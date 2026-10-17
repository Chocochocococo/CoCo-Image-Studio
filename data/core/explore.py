# -*- coding: utf-8 -*-
"""
explore.py —— 探索的特徵（工作行程這端）

SigLIP2 base：圖與文字算成同一個空間的 768 維向量（正規化過，內積就是相似度）。
前處理照模型的設定：縮成 224×224（雙線性）、除以 255、減 0.5 除 0.5。文字補到 64 個 token。
模型只在要用時載入（建立索引、用文字找圖；以圖找圖用的是存好的特徵，不用模型），
閒置 IDLE 秒就卸載，把約 0.7 GB 還給顯示卡；連續用（索引一批接一批、連續查文字）不會一直重載。
"""

import threading
import time
from pathlib import Path

SIZE = 224
TEXT_LEN = 64
BATCH = 32
_M = {}                    # 模型資料夾 → (model, tokenizer)
IDLE = 120                 # 幾秒沒用就卸載（載入一次約 4 秒）
_LOCK = threading.RLock()  # 算特徵與卸載不能同時
_LAST = [0.0]              # 最後一次用的時間
_TIMER = [None]
on_release = None          # 卸載時通知（工作行程記一行紀錄）
ERRORS = []                # 最近讀不了的圖（路徑, 原因），除錯用


def _touch():
    """用過一次：重新倒數，IDLE 秒內沒再用就卸載"""
    _LAST[0] = time.monotonic()
    if _TIMER[0] is not None:
        _TIMER[0].cancel()
    t = threading.Timer(IDLE + 0.05, _idle_release)
    t.daemon = True
    _TIMER[0] = t
    t.start()


def _idle_release():
    with _LOCK:
        if _M and time.monotonic() - _LAST[0] >= IDLE:
            release()
            if on_release:
                on_release()


def load(model_dir):
    key = str(model_dir)
    if key not in _M:
        import torch
        from transformers import AutoTokenizer, SiglipModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = SiglipModel.from_pretrained(key, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
        model.to(dev).eval()
        _M.clear()
        _M[key] = (model, AutoTokenizer.from_pretrained(key))
    return _M[key]


def _pixels(paths):
    """圖 → (張量, 讀得到的索引)。讀不了的圖略過（回傳哪幾張成功）"""
    import numpy as np
    import torch
    from PIL import Image
    arrs, ok = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im.draft("RGB", (SIZE * 2, SIZE * 2))          # JPEG 先用低解析度解碼，快很多
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im)
                im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                arrs.append(np.asarray(im, dtype=np.float32))
                ok.append(i)
        except Exception as e:
            ERRORS.append((str(p), f"{type(e).__name__}: {e}"))
            del ERRORS[:-20]
            continue
    if not arrs:
        return None, ok
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2) / 255.0
    return (x - 0.5) / 0.5, ok


def embed_images(model_dir, paths):
    """圖的路徑 → (N×768 float16 的 numpy 陣列, 成功的索引)"""
    with _LOCK:
        try:
            return _embed_images(model_dir, paths)
        finally:
            _touch()


def _embed_images(model_dir, paths):
    import numpy as np
    import torch
    model, _ = load(model_dir)
    p = next(model.parameters())
    outs, oks = [], []
    for s in range(0, len(paths), BATCH):
        x, ok = _pixels(paths[s:s + BATCH])
        if x is None:
            continue
        with torch.inference_mode():
            v = model.get_image_features(pixel_values=x.to(p.device, p.dtype))
            v = getattr(v, "pooler_output", v)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
        outs.append(v.cpu().numpy().astype(np.float16))
        oks += [s + i for i in ok]
    if not outs:
        return np.zeros((0, 768), np.float16), []
    return np.concatenate(outs), oks


def embed_text(model_dir, text):
    """文字 → 768 維（float32 list）"""
    with _LOCK:
        try:
            return _embed_text(model_dir, text)
        finally:
            _touch()


def _embed_text(model_dir, text):
    import torch
    model, tok = load(model_dir)
    p = next(model.parameters())
    t = tok([str(text)], padding="max_length", max_length=TEXT_LEN, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        v = model.get_text_features(input_ids=t["input_ids"].to(p.device))
        v = getattr(v, "pooler_output", v)
        v = torch.nn.functional.normalize(v.float(), dim=-1)
    return v[0].cpu().tolist()


def release():
    """卸載模型、把快取還給顯示卡"""
    with _LOCK:
        if not _M:
            return
        _M.clear()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
