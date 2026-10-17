# -*- coding: utf-8 -*-
"""
resume.py —— 訓練進度：停下來時存、之後從那裡接著練（使用者要求，2026-10-07 修改核對表 F3）。
sdxl_train.py、anima_train.py、slider_train.py 共用。

- **存什麼**：LoRA 權重（float32，跟訓練中一樣）、優化器與學習率排程的狀態、GradScaler、走到第幾步、
  這個 epoch 的批次順序與做到第幾批、Python 亂數的狀態、loss 的平均；Slider 另外存底圖的 latent（自動畫的底圖是暫存檔，
  接續時不用重畫）與還沒輪到的順序。每一步的雜訊是用 步數 重新設 seed 的，不用存。
  同樣的設定、同一份資料集：停下來再接續跟一次練完的每一步都一樣（同一批圖、同一個雜訊、同一個學習率）；
  CPU 上逐位元相同（tests/test_train_resume.py）。顯示卡上同樣的設定練兩次本身就不逐位元相同（fp16 的累加順序），
  接續跟一次練完的差跟它同一個量級（2026-10-07 RTX 5080：SDXL 16 步 3.1e-6 對 1.8e-6、Slider 40 步 1.3e-3 對 1.1e-3）。
- **放哪**：LoRA 資料夾裡的 _coco_progress/state.bin。不用 .pt：資源庫把 .pt 當 LoRA 列出來。
  先寫到 .tmp 再換名字：存到一半當掉不會把上一份弄壞。
- **資料集**：記下每張圖的檔名（順序）與內容的指紋。圖的清單不一樣（加了、刪了、改名）就不能接（批次順序對不上）；
  只是改了標籤或換了圖的內容，照新的練、寫一行提醒。
"""

import hashlib
import os
from pathlib import Path

import i18n

DIR = "_coco_progress"
FILE = "state.bin"


def path_of(out_dir):
    return Path(out_dir) / DIR / FILE


def lora_state(groups):
    """[(前綴, {名稱: LoRALinear})] → {前綴.名稱: {down, up}}（CPU、float32 的複本）"""
    return {f"{prefix}.{name}": {"down": m.lora_down.weight.detach().float().cpu().clone(),
                                 "up": m.lora_up.weight.detach().float().cpu().clone()}
            for prefix, mods in groups for name, m in mods.items()}


def load_lora(groups, saved):
    """存的權重放回去；層的名稱或形狀對不上（rank 改了、換了底模）就停下來，不要練出奇怪的東西"""
    import torch
    have = {f"{prefix}.{name}": m for prefix, mods in groups for name, m in mods.items()}
    if set(have) != set(saved):
        raise ValueError(i18n.tr("eng.res.model"))
    with torch.no_grad():
        for k, m in have.items():
            for part, w in (("down", m.lora_down.weight), ("up", m.lora_up.weight)):
                src = saved[k][part]
                if tuple(src.shape) != tuple(w.shape):
                    raise ValueError(i18n.tr("eng.res.rank"))
                w.copy_(src.to(w.device, w.dtype))


def dataset_sig(items):
    """資料集的指紋：檔名的順序（決定批次）、內容（圖的大小與修改時間、標籤文字）"""
    names = [Path(it["path"]).name for it in items]
    h = hashlib.sha1()
    for it in items:
        try:
            st = os.stat(it["path"])
            h.update(f"{st.st_size}:{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(b"-\n")
        h.update((it.get("caption") or "").encode("utf-8") + b"\n")
    return {"names": names, "content": h.hexdigest()}


def check_dataset(saved, items):
    """→ None（一樣）、"changed"（只改了內容：照新的練）；圖的清單不一樣丟出錯誤"""
    now = dataset_sig(items)
    if (saved or {}).get("names") != now["names"]:
        raise ValueError(i18n.tr("eng.res.dataset"))
    return None if saved.get("content") == now["content"] else "changed"


def save(out_dir, payload, out):
    """寫進度檔、回報 {"type": "state"}（server/train.py 記下來，03 訓練才列得出來）"""
    import torch
    p = path_of(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    out({"type": "state", "path": str(p), "step": int(payload["step"]), "done": bool(payload.get("done"))})


def load(path):
    """weights_only：進度檔只放張量、數字、清單，不執行檔案裡的程式（別處拿來的檔也安全）"""
    import torch
    p = Path(path)
    if not p.is_file():
        raise ValueError(i18n.tr("eng.res.missing"))
    return torch.load(p, map_location="cpu", weights_only=True)


def rng_state(rng):
    """random.Random 的狀態（tuple 裡有 tuple）存成清單；set_rng 換回來"""
    v, inner, gauss = rng.getstate()
    return [v, list(inner), gauss]


def set_rng(rng, st):
    rng.setstate((st[0], tuple(st[1]), st[2]))
