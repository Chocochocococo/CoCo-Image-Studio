# -*- coding: utf-8 -*-
"""
embeddings.py —— Textual inversion（資源庫的 Embeddings）

跟 A1111 一樣：提示詞裡寫 embedding 的檔名（例如 lazyhand）就套用，不分大小寫；
也接受 ComfyUI 的寫法 embedding:lazyhand（可以帶副檔名）。

做法：
  - 載入時不用檔名當 token，用 cocoti0、cocoti1…（檔名可能剛好是詞彙表裡的字，
    例如叫 smile 的 embedding，diffusers 會拒絕；大小寫也會對不上 CLIP 的 tokenizer）。
    一個 embedding 有 n 個向量，diffusers 會建 cocoti0、cocoti0_1、…、cocoti0_{n-1}
  - 編碼前把提示詞裡的名字換成那一串 token（expand），A1111 分塊編碼與管線自己編碼
    （native）兩條路都吃得到；寫進中繼資料的還是原本的提示詞
  - SDXL 的檔案有 clip_l（768 維）與 clip_g（1280 維）兩份，分別載進兩個編碼器；
    SD1.5 只有一份（A1111 的 .pt / safetensors 的 emb_params，或 diffusers 格式）
  - 載過的記在 pipe._coco_ti，換模型時整個管線重建，自然就清掉了

這個模組最上層不 import torch：伺服器也用 used() 找提示詞用到哪些 embedding。
"""

import re
from pathlib import Path

import i18n

EXTS = (".safetensors", ".pt", ".bin")
TOKEN = "cocoti"


def _pattern(names):
    """名字前後不能接著字母、數字、底線、連字號（lazyhand 不會吃掉 lazyhands）"""
    alts = "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
    return re.compile(rf"(?<![\w\-.:])(?:embedding:)?({alts})(?:\.(?:safetensors|pt|bin))?(?![\w\-])", re.I)


def used(text, names):
    """text 裡用到的 embedding（回傳小寫名字的集合）"""
    names = [n for n in names if n]
    if not text or not names:
        return set()
    return {m.group(1).lower() for m in _pattern(names).finditer(text)}


def expand(text, ti, full=True):
    """把名字換成載入時的 token。ti：{小寫名字: (token, 向量數)}

    full：換成整串 cocoti0 cocoti0_1 …（CoCo 自己的 A1111 分塊編碼用）。
    管線自己編碼（native）時只換成第一個：diffusers 的 maybe_convert_prompt 會再展開，
    先展開的話它會把 cocoti0_1 裡的 cocoti0 又展開一次"""
    if not text or not ti:
        return text

    def rep(m):
        tok, n = ti[m.group(1).lower()]
        return " ".join([tok] + [f"{tok}_{i}" for i in range(1, n)]) if full else tok
    return _pattern(list(ti)).sub(rep, text)


def read(path):
    """讀檔 → {"clip_l": 張量, "clip_g": 張量}（SD1.5 只有 clip_l）。認不得的格式丟 ValueError"""
    import torch
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(i18n.tr("srv.badFormat"))
    if "clip_l" in sd and "clip_g" in sd:                       # SDXL
        return {"clip_l": sd["clip_l"], "clip_g": sd["clip_g"]}
    if "emb_params" in sd:                                     # A1111 存的 safetensors
        return {"clip_l": sd["emb_params"]}
    if isinstance(sd.get("string_to_param"), dict):            # A1111 的 .pt
        v = list(sd["string_to_param"].values())
        if v:
            return {"clip_l": v[0]}
    tensors = [v for v in sd.values() if hasattr(v, "shape")]
    if len(tensors) == 1:                                      # diffusers：{"<token>": 張量}
        return {"clip_l": tensors[0]}
    raise ValueError(i18n.tr("eng.emb.unknown"))


def arch(vec):
    """sdxl / sd15（看向量的維度）"""
    if "clip_g" in vec:
        return "sdxl"
    return "sd15" if vec["clip_l"].shape[-1] == 768 else ""


def ensure(pipe, items, log=print):
    """確定這些 embedding 已經載進管線。items：[(名字, 路徑), …]；回傳這次真的套用的名字。

    跟模型不合的（SD1.5 的用在 SDXL 上）略過並提醒，不讓整張圖失敗。"""
    ti = pipe.__dict__.setdefault("_coco_ti", {})
    sdxl = getattr(pipe, "text_encoder_2", None) is not None
    ok = []
    for name, path in items:
        key = str(name).lower()
        if key in ti:
            ok.append(name)
            continue
        try:
            vec = read(path)
        except Exception as e:
            log("[!] " + i18n.tr("eng.emb.unreadable", name=name, e=e))
            continue
        if arch(vec) != ("sdxl" if sdxl else "sd15"):
            log("[!] " + i18n.tr("eng.emb.mismatch", name=name, arch=arch(vec) or '?'))
            continue
        # 編號另外數：載到一半失敗的那個 token 已經在 tokenizer 裡了，不能再用同一個名字
        n = pipe.__dict__.get("_coco_ti_n", 0)
        pipe._coco_ti_n = n + 1
        tok = f"{TOKEN}{n}"
        vl = vec["clip_l"].reshape(-1, vec["clip_l"].shape[-1])
        try:
            pipe.load_textual_inversion(vl, token=tok, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            if sdxl:
                vg = vec["clip_g"].reshape(-1, vec["clip_g"].shape[-1])
                pipe.load_textual_inversion(vg, token=tok, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
        except Exception as e:
            log("[!] " + i18n.tr("eng.emb.fail", name=name, e=e))
            continue
        ti[key] = (tok, int(vl.shape[0]))
        log(i18n.tr("eng.emb.loaded", name=name, n=vl.shape[0]))
        ok.append(name)
    return ok


def short_hash(path):
    """A1111 的 TI hashes：檔案 SHA-256 的前 10 碼"""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]
