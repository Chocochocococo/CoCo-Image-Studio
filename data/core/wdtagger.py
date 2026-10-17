# -*- coding: utf-8 -*-
"""
wdtagger.py —— WD14 打標籤（訓練空間的打標台，工作行程這端）

SmilingWolf 的 WD v3 tagger（Apache-2.0）。舊的獨立打標台用 onnxruntime 跑 model.onnx，但 onnxruntime-gpu
要 CUDA 12 的函式庫，CoCo 的 PyTorch 是 CUDA 13，兩邊的 DLL 對不上（舊打標台為了這個輪流試三種 DLL 來源，
還有一顆「診斷 GPU」）。這裡自己讀檔、用 PyTorch 跑，不需要 onnxruntime。

**認得的模型。** ViT 系（wd-vit-large-tagger-v3、wd-vit-tagger-v3）：patch 卷積＋位置編碼（沒有 cls token）、
前置 LayerNorm（eps 1e-6）的標準 Transformer、tanh 近似的 GELU、最後 LayerNorm 之後整張平均、一層線性、sigmoid。
EVA02、SwinV2、ConvNeXt 版結構不同，認得出來但說還不支援（`arch_of`）。

**讀檔。** 資料夾裡的 model.onnx（官方的、舊打標台用的那個）或 timm 的 model.safetensors，旁邊要有 selected_tags.csv。
ONNX 不裝 onnx 套件，自己讀 protobuf（`read_onnx`，只要權重與節點名稱）：有名字的權重直接用（拿掉 core_model.）；
矩陣乘法的權重匯出時被改名成 onnx::MatMul_N、存成轉置的，照用到它的節點名稱
（/core_model/blocks/blocks.0/attn/qkv/MatMul）對回去再轉置。前處理的兩個常數（除、減）從圖裡讀。

**前處理。** 跟官方範例（與舊打標台）一樣：透明的地方鋪白底、補成白底正方形置中、縮成模型的大小（bicubic）、
RGB 轉 BGR、0～255；模型裡再 ÷127.5 − 1。

**模型只在要用時載入**，閒置 IDLE 秒就卸載（跟探索一樣），把約 0.7 GB 還給顯示卡。
"""

import csv
import mmap
import re
import threading
import time
from pathlib import Path

import numpy as np

import i18n

BATCH = 8
FLOOR = 0.1                # 回傳的最低信心：門檻在伺服器那端套，改門檻不用重跑模型
IDLE = 120
_M = {}                    # 資料夾 → (模型, 標籤清單, 輸入大小)
_LOCK = threading.RLock()
_LAST = [0.0]
_TIMER = [None]
on_release = None
ERRORS = []

# selected_tags.csv 的分類：0 一般、4 角色、9 分級（general／sensitive／…，打標不用）
GENERAL, CHARACTER, RATING = 0, 4, 9


# ---------------------------------------------------------------- 讀 ONNX（最小的 protobuf 解析）

def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b, i, end):
    """protobuf 一層的欄位：(欄位號, 型別, 值或起點, 終點)"""
    while i < end:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v, None
        elif wt == 1:
            yield f, wt, i, i + 8
            i += 8
        elif wt == 5:
            yield f, wt, i, i + 4
            i += 4
        elif wt == 2:
            n, i = _varint(b, i)
            yield f, wt, i, i + n
            i += n
        else:
            raise ValueError(i18n.tr("eng.wd.onnx", t=wt))


_NP = {1: np.float32, 7: np.int64, 10: np.float16, 11: np.float64}


def _tensor(b, a, z):
    """TensorProto → (名字, numpy 陣列)。資料在 raw_data，或小常數放在 float_data／int64_data"""
    name, dims, dt, raw, packed = "", [], 1, None, []
    for f, wt, x, y in _fields(b, a, z):
        if f == 1:
            if wt == 0:
                dims.append(x)
            else:
                j = x
                while j < y:
                    v, j = _varint(b, j)
                    dims.append(v)
        elif f == 2:
            dt = x
        elif f == 8:
            name = bytes(b[x:y]).decode("utf-8")
        elif f == 9:
            raw = (x, y)
        elif f == 4 and wt == 2:                      # float_data（packed）
            packed = np.frombuffer(bytes(b[x:y]), np.float32)
        elif f == 4 and wt == 5:
            packed = np.append(packed, np.frombuffer(bytes(b[x:y]), np.float32))
        elif f == 7 and wt == 0:                      # int64_data（沒 packed）
            packed = np.append(packed, x)
    dtype = _NP.get(dt)
    if dtype is None:
        return name, None
    if raw is not None:
        # 複製出來：留著指向 mmap 的陣列，檔案就關不掉
        arr = np.frombuffer(b, dtype, count=(raw[1] - raw[0]) // np.dtype(dtype).itemsize, offset=raw[0]).copy()
    else:
        arr = np.asarray(packed, dtype)
    if arr.size != int(np.prod(dims)):        # 沒處理的存法（packed int64 這類形狀常數）：用不到，略過
        return name, None
    return name, arr.reshape(dims)


def read_onnx(path):
    """model.onnx → (權重 dict（拿掉 core_model.、矩陣乘法的已轉置回 PyTorch 的方向）, 前處理的 (除, 減))

    ONNX 匯出把 nn.Linear 的權重改名成 onnx::MatMul_N 並轉置成 (in, out)；用到它的 MatMul 節點名稱
    保留了模組路徑（/core_model/blocks/blocks.0/attn/qkv/MatMul），照它對回 blocks.0.attn.qkv.weight。"""
    with open(path, "rb") as fh:
        b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            inits, nodes = {}, []
            for f, _, a, z in _fields(b, 0, len(b)):
                if f != 7:                            # ModelProto.graph
                    continue
                for gf, _, ga, gz in _fields(b, a, z):
                    if gf == 5:                       # GraphProto.initializer
                        name, arr = _tensor(b, ga, gz)
                        if arr is not None:
                            inits[name] = arr
                    elif gf == 1:                     # GraphProto.node
                        ins, op, nm = [], "", ""
                        for nf, _, na, nz in _fields(b, ga, gz):
                            if nf == 1:
                                ins.append(bytes(b[na:nz]).decode("utf-8"))
                            elif nf == 3:
                                nm = bytes(b[na:nz]).decode("utf-8")
                            elif nf == 4:
                                op = bytes(b[na:nz]).decode("utf-8")
                        nodes.append((op, nm, ins))
            sd, div, sub = {}, None, None
            for name, arr in inits.items():
                if name.startswith("core_model."):
                    sd[name[len("core_model."):]] = arr
            for op, nm, ins in nodes:
                if op == "MatMul" and len(ins) == 2 and ins[1] in inits and nm.startswith("/core_model/"):
                    sd[_module_path(nm) + ".weight"] = inits[ins[1]].T      # 建模型時才轉成連續的
                elif op == "Div" and nm == "/Div" and ins[1] in inits:
                    div = float(inits[ins[1]])
                elif op == "Sub" and nm == "/Sub" and ins[1] in inits:
                    sub = float(inits[ins[1]])
            return sd, (div if div is not None else 127.5, sub if sub is not None else 1.0)
        finally:
            b.close()


def _module_path(node_name):
    """/core_model/blocks/blocks.0/attn/qkv/MatMul → blocks.0.attn.qkv

    PyTorch 匯出的節點名稱把容器與子模組都寫一次（blocks/blocks.0），後一段以前一段加「.」開頭的，前一段拿掉"""
    parts = node_name.strip("/").split("/")[1:-1]          # 去掉 core_model 與最後的運算名稱
    keep = [p for i, p in enumerate(parts) if not (i + 1 < len(parts) and parts[i + 1].startswith(p + "."))]
    return ".".join(keep)


# ---------------------------------------------------------------- 架構

def arch_of(keys):
    """從權重名字認架構：vit（支援）、eva02／swinv2／convnext（認得、不支援）、None（不是 WD tagger）"""
    keys = set(keys)
    if any(k.startswith("stages.") for k in keys):
        return "convnext"
    if any(k.startswith("layers.") and ".blocks." in k for k in keys):
        return "swinv2"
    if any("rope" in k or ".mlp.fc1_g." in k or ".attn.q_proj." in k for k in keys):
        return "eva02"
    if "pos_embed" in keys and "blocks.0.attn.qkv.weight" in keys and "head.weight" in keys:
        return "vit"
    return None


def _build(sd):
    import torch
    from torch import nn
    import torch.nn.functional as F

    dim, _, patch, _ = sd["patch_embed.proj.weight"].shape
    tokens = sd["pos_embed"].shape[1]
    grid = int(round(tokens ** 0.5))
    if grid * grid != tokens:
        raise ValueError(i18n.tr("eng.wd.cls"))
    depth = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k)))
    heads = dim // 64
    classes = sd["head.weight"].shape[0]

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, N, _ = x.shape
            q, k, v = self.qkv(x).reshape(B, N, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
            return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, dim))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = sd["blocks.0.mlp.fc1.weight"].shape[0]
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attn()
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, dim, patch, patch)

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.pos_embed = nn.Parameter(torch.zeros(1, tokens, dim))
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.head = nn.Linear(dim, classes)

        def forward(self, x):
            x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) + self.pos_embed
            for blk in self.blocks:
                x = blk(x)
            return torch.sigmoid(self.head(self.norm(x).mean(1)))

    with torch.device("meta"):
        model = ViT()
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()},
                          strict=True, assign=True)
    return model, grid * patch


def find(model_dir):
    """資料夾 → (權重檔, selected_tags.csv)；ONNX 與 safetensors 都有時用 safetensors（不用解析）"""
    d = Path(model_dir)
    tags = d / "selected_tags.csv"
    for name in ("model.safetensors", "model.onnx"):
        if (d / name).is_file() and tags.is_file():
            return d / name, tags
    return None, None


def read_tags(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append((row["name"], int(row.get("category") or 0)))
    return out


def load_weights(path):
    """權重檔 → (權重 dict, (除, 減))"""
    path = Path(path)
    if path.suffix.lower() == ".onnx":
        return read_onnx(path)
    from safetensors.numpy import load_file
    sd = load_file(str(path))
    return {k[len("core_model."):] if k.startswith("core_model.") else k: v for k, v in sd.items()}, (127.5, 1.0)


def load(model_dir):
    key = str(model_dir)
    if key not in _M:
        import torch
        weights, tags_csv = find(model_dir)
        if weights is None:
            raise FileNotFoundError(i18n.tr("eng.wd.files", dir=model_dir))
        sd, norm = load_weights(weights)
        arch = arch_of(sd)
        if arch != "vit":
            raise ValueError(i18n.tr("eng.wd.arch", arch=arch or i18n.tr("eng.wd.unknownArch")))
        model, size = _build(sd)
        del sd
        tags = read_tags(tags_csv)
        if len(tags) != model.head.out_features:
            raise ValueError(i18n.tr("eng.wd.tags", n=len(tags), m=model.head.out_features))
        # 一律 float32：跟舊打標台（onnxruntime）比 32 張圖，float32 的 982 個標籤全部一樣、fp16 差 1 個
        # （剛好在門檻上的）；速度 2.0 對 1.25 秒／32 張，幾百張的資料集也只差幾秒。實測 RTX 5080，2026-10-06
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev, torch.float32).eval()
        _M.clear()
        _M[key] = (model, tags, size, norm)
    return _M[key]


# ---------------------------------------------------------------- 推論

def prep(path, size):
    """圖 → (size, size, 3) 的 BGR float32（0～255）：鋪白底、補成白底正方形置中、bicubic 縮放"""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    im = bg.convert("RGB")
    w, h = im.size
    m = max(w, h)
    sq = Image.new("RGB", (m, m), (255, 255, 255))
    sq.paste(im, ((m - w) // 2, (m - h) // 2))
    if m != size:
        sq = sq.resize((size, size), Image.BICUBIC)
    return np.asarray(sq, dtype=np.float32)[:, :, ::-1]


def _touch():
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


def probs(model_dir, paths):
    """圖的路徑 → (N × 標籤數 的 float32 機率, 成功的索引)。讀不了的圖略過"""
    with _LOCK:
        try:
            return _probs(model_dir, paths)
        finally:
            _touch()


def _probs(model_dir, paths):
    import torch
    model, tags, size, (div, sub) = load(model_dir)
    p = next(model.parameters())
    outs, oks = [], []
    for s in range(0, len(paths), BATCH):
        arrs, ok = [], []
        for i, path in enumerate(paths[s:s + BATCH]):
            try:
                arrs.append(prep(path, size))
                ok.append(s + i)
            except Exception as e:
                ERRORS.append((str(path), f"{type(e).__name__}: {e}"))
                del ERRORS[:-20]
        if not arrs:
            continue
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(p.device)
        x = (x / div - sub).to(p.dtype)
        with torch.inference_mode():
            outs.append(model(x).float().cpu().numpy())
        oks += ok
    if not outs:
        return np.zeros((0, len(tags)), np.float32), []
    return np.concatenate(outs), oks


def predict(model_dir, paths, floor=FLOOR):
    """圖的路徑 → 每張一個清單 [[標籤, 分類, 機率], …]（一般與角色、機率 ≥ floor，高的在前）；讀不了的是 None"""
    pr, ok = probs(model_dir, paths)
    _, tags, _, _ = _M[str(model_dir)]
    res = [None] * len(paths)
    keep = np.array([c in (GENERAL, CHARACTER) for _, c in tags])
    for row, i in zip(pr, ok):
        idx = np.nonzero((row >= floor) & keep)[0]
        idx = idx[np.argsort(-row[idx], kind="stable")]
        res[i] = [[tags[j][0], tags[j][1], round(float(row[j]), 4)] for j in idx]
    return res


def release():
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
