# -*- coding: utf-8 -*-
"""去背（精修的「去背」方式）：模型的清單、下載、刪除，以及把去背的結果存成新圖。

使用者筆記：不是每個人都會用去背，模型不內建、不自動下載——精修頁先列出可以下載的模型（用途、大小、授權），
按「下載」才下載；不用了可以刪掉。自動下載失敗（例如防毒軟體擋 SSL）時，畫面上有網址與資料夾，
自己用瀏覽器下載放進去也認得。

不限定哪些檔案：CoCo 認的是「架構」。models/rembg（以及模型總資料夾裡的 rembg）裡的 .safetensors，
讀檔頭的權重名字與形狀判斷是不是認得的架構（IS-Net、Swin 主幹的 BiRefNet），認得的都能選來用（別人微調過的也一樣）。
只收 .safetensors：.pth／.ckpt 這類載入時可能執行檔案裡夾帶的程式。認不得的列出來並說原因。

IS-Net 不同來源的前處理不一樣：anime-segmentation 系是 0～1、補成正方形；DIS 官方系是再減 0.5、
直接縮成正方形。權重名字前面有 "net."（anime-segmentation 的存法）就先猜前者，不然猜後者，使用者可以換。
BiRefNet 只有官方那一種（ImageNet 標準化）。

下載完照 Hugging Face 公布的 SHA256 檢查，對不上就刪掉（檔案壞了或被換過）；中斷了下次接著下載（download.py）。
實際跑模型在工作行程（core/bgremove.py）；這裡把遮罩套到原圖上存成透明背景的 PNG，
中繼資料記 init（精修頁的版本列靠它），生成參數照原圖的留著。
"""

import json
import time
from pathlib import Path

from . import download
from .state import C, SET

from i18n import tr

# 可以下載的模型（按了才下載）。下載下來的檔案跟自己放的一樣，靠架構判斷來用
MODELS = {
    "isnet-anime": {
        "file": "isnet-anime.safetensors",
        "alias": ("isnetis.safetensors", "anime-seg.safetensors", "model.safetensors"),   # 自己下載的沒改名也認得
        "url": "https://huggingface.co/skytnt/anime-seg/resolve/main/model.safetensors",
        "page": "https://huggingface.co/skytnt/anime-seg",
        "bytes": 203982056,
        "sha256": "3351563ba8b61a01a66bacb79cf36aabce2da62d88b7606a719f573f23fe5d3e",
        "license": "Apache-2.0",
    },
    "birefnet-lite": {
        "file": "birefnet-lite.safetensors",
        "alias": ("BiRefNet_lite.safetensors", "model.safetensors"),
        "url": "https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/main/model.safetensors",
        "page": "https://huggingface.co/ZhengPeng7/BiRefNet_lite",
        "bytes": 177634392,
        "sha256": "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815",
        "license": "MIT",
    },
}
ARCHES = ("isnet", "birefnet")         # 認得的架構
PREPS = ("anime", "dis")               # 前處理：anime-segmentation 系、DIS 官方系
UNSAFE = (".pth", ".pt", ".ckpt", ".bin", ".pkl", ".pickle")
PREFIXES = ("", "net.", "model.", "module.")
_DL = download.Jobs()          # 模型 → 下載進度
_PROBE = {}                    # (路徑, 大小, 修改時間) → 判斷結果（檔頭不用每次都讀）


def home():
    from . import modelhome
    return modelhome.own("rembg")


def dirs():
    out = [home()]
    root = str(SET.load().get("MODELS_ROOT") or "").strip()
    if root and root != ".":
        out.append(Path(root) / "rembg")
    return out


# ── 認架構 ──
def read_header(path):
    """safetensors 的檔頭：{權重名字: 形狀}。只讀開頭那段 JSON，不載入權重；不是 safetensors 回 None"""
    try:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if not 2 <= n <= 64 * 1024 * 1024:
                return None
            head = json.loads(fh.read(n))
    except (OSError, ValueError):
        return None
    if not isinstance(head, dict):
        return None
    return {k: v.get("shape") for k, v in head.items() if k != "__metadata__" and isinstance(v, dict)}


def detect(shapes):
    """權重的名字與形狀 → {arch, prefix, prep}；認不得回 {why}"""
    if not shapes:
        return {"why": "notSafetensors"}
    for p in PREFIXES:
        g = shapes.get
        if (g(p + "conv_in.weight") == [64, 3, 3, 3] and g(p + "side1.weight") == [1, 64, 3, 3]
                and g(p + "stage1.rebnconvin.conv_s1.weight") and g(p + "stage6.rebnconv4.conv_s1.weight")
                and not g(p + "outconv.weight")):
            return {"arch": "isnet", "prefix": p, "prep": "anime" if p == "net." else "dis"}
    for p in PREFIXES:
        g = shapes.get
        if g(p + "squeeze_module.0.conv_in.weight") and g(p + "decoder.conv_out1.0.weight"):
            pe = g(p + "bb.patch_embed.proj.weight")
            if not pe or list(pe[1:]) != [3, 4, 4]:
                return {"why": "birefnetBackbone"}          # PVT、VGG、ResNet 之類的主幹：還不支援
            return {"arch": "birefnet", "prefix": p, "prep": "imagenet", "variant": _swin_size(shapes, p, pe[0])}
    names = list(shapes)
    if any(k.endswith("outconv.weight") for k in names) and any("rebnconvin" in k for k in names):
        return {"why": "u2net"}
    return {"why": "unknown"}


def _swin_size(shapes, p, dim):
    """Swin 主幹是哪一種：T（96、第三層 6 塊）、S（96、18 塊）、B（128）、L（192）"""
    n = 0
    while f"{p}bb.layers.2.blocks.{n}.norm1.weight" in shapes:
        n += 1
    return {96: "Swin-T" if n <= 6 else "Swin-S", 128: "Swin-B", 192: "Swin-L"}.get(dim, f"Swin {dim}")


def _probe(f):
    try:
        st = f.stat()
    except OSError:
        return {"why": "unreadable"}
    key = (str(f), st.st_size, st.st_mtime)
    if key not in _PROBE:
        _PROBE[key] = detect(read_header(f))
    return _PROBE[key]


def scan():
    """兩個資料夾裡的模型：認得的 [{id, name, known, arch, prefix, prep, path, bytes, own}]、認不得的 [{name, dir, why}]。
    同名的只算第一個（CoCo 的資料夾優先）"""
    models, skipped, seen = [], [], set()
    for d in dirs():
        try:
            files = sorted(x for x in d.iterdir() if x.is_file())
        except OSError:
            continue
        for f in files:
            ext = f.suffix.lower()
            if ext in UNSAFE:
                skipped.append({"name": f.name, "dir": str(d), "why": "unsafe"})
                continue
            if ext != ".safetensors" or f.name.lower() in seen:
                continue
            info = _probe(f)
            if not info.get("arch"):
                skipped.append({"name": f.name, "dir": str(d), "why": info.get("why", "unknown")})
                continue
            seen.add(f.name.lower())
            size = f.stat().st_size
            known = next((k for k, m in MODELS.items() if size == m["bytes"] and f.name in (m["file"], *m["alias"])), None)
            models.append({"id": f.name, "name": f.stem, "known": known, "arch": info["arch"], "prefix": info["prefix"],
                           "prep": info["prep"], "variant": info.get("variant", ""), "path": str(f), "bytes": size,
                           "own": f.parent == home()})
    return models, skipped


def find_model(mid):
    return next((m for m in scan()[0] if m["id"] == mid), None)


def find(key):
    """可以下載的那幾個：已經有了嗎（下載的、或自己放的同一個檔案）"""
    m = next((x for x in scan()[0] if x["known"] == key), None)
    return Path(m["path"]) if m else None


def status():
    """精修頁要的：可以下載的模型（有沒有、下載進度）、所有能用的模型、認不得的檔案、資料夾"""
    models, skipped = scan()
    have = {m["known"] for m in models if m["known"]}
    catalog = []
    for key, m in MODELS.items():
        catalog.append({"key": key, "installed": key in have, "bytes": m["bytes"], "url": m["url"], "page": m["page"],
                        "part": 0 if key in have else download.partial(home() / m["file"], m["bytes"]),
                        "license": m["license"], "file": m["file"], "download": _DL.progress(key)})
    return {"catalog": catalog, "models": models, "skipped": skipped, "dir": str(home()),
            "dirs": [str(d) for d in dirs()], "arches": list(ARCHES)}


def _download(key, opener=None):
    m = MODELS[key]
    download.run(_DL[key], [{**m, "dest": home() / m["file"]}], opener, hint=tr("srv.dl.manualHint"))


def start_download(key):
    if key not in MODELS:
        return {"error": tr("srv.unknownModel")}
    if find(key):
        return {"ok": True, "already": True}
    return _DL.begin(key, MODELS[key]["bytes"], _download, (key,))


def cancel_download(key):
    return _DL.cancel(key)


def remove(mid):
    """刪掉 CoCo 資料夾（models/rembg）裡的模型；模型總資料夾裡的是使用者自己的資料夾，不動"""
    m = find_model(mid)
    if not m:
        return {"ok": True}
    if not m["own"]:
        return {"error": tr("srv.rb.notOwn", path=m['path'])}
    try:
        Path(m["path"]).unlink()
    except OSError as e:
        return {"error": str(e)}
    return {"ok": True}


# ── 套用遮罩、存成新圖 ──
def _source_meta(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.info.get("parameters") or "", im.info.get("coco") or ""
    except OSError:
        return "", ""


SOFT = (16, 240)               # 柔和：低於 16 的當全透明（地板倒影之類的淡影）、高於 240 的當全不透明（角色本體）


def soften(mask, edge):
    """邊緣：soft＝保留模型的柔邊，只把兩端清乾淨（ISNet 在角色身上常給 240 幾、背景常留一點點）；
    sharp＝把半透明收窄（邊緣比較乾淨，但細毛會變硬）"""
    if edge == "sharp":
        return mask.point(lambda v: max(0, min(255, round((v - 128) * 3 + 128))))
    lo, hi = SOFT
    return mask.point(lambda v: max(0, min(255, round((v - lo) * 255 / (hi - lo)))))


def save_result(src_path, mask, key, edge, out_dir=None, prep=None):
    """原圖＋遮罩 → 透明背景的 PNG（顏色照原圖，只換透明度）。回傳存好的路徑"""
    from PIL import Image, PngImagePlugin
    src = Path(src_path)
    with Image.open(src) as im:
        rgba = im.convert("RGBA")
    if mask.size != rgba.size:
        mask = mask.resize(rgba.size, Image.BILINEAR)
    a = soften(mask, edge)
    if rgba.getchannel("A").getextrema()[0] < 255:          # 原圖本來就有透明的地方：兩個取小的
        from PIL import ImageChops
        a = ImageChops.multiply(a, rgba.getchannel("A"))
    rgba.putalpha(a)
    params, coco = _source_meta(src)
    try:
        c = json.loads(coco) if coco else {}
    except ValueError:
        c = {}
    c = {k: v for k, v in c.items() if k not in ("init", "strength", "inpaint", "faces", "edit_of")}
    c.update({"init": str(src), "rembg": {"model": key, "edge": edge, **({"prep": prep} if prep else {})}})
    info = PngImagePlugin.PngInfo()
    if params:
        info.add_text("parameters", params)
    info.add_text("coco", json.dumps(c, ensure_ascii=False))
    folder = Path(out_dir or (Path(C.OUTPUT_DIR) / time.strftime("%Y-%m-%d")))
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{src.stem}_nobg.png"
    n = 2
    while out.exists():
        out = folder / f"{src.stem}_nobg_{n}.png"
        n += 1
    rgba.save(out, "PNG", pnginfo=info)
    return out
