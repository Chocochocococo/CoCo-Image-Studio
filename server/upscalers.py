# -*- coding: utf-8 -*-
"""高清化（精修的「高清化」）：放大模型的清單、倍數與尺寸，以及「只放大」的結果怎麼存。

兩種做法：
  只放大：工作行程用放大模型（core/upscale.py）放大，這裡存成新圖——很快，不重畫
  放大＋補細節：走生成佇列（pipeline._hires）：放大後切成約 1MP 的重疊小塊，每塊用低修改幅度重畫再貼回
1 倍是大小不變：補細節時在放大的畫布上重畫再縮回（core/upscale.detail_size）；只放大時要選放大模型，不然圖不會變。
放大模型照模型總資料夾的分類（ESRGAN、RealESRGAN、upscale_models…）找，同一個檔案放在兩個地方只列一次。
"""

import json
import time
from pathlib import Path

from .resources import res_dirs
from .state import C

SCALES = (1, 1.5, 2, 3, 4)
MAX_SIDE = 4096            # 跟精修的上限一樣（refine.SIZE_MAX）
EXTS = (".safetensors", ".pth", ".pt")


def display_name(filename):
    """檔名 → 顯示的名字：副檔名拿掉（4x-UltraSharp.pth.pt 這種疊了兩層的也一起拿掉）"""
    name = filename
    changed = True
    while changed:
        changed = False
        for ext in EXTS:
            if name.lower().endswith(ext) and len(name) > len(ext):
                name, changed = name[:-len(ext)], True
    return name


def scan_files():
    """放大模型資料夾裡的檔案（ESRGAN 的模型多半是 .pth，資源庫的掃描不收，所以這裡自己找）"""
    files = []
    for d in res_dirs("upscaler"):
        try:
            files += sorted(x for x in Path(d).rglob("*") if x.is_file())
        except OSError:
            continue
    return files


def upscalers():
    """放大模型：[{name, path, bytes}]，照名字排；同名、同大小的（同一個檔案放在兩個資料夾）只留一個"""
    out, seen = [], set()
    for p in scan_files():
        if not p.name.lower().endswith(EXTS):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        name = display_name(p.name)
        if (name.lower(), size) in seen:
            continue
        seen.add((name.lower(), size))
        out.append({"name": name, "path": str(p), "bytes": size})
    out.sort(key=lambda x: x["name"].lower())
    return out


def find(name):
    return next((m for m in upscalers() if m["name"] == name), None)


def target_size(w, h, scale):
    """放大後的大小（對齊 16，生成要求）；超過上限回 None"""
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return None
    if scale not in SCALES:
        return None
    tw, th = round(w * scale / 16) * 16, round(h * scale / 16) * 16
    return (tw, th) if max(tw, th) <= MAX_SIDE else None


def _source_meta(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.info.get("parameters") or "", im.info.get("coco") or ""
    except OSError:
        return "", ""


def save_result(src_path, img, scale, model_name, out_dir=None):
    """只放大的結果存成新圖：生成參數照原圖的留著，coco JSON 記 init（版本列靠它）與 upscale"""
    from PIL import PngImagePlugin
    src = Path(src_path)
    params, coco = _source_meta(src)
    try:
        c = json.loads(coco) if coco else {}
    except ValueError:
        c = {}
    c = {k: v for k, v in c.items() if k not in ("init", "strength", "inpaint", "faces", "edit_of", "rembg", "upscale")}
    c.update({"init": str(src), "upscale": {"scale": scale, "model": model_name or "Lanczos", "detail": False}})
    info = PngImagePlugin.PngInfo()
    if params:
        info.add_text("parameters", params)
    info.add_text("coco", json.dumps(c, ensure_ascii=False))
    folder = Path(out_dir or (Path(C.OUTPUT_DIR) / time.strftime("%Y-%m-%d")))
    folder.mkdir(parents=True, exist_ok=True)
    tag = f"x{scale:g}"
    out = folder / f"{src.stem}_{tag}.png"
    n = 2
    while out.exists():
        out = folder / f"{src.stem}_{tag}_{n}.png"
        n += 1
    img.save(out, "PNG", pnginfo=info)
    return out
