# -*- coding: utf-8 -*-
"""精修（img2img）：底圖的查詢、同一張底圖做出來的各個版本、搬移後跟著改路徑。

版本關係不另外存檔：img2img 的輸出在中繼資料裡記著底圖的完整路徑
（coco JSON 的 init），圖庫索引把它放進 images.extra，這裡直接查。
"""

import json
from pathlib import Path

from .library import thumb_ver
from .state import C

from i18n import tr

# 模型（SDXL 的 VAE 是 8 倍、Anima 還要再 ×2）要求寬高是 16 的倍數
SIZE_STEP = 16
SIZE_MIN, SIZE_MAX = 256, 4096

_COLS = ("id, filename, path, mtime, width, height, rating, model, seed, prompt, negative, loras, extra, "
         "sampler, scheduler, steps, cfg, clip_skip")


# 用指示修圖的其他參考圖（圖2、圖3）最多幾張：每多一張，主模型要看的圖片 token 多約 4096 個（Qwen 的文字編碼器另外多約 1000 個）
EDIT_REFS_MAX = 2
# 能用指示修圖的架構：Qwen Image 2.1、FLUX.2 klein（4B、9B）、FLUX.1 Kontext
EDIT_ARCHS = ("qwen21", "klein4", "klein9", "kontext")


def snap(v, step=SIZE_STEP):
    """寬高對齊到 16 的倍數（四捨五入），並限制在合理範圍"""
    v = int(round(float(v) / step)) * step
    return max(SIZE_MIN, min(SIZE_MAX, v))


def _item(r):
    d = dict(r)
    d["v"] = thumb_ver(d["path"], d.pop("mtime"))
    ex = json.loads(d.pop("extra") or "{}")
    d["strength"] = ex.get("strength")
    d["init"] = ex.get("init") or ""
    # 開著品質標籤生成的圖：精修頁預填使用者自己寫的，不是含自動標籤的完整版
    d["inpaint"] = ex.get("inpaint") or ""           # 局部重繪的範圍（whole / masked），整張重繪是空的
    d["faces"] = ex.get("faces") or 0                # 修臉修了幾張臉
    d["rembg"] = ex.get("rembg") or 0                # 去背出來的（版本列標「去背」）
    d["upscale"] = ex.get("upscale") or None          # 高清化出來的：{scale, detail}
    d["outpaint"] = ex.get("outpaint") or 0           # 擴圖出來的（版本列標「擴圖」）
    d["edit"] = ex.get("edit") or 0                   # 用 Qwen 修圖出來的（版本列標「修圖」）
    d["user_prompt"] = ex.get("user_prompt")
    d["user_negative"] = ex.get("user_negative")
    # 精修照原圖的模型與參數時要用（Anima 的 shift 記在 extra）
    shift = ex.get("Shift", ex.get("shift"))
    try:
        d["shift"] = float(shift) if shift not in (None, "") else None
    except (TypeError, ValueError):
        d["shift"] = None
    # CoCo 存圖時 <lora:…> 不在提示詞裡、另外記在 loras 欄位；精修頁要補回提示詞，不然重畫時 LoRA 就掉了
    try:
        loras = json.loads(d.pop("loras") or "[]")
    except ValueError:
        loras = []
    d["loras"] = [{"file": str(x.get("file") or ""), "weight": x.get("weight", 1)}
                  for x in loras if isinstance(x, dict) and x.get("file")]
    return d


def get_image(con, image_id):
    try:
        r = con.execute(f"SELECT {_COLS} FROM images WHERE id = ?", (int(image_id),)).fetchone()
    except (TypeError, ValueError):
        return None
    return _item(r) if r else None


OUTPAINT_MAX = 2048            # 每一邊最多擴幾像素


def clean_outpaint(sides, size):
    """擴圖送來的四邊：整數、不是負的、每邊最多 OUTPAINT_MAX、加大後不超過 SIZE_MAX。
    回傳 {sides: {left, top, right, bottom}, size: (寬, 高)} 或 {error}"""
    if not isinstance(sides, dict):
        return {"error": tr("srv.rf.outBad")}
    out = {}
    for k in ("left", "top", "right", "bottom"):
        try:
            v = int(round(float(sides.get(k) or 0)))
        except (TypeError, ValueError):
            return {"error": tr("srv.rf.outNum")}
        if v < 0 or v > OUTPAINT_MAX:
            return {"error": tr("srv.rf.outMax", n=OUTPAINT_MAX)}
        out[k] = v
    if not any(out.values()):
        return {"error": tr("srv.rf.outSide")}
    W, H = size[0] + out["left"] + out["right"], size[1] + out["top"] + out["bottom"]
    if max(W, H) > SIZE_MAX:
        return {"error": tr("srv.rf.outTooBig", n=SIZE_MAX)}
    return {"sides": out, "size": (W, H)}


def items(con, ids, limit=500):
    """好幾張圖的資料（跟 versions 的 source 一樣，不含路徑）；ids 是「1,2,3」，找不到的略過、照給的順序"""
    out = []
    for x in str(ids).split(",")[:limit]:
        it = get_image(con, x.strip()) if x.strip() else None
        if it:
            out.append({k: v for k, v in it.items() if k != "path"})
    return out


def clean_faces(faces, size, limit=12):
    """修臉送來的臉：外框要在圖裡、有大小；提示詞限長。回傳 {faces} 或 {error}"""
    if not isinstance(faces, list) or not faces:
        return {"error": tr("srv.rf.noFace")}
    W, H = size
    out = []
    for f in faces[:limit]:
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in (f.get("box") or [])[:4])
        except (TypeError, ValueError, AttributeError):
            return {"error": tr("srv.rf.badFace")}
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return {"error": tr("srv.rf.badFace")}
        item = {"box": [x0, y0, x1, y1]}
        p = str(f.get("prompt") or "").strip()[:500]
        if p:
            item["prompt"] = p
        out.append(item)
    return {"faces": out}


def init_source(con, image_id):
    """生成用的底圖：回傳 {path, width, height}；找不到或檔案不見了回 {error}"""
    img = get_image(con, image_id)
    if not img:
        return {"error": tr("srv.rf.noBase")}
    if not Path(img["path"]).is_file():
        return {"error": tr("srv.rf.baseGone", file=img['filename'])}
    return {"path": img["path"], "width": img["width"], "height": img["height"]}


def versions(con, image_id):
    """這張圖的精修版本（用它當底圖做出來的，新的在前），以及它自己的底圖（上一版）"""
    src = get_image(con, image_id)
    if not src:
        return {"error": tr("srv.imgNotFound")}
    rows = con.execute(
        f"""SELECT {_COLS} FROM images
            WHERE lower(json_extract(extra, '$.init')) = lower(?)
            ORDER BY mtime DESC, id DESC""", (src["path"],)).fetchall()
    parent = None
    if src["init"]:
        r = con.execute(f"SELECT {_COLS} FROM images WHERE lower(path) = lower(?)", (src["init"],)).fetchone()
        parent = _item(r) if r else None
    strip = lambda d: {k: v for k, v in d.items() if k != "path"}
    return {"source": strip(src), "parent": strip(parent) if parent else None,
            "items": [strip(_item(r)) for r in rows]}


MASK_DIR = Path(getattr(C, "MASK_DIR", None) or (Path(C.DB_PATH).parent / "masks"))
MASK_MAX_BYTES = 8 * 1024 * 1024


def save_mask(data_url, size):
    """精修頁畫的遮罩（canvas 匯出的 PNG data URL）→ 存成黑白 PNG，回傳路徑或 {error}

    存在 masks/，檔名是內容的雜湊：同一個遮罩重畫好幾次只存一份，
    版本的中繼資料記著路徑，之後可以「用同一個遮罩再畫一次」
    """
    import base64
    import hashlib
    import io
    try:
        from core import inpaint as IP
    except ImportError:
        import inpaint as IP
    from PIL import Image

    text = str(data_url or "")
    head, _, body = text.partition(",")
    if not head.startswith("data:image/png;base64") or not body:
        return {"error": tr("srv.rf.maskFmt")}
    try:
        raw = base64.b64decode(body, validate=True)
        if len(raw) > MASK_MAX_BYTES:
            return {"error": tr("srv.rf.maskBig")}
        mask = IP.load_mask(Image.open(io.BytesIO(raw)), size)
    except Exception:
        return {"error": tr("srv.rf.maskBad")}
    if not mask.getbbox():
        return {"error": tr("srv.rf.maskEmpty")}
    buf = io.BytesIO()
    mask.save(buf, "PNG", optimize=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    path = MASK_DIR / f"{hashlib.sha1(buf.getvalue()).hexdigest()[:16]}.png"
    if not path.exists():
        path.write_bytes(buf.getvalue())
    return {"path": str(path)}


def save_sketch(data_url, size):
    """精修頁塗的顏色（canvas 匯出、透明底的 PNG data URL）→ 存成 RGBA PNG，回傳 {path}、{path: None}（沒塗）或 {error}

    跟遮罩一樣存在 masks/、檔名是內容的雜湊（前面加 sk-）。重畫時先把顏色蓋在底圖上當起點
    """
    import base64
    import hashlib
    import io
    from PIL import Image

    text = str(data_url or "")
    head, _, body = text.partition(",")
    if not head.startswith("data:image/png;base64") or not body:
        return {"error": tr("srv.rf.sketchFmt")}
    try:
        raw = base64.b64decode(body, validate=True)
        if len(raw) > MASK_MAX_BYTES * 4:
            return {"error": tr("srv.rf.sketchBig")}
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGBA")
    except Exception:
        return {"error": tr("srv.rf.sketchBad")}
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.NEAREST)
    if not im.getchannel("A").getbbox():
        return {"path": None}
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    path = MASK_DIR / f"sk-{hashlib.sha1(buf.getvalue()).hexdigest()[:16]}.png"
    if not path.exists():
        path.write_bytes(buf.getvalue())
    return {"path": str(path)}


def rename_inits(con, moved):
    """底圖被搬走了：版本裡記的底圖路徑跟著改，不然版本清單會斷掉"""
    n = 0
    for old, new in (moved or {}).items():
        cur = con.execute(
            "UPDATE images SET extra = json_set(extra, '$.init', ?) "
            "WHERE lower(json_extract(extra, '$.init')) = lower(?)", (new, old))
        n += cur.rowcount
    return n
