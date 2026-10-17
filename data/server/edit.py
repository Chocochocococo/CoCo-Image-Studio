"""06 輸出：編輯中的工作檔（圖層）與輸出。

工作檔（使用者選的做法 A）：edits/<id>/project.json＋每個圖層一張 PNG（清單的 file，layer-<圖層 id>-<亂數>.png；
以前的工作檔沒記 file，是 layer-<圖層 id>.png）。改過的圖層每次存成新檔，清單原子換上去之後才刪舊的。
編輯時前端自動存（只送改過的圖層）；再打開同一張圖時問要不要接著做；「捨棄編輯」才刪。
一張底圖只有一份工作檔。文字圖層另外記文字的設定（text），之後還能再改字；浮水印圖層記 wm；鎖定透明記 lock。
浮水印的圖片（logo）存在 edits/watermark.png，每張圖、批次輸出都共用。

輸出：前端把看得到的圖層合併成一張 PNG 送來，這裡照設定縮放、轉檔、存成新檔。
存進圖庫一律保留生成參數；匯出到別的資料夾可以選要不要保留（使用者筆記）。
PNG 的參數寫 parameters 文字、coco JSON 記從哪張編輯來的（edit_of）；JPG／WebP 寫 EXIF UserComment
（跟 A1111 一樣，gallery_index 讀得回來）。
"""

import base64
import binascii
import io
import json
import re
import secrets
import shutil
import threading
import time

from i18n import tr
from pathlib import Path

from .state import C, SET

EDIT_DIR = Path(getattr(C, "EDIT_DIR", None) or (Path(C.DB_PATH).parent / "edits"))
_ID = re.compile(r"^[0-9a-f]{6,32}$")
_FILE = re.compile(r"^layer-[0-9a-f]{6,32}(?:-[0-9a-f]{8})?\.png$")     # 圖層的圖檔名（清單裡記的）
MAX_LAYERS = 32
MAX_SIDE = 8192
MAX_BYTES = 96 * 1024 * 1024          # 一張圖層的 data URL（8K 的 PNG 也放得下）
MAX_TEXT = 20000                      # 文字圖層的設定（JSON）最多這麼長
FORMATS = {"png": ("PNG", ".png"), "jpg": ("JPEG", ".jpg"), "webp": ("WEBP", ".webp")}
_LOCK = threading.Lock()


class _Bad(ValueError):
    """存工作檔時送來的東西不對：回錯誤訊息（這次新寫的圖先刪掉）"""


def _png(data_url, size=None):
    """data URL → PNG 位元組（驗過是圖、大小對）；不對回 None"""
    from PIL import Image
    text = str(data_url or "")
    head, _, body = text.partition(",")
    if not head.startswith("data:image/png;base64") or not body or len(body) > MAX_BYTES:
        return None
    try:
        raw = base64.b64decode(body, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            if size and im.size != tuple(size):
                return None
            if max(im.size) > MAX_SIDE:
                return None
    except (binascii.Error, ValueError, OSError):
        return None
    return raw


def _dir(pid):
    return EDIT_DIR / pid if _ID.match(str(pid or "")) else None


def _meta(pid):
    d = _dir(pid)
    try:
        m = json.loads((d / "project.json").read_text(encoding="utf-8")) if d else None
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def _src_row(con, sid):
    try:
        return con.execute("SELECT id, path, filename, width, height, prompt, seed, sampler, steps, cfg, model, loras "
                           "FROM images WHERE id = ?", (int(sid),)).fetchone()
    except (TypeError, ValueError):
        return None


def _layer_file(entry):
    """清單裡這個圖層的圖檔名（以前的工作檔沒記，就是 layer-<id>.png）"""
    name = str(entry.get("file") or "")
    return name if _FILE.match(name) else f"layer-{entry.get('id')}.png"


def _write_layer(d, lid, save, written):
    """新的圖層圖存成新檔名，不蓋掉清單現在指著的那張：save(路徑) 寫入，寫好的記進 written。回傳檔名"""
    while True:
        name = f"layer-{lid}-{secrets.token_hex(4)}.png"
        if not (d / name).exists():
            break
    written.append(d / name)
    save(d / name)
    return name


def _switch(d, meta):
    """清單原子換上去（這一步之前出錯，舊的清單與圖都還在）"""
    tmp = d / "project.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / "project.json")


def _sweep(d, meta):
    """清單換好之後，刪掉清單沒用到的圖層圖；刪不掉的下次存檔再刪"""
    used = {x["file"] for x in meta["layers"]}
    for f in d.glob("layer-*.png"):
        if f.name not in used:
            try:
                f.unlink()
            except OSError:
                pass


def save_project(con, data):
    """新建或更新工作檔。data：{id?, src_id, w, h, active, layers: [{id, name, visible, opacity, png?}]}
    （png 是改過的圖層才有；沒給的沿用上次存的）。回傳 {id} 或 {error}。
    改過的圖層存成新檔、全部存好才換清單、換好才刪舊圖：中途出錯（第二張圖不對、磁碟滿了），舊的工作檔完整留著"""
    row = _src_row(con, data.get("src_id"))
    if not row:
        return {"error": tr("srv.ed.noBase")}
    try:
        w, h = int(data.get("w")), int(data.get("h"))
    except (TypeError, ValueError):
        return {"error": tr("srv.ed.badSize")}
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        return {"error": tr("srv.ed.badSize")}
    layers = data.get("layers") or []
    if not isinstance(layers, list) or not layers or len(layers) > MAX_LAYERS:
        return {"error": tr("srv.ed.badLayer")}
    with _LOCK:
        pid = str(data.get("id") or "")
        old = _meta(pid) if pid else None
        if not old:
            pid = find_project(row["path"]) or secrets.token_hex(6)
            old = _meta(pid)
        d = EDIT_DIR / pid
        d.mkdir(parents=True, exist_ok=True)
        files = {x.get("id"): _layer_file(x) for x in (old or {}).get("layers") or [] if isinstance(x, dict)}
        keep, written = [], []
        try:
            for L in layers:
                lid = str(L.get("id") or "")
                if not _ID.match(lid):
                    raise _Bad(tr("srv.ed.badLayer"))
                if L.get("png"):
                    raw = _png(L["png"], (w, h))
                    if raw is None:
                        raise _Bad(tr("srv.ed.badLayerImg"))
                    name = _write_layer(d, lid, lambda f, raw=raw: f.write_bytes(raw), written)
                else:
                    name = files.get(lid) or f"layer-{lid}.png"
                    if not (d / name).is_file():
                        raise _Bad(tr("srv.ed.noLayerImg"))
                try:
                    op = max(0.0, min(1.0, float(L.get("opacity", 1))))
                except (TypeError, ValueError):
                    op = 1.0
                entry = {"id": lid, "file": name, "name": str(L.get("name") or "")[:80],
                         "visible": bool(L.get("visible", True)), "opacity": op}
                tx = L.get("text")
                if isinstance(tx, dict) and len(json.dumps(tx, ensure_ascii=False)) <= MAX_TEXT:
                    entry["text"] = tx
                if L.get("wm"):
                    entry["wm"] = True
                if L.get("lock"):                           # 鎖定透明
                    entry["lock"] = True
                keep.append(entry)
            meta = {"src": row["path"], "w": w, "h": h, "layers": keep,
                    "active": str(data.get("active") or keep[-1]["id"]),
                    "created": (old or {}).get("created") or time.time(), "updated": time.time()}
            _switch(d, meta)
        except BaseException as e:
            for f in written:
                f.unlink(missing_ok=True)
            if isinstance(e, _Bad):
                return {"error": str(e)}
            raise
        _sweep(d, meta)
    return {"ok": True, "id": pid}


def find_project(path):
    """這張底圖（路徑）有沒有工作檔：回傳 id 或 None"""
    if not EDIT_DIR.is_dir():
        return None
    low = str(Path(path)).lower()
    for d in EDIT_DIR.iterdir():
        m = _meta(d.name) if d.is_dir() else None
        if m and str(Path(m.get("src", ""))).lower() == low:
            return d.name
    return None


def project_for(con, sid):
    """打開某張圖時：有工作檔就回傳內容（圖層清單，圖另外抓），沒有回 None"""
    row = _src_row(con, sid)
    pid = find_project(row["path"]) if row else None
    m = _meta(pid) if pid else None
    return {"id": pid, **m} if m else None


def list_projects(con):
    """未完成的編輯：新的在前，附上底圖的 id（底圖不在圖庫了就沒有）"""
    from .library import thumb_ver
    out = []
    if not EDIT_DIR.is_dir():
        return out
    for d in EDIT_DIR.iterdir():
        m = _meta(d.name) if d.is_dir() else None
        if not m:
            continue
        r = con.execute("SELECT id, path, mtime, filename FROM images WHERE lower(path) = lower(?)", (m.get("src", ""),)).fetchone()
        out.append({"id": d.name, "src_id": r[0] if r else None, "v": thumb_ver(r[1], r[2]) if r else "",
                    "filename": r[3] if r else Path(m.get("src", "")).name, "layers": len(m.get("layers") or []),
                    "w": m.get("w"), "h": m.get("h"), "updated": m.get("updated", 0)})
    out.sort(key=lambda x: -x["updated"])
    return out


def layer_path(pid, lid):
    d = _dir(pid)
    if not d or not _ID.match(str(lid or "")):
        return None
    hit = next((x for x in (_meta(pid) or {}).get("layers") or [] if isinstance(x, dict) and x.get("id") == lid), None)
    f = d / (_layer_file(hit) if hit else f"layer-{lid}.png")
    return f if f.is_file() else None


# ── 浮水印的圖片 ──
def logo_file():
    return EDIT_DIR / "watermark.png"


def logo_version():
    f = logo_file()
    return int(f.stat().st_mtime * 1000) if f.is_file() else 0


def save_logo(data):
    raw = _png(data.get("png"))
    if raw is None:
        return {"error": tr("srv.ed.badImg")}
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EDIT_DIR / "watermark.tmp"
    tmp.write_bytes(raw)
    tmp.replace(logo_file())
    return {"ok": True, "v": logo_version()}


def clear_logo():
    logo_file().unlink(missing_ok=True)
    return {"ok": True, "v": 0}


def discard(pid):
    d = _dir(pid)
    if not d or not d.is_dir():
        return False
    with _LOCK:
        shutil.rmtree(d, ignore_errors=True)
    return True


def _source_meta(path):
    """底圖的生成參數：(parameters 文字, coco JSON 字串)。讀不到就是空的"""
    from PIL import Image
    params, coco = "", ""
    try:
        with Image.open(path) as im:
            info = dict(im.info or {})
            params, coco = str(info.get("parameters") or ""), str(info.get("coco") or "")
            if not params and Path(path).suffix.lower() in (".jpg", ".jpeg", ".webp"):
                try:
                    from core import meta_formats as MF
                except ImportError:
                    import meta_formats as MF
                exif = im.getexif()
                try:
                    sub = exif.get_ifd(0x8769)
                except Exception:
                    sub = {}
                params = MF.user_comment(sub.get(0x9286) or exif.get(0x9286)).strip()
    except OSError:
        pass
    return params, coco


def _target(row, base, ext, size, name_only=False):
    """06 輸出的檔名（core/naming.py）：編輯用原圖的檔名與參數當變數，拼接的 {name} 是 collage"""
    try:
        import naming
    except ImportError:
        from core import naming
    cfg = SET.load()
    req = {"width": size[0], "height": size[1]}
    if row is None:
        kind, vals = "collage", naming.values(req, "")
        vals["name"] = "collage"
    else:
        kind = "edit"
        try:
            loras = [(str(x.get("file") or ""), x.get("weight", 1)) for x in json.loads(row["loras"] or "[]") if isinstance(x, dict)]
        except (ValueError, TypeError):
            loras = []
        req.update(steps=row["steps"] if row["steps"] is not None else "", cfg=row["cfg"] if row["cfg"] is not None else "",
                   prompt=row["prompt"] or "")
        vals = naming.values(req, row["seed"] if row["seed"] is not None else "", row["model"] or "", loras, row["sampler"] or "")
        vals["name"] = Path(row["filename"]).stem
    pattern = cfg.get("COLLAGE_NAME" if kind == "collage" else "EDIT_NAME") or naming.DEFAULTS[kind]
    return naming.target(base, pattern, vals, ext=ext, name_only=name_only, default=naming.DEFAULTS[kind])


def export(con, data):
    """輸出合併好的圖。data：{src_id, png, format, quality, width, height, dest: "library" | "folder", folder, keep_meta}
    拼接（collage: [圖庫 id…]）沒有單一來源圖：不帶生成參數。
    檔名照 ⚙ 設定的格式（EDIT_NAME／COLLAGE_NAME，core/naming.py）：存進圖庫時整個格式都用（/ 前面是輸出資料夾底下的資料夾），
    匯出到其他資料夾時只用檔名那段（使用者選的資料夾就是要存的地方）。
    回傳 {path, id?（存進圖庫的話）} 或 {error}"""
    from PIL import Image, PngImagePlugin
    collage = bool(data.get("collage"))
    row = None if collage else _src_row(con, data.get("src_id"))
    if not row and not collage:
        return {"error": tr("srv.ed.noBase")}
    raw = _png(data.get("png"))
    if raw is None:
        return {"error": tr("srv.ed.badImg")}
    fmt = str(data.get("format") or "png").lower()
    if fmt not in FORMATS:
        return {"error": tr("srv.ed.badType")}
    pil_fmt, ext = FORMATS[fmt]
    try:
        quality = max(1, min(100, int(data.get("quality") or 92)))
    except (TypeError, ValueError):
        quality = 92
    to_library = data.get("dest") != "folder"
    keep = False if collage else True if to_library else bool(data.get("keep_meta", True))    # 圖庫裡的一律保留
    if to_library:
        folder = Path(C.OUTPUT_DIR)
    else:
        folder = Path(str(data.get("folder") or "").strip().strip('"'))
        if not str(folder) or str(folder) == ".":
            return {"error": tr("srv.pickFolder")}
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"error": tr("srv.mkdirFail", e=e)}

    im = Image.open(io.BytesIO(raw))
    im.load()
    try:
        w, h = int(data.get("width") or im.width), int(data.get("height") or im.height)
    except (TypeError, ValueError):
        w, h = im.size
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        return {"error": tr("srv.ed.badOutSize")}
    if (w, h) != im.size:
        im = im.convert("RGBA").resize((w, h), Image.LANCZOS)
    alpha = im.mode in ("RGBA", "LA") and im.getchannel("A").getextrema()[0] < 255
    if fmt == "jpg" or not alpha:
        if alpha:                                   # JPG 沒有透明：鋪白底
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
    else:
        im = im.convert("RGBA")

    params, coco = _source_meta(row["path"]) if keep else ("", "")
    kw = {}
    if fmt == "png":
        info = PngImagePlugin.PngInfo()
        if params:
            info.add_text("parameters", params)
        if keep:
            try:
                c = json.loads(coco) if coco else {}
            except ValueError:
                c = {}
            c = {k: v for k, v in c.items() if k not in ("init", "strength", "inpaint", "faces")}   # 不是 img2img 的版本
            c["edit_of"] = row["path"]
            info.add_text("coco", json.dumps(c, ensure_ascii=False))
        kw = {"pnginfo": info, "optimize": False}
    else:
        kw = {"quality": quality}
        if fmt == "webp" and quality >= 100:
            kw = {"lossless": True}
        if params:
            exif = Image.Exif()
            exif[0x8769] = {0x9286: b"UNICODE\x00" + params.encode("utf-16-be")}
            kw["exif"] = exif.tobytes()

    out = _target(row, folder, ext, (w, h), name_only=not to_library)     # 先建了空檔佔住名字（core/naming.py）
    try:
        im.save(out, pil_fmt, **kw)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    res = {"ok": True, "path": str(out)}
    if to_library:
        try:
            import gallery_index as GI
        except ImportError:
            from core import gallery_index as GI
        if GI.index_file(con, out):
            con.commit()
            r = con.execute("SELECT id FROM images WHERE path = ?", (str(out),)).fetchone()
            res["id"] = r[0] if r else None
    return res
