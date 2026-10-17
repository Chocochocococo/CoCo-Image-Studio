""".kra（Krita 的檔案）：把 06 輸出的工作檔存成 .kra，也能把 .kra 讀回來接著編輯。

使用者筆記：想把編輯到一半的檔案存起來、之後讀回來接著做；也想拿到 Krita 裡繼續修。
所以一個檔案兩邊通用——CoCo 存的 .kra 用 Krita 打開就是一般的圖層檔；Krita 存的 .kra 也讀得回來。

.kra 是 zip：
  mimetype（第一個、不壓縮）＝application/x-krita
  maindoc.xml：圖的大小、圖層清單（最上面的寫在最前面；名字、顯示、不透明度 0～255、鎖定透明）
  <名字>/layers/layerN：每層的像素，Krita 的磚塊格式——
      VERSION 2 / TILEWIDTH 64 / TILEHEIGHT 64 / PIXELSIZE 4 / DATA 磚塊數，
      每塊「x,y,LZF,長度」一行，接著 1 個位元組（1＝LZF 壓縮過、0＝沒壓縮）＋資料；
      資料是 64×64 的 BGRA、沒有預乘透明度。沒壓縮的是逐像素排列（BGRA BGRA…）；
      壓縮過的是先按色版分開排（先全部的 B，再 G、R、A）再壓縮（用 Krita 5.2 實際讀寫確認過）
  mergedimage.png（合併後的樣子）、preview.png（縮圖）
  coco.json：CoCo 自己的資訊（原圖是哪張、文字圖層的設定、浮水印圖層），Krita 不會讀也不會留

讀 Krita 存的檔：一般圖層（paintlayer）照讀；群組攤平成裡面的圖層（顯示、不透明度一起算）；
其他種類（濾鏡、填色、向量、檔案圖層…）讀不了就跳過並告訴使用者；一層都讀不到就用合併後的圖當一層。
8 位元 RGBA 以外的色彩空間（16 位元、CMYK…）也一樣跳過。
"""

import hashlib
import io
import json
import secrets
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from . import edit
from .state import C

from i18n import tr

MIME = b"application/x-krita"
NS = "http://www.calligra.org/DTD/krita"
TILE = 64
DOC_NAME = "coco"                     # zip 裡圖層資料夾的名字
# 色彩設定檔：LittleCMS 內建的 sRGB（Krita 自己存檔也常用這個，名字一樣是「sRGB built-in」）。
# 整張圖與每一層都附上，Krita 開檔才不會說「無法載入設定檔」；CoCo 的圖本來就是 sRGB，顏色不受影響
ICC_NAME = "sRGB built-in"
_ICC = None


def srgb_icc():
    global _ICC
    if _ICC is None:
        from PIL import ImageCms
        _ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _ICC


# ── 磚塊格式 ──
def lzf_decompress(data, out_len):
    """liblzf 的解壓縮（Krita 磚塊用的）。往回複製的距離比長度短時要一個一個來（會用到剛複製出來的）"""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:                                   # 原樣的一段：ctrl＋1 個位元組
            out += data[i:i + ctrl + 1]
            i += ctrl + 1
            continue
        length = ctrl >> 5
        if length == 7:
            length += data[i]
            i += 1
        ref = len(out) - ((ctrl & 0x1F) << 8) - data[i] - 1
        i += 1
        length += 2
        if ref < 0:
            raise ValueError(tr("srv.kra.lzfBad"))
        if ref + length <= len(out):
            out += out[ref:ref + length]
        else:
            for k in range(length):
                out.append(out[ref + k])
    if len(out) != out_len:
        raise ValueError(tr("srv.kra.lzfLen"))
    return bytes(out)


def encode_layer(arr):
    """RGBA 的 numpy 陣列（H×W×4）→ Krita 的圖層資料。全透明的磚塊不寫（預設就是透明）"""
    import numpy as np
    h, w = arr.shape[:2]
    tiles = []
    for ty in range(0, h, TILE):
        for tx in range(0, w, TILE):
            part = arr[ty:ty + TILE, tx:tx + TILE]
            if not part[..., 3].any():
                continue
            tile = np.zeros((TILE, TILE, 4), np.uint8)
            tile[:part.shape[0], :part.shape[1]] = part
            px = tile[..., [2, 1, 0, 3]].tobytes()      # BGRA，沒壓縮就是逐像素排列
            tiles.append(f"{tx},{ty},LZF,{len(px) + 1}\n".encode() + b"\x00" + px)
    head = f"VERSION 2\nTILEWIDTH {TILE}\nTILEHEIGHT {TILE}\nPIXELSIZE 4\nDATA {len(tiles)}\n".encode()
    return head + b"".join(tiles)


def decode_layer(raw, w, h, ox=0, oy=0):
    """Krita 的圖層資料 → W×H 的 RGBA numpy 陣列（ox, oy 是這層的位移；超出圖的部分丟掉）"""
    import numpy as np
    out = np.zeros((h, w, 4), np.uint8)
    pos = 0

    def line():
        nonlocal pos
        end = raw.index(b"\n", pos)
        s = raw[pos:end].decode("ascii")
        pos = end + 1
        return s

    head = {}
    while True:
        k, _, v = line().partition(" ")
        head[k] = v
        if k == "DATA":
            break
    tw, th, ps = int(head.get("TILEWIDTH", TILE)), int(head.get("TILEHEIGHT", TILE)), int(head.get("PIXELSIZE", 4))
    if ps != 4:
        raise ValueError(tr("srv.kra.only8bit"))
    size = tw * th * ps
    for _ in range(int(head["DATA"])):
        x, y, _codec, n = line().split(",")
        x, y, n = int(x) + ox, int(y) + oy, int(n)
        blob = raw[pos:pos + n]
        pos += n
        if blob[0]:                                     # 壓縮過：解開後是按色版分開排的
            tile = np.frombuffer(lzf_decompress(blob[1:], size), np.uint8).reshape(4, th * tw).T.reshape(th, tw, 4)
        else:                                           # 沒壓縮：一般的逐像素排列
            tile = np.frombuffer(bytes(blob[1:1 + size]), np.uint8).reshape(th, tw, 4)
        tile = tile[..., [2, 1, 0, 3]]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + tw), min(h, y + th)
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] = tile[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


def _digest(arr):
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ── 存成 .kra ──
def export_kra(con, data):
    """data：{id（工作檔）, folder}。回傳 {path} 或 {error}"""
    import numpy as np
    from PIL import Image
    pid = str(data.get("id") or "")
    meta = edit._meta(pid)
    if not meta:
        return {"error": tr("srv.kra.noWork")}
    folder = Path(str(data.get("folder") or "").strip().strip('"'))
    if not str(folder) or str(folder) == ".":
        return {"error": tr("srv.pickFolder")}
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"error": tr("srv.mkdirFail", e=e)}
    w, h = int(meta["w"]), int(meta["h"])
    layers = []
    for i, L in enumerate(meta["layers"]):
        f = edit.layer_path(pid, L["id"])
        if not f:
            return {"error": tr("srv.ed.noLayerImg")}
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGBA"))
        layers.append((f"layer{i + 1}", L, arr))
    merged = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for _, L, arr in layers:
        if L.get("visible", True):
            im = Image.fromarray(arr, "RGBA")
            if L.get("opacity", 1) < 1:
                a = np.asarray(im.getchannel("A"), np.float32) * float(L["opacity"])
                im.putalpha(Image.fromarray(a.round().astype(np.uint8), "L"))
            merged = Image.alpha_composite(merged, im)
    sel = ' selected="true"'
    xml_layers = "".join(
        f'   <layer name="{_attr(L.get("name") or fn)}" filename="{fn}" nodetype="paintlayer" x="0" y="0" '
        f'visible="{1 if L.get("visible", True) else 0}" opacity="{round(255 * float(L.get("opacity", 1)))}" compositeop="normal" '
        f'colorspacename="RGBA" channelflags="" channellockflags="{"1110" if L.get("lock") else ""}" '
        f'uuid="{{{uuid.uuid4()}}}" locked="0" collapsed="0" intimeline="0" onionskin="0" colorlabel="0"'
        f'{sel if L["id"] == meta.get("active") else ""}/>\n'
        for fn, L, _ in reversed(layers))                  # Krita：最上面的寫在最前面
    title = Path(meta.get("src", "")).stem or "coco"
    maindoc = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
               "<!DOCTYPE DOC PUBLIC '-//KDE//DTD krita 2.0//EN' 'http://www.calligra.org/DTD/krita-2.0.dtd'>\n"
               f'<DOC xmlns="{NS}" kritaVersion="5.2.0" syntaxVersion="2" editor="Krita">\n'
               f' <IMAGE mime="application/x-kra" name="{DOC_NAME}" width="{w}" height="{h}" x-res="100" y-res="100" '
               f'colorspacename="RGBA" profile="{ICC_NAME}" description="{_attr(title)}">\n'
               f'  <layers>\n{xml_layers}  </layers>\n </IMAGE>\n</DOC>\n')
    coco = {"version": 1, "src": meta.get("src", ""), "active": meta.get("active"),
            "layers": [{"file": fn, "id": L["id"], "name": L.get("name", ""), "digest": _digest(arr),
                        **{k: L[k] for k in ("text", "wm", "lock") if L.get(k)}} for fn, L, arr in layers]}
    out = _unique(folder / f"{title}.kra")
    tmp = out.with_suffix(".kra.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), MIME, compress_type=zipfile.ZIP_STORED)   # 一定要第一個、不壓縮
        z.writestr("maindoc.xml", maindoc)
        icc = srgb_icc()
        z.writestr(f"{DOC_NAME}/annotations/icc", icc)
        for fn, _, arr in layers:
            z.writestr(f"{DOC_NAME}/layers/{fn}", encode_layer(arr))
            z.writestr(f"{DOC_NAME}/layers/{fn}.defaultpixel", b"\x00\x00\x00\x00")
            z.writestr(f"{DOC_NAME}/layers/{fn}.icc", icc)     # 沒有的話 Krita 開檔會說「無法載入設定檔」
        z.writestr("mergedimage.png", _png_bytes(merged))
        thumb = merged.copy()
        thumb.thumbnail((256, 256))
        z.writestr("preview.png", _png_bytes(thumb))
        z.writestr("coco.json", json.dumps(coco, ensure_ascii=False))
    tmp.replace(out)
    return {"ok": True, "path": str(out)}


def _attr(s):
    return (str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;"))


def _png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _unique(path):
    if not path.exists():
        return path
    for n in range(2, 10000):
        p = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not p.exists():
            return p
    return path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")


# ── 讀 .kra ──
def _entries(z):
    """zip 裡的檔名 → ZipInfo。Krita 用文件名稱當資料夾（中文介面預設是「未命名」），
    寫的是 UTF-8 卻沒標記，Python 會當成舊的 cp437 解讀成亂碼；沒標記的一律換回 UTF-8"""
    out = {}
    for info in z.infolist():
        name = info.filename
        if not info.flag_bits & 0x800:
            try:
                name = name.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        out[name] = info
    return out


def read_kra(raw):
    """.kra 的位元組 → {w, h, layers: [{name, visible, opacity, lock, arr, file}]（下面的在前）, merged, coco, skipped: [名字…]}"""
    import numpy as np
    from PIL import Image
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ValueError(tr("srv.kra.notZip"))
    names = _entries(z)
    if "maindoc.xml" not in names:
        raise ValueError(tr("srv.kra.noMain"))
    root = ET.fromstring(z.read(names["maindoc.xml"]))
    img = root.find(f"{{{NS}}}IMAGE")
    if img is None:
        img = root.find("IMAGE")
    if img is None:
        raise ValueError(tr("srv.kra.noImage"))
    w, h, doc = int(img.get("width")), int(img.get("height")), img.get("name") or ""
    if not (1 <= w <= edit.MAX_SIDE and 1 <= h <= edit.MAX_SIDE):
        raise ValueError(tr("srv.kra.tooBig", w=w, h=h, max=edit.MAX_SIDE))
    out, skipped = [], []

    def walk(layers_el, vis=True, op=1.0, ox=0, oy=0):
        for el in list(layers_el)[::-1]:                    # XML 是上面的在前；倒過來變成由下往上
            tag = el.tag.split("}")[-1]
            if tag != "layer":                              # 遮罩（mask）之類的不是圖層
                continue
            kind, name = el.get("nodetype"), el.get("name") or ""
            v = vis and el.get("visible", "1") != "0"
            o = op * int(el.get("opacity", "255")) / 255
            x, y = ox + int(float(el.get("x", "0"))), oy + int(float(el.get("y", "0")))
            if kind == "grouplayer":
                sub = el.find(f"{{{NS}}}layers")
                if sub is None:
                    sub = el.find("layers")
                if sub is not None:
                    walk(sub, v, o, x, y)
                continue
            fn = el.get("filename") or ""
            path = f"{doc}/layers/{fn}"
            if kind != "paintlayer" or el.get("colorspacename", "RGBA") != "RGBA" or path not in names:
                skipped.append(name)
                continue
            try:
                arr = decode_layer(z.read(names[path]), w, h, x, y)
            except (ValueError, KeyError, IndexError):
                skipped.append(name)
                continue
            flags = el.get("channellockflags") or ""
            out.append({"name": name, "visible": v, "opacity": o, "arr": arr, "file": fn,
                        "lock": len(flags) == 4 and flags[3] == "0"})

    if img.get("colorspacename", "RGBA") == "RGBA":
        top = img.find(f"{{{NS}}}layers")
        if top is None:
            top = img.find("layers")
        if top is not None:
            walk(top)
    merged = None
    if "mergedimage.png" in names:
        with Image.open(io.BytesIO(z.read(names["mergedimage.png"]))) as im:
            merged = im.convert("RGBA")
            merged.load()
    if not out:                                         # 一層都讀不到：用合併後的圖當一層
        if merged is None or merged.size != (w, h):
            raise ValueError(tr("srv.kra.noLayers"))
        out.append({"name": "Krita", "visible": True, "opacity": 1.0, "arr": np.asarray(merged), "file": "", "lock": False})
    try:
        coco = json.loads(z.read(names["coco.json"])) if "coco.json" in names else None
    except ValueError:
        coco = None
    return {"w": w, "h": h, "layers": out, "merged": merged, "coco": coco if isinstance(coco, dict) else None, "skipped": skipped}


def import_kra(con, data):
    """data：{name（檔名）, data（base64 或 data URL）, replace}。
    CoCo 存的 .kra 接回原本那張圖；不是的話合併後的圖先存進圖庫（output/imports），工作檔接在它上面。
    那張圖已經有工作檔、又沒說要換掉：回 {exists, filename} 讓前端問。成功回 {ok, src_id, layers, skipped}"""
    import base64
    import binascii
    import numpy as np
    from PIL import Image
    try:
        import gallery_index as GI
    except ImportError:
        from core import gallery_index as GI
    body = str(data.get("data") or "")
    body = body.partition(",")[2] if body.startswith("data:") else body
    try:
        raw = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        return {"error": tr("srv.kra.unreadable")}
    try:
        k = read_kra(raw)
    except (ValueError, KeyError, ET.ParseError, OSError) as e:
        return {"error": str(e)}
    if len(k["layers"]) > edit.MAX_LAYERS:
        return {"error": tr("srv.kra.tooMany", n=len(k['layers']), max=edit.MAX_LAYERS)}
    coco = k["coco"] or {}
    row = None
    if coco.get("src"):
        row = con.execute("SELECT id, path, filename FROM images WHERE lower(path) = lower(?)", (coco["src"],)).fetchone()
    if row:
        pid = edit.find_project(row["path"])
        if pid and not data.get("replace"):
            return {"exists": True, "filename": row["filename"]}
    else:
        # 不是 CoCo 存的（或原圖不在了）：合併後的圖存進圖庫當底圖
        merged = k["merged"]
        if merged is None or merged.size != (k["w"], k["h"]):
            merged = Image.new("RGBA", (k["w"], k["h"]), (0, 0, 0, 0))
            for L in k["layers"]:
                if L["visible"]:
                    merged = Image.alpha_composite(merged, Image.fromarray(L["arr"], "RGBA"))
        # 同名的 .kra 開過了（圖庫裡已經有 imports/<名字>.png）：先問要不要換成這次的，不要一直多出一張
        stem = Path(str(data.get("name") or "import")).stem or "import"
        dest = Path(C.OUTPUT_DIR) / "imports" / f"{stem}.png"
        if dest.exists() and not data.get("replace"):
            return {"exists": True, "imported": True, "filename": dest.name}
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".png.tmp")
        merged.save(tmp, "PNG")
        tmp.replace(dest)
        GI.index_file(con, dest)
        con.commit()
        row = con.execute("SELECT id, path, filename FROM images WHERE lower(path) = lower(?)", (str(dest),)).fetchone()
        if not row:
            return {"error": tr("srv.kra.saveLibFail")}
        pid = edit.find_project(row["path"])
    by_file = {x.get("file"): x for x in coco.get("layers") or [] if isinstance(x, dict)}
    pid = pid or secrets.token_hex(6)
    d = edit.EDIT_DIR / pid
    with edit._LOCK:
        # 跟存工作檔同一套（server/edit.py）：圖層存成新檔、清單原子換上去之後才刪舊的，
        # 換成這個 .kra 途中出錯時，原本的工作檔完整留著
        d.mkdir(parents=True, exist_ok=True)
        keep, used, written = [], set(), []
        try:
            for L in k["layers"]:
                info = by_file.get(L["file"]) or {}
                lid = str(info.get("id") or "")
                if not edit._ID.match(lid) or lid in used:
                    lid = secrets.token_hex(6)
                used.add(lid)
                im = Image.fromarray(np.ascontiguousarray(L["arr"]), "RGBA")
                name = edit._write_layer(d, lid, lambda f, im=im: im.save(f, "PNG"), written)
                entry = {"id": lid, "file": name, "name": L["name"][:80], "visible": L["visible"],
                         "opacity": round(L["opacity"], 4)}
                # 文字圖層：像素跟存的時候一樣（沒在 Krita 改過）才還原成可以改字的文字圖層
                if info.get("text") and info.get("digest") == _digest(L["arr"]):
                    entry["text"] = info["text"]
                if info.get("wm"):
                    entry["wm"] = True
                if L["lock"] or info.get("lock"):
                    entry["lock"] = True
                keep.append(entry)
            ids = [x["id"] for x in keep]
            meta = {"src": row["path"], "w": k["w"], "h": k["h"], "layers": keep,
                    "active": coco.get("active") if coco.get("active") in ids else ids[-1],
                    "created": time.time(), "updated": time.time()}
            edit._switch(d, meta)
        except BaseException:
            for f in written:
                f.unlink(missing_ok=True)
            raise
        edit._sweep(d, meta)
    return {"ok": True, "src_id": row["id"], "layers": len(keep), "skipped": k["skipped"]}
