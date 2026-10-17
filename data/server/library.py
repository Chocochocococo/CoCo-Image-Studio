# -*- coding: utf-8 -*-
"""圖庫索引、查詢與縮圖。
"""

import hashlib
import importlib
import io
import math
import time
from pathlib import Path

from .state import (C, DB_PATH, DB_WRITE_LOCK, INDEX_LOCK, INDEX_STATE,
                    SETTINGS_APPLY_LOCK, THUMB_DIR, THUMB_W, res_stem)

from i18n import tr


def run_index(quiet=False):
    """呼叫 gallery_index 做一次增量掃描"""
    import gallery_index as GI

    if not INDEX_LOCK.acquire(blocking=False):
        return {"error": tr("srv.lib.indexing")}
    INDEX_STATE.update(running=True, msg=tr("srv.lib.scanning"))
    try:
        with SETTINGS_APPLY_LOCK:
            importlib.reload(C)      # 設定可能剛剛才改過
            scan_dirs = list(C.SCAN_DIRS)
        GI.SCAN_DIRS = scan_dirs     # 這行不補的話會一直掃舊資料夾
        GI.DB_PATH = DB_PATH
        GI.REBUILD = False
        with DB_WRITE_LOCK:
            con = GI.open_db()
            try:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA busy_timeout=30000")
                if quiet:
                    import io as _io
                    import contextlib
                    buf = _io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        st = GI.scan(con, GI.SCAN_DIRS)
                else:
                    st = GI.scan(con, GI.SCAN_DIRS)
            finally:
                con.close()
        msg = tr("srv.lib.done", added=st['added'], updated=st['updated'], skipped=st['skipped'], removed=st['removed'])
        INDEX_STATE.update(msg=msg, at=time.time())
        return {"ok": True, **st, "msg": msg}
    except Exception as e:
        INDEX_STATE["msg"] = tr("srv.app.indexErr", e=e)
        return {"error": str(e)}
    finally:
        INDEX_STATE["running"] = False
        INDEX_LOCK.release()


# ---------------------------------------------------------------- 查詢
def _vals(p, key):
    """篩選值可以是一個字串或一串（網址重複帶同一個參數），統一成清單"""
    v = p.get(key)
    items = v if isinstance(v, (list, tuple)) else [v]
    return [str(x) for x in items if x not in (None, "")]


_LORA_IN = ("EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(loras) "
            "THEN loras ELSE '[]' END) j "
            "WHERE coco_stem(json_extract(j.value, '$.file')) IN ({}))")


def build_query(p):
    """依篩選條件組出 SQL

    q：空格或逗號分開的詞都要出現；前面加 - 的詞要不出現。
    model / folder / root：可以多個，符合任一個就算（一張圖只會有一個模型）。
    lora：可以多個；lora_mode=any 是任一個，其他（預設）是全部都要有。
    """
    where, args = [], []

    include, exclude = fts_terms(p.get("q"))
    if include:
        where.append(
            "id IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)")
        args.append(" ".join(include))
    for term in exclude:
        where.append(
            "id NOT IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)")
        args.append(term)

    # 模型與 LoRA 用 coco_stem 比對（連線要先註冊，見 state.db）：
    # 短名、完整路徑、有沒有副檔名都算同一個
    models = [res_stem(m) for m in _vals(p, "model")]
    if models:
        where.append(f"coco_stem(model) IN ({','.join('?' * len(models))})")
        args += models
    loras = [res_stem(x) for x in _vals(p, "lora")]
    if loras and p.get("lora_mode") == "any":
        where.append(_LORA_IN.format(",".join("?" * len(loras))))
        args += loras
    else:
        for x in loras:
            where.append(_LORA_IN.format("?"))
            args.append(x)
    # 來源：掃描的根目錄（CoCo 的 outputs、Forge 的 Text2Img…）底下的所有圖
    roots = [str(r).rstrip("\\/").lower() for r in _vals(p, "root")]
    if roots:
        where.append("(" + " OR ".join(
            "lower(folder) = ? OR substr(lower(folder), 1, ?) IN (?, ?)" for _ in roots) + ")")
        for root in roots:
            args += [root, len(root) + 1, root + "\\", root + "/"]
    if p.get("sampler"):
        where.append("sampler = ?")
        args.append(p["sampler"])
    folders = _vals(p, "folder")
    if folders:
        where.append(f"folder IN ({','.join('?' * len(folders))})")
        args += folders

    # 評分：勾哪幾個星級就是哪幾個（可以複選），0＝未評分。舊網址的 -1 也是未評分
    stars = sorted({0 if v == "-1" else int(v) for v in _vals(p, "rating")
                    if v.lstrip("-").isdigit() and -1 <= int(v) <= 5})
    if stars:
        where.append(f"COALESCE(rating, 0) IN ({','.join('?' * len(stars))})")
        args += stars

    if p.get("orient") == "portrait":
        where.append("height > width")
    elif p.get("orient") == "landscape":
        where.append("width > height")
    elif p.get("orient") == "square":
        where.append("width = height")

    if p.get("noparam") == "1":
        where.append("prompt = ''")

    # 勝出：任何一個比較組裡標了勝出的圖（比較組用路徑記，這裡照路徑比對，不分大小寫）
    if p.get("win") == "1":
        from .compare import winner_paths
        wins = winner_paths()
        if wins:
            where.append(f"lower(path) IN ({','.join('?' * len(wins))})")
            args += wins
        else:
            where.append("0")

    sort = {
        "new": "mtime DESC", "old": "mtime ASC",
        "rating": "rating DESC, mtime DESC",
        "name": "filename ASC",
    }.get(p.get("sort") or "new", "mtime DESC")

    # 生成頁的牆開著篩選時（keep_new＝這次改篩選的時間，秒）：之後才生成的圖不管符不符合都列出——使用者選的（2026-10-10）：
    # 生成時一定看得到結果，例如篩「4 顆星以上」時剛生成、還沒評分的圖不會不見。照時間排，新圖自然在最前面。
    # 從改篩選的時間算、不是從開 CoCo 算（使用者回報：取消再重新篩選，之前的新圖要當成舊圖照篩選）
    since = _since(p.get("keep_new"))
    if where and since:
        return " WHERE ((" + ") AND (".join(where) + ") OR mtime >= ?)", args + [since], sort

    sql = " WHERE " + " AND ".join(where) if where else ""
    return sql, args, sort


def _since(v):
    """keep_new 的時間（秒）；不是正的數字就當沒帶"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def lib_facets(con, roots):
    """新介面圖庫的篩選清單：來源、模型、LoRA、資料夾，各附張數

    模型與 LoRA 依 res_stem 合併（同一個模型有的圖寫短名、有的寫完整路徑），
    顯示名稱取最常出現的那個寫法、去掉路徑與副檔名。
    """
    import collections
    import json
    models, loras = collections.Counter(), collections.Counter()
    shown = collections.defaultdict(collections.Counter)
    folders = collections.Counter()
    root_n = collections.Counter()
    norm = [(r, str(r).rstrip("\\/").lower()) for r in roots]
    total = 0
    for model, lora_json, folder in con.execute(
            "SELECT model, loras, folder FROM images"):
        total += 1
        if model:
            k = res_stem(model)
            models[k] += 1
            shown[("m", k)][str(model).replace("\\", "/").rsplit("/", 1)[-1]] += 1
        try:
            items = json.loads(lora_json or "[]")
        except (TypeError, ValueError):
            items = []
        seen = set()
        for it in items if isinstance(items, list) else []:
            f = it.get("file") if isinstance(it, dict) else None
            k = res_stem(f)
            if k and k not in seen:          # 同一張圖重複列同一個 LoRA 只算一次
                seen.add(k)
                loras[k] += 1
                shown[("l", k)][str(f).replace("\\", "/").rsplit("/", 1)[-1]] += 1
        if folder:
            folders[folder] += 1
            low = folder.lower()
            for r, rl in norm:
                if low == rl or low.startswith(rl + "\\") or low.startswith(rl + "/"):
                    root_n[r] += 1
                    break

    def named(counter, kind):
        out = []
        for k, c in counter.most_common():
            name = shown[(kind, k)].most_common(1)[0][0]
            for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"):
                if name.lower().endswith(ext):
                    name = name[: -len(ext)]
            out.append({"v": k, "name": name, "c": c})
        return out

    from .compare import winner_paths
    wins = winner_paths()
    won = 0
    for i in range(0, len(wins), 500):
        part = wins[i:i + 500]
        won += con.execute(f"SELECT COUNT(*) FROM images WHERE lower(path) IN ({','.join('?' * len(part))})", part).fetchone()[0]

    return {"total": total, "winners": won,
            "roots": [{"v": r, "c": root_n[r]} for r in roots],
            "models": named(models, "m"), "loras": named(loras, "l"),
            "folders": [{"v": f, "c": c} for f, c in folders.most_common()]}


def fts_terms(q):
    """把使用者輸入拆成 FTS 片語：(要出現的, 要排除的)

    每個詞都包成片語，避免引號、括號、星號被當成語法而報錯。
    詞尾的 * 保留為前綴搜尋；詞前面的 - 代表排除。
    """
    import re as _re
    include, exclude = [], []
    for t in _re.split(r"[\s,]+", (q or "").strip()):
        neg = len(t) > 1 and t.startswith("-")
        if neg:
            t = t[1:]
        star = t.endswith("*")
        t = t.strip('"*()').replace('"', "")
        if not t:
            continue
        (exclude if neg else include).append(f'"{t}"' + ("*" if star else ""))
    return include, exclude


# ---------------------------------------------------------------- 縮圖
# 縮圖格式改變時就加一，舊快取會自動作廢（檔名不同）。
#   2：有透明度的圖改存 WebP。之前一律轉 RGB 存 JPEG，透明處底下殘留的
#      像素顏色會露出來——實測去背圖的透明區底下是紫色 (125, 93, 152)。
#   3：全部改 WebP，只留一種格式。實測 60 張平均 35.5 KB（JPEG 45.5 KB），
#      編碼每張約 26 ms（JPEG 1.4 ms），縮圖是看到才做，所以感覺不出來。
THUMB_VERSION = 3


def thumb_ver(path, mtime):
    """縮圖網址的版本碼：取自路徑與檔案時間，同一秒建立的檔案也不會撞號。
    索引重建後 id 會重新分配，靠這個才不會拿到別張圖的快取。
    也算進縮圖寬度：設定裡改了寬度（重開後生效），網址跟著變，瀏覽器才不會繼續用舊縮圖。"""
    return hashlib.sha1(f"{path}|{mtime}|{THUMB_W}".encode("utf-8")).hexdigest()[:10]


def thumb_path(row):
    key = f"{row['path']}|{row['mtime']}|{THUMB_W}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return THUMB_DIR / f"v{THUMB_VERSION}-{h}.webp"


def _has_transparency(im):
    """真的有透明的地方才算；整張 alpha 都是 255 的 RGBA 就是不透明的圖。

    很多生圖工具存 alpha 全滿的 RGBA，那種縮圖存 RGB，不必多帶一個 alpha。
    """
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        return im.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return False


def make_thumb(row):
    from PIL import Image
    out = thumb_path(row)
    if out.exists():
        return out
    THUMB_DIR.mkdir(exist_ok=True)
    src = Path(row["path"])
    if not src.exists():
        return None
    with Image.open(src) as im:
        alpha = _has_transparency(im)
        im = im.convert("RGBA" if alpha else "RGB")
        w, h = im.size
        if w > THUMB_W:
            im = im.resize((THUMB_W, max(1, round(h * THUMB_W / w))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=82, method=4)
    out.write_bytes(buf.getvalue())
    return out


# ---------------------------------------------------------------- 在資料夾中顯示
def open_folder(path):
    """打開資料夾本身（設定畫面的「打開」：CoCo 的模型資料夾）"""
    import os
    import subprocess
    if os.name == "nt":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


def reveal(path):
    """打開檔案所在的資料夾並選取它。

    Windows 用殼層的 SHOpenFolderAndSelectItems：資料夾已經開著的話沿用那個視窗、
    帶到前面再選取；直接叫 explorer /select 每次都開新視窗，已開著的那個不會跟著選。
    殼層呼叫失敗才退回 explorer /select。其他系統只能打開資料夾。
    """
    import os
    import subprocess
    f = Path(path)
    if os.name != "nt":
        subprocess.Popen(["xdg-open", str(f.parent)])
        return "folder"
    try:
        import ctypes
        from ctypes import wintypes
        shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
        shell32.ILCreateFromPathW.restype = ctypes.c_void_p
        shell32.ILCreateFromPathW.argtypes = [wintypes.LPCWSTR]
        shell32.ILFree.argtypes = [ctypes.c_void_p]
        shell32.SHOpenFolderAndSelectItems.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p, wintypes.DWORD]
        shell32.SHOpenFolderAndSelectItems.restype = ctypes.c_long
        ole32.CoInitialize(None)
        try:
            pidl = shell32.ILCreateFromPathW(str(f))
            if not pidl:
                raise OSError("ILCreateFromPathW")
            try:
                hr = shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0)
            finally:
                shell32.ILFree(pidl)
            if hr != 0:
                raise OSError(f"SHOpenFolderAndSelectItems {hr:#x}")
        finally:
            ole32.CoUninitialize()
        return "shell"
    except (OSError, AttributeError):
        subprocess.Popen(f'explorer /select,"{f}"')
        return "explorer"
