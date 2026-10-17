# -*- coding: utf-8 -*-
"""探索（工作室的 02）：找相似的圖、用文字找圖。

看圖的模型是 SigLIP2 base（Google，Apache-2.0）：把每張圖算成一個特徵向量，相似的圖向量方向接近；
文字也算成同一個空間的向量，所以可以用文字（含中文）找圖。模型不內建，按了才下載（約 1.5 GB；中斷了下次接著下載，download.py）。

搜尋的範圍：CoCo 的圖庫，加上「參考圖來源」——一般資料夾，或 Eagle 的檔案庫（只讀，不寫入）。
特徵存在 explore.db（跟 gallery.db 同一個資料夾），換了檔案（修改時間不同）才重算。
算特徵交給工作行程（顯示卡），這裡負責排程、存取與搜尋。
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from . import download
from .state import C

from i18n import tr

MODEL_ID = "siglip2-base-patch16-224"
MODEL_REV = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"        # 固定版本：以後上游更新也拿到一樣的東西
MODEL_PAGE = "https://huggingface.co/google/siglip2-base-patch16-224"
_BASE = f"https://huggingface.co/google/siglip2-base-patch16-224/resolve/{MODEL_REV}/"
# 要下載的檔案：大小、SHA256（大檔有，小的設定檔沒有就只看大小）
MODEL_FILES = {
    "config.json": (253, None),
    "preprocessor_config.json": (394, None),
    "special_tokens_map.json": (636, None),
    "tokenizer_config.json": (47164, None),
    "tokenizer.json": (34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
    "model.safetensors": (1500800904, "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"),
}
MODEL_BYTES = sum(b for b, _ in MODEL_FILES.values())
_DL = download.Jobs()          # 只有一個下載工作："model"
_LOCK = threading.Lock()


def model_dir():
    from . import modelhome
    return modelhome.own("explore") / MODEL_ID


def model_ready():
    d = model_dir()
    return all((d / f).is_file() and (d / f).stat().st_size == b for f, (b, _) in MODEL_FILES.items())


def model_status():
    """part：已經下載了多少（關掉 CoCo 再打開時說上次下載到哪裡）"""
    dl = _DL.progress("model")
    ready = model_ready()
    return {"ready": ready, "bytes": MODEL_BYTES, "page": MODEL_PAGE, "dir": str(model_dir()),
            "part": 0 if ready else download.on_disk(_files()),
            "download": dl if dl and (dl["running"] or dl["error"]) else None}


def _files():
    d = model_dir()
    return [{"url": _BASE + name, "bytes": size, "sha256": sha, "dest": d / name} for name, (size, sha) in MODEL_FILES.items()]


def _download(opener=None):
    download.run(_DL["model"], _files(), opener)


def start_download():
    if model_ready():
        return {"ok": True, "already": True}
    return _DL.begin("model", MODEL_BYTES, _download)


def cancel_download():
    return _DL.cancel("model")


# ---------------------------------------------------------------- 參考圖來源
# 設定的 REF_SOURCES：[{"kind": "folder", "path": …} 或 {"kind": "eagle", "path": "…\\X.library", "folders": [Eagle 資料夾 id…]}]
# Eagle 選了資料夾就連底下的子資料夾一起；folders 空的＝整個檔案庫。Eagle 的檔案庫只讀，不寫入任何東西
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def clean_sources(v):
    try:
        from core.settings import as_refsources
    except ImportError:
        from settings import as_refsources
    return as_refsources(v)


def eagle_info(path):
    """Eagle 檔案庫的資料夾樹：{name, path, folders: [{id, name, children}]} 或 {error}"""
    p = Path(str(path or "").strip().strip('"'))
    meta = p / "metadata.json"
    if not (p.is_dir() and meta.is_file() and (p / "images").is_dir()):
        return {"error": tr("srv.x.notEagle")}
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": tr("srv.x.eagleErr", e=e)}

    def tree(fs):
        return [{"id": str(f.get("id")), "name": str(f.get("name") or ""), "children": tree(f.get("children") or [])}
                for f in fs or [] if isinstance(f, dict) and f.get("id")]
    return {"name": p.name.rsplit(".library", 1)[0], "path": str(p), "folders": tree(m.get("folders"))}


def _descendants(folders, chosen):
    """選了的資料夾，加上它們底下所有的子資料夾"""
    out, want = set(), set(chosen)

    def walk(fs, inside):
        for f in fs:
            on = inside or f["id"] in want
            if on:
                out.add(f["id"])
            walk(f["children"], on)
    walk(folders, False)
    return out


def _eagle_items(src, known):
    """Eagle 檔案庫裡選到的圖：[{key, path, thumb, name, tags, url, eagle, folders, w, h, mtime}]。
    known：上次讀過的 {eagle id: (mtime, item)}，修改時間（mtime.json）沒變就不重讀 metadata.json"""
    info = eagle_info(src["path"])
    if "error" in info:
        return [], info["error"]
    lib = Path(info["path"])
    allowed = _descendants(info["folders"], src.get("folders") or []) if src.get("folders") else None
    try:
        mt = json.loads((lib / "mtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        mt = {}
    out, nread = [], 0
    for d in (lib / "images").iterdir():
        if not d.name.endswith(".info"):
            continue
        eid = d.name[:-5]
        stamp = mt.get(eid)
        old = known.get(eid)
        if old and stamp is not None and old[0] == stamp:
            it = old[1]
        else:
            try:
                m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            nread += 1
            ext = "." + str(m.get("ext") or "").lower()
            if m.get("isDeleted") or ext not in IMAGE_EXT:
                it = None
            else:
                name = str(m.get("name") or eid)
                f = d / f"{name}{ext}"
                th = d / f"{name}_thumbnail.png"
                it = {"key": f"eagle:{eid}", "path": str(f), "thumb": str(th) if th.is_file() else "",
                      "name": name, "tags": [str(t) for t in m.get("tags") or []][:50], "url": str(m.get("url") or ""),
                      "eagle": eid, "folders": [str(x) for x in m.get("folders") or []],
                      "w": int(m.get("width") or 0), "h": int(m.get("height") or 0), "mtime": float(stamp or m.get("mtime") or 0)}
            known[eid] = (stamp, it)
        if it and (allowed is None or allowed.intersection(it["folders"])):
            out.append(it)
    known["_read"] = nread
    return out, ""


def _folder_items(src):
    p = Path(src["path"])
    if not p.is_dir():
        return [], tr("srv.x.noDir", p=p)
    out = []
    for f in p.rglob("*"):
        if f.suffix.lower() in IMAGE_EXT and f.is_file():
            try:
                st = f.stat()
            except OSError:
                continue
            out.append({"key": f"file:{str(f).lower()}", "path": str(f), "thumb": "", "name": f.stem, "tags": [], "url": "",
                        "eagle": "", "folders": [], "w": 0, "h": 0, "mtime": st.st_mtime})
    return out, ""


# ---------------------------------------------------------------- 特徵資料庫
# explore.db：vecs（每張圖的特徵，key＋算的時候的修改時間）、refs（參考圖的資料：路徑、Eagle 的標籤與來源網址…）。
# key：圖庫的圖是 lib:路徑（圖庫重建索引時 id 會變，路徑不會），參考圖是 eagle:Eagle 的 id 或 file:路徑
DB_FILE = Path(C.DB_PATH).parent / "explore.db"
INDEX_BATCH = 64
DIM = 768
_EAGLE_CACHE = {}          # 檔案庫路徑 → {eagle id: (mtime, item)}
_IDX = {"running": False, "phase": "", "done": 0, "total": 0, "error": "", "warn": [], "finished": 0.0}
_MAT = {"ver": -1, "keys": [], "mat": None}
_VER = [0]


def db():
    con = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS vecs (key TEXT PRIMARY KEY, mtime REAL, vec BLOB);
        CREATE TABLE IF NOT EXISTS refs (id INTEGER PRIMARY KEY, key TEXT UNIQUE, src TEXT, path TEXT, thumb TEXT,
                                         name TEXT, tags TEXT, url TEXT, eagle TEXT, w INT, h INT, mtime REAL);
        -- Eagle 每一項讀過的結果（不是圖、在垃圾桶的記 null）：重開 CoCo 也只要讀改過的（第一次讀 27,000 項要兩分鐘）
        CREATE TABLE IF NOT EXISTS eagle (lib TEXT, eid TEXT, stamp INTEGER, item TEXT, PRIMARY KEY (lib, eid));
        -- 存回 Eagle 存過的（server/eagle.py）：同一張再存先提醒
        CREATE TABLE IF NOT EXISTS eagle_saved (path TEXT, lib TEXT, folder TEXT, time REAL);
    """)
    # 參考圖加進圖庫後的那份（精修、編輯用；同一張只複製一次）。舊的資料庫沒有這欄就補上
    if "lib_path" not in {r[1] for r in con.execute("PRAGMA table_info(refs)")}:
        con.execute("ALTER TABLE refs ADD COLUMN lib_path TEXT")
    return con


def _sync_refs(con, sources):
    """參考圖來源 → refs 表（新的加、改的更新、拿掉的連特徵一起刪）。回傳 (參考圖數, 警告)"""
    items, warns = [], []
    for s in sources:
        if s["kind"] == "eagle":
            known = _eagle_known(con, s["path"])
            got, err = _eagle_items(s, known)
            if known.pop("_read", 0):
                con.execute("DELETE FROM eagle WHERE lib = ?", (s["path"],))
                con.executemany("INSERT INTO eagle(lib, eid, stamp, item) VALUES (?,?,?,?)",
                                [(s["path"], eid, st, json.dumps(it, ensure_ascii=False) if it else None)
                                 for eid, (st, it) in known.items()])
        else:
            got, err = _folder_items(s)
        if err:
            warns.append(err)
        for it in got:
            it["src"] = s["path"]
        items += got
    seen = {it["key"] for it in items}
    gone = {r["key"] for r in con.execute("SELECT key FROM refs")} - seen
    con.executemany("DELETE FROM refs WHERE key = ?", [(k,) for k in gone])
    con.executemany("DELETE FROM vecs WHERE key = ?", [(k,) for k in gone])
    con.executemany("""INSERT INTO refs(key, src, path, thumb, name, tags, url, eagle, w, h, mtime)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(key) DO UPDATE SET src=excluded.src, path=excluded.path, thumb=excluded.thumb,
                         name=excluded.name, tags=excluded.tags, url=excluded.url, w=excluded.w, h=excluded.h,
                         mtime=excluded.mtime""",
                    [(it["key"], it["src"], it["path"], it["thumb"], it["name"], json.dumps(it["tags"], ensure_ascii=False),
                      it["url"], it["eagle"], it["w"], it["h"], it["mtime"]) for it in items])
    con.commit()
    return len(items), warns


def _eagle_known(con, lib):
    """Eagle 讀過的結果：記憶體裡有就用，沒有就從 explore.db 讀回來"""
    known = _EAGLE_CACHE.get(lib)
    if known is None:
        known = {r["eid"]: (r["stamp"], json.loads(r["item"]) if r["item"] else None)
                 for r in con.execute("SELECT eid, stamp, item FROM eagle WHERE lib = ?", (lib,))}
        _EAGLE_CACHE[lib] = known
    return known


def lib_key(path):
    return f"lib:{str(path).lower()}"


def _lib_items(gcon):
    """CoCo 圖庫的圖。算特徵用縮圖（快，224 也夠）"""
    from .library import thumb_path
    out = []
    for r in gcon.execute("SELECT path, mtime FROM images"):
        th = thumb_path({"path": r["path"], "mtime": r["mtime"]})
        out.append({"key": lib_key(r["path"]), "path": r["path"], "thumb": str(th) if th.exists() else "",
                    "mtime": float(r["mtime"] or 0)})
    return out


def index_status():
    con = db()
    try:
        n = {"vecs": con.execute("SELECT COUNT(*) FROM vecs").fetchone()[0],
             "refs": con.execute("SELECT COUNT(*) FROM refs").fetchone()[0]}
    finally:
        con.close()
    return {**_IDX, "warn": list(_IDX["warn"]), **n}


def start_index(gallery_db, sources, embed):
    """背景更新索引：同步參考圖、找出沒算過（或檔案改過）的、一批一批交給工作行程算。
    embed(paths) → (向量陣列, 成功的索引) 或 {error}"""
    with _LOCK:
        if _IDX["running"]:
            return {"ok": True}
        if not model_ready():
            return {"error": tr("srv.noExploreModel")}
        _IDX.update(running=True, phase="scan", done=0, total=0, error="", warn=[])
    threading.Thread(target=_index, args=(gallery_db, sources, embed), daemon=True).start()
    return {"ok": True}


def _index(gallery_db, sources, embed):
    con = db()
    try:
        _, warns = _sync_refs(con, sources)
        _IDX["warn"] = warns
        gcon = sqlite3.connect(gallery_db, timeout=30)
        gcon.row_factory = sqlite3.Row
        try:
            items = _lib_items(gcon)
        finally:
            gcon.close()
        items += [dict(r) for r in con.execute("SELECT key, path, thumb, mtime FROM refs")]
        have = {r["key"]: r["mtime"] for r in con.execute("SELECT key, mtime FROM vecs")}
        lib_keys = {it["key"] for it in items if it["key"].startswith("lib:")}
        stale = [k for k in have if k.startswith("lib:") and k not in lib_keys]       # 圖庫裡已經沒有的
        con.executemany("DELETE FROM vecs WHERE key = ?", [(k,) for k in stale])
        con.commit()
        todo = [it for it in items if have.get(it["key"]) != it["mtime"]]
        _IDX.update(phase="embed", total=len(todo), done=0)
        for s in range(0, len(todo), INDEX_BATCH):
            part = todo[s:s + INDEX_BATCH]
            got = embed([it["thumb"] or it["path"] for it in part])
            if isinstance(got, dict):
                raise RuntimeError(got.get("error") or tr("srv.embedFail"))
            vecs, ok = got
            con.executemany("INSERT OR REPLACE INTO vecs(key, mtime, vec) VALUES (?,?,?)",
                            [(part[i]["key"], part[i]["mtime"], np.asarray(vecs[j], np.float16).tobytes())
                             for j, i in enumerate(ok)])
            con.commit()
            _IDX["done"] = min(len(todo), s + len(part))
            _VER[0] += 1
        _VER[0] += 1
    except Exception as e:
        _IDX["error"] = f"{type(e).__name__}: {e}"
    finally:
        con.close()
        _IDX.update(running=False, phase="", finished=time.time())


def _matrix():
    """全部的向量（快取；索引有變才重讀）：(keys, N×768 float32)"""
    if _MAT["ver"] != _VER[0] or _MAT["mat"] is None:
        con = db()
        try:
            rows = con.execute("SELECT key, vec FROM vecs").fetchall()
        finally:
            con.close()
        keys = [r["key"] for r in rows]
        mat = (np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float16).reshape(-1, DIM).astype(np.float32)
               if rows else np.zeros((0, DIM), np.float32))
        _MAT.update(ver=_VER[0], keys=keys, mat=mat, pos={k: i for i, k in enumerate(keys)})
    return _MAT["keys"], _MAT["mat"]


def vector_of(key):
    _matrix()
    i = _MAT["pos"].get(key)
    return None if i is None else _MAT["mat"][i]


def search(vec, scope="all", limit=200, exclude=()):
    """跟 vec 最像的：[(key, 相似度)]。scope：all／lib（我的圖）／ref（參考圖）"""
    keys, mat = _matrix()
    if not keys:
        return []
    sim = mat @ np.asarray(vec, np.float32)
    if scope in ("lib", "ref"):
        mask = np.array([k.startswith("lib:") == (scope == "lib") for k in keys])
        sim = np.where(mask, sim, -9.0)
    for k in exclude or ():
        i = _MAT["pos"].get(k)
        if i is not None:
            sim[i] = -9.0
    top = np.argsort(-sim)[:limit]
    return [(keys[i], float(sim[i])) for i in top if sim[i] > -9]


def _gi():
    try:
        from core import gallery_index as GI
    except ImportError:
        import gallery_index as GI
    return GI


def _num(v, kind=float):
    try:
        x = kind(float(v)) if kind is int else float(v)
    except (TypeError, ValueError):
        return None
    return x


def ref_detail(rid):
    """參考圖的詳細資料：跟圖庫的 /api/detail 同一種格式（看大圖直接用），加上 ref（Eagle 的標籤、來源網址…）。
    圖裡有生成參數（例如存進 Eagle 的 AI 圖）就一起讀出來，送到生成區、存成 Recipe 都能用"""
    con = db()
    try:
        r = con.execute("SELECT * FROM refs WHERE id = ?", (rid,)).fetchone()
    finally:
        con.close()
    if not r:
        return {"error": tr("srv.refNotFound")}
    f = Path(r["path"])
    meta = _gi().read_meta(f) if f.is_file() else {}
    if "error" in meta:
        meta = {}
    w, h = _num(meta.get("width"), int) or r["w"], _num(meta.get("height"), int) or r["h"]
    if not (w and h) and f.is_file():
        from PIL import Image
        with Image.open(f) as im:
            w, h = im.size
    loras = meta.get("loras") or []
    try:
        st = f.stat()
        size, mtime = st.st_size, st.st_mtime
    except OSError:
        size, mtime = 0, r["mtime"]
    return {"id": f"r{r['id']}", "filename": f.name, "path": str(f), "mtime": mtime, "filesize": size,
            "width": w, "height": h, "rating": None, "exists": f.is_file(),
            "prompt": meta.get("prompt") or "", "negative": meta.get("negative") or "",
            "model": _gi().normalize_model(meta.get("model")) if meta.get("model") else "",
            "sampler": meta.get("sampler") or "", "scheduler": meta.get("scheduler") or "",
            "steps": _num(meta.get("steps"), int), "cfg": _num(meta.get("cfg")), "seed": _num(meta.get("seed"), int),
            "clip_skip": _num(meta.get("clip_skip"), int),
            "loras": loras if isinstance(loras, list) else [], "extra": meta.get("extra") or {},
            "ref": {"rid": r["id"], "key": r["key"], "name": r["name"], "tags": json.loads(r["tags"] or "[]"),
                    "url": r["url"], "eagle": r["eagle"], "src": r["src"]}}


def import_ref(rid, gcon, out_dir, lock):
    """參考圖加進 CoCo 的圖庫（精修、編輯、放進比較要用）：複製一份到 outputs/imports，同一張只複製一次。
    原檔（Eagle 的檔案庫）不動。回傳 {id: 圖庫的 id} 或 {error}"""
    import shutil
    con = db()
    try:
        r = con.execute("SELECT * FROM refs WHERE id = ?", (rid,)).fetchone()
        if not r:
            return {"error": tr("srv.refNotFound")}
        src = Path(r["path"])
        old = r["lib_path"] if "lib_path" in r.keys() else None
        if old and Path(old).is_file():
            hit = gcon.execute("SELECT id FROM images WHERE lower(path) = lower(?)", (old,)).fetchone()
            if hit:
                return {"id": hit[0], "path": old}
        if not src.is_file():
            return {"error": tr("srv.x.refFileGone")}
        dest_dir = Path(out_dir) / "imports"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        n = 2
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        shutil.copy2(src, dest)
        with lock:
            _gi().index_file(gcon, dest)
            gcon.commit()
        con.execute("UPDATE refs SET lib_path = ? WHERE id = ?", (str(dest), rid))
        con.commit()
        hit = gcon.execute("SELECT id FROM images WHERE lower(path) = lower(?)", (str(dest),)).fetchone()
        return {"id": hit[0], "path": str(dest)} if hit else {"error": tr("srv.x.importFail")}
    finally:
        con.close()


def ref_rows(con, keys):
    """參考圖的資料（給前端顯示）：{key: {...}}"""
    if not keys:
        return {}
    out = {}
    for s in range(0, len(keys), 500):
        part = keys[s:s + 500]
        q = f"SELECT * FROM refs WHERE key IN ({','.join('?' * len(part))})"
        for r in con.execute(q, part):
            d = dict(r)
            d["tags"] = json.loads(d["tags"] or "[]")
            out[d["key"]] = d
    return out
