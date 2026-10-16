# -*- coding: utf-8 -*-
"""Recipe：可重複使用的組合——模型、LoRA、VAE、編碼器、取樣設定、尺寸、負面提示詞，
以及（存的時候勾了才有）提示詞。

以前提示詞不存（Recipe 只管「怎麼畫」）；使用者筆記：從圖庫存成 Recipe 要連提示詞一起存。
存在 recipes.json；封面是圖庫裡的一張圖，用路徑記錄（跟比較組一樣，索引重建不會跑掉）。
"""

import json
import secrets
import threading
import time
from pathlib import Path

from .library import thumb_ver
from .state import C

RECIPES_FILE = getattr(C, "RECIPES_FILE", None) or (Path(C.DB_PATH).parent / "recipes.json")
_LOCK = threading.Lock()

# Recipe 內容只收這些欄位；型別不對的丟掉，免得壞資料一路傳到生成
_FIELDS = {
    "model": str, "vae": str, "te": str, "arch": str,
    "sampler": str, "schedule": str, "negative": str, "prompt": str,
    "steps": int, "width": int, "height": int, "clip_skip": int,
    "cfg": float, "shift": float,
}


def clean(data):
    out = {}
    for k, typ in _FIELDS.items():
        v = (data or {}).get(k)
        if v is None or v == "":
            continue
        try:
            # 提示詞可能很長（整段模板），其他文字欄位 4000 字就夠
            out[k] = typ(v) if typ is not str else str(v)[:20000 if k in ("prompt", "negative") else 4000]
        except (TypeError, ValueError):
            continue
    loras = []
    for it in (data or {}).get("loras") or []:
        if not isinstance(it, dict) or not str(it.get("name") or "").strip():
            continue
        try:
            w = float(it.get("weight", 1))
        except (TypeError, ValueError):
            w = 1.0
        loras.append({"name": str(it["name"]).strip()[:200], "weight": round(w, 3)})
    out["loras"] = loras[:32]
    return out


def _load():
    try:
        d = json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    tmp = RECIPES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(RECIPES_FILE)


def _stamp(d):
    """最後修改時間一定比現有的都新（Windows 時鐘約 15 ms 才跳一次）"""
    last = max((r.get("updated", 0) for r in d.values()), default=0)
    return max(time.time(), last + 1e-3)


def list_recipes(con):
    out = []
    for rid, r in _load().items():
        cover = None
        cp = r.get("cover")
        if cp:
            row = con.execute("SELECT id, path, mtime FROM images WHERE path = ?", (cp,)).fetchone()
            if row:
                cover = {"id": row[0], "v": thumb_ver(row[1], row[2])}
        out.append({"id": rid, "name": r.get("name", ""), "note": r.get("note", ""),
                    "data": r.get("data") or {}, "cover": cover,
                    "created": r.get("created", 0), "updated": r.get("updated", 0),
                    "used": r.get("used", 0)})
    out.sort(key=lambda x: -x["updated"])
    return out


def save_recipe(con, rid, name, data, cover_id=None, note=None):
    """新增（rid 空）或覆寫。cover_id 是圖庫裡的圖；給了就換封面"""
    cover = None
    if cover_id not in (None, ""):
        row = con.execute("SELECT path FROM images WHERE id = ?", (int(cover_id),)).fetchone()
        cover = row[0] if row else None
    with _LOCK:
        d = _load()
        if not rid or rid not in d:
            rid = secrets.token_hex(4)
            while rid in d:
                rid = secrets.token_hex(4)
            d[rid] = {"created": time.time()}
        r = d[rid]
        if name is not None:
            r["name"] = str(name).strip()[:80]
        if data is not None:
            r["data"] = clean(data)
        if cover:
            r["cover"] = cover
        if note is not None:
            r["note"] = str(note)[:2000]
        r["updated"] = _stamp(d)
        _save(d)
    return rid


def touch(rid):
    """套用一次：記「最近用過」，清單排到最前面"""
    with _LOCK:
        d = _load()
        if rid not in d:
            return False
        d[rid]["used"] = time.time()
        _save(d)
    return True


def delete_recipe(rid):
    with _LOCK:
        d = _load()
        if d.pop(rid, None) is None:
            return False
        _save(d)
    return True


def forget_covers(paths):
    """當封面的圖丟掉或刪掉了：換回沒有封面（清單上顯示 R）"""
    gone = {str(Path(p)).lower() for p in paths or []}
    if not gone:
        return
    with _LOCK:
        d = _load()
        changed = False
        for r in d.values():
            if r.get("cover") and str(Path(r["cover"])).lower() in gone:
                r.pop("cover")
                changed = True
        if changed:
            _save(d)


def rename_paths(mapping):
    """圖庫搬移了檔案：封面路徑跟著改"""
    if not mapping:
        return
    low = {str(Path(a)).lower(): b for a, b in mapping.items()}
    with _LOCK:
        d = _load()
        changed = False
        for r in d.values():
            nb = low.get(str(Path(r.get("cover") or "")).lower()) if r.get("cover") else None
            if nb:
                r["cover"] = nb
                changed = True
        if changed:
            _save(d)
