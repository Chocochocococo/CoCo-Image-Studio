# -*- coding: utf-8 -*-
"""比較組：一組拿來互相比較的圖，存在 compare_sets.json。

用檔案路徑記錄，不用資料庫的 id——索引重建後 id 會重新分配，路徑不會。
圖庫「搬移」時由 handler.file_op 呼叫 rename_paths() 跟著改。
"""

import json
import secrets
import threading
import time
from pathlib import Path

from .library import thumb_ver
from .state import C

SETS_FILE = getattr(C, "COMPARE_FILE", None) or (Path(C.DB_PATH).parent / "compare_sets.json")
_LOCK = threading.Lock()

# 比較時要看的欄位（images 表），前端拿來算差異
FIELDS = ("id", "path", "mtime", "filename", "width", "height", "prompt", "negative", "steps", "cfg",
          "seed", "sampler", "scheduler", "model", "clip_skip", "loras", "extra", "rating")


def _load():
    try:
        d = json.loads(SETS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    tmp = SETS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(SETS_FILE)


def _stamp(d):
    """「最後修改」時間：一定比現有的都新。Windows 的時鐘約 15 ms 才跳一次，
    連續兩次操作可能拿到同一個值，清單順序就會亂"""
    last = max((s.get("updated", 0) for s in d.values()), default=0)
    return max(time.time(), last + 1e-3)


def _norm(p):
    return str(Path(p)).lower()


def _auto_archive(d, days, now):
    """自動建的組（生成後送去比較）超過 days 天沒打開就封存。回傳有沒有改"""
    if not days or days <= 0:
        return False
    changed = False
    for s in d.values():
        if s.get("auto") and not s.get("archived") and now - (s.get("seen") or s.get("created") or now) > days * 86400:
            s["archived"] = True
            changed = True
    return changed


def list_sets(con, auto_archive_days=0):
    """清單：名稱、張數、前四張的縮圖、狀態（比較中／已完成／封存），新的在前。
    auto_archive_days：自動建的組幾天沒打開就封存（0＝不封存，介面偏好 cmpautoarch）"""
    out = []
    with _LOCK:
        d = _load()
        if _auto_archive(d, auto_archive_days, time.time()):
            _save(d)
    for sid, s in d.items():
        items = s.get("items") or []
        covers = []
        for it in items[:4]:
            r = con.execute("SELECT id, path, mtime FROM images WHERE path = ?", (it["path"],)).fetchone()
            if r:
                covers.append({"id": r[0], "v": thumb_ver(r[1], r[2])})
        out.append({"id": sid, "name": s.get("name", ""), "count": len(items),
                    "winners": sum(1 for it in items if it.get("winner")),
                    "updated": s.get("updated", 0), "covers": covers,
                    "done": bool(s.get("done")), "archived": bool(s.get("archived")), "auto": bool(s.get("auto"))})
    out.sort(key=lambda x: -x["updated"])
    return out


def get_set(con, sid):
    """一組的內容，每張附上目前資料庫裡的參數；檔案不在索引裡的標 missing"""
    with _LOCK:
        d = _load()
        s = d.get(sid)
        if not s:
            return None
        # 最後打開的時間（自動封存看這個）；不動 updated，清單的順序才不會因為點開就跳
        s["seen"] = time.time()
        _save(d)
    items = []
    for it in s.get("items") or []:
        r = con.execute(f"SELECT {', '.join(FIELDS)} FROM images WHERE path = ?", (it["path"],)).fetchone()
        row = dict(zip(FIELDS, r)) if r else None
        if row:
            row["v"] = thumb_ver(row.pop("path"), row.pop("mtime"))
            for k in ("loras", "extra"):
                try:
                    row[k] = json.loads(row[k] or ("[]" if k == "loras" else "{}"))
                except ValueError:
                    row[k] = [] if k == "loras" else {}
        items.append({"path": it["path"], "winner": bool(it.get("winner")), "note": it.get("note", ""),
                      "reach": int(it.get("reach") or 1), "adv": bool(it.get("adv")),
                      "img": row, "missing": row is None})
    return {"id": sid, "name": s.get("name", ""), "created": s.get("created", 0),
            "updated": s.get("updated", 0), "items": items,
            # 淘汰賽（web/js/cup.js）：目前第幾輪、選好了沒（＝已完成）；封存的組清單上另外一頁
            "round": int(s.get("round") or 1), "done": bool(s.get("done")), "archived": bool(s.get("archived"))}


def add_images(con, sid, ids, name="", paths=None, auto=""):
    """把圖庫裡的圖加進比較組；sid 空的就開新的一組。已經在組裡的不重複加。

    paths：用檔案路徑指定（一批生完自動送來比較時用；只收圖庫裡有的）。
    auto：這一批的識別（第一張的路徑）。好幾個 CoCo 視窗都會偵測到「這批生完了」，
    同一批只建一組，後來的直接回傳那一組。
    新加進來的圖直接加入目前這一輪（淘汰賽已經比到後面的話，不會只待在第一輪）"""
    found = []
    for i in ids or []:
        r = con.execute("SELECT path FROM images WHERE id = ?", (int(i),)).fetchone()
        if r:
            found.append(r[0])
    for p in paths or []:
        r = con.execute("SELECT path FROM images WHERE lower(path) = lower(?)", (str(p),)).fetchone()
        if r:
            found.append(r[0])
    paths = found
    with _LOCK:
        d = _load()
        if auto:
            for k, s in d.items():
                if s.get("auto") == auto:
                    return {"id": k, "added": 0, "count": len(s.get("items") or []), "existing": True}
        now = time.time()
        if not sid or sid not in d:
            sid = secrets.token_hex(4)
            while sid in d:
                sid = secrets.token_hex(4)
            d[sid] = {"name": (name or "").strip()[:80], "created": now, "items": []}
            if auto:
                d[sid]["auto"] = str(auto)
        s = d[sid]
        have = {_norm(it["path"]) for it in s["items"]}
        added = 0
        rnd = int(s.get("round") or 1)
        for p in paths:
            if _norm(p) not in have:
                s["items"].append({"path": p, **({"reach": rnd} if rnd > 1 else {})})
                have.add(_norm(p))
                added += 1
        s["updated"] = _stamp(d)
        _save(d)
    return {"id": sid, "added": added, "count": len(d[sid]["items"])}


def update_set(sid, name=None, items=None, round=None, done=None, archived=None):
    """改名、改順序、改勝出與備註、拿掉圖（items 是完整的新清單：[{path, winner, note, reach, adv}]）；
    round／done 是淘汰賽的輪次與「選好了」（done 也是清單上的「已完成」）；archived 封存"""
    with _LOCK:
        d = _load()
        s = d.get(sid)
        if s is None:
            return None
        if name is not None:
            s["name"] = str(name).strip()[:80]
        if items is not None:
            known = {_norm(it["path"]): it for it in s["items"]}
            new = []
            for it in items:
                p = str(it.get("path") or "")
                if _norm(p) not in known:
                    continue          # 只能重排或拿掉，不能經這裡塞進任意路徑
                keep = {"path": known[_norm(p)]["path"]}
                if it.get("winner"):
                    keep["winner"] = True
                if it.get("note"):
                    keep["note"] = str(it["note"])[:2000]
                try:
                    reach = max(1, min(999, int(it.get("reach") or 1)))
                except (TypeError, ValueError):
                    reach = 1
                if reach > 1:
                    keep["reach"] = reach
                if it.get("adv"):
                    keep["adv"] = True
                new.append(keep)
            s["items"] = new
        if round is not None:
            try:
                s["round"] = max(1, min(999, int(round)))
            except (TypeError, ValueError):
                pass
        if done is not None:
            s["done"] = bool(done)
        if archived is not None:
            s["archived"] = bool(archived)
            s["seen"] = time.time()       # 取消封存之後重新算，不會馬上又被自動封存
        s["updated"] = _stamp(d)
        _save(d)
    return True


BULK = ("archive", "unarchive", "done", "undone", "delete")


def bulk(ids, action):
    """一次處理好幾組（清單的「選取」）：封存、取消封存、標成已完成／比較中、刪除。回傳處理了幾組"""
    if action not in BULK:
        return None
    n = 0
    with _LOCK:
        d = _load()
        now = time.time()
        for sid in ids or []:
            s = d.get(str(sid))
            if s is None:
                continue
            if action == "delete":
                del d[str(sid)]
            elif action in ("archive", "unarchive"):
                s["archived"] = action == "archive"
                s["seen"] = now
            else:
                s["done"] = action == "done"
            n += 1
        if n:
            _save(d)
    return n


def delete_set(sid):
    with _LOCK:
        d = _load()
        if d.pop(sid, None) is None:
            return False
        _save(d)
    return True


def winner_paths():
    """每一組標了勝出的圖（路徑小寫；圖庫的「勝出」篩選用，使用者筆記：勝出的圖不必一組一組點開找）"""
    return sorted({_norm(it["path"]) for s in _load().values() for it in s.get("items") or []
                   if it.get("winner") and it.get("path")})


def forget_paths(paths):
    """圖丟進回收桶或刪掉了：從每一組拿掉；拿光的那組一起刪（使用者筆記：不要留下空白的圖或空的組）。
    回傳 {items: 拿掉幾張, sets: 刪了幾組}"""
    gone = {_norm(p) for p in paths or []}
    if not gone:
        return {"items": 0, "sets": 0}
    items = sets = 0
    with _LOCK:
        d = _load()
        for sid in list(d):
            s = d[sid]
            before = s.get("items") or []
            keep = [it for it in before if _norm(it.get("path", "")) not in gone]
            if len(keep) == len(before):
                continue
            items += len(before) - len(keep)
            if keep:
                s["items"] = keep
                s["updated"] = _stamp(d)
            else:
                del d[sid]
                sets += 1
        if items:
            _save(d)
    return {"items": items, "sets": sets}


def rename_paths(mapping):
    """圖庫搬移了檔案：{舊路徑: 新路徑}，比較組裡的也跟著改"""
    if not mapping:
        return
    low = {_norm(a): b for a, b in mapping.items()}
    with _LOCK:
        d = _load()
        changed = False
        for s in d.values():
            for it in s.get("items") or []:
                nb = low.get(_norm(it["path"]))
                if nb:
                    it["path"] = nb
                    changed = True
        if changed:
            _save(d)
