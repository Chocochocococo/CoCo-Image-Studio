"""CoCo 自己的回收桶（.trash 資料夾）：看裡面有什麼、還原、永久刪除、清空。

丟進回收桶時（handler.file_op）把原本的路徑與整列索引記在 .trash/trash.json，
還原時放回原位、索引照原樣補回去（評分、提示詞都在，不用重新解析）。
以前丟進來、沒有紀錄的檔案不知道原本在哪，還原時要指定資料夾，索引重新讀檔建立。
"""

import json
import threading
import time
from pathlib import Path

import config as C

from . import fileops
from .library import make_thumb

MANIFEST = "trash.json"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
# 還原時補回索引的欄位（紀錄檔是本機檔案，還是只收認得的欄位名稱，不直接拼進 SQL）
COLS = ("path", "filename", "folder", "mtime", "filesize", "width", "height", "prompt", "negative", "steps", "cfg",
        "seed", "sampler", "scheduler", "model", "clip_skip", "loras", "extra", "raw", "rating", "indexed")
_LOCK = threading.Lock()


def trash_dir():
    return Path(C.TRASH_DIR)          # 設定存檔會 reload config，每次現取


def _load():
    p = trash_dir() / MANIFEST
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    p = trash_dir() / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def record(entries):
    """丟進回收桶的紀錄。entries：{回收桶裡的檔名: {orig: 原本的路徑, row: 索引那一列（dict，不含 id）}}"""
    if not entries:
        return
    with _LOCK:
        d = _load()
        now = time.time()
        for name, e in entries.items():
            d[name] = {"orig": str(e["orig"]), "time": now, "row": e.get("row") or None}
        _save(d)


def _safe(name):
    """只收回收桶裡的檔名（不能帶路徑、不能是紀錄檔），回傳完整路徑或 None"""
    name = str(name or "")
    if not name or name != Path(name).name or name in (MANIFEST, ".", ".."):
        return None
    p = trash_dir() / name
    return p if p.suffix.lower() in EXTS and p.is_file() else None


def list_items():
    """回收桶裡的圖，最近丟的在前面"""
    d = _load()
    out = []
    base = trash_dir()
    if not base.is_dir():
        return {"items": [], "bytes": 0}
    total = 0
    for p in base.iterdir():
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        st = p.stat()
        total += st.st_size
        e = d.get(p.name) or {}
        out.append({"name": p.name, "size": st.st_size, "v": f"{int(st.st_mtime)}-{st.st_size}",
                    "orig": e.get("orig") or None, "time": e.get("time") or st.st_mtime})
    out.sort(key=lambda it: it["time"], reverse=True)
    return {"items": out, "bytes": total}


def thumb(name):
    p = _safe(name)
    if not p:
        return None
    return make_thumb({"path": str(p), "mtime": p.stat().st_mtime})


def restore(con, names, dest=None):
    """放回原位（沒有紀錄的放到 dest）。回傳 {done, failed, need: 沒紀錄又沒給 dest 的檔名, errors}"""
    import gallery_index as GI
    with _LOCK:
        d = _load()
        done = failed = 0
        need, errs = [], []
        for name in names or []:
            p = _safe(name)
            if not p:
                failed += 1
                continue
            e = d.get(name) or {}
            if e.get("orig"):
                target = Path(e["orig"])
            elif dest:
                target = Path(dest) / name
            else:
                need.append(name)
                continue
            try:
                # 圖片和同名的 .txt 一起放回去：兩個名字都要沒人用，一起成功或一起維持原狀（server/fileops.py）
                target.parent.mkdir(parents=True, exist_ok=True)
                extra = fileops.companions(p)
                target = fileops.free_name(target.parent, target.name, extra)
                undo = fileops.transfer([(p, target)] + [(x, target.with_suffix(x.suffix)) for x in extra])
            except Exception as ex:
                failed += 1
                if len(errs) < 3:
                    errs.append(f"{name}: {str(ex)[:70]}")
                continue
            try:
                row = e.get("row")
                if row:
                    # 索引照原樣補回去（評分、提示詞都在），只換路徑與檔名
                    row = {**row, "path": str(target), "filename": target.name, "folder": str(target.parent)}
                    cols = [c for c in COLS if c in row]
                    con.execute(f"INSERT OR IGNORE INTO images ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                [row[c] for c in cols])
                else:
                    GI.index_file(con, target)
                d.pop(name, None)
                done += 1
            except Exception as ex:
                undo()                         # 索引補不回去：檔案也放回回收桶，下次再試
                failed += 1
                if len(errs) < 3:
                    errs.append(f"{name}: {str(ex)[:70]}")
        con.commit()
        _save(d)
    return {"ok": True, "done": done, "failed": failed, "need": need, "errors": errs}


def delete(names):
    """永久刪除（連同同名的 .txt）"""
    with _LOCK:
        d = _load()
        done = failed = 0
        for name in names or []:
            p = _safe(name)
            if not p:
                failed += 1
                continue
            try:
                p.unlink()
                txt = p.with_suffix(".txt")
                if txt.exists():
                    txt.unlink()
                d.pop(name, None)
                done += 1
            except OSError:
                failed += 1
        _save(d)
    return {"ok": True, "done": done, "failed": failed}


def empty():
    """清空：回收桶裡的圖（與同名的 .txt）全部刪掉"""
    return delete([it["name"] for it in list_items()["items"]])
