"""圖在哪裡被用到：比較組（含勝出，重組拿它當來源）、Recipe 的封面、資源（模型、LoRA…）的封面、精修的底圖。

丟進回收桶或刪除之前問一次（使用者筆記：刪了被用到的圖要先提醒，按「繼續」就直接刪，不必再按一次）；
刪掉之後把這些地方清乾淨：從比較組拿掉（空掉的比較組一起刪）、封面換回預設、精修的底圖清掉。
"""

from pathlib import Path

from . import assets, compare, recipes
from .state import load_ui, update_ui


def _norm(p):
    return str(Path(str(p))).lower()


def image_uses(con, ids):
    """回傳 [{id, filename, uses: [{kind, name, winner?}]}]，只列有被用到的"""
    rows = {}
    for i in ids or []:
        r = con.execute("SELECT id, path, filename FROM images WHERE id = ?", (int(i),)).fetchone()
        if r:
            rows[_norm(r[1])] = {"id": r[0], "filename": r[2], "uses": []}
    if not rows:
        return []
    for s in compare._load().values():
        for it in s.get("items") or []:
            hit = rows.get(_norm(it.get("path", "")))
            if hit:
                hit["uses"].append({"kind": "compare", "name": s.get("name") or "", "winner": bool(it.get("winner"))})
    for r in recipes._load().values():
        hit = rows.get(_norm(r["cover"])) if r.get("cover") else None
        if hit:
            hit["uses"].append({"kind": "recipe", "name": r.get("name") or ""})
    for key, n in assets.load_notes().items():
        hit = rows.get(_norm(n["cover"])) if n.get("cover") else None
        if hit:
            kind, _, name = key.partition(":")
            hit["uses"].append({"kind": "cover", "name": name, "res": kind})
    try:
        src = int(load_ui().get("rsrc") or 0)
    except (TypeError, ValueError):
        src = 0
    for hit in rows.values():
        if src and hit["id"] == src:
            hit["uses"].append({"kind": "refine", "name": ""})
    return [v for v in rows.values() if v["uses"]]


def forget(paths, ids):
    """圖已經丟掉或刪掉了：各處清乾淨。回傳 {items: 從比較組拿掉幾張, sets: 刪了幾個空掉的比較組}"""
    out = compare.forget_paths(paths)
    recipes.forget_covers(paths)
    assets.forget_covers(paths)
    def clear_refine(ui):                       # 是這幾張才清：看與改在同一把鎖裡（state.update_ui）
        try:
            if int(ui.get("rsrc") or 0) in {int(i) for i in ids or []}:
                ui["rsrc"] = 0
                return True
        except (TypeError, ValueError):
            pass
        return False
    update_ui(clear_refine)
    return out
