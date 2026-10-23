# -*- coding: utf-8 -*-
"""變數與萬用字元的讀寫。
"""

import threading

from .state import C, ROOT

from i18n import tr

# 變數檔的讀、改、寫回在同一把鎖裡：兩個視窗同時新增不同的變數，不會只剩一筆
_VARS_LOCK = threading.Lock()

# 隨附的範例放這裡（git 追蹤、跟著 zip 發布）。使用者的 variables.txt 與 wildcards\ 不進 git、不在 zip 裡：
# 照說明「解壓縮到同一個資料夾、取代檔案」更新時，改過的才不會被蓋回範例
EXAMPLES = ROOT / "assets" / "examples"


def prepare_files():
    """啟動時：舊版的變數檔先搬過來，再補上缺的範例（反過來的話範例先佔了 variables.txt，舊的變數就搬不過來）"""
    migrate_vars()
    ensure_examples()


def ensure_examples():
    """第一次啟動時放進範例：variables.txt 不存在才放；wildcards\\ 整個資料夾不存在才放
    （資料夾在就不補個別的檔：使用者刪掉的範例不會每次啟動又跑回來）。已經有的檔一律不寫"""
    jobs = [(EXAMPLES / "variables.txt", C.VARS_FILE)]      # 已經有了就寫不進去（下面的排他建立）
    if not C.WILDCARD_DIR.exists():
        src = EXAMPLES / "wildcards"
        jobs += [(f, C.WILDCARD_DIR / f.relative_to(src)) for f in sorted(src.rglob("*.txt"))]
    for src, dst in jobs:
        try:
            data = src.read_bytes()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "xb") as fh:          # 排他建立：檢查之後才冒出來的同名檔也不會被蓋掉
                fh.write(data)
        except OSError:
            pass                                 # 放不進去就是沒有範例，其他功能照常


# ---------------------------------------------------------------- 萬用字元
def wc_list():
    d = C.WILDCARD_DIR
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.rglob("*.txt")):
        rel = f.relative_to(d).as_posix()
        if rel.startswith("_") or rel.endswith(".bak"):
            continue                      # 變數檔與備份不算萬用字元
        try:
            n = len([l for l in f.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.strip().startswith("#")])
        except Exception:
            n = 0
        out.append({"name": rel[:-4], "file": rel, "lines": n})
    return out


VARS_HEADER = (
    "# 變數：在工作台的「變數」分頁新增與編輯，不需要手動改這個檔\n"
    "# 格式： 名稱 = 值      （值可以包含 {A|B} 之類的模板語法）\n"
    "#\n"
    "# 變數與萬用字元不同：\n"
    "#   變數     一個名稱對應一段固定內容，用 ${名稱} 取用\n"
    "#   萬用字元 wildcards\\ 底下的清單檔，用 __檔名__ 隨機抽一行\n\n")

def migrate_vars():
    """舊版把變數放在 wildcards\\_variables.txt，搬到根目錄"""
    old = C.WILDCARD_DIR / "_variables.txt"
    if old.is_file() and not C.VARS_FILE.exists():
        C.VARS_FILE.write_text(old.read_text(encoding="utf-8"),
                               encoding="utf-8")
    if old.is_file():
        try:
            old.rename(old.with_suffix(".txt.bak"))
        except Exception:
            pass


def vars_read():
    f = C.VARS_FILE
    out = []
    if not f.is_file():
        return out
    for raw in f.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip():
            out.append({"name": k.strip(), "value": v.strip()})
    return out


def vars_write(items):
    width = max([len(i["name"]) for i in items] + [4])
    body = "".join(f'{i["name"]:<{width}} = {i["value"]}\n' for i in items)
    # 先寫同資料夾的暫存檔再原子取代：寫到一半出錯也不會只剩半份
    tmp = C.VARS_FILE.with_name(C.VARS_FILE.name + ".tmp")
    try:
        tmp.write_text(VARS_HEADER + body, encoding="utf-8")
        tmp.replace(C.VARS_FILE)
    finally:
        tmp.unlink(missing_ok=True)
    return items


def vars_apply(data):
    with _VARS_LOCK:
        return _vars_apply(data)


def _vars_apply(data):
    items = vars_read()
    act = data.get("action")
    name = (data.get("name") or "").strip()
    old = (data.get("old_name") or "").strip()
    value = (data.get("value") or "").strip()

    if act == "delete":
        items = [i for i in items if i["name"] != name]
    elif act in ("add", "update"):
        if not name:
            return {"error": tr("srv.ud.nameEmpty")}
        if "=" in name:
            return {"error": tr("srv.ud.nameEq")}

        if act == "update" and old:
            hit = next((i for i in items if i["name"] == old), None)
            if hit is None:
                return {"error": tr("srv.ud.notFound", name=old)}
            if name != old and any(i["name"] == name for i in items):
                return {"error": tr("srv.ud.exists", name=name)}
            hit["name"], hit["value"] = name, value
        else:
            if any(i["name"] == name for i in items):
                return {"error": tr("srv.ud.existsEdit", name=name)}
            items.append({"name": name, "value": value})
    else:
        return {"error": tr("srv.badAction")}

    items.sort(key=lambda i: i["name"].lower())
    vars_write(items)
    return {"ok": True, "vars": items}


def wc_read(rel):
    f = (C.WILDCARD_DIR / rel).resolve()
    C.WILDCARD_DIR.resolve()
    try:
        f.relative_to(C.WILDCARD_DIR.resolve())
    except ValueError:
        return None
    return f.read_text(encoding="utf-8") if f.is_file() else ""


def wc_write(rel, text, create=False):
    """create：「新增」——已經有同名的檔就不寫、回 exists（以前新增同名的會把原檔清空）；不然是編輯後的儲存（覆寫）"""
    if not rel.endswith(".txt"):
        rel += ".txt"
    f = (C.WILDCARD_DIR / rel).resolve()
    try:
        f.relative_to(C.WILDCARD_DIR.resolve())
    except ValueError:
        return {"error": tr("srv.ud.outside")}
    f.parent.mkdir(parents=True, exist_ok=True)
    if create:
        try:
            with open(f, "x", encoding="utf-8") as fh:     # 排他建立：檢查與建立之間不會被別的請求搶先
                fh.write(text or "")
        except FileExistsError:
            return {"error": tr("srv.ud.wcExists", name=rel[:-4]), "exists": True}
    else:
        f.write_text(text or "", encoding="utf-8")
    return {"ok": True, "file": f.relative_to(C.WILDCARD_DIR).as_posix()}
