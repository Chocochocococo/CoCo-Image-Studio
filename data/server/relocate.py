# -*- coding: utf-8 -*-
"""CoCo 的東西都放進 data\\（2026-10-16 使用者要求：根目錄只留 CoCo.bat，一看就知道要點哪個）。

以前所有東西都在 CoCo.bat 旁邊。換成新的配置之後第一次啟動：CoCo.bat 先把 Python 與 venv 搬進 data\\
（程式不能搬自己正在用的），這裡再把其他 CoCo 的東西（圖、模型、設定、資料庫、快取）搬進來：
- 只搬 CoCo 自己會產生的（ITEMS），使用者自己放在旁邊的不碰；data\\ 裡已經有同名的不蓋掉（資料夾就合併、
  同名的檔案留在原處），寫進紀錄
- 狀態檔（JSON）與資料庫裡指到舊位置的路徑跟著改，改之前備份到 backups\\；路徑的寫法（\\ 或 /）照原本的
- 縮圖的檔名是用圖的路徑算的（server/library.py 的 thumb_path），跟著改名，不用重做
- 搬了什麼寫進 backups\\relocate.txt。搬到一半中斷的話（PENDING 還在），下次啟動照樣把路徑改完
"""

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from i18n import tr

# CoCo 自己會產生、以前放在 CoCo.bat 旁邊的東西。venv、python、install_mode.txt 由 CoCo.bat 先搬（它要先知道裝了什麼）；
# venv、python 不在這裡：兩邊都有的時候不能把兩個 venv 合併
ITEMS = ("outputs", "models", "hf_cache", "backups", "datasets", ".thumbs", ".trash", "masks", "covers", "edits", "cnrefs",
         "loras", "vae", "text_encoders", "embeddings", "wildcards",
         "settings.json", "gallery_ui.json", "asset_notes.json", "compare_sets.json", "recipes.json", "queue.json",
         "presets.json", "datasets.json", "file_hashes.json", "lora_hashes.json", "install_mode.txt", "prompts_out.txt",
         "variables.txt", "user_tags.db", "user_tags.db-journal", "tags_user.db",
         "gallery.db", "gallery.db-wal", "gallery.db-shm", "explore.db", "explore.db-wal", "explore.db-shm")
# 存了路徑的 JSON：data\\ 最上層這幾個，加上編輯的專案與回收筒的清單
JSON_FILES = ("settings.json", "gallery_ui.json", "asset_notes.json", "compare_sets.json", "recipes.json", "queue.json",
              "presets.json", "datasets.json", "file_hashes.json", "lora_hashes.json")
JSON_DIRS = ("edits", ".trash")
DBS = ("gallery.db", "explore.db")
PENDING = "relocate.pending"
EXE_NAME = "CoCo Image Studio.exe"        # 最外層的啟動程式（tools/make_launcher.py）
LOG = "relocate.txt"


def _key(p):
    return os.path.normcase(os.path.abspath(str(p)))


def _is_link(p):
    try:
        return p.is_symlink() or bool(getattr(os.lstat(p), "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def _move(src, dst, moves, skipped):
    """src 搬到 dst：同名的不覆蓋（兩邊都是資料夾就把裡面的一個一個搬進去）；連結（junction）整個搬，不進去搬裡面"""
    if dst.exists() or _is_link(dst):
        if src.is_dir() and dst.is_dir() and not _is_link(src) and not _is_link(dst):
            for x in sorted(src.iterdir()):
                _move(x, dst / x.name, moves, skipped)
            try:
                src.rmdir()                    # 只拿掉搬空的
            except OSError:
                pass
            return
        skipped.append(src)
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moves.append((src, dst))
    except OSError:
        skipped.append(src)


class Mapper:
    """舊位置的路徑 → data\\ 裡的。top 底下、不在 data 底下、第一層的東西在 data 裡有（不管是誰搬的：
    這裡、CoCo.bat、git），而且舊位置那個檔已經不在了（同名沒搬的留在原處，照舊指著它）就換；寫法（\\ 或 /）照原本的"""

    def __init__(self, top, data):
        self.top, self.data = Path(top), Path(data)
        self.tk, self.dk = _key(top).rstrip(os.sep) + os.sep, _key(data).rstrip(os.sep) + os.sep
        self.n = 0

    def __call__(self, value):
        if not isinstance(value, str) or len(value) < 3 or (":" not in value and not value.startswith("\\\\")):
            return value
        try:
            k = _key(value) if not value.startswith(("http:", "https:")) else ""
        except (ValueError, OSError):            # 不是路徑的字串（NUL 之類）
            return value
        if not k.startswith(self.tk) or k.startswith(self.dk):
            return value
        first = k[len(self.tk):].split(os.sep, 1)[0]
        if not first or not (self.data / first).exists() or os.path.lexists(value):
            return value
        rest = value[len(str(self.top).rstrip("\\/")):]
        new = str(self.data).rstrip("\\/")
        if "/" in value and "\\" not in value:
            new = new.replace("\\", "/")
        self.n += 1
        return new + rest

    def walk(self, o):
        if isinstance(o, dict):
            return {self(k): self.walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self.walk(x) for x in o]
        return self(o)


def _backup(f, data, stamp):
    # 檔名帶上它在 data 裡的資料夾：編輯的專案每個都叫 project.json，只用檔名會互相蓋掉
    rel = f.relative_to(data).as_posix().replace("/", "__") if f.is_relative_to(data) else f.name
    bak = data / "backups" / f"{rel}.{stamp}"
    bak.parent.mkdir(parents=True, exist_ok=True)
    if f.suffix == ".db":
        src = sqlite3.connect(str(f))
        try:
            dst = sqlite3.connect(str(bak))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    else:
        shutil.copy2(f, bak)
    return bak


def _json_files(data):
    for name in JSON_FILES:
        f = data / name
        if f.is_file():
            yield f
    for d in JSON_DIRS:
        p = data / d
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*.json") if x.is_file())


def _rewrite_json(data, mapper, stamp, changed):
    for f in _json_files(data):
        try:
            raw = f.read_text(encoding="utf-8")
            obj = json.loads(raw)
        except (OSError, ValueError):
            continue
        before = mapper.n
        new = mapper.walk(obj)
        if mapper.n == before:
            continue
        _backup(f, data, stamp)
        indent = 2 if raw.lstrip().startswith(("{\n", "[\n")) or "\n  " in raw else None
        f.write_bytes(json.dumps(new, ensure_ascii=False, indent=indent).encode("utf-8"))
        changed.append(f)


def _rewrite_db(f, data, mapper, stamp, changed, renamed):
    """每個一般表格（不含全文搜尋的虛擬表格與它的附屬表格）的文字欄位：指到舊位置的換掉。
    圖庫的 images 換了路徑，縮圖跟著改名"""
    con = sqlite3.connect(str(f))
    try:
        todo = []
        for _, name, typ, *_ in con.execute("pragma table_list").fetchall():
            if typ != "table" or name.startswith("sqlite_"):
                continue
            cols = [r[1] for r in con.execute(f"pragma table_info('{name}')") if (r[2] or "").upper() in ("TEXT", "")]
            for c in cols:
                rows = con.execute(f'select rowid, "{c}" from "{name}" where "{c}" like ? or "{c}" like ?',
                                   (str(mapper.top).rstrip("\\/") + "%", str(mapper.top).rstrip("\\/").replace("\\", "/") + "%")).fetchall()
                for rid, v in rows:
                    nv = mapper(v)
                    if nv != v:
                        todo.append((name, c, rid, v, nv))
        if not todo:
            return
        con.close()
        _backup(f, data, stamp)
        con = sqlite3.connect(str(f))
        thumbs = []
        with con:
            for name, c, rid, v, nv in todo:
                con.execute(f'update or ignore "{name}" set "{c}" = ? where rowid = ?', (nv, rid))
                if f.name == "gallery.db" and name == "images" and c == "path":
                    thumbs.append((rid, v, nv))
        changed.append(f)
        if thumbs:
            _rename_thumbs(con, thumbs, renamed)
    finally:
        con.close()


def _rename_thumbs(con, rows, renamed):
    try:
        from .library import thumb_path
    except Exception:
        return
    for rid, old, new in rows:
        r = con.execute("select mtime from images where rowid = ?", (rid,)).fetchone()
        if not r:
            continue
        a, b = thumb_path({"path": old, "mtime": r[0]}), thumb_path({"path": new, "mtime": r[0]})
        try:
            if a.is_file() and not b.exists():
                a.replace(b)
                renamed.append(b)
        except OSError:
            pass


def run(top, data, log=print):
    """舊的配置（東西在 top）→ data。回傳 {moved: [(舊, 新)], skipped: [舊], changed: [改了路徑的檔], thumbs: 改名的縮圖數}"""
    top, data = Path(top), Path(data)
    out = {"moved": [], "skipped": [], "changed": [], "thumbs": 0}
    # 只在新的配置裡做：data 就在 CoCo 的入口旁邊（CoCo Image Studio.exe，或過渡期放在最外層的 CoCo.bat）；
    # 舊的配置裡 top 是 CoCo 資料夾的上一層，不能去動那裡
    entry = (top / EXE_NAME).is_file() or (top / "CoCo.bat").is_file()
    if _key(top) == _key(data) or data.name.lower() != "data" or _key(data.parent) != _key(top) or not entry:
        return out
    moves, skipped = out["moved"], out["skipped"]
    pending = data / "backups" / PENDING
    todo = [n for n in ITEMS if (top / n).exists() or _is_link(top / n)]
    if not todo and not pending.exists():
        return out
    if todo:
        log("  " + tr("srv.rel.title"))
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(b"")
    for n in todo:
        _move(top / n, data / n, moves, skipped)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    mapper, renamed = Mapper(top, data), []
    _rewrite_json(data, mapper, stamp, out["changed"])
    for name in DBS:
        f = data / name
        if f.is_file():
            try:
                _rewrite_db(f, data, mapper, stamp, out["changed"], renamed)
            except sqlite3.Error as e:
                log("  [!] " + tr("srv.rel.dbErr", name=name, e=e))
    out["thumbs"] = len(renamed)
    _record(data, out, mapper.n)
    pending.unlink(missing_ok=True)
    for a, b in moves:
        if a.parent == top:
            log("  " + tr("srv.rel.moved", name=a.name))
    for a in skipped:
        log("  [!] " + tr("srv.rel.skipped", path=a))
    if mapper.n:
        log("  " + tr("srv.rel.rewrote", n=mapper.n, files=len(out["changed"])))
    return out


def _startup_lang(top, data):
    """搬家的訊息用哪個語言：跟 server/app.py 的 startup_lang 一樣（介面存的語言，「跟著瀏覽器」時看 Windows 的
    顯示語言）。這時候介面偏好可能還在舊位置，先找到的那份算"""
    pref = ""
    for f in (data / "gallery_ui.json", top / "gallery_ui.json"):
        if f.is_file():
            try:
                pref = json.loads(f.read_text(encoding="utf-8")).get("lang") or ""
            except (OSError, ValueError, AttributeError):
                pass
            break
    if not pref:
        try:
            import ctypes
            pref = "zh-TW" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x04 else "en"
        except Exception:
            pass
    return pref


def main(top, data):
    """tools/studio.py 在載入伺服器之前呼叫：伺服器一載入就讀設定，設定要先在 data 裡"""
    try:
        import i18n
        i18n.set_lang(_startup_lang(Path(top), Path(data)))
        return run(top, data)
    except Exception as e:
        print("  [!] " + tr("srv.app.relocateErr", e=e))
        return None


def _record(data, out, n):
    lines = [time.strftime("%Y-%m-%d %H:%M:%S")]
    lines += [f"{a} -> {b}" for a, b in out["moved"]]
    lines += [f"[skip] {a}" for a in out["skipped"]]
    lines += [f"[paths] {n} in {', '.join(f.name for f in out['changed'])}", f"[thumbs] {out['thumbs']}"]
    try:
        with open(data / "backups" / LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n\n")
    except OSError:
        pass
