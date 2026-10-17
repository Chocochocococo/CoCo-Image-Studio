# -*- coding: utf-8 -*-
"""存回 Eagle：把 CoCo 的圖（精修、編輯完的也行）加進 Eagle。

走 Eagle 開著時的本機 API（http://localhost:41595，預設就開著）。權限金鑰（Eagle 4 的 developer.apiToken）
從 /api/application/info 讀，使用者不用自己貼。Eagle 的 API 只能加進「目前開著的那個檔案庫」，
所以存之前會先告訴使用者現在開的是哪一個。加進去的是複本（Eagle 自己複製一份），CoCo 的原檔不動。
存過的記在 explore.db 的 eagle_saved（同一張再存先提醒，不會靜靜多一份）。
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from i18n import tr
from pathlib import Path

BASE = "http://localhost:41595"
TIMEOUT = 10
_TOKEN = {"v": None}


def _call(path, body=None, token=True):
    """呼叫 Eagle 的 API → data；Eagle 沒開回 None；Eagle 回錯誤丟 RuntimeError。
    連不上先再試一次（實測 Eagle 偶爾一下子沒回應，不要因此就說它沒開）"""
    got = _call_once(path, body, token)
    if got is None and body is None:
        time.sleep(0.5)
        got = _call_once(path, body, token)
    return got


def _call_once(path, body=None, token=True):
    q = {}
    if token and _TOKEN["v"]:
        q["token"] = _TOKEN["v"]
    url = BASE + path + (("?" + urllib.parse.urlencode(q)) if q else "")
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(tr("srv.eg.httpErr", code=e.code)) from e
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if d.get("status") != "success":
        raise RuntimeError(str(d.get("message") or d.get("data") or tr("srv.eg.err")))
    return d.get("data")


def _ensure_token():
    if _TOKEN["v"]:
        return True
    info = _call("/api/application/info", token=False)
    if info is None:
        return False
    _TOKEN["v"] = (((info.get("preferences") or {}).get("developer") or {}).get("apiToken")) or ""
    return True


def status():
    """Eagle 開著嗎、現在開的是哪個檔案庫、它的資料夾（存的時候選）"""
    try:
        if not _ensure_token():
            return {"running": False}
        lib = _call("/api/library/info")
        if lib is None:
            return {"running": False}
    except RuntimeError as e:
        return {"running": True, "error": str(e)}

    def tree(fs):
        return [{"id": str(f.get("id")), "name": str(f.get("name") or ""), "children": tree(f.get("children") or [])}
                for f in fs or [] if isinstance(f, dict) and f.get("id")]
    L = lib.get("library") or {}
    return {"running": True, "library": {"name": L.get("name") or "", "path": L.get("path") or ""},
            "folders": tree(lib.get("folders"))}


def tag_key(tag):
    """比對用：大小寫、底線與空白都不算差別（best_quality ＝ Best Quality）"""
    return re.sub(r"\s+", " ", str(tag).replace("_", " ")).strip().lower()


def prompt_tags(prompt, skip=""):
    """正面提示詞 → 標籤：依逗號、換行、BREAK 切開，拿掉 <lora:…> 這類語法、權重與強調括號
    （(blue eyes:1.2) → blue eyes；跳脫的 \\( \\) 是名字的一部分，留成括號）。大小寫不同的算同一個。
    skip：使用者設定要略過的標籤（逗號隔開的字串或清單）"""
    s = re.sub(r"<[^>]*>", ",", prompt or "")
    s = s.replace("\\(", "\x01").replace("\\)", "\x02")
    s = re.sub(r":\s*-?\d*\.?\d+\s*(?=[)\]]|,|$)", "", s, flags=re.M)
    s = re.sub(r"[()\[\]{}]", ",", s)
    if isinstance(skip, str):
        skip = skip.replace("，", ",").split(",")
    out, seen = [], {tag_key(x) for x in skip or [] if str(x).strip()}
    for part in re.split(r",|\n|\bBREAK\b|\|", s):
        t = re.sub(r"\s+", " ", part.replace("\x01", "(").replace("\x02", ")")).strip()[:100]
        if t and tag_key(t) not in seen:
            seen.add(tag_key(t))
            out.append(t)
    return out


def saved_before(con, path):
    r = con.execute("SELECT lib, time FROM eagle_saved WHERE path = ? ORDER BY time DESC LIMIT 1", (str(path).lower(),)).fetchone()
    return {"lib": r[0], "time": r[1]} if r else None


def save(con, path, name, folder="", tags=(), annotation="", website=""):
    """把 path 加進 Eagle 目前開著的檔案庫。回傳 {ok, library} 或 {error}"""
    p = Path(path)
    if not p.is_file():
        return {"error": tr("srv.eg.noFile")}
    try:
        if not _ensure_token():
            return {"error": tr("srv.eg.notRunning"), "notRunning": True}
        lib = (_call("/api/library/info") or {}).get("library") or {}
        body = {"path": str(p), "name": name or p.stem, "tags": [str(t)[:100] for t in tags if str(t).strip()][:300]}
        if folder:
            body["folderId"] = str(folder)
        if annotation:
            body["annotation"] = str(annotation)[:5000]
        if website:
            body["website"] = str(website)[:2000]
        try:
            got = _call("/api/item/addFromPath", body)
        except RuntimeError:
            _TOKEN["v"] = None                      # 金鑰可能換過：重拿一次再試
            if not _ensure_token():
                return {"error": tr("srv.eg.notRunning"), "notRunning": True}
            got = _call("/api/item/addFromPath", body)
        if got is None:
            return {"error": tr("srv.eg.notRunning"), "notRunning": True}
    except RuntimeError as e:
        return {"error": tr("srv.eg.rejected", e=e)}
    con.execute("INSERT INTO eagle_saved(path, lib, folder, time) VALUES (?,?,?,?)",
                (str(p).lower(), str(lib.get("path") or ""), str(folder or ""), time.time()))
    con.commit()
    return {"ok": True, "library": lib.get("name") or ""}
