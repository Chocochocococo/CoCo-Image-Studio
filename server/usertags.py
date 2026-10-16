# -*- coding: utf-8 -*-
"""你的標籤庫（使用者筆記 #1、#2；資源庫的「標籤」分頁）。

三樣東西，存在 CoCo 資料夾的 user_tags.db（不放進 gallery.db：那是可以重建的索引，這是你整理的資料）：
  自己的標籤   內建庫沒有的詞（角色名、LoRA 觸發詞、自創的詞），可以設分類與中文。
               自動完成與標籤模式的顏色都會一起查（prompts.tag_search／tag_lookup）
  收藏         內建庫的標籤加星號：自動完成排前面，標籤分頁點一下就加進提示詞
  標籤組合     一串標籤存成一組（例如自己角色的外觀），點一下整組加進提示詞
               ——跟萬用字元不同，萬用字元是每次抽一個，這是整組一起用

圖庫裡出現內建庫沒有的詞時不直接收：先列在「待確認」（依出現張數排序），收下才算；
略過的記起來，之後不再出現（打錯的字、句子、一次性的實驗詞才不會混進來）。

新增前比對內建庫：名稱相同的就是內建標籤（改成收藏），是別名的指出正名
（6.5 萬個同義詞：boobs → breasts），不另外收一份。
"""

import re
import sqlite3
import threading
import time

from .prompts import TAGS_DB, extract_loras, tag_key, with_zh
from .state import ROOT

from i18n import tr

DB = ROOT / "user_tags.db"
_LOCK = threading.Lock()

# 分類就是標籤模式的十種顏色（web/js/chips.js 的 CAT_GROUP），每種存一個內建庫的分類名
CATS = {"subject": "subject", "hair": "hair", "eyes": "eyes", "body": "appearance", "wear": "clothing",
        "face": "expression", "pose": "pose", "scene": "scene", "meta": "meta", "who": "character"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS mine (
  name  TEXT PRIMARY KEY,          -- 標籤庫的寫法（小寫、底線）
  own   INTEGER NOT NULL,          -- 1 自己的標籤（內建庫沒有）、0 收藏的內建標籤
  zh    TEXT DEFAULT '',
  cat   TEXT DEFAULT '',
  fav   INTEGER DEFAULT 0,
  note  TEXT DEFAULT '',
  added REAL,
  uses  INTEGER DEFAULT 0,
  used  REAL
);
CREATE TABLE IF NOT EXISTS groups (
  id      INTEGER PRIMARY KEY,
  name    TEXT NOT NULL,
  folder  TEXT DEFAULT '',
  text    TEXT NOT NULL,           -- 照提示詞的寫法（可以有權重）
  fav     INTEGER DEFAULT 0,
  note    TEXT DEFAULT '',
  added   REAL,
  updated REAL,
  uses    INTEGER DEFAULT 0,
  used    REAL
);
CREATE TABLE IF NOT EXISTS ignored (name TEXT PRIMARY KEY, added REAL);
"""

MAX_NAME = 120
MAX_TEXT = 5000


def connect():
    con = sqlite3.connect(str(DB), timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _base():
    return sqlite3.connect(f"file:{TAGS_DB}?mode=ro", uri=True) if TAGS_DB.exists() else None


def base_info(keys):
    """{鍵: {"name": 正名, "cat", "zh", "posts", "alias": 是不是別名}}，內建庫沒有的不放"""
    out = {}
    base = _base()
    if base is None:
        return out
    try:
        keys = [k for k in dict.fromkeys(keys) if k]
        for i in range(0, len(keys), 300):
            part = keys[i:i + 300]
            q = ",".join("?" * len(part))
            for r in base.execute(f"SELECT name, cat, zh, posts FROM tags WHERE name IN ({q})", part):
                out[r[0]] = with_zh({"name": r[0], "cat": r[1] or "", "zh": r[2] or "", "posts": r[3] or 0, "alias": False})
            rest = [k for k in part if k not in out]
            if rest:
                q = ",".join("?" * len(rest))
                for r in base.execute(f"""SELECT a.alias, t.name, t.cat, t.zh, t.posts FROM aliases a JOIN tags t ON t.id = a.tag_id
                                           WHERE a.alias IN ({q}) ORDER BY t.posts DESC""", rest):
                    out.setdefault(r[0], with_zh({"name": r[1], "cat": r[2] or "", "zh": r[3] or "", "posts": r[4] or 0, "alias": True}))
    finally:
        base.close()
    return out


def check(name):
    """新增前的比對：{key, base（內建庫的正名）, alias（是別名）, mine（已經在你的標籤庫）}"""
    key = tag_key(name)
    if not key:
        return {"error": tr("srv.ut.nameEmpty")}
    b = base_info([key]).get(key)
    with _LOCK:
        con = connect()
        try:
            row = con.execute("SELECT own, fav FROM mine WHERE name = ?", (b["name"] if b else key,)).fetchone()
        finally:
            con.close()
    return {"key": key, "base": b["name"] if b else "", "alias": bool(b and b["alias"]),
            "mine": bool(row and (row["own"] or row["fav"]))}


# ---------------------------------------------------------------- 清單

def listing():
    with _LOCK:
        con = connect()
        try:
            tags = [dict(r) for r in con.execute("SELECT * FROM mine")]
            groups = [dict(r) for r in con.execute("SELECT * FROM groups")]
            ignored = con.execute("SELECT COUNT(*) FROM ignored").fetchone()[0]
        finally:
            con.close()
    # 收藏的內建標籤：分類與中文沒改過的照內建庫
    info = base_info([t["name"] for t in tags if not t["own"]])
    for t in tags:
        b = info.get(t["name"])
        t["base_zh"], t["base_cat"], t["posts"] = (b["zh"], b["cat"], b["posts"]) if b else ("", "", 0)
    return {"tags": tags, "groups": groups, "ignored": ignored, "cats": list(CATS)}


# ---------------------------------------------------------------- 標籤

def save_tag(data):
    """新增或修改。內建庫的別名不收（回 alias_of 讓畫面問要不要收藏正名）；名稱跟內建庫相同的存成收藏"""
    key = tag_key(data.get("name"))
    if not key or len(key) > MAX_NAME:
        return {"error": tr("srv.ut.nameBad")}
    b = base_info([key]).get(key)
    if b and b["alias"]:
        return {"error": "alias", "alias_of": b["name"]}
    own = 0 if b else 1
    cat = str(data.get("cat") or "")
    if cat and cat not in CATS.values():
        cat = CATS.get(cat, "")
    now = time.time()
    with _LOCK:
        con = connect()
        try:
            old = con.execute("SELECT * FROM mine WHERE name = ?", (key,)).fetchone()
            row = dict(old) if old else {"name": key, "own": own, "zh": "", "cat": "", "fav": 0 if own else 1, "note": "",
                                         "added": now, "uses": 0, "used": None}
            for k in ("zh", "note"):
                if k in data:
                    row[k] = str(data[k] or "")[:MAX_NAME if k == "zh" else 1000].strip()
            if "cat" in data:
                row["cat"] = cat
            if "fav" in data:
                row["fav"] = 1 if data["fav"] else 0
            row["own"] = own
            if not own and not row["fav"] and not row["zh"] and not row["note"] and not row["cat"]:
                con.execute("DELETE FROM mine WHERE name = ?", (key,))      # 取消收藏、又沒改過什麼：不留
                con.commit()
                return {"ok": True, "name": key, "removed": True}
            con.execute("""INSERT OR REPLACE INTO mine (name, own, zh, cat, fav, note, added, uses, used)
                           VALUES (:name, :own, :zh, :cat, :fav, :note, :added, :uses, :used)""", row)
            con.execute("DELETE FROM ignored WHERE name = ?", (key,))
            con.commit()
        finally:
            con.close()
    return {"ok": True, "name": key, "own": own}


def delete_tag(name):
    key = tag_key(name)
    with _LOCK:
        con = connect()
        try:
            con.execute("DELETE FROM mine WHERE name = ?", (key,))
            con.commit()
        finally:
            con.close()
    return {"ok": True}


def used(names=(), group=None):
    """標籤分頁點了加進提示詞：記次數與時間（「最常用」「最近用過」排序用）"""
    now = time.time()
    with _LOCK:
        con = connect()
        try:
            for n in names or ():
                con.execute("UPDATE mine SET uses = uses + 1, used = ? WHERE name = ?", (now, tag_key(n)))
            if group is not None:
                con.execute("UPDATE groups SET uses = uses + 1, used = ? WHERE id = ?", (now, int(group)))
            con.commit()
        finally:
            con.close()
    return {"ok": True}


# ---------------------------------------------------------------- 組合

def save_group(data):
    name = str(data.get("name") or "").strip()[:MAX_NAME]
    text = str(data.get("text") or "").strip()[:MAX_TEXT]
    if not name:
        return {"error": tr("srv.ut.groupName")}
    if not text:
        return {"error": tr("srv.ut.groupEmpty")}
    folder = str(data.get("folder") or "").strip()[:MAX_NAME]
    now = time.time()
    with _LOCK:
        con = connect()
        try:
            gid = data.get("id")
            if gid not in (None, ""):
                cur = con.execute("UPDATE groups SET name = ?, folder = ?, text = ?, fav = ?, note = ?, updated = ? WHERE id = ?",
                                  (name, folder, text, 1 if data.get("fav") else 0, str(data.get("note") or "")[:1000], now, int(gid)))
                if not cur.rowcount:
                    return {"error": tr("srv.ut.groupNotFound")}
            else:
                gid = con.execute("INSERT INTO groups (name, folder, text, fav, note, added, updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                  (name, folder, text, 1 if data.get("fav") else 0, str(data.get("note") or "")[:1000], now, now)).lastrowid
            con.commit()
        finally:
            con.close()
    return {"ok": True, "id": int(gid)}


def delete_group(gid):
    with _LOCK:
        con = connect()
        try:
            con.execute("DELETE FROM groups WHERE id = ?", (int(gid),))
            con.commit()
        finally:
            con.close()
    return {"ok": True}


# ---------------------------------------------------------------- 待確認（圖庫裡出現、內建庫沒有的）

_DYNAMIC = re.compile(r"__|\||\{|\}|\[\[|\]\]")
_WEIGHT = re.compile(r":\s*[-+]?[\d.]+\s*$")


def prompt_tags(prompt):
    """提示詞 → 標籤庫的鍵（去掉 LoRA、權重、括號；動態語法與註解不算）"""
    text, _ = extract_loras(prompt or "")
    text = re.sub(r"<[^>]*>", "", text)
    out = []
    for line in text.splitlines():
        line = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]
        for raw in re.split(r",|\bBREAK\b", line):
            t = raw.strip()
            if not t or _DYNAMIC.search(t):
                continue
            for _ in range(3):                                  # ((詞:1.2)) 這種一層一層剝
                t = _WEIGHT.sub("", t.strip())
                t = re.sub(r"^(?<!\\)[(\[]+|(?<!\\)[)\]]+$", "", t).strip()
            key = tag_key(t)
            if key and len(key) <= 60 and not re.fullmatch(r"[\d._\-:]+", key):
                out.append(key)
    return out


def candidates(gallery, limit=400):
    """圖庫提示詞裡、內建庫沒有、也還沒收下或略過的詞，依出現張數排序。
    words＝幾個字（5 個以上多半是句子，畫面預設收起來）；sample＝最近一張用到的圖"""
    seen = {}
    for r in gallery.execute("SELECT id, mtime, prompt FROM images WHERE prompt != '' ORDER BY mtime DESC"):
        for key in set(prompt_tags(r["prompt"])):
            hit = seen.get(key)
            if hit:
                hit[0] += 1
            else:
                seen[key] = [1, r["id"], r["mtime"]]
    known = base_info(list(seen))
    with _LOCK:
        con = connect()
        try:
            skip = {r[0] for r in con.execute("SELECT name FROM mine")} | {r[0] for r in con.execute("SELECT name FROM ignored")}
        finally:
            con.close()
    rows = [{"name": k, "images": v[0], "sample": {"id": v[1], "v": v[2]}, "words": len([w for w in k.split("_") if w])}
            for k, v in seen.items() if k not in known and k not in skip]
    rows.sort(key=lambda x: (-x["images"], x["name"]))
    return {"items": rows[:limit], "total": len(rows)}


def accept(names, cat=""):
    """待確認的收下：當成自己的標籤，可以一次指定分類"""
    done = 0
    for n in names or ():
        r = save_tag({"name": n, "cat": cat})
        done += 1 if r.get("ok") else 0
    return {"ok": True, "added": done}


def ignore(names, undo=False):
    now = time.time()
    with _LOCK:
        con = connect()
        try:
            for n in names or ():
                key = tag_key(n)
                if undo:
                    con.execute("DELETE FROM ignored WHERE name = ?", (key,))
                else:
                    con.execute("INSERT OR REPLACE INTO ignored (name, added) VALUES (?, ?)", (key, now))
            con.commit()
        finally:
            con.close()
    return {"ok": True}


def ignored_list():
    with _LOCK:
        con = connect()
        try:
            return {"items": [r[0] for r in con.execute("SELECT name FROM ignored ORDER BY added DESC")]}
        finally:
            con.close()


# ---------------------------------------------------------------- 給自動完成與標籤模式

def mine_search(q, limit=15):
    """自動完成：自己的標籤（名稱開頭、或中文含這幾個字）與收藏的名字。回傳 ([自己的標籤], {收藏的名字})"""
    q = (q or "").strip().lower().replace(" ", "_")
    if not DB.exists():
        return [], set()
    with _LOCK:
        con = connect()
        try:
            favs = {r[0] for r in con.execute("SELECT name FROM mine WHERE own = 0 AND fav = 1")}
            like = q.replace("%", r"\%").replace("_", r"\_") + "%"
            rows = con.execute("""SELECT name, zh, cat, fav FROM mine WHERE own = 1
                                    AND (name LIKE ?1 ESCAPE '\\' OR (?2 != '' AND zh LIKE '%' || ?2 || '%'))
                                  ORDER BY CASE WHEN name = ?3 THEN 0 ELSE 1 END, fav DESC, uses DESC, name LIMIT ?4""",
                               (like, q, q, limit)).fetchall()
        finally:
            con.close()
    return [dict(r) for r in rows], favs


def mine_lookup(keys):
    """標籤模式的顏色與中文：{鍵: {name, cat, zh}}。自己的標籤，以及改過中文或分類的收藏"""
    if not DB.exists() or not keys:
        return {}
    out = {}
    keys = list(dict.fromkeys(keys))
    with _LOCK:
        con = connect()
        try:
            for i in range(0, len(keys), 300):
                part = keys[i:i + 300]
                q = ",".join("?" * len(part))
                for r in con.execute(f"SELECT name, own, zh, cat FROM mine WHERE name IN ({q})", part):
                    if r["own"] or r["zh"] or r["cat"]:
                        out[r["name"]] = {"name": r["name"], "cat": r["cat"] or "", "zh": r["zh"] or "", "own": bool(r["own"])}
        finally:
            con.close()
    return out
