# -*- coding: utf-8 -*-
"""標籤庫搜尋與提示詞裡的 <lora:…> 解析。
"""

import re
import sqlite3

from .state import ROOT


# ---------------------------------------------------------------- 標籤庫
TAGS_DB = ROOT / "assets" / "tags" / "base.db"

# 中文翻譯與細分類的疊加檔：一行「標籤<TAB>中文<TAB>分類」，# 開頭的是註解。蓋過 base.db 的（base.db 是 20 MB 的
# 二進位檔，改一次 git 就多存一份；文字檔看得到差異、也可以直接打開來改）。中文或分類空著的照 base.db；
# 中文寫「-」＝清掉 base.db 的翻譯（base.db 有些是規則亂組的，例如把畫師 gray_gray 翻成「灰色灰色」）
ZH_CLEAR = "-"
ZH_FILE = ROOT / "assets" / "tags" / "zh_TW.tsv"
# 細分類（web/js/chips.js 的 CAT_GROUP 對得到顏色組的）；疊加檔寫了別的照 base.db
FINE_CATS = frozenset(("subject", "species", "age", "profession", "hair", "eyes", "appearance", "clothing", "accessory",
                       "expression", "mood", "pose", "action", "scene", "lighting", "weather_time", "composition", "effect",
                       "object", "quality", "meta", "negative", "uncategorized"))
_ZH = {"key": None, "map": {}}


def zh_overlay():
    """{標籤: (中文, 分類)}。檔案改了下次查詢就用新的，不用重開"""
    try:
        st = ZH_FILE.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    if _ZH["key"] != key:
        out = {}
        for line in ZH_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, zh, cat = (line.split("\t") + ["", ""])[:3]
            zh, cat = zh.strip(), cat.strip()
            out[name.strip()] = (zh, cat if cat in FINE_CATS else "")
        _ZH.update(key=key, map=out)
    return _ZH["map"]


def with_zh(row):
    """內建庫查到的一筆（dict，要有 name）換上疊加檔的中文與分類"""
    o = zh_overlay().get(row.get("name"))
    if o:
        if o[0]:
            row["zh"] = "" if o[0] == ZH_CLEAR else o[0]
        if o[1]:
            row["cat"] = o[1]
    return row


# 畫師與角色名佔了整個庫的 68%，對描述性標籤的自動完成是雜訊。
# 不排除（Illustrious 系模型很吃畫師名），但排在後面。
_TAG_BACK = ("artist", "character", "copyright", "contributor")

# 查詢裡有中文字就用中文找（打「紅髮」找到 red_hair）
CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_ZHIDX = {"key": None, "rows": []}


def _zh_index():
    """中文搜尋的索引：[(中文, 標籤, 使用次數, 慣例標籤嗎)]。疊加檔的中文加上 base.db 原本的
    （疊加檔改過或清掉的照疊加檔）。疊加檔改了就重建（約 0.2 秒），之後每次查詢只掃這份清單"""
    over = zh_overlay()
    if _ZHIDX["key"] == (_ZH["key"], TAGS_DB) and _ZHIDX["rows"]:
        return _ZHIDX["rows"]
    rows = []
    if TAGS_DB.exists():
        con = sqlite3.connect(f"file:{TAGS_DB}?mode=ro", uri=True)
        try:
            zh = {n: z for n, z in con.execute("SELECT name, zh FROM tags WHERE zh != ''")}
            for name, (z, _) in over.items():
                if z:
                    zh[name] = z               # 清掉的變成「-」，中文查詢不會比對到
            names = list(zh)
            for i in range(0, len(names), 500):
                part = names[i:i + 500]
                for n, posts, pin in con.execute(
                        f"SELECT name, posts, pin FROM tags WHERE name IN ({','.join('?' * len(part))})", part):
                    rows.append((zh[n], n, posts or 0, pin or 0))
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
    _ZHIDX.update(key=(_ZH["key"], TAGS_DB), rows=rows)
    return rows


def _zh_search(q, limit):
    """中文找標籤：中文完全相同的最前面，接著開頭相同、包含；同一級照慣例標籤、使用次數。
    不像英文那樣把角色排後面：畫師沒有中文，角色有中文的都是常用的（打「初音」要先看到初音未來）"""
    hits = []
    for zh, name, posts, pin in _zh_index():
        i = zh.find(q)
        if i >= 0:
            hits.append((0 if zh == q else 1 if i == 0 else 2, -pin, -posts, name))
    names = [h[-1] for h in sorted(hits)[:limit]]
    if not names:
        return []
    con = sqlite3.connect(f"file:{TAGS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        got = {r["name"]: with_zh(dict(r)) for r in con.execute(
            f"SELECT name, zh, cat, kind, posts, nsfw, '' AS via FROM tags WHERE name IN ({','.join('?' * len(names))})", names)}
    except sqlite3.Error:
        got = {}
    finally:
        con.close()
    return [got[n] for n in names if n in got]


def tag_search(q, limit=15):
    """標籤自動完成。空字串就回熱門的；有中文字就用中文找（_zh_search）"""
    q = (q or "").strip()
    if CJK.search(q):
        return _with_mine(q, _zh_search(q, limit) if TAGS_DB.exists() else [], limit)
    q = q.lower().replace(" ", "_")
    if not TAGS_DB.exists():
        return []
    con = sqlite3.connect(f"file:{TAGS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        like = q.replace("%", r"\%").replace("_", r"\_") + "%" if q else "%"
        # 排序一定要在 SQL 裡做完。先前是撈一批再用 Python 重排，
        # 但 SQL 的 LIMIT 是按熱門度截的——查 dark 會被
        # dark_blue_eyes（別名，170 萬）之類擠掉，真正的 dark（1.5 萬）
        # 根本進不了候選名單。
        #   rank 0 名稱完全相符　1 名稱開頭或別名完全相符　2 別名開頭相符
        # 別名完全相符放在第 1 級而不是第 0 級，讓熱門度去分高下：
        #   boobs → breasts（別名相符 340 萬）勝過 boobs_on_face（名稱相符 26）
        #   twin  → twintails（名稱相符 90 萬）勝過 twins_(lore)（別名相符 9 千）
        # 放第 0 級的話後者會贏，但那顯然不是使用者要的
        rows = con.execute("""
            SELECT name, zh, cat, kind, posts, nsfw, via FROM (
              SELECT name, zh, cat, kind, posts, nsfw, pin, '' AS via,
                     CASE WHEN name = ?2 THEN 0 ELSE 1 END AS rank
                FROM tags WHERE name LIKE ?1 ESCAPE '\\'
              UNION ALL
              SELECT t.name, t.zh, t.cat, t.kind, t.posts, t.nsfw, t.pin,
                     a.alias,
                     CASE WHEN a.alias = ?2 THEN 1 ELSE 2 END
                FROM aliases a JOIN tags t ON t.id = a.tag_id
               WHERE a.alias LIKE ?1 ESCAPE '\\'
            )
            GROUP BY name
            ORDER BY MIN(rank),
                     MAX(COALESCE(pin,0)) DESC,   -- 品質詞等慣例標籤排前面
                     MIN(CASE WHEN kind IN
                       ('artist','character','copyright','contributor')
                     THEN 1 ELSE 0 END), posts DESC
            LIMIT ?3
        """, (like, q, limit)).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    return _with_mine(q, [with_zh(dict(r)) for r in rows], limit)


def _with_mine(q, rows, limit):
    """併進你的標籤庫（server/usertags.py）：完全相符的最前面，接著你自己的標籤，內建的收藏排在其他內建標籤前面"""
    from . import usertags
    try:
        own, favs = usertags.mine_search(q, limit)
    except sqlite3.Error:
        return rows
    for r in rows:
        r["fav"] = 1 if r["name"] in favs else 0
    own = [{"name": r["name"], "zh": r["zh"], "cat": r["cat"], "kind": "mine", "posts": 0, "nsfw": 0, "via": "", "fav": r["fav"],
            "mine": 1} for r in own if r["name"] not in {x["name"] for x in rows}]
    exact = [r for r in rows + own if r["name"] == q]
    rest = own + sorted((r for r in rows if r["name"] != q), key=lambda r: -r["fav"])
    return (exact + [r for r in rest if r not in exact])[:limit]


def tag_key(text):
    """提示詞裡的寫法 → 標籤庫的鍵：Blue Eyes → blue_eyes、hakurei reimu \\(touhou\\) → hakurei_reimu_(touhou)"""
    return re.sub(r"\\([()])", r"\1", (text or "").strip()).replace(" ", "_").lower()


def tag_lookup(names, limit=400):
    """一次查一批標籤的分類與中文（標籤模式的顏色用）。

    回傳 {原本的寫法: {name, cat, zh}}，查不到的不放。先比名稱、再比別名——
    提示詞裡常寫別名（boobs），顏色要跟正名（breasts）一樣。
    """
    want = {}
    for n in list(dict.fromkeys(names or []))[:limit]:
        if isinstance(n, str) and n.strip():
            want.setdefault(tag_key(n), []).append(n)
    if not want:
        return {}
    if not TAGS_DB.exists():                       # 沒有內建庫照樣查你的標籤
        con = sqlite3.connect(":memory:")
        con.executescript("CREATE TABLE tags (id, name, cat, zh, posts); CREATE TABLE aliases (alias, tag_id);")
    else:
        con = sqlite3.connect(f"file:{TAGS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    hits = {}
    try:
        keys = list(want)
        for i in range(0, len(keys), 200):
            part = keys[i:i + 200]
            q = ",".join("?" * len(part))
            for r in con.execute(f"SELECT name, cat, zh FROM tags WHERE name IN ({q})", part):
                hits[r["name"]] = with_zh(dict(r))
            rest = [k for k in part if k not in hits]
            if rest:
                q = ",".join("?" * len(rest))
                for r in con.execute(f"""
                        SELECT a.alias, t.name, t.cat, t.zh FROM aliases a JOIN tags t ON t.id = a.tag_id
                         WHERE a.alias IN ({q}) ORDER BY t.posts DESC""", rest):
                    hits.setdefault(r["alias"], with_zh({"name": r["name"], "cat": r["cat"], "zh": r["zh"]}))
    except sqlite3.Error:
        hits = {}
    finally:
        con.close()
    # 你的標籤庫：自己的標籤補上（內建庫沒有的），收藏的改過中文或分類就照你的
    from . import usertags
    try:
        mine = usertags.mine_lookup(list(want) + [v["name"] for v in hits.values()])
    except sqlite3.Error:
        mine = {}
    for k in want:
        if k not in hits and k in mine:
            hits[k] = mine[k]
    out = {}
    for k, v in hits.items():
        m = mine.get(v["name"]) or {}
        item = {"name": v["name"], "cat": m.get("cat") or v["cat"] or "", "zh": m.get("zh") or v["zh"] or ""}
        if m.get("own"):
            item["mine"] = 1
        for orig in want.get(k, []):
            out[orig] = item
    return out


_LORA_TAG = re.compile(r"<lora:([^:>]+?)(?::([-\d.]+))?>", re.I)

# 動態語法的標記。有這些的話，token 計數量到的是**模板**而不是實際送去
# 編碼的字串——展開之後每一則的長度都不一樣（見 core/template.py）。
_DYN_SYNTAX = re.compile(r"__[^_]+__|\{[^}]*\}|\[\[")


def extract_loras(prompt):
    """把提示詞裡的 <lora:名字:權重> 抽出來

    回傳 (去掉標記的提示詞, [(名字, 權重), ...])。

    為什麼要有這個：LoRA 用點卡片選很直覺，但**沒辦法做對照測試**。
    Forge 的做法是把 LoRA 寫在提示詞裡，於是可以用動態語法包起來：

        [[<lora:A:0.8>|<lora:B:0.8>|]]

    展開成三組（A、B、都不用），配上固定種子，就是標準的 LoRA 版本對照。
    卡片式介面做不到這件事，所以兩種並存：卡片管常用的，
    提示詞標記管這次要比較的。
    """
    found = []
    for m in _LORA_TAG.finditer(prompt or ""):
        name = m.group(1).strip()
        try:
            w = float(m.group(2)) if m.group(2) else 1.0
        except ValueError:
            w = 1.0
        if name:
            found.append((name, w))
    clean = _LORA_TAG.sub("", prompt or "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"(,\s*){2,}", ", ", clean).strip().strip(",").strip()
    return clean, found
