# -*- coding: utf-8 -*-
"""後端（伺服器與生成子行程）給使用者看的訊息，照介面目前的語言產生。

翻譯跟前端共用同一份 web/locales/*.json（後端的鍵用 srv. 與 eng. 開頭），使用者改 zh-TW.json 就改得到後端的用字，
兩種語言鍵與變數一致的測試也一起涵蓋。缺字時跟前端一樣：退回繁體中文，再缺就顯示鍵名。

語言是整個行程共用一個：前端每個請求都帶 X-CoCo-Lang（web/js/api.js），伺服器收到就換；
生成子行程啟動時從環境變數 COCO_LANG 拿，之後每個指令都帶 lang（server/worker_client.send）。
介面只有一位使用者、各視窗的語言設定相同，所以不分請求各記一份；換語言之前已經產生的訊息（紀錄、排程的錯誤）維持原本的語言。
"""

import json
import os
import re
import sys
import threading
from pathlib import Path

LANGS = ("zh-TW", "en")
SOURCE = "zh-TW"
LOCALES = Path(__file__).resolve().parent.parent / "web" / "locales"

# 測試與程式有時用 `import i18n`、有時用 `core.i18n`，兩個名字要是同一個模組物件，
# 不然一邊換了語言另一邊不知道（見 CLAUDE.md「常踩的坑」的 anima 例子）
sys.modules.setdefault("i18n", sys.modules[__name__])
sys.modules.setdefault("core.i18n", sys.modules[__name__])

_lock = threading.Lock()
_cache = {}
_lang = SOURCE


def _norm(lang):
    """瀏覽器或設定給的語言碼；認不得的照前端的規則：中文用繁中，其餘英文"""
    if not lang:
        return None
    lang = str(lang).strip()
    if lang in LANGS:
        return lang
    return SOURCE if lang.lower().startswith("zh") else "en"


def set_lang(lang):
    global _lang
    got = _norm(lang)
    if got:
        _lang = got


def lang():
    return _lang


def _dict(lang):
    with _lock:
        if lang not in _cache:
            try:
                _cache[lang] = json.loads((LOCALES / f"{lang}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _cache[lang] = {}
        return _cache[lang]


def tr(key, **params):
    """照目前語言取出 key 的文字，{名稱} 換成 params；跟前端 t() 的規則一樣"""
    s = _dict(_lang).get(key)
    if s is None:
        s = _dict(SOURCE).get(key, key)
    if not params:
        return s
    return re.sub(r"\{(\w+)\}", lambda m: str(params[m.group(1)]) if m.group(1) in params else m.group(0), s)


def join(items):
    """列舉用的分隔：中文用頓號，英文用逗號"""
    return tr("srv.listSep").join(str(x) for x in items)


set_lang(os.environ.get("COCO_LANG"))
