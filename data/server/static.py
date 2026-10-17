# -*- coding: utf-8 -*-
"""介面的靜態檔（web/ 底下），掛在網站根目錄。

/next 是新介面還在預覽時的網址，留著轉址到 /，舊書籤才不會壞掉。
"""

from pathlib import Path
from urllib.parse import unquote

from .state import ROOT

WEB_DIR = ROOT / "web"
OLD_PREFIX = "/next"

# 只送這些副檔名。沒列在這裡的一律 404，免得不小心把別的東西送出去。
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


def old_redirect(url_path):
    """/next、/next/… → 對應的新網址；不是舊網址就回 None"""
    if url_path == OLD_PREFIX or url_path.startswith(OLD_PREFIX + "/"):
        return "/" + url_path[len(OLD_PREFIX):].lstrip("/")
    return None


def resolve(url_path, root=None):
    """/… → (檔案路徑, MIME)；不在 web 裡、不存在或類型不允許就回 None

    路徑先 URL 解碼再 resolve，%2e%2e 之類的寫法跟 ../ 一樣會被擋下；
    resolve 之後必須仍在 root 底下，才算數。
    """
    root = Path(root or WEB_DIR).resolve()
    if not url_path.startswith("/"):
        return None
    rel = unquote(url_path).lstrip("/\\") or "index.html"
    try:
        p = (root / rel).resolve()
        p.relative_to(root)
    except (ValueError, OSError):
        return None
    if p.is_dir():
        p = p / "index.html"
    mime = MIME.get(p.suffix.lower())
    if not mime or not p.is_file():
        return None
    return p, mime
