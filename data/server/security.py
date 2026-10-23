# -*- coding: utf-8 -*-
"""本機 HTTP 服務的 Host / Origin 檢查。
"""

from urllib.parse import urlparse

from .state import APP_PORT


def _is_loopback_netloc(value):
    """Host header 只接受目前這個 loopback 服務。

    只綁 127.0.0.1 仍擋不住 DNS rebinding：瀏覽器可以把攻擊者控制的
    網域解析到 127.0.0.1，再用那個網域當 Host 呼叫本機服務。
    """
    try:
        u = urlparse("//" + str(value or ""))
        return (u.hostname in {"127.0.0.1", "localhost"}
                and u.port == APP_PORT
                and not u.username and not u.password)
    except ValueError:
        return False


def _is_loopback_origin(value):
    """Origin 若存在，只接受同一個本機 HTTP 服務。"""
    if not value:
        return True
    try:
        u = urlparse(str(value))
        return (u.scheme == "http"
                and u.hostname in {"127.0.0.1", "localhost"}
                and u.port == APP_PORT
                and not u.username and not u.password)
    except ValueError:
        return False


def _is_same_site_request(headers):
    """GET /api/… 只接受 CoCo 自己的頁面（或使用者自己在網址列打的）發出的請求。

    GET 不帶 token，別的網站用 <img src> 或 fetch 就能打有副作用的 GET（/api/dataset/open 會寫
    datasets.json、把整個資料夾當資料集掃一遍），Host 擋不了這種（Host 照樣是 127.0.0.1）。
    瀏覽器會帶 Sec-Fetch-Site：same-origin＝自己的頁面、none＝使用者在網址列打的或書籤；cross-site
    與 same-site（同一台電腦別的 port 的網頁）都擋。沒有這個標頭的（curl、舊瀏覽器）看 Origin：
    跨來源的 fetch 才會帶，不是本機就擋。首頁與靜態檔不走這裡：從別處點連結打開 CoCo 要能開。"""
    site = str(headers.get("Sec-Fetch-Site") or "").strip().lower()
    if site and site not in ("same-origin", "none"):
        return False
    return _is_loopback_origin(headers.get("Origin"))
