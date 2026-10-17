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
