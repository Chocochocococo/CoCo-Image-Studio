# -*- coding: utf-8 -*-
"""按了才下載的共用部分：ControlNet（controlnet.py）、去背（rembg.py）、探索（explore.py）、
下載模型（modeldl.py，試用模型與拆分式模型的零件）都用這份。

一個下載工作＝一串檔案（網址、大小、SHA256、放到哪裡）＋一個進度 dict（done、total、error、file、running、
cancel、finished），前端輪詢狀態時看進度。先下載到旁邊的 .part，大小、SHA256 對了才換成正式的檔名：
  中斷（網路斷掉、關掉 CoCo）  .part 留著，下次用 HTTP Range 從斷掉的地方接著下載；伺服器不回 206 就從頭寫。
                              連線提早結束時 http.client 不會丟錯誤（只是讀到空的），所以讀完比該有的小也算中斷
  取消、內容不對               刪掉 .part
  SSL 憑證錯誤                直接的 SSLError，或 urlopen 包在 URLError 裡的（reason 是 SSLError），都說是憑證問題
已經有（大小對）的檔案不重下，算進進度。連線用系統的憑證庫（ssl 預設會載入 Windows 的）：
防毒軟體檢查 SSL 時，它的根憑證通常在這裡面。
"""

import hashlib
import http.client
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from i18n import tr

CHUNK = 1 << 20
TIMEOUT = 30                     # 秒：這麼久沒收到資料就算中斷


class BadFile(ValueError):
    """下載完的內容不對（比該有的大，或 SHA256 不對）"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def part_of(dest):
    dest = Path(dest)
    return dest.with_name(dest.name + ".part")


def partial(dest, total):
    """上次下載到一半留下多少（.part 的大小；比完整的還大的會被丟掉，不算）：畫面說「上次下載到 X%」用"""
    try:
        n = part_of(dest).stat().st_size
    except OSError:
        return 0
    return n if n <= total else 0


def on_disk(files):
    """一串檔案已經下載了多少：下載好的（大小對）整個算，其他的算 .part（多檔的下載說「上次下載到 N%」用）"""
    n = 0
    for f in files:
        dest = Path(f["dest"])
        try:
            done = dest.stat().st_size == f["bytes"]
        except OSError:
            done = False
        n += f["bytes"] if done else partial(dest, f["bytes"])
    return n


def fetch(f, st, open_):
    """下載一個檔。f：{url, bytes, sha256（None＝只看大小）, dest}。下載了新的回 True，本來就有回 False"""
    dest = Path(f["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == f["bytes"]:
        st["done"] += f["bytes"]
        return False
    part = part_of(dest)
    have = part.stat().st_size if part.is_file() else 0
    if have > f["bytes"]:
        part.unlink()
        have = 0
    st["file"] = dest.name
    if have == f["bytes"]:
        st["done"] += have                       # 上次下載完了還沒檢查：不用再連線（送 Range 會拿到 416）
    else:
        headers = {"User-Agent": "CoCo-Image-Studio"}
        if have:
            headers["Range"] = f"bytes={have}-"
        st["done"] += have                       # 先算進去：等伺服器回應時進度不會先跳回 0
        with open_(urllib.request.Request(f["url"], headers=headers)) as r:
            if have and getattr(r, "status", 200) != 206:
                st["done"] -= have
                have = 0                         # 伺服器從頭給：從頭寫
            with open(part, "ab" if have else "wb") as out:
                while True:
                    if st.get("cancel"):
                        raise InterruptedError
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    st["done"] += len(chunk)
    size = part.stat().st_size
    if size < f["bytes"]:
        # 連線在中途斷掉：http.client 不會丟錯誤（只是讀不到東西），SSL 也把突然斷掉當成結束。留著接著下載
        raise ConnectionError(tr("srv.dl.cutShort", pct=size * 100 // f['bytes']))
    if size != f["bytes"] or (f.get("sha256") and sha256(part) != f["sha256"]):
        part.unlink(missing_ok=True)
        what = "SHA256" if f.get("sha256") else tr("srv.dl.size")
        raise BadFile(tr("srv.dl.bad", file=dest.name, what=what))
    part.replace(dest)
    return True


def run(st, files, opener=None, hint="", on_saved=None):
    """依序下載 files（可以是 generator：放到哪裡要輪到才算，算的時候出錯也照樣回報），結果寫進 st（error 空的＝成功）。hint：SSL 錯誤時告訴使用者怎麼自己下載；
    on_saved(f)：每個新下載好的檔案換成正式檔名之後呼叫。opener 是測試用的（取代 urlopen）"""
    f = None
    try:
        ctx = ssl.create_default_context()
        open_ = opener or (lambda r: urllib.request.urlopen(r, timeout=TIMEOUT, context=ctx))
        for f in files:
            if fetch(f, st, open_) and on_saved:
                on_saved(f)
    except InterruptedError:
        part_of(f["dest"]).unlink(missing_ok=True)
        st["error"] = tr("srv.cancelled")
    except BadFile as e:
        st["error"] = str(e)
    except ssl.SSLError as e:
        st["error"] = _ssl_error(e, hint)
    except (OSError, http.client.HTTPException) as e:     # URLError 也是 OSError
        reason = getattr(e, "reason", None)
        if isinstance(reason, ssl.SSLError):
            st["error"] = _ssl_error(reason, hint)
        else:
            st["error"] = tr("srv.dl.interrupted", why=_net_error(e))
    except Exception as e:
        st["error"] = f"{type(e).__name__}: {e}"
    finally:
        st["running"] = False
        st["finished"] = time.time()


def _ssl_error(e, hint):
    return tr("srv.dl.ssl", e=e) + (tr("srv.dl.sslHint", hint=hint) if hint else "")


def _net_error(e):
    """網路錯誤 → 給使用者看的原因（常見的換成白話，拿掉 URLError 外面那層 <urlopen error …>）"""
    if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError) and isinstance(e.reason, BaseException):
        e = e.reason
    if isinstance(e, TimeoutError):
        return tr("srv.dl.timeout", n=TIMEOUT)
    if isinstance(e, socket.gaierror):
        return tr("srv.dl.offline")
    return str(e)


class Jobs(dict):
    """下載工作 → 進度。begin 開始（已經在下載就不再開一個）、cancel 取消、progress 給前端看的（不含 cancel）"""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

    def begin(self, job, total, target, args=()):
        with self.lock:
            cur = self.get(job)
            if cur and cur.get("running"):
                return {"ok": True}
            self[job] = {"done": 0, "total": total, "error": "", "file": "", "running": True, "cancel": False}
        threading.Thread(target=target, args=args, daemon=True).start()
        return {"ok": True}

    def cancel(self, job):
        with self.lock:
            cur = self.get(job)
            if cur and cur.get("running"):
                cur["cancel"] = True
        return {"ok": True}

    def progress(self, job):
        with self.lock:
            d = dict(self.get(job) or {})
        d.pop("cancel", None)
        return d or None
