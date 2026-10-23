# -*- coding: utf-8 -*-
"""HTTP 路由。介面的靜態檔在 web/（見 static.py）。
"""

import html
import importlib
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import static
from . import (assets, compare, controlnet, dataset, eagle, train, edit, explore, fileops, install, jobqueue, kra, lorahash, modeldl, pairmaker, presets, recipes,
               refine, rembg, shortcut, trash, upscalers, usage, usertags)
from .library import build_query, lib_facets, make_thumb, open_folder, reveal, run_index, thumb_ver
from .prompts import extract_loras, tag_lookup, tag_search
from .resources import (KINDS, ROOT_KINDS, browse, detectors, model_arch, res_cover_path, res_dirs, res_scan, root_folders, set_kind,
                        resolve_lora)
from .security import _is_loopback_netloc, _is_loopback_origin, _is_same_site_request
from .settings_api import arch_switch, settings_out, update_settings
from .state import (API_TOKEN, APP_VERSION, C, DB_WRITE_LOCK, EDIT_REQUEST_BODY, INDEX_STATE, MAX_REQUEST_BODY,
                    PAGE_SIZE, ROOT, SET, SETTINGS_APPLY_LOCK, TPL, VERSION, db, load_ui,
                    save_ui, update_ui)
from .userdata import vars_apply, vars_read, wc_list, wc_read, wc_write
from .worker_client import WORKER

import i18n
from i18n import tr

class Server(ThreadingHTTPServer):
    """Windows 上把每條連線的送出緩衝設成 0，避免回應尾巴被丟掉

    實測在這台 Windows 上，一次送出 15 MB 的字型檔，送完馬上關線，
    16 次全部在最後 20～60 KB 收到連線重設（RST）：close（或
    shutdown）的當下資料還在系統的送出緩衝區，整段被丟掉。送完等
    3 秒再關就 8/8 完整，所以是「還沒送到就關」的問題；很可能是防毒
    之類的網路過濾在攔。先 shutdown(SHUT_WR) 的 lingering close 一樣
    0/8，因為 shutdown 本身就會觸發。

    SO_SNDBUF=0 時 Windows 直接從程式的緩衝區送，send 要等對方收到
    才回傳，關線時就沒有殘量。實測 16/16 完整，15 MB 約 0.4 秒，跟原本
    一樣快。Linux 設 0 會被夾到極小的緩衝、反而變慢，所以只在 Windows 用。
    """
    daemon_threads = True

    def get_request(self):
        sock, addr = super().get_request()
        if os.name == "nt":
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 0)
            except OSError:
                pass
        return sock, addr


# 寫回去時連線已經被瀏覽器關掉（見 Handler._send）
CLIENT_GONE = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)

# flow 架構的 shift 一個架構記一格：欄位與官方預設（前端 web/js/options.js 的 shiftKey 同一張表）
SHIFT_FIELDS = {"anima": ("ANIMA_SHIFT", 3.0), "zimage": ("ZIMAGE_SHIFT", 6.0), "krea2": ("KREA2_SHIFT", 3.16),
                "qwen21": ("QWEN21_SHIFT", 2.0), "klein4": ("FLUX2_SHIFT", 9.89), "klein9": ("FLUX2_SHIFT", 9.89),
                "flux1": ("FLUX1_SHIFT", 3.16), "flux1s": ("FLUX1S_SHIFT", 1.0), "kontext": ("FLUX1_SHIFT", 3.16)}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    @property
    def con(self):
        con = getattr(self, "_con", None)
        if con is None:
            con = db()
            self._con = con
        return con

    def finish(self):
        try:
            super().finish()
        finally:
            con = getattr(self, "_con", None)
            if con is not None:
                con.close()
                self._con = None

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; "
            # script 只准 web/js 的檔：首頁沒有 inline script（token 與版本號在 <meta>），javascript: 網址也跑不了
            "script-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )

    def _local_host(self):
        return _is_loopback_netloc(self.headers.get("Host"))

    def _post_authorized(self):
        if not self._local_host():
            return False
        if not _is_loopback_origin(self.headers.get("Origin")):
            return False
        got = self.headers.get("X-CoCo-Token") or ""
        return secrets.compare_digest(got, API_TOKEN)

    def _discard_small_request_body(self):
        """拒絕 POST 前讀完一般大小的 body，避免 Windows close() 送 RST。"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return
        if 0 < n <= MAX_REQUEST_BODY:
            self.rfile.read(n)

    def _send(self, body=b""):
        """送出標頭與內容。瀏覽器已經不等了（換頁、圖片被換掉時會取消還在載入的請求、關掉連線）就算了：
        不是 CoCo 的錯，回應也沒人要，不印錯誤（使用者回報：精修時視窗印出整段 WinError 10053）。
        Server 的 SO_SNDBUF=0 讓送出要等對方收到才回傳，所以取消的請求多半在這裡碰到。其他的錯照常丟出去"""
        try:
            self.end_headers()
            self.wfile.write(body)
        except CLIENT_GONE:
            self.close_connection = True

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self._send(b)

    def _bytes(self, data, mime, cache=True, tag=None):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        # 網址帶了 v=mtime，內容改變時網址就會變，所以可以放心長快取。
        # 沒帶 v 的一律不快取，避免索引重建後 id 對到別張圖。
        self.send_header("Cache-Control",
                         "max-age=604800, immutable" if cache else "no-store")
        if tag:
            self.send_header("ETag", f'"{tag}"')
        self._security_headers()
        self._send(data)

    def _row(self, rid):
        return self.con.execute(
            "SELECT * FROM images WHERE id=?", (rid,)).fetchone()

    def do_GET(self):
        i18n.set_lang(self.headers.get("X-CoCo-Lang"))   # 回應裡的訊息照介面目前的語言（web/js/api.js 每個請求都帶）
        if not self._local_host():
            return self._json({"error": "forbidden"}, 403)
        u = urlparse(self.path)
        if u.path.startswith("/api/") and not _is_same_site_request(self.headers):
            # 別的網站觸發的 GET（CSRF，見 server/security.py）：只擋 API，首頁與靜態檔照常
            return self._json({"error": "forbidden"}, 403)
        p = {k: v[0] for k, v in parse_qs(u.query).items()}

        moved = static.old_redirect(u.path)
        if moved is not None:
            # /next 是新介面預覽時的網址，舊書籤轉到新的位置
            self.send_response(301)
            self.send_header("Location", moved + (f"?{u.query}" if u.query else ""))
            self.send_header("Content-Length", "0")
            self._security_headers()
            self._send()

        elif not u.path.startswith("/api/"):
            hit = static.resolve(u.path)
            if not hit:
                return self._json({"error": "not found"}, 404)
            f, mime = hit
            b = f.read_bytes()
            if f.suffix == ".html":
                # __COCO_API_TOKEN__ 換成這次啟動的 token；__COCO_APP_VERSION__ 換成版本號（關於頁顯示）。
                # 兩個都在 <meta> 的 content 屬性裡（api.js、about.js 讀），所以照 HTML 屬性跳脫；
                # 不用 inline script，CSP 的 script-src 才能只留 'self'
                b = b.replace(b"__COCO_API_TOKEN__", html.escape(API_TOKEN, quote=True).encode("utf-8"))
                b = b.replace(b"__COCO_APP_VERSION__", html.escape(APP_VERSION, quote=True).encode("utf-8"))
            # 字型不會變（檔名含版本就換檔名），可以長快取；14 MB 的楷體
            # 每次都重抓太浪費。HTML / JS / CSS 開發中常改，一律不快取。
            self._bytes(b, mime, cache=f.suffix in (".ttf", ".woff2"))

        elif u.path == "/api/list":
            try:
                # 篩選可以重複帶同一個參數（多個 LoRA、多個模型…），這幾個要整串
                multi = parse_qs(u.query)
                for k in ("model", "lora", "folder", "root"):
                    if k in multi:
                        p[k] = multi[k]
                where, args, sort = build_query(p)
                off = int(p.get("offset") or 0)
                total = self.con.execute(
                    f"SELECT COUNT(*) FROM images{where}", args).fetchone()[0]
                rows = self.con.execute(
                    f"""SELECT id, filename, path, mtime, width, height,
                               rating, model, seed, prompt FROM images{where}
                        ORDER BY {sort} LIMIT ? OFFSET ?""",
                    args + [PAGE_SIZE, off]).fetchall()
                items = []
                for r in rows:
                    d = dict(r)
                    d["v"] = thumb_ver(d.pop("path"), d.pop("mtime"))
                    items.append(d)
                self._json({"total": total, "offset": off, "items": items})
            except sqlite3.OperationalError as e:
                self._json({"error": tr("srv.queryFail", e=e), "items": [], "total": 0})

        elif u.path == "/api/presets":
            self._json(presets.all_presets())

        elif u.path == "/api/presets/resolve":
            model = p.get("model") or SET.load().get("MODEL") or ""
            self._json(presets.resolve(model, p.get("arch") or model_arch(model)))

        elif u.path == "/api/refine/upscalers":
            self._json({"items": [{"name": m["name"], "bytes": m["bytes"]} for m in upscalers.upscalers()],
                        "scales": list(upscalers.SCALES), "max": upscalers.MAX_SIDE})

        elif u.path == "/api/rembg/status":
            self._json(rembg.status())

        elif u.path == "/api/controlnet/status":
            self._json(controlnet.status())

        elif u.path == "/api/modeldl/status":
            self._json(modeldl.status())

        elif u.path == "/api/controlnet/upload":
            # 從電腦選的參考圖（只給 cnrefs/ 裡的，用名字指定）
            f = controlnet.upload_path(p.get("name"))
            if not f:
                return self._json({"error": tr("srv.refNotFound")}, 404)
            self._bytes(f.read_bytes(), "image/png", cache=True)

        elif u.path == "/api/queue":
            self._json(jobqueue.listing())

        elif u.path == "/api/naming/preview":
            self._json(self.naming_preview(p.get("pattern", ""), p.get("kind", "gen")))

        elif u.path == "/api/dataset/open":
            self._json(dataset.open_dataset(p.get("dir", "")))

        elif u.path == "/api/dataset/state":
            self._json(dataset.job_state())

        elif u.path == "/api/dataset/stats":
            self._json(dataset.stats(p.get("dir", "")))

        elif u.path == "/api/dataset/opts":
            self._json({"opts": dataset.get_opts(), "recent": dataset.recent(), "model": dataset.model_status()})

        elif u.path == "/api/dataset/library":
            self._json({"items": dataset.library()})

        elif u.path == "/api/dataset/info":
            self._json(dataset.dataset_info(p.get("dir", "")))

        elif u.path in ("/api/dataset/thumb", "/api/dataset/full"):
            # 資料集的圖不在圖庫裡：路徑一定要在使用者開的資料集資料夾裡
            root = dataset.root_of(p.get("dir", ""))
            img = dataset.inside(root, p.get("path", "")) if root else None
            if not img:
                return self._json({"error": tr("srv.notFound")}, 404)
            if u.path == "/api/dataset/thumb":
                # 只有做縮圖的錯才算縮圖失敗；送出時的錯不能吞掉再送一次（見 /api/thumb）
                try:
                    t = make_thumb({"path": str(img), "mtime": int(img.stat().st_mtime)})
                    data = t.read_bytes() if t else None
                except Exception:
                    data = None
                if data is None:
                    return self._json({"error": tr("srv.thumbFail")}, 500)
                return self._bytes(data, "image/webp", cache=bool(p.get("v")), tag=t.stem)
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                    ".bmp": "image/bmp"}.get(img.suffix.lower(), "application/octet-stream")
            return self._bytes(img.read_bytes(), mime, cache=bool(p.get("v")))

        elif u.path == "/api/train/options":
            self._json(train.options())

        elif u.path == "/api/train/state":
            self._json(train.state())

        elif u.path == "/api/pairs/state":
            self._json(pairmaker.state())

        elif u.path == "/api/explore/status":
            self._json({"model": explore.model_status(), "index": explore.index_status(),
                        "sources": SET.load().get("REF_SOURCES") or []})

        elif u.path == "/api/explore/ref/detail":
            # 參考圖的詳細資料（跟 /api/detail 同格式，看大圖直接用）
            try:
                rid = int(str(p.get("id", "")).lstrip("r"))
            except ValueError:
                return self._json({"error": tr("srv.notFound")}, 404)
            d = explore.ref_detail(rid)
            self._json(d, 404 if "error" in d else 200)

        elif u.path == "/api/eagle/status":
            # 存回 Eagle：Eagle 開著嗎、現在開的是哪個檔案庫、它的資料夾
            self._json(eagle.status())

        elif u.path == "/api/explore/eagle":
            # 設定裡選 Eagle 資料夾用：檔案庫的資料夾樹
            self._json(explore.eagle_info(p.get("path", "")))

        elif u.path in ("/api/explore/ref/thumb", "/api/explore/ref/full"):
            # 參考圖：只給索引裡有的（用 id 指定，不收任意路徑）
            con = explore.db()
            try:
                r = con.execute("SELECT path, thumb, mtime FROM refs WHERE id = ?", (p.get("id"),)).fetchone()
            finally:
                con.close()
            if not r or not Path(r["path"]).is_file():
                return self._json({"error": tr("srv.refNotFound")}, 404)
            if u.path.endswith("thumb"):
                # Eagle 自己做好的縮圖比較小，從它縮快很多（原圖常是好幾千像素）
                small = r["thumb"] if r["thumb"] and Path(r["thumb"]).is_file() else r["path"]
                t = make_thumb({"path": small, "mtime": r["mtime"]})
                if t:
                    return self._bytes(t.read_bytes(), "image/webp", cache=True)
            f = Path(r["path"])
            mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}.get(f.suffix.lower(), "image/jpeg")
            self._bytes(f.read_bytes(), mime, cache=True)

        elif u.path == "/api/refine/versions":
            self._json(refine.versions(self.con, p.get("id")))

        elif u.path == "/api/refine/items":
            # 批次精修：選的這些圖的提示詞、模型與參數（照原圖重畫要用）
            self._json({"items": refine.items(self.con, p.get("ids") or "")})

        elif u.path == "/api/ui":
            self._json(load_ui())

        elif u.path == "/api/gen/state":
            st = dict(WORKER.state)
            st["alive"] = WORKER.alive()
            st["log"] = WORKER.log[-60:]
            st["table"] = WORKER.table
            st["queue_run"] = jobqueue.summary()       # 排程生成在做什麼（生成台的排程鈕、進度用）
            self._json(st)

        elif u.path == "/api/index_state":
            self._json(dict(INDEX_STATE))

        elif u.path == "/api/settings":
            self._json(settings_out())

        elif u.path == "/api/install":
            self._json(install.status())

        elif u.path == "/api/shortcut":
            try:
                self._json(shortcut.status())
            except (OSError, subprocess.SubprocessError, ValueError):
                self._json({"error": tr("srv.sc.fail")})

        elif u.path == "/api/tags":
            qs = parse_qs(u.query)
            self._json({"items": tag_search(
                (qs.get("q") or [""])[0],
                int((qs.get("limit") or ["15"])[0]))})

        elif u.path == "/api/vars":
            self._json({"vars": vars_read()})

        # 你的標籤庫（資源庫的「標籤」分頁，server/usertags.py）
        elif u.path == "/api/mytags":
            self._json(usertags.listing())
        elif u.path == "/api/mytags/check":
            self._json(usertags.check((parse_qs(u.query).get("name") or [""])[0]))
        elif u.path == "/api/mytags/candidates":
            self._json(usertags.candidates(self.con))
        elif u.path == "/api/mytags/ignored":
            self._json(usertags.ignored_list())

        elif u.path == "/api/wildcards":
            self._json({"files": wc_list(),
                        "dir": str(C.WILDCARD_DIR)})

        elif u.path == "/api/wildcard":
            t = wc_read(p.get("file", ""))
            self._json({"error": tr("srv.notFound")} if t is None else {"text": t})

        elif u.path == "/api/res":
            kind = p.get("kind", "model")
            self._json({"items": res_scan(kind),
                        "dirs": [str(d) for d in res_dirs(kind)]})

        elif u.path == "/api/res/cover":
            # LoRA 也可以用名字指定（生成台的 LoRA 格：提示詞裡的 <lora:名字> 只有名字）
            path = p.get("path", "")
            if not path and p.get("kind") == "lora" and p.get("name"):
                path = resolve_lora(p["name"]) or ""
            c = (assets.custom_cover(p.get("kind", "model"), path)
                 or res_cover_path(p.get("kind", "model"), path)) if path else None
            if not c:
                return self._json({"error": tr("srv.noCover")}, 404)
            # thumb=1：資源庫的卡片用縮圖（civitai 的封面常是好幾千像素的原圖），
            # 跟圖庫共用縮圖快取；詳細欄的大圖才送原檔
            if p.get("thumb"):
                try:
                    t = make_thumb({"path": str(c), "mtime": c.stat().st_mtime})
                    if t:
                        return self._bytes(t.read_bytes(), "image/webp", cache=True)
                except Exception:
                    pass          # 縮不了就照樣送原檔
            mime = {".png": "image/png", ".webp": "image/webp"}.get(c.suffix.lower(), "image/jpeg")
            self._bytes(c.read_bytes(), mime, cache=True)

        elif u.path == "/api/browse":
            self._json(browse(p.get("path", "")))

        elif u.path == "/api/facets":
            def col(c):
                return [dict(r) for r in self.con.execute(
                    f"""SELECT {c} v, COUNT(*) c FROM images
                        WHERE {c} IS NOT NULL AND {c} != ''
                        GROUP BY {c} ORDER BY c DESC LIMIT 40""")]
            roots = []
            try:
                import gallery_index as GI
                roots = [str(Path(d)) for d in GI.SCAN_DIRS]
            except Exception:
                pass
            self._json({"models": col("model"), "samplers": col("sampler"),
                        "folders": col("folder"), "roots": roots,
                        "total": self.con.execute(
                            "SELECT COUNT(*) FROM images").fetchone()[0]})

        elif u.path == "/api/lib/facets":
            try:
                import gallery_index as GI
                roots = [str(Path(d)) for d in GI.SCAN_DIRS]
            except Exception:
                roots = []
            self._json(lib_facets(self.con, roots))

        elif u.path == "/api/res/home":
            # CoCo 自己的模型資料夾與每一格（設定畫面用）
            from . import modelhome
            self._json(modelhome.summary())

        elif u.path == "/api/res/folders":
            # 模型總資料夾底下的子資料夾與分類（設定畫面用；root 給了就看那個資料夾，還沒存也能預覽）
            root = p.get("root")
            self._json({"root": root if root is not None else (SET.load().get("MODELS_ROOT") or ""),
                        "folders": root_folders(root=root, count=True) if root is not None else root_folders(count=True),
                        "kinds": list(ROOT_KINDS)})

        elif u.path == "/api/assets":
            kind = p.get("kind", "model")
            if kind not in KINDS:
                return self._json({"error": tr("srv.badKind", kinds=" / ".join(KINDS))}, 400)
            arch = model_arch(SET.load().get("MODEL") or "")
            self._json({"items": assets.asset_list(self.con, kind, arch),
                        "arch": arch, "dirs": [str(d) for d in res_dirs(kind)]})

        elif u.path == "/api/assets/images":
            self._json({"items": assets.used_images(self.con, p.get("kind", ""), p.get("key", ""))})

        elif u.path == "/api/recipes":
            self._json({"recipes": recipes.list_recipes(self.con)})

        elif u.path == "/api/compare/sets":
            try:
                days = int(load_ui().get("cmpautoarch") or 0)
            except (TypeError, ValueError):
                days = 0
            self._json({"sets": compare.list_sets(self.con, days)})

        elif u.path == "/api/compare/set":
            s = compare.get_set(self.con, p.get("id", ""))
            self._json(s if s else {"error": tr("srv.setNotFound")}, 200 if s else 404)

        elif u.path == "/api/detail":
            r = self._row(int(p.get("id", 0)))
            if not r:
                return self._json({"error": tr("srv.notFound")}, 404)
            d = dict(r)
            d["loras"] = json.loads(d.get("loras") or "[]")
            d["extra"] = json.loads(d.get("extra") or "{}")
            d["exists"] = Path(d["path"]).exists()
            self._json(d)

        elif u.path == "/api/edit/project":
            self._json({"project": edit.project_for(self.con, p.get("src", 0))})

        elif u.path == "/api/edit/projects":
            self._json({"projects": edit.list_projects(self.con)})

        elif u.path == "/api/edit/wmlogo":
            f = edit.logo_file()
            if not f.is_file():
                return self._json({"error": tr("srv.notFound"), "v": 0}, 404)
            return self._bytes(f.read_bytes(), "image/png", cache=bool(p.get("v")), tag=f"wm-{p.get('v', '')}")

        elif u.path == "/api/edit/wminfo":
            self._json({"v": edit.logo_version()})

        elif u.path == "/api/edit/layer":
            f = edit.layer_path(p.get("id", ""), p.get("layer", ""))
            if not f:
                return self._json({"error": tr("srv.notFound")}, 404)
            return self._bytes(f.read_bytes(), "image/png", cache=bool(p.get("v")), tag=f"{p.get('id')}-{f.stem}-{p.get('v', '')}")

        elif u.path == "/api/trash":
            self._json(trash.list_items())

        elif u.path == "/api/trash/thumb":
            try:
                t = trash.thumb(p.get("name", ""))
            except Exception:
                t = None
            if not t:
                return self._json({"error": tr("srv.notFound")}, 404)
            return self._bytes(t.read_bytes(), "image/webp", cache=bool(p.get("v")), tag=t.stem)

        elif u.path in ("/api/thumb", "/api/full"):
            r = self._row(int(p.get("id", 0)))
            if not r:
                return self._json({"error": tr("srv.notFound")}, 404)
            versioned = bool(p.get("v"))
            if u.path == "/api/thumb":
                # 只有做縮圖的錯才算縮圖失敗。以前送出也包在裡面：瀏覽器取消了，就被當成縮圖失敗、
                # 往斷掉的連線再送一次錯誤訊息
                try:
                    t = make_thumb(r)
                    data = t.read_bytes() if t else None
                except Exception:
                    data = None
                if data is None:
                    return self._json({"error": tr("srv.thumbFail")}, 500)
                return self._bytes(data, "image/webp", cache=versioned, tag=t.stem)
            src = Path(r["path"])
            if not src.exists():
                return self._json({"error": tr("srv.fileMissing")}, 404)
            mime = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
                        src.suffix.lower(), "application/octet-stream")
            self._bytes(src.read_bytes(), mime, cache=versioned)

        else:
            self._json({"error": "not found"}, 404)

    def gen_run(self, data):
        if train.busy():
            return {"error": tr("srv.h.training")}
        plan = self.plan_run(data)
        if "error" in plan:
            return plan
        if not WORKER.submit(plan["load"], plan["jobs"]):
            return {"error": tr("srv.h.busy")}
        r = {"ok": True, "queued": len(plan["jobs"])}
        if plan["warns"]:
            r["warn"] = "；".join(plan["warns"])
        return r

    LABEL = {k: f"srv.h.lbl.{k}" for k in ("batch", "steps", "cfg", "clip_skip", "shift", "seed")}     # 錯誤訊息裡的名稱，tr() 的鍵

    def plan_run(self, data, edit_path=None):
        """一次生成要做的事：回傳 {load: 載入設定, jobs: 工作, warns: 提醒} 或 {error}。
        生成台、精修送出用 gen_run；批次精修一張一張排好再一起送（refine_batch）。
        edit_path：成對圖輔助（server/pairmaker.py）用資料集的圖當條件做 Qwen 修圖；只從程式裡傳，API 送來的資料指定不了路徑"""
        cfg = SET.load()
        model = (data.get("model") or cfg.get("MODEL") or "").strip()
        if not model:
            return {"error": tr("srv.h.noModel")}

        # shift 一個架構記一格（Anima 3.0、Z-Image 6.0、Krea 2 3.16、Qwen Image 2.1 2.0、FLUX.2 klein 9.89、
        # FLUX.1 dev 3.16、schnell 1）：跟著這次的模型
        shift_field, shift_default = SHIFT_FIELDS.get(model_arch(model), SHIFT_FIELDS["anima"])

        prompts = data.get("prompts")
        if not prompts:
            p = (data.get("prompt") or cfg.get("PROMPT") or "").strip()
            if not p:
                return {"error": tr("srv.h.emptyPrompt")}
            prompts = [p]
        # 數字都在伺服器這邊再檢查一次（R17）：不是數字的回錯誤，超出範圍的夾回去並提醒
        warns = []
        nums = {}
        for key, field, typ, default in (("batch", "BATCH", int, 1), ("steps", "STEPS", int, 28), ("cfg", "CFG", float, 5.0),
                                         ("clip_skip", "CLIP_SKIP", int, 2), ("shift", shift_field, float, shift_default),
                                         ("seed", "SEED", int, -1)):
            raw = data.get(key, cfg.get(field))
            if raw in (None, ""):
                raw = default
            try:
                if isinstance(raw, bool):
                    raise TypeError
                if isinstance(raw, (int, float)):
                    v = raw if typ is float else int(raw)          # 5 還是 5（寫進圖的參數不會變成 5.0）
                elif typ is float:
                    v = float(raw)
                else:
                    v = int(raw) if key == "seed" else int(float(raw))   # seed 很大，不經過小數才不會失準
            except (TypeError, ValueError, OverflowError):
                return {"error": tr("srv.h.notNumber", what=tr(self.LABEL[key]))}
            if v != v or v in (float("inf"), float("-inf")):     # NaN、無限大
                return {"error": tr("srv.h.notNumber", what=tr(self.LABEL[key]))}
            c = SET.clamp(field, v)
            if c != v:
                warns.append(tr("srv.h.clamped", what=tr(self.LABEL[key]), raw=raw, v=f"{c:g}" if typ is float else c))
            nums[key] = c
        n = nums["batch"]
        if any(len(str(p)) > SET.MAX_PROMPT for p in prompts) or len(str(data.get("negative") or "")) > SET.MAX_PROMPT:
            return {"error": tr("srv.h.promptLong", n=SET.MAX_PROMPT)}
        if len(prompts) * n > SET.MAX_JOBS:
            return {"error": tr("srv.h.tooMany", max=SET.MAX_JOBS, combos=len(prompts), n=n)}

        # 一律存在 CoCo 的輸出資料夾（以前會照請求裡的 out_dir，等於誰都能指定寫到哪裡）
        out_dir = Path(C.OUTPUT_DIR)
        base = nums["seed"]
        fixed = bool(data.get("fix_seed", cfg.get("FIX_SEED")))
        if fixed and base < 0:
            import random
            base = random.randrange(1 << 31)   # 整批共用這一個

        cond_mode = data.get("cond_mode", cfg.get("COND_MODE", "auto"))
        modes = ["a1111", "native"] if cond_mode == "both" else [cond_mode]
        if cond_mode == "both" and model_arch(model) in SHIFT_FIELDS:
            return {"error": tr("srv.h.noCondCompare")}

        loras = data.get("loras")
        if loras is None:
            loras = [[l["path"], l["weight"]] for l in cfg.get("LORAS", [])]

        # img2img：底圖用圖庫的 id 指定（不收任意路徑），寬高預設跟底圖一樣
        init = {}
        width = data.get("width", cfg.get("WIDTH"))
        height = data.get("height", cfg.get("HEIGHT"))
        # 一般生成的寬高：256～4096、對齊 16（目前與之後的架構都要 16 的倍數，見 server/refine.snap）
        try:
            width, height = refine.snap(width or 1024), refine.snap(height or 1024)
        except (TypeError, ValueError):
            return {"error": tr("srv.h.sizeNum")}
        if edit_path:
            if model_arch(model) != "qwen21":
                return {"error": tr("srv.h.needQwen")}
            init = {"edit_image": str(edit_path)}
        elif data.get("init_id") not in (None, ""):
            src = refine.init_source(self.con, data.get("init_id"))
            if "error" in src:
                return src
            try:
                strength = float(data.get("strength", 0.5))
            except (TypeError, ValueError):
                return {"error": tr("srv.h.strengthNum")}
            init = {"init_image": src["path"], "strength": min(1.0, max(0.05, strength))}
            width = refine.snap(data.get("width") or src["width"])
            height = refine.snap(data.get("height") or src["height"])
            # 用指示修圖（Qwen Image 2.1、FLUX.2 klein 或 FLUX.1 Kontext）：這張圖當條件，從頭畫一張照指示改過的；
            # 大小照精修頁送來的（預設跟原圖一樣）
            if data.get("edit"):
                if model_arch(model) not in refine.EDIT_ARCHS:
                    return {"error": tr("srv.h.needEditModel")}
                init = {"edit_image": src["path"]}
                # 其他參考圖（圖2、圖3）：一樣只收圖庫的 id
                refs = data.get("edit_refs") or []
                if not isinstance(refs, list) or len(refs) > refine.EDIT_REFS_MAX:
                    return {"error": tr("srv.h.refsMax", n=refine.EDIT_REFS_MAX)}
                paths = []
                for rid in refs:
                    r = refine.init_source(self.con, rid)
                    if "error" in r:
                        return {"error": tr("srv.h.refGone")}
                    paths.append(r["path"])
                if paths:
                    init["edit_refs"] = paths
            # 局部重繪：遮罩是精修頁畫的 PNG（data URL）
            elif data.get("mask"):
                m = refine.save_mask(data["mask"], (width, height))
                if "error" in m:
                    return m
                try:
                    blur = max(0, min(64, int(data.get("mask_blur", 8))))
                except (TypeError, ValueError):
                    blur = 8
                init.update(mask_image=m["path"], mask_blur=blur,
                            inpaint_area="masked" if data.get("inpaint_area") == "masked" else "whole",
                            inpaint_fill="fill" if data.get("inpaint_fill") == "fill" else "original")
                # 塗色：蓋在底圖上當起點（只在遮罩裡有用）
                if data.get("sketch"):
                    sk = refine.save_sketch(data["sketch"], (width, height))
                    if "error" in sk:
                        return sk
                    if sk["path"]:
                        init["sketch_image"] = sk["path"]
            # 高清化（放大＋補細節）：放大到倍數的大小，再分塊重畫
            elif isinstance(data.get("upscale"), dict):
                up = data["upscale"]
                size = upscalers.target_size(src["width"], src["height"], up.get("scale"))
                if not size:
                    return {"error": tr("srv.h.upTooBig", n=upscalers.MAX_SIDE)}
                name = str(up.get("model") or "")
                upm = upscalers.find(name) if name else None          # 不要叫 model：那是生成用的模型
                if name and not upm:
                    return {"error": tr("srv.h.upGone")}
                width, height = size
                try:
                    blur = max(0, min(64, int(data.get("mask_blur", 32))))
                except (TypeError, ValueError):
                    blur = 32
                init.update(upscale={"model": upm["path"] if upm else "", "scale": float(up["scale"])}, mask_blur=blur)
            # 擴圖：畫布往四邊加大，新的地方交給局部重繪補上（pipeline._outpaint）
            elif isinstance(data.get("outpaint"), dict):
                op = refine.clean_outpaint(data["outpaint"], (src["width"], src["height"]))
                if "error" in op:
                    return op
                width, height = op["size"]
                try:
                    blur = max(0, min(64, int(data.get("mask_blur", 16))))
                except (TypeError, ValueError):
                    blur = 16
                init.update(outpaint=op["sides"], mask_blur=blur)
            # 批次精修的修臉：臉在工作裡找（core/worker.py），全部都修
            elif data.get("faces") == "auto":
                found = detectors()
                if not found:
                    return {"error": tr("srv.h.noDetector")}
                try:
                    blur = max(0, min(64, int(data.get("mask_blur", 8))))
                except (TypeError, ValueError):
                    blur = 8
                init.update(faces="auto", mask_blur=blur, face_model=found[0]["path"])
            # 修臉：精修頁偵測到、勾起來的臉（原圖座標）與每張另外加的提示詞
            elif data.get("faces") is not None:
                fs = refine.clean_faces(data.get("faces"), (src["width"], src["height"]))
                if "error" in fs:
                    return fs
                try:
                    blur = max(0, min(64, int(data.get("mask_blur", 8))))
                except (TypeError, ValueError):
                    blur = 8
                init.update(faces=fs["faces"], face_size=[src["width"], src["height"]], mask_blur=blur,
                            face_model=str(data.get("face_model") or ""))

        # 生成後自動修臉（像 ADetailer）：只有一般生成才做，精修本身就是在修
        auto_face = None
        if not init and data.get("auto_face", cfg.get("AUTO_FACE")):
            found = detectors()
            if found:
                try:
                    fs = min(1.0, max(0.05, float(cfg.get("AUTO_FACE_STRENGTH") or 0.4)))
                except (TypeError, ValueError):
                    fs = 0.4
                auto_face = {"model": found[0]["path"], "strength": fs, "blur": 8, "keep": bool(cfg.get("AUTO_FACE_KEEP"))}
            else:
                warns.append(tr("srv.h.noDetectorAuto"))

        # ControlNet（server/controlnet.py）：生成台的參考圖只用在一般生成；精修的擴圖、局部重繪可以用 union 的「重繪」
        cn_jobs = []
        sdxl = model_arch(model) == "sdxl"
        units = [u for u in (data.get("controlnets") or []) if isinstance(u, dict) and not u.get("off")][:3]
        if units and not init:
            if not sdxl:
                warns.append(tr("srv.h.cnSdxlOnly"))
            else:
                for u in units:
                    path = self.cn_image(u.get("src"))
                    if not path:
                        return {"error": tr("srv.h.cnRefGone")}
                    j = controlnet.unit_for_job(u, path)
                    if "error" in j:
                        return j
                    j["resize"] = "fit" if u.get("resize") == "fit" else "crop"
                    cn_jobs.append(j)
        if init and data.get("cn_repaint") and ("mask_image" in init or "outpaint" in init):
            um = controlnet.union_model()
            if not sdxl:
                warns.append(tr("srv.h.seamSdxlOnly"))
            elif not um or not um["config"]:
                return {"error": tr("srv.h.seamNeedUnion")}
            else:
                init["cn_repaint"] = {"model": um["path"], "config": um["config"], "name": um["name"],
                                      "scale": controlnet.clamp(data.get("cn_repaint_scale"), 0, 2, 1.0)}

        # 品質標籤與預設負面（server/presets.py）：開關存在設定裡，關掉會記住
        negative = data.get("negative", cfg.get("NEGATIVE", ""))
        use_preset = bool(data.get("preset", cfg.get("QUALITY_PRESET", True)))
        preset = presets.resolve(model, model_arch(model)) if use_preset else None

        jobs = []
        missing = set()
        _hashes = {}
        mhash = lorahash.model_hash(model)

        def lora_hash(path):
            """同一次送出裡同一個檔案只算一次（第一次遇到沒有說明檔、檔頭也沒記的才要讀整個檔，約 1 秒，之後有快取）"""
            if path not in _hashes:
                _hashes[path] = lorahash.short(path)
            return _hashes[path]
        logical_index = 0
        # Embeddings（core/embeddings.py）：提示詞裡寫檔名就套用。跟模型架構不合的不帶，另外提醒
        import embeddings
        arch_now = model_arch(model)
        emb_all = [(it["name"], it["path"]) for it in res_scan("embedding")]
        emb_bad = set()
        for i, pr in enumerate(prompts):
            # 提示詞裡的 <lora:名字:權重> 只對這一則生效，
            # 所以每個工作各自帶一份 LoRA 清單
            pr, tags = extract_loras(pr)
            # 動態語法已經在前端展開過，這裡每一則各自補上還沒有的品質標籤。
            # 中繼資料記完整的，另外記「使用者寫的」，送回生成區時才不會一直疊上去
            full_p, full_n, pinfo = pr, negative, {"preset": {"on": False}}
            if preset:
                full_p, full_n, added = presets.apply(pr, negative, preset)
                pinfo = {"preset": {"on": True, **added}}
                if added["quality"] or added["negative"]:
                    pinfo.update(user_prompt=pr, user_negative=negative)
            job_loras = [list(x) for x in loras]
            for nm, w in tags:
                path = resolve_lora(nm)
                if not path:
                    missing.add(nm)
                    continue
                for row in job_loras:
                    if Path(row[0]).stem.lower() == Path(path).stem.lower():
                        row[1] = w        # 卡片也選了同一個，以提示詞為準
                        break
                else:
                    job_loras.append([path, w])
            job_emb = []
            for name in sorted(embeddings.used(f"{full_p}\n{full_n}", [x for x, _ in emb_all])):
                hit = next(((x, pth) for x, pth in emb_all if x.lower() == name), None)
                ea = assets.asset_arch("embedding", hit[1]) if hit else ""
                if hit and (not ea or assets.compatible("embedding", ea, arch_now)):
                    job_emb.append(list(hit))
                elif hit:
                    emb_bad.add(hit[0])
            for k in range(n):
                if fixed:
                    job_seed = base
                elif base >= 0:
                    job_seed = base + logical_index
                elif len(modes) > 1:
                    # 對照模式即使選了隨機 seed，兩條路也必須共用同一顆。
                    import random
                    job_seed = random.randrange(1 << 31)
                else:
                    job_seed = -1
                for job_mode in modes:
                    jobs.append({
                        "cmd": "gen", "id": len(jobs),
                        "loras": job_loras,
                        # 每個 LoRA 的雜湊（Forge 的寫法，存圖時寫進參數；之後改了名字也對得上）
                        "lora_hashes": {Path(p).stem: lora_hash(p) for p, _ in job_loras if lora_hash(p)},
                        "embeddings": job_emb,
                        "prompt": full_p,
                        "negative": full_n,
                        **pinfo,
                        "width": width,
                        "height": height,
                        "steps": nums["steps"],
                        "cfg": nums["cfg"],
                        "seed": job_seed,
                        "sampler": data.get("sampler", cfg.get("SAMPLER")),
                        "schedule": data.get("schedule", cfg.get("SCHEDULE")),
                        "clip_skip": nums["clip_skip"],
                        "cond_mode": job_mode,
                        "out_dir": str(out_dir),
                        # 寫進圖的參數：模型的雜湊（知道的才有，第一次在背景算）、CoCo 的版本
                        **({"model_hash": mhash} if mhash else {}),
                        **({"coco_version": VERSION} if VERSION else {}),
                        # 檔名格式（⚙ 設定 → 生成 → 存檔名稱，core/naming.py）
                        "name_pattern": cfg.get("SAVE_NAME") or "",
                        # 亂數在哪裡產生（⚙ 設定 → 生成 →「用 CPU 產生亂數」，預設開，跟 Forge 一樣）
                        "rng": "cpu" if data.get("rng_cpu", cfg.get("RNG_CPU", True)) else "gpu",
                        # 加權方式與 NGMS（⚙ 設定 → 生成 → 畫質；只影響 SDXL／SD 1.5）
                        "emphasis": "Original" if cfg.get("EMPHASIS_NORM", True) else "No norm",
                        "ngms": float(cfg.get("NGMS") or 0),
                        **init,
                        **({"auto_face": auto_face} if auto_face else {}),
                        **({"controlnets": cn_jobs} if cn_jobs else {}),
                    })
                logical_index += 1

        # 拆分式模型（Anima、Z-Image、Krea 2、Qwen Image 2.1、FLUX.2 klein、FLUX.1）的 VAE 與文字編碼器：這個架構還沒選的自動補上，
        # 補上的記進設定（生成台的零件欄、資源庫看得到，之後要換也是在資源庫）。
        # FLUX.1 的全包檔：資源庫裡找不到的零件用檔案自帶的；CLIP-L 不佔欄位，另外找（assets.clip_for）
        vae_ov = data.get("vae_path", cfg.get("VAE_OVERRIDE", ""))
        te_ov = data.get("te_path", cfg.get("TE_OVERRIDE", ""))
        arch_now = model_arch(model)
        clip_path = ""
        if arch_now in assets.SPLIT_ARCHS:
            builtin = assets.builtin_parts(model) if arch_now in assets.FLUX1_ARCHS else set()
            got = assets.fill_parts(arch_now, {"vae": vae_ov, "te": te_ov}, builtin)
            if "error" in got:
                return got
            if arch_now in assets.FLUX1_ARCHS:
                clip = assets.clip_for(arch_now, "clip" in builtin)
                if "error" in clip:
                    return clip
                clip_path = clip["path"]
            vae_ov, te_ov = got["vae"], got["te"]
            if got["picked"]:
                key = {"vae": "VAE_OVERRIDE", "te": "TE_OVERRIDE"}
                update_settings({key[k]: {arch_now: (vae_ov if k == "vae" else te_ov)[arch_now]} for k, _, _ in got["picked"]})
                for k, name, old in got["picked"]:
                    lbl = assets.part_label(k)
                    warns.append(tr("srv.h.partSwapped", old=old, part=lbl, name=name) if old else
                                 tr("srv.h.partPicked", arch=assets.ARCH_TITLE[arch_now], part=lbl, name=name))

        load = {
            "model": model, "loras": loras,
            "vram_mode": data.get("vram_mode", cfg.get("VRAM_MODE")),
            "low_decode": data.get("low_decode",
                                   cfg.get("LOW_DECODE", "auto")),
            "tile_vae": data.get("tile_vae", cfg.get("TILE_VAE") or "auto"),
            "vae_path": vae_ov,
            "te_path": te_ov,
            **({"clip_path": clip_path} if clip_path else {}),
            # VAE 用 32 位元（⚙ 設定 → 生成；預設關，模型自帶的 SDXL VAE 用 bf16）。換了要重新載入模型
            "vae_fp32": bool(cfg.get("VAE_FP32")),
            # Z-Image 與 bf16 版 Krea 2、Qwen Image 2.1 的主模型用 fp8 存放（⚙ 設定 → 生成；出圖會變，換了要重新載入）
            "fp8_weights": bool(cfg.get("FP8_WEIGHTS")),
            "sampler": data.get("sampler", cfg.get("SAMPLER")),
            "schedule": data.get("schedule", cfg.get("SCHEDULE")),
            "shift": nums["shift"],
            "cond_mode": "auto" if cond_mode == "both" else cond_mode,
            # Qwen 修圖的其他參考圖幾張：生成子行程載入時多留空間，放不下就邊算邊搬（core/qwen21.refs_room、needs_reload）
            **({"edit_refs": len(init["edit_refs"])} if init.get("edit_refs") else {}),
        }
        if missing:
            warns.append(tr("srv.h.loraMissing", names=i18n.join(sorted(missing))))
        if emb_bad:
            warns.append(tr("srv.h.embBad", names=i18n.join(sorted(emb_bad))))
        return {"load": load, "jobs": jobs, "warns": warns}

    def eagle_save(self, data):
        """存回 Eagle：圖庫的圖（id）或資料夾來源的參考圖（rid）。生成參數寫進 Eagle 的註解、參考圖的來源網址帶過去。
        存過的再存要 force（前端先問過使用者）"""
        website, annotation = "", ""
        if data.get("rid") not in (None, ""):
            try:
                d = explore.ref_detail(int(str(data["rid"]).lstrip("r")))
            except ValueError:
                d = {"error": tr("srv.refNotFound")}
            if "error" in d:
                return d
            path, name, website, prompt = d["path"], Path(d["path"]).stem, d["ref"]["url"], d.get("prompt") or ""
            raw = explore._gi().read_meta(Path(path)).get("raw") if data.get("annotate", True) else ""
        else:
            r = self._row(int(data.get("id") or 0))
            if not r:
                return {"error": tr("srv.imgNotFound")}
            path, name, raw, prompt = r["path"], Path(r["path"]).stem, r["raw"], r["prompt"] or ""
        if data.get("annotate", True) and raw:
            annotation = raw
        tags = [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()]
        if data.get("promptTags"):                  # 正面提示詞也存成標籤（跟上面重複的、設定裡要略過的不加）
            tags += eagle.prompt_tags(prompt, [*tags, *str(SET.load().get("EAGLE_SKIP_TAGS") or "").replace("，", ",").split(",")])
        con = explore.db()
        try:
            before = eagle.saved_before(con, path)
            if before and not data.get("force"):
                return {"already": before}
            return eagle.save(con, path, name, str(data.get("folder") or ""), tags, annotation, website)
        finally:
            con.close()

    def cn_image(self, src):
        """ControlNet 參考圖的來源 → 檔案路徑：圖庫的圖（id）、探索的參考圖（rid）、從電腦選的（cnrefs 的名字）"""
        if not isinstance(src, dict):
            return None
        kind = src.get("kind")
        try:
            if kind == "lib":
                r = self._row(int(src.get("id") or 0))
                return r["path"] if r and Path(r["path"]).is_file() else None
            if kind == "ref":
                d = explore.ref_detail(int(str(src.get("rid") or "").lstrip("r")))
                return d["path"] if "error" not in d and Path(d["path"]).is_file() else None
        except (TypeError, ValueError):
            return None
        if kind == "upload":
            f = controlnet.upload_path(src.get("name"))
            return str(f) if f else None
        return None

    def controlnet_restore(self, recorded):
        """送到生成區：圖裡記的參考圖（路徑）→ 生成台的參考圖設定（圖庫的 id、探索的參考圖、cnrefs）"""
        def find_lib(path):
            r = self.con.execute("SELECT id FROM images WHERE path = ? COLLATE NOCASE", (path,)).fetchone()
            return r[0] if r else None

        def find_ref(path):
            con = explore.db()
            try:
                r = con.execute("SELECT id FROM refs WHERE path = ? COLLATE NOCASE", (path,)).fetchone()
            finally:
                con.close()
            return r[0] if r else None
        return controlnet.restore_units(recorded, find_lib, find_ref)

    def controlnet_preview(self, data):
        """參考圖視窗的「看處理結果」：照目前的生成大小裁好、做前處理，回傳縮小的 PNG（data URL）"""
        import base64
        path = self.cn_image(data.get("src"))
        if not path:
            return {"error": tr("srv.h.cnRefMissing")}
        typ = str(data.get("type") or "")
        prep = str(data.get("prep") or "none")
        if typ not in controlnet.PREPS or prep not in controlnet.PREPS[typ]:
            prep = "none"
        need = controlnet.PREP_NEEDS.get(prep)
        files = controlnet.prep_paths(need) if need else {}
        if need and not files:
            return {"error": tr("srv.h.needPrep"), "need": need}
        cfg = SET.load()
        w = refine.snap(data.get("width") or cfg.get("WIDTH") or 1024)
        h = refine.snap(data.get("height") or cfg.get("HEIGHT") or 1024)
        tmp = Path(tempfile.gettempdir()) / f"coco-cnprep-{secrets.token_hex(6)}.png"
        try:
            got = WORKER.ask_cnprep(path, prep, files, tmp, w, h, "fit" if data.get("resize") == "fit" else "crop",
                                    controlnet.clamp(data.get("low"), 1, 254, 100), controlnet.clamp(data.get("high"), 2, 255, 200), typ)
            if not got.get("ready"):
                return {"error": got.get("why") or tr("srv.h.prepFail")}
            from PIL import Image
            import io
            with Image.open(tmp) as im:
                im.thumbnail((768, 768))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "PNG")
            return {"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "width": w, "height": h,
                    **({"people": got["people"]} if "people" in got else {})}
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def naming_preview(pattern, kind="gen"):
        """⚙ 設定的存檔名稱：照這個格式，下一張圖會存成什麼（用生成台目前的設定當例子，seed 用 1234567890）。
        kind：gen（生成）、edit（06 輸出的編輯，原圖檔名當例子）、collage（拼接）"""
        import naming
        kind = kind if kind in naming.DEFAULTS else "gen"
        cfg = SET.load()
        req = {k: cfg.get(K) for k, K in (("width", "WIDTH"), ("height", "HEIGHT"), ("steps", "STEPS"), ("cfg", "CFG"), ("sampler", "SAMPLER"))}
        req["prompt"] = cfg.get("PROMPT") or "1girl, solo, smile"
        loras = [(x.get("path", ""), x.get("weight", 1)) for x in cfg.get("LORAS") or [] if isinstance(x, dict)]
        out = Path(C.OUTPUT_DIR)
        pat = (pattern or "").strip() or naming.DEFAULTS[kind]
        vals = naming.values(req, 1234567890, cfg.get("MODEL") or "", loras)
        vals["name"] = {"gen": "", "edit": "000001-1234567890-1024x1344", "collage": "collage"}[kind]
        path = naming.target(out, pat, vals, create=False, default=naming.DEFAULTS[kind])
        return {"path": str(path.relative_to(out)), "unknown": naming.unknown(pat),
                "noUnique": not any(f"{{{k}" in pat for k in ("seq", "seed", "time", *(("name",) if kind == "edit" else ())))}

    @staticmethod
    def _embed(paths):
        """探索：一批圖交給工作行程算特徵 → (向量, 成功的索引) 或 {error}"""
        import numpy as np
        tmp = Path(tempfile.gettempdir()) / f"coco-embed-{secrets.token_hex(6)}.npy"
        try:
            got = WORKER.ask_embed(explore.model_dir(), paths, tmp)
            if not got.get("ready"):
                return {"error": got.get("why") or tr("srv.embedFail")}
            return np.load(tmp), got.get("ok") or []
        finally:
            tmp.unlink(missing_ok=True)

    def explore_search(self, data):
        """找相似：以圖（圖庫的 id 或參考圖的 key）或文字。回傳 {items: [...], query}"""
        if not explore.model_ready():
            return {"error": tr("srv.noExploreModel")}
        scope = data.get("scope") if data.get("scope") in ("all", "lib", "ref") else "all"
        try:
            limit = max(1, min(500, int(data.get("limit") or 200)))
        except (TypeError, ValueError):
            limit = 200
        exclude = []
        if str(data.get("text") or "").strip():
            got = WORKER.ask_embed_text(explore.model_dir(), str(data["text"]).strip()[:300])
            if not got.get("ready"):
                return {"error": got.get("why") or tr("srv.h.textEmbedFail")}
            vec = got["vec"]
        else:
            if data.get("id") not in (None, ""):
                r = self._row(int(data["id"]))
                if not r:
                    return {"error": tr("srv.imgNotFound")}
                key = explore.lib_key(r["path"])
            else:
                key = str(data.get("key") or "")
            vec = explore.vector_of(key)
            if vec is None:
                return {"error": tr("srv.h.unindexed"), "unindexed": True}
            exclude = [key]
        hits = explore.search(vec, scope, limit, exclude)
        lib_paths = [k[4:] for k, _ in hits if k.startswith("lib:")]
        libs = {}
        for s in range(0, len(lib_paths), 500):
            part = lib_paths[s:s + 500]
            q = f"SELECT id, path, filename, width, height, mtime, model, seed, prompt FROM images WHERE lower(path) IN ({','.join('?' * len(part))})"
            for r in self.con.execute(q, part):
                libs[str(r["path"]).lower()] = r
        econ = explore.db()
        try:
            refs = explore.ref_rows(econ, [k for k, _ in hits if not k.startswith("lib:")])
        finally:
            econ.close()
        items = []
        for k, sim in hits:
            if k.startswith("lib:"):
                r = libs.get(k[4:])
                if r:
                    items.append({"kind": "lib", "id": r["id"], "filename": r["filename"], "width": r["width"], "height": r["height"],
                                  "v": thumb_ver(r["path"], r["mtime"]), "model": r["model"], "prompt": r["prompt"], "sim": round(sim, 4)})
            else:
                r = refs.get(k)
                if r:
                    items.append({"kind": "ref", "key": k, "id": r["id"], "name": r["name"], "width": r["w"], "height": r["h"],
                                  "tags": r["tags"], "url": r["url"], "eagle": r["eagle"], "src": r["src"], "sim": round(sim, 4)})
        return {"items": items}

    def refine_batch(self, data):
        """批次精修（修臉、高清化的補細節）：items 是每張圖各自的送出內容（跟精修頁送的一樣，各自照原圖的模型與參數）。
        原圖的模型一樣的排在一起，一組做完換下一組的模型（worker_client 的 pending）。
        有問題的那張不做、列在 skipped，其他照做"""
        items = data.get("items")
        if not isinstance(items, list) or not items:
            return {"error": tr("srv.h.noPick")}
        groups, skipped, warns = {}, [], []
        for it in items[:500]:
            if not isinstance(it, dict):
                continue
            plan = self.plan_run(it)
            if "error" in plan:
                skipped.append({"id": it.get("init_id"), "error": plan["error"]})
                continue
            key = json.dumps(plan["load"], sort_keys=True, ensure_ascii=False)
            groups.setdefault(key, [plan["load"], []])[1].extend(plan["jobs"])
            warns += [w for w in plan["warns"] if w not in warns]
        if not groups:
            return {"error": skipped[0]["error"] if skipped else tr("srv.h.nothingToDo"), "skipped": skipped}
        seq = 0
        for _, jobs in groups.values():
            for j in jobs:
                j["id"] = seq                       # 進度表靠編號對，整批要不一樣
                seq += 1
        (load, first), *more = groups.values()
        if not WORKER.submit(load, first, more=[tuple(g) for g in more]):
            return {"error": tr("srv.h.busy")}
        r = {"ok": True, "queued": seq, "skipped": skipped}
        if warns:
            r["warn"] = "；".join(warns)
        return r

    def file_op(self, data):
        """搬移 / 複製 / 丟進回收桶 / 刪除。回傳統計。"""
        mode = data.get("mode", "move")
        ids = [int(i) for i in data.get("ids", [])]
        if not ids:
            return {"error": tr("srv.noSelection")}
        if mode == "delete":
            # 無法復原：只有在設定裡開了才做，前端送錯了也刪不掉
            if SET.load().get("HARD_DELETE") is not True:
                return {"error": tr("srv.h.hardDeleteOff")}
            return self._delete_files(ids)

        if mode == "trash":
            dest = C.TRASH_DIR
        else:
            raw = (data.get("dest") or "").strip().strip('"')
            if not raw:
                return {"error": tr("srv.h.needDest")}
            dest = Path(raw)
            if dest.exists() and not dest.is_dir():
                return {"error": tr("srv.h.destNotDir")}
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"error": tr("srv.mkdirFail", e=e)}

        done = failed = 0
        errs = []
        moved = {}          # 舊路徑 → 新路徑，比較組要跟著改
        trashed = {}        # 丟進回收桶的：記原本的位置與索引，之後可以還原（server/trash.py）
        gone_ids = []       # 丟掉的：比較組、封面、精修的底圖要清掉（server/usage.py）
        for i in ids:
            r = self._row(i)
            if not r:
                failed += 1
                continue
            src = Path(r["path"])
            if not src.exists():
                # 檔案早就不見了，順手清掉索引
                self.con.execute("DELETE FROM images WHERE id=?", (i,))
                failed += 1
                continue

            # 同名的標籤檔一起處理，資料集才完整；圖片與 .txt 兩個名字都要沒人用（server/fileops.py）
            extra = fileops.companions(src)
            target = fileops.free_target(dest, src, extra)
            if target.resolve() == src.resolve():
                continue          # 已經在目的地

            try:
                # 一組要嘛全部到位、要嘛維持原狀；後面更新索引出錯也把檔案搬回去
                undo = fileops.transfer([(src, target)] + [(e, target.with_suffix(e.suffix)) for e in extra],
                                        "copy" if mode == "copy" else "move")
            except Exception as e:
                failed += 1
                if len(errs) < 3:
                    errs.append(f"{src.name}: {str(e)[:70]}")
                continue
            try:
                if mode == "copy":
                    pass                      # 原檔還在，索引不用動
                elif mode == "trash":
                    trashed[target.name] = {"orig": str(src), "row": {k: r[k] for k in r.keys() if k != "id"}}
                    gone_ids.append(i)
                    self.con.execute("DELETE FROM images WHERE id=?", (i,))
                else:
                    self.con.execute(
                        "UPDATE images SET path=?, folder=?, filename=? "
                        "WHERE id=?",
                        (str(target), str(target.parent), target.name, i))
                    moved[str(src)] = str(target)
                done += 1
            except Exception as e:
                undo()
                trashed.pop(target.name, None)
                moved.pop(str(src), None)
                if gone_ids and gone_ids[-1] == i:
                    gone_ids.pop()
                failed += 1
                if len(errs) < 3:
                    errs.append(f"{src.name}: {str(e)[:70]}")

        if mode == "move":
            refine.rename_inits(self.con, moved)     # 跟著這次的交易一起 commit
        self.con.commit()
        trash.record(trashed)
        cleaned = usage.forget([e["orig"] for e in trashed.values()], gone_ids) if trashed else None
        if mode == "move":
            compare.rename_paths(moved)
            recipes.rename_paths(moved)
            assets.rename_paths(moved)

        if mode in ("move", "copy"):
            def remember(ui):                        # 最近用過的目的地：讀、改、存在同一把鎖裡（state.update_ui）
                ui["dests"] = ([str(dest)] + [d for d in ui.get("dests", []) if d != str(dest)])[:8]
            update_ui(remember)

        return {"ok": True, "mode": mode, "done": done, "failed": failed,
                "errors": errs, "dest": str(dest), **({"cleaned": cleaned} if cleaned else {})}

    def _delete_files(self, ids):
        """直接刪掉圖檔（連同同名的標籤檔 .txt）與索引。檔案早就不見的，只清索引"""
        done = failed = 0
        errs = []
        gone_paths, gone_ids = [], []
        for i in ids:
            r = self._row(i)
            if not r:
                failed += 1
                continue
            src = Path(r["path"])
            existed = src.exists()
            try:
                if existed:
                    src.unlink()
            except Exception as e:
                # 圖還在（被別的程式開著、沒有權限）：索引、比較組、封面都不動
                failed += 1
                if len(errs) < 3:
                    errs.append(f"{src.name}: {str(e)[:70]}")
                continue
            # 圖已經不在了：索引與各處的關聯都要清（usage.forget）
            gone_paths.append(str(src))
            gone_ids.append(i)
            self.con.execute("DELETE FROM images WHERE id=?", (i,))
            done += 1
            txt = src.with_suffix(".txt")
            try:
                if existed and txt.exists():
                    txt.unlink()
            except Exception as e:                      # 圖已經刪了，標籤檔刪不掉只說一聲
                if len(errs) < 3:
                    errs.append(f"{txt.name}: {str(e)[:70]}")
        self.con.commit()
        cleaned = usage.forget(gone_paths, gone_ids) if gone_paths else None
        return {"ok": True, "mode": "delete", "done": done, "failed": failed, "errors": errs,
                **({"cleaned": cleaned} if cleaned else {})}

    def do_POST(self):
        i18n.set_lang(self.headers.get("X-CoCo-Lang"))
        if not self._post_authorized():
            self._discard_small_request_body()
            return self._json({"error": "forbidden"}, 403)
        content_type = (self.headers.get("Content-Type") or "").split(
            ";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._discard_small_request_body()
            return self._json({"error": tr("srv.h.badContentType")},
                              415)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return self._json({"error": tr("srv.h.badLength")}, 400)
        limit = EDIT_REQUEST_BODY if urlparse(self.path).path.startswith("/api/edit/") else MAX_REQUEST_BODY
        if n < 0 or n > limit:
            return self._json({"error": tr("srv.h.tooLarge")}, 413)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json({"error": tr("srv.h.badJson")}, 400)
        if not isinstance(data, dict):
            return self._json({"error": tr("srv.h.jsonNotObject")}, 400)
        u = urlparse(self.path)

        # 輕量安裝沒有生成引擎：在這裡就擋下，不去開工作行程再出錯（server/install.py）
        if u.path in install.FULL_ONLY and install.lite():
            return self._json({"error": tr("srv.lite.needFull")})

        if u.path == "/api/ui":
            save_ui(data)
            self._json({"ok": True})

        elif u.path == "/api/install/full":
            self._json(install.request_full())

        elif u.path == "/api/shortcut/remove":
            try:
                self._json(shortcut.remove())
            except (OSError, subprocess.SubprocessError, ValueError):
                self._json({"error": tr("srv.sc.fail")})

        elif u.path == "/api/files":
            with DB_WRITE_LOCK:
                result = self.file_op(data)
            self._json(result)

        elif u.path == "/api/edit/save":
            self._json(edit.save_project(self.con, data))

        elif u.path == "/api/edit/wmlogo":
            self._json(edit.save_logo(data) if data.get("png") else edit.clear_logo())

        elif u.path == "/api/edit/kra/export":
            self._json(kra.export_kra(self.con, data))

        elif u.path == "/api/edit/kra/import":
            with DB_WRITE_LOCK:
                self._json(kra.import_kra(self.con, data))

        elif u.path == "/api/edit/discard":
            self._json({"ok": edit.discard(str(data.get("id") or ""))})

        elif u.path == "/api/edit/export":
            with DB_WRITE_LOCK:
                self._json(edit.export(self.con, data))

        elif u.path == "/api/files/usage":
            ids = data.get("ids") or []
            if not isinstance(ids, list):
                return self._json({"error": tr("srv.badFormat")})
            self._json({"items": usage.image_uses(self.con, ids)})

        elif u.path == "/api/trash":
            # 回收桶：還原（沒有紀錄的放到 dest）、永久刪除、清空
            act, names = data.get("action"), data.get("names") or []
            if not isinstance(names, list):
                return self._json({"error": tr("srv.badFormat")})
            if act == "restore":
                with DB_WRITE_LOCK:
                    self._json(trash.restore(self.con, [str(n) for n in names], (data.get("dest") or "").strip() or None))
            elif act == "delete":
                self._json(trash.delete([str(n) for n in names]))
            elif act == "empty":
                self._json(trash.empty())
            else:
                self._json({"error": tr("srv.badAction")})

        elif u.path == "/api/gen/tokens":
            # 開著品質標籤時，數的是實際會送出的（含自動加上的）
            text, neg = data.get("prompt", ""), data.get("negative", "")
            if data.get("preset"):
                model = SET.load().get("MODEL") or ""
                text, neg, _ = presets.apply(text, neg, presets.resolve(model, model_arch(model)))
            self._json(WORKER.ask_tokens(text, neg))

        elif u.path == "/api/presets":
            self._json(presets.update(data.get("scope"), data.get("key"), data))

        elif u.path == "/api/gen/run":
            self._json(self.gen_run(data))

        elif u.path == "/api/explore/model/download":
            self._json(explore.start_download())

        elif u.path == "/api/explore/model/cancel":
            self._json(explore.cancel_download())

        elif u.path == "/api/explore/index":
            self._json(explore.start_index(C.DB_PATH, SET.load().get("REF_SOURCES") or [], self._embed))

        elif u.path == "/api/explore/ref/import":
            # 參考圖加進圖庫（精修、編輯、放進比較要用）：複製一份到 outputs/imports，原檔不動
            try:
                rid = int(str(data.get("id", "")).lstrip("r"))
            except ValueError:
                return self._json({"error": tr("srv.refNotFound")})
            self._json(explore.import_ref(rid, self.con, C.OUTPUT_DIR, DB_WRITE_LOCK))

        elif u.path == "/api/eagle/save":
            self._json(self.eagle_save(data))

        elif u.path == "/api/explore/search":
            self._json(self.explore_search(data))

        elif u.path == "/api/dataset/opts":
            self._json({"opts": dataset.set_opts(data.get("opts") or {})})

        elif u.path == "/api/dataset/library":
            # 01 資料集：加進清單、從清單移除（檔案不動）、改封面／每張重複幾次／備註
            op = data.get("op")
            if op == "add":
                self._json(dataset.lib_add(data.get("dir")))
            elif op == "remove":
                self._json(dataset.lib_remove(str(data.get("dir") or "")))
            elif op == "update":
                self._json(dataset.lib_update(str(data.get("dir") or ""),
                                              {k: data[k] for k in ("cover", "repeats", "note") if k in data}))
            else:
                self._json({"error": tr("srv.badAction")})

        elif u.path == "/api/dataset/trigger":
            root = dataset.root_of(data.get("dir"))
            self._json({"trigger": dataset.set_trigger(root, data.get("trigger"))} if root else {"error": tr("srv.noDir")})

        elif u.path == "/api/dataset/save":
            root = dataset.root_of(data.get("dir"))
            if not root:
                return self._json({"error": tr("srv.noDir")})
            if data.get("tags") is not None:
                self._json(dataset.save(root, data.get("path"), tags=str(data.get("tags"))[:20000],
                                        desc=str(data.get("desc") or "")[:20000]))
            else:
                self._json(dataset.save(root, data.get("path"), str(data.get("text") or "")[:20000]))

        elif u.path == "/api/dataset/tag":
            targets = data.get("targets")
            self._json(dataset.start_tag(data.get("dir"), targets if isinstance(targets, list) else [],
                                         WORKER.ask_tag, dataset.model_dir()))

        elif u.path == "/api/dataset/caption":
            targets = data.get("targets")
            self._json(dataset.start_caption(data.get("dir"), targets if isinstance(targets, list) else [],
                                             WORKER.ask_caption, dataset.joycaption_files()))

        elif u.path == "/api/dataset/cancel":
            self._json(dataset.cancel())

        elif u.path == "/api/dataset/bulk":
            targets = data.get("targets")
            self._json(dataset.bulk(data.get("dir"), str(data.get("op") or ""), str(data.get("tag") or ""),
                                    str(data.get("to") or ""), targets if isinstance(targets, list) else None))

        elif u.path == "/api/dataset/trash":
            targets = data.get("targets")
            self._json(dataset.trash(data.get("dir"), targets if isinstance(targets, list) else []))

        elif u.path == "/api/dataset/from_gallery":
            ids = data.get("ids")
            try:
                ids = [int(i) for i in ids][:5000] if isinstance(ids, list) else []
            except (TypeError, ValueError):
                ids = []
            self._json(dataset.from_gallery(ids, data.get("dir"), self.con))

        elif u.path == "/api/dataset/translate":
            segs = data.get("segments")
            try:
                self._json(dataset.translate([str(s)[:2000] for s in segs][:300] if isinstance(segs, list) else [],
                                             str(data.get("target") or "zh-TW")))
            except RuntimeError as e:
                self._json({"error": str(e)})

        elif u.path == "/api/train/preview":
            self._json(train.preview(data))

        elif u.path == "/api/train/start":
            self._json(train.start(data, WORKER.stop))

        elif u.path == "/api/train/stop":
            self._json(train.stop())

        elif u.path == "/api/pairs/estimate":
            self._json(pairmaker.estimate(data))

        elif u.path == "/api/pairs/start":
            self._json(pairmaker.start(data, lambda d, path: self.plan_run(d, edit_path=path), WORKER, train.busy))

        elif u.path == "/api/pairs/cancel":
            self._json(pairmaker.cancel(WORKER))

        elif u.path == "/api/train/resume":
            self._json(train.resume(data, WORKER.stop))

        elif u.path == "/api/train/progress":
            self._json(train.discard_progress(data.get("dir")) if data.get("op") == "discard" else {"progress": train.progress_list()})

        elif u.path == "/api/train/saved":
            op = data.get("op")
            self._json(train.save_config(data.get("name"), data.get("cfg") or {}) if op == "save"
                       else train.delete_config(data.get("name")) if op == "delete" else {"error": tr("srv.badAction")})

        elif u.path == "/api/dataset/model/download":
            self._json(dataset.start_download())

        elif u.path == "/api/dataset/model/cancel":
            self._json(dataset.cancel_download())

        elif u.path == "/api/queue/add":
            items = data.get("items")
            self._json(jobqueue.add(items if isinstance(items, list) else []))

        elif u.path == "/api/queue/remove":
            self._json(jobqueue.remove(str(data.get("id") or "")))

        elif u.path == "/api/queue/move":
            self._json(jobqueue.move(str(data.get("id") or ""), data.get("to")))

        elif u.path == "/api/queue/retry":
            self._json(jobqueue.retry(str(data.get("id") or "")))

        elif u.path == "/api/queue/clear":
            self._json(jobqueue.clear_finished())

        elif u.path == "/api/queue/cool":
            self._json({"ok": True, "cool": jobqueue.set_cool(data)})

        elif u.path == "/api/queue/start":
            self._json(jobqueue.start())

        elif u.path == "/api/queue/pause":
            self._json(jobqueue.pause())

        elif u.path == "/api/refine/batch":
            self._json(self.refine_batch(data))

        elif u.path == "/api/gen/cancel":
            self._json({"ok": True, "dropped": WORKER.cancel()})

        elif u.path == "/api/gen/stop":
            WORKER.stop()
            self._json({"ok": True})

        elif u.path == "/api/reindex":
            self._json(run_index(quiet=True))

        elif u.path == "/api/settings":
            self._json(update_settings(data))

        elif u.path == "/api/wildcard":
            self._json(wc_write(data.get("file", ""), data.get("text", ""), create=data.get("create") is True))

        elif u.path == "/api/vars":
            self._json(vars_apply(data))

        elif u.path == "/api/tags/lookup":
            names = data.get("names")
            self._json({"items": tag_lookup(names if isinstance(names, list) else [])})

        elif u.path == "/api/mytags/tag":
            self._json(usertags.save_tag(data))
        elif u.path == "/api/mytags/tag/delete":
            self._json(usertags.delete_tag(data.get("name")))
        elif u.path == "/api/mytags/group":
            self._json(usertags.save_group(data))
        elif u.path == "/api/mytags/group/delete":
            self._json(usertags.delete_group(data.get("id")))
        elif u.path == "/api/mytags/used":
            names = data.get("names")
            self._json(usertags.used(names if isinstance(names, list) else [], data.get("group")))
        elif u.path == "/api/mytags/accept":
            names = data.get("names")
            self._json(usertags.accept(names if isinstance(names, list) else [], data.get("cat") or ""))
        elif u.path == "/api/mytags/ignore":
            names = data.get("names")
            self._json(usertags.ignore(names if isinstance(names, list) else [], bool(data.get("undo"))))

        elif u.path == "/api/expand":
            try:
                res = TPL.expand(
                    data.get("template", ""),
                    count=int(data.get("count") or 1),
                    seed=(None if data.get("seed") in ("", None)
                          else int(data["seed"])),
                    wildcard_dir=C.WILDCARD_DIR,
                    vars_file=C.VARS_FILE,
                    lock_random=bool(data.get("lock", True)),
                    limit=3000)
                self._json(res)
            except ValueError as e:
                self._json({"error": str(e)})
            except Exception as e:
                self._json({"error": tr("srv.h.expandFail", e=e)})

        elif u.path == "/api/save_prompts":
            try:
                items = data.get("prompts", [])
                fmt = data.get("format", "plain")
                seed = data.get("seed")
                if fmt == "a1111":
                    tail = "" if seed in ("", None) else f" --seed {int(seed)}"
                    lines = [f'--prompt "{t}"{tail}' for t in items]
                else:
                    lines = list(items)
                out = ROOT / "prompts_out.txt"
                out.write_text("\n".join(lines), encoding="utf-8")
                self._json({"ok": True, "path": str(out), "n": len(lines)})
            except Exception as e:
                self._json({"error": str(e)})

        elif u.path == "/api/res/pick":
            kind = data.get("kind", "model")
            path = (data.get("path") or "").strip()
            with SETTINGS_APPLY_LOCK:
                cur = SET.load()
                if kind == "model":
                    SET.save({"MODEL": path, **arch_switch(cur, path)})
                else:
                    ls = list(cur.get("LORAS") or [])
                    hit = next((l for l in ls if l["path"] == path), None)
                    if hit:
                        ls = [l for l in ls if l["path"] != path]
                    else:
                        ls.append({"path": path,
                                   "weight": float(
                                       data.get("weight", 1.0))})
                    SET.save({"LORAS": ls})
                importlib.reload(C)
                current = settings_out()
            self._json({"ok": True, "settings": current})

        elif u.path == "/api/mkdir":
            try:
                d = Path(data.get("parent", "")) / str(data.get("name", "")).strip()
                d.mkdir(parents=True, exist_ok=True)
                self._json({"ok": True, "path": str(d)})
            except Exception as e:
                self._json({"error": str(e)})

        elif u.path == "/api/assets/meta":
            kind = data.get("kind", "")
            name = str(data.get("name") or "").strip()
            if kind not in KINDS or not name:
                return self._json({"error": tr("srv.h.needKindName")}, 400)
            trig = data.get("triggers")
            meta = assets.update_note(kind, name, fav=data.get("fav"), note=data.get("note"),
                                      touch=bool(data.get("touch")),
                                      triggers=trig if isinstance(trig, list) else None,
                                      reset_triggers=bool(data.get("resetTriggers")))
            self._json({"ok": True, "meta": meta})

        elif u.path == "/api/refine/detect":
            # 修臉的找臉：底圖用圖庫的 id 指定；偵測模型用檔名挑（沒給就用最好的那個）
            src = refine.init_source(self.con, data.get("id"))
            if "error" in src:
                return self._json(src, 404)
            models = detectors()
            if not models:
                return self._json({"error": tr("srv.h.noDetector"), "models": []})
            want = str(data.get("model") or "")
            pick = next((m for m in models if m["name"] == want), models[0])
            got = WORKER.ask_detect(src["path"], pick["path"])
            if not got.get("ready"):
                return self._json({"error": got.get("why") or tr("srv.h.detectFail"), "models": [m["name"] for m in models]})
            self._json({"faces": got.get("faces", []), "size": got.get("size"), "model": pick["name"],
                        "models": [m["name"] for m in models]})

        elif u.path == "/api/controlnet/download":
            self._json(controlnet.start_download(str(data.get("key") or "")))

        elif u.path == "/api/modeldl/download":
            self._json(modeldl.start(str(data.get("job") or "")))

        elif u.path == "/api/modeldl/cancel":
            self._json(modeldl.cancel(str(data.get("job") or "")))

        elif u.path == "/api/controlnet/cancel":
            self._json(controlnet.cancel_download(str(data.get("key") or "")))

        elif u.path == "/api/controlnet/remove":
            self._json(controlnet.remove(str(data.get("key") or "")))

        elif u.path == "/api/controlnet/upload":
            self._json(controlnet.save_upload(data.get("png")))

        elif u.path == "/api/lora/resolve":
            # 送到生成區：圖裡的 LoRA 名字找不到時，用圖裡記的雜湊值找改過名的（server/lorahash.py）
            files = [(Path(it["path"]).stem, it["path"]) for it in res_scan("lora")]
            wanted = data.get("hashes") if isinstance(data.get("hashes"), dict) else {}
            self._json(lorahash.resolve({str(k)[:200]: str(v)[:64] for k, v in list(wanted.items())[:50]}, files))

        elif u.path == "/api/controlnet/restore":
            self._json(self.controlnet_restore(data.get("units")))

        elif u.path == "/api/controlnet/preview":
            self._json(self.controlnet_preview(data))

        elif u.path == "/api/rembg/download":
            self._json(rembg.start_download(str(data.get("key") or "")))

        elif u.path == "/api/rembg/cancel":
            self._json(rembg.cancel_download(str(data.get("key") or "")))

        elif u.path == "/api/rembg/remove":
            self._json(rembg.remove(str(data.get("key") or "")))

        elif u.path == "/api/refine/upscale":
            # 高清化的「只放大」：工作行程放大，這裡存成新圖（精修的新版本）
            src = refine.init_source(self.con, data.get("id"))
            if "error" in src:
                return self._json(src, 404)
            size = upscalers.target_size(src["width"], src["height"], data.get("scale"))
            if not size:
                return self._json({"error": tr("srv.h.upTooBig", n=upscalers.MAX_SIDE)})
            name = str(data.get("model") or "")
            model = upscalers.find(name) if name else None
            if name and not model:
                return self._json({"error": tr("srv.h.upGone")})
            if not model and float(data.get("scale")) == 1:
                return self._json({"error": tr("srv.h.up1x")})
            tmp =Path(tempfile.gettempdir()) / f"coco-upscale-{secrets.token_hex(6)}.png"
            try:
                got = WORKER.ask_upscale(src["path"], model["path"] if model else "", tmp, size)
                if not got.get("ready"):
                    return self._json({"error": got.get("why") or tr("srv.h.upFail")})
                from PIL import Image
                with Image.open(tmp) as im:
                    im.load()
                    out = upscalers.save_result(src["path"], im, float(data.get("scale")), name)
            finally:
                tmp.unlink(missing_ok=True)
            try:
                import gallery_index as GI
            except ImportError:
                from core import gallery_index as GI
            with DB_WRITE_LOCK:
                GI.index_file(self.con, out)
                self.con.commit()
            r = self.con.execute("SELECT id FROM images WHERE lower(path) = lower(?)", (str(out),)).fetchone()
            self._json({"ok": True, "id": r[0] if r else None, "path": str(out)})

        elif u.path == "/api/refine/rembg":
            # 去背：工作行程算遮罩，這裡套到原圖上存成新圖（精修的新版本）
            src = refine.init_source(self.con, data.get("id"))
            if "error" in src:
                return self._json(src, 404)
            model = rembg.find_model(str(data.get("model") or ""))
            if not model:
                return self._json({"error": tr("srv.h.rbGone")})
            # IS-Net 的前處理可以換；BiRefNet 只有官方那一種
            prep = data.get("prep") if model["arch"] == "isnet" and data.get("prep") in rembg.PREPS else model["prep"]
            tmp = Path(tempfile.gettempdir()) / f"coco-rembg-{secrets.token_hex(6)}.png"
            try:
                got = WORKER.ask_rembg(src["path"], model["path"], tmp, model["arch"], prep)
                if not got.get("ready"):
                    return self._json({"error": got.get("why") or tr("srv.h.rbFail")})
                from PIL import Image
                with Image.open(tmp) as m:
                    mask = m.convert("L")
                out = rembg.save_result(src["path"], mask, model["known"] or model["id"], str(data.get("edge") or "soft"), prep=prep)
            finally:
                tmp.unlink(missing_ok=True)
            try:
                import gallery_index as GI
            except ImportError:
                from core import gallery_index as GI
            with DB_WRITE_LOCK:
                GI.index_file(self.con, out)
                self.con.commit()
            r = self.con.execute("SELECT id FROM images WHERE lower(path) = lower(?)", (str(out),)).fetchone()
            self._json({"ok": True, "id": r[0] if r else None, "path": str(out)})

        elif u.path == "/api/res/kind":
            r = set_kind(str(data.get("path") or ""), str(data.get("kind") or ""))
            self._json(r, 400 if "error" in r else 200)

        elif u.path == "/api/assets/cover":
            # 自訂封面：id＝圖庫裡的圖、data＝上傳的圖（data URL），都沒給就用回預設
            kind = data.get("kind", "")
            name = str(data.get("name") or "").strip()
            if kind not in KINDS or not name:
                return self._json({"error": tr("srv.h.needKindName")}, 400)
            if data.get("id") not in (None, ""):
                try:
                    r = self._row(int(data["id"]))
                except Exception:
                    r = None
                if not r:
                    return self._json({"error": tr("srv.imgNotFound")}, 404)
                meta = assets.set_cover(kind, name, image=r["path"])
            elif data.get("data"):
                up = assets.save_upload(data["data"])
                if "error" in up:
                    return self._json(up, 400)
                meta = assets.set_cover(kind, name, file=up["file"])
            else:
                meta = assets.set_cover(kind, name)
            self._json({"ok": True, "meta": meta})

        elif u.path == "/api/recipes/save":
            if data.get("data") is not None and not isinstance(data.get("data"), dict):
                return self._json({"error": tr("srv.h.dataNotObject")}, 400)
            rid = recipes.save_recipe(self.con, str(data.get("id") or ""), data.get("name"),
                                      data.get("data"), data.get("cover"), data.get("note"))
            self._json({"ok": True, "id": rid})

        elif u.path == "/api/recipes/use":
            ok = recipes.touch(str(data.get("id") or ""))
            self._json({"ok": True} if ok else {"error": tr("srv.h.recipeNotFound")}, 200 if ok else 404)

        elif u.path == "/api/recipes/delete":
            ok = recipes.delete_recipe(str(data.get("id") or ""))
            self._json({"ok": True} if ok else {"error": tr("srv.h.recipeNotFound")}, 200 if ok else 404)

        elif u.path == "/api/compare/add":
            ids = [int(i) for i in data.get("ids", []) if str(i).lstrip("-").isdigit()]
            paths = [str(p) for p in data.get("paths") or [] if isinstance(p, str)]
            if not ids and not paths:
                return self._json({"error": tr("srv.noSelection")}, 400)
            self._json({"ok": True, **compare.add_images(self.con, data.get("id") or "", ids,
                                                         data.get("name") or "", paths=paths,
                                                         auto=str(data.get("auto") or ""))})

        elif u.path == "/api/compare/update":
            items = data.get("items")
            ok = compare.update_set(str(data.get("id") or ""), name=data.get("name"),
                                    items=items if isinstance(items, list) else None,
                                    round=data.get("round"), done=data.get("done"), archived=data.get("archived"))
            self._json({"ok": True} if ok else {"error": tr("srv.setNotFound")}, 200 if ok else 404)

        elif u.path == "/api/compare/bulk":
            ids = data.get("ids") or []
            n = compare.bulk([str(i) for i in ids] if isinstance(ids, list) else [], str(data.get("action") or ""))
            self._json({"ok": True, "n": n} if n is not None else {"error": tr("srv.badAction")})

        elif u.path == "/api/compare/delete":
            ok = compare.delete_set(str(data.get("id") or ""))
            self._json({"ok": True} if ok else {"error": tr("srv.setNotFound")}, 200 if ok else 404)

        elif u.path == "/api/res/open_home":
            # 打開 CoCo 自己的模型資料夾（或其中一格）：只開 models 底下的，別的路徑不接受
            from . import modelhome
            sub = str(data.get("sub") or "")
            names = {n for ns in modelhome.SUBDIRS.values() for n in ns}
            if sub and sub not in names:
                self._json({"error": tr("srv.h.badFolder")})
            else:
                d = modelhome.home() / sub if sub else modelhome.home()
                d.mkdir(parents=True, exist_ok=True)
                open_folder(d)
                self._json({"ok": True})

        elif u.path == "/api/reveal":
            try:
                r = self._row(int(data.get("id", 0)))
                reveal(r["path"])
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)})

        elif u.path == "/api/rate_bulk":
            ids = [int(i) for i in data.get("ids", [])]
            r = int(data.get("rating", 0))
            with DB_WRITE_LOCK:
                self.con.executemany("UPDATE images SET rating=? WHERE id=?",
                                     [(r, i) for i in ids])
                self.con.commit()
            self._json({"ok": True, "count": len(ids)})

        elif u.path == "/api/rate":
            with DB_WRITE_LOCK:
                self.con.execute("UPDATE images SET rating=? WHERE id=?",
                                 (int(data.get("rating", 0)),
                                  int(data["id"])))
                self.con.commit()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


# 排程生成（server/jobqueue.py）用同一套規劃：每一項就是生成台送出的內容
jobqueue.PLAN = lambda data: Handler.__new__(Handler).plan_run(data)
