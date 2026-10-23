# -*- coding: utf-8 -*-
"""啟動：更新圖庫索引、做資料遷移，然後開 HTTP 服務。
"""

import os
import threading
import webbrowser
from pathlib import Path

from .handler import Handler, Server
from .library import run_index
from .resources import model_arch
from .state import APP_PORT, APP_VERSION, DB_PATH, SET, C, db, load_ui
from .userdata import prepare_files
from .worker_client import WORKER

import i18n
from i18n import tr


def startup_lang():
    """瀏覽器還沒連上之前（主控台的訊息）用哪個語言：介面存的語言；「跟著瀏覽器」時看 Windows 的顯示語言"""
    pref = load_ui().get("lang") or ""
    if not pref:
        try:
            import ctypes
            # LANGID 的低 10 位元是主要語言，0x04＝中文
            pref = "zh-TW" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x04 else "en"
        except Exception:
            pref = ""
    i18n.set_lang(pref)


def ensure_default_output():
    """CoCo 預設的輸出資料夾（data\\outputs）要第一次生成才會建出來：全新安裝一啟動掃圖庫就印
    「[!] 資料夾不存在」，新使用者第一眼看到像是出錯。只建預設的那個、而且它在圖庫掃描範圍裡才建；
    使用者自己加的資料夾不存在時照樣提醒（多半是外接碟沒插，不能自動建一個空的）"""
    key = lambda p: os.path.normcase(os.path.abspath(str(p)))
    out = Path(C.OUTPUT_DIR_DEFAULT)
    if key(out) not in {key(d) for d in C.SCAN_DIRS}:
        return
    try:
        out.mkdir(exist_ok=True)
    except OSError:
        pass         # 建不了就跟以前一樣，掃描時印提醒


def title_line():
    """主控台開頭的標題，後面接版本號：回報問題時貼這段就知道是哪一版"""
    return "  " + tr("srv.app.title") + (f"  v{APP_VERSION}" if APP_VERSION else "")


def main():
    import certs
    certs.use_system_certs()     # 防毒軟體掃 HTTPS 時 huggingface_hub 才下載得到（core/certs.py）
    startup_lang()
    print("=" * 52)
    print(title_line())
    print("=" * 52)
    # CoCo 自己的模型資料夾：建好分格、把舊位置的檔案搬進來（先做，後面的掃描才看得到）
    try:
        from . import modelhome
        modelhome.ensure()
        modelhome.migrate()
    except Exception as e:
        print("  [!] " + tr("srv.app.modelHomeErr", e=e))
    try:
        import gallery_index as GI
        ensure_default_output()
        for d in GI.SCAN_DIRS:
            print("  " + tr("srv.app.scan", d=d))
        print("\n  " + tr("srv.app.indexing"))
        r = run_index()
        if r.get("error"):
            print(f"  [!] {r['error']}")
        else:
            print(f"  {r['msg']}")
    except Exception as e:
        print("  [!] " + tr("srv.app.indexErr", e=e))

    if not DB_PATH.exists():
        print("\n[X] " + tr("srv.app.noDb", path=DB_PATH) + "\n")
        return

    prepare_files()          # 變數與萬用字元：舊的變數檔搬過來、第一次啟動放進範例（不蓋掉已經有的）
    moved = SET.migrate_overrides(model_arch)
    if moved:
        arch, keys = moved
        print("  " + tr("srv.app.overrides", arch=arch or tr("srv.app.unknownArch"), keys=i18n.join(keys)))
    con = db()
    try:
        n = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    finally:
        con.close()
    srv = Server(("127.0.0.1", APP_PORT), Handler)
    url = f"http://127.0.0.1:{APP_PORT}"
    print("  " + tr("srv.app.started", url=url))
    print("  " + tr("srv.app.count", n=n))
    print("  " + tr("srv.app.closeHint"))
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        WORKER.stop()

