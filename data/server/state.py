# -*- coding: utf-8 -*-
"""共用的路徑、設定、鎖與資料庫連線。

從原本的 tools/studio.py 拆出來，內容未改。
"""

import json
import secrets
import sqlite3
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
import config as C
import settings as SET


def _version():
    """CoCo 的版本（git 的 commit 前 7 碼）：寫進圖的參數（Version: CoCo …），引擎改過之後看得出是哪一版畫的"""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    try:
        git = ROOT.parent / ".git" if not (ROOT / ".git").exists() else ROOT / ".git"     # 程式在 data 裡，repo 在上一層
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            head = (git / head[4:].strip()).read_text(encoding="utf-8").strip()
        return head[:7]
    except OSError:
        return ""


VERSION = _version()


def _app_version():
    """發布的版本號（data\\VERSION，例如 0.1）：⚙ 設定 → 關於、主控台開頭顯示；發布工具 tools/release.py 也讀這個檔。
    公開版沒有 .git，上面的 commit 碼是空的，使用者回報問題時看這個。讀不到就空字串（不顯示）"""
    try:
        return (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


APP_VERSION = _app_version()
import template as TPL

HERE = ROOT
DB_PATH = C.DB_PATH
THUMB_DIR = C.THUMB_DIR
UI_FILE = C.UI_FILE
UI_DEFAULT = {"fs": 1.0, "colw": 260, "genw": 190, "recent": [],
              "order": "just", "gorder": "just", "genzoom": True,
              "extw": 320, "parw": 380, "midtop": 42,
              "dests": [],
              # 介面：語言空字串＝跟著瀏覽器；配色與深淺見 web/js/theme.js
              "lang": "", "theme": "cherry", "mode": "dark",
              # 圖庫空間的「滑過放大」；工作室的牆沿用舊介面生成頁的 genzoom
              "libzoom": False,
              # 生成台目前套用的 Recipe（id）；空字串＝沒有
              "recipe": "",
              # 新介面的介面大小（整頁縮放）；舊介面的 fs 只放大文字，兩者意思不同，分開存
              "scale": 1.0,
              # 生成台的提示詞用標籤模式（False＝純文字）
              "chips": False,
              # 看大圖的提示詞用標籤模式顯示（點一下複製那個標籤）
              "vchips": False,
              # 精修頁：目前的底圖（圖庫 id，0＝還沒選）與修改幅度
              "rsrc": 0, "rstrength": 0.5,
              # 精修頁：方式（whole／inpaint）、局部重繪的修改幅度、筆刷、工具、範圍、邊緣柔化
              "rmode": "whole", "rinstrength": 0.65, "rrmstrength": 0.85, "rbrush": 48, "rtool": "brush", "rarea": "whole", "rblur": 8,
              # 精修頁：遮罩工具是塗還是擦（每種工具共用）、局部重繪的起點（original／fill＝先抹掉，移除東西用）
              "rerase": False, "rfill": "original",
              # 精修頁：塗色的顏色、魔術棒的容許度
              "rcolor": "#d04848", "rtol": 32,
              # 精修頁：修臉的修改幅度、偵測模型（檔名，空的＝用最好的那個）
              "rfstrength": 0.4, "rfmodel": "",
              # 精修頁：照原圖的模型與參數（找不到原圖的模型時自動用生成台的）
              "rsrcp": True,
              # 精修頁：去背的模型（id，空的＝第一個）、邊緣、IS-Net 的前處理（每個模型各自記）
              "rbmodel": "", "rbedge": "soft", "rbprep": {},
              # 精修頁：高清化的放大模型（null＝第一個，空字串＝一般放大）、倍數、補細節、補細節的修改幅度
              "rupmodel": None, "rupscale": 2, "rupdetail": True, "rustrength": 0.3,
              # 精修頁：擴圖的範圍（方式、比例、原圖放哪、自訂四邊）與修改幅度
              "rout": {}, "rostrength": 0.9,
              # 精修頁：用 Qwen 修圖的模型、大小（src／mp1）、步數、CFG、取樣器（沒選過的用官方建議）
              "redit": {},
              # 批次精修：上次選的方式（face／upscale／rembg）
              "rbatch": "face",
              # 生成台：使用者自己存的常用尺寸 [[寬, 高], …]
              "sizes": [],
              # 探索：找的範圍（all／lib＝我的圖／ref＝參考圖）
              "xscope": "all",
              # 存回 Eagle：每個檔案庫上次選的資料夾 {檔案庫路徑: 資料夾 id}、標籤、要不要把生成參數寫進註解
              "eaglefolder": {}, "eaglenote": True, "rcnrepaint": False,
              # 比較頁：排法（空字串＝還沒選過，依張數決定）、格狀的大小、要不要排成 XY 對照表
              "cmpview": "", "cmpsize": "m", "cmpxy": True,
              # 多視窗的 ⧉：auto＝偵測到第二個螢幕才出現；on / off 手動（web/js/sync.js）
              "multiwin": "auto",
              # 生成台：一批生完（2 張以上）自動送去比較
              "autocmp": False,
              # 比較組：自動建的組幾天沒打開就封存（0＝不封存）
              "cmpautoarch": 0,
              # 06 輸出：正在編輯的底圖（圖庫 id），輸出的檔案類型、品質、目的地（library／folder）、資料夾、
              # 匯出到資料夾時要不要保留生成參數（存進圖庫一律保留）
              "esrc": 0, "eformat": "png", "equality": 92, "edest": "library", "efolder": "", "ekeep": True,
              # 06 輸出：筆刷（顏色、大小、不透明度、硬度）、模糊半徑與銳利強度、選取（形狀、魔術棒、擴張收縮的幅度）、
              # 仿製印章（對齊、取樣）
              "ebrush": {}, "efx": {}, "esel": {}, "eclone": {},
              # 文字工具上次用的設定、浮水印的設定、批次輸出的尺寸與要不要加浮水印
              "etext": {}, "ewm": {}, "ebatch": {}, "collage": {},
              # 存 .kra 的資料夾
              "ekradir": "",
              # 工作室最後開的那一步（頂端的「工作室」連結指到這裡）
              "laststep": "generate",
              # 訓練空間的資料集：最後開的資料夾、描述換圖時自動翻譯、左欄收起來
              "dsdir": "", "dstrans": False, "dsfold": False,
              # 訓練空間最後開的那一步（頂端的「訓練」連結指到這裡）
              "trainstep": "datasets"}


def load_ui():
    cfg = dict(UI_DEFAULT)
    if UI_FILE.exists():
        try:
            cfg.update(json.loads(UI_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


_UI_LOCK = threading.RLock()


def save_ui(cfg):
    """cfg 只要帶改了的欄位（前端都是這樣送），沒帶的照舊。讀、合併、寫在同一把鎖裡、原子取代檔案：
    兩個視窗同時存不同的欄位，不會被對方讀到的舊值蓋回去"""
    with _UI_LOCK:
        old = load_ui()
        keep = {k: cfg.get(k, old.get(k, v)) for k, v in UI_DEFAULT.items()}
        tmp = UI_FILE.with_name(UI_FILE.name + ".tmp")
        try:
            tmp.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(UI_FILE)
        finally:
            tmp.unlink(missing_ok=True)


def update_ui(fn):
    """要看現在的值才知道怎麼改的（清單加一筆、是這張才清掉）：fn(ui) 就地改，回傳 False 就不存；整段在鎖裡。
    自己 load_ui 改完再整份 save_ui 的話，中間別的請求存的欄位會被這份舊的蓋回去"""
    with _UI_LOCK:
        ui = load_ui()
        if fn(ui) is not False:
            save_ui(ui)
APP_PORT = C.GALLERY_PORT
PAGE_SIZE = C.PAGE_SIZE
THUMB_W = C.THUMB_WIDTH
API_TOKEN = secrets.token_urlsafe(32)
# 模板一次最多可存 3000 則提示詞，1 MiB 會截掉合法批次；仍需有明確上限，
# 避免 localhost 被錯誤請求一次吃進無界記憶體。
MAX_REQUEST_BODY = 8 * 1024 * 1024
# 06 輸出的請求（圖層、合併的圖、.kra）一次可能好幾張大 PNG：放寬到 1 GB
EDIT_REQUEST_BODY = 1024 * 1024 * 1024

_RES_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


def res_stem(name):
    """模型或 LoRA 名稱的比對用鍵：去掉資料夾與副檔名、轉小寫

    圖的中繼資料有的寫短名、有的寫完整路徑、有的帶副檔名
    （anima_baseV10 與 C:/…/anima_baseV10 其實是同一個），比對前先統一。
    """
    s = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    low = s.lower()
    for ext in _RES_EXT:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def db():
    # Handler instance 與 request 是一對一；連線不跨 thread 共用。
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    # 查詢裡用 coco_stem(model) 比對模型與 LoRA（見 library.build_query）
    con.create_function("coco_stem", 1, res_stem, deterministic=True)
    return con


INDEX_STATE = {"running": False, "msg": "", "at": 0}
INDEX_LOCK = threading.Lock()
DB_WRITE_LOCK = threading.RLock()
SETTINGS_APPLY_LOCK = threading.RLock()
