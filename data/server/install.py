# -*- coding: utf-8 -*-
"""安裝方式：完整安裝或輕量安裝。

- 完整：全部功能（生成、精修、訓練要 NVIDIA 顯示卡）。CUDA 版 PyTorch＋requirements-lock.txt，約 4 GB。
- 輕量：圖庫、探索、比較、重組、輸出、資源庫，以及生成台的提示詞與標籤（只有生成要完整安裝）。CPU 版 PyTorch＋requirements-lite.txt（版本照 requirements-lock.txt），
  沒有 diffusers 這些生成引擎的套件。探索的特徵用 CPU 算（2026-10-15 實測 Ryzen 7 9700X：每秒 12.2 張，顯示卡 47 張；
  用文字找圖一次約 0.06 秒）。

CoCo.bat 第一次安裝時問，選擇記在 install_mode.txt（full／lite；這個檔出現以前裝的一律當完整）。
目前實際裝的是哪種看 diffusers 在不在：檔案寫 full 但還沒補裝時，實際還是輕量。
介面按「改成完整安裝」只把檔案改成 full，下次用 CoCo.bat 開啟時補裝（torch 不是 CUDA 版就整個重裝）。
完整改回輕量不做：要重建整個 venv（使用者決定的，2026-10-15）。
"""

import importlib.util
import shutil
from pathlib import Path

from i18n import tr

ROOT = Path(__file__).resolve().parent.parent
MODE_FILE = ROOT / "install_mode.txt"

# 輕量安裝擋掉的 POST（要用生成引擎、訓練，或下載只有它們用得到的模型）。
# 數 token、找相似的特徵在輕量也能用（transformers＋CPU 版 PyTorch）；排程只擋開始，加進排程不會跑
FULL_ONLY = frozenset({
    "/api/gen/run", "/api/queue/start",
    "/api/refine/batch", "/api/refine/detect", "/api/refine/rembg", "/api/refine/upscale",
    "/api/rembg/download", "/api/rembg/remove", "/api/controlnet/download", "/api/controlnet/preview",
    "/api/train/start", "/api/train/resume", "/api/pairs/start",
    "/api/dataset/tag", "/api/dataset/caption", "/api/dataset/model/download",
})


def lite():
    """目前裝的是輕量（沒有生成引擎）"""
    return importlib.util.find_spec("diffusers") is None


def wanted():
    """install_mode.txt 寫的（full、lite；沒有這個檔是 None）"""
    try:
        return MODE_FILE.read_text(encoding="utf-8").strip().lower() or None
    except OSError:
        return None


def has_nvidia():
    """裝了 NVIDIA 驅動就有 nvidia-smi（在 System32）；CoCo.bat 第一次安裝時也是這樣判斷"""
    return shutil.which("nvidia-smi") is not None


def status():
    """lite：目前是輕量；pending：已經改成完整、下次開啟時補裝；gpu：有沒有 NVIDIA 顯示卡（只在輕量時查）"""
    lt = lite()
    return {"lite": lt, "pending": lt and wanted() == "full", "gpu": True if not lt else has_nvidia()}


def request_full():
    """改成完整安裝：下次用 CoCo.bat 開啟時補裝"""
    if not lite():
        return status()
    if not has_nvidia():
        return {**status(), "error": tr("srv.lite.noGpu")}
    MODE_FILE.write_bytes(b"full\r\n")          # CoCo.bat 的 echo 寫出來也是 CRLF；set /p 讀第一行
    return status()
