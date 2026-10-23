# -*- coding: utf-8 -*-
"""桌面與開始功能表上開 CoCo 的捷徑（CoCo.bat 的 :SHORTCUT 建的）：⚙ 設定 →「其他」的「移除捷徑」。

不再用 CoCo 時先移除捷徑、再刪掉整個 CoCo 資料夾，電腦上就不留東西（pip 的下載快取放在 data\\pip-cache、
裝好就刪，見 CoCo.bat）。刪資料夾時 Windows 不會通知任何程式，捷徑只能先這樣拿掉。

只動開這份 CoCo 的捷徑：名稱一樣、目標是這份的 exe（沒有 exe 的舊安裝開 data\\CoCo.bat）。
另一份 CoCo 的捷徑（例如另一個資料夾的測試版）不碰；目標已經不在的 CoCo 捷徑（搬走 CoCo 資料夾後留下的）一起拿掉。
捷徑是 CoCo 自己建的，直接刪、不丟資源回收筒；要再建：刪掉 venv\\.coco-shortcut 再開 CoCo（README）。
"""

import ctypes
import json
import os
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "CoCo Image Studio.lnk"                   # 跟 CoCo.bat 建的一樣
# CoCo.bat 用 [Environment]::GetFolderPath('Desktop')、('Programs')，對應的 Known Folder：
# 桌面（使用者可能把它搬到別的磁碟）、開始功能表的「程式集」
FOLDERS = ("B4BFCC3A-DB2C-424C-B029-7FE99A87C641", "A77F5D77-2E2B-44C3-A6A2-ABA601054A51")
# 讀捷徑的目標：WScript.Shell，跟 CoCo.bat 建捷徑用的同一個。路徑從環境變數進來（| 不會出現在 Windows 路徑裡），
# 輸出改成 UTF-8：預設是主控台的字碼頁，中文路徑會變成問號
READ = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; $w = New-Object -ComObject WScript.Shell; "
        "$r = @($env:COCO_SC_LINKS -split '\\|' | ForEach-Object { $w.CreateShortcut($_).TargetPath }); "
        "ConvertTo-Json -InputObject $r -Compress")


def _known_folder(guid):
    """SHGetKnownFolderPath；拿不到（例如沒有這個資料夾）回 None"""
    gid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(guid).bytes_le)
    buf = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(gid), 0, None, ctypes.byref(buf)) != 0:
        return None
    try:
        return Path(buf.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(buf)


def folders():
    """建捷徑的兩個地方"""
    if os.name != "nt":
        return []
    return [p for p in map(_known_folder, FOLDERS) if p]


def _targets(links):
    """每個捷徑的目標（順序跟 links 一樣）"""
    if not links:
        return []
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", READ],
                       env={**os.environ, "COCO_SC_LINKS": "|".join(map(str, links))},
                       capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode:
        raise OSError(r.stderr.decode("utf-8", "replace").strip())
    out = json.loads(r.stdout.decode("utf-8-sig") or "[]")
    return [str(t or "") for t in (out if isinstance(out, list) else [out])]


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def ours(target):
    """這個目標是不是這份 CoCo（或已經不在的 CoCo）。讀不到目標（空字串）的檔名是空的，不算"""
    if any(_same(target, p) for p in (ROOT.parent / "CoCo Image Studio.exe", ROOT / "CoCo.bat")):
        return True
    # 搬走或刪掉的那份留下的：目標不在了、檔名是 CoCo 的。在的話是另一份 CoCo，不碰
    return Path(target).name.lower() in ("coco image studio.exe", "coco.bat") and not os.path.exists(target)


def find():
    """開這份 CoCo 的捷徑"""
    links = [d / NAME for d in folders() if (d / NAME).is_file()]
    return [p for p, t in zip(links, _targets(links)) if ours(t)]


def status():
    return {"found": len(find())}


def remove():
    """移除這份 CoCo 的捷徑；回傳移除了幾個"""
    n = 0
    for p in find():
        p.unlink()
        n += 1
    return {"removed": n, "found": 0}
