# -*- coding: utf-8 -*-
"""搬移、複製圖片（連同同名的標籤檔 .txt）：一組檔案要嘛全部到位，要嘛全部維持原狀（審查紀錄 R16）。

以前圖片和 .txt 分兩步搬，第二步失敗會留下半套：圖片已經在新的地方，索引還指著舊的（圖庫找不到這張），
.txt 留在原地。目的地剛好有同名的 .txt 也會被蓋掉。現在：
  - 挑名字時圖片和 .txt 兩個名字都要沒人用（重名就加 _1、_2…）
  - 一組裡任何一個失敗，已經搬過去的搬回來、複製出來的刪掉，再把錯誤丟出去
  - 呼叫的人更新索引失敗時，用 undo() 把檔案搬回去
不會蓋掉任何已經存在的檔案。
"""

import shutil

from i18n import tr
from pathlib import Path


def companions(src):
    """跟著圖片一起走的檔案：同名的 .txt（訓練用的標籤檔）"""
    txt = Path(src).with_suffix(".txt")
    return [txt] if txt.is_file() else []


def free_name(dest, name, extra=()):
    """dest 裡可以用的名字：圖片跟要一起走的檔案（同一個名字、換副檔名）都沒人用，重名就加 _1、_2…"""
    p = Path(name)
    target = Path(dest) / p.name
    n = 1
    while target.exists() or any(target.with_suffix(Path(e).suffix).exists() for e in extra):
        target = Path(dest) / f"{p.stem}_{n}{p.suffix}"
        n += 1
    return target


def free_target(dest, src, extra=()):
    """搬 src 到 dest 要用的名字；已經就在目的地的回傳原路徑（呼叫的人略過）"""
    src = Path(src)
    target = Path(dest) / src.name
    if target.exists() and target.resolve() == src.resolve():
        return target
    return free_name(dest, src.name, extra)


def _move(a, b):
    if Path(b).exists():
        raise FileExistsError(tr("srv.fo.exists", name=Path(b).name))
    shutil.move(str(a), str(b))


def _copy(a, b):
    if Path(b).exists():
        raise FileExistsError(tr("srv.fo.exists", name=Path(b).name))
    shutil.copy2(str(a), str(b))


def transfer(pairs, mode="move"):
    """[(來源, 目的地)…] 一起搬或複製。失敗時復原已經做的、再丟出錯誤；成功回傳 undo（呼叫的人後續出錯時用）"""
    done = []

    def undo():
        for a, b in reversed(done):
            try:
                if mode == "copy":
                    Path(b).unlink(missing_ok=True)
                else:
                    _move(b, a)
            except Exception:
                pass                       # 盡力復原；原本的錯誤比較重要
        done.clear()

    try:
        for a, b in pairs:
            (_copy if mode == "copy" else _move)(a, b)
            done.append((a, b))
    except Exception:
        undo()
        raise
    return undo


def recycle(paths):
    """丟進 Windows 的資源回收筒（可以還原），不是永久刪除。不在 Windows 上丟出錯誤，不改用永久刪除

    用殼層的 SHFileOperationW（FO_DELETE＋FOF_ALLOWUNDO），跟檔案總管按 Delete 一樣；不出確認視窗、不顯示進度。
    檔案清單是一串以 \0 隔開、最後兩個 \0 結尾的路徑，一定要完整路徑才進得了回收筒。"""
    files = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    if not files:
        return
    import ctypes
    from ctypes import wintypes
    if not hasattr(ctypes, "windll"):
        raise OSError(tr("srv.fo.winOnly"))

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", ctypes.c_void_p),
                    ("pTo", ctypes.c_void_p), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", ctypes.c_void_p)]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    buf = ctypes.create_unicode_buffer("\0".join(files) + "\0\0")
    op = SHFILEOPSTRUCTW(None, FO_DELETE, ctypes.addressof(buf), None,
                         FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc or op.fAnyOperationsAborted:
        raise OSError(tr("srv.fo.recycleFail", code=rc))
