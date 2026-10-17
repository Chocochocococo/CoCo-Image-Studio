# -*- coding: utf-8 -*-
"""
make_launcher.py —— 編 CoCo Image Studio.exe（最外層的啟動程式，原始碼 tools/launcher.cs）

用 Windows 內建的 .NET Framework 4 的 C# 編譯器（csc.exe，Windows 10／11 都有），不用另外裝東西。
圖示用 assets/coco.ico（tools/make_icon.py 產生的）。原始碼的雜湊值寫進 exe（AssemblyInformationalVersion），
tests/test_launcher.py 拿它確認 exe 是照現在的原始碼編的——改了 launcher.cs 或圖示就要重跑這支。
編出來的 exe 沒有程式碼簽章：從網路下載的人第一次打開會看到 Windows 的「Windows 已保護您的電腦」（使用者決定的，2026-10-17）。

跑法（在 data 資料夾裡）：venv\\Scripts\\python.exe tools\\make_launcher.py
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # data
SRC = ROOT / "tools" / "launcher.cs"
ICON = ROOT / "assets" / "coco.ico"
EXE_NAME = "CoCo Image Studio.exe"
OUT = ROOT.parent / EXE_NAME


def source_hash():
    """原始碼＋圖示的雜湊值（前 16 碼）：兩個任一改了，exe 就要重編"""
    h = hashlib.sha1(SRC.read_bytes().replace(b"\r\n", b"\n"))
    h.update(ICON.read_bytes())
    return h.hexdigest()[:16]


def csc():
    win = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for fw in ("Framework64", "Framework"):
        p = win / "Microsoft.NET" / fw / "v4.0.30319" / "csc.exe"
        if p.exists():
            return p
    raise SystemExit("找不到 C# 編譯器（.NET Framework 4 的 csc.exe）")


def main():
    code = SRC.read_text(encoding="utf-8").replace('"SOURCE_HASH"', f'"{source_hash()}"')
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "launcher.cs"
        src.write_bytes(code.encode("utf-8"))
        r = subprocess.run([str(csc()), "/nologo", "/target:exe", "/optimize", "/codepage:65001", f"/win32icon:{ICON}",
                            f"/out:{OUT}", str(src)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("編譯失敗：\n" + r.stdout + r.stderr)
    print("寫好了：", OUT, f"（{OUT.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()
