# -*- coding: utf-8 -*-
"""
studio.py —— CoCo Image Studio 的啟動入口（CoCo.bat 呼叫這個檔）

伺服器程式在 server/，前端頁面在 web/。
這個檔案只負責設定匯入路徑並啟動，實際內容見 server/app.py。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

if __name__ == "__main__":
    # CoCo 的東西改放在 data\：舊的配置第一次啟動時把旁邊的搬進來（server/relocate.py）。
    # 要在載入伺服器之前做——伺服器一載入就讀設定，設定要先搬進 data
    from server import relocate
    relocate.main(ROOT.parent, ROOT)

    from server.app import main
    main()
