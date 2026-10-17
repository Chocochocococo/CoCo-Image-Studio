"""CoCo Image Studio 的本機伺服器（原本整個寫在 tools/studio.py）。"""

import sys
from pathlib import Path

# 每個模組都要能 `import i18n`（core/i18n.py，訊息照介面語言）。state.py 也會加，
# 但 download、eagle 這類不經過 state 的模組被單獨匯入時（測試）還沒加到
_CORE = str(Path(__file__).resolve().parent.parent / "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
