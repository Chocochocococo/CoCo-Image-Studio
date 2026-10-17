# -*- coding: utf-8 -*-
"""HTTPS 改用 Windows 的憑證（truststore）。

防毒軟體掃 HTTPS 時（Norton 等）會用自己的憑證重新簽每個網站。那張憑證裝在 Windows 的憑證庫裡，
但 httpx 只認 certifi 的清單：huggingface_hub 的下載（diffusers 第一次載入 SDXL 單檔要抓的設定檔、
transformers 的檔案）就 CERTIFICATE_VERIFY_FAILED，全新安裝第一次生成就失敗（2026-10-16 這台實測；
CoCo 自己的下載用 urllib，本來就讀 Windows 的憑證，不受影響）。
truststore 讓 ssl 改問 Windows 驗證，沒有防毒軟體時結果一樣。沒裝 truststore（還沒補裝的舊環境）就照舊。

每個行程的入口呼叫一次：伺服器（server/app.py）、生成子行程（worker.py）、訓練行程（*_train.py）。
"""

_STATE = []


def use_system_certs():
    """回傳有沒有改成功（只做一次）"""
    if not _STATE:
        try:
            import truststore
            truststore.inject_into_ssl()
            _STATE.append(True)
        except Exception:
            _STATE.append(False)
    return _STATE[0]
