# Qwen Image 2.1（從 diffusers 主線抄來的）

CoCo 用的 diffusers 0.39（以及 0.40）還沒有 Qwen Image 2.1，這三個檔案是從 diffusers 的開發主線原封不動抄來的，
等 diffusers 正式版收進去之後就可以刪掉、改用套件裡的。

| 檔案 | 來源 |
|---|---|
| `transformer_qwenimage21.py` | `src/diffusers/models/transformers/transformer_qwenimage21.py` |
| `autoencoder_kl_qwenimage21.py` | `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py` |
| `pipeline_qwenimage21.py` | `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` |

- 來源：https://github.com/huggingface/diffusers ，commit `8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd`（2026-10-04 取得；
  Qwen Image 2.1 是 PR #14804 加進去的）
- 授權：Apache License 2.0，Copyright 2026 Qwen-Image Team, The HuggingFace Team（各檔案開頭保留原本的授權聲明）
- 改過的地方：只有 import——相對匯入（`from ...utils import …`）改成從安裝好的 `diffusers` 套件匯入，
  三個檔案彼此之間改成從這個資料夾匯入。程式本身沒有改。

CoCo 怎麼用它們見 `core/qwen21.py`（ComfyUI 單檔的鍵名對照、int8 + convrot 的還原、文生圖的流程）。
