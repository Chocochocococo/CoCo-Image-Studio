# Prodigy 優化器（prodigyopt 1.1.2）

訓練 LoRA 用的 Prodigy（會自己估學習率的 Adam）。使用者在 OneTrainer 用的就是這個版本（OneTrainer 的 PRODIGY 選項
呼叫 `prodigyopt.Prodigy`），CoCo 原封不動帶進來，訓練出來的行為才會一樣；也不用多裝一個套件。

| 檔案 | 來源 |
|---|---|
| `prodigy.py` | prodigyopt 1.1.2 的 `prodigyopt/prodigy.py` |
| `LICENSE` | 同一版的授權全文 |

- 來源：https://github.com/konstmish/prodigy （PyPI 的 prodigyopt 1.1.2；2026-10-06 從使用者已經裝好的 OneTrainer 環境複製，
  內容跟 PyPI 的同一版相同）
- 授權：MIT License，Copyright (c) 2023 Konstantin Mishchenko
- 改過的地方：沒有。`__init__.py` 只匯出 `Prodigy`
