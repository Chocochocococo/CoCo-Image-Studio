# CoCo Image Studio

本地 AI 生圖與 LoRA 訓練工作台。

---

## 系統需求

| 項目 | 需求 |
|---|---|
| 作業系統 | **Windows**（`CoCo Image Studio.exe` 是唯一入口） |
| 顯卡 | 完整安裝：**NVIDIA，RTX 20 系列以上**（生成、精修、訓練沒有 CPU 模式，AMD / Intel / Mac 不支援）。輕量安裝不需要顯卡 |
| 驅動 | 完整安裝：**580 以上**（PyTorch 用的是 CUDA 13 版） |
| Python | 不用另外安裝：第一次執行時自動下載官方的 Python 3.11.9，放在 `data` 資料夾裡 |
| VRAM | 8GB 可跑（設定頁把 VRAM 策略改成「省 VRAM」，或打開「主模型用 fp8 存放」），12GB 以上較舒服 |

Anima 架構額外需要 **RTX 30 系列以上**——它用 bfloat16，Turing（20 系列）沒有
原生支援會非常慢。SDXL 與 SD1.5 走 fp16，20 系列沒問題。

模型檔不隨專案附帶。`data\models\` 底下一種一格，名稱跟 ComfyUI 一樣（`checkpoints`、`diffusion_models`、`loras`、`vae`、`text_encoders`、`embeddings`、`controlnet`、`upscale_models`、`adetailer`、`rembg`…），照用途放進去就好；也可以在「設定 → 資源資料夾」指向其他工具（StabilityMatrix、ComfyUI、Forge）的模型資料夾，CoCo 自己的這些一樣會讀。舊版放在 `loras\`、`vae\` 這些地方的檔案，啟動時會自動搬進對應的分格。
還沒有模型的話，可以在「設定 → 資源資料夾 → 下載模型」下載試用模型（Anima，約 5 GB）。

### 會不會動到我現有的環境

不會。PyTorch 與所有套件都裝在 `data\venv\`，版本由 `data\CoCo.bat` 與
`requirements-lock.txt` 固定（torch 2.14.0+cu130 / diffusers 0.39.0；輕量安裝是 CPU 版 torch 加
`requirements-lite.txt`，版本一樣照 `requirements-lock.txt`），
跟系統上的 Python、A1111、ComfyUI 各自獨立。Python 也是 CoCo 自己的一份（`data\python\`，python.org 的官方套件、核對過 SHA512）。

下載的 diffusers 模型也存在 `data\hf_cache\`，不會堆到 C 槽使用者目錄。

---

## 開始使用

1. 雙擊 **`CoCo Image Studio.exe`**
   > 第一次打開時 Windows 可能顯示「Windows 已保護您的電腦」：按「其他資訊」，再按「仍要執行」。
   > 這是因為 CoCo 的啟動程式沒有付費的程式碼簽章，Windows 不認得它；它只負責執行 `data\CoCo.bat`
   > （原始碼在 `data\tools\launcher.cs`）。之後再打開就不會再問。
2. 第一次會詢問安裝方式，輸入 1 或 2：
   - **1 完整安裝**（約 4 GB，需要 NVIDIA 顯卡）：全部功能。
   - **2 輕量安裝**（約 1 GB）：圖庫、探索、比較、重組、輸出、資源庫，以及生成台的提示詞與標籤；不能生成圖片，精修與訓練只能看。探索用 CPU 計算。

   偵測不到 NVIDIA 顯卡時建議輕量安裝。接著會問要不要在桌面和開始功能表建立 CoCo 的捷徑（Y／N）。
   選好後自動建立環境、安裝套件，完成後直接開啟 Studio
3. 之後執行會先更新圖庫索引，再開啟 Studio

> 輕量安裝之後想用生成：到「⚙ 設定 → 其他」點擊「改為完整安裝」，關閉 CoCo 後重新開啟，會自動補裝其餘套件。

> 第一次會自動下載官方的 Python 3.11.9（約 17 MB）放在 `data` 資料夾裡，不會影響電腦上已經裝好的 Python；沒有網路時才改用電腦上的 Python 3.11～3.13。
> `CoCo Image Studio.exe` 也可以釘選到工作列。
> 整個 CoCo 資料夾可以搬移或改名，不用重裝；搬了之後捷徑要重建（刪掉 `data\venv\.coco-shortcut` 再打開 CoCo 就會再問一次）。

---

## 設定在介面上改

開啟 Studio 後切到「設定」分頁，所有東西都能在那裡調：
掃描資料夾、模型、LoRA、提示詞、步數、CFG、取樣器、VRAM 策略……

改動存進 `data\settings.json`，**`data\config.py` 不會被改寫**。
`config.py` 是預設值，想手動編輯仍然可以，兩者並存。

---

## 資料夾

要開 CoCo 就點 `CoCo Image Studio.exe`。其他東西（程式、Python、套件、你的圖、模型、設定）都在旁邊的 `data` 資料夾裡。

```
CoCo-Image-Studio\
  CoCo Image Studio.exe  ← 唯一入口（有 CoCo 的圖示，只負責執行 data\CoCo.bat）
  data\             ← CoCo 的所有東西
    CoCo.bat          ← 首次安裝，之後掃描並啟動（直接執行它也一樣）
    outputs\          ← 生成結果
    models\           ← 模型，一種一格（checkpoints、loras、vae…），自行放置
    settings.json     ← 介面存的設定（自動產生）
    config.py         ← 預設值
    variables.txt     ← 變數（名稱 = 值）
    wildcards\        ← 萬用字元清單檔
    install_mode.txt  ← 安裝方式（full／lite，第一次安裝時產生）
    python\           ← CoCo 自己的 Python（第一次安裝時下載）
    venv\             ← 自動建立
    requirements-lock.txt
    requirements-lite.txt  ← 輕量安裝的套件（版本照 requirements-lock.txt）
    core\             ← Studio 核心
      conditioning.py   A1111 風格的提示詞編碼
      template.py       模板展開引擎
      gallery_index.py  圖庫索引器
      settings.py       設定的讀寫
    server\           ← 本機伺服器：HTTP 路由、圖庫、生成佇列、設定
    web\              ← 介面（index.html、js\、css\、locales\）
    tools\
      studio.py         啟動入口（本體在 server\）
```

> 以前的版本把這些東西都放在 CoCo 資料夾的最外層。換成這一版之後第一次開啟，CoCo 會自動把它們搬進 `data`，
> 設定、比較組、圖庫裡記的位置也會跟著改（原本的檔案備份在 `data\backups`）。

---

## 模板語法速查

| 寫法 | 作用 |
|---|---|
| `[[A\|B\|C]]` | 笛卡兒展開，多個會相乘（立繪組） |
| `{A\|B\|C}` | 隨機擇一 |
| `{3::A\|B}` | 加權隨機（A 機率是 B 的三倍） |
| `{A\|}` | 有機會什麼都不放 |
| `__pose__` | 讀 `data\wildcards\pose.txt` 抽一行 |
| `[[__pose__]]` | 對該檔每一行各展開一組 |
| `{set:c=red\|blue}` `{get:c}` | 抽一次，之後重複用同一個值 |
| `{2$$A\|B\|C}` | 抽 2 個不重複 |
| `{1-3$$A\|B\|C}` | 抽 1～3 個 |
| `{2$$ and $$A\|B\|C}` | 自訂分隔符 |
| `${c={red\|blue}}` `${c}` | 同上，Dynamic Prompts 風格 |
| `# 註解` | 到行尾都會被忽略 |

### 變數與萬用字元的差別

| | 是什麼 | 存在哪 | 怎麼用 |
|---|---|---|---|
| **變數** | 一個名稱對應一段固定內容 | `data\variables.txt` | `${my_oc}` |
| **萬用字元** | 一個檔案裡很多行選項 | `data\wildcards\*.txt` | `__pose__` 隨機抽一行 |

變數在工作台的「變數」分頁用表單新增與編輯，不用手動改檔案。
萬用字元在「萬用字元」分頁編輯，一行一個選項。

```
# variables.txt
my_oc     = silver long hair, blue eyes, hair ribbon
quality   = masterpiece, best quality, amazing quality, newest
```

**提示詞欄位本身就是模板**，直接寫動態語法即可，左欄可以預覽展開結果。

### 固定 seed（測 LoRA 用）

在「模板」分頁把格式設為「Forge 批次」並填圖片種子，
`data\prompts_out.txt` 會輸出成：

```
--prompt "..." --seed 12345
```

直接貼進 Forge 的 **Prompts from file or textbox**，所有變體共用同一個
seed，差異就只來自提示詞。搭配 `[[<lora:A:0.8>|<lora:B:0.8>|]]`
就是標準的 LoRA 對照測試。

`TPL_LOCK_RANDOM = True` 時，同一輪的所有組合共用隨機值——
換姿勢換表情但角色不變，這就是角色立繪組。

---

## 模型格式

三種都支援：

| 格式 | 長相 | 怎麼用 |
|---|---|---|
| **單檔** | `model.safetensors` | 放進模型資料夾，掃描後點卡片 |
| **diffusers 資料夾** | 含 `model_index.json` 的資料夾 | 同上，卡片會標「資料夾」 |
| **拆分零件** | transformer / vae / text_encoder 各一份 | 主模型照選，再到「VAE」與「文字編碼器」分頁各點一張卡片 |

零件覆寫就是 VAE 與文字編碼器兩個分頁。兩邊都有「模型自帶」那張卡片，
選它等於不覆寫；選別的就覆寫——可以讓多個模型共用同一份 VAE，省下數 GB。

### diffusers 模型要自己下載嗎

把整個資料夾放進模型資料夾即可，不必連網，掃描後卡片會標「資料夾」。

自動下載的模型會存進 `data\hf_cache\`，不會跑到 C 槽的使用者目錄。
要清掉直接刪那個資料夾就好。

---

## 常見問題

**這版 PyTorch 不支援你的顯卡（sm_120）**
RTX 50 系列需要 CUDA 12.8 以上的版本。刪除 `data\venv\.coco-ready` 後重新打開 CoCo，
它會檢查已安裝的版本，對不上就重新安裝驗證過的 cu130 版本。
（`CoCo.bat` 更新了鎖定的版本時也會自動做這件事，不必手動刪。）

**啟動時說「找不到你的顯示卡」（英文：cannot see your GPU）**
PyTorch 的版本是對的，但它看不到顯卡，所以**重裝沒有用**——問題在驅動或硬體。
先更新 NVIDIA 驅動（CUDA 13 需要 580 以上），再確認裝置管理員裡顯卡正常。
這個狀態下圖庫仍然可用，只有生成會失敗。

**LoRA 載入失敗**
若是 LyCORIS（LoHa / LoKr），diffusers 目前不支援。一般 kohya LoRA 若
diffusers 載入層出錯，程式會自動改用手動融合，詳細原因會顯示在生成紀錄。

**瀏覽方式**

標題列可切換三種排版：

| 選項 | 英文名 | 順序 | 空隙 |
|---|---|---|---|
| 響應式・齊行（預設） | justified layout | 正確 | 每列填滿 |
| 瀑布流・緊密 | masonry, shortest column | 會跳 | 密 |
| 瀑布流・依序 | masonry, round-robin | 正確 | 底部參差 |

齊行是把連續的圖組成一列、整列縮放到填滿寬度；瀑布流是垂直堆疊成數欄。
兩者是不同的排版方式，不是同一種的變體。

**圖庫顯示「無參數」**
那些圖的中繼資料被剝掉了（多半是外部下載或編輯過），不是錯誤。

**啟動時出現 `is not recognized as an internal command`**
請確認專案完整解壓縮：`CoCo Image Studio.exe` 旁邊要有 `data` 資料夾，裡面有 `CoCo.bat`、`requirements-lock.txt`、`requirements-lite.txt`、`core\`、
`tools\studio.py`、`server\`、`web\`。

**V-Pred 模型（NoobAI V-Pred 這類）**
直接選就好，CoCo 會從模型檔認出來、自動換成對的算法。官方建議取樣器用 Euler、CFG 4～5、28～35 步；
選了不適用的取樣器（LMS、Heun、DPM2、DPM++ SDE、DEIS、SA-Solver）會自動改用 Euler，生成紀錄會說明。

**VAE 怎麼選**

「模型 / LoRA / VAE」的 VAE 分頁有三種選擇：

| 選項 | 說明 |
|---|---|
| **sdxl-vae-fp16-fix**（預設） | SDXL 原版 VAE 在 fp16 會溢位產生黑圖，diffusers 的對策是解碼時升到 fp32，那一瞬間 VRAM 會暴衝。換成修正版就全程 fp16，不暴衝 |
| 模型自帶 | 畫面完全依照模型，但 SDXL 解碼時 VRAM 會暴衝 |
| 自訂 | 掃描 `data\models\vae\` 與你設定的 VAE 資料夾裡的檔案 |

「SDXL 分階段解碼」預設為自動；大圖會先卸載 UNet 再執行 VAE decode。

**VRAM 比 Forge 高**
在設定頁把 VRAM 策略改成「平衡」或「省 VRAM」。SDXL 另請維持分階段解碼
為「自動」；Anima 的平衡與省 VRAM 模式會在各階段卸載不用的元件。
還不夠的話，打開「設定 → 生成 → 主模型用 fp8 存放」：主模型只佔一半的顯示卡記憶體，
速度慢一點點，生成的圖會跟關掉時有一點不同。

---

## 回報問題

遇到問題或有建議，歡迎在 GitHub 的 Issues 回報。

這個專案**不接受程式碼貢獻（Pull Request）**：作者沒辦法自己審查別人寫的程式碼，無法確保裡面沒有惡意程式。

---

## 贊助

CoCo 免費、沒有廣告。如果它幫上了你的忙，歡迎贊助支持作者繼續開發。

| 平台 | 適合 | 連結 |
|---|---|---|
| 綠界 | 台灣：信用卡、ATM、超商等 | [前往贊助](https://p.ecpay.com.tw/D6622EF) |
| Ko-fi | 海外：信用卡、PayPal | [前往贊助](https://ko-fi.com/chocochocococo) |

CoCo 裡也找得到：⚙ 設定 → 關於。

---

## 授權

CoCo Image Studio 採用 [PolyForm Noncommercial License 1.0.0](LICENSE)，加上一段額外許可：
**程式本身禁止商業用途，用它生成的圖不限。**
下面是簡單的說明，實際以 [LICENSE](LICENSE) 的英文條文為準。

**可以，不用問作者：**

- 用 CoCo 生成的圖，做什麼都可以：接案、販售、授權、商業作品都行
- 個人的興趣、學習、研究、測試、私人娛樂
- 學校、慈善團體、公益組織、政府機關使用
- 修改程式，或把原版、改過的版本分享給別人：要一起附上 LICENSE 的條文，以及第一行的著作權聲明（`Required Notice:` 那一行）

**不可以，要先取得作者同意：**

- 販售 CoCo 或改過的版本、分享時收費
- 把 CoCo 放進收費的產品，或拿它架設收費的服務

**其他：**

- 專案裡附帶的第三方檔案與程式（字型、tokenizer、從 diffusers 抄來的程式等）各自保留原本的授權，見 [THIRD_PARTY_NOTICES.md](data/THIRD_PARTY_NOTICES.md)。
- CoCo 不附帶生成用的模型。你自己放進來的模型，以及用它們生成的圖，還要受各個模型自己的授權規範。
