# -*- coding: utf-8 -*-
"""
config.py —— CoCo Image Studio 的所有設定

**要改設定只要動這一個檔案。** 其他 .py 都不用碰。

用記事本或 VS Code 打開，改完存檔即可。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════
#  一、圖庫
# ══════════════════════════════════════════════════════════════

# 要納入圖庫的資料夾（可放多個，會含子資料夾）
SCAN_DIRS = [
    ROOT / "outputs",
    # ROOT / "another-image-folder",
]

GALLERY_PORT = 7789          # 圖庫網頁的連接埠
THUMB_WIDTH = 420            # 縮圖寬度
PAGE_SIZE = 60               # 每次捲動載入幾張


# ══════════════════════════════════════════════════════════════
#  二、生成預設值
# ══════════════════════════════════════════════════════════════

# 模型：單一 .safetensors 檔、diffusers 資料夾，或 HF 名稱
MODEL = str(ROOT / "models" / "waiIllustriousSDXL_v160.safetensors")

# Anima 的 flow-match shift。官方值 3.0，影響很大，其他架構用不到。
ANIMA_SHIFT = 3.0
# Z-Image 的 shift（跟 Anima 分開記）：官方 Base 6.0、Turbo 3.0
ZIMAGE_SHIFT = 6.0
# Krea 2 的 shift：官方 Turbo 固定 mu 1.15，等於 shift 3.16（Base 版照解析度算，1024² 約 2.5）
KREA2_SHIFT = 3.16
# Qwen Image 2.1 的 shift（1024² 時）：官方依解析度算，1024² 是 2.0，其他解析度 CoCo 照官方的斜率自動調整
QWEN21_SHIFT = 2.0
# FLUX.2 klein 的 shift（1024²、4 步時；4B、9B 共用）：預設值＝官方依解析度與步數自動算，改了才用自己的值
FLUX2_SHIFT = 9.89
# FLUX.1 dev 的 shift（1024² 時）：官方依解析度算，1024² 是 3.16（μ 1.15），預設值＝照官方自動算；schnell 官方不平移（1）
FLUX1_SHIFT = 3.16
FLUX1S_SHIFT = 1.0

# 生成時自動加上模型的品質標籤與預設負面（在資源庫的模型詳細裡設定）
QUALITY_PRESET = True

# 生成後自動修臉（像 ADetailer）：找臉、每張臉放大重畫再貼回。
# KEEP = True 時沒修過的也存，修好的存成它的新版本
AUTO_FACE = False
AUTO_FACE_KEEP = False
AUTO_FACE_STRENGTH = 0.4

# 模型自帶的 SDXL VAE 解碼時升 32 位元。False（預設）＝支援 bf16 的顯卡用 bf16：
# 精修 1024² 顯示卡記憶體峰值 15.4 → 10.2 GiB，出圖平均差 0.31/255
VAE_FP32 = False

# 主模型用 fp8 存放（⚙ 設定 → 生成）：Anima、SDXL、Z-Image、bf16 版的 Krea 2 與 Qwen Image 2.1。True＝記憶體減半，但構圖會跟 bf16 不同
# （RTX 5080、高速模式第二張：Anima 1024×1344 顯示卡峰值 8.05 → 6.21 GiB、16.1 → 16.9 秒；
#  SDXL 832×1216 8.90 → 6.47 GiB、4.6 → 5.2 秒。Z-Image 32 GB、1024²：bf16 邊算邊搬（權重鎖定）1.42 秒／步、
#  fp8 整個放得進顯示卡 1.4 秒／步；系統記憶體不夠、權重鎖定不了時 bf16 邊算邊搬是 2.2 秒／步）
FP8_WEIGHTS = False

# 用 CPU 產生亂數：同一個 seed 跟 Forge／A1111（RNG: CPU）畫出同樣的構圖。False＝用顯示卡（CoCo 以前的做法）
RNG_CPU = True

# 提示詞加權（(詞:1.2) 這類）後拉回整體強度：True＝A1111／Forge 預設的 Original（加重一個詞，其他詞會稍微變淡）；
# False＝Forge 的 No norm（只乘權重，加重的詞直接變強）。只影響 SDXL／SD 1.5（core/conditioning.encode_chunk）
EMPHASIS_NORM = True

# NGMS（Forge 的 Negative Guidance minimum sigma）：雜訊低於這個值的步驟每隔一步只算正面提示詞，比較快、細節略有不同。
# 0＝關（Forge 預設）。只影響 SDXL／SD 1.5（core/pipeline._ngms）
NGMS = 0.0

# 探索的參考圖來源：[{"kind": "folder", "path": …} 或 {"kind": "eagle", "path": "…\X.library", "folders": [Eagle 資料夾 id…]}]
# 在 ⚙ 設定 → 參考圖 裡加
REF_SOURCES = []
# 打標台的 WD14 模型資料夾（裡面有 model.onnx 或 model.safetensors，加上 selected_tags.csv）；空的＝用 CoCo 下載的
TAGGER_DIR = ""
# 打標台寫描述用的 JoyCaption：放 GGUF 的資料夾（主體＋檔名有 mmproj 的看圖部分）
JOYCAPTION_DIR = ""
# 生成的圖存成什麼檔名（core/naming.py）：/ 前面是資料夾；預設 outputs/日期/000001-seed-寬x高.png
SAVE_NAME = "{date}/{seq}-{seed}-{width}x{height}"
# 06 輸出的檔名：編輯（單張、批次輸出；{name} 是原圖的檔名）與拼接。匯出到其他資料夾時只用檔名那段
EDIT_NAME = "{date}/{name}_edit"
COLLAGE_NAME = "{date}/collage-{date}-{time}"
# 存回 Eagle：預設帶的標籤（逗號隔開）、標籤組合（[{"name": …, "tags": [...]}]，存的時候點一下套用）、
# 要不要把正面提示詞也存成標籤（存的小視窗裡可以每次改）
EAGLE_TAGS = "CoCo"
EAGLE_TAG_SETS = []
EAGLE_PROMPT_TAGS = False
# 生成台的 ControlNet 參考圖（server/controlnet.py）：最多 3 張，每張的來源、模型、種類、強度…
CONTROLNETS = []
# 提示詞存成標籤時略過的標籤（逗號隔開；品質標籤這類每張都有的）
EAGLE_SKIP_TAGS = ""

# 零件覆寫：一個架構記一格，因為零件跟架構是綁死的。
# 沒有列到的架構＝用模型自帶的。值可以是檔案路徑，SDXL 另外接受 "fp16fix"。
VAE_OVERRIDE = {
    "sdxl": "fp16fix",       # 原版 SDXL VAE 解碼會升 fp32，1536² 實測慢 70 倍
}
TE_OVERRIDE = {}             # Anima 需要 Qwen3-0.6B，在介面上點卡片指定

# 額外掃描的資源資料夾（可指向 StabilityMatrix、ComfyUI、Forge 既有目錄）。
# CoCo 自己的 models\ 底下一種一格（checkpoints、loras、vae…，名稱跟 ComfyUI 一樣），不用寫在這裡也一定會讀，
# 見 server/modelhome.py
MODEL_DIRS = []
LORA_DIRS = []
VAE_DIRS = []
TE_DIRS = []
EMBEDDING_DIRS = []           # Textual inversion：提示詞裡寫檔名就套用（A1111 的寫法）
DETECTOR_DIRS = []            # 修臉用的偵測模型（ADetailer 的 face_yolov8*.pt）

# 模型總資料夾：放所有模型的那一層（例如 StabilityMatrix 的 Data\Models）。
# 底下的子資料夾依名稱自動分類（StableDiffusion、Lora、VAE…，見 server/resources.py），
# 認錯的在設定裡改（MODELS_ROOT_MAP：{子資料夾名稱: 種類或 "skip"}）。
# 上面各類的資料夾清單照樣有效，當成額外的資料夾
MODELS_ROOT = ""
MODELS_ROOT_MAP = {}
RES_KIND = {}                 # 單一檔案改分類：{完整路徑: 種類}，在資源庫的詳細欄改

# LoRA：(路徑, 權重)，不用就留空清單
LORAS = [
    # (str(ROOT / "loras" / "SimpleLine-v1.safetensors"), 1.0),
]

# 品質標籤（masterpiece, best quality…）不寫在這裡：生成時照模型自動加（資源庫的「品質標籤與預設負面」）
PROMPT = (
    "1girl, solo, long hair, blue eyes, "
    "(white dress:1.2), standing, looking at viewer, "
    "outdoors, garden, soft lighting"
)

NEGATIVE = (
    "worst quality, low quality, blurry, bad anatomy, bad hands, "
    "text, watermark, signature"
)

WIDTH, HEIGHT = 832, 1216
STEPS = 28
CFG = 5.0
SEED = 12345                 # -1 為隨機
FIX_SEED = False             # True = 整批共用同一個種子
SAMPLER = "Euler a"
SCHEDULE = "Normal"    # Normal / Karras / Exponential / Beta / SGM Uniform
CLIP_SKIP = 2

# 條件編碼："auto"（建議）/ "a1111" / "native" / "both"
COND_MODE = "auto"
BATCH = 1                    # 一次生成幾張（一張一張跑）

# VRAM 策略（實測數據見 pipeline.VRAM_MODES 上面的註解）
#   speed    全部常駐 GPU
#   balanced 文字編碼完就把編碼器搬回 CPU，去噪主模型常駐（預設）
#   low      用完的元件都搬回 CPU，最省但較慢
# 高解析度的 VAE 分塊解碼三種模式都會自動開，跟模式無關。
VRAM_MODE = "balanced"
# SDXL 解碼策略：auto＝大於 1024² 時先卸載 UNet；on/off＝強制開／關。
# 舊版的 True / False 會由 settings.py 自動遷移成 on / auto。
LOW_DECODE = "auto"
TILE_VAE = "auto"            # VAE 分塊解碼：auto 依解析度 / on 一律分塊 / off 一律不分塊
DEBUG_LORA = False


# ══════════════════════════════════════════════════════════════
#  三、提示詞模板
# ══════════════════════════════════════════════════════════════

TEMPLATE = """
masterpiece, best quality, 1girl, solo,
{set:hair=silver|black|pink} {get:hair} long hair,
{set:cloth=white|navy|crimson} {get:cloth} dress,
[[standing|sitting|leaning against wall]],
[[smile|neutral expression|angry|surprised]],
{simple background|garden background|indoor background},
{2::soft lighting|dramatic lighting}
"""

TPL_COUNT = 1                # 每個組合各產生幾組隨機變體
TPL_LOCK_RANDOM = True       # True = 立繪組（同一輪角色不變）
TPL_SHOW_ALL = False         # True = 印出全部
TPL_SAVE = True              # 存成 prompts_out.txt

                             # 測 LoRA 時固定它，差異才只來自提示詞


# ══════════════════════════════════════════════════════════════
#  四、路徑（通常不用改）
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = ROOT / "outputs"       # 生成、精修、06 輸出存圖的地方（⚙ 設定 → 圖庫 可以換）
OUTPUT_DIR_DEFAULT = OUTPUT_DIR
WILDCARD_DIR = ROOT / "wildcards"    # 萬用字元：一行一個選項的清單檔
VARS_FILE = ROOT / "variables.txt"   # 變數：名稱 = 值
DB_PATH = ROOT / "gallery.db"
THUMB_DIR = ROOT / ".thumbs"
TRASH_DIR = ROOT / ".trash"
UI_FILE = ROOT / "gallery_ui.json"
ASSET_NOTES_FILE = ROOT / "asset_notes.json"   # 資源庫的收藏、備註、最近用過
COMPARE_FILE = ROOT / "compare_sets.json"       # 比較組
RECIPES_FILE = ROOT / "recipes.json"            # Recipe
DATASETS_FILE = ROOT / "datasets.json"          # 打標台：清理規則與門檻、各資料集的觸發詞、最近開過的資料集
PRESETS_FILE = ROOT / "presets.json"            # 各模型、各架構的品質標籤與預設負面
MASK_DIR = ROOT / "masks"                       # 局部重繪的遮罩（精修頁畫的）
COVER_DIR = ROOT / "covers"                     # 資源庫自己上傳的封面
SETTINGS_FILE = ROOT / "settings.json"
HF_CACHE = ROOT / "hf_cache"      # diffusers 下載的模型都放這裡


def _set_hf_home():
    """把 Hugging Face 的快取指到專案底下。

    預設在使用者家目錄的 .cache/huggingface，不在專案資料夾裡，
    很容易在 C 槽默默堆積數十 GB 而且不知道是誰下載的。
    """
    import os
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))


_set_hf_home()


# ══════════════════════════════════════════════════════════════
#  介面設定會覆蓋上面的預設值
#  （在工作台的「設定」分頁改的東西存在 settings.json）
# ══════════════════════════════════════════════════════════════

def _apply_settings():
    import json
    if not SETTINGS_FILE.exists():
        return
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    g = globals()
    for k, v in saved.items():
        if k == "SCAN_DIRS":
            g[k] = [Path(x) for x in v]
        elif k == "OUTPUT_DIR":
            if str(v or "").strip():                 # 空的＝用預設（CoCo 資料夾裡的 outputs）
                g[k] = Path(str(v).strip())
        elif k == "LORAS":
            g[k] = [(it["path"], it["weight"]) for it in v]
        elif k in g and isinstance(g[k], dict) and isinstance(v, dict):
            # 零件覆寫是一個架構記一格，要逐架構合併。
            # 整份取代的話，使用者替 anima 挑了 VAE，
            # 上面替 sdxl 準備的 fp16fix 就會消失。
            g[k] = {**g[k], **v}
        elif k in g:
            g[k] = v


_apply_settings()
