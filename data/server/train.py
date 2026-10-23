# -*- coding: utf-8 -*-
"""訓練空間的 02 訓練：SDXL 與 Anima 的 LoRA（core/sdxl_train.py、core/anima_train.py 在另一個行程跑）

- **預設**照使用者在 OneTrainer 常用的三份設定（`PRESETS`）：200MB（rank 16／alpha 8、10 epoch、前 4 個不存）、
  400MB（rank 32／alpha 16、10 epoch）、一張圖（rank 8／alpha 8、8 epoch、前 2 個不存）。其他照三份共同的
  （Prodigy 1.0、cosine、1024、batch 1、兩個文字編碼器一起訓練、crop jitter），寫死在 core/sdxl_train.py。
- **Anima** 的預設照使用者在 Anima-Standalone-Trainer 的設定（`ANIMA_PRESETS`）：rank 32／alpha 32、10 epoch、每個都存，
  優化器 Prodigy（最近幾次用的）或 AdamW 1e-4（之前用的）；只訓練 DiT。VAE 與文字編碼器用生圖時選的（沒選照生成台的規則找，
  assets.fill_parts，不寫回設定）。
- **開始前關掉生成引擎**：SDXL 訓練要 10 GB 上下，跟生成模型、JoyCaption 一起放不下；訓練中生圖、打標籤、寫描述都先擋下來
  （`busy()`，worker_client 啟動前問它）。
- 訓練行程用 JSON lines 回報（log、cache、progress、saved、done、error）；「停止」建停止檔（設定的 stop_file），它把這一步做完、存檔再結束。
  不走 stdin：Windows 上訓練行程有執行緒卡在讀管道時，載入 numpy 的 DLL 會跟著卡死。
- 存到 LoRA 資料夾底下的 名稱/ 資料夾（使用者在 OneTrainer 也是這樣放），中間的 epoch 存成 名稱-e05，最後一個是 名稱。
- 上次的設定記在 datasets.json 的 train（下次打開帶回來）。
- **Slider LoRA**（kind＝"slider"，core/slider_train.py）：中性、正向、反向、錨點四則提示詞，底圖預設照中性提示詞自動畫
  （`gen_req`，跟平常生圖一樣的設定），也可以選資料集；每 100 步存一次並試畫 −2、−1、+1、+2（768²）。設定的鍵都是 sl_ 開頭，
  跟一般 LoRA 的 rank、epoch 分開記。
- **試畫**（`sample_jobs`）：開始前畫一次不掛 LoRA 的，之後每存一個 epoch 畫一次，訓練行程直接用掛著的 LoRA 畫。
  取樣器、步數、CFG、CLIP skip、負面、品質標籤照平常生圖的設定（⚙ 設定），1024×1024、seed 固定；
  提示詞一行一則（最多 4 則），空白就用資料集第一張圖的標籤。存到 輸出資料夾/training_samples/名稱/（以前的「訓練試畫」照舊認），
  收到就加進圖庫、放進比較組（一則提示詞一組，`名稱 試畫`），挑哪個 epoch 最好可以直接用淘汰賽。
"""

import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .state import C, ROOT, SET

import i18n
from i18n import tr

PRESETS = {
    "200mb": {"rank": 16, "alpha": 8.0, "epochs": 10, "save_skip": 4},
    "400mb": {"rank": 32, "alpha": 16.0, "epochs": 10, "save_skip": 1},
    "single": {"rank": 8, "alpha": 8.0, "epochs": 8, "save_skip": 2},
}
# 介面能改的欄位與範圍：(型別, 最小, 最大)
FIELDS = {
    "rank": (int, 1, 256), "alpha": (float, 0.1, 256), "epochs": (int, 1, 200), "repeats": (int, 1, 100),
    "save_skip": (int, 0, 200), "keep_tags": (int, 0, 20), "seed": (int, 0, 2**31 - 1),
}
ANIMA_PRESETS = {
    "prodigy": {"rank": 32, "alpha": 32.0, "epochs": 10, "save_skip": 0, "optimizer": "prodigy"},
    "adamw": {"rank": 32, "alpha": 32.0, "epochs": 10, "save_skip": 0, "optimizer": "adamw"},
}
ARCHS = ("sdxl", "anima")
OPTIMIZERS = ("prodigy", "adamw")
KINDS = ("lora", "slider")
SLIDER_FIELDS = {"sl_steps": (int, 50, 5000), "sl_pair_steps": (int, 50, 5000), "sl_rank": (int, 1, 128), "sl_alpha": (float, 0.1, 128),
                 "sl_guidance": (float, 0.5, 10.0), "sl_n_base": (int, 4, 40)}
SLIDER_TEXT = ("sl_positive", "sl_negative", "sl_anchor", "sl_dataset", "sl_pos_dir", "sl_neg_dir", "sl_pair_prompt")
# Slider 的兩種教法：words＝正向、反向的詞（底模自己對那些詞的理解）；pairs＝成對圖片（同一張圖的正向版、反向版，
# 照 ai-toolkit 的 ImageReferenceSliderTrainer：兩個資料夾裡同名的圖是一對）
SLIDER_MODES = ("words", "pairs")
# Slider 的中性提示詞可以寫好幾則（介面一格一則，存成一行一則）：只寫一則短的，Slider 只在短提示詞上推得動
# （2026-10-06 實測：中性「1boy, solo」訓練的明暗 slider，使用者 35 個標籤的提示詞 ±2 亮度只差 203→184，前 8 個標籤 195→100）
MAX_NEUTRAL = 4
SLIDER_DEFAULT = {"sl_steps": 400, "sl_rank": 8, "sl_alpha": 4.0, "sl_guidance": 3.0, "sl_n_base": 12,
                  "sl_neutral": "1girl, solo", "sl_positive": "", "sl_negative": "", "sl_anchor": "", "sl_dataset": "",
                  "sl_mode": "words", "sl_pos_dir": "", "sl_neg_dir": "", "sl_pair_prompt": "",
                  # 成對圖片的步數另外記：學得快，400 步會練過頭（2026-10-08 實測 12 對：400 步 ±2 崩成訓練圖的畫風，
                  # 200 步撐到 ±1.5、100 步撐到 ±2 但效果弱；使用者選 200）
                  "sl_pair_steps": 200,
                  "sl_quality": True, "sl_sfw": True}
SLIDER_EVERY = 100            # 每幾步存一次並試畫（core/slider_train.py 的 save_steps）
SLIDER_MULTS = 4              # 每次試畫幾個強度（−2、−1、+1、+2）
SLIDER_RES = 768
# Slider 的底圖與試畫「避開 NSFW」（左欄的勾選，預設勾）加的負面：中性提示詞很短，SDXL 系的動漫模型常常自己跑去畫裸露的畫面，正向或反向帶到的標籤
# 有別的意思時更明顯（2026-10-06 實測：年齡 slider 的 mature female 一路變成性感的畫面）。底圖是訓練的材料，要乾淨
SLIDER_SAFE_NEG = "nsfw, explicit"
# Anima 只靠負面壓不住（同一次實測：不掛 LoRA 的底圖就已經很性感），要在提示詞加分級標籤 safe（Anima 官方的寫法）
SLIDER_SAFE_POS = {"anima": "safe"}
BOOLS = ("train_te", "flip", "shuffle_tags", "sample", "sl_quality", "sl_sfw", "keep_state")
DEFAULT = {"preset": "200mb", **PRESETS["200mb"], "repeats": 4, "train_te": True, "flip": False, "shuffle_tags": False,
           "keep_state": False,
           "keep_tags": 1, "seed": 42, "sample": True, "sample_prompts": "", "kind": "lora", **SLIDER_DEFAULT}
# 估時間用（RTX 5080、1024、batch 1；2026-10-06 實測）：SDXL 連文字編碼器一起訓練、Anima 只訓練 DiT
SEC_PER_STEP = {"sdxl": 1.27, "anima": 1.7}
# 試畫一張（1024²、28 步、含存檔；RTX 5080 2026-10-06 實測）：Anima 跟平常生圖一樣慢
SEC_PER_SAMPLE = {"sdxl": 6.5, "anima": 16.5}
# Slider（768²）：每步（不掛 LoRA 的三個預測＋強度 ±1 各一次前向與反向）、畫一張底圖、試畫一張
SEC_SLIDER = {"sdxl": {"step": 1.55, "base": 3.0, "sample": 3.9}, "anima": {"step": 1.9, "base": 7.2, "sample": 8.8}}
# 成對圖片每步只有 ±1 兩次前向與反向（不用先算不掛 LoRA 的三個預測）：SDXL 1.34 秒（2026-10-08 實測 12 對、400 步 9.0 分鐘）；
# Anima 未量測，照 SDXL 的比例先估
SEC_PAIR_STEP = {"sdxl": 1.34, "anima": 1.65}
MAX_SAMPLES = 4               # 試畫的提示詞最多幾則
SAMPLE_FOLDER = "training_samples"      # 輸出資料夾底下；固定英文、不跟著介面語言
LEGACY_SAMPLE_FOLDER = "訓練試畫"       # 2026-10-14 以前的名稱：同名 LoRA 已經有試畫在那裡的照舊放（加練、接著練的跟之前的放一起）
_LOCK = threading.RLock()
STATE = {"running": False, "phase": "", "step": 0, "total": 0, "epoch": 0, "epochs": 0, "loss": [], "lr": 0.0,
         "sec": 0.0, "eta": 0, "log": [], "saved": [], "error": "", "name": "", "started": 0.0, "cache": [0, 0],
         "samples": [], "sets": {}, "kind": "lora"}
_PROC = {"p": None, "stop_file": None, "lang_file": None, "run": {}, "kind": "lora", "n_images": 0, "resumed": "",
         "state_saved": False}
RUN = {}                       # 這次一般 LoRA 訓練的資料集與設定：訓練完寫進資料集的訓練紀錄（dataset.add_run）


def busy():
    """訓練中：生成、打標籤、寫描述先不要開工作行程（顯示卡放不下）"""
    return bool(STATE["running"])


def _log(msg, kind="info"):
    STATE["log"].append({"t": time.strftime("%H:%M:%S"), "m": str(msg), "k": kind})
    del STATE["log"][:-300]


def state():
    with _LOCK:
        d = {k: v for k, v in STATE.items() if k not in ("log", "loss", "saved", "samples", "sets")}
        d["log"] = list(STATE["log"][-120:])
        d["loss"] = list(STATE["loss"])
        d["saved"] = list(STATE["saved"])
        d["samples"] = list(STATE["samples"])
        d["sets"] = dict(STATE["sets"])
        return d


# ---------------------------------------------------------------- 設定

def clean(data):
    """介面送來的設定 → 收得下的欄位（範圍夾住）；預設套用後再蓋上使用者改的"""
    out = {}
    if "preset" in (data or {}):
        preset = str(data["preset"] or "")
        out["preset"] = preset if preset in PRESETS or preset in ANIMA_PRESETS else ""     # ""＝自己改過數字，下次打開不要又亮預設
    if (data or {}).get("optimizer") in OPTIMIZERS:
        out["optimizer"] = data["optimizer"]
    for k, (typ, lo, hi) in FIELDS.items():
        v = (data or {}).get(k)
        if v is None or v == "":
            continue
        try:
            out[k] = typ(min(hi, max(lo, typ(v))))
        except (TypeError, ValueError):
            continue
    for k, (typ, lo, hi) in SLIDER_FIELDS.items():
        v = (data or {}).get(k)
        if v is None or v == "":
            continue
        try:
            out[k] = typ(min(hi, max(lo, typ(v))))
        except (TypeError, ValueError):
            continue
    for k in SLIDER_TEXT:
        if (data or {}).get(k) is not None:
            out[k] = str(data[k]).strip()[:1000]
    if (data or {}).get("sl_neutral") is not None:
        out["sl_neutral"] = str(data["sl_neutral"]).rstrip()[:4000]      # 介面是好幾格：前面空的格子要留著
    if (data or {}).get("kind") in KINDS:
        out["kind"] = data["kind"]
    if (data or {}).get("sl_mode") in SLIDER_MODES:
        out["sl_mode"] = data["sl_mode"]
    for k in BOOLS:
        if k in (data or {}):
            out[k] = bool(data[k])
    for k in ("name", "model", "dataset", "out_root"):
        if (data or {}).get(k) is not None:
            out[k] = str(data[k]).strip()[:500]
    if (data or {}).get("sample_prompts") is not None:
        out["sample_prompts"] = str(data["sample_prompts"]).rstrip()[:4000]      # 介面是 4 格：前面空的格子要留著
    return out


def safe_name(name):
    """LoRA 的名字：Windows 不能用的字元換掉、頭尾的點與空白拿掉"""
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip(" .")
    return n[:80]


def _store_load():
    from . import dataset
    with dataset._LOCK:
        return dataset._load()


def last():
    d = _store_load().get("train")
    return {**DEFAULT, **clean(d if isinstance(d, dict) else {})}


def remember(cfg):
    from . import dataset
    with dataset._LOCK:
        d = dataset._load()
        d["train"] = clean(cfg)
        dataset._save(d)


# 存成設定檔（像 OneTrainer 的 config）：datasets.json 的 train_saved，[{name, cfg, date}]，最近存的在前。
# 「上次的設定」（remember）只有一份、按開始才記；這個是使用者自己取名字存、要的時候載入
MAX_SAVED = 50


def _saved(d):
    lst = d.get("train_saved")
    if not isinstance(lst, list):
        return []
    return [e for e in lst if isinstance(e, dict) and isinstance(e.get("name"), str) and isinstance(e.get("cfg"), dict)]


def saved_configs():
    return _saved(_store_load())


def save_config(name, cfg):
    """同名的蓋過去（搬到最前面）"""
    from . import dataset
    name = str(name or "").strip()[:80]
    if not name:
        return {"error": tr("srv.tr.cfgName")}
    with dataset._LOCK:
        d = dataset._load()
        lst = [e for e in _saved(d) if e["name"] != name]
        d["train_saved"] = [{"name": name, "cfg": clean(cfg), "date": time.strftime("%Y-%m-%d %H:%M")}] + lst[:MAX_SAVED - 1]
        dataset._save(d)
    return {"ok": True, "saved": saved_configs()}


def delete_config(name):
    from . import dataset
    with dataset._LOCK:
        d = dataset._load()
        d["train_saved"] = [e for e in _saved(d) if e["name"] != str(name or "")]
        dataset._save(d)
    return {"ok": True, "saved": saved_configs()}


def lora_root():
    """預設存到哪：LoRA 資料夾清單的第一個（使用者的在前、CoCo 自己的 models\\loras 最後）"""
    from . import modelhome
    from .resources import res_dirs
    dirs = [d for d in res_dirs("lora") if Path(d).is_dir()]
    return str(dirs[0]) if dirs else str(modelhome.own("lora"))


def save_epochs(cfg):
    """哪幾個 epoch 會存檔（core/sdxl_train.py 同一條規則：最後一個一定存，前 save_skip 個不存）"""
    n = int(cfg["epochs"])
    return [e for e in range(1, n + 1) if e == n or e > int(cfg["save_skip"])]


def pairs_of(cfg):
    """成對圖片：正向、反向兩個資料夾裡檔名（不含副檔名，不分大小寫）一樣的圖是一對。
    提示詞用正向那張旁邊的 .txt，沒有就用反向那張的，都沒有用 sl_pair_prompt（照 ai-toolkit）→ [{pos, neg, prompt}]"""
    from . import dataset
    pos, neg = dataset.root_of(cfg.get("sl_pos_dir")), dataset.root_of(cfg.get("sl_neg_dir"))
    if not pos or not neg:
        return []
    by = {}
    for img in dataset.list_images(neg):
        by.setdefault(img.stem.lower(), img)
    default = str(cfg.get("sl_pair_prompt") or "").strip()
    out = []
    for img in dataset.list_images(pos):
        other = by.get(img.stem.lower())
        if other is None or other.resolve() == img.resolve():
            continue
        text = dataset.read_text(dataset.caption_path(img)) or dataset.read_text(dataset.caption_path(other))
        text = ", ".join(x.strip() for x in text.replace("\r", "\n").replace("\n", ",").split(",") if x.strip())
        out.append({"pos": str(img), "neg": str(other), "prompt": text or default})
    return out


def neutrals(cfg):
    """Slider 的中性提示詞：一行一則、空的拿掉、最多 MAX_NEUTRAL 則"""
    return [ln.strip() for ln in str(cfg.get("sl_neutral") or "").splitlines() if ln.strip()][:MAX_NEUTRAL]


def with_words(neutral, words):
    """中性提示詞接上正向／反向要加的詞；中性已經有的標籤不重複加（以前正向寫整句的，同一則中性照樣是同一句）"""
    from . import presets
    add = presets.missing(neutral, words)
    return ", ".join(x for x in (neutral.strip(" ,"), ", ".join(add)) if x)


def sample_prompts(cfg):
    """試畫的提示詞（介面一格一則，存成一行一則，最多 MAX_SAMPLES 則，空的不畫）；沒開試畫是空的。
    第一格空白用預設（Slider 是中性提示詞，LoRA 是資料集第一張圖的標籤）：介面上第一格的淡字就是它"""
    if not cfg.get("sample"):
        return []
    lines = [ln.strip() for ln in str(cfg.get("sample_prompts") or "").splitlines()][:MAX_SAMPLES] or [""]
    if not lines[0]:
        lines[0] = slider_prompt(cfg) if cfg.get("kind") == "slider" else default_prompt(cfg.get("dataset"))
    return [ln for ln in lines if ln]


def slider_prompt(cfg):
    """Slider 試畫第一格空白時用的：用詞的是第一則中性提示詞，成對圖片的是沒有 .txt 時用的提示詞（沒寫就用第一對的）"""
    if cfg.get("sl_mode") == "pairs":
        return str(cfg.get("sl_pair_prompt") or "").strip() or next((p["prompt"] for p in pairs_of(cfg)), "")
    return (neutrals(cfg) or [""])[0]


def default_prompt(ds):
    """沒寫試畫的提示詞：資料集第一張圖的標籤（不含描述）"""
    from . import dataset
    root = dataset.root_of(ds) if ds else None
    imgs = dataset.list_images(root) if root else []
    return ", ".join(dataset._tags_of(root, imgs[0])) if imgs else ""


def slider_saves(steps):
    """Slider 哪幾步存檔（core/slider_train.py 的 save_steps 同一條規則：每 SLIDER_EVERY 步，最後一步一定存）"""
    return sorted(set(range(SLIDER_EVERY, int(steps) + 1, SLIDER_EVERY)) | {int(steps)})


def estimate(n_images, cfg):
    """步數與大約幾分鐘（1024、batch 1；試畫的時間也算進去）"""
    if cfg.get("kind") == "slider":
        arch = cfg.get("arch") if cfg.get("arch") in ARCHS else "sdxl"
        t = {**SEC_SLIDER[arch], **({"step": SEC_PAIR_STEP[arch]} if cfg.get("sl_mode") == "pairs" else {})}
        steps = int(cfg["sl_steps"])
        n_samples = len(sample_prompts(cfg)) * (1 + len(slider_saves(steps)) * SLIDER_MULTS)
        n_base = 0 if cfg.get("sl_dataset") or cfg.get("sl_mode") == "pairs" else int(cfg["sl_n_base"])
        return {"steps": steps, "samples": n_samples, "base": n_base,
                "minutes": round((steps * t["step"] + n_base * t["base"] + n_samples * t["sample"]) / 60, 1)}
    steps = n_images * int(cfg["repeats"]) * int(cfg["epochs"])
    n_samples = len(sample_prompts(cfg)) * (len(save_epochs(cfg)) + 1)
    arch = cfg.get("arch") if cfg.get("arch") in ARCHS else "sdxl"
    return {"steps": steps, "samples": n_samples,
            "minutes": round((steps * SEC_PER_STEP[arch] + n_samples * SEC_PER_SAMPLE[arch]) / 60, 1)}


def preview(data):
    """還沒開始前：這樣設定共幾步、大約幾分鐘"""
    c, err = check(data)
    if err:
        return {"error": err}
    return {"ok": True, "n_images": c["n_images"], "out_dir": c["out_dir"], **estimate(c["n_images"], c),
            "default_prompt": slider_prompt(c) if c["kind"] == "slider" else default_prompt(c["dataset"])}


def options():
    from . import dataset, pairmaker
    from .resources import model_arch, res_scan
    models = [{"path": it["path"], "name": it.get("name") or Path(it["path"]).stem, "arch": a} for it in res_scan("model")
              for a in [model_arch(it["path"])] if a in ARCHS]
    cur = str(SET.load().get("MODEL") or "")
    lst = last()
    if not lst.get("model") or not any(m["path"] == lst["model"] for m in models):
        # 沒訓練過：用生圖正在用的模型（能訓練的話），不然清單第一個
        lst["model"] = cur if any(m["path"] == cur for m in models) else (models[0]["path"] if models else "")
    reps = {e["path"]: e["repeats"] for e in dataset.library() if e.get("repeats")}
    return {"last": lst, "presets": {"sdxl": PRESETS, "anima": ANIMA_PRESETS}, "models": models, "datasets": dataset.recent(),
            "ds_repeats": reps, "saved": saved_configs(), "lora_root": lora_root(), "progress": progress_list(),
            "qwen_models": pairmaker.qwen_models(), "pairs_root": pairmaker.default_root(),
            "max_samples": MAX_SAMPLES, "max_neutral": MAX_NEUTRAL, "state": state()}


SCRIPT = {"sdxl": "sdxl_train.py", "anima": "anima_train.py"}
ARCH_NAME = {"sdxl": "SDXL", "anima": "Anima"}


def sample_folder(name):
    """試畫存在輸出資料夾底下的哪裡（相對路徑）。名稱裡的 { } 會被當成檔名變數，先換掉"""
    safe = re.sub(r'[{}]', '_', name)
    old = Path(C.OUTPUT_DIR) / LEGACY_SAMPLE_FOLDER / safe
    top = LEGACY_SAMPLE_FOLDER if old.is_dir() and not (Path(C.OUTPUT_DIR) / SAMPLE_FOLDER / safe).exists() else SAMPLE_FOLDER
    return f"{top}/{safe}"


def anima_parts():
    """Anima 的 VAE 與文字編碼器：生圖時選的；沒選照生成台的規則找（資源庫裡只有一個、或檔名跟官方一樣的），不寫回設定"""
    from . import assets
    cfg = SET.load()
    got = assets.fill_parts("anima", {"vae": cfg.get("VAE_OVERRIDE"), "te": cfg.get("TE_OVERRIDE")})
    if "error" in got:
        return {"error": got["error"]}
    pick = {k: (got[k].get("anima") if isinstance(got[k], dict) else got[k]) or "" for k in ("vae", "te")}
    return pick


def check(data):
    """開始前檢查：回傳 (完整設定, None) 或 (None, 錯誤訊息)"""
    from . import dataset
    from .resources import model_arch
    c = {**DEFAULT, **clean(data)}          # 介面選預設時已經把 rank、epoch 這些填好送來
    model = c.get("model") or ""
    if not model or not Path(model).is_file():
        return None, tr("srv.tr.needBase")
    arch = model_arch(model)
    if arch not in ARCHS:
        return None, tr("srv.tr.archs")
    if arch == "anima":
        parts = anima_parts()
        if "error" in parts:
            return None, parts["error"]
        c.update(parts)
        if c.get("optimizer") not in OPTIMIZERS:
            c["optimizer"] = "prodigy"
    if c["kind"] == "slider" and c.get("sl_mode") == "pairs":
        if not dataset.root_of(c.get("sl_pos_dir")) or not dataset.root_of(c.get("sl_neg_dir")):
            return None, tr("srv.tr.needPairDirs")
        if dataset.root_of(c["sl_pos_dir"]) == dataset.root_of(c["sl_neg_dir"]):
            return None, tr("srv.tr.pairDirsSame")
        n = len(pairs_of(c))
        if not n:
            return None, tr("srv.tr.noPairs")
        c.update(sl_pos_dir=str(dataset.root_of(c["sl_pos_dir"])), sl_neg_dir=str(dataset.root_of(c["sl_neg_dir"])),
                 sl_steps=c["sl_pair_steps"])          # 之後估時間、送給訓練行程都用這個
    elif c["kind"] == "slider":
        if not neutrals(c):
            return None, tr("srv.tr.needNeutral")
        for k, msg in (("sl_positive", "srv.tr.needPos"), ("sl_negative", "srv.tr.needNeg")):
            if not str(c.get(k) or "").strip():
                return None, tr(msg)
        n = int(c["sl_n_base"])
        if c.get("sl_dataset"):
            root = dataset.root_of(c["sl_dataset"])
            n = len(dataset.list_images(root)) if root else 0
            if not n:
                return None, tr("srv.tr.baseDirEmpty")
            c["sl_dataset"] = str(root)
    else:
        root = dataset.root_of(c.get("dataset"))
        if not root:
            return None, tr("srv.tr.needDataset")
        n = len(dataset.list_images(root))
        if not n:
            return None, tr("srv.tr.datasetEmpty")
        c["dataset"] = str(root)
    name = safe_name(c.get("name"))
    if not name:
        return None, tr("srv.tr.needName")
    out_root = Path(c.get("out_root") or lora_root())
    if not out_root.is_absolute():
        return None, tr("srv.needFullPathSave")
    c.update(name=name, out_dir=str(out_root / name), n_images=n, arch=arch)
    return c, None


def sample_jobs(c):
    """試畫的工作（core/sdxl_train.py 的 make_sampler 交給生成引擎畫）：跟平常生圖一樣的設定與品質標籤"""
    if c.get("kind") == "slider":
        return [gen_req(c, pr, SLIDER_RES, quality=bool(c.get("sl_quality")), sfw=bool(c.get("sl_sfw"))) for pr in sample_prompts(c)]
    return [gen_req(c, pr, 1024) for pr in sample_prompts(c)]


def gen_req(c, pr, size, quality=None, sfw=False):
    """一則提示詞 → 生成引擎的請求（品質標籤、負面、取樣器、步數、CFG 跟平常生圖一樣）。
    quality：加不加這個模型的品質標籤與預設負面；None＝照生成台的開關（一般 LoRA 的試畫），True／False＝Slider 左欄的勾選
    （預設勾：中性提示詞本來就很短，沒加的話底圖又灰又糊，2026-10-06 實測）。
    sfw：負面加 SLIDER_SAFE_NEG、Anima 的提示詞加 SLIDER_SAFE_POS（Slider 左欄的「避開 NSFW」，預設勾）。
    使用者已經寫了的標籤都不重複加"""
    from . import presets
    from .lorahash import model_hash
    from .state import VERSION
    cfg = SET.load()
    negative = str(cfg.get("NEGATIVE") or "")
    arch = c.get("arch") or "sdxl"
    use = cfg.get("QUALITY_PRESET", True) if quality is None else quality
    preset = presets.resolve(c["model"], arch) if use else None
    mh = model_hash(c["model"])
    full_p, full_n, pinfo = pr, negative, {"preset": {"on": False}}
    if preset:
        full_p, full_n, added = presets.apply(pr, negative, preset)
        pinfo = {"preset": {"on": True, **added}}
        if added["quality"] or added["negative"]:
            pinfo.update(user_prompt=pr, user_negative=negative)
    if sfw:
        tag = SLIDER_SAFE_POS.get(arch, "")
        if tag and presets.missing(full_p, tag):
            full_p = f"{full_p.strip(' ,')}, {tag}"
        missing = presets.missing(full_n, SLIDER_SAFE_NEG)
        if missing:
            full_n = ", ".join([x for x in (full_n.strip(" ,"), ", ".join(missing)) if x])
    return {"prompt": full_p, "negative": full_n, **pinfo, "width": size, "height": size,
            "steps": int(cfg.get("STEPS") or 28), "cfg": float(cfg.get("CFG") or 5.0), "seed": int(c["seed"]),
            "sampler": cfg.get("SAMPLER") or "Euler a", "schedule": cfg.get("SCHEDULE") or "Normal",
            "clip_skip": int(cfg.get("CLIP_SKIP") or 2), "rng": "cpu" if cfg.get("RNG_CPU", True) else "gpu",
            "out_dir": str(C.OUTPUT_DIR), "folder": sample_folder(c['name']),
            **({"model_hash": mh} if mh else {}), **({"coco_version": VERSION} if VERSION else {}),
            # Anima：取樣的 shift 照生成台、記下用了哪個 VAE 與文字編碼器（圖庫存 Recipe 時帶得到）
            **({"shift": float(cfg.get("ANIMA_SHIFT") or 3.0), "parts": {"vae": c["vae"], "te": c["te"]}}
               if arch == "anima" else {})}


def add_sample(m):
    """訓練行程畫好一張試畫：加進圖庫、放進這則提示詞的比較組（第一張時開新的一組）"""
    from . import compare
    from .state import DB_WRITE_LOCK, db
    try:
        import gallery_index as GI
    except ImportError:
        from core import gallery_index as GI
    idx = int(m.get("index") or 0)
    item = {"path": m["path"], "epoch": int(m.get("epoch") or 0), "index": idx, "lora": m.get("lora") or "", "id": None,
            "mult": float(m.get("mult", 1.0))}
    con = db()
    try:
        with DB_WRITE_LOCK:
            GI.index_file(con, m["path"])
            con.commit()
        r = con.execute("SELECT id, mtime FROM images WHERE lower(path) = lower(?)", (m["path"],)).fetchone()
        if r:
            from .library import thumb_ver
            item.update(id=r[0], v=thumb_ver(m["path"], r[1]))
            with _LOCK:
                sid = STATE["sets"].get(str(idx), "")
                n = STATE["name"]
            name = tr("srv.tr.sampleSet", name=n) + (f" {idx + 1}" if idx else "")
            got = compare.add_images(con, sid, [], name=name, paths=[m["path"]])
            with _LOCK:
                STATE["sets"][str(idx)] = got["id"]
    except Exception as e:                 # 圖庫或比較組出問題不影響訓練
        with _LOCK:
            _log(tr("srv.tr.sampleLibFail", e=f"{type(e).__name__}: {e}"), "warn")
    finally:
        con.close()
    with _LOCK:
        STATE["samples"].append(item)


def lora_run(c):
    """一般 LoRA 訓練行程的設定"""
    keys = (("arch", "model", "dataset", "out_dir", "name", "rank", "alpha", "epochs", "repeats", "save_skip", "seed")
            + (("vae", "te", "optimizer") if c["arch"] == "anima" else ("train_te", "flip", "shuffle_tags", "keep_tags")))
    return {k: c[k] for k in keys}


def slider_run(c):
    """Slider 訓練行程（core/slider_train.py）的設定。每則中性提示詞一組（中性、中性＋正向的詞、中性＋反向的詞），
    底圖的生圖請求也是每則一個（768²），訓練行程輪流畫。成對圖片：pairs 是每一對的兩張圖與提示詞，不用底圖"""
    if c.get("sl_mode") == "pairs":
        return {"arch": c["arch"], "model": c["model"], "name": c["name"], "out_dir": c["out_dir"], "seed": c["seed"],
                "pairs": pairs_of(c), "positive": Path(c["sl_pos_dir"]).name, "negative": Path(c["sl_neg_dir"]).name,
                "neutral": "", "anchor": "", "guidance": 1.0, "steps": c["sl_steps"], "rank": c["sl_rank"], "alpha": c["sl_alpha"],
                "save_every": SLIDER_EVERY, "resolution": SLIDER_RES, **({"vae": c["vae"], "te": c["te"]} if c["arch"] == "anima" else {})}
    neus = neutrals(c)
    bases = [{k: v for k, v in gen_req(c, n, SLIDER_RES, quality=bool(c.get("sl_quality")), sfw=bool(c.get("sl_sfw"))).items()
              if k not in ("out_dir", "folder")} for n in neus]
    return {"arch": c["arch"], "model": c["model"], "name": c["name"], "out_dir": c["out_dir"], "seed": c["seed"],
            "prompts": [{"neu": n, "pos": with_words(n, c["sl_positive"]), "neg": with_words(n, c["sl_negative"])} for n in neus],
            "neutral": "\n".join(neus), "positive": c["sl_positive"], "negative": c["sl_negative"], "anchor": c["sl_anchor"],
            "steps": c["sl_steps"], "rank": c["sl_rank"], "alpha": c["sl_alpha"], "guidance": c["sl_guidance"],
            "n_base": c["sl_n_base"], "dataset": c["sl_dataset"], "save_every": SLIDER_EVERY, "resolution": SLIDER_RES,
            "bases": bases, **({"vae": c["vae"], "te": c["te"]} if c["arch"] == "anima" else {})}


# ---------------------------------------------------------------- 執行

def start(data, stop_worker):
    with _LOCK:
        if STATE["running"]:
            return {"error": tr("srv.tr.busy")}
        c, err = check(data)
        if err:
            return {"error": err}
        remember({**data, "name": c["name"]})
        slider = c.get("kind") == "slider"
        run = slider_run(c) if slider else lora_run(c)
        run["samples"] = sample_jobs(c)
        run["keep_state"] = bool(c.get("keep_state"))
        RUN.clear()
        if not slider:
            RUN.update(dataset=c["dataset"], name=c["name"], model=Path(c["model"]).name, arch=c["arch"],
                       epochs=int(c["epochs"]), repeats=int(c["repeats"]), rank=int(c["rank"]),
                       cfg=clean({**data, "name": c["name"]}))
        STATE.update(loss=[], log=[], sets={})
        if slider and c.get("sl_mode") == "pairs":
            _log(tr("srv.tr.startPairs", name=c['name'], arch=ARCH_NAME[c['arch']], steps=c['sl_steps'], rank=c['sl_rank'], alpha=c['sl_alpha'],
                    n=c['n_images'], pos=Path(c['sl_pos_dir']).name, neg=Path(c['sl_neg_dir']).name))
        elif slider:
            _log(tr("srv.tr.startSlider", name=c['name'], arch=ARCH_NAME[c['arch']], steps=c['sl_steps'], rank=c['sl_rank'], alpha=c['sl_alpha'],
                    g=c['sl_guidance'], pos=c['sl_positive'], neg=c['sl_negative'], n=len(neutrals(c))))
        else:
            opt = tr("srv.tr.optimizer", o=c['optimizer'].replace('prodigy', 'Prodigy').replace('adamw', 'AdamW')) if c["arch"] == "anima" else ""
            _log(tr("srv.tr.startLora", name=c['name'], arch=ARCH_NAME[c['arch']], n=c['n_images'], repeats=c['repeats'], epochs=c['epochs'],
                    rank=c['rank'], alpha=c['alpha']) + opt)
        _launch(run, c.get("kind", "lora"), c["n_images"], stop_worker)
    return {"ok": True}


def _launch(run, kind, n_images, stop_worker):
    """開訓練行程（新開始與接續共用）；STATE 的 loss、log、sets 呼叫的人先設好（接續時帶上次的）"""
    stop_worker()                    # 生成引擎讓出顯示卡
    cfg_path = Path(tempfile.gettempdir()) / f"coco-train-{int(time.time())}.json"
    stop_file = cfg_path.with_suffix(".stop")
    stop_file.unlink(missing_ok=True)
    run = {**run, "stop_file": str(stop_file)}
    cfg_path.write_bytes(json.dumps(run, ensure_ascii=False).encode("utf-8"))
    slider = kind == "slider"
    STATE.update(running=True, phase="load", step=0, total=0, epoch=0, epochs=0 if slider else int(run["epochs"]),
                 lr=0.0, sec=0.0, eta=0, saved=[], error="", name=run["name"], started=time.time(),
                 cache=[0, n_images], samples=[], kind=kind)
    _log(tr("srv.tr.engineOff"))
    lang_file = cfg_path.with_suffix(".lang")       # 訓練中換介面語言，之後的訊息跟著換（_lang_changed）
    lang_file.write_bytes(i18n.lang().encode("utf-8"))
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "COCO_LANG": i18n.lang(),
           "COCO_LANG_FILE": str(lang_file)}
    script = "slider_train.py" if slider else SCRIPT[run["arch"]]
    p = subprocess.Popen([sys.executable, "-u", str(Path(ROOT) / "core" / script), str(cfg_path)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", cwd=str(ROOT), env=env,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _PROC.update(p=p, stop_file=stop_file, lang_file=lang_file, run={k: v for k, v in run.items() if k not in ("stop_file", "resume")},
                 kind=kind, n_images=n_images, resumed=run.get("resume") or "", state_saved=False)
    threading.Thread(target=_read, args=(p, cfg_path, stop_file, lang_file), daemon=True).start()
    threading.Thread(target=_read_err, args=(p,), daemon=True).start()


def _lang_changed(lang):
    """訓練行程沒有指令可以帶語言（生成子行程是每個指令帶），改寫語言檔，它每次產生訊息前讀（core/i18n.py）。
    以前只在啟動時給：訓練中換語言，之後的試畫、完成、錯誤紀錄還是原本的語言（R68）。訓練結束時 _read 刪掉這個檔，之後就不寫"""
    f = _PROC.get("lang_file")
    if f and f.exists():
        try:
            f.write_bytes(lang.encode("utf-8"))
        except OSError:
            pass


i18n.on_change(_lang_changed)


# ---------------------------------------------------------------- 暫停、接續（使用者要求，2026-10-07 修改核對表 F3）
# 停止時訓練行程存進度（core/resume.py：LoRA 資料夾裡的 _coco_progress/state.bin），這裡在旁邊寫 info.json：
# 訓練行程的設定、步數、loss 曲線、試畫的比較組——接續時照原本的設定開，不用介面上的設定剛好一樣。
# 有哪些進度記在 datasets.json 的 train_progress（資料夾清單）。接續的那次練完（沒勾「練完也存進度」）就把進度丟進資源回收筒。
PROGRESS_DIR = "_coco_progress"
MAX_PROGRESS = 30


def _progress_index(d):
    lst = d.get("train_progress")
    return [x for x in lst if isinstance(x, str)] if isinstance(lst, list) else []


def _set_progress_index(fn):
    from . import dataset
    with dataset._LOCK:
        d = dataset._load()
        d["train_progress"] = fn(_progress_index(d))[:MAX_PROGRESS]
        dataset._save(d)


def write_progress(m):
    """訓練行程存好進度（{"type": "state"}）：旁邊寫 info.json、記進清單"""
    run = _PROC.get("run") or {}
    folder = Path(m["path"]).parent
    with _LOCK:
        info = {"name": run.get("name", ""), "kind": _PROC.get("kind", "lora"), "arch": run.get("arch", ""),
                "step": int(m.get("step") or 0), "total": int(STATE.get("total") or 0), "done": bool(m.get("done")),
                "date": time.strftime("%Y-%m-%d %H:%M"), "n_images": _PROC.get("n_images", 0), "run": run,
                "loss": list(STATE["loss"]), "sets": dict(STATE["sets"]), "record": dict(RUN)}
    _PROC["state_saved"] = True
    try:
        (folder / "info.json").write_bytes(json.dumps(info, ensure_ascii=False).encode("utf-8"))
        key = str(folder)
        _set_progress_index(lambda lst: [key] + [x for x in lst if x.lower() != key.lower()])
        _log(tr("srv.tr.progSaved") if not info["done"] else tr("srv.tr.progSavedDone"), "ok")
    except OSError as e:
        _log(tr("srv.tr.progInfoFail", e=e), "warn")


def _read_info(folder):
    try:
        info = json.loads((Path(folder) / "info.json").read_text(encoding="utf-8"))
        st = Path(folder) / "state.bin"
        if not isinstance(info, dict) or not isinstance(info.get("run"), dict):
            return None
        return {**info, "dir": str(folder), "mb": round(st.stat().st_size / 2 ** 20)}     # 進度檔不見了：stat 丟出錯誤，不列
    except (OSError, ValueError):
        return None


def progress_list():
    """03 訓練列出來的進度（新的在前）：找不到的（搬走、刪掉）不列"""
    out = []
    for folder in _progress_index(_store_load()):
        info = _read_info(folder)
        if info:
            out.append({k: info.get(k) for k in ("dir", "name", "kind", "arch", "step", "total", "done", "date", "mb")}
                       | {"epochs": int(info["run"].get("epochs") or 0), "steps": int(info["run"].get("steps") or 0)})
    return out


def discard_progress(folder):
    """丟掉進度：整個 _coco_progress 資料夾丟進資源回收筒（只認清單裡的、名字對的資料夾）"""
    from .fileops import recycle
    key = str(folder or "")
    if key.lower() not in {x.lower() for x in _progress_index(_store_load())} or Path(key).name != PROGRESS_DIR:
        return {"error": tr("srv.tr.progNotFound")}
    try:
        recycle([key])
    except OSError as e:
        return {"error": str(e)}
    _set_progress_index(lambda lst: [x for x in lst if x.lower() != key.lower()])
    return {"ok": True, "progress": progress_list()}


def _keep_final(run, kind):
    """練完的再加練：最後那個檔（名稱.safetensors）先改名成 名稱-e10／名稱-s0400，不然會被新的蓋掉"""
    out = Path(run["out_dir"])
    final = out / f"{run['name']}.safetensors"
    if not final.is_file():
        return
    tag = f"-s{int(run['steps']):04d}" if kind == "slider" else f"-e{int(run['epochs']):02d}"
    dst = final.with_name(f"{run['name']}{tag}.safetensors")
    if dst.exists():
        return
    final.rename(dst)
    side = final.with_suffix(".json")
    if side.is_file() and not dst.with_suffix(".json").exists():
        side.rename(dst.with_suffix(".json"))
    _log(tr("srv.tr.renamed", old=dst.name, new=final.name))


def resume(data, stop_worker):
    """從進度接著練。more：多練幾個 epoch（Slider 是幾步）；練完的進度一定要多練"""
    with _LOCK:
        if STATE["running"]:
            return {"error": tr("srv.tr.busy")}
        folder = str(data.get("dir") or "")
        if folder.lower() not in {x.lower() for x in _progress_index(_store_load())}:
            return {"error": tr("srv.tr.progNotFound")}
        info = _read_info(folder)
        if not info:
            return {"error": tr("srv.tr.progBroken")}
        kind = info.get("kind") or "lora"
        try:
            more = max(0, int(data.get("more") or 0))
        except (TypeError, ValueError):
            more = 0
        unit = "steps" if kind == "slider" else "epochs"
        more = min(more, SLIDER_FIELDS["sl_steps"][2] if kind == "slider" else FIELDS["epochs"][2])
        if info.get("done") and not more:
            return {"error": tr("srv.tr.doneSteps") if kind == "slider" else tr("srv.tr.doneEpochs")}
        run = dict(info["run"])
        if not Path(run.get("model") or "").is_file():
            return {"error": tr("srv.tr.baseGone")}
        if info.get("done"):
            try:
                _keep_final(run, kind)
            except OSError as e:
                return {"error": tr("srv.tr.renameFail", e=e)}
        run[unit] = int(run[unit]) + more
        run["resume"] = str(Path(folder) / "state.bin")
        run["keep_state"] = bool(data.get("keep_state"))
        RUN.clear()
        rec = info.get("record") or {}
        if rec:
            RUN.update(rec, epochs=int(run.get("epochs") or rec.get("epochs") or 0))
        STATE.update(loss=[x for x in info.get("loss") or [] if isinstance(x, list)], log=[],
                     sets={str(k): v for k, v in (info.get("sets") or {}).items()})
        _log(tr("srv.tr.resume", name=run['name'], step=info.get('step', 0)) + ((tr("srv.tr.moreSteps", n=more) if kind == 'slider' else tr("srv.tr.moreEpochs", n=more)) if more else ""))
        _launch(run, kind, int(info.get("n_images") or 0), stop_worker)
    return {"ok": True}


def handle(m):
    """訓練行程的一行回報 → STATE"""
    t = m.get("type")
    if t == "sample":
        add_sample(m)
        return
    with _LOCK:
        if t == "log":
            _log(m.get("msg", ""), m.get("kind", "info"))
        elif t == "sampling":
            STATE["phase"] = "sample"
        elif t == "base":
            STATE.update(phase="base", cache=[int(m.get("done", 0)), int(m.get("total", 0))])
        elif t == "cache":
            STATE.update(phase="cache", cache=[int(m.get("done", 0)), int(m.get("total", 0))])
        elif t == "progress":
            STATE.update(phase="train", step=m["step"], total=m["total"], epoch=m["epoch"], epochs=m["epochs"],
                         lr=m.get("lr", 0.0), sec=m.get("sec", 0.0), eta=m.get("eta", 0))
            STATE["loss"].append([m["step"], m.get("loss_avg", m.get("loss"))])
            if len(STATE["loss"]) > 600:                 # 長的訓練：每兩點留一點
                STATE["loss"] = STATE["loss"][::2]
        elif t == "saved":
            STATE["saved"].append({"path": m["path"], "epoch": m.get("epoch"), "step": m.get("step")})
            _log(tr("srv.tr.saved", name=Path(m['path']).name), "ok")
            write_sidecar(m["path"])
        elif t == "state":
            write_progress(m)
        elif t == "done":
            STATE.update(running=False, phase="done")
            record_run()
            used_up()
        elif t == "error":
            STATE.update(running=False, phase="error", error=m.get("msg", ""))
            _log(tr("srv.tr.failed", msg=m.get('msg', '')), "err")


def used_up():
    """接續的那次練完、這次沒再存進度（沒按停止、沒勾「練完也存進度」）：用過的進度丟進資源回收筒"""
    folder = Path(_PROC.get("resumed") or "").parent
    if not _PROC.get("resumed") or _PROC.get("state_saved") or folder.name != PROGRESS_DIR:
        return
    _PROC["resumed"] = ""
    try:
        from .fileops import recycle
        recycle([str(folder)])
        key = str(folder).lower()
        _set_progress_index(lambda lst: [x for x in lst if x.lower() != key])
    except Exception as e:                # 丟不掉不影響訓練結果，03 訓練還列得出來、可以自己丟
        _log(tr("srv.tr.recycleFail", e=e), "warn")


def write_sidecar(path):
    """一般 LoRA 存好一個檔：旁邊存一份說明檔（StabilityMatrix 的 .json 格式），觸發詞用資料集的——資源庫、生成台的 LoRA
    就看得到觸發詞（使用者回報：練好的 LoRA 沒有記觸發詞）。資料集沒設觸發詞就不存；別處來的同名說明檔不蓋掉"""
    if not RUN.get("dataset"):
        return
    from . import dataset
    try:
        words = [w.strip() for w in dataset.get_trigger(RUN["dataset"]).split(",") if w.strip()]
        if not words:
            return
        p = Path(path).with_suffix(".json")
        if p.is_file():
            try:
                if not json.loads(p.read_text(encoding="utf-8")).get("coco_trained"):
                    return
            except ValueError:
                return
        body = {"trainedWords": words, "baseModel": RUN.get("model", ""), "coco_trained": True}
        p.write_bytes(json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8"))
    except OSError as e:                  # 說明檔存不了不影響訓練結果
        _log(tr("srv.tr.triggerFail", e=e), "warn")


def record_run():
    """一般 LoRA 訓練完：寫進資料集的訓練紀錄（01 資料集的卡片、打標頁看得到）。按停止的也記（stopped）"""
    if not RUN or not STATE["saved"]:
        return
    from . import dataset
    last = STATE["saved"][-1]
    try:
        dataset.add_run(RUN["dataset"], {
            "name": RUN["name"], "file": Path(last["path"]).name, "dir": str(Path(last["path"]).parent),
            "model": RUN["model"], "arch": RUN["arch"], "date": time.strftime("%Y-%m-%d %H:%M"),
            "epochs": RUN["epochs"], "repeats": RUN["repeats"], "rank": RUN["rank"], "steps": int(last.get("step") or 0),
            **({"stopped": True} if "-step" in Path(last["path"]).stem else {}),
            **({"cfg": RUN["cfg"]} if RUN.get("cfg") else {})})      # 02 打標的訓練紀錄按「用這組設定」再訓練一次
    except Exception as e:                 # 紀錄寫不進去不影響訓練結果
        _log(tr("srv.tr.recordFail", e=f"{type(e).__name__}: {e}"), "warn")
    RUN.clear()


def _read(p, *temp_files):
    try:
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                handle(json.loads(line))
            except (ValueError, KeyError):
                pass
    finally:
        code = p.wait()
        for f in temp_files:                           # 暫存的設定檔、停止檔與語言檔
            f.unlink(missing_ok=True)
        with _LOCK:
            if STATE["running"]:                       # 沒有說 done 就結束：當掉了
                STATE.update(running=False, phase="error", error=STATE["error"] or tr("srv.tr.exited", code=code))
                _log(STATE["error"], "err")


# 套件的警告、進度條不是錯誤
_NOISE = re.compile(r"warn|Loading pipeline|it/s\]|%\||Using decoupled|FutureWarning|UserWarning|should be kept in float32|experimental feature|^\s*deprecate\(|not created using `ComponentSpec`|^\s*$|lv = float", re.I)


def _read_err(p):
    try:
        for line in p.stderr:
            if not _NOISE.search(line):
                with _LOCK:
                    _log(line.rstrip()[:300], "err")
    except Exception:
        pass


def stop():
    """這一步做完、存檔再結束"""
    p = _PROC["p"]
    if p is None or p.poll() is not None:
        return {"ok": True}
    try:
        _PROC["stop_file"].write_bytes(b"stop")
        with _LOCK:
            _log(tr("srv.tr.stopping"), "warn")
    except OSError:
        p.kill()
    return {"ok": True}
