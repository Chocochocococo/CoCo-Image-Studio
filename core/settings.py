# -*- coding: utf-8 -*-
"""
settings.py —— 設定的讀寫

config.py 是預設值，settings.json 是介面寫入的使用者設定。
讀取時以 settings.json 優先。

**介面不會去改寫 config.py。** 程式改寫程式碼容易出錯又難還原，
所以介面只寫 JSON，config.py 永遠保持乾淨、可手動編輯。
"""

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "settings.json"
_WRITE_LOCK = threading.RLock()

# 介面可以調整的欄位，以及型別
FIELDS = {
    # 圖庫
    "SCAN_DIRS": "paths",
    "GALLERY_PORT": "int",
    "THUMB_WIDTH": "int",
    "PAGE_SIZE": "int",
    # 圖庫的「丟進回收桶」改成直接刪除檔案（無法復原，預設關；伺服器刪之前也會檢查這個）
    "HARD_DELETE": "bool",
    # 看大圖：「只送提示詞」時保留生成台的 LoRA、複製提示詞時連 <lora:…> 一起複製（預設都關；只有前端讀）
    "SEND_KEEP_LORAS": "bool",
    "COPY_WITH_LORAS": "bool",
    # 生成
    "MODEL": "str",
    # 零件跟架構是綁死的（SDXL 吃 fp16fix、Anima 吃 Qwen 那組），
    # 所以覆寫要一個架構記一格，不能只有一格全域的
    "VAE_OVERRIDE": "byarch",
    "TE_OVERRIDE": "byarch",
    "MODEL_DIRS": "paths",
    "LORA_DIRS": "paths",
    "VAE_DIRS": "paths",
    "TE_DIRS": "paths",
    "EMBEDDING_DIRS": "paths",
    "DETECTOR_DIRS": "paths",
    # 模型總資料夾與分類的修正（server/resources.py）
    "MODELS_ROOT": "str",
    # 存圖的資料夾（空的＝CoCo 資料夾裡的 outputs）；換了會自動加進圖庫要掃的資料夾（server/settings_api.py）
    "OUTPUT_DIR": "str",
    "MODELS_ROOT_MAP": "strmap",
    "RES_KIND": "strmap",
    "LORAS": "loras",
    "PROMPT": "text",
    "NEGATIVE": "text",
    # 標籤模式暫時關掉的標籤（不在提示詞裡）：[{text, after}]
    "PROMPT_OFF": "chipoff",
    "NEGATIVE_OFF": "chipoff",
    "WIDTH": "int",
    "HEIGHT": "int",
    "STEPS": "int",
    "CFG": "float",
    # 各架構記住的步數、CFG、取樣器、排程器（換架構時 server/settings_api.arch_switch 存取）
    "ARCH_PARAMS": "archparams",
    "SEED": "int",
    "FIX_SEED": "bool",
    "SAMPLER": "str",
    "SCHEDULE": "str",
    "CLIP_SKIP": "int",
    "COND_MODE": "str",
    "BATCH": "int",
    "VRAM_MODE": "str",
    "LOW_DECODE": "decode_mode",
    "TILE_VAE": "str",
    "ANIMA_SHIFT": "float",
    "ZIMAGE_SHIFT": "float",
    "KREA2_SHIFT": "float",
    "QWEN21_SHIFT": "float",
    "FLUX2_SHIFT": "float",
    "FLUX1_SHIFT": "float",
    "FLUX1S_SHIFT": "float",
    # 生成時自動加上模型的品質標籤與預設負面（server/presets.py）
    "QUALITY_PRESET": "bool",
    # 生成後自動修臉（像 ADetailer）：開關在生成鈕旁邊；修改幅度、要不要留沒修過的在 ⚙ 設定 → 生成
    "AUTO_FACE": "bool",
    "AUTO_FACE_KEEP": "bool",
    "AUTO_FACE_STRENGTH": "float",
    # 模型自帶的 SDXL VAE 解碼時升 32 位元（出圖跟以前逐位元一樣，但多吃約 5 GB；預設關，用 bf16）
    "VAE_FP32": "bool",
    "FP8_WEIGHTS": "bool",
    # 用 CPU 產生亂數（跟 Forge／A1111 一樣，同一個 seed 構圖相同）；關掉是 CoCo 以前的（顯示卡）
    "RNG_CPU": "bool",
    # 加權後拉回整體強度（Forge 的 Emphasis：Original／No norm）、NGMS（0＝關）：只影響 SDXL／SD 1.5
    "EMPHASIS_NORM": "bool",
    "NGMS": "float",
    # 生成的圖的檔名格式（core/naming.py）：日期、序號、seed、模型、提示詞…這些變數
    "SAVE_NAME": "text",
    "EDIT_NAME": "text",
    "COLLAGE_NAME": "text",
    # 探索的參考圖來源（server/explore.py）：資料夾或 Eagle 的檔案庫（可以只選幾個 Eagle 資料夾）
    "REF_SOURCES": "refsources",
    # 打標台的 WD14 模型資料夾（server/dataset.py）
    "TAGGER_DIR": "str",
    # 打標台寫描述的 JoyCaption 資料夾（GGUF）
    "JOYCAPTION_DIR": "str",
    # 存回 Eagle（server/eagle.py）：預設標籤、標籤組合、正面提示詞存成標籤
    "EAGLE_TAGS": "text",
    "EAGLE_TAG_SETS": "tagsets",
    "EAGLE_PROMPT_TAGS": "bool",
    "EAGLE_SKIP_TAGS": "text",
    # 生成台的 ControlNet 參考圖（server/controlnet.py）
    "CONTROLNETS": "cnunits",
    # 模板
    "TEMPLATE": "text",
    "TPL_COUNT": "int",
    "TPL_LOCK_RANDOM": "bool",
    "TPL_SHOW_ALL": "bool",
    "TPL_SAVE": "bool",
}


# 數字欄位的上下限（審查紀錄 R17）：存設定時夾在範圍裡，生成時伺服器也照這份檢查（server/handler.plan_run）。
# 跟介面上的格子一樣（web/js/generate.js、setup.js；tests/test_limits.py 對得上）
RANGES = {
    "WIDTH": (256, 4096), "HEIGHT": (256, 4096),
    "STEPS": (1, 200), "CFG": (0.0, 30.0), "BATCH": (1, 100), "CLIP_SKIP": (1, 12), "ANIMA_SHIFT": (0.1, 20.0), "ZIMAGE_SHIFT": (0.1, 20.0),
    "KREA2_SHIFT": (0.1, 20.0), "QWEN21_SHIFT": (0.1, 20.0), "FLUX2_SHIFT": (0.1, 20.0), "FLUX1_SHIFT": (0.1, 20.0),
    "FLUX1S_SHIFT": (0.1, 20.0),
    "SEED": (-1, 2 ** 63 - 1),                 # 圖庫（SQLite）的整數存得下的最大值
    "AUTO_FACE_STRENGTH": (0.05, 1.0), "NGMS": (0.0, 15.0),
    "THUMB_WIDTH": (120, 1024), "PAGE_SIZE": (20, 500), "GALLERY_PORT": (1024, 65535),
}
MAX_JOBS = 1000               # 一次送出最多幾張（動態語法展開 × 張數 × 對照）
MAX_PROMPT = 50000            # 提示詞最多幾個字


def clamp(key, v):
    """夾在 RANGES 裡（沒有範圍的照原樣）"""
    lo, hi = RANGES.get(key, (None, None))
    if lo is not None and v < lo:
        return type(v)(lo)
    if hi is not None and v > hi:
        return type(v)(hi)
    return v


def as_chipoff(v):
    """暫時關掉的標籤 [{text, after}]：只收字串、去掉空的，最多 200 個"""
    out = []
    for m in v if isinstance(v, list) else []:
        if isinstance(m, dict) and isinstance(m.get("text"), str) and m["text"].strip():
            out.append({"text": m["text"].strip()[:2000], "after": str(m.get("after") or "")[:2000]})
    return out[:200]


ARCH_PARAM_KEYS = ("STEPS", "CFG", "SAMPLER", "SCHEDULE")


def as_archparams(v):
    """{架構: {STEPS, CFG, SAMPLER, SCHEDULE}}：數字夾在範圍裡，認不得的丟掉"""
    out = {}
    if not isinstance(v, dict):
        return out
    for a, p in v.items():
        if not isinstance(a, str) or not a or not isinstance(p, dict):
            continue
        q = {}
        for k, typ in (("STEPS", int), ("CFG", float)):
            try:
                if p.get(k) is not None:
                    q[k] = clamp(k, typ(float(p[k])))
            except (TypeError, ValueError):
                pass
        for k in ("SAMPLER", "SCHEDULE"):
            if isinstance(p.get(k), str) and p[k].strip():
                q[k] = p[k].strip()[:60]
        if q:
            out[a[:20]] = q
    return out


def as_byarch(v):
    """統一成 {架構: 路徑}。舊設定是單一字串，這裡一律當成空的，
    真正的搬移交給 migrate_overrides，那裡才知道原本配的是哪個架構。"""
    if isinstance(v, dict):
        return {str(k): str(x) for k, x in v.items() if str(k)}
    return {}


def as_decode_mode(v):
    """統一成 auto / on / off，並遷移舊版 checkbox 的布林值。"""
    if isinstance(v, bool):
        # False 是舊版預設，不代表使用者明確要求永遠關閉。
        return "on" if v else "auto"
    s = str(v or "auto").strip().lower()
    return s if s in ("auto", "on", "off") else "auto"


def as_strmap(v):
    """{字串: 字串}；空的鍵、空的值都不留"""
    if not isinstance(v, dict):
        return {}
    return {str(k).strip(): str(x).strip() for k, x in v.items() if str(k).strip() and str(x).strip()}


def as_refsources(v):
    """探索的參考圖來源：[{"kind": "folder"|"eagle", "path": …, "folders": [Eagle 資料夾 id…]（eagle 才有）}]"""
    out = []
    for s in v if isinstance(v, list) else []:
        if not isinstance(s, dict) or not str(s.get("path") or "").strip():
            continue
        kind = "eagle" if s.get("kind") == "eagle" else "folder"
        item = {"kind": kind, "path": str(s["path"]).strip()}
        if kind == "eagle":
            item["folders"] = [str(x) for x in (s.get("folders") or []) if str(x).strip()][:500]
        out.append(item)
    return out[:50]


def as_tagsets(v):
    """存回 Eagle 的標籤組合：[{"name": 名稱, "tags": [標籤…]}]；沒有標籤的不留"""
    out = []
    for s in v if isinstance(v, list) else []:
        if not isinstance(s, dict):
            continue
        raw = s.get("tags")
        if isinstance(raw, str):
            raw = raw.replace("，", ",").split(",")
        tags = []
        for x in raw if isinstance(raw, list) else []:
            x = str(x).strip()[:100]
            if x and x.lower() not in (y.lower() for y in tags):
                tags.append(x)
        if tags:
            out.append({"name": str(s.get("name") or "").strip()[:60], "tags": tags[:100]})
    return out[:50]


def _num(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def as_cnunits(v):
    """生成台的 ControlNet 參考圖：[{src: {kind, id/rid/name}, model, type, prep, scale, start, end, resize, low, high, off}]，最多 3 張"""
    out = []
    for u in v if isinstance(v, list) else []:
        if not isinstance(u, dict):
            continue
        src = u.get("src") if isinstance(u.get("src"), dict) else None
        if src:
            src = {k: src[k] for k in ("kind", "id", "rid", "name", "v") if isinstance(src.get(k), (str, int, float))}
        out.append({"src": src or None, "model": str(u.get("model") or "")[:300], "type": str(u.get("type") or "")[:20],
                    "prep": str(u.get("prep") or "")[:20], "scale": _num(u.get("scale"), 0, 2, 1.0),
                    "start": _num(u.get("start"), 0, 1, 0.0), "end": _num(u.get("end"), 0, 1, 1.0),
                    "resize": "fit" if u.get("resize") == "fit" else "crop",
                    "low": int(_num(u.get("low"), 1, 254, 100)), "high": int(_num(u.get("high"), 2, 255, 200)), "off": bool(u.get("off"))})
    return out[:3]


def pick(value, arch, fallback=""):
    """取出某個架構要用的覆寫"""
    return as_byarch(value).get(arch or "", fallback)


def _defaults():
    import config
    out = {}
    for k in FIELDS:
        v = getattr(config, k, None)
        if FIELDS[k] == "paths":
            v = [str(x) for x in (v or [])]
        elif FIELDS[k] == "loras":
            v = [{"path": str(p), "weight": float(w)} for p, w in (v or [])]
        elif FIELDS[k] == "byarch":
            v = as_byarch(v)
        elif FIELDS[k] == "decode_mode":
            v = as_decode_mode(v)
        elif FIELDS[k] == "strmap":
            v = as_strmap(v)
        elif FIELDS[k] == "refsources":
            v = as_refsources(v)
        elif FIELDS[k] == "tagsets":
            v = as_tagsets(v)
        elif FIELDS[k] == "cnunits":
            v = as_cnunits(v)
        elif FIELDS[k] == "archparams":
            v = as_archparams(v)
        elif FIELDS[k] == "chipoff":
            v = as_chipoff(v)
        elif isinstance(v, Path):
            v = str(v)
        out[k] = v
    return out


def _raw():
    if not FILE.exists():
        return {}
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data):
    """先完整寫入同目錄暫存檔，再原子取代 settings.json。"""
    tmp = FILE.with_name(FILE.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(FILE)
    finally:
        tmp.unlink(missing_ok=True)


def load():
    """回傳合併後的設定"""
    cfg = _defaults()
    for k, v in _raw().items():
        if k not in FIELDS:
            continue
        if FIELDS[k] == "byarch":
            # 逐架構合併，不整份取代：config.py 替 sdxl 準備的 fp16fix
            # 不該因為使用者替 anima 挑了 VAE 就消失。
            # 使用者選「模型自帶」時存的是空字串，一樣會蓋過預設值。
            cfg[k] = {**cfg.get(k, {}), **as_byarch(v)}
        elif FIELDS[k] == "decode_mode":
            cfg[k] = as_decode_mode(v)
        elif FIELDS[k] == "strmap":
            cfg[k] = as_strmap(v)
        elif FIELDS[k] == "refsources":
            cfg[k] = as_refsources(v)
        elif FIELDS[k] == "tagsets":
            cfg[k] = as_tagsets(v)
        elif FIELDS[k] == "cnunits":
            cfg[k] = as_cnunits(v)
        elif FIELDS[k] == "archparams":
            cfg[k] = as_archparams(v)
        elif FIELDS[k] == "chipoff":
            cfg[k] = as_chipoff(v)
        else:
            cfg[k] = v
    return cfg


def migrate_overrides(arch_of):
    """把舊的單一字串覆寫搬到對應架構底下（一次性）

    只有這裡知道那個字串原本是配給誰的——就是當時設定的模型的架構。
    arch_of 傳入模型路徑、回傳架構名稱。
    """
    with _WRITE_LOCK:
        cur = _raw()
        legacy = {k: v for k, v in cur.items()
                  if k in ("VAE_OVERRIDE", "TE_OVERRIDE")
                  and isinstance(v, str)}
        if not legacy:
            return None
        arch = ""
        try:
            arch = arch_of(cur.get("MODEL") or "") or ""
        except Exception:
            pass
        for k, v in legacy.items():
            cur[k] = {arch: v} if (v and arch) else {}
        _write(cur)
        return arch, sorted(legacy)


def save(data):
    """只存介面認得的欄位，並做基本型別轉換"""
    with _WRITE_LOCK:
        cur = _raw()
        for k, v in (data or {}).items():
            t = FIELDS.get(k)
            if t is None:
                continue
            try:
                if t == "int":
                    cur[k] = None if v in ("", None) else clamp(k, int(v) if isinstance(v, int) else int(float(v)))
                elif t == "float":
                    cur[k] = None if v in ("", None) else clamp(k, float(v))
                elif t == "bool":
                    cur[k] = bool(v)
                elif t == "paths":
                    cur[k] = [str(x).strip() for x in (v or [])
                              if str(x).strip()]
                elif t == "byarch":
                    # 一次只送一個架構的選擇，所以是合併不是取代，
                    # 否則替 Anima 挑 VAE 會把 SDXL 的那格洗掉
                    merged = as_byarch(cur.get(k))
                    # 空字串代表「這個架構明確要用模型自帶的」，要存下來，
                    # 不能刪掉——刪掉的話 config.py 的預設值會又冒出來
                    merged.update(as_byarch(v))
                    cur[k] = merged
                elif t == "decode_mode":
                    cur[k] = as_decode_mode(v)
                elif t == "strmap":
                    cur[k] = as_strmap(v)        # 整份取代：介面送的是改好的完整對照表
                elif t == "refsources":
                    cur[k] = as_refsources(v)
                elif t == "tagsets":
                    cur[k] = as_tagsets(v)
                elif t == "cnunits":
                    cur[k] = as_cnunits(v)
                elif t == "archparams":
                    cur[k] = as_archparams(v)
                elif t == "chipoff":
                    cur[k] = as_chipoff(v)
                elif t == "loras":
                    out = []
                    for it in (v or []):
                        p = str(it.get("path", "")).strip()
                        if p:
                            out.append({"path": p,
                                        "weight": float(
                                            it.get("weight", 1.0))})
                    cur[k] = out
                else:
                    cur[k] = "" if v is None else str(v)
            except (TypeError, ValueError):
                continue

        _write(cur)
        return cur


def get(key, fallback=None):
    return load().get(key, fallback)


def as_loras(value):
    """把介面格式轉回 [(path, weight), ...]"""
    return [(it["path"], it["weight"]) for it in (value or [])]
