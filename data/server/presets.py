# -*- coding: utf-8 -*-
"""品質標籤與預設負面提示詞：生成時自動加上去（NovelAI 的「Add Quality Tags」「UC Preset」那種）。

不同模型要的品質標籤差很多（Illustrious 的 masterpiece, best quality、Anima／Pony 的 score_*），
所以跟著模型走：模型自己有設就用自己的，沒有就用它那個架構的預設。存在 presets.json：

    {"arch":  {"sdxl": {"quality": "...", "negative": "...", "position": "after"}, ...},
     "model": {"<模型的 res_stem>": {"mode": "custom" | "none", "quality": ..., "negative": ..., "position": ...}}}

套用時只加提示詞裡還沒有的標籤（比對時忽略大小寫、底線與空格、權重括號），
Recipe 自帶的負面、舊提示詞裡本來就寫的 masterpiece 都不會重複。
"""

import json
import re
import threading
from pathlib import Path

from .state import C, res_stem

from i18n import tr

PRESETS_FILE = getattr(C, "PRESETS_FILE", None) or (Path(C.DB_PATH).parent / "presets.json")
_LOCK = threading.Lock()
POSITIONS = ("before", "after")

# 第一次沒有 presets.json 時的架構預設，之後都在資源庫的模型詳細裡改。
# SDXL：每個模型的官方建議不同（Illustrious、NoobAI、Pony），用圖庫裡實際出現在七成以上的圖的標籤（2026-09-28 統計 993 張）。
# Anima：官方模型頁給 Base、Turbo 的建議（circlestone-labs/Anima），分級標籤 safe 不放（使用者決定：新使用者畫 NSFW
# 要先拿掉、跟提示詞寫的 nsfw 互相抵消）。Aesthetic 版官方說不要用 score 標籤，下載試用模型時另外設（AESTHETIC）。
# Z-Image、Krea 2、Qwen Image 2.1 官方沒有品質標籤（用一般句子描述），不設
DEFAULTS = {
    "sdxl": {"quality": "masterpiece, best quality, amazing quality",
             "negative": "worst quality, bad quality, worst detail, censor", "position": "before"},
    "anima": {"quality": "masterpiece, best quality, score_7",
              "negative": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, "
                          "chromatic aberration", "position": "before"},
}
# Anima Aesthetic：官方建議去掉 score 標籤（已經調整過畫質）
AESTHETIC = {"quality": "masterpiece, best quality",
             "negative": "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration",
             "position": "before"}


def _load():
    try:
        d = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            return {"arch": dict(d.get("arch") or {}), "model": dict(d.get("model") or {})}
    except (OSError, ValueError):
        pass
    return {"arch": json.loads(json.dumps(DEFAULTS)), "model": {}}


def _save(d):
    tmp = PRESETS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(PRESETS_FILE)


def _clean(p):
    p = p if isinstance(p, dict) else {}
    pos = p.get("position") if p.get("position") in POSITIONS else "before"
    return {"quality": str(p.get("quality") or "").strip()[:2000],
            "negative": str(p.get("negative") or "").strip()[:4000], "position": pos}


def all_presets():
    d = _load()
    return {"arch": {k: _clean(v) for k, v in d["arch"].items()},
            "model": {k: {**_clean(v), "mode": "none" if v.get("mode") == "none" else "custom"}
                      for k, v in d["model"].items()}}


def resolve(model, arch):
    """這個模型實際要用的：回傳 {quality, negative, position, source}，source 是 model / arch / none"""
    d = _load()
    m = d["model"].get(res_stem(model or ""))
    if m:
        if m.get("mode") == "none":
            return {"quality": "", "negative": "", "position": "before", "source": "none"}
        return {**_clean(m), "source": "model"}
    a = d["arch"].get(arch or "")
    if a:
        return {**_clean(a), "source": "arch"}
    return {"quality": "", "negative": "", "position": "before", "source": "none"}


def set_default(model, preset):
    """模型還沒有自己的設定才設（使用者改過、關掉的不動）。回傳有沒有設"""
    with _LOCK:
        if res_stem(model) in _load()["model"]:
            return False
    update("model", model, {**preset, "mode": "custom"})
    return True


def update(scope, key, data):
    """scope="arch"：key 是架構；scope="model"：key 是模型路徑或名稱，data.mode 是
    custom / none / inherit（inherit＝拿掉自訂、回到架構預設）"""
    if scope not in ("arch", "model") or not str(key or "").strip():
        return {"error": tr("srv.badParams")}
    with _LOCK:
        d = _load()
        if scope == "arch":
            d["arch"][str(key)] = _clean(data)
        else:
            k = res_stem(key)
            mode = (data or {}).get("mode", "custom")
            if mode == "inherit":
                d["model"].pop(k, None)
            else:
                d["model"][k] = {**_clean(data), "mode": "none" if mode == "none" else "custom"}
        _save(d)
    return {"ok": True, **all_presets()}


# ---------------------------------------------------------------- 套用
def _split(text):
    """依最外層的逗號與換行切（括號、<lora:…> 裡的逗號不算）"""
    out, buf, depth = [], "", 0
    for ch in text or "":
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch in ",\n" and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    out.append(buf)
    return [x.strip() for x in out if x.strip()]


def _key(tag):
    """比對用：(masterpiece:1.2)、Best_Quality、[best quality] 都算同一個"""
    t = re.sub(r":\s*-?[\d.]+\s*\)?$", "", tag.strip())
    t = re.sub(r"[()\[\]{}]", "", t).replace("_", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def missing(text, preset_text):
    """preset_text 裡、text 還沒有的標籤（照 preset_text 的順序）"""
    have = {_key(t) for t in _split(text)}
    return [t for t in _split(preset_text) if _key(t) not in have]


def _join(base, tags, position):
    if not tags:
        return base
    add = ", ".join(tags)
    base = (base or "").strip()
    if not base:
        return add
    if position == "before":
        return f"{add}, {base}"
    # 提示詞最後一行是 # 註解的話，接在同一行會變成註解的一部分，要換行
    sep = "\n" if base.rsplit("\n", 1)[-1].lstrip().startswith("#") else ", "
    return base.rstrip(", ") + sep + add


def apply(prompt, negative, preset):
    """回傳 (完整提示詞, 完整負面, 實際加上的 {quality, negative})"""
    q = missing(prompt, preset.get("quality", ""))
    n = missing(negative, preset.get("negative", ""))
    full_p = _join(prompt, q, preset.get("position", "before"))
    full_n = _join(negative, n, "after")
    return full_p, full_n, {"quality": ", ".join(q), "negative": ", ".join(n)}
