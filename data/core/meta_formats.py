# -*- coding: utf-8 -*-
"""
meta_formats.py —— A1111 以外的生成參數格式

gallery_index.read_meta 讀到 PNG 文字區塊（或 EXIF）後交給這裡。回傳的欄位跟
parse_a1111 一樣（prompt、negative、steps、cfg、seed、sampler、scheduler、model、
clip_skip、loras、extra），圖庫、看大圖、比較、送到生成區都不必知道圖是哪裡來的。

  ComfyUI    PNG 的 prompt 區塊是「執行圖」：{節點 id: {class_type, inputs}}，
             inputs 的值不是常數就是 [來源節點 id, 輸出編號] 的連結。
             WebP / AVIF 存在 EXIF 0x0110，內容是 "prompt:{…}"。
             workflow 區塊是介面的版面，這裡不用。
  TensorArt  generation_data 區塊是整理好的 JSON（底下其實也跑 ComfyUI，
             另外附一份 prompt 執行圖）。seed 常是 -1，真正的值在執行圖裡。
  NovelAI    PNG 的 Software 是 "NovelAI"，Comment 是整包參數的 JSON（V4 起提示詞在
             v4_prompt，角色各自的提示詞在 char_captions）。只有 PNG 文字區塊這一種；
             另存過、文字區塊被剝掉的圖，NovelAI 藏在 alpha 通道的那份這裡不讀。
  SwarmUI    {"sui_image_params": {…}} JSON。PNG 放在 parameters 區塊（跟 A1111
             同一個名字，要先認出是 JSON），JPG 放在 EXIF 的 UserComment（UTF-16）。

節點名稱與輸入欄位對照 ComfyUI 原始碼（nodes.py、comfy_extras/nodes_custom_sampler.py）。
自訂節點五花八門，這裡用「看輸入欄位」而不是「看節點名稱」判斷，認不得的節點
只要有 model / conditioning / clip 之類的輸入就當成透傳往上游找。
"""

import json
import re

# ComfyUI 的取樣器、排程名稱 → A1111／CoCo 的寫法（comfy/samplers.py 的
# KSAMPLER_NAMES、SCHEDULER_HANDLERS）。對不上的照原樣留著。
SAMPLERS = {
    "euler": "Euler", "euler_ancestral": "Euler a", "heun": "Heun", "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a", "lms": "LMS", "dpmpp_2m": "DPM++ 2M", "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE", "dpmpp_sde": "DPM++ SDE", "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE", "dpmpp_2s_ancestral": "DPM++ 2S a", "dpmpp_3m_sde_gpu": "DPM++ 3M SDE", "ddim": "DDIM",
    "uni_pc": "UniPC", "deis": "DEIS", "sa_solver": "SA-Solver", "lcm": "LCM", "er_sde": "ER-SDE",
}
SCHEDULES = {
    "normal": "Normal", "karras": "Karras", "exponential": "Exponential", "sgm_uniform": "SGM Uniform",
    "simple": "Simple", "beta": "Beta", "ddim_uniform": "DDIM Uniform",
    "linear_quadratic": "Linear Quadratic", "kl_optimal": "KL Optimal",
}

_LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([-\d.]+))?(?::[-\d.]+)?>", re.I)


def sampler_name(name):
    return SAMPLERS.get(str(name or "").strip(), name or None)


def schedule_name(name):
    return SCHEDULES.get(str(name or "").strip(), name or None)


def _num(v, cast):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- ComfyUI
class _Graph:
    def __init__(self, nodes):
        self.nodes = {str(k): v for k, v in nodes.items() if isinstance(v, dict) and "inputs" in v}

    def link(self, v):
        """[節點 id, 輸出編號] → (節點, 輸出編號)；不是連結回 (None, None)"""
        if isinstance(v, list) and len(v) == 2 and str(v[0]) in self.nodes and isinstance(v[1], int):
            return self.nodes[str(v[0])], v[1]
        return None, None

    def value(self, v, keys, depth=0):
        """常數直接回；連結就到來源節點找 keys 裡的欄位（seed、文字常被拉成獨立的節點）"""
        node, _ = self.link(v)
        if node is None:
            return v
        if depth > 8:
            return None
        for k in keys:
            if k in node["inputs"]:
                return self.value(node["inputs"][k], keys, depth + 1)
        return None

    def cls(self, node):
        return str(node.get("class_type", ""))


_TEXT_KEYS = ("text", "text_g", "prompt", "string", "value", "text_positive", "positive_prompt", "wildcard_text", "populated_text")
_SEED_KEYS = ("seed", "noise_seed", "value", "int", "number")


# rgthree 的 Context：把九樣東西包成一條線，輸出的順序固定（看它的 workflow 定義）
_CTX = ("context", "model", "clip", "vae", "positive", "negative", "latent", "images", "seed")


def _unwrap(g, v, depth=0):
    """穿過 rgthree 的 Context／Context Switch：回傳真正的來源連結"""
    node, idx = g.link(v)
    if node is None or depth > 30:
        return v
    ins = node["inputs"]
    ctxs = [ins[k] for k in sorted(ins) if k.startswith("ctx_") and g.link(ins[k])[0]]
    if ctxs:                                   # Context Switch：第一個有接的，輸出順序不變
        return _unwrap(g, [ctxs[0][0], idx], depth + 1)
    if "Context" in g.cls(node) and 0 < idx < len(_CTX):
        key = _CTX[idx]
        if key in ins:
            return _unwrap(g, ins[key], depth + 1)
        if "base_ctx" in ins:
            return _unwrap(g, [ins["base_ctx"][0], idx], depth + 1)
    return v


def _is_sampler(n):
    ins = n["inputs"]
    return ("positive" in ins and "negative" in ins and ("latent_image" in ins or "latent" in ins)) \
        or ("guider" in ins and "sigmas" in ins and "sampler" in ins)


def _texts(g, v, seen, depth=0):
    """條件（conditioning）往上游追到文字。遇到同時有 positive / negative 的節點
    （例如 ControlNet），照連結的輸出編號決定走哪一條（0＝正、1＝負）"""
    node, idx = g.link(_unwrap(g, v))
    if node is None or depth > 30 or id(node) in seen:
        return []
    seen.add(id(node))
    ins = node["inputs"]
    # Qwen Image 這類：同一個節點輸出正、負兩組條件（輸出 0＝正、1＝負）
    if "prompt" in ins and "negative_prompt" in ins:
        key = "negative_prompt" if idx == 1 else "prompt"
        t = g.value(ins[key], _TEXT_KEYS)
        return [t] if isinstance(t, str) else []
    for k in _TEXT_KEYS:
        if k in ins and not g.link(ins[k])[0] and isinstance(ins[k], str):
            return [ins[k]]
        if k in ins and g.link(ins[k])[0] and "clip" in ins:        # 文字拉成獨立節點
            t = g.value(ins[k], _TEXT_KEYS)
            if isinstance(t, str):
                return [t]
    # 透傳：ControlNet 之類同時有 positive / negative 的，照輸出編號選一邊；
    # 合併（ConditioningCombine…）就兩邊都收
    if "positive" in ins and "negative" in ins:
        return _texts(g, ins["negative" if idx == 1 else "positive"], seen, depth + 1)
    out = []
    for k, sub in ins.items():
        if k.startswith("conditioning") and g.link(sub)[0]:
            out += _texts(g, sub, seen, depth + 1)
    return out


def _clip_skip(g, v, depth=0):
    """文字編碼器的 clip 輸入往上找 CLIPSetLastLayer（-2 就是 A1111 的 Clip skip 2）"""
    node, _ = g.link(v)
    if node is None or depth > 20:
        return None
    ins = node["inputs"]
    if "stop_at_clip_layer" in ins:
        n = _num(g.value(ins["stop_at_clip_layer"], ("value", "int")), int)
        return -n if n and n < 0 else None
    return _clip_skip(g, ins.get("clip"), depth + 1) if "clip" in ins else None


def _first_clip(g, v, seen=None, depth=0):
    """條件往上游找第一個文字編碼器的 clip 輸入"""
    seen = seen if seen is not None else set()
    node, _ = g.link(_unwrap(g, v))
    if node is None or depth > 30 or id(node) in seen:
        return None
    seen.add(id(node))
    ins = node["inputs"]
    if "clip" in ins and any(k in ins for k in _TEXT_KEYS):
        return ins["clip"]
    for k, sub in ins.items():
        if (k.startswith("conditioning") or k in ("positive", "negative")) and g.link(sub)[0]:
            c = _first_clip(g, sub, seen, depth + 1)
            if c:
                return c
    return None


def _model(g, v, groups, depth=0):
    """model 輸入往上游：一路收 LoRA（每個節點一組，groups 由外往內），停在載入模型的節點"""
    node, _ = g.link(_unwrap(g, v))
    if node is None or depth > 30:
        return None
    ins = node["inputs"]
    loras = []
    groups.append(loras)
    # model_name 不收：臉部偵測、放大模型的節點也叫這個名字
    for k in ("ckpt_name", "unet_name", "base_ckpt_name"):
        if isinstance(ins.get(k), str):
            return ins[k]
    if isinstance(ins.get("lora_name"), str):
        w = _num(g.value(ins.get("strength_model", ins.get("strength", 1)), ("value", "float")), float)
        if w != 0:
            loras.append({"file": ins["lora_name"], "weight": 1.0 if w is None else w})
    if isinstance(ins.get("text"), str) and "<lora:" in ins["text"]:        # LoraTagLoader（TensorArt）
        for name, w in _LORA_TAG.findall(ins["text"]):
            loras.append({"file": name, "weight": _num(w, float) if w else 1.0})
    for k, sub in ins.items():                                                # rgthree 的 Power Lora Loader
        if k.lower().startswith("lora_") and isinstance(sub, dict) and sub.get("on") and sub.get("lora"):
            loras.append({"file": sub["lora"], "weight": _num(sub.get("strength", 1), float) or 1.0})
    # "*" 是 smZ Settings 之類「什麼都能接」的透傳節點
    nxt = ins.get("model", ins.get("*"))
    return _model(g, nxt, groups, depth + 1) if nxt is not None else None


def _latent_origin(g, v, depth=0):
    """latent 往上游找：Empty…Latent（文生圖）還是從圖來的（圖生圖、修臉）"""
    node, _ = g.link(v)
    if node is None or depth > 20:
        return None
    if "Empty" in g.cls(node) and "Latent" in g.cls(node):
        return node
    for k in ("latent_image", "samples", "latent", "on_false", "on_true"):
        if k in node["inputs"] and g.link(node["inputs"][k])[0]:
            got = _latent_origin(g, node["inputs"][k], depth + 1)
            if got:
                return got
    return None


def parse_comfy(text):
    """ComfyUI 的 prompt 執行圖 → 參數；看不懂回 {}"""
    try:
        nodes = json.loads(text) if isinstance(text, (str, bytes)) else text
    except (TypeError, ValueError):
        return {}
    if not isinstance(nodes, dict):
        return {}
    g = _Graph(nodes)
    samplers = [n for n in g.nodes.values() if _is_sampler(n)]
    if not samplers:
        return {}

    # 修臉、高解析重繪會有好幾個取樣器：主要的那個是從空白 latent 開始、denoise 最大的
    def rank(n):
        ins = n["inputs"]
        den = _num(g.value(ins.get("denoise", 1.0), ("value", "float")), float)
        return (_latent_origin(g, ins.get("latent_image", ins.get("latent"))) is not None, den if den is not None else 1.0)
    main = max(samplers, key=rank)
    ins = main["inputs"]
    out = {"extra": {"source": "ComfyUI"}}

    pos_v, neg_v, model_v = ins.get("positive"), ins.get("negative"), ins.get("model")
    if "guider" in ins:                                       # SamplerCustomAdvanced：參數拆在好幾個節點
        guider, _ = g.link(ins["guider"])
        if guider:
            gi = guider["inputs"]
            pos_v, neg_v = gi.get("positive", gi.get("conditioning")), gi.get("negative")
            model_v = gi.get("model")
            out["cfg"] = _num(g.value(gi.get("cfg"), ("value", "float")), float)
        noise, _ = g.link(ins.get("noise"))
        if noise:
            out["seed"] = _num(g.value(noise["inputs"].get("noise_seed"), _SEED_KEYS), int)
    else:
        out["seed"] = _num(g.value(ins.get("seed", ins.get("noise_seed")), _SEED_KEYS), int)
        out["steps"] = _num(g.value(ins.get("steps"), ("value", "int")), int)
        out["cfg"] = _num(g.value(ins.get("cfg"), ("value", "float")), float)
        out["sampler"] = sampler_name(g.value(ins.get("sampler_name"), ("value",)))
        out["scheduler"] = schedule_name(g.value(ins.get("scheduler"), ("value",)))
        den = _num(g.value(ins.get("denoise"), ("value", "float")), float)
        if den is not None and den < 0.999:
            out["extra"]["Denoising strength"] = den
    # SamplerCustom／SamplerCustomAdvanced：取樣器在 KSamplerSelect、步數與排程在 BasicScheduler 之類
    smp, _ = g.link(ins.get("sampler"))
    if smp and not out.get("sampler"):
        out["sampler"] = sampler_name(g.value(smp["inputs"].get("sampler_name"), ("value",)))
    sig, _ = g.link(ins.get("sigmas"))
    if sig:
        si = sig["inputs"]
        out["scheduler"] = out.get("scheduler") or schedule_name(g.value(si.get("scheduler"), ("value",)))             or g.cls(sig).replace("Scheduler", "") or None
        out["steps"] = out.get("steps") or _num(g.value(si.get("steps"), ("value", "int")), int)
        model_v = model_v or si.get("model")

    pos = [t for t in _texts(g, pos_v, set()) if t.strip()]
    neg = [t for t in _texts(g, neg_v, set()) if t.strip()] if neg_v is not None else []
    out["prompt"] = "\n".join(dict.fromkeys(pos)).strip()
    out["negative"] = "\n".join(dict.fromkeys(neg)).strip()
    clip = _first_clip(g, pos_v)
    out["clip_skip"] = _clip_skip(g, clip) if clip else None

    groups = []
    out["model"] = _model(g, model_v, groups)
    # 從模型往外疊的順序：節點順序反過來，同一個節點裡的（LoraTagLoader 一次好幾個）照原順序
    out["loras"] = [x for grp in reversed(groups) for x in grp]

    # VAE、FluxGuidance 之類放 extra
    for n in g.nodes.values():
        ni = n["inputs"]
        if isinstance(ni.get("vae_name"), str):
            out["extra"].setdefault("VAE", ni["vae_name"])
        if "guidance" in ni and "conditioning" in ni:
            out["extra"].setdefault("Guidance", g.value(ni["guidance"], ("value", "float")))
    lat = _latent_origin(g, ins.get("latent_image", ins.get("latent")))
    if lat:
        w = _num(g.value(lat["inputs"].get("width"), ("value", "int")), int)
        h = _num(g.value(lat["inputs"].get("height"), ("value", "int")), int)
        if w and h:
            out["width"], out["height"] = w, h
    return {k: v for k, v in out.items() if v not in (None, "")} | {"prompt": out["prompt"], "negative": out["negative"]}


# ---------------------------------------------------------------- TensorArt
def parse_tensorart(text, comfy=None):
    """TensorArt 的 generation_data → 參數。comfy 是同一張圖的 prompt 執行圖（補 seed 用）"""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(d, dict) or not ("prompt" in d or "baseModel" in d):
        return {}
    graph = parse_comfy(comfy) if comfy else {}
    seed = _num(d.get("seed"), int)
    if seed is None or seed < 0:
        seed = graph.get("seed")                  # 平台常寫 -1，實際用的在執行圖裡
    base = d.get("baseModel") or {}
    loras = [{"file": m.get("modelFileName") or m.get("label") or "", "weight": _num(m.get("weight"), float) or 1.0}
             for m in d.get("models") or [] if str(m.get("type", "")).upper() == "LORA"]
    extra = {"source": "TensorArt"}
    if d.get("sdVae"):
        extra["VAE"] = d["sdVae"]
    if base.get("baseModel"):
        extra["Base model"] = base["baseModel"]
    ad = d.get("adetailer") or {}
    if ad.get("enableAdetailer"):
        extra["ADetailer"] = ", ".join(a.get("adModel", "") for a in ad.get("args") or [] if a.get("adModel")) or "on"
    clip = _num(d.get("clipSkip"), int)
    return {
        "prompt": d.get("prompt") or "",
        "negative": (d.get("negativePrompt") or "").rstrip(", "),
        "steps": _num(d.get("steps"), int),
        "cfg": _num(d.get("cfgScale"), float),
        "seed": seed,
        "sampler": d.get("samplerName") or sampler_name(d.get("ksamplerName")),
        "scheduler": schedule_name(d.get("schedule")) or graph.get("scheduler"),
        "model": base.get("modelFileName") or graph.get("model"),
        "clip_skip": clip,
        "loras": loras,
        "width": _num(d.get("width"), int), "height": _num(d.get("height"), int),
        "extra": extra,
    }


# ---------------------------------------------------------------- SwarmUI
def is_swarm(text):
    return isinstance(text, str) and text.lstrip().startswith("{") and "sui_image_params" in text


def parse_swarm(text):
    try:
        d = json.loads(text).get("sui_image_params") or {}
    except (TypeError, ValueError, AttributeError):
        return {}
    if not isinstance(d, dict) or not d:
        return {}
    names, weights = d.get("loras") or [], d.get("loraweights") or []
    loras = [{"file": n, "weight": _num(weights[i], float) if i < len(weights) else 1.0} for i, n in enumerate(names)]
    extra = {"source": "SwarmUI"}
    if d.get("vae"):
        extra["VAE"] = d["vae"]
    return {
        "prompt": d.get("prompt") or "",
        "negative": d.get("negativeprompt") or "",
        "steps": _num(d.get("steps"), int),
        "cfg": _num(d.get("cfgscale"), float),
        "seed": _num(d.get("seed"), int),
        "sampler": sampler_name(d.get("sampler")),
        "scheduler": schedule_name(d.get("scheduler")),
        "model": d.get("model"),
        "clip_skip": _num(d.get("clipstopatlayer"), int) and abs(_num(d.get("clipstopatlayer"), int)),
        "loras": loras,
        "extra": extra,
    }


def user_comment(v):
    """EXIF UserComment → 字串。開頭 8 個位元組是編碼標記：UNICODE 的是 UTF-16（SwarmUI 用），
    ASCII 或沒標記的當 UTF-8"""
    if isinstance(v, str):
        return v
    if not isinstance(v, (bytes, bytearray)):
        return ""
    head, body = bytes(v[:8]), bytes(v[8:])
    if head.startswith(b"UNICODE"):
        # 位元組順序沒有標記：SwarmUI 是小端、A1111（piexif）是大端。參數裡大多是 ASCII
        # 的標點與欄位名，ASCII 字元在小端的奇數位、大端的偶數位是 0，數整段就分得出來。
        # 不能只看開頭兩個位元組——開頭是中文時兩邊都不是 0
        if body[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return body.decode("utf-16", "ignore")
        odd = body[1::2].count(0)
        even = body[0::2].count(0)
        return body.decode("utf-16-le" if odd > even else "utf-16-be", "ignore")
    if head.startswith(b"ASCII") or head == b"\x00" * 8:
        return body.decode("utf-8", "ignore")
    return bytes(v).decode("utf-8", "ignore")


# ---------------------------------------------------------------- NovelAI
def is_novelai(info):
    return str(info.get("Software", "")).strip() == "NovelAI" and bool(info.get("Comment"))


def _captions(block):
    """v4_prompt／v4_negative_prompt → (整體的提示詞, [角色 1, 角色 2…])"""
    cap = (block or {}).get("caption") or {}
    chars = [c.get("char_caption", "") for c in cap.get("char_captions") or [] if isinstance(c, dict)]
    return cap.get("base_caption") or "", [c for c in chars if c.strip()]


def parse_novelai(info):
    """NovelAI 的 PNG 文字區塊 → 參數"""
    try:
        d = json.loads(info.get("Comment") or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    base, chars = _captions(d.get("v4_prompt"))
    nbase, nchars = _captions(d.get("v4_negative_prompt"))
    # 角色各自的提示詞接在後面，圖庫的全文搜尋才找得到（看大圖也看得到）
    prompt = "\n\n".join(x for x in [base or d.get("prompt") or info.get("Description") or "", *chars] if x)
    negative = "\n\n".join(x for x in [d.get("uc") or nbase, *nchars] if x)
    smp = str(d.get("sampler") or "")
    extra = {"source": "NovelAI"}
    if d.get("model_hash"):
        extra["Model hash"] = d["model_hash"]
    if _num(d.get("cfg_rescale"), float):
        extra["CFG rescale"] = d["cfg_rescale"]
    if d.get("sm"):
        extra["SMEA"] = "DYN" if d.get("sm_dyn") else "on"
    if chars:
        extra["Characters"] = len(chars)
    return {
        "prompt": prompt,
        "negative": negative,
        "steps": _num(d.get("steps"), int),
        "cfg": _num(d.get("scale"), float),
        "seed": _num(d.get("seed"), int),
        "sampler": sampler_name(smp[2:] if smp.startswith("k_") else smp),
        "scheduler": schedule_name(d.get("noise_schedule")),
        "model": d.get("model_name") or info.get("Source"),
        "loras": [],
        "extra": extra,
    }
