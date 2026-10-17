# -*- coding: utf-8 -*-
"""
flux1.py —— FLUX.1（Black Forest Labs 的 12B：dev、schnell，以及以它們為底的微調）

模型檔有兩種，都是 BFL 原始命名（double_blocks、single_blocks、img_in、vector_in…）：
  只有主模型（ComfyUI 的 diffusion_models）        VAE 與文字編碼器要另外選
  全包（ComfyUI 的 checkpoints，CivitAI 上很常見）  主模型在 model.diffusion_model.、T5 與 CLIP-L 在 text_encoders.、
                                                    VAE 在 vae.（有些是 first_stage_model.）
零件：
  VAE          FLUX.1 的 VAE（ae.safetensors，16 個 latent 通道；Z-Image 用的就是它）
  文字編碼器   T5-XXL（t5xxl_fp16 或 fp8 版），在生成台的「文字編碼器」欄選
  CLIP-L       clip_l.safetensors，不佔欄位：模型檔自帶的、或資源庫裡找到的（server/assets.clip_for）
沒選的零件生成前自動找資源庫裡相符的（assets.fill_parts）；找不到但模型檔自帶的，用模型檔裡的。
架構代號：flux1（有 guidance 輸入的：dev、Krea dev 與大部分微調）、flux1s（schnell）、kontext（Kontext dev：
權重的形狀跟 dev 一模一樣、官方的檔也沒有中繼資料，只能看檔名有沒有 kontext）。
Fill、Canny、Depth 的輸入通道不同、Chroma 拿掉了 CLIP 與每層的調變，都認成 other。

diffusers 0.39 有 FluxPipeline 與單檔的鍵名轉換；transformer、T5、CLIP 的設定從權重形狀推，不用連網。

**fp8。** 常見的 fp8 檔是把 bf16 直接轉成 float8_e4m3fn（沒有比例，連 bias、norm 都轉），也有 ComfyUI 的 scaled fp8
（每個矩陣一個比例：舊的 scale_weight、新的 weight_scale）。線性層照原樣用 fp8 存放、算的時候還原（沒有比例的當 1，
Krea 2 那套 ScaledFP8Linear），其他的還原成 bf16。T5 的 fp16 檔也照原樣存放、算的時候轉 bf16（直接映射檔案，
不複製 9 GB 進系統記憶體），跟載入時就轉成 bf16 逐位元相同。

**guidance。** dev、Kontext 是蒸餾過 guidance 的模型：介面的 CFG 欄就是官方的 guidance（dev 3.5、Kontext 2.5），
每一步只跑一次、不看負面提示詞（Forge 的 Distilled CFG、ComfyUI 的 FluxGuidance 也是這樣）。schnell 沒有 guidance 輸入，
CFG 大於 1 時照一般的 CFG 跑負面提示詞（官方管線的 true_cfg_scale）。

**排程。** dev 的 σ 依「latent 格數」平移（官方 calculate_shift：256 格 0.5、4096 格 1.15，指數平移）；schnell 不平移。
CoCo 自己算好，用 shift 1 的排程器跑（flux2 那套 _use_sigmas），跟官方逐位元相同。
介面的 shift：dev、Kontext 是「1024² 時」的值（預設 3.16＝官方），schnell 是固定值（預設 1＝官方）。

**用指示修圖（Kontext）。** 精修的「用指示修圖」跟 Qwen Image 2.1、klein 共用：正在修的這張交給官方的 FluxKontextPipeline
（縮到最接近的官方解析度、VAE 編碼後接在畫的這張後面）。官方只收一張，其他參考圖照 ComfyUI 官方工作流程的做法
（Image Stitch）高度對齊接在右邊（edit_images）。σ 照畫的這張的大小算（官方也是）。
"""

import math
import re
import time
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
CLIP_TOKENIZER = ROOT / "assets" / "flux1" / "tokenizer"        # openai/clip-vit-large-patch14 的 tokenizer
T5_TOKENIZER = ROOT / "assets" / "anima" / "tokenizer_2"        # google/t5-v1_1-xxl 的 tokenizer（Anima 用的同一份）

PREFIX = "model.diffusion_model."
T5_PREFIX = "text_encoders.t5xxl.transformer."
CLIP_PREFIX = "text_encoders.clip_l.transformer."
VAE_PREFIXES = ("vae.", "first_stage_model.")

CLIP_TOKENS = 77
T5_TOKENS = {"flux1": 512, "flux1s": 256, "kontext": 512}   # BFL 官方：schnell 256、dev 512（補到這個長度，超過的截掉）
MULTIPLE = 16                                       # 寬高要 16 的倍數（VAE 縮小 8 倍、再 2×2 打包）

TITLE = {"flux1": "FLUX.1 dev", "flux1s": "FLUX.1 schnell", "kontext": "FLUX.1 Kontext"}
# dev 官方範例 50 步，ComfyUI 的範例 20 步；schnell 官方 4 步；Kontext 官方範例 28 步
DEFAULT_STEPS = {"flux1": 20, "flux1s": 4, "kontext": 28}
DEFAULT_CFG = {"flux1": 3.5, "flux1s": 1.0, "kontext": 2.5}    # guidance：dev 3.5、Kontext 2.5（官方）；schnell：不跑 CFG
SEQ_1024 = (1024 // 16) ** 2


def official_mu(seq):
    """官方的 calculate_shift（diffusers 的 pipeline_flux，預設值照 FLUX.1-dev 的 scheduler_config，運算順序照抄）"""
    base_seq_len, max_seq_len, base_shift, max_shift = 256, 4096, 0.5, 1.15
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return seq * m + b


SHIFT = {"flux1": round(math.exp(official_mu(SEQ_1024)), 2), "flux1s": 1.0}      # 介面的預設值（dev ≈ 3.16）
SHIFT["kontext"] = SHIFT["flux1"]                   # Kontext 的排程設定跟 dev 一樣


# ---------------------------------------------------------------- 認檔

def _strip(k):
    return k[len(PREFIX):] if k.startswith(PREFIX) else k


def is_flux1(keys):
    """FLUX.1 的主模型（只有主模型或全包）：每個雙流層有自己的 img_mod、有 vector_in（CLIP 的向量）。
    也收 diffusers 命名的。FLUX.2 的調變全模型共用、沒有 vector_in；Chroma 沒有 vector_in"""
    ks = {_strip(k) for k in keys}
    return ("double_blocks.0.img_mod.lin.weight" in ks and "vector_in.in_layer.weight" in ks) \
        or ("transformer_blocks.0.norm1.linear.weight" in ks and "time_text_embed.text_embedder.linear_1.weight" in ks
            and "single_transformer_blocks.0.proj_mlp.weight" in ks)


def arch_of(keys, shape, name=""):
    """flux1（有 guidance 輸入）／flux1s（schnell）／kontext（有 guidance 輸入、檔名有 kontext）；
    輸入不是 64 個通道的（Fill、Canny、Depth）回 other。shape(鍵) → 那個張量的形狀；name：檔名"""
    ks = {_strip(k): k for k in keys}
    for key in ("img_in.weight", "x_embedder.weight"):
        if key in ks and list(shape(ks[key]))[-1] != 64:
            return "other"
    guided = any(k.startswith(("guidance_in.", "time_text_embed.guidance_embedder.")) for k in ks)
    if not guided:
        return "flux1s"
    return "kontext" if "kontext" in str(name).lower() else "flux1"


def builtin_parts(keys):
    """全包的檔裡自帶哪些零件：{"vae", "te"（T5）, "clip"} → True／False"""
    return {"vae": any(k.startswith(VAE_PREFIXES) for k in keys),
            "te": any(k.startswith(T5_PREFIX) for k in keys),
            "clip": any(k.startswith(CLIP_PREFIX) for k in keys)}


# ---------------------------------------------------------------- 鍵名

# BFL 原始的線性層 → diffusers 的模組（合起來的 qkv、單流的 linear1 對到拆開的好幾個）。轉換權重用 diffusers 的
# convert_flux_transformer_checkpoint_to_diffusers；這張表給 fp8 的比例、LoRA 對名字（tests/test_flux1.py 核對過兩邊一致）
_TOP = {"img_in": "x_embedder", "txt_in": "context_embedder",
        "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
        "final_layer.linear": "proj_out"}
# diffusers 把最後一層的調變前後兩半對調了：比例（一個純量）照樣對，LoRA 照原本的順序加上去會加錯，不放進 LoRA 的表
_SWAPPED = {"final_layer.adaLN_modulation.1": "norm_out.linear"}
_DOUBLE = {"img_mod.lin": ("norm1.linear",), "txt_mod.lin": ("norm1_context.linear",),
           "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"), "img_attn.proj": ("attn.to_out.0",),
           "img_mlp.0": ("ff.net.0.proj",), "img_mlp.2": ("ff.net.2",),
           "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"), "txt_attn.proj": ("attn.to_add_out",),
           "txt_mlp.0": ("ff_context.net.0.proj",), "txt_mlp.2": ("ff_context.net.2",)}
_SINGLE = {"linear1": ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"), "linear2": ("proj_out",),
           "modulation.lin": ("norm.linear",)}


def module_names(name, swapped=False):
    """BFL 的線性層名稱 → diffusers 的模組路徑（tuple；認不得回空的）。swapped：連前後對調的那層也對（比例用）"""
    if name in _TOP:
        return (_TOP[name],)
    if swapped and name in _SWAPPED:
        return (_SWAPPED[name],)
    parts = name.split(".")
    if len(parts) >= 3 and parts[1].isdigit():
        rest = ".".join(parts[2:])
        if parts[0] == "double_blocks" and rest in _DOUBLE:
            return tuple(f"transformer_blocks.{parts[1]}.{m}" for m in _DOUBLE[rest])
        if parts[0] == "single_blocks" and rest in _SINGLE:
            return tuple(f"single_transformer_blocks.{parts[1]}.{m}" for m in _SINGLE[rest])
    return ()


# ComfyUI 量化的附屬資料：用不到（scale_input／input_scale 是輸入的比例，CoCo 照 Krea 2 的算法；scaled_fp8 只是標記）
_DROP = (".scale_input", ".input_scale", ".comfy_quant")


def _quant_keys(sd, what):
    """整理量化的附屬資料：舊的 scale_weight 改名 weight_scale，用不到的丟掉；4 位元、bitsandbytes 的說清楚不支援"""
    if any(k.endswith(".weight_scale_2") for k in sd):
        raise RuntimeError(i18n.tr("eng.f1.nvfp4", what=what))
    if any(k.endswith((".absmax", ".quant_map", ".nested_absmax")) or ".quant_state." in k for k in sd):
        raise RuntimeError(i18n.tr("eng.f1.nf4", what=what))
    out = {}
    for k, v in sd.items():
        if k == "scaled_fp8" or k.endswith(_DROP):
            continue
        if k.endswith(".scale_weight"):
            k = k[:-len(".scale_weight")] + ".weight_scale"
        out[k] = v
    return out


def _unit_scales(sd, is_linear):
    """沒有比例的 fp8 線性層（直接把 bf16 轉成 fp8 存的檔）補上比例 1：照 fp8 存放、算的時候還原，不先還原成 bf16
    （主模型 12 GB 會變成 24 GB）。還原就是轉型，跟 ComfyUI 的算法一樣"""
    import torch
    fp8 = _krea2()._fp8_dtypes()
    for k, v in list(sd.items()):
        if k.endswith(".weight") and v.dtype in fp8 and v.dim() == 2 and is_linear(k[:-len(".weight")]):
            sd.setdefault(k[:-len(".weight")] + ".weight_scale", torch.ones((), dtype=torch.float32))
    return sd


def convert(raw):
    """主模型 → diffusers 的鍵名。fp8 的比例跟著權重改名，合起來的 qkv、linear1 拆開時比例照抄（同一個純量）"""
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    sd = _quant_keys({_strip(k): v for k, v in raw.items()}, i18n.tr("eng.lbl.main"))
    if "x_embedder.weight" in sd:                               # 已經是 diffusers 的命名
        return _unit_scales(sd, lambda m: True)
    scales = {k: sd.pop(k) for k in [k for k in sd if k.endswith(".weight_scale")]}
    out = convert_flux_transformer_checkpoint_to_diffusers(sd)
    if sd:
        raise RuntimeError(i18n.tr("eng.f1.unknownTensors", n=len(sd), k=next(iter(sd))))
    for k, v in scales.items():
        names = module_names(k[:-len(".weight_scale")], swapped=True)
        if not names:
            raise RuntimeError(i18n.tr("eng.f1.noModule", k=k))
        for n in names:
            out[n + ".weight_scale"] = v
    return _unit_scales(out, lambda m: True)


def _count(sd, prefix):
    idx = {int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)}
    return max(idx) + 1 if idx else 0


def _transformer_config(sd):
    inner, cin = sd["x_embedder.weight"].shape
    head = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    return {"patch_size": 1, "in_channels": cin, "out_channels": None,
            "num_layers": _count(sd, "transformer_blocks."), "num_single_layers": _count(sd, "single_transformer_blocks."),
            "attention_head_dim": head, "num_attention_heads": inner // head,
            "joint_attention_dim": sd["context_embedder.weight"].shape[1],
            "pooled_projection_dim": sd["time_text_embed.text_embedder.linear_1.weight"].shape[1],
            "guidance_embeds": any(k.startswith("time_text_embed.guidance_embedder.") for k in sd),
            "axes_dims_rope": (16, 56, 56)}


# ---------------------------------------------------------------- 載入

def _z():
    try:
        from core import zimage
    except ImportError:
        import zimage
    return zimage


def _krea2():
    try:
        from core import krea2
    except ImportError:
        import krea2
    return krea2


def _q21():
    try:
        from core import qwen21
    except ImportError:
        import qwen21
    return qwen21


def _flux2():
    try:
        from core import flux2
    except ImportError:
        import flux2
    return flux2


_PIPE = None


def _pipe_class():
    """跟 Z-Image 一樣：文字編碼器平常在系統記憶體，diffusers 挑到它會整條判成 CPU。有顯示卡就一律在顯示卡上算"""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import FluxPipeline

        class CocoFluxPipeline(FluxPipeline):
            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _PIPE = CocoFluxPipeline
    return _PIPE


def _part(raw, prefix):
    """全包檔裡的一個零件：鍵名去掉前綴（prefix 空字串＝整個檔都是）"""
    if not prefix:
        return dict(raw)
    return {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}


def _header(path):
    import json
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))


def _dtype(path, key, name):
    import torch
    t = _z()._DTYPES.get(name)
    if t is None:
        raise RuntimeError(i18n.tr("eng.badDtype", name=Path(path).name, k=key, dtype=name))
    return getattr(torch, t)


# 一次讀多少（Windows 的 ReadFile 一次最多 4 GB 不到）
READ_CHUNK = 256 * 2 ** 20


def read_part(path, prefix="", skip=()):
    """鍵名是 prefix 開頭（skip 開頭的不要）的張量讀進系統記憶體，鍵名去掉前綴。
    不用 safetensors 的 get_tensor：它把整個檔案用 copy-on-write 映射，Windows 會先預留整個檔案大小的承諾額度
    （全包的檔 17 GB），其他程式開著時就不夠（2026-10-09 實測：額度只剩 24.5 GB，讀 CLIP 時整個程式當掉）。
    這裡一個張量一個張量讀檔案，只佔讀進來的大小"""
    import torch
    base, header = _header(path)
    out = {}
    with open(path, "rb", buffering=0) as fh:
        for k, v in header.items():
            if k == "__metadata__" or not k.startswith(prefix) or (skip and k.startswith(skip)):
                continue
            t = torch.empty(v["shape"], dtype=_dtype(path, k, v["dtype"]))
            a, b = v["data_offsets"]
            buf = t.reshape(-1).view(torch.uint8).numpy()
            if len(buf) != b - a:
                raise RuntimeError(i18n.tr("eng.f1.badSize", name=Path(path).name, k=k))
            fh.seek(base + a)
            at = 0
            while at < len(buf):
                got = fh.readinto(memoryview(buf)[at:at + READ_CHUNK])
                if not got:
                    raise RuntimeError(i18n.tr("eng.f1.truncated", name=Path(path).name))
                at += got
            out[k[len(prefix):]] = t
    return out


def map_part(path, prefix=""):
    """鍵名是 prefix 開頭的張量直接映射檔案（不複製），只映射用得到的那一段，而且唯讀：
    唯讀的映射不佔 Windows 的承諾額度（copy-on-write 的要預留整段的大小，見 read_part）。
    寫不進去，所以映射的權重不能就地改：T5 的線性層一律照原樣存放、LoRA 掛在旁邊（t5_from）。
    回傳 ({鍵（去掉前綴）: 張量}, mmap)"""
    import mmap
    import warnings
    import torch
    base, header = _header(path)
    items = [(k, v) for k, v in header.items() if k != "__metadata__" and k.startswith(prefix)]
    spans = [v["data_offsets"] for _, v in items if v["data_offsets"][1] > v["data_offsets"][0]]
    if not spans:
        return {}, None
    lo, hi = base + min(a for a, _ in spans), base + max(b for _, b in spans)
    start = lo - lo % mmap.ALLOCATIONGRANULARITY
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), hi - start, access=mmap.ACCESS_READ, offset=start)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # 「buffer 不能寫」：本來就不寫
        for k, v in items:
            dt = _dtype(path, k, v["dtype"])
            a, b = v["data_offsets"]
            out[k[len(prefix):]] = torch.empty(v["shape"], dtype=dt) if a == b else                 torch.frombuffer(mm, dtype=dt, count=(b - a) // dt.itemsize, offset=base + a - start).view(v["shape"])
    return out, mm


def build_stored(cls, cfg, sd, dtype, label, log, is_linear, store_all=False):
    """在 meta 上建好、權重直接指派（anima._build）。線性層的權重不是計算精度的（fp8、fp16）照原樣存放、
    算的時候還原（krea2 的 ScaledFP8Linear，比例預設 1），其他的照 krea2.prepare_weights 轉成計算精度。
    store_all：計算精度的線性層也這樣存（權重是唯讀映射的：LoRA 掛在旁邊、不就地改）。
    回傳 (模型, 有沒有量化存放)"""
    try:
        from core.anima import _build
    except ImportError:
        from anima import _build
    K = _krea2()
    scales = {k[:-len(".weight_scale")]: sd.pop(k) for k in [k for k in sd if k.endswith(".weight_scale")]}
    stored, keep = {}, {}
    for k in list(sd):
        mod = k[:-len(".weight")] if k.endswith(".weight") else None
        v = sd[k]
        if mod is None or v.dim() != 2 or not is_linear(mod) or not v.is_floating_point():
            continue
        s = scales.get(mod)
        if v.dtype == dtype and s is None and not store_all:
            continue
        if s is not None and s.numel() != 1:
            continue                    # 每列一個比例：交給 prepare_weights 還原
        stored[k], keep[mod] = sd.pop(k), float(s) if s is not None else 1.0
        scales.pop(mod, None)
    sd.update({m + ".weight_scale": s for m, s in scales.items()})
    weights, more = K.prepare_weights(sd, dtype)
    weights.update(stored)
    keep.update(more)
    model = _build(cls, cfg, weights, label, log, prepare=lambda m: K.swap_fp8(m, keep, weights))
    if keep:
        kinds = sorted({str(w.dtype).replace("torch.", "") for k, w in weights.items()
                        if k.endswith(".weight") and k[:-len(".weight")] in keep})
        log(label + "　" + i18n.tr("eng.f1.storedAsIs", n=len(keep), kinds=i18n.join(kinds)))
    return model, bool(keep)


def _t5_linear(mod):
    return mod.split(".")[-1] in ("q", "k", "v", "o", "wi_0", "wi_1", "wo", "wi")


def t5_weights(raw, prefix=""):
    """T5 檔（或全包檔裡的那段）→ T5EncoderModel 的鍵名。解碼器、lm_head 用不到；
    embed_tokens 跟 shared 是同一份（官方的 tie）"""
    sd = _quant_keys({k: v for k, v in _part(raw, prefix).items() if not k.startswith(("decoder.", "lm_head."))}, "T5")
    if "shared.weight" not in sd and "encoder.embed_tokens.weight" in sd:
        sd["shared.weight"] = sd["encoder.embed_tokens.weight"]
    if "shared.weight" in sd:
        sd["encoder.embed_tokens.weight"] = sd["shared.weight"]
    return sd


def _t5_config(sd):
    from transformers import T5Config
    attn = "encoder.block.0.layer.0.SelfAttention."
    d_ff, d_model = sd["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape
    buckets, heads = sd[attn + "relative_attention_bias.weight"].shape
    return T5Config(vocab_size=sd["shared.weight"].shape[0], d_model=d_model, d_kv=sd[attn + "q.weight"].shape[0] // heads,
                    d_ff=d_ff, num_layers=_count(sd, "encoder.block."), num_heads=heads,
                    relative_attention_num_buckets=buckets, relative_attention_max_distance=128, dropout_rate=0.1,
                    layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu", pad_token_id=0, eos_token_id=1,
                    decoder_start_token_id=0, tie_word_embeddings=False, use_cache=False)


def t5_from(path, prefix, dtype, log=print, width=None):
    """T5-XXL 的編碼器。權重直接映射檔案，不另外讀進系統記憶體；fp8、fp16 的線性層照原樣存放（build_stored）。
    width：主模型要的文字寬度（txt_in 的輸入，FLUX.1 是 4096）。編碼時邊算邊搬（place）"""
    from transformers import T5EncoderModel
    raw, mm = map_part(path, prefix)
    sd = t5_weights(raw)
    del raw
    name = Path(path).name
    if "encoder.block.0.layer.1.DenseReluDense.wi_0.weight" not in sd or "shared.weight" not in sd:
        raise RuntimeError(i18n.tr("eng.f1.notT5", name=name))
    vocab = sd["shared.weight"].shape[0]
    if vocab != 32128:
        raise RuntimeError(i18n.tr("eng.f1.vocab", name=name, n=vocab))
    d_model = sd["shared.weight"].shape[1]
    if width and d_model != width:
        raise RuntimeError(i18n.tr("eng.f1.width", name=name, d=d_model, w=width))
    te, _ = build_stored(T5EncoderModel, _t5_config(sd), sd, dtype, "　T5", log, _t5_linear, store_all=True)
    te.encoder.embed_tokens.weight = te.shared.weight         # 同一份，不用兩份（fp16 的檔轉 bf16 時會各複製一份）
    ff = _t5_ff()
    for m in te.modules():
        if type(m).__name__ == "T5DenseGatedActDense":
            m.__class__ = ff
    # 檔案路徑不給：映射的只是檔案的一段，zimage._up 照位址讀檔案會讀錯地方（T5 一律邊算邊搬，不會整個搬）
    te._coco_mapped, te._coco_file = True, (mm, None)
    return te.eval()


_FF = None


def _t5_ff():
    """T5 的 MLP：transformers 會把中間結果轉成 wo 權重的型別（為了 flan-t5 的 wo 留 fp32），wo 照原樣用 fp8、fp16
    存放時就變成用 fp8、fp16 算（fp8 整個壞掉，fp16 會溢位）。官方的權重是 bf16、本來就不會轉，所以拿掉這一步，
    跟官方逐位元相同"""
    global _FF
    if _FF is None:
        from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

        class CocoT5DenseGatedActDense(T5DenseGatedActDense):
            def forward(self, hidden_states):
                h = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
                return self.wo(self.dropout(h))

        _FF = CocoT5DenseGatedActDense
    return _FF


def clip_weights(raw, prefix=""):
    """CLIP-L 檔 → transformers 5 的 CLIPTextModel 鍵名（少了外面那層 text_model）。投影層、logit_scale、
    position_ids（不存檔的緩衝區）用不到"""
    out = {}
    for k, v in _part(raw, prefix).items():
        if k.startswith("text_model."):
            k = k[len("text_model."):]
        if k.startswith(("embeddings.token_embedding.", "embeddings.position_embedding.", "encoder.", "final_layer_norm.")):
            out[k] = v
    return _quant_keys(out, "CLIP-L")


def _clip_config(sd):
    """官方 FLUX.1 的 text_encoder/config.json（openai/clip-vit-large-patch14 的文字部分）"""
    from transformers import CLIPTextConfig
    vocab, hidden = sd["embeddings.token_embedding.weight"].shape
    return CLIPTextConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=sd["encoder.layers.0.mlp.fc1.weight"].shape[0],
                          num_hidden_layers=_count(sd, "encoder.layers."), num_attention_heads=hidden // 64,
                          max_position_embeddings=sd["embeddings.position_embedding.weight"].shape[0],
                          hidden_act="quick_gelu", layer_norm_eps=1e-5, projection_dim=768,
                          bos_token_id=0, eos_token_id=2, pad_token_id=1)


def clip_from(path, prefix, dtype, log=print):
    """CLIP-L（只用它的整句向量 pooler_output）。246 MB，轉成 bf16 常駐顯示卡"""
    from transformers import CLIPTextModel
    raw = read_part(path, prefix)
    sd = clip_weights(raw)
    del raw
    emb = sd.get("embeddings.token_embedding.weight")
    if emb is None:
        raise RuntimeError(i18n.tr("eng.f1.notClip", name=Path(path).name))
    if emb.shape[1] != 768:
        raise RuntimeError(i18n.tr("eng.f1.clipWidth", name=Path(path).name, d=emb.shape[1]))
    te, _ = build_stored(CLIPTextModel, _clip_config(sd), sd, dtype, "　CLIP-L", log, lambda m: False)
    return te.eval()


def load_vae(path, prefix, dtype, log=print):
    """FLUX.1 的 VAE（跟 Z-Image 同一個，設定照 zimage.VAE）：BFL 原始命名照 convert_ldm_vae_checkpoint 轉"""
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    try:
        from core.anima import _build
    except ImportError:
        from anima import _build
    Z = _z()
    raw = read_part(path, prefix)
    vsd = raw if "encoder.down_blocks.0.resnets.0.conv1.weight" in raw else convert_ldm_vae_checkpoint(raw, Z.VAE)
    conv = vsd.get("decoder.conv_in.weight")
    if conv is None or conv.shape[1] != 16 or "bn.running_mean" in vsd:
        raise RuntimeError(i18n.tr("eng.f1.notVae", name=Path(path).name))
    return _build(AutoencoderKL, dict(Z.VAE), {k: v.to(dtype) for k, v in vsd.items()}, "　vae", log)


def _source(chosen, model_path, has, prefix, what, need):
    """零件從哪裡讀：(路徑, 前綴)。有選的用選的，沒有就用模型檔自帶的"""
    if chosen:
        if not Path(chosen).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=chosen))
        return str(chosen), ""
    if has:
        return str(model_path), prefix
    raise RuntimeError(need)


def load(model_path, vae_path, te_path, clip_path=None, log=print, dtype=None, fp8=False):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from transformers import CLIPTokenizer, T5TokenizerFast
    K = _krea2()

    dtype = dtype or torch.bfloat16
    _, header = _header(model_path)
    keys = [k for k in header if k != "__metadata__"]
    arch = arch_of(keys, lambda k: header[k]["shape"], Path(model_path).name)
    if arch not in TITLE:
        raise RuntimeError(i18n.tr("eng.f1.fillUnsupported"))
    title = TITLE[arch]
    has = builtin_parts(keys)
    vae_src = _source(vae_path, model_path, has["vae"], next((p for p in VAE_PREFIXES if any(k.startswith(p) for k in keys)), ""),
                      "VAE", i18n.tr("eng.f1.needVae"))
    t5_src = _source(te_path, model_path, has["te"], T5_PREFIX, i18n.tr("eng.teCap"), i18n.tr("eng.f1.needT5"))
    clip_src = _source(clip_path, model_path, has["clip"], CLIP_PREFIX, "CLIP-L", i18n.tr("eng.f1.needClip"))
    builtin = [n for n, src in (("VAE", vae_src), ("T5", t5_src), ("CLIP-L", clip_src)) if src[0] == str(model_path)]
    log(i18n.tr("eng.loading", model=title) + (i18n.tr("eng.f1.builtin", parts=i18n.join(builtin)) if builtin else ""))
    # 主模型讀進系統記憶體（全包的檔只讀主模型那段）
    raw = read_part(model_path, skip=("text_encoders.", *VAE_PREFIXES))
    sd = convert(raw)
    del raw
    tcfg = _transformer_config(sd)
    log("　" + i18n.tr("eng.tfFlux", d=tcfg['num_layers'], s=tcfg['num_single_layers'], h=tcfg['num_attention_heads'],
                     dim=tcfg['num_attention_heads'] * tcfg['attention_head_dim'])
        + ("" if tcfg["guidance_embeds"] else i18n.tr("eng.f1.noGuidance")))
    transformer, is_fp8 = K._build(FluxTransformer2DModel, tcfg, sd, dtype, "　transformer", log)
    del sd
    if fp8 and not is_fp8:
        # bf16 的檔案：權重用 fp8 存、算的時候一層一層轉回 bf16（跟 Z-Image、Krea 2 同一個設定）
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        is_fp8 = True
        log("　" + i18n.tr("eng.fp8Store"))

    vae = load_vae(*vae_src, dtype, log)
    t5 = t5_from(*t5_src, dtype, log, width=tcfg["joint_attention_dim"])
    clip = clip_from(*clip_src, dtype, log)

    tokenizer = CLIPTokenizer.from_pretrained(str(CLIP_TOKENIZER))
    tokenizer_2 = T5TokenizerFast.from_pretrained(str(T5_TOKENIZER))
    tokenizer_2.model_max_length = T5_TOKENS[arch]          # 生成台的 token 計數照這個上限顯示
    # σ 由 CoCo 照官方算好（schedule），排程器的 shift 固定 1、不做動態平移
    pipe = _pipe_class()(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0,
                                                                    use_dynamic_shifting=False),
                          vae=vae, text_encoder=clip, tokenizer=tokenizer, text_encoder_2=t5, tokenizer_2=tokenizer_2,
                          transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.update_components = lambda **kw: pipe.register_modules(**kw)     # 取樣器沿用 Anima 那套（見 zimage.load）
    pipe._coco_is_flux1 = True
    pipe._coco_arch = arch
    pipe._coco_dtype = dtype
    pipe._coco_fp8 = is_fp8
    pipe._coco_weights = K.weights_label(transformer, 0)
    pipe._coco_embeds = None
    pipe._coco_f1_shift = SHIFT[arch]
    pipe._coco_te_map = t5._coco_file[0]           # 編碼完把讀過的頁面還給系統快取（zimage._trim）
    pipe._coco_count_tokenizers = ("tokenizer_2",)  # token 計數只看 T5（CLIP 只拿整句的向量，超過 77 截掉是正常的）
    log("　" + i18n.tr("eng.assembled"))
    return pipe


# ---------------------------------------------------------------- LoRA

def lora_alias(transformer):
    """LoRA 的模組名（BFL 原始命名：kohya 的 lora_unet_double_blocks_0_img_attn_qkv 去掉前綴，
    或 ComfyUI 的 diffusion_model.double_blocks.0.img_attn.qkv 換成底線）→ diffusers 的模組路徑。
    合起來的 qkv、單流的 linear1 對到好幾個模組（tuple），融合時照各模組的寬度切開（pipeline._fuse_lora）。
    用 diffusers 命名訓練的（ai-toolkit 的 transformer.transformer_blocks.0.attn.to_q）本來就查得到"""
    names = list(_TOP)
    for i in range(len(getattr(transformer, "transformer_blocks", []))):
        names += [f"double_blocks.{i}.{n}" for n in _DOUBLE]
    for i in range(len(getattr(transformer, "single_transformer_blocks", []))):
        names += [f"single_blocks.{i}.{n}" for n in _SINGLE]
    have = {n for n, _ in transformer.named_modules()}
    out = {}
    for n in names:
        mods = module_names(n)
        if mods and all(m in have for m in mods):
            out[n.replace(".", "_")] = mods if len(mods) > 1 else mods[0]
    return out


# ---------------------------------------------------------------- 記憶體

def parked(vram_mode):
    return ("text_encoder", "text_encoder_2")


def place(pipe, vram_mode, log=print):
    """主模型放得下就整個放、放不下就邊算邊搬（zimage.place）；T5（9 GB）一律邊算邊搬
    （qwen21.stream_text_encoder），CLIP-L（246 MB）常駐顯示卡"""
    import torch
    label = TITLE.get(getattr(pipe, "_coco_arch", ""), "FLUX.1")
    _z().place(pipe, vram_mode, log, label=label)
    stream_t5(pipe, log, label)
    if torch.cuda.is_available():
        pipe.text_encoder.to("cuda")


def stream_t5(pipe, log=print, label="FLUX.1"):
    """T5 邊算邊搬：算到哪一層才搬哪一層（qwen21.stream_text_encoder 那套）。T5 的 24 層在 encoder.block 底下
    （不是最外層），所以裝在 encoder 上（裝在最外層的話整個 T5 會被當成一組、每次整個搬；diffusers 的 block_modules
    遇到 use_stream 會重複註冊而出錯）。最外層的 shared 跟 encoder 的 embed_tokens 是同一份，跟著搬。
    權重是映射的檔案，不鎖定"""
    import torch
    te = pipe.text_encoder_2
    if not torch.cuda.is_available() or getattr(te, "_coco_grouped", False):
        return
    from diffusers.hooks import apply_group_offloading
    _z()._to(pipe, "text_encoder_2", "cpu")
    apply_group_offloading(te.encoder, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                           offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=False,
                           low_cpu_mem_usage=True)
    te._coco_grouped = True
    log(i18n.tr("eng.f1.t5Mapped", label=label))


def configure_memory(pipe, vram_mode, log=print):
    pass


# ---------------------------------------------------------------- 排程

def _shift_now(pipe):
    return getattr(pipe, "_coco_f1_shift", SHIFT.get(getattr(pipe, "_coco_arch", "flux1"), 1.0))


def set_scheduler(pipe, sampler, schedule, shift, log=print):
    """取樣器沿用 Anima 那套，取樣器自己的 shift 一律 1（σ 在 schedule 裡算好）；介面的 shift 另外記"""
    try:
        from core import anima as AN
    except ImportError:
        import anima as AN
    arch = getattr(pipe, "_coco_arch", "flux1")
    try:
        pipe._coco_f1_shift = float(shift) if shift not in (None, "") else SHIFT[arch]
    except (TypeError, ValueError):
        pipe._coco_f1_shift = SHIFT[arch]
    AN.set_scheduler(pipe, sampler, schedule, 1.0, lambda m: log(re.sub(r"\s*· shift 1\.0", "", m)))
    s = pipe._coco_f1_shift
    if arch != "flux1s":
        log(i18n.tr("eng.f1.shift", v=i18n.tr("eng.f1.shiftOfficial") if is_official(arch, s) else i18n.tr("eng.f1.shiftAt", s=f"{s:g}")))
    else:
        log(i18n.tr("eng.f1s.shift", v=i18n.tr("eng.f1s.shiftOfficial") if is_official(arch, s) else f"{s:g}"))


def is_official(arch, shift):
    return abs(float(shift) - SHIFT[arch]) < 1e-9


def schedule(arch, shift, width, height, steps):
    """這次的 σ。dev、Kontext：官方依 latent 格數（只算畫的這張，不算參考圖）算 μ、做指數平移 e^μ / (e^μ + (1/t − 1))（FlowMatchEuler 的
    _time_shift_exponential）；介面改過 shift 的照官方的斜率跟著解析度調整。schnell：官方不平移，
    排程器的 shift·t / (1 + (shift − 1)·t)（shift 1＝原樣）。t＝linspace(1, 1/n) 先轉 float32，運算順序照官方，逐位元相同"""
    import numpy as np
    t = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    if arch != "flux1s":
        mu = official_mu((int(height) // 16) * (int(width) // 16))
        e = math.exp(mu) if is_official(arch, shift) else float(shift) * math.exp(mu - official_mu(SEQ_1024))
        return e / (e + (1 / t - 1) ** 1.0)
    s = float(shift)
    return s * t / (1 + (s - 1) * t)


# ---------------------------------------------------------------- 提示詞

def embed(pipe, text):
    """一則提示詞 → (T5 的逐字向量 (1, n, 4096), CLIP-L 的整句向量 (1, 768))：官方 encode_prompt 的做法
    （兩個都補到固定長度、超過的截掉；T5 不給 attention mask）"""
    import torch
    dtype = getattr(pipe, "_coco_dtype", torch.bfloat16)
    dev = _q21()._te_device(pipe.text_encoder_2)
    ids = pipe.tokenizer(text, padding="max_length", max_length=CLIP_TOKENS, truncation=True, return_tensors="pt").input_ids
    pooled = pipe.text_encoder(ids.to(next(pipe.text_encoder.parameters()).device), output_hidden_states=False).pooler_output
    ids2 = pipe.tokenizer_2(text, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, truncation=True,
                            return_tensors="pt").input_ids
    emb = pipe.text_encoder_2(ids2.to(dev), output_hidden_states=False)[0]
    return emb.to(device=dev, dtype=dtype), pooled.to(device=dev, dtype=dtype)


def encode(pipe, prompt, negative, use_cfg, log=print):
    """回傳 ((正面, 正面的整句向量), (負面, …) 或 None)；同一組提示詞直接用上次的。
    文字編碼器的 LoRA 換了（pipe._coco_lora_ver）也重新編碼"""
    import torch
    key = (prompt, negative if use_cfg else None, getattr(pipe, "_coco_lora_ver", 0))
    if pipe._coco_embeds and pipe._coco_embeds[0] == key:
        return pipe._coco_embeds[1]
    t = time.perf_counter()
    try:
        with torch.no_grad():
            pos = embed(pipe, prompt)
            neg = embed(pipe, negative) if use_cfg else None
    finally:
        _z()._trim(pipe)
    log(i18n.tr("eng.f1.encode", s=f"{time.perf_counter() - t:.1f}"))
    pipe._coco_embeds = (key, (pos, neg))
    return pos, neg


# ---------------------------------------------------------------- img2img／局部重繪

def _pack(x):
    """(1, 16, h, w) → 管線在去噪迴圈裡用的 (1, h/2 · w/2, 64)（FluxPipeline._pack_latents）"""
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def _encode_init(pipe, req, width, height):
    """底圖 → FLUX.1 的 latent（(1, 16, h/8, w/8)，float32）：跟 Z-Image 同一個 VAE、同一套（zimage._encode_init）。
    用分佈的眾數不抽樣：同一張底圖每次編出一樣的 latent，差異只來自 seed"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P
    img = P.load_init_image(req["init_image"], width, height)
    vae = pipe.vae
    w = next(vae.parameters())
    x = torch.from_numpy(np.asarray(img)).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None].to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    return ((z - vae.config.shift_factor) * vae.config.scaling_factor).to(torch.float32)


def _prepare_img2img(pipe, req, generator, width, height, shift, log):
    """跟 FLUX.2 klein 同一套（flux2._prepare_img2img）：完整排程照文生圖算，跳過前面幾步、從切點的 σ 開始；
    起點 x = (1−σ)·x₀ + σ·ε。局部重繪另外每一步把遮罩外換回底圖的加噪版本（anima.mask_steps，用打包後的格式）。
    回傳 (打包好的 latents, 這次的 σ, 跑幾步, 還原函式)"""
    import torch
    try:
        from core import anima as AN, pipeline as P
    except ImportError:
        import anima as AN
        import pipeline as P

    arch = getattr(pipe, "_coco_arch", "flux1")
    n_all, run = P.img2img_steps(req["steps"], req.get("strength", 0.5))
    full = schedule(arch, shift, width, height, n_all)
    skip = n_all - run
    s0 = float(full[skip])
    x0 = _encode_init(pipe, req, width, height)
    eps = torch.randn(x0.shape, generator=generator, device=generator.device, dtype=torch.float32).to(x0.device)
    latents = _pack((1.0 - s0) * x0 + s0 * eps)
    restore = None
    if req.get("_mask") is not None:
        try:
            from core import inpaint as IP
        except ImportError:
            import inpaint as IP
        h, w = x0.shape[-2:]
        m = torch.from_numpy(IP.latent_mask(req["_mask"], w, h))[None, None].to(x0.device).expand_as(x0)
        restore = AN.mask_steps(pipe.scheduler, _pack(x0), _pack(eps), _pack(m.contiguous()))
    strength = min(1.0, max(P.STRENGTH_MIN, float(req.get("strength", 0.5))))
    log(i18n.tr("eng.i2iFrom", model="FLUX.1", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))
    return latents, full[skip:], run, restore or (lambda: None)


# ---------------------------------------------------------------- 用指示修圖（Kontext）

_KONTEXT = None


def _kontext_class():
    """官方的 FluxKontextPipeline（跟 _pipe_class 一樣：有顯示卡就一律在顯示卡上算）"""
    global _KONTEXT
    if _KONTEXT is None:
        import torch
        from diffusers import FluxKontextPipeline

        class CocoFluxKontextPipeline(FluxKontextPipeline):
            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _KONTEXT = CocoFluxKontextPipeline
    return _KONTEXT


def kontext_pipe(pipe):
    """同一組元件組成官方的 Kontext 管線（取樣器可能換過，每次照現在的組）"""
    k = _kontext_class()(scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                         text_encoder_2=pipe.text_encoder_2, tokenizer_2=pipe.tokenizer_2, transformer=pipe.transformer)
    k.set_progress_bar_config(disable=True)
    return k


def _flat(f):
    """一張圖 → RGB，透明的地方鋪白（管線轉 RGB 時直接丟掉透明度，底下的顏色常常是黑的或亂的）"""
    from PIL import Image
    im = Image.open(f)
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    return im.convert("RGB")


def edit_images(req):
    """修圖的條件圖：正在修的這張；有其他參考圖時照 ComfyUI 官方 Kontext 工作流程的 Image Stitch（往右接、
    高度對齊第一張、照比例縮放），接成一張（官方管線只收一張，之後縮到最接近的官方解析度）"""
    from PIL import Image
    first = _flat(req["edit_image"])
    refs = [_flat(f) for f in (req.get("edit_refs") or [])]
    if not refs:
        return first
    h = first.height
    refs = [r if r.height == h else r.resize((max(1, round(r.width * h / r.height)), h), Image.LANCZOS) for r in refs]
    out = Image.new("RGB", (first.width + sum(r.width for r in refs), h), (255, 255, 255))
    x = 0
    for im in (first, *refs):
        out.paste(im, (x, 0))
        x += im.width
    return out


# ---------------------------------------------------------------- 生成

def _snap(v):
    return max(MULTIPLE * 8, -(-int(v) // MULTIPLE) * MULTIPLE)


def generate(pipe, req, on_step=None, log=print):
    """回傳 (PIL Image, seed)。介面與 pipeline.generate 對齊"""
    import torch
    from PIL import Image
    try:
        from core.anima import _bind_scheduler_generator
    except ImportError:
        from anima import _bind_scheduler_generator
    Z = _z()
    arch = getattr(pipe, "_coco_arch", "flux1")

    seed = int(req.get("seed", -1))
    if seed < 0:
        import random
        seed = random.randrange(1 << 31)
    generator = torch.Generator("cpu" if str(req.get("rng") or "gpu").lower() == "cpu" else "cuda").manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scheduler = pipe.scheduler
    _bind_scheduler_generator(scheduler)
    scheduler._coco_generator = generator

    # dev：CFG 欄就是官方的 guidance，不跑負面；schnell：CFG 照一般的定義（負 + CFG ×（正 − 負）），1（含以下）不跑負面
    cfg = float(req.get("cfg", DEFAULT_CFG[arch]))
    guided = bool(pipe.transformer.config.guidance_embeds)
    use_cfg = not guided and cfg > 1.0
    prompt, negative = Z._clean(req.get("prompt", "")), Z._clean(req.get("negative", ""))
    if prompt != (req.get("prompt") or "").strip():
        log(i18n.tr("eng.noWeightSyntax", model="FLUX.1"))
    if guided and negative:
        log(i18n.tr("eng.f1.guidance", model=TITLE[arch], cfg=f"{cfg:g}"))
    pos, neg = encode(pipe, prompt, negative, use_cfg, log)

    if torch.cuda.is_available() and not getattr(pipe.transformer, "_coco_grouped", False):
        Z._to(pipe, "transformer", "cuda")
    Z._to(pipe, "vae", "cuda" if torch.cuda.is_available() else "cpu")

    width, height = int(req["width"]), int(req["height"])
    w, h = _snap(width), _snap(height)
    if (w, h) != (width, height):
        log(i18n.tr("eng.multipleOf", model="FLUX.1", m=16, w=w, h=h, tw=width, th=height))
    steps = int(req["steps"])
    shift = _shift_now(pipe)
    sig = schedule(arch, shift, w, h, steps)
    extra, restore, run = {}, None, pipe
    if req.get("edit_image"):
        if arch != "kontext":
            raise RuntimeError(i18n.tr("eng.f1.noEdit", model=TITLE[arch]))
        # 修圖：條件圖交給官方的 Kontext 管線；輸出照要的大小（max_area＝寬×高，管線就不會再縮放），
        # σ 照畫的這張的大小算（官方也是）
        cond = edit_images(req)
        names = [Path(req["edit_image"]).name, *(Path(f).name for f in (req.get("edit_refs") or []))]
        log(i18n.tr("eng.f1.kontext", name=names[0]) + (i18n.tr("eng.f1.kontextRefs", names=i18n.join(names[1:]), w=cond.width, h=cond.height)
                                                    if len(names) > 1 else ""))
        run, extra = kontext_pipe(pipe), {"image": cond, "max_area": w * h}
    elif req.get("init_image"):
        latents, sig, steps, restore = _prepare_img2img(pipe, req, generator, w, h, shift, log)
        extra = {"latents": latents}
    restores = [_flux2()._use_sigmas(scheduler, sig), *([restore] if restore else [])]

    def cb(p, i, t, kw):
        if on_step:
            on_step(i + 1, steps)
        return kw

    try:
        out = run(prompt_embeds=pos[0], pooled_prompt_embeds=pos[1],
                   negative_prompt_embeds=neg[0] if use_cfg else None,
                   negative_pooled_prompt_embeds=neg[1] if use_cfg else None,
                   true_cfg_scale=cfg if use_cfg else 1.0, guidance_scale=cfg if guided else 1.0,
                   height=h, width=w, num_inference_steps=steps, generator=generator, callback_on_step_end=cb,
                   max_sequence_length=pipe.tokenizer_2.model_max_length, **extra)
    finally:
        for r in reversed(restores):
            r()
        scheduler._coco_generator = None
    img = out.images[0]
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img, seed
