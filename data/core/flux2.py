# -*- coding: utf-8 -*-
"""
flux2.py —— FLUX.2 [klein]（Black Forest Labs 的 4B 與 9B；蒸餾版 4 步、Base 版 50 步）

模型檔是單檔的 transformer（BFL 原始命名：double_blocks、single_blocks、img_in、txt_in…，ComfyUI 用的就是這個檔），
VAE 與文字編碼器要另外選：
  VAE          FLUX.2 的 VAE（flux2-vae.safetensors，32 個 latent 通道；4B、9B 共用）
  文字編碼器   4B 用 Qwen3-4B（qwen_3_4b，跟 Z-Image 同一個檔）、9B 用 Qwen3-8B（qwen_3_8b…）
架構代號：klein4、klein9。文字編碼器不同、LoRA 不能互用，所以零件、步數、CFG 各記各的；程式共用這一份。
FLUX.2 [dev]（32B，文字編碼器是 Mistral）不支援，認成 other。

diffusers 0.39 有 Flux2KleinPipeline 與單檔的鍵名轉換；transformer 的設定從權重形狀推，不用連網。
蒸餾版與 Base 版的檔案分不出來（鍵名、形狀一樣，沒有中繼資料），預設照蒸餾版（4 步、CFG 1），用 Base 版的人改了 CoCo 會記住。

**文字編碼器。** 官方取 Qwen3 第 9、18、27 層的輸出疊起來（36 層的後 9 層用不到），所以只建前 27 層、
最後的 norm 換成什麼都不做（transformers 5 的 hidden_states 最後一格是 norm 之後的）；檔案照 Z-Image 那套直接映射，
編碼時邊算邊搬（Qwen Image 2.1 那套），主模型不用讓開。跟官方（from_pretrained 的完整 36 層）逐位元相同。
9B 的 qwen_3_8b_fp8mixed 混了 fp8 與 NVFP4 的層，照 Krea 2 那套照原樣存放、算的時候還原（krea2.prepare_weights）。

**排程。** 官方的 σ 依「latent 格數」與步數算（compute_empirical_mu，指數平移）。CoCo 自己算好，用 shift 1 的排程器跑，
跟官方逐位元相同。管線遇到 use_flow_sigmas 的取樣器（UniPC、DPM++）會把傳進去的 σ 丟掉，所以不從參數傳，
改成包住排程器的 set_timesteps（_use_sigmas）。介面的 shift 是「1024²、4 步」時的值，預設值＝照官方自動算。

**用指示修圖。** 精修的修圖跟 Qwen Image 2.1 共用同一個模式：正在修的這張與其他參考圖照順序交給官方管線
（edit_images），提示詞照一般的編碼。
"""

import math
import re
import time
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "assets" / "anima" / "tokenizer"        # Qwen3 各尺寸同一份詞彙（跟官方的 token 逐一核對過）

PREFIX = "model.diffusion_model."
TEXT_LAYERS = (9, 18, 27)                 # 官方取這三層的輸出（hidden_states 的索引，0 是嵌入層）
MAX_TOKENS = 512                          # 官方固定補到 512 個 token，超過的截掉
# Qwen3 的對話模板（enable_thinking=False）：跟官方 tokenizer 的 apply_chat_template 逐字相同
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MULTIPLE = 16                             # 寬高要 16 的倍數（VAE 縮小 8 倍、再 2×2 打包）

# 文字的維度（txt_in 的輸入＝文字編碼器的寬度 × 3 層）→ 架構代號
ARCHS = {7680: "klein4", 12288: "klein9"}
TITLE = {"klein4": "FLUX.2 klein 4B", "klein9": "FLUX.2 klein 9B"}
TE_FILE = {"klein4": ("Qwen3-4B", "qwen_3_4b"), "klein9": ("Qwen3-8B", "qwen_3_8b")}


def te_name(arch):
    """「Qwen3-4B（qwen_3_4b）」，照介面語言"""
    return i18n.tr("srv.part.hint", name=TE_FILE[arch][0], file=TE_FILE[arch][1])

DEFAULT_CFG, DEFAULT_STEPS = 1.0, 4       # 蒸餾版的官方用法；Base 版官方是 50 步、CFG 4
SEQ_1024 = (1024 // 16) ** 2


def official_mu(seq, steps):
    """官方的 compute_empirical_mu（diffusers 的 pipeline_flux2_klein，照抄）"""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq > 4300:
        return float(a2 * seq + b2)
    m_200 = a2 * seq + b2
    m_10 = a1 * seq + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * steps + b)


SHIFT = round(math.exp(official_mu(SEQ_1024, DEFAULT_STEPS)), 2)     # 介面的預設值（≈ 9.88）


# ---------------------------------------------------------------- 認檔

def _strip(k):
    return k[len(PREFIX):] if k.startswith(PREFIX) else k


def is_flux2(keys):
    """FLUX.2 的 transformer 單檔：BFL 原始命名或 diffusers 命名（可能帶 model.diffusion_model. 前綴）"""
    ks = {_strip(k) for k in keys}
    return "single_stream_modulation.lin.weight" in ks or "single_stream_modulation.linear.weight" in ks


def arch_of(keys, shape):
    """klein4／klein9；認不出尺寸的（FLUX.2 dev 這類）回 other。shape(鍵) → 那個張量的形狀"""
    ks = {_strip(k): k for k in keys}
    for name in ("txt_in.weight", "context_embedder.weight"):
        if name in ks:
            dim = list(shape(ks[name]))[-1]
            return ARCHS.get(dim, "other")
    return "other"


# ---------------------------------------------------------------- 鍵名

# BFL 原始的線性層 → diffusers 的模組（合起來的 qkv 對到拆開的三個）。轉換權重用 diffusers 的
# convert_flux2_transformer_checkpoint_to_diffusers；這張表給 fp8 的比例、LoRA、直接用 fp8 算的層對名字
# （tests/test_flux2.py 核對過兩邊一致）。final_layer.adaLN_modulation.1 不放：diffusers 把它的前後兩半對調了，
# LoRA 照原本的順序加上去會加錯
_TOP = {"img_in": "x_embedder", "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out"}
_DOUBLE = {"img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"), "img_attn.proj": ("attn.to_out.0",),
           "img_mlp.0": ("ff.linear_in",), "img_mlp.2": ("ff.linear_out",),
           "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"), "txt_attn.proj": ("attn.to_add_out",),
           "txt_mlp.0": ("ff_context.linear_in",), "txt_mlp.2": ("ff_context.linear_out",)}
_SINGLE = {"linear1": ("attn.to_qkv_mlp_proj",), "linear2": ("attn.to_out",)}


def module_names(name):
    """BFL 的線性層名稱 → diffusers 的模組路徑（tuple；認不得回空的）"""
    if name in _TOP:
        return (_TOP[name],)
    parts = name.split(".")
    if len(parts) >= 3 and parts[1].isdigit():
        rest = ".".join(parts[2:])
        if parts[0] == "double_blocks" and rest in _DOUBLE:
            return tuple(f"transformer_blocks.{parts[1]}.{m}" for m in _DOUBLE[rest])
        if parts[0] == "single_blocks" and rest in _SINGLE:
            return tuple(f"single_transformer_blocks.{parts[1]}.{m}" for m in _SINGLE[rest])
    return ()


_QUANT = (".weight_scale", ".input_scale", ".comfy_quant", ".weight_scale_2")


def convert(raw):
    """單檔 → diffusers 的鍵名。fp8 檔（ComfyUI 的 scaled fp8：weight × 一個純量 weight_scale）的比例
    跟著權重改名，合起來的 qkv 拆成三個時比例照抄三份（同一個純量）；input_scale 用不到（CoCo 照 Krea 2 的算法）"""
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    sd = {_strip(k): v for k, v in raw.items()}
    if any(k.endswith(".weight_scale_2") for k in sd):
        raise RuntimeError(i18n.tr("eng.f2.fp4"))
    if "single_stream_modulation.linear.weight" in sd:          # 已經是 diffusers 的命名
        return {k: v for k, v in sd.items() if not k.endswith((".input_scale", ".comfy_quant"))}
    extra = {k: sd.pop(k) for k in [k for k in sd if k.endswith(_QUANT)]}
    out = convert_flux2_transformer_checkpoint_to_diffusers(sd)
    for k, v in extra.items():
        if not k.endswith(".weight_scale"):
            continue
        names = module_names(k[:-len(".weight_scale")])
        if not names:
            raise RuntimeError(i18n.tr("eng.f2.noModule", k=k))
        for n in names:
            out[n + ".weight_scale"] = v
    return out


def _count(sd, prefix):
    idx = {int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)}
    return max(idx) + 1 if idx else 0


def _transformer_config(sd):
    inner, cin = sd["x_embedder.weight"].shape
    head = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    out = sd["proj_out.weight"].shape[0]
    return {"patch_size": 1, "in_channels": cin, "out_channels": None if out == cin else out,
            "num_layers": _count(sd, "transformer_blocks."), "num_single_layers": _count(sd, "single_transformer_blocks."),
            "attention_head_dim": head, "num_attention_heads": inner // head,
            "joint_attention_dim": sd["context_embedder.weight"].shape[1],
            "timestep_guidance_channels": sd["time_guidance_embed.timestep_embedder.linear_1.weight"].shape[1],
            "mlp_ratio": sd["transformer_blocks.0.ff.linear_in.weight"].shape[0] / 2 / inner,
            "axes_dims_rope": (head // 4,) * 4, "rope_theta": 2000, "eps": 1e-6,          # 官方 128 → 32×4
            "guidance_embeds": any(k.startswith("time_guidance_embed.guidance_embedder.") for k in sd)}


def fp8_mm_layers(metadata):
    """檔案的 _quantization_metadata → 可以直接用 fp8 算的層（diffusers 的模組路徑），規則跟 Krea 2 一樣
    （krea2.fp8_mm_layers：標了 full_precision_matrix_mult 的用完整精度算）"""
    import json
    try:
        layers = json.loads((metadata or {}).get("_quantization_metadata") or "{}").get("layers") or {}
    except (ValueError, AttributeError):
        return set()
    out = set()
    for name, conf in layers.items():
        if isinstance(conf, dict) and "float8" in str(conf.get("format", "")) and not conf.get("full_precision_matrix_mult"):
            out.update(module_names(_strip(name)))
    return out


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


_PIPE = None


def _pipe_class():
    """跟 Z-Image 一樣：文字編碼器平常在系統記憶體，diffusers 挑到它會整條判成 CPU。有顯示卡就一律在顯示卡上算"""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import Flux2KleinPipeline

        class CocoFlux2KleinPipeline(Flux2KleinPipeline):
            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _PIPE = CocoFlux2KleinPipeline
    return _PIPE


def te_weights(raw):
    """文字編碼器檔案裡用得到的：model.* 的前 27 層與嵌入層（lm_head、量化的附帶資料不要；
    最後的 norm 載入時要有，建好之後換掉，見 text_encoder_from）"""
    out, last = {}, TEXT_LAYERS[-1]
    for k, v in raw.items():
        if not k.startswith("model."):
            continue
        k = k[len("model."):]
        if k.endswith((".comfy_quant", ".input_scale")):
            continue
        if k.startswith("layers.") and int(k.split(".")[1]) >= last:
            continue
        out[k] = v
    return out


def text_encoder_from(te_path, dtype, arch, log=print):
    """Qwen3 文字編碼器（只建前 27 層）。權重直接映射檔案，不另外讀進系統記憶體；fp8、NVFP4 的層照原樣存放。
    編碼時才搬上顯示卡（zimage._to）"""
    import torch
    from transformers import Qwen3Model
    Z = _z()
    raw, mm = Z._mapped(te_path)
    sd = te_weights(raw)
    del raw
    name = Path(te_path).name
    if "embed_tokens.weight" not in sd:
        raise RuntimeError(i18n.tr("eng.notQwen3", name=name))
    if any(k.startswith(("visual.", "language_model.")) for k in sd):
        raise RuntimeError(i18n.tr("eng.f2.isVl", name=name, model=TITLE[arch], te=te_name(arch)))
    width = sd["embed_tokens.weight"].shape[1]
    want = next(d for d, a in ARCHS.items() if a == arch) // len(TEXT_LAYERS)
    if width != want:
        raise RuntimeError(i18n.tr("eng.teDim", got=width, want=want, model=TITLE[arch], te=te_name(arch)))
    if _count(sd, "layers.") < TEXT_LAYERS[-1]:
        raise RuntimeError(i18n.tr("eng.f2.layers", name=name, n=_count(sd, 'layers.'), need=TEXT_LAYERS[-1]))
    te, _ = _krea2()._build(Qwen3Model, Z._qwen3_config(sd), sd, dtype, "　text_encoder", log)
    te.norm = torch.nn.Identity()
    te._coco_mapped, te._coco_file = True, (mm, str(te_path))
    return te.eval()


# 官方的 vae/config.json（FLUX.2-klein-4B）
VAE = {"act_fn": "silu", "batch_norm_eps": 1e-4, "batch_norm_momentum": 0.1,
       "block_out_channels": (128, 256, 512, 512), "down_block_types": ("DownEncoderBlock2D",) * 4,
       "up_block_types": ("UpDecoderBlock2D",) * 4, "force_upcast": True, "in_channels": 3, "out_channels": 3,
       "latent_channels": 32, "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32,
       "patch_size": (2, 2), "sample_size": 1024, "use_post_quant_conv": True, "use_quant_conv": True}


def convert_vae(vsd):
    """FLUX.2 的 VAE 有兩種命名：diffusers 的（BFL 官方庫的 vae/、ComfyUI 的 flux2-dev 那份）照原樣；
    BFL 原始的（encoder.down.N.block…、quant_conv 放在 encoder／decoder 底下；ComfyUI 的 klein 那份）
    照 FLUX.1 的 VAE 轉（convert_ldm_vae_checkpoint），bn 的統計值另外帶過去"""
    if "decoder.conv_in.weight" in vsd and "encoder.down_blocks.0.resnets.0.conv1.weight" in vsd:
        return dict(vsd)
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    moved = {"encoder.quant_conv.": "quant_conv.", "decoder.post_quant_conv.": "post_quant_conv."}
    sd = {}
    for k, v in vsd.items():
        for a, b in moved.items():
            if k.startswith(a):
                k = b + k[len(a):]
        sd[k] = v
    out = convert_ldm_vae_checkpoint(sd, VAE)
    out.update({k: v for k, v in sd.items() if k.startswith("bn.")})
    return out


def load_vae(vae_path, dtype, log=print):
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file
    try:
        from core.anima import _build
    except ImportError:
        from anima import _build
    vsd = convert_vae(load_file(str(vae_path)))
    conv = vsd.get("decoder.conv_in.weight")
    if conv is None or conv.shape[1] != 32 or "bn.running_mean" not in vsd:
        raise RuntimeError(i18n.tr("eng.f2.notVae", name=Path(vae_path).name))
    # diffusers 載入時整顆轉成 bf16，bn 的統計值也是
    return _build(AutoencoderKLFlux2, dict(VAE), {k: v.to(dtype) if v.is_floating_point() else v for k, v in vsd.items()},
                  "　vae", log)


def load(model_path, vae_path, te_path, log=print, dtype=None, fp8=False):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer
    K = _krea2()

    dtype = dtype or torch.bfloat16
    with safe_open(str(model_path), framework="pt") as f:
        arch = arch_of(list(f.keys()), lambda k: f.get_slice(k).get_shape())
        mm_names = fp8_mm_layers(f.metadata())
    if arch not in TITLE:
        raise RuntimeError(i18n.tr("eng.f2.devUnsupported"))
    title = TITLE[arch]
    if not vae_path or not te_path:
        raise RuntimeError(i18n.tr("eng.needParts", model=title, vae=i18n.tr("eng.vaeOf", name="FLUX.2", file="flux2-vae"), te=te_name(arch)))
    for p, what in ((vae_path, "VAE"), (te_path, i18n.tr("eng.teCap"))):
        if not Path(p).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=p))

    log(i18n.tr("eng.loading", model=title))
    sd = convert(load_file(str(model_path)))
    tcfg = _transformer_config(sd)
    log("　" + i18n.tr("eng.tfFlux", d=tcfg['num_layers'], s=tcfg['num_single_layers'], h=tcfg['num_attention_heads'],
                     dim=tcfg['num_attention_heads'] * tcfg['attention_head_dim']))
    transformer, is_fp8 = K._build(Flux2Transformer2DModel, tcfg, sd, dtype, "　transformer", log)
    del sd
    direct = K.enable_fp8_mm(transformer, mm_names, log) if is_fp8 else 0
    if fp8 and not is_fp8:
        # bf16 的檔案：權重用 fp8 存、算的時候一層一層轉回 bf16（跟 Z-Image、Krea 2 同一個設定）
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        is_fp8 = True
        log("　" + i18n.tr("eng.fp8Store"))

    vae = load_vae(vae_path, dtype, log)
    text_encoder = text_encoder_from(te_path, dtype, arch, log)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(TOKENIZER))
    tokenizer.model_max_length = MAX_TOKENS                 # 生成台的 token 計數照這個上限顯示
    # σ 由 CoCo 照官方算好（sigmas），排程器的 shift 固定 1、不做動態平移
    pipe = _pipe_class()(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0,
                                                                    use_dynamic_shifting=False),
                          vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.update_components = lambda **kw: pipe.register_modules(**kw)     # 取樣器沿用 Anima 那套（見 zimage.load）
    pipe._coco_is_flux2 = True
    pipe._coco_arch = arch
    pipe._coco_dtype = dtype
    pipe._coco_fp8 = is_fp8
    pipe._coco_weights = K.weights_label(transformer, direct)
    pipe._coco_embeds = None
    pipe._coco_f2_shift = SHIFT
    pipe._coco_te_map = text_encoder._coco_file[0]      # 編碼完把讀過的頁面還給系統快取（zimage._trim）
    log("　" + i18n.tr("eng.assembled"))
    return pipe


# ---------------------------------------------------------------- LoRA

def lora_alias(transformer):
    """LoRA 的模組名（BFL 原始命名：kohya 的 lora_unet_double_blocks_0_img_attn_qkv 去掉前綴，
    或 ai-toolkit／ComfyUI 的 diffusion_model.double_blocks.0.img_attn.qkv 換成底線）→ diffusers 的模組路徑。
    合起來的 qkv 對到三個模組（tuple），融合時照順序切三段（pipeline._fuse_lora）。
    用 diffusers 命名訓練的（transformer.transformer_blocks.0.attn.to_q）本來就查得到"""
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


# ---------------------------------------------------------------- 記憶體（主模型跟 Z-Image 同一套，文字編碼器跟 Qwen Image 2.1）

def parked(vram_mode):
    return ("text_encoder",)


def place(pipe, vram_mode, log=print):
    """主模型放得下就整個放（zimage.place）；文字編碼器一律邊算邊搬（qwen21.stream_text_encoder），主模型不用讓開。
    2026-10-09 RTX 5080、4B、換提示詞編碼（含主模型搬回顯示卡）：整個搬 3.4～3.6 秒（主模型 7.75 GB 先讓開再搬回來），
    邊算邊搬 0.6～0.7 秒；fp8 存放（兩個放得下）整個搬 1.1 秒、邊算邊搬 0.6 秒。編出來逐位元相同"""
    label = TITLE.get(getattr(pipe, "_coco_arch", ""), "FLUX.2")
    _z().place(pipe, vram_mode, log, label=label)
    _q21().stream_text_encoder(pipe, log, label=label)


def configure_memory(pipe, vram_mode, log=print):
    _q21()._te_to(pipe, "cpu")


# ---------------------------------------------------------------- 排程

def set_scheduler(pipe, sampler, schedule, shift, log=print):
    """取樣器沿用 Anima 那套，取樣器自己的 shift 一律 1（σ 在 sigmas 裡算好）；介面的 shift 另外記"""
    try:
        from core import anima as AN
    except ImportError:
        import anima as AN
    try:
        pipe._coco_f2_shift = float(shift) if shift not in (None, "") else SHIFT
    except (TypeError, ValueError):
        pipe._coco_f2_shift = SHIFT
    AN.set_scheduler(pipe, sampler, schedule, 1.0, lambda m: log(re.sub(r"\s*· shift 1\.0", "", m)))
    log(i18n.tr("eng.f2.shift", v=i18n.tr("eng.f2.shiftOfficial") if is_official(pipe._coco_f2_shift) else i18n.tr("eng.f2.shiftCustom", s=f"{pipe._coco_f2_shift:g}")))


def is_official(shift):
    return abs(float(shift) - SHIFT) < 1e-9


def effective_shift(shift, width, height, steps):
    """介面的 shift → 這個解析度、步數實際用的。預設值就是官方的；改過的照官方的斜率跟著解析度調整"""
    seq = (int(height) // 16) * (int(width) // 16)
    mu = official_mu(seq, steps)
    if is_official(shift):
        return math.exp(mu)
    return float(shift) * math.exp(mu - official_mu(SEQ_1024, steps))


def sigmas(steps, shift):
    """官方的 σ：linspace(1, 1/n) 先轉 float32，再做指數平移 e^μ / (e^μ + (1/t − 1))（FlowMatchEuler 的
    _time_shift_exponential，shift＝e^μ）。運算順序照官方，算出來逐位元相同"""
    import numpy as np
    t = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    return shift / (shift + (1 / t - 1) ** 1.0)


def _use_sigmas(scheduler, sig):
    """這次生成的 σ 一律用 sig：管線呼叫 set_timesteps 時不管傳了什麼都換成這組。回傳還原用的函式。
    UniPC、DPM++ 這類 use_flow_sigmas 的取樣器，管線本來會把 σ 丟掉、改用排程器自己算的（沒有平移）"""
    import functools
    inner = scheduler.set_timesteps

    @functools.wraps(inner)                 # 管線用 inspect.signature 看收不收 sigmas
    def set_timesteps(*a, device=None, mu=None, **kw):
        return inner(sigmas=sig, device=device, mu=mu)

    scheduler.set_timesteps = set_timesteps
    return lambda: scheduler.__dict__.pop("set_timesteps", None)


# ---------------------------------------------------------------- 提示詞

def embed(pipe, text):
    """一則提示詞 → (1, 512, 3 × 寬度)：第 9、18、27 層的輸出接起來（官方 _get_qwen3_prompt_embeds 的做法）"""
    import torch
    te = pipe.text_encoder
    dev = _q21()._te_device(te)
    tok = pipe.tokenizer(TEMPLATE.format(text), return_tensors="pt", padding="max_length", truncation=True,
                         max_length=MAX_TOKENS)
    out = te(input_ids=tok.input_ids.to(dev), attention_mask=tok.attention_mask.to(dev),
             output_hidden_states=True, use_cache=False)
    h = torch.stack([out.hidden_states[k] for k in TEXT_LAYERS], dim=1).to(getattr(pipe, "_coco_dtype", torch.bfloat16))
    b, c, n, d = h.shape
    return h.permute(0, 2, 1, 3).reshape(b, n, c * d)


def encode(pipe, prompt, negative, use_cfg, log=print):
    """回傳 (正面, 負面)；同一組提示詞直接用上次的。文字編碼器平常邊算邊搬（place）；沒有的話（沒顯示卡）
    整個搬，主模型整個放在顯示卡、又擠不下時先讓開（跟 Qwen Image 2.1 的 encode 一樣）"""
    import torch
    Z, Q = _z(), _q21()
    key = (prompt, negative if use_cfg else None)
    if pipe._coco_embeds and pipe._coco_embeds[0] == key:
        return pipe._coco_embeds[1]
    tr = pipe.transformer
    moved_tr = False
    if torch.cuda.is_available():
        if Z._device(tr) == "cuda" and not getattr(tr, "_coco_grouped", False):
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            if free < Q._te_room(pipe) + (1.5 * 2 ** 30):
                Z._to(pipe, "transformer", "cpu")
                moved_tr = True
        Q._te_to(pipe, "cuda")
    t = time.perf_counter()
    try:
        with torch.no_grad():
            pos = embed(pipe, prompt)
            neg = embed(pipe, negative) if use_cfg else None
    finally:
        Q._te_to(pipe, "cpu")
        Z._trim(pipe)
        if moved_tr:
            Z._to(pipe, "transformer", "cuda")
    log(i18n.tr("eng.encodeTime", model="FLUX.2", s=f"{time.perf_counter() - t:.1f}") + (i18n.tr("eng.trMoved") if moved_tr else ""))
    pipe._coco_embeds = (key, (pos, neg))
    return pos, neg


# ---------------------------------------------------------------- img2img／局部重繪

def _pack(x):
    """(1, 128, h, w) → 管線在去噪迴圈裡用的 (1, h·w, 128)（Flux2KleinPipeline._pack_latents）"""
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).permute(0, 2, 1)


def _encode_init(pipe, req, width, height):
    """底圖 → 正規化過的 latent（(1, 128, h/16, w/16)，float32）：官方的 _encode_vae_image，
    VAE 編碼（眾數，不抽樣）→ 2×2 打包 → 用 bn 的統計值正規化。同一張底圖每次編出一樣的 latent"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P
    img = P.load_init_image(req["init_image"], width, height)
    vae = pipe.vae
    w = next(vae.parameters())
    x = torch.from_numpy(np.array(img)).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None].to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        z = pipe._patchify_latents(vae.encode(x).latent_dist.mode())
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z.device, z.dtype)
    return ((z - mean) / std).to(torch.float32)


def _prepare_img2img(pipe, req, generator, width, height, shift, log):
    """跟 Anima、Z-Image、Qwen Image 2.1 同一套：完整排程照文生圖算（官方的 σ），跳過前面幾步、從切點的 σ 開始；
    起點 x = (1−σ)·x₀ + σ·ε。局部重繪另外每一步把遮罩外換回底圖的加噪版本（anima.mask_steps，用打包後的格式）。
    回傳 (latents（沒打包，管線自己打包）, 這次的 σ, 跑幾步, 還原函式)"""
    import torch
    try:
        from core import anima as AN, pipeline as P
    except ImportError:
        import anima as AN
        import pipeline as P

    n_all, run = P.img2img_steps(req["steps"], req.get("strength", 0.5))
    full = sigmas(n_all, shift)
    skip = n_all - run
    s0 = float(full[skip])
    x0 = _encode_init(pipe, req, width, height)
    eps = torch.randn(x0.shape, generator=generator, device=generator.device, dtype=torch.float32).to(x0.device)
    latents = (1.0 - s0) * x0 + s0 * eps
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
    log(i18n.tr("eng.i2iFrom", model="FLUX.2", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))
    return latents, full[skip:], run, restore or (lambda: None)


# ---------------------------------------------------------------- 生成

def _snap(v):
    return max(MULTIPLE * 8, -(-int(v) // MULTIPLE) * MULTIPLE)


# ---------------------------------------------------------------- 用指示修圖（參考圖）

def edit_images(req):
    """修圖的參考圖：正在修的這張（圖1）＋其他參考圖（圖2、圖3），照順序交給官方管線
    （每張縮到約 100 萬像素以內、對齊 16，VAE 編碼後接在畫的這張後面，位置編碼的時間軸照順序分開）。
    透明的地方鋪白：管線轉 RGB 時直接丟掉透明度，底下的顏色常常是黑的或亂的"""
    from PIL import Image
    out = []
    for f in [req["edit_image"], *(req.get("edit_refs") or [])]:
        im = Image.open(f)
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        out.append(im.convert("RGB"))
    return out


def generate(pipe, req, on_step=None, log=print):
    """回傳 (PIL Image, seed)。介面與 pipeline.generate 對齊"""
    import torch
    from PIL import Image
    try:
        from core.anima import _bind_scheduler_generator
    except ImportError:
        from anima import _bind_scheduler_generator
    Z = _z()

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

    # CFG 照一般的定義（官方管線就是：負 + CFG ×（正 − 負））。CFG 1（含以下）不跑負面：蒸餾版官方就是這樣用
    cfg = float(req.get("cfg", DEFAULT_CFG))
    use_cfg = cfg > 1.0
    prompt, negative = Z._clean(req.get("prompt", "")), Z._clean(req.get("negative", ""))
    if prompt != (req.get("prompt") or "").strip():
        log(i18n.tr("eng.noWeightSyntax", model="FLUX.2"))
    pos, neg = encode(pipe, prompt, negative, use_cfg, log)

    if torch.cuda.is_available() and not getattr(pipe.transformer, "_coco_grouped", False):
        Z._to(pipe, "transformer", "cuda")
    Z._to(pipe, "vae", "cuda" if torch.cuda.is_available() else "cpu")

    width, height = int(req["width"]), int(req["height"])
    w, h = _snap(width), _snap(height)
    if (w, h) != (width, height):
        log(i18n.tr("eng.multipleOf", model="FLUX.2", m=16, w=w, h=h, tw=width, th=height))
    steps = int(req["steps"])
    shift = effective_shift(getattr(pipe, "_coco_f2_shift", SHIFT), w, h, steps)
    sig = sigmas(steps, shift)
    extra, restore = {}, None
    if req.get("edit_image"):
        # 修圖：提示詞照一般的編碼（klein 的文字編碼器不看圖），參考圖交給管線；σ 照畫的這張的大小算（官方也是）
        conds = edit_images(req)
        names = [Path(req["edit_image"]).name, *(Path(f).name for f in (req.get("edit_refs") or []))]
        log(i18n.tr("eng.editFrom", model="FLUX.2", name=names[0], w=conds[0].width, h=conds[0].height)
            + (i18n.tr("eng.editRefs", names=i18n.join(names[1:])) if len(names) > 1 else ""))
        extra = {"image": conds}
    elif req.get("init_image"):
        latents, sig, steps, restore = _prepare_img2img(pipe, req, generator, w, h, shift, log)
        extra = {"latents": latents}
    restores = [_use_sigmas(scheduler, sig), *([restore] if restore else [])]

    def cb(p, i, t, kw):
        if on_step:
            on_step(i + 1, steps)
        return kw

    try:
        out = pipe(prompt_embeds=pos, negative_prompt_embeds=neg if use_cfg else None,
                   guidance_scale=cfg if use_cfg else 1.0, height=h, width=w, num_inference_steps=steps,
                   generator=generator, callback_on_step_end=cb, max_sequence_length=MAX_TOKENS, **extra)
    finally:
        for r in reversed(restores):
            r()
        scheduler._coco_generator = None
    img = out.images[0]
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img, seed
