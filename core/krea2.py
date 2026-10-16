# -*- coding: utf-8 -*-
"""
krea2.py —— Krea 2（Krea AI 的 12B 單流 MMDiT；Turbo 是 8 步的蒸餾版）

模型檔是 ComfyUI 的拆分式單檔：只有擴散 transformer，而且是 Krea 原始的命名
（blocks.N.attn.wq、txtfusion、first／last、tmlp…），這裡對照成 diffusers 的 Krea2Transformer2DModel。
VAE 與文字編碼器要另外選：
  VAE          Qwen Image 的 VAE（qwen_image_vae.safetensors，跟 Anima 同一顆）
  文字編碼器   Qwen3-VL 4B（qwen3vl_4b_fp8_scaled.safetensors；只載文字的部分，看圖的部分用不到）
設定從權重形狀推，推不出來的照官方（krea/Krea-2 的 config.json），不用連網。

**fp8。** 官方給 ComfyUI 的檔案是 scaled fp8（權重 float8 × 每個矩陣一個比例），12B 的主模型 13 GB。
載入時不還原（還原成 bf16 要 24 GB，32 GB 的電腦放不下），算的時候一層一層還原（ScaledFP8Linear），
結果跟先整個還原成 bf16 逐位元相同。文字編碼器也一樣（4.4 GB，還原要 8 GB）。
主模型的 NVFP4 版（4 位元）也能用（Nvfp4Linear）。官方另外的 int8、mxfp8 版還不支援：載入前只看檔頭就擋下（check_quant）。

**記憶體。** 跟 Z-Image 同一套（zimage.place）：文字編碼器直接映射檔案、不讀進系統記憶體，編碼時才搬上顯示卡
（RTX 5080、1024×1344：系統記憶體峰值 18.1 → 15.5 GB，編碼 1.7 → 0.9 秒）；主模型放得下就整個放，放不下就邊算邊搬。
"""

import json
import math
import time
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "assets" / "anima" / "tokenizer"        # Qwen 系同一份詞彙與對話模板（Qwen3-VL 也是）

# 官方 krea/Krea-2 的設定；形狀推得出來的會照檔案覆蓋
TRANSFORMER = {"in_channels": 64, "num_layers": 28, "attention_head_dim": 128, "num_attention_heads": 48,
               "num_key_value_heads": 12, "intermediate_size": 16384, "timestep_embed_dim": 256,
               "text_hidden_dim": 2560, "num_text_layers": 12, "text_num_attention_heads": 20,
               "text_num_key_value_heads": 20, "text_intermediate_size": 6912, "num_layerwise_text_blocks": 2,
               "num_refiner_text_blocks": 2, "axes_dims_rope": (32, 48, 48), "rope_theta": 1000.0, "norm_eps": 1e-5}
# Qwen3-VL 4B 文字部分；RoPE 是三軸的 mrope（文字三軸位置相同，等於一般的 RoPE），theta 五百萬
QWEN3VL = {"rope_parameters": {"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20],
                               "mrope_interleaved": True},
           "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "attention_bias": False, "use_cache": False}
# Turbo（蒸餾版）官方排程固定 mu＝1.15，等於 shift e^1.15 ≈ 3.16；Base 版照解析度算（1024² 約 2.5）
SHIFT = round(math.exp(1.15), 2)
DEFAULT_CFG, DEFAULT_STEPS = 1.0, 8
MAX_TOKENS = 512                     # 提示詞上限（官方固定長度），超過的截掉

PREFIX = "model.diffusion_model."


def _strip(keys):
    return [k[len(PREFIX):] if k.startswith(PREFIX) else k for k in keys]


def is_krea2(keys):
    """ComfyUI 的 Krea 2 單檔（Krea 原始命名，可能帶 model.diffusion_model. 前綴）"""
    ks = set(_strip(keys))
    return "txtfusion.projector.weight" in ks and "tproj.1.weight" in ks and "first.weight" in ks


# ---------------------------------------------------------------- 鍵名對照（Krea 原始 → diffusers）

_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_TOP = {"first.weight": "img_in.weight", "first.bias": "img_in.bias",
        "last.linear.weight": "final_layer.linear.weight", "last.linear.bias": "final_layer.linear.bias",
        "last.norm.scale": "final_layer.norm.weight", "last.modulation.lin": "final_layer.scale_shift_table",
        "tmlp.0.weight": "time_embed.linear_1.weight", "tmlp.0.bias": "time_embed.linear_1.bias",
        "tmlp.2.weight": "time_embed.linear_2.weight", "tmlp.2.bias": "time_embed.linear_2.bias",
        "tproj.1.weight": "time_mod_proj.weight", "tproj.1.bias": "time_mod_proj.bias",
        "txtmlp.0.scale": "txt_in.norm.weight", "txtmlp.1.weight": "txt_in.linear_1.weight",
        "txtmlp.1.bias": "txt_in.linear_1.bias", "txtmlp.3.weight": "txt_in.linear_2.weight",
        "txtmlp.3.bias": "txt_in.linear_2.bias", "txtfusion.projector.weight": "text_fusion.projector.weight"}
_BLOCK = {"attn.qknorm.qnorm.scale": "attn.norm_q.weight", "attn.qknorm.knorm.scale": "attn.norm_k.weight",
          "prenorm.scale": "norm1.weight", "postnorm.scale": "norm2.weight", "mod.lin": "scale_shift_table"}


def _block(rest):
    """一個 block 裡面的名字（blocks.N. 之後的部分）"""
    if rest in _BLOCK:
        return _BLOCK[rest]
    p = rest.split(".")
    if len(p) == 3 and p[0] == "attn" and p[1] in _ATTN:              # attn.wq.weight（.weight_scale）
        return f"attn.{_ATTN[p[1]]}.{p[2]}"
    if len(p) == 3 and p[0] == "mlp" and p[1] in ("up", "gate", "down"):
        return f"ff.{p[1]}.{p[2]}"
    return None


def rename(key):
    """Krea 原始的鍵名 → diffusers 的；不認得回傳 None"""
    if key in _TOP:
        return _TOP[key]
    p = key.split(".")
    if p[0] == "blocks" and len(p) > 2 and p[1].isdigit():
        inner = _block(".".join(p[2:]))
        return inner and f"transformer_blocks.{p[1]}.{inner}"
    if p[0] == "txtfusion" and len(p) > 3 and p[1] in ("layerwise_blocks", "refiner_blocks") and p[2].isdigit():
        inner = _block(".".join(p[3:]))
        return inner and f"text_fusion.{p[1]}.{p[2]}.{inner}"
    return None


def convert(raw):
    """整份對照。ComfyUI 量化的附屬資料（comfy_quant）丟掉；weight_scale 留著給 ScaledFP8Linear。
    每個 block 的調變表在檔案裡是攤平的（6 × 寬度），diffusers 是 (6, 寬度)"""
    out, unknown = {}, []
    for k, v in raw.items():
        k = k[len(PREFIX):] if k.startswith(PREFIX) else k
        if k.endswith(".comfy_quant"):
            continue
        d = rename(k)
        if d is None:
            unknown.append(k)
            continue
        if d.endswith("scale_shift_table") and v.dim() == 1:
            v = v.view(-1, v.shape[0] // 6 if d.startswith("transformer_blocks.") else v.shape[0] // 2)
        out[d] = v
    if unknown:
        raise RuntimeError(i18n.tr("eng.k.unknownKeys", n=len(unknown)) + "\n" + "\n".join(unknown[:5]))
    return out


def _count(sd, prefix):
    n = {int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)}
    return 1 + max(n) if n else 0


def _transformer_config(sd):
    cfg = dict(TRANSFORMER)
    head = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    hidden, in_ch = sd["img_in.weight"].shape
    t_head = sd["text_fusion.refiner_blocks.0.attn.norm_q.weight"].shape[0]
    t_dim = sd["txt_in.linear_1.weight"].shape[1]
    cfg.update(in_channels=in_ch, num_layers=_count(sd, "transformer_blocks."), attention_head_dim=head,
               num_attention_heads=hidden // head,
               num_key_value_heads=sd["transformer_blocks.0.attn.to_k.weight"].shape[0] // head,
               intermediate_size=sd["transformer_blocks.0.ff.up.weight"].shape[0],
               timestep_embed_dim=sd["time_embed.linear_1.weight"].shape[1], text_hidden_dim=t_dim,
               num_text_layers=sd["text_fusion.projector.weight"].shape[1],
               text_num_attention_heads=t_dim // t_head,
               text_num_key_value_heads=sd["text_fusion.refiner_blocks.0.attn.to_k.weight"].shape[0] // t_head,
               text_intermediate_size=sd["text_fusion.refiner_blocks.0.ff.up.weight"].shape[0],
               num_layerwise_text_blocks=_count(sd, "text_fusion.layerwise_blocks."),
               num_refiner_text_blocks=_count(sd, "text_fusion.refiner_blocks."))
    return cfg


def _qwen3vl_config(sd):
    from transformers import Qwen3VLTextConfig
    emb = sd["embed_tokens.weight"]
    head = sd["layers.0.self_attn.q_norm.weight"].shape[0]
    cfg = Qwen3VLTextConfig(vocab_size=emb.shape[0], hidden_size=emb.shape[1], head_dim=head,
                            intermediate_size=sd["layers.0.mlp.gate_proj.weight"].shape[0],
                            num_hidden_layers=_count(sd, "layers."),
                            num_attention_heads=sd["layers.0.self_attn.q_proj.weight"].shape[0] // head,
                            num_key_value_heads=sd["layers.0.self_attn.k_proj.weight"].shape[0] // head, **QWEN3VL)
    cfg._attn_implementation = "sdpa"
    return cfg


# ---------------------------------------------------------------- scaled fp8

_FP8_CLS = None


_FP8_MM = {}


def fp8_mm_ok(device):
    """這張顯示卡能不能直接用 fp8 算（RTX 40／50 系列、Hopper 以後，計算能力 8.9 起）"""
    import torch
    key = str(device)
    if key not in _FP8_MM:
        try:
            _FP8_MM[key] = torch.cuda.get_device_capability(device) >= (8, 9) and hasattr(torch, "_scaled_mm")
        except Exception:
            _FP8_MM[key] = False
    return _FP8_MM[key]


def fp8_linear():
    """權重用 fp8 存（ComfyUI 的 scaled fp8：weight × weight_scale）。兩種算法：
      直接算（_coco_mm）  輸入也壓成 fp8（整個張量一個比例，PyTorch 在 RTX 50 上只支援這種），用顯示卡的 fp8 運算
                          （torch._scaled_mm）。只用在官方標成可以的層（檔案的 _quantization_metadata 沒寫
                          full_precision_matrix_mult 的：Q、K、V、MLP 的 gate 與 up），跟 ComfyUI 一樣。
                          RTX 5080 大的層快 2 倍（16384×6144：8.6 → 4.2 ms），相對誤差約 2.7%
      先還原再算          還原成計算用的精度。比例存成 Python 浮點數：bf16 張量 × 浮點數在 PyTorch 裡用 32 位元乘、只捨入一次，
                          跟「先轉 32 位元乘比例再轉 bf16」（zimage.dequant 的還原）逐位元相同；
                          存成 0 維張量的話比例會先被捨入成 bf16
    顯示卡不支援 fp8 運算（RTX 30 以前）或形狀不合（寬度不是 16 的倍數）時一律先還原再算"""
    global _FP8_CLS
    if _FP8_CLS is None:
        import torch
        import torch.nn.functional as F

        class LowRankLinear(torch.nn.Linear):
            """量化存放的線性層共用的部分：LoRA 不併進權重（fp8 只有 3 位元、NVFP4 只有 1 位元的精度，
            小小的變化量捨入後大多不見），另外走一條低秩的路：y = Wx + up·(down·x)，up 已經乘好強度。
            子類別的 _compute 算 Wx"""
            _coco_lora = ()

            def forward(self, x):
                y = self._compute(x)
                for down, up in self._coco_lora:
                    y = y + F.linear(F.linear(x, down.to(device=x.device, dtype=x.dtype)),
                                     up.to(device=x.device, dtype=x.dtype))
                return y

            def add_lowrank(self, down, up):
                """放在顯示卡上（不歸 group offload 管，主模型邊算邊搬也不用跟著搬；一個 rank 64 的 LoRA 幾十 MB）"""
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                pair = (down.detach().to(dev, torch.bfloat16), up.detach().to(dev, torch.bfloat16))
                self._coco_lora = (*self._coco_lora, pair)

            def clear_lowrank(self):
                self._coco_lora = ()

        class ScaledFP8Linear(LowRankLinear):
            weight_scale = 1.0
            _coco_mm = False            # 官方標成可以直接用 fp8 算的層

            def _direct(self, x):
                k, n = self.in_features, self.out_features
                if not (self._coco_mm and x.is_cuda and k % 16 == 0 and n % 16 == 0 and fp8_mm_ok(x.device)):
                    return None
                x2 = x.reshape(-1, k)
                xs = x2.abs().amax().float().clamp_(min=1e-12).div_(448.0)
                ws = torch.full((), self.weight_scale, dtype=torch.float32, device=x.device)
                y = torch._scaled_mm((x2 / xs).to(self.weight.dtype), self.weight.t(), scale_a=xs, scale_b=ws,
                                     bias=self.bias.to(x.dtype) if self.bias is not None else None, out_dtype=x.dtype)
                return y.view(*x.shape[:-1], n)

            def _compute(self, x):
                y = self._direct(x)
                if y is None:
                    w = self.weight.to(x.dtype) * self.weight_scale
                    y = F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)
                return y

        class Nvfp4Linear(LowRankLinear):
            """NVFP4（4 位元）：weight 每個位元組裝兩個 E2M1、weight_scale 每 16 個一個 fp8 比例（cuBLAS 的分塊排列）、
            weight_scale_2 整個矩陣一個比例（兩個都是緩衝區）。官方把每一層都標成 full_precision_matrix_mult，
            所以一律先還原成計算精度再算（comfy_kitchen 的 dequantize_nvfp4，跟 ComfyUI 一樣）"""

            def _compute(self, x):
                ck = _nvfp4_kitchen()
                w = ck.dequantize_nvfp4(self.weight, self.weight_scale_2, self.weight_scale, output_type=x.dtype)
                return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

        _FP8_CLS = ScaledFP8Linear
        _CLASSES["nvfp4"] = Nvfp4Linear
        _CLASSES["lowrank"] = LowRankLinear
    return _FP8_CLS


_CLASSES = {}


def nvfp4_linear():
    fp8_linear()
    return _CLASSES["nvfp4"]


def lowrank_linear():
    """量化存放的線性層共用的基底（LoRA 掛低秩的路）；Qwen Image 2.1 的 Int8Linear 也用"""
    fp8_linear()
    return _CLASSES["lowrank"]


def _nvfp4_kitchen():
    """NVFP4 的還原要 comfy_kitchen（ComfyUI 官方的運算套件）；沒有就說清楚"""
    try:
        from core.qwen21 import kitchen
    except ImportError:
        from qwen21 import kitchen
    ck = kitchen()
    if ck is None or not callable(getattr(ck, "dequantize_nvfp4", None)):
        raise RuntimeError(i18n.tr("eng.k.needKitchen"))
    return ck


def _fp8_dtypes():
    import torch
    return {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}


def _check_fp4(keys, what):
    if any(k.endswith("weight_scale_2") for k in keys):
        raise RuntimeError(i18n.tr("eng.fp4", what=what, alt=i18n.tr("eng.alt.fp8bf16")))


# ComfyUI 的量化格式（comfy/quant_ops.py 的 QUANT_ALGOS）：主模型認得 scaled fp8 與 NVFP4，文字編碼器只認得 fp8。
# 官方的 Krea 2 另外有 int8（int8_tensorwise＋convrot）與 mxfp8 版；不擋的話 int8 會靜靜載入、生成時才出看不懂的型別錯誤，
# mxfp8 的分塊比例被當成每列的比例乘，載入時出形狀錯誤，而且都是讀完整個 13 GB 才出錯
QUANT_OK = {"float8_e4m3fn", "float8_e5m2", "nvfp4"}
QUANT_OK_TE = {"float8_e4m3fn", "float8_e5m2"}
_QUANT_LABEL = {"int8_tensorwise": "int8", "mxfp8": "mxfp8", "nvfp4": "NVFP4", "convrot_w4a4": "W4A4",
                "asym_w4a8_int8": "W4A8", "w6a8_int8": "W6A8"}


def quant_formats(path):
    """不讀權重、只看檔頭：檔案用了哪些 ComfyUI 量化格式。每層的 comfy_quant（json 存成 uint8，只讀這幾個小張量）
    與檔頭 metadata 的 _quantization_metadata 都看；另外照權重的型別判斷，沒有這兩份資料的檔案也認得：
    int8 的 weight → int8_tensorwise；fp8 的 weight 配上分塊（多維或 e8m0）的比例 → mxfp8（每列一個比例的 fp8 支援，不算）"""
    import struct
    fmts = set()
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        try:
            layers = json.loads((header.get("__metadata__") or {}).get("_quantization_metadata") or "{}").get("layers") or {}
        except (ValueError, AttributeError):
            layers = {}
        fmts |= {str(c.get("format", "")) for c in layers.values() if isinstance(c, dict)}
        for k, v in header.items():
            if k.endswith(".comfy_quant") and isinstance(v, dict):
                a, b = v["data_offsets"]
                fh.seek(8 + n + a)
                try:
                    fmts.add(str(json.loads(fh.read(b - a).decode("utf-8")).get("format", "")))
                except (ValueError, UnicodeDecodeError, AttributeError):
                    pass
    for k, v in header.items():
        if not k.endswith(".weight") or not isinstance(v, dict):
            continue
        if v.get("dtype") == "I8":
            fmts.add("int8_tensorwise")
        elif v.get("dtype") in ("F8_E4M3", "F8_E5M2"):
            s = header.get(k + "_scale")
            if isinstance(s, dict) and (len(s.get("shape", ())) > 1 or s.get("dtype") in ("U8", "F8_E8M0")):
                fmts.add("mxfp8")
    fmts.discard("")
    return fmts


def check_quant(path, what, ok=QUANT_OK, instead=None):
    """用了還不支援的量化格式就直接說（載入權重之前）"""
    bad = sorted(quant_formats(path) - set(ok))
    if bad:
        label = "、".join(_QUANT_LABEL.get(f, f) for f in bad)
        raise RuntimeError(i18n.tr("eng.quantUnsupported", what=what, label=label, alt=instead or i18n.tr("eng.alt.fp8nv")))


def _is_nvfp4(sd, mod):
    import torch
    w = sd.get(mod + ".weight")
    return w is not None and w.dtype == torch.uint8 and (mod + ".weight_scale_2") in sd


def prepare_weights(sd, dtype, fp32_names=()):
    """把對照好的權重整理成可以直接指派的樣子，回傳 (權重, {模組路徑: 比例，NVFP4 的是 "nvfp4"})。
      NVFP4                  → 原樣留著（weight、分塊比例、整體比例），模組之後換成 Nvfp4Linear
      fp8 ＋ 純量比例        → 原樣留著，模組之後換成 ScaledFP8Linear
      fp8 ＋ 每列一個比例    → 載入時還原成 dtype（少見，只有少數量化工具這樣存）
      其他浮點數             → dtype；fp32_names（模型要求 32 位元的，例如 Krea 2 的 RMSNorm）→ float32"""
    import torch
    fp8 = _fp8_dtypes()
    scales = {k[:-len(".weight_scale")]: sd.pop(k) for k in [k for k in sd if k.endswith(".weight_scale")]}
    out, keep = {}, {}
    for k, v in sd.items():
        mod = k[:-len(".weight")] if k.endswith(".weight") else None
        if mod is not None and v.dtype == torch.uint8 and (mod + ".weight_scale_2") in sd:
            out[k], out[mod + ".weight_scale"] = v, scales[mod]
            out[mod + ".weight_scale_2"] = sd[mod + ".weight_scale_2"].to(torch.float32)
            keep[mod] = "nvfp4"
            continue
        if k.endswith(".weight_scale_2"):
            continue
        if v.dtype in fp8:
            mod = k[:-len(".weight")] if k.endswith(".weight") else None
            s = scales.get(mod)
            if s is not None and s.numel() == 1:
                out[k], keep[mod] = v, float(s)
                continue
            w = v.to(torch.float32)
            if s is not None:
                w = w * (s.to(torch.float32).view(-1, *([1] * (w.dim() - 1))) if s.dim() == 1 else s.to(torch.float32))
            out[k] = w.to(dtype)
        elif v.is_floating_point():
            out[k] = v.to(torch.float32 if any(n in k.split(".") for n in fp32_names) else dtype)
        else:
            out[k] = v
    return out, keep


def swap_fp8(model, keep, weights=None):
    """把要用 fp8 算的 Linear 換成 ScaledFP8Linear、NVFP4 的換成 Nvfp4Linear（在載入權重之前，模型還在 meta 上）。
    NVFP4 的 weight 形狀跟原本不同（兩個一組裝成一個位元組），連同兩個比例的緩衝區照 weights 的形狀先換好"""
    import torch
    cls, nv = fp8_linear(), nvfp4_linear()
    for name, s in keep.items():
        m = model.get_submodule(name)
        if not isinstance(m, torch.nn.Linear):
            raise RuntimeError(i18n.tr("eng.k.notLinear", name=name))
        if s == "nvfp4":
            m.__class__ = nv
            meta = lambda k: torch.empty(weights[f"{name}.{k}"].shape, dtype=weights[f"{name}.{k}"].dtype, device="meta")  # noqa: E731
            m.weight = torch.nn.Parameter(meta("weight"), requires_grad=False)
            m.register_buffer("weight_scale", meta("weight_scale"))
            m.register_buffer("weight_scale_2", meta("weight_scale_2"))
        else:
            m.__class__ = cls
            m.weight_scale = s
    model.requires_grad_(False)


def _build(cls, cfg, sd, dtype, label, log):
    try:
        from core.anima import _build as build
    except ImportError:
        from anima import _build as build
    weights, keep = prepare_weights(sd, dtype, getattr(cls, "_keep_in_fp32_modules", None) or ())
    model = build(cls, cfg, weights, label, log, prepare=lambda m: swap_fp8(m, keep, weights))
    nv = sum(1 for v in keep.values() if v == "nvfp4")
    if nv:
        _nvfp4_kitchen()
        log(label + "　" + i18n.tr("eng.k.nvStore", n=nv))
    if len(keep) > nv:
        log(label + "　" + i18n.tr("eng.fp8StoreN", n=len(keep) - nv))
    return model, bool(keep)


def fp8_mm_layers(metadata):
    """檔案的 _quantization_metadata（ComfyUI 存的每層量化設定）→ 可以直接用 fp8 算的層（diffusers 的模組路徑）。
    官方標了 full_precision_matrix_mult 的（attn 的 gate、wo、MLP 的 down）要用完整精度算；沒有這份資料的檔案一律先還原再算"""
    try:
        layers = json.loads((metadata or {}).get("_quantization_metadata") or "{}").get("layers") or {}
    except (ValueError, AttributeError):
        return set()
    out = set()
    for name, conf in layers.items():
        if not isinstance(conf, dict) or "float8" not in str(conf.get("format", "")) or conf.get("full_precision_matrix_mult"):
            continue
        d = rename((name[len(PREFIX):] if name.startswith(PREFIX) else name) + ".weight")
        if d:
            out.add(d[:-len(".weight")])
    return out


def enable_fp8_mm(model, names, log=print):
    """把可以直接用 fp8 算的層打開（ScaledFP8Linear._coco_mm）。回傳打開了幾層"""
    import torch
    cls, n = fp8_linear(), 0
    for name, m in model.named_modules():
        if isinstance(m, cls):
            m._coco_mm = name in names
            n += m._coco_mm
    if n:
        gpu = torch.cuda.is_available() and fp8_mm_ok(torch.device("cuda"))
        log("　" + i18n.tr("eng.k.fp8Compute", n=n) + ("" if gpu else i18n.tr("eng.k.noFp8Gpu")))
    return n


def weights_label(transformer, direct):
    """圖的參數裡的 Model dtype：NVFP4（4 位元存放，出圖跟 fp8 版不同）、fp8 W8A8（有直接用 fp8 算，
    跟先還原再算有極小的差異）；其他的照 _coco_fp8 記成 fp8 或不記"""
    import torch
    if any(isinstance(m, nvfp4_linear()) for m in transformer.modules()):
        return "nvfp4"
    if direct and torch.cuda.is_available() and fp8_mm_ok(torch.device("cuda")):
        return "fp8 W8A8"
    return ""


# ---------------------------------------------------------------- 載入

_PIPE = None


def _pipe_class():
    """跟 Z-Image 一樣：文字編碼器平常在系統記憶體，diffusers 挑到它會整條判成 CPU。有顯示卡就一律在顯示卡上算"""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import Krea2Pipeline

        class CocoKrea2Pipeline(Krea2Pipeline):
            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _PIPE = CocoKrea2Pipeline
    return _PIPE


def _te_weights(raw):
    """Qwen3-VL 的檔案只取文字的部分：ComfyUI 的是 model.layers…、Hugging Face 的是 model.language_model.layers…；
    看圖的（model.visual.）與輸出層（lm_head）用不到"""
    out, visual = {}, False
    for k, v in raw.items():
        if k.startswith(("model.visual.", "visual.")):
            visual = True
            continue
        for pre in ("model.language_model.", "model."):
            if k.startswith(pre):
                if not k.endswith(".comfy_quant"):
                    out[k[len(pre):]] = v
                break
    return out, visual


def text_encoder_from(te_path, dtype, dim, log=print):
    """Qwen3-VL 4B 的文字部分。權重直接映射檔案（zimage._mapped），fp8 的層照原樣存放，不另外讀進系統記憶體。
    編碼時才搬上顯示卡（zimage._to）"""
    from transformers import Qwen3VLTextModel
    check_quant(te_path, i18n.tr("eng.lbl.te"), QUANT_OK_TE, i18n.tr("eng.alt.fp8bf16"))
    raw, mm = _z()._mapped(te_path)
    _check_fp4(raw, i18n.tr("eng.lbl.te"))
    sd, visual = _te_weights(raw)
    del raw
    if "embed_tokens.weight" not in sd:
        raise RuntimeError(i18n.tr("eng.notQwen3vl", name=Path(te_path).name))
    if sd["embed_tokens.weight"].shape[1] != dim:
        raise RuntimeError(i18n.tr("eng.teDim", got=sd['embed_tokens.weight'].shape[1], want=dim, model="Krea 2", te="Qwen3-VL 4B"))
    if not visual:
        log("[!] " + i18n.tr("eng.noVision", model="Krea 2", te="Qwen3-VL 4B", hint=i18n.tr("eng.k.noVisionHint")))
    te, _ = _build(Qwen3VLTextModel, _qwen3vl_config(sd), sd, dtype, "　text_encoder", log)
    te._coco_mapped, te._coco_file = True, (mm, str(te_path))
    return te.eval()


def load(model_path, vae_path, te_path, log=print, dtype=None, fp8=False):
    import json
    import inspect
    import torch
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer

    try:
        from core.anima import ASSETS, _build as build, lean_vae
    except ImportError:
        from anima import ASSETS, _build as build, lean_vae

    dtype = dtype or torch.bfloat16
    if not vae_path or not te_path:
        raise RuntimeError(i18n.tr("eng.needParts", model="Krea 2", vae=i18n.tr("eng.vaeOf", name="Qwen Image", file="qwen_image_vae"), te=i18n.tr("srv.part.hint", name="Qwen3-VL 4B", file="qwen3vl_4b")))
    for p, what in ((vae_path, "VAE"), (te_path, i18n.tr("eng.teCap"))):
        if not Path(p).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=p))

    log(i18n.tr("eng.loading", model="Krea 2"))
    check_quant(model_path, i18n.tr("eng.lbl.main"))
    raw = load_file(str(model_path))
    from safetensors import safe_open
    with safe_open(str(model_path), framework="pt") as f:
        mm_names = fp8_mm_layers(f.metadata())
    sd = convert(raw)
    del raw
    tcfg = _transformer_config(sd)
    log("　" + i18n.tr("eng.tf3", layers=tcfg['num_layers'], heads=tcfg['num_attention_heads'], dim=tcfg['num_attention_heads'] * tcfg['attention_head_dim']))
    transformer, is_fp8 = _build(Krea2Transformer2DModel, tcfg, sd, dtype, "　transformer", log)
    del sd
    direct = enable_fp8_mm(transformer, mm_names, log) if is_fp8 else 0
    if fp8 and not is_fp8:
        # bf16 的檔案：權重用 fp8 存、算的時候一層一層轉回 bf16（跟 Z-Image 的同一個設定）
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        is_fp8 = True
        log("　" + i18n.tr("eng.fp8Store"))

    vcfg = json.loads((ASSETS / "vae" / "config.json").read_text("utf-8"))
    accept = set(inspect.signature(AutoencoderKLQwenImage.__init__).parameters)
    vcfg = {k: v for k, v in vcfg.items() if not k.startswith("_") and k in accept}
    vsd = convert_wan_vae_to_diffusers(load_file(str(vae_path)))
    if "decoder.conv_in.weight" not in vsd or vsd["decoder.conv_in.weight"].shape[1] != vcfg.get("z_dim", 16):
        raise RuntimeError(i18n.tr("eng.notQwenVae", name=Path(vae_path).name))
    vae = build(AutoencoderKLQwenImage, vcfg, {k: v.to(dtype) if v.is_floating_point() else v for k, v in vsd.items()},
                "　vae", log)
    lean_vae(vae)
    del vsd

    text_encoder = text_encoder_from(te_path, dtype, tcfg["text_hidden_dim"], log)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(TOKENIZER))
    tokenizer.model_max_length = MAX_TOKENS                 # 生成台的 token 計數照這個上限顯示
    # 排程用固定的 shift（use_dynamic_shifting 關掉，管線算的 mu 就不用）：Turbo 官方的 mu＝1.15 就是 shift e^1.15
    pipe = _pipe_class()(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=SHIFT,
                                                                    use_dynamic_shifting=False),
                          vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.update_components = lambda **kw: pipe.register_modules(**kw)     # 取樣器沿用 Anima 那套（見 zimage.load）
    pipe._coco_is_krea2 = True
    pipe._coco_fp8 = is_fp8
    pipe._coco_weights = weights_label(transformer, direct)
    pipe._coco_embeds = None
    log("　" + i18n.tr("eng.assembled"))
    return pipe


# ---------------------------------------------------------------- LoRA

_TOP_LINEAR = ("first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1", "txtmlp.1", "txtmlp.3", "txtfusion.projector")
_BLOCK_LINEAR = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.up", "mlp.gate", "mlp.down")


def lora_alias(transformer):
    """LoRA 的模組名（Krea 原始命名：kohya 的 lora_unet_blocks_0_attn_wq 去掉前綴，
    或 ComfyUI 的 diffusion_model.blocks.0.attn.wq 換成底線）→ diffusers 的模組路徑。
    用 diffusers 命名訓練的（transformer.transformer_blocks.0.attn.to_q）本來就查得到"""
    names = list(_TOP_LINEAR)
    for i in range(len(getattr(transformer, "transformer_blocks", []))):
        names += [f"blocks.{i}.{n}" for n in _BLOCK_LINEAR]
    fusion = getattr(transformer, "text_fusion", None)
    for kind in ("layerwise_blocks", "refiner_blocks"):
        for i in range(len(getattr(fusion, kind, []))):
            names += [f"txtfusion.{kind}.{i}.{n}" for n in _BLOCK_LINEAR]
    have = {n for n, _ in transformer.named_modules()}
    out = {}
    for n in names:
        d = rename(n + ".weight")
        if d and d[:-len(".weight")] in have:
            out[n.replace(".", "_")] = d[:-len(".weight")]
    return out


# ---------------------------------------------------------------- 記憶體（跟 Z-Image 同一套）

def _z():
    try:
        from core import zimage
    except ImportError:
        import zimage
    return zimage


def parked(vram_mode):
    return ("text_encoder",)


def place(pipe, vram_mode, log=print):
    _z().place(pipe, vram_mode, log, label="Krea 2")


def configure_memory(pipe, vram_mode, log=print):
    _z()._to(pipe, "text_encoder", "cpu")


# ---------------------------------------------------------------- 提示詞

def encode(pipe, prompt, negative, use_cfg, log=print, max_len=MAX_TOKENS):
    """回傳 (正面, 正面遮罩, 負面, 負面遮罩)；同一組提示詞直接用上次的。
    文字編碼器搬上 GPU 編碼完就搬回系統記憶體；主模型整個放在顯示卡、又擠不下時先讓開"""
    import torch
    Z = _z()
    key = (prompt, negative if use_cfg else None, max_len)
    if pipe._coco_embeds and pipe._coco_embeds[0] == key:
        return pipe._coco_embeds[1]
    te, tr = pipe.text_encoder, pipe.transformer
    moved_tr = False
    if torch.cuda.is_available():
        if Z._device(tr) == "cuda" and not getattr(tr, "_coco_grouped", False):
            free, _ = torch.cuda.mem_get_info()
            if free < Z._bytes(te) + (1.5 * 2 ** 30):
                Z._to(pipe, "transformer", "cpu")
                moved_tr = True
        Z._to(pipe, "text_encoder", "cuda")
    t = time.perf_counter()
    try:
        with torch.no_grad():
            dev = next(te.parameters()).device
            pos, pmask = pipe.encode_prompt(prompt=prompt, device=dev, max_sequence_length=max_len)
            neg = nmask = None
            if use_cfg:
                neg, nmask = pipe.encode_prompt(prompt=negative or "", device=dev, max_sequence_length=max_len)
    finally:
        Z._to(pipe, "text_encoder", "cpu")
        if moved_tr:
            Z._to(pipe, "transformer", "cuda")
    log(i18n.tr("eng.encodeTime", model="Krea 2", s=f"{time.perf_counter() - t:.1f}") + (i18n.tr("eng.trMoved") if moved_tr else ""))
    out = (pos, pmask, neg, nmask)
    pipe._coco_embeds = (key, out)
    return out


# ---------------------------------------------------------------- img2img／局部重繪

def _pack(pipe, x):
    """(1, 16, h, w) → 管線吃的 2×2 打包格式 (1, h/2·w/2, 64)"""
    b, c, h, w = x.shape
    return pipe._pack_latents(x, b, c, h, w)


def _encode_init(pipe, req):
    """底圖 → 正規化過的 latent（(1, 16, h/8, w/8)，float32）。Qwen Image 的 VAE 解碼時是 latent × std + mean
    （跟 Anima 同一顆，anima._encode_init），這裡反過來。用分佈的眾數不抽樣：同一張底圖每次編出一樣的 latent"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P
    img = P.load_init_image(req["init_image"], req["width"], req["height"])
    vae = pipe.vae
    w = next(vae.parameters())
    x = torch.from_numpy(np.array(img)).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None, :, None].to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1).to(z)
    std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1).to(z)
    return ((z - mean) / std)[:, :, 0].to(torch.float32)


def _prepare_img2img(pipe, req, generator, log):
    """跟 Anima、Z-Image 同一套：完整排程照文生圖算（管線預設的 linspace(1, 1/n)，shift 在排程器裡套），
    跳過前面幾步、從切點的 σ 開始；起點 x = (1−σ)·x₀ + σ·ε。局部重繪另外每一步把遮罩外換回底圖的加噪版本
    （anima.mask_steps）。latent 先在 (1, 16, h, w) 算好再打包成管線的格式。
    回傳 (打包好的 latents, sigmas, 跑幾步, 還原函式)"""
    import numpy as np
    import torch
    try:
        from core import anima as AN, pipeline as P
    except ImportError:
        import anima as AN
        import pipeline as P

    n_all, run = P.img2img_steps(req["steps"], req.get("strength", 0.5))
    skip = n_all - run
    sigmas = np.linspace(1.0, 1 / n_all, n_all)             # 跟文生圖同一條（Krea2Pipeline 的預設）
    sched = pipe.scheduler
    sched.set_timesteps(sigmas=sigmas, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    s0 = float(sched.sigmas[skip])

    x0 = _encode_init(pipe, req)
    eps = torch.randn(x0.shape, generator=generator, device=generator.device, dtype=torch.float32).to(x0.device)
    x0p, epsp = _pack(pipe, x0), _pack(pipe, eps)
    latents = (1.0 - s0) * x0p + s0 * epsp
    restores = [AN._once_partway(sched, skip)]
    if req.get("_mask") is not None:
        try:
            from core import inpaint as IP
        except ImportError:
            import inpaint as IP
        h, w = x0.shape[-2:]
        m = torch.from_numpy(IP.latent_mask(req["_mask"], w, h))[None, None].to(x0.device).expand_as(x0)
        restores.append(AN.mask_steps(sched, x0p, epsp, _pack(pipe, m.contiguous())))
    strength = min(1.0, max(P.STRENGTH_MIN, float(req.get("strength", 0.5))))
    log(i18n.tr("eng.i2iFrom", model="Krea 2", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))

    def restore():
        for r in reversed(restores):
            r()
    return latents, sigmas, run, restore


# ---------------------------------------------------------------- 生成

def generate(pipe, req, on_step=None, log=print):
    """回傳 (PIL Image, seed)。介面與 pipeline.generate 對齊"""
    import torch
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

    # CFG 照一般的定義；diffusers 的 Krea 2 寫成 正 + w ×（正 − 負），w＝CFG − 1（跟 Z-Image 一樣）。
    # CFG 1（含以下）不跑負面：Turbo 官方就是這樣用
    cfg = float(req.get("cfg", DEFAULT_CFG))
    use_cfg = cfg > 1.0
    prompt, negative = Z._clean(req.get("prompt", "")), Z._clean(req.get("negative", ""))
    if prompt != (req.get("prompt") or "").strip():
        log(i18n.tr("eng.noWeightSyntax", model="Krea 2"))
    pos, pmask, neg, nmask = encode(pipe, prompt, negative, use_cfg, log)

    if torch.cuda.is_available() and not getattr(pipe.transformer, "_coco_grouped", False):
        Z._to(pipe, "transformer", "cuda")
    Z._to(pipe, "vae", "cuda" if torch.cuda.is_available() else "cpu")

    total = int(req["steps"])
    extra, restore = {}, None
    if req.get("init_image"):
        latents, sigmas, total, restore = _prepare_img2img(pipe, req, generator, log)
        extra = {"latents": latents, "sigmas": sigmas}

    def cb(p, i, t, kw):
        if on_step:
            on_step(i + 1, total)
        return kw

    try:
        out = pipe(prompt_embeds=pos, prompt_embeds_mask=pmask,
                   negative_prompt_embeds=neg if use_cfg else None,
                   negative_prompt_embeds_mask=nmask if use_cfg else None,
                   height=int(req["height"]), width=int(req["width"]), num_inference_steps=int(req["steps"]),
                   guidance_scale=cfg - 1.0 if use_cfg else 0.0, generator=generator, callback_on_step_end=cb, **extra)
    finally:
        if restore:
            restore()
        scheduler._coco_generator = None
    return out.images[0], seed
