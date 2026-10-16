# -*- coding: utf-8 -*-
"""
qwen21.py —— Qwen Image 2.1（通義千問的 7B 單流 transformer；文字與圖接成一條序列、區塊因果 attention）

模型檔是 ComfyUI 的拆分式單檔，命名跟 diffusers 一樣（只有 MLP 的 gate 與 up 合成一個 gate_up）。
VAE 與文字編碼器要另外選：
  VAE          Qwen Image 2.1 的 VAE（qwen_image_2.1_vae；64 個 latent 通道、縮小 16 倍，輸出 RGBA）
  文字編碼器   Qwen3-VL 8B（qwen3vl_8b；只載文字的部分）
diffusers 0.39／0.40 還沒有 2.1：模型、VAE、管線三個檔案從 diffusers 主線抄進 core/vendor/qwenimage21/
（Apache 2.0，見那裡的 README）。

**int8 + convrot。** ComfyUI 的 int8 檔：權重先每 256 個一組乘上正規化的 Hadamard 矩陣（旋轉）再以每列一個比例
存成 int8，W = (q × 比例) · H。載入時不還原（主模型 7.3 GB、文字編碼器 7.5 GB）。主模型的算法：
  直接算（W8A8）  一律用這個。輸入也旋轉、壓成 int8，用顯示卡的 int8 運算，跟 ComfyUI 一樣交給 comfy_kitchen（Apache 2.0）
                  合併過的 kernel。RTX 5080 每層快 4～5 倍，相對誤差約 1%（輸入的量化誤差）；
                  1024² 整步 0.96 → 0.37 秒（邊算邊搬 0.43 秒），出圖平均差 0.5/255
  先還原再算      沒裝 comfy_kitchen 或它用不了時退回這個（記錄裡會說）。Int8Linear 把權重還原成 bf16 再算，
                  跟整個還原成 bf16 的模型逐位元相同；q · H 在 32 位元裡是精確的（整數乘 ±1/16），只在最後乘比例轉 bf16 時捨入一次
兩種畫出來平均只差 0.5/255，看不出來，所以不做成設定（set_fast 留給測試與除錯）。文字編碼器一次只編一句，一律先還原再算。
Hadamard 的定義與存法照 comfy_kitchen 的說明。

**排程。** 官方：shift 依解析度（mu＝0.5 ～ 0.9，序列長 256 ～ 8192），最後把 σ 拉到 0.02。CoCo 先把整條 σ 算好
交給取樣器（取樣器自己的 shift 設成 1），所以每一種取樣器都照官方的排程走。介面上的 shift 是「1024² 時的 shift」，
預設 2.0 就是官方值，其他解析度照官方的斜率自動調整。

**記憶體。** 跟 Z-Image、Krea 2 同一套（zimage.place）。主模型 7.3 GB，16 GB 的卡整個放得下。
文字編碼器（7 GB，修圖再加看圖的 1 GB）不另外讀進系統記憶體：權重直接映射檔案（_mapped），編碼時一層一層搬上顯示卡，
編完把讀過的頁面還給系統快取（_trim）。以前整份複製並鎖定在系統記憶體，系統記憶體多佔 7 GB，
工作管理員還把鎖定的那份算成「共用 GPU 記憶體」，看起來像顯示卡溢出。
"""

import json
import math
import time
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "assets" / "anima" / "tokenizer"        # Qwen 系同一份詞彙（含 <|image_pad|> 這些看圖用的記號）

PREFIX = "model.diffusion_model."
# 官方 Qwen/Qwen-Image-2.1 的 text_encoder/config.json（Qwen3-VL 8B 的文字部分）
QWEN3VL = {"rope_parameters": {"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20],
                               "mrope_interleaved": True},
           "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "attention_bias": False, "use_cache": False}
# 官方 text_encoder/config.json 的 vision_config（Qwen3-VL 8B 看圖的部分；修圖時才載入）
VISION = {"deepstack_visual_indexes": [8, 16, 24], "depth": 27, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152,
          "in_channels": 3, "intermediate_size": 4304, "num_heads": 16, "num_position_embeddings": 2304,
          "out_hidden_size": 4096, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}
VISION_TOKENS = {"image_token_id": 151655, "video_token_id": 151656, "vision_start_token_id": 151652,
                 "vision_end_token_id": 151653}
# 官方 processor/preprocessor_config.json（Qwen2VLImageProcessor：16 像素一格、2×2 合成一個圖片 token）
IMAGE_PROC = {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
              "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16,
              "resample": 3, "rescale_factor": 1 / 255, "size": {"longest_edge": 16777216, "shortest_edge": 65536},
              "temporal_patch_size": 2}
# 官方 scheduler/scheduler_config.json：exponential 的動態 shift、終點 σ 0.02
BASE_SEQ, MAX_SEQ, BASE_MU, MAX_MU, TERMINAL = 256, 8192, 0.5, 0.9, 0.02
SEQ_1024 = (1024 // 16) ** 2


def official_mu(seq):
    return BASE_MU + (MAX_MU - BASE_MU) * (seq - BASE_SEQ) / (MAX_SEQ - BASE_SEQ)


SHIFT_1024 = math.exp(official_mu(SEQ_1024))       # ≈ 2.0007
SHIFT = round(SHIFT_1024, 2)                         # 介面的預設值
DEFAULT_CFG, DEFAULT_STEPS = 1.0, 40                 # 官方：不開 CFG、40 步
OPAQUE = 240                                         # 透明度最低值在這以上就當成不透明
MULTIPLE = 32                                        # 寬高要 32 的倍數（VAE 縮小 16 倍、文字的一格對 2×2 個 latent）
SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = ("<|im_start|>system\n" + SYS_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def _strip(keys):
    return [k[len(PREFIX):] if k.startswith(PREFIX) else k for k in keys]


def is_qwen21(keys):
    """ComfyUI 的 Qwen Image 2.1 單檔：共用的 modulation、txt_in 的 in_layer／out_layer、img_mlp"""
    ks = set(_strip(keys))
    return "modulation.1.weight" in ks and "txt_in.in_layer.weight" in ks \
        and any(".img_mlp." in k for k in ks)


# ---------------------------------------------------------------- int8 + convrot

_H = {}


def hadamard(size, device=None):
    """正規化的「regular」Hadamard 矩陣（4×4 的基本塊做 Kronecker 積，再除以 √size）；對稱且正交，H·H = I"""
    import torch
    key = (size, str(device))
    if key not in _H:
        if size < 4 or 4 ** round(math.log(size, 4)) != size:
            raise ValueError(i18n.tr("eng.q.convrot", size=size))
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        _H[key] = (h / math.sqrt(size)).to(device)
    return _H[key]


def dequant_rows(q, scale, group):
    """int8 的列 → 32 位元：(q · H) × 比例。q 是 (列數, 寬)，比例是 (列數,)"""
    import torch
    w = q.to(torch.float32)
    if group:
        n, d = w.shape
        w = (w.view(n, d // group, group) @ hadamard(group, w.device)).view(n, d)
    return w * scale.to(torch.float32).view(-1, 1)


_CLS = {}
_KITCHEN = {}


def kitchen():
    """comfy_kitchen 模組；沒裝或用不了回傳 None（原因記在 _KITCHEN["why"]）"""
    if "mod" not in _KITCHEN:
        try:
            import comfy_kitchen as ck
            _KITCHEN["mod"] = ck if callable(getattr(ck, "int8_linear", None)) else None
            _KITCHEN["why"] = "" if _KITCHEN["mod"] else i18n.tr("eng.q.noInt8")
        except Exception as e:                       # 沒裝、或 CUDA 執行檔載不進來
            _KITCHEN["mod"], _KITCHEN["why"] = None, f"{type(e).__name__}: {str(e)[:80]}"
    return _KITCHEN["mod"]


def int8_classes():
    """(Int8Linear, Int8Embedding)。權重 int8、每列一個比例（weight_scale 緩衝區，跟著模組搬上搬下）、
    _coco_group 是旋轉的分組大小（0＝沒旋轉）。算的時候整個矩陣還原成計算精度"""
    if not _CLS:
        import torch
        import torch.nn.functional as F
        try:
            from core.krea2 import lowrank_linear
        except ImportError:
            from krea2 import lowrank_linear

        class Int8Linear(lowrank_linear()):
            """LoRA 跟 Krea 2 的量化層一樣不併進權重、掛低秩的路（int8 併進去小的變化量會被捨掉）"""
            _coco_group = 0
            _coco_fast = False          # True：W8A8（comfy_kitchen）；False：先還原成計算精度再算

            def _compute(self, x):
                if self._coco_fast and x.is_cuda:
                    ck = kitchen()
                    if ck is not None:
                        g = self._coco_group
                        return ck.int8_linear(x, self.weight, self.weight_scale,
                                              self.bias.to(x.dtype) if self.bias is not None else None,
                                              out_dtype=x.dtype, convrot=bool(g), convrot_groupsize=g or 256)
                w = dequant_rows(self.weight, self.weight_scale, self._coco_group).to(x.dtype)
                return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

        class Int8Embedding(torch.nn.Embedding):
            _coco_group = 0

            def forward(self, ids):
                # 在權重所在的地方查表：文字編碼器邊算邊搬時，看圖的流程會在文字編碼器還沒搬上來前先查（只查用到的幾列）
                flat = ids.reshape(-1).to(self.weight.device)
                rows = dequant_rows(self.weight[flat], self.weight_scale[flat], self._coco_group)
                return rows.view(*ids.shape, -1).to(device=ids.device, dtype=torch.bfloat16)

        _CLS.update(linear=Int8Linear, embedding=Int8Embedding)
    return _CLS["linear"], _CLS["embedding"]


def quant_info(raw, strip=lambda k: k):
    """各模組的 comfy_quant（json 存成 uint8）→ {模組路徑: 旋轉的分組大小（0＝沒旋轉）}；不支援的格式直接說"""
    out = {}
    for k, v in raw.items():
        if not k.endswith(".comfy_quant"):
            continue
        try:
            conf = json.loads(bytes(v.tolist()).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        fmt = conf.get("format", "")
        if fmt != "int8_tensorwise":
            raise RuntimeError(i18n.tr("eng.q.fmt", fmt=fmt))
        out[strip(k[:-len(".comfy_quant")])] = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 0
    return out


def prepare_weights(sd, quant, dtype, fp32_names=()):
    """int8 的權重原樣留著（比例攤成一維，留給 Int8Linear 的 weight_scale），其他浮點數轉 dtype
    （fp32_names 裡的轉 float32）。回傳 (權重, {模組路徑: 分組大小})"""
    import torch
    out, keep = {}, {}
    for k, v in sd.items():
        mod = k[:-len(".weight")] if k.endswith(".weight") else None
        if v.dtype == torch.int8 and mod is not None:
            s = sd.get(mod + ".weight_scale")
            if s is None:
                raise RuntimeError(i18n.tr("eng.q.noScale", k=k))
            out[k], out[mod + ".weight_scale"] = v, s.to(torch.float32).reshape(-1).expand(v.shape[0]).contiguous()
            keep[mod] = quant.get(mod, 0)
        elif k.endswith(".weight_scale"):
            continue
        elif v.is_floating_point():
            out[k] = v.to(torch.float32 if any(n in k.split(".") for n in fp32_names) else dtype)
        else:
            out[k] = v
    return out, keep


def swap_int8(model, keep):
    """把 int8 的線性層、嵌入層換成 Int8Linear／Int8Embedding，補上比例的緩衝區（模型還在 meta 上）"""
    import torch
    lin, emb = int8_classes()
    for name, group in keep.items():
        m = model.get_submodule(name)
        if isinstance(m, torch.nn.Linear):
            m.__class__ = lin
        elif isinstance(m, torch.nn.Embedding):
            m.__class__ = emb
        else:
            raise RuntimeError(i18n.tr("eng.q.notLinear", name=name))
        m.register_buffer("weight_scale", torch.empty(m.weight.shape[0], device="meta"))
        m._coco_group = group
    model.requires_grad_(False)


def _build(cls, cfg, sd, quant, dtype, label, log):
    try:
        from core.anima import _build as build
    except ImportError:
        from anima import _build as build
    weights, keep = prepare_weights(sd, quant, dtype, getattr(cls, "_keep_in_fp32_modules", None) or ())
    model = build(cls, cfg, weights, label, log, prepare=lambda m: swap_int8(m, keep))
    if keep:
        rot = sum(1 for g in keep.values() if g)
        log(label + "　" + i18n.tr("eng.q.int8Store", n=len(keep)) + (i18n.tr("eng.q.rot", n=rot) if rot else ""))
    return model, bool(keep)


# ---------------------------------------------------------------- LoRA

def lora_alias(transformer):
    """LoRA 的模組名 → diffusers 的模組路徑。ComfyUI 的模型把 MLP 的 gate 與 up 合成一個 gate_up（gate 在前），
    照那個訓練的 LoRA 也是一個：對到拆開的兩個（tuple），融合時把變化量照順序切開（pipeline._fuse_lora）。
    其他的名字跟 diffusers 一樣，本來就查得到"""
    out = {}
    for i in range(len(getattr(transformer, "transformer_blocks", []))):
        mlp = f"transformer_blocks.{i}.img_mlp"
        out[f"transformer_blocks_{i}_img_mlp_gate_up"] = (f"{mlp}.gate_layer", f"{mlp}.proj")
    return out


# ---------------------------------------------------------------- 鍵名

def convert(raw):
    """主模型：去掉前綴；ComfyUI 把 MLP 的 gate 與 up 合成一個 gate_up（gate 在前），拆回 diffusers 的
    gate_layer／proj（int8 的每列比例也一起切）。comfy_quant 另外讀（quant_info）"""
    out = {}
    for k, v in raw.items():
        k = k[len(PREFIX):] if k.startswith(PREFIX) else k
        if k.endswith(".comfy_quant"):
            continue
        if ".img_mlp.gate_up." in k:
            gate, up = v.chunk(2, dim=0)
            out[k.replace("gate_up", "gate_layer")], out[k.replace("gate_up", "proj")] = gate, up
        else:
            out[k] = v
    return out


def _gate_up_quant(quant):
    q = {}
    for k, g in quant.items():
        if k.endswith(".img_mlp.gate_up"):
            q[k.replace("gate_up", "gate_layer")] = q[k.replace("gate_up", "proj")] = g
        else:
            q[k] = g
    return q


def _transformer_config(sd):
    head = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    inner, in_ch = sd["img_in.weight"].shape
    return {"patch_size": 1, "in_channels": in_ch, "out_channels": sd["proj_out.weight"].shape[0],
            "num_layers": 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")),
            "attention_head_dim": head, "num_attention_heads": inner // head,
            "context_in_dim": sd["txt_in.in_layer.weight"].shape[1],
            "mlp_ratio": sd["transformer_blocks.0.img_mlp.proj.weight"].shape[0] // inner}


def convert_vae(raw):
    """VAE：ComfyUI（Wan 2.2 式）的命名 → diffusers 的 AutoencoderKLQwenImage21。卷積核的時間維（1）拿掉。
    downsamples.I.downsamples.J／upsamples.I.upsamples.J：最後一個（resample、time_conv）是縮放層，其他是 resnet"""
    res = {"residual.0.gamma": "norm1.gamma", "residual.2": "conv1", "residual.3.gamma": "norm2.gamma",
           "residual.6": "conv2", "shortcut": "conv_shortcut"}
    top = {"conv1": "quant_conv", "conv2": "post_quant_conv"}
    out = {}

    def resnet(rest):
        for a, b in res.items():
            if rest == a or rest.startswith(a + "."):
                return b + rest[len(a):]
        return None

    for k, v in raw.items():
        p = k.split(".")
        new = None
        if p[0] in top:
            new = ".".join([top[p[0]]] + p[1:])
        elif p[0] in ("encoder", "decoder"):
            side, part = p[0], p[1]
            if part == "conv1":
                new = f"{side}.conv_in." + ".".join(p[2:])
            elif part == "head":
                new = f"{side}.norm_out.gamma" if p[2] == "0" else f"{side}.conv_out." + ".".join(p[3:])
            elif part == "middle":
                i, rest = p[2], ".".join(p[3:])
                new = f"{side}.mid_block.attentions.0.{rest}" if i == "1" else \
                    f"{side}.mid_block.resnets.{0 if i == '0' else 1}.{resnet(rest)}"
            elif part in ("downsamples", "upsamples"):
                blocks, sampler = ("down_blocks", "downsampler") if part == "downsamples" else ("up_blocks", "upsampler")
                i, j, rest = p[2], p[4], ".".join(p[5:])
                if p[5] in ("resample", "time_conv"):
                    new = f"{side}.{blocks}.{i}.{sampler}.{rest}"
                else:
                    r = resnet(rest)
                    new = r and f"{side}.{blocks}.{i}.resnets.{j}.{r}"
        if new is None or "None" in new:
            raise RuntimeError(i18n.tr("eng.q.vaeKey", k=k))
        out[new] = v.squeeze(2) if v.dim() == 5 else v
    return out


# ---------------------------------------------------------------- 載入

_PIPE = None


def _pipe_class():
    """官方管線的 __init__ 要 Qwen3VLProcessor（看圖用）；文生圖只要 tokenizer，這裡自己設好那幾個屬性。
    跟 Z-Image 一樣：有顯示卡就一律在顯示卡上算"""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        try:
            from core.vendor.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
        except ImportError:
            from vendor.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

        class CocoQwen21Pipeline(QwenImage21Pipeline):
            def __init__(self, scheduler, vae, text_encoder, tokenizer, transformer):
                DiffusionPipeline.__init__(self)
                self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer,
                                      scheduler=scheduler)
                self.vae_scale_factor = 16
                self.latent_channels = vae.config.z_dim
                self.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=self.latent_channels)
                self.sys_prompt = SYS_PROMPT
                self.prompt_template_t2i = TEMPLATE
                self._drop_idx = len(tokenizer("<|im_start|>system\n" + SYS_PROMPT + "<|im_end|>\n").input_ids)
                self._img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
                """修圖（有條件圖）時官方管線自己編碼提示詞，這裡換成 CoCo 的元件（edit_embeds）"""
                return edit_embeds(self, prompt, image)

        _PIPE = CocoQwen21Pipeline
    return _PIPE


def _vlm(pipe, log=print):
    """修圖用的 Qwen3-VL（看圖＋文字）。第一次修圖時才組看圖的那半（約 1 GB，bf16，直接用映射的檔案），
    文字的那半共用 pipe.text_encoder（不重複載入）"""
    if getattr(pipe, "_coco_vlm", None) is not None:
        return pipe._coco_vlm
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLModel
    src = getattr(pipe, "_coco_te_visual", None)
    if src is None:
        src, pipe._coco_te_map = _z()._mapped(getattr(pipe, "_coco_te_path", ""))
    vis = {}
    for k, v in src.items():
        for pre in ("model.visual.", "visual."):
            if k.startswith(pre):
                vis[k[len(pre):]] = v.to(torch.bfloat16)
    if not vis:
        raise RuntimeError(i18n.tr("eng.q.noVisionEdit"))
    cfg = Qwen3VLConfig(text_config=pipe.text_encoder.config.to_dict(), vision_config=VISION, **VISION_TOKENS)
    with init_empty_weights():
        vlm = Qwen3VLModel(cfg)
    vlm.language_model = pipe.text_encoder
    miss = vlm.visual.load_state_dict(vis, strict=False, assign=True)
    if miss.missing_keys or miss.unexpected_keys:
        raise RuntimeError(i18n.tr("eng.q.visionKeys", miss=miss.missing_keys[:3], extra=miss.unexpected_keys[:3]))
    vlm.requires_grad_(False).eval()
    pipe._coco_vlm = vlm
    log(i18n.tr("eng.q.visionLoad", n=len(vis)))
    return vlm


def _processor(pipe):
    """官方的 Qwen3VLProcessor：模板裡的 <|image_pad|> 依圖的大小展開、圖切成 16 像素的格子。
    影片處理器用不到，但它一定要有一個"""
    if getattr(pipe, "_coco_proc", None) is None:
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        pipe._coco_proc = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(**IMAGE_PROC), tokenizer=pipe.tokenizer,
                                           video_processor=Qwen3VLVideoProcessor())
    return pipe._coco_proc


def edit_embeds(pipe, prompt, images):
    """修圖的提示詞編碼（官方 QwenImage21Pipeline._get_qwen_prompt_embeds 的做法，Apache 2.0）：模板裡放條件圖的位置，
    文字與圖一起進 Qwen3-VL，取最後的 norm 之前；去掉系統提示的那幾個 token；記下哪些位置是圖。
    同一組（提示詞、條件圖）不重算。文字編碼器搬上顯示卡、主模型放不下時先讓開（跟 encode 一樣）"""
    import torch
    from PIL import Image as PILImage
    prompt = [prompt] if isinstance(prompt, str) else list(prompt)
    prompt = [" " if not p else p for p in prompt]
    key = (tuple(prompt), getattr(pipe, "_coco_edit_key", None))
    cache = pipe.__dict__.setdefault("_coco_edit_cache", {})
    if key in cache:
        return cache[key]
    slots = "<image1><|vision_start|><|image_pad|><|vision_end|>" + "".join(
        f" <image{i}><|vision_start|><|image_pad|><|vision_end|>" for i in range(2, len(images) + 1))
    texts = [TEMPLATE.replace("<|im_start|>user\n", "<|im_start|>user\n" + slots, 1).format(p) for p in prompt]
    pics = []
    for _ in prompt:
        for img in images:
            if img.mode == "RGBA":                 # 官方：看圖那邊把透明的地方鋪白底（VAE 那邊照樣讀四個通道）
                white = PILImage.new("RGB", img.size, (255, 255, 255))
                white.paste(img, mask=img.getchannel("A"))
                img = white
            pics.append(img)
    Z = _z()
    vlm = _vlm(pipe)
    te, tr = pipe.text_encoder, pipe.transformer
    moved_tr = False
    t = time.perf_counter()
    if torch.cuda.is_available():
        if Z._device(tr) == "cuda" and not getattr(tr, "_coco_grouped", False):
            torch.cuda.empty_cache()                 # PyTorch 留著沒在用的不算進剩多少，先還回去
            free, _ = torch.cuda.mem_get_info()
            if free < _te_room(pipe) + Z._bytes(vlm.visual) + (2 * 2 ** 30):
                Z._to(pipe, "transformer", "cpu")
                moved_tr = True
        _te_to(pipe, "cuda")
        _z()._up(vlm.visual, getattr(pipe, "_coco_te_map", None), getattr(pipe, "_coco_te_path", None))
    try:
        dev = _te_device(te)
        inputs = _processor(pipe)(text=texts, images=pics, padding=True, padding_side="left", return_tensors="pt").to(dev)
        kw = {k: inputs[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
              if k in inputs}
        kw["pixel_values"] = kw["pixel_values"].to(torch.bfloat16)
        handle = te.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            with torch.no_grad():
                hidden = vlm(**kw).last_hidden_state
        finally:
            handle.remove()
    finally:
        _z()._home(vlm.visual)
        _te_to(pipe, "cpu")
        _z()._trim(pipe)
        if moved_tr:
            Z._to(pipe, "transformer", "cuda")
    mask = inputs["attention_mask"].bool()
    drop = pipe._drop_idx
    split = [h[m][drop:] for h, m in zip(hidden, mask)]
    pads = [(ids[m] == pipe._img_token_id)[drop:] for ids, m in zip(inputs["input_ids"], mask)]
    n = max(e.size(0) for e in split)
    embeds = torch.stack([torch.cat([u, u.new_zeros(n - u.size(0), u.size(1))]) for u in split])
    att = torch.stack([torch.cat([torch.ones(u.size(0), dtype=torch.long, device=u.device),
                                  torch.zeros(n - u.size(0), dtype=torch.long, device=u.device)]) for u in split])
    pad = torch.stack([torch.cat([u, u.new_zeros(n - u.size(0))]) for u in pads])
    out = (embeds, att, pad)
    if len(cache) >= 4:
        cache.pop(next(iter(cache)))
    cache[key] = out
    print_log = getattr(pipe, "_coco_log", None)
    if print_log:
        print_log(i18n.tr("eng.q.editEncode", s=f"{time.perf_counter() - t:.1f}", n=int(pad.sum()))
                  + (i18n.tr("eng.trMoved") if moved_tr else ""))
    return out


def _te_weights(raw):
    """Qwen3-VL 只取文字的部分（model.layers…；Hugging Face 的是 model.language_model.layers…），
    看圖的與輸出層（lm_head）用不到。回傳 (權重, {模組: 分組大小}, 有沒有看圖的部分)"""
    out, visual = {}, False

    def strip(k):
        for pre in ("model.language_model.", "model."):
            if k.startswith(pre):
                return k[len(pre):]
        return None

    for k, v in raw.items():
        if k.startswith(("model.visual.", "visual.")):
            visual = True
            continue
        s = strip(k)
        if s is not None and not k.endswith(".comfy_quant"):
            out[s] = v
    quant = quant_info({k: v for k, v in raw.items() if strip(k) is not None and not k.startswith("model.visual.")},
                       lambda k: strip(k))
    return out, quant, visual


def set_fast(pipe, on, log=print):
    """主模型的 int8 層要不要直接用 int8 算（W8A8）。載入時一律開（finish）；不用重新載入，下一張生效。
    回傳實際有沒有開"""
    lin, _ = int8_classes()
    layers = [m for m in pipe.transformer.modules() if isinstance(m, lin)]
    if not layers:
        return False
    fast = bool(on) and kitchen() is not None
    if bool(on) and not fast:
        log("[!] " + i18n.tr("eng.q.kitchenOff", why=_KITCHEN.get('why', '')))
    if getattr(pipe, "_coco_int8_fast", None) != fast:
        for m in layers:
            m._coco_fast = fast
        pipe._coco_int8_fast = fast
        log(i18n.tr("eng.q.mainMode", mode=i18n.tr("eng.q.fast") if fast else i18n.tr("eng.q.slow")))
    pipe._coco_weights = "int8 W8A8" if fast else "int8"
    return fast


def load(model_path, vae_path, te_path, log=print, dtype=None, fp8=False):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer, Qwen3VLTextModel
    try:
        from core.krea2 import _qwen3vl_config, _check_fp4
        from core.vendor.qwenimage21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        from core.vendor.qwenimage21.transformer_qwenimage21 import QwenImage21Transformer2DModel
        from core.anima import _build as build, lean_vae
    except ImportError:
        from krea2 import _qwen3vl_config, _check_fp4
        from vendor.qwenimage21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        from vendor.qwenimage21.transformer_qwenimage21 import QwenImage21Transformer2DModel
        from anima import _build as build, lean_vae

    dtype = dtype or torch.bfloat16
    if not vae_path or not te_path:
        raise RuntimeError(i18n.tr("eng.needParts", model="Qwen Image 2.1", vae=i18n.tr("eng.vaeOf", name="Qwen Image 2.1", file="qwen_image_2.1_vae"), te=i18n.tr("srv.part.hint", name="Qwen3-VL 8B", file="qwen3vl_8b")))
    for p, what in ((vae_path, "VAE"), (te_path, i18n.tr("eng.teCap"))):
        if not Path(p).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=p))

    log(i18n.tr("eng.loading", model="Qwen Image 2.1"))
    raw = load_file(str(model_path))
    _check_fp4(raw, i18n.tr("eng.lbl.model"))
    quant = _gate_up_quant(quant_info(raw, lambda k: k[len(PREFIX):] if k.startswith(PREFIX) else k))
    sd = convert(raw)
    del raw
    tcfg = _transformer_config(sd)
    log("　" + i18n.tr("eng.tf3", layers=tcfg['num_layers'], heads=tcfg['num_attention_heads'], dim=tcfg['num_attention_heads'] * tcfg['attention_head_dim']))
    transformer, quantized = _build(QwenImage21Transformer2DModel, tcfg, sd, quant, dtype, "　transformer", log)
    del sd
    if fp8 and not quantized:
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        quantized = True
        log("　" + i18n.tr("eng.fp8Store"))

    vsd = load_file(str(vae_path))
    if "encoder.conv1.weight" not in vsd or vsd["encoder.conv1.weight"].shape[1] != 4:
        raise RuntimeError(i18n.tr("eng.q.notVae", name=Path(vae_path).name))
    vae = build(AutoencoderKLQwenImage21, {}, {k: v.to(dtype) for k, v in convert_vae(vsd).items()}, "　vae", log)
    lean_vae(vae)
    del vsd

    te_raw, te_map = _z()._mapped(te_path)
    _check_fp4(te_raw, i18n.tr("eng.lbl.te"))
    te_sd, te_quant, visual = _te_weights(te_raw)
    te_visual = {k: v for k, v in te_raw.items() if k.startswith(("model.visual.", "visual."))}
    del te_raw
    if "embed_tokens.weight" not in te_sd:
        raise RuntimeError(i18n.tr("eng.notQwen3vl", name=Path(te_path).name))
    if te_sd["embed_tokens.weight"].shape[1] != tcfg["context_in_dim"]:
        raise RuntimeError(i18n.tr("eng.teDim", got=te_sd['embed_tokens.weight'].shape[1], want=tcfg['context_in_dim'], model="Qwen Image 2.1", te="Qwen3-VL 8B"))
    if not visual:
        log("[!] " + i18n.tr("eng.noVision", model="Qwen Image 2.1", te="Qwen3-VL 8B", hint=i18n.tr("eng.q.noVisionHint")))
    te_cfg = _qwen3vl_config(te_sd)                  # Qwen3-VL 的 RoPE 設定 4B、8B 一樣（QWEN3VL 是官方 8B 的值，對照用）
    assert te_cfg.rope_parameters["rope_theta"] == QWEN3VL["rope_parameters"]["rope_theta"]
    text_encoder, _ = _build(Qwen3VLTextModel, te_cfg, te_sd, te_quant, dtype, "　text_encoder", log)
    text_encoder.eval()
    del te_sd

    tokenizer = Qwen2Tokenizer.from_pretrained(str(TOKENIZER))
    # 排程：σ 由 CoCo 照官方算好再交給取樣器（見 sigmas），取樣器自己的 shift 一律是 1
    pipe = _pipe_class()(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0,
                                                                    use_dynamic_shifting=False),
                          vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.update_components = lambda **kw: pipe.register_modules(**kw)     # 取樣器沿用 Anima 那套（見 zimage.load）
    finish(pipe, quantized, fp8, log)
    pipe._coco_te_path = str(te_path)
    pipe._coco_te_map, pipe._coco_te_visual = te_map, te_visual
    text_encoder._coco_mapped, text_encoder._coco_file = True, (te_map, str(te_path))
    log("　" + i18n.tr("eng.assembled"))
    return pipe


def finish(pipe, quantized, fp8, log=print):
    """組裝好之後的狀態。int8 的主模型一律直接用 int8 算（沒有 comfy_kitchen 時 set_fast 會退回、說清楚）"""
    pipe._coco_is_qwen21 = True
    pipe._coco_fp8 = False                 # int8 不算 fp8；Model dtype 另外記
    pipe._coco_weights = "int8" if quantized and not fp8 else ("fp8" if quantized else "")
    pipe._coco_embeds = None
    pipe._coco_q21_shift = SHIFT
    pipe._coco_vlm = None                  # 修圖用的看圖部分，第一次修圖時才載入（_vlm）
    pipe._coco_kv, pipe._coco_parked_blocks = [], []
    tr = getattr(pipe, "transformer", None)
    if isinstance(tr, __import__("torch").nn.Module):
        tr.register_forward_pre_hook(lambda m, args, kwargs: _remember_kv(pipe, kwargs), with_kwargs=True)
    set_fast(pipe, True, log)


# ---------------------------------------------------------------- 記憶體（跟 Z-Image 同一套）

def _z():
    try:
        from core import zimage
    except ImportError:
        import zimage
    return zimage


def _te_device(te):
    """文字編碼器在哪裡算：邊算邊搬（group offload）時權重平常在系統記憶體，但一律在顯示卡上算"""
    import torch
    if getattr(te, "_coco_grouped", False) and torch.cuda.is_available():
        return torch.device("cuda")
    return next(te.parameters()).device


def parked(vram_mode):
    return ("text_encoder",)


def place(pipe, vram_mode, log=print):
    _z().place(pipe, vram_mode, log, label="Qwen Image 2.1", extra=getattr(pipe, "_coco_room", 0))
    stream_text_encoder(pipe, log)


# 修圖的其他參考圖（圖2、圖3）：每多一張，去噪時多用約 2.25 GiB（官方管線把條件圖與文字的 K、V 留著每一步用，加上序列變長）。
# 2026-10-09 RTX 5080、896×1152、40 步、整段可以延伸的配置區段：主模型整個放時條件圖 1／2 張 reserved 11.85／14.04 GiB；
# 3 張 allocated 15.77 GiB 擠出顯示卡，有一次 38 秒、有一次 522 秒。邊算邊搬（從載入就是）1／2／3 張 5.9／7.8／10.5 GiB、28／29／33 秒。
# 試過生成途中才改成邊算邊搬（途中鎖定主模型的權重），接下來的修圖會出錯（文字編碼器讀到壞掉的資料），所以一律在載入時決定
EDIT_BASE_BYTES = int(4.3 * 2 ** 30)       # 一張條件圖時去噪要的（RESIDENT_MARGIN 的 4 GiB 差不多就是這個）
EDIT_REF_BYTES = int(2.25 * 2 ** 30)


def refs_room(n_refs):
    """載入時主模型整個放之外，還要另外留多少（給 zimage.place 的 extra）"""
    return max(0, int(n_refs)) * EDIT_REF_BYTES


def needs_reload(pipe, n_refs):
    """已經載入、主模型整個放在顯示卡，這次帶的參考圖卻放不下：要重新載入（改成邊算邊搬）。
    已經是邊算邊搬就照用（之後只有一張的修圖也照用，不來回重新載入）"""
    import torch
    if not n_refs or not getattr(pipe, "_coco_is_qwen21", False) or not torch.cuda.is_available():
        return False
    tr = pipe.transformer
    if getattr(tr, "_coco_grouped", False) or _z()._device(tr) != "cuda":
        return False
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    return free < EDIT_BASE_BYTES + n_refs * EDIT_REF_BYTES


def stream_text_encoder(pipe, log=print, label="Qwen Image 2.1"):
    """文字編碼器（7 GB）邊算邊搬：算到哪一層才搬哪一層（group offload）。
    以前是整個搬上顯示卡、主模型放不下就先讓開再搬回來：RTX 5080 每換一次提示詞光搬就要約 5 秒，
    計算只要 0.5 秒。出圖逐位元相同。
    權重是映射的檔案（_mapped）時不鎖定：鎖定的頁面會一直算在系統記憶體的「使用中」"""
    import torch
    te = pipe.text_encoder
    if not torch.cuda.is_available() or getattr(te, "_coco_grouped", False):
        return
    from diffusers.hooks import apply_group_offloading
    Z = _z()
    Z._to(pipe, "text_encoder", "cpu")
    pinned = 0 if getattr(te, "_coco_mapped", False) else Z.pin(te, log, what=i18n.tr("eng.lbl.te"))
    apply_group_offloading(te, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                           offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=False,
                           low_cpu_mem_usage=True)
    te._coco_grouped = True
    log(i18n.tr("eng.q.teStream", label=label) + (i18n.tr("eng.z.pinned", gb=f"{pinned / 2 ** 30:.1f}") if pinned else
                                               i18n.tr("eng.q.teMapped") if getattr(te, "_coco_mapped", False) else ""))


def _te_to(pipe, device):
    """文字編碼器整個搬（邊算邊搬時不動，搬了會把 group offload 弄亂）"""
    if not getattr(pipe.text_encoder, "_coco_grouped", False):
        _z()._to(pipe, "text_encoder", device)


def _te_room(pipe):
    """編碼時文字編碼器在顯示卡上要的空間：邊算邊搬時只要一層"""
    Z = _z()
    te = pipe.text_encoder
    if getattr(te, "_coco_grouped", False):
        return max((Z._bytes(b) for b in getattr(te, "layers", [])), default=0)
    return Z._bytes(te)


def configure_memory(pipe, vram_mode, log=print):
    _te_to(pipe, "cpu")


# ---------------------------------------------------------------- 排程

def set_scheduler(pipe, sampler, schedule, shift, log=print):
    """取樣器沿用 Anima 那套，取樣器自己的 shift 一律 1（σ 在 sigmas 裡算好）；介面的 shift 另外記"""
    try:
        from core import anima as AN
    except ImportError:
        import anima as AN
    try:
        pipe._coco_q21_shift = float(shift) if shift not in (None, "") else SHIFT
    except (TypeError, ValueError):
        pipe._coco_q21_shift = SHIFT
    AN.set_scheduler(pipe, sampler, schedule, 1.0, lambda m: log(m.replace(" · shift 1.0", "")))
    log(i18n.tr("eng.q.shift", s=f"{pipe._coco_q21_shift:g}"))


def effective_shift(shift, width, height):
    """介面的 shift（1024² 時）→ 這個解析度實際用的。預設值（2.0）就是官方的動態 shift"""
    seq = (int(height) // 16) * (int(width) // 16)
    base = SHIFT_1024 if abs(float(shift) - SHIFT) < 1e-9 else float(shift)
    return base * math.exp(official_mu(seq) - official_mu(SEQ_1024))


def sigmas(steps, shift):
    """官方的 σ：linspace(1, 1/n) → shift → 把終點拉到 0.02（FlowMatchEuler 的 stretch_shift_to_terminal）"""
    import numpy as np
    s = np.linspace(1.0, 1.0 / steps, steps)
    s = shift * s / (1 + (shift - 1) * s)
    one_minus = 1.0 - s
    return 1.0 - one_minus / (one_minus[-1] / (1.0 - TERMINAL))


# ---------------------------------------------------------------- 提示詞

def encode(pipe, prompt, negative, use_cfg, log=print):
    """回傳 (正面, 負面)；同一組提示詞直接用上次的。Qwen3-VL 最後一層的輸出要在最後的 RMSNorm **之前**
    （官方：transformers 5 起 hidden_states[-1] 變成正規化後的，所以掛一個鉤子讓 norm 直接回傳輸入）"""
    import torch
    Z = _z()
    key = (prompt, negative if use_cfg else None)
    if pipe._coco_embeds and pipe._coco_embeds[0] == key:
        return pipe._coco_embeds[1]
    te, tr = pipe.text_encoder, pipe.transformer
    moved_tr = False
    if torch.cuda.is_available():
        if Z._device(tr) == "cuda" and not getattr(tr, "_coco_grouped", False):
            torch.cuda.empty_cache()                 # PyTorch 留著沒在用的不算進剩多少，先還回去
            free, _ = torch.cuda.mem_get_info()
            if free < _te_room(pipe) + (1.5 * 2 ** 30):
                Z._to(pipe, "transformer", "cpu")
                moved_tr = True
        _te_to(pipe, "cuda")
    t = time.perf_counter()

    def one(text):
        dev = _te_device(te)
        ids = pipe.tokenizer(TEMPLATE.format(text or " "), return_tensors="pt").input_ids.to(dev)
        handle = te.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            h = te(input_ids=ids).last_hidden_state
        finally:
            handle.remove()
        return h[:, pipe._drop_idx:]

    try:
        with torch.no_grad():
            pos = one(prompt)
            neg = one(negative) if use_cfg else None
    finally:
        _te_to(pipe, "cpu")
        _z()._trim(pipe)
        if moved_tr:
            Z._to(pipe, "transformer", "cuda")
    log(i18n.tr("eng.encodeTime", model="Qwen Image 2.1", s=f"{time.perf_counter() - t:.1f}") + (i18n.tr("eng.trMoved") if moved_tr else ""))
    pipe._coco_embeds = (key, (pos, neg))
    return pos, neg


# ---------------------------------------------------------------- img2img／局部重繪

def _pack(x):
    """(1, 64, h, w) → 管線吃的 (1, h·w, 64)（2.1 的 latent 不再分塊，只攤平）"""
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).transpose(1, 2)


def _encode_init(pipe, req, width, height):
    """底圖 → 正規化過的 latent（(1, 64, h/16, w/16)，float32）。VAE 吃 RGBA：一般的圖補一個全不透明的 alpha。
    解碼時是 latent × std + mean，這裡反過來；用分佈的眾數不抽樣（同一張底圖每次編出一樣的 latent）"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P
    img = P.load_init_image(req["init_image"], width, height)
    vae = pipe.vae
    w = next(vae.parameters())
    rgb = torch.from_numpy(np.array(img)).to(torch.float32) / 127.5 - 1.0
    x = torch.cat([rgb, torch.ones_like(rgb[..., :1])], dim=-1)          # alpha＝1（全不透明）
    x = x.permute(2, 0, 1)[None, :, None].to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1).to(z)
    std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1).to(z)
    return ((z - mean) / std)[:, :, 0].to(torch.float32)


def _prepare_img2img(pipe, req, generator, width, height, shift, log):
    """跟 Anima、Z-Image、Krea 2 同一套：完整排程照文生圖算（官方的 σ），跳過前面幾步、從切點的 σ 開始；
    起點 x = (1−σ)·x₀ + σ·ε。σ 本來就是 CoCo 算好交給管線的，直接交切掉前面的那段（不用改排程器）。
    局部重繪另外每一步把遮罩外換回底圖的加噪版本（anima.mask_steps）。
    回傳 (打包好的 latents, 這次的 σ, 跑幾步, 還原函式)"""
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
    x0p, epsp = _pack(x0), _pack(eps)
    latents = (1.0 - s0) * x0p + s0 * epsp
    restore = None
    if req.get("_mask") is not None:
        try:
            from core import inpaint as IP
        except ImportError:
            import inpaint as IP
        h, w = x0.shape[-2:]
        m = torch.from_numpy(IP.latent_mask(req["_mask"], w, h))[None, None].to(x0.device).expand_as(x0)
        restore = AN.mask_steps(pipe.scheduler, x0p, epsp, _pack(m.contiguous()))
    strength = min(1.0, max(P.STRENGTH_MIN, float(req.get("strength", 0.5))))
    log(i18n.tr("eng.i2iFrom", model="Qwen Image 2.1", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))
    return latents, full[skip:], run, restore or (lambda: None)


# ---------------------------------------------------------------- 生成

# VAE 解碼每百萬像素實際用到約 3.5 GiB（不留影格快取，見 anima.lean_vae；以前 6.3 GiB），
# 配置器的零碎另外約四成（見 anima._expandable；只留實際用到的量會擠出顯示卡、借系統記憶體）
DECODE_PER_MP = 5.0 * 2 ** 30


def _remember_kv(pipe, kwargs):
    """記下這次生成的 KV 快取（官方管線的區域變數），最後一步算完可以先清掉、騰出空間給解碼"""
    kv = kwargs.get("kv_cache")
    if kv is not None and all(kv is not c for c in pipe._coco_kv):
        pipe._coco_kv.append(kv)


def _room_for_decode(pipe, px, log):
    """主模型整個放在顯示卡、解碼又放不下時，最後一步算完騰出空間（出圖不變）：
      1. 清掉 KV 快取（條件圖與文字的 K、V，最後一步之後用不到；修圖時約 2 GB）
      2. 還不夠就把主模型最後面幾層挪回系統記憶體，挪到剛好夠（下一張開始時搬回來，_restore_blocks）
    以前一律整個主模型挪開再搬回來（約 3.5 秒）；不挪的話擠出顯示卡，解碼慢好幾倍。分塊解碼（大圖自動開）時用不著"""
    import torch
    Z = _z()
    tr = pipe.transformer
    for kv in getattr(pipe, "_coco_kv", []):
        for lc in kv.layer_caches:
            lc.k = lc.v = None
    pipe._coco_kv = []
    if not torch.cuda.is_available() or getattr(tr, "_coco_grouped", False) or Z._device(tr) != "cuda" \
            or getattr(pipe, "_coco_tiled", False):
        return
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    need = DECODE_PER_MP * px / 1e6
    if free >= need:
        return
    parked, freed = [], 0
    for blk in reversed(tr.transformer_blocks):
        if free + freed >= need:
            break
        blk.to("cpu")
        parked.append(blk)
        freed += Z._bytes(blk)
    torch.cuda.empty_cache()
    pipe._coco_parked_blocks = parked
    log(i18n.tr("eng.q.park", n=len(parked), free=f"{free / 2 ** 30:.1f}", need=f"{need / 2 ** 30:.1f}"))


def _restore_blocks(pipe):
    """上一張解碼前挪開的幾層搬回顯示卡"""
    for blk in getattr(pipe, "_coco_parked_blocks", []):
        blk.to("cuda")
    pipe._coco_parked_blocks = []


def _snap(v):
    return max(MULTIPLE * 4, -(-int(v) // MULTIPLE) * MULTIPLE)


def generate(pipe, req, on_step=None, log=print):
    """回傳 (PIL Image, seed)。介面與 pipeline.generate 對齊"""
    import torch
    from PIL import Image
    try:
        from core.anima import _bind_scheduler_generator, _expandable
    except ImportError:
        from anima import _bind_scheduler_generator, _expandable
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

    # CFG 照一般的定義（官方管線的 true_cfg_scale 就是：負 + CFG ×（正 − 負））；官方建議不開（CFG 1）
    cfg = float(req.get("cfg", DEFAULT_CFG))
    use_cfg = cfg > 1.0
    prompt, negative = Z._clean(req.get("prompt", "")), Z._clean(req.get("negative", ""))
    if prompt != (req.get("prompt") or "").strip():
        log(i18n.tr("eng.noWeightSyntax", model="Qwen Image 2.1"))
    pos, neg = (None, None) if req.get("edit_image") else encode(pipe, prompt, negative, use_cfg, log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()                     # 上一張解碼留著的還回去（不然剩下的空間看起來是 0）
    if torch.cuda.is_available() and not getattr(pipe.transformer, "_coco_grouped", False):
        _restore_blocks(pipe)
        Z._to(pipe, "transformer", "cuda")
    Z._to(pipe, "vae", "cuda" if torch.cuda.is_available() else "cpu")
    pipe._coco_kv = []

    width, height = int(req["width"]), int(req["height"])
    w, h = _snap(width), _snap(height)
    if (w, h) != (width, height):
        log(i18n.tr("eng.multipleOf", model="Qwen Image 2.1", m=32, w=w, h=h, tw=width, th=height))
    steps = int(req["steps"])
    shift = effective_shift(getattr(pipe, "_coco_q21_shift", SHIFT), w, h)
    sig = sigmas(steps, shift)
    extra, restore = {}, None
    if req.get("edit_image"):
        # 條件圖：正在修的這張是圖1，其他參考圖接在後面（圖2、圖3），提示詞可以用「圖1」「圖2」指（官方的模板每張前面有 <imageN>）
        files = [Path(req["edit_image"]), *(Path(f) for f in (req.get("edit_refs") or []))]
        conds = []
        for f in files:
            im = Image.open(f)
            im.load()
            conds.append(im)
        pipe._coco_edit_key = tuple((str(f), f.stat().st_mtime) for f in files)
        pipe._coco_log = log
        log(i18n.tr("eng.editFrom", model="Qwen Image 2.1", name=files[0].name, w=conds[0].width, h=conds[0].height)
            + (i18n.tr("eng.editRefs", names=i18n.join(f.name for f in files[1:])) if len(files) > 1 else ""))
        extra = {"image": conds}
    elif req.get("init_image"):
        latents, sig, steps, restore = _prepare_img2img(pipe, req, generator, w, h, shift, log)
        extra = {"latents": latents}

    def cb(p, i, t, kw):
        if on_step:
            on_step(i + 1, steps)
        if i + 1 == steps:
            _room_for_decode(pipe, w * h, log)
        return kw

    # 修圖：整段用可以延伸的配置區段（anima._expandable）。序列一長，一般的區段切不出來就再要新的、舊的留著：
    # 2026-10-09 RTX 5080、高速模式、896×1152：條件圖 2 張 allocated 13.19、reserved 15.78 GiB，擠出顯示卡、35 秒；
    # 整段延伸只留 14.04 GiB、26.5 秒。只有一張時沒差（11.86 → 11.85 GiB）。
    # 試過放不下時改成邊算邊搬（生成途中鎖定主模型的權重），接下來的修圖會出錯（文字編碼器讀到壞掉的資料），不做
    grow = "image" in extra and _expandable(True)
    try:
        if "image" in extra:
            # 修圖：提示詞要跟條件圖一起編碼，交給管線（edit_embeds）；文生圖的編碼結果用不到
            out = pipe(prompt=prompt, negative_prompt=(negative or " ") if use_cfg else None,
                       true_cfg_scale=cfg if use_cfg else 1.0, height=h, width=w, num_inference_steps=steps,
                       sigmas=sig, generator=generator, callback_on_step_end=cb, **extra)
        else:
            out = pipe(prompt_embeds=pos, negative_prompt_embeds=neg if use_cfg else None,
                       true_cfg_scale=cfg if use_cfg else 1.0, height=h, width=w, num_inference_steps=steps,
                       sigmas=sig, generator=generator, callback_on_step_end=cb, **extra)
    finally:
        if grow:
            _expandable(False)
        if restore:
            restore()
        scheduler._coco_generator = None
    img = out.images[0]
    # VAE 輸出 RGBA。一般的圖透明度只在 253～255 跳（解碼的誤差），當成一般的圖；真的有透明的地方才留 RGBA
    if img.mode == "RGBA" and img.getchannel("A").getextrema()[0] >= OPAQUE:
        img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img, seed
