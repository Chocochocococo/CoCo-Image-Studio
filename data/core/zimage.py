# -*- coding: utf-8 -*-
"""
zimage.py —— Z-Image（通義實驗室的 6B 單流 DiT；Base 與 Turbo）

模型檔是 ComfyUI 的拆分式單檔：只有擴散 transformer（cap_embedder、context_refiner、noise_refiner、layers…），
VAE 與文字編碼器要另外選：
  VAE          Flux 的 VAE（ae.safetensors，16 個 latent 通道）
  文字編碼器   Qwen3-4B（qwen_3_4b.safetensors；ComfyUI 的 fp8 版也行，載入時還原成 bf16）
diffusers 0.39 有 ZImagePipeline 與單檔的鍵名轉換；transformer 的設定從權重形狀推，
推不出來的照官方的設定（Tongyi-MAI/Z-Image 的 config.json），不用連網。

**記憶體。** transformer 12.3 GB、文字編碼器 8 GB，16 GB 的卡放不下兩個。文字編碼器不讀進系統記憶體：
權重直接映射檔案（_mapped），只在編碼提示詞的那一下讀檔案搬上顯示卡（_up），編完換回映射的那份（_home）。
以前整份複製在系統記憶體，RTX 5080、1024×1344 生成時系統記憶體多佔 7 GB（峰值 20.2 → 13.3 GB），
編碼連搬帶算 2.7 → 1.4 秒。空間不夠同時放兩個時，先把 transformer 讓開（位置不同、算出來一樣）。
同一組提示詞連續生成（一次多張、固定提示詞換 seed）不重算。
省 VRAM 模式：transformer 一個 block 一個 block 搬（group offload，跟 Anima 同一套）。
"""

import json
import os
import time
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "assets" / "anima" / "tokenizer"        # Qwen3 各尺寸同一份詞彙與對話模板

# 官方 Tongyi-MAI/Z-Image 的設定（Turbo 的 axes_lens 是 1024，只影響最長的位置，權重一樣）
TRANSFORMER = {"all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 16, "dim": 3840, "n_layers": 30,
               "n_refiner_layers": 2, "n_heads": 30, "n_kv_heads": 30, "norm_eps": 1e-5, "qk_norm": True,
               "cap_feat_dim": 2560, "rope_theta": 256.0, "t_scale": 1000.0, "axes_dims": [32, 48, 48],
               "axes_lens": [1536, 512, 512]}
# Flux 的 VAE；force_upcast 關掉：用 bf16 解碼（數值範圍跟 fp32 一樣、不會溢位，跟 SDXL 的做法一致）
VAE = {"act_fn": "silu", "block_out_channels": [128, 256, 512, 512], "in_channels": 3, "out_channels": 3,
       "down_block_types": ["DownEncoderBlock2D"] * 4, "up_block_types": ["UpDecoderBlock2D"] * 4,
       "latent_channels": 16, "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32,
       "sample_size": 1024, "scaling_factor": 0.3611, "shift_factor": 0.1159, "force_upcast": False,
       "use_post_quant_conv": False, "use_quant_conv": False}
# Qwen3 系列共用；形狀推不出來的就是這幾個（RoPE 的 theta 不給會用 transformers 的預設 10000，Qwen3 是一百萬）
QWEN3 = {"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}, "rms_norm_eps": 1e-6,
         "max_position_embeddings": 40960, "tie_word_embeddings": True, "attention_bias": False}
# 官方排程：Base 是 6.0；Turbo 是 3.0（認不出是哪個，shift 可以在生成台改）
SHIFT = 6.0
DEFAULT_CFG, DEFAULT_STEPS = 4.0, 28
MAX_TOKENS = 512                     # 提示詞上限（含對話模板的幾個 token），超過的截掉

PREFIX = "model.diffusion_model."


def is_zimage(keys):
    """ComfyUI 的 Z-Image 單檔（可能帶 model.diffusion_model. 前綴）"""
    ks = set(k[len(PREFIX):] if k.startswith(PREFIX) else k for k in keys)
    return "cap_embedder.1.weight" in ks and any(k.startswith("context_refiner.") for k in ks) \
        and any(k.startswith("noise_refiner.") for k in ks)


def _transformer_config(sd):
    cfg = dict(TRANSFORMER)
    head = sd["layers.0.attention.norm_q.weight"].shape[0]
    dim = sd["cap_embedder.1.weight"].shape[0]
    cfg.update(dim=dim, cap_feat_dim=sd["cap_embedder.1.weight"].shape[1],
               n_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers.")),
               n_refiner_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("noise_refiner.")),
               n_heads=dim // head, n_kv_heads=sd["layers.0.attention.to_k.weight"].shape[0] // head,
               in_channels=sd["all_x_embedder.2-1.weight"].shape[1] // 4)
    return cfg


def _qwen3_config(sd):
    from transformers import Qwen3Config
    emb = sd["embed_tokens.weight"]
    head = sd["layers.0.self_attn.q_norm.weight"].shape[0]
    return Qwen3Config(vocab_size=emb.shape[0], hidden_size=emb.shape[1], head_dim=head,
                       intermediate_size=sd["layers.0.mlp.gate_proj.weight"].shape[0],
                       num_hidden_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers.")),
                       num_attention_heads=sd["layers.0.self_attn.q_proj.weight"].shape[0] // head,
                       num_key_value_heads=sd["layers.0.self_attn.k_proj.weight"].shape[0] // head, **QWEN3)


# ---------------------------------------------------------------- ComfyUI 的 fp8 文字編碼器

_QUANT_EXTRA = ("comfy_quant", "weight_scale", "weight_scale_2", "scale_weight", "input_scale", "scaled_fp8")


def dequant(sd, dtype):
    """ComfyUI 的 fp8 權重還原：weight（float8）× weight_scale。沒壓縮的照原樣轉 dtype。
    nvfp4（4 位元，_fp4_ 版）還不支援，直接說"""
    import torch
    fp8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}
    if any(k.endswith("weight_scale_2") for k in sd):
        raise RuntimeError(i18n.tr("eng.fp4Te"))
    out = {}
    for k, v in sd.items():
        if k.rsplit(".", 1)[-1] in _QUANT_EXTRA:
            continue
        if v.dtype in fp8:
            base = k[:-len("weight")] if k.endswith("weight") else k + "."
            scale = sd.get(base + "weight_scale", sd.get(base + "scale_weight"))
            w = v.to(torch.float32)
            if scale is not None:
                s = scale.to(torch.float32)
                w = w * (s.view(-1, *([1] * (w.dim() - 1))) if s.dim() == 1 and s.shape[0] == w.shape[0] else s)
            out[k] = w.to(dtype)
        else:
            out[k] = v.to(dtype) if v.is_floating_point() else v
    return out


# ---------------------------------------------------------------- 載入

_PIPE = None


def _pipe_class():
    """diffusers 判斷在哪裡算（_execution_device）是從元件裡隨便挑一個看，文字編碼器平常在系統記憶體，
    挑到它就整條判成 CPU：起始的雜訊配在 CPU、跟顯示卡上的 transformer 對不起來。有顯示卡就一律在顯示卡上算"""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import ZImagePipeline

        class CocoZImagePipeline(ZImagePipeline):
            @property
            def _execution_device(self):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _PIPE = CocoZImagePipeline
    return _PIPE


def text_encoder_from(te_path, dtype, dim, build, log=print):
    """Qwen3-4B 文字編碼器。權重直接映射檔案（_mapped），不另外讀進系統記憶體；bf16 的檔案直接用映射的那份，
    fp8 的檔案還原成 dtype 時才複製。編碼時才搬上顯示卡（_to）"""
    from transformers import Qwen3Model
    raw, mm = _mapped(te_path)
    sd = {k[6:]: v for k, v in raw.items() if k.startswith("model.")}
    del raw
    if not sd:
        raise RuntimeError(i18n.tr("eng.notQwen3", name=Path(te_path).name))
    sd = dequant(sd, dtype)
    if sd["embed_tokens.weight"].shape[1] != dim:
        raise RuntimeError(i18n.tr("eng.teDim", got=sd['embed_tokens.weight'].shape[1], want=dim, model="Z-Image", te="Qwen3-4B"))
    te = build(Qwen3Model, _qwen3_config(sd), sd, "　text_encoder", log).eval()
    te._coco_mapped, te._coco_file = True, (mm, str(te_path))
    return te


def load(model_path, vae_path, te_path, log=print, dtype=None, fp8=False):
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint, convert_z_image_transformer_checkpoint_to_diffusers
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer

    try:
        from core.anima import _build
    except ImportError:
        from anima import _build

    dtype = dtype or torch.bfloat16
    if not vae_path or not te_path:
        raise RuntimeError(i18n.tr("eng.needParts", model="Z-Image", vae=i18n.tr("eng.vaeOf", name="Flux", file="ae.safetensors"), te=i18n.tr("srv.part.hint", name="Qwen3-4B", file="qwen_3_4b")))
    for p, what in ((vae_path, "VAE"), (te_path, i18n.tr("eng.teCap"))):
        if not Path(p).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=p))

    log(i18n.tr("eng.loading", model="Z-Image"))
    raw = load_file(str(model_path))
    sd = {k: v.to(dtype) if v.is_floating_point() else v
          for k, v in convert_z_image_transformer_checkpoint_to_diffusers(raw).items()}
    del raw
    tcfg = _transformer_config(sd)
    log("　" + i18n.tr("eng.tf3", layers=tcfg['n_layers'], heads=tcfg['n_heads'], dim=tcfg['dim']))
    transformer = _build(ZImageTransformer2DModel, tcfg, sd, "　transformer", log)
    del sd
    if fp8:
        # 權重用 fp8 存、算的時候一層一層轉回 bf16（diffusers 的 layerwise casting；精度敏感的嵌入層照舊 bf16）
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        log("　" + i18n.tr("eng.fp8Store"))

    vsd = convert_ldm_vae_checkpoint(load_file(str(vae_path)), VAE)
    if "encoder.conv_in.weight" not in vsd or vsd["encoder.conv_in.weight"].shape[0] != 128 \
            or vsd.get("decoder.conv_in.weight", torch.empty(0, 0)).shape[1] != 16:
        raise RuntimeError(i18n.tr("eng.z.notFluxVae", name=Path(vae_path).name))
    vae = _build(AutoencoderKL, VAE, {k: v.to(dtype) for k, v in vsd.items()}, "　vae", log)

    text_encoder = text_encoder_from(te_path, dtype, tcfg["cap_feat_dim"], _build, log)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(TOKENIZER))
    tokenizer.model_max_length = MAX_TOKENS                 # 生成台的 token 計數照這個上限顯示
    pipe = _pipe_class()(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=SHIFT,
                                                                    use_dynamic_shifting=False),
                          vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    # 取樣器沿用 Anima 那套（anima.set_scheduler），它換元件用 modular pipeline 的 update_components
    pipe.update_components = lambda **kw: pipe.register_modules(**kw)
    pipe._coco_is_zimage = True
    pipe._coco_fp8 = bool(fp8)
    pipe._coco_embeds = None
    log("　" + i18n.tr("eng.assembled"))
    return pipe


# ---------------------------------------------------------------- LoRA

# 原始模型（ComfyUI、訓練工具）的名字 → diffusers 改過的名字
_RENAMED = {"x_embedder": "all_x_embedder.2-1", "final_layer_linear": "all_final_layer.2-1.linear",
            "final_layer_adaLN_modulation_1": "all_final_layer.2-1.adaLN_modulation.1"}


def lora_alias(transformer):
    """LoRA 的模組名（kohya 的 lora_unet_layers_0_attention_qkv 去掉前綴，或 diffusion_model.layers.0.… 換成底線）
    → diffusers 的模組路徑。原始模型的 Q、K、V 是一個合起來的 qkv，diffusers 拆成 to_q／to_k／to_v：
    對到三個模組（tuple），融合時把變化量照順序切三段（pipeline._fuse_lora）"""
    names = {n for n, _ in transformer.named_modules()}
    out = {k: v for k, v in _RENAMED.items() if v in names}
    for kind in ("layers", "context_refiner", "noise_refiner"):
        for i in range(len(getattr(transformer, kind, []))):
            att = f"{kind}.{i}.attention"
            out[f"{kind}_{i}_attention_qkv"] = (f"{att}.to_q", f"{att}.to_k", f"{att}.to_v")
            out[f"{kind}_{i}_attention_out"] = f"{att}.to_out.0"
    return out


# ---------------------------------------------------------------- 記憶體

def _device(module):
    try:
        return next(module.parameters()).device.type
    except (AttributeError, StopIteration):
        return ""


# safetensors 的型別 → torch
_DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64", "F16": "float16",
           "BF16": "bfloat16", "F32": "float32", "F64": "float64", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}


def _mapped(path):
    """safetensors 檔直接映射成張量，不複製。回傳 ({鍵: 張量}, mmap)；張量用到哪裡才從檔案（或系統快取）讀哪裡。
    用 copy-on-write 映射：萬一有地方就地改權重（LoRA 融合），改的是這個程式自己的那一頁，檔案不會動"""
    import mmap
    import struct
    import torch
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        name = _DTYPES.get(v["dtype"])
        if name is None:
            raise RuntimeError(i18n.tr("eng.badDtype", name=Path(path).name, k=k, dtype=v['dtype']))
        dt = getattr(torch, name)
        a, b = v["data_offsets"]
        if b == a:
            out[k] = torch.empty(v["shape"], dtype=dt)
        else:
            out[k] = torch.frombuffer(mm, dtype=dt, count=(b - a) // dt.itemsize, offset=8 + n + a).view(v["shape"])
    return out, mm


def _trim(pipe):
    """編碼完把映射的檔案讀過的頁面移出這個程式（Windows 的 VirtualUnlock：範圍沒鎖定時就是移出工作集）。
    內容留在系統的檔案快取，工作管理員不算「使用中」，下次編碼從快取讀回來（8 GB 約 1 秒）。
    其他系統的檔案快取本來就不算使用中，不用做"""
    _release(getattr(pipe, "_coco_te_map", None))


def _release(mm, t=None):
    """映射的檔案讀過的頁面移出這個程式：整塊，或只有張量 t 的那段（t 不在映射的範圍裡就不動：
    其他的是這個程式自己的記憶體，移出去會被寫進分頁檔）"""
    if mm is None or os.name != "nt":
        return
    import ctypes
    import torch
    base = torch.frombuffer(mm, dtype=torch.uint8).data_ptr()
    ptr, n = base, len(mm)
    if t is not None:
        ptr, n = t.data_ptr(), t.numel() * t.element_size()
        if not base <= ptr < base + len(mm):           # 映射裡的張量整個都在裡面
            return
    ctypes.windll.kernel32.VirtualUnlock(ctypes.c_void_p(ptr), ctypes.c_size_t(n))


# 從檔案讀上顯示卡時，一次讀多少進鎖定的緩衝區（兩塊輪流：一塊在搬、一塊在讀）
READ_CHUNK = 64 * 2 ** 20


def _up(module, mm=None, path=None):
    """整個搬上顯示卡，記住原本在系統記憶體的那份（映射的檔案）；_home 換回來，不用再複製一次。
    給了映射的檔案（mm）與它的路徑（path）：映射範圍裡的張量直接讀檔案進鎖定的緩衝區再搬（_read_up），
    不碰映射的頁面。直接搬映射的張量要一頁一頁從系統快取載入：RTX 5080、Z-Image 的 7.5 GB 要 2.8 秒，讀檔案 1.2 秒。
    只給 mm（被 LoRA 改過、檔案裡的是舊的）就直接搬，每搬完一個張量把它讀過的頁面還給系統快取，
    不然搬的那一下整份都會算在系統記憶體的使用中"""
    import itertools
    ts = [t for t in itertools.chain(module.parameters(), module.buffers()) if t.device.type == "cpu"]
    rest = _read_up(ts, mm, path) if mm is not None and path else ts
    for t in rest:
        t._coco_home = t.data
        t.data = t.data.to("cuda", non_blocking=True)
        _release(mm, t._coco_home)


def _read_up(ts, mm, path):
    """ts 裡在映射範圍內的張量從檔案讀上顯示卡，回傳不是的（交給 _up 直接搬）"""
    import torch
    base = torch.frombuffer(mm, dtype=torch.uint8).data_ptr()
    rest, mine = [], []
    for t in ts:
        (mine if t.is_contiguous() and base <= t.data_ptr() < base + len(mm) else rest).append(t)
    bufs = [torch.empty(READ_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    done_ev, k = [None, None], 0
    with open(path, "rb", buffering=0) as fh:
        for t in mine:
            g = torch.empty(t.shape, dtype=t.dtype, device="cuda")
            dst = g.view(-1).view(torch.uint8)
            fh.seek(t.data_ptr() - base)
            size, at = dst.numel(), 0
            while at < size:
                n = min(READ_CHUNK, size - at)
                buf = bufs[k % 2]
                if done_ev[k % 2] is not None:
                    done_ev[k % 2].synchronize()          # 這塊緩衝區上次的搬完了才能再寫
                if fh.readinto(memoryview(buf.numpy())[:n]) != n:
                    raise RuntimeError(i18n.tr("eng.truncated", name=Path(path).name))
                dst[at:at + n].copy_(buf[:n], non_blocking=True)
                done_ev[k % 2] = torch.cuda.Event()
                done_ev[k % 2].record()
                at += n
                k += 1
            t._coco_home = t.data
            t.data = g
    return rest


def _home(module):
    import itertools
    for t in itertools.chain(module.parameters(), module.buffers()):
        home = t.__dict__.pop("_coco_home", None)
        if home is not None:
            t.data = home


def _to(pipe, name, device):
    """整個搬。映射檔案的文字編碼器（_coco_mapped）上顯示卡記住原本那份、回系統記憶體換回來（_up／_home），
    不另外複製一份，換回來之後把讀過的頁面還給系統快取（_release；檔案記在 _coco_file）。
    有 LoRA 併進文字編碼器（pipe._coco_lora_dirty 裡有它的模組）時檔案裡的是舊的，改成直接搬映射的那份。
    LoRA 只在文字編碼器待在系統記憶體時併（編碼完一定搬回來），在顯示卡上改的 _home 會丟掉"""
    import torch
    m = getattr(pipe, name, None)
    if m is None or _device(m) == device:
        return False
    if getattr(m, "_coco_mapped", False) and device in ("cuda", "cpu"):
        mm, path = getattr(m, "_coco_file", (None, None))
        if device == "cuda":
            fused = any(model is m for model, _ in getattr(pipe, "_coco_lora_dirty", ()))
            _up(m, mm, None if fused else path)
        else:
            _home(m)
            _release(mm)
    else:
        m.to(device)
    if device == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def parked(vram_mode):
    """載入後先不搬上 GPU 的元件：文字編碼器一律待在系統記憶體（只在編碼時上來）"""
    return ("text_encoder",)


# 鎖定主模型的權重之後，系統記憶體至少還要剩這麼多（文字編碼器、圖庫、其他程式）
PIN_KEEP_FREE = 12 * 2 ** 30
_PINNED = {"bytes": 0}                  # 目前總共鎖了多少（主模型、文字編碼器各鎖一次時要加起來算）


def _unpin(ptrs, n=0):
    _PINNED["bytes"] = max(0, _PINNED["bytes"] - n)
    try:
        import torch
        cr = torch.cuda.cudart()
        for ptr in ptrs:
            cr.cudaHostUnregister(ptr)
    except Exception:
        pass


def pin(module, log=print, what=None):
    """邊算邊搬時把權重原地鎖定在系統記憶體（cudaHostRegister），回傳鎖了幾個位元組。
    diffusers 搬沒鎖定的權重，每次都先複製到一塊鎖定的記憶體再搬；原地鎖定就直接搬，也不多佔記憶體。
    RTX 5080、Krea 2（fp8 12.2 GB）1024² 8 步：2.5 → 1.8 秒／步，出圖逐位元相同。
    鎖定的記憶體系統不能挪用，所以鎖完剩不到 PIN_KEEP_FREE 就不鎖（照舊、慢一點）。
    權重只會就地改（LoRA 融合、還原都是 copy_／+=），位址不變；模型卸載時解除鎖定。
    只鎖一組一組搬的 block（最外層的 ModuleList）：其他的（嵌入層、最後一層…）diffusers 是整個 .to() 搬，
    搬上顯示卡時系統記憶體那份就釋放了，鎖定卻還在，之後配置到同一塊位址的張量搬回來時 CUDA 回報
    「Copy is larger than memobj size」（2026-10-09 FLUX.1：轉換時新配置的張量在這組裡，第一步就出錯）"""
    import weakref
    import torch
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:
        return 0
    blocks = [c for c in module.children() if isinstance(c, (torch.nn.ModuleList, torch.nn.Sequential))]
    params = [p for b in blocks for p in b.parameters() if p.device.type == "cpu" and p.numel() and not p.is_pinned()]
    need = sum(p.numel() * p.element_size() for p in params)
    if total - _PINNED["bytes"] - need < PIN_KEEP_FREE:
        log(i18n.tr("eng.z.noPin", gb=f"{total / 2 ** 30:.0f}", what=what or i18n.tr("eng.lbl.main")))
        return 0
    ptrs, done = [], 0
    try:
        cr = torch.cuda.cudart()
        for p in params:
            n = p.numel() * p.element_size()
            if int(cr.cudaHostRegister(p.data_ptr(), n, 0)) == 0:
                ptrs.append(p.data_ptr())
                done += n
    except Exception as e:                      # 鎖不了就照舊（慢一點），不影響出圖
        log("[!] " + i18n.tr("eng.z.pinFail", what=what or i18n.tr("eng.lbl.main"), e=str(e)[:80]))
    _PINNED["bytes"] += done
    weakref.finalize(module, _unpin, ptrs, done)
    return done


# 主模型整個放顯示卡時，除了權重還要留給計算的空間（1024² 開 CFG 實測約 1.3 GiB，大圖更多，加上配置器的零碎）
RESIDENT_MARGIN = 4 * 2 ** 30


def place(pipe, vram_mode, log=print, label="Z-Image", extra=0):
    """依 VRAM 模式放主模型（pipeline._to_device 呼叫）。三種出圖逐位元相同，只差快慢與佔用：
      speed、balanced  放得下就整個放，放不下就邊算邊搬
      low              一律邊算邊搬
    高速模式以前一律整個放：放不下時擠出顯示卡、借系統記憶體，反而慢好幾倍
    （RTX 5080、Krea 2 1024²：第一張 2.05 秒／步、第二張 17.65 秒／步；邊算邊搬 1.6 秒）。
    邊算邊搬＝diffusers 的 group offload 加 use_stream：下一個 block 在算這一個時就開始搬；
    low_cpu_mem_usage 不先在系統記憶體另外複製一份釘選的（32 GB 的電腦會不夠），而且搬回來不用複製（權重沒變）。
    RTX 5080、1024² 12 步實測：整個放（擠出顯示卡）3.21 秒／步、峰值 17.6 GB；不開 stream 的分組 3.20 秒、6.2 GB；
    邊算邊搬 2.21 秒、6.5 GB。邊算邊搬時權重原地鎖定（見 pin）"""
    import torch
    gpu = torch.cuda.is_available()
    _to(pipe, "text_encoder", "cpu")
    if not gpu:
        return
    _to(pipe, "vae", "cuda")
    tr = pipe.transformer
    stream, why = vram_mode == "low", i18n.tr("eng.z.lowMode")
    if vram_mode in ("balanced", "speed"):
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        need = _bytes(tr) + RESIDENT_MARGIN + extra          # extra：這次另外要的空間（Qwen 修圖的其他參考圖，qwen21.refs_room）
        stream = free < need
        why = i18n.tr("eng.z.noRoom", free=f"{free / 2 ** 30:.1f}", need=f"{need / 2 ** 30:.1f}")
    if stream:
        pinned = pin(tr, log)
        tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                offload_type="block_level", num_blocks_per_group=1, use_stream=True,
                                record_stream=False, low_cpu_mem_usage=True)
        tr._coco_grouped = True
        log(i18n.tr("eng.z.stream", label=label, why=why) + (i18n.tr("eng.z.pinned", gb=f"{pinned / 2 ** 30:.1f}") if pinned else ""))
    else:
        tr.to("cuda")
        log(i18n.tr("eng.z.onGpu", label=label))


def configure_memory(pipe, vram_mode, log=print):
    _to(pipe, "text_encoder", "cpu")


# ---------------------------------------------------------------- 提示詞

def _clean(text):
    """拿掉 (詞:1.2) 這類權重語法（Z-Image 的文字編碼器是語言模型，括號和數字會被當成字）；BREAK 也拿掉"""
    try:
        from core import conditioning
    except ImportError:
        import conditioning
    return "".join(t for t, w in conditioning.parse_attention(text or "") if not (t == "BREAK" and w == -1)).strip()


def encode(pipe, prompt, negative, use_cfg, log=print, max_len=MAX_TOKENS):
    """回傳 (正面, 負面)；同一組提示詞直接用上次的。文字編碼器搬上 GPU 編碼完就搬回系統記憶體"""
    import torch
    key = (prompt, negative if use_cfg else None, max_len)
    if pipe._coco_embeds and pipe._coco_embeds[0] == key:
        return pipe._coco_embeds[1]
    te, tr = pipe.text_encoder, pipe.transformer
    moved_tr = False
    if torch.cuda.is_available():
        if _device(tr) == "cuda" and not getattr(tr, "_coco_grouped", False):
            free, _ = torch.cuda.mem_get_info()
            if free < _bytes(te) + (1.5 * 2 ** 30):
                _to(pipe, "transformer", "cpu")       # 放不下兩個：transformer 先讓開
                moved_tr = True
        _to(pipe, "text_encoder", "cuda")
    t = time.perf_counter()
    try:
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(prompt=prompt, negative_prompt=negative if use_cfg else None,
                                          do_classifier_free_guidance=use_cfg, max_sequence_length=max_len,
                                          device=next(te.parameters()).device)
    finally:
        _to(pipe, "text_encoder", "cpu")
        if moved_tr:
            _to(pipe, "transformer", "cuda")
    log(i18n.tr("eng.encodeTime", model="Z-Image", s=f"{time.perf_counter() - t:.1f}") + (i18n.tr("eng.trMoved") if moved_tr else ""))
    pipe._coco_embeds = (key, (pos, neg))
    return pos, neg


# ---------------------------------------------------------------- img2img／局部重繪

def _encode_init(pipe, req):
    """底圖 → Z-Image 的 latent（(1, 16, h/8, w/8)，float32）：Flux 的 VAE 解碼時是 latent ÷ 縮放 + 偏移，這裡反過來。
    用分佈的眾數不抽樣：同一張底圖每次編出一樣的 latent，差異只來自 seed"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P
    img = P.load_init_image(req["init_image"], req["width"], req["height"])
    vae = pipe.vae
    w = next(vae.parameters())
    x = torch.from_numpy(np.asarray(img)).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None].to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    return ((z - vae.config.shift_factor) * vae.config.scaling_factor).to(torch.float32)


def _prepare_img2img(pipe, req, generator, log):
    """跟 Anima 同一套（anima._prepare_img2img）：完整排程照文生圖算，跳過前面幾步，從切點的 σ 開始；
    起點 x = (1−σ)·x₀ + σ·ε。局部重繪另外每一步把遮罩外換回底圖的加噪版本（anima.mask_steps）。
    回傳 (latents, sigmas, 跑幾步, 還原函式)"""
    import torch
    from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas
    try:
        from core import anima as AN, pipeline as P
    except ImportError:
        import anima as AN
        import pipeline as P

    n_all, run = P.img2img_steps(req["steps"], req.get("strength", 0.5))
    skip = n_all - run
    sigmas = get_default_z_image_sigmas(n_all)              # 跟文生圖同一條
    sched = pipe.scheduler
    sched.set_timesteps(sigmas=sigmas, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    s0 = float(sched.sigmas[skip])

    x0 = _encode_init(pipe, req)
    eps = torch.randn(x0.shape, generator=generator, device=generator.device, dtype=torch.float32).to(x0.device)
    latents = (1.0 - s0) * x0 + s0 * eps
    restores = [AN._once_partway(sched, skip)]
    if req.get("_mask") is not None:
        try:
            from core import inpaint as IP
        except ImportError:
            import inpaint as IP
        h, w = x0.shape[-2:]
        m = torch.from_numpy(IP.latent_mask(req["_mask"], w, h))[None, None].to(x0.device)
        restores.append(AN.mask_steps(sched, x0, eps, m))
    strength = min(1.0, max(P.STRENGTH_MIN, float(req.get("strength", 0.5))))
    log(i18n.tr("eng.i2iFrom", model="Z-Image", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))

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

    seed = int(req.get("seed", -1))
    if seed < 0:
        import random
        seed = random.randrange(1 << 31)
    generator = torch.Generator("cpu" if str(req.get("rng") or "gpu").lower() == "cpu" else "cuda").manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scheduler = pipe.scheduler
    _bind_scheduler_generator(scheduler)            # diffusers 的迴圈不把 generator 傳給 step，SDE 系的雜訊才同 seed
    scheduler._coco_generator = generator

    # CFG 照一般的定義（Forge／ComfyUI：負 + CFG ×（正 − 負））。diffusers 的 Z-Image 寫成 正 + w ×（正 − 負），
    # w＝CFG − 1：官方範例的 guidance_scale=4 就是 CFG 5。CFG 1（含以下）不跑負面，跟 Turbo 的用法一樣
    cfg = float(req.get("cfg", DEFAULT_CFG))
    use_cfg = cfg > 1.0
    prompt, negative = _clean(req.get("prompt", "")), _clean(req.get("negative", ""))
    if prompt != (req.get("prompt") or "").strip():
        log(i18n.tr("eng.noWeightSyntax", model="Z-Image"))
    pos, neg = encode(pipe, prompt, negative, use_cfg, log)

    if torch.cuda.is_available() and not getattr(pipe.transformer, "_coco_grouped", False):
        _to(pipe, "transformer", "cuda")
    _to(pipe, "vae", "cuda" if torch.cuda.is_available() else "cpu")

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
        out = pipe(prompt_embeds=pos, negative_prompt_embeds=neg if use_cfg else None,
                   height=int(req["height"]), width=int(req["width"]), num_inference_steps=int(req["steps"]),
                   guidance_scale=cfg - 1.0 if use_cfg else 0.0, generator=generator, callback_on_step_end=cb, **extra)
    finally:
        if restore:
            restore()
        scheduler._coco_generator = None
    return out.images[0], seed
