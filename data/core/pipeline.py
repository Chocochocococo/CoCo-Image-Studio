# -*- coding: utf-8 -*-
"""
pipeline.py —— 模型載入與生成

Studio 工作行程使用的模型載入、取樣與圖片儲存共用邏輯。
這裡不做任何輸出入，純粹是函式庫。
"""

import json
import re
import time
from pathlib import Path

import i18n

# 取樣器：每一步怎麼去噪
SAMPLERS = {
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "LMS": ("LMSDiscreteScheduler", {}),
    "Heun": ("HeunDiscreteScheduler", {}),
    "DPM2": ("KDPM2DiscreteScheduler", {}),
    "DPM2 a": ("KDPM2AncestralDiscreteScheduler", {}),
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M SDE": ("DPMSolverMultistepScheduler",
                     {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ SDE": ("DPMSolverSinglestepScheduler", {}),
    "DPM++ 3M SDE": ("DPMSolverMultistepScheduler",
                     {"algorithm_type": "sde-dpmsolver++", "solver_order": 3}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "DDIM": ("DDIMScheduler", {}),
    "DEIS": ("DEISMultistepScheduler", {}),
    # diffusers 本來就有、之前沒接上的
    "SA-Solver": ("SASolverScheduler", {}),      # ComfyUI 的 sa_solver
    "LCM": ("LCMScheduler", {}),                 # 蒸餾模型用，步數很少
    "TCD": ("TCDScheduler", {}),                 # 同上
}

# 支援零終端 SNR（rescale_betas_zero_snr）的取樣器：NoobAI V-Pred 這類模型只能用這些
ZSNR_OK = ("Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE", "UniPC", "DDIM", "LCM", "TCD")

# 排程器：那些步驟的雜訊強度怎麼分佈
SCHEDULES = {
    "Normal": {},
    "Karras": {"use_karras_sigmas": True},
    "Exponential": {"use_exponential_sigmas": True},
    "Beta": {"use_beta_sigmas": True},
    "SGM Uniform": {"timestep_spacing": "trailing"},
    # 以下三種 diffusers 也有，之前沒接上。
    # 不是每個取樣器都支援，不支援的會被 apply_scheduler 逐項剔除並記錄。
    "Flow": {"use_flow_sigmas": True},
    "Lu Lambdas": {"use_lu_lambdas": True},      # 只有 DPM++ 2M 系列支援
    "Linspace": {"timestep_spacing": "linspace"},
    "Leading": {"timestep_spacing": "leading"},
}

# diffusers 的 use_beta_sigmas 要 scipy（requirements-lock.txt 有鎖）。
# 舊安裝、離線裝不起來時沒有它：那時選 Beta 要明講並改用 Normal，
# 不能讓排程器建構失敗——SDXL 會靜靜沿用上一個排程器，Anima 則整個生成失敗。
BETA_KEY = "use_beta_sigmas"


def beta_available():
    # anima.set_scheduler 有同一個判斷；兩邊都用 find_spec，測試才能用
    # sys.modules["scipy"] = None 模擬沒裝
    import importlib.util
    return importlib.util.find_spec("scipy") is not None


# 舊設定裡把兩者寫在一起的名稱，載入時自動拆開
LEGACY = {
    "DPM++ 2M Karras": ("DPM++ 2M", "Karras"),
    "DPM++ 2M SDE Karras": ("DPM++ 2M SDE", "Karras"),
    "DPM++ SDE Karras": ("DPM++ SDE", "Karras"),
}


# 只有 flow-match（Anima）才有的取樣器。選在 SDXL 上會明講後略過，
# 對應的實作在 core/samplers.py 與 anima.set_scheduler。
# 新名字與舊名字都要列：已產出的圖中繼資料裡寫的是舊名，
# 拿掉的話那些圖在 SDXL 上會落到比較籠統的「不認得取樣器」訊息。
# 對應的別名表在 anima._ALIASES。
FLOW_ONLY = ("ER-SDE", "ER-SDE 1", "ER-ODE", "ER-SDE ve",
             "Euler a 0.5", "Euler Churn",
             "Euler a CoCo", "Euler a CoCo 0.5",
             "Euler a renoise", "Euler a renoise 0.25", "Euler a renoise 0.5",
             "Euler a renoise late", "Euler a renoise late-half", "Euler a renoise 0.5 late",
             # 舊名字（別名，見 core/anima.py 的 _ALIASES）
             "Euler a RF", "Euler a RF 0.5", "ER-SDE ODE", "Euler a RF eta 0.5", "Euler churn",
             "Euler a eta 0.25", "Euler a eta 0.5",
             "Euler a late-off", "Euler a late-off 0.5",
             "Euler a eta 0.5 late-off",
             "Euler a kdiff", "Euler a kdiff eta 0.5")


def split_sampler(name):
    """把舊的合併名稱拆成 (取樣器, 排程器)"""
    if name in LEGACY:
        return LEGACY[name]
    return name, None


# ---------------------------------------------------------------- 載入
def vpred_info(path):
    """SDXL／SD 1.5 單檔是不是 v-prediction（NoobAI V-Pred 這類）：(v-prediction, 零終端 SNR)。
    照 ComfyUI、Forge 的認法：檔案裡有空的 v_pred 標記就是 v-prediction、ztsnr 標記就是零終端 SNR。
    沒有標記、但檔頭的 metadata 寫 modelspec.prediction_type: v 的也算 v-prediction（零終端 SNR 只認標記）。
    diffusers 載入單檔時不看這兩個標記，照 epsilon 算，出圖整個壞掉"""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys, meta = set(f.keys()), f.metadata() or {}
    except Exception:
        return False, False
    said = str(meta.get("modelspec.prediction_type") or meta.get("modelspec.predict_key") or "").strip().lower()
    return "v_pred" in keys or said in ("v", "v_prediction", "v-prediction"), "ztsnr" in keys


# v-prediction 模型的排程設定：零終端 SNR 的要從最後一個時間步（雜訊最大）開始（trailing），
# 不然第一步的雜訊強度跟模型學的對不上（diffusers 預設的 leading 從第 946 步開始）
def _set_vpred(pipe, zsnr, log):
    extra = {"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"} if zsnr else {}
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, prediction_type="v_prediction", **extra)
    pipe._coco_vpred = (True, bool(zsnr))
    log(i18n.tr("eng.p.vpred") + (i18n.tr("eng.p.zsnr") if zsnr else ""))


# 「Normal」排程照 A1111／Forge：時間步從 999 等距走到 0（k-diffusion 的 get_sigmas，Forge 的 Normal、Automatic 都是這個）。
# diffusers 的 SDXL、SD 1.5 設定檔寫 leading，28 步時從第 946 步（σ 10.72，Forge 是 14.61）開始，少了雜訊最大、
# 決定大致形狀的那一段。2026-10-11 實測 Forge 自己畫的 9 張（Euler，同 seed、同參數）：CoCo 跟 Forge 平均差
# 26.1 → 13.4（另一半差距是 conditioning._pad_pair 的補零）。舊的做法留在排程選單的「Leading」，舊圖要逐位元重現時用。
# 只換 leading：零終端 SNR 的 V-Pred 在 _set_vpred 已經改成 trailing，照舊
def _a1111_spacing(pipe):
    sched = getattr(pipe, "scheduler", None)
    cfg = getattr(sched, "config", None)
    if cfg is None or getattr(cfg, "timestep_spacing", None) != "leading":
        return
    pipe.scheduler = type(sched).from_config(cfg, timestep_spacing="linspace")


def detect_arch(path: Path):
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            # FLUX.2：4B、9B 的鍵名一樣，看文字的維度分（要讀形狀，檔案還開著時判斷）
            if _flux2().is_flux2(keys):
                return _flux2().arch_of(keys, lambda k: f.get_slice(k).get_shape())
            # FLUX.1：dev 系（有 guidance 輸入）、schnell、Kontext（形狀跟 dev 一樣，看檔名）分開；Fill、Canny、Depth 回 other
            if _flux1().is_flux1(keys):
                return _flux1().arch_of(keys, lambda k: f.get_slice(k).get_shape(), Path(path).name)
    except Exception:
        return "sdxl"
    j = "\n".join(keys)
    # Anima 的 ComfyUI 式單檔：只有去噪網路與 LLM 轉接層，沒有 VAE 也沒有文字編碼器。
    # 不先認出來的話，會被當成 sd15 拿去 from_single_file，
    # 然後報一個跟真正原因無關的「缺 CLIPTextModel」。
    if _anima().is_anima(j):
        return "anima"
    # Z-Image 的 ComfyUI 式單檔：一樣只有 transformer（cap_embedder、context_refiner、noise_refiner）
    if _zimage().is_zimage(keys):
        return "zimage"
    # Krea 2 的 ComfyUI 式單檔：Krea 原始命名（blocks、txtfusion、first／last）
    if _krea2().is_krea2(keys):
        return "krea2"
    # Qwen Image 2.1 的 ComfyUI 式單檔：共用的 modulation、txt_in.in_layer
    if _qwen21().is_qwen21(keys):
        return "qwen21"
    if ("conditioner.embedders.1" in j or "text_encoders.clip_g" in j
            or "text_encoder_2." in j or "add_embedding" in j):
        return "sdxl"
    if "double_blocks." in j or "model.diffusion_model.joint_blocks" in j:
        return "other"
    if not any(k.startswith(("first_stage_model.", "model.diffusion_model.",
                             "cond_stage_model.", "vae.", "unet."))
               for k in keys):
        return "partial"     # 認不出來，而且看起來不是完整的檢查點
    return "sd15"


def _anima():
    """工作行程把 core/ 也放進了 sys.path，兩種匯入方式都要能通"""
    try:
        from core import anima
    except ImportError:
        import anima
    return anima


def _zimage():
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


def _qwen21():
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


def _flux1():
    try:
        from core import flux1
    except ImportError:
        import flux1
    return flux1


# FLUX.2 [klein] 的兩個架構代號（core/flux2.py 共用）
FLUX2_ARCHS = ("klein4", "klein9")
# FLUX.1 的架構代號（core/flux1.py 共用）：dev 系、schnell、Kontext
FLUX1_ARCHS = ("flux1", "flux1s", "kontext")


def is_flow(pipe):
    """flow matching 的拆分式架構（Anima、Z-Image、Krea 2、Qwen Image 2.1、FLUX.2 klein、FLUX.1）：取樣器、記憶體、存圖參數走同一套"""
    return any(getattr(pipe, a, False) for a in ("_coco_is_anima", "_coco_is_zimage", "_coco_is_krea2", "_coco_is_qwen21",
                                                 "_coco_is_flux2", "_coco_is_flux1"))


PREFIXES = ("", "model.", "module.", "text_model.", "transformer.",
            "first_stage_model.", "cond_stage_model.", "vae.",
            "text_encoder.", "text_encoders.")


def _load_into(module, path, log, label=""):
    """把單檔 .safetensors 的權重載進既有的模組。

    ComfyUI 式的拆分模型（例如 Anima 的 text_encoders/qwen_3_06b_base）
    就是單一檔案，沒有 config.json，所以不能用 from_pretrained。
    做法是沿用管線裡既有的架構，只把權重換掉。
    鍵名前綴常常不一樣，所以會試幾種常見的前綴。
    """
    import torch
    from safetensors.torch import load_file

    sd = load_file(str(path))
    want = set(module.state_dict().keys())

    best, best_n, best_pre = sd, 0, ""
    for pre in PREFIXES:
        if pre:
            cand = {k[len(pre):]: v for k, v in sd.items()
                    if k.startswith(pre)}
        else:
            cand = sd
        n = len(want & set(cand))
        if n > best_n:
            best, best_n, best_pre = cand, n, pre

    # 只對上零星幾個就照做的話，其餘權重會維持原樣，混出一顆壞掉的模組。
    # 實測用 LDM 命名的 sdxl_vae 只對上 12/248，出圖灰濛濛還有白斑點，
    # 而且沒有任何錯誤訊息。寧可擋下來。
    if best_n < len(want) * 0.9:
        raise RuntimeError(i18n.tr("eng.p.keysPartial", n=best_n, total=len(want), file=len(sd)))

    best = {k: v.to(torch.float16) if v.is_floating_point() else v
            for k, v in best.items()}
    miss = module.load_state_dict(best, strict=False)
    log(label + i18n.tr("eng.p.loadedN", n=best_n, total=len(want))
        + (i18n.tr("eng.p.prefix", p=best_pre) if best_pre else "")
        + (i18n.tr("eng.p.missingN", n=len(miss.missing_keys)) if miss.missing_keys else ""))
    return module


def _load_ldm_vae(pipe, path: Path, log):
    """載入原始 LDM 命名的 VAE（Civitai 上的 sdxl_vae 等等）

    那種檔案的鍵名是 decoder.up.0.block.0.…，diffusers 用的是
    decoder.up_blocks.0.resnets.0.…，直接套只會對上零星幾個。
    from_single_file 會做這層轉換，但它要連 HF 抓設定；
    這裡改用管線裡現成 VAE 的設定，離線也能用。
    """
    import torch
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    sd = load_file(str(path))
    conv = convert_ldm_vae_checkpoint(sd, pipe.vae.config)
    if not conv:
        raise RuntimeError(i18n.tr("eng.p.ldmEmpty"))
    conv = {k: v.to(torch.float16) if v.is_floating_point() else v
            for k, v in conv.items()}
    miss = pipe.vae.load_state_dict(conv, strict=False)
    want = len(pipe.vae.state_dict())
    if len(miss.missing_keys) > want * 0.1:
        raise RuntimeError(i18n.tr("eng.p.ldmMissing", n=len(miss.missing_keys), total=want))
    log(i18n.tr("eng.p.ldmLoaded", name=path.name, n=want - len(miss.missing_keys), total=want))


def _load_component(cls_name, path, log):
    """資料夾走 from_pretrained，單檔走 from_single_file"""
    import torch
    import diffusers
    import transformers

    cls = getattr(diffusers, cls_name, None) or getattr(
        transformers, cls_name, None)
    if cls is None:
        raise RuntimeError(i18n.tr("eng.p.noClass", name=cls_name))

    f = Path(path)
    if f.is_dir():
        return cls.from_pretrained(str(f), torch_dtype=torch.float16)
    if not hasattr(cls, "from_single_file"):
        raise RuntimeError(i18n.tr("eng.p.noSingle", name=cls_name))
    return cls.from_single_file(str(f), torch_dtype=torch.float16)


# ---------------------------------------------------------------- VRAM
# 分階段實測（RTX 5080 / 15.9 GB）：
#   SDXL  1536²　權重 6.46 GB（unet 4.78、兩顆文字編碼器 1.52、vae 0.16）
#                尖峰 12.16 GB → 中間張量 5.58 GB
#   Anima 1536²　權重 5.24 GB　尖峰 14.61 GB → 中間張量 9.31 GB
#
# 所以大頭是中間張量而不是權重，而中間張量的尖峰落在 VAE 解碼。
# 這決定了三種模式的設計：先處理解碼，再處理閒置的權重。
VRAM_MODES = ("speed", "balanced", "low")

# 超過這個像素量就分塊解碼。1024² 尖峰還在安全範圍，1536² 會逼近上限。
# R21 實測（2026-10-03，12 步，峰值 reserved）：SDXL 1536² 平衡模式兩個都開 6.66 GiB、只分段 9.83、
# 只分塊 9.96、都不開 15.06，兩個一起開不衝突、最省，只多 1.3 秒；896×1536 不分塊，分段就降到 6.65。
# Anima 1536² 不分塊 12.29（高速 17.27，超過顯示卡）、分塊 5.38。分塊的出圖差平均 0.7～1.3/255、
# 散在整張圖，沒有接縫。省 VRAM 模式的峰值在解碼：Anima 896×1536 不分塊 7.35、分塊 1.65（說明有教）
TILE_ABOVE = 1280 * 1280

# SDXL 在這個像素量以上，UNet 與 VAE decode 工作集疊在一起很容易逼近
# 16 GB 卡的上限。auto 模式改走 latent → 卸載 UNet → decode。
STAGED_DECODE_ABOVE = 1024 * 1024

# VAE 解碼時 PyTorch 向驅動**新要**的 VRAM，GiB／百萬像素（key：是否升 fp32）。
# 量的是 reserved 的增量，不是 allocated——實際吃掉顯卡的是前者。
# SDXL 1024×1344 與 1536² 實測（先 empty_cache）：
#   升 fp32   allocated 3.7/MP，reserved 5.5～7.4/MP（配置器取整與快取，且不穩定）
#   fp16-fix  allocated 2.3/MP，reserved 3.96～3.98/MP
# 取最大值。原版 VAE 因此在 16 GB 卡上 1024×1344 就放不下，照樣會搬 UNet；
# 換 fp16-fix 才會留著。
DECODE_GIB_PER_MP = {True: 7.4, False: 4.0}
DECODE_MARGIN_GIB = 1.0


def decode_budget(px, upcast, free_bytes):
    """解碼放不放得下，回傳 (放得下, 可用 GiB, 需要 GiB)

    free_bytes 必須是 empty_cache() **之後**驅動回報的剩餘 VRAM。

    **不能只看總容量**：桌面程式本身就會吃掉幾 GB，而 Windows 在 VRAM
    不夠時不會報錯，會把一部分挪到系統記憶體，速度靜靜掉一截。

    **也不能把 PyTorch 已保留但沒在用的快取算進來**：那些是去噪留下的
    碎片，塞不下解碼的大張量。實測 1024×1344 fp32：快取閒置 1.08 GiB、
    驅動剩 6.88 GiB，解碼實際只用 5.08 GiB，卻向驅動新要了 6.94 GiB——
    超出剩餘量。先 empty_cache 把碎片還回去，剩餘量才是真的。

    需要量同理要用 reserved 的增量估（DECODE_GIB_PER_MP），不能用
    allocated：同一次解碼 allocated 5.08、reserved 卻漲 7.62 GiB。
    """
    avail = free_bytes / 2 ** 30
    need = px / 1e6 * DECODE_GIB_PER_MP[bool(upcast)] + DECODE_MARGIN_GIB
    return avail >= need, avail, need


def _tile_vae(pipe, req, log):
    """依解析度決定要不要分塊解碼

    「選 low 才 tiling」是錯的分法——尖峰跟解析度有關，跟模式無關。
    實測 Anima 1536² 開了是 14.61 → 5.75 GB，出圖差異平均 0.4/255。
    小圖不開，因為分塊終究不是完全等價。

    這裡曾經寫成 `px > TILE_ABOVE or vram == "low"`，跟上面那句話自相
    矛盾，也是「省 VRAM 模式跟高速模式出圖不一樣」的唯一原因：分組載入
    本身是數學透明的（實測差異 0.0000），分塊不是。兩邊改用同一條規則
    之後，Anima 896×1536 與 1536² 都逐像素相同。

    真的需要再擠的人用 tile_vae 明確指定，不要綁在 VRAM 模式上。
    """
    px = int(req.get("width", 0)) * int(req.get("height", 0))
    force = str(req.get("tile_vae", "auto") or "auto").lower()
    if force == "on":
        want = True
    elif force == "off":
        want = False
    else:
        want = px > TILE_ABOVE
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    try:
        if want and not getattr(pipe, "_coco_tiled", False):
            vae.enable_tiling()
            pipe._coco_tiled = True
            log(i18n.tr("eng.p.tileOn") if force == "on" else i18n.tr("eng.p.tileAuto", mp=f"{px // 10000 / 100:.2f}"))
        elif not want and getattr(pipe, "_coco_tiled", False):
            vae.disable_tiling()
            pipe._coco_tiled = False
    except Exception as e:
        log("[!] " + i18n.tr("eng.p.tileFail", e=f"{type(e).__name__}: {str(e)[:90]}"))


def _park_encoders(pipe, log):
    """文字編碼器編完就搬回 CPU

    SDXL 有兩顆，加起來 1.52 GB，而它們在幾十步的去噪期間完全沒事做。
    去噪主模型不搬——那個每一步都要用，搬來搬去只會變慢。

    除了搬走還要把屬性設成 None：diffusers 的 pipe.device 是從
    _get_signature_keys 回傳的**集合**裡取「第一個」模組的裝置，
    集合沒有順序，所以只要有元件在 CPU 上，整條管線就可能被判成 cpu，
    接著 prepare_latents 會拿 cpu 裝置配 latents 而 generator 是 cuda，
    直接拋錯。設成 None 就不列入判斷。
    SDXL 在傳入 prompt_embeds 時本來就支援 text_encoder_2=None
    （投影維度會改從 pooled 向量的形狀推得）。
    """
    import torch
    parked = getattr(pipe, "_coco_parked", None)
    if parked is None:
        parked = {}
        pipe._coco_parked = parked
    moved = []
    for name in ("text_encoder", "text_encoder_2"):
        m = getattr(pipe, name, None)
        if m is None or not hasattr(m, "to"):
            continue
        try:
            m.to("cpu")
            parked[name] = m
            setattr(pipe, name, None)
            moved.append(name)
        except Exception as e:
            log("[!] " + i18n.tr("eng.p.toCpuFail", name=name, e=f"{type(e).__name__}: {str(e)[:70]}"))
    if moved:
        torch.cuda.empty_cache()
    return moved


def _wake_encoders(pipe, log=print):
    parked = getattr(pipe, "_coco_parked", None)
    if not parked:
        return
    for name, m in list(parked.items()):
        try:
            setattr(pipe, name, m.to("cuda"))
        except Exception as e:
            setattr(pipe, name, m)
            log("[!] " + i18n.tr("eng.p.toGpuFail", name=name, e=f"{type(e).__name__}: {str(e)[:70]}"))
        parked.pop(name, None)


def _for_arch(value, arch):
    """覆寫是一個架構記一格。舊的單一字串照舊當成通用值。"""
    if isinstance(value, dict):
        return value.get(arch or "", "")
    return value or ""


def _vae_bf16(pipe):
    """會升 fp32 的 VAE（SDXL 原版，config 的 force_upcast）改用 bf16 跑，Forge／ComfyUI 在新顯卡上也是這樣。

    原版 SDXL VAE 在 fp16 會溢位出黑圖，所以 diffusers 解碼、img2img 編碼時整顆升 fp32：
    RTX 5080、1024² 實測 reserved 從 6 衝到 13.5～13.8 GiB（UNet 去噪只要 6），
    加上桌面本身用的就超過 16 GB，Windows 挪到系統記憶體、變慢。
    bf16 的數值範圍跟 fp32 一樣、不會溢位：峰值 8.8 GiB（跟 fp16-fix 一樣），
    一般生成跟 fp32 解碼平均差 0.31/255（99% 的像素差 2 以內）。

    diffusers 只在 VAE 是 fp16 時才升精度，其他時候不會幫忙轉輸入的型別（latent 是 fp16），
    所以編碼、解碼包一層，進去前轉成 bf16。回傳有沒有改"""
    import torch
    vae = getattr(pipe, "vae", None)
    cfg = getattr(vae, "config", None)
    if vae is None or not getattr(cfg, "force_upcast", False) or getattr(vae, "dtype", None) != torch.float16:
        return False
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return False
    except Exception:
        return False
    vae.to(dtype=torch.bfloat16)
    vae.register_to_config(force_upcast=False)
    enc, dec = vae.encode, vae.decode
    vae.encode = lambda x, *a, **k: enc(x.to(torch.bfloat16), *a, **k)
    vae.decode = lambda z, *a, **k: dec(z.to(torch.bfloat16), *a, **k)
    vae._coco_bf16 = True
    return True


def used_parts(model, vae_path, te_path):
    """這次實際用的 VAE 與文字編碼器（跟 load_model 同一套判斷）。存圖時記進 coco JSON，
    圖庫「存成 Recipe」才帶得到。值：路徑、"fp16fix"，或 "builtin"（模型自帶）"""
    p = Path(model)
    arch = detect_arch(p) if p.suffix.lower() in (".safetensors", ".ckpt") and p.exists() else ""
    vae, te = _for_arch(vae_path, arch).strip(), _for_arch(te_path, arch).strip()
    if vae == "fp16fix" and arch not in ("sdxl", ""):
        vae = ""                       # fp16fix 只給 SDXL（load_model 也是這樣略過）
    return {"arch": arch, "vae": vae or "builtin", "te": te or "builtin"}


# Anima 省 VRAM 模式的 transformer 一次載入幾個 block 到 GPU（審查紀錄 R26）。
# RTX 5080 實測（anima_baseV10，1024² 12 步，第二張）：1／2／4／7／14 個一組都是 21 秒上下，
# 峰值 reserved 5.76／5.88／6.23／6.54／7.47 GiB——搬運次數多不會變慢（瓶頸不在搬運），一組越大只是越佔 VRAM，
# 所以用 1。diffusers 的 use_stream（邊算邊搬）快到 6.5 秒，但峰值 9.38 GiB，比平衡模式（6.6 秒、5.65 GiB）還佔，
# 不用。出圖全部逐位元相同。
ANIMA_BLOCKS_PER_GROUP = 1


def _to_device(pipe, vram_mode, log):
    """搬上 GPU。

    modular pipeline 沒有 enable_model_cpu_offload，所以 low 模式改用
    transformer 的 block-level group offload：一次只把幾個 block 載上
    GPU，用完換下一組。比整包常駐省，又比逐層 offload 快。

    Anima 的 balanced／low 待機時有些元件放 CPU（anima.parked），那些一開始就不搬上 GPU（R22、R23）：
    以前先整包搬上去、configure_memory 再搬回來，balanced 載入的瞬間整個模型在 VRAM（實測 5.30 GiB），
    low 的文字元件白跑一趟（1.60 GiB）。

    VAE 分塊解碼不在這裡處理——那個依解析度自動決定，見 _tile_vae。
    """
    import torch

    anima = getattr(pipe, "_coco_is_anima", False)
    zimg = getattr(pipe, "_coco_is_zimage", False)
    if zimg:                       # Z-Image 自己決定主模型放哪（整個放或邊算邊搬，見 zimage.place）
        _zimage().place(pipe, vram_mode, log)
        pipe._coco_vram = vram_mode
        return pipe
    if getattr(pipe, "_coco_is_krea2", False):      # Krea 2 同一套
        _krea2().place(pipe, vram_mode, log)
        pipe._coco_vram = vram_mode
        return pipe
    if getattr(pipe, "_coco_is_qwen21", False):     # Qwen Image 2.1 也是
        _qwen21().place(pipe, vram_mode, log)
        pipe._coco_vram = vram_mode
        return pipe
    if getattr(pipe, "_coco_is_flux2", False):      # FLUX.2 klein 也是
        _flux2().place(pipe, vram_mode, log)
        pipe._coco_vram = vram_mode
        return pipe
    if getattr(pipe, "_coco_is_flux1", False):      # FLUX.1 也是
        _flux1().place(pipe, vram_mode, log)
        pipe._coco_vram = vram_mode
        return pipe
    park = _anima().parked(vram_mode) if anima else _zimage().parked(vram_mode) if zimg else ()

    def to_gpu(names):
        for name in names:
            m = getattr(pipe, name, None)
            if m is not None and hasattr(m, "to") and name not in park:
                try:
                    m.to("cuda")
                except Exception:
                    pass

    if vram_mode == "low":
        # Anima 的記憶體一律由 CoCo 自己管（group offload＋anima 的掛鉤），不用 diffusers 的 cpu offload（R24）：
        # 現在 Anima 的 modular pipeline 沒有這個方法所以走不到，但 diffusers 哪天加上了，這裡就會成功、提早
        # return、不裝掛鉤，anima.generate 卻照樣搬元件，兩套搬移打架
        if not anima and not zimg:
            try:
                pipe.enable_model_cpu_offload()
                pipe._coco_vram = vram_mode
                return pipe
            except Exception:
                pass

        t = getattr(pipe, "transformer", None)
        done = False
        if t is not None and hasattr(t, "enable_group_offload"):
            try:
                t.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=ANIMA_BLOCKS_PER_GROUP)
                done = True
                t._coco_grouped = True
                log(i18n.tr("eng.p.grouped", n=ANIMA_BLOCKS_PER_GROUP))
            except Exception as e:
                log("[!] " + i18n.tr("eng.p.groupFail", e=str(e)[:90]))

        # 其他元件上 GPU（Anima 待機放 CPU 的除外）；transformer 已經交給 group offload 管了
        to_gpu(("text_encoder", "text_conditioner", "vae"))
        if not done:
            pipe.to("cuda")
    elif park:
        # Anima balanced：只搬待機時要留在 GPU 的（VAE），其他的要用時 anima 的掛鉤才搬上來
        to_gpu(("text_encoder", "text_conditioner", "transformer", "vae"))
    else:
        pipe.to("cuda")

    pipe._coco_vram = vram_mode
    if getattr(pipe, "_coco_is_anima", False):
        _anima().configure_memory(pipe, vram_mode, log)
    if zimg:
        _zimage().configure_memory(pipe, vram_mode, log)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def load_model(model, vram_mode="normal", log=print,
               vae_path=None, te_path=None, vae_fp32=False, fp8_weights=False, room=0, clip_path=None):
    """回傳已載入並搬到 GPU 的 pipeline

    支援三種格式：
      單檔 .safetensors        → 讀標頭判斷架構後 from_single_file
      diffusers 資料夾          → from_pretrained（含 model_index.json）
      拆分零件（vae/te 覆寫）    → 各自載入後注入，架在 diffusers 之上
    """
    import torch
    import diffusers
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(i18n.tr("eng.p.noCuda"))

    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    log(i18n.tr("eng.p.gpu", name=torch.cuda.get_device_name(0), sm=f"sm_{cap[0]}{cap[1]}", torch=torch.__version__))
    if f"sm_{cap[0]}{cap[1]}" not in archs:
        raise RuntimeError(i18n.tr("eng.p.smUnsupported", sm=f"sm_{cap[0]}{cap[1]}"))

    p = Path(model)
    t0 = time.time()

    arch = ""
    if p.suffix.lower() in (".safetensors", ".ckpt") and p.exists():
        arch = detect_arch(p)
    vae_path = _for_arch(vae_path, arch)
    te_path = _for_arch(te_path, arch)

    if p.suffix.lower() in (".safetensors", ".ckpt"):
        if not p.exists():
            raise RuntimeError(i18n.tr("eng.p.noModelFile", p=p))
        if arch == "anima":
            # Anima 走自己的組裝路徑：這個檔案只有 transformer 與
            # LLM 轉接層，VAE 與文字編碼器要另外指定。
            pipe = _anima().load(p, vae_path, te_path, log=log, fp8=fp8_weights)
            pipe._coco_fp8 = bool(fp8_weights)    # 圖的參數記 Model dtype: fp8
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch == "zimage":
            # Z-Image 同樣只有 transformer：VAE（Flux 的）與文字編碼器（Qwen3-4B）要另外指定
            pipe = _zimage().load(p, vae_path, te_path, log=log, fp8=fp8_weights)
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch == "krea2":
            # Krea 2 也只有 transformer：VAE（Qwen Image 的）與文字編碼器（Qwen3-VL 4B）要另外指定
            pipe = _krea2().load(p, vae_path, te_path, log=log, fp8=fp8_weights)
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch == "qwen21":
            # Qwen Image 2.1 也只有 transformer：VAE（Qwen Image 2.1 的）與文字編碼器（Qwen3-VL 8B）要另外指定
            pipe = _qwen21().load(p, vae_path, te_path, log=log, fp8=fp8_weights)
            pipe._coco_room = room          # 修圖的其他參考圖要另外留的空間（qwen21.refs_room），放不下就邊算邊搬
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch in FLUX2_ARCHS:
            # FLUX.2 klein 也只有 transformer：VAE（FLUX.2 的）與文字編碼器（4B：Qwen3-4B、9B：Qwen3-8B）要另外指定
            pipe = _flux2().load(p, vae_path, te_path, log=log, fp8=fp8_weights)
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch in FLUX1_ARCHS:
            # FLUX.1：只有主模型的檔，或全包的檔（自帶的零件沒另外選就用檔案裡的）；文字編碼器欄是 T5，CLIP-L 另外帶
            pipe = _flux1().load(p, vae_path, te_path, clip_path or None, log=log, fp8=fp8_weights)
            _to_device(pipe, vram_mode, log)
            log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
            return pipe
        if arch == "partial":
            raise RuntimeError(i18n.tr("eng.p.partial", name=p.name))
        order = {"sdxl": ["StableDiffusionXLPipeline", "StableDiffusionPipeline"],
                 "sd15": ["StableDiffusionPipeline", "StableDiffusionXLPipeline"],
                 "other": ["StableDiffusionXLPipeline",
                           "StableDiffusionPipeline"]}[arch]
        log(i18n.tr("eng.p.single", arch=arch, name=p.name))
        errs, pipe = [], None
        for name in order:
            cls = getattr(diffusers, name, None)
            if cls is None or not hasattr(cls, "from_single_file"):
                continue
            try:
                pipe = cls.from_single_file(str(p), torch_dtype=torch.float16,
                                            use_safetensors=True)
                log(i18n.tr("eng.p.pipeType", name=name))
                break
            except Exception as e:
                errs.append(f"{name}: {str(e)[:150]}")
        if pipe is None:
            raise RuntimeError(i18n.tr("eng.p.singleFail") + "\n" + "\n".join(errs))
        v, zsnr = vpred_info(p)
        if v:
            _set_vpred(pipe, zsnr, log)
    else:
        kind = i18n.tr("eng.p.kindDir") if p.is_dir() else i18n.tr("eng.p.kindHf")
        log(i18n.tr("eng.p.diffusers", kind=kind, model=model))
        pipe = DiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16)
        log(i18n.tr("eng.p.pipeType", name=type(pipe).__name__))
    _a1111_spacing(pipe)

    # VAE：""＝模型自帶、"fp16fix"＝SDXL 修正版、其餘視為路徑
    want = (vae_path or "").strip()
    if want == "fp16fix" and getattr(pipe, "text_encoder_2", None) is None:
        want = ""                     # 不是 SDXL 就沒必要換
    if want == "fp16fix":
        try:
            from diffusers import AutoencoderKL
            # 先只讀本機快取。下載過一次之後就不必連網；而且同一個行程
            # 載第二個模型時，hub 的連線可能已經被關掉，連網那條路會拋
            # 「client has been closed」，然後靜靜退回模型自帶的 VAE。
            try:
                pipe.vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                    local_files_only=True)
            except Exception:
                pipe.vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            log("VAE：sdxl-vae-fp16-fix")
        except Exception as e:
            log("[!] " + i18n.tr("eng.p.fp16VaeFail", e=str(e)[:90]))
    elif want:
        try:
            pipe.vae = _load_component("AutoencoderKL", want, log)
            log(f"VAE：{Path(want).name}")
        except Exception as e:
            try:
                _load_ldm_vae(pipe, Path(want), log)
            except Exception as e2:
                # 不再默默沿用模型自帶的。SDXL 用原版 VAE 解碼會升 fp32，
                # 1536² 實測從 4 秒變 305 秒，而且訊息藏在紀錄裡看不到。
                raise RuntimeError(i18n.tr("eng.p.vaeOverride", name=Path(want).name, arch=arch or i18n.tr("eng.unknown"))
                                   + "\n　" + i18n.tr("eng.reason", e=f"{str(e)[:70]} / {str(e2)[:70]}")) from e2
    else:
        log(i18n.tr("eng.p.vaeBuiltin"))
    # 設定裡選了 32 位元（出圖跟以前逐位元一樣，但多吃約 5 GB）就照 diffusers 原本的升 fp32
    if vae_fp32:
        if getattr(getattr(pipe, "vae", None), "config", None) is not None and getattr(pipe.vae.config, "force_upcast", False):
            log(i18n.tr("eng.p.vaeFp32"))
    elif _vae_bf16(pipe):
        log(i18n.tr("eng.p.vaeBf16"))

    if te_path:
        te = getattr(pipe, "text_encoder", None)
        if te is None:
            log("[!] " + i18n.tr("eng.p.noTe"))
        else:
            f = Path(te_path)
            try:
                if f.is_dir():
                    pipe.text_encoder = _load_component(
                        type(te).__name__, str(f), log)
                    log(i18n.tr("eng.p.te", name=f.name))
                else:
                    _load_into(te, f, log, i18n.tr("eng.p.teLabel"))
            except Exception as e:
                raise RuntimeError(i18n.tr("eng.p.teOverride", name=f.name, arch=arch or i18n.tr("eng.unknown"))
                                   + "\n　" + i18n.tr("eng.reason", e=str(e)[:110])) from e

    # 主模型用 fp8 存放（⚙ 設定 → 生成）：UNet 的權重存成 fp8、算的時候一層一層轉回 fp16，
    # 顯示卡記憶體約減半，出圖會變（同一個 seed 構圖不同）
    if fp8_weights and getattr(pipe, "unet", None) is not None:
        pipe.unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
        pipe._coco_fp8 = True
        log(i18n.tr("eng.fp8Store"))

    # PyTorch 2 的 SDPA 已經是記憶體效率版的注意力，
    # 再開 attention slicing 只會拖慢速度（實測慢 30～50%），
    # 所以只有真的缺 VRAM 的 low 模式才開。
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # VAE 的輸入尺寸每張圖都可能不同。開啟 autotuner 會讓 cuDNN 為解碼
        # 卷積挑選需要巨大 workspace 的演算法：RTX 5080 / SDXL 896x1536
        # 實測整輪 reserved 由 6.86 GiB 暴增到 12.65 GiB。這不是模型權重，
        # 也不是 cache leak；關閉 benchmark 才能讓 VRAM 峰值保持可預期。
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    if vram_mode not in VRAM_MODES:
        vram_mode = "balanced"
    if vram_mode == "low":
        pipe.enable_model_cpu_offload()
        # 這裡本來還會開 attention slicing，實測是純虧的：
        #   1536²  開 8.59 GB / 8.4s　關 7.32 GB / 6.4s
        #   1024²  開 7.10 GB / 4.1s　關 6.93 GB / 3.6s
        # 省 1.27 GB 而且快 24%——不是取捨，是兩邊都輸。
        # 原因是 PyTorch 2 的 SDPA 本來就是記憶體效率版，
        # 再切片反而逼它走回分塊路徑，配置更多也算更慢。
    else:
        pipe = pipe.to("cuda")
    pipe._coco_vram = vram_mode

    # VAE slicing 幾乎不影響速度，tiling 才會；tiling 留給 low 模式
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=True)
    log(i18n.tr("eng.loadedIn", s=f"{time.time() - t0:.1f}"))
    return pipe


def apply_scheduler(pipe, sampler, schedule=None, log=print, shift=None):
    import diffusers

    if getattr(pipe, "_coco_is_anima", False):
        _anima().set_scheduler(pipe, sampler, schedule, shift, log)
        return
    if getattr(pipe, "_coco_is_zimage", False):
        # 同一套 flow matching 取樣器，預設 shift 照 Z-Image 的（Base 6.0）
        _anima().set_scheduler(pipe, sampler, schedule, shift, log, default_shift=_zimage().SHIFT)
        return
    if getattr(pipe, "_coco_is_krea2", False):
        # 預設 shift 照 Krea 2 Turbo 的（官方固定 mu 1.15，等於 shift 3.16）
        _anima().set_scheduler(pipe, sampler, schedule, shift, log, default_shift=_krea2().SHIFT)
        return
    if getattr(pipe, "_coco_is_qwen21", False):
        # 取樣器同一套；σ 照官方依解析度算好再交給取樣器（qwen21.sigmas），shift 是「1024² 時」的值
        _qwen21().set_scheduler(pipe, sampler, schedule, shift, log)
        return
    if getattr(pipe, "_coco_is_flux2", False):
        # 同上：σ 照官方依解析度與步數算好（flux2.sigmas），shift 是「1024²、4 步時」的值
        _flux2().set_scheduler(pipe, sampler, schedule, shift, log)
        return
    if getattr(pipe, "_coco_is_flux1", False):
        # 同上：σ 照官方算好（flux1.schedule）；dev 的 shift 是「1024² 時」的值，schnell 的是固定值
        _flux1().set_scheduler(pipe, sampler, schedule, shift, log)
        return

    sampler, legacy = split_sampler(sampler or "Euler a")
    if legacy and not schedule:
        schedule = legacy
    pipe._coco_sampler = None            # 實際用的取樣器跟選的不同時才記（存圖的參數照這個寫）
    if getattr(pipe, "_coco_vpred", (False, False))[1] and sampler in SAMPLERS and sampler not in ZSNR_OK:
        # 零終端 SNR 的 V-Pred 模型：不支援的取樣器算出來不對（官方也說只用 Euler），改用 Euler 並記下來
        log("[!] " + i18n.tr("eng.p.noZsnr", s=sampler))
        sampler = pipe._coco_sampler = "Euler"
    if sampler in FLOW_ONLY:
        # 這些只實作了 flow-match 版（Anima）。SDXL 的 VP 參數化不同，
        # 直接套會出錯，所以在這裡明講而不是靜靜換掉，也不要讓它掉進
        # 下面那句比較籠統的「不認得取樣器」。
        log("[!] " + i18n.tr("eng.p.flowOnly", s=sampler))
        return
    if sampler not in SAMPLERS:
        log("[!] " + i18n.tr("eng.p.unknownSampler", s=sampler))
        return

    name, kw = SAMPLERS[sampler]
    kw = dict(kw)
    sched_kw = dict(SCHEDULES.get(schedule or "Normal", {}))

    cls = getattr(diffusers, name, None)
    if cls is None:
        log("[!] " + i18n.tr("eng.p.noScheduler", name=name))
        return

    if sched_kw.get(BETA_KEY) and not beta_available():
        log("[!] " + i18n.tr("eng.betaMissing"))
        sched_kw.pop(BETA_KEY)
    # from_config 遇到取樣器不認得的參數會**默默丟掉**（不會丟 TypeError），
    # 以前紀錄照樣寫「排程 Beta」，其實沒套上。先照建構子的參數表篩一次，講清楚。
    try:
        import inspect
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = None
    if accepted is not None:
        for k in [k for k in sched_kw if k not in accepted]:
            sched_kw.pop(k)
            log("[!] " + i18n.tr("eng.p.noSchedule", s=sampler, sch=schedule, k=k))
    kw.update(sched_kw)
    # 實際生效的排程：中繼資料寫這個，不寫介面上選的（見 save_image）
    want = schedule or "Normal"
    effective = want if want in SCHEDULES and (sched_kw or not SCHEDULES[want]) else "Normal"

    # 一律從模型**原本**的排程設定建，不能從上一個排程器的設定建：Normal 是空的
    # {}，上一次設的 use_karras_sigmas、timestep_spacing 會原封不動帶過來——
    # 以前選過 Karras 之後再選 Normal，實際上一直是 Karras，直到重新載入模型。
    base = getattr(pipe, "_coco_sched_base", None)
    if base is None:
        base = pipe._coco_sched_base = dict(pipe.scheduler.config)

    # 有些排程參數不是每個取樣器都支援，不支援就一個一個拿掉
    while True:
        try:
            pipe.scheduler = cls.from_config(base, **kw)
            break
        except TypeError as e:
            bad = [k for k in kw if k in str(e)]
            if not bad:
                log("[!] " + i18n.tr("eng.p.samplerFail", e=str(e)[:90]))
                return
            for k in bad:
                kw.pop(k, None)
                if k in sched_kw:
                    effective = "Normal"
                log("[!] " + i18n.tr("eng.p.optSkipped", s=sampler, k=k))
        except Exception as e:
            log("[!] " + i18n.tr("eng.p.samplerFail", e=str(e)[:90]))
            return
    pipe._coco_schedule = effective
    log(i18n.tr("eng.p.sampler", s=sampler, sch=effective))


# ---------------------------------------------------------------- LoRA
def _get_module(root, path):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    if hasattr(m, "base_layer"):        # 被 peft 包過的話，權重在裡面
        m = m.base_layer
    return m


def _flat_index(model):
    """建立「底線化名稱 → 實際模組路徑」的對照表

    kohya 的鍵長這樣：
        lora_unet_down_blocks_0_attentions_0_..._attn1_to_q
    模組路徑長這樣：
        down_blocks.0.attentions.0....attn1.to_q
    模組名稱本身也含底線，所以不能直接把 _ 換成 . 硬拆。
    改成把每個模組路徑的 . 換成 _ 之後建索引，直接查表，不會猜錯。
    """
    idx = {}
    if model is None:
        return idx
    for name, mod in model.named_modules():
        if not name or not hasattr(mod, "weight"):
            continue
        idx[name.replace(".", "_")] = name
    return idx


def _lora_targets(pipe):
    """回傳 [(前綴, 子模型, 對照表), ...]"""
    out = []
    unet = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    if unet is not None:
        idx = _flat_index(unet)
        if getattr(pipe, "_coco_is_anima", False):
            # Anima 的 LoRA 用的是拆分式模型的鍵名，跟 diffusers 的
            # 模組路徑對不上，補一組別名進去
            idx.update(_anima().lora_alias(len(unet.transformer_blocks)))
        if getattr(pipe, "_coco_is_zimage", False):
            # Z-Image：合起來的 qkv 對到拆開的三個、幾個改過名的模組
            idx.update(_zimage().lora_alias(unet))
        if getattr(pipe, "_coco_is_krea2", False):
            # Krea 2：原始命名（blocks.N.attn.wq、txtfusion…）對到 diffusers 的
            idx.update(_krea2().lora_alias(unet))
        if getattr(pipe, "_coco_is_qwen21", False):
            # Qwen Image 2.1：ComfyUI 合起來的 gate_up 對到拆開的 gate_layer／proj
            idx.update(_qwen21().lora_alias(unet))
        if getattr(pipe, "_coco_is_flux2", False):
            # FLUX.2：BFL 原始命名（double_blocks.N.img_attn.qkv…）對到 diffusers 的，qkv 拆三個
            idx.update(_flux2().lora_alias(unet))
        if getattr(pipe, "_coco_is_flux1", False):
            # FLUX.1：同樣是 BFL 原始命名，qkv 拆三個、單流的 linear1 拆四個（q、k、v、MLP）
            idx.update(_flux1().lora_alias(unet))
        # diffusion_model.／transformer. 開頭的是 ComfyUI／diffusers 的寫法（點分隔，查表前換成底線）
        out.append((("lora_unet_", "diffusion_model.", "transformer."), unet, idx))
    te1 = getattr(pipe, "text_encoder", None)
    if te1 is not None:
        out.append((("lora_te1_", "lora_te_"), te1, _te_index(te1)))
    te2 = getattr(pipe, "text_encoder_2", None)
    if te2 is not None:
        out.append((("lora_te2_",), te2, _te_index(te2)))
    return out


def _te_index(model):
    """文字編碼器的對照表。kohya 的鍵是 text_model_encoder_layers_…（舊版 transformers 的 CLIPTextModel 外面
    包一層 text_model）；新版的 CLIPTextModel 少了那一層，模組名稱是 encoder.layers.…，直接查會全部對不上，
    LoRA 的 CLIP-L 部分整段被靜靜略過（實測 SDXL 的 LoRA 少套 72 個模組，畫面跟 Forge 差很多）。
    兩種寫法都放進表裡"""
    idx = _flat_index(model)
    for k, v in list(idx.items()):
        if not k.startswith("text_model_"):
            idx.setdefault(f"text_model_{k}", v)
    return idx


def _te_changed(pipe, model):
    """動到文字編碼器的權重（LoRA 的 te 部分）：快取編碼結果的架構（FLUX.1）看這個號碼決定要不要重新編碼"""
    if model is not (getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)):
        pipe._coco_lora_ver = getattr(pipe, "_coco_lora_ver", 0) + 1


def _wrote(pipe):
    """LoRA 要開始改權重了（改之前記）：apply_loras 用這個號碼分辨失敗時權重有沒有被動過"""
    pipe._coco_lora_writes = getattr(pipe, "_coco_lora_writes", 0) + 1


def _backup_store(pipe):
    """原始權重的備份，以及「上次還原之後被動過」的模組集合

    備份是跟著 pipe 走的常駐快取，還原之後**不清掉**：還原完權重就等於
    備份，下次再動同一個模組不必重新備份。以前每次換 LoRA 都清掉重備，
    SDXL 光備份就要 0.6 秒（915 個模組、2.8B 參數）。代價是移除全部
    LoRA 之後備份仍佔著系統記憶體，換模型時隨舊的 pipe 一起釋放。
    """
    d = getattr(pipe, "_coco_lora_backup", None)
    if d is None:
        d = {}
        pipe._coco_lora_backup = d
    dirty = getattr(pipe, "_coco_lora_dirty", None)
    if dirty is None:
        dirty = set()
        pipe._coco_lora_dirty = dirty
    return d, dirty


def restore_weights(pipe, log=print):
    """把被 LoRA 動過的權重還原成備份裡的原始值

    不用「再套一次負權重」還原，因為那在 fp16 / bfloat16 底下不精確：
    就地累加會把和捨入回低精度，加了再減不會回到原點。實測 Anima
    （bfloat16）殘差最大 2^-13，出圖差異約是 LoRA 效果的 7%。

    Forge、Neo Forge、ComfyUI、A1111 四個都是存一份原始權重再複製回來
    （ComfyUI 的 ModelPatcher.backup、A1111 的 network_weights_backup），
    這裡採同樣做法。備份放 CPU，不佔 VRAM。只還原 dirty 裡的模組——
    不在裡面的本來就等於備份。
    """
    store, dirty = _backup_store(pipe)
    if not dirty:
        return 0
    n = 0
    for slot in list(dirty):
        model, name = slot
        try:
            m = _get_module(model, name)
            _te_changed(pipe, model)
            if hasattr(m, "clear_lowrank"):          # Krea 2 的 fp8 層：LoRA 掛在旁邊，拿掉就好
                m.clear_lowrank()
                dirty.discard(slot)
                n += 1
                continue
            W = m.weight
            W.data.copy_(store[slot].to(W.device))
            dirty.discard(slot)
            n += 1
        except Exception as e:
            log("[!] " + i18n.tr("eng.p.restoreFail", name=name, e=str(e)[:70]))
    return n


def fuse_lora(pipe, path: Path, scale: float):
    """把 kohya 格式的 LoRA 併進權重。

    不使用 diffusers 的私有轉換函式 —— 那是 _ 開頭的內部 API，
    改版就會壞（0.39 能用、後來的版本回傳空的）。
    這裡自己做鍵名對照，只依賴 safetensors 與模組樹本身。

    每個模組第一次被動到時，會先把原始權重備份到 CPU，
    之後要移除或改權重就走 restore_weights，不靠負權重回推。

    B @ A 一律在 GPU 上算（fp32）。在 CPU 算的話，SDXL 一個 LoRA 要
    產生 2.8B 個 fp32 數字，實測 1.95 秒；GPU 上 0.22 秒。

    算的時候要暫時關掉 TF32（load_model 為了速度全域打開了）。TF32 只有
    約三位有效數字，實測讓同 seed 的出圖跟 CPU 算的差到平均 2/255；
    關掉之後 GPU 的 fp32 結果跟 CPU 逐位元相同。
    """
    import torch

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _fuse_lora(pipe, path, scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _fuse_lora(pipe, path, scale):
    import torch
    from safetensors.torch import load_file

    sd = load_file(str(path))
    targets = _lora_targets(pipe)
    store, dirty = _backup_store(pipe)
    calc = "cuda" if torch.cuda.is_available() else "cpu"

    # 先把同一個模組的 down / up / alpha 收在一起
    mods = {}
    for k, v in sd.items():
        for suf in (".lora_down.weight", ".lora_up.weight", ".alpha",
                    ".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suf):
                base = k[: -len(suf)]
                slot = {".lora_down.weight": "down", ".lora_A.weight": "down",
                        ".lora_up.weight": "up", ".lora_B.weight": "up",
                        ".alpha": "alpha"}[suf]
                mods.setdefault(base, {})[slot] = v
                break
    # 會讓整個檔案失敗的先查：在迴圈裡才發現的話，前面的模組已經改了（apply_loras 得整組還原重套）
    for base, parts in mods.items():
        if "alpha" in parts and parts["alpha"].numel() != 1:
            raise ValueError(i18n.tr("eng.p.alphaShape", base=base, shape=tuple(parts['alpha'].shape)))

    merged = 0
    why = {"no_pair": 0, "no_target": 0, "no_module": 0, "shape": 0}
    sample = []

    for base, parts in mods.items():
        if "down" not in parts or "up" not in parts:
            why["no_pair"] += 1
            continue

        hit = None
        for prefixes, model, idx in targets:
            for pre in prefixes:
                if base.startswith(pre):
                    name = idx.get(base[len(pre):]) or idx.get(base[len(pre):].replace(".", "_"))
                    if name:
                        hit = (model, name)
                    break
            if hit:
                break

        if hit is None:
            why["no_target"] += 1
            if len(sample) < 3:
                sample.append(base)
            continue

        # 一個 LoRA 對到好幾個模組（Z-Image 合起來的 qkv → 拆開的 to_q／to_k／to_v）：變化量照順序切開
        names = hit[1] if isinstance(hit[1], tuple) else (hit[1],)
        try:
            ms = [_get_module(hit[0], n) for n in names]
            Ws = [m.weight for m in ms]
        except Exception:
            why["no_module"] += 1
            continue

        A, B = parts["down"], parts["up"]
        rank = A.shape[0]
        alpha = float(parts["alpha"]) if "alpha" in parts else rank
        s = scale * (alpha / rank)

        A2 = A.reshape(rank, -1).to(calc, torch.float32)
        B2 = B.reshape(B.shape[0], -1).to(calc, torch.float32)
        if all(hasattr(m, "add_lowrank") for m in ms):
            # Krea 2 的 scaled fp8、NVFP4 層：不併進權重，另外掛一條低秩的路（krea2 的 LowRankLinear.add_lowrank）
            # 用模組記的維度：NVFP4 的權重兩個一組裝成一個位元組，張量的寬度只有一半
            rows = [m.out_features for m in ms]
            if B2.shape[0] != sum(rows) or any(A2.shape[1] != m.in_features for m in ms):
                why["shape"] += 1
                continue
            _wrote(pipe)
            for name, m, up in zip(names, ms, B2.split(rows)):
                m.add_lowrank(A2, s * up)
                dirty.add((hit[0], name))
            _te_changed(pipe, hit[0])
            merged += 1
            continue
        try:
            full = B2 @ A2
            rows = [W.shape[0] for W in Ws]
            if full.shape[0] != sum(rows):
                raise ValueError
            deltas = [d.reshape(W.shape) for d, W in zip(full.split(rows), Ws)]
        except Exception:
            why["shape"] += 1
            continue

        _wrote(pipe)
        for name, W, delta in zip(names, Ws, deltas):
            slot = (hit[0], name)
            if slot not in store:
                # 第一次被動到：現在的值就是原始值（見 _backup_store）
                store[slot] = W.data.detach().to("cpu", copy=True)
            dirty.add(slot)
            if W.dtype.itemsize == 1:
                # fp8 存放（Z-Image 的 ⚙ 設定）：fp8 不能直接相加，轉回 32 位元加完再存回去
                W.data.copy_((W.data.float() + (s * delta).to(W.device)).to(W.dtype))
            else:
                W.data += (s * delta).to(W.dtype).to(W.device)
        _te_changed(pipe, hit[0])
        merged += 1

    return merged, {"total": len(mods), "why": why, "sample": sample,
                    "keys": len(sd)}


def apply_loras(pipe, loras, log=print, applied=None):
    """套用 LoRA。applied 是已生效的清單，會就地更新。

    純粹新增（既有的都沒動）→ 直接往上疊，不必碰已經生效的。
    有任何一個要移除或改權重 → 先把權重還原成備份的原始值，再整組重套。
    兩種情況都不需要重新載入模型。
    某個 LoRA 併到一半才失敗（顯示卡記憶體不足之類）→ 也是整組還原、其他的重套：
    不然改了一半的權重不在 applied 裡，之後拿掉 LoRA 也不會還原，後面的圖都帶著它。
    """
    cur = {p: w for p, w in (applied if applied is not None else [])}
    want = {str(p): float(w) for p, w in (loras or [])}

    gone = [p for p, w in cur.items() if want.get(p) != w]
    if gone:
        n = restore_weights(pipe, log)
        names = i18n.join(Path(p).name for p in gone)
        log(i18n.tr("eng.p.restoreN", n=n, names=names))
        cur = {}

    for p, w in want.items():
        if cur.get(p) == w:
            continue
        f = Path(p)
        if not f.exists():
            log("[!] " + i18n.tr("eng.p.loraMissing", f=f))
            continue
        writes = getattr(pipe, "_coco_lora_writes", 0)
        try:
            n, info = fuse_lora(pipe, f, w)
        except Exception as e:
            log("[X] " + i18n.tr("eng.p.loraFail", name=f.name, e=f"{type(e).__name__}: {str(e)[:100]}"))
            if getattr(pipe, "_coco_lora_writes", 0) != writes:
                n = restore_weights(pipe, log)
                log(i18n.tr("eng.p.restoreHalf", n=n, name=f.name))
                if applied is not None:
                    applied[:] = []
                return apply_loras(pipe, [(q, v) for q, v in want.items() if q != p], log, applied)
            continue
        if n:
            log(i18n.tr("eng.p.loraOk", name=f.name, w=w, n=n))
            cur[p] = w
        else:
            log("[X] " + i18n.tr("eng.p.loraNone", name=f.name, keys=info['keys'], total=info['total'], why=info['why']))
            for smp in info["sample"]:
                log("    " + i18n.tr("eng.p.loraKey", k=smp))

    if applied is not None:
        applied[:] = list(cur.items())
    return list(cur.items())


# ---------------------------------------------------------------- 生成
def _decode_low(pipe, latents, log, may_keep=False):
    """先把 UNet 移出 VRAM 再解碼。

    構想來自 Forge 與 ComfyUI：解碼階段其實用不到 UNet，
    先把它搬離 VRAM，VAE 升到 fp32 時就不會疊在一起。

    早期實測這條路反而更貴，原因是這裡沒有包 no_grad——
    pipe() 內部自己有，手動呼叫 vae.decode 就沒有，autograd 把
    整條解碼的中間張量都留著。補上之後 896x1536 的解碼尖峰從
    16.90 GB 降到 4.82 GB。

    搬出再搬回一趟要 1.4 秒（SDXL 1024×1344），比 fp32 解碼本身還久。
    高速模式下若剩餘 VRAM 放得下解碼，就把 UNet 留著。出圖不受影響：
    搬不搬只是位置不同，分階段與否的出圖差異本來就是 0.0000。
    may_keep 只在「自動」時為真；手動指定開啟就是要省 VRAM，照搬。
    """
    import torch

    vae = pipe.vae
    need = (vae.dtype == torch.float16
            and getattr(vae.config, "force_upcast", False))
    unet = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    moved = False
    if unet is not None and str(getattr(unet, "device", "")) != "cpu":
        keep = False
        if may_keep and getattr(pipe, "_coco_vram", "") == "speed":
            try:
                torch.cuda.empty_cache()      # 見 decode_budget：碎片不算數
                free, _ = torch.cuda.mem_get_info()
                px = latents.shape[-2] * latents.shape[-1] * 64
                keep, avail, want = decode_budget(px, need, free)
                if keep:
                    log(i18n.tr("eng.p.unetStays", avail=f"{avail:.1f}", want=f"{want:.1f}"))
            except Exception:
                keep = False
        if not keep:
            unet.to("cpu")
            torch.cuda.empty_cache()
            moved = True
            log(i18n.tr("eng.p.unetOut"))

    try:
        with torch.no_grad():
            if need:
                vae.to(dtype=torch.float32)
            # 跟 diffusers 一般解碼的順序一樣，出圖才逐位元相同（R21）：bf16 的 VAE（_vae_bf16）
            # 先在 latent 原本的精度（fp16）除縮放係數，進 decode 才轉 bf16；先轉 bf16 再除，
            # 精度少兩個位元，896×1536 實測差 max 30／平均 0.43
            if getattr(vae, "_coco_bf16", False):
                lat = latents
            else:
                lat = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype) \
                    if hasattr(vae, "post_quant_conv") else latents.to(vae.dtype)
            img = vae.decode(lat / vae.config.scaling_factor,
                             return_dict=False)[0]
            if need:
                vae.to(dtype=torch.float16)
            img = pipe.image_processor.postprocess(img.detach(),
                                                   output_type="pil")[0]
    finally:
        if moved:
            unet.to("cuda")
    return img


def _use_staged_decode(req, value):
    """解析 auto/on/off（並相容舊版 bool）後決定是否分階段解碼。"""
    if isinstance(value, bool):
        return value, "手動"
    mode = str(value or "auto").strip().lower()
    if mode == "on":
        return True, "手動"
    if mode == "off":
        return False, "手動"
    px = int(req.get("width", 0)) * int(req.get("height", 0))
    # 用了參考圖（ControlNet）也分階段：解碼前先把 ControlNet（SDXL 的約 2.4 GB）搬回系統記憶體。
    # 2026-10-08 實測 RTX 5080、高速模式、896×1152＋NoobAI 的 openpose：峰值 reserved 13.76 → 10.98 GiB、
    # 7.8 → 7.4 秒、出圖逐位元相同；不分階段時使用者的顯示卡溢出到共用記憶體 1 GB 多
    return px > STAGED_DECODE_ABOVE or bool(req.get("controlnets")), "自動"


def rng_device(req):
    """亂數在哪裡產生。cpu（預設）：跟 Forge／A1111 一樣——同一個 seed 畫出同樣的構圖（Forge 的參數寫 RNG: CPU）；
    gpu：CoCo 以前的做法，同一個 seed 的雜訊跟 Forge 完全不同。
    diffusers 拿到 CPU 的 generator 會在 CPU 上產生再搬到顯示卡（實測 fp16 跟 Forge 的 fp32 轉 fp16 逐位元相同），
    起始雜訊與每一步的雜訊都從同一個 generator 依序取，跟 Forge 的 ImageRNG 順序一樣"""
    return "cpu" if str(req.get("rng") or "gpu").lower() == "cpu" else "cuda"


def _clip_skip_floor(pipe):
    """A1111／Forge 層號的下限。SDXL 兩顆 encoder 最低 2、SD1.x 最低 1。"""
    return 2 if getattr(pipe, "text_encoder_2", None) is not None else 1


def _effective_clip_skip(pipe, req, log=print):
    """回傳**實際會套用**的層號，不是介面上選的值。

    介面選比下限小的值會被夾住（SDXL 選 1 實際跑 2），中繼資料必須寫
    夾限後的值，否則同一張圖的參數重跑會對不起來。

    夾上限的部分留在 encode_chunk 裡——SDXL 兩顆 encoder 深度不同
    （CLIP-L 12 層、CLIP-G 32 層），選超過深度的值時本來就沒有單一
    正確答案，而那是不會發生在正常操作裡的輸入。
    """
    try:
        wanted = int(req.get("clip_skip", 2))
    except (TypeError, ValueError):
        wanted = 2
    floor = _clip_skip_floor(pipe)
    if wanted < floor:
        log(i18n.tr("eng.p.clipFloor", w=wanted, f=floor))
    return max(wanted, floor)


# ---------------------------------------------------------------- img2img
# 強度太小等於沒改，太接近 0 時排程要拉到上千步才湊得出指定的步數
STRENGTH_MIN = 0.05


def img2img_steps(steps, strength):
    """img2img 的步數：回傳 (排程總步數, 實際要跑的步數)

    **介面上的「步數」就是實際跑的步數**（ComfyUI 的做法）：排程照
    steps / strength 步算完整一條，只跑後面那 steps 步。A1111／Forge 預設
    是反過來——排程照 steps 算、只跑 steps × strength 步，強度 0.3、28 步
    只剩 8 步，細節容易糊。

    diffusers 的 img2img 用 int(總步數 × strength) 決定跑幾步，浮點誤差會
    差一步（28 / 0.3 → 93 → 27.9 → 27），所以往上找到剛好夠的那個總步數。
    """
    steps = max(1, int(steps))
    strength = min(1.0, max(STRENGTH_MIN, float(strength)))
    total = max(steps, round(steps / strength))
    while min(int(total * strength), total) < steps:
        total += 1
    return total, min(int(total * strength), total)


def load_init_image(path, width, height):
    """讀底圖、轉 RGB、縮放到要生成的大小

    透明的部分鋪白底：直接丟掉 alpha 的話，透明區常殘留去背前的顏色
    （縮圖那邊實際踩過，見 tests/test_thumbs.py 的紫色）。
    """
    from PIL import Image
    if isinstance(path, Image.Image):
        im = path.copy()
    else:
        im = Image.open(path)
        im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        # 其實不透明的（有些工具存成 alpha 254）直接丟掉 alpha：鋪白底會讓這些像素偏一階，
        # 局部重繪、修臉的「遮罩外逐位元不變」就對不上原檔了
        if im.getchannel("A").getextrema()[0] < 250:
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    size = (int(width), int(height))
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def _generate_impl(pipe, req, cond_mode="auto", on_step=None, log=print,
                   low_decode="auto"):
    """req 是一個 dict，回傳 (PIL Image, seed, cond_mode, 實際的 clip_skip)"""
    import torch

    vram = getattr(pipe, "_coco_vram", "balanced")
    _tile_vae(pipe, req, log)      # 尖峰在解碼，跟解析度有關而不是跟模式有關

    # Anima 走自己的路：flow-match 排程器、條件走 text_conditioner，
    # 底下那套 CLIP 分塊加權（conditioning.py）對它完全不適用。
    if getattr(pipe, "_coco_is_anima", False):
        # low_decode 對 Anima 無效，不傳進去：它的尖峰在編碼＋取樣
        # （896x1536 是 10.73 GB），解碼只有 7.03 GB。先卸載 DiT 實測
        # 10.73 → 10.73 GB，只白花 1.3 秒。詳見 SPEC §6.3.4。
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model="Anima"))
        img, seed = _anima().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, "anima", None
    if getattr(pipe, "_coco_is_zimage", False):
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model="Z-Image"))
        img, seed = _zimage().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, "zimage", None
    if getattr(pipe, "_coco_is_krea2", False):
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model="Krea 2"))
        img, seed = _krea2().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, "krea2", None
    if getattr(pipe, "_coco_is_qwen21", False):
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model="Qwen Image 2.1"))
        img, seed = _qwen21().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, "qwen21", None
    if getattr(pipe, "_coco_is_flux2", False):
        arch = pipe._coco_arch
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model=_flux2().TITLE[arch]))
        img, seed = _flux2().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, arch, None
    if getattr(pipe, "_coco_is_flux1", False):
        arch = pipe._coco_arch
        if req.get("controlnets"):
            log("[!] " + i18n.tr("eng.p.noCn", model=_flux1().TITLE[arch]))
        img, seed = _flux1().generate(pipe, req, on_step=on_step, log=log)
        return img, seed, arch, None

    staged, staged_by = _use_staged_decode(req, low_decode)
    if staged:
        log(i18n.tr("eng.p.staged", by=i18n.tr("eng.p.byAuto") if staged_by == "自動" else i18n.tr("eng.p.byManual")))

    seed = int(req.get("seed", -1))
    if seed < 0:
        import random
        seed = random.randrange(1 << 31)
    g = torch.Generator(rng_device(req)).manual_seed(seed)
    # 全域 RNG 也要種，跟 anima.generate 對齊。傳入的 generator 只覆蓋
    # 有接受它的那些呼叫；只要有任何一個運算退回全域 RNG，結果就會取決於
    # 這個 process 之前跑過什麼——症狀是「同參數偶爾出不同的圖、無法重現」。
    # 這裡不是已知 bug 的修復，是把兩條路徑的嚴謹度拉齊。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    kw = dict(width=int(req["width"]), height=int(req["height"]),
              num_inference_steps=int(req["steps"]),
              guidance_scale=float(req["cfg"]), generator=g)

    # 自動判定之前一定要先把文字編碼器接回來。平衡模式會在編碼後把
    # pipe.text_encoder 設成 None（見 _park_encoders），而 inspect 正是
    # 看那個屬性在不在——不先接回來的話，第二次以後的生成都會誤判成
    # native，同參數同 seed 卻出不一樣的圖。
    if vram == "balanced":
        _wake_encoders(pipe, log)

    # Embeddings（core/embeddings.py）：載進編碼器，編碼前把提示詞裡的名字換成對應的 token。
    # 中繼資料寫的還是 req 裡原本的提示詞
    import embeddings
    if req.get("embeddings"):
        embeddings.ensure(pipe, req["embeddings"], log)
    ti = pipe.__dict__.get("_coco_ti")

    mode = cond_mode
    if mode == "auto":
        try:
            import conditioning
            mode = "a1111" if conditioning.inspect(pipe)["ok"] else "native"
        except Exception:
            mode = "native"

    clip_skip = _effective_clip_skip(pipe, req, log)

    if mode == "a1111":
        import conditioning
        pe, pp, ne, npd = conditioning.encode_prompt(
            pipe, embeddings.expand(req["prompt"], ti), embeddings.expand(req.get("negative", ""), ti),
            clip_skip=clip_skip, norm=req.get("emphasis") != "No norm")
        kw.update(prompt_embeds=pe, negative_prompt_embeds=ne)
        if pp is not None:
            kw.update(pooled_prompt_embeds=pp,
                      negative_pooled_prompt_embeds=npd)
        # 編完就把文字編碼器搬回 CPU。a1111 模式已經把提示詞算成向量，
        # 接下來幾十步去噪完全用不到它們，讓它們佔著 1.5 GB 沒有道理。
        # low 模式不用管，enable_model_cpu_offload 自己會處理。
        if vram == "balanced":
            moved = _park_encoders(pipe, log)
            if moved:
                log(i18n.tr("eng.p.teToCpu", names=i18n.join(moved)))
    else:
        # native 模式是管線自己在內部呼叫編碼器，不能搬走
        kw.update(prompt=embeddings.expand(req["prompt"], ti, full=False),
                  negative_prompt=embeddings.expand(req.get("negative", ""), ti, full=False))
        # native 超過上限就直接截斷而且不會說。使用者偏好 native 的出圖，
        # 沒有這段提示的話會在不知情下被丟掉後半段提示詞。
        try:
            import conditioning
            for name, n, limit in conditioning.native_overflow(
                    pipe, embeddings.expand(req["prompt"], ti), embeddings.expand(req.get("negative", ""), ti)):
                log("[!] " + i18n.tr("eng.p.nativeCut", field=conditioning.field_label(name), n=n, limit=limit, cut=n - limit))
        except Exception:
            pass
        # diffusers 的參數是「從下限再往前跳幾層」，不是 A1111 的層號本身，
        # 所以要換算後再傳入。clip_skip 已經夾過下限，相減不會是負的。
        kw["clip_skip"] = clip_skip - _clip_skip_floor(pipe)

    # img2img：換成共用同一組元件的 Img2Img 管線（不會重新配置記憶體）。
    # 要在這裡才建：文字編碼器可能剛被 _park_encoders 設成 None，
    # 排程器也可能剛被 apply_scheduler 換掉，建早了會拿到舊的。
    run = pipe
    total = int(req["steps"])
    if req.get("init_image"):
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
        n_all, total = img2img_steps(req["steps"], req.get("strength", 0.5))
        mask = req.get("_mask")
        # torch_dtype=None 不能省：from_pipe 預設是 float32，會把**共用的**
        # UNet、VAE、文字編碼器整組轉成 fp32（實測 UNet 4.78 → 9.56 GiB），
        # VRAM 爆掉溢到系統記憶體，每步 0.18 → 6.6 秒，之後的 txt2img 也一起變慢
        cls = StableDiffusionXLInpaintPipeline if mask is not None else StableDiffusionXLImg2ImgPipeline
        run = cls.from_pipe(pipe, torch_dtype=None)
        try:
            run.set_progress_bar_config(disable=True)
        except Exception:
            pass
        if mask is None:
            kw.pop("width")
            kw.pop("height")
        else:
            # 局部重繪：每一步把遮罩外換回底圖的加噪版本（diffusers 內建）。
            # 給它柔化遮罩「有碰到」的範圍，柔化的邊才會落在重畫過的地方
            kw["mask_image"] = mask.point(lambda v: 255 if v > 0 else 0)
        kw.update(image=load_init_image(req["init_image"], req["width"], req["height"]),
                  strength=min(1.0, max(STRENGTH_MIN, float(req.get("strength", 0.5)))),
                  num_inference_steps=n_all)
        log(i18n.tr("eng.p.i2i", mode=i18n.tr("eng.inpaint") if mask is not None else "img2img", s=f"{kw['strength']:g}", run=total, n=n_all))

    # ControlNet（core/controlnet_run.py）：換成同一組元件加上 ControlNet 的管線；參考圖在這裡才做前處理（抽姿勢、深度…）
    cn_cleanup = None
    if req.get("controlnets"):
        # 先找工作行程的那份（worker 用 import controlnet_run），不然同一個模組會載成兩份、快取與倒數各管各的
        try:
            import controlnet_run as CR
        except ImportError:
            from core import controlnet_run as CR
        run, extra, cn_cleanup = CR.build(pipe, req["controlnets"], int(req["width"]), int(req["height"]),
                                          init=bool(req.get("init_image")), mask=req.get("_mask") is not None, log=log)
        kw.update(extra)

    if on_step:
        import inspect as _i
        try:
            params = _i.signature(run.__call__).parameters
        except Exception:
            params = {}

        if "callback_on_step_end" in params:
            def cb(pipe_, step, timestep, cbkw):
                on_step(step + 1, total)
                return cbkw
            kw["callback_on_step_end"] = cb
        elif "callback" in params:          # 舊版 diffusers
            def cb_old(step, timestep, latents):
                on_step(step + 1, total)
            kw["callback"] = cb_old
            kw["callback_steps"] = 1

    try:
        if staged:
            kw["output_type"] = "latent"
            with _ngms(run, ngms_of(req)):
                lat = run(**kw).images
            if cn_cleanup:                      # ControlNet 先搬回系統記憶體，解碼才有空間
                cn_cleanup()
                cn_cleanup = None
            img = _decode_low(pipe, lat, log, may_keep=(staged_by == "自動"))
        else:
            # 交給 diffusers 自己處理 VAE 的精度，它會在需要時升到 fp32
            with _ngms(run, ngms_of(req)):
                img = run(**kw).images[0]
    finally:
        if cn_cleanup:
            cn_cleanup()
    return img, seed, mode, clip_skip


def ngms_of(req):
    """NGMS 的值（0＝關）。只有 SDXL／SD 1.5 用得到（_generate_impl）"""
    try:
        v = float(req.get("ngms") or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 and v == v else 0.0


def _ngms(run, value):
    """Forge／A1111 的 NGMS（Negative Guidance minimum sigma）：雜訊低於 value 的步驟，每隔一步只算正面提示詞（CFG 當 1），
    負面那一半不算，所以比較快。跟 Forge 的 sd_samplers_cfg_denoiser 一樣：從這一輪第一次算 UNet 起數，第 1、3、5…次
    （從 0 數）而且當下的 σ 小於 value 才省略。

    做法：包住 UNet。要省略的那一次只拿批次的後半（diffusers 把負面放前半、正面放後半）去算，結果複製成兩份，
    管線的 CFG 算出來就剛好等於正面那份（負面＋CFG×(正面−負面)，兩份一樣時就是正面）。
    要省略哪一次看排程器的 step_index（img2img 減掉 begin_index）；DPM++ 這類第一次呼叫時還沒有 step_index，當成第 0 次"""
    import contextlib
    import torch

    unet, sched = getattr(run, "unet", None), getattr(run, "scheduler", None)
    if not value or unet is None or sched is None:
        return contextlib.nullcontext()

    def skip(n):
        i = getattr(sched, "step_index", None)
        if i is None or not getattr(run, "do_classifier_free_guidance", False):
            return False
        k = i - (getattr(sched, "begin_index", None) or 0)
        return k % 2 == 1 and float(sched.sigmas[i]) < value

    def half(x, n):
        if torch.is_tensor(x):
            return x[n // 2:] if x.dim() and x.shape[0] == n else x
        if isinstance(x, dict):
            return {k: half(v, n) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(half(v, n) for v in x)
        return x

    @contextlib.contextmanager
    def wrap():
        own = "forward" in unet.__dict__
        orig = unet.forward

        def forward(sample, *a, **kw):
            n = sample.shape[0]
            if not skip(n):
                return orig(sample, *a, **kw)
            out = orig(half(sample, n), *half(a, n), **half(kw, n))
            if isinstance(out, tuple):
                return (torch.cat([out[0], out[0]]),) + tuple(out[1:])
            out.sample = torch.cat([out.sample, out.sample])
            return out

        unet.forward = forward
        try:
            yield
        finally:
            if own:
                unet.forward = orig
            else:
                del unet.forward
    return wrap()


def attention_priority():
    """讓 SDPA 優先選 cuDNN 注意力 kernel

    PyTorch 預設的順序是 flash → efficient → math → cuDNN，而 Windows 版
    的 torch 沒有編 flash，所以實際一律落到 efficient。cuDNN 被排在最後，
    等於從來不會被選到。

    RTX 5080、bf16、16 頭 × 128 維、5632 token（Anima 1024×1408）實測：
    efficient 5.8 ms、cuDNN 2.4 ms。Anima 整步 0.97 → 0.77 秒。
    torch 2.11 與 2.14 數字一樣，所以這不是升版能解的。

    cuDNN 不支援的形狀（例如 VAE 裡的單頭 512 維）會自動往下一個退，
    不會報錯。**會改變出圖**：不同 kernel 的浮點累加順序不同，
    同 seed 的圖會有極小的逐像素差異。
    """
    import contextlib
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.MATH], set_priority=True)
    except Exception:          # 舊版 torch 沒有 set_priority
        return contextlib.nullcontext()


def generate(pipe, req, cond_mode="auto", on_step=None, log=print,
             low_decode="auto"):
    """回傳 (PIL Image, seed, cond_mode, 實際的 clip_skip)。

    所有自製前處理也要在 inference mode 裡，不能只依賴 pipe() 的 decorator。
    """
    import torch
    with torch.inference_mode(), attention_priority():
        if req.get("upscale"):
            return _hires(pipe, req, cond_mode, on_step, log, low_decode)
        if req.get("outpaint"):
            return _outpaint(pipe, req, cond_mode, on_step, log, low_decode)
        if req.get("faces"):
            return _face_fix(pipe, req, cond_mode, on_step, log, low_decode)
        if req.get("mask_image"):
            return _inpaint(pipe, req, cond_mode, on_step, log, low_decode)
        return _generate_impl(pipe, req, cond_mode, on_step, log, low_decode)


def _face_fix(pipe, req, cond_mode, on_step, log, low_decode):
    """修臉（core/faces.py）：每張臉的外框當遮罩，一張接一張用局部重繪的「只重畫遮罩附近」修在同一張圖上。

    req["faces"]：[{"box": 原圖座標, "prompt": 這張臉另外加的提示詞}]；req["face_size"]：偵測時的原圖大小。
    每張臉的 seed 依序加一；進度條把幾張臉接成一條"""
    import random
    try:
        from core import faces as F
    except ImportError:
        import faces as F
    W, H = int(req["width"]), int(req["height"])
    img = load_init_image(req["init_image"], W, H)
    todo = req["faces"]
    src = req.get("face_size") or [W, H]
    seed = int(req.get("seed", -1))
    if seed < 0:
        seed = random.randrange(1 << 31)
    n = len(todo)
    mode = clip = None
    for i, f in enumerate(todo):
        box = F.scale_box(f["box"], src, (W, H))
        extra = str(f.get("prompt") or "").strip()
        log(i18n.tr("eng.p.face", i=i + 1, n=n, w=box[2] - box[0], h=box[3] - box[1]) + (i18n.tr("eng.p.faceExtra", t=extra) if extra else ""))
        sub = {k: v for k, v in req.items() if k not in ("faces", "face_size", "mask_image")}
        # 臉常常只有幾十像素寬：放大上限放寬到 8 倍，才畫得到 SDXL 擅長的 1MP 級（ADetailer 也是放大到接近原圖大小）
        sub.update(init_image=img, mask_image=F.box_mask((W, H), box), inpaint_area="masked", seed=seed + i, max_upscale=8,
                   prompt=f"{req['prompt']}, {extra}" if extra else req["prompt"])
        step = (lambda k: (lambda s, total: on_step(k * total + s, n * total)))(i) if on_step else None
        img, _, mode, clip = _inpaint(pipe, sub, cond_mode, step, log, low_decode)
    return img, seed, mode, clip


HIRES_MARGIN = 96         # 補細節每塊往外多拿的範圍：上下文 32＋遮罩柔化（32）往外擴約 57，都在這裡面


def _hires(pipe, req, cond_mode, on_step, log, low_decode):
    """高清化（放大＋補細節）：先用放大模型放大到 width×height（core/upscale.py，沒選模型就 Lanczos），
    再切成重疊的小塊（每塊加上四周的上下文剛好 1MP），一塊接一塊用局部重繪的「只重畫遮罩附近」補細節。
    每塊只改自己那塊（外面逐位元不變），接縫靠遮罩柔化與重疊蓋過去。每塊的 seed 依序加一，進度條接成一條。
    1 倍（大小不變）：先放大到 upscale.detail_size 補細節，最後用 Lanczos 縮回 width×height。
    req["upscale"]：{"model": 放大模型的路徑（空的＝Lanczos）, "scale": 倍數}"""
    import random
    from PIL import Image
    try:
        from core import faces as F, upscale as U
    except ImportError:
        import faces as F
        import upscale as U
    up = req["upscale"]
    FW, FH = int(req["width"]), int(req["height"])            # 最後的大小
    W, H = U.detail_size(FW, FH) if float(up.get("scale") or 0) == 1 else (FW, FH)   # 補細節的畫布
    with Image.open(req["init_image"]) as im0:
        sw, sh = im0.size
    src = load_init_image(req["init_image"], sw, sh)          # 透明的地方鋪白底（重畫要的是不透明的圖）
    log(i18n.tr("eng.p.upscale", sw=sw, sh=sh, fw=FW, fh=FH, m=Path(up['model']).name if up.get('model') else 'Lanczos')
        + (i18n.tr("eng.p.upscaleDetail", w=W, h=H) if (W, H) != (FW, FH) else ""))
    img = U.upscale_to(src, (W, H), up.get("model") or None)
    U.release()                                             # 放大模型用完就卸載，顯示卡留給補細節
    boxes = U.tile_boxes(W, H)
    seed = int(req.get("seed", -1))
    if seed < 0:
        seed = random.randrange(1 << 31)
    n = len(boxes)
    mode = clip = None
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        log(i18n.tr("eng.p.detail", i=i + 1, n=n))
        # 只拿這塊四周 HIRES_MARGIN 的範圍去做（遮罩柔化與上下文都在裡面）：整張大圖的柔化、貼回很慢，
        # 實測 3072² 每塊多花 7 秒、而且圖越大越慢。範圍外本來就不會動，整塊貼回去結果一樣
        rx0, ry0 = max(0, x0 - HIRES_MARGIN), max(0, y0 - HIRES_MARGIN)
        rx1, ry1 = min(W, x1 + HIRES_MARGIN), min(H, y1 + HIRES_MARGIN)
        region = img.crop((rx0, ry0, rx1, ry1))
        sub = {k: v for k, v in req.items() if k not in ("upscale", "faces", "face_size", "mask_image")}
        sub.update(init_image=region, width=rx1 - rx0, height=ry1 - ry0, inpaint_area="masked", seed=seed + i, max_upscale=1,
                   mask_image=F.box_mask((rx1 - rx0, ry1 - ry0), (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)))
        step = (lambda k: (lambda s, total: on_step(k * total + s, n * total)))(i) if on_step else None
        out, _, mode, clip = _inpaint(pipe, sub, cond_mode, step, log, low_decode)
        img.paste(out, (rx0, ry0))
        # 每塊的大小不完全一樣（邊上的小一點），PyTorch 每塊留下不同大小的快取、越積越多：
        # 實測 2048² 九塊 reserved 衝到 20.7 GiB（16 GB 的卡，超出的被 Windows 挪到系統記憶體、變慢）。
        # 每塊做完就還給系統，峰值就停在一塊的量
        _release_cache()
    if img.size != (FW, FH):
        img = img.resize((FW, FH), Image.LANCZOS)
    return img, seed, mode, clip


OUTPAINT_OVERLAP = 32          # 遮罩往原圖多蓋幾像素：接縫落在重畫過的地方，才接得起來
OUTPAINT_MAX_PX = 1536 * 1024  # 畫布比這大就整張縮小來畫（SDXL、Anima 是 1MP 級，太大構圖會亂），貼回時放大


def _outpaint(pipe, req, cond_mode, on_step, log, low_decode):
    """擴圖：畫布往四邊加大，新的地方先用邊緣的顏色延伸填滿當起點，遮罩蓋住新的地方再往原圖多蓋一點，
    整張丟給局部重繪（模型看得到整張原圖，延伸出去的才接得上）。原圖除了接縫那一圈，逐位元不變。
    req["outpaint"]：{"left", "top", "right", "bottom"}（像素）；width、height 是加大後的畫布"""
    from PIL import Image, ImageDraw
    try:
        from core import inpaint as IP
    except ImportError:
        import inpaint as IP
    ext = req["outpaint"]
    L, T, Rt, B = (max(0, int(ext.get(k, 0))) for k in ("left", "top", "right", "bottom"))
    with Image.open(req["init_image"]) as im0:
        sw, sh = im0.size
    src = load_init_image(req["init_image"], sw, sh)          # 透明的地方鋪白底
    W, H = sw + L + Rt, sh + T + B
    canvas = Image.new("RGB", (W, H))
    canvas.paste(src, (L, T))
    new = Image.new("L", (W, H), 255)                         # 新加的地方
    ImageDraw.Draw(new).rectangle((L, T, L + sw - 1, T + sh - 1), fill=0)
    ov = OUTPAINT_OVERLAP
    hard = Image.new("L", (W, H), 255)                        # 要重畫的：新的地方＋往原圖多蓋一圈（只在有擴的那幾邊）
    ImageDraw.Draw(hard).rectangle((L + (ov if L else 0), T + (ov if T else 0),
                                    L + sw - 1 - (ov if Rt else 0), T + sh - 1 - (ov if B else 0)), fill=0)
    soft = IP.feather(hard, req.get("mask_blur", IP.MASK_BLUR))
    start = IP.fill(canvas, new)                              # 新的地方用周圍的顏色延伸開
    scale = min(1.0, (OUTPAINT_MAX_PX / (W * H)) ** 0.5)
    p = {"box": (0, 0, W, H), "work": (IP._snap(W * scale), IP._snap(H * scale))}
    log(i18n.tr("eng.p.outpaint", sw=sw, sh=sh, w=W, h=H, l=L, t=T, r=Rt, b=B) +
        (i18n.tr("eng.p.outpaintSmall", w=p['work'][0], h=p['work'][1]) if scale < 1 else ""))
    img, mask = IP.crop_inputs(start, soft, p)
    sub = {k: v for k, v in req.items() if k not in ("outpaint", "mask_image", "faces", "upscale", "controlnets")}
    sub.update(init_image=img, _mask=mask, width=p["work"][0], height=p["work"][1], **_repaint_units(req, img, mask))
    out, seed, mode, clip = _generate_impl(pipe, sub, cond_mode, on_step, log, low_decode)
    return IP.paste_back(canvas, out, soft, p), seed, mode, clip


def _repaint_units(req, img, mask):
    """擴圖、局部重繪用 union 模型的「重繪」：控制圖是要重畫的那塊塗黑的原圖（xinsir 的用法），
    模型照周圍的內容把黑的地方接起來，接縫比只靠局部重繪自然。req["cn_repaint"]：{model, config, name, scale}"""
    r = req.get("cn_repaint")
    if not r:
        return {"controlnets": []}
    from PIL import Image
    black = Image.new("RGB", img.size)
    ctrl = Image.composite(black, img.convert("RGB"), mask.point(lambda v: 255 if v > 0 else 0))
    return {"controlnets": [{"model": r["model"], "arch": "sdxl-union", "config": r["config"], "name": r.get("name", ""),
                             "type": "repaint", "prep": "none", "_image": ctrl, "scale": float(r.get("scale", 1.0)),
                             "start": 0.0, "end": 1.0}]}


def _release_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _inpaint(pipe, req, cond_mode, on_step, log, low_decode):
    """局部重繪：依範圍裁出要重畫的那塊（masked 會放大），交給引擎重畫，再貼回原圖。
    遮罩外的像素直接用原圖的，所以逐位元不變（見 core/inpaint.py）"""
    try:
        from core import inpaint as IP
    except ImportError:
        import inpaint as IP
    W, H = int(req["width"]), int(req["height"])
    full = load_init_image(req["init_image"], W, H)
    soft = IP.feather(IP.load_mask(req["mask_image"], (W, H)), req.get("mask_blur", IP.MASK_BLUR))
    p = IP.plan((W, H), soft, req.get("inpaint_area", "whole"), int(req.get("mask_padding", IP.MASK_PADDING)),
                max_up=float(req.get("max_upscale") or IP.MAX_UPSCALE))
    if p is None:
        raise ValueError(i18n.tr("srv.rf.maskEmpty"))
    # 塗色：顏色先蓋在底圖上當起點（貼回時遮罩外還是用原本的 full）
    sketch = IP.load_sketch(req["sketch_image"], (W, H)) if req.get("sketch_image") else None
    start = IP.apply_sketch(full, sketch, soft) if sketch else full
    img, mask = IP.crop_inputs(start, soft, p)
    if req.get("inpaint_fill") == "fill":
        # 移除東西：遮罩裡先用周圍的顏色糊掉，模型才不會照原本的畫回去；塗了顏色的地方不抹
        img = IP.fill(img, IP.crop_inputs(start, IP.unsketched(soft, sketch), p)[1] if sketch else mask)
    # 局部重繪：生成台的參考圖對不上裁出來的那塊，不用；union 的「重繪」照裁出來的那塊做
    sub = dict(req, init_image=img, _mask=mask, width=p["work"][0], height=p["work"][1], **_repaint_units(req, img, mask))
    if p["box"] != (0, 0, W, H):
        log(i18n.tr("eng.p.masked", w=p['box'][2] - p['box'][0], h=p['box'][3] - p['box'][1], ww=p['work'][0], wh=p['work'][1]))
    out, seed, mode, clip = _generate_impl(pipe, sub, cond_mode, on_step, log, low_decode)
    return IP.paste_back(full, out, soft, p), seed, mode, clip


def render_env(pipe, cfg):
    """存圖時記下會影響出圖、但不在請求裡的狀態（使用者筆記 #4：偶爾出一張不一樣的圖時可以直接比對）：
    Anima 的 shift、VAE 的精度與有沒有分塊解碼、VRAM 模式（出圖相同，除錯用）、PyTorch 與 diffusers 的版本"""
    env = {"vram": (cfg or {}).get("vram_mode") or ""}
    try:
        import torch
        import diffusers
        env.update(torch=torch.__version__, diffusers=diffusers.__version__)
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            if getattr(vae, "_coco_bf16", False):
                env["vae_dtype"] = "bf16"
            elif vae.dtype == torch.float16 and getattr(getattr(vae, "config", None), "force_upcast", False):
                env["vae_dtype"] = "fp32"            # diffusers 解碼時整顆升 32 位元
            else:
                env["vae_dtype"] = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(vae.dtype, str(vae.dtype))
        env["vae_tiling"] = bool(getattr(pipe, "_coco_tiled", False))
        if getattr(pipe, "_coco_weights", ""):
            env["weights"] = pipe._coco_weights        # 主模型的存放精度（Qwen Image 2.1 的 int8 檔）
        elif getattr(pipe, "_coco_fp8", False):
            env["weights"] = "fp8"                     # 主模型用 fp8 存放：出圖跟 bf16 不同
    except Exception:
        pass
    if is_flow(pipe) and (cfg or {}).get("shift") not in (None, ""):
        env["shift"] = float(cfg["shift"])
    # Anima 實際用的排程器跟選的名字不同時（例如選 LMS，Anima 沒有，實際是 FlowMatch Euler）
    impl = getattr(pipe, "_coco_sched_name", "") if is_flow(pipe) else ""
    if impl and impl.split(" + ")[0].removeprefix("FlowMatch ") != getattr(pipe, "_coco_sampler", ""):
        env["sampler_impl"] = impl
    return env


def save_image(img, out_dir: Path, req, seed, mode, model_name, loras,
               sampler=None, clip_skip=None, schedule=None, parts=None, env=None):
    """sampler 與 clip_skip 傳進來時會覆蓋 req 裡的值。

    中繼資料只寫**實際套用**的參數，不寫介面上的原始值，否則同一張圖
    照著參數重跑會對不起來。目前有兩個會不一致：

      sampler    Anima 實際跑的是 flow-match，跟介面選的取樣器名稱不同
      clip_skip  介面選比架構下限小的值會被夾住（見 _effective_clip_skip）
      schedule   取樣器不支援、或缺 scipy 的 Beta 會改用 Normal（見 apply_scheduler）
    """
    from PIL import PngImagePlugin
    try:
        from core import naming
    except ImportError:
        import naming
    anima = mode in ("anima", "zimage", "krea2", "qwen21", *FLUX2_ARCHS, *FLUX1_ARCHS)  # flow 架構：沒有 clip skip、排程照實際套用的
    smp = sampler or req.get("sampler", "")

    # Anima 只收實際套用的排程（core/anima._record，沒套用的是 Normal）；以前一律寫 Normal、排程併在取樣器名字裡
    sched = schedule if anima else (schedule or req.get("schedule", "Normal"))
    clip = "" if anima else (clip_skip if clip_skip is not None
                             else req.get("clip_skip", ""))

    # img2img 的底圖與強度。A1111 的欄位名是 Denoising strength，
    # 照寫其他工具才讀得懂；底圖路徑只放 coco JSON（圖庫用它找「同一張的其他版本」）
    init = req.get("init_image") or ""
    strength = (min(1.0, max(STRENGTH_MIN, float(req.get("strength", 0.5))))
                if init else None)
    i2i = f", Denoising strength: {strength:g}" if init else ""
    # 局部重繪：A1111 的寫法是 Mask blur、Inpaint area；遮罩檔的路徑只放 coco JSON
    mask = req.get("mask_image") or ""
    area = req.get("inpaint_area") or "whole"
    blur = int(req.get("mask_blur", 8) or 0)
    if mask:
        i2i += f", Mask blur: {blur}, Inpaint area: {'Only masked' if area == 'masked' else 'Whole picture'}"
    # 修臉：照 ADetailer 的欄位寫，Forge／civitai 看得懂。生成後自動修臉、不留原圖時沒有底圖，
    # 幅度另外放在 face_strength
    faces = req.get("faces") or []
    if faces:
        fs = req.get("face_strength", strength)
        i2i += (f", ADetailer model: {Path(str(req.get('face_model') or 'face')).name}, ADetailer confidence: 0.3, "
                f"ADetailer mask blur: {blur}, ADetailer denoising strength: {fs:g}, "
                f"ADetailer inpaint only masked: True, ADetailer inpaint padding: 32, ADetailer version: CoCo")
    # 亂數在 CPU 產生：照 A1111／Forge 寫 RNG: CPU（它們也是 GPU 時不寫）
    if rng_device(req) == "cpu":
        i2i += ", RNG: CPU"
    # NGMS 與加權方式（只有 SDXL／SD 1.5）：照 Forge 的寫法。NGMS 有開就寫；Emphasis 只在 No norm、提示詞裡有加權括號時寫
    ngms = 0.0 if anima else ngms_of(req)
    no_norm = (mode == "a1111" and req.get("emphasis") == "No norm"
               and any(c in f"{req['prompt']}{req.get('negative', '')}" for c in "(["))
    if ngms:
        i2i += f", NGMS: {ngms:g}"
    if no_norm:
        i2i += ", Emphasis: No norm"
    # 擴圖：A1111 沒有固定的欄位，照「Outpaint: 左 上 右 下」寫
    outp = req.get("outpaint") or {}
    if outp:
        i2i += (f", Mask blur: {blur}, Outpaint: left {int(outp.get('left', 0))} top {int(outp.get('top', 0))} "
                f"right {int(outp.get('right', 0))} bottom {int(outp.get('bottom', 0))}")
    # 高清化：照 A1111 的 hires fix 欄位寫
    hires = req.get("upscale") or {}
    if hires:
        i2i += (f", Hires upscale: {float(hires.get('scale', 2)):g}, "
                f"Hires upscaler: {Path(hires['model']).stem if hires.get('model') else 'Lanczos'}")
    # LoRA：Forge 的寫法是 Lora hashes: "名字: 前 12 碼, …"（server/lorahash.py 算好帶在工作裡）；
    # 送回生成區時，LoRA 改過名也能用它找到
    # 只記實際套用的（loras 是生效中的清單）：Z-Image 這類還不能套 LoRA 的，提示詞裡寫了也不算
    applied = {Path(str(p)).stem for p, _ in (loras or [])}
    lhs = {n: h for n, h in (req.get("lora_hashes") or {}).items() if n in applied}
    if lhs:
        i2i += f', Lora hashes: "{", ".join(f"{n}: {h}" for n, h in lhs.items())}"'
    # Embeddings：A1111 的寫法是 TI hashes: "名字: 檔案雜湊前 10 碼, …"（civitai 靠它認資源）
    tis = []
    for name, p in req.get("embeddings") or []:
        try:
            import embeddings
            tis.append(f"{name}: {embeddings.short_hash(p)}")
        except OSError:
            pass
    if tis:
        i2i += f', TI hashes: "{", ".join(tis)}"'
    # 換過的 VAE：A1111 的寫法是 VAE: 檔名（模型自帶的不寫）
    vae_used = (parts or {}).get("vae") or "builtin"
    if vae_used != "builtin":
        i2i += f", VAE: {'sdxl-vae-fp16-fix' if vae_used == 'fp16fix' else Path(vae_used).name}"
    # 換過的文字編碼器、VAE 精度與分塊、Anima 的 shift（render_env 收集的；影響出圖，比對時用）
    te_used = (parts or {}).get("te") or "builtin"
    if te_used != "builtin":
        i2i += f", Text encoder: {Path(te_used).name}"
    env = env or {}
    if env.get("shift") is not None:
        i2i += f", Shift: {env['shift']:g}"
    if env.get("vae_dtype"):
        i2i += f", VAE dtype: {env['vae_dtype']}"
    if env.get("weights"):
        i2i += f", Model dtype: {env['weights']}"
    if env.get("vae_tiling"):
        i2i += ", VAE tiling: on"
    # ControlNet：生成台的參考圖，或擴圖／局部重繪用的 union「重繪」（Anima 沒有套用，不寫）
    cn_units = [] if anima else [u for u in (req.get("controlnets") or []) if u.get("model")]
    if not anima and req.get("cn_repaint") and (outp or mask):
        r = req["cn_repaint"]
        cn_units.append({"model": r["model"], "name": r.get("name", ""), "type": "repaint", "prep": "none",
                         "scale": float(r.get("scale", 1.0)), "start": 0.0, "end": 1.0})
    if cn_units:
        # 先找工作行程的那份（worker 用 import controlnet_run），不然同一個模組會載成兩份、快取與倒數各管各的
        try:
            import controlnet_run as CR
        except ImportError:
            from core import controlnet_run as CR
        i2i += ", " + CR.params_text(cn_units)

    # 模型的雜湊（A1111／Forge 寫在 Model 前面）與 CoCo 的版本（A1111 的 Version 寫在最後）
    mh = f"Model hash: {req['model_hash']}, " if req.get("model_hash") else ""
    ver = f", Version: CoCo {req['coco_version']}" if req.get("coco_version") else ""
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters",
                  f"{req['prompt']}\nNegative prompt: {req.get('negative','')}\n"
                  f"Steps: {req['steps']}, Sampler: {smp}, "
                  f"Schedule type: {sched or 'Normal'}, "
                  f"CFG scale: {req['cfg']}, Seed: {seed}, "
                  f"Size: {req['width']}x{req['height']}, "
                  f"Clip skip: {clip}, {mh}Model: {model_name}{i2i}{ver}")
    extra = {"init": str(init), "strength": strength} if init else {}
    # 修圖（Qwen Image 2.1）：條件圖也記成底圖，精修頁的版本列與前後對照滑桿照樣找得到
    if req.get("edit_image"):
        extra.update(init=str(req["edit_image"]), edit=1)
        if req.get("edit_refs"):                       # 其他參考圖（圖2、圖3）
            extra["edit_refs"] = [str(p) for p in req["edit_refs"]]
    # 品質標籤（server/presets.py）：prompt／negative 是實際送出的完整版，
    # 另外記開關、自動加上的部分與使用者自己寫的，送回生成區時只帶使用者寫的
    for k in ("preset", "user_prompt", "user_negative"):
        if req.get(k) is not None:
            extra[k] = req[k]
    if mask:
        extra["inpaint"] = {"mask": str(mask), "area": area, "blur": blur,
                            **({"fill": "fill"} if req.get("inpaint_fill") == "fill" else {}),
                            **({"sketch": str(req["sketch_image"])} if req.get("sketch_image") else {})}
    if req.get("embeddings"):
        extra["embeddings"] = [name for name, _ in req["embeddings"]]
    # 這次實際用的 VAE 與文字編碼器（路徑、"fp16fix" 或 "builtin"）：圖庫存成 Recipe 時帶得到
    if parts:
        extra["parts"] = {"vae": parts.get("vae") or "builtin", "te": parts.get("te") or "builtin"}
    if rng_device(req) == "cpu":
        extra["rng"] = "cpu"
    if ngms:
        extra["ngms"] = ngms
    if no_norm:
        extra["emphasis"] = "No norm"
    if outp:
        extra["outpaint"] = {k: int(outp.get(k, 0)) for k in ("left", "top", "right", "bottom")}
    if cn_units:
        extra["controlnet"] = [{"model": Path(u["model"]).name, "name": u.get("name", ""), "type": u.get("type", ""),
                                "prep": u.get("prep", "none"), "scale": float(u.get("scale", 1)), "start": float(u.get("start", 0)),
                                "end": float(u.get("end", 1)), **({"image": str(u["image"])} if u.get("image") else {}),
                                **({"resize": u["resize"]} if u.get("resize") else {})} for u in cn_units]
    if hires:
        extra["upscale"] = {"scale": float(hires.get("scale", 2)), "detail": True,
                            "model": Path(hires["model"]).name if hires.get("model") else "Lanczos"}
    if faces:
        extra["faces"] = {"model": Path(str(req.get("face_model") or "")).name, "blur": blur,
                          **({"auto": True} if req.get("auto_face") else {}),
                          "items": [{"box": f["box"], **({"prompt": f["prompt"]} if f.get("prompt") else {})} for f in faces]}
    info.add_text("coco", json.dumps({
        "prompt": req["prompt"], "negative": req.get("negative", ""),
        "steps": req["steps"], "cfg": req["cfg"], "seed": seed,
        "sampler": smp, "schedule": sched or None,
        "model": model_name,
        "clip_skip": clip or None, "conditioning": mode,
        # 影響出圖的狀態（render_env）、模型雜湊、CoCo 版本：偶爾出一張不一樣的圖時比對用
        **({"env": env} if env else {}),
        **({"model_hash": req["model_hash"]} if req.get("model_hash") else {}),
        **({"version": req["coco_version"]} if req.get("coco_version") else {}),
        "loras": [{"file": Path(p).name, "weight": w, **({"hash": lhs[Path(p).stem]} if Path(p).stem in lhs else {})}
                  for p, w in (loras or [])],
        **extra,
    }, ensure_ascii=False))
    # 檔名照 ⚙ 設定 → 生成 → 存檔名稱（req["name_pattern"]），預設 outputs/2026-08-07/000001-24680-896x1344.png：
    # 依日期分資料夾，序號取同資料夾裡最大的再加一，手動刪掉幾張也不會撞名（core/naming.py）。
    # target 會先建一個空檔佔住名字，所以放在存圖前一刻，存失敗就刪掉
    path = naming.target(out_dir, req.get("name_pattern") or naming.DEFAULT,
                         naming.values(req, seed, model_name, loras, smp))
    try:
        img.save(path, pnginfo=info)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path
