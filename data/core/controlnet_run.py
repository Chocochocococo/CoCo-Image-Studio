# -*- coding: utf-8 -*-
"""ControlNet 的模型載入與管線組裝（pipeline._generate_impl 用；只有 SDXL）。

req["controlnets"]：[{model, arch: sdxl / sdxl-union, config, name, type, prep, prep_files, image（檔案路徑）或 _image（PIL）,
                      scale, start, end, low, high, resize: crop / fit}]
  一般的 SDXL ControlNet 用 ControlNetModel（結構照 SDXL 的標準，檔頭的權重名字已經在 server/controlnet.py 對過）；
  union／promax 用 ControlNetUnionModel 加旁邊的 config_promax.json。兩種不能混在同一次生成（diffusers 的管線不支援）。

模型只在生成真的用到參考圖時才載入（沒開就完全不碰）。權重直接對應到 .safetensors 檔案（不另外複製一份，
2.5 GB 的模型在系統記憶體只多幾 MB），生成時才搬上顯示卡（同時轉成生成用的精度，BF16 的檔案也不會在
系統記憶體多一份）；生成完不搬回來，而是重新對應回檔案（0.04 秒），顯示卡馬上空出來。
閒置 IDLE 秒（生成、看處理結果都算用到）就連同抽姿勢、算深度的模型一起放掉；生成途中不會放。
"""

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

import i18n

KEEP = 2
_CACHE = OrderedDict()          # (路徑, 修改時間) → 模型（CPU）
IDLE = 120                      # 幾秒沒用就卸載（重新載入一個約 2～3 秒）
_LOCK = threading.RLock()
_LAST = [0.0]
_TIMER = [None]
_BUSY = [0]                     # 正在生成的（用著的模型不能放）
on_release = None               # 卸載時通知（工作行程記一行紀錄）


def touch():
    """用過一次：重新倒數，IDLE 秒內沒再用就卸載"""
    with _LOCK:
        _LAST[0] = time.monotonic()
        if _TIMER[0] is not None:
            _TIMER[0].cancel()
        t = threading.Timer(IDLE + 0.05, _idle_release)
        t.daemon = True
        _TIMER[0] = t
        t.start()


def _loaded():
    # 先找工作行程的那份（worker 用 import controlnet_prep），不然同一個模組會載成兩份、快取與倒數各管各的
    try:
        import controlnet_prep as CP
    except ImportError:
        from core import controlnet_prep as CP
    return bool(_CACHE) or CP.loaded()


def _idle_release():
    with _LOCK:
        if _BUSY[0] or time.monotonic() - _LAST[0] < IDLE or not _loaded():
            return
        release()
        if on_release:
            on_release()

# SDXL ControlNet 的標準結構（diffusers 格式；跟 SDXL 的 UNet 前半一樣）
SDXL_CONFIG = {
    "act_fn": "silu", "addition_embed_type": "text_time", "addition_embed_type_num_heads": 64, "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20], "block_out_channels": [320, 640, 1280], "class_embed_type": None,
    "conditioning_channels": 3, "conditioning_embedding_out_channels": [16, 32, 96, 256],
    "controlnet_conditioning_channel_order": "rgb", "cross_attention_dim": 2048,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], "downsample_padding": 1,
    "encoder_hid_dim": None, "encoder_hid_dim_type": None, "flip_sin_to_cos": True, "freq_shift": 0,
    "global_pool_conditions": False, "in_channels": 4, "layers_per_block": 2, "mid_block_scale_factor": 1, "norm_eps": 1e-05,
    "norm_num_groups": 32, "num_attention_heads": None, "num_class_embeds": None, "only_cross_attention": False,
    "projection_class_embeddings_input_dim": 2816, "resnet_time_scale_shift": "default", "transformer_layers_per_block": [1, 2, 10],
    "upcast_attention": None, "use_linear_projection": True,
}
UNION_MODE = {"pose": 0, "depth": 1, "softedge": 2, "canny": 3, "lineart": 3, "tile": 6, "repaint": 7}
# 存圖參數裡寫的前處理名字（跟 A1111 的 ControlNet 擴充寫法接近）
MODULE_NAME = {"dwpose": "dw_openpose_full", "depth": "depth_anything_v2", "canny": "canny", "lineart": "lineart_xdog",
               "softedge": "softedge_coco", "tile": "tile", "none": "none", "repaint": "inpaint_only"}


def _layers(keys):
    """權重名字 → 每一層有幾個 transformer block（有的「精簡版」ControlNet 比標準的少）"""
    out = []
    for i in range(3):
        n = 0
        while f"down_blocks.{i}.attentions.0.transformer_blocks.{n}.attn1.to_q.weight" in keys:
            n += 1
        out.append(max(1, n))
    return out


def load(unit, dtype, log=print):
    """ControlNet 模型（權重對應到檔案、照檔案的精度；dtype 是上顯示卡時要轉成的，這裡不用）；同一個檔案載過就直接用"""
    import torch
    from safetensors.torch import load_file
    path = Path(unit["model"])
    key = (str(path), path.stat().st_mtime)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    log(i18n.tr("eng.cn.load", name=path.name))
    sd = load_file(str(path))                   # 對應到檔案，不讀進記憶體
    if unit.get("arch") == "sdxl-union":
        from diffusers import ControlNetUnionModel
        cfg = json.loads(Path(unit["config"]).read_text(encoding="utf-8"))
        cfg.pop("_class_name", None)
        with torch.device("meta"):
            m = ControlNetUnionModel.from_config(cfg)
    else:
        from diffusers import ControlNetModel
        cfg = dict(SDXL_CONFIG, transformer_layers_per_block=_layers(sd))
        hint = sd.get("controlnet_cond_embedding.conv_in.weight")
        if hint is not None:
            cfg["conditioning_channels"] = int(hint.shape[1])
        with torch.device("meta"):
            m = ControlNetModel.from_config(cfg)
    m.load_state_dict(sd, strict=True, assign=True)    # 照檔案的精度，上顯示卡時才轉
    m.eval().requires_grad_(False)
    m._coco_path = str(path)
    _CACHE[key] = m
    while len(_CACHE) > KEEP:
        _CACHE.popitem(last=False)
    return m


def _remap(m):
    """生成完：權重重新對應回檔案（不搬回系統記憶體、不複製），顯示卡上那份就放掉了"""
    from safetensors.torch import load_file
    try:
        m.load_state_dict(load_file(m._coco_path), strict=True, assign=True)
    except Exception:
        # 檔案被搬走了之類的：這個模型不能再用，從快取拿掉（下次重新載入），顯示卡上那份照樣放掉
        for k in [k for k, v in _CACHE.items() if v is m]:
            del _CACHE[k]
        m.to("meta")


def fit(img, w, h, mode="crop"):
    """參考圖變成生成的大小：crop＝等比例放大填滿、裁掉多的（預設）；fit＝整張放進去、旁邊補黑"""
    from PIL import Image
    img = img.convert("RGB")
    if img.size == (w, h):
        return img
    if mode == "fit":
        k = min(w / img.width, h / img.height)
        sm = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
        out = Image.new("RGB", (w, h))
        out.paste(sm, ((w - sm.width) // 2, (h - sm.height) // 2))
        return out
    k = max(w / img.width, h / img.height)
    big = img.resize((max(w, round(img.width * k)), max(h, round(img.height * k))), Image.LANCZOS)
    x, y = (big.width - w) // 2, (big.height - h) // 2
    return big.crop((x, y, x + w, y + h))


def control_image(unit, w, h, log=print):
    """一張參考圖的控制圖（生成的大小）：先照比例縮成 w×h，再做前處理"""
    from PIL import Image
    # 先找工作行程的那份（worker 用 import controlnet_prep），不然同一個模組會載成兩份、快取與倒數各管各的
    try:
        import controlnet_prep as CP
    except ImportError:
        from core import controlnet_prep as CP
    if unit.get("_image") is not None:
        src = unit["_image"]
    else:
        with Image.open(unit["image"]) as im:
            src = im.convert("RGB")
    src = fit(src, w, h, unit.get("resize", "crop"))
    out, info = CP.run(src, unit.get("prep", "none"), unit.get("prep_files") or {}, unit.get("low", 100), unit.get("high", 200))
    if unit.get("prep") == "none" and unit.get("type") in ("lineart", "canny", "softedge"):
        out = ink_on_black(out)
    if unit.get("prep") == "dwpose":
        log(i18n.tr("eng.cn.pose", n=info.get('people', 0)) + (i18n.tr("eng.cn.noPeople") if not info.get("people") else ""))
    return out


def ink_on_black(img):
    """自己畫的線稿多半是黑線白底，ControlNet 要的是白線黑底：底色偏亮就反過來"""
    from PIL import ImageOps, ImageStat
    g = img.convert("L")
    return ImageOps.invert(img.convert("RGB")) if ImageStat.Stat(g).mean[0] > 127 else img


def build(pipe, units, w, h, init=False, mask=False, log=print):
    """組 ControlNet 管線：回傳 (管線, 要加進呼叫的參數, 用完要呼叫的 cleanup)"""
    import torch
    import diffusers as D
    from diffusers.models.controlnets import MultiControlNetModel, MultiControlNetUnionModel
    union = [u.get("arch") == "sdxl-union" for u in units]
    if any(union) and not all(union):
        raise ValueError(i18n.tr("eng.cn.mixed"))
    union = all(union)
    dtype = getattr(pipe.unet, "dtype", torch.float16)
    used, done = [], [False]

    def cleanup():
        """生成完（或中途出錯）：模型搬回系統記憶體、快取還給系統、重新倒數"""
        if done[0]:
            return
        done[0] = True
        for m in used:
            _remap(m)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with _LOCK:
            _BUSY[0] -= 1
            touch()

    with _LOCK:
        _BUSY[0] += 1                   # 生成途中倒數到了也不卸載
        touch()
    try:
        models = [load(u, dtype, log) for u in units]
        images = [control_image(u, w, h, log) for u in units]
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        used += list({id(m): m for m in models}.values())
        for m in used:
            # 繞過 diffusers 的 to()（它會為了 fp32 模組的提醒印一行字）；搬上去的同時轉精度
            torch.nn.Module.to(m, device=dev, dtype=dtype)
        if union:
            cn = models[0] if len(models) == 1 else MultiControlNetUnionModel(models)
            cls = (D.StableDiffusionXLControlNetUnionInpaintPipeline if mask else
                   D.StableDiffusionXLControlNetUnionImg2ImgPipeline if init else D.StableDiffusionXLControlNetUnionPipeline)
        else:
            cn = models[0] if len(models) == 1 else MultiControlNetModel(models)
            cls = (D.StableDiffusionXLControlNetInpaintPipeline if mask else
                   D.StableDiffusionXLControlNetImg2ImgPipeline if init else D.StableDiffusionXLControlNetPipeline)
        # torch_dtype=None：不要讓 from_pipe 把共用的 UNet 轉成 fp32（見 _generate_impl 的 img2img）
        run = cls.from_pipe(pipe, controlnet=cn, torch_dtype=None)
    except Exception:
        cleanup()
        raise
    try:
        run.set_progress_bar_config(disable=True)
    except Exception:
        pass
    one = len(units) == 1
    pick = lambda k, d: (float(units[0].get(k, d)) if one else [float(u.get(k, d)) for u in units])
    kw = {"controlnet_conditioning_scale": pick("scale", 1.0),
          "control_guidance_start": pick("start", 0.0), "control_guidance_end": pick("end", 1.0)}
    imgs = images[0] if one and not union else images
    if union:
        kw["control_image"] = images
        kw["control_mode"] = [UNION_MODE[u["type"]] for u in units]
    elif init or mask:
        kw["control_image"] = imgs
    else:
        kw["image"] = imgs
    names = "、".join(f"{u.get('name') or Path(u['model']).stem}（{u['type']}）" for u in units)
    log(f"ControlNet：{names}")
    return run, kw, cleanup


def params_text(units):
    """存圖參數（A1111 的 ControlNet 擴充的寫法）"""
    out = []
    for i, u in enumerate(units):
        mod = MODULE_NAME.get("repaint" if u.get("type") == "repaint" else u.get("prep", "none"), u.get("prep", "none"))
        out.append(f'ControlNet {i}: "Module: {mod}, Model: {u.get("name") or Path(u["model"]).stem}, '
                   f'Weight: {float(u.get("scale", 1)):g}, Guidance Start: {float(u.get("start", 0)):g}, '
                   f'Guidance End: {float(u.get("end", 1)):g}"')
    return ", ".join(out)


def release():
    """載過的 ControlNet 與前處理的模型都放掉，記憶體還給系統"""
    # 先找工作行程的那份（worker 用 import controlnet_prep），不然同一個模組會載成兩份、快取與倒數各管各的
    try:
        import controlnet_prep as CP
    except ImportError:
        from core import controlnet_prep as CP
    with _LOCK:
        _CACHE.clear()
        CP.release()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
