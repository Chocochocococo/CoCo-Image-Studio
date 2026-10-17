# -*- coding: utf-8 -*-
"""
sdxl_train.py —— 訓練 SDXL 的 LoRA（訓練空間的 02 訓練）。另開一個行程跑：python core/sdxl_train.py 設定.json

使用者在 OneTrainer 訓練 Illustrious 系的 SDXL LoRA，預設值照他常用的三份設定；做法照 OneTrainer 的行為自己寫
（OneTrainer 是 AGPL，只讀它的程式了解做法，不抄程式碼）：

- **LoRA 掛哪些層**：UNet 名稱含 "attentions" 的 Linear（OneTrainer 的 attn-mlp：proj_in／out、attn1、attn2、ff），
  兩個文字編碼器的所有 Linear（含 TE2 的 text_projection）。down 用 kaiming uniform（a=√5）、up 全 0，輸出 × alpha／rank。
- **存檔**：kohya 式的名稱（lora_unet_…、lora_te1_text_model_…、lora_te2_…，各有 alpha、lora_down、lora_up，float32），
  Forge、ComfyUI、CoCo 都讀得到。新版 transformers 的 CLIPTextModel 少了外面那層 text_model，存的時候補回去。
- **資料**：資料集資料夾裡的圖與同名 .txt，每個 epoch 每張重複 repeats 次、打亂順序。長寬比分桶：1:1、1:1.25…1:4 與反過來，
  面積對到 解析度²、邊長取 64 的倍數，選比例最接近的；縮放（bilinear＋antialias）蓋滿那一格再裁切（crop jitter：位置隨機）。
  latent 先算好（VAE float32、取平均值），一張一次。
- **每一步**：文字截到 77 個 token、兩個 CLIP 取倒數第二層（不加最後的 LayerNorm）、pooled 取 TE2 的投影；
  時間步 0～999 均勻抽、一般高斯雜訊（DDPM、scaled_linear 0.00085～0.012）、預測雜訊（epsilon）、MSE；
  SDXL 的尺寸條件是（原圖高、寬、裁切的上、左、裁切後高、寬）。
- **優化**：Prodigy（vendor/prodigyopt，跟 OneTrainer 同一版）學習率 1.0、betas (0.9, 0.99)、weight decay 0.01、
  decouple、bias correction、safeguard warmup、d0 1e-6、slice_p 11；cosine 降到 0、沒有 warmup；梯度裁切 1.0。
  UNet 與文字編碼器 fp16 不動、LoRA 權重 float32，算的時候 fp16 autocast＋GradScaler；UNet 開梯度檢查點。

跟 server 用 JSON lines 溝通（stdout）：log、progress（步、epoch、loss、Prodigy 的學習率、每步秒數）、saved（存好的檔）、
done、error。設定裡的 stop_file 出現了就在這一步做完後存檔、結束。
"""

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import i18n

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SKIP_DIRS = ("_tags_only", "_deleted")
ASPECTS = [(1.0, 1.0), (1.0, 1.25), (1.0, 1.5), (1.0, 1.75), (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (1.0, 3.5), (1.0, 4.0)]
QUANT = 64

# 預設值：使用者 OneTrainer 的設定（三份共同的部分）
DEFAULTS = {
    "resolution": 1024, "rank": 16, "alpha": 8.0, "epochs": 10, "repeats": 4, "batch": 1,
    "train_te": True, "layer_filter": "attentions", "flip": False, "crop_jitter": True,
    "shuffle_tags": False, "keep_tags": 1, "save_every": 1, "save_skip": 0, "seed": 42,
    "lr": 1.0, "grad_ckpt": True,
}


# ---------------------------------------------------------------- 資料

def list_images(root):
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return sorted(out, key=lambda p: [s.lower() for s in p.relative_to(root).parts])


def read_caption(img):
    try:
        return Path(img).with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def buckets(resolution):
    """這個解析度的所有分桶 [(高, 寬)]：面積對到 解析度²、邊長取 64 的倍數（OneTrainer 的 AspectBucketing）"""
    out = []
    for h, w in ASPECTS + [(w, h) for h, w in ASPECTS]:
        s = math.sqrt(h * w)
        out.append((round(h / s * resolution / QUANT) * QUANT, round(w / s * resolution / QUANT) * QUANT))
    return sorted(set(out))


def bucket_for(h, w, resolution):
    """原圖 (高, 寬) → 比例最接近的分桶"""
    bs = buckets(resolution)
    return min(bs, key=lambda b: abs(b[0] / b[1] - h / w))


def scale_crop(h, w, bucket, rng, jitter):
    """縮放成蓋滿分桶的大小 → (縮放後高, 寬, 裁切的上, 左)。比例較高的圖照寬度縮、較寬的照高度縮"""
    bh, bw = bucket
    if h / w > bh / bw:
        sh, sw = round(h * bw / w), bw
    else:
        sh, sw = bh, round(w * bh / h)
    dy, dx = sh - bh, sw - bw
    if jitter:
        return sh, sw, rng.randint(0, dy), rng.randint(0, dx)
    return sh, sw, dy // 2, dx // 2


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def shuffle_caption(text, keep, rng):
    """打亂標籤順序，前 keep 個（觸發詞）不動；描述那段（像句子的）接在最後不打亂"""
    parts = [p.strip() for p in text.split(",")]
    tags, desc = [], []
    for i, p in enumerate(parts):
        if p and (len(p.split()) >= 7 or re.search(r"[.!?](\s|$)", p)):
            desc = parts[i:]
            break
        if p:
            tags.append(p)
    head, rest = tags[:keep], tags[keep:]
    rng.shuffle(rest)
    return ", ".join(head + rest + [d for d in desc if d])


# ---------------------------------------------------------------- LoRA

def _lora_class():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class LoRALinear(nn.Module):
        """原本的 Linear ＋ up(down(x)) × alpha／rank × mult。原本的權重不動。
        mult 是強度（一般訓練一直是 1；Slider 訓練在 +1、−1、0 之間切換，core/slider_train.py）"""

        def __init__(self, orig, rank, alpha):
            super().__init__()
            self.orig = orig
            self.rank, self.alpha, self.mult = rank, float(alpha), 1.0
            dev = orig.weight.device
            self.lora_down = nn.Linear(orig.in_features, rank, bias=False, device=dev, dtype=torch.float32)
            self.lora_up = nn.Linear(rank, orig.out_features, bias=False, device=dev, dtype=torch.float32)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x):
            d = self.lora_down.weight.dtype
            if x.dtype != d and not torch.is_autocast_enabled("cuda"):
                # 試畫（沒有 autocast）：LoRA 的權重轉成模型的型別來算。權重小、中間結果大，轉權重比較快：
                # 1024²、28 步每張 6.2 秒，把輸入轉成 float32 是 8.0 秒（不掛 LoRA 4.8 秒；出圖平均差 1.8/255）。
                # 訓練時有 autocast，走下面那行不變
                return self.orig(x) + F.linear(F.linear(x, self.lora_down.weight.to(x.dtype)),
                                               self.lora_up.weight.to(x.dtype)) * (self.alpha / self.rank * self.mult)
            return self.orig(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank * self.mult)

    return LoRALinear


def inject(model, rank, alpha, name_filter=None):
    """把 model 裡的 Linear 換成 LoRALinear → {模組名稱: LoRALinear}。
    name_filter：名稱要含這個字才換；也可以是函式（名稱 → 要不要換，core/anima_train.py 用）"""
    from torch import nn
    LoRALinear = _lora_class()
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and (not name_filter or (name_filter(n) if callable(name_filter) else name_filter in n))]
    out = {}
    for name, mod in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lora = LoRALinear(mod, rank, alpha)
        setattr(parent, attr, lora)
        out[name] = lora
    return out


def kohya_key(prefix, name):
    """模組名稱 → kohya 式的鍵：lora_unet_down_blocks_1_attentions_0_proj_in。
    文字編碼器：新版 transformers 的 CLIPTextModel 少了 text_model 那層，補回去（text_projection 不用）"""
    if prefix in ("lora_te1", "lora_te2") and not name.startswith("text_model.") and not name.startswith("text_projection"):
        name = "text_model." + name
    return f"{prefix}_{name.replace('.', '_')}"


def state_dict(groups, key=None, dtype=None):
    """[(前綴, {名稱: LoRALinear})] → 存檔用的 dict（每層有 alpha）。
    key：(前綴, 名稱) → 鍵名，預設 kohya_key；dtype：預設 float32（OneTrainer），Anima 是 bf16（sd-scripts 的 save_precision）"""
    import torch
    key = key or kohya_key
    dt = dtype or torch.float32
    out = {}
    for prefix, mods in groups:
        for name, m in mods.items():
            k = key(prefix, name)
            out[k + ".lora_down.weight"] = m.lora_down.weight.detach().to(dt).cpu().contiguous()
            out[k + ".lora_up.weight"] = m.lora_up.weight.detach().to(dt).cpu().contiguous()
            out[k + ".alpha"] = torch.tensor(float(m.alpha), dtype=dt)
    return out


def save_lora(path, groups, meta, key=None, dtype=None):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(state_dict(groups, key, dtype), str(tmp), metadata={k: str(v) for k, v in meta.items()})
    os.replace(tmp, path)


# ---------------------------------------------------------------- 排程

def cosine(step, total):
    """OneTrainer 的 COSINE（沒有 warmup、降到 0、一個週期）：0.5 × (1 + cos(π × step／total))"""
    return 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(1, total)))


def plan_steps(n_images, repeats, epochs, batch):
    per_epoch = math.ceil(n_images * repeats / batch)
    return per_epoch, per_epoch * epochs


# ---------------------------------------------------------------- 訓練

def _out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg, kind="info"):
    _out({"type": "log", "msg": str(msg), "kind": kind})


def stop_requested(cfg):
    """server/train.py 要停時建 stop_file。不用 stdin：Windows 上另一個執行緒卡在讀管道時，
    主執行緒載入 numpy 的 DLL 會一起卡住（實測卡死在 import torch，從 CoCo 開訓練一直停在載入模型）"""
    f = cfg.get("stop_file")
    return bool(f) and os.path.exists(f)


def encode_text(pipe, captions, device):
    """兩個 CLIP：截到 77、取倒數第二層、不加最後的 LayerNorm；pooled 取 TE2 的投影"""
    import torch
    t1 = pipe.tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    t2 = pipe.tokenizer_2(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    o1 = pipe.text_encoder(t1, output_hidden_states=True)
    o2 = pipe.text_encoder_2(t2, output_hidden_states=True)
    hidden = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
    return hidden, o2.text_embeds


def cache_latents(pipe, items, resolution, jitter, flip, rng, device):
    """每張圖：選分桶、縮放裁切、VAE（float32）編碼取平均 → item 補上 latent 與尺寸條件"""
    import torch
    import torchvision.transforms.functional as TF
    vae = pipe.vae
    vae.to(device, torch.float32)
    sf = vae.config.scaling_factor
    for i, it in enumerate(items):
        im = load_rgb(it["path"])
        w, h = im.size
        b = bucket_for(h, w, resolution)
        sh, sw, top, left = scale_crop(h, w, b, rng, jitter)
        x = TF.to_tensor(im)
        x = TF.resize(x, [sh, sw], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x = x[:, top:top + b[0], left:left + b[1]]
        if flip and rng.random() < 0.5:
            x = x.flip(-1)
        x = (x * 2 - 1)[None].to(device, torch.float32)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mean * sf
        it.update(latent=lat[0].to(torch.float16).cpu(), time_ids=[h, w, top, left, b[0], b[1]], bucket=b)
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            _out({"type": "cache", "done": i + 1, "total": len(items)})
    vae.to("cpu")
    torch.cuda.empty_cache()


def save_epochs(epochs, save_skip, save_every=1):
    """哪幾個 epoch 存檔：最後一個一定存（名字本身），其他是 save_skip 之後每 save_every 個"""
    return [e for e in range(1, epochs + 1) if e == epochs or (e > save_skip and e % max(1, save_every) == 0)]


def sample_tag(epoch):
    """試畫的檔名用：開始前（不掛 LoRA）是 base，之後是 e05 這樣"""
    return "base" if not epoch else f"e{int(epoch):02d}"


def make_sampler(pipe, c, models, device, prepare=None, before=None, after=None):
    """試畫：用生成引擎的 generate／save_image 畫 c["samples"] 的每一則（server/train.py 照平常生圖的設定組好），
    LoRA 已經掛在模型上，不用重新載入。沒有要試畫回傳 None。
    prepare：第一次畫之前做一次（預設是 SDXL 的 VAE）；before／after：每次畫的前後（Anima 搬文字編碼器）。
    回傳 sample(epoch, lora_path, mults=None, tag=None, set_mult=None)：畫完每張回報 {"type": "sample", path, index, epoch, lora, mult}。
    mults：Slider 每則提示詞畫好幾個強度（set_mult 切換 LoRA 的強度，0 就是不掛）；tag：檔名用的標記（預設 e05 這樣）"""
    jobs = [j for j in (c.get("samples") or []) if isinstance(j, dict) and j.get("prompt")]
    if not jobs:
        return None
    import torch
    import pipeline as P
    P.apply_scheduler(pipe, jobs[0].get("sampler") or "Euler a", jobs[0].get("schedule") or "Normal", lambda *a: None,
                      shift=jobs[0].get("shift"))
    pipe._coco_vram = "normal"            # 不要把文字編碼器搬回 CPU（訓練還要用）
    model_name = Path(c["model"]).name
    stem = re.sub(r"[{}]", "_", str(c["name"]))      # 檔名格式裡 {…} 是變數
    ready = []

    def plog(msg):
        if str(msg).startswith("[!]"):
            log(str(msg)[3:].strip(), "warn")

    def sample(epoch, lora_path, mults=None, tag=None, set_mult=None):
        t0 = time.time()
        _out({"type": "sampling", "epoch": int(epoch)})
        if not ready:
            if prepare:
                prepare()
            else:
                # VAE 算完 latent 留在 CPU（float32）；試畫照生成引擎的做法用 bf16 解碼（原版 SDXL VAE 在 fp16 會溢位出黑圖）
                pipe.vae.to(device, torch.float16)
                P._vae_bf16(pipe)
            ready.append(True)
        was = [(m, m.training) for m in models]
        for m, _ in was:
            m.eval()
        try:
            if before:
                before()
            for i, job in enumerate(jobs):
                for m in (mults or [1.0]):
                    w = f"-w{float(m):+g}" if mults else ""          # Slider：-w-2、-w+1 這樣
                    req = {**job, "name_pattern": f"{job.get('folder') or 'train'}/{stem}-{tag or sample_tag(epoch)}-{i + 1}{w}"}
                    if set_mult:
                        set_mult(float(m))
                    img, seed, mode, clip = P.generate(pipe, req, log=plog)
                    loras = [(str(lora_path), float(m))] if lora_path and m else []
                    path = P.save_image(img, Path(job["out_dir"]), req, seed, mode, model_name, loras,
                                        clip_skip=clip, parts=job.get("parts"), env=P.render_env(pipe, {"shift": job.get("shift")}))
                    _out({"type": "sample", "path": str(path), "index": i, "epoch": int(epoch), "mult": float(m),
                          "lora": Path(lora_path).name if lora_path and m else ""})
        except Exception as e:            # 試畫失敗不影響訓練
            log(i18n.tr("eng.t.sampleFail", e=f"{type(e).__name__}: {str(e)[:200]}"), "warn")
            return time.time() - t0
        finally:
            if set_mult:
                set_mult(1.0)
            if after:
                after()
            for m, w in was:
                m.train(w)
            torch.cuda.empty_cache()
        took = time.time() - t0
        log(i18n.tr("eng.t.sampled", n=len(jobs) * len(mults or [1]), s=f"{took:.0f}"))
        return took

    return sample


def train(cfg):
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vendor.prodigyopt import Prodigy
    import resume as R

    c = {**DEFAULTS, **cfg}
    device = "cuda"
    rng = random.Random(int(c["seed"]))
    torch.manual_seed(int(c["seed"]))

    imgs = list_images(c["dataset"])
    if not imgs:
        raise ValueError(i18n.tr("eng.t.noImages"))
    items = [{"path": str(p), "caption": read_caption(p)} for p in imgs]
    no_cap = sum(1 for it in items if not it["caption"])
    if no_cap:
        log(i18n.tr("eng.t.noCaption", n=no_cap), "warn")
    per_epoch, total = plan_steps(len(items), int(c["repeats"]), int(c["epochs"]), int(c["batch"]))
    log(i18n.tr("eng.t.plan", n=len(items), repeats=c['repeats'], epochs=c['epochs'], batch=c['batch'], total=total))

    t0 = time.time()
    log(i18n.tr("eng.t.loading"))
    pipe = StableDiffusionXLPipeline.from_single_file(c["model"], torch_dtype=torch.float16, use_safetensors=True)
    if pipe.scheduler.config.get("prediction_type") == "v_prediction":
        raise ValueError(i18n.tr("eng.t.vpred"))
    log(i18n.tr("eng.t.loaded", s=f"{time.time() - t0:.0f}"))

    log(i18n.tr("eng.t.latents"))
    cache_latents(pipe, items, int(c["resolution"]), bool(c["crop_jitter"]), bool(c["flip"]), rng, device)

    unet, te1, te2 = pipe.unet, pipe.text_encoder, pipe.text_encoder_2
    for m in (unet, te1, te2):
        m.requires_grad_(False)
        m.to(device)
    rank, alpha = int(c["rank"]), float(c["alpha"])
    groups = [("lora_unet", inject(unet, rank, alpha, c["layer_filter"] or None))]
    if c["train_te"]:
        groups = [("lora_te1", inject(te1, rank, alpha)), ("lora_te2", inject(te2, rank, alpha))] + groups
    if c["grad_ckpt"]:
        unet.enable_gradient_checkpointing()
    unet.train()
    te1.train(bool(c["train_te"]))
    te2.train(bool(c["train_te"]))
    log(i18n.tr("eng.t.loraLayers", groups=i18n.join(i18n.tr("eng.t.loraGroup", p=p, n=len(m)) for p, m in groups)))

    params = [{"params": [p for m in mods.values() for p in (m.lora_down.weight, m.lora_up.weight)], "lr": float(c["lr"])}
              for _, mods in groups]
    for g in params:
        for p in g["params"]:
            p.requires_grad_(True)
    opt = Prodigy(params, lr=float(c["lr"]), betas=(0.9, 0.99), beta3=None, eps=1e-8, weight_decay=0.01, decouple=True,
                  use_bias_correction=True, safeguard_warmup=True, d0=1e-6, d_coef=1.0, growth_rate=float("inf"), slice_p=11)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total))
    scaler = torch.amp.GradScaler("cuda")
    noise_sched = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    alphas = noise_sched.alphas_cumprod.to(device)
    all_params = [p for g in params for p in g["params"]]

    meta = {"ss_base_model_version": "sdxl_base_v1-0", "ss_network_module": "networks.lora", "ss_network_dim": rank,
            "ss_network_alpha": alpha, "ss_resolution": c["resolution"], "ss_num_train_images": len(items),
            "ss_num_epochs": c["epochs"], "ss_output_name": c["name"], "coco_trainer": "CoCo Image Studio",
            "modelspec.architecture": "stable-diffusion-xl-v1-base/lora", "modelspec.title": c["name"]}
    out_dir = Path(c["out_dir"])
    gen = torch.Generator(device=device)
    sampler = make_sampler(pipe, c, [unet, te1, te2], device)
    saves = save_epochs(int(c["epochs"]), int(c["save_skip"]), int(c["save_every"]))
    per_sample = 0.0
    step, ema_loss, t_start, spent = 0, None, time.time(), 0.0
    start_epoch, batches, pos = 1, None, 0
    if c.get("resume"):
        st = R.load(c["resume"])
        R.load_lora(groups, st["lora"])
        if R.check_dataset(st["data"], items):
            log(i18n.tr("eng.t.changed"), "warn")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        R.set_rng(rng, st["rng"])
        step, ema_loss, start_epoch, batches, pos = st["step"], st["ema"], st["epoch"], st["batches"], st["pos"]
        log(i18n.tr("eng.t.resume", step=step, total=total))
    elif sampler:
        log(i18n.tr("eng.t.sampleFirst"))
        per_sample = sampler(0, None)
    step0 = step

    def save(path, epoch):
        save_lora(path, groups, {**meta, "ss_epoch": epoch, "ss_steps": step})
        _out({"type": "saved", "path": str(path), "epoch": epoch, "step": step})

    def save_state(epoch, batches, pos, done=False):
        """batches＝None：下一個 epoch 從頭排"""
        R.save(out_dir, {"lora": R.lora_state(groups), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batches": batches, "pos": pos,
                         "rng": R.rng_state(rng), "ema": ema_loss, "data": R.dataset_sig(items), "done": done}, _out)

    for epoch in range(start_epoch, int(c["epochs"]) + 1):
        if batches is None:
            order = [i for i in range(len(items)) for _ in range(int(c["repeats"]))]
            rng.shuffle(order)
            # batch > 1 時同一個分桶的才放一起
            batches, pending = [], {}
            for i in order:
                b = items[i]["bucket"]
                pending.setdefault(b, []).append(i)
                if len(pending[b]) == int(c["batch"]):
                    batches.append(pending.pop(b))
            batches += [v for v in pending.values() if v]
            pos = 0
        while pos < len(batches):
            bi = batches[pos]
            if stop_requested(c):
                break
            gen.manual_seed(int(c["seed"]) * 100003 + step)
            lat = torch.stack([items[i]["latent"] for i in bi]).to(device, torch.float32)
            caps = [shuffle_caption(items[i]["caption"], int(c["keep_tags"]), rng) if c["shuffle_tags"] else items[i]["caption"]
                    for i in bi]
            time_ids = torch.tensor([items[i]["time_ids"] for i in bi], device=device, dtype=torch.float16)
            t = (torch.rand(len(bi), generator=gen, device=device) * 1000).long().clamp_(0, 999)
            noise = torch.randn(lat.shape, generator=gen, device=device)
            a = alphas[t].view(-1, 1, 1, 1)
            noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.float16):
                if c["train_te"]:
                    hidden, pooled = encode_text(pipe, caps, device)
                else:
                    with torch.no_grad():
                        hidden, pooled = encode_text(pipe, caps, device)
                pred = unet(noisy.half(), t, encoder_hidden_states=hidden.half(),
                            added_cond_kwargs={"text_embeds": pooled.half(), "time_ids": time_ids}).sample
            loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pos += 1
            lv = float(loss.detach())
            ema_loss = lv if ema_loss is None else ema_loss * 0.98 + lv * 0.02
            g0 = opt.param_groups[0]
            el = time.time() - t_start - spent           # 試畫的時間不算進每步秒數
            sec = el / (step - step0)
            left = sum(1 for e in saves if e >= epoch) * per_sample
            _out({"type": "progress", "step": step, "total": total, "epoch": epoch, "epochs": int(c["epochs"]),
                  "loss": round(lv, 5), "loss_avg": round(ema_loss, 5), "lr": float(g0.get("d", 0) * g0["lr"]),
                  "sec": round(sec, 3), "eta": round(sec * (total - step) + left)})
        if stop_requested(c):
            save(out_dir / f"{c['name']}-step{step}.safetensors", epoch)
            save_state(epoch, batches, pos)
            log(i18n.tr("eng.t.stopped", step=step), "warn")
            return
        batches = None
        if epoch in saves:
            path = out_dir / (f"{c['name']}.safetensors" if epoch == int(c["epochs"]) else f"{c['name']}-e{epoch:02d}.safetensors")
            save(path, epoch)
            if sampler:
                per_sample = sampler(epoch, path)
                spent += per_sample
    if c.get("keep_state"):
        save_state(int(c["epochs"]) + 1, None, 0, done=True)
    log(i18n.tr("eng.t.done", step=step, min=f"{(time.time() - t_start) / 60:.1f}"), "ok")


def main():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # noqa: F401  （設定 HF_HOME：diffusers 讀單檔時要的設定檔放在 CoCo 的 hf_cache）
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        train(cfg)
        _out({"type": "done"})
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        _out({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    import certs
    certs.use_system_certs()     # 防毒軟體掃 HTTPS 時 huggingface_hub 才下載得到（certs.py）
    main()
