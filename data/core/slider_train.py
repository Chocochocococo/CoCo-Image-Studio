# -*- coding: utf-8 -*-
"""
slider_train.py —— 訓練 Slider LoRA（可以調強度的滑桿：+1 往「正向」、−1 往「反向」、0 不變）。
另開一個行程跑：python core/slider_train.py 設定.json（server/train.py 開，SDXL 與 Anima 都可以）

做法照 ai-toolkit（MIT）新版的 ConceptSliderTrainer（只讀程式了解做法）：
- **底圖**：拿真的圖加雜訊當模型的輸入，不從雜訊一步步畫（使用者以前用 ai-toolkit 舊版 slider 失敗，那個每一步都要先畫
  幾十步，又慢又吃記憶體）。底圖預設照「中性」提示詞用 CoCo 自己的引擎畫（server/train.py 組好跟平常生圖一樣的設定），
  也可以指定資料集資料夾。
- **中性提示詞可以好幾則**（c["prompts"]：每則一組 中性／中性＋正向的詞／中性＋反向的詞，server/train.py 組好）：
  底圖輪流照每一則畫，每一步用那張底圖那一則；資料集的底圖每一步輪流換一則。只用一則短的，Slider 只在短提示詞上推得動
  （2026-10-06 實測，見 server/train.py 的 MAX_NEUTRAL）。只有一則時跟以前一樣。
- **每一步**：同一個加了雜訊的底圖，不掛 LoRA 各算一次「正向」「中性」「反向」（有錨點再加「錨點」）的預測；
  目標 ＝ 中性 ＋ 強度 ×（正向 − 反向），再把平均與標準差調成跟中性一樣（norm_like）。LoRA ×(+1) 用中性提示詞去逼近這個目標，
  LoRA ×(−1) 逼近 中性 ＋ 強度 ×（反向 − 正向）。有錨點時錨點提示詞的預測要維持不掛 LoRA 的樣子（不想被影響的東西）。
  loss 照 ai-toolkit：(2 × 目標的 MSE ＋ 錨點強度 × 錨點的 MSE) ／ 3，+1 與 −1 各一次，平均。
- **LoRA**：SDXL 掛 UNet 的注意力層（跟一般 LoRA 訓練一樣）、Anima 掛 DiT 的 Block；文字編碼器不訓練。
  AdamW 2e-4（weight decay 1e-4）、學習率固定、梯度裁切 1.0；存檔格式跟一般 LoRA 訓練一樣，生圖時權重給負的就是往反向。
- **試畫**：每存一次畫 −2、−1、+1、+2（開始前畫一張不掛 LoRA 的當 0），全部進比較組。

2026-10-06 原型（SDXL、768²、rank 8／alpha 4、強度 3、2e-4）：100 步就看得出年齡變化，400 步 ±2 很明顯，每步 1.4 秒。
"""

import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdxl_train as S  # noqa: E402
import resume as R  # noqa: E402
import i18n  # noqa: E402

DEFAULTS = {"resolution": 768, "rank": 8, "alpha": 4.0, "steps": 400, "lr": 2e-4, "guidance": 3.0, "anchor_strength": 1.0,
            "n_base": 12, "save_every": 100, "seed": 42, "shift": 3.0, "grad_ckpt": True}
MULTS = (-2.0, -1.0, 1.0, 2.0)            # 試畫的強度（0 是開始前那張）


# ---------------------------------------------------------------- 算目標與 loss

def norm_like(x, ref):
    """x 的平均與標準差調成跟 ref 一樣（ai-toolkit 的 norm_like_tensor）"""
    return (x - x.mean()) / (x.std() + 1e-8) * ref.std() + ref.mean()


def targets(pos, neu, neg, guidance):
    """不掛 LoRA 的三個預測 → (LoRA ×+1 的目標, LoRA ×−1 的目標)"""
    return (norm_like(neu + guidance * (pos - neg), neu),
            norm_like(neu + guidance * (neg - pos), neu))


def slider_loss(pred, target, anchor_pred=None, anchor_target=None, anchor_strength=1.0):
    """ai-toolkit：enhance 與 erase 兩個目標其實一樣，所以是 (2 × MSE ＋ 錨點強度 × 錨點的 MSE) ／ 3"""
    import torch.nn.functional as F
    main = F.mse_loss(pred.float(), target.float())
    anchor = (F.mse_loss(anchor_pred.float(), anchor_target.float()) * anchor_strength
              if anchor_pred is not None else main * 0)
    return (2 * main + anchor) / 3


def save_steps(steps, every):
    """哪幾步存檔：每 every 步，最後一步一定存"""
    every = max(1, int(every))
    return sorted({s for s in range(every, steps + 1, every)} | {steps})


def step_tag(step):
    return f"s{int(step):04d}"


def set_mult(mods, v):
    for m in mods.values():
        m.mult = float(v)


# ---------------------------------------------------------------- 兩種架構

class SDXL:
    """SDXL：UNet 預測雜訊（epsilon），時間步 0～999 均勻、DDPM scaled_linear"""
    def __init__(self, c, device):
        import torch
        from diffusers import DDPMScheduler, StableDiffusionXLPipeline
        self.c, self.device = c, device
        self.pipe = StableDiffusionXLPipeline.from_single_file(c["model"], torch_dtype=torch.float16, use_safetensors=True)
        if self.pipe.scheduler.config.get("prediction_type") == "v_prediction":
            raise ValueError(i18n.tr("eng.t.vpred"))
        self.pipe.to(device)
        self.pipe._coco_vram = "normal"
        sch = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.alphas = sch.alphas_cumprod.to(device)
        self.model = self.pipe.unet

    def cache(self, paths):
        items = [{"path": str(p)} for p in paths]
        S.cache_latents(self.pipe, items, int(self.c["resolution"]), False, False, random.Random(0), self.device)
        return [{"latent": it["latent"], "time_ids": it["time_ids"]} for it in items]

    def encode(self, text):
        # 跟生圖同一套（conditioning.encode_prompt：每 75 個 token 一塊、權重語法也認得）。以前用 sdxl_train.encode_text 截到 77，
        # 長的中性提示詞接上正向、反向的詞之後，加的詞剛好被截掉，三則變成一樣、什麼都學不到（2026-10-06 使用者的設定：
        # 四則裡兩則 104、123 個 token）。77 以內的跟以前逐位元一樣（同一天量過三則）
        import torch
        import conditioning as CD
        with torch.no_grad():
            h, p, _, _ = CD.encode_prompt(self.pipe, text, "", clip_skip=2)
        return h.half(), p.half()

    def text_off(self):
        for m in (self.pipe.text_encoder, self.pipe.text_encoder_2):
            m.to("cpu")

    def text_on(self):
        for m in (self.pipe.text_encoder, self.pipe.text_encoder_2):
            m.to(self.device)

    def lora(self, rank, alpha):
        return S.inject(self.model, rank, alpha, "attentions"), S.kohya_key, None

    def noisy(self, item, gen):
        import torch
        z = item["latent"][None].to(self.device, torch.float32)
        t = torch.randint(0, 1000, (1,), device=self.device, generator=gen)
        noise = torch.randn(z.shape, device=self.device, generator=gen)
        a = self.alphas[t].view(-1, 1, 1, 1)
        return (a.sqrt() * z + (1 - a).sqrt() * noise).half(), t

    def pair_noisy(self, a, b, gen):
        """成對圖片：兩張加同一份雜訊、同一個時間步 → (正向的輸入, 反向的輸入, t, 正向的目標, 反向的目標)；目標都是雜訊（epsilon）"""
        import torch
        zp, zn = (it["latent"][None].to(self.device, torch.float32) for it in (a, b))
        t = torch.randint(0, 1000, (1,), device=self.device, generator=gen)
        noise = torch.randn(zp.shape, device=self.device, generator=gen)
        al = self.alphas[t].view(-1, 1, 1, 1)
        return ((al.sqrt() * zp + (1 - al).sqrt() * noise).half(), (al.sqrt() * zn + (1 - al).sqrt() * noise).half(), t,
                noise, noise)

    def predict(self, x, t, conds, item):
        import torch
        if len({c[0].shape[1] for c in conds}) > 1:
            # 長度不一樣（長的提示詞多一塊）：分開算，不補零（補零的位置也會被注意力看到，預測就不是那則提示詞的了）
            return torch.cat([self.predict(x, t, [c], item) for c in conds])
        n = len(conds)
        h = torch.cat([c[0] for c in conds])
        p = torch.cat([c[1] for c in conds])
        tid = torch.tensor([item["time_ids"]] * n, device=self.device, dtype=torch.float16)
        return self.model(x.expand(n, -1, -1, -1), t.expand(n), encoder_hidden_states=h,
                          added_cond_kwargs={"text_embeds": p, "time_ids": tid}).sample

    def autocast(self):
        import torch
        return torch.autocast("cuda", dtype=torch.float16)

    def sampler_hooks(self):
        # VAE 照預設（make_sampler 轉 bf16）；文字編碼器不訓練，算完放 CPU，試畫時才搬上來
        return {"before": self.text_on, "after": self.text_off}


class Anima:
    """Anima：DiT 預測速度（ε − x₀），t ＝ logit_normal＋shift（跟 core/anima_train.py 一樣）"""
    def __init__(self, c, device):
        import torch
        import anima_train as AT
        self.AT, self.c, self.device = AT, c, device
        A = AT._anima()
        A.FAST_ATTN = False               # 融合版注意力不能反向傳播
        self.pipe = A.load(c["model"], c["vae"], c["te"], log=lambda *a: None, dtype=torch.bfloat16)
        for name in ("text_encoder", "text_conditioner", "vae", "transformer"):
            getattr(self.pipe, name).to(device)
        self.pipe._coco_vram = "normal"
        self.model = self.pipe.transformer
        self.pad = {}

    def cache(self, paths):
        out = []
        for p in paths:
            arr, b = self.AT.load_image(p, int(self.c["resolution"]))
            out.append({"latent": self.AT.encode_latent(self.pipe.vae, arr, self.device), "bucket": b})
        return out

    def encode(self, text):
        return self.AT.encode_caption(self.pipe, text, self.device).to(self.device)

    def text_off(self):
        self.pipe.text_encoder.to("cpu")

    def text_on(self):
        self.pipe.text_encoder.to(self.device)

    def lora(self, rank, alpha):
        import torch
        names = self.AT.lora_names(len(self.model.transformer_blocks))
        mods = S.inject(self.model, rank, alpha, lambda n: n in names)
        return mods, (lambda prefix, name: names[name]), torch.bfloat16

    def noisy(self, item, gen):
        import torch
        z = item["latent"].to(self.device)
        t = self.AT.sample_t(gen, 1, float(self.c["shift"]), self.device)
        noise = torch.randn(z.shape, device=self.device, generator=gen)
        tv = t.view(-1, 1, 1, 1, 1)
        self._noise, self._z = noise, z
        return ((1 - tv) * z + tv * noise).to(torch.bfloat16), t.to(torch.bfloat16)

    def pair_noisy(self, a, b, gen):
        """成對圖片：同一份雜訊、同一個 t；目標是速度（雜訊 − 那張的 latent，跟 core/anima_train.py 一樣）"""
        import torch
        zp, zn = a["latent"].to(self.device), b["latent"].to(self.device)
        t = self.AT.sample_t(gen, 1, float(self.c["shift"]), self.device)
        noise = torch.randn(zp.shape, device=self.device, generator=gen)
        tv = t.view(-1, 1, 1, 1, 1)
        return (((1 - tv) * zp + tv * noise).to(torch.bfloat16), ((1 - tv) * zn + tv * noise).to(torch.bfloat16),
                t.to(torch.bfloat16), noise - zp, noise - zn)

    def predict(self, x, t, conds, item):
        import torch
        n = len(conds)
        bw, bh = item["bucket"]
        if (bw, bh) not in self.pad:
            self.pad[(bw, bh)] = torch.zeros(1, 1, bh, bw, device=self.device, dtype=torch.bfloat16)
        return self.model(hidden_states=x.expand(n, -1, -1, -1, -1), timestep=t.expand(n), encoder_hidden_states=torch.cat(conds),
                          padding_mask=self.pad[(bw, bh)], return_dict=False)[0]

    def autocast(self):
        import torch
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def sampler_hooks(self):
        return {"prepare": lambda: None, "before": self.text_on, "after": self.text_off}


ARCH = {"sdxl": SDXL, "anima": Anima}


# ---------------------------------------------------------------- 底圖

def base_reqs(c):
    """底圖的生圖請求：每則中性提示詞一個（c["bases"]）；以前的設定只有一個（c["base"]）"""
    return c.get("bases") or [c["base"]]


def prompt_sets(c):
    """每則中性提示詞一組 {neu, pos, neg}；以前的設定只有一組（neutral／positive／negative）"""
    return c.get("prompts") or [{"neu": c["neutral"], "pos": c["positive"], "neg": c["negative"]}]


def make_base(arch, c, folder):
    """照中性提示詞畫底圖（第 i 張用第 i % 則的請求），存成 png → 路徑清單"""
    import pipeline as P
    reqs = base_reqs(c)
    P.apply_scheduler(arch.pipe, reqs[0].get("sampler") or "Euler a", reqs[0].get("schedule") or "Normal", lambda *a: None,
                      shift=reqs[0].get("shift"))
    paths = []
    for i in range(int(c["n_base"])):
        req = reqs[i % len(reqs)]
        img, *_ = P.generate(arch.pipe, {**req, "seed": int(c["seed"]) * 1000 + i}, log=lambda *a: None)
        p = Path(folder) / f"base_{i:02d}.png"
        img.save(p)
        paths.append(p)
        S._out({"type": "base", "done": i + 1, "total": int(c["n_base"])})
    return paths


def make_pairs(arch, c, folder):
    """成對圖片 → [{pos, neg, prompt}]（各自的 latent）。反向那張大小跟正向不一樣就先縮放成一樣，
    分桶與裁切才會一樣、兩張的 latent 對得上（Qwen 修圖做的成對圖邊長對齊 16，可能差幾個像素）"""
    from PIL import Image
    pos, neg = [], []
    for i, p in enumerate(c["pairs"]):
        a, b = S.load_rgb(p["pos"]), S.load_rgb(p["neg"])
        if b.size != a.size:
            path = Path(folder) / f"neg_{i:04d}.png"
            b.resize(a.size, Image.LANCZOS).save(path)
            neg.append(path)
        else:
            neg.append(Path(p["neg"]))
        pos.append(Path(p["pos"]))
    P, N = arch.cache(pos), arch.cache(neg)
    return [{"pos": P[i], "neg": N[i], "prompt": c["pairs"][i]["prompt"]} for i in range(len(pos))]


# ---------------------------------------------------------------- 訓練

def train(cfg):
    import torch

    c = {**DEFAULTS, **cfg}
    if c.get("arch") not in ARCH:
        raise ValueError(i18n.tr("eng.t.badArch", arch=c.get('arch')))
    device = "cuda"
    rng = random.Random(int(c["seed"]))
    torch.manual_seed(int(c["seed"]))
    steps = int(c["steps"])

    t0 = time.time()
    S.log(i18n.tr("eng.t.loading"))
    arch = ARCH[c["arch"]](c, device)
    S.log(i18n.tr("eng.t.loaded", s=f"{time.time() - t0:.0f}"))

    pairs = bool(c.get("pairs"))                  # 成對圖片（server/train.py 的 sl_mode＝pairs）
    sets = None if pairs else prompt_sets(c)
    st = R.load(c["resume"]) if c.get("resume") else None
    tmp = None
    if st:
        items = st["items"]                       # 底圖的 latent 存在進度檔裡：不用重畫，跟停下來前同一批
        S.log(i18n.tr("eng.t.baseResume", n=len(items)))
    elif pairs:
        tmp = tempfile.mkdtemp(prefix="coco-slider-")
        S.log(i18n.tr("eng.t.pairs", n=len(c['pairs'])))
    elif c.get("dataset"):
        paths = S.list_images(c["dataset"])
        if not paths:
            raise ValueError(i18n.tr("eng.t.noImages"))
        S.log(i18n.tr("eng.t.baseDataset", n=len(paths)))
    else:
        tmp = tempfile.mkdtemp(prefix="coco-slider-")
        S.log(i18n.tr("eng.t.drawBase", n=c['n_base']))
        t1 = time.time()
        paths = make_base(arch, c, tmp)
        S.log(i18n.tr("eng.t.baseDone", s=f"{time.time() - t1:.0f}"))
    if not st:
        try:
            items = make_pairs(arch, c, tmp) if pairs else arch.cache(paths)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    if not st and not pairs and not c.get("dataset"):
        for i, it in enumerate(items):
            it["set"] = i % len(sets)              # 自動畫的底圖：用畫它的那一則

    if pairs:
        # 每一對用自己的提示詞（同樣的只算一次）
        texts = sorted({it["prompt"] for it in items})
        for it in items:
            it["ci"] = texts.index(it["prompt"])
        pconds = [arch.encode(x) for x in texts]
        conds, anchor = [], None
    else:
        # 每一組照 正向、中性、反向（、錨點）的順序算：只有一組時跟以前同一批、同一個順序
        conds = [{k: arch.encode(s[k]) for k in ("pos", "neu", "neg")} for s in sets]
        anchor = arch.encode(c["anchor"]) if c.get("anchor") else None
        if len(sets) > 1:
            S.log(i18n.tr("eng.t.neutrals", n=len(sets)))
    arch.text_off()
    torch.cuda.empty_cache()

    model = arch.model
    model.requires_grad_(False)
    rank, alpha = int(c["rank"]), float(c["alpha"])
    mods, key, save_dtype = arch.lora(rank, alpha)
    if c["grad_ckpt"]:
        model.enable_gradient_checkpointing()
    if pairs:
        S.log(i18n.tr("eng.t.slLoraPairs", n=len(mods), rank=rank, alpha=f"{alpha:g}", pairs=len(items)))
    else:
        S.log(i18n.tr("eng.t.slLora", n=len(mods), rank=rank, alpha=f"{alpha:g}", g=f"{c['guidance']:g}")
              + (i18n.tr("eng.t.anchor", a=c['anchor']) if c.get("anchor") else ""))
    params = [p for m in mods.values() for p in (m.lora_down.weight, m.lora_up.weight)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=float(c["lr"]), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=c["arch"] == "sdxl")      # SDXL 用 fp16；Anima 是 bf16，不用

    meta = {"ss_base_model_version": "sdxl_base_v1-0" if c["arch"] == "sdxl" else "anima",
            "ss_network_module": "networks.lora" if c["arch"] == "sdxl" else "networks.lora_anima",
            "ss_network_dim": rank, "ss_network_alpha": alpha, "ss_resolution": c["resolution"],
            "ss_output_name": c["name"], "coco_trainer": "CoCo Image Studio", "modelspec.title": c["name"],
            "ss_training_comment": f"Slider：+ {c['positive']} ／ − {c['negative']}",
            "coco_slider": json.dumps({k: c.get(k, "") for k in ("neutral", "positive", "negative", "anchor", "guidance")},
                                      ensure_ascii=False)}
    out_dir = Path(c["out_dir"])
    step = 0

    def save(path):
        S.save_lora(path, [("lora_unet", mods)], {**meta, "ss_steps": step}, key=key, dtype=save_dtype)
        S._out({"type": "saved", "path": str(path), "epoch": step, "step": step})

    def pair_step(item, gen):
        """成對圖片的一步（ai-toolkit 的 ImageReferenceSliderTrainer）：LoRA ×(+1) 照正向那張、×(−1) 照反向那張預測雜訊，
        一般的 MSE，兩個平均。兩張共同的部分（資料集的畫風）在 +1、−1 被往相反方向拉、互相抵掉，學到的只有兩張不一樣的地方"""
        import torch.nn.functional as F
        xp, xn, t, tp, tn = arch.pair_noisy(item["pos"], item["neg"], gen)
        cond = pconds[item["ci"]]
        total = 0.0
        for mult, x, tgt, it in ((1.0, xp, tp, item["pos"]), (-1.0, xn, tn, item["neg"])):
            set_mult(mods, mult)
            with arch.autocast():
                out = arch.predict(x, t, [cond], it)
            loss = F.mse_loss(out.float(), tgt.float()) / 2
            scaler.scale(loss).backward()
            total += float(loss.detach())
        set_mult(mods, 1.0)
        return total

    def save_state(done=False):
        R.save(out_dir, {"lora": R.lora_state([("lora_unet", mods)]), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                         "step": step, "order": list(order), "rng": R.rng_state(rng), "ema": ema, "items": items,
                         "done": done}, S._out)

    sampler = S.make_sampler(arch.pipe, c, [model], device, **arch.sampler_hooks())
    per_sample = 0.0
    saves = save_steps(steps, c["save_every"])
    order = []
    gen = torch.Generator(device=device)
    t_start, spent, ema = time.time(), 0.0, None
    if st:
        R.load_lora([("lora_unet", mods)], st["lora"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        R.set_rng(rng, st["rng"])
        step, order, ema = st["step"], list(st["order"]), st["ema"]
        S.log(i18n.tr("eng.t.resume", step=step, total=steps))
    elif sampler:
        S.log(i18n.tr("eng.t.slSampleFirst"))
        per_sample = sampler(0, None, mults=[0.0], tag="base", set_mult=lambda v: set_mult(mods, v))
    step0 = step
    model.train()

    while step < steps:
        if S.stop_requested(c):
            save(out_dir / f"{c['name']}-{step_tag(step)}.safetensors")
            save_state()
            S.log(i18n.tr("eng.t.stopped", step=step), "warn")
            return
        if not order:
            order = list(range(len(items)))
            rng.shuffle(order)
        item = items[order.pop()]
        if pairs:
            gen.manual_seed(int(c["seed"]) * 100003 + step)
            total = pair_step(item, gen)
        else:
            cond = conds[item.get("set", step % len(sets))]       # 資料集的底圖：每一步輪流換一則
            if anchor is not None:
                cond = {**cond, "anc": anchor}
            gen.manual_seed(int(c["seed"]) * 100003 + step)
            x, t = arch.noisy(item, gen)
            with torch.no_grad():
                set_mult(mods, 0.0)
                model.eval()
                names = list(cond)
                preds = dict(zip(names, arch.predict(x, t, [cond[k] for k in names], item).float().chunk(len(names))))
                tgt = dict(zip((1.0, -1.0), targets(preds["pos"], preds["neu"], preds["neg"], float(c["guidance"]))))
                model.train()
            total = 0.0
            use = [cond["neu"]] + ([cond["anc"]] if "anc" in cond else [])
            for mult in (1.0, -1.0):
                set_mult(mods, mult)
                with arch.autocast():
                    out = arch.predict(x, t, use, item)
                parts = out.chunk(len(use))
                loss = slider_loss(parts[0], tgt[mult], parts[1] if len(use) > 1 else None, preds.get("anc"),
                                   float(c["anchor_strength"])) / 2
                scaler.scale(loss).backward()
                total += float(loss.detach())
            set_mult(mods, 1.0)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        ema = total if ema is None else ema * 0.98 + total * 0.02
        el = time.time() - t_start - spent
        left = sum(1 for s in saves if s > step) * per_sample * len(MULTS)
        S._out({"type": "progress", "step": step, "total": steps, "epoch": step, "epochs": steps, "loss": round(total, 5),
                "loss_avg": round(ema, 5), "lr": float(c["lr"]), "sec": round(el / (step - step0), 3),
                "eta": round(el / (step - step0) * (steps - step) + left)})
        if step in saves:
            path = out_dir / (f"{c['name']}.safetensors" if step == steps else f"{c['name']}-{step_tag(step)}.safetensors")
            save(path)
            if sampler:
                took = sampler(step, path, mults=list(MULTS), tag=step_tag(step), set_mult=lambda v: set_mult(mods, v))
                spent += took
                per_sample = took / len(MULTS)
    if c.get("keep_state"):
        save_state(done=True)
    S.log(i18n.tr("eng.t.done", step=step, min=f"{(time.time() - t_start) / 60:.1f}"), "ok")


def main():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # noqa: F401  （設定 HF_HOME）
    # 長的提示詞 CoCo 自己分段（conditioning.py），tokenizer 那句「Token indices sequence length is longer…」是預期中的
    # （跟 core/worker.py 一樣；使用者看到以為出錯了）
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        train(cfg)
        S._out({"type": "done"})
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        S._out({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    import certs
    certs.use_system_certs()     # 防毒軟體掃 HTTPS 時 huggingface_hub 才下載得到（certs.py）
    main()
