# -*- coding: utf-8 -*-
"""
anima_train.py —— 訓練 Anima 的 LoRA（訓練空間的 02 訓練）。另開一個行程跑：python core/anima_train.py 設定.json

使用者用 Anima-Standalone-Trainer（sd-scripts 系，Apache 2.0）訓練 Anima，預設值照他的設定；做法照它的行為：

- **LoRA 掛哪些層**：DiT 每個 Block 裡的所有 Linear——self_attn／cross_attn 的 q、k、v、output_proj，mlp 的 layer1／2，
  三組 adaln_modulation 的 1／2，一層 16 個、28 層共 448 個。LLM adapter 與文字編碼器不訓練（network_train_unet_only）。
  down 用 kaiming uniform（a=√5）、up 全 0，輸出 × alpha／rank（跟 SDXL 同一個 LoRALinear）。
- **存檔**：ComfyUI 的名稱（lora_unet_blocks_0_self_attn_q_proj…，用 anima.lora_alias 反查 diffusers 的模組），
  每層 alpha、lora_down、lora_up，全部 bf16（sd-scripts 的 save_precision）。跟使用者的 LoRA 1,344 個鍵一樣。
- **資料**：sd-scripts 的分桶（bucket_no_upscale）：比 解析度² 大的圖等比縮小到那個面積（長邊或短邊取 64 的倍數，
  比例誤差小的那個），小的不放大；再往下切到 64 的倍數、置中裁切。縮小用 cv2 的 INTER_AREA（sd-scripts 的預設）。
  latent 先算好（VAE 的平均值、照 latents_mean／std 正規化，跟 anima._encode_init 一樣）。
- **文字**：LLM adapter 不訓練，所以每張圖的條件（Qwen3 → text_conditioner）先算好，走 diffusers 生成時同一段程式；
  每一步有 caption_dropout（5%）的機率換成空字串的條件（sd-scripts 的 cache_uncond_embeddings）。
- **每一步**：t ＝ sigmoid(常態分布)，再照 shift 3 偏移成 3t／(1＋2t)（logit_normal＋discrete_flow_shift），夾在 1e-5～1−1e-5；
  x_t ＝ (1−t)·x₀ ＋ t·ε，模型預測 ε − x₀（diffusers 生成時也是這樣用：timestep 傳 σ、輸出是速度），MSE。
  weighting_scheme 是 logit_normal 時 sd-scripts 的權重全是 1，所以不另外加權。
- **優化**：Prodigy（學習率 1、weight decay 0.1、decouple，其餘照 prodigyopt 的預設）或 AdamW（1e-4、weight decay 0.01），
  cosine 降到 0、沒有 warmup；梯度裁切 1.0；DiT bf16 不動、LoRA float32、bf16 autocast；梯度檢查點。

CoCo 生圖時換上的融合版注意力（anima.FAST_ATTN）用了 out= 的寫法，不能反向傳播，訓練行程關掉它。
跟 server 的溝通、停止、試畫都跟 core/sdxl_train.py 一樣（共用那邊的函式）。
"""

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdxl_train as S  # noqa: E402
import resume as R  # noqa: E402
import i18n  # noqa: E402

DEFAULTS = {"resolution": 1024, "rank": 32, "alpha": 32.0, "epochs": 10, "repeats": 2, "batch": 1, "optimizer": "prodigy",
            "caption_dropout": 0.05, "shift": 3.0, "save_every": 1, "save_skip": 0, "seed": 42, "grad_ckpt": True}
STEP = 64
LR = {"prodigy": 1.0, "adamw": 1e-4}


# ---------------------------------------------------------------- 資料

def bucket(w, h, resolution, step=STEP):
    """sd-scripts 的 BucketManager.select_bucket（bucket_no_upscale）：原圖 (寬, 高) → (分桶 (寬, 高), 縮放後 (寬, 高))"""
    def round_to_steps(x):
        x = int(x + 0.5)
        return x - x % step
    max_area = resolution * resolution
    ar = w / h
    if w * h > max_area:
        rw = math.sqrt(max_area * ar)
        rh = max_area / rw
        bw = round_to_steps(rw)
        bh_in_w = round_to_steps(bw / ar)
        bh = round_to_steps(rh)
        bw_in_h = round_to_steps(bh * ar)
        if abs(bw / bh_in_w - ar) < abs(bw_in_h / bh - ar):
            resized = (bw, int(bw / ar + 0.5))
        else:
            resized = (int(bh * ar + 0.5), bh)
    else:
        resized = (w, h)
    return (resized[0] - resized[0] % step, resized[1] - resized[1] % step), resized


def load_image(path, resolution):
    """讀圖、縮到分桶的大小、置中裁切 → (uint8 陣列 高×寬×3, 分桶 (寬, 高))"""
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    (bw, bh), (rw, rh) = bucket(w, h, resolution)
    if (rw, rh) != (w, h):
        a = cv2.resize(a, (rw, rh), interpolation=cv2.INTER_AREA)      # 只會縮小（不放大）
    top, left = (rh - bh) // 2, (rw - bw) // 2
    return a[top:top + bh, left:left + bw], (bw, bh)


def sample_t(gen, n, shift, device):
    """logit_normal 的時間步，再照 shift 偏移（sd-scripts 的 get_noisy_model_input_and_timesteps）"""
    import torch
    t = torch.sigmoid(torch.randn(n, generator=gen, device=device))
    if shift and shift != 1.0:
        t = (t * shift) / (1 + (shift - 1) * t)
    return t.clamp(1e-5, 1.0 - 1e-5)


# ---------------------------------------------------------------- LoRA 的名稱

def lora_names(num_layers):
    """diffusers 的模組路徑 → ComfyUI／sd-scripts 的 LoRA 名稱（lora_unet_blocks_0_self_attn_q_proj）。
    只有 Block 裡的 Linear（x_embedder、t_embedder、final_layer 那些 sd-scripts 預設不掛；q_norm／k_norm 不是 Linear）"""
    A = _anima()
    return {dst: f"lora_unet_{src}" for src, dst in A.lora_alias(num_layers).items()
            if src.startswith("blocks_") and not dst.endswith((".norm_q", ".norm_k"))}


def _anima():
    import pipeline as P
    return P._anima()


# ---------------------------------------------------------------- 先算好

def encode_caption(pipe, text, device):
    """一則標籤 → DiT 的 cross attention 輸入（diffusers 生成時同一段：Qwen3 最後一層 → text_conditioner），放 CPU"""
    import torch
    from diffusers.modular_pipelines.anima.before_denoise import AnimaTextConditioningStep
    from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep
    with torch.no_grad():
        q, qm = AnimaTextEncoderStep._get_qwen_prompt_embeds(pipe, text, 512, device, torch.bfloat16)
        ids, im = AnimaTextEncoderStep._get_t5_prompt_ids(pipe, text, 512, device)
        out = AnimaTextConditioningStep._condition_prompt_embeds(pipe, q, qm, ids, im, device,
                                                                 pipe.text_conditioner.dtype, pipe.transformer.dtype)
    return out.cpu()


def encode_latent(vae, arr, device):
    """uint8 陣列 → 正規化的 latent (1, C, 1, h, w) float32，放 CPU（anima._encode_init 同一套）"""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(arr.copy()).to(torch.float32) / 127.5 - 1.0
        x = x.permute(2, 0, 1)[None, :, None].to(device=device, dtype=vae.dtype)
        z = vae.encode(x).latent_dist.mode()
        c = vae.config.z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1).to(z)
        std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1).to(z)
        return ((z - mean) / std).float().cpu()


# ---------------------------------------------------------------- 訓練

def make_optimizer(params, c):
    import torch
    lr = float(c.get("lr") or LR[c["optimizer"]])
    if c["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    from vendor.prodigyopt import Prodigy
    return Prodigy(params, lr=lr, weight_decay=0.1, decouple=True)      # 使用者的 optimizer_args，其他照預設


def train(cfg):
    import torch
    import torch.nn.functional as F

    c = {**DEFAULTS, **cfg}
    if c["optimizer"] not in LR:
        raise ValueError(i18n.tr("eng.t.badOptimizer", o=c['optimizer']))
    device = "cuda"
    rng = random.Random(int(c["seed"]))
    torch.manual_seed(int(c["seed"]))

    imgs = S.list_images(c["dataset"])
    if not imgs:
        raise ValueError(i18n.tr("eng.t.noImages"))
    items = [{"path": str(p), "caption": S.read_caption(p)} for p in imgs]
    no_cap = sum(1 for it in items if not it["caption"])
    if no_cap:
        S.log(i18n.tr("eng.t.noCaption", n=no_cap), "warn")
    per_epoch, total = S.plan_steps(len(items), int(c["repeats"]), int(c["epochs"]), int(c["batch"]))
    S.log(i18n.tr("eng.t.plan", n=len(items), repeats=c['repeats'], epochs=c['epochs'], batch=c['batch'], total=total))

    t0 = time.time()
    S.log(i18n.tr("eng.t.loading"))
    A = _anima()
    A.FAST_ATTN = False                   # 融合版注意力不能反向傳播
    pipe = A.load(c["model"], c["vae"], c["te"], log=lambda *a: None, dtype=torch.bfloat16)
    for name in ("text_encoder", "text_conditioner", "vae"):
        getattr(pipe, name).to(device)
    S.log(i18n.tr("eng.t.loaded", s=f"{time.time() - t0:.0f}"))

    S.log(i18n.tr("eng.t.latentsTags"))
    for i, it in enumerate(items):
        arr, it["bucket"] = load_image(it["path"], int(c["resolution"]))
        it["latent"] = encode_latent(pipe.vae, arr, device)
        it["cond"] = encode_caption(pipe, it["caption"], device)
        S._out({"type": "cache", "done": i + 1, "total": len(items)})
    uncond = encode_caption(pipe, "", device)
    pipe.text_encoder.to("cpu")           # 試畫時才搬回來
    torch.cuda.empty_cache()

    dit = pipe.transformer
    dit.requires_grad_(False)
    dit.to(device)
    names = lora_names(len(dit.transformer_blocks))
    linear = {n for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("transformer_blocks.")}
    if linear != set(names):              # 每個 Block 裡的 Linear 都要對得上 ComfyUI 的名稱，不能漏也不能多
        raise RuntimeError(i18n.tr("eng.t.loraMismatch", a=len(linear), b=len(names), less=sorted(linear - set(names))[:3], more=sorted(set(names) - linear)[:3]))
    rank, alpha = int(c["rank"]), float(c["alpha"])
    mods = S.inject(dit, rank, alpha, lambda n: n in names)
    groups = [("lora_unet", mods)]
    if c["grad_ckpt"]:
        dit.enable_gradient_checkpointing()
    dit.train()
    S.log(i18n.tr("eng.t.animaLora", n=len(mods), rank=rank, alpha=f"{alpha:g}"))

    params = [p for m in mods.values() for p in (m.lora_down.weight, m.lora_up.weight)]
    for p in params:
        p.requires_grad_(True)
    opt = make_optimizer([{"params": params, "lr": float(c.get("lr") or LR[c["optimizer"]])}], c)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: S.cosine(s, total))

    meta = {"ss_base_model_version": "anima", "ss_network_module": "networks.lora_anima", "ss_network_dim": rank,
            "ss_network_alpha": alpha, "ss_resolution": c["resolution"], "ss_num_train_images": len(items),
            "ss_num_epochs": c["epochs"], "ss_output_name": c["name"], "ss_optimizer": c["optimizer"],
            "ss_timestep_sample_method": "logit_normal", "ss_discrete_flow_shift": c["shift"],
            "ss_caption_dropout_rate": c["caption_dropout"], "coco_trainer": "CoCo Image Studio",
            "modelspec.title": c["name"]}
    out_dir = Path(c["out_dir"])
    gen = torch.Generator(device=device)

    def key(prefix, name):
        return names[name]

    def save(path, epoch):
        S.save_lora(path, groups, {**meta, "ss_epoch": epoch, "ss_steps": step}, key=key, dtype=torch.bfloat16)
        S._out({"type": "saved", "path": str(path), "epoch": epoch, "step": step})

    def te_to(dev):
        pipe.text_encoder.to(dev)

    sampler = S.make_sampler(pipe, c, [dit], device, prepare=lambda: None,
                             before=lambda: te_to(device), after=lambda: te_to("cpu"))
    saves = S.save_epochs(int(c["epochs"]), int(c["save_skip"]), int(c["save_every"]))
    per_sample = 0.0
    step = 0
    ema_loss, t_start, spent = None, time.time(), 0.0
    start_epoch, batches, pos = 1, None, 0
    if c.get("resume"):
        st = R.load(c["resume"])
        R.load_lora(groups, st["lora"])
        if R.check_dataset(st["data"], items):
            S.log(i18n.tr("eng.t.changed"), "warn")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        R.set_rng(rng, st["rng"])
        step, ema_loss, start_epoch, batches, pos = st["step"], st["ema"], st["epoch"], st["batches"], st["pos"]
        S.log(i18n.tr("eng.t.resume", step=step, total=total))
    elif sampler:
        S.log(i18n.tr("eng.t.sampleFirst"))
        per_sample = sampler(0, None)
    step0 = step
    pad = {}

    def save_state(epoch, batches, pos, done=False):
        R.save(out_dir, {"lora": R.lora_state(groups), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "step": step, "epoch": epoch, "batches": batches, "pos": pos,
                         "rng": R.rng_state(rng), "ema": ema_loss, "data": R.dataset_sig(items), "done": done}, S._out)

    for epoch in range(start_epoch, int(c["epochs"]) + 1):
        if batches is None:
            order = [i for i in range(len(items)) for _ in range(int(c["repeats"]))]
            rng.shuffle(order)
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
            if S.stop_requested(c):
                break
            gen.manual_seed(int(c["seed"]) * 100003 + step)
            lat = torch.cat([items[i]["latent"] for i in bi]).to(device)
            cond = torch.cat([uncond if rng.random() < float(c["caption_dropout"]) else items[i]["cond"]
                              for i in bi]).to(device)
            bw, bh = items[bi[0]]["bucket"]
            if (bw, bh) not in pad:
                pad[(bw, bh)] = torch.zeros(1, 1, bh, bw, device=device, dtype=torch.bfloat16)
            t = sample_t(gen, len(bi), float(c["shift"]), device)
            noise = torch.randn(lat.shape, generator=gen, device=device)
            tv = t.view(-1, 1, 1, 1, 1)
            noisy = (1 - tv) * lat + tv * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit(hidden_states=noisy.to(torch.bfloat16), timestep=t.to(torch.bfloat16),
                           encoder_hidden_states=cond, padding_mask=pad[(bw, bh)], return_dict=False)[0]
            loss = F.mse_loss(pred.float(), (noise - lat).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pos += 1
            lv = float(loss.detach())
            ema_loss = lv if ema_loss is None else ema_loss * 0.98 + lv * 0.02
            g0 = opt.param_groups[0]
            el = time.time() - t_start - spent
            sec = el / (step - step0)
            left = sum(1 for e in saves if e >= epoch) * per_sample
            lr_now = float(g0.get("d", 1.0) * g0["lr"]) if c["optimizer"] == "prodigy" else float(g0["lr"])
            S._out({"type": "progress", "step": step, "total": total, "epoch": epoch, "epochs": int(c["epochs"]),
                    "loss": round(lv, 5), "loss_avg": round(ema_loss, 5), "lr": lr_now,
                    "sec": round(sec, 3), "eta": round(sec * (total - step) + left)})
        if S.stop_requested(c):
            save(out_dir / f"{c['name']}-step{step}.safetensors", epoch)
            save_state(epoch, batches, pos)
            S.log(i18n.tr("eng.t.stopped", step=step), "warn")
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
    S.log(i18n.tr("eng.t.done", step=step, min=f"{(time.time() - t_start) / 60:.1f}"), "ok")


def main():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # noqa: F401  （設定 HF_HOME）
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
