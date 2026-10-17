# -*- coding: utf-8 -*-
"""
anima.py —— Anima 的載入與生成

Anima 在 diffusers 0.39 是 modular pipeline，元件如下：

    transformer        CosmosTransformer3DModel     Anima 重用 Cosmos Predict2 DiT
    text_conditioner   AnimaTextConditioner         唯一的模型專屬模組（llm_adapter）
    text_encoder       Qwen3Model                   Qwen3-0.6B
    vae                AutoencoderKLQwenImage
    scheduler          FlowMatchEulerDiscreteScheduler

使用者手上的是 ComfyUI 拆分式的單檔（鍵名開頭是 net.），diffusers 沒有
對應的 from_single_file。但兩邊的差異純粹是改名 —— 實測 567/567、118/118、
194/194、310/310 全部對得上，缺鍵與形狀不符都是 0。

**架構本身完全交給 diffusers，這裡只做鍵名對照。**
這跟 §6.3.3 的 LoRA 鍵名對照表是同一招，不是自己扛第 2 層。

超參數不需要外部設定檔，全部從權重形狀推得（做法與 ComfyUI / Neo Forge
的偵測邏輯相同）。只有 VAE 的 latents_mean / latents_std 與排程器的 shift
推不出來，那兩份 JSON 放在 assets/anima/。
"""

import json
import os
import re
from pathlib import Path

import i18n

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "anima"


# ---------------------------------------------------------------- 鍵名對照
_TOP = {
    "x_embedder.proj.1":              "patch_embed.proj",
    "t_embedder.1.linear_1":          "time_embed.t_embedder.linear_1",
    "t_embedder.1.linear_2":          "time_embed.t_embedder.linear_2",
    "t_embedding_norm":               "time_embed.norm",
    "final_layer.linear":             "proj_out",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
}
_ATTN = {"q_proj": "to_q", "k_proj": "to_k", "v_proj": "to_v",
         "output_proj": "to_out.0", "q_norm": "norm_q", "k_norm": "norm_k"}
# self_attn → attn1（自注意力）、cross_attn → attn2（對文字的交叉注意力）
_BLOCK = {"self_attn": "attn1", "cross_attn": "attn2"}
# CosmosTransformerBlock 的 norm1/2/3 依序在 attn1、attn2、ff 之前
_ADALN = {"adaln_modulation_self_attn": "norm1",
          "adaln_modulation_cross_attn": "norm2",
          "adaln_modulation_mlp": "norm3"}
_MLP = {"layer1": "ff.net.0.proj", "layer2": "ff.net.2"}


def _rename(key):
    """net.* → CosmosTransformer3DModel 的鍵名，認不出來回 None"""
    stem, suffix = key.rsplit(".", 1)
    if stem in _TOP:
        return f"{_TOP[stem]}.{suffix}"
    m = re.match(r"blocks\.(\d+)\.(.+)$", stem)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    for src, dst in _BLOCK.items():
        if rest.startswith(src + "."):
            part = rest[len(src) + 1:]
            return (f"transformer_blocks.{i}.{dst}.{_ATTN[part]}.{suffix}"
                    if part in _ATTN else None)
    for src, dst in _ADALN.items():
        if rest.startswith(src + "."):
            return (f"transformer_blocks.{i}.{dst}."
                    f"linear_{rest[len(src) + 1:]}.{suffix}")
    if rest.startswith("mlp."):
        part = rest[4:]
        return (f"transformer_blocks.{i}.{_MLP[part]}.{suffix}"
                if part in _MLP else None)
    return None


def lora_alias(num_layers):
    """kohya 的模組名 → diffusers 的模組路徑

    Anima 的 LoRA 是拿 ComfyUI 拆分式的模型訓出來的（sd-scripts 的
    networks.lora_anima），所以鍵名沿用 net.* 那一套：

        lora_unet_blocks_0_cross_attn_k_proj
          → transformer_blocks.0.attn2.to_k

    融合機制本身不用改，只要給它一組別名就查得到。
    別名由上面同一組對照表生成，不會跟載入路徑走鐘。
    """
    out = {src.replace(".", "_"): dst for src, dst in _TOP.items()}
    for i in range(num_layers):
        for a_src, a_dst in _BLOCK.items():
            for p_src, p_dst in _ATTN.items():
                out[f"blocks_{i}_{a_src}_{p_src}"] = \
                    f"transformer_blocks.{i}.{a_dst}.{p_dst}"
        for n_src, n_dst in _ADALN.items():
            for n in ("1", "2"):
                out[f"blocks_{i}_{n_src}_{n}"] = \
                    f"transformer_blocks.{i}.{n_dst}.linear_{n}"
        for m_src, m_dst in _MLP.items():
            out[f"blocks_{i}_mlp_{m_src}"] = f"transformer_blocks.{i}.{m_dst}"
    return out


# ComfyUI 存檔（Save Checkpoint）的寫法：官方的 Aesthetic、Turbo 版是這樣存的，Base 版與 Preview 是 net.*。
# 後面的鍵名、形狀都一樣，讀進來先換成 net.*（net_keys）
ALT_PREFIX = "model.diffusion_model."


def is_anima(keys):
    """去噪網路（net.blocks）與 LLM 轉接層（net.llm_adapter）。ComfyUI 存檔的寫法只認轉接層：
    model.diffusion_model.blocks. 別的架構也有"""
    j = "\n".join(keys) if not isinstance(keys, str) else keys
    return "net.llm_adapter." in j or "net.blocks." in j or ALT_PREFIX + "llm_adapter." in j


def net_keys(raw):
    """權重的鍵名統一成 net.*（ComfyUI 存檔的 model.diffusion_model.* 換掉），不複製權重"""
    return {("net." + k[len(ALT_PREFIX):] if k.startswith(ALT_PREFIX) else k): v for k, v in raw.items()}


# ---------------------------------------------------------------- 超參數
def _transformer_config(sd):
    """從權重形狀推出 CosmosTransformer3DModel 的設定"""
    proj = sd["net.x_embedder.proj.1.weight"]
    model_channels = proj.shape[0]
    # patch 是 1×2×2，所以輸入通道 = shape[1]/4，再扣掉 padding mask 那一條
    in_channels = proj.shape[1] // 4 - 1
    n_blocks = 1 + max(int(k.split(".")[2]) for k in sd
                       if k.startswith("net.blocks."))
    head_dim = sd["net.blocks.0.self_attn.q_norm.weight"].shape[0]
    num_heads = sd["net.blocks.0.self_attn.q_proj.weight"].shape[0] // head_dim
    cross_dim = sd["net.blocks.0.cross_attn.k_proj.weight"].shape[1]
    adaln = sd["net.blocks.0.adaln_modulation_self_attn.1.weight"].shape[0]
    return dict(
        in_channels=in_channels, out_channels=in_channels,
        num_attention_heads=num_heads, attention_head_dim=head_dim,
        num_layers=n_blocks, mlp_ratio=4.0,
        text_embed_dim=cross_dim, adaln_lora_dim=adaln,
        patch_size=(1, 2, 2), concat_padding_mask=True,
        # rope_scale 是 (t, h, w)。Anima 的外插比例是 t=1、h=w=4，
        # 兩邊算 ntk_factor 的公式一模一樣，直接照填。
        rope_scale=(1.0, 4.0, 4.0),
        # 檢查點裡沒有 pos_emb_*，所以不要那組可學習的位置編碼
        extra_pos_embed_type=None,
    )


def _conditioner_config(sd):
    """從 llm_adapter 的形狀推出 AnimaTextConditioner 的設定"""
    p = "net.llm_adapter."
    emb = sd[p + "embed.weight"]
    n_layers = 1 + max(int(k.split(".")[3]) for k in sd
                       if k.startswith(p + "blocks."))
    head_dim = sd[p + "blocks.0.self_attn.q_norm.weight"].shape[0]
    model_dim = sd[p + "blocks.0.self_attn.q_proj.weight"].shape[1]
    return dict(
        source_dim=sd[p + "blocks.0.cross_attn.k_proj.weight"].shape[1],
        target_dim=sd[p + "out_proj.weight"].shape[0],
        model_dim=model_dim, num_layers=n_layers,
        num_attention_heads=sd[p + "blocks.0.self_attn.q_proj.weight"].shape[0]
        // head_dim,
        target_vocab_size=emb.shape[0],
    )


def _qwen3_config(sd):
    """從權重形狀推出 Qwen3 的設定（0.6B base）。

    形狀推不出 RoPE 的 theta：不給的話 transformers 用預設的 10000，Qwen3 全系列官方（ComfyUI 也是）是一百萬，
    文字的位置編碼就跟官方不同（審查紀錄 R41，2026-10-04 修正；之前 CoCo 畫的 Anima 圖因此重現不了，使用者決定以後的圖為重）"""
    from transformers import Qwen3Config
    emb = sd["embed_tokens.weight"]
    head_dim = sd["layers.0.self_attn.q_norm.weight"].shape[0]
    return Qwen3Config(
        vocab_size=emb.shape[0], hidden_size=emb.shape[1],
        intermediate_size=sd["layers.0.mlp.gate_proj.weight"].shape[0],
        num_hidden_layers=1 + max(int(k.split(".")[1]) for k in sd
                                  if k.startswith("layers.")),
        num_attention_heads=sd["layers.0.self_attn.q_proj.weight"].shape[0]
        // head_dim,
        num_key_value_heads=sd["layers.0.self_attn.k_proj.weight"].shape[0]
        // head_dim,
        head_dim=head_dim,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
    )


# ---------------------------------------------------------------- 載入
def _krea2():
    try:
        from core import krea2
    except ImportError:
        import krea2
    return krea2


def fp8_blocks(trunk):
    """主模型用 fp8 存放：每個 block 裡的矩陣（注意力、MLP、adaLN）存成 fp8 加一個比例（整個矩陣最大的值對到 fp8 的 448），
    算的時候還原（krea2 的 ScaledFP8Linear，LoRA 走旁路）。嵌入層、norm、最後一層照舊 bf16。
    不直接轉 fp8（diffusers 的 layerwise casting）：Anima 的權重有兩成多小到 fp8 只剩次正規數的精度，直接轉誤差 4.8%、
    畫風明顯變；加比例 2.7%（SDXL 直接轉 2.9%、加比例 2.7%，所以 SDXL 照舊直接轉，連卷積層也省）。
    回傳 (權重, {模組路徑: 比例})"""
    import torch
    out, keep = dict(trunk), {}
    for k, w in trunk.items():
        if not (k.startswith("transformer_blocks.") and k.endswith(".weight") and w.dim() == 2):
            continue
        s = float(w.float().abs().max()) / 448.0 or 1.0
        out[k] = (w.float() / s).to(torch.float8_e4m3fn)
        keep[k[:-len(".weight")]] = s
    return out, keep


def _build(cls, cfg, weights, label, log, prepare=None):
    """在 meta 上建好架構再把權重指派進去，避免先配置一份再覆蓋掉。

    include_buffers 預設是 False，非持久化的緩衝區（例如旋轉位置編碼的
    inv_freq）會照常在 CPU 上算好——那些不在檢查點裡，留在 meta 會當掉。
    prepare：指派權重之前對模型做的調整（Krea 2 把 fp8 的線性層換掉，見 krea2.swap_fp8）。
    """
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = cls(**cfg) if isinstance(cfg, dict) else cls(cfg)
    if prepare:
        prepare(model)
    miss = model.load_state_dict(weights, strict=False, assign=True)
    if miss.missing_keys or miss.unexpected_keys:
        raise RuntimeError(i18n.tr("eng.a.keys", label=label, miss=len(miss.missing_keys), extra=len(miss.unexpected_keys),
                              missList=miss.missing_keys[:3], extraList=miss.unexpected_keys[:3]))
    left = [n for n, t in list(model.named_parameters())
            + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(i18n.tr("eng.a.meta", label=label, n=len(left), names=left[:3]))
    log(label + "　" + i18n.tr("eng.a.tensors", n=len(weights)))
    return model


def lean_vae(vae):
    """單張圖編碼、解碼時不留影格快取（Qwen Image 的 VAE：Anima、Krea 2；Qwen Image 2.1 的 VAE）。
    這兩種 VAE 是從影片的 Wan VAE 來的，每一層都把輸入複製一份留給下一格（feat_cache），整個解碼完才清；
    一張圖只有一格，那些複製一份都用不到，卻讓解碼的顯示卡記憶體多一倍。
    只有一格時不給快取，每一層走的路跟第一格一樣（第一格本來就沒有前一格可以用），出圖逐位元相同。
    RTX 5080 解碼峰值：Qwen Image 1024×1408 5.7 → 2.8 GiB；2.1 1024×1344 8.6 → 4.8 GiB（1344×1792 解碼 7 → 1 秒）"""
    for part, inner_name in (("decoder", "_decode"), ("encoder", "_encode")):
        mod, inner = getattr(vae, part), getattr(vae, inner_name)

        def run(x, *a, _inner=inner, _dec=part == "decoder", **k):
            vae._coco_frames = x.shape[2]
            grow = _dec and _expandable(True)
            try:
                return _inner(x, *a, **k)
            finally:
                vae._coco_frames = None
                if grow:
                    _expandable(False)

        def forward(x, feat_cache=None, feat_idx=None, _fwd=mod.forward, **k):
            if getattr(vae, "_coco_frames", None) == 1:
                feat_cache = None
            return _fwd(x, feat_cache=feat_cache, feat_idx=feat_idx, **k)

        setattr(vae, inner_name, run)
        mod.forward = forward


def _expandable(on):
    """解碼時 PyTorch 的配置器改用可以延伸的區段（expandable_segments），解完換回來；回傳有沒有換。
    解碼的張量一層比一層大，一般的區段切不出來就再要一塊新的，舊的留著沒在用：2.1 的 1024×1344 實際用 4.8 GiB，
    留著的卻長到 7.8 GiB。Windows 的顯示卡驅動在顯示卡滿了時改借系統記憶體（工作管理員的「共用 GPU 記憶體」），
    配置永遠成功，PyTorch 就不會先把沒在用的還回去：主模型整個放在顯示卡時就擠出顯示卡，解碼 0.5 → 4 秒。
    可以延伸的區段同一塊接著長，只留 6.8 GiB。只在解碼時用：一直開著會影響其他模型，使用者自己設了就不動"""
    import torch
    if not torch.cuda.is_available() or "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return False
    setting = getattr(torch._C, "_accelerator_setAllocatorSettings", None) or torch.cuda.memory._set_allocator_settings
    try:
        setting(f"expandable_segments:{on}")
    except (AttributeError, RuntimeError):
        return False
    return True


def load(model_path, vae_path, te_path, log=print, dtype=None, fp8=False):
    """組出 Anima 的 modular pipeline

    三個路徑都是使用者在介面上選的單檔。
    fp8：主模型用 fp8 存放（⚙ 設定 → 生成），見 fp8_blocks。
    """
    import torch
    from safetensors.torch import load_file
    from diffusers import (AnimaTextConditioner, AutoencoderKLQwenImage,
                           CosmosTransformer3DModel,
                           FlowMatchEulerDiscreteScheduler)
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from diffusers import AnimaModularPipeline
    from diffusers.modular_pipelines.anima import AnimaAutoBlocks
    from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

    if dtype is None:
        dtype = torch.bfloat16

    if not vae_path or not te_path:
        raise RuntimeError(i18n.tr("eng.needParts", model="Anima", vae=i18n.tr("eng.vaeOf", name="Qwen Image", file="qwen_image_vae"), te=i18n.tr("srv.part.hint", name="Qwen3-0.6B", file="qwen_3_06b_base")))
    for p, what in ((vae_path, "VAE"), (te_path, i18n.tr("eng.teCap"))):
        if not Path(p).is_file():
            raise RuntimeError(i18n.tr("eng.fileMissing", what=what, p=p))

    def cast(d):
        return {k: v.to(dtype) if v.is_floating_point() else v
                for k, v in d.items()}

    log(i18n.tr("eng.loading", model="Anima"))

    # ---- transformer 與 text_conditioner 都在同一個檔案裡 ----
    raw = net_keys(load_file(str(model_path)))
    tcfg = _transformer_config(raw)
    log("　" + i18n.tr("eng.a.tf", layers=tcfg['num_layers'], heads=tcfg['num_attention_heads'], dim=tcfg['attention_head_dim']))

    trunk, unknown = {}, []
    for k, v in raw.items():
        if not k.startswith("net.") or k.startswith("net.llm_adapter."):
            continue
        d = _rename(k[4:])
        if d is None:
            unknown.append(k)
        else:
            trunk[d] = v.to(dtype)
    if unknown:
        raise RuntimeError(i18n.tr("eng.a.unknownKeys", n=len(unknown)) + "\n" + "\n".join(unknown[:5]))
    keep = {}
    if fp8:
        trunk, keep = fp8_blocks(trunk)
    transformer = _build(CosmosTransformer3DModel, tcfg, trunk,
                         "　transformer", log, prepare=(lambda m: _krea2().swap_fp8(m, keep)) if keep else None)
    if keep:
        log("　" + i18n.tr("eng.a.fp8", n=len(keep)))
    _fast_attention(transformer, log)

    ada = {k[len("net.llm_adapter."):]: v.to(dtype) for k, v in raw.items()
           if k.startswith("net.llm_adapter.")}
    conditioner = _build(AnimaTextConditioner, _conditioner_config(raw), ada,
                         "　text_conditioner", log)
    del raw, trunk, ada

    # ---- VAE：ComfyUI 命名，用 diffusers 現成的轉換器 ----
    vcfg = json.loads((ASSETS / "vae" / "config.json").read_text("utf-8"))
    import inspect
    accept = set(inspect.signature(AutoencoderKLQwenImage.__init__).parameters)
    vcfg = {k: v for k, v in vcfg.items()
            if not k.startswith("_") and k in accept}
    vae = _build(AutoencoderKLQwenImage, vcfg,
                 cast(convert_wan_vae_to_diffusers(load_file(str(vae_path)))),
                 "　vae", log)
    lean_vae(vae)

    # ---- 文字編碼器：標準 HF 命名，只要去掉 model. 前綴 ----
    te_sd = {k[6:]: v for k, v in load_file(str(te_path)).items()
             if k.startswith("model.")}
    if not te_sd:
        raise RuntimeError(i18n.tr("eng.notQwen3", name=Path(te_path).name))
    text_encoder = _build(Qwen3Model, _qwen3_config(te_sd), cast(te_sd),
                          "　text_encoder", log)
    del te_sd

    scfg = json.loads(
        (ASSETS / "scheduler" / "scheduler_config.json").read_text("utf-8"))
    scheduler = FlowMatchEulerDiscreteScheduler(
        **{k: v for k, v in scfg.items() if not k.startswith("_")})

    # 一定要用 AnimaModularPipeline 而不是通用的 ModularPipeline：
    # vae_scale_factor、default_sample_size 這些是它才有的屬性
    pipe = AnimaModularPipeline(blocks=AnimaAutoBlocks())
    pipe.update_components(
        transformer=transformer, text_conditioner=conditioner,
        text_encoder=text_encoder, vae=vae, scheduler=scheduler,
        tokenizer=Qwen2Tokenizer.from_pretrained(str(ASSETS / "tokenizer")),
        t5_tokenizer=T5TokenizerFast.from_pretrained(
            str(ASSETS / "tokenizer_2")),
    )
    pipe._coco_is_anima = True
    _hush(pipe)
    log("　" + i18n.tr("eng.assembled"))
    return pipe


# ---------------------------------------------------------------- 注意力
# 關掉就是 diffusers 原本的 CosmosAttnProcessor2_0。對照測試用。
FAST_ATTN = True


def _rms(x, norm):
    """跟 diffusers RMSNorm 逐位元相同，但 6 個 kernel 變 2 個

    diffusers 的寫法：升 fp32 → 平方 → 平均 → rsqrt → 相乘 → 捨入回 bf16
    → 乘權重。F.rms_norm 不帶權重時是一個融合 kernel，內部同樣用 fp32
    算、最後捨入一次，結果跟前五步完全相同；權重照原樣另外乘。
    **不能把權重交給 F.rms_norm**：那會在 fp32 裡先乘權重再捨入，
    少了中間那次 bf16 捨入，實測 26% 的元素差一個 bf16 單位。
    """
    import torch.nn.functional as F
    return F.rms_norm(x, (x.shape[-1],), None, norm.eps) * norm.weight


def _rope(x, emb):
    """跟 diffusers apply_rotary_emb（use_real_unbind_dim=-2）逐位元相同

    原式是 x·cos + cat(−x₂, x₁)·sin，全部在 fp32 算完再捨入回 bf16。
    Cosmos 的 cos / sin 是 cat([t, h, w] * 2) 組出來的，前後兩半相同，
    所以可以直接寫成
        前半 = x₁·cos − x₂·sin　　後半 = x₂·cos + x₁·sin
    每一個乘法、加法的捨入都跟原式一樣（取負是精確的），但省掉取負、
    串接與兩次升 fp32 的複製：一個 [1, 16, 5632, 128] 的張量從 8 趟
    讀寫變成 4 趟。最後一步直接寫進 bf16 的輸出，運算仍在 fp32 完成。
    """
    import torch
    cos, sin = emb
    h = x.shape[-1] // 2
    p = x * cos[None, None]          # bf16 × fp32 → fp32，升精度是精確的
    q = x * sin[None, None]
    out = torch.empty_like(x)
    torch.sub(p[..., :h], q[..., h:], out=out[..., :h])
    torch.add(p[..., h:], q[..., :h], out=out[..., h:])
    return out


class FastCosmosAttnProcessor:
    """CosmosAttnProcessor2_0 的等價版本，出圖逐位元相同

    Anima 每步約四分之一的時間花在注意力周邊的小運算上（RTX 5080、
    1024×1408 一次前向：RoPE 約 33 ms、q/k 的 RMSNorm 約 25 ms，
    整次前向 340 ms）。ComfyUI 用自己的融合 kernel；這裡只重排
    PyTorch 的運算，不引入新的相依。

    另外拿掉了原版的 repeat_interleave：它的倍數取的是 head_dim 的比值，
    q、k 同維度時恆為 1，卻每次都複製一份 K 和 V。
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        q = _rms(q, attn.norm_q)
        k = _rms(k, attn.norm_k)
        if image_rotary_emb is not None:
            q = _rope(q, image_rotary_emb)
            k = _rope(k, image_rotary_emb)
        if k.size(3) != q.size(3):
            k = k.repeat_interleave(q.size(3) // k.size(3), dim=3)
            v = v.repeat_interleave(q.size(3) // v.size(3), dim=3)

        out = dispatch_attention_fn(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        out = out.flatten(2, 3).type_as(q)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


def _fast_attention(transformer, log):
    """符合前提的注意力層換成 FastCosmosAttnProcessor，回傳換了幾個

    前提逐一檢查，任何一個不成立就保留原版：
      - 原本是 CosmosAttnProcessor2_0（Anima 用的那種）
      - q/k norm 是權重為 bf16 / fp16、沒有 bias 的 RMSNorm
        （fp32 權重時 diffusers 不會先捨入，_rms 的等價性不成立）
      - 位置編碼是 CosmosRotaryPosEmbed（_rope 依賴它前後兩半相同）
    """
    import torch
    if not FAST_ATTN or type(getattr(transformer, "rope", None)).__name__ \
            != "CosmosRotaryPosEmbed":
        return 0
    n = 0
    for m in transformer.modules():
        proc = getattr(m, "processor", None)
        if type(proc).__name__ != "CosmosAttnProcessor2_0":
            continue
        norms = [getattr(m, "norm_q", None), getattr(m, "norm_k", None)]
        if not all(type(x).__name__ == "RMSNorm"
                   and getattr(x, "bias", None) is None
                   and getattr(x, "weight", None) is not None
                   and x.weight.dtype in (torch.float16, torch.bfloat16)
                   for x in norms):
            continue
        m.processor = FastCosmosAttnProcessor()
        n += 1
    if n:
        log("　" + i18n.tr("eng.a.fused", n=n))
    return n


class _Bar:
    """假裝成 tqdm，把去噪迴圈的每一步轉成 on_step 回呼

    去噪迴圈寫的是 `with self.progress_bar(total=n) as bar: ... bar.update()`，
    所以把那個工廠換掉就能拿到進度，不必動 diffusers 的程式碼。
    """

    def __init__(self, total, pipe):
        self.total, self.n, self.pipe = total or 0, 0, pipe

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, n=1):
        self.n += n
        cb = getattr(self.pipe, "_coco_on_step", None)
        if cb:
            try:
                cb(min(self.n, self.total), self.total)
            except Exception:
                pass

    def close(self):
        pass


def _hush(pipe):
    """關掉 tqdm，並把進度接到 on_step

    pipe.set_progress_bar_config 只會傳給第一層的子區塊，
    但去噪迴圈在更裡面，所以要自己走完整棵樹。
    不接管的話那條進度條會從 stderr 灌進介面的「紀錄」面板，
    而且工作台的進度列不會動。
    """
    seen = set()

    def walk(b):
        if id(b) in seen:
            return
        seen.add(id(b))
        if hasattr(b, "progress_bar"):
            b.progress_bar = (lambda total=None, iterable=None, _p=pipe:
                              _Bar(total, _p))
        for sub in getattr(getattr(b, "sub_blocks", None), "values",
                           lambda: [])():
            walk(sub)

    walk(getattr(pipe, "_blocks", None) or pipe)


# ---------------------------------------------------------------- 排程器
# FlowMatchEuler 認得的 sigma 分佈，跟 SDXL 那張 SCHEDULES 表對應
_SIGMAS = {"Karras": {"use_karras_sigmas": True},
           "Exponential": {"use_exponential_sigmas": True},
           "Beta": {"use_beta_sigmas": True}}


# 舊名字 → 新名字。**不能拿掉**：已經產出的圖，中繼資料裡寫的是舊名，
# 拿掉之後那些圖就再也重現不了（會靜靜落到 FlowMatch Euler，不會報錯）。
#
# `kdiff` 那組改名不只是美觀問題：那個名字是在還沒讀 ComfyUI 原始碼時取的，
# 當時誤以為它就是 k-diffusion 的公式。實際讀完發現 ComfyUI 對 flow 模型走
# 的是另一條式子（見 _REFERENCE_VARIANTS），那支其實是本專案自己的推導，
# 所以改叫 CoCo。`ER-SDE ODE` 則是自相矛盾（SDE 又 ODE），改成 ER-ODE。
#
# 2026-10-03 改名（使用者：名字不要那麼容易混淆）：Euler a 在 Anima 上改用跟 Forge／ComfyUI 一樣的 RF 算法，
# 原本「Euler a RF」那支就是它；CoCo 以前的 Euler a（每步把雜訊整個換新）改叫 Euler a renoise，
# 照它改的那幾支也都掛 renoise。舊名字照樣認得：除了「Euler a」本身，舊圖都重現得了。
_ALIASES = {
    "euler a kdiff": "euler a coco",
    "euler a kdiff eta 0.5": "euler a coco 0.5",
    "euler a rf": "euler a",
    "euler a rf 0.5": "euler a 0.5",
    "euler a rf eta 0.5": "euler a 0.5",
    "er-sde ode": "er-ode",
    "euler a eta 0.25": "euler a renoise 0.25",
    "euler a eta 0.5": "euler a renoise 0.5",
    "euler a late-off": "euler a renoise late",
    "euler a late-off 0.5": "euler a renoise late-half",
    "euler a eta 0.5 late-off": "euler a renoise 0.5 late",
    # 更早以前存圖寫的是實際的排程器（_coco_sched_name）
    "flowmatch euler a": "euler a renoise",
    "flowmatch euler sde": "euler a renoise",
    "flowmatch euler": "euler",
    "flowmatch heun": "heun",
}


def _record(pipe, sampler, schedule):
    """存圖用的名字：介面上選的取樣器、實際有套用的排程（A1111／Forge 的寫法，分開兩欄）。
    _coco_sched_name 另外記實際的排程器（例如選 LMS、實際是 FlowMatch Euler），給紀錄與比較工具看。
    以前圖裡寫的是後者（「FlowMatch Euler a + Karras」），送到生成區讀不回來"""
    pipe._coco_sampler = sampler
    pipe._coco_schedule = schedule if schedule in _SIGMAS else "Normal"


def _is_ancestral(name):
    """名字判定：Euler a、DPM2 a 這類 ancestral，以及沒有專屬實作的 SDE

    FlowMatchEuler 的 `stochastic_sampling` 做的是

        x₀ = x − σ·v;   x' = (1−σ_next)·x₀ + σ_next·ε   （ε 是全新雜訊）

    先完全去噪到 x₀，再把雜訊加回下一個層級——**這正是 ancestral 的定義**。
    真正的 SDE 解算器（例如 ER-SDE）會保留一部分原本的軌跡雜訊，這裡沒有。
    diffusers 對 FlowMatchEuler 只提供這一種隨機模式，所以「Euler a」與
    「Euler SDE」在 Anima 上是同一個東西，統一標成 a，
    compare_samplers 的重複偵測才抓得到它們出圖相同。

    結尾的 " a" 要用整個 token 比對，不能用 `"a" in name`——那會把
    sa-solver 之類的名字一起吃掉。
    """
    return name.endswith(" a") or "ancestral" in name or "sde" in name


def _takes_sigmas(cls):
    import inspect
    try:
        return "sigmas" in inspect.signature(cls.set_timesteps).parameters
    except Exception:
        return False


def _bind_scheduler_generator(scheduler):
    """modular Anima loop 沒把 generator 傳給 scheduler.step，補上這層轉接。"""
    import inspect
    if getattr(scheduler, "_coco_generator_bound", False):
        return scheduler
    try:
        if "generator" not in inspect.signature(scheduler.step).parameters:
            return scheduler
    except (TypeError, ValueError):
        return scheduler
    inner = scheduler.step

    def step(*args, **kwargs):
        if kwargs.get("generator") is None:
            generator = getattr(scheduler, "_coco_generator", None)
            if generator is not None:
                kwargs["generator"] = generator
        return inner(*args, **kwargs)

    scheduler.step = step
    scheduler._coco_generator_bound = True
    return scheduler


# 介面上的名字 → (samplers 裡的類別名, 額外 config)。鍵是小寫，
# 因為 set_scheduler 一開始就把 sampler 轉成小寫比對。
# 對照 pipeline.SAMPLERS 那張 SDXL 的表，只收有 flow 版的那幾支。
_FLOW_SOLVERS = {
    "dpm++ 2m": ("FlowDPMSolverMultistepScheduler", {}),
    "dpm++ 2m sde": ("FlowDPMSolverMultistepScheduler",
                     {"algorithm_type": "sde-dpmsolver++"}),
    "dpm++ 3m sde": ("FlowDPMSolverMultistepScheduler",
                     {"algorithm_type": "sde-dpmsolver++",
                      "solver_order": 3}),
    # 跟 pipeline.SAMPLERS 一致：diffusers 的 Singlestep 沒有 sde
    # algorithm_type，所以這格其實是確定性的，名字裡的 SDE 是沿用
    # A1111 的叫法。這是既有的近似，不是這次改出來的。
    "dpm++ sde": ("FlowDPMSolverSinglestepScheduler", {}),
    "deis": ("FlowDEISMultistepScheduler", {}),
    "sa-solver": ("FlowSASolverScheduler", {}),
}

# 這幾支都是 VP 參數化的 sigma 分佈轉換，套在 flow sigma 上是錯的
# （理由跟 UniPC 那邊一樣，見 _set_unipc 的註解）。
_FLOW_SOLVER_BAD_SCHEDULES = ("Karras", "Exponential", "Beta")


def _set_flow_solver(pipe, name, schedule, steps, sh, log):
    """DPM++ / DEIS / SA-Solver 的 flow 版（R30）

    這幾支 diffusers 都支援 `use_flow_sigmas` + `flow_prediction`，
    解算器本身對 Anima 完全適用，卡的只有 set_timesteps 不吃 sigmas。
    samplers._AcceptsSigmas 補上那層之後就能直接用——包含 shift 的套法，
    所以跟 FlowMatch Euler、UniPC 的 sigma 排程完全對齊，
    比出來的差異只來自解算器。詳見 samplers.py 裡的說明。
    """
    try:
        from core import samplers
    except ImportError:
        import samplers

    cls_name, extra = _FLOW_SOLVERS[name]
    cls = getattr(samplers, cls_name)

    if schedule in _FLOW_SOLVER_BAD_SCHEDULES:
        log("[!] " + i18n.tr("eng.a.vpSchedule", name=name, sch=schedule))

    sched = cls(num_train_timesteps=steps,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=sh,
                **extra)
    pipe.update_components(scheduler=_bind_scheduler_generator(sched))
    used = _FLOW_SOLVERS_LABEL.get(name, name)
    pipe._coco_sched_name = used
    _record(pipe, used, "Normal")
    pipe._coco_shift = sh
    log(i18n.tr("eng.a.sched", used=used, sh=sh))


# ancestral 的變體（R32）。名字全用 ASCII，中繼資料才能跟其他工具互通，
# 而且餵回來要解析得回同一支（有測試釘 round-trip）。
#
#   eta       0 = 確定性 Euler，1 = diffusers 原本的 stochastic_sampling
#   late_off  最後這個比例的步驟走確定性，不再注入雜訊
#
# late_off 用**比例**不用 sigma 門檻：原本寫 s_tmin=0.15，實測 16 步
# 以下一步都關不到（整個參數是 no-op），20–35 步也只關到 1 步，
# 同一個設定在不同步數下意義完全不同。0.25 則不管幾步都是「最後
# 四分之一安靜」。詳見 samplers.FlowMatchEulerAncestralScheduler。
#
# eta 0.25 與 late-off 0.5 這兩個點是實測回饋加上去的：使用者觀察到
# ancestral 變體「衣服細節比其他取樣器弱」，合成量測也看到銳利度隨 eta
# 單調下降（η=0.25 −9%、0.5 −31%、1.0 −77%），也就是**隨機性一路在拿
# 細節換東西**。但 eta=1 + late-off 那支同時「構圖像 Euler SDE」又「比
# 沒 late-off 時清楚」，說明分工是可行的：前段隨機決定構圖、後段確定性
# 保住細節。所以往「拉長確定性尾段」推，而不是往 eta 中間值推。
_ANCESTRAL_VARIANTS = {
    # renoise 系：每步把雜訊換新（數字是換掉的比例），late＝最後 1/4 步不加新雜訊、late-half＝最後一半
    "euler a renoise 0.25": ("Euler a renoise 0.25", {"eta": 0.25}),
    "euler a renoise 0.5": ("Euler a renoise 0.5", {"eta": 0.5}),
    "euler a renoise late": ("Euler a renoise late",
                             {"eta": 1.0, "late_off": 0.25}),
    "euler a renoise late-half": ("Euler a renoise late-half",
                                  {"eta": 1.0, "late_off": 0.5}),
    "euler a renoise 0.5 late": ("Euler a renoise 0.5 late",
                                 {"eta": 0.5, "late_off": 0.25}),
    # k-diffusion 系（ComfyUI / A1111 / Forge 共用的那套）的注入量，
    # 多了一個步長因子 √(1−(σ_t/σ_s)²)。合成量測：20 步下銳利度是
    # diffusers 那條的 3.4 倍，而且不隨步數崩潰（見 samplers.py 的說明）。
    # 照公開演算法自己推的，沒有沿用任何 GPL/AGPL 專案的程式碼。
    "euler a coco": ("Euler a CoCo", {"eta": 1.0, "step_aware": True}),
    "euler a coco 0.5": ("Euler a CoCo 0.5",
                         {"eta": 0.5, "step_aware": True}),
}


# ER-SDE 自己的旋鈕（R33）。`FlowMatchERSDEScheduler` 一直支援
# max_stage 與 s_noise，但 set_scheduler 從來沒把它們接出來，
# 於是永遠是 3 階 + 全額雜訊。
#
# **不要用 s_noise 當隨機性的旋鈕。** 第一版做了 `s_noise 0.5` 與
# `det`（s_noise=0），實測兩張都是純色圖：收縮係數 r 是假設全額雜訊會被
# 補回來才推導出來的（大步時 r 可以小到 0.08），只砍雜訊就留下沒有補償
# 的收縮，高頻結構被壓掉。要確定性版本得換 φ(λ)=λ，見 _er_phi_identity。
#
# **`ER-ODE` 是診斷用的關鍵點。** ER-SDE 的雜訊項本來就會隨步長
# 自動縮小，所以它同時銳利又穩定。換成純 ODE 版之後如果手部表現仍然好，
# 代表功勞在**3 階的精度**而不是隨機性，那整個搜尋方向就要改。
# 階數用 1 對照而不是 2：實測 3 階相對 2 階只差 0.24–0.43%（3 階本來就
# 只是微修正），2 階那格在選單裡等於死重；1 階才差得出來（約 5%）。
#
# 這三支合起來是一組對照，用來回答「ER-SDE 的優勢來自階數還是隨機性」：
#   ER-SDE      3 階 + 雜訊    基準
#   ER-ODE      3 階 + 無雜訊  只拿掉隨機性
#   ER-SDE 1    1 階 + 雜訊    只拿掉階數
_ERSDE_VARIANTS = {
    "er-sde": ("ER-SDE", {}),
    "er-sde 1": ("ER-SDE 1", {"max_stage": 1}),
    "er-ode": ("ER-ODE", {"deterministic": True}),
}


# R34：照 ComfyUI / Neo Forge 的演算法實作的對照組（2026-10-03 只照公式重寫程式，見 samplers.py）。
# 目的是在**同一支程式、同一個 seed** 底下跟自家推導 A/B，
# 跨程式比會混進提示詞編碼、VAE、CFG 實作等變因，回答不了問題。
#
# 讀過兩邊的原始碼：**Neo Forge 與 ComfyUI 的 k_diffusion 是同一份**，
# 只差排版、註解與取 model_sampling 的方式，所以只有一份參考實作。
_REFERENCE_VARIANTS = {
    # ComfyUI 對 flow 模型走的是專屬的 sample_euler_ancestral_RF，
    # 跟本專案 _ANCESTRAL_VARIANTS 那套的公式**不一樣**（見 samplers.py）。
    # 2026-10-03 起 Anima 的「Euler a」就是這支（跟 Forge／ComfyUI 一樣），以前叫 Euler a RF
    "euler a": ("Euler a", "FlowMatchEulerAncestralRFScheduler",
                {"eta": 1.0}),
    "euler a 0.5": ("Euler a 0.5",
                    "FlowMatchEulerAncestralRFScheduler",
                    {"eta": 0.5}),
    # 它們的 sample_euler 帶 s_churn，是 Karras et al. 2022 Algorithm 2：
    # 先把 sigma 往上推再用更大的確定性步走下來，跟 ancestral 是不同機制。
    "euler churn": ("Euler Churn", "FlowMatchEulerChurnScheduler",
                    {"s_churn": 4.0}),
    # ER-SDE 照它們的 VE 空間寫法重寫一份。實測跟自家的 flow 空間版
    # 差 3e-7（float32 湊整），也就是兩份推導互相驗證通過。
    "er-sde ve": ("ER-SDE ve", "ERSDEVEScheduler", {}),
}


def _set_reference_variant(pipe, name, schedule, steps, sh, log):
    """ComfyUI / Neo Forge 演算法的對照組（R34）"""
    try:
        from core import samplers
    except ImportError:
        import samplers

    label, cls_name, extra = _REFERENCE_VARIANTS[name]
    cls = getattr(samplers, cls_name)
    kw = dict(_SIGMAS.get(schedule or "", {}))
    sched = cls(num_train_timesteps=steps, shift=sh, **kw, **extra)
    pipe.update_components(scheduler=_bind_scheduler_generator(sched))
    used = label + (f" + {schedule}" if schedule in _SIGMAS else "")
    pipe._coco_sched_name = used
    _record(pipe, label, schedule)
    pipe._coco_shift = sh
    log(i18n.tr("eng.a.sched", used=used, sh=sh))


def _set_ancestral_variant(pipe, name, schedule, steps, sh, log):
    """可調 eta／注入範圍的 ancestral Euler（R32）

    這幾支是 FlowMatchEuler 的子類別，所以 Karras / Exponential / Beta
    照樣能用——父類別是「先套 shift 再轉分佈」的串接式做法，
    不像 UniPC 那樣互斥（見 _set_unipc 的說明）。
    """
    try:
        from core.samplers import FlowMatchEulerAncestralScheduler
    except ImportError:
        from samplers import FlowMatchEulerAncestralScheduler

    label, kw = _ANCESTRAL_VARIANTS[name]
    sched = FlowMatchEulerAncestralScheduler(
        num_train_timesteps=steps, shift=sh,
        **dict(_SIGMAS.get(schedule or "", {})), **kw)
    pipe.update_components(scheduler=_bind_scheduler_generator(sched))
    used = label + (f" + {schedule}" if schedule in _SIGMAS else "")
    pipe._coco_sched_name = used
    _record(pipe, label, schedule)
    pipe._coco_shift = sh
    log(i18n.tr("eng.a.sched", used=used, sh=sh))


# 印出來的名字。跟輸入一樣時 compare_samplers 就不會標成「實際 X」。
_FLOW_SOLVERS_LABEL = {
    "dpm++ 2m": "DPM++ 2M",
    "dpm++ 2m sde": "DPM++ 2M SDE",
    "dpm++ 3m sde": "DPM++ 3M SDE",
    "dpm++ sde": "DPM++ SDE",
    "deis": "DEIS",
    "sa-solver": "SA-Solver",
}


def _set_unipc(pipe, schedule, steps, sh, log):
    """UniPC，flow-match 版

    **為什麼只有 UniPC 接得上。** diffusers 0.39 的多步解算器裡，
    DPMSolverMultistep / DPMSolverSinglestep / DEIS / SA-Solver 都支援
    use_flow_sigmas，但它們的 set_timesteps **不吃 sigmas**（只吃
    timesteps）。Anima 的 before_denoise 一律會餵 sigmas，而
    retrieve_timesteps 對不收 sigmas 的排程器是直接丟例外，所以那幾支
    現在接不上。UniPC 是唯一簽名有 sigmas 的，可以原樣放行。

    **參數為什麼是這幾個。** prediction_type="flow_prediction" 讓
    convert_model_output 走 x₀ = x − σ·v，跟 flow matching 的定義一致
    （也跟 samplers.FlowMatchERSDEScheduler 的算法相同）；
    use_flow_sigmas=True 是 UniPC 允許外部餵 sigmas 的前提條件；
    flow_shift 套的公式是 shift·σ/(1+(shift−1)·σ)，跟
    FlowMatchEulerDiscreteScheduler.set_timesteps 第 2 步逐字相同，
    所以它就是介面上那個 shift，兩個取樣器的 sigma 排程完全對齊，
    比較出來的差異才只來自解算器本身。

    solver_order 用 diffusers 的預設 2（文件建議：有 CFG 用 2，
    無條件生成才用 3）。UniPC 是確定性解算器，step() 沒有 generator，
    _bind_scheduler_generator 會偵測到並原樣回傳。
    """
    from diffusers import UniPCMultistepScheduler

    if schedule in _SIGMAS:
        # UniPC 的 karras / exponential / beta 跟 use_flow_sigmas 在
        # set_timesteps 裡是 if/elif **互斥**的，跟 FlowMatchEuler
        # 「先套 shift 再轉分佈」的串接式做法不一樣：
        #
        #   karras       完全不套 flow_shift，而且轉完會再做一次
        #                σ/(σ+1)，把起始 sigma 從 1.0 壓到 0.5——
        #                等於一開始就當作圖已經去噪一半，出圖是糊的
        #   exponential  同樣不套 shift，且直接拿 flow sigma 去查
        #                _sigma_to_t（那支是給 VP 參數化用的），結果無意義
        #   beta         同上
        #
        # 三條都不是「差一點」而是算錯，所以擋掉而不是靜靜放行。
        log("[!] " + i18n.tr("eng.a.unipcSchedule", sch=schedule))

    sched = UniPCMultistepScheduler(
        num_train_timesteps=steps,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=sh,
        solver_order=2,
        final_sigmas_type="zero",
    )
    pipe.update_components(scheduler=_bind_scheduler_generator(sched))
    pipe._coco_sched_name = "UniPC"
    _record(pipe, "UniPC", "Normal")
    pipe._coco_shift = sh
    log(i18n.tr("eng.a.unipc", sh=sh))


def set_scheduler(pipe, sampler=None, schedule=None, shift=None, log=print, default_shift=None):
    """把介面上的取樣器／排程器選擇對應到 flow-match 的等價設定

    Anima 只吃 flow-match，但 FlowMatchEuler 本身認得 Karras / Exponential /
    Beta 三種 sigma 分佈，也有 stochastic_sampling（SDE 版），
    所以現有的兩個下拉選單對 Anima 仍然有意義，不必另做一套 UI。

    UniPC 可以直接接（見 _set_unipc）。DPM++ / DEIS / SA-Solver 的
    set_timesteps 不吃 sigmas，由 samplers._AcceptsSigmas 補上那層再接
    （見 _set_flow_solver）。所以現在只剩 LMS、DDIM、LCM、TCD 這些
    沒有 flow 版的會落到 FlowMatch Euler；名字是 ancestral 的（Euler a、
    DPM2 a）則落到開了 stochastic_sampling 的 FlowMatch Euler a。

    shift 是 Anima 特有且影響很大的旋鈕（實測換個值出圖差異 30 以上），
    留空就用 assets/anima/scheduler 裡的官方值 3.0。

    Z-Image 也走這裡（同樣是 flow matching），default_shift 給它自己的官方值（Base 6.0）。
    """
    from diffusers import (FlowMatchEulerDiscreteScheduler,
                           FlowMatchHeunDiscreteScheduler)

    cfg = json.loads(
        (ASSETS / "scheduler" / "scheduler_config.json").read_text("utf-8"))
    steps = int(cfg.get("num_train_timesteps", 1000))
    base = float(default_shift if default_shift is not None else cfg.get("shift", 3.0))
    try:
        sh = float(shift) if shift not in (None, "") else base
    except (TypeError, ValueError):
        sh = base

    # Beta 分佈要 scipy（跟 pipeline.beta_available 同一個判斷）。沒有的話 Euler 系
    # 的排程器一建構就丟 ImportError，整個生成失敗；在這裡先換成 Normal 並講清楚。
    if schedule == "Beta":
        import importlib.util
        if importlib.util.find_spec("scipy") is None:
            log("[!] " + i18n.tr("eng.betaMissing"))
            schedule = "Normal"

    name = _ALIASES.get((sampler or "").lower(), (sampler or "").lower())
    # 要放在 er-sde 判斷之前：「ER-SDE ve」含 er-sde，會被那條先吃掉。
    if name in _REFERENCE_VARIANTS:
        _set_reference_variant(pipe, name, schedule, steps, sh, log)
        return
    # 查表要放在子字串判斷**之前**：ER-ODE 改名之後就不含 er-sde 了，
    # 只靠子字串會整條漏掉（而且不會報錯，會靜靜落到 FlowMatch Euler）。
    if name in _ERSDE_VARIANTS or "er-sde" in name or "er_sde" in name:
        try:
            from core.samplers import FlowMatchERSDEScheduler
        except ImportError:
            from samplers import FlowMatchERSDEScheduler
        label, extra = _ERSDE_VARIANTS.get(name, ("ER-SDE", {}))
        # Karras／Exponential／Beta 照樣能用（父類別是 FlowMatch Euler，先套 shift 再轉分佈）。
        # 以前這裡沒傳、一律 Normal：Forge 的「ER SDE + Beta」送過來畫出來就不一樣
        sched = FlowMatchERSDEScheduler(num_train_timesteps=steps, shift=sh,
                                        **dict(_SIGMAS.get(schedule or "", {})), **extra)
        pipe.update_components(scheduler=_bind_scheduler_generator(sched))
        used = label + (f" + {schedule}" if schedule in _SIGMAS else "")
        pipe._coco_sched_name = used
        _record(pipe, label, schedule)
        pipe._coco_shift = sh
        log(i18n.tr("eng.a.schedStage", used=used, stage=sched.max_stage, noise=f"{sched.s_noise:g}", sh=sh))
        return
    if "unipc" in name:
        _set_unipc(pipe, schedule, steps, sh, log)
        return
    # 這個查表要放在下面的 "sde" 關鍵字判斷**之前**，否則
    # 「DPM++ 2M SDE」會先被當成 Euler SDE 吃掉。
    if name in _FLOW_SOLVERS:
        _set_flow_solver(pipe, name, schedule, steps, sh, log)
        return
    # 同理要放在 _is_ancestral 之前，否則「Euler a eta 0.5」會先被
    # 當成普通的 Euler a（結尾不是 " a"，但含不含 sde 不是重點——
    # 真正的風險是將來加了含 " a" 結尾的變體名）。
    if name in _ANCESTRAL_VARIANTS:
        _set_ancestral_variant(pipe, name, schedule, steps, sh, log)
        return
    # Anima 的 before_denoise 一律會餵自訂 sigmas（沒給就自己算 linspace），
    # 所以 set_timesteps 不吃 sigmas 的排程器根本用不了。
    # FlowMatchHeun 就是這種，選了會在去噪那一步才炸。
    if "heun" in name and _takes_sigmas(FlowMatchHeunDiscreteScheduler):
        sched = FlowMatchHeunDiscreteScheduler(num_train_timesteps=steps,
                                               shift=sh)
        used = "FlowMatch Heun"
    elif _is_ancestral(name) and name != "euler a renoise":
        # DPM2 a 這類 Anima 沒有專屬實作的 ancestral：跟 Euler a 一樣走 RF（以前是每步雜訊整個換新那支）
        _set_reference_variant(pipe, "euler a", schedule, steps, sh, log)
        _record(pipe, (sampler or "").strip(), pipe._coco_schedule)
        return
    else:
        if "heun" in name:
            log("[!] " + i18n.tr("eng.a.heun"))
        kw = dict(_SIGMAS.get(schedule or "", {}))
        renoise = name == "euler a renoise"   # CoCo 以前的 Euler a：diffusers 的 stochastic_sampling
        if renoise:
            kw["stochastic_sampling"] = True
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=steps,
                                                shift=sh, **kw)
        used = "FlowMatch Euler" + (" a" if renoise else "")
        if schedule in _SIGMAS:
            used += f" + {schedule}"
    pipe.update_components(scheduler=_bind_scheduler_generator(sched))
    pipe._coco_sched_name = used
    picked = "Euler a renoise" if name == "euler a renoise" else (sampler or "").strip() or "Euler"
    _record(pipe, picked, schedule if used.endswith(f" + {schedule}") else "Normal")
    pipe._coco_shift = sh
    log(i18n.tr("eng.a.sched", used=used, sh=sh))


# ---------------------------------------------------------------- 提示詞權重
def _parse(text):
    """用 conditioning.py 的權重語法解析（跟 SDXL 同一套，純文字處理、跟模型無關）"""
    try:
        from core import conditioning
    except ImportError:
        import conditioning
    return conditioning.parse_attention(text or "")


def weigh_prompt(tokenizer, text, max_len=512):
    """回傳 (去掉語法標記的提示詞, 逐 token 權重)

    Anima 的權重要乘在 text_conditioner 的輸出上，而那是以 T5 的 token
    位置對齊的（做法與 Neo Forge 的 anima_engine 相同）。

    對齊不靠「分段各自 tokenize 再接起來」——sentencepiece 在段落邊界的
    切法會變，接起來不保證等於整句的切法。改用 fast tokenizer 的字元
    偏移：每個 token 落在哪個片段的字元範圍內，就拿那個片段的權重。
    """
    parsed = [(t, w) for t, w in _parse(text) if not (t == "BREAK" and w == -1)]
    clean, spans, pos = "", [], 0
    for t, w in parsed:
        clean += t
        spans.append((pos, pos + len(t), w))
        pos += len(t)

    if all(abs(w - 1.0) < 1e-6 for _, _, w in spans):
        return clean, None                    # 沒有加權，不必多做事

    enc = tokenizer(clean, padding="longest", max_length=max_len,
                    truncation=True, return_offsets_mapping=True)
    weights = []
    for start, end in enc["offset_mapping"]:
        if start == end:                      # 特殊 token（</s> 之類）
            weights.append(1.0)
            continue
        w = 1.0
        for a, b, ww in spans:
            if a <= start < b:
                w = ww
                break
        weights.append(w)
    return clean, weights


def _install_weights(pipe):
    """讓 text_conditioner 的輸出依 T5 token 位置加權

    nn.Module.__call__ 會去查 self.forward，所以在實例上蓋一個同名屬性
    就能攔截，不必動 diffusers 的類別。
    正負提示詞是分兩次呼叫的，用 t5 ids 本身比對來認人，比記呼叫順序可靠。
    """
    import torch
    tc = pipe.text_conditioner
    if getattr(tc, "_coco_wrapped", False):
        return
    inner = tc.forward

    def forward(source_hidden_states, target_input_ids, **kw):
        out = inner(source_hidden_states=source_hidden_states,
                    target_input_ids=target_input_ids, **kw)
        for ids, w in getattr(pipe, "_coco_weights", []):
            if (ids.shape != target_input_ids.shape[-len(ids.shape):]
                    or not torch.equal(ids.to(target_input_ids.device),
                                       target_input_ids.reshape(ids.shape))):
                continue
            # conditioner 會把輸出往後補到 min_sequence_length（預設 512），
            # 補出來的位置是零，權重給 1 即可
            v = w.to(out.device, out.dtype)
            if v.numel() < out.shape[1]:
                v = torch.cat([v, torch.ones(out.shape[1] - v.numel(),
                                             device=out.device,
                                             dtype=out.dtype)])
            return out * v[:out.shape[1]].view(1, -1, 1)
        return out

    tc.forward = forward
    tc._coco_wrapped = True


def _component_device(module):
    try:
        return next(module.parameters()).device.type
    except (AttributeError, StopIteration):
        return ""


def _move_components(pipe, names, device, log):
    """搬移 Anima 元件並回傳實際移動的名稱。"""
    import torch
    moved = []
    for name in names:
        module = getattr(pipe, name, None)
        if module is None or _component_device(module) == device:
            continue
        try:
            module.to(device)
            moved.append(name)
        except Exception as e:
            log("[!] " + i18n.tr("eng.a.moveFail", name=name, device=device, e=f"{type(e).__name__}: {str(e)[:70]}"))
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def _install_memory_hooks(pipe):
    """在 modular pipeline 的階段邊界執行 balanced 元件交接。"""
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and not getattr(
            transformer, "_coco_balance_wrapped", False):
        inner_forward = transformer.forward

        def forward(*args, **kwargs):
            if getattr(pipe, "_coco_vram", "speed") in ("balanced", "low"):
                log = getattr(pipe, "_coco_log", print)
                moved = _move_components(
                    pipe, ("text_encoder", "text_conditioner"), "cpu", log)
                if moved:
                    log(i18n.tr("eng.a.teOut", names=i18n.join(moved)))
            return inner_forward(*args, **kwargs)

        transformer.forward = forward
        transformer._coco_balance_wrapped = True

    vae = getattr(pipe, "vae", None)
    if vae is not None and not getattr(vae, "_coco_balance_wrapped", False):
        inner_decode = vae.decode

        def decode(*args, **kwargs):
            if getattr(pipe, "_coco_vram", "speed") == "balanced":
                log = getattr(pipe, "_coco_log", print)
                moved = _move_components(pipe, ("transformer",), "cpu", log)
                if moved:
                    log(i18n.tr("eng.a.balancedOut"))
            return inner_decode(*args, **kwargs)

        vae.decode = decode
        vae._coco_balance_wrapped = True


def parked(vram_mode):
    """待機時放在 CPU、要用時才搬上 GPU 的元件（balanced：文字兩個＋transformer；low：文字兩個，transformer 交給
    group offload）。pipeline._to_device 照這份決定哪些「不要」先搬上 GPU（R22、R23：以前先整包搬上去再搬回來，
    balanced 載入的瞬間整個模型都在 VRAM，low 的文字元件白跑一趟）"""
    if vram_mode == "balanced":
        return ("text_encoder", "text_conditioner", "transformer")
    if vram_mode == "low":
        return ("text_encoder", "text_conditioner")
    return ()


def configure_memory(pipe, vram_mode, log=print):
    """設定 Anima 的 speed / balanced 生命週期；low 由 group offload 處理。"""
    if vram_mode not in ("balanced", "low"):
        return
    _install_memory_hooks(pipe)
    names = parked(vram_mode)
    moved = _move_components(pipe, names, "cpu", log)
    if moved:
        log(i18n.tr("eng.a.idle", mode=vram_mode))


# ---------------------------------------------------------------- img2img
# diffusers 的 Anima modular pipeline 只有 text2image（ROADMAP C+.0）。
# flow matching 的 img2img 不必動模型，只有兩件事：
#   1. 排程照常算完整一條，**跳過前面幾步**，從中途的 σ 開始
#   2. 起始 latent 用 x = (1−σ)·x₀ + σ·ε 混進雜訊（x₀ 是底圖的 latent）
# 為什麼不直接餵一段比較短的 sigmas：Karras／Beta 這類分佈是依整段的範圍
# 算出來的，只給後半段會算出另一條曲線。先照常算、再切掉前段，每一支
# 排程器（包括 samplers.py 自製的）拿到的都是 txt2img 那條的後半段。

def start_partway(scheduler, skip):
    """set_timesteps 之後呼叫：把前 skip 步切掉，從第 skip 步開始跑"""
    if skip <= 0:
        return
    scheduler.timesteps = scheduler.timesteps[skip:]
    scheduler.sigmas = scheduler.sigmas[skip:]
    n = len(scheduler.timesteps)
    scheduler.num_inference_steps = n
    # DPM++ SDE（singlestep）每一步的階數是照總步數排好的一張表，要重排
    if hasattr(scheduler, "order_list") and hasattr(scheduler, "get_order_list"):
        scheduler.order_list = scheduler.get_order_list(n)
    scheduler._step_index = None
    scheduler._begin_index = None


def _once_partway(scheduler, skip):
    """下一次 set_timesteps 之後自動切掉前 skip 步；回傳還原用的函式

    Anima 的 before_denoise 是在管線裡面呼叫 set_timesteps，外面插不進去，
    所以在實例上蓋一層。用完一定要還原，不然下一張 txt2img 也會被切。
    """
    import functools
    inner = scheduler.set_timesteps

    # wraps 不能省：before_denoise 的 retrieve_timesteps 用 inspect.signature
    # 檢查收不收 sigmas，*a / **kw 的簽名會被判成「不支援自訂 sigmas」直接丟例外
    @functools.wraps(inner)
    def set_timesteps(*a, **kw):
        out = inner(*a, **kw)
        start_partway(scheduler, skip)
        return out

    scheduler.set_timesteps = set_timesteps

    def restore():
        scheduler.__dict__.pop("set_timesteps", None)
    return restore


def mask_steps(scheduler, x0, eps, m):
    """局部重繪：每一步之後，遮罩外（m=0）換回底圖在「下一個 σ」的加噪版本
    x = (1−σ)·x₀ + σ·ε，模型看得到完整畫面，遮罩裡畫的才接得上外面。
    最後一步 σ=0，遮罩外就是 x₀ 本身。回傳還原用的函式。

    包在 scheduler.step 外面：Anima 的去噪迴圈在 diffusers 裡，外面插不進去。
    原本可能已經被 _bind_scheduler_generator 包過一層，還原時要放回那一層
    """
    import functools
    prev = scheduler.__dict__.get("step")
    inner = scheduler.step

    @functools.wraps(inner)
    def step(*a, **kw):
        out = inner(*a, **kw)
        lat = out[0] if isinstance(out, tuple) else out.prev_sample
        i = scheduler.step_index
        sig = float(scheduler.sigmas[i]) if i is not None and i < len(scheduler.sigmas) else 0.0
        known = ((1.0 - sig) * x0 + sig * eps).to(device=lat.device, dtype=lat.dtype)
        mm = m.to(device=lat.device, dtype=lat.dtype)
        lat = mm * lat + (1.0 - mm) * known
        if isinstance(out, tuple):
            return (lat, *out[1:])
        out.prev_sample = lat
        return out

    scheduler.step = step

    def restore():
        if prev is None:
            scheduler.__dict__.pop("step", None)
        else:
            scheduler.step = prev
    return restore


def _encode_init(pipe, req, log):
    """底圖 → 正規化過的 latent（(1, C, 1, h, w)，float32）

    解碼時是 latents × std + mean（decoders.py 存的是 1/std），這裡反過來。
    用分佈的眾數不抽樣：同一張底圖每次編出一樣的 latent，差異只來自 seed。
    """
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P

    img = P.load_init_image(req["init_image"], req["width"], req["height"])
    vae = pipe.vae
    dev = _component_device(vae) or torch.device("cuda")
    x = torch.from_numpy(np.asarray(img)).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None, :, None].to(device=dev, dtype=vae.dtype)
    z = vae.encode(x).latent_dist.mode()
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1).to(z)
    std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1).to(z)
    return ((z - mean) / std).to(torch.float32)


def _prepare_img2img(pipe, req, generator, log):
    """算排程、編碼底圖、混雜訊。回傳 (latents, sigmas, 還原函式)"""
    import numpy as np
    import torch
    try:
        from core import pipeline as P
    except ImportError:
        import pipeline as P

    n_all, run = P.img2img_steps(req["steps"], req.get("strength", 0.5))
    skip = n_all - run
    # 跟 before_denoise 一模一樣的輸入，先自己算一次，拿到切點的 σ
    sigmas = np.linspace(1.0, 1 / n_all, n_all)
    sched = pipe.scheduler
    sched.set_timesteps(sigmas=sigmas, device=torch.device("cuda"))
    s0 = float(sched.sigmas[skip])

    x0 = _encode_init(pipe, req, log)
    eps = torch.randn(x0.shape, generator=generator, device=generator.device, dtype=torch.float32).to(x0.device)
    latents = (1.0 - s0) * x0 + s0 * eps
    restores = [_once_partway(sched, skip)]
    if req.get("_mask") is not None:
        try:
            from core import inpaint as IP
        except ImportError:
            import inpaint as IP
        h, w = x0.shape[-2:]
        m = torch.from_numpy(IP.latent_mask(req["_mask"], w, h))[None, None, None].to(x0.device)
        restores.append(mask_steps(sched, x0, eps, m))
    strength = min(1.0, max(P.STRENGTH_MIN, float(req.get("strength", 0.5))))
    log(i18n.tr("eng.a.i2i", mode=i18n.tr("eng.inpaint") if req.get('_mask') is not None else "img2img", s=f"{strength:g}", s0=f"{s0:.3f}", run=run, n=n_all))

    def restore():
        for r in reversed(restores):
            r()
    return latents, sigmas, restore


def generate(pipe, req, on_step=None, log=print):
    """回傳 (PIL Image, seed)。介面與 pipeline.generate 對齊。"""
    import torch
    from diffusers import ClassifierFreeGuidance

    seed = int(req.get("seed", -1))
    if seed < 0:
        import random
        seed = random.randrange(1 << 31)

    vram_mode = getattr(pipe, "_coco_vram", "speed")
    pipe._coco_log = log
    if vram_mode in ("balanced", "low"):
        names = (("text_encoder", "text_conditioner", "transformer", "vae")
                 if vram_mode == "balanced"
                 else ("text_encoder", "text_conditioner", "vae"))
        moved = _move_components(
            pipe, names, "cuda", log)
        if moved:
            log(i18n.tr("eng.a.toGpu", mode=vram_mode, names=i18n.join(moved)))

    # 初始 latent 使用傳入的 generator；modular denoise loop 卻不會把它繼續
    # 傳給 scheduler.step。兩邊都綁定，SDE / ER-SDE 的逐步雜訊才會同 seed。
    # 亂數在 CPU 或顯示卡產生（設定的「用 CPU 產生亂數」，見 pipeline.rng_device）
    generator = torch.Generator("cpu" if str(req.get("rng") or "gpu").lower() == "cpu" else "cuda").manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scheduler = pipe.scheduler
    _bind_scheduler_generator(scheduler)
    scheduler._coco_generator = generator

    # CFG 是每次生成都可能改的，所以在這裡換掉 guider
    pipe.update_components(
        guider=ClassifierFreeGuidance(guidance_scale=float(req["cfg"])))

    # 提示詞的 (權重:1.2) 語法：標記要拿掉，權重改乘在條件向量上。
    # 不拿掉的話那些括號和數字會被 tokenizer 當成一般文字送進模型。
    maxlen = int(req.get("max_tokens") or 512)
    pos, wpos = weigh_prompt(pipe.t5_tokenizer, req["prompt"], maxlen)
    neg, wneg = weigh_prompt(pipe.t5_tokenizer, req.get("negative", ""), maxlen)

    slots = []
    for text, w in ((pos, wpos), (neg, wneg)):
        if w is None:
            continue
        ids = pipe.t5_tokenizer(text, padding="longest", max_length=maxlen,
                                truncation=True,
                                return_tensors="pt").input_ids[0]
        slots.append((ids, torch.tensor(w, dtype=torch.float32)))
    pipe._coco_weights = slots
    if slots:
        _install_weights(pipe)
        log(i18n.tr("eng.a.weights", n=len(slots)))

    pipe._coco_on_step = on_step
    extra, restore = {}, None
    try:
        if req.get("init_image"):
            latents, sigmas, restore = _prepare_img2img(pipe, req, generator, log)
            extra = {"latents": latents, "sigmas": sigmas}
        out = pipe(
            prompt=pos, negative_prompt=neg or None,
            height=int(req["height"]), width=int(req["width"]),
            num_inference_steps=int(req["steps"]),
            max_sequence_length=maxlen,
            generator=generator,
            output="images", **extra)
    finally:
        if restore:
            restore()
        pipe._coco_on_step = None
        pipe._coco_weights = []
        scheduler._coco_generator = None
        if vram_mode in ("balanced", "low"):
            names = (("text_encoder", "text_conditioner", "transformer")
                     if vram_mode == "balanced"
                     else ("text_encoder", "text_conditioner"))
            _move_components(pipe, names, "cpu", log)
        pipe._coco_log = print
    img = out[0] if isinstance(out, (list, tuple)) else out
    return img, seed
