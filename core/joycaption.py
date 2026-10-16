# -*- coding: utf-8 -*-
"""
joycaption.py —— 打標台的 JoyCaption（自然語言描述），直接讀使用者的 GGUF，不需要 llama-server

JoyCaption Beta One（fancyfeast/llama-joycaption-beta-one-hf-llava）是 LLaVA：SigLIP2 so400m（384×384、729 個圖塊）
看圖、兩層 MLP 投影到 4096，接 Llama 3.1 8B 寫字。使用者手上的是社群轉的 GGUF：主體 Q8_0、看圖的部分（mmproj）F16。

**看圖照官方的設定，不照 mmproj 寫的。** 這份 mmproj 的設定寫成 CLIP 的正規化（0.481…／0.268…）、use_gelu=False，
但官方的 preprocessor_config 是 0.5／0.5、Lanczos 縮成 384×384，SigLIP 的激活是 tanh 近似的 GELU（2026-10-06 核對）。
mmproj 只存了 26 層（官方取倒數第二層的輸出，最後一層與 post-LN 用不到）；llama.cpp 的命名把 MLP 的兩層取反了
（ffn_down 才是第一層 1152→4304，看形狀就知道）。投影層是 Linear → GELU（erf）→ Linear。

**寫字的部分為了快，權重在顯示卡上存成 int8 直接算**（comfy_kitchen 的 int8_linear，W8A8）：
Q8_0 是每 32 個值一個比例，int8_linear 要每一列一個比例，所以讀檔時重新量化一次。先乘 Hadamard（每 256 個一組，
跟 Qwen Image 2.1 的 convrot 同一套）把離群值打散再量化，權重誤差 0.87～0.94%（不旋轉 1.0～1.6%）。
Q、K、V 接成一個矩陣、gate 與 up 接成一個矩陣；RMSNorm、SwiGLU 併進量化的輸入端，殘差相加併進輸出端。
產生每個字的計算錄成 CUDA graph 重播（省掉 Python 與啟動的開銷），注意力用 comfy_kitchen 的 decode 版本。

**RoPE 用相鄰兩兩一組的轉法**：GGUF 轉檔時 q、k 的權重已經重排成 llama.cpp 的這種轉法，數學上跟官方的一樣。
Llama 3.1 的長度縮放存在 rope_freqs.weight（每個頻率除以它）。

**tokenizer** 用檔案裡的詞彙表與 merges 組（tokenizers 套件）：Llama 3 的切字規則、ignore_merges。
對話照檔案裡的模板：系統、使用者（圖的三個特殊 token 在提示詞前面，中間那個換成 729 個圖塊）、助手。

**實測**（RTX 5080，同一份 GGUF、同樣 5 張圖、溫度 0，2026-10-06）：
            看圖＋讀提示詞（約 790 token）   寫字           每張圖
  CoCo      35＋45 毫秒                       96.4 字／秒    0.85～0.99 秒
  llama.cpp 150～185 毫秒                     95.9 字／秒    1.12～1.25 秒（llama-server b10199）
tokenizer 切出的 token 跟 llama.cpp 一樣；第一個字的機率跟完整精度（Q8_0 還原成 float32、不量化輸入）比，
KL 0.0036。載入約 6 秒，顯示卡約 8.6 GB。RoPE 一定要給 4 維／6 維，comfy_kitchen 才用 CUDA 合併版（不然每個字多 0.35 毫秒）。
"""

import threading

import math
import time

import numpy as np

import i18n

IMAGE_SIZE = 384
IMAGE_TOKENS = 729
GROUP = 256                          # Hadamard 旋轉的分組
EOS = (128001, 128008, 128009)
SYSTEM = "You are a helpful image captioner."
DATE = "26 July 2024"                # 官方模板沒給日期時用的
# Llama 3 的切字規則（tokenizer.json 的 pre_tokenizer）
PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
           r"\s+(?!\S)|\s+")


# ---------------------------------------------------------------- tokenizer

def build_tokenizer(g):
    """GGUF 裡的詞彙表與 merges → tokenizers.Tokenizer（跟官方 tokenizer.json 同樣的設定）"""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    tokens = g.meta["tokenizer.ggml.tokens"]
    types = g.meta["tokenizer.ggml.token_type"]
    vocab = {t: i for i, t in enumerate(tokens)}
    merges = [tuple(m.split(" ", 1)) for m in g.meta["tokenizer.ggml.merges"]]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    from tokenizers import AddedToken
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t, ty in zip(tokens, types) if int(ty) == 3])
    return tok


def chat_ids(tok, prompt, system=SYSTEM):
    """官方模板 → token id。圖的位置是 <|reserved_special_token_69|>（之後換成 729 個圖塊）"""
    text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Cutting Knowledge Date: December 2023\nToday Date: {DATE}\n\n{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return tok.encode(text, add_special_tokens=False).ids


# ---------------------------------------------------------------- 權重

def hadamard(size, device):
    import torch
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return (h / math.sqrt(size)).to(device)


def q8_0_to_float(raw, shape, device):
    """Q8_0 的位元組（numpy uint8）→ 顯示卡上的 float32 (N, K)"""
    import torch
    b = torch.from_numpy(np.array(raw)).to(device).view(-1, 34)        # 複製一份：mmap 的頁面是唯讀的
    d = b[:, :2].contiguous().view(torch.float16).float()
    q = b[:, 2:].contiguous().view(torch.int8).float()
    return (q * d).view(shape)


def requant(w, H):
    """float32 (N, K) → (int8 (N, K), 每列比例 float32 (N,))：先乘 Hadamard（每 GROUP 個一組）再每列量化。
    還原是 (q · H) × 比例（H 對稱正交，H·H = I），comfy_kitchen 的 convrot 會把輸入也乘同一個 H"""
    import torch
    n, k = w.shape
    wr = (w.view(n, k // GROUP, GROUP) @ H).view(n, k)
    s = wr.abs().amax(1).clamp(min=1e-12) / 127
    return (wr / s[:, None]).round_().clamp_(-127, 127).to(torch.int8), s


class Int8W:
    """一個量化的矩陣（可以是好幾個上下接起來）"""
    __slots__ = ("q", "s")

    def __init__(self, q, s):
        self.q, self.s = q, s


def load_q8(g, names, H, device, chunk=8192):
    """好幾個 Q8_0 矩陣（列數可以不同、欄數相同）上下接起來，分批還原、旋轉、量化（大的矩陣不會一次佔滿顯示卡）"""
    import torch
    qs, ss = [], []
    for name in names:
        n, k = g.shape(name)
        raw = g.raw(name)
        per_row = k // 32 * 34
        for r in range(0, n, chunk):
            rows = min(chunk, n - r)
            w = q8_0_to_float(raw[r * per_row:(r + rows) * per_row], (rows, k), device)
            q, s = requant(w, H)
            qs.append(q)
            ss.append(s)
            del w
    return Int8W(torch.cat(qs), torch.cat(ss))


# ---------------------------------------------------------------- 看圖（SigLIP so400m＋投影）

def prep_image(path):
    """官方的前處理：透明鋪白、轉 RGB、Lanczos 縮成 384×384（不保留比例）、÷255、−0.5、÷0.5 → (1, 3, 384, 384)"""
    import torch
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None] / 255.0
    return (x - 0.5) / 0.5


class Vision:
    """mmproj 的 SigLIP（26 層）＋投影，bf16"""

    def __init__(self, g, device, dtype):
        import torch
        t = lambda n: torch.from_numpy(g.array(n)).to(device, dtype)        # noqa: E731
        self.dtype = dtype
        self.patch_w, self.patch_b = t("v.patch_embd.weight"), t("v.patch_embd.bias")
        self.pos = t("v.position_embd.weight")
        self.heads = int(g.meta["clip.vision.attention.head_count"])
        self.eps = float(g.meta["clip.vision.attention.layer_norm_epsilon"])
        n = 1 + max(int(k.split(".")[2]) for k in g.tensors if k.startswith("v.blk."))
        self.blocks = []
        for i in range(n):
            p = f"v.blk.{i}."
            self.blocks.append({
                "ln1": (t(p + "ln1.weight"), t(p + "ln1.bias")), "ln2": (t(p + "ln2.weight"), t(p + "ln2.bias")),
                "qkv": (torch.cat([t(p + f"attn_{c}.weight") for c in "qkv"]), torch.cat([t(p + f"attn_{c}.bias") for c in "qkv"])),
                "out": (t(p + "attn_out.weight"), t(p + "attn_out.bias")),
                # llama.cpp 的命名取反了：ffn_down 是第一層（1152→4304）、ffn_up 是第二層
                "fc1": (t(p + "ffn_down.weight"), t(p + "ffn_down.bias")), "fc2": (t(p + "ffn_up.weight"), t(p + "ffn_up.bias"))})
        self.mm0 = (t("mm.0.weight"), t("mm.0.bias"))
        self.mm2 = (t("mm.2.weight"), t("mm.2.bias"))

    def __call__(self, pixels):
        """(1, 3, 384, 384) → (729, 4096)"""
        import torch.nn.functional as F
        x = F.conv2d(pixels.to(self.patch_w.device, self.dtype), self.patch_w, self.patch_b, stride=14)
        x = x.flatten(2).transpose(1, 2) + self.pos                           # (1, 729, 1152)
        B, N, D = x.shape
        hd = D // self.heads
        for b in self.blocks:
            h = F.layer_norm(x, (D,), *b["ln1"], eps=self.eps)
            q, k, v = F.linear(h, *b["qkv"]).view(B, N, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
            x = x + F.linear(a, *b["out"])
            h = F.layer_norm(x, (D,), *b["ln2"], eps=self.eps)
            x = x + F.linear(F.gelu(F.linear(h, *b["fc1"]), approximate="tanh"), *b["fc2"])
        return F.linear(F.gelu(F.linear(x[0], *self.mm0)), *self.mm2)


# ---------------------------------------------------------------- Llama 3.1 8B

class Llama:
    def __init__(self, g, device, max_len=2048, log=print):
        import torch
        self.device = device
        self.max_len = max_len
        m = g.meta
        self.n_layer = int(m["llama.block_count"])
        self.dim = int(m["llama.embedding_length"])
        self.n_head = int(m["llama.attention.head_count"])
        self.n_kv = int(m["llama.attention.head_count_kv"])
        self.hd = self.dim // self.n_head
        self.ffn = int(m["llama.feed_forward_length"])
        self.eps = float(m["llama.attention.layer_norm_rms_epsilon"])
        H = hadamard(GROUP, device)
        t0 = time.time()
        f32 = lambda n: torch.from_numpy(g.array(n)).to(device, torch.bfloat16)    # noqa: E731
        self.layers = []
        for i in range(self.n_layer):
            p = f"blk.{i}."
            self.layers.append({
                "attn_norm": f32(p + "attn_norm.weight"), "ffn_norm": f32(p + "ffn_norm.weight"),
                "qkv": load_q8(g, [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], H, device),
                "o": load_q8(g, [p + "attn_output.weight"], H, device),
                "gu": load_q8(g, [p + "ffn_gate.weight", p + "ffn_up.weight"], H, device),
                "down": load_q8(g, [p + "ffn_down.weight"], H, device)})
        self.norm = f32("output_norm.weight")
        self.one = torch.ones(self.dim, device=device, dtype=torch.bfloat16)     # 殘差相加的倍數（int8_linear 一定要給）
        self.head = load_q8(g, ["output.weight"], H, device)
        # 詞嵌入：只查用到的幾列，直接存 Q8_0（int8＋每 32 個一個比例），查的時候還原
        raw = torch.from_numpy(np.array(g.raw("token_embd.weight"))).to(device).view(-1, 34)
        self.emb_d = raw[:, :2].contiguous().view(torch.float16).view(-1, self.dim // 32)
        self.emb_q = raw[:, 2:].contiguous().view(torch.int8).view(-1, self.dim)
        del raw
        log(i18n.tr("eng.jc.loaded", s=f"{time.time() - t0:.1f}"))
        # RoPE 的 cos／sin 表：Llama 3.1 的頻率除以 rope_freqs（長度縮放）
        base = float(m["llama.rope.freq_base"])
        inv = 1.0 / base ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd)
        if "rope_freqs.weight" in g.tensors:
            inv = inv / torch.from_numpy(g.array("rope_freqs.weight")).double()
        ang = torch.arange(max_len, dtype=torch.float64)[:, None] * inv[None]
        c, s = ang.cos().float(), ang.sin().float()
        # comfy_kitchen 的格式：每個位置、每對一個 2×2 旋轉矩陣，(長度, 1, 64, 2, 2)（中間的 1 對到頭數）
        self.freqs = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)[:, None].to(device)
        # KV 快取：每層 (1, 長度, KV 頭數, 128)
        self.kc = torch.zeros(self.n_layer, 1, max_len, self.n_kv, self.hd, device=device, dtype=torch.bfloat16)
        self.vc = torch.zeros_like(self.kc)
        self.graph = None
        self._capture()

    def embed(self, ids):
        import torch
        q = self.emb_q[ids].view(-1, self.dim // 32, 32).float()
        return (q * self.emb_d[ids].float()[..., None]).view(-1, self.dim).to(torch.bfloat16)

    def _lin(self, x, w, **kw):
        import comfy_kitchen as ck
        import torch
        return ck.int8_linear(x, w.q, w.s, None, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=GROUP, **kw)

    def _layer(self, i, L, x, pos, f, decode, kv_len=None):
        """一層。f：這幾個位置的 RoPE 矩陣（每次只查一次表，32 層共用）。
        RoPE 是相鄰兩兩一組轉：GGUF 轉檔時 q、k 的權重已經照 llama.cpp 的這種轉法重排，數學上跟官方的一樣"""
        import comfy_kitchen as ck
        import torch.nn.functional as F
        T = x.shape[0]
        qkv = self._lin(x, L["qkv"], input_act="rms_norm", input_act_weight=L["attn_norm"], input_act_eps=self.eps)
        q = qkv[:, :self.dim].view(T, self.n_head, self.hd)
        k = qkv[:, self.dim:self.dim + self.n_kv * self.hd].view(T, self.n_kv, self.hd)
        v = qkv[:, self.dim + self.n_kv * self.hd:].view(T, self.n_kv, self.hd)
        # 4 維／6 維才會用 CUDA 合併版（3 維會退回一步一步算：每個字多 0.35 毫秒，RTX 5080 實測）
        q, k = (t[0] for t in ck.apply_rope(q[None], k[None], f))
        self.kc[i, 0].index_copy_(0, pos, k)
        self.vc[i, 0].index_copy_(0, pos, v)
        if decode:
            a = ck.flash_attention_decode(q[None], self.kc[i], self.vc[i], kv_len).reshape(1, self.dim)
        else:
            a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                               is_causal=True, enable_gqa=True)[0].transpose(0, 1).reshape(T, self.dim)
        x = self._lin(a, L["o"], residual=x, residual_scale=self.one)
        h = self._lin(x, L["gu"], input_act="rms_norm", input_act_weight=L["ffn_norm"], input_act_eps=self.eps)
        return self._lin(h, L["down"], input_act="swiglu", residual=x, residual_scale=self.one)

    def logits(self, x):
        return self._lin(x, self.head, input_act="rms_norm", input_act_weight=self.norm, input_act_eps=self.eps).float()

    def prefill(self, x):
        """整段提示詞（含圖塊）一次算完，回傳最後一個位置的 logits。x：(T, 4096) bf16"""
        import torch
        T = x.shape[0]
        if T >= self.max_len - 1:
            raise ValueError(i18n.tr("eng.jc.long", n=T))
        pos = torch.arange(T, device=self.device)
        with torch.inference_mode():
            f = self.freqs[:T][None]
            for i, L in enumerate(self.layers):
                x = self._layer(i, L, x, pos, f, False)
            return self.logits(x[-1:])

    # 產生一個字：錄成 CUDA graph。輸入（token、位置、KV 長度）放在固定的張量裡，重播前改內容
    def _decode_body(self):
        import torch
        x = self.embed(self._tok)
        f = self.freqs[self._pos][None]
        for i, L in enumerate(self.layers):
            x = self._layer(i, L, x, self._pos, f, True, self._kvlen)
        self._out.copy_(self.logits(x))

    def _capture(self):
        """錄之前要先真的跑兩次：位置放在 KV 快取的最後一格（永遠用不到），不然會蓋掉第 0 格（<|begin_of_text|>）"""
        import torch
        self._tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._pos = torch.full((1,), self.max_len - 1, dtype=torch.long, device=self.device)
        self._kvlen = torch.ones(1, dtype=torch.int32, device=self.device)
        self._out = torch.zeros(1, self.head.q.shape[0], dtype=torch.float32, device=self.device)
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(2):                    # 先跑兩次：讓配置器與核心都準備好，錄的時候才不會配置新的記憶體
                self._decode_body()
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            self._decode_body()
        self.graph = g

    def step(self, token, pos):
        """位置 pos 放 token，回傳下一個字的 logits（之前的位置要已經在 KV 快取裡）"""
        if not 0 <= pos < self.max_len - 1:
            raise ValueError(i18n.tr("eng.jc.maxLen"))
        self._tok.fill_(int(token))
        self._pos.fill_(int(pos))
        self._kvlen.fill_(int(pos) + 1)
        self.graph.replay()
        return self._out


# ---------------------------------------------------------------- 整體

def sample(logits, temperature, top_p, gen):
    """溫度＋top-p（官方建議 0.6／0.9）；溫度 0 是每次挑最可能的"""
    import torch
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits[0] / temperature, -1)
    p, idx = probs.sort(descending=True)
    keep = p.cumsum(0) - p < top_p
    p = p * keep
    return int(idx[torch.multinomial(p / p.sum(), 1, generator=gen)])


class JoyCaption:
    def __init__(self, main_path, mmproj_path, device="cuda", log=print, max_len=2048):
        import torch
        import gguf
        self.log = log
        gm, gv = gguf.GGUF(main_path), gguf.GGUF(mmproj_path)
        try:
            if gm.meta.get("general.architecture") != "llama":
                raise ValueError(i18n.tr("eng.jc.notLlama"))
            if gv.meta.get("general.architecture") != "clip" or not gv.meta.get("clip.has_llava_projector"):
                raise ValueError(i18n.tr("eng.jc.notMmproj"))
            self.tok = build_tokenizer(gm)
            self.vision = Vision(gv, device, torch.bfloat16)
            self.llm = Llama(gm, device, max_len, log)
        finally:
            gm.close()
            gv.close()
        self.image_id = self.tok.token_to_id("<|reserved_special_token_69|>")
        self.device = device

    def caption(self, path, prompt, max_new=300, temperature=0.6, top_p=0.9, seed=None, stats=None):
        import torch
        ids = chat_ids(self.tok, prompt)
        at = ids.index(self.image_id)
        t0 = time.perf_counter()
        with torch.inference_mode():
            img = self.vision(prep_image(path))
            ids_t = torch.tensor(ids, device=self.device)
            emb = self.llm.embed(ids_t)
            x = torch.cat([emb[:at], img.to(emb.dtype), emb[at + 1:]])
        t1 = time.perf_counter()
        logits = self.llm.prefill(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % 2**31)
        out = []
        pos = x.shape[0]
        for _ in range(max_new):
            tok = sample(logits, temperature, top_p, gen)
            if tok in EOS:
                break
            out.append(tok)
            if pos >= self.llm.max_len - 1:
                break
            logits = self.llm.step(tok, pos)
            pos += 1
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if stats is not None:
            stats.update(prompt_tokens=x.shape[0], new_tokens=len(out), vision=t1 - t0, prefill=t2 - t1, decode=t3 - t2,
                         tok_per_s=len(out) / max(1e-9, t3 - t2))
        return self.tok.decode(out, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- 工作行程用：要用時才載入，閒置就卸載

IDLE = 120                 # 幾秒沒用就卸載（8.6 GB 不該一直佔著顯示卡；載入一次約 6 秒）
_M = {}                    # (主體, mmproj) → JoyCaption
_LOCK = threading.RLock()
_LAST = [0.0]
_TIMER = [None]
on_release = None


def loaded():
    return bool(_M)


def get(main_path, mmproj_path, log=print):
    key = (str(main_path), str(mmproj_path))
    with _LOCK:
        if key not in _M:
            release()
            _M[key] = JoyCaption(main_path, mmproj_path, log=log)
        return _M[key]


def _touch():
    _LAST[0] = time.monotonic()
    if _TIMER[0] is not None:
        _TIMER[0].cancel()
    t = threading.Timer(IDLE + 0.05, _idle_release)
    t.daemon = True
    _TIMER[0] = t
    t.start()


def _idle_release():
    with _LOCK:
        if _M and time.monotonic() - _LAST[0] >= IDLE:
            release()
            if on_release:
                on_release()


def caption_many(main_path, mmproj_path, paths, prompts, max_new=300, temperature=0.6, top_p=0.9, log=print):
    """好幾張圖 → 每張的描述（讀不了的圖是 None，原因記在 ERRORS）與計時"""
    with _LOCK:
        try:
            m = get(main_path, mmproj_path, log)
            out, stats = [], []
            for path, prompt in zip(paths, prompts):
                st = {}
                try:
                    out.append(m.caption(path, prompt, max_new=max_new, temperature=temperature, top_p=top_p, stats=st))
                    stats.append(st)
                except (OSError, ValueError) as e:
                    ERRORS.append((str(path), f"{type(e).__name__}: {e}"))
                    del ERRORS[:-20]
                    out.append(None)
            return out, stats
        finally:
            _touch()


ERRORS = []


def release():
    """卸載、把顯示卡還回去（CUDA graph 也一起丟）"""
    with _LOCK:
        if not _M:
            return
        _M.clear()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
