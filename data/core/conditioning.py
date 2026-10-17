# -*- coding: utf-8 -*-
"""
conditioning.py —— A1111 / Forge 風格的提示詞編碼

為什麼要自己寫：
  diffusers 原生    → 一律補滿 77 個 token，但不支援加權、超過就截斷
  compel            → 支援加權與長提示詞，但補位長度跟 A1111 不同
                      （短提示詞時補到實際長度而非 77，條件向量因此不一樣）

A1111 的做法是：
  1. 解析 (word:1.2) / (word) / [word] 取得每個 token 的權重
  2. 每 75 個 token 切一塊，前後加 BOS/EOS 湊成 77
  3. 逗號附近優先斷句（comma padding backtrack），避免把詞組切開
  4. 每塊各自編碼，權重乘上去之後做均值還原
  5. 所有塊沿序列軸接起來

這個檔案就是把上面五步實作出來，屬於規格書裡「第 4 層自己寫」的部分。
"""

import math
import re

import torch

COMMA_BACKTRACK = 20        # A1111 的預設值
CHUNK = 75                  # 每塊的內容長度，加 BOS/EOS 後為 77

# ---------------------------------------------------------------- 權重語法
# 規則（跟 A1111／Forge 寫的提示詞相容，同一句話拿到同樣的權重）：
#   (詞)      ×1.1，可以疊：((詞)) ×1.21
#   [詞]      ÷1.1
#   (詞:1.3)  指定倍數；數字前後可以有空白、可以帶正負號
#   \( \) \[ \] \\   照字面，不當語法
#   BREAK     另起一塊（大寫、前後是空白或標點）
# 括號沒關到最後就關在句尾；多出來的右括號照字面留著。圓括號與方括號各算各的，
# 交錯也可以：( a [ b ) c ] 的 a、b 乘 1.1，b、c 除 1.1。
#
# 做法分兩趟：先切成片段並記下每組括號蓋住哪些片段，最後才一起算權重。
# 倍數照「括號關起來的先後」乘上去（句尾才關的：先圓括號、再方括號，各自照開的順序），
# 浮點數才跟其他工具算出來的逐位元一樣，相鄰同權重的片段也才合併得一樣。
ROUND_MUL = 1.1
SQUARE_MUL = 1 / 1.1
BREAK_MARK = -1                 # BREAK 在結果裡寫成 ("BREAK", -1)

_ESCAPABLE = "()[]\\"
_SYNTAX = "\\()[]:"
_WEIGHT_CLOSE = re.compile(r":\s*([+-]?[.\d]+)\s*\)")
_BREAK = re.compile(r"\s*\bBREAK\b\s*")


def _pieces(text):
    """把提示詞切成語法單位：("lit", 字) 跳脫的字、("open"/"close", 括號)、
    ("weight", (原文, 倍數或 None)) 冒號加數字加右括號、("text", 文字)"""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            nxt = text[i + 1:i + 2]
            if nxt and nxt in _ESCAPABLE:
                yield "lit", nxt
                i += 2
            else:
                yield "lit", ""          # 後面不是可跳脫的字：反斜線本身拿掉（相容）
                i += 1
        elif c in "([":
            yield "open", c
            i += 1
        elif c in ")]":
            yield "close", c
            i += 1
        elif c == ":":
            m = _WEIGHT_CLOSE.match(text, i)
            if m:
                try:
                    num = float(m.group(1))
                except ValueError:       # 「:.)」「:1.2.3)」：以前整個生成會失敗，現在照字面
                    num = None
                yield "weight", (m.group(0), num)
                i = m.end()
            else:
                yield "text", ":"
                i += 1
        else:
            j = i + 1
            while j < n and text[j] not in _SYNTAX:
                j += 1
            yield "text", text[i:j]
            i = j


def parse_attention(text):
    """把提示詞拆成 [[文字, 權重], ...]；BREAK 是 ["BREAK", -1]"""
    segs = []                  # 每段的文字；BREAK 用 None 佔位
    spans = []                 # 已經關起來的括號：(開始的片段, 結束的片段, 倍數)，照關的先後
    open_round, open_square = [], []

    def add_text(s):
        for k, part in enumerate(_BREAK.split(s)):
            if k:
                segs.append(None)
            segs.append(part)

    for kind, val in _pieces(text):
        if kind == "lit":
            segs.append(val)
        elif kind == "open":
            (open_round if val == "(" else open_square).append(len(segs))
        elif kind == "close":
            stack, mul = (open_round, ROUND_MUL) if val == ")" else (open_square, SQUARE_MUL)
            if stack:
                spans.append((stack.pop(), len(segs), mul))
            else:
                add_text(val)
        elif kind == "weight":
            raw, num = val
            if open_round and num is not None:
                spans.append((open_round.pop(), len(segs), num))
            else:
                add_text(raw)
        else:
            add_text(val)

    end = len(segs)
    spans += [(s, end, ROUND_MUL) for s in open_round]
    spans += [(s, end, SQUARE_MUL) for s in open_square]

    weights = [1.0 if s is not None else BREAK_MARK for s in segs]
    for start, stop, mul in spans:
        for k in range(start, stop):
            weights[k] *= mul          # BREAK 也會被乘到，跟 A1111／Forge 一樣（括號裡的 BREAK 不算分塊）

    out = []
    for s, w in zip(segs, weights):
        s = "BREAK" if s is None else s
        if out and out[-1][1] == w:
            out[-1][0] += s
        else:
            out.append([s, w])
    return out or [["", 1.0]]


def tokenize_weighted(tokenizer, text):
    """回傳 [(token_ids, weights), ...]，每項是一塊 75 個 token 的內容"""
    parsed = parse_attention(text)

    ids, wts = [], []
    for chunk_text, w in parsed:
        if chunk_text == "BREAK" and w == -1:
            # 補滿目前這塊
            while len(ids) % CHUNK != 0:
                ids.append(tokenizer.eos_token_id)
                wts.append(1.0)
            continue
        toks = tokenizer(chunk_text, truncation=False,
                         add_special_tokens=False).input_ids
        ids.extend(toks)
        wts.extend([w] * len(toks))

    # 切塊，逗號附近優先斷
    comma_id = tokenizer(",", add_special_tokens=False).input_ids
    comma_id = comma_id[0] if comma_id else None

    chunks, i = [], 0
    while i < len(ids):
        end = min(i + CHUNK, len(ids))
        if end < len(ids) and comma_id is not None:
            # 往回找逗號，但不超過 COMMA_BACKTRACK
            for back in range(0, min(COMMA_BACKTRACK, end - i)):
                if ids[end - 1 - back] == comma_id:
                    end = end - back
                    break
        chunks.append((ids[i:end], wts[i:end]))
        i = end

    if not chunks:
        chunks = [([], [])]
    return chunks


def encode_chunk(tokenizer, text_encoder, token_ids, weights,
                 want_pooled=False, clip_skip=2, min_clip_skip=1, norm=True):
    """編碼單一塊，回傳 (hidden_states, pooled 或 None)

    序列組法必須是  [BOS] 內容 [EOS] 補位...  共 77 個。
    EOS 一定要緊接在內容後面，不能放到最尾端——
    SDXL 的 tokenizer_2 其 pad token 是 0、eos 是 49407，兩者不同，
    而 pooled 向量是從 EOS 的位置取出來的，位置錯了整個風格就跑掉。
    """
    device = text_encoder.device
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    pad = eos if pad is None else pad

    body = list(token_ids[:CHUNK])
    w = list(weights[:CHUNK])

    ids = [bos] + body + [eos]
    ws = [1.0] + w + [1.0]
    while len(ids) < CHUNK + 2:
        ids.append(pad)
        ws.append(1.0)

    full = torch.tensor([ids], device=device)
    full_w = torch.tensor([ws], device=device, dtype=torch.float32)

    out = text_encoder(full, output_hidden_states=True)
    try:
        layer = max(int(clip_skip), int(min_clip_skip))
    except (TypeError, ValueError):
        layer = int(min_clip_skip)
    layer = min(layer, len(out.hidden_states))
    z = out.hidden_states[-layer]
    pooled = None
    if want_pooled:
        pooled = getattr(out, "text_embeds", None)
        if pooled is None:
            pooled = out[0]

    # 加權後還原均值，避免整體亮度／對比被權重帶偏（A1111／Forge 的 Emphasis「Original」）。
    # norm=False 是 Forge 的「No norm」：只乘權重、不拉回，加重的詞直接變強，其他詞不受影響
    #
    # 對齊 z 的裝置而不是上面那個 device：省 VRAM 模式下文字編碼器
    # 平常放在 CPU，呼叫時 accelerate 的 hook 才把模組與輸入搬上 GPU，
    # 所以 text_encoder.device 讀到的是 cpu，輸出卻在 cuda。
    if any(abs(x - 1.0) > 1e-6 for x in ws):
        orig_mean = z.mean()
        z = z * full_w.to(z.device, z.dtype).unsqueeze(-1)
        if norm:
            z = z * (orig_mean / z.mean())
    return z, pooled


def inspect(pipe):
    """判斷這個管線該用哪種 conditioning。

    回傳 dict：
      ok      —— 能不能用 A1111 風格編碼
      why     —— 不能用時的原因
      limit   —— 文字序列長度上限
      encoder —— 文字編碼器類別名稱
      chunked —— 能不能靠分塊突破上限
    """
    tok = getattr(pipe, "tokenizer", None)
    te = getattr(pipe, "text_encoder", None)

    if tok is None or te is None:
        return {"ok": False, "why": "這個管線沒有獨立的文字編碼器",
                "limit": None, "encoder": "無", "chunked": False}

    cls = type(te).__name__
    limit = getattr(tok, "model_max_length", None)
    if isinstance(limit, int) and limit > 100000:      # 有些 tokenizer 填很大的值
        limit = None

    # 管線自己宣告的固定文字長度（LLM 條件模型會有）
    fixed = None
    for attr in ("max_sequence_length", "tokenizer_max_length"):
        v = getattr(pipe, attr, None)
        if isinstance(v, int):
            fixed = v
            break
    if fixed is None:
        try:
            import inspect as _i
            sig = _i.signature(pipe.__call__)
            p = sig.parameters.get("max_sequence_length")
            if p is not None and isinstance(p.default, int):
                fixed = p.default
        except Exception:
            pass

    if "CLIP" not in cls:
        return {"ok": False,
                "why": f"文字編碼器是 {cls}，不是 CLIP；"
                       "A1111 的權重是 CLIP embedding 的插值技巧，此處不適用",
                "limit": fixed or limit, "encoder": cls, "chunked": False}

    if limit not in (77, None) and limit and limit > 100:
        return {"ok": False,
                "why": f"CLIP 但序列上限是 {limit}，不是標準的 77",
                "limit": limit, "encoder": cls, "chunked": False}

    return {"ok": True, "why": "", "limit": 77, "encoder": cls,
            "chunked": True}


def summarize(pipe):
    """給使用者看的一段說明"""
    info = inspect(pipe)
    lines = [f"  文字編碼器：{info['encoder']}"]
    if info["ok"]:
        lines.append("  序列上限　：77／塊，可無限分塊")
        lines.append("  → 使用 a1111 編碼：支援權重語法與長提示詞")
    else:
        lim = info["limit"]
        lines.append(f"  序列上限　：{lim if lim else '未知'}"
                     "（固定長度，不能分塊）")
        lines.append(f"  → 使用 native 編碼：{info['why']}")
        if lim and lim >= 256:
            lines.append(f"    這類模型本來就給足長度（{lim} tokens），"
                         "不需要分塊；強調請改用自然語言描述")
    return "\n".join(lines)


@torch.inference_mode()
def encode_prompt(pipe, prompt, negative, clip_skip=2, norm=True):
    """回傳 diffusers 管線需要的四個張量"""
    is_sdxl = getattr(pipe, "text_encoder_2", None) is not None

    minimum = 2 if is_sdxl else 1

    def run(text, tokenizer, encoder, want_pooled):
        chunks = tokenize_weighted(tokenizer, text)
        zs, pooled = [], None
        for i, (ids, wts) in enumerate(chunks):
            z, p = encode_chunk(tokenizer, encoder, ids, wts,
                                want_pooled and i == 0,
                                clip_skip=clip_skip,
                                min_clip_skip=minimum, norm=norm)
            zs.append(z)
            if p is not None and pooled is None:
                pooled = p
        return torch.cat(zs, dim=1), pooled

    if not is_sdxl:
        pe, _ = run(prompt, pipe.tokenizer, pipe.text_encoder, False)
        ne, _ = run(negative, pipe.tokenizer, pipe.text_encoder, False)
        pe, ne = _pad_pair(pe, ne)
        return pe, None, ne, None

    pe1, _ = run(prompt, pipe.tokenizer, pipe.text_encoder, False)
    pe2, pp = run(prompt, pipe.tokenizer_2, pipe.text_encoder_2, True)
    ne1, _ = run(negative, pipe.tokenizer, pipe.text_encoder, False)
    ne2, npd = run(negative, pipe.tokenizer_2, pipe.text_encoder_2, True)

    pe = torch.cat([pe1, pe2], dim=-1)
    ne = torch.cat([ne1, ne2], dim=-1)
    if not (negative or "").strip():
        # 負面提示詞空白：SDXL 用全零（Forge sdxl.py 的 force_zero_negative_prompt，SDXL 官方也這樣），
        # 不是把空字串拿去編碼。2026-10-11 實測 Forge 自己畫的 2 張（Euler、空白負面）：
        # 拿空字串編碼的跟 Forge 平均差 33～35，構圖細節整個不同
        ne, npd = torch.zeros_like(ne), torch.zeros_like(npd)
    pe, ne = _pad_pair(pe, ne)
    return pe, pp, ne, npd


def _pad_pair(a, b):
    """正負提示詞的序列長度必須一致（同一批算）：兩邊各自重複到兩個長度的最小公倍數。

    跟 Forge（backend/sampling/condition.py 的 ConditionCrossAttn.concat）一樣。注意力對整串重複幾次的序列，
    結果跟原本那串一樣（每個位置的權重平均分給幾份，加回來不變），等於兩邊分開算。
    以前短的那邊補零：77 個零的鍵分走注意力、值又是零，短的那邊（通常是負面提示詞，連 lazyhand 這類
    embedding）整個被稀釋。2026-10-11 實測 Forge 自己畫的 9 張（Euler，同 seed、同參數、Normal 已照 Forge）：
    跟 Forge 平均差 13.4 → 2.9，正負長度不同的 5 張從 15～25 降到 1.4～3.8"""
    if a.shape[1] == b.shape[1]:
        return a, b
    n = math.lcm(a.shape[1], b.shape[1])
    return a.repeat(1, n // a.shape[1], 1), b.repeat(1, n // b.shape[1], 1)


def count_tokens(pipe, prompt, negative=""):
    """native 編碼下每個欄位的 token 數，回傳 [(欄位, token 數, 上限), ...]

    **全部都列**，不只超過的——介面要拿它做即時計數（78/77 這種顯示），
    只回報超出的話使用者就只能在爆掉之後才知道，那太晚了。

    三個容易算錯的地方：

    1. **上限包含特殊 token。** CLIP 的 77 是含 BOS/EOS 的，所以內容
       只有 75 個位置。這裡用 `add_special_tokens` 的預設（會加），
       算出來的數字才跟上限可以直接比。
    2. **要逐一檢查每個 tokenizer。** SDXL 有兩個（CLIP-L 與 CLIP-G），
       詞彙表不同，同一段文字的 token 數也不同，可能只有一邊爆掉。
    3. **native 看不懂權重語法。** `(word:1.2)` 的括號與數字會被當成
       普通文字吃掉 token，所以這裡**不做** parse_attention，
       直接量原始字串——那才是管線實際會拿到的東西。
    """
    out = []
    # FLUX.1 只看 T5（tokenizer_2）：CLIP 只拿整句的向量，超過 77 截掉是正常的，列出來反而像出錯
    attrs = getattr(pipe, "_coco_count_tokenizers", None) or ("tokenizer", "tokenizer_2")
    for label, text in (("正面", prompt), ("負面", negative)):
        if text is None:
            continue
        for attr in attrs:
            tok = getattr(pipe, attr, None)
            if tok is None:
                continue
            limit = getattr(tok, "model_max_length", None)
            if not isinstance(limit, int) or limit <= 0 or limit > 100000:
                continue
            try:
                n = len(tok(text, truncation=False).input_ids)
            except Exception:
                continue
            name = label if attr == attrs[0] else f"{label}（第二組）"
            out.append((name, n, limit))
    return out


def field_label(name):
    """count_tokens 的欄位名（正面、負面、正面（第二組）；前端拿它分正負面，所以本身不翻）→ 給使用者看的名稱"""
    import i18n
    out = name.replace("（第二組）", i18n.tr("eng.c.second"))
    return out.replace("正面", i18n.tr("eng.c.pos")).replace("負面", i18n.tr("eng.c.neg"))


def native_overflow(pipe, prompt, negative=""):
    """只回會被截斷的欄位，沒有就是空 list

    **為什麼需要這支。** native 是把原始字串交給管線自己編碼，超過上限
    就直接砍掉後面，而且**不會有任何訊息**。使用者偏好 native 的出圖，
    卻可能一直在不知情的狀況下被丟掉半段提示詞。
    """
    return [(name, n, limit)
            for name, n, limit in count_tokens(pipe, prompt, negative)
            if n > limit and (name.startswith("正面") or negative)]


def describe(pipe, prompt, negative):
    """給使用者看的摘要"""
    tok = pipe.tokenizer
    lines = []
    for label, text in (("正面", prompt), ("負面", negative)):
        chunks = tokenize_weighted(tok, text)
        total = sum(len(c[0]) for c in chunks)
        ws = [w for c in chunks for w in c[1] if abs(w - 1.0) > 1e-6]
        lines.append(f"  {label}：{total} tokens → {len(chunks)} 塊"
                     f"（{len(chunks) * 77} 個條件位置）"
                     + (f"，{len(ws)} 個 token 有加權" if ws else ""))
    return "\n".join(lines)
