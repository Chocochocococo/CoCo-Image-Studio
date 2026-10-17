# -*- coding: utf-8 -*-
"""資源庫：模型、LoRA、VAE、文字編碼器的清單，加上架構、用量、代表圖、收藏、備註與自訂封面。

架構只讀 safetensors 的標頭（開頭一小段 JSON），不載入權重、不需要 torch；
230 個 LoRA 第一次約 0.6 秒，之後依檔案修改時間快取。
"""

import json
import struct
import threading
from pathlib import Path

from .library import thumb_ver
from .resources import _real, kind_overrides, model_arch, res_scan
from .state import C, res_stem

import i18n
from i18n import tr

NOTES_FILE = getattr(C, "ASSET_NOTES_FILE", None) or (Path(C.DB_PATH).parent / "asset_notes.json")
COVER_DIR = getattr(C, "COVER_DIR", None) or (Path(C.DB_PATH).parent / "covers")
COVER_MAX = 1024          # 上傳的封面縮到最長邊這麼大
COVER_BYTES = 6 * 2 ** 20     # 請求上限是 8 MB（state.MAX_REQUEST_BODY），前端會先縮到 1024 再送
_NOTES_LOCK = threading.Lock()
_ARCH = {}          # (路徑, 修改時間) → 架構


# ---------------------------------------------------------------- 架構
def _header(path):
    """safetensors 標頭：{張量名: {dtype, shape, …}, "__metadata__": {…}}"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100_000_000:
            return {}
        return json.loads(f.read(n))


# FLUX.1 的架構代號（core/flux1.py：dev 系、schnell、Kontext）
FLUX1_ARCHS = ("flux1", "flux1s", "kontext")
# FLUX.2 klein 的 LoRA：隱藏層寬度 → 架構（4B 是 3072、9B 是 4096）
KLEIN_WIDTH = {3072: "klein4", 4096: "klein9"}


def _klein_lora(h, keys, base):
    """FLUX.2 klein 的 LoRA → klein4／klein9；不是回 ""、是 FLUX.2 但認不出哪一個回 other。
    BFL 命名（double_blocks、single_blocks）FLUX.1 也用：FLUX.1 每層有自己的 img_mod、雙流 19 層，
    klein 的 modulation 全模型共用、雙流最多 8 層。diffusers 命名的 to_qkv_mlp_proj 只有 FLUX.2 有"""
    import re
    joined = "\n".join(keys)
    named = "klein" in base or any(x in base for x in ("flux.2", "flux2", "flux-2", "flux_2"))
    if "flux" in base and not named:
        return ""                                  # 中繼資料寫明是 FLUX.1
    if not named and "to_qkv_mlp_proj" not in joined:
        if "double_blocks" not in joined or "img_mod" in joined:
            return ""
        idx = [int(m) for m in re.findall(r"double_blocks[._](\d+)[._]", joined)]
        if not idx or max(idx) >= 8:
            return ""
    for k in keys:
        if (k.endswith((".lora_down.weight", ".lora_A.weight"))
                and any(x in k for x in ("img_attn", "attn.to_q.", "to_qkv_mlp_proj", "linear1"))):
            shape = (h.get(k) or {}).get("shape") or []
            if len(shape) == 2:
                return KLEIN_WIDTH.get(shape[1], "other")
    return "other"


_FLUX2_NAMES = ("flux.2", "flux2", "flux-2", "flux_2", "klein")


def _flux1_lora(keys, base):
    """FLUX.1 的 LoRA（dev、schnell 共用）。中繼資料寫 flux 而不是 FLUX.2 的就是；沒寫的看張量：
    BFL 命名的（klein 的先認過了：雙流 8 層以內、沒有每層的 img_mod，見 _klein_lora）；
    diffusers 命名有 single_transformer_blocks、沒有 FLUX.2 才有的 to_qkv_mlp_proj"""
    if any(x in base for x in _FLUX2_NAMES):
        return False
    if "flux" in base:
        return True
    if base.strip():
        return False                                    # 寫了別的架構
    joined = "\n".join(keys)
    if "double_blocks" in joined:
        return True
    return "single_transformer_blocks" in joined and "to_qkv_mlp_proj" not in joined


def arch_from_header(kind, h, name=""):
    """依標頭判斷適用的架構：sdxl / sd15 / anima / zimage / krea2 / klein4、klein9（FLUX.2 klein）/
    flux1（FLUX.1 的 LoRA，dev、schnell 共用）/ sd（SD 系列通用的 VAE、CLIP）/ clipl（CLIP-L：SD 系列與 FLUX.1）/
    t5（T5-XXL，FLUX.1 的文字編碼器）/ qwenimage（Qwen Image 的 VAE，Anima 與 Krea 2 共用）/ qwen21（Qwen Image 2.1）/
    flux2（FLUX.2 的 VAE，klein 4B、9B 共用）/ other / ""（認不出來）。
    VAE 的 zimage＝FLUX.1 的 VAE（ae.safetensors，Z-Image 與 FLUX.1 共用）

    LoRA 先看訓練工具寫的中繼資料（kohya 的 ss_base_model_version、modelspec），
    沒有的話看張量名稱：SDXL 有兩個文字編碼器（te1 / te2）、SD1.5 只有一個（te）、
    Anima 的去噪網路是 DiT，名稱裡是 blocks 加 adaln_modulation / cross_attn。
    """
    meta = h.get("__metadata__") or {}
    keys = [k for k in h if k != "__metadata__"]
    joined = "\n".join(keys[:4000])
    if kind == "lora":
        base = " ".join(str(meta.get(k, "")) for k in
                        ("ss_base_model_version", "modelspec.architecture")).lower()
        if "anima" in base:
            return "anima"
        if "sdxl" in base or "stable-diffusion-xl" in base:
            return "sdxl"
        if "sd_v1" in base or "stable-diffusion-v1" in base:
            return "sd15"
        if "zimage" in base.replace("-", "").replace("_", ""):
            return "zimage"
        if "krea" in base:
            return "krea2"
        if "qwen" in base and any(x in base for x in ("2.1", "2_1", "21")):
            return "qwen21"
        klein = _klein_lora(h, keys, base)
        if klein:
            return klein
        if _flux1_lora(keys, base):
            return "flux1"
        if base.strip() and any(x in base for x in ("flux", "sd3", "wan", "qwen")):
            return "other"
        if "lora_te1_" in joined or "lora_te2_" in joined:
            return "sdxl"
        # Krea 2 的注意力層叫 attn.wq／wk／wv（Anima 的是 self_attn.q_proj），文字融合層叫 txtfusion
        if "txtfusion" in joined or "_attn_wq" in joined or ".attn.wq." in joined:
            return "krea2"
        # Qwen Image 2.1：單流，MLP 是 gate_up（ComfyUI）或 gate_layer／proj（diffusers）、沒有 txt_mlp
        # （第一代 Qwen-Image 是雙流，有 txt_mlp、MLP 叫 net.0.proj）
        if "img_mlp" in joined and "txt_mlp" not in joined and ("gate_up" in joined or "gate_layer" in joined):
            return "qwen21"
        if "blocks_0_adaln_modulation" in joined or "blocks.0.adaln_modulation" in joined \
                or "lora_unet_blocks_" in joined or "diffusion_model.blocks." in joined:
            return "anima"
        if "context_refiner" in joined or "noise_refiner" in joined:
            return "zimage"                   # Z-Image 的 transformer（CoCo 還不能套，資源庫照樣標出來）
        if "double_blocks" in joined:
            return "other"
        if "input_blocks" in joined:
            # SDXL 的 UNet 每層有多個 transformer block，SD1.5 只有一個
            return "sdxl" if "transformer_blocks_1" in joined or "transformer_blocks.1." in joined else "sd15"
        if "lora_te_" in joined or "down_blocks" in joined:
            return "sd15"
        return ""
    if kind == "vae":
        # Flux.1 的 VAE（Z-Image 用的就是它）：16 個 latent 通道；Flux.2 的是 32 個（klein 用）
        conv = (h.get("decoder.conv_in.weight") or {}).get("shape") or []
        if len(conv) == 4 and conv[1] == 16:
            return "zimage"
        if len(conv) == 4 and conv[1] == 32 and "bn.running_mean" in h:
            return "flux2"
        base = str(meta.get("modelspec.architecture", "")).lower()
        if "flux" in base or "bn.running_mean" in joined:
            return "other"
        # Qwen Image 2.1 的 VAE：輸入 4 個通道（RGBA），要先認出來（命名跟下面的 Qwen Image VAE 一樣從 conv1 開始）
        enc = (h.get("encoder.conv1.weight") or {}).get("shape") or []
        if (len(enc) == 5 and enc[1] == 4) or "qwen_image_2.1" in str(meta.get("modelspec.architecture", "")):
            return "qwen21"
        # Qwen Image 的 VAE（Wan 2.1 系，Anima 與 Krea 2 共用）：名稱從 conv1 / encoder.downsamples 開始
        if keys and ("conv1.bias" in h or "encoder.downsamples.0.residual.0.gamma" in joined):
            return "qwenimage"
        if "decoder.conv_in" in joined or "first_stage_model." in joined:
            return "other" if "Flux" in str(meta) else "sd"
        return ""
    if kind == "embedding":
        # SDXL 的有 clip_l 與 clip_g 兩份；SD1.5 的是 A1111 的 emb_params（.pt 讀不了標頭，認不出來）
        if "clip_g" in h and "clip_l" in h:
            return "sdxl"
        if "emb_params" in h or "clip_l" in h:
            return "sd15"
        return ""
    if kind == "te":
        if "text_model.embeddings" in joined or "text_model.encoder" in joined:
            # CLIP：SDXL / SD1.5 自帶，單獨換用的情況少；CLIP-L（寬 768）FLUX.1 要另外放一個
            emb = (h.get("text_model.embeddings.token_embedding.weight") or {}).get("shape") or []
            return "clipl" if len(emb) == 2 and emb[1] == 768 else "sd"
        # T5-XXL（FLUX.1 的文字編碼器，寬 4096）：Wan 用的 umT5 鍵名一樣、詞彙表 25 萬；T5-XL 寬 2048
        t5 = (h.get("shared.weight") or h.get("encoder.embed_tokens.weight") or {}).get("shape") or []
        if len(t5) == 2 and "encoder.block.0.layer.0.SelfAttention.q.weight" in h:
            return "t5" if t5 == [32128, 4096] else "other"
        emb = h.get("model.embed_tokens.weight", {}).get("shape") or []
        # Anima 的文字編碼器是 Qwen3 0.6B：詞彙表約 15 萬、隱藏層 1024
        if len(emb) == 2 and 150_000 <= emb[0] <= 160_000 and emb[1] == 1024:
            return "anima"
        # Z-Image 的是 Qwen3-4B（隱藏層 2560）；Qwen3-VL 4B 同樣 2560，但多了看圖的部分（Krea 2 用的）
        if len(emb) == 2 and 150_000 <= emb[0] <= 160_000 and emb[1] == 2560:
            return "krea2" if any(k.startswith(("model.visual.", "visual.")) for k in keys) else "zimage"
        # Qwen Image 2.1 的是 Qwen3-VL 8B（4096）；同樣 4096、沒有看圖部分的是 Qwen3 8B（FLUX.2 klein 9B 用的）
        if len(emb) == 2 and 150_000 <= emb[0] <= 160_000 and emb[1] == 4096:
            return "qwen21" if any(k.startswith(("model.visual.", "visual.")) for k in keys) else "klein9"
        return "other" if keys else ""
    return ""


def asset_arch(kind, path):
    p = Path(path)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return ""
    if key in _ARCH:
        return _ARCH[key]
    if kind == "model":
        arch = model_arch(str(p)) if p.is_file() else ("anima" if "anima" in p.name.lower() else "")
    elif p.suffix.lower() not in (".safetensors", ".sft"):
        arch = ""
    else:
        try:
            arch = arch_from_header(kind, _header(p), p.name)
        except (OSError, ValueError, struct.error):
            arch = ""
    _ARCH[key] = arch
    return arch


def compatible(kind, asset, model):
    """這個資源能不能搭配目前的模型架構。認不出來的一律當成可以（不擋，只是不保證）"""
    if not asset or not model or kind == "model":
        return True
    if asset in ("sd", "clipl"):
        return model in ("sdxl", "sd15")
    if asset in ("flux1", "t5"):
        return model in FLUX1_ARCHS                 # FLUX.1 的 LoRA 大多 dev、schnell、Kontext 都能套
    if asset == "qwenimage":
        return model in ("anima", "krea2")
    if asset == "flux2":
        return model in ("klein4", "klein9")
    if asset == "zimage" and kind == "te":
        return model in ("zimage", "klein4")          # Qwen3-4B：Z-Image 與 FLUX.2 klein 4B 共用同一個檔
    if asset == "zimage" and kind == "vae":
        return model in ("zimage", *FLUX1_ARCHS)      # FLUX.1 的 VAE（ae.safetensors）：Z-Image 用的就是它
    return asset == model


# ---------------------------------------------------------------- 拆分式模型的零件
# 模型檔只有擴散模型、VAE 與文字編碼器要另外選的架構（FLUX.1 也有全包的檔：自帶的零件沒另外選就用檔案裡的）
SPLIT_ARCHS = ("anima", "zimage", "krea2", "qwen21", "klein4", "klein9", *FLUX1_ARCHS)
# clip：FLUX.1 的 CLIP-L（不佔生成台的欄位，見 clip_for）
def part_label(kind):
    """零件的名稱（VAE、文字編碼器、CLIP-L），照介面語言"""
    return tr(f"srv.part.{kind}")


ARCH_TITLE = {"anima": "Anima", "zimage": "Z-Image", "krea2": "Krea 2", "qwen21": "Qwen Image 2.1",
              "klein4": "FLUX.2 klein 4B", "klein9": "FLUX.2 klein 9B", "flux1": "FLUX.1 dev", "flux1s": "FLUX.1 schnell",
              "kontext": "FLUX.1 Kontext"}
# 一個都找不到時，告訴使用者要放哪個檔（ComfyUI 的檔名）
# （名稱, 檔名）；VAE 寫成「某某的 VAE」，照介面語言組（part_hint）
PART_HINT = {("anima", "vae"): ("Qwen Image", "qwen_image_vae"), ("anima", "te"): ("Qwen3 0.6B", "qwen_3_06b_base"),
             ("zimage", "vae"): ("Flux.1", "ae.safetensors"), ("zimage", "te"): ("Qwen3-4B", "qwen_3_4b"),
             ("krea2", "vae"): ("Qwen Image", "qwen_image_vae"), ("krea2", "te"): ("Qwen3-VL 4B", "qwen3vl_4b"),
             ("qwen21", "vae"): ("Qwen Image 2.1", "qwen_image_2.1_vae"), ("qwen21", "te"): ("Qwen3-VL 8B", "qwen3vl_8b"),
             ("klein4", "vae"): ("FLUX.2", "flux2-vae"), ("klein4", "te"): ("Qwen3-4B", "qwen_3_4b"),
             ("klein9", "vae"): ("FLUX.2", "flux2-vae"), ("klein9", "te"): ("Qwen3-8B", "qwen_3_8b"),
             **{(a, k): v for a in FLUX1_ARCHS for k, v in (("vae", ("FLUX.1", "ae.safetensors")),
                                                            ("te", ("T5-XXL", "t5xxl_fp16")), ("clip", ("CLIP-L", "clip_l")))}}


def part_hint(arch, kind):
    """找不到零件時說要放哪個檔，例如「Qwen Image 的 VAE（qwen_image_vae）」"""
    hit = PART_HINT.get((arch, kind))
    if not hit:
        return part_label(kind)
    return tr("srv.part.vaeHint" if kind == "vae" else "srv.part.hint", name=hit[0], file=hit[1])
# 官方工作流程用的檔名開頭：資源庫裡相符的有好幾個時，對得上的只有一個就用它
# （Krea 2 的 VAE：qwen_image_vae 與 wan_2.1_vae 都裝得上，官方用 qwen_image_vae）
PART_NAME = {("anima", "vae"): "qwen_image_vae", ("anima", "te"): "qwen_3_06b",
             ("zimage", "vae"): "ae", ("zimage", "te"): "qwen_3_4b",
             ("krea2", "vae"): "qwen_image_vae", ("krea2", "te"): "qwen3vl_4b",
             ("qwen21", "vae"): "qwen_image_2.1_vae", ("qwen21", "te"): "qwen3vl_8b",
             ("klein4", "vae"): "flux2-vae", ("klein4", "te"): "qwen_3_4b",
             ("klein9", "vae"): "flux2-vae", ("klein9", "te"): "qwen_3_8b",
             **{(a, k): v for a in FLUX1_ARCHS for k, v in (("vae", "ae"), ("te", "t5xxl_fp16"), ("clip", "clip_l"))}}


def part_candidates(kind, arch):
    """資源庫裡認得出來、跟這個架構相符的 VAE／文字編碼器（認不出來的不算，免得選錯）。
    kind＝clip：文字編碼器資料夾裡的 CLIP-L（FLUX.1 用）"""
    out = []
    scan = "te" if kind == "clip" else kind
    for it in res_scan(scan):
        a = asset_arch(scan, it["path"])
        ok = a == "clipl" if kind == "clip" else \
            (a and a not in ("other", "partial", "sd") and compatible(kind, a, arch))
        if ok:
            out.append({"name": it["name"], "path": it["path"]})
    return out


def builtin_parts(model):
    """全包的模型檔（FLUX.1 的 checkpoint）自帶哪些零件：{"vae", "te", "clip"} 裡的幾個"""
    from core.flux1 import builtin_parts as parts
    try:
        keys = [k for k in _header(model) if k != "__metadata__"]
    except (OSError, ValueError, struct.error):
        return set()
    return {k for k, v in parts(keys).items() if v}


def clip_for(arch, builtin=False):
    """FLUX.1 的 CLIP-L（不佔生成台的欄位）：資源庫裡的（好幾個時檔名是 clip_l 的優先，不然照檔名排第一個）；
    沒有就用模型檔自帶的（builtin，回空字串）；都沒有說要放哪個檔。回傳 {"path"} 或 {"error"}"""
    found = sorted(part_candidates("clip", arch), key=lambda f: f["name"].lower())
    want = [f for f in found if res_stem(f["name"]) == PART_NAME[(arch, "clip")]]
    if want or found:
        return {"path": (want or found)[0]["path"]}
    if builtin:
        return {"path": ""}
    return {"error": tr("srv.part.needClip", arch=ARCH_TITLE.get(arch, arch))}


def fill_parts(arch, current, builtin=()):
    """拆分式模型生成前補上沒選的零件。current＝{"vae": 設定值, "te": 設定值}（每個架構一格的 dict，或舊的單一字串）。
    這個架構還沒選、或選的檔案不見了：資源庫裡相符的只有一個就用它；好幾個時檔名跟官方一樣的只有一個就用它（PART_NAME），
    不然請使用者選；一個都沒有時，模型檔自帶的（builtin，FLUX.1 的全包檔）就用檔案裡的，不然說要放哪個檔。
    回傳 {"vae", "te"（補好的設定值）, "picked": [(種類, 檔名, 換掉的舊檔名或 "")]} 或 {"error"}"""
    out, picked = {}, []
    for kind in ("vae", "te"):
        val = current.get(kind)
        by_arch = dict(val) if isinstance(val, dict) else ({} if not val else {arch: val})
        cur = by_arch.get(arch) or ""
        if cur and Path(cur).is_file():
            out[kind] = val
            continue
        found = part_candidates(kind, arch)
        want = PART_NAME.get((arch, kind))
        if len(found) > 1 and want:
            same = [f for f in found if res_stem(f["name"]) == want]
            named = same or [f for f in found if res_stem(f["name"]).startswith(want + "_")]
            found = named if len(named) == 1 else found
        label, title = part_label(kind), ARCH_TITLE.get(arch, arch)
        if not found and kind in builtin:
            if cur:                             # 選的不見了：改用模型檔自帶的（記成空的＝沒選）
                by_arch[arch] = ""
                picked.append((kind, tr("srv.part.builtin"), Path(cur).name))
            out[kind] = by_arch if cur else val
            continue
        if len(found) > 1:
            names = i18n.join(f["name"] for f in found[:3]) + ("…" if len(found) > 3 else "")
            return {"error": tr("srv.part.pickOne", arch=title, part=label, n=len(found), names=names)}
        if not found:
            gone = tr("srv.part.gone", name=Path(cur).name) if cur else ""
            return {"error": gone + tr("srv.part.none", arch=title, part=label, hint=part_hint(arch, kind))}
        by_arch[arch] = found[0]["path"]
        out[kind] = by_arch
        picked.append((kind, found[0]["name"], Path(cur).name if cur else ""))
    out["picked"] = picked
    return out


# ---------------------------------------------------------------- 收藏、備註、最近用過
def load_notes():
    try:
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def note_key(kind, name):
    """用「種類:檔名主體」當鍵：檔案搬到別的資料夾，收藏與備註還在"""
    return f"{kind}:{res_stem(name)}"


def _save_notes(d):
    tmp = NOTES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(NOTES_FILE)


def _edit(kind, name, fn):
    """改一筆：fn(cur) 就地修改；空值拿掉，全空就整筆刪掉"""
    k = note_key(kind, name)
    with _NOTES_LOCK:
        d = load_notes()
        cur = dict(d.get(k) or {})
        fn(cur)
        cur = {x: y for x, y in cur.items() if y not in (None, "", False, 0)}
        if cur:
            d[k] = cur
        else:
            d.pop(k, None)
        _save_notes(d)
    return cur


def update_note(kind, name, fav=None, note=None, touch=False, triggers=None, reset_triggers=False):
    """triggers：自己改的觸發詞（清單；空清單＝沒有觸發詞，也會記住）；reset_triggers：用回檔案裡的"""
    import time

    def fn(cur):
        if fav is not None:
            cur["fav"] = bool(fav)
        if note is not None:
            cur["note"] = str(note)[:4000]
        if reset_triggers:
            cur.pop("triggers", None)
        elif triggers is not None:
            cur["triggers"] = [str(x).strip()[:300] for x in triggers if str(x).strip()][:50]
        if touch:
            cur["last"] = time.time()
            cur["picks"] = int(cur.get("picks") or 0) + 1
    return _edit(kind, name, fn)


# ---------------------------------------------------------------- 自訂封面
# 優先順序：自己指定的 → 模型旁邊的 .preview 圖（civitai、StabilityMatrix 下載的）→ 圖庫裡最新用過的一張。
# 指定圖庫的圖記完整路徑（跟 Recipe 的封面一樣，搬移時 rename_paths 跟著改）；
# 上傳的圖縮小後存在 covers/，檔名是內容的雜湊。模型資料夾裡的檔案一律不動
def set_cover(kind, name, image=None, file=None):
    """image：圖庫裡那張圖的路徑；file：covers/ 裡的檔名；兩個都沒給＝用回預設"""
    def fn(cur):
        cur.pop("cover", None)
        cur.pop("cover_file", None)
        if image:
            cur["cover"] = str(image)
        elif file:
            cur["cover_file"] = Path(str(file)).name
    return _edit(kind, name, fn)


def save_upload(data_url):
    """上傳的封面（data URL）→ 縮小存成 WebP，回傳 {file} 或 {error}"""
    import base64
    import binascii
    import hashlib
    import io

    from PIL import Image
    if not isinstance(data_url, str) or not data_url.startswith("data:image/") or "," not in data_url:
        return {"error": tr("srv.coverNotImage")}
    b64 = data_url.split(",", 1)[1]
    if len(b64) > COVER_BYTES * 4 // 3 + 8:
        return {"error": tr("srv.coverTooBig")}
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        im.load()
    except (binascii.Error, ValueError, OSError, Image.DecompressionBombError):
        return {"error": tr("srv.imgUnreadable")}
    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    im.thumbnail((COVER_MAX, COVER_MAX), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=88, method=4)
    data = buf.getvalue()
    name = hashlib.sha1(data).hexdigest()[:16] + ".webp"
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    (COVER_DIR / name).write_bytes(data)
    return {"file": name}


def custom_cover(kind, path, notes=None):
    """自己指定的封面檔（不存在了就當沒有）"""
    n = (load_notes() if notes is None else notes).get(note_key(kind, Path(str(path)).name)) or {}
    if n.get("cover_file"):
        f = COVER_DIR / Path(n["cover_file"]).name
        if f.is_file():
            return f
    if n.get("cover"):
        f = Path(n["cover"])
        if f.is_file():
            return f
    return None


def forget_covers(paths):
    """當封面的圖丟掉或刪掉了：換回預設封面（全空的那筆整筆拿掉，跟 _edit 一樣）"""
    gone = {str(Path(p)).lower() for p in paths or []}
    if not gone:
        return
    with _NOTES_LOCK:
        d = load_notes()
        changed = False
        for key in list(d):
            n = d[key]
            if n.get("cover") and str(Path(n["cover"])).lower() in gone:
                n.pop("cover")
                changed = True
                if not n:
                    del d[key]
        if changed:
            _save_notes(d)


def rename_paths(mapping):
    """圖庫搬移了檔案：當封面的那張跟著改路徑"""
    if not mapping:
        return
    low = {str(Path(a)).lower(): b for a, b in mapping.items()}
    with _NOTES_LOCK:
        d = load_notes()
        changed = False
        for n in d.values():
            nb = low.get(str(Path(n["cover"])).lower()) if n.get("cover") else None
            if nb:
                n["cover"] = nb
                changed = True
        if changed:
            _save_notes(d)


def used_images(con, kind, key, limit=48):
    """用過這個模型／LoRA 的圖（最新的在前），給挑封面用"""
    from .library import build_query
    if kind not in ("model", "lora") or not key:
        return []
    where, args, _ = build_query({kind: key})
    rows = con.execute(f"SELECT id, path, mtime FROM images{where} ORDER BY mtime DESC LIMIT ?",
                       args + [int(limit)]).fetchall()
    return [{"id": r[0], "v": thumb_ver(r[1], r[2])} for r in rows]


# ---------------------------------------------------------------- 圖庫用量
def usage(con):
    """每個模型、LoRA 在圖庫裡用過幾張，以及最新的一張（沒有封面時拿來當代表圖）"""
    out = {"model": {}, "lora": {}}
    rows = con.execute("SELECT id, path, mtime, model, loras FROM images ORDER BY mtime DESC")
    for rid, path, mtime, model, loras in rows:
        pic = None
        if model:
            k = res_stem(model)
            u = out["model"].setdefault(k, {"uses": 0, "sample": None})
            u["uses"] += 1
            if u["sample"] is None:
                pic = pic or {"id": rid, "v": thumb_ver(path, mtime)}
                u["sample"] = pic
        try:
            items = json.loads(loras or "[]")
        except ValueError:
            items = []
        seen = set()
        for it in items if isinstance(items, list) else []:
            k = res_stem(it.get("file") if isinstance(it, dict) else "")
            if not k or k in seen:
                continue
            seen.add(k)
            u = out["lora"].setdefault(k, {"uses": 0, "sample": None})
            u["uses"] += 1
            if u["sample"] is None:
                pic = pic or {"id": rid, "v": thumb_ver(path, mtime)}
                u["sample"] = pic
    return out


def asset_list(con, kind, model_arch_now):
    notes = load_notes()
    use = usage(con).get(kind, {}) if kind in ("model", "lora") else {}
    moved = kind_overrides()
    items = []
    for it in res_scan(kind):
        stem = res_stem(it["name"])
        arch = asset_arch(kind, it["path"])
        n = notes.get(note_key(kind, it["name"])) or {}
        u = use.get(stem) or {}
        mine = custom_cover(kind, it["path"], notes)
        # 觸發詞：自己改過的優先（記在 asset_notes），不然用模型旁資料檔裡的
        edited = isinstance(n.get("triggers"), list)
        items.append({
            "name": it["name"], "path": it["path"], "rel": it["rel"], "root": it["root"],
            "size": it["size"], "cover": bool(mine) or it["cover"],
            "triggers": n["triggers"] if edited else (it.get("triggers") or []),
            "fileTriggers": it.get("triggers") or [], "trigEdited": edited,
            # 自訂封面換了網址也要跟著變，瀏覽器才不會拿快取裡的舊圖
            "moved": bool(moved) and _real(it["path"]) in moved,
            "custom": bool(mine), "sidecar": it["cover"], "coverv": f"{mine.name}-{int(mine.stat().st_mtime)}" if mine else "",
            "kindTag": it.get("kindTag"), "key": stem, "arch": arch,
            "ok": compatible(kind, arch, model_arch_now),
            "uses": u.get("uses", 0), "sample": u.get("sample"),
            "fav": bool(n.get("fav")), "note": n.get("note", ""),
            "last": n.get("last", 0), "picks": n.get("picks", 0),
        })
    return items
