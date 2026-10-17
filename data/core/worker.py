# -*- coding: utf-8 -*-
"""
worker.py —— 生成用的常駐工作行程

以獨立行程執行，跟工作台用 stdin / stdout 的 JSON 行協定溝通。

為什麼要獨立行程：
  - 生成是重度 CUDA 操作，原生層崩潰會直接殺掉整個行程。
    放在工作台裡的話，一次崩潰整個介面就沒了。
  - 模型常駐在這個行程，生成之間不用重載。
  - 要釋放 VRAM 只要結束這個行程，不必關掉工作台。

協定（每行一個 JSON）
  進來：
    {"cmd":"load", "model":..., "loras":[[path,weight],...],
     "vram_mode":..., "vae_path":..., "sampler":..., "cond_mode":...}
    {"cmd":"gen", "id":..., "prompt":..., "negative":..., "width":..., ...}
    {"cmd":"ping"}
    {"cmd":"quit"}
  出去：
    {"type":"log",     "msg":...}
    {"type":"ready"}
    {"type":"progress","id":...,"step":n,"total":m}
    {"type":"done",    "id":...,"path":...,"seed":...,"secs":...}
    {"type":"skip",    "id":...,"msg":...}      批次修臉沒找到臉，這張不做
    {"type":"error",   "id":...,"msg":...}
"""

import json
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import i18n  # noqa: E402  訊息照介面語言（啟動時讀 COCO_LANG，每個指令也帶 lang）
import certs  # noqa: E402

certs.use_system_certs()   # 防毒軟體掃 HTTPS 時 huggingface_hub 才下載得到（certs.py）；要在 diffusers 之前

# 超過 77 token 的提示詞 CoCo 自己分段（A1111 的做法，見 conditioning.py），
# tokenizer 那句「Token indices sequence length is longer…」是預期中的，不要讓它出現在介面上
import logging  # noqa: E402
import warnings  # noqa: E402
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# stderr 的每一行，工作台會分成警告與錯誤（server/worker_client.stderr_kind）。
# diffusers 的 logger 只印訊息本身，Python 的警告會多印一行原始碼，兩種都分不出來，
# 這裡統一成一行、帶前綴
warnings.formatwarning = lambda message, category, filename, lineno, line=None: f"[{category.__name__}] {message}\n"


def _prefix_diffusers_logs():
    try:
        from diffusers.utils import logging as dlog
        dlog.get_logger("diffusers")                 # 讓 diffusers 先裝好它自己的 handler
        for h in logging.getLogger("diffusers").handlers:
            h.setFormatter(logging.Formatter("[diffusers] %(message)s"))
    except Exception:
        pass


_prefix_diffusers_logs()

STATE = {"pipe": None, "cfg": {}, "loras": []}

# 這些設定不需要重新載入模型，但每次 load 指令都必須更新；否則 UI 已切換，
# worker 還會沿用第一次載模時的值。
DYNAMIC_CFG = ("sampler", "schedule", "shift", "cond_mode",
               "low_decode", "tile_vae")

# Windows 的管道預設用系統編碼（繁中是 cp950），寫中文會直接拋
# UnicodeEncodeError 把整個行程弄死。強制 UTF-8。
for _s in ("stdin", "stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_OUT_LOCK = threading.Lock()     # 探索的模型閒置卸載是另一個執行緒在記紀錄：一行一行寫，不能交錯


def out(obj):
    with _OUT_LOCK:
        try:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            # 連寫都寫不出去就退回純 ASCII，寧可訊息難看也不要讓行程死掉
            try:
                sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
                sys.stdout.flush()
            except Exception:
                pass


def log(msg):
    out({"type": "log", "msg": str(msg)})


def do_load(m):
    import pipeline as P

    loras = [(str(p), float(w)) for p, w in (m.get("loras") or [])]

    # 只有模型本身或 VRAM 設定變了才需要重新載入。
    # LoRA 換了就地增減即可 —— 融合是可逆的，用負權重還原就好。
    same_model = (STATE["pipe"] is not None
                  and STATE["cfg"].get("model") == m.get("model")
                  and STATE["cfg"].get("vram_mode") == m.get("vram_mode")
                  and STATE["cfg"].get("vae_path") == m.get("vae_path")
                  and STATE["cfg"].get("te_path") == m.get("te_path")
                  and STATE["cfg"].get("clip_path") == m.get("clip_path")
                  and bool(STATE["cfg"].get("vae_fp32")) == bool(m.get("vae_fp32"))
                  and bool(STATE["cfg"].get("fp8_weights")) == bool(m.get("fp8_weights")))
    # Qwen 修圖帶了其他參考圖、主模型整個放卻放不下：重新載入、改成邊算邊搬（途中才改會出錯，見 qwen21.needs_reload）
    refs = int(m.get("edit_refs") or 0)
    if same_model and refs and P._qwen21().needs_reload(STATE["pipe"], refs):
        log(i18n.tr("eng.w.refsReload", n=refs))
        same_model = False
    if same_model:
        P.apply_scheduler(STATE["pipe"], m.get("sampler"),
                          m.get("schedule"), log, shift=m.get("shift"))
        if [(p, w) for p, w in STATE["loras"]] != loras:
            P.apply_loras(STATE["pipe"], loras, log, STATE["loras"])
        for key in DYNAMIC_CFG:
            if key in m:
                STATE["cfg"][key] = m[key]
        out({"type": "ready"})
        return

    _release_joycaption()                  # JoyCaption（8.6 GB）跟生成模型一起放不下：要生圖就先讓出來
    _release_rembg()
    if STATE["pipe"] is not None:
        log(i18n.tr("eng.w.reload"))
        STATE["pipe"] = None
        STATE["loras"] = []
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    pipe = P.load_model(m["model"], m.get("vram_mode", "balanced"), log,
                        vae_path=m.get("vae_path"),
                        te_path=m.get("te_path") or None,
                        vae_fp32=bool(m.get("vae_fp32")),
                        fp8_weights=bool(m.get("fp8_weights")),
                        room=P._qwen21().refs_room(refs),
                        clip_path=m.get("clip_path") or None)
    P.apply_scheduler(pipe, m.get("sampler"), m.get("schedule"), log,
                      shift=m.get("shift"))
    STATE["loras"] = []
    P.apply_loras(pipe, loras, log, STATE["loras"])

    STATE["pipe"] = pipe
    STATE["cfg"] = dict(m)
    # 這次實際用的 VAE 與文字編碼器：存圖時記下來，圖庫存成 Recipe 才帶得到
    try:
        STATE["parts"] = P.used_parts(m["model"], m.get("vae_path"), m.get("te_path"))
    except Exception:
        STATE["parts"] = None
    out({"type": "ready"})


_CLIP_TOK = "unset"          # 快取；找不到時記 None，不要每次重找
_CLIP_LOCK = threading.Lock()


def _clip_tokenizer():
    """獨立載一份 CLIP tokenizer，不碰模型權重

    **為什麼要有這條路。** 為了數 token 而先載入 6 GB 的模型不合理。
    CLIP 的 tokenizer 只是個 1.6 MB 的詞彙表，跟權重無關，實測載入 0.05 秒。

    對照 Forge 的做法（`modules/ui.py` 的 `update_token_counter`）：它是
    `sd_model.get_prompt_lengths_on_ui`，**沒載入模型就直接顯示 `?/?`**。
    Forge 啟動時會自動載模型所以實務上總是有，但這裡的模型是使用者按下
    生成才載的，照抄那個做法等於大部分時間都看不到數字。

    來源是 diffusers 之前為了 from_single_file 抓下來的快取
    （hf_cache/hub/models--*/snapshots/*/tokenizer）。找不到就回 None。
    """
    global _CLIP_TOK
    if _CLIP_TOK != "unset":
        return _CLIP_TOK
    with _CLIP_LOCK:
        if _CLIP_TOK != "unset":     # 暖機執行緒可能已經載好了
            return _CLIP_TOK
        return _load_clip_tokenizer()


def _load_clip_tokenizer():
    """實際去載。**呼叫前必須先拿 _CLIP_LOCK。**

    `_CLIP_TOK` 只在**載完之後**才寫。第一版是一進來就設成 None 再開始載，
    但載入要約 2 秒（transformers 的 import），這段期間 `_clip_tokenizer`
    的快取檢查（`!= "unset"`）就會拿到 None，回報「找不到 tokenizer」。
    暖機執行緒正好把這個窗口撐在使用者打字的當下，所以每次都會踩到。
    """
    global _CLIP_TOK
    found = None
    try:
        import glob
        from transformers import CLIPTokenizer
        pats = ["hf_cache/hub/models--stabilityai--stable-diffusion-xl-*"
                "/snapshots/*/tokenizer",
                "hf_cache/hub/models--*stable-diffusion-v1-5*"
                "/snapshots/*/tokenizer",
                "hf_cache/hub/models--*/snapshots/*/tokenizer"]
        for pat in pats:
            for d in sorted(glob.glob(str(ROOT / pat))):
                try:
                    found = CLIPTokenizer.from_pretrained(d)
                    break
                except Exception:
                    continue
            if found is not None:
                break
    except Exception:
        pass
    _CLIP_TOK = found            # 只有這裡會寫，寫的時候值已經是最終的
    return found


def _fallback_fields(text_pairs, model_path):
    """模型沒載入時的估計值。回 (fields, 說明) —— 算不出來就 fields 是 None

    只有 CLIP 系（SD1.5 / SDXL）估得準。Anima 走 Qwen，上限與詞彙表完全
    不同，拿 CLIP 去估會給出一個看起來很篤定卻是錯的數字——**那比不顯示
    更糟**，所以寧可說「要載入模型」。
    """
    import pipeline as P

    arch = None
    if model_path:
        try:
            arch = P.detect_arch(Path(model_path))
        except Exception:
            arch = None
    if arch == "anima":
        return None, i18n.tr("eng.w.tokLoad", model="Anima", enc="Qwen")
    if arch == "zimage":
        return None, i18n.tr("eng.w.tokLoad", model="Z-Image", enc="Qwen")
    if arch == "krea2":
        return None, i18n.tr("eng.w.tokLoad", model="Krea 2", enc="Qwen")
    if arch == "qwen21":
        return None, i18n.tr("eng.w.tokLoad", model="Qwen Image 2.1", enc="Qwen")
    if arch in P.FLUX2_ARCHS:
        return None, i18n.tr("eng.w.tokLoad", model="FLUX.2 klein", enc="Qwen")
    if arch in P.FLUX1_ARCHS:
        return None, i18n.tr("eng.w.tokLoad", model="FLUX.1", enc="T5")

    tok = _clip_tokenizer()
    if tok is None:
        return None, i18n.tr("eng.w.noTokenizer")

    limit = getattr(tok, "model_max_length", 77)
    fields = []
    for label, text in text_pairs:
        if text is None:
            continue
        try:
            n = len(tok(text, truncation=False).input_ids)
        except Exception:
            continue
        fields.append({"name": label, "n": n, "limit": limit,
                       "over": max(0, n - limit)})
    return fields, ""


def do_tokens(m):
    """算提示詞的 token 數，給介面即時顯示用

    **不生成、不動模型**，只查已載入管線的 tokenizer。

    為什麼要問 worker 而不是在 studio 那端自己算：tokenizer 是跟**實際
    載入的模型**綁在一起的。SDXL 有兩顆 CLIP、Anima 走 Qwen，上限與詞彙表
    都不同；在 studio 端猜一個標準 CLIP 來算，數字會跟真正跑的時候不一樣，
    那比不顯示還糟。

    模型還沒載入時**不是直接放棄**，改用獨立的 CLIP tokenizer 估算
    （見 `_clip_tokenizer`），並標 `estimated`，介面會顯示成約略值。
    為了數 token 而先載入 6 GB 的模型不合理。

    注意 worker 的主迴圈是**串行**的：生成中不會讀 stdin，所以這個查詢
    會被排到生成結束後才處理。呼叫端要自己設 timeout（見 studio 的
    `ask_tokens`）。使用者打提示詞時 worker 通常是閒的，實務上不成問題。
    """
    pipe = STATE["pipe"]
    qid = m.get("qid")
    if pipe is None:
        pairs = [("正面", m.get("prompt", "")), ("負面", m.get("negative", ""))]
        fields, why = _fallback_fields(pairs, m.get("model"))
        if fields is None:
            out({"type": "tokens", "qid": qid, "ready": False, "why": why})
            return
        # 沒載入模型就不知道會走 a1111 還是 native，一律當成會截斷的那個
        # （native）來提醒——少報一次截斷比多報一次難發現。
        out({"type": "tokens", "qid": qid, "ready": True, "estimated": True,
             "mode": "native", "fields": fields})
        return

    import conditioning
    import embeddings
    # 載過的 embedding 一個名字佔好幾個 token，算的時候要先換開
    ti = pipe.__dict__.get("_coco_ti")
    fields = []
    for name, n, limit in conditioning.count_tokens(
            pipe, embeddings.expand(m.get("prompt", ""), ti), embeddings.expand(m.get("negative", ""), ti)):
        fields.append({"name": name, "n": n, "limit": limit,
                       "over": max(0, n - limit)})

    # 實際會用哪種編碼決定了「超過」要不要緊：a1111 會分塊所以不截斷，
    # native 才會。auto 的判定跟 _generate_impl 用的是同一支 inspect。
    mode = STATE["cfg"].get("cond_mode", "auto")
    if mode == "auto":
        try:
            mode = "a1111" if conditioning.inspect(pipe)["ok"] else "native"
        except Exception:
            mode = "native"

    out({"type": "tokens", "qid": qid, "ready": True,
         "mode": mode, "fields": fields})


def _release_cache():
    """去背、找臉用完把 PyTorch 留著的快取還給系統：模型本身很小（幾百 MB，留著下一張不用重載），
    但 1024 的圖跑一次會留下約 2.5 GB 快取，閒著也佔著顯示卡"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def do_detect(m):
    """修臉的找臉（core/faces.py）：不動生成用的管線，YOLO 另外載、另外快取"""
    qid = m.get("qid")
    try:
        import faces
        got = faces.detect(m["path"], m["model"])
        _release_cache()
        out({"type": "detect", "qid": qid, "ready": True, **got})
    except Exception as e:
        out({"type": "detect", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def do_rembg(m):
    """去背（core/bgremove.py）：算出遮罩存成 PNG，套到原圖上由 studio 那端做。模型另外載、另外快取"""
    qid = m.get("qid")
    try:
        import bgremove
        try:
            bgremove.predict(m["path"], m["model"], arch=m.get("arch", "isnet"), prep=m.get("prep", "anime")).save(m["out"], "PNG")
        finally:
            _release_cache()
        out({"type": "rembg", "qid": qid, "ready": True})
    except Exception as e:
        out({"type": "rembg", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def do_cnprep(m):
    """ControlNet 參考圖的預覽（core/controlnet_prep.py）：照生成的大小裁好、做前處理（抽姿勢、深度…）存到 out"""
    qid = m.get("qid")
    try:
        import controlnet_run as CR
        from PIL import Image
        with Image.open(m["path"]) as im:
            src = CR.fit(im, int(m["w"]), int(m["h"]), m.get("resize", "crop"))
        import controlnet_prep as CP
        try:
            img, info = CP.run(src, m.get("prep", "none"), m.get("files") or {}, m.get("low", 100), m.get("high", 200))
        finally:
            _release_cache()
            CR.touch()                     # 看處理結果也算用到：閒置倒數重來
        if m.get("prep") == "none" and m.get("type") in ("lineart", "canny", "softedge"):
            img = CR.ink_on_black(img)
        img.save(m["out"], "PNG")
        out({"type": "cnprep", "qid": qid, "ready": True, **info})
    except Exception as e:
        out({"type": "cnprep", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def do_embed(m):
    """探索的特徵（core/explore.py）：圖 → 存成 .npy 交回；文字 → 直接回傳向量"""
    qid = m.get("qid")
    try:
        import explore
        explore.on_release = lambda: log(i18n.tr("eng.w.relExplore"))
        if m.get("text") is not None:
            out({"type": "embed", "qid": qid, "ready": True, "vec": explore.embed_text(m["model"], m["text"])})
            return
        import numpy as np
        vecs, ok = explore.embed_images(m["model"], m["paths"])
        np.save(m["out"], vecs)
        out({"type": "embed", "qid": qid, "ready": True, "ok": ok})
    except Exception as e:
        out({"type": "embed", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def do_tag(m):
    """打標台的 WD14（core/wdtagger.py）：一批圖 → 每張的標籤、分類、機率（門檻與清理在 studio 那端套）"""
    qid = m.get("qid")
    try:
        import wdtagger
        wdtagger.on_release = lambda: log(i18n.tr("eng.w.relTagger"))
        out({"type": "tag", "qid": qid, "ready": True, "tags": wdtagger.predict(m["model"], m["paths"])})
    except Exception as e:
        out({"type": "tag", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def _release_joycaption():
    import sys
    jc = sys.modules.get("joycaption")
    if jc is not None and jc.loaded():
        jc.release()
        log(i18n.tr("eng.w.relJcForGen"))


def _release_rembg():
    """去背的模型（幾百 MB，用過的那個會留著）：換生成模型時一起讓出來"""
    import sys
    bg = sys.modules.get("bgremove")
    if bg is not None and bg._MODELS:
        bg.release()
        log(i18n.tr("eng.w.relRembgForGen"))


def _release_pipe_for(what):
    """要載入很大的模型（JoyCaption 8.6 GB）前先卸載生成用的模型；下次生圖時主行程會送 load，重新載入"""
    if STATE["pipe"] is None:
        return
    log(i18n.tr("eng.w.releasePipe", what=what))
    STATE["pipe"] = None
    STATE["loras"] = []
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def do_caption(m):
    """打標台的 JoyCaption（core/joycaption.py）：一批圖 → 每張的自然語言描述。直接讀 GGUF，不需要 llama-server"""
    qid = m.get("qid")
    try:
        import joycaption
        joycaption.on_release = lambda: log(i18n.tr("eng.w.relJc"))
        if not joycaption.loaded():
            _release_pipe_for(i18n.tr("eng.w.describe"))
        texts, stats = joycaption.caption_many(m["model"], m["mmproj"], m["paths"], m["prompts"],
                                               max_new=int(m.get("max_new", 300)), temperature=float(m.get("temperature", 0.6)),
                                               top_p=float(m.get("top_p", 0.9)), log=log)
        out({"type": "caption", "qid": qid, "ready": True, "texts": texts, "stats": stats})
    except Exception as e:
        out({"type": "caption", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def do_upscale(m):
    """高清化的「只放大」（core/upscale.py）：放大到指定大小存成 PNG，存檔與中繼資料由 studio 那端做"""
    qid = m.get("qid")
    try:
        import upscale
        try:
            upscale.upscale_to(m["path"], m["size"], m.get("model") or None).save(m["out"], "PNG")
        finally:
            upscale.release()                   # 放大模型與快取用完就還，不佔著顯示卡
        out({"type": "upscale", "qid": qid, "ready": True})
    except Exception as e:
        out({"type": "upscale", "qid": qid, "ready": False, "why": f"{type(e).__name__}: {e}"})


def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timing_summary(marks, calls=0, batch=None, lora_s=None, save_s=None):
    """把分段計時的記號整理成一行紀錄

    marks 是 [開始, 第 1 步結束, …, 第 N 步結束, 生成結束]，單位秒，同一個基準。

    第一步之前那段混著「文字編碼＋準備」與「第一步本身」，拆不開，
    所以用其餘各步的中位數當作一步的時間扣掉。用中位數不用平均，
    是因為 group offload 或第一次呼叫的暖機會讓個別幾步特別慢。
    """
    start, steps, end = marks[0], marks[1:-1], marks[-1]
    parts = []
    if lora_s:
        parts.append(i18n.tr("eng.w.tLora", s=f"{lora_s:.1f}"))
    if not steps:
        parts.append(i18n.tr("eng.w.tGenNoSteps", s=f"{end - start:.1f}"))
    else:
        n = len(steps)
        gaps = sorted(b - a for a, b in zip(steps, steps[1:]))
        per = gaps[len(gaps) // 2] if gaps else steps[0] - start
        prep = max(0.0, steps[0] - start - per)
        denoise = steps[-1] - steps[0] + per
        rate = i18n.tr("eng.w.tRate", r=f"{1 / per:.2f}") if per > 0 else ""
        parts.append(i18n.tr("eng.w.tPrep", s=f"{prep:.1f}"))
        parts.append(i18n.tr("eng.w.tDenoise", n=n, s=f"{denoise:.1f}", per=f"{per:.2f}", rate=rate))
        if gaps and gaps[-1] > per * 1.5:
            parts.append(i18n.tr("eng.w.tSlowest", s=f"{gaps[-1]:.2f}"))
        parts.append(i18n.tr("eng.w.tDecode", s=f"{end - steps[-1]:.1f}"))
    if save_s is not None:
        parts.append(i18n.tr("eng.w.tSave", s=f"{save_s:.1f}"))
    line = i18n.tr("eng.w.timing") + " · ".join(parts)
    if calls and steps:
        # CFG 的正負提示詞是分兩次 batch 1 跑，還是合成一次 batch 2，
        # 會直接反映在每步時間上，所以一起記下來
        line += i18n.tr("eng.w.tCalls", n=f"{calls / len(steps):g}")
        if batch:
            line += f"（batch {batch}）"
    return line


class StageTimer:
    """單張圖的分段計時

    GPU 那段用 CUDA event 量，不插 synchronize。每步都 synchronize 的話，
    CPU 就不能先把下一步排進佇列，量的動作本身會拖慢生成。event 只是在
    串流裡放一個記號，最後統一同步一次再讀。

    另外在去噪網路（Anima 的 transformer、SDXL 的 unet）掛一個
    forward pre-hook，數每步被呼叫幾次、batch 多大。
    """

    def __init__(self, pipe):
        self.events, self.calls, self.batch, self.hook = [], 0, None, None
        try:
            import torch
            self.torch = torch if torch.cuda.is_available() else None
        except Exception:
            self.torch = None
        net = None
        for name in ("transformer", "unet"):
            try:
                net = getattr(pipe, name, None)
            except Exception:
                net = None
            if net is not None:
                break
        if self.torch is not None and net is not None:
            try:
                self.hook = net.register_forward_pre_hook(self._count,
                                                          with_kwargs=True)
            except Exception:
                self.hook = None

    def _count(self, module, args, kwargs):
        # 掛在生成路徑上，拋出例外會讓那張圖失敗，所以一律吞掉
        try:
            self.calls += 1
            if self.batch is None:
                x = kwargs.get("hidden_states", args[0] if args else None)
                if hasattr(x, "shape") and len(x.shape):
                    self.batch = int(x.shape[0])
        except Exception:
            pass

    def mark(self):
        if self.torch is None:
            return
        e = self.torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def summary(self, lora_s=None, save_s=None):
        if self.torch is None or len(self.events) < 2:
            return None
        self.torch.cuda.synchronize()
        first = self.events[0]
        marks = [0.0] + [first.elapsed_time(e) / 1000
                         for e in self.events[1:]]
        return timing_summary(marks, self.calls, self.batch, lora_s, save_s)


FACE_MAX = 12              # 一張圖最多修幾張臉（跟精修頁送來的上限一樣，server/refine.clean_faces）


def _find_faces(image, model_path):
    """找臉（core/faces.py）→ ([{"box": …}, …], 圖的大小)；最多 FACE_MAX 張"""
    import faces
    got = faces.detect(image, model_path)
    return [{"box": f["box"]} for f in got["faces"][:FACE_MAX]], got["size"]


def do_gen(m):
    import pipeline as P

    if STATE["pipe"] is None:
        raise RuntimeError(i18n.tr("eng.w.notLoaded"))

    # 提示詞裡的 <lora:名字:權重> 是逐則生效的，所以每個工作可能帶不同的
    # LoRA 清單。跟目前生效的不一樣就重套——還原是從備份複製回來的，
    # 位元完全一致，不會累積誤差（見 pipeline.restore_weights）。
    want = m.get("loras")
    lora_s = None
    if want is not None:
        want = [(p, float(w)) for p, w in want]
        if sorted(want) != sorted(STATE["loras"]):
            t = time.perf_counter()
            P.apply_loras(STATE["pipe"], want, log, STATE["loras"])
            _cuda_sync()
            lora_s = time.perf_counter() - t

    jid = m.get("id")
    t0 = time.time()

    # 批次精修的修臉：臉在工作裡找（排進佇列時還沒輪到，不先找）。沒有臉就略過這張
    if m.get("faces") == "auto":
        found, size = _find_faces(m["init_image"], m["face_model"])
        if not found:
            log(i18n.tr("eng.w.noFaceSkip", name=Path(m['init_image']).name))
            out({"type": "skip", "id": jid, "msg": i18n.tr("eng.w.noFace")})
            return
        log(i18n.tr("eng.w.facesFound", n=len(found)))
        m["faces"], m["face_size"] = found, size
    last = [0.0]
    timer = StageTimer(STATE["pipe"])

    def on_step(step, total):
        timer.mark()
        now = time.time()
        if now - last[0] > 0.25 or step == total:
            last[0] = now
            out({"type": "progress", "id": jid, "step": step, "total": total})

    # 分塊是整批共用的設定，掛在 cfg 上，補進單筆工作讓 _tile_vae 讀得到
    m.setdefault("tile_vae", STATE["cfg"].get("tile_vae") or "auto")
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None
    timer.mark()
    try:
        img, seed, mode, clip_skip = P.generate(
            STATE["pipe"], m,
            m.get("cond_mode", STATE["cfg"].get("cond_mode", "auto")),
            on_step, log, low_decode=STATE["cfg"].get("low_decode", "auto"))
        timer.mark()
    finally:
        timer.close()
    if torch is not None:
        try:
            torch.cuda.synchronize()
            peak_a = torch.cuda.max_memory_allocated() / 2 ** 30
            peak_r = torch.cuda.max_memory_reserved() / 2 ** 30
            log(i18n.tr("eng.w.vramPeak", a=f"{peak_a:.2f}", r=f"{peak_r:.2f}"))
            # 解碼、編碼留下的大塊快取（Anima 的 Qwen VAE 留數 GB cuDNN workspace、
            # SDXL 解碼也有 2～4 GB）：圖已經回到 CPU，還給系統，不然工作管理員一直停在峰值，
            # 閒著也佔著顯示卡。下一張再要回來只要幾毫秒（Forge 每次生成完也這樣做）
            torch.cuda.empty_cache()
            now_r = torch.cuda.memory_reserved() / 2 ** 30
            log(i18n.tr("eng.w.vramFreed", r=f"{now_r:.2f}"))
        except Exception:
            pass
    pipe = STATE["pipe"]

    save_s = [0.0]          # 存檔花的時間（自動修臉留原圖時存兩次）

    def save(image, req, md, cs):
        t = time.perf_counter()
        try:
            return P.save_image(image, Path(m["out_dir"]), req, seed, md,
                                Path(STATE["cfg"]["model"]).name,
                                STATE["loras"],
                                # Anima：介面上選的名字（_coco_sampler）；實際的排程器記在 coco JSON 的 env
                                sampler=getattr(pipe, "_coco_sampler", None) or getattr(pipe, "_coco_sched_name", None),
                                clip_skip=cs,
                                schedule=getattr(pipe, "_coco_schedule", None),
                                parts=STATE.get("parts"),
                                env=P.render_env(pipe, STATE["cfg"]))
        finally:
            save_s[0] += time.perf_counter() - t

    path = None
    if m.get("auto_face") and not m.get("init_image"):
        t = time.perf_counter()
        img, path = _auto_face(P, img, m, seed, mode, clip_skip, save, jid)
        if path is not None:
            log(i18n.tr("eng.w.autoFaceTime", s=f"{time.perf_counter() - t - save_s[0]:.1f}"))
    if path is None:
        path = save(img, m, mode, clip_skip)
    try:
        line = timer.summary(lora_s, save_s[0])
        if line:
            log(line)
    except Exception as e:
        log(i18n.tr("eng.w.timingFail", e=e))
    out({"type": "done", "id": jid, "path": str(path), "seed": seed,
         "secs": round(time.time() - t0, 1), "conditioning": mode})


def _auto_face(P, img, m, seed, mode, clip_skip, save, jid):
    """生成後自動修臉（像 ADetailer）：找臉、每張臉用局部重繪的「只重畫遮罩附近」修。
    m["auto_face"]：{"model": 偵測模型, "strength": 修改幅度, "blur": 遮罩柔化, "keep": 沒修過的也存}。
    keep：沒修過的照常存，修好的存成它的新版本（精修頁的版本列對照得到）；
    不留：只存修好的，參數照 ADetailer 寫。回傳 (圖, 存好的路徑)；沒有臉時路徑是 None（照常存）"""
    af = m["auto_face"]
    try:
        found, size = _find_faces(img, af["model"])
    except Exception as e:
        log(i18n.tr("eng.w.autoFaceFail", e=f"{type(e).__name__}: {e}"))
        return img, None
    if not found:
        log(i18n.tr("eng.w.autoFaceNone"))
        return img, None
    base = save(img, m, mode, clip_skip) if af.get("keep") else None
    face = {"faces": found, "face_size": size, "face_model": af["model"],
            "mask_blur": int(af.get("blur", 8)), "face_strength": float(af["strength"])}
    fm = dict(m, **face, init_image=str(base) if base else img, strength=float(af["strength"]), seed=seed)
    log(i18n.tr("eng.w.autoFaceN", n=len(found)))
    last = [0.0]

    def on_step(step, total):
        now = time.time()
        if now - last[0] > 0.25 or step == total:
            last[0] = now
            out({"type": "progress", "id": jid, "step": step, "total": total})

    fixed, _, md, cs = P.generate(STATE["pipe"], fm, m.get("cond_mode", STATE["cfg"].get("cond_mode", "auto")),
                                  on_step, log, low_decode=STATE["cfg"].get("low_decode", "auto"))
    # 不留原圖：存的是生成的參數＋修臉的欄位（沒有底圖，不算 img2img）
    return fixed, save(fixed, fm if base else dict(m, **face), md, cs)


def _cn_release_note():
    """ControlNet 閒置卸載時記一行（core/controlnet_run.py 的倒數）"""
    try:
        import controlnet_run
        controlnet_run.on_release = lambda: log(i18n.tr("eng.w.relCn"))
    except Exception:
        pass


def main():
    log(i18n.tr("eng.w.started"))
    _cn_release_note()
    # 背景把 CLIP tokenizer 暖好。第一次載要約 2 秒（大部分是 transformers
    # 的 import），而 studio 那端的查詢只等 1.5 秒——不先暖的話，使用者
    # 打的第一段提示詞一定會逾時，看到「忙碌中」而不是數字。
    warm = threading.Thread(target=_clip_tokenizer, daemon=True)
    warm.start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue

        i18n.set_lang(m.get("lang"))
        cmd = m.get("cmd")
        # 暖機還在 import transformers 時，載模型這類指令又去 import diffusers（裡面也會 import transformers），
        # 兩個執行緒同時載同一個套件會撞壞（實測：剛啟動就按生成，「cannot import name 'AutoImageProcessor'」，
        # 工作行程直接結束）。數 token 本來就在等暖機的鎖，其他的等暖機做完（最多約 2 秒）
        if cmd not in ("tokens", "ping", "quit") and warm.is_alive():
            warm.join()
        try:
            if cmd == "load":
                do_load(m)
            elif cmd == "gen":
                do_gen(m)
            elif cmd == "tokens":
                do_tokens(m)
            elif cmd == "detect":
                do_detect(m)
            elif cmd == "rembg":
                do_rembg(m)
            elif cmd == "upscale":
                do_upscale(m)
            elif cmd == "embed":
                do_embed(m)
            elif cmd == "tag":
                do_tag(m)
            elif cmd == "caption":
                do_caption(m)
            elif cmd == "cnprep":
                do_cnprep(m)
            elif cmd == "ping":
                out({"type": "pong"})
            elif cmd == "quit":
                break
        except Exception as e:
            out({"type": "error", "id": m.get("id"),
                 "msg": f"{type(e).__name__}: {e}"})
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()


if __name__ == "__main__":
    main()
