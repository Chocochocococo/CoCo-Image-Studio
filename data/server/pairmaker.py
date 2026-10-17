# -*- coding: utf-8 -*-
"""成對圖輔助（使用者要求，2026-10-08）：用 Qwen 修圖把一批圖改成「正向版」與「反向版」，給成對圖片的 Slider
（server/train.py 的 sl_mode＝pairs）用。

- **圖從哪來**（`SRC_MODES`）：folder＝選一個資料夾裡的圖；auto＝先用 Slider 的底模照提示詞自動畫底圖（`auto_plan`，
  存在 存放位置/名稱/底圖，旁邊的 .txt 是畫它的提示詞），畫完接著改。自動畫的底圖是底模自己的畫風、內容也多樣：
  2026-10-08 實測用資料集的 12 張（同一種粗墨線畫風）練，±2 時資料集的畫風會漏出來（見 DEVLOG）。
- **做法**（`MODES`）：both＝兩邊都改（預設：Qwen 修圖的痕跡兩邊都有，訓練時在 +1、−1 互相抵掉，LoRA 只學到指示要的差別）；
  orig_neg＝原圖當反向、只改正向；orig_pos＝原圖當正向、只改反向（改的圖數少一半，但修圖的痕跡會一起學進去）。
- **存哪**：資料夾來源存到原資料夾旁邊的 <名稱>_pairs/positive、negative（不放在原資料夾裡面：資料集的圖會連子資料夾一起算）；
  自動畫的存到 存放位置/名稱/positive、negative（底圖在 base）。2026-10-14 以前是中文名稱（_成對、正向、反向、底圖），
  那種資料夾已經在的話照舊用它，做到一半的才接得下去（`_dir`）；用原圖那邊的就直接是原圖的資料夾。檔名跟原圖一樣（副檔名是 .png）、
  旁邊的 .txt 一起複製，兩個資料夾裡同名的就是一對。做好的兩個資料夾加進資料集清單。
- **接著做**：已經有的（底圖、改好的圖）跳過，中斷了再按一次就從沒做的開始。
- **怎麼做**：一張一個工作交給生成子行程（handler.plan_run，修圖走 edit_path，跟精修頁的「用 Qwen 修圖」同一條路），
  參數用那個架構記的那組（`arch_params`），修圖不加品質標籤、都不掛 LoRA。存到暫存資料夾、做好一張搬一張（不進圖庫、
  工作標 kind＝pairs，生成頁的「生成後送去比較」跳過）。修圖的大小縮到約 100 萬像素、邊長對齊 32（Slider 用 768² 訓練，原圖再大也用不到）。
"""

import shutil
import tempfile
import threading
import time
from pathlib import Path

from i18n import tr

MODES = ("both", "orig_neg", "orig_pos")
SRC_MODES = ("folder", "auto")
MAX_IMAGES = 100               # 兩邊都改是 200 個工作；生成子行程的進度表只留最近 300 列（worker_client.TABLE_MAX）
AREA = 1024 * 1024             # 修圖的大小：約這麼多像素，長寬比照原圖
# 資料夾名稱固定用英文、不跟著介面語言（換了語言還要接得上做到一半的那批）；以前的中文名稱在的話沿用（_dir）
POS, NEG, BASE = "positive", "negative", "base"
PAIRS_SUFFIX = "_pairs"
LEGACY = {POS: "正向", NEG: "反向", BASE: "底圖", PAIRS_SUFFIX: "_成對"}
BASE_SIZE = 1024               # 自動畫的底圖（各架構的原生大小，修圖時照 AREA 縮放）
N_BASE = (4, 40, 12)           # 自動畫幾張：最少、最多、預設
SEC_PER_EDIT = 24              # 估時間用：Qwen 修圖一張（RTX 5080，2026-10-08 實測 24 張 577 秒，960×1280 的圖改成約 100 萬像素）
SEC_PER_BASE = {"sdxl": 6.5, "anima": 16.5}     # 畫一張 1024² 的底圖（同 server/train.py 的 SEC_PER_SAMPLE）
_LOCK = threading.Lock()
STATE = {"running": False, "phase": "", "total": 0, "done": 0, "failed": 0, "skipped": 0, "pos_dir": "", "neg_dir": "",
         "msg": "", "error": ""}


def state():
    with _LOCK:
        return dict(STATE)


def _dir(parent, name, suffix_of=None):
    """parent 底下的 name（suffix_of 給的話是 <suffix_of><name>）。同一層裡一律用同一套名稱，不會「正向」配「negative」：
    新名稱一個都還沒有、舊的中文名稱有任何一個在，就整層用舊的"""
    names = (PAIRS_SUFFIX,) if suffix_of else (POS, NEG, BASE)
    pre = suffix_of or ""
    legacy = (not any((parent / f"{pre}{n}").exists() for n in names)
              and any((parent / f"{pre}{LEGACY[n]}").is_dir() for n in names))
    return parent / f"{pre}{LEGACY[name] if legacy else name}"


def out_dirs(src, base=None):
    if not base:
        base = _dir(Path(src).parent, PAIRS_SUFFIX, suffix_of=Path(src).name)
    return _dir(Path(base), POS), _dir(Path(base), NEG)


def edit_size(w, h):
    """修圖的大小：面積約 AREA、長寬比照原圖、邊長對齊 32（Qwen Image 2.1 要 32 的倍數，core/qwen21.MULTIPLE；
    對齊 16 的話它會畫大一點再縮回來，多一次縮放。2026-10-08 實測 960×1280 的圖：880×1184 → 畫 896×1184 再縮）"""
    from .refine import snap
    k = (AREA / float(w * h)) ** 0.5
    return snap(w * k, 32), snap(h * k, 32)


def qwen_models():
    """能用的 Qwen Image 2.1 主模型"""
    from .resources import model_arch, res_scan
    return [{"path": it["path"], "name": it.get("name") or Path(it["path"]).stem}
            for it in res_scan("model") if model_arch(it["path"]) == "qwen21"]


def default_root():
    """自動畫的成對圖預設放哪：最近用的資料集的上一層（使用者放資料集的地方），沒有就是 CoCo 的 datasets 資料夾"""
    from . import dataset
    from .state import ROOT
    for d in dataset.recent():
        if Path(d).is_dir():
            return str(Path(d).parent)
    return str(Path(ROOT) / "datasets")


def _edit_opts(data):
    """修圖的做法、兩邊的指示、Qwen 模型 → {mode, sides, texts, model} 或 {error}"""
    mode = data.get("mode") if data.get("mode") in MODES else "both"
    texts = {POS: str(data.get("pos_text") or "").strip()[:2000], NEG: str(data.get("neg_text") or "").strip()[:2000]}
    sides = {"both": (POS, NEG), "orig_neg": (POS,), "orig_pos": (NEG,)}[mode]
    for side in sides:
        if not texts[side]:
            return {"error": tr("srv.pm.needText", side=tr("srv.pm.pos" if side == POS else "srv.pm.neg"))}
    model = str(data.get("model") or "")
    if model not in {m["path"] for m in qwen_models()}:
        return {"error": tr("srv.pm.needQwen")}
    return {"mode": mode, "sides": sides, "texts": texts, "model": model}


def plan(data, out_base=None):
    """要改哪些：回傳 {tasks: [(原圖, 哪邊, 指示, 存成)], pos_dir, neg_dir, skipped, model} 或 {error}。
    out_base：自動畫的底圖改好放哪（存放位置/名稱），只從程式裡給"""
    from . import dataset
    src = dataset.root_of(data.get("src"))
    if not src:
        return {"error": tr("srv.pm.needSrc")}
    imgs = dataset.list_images(src)
    if not imgs:
        return {"error": tr("srv.noImagesInDir")}
    if len(imgs) > MAX_IMAGES:
        return {"error": tr("srv.pm.tooMany", max=MAX_IMAGES, n=len(imgs))}
    # 成對圖靠檔名配對（server/train.pairs_of：不看副檔名、不分大小寫，子資料夾的也算），改好的圖也照檔名存：
    # 同名的兩張會存到同一個地方、後一張蓋掉前一張，.txt 卻留著前一張的
    seen, dup = {}, []
    for img in imgs:
        first = seen.setdefault(img.stem.lower(), img)
        if first is not img:
            dup.append(f"{first.relative_to(src).as_posix()}、{img.relative_to(src).as_posix()}")
    if dup:
        return {"error": tr("srv.pm.dupNames", names=tr("srv.pm.dupSep").join(dup[:3]) + (" …" if len(dup) > 3 else ""))}
    e = _edit_opts(data)
    if "error" in e:
        return e
    pos_dir, neg_dir = out_dirs(src, out_base)
    if e["mode"] == "orig_neg":
        neg_dir = src
    elif e["mode"] == "orig_pos":
        pos_dir = src
    dirs = {POS: pos_dir, NEG: neg_dir}
    tasks, skipped = [], 0
    for img in imgs:
        for side in e["sides"]:
            target = dirs[side] / f"{img.stem}.png"
            if target.exists():
                skipped += 1
                continue
            tasks.append((img, side, e["texts"][side], target))
    return {"tasks": tasks, "pos_dir": str(pos_dir), "neg_dir": str(neg_dir), "skipped": skipped, "model": e["model"]}


def auto_plan(data):
    """自動畫底圖要畫哪些：回傳 {tasks: [(第幾張, 提示詞, 存成)], base, base_dir, model, arch, n, skipped} 或 {error}"""
    from . import train
    from .resources import model_arch
    name = train.safe_name(data.get("name"))
    if not name:
        return {"error": tr("srv.pm.needName")}
    try:
        root = Path(str(data.get("root") or "")).expanduser()
    except (TypeError, ValueError):
        return {"error": tr("srv.needFullPathSave")}
    if not root.is_absolute():
        return {"error": tr("srv.needFullPathSave")}
    model = str(data.get("base_model") or "")
    arch = model_arch(model) if model and Path(model).is_file() else ""
    if not arch:
        return {"error": tr("srv.pm.needBase")}
    prompts = [ln.strip() for ln in str(data.get("prompts") or "").splitlines() if ln.strip()][:20]
    if not prompts:
        return {"error": tr("srv.pm.needPrompts")}
    try:
        n = int(data.get("n_base") or N_BASE[2])
    except (TypeError, ValueError):
        n = N_BASE[2]
    n = max(N_BASE[0], min(N_BASE[1], n))
    base = root / name
    bdir = _dir(base, BASE)
    tasks, skipped = [], 0
    for i in range(n):
        target = bdir / f"base_{i:02d}.png"
        if target.exists():
            skipped += 1
            continue
        tasks.append((i, prompts[i % len(prompts)], target))
    return {"tasks": tasks, "base": str(base), "base_dir": str(bdir), "model": model, "arch": arch, "n": n, "skipped": skipped}


def estimate(data):
    """還沒開始：要畫幾張、要改幾張、大約幾分鐘、存到哪"""
    if data.get("src_mode") == "auto":
        a = auto_plan(data)
        if "error" in a:
            return a
        e = _edit_opts(data)
        if "error" in e:
            return e
        pos_dir, neg_dir = out_dirs(None, a["base"])
        dirs = {POS: pos_dir, NEG: neg_dir}
        n_edit = sum(1 for i in range(a["n"]) for side in e["sides"] if not (dirs[side] / f"base_{i:02d}.png").exists())
        sec = len(a["tasks"]) * SEC_PER_BASE.get(a["arch"], 10) + n_edit * SEC_PER_EDIT
        return {"ok": True, "n_base": len(a["tasks"]), "n": n_edit, "skipped": a["skipped"], "minutes": round(sec / 60, 1),
                "pos_dir": str(pos_dir if e["mode"] != "orig_pos" else a["base_dir"]),
                "neg_dir": str(neg_dir if e["mode"] != "orig_neg" else a["base_dir"])}
    p = plan(data)
    if "error" in p:
        return p
    n = len(p["tasks"])
    return {"ok": True, "n_base": 0, "n": n, "skipped": p["skipped"], "minutes": round(n * SEC_PER_EDIT / 60, 1),
            "pos_dir": p["pos_dir"], "neg_dir": p["neg_dir"]}


def arch_params(arch):
    """這個架構的步數、CFG、取樣器、排程器：生成台現在用的模型是這個架構就用現在的，不然用記下的那組（settings_api 的
    ARCH_PARAMS），都沒有就是官方建議的"""
    from . import settings_api
    from .resources import model_arch
    from .state import SET
    cfg = SET.load()
    if model_arch(cfg.get("MODEL") or "") == arch:
        mem = {k: cfg.get(k) for k in SET.ARCH_PARAM_KEYS if cfg.get(k) is not None}
    else:
        mem = SET.as_archparams(cfg.get("ARCH_PARAMS")).get(arch) or {}
    p = {**settings_api.ARCH_DEFAULTS.get(arch, settings_api.ARCH_DEFAULTS["sdxl"]), **mem}
    return {"steps": p["STEPS"], "cfg": p["CFG"], "sampler": p["SAMPLER"], "schedule": p["SCHEDULE"]}


def qwen_params():
    """Qwen Image 2.1 的參數（修圖用）"""
    return arch_params("qwen21")


def _seed(data):
    try:
        return int(data.get("seed", 42))
    except (TypeError, ValueError):
        return 42


def start(data, plan_one, worker, busy=lambda: False):
    """plan_one(資料, 原圖路徑或 None) → handler.plan_run 的結果（{load, jobs, warns} 或 {error}）；worker：生成子行程"""
    with _LOCK:
        if STATE["running"]:
            return {"error": tr("srv.pm.busy")}
    if busy():
        return {"error": tr("srv.pm.training")}
    if data.get("src_mode") == "auto":
        a = auto_plan(data)
        if "error" in a:
            return a
        e = _edit_opts(data)                    # 修圖的設定先檢查：不要畫完底圖才說指示沒寫
        if "error" in e:
            return e
        return _start_bases(data, a, plan_one, worker)
    return _start_edits(data, plan_one, worker)


def _start_bases(data, a, plan_one, worker):
    """第一段：照提示詞畫底圖。畫完（_watch 的 then）接著改"""
    from . import train
    edits = {**data, "src_mode": "folder", "src": a["base_dir"]}

    def then():
        return _start_edits(edits, plan_one, worker, out_base=a["base"])
    if not a["tasks"]:
        return _start_edits(edits, plan_one, worker, out_base=a["base"])
    from . import presets
    sfw = bool(data.get("sfw", True))             # 照 Slider 左欄的「避開 NSFW」與「加品質標籤」（底圖是訓練的材料，要乾淨）
    neg = train.SLIDER_SAFE_NEG if sfw else ""
    tag = train.SLIDER_SAFE_POS.get(a["arch"], "") if sfw else ""
    stage = Path(tempfile.mkdtemp(prefix="coco-pairs-"))
    params = arch_params(a["arch"])
    jobs, load, todo = [], None, {}
    for k, (i, prompt, target) in enumerate(a["tasks"]):
        pr = f"{prompt.strip(' ,')}, {tag}" if tag and presets.missing(prompt, tag) else prompt
        got = plan_one({"model": a["model"], "prompts": [pr], "negative": neg, "batch": 1, "seed": _seed(data) * 1000 + i,
                        "fix_seed": True, "preset": bool(data.get("quality", True)), "loras": [], "auto_face": False,
                        "width": BASE_SIZE, "height": BASE_SIZE, **params}, None)
        if "error" in got:
            shutil.rmtree(stage, ignore_errors=True)
            return got
        jobs.append({**got["jobs"][0], "id": k, "out_dir": str(stage), "name_pattern": "", "kind": "pairs"})
        load = load or got["load"]
        todo[k] = (None, target, prompt)           # .txt 寫畫它的提示詞（不含品質標籤）：訓練時那一對的提示詞
    if not worker.submit(load, jobs):
        shutil.rmtree(stage, ignore_errors=True)
        return {"error": tr("srv.pm.genBusy")}
    batch = worker.batch
    with _LOCK:
        STATE.update(running=True, phase="base", total=len(jobs), done=0, failed=0, skipped=a["skipped"], pos_dir="", neg_dir="",
                     error="", msg="")
    threading.Thread(target=_watch, args=(worker, batch, todo, stage, {"pos_dir": "", "neg_dir": ""}),
                     kwargs={"then": then}, daemon=True).start()
    return {"ok": True, **state()}


def _start_edits(data, plan_one, worker, out_base=None):
    """修圖那一段（資料夾來源直接從這裡開始；自動畫的畫完底圖從 _watch 接過來）"""
    from PIL import Image
    p = plan(data, out_base)
    if "error" in p:
        return p
    if not p["tasks"]:
        _remember(p)
        with _LOCK:
            STATE.update(running=False, phase="", total=0, done=0, failed=0, skipped=p["skipped"], pos_dir=p["pos_dir"],
                         neg_dir=p["neg_dir"], error="", msg=tr("srv.pm.allDone"))
        return {"ok": True, **state()}
    stage = Path(tempfile.mkdtemp(prefix="coco-pairs-"))
    params = qwen_params()
    jobs, load, todo = [], None, {}
    for k, (img, side, text, target) in enumerate(p["tasks"]):
        try:
            with Image.open(img) as im:
                w, h = edit_size(*im.size)
        except OSError:
            continue                                    # 讀不了的圖跳過
        got = plan_one({"model": p["model"], "prompts": [text], "negative": "", "batch": 1, "seed": _seed(data), "fix_seed": True,
                        "preset": False, "loras": [], "auto_face": False, "width": w, "height": h, **params}, str(img))
        if "error" in got:
            shutil.rmtree(stage, ignore_errors=True)
            return got
        job = {**got["jobs"][0], "id": k, "out_dir": str(stage), "name_pattern": "", "kind": "pairs"}
        jobs.append(job)
        load = load or got["load"]
        todo[k] = (img, target)
    if not jobs:
        shutil.rmtree(stage, ignore_errors=True)
        return {"error": tr("srv.pm.allUnreadable")}
    if not worker.submit(load, jobs):
        shutil.rmtree(stage, ignore_errors=True)
        return {"error": tr("srv.pm.genBusy")}
    batch = worker.batch
    with _LOCK:
        STATE.update(running=True, phase="edit", total=len(jobs), done=0, failed=0, skipped=p["skipped"], pos_dir=p["pos_dir"],
                     neg_dir=p["neg_dir"], error="", msg="")
    threading.Thread(target=_watch, args=(worker, batch, todo, stage, p), daemon=True).start()
    return {"ok": True, **state()}


ENDED = ("failed", "aborted", "cancelled", "skipped")


def collect(worker, batch, todo):
    """看生成子行程的進度表：做好的搬到該去的地方 → (這次搬了幾張, 這次失敗幾張)。todo 處理過的拿掉。
    todo 的值：(原圖, 存成)＝改好的圖，原圖的 .txt 沒有才複製；(None, 存成, 提示詞)＝自動畫的底圖，.txt 寫提示詞"""
    from . import dataset
    with worker.lock:
        rows = [dict(r) for r in worker.table if r["b"] == batch and r["i"] in todo]
    moved = failed = 0
    for r in rows:
        if r["status"] == "done" and r.get("path"):
            img, target, *text = todo.pop(r["i"])
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():                 # 做的期間有同名的檔放進來了：不蓋掉
                    raise FileExistsError(str(target))
                shutil.move(r["path"], target)
                if text:
                    dataset.write_text(target.with_suffix(".txt"), text[0])
                else:
                    cap = dataset.caption_path(img)
                    if cap.is_file() and not target.with_suffix(".txt").exists():
                        shutil.copy2(cap, target.with_suffix(".txt"))
                moved += 1
            except OSError:
                failed += 1
        elif r["status"] in ENDED:
            todo.pop(r["i"])
            failed += 1
    return moved, failed


def _watch(worker, batch, todo, stage, p, every=1.0, then=None):
    """每秒收一次做好的；這一批做完，有 then（下一段）就接著開，不然記住資料夾、結束"""
    nxt = None
    try:
        while todo:
            moved, failed = collect(worker, batch, todo)
            with _LOCK:
                STATE["done"] += moved
                STATE["failed"] += failed
            if todo and not worker.state.get("running") and not worker.jobs and not worker.pending:
                time.sleep(every)                       # 最後一張剛做完：再看一次
                moved, failed = collect(worker, batch, todo)
                with _LOCK:
                    STATE["done"] += moved
                    STATE["failed"] += failed + len(todo)
                todo.clear()
                break
            time.sleep(every)
        if then and not state()["failed"]:
            nxt = then()
        elif then:
            nxt = {"error": tr("srv.pm.baseFailed", n=state()['failed'])}
        else:
            _remember(p)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if not (nxt and nxt.get("ok") and nxt.get("running")):       # 下一段開始了就交給它
            with _LOCK:
                STATE["running"] = False
                if nxt and nxt.get("error"):
                    STATE.update(error=nxt["error"], msg=nxt["error"])
                elif not (nxt and nxt.get("msg")):
                    STATE["msg"] = tr("srv.pm.done") if not STATE["failed"] else tr("srv.pm.doneFailed", n=STATE['failed'])


def _remember(p):
    """做好的資料夾加進資料集清單（03 訓練的正向、反向選得到）"""
    from . import dataset
    for d in (p["neg_dir"], p["pos_dir"]):
        if d and Path(d).is_dir():
            dataset._remember(Path(d))


def cancel(worker):
    if state()["running"]:
        worker.cancel()
    return {"ok": True}
