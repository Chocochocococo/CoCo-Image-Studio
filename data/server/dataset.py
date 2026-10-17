# -*- coding: utf-8 -*-
"""訓練空間的打標台：資料集資料夾、打標籤、審核、整批改標籤、從圖庫建資料集、翻譯描述

從舊的獨立打標台（tagger_studio.py）搬進來的。資料集就是一個放圖的資料夾，每張圖旁邊一個同名的 .txt
（sd-scripts、OneTrainer 通用的格式），CoCo 不另外存一份標籤。

- **打標籤**：WD14（core/wdtagger.py，工作行程跑）回每張圖的標籤與機率，這裡套門檻與清理規則寫成 .txt。
  門檻、清理規則、角色名要不要留的預設值跟舊打標台一樣；觸發詞每個資料集各自記。
- **_tags_only**：舊打標台追加自然語言描述前，把純標籤版備份到資料集底下的 _tags_only/（檔名同 .txt），
  重跑描述時從那裡重新開始。這裡照樣認得：重新打標籤時備份一起更新（不然重跑描述會把舊的標籤還原回來），
  整批改標籤時備份也一起改。_tags_only 與 _deleted（舊打標台的刪除區）裡的圖不算資料集的圖。
- **寫描述**：JoyCaption（core/joycaption.py，工作行程跑，直接讀使用者的 GGUF）。描述接在標籤後面（設定的分隔），
  已經有描述的跳過（勾覆蓋或選取的照做）；標籤從 _tags_only 的備份或 .txt 的標籤那段拿，重寫描述不會疊上去。
  提示詞可以改，{length} 換成描述長度（短、中、長，也決定最多寫幾個字）；「把標籤當參考」時標籤（不含觸發詞）放在提示詞前面。
  重新打標籤時保留已經寫好的描述，只換標籤（舊打標台會一起蓋掉）。
- **刪除**丟進 Windows 的資源回收筒（圖、.txt、備份），不再搬到 _deleted。
- **整批改標籤**只比對整個標籤（逗號隔開的一段），大小寫、底線與空格、括號跳脫都不算差別；
  自然語言描述那一段不會被當成標籤改到。
- **翻譯**：標籤的中文用 CoCo 內建的翻譯（前端的標籤模式）；自然語言描述送 Google 翻譯——
  只有按了翻譯（或打開換圖自動翻）才送。用 Google 網頁翻譯的端點（translate-pa.googleapis.com 的 translateHtml，
  金鑰從 Google 翻譯自己的程式碼抓，抓不到用它公開的那把；做法同 CoCo-Translate）。舊打標台用的
  translate_a/single 常常回「超出請求額度」。

設定存在 datasets.json（不進 git）：清理規則與門檻（opts）、每個資料集的觸發詞（trigger）、資料集清單（library）。

**資料集清單**（訓練空間的 01 資料集，像 OneTrainer 的 concept 清單）：每個資料集記資料夾、封面（沒選就是第一張）、
每張重複幾次（訓練時帶入）、備註、訓練紀錄（訓練完自動加，server/train.py）。開過的資料夾自動加進來，最近用的排前面；
從清單移除只是不顯示，資料夾裡的檔案不動。以前的「最近開過」（recent）第一次讀的時候併進清單。
"""

import html
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import download, fileops

from i18n import tr
from .state import C, SET

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BACKUP_DIR = "_tags_only"
SKIP_DIRS = (BACKUP_DIR, "_deleted")
DATASETS_FILE = getattr(C, "DATASETS_FILE", None) or (Path(C.DB_PATH).parent / "datasets.json")
CHUNK = 8                       # 一次交給工作行程幾張（中間可以插生成、可以取消）
CAP_CHUNK = 2                   # 寫描述一次幾張（一張約 1 秒）
# 描述長度 → 提示詞裡的 {length}、最多寫幾個 token
LENGTHS = {"short": ("short", 120), "medium": ("medium-length", 250), "long": ("long", 400)}
DEFAULT_NL = ("Write a {length} descriptive caption for this image in a casual tone. Describe the appearance, clothing, pose, "
              "expression and background. Do not mention the artist, copyright, or any character name.")
_LOCK = threading.RLock()
# .txt 的讀改寫（審核頁存檔、背景打標籤與寫描述、整批改標籤）：同一張不會一邊讀一邊被另一邊寫掉
_TXT = threading.Lock()

# 清理規則與門檻的預設值：跟舊打標台一樣
OPTS = {
    "general": 0.35,            # 一般標籤門檻
    "character": 0.85,          # 角色標籤門檻
    "keep_character": False,    # 保留角色名標籤
    "remove_prefixes": "copyright:, character:, artist:",   # 整個刪掉的前綴
    "strip_labels": "meta:",    # 只刪前綴文字
    "underscore": True,         # 底線換成空格
    "escape": True,             # 括號加上反斜線
    "dedupe": True,             # 移除重複標籤
    "separator": ", ",          # 標籤與描述的分隔
    "overwrite": False,         # 已經有標籤的圖也重做
    "nl_prompt": DEFAULT_NL,    # 寫描述的提示詞（JoyCaption 官方建議的寫法）
    "nl_length": "medium",      # 描述長度：short／medium／long
    "use_tags": True,           # 把標籤當參考給模型看
}
_TYPES = {k: type(v) for k, v in OPTS.items()}


# ---------------------------------------------------------------- 設定

def _load():
    try:
        d = json.loads(DATASETS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    tmp = DATASETS_FILE.with_suffix(".json.tmp")
    tmp.write_bytes(json.dumps(d, ensure_ascii=False, indent=1).encode("utf-8"))
    tmp.replace(DATASETS_FILE)


def clean_opts(data):
    """只收認得的欄位、型別對的；門檻夾在 0～1"""
    out = {}
    for k, typ in _TYPES.items():
        v = (data or {}).get(k)
        if v is None:
            continue
        try:
            if typ is bool:
                out[k] = bool(v)
            elif typ is float:
                out[k] = min(1.0, max(0.0, float(v)))
            elif k == "nl_length":
                if str(v) in LENGTHS:
                    out[k] = str(v)
            else:
                out[k] = str(v)[:4000 if k == "nl_prompt" else 500]
        except (TypeError, ValueError):
            continue
    return out


def get_opts():
    with _LOCK:
        return {**OPTS, **clean_opts(_load().get("opts"))}


def set_opts(data):
    with _LOCK:
        d = _load()
        d["opts"] = {**clean_opts(d.get("opts")), **clean_opts(data)}
        _save(d)
        return {**OPTS, **d["opts"]}


def _key(root):
    return str(Path(root)).lower()


def get_trigger(root):
    with _LOCK:
        return str((_load().get("trigger") or {}).get(_key(root), ""))


def set_trigger(root, text):
    with _LOCK:
        d = _load()
        t = d.setdefault("trigger", {})
        text = str(text or "").strip().rstrip(",").strip()[:200]
        if text:
            t[_key(root)] = text
        else:
            t.pop(_key(root), None)
        _save(d)
        return text


MAX_RUNS = 30                   # 每個資料集記幾筆訓練紀錄
REPEATS_RANGE = (1, 100)


def _library(d):
    """datasets.json 的 library（[{path, cover, repeats, note, runs}]），沒有就從以前的 recent 建。會改到 d"""
    lib = d.get("library")
    if not isinstance(lib, list):
        lib = [{"path": r} for r in (d.get("recent") or []) if isinstance(r, str)]
        d["library"] = lib
    return [e for e in lib if isinstance(e, dict) and isinstance(e.get("path"), str)]


def _entry(lib, root):
    for e in lib:
        if _key(e["path"]) == _key(root):
            return e
    return None


def recent():
    """資料集的路徑，最近用的在前面（圖庫「加進資料集」、訓練頁的清單用）"""
    with _LOCK:
        return [e["path"] for e in _library(_load())]


def _remember(root):
    """開了這個資料夾：加進清單（已經在就搬到最前面，設定照舊）"""
    with _LOCK:
        d = _load()
        lib = _library(d)
        e = _entry(lib, root) or {"path": str(root)}
        d["library"] = [e] + [x for x in lib if x is not e]
        d.pop("recent", None)
        _save(d)


def library():
    """01 資料集的卡片：名字、張數、有標籤的張數、封面、觸發詞、每張重複幾次、備註、訓練紀錄；資料夾不見的標 missing"""
    with _LOCK:
        d = _load()
        lib = _library(d)
        trig = d.get("trigger") or {}
    out = []
    for e in lib:
        root = root_of(e["path"])
        card = {"path": e["path"], "name": Path(e["path"]).name, "repeats": e.get("repeats"), "note": e.get("note", ""),
                "runs": list(e.get("runs") or [])[-5:][::-1], "n_runs": len(e.get("runs") or []),
                "trigger": str(trig.get(_key(e["path"]), "")), "missing": root is None}
        if root:
            imgs = list_images(root)
            cover = inside(root, e["cover"]) if e.get("cover") else None
            cover = cover or (imgs[0] if imgs else None)
            card.update(n=len(imgs), tagged=sum(1 for i in imgs if caption_path(i).is_file()),
                        cover=str(cover) if cover else "", cover_v=int(cover.stat().st_mtime) if cover else 0)
        out.append(card)
    return out


def lib_add(path):
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    _remember(root)
    return {"ok": True, "dir": str(root)}


def lib_remove(path):
    """從清單拿掉（資料夾裡的檔案不動）"""
    with _LOCK:
        d = _load()
        lib = _library(d)
        d["library"] = [e for e in lib if _key(e["path"]) != _key(path)]
        _save(d)
    return {"ok": True}


def lib_update(path, data):
    """改封面（資料集裡的一張圖，空字串＝回到第一張）、每張重複幾次（空＝沒設定）、備註"""
    with _LOCK:
        d = _load()
        lib = _library(d)
        e = _entry(lib, path)
        if e is None:
            return {"error": tr("srv.ds.notListed")}
        if "cover" in data:
            if data["cover"]:
                img = inside(Path(e["path"]), data["cover"])
                if not img:
                    return {"error": tr("srv.ds.imgNotIn")}
                e["cover"] = str(img)
            else:
                e.pop("cover", None)
        if "repeats" in data:
            try:
                e["repeats"] = max(REPEATS_RANGE[0], min(REPEATS_RANGE[1], int(data["repeats"])))
            except (TypeError, ValueError):
                e.pop("repeats", None)
        if "note" in data:
            note = str(data["note"] or "").strip()[:2000]
            if note:
                e["note"] = note
            else:
                e.pop("note", None)
        d["library"] = lib
        _save(d)
        return {"ok": True, "cover": e.get("cover", ""), "repeats": e.get("repeats"), "note": e.get("note", "")}


def dataset_info(path):
    """打標頁的「資料集設定」：每張重複幾次、備註、封面、訓練紀錄（新的在前）"""
    with _LOCK:
        e = _entry(_library(_load()), path) or {}
    return {"repeats": e.get("repeats"), "note": e.get("note", ""), "cover": e.get("cover", ""),
            "runs": list(e.get("runs") or [])[::-1]}


def add_run(path, run):
    """訓練完：記進這個資料集的訓練紀錄（server/train.py 呼叫；不在清單裡就先加進去）"""
    with _LOCK:
        d = _load()
        lib = _library(d)
        e = _entry(lib, path)
        if e is None:
            e = {"path": str(path)}
            lib.insert(0, e)
        e["runs"] = (list(e.get("runs") or []) + [dict(run)])[-MAX_RUNS:]
        d["library"] = lib
        _save(d)


# ---------------------------------------------------------------- 資料夾與檔案

def root_of(path):
    """使用者給的資料集資料夾 → 完整路徑；不是資料夾回 None"""
    try:
        p = Path(str(path or "")).expanduser()
        if not p.is_absolute():
            return None
        p = p.resolve()
        return p if p.is_dir() else None
    except (OSError, ValueError):
        return None


def inside(root, path):
    """root 底下的圖片檔（不在 _tags_only、_deleted 裡）→ Path；其他回 None。API 收到的路徑都要過這關"""
    try:
        p = Path(str(path)).resolve()
        rel = p.relative_to(Path(root))
    except (OSError, ValueError):
        return None
    if any(part in SKIP_DIRS for part in rel.parts[:-1]):
        return None
    return p if p.suffix.lower() in IMAGE_EXTS and p.is_file() else None


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


def caption_path(img):
    return Path(img).with_suffix(".txt")


def backup_path(root, img):
    """舊打標台的純標籤備份：資料集底下 _tags_only/同名.txt（子資料夾的圖也放在同一層，照舊打標台）"""
    return Path(root) / BACKUP_DIR / caption_path(img).name


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def write_text(path, text):
    Path(path).write_bytes(str(text).strip().encode("utf-8"))


def items(root):
    """資料集的每張圖：名字、路徑、標籤、有沒有 .txt、縮圖用的修改時間"""
    root = Path(root)
    out = []
    for p in list_images(root):
        txt = caption_path(p)
        try:
            mtime = int(p.stat().st_mtime)
        except OSError:
            mtime = 0
        text = read_text(txt)
        tags, desc = split_caption(text)
        out.append({"name": p.name, "rel": str(p.relative_to(root)), "path": str(p), "text": text,
                    "tags": ", ".join(tags), "desc": desc, "has_txt": txt.is_file(), "mtime": mtime})
    return out


def open_dataset(path):
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    _remember(root)
    its = items(root)
    return {"dir": str(root), "items": its, "tagged": sum(1 for i in its if i["has_txt"]),
            "trigger": get_trigger(root), "has_backup": (root / BACKUP_DIR).is_dir()}


def save(root, path, text=None, tags=None, desc=""):
    """審核頁改了標籤：寫回 .txt。給 tags（與 desc）時用設定的分隔接起來；給 text 就照寫"""
    img = inside(root, path)
    if not img:
        return {"error": tr("srv.ds.imgNotIn")}
    if tags is not None:
        text = join_caption([x.strip() for x in str(tags).split(",") if x.strip()], str(desc or "").strip(),
                            get_opts()["separator"])
    with _TXT:
        write_text(caption_path(img), text or "")
    return {"ok": True, "text": read_text(caption_path(img))}


# ---------------------------------------------------------------- 清理規則（照舊打標台的 clean_tags）

def _list(s):
    return [p.strip().lower() for p in str(s or "").split(",") if p.strip()]


def clean_tags(text, opts, trigger=""):
    """標籤文字 → 清理後的「a, b, c」：整個刪掉某些前綴的、拿掉某些前綴文字、底線換空格、括號跳脫、
    去除重複（不分大小寫），觸發詞放最前面"""
    remove, strip = _list(opts.get("remove_prefixes")), _list(opts.get("strip_labels"))
    out, seen = [], set()
    for raw in str(text).replace("\n", ",").split(","):
        tag = raw.strip()
        low = tag.lower()
        if not tag or any(low.startswith(p) for p in remove):
            continue
        for p in strip:
            if low.startswith(p):
                tag = tag[len(p):].strip()
                break
        if opts.get("underscore"):
            tag = tag.replace("_", " ")
        if opts.get("escape"):
            tag = tag.replace("\\(", "(").replace("\\)", ")").replace("(", "\\(").replace(")", "\\)")
        tag = re.sub(r"\s+", " ", tag).strip()
        if not tag:
            continue
        if opts.get("dedupe"):
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())
        out.append(tag)
    trig = str(trigger or "").strip().rstrip(",").strip()
    if trig:
        out = [trig] + [t for t in out if t.lower() != trig.lower()]
    return ", ".join(out)


def pick(pred, opts):
    """WD14 的結果（[[標籤, 分類, 機率], …]，高的在前）→ 過了門檻的標籤，逗號隔開。
    一般標籤看一般門檻；角色標籤要打開「保留角色名」並過角色門檻"""
    out = []
    for name, cat, p in pred or []:
        if cat == 0 and p >= opts["general"]:
            out.append(name)
        elif cat == 4 and opts.get("keep_character") and p >= opts["character"]:
            out.append(name)
    return ", ".join(out)


# ---------------------------------------------------------------- 打標籤（背景工作）

JOB = {"running": False, "cancel": False, "done": 0, "total": 0, "dir": "", "kind": "", "log": [], "changed": []}


def _log(msg, kind="info"):
    JOB["log"].append({"t": time.strftime("%H:%M:%S"), "m": str(msg), "k": kind})
    del JOB["log"][:-300]


def job_state():
    with _LOCK:
        d = {k: v for k, v in JOB.items() if k != "cancel"}
        d["log"] = list(JOB["log"][-200:])
        d["changed"] = list(JOB["changed"])
        return d


def cancel():
    with _LOCK:
        if JOB["running"]:
            JOB["cancel"] = True
            _log(tr("srv.ds.cancelReq"))
    return {"ok": True}


def _begin(path, targets, kind):
    """打標籤、寫描述共用：(資料夾, 圖) 或 {error}。成功就佔住 JOB"""
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    if targets:
        imgs = [p for p in (inside(root, t) for t in targets) if p]
        if not imgs:
            return {"error": tr("srv.ds.selNotIn")}
    else:
        imgs = list_images(root)
        if not imgs:
            return {"error": tr("srv.noImagesInDir")}
    with _LOCK:
        if JOB["running"]:
            return {"error": tr("srv.ds.busy")}
        JOB.update(running=True, cancel=False, done=0, total=len(imgs), dir=str(root), kind=kind, log=[], changed=[])
    return root, imgs


def start_tag(path, targets, ask, model):
    """開始打標籤：targets 是選取的圖（空的就是整個資料夾）。ask(模型, 路徑清單) 是工作行程的 WD14"""
    if not model:
        return {"error": tr("srv.ds.noTagger"), "need_model": True}
    got = _begin(path, targets, "tag")
    if isinstance(got, dict):
        return got
    root, imgs = got
    opts = get_opts()
    # 選了圖就是要重做那幾張；整個資料夾時已經有標籤的跳過，除非勾了覆蓋
    force = bool(targets) or opts["overwrite"]
    threading.Thread(target=_run_tag, args=(root, imgs, opts, get_trigger(root), force, ask, model),
                     daemon=True).start()
    return {"ok": True, "total": len(imgs)}


def _run_tag(root, imgs, opts, trigger, force, ask, model):
    try:
        todo = imgs if force else [p for p in imgs if not caption_path(p).is_file()]
        skipped = len(imgs) - len(todo)
        if skipped:
            _log(tr("srv.ds.skipTagged", n=skipped))
        with _LOCK:
            JOB["done"] = skipped
        _log(tr("srv.ds.tagStart", n=len(todo)))
        ok = 0
        for s in range(0, len(todo), CHUNK):
            if JOB["cancel"]:
                _log(tr("srv.cancelled"), "err")
                return
            chunk = todo[s:s + CHUNK]
            got = ask(model, [str(p) for p in chunk])
            if not got.get("ready"):
                _log(tr("srv.ds.tagFail", why=got.get('why') or tr("srv.ds.noReply")), "err")
                return
            for img, pred in zip(chunk, got.get("tags") or []):
                if pred is None:
                    _log(tr("srv.ds.unreadable", name=img.name), "err")
                    continue
                text = clean_tags(pick(pred, opts), opts, trigger)
                bak = backup_path(root, img)
                with _TXT:
                    if bak.is_file():
                        write_text(bak, text)
                    # 已經寫好的描述留著，只換標籤
                    desc = split_caption(read_text(caption_path(img)))[1]
                    write_text(caption_path(img), join_caption([text] if text else [], desc, opts["separator"]))
                ok += 1
                with _LOCK:
                    JOB["changed"].append(str(img))
                _log(f"{img.name} → {text[:90]}", "ok")
            with _LOCK:
                JOB["done"] = skipped + s + len(chunk)
        _log(tr("srv.ds.tagDone", n=ok), "ok")
    except Exception as e:
        _log(tr("srv.ds.tagErr", e=f"{type(e).__name__}: {e}"), "err")
    finally:
        with _LOCK:
            JOB["running"] = False
            JOB["cancel"] = False


# ---------------------------------------------------------------- 寫描述（JoyCaption）

def nl_prompt(opts, tags, trigger=""):
    """寫描述的提示詞：{length} 換成長度；把標籤當參考時（不含觸發詞）放在前面（照舊打標台的寫法）"""
    text = (opts.get("nl_prompt") or DEFAULT_NL).strip()
    text = text.replace("{length}", LENGTHS.get(opts.get("nl_length"), LENGTHS["medium"])[0])
    trig = tag_key(trigger) if trigger else ""
    ref = [t for t in tags if tag_key(t) != trig]
    if opts.get("use_tags") and ref:
        text = f"The following tags describe this image: {', '.join(ref)}\n\n{text}"
    return text


def start_caption(path, targets, ask, files):
    """開始寫描述：files 是 JoyCaption 的 {main, mmproj}（沒設定就是 None）。ask(主體, mmproj, 路徑, 提示詞, 最多幾個 token)"""
    if not files or not files.get("main") or not files.get("mmproj"):
        return {"error": (files or {}).get("error") or tr("srv.ds.noJc"), "need_jc": True}
    got = _begin(path, targets, "desc")
    if isinstance(got, dict):
        return got
    root, imgs = got
    opts = get_opts()
    force = bool(targets) or opts["overwrite"]
    threading.Thread(target=_run_caption, args=(root, imgs, opts, get_trigger(root), force, ask, files),
                     daemon=True).start()
    return {"ok": True, "total": len(imgs)}


def _tags_of(root, img):
    """這張圖的純標籤：舊打標台的 _tags_only 備份優先，不然是 .txt 的標籤那段"""
    bak = backup_path(root, img)
    if bak.is_file():
        return split_caption(read_text(bak))[0]
    return split_caption(read_text(caption_path(img)))[0]


def _run_caption(root, imgs, opts, trigger, force, ask, files):
    try:
        todo = imgs if force else [p for p in imgs if not split_caption(read_text(caption_path(p)))[1]]
        skipped = len(imgs) - len(todo)
        if skipped:
            _log(tr("srv.ds.skipDescribed", n=skipped))
        with _LOCK:
            JOB["done"] = skipped
        _log(tr("srv.ds.descStart", n=len(todo)))
        max_new = LENGTHS.get(opts["nl_length"], LENGTHS["medium"])[1]
        ok, t0 = 0, time.time()
        for s in range(0, len(todo), CAP_CHUNK):
            if JOB["cancel"]:
                _log(tr("srv.cancelled"), "err")
                return
            chunk = todo[s:s + CAP_CHUNK]
            tags = [_tags_of(root, p) for p in chunk]
            was = [read_text(caption_path(p)) for p in chunk]       # 寫回前比對：等模型的這幾秒裡有沒有被改過
            got = ask(files["main"], files["mmproj"], [str(p) for p in chunk],
                      [nl_prompt(opts, t, trigger) for t in tags], max_new)
            if not got.get("ready"):
                _log(tr("srv.ds.descFail", why=got.get('why') or tr("srv.ds.noReply")), "err")
                return
            for img, t, before, text in zip(chunk, tags, was, got.get("texts") or []):
                if text is None:
                    _log(tr("srv.ds.unreadable", name=img.name), "err")
                    continue
                desc = " ".join(str(text).split())
                with _TXT:
                    now = read_text(caption_path(img))
                    if now != before:
                        # 等模型的時候這張被改過（審核頁存了、整批改了標籤）：標籤用現在的；描述也改過就整個留使用者的
                        cur_tags, cur_desc = split_caption(now)
                        if cur_desc != split_caption(before)[1]:
                            _log(tr("srv.ds.keptYours", name=img.name))
                            continue
                        t = cur_tags
                    write_text(caption_path(img), join_caption(t, desc, opts["separator"]))
                ok += 1
                with _LOCK:
                    JOB["changed"].append(str(img))
                _log(f"{img.name} → {desc[:90]}", "ok")
            with _LOCK:
                JOB["done"] = skipped + s + len(chunk)
        _log(tr("srv.ds.descDone", n=ok, s=f"{time.time() - t0:.0f}"), "ok")
    except Exception as e:
        _log(tr("srv.ds.descErr", e=f"{type(e).__name__}: {e}"), "err")
    finally:
        with _LOCK:
            JOB["running"] = False
            JOB["cancel"] = False


def joycaption_files():
    """設定的 JoyCaption 資料夾（JOYCAPTION_DIR）裡的 GGUF：檔名有 mmproj 的是看圖的部分，其他的是主體（好幾個時取最大的）。
    沒設定時用 CoCo 自己的 models\\joycaption（裡面有 GGUF 才算）"""
    from . import modelhome
    d = str(SET.load().get("JOYCAPTION_DIR") or "").strip()
    if not d:
        own = modelhome.own("joycaption")
        try:
            has = any(f.suffix.lower() == ".gguf" for f in own.iterdir())
        except OSError:
            has = False
        if not has:
            return {"dir": "", "main": "", "mmproj": ""}
        d = str(own)
    try:
        files = [f for f in Path(d).iterdir() if f.suffix.lower() == ".gguf" and f.is_file()]
    except OSError:
        return {"dir": d, "main": "", "mmproj": "", "error": tr("srv.ds.jcNoDir")}
    proj = sorted((f for f in files if "mmproj" in f.name.lower()), key=lambda f: -f.stat().st_size)
    main = sorted((f for f in files if "mmproj" not in f.name.lower()), key=lambda f: -f.stat().st_size)
    out = {"dir": d, "main": str(main[0]) if main else "", "mmproj": str(proj[0]) if proj else ""}
    if not main or not proj:
        out["error"] = tr("srv.ds.jcFiles")
    return out


# ---------------------------------------------------------------- 標籤統計與整批修改

def tag_key(tag):
    """比對用：大小寫、底線與空格、括號跳脫、多餘空白都不算差別"""
    t = str(tag).replace("\\(", "(").replace("\\)", ")").replace("_", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def is_sentence(seg):
    """自然語言描述的一段（不是標籤）：七個字以上，或有句號、問號、驚嘆號接空白或結尾"""
    s = seg.strip()
    return len(s.split()) >= 7 or bool(re.search(r"[.!?](\s|$)", s))


def split_caption(text):
    """標籤檔 → (標籤清單, 描述)。描述是第一段像句子的開始到最後"""
    segs = [s.strip() for s in str(text).split(",")]
    for i, s in enumerate(segs):
        if s and is_sentence(s):
            return [t for t in segs[:i] if t], ", ".join(segs[i:]).strip(", ")
    return [t for t in segs if t], ""


def join_caption(tags, desc, sep=", "):
    t = ", ".join(tags)
    if desc:
        return f"{t}{sep}{desc}" if t else desc
    return t


def stats(path):
    """整個資料集的標籤：每個標籤出現在幾張圖（同一個標籤不同寫法算一個，顯示最常見的寫法）"""
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    count, forms = {}, {}
    imgs = list_images(root)
    for p in imgs:
        tags, _ = split_caption(read_text(caption_path(p)))
        seen = set()
        for t in tags:
            k = tag_key(t)
            if k in seen:
                continue
            seen.add(k)
            count[k] = count.get(k, 0) + 1
            f = forms.setdefault(k, {})
            f[t] = f.get(t, 0) + 1
    trig = tag_key(get_trigger(root)) if get_trigger(root) else ""
    rows = [{"tag": max(forms[k].items(), key=lambda kv: kv[1])[0], "count": n, "trigger": k == trig}
            for k, n in count.items()]
    rows.sort(key=lambda r: (-r["count"], r["tag"].lower()))
    return {"images": len(imgs), "tags": rows}


def _edit(tags, op, key, to, front=False):
    """一張圖的標籤清單套用整批修改，回傳新的清單（沒變回 None）。front：加的標籤放最前面（觸發詞）"""
    if op == "remove":
        new = [t for t in tags if tag_key(t) != key]
    elif op == "replace":
        new, seen = [], set()
        for t in tags:
            t = to if tag_key(t) == key else t
            if tag_key(t) not in seen:
                seen.add(tag_key(t))
                new.append(t)
    elif op == "add":
        if any(tag_key(t) == key for t in tags):
            return None
        new = [to] + list(tags) if front else list(tags) + [to]
    else:
        return None
    return new if new != list(tags) else None


def bulk(path, op, tag, to="", targets=None):
    """整批改標籤：remove（拿掉）、replace（換成 to）、add（加上 tag，已經有的不重複）。
    只動標籤那段，描述不動；_tags_only 的備份一起改。targets 有給就只改那幾張"""
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    if op not in ("remove", "replace", "add"):
        return {"error": tr("srv.ds.badEdit")}
    key = tag_key(tag)
    if not key:
        return {"error": tr("srv.ds.noTag")}
    to = re.sub(r"\s+", " ", str(to or "").replace(",", " ")).strip()
    if op == "replace" and not to:
        return {"error": tr("srv.ds.replaceWith")}
    if op == "add":
        to = re.sub(r"\s+", " ", str(tag).replace(",", " ")).strip()
    imgs = [p for p in (inside(root, t) for t in targets) if p] if targets else list_images(root)
    trig = get_trigger(root)
    front = op == "add" and bool(trig) and tag_key(trig) == key        # 加的是觸發詞：放最前面，跟打標籤時一樣
    changed = 0
    for img in imgs:
        hit = False
        for f in (caption_path(img), backup_path(root, img)):
            if not f.is_file() and not (op == "add" and f == caption_path(img)):
                continue
            with _TXT:
                tags, desc = split_caption(read_text(f))
                new = _edit(tags, op, key, to, front)
                if new is None:
                    continue
                write_text(f, join_caption(new, desc))
            hit = hit or f == caption_path(img)
        changed += hit
    return {"ok": True, "changed": changed}


# ---------------------------------------------------------------- 刪除、從圖庫建資料集

def trash(path, targets):
    """選取的圖連同 .txt 與 _tags_only 的備份丟進資源回收筒"""
    root = root_of(path)
    if not root:
        return {"error": tr("srv.noDir")}
    imgs = [p for p in (inside(root, t) for t in targets or []) if p]
    files = []
    for img in imgs:
        files += [img, caption_path(img), backup_path(root, img)]
    try:
        fileops.recycle(files)
    except OSError as e:
        return {"error": str(e)}
    return {"ok": True, "moved": len(imgs)}


def from_gallery(ids, path, con):
    """圖庫選的圖複製進資料集資料夾（原圖不動；重名加 _1；同名的 .txt 不帶，標籤之後打）。資料夾不在就建"""
    try:
        root = Path(str(path or "")).expanduser()
        if not root.is_absolute():
            return {"error": tr("srv.needFullPath")}
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
    except OSError as e:
        return {"error": tr("srv.mkdirFail", e=e)}
    pairs = []
    for i in ids or []:
        r = con.execute("SELECT path FROM images WHERE id = ?", (int(i),)).fetchone()
        if r and Path(r[0]).is_file() and Path(r[0]).suffix.lower() in IMAGE_EXTS:
            pairs.append(Path(r[0]))
    copied = 0
    for src in pairs:
        # 一張一張複製：同一批裡兩張同名時，後面那張挑名字時前一張已經在了，自然會加 _1
        fileops.transfer([(src, fileops.free_name(root, src.name, extra=(".txt",)))], mode="copy")
        copied += 1
    _remember(root)
    return {"ok": True, "copied": copied, "dir": str(root)}


# ---------------------------------------------------------------- 打標籤的模型（WD14）

# 可以下載的（按了才下載）：SmilingWolf/wd-vit-large-tagger-v3（Apache-2.0）。跟舊打標台用的 model.onnx 逐位元相同
# （2026-10-06 核對 SHA256；selected_tags.csv 核對 git 的雜湊）。網址鎖定版本
TAGGER = {
    "name": "wd-vit-large-tagger-v3",
    "page": "https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3",
    "license": "Apache-2.0",
    "files": [
        {"file": "model.onnx", "bytes": 1260645673,
         "sha256": "e4c8001b000a6c98f2db10794f7c406daa79873d071d6ca924330fa053fa1845",
         "url": "https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/resolve/ae469aa2e4706a3af08d3673cf73a11d1add314c/model.onnx"},
        {"file": "selected_tags.csv", "bytes": 308468,
         "sha256": "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
         "url": "https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/resolve/ae469aa2e4706a3af08d3673cf73a11d1add314c/selected_tags.csv"},
    ],
}
_DL = download.Jobs()


def home():
    from . import modelhome
    return modelhome.own("tagger") / TAGGER["name"]


def is_model_dir(d):
    """資料夾裡有沒有 WD tagger（model.onnx 或 model.safetensors，加上 selected_tags.csv）"""
    try:
        d = Path(d)
        return (d / "selected_tags.csv").is_file() and any((d / n).is_file() for n in ("model.onnx", "model.safetensors"))
    except (OSError, ValueError):
        return False


def model_dir():
    """要用的 WD tagger 資料夾：設定裡指定的（TAGGER_DIR）優先，再來是 CoCo 下載的；都沒有回 None"""
    own = str(SET.load().get("TAGGER_DIR") or "").strip()
    for d in ([own] if own else []) + [str(home())]:
        if d and is_model_dir(d):
            return d
    return None


def model_status():
    d = model_dir()
    files = [{"dest": home() / f["file"], **f} for f in TAGGER["files"]]
    return {"dir": d or "", "own": str(SET.load().get("TAGGER_DIR") or ""), "home": str(home()), "jc": joycaption_files(),
            "name": TAGGER["name"], "page": TAGGER["page"], "license": TAGGER["license"],
            "bytes": sum(f["bytes"] for f in TAGGER["files"]), "part": download.on_disk(files),
            "download": _DL.progress("wd")}


def _download():
    files = [{**f, "dest": home() / f["file"]} for f in TAGGER["files"]]
    download.run(_DL["wd"], files, hint=tr("srv.dl.manualHint"))


def start_download():
    if is_model_dir(home()):
        return {"ok": True, "already": True}
    return _DL.begin("wd", sum(f["bytes"] for f in TAGGER["files"]), _download)


def cancel_download():
    return _DL.cancel("wd")


# ---------------------------------------------------------------- 翻譯描述（Google）

GOOGLE_URL = "https://translate-pa.googleapis.com/v1/translateHtml"
# Google 翻譯網頁小工具的程式碼，裡面寫著它自己用的金鑰（版本會換，抓不到就用下面那把）
GOOGLE_JS = ("https://translate.googleapis.com/_/translate_http/_/js/k=translate_http.tr.en_US.YusFYy3P_ro.O/am=AAg/"
             "d=1/exm=el_conf/ed=1/rs=AN8SPfq1Hb8iJRleQqQc8zhdzXmF9E56eQ/m=el_main")
GOOGLE_KEY = "AIzaSyATBXajvzQLTDHEQbcpq0Ihe0vWDHmO520"     # 同一個小工具公開的金鑰
_GKEY = {"key": None, "at": 0.0}
MAX_ITEMS, MAX_CHARS = 100, 5000


def _google_key(opener=urllib.request.urlopen):
    """抓到的金鑰 20 分鐘更新一次，抓不到 5 分鐘再試；這之間用公開的那把"""
    age = time.time() - _GKEY["at"]
    if age < (1200 if _GKEY["key"] else 300):
        return _GKEY["key"] or GOOGLE_KEY
    _GKEY["at"] = time.time()
    try:
        with opener(GOOGLE_JS, timeout=15) as r:
            m = re.search(r"""['"]x-goog-api-key['"]\s*:\s*['"](\w{39})['"]""", r.read().decode("utf-8", "replace"), re.I)
        _GKEY["key"] = m.group(1) if m else None
    except Exception:
        _GKEY["key"] = None
    return _GKEY["key"] or GOOGLE_KEY


def translate(segments, target="zh-TW", opener=urllib.request.urlopen):
    """一串文字 → 同樣數量的翻譯。translateHtml 把輸入當 HTML：先跳脫、回來再還原"""
    segs = [str(s) for s in segments if str(s).strip()]
    if not segs:
        return {"lines": []}
    if target not in ("zh-TW", "zh-CN", "ja", "en", "ko"):
        target = "zh-TW"
    out = []
    batch, size = [], 0
    for s in segs + [None]:
        if s is None or len(batch) >= MAX_ITEMS or (batch and size + len(s) > MAX_CHARS):
            if batch:
                out += _google(batch, target, opener)
            batch, size = [], 0
        if s is not None:
            batch.append(s)
            size += len(s)
    return {"lines": out}


def _google(texts, target, opener):
    body = json.dumps([[[html.escape(t, quote=False) for t in texts], "auto", target], "te"]).encode("utf-8")
    req = urllib.request.Request(GOOGLE_URL, data=body, method="POST", headers={
        "Content-Type": "application/json+protobuf", "X-goog-api-key": _google_key(opener)})
    try:
        with opener(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError(tr("srv.ds.gtRate")) from None
        raise RuntimeError(tr("srv.ds.gtErr", code=e.code)) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError(tr("srv.ds.gtOffline")) from None
    if not isinstance(data, list) or not isinstance(data[0], list) or len(data[0]) != len(texts):
        raise RuntimeError(tr("srv.ds.gtBad"))
    return [html.unescape(str(t or "")) for t in data[0]]
