# -*- coding: utf-8 -*-
"""模型、LoRA、VAE、文字編碼器、Embeddings 的掃描與架構判斷，以及資料夾瀏覽。

資源從三個地方來：各類自己的資料夾清單（MODEL_DIRS…）、模型總資料夾（MODELS_ROOT）底下依名稱自動分類的子資料夾，
以及 CoCo 自己的模型資料夾（server/modelhome.py：models 底下一種一格，一定會讀）。單一檔案可以改分類（RES_KIND）。
"""

import json
import os
from pathlib import Path

from .state import HERE, SET

from i18n import tr


# ---------------------------------------------------------------- 資源
RES_EXT = {".safetensors", ".ckpt", ".pt", ".sft"}
KINDS = ("model", "lora", "vae", "te", "embedding")
DIR_KEY = {"model": "MODEL_DIRS", "lora": "LORA_DIRS", "vae": "VAE_DIRS", "te": "TE_DIRS",
           "embedding": "EMBEDDING_DIRS", "detector": "DETECTOR_DIRS"}

# 模型總資料夾底下的子資料夾名稱 → 種類（StabilityMatrix、ComfyUI、A1111／Forge 的慣例，不分大小寫）。
# LyCORIS 跟 LoRA 一樣用 <lora:…>，併進 LoRA。
# upscaler、detector、controlnet 是之後的功能（放大、修臉…）才用得到的，先認得，資源庫還不列
FOLDER_KINDS = {
    "model": ("stablediffusion", "stable-diffusion", "checkpoints", "diffusionmodels", "diffusion_models", "unet", "diffusers"),
    "lora": ("lora", "loras", "lycoris"),
    "vae": ("vae",),
    "te": ("textencoders", "text_encoders", "clip"),
    "embedding": ("embeddings", "embedding", "textualinversion", "textual_inversion"),
    "upscaler": ("esrgan", "realesrgan", "upscale_models", "upscalers", "swinir", "scunet", "bsrgan", "ldsr",
                 "latent_upscale_models"),
    "detector": ("afterdetailer", "adetailer", "ultralytics", "sams"),
    "controlnet": ("controlnet", "t2iadapter", "ipadapter", "ipadapters15", "ipadaptersxl", "clipvision", "clip_vision"),
}
FOLDER_KIND = {name: kind for kind, names in FOLDER_KINDS.items() for name in names}
ROOT_KINDS = (*KINDS, "upscaler", "detector", "controlnet", "skip")
MODEL_FILE_EXT = RES_EXT | {".pth", ".bin", ".onnx", ".gguf"}


def _real(p):
    """比對用的路徑：連結解開、大小寫統一"""
    try:
        return os.path.normcase(str(Path(p).resolve()))
    except OSError:
        return os.path.normcase(os.path.abspath(str(p)))


def root_folders(root=None, mapping=None, count=False):
    """模型總資料夾底下的子資料夾：[{name, path, auto, kind, dup, files}]

    kind 是自動判斷的（auto），使用者在 MODELS_ROOT_MAP 改過的以改的為準；認不得的是 ""。
    指到同一個地方的（StabilityMatrix 的 diffusion_models 是 DiffusionModels 的連結）只算一次：
    真的資料夾優先，連結標 dup。count 才數檔案（設定畫面用；資源清單不需要，每次都數太慢）"""
    s = SET.load() if root is None or mapping is None else {}
    root = Path(str(s.get("MODELS_ROOT") or "") if root is None else str(root))
    over = {str(k).lower(): v for k, v in ((s.get("MODELS_ROOT_MAP") or {}) if mapping is None else mapping).items()}
    if not str(root).strip() or str(root) == "." or not root.is_dir():
        return []
    try:
        subs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return []

    def is_link(d):
        # Windows 的 junction（StabilityMatrix 用的）is_symlink() 認不得，Python 3.11 也還沒有
        # is_junction()：看檔案屬性的 reparse point（0x400）
        try:
            return d.is_symlink() or bool(getattr(os.lstat(d), "st_file_attributes", 0) & 0x400)
        except OSError:
            return False
    subs.sort(key=lambda d: (is_link(d), d.name.lower()))
    out, seen = [], set()
    for d in subs:
        real = _real(d)
        auto = FOLDER_KIND.get(d.name.lower(), "")
        it = {"name": d.name, "path": str(d), "auto": auto, "kind": over.get(d.name.lower(), auto),
              "dup": real in seen}
        seen.add(real)
        if count:
            n = 0
            try:
                for f in d.rglob("*"):
                    if f.suffix.lower() in MODEL_FILE_EXT:
                        n += 1
            except OSError:
                pass
            it["files"] = n
        out.append(it)
    out.sort(key=lambda it: it["name"].lower())
    return out
COVER_EXT = (".png", ".jpg", ".jpeg", ".webp")
TRIGGER_FILES = (".civitai.info", ".cm-info.json", ".json")


def res_dirs(kind):
    """這一類要掃的資料夾：自己的清單＋模型總資料夾裡分到這一類的＋CoCo 自己的（放最後：存檔、下載先找使用者的）；
    同一個地方只算一次"""
    from . import modelhome
    # 沒有自己清單的種類（放大、ControlNet）只看模型總資料夾，不要拿到 LoRA 的清單
    dirs = [Path(d) for d in (SET.load().get(DIR_KEY[kind]) or [])] if kind in DIR_KEY else []
    dirs += [Path(f["path"]) for f in root_folders() if f["kind"] == kind and not f["dup"]]
    dirs += modelhome.own_dirs(kind)
    out, seen = [], set()
    for d in dirs:
        k = _real(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def detectors():
    """修臉用的臉部偵測模型：偵測類資料夾裡檔名有 face 的 .pt（ADetailer 的 face_yolov8n/s），大的排前面"""
    out = []
    for it in res_scan("detector"):
        if "face" in it["name"].lower() and it["path"].lower().endswith(".pt"):
            out.append({"name": Path(it["path"]).name, "path": it["path"]})
    # yolov8s 比 n 準，m 更準：照 x > l > m > s > n 排
    rank = {c: i for i, c in enumerate("xlmsn")}
    out.sort(key=lambda d: (rank.get(Path(d["name"]).stem.lower()[-1:], 9), d["name"].lower()))
    return out


def kind_overrides():
    """單一檔案改過的分類：{比對用的路徑: (原本寫的路徑, 種類)}。
    比對用的路徑大小寫統一過，不能拿來顯示"""
    return {_real(p): (Path(p), k) for p, k in (SET.load().get("RES_KIND") or {}).items() if k in KINDS}


def _find_cover(f: Path):
    for suf in ("", ".preview"):
        for ext in COVER_EXT:
            c = f.with_suffix("")
            c = c.parent / (c.name + suf + ext)
            if c.is_file():
                return c
    return None


def _find_triggers(f: Path):
    """從 civitai / StabilityMatrix 的中繼資料撈觸發詞"""
    stem = f.with_suffix("")
    for suf in TRIGGER_FILES:
        p = stem.parent / (stem.name + suf)
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("trainedWords", "TrainedWords", "trigger_words"):
            v = d.get(key)
            if isinstance(v, list) and v:
                return [str(x) for x in v]
        mo = d.get("ModelDescription") or {}
        if isinstance(mo, dict):
            v = mo.get("TrainedWords")
            if isinstance(v, list) and v:
                return [str(x) for x in v]
    t = stem.parent / (stem.name + ".txt")
    if t.is_file():
        try:
            line = t.read_text(encoding="utf-8").strip().splitlines()
            if line:
                return [line[0]]
        except Exception:
            pass
    return []

def resolve_lora(name):
    """名字 → 檔案路徑。比對不分大小寫，也接受不含副檔名的寫法。"""
    want = name.strip().replace("\\", "/").split("/")[-1].lower()
    want_stem = want.rsplit(".", 1)[0]
    for it in res_scan("lora"):
        p = Path(it["path"])
        if p.stem.lower() == want_stem or p.name.lower() == want:
            return str(p)
    # 退而求其次：路徑裡任一段對得上（讓 子資料夾/名字 也能寫）
    for it in res_scan("lora"):
        if want_stem in Path(it["path"]).stem.lower():
            return it["path"]
    return None


def _file_item(f, root):
    try:
        size = f.stat().st_size
    except Exception:
        size = 0
    return {
        "name": f.stem,
        "path": str(f),
        "folder": str(f.parent),
        "rel": str(f.parent.relative_to(root)) if root and f.parent != root and root in f.parents else "",
        "root": str(root or f.parent),
        "size": round(size / 2 ** 30, 2),
        "cover": bool(_find_cover(f)),
        "kindTag": "單檔",
        "triggers": _find_triggers(f),
    }


def tool_dirs():
    """CoCo 自己的工具模型資料夾（ControlNet、放大、修臉偵測、去背、探索、打標籤、JoyCaption）。
    以前的預設 MODEL_DIRS 是整個 models（設定裡可能還留著），這些不是主模型，不能被列成模型"""
    from . import modelhome                      # 它會 import 這裡，放函式裡免得循環
    from .state import C
    return [*(d for k in modelhome.TOOL_KINDS for d in modelhome.own_dirs(k)),
            *(Path(d) for d in (getattr(C, "DETECTOR_DIRS", None) or []))]


def _tool_skips(root):
    """root 底下（不含 root 本身）的工具模型資料夾，用 root 的寫法回傳，好跟 rglob 出來的路徑比。
    root 就是工具資料夾或在它裡面（例如修臉偵測直接掃 models/adetailer）就不跳"""
    r = _real(root).rstrip(os.sep) + os.sep
    return [root / t[len(r):] for t in map(_real, tool_dirs()) if t.startswith(r)]


def res_scan(kind):
    out, seen = [], set()
    over = kind_overrides()
    for root in res_dirs(kind):
        if not root.is_dir():
            continue
        skips = _tool_skips(root)

        def skipped(f):
            return any(s in f.parents for s in skips)

        dif_dirs = []
        # diffusers 資料夾：有 model_index.json 就整個資料夾算一個模型
        if kind == "model":
            for idx in sorted(root.rglob("model_index.json")):
                d = idx.parent
                if _real(d) in seen or skipped(idx):
                    continue
                seen.add(_real(d))
                dif_dirs.append(d)
                size = sum(x.stat().st_size for x in d.rglob("*")
                           if x.is_file()) if d.is_dir() else 0
                out.append({
                    "name": d.name, "path": str(d), "folder": str(d.parent),
                    "rel": str(d.parent.relative_to(root))
                           if d.parent != root else "",
                    "root": str(root),
                    "size": round(size / 2 ** 30, 2),
                    "cover": bool(_find_cover(d / d.name)),
                    "kindTag": "diffusers",
                    "triggers": [],
                })

        for f in sorted(root.rglob("*")):
            # 同一個檔從兩個資料夾都看得到（舊設定的 MODEL_DIRS 是整個 models、CoCo 自己的 models\diffusion_models 也在；
            # 連結、大小寫不同）只列一次：先掃到的（使用者的資料夾）為準
            if f.suffix.lower() not in RES_EXT or _real(f) in seen or skipped(f):
                continue
            # diffusers 資料夾底下的零件不另外列（可能在子資料夾裡）
            if any(d == f.parent or d in f.parents for d in dif_dirs):
                continue
            seen.add(_real(f))
            if over and over.get(_real(f), (f, kind))[1] != kind:
                continue                    # 使用者把它改到別類了
            out.append(_file_item(f, root))
    # 從別類改過來的
    mine = [(real, f) for real, (f, k) in over.items() if k == kind]
    have = {_real(it["path"]) for it in out} if mine else set()
    for real, f in mine:
        if real not in have and f.is_file():
            out.append(_file_item(f, None))
    return out


def res_cover_path(kind, raw):
    """只允許讀取設定的資源資料夾底下的檔案（加上使用者改到這一類的單檔）"""
    try:
        p = Path(raw).resolve()
    except Exception:
        return None
    for root in res_dirs(kind):
        try:
            p.relative_to(root.resolve())
        except Exception:
            continue
        return _find_cover(p)
    if kind_overrides().get(_real(p), (p, ""))[1] == kind:
        return _find_cover(p)
    return None


def set_kind(path, kind):
    """單一檔案改分類（kind 空＝用回資料夾的分類）。只收目前清單裡有的檔案"""
    if kind and kind not in KINDS:
        return {"error": tr("srv.res.noKind")}
    want = _real(path)
    if not any(_real(it["path"]) == want for k in KINDS for it in res_scan(k)):
        return {"error": tr("srv.res.notListed")}
    cur = {p: k for p, k in (SET.load().get("RES_KIND") or {}).items() if _real(p) != want}
    if kind:
        cur[str(Path(path))] = kind
    SET.save({"RES_KIND": cur})
    return {"ok": True}


# ---------------------------------------------------------------- 資料夾
def browse(path):
    """列出資料夾。path 留空時回傳磁碟機與常用位置。"""
    import os
    import string

    if not path:
        roots = []
        if os.name == "nt":
            for d in string.ascii_uppercase:
                r = f"{d}:\\"
                if Path(r).exists():
                    roots.append({"name": f"{d}:", "path": r})
            for label, sub in (("桌面", "Desktop"), ("文件", "Documents"),
                               ("圖片", "Pictures"), ("下載", "Downloads")):
                pp = Path.home() / sub
                if pp.is_dir():
                    roots.append({"name": label, "path": str(pp)})
        else:
            roots = [{"name": "/", "path": "/"},
                     {"name": "家目錄", "path": str(Path.home())}]
        roots.append({"name": "專案資料夾", "path": str(HERE)})
        return {"path": "", "parent": None, "dirs": roots}

    p = Path(path)
    if not p.is_dir():
        return {"error": tr("srv.res.notDir", path=path)}
    dirs = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if c.is_dir() and not c.name.startswith("$"):
                    dirs.append({"name": c.name, "path": str(c)})
            except OSError:
                pass
    except PermissionError:
        return {"error": tr("srv.res.noPerm")}
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs}

_ARCH_CACHE = {}


def model_arch(path):
    """判斷模型架構。只讀 safetensors 標頭，約 40 ms，結果快取起來。"""
    if not path:
        return ""
    key = str(path)
    if key in _ARCH_CACHE:
        return _ARCH_CACHE[key]
    arch = ""
    try:
        p = Path(key)
        if p.suffix.lower() in (".safetensors", ".ckpt") and p.is_file():
            from core.pipeline import detect_arch
            arch = detect_arch(p)
    except Exception:
        arch = ""
    _ARCH_CACHE[key] = arch
    return arch
