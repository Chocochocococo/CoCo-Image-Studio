# -*- coding: utf-8 -*-
"""CoCo 自己的模型資料夾：models\\ 底下一種一格，名稱跟 ComfyUI 一樣（別的工具的模型可以整個資料夾搬過來，
模型總資料夾也認得這些名字）。不管使用者有沒有設模型總資料夾、各類的資料夾清單是不是空的，這裡一定會讀
（server/resources.py 的 res_dirs）。下載的、訓練的東西沒有別的地方可放時也放這裡。

以前的預設是 models\\ 整個當主模型、根目錄的 loras\\、vae\\、text_encoders\\、embeddings\\，清單被清空就讀不到，
放大與 JoyCaption 也沒有預設的地方（2026-10-06 使用者要求：每種用途分門別類，所有使用者都適用）。
啟動時 `migrate` 把舊位置的檔案搬進分格：不覆蓋同名的、不碰連結（junction 的真正檔案可能在別的磁碟）、
預覽圖與說明（同檔名開頭的）跟著搬、狀態檔裡指到舊路徑的跟著改（改之前備份到 backups\\），搬了什麼寫進 models\\moved.txt。
"""

import json
import os
import shutil
import time
from pathlib import Path

from .state import C, HERE

from i18n import tr

# 種類 → 子資料夾（第一個是放新東西的地方）
SUBDIRS = {
    "model": ("checkpoints", "diffusion_models"),   # SDXL、SD1.5 單檔 ／ 拆分式（Anima、Z-Image、Krea 2、Qwen）
    "lora": ("loras",),
    "vae": ("vae",),
    "te": ("text_encoders",),
    "embedding": ("embeddings",),
    "controlnet": ("controlnet",),
    "upscaler": ("upscale_models",),
    "detector": ("adetailer",),                       # 修臉的臉部偵測
    "rembg": ("rembg",),                              # 去背
    "explore": ("explore",),                          # 找相似（SigLIP2）
    "tagger": ("tagger",),                            # WD14 打標籤
    "joycaption": ("joycaption",),                    # JoyCaption 寫描述（GGUF）
}
# 不是生圖資源、是工具自己用的（主模型的掃描要跳過，見 resources.tool_dirs）
TOOL_KINDS = ("controlnet", "upscaler", "detector", "rembg", "explore", "tagger", "joycaption")
# 舊版的預設資料夾（根目錄底下）→ 搬到哪一格
LEGACY = {"loras": "loras", "vae": "vae", "text_encoders": "text_encoders", "embeddings": "embeddings"}
MODEL_EXT = {".safetensors", ".ckpt", ".pt", ".sft", ".gguf"}
SPLIT_ARCHS = ("anima", "zimage", "krea2", "qwen21", "klein4", "klein9", "flux1", "flux1s", "kontext")
# 存了模型路徑的狀態檔（搬了檔案要跟著改）
STATE_FILES = ("settings.json", "recipes.json", "datasets.json", "queue.json", "presets.json", "gallery_ui.json")
LOG_NAME = "moved.txt"


def root():
    return Path(getattr(C, "ROOT", HERE))


def home():
    return root() / "models"


def own_dirs(kind):
    """這一類 CoCo 自己的資料夾（不存在的也列，呼叫的地方自己看）"""
    return [home() / n for n in SUBDIRS.get(kind, ())]


def own(kind):
    """這一類放新東西的地方"""
    return home() / SUBDIRS[kind][0]


def ensure():
    """建好空的分格：使用者一打開 models 就知道東西放哪"""
    made = []
    for names in SUBDIRS.values():
        for n in names:
            d = home() / n
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                made.append(d)
    return made


def is_link(p):
    """symlink 或 Windows 的 junction（is_symlink 認不得 junction，看 reparse point 屬性）"""
    try:
        return p.is_symlink() or bool(getattr(os.lstat(p), "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def _key(p):
    return os.path.normcase(os.path.abspath(str(p)))


# ---------------------------------------------------------------- 搬檔案

def _sidecars(f):
    """跟模型同檔名開頭的（foo.preview.png、foo.civitai.info、foo.json、foo.png…）"""
    stem = f.name[:-len(f.suffix)] if f.suffix else f.name
    try:
        return [x for x in f.parent.iterdir() if x.is_file() and x != f and x.name.startswith(stem + ".")]
    except OSError:
        return []


def _move(src, dst, moves, skipped):
    """src 搬到 dst：同名的不覆蓋（資料夾就合併進去）、連結不碰"""
    if is_link(src):
        skipped.append((src, tr("srv.mh.link")))
        return
    if dst.exists():
        if src.is_dir() and dst.is_dir() and not is_link(dst):
            for x in sorted(src.iterdir()):
                _move(x, dst / x.name, moves, skipped)
            try:
                src.rmdir()                    # 只拿掉空的資料夾
            except OSError:
                pass
            return
        skipped.append((src, tr("srv.mh.same")))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    moves.append((src, dst))


def _arch(path):
    try:
        from .resources import model_arch
        return model_arch(str(path))
    except Exception:
        return ""


def migrate(log=print):
    """舊位置 → 分格。回傳 (moves, skipped)：[(舊, 新)]、[(路徑, 原因)]"""
    moves, skipped = [], []
    base, h = root(), home()
    for old, new in LEGACY.items():
        d = base / old
        if not d.is_dir():
            continue
        if is_link(d):
            skipped.append((d, tr("srv.mh.link")))
            continue
        for x in sorted(d.iterdir()):
            _move(x, h / new / x.name, moves, skipped)
        try:
            d.rmdir()
        except OSError:
            pass
    # models\ 最上層的主模型（以前 MODEL_DIRS 預設是整個 models）：拆分式放 diffusion_models，其他 checkpoints
    if h.is_dir() and not is_link(h):
        for f in sorted(h.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_EXT:
                sub = "diffusion_models" if _arch(f) in SPLIT_ARCHS else "checkpoints"
                side = _sidecars(f)
                _move(f, h / sub / f.name, moves, skipped)
                if moves and moves[-1][0] == f:
                    for s in side:
                        _move(s, h / sub / s.name, moves, skipped)
    if moves:
        rewrite_state(moves, log)
    _record(moves, skipped)
    for src, dst in moves:
        log("  " + tr("srv.mh.moved", src=src, dst=dst))
    for src, why in skipped:
        log("  [!] " + tr("srv.mh.skipped", why=why, src=src))
    return moves, skipped


def _record(moves, skipped):
    if not moves and not skipped:
        return
    lines = [time.strftime("%Y-%m-%d %H:%M:%S")]
    lines += [tr("srv.mh.recMoved", a=a, b=b) for a, b in moves]
    lines += [tr("srv.mh.skipped", why=why, src=a) for a, why in skipped]
    try:
        with open(home() / LOG_NAME, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n\n")
    except OSError:
        pass


# ---------------------------------------------------------------- 狀態檔的路徑跟著改

def remap(value, moves):
    """一個路徑字串：等於搬走的東西、或在搬走的資料夾裡 → 新的位置；其他照舊"""
    if not isinstance(value, str) or not value:
        return value
    k = _key(value)
    for src, dst in moves:
        s = _key(src)
        if k == s:
            return str(dst)
        if k.startswith(s.rstrip(os.sep) + os.sep):
            return str(dst) + value[len(str(src).rstrip("\\/")):]
    # 舊的預設資料夾本身（清單裡寫的 C:\…\loras）：整個資料夾搬空了，指到新的那格
    for old, new in LEGACY.items():
        if k == _key(root() / old):
            return str(home() / new)
    return value


def _walk(o, moves):
    if isinstance(o, dict):
        return {remap(k, moves): _walk(v, moves) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(x, moves) for x in o]
    return remap(o, moves)


def rewrite_state(moves, log=print):
    """狀態檔裡指到舊路徑的跟著改；有改的先備份到 backups\\（檔名加時間）"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    changed = []
    for name in STATE_FILES:
        f = root() / name
        if not f.is_file():
            continue
        try:
            raw = f.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            continue
        new = _walk(data, moves)
        if new == data:
            continue
        bak = root() / "backups" / f"{name}.{stamp}"
        bak.parent.mkdir(exist_ok=True)
        bak.write_bytes(raw.encode("utf-8"))
        f.write_bytes(json.dumps(new, ensure_ascii=False, indent=2).encode("utf-8"))
        changed.append(name)
        log("  " + tr("srv.mh.rewrote", name=name, bak=bak.name))
    return changed


def summary():
    """設定畫面用：CoCo 的模型資料夾與每一格（種類、名稱、幾個檔案）"""
    out = []
    for kind, names in SUBDIRS.items():
        for n in names:
            d = home() / n
            files = 0
            if d.is_dir():
                try:
                    files = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in MODEL_EXT | {".pth", ".bin", ".onnx"})
                except OSError:
                    pass
            out.append({"kind": kind, "name": n, "path": str(d), "files": files})
    return {"home": str(home()), "folders": out}
