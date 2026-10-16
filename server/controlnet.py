# -*- coding: utf-8 -*-
"""ControlNet：模型的清單（認架構、能不能用）、可以下載的模型、單張參考圖的設定整理。

用的地方：生成台的「參考圖」小視窗（照姿勢、線稿、深度、輪廓、原圖構圖來畫），以及精修的擴圖與局部重繪
（union 模型的「重繪」讓接縫更自然）。實際跑模型、抽姿勢與深度在工作行程（core/controlnet_prep.py、
core/pipeline.py）。

不限定哪些檔案：模型總資料夾裡分到 ControlNet 的資料夾（StabilityMatrix 的 ControlNet）與 models/controlnet，
裡面的 .safetensors 讀檔頭判斷：
  sdxl        一般的 SDXL ControlNet（diffusers 格式），一個模型管一種（從檔名猜是哪一種，猜不到讓使用者選）
  sdxl-union  xinsir 的 union／promax：一個模型管好幾種（姿勢、深度、軟邊緣、線稿、Tile、重繪）
  sd15        SD 1.5 用的：跟 SDXL 的結構對不上，列出來但標不能用
只收 .safetensors（.pth 這類載入時可能執行檔案裡夾帶的程式）。

可以下載的（按了才下載，下載完照 Hugging Face 公布的 SHA256 檢查；版本固定；中斷了下次接著下載，download.py）：
  union-promax  xinsir/controlnet-union-sdxl-1.0 的 promax（Apache-2.0）
  dwpose        yzd-v/DWPose 的 YOLOX＋DWPose ONNX（Apache-2.0）：從一般的圖抽出姿勢
  depth         depth-anything/Depth-Anything-V2-Small-hf（Apache-2.0）：從一般的圖算深度
"""

import hashlib
import json
import re

from i18n import tr
from pathlib import Path

from . import download
from .resources import res_dirs
from .state import C

UNSAFE = (".pth", ".pt", ".ckpt", ".bin", ".pkl", ".pickle")

# 控制的種類 → union promax 的 control_mode 編號（xinsir 的定義）
TYPES = ("pose", "depth", "softedge", "canny", "lineart", "tile", "repaint")
UNION_MODE = {"pose": 0, "depth": 1, "softedge": 2, "canny": 3, "lineart": 3, "tile": 6, "repaint": 7}
# 前處理：參考圖要先變成什麼（none＝已經是姿勢圖／線稿／深度圖，直接用）
PREPS = {"pose": ("dwpose", "none"), "depth": ("depth", "none"), "softedge": ("softedge", "none"),
         "canny": ("canny", "none"), "lineart": ("lineart", "none"), "tile": ("tile", "none")}
# 前處理要用的模型（下載項目）
PREP_NEEDS = {"dwpose": "dwpose", "depth": "depth"}

_REV = {"union": "801a4a3fa3d4c936f4feea95b98607bc6726f80c", "dwpose": "1a7144101628d69ee7a3768d1ee3a094070dc388",
        "depth": "5426e4f0f36572d16453bbda7a8389317b1bef99"}
CATALOG = {
    "union-promax": {
        "dir": "union-promax", "page": "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0", "license": "Apache-2.0",
        "files": [
            {"name": "config_promax.json", "url": f"https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/{_REV['union']}/config_promax.json",
             "bytes": 1260, "sha256": "6653ad6a0ed181f0d4a7225f1b1c037405d98573fbf0862b1ca3ab6c99b52c21"},
            {"name": "controlnet-union-sdxl-promax.safetensors",
             "url": f"https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/{_REV['union']}/diffusion_pytorch_model_promax.safetensors",
             "bytes": 2513342408, "sha256": "9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc"},
        ]},
    "dwpose": {
        "dir": "dwpose", "page": "https://huggingface.co/yzd-v/DWPose", "license": "Apache-2.0",
        "files": [
            {"name": "yolox_l.onnx", "url": f"https://huggingface.co/yzd-v/DWPose/resolve/{_REV['dwpose']}/yolox_l.onnx",
             "bytes": 216746733, "sha256": "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"},
            {"name": "dw-ll_ucoco_384.onnx", "url": f"https://huggingface.co/yzd-v/DWPose/resolve/{_REV['dwpose']}/dw-ll_ucoco_384.onnx",
             "bytes": 134399116, "sha256": "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"},
        ]},
    "depth": {
        "dir": "depth-anything-v2-small", "page": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf", "license": "Apache-2.0",
        "files": [
            {"name": "config.json", "url": f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/{_REV['depth']}/config.json",
             "bytes": 950, "sha256": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75"},
            {"name": "preprocessor_config.json",
             "url": f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/{_REV['depth']}/preprocessor_config.json",
             "bytes": 775, "sha256": "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84"},
            {"name": "model.safetensors", "url": f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/{_REV['depth']}/model.safetensors",
             "bytes": 99173660, "sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"},
        ]},
}

_DL = download.Jobs()          # 下載項目 → 進度
_PROBE = {}                    # (路徑, 大小, 修改時間) → 判斷結果


def home():
    from . import modelhome
    return modelhome.own("controlnet")


def dirs():
    """要找 ControlNet 的資料夾：CoCo 自己的（下載的放這裡）＋模型總資料夾裡分到 ControlNet 的"""
    out, seen = [], set()
    for d in [home(), *res_dirs("controlnet")]:
        k = str(Path(d)).lower()
        if k not in seen:
            seen.add(k)
            out.append(Path(d))
    return out


# ── 認架構 ──
def read_header(path):
    """safetensors 的檔頭：{權重名字: 形狀}。只讀開頭那段 JSON，不載入權重；不是 safetensors 回 None"""
    try:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if not 2 <= n <= 64 * 1024 * 1024:
                return None
            head = json.loads(fh.read(n))
    except (OSError, ValueError):
        return None
    if not isinstance(head, dict):
        return None
    return {k: v.get("shape") for k, v in head.items() if k != "__metadata__" and isinstance(v, dict)}


def detect(shapes):
    """檔頭 → {arch: sdxl / sdxl-union / sd15}；不是 ControlNet 或還不支援的回 {why}"""
    if not shapes:
        return {"why": "notSafetensors"}
    g = shapes.get
    if g("task_embedding") is not None or any(k.startswith("transformer_layes.") for k in shapes):
        if g("add_embedding.linear_1.weight") == [1280, 2816]:
            return {"arch": "sdxl-union"}
        return {"why": "unknown"}
    if g("controlnet_cond_embedding.conv_in.weight") is not None:
        if g("add_embedding.linear_1.weight") == [1280, 2816]:
            return {"arch": "sdxl"}
        ctx = next((v for k, v in shapes.items() if k.endswith("attn2.to_k.weight")), None)
        return {"arch": "sd15"} if ctx and ctx[-1] == 768 else {"why": "unknown"}
    if g("control_model.input_hint_block.0.weight") is not None:
        ctx = next((v for k, v in shapes.items() if k.endswith("attn2.to_k.weight")), None)
        if ctx and ctx[-1] == 768:
            return {"arch": "sd15"}
        return {"why": "ldmFormat"}                 # SDXL 的舊格式：還不支援
    if any("lora" in k for k in list(shapes)[:50]):
        return {"why": "controlLora"}
    return {"why": "notControlnet"}


# 從檔名猜一般 ControlNet 管哪一種（猜不到 ""，讓使用者選）
_GUESS = (("pose", r"pose"), ("depth", r"depth|zoe|midas"), ("canny", r"canny"), ("lineart", r"line|anime_?line|manga"),
          ("softedge", r"soft|hed|pidi|scribble|sketch"), ("tile", r"tile|blur"), ("repaint", r"inpaint|repaint"))


def guess_type(name):
    n = name.lower()
    return next((t for t, rx in _GUESS if re.search(rx, n)), "")


def sidecar(f):
    """StabilityMatrix 下載時放在旁邊的 .cm-info.json（Civitai 的模型名稱與版本名稱）：顯示名稱與猜種類用"""
    try:
        d = json.loads(f.with_name(f.stem + ".cm-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    name, ver = str(d.get("ModelName") or "").strip(), str(d.get("VersionName") or "").strip()
    return (f"{name} · {ver}" if name and ver else name or ver), str(d.get("BaseModel") or "")


def _probe(f):
    try:
        st = f.stat()
    except OSError:
        return {"why": "unreadable"}
    key = (str(f), st.st_size, st.st_mtime)
    if key not in _PROBE:
        _PROBE[key] = detect(read_header(f))
    return _PROBE[key]


def scan():
    """[{id, name, arch, type, path, bytes, own, ok}]（ok＝SDXL 能用）、認不得的 [{name, dir, why}]。
    同名的只算第一個（CoCo 的資料夾優先）；資料夾往下找一層（下載的 union 放在自己的資料夾裡）"""
    models, skipped, seen = [], [], set()
    for d in dirs():
        try:
            files = sorted([x for x in d.iterdir() if x.is_file()] + [y for x in d.iterdir() if x.is_dir() for y in x.iterdir() if y.is_file()])
        except OSError:
            continue
        for f in files:
            ext = f.suffix.lower()
            if ext in UNSAFE:
                skipped.append({"name": f.name, "dir": str(f.parent), "why": "unsafe"})
                continue
            if ext != ".safetensors" or f.name.lower() in seen:
                continue
            info = _probe(f)
            if not info.get("arch"):
                if info.get("why") != "notControlnet":     # IP-Adapter、CLIP Vision 這類放在一起的，不用列
                    skipped.append({"name": f.name, "dir": str(f.parent), "why": info.get("why", "unknown")})
                continue
            seen.add(f.name.lower())
            arch = info["arch"]
            label, base = sidecar(f)
            models.append({"id": f.name, "name": f.stem, "label": label, "base": base, "arch": arch, "path": str(f),
                           "bytes": f.stat().st_size, "own": home() in f.parents, "ok": arch in ("sdxl", "sdxl-union"),
                           "type": "" if arch == "sdxl-union" else (guess_type(f.stem) or guess_type(label)),
                           "config": str(f.parent / "config_promax.json") if (f.parent / "config_promax.json").is_file() else ""})
    models.sort(key=lambda m: (not m["ok"], m["arch"] != "sdxl-union", m["name"].lower()))
    return models, skipped


def find_model(mid):
    return next((m for m in scan()[0] if m["id"] == mid or m["path"] == mid), None)


def union_model():
    """能用的 union 模型（擴圖、局部重繪的「接縫更自然」用）；沒有回 None"""
    return next((m for m in scan()[0] if m["arch"] == "sdxl-union"), None)


# ── 可以下載的 ──
def package_dir(key):
    return home() / CATALOG[key]["dir"]


def installed(key):
    """下載項目的檔案都在、大小對"""
    d = package_dir(key)
    for f in CATALOG[key]["files"]:
        p = d / f["name"]
        try:
            if p.stat().st_size != f["bytes"]:
                return False
        except OSError:
            return False
    return True


def prep_paths(key):
    """前處理模型的檔案位置（沒裝回 None）"""
    return {f["name"]: str(package_dir(key) / f["name"]) for f in CATALOG[key]["files"]} if installed(key) else None


def status():
    models, skipped = scan()
    catalog = []
    for key, m in CATALOG.items():
        ok = installed(key)
        catalog.append({"key": key, "installed": ok, "bytes": sum(f["bytes"] for f in m["files"]),
                        "part": 0 if ok else download.on_disk(_files(key)),
                        "page": m["page"], "license": m["license"], "dir": str(package_dir(key)),
                        "files": [{"name": f["name"], "url": f["url"]} for f in m["files"]], "download": _DL.progress(key)})
    return {"catalog": catalog, "models": models, "skipped": skipped, "dir": str(home()),
            "dirs": [str(d) for d in dirs()], "types": list(TYPES), "preps": PREPS}


def _files(key):
    d = package_dir(key)
    return [{**f, "dest": d / f["name"]} for f in CATALOG[key]["files"]]


def _download(key, opener=None):
    download.run(_DL[key], _files(key), opener, hint=tr("srv.dl.manualHint"))


def start_download(key):
    if key not in CATALOG:
        return {"error": tr("srv.unknownModel")}
    if installed(key):
        return {"ok": True, "already": True}
    return _DL.begin(key, sum(f["bytes"] for f in CATALOG[key]["files"]), _download, (key,))


def cancel_download(key):
    return _DL.cancel(key)


def remove(key):
    """刪掉 CoCo 下載的項目（models/controlnet 底下，連下載到一半的 .part）；模型總資料夾裡的是使用者自己的，不動"""
    if key not in CATALOG:
        return {"error": tr("srv.unknownModel")}
    d = package_dir(key)
    try:
        for f in CATALOG[key]["files"]:
            (d / f["name"]).unlink(missing_ok=True)
            download.part_of(d / f["name"]).unlink(missing_ok=True)
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    except OSError as e:
        return {"error": str(e)}
    return {"ok": True}


# ── 一張參考圖的設定（生成台送來的）→ 工作行程要的 ──
def clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def unit_for_job(u, image_path):
    """生成台的一格參考圖 → 工作的 controlnets 一項；不能用回 {error}。
    u：{model（id）, type, prep, scale, start, end, low, high（canny 的門檻）}"""
    m = find_model(str(u.get("model") or ""))
    if not m:
        return {"error": tr("srv.cn.notFound")}
    if not m["ok"]:
        return {"error": tr("srv.cn.sd15", name=m['name'])}
    typ = str(u.get("type") or m["type"] or "")
    if typ not in TYPES or typ == "repaint":
        return {"error": tr("srv.cn.needType", name=m['name'])}
    prep = str(u.get("prep") or PREPS[typ][0])
    if prep not in PREPS[typ]:
        prep = PREPS[typ][0]
    need = PREP_NEEDS.get(prep)
    paths = prep_paths(need) if need else {}
    if need and not paths:
        return {"error": tr({"dwpose": "srv.cn.needDwpose", "depth": "srv.cn.needDepth"}[need])}
    if m["arch"] == "sdxl-union" and not m["config"]:
        return {"error": tr("srv.cn.noConfig", name=m['name'])}
    return {"model": m["path"], "arch": m["arch"], "config": m["config"], "name": m["name"], "type": typ, "prep": prep,
            "prep_files": paths or {}, "image": str(image_path),
            "scale": clamp(u.get("scale"), 0, 2, 1.0), "start": clamp(u.get("start"), 0, 1, 0.0), "end": clamp(u.get("end"), 0, 1, 1.0),
            "low": int(clamp(u.get("low"), 1, 254, 100)), "high": int(clamp(u.get("high"), 2, 255, 200))}


# ── 送到生成區：圖裡記的 ControlNet → 生成台的參考圖設定 ──
def restore_units(recorded, find_lib, find_ref):
    """recorded：圖的 coco JSON 裡的 controlnet 清單（model 是檔名、image 是參考圖的完整路徑）。
    find_lib(路徑) → 圖庫 id 或 None；find_ref(路徑) → 探索參考圖的 id 或 None。
    回傳 {units: 生成台的格式, missing: [(種類, 名字)]}。擴圖、局部重繪的「重繪」不帶（那是精修的）。
    參考圖不在圖庫、探索、cnrefs 裡（例如從別的資料夾拖進來又搬走了）但檔案還在的，複製一份進 cnrefs"""
    import base64
    units, missing = [], []
    models = scan()[0]
    for r in recorded if isinstance(recorded, list) else []:
        if not isinstance(r, dict) or r.get("type") == "repaint" or r.get("type") not in TYPES:
            continue
        fname = str(r.get("model") or "")
        m = next((x for x in models if x["id"].lower() == fname.lower()), None) or             next((x for x in models if x["name"].lower() == Path(fname).stem.lower()), None)
        if not m or not m["ok"]:
            missing.append(("model", r.get("name") or Path(fname).stem))
        img = Path(str(r.get("image") or ""))
        src = None
        if str(img) and img.is_file():
            if img.parent == UPLOAD_DIR and _NAME.match(img.name):
                src = {"kind": "upload", "name": img.name}
            elif (lid := find_lib(str(img))) is not None:
                src = {"kind": "lib", "id": lid}
            elif (rid := find_ref(str(img))) is not None:
                src = {"kind": "ref", "rid": rid}
            else:
                try:
                    up = save_upload("data:image/png;base64," + base64.b64encode(img.read_bytes()).decode())
                except OSError:
                    up = {"error": tr("srv.unreadable")}
                if "name" in up:
                    src = {"kind": "upload", "name": up["name"]}
        if src is None:
            missing.append(("image", img.name or "?"))
            continue
        units.append({"src": src, "model": m["id"] if m and m["ok"] else "", "type": r["type"],
                      "prep": str(r.get("prep") or PREPS[r["type"]][0]), "scale": clamp(r.get("scale"), 0, 2, 1.0),
                      "start": clamp(r.get("start"), 0, 1, 0.0), "end": clamp(r.get("end"), 0, 1, 1.0),
                      "resize": "fit" if r.get("resize") == "fit" else "crop", "low": 100, "high": 200, "off": False})
    return {"units": units[:3], "missing": missing}


# ── 從電腦選的參考圖：存一份在 cnrefs/（檔名是內容的雜湊，同一張只存一份）──
UPLOAD_DIR = Path(C.DB_PATH).parent / "cnrefs"
UPLOAD_MAX = 64 * 1024 * 1024
UPLOAD_SIDE = 4096
_NAME = re.compile(r"^[0-9a-f]{16}\.png$")


def save_upload(data_url):
    """data URL（PNG／JPG／WebP）→ 轉成 PNG 存起來，回傳 {name, width, height} 或 {error}"""
    import base64
    import io
    from PIL import Image
    head, _, body = str(data_url or "").partition(",")
    if not re.match(r"^data:image/(png|jpeg|webp);base64$", head) or not body:
        return {"error": tr("srv.cn.badType")}
    try:
        raw = base64.b64decode(body, validate=True)
        if len(raw) > UPLOAD_MAX:
            return {"error": tr("srv.imgTooBig")}
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
    except Exception:
        return {"error": tr("srv.imgUnreadable2")}
    if max(im.size) > UPLOAD_SIDE:
        k = UPLOAD_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{hashlib.sha1(buf.getvalue()).hexdigest()[:16]}.png"
    path = UPLOAD_DIR / name
    if not path.exists():
        path.write_bytes(buf.getvalue())
    return {"name": name, "width": im.width, "height": im.height}


def upload_path(name):
    """上傳過的參考圖的路徑；名字不對或檔案不在回 None（不讓人拿到 cnrefs 以外的檔案）"""
    name = str(name or "")
    if not _NAME.match(name):
        return None
    p = UPLOAD_DIR / name
    return p if p.is_file() else None
