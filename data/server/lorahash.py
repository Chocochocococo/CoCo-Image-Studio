# -*- coding: utf-8 -*-
"""檔案的雜湊值：改過名的 LoRA 用雜湊值對上（送到生成區時）；CoCo 存圖記下 LoRA 與模型的雜湊（Forge 的寫法）。

Forge／A1111 存圖時寫 Lora hashes: "名字: 前 12 碼"，雜湊是哪一種跟檔案有關（實測兩種都有）：
  - 檔頭的 __metadata__ 有 sshs_model_hash（kohya sd-scripts 訓練的 LoRA 都有，只算權重、不含檔頭）就用它
    （有些 LoRA 是這種，改過檔名的也找得到）
  - 沒有的話是整個檔案的 SHA256（有些 LoRA 是這種）
整個檔案的 SHA256 從哪裡來，便宜的先：StabilityMatrix 放在旁邊的 <名字>.cm-info.json（Hashes.SHA256）→
算過的記在 lora_hashes.json（路徑＋大小＋修改時間對得上才用）→ 都沒有才讀整個檔案算（幾百 MB 約 1 秒），算完記起來。
找改過名的：先看不用讀檔就知道的；還找不到，只有「檔頭沒有 sshs、也還不知道整檔雜湊」的檔案要算，
名字像的先算，最多算 BUDGET 秒，剩下的在背景接著算（下次就快了）。
"""

import difflib
import hashlib
import json
import threading
import time
from pathlib import Path

from .state import C

BUDGET = 20.0                  # 找改過名的 LoRA 最多花幾秒算雜湊，剩下的背景接著算
CHUNK = 1 << 20
_LOCK = threading.Lock()
_BG = {"thread": None}


def cache_file():
    """算過的整檔雜湊（LoRA 與模型）。以前叫 lora_hashes.json，有的話改名接著用"""
    root = Path(getattr(C, "ROOT", Path(C.DB_PATH).parent))
    f = root / "file_hashes.json"
    old = root / "lora_hashes.json"
    if not f.exists() and old.exists():
        try:
            old.replace(f)
        except OSError:
            return old
    return f


def _load():
    try:
        d = json.loads(cache_file().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    f = cache_file()
    tmp = f.with_name(f.name + ".tmp")
    try:
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)
    except OSError:
        tmp.unlink(missing_ok=True)


def _sig(p):
    st = p.stat()
    return [st.st_size, round(st.st_mtime, 3)]


def header_hash(path):
    """檔頭 __metadata__ 的 sshs_model_hash（kohya sd-scripts 寫的）；沒有回 ''（只讀檔頭，很快）"""
    try:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if not 2 <= n <= 64 * 1024 * 1024:
                return ""
            meta = (json.loads(fh.read(n)) or {}).get("__metadata__") or {}
        h = str(meta.get("sshs_model_hash") or "").lower()
        return h if len(h) >= 12 else ""
    except (OSError, ValueError, AttributeError):
        return ""


def sidecar(path):
    """StabilityMatrix 的 .cm-info.json 裡記的 SHA256（小寫）；沒有回 ''"""
    p = Path(path)
    try:
        d = json.loads(p.with_name(p.stem + ".cm-info.json").read_text(encoding="utf-8"))
        h = str((d.get("Hashes") or {}).get("SHA256") or "").lower()
        return h if len(h) == 64 else ""
    except (OSError, ValueError, AttributeError):
        return ""


def known(path):
    """不用讀整個檔就知道的整檔 SHA256（說明檔或之前算過的）；不知道回 ''"""
    p = Path(path)
    h = sidecar(p)
    if h:
        return h
    try:
        e = _load().get(str(p))
        if e and e.get("sig") == _sig(p):
            return e["sha256"]
    except OSError:
        pass
    return ""


def compute(path):
    """讀整個檔案算 SHA256，記進 lora_hashes.json"""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with _LOCK:
        d = _load()
        d[str(p)] = {"sig": _sig(p), "sha256": sha}
        _save(d)
    return sha


def full(path, allow_compute=True):
    return known(path) or (compute(path) if allow_compute else "")


def short(path, allow_compute=True):
    """Forge 寫在圖裡的那種：檔頭有 sshs_model_hash 就用它，不然是整檔 SHA256；取前 12 碼。算不了回 ''"""
    try:
        return (header_hash(path) or full(path, allow_compute))[:12]
    except OSError:
        return ""


def resolve(wanted, files, budget=BUDGET):
    """wanted：{圖裡的名字: 雜湊前幾碼（10～12 碼都可以）}；files：[(名字, 路徑)…]（目前找得到的 LoRA）。
    回傳 {"found": {圖裡的名字: 現在的名字}, "partial": 還沒全部算完}"""
    want = {n: str(h).lower() for n, h in (wanted or {}).items() if len(str(h)) >= 8}
    found = {}

    def take(name, *hashes):
        for n, h in want.items():
            if n not in found and any(x and x.startswith(h) for x in hashes):
                found[n] = name

    pending = []
    for name, path in files:
        head = header_hash(path)
        try:
            sha = known(path)
        except OSError:
            continue
        take(name, head, sha)
        if not head and not sha:
            pending.append((name, path))          # Forge 對這種用整檔雜湊：要算才知道
    if len(found) == len(want):
        return {"found": found, "partial": False}
    # 還沒找到的：名字像的先算
    left = [n for n in want if n not in found]
    score = lambda nm: max(difflib.SequenceMatcher(None, nm.lower(), w.lower()).ratio() for w in left)
    pending.sort(key=lambda x: -score(x[0]))
    start = time.monotonic()
    for i, (name, path) in enumerate(pending):
        if len(found) == len(want):
            break
        if time.monotonic() - start >= budget:
            _background(pending[i:])
            return {"found": found, "partial": True}
        try:
            take(name, compute(path))
        except OSError:
            continue
    return {"found": found, "partial": False}


_PENDING = set()


def model_hash(path):
    """模型的雜湊（A1111／Forge 的 Model hash：整檔 SHA256 前 10 碼）。不用讀檔就知道的直接給；
    不知道的在背景算（模型幾 GB，不能讓生成等），這次回 ''，算好之後的圖就有了。資料夾格式的模型不算"""
    p = Path(str(path or ""))
    if not p.is_file():
        return ""
    try:
        sha = known(p)
    except OSError:
        return ""
    if sha:
        return sha[:10]
    with _LOCK:
        if str(p) in _PENDING:
            return ""
        _PENDING.add(str(p))

    def run():
        try:
            compute(p)
        except OSError:
            pass
        finally:
            with _LOCK:
                _PENDING.discard(str(p))
    threading.Thread(target=run, daemon=True).start()
    return ""


def _background(rest):
    """剩下的檔案在背景算完（下次找就快了）；已經在算就不再開"""
    t = _BG["thread"]
    if t is not None and t.is_alive():
        return

    def run():
        for _, path in rest:
            try:
                if not known(path):
                    compute(path)
            except OSError:
                pass
    _BG["thread"] = threading.Thread(target=run, daemon=True)
    _BG["thread"].start()
