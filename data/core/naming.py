# -*- coding: utf-8 -*-
"""生成的圖要存成什麼檔名（⚙ 設定 → 生成 → 存檔名稱）。

格式像 "{date}/{seq}-{seed}-{width}x{height}"（預設，就是以前固定的 outputs/日期/000001-seed-寬x高.png）：
/ 前面是資料夾（在輸出資料夾底下），最後一段是檔名（副檔名自己加 .png）。
變數可以加 :數字——{seq:4} 是補零到 4 位，其他的是最多幾個字（{prompt:30}）。

序號：同一個資料夾裡、符合這個格式的檔案取最大的序號再加一（以前是「當天資料夾最大的六位數加一」，
預設格式下結果一樣）。格式裡沒有序號又剛好重名時，在後面加 -2、-3…，不會蓋掉。
選好的名字當場建一個空檔佔住（生成與 06 輸出在不同行程，可能同時存進同一個資料夾）。
檔名太長時縮短變數的內容，序號與格式裡寫死的字留著。
Windows 檔名不能用的字元換成 _，資料夾一定在輸出資料夾裡面（.. 這類會被拿掉）。
"""

import re
import time
from pathlib import Path

import i18n

DEFAULT = "{date}/{seq}-{seed}-{width}x{height}"
# 06 輸出：編輯（單張、批次輸出）與拼接。{name} 是原圖的檔名（拼接是 collage）
EDIT_DEFAULT = "{date}/{name}_edit"
COLLAGE_DEFAULT = "{date}/collage-{date}-{time}"
DEFAULTS = {"gen": DEFAULT, "edit": EDIT_DEFAULT, "collage": COLLAGE_DEFAULT}
VARS = ("date", "time", "year", "month", "day", "seq", "seed", "width", "height",
        "model", "sampler", "steps", "cfg", "prompt", "loras", "name")
TOKEN = re.compile(r"\{(\w+)(?::(\d{1,3}))?\}")
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
MAX_NAME = 150           # 檔名（不含副檔名）最多幾個字，路徑太長 Windows 會存不了
PROMPT_LEN = 40          # {prompt} 沒指定長度時取前幾個字


def clean(seg, limit=MAX_NAME):
    """一段資料夾名稱或檔名：不能用的字元換成 _、頭尾的空白與點拿掉、保留字加 _；limit＝0 不截長度"""
    s = re.sub(r"\s+", " ", _BAD.sub("_", str(seg))).strip(" .")
    if s.split(".")[0].upper() in _RESERVED:
        s += "_"
    return s[:limit] if limit else s


def prompt_words(prompt):
    """{prompt} 用的文字：拿掉 <lora:…> 這類語法、權重與強調括號、BREAK，逗號留著"""
    s = re.sub(r"<[^>]*>", "", prompt or "")
    s = re.sub(r":\s*-?\d*\.?\d+\s*(?=[)\]])", "", s)
    s = re.sub(r"[()\[\]{}]|\bBREAK\b", "", s)
    s = re.sub(r"\s*,\s*", ", ", s.replace("\n", ", "))
    return re.sub(r"(, )+", ", ", s).strip(" ,")


def values(req, seed, model_name="", loras=(), sampler="", now=None):
    """這張圖的變數值（序號另外算）"""
    t = time.localtime(now if now is not None else time.time())
    cfg = req.get("cfg", "")
    try:
        cfg = f"{float(cfg):g}"
    except (TypeError, ValueError):
        cfg = str(cfg or "")
    return {
        "date": time.strftime("%Y-%m-%d", t), "time": time.strftime("%H%M%S", t),
        "year": time.strftime("%Y", t), "month": time.strftime("%m", t), "day": time.strftime("%d", t),
        "seed": str(seed), "width": str(req.get("width", "")), "height": str(req.get("height", "")),
        "model": Path(str(model_name or "")).stem, "sampler": str(sampler or req.get("sampler") or ""),
        "steps": str(req.get("steps", "")), "cfg": cfg,
        "prompt": prompt_words(req.get("user_prompt") or req.get("prompt") or ""),
        "loras": "+".join(Path(str(p)).stem for p, _ in (loras or [])),
    }


def _fill(template, vals, seq=None, cap=None):
    """cap：每個變數最多幾個字（檔名太長時縮短用，見 _stem）"""
    def one(m):
        key, n = m.group(1), m.group(2)
        if key not in VARS:
            return m.group(0)
        if key == "seq":
            return "" if seq is None else str(seq).zfill(int(n) if n else 6)
        v = vals.get(key, "")
        lim = int(n) if n else (PROMPT_LEN if key == "prompt" else 0)
        v = v[:lim].rstrip(" ,") if lim else v
        return v[:cap].rstrip(" ,") if cap is not None else v
    return TOKEN.sub(one, template)


def _stem(template, vals, seq):
    """檔名那段：填好、清好。超過 MAX_NAME 時縮短變數的內容，序號與字面的部分留著——整段截掉的話
    結尾的序號會一起不見，撞名時換序號也還是同一個名字（以前會一直換下去，存不了檔）。
    縮法：每個變數最多 L 個字，找放得下的最大 L（長的先縮、短的不動）；字面的部分就超過的才整段截"""
    full = lambda cap=None: clean(_fill(template, vals, seq, cap), limit=0)     # noqa: E731
    s = full()
    if len(s) <= MAX_NAME:
        return s
    lo, hi = 0, MAX_NAME
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(full(mid)) <= MAX_NAME:
            lo = mid
        else:
            hi = mid - 1
    return full(lo)[:MAX_NAME]


def _seq_pattern(template, ext=".png"):
    """檔名那段的格式 → 找同資料夾裡已有序號的正規式：序號是數字，其他變數什麼都可以"""
    lit = lambda s: re.escape(re.sub(r"\s+", " ", _BAD.sub("_", s)))     # 字面的部分照存檔時的換法
    out, pos, has = "", 0, False
    for m in TOKEN.finditer(template):
        out += lit(template[pos:m.start()])
        if m.group(1) == "seq" and not has:
            out += r"(\d+)"
            has = True
        elif m.group(1) in VARS:
            out += ".*?"
        else:
            out += re.escape(m.group(0))
        pos = m.end()
    out += lit(template[pos:])
    return (re.compile(rf"^{out}{re.escape(ext)}$", re.I) if has else None)


def unknown(pattern):
    """格式裡不認得的變數（照字面留在檔名裡）"""
    return sorted({m.group(1) for m in TOKEN.finditer(pattern or "") if m.group(1) not in VARS})


def split(pattern, default=DEFAULT):
    """格式 → 每一段；最後是 / 的（只寫了資料夾）檔名用預設的"""
    pat = (pattern or "").strip()
    parts = [p for p in re.split(r"[/\\]", pat) if p.strip()]
    if not parts:
        return split(default, default)
    if pat[-1] in "/\\":
        parts.append(default.split("/")[-1])
    return parts


def target(out_dir, pattern, vals, create=True, ext=".png", name_only=False, default=DEFAULT):
    """回傳要存的完整路徑（資料夾會建好、檔案先建一個空的佔住名字；create=False 是預覽，都不建）。pattern 空的就用 default。
    ext：副檔名（06 輸出可以是 .jpg、.webp）；name_only：只用檔名那段（匯出到使用者選的資料夾時不再開子資料夾）"""
    parts = split(pattern or default, default)
    if name_only:
        parts = parts[-1:]
    folder = Path(out_dir)
    for seg in parts[:-1]:
        s = clean(_fill(seg, vals, seq=""))
        if s:
            folder = folder / s
    name_t = parts[-1]
    if not clean(_fill(name_t, vals, seq=1)):
        name_t = split(default, default)[-1]        # 檔名那段整個是空的：用預設
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    rx = _seq_pattern(name_t, ext)
    seq = 1
    if rx and folder.is_dir():
        nums = [int(m.group(1)) for f in folder.iterdir() if (m := rx.match(f.name))]
        seq = max(nums) + 1 if nums else 1
    prev = None
    while True:
        stem = _stem(name_t, vals, seq)
        path = folder / f"{stem}{ext}"
        if _claim(path, create):
            return path
        if not rx or stem == prev:
            break       # 格式裡沒有序號，或換了序號名字還是一樣（字面的部分就超過長度）：改在後面加 -2、-3…
        prev, seq = stem, seq + 1
    for i in range(2, 100000):
        path = folder / f"{stem[:MAX_NAME - 7]}-{i}{ext}"
        if _claim(path, create):
            return path
    raise FileExistsError(i18n.tr("eng.name.tooMany", p=folder / stem))


def _claim(path, create):
    """這個名字能不能用。create 時順便建一個空檔佔住：生成（工作行程）與 06 輸出（主行程）可能同時往同一個資料夾存，
    只看「還不存在」的話兩邊會拿到同一個名字，後存的蓋掉先存的。存圖的人直接寫進這個檔，存失敗要自己刪掉"""
    if path.exists():
        return False
    if not create:
        return True
    try:
        with open(path, "xb"):
            pass
    except FileExistsError:
        return False
    return True
