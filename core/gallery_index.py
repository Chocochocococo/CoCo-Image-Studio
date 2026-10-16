# -*- coding: utf-8 -*-
"""
gallery_index.py —— 圖庫索引器（階段 A・第一次）

掃描資料夾裡的圖片，讀出 PNG 內嵌的生成參數，建成 SQLite 索引。

特性：
  - 增量掃描：檔案沒變就跳過，重跑很快
  - 評星不會被覆蓋：重新索引時保留既有評分
  - 全文檢索：提示詞用 SQLite FTS5 建索引
  - 支援 A1111 參數格式、本專案的 coco JSON、ComfyUI 與 TensorArt（見 meta_formats.py）

執行： run_index.bat
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C

import i18n

HERE = ROOT
SCAN_DIRS = C.SCAN_DIRS
DB_PATH = C.DB_PATH
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
REBUILD = False          # True = 砍掉重建（評星也會消失）

# 認得的格式變多時加一：之前讀不到參數的圖（prompt 與 raw 都空）會在下次掃描重讀一次，
# 不必整個重建。檔案沒變的圖平常是直接跳過的，不加這個它們永遠不會被重新解析。
#   2  ComfyUI、TensorArt、SwarmUI
#   3  NovelAI
PARSER_VERSION = 3


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    filename  TEXT,
    folder    TEXT,
    mtime     REAL,
    filesize  INTEGER,
    width     INTEGER,
    height    INTEGER,
    prompt    TEXT DEFAULT '',
    negative  TEXT DEFAULT '',
    steps     INTEGER,
    cfg       REAL,
    seed      INTEGER,
    sampler   TEXT,
    scheduler TEXT,
    model     TEXT,
    clip_skip INTEGER,
    loras     TEXT DEFAULT '[]',
    extra     TEXT DEFAULT '{}',
    raw       TEXT DEFAULT '',
    rating    INTEGER DEFAULT 0,
    indexed   REAL
);
CREATE TABLE IF NOT EXISTS index_meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS idx_folder ON images(folder);
CREATE INDEX IF NOT EXISTS idx_model  ON images(model);
CREATE INDEX IF NOT EXISTS idx_rating ON images(rating);
CREATE INDEX IF NOT EXISTS idx_mtime  ON images(mtime);

CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    prompt, negative, content='images', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, prompt, negative)
    VALUES (new.id, new.prompt, new.negative);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt, negative)
    VALUES ('delete', old.id, old.prompt, old.negative);
END;
CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, prompt, negative)
    VALUES ('delete', old.id, old.prompt, old.negative);
    INSERT INTO images_fts(rowid, prompt, negative)
    VALUES (new.id, new.prompt, new.negative);
END;
"""


# ---------------------------------------------------------------- 參數解析
_re_kv = re.compile(r"\s*([\w \-/]+):\s*"
                    r"(\"[^\"]*\"|\{[^}]*\}|[^,]*)(?:,|$)")
_re_lora = re.compile(r"<lora:([^:>]+):([\d.\-]+)(?::[\d.\-]+)?>")


def parse_a1111(text):
    """解析 A1111 的 parameters 字串"""
    if not text:
        return {}
    lines = text.split("\n")

    neg_i = next((i for i, l in enumerate(lines)
                  if l.startswith("Negative prompt:")), None)

    # 最後一行若含有 "Key: value" 就是參數列
    tail_i = len(lines) - 1
    while tail_i >= 0 and not lines[tail_i].strip():
        tail_i -= 1
    tail = lines[tail_i] if tail_i >= 0 else ""
    has_params = bool(re.search(r"\b(Steps|Sampler|Seed|CFG scale):", tail))

    if neg_i is not None:
        prompt = "\n".join(lines[:neg_i]).strip()
        end = tail_i if has_params else len(lines)
        neg = "\n".join(lines[neg_i:end]).strip()
        neg = neg[len("Negative prompt:"):].strip()
    else:
        end = tail_i if has_params else len(lines)
        prompt = "\n".join(lines[:end]).strip()
        neg = ""

    kv = {}
    if has_params:
        for m in _re_kv.finditer(tail):
            k = m.group(1).strip()
            v = m.group(2).strip().strip('"')
            if k:
                kv[k] = v

    def num(key, cast, *alts):
        for k in (key,) + alts:
            if k in kv:
                try:
                    return cast(kv[k])
                except Exception:
                    return None
        return None

    size = kv.get("Size", "")
    w = h = None
    if "x" in size:
        try:
            w, h = (int(x) for x in size.split("x")[:2])
        except Exception:
            pass

    known = {"Steps", "Sampler", "CFG scale", "Seed", "Size", "Model",
             "Model hash", "Clip skip", "Schedule type", "Scheduler"}
    return {
        "prompt": prompt,
        "negative": neg,
        "steps": num("Steps", int),
        "cfg": num("CFG scale", float),
        "seed": num("Seed", int),
        "sampler": kv.get("Sampler"),
        "scheduler": kv.get("Schedule type") or kv.get("Scheduler"),
        "model": kv.get("Model") or kv.get("Model hash"),
        "clip_skip": num("Clip skip", int),
        "width": w, "height": h,
        "extra": {k: v for k, v in kv.items() if k not in known},
    }


_re_hash = re.compile(r"\s*\[[0-9a-fA-F]{6,12}\]\s*$")
_re_seq = re.compile(r"[_\-]\d{4,6}[_\-]?$")


def normalize_model(name):
    """把同一個模型的各種寫法統一

    waiIllustriousSDXL_v160.safetensors → waiIllustriousSDXL_v160
    someModel [a1b2c3d4]                → someModel
    synta-XL IL Mix_00001_              → synta-XL IL Mix
    """
    if not name:
        return None
    n = str(name).strip().strip('"')
    for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".sft"):
        if n.lower().endswith(ext):
            n = n[: -len(ext)]
    n = _re_hash.sub("", n)
    n = _re_seq.sub("", n)
    return n.strip(" _-") or None


def parse_a1111_owned(raw):
    """raw 是 A1111 的文字（不是 SwarmUI 的 JSON）：那種才沿用它寫的 Size"""
    return bool(raw) and not raw.lstrip().startswith("{")


def read_meta(path: Path):
    """從圖片讀出生成參數。回傳 dict，讀不到就回空的"""
    from PIL import Image

    out = {}
    try:
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            info = dict(getattr(im, "info", {}) or {})

            raw = info.get("parameters", "")
            comfy = info.get("prompt", "")              # ComfyUI 的執行圖
            tensorart = info.get("generation_data", "")
            novelai = {k: info.get(k) for k in ("Software", "Comment", "Description", "Source")}

            # 非 PNG 走 EXIF。UserComment 在 Exif 子目錄（0x8769）裡，A1111／Forge 與
            # SwarmUI 都寫這裡；以前只看最上層，JPG 的參數一直讀不到
            if not raw and path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
                try:
                    import meta_formats as MF
                    exif = im.getexif()
                    try:
                        sub = exif.get_ifd(0x8769)
                    except Exception:
                        sub = {}
                    for v in (sub.get(0x9286), exif.get(0x9286)):
                        text = MF.user_comment(v).strip()
                        if "Steps:" in text or MF.is_swarm(text):
                            raw = text
                            break
                    # ComfyUI 存 WebP／AVIF 時把執行圖放在 0x0110，內容是 "prompt:{…}"
                    v = exif.get(0x0110)
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", "ignore")
                    if not comfy and isinstance(v, str) and v.startswith("prompt:"):
                        comfy = v[len("prompt:"):]
                except Exception:
                    pass

            coco = info.get("coco", "")
    except Exception as e:
        return {"error": str(e)}

    import meta_formats as MF
    if MF.is_swarm(raw):
        parsed = MF.parse_swarm(raw)             # SwarmUI 也用 parameters 這個名字，但放的是 JSON
    elif not raw and MF.is_novelai(novelai):
        parsed = MF.parse_novelai(novelai)
    else:
        parsed = parse_a1111(raw) if raw else {}
    if not raw and (tensorart or comfy):
        parsed = (MF.parse_tensorart(tensorart, comfy) if tensorart else {}) or \
            (MF.parse_comfy(comfy) if comfy else {})
    if not parse_a1111_owned(raw):
        # 寬高用實際的圖：執行圖裡的是 latent 的大小，之後可能又放大過
        parsed.pop("width", None)
        parsed.pop("height", None)
    # coco JSON 的欄位優先，因為是結構化的
    if coco:
        try:
            c = json.loads(coco)
            for k_src, k_dst in (("prompt", "prompt"), ("negative", "negative"),
                                 ("steps", "steps"), ("cfg", "cfg"),
                                 ("seed", "seed"), ("sampler", "sampler"),
                                 ("model", "model"),
                                 ("clip_skip", "clip_skip")):
                if c.get(k_src) not in (None, ""):
                    parsed[k_dst] = c[k_src]
            if c.get("loras"):
                parsed["loras"] = c["loras"]
            if c.get("conditioning"):
                parsed.setdefault("extra", {})["conditioning"] = \
                    c["conditioning"]
            # img2img 的底圖（完整路徑）與強度：精修頁靠 init 找同一張的其他版本
            if c.get("init"):
                ex = parsed.setdefault("extra", {})
                ex["init"] = c["init"]
                if c.get("strength") is not None:
                    ex["strength"] = c["strength"]
            # 局部重繪：遮罩檔（精修頁「用同一個遮罩」）、範圍、柔化
            inp = c.get("inpaint")
            if isinstance(inp, dict) and inp.get("mask"):
                ex = parsed.setdefault("extra", {})
                ex["inpaint"] = inp.get("area") or "whole"
                ex["mask"] = inp["mask"]
                if inp.get("blur") is not None:
                    ex["mask_blur"] = inp["blur"]
            # 用了哪個 VAE 與文字編碼器（圖庫存成 Recipe 用）
            pt = c.get("parts")
            if isinstance(pt, dict):
                parsed.setdefault("extra", {})["parts"] = {k: str(pt[k]) for k in ("vae", "te") if pt.get(k)}
            # 修臉：修了幾張臉（精修頁的版本列標「修臉」）
            fc = c.get("faces")
            if isinstance(fc, dict) and fc.get("items"):
                parsed.setdefault("extra", {})["faces"] = len(fc["items"])
            # 去背：精修頁的版本列標「去背」
            if isinstance(c.get("rembg"), dict):
                parsed.setdefault("extra", {})["rembg"] = 1
            # 高清化：精修頁的版本列標「高清化 ×2」
            if isinstance(c.get("outpaint"), dict):
                parsed.setdefault("extra", {})["outpaint"] = 1
            # 用 Qwen 修圖（Qwen Image 2.1）：精修頁的版本列標「修圖」
            if c.get("edit"):
                parsed.setdefault("extra", {})["edit"] = 1
            # ControlNet：參考圖、模型、種類、強度（送到生成區時帶回參考圖；比較頁比得出差在哪）
            cn = c.get("controlnet")
            if isinstance(cn, list) and cn:
                parsed.setdefault("extra", {})["controlnet"] = [u for u in cn if isinstance(u, dict)][:5]
            up = c.get("upscale")
            if isinstance(up, dict) and up.get("scale"):
                parsed.setdefault("extra", {})["upscale"] = {"scale": up["scale"], "detail": bool(up.get("detail"))}
            # 品質標籤：開關（比較頁看得出差在哪）、自動加上的、使用者自己寫的（送到生成區用）
            pre = c.get("preset")
            if isinstance(pre, dict):
                ex = parsed.setdefault("extra", {})
                ex["preset"] = "on" if pre.get("on") else "off"
                if pre.get("quality"):
                    ex["preset_quality"] = pre["quality"]
                if pre.get("negative"):
                    ex["preset_negative"] = pre["negative"]
                for k in ("user_prompt", "user_negative"):
                    if c.get(k) is not None:
                        ex[k] = c[k]
        except Exception:
            pass

    # 提示詞裡的 <lora:名稱:權重>
    if "loras" not in parsed:
        found = _re_lora.findall(parsed.get("prompt", "") or "")
        parsed["loras"] = [{"file": n, "weight": float(w)} for n, w in found]

    parsed.setdefault("width", out.get("width"))
    parsed.setdefault("height", out.get("height"))
    parsed["raw"] = raw
    if not parsed.get("width"):
        parsed["width"], parsed["height"] = out.get("width"), out.get("height")
    return parsed


# ---------------------------------------------------------------- 索引
def open_db():
    if REBUILD and DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


UPSERT = """
    INSERT INTO images (path, filename, folder, mtime, filesize,
        width, height, prompt, negative, steps, cfg, seed, sampler,
        scheduler, model, clip_skip, loras, extra, raw, rating, indexed)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(path) DO UPDATE SET
        mtime=excluded.mtime, filesize=excluded.filesize,
        width=excluded.width, height=excluded.height,
        prompt=excluded.prompt, negative=excluded.negative,
        steps=excluded.steps, cfg=excluded.cfg, seed=excluded.seed,
        sampler=excluded.sampler, scheduler=excluded.scheduler,
        model=excluded.model, clip_skip=excluded.clip_skip,
        loras=excluded.loras, extra=excluded.extra, raw=excluded.raw,
        indexed=excluded.indexed
"""


def _sql_int(v):
    """SQLite 的整數只到 ±2^63：更大的（ComfyUI 的 seed 可以到 2^64）存成空的，
    不然整張圖寫不進索引、圖庫看不到。完整的值還在圖的參數文字（raw）裡"""
    if isinstance(v, int) and not -(1 << 63) <= v < (1 << 63):
        return None
    return v


def file_row(f, st, meta, rating=0):
    """UPSERT 用的一列"""
    return (
        str(f), f.name, str(f.parent), st.st_mtime, st.st_size,
        meta.get("width"), meta.get("height"),
        meta.get("prompt") or "", meta.get("negative") or "",
        _sql_int(meta.get("steps")), meta.get("cfg"), _sql_int(meta.get("seed")),
        meta.get("sampler"), meta.get("scheduler"),
        normalize_model(meta.get("model")),
        meta.get("clip_skip"),
        json.dumps(meta.get("loras", []), ensure_ascii=False),
        json.dumps(meta.get("extra", {}), ensure_ascii=False),
        meta.get("raw", ""),
        rating,
        time.time(),
    )


def index_file(con, path, rating=0):
    """單一檔案加進索引（回收桶還原、沒有留下索引資料的檔案用）。讀不出來回 False"""
    f = Path(path)
    meta = read_meta(f)
    if "error" in meta:
        return False
    con.execute(UPSERT, file_row(f, f.stat(), meta, rating))
    return True


def scan(con, roots):
    cur = con.cursor()
    known = {r["path"]: (r["mtime"], r["filesize"], r["rating"], r["nometa"])
             for r in cur.execute(
                 "SELECT path, mtime, filesize, rating, "
                 "(prompt = '' AND raw = '') AS nometa FROM images")}
    got = cur.execute("SELECT v FROM index_meta WHERE k = 'parser'").fetchone()
    reparse = (int(got[0]) if got else 1) < PARSER_VERSION

    seen = set()
    added = updated = skipped = failed = 0
    t0 = time.time()

    files = []
    configured_roots = [Path(root) for root in roots]
    missing_roots = []
    for root in configured_roots:
        if not root.is_dir():
            print("  [!] " + i18n.tr("eng.gi.noDir", root=root))
            missing_roots.append(root)
            continue
        files += [p for p in root.rglob("*") if p.suffix.lower() in EXTS]

    print("  " + i18n.tr("eng.gi.found", n=len(files)) + "\n")

    for i, f in enumerate(files, 1):
        key = str(f)
        seen.add(key)
        try:
            st = f.stat()
        except Exception:
            failed += 1
            continue

        prev = known.get(key)
        if prev and abs(prev[0] - st.st_mtime) < 1e-6 and prev[1] == st.st_size \
                and not (reparse and prev[3]):
            skipped += 1
            continue

        meta = read_meta(f)
        if "error" in meta:
            print("  [!] " + i18n.tr("eng.gi.readFail", name=f.name, e=meta['error'][:60]))
            failed += 1
            continue

        cur.execute(UPSERT, file_row(f, st, meta, prev[2] if prev else 0))   # 保留既有評星

        if prev:
            updated += 1
        else:
            added += 1
        if i % 200 == 0:
            con.commit()
            print(f"  ... {i}/{len(files)}")

    # 只刪除確定不存在的索引。某個設定中的 root 暫時離線時，本輪根本無法
    # 判斷檔案是否消失，必須保留舊資料；從設定移除的 root 則照常清掉。
    #
    # 這裡刻意只做字串比較，不用 resolve()：離線的 root 通常正是拔掉的外接碟
    # 或斷線的網路磁碟，resolve() 會去碰檔案系統，每個候選路徑都可能卡到
    # timeout。abspath 只做語彙正規化（分隔符號與 ..），normcase 處理
    # Windows 不分大小寫，兩者都不發出 I/O。
    def norm(p):
        return os.path.normcase(os.path.abspath(str(p)))

    # 尾端補分隔符號，否則 rootB 會誤蓋到 rootBB 底下的檔案
    offline = [norm(root) + os.sep for root in missing_roots]

    gone = []
    for p in known:
        if p in seen:
            continue
        if offline:
            np = norm(p)
            if any(np.startswith(pre) for pre in offline):
                continue          # root 離線，這輪無從判斷，保留
        gone.append(p)
    for p in gone:
        cur.execute("DELETE FROM images WHERE path=?", (p,))

    # 既有列也套用正規化（規則調整後不必整個重建）
    fixed = 0
    for r in cur.execute(
            "SELECT id, model FROM images WHERE model IS NOT NULL").fetchall():
        nm = normalize_model(r["model"])
        if nm != r["model"]:
            cur.execute("UPDATE images SET model=? WHERE id=?", (nm, r["id"]))
            fixed += 1

    cur.execute("INSERT OR REPLACE INTO index_meta (k, v) VALUES ('parser', ?)",
                (str(PARSER_VERSION),))
    con.commit()
    return {"added": added, "updated": updated, "skipped": skipped,
            "normalized": fixed,
            "failed": failed, "removed": len(gone),
            "secs": time.time() - t0}


def report(con):
    cur = con.cursor()
    total = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"\n  索引總數：{total}")
    if not total:
        return

    withp = cur.execute(
        "SELECT COUNT(*) FROM images WHERE prompt != ''").fetchone()[0]
    print(f"  有參數的：{withp}（{withp * 100 // max(total,1)}%）")

    print("\n  依模型：")
    for r in cur.execute("""SELECT COALESCE(model,'(未知)') m, COUNT(*) c
                            FROM images GROUP BY m ORDER BY c DESC LIMIT 8"""):
        print(f"    {r['c']:5d}  {r['m']}")

    print("\n  依取樣器：")
    for r in cur.execute("""SELECT COALESCE(sampler,'(未知)') s, COUNT(*) c
                            FROM images GROUP BY s ORDER BY c DESC LIMIT 5"""):
        print(f"    {r['c']:5d}  {r['s']}")

    print("\n  最近三張：")
    for r in cur.execute("""SELECT filename, seed, steps, cfg, sampler, prompt
                            FROM images ORDER BY mtime DESC LIMIT 3"""):
        print(f"    {r['filename']}")
        print(f"      seed={r['seed']} steps={r['steps']} "
              f"cfg={r['cfg']} sampler={r['sampler']}")
        p = (r["prompt"] or "").replace("\n", " ")
        print(f"      {p[:90]}{'...' if len(p) > 90 else ''}")

    # 檢索示範
    print("\n  全文檢索示範（搜尋 'masterpiece'）：")
    try:
        n = cur.execute("""SELECT COUNT(*) FROM images_fts
                           WHERE images_fts MATCH ?""",
                        ("masterpiece",)).fetchone()[0]
        print(f"    命中 {n} 張")
    except Exception as e:
        print(f"    [!] {e}")


def main():
    print("=" * 58)
    print("  圖庫索引器")
    print("=" * 58)
    print(f"  資料庫：{DB_PATH}")
    for d in SCAN_DIRS:
        print(f"  掃描　：{d}")
    print()

    con = open_db()
    st = scan(con, SCAN_DIRS)
    print(f"\n  新增 {st['added']} · 更新 {st['updated']} · "
          f"跳過 {st['skipped']} · 失敗 {st['failed']} · "
          f"移除 {st['removed']}")
    if st.get("normalized"):
        print(f"  模型名稱正規化 {st['normalized']} 筆")
    print(f"  耗時 {st['secs']:.1f} 秒")
    report(con)
    con.close()
    print("\n  完成。再跑一次會只處理有變動的檔案。\n")


if __name__ == "__main__":
    main()
