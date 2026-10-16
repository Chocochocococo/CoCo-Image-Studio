# -*- coding: utf-8 -*-
"""設定的讀取與儲存；只有掃描資料夾真的改變時才重建索引。
"""

import importlib
import os
from pathlib import Path

from .library import run_index
from .resources import model_arch
from .state import C, SET, SETTINGS_APPLY_LOCK

from i18n import tr


def settings_out():
    """設定加上目前模型的架構與預設的輸出資料夾。

    零件覆寫是一個架構記一格，前端要知道現在該讀哪一格、
    點卡片又該寫進哪一格。這些是算出來的欄位，不存進 settings.json。
    """
    s = SET.load()
    s["MODEL_ARCH"] = model_arch(s.get("MODEL") or "")
    s["OUTPUT_DIR_DEFAULT"] = str(getattr(C, "OUTPUT_DIR_DEFAULT", C.OUTPUT_DIR))
    return s


# 換了不同架構的主模型：步數、CFG、取樣器、排程器跟著換（使用者要求：換架構後還是上一個架構的，常常不適合）。
# 換之前先記下舊架構的；新架構記過就用記的，沒用過用官方建議（shift 本來就一個架構一格）。
# 官方建議：Anima 是官方模型頁（30～50 步、CFG 4～5、er_sde；Turbo 版要 CFG 1、8～12 步，第一次之後會記住使用者改的），
# Z-Image、Krea 2（Turbo）、Qwen 2.1、FLUX.2 klein 照各自模組的 DEFAULT_CFG／DEFAULT_STEPS，SDXL、SD 1.5 照 config 的預設
ARCH_DEFAULTS = {
    "sdxl": {"STEPS": 28, "CFG": 5.0, "SAMPLER": "Euler a", "SCHEDULE": "Normal"},
    "sd15": {"STEPS": 28, "CFG": 5.0, "SAMPLER": "Euler a", "SCHEDULE": "Normal"},
    "anima": {"STEPS": 30, "CFG": 4.0, "SAMPLER": "ER-SDE", "SCHEDULE": "Normal"},
    "zimage": {"STEPS": 28, "CFG": 4.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    "krea2": {"STEPS": 8, "CFG": 1.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    "qwen21": {"STEPS": 40, "CFG": 1.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    # FLUX.2 klein 蒸餾版的官方用法（core/flux2.DEFAULT_*）；Base 版官方是 50 步、CFG 4，改了會記住
    "klein4": {"STEPS": 4, "CFG": 1.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    "klein9": {"STEPS": 4, "CFG": 1.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    # FLUX.1（core/flux1.DEFAULT_*）：dev 的 CFG 欄是 guidance（官方 3.5）；schnell 官方 4 步、不跑 CFG
    "flux1": {"STEPS": 20, "CFG": 3.5, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    "flux1s": {"STEPS": 4, "CFG": 1.0, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
    # Kontext 官方範例：28 步、guidance 2.5
    "kontext": {"STEPS": 28, "CFG": 2.5, "SAMPLER": "Euler", "SCHEDULE": "Normal"},
}


def arch_switch(before, model):
    """設定換成 model 時要一起改的：不同架構才有；回傳要存的 patch（含更新過的 ARCH_PARAMS）"""
    old, new = model_arch(before.get("MODEL") or ""), model_arch(model or "")
    if not new or old == new:
        return {}
    mem = SET.as_archparams(before.get("ARCH_PARAMS"))
    if old:
        mem[old] = {k: before[k] for k in SET.ARCH_PARAM_KEYS if before.get(k) is not None}
    return {**ARCH_DEFAULTS.get(new, {}), **mem.get(new, {}), "ARCH_PARAMS": mem}


def _inside(path, roots):
    """path 在 roots 其中一個裡面（或就是它）"""
    p = os.path.normcase(os.path.normpath(str(path)))
    for r in roots or []:
        r = os.path.normcase(os.path.normpath(str(r).strip()))
        if r and (p == r or p.startswith(r.rstrip("\\/") + os.sep)):
            return True
    return False


def _check_output(data, before):
    """換輸出資料夾：要是完整路徑、建得出來；不在圖庫要掃的資料夾裡就加進去（圖才看得到）。
    回傳錯誤訊息或 None；會直接改 data（補上 SCAN_DIRS）"""
    if "OUTPUT_DIR" not in data:
        return None
    raw = str(data.get("OUTPUT_DIR") or "").strip().strip('"')
    data["OUTPUT_DIR"] = raw
    out = Path(raw) if raw else Path(getattr(C, "OUTPUT_DIR_DEFAULT", C.OUTPUT_DIR))
    if raw and not out.is_absolute():
        return tr("srv.set.outFull")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return tr("srv.set.outMkdir", e=e)
    scans = data.get("SCAN_DIRS", before.get("SCAN_DIRS")) or []
    if not _inside(out, scans):
        data["SCAN_DIRS"] = [*[str(x) for x in scans], str(out)]
    return None


def _scan_dirs_key(values):
    """不碰檔案系統地比較掃描根目錄；離線磁碟不能在這裡 resolve。"""
    return tuple(sorted({
        os.path.normcase(os.path.normpath(str(v).strip()))
        for v in (values or []) if str(v).strip()
    }))


def update_settings(data):
    """儲存設定；只有掃描根目錄真的改變時才重建圖庫索引。"""
    with SETTINGS_APPLY_LOCK:
        before = SET.load()
        data = dict(data or {})
        err = _check_output(data, before)
        if err:
            return {"error": err}
        if "MODEL" in data:
            # 同一次也送了步數這些的（例如套用 Recipe）以送來的為準
            data = {**arch_switch(before, data["MODEL"]), **data}
        SET.save(data)
        importlib.reload(C)
        after = settings_out()
        scan_changed = (_scan_dirs_key(before.get("SCAN_DIRS")) !=
                        _scan_dirs_key(after.get("SCAN_DIRS")))

    idx = run_index(quiet=True) if scan_changed else {}
    out = {"ok": True, "settings": after,
           "index": idx.get("msg", ""),
           "reindexed": bool(idx.get("ok"))}
    if idx.get("error"):
        # 設定本身已成功寫入；索引失敗要獨立回報，不能把兩件事混成同一個 error。
        out["index_error"] = idx["error"]
    return out
