# -*- coding: utf-8 -*-
"""排程生成（使用者筆記 #7）：一次排好好幾項，依序自動生成；每 N 張休息、或等顯示卡降溫再繼續。

每一項是按「加入排程」那一刻生成台的完整設定（模型、LoRA、提示詞、參數、張數；動態語法已經展開），
所以可以排不同模型的工作。開始之後一張一張送進工作行程（worker_client），每張之間檢查：
要不要暫停、要不要休息、溫度降下來了沒。同一個模型連續送，工作行程不會重新載入。

存在 queue.json：關掉 CoCo 再開，清單還在、做到第幾張也記得，按「開始」就接著做（不會自己開始）。
做到一半的那一項：重新規劃後跳過已經處理過的那幾張（固定 seed 時 seed 照樣接得上）。
生成台自己按「生成」不受影響：排程等它做完再送下一張。
"""

import json
import secrets
import subprocess
import threading
import time
from pathlib import Path

from .state import C
from .worker_client import WORKER

from i18n import tr

QUEUE_FILE = getattr(C, "QUEUE_FILE", None) or (Path(C.DB_PATH).parent / "queue.json")
MAX_ITEMS = 500
COOL_MODES = ("off", "count", "temp")
COOL_DEFAULT = {"mode": "off", "every": 10, "rest": 30, "temp": 70}
TEMP_POLL = 3.0            # 等降溫時幾秒看一次

# 規劃一項要做的工作：handler 啟動時接上 Handler.plan_run（這裡不 import handler，免得互相 import）
PLAN = None

_LOCK = threading.RLock()
_RUN = {"thread": None, "state": "idle", "pausing": False, "msg": "", "until": 0.0, "temp": None,
        "current": None, "count": 0}


# ---------------------------------------------------------------- 存檔
def _load():
    try:
        d = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    items = [it for it in d.get("items") or [] if isinstance(it, dict) and isinstance(it.get("data"), dict)]
    for it in items:
        if it.get("status") == "running":      # 上次關掉時正在做：回到等待，做過的張數記著
            it["status"] = "waiting"
    return {"items": items, "cool": clean_cool(d.get("cool") or {})}


def _save(d):
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(QUEUE_FILE)


_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = _load()
    return _DATA


def _commit():
    _save(_data())


def clean_cool(c):
    out = dict(COOL_DEFAULT)
    if c.get("mode") in COOL_MODES:
        out["mode"] = c["mode"]
    for k, lo, hi in (("every", 1, 1000), ("rest", 1, 3600), ("temp", 40, 95)):
        try:
            out[k] = max(lo, min(hi, int(c.get(k, out[k]))))
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------- 清單
def _label(data):
    p = (data.get("prompts") or [data.get("prompt") or ""])[0]
    return " ".join(str(p).split())[:120]


def add(datas):
    """加入好幾項（每項是生成台送出的內容）。先規劃一次：設定有問題的當場告訴使用者，也算出幾張"""
    added, errors = [], []
    with _LOCK:
        d = _data()
        for data in datas[:MAX_ITEMS]:
            if not isinstance(data, dict):
                continue
            if len(d["items"]) >= MAX_ITEMS:
                errors.append(tr("srv.q.max", n=MAX_ITEMS))
                break
            data = {k: v for k, v in data.items() if k not in ("init_id", "mask", "sketch", "faces", "upscale", "outpaint", "edit")}
            plan = PLAN(data)
            if "error" in plan:
                errors.append(plan["error"])
                continue
            it = {"id": secrets.token_hex(6), "added": time.time(), "label": _label(data),
                  "model": Path(str(plan["load"].get("model") or "")).stem, "size": [data.get("width"), data.get("height")],
                  "data": data, "total": len(plan["jobs"]), "done": 0, "failed": 0, "status": "waiting", "error": "", "files": []}
            d["items"].append(it)
            added.append(it["id"])
        _commit()
    return {"ok": bool(added), "added": len(added), "errors": errors}


def remove(item_id):
    with _LOCK:
        d = _data()
        it = next((x for x in d["items"] if x["id"] == item_id), None)
        if not it:
            return {"error": tr("srv.q.notFound")}
        if it["status"] == "running":
            return {"error": tr("srv.q.running")}
        d["items"].remove(it)
        _commit()
    return {"ok": True}


def move(item_id, to):
    with _LOCK:
        items = _data()["items"]
        it = next((x for x in items if x["id"] == item_id), None)
        if not it:
            return {"error": tr("srv.q.notFound")}
        items.remove(it)
        try:
            to = max(0, min(len(items), int(to)))
        except (TypeError, ValueError):
            to = len(items)
        items.insert(to, it)
        _commit()
    return {"ok": True}


def clear_finished():
    with _LOCK:
        d = _data()
        n = len(d["items"])
        d["items"] = [x for x in d["items"] if x["status"] not in ("done", "failed")]
        _commit()
    return {"ok": True, "removed": n - len(d["items"])}


def retry(item_id):
    """有失敗的那一項：失敗的那幾張重做（做好的不重做）"""
    with _LOCK:
        it = next((x for x in _data()["items"] if x["id"] == item_id), None)
        if not it or it["status"] != "failed":
            return {"error": tr("srv.q.onlyFailed")}
        it.update(status="waiting", failed=0, error="", seen=list(it.get("ok_idx") or []))
        _commit()
    return {"ok": True}


def set_cool(c):
    with _LOCK:
        _data()["cool"] = clean_cool(c or {})
        _commit()
        return dict(_data()["cool"])


def listing():
    with _LOCK:
        d = _data()
        items = [{k: v for k, v in it.items() if k != "data"} for it in d["items"]]
        return {"items": items, "cool": dict(d["cool"]), "run": summary()}


def summary():
    """給生成台輪詢用（/api/gen/state 一起帶）：排程在做什麼、還剩幾張"""
    with _LOCK:
        items = _data()["items"]
        left = sum(max(0, it["total"] - it["done"] - it["failed"]) for it in items if it["status"] in ("waiting", "running"))
        return {"state": _RUN["state"], "pausing": _RUN["pausing"], "msg": _RUN["msg"], "left": left,
                "waiting": sum(1 for it in items if it["status"] in ("waiting", "running")),
                "until": _RUN["until"], "temp": _RUN["temp"], "target": _data()["cool"]["temp"], "current": _RUN["current"]}


# ---------------------------------------------------------------- 溫度
_NVML = {"handle": None, "failed": False}


def gpu_temp():
    """顯示卡溫度（°C）；讀不到回 None。先用 NVML（nvidia-ml-py，requirements-lock 直接列），沒有就問 nvidia-smi"""
    if not _NVML["failed"]:
        try:
            import pynvml
            if _NVML["handle"] is None:
                pynvml.nvmlInit()
                _NVML["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetTemperature(_NVML["handle"], pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            _NVML["failed"] = True
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------- 執行
def start():
    with _LOCK:
        t = _RUN["thread"]
        if t and t.is_alive():
            _RUN["pausing"] = False                # 暫停中又按開始：取消暫停
            return {"ok": True}
        if not any(it["status"] == "waiting" for it in _data()["items"]):
            return {"error": tr("srv.q.empty")}
        _RUN.update(pausing=False, state="running", msg="", count=0)
        t = threading.Thread(target=_run, daemon=True)
        _RUN["thread"] = t
        t.start()
    return {"ok": True}


def pause():
    """做完正在生成的這張就停"""
    with _LOCK:
        t = _RUN["thread"]
        if not (t and t.is_alive()):
            return {"ok": True}
        _RUN["pausing"] = True
    return {"ok": True}


def _set(state, msg="", **kw):
    with _LOCK:
        _RUN.update(state=state, msg=msg, **kw)


def _sleep_until(pred, step=0.3):
    """等到 pred() 為真；中途按了暫停回 False"""
    while not pred():
        if _RUN["pausing"]:
            return False
        time.sleep(step)
    return True


def _cool_down(cool, sleep=time.sleep):
    """每張生成之前：該休息就休息、太熱就等。中途按了暫停回 False"""
    if cool["mode"] == "count" and _RUN["count"] and _RUN["count"] % cool["every"] == 0:
        until = time.time() + cool["rest"]
        _set("cooling", tr("srv.q.rest", s=cool['rest']), until=until)
        if not _sleep_until(lambda: time.time() >= until):
            return False
        _RUN["count"] = 0
    if cool["mode"] == "temp":
        while True:
            t = gpu_temp()
            _set("hot" if t is not None and t > cool["temp"] else "running", "", temp=t, until=0.0)
            if t is None or t <= cool["temp"]:
                break
            end = time.time() + TEMP_POLL
            if not _sleep_until(lambda: time.time() >= end):
                return False
    _set("running", "", until=0.0)
    return True


def _one(load, job):
    """送一張進工作行程，等它做完；回傳進度表上那一列（狀態、檔案）"""
    if not _sleep_until(lambda: not WORKER.state.get("running") and not WORKER.jobs and not WORKER.pending):
        return None
    while not WORKER.submit(load, [dict(job, id=0)]):     # 剛好被生成台搶先：等它做完再送
        if not _sleep_until(lambda: not WORKER.state.get("running")):
            return None
    batch = WORKER.batch
    while WORKER.state.get("running") or WORKER.jobs:      # 按了暫停也要等這張做完
        time.sleep(0.3)
    return next((r for r in reversed(WORKER.table) if r["b"] == batch), None)


def _run():
    try:
        while not _RUN["pausing"]:
            with _LOCK:
                it = next((x for x in _data()["items"] if x["status"] == "waiting"), None)
                cool = dict(_data()["cool"])
            if not it:
                break
            plan = PLAN(it["data"])
            if "error" in plan:
                with _LOCK:
                    it.update(status="failed", error=plan["error"])
                    _commit()
                continue
            with _LOCK:
                it["status"] = "running"
                _RUN["current"] = it["id"]
                _commit()
            jobs = plan["jobs"]
            seen = set(it.get("seen") or [])       # 處理過的（成功或失敗）：接著做時跳過
            stopped = False
            for idx, job in enumerate(jobs):
                if idx in seen:
                    continue
                if _RUN["pausing"] or not _cool_down(cool):
                    stopped = True
                    break
                row = _one(plan["load"], job)
                status = (row or {}).get("status")
                if status in (None, "aborted", "cancelled"):
                    # 工作行程被停掉、或這張被取消：這張不算，整個排程先停
                    _set("idle", tr("srv.q.crashed") if status == "aborted" else "")
                    stopped = True
                    _RUN["pausing"] = True
                    break
                with _LOCK:
                    it.setdefault("seen", []).append(idx)
                    if status in ("done", "skipped"):
                        it["done"] += 1
                        it.setdefault("ok_idx", []).append(idx)
                        if row.get("path"):
                            it["files"] = (it["files"] + [row["path"]])[-8:]
                    else:
                        it["failed"] += 1
                        it["error"] = tr("srv.q.someFailed")
                    _commit()
                _RUN["count"] += 1
            with _LOCK:
                if stopped and it["done"] + it["failed"] < it["total"]:
                    it["status"] = "waiting"
                else:
                    it["status"] = "failed" if it["failed"] else "done"
                _commit()
    finally:
        with _LOCK:
            msg = _RUN["msg"] if _RUN["state"] == "idle" else ""
            _RUN.update(state="idle", pausing=False, current=None, until=0.0, msg=msg)
