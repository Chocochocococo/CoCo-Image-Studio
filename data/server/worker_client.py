# -*- coding: utf-8 -*-
"""管理生成用的子行程（core/worker.py）。
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import i18n
from i18n import tr

from .state import C, ROOT, SET
from .prompts import _DYN_SYNTAX, extract_loras


# ---------------------------------------------------------------- 生成
# stderr 不全是錯誤：套件的警告（[transformers] …、FutureWarning…）也走這裡。
# 警告記成 warn，只有真的錯誤才標紅、顯示在生成鈕下面。
# 套件 logger 印的一律算警告：真的出錯會變成 exception，traceback 那幾行才是錯誤
_LIB = re.compile(r"^\[(transformers|diffusers|accelerate|huggingface_hub)\]")
_WARN = re.compile(r"warn(ing)?\b", re.I)
_ERR = re.compile(r"error|exception|traceback", re.I)


# diffusers／transformers 載入時的 tqdm 進度條（「Loading pipeline components...: 100%|███|」）不是錯誤
_PROGRESS = re.compile(r"\d+%\|")


def stderr_kind(line):
    if _PROGRESS.search(line):
        return "info"
    if _LIB.match(line):
        return "warn"
    return "warn" if _WARN.search(line) and not _ERR.search(line) else "err"


class Worker:
    """管理生成用的子行程。模型常駐在那邊，崩潰也只影響它自己。"""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        # 啟動與停止：兩個請求同時叫起工作行程（查 token 與送出生成），只能起一個
        self._life = threading.Lock()
        self.state = {"running": False, "step": 0, "total": 0,
                      "queue": 0, "msg": "", "last": None, "alive": False}
        self.log = []
        self.jobs = []
        # 批次精修：原圖的模型不一樣時分成好幾組，一組做完換下一組的模型接著做 [(載入設定, 工作), …]
        self.pending = []
        # 進度表保留整個工作台行程的歷史，關掉程式才清空。
        # 每次送出算一個 batch，因為工作編號每批都從 0 重新算，
        # 只靠編號會標錯列。
        self.table = []
        self.batch = 0
        self.seq = 0         # 跨批次的連續編號，給介面顯示用
        self._reader = None
        # token 計數查詢：qid → (Event, 結果)。worker 的主迴圈是串行的，
        # 生成中不讀 stdin，所以一定要有 timeout（見 ask_tokens）。
        self._tok_seq = 0
        self._tok_waits = {}

    TABLE_MAX = 300          # 只留最近這麼多列，免得跑一整天塞爆記憶體

    # -- 行程 --
    def alive(self):
        p = self.proc
        return p is not None and p.poll() is None

    def start(self):
        with self._life:
            if self.alive():
                return True
            from . import train
            if train.busy():                     # 訓練中：顯示卡留給訓練
                return False
            import sys as _s
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"     # 中文訊息才不會弄死子行程
            env["PYTHONUTF8"] = "1"
            env["COCO_LANG"] = i18n.lang()     # 子行程的訊息照介面語言（之後每個指令也帶，見 send）
            proc = subprocess.Popen(
                [_s.executable, "-u", str(ROOT / "core" / "worker.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", env=env, cwd=str(ROOT),
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if os.name == "nt" else 0))
            self.errbuf = []
            self.errshown = 0
            self.proc = proc
            # 讀取的執行緒綁定這個行程：停掉換新的之後，舊的收尾不會動到新的佇列（見 _read 的最後）
            self._reader = threading.Thread(target=self._read, args=(proc,), daemon=True)
            self._reader.start()
            threading.Thread(target=self._read_err, args=(proc,), daemon=True).start()
            self.state["alive"] = True
            return True

    def stop(self):
        with self._life:
            proc = self.proc
            if proc is not None and proc.poll() is None:
                try:
                    self.send({"cmd": "quit"})
                    proc.wait(timeout=8)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self.proc = None
            with self.lock:
                self._abandon()                  # 舊行程的讀取執行緒不再收尾（已經不是目前的），剩下的這裡標掉
            self.state.update(alive=False, running=False, step=0, total=0,
                              queue=0, msg=tr("srv.wc.stopped"))

    def _abandon(self):
        """工作行程沒了：還沒做的標「已中斷」、清掉（呼叫的人拿著 self.lock）。回傳有沒有清到東西"""
        left = self.jobs + [j for _, js in self.pending for j in js]
        for j in left:
            self._mark(j["id"], "aborted")
        self.jobs = []
        self.pending = []
        for r in self.table:
            if r["status"] in ("running", "waiting") and r["b"] == self.batch:
                r["status"] = "aborted"
        return bool(left)

    def send(self, obj):
        if not self.alive():
            return False
        try:
            self.proc.stdin.write(json.dumps({**obj, "lang": i18n.lang()}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            return True
        except Exception:
            return False

    def ask_tokens(self, prompt, negative, timeout=1.5):
        """問 worker 這段提示詞幾個 token。逾時或沒模型都回 ready=False。

        worker 的主迴圈是串行的：生成中不讀 stdin，這個查詢會排到生成
        結束才處理。所以**一定要有 timeout**，否則使用者一邊生成一邊打字
        就會把介面卡住。逾時不是錯誤，只是「現在算不出來」。
        """
        # 沒在跑就順手起來。**啟動很便宜**——worker 只是個等指令的 Python
        # 行程，重的是 cmd:load。不啟動的話使用者在第一次生成之前
        # 永遠看不到 token 數，而那正是最需要看到的時候。
        cold = not self.alive()
        if cold:
            try:
                self.start()
            except Exception:
                return {"ready": False, "why": tr("srv.wc.noStart")}
            # 剛起來的話要等它 import transformers 並暖好 tokenizer，
            # 實測約 2 秒。用平常的 1.5 秒會讓**第一次**查詢必定逾時。
            timeout = max(timeout, 8.0)
        if not self.alive():
            return {"ready": False, "why": tr("srv.wc.notRunning")}
        with self.lock:
            self._tok_seq += 1
            qid = self._tok_seq
            ev = threading.Event()
            self._tok_waits[qid] = [ev, None]
        try:
            # 帶上選到的模型：沒載入時 worker 要靠它判斷架構，
            # Anima 走 Qwen 不能拿 CLIP 去估（見 _fallback_fields）。
            try:
                model = SET.load().get("MODEL") or C.MODEL
            except Exception:
                model = None
            # **要送跟生成時同一份文字。** <lora:名字:權重> 在建工作時就被
            # extract_loras 拿掉了，編碼器根本看不到；照原字串數會多算
            # （一個 LoRA 名字就十幾個 token）。這裡刻意呼叫同一支
            # 函式而不是另寫一個正規表示式，免得兩邊的規則哪天走鐘。
            prompt = extract_loras(prompt)[0]
            negative = extract_loras(negative)[0]
            dyn = bool(_DYN_SYNTAX.search(prompt or ""))
            if not self.send({"cmd": "tokens", "qid": qid, "model": model,
                              "prompt": prompt, "negative": negative}):
                return {"ready": False, "why": tr("srv.wc.sendFail")}
            if not ev.wait(timeout):
                return {"ready": False, "why": tr("srv.wc.busy")}
            with self.lock:
                got = self._tok_waits[qid][1] or {"ready": False}
            # 動態語法是前端展開後才送去生成的，這裡數的是**模板**，
            # 不是實際會被編碼的那幾則。標出來，不要讓數字看起來很篤定。
            if dyn and got.get("ready"):
                got = dict(got, dynamic=True)
            return got
        finally:
            with self.lock:
                self._tok_waits.pop(qid, None)

    def ask_detect(self, path, model, timeout=120):
        """修臉的找臉：交給 worker 跑 YOLO。生成中要等這批做完才輪得到，所以等久一點"""
        return self._ask({"cmd": "detect", "path": str(path), "model": str(model)}, timeout)

    def ask_rembg(self, path, model, out, arch="isnet", prep="anime", timeout=300):
        """去背：worker 算出遮罩存到 out。第一次要載模型，生成中也要等這批做完"""
        return self._ask({"cmd": "rembg", "path": str(path), "model": str(model), "out": str(out), "arch": arch, "prep": prep}, timeout)

    def ask_upscale(self, path, model, out, size, timeout=300):
        """高清化的只放大：worker 放大到 size 存到 out"""
        return self._ask({"cmd": "upscale", "path": str(path), "model": str(model or ""), "out": str(out), "size": list(size)}, timeout)

    def ask_embed(self, model, paths, out, timeout=600):
        """探索：算一批圖的特徵存到 out（.npy）。生成中要等這批做完才輪得到"""
        return self._ask({"cmd": "embed", "model": str(model), "paths": [str(p) for p in paths], "out": str(out)}, timeout)

    def ask_tag(self, model, paths, timeout=600):
        """打標台：一批圖的 WD14 標籤（每張 [[標籤, 分類, 機率], …]）。生成中要等這批做完才輪得到"""
        return self._ask({"cmd": "tag", "model": str(model), "paths": [str(p) for p in paths]}, timeout)

    def ask_caption(self, model, mmproj, paths, prompts, max_new=300, timeout=900):
        """打標台：一批圖的 JoyCaption 描述。第一次要載入約 6 秒；生成中要等這批做完才輪得到"""
        return self._ask({"cmd": "caption", "model": str(model), "mmproj": str(mmproj), "paths": [str(p) for p in paths],
                          "prompts": [str(p) for p in prompts], "max_new": int(max_new)}, timeout)

    def ask_embed_text(self, model, text, timeout=120):
        return self._ask({"cmd": "embed", "model": str(model), "text": str(text)}, timeout)

    def ask_cnprep(self, path, prep, files, out, w, h, resize="crop", low=100, high=200, typ="", timeout=300):
        """ControlNet 參考圖的預覽：worker 照生成的大小裁好、做前處理存到 out"""
        return self._ask({"cmd": "cnprep", "path": str(path), "prep": prep, "files": files or {}, "out": str(out),
                          "w": int(w), "h": int(h), "resize": resize, "low": int(low), "high": int(high), "type": typ}, timeout)

    def _ask(self, msg, timeout):
        """送一個要等回覆的指令（找臉、去背），用 qid 對回來"""
        from . import train
        if train.busy():
            return {"ready": False, "why": tr("srv.wc.training")}
        if not self.alive():
            try:
                self.start()
            except Exception:
                return {"ready": False, "why": tr("srv.wc.noStart")}
        with self.lock:
            self._tok_seq += 1
            qid = self._tok_seq
            ev = threading.Event()
            self._tok_waits[qid] = [ev, None]
        try:
            if not self.send({**msg, "qid": qid}):
                return {"ready": False, "why": tr("srv.wc.sendFail")}
            if not ev.wait(timeout):
                return {"ready": False, "why": tr("srv.wc.busyBatch")}
            with self.lock:
                return self._tok_waits[qid][1] or {"ready": False}
        finally:
            with self.lock:
                self._tok_waits.pop(qid, None)

    # 進度表的狀態是代碼（waiting、running、done、failed、aborted、cancelled、skipped），
    # 介面照語言顯示（web/js/generate.js 的 STATUS）；排程與成對圖也拿它比對，所以不能是會翻譯的文字
    def _mark(self, jid, status, **kw):
        # 只找目前這一批的，舊批次的同編號不能動到
        for r in reversed(self.table):
            if r["b"] == self.batch and r["i"] == jid:
                r["status"] = status
                r.update(kw)
                return

    def _note(self, msg, kind="info"):
        self.log.append({"t": time.strftime("%H:%M:%S"), "m": str(msg),
                         "k": kind})
        del self.log[:-200]

    def _read_err(self, proc):
        try:
            for line in proc.stderr:
                if not line.strip():
                    continue
                self.errbuf.append(line.rstrip())
                del self.errbuf[:-40]
                self.errshown += 1
                # 不截斷。錯誤訊息被切掉的話最關鍵的那句常常剛好不見，
                # 例如「…does not support custom sigmas schedules. Pl」
                self._note(line.rstrip(), stderr_kind(line))
        except Exception:
            pass

    def _read(self, proc):
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    self._note(line[:200])
                    continue
                t = m.get("type")
                if t in ("tokens", "detect", "rembg", "upscale", "embed", "cnprep", "tag", "caption"):
                    with self.lock:
                        slot = self._tok_waits.get(m.get("qid"))
                        if slot:
                            slot[1] = m
                            slot[0].set()
                    continue
                if t == "log":
                    self._note(m.get("msg", ""))
                elif t == "ready":
                    self.state["msg"] = tr("srv.wc.ready")
                    self._next()
                elif t == "progress":
                    self.state.update(step=m.get("step", 0),
                                      total=m.get("total", 0))
                    self._mark(m.get("id"), "running",
                               step=m.get("step", 0), total=m.get("total", 0))
                elif t == "done":
                    self._note(tr("srv.wc.done", file=Path(m['path']).name, seed=m['seed'], secs=m['secs']), "ok")
                    self._mark(m.get("id"), "done", secs=m.get("secs"),
                               seed=m.get("seed"),
                               file=Path(m["path"]).name, path=str(m["path"]))
                    for r in self.table:
                        if r["b"] == self.batch and r["i"] == m.get("id"):
                            r["step"] = r["total"]
                    self.state["last"] = m.get("path")
                    self._next()
                elif t == "skip":
                    # 批次修臉沒找到臉：這張不做，接下一張
                    self._note(tr("srv.wc.skipped", msg=m.get('msg', '')))
                    self._mark(m.get("id"), "skipped")
                    self._next()
                elif t == "error":
                    self._note(m.get("msg", tr("srv.wc.unknownErr")), "err")
                    self._mark(m.get("id"), "failed")
                    self.state.update(running=False, msg=m.get("msg", ""))
                    self._next()
        except Exception:
            pass
        code = None
        try:
            code = proc.wait(timeout=3)
        except Exception:
            pass
        with self.lock:
            if self.proc is not proc:
                # 已經停掉、或換了新的工作行程：佇列、進度表、狀態都是新的那個的，不能動
                self._note(tr("srv.wc.oldExited", code=code))
                return
            self.state.update(alive=False, running=False, step=0, total=0)
            if self._abandon():
                self.state["msg"] = tr("srv.wc.crashed")
        # 結束時補上還沒顯示的 stderr，不然只會看到一句「已結束」
        buf = getattr(self, "errbuf", [])
        for l in buf[getattr(self, "errshown", 0):][-6:]:
            self._note(l, "err")
        self._note(tr("srv.wc.exited", code=code), "err")

    # -- 佇列 --
    def _left(self):
        return len(self.jobs) + sum(len(js) for _, js in self.pending)

    def _next(self):
        with self.lock:
            if not self.jobs and self.pending:
                # 這組做完了，換下一組的模型（ready 之後會再回到這裡接第一個）
                cfg, self.jobs = self.pending.pop(0)
                self.state.update(running=True, queue=self._left(), step=0, total=0, msg=tr("srv.wc.switching"))
                job = {"cmd": "load", **cfg}
            elif not self.jobs:
                self.state.update(running=False, step=0, total=0, queue=0,
                                  msg=tr("srv.wc.allDone"))
                return
            else:
                job = self.jobs.pop(0)
                self._mark(job["id"], "running", step=0)
                self.state.update(running=True, queue=self._left(),
                                  cur=job["id"],
                                  msg=tr("srv.wc.running", n=self._left()))
        self.send(job)

    def submit(self, cfg, jobs, more=None):
        """送出一批。more：之後要接著做的 [(載入設定, 工作), …]（批次精修：原圖的模型不一樣時）。
        工作編號整批要不一樣（進度表靠它對）"""
        self.start()
        more = [(c, list(js)) for c, js in (more or []) if js]
        with self.lock:
            if self.state.get("running") or self.jobs or self.pending:
                return False
            self.jobs = list(jobs)
            self.pending = more
            self.batch += 1
            for j in list(jobs) + [j for _, js in more for j in js]:
                self.seq += 1
                self.table.append(
                    {"i": j["id"], "b": self.batch, "n": self.seq,
                     "prompt": j["prompt"][:160],
                     # 不是生成台送的（成對圖輔助，server/pairmaker.py）：生成頁的「生成後送去比較」不要拿去開比較組
                     **({"kind": j["kind"]} if j.get("kind") else {}),
                     "seed": j["seed"], "status": "waiting",
                     "step": 0, "total": int(j.get("steps") or 0),
                     "secs": None, "file": None})
            del self.table[:-self.TABLE_MAX]
            self.state.update(running=True, queue=self._left(),
                              step=0, total=0, msg=tr("srv.wc.loading"))
        if self.send({"cmd": "load", **cfg}):  # ready 之後會自動接第一個
            return True
        with self.lock:
            for j in self.jobs + [j for _, js in self.pending for j in js]:
                self._mark(j["id"], "aborted")
            self.jobs = []
            self.pending = []
            self.state.update(running=False, queue=0, msg=tr("srv.wc.noCmd"))
        return False

    def cancel(self):
        with self.lock:
            rest = self.jobs + [j for _, js in self.pending for j in js]
            n = len(rest)
            for j in rest:
                self._mark(j["id"], "cancelled")
            self.jobs = []
            self.pending = []
        self.state["msg"] = tr("srv.wc.cancelled", n=n)
        return n


WORKER = Worker()
