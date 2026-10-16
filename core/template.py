# -*- coding: utf-8 -*-
"""
template.py —— 提示詞模板引擎（階段 B）

把一段模板展開成多組提示詞。純文字處理，不需要 GPU。

語法
────────────────────────────────────────────────
  [[A|B|C]]        笛卡兒展開。每個選項各產生一組提示詞。
                   多個 [[ ]] 會相乘 —— 立繪組就是靠這個。
                   **裡面沒有 | 就不是展開**，例如 [[tag]] 會原樣留給
                   提示詞的 A1111 減弱語法（權重 0.826）。

  {A|B|C}          隨機擇一
  {2::A|B|3::C}    加權隨機（A 權重 2、B 權重 1、C 權重 3）
  {A|B|}           留空的選項代表「有機會什麼都不放」

  __chara/my_oc__       讀 wildcards/chara/my_oc.txt，隨機抽一行
  [[__chara/all__]]     對該檔案的每一行各展開一組

  {set:color=red|blue}  抽一個值存起來，本身不輸出
  {get:color}           取用先前抽到的值（同一組提示詞內保證一致）

  巢狀可用：{A|{B|C}}、萬用字元檔內也能寫上述語法

範例
────────────────────────────────────────────────
  masterpiece, 1girl, {set:cloth=red|blue|white} {get:cloth} dress,
  [[standing|sitting|leaning]], [[smile|neutral|angry|surprised]],
  {simple background|detailed background}

  → 12 組（3 姿勢 × 4 表情），每組的服裝顏色前後一致
"""

import itertools
import random
import re
from pathlib import Path

import i18n

MAX_PASSES = 40          # 巢狀解析的上限，防呆用
DEFAULT_LIMIT = 2000     # 展開組數上限

_re_cart = re.compile(r"\[\[(.*?)\]\]", re.S)
_re_brace = re.compile(r"\{([^{}]*)\}")          # 最內層的 { }
_re_wild = re.compile(r"__([A-Za-z0-9_\-/\\. ]+?)__")
_re_weight = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*::(.*)$", re.S)
_re_hold = re.compile("\x01([^\x01]*)\x01")   # 尚未解析的 ${name} 暫存標記
_re_comment = re.compile(r"#[^\n]*")
_re_count = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?\$\$(.*)$", re.S)


# ---------------------------------------------------------------- 萬用字元
class Wildcards:
    """讀取 wildcards 資料夾裡的文字檔，結果會快取"""

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None
        self._cache = {}
        self.missing = set()

    def lines(self, name):
        if name in self._cache:
            return self._cache[name]
        out = []
        if self.root:
            for ext in (".txt", ""):
                f = (self.root / (name + ext)).resolve()
                try:
                    f.relative_to(self.root)
                except ValueError:
                    continue
                if f.is_file():
                    out = [l.strip() for l in
                           f.read_text(encoding="utf-8").splitlines()]
                    out = [l for l in out if l and not l.startswith("#")]
                    break
            else:
                self.missing.add(name)
        else:
            self.missing.add(name)
        self._cache[name] = out
        return out


def load_vars(path):
    """讀變數檔，格式為 `名稱 = 值`（值可含模板語法）

    變數與萬用字元是不同的東西：
      變數     —— 一個名稱對應一段固定內容，用 ${名稱} 取用
      萬用字元 —— 一個檔案裡有很多行選項，用 __檔名__ 隨機抽一行

    所以變數放在專案根目錄的 variables.txt，不放在 wildcards 資料夾。
    """
    out = {}
    if not path:
        return out
    f = Path(path)
    if f.is_dir():                      # 相容舊版：傳進來的是 wildcards 資料夾
        f = f / "_variables.txt"
    if not f.is_file():
        return out
    for raw in f.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


# ---------------------------------------------------------------- 選項拆解
def split_options(text):
    """依最外層的 | 拆開，忽略巢狀括號內的 |"""
    parts, buf, depth = [], [], 0
    i = 0
    while i < len(text):
        c = text[i]
        if text.startswith("[[", i):
            depth += 1
            buf.append("[[")
            i += 2
            continue
        if text.startswith("]]", i):
            depth -= 1
            buf.append("]]")
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def pick_many(body, rng):
    """處理 {2$$A|B|C}、{1-3$$A|B|C}、{2$$ and $$A|B|C}
    回傳字串；不是這個語法就回傳 None"""
    m = _re_count.match(body)
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    rest = m.group(3)

    sep = ", "
    if "$$" in rest:                       # 自訂分隔符
        sep_part, rest = rest.split("$$", 1)
        sep = sep_part

    opts = [o.strip() for o in split_options(rest)]
    n = rng.randint(min(lo, hi), max(lo, hi))
    n = min(n, len(opts))
    if n <= 0:
        return ""
    return sep.join(rng.sample(opts, n))


def weighted(options):
    """把 ['2::A', 'B'] 拆成 (權重, 內容) 清單"""
    out = []
    for o in options:
        m = _re_weight.match(o)
        if m:
            out.append((float(m.group(1)), m.group(2)))
        else:
            out.append((1.0, o))
    return out


# ---------------------------------------------------------------- 展開
def _cart_options(inner, wc):
    """笛卡兒括號內的選項。整段就是一個萬用字元時，展開該檔的每一行。"""
    inner = inner.strip()
    m = _re_wild.fullmatch(inner)
    if m:
        lines = wc.lines(m.group(1))
        return lines if lines else [inner]
    return [p.strip() for p in split_options(inner)]


def is_cartesian(inner):
    """[[ ]] 裡面的內容算不算笛卡兒展開

    這個判斷是為了跟 A1111 的減弱語法共存：`[[tag]]` 在 Forge 是
    「減弱兩次」（權重 0.826），但在這裡本來會被當成單一選項的展開
    吃掉，括號消失、減弱效果無聲不見。

    規則：裡面有 | 才算展開，或者整段是一個萬用字元（那是刻意支援的
    `[[__chara/all__]]` 寫法）。單純一個詞就讓給減弱語法——
    反正「只有一個選項的笛卡兒展開」本來就沒有作用。
    """
    s = inner.strip()
    if _re_wild.fullmatch(s):
        return True
    return len(split_options(s)) > 1


def cartesian(template, wc, limit=DEFAULT_LIMIT):
    """處理所有 [[ ]]，回傳 [(展開後的模板, {軸: 值}), ...]"""
    groups = []
    def grab(m):
        if not is_cartesian(m.group(1)):
            return m.group(0)      # 原樣留著，交給提示詞的減弱語法
        groups.append(_cart_options(m.group(1), wc))
        return f"\x00{len(groups) - 1}\x00"

    marked = _re_cart.sub(grab, template)
    if not groups:
        return [(template, {})]

    total = 1
    for g in groups:
        total *= max(1, len(g))
    if total > limit:
        raise ValueError(i18n.tr("eng.tpl.tooMany", n=total, limit=limit) + i18n.tr("eng.tpl.reduce"))

    out = []
    for combo in itertools.product(*groups):
        text = marked
        axes = {}
        for i, v in enumerate(combo):
            text = text.replace(f"\x00{i}\x00", v)
            axes[f"軸{i + 1}"] = v
        out.append((text, axes))
    return out


def resolve(text, rng, wc, preset_vars=None):
    """處理 { } 隨機、變數與萬用字元，由內而外反覆解析"""
    vars_ = dict(preset_vars or {})

    def dollars(text):
        """處理 ${name} 與 ${name=...}。用括號配對掃描，
        才能吃下 ${c={red|blue}} 這種巢狀寫法。
        還沒定義的 ${name} 原樣留著，等下一輪再解 —— 這樣
        定義寫在後面也不會失效。"""
        # 上一輪留下的暫存標記還原成 ${name}，這輪或許已經有定義了
        text = _re_hold.sub(lambda m: "${" + m.group(1) + "}", text)
        out, i = [], 0
        while i < len(text):
            j = text.find("${", i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            depth, k = 0, j + 1
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if k >= len(text):
                out.append(text[j:])
                break
            body = text[j + 2:k]
            if "=" in body:
                name, opts = body.split("=", 1)
                name = name.strip()
                opts = opts.strip()
                if opts.startswith("{") and opts.endswith("}"):
                    opts = opts[1:-1]
                picks = weighted(split_options(opts))
                val = rng.choices([p[1] for p in picks],
                                  weights=[p[0] for p in picks])[0].strip()
                # 定義裡引用其他變數時，先代換掉
                val = re.sub(
                    r"\$\{([A-Za-z0-9_\-]+)\}",
                    lambda mm: vars_.get(mm.group(1),
                                         "\x01" + mm.group(1) + "\x01"),
                    val)
                vars_[name] = val
                # 定義本身不輸出，取值要另外寫 ${name}
            else:
                name = body.strip()
                # 還沒定義就先包成標記，避免被 { } 的規則吃掉
                out.append(vars_[name] if name in vars_
                           else "\x01" + name + "\x01")
            i = k + 1
        return "".join(out)

    def brace(m):
        body = m.group(1)

        many = pick_many(body, rng)
        if many is not None:
            return many

        if body.startswith("set:"):
            body = body[4:]
            if "=" not in body:
                return ""
            name, opts = body.split("=", 1)
            picks = weighted(split_options(opts))
            val = rng.choices([p[1] for p in picks],
                              weights=[p[0] for p in picks])[0]
            vars_[name.strip()] = val.strip()
            return ""

        if body.startswith("get:"):
            return vars_.get(body[4:].strip(), "")

        picks = weighted(split_options(body))
        return rng.choices([p[1] for p in picks],
                           weights=[p[0] for p in picks])[0]

    def wild(m):
        lines = wc.lines(m.group(1))
        return rng.choice(lines) if lines else m.group(0)

    for _ in range(MAX_PASSES):
        new = _re_wild.sub(wild, text)
        new = dollars(new)
        new = _re_brace.sub(brace, new)
        if new == text:
            break
        text = new
    return _re_hold.sub("", text)        # 清掉始終沒有定義的


def tidy(text):
    """收尾：清掉註解、多餘的逗號、空白與換行"""
    text = _re_comment.sub("", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def expand(template, count=1, seed=None, wildcard_dir=None,
           limit=DEFAULT_LIMIT, lock_random=True, vars_file=None):
    """把模板展開成多組提示詞

    count       —— 沒有 [[ ]] 時要產生幾組隨機變體；
                   有 [[ ]] 時，每個組合各產生 count 組
    seed        —— 給定就可重現
    lock_random —— True（預設）：同一輪變體裡，所有笛卡兒組合共用同一組隨機值。
                   角色立繪組要的就是這個 —— 換姿勢換表情，但髮色服裝不變。
                   False：每個組合各自隨機，適合「大量不同角色」的批次。
    """
    wc = Wildcards(wildcard_dir)
    preset = load_vars(vars_file or wildcard_dir)
    combos = cartesian(template, wc, limit)

    total = len(combos) * max(1, count)
    if total > limit:
        raise ValueError(i18n.tr("eng.tpl.tooMany", n=total, limit=limit))

    base = random.randrange(1 << 30) if seed is None else int(seed)
    out = []
    for ci, (tmpl, axes) in enumerate(combos):
        for k in range(max(1, count)):
            # 鎖定時只看變體序號，所有組合抽到相同的值
            rng = random.Random(base + k * 7919
                                + (0 if lock_random else ci * 10007))
            out.append({
                "index": len(out),
                "prompt": tidy(resolve(tmpl, rng, wc, preset)),
                "axes": axes,
                "seed_base": base,
            })
    return {"items": out, "missing": sorted(wc.missing),
            "combos": len(combos), "count": max(1, count),
            "lock_random": lock_random, "vars": sorted(preset)}
