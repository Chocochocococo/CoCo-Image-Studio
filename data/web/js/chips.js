// 標籤模式與自動完成。
//
// 標籤模式只是提示詞的另一種畫法：真正存的還是那段文字（read / write），
// 所以隨時可以切回純文字，兩邊不會不同步。解析在 prompttext.js。
// 自動完成兩種模式都有：純文字模式補游標前的那個字，標籤模式補新增欄。

import {post, get} from "./api.js";
import {h, zoom} from "./dom.js";
import {lang, t} from "./i18n.js";
import {fmtWeight, loraTags, setLoraWeight} from "./options.js";
import {cartTotal, chipText, insertTag, joinPrompt, offOf, parseChip, parsePrompt, syntaxAt, tagOut, tokenAt, withOff} from "./prompttext.js";

// 標籤庫的分類 → 顏色組（CSS 的 .chip[data-c]）。沒列到的（uncategorized）不上色
const CAT_GROUP = {
  subject: "subject", species: "subject", age: "subject", profession: "subject",
  hair: "hair", eyes: "eyes", appearance: "body",
  clothing: "wear", accessory: "wear",
  expression: "face", mood: "face",
  pose: "pose", action: "pose",
  scene: "scene", lighting: "scene", weather_time: "scene", composition: "scene", effect: "scene", object: "scene",
  quality: "meta", meta: "meta", negative: "meta",
  artist: "who", character: "who", copyright: "who", lore: "who", contributor: "who",
};
// 顏色組的名稱（滑過標籤時顯示）；寫成字面上的鍵，測試才找得到
const GROUP_LABEL = {subject: "chip.cat.subject", hair: "chip.cat.hair", eyes: "chip.cat.eyes", body: "chip.cat.body",
                     wear: "chip.cat.wear", face: "chip.cat.face", pose: "chip.cat.pose", scene: "chip.cat.scene",
                     meta: "chip.cat.meta", who: "chip.cat.who"};
const groupOf = cat => CAT_GROUP[cat] || "";

// 寫法 → {name, cat, zh}；null＝查過、標籤庫沒有
const INFO = new Map();
// 查詢中的：寫法 → 那次查詢。別的畫面要同一個字就等同一個、回來一樣補畫（R67：以前用 null 佔位，
// 第二個畫面當成查過、不補畫；查失敗也一直留著 null，重新整理之前都不再查）
const ASKING = new Map();

// 回傳 names 裡有沒有新查到的（有才需要重畫）
async function lookup(names) {
  const mine = [...new Set(names)].filter(n => n && !INFO.has(n));
  const want = mine.filter(n => !ASKING.has(n));
  if (want.length) {
    const ask = post("/api/tags/lookup", {names: want}).then(d => {
      for (const [k, v] of Object.entries(d.items)) INFO.set(k, v);
      for (const n of want) if (!INFO.has(n)) INFO.set(n, null);
    }).catch(() => {})        // 連不上、或回錯誤（沒有 items）：不記，下次再問
      .finally(() => want.forEach(n => ASKING.delete(n)));
    for (const n of want) ASKING.set(n, ask);
  }
  await Promise.all(new Set(mine.map(n => ASKING.get(n))));
  return mine.some(n => INFO.get(n));
}

// 標籤 → 顏色組（重組頁拆提示詞用）：先把沒查過的一次查完，回傳查表函式；查不到的是 ""
export async function tagGrouper(names) {
  await lookup(names);
  return name => groupOf(INFO.get(name)?.cat);
}

const showZh = () => lang().startsWith("zh");

// ── 唯讀的標籤（看大圖的提示詞，使用者要求 2026-10-16）──
// 跟標籤模式一樣的顏色、權重與中文翻譯，但不能改：wrap(el, text) 由呼叫端決定點了做什麼（看大圖是複製那個標籤）。
// 標籤庫的翻譯查回來再補畫一次
export function chipView(text, wrap) {
  const list = parsePrompt(text);
  const box = h("div.vchips");
  const one = c => {
    const info = c.kind === "tag" ? INFO.get(c.text) : null;
    let label = c.text, w = c.kind === "tag" ? c.weight : null;
    if (c.kind === "lora") { const [l] = loraTags(c.text); if (l) { label = l.name; w = l.weight; } }
    const shown = w != null ? Math.round(w * 100) / 100 : 1;
    return wrap(h("span.chip", {"data-kind": c.kind, "data-c": groupOf(info?.cat) || null},
      c.kind === "lora" ? h("span.lk", {text: "LoRA"}) : null,
      h("span.t", {text: label || t("chip.none")}),
      showZh() && info?.zh ? h("span.zh", {text: info.zh}) : null,
      Math.abs(shown - 1) >= 0.005 ? h("span.w", {text: fmtWeight(shown), "data-lo": shown < 1 ? "" : null}) : null), chipText(c));
  };
  const draw = () => box.replaceChildren(...list.map(c => c.opts
    ? h("span.grp", {"data-kind": c.kind}, h("span.gh", {}, h("span", {text: t(c.kind === "cart" ? "chip.cart" : "chip.rand", {n: c.opts.length})})),
        h("span.gb", {}, c.opts.filter(o => o.trim()).map(o => one(parseChip(o)))))
    : one(c)));
  draw();
  const names = list.flatMap(c => c.opts ? c.opts.map(parseChip) : [c]).filter(c => c.kind === "tag").map(c => c.text);
  lookup(names).then(found => { if (found && box.isConnected) draw(); });
  return box;
}
const clampW = w => Math.max(-2, Math.min(3, Math.round(w * 100) / 100));

// ── 標籤模式 ─────────────────────────────────────
export class ChipBox {
  // read()：目前的提示詞；write(text)：改好的提示詞（呼叫端負責存檔、更新 token 數）；
  // loraBad(name)：這個 LoRA 跟主模型的架構合不合（不合的標紅框，照樣可以生成）；
  // readOff()／writeOff(list)：暫時關掉的標籤 [{text, after}]（after＝它前面那個開著的標籤，切回標籤模式時放回原位）
  constructor({read, write, label, loraBad, readOff, writeOff}) {
    this.read = read;
    this.write = write;
    this.readOff = readOff || (() => []);
    this.writeOff = writeOff || null;
    this.loraBad = loraBad || (() => false);
    this.list = [];
    this.armed = false;               // 空白欄按一次 Backspace 先標記最後一個，再按才刪
    this.el = h("div.chipbox", {role: "group", "aria-label": label});
    this.input = h("input.chipin", {type: "text", spellcheck: "false", autocomplete: "off",
                                    "aria-label": t("chip.add"), placeholder: t("chip.add")});
    attachAC(this.input, {mode: "input", onPick: text => this.add(text)});
    this.input.addEventListener("keydown", e => this.onInputKey(e));
    this.input.addEventListener("paste", e => {
      const text = e.clipboardData?.getData("text") || "";
      if (!/[,\n]/.test(text)) return;          // 貼一整串的時候直接拆成標籤
      e.preventDefault();
      this.add(text);
    });
    this.el.addEventListener("click", e => { if (e.target === this.el) this.input.focus(); });
    this.dnd();
  }

  // 暫時關掉的標籤（使用者要求）：不在提示詞裡（純文字模式看不到、不會生成），另外記它跟在哪個開著的標籤後面；
  // 純文字模式改動很多、那個標籤不見了，就放到最後
  refresh() {
    this.list = withOff(parsePrompt(this.read()), this.readOff() || []);
    this.draw();
  }

  commit() {
    this.write(joinPrompt(this.list.filter(c => !c.off)));
    if (this.writeOff) this.writeOff(offOf(this.list));
    this.draw();
  }

  toggleOff(i) {
    const c = this.list[i];
    if (!c) return;
    c.off = !c.off;
    this.commit();
    this.focusChip(i, null);
    this.pin(i, null);
  }

  add(text) {
    const items = parsePrompt(text);
    if (!items.length) return;
    this.list.push(...items);
    this.input.value = "";
    this.commit();
    this.input.focus();
  }

  onInputKey(e) {
    if (e.defaultPrevented) return;
    if ((e.key === "Enter" && !e.ctrlKey && !e.metaKey) || e.key === ",") {
      if (!this.input.value.trim()) return;
      e.preventDefault();
      this.add(this.input.value);
    } else if (e.key === "Backspace" && !this.input.value && this.list.length) {
      e.preventDefault();
      if (this.armed) { this.armed = false; this.list.pop(); this.commit(); this.input.focus(); }
      else { this.armed = true; this.top().pop()?.classList.add("armed"); }
    } else if (this.armed) {
      this.armed = false;
      this.el.querySelector(".armed")?.classList.remove("armed");
    }
  }

  draw() {
    this.armed = false;
    this.el.removeAttribute("data-pinned");      // 重畫後沒有被釘住的標籤了（要釘的由 weigh 重新釘）
    const keepFocus = document.activeElement === this.input;
    const {parts, total} = cartTotal(this.list);
    this.el.replaceChildren(...[
      ...this.list.map((c, i) => c.opts ? this.group(c, i) : this.chip(c, i)),
      this.input,
      parts.length ? h("span.chipsum", {text: parts.length > 1
        ? t("chip.sum", {parts: parts.join(" × "), n: total.toLocaleString(lang())})
        : t("chip.sum1", {n: total.toLocaleString(lang())})}) : null,
    ].filter(Boolean));
    if (keepFocus) this.input.focus();
    this.fill();
  }

  // 補上分類顏色與中文：只問沒查過的，有查到才重畫
  async fill() {
    const names = [];
    for (const c of this.list) {
      if (c.kind === "tag") names.push(c.text);
      for (const o of c.opts || []) { const x = parseChip(o); if (x.kind === "tag") names.push(x.text); }
    }
    if (await lookup(names)) this.draw();
  }

  // 一個標籤。j != null 表示它是第 i 組的第 j 個選項
  chip(c, i, j = null) {
    const sub = j != null;
    const info = c.kind === "tag" ? INFO.get(c.text) : null;
    const g = groupOf(info?.cat);
    let label = c.text, w = c.kind === "tag" ? c.weight : null, bad = false;
    if (c.kind === "lora") {
      const [l] = loraTags(c.text);
      if (l) { label = l.name; w = l.weight; bad = this.loraBad(l.name); }
    }
    const shown = w != null ? Math.round(w * 100) / 100 : 1;
    const tip = [bad ? t("gen.slot.loraBadOne") : "",
                 c.off ? t("chip.offNote") : "",
                 c.kind === "tag" && info && info.name !== c.text.replace(/ /g, "_").toLowerCase() ? `→ ${info.name}` : "",
                 g ? t(GROUP_LABEL[g]) : "", t(`chip.kind.${c.kind}`), t("chip.editHint")].filter(Boolean).join(" · ");
    const el = h("span.chip", {"data-kind": c.kind, "data-c": g || null, "data-i": i, "data-j": sub ? j : null, "data-bad": bad ? "" : null,
                               "data-off": c.off ? "" : null,
                               draggable: sub ? null : "true", tabindex: "0", title: tip},
      c.kind === "lora" ? h("span.lk", {text: "LoRA"}) : null,
      h("span.t", {text: label || t("chip.none")}),
      showZh() && info?.zh ? h("span.zh", {text: info.zh}) : null,
      Math.abs(shown - 1) >= 0.005 ? h("span.w", {text: fmtWeight(shown), "data-lo": shown < 1 ? "" : null}) : null,
      // 按鈕滑過才浮在標籤上方，不佔位置也不蓋住標籤——不然每個標籤都變寬，滑過時整排還會跳動
      h("span.ctl", {},
        w != null ? [
          h("button.cb", {type: "button", tabindex: "-1", "aria-label": t("chip.less"), title: t("chip.less"), text: "−", on: {click: () => this.weigh(i, j, -0.1)}}),
          h("button.cb", {type: "button", tabindex: "-1", "aria-label": t("chip.more"), title: t("chip.more"), text: "+", on: {click: () => this.weigh(i, j, 0.1)}}),
        ] : null,
        sub ? null : this.offBtn(c, i),
        h("button.cb.x", {type: "button", tabindex: "-1", "aria-label": t("chip.remove"), title: t("chip.remove"), text: "×", on: {click: () => this.remove(i, j)}})));
    el.addEventListener("dblclick", e => { e.stopPropagation(); this.edit(el, i, j); });
    el.addEventListener("keydown", e => this.chipKey(e, el, i, j));
    el.addEventListener("mouseenter", () => this.place(el));
    el.addEventListener("focus", () => this.place(el));
    return el;
  }

  // ⊘：暫時關掉／打開（使用者選的：點兩下維持改字）
  offBtn(c, i) {
    return h("button.cb.off", {type: "button", tabindex: "-1", "aria-pressed": String(!!c.off),
                               "aria-label": t(c.off ? "chip.on" : "chip.off"), title: t(c.off ? "chip.on" : "chip.off"), text: "⊘",
                               on: {click: () => this.toggleOff(i)}});
  }

  // 工具列預設浮在標籤上方；上面剩的空間放不下（第一排、或捲到一半）就改浮在下方
  place(el) {
    const room = el.getBoundingClientRect().top - this.el.getBoundingClientRect().top;
    el.toggleAttribute("data-flip", room < 32);
  }

  group(c, i) {
    const body = h("span.gb", {}, c.opts.map((o, j) => o.trim()
      ? this.chip(parseChip(o), i, j)
      : h("span.chip", {"data-kind": "none", "data-i": i, "data-j": j, tabindex: "0", title: t("chip.editHint")},
          h("span.t", {text: t("chip.none")}),
          h("span.ctl", {}, h("button.cb.x", {type: "button", tabindex: "-1", "aria-label": t("chip.remove"), title: t("chip.remove"),
                                              text: "×", on: {click: () => this.remove(i, j)}})))));
    // 空選項也可以點兩下改
    body.querySelectorAll('.chip[data-kind="none"]').forEach(el => {
      const j = Number(el.dataset.j);
      el.addEventListener("mouseenter", () => this.place(el));
      el.addEventListener("dblclick", e => { e.stopPropagation(); this.edit(el, i, j); });
      el.addEventListener("keydown", e => this.chipKey(e, el, i, j));
    });
    return h("span.grp", {"data-kind": c.kind, "data-i": i, draggable: "true", "data-off": c.off ? "" : null,
                          title: t(c.kind === "cart" ? "chip.cartHint" : "chip.randHint")},
      h("span.gh", {},
        h("span", {text: t(c.kind === "cart" ? "chip.cart" : "chip.rand", {n: c.opts.length})}),
        h("button.cb", {type: "button", "aria-label": t("chip.addOpt"), title: t("chip.addOpt"), text: "+",
                        on: {click: () => this.addOpt(body, i)}}),
        this.offBtn(c, i),
        h("button.cb.x", {type: "button", "aria-label": t("chip.removeGroup"), title: t("chip.removeGroup"), text: "×",
                          on: {click: () => this.remove(i)}})),
      body);
  }

  // 最外層的標籤與組（不含新增欄與總數）
  top() { return [...this.el.children].filter(x => x.dataset.i != null); }

  item(i, j) {
    if (j == null) return this.list[i];
    const g = this.list[i];
    return g?.opts ? parseChip(g.opts[j] || "") : null;
  }

  put(i, j, c) {
    if (j == null) this.list[i] = c;
    else this.list[i].opts[j] = chipText(c);
  }

  weigh(i, j, d) {
    const c = this.item(i, j);
    if (!c) return;
    if (c.kind === "tag") c.weight = clampW(c.weight + d);
    else if (c.kind === "lora") {
      const [l] = loraTags(c.text);
      if (!l) return;
      c.text = setLoraWeight(c.text, l.name, clampW(l.weight + d));
    } else return;
    this.put(i, j, c);
    this.commit();
    this.focusChip(i, j);
    this.pin(i, j);
  }

  // 調權重之後整排重畫，工具列原本只在滑過時出現：游標底下變成別的標籤、跳出它的按鈕，要一直移回去（使用者回報）。
  // 剛調的那個標籤釘住工具列、別的標籤先不跳；滑鼠離開它（含工具列）才放開
  pin(i, j) {
    const el = j == null ? this.top()[i] : this.el.querySelector(`.chip[data-i="${i}"][data-j="${j}"]`);
    if (!el) return;
    el.setAttribute("data-pin", "");
    this.el.setAttribute("data-pinned", "");
    el.addEventListener("mouseleave", () => this.unpin(), {once: true});
  }

  unpin() {
    this.el.removeAttribute("data-pinned");
    this.el.querySelector(".chip[data-pin]")?.removeAttribute("data-pin");
  }

  remove(i, j = null) {
    if (j == null) this.list.splice(i, 1);
    else {
      const g = this.list[i];
      if (!g?.opts) return;
      g.opts.splice(j, 1);
      // 只剩一個選項就不算「多選一」了，攤回一般標籤
      if (g.opts.length <= 1) {
        const rest = (g.opts[0] || "").trim();
        if (rest) this.list[i] = parseChip(rest);
        else this.list.splice(i, 1);
      }
    }
    this.commit();
    this.input.focus();
  }

  move(i, to) {
    if (to < 0 || to >= this.list.length || to === i) return;
    const [m] = this.list.splice(i, 1);
    this.list.splice(to, 0, m);
    this.commit();
    this.focusChip(to, null);
  }

  focusChip(i, j) {
    (j == null ? this.top()[i] : this.el.querySelector(`.chip[data-i="${i}"][data-j="${j}"]`))?.focus();
  }

  // 標籤上的鍵盤：Enter／F2 改字、Delete 刪、+／- 調權重、Alt＋←→ 搬位置
  chipKey(e, el, i, j) {
    if (e.target !== el) return;
    if (e.key === "Enter" || e.key === "F2") { e.preventDefault(); this.edit(el, i, j); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); this.remove(i, j); }
    else if (e.key === "+" || e.key === "=") { e.preventDefault(); this.weigh(i, j, 0.1); }
    else if (e.key === "-") { e.preventDefault(); this.weigh(i, j, -0.1); }
    else if (e.altKey && j == null && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      this.move(i, i + (e.key === "ArrowLeft" ? -1 : 1));
    }
  }

  // 點兩下改原始文字：動態語法、LoRA、權重都能直接改字串，最實在
  edit(el, i, j) {
    const c = j == null ? this.list[i] : null;
    const raw = j == null ? chipText(c) : this.list[i].opts[j];
    const box = h("input.chipedit", {type: "text", value: raw, spellcheck: "false", "aria-label": t("chip.edit"),
                                     size: String(Math.max(6, Math.min(60, raw.length + 2)))});
    let done = false;
    const finish = ok => {
      if (done) return;
      done = true;
      if (!ok) { this.draw(); return; }
      const v = box.value.trim();
      const items = parsePrompt(v);
      if (c?.off) for (const x of items) x.off = true;      // 關掉的改了字還是關著
      if (j == null) this.list.splice(i, 1, ...items);
      else this.list[i].opts[j] = v;          // 空白＝這組不用任何東西
      this.commit();
      this.input.focus();
    };
    box.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    box.addEventListener("blur", () => finish(true));
    el.replaceWith(box);
    box.focus();
    box.select();
  }

  addOpt(body, i) {
    const box = h("input.chipedit", {type: "text", size: "12", spellcheck: "false", placeholder: t("chip.optHint"),
                                     "aria-label": t("chip.addOpt")});
    let done = false;
    const finish = (ok, keepBlank) => {
      if (done) return;
      done = true;
      const v = box.value.trim();
      if (ok && (v || keepBlank)) { this.list[i].opts.push(v); this.commit(); }
      else this.draw();
    };
    box.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); finish(true, true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    box.addEventListener("blur", () => finish(true, false));
    body.append(box);
    box.focus();
  }

  // 拖曳排序：只排最外層（組裡的選項不單獨拖）
  dnd() {
    let from = null;
    // 從事件的元素往上找，找到最外層那一格（組裡的選項算整組）
    const target = e => this.top().find(x => x.contains(e.target)) || null;
    const clear = () => this.el.querySelectorAll(".over, .drag").forEach(x => x.classList.remove("over", "drag"));
    this.el.addEventListener("dragstart", e => {
      const it = target(e);
      if (!it || it.dataset.j != null) return;
      from = Number(it.dataset.i);
      it.classList.add("drag");
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", chipText(this.list[from]));
    });
    this.el.addEventListener("dragover", e => {
      const it = target(e);
      if (from == null || !it) return;
      e.preventDefault();
      this.el.querySelectorAll(".over").forEach(x => x.classList.remove("over"));
      it.classList.add("over");
    });
    this.el.addEventListener("drop", e => {
      const it = target(e);
      if (from == null || !it) return;
      e.preventDefault();
      const to = Number(it.dataset.i), f = from;
      from = null;
      clear();
      this.move(f, to);
    });
    this.el.addEventListener("dragend", () => { from = null; clear(); });
  }
}

// ── 自動完成 ─────────────────────────────────────
// mode "input"：整格就是關鍵字（標籤模式的新增欄）；選到就交給 onPick。
// mode "text"：純文字框，關鍵字是游標前那個字；選到就原地換掉。
const AC = {box: null, field: null, items: [], sel: 0, seq: 0, timer: null, tok: null, opts: null};

function acClose() {
  AC.box?.remove();
  AC.box = null;
  AC.items = [];
  AC.field?.removeAttribute("aria-activedescendant");
  AC.field?.setAttribute("aria-expanded", "false");
}

// 純文字框：游標所在的那個字（取詞與插入在 prompttext.js 的 tokenAt／insertTag）
const tokenOf = ta => (ta.selectionStart === ta.selectionEnd ? tokenAt(ta.value, ta.selectionStart) : null);

// 游標在文字框裡的座標（鏡像 div 量出來，單位是放大前的像素）
function caretXY(ta, pos) {
  const cs = getComputedStyle(ta);
  const m = document.createElement("div");
  for (const p of ["boxSizing", "width", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "borderTopWidth",
                   "borderLeftWidth", "borderRightWidth", "fontFamily", "fontSize", "fontWeight", "letterSpacing",
                   "lineHeight", "wordSpacing", "tabSize"]) m.style[p] = cs[p];
  Object.assign(m.style, {position: "absolute", visibility: "hidden", whiteSpace: "pre-wrap", overflowWrap: "break-word",
                          top: "0", left: "-9999px", borderStyle: "solid", borderColor: "transparent"});
  m.textContent = ta.value.slice(0, pos);
  const s = document.createElement("span");
  s.textContent = "​";
  m.append(s);
  document.body.append(m);
  const out = {x: s.offsetLeft, y: s.offsetTop - ta.scrollTop, lh: s.offsetHeight};
  m.remove();
  return out;
}

function acPlace() {
  const z = zoom(), r = AC.field.getBoundingClientRect();
  let x = r.left / z, y = r.bottom / z + 4;
  if (AC.opts.mode === "text") {
    const c = caretXY(AC.field, AC.tok.start);
    x = r.left / z + c.x;
    y = r.top / z + Math.min(c.y + c.lh, AC.field.clientHeight) + 4;
  }
  const W = innerWidth / z, H = innerHeight / z, bw = AC.box.offsetWidth, bh = AC.box.offsetHeight;
  if (y + bh > H - 8) y = Math.max(8, (AC.opts.mode === "text" ? r.top / z : r.top / z) - bh - 4);
  AC.box.style.left = `${Math.max(8, Math.min(x, W - bw - 8))}px`;
  AC.box.style.top = `${y}px`;
}

const fmtPosts = n => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n || 0);

function acShow(items) {
  acClose();
  if (!items.length) return;
  AC.items = items;
  AC.sel = 0;
  AC.box = h("div.acbox", {role: "listbox", id: "acList", "aria-label": t("chip.suggest")},
    items.map((x, k) => h("div.aci", {role: "option", id: `acOpt${k}`, "aria-selected": String(k === 0),
                                      "data-syntax": x.syntax || null,
                                      on: {mousedown: e => { e.preventDefault(); acPick(k); },
                                           mousemove: () => acSel(k)}},
      ...(x.syntax ? synRow(x, k) : tagRow(x)))));
  document.body.append(AC.box);
  AC.field.setAttribute("aria-expanded", "true");
  AC.field.setAttribute("aria-activedescendant", "acOpt0");
  acPlace();
  for (const x of items) {           // 順便記下分類，加成標籤時顏色馬上有
    if (x.syntax) continue;
    const k = tagOut(x.name);
    if (!INFO.get(k)) INFO.set(k, {name: x.name, cat: x.cat || "", zh: x.zh || ""});
  }
}

// 萬用字元、變數的一列：名稱、幾行或內容；萬用字元多一顆「每行一張」（[[ ]]，Shift＋Enter 也是）
function synRow(x, k) {
  if (x.syntax === "none") return [h("span.z", {text: x.name})];
  return [h("span.nm", {text: x.syntax === "var" ? `\${${x.name}}` : `__${x.name}__`}),
          h("span.z", {text: x.info || ""}),
          h("span.sp"),
          x.each ? h("button.acx", {type: "button", tabindex: "-1", text: t("chip.eachLine"), title: t("chip.eachLineHint"),
                                    on: {mousedown: e => { e.preventDefault(); e.stopPropagation(); acPick(k, true); }}}) : null];
}

function tagRow(x) {
  return [
      h("span.dot", {"data-c": groupOf(x.cat) || null}),
      x.fav ? h("span.star", {text: "★", title: t("chip.fav")}) : null,
      h("span.nm", {text: tagOut(x.name)}),
      showZh() && x.zh ? h("span.z", {text: x.zh}) : null,
      x.via ? h("span.z", {text: `← ${x.via.replace(/_/g, " ")}`}) : null,
      h("span.sp"),
      groupOf(x.cat) ? h("span.c", {text: t(GROUP_LABEL[groupOf(x.cat)])}) : null,
      x.mine ? h("span.p.mine", {text: t("chip.mine"), title: t("chip.mineHint")}) : h("span.p", {text: fmtPosts(x.posts)})];
}

function acSel(k) {
  if (!AC.box) return;
  AC.sel = (k + AC.items.length) % AC.items.length;
  [...AC.box.children].forEach((el, n) => el.setAttribute("aria-selected", String(n === AC.sel)));
  AC.box.children[AC.sel].scrollIntoView({block: "nearest"});
  AC.field.setAttribute("aria-activedescendant", `acOpt${AC.sel}`);
}

function acPick(k, each = false) {
  const x = AC.items[k];
  if (!x || x.syntax === "none") return;
  const text = x.syntax ? (each && x.each ? x.each : x.text) : tagOut(x.name), f = AC.field, opts = AC.opts, tok = AC.tok;
  acClose();
  if (opts.mode === "input") { opts.onPick(text); return; }
  const out = insertTag(f.value, x.syntax ? tok : (tokenOf(f) || tok), text);
  f.value = out.value;
  f.setSelectionRange(out.caret, out.caret);
  f.dispatchEvent(new Event("input", {bubbles: true}));
  acClose();                         // input 事件會再觸發一次查詢，這裡先關掉
  clearTimeout(AC.timer);
}

// 萬用字元與變數的清單（打 __ 或 $ 時列出來）：幾秒內重用，語法視窗改了下次就讀到
const SYN = {at: 0, wild: [], vars: []};
async function synLists() {
  if (Date.now() - SYN.at < 5000) return SYN;
  const [w, v] = await Promise.all([get("/api/wildcards").catch(() => ({})), get("/api/vars").catch(() => ({}))]);
  Object.assign(SYN, {at: Date.now(), wild: w.files || [], vars: v.vars || []});
  return SYN;
}

async function synQuery(field, opts, syn) {
  const seq = ++AC.seq;
  const L = await synLists();
  if (seq !== AC.seq || document.activeElement !== field) return;
  const q = syn.q.toLowerCase();
  let items = syn.kind === "wild"
    ? L.wild.filter(x => x.name.toLowerCase().includes(q)).slice(0, 12)
      .map(x => ({syntax: "wild", name: x.name, info: t("chip.wildLines", {n: x.lines}), text: `__${x.name}__`, each: `[[__${x.name}__]]`}))
    : L.vars.filter(x => x.name.toLowerCase().includes(q)).slice(0, 12)
      .map(x => ({syntax: "var", name: x.name, info: x.value, text: `\${${x.name}}`}));
  if (!items.length) items = [{syntax: "none", name: t(syn.kind === "wild" ? "chip.noWild" : "chip.noVar")}];
  AC.field = field;
  AC.opts = opts;
  AC.tok = syn;
  acShow(items);
}

async function acQuery(field, opts) {
  // 打 __ 或 $：列萬用字元、變數（使用者要求：不用再開語法視窗記寫法）
  const at = field.selectionStart, one = field.selectionStart === field.selectionEnd;
  const syn = one ? syntaxAt(field.value, at) : null;
  if (syn) { synQuery(field, opts, syn); return; }
  const tok = opts.mode === "text" ? tokenOf(field) : {q: field.value.trim()};
  if (!tok || !tok.q) { acClose(); return; }
  const seq = ++AC.seq;
  const d = await get(`/api/tags?q=${encodeURIComponent(tok.q)}&limit=12`).catch(() => ({items: []}));
  if (seq !== AC.seq || document.activeElement !== field) return;     // 已經有更新的查詢，或游標離開了
  AC.field = field;
  AC.opts = opts;
  AC.tok = tok;
  // 只有一個結果而且就是打好的那個字，不必再問
  const items = (d.items || []).filter(x => !(d.items.length === 1 && tagOut(x.name).toLowerCase() === tok.q.toLowerCase()));
  acShow(items);
}

export function attachAC(field, opts) {
  field.setAttribute("aria-autocomplete", "list");
  field.setAttribute("aria-controls", "acList");
  field.setAttribute("aria-expanded", "false");
  const later = () => {
    clearTimeout(AC.timer);
    AC.timer = setTimeout(() => acQuery(field, opts), 120);
  };
  // 注音、拼音選字中（框裡是「ㄏㄨㄥˊ」這種還沒選好的字）不查，選好字才查
  field.addEventListener("input", e => { if (!e.isComposing) later(); });
  field.addEventListener("compositionend", later);
  field.addEventListener("keydown", e => {
    if (e.isComposing || e.keyCode === 229) return;     // 選字時的 Enter、方向鍵是輸入法的
    if (!AC.box || AC.field !== field) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); acSel(AC.sel + (e.key === "ArrowDown" ? 1 : -1)); }
    else if ((e.key === "Enter" || e.key === "Tab") && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      acPick(AC.sel);
    } else if (e.key === "Enter" && e.shiftKey && AC.items[AC.sel]?.each) {
      e.preventDefault();
      acPick(AC.sel, true);                    // 萬用字元：Shift＋Enter＝每行一張
    } else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); acClose(); }
  });
  field.addEventListener("blur", () => setTimeout(() => { if (AC.field === field && document.activeElement !== field) acClose(); }, 120));
  field.addEventListener("scroll", () => { if (AC.field === field) acClose(); });
}

addEventListener("resize", acClose);
