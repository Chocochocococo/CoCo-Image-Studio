// 圖庫牆：三種排法、捲到底自動載入、可選取。工作室右側與圖庫空間各用一個。
//
// 排法：
//   just   齊行：連續幾張組成一列，整列縮放到剛好填滿寬度。
//          順序嚴格由左到右、沒有空隙，代價是每列高度不同。
//   packed 瀑布流・緊密：每張放進目前最矮的那欄。排得最密，順序會稍微跳。
//   seq    瀑布流・依序：第 i 張固定放第 i % 欄數 欄。順序正確，底部參差。
// 都是自己排，不用 CSS columns：那是一欄填滿才換下一欄，最新的圖會全擠在
// 第一欄，而且每載入一批整面牆都會重排。

import {get, post} from "./api.js";
import {h, zoom} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {openPop, openedPop} from "./pop.js";

export const ORDERS = ["just", "packed", "seq"];
const GAP = 12;

// 高寬比，不必等圖載入就能排
const ratio = it => (it.width && it.height ? it.height / it.width : 1.4);

// 右下角的 ↑：所有的牆一起更新（捲動事件不冒泡，在捕獲階段收；負責捲動的卡片換頁時是新的，不必重綁）
// 每次捲動只讀兩個數字、改一個屬性，直接做（不等下一格畫面：分頁不在前景時 requestAnimationFrame 不跑）
const WALLS = new Set();
document.addEventListener("scroll", () => { for (const w of WALLS) w.syncTop(); }, {capture: true, passive: true});

export class Wall {
  // name：區分小視窗用；orderKey / sizeKey / zoomKey：存在 ui 偏好裡的鍵；
  // sizes：[["s", px], …]；query()：附加在 /api/list 後面的查詢字串；emptyKey：沒有圖時的字（鍵，或回傳鍵的函式）；
  // onOpen(item)：看大圖。可選取的牆（selectable: true）點一下是選取，雙擊、Enter 或角落的按鈕才看大圖；
  // selectable: "mod"（工作室的牆）點一下照樣看大圖，Ctrl／Shift＋點或角落的勾才是選取，
  // 已經有選取時點一下就是加選或取消
  constructor({name, ui, orderKey, sizeKey, zoomKey, sizes, query = () => "", selectable = false,
               emptyKey = "wall.empty", onSelect, onCount, onOpen}) {
    Object.assign(this, {name, ui, orderKey, sizeKey, zoomKey, sizes, query, selectable, emptyKey, onSelect, onCount, onOpen});
    WALLS.add(this);
    this.items = [];
    this.total = 0;
    this.offset = 0;
    this.loading = false;
    this.gen = 0;              // 每次重設加一，舊的請求回來時就知道要丟掉
    this.nodes = new Map();    // id → 圖塊，重新排版時搬動同一個元素，圖不用重新載入
    this.dev = null;           // 「顯影中」那格
    this.devRatio = 1.4;
    this.cols = {els: [], heights: [], i: 0, w: 0};
    this.scrollY = 0;          // 換頁前的捲動位置（saveScroll／restoreScroll）
    this.sel = [];             // 選取的 id，照點選的順序（紙膠帶上的號碼）
    this.anchor = null;        // Shift 連選的起點
    this.mod = selectable === "mod";

    this.box = h("div.mason", {role: selectable ? "listbox" : null, "aria-multiselectable": selectable ? "true" : null});
    this.status = h("p.dim");
    this.more = h("button.pill", {type: "button", hidden: true, "data-i18n": "wall.more", on: {click: () => this.loadMore()}});
    this.end = h("div.end", {}, this.status, this.more);
    // 回到頂端（使用者筆記）：捲下去一段才出現，固定在看得到的範圍右下角（sticky，不佔版面）。
    // 畫面裡放在 end 後面：w.box, w.end, w.top
    this.top = h("div.totop", {hidden: true},
      h("button", {type: "button", "data-i18n-aria": "wall.toTop", "data-i18n-title": "wall.toTop",
                   on: {click: () => this.toTop(true)}}, "↑"));
    // 捲到底自動載入；按鈕留著給鍵盤與不支援的瀏覽器
    new IntersectionObserver(es => { if (es[0].isIntersecting && this.total) this.loadMore(); }).observe(this.end);
    // 寬度改變時重排：齊行每次都要重算，瀑布流只有欄數變了才要
    new ResizeObserver(() => {
      if (!this.box.clientWidth || this.box.clientWidth === this.cols.w) return;
      if (this.order() === "just" || this.cols.els.length !== this.colCount()) this.layout();
      else this.cols.w = this.box.clientWidth;
    }).observe(this.box);
    this.box.addEventListener("click", e => {
      if (e.target.closest(".open")) { e.stopPropagation(); this.open(e); return; }
      if (e.target.closest(".pick")) { e.stopPropagation(); this.onClick(e); return; }
      if (!selectable) { this.open(e); return; }
      if (this.mod && !(e.ctrlKey || e.metaKey || e.shiftKey || this.sel.length)) { this.open(e); return; }
      this.onClick(e);
    });
    if (selectable && !this.mod) this.box.addEventListener("dblclick", e => this.open(e));
    this.box.addEventListener("keydown", e => {
      if (!e.target.matches("figure.im[data-id]")) return;
      if (e.key === "Enter") { e.preventDefault(); this.open(e); }
      else if (e.key === " " && selectable) { e.preventDefault(); this.onClick(e); }
    });
    hoverZoom(this);
  }

  itemOf(e) {
    const fig = e.target.closest("figure.im[data-id]");
    return fig ? this.items.find(i => i.id === Number(fig.dataset.id)) : null;
  }

  open(e) {
    const it = this.itemOf(e);
    if (it) { hideZoom(); this.onOpen?.(it); }
  }

  // ── 設定 ──
  order() { return ORDERS.includes(this.ui[this.orderKey]) ? this.ui[this.orderKey] : "just"; }
  tileSize() { return Number(this.ui[this.sizeKey]) || this.sizes[1][1]; }
  sizeName() {
    const px = this.tileSize();
    return this.sizes.reduce((a, b) => (Math.abs(b[1] - px) < Math.abs(a[1] - px) ? b : a))[0];
  }
  colCount() {
    return Math.max(1, Math.floor((this.box.clientWidth + GAP) / (this.tileSize() + GAP)));
  }

  viewButton() {
    return h("button.ghost", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "wall.view",
                              on: {click: e => openPop(`popView-${this.name}`, e.currentTarget, () => this.viewPop())}});
  }

  viewPop() {
    return [
      h("div.hd", {}, h("h3.title", {"data-i18n": "wall.view.title"})),
      h("div.row", {}, h("span.lbl", {"data-i18n": "wall.order"}),
        h("div.opts", {}, ORDERS.map(o => h("button.opt", {
          type: "button", "aria-pressed": String(this.order() === o), on: {click: () => this.setView({[this.orderKey]: o})}},
          h("b", {"data-i18n": `wall.order.${o}`}), h("span", {"data-i18n": `wall.order.${o}.hint`}))))),
      h("div.row", {}, h("span.lbl", {"data-i18n": "wall.size"}),
        h("div.seg", {}, this.sizes.map(([k, px]) => h("button", {
          type: "button", "aria-pressed": String(this.sizeName() === k), "data-i18n": `wall.size.${k}`,
          on: {click: () => this.setView({[this.sizeKey]: px})}})))),
      this.zoomKey ? h("div.row.swrow", {},
        h("button.switch", {type: "button", role: "switch", "aria-checked": String(!!this.ui[this.zoomKey]),
                            "aria-labelledby": `zoomLbl-${this.name}`,
                            on: {click: () => this.setView({[this.zoomKey]: !this.ui[this.zoomKey]})}}, h("i")),
        h("span", {}, h("b", {id: `zoomLbl-${this.name}`, "data-i18n": "wall.zoom"}),
          h("span.hint", {"data-i18n": "wall.zoom.hint"}))) : null];
  }

  setView(patch) {
    Object.assign(this.ui, patch);
    if (!(this.zoomKey in patch)) this.layout();
    const pop = openedPop(`popView-${this.name}`);
    if (pop) {
      pop.querySelectorAll(".opt").forEach((b, i) => b.setAttribute("aria-pressed", String(ORDERS[i] === this.order())));
      pop.querySelectorAll(".seg button").forEach((b, i) => b.setAttribute("aria-pressed", String(this.sizes[i][0] === this.sizeName())));
      pop.querySelector(".switch")?.setAttribute("aria-checked", String(!!this.ui[this.zoomKey]));
    }
    post("/api/ui", patch).catch(() => { /* 沒存到只影響下次開啟 */ });
  }

  // ── 排版 ──
  entries() {
    const out = this.dev ? [[this.dev, this.devRatio]] : [];
    for (const it of this.items) out.push([this.nodeFor(it), ratio(it)]);
    return out;
  }

  place(node, r) {
    const c = this.cols;
    node.style.width = node.style.height = "";
    const i = this.order() === "seq" ? c.i % c.els.length : c.heights.indexOf(Math.min(...c.heights));
    c.i++;
    c.els[i].append(node);
    c.heights[i] += r + GAP / this.tileSize();
  }

  // 齊行：一張張加進目前這列，寬度夠了就把整列縮放到剛好填滿。
  // 加進這張會讓整列比目標矮時，比較「這張放進來、整列壓矮」和「這張留到下一列、
  // 這列拉高」哪個比較接近目標高度——不這樣做，一張很寬的圖會把整列壓得很扁。
  // 排法是固定的，所以載入更多時前面已經排滿的列不會變，只有最後一列會重排。
  justify(list) {
    const W = this.box.clientWidth, H = this.tileSize();
    const rows = [];
    let buf = [], sum = 0;
    for (const [node, r] of list) {
      // 太寬或太細長的圖（例如 10:1 的橫幅）當成最多 3:1 來排、縮圖裁中間
      const ar = Math.min(3, Math.max(1 / 3, 1 / r));
      const hWith = (W - GAP * buf.length) / (sum + ar);
      if (hWith > H) { buf.push([node, ar]); sum += ar; continue; }     // 還放得下
      const hWithout = buf.length ? (W - GAP * (buf.length - 1)) / sum : Infinity;
      if (hWithout - H < H - hWith && hWithout <= H * 1.6) {
        rows.push([buf, hWithout]);
        buf = [[node, ar]]; sum = ar;
      } else {
        buf.push([node, ar]);
        rows.push([buf, hWith]);
        buf = []; sum = 0;
      }
    }
    if (buf.length) rows.push([buf, H]);      // 最後一列不拉伸
    this.box.replaceChildren(...rows.map(([items, ht]) => h("div.jrow", {}, items.map(([node, ar]) => {
      node.style.width = `${ar * ht}px`;
      node.style.height = `${ht}px`;
      return node;
    }))));
  }

  layout() {
    const mode = this.order(), c = this.cols;
    this.box.classList.toggle("just", mode === "just");
    c.w = this.box.clientWidth;
    if (!c.w) return;                         // 還沒顯示在畫面上；顯示時 ResizeObserver 會再排
    if (mode === "just") { c.els = []; this.justify(this.entries()); return; }
    const n = this.colCount();
    c.els = Array.from({length: n}, () => h("div.col"));
    c.heights = Array(n).fill(0);
    c.i = 0;
    this.box.replaceChildren(...c.els);
    for (const [node, r] of this.entries()) this.place(node, r);
  }

  // 剛放回畫面上時呼叫：寬度跟上次排版時不同就重排。
  // 不只靠 ResizeObserver——牆不在畫面上時載入的圖量不到寬度、排不了，
  // 等它回到畫面上要馬上補排，不能等下一次通知（分頁在背景時通知會延後）。
  refit() {
    if (this.box.clientWidth && this.box.clientWidth !== this.cols.w) this.layout();
  }

  // 載入更多：瀑布流接著往下放，已經放好的不動；齊行整面重算（結果一樣，只有最後一列變）
  append(items) {
    const c = this.cols;
    if (this.order() !== "just" && c.els.length && c.els.length === this.colCount() && c.els[0].isConnected) {
      for (const it of items) this.place(this.nodeFor(it), ratio(it));
    } else this.layout();
  }

  nodeFor(it) {
    let n = this.nodes.get(it.id);
    if (!n) { n = this.tile(it); this.nodes.set(it.id, n); }
    return n;
  }

  tile(it) {
    const img = h("img", {src: `/api/thumb?id=${encodeURIComponent(it.id)}&v=${encodeURIComponent(it.v)}`,
                          alt: "", loading: "lazy", decoding: "async",
                          width: it.width || null, height: it.height || null});
    if (it.width && it.height) img.style.aspectRatio = `${it.width} / ${it.height}`;
    return h("figure.im", {title: it.filename || "", "data-id": it.id, tabindex: "0",
                           role: this.selectable ? "option" : "button",
                           "aria-selected": this.selectable ? "false" : null,
                           "aria-label": this.selectable ? null : t("wall.openAria", {name: it.filename || ""})},
      img,
      // 可選取的牆：點圖是選取，看大圖用角落這顆（滑過才出現）
      this.selectable && !this.mod ? h("span.open", {role: "button", "data-i18n-title": "wall.open", title: t("wall.open"), "aria-hidden": "true"}, "⤢") : null,
      // 工作室的牆：點圖是看大圖，選取用角落的勾（或 Ctrl／Shift＋點）
      this.mod ? h("span.pick", {role: "button", "data-i18n-title": "wall.pick", title: t("wall.pick"), "aria-hidden": "true"}, "✓") : null);
  }

  // ── 捲動位置（使用者筆記：切頁再回來，圖庫都跳回最上面）──
  // 捲的是牆外面那張卡片（寬螢幕）或整頁（窄螢幕）。切頁時整個畫面重建、卡片是新的，
  // 牆上的圖沿用，所以換頁前記下捲到哪，回來排好之後捲回去
  scroller() {
    for (let el = this.box.parentElement; el && el !== document.body; el = el.parentElement) {
      const oy = getComputedStyle(el).overflowY;
      if ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight) return el;
    }
    return document.scrollingElement;
  }

  saveScroll() {
    if (this.box.isConnected) this.scrollY = this.scroller()?.scrollTop || 0;
  }

  // refit()／layout() 是同步排好的，排完馬上捲回去（不等下一格畫面，免得先閃一下最上面）
  restoreScroll() {
    if (!this.scrollY || !this.box.isConnected) return;
    const sc = this.scroller();
    if (sc) sc.scrollTop = this.scrollY;
  }

  // 回到最上面（開始生成時看「顯影中」那格；右下角的 ↑）。不在畫面上的話，回來時就在最上面
  // 窄螢幕捲的是整頁（牆在生成台下面），只捲到牆的頂端、而且只往上捲。
  // 開始生成時直接跳，不用 smooth：新格子插在最前面時瀏覽器的捲動錨定會把平滑捲動推走；按 ↑ 的才平滑
  toTop(smooth = false) {
    this.scrollY = 0;
    if (!this.box.isConnected) return;
    const sc = this.scroller();
    const top = sc === document.scrollingElement
      ? Math.max(0, this.box.getBoundingClientRect().top + sc.scrollTop - 80) : 0;
    if (sc.scrollTop <= top) return;
    if (smooth) sc.scrollTo({top, behavior: "smooth"});
    else sc.scrollTop = top;
  }

  // 捲超過一個畫面高才出現 ↑
  syncTop() {
    if (!this.box.isConnected) return;
    const sc = this.scroller();
    const past = sc === document.scrollingElement ? -this.box.getBoundingClientRect().top : sc.scrollTop;
    this.top.hidden = !(past > Math.max(400, (sc.clientHeight || innerHeight) * 0.8));
  }

  // ── 載入 ──
  drawCount() {
    this.more.hidden = this.offset >= this.total;
    this.onCount?.(this.total);
  }

  // 牆底下那行字。不帶參數的（尚無圖片、載入中）掛 data-i18n，換語言時 render() 的 translate() 會跟著換；
  // 直接寫 textContent 的會停在原本的語言，要重新整理才換。清掉時鍵也要拿掉，不然重畫時「尚無圖片」又被補回有圖的牆上
  setStatus(key, params) {
    if (key && !params) this.status.dataset.i18n = key;
    else this.status.removeAttribute("data-i18n");
    this.status.textContent = key ? t(key, params) : "";
  }

  async reset() {
    this.gen++;
    this.scrollY = 0;
    this.items = [];
    this.offset = 0;
    this.total = 0;
    this.loading = false;
    this.clearSelection();
    this.box.replaceChildren();
    await this.loadMore();
  }

  async loadMore() {
    if (this.loading || (this.offset && this.offset >= this.total)) return;
    this.loading = true;
    const gen = this.gen, q = this.query();
    this.setStatus("wall.loading");
    try {
      const d = await get(`/api/list?offset=${this.offset}${q ? `&${q}` : ""}`);
      if (gen !== this.gen) return;           // 篩選已經換了，這批丟掉
      if (d.error) throw new Error(d.error);
      this.items.push(...d.items);
      this.total = d.total;
      this.offset += d.items.length;
      this.append(d.items);
      this.setStatus(this.total ? "" : typeof this.emptyKey === "function" ? this.emptyKey() : this.emptyKey);
    } catch (e) {
      if (gen === this.gen) this.setStatus("wall.error", {msg: e.message || e});
    } finally {
      if (gen === this.gen) { this.loading = false; this.drawCount(); }
    }
  }

  // 剛生成的圖：拿第一頁，比已經有的還新的就放到最前面（顯影中那格之後）
  async refreshNewest() {
    try {
      const q = this.query(), gen = this.gen;
      const d = await get(`/api/list?offset=0${q ? `&${q}` : ""}`);
      if (gen !== this.gen || d.error || !d.items) return;
      const known = new Set(this.items.map(i => i.id));
      const fresh = [];
      for (const it of d.items) { if (known.has(it.id)) break; fresh.push(it); }
      this.total = d.total;
      if (fresh.length) {
        this.items.unshift(...fresh);
        this.offset += fresh.length;
        this.layout();                         // 新圖要排在最前面，只能整面重排
        this.setStatus("");
      }
      this.drawCount();
    } catch { /* 下一張出來時會再試 */ }
  }

  // 生成中在最前面放一格「顯影中」，比例跟要生成的尺寸一樣，由下往上顯影
  setDev(state, s) {
    if (!state.running) {
      if (this.dev) { this.dev = null; this.layout(); }
      return;
    }
    if (!this.dev) {
      this.dev = h("figure.im.dev", {"aria-hidden": "true"},
        h("div.devl", {}, h("b"), h("span", {"data-i18n": "wall.developing"})));
      translate(this.dev);
      this.devRatio = s?.WIDTH && s?.HEIGHT ? s.HEIGHT / s.WIDTH : 1.4;
      this.layout();
      this.toTop();                    // 開始生成：捲回最上面看正在生成的那張（使用者筆記）
    }
    if (s?.WIDTH && s?.HEIGHT) this.dev.style.aspectRatio = `${s.WIDTH} / ${s.HEIGHT}`;
    this.dev.style.setProperty("--p", state.total ? (state.step / state.total).toFixed(3) : "0");
    this.dev.querySelector("b").textContent = state.total ? `${state.step} / ${state.total}` : "…";
  }

  // ── 選取 ──
  onClick(e) {
    const fig = e.target.closest("figure.im[data-id]");
    if (!fig) return;
    const id = Number(fig.dataset.id);
    const idx = this.items.findIndex(i => i.id === id);
    if (e.shiftKey && this.anchor != null && idx >= 0) {
      const [a, b] = this.anchor < idx ? [this.anchor, idx] : [idx, this.anchor];
      for (const it of this.items.slice(a, b + 1)) if (!this.sel.includes(it.id)) this.sel.push(it.id);
    } else {
      const k = this.sel.indexOf(id);
      if (k >= 0) this.sel.splice(k, 1); else this.sel.push(id);
    }
    this.anchor = idx;
    this.drawSel();
  }

  selected() {
    const byId = new Map(this.items.map(i => [i.id, i]));
    return this.sel.map(id => byId.get(id)).filter(Boolean);
  }

  // 全選目前載入的（還沒捲到的不算）
  selectAll() {
    const have = new Set(this.sel);
    for (const it of this.items) if (!have.has(it.id)) this.sel.push(it.id);
    this.drawSel();
  }

  clearSelection() {
    if (!this.sel.length) return;
    this.sel = [];
    this.anchor = null;
    this.drawSel();
  }

  drawSel() {
    this.box.classList.toggle("selecting", this.sel.length > 0);
    for (const [id, node] of this.nodes) {
      const k = this.sel.indexOf(id);
      node.classList.toggle("sel", k >= 0);
      if (this.selectable) node.setAttribute("aria-selected", String(k >= 0));
      let tape = node.querySelector(".tape");
      if (k >= 0) {
        if (!tape) { tape = h("span.tape"); node.append(tape); }
        tape.textContent = String(k + 1);
      } else tape?.remove();
    }
    this.onSelect?.(this.selected());
  }

  // 丟進回收桶之後：從牆上拿掉，不必重新載入
  remove(ids) {
    const gone = new Set(ids);
    const before = this.items.length;
    this.items = this.items.filter(i => !gone.has(i.id));
    const n = before - this.items.length;
    this.offset -= n;
    this.total = Math.max(0, this.total - n);
    for (const id of gone) this.nodes.delete(id);
    this.sel = this.sel.filter(id => !gone.has(id));
    this.layout();
    this.drawSel();
    this.drawCount();
  }

  countText() { return t("wall.count", {n: this.total.toLocaleString(lang())}); }
}

// ── 滑過放大 ─────────────────────────────────────
// 游標停在圖上 0.22 秒就在旁邊浮出大圖，移開就收。開關在每道牆的「排法」小視窗裡。
// 大圖優先放游標右邊，放不下才放左邊；先顯示縮圖，原圖載好再換上。
let zoomEl = null, zoomTimer = null, zoomSeq = 0;

function hideZoom() {
  clearTimeout(zoomTimer);
  zoomSeq++;
  if (zoomEl) zoomEl.hidden = true;
}

function placeZoom(e) {
  if (!zoomEl || zoomEl.hidden) return;
  const z = zoom(), w = zoomEl.offsetWidth * z, ht = zoomEl.offsetHeight * z, pad = 16;
  let x = e.clientX + 24;
  if (x + w > innerWidth - pad) x = Math.max(pad, e.clientX - w - 24);
  const y = Math.max(pad, Math.min(innerHeight - ht - pad, e.clientY - ht / 2));
  zoomEl.style.left = `${x / z}px`;
  zoomEl.style.top = `${y / z}px`;
}

function showZoom(it, e) {
  if (!zoomEl) {
    zoomEl = h("div.hzoom", {hidden: true, "aria-hidden": "true"}, h("img", {alt: ""}));
    document.body.append(zoomEl);
  }
  const seq = ++zoomSeq;
  const img = zoomEl.firstChild;
  // 盡量大，但不超過視窗的一半寬、八成五高
  const ar = it.width && it.height ? it.width / it.height : 0.75;
  const z = zoom();
  const maxW = Math.min(innerWidth * 0.5, 900) / z, maxH = innerHeight * 0.85 / z;
  const w = Math.min(maxW, maxH * ar);
  zoomEl.style.width = `${Math.round(w)}px`;
  zoomEl.style.height = `${Math.round(w / ar)}px`;
  img.src = `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v)}`;
  const full = new Image();
  full.onload = () => { if (seq === zoomSeq) img.src = full.src; };
  full.src = `/api/full?id=${it.id}&v=${encodeURIComponent(it.v)}`;
  zoomEl.hidden = false;
  placeZoom(e);
}

function hoverZoom(wall) {
  if (!wall.zoomKey) return;
  wall.box.addEventListener("mouseover", e => {
    const it = wall.itemOf(e);
    if (!it || !wall.ui[wall.zoomKey] || document.body.classList.contains("viewing")) return;
    clearTimeout(zoomTimer);
    const at = {clientX: e.clientX, clientY: e.clientY};
    // 等的這 0.22 秒裡牆可能已經不在畫面上（看大圖按「找相似」：看大圖一關，游標底下換成牆上的縮圖，接著就切到探索）：
    // 不在了就不浮出來，不然再也等不到 mouseout，預覽會一直卡在下一個畫面上
    zoomTimer = setTimeout(() => { if (wall.box.isConnected) showZoom(it, at); }, 220);
  });
  wall.box.addEventListener("mousemove", e => {
    if (zoomEl && !zoomEl.hidden) placeZoom(e);
  });
  wall.box.addEventListener("mouseout", e => {
    const from = e.target.closest("figure.im[data-id]");
    if (from && !from.contains(e.relatedTarget)) hideZoom();
  });
  wall.box.addEventListener("mousedown", hideZoom);
}
addEventListener("wheel", hideZoom, {passive: true});
addEventListener("blur", hideZoom);
addEventListener("hashchange", hideZoom);       // 換畫面：已經浮出來的也收起來（原本的牆被換掉，mouseout 不會來）
