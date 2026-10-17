// 圖庫篩選列的畫面：圖庫空間與工作室的牆各一份（使用者要求 2026-10-10：生成頁的圖也要能篩選）。
// 條件與查詢字串在 filterq.js；這裡是搜尋框、篩選鈕、勾選的小視窗、清除鈕。
// 圖庫空間是一整列；工作室的牆收成一顆「篩選」鈕（menu），點了開小視窗，跟「排法」一樣（使用者要求 2026-10-10）。

import {get} from "./api.js";
import {$, h} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {closePop, openPop, openedPop} from "./pop.js";
import {LORA_MODES, MULTI, RATINGS, activeOf, blank, countOf, nowSec, queryOf} from "./filterq.js";

export const lastPart = p => String(p).replace(/[\\/]+$/, "").split(/[\\/]/).pop();
const SORTS = ["new", "old", "rating", "name"];
const starText = v => (v === "0" ? t("lib.rating.0") : "★".repeat(Number(v)));

// 篩選清單（來源、模型、LoRA、資料夾，各附張數）兩份共用。圖庫有變就丟掉，下次要用時重抓；同時要的只抓一次
let FACETS = null, facetsReq = null;
export const facets = () => FACETS;
export function dropFacets() { FACETS = null; }
export function loadFacets() {
  facetsReq ??= get("/api/lib/facets").then(d => { FACETS = d; return d; }).finally(() => { facetsReq = null; });
  return facetsReq;
}

export class Filters {
  // id：這一列與清除鈕的 id 前綴（`${id}Filters`、`${id}Clear`），小視窗也用它分開；
  // keepNew：查詢帶 keep_new（工作室的牆：開著篩選時新生成的圖照樣列出）；build(F)：這一列放哪些東西；
  // menu：收成一顆「篩選」鈕（`${id}Menu`），build(F) 是點開的小視窗（`pop-${id}`）裡放的東西
  constructor({id, keepNew = false, menu = false, build}) {
    Object.assign(this, {id, keepNew, menu, build, wall: null, f: blank(), searchTimer: null, since: nowSec()});
  }

  query() { return queryOf(this.f, {keepNew: this.keepNew, since: this.since}); }
  active() { return activeOf(this.f); }
  count() { return countOf(this.f); }

  // 一整列：改了就整列重畫（來源、勝出這些按鈕的狀態跟著換）。
  // 收成小視窗：在小視窗裡改的，小視窗留著、字就地更新（整個重畫會把正在打字的搜尋框換掉）
  set(patch) {
    Object.assign(this.f, patch);
    if (this.menu) this.updateChips();
    else { closePop(); this.draw(); }
    this.restart();
  }

  // 換成只有這些條件（從別處跳過來看「用了這個 LoRA 的圖」）
  only(patch = {}) { this.f = {...blank(), ...patch}; this.since = nowSec(); }

  // 條件改了：牆重新載入。keepNew 的起點跟著換成現在——在這之前生成的都當成舊圖照篩選
  // （使用者回報：取消篩選再重新篩選，之前列出的新圖還混在裡面）
  restart() {
    this.since = nowSec();
    this.wall?.reset();
  }

  // 在小視窗裡勾選：小視窗不關，可以連續選好幾個；篩選列的字就地更新
  toggle(kind, v, btn) {
    const arr = this.f[kind];
    const i = arr.indexOf(v);
    if (i >= 0) arr.splice(i, 1); else arr.push(v);
    btn?.setAttribute("aria-pressed", String(i < 0));
    this.updateChips();
    this.restart();
  }

  nameOf(kind, v) {
    if (kind === "folder" || kind === "root") return lastPart(v);
    const x = (FACETS?.[`${kind}s`] || []).find(i => i.v === v);
    return x?.name || lastPart(v);
  }

  isSet(kind) { return MULTI.includes(kind) || kind === "rating" ? this.f[kind].length > 0 : false; }

  // back：收成小視窗時，清單最上面的「‹ 篩選」
  pickPop(kind, back = null) {
    const f = this.f;
    const list = (FACETS?.[`${kind}s`] || []).map(x => ({v: x.v, name: x.name || lastPart(x.v), c: x.c, title: x.v}));
    const box = h("div.plist");
    const fill = needle => {
      const n = needle.trim().toLowerCase();
      // 已經選的排最前面，才不會被搜尋或長清單埋掉
      const shown = list.filter(x => !n || x.name.toLowerCase().includes(n))
        .sort((a, b) => f[kind].includes(b.v) - f[kind].includes(a.v)).slice(0, 200);
      box.replaceChildren(...(shown.length ? shown.map(x => h("button", {
        type: "button", "aria-pressed": String(f[kind].includes(x.v)), title: x.title,
        on: {click: e => this.toggle(kind, x.v, e.currentTarget)}},
        h("span", {text: x.name}), h("span.c.mono", {text: x.c.toLocaleString(lang())})))
        : [h("p.dim", {text: t("lib.noMatch")})]));
    };
    fill("");
    const mode = kind !== "lora" ? null : h("div.row", {},
      h("div.infoline", {}, h("div.seg", {role: "group", "data-i18n-aria": "lib.loraMode"}, LORA_MODES.map(m => h("button", {
        type: "button", "aria-pressed": String(f.loraMode === m), "data-i18n": `lib.loraMode.${m}`,
        on: {click: e => {
          f.loraMode = m;
          e.currentTarget.parentElement.querySelectorAll("button")
            .forEach(b => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
          this.updateChips();
          if (f.lora.length > 1) this.restart();
        }}})))));
    return [
      h("div.hd", {}, back, h("h3.title", {"data-i18n": `lib.${kind}`}), h("span.dim", {"data-i18n": "lib.multiHint"})),
      mode,
      h("input.find", {type: "search", "data-i18n-placeholder": "lib.find", "data-i18n-aria": "lib.find",
                       on: {input: e => fill(e.target.value)}}),
      box];
  }

  ratingPop(back = null) {
    return [
      h("div.hd", {}, back, h("h3.title", {"data-i18n": "lib.rating"}), h("span.dim", {"data-i18n": "lib.multiHint"})),
      h("div.plist.prating", {}, RATINGS.map(v => h("button", {
        type: "button", "aria-pressed": String(this.f.rating.includes(v)), title: v === "0" ? null : t("lib.stars", {n: v}),
        on: {click: e => this.toggle("rating", v, e.currentTarget)}},
        v === "0" ? h("span", {"data-i18n": "lib.rating.0"})
          : h("span.st", {}, "★".repeat(Number(v)), h("i", {text: "★".repeat(5 - Number(v))})))))];
  }

  listPop(title, key, values, labelPrefix) {
    return [
      h("div.hd", {}, h("h3.title", {"data-i18n": title})),
      h("div.plist", {}, values.map(v => h("button", {
        type: "button", "aria-pressed": String(this.f[key] === v), on: {click: () => this.set({[key]: v})}},
        h("span", {"data-i18n": `${labelPrefix}.${v || "any"}`}))))];
  }

  // 篩選鈕上的字：沒選「模型 ▾」；選一個「模型：名字」；選好幾個「模型：名字 +2」。
  // bare：小視窗裡左邊已經有「模型」這些字，鈕上只寫選了什麼（沒選是「不限 ▾」）
  chipText(kind, label, bare = false) {
    const f = this.f;
    const none = `${t(bare ? "lib.any" : label)} ▾`, head = bare ? "" : `${t(label)}：`;
    if (kind === "rating") {
      const on = RATINGS.filter(v => f.rating.includes(v));           // 照星級高到低排，不照點的順序
      return on.length ? `${head}${on.map(starText).join("、")}` : none;
    }
    if (kind === "sort") return `${t(label)}：${t(`lib.sort.${f.sort}`)}`;
    const vals = f[kind];
    if (!vals.length) return none;
    const mode = kind === "lora" && vals.length > 1 ? `（${t(`lib.loraMode.${f.loraMode}`)}）` : "";
    const names = `${this.nameOf(kind, vals[0])}${vals.length > 1 ? ` +${vals.length - 1}` : ""}`;
    return bare ? `${names}${mode}` : `${t(label)}${mode}：${names}`;
  }

  dropChip(kind, label, bare = false) {
    const set = this.isSet(kind);
    return h("span.chipg", {"data-set": set ? "1" : null, "data-kind": kind, "data-label": label, "data-bare": bare ? "1" : null},
      h("button.chip", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false",
                        title: MULTI.includes(kind) && set ? this.f[kind].map(v => this.nameOf(kind, v)).join("、") : null,
                        on: {click: e => this.openPick(kind, e.currentTarget)}},
        this.chipText(kind, label, bare)),
      kind !== "sort" ? h("button.x", {type: "button", "data-i18n-aria": "lib.clearOne", hidden: !set,
                                       on: {click: () => this.set({[kind]: MULTI.includes(kind) || kind === "rating" ? [] : ""})}}, "✕") : null);
  }

  // 勾選的小視窗。收成小視窗時：清單換掉篩選的小視窗、貼在同一顆「篩選」鈕下面，最上面「‹ 篩選」回去
  // （小視窗同一時間只開一個）
  openPick(kind, chip) {
    const anchor = this.menu ? $(`${this.id}Menu`) : chip;
    if (!anchor) return;
    if (this.menu && !FACETS && MULTI.includes(kind)) {
      loadFacets().catch(() => null).then(() => { if (FACETS) this.openPick(kind, chip); });
      return;
    }
    const back = this.menu ? h("button.ghost", {type: "button", on: {click: () => this.openMenu()}},
                               "‹ ", h("span", {"data-i18n": "lib.filters"})) : null;
    const id = `pop-${this.id}-${kind}`;
    openPop(id, anchor, () => (kind === "rating" ? this.ratingPop(back)
      : kind === "sort" ? this.listPop("lib.sort", "sort", SORTS, "lib.sort")
      : this.pickPop(kind, back)));
    if (this.menu) openedPop(id)?.querySelector(".find")?.focus();   // 開清單多半是要找名字，不是要按「‹ 篩選」
  }

  // 清除篩選：搜尋與排序留著（各有自己的地方改）；收成小視窗時搜尋框也在小視窗裡，一起清
  clearable() { return this.menu ? this.count() > 0 : this.active(); }
  clearButton() {
    return h("button.ghost", {type: "button", id: `${this.id}Clear`, hidden: !this.clearable(), "data-i18n": "lib.clear",
                              on: {click: () => {
                                clearTimeout(this.searchTimer);
                                this.set({...blank(), q: this.menu ? "" : this.f.q, sort: this.f.sort});
                                const box = this.menu && $(`${this.id}Search`);
                                if (box) box.value = "";
                              }}});
  }

  // ── 收成一顆「篩選」鈕（工作室的牆） ──
  // 開著篩選時鈕上寫幾個、換成強調色：牆上看得出現在只列一部分；滑過看是哪些
  menuButton() {
    const b = h("button.ghost", {type: "button", id: `${this.id}Menu`, "aria-haspopup": "dialog", "aria-expanded": "false",
                                 on: {click: e => this.openMenu(e.currentTarget)}});
    this.paintMenu(b);
    return b;
  }

  paintMenu(b) {
    const n = this.count();
    b.textContent = n ? t("lib.filtersN", {n}) : t("lib.filters");
    b.toggleAttribute("data-on", n > 0);
    b.title = n ? this.summary() : "";
  }

  // 開了哪些條件，一行一個
  summary() {
    const f = this.f;
    return [f.q ? `${t("lib.searchShort")}：${f.q}` : null,
            ...[...MULTI, "rating"].filter(k => this.isSet(k)).map(k => this.chipText(k, `lib.${k}`))]
      .filter(Boolean).join("\n");
  }

  openMenu(btn = $(`${this.id}Menu`)) {
    if (!btn) return;
    openPop(`pop-${this.id}`, btn, () => this.build(this));
    $(`${this.id}Search`)?.focus();
    // 清單（模型、LoRA、資料夾的名字）還沒抓到：抓到再把鈕上的字換成名字
    if (!FACETS) loadFacets().then(() => this.updateChips()).catch(() => { /* 搜尋照樣能用 */ });
  }

  // 搜尋框：打完字停 0.35 秒才查
  searchInput(id) {
    return h("input", {id, type: "search", value: this.f.q, "data-i18n-placeholder": "lib.search",
                       "data-i18n-aria": "lib.search", "data-i18n-title": "lib.searchHint",
                       autocomplete: "off", spellcheck: "false",
                       on: {input: e => {
                         clearTimeout(this.searchTimer);
                         const v = e.target.value.trim();
                         this.searchTimer = setTimeout(() => { if (v !== this.f.q) this.set({q: v}); }, 350);
                       }}});
  }

  // 小視窗開著時不能整列重畫（會把小視窗的依附對象換掉），只更新字與清除鈕
  updateChips() {
    for (const g of document.querySelectorAll(`#${this.id}Filters .chipg[data-kind], #pop-${this.id} .chipg[data-kind]`)) {
      const kind = g.dataset.kind, set = this.isSet(kind);
      const chip = g.querySelector(".chip");
      chip.textContent = this.chipText(kind, g.dataset.label, !!g.dataset.bare);
      chip.title = MULTI.includes(kind) && set ? this.f[kind].map(v => this.nameOf(kind, v)).join("、") : "";
      g.toggleAttribute("data-set", set);
      const x = g.querySelector(".x");
      if (x) x.hidden = !set;
    }
    const clear = $(`${this.id}Clear`);
    if (clear) clear.hidden = !this.clearable();
    const menu = $(`${this.id}Menu`);
    if (menu) this.paintMenu(menu);
  }

  draw() {
    if (this.menu) { this.updateChips(); return; }
    const row = $(`${this.id}Filters`);
    if (!row) return;
    row.replaceChildren(...this.build(this));
    translate(row);
  }
}
