// 從圖庫挑圖（使用者筆記：比較組建好之後，要能直接在比較頁加圖）。
// 蓋在畫面上的視窗：搜尋（跟圖庫同一個搜尋，/api/list 的 q）、點一下選、Shift 選一段、捲到底載入更多。
// have：已經在裡面的圖（id），標「已加入」不能再選
// single：只挑一張，點了就選定（精修的「換一張」；不用再按確定，也不會留下圖庫的勾選）

import {get} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {closePop} from "./pop.js";

const P = {el: null, items: [], total: 0, q: "", sel: [], last: null, have: new Set(), onAdd: null, back: null, seq: 0, busy: false,
           single: false};

export function openPicker({title, have = [], onAdd, single = false}) {
  closePop();
  Object.assign(P, {items: [], total: 0, q: "", sel: [], last: null, have: new Set(have), onAdd, single, back: document.activeElement});
  P.el?.remove();
  P.el = h("div.assets.picker", {role: "dialog", "aria-modal": "true", "aria-labelledby": "pickTitle"});
  P.el.addEventListener("click", e => { if (e.target === P.el) closePicker(); });
  const find = h("input.find", {type: "search", id: "pickFind", spellcheck: "false", "data-i18n-placeholder": "pick.find",
                                "data-i18n-aria": "pick.find",
                                on: {input: e => { clearTimeout(P.timer); P.timer = setTimeout(() => search(e.target.value.trim()), 250); }}});
  const grid = h("div.tgrid.pkgrid", {id: "pickGrid", role: "listbox", "aria-multiselectable": "true", "data-i18n-aria": "pick.list"});
  const more = h("p.dim.pkmore", {id: "pickMore"});
  new IntersectionObserver(es => { if (es[0].isIntersecting) loadMore(); }, {root: grid}).observe(more);
  grid.append(more);
  P.el.append(h("div.apanel.tpanel", {},
    h("div.ahd", {},
      h("h2.title", {id: "pickTitle", text: title}),
      find,
      h("span.sp"),
      h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: closePicker}}, "✕")),
    grid,
    h("div.sfoot", {},
      h("span.msg", {id: "pickMsg", role: "status"}), h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "pick.cancel", on: {click: closePicker}}),
      single ? null : h("button.pill.primary", {type: "button", id: "pickAdd", disabled: true, on: {click: done}}))));
  translate(P.el);
  document.body.append(P.el);
  document.body.classList.add("viewing");
  syncSel();
  search("");
  find.focus();
}

export function closePicker() {
  if (!P.el) return;
  P.el.remove();
  P.el = null;
  document.body.classList.remove("viewing");
  P.back?.focus?.();
}

async function search(q) {
  P.q = q;
  P.items = [];
  P.total = 0;
  const grid = $("pickGrid");
  if (!grid) return;
  grid.querySelectorAll(".tcard").forEach(c => c.remove());
  await loadMore(true);
}

async function loadMore(fresh = false) {
  if (!P.el || (P.busy && !fresh) || (!fresh && P.items.length >= P.total)) return;
  P.busy = true;
  const seq = ++P.seq;
  const qs = new URLSearchParams({sort: "new", offset: String(P.items.length)});
  if (P.q) qs.set("q", P.q);
  const d = await get(`/api/list?${qs}`).catch(() => ({items: [], total: 0}));
  P.busy = false;
  if (seq !== P.seq || !P.el) return;
  P.total = d.total || 0;
  const grid = $("pickGrid"), more = $("pickMore");
  for (const it of d.items || []) {
    P.items.push(it);
    grid.insertBefore(card(it), more);
  }
  more.textContent = P.items.length < P.total ? t("pick.loading") : (P.total ? "" : t("pick.none"));
  // 一頁還填不滿格子時，觀察者不會再觸發：自己再抓一次
  if (P.items.length < P.total && grid.scrollHeight <= grid.clientHeight + 10) loadMore();
}

function card(it) {
  const inSet = P.have.has(it.id);
  return h("button.tcard", {type: "button", "data-id": String(it.id), "aria-pressed": "false", disabled: inSet,
                            title: it.filename, on: {click: e => pick(it.id, e.shiftKey)}},
    h("img", {src: `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v)}`, alt: "", loading: "lazy"}),
    h("span.tmeta", {}, h("b", {text: it.filename}), inSet ? h("span.dim", {text: t("pick.have")}) : null));
}

function pick(id, shift) {
  if (P.single) { P.sel = [id]; done(); return; }
  const ids = P.items.filter(it => !P.have.has(it.id)).map(it => it.id);
  const i = ids.indexOf(id), j = P.last == null ? -1 : ids.indexOf(P.last);
  if (shift && i >= 0 && j >= 0) {
    for (const x of ids.slice(Math.min(i, j), Math.max(i, j) + 1)) if (!P.sel.includes(x)) P.sel.push(x);
  } else if (P.sel.includes(id)) P.sel = P.sel.filter(x => x !== id);
  else P.sel.push(id);
  P.last = id;
  syncSel();
}

function syncSel() {
  P.el?.querySelectorAll(".tcard").forEach(c => c.setAttribute("aria-pressed", String(P.sel.includes(Number(c.dataset.id)))));
  const b = $("pickAdd");
  if (b) { b.disabled = !P.sel.length; b.textContent = P.sel.length ? t("pick.addN", {n: P.sel.length}) : t("pick.add"); }
}

async function done() {
  if (!P.sel.length) return;
  const ids = P.sel.slice(), onAdd = P.onAdd;
  closePicker();
  await onAdd?.(ids);
}

document.addEventListener("keydown", e => {
  if (!P.el || e.key !== "Escape" || e.defaultPrevented) return;
  e.preventDefault();
  closePicker();
});
