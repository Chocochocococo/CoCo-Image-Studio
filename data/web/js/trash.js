// 回收桶（使用者筆記：丟進回收桶之後要自己開資料夾整理很奇怪）：看 CoCo 的 .trash 裡有什麼，
// 還原、永久刪除、清空。後端在 server/trash.py；丟進去時記了原本的位置，還原就放回去，
// 早期丟進來、沒有紀錄的要選資料夾。永久刪除與清空都要按兩次

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {closePop} from "./pop.js";
import {pickFolder} from "./setup.js";

const T = {el: null, items: [], bytes: 0, sel: new Set(), last: null, onChanged: null, back: null, arm: {}};

export async function openTrash(onChanged) {
  closePop();
  T.onChanged = onChanged;
  T.back = document.activeElement;
  T.sel = new Set();
  T.last = null;
  T.el?.remove();
  T.el = h("div.assets.trash", {role: "dialog", "aria-modal": "true", "aria-labelledby": "trashTitle"});
  T.el.addEventListener("click", e => { if (e.target === T.el) closeTrash(); });
  document.body.append(T.el);
  document.body.classList.add("viewing");
  await load();
}

export function closeTrash() {
  if (!T.el) return;
  for (const k of Object.keys(T.arm)) clearTimeout(T.arm[k]);
  T.arm = {};
  T.el.remove();
  T.el = null;
  document.body.classList.remove("viewing");
  T.back?.focus?.();
}

async function load() {
  const d = await get("/api/trash").catch(() => ({items: [], bytes: 0}));
  T.items = d.items || [];
  T.bytes = d.bytes || 0;
  const names = new Set(T.items.map(it => it.name));
  T.sel = new Set([...T.sel].filter(n => names.has(n)));
  draw();
}

const mb = b => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(1)} GB` : `${(b / 1024 ** 2).toFixed(1)} MB`);
const when = s => new Date(s * 1000).toLocaleString(lang() || undefined, {year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
const folderOf = p => p.replace(/[\\/][^\\/]*$/, "");

function say(text, kind = "ok") {
  const m = $("trashMsg");
  if (!m) return;
  m.textContent = text;
  m.dataset.kind = kind;
}

// 點一下選／取消；Shift＋點選一整段
function pick(name, shift) {
  const idx = T.items.findIndex(it => it.name === name);
  const from = T.last == null ? -1 : T.items.findIndex(it => it.name === T.last);
  if (shift && from >= 0 && idx >= 0) {
    const [a, b] = from < idx ? [from, idx] : [idx, from];
    for (const it of T.items.slice(a, b + 1)) T.sel.add(it.name);
  } else if (T.sel.has(name)) T.sel.delete(name);
  else T.sel.add(name);
  T.last = name;
  syncSel();
}

function syncSel() {
  T.el?.querySelectorAll(".tcard").forEach(c => c.setAttribute("aria-pressed", String(T.sel.has(c.dataset.name))));
  const n = T.sel.size;
  const r = $("trashRestore"), d = $("trashDelete"), all = $("trashAll");
  if (r) { r.disabled = !n; r.textContent = n ? t("trash.restoreN", {n}) : t("trash.restore"); }
  if (d && !T.arm.del) { d.disabled = !n; d.textContent = t("trash.delete"); }
  if (all) all.textContent = t(n && n === T.items.length ? "trash.none" : "trash.all");
}

// 要按兩次的按鈕：第一次變成「確定？」，4 秒內再按才做
function armed(key, btn, sure, run) {
  if (!T.arm[key]) {
    btn.textContent = sure;
    btn.classList.add("armed");
    T.arm[key] = setTimeout(() => { delete T.arm[key]; btn.classList.remove("armed"); syncSel(); if (key === "empty") btn.textContent = t("trash.empty"); }, 4000);
    return;
  }
  clearTimeout(T.arm[key]);
  delete T.arm[key];
  btn.classList.remove("armed");
  run();
}

async function restore() {
  const names = [...T.sel];
  if (!names.length) return;
  let d = await post("/api/trash", {action: "restore", names}).catch(e => ({error: e.message}));
  if (d.error) { say(t("lib.err", {msg: d.error}), "err"); return; }
  let done = d.done, failed = d.failed;
  // 早期丟進來的不知道原本在哪：選一個資料夾放
  if (d.need?.length) {
    say(t("trash.needDest", {n: d.need.length}), "warn");
    const dest = await pickFolder("");
    if (dest) {
      const d2 = await post("/api/trash", {action: "restore", names: d.need, dest}).catch(e => ({error: e.message}));
      if (!d2.error) { done += d2.done; failed += d2.failed; }
    }
  }
  T.sel.clear();
  await load();
  say(t("trash.restored", {n: done}) + (failed ? " · " + t("lib.failed", {n: failed}) : ""), failed ? "err" : "ok");
  if (done) T.onChanged?.();
}

async function remove(action) {
  const names = action === "empty" ? [] : [...T.sel];
  const d = await post("/api/trash", {action, names}).catch(e => ({error: e.message}));
  if (d.error) { say(t("lib.err", {msg: d.error}), "err"); return; }
  T.sel.clear();
  await load();
  say(t("trash.deleted", {n: d.done}) + (d.failed ? " · " + t("lib.failed", {n: d.failed}) : ""), d.failed ? "err" : "ok");
}

function draw() {
  const [infoBtn, infoText] = info("trash.hint");
  const card = it => h("button.tcard", {
    type: "button", "data-name": it.name, "aria-pressed": String(T.sel.has(it.name)),
    title: it.orig || t("trash.noOrig"), on: {click: e => pick(it.name, e.shiftKey)}},
    h("img", {src: `/api/trash/thumb?name=${encodeURIComponent(it.name)}&v=${encodeURIComponent(it.v)}`, alt: "", loading: "lazy"}),
    h("span.tmeta", {},
      h("b", {text: it.name}),
      h("span.dim", {text: it.orig ? folderOf(it.orig) : t("trash.noOrig")}),
      h("span.dim.mono", {text: when(it.time)})));
  const del = h("button.pill", {type: "button", id: "trashDelete", disabled: true, "data-i18n": "trash.delete",
                                on: {click: e => armed("del", e.currentTarget, t("trash.deleteSure", {n: T.sel.size}), () => remove("delete"))}});
  T.el.replaceChildren(h("div.apanel.tpanel", {},
    h("div.ahd", {},
      h("h2.title", {id: "trashTitle", "data-i18n": "trash.title"}), infoBtn,
      h("span.dim.mono", {text: T.items.length ? t("trash.count", {n: T.items.length, size: mb(T.bytes)}) : ""}),
      h("span.sp"),
      h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: closeTrash}}, "✕")),
    infoText,
    T.items.length
      ? h("div.tgrid", {role: "listbox", "aria-multiselectable": "true", "data-i18n-aria": "trash.title"}, T.items.map(card))
      : h("div.tnone", {}, h("p.dim", {"data-i18n": "trash.emptyNow"})),
    h("div.sfoot", {},
      h("span.msg", {id: "trashMsg", role: "status"}), h("span.sp"),
      T.items.length ? h("button.pill", {type: "button", id: "trashAll", on: {click: () => {
        if (T.sel.size === T.items.length) T.sel.clear(); else T.items.forEach(it => T.sel.add(it.name));
        syncSel();
      }}}) : null,
      T.items.length ? h("button.pill", {type: "button", id: "trashEmpty", "data-i18n": "trash.empty",
                                         on: {click: e => armed("empty", e.currentTarget, t("trash.emptySure", {n: T.items.length}), () => remove("empty"))}}) : null,
      T.items.length ? del : null,
      T.items.length ? h("button.pill.primary", {type: "button", id: "trashRestore", disabled: true, on: {click: restore}}) : null)));
  translate(T.el);
  syncSel();
}

document.addEventListener("keydown", e => {
  // 選資料夾的小視窗開著時，它在捕獲階段就把 Esc 攔下了，不會到這裡
  if (!T.el || e.key !== "Escape" || e.defaultPrevented) return;
  e.preventDefault();
  closeTrash();
});
