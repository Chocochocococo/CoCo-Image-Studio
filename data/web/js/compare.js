// 比較模式（工作室的 03 比較）：一組圖並排、找出彼此哪裡不一樣、標出勝出的、匯出成一張比較圖。
//
// 比較組存在伺服器（server/compare.py），以檔案路徑記錄。放進比較組的入口在
// 圖庫與工作室的動作列、看大圖，以及「一批生完自動送來」；這裡是看與整理。
// 排法：並排、格狀、兩張滑桿、淘汰賽（一輪一輪挑晉級的，覺得夠了就「就選這些」，邏輯在 cup.js）。

import {get, post} from "./api.js";
import {xyExportLayout, xyLayout} from "./cmpaxes.js";
import {advance, alive, back, canAdvance, finish, restart, roundOf, route} from "./cup.js";
import {$, canvasScale, h, seqLabel, stem} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {saveFromImagePop} from "./recipes.js";
import {closePop, openPop} from "./pop.js";
import {openViewer} from "./viewer.js";
import {openPicker} from "./picker.js";

const C = {sets: [], cur: null, curId: "", view: "side", loupe: false, zoom: 3, pickA: 0, pickB: 1, diffOpen: false,
           delArm: null, saveTimer: null, dirty: null /* 排著還沒送的那組 {set, patch} */, ui: null, swap: false, active: ""};

// 格狀的大小：卡片的最小寬度
const GRID_SIZES = {s: 160, m: 230, l: 320};
// 超過這麼多張、而且沒選過排法時，預設用格狀（並排會太擠）
const GRID_FROM = 6;

// 排法、大小、XY 開關存在 ui 偏好（cmpview 空字串＝還沒選過，依張數決定）
export function initCompare(ui) { C.ui = ui; }

// ── 比較組的狀態（使用者筆記：比較組越來越多要能管理）──
// 比較中、已完成（淘汰賽「就選這些」或自己標）、封存（清單上另外一頁，勝出照樣算）
const TABS = ["active", "done", "archived"];
const L2 = {tab: "active", q: "", selMode: false, picked: new Set(), arm: null};
const statusOf = s => (s.archived ? "archived" : s.done ? "done" : "active");
const visible = () => C.sets.filter(s => statusOf(s) === L2.tab && (!L2.q || low(s.name).includes(low(L2.q))));
const pref = (k, fallback) => C.ui?.[k] ?? fallback;
function setPref(patch) {
  if (!C.ui) return;
  Object.assign(C.ui, patch);
  post("/api/ui", patch).catch(() => { /* 只影響下次開啟 */ });
}
const viewOf = s => pref("cmpview", "") || (s.items.length > GRID_FROM ? "grid" : "side");

const label = seqLabel;

// 超過這麼多張就提醒：並排每張會很小、差異說明很長，匯出的圖也可能要縮小。不擋，只提醒
export const MANY = 30;
const low = s => String(s || "").toLowerCase();

// ── 資料 ──
async function loadSets() {
  try { C.sets = (await get("/api/compare/sets")).sets || []; } catch { C.sets = []; }
}

async function loadSet(id) {
  C.curId = id;
  if (!id) { C.cur = null; return; }
  try {
    const d = await get(`/api/compare/set?id=${encodeURIComponent(id)}`);
    C.cur = d.error ? null : d;
  } catch { C.cur = null; }
  C.pickA = 0;
  C.pickB = Math.min(1, (C.cur?.items.length || 1) - 1);
}

// 改順序、勝出、拿掉、改名：畫面先變，停 0.25 秒再存（整組一起送）。
// 換狀態、加圖之前先 flushSave()：以前是直接取消排著的這次，名稱、勝出這些還沒存的改動就丟了（Codex 第三輪審查）
function save(patch = {}) {
  const s = C.cur;
  if (!s) return;
  if (C.dirty && C.dirty.set !== s) flushSave();     // 前一組還沒送的先送
  C.dirty = {set: s, patch: {...(C.dirty?.patch || {}), ...patch}};
  clearTimeout(C.saveTimer);
  C.saveTimer = setTimeout(flushSave, 250);
}

// 排著的那次馬上送；沒有排著的就什麼都不做
async function flushSave() {
  clearTimeout(C.saveTimer);
  const job = C.dirty;
  C.dirty = null;
  if (!job) return;
  const s = job.set;
  await post("/api/compare/update", {id: s.id, name: s.name, round: roundOf(s), done: !!s.done, ...job.patch,
    items: s.items.map(it => ({path: it.path, winner: it.winner, note: it.note, reach: it.reach, adv: it.adv}))}).catch(() => {});
  await loadSets();
  // 淘汰賽「就選這些」或回去重選：狀態變了，清單切到它所在的那一頁，不要從清單上消失
  const cur = C.sets.find(x => x.id === s.id);
  if (cur && !L2.selMode && statusOf(cur) !== L2.tab) L2.tab = statusOf(cur);
  drawSets();
}

// ── 差異 ──
// 提示詞拆成標籤比較：去掉 <lora:…>、以逗號與換行分開、忽略大小寫與前後空白
function tags(text) {
  return String(text || "").replace(/<lora:[^>]*>/gi, "").split(/[,\n]+/).map(x => x.trim().toLowerCase()).filter(Boolean);
}

const FIELDS = [
  ["model", "cmp.f.model", im => stem(im.model)],
  ["loras", "cmp.f.loras", im => (im.loras || []).map(l => `${stem(l.file)} ${Number(l.weight)}`).sort().join(" · ")],
  ["sampler", "cmp.f.sampler", im => im.sampler],
  ["scheduler", "cmp.f.schedule", im => im.scheduler],
  ["steps", "cmp.f.steps", im => im.steps],
  ["cfg", "cmp.f.cfg", im => im.cfg],
  ["seed", "cmp.f.seed", im => im.seed],
  ["size", "cmp.f.size", im => (im.width && im.height ? `${im.width}×${im.height}` : "")],
  ["clip", "cmp.f.clip", im => im.clip_skip],
  ["shift", "cmp.f.shift", im => (im.extra || {}).Shift ?? (im.extra || {}).shift],
  // 品質標籤開關：只有 CoCo 自己生的圖有記，其他的空白（XY 那邊把空白當「沒記錄」）
  ["preset", "cmp.f.preset", im => ({on: t("cmp.presetOn"), off: t("cmp.presetOff")})[(im.extra || {}).preset] || ""],
  // ControlNet：用了哪些參考圖模型、種類、強度（CoCo 生的圖才有記）
  ["controlnet", "cmp.f.controlnet", im => ((im.extra || {}).controlnet || [])
    .map(u => `${u.name || stem(u.model)} · ${t(`cn.type.${u.type}`)} ${Number(u.scale ?? 1)}`).join(" + ")],
];

// 回傳 {same: [[鍵, 值]], diff: [[鍵, 每張的值]], per: 每張的差異短語}
function diffs(items) {
  const imgs = items.map(it => it.img || {});
  const out = {same: [], diff: [], per: imgs.map(() => [])};
  if (!imgs.length) return out;
  for (const [key, lab, get1] of FIELDS) {
    const vals = imgs.map(im => { const v = get1(im); return v == null ? "" : String(v); });
    if (new Set(vals).size <= 1) { if (vals[0]) out.same.push([lab, vals[0]]); continue; }
    out.diff.push([lab, vals]);
    if (key === "loras") {
      // 只列跟別張不一樣的那幾個 LoRA
      const sets = imgs.map(im => (im.loras || []).map(l => `${stem(l.file)} ${Number(l.weight)}`));
      const common = sets.reduce((a, b) => a.filter(x => b.includes(x)));
      sets.forEach((s, i) => s.filter(x => !common.includes(x)).forEach(x => out.per[i].push(x)));
      sets.forEach((s, i) => { if (!s.length) out.per[i].push(t("cmp.noLora")); });
    } else {
      vals.forEach((v, i) => out.per[i].push(`${t(lab)} ${v || "—"}`));
    }
  }
  for (const [key, lab] of [["prompt", "cmp.f.prompt"], ["negative", "cmp.f.negative"]]) {
    const ts = imgs.map(im => tags(im[key]));
    const common = ts.reduce((a, b) => a.filter(x => b.includes(x)));
    const uniq = ts.map(s => [...new Set(s.filter(x => !common.includes(x)))]);
    if (uniq.some(u => u.length)) {
      out.diff.push([lab, uniq.map(u => (u.length ? `+ ${u.join(", ")}` : "—"))]);
      uniq.forEach((u, i) => {
        const shown = u.slice(0, 5).map(x => (key === "negative" ? `${t("cmp.neg")} +${x}` : `+${x}`));
        if (u.length > 5) shown.push(t("cmp.more", {n: u.length - 5}));
        out.per[i].push(...shown);
      });
    } else if (ts[0].length) out.same.push([lab, t("cmp.sameTags", {n: ts[0].length})]);
  }
  return out;
}

// 每個項目在每張圖的值，給 XY 對照表挑軸用（cmpaxes.js）。
// 提示詞只看「跟大家都有的相比多出來的標籤」，這樣換一個表情標籤的比較也能排成表。
function dimsOf(items) {
  const imgs = items.map(it => it.img || {});
  const dims = FIELDS.map(([key, lab, get1]) => ({key, lab, blankUnknown: true,
                                                   vals: imgs.map(im => { const v = get1(im); return v == null ? "" : String(v); })}));
  for (const [key, lab] of [["prompt", "cmp.f.prompt"], ["negative", "cmp.f.negative"]]) {
    const ts = imgs.map(im => tags(im[key]));
    // 組裡的圖全被搬走或刪除時一張都沒有：沒有初始值的 reduce 會丟錯，整個比較頁畫不出來
    const common = ts.length ? ts.reduce((a, b) => a.filter(x => b.includes(x))) : [];
    dims.push({key, lab, vals: ts.map(u => [...new Set(u.filter(x => !common.includes(x)))].map(x => `+${x}`).join(", "))});
  }
  return dims;
}

// ── 放進比較（圖庫的動作列、看大圖用） ──
export function addToCompare(ids, anchor, onDone) {
  openPop("popCompare", anchor, () => {
    const name = h("input.find", {type: "text", "data-i18n-placeholder": "cmp.newName", "data-i18n-aria": "cmp.newName",
                                  on: {input: () => { create.disabled = !name.value.trim(); },
                                       keydown: e => { if (e.key === "Enter" && name.value.trim()) { e.preventDefault(); add("", name.value.trim()); } }}});
    // 取了名字才能按（空的名字以前會建出「未命名」，按鈕又一直是灰的，看不出能不能按）
    const create = h("button.pill.primary", {type: "button", disabled: true, "data-i18n": "cmp.create",
                                             on: {click: () => add("", name.value.trim())}});
    const add = async (id, nm) => {
      closePop();
      const d = await post("/api/compare/add", {id, name: nm, ids}).catch(e => ({error: e.message}));
      if (!d.error) { C.curId = d.id; await loadSets(); }
      onDone?.(d);
    };
    const list = h("div.plist", {}, C.sets.length ? C.sets.map(s => h("button", {type: "button", on: {click: () => add(s.id)}},
      h("span", {text: s.name || t("cmp.untitled")}), h("span.c.mono", {text: String(s.count)})))
      : [h("p.dim", {"data-i18n": "cmp.noSets"})]);
    loadSets().then(() => {
      list.replaceChildren(...(C.sets.length ? C.sets.map(s => h("button", {type: "button", on: {click: () => add(s.id)}},
        h("span", {text: s.name || t("cmp.untitled")}), h("span.c.mono", {text: String(s.count)})))
        : [h("p.dim", {text: t("cmp.noSets")})]));
    });
    return [
      h("div.hd", {}, h("h3.title", {text: t("cmp.addTitle", {n: ids.length})})),
      ids.length > MANY ? h("p.hint.warn", {text: t("cmp.manyAdd", {n: ids.length, max: MANY})}) : null,
      h("div.row", {}, h("span.lbl", {"data-i18n": "cmp.newSet"}),
        h("div.newset", {}, name, create)),
      h("div.row", {}, h("span.lbl", {"data-i18n": "cmp.existing"}), list)];
  });
}

// ── 快速滑桿對照：任選兩張，不必先建比較組 ──
// items：[{id, v, width, height, filename}, …] 兩張。蓋在畫面上，Esc 關閉，←／→ 移動分隔線。
const Q = {el: null, items: [], x: 50, back: null};

export function openSlider(items) {
  if (items.length < 2) return;
  closePop();
  Q.items = items.slice(0, 2);
  Q.x = 50;
  Q.back = document.activeElement;
  Q.el?.remove();
  Q.el = h("div.qslider", {role: "dialog", "aria-modal": "true", "aria-labelledby": "qsTitle"});
  Q.el.addEventListener("click", e => { if (e.target === Q.el || e.target.classList.contains("qstage")) closeSlider(); });
  document.body.append(Q.el);
  document.body.classList.add("viewing");
  drawSlider();
  addEventListener("resize", fitSlider);
}

export function closeSlider() {
  if (!Q.el) return;
  Q.el.remove();
  Q.el = null;
  document.body.classList.remove("viewing");
  removeEventListener("resize", fitSlider);
  Q.back?.focus?.();
}

function drawSlider() {
  const [a, b] = Q.items;
  const src = it => `/api/full?id=${it.id}&v=${encodeURIComponent(it.v)}`;
  const range = h("input", {type: "range", min: "0", max: "100", step: "0.5", value: String(Q.x), "data-i18n-aria": "qs.divider",
                            on: {input: e => setX(Number(e.target.value))}});
  Q.el.replaceChildren(
    h("div.qhd", {},
      h("h2.title", {id: "qsTitle", "data-i18n": "lib.slider"}),
      h("span.qname", {title: a.filename}, h("b", {"data-i18n": "cmp.pickA"}), a.filename || ""),
      h("span.dim", {text: "↔"}),
      h("span.qname", {title: b.filename}, h("b", {"data-i18n": "cmp.pickB"}), b.filename || ""),
      h("span.sp"),
      h("span.dim.qhint", {"data-i18n": "qs.hint"}),
      h("button.pill", {type: "button", "data-i18n": "qs.swap", on: {click: () => { Q.items.reverse(); drawSlider(); }}}),
      h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.toCompare",
                        on: {click: e => {
                          const btn = e.currentTarget;
                          addToCompare(Q.items.map(it => it.id), btn, r => {
                            if (r.error) return;
                            btn.replaceWith(h("button.pill", {type: "button", on: {click: () => { closeSlider(); openCompare(r.id); }}},
                              t("cmp.addedGo")));
                          });
                        }}}),
      h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: closeSlider}}, "✕")),
    h("div.qstage", {},
      h("div.ba.qba", {id: "qsBox"},
        h("img.layer.after", {src: `/api/thumb?id=${b.id}&v=${encodeURIComponent(b.v)}`, alt: "", "data-full": src(b)}),
        h("img.layer.before", {src: `/api/thumb?id=${a.id}&v=${encodeURIComponent(a.v)}`, alt: "", "data-full": src(a)}),
        h("span.lab", {style: "left:10px", "data-i18n": "cmp.pickA"}), h("span.lab", {style: "right:10px", "data-i18n": "cmp.pickB"}),
        h("div.line"), h("div.knob", {text: "⇆"}),
        range)));
  translate(Q.el);
  // 先放縮圖（已在快取），原圖載好再換上
  Q.el.querySelectorAll("img[data-full]").forEach(img => {
    const full = new Image();
    full.onload = () => { if (img.isConnected) img.src = full.src; };
    full.src = img.dataset.full;
  });
  setX(Q.x);
  fitSlider();
  range.focus();
}

function setX(x) {
  Q.x = Math.max(0, Math.min(100, x));
  const box = $("qsBox");
  if (!box) return;
  box.style.setProperty("--x", `${Q.x}%`);
  const r = box.querySelector("input");
  if (Number(r.value) !== Q.x) r.value = String(Q.x);
}

// 依可見範圍算大小（跟看大圖一樣，不靠 CSS 的百分比高度）；比例照左邊那張，右邊那張比例不同時會留邊
function fitSlider() {
  const box = $("qsBox"), stage = Q.el?.querySelector(".qstage");
  if (!box || !stage) return;
  const a = Q.items[0];
  const iw = a.width || 3, ih = a.height || 4;
  const k = Math.min(stage.clientWidth / iw, stage.clientHeight / ih);
  box.style.width = `${Math.round(iw * k)}px`;
  box.style.height = `${Math.round(ih * k)}px`;
}

document.addEventListener("keydown", e => {
  if (!Q.el || e.defaultPrevented) return;
  if (e.key === "Escape") { e.preventDefault(); closeSlider(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    setX(Q.x + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 5));
  }
});

export function openCompare(id) {
  if (id) C.curId = id;
  location.hash = "#compare";
}

// 直接用淘汰賽打開（一批生完自動送來比較之後的「去淘汰賽」）
export function openCup(id) {
  setPref({cmpview: "cup"});
  openCompare(id);
}

// ── 畫面 ──
export function compareScreen() {
  return h("div.cmp", {},
    h("aside.card.csets", {"aria-labelledby": "csetTitle"},
      h("div.csethd", {}, h("h2.title", {id: "csetTitle", "data-i18n": "cmp.sets"}), h("span.sp"),
        h("button.mini", {type: "button", id: "cmpSelMode", "aria-pressed": "false", "data-i18n": "cmp.select",
                          on: {click: () => { L2.selMode = !L2.selMode; L2.picked.clear(); drawSets(); }}})),
      h("div.csetbar", {id: "cmpSetBar"}),
      h("div.clist", {id: "cmpList", role: "listbox", "data-i18n-aria": "cmp.sets"}),
      h("div.cbulk", {id: "cmpBulk", hidden: true})),
    h("section.card.cmain", {id: "cmpMain", "aria-live": "polite"}));
}

export async function showCompare() {
  await loadSets();
  // 從別處打開某一組（例如生成後送去比較）：切到它所在的那一頁
  const want = C.sets.find(s => s.id === C.curId);
  if (want) L2.tab = statusOf(want);
  if (!visible().some(s => s.id === C.curId)) C.curId = visible()[0]?.id || "";
  await loadSet(C.curId);
  drawSets();
  drawMain();
}

function drawSets() {
  const box = $("cmpList");
  if (!box) return;
  const n = k => C.sets.filter(s => statusOf(s) === k).length;
  const bar = $("cmpSetBar");
  if (bar && !bar.contains(document.activeElement)) {        // 打字中不重畫搜尋框
    bar.replaceChildren(...[          // 比較組少的時候沒有搜尋框（null），replaceChildren 不會略過
      h("div.seg", {role: "group", "data-i18n-aria": "cmp.status"}, TABS.map(k => h("button", {
        type: "button", "aria-pressed": String(L2.tab === k),
        on: {click: () => { L2.tab = k; L2.picked.clear(); drawSets(); }}},
        t(`cmp.st.${k}`), h("span.c.mono", {text: String(n(k))})))),
      C.sets.length > 6 ? h("input.find", {type: "search", value: L2.q, "data-i18n-placeholder": "cmp.findSet", "data-i18n-aria": "cmp.findSet",
                                           on: {input: e => { L2.q = e.target.value.trim(); drawSets(); }}}) : null].filter(Boolean));
    translate(bar);
  }
  $("cmpSelMode")?.setAttribute("aria-pressed", String(L2.selMode));
  const list = visible();
  box.toggleAttribute("data-sel", L2.selMode);
  box.replaceChildren(...(list.length ? list.map(s => h("button.cset", {
    type: "button", role: "option", "aria-selected": String(L2.selMode ? L2.picked.has(s.id) : s.id === C.curId),
    on: {click: async () => {
      if (L2.selMode) { L2.picked.has(s.id) ? L2.picked.delete(s.id) : L2.picked.add(s.id); drawSets(); return; }
      await loadSet(s.id); drawSets(); drawMain();
    }}},
    h("span.cmos", {}, s.covers.map(c => h("img", {src: `/api/thumb?id=${c.id}&v=${encodeURIComponent(c.v)}`, alt: "", loading: "lazy"}))),
    h("span.cinfo", {},
      h("b", {text: s.name || t("cmp.untitled")}),
      h("span.dim", {text: [t("cmp.count", {n: s.count}), s.winners ? t("cmp.winners", {n: s.winners}) : ""].filter(Boolean).join(" · ")}))))
    : [h("p.dim", {text: t(C.sets.length ? (L2.q ? "cmp.noMatch" : `cmp.none.${L2.tab}`) : "cmp.empty")})]));
  translate(box);
  drawBulk();
}

// 選取模式底下的動作：封存或取消封存、刪除（按兩次）
function drawBulk() {
  const box = $("cmpBulk");
  if (!box) return;
  box.hidden = !L2.selMode;
  if (!L2.selMode) return;
  const ids = [...L2.picked], k = ids.length;
  const run = async action => {
    await post("/api/compare/bulk", {ids, action}).catch(() => {});
    L2.picked.clear();
    await loadSets();
    if (!C.sets.some(s => s.id === C.curId)) { C.curId = ""; C.cur = null; }
    if (!visible().some(s => s.id === C.curId)) await loadSet(visible()[0]?.id || "");
    drawSets();
    drawMain();
  };
  const del = h("button.pill", {type: "button", disabled: !k, text: t("cmp.bulkDelete", {n: k}), on: {click: e => {
    const b = e.currentTarget;
    if (!L2.arm) {
      b.textContent = t("cmp.bulkDeleteSure", {n: k});
      b.classList.add("armed");
      L2.arm = setTimeout(() => { L2.arm = null; b.textContent = t("cmp.bulkDelete", {n: k}); b.classList.remove("armed"); }, 4000);
      return;
    }
    clearTimeout(L2.arm);
    L2.arm = null;
    run("delete");
  }}});
  box.replaceChildren(
    h("button.pill", {type: "button", "data-i18n": "cmp.pickAll", on: {click: () => {
      const all = visible().map(s => s.id);
      if (all.every(id => L2.picked.has(id))) L2.picked.clear(); else all.forEach(id => L2.picked.add(id));
      drawSets();
    }}}),
    h("span.sp"),
    L2.tab === "archived"
      ? h("button.pill", {type: "button", disabled: !k, text: t("cmp.bulkUnarchive", {n: k}), on: {click: () => run("unarchive")}})
      : h("button.pill", {type: "button", disabled: !k, text: t("cmp.bulkArchive", {n: k}), on: {click: () => run("archive")}}),
    del);
  translate(box);
}

// 目前這組：標成已完成／改回比較中、封存／取消封存。改了之後清單切到它所在的那一頁
async function setStatus(patch) {
  const s = C.cur;
  if (!s) return;
  if (document.activeElement?.classList.contains("cname")) document.activeElement.blur();   // 名稱還在改：先收進來
  await flushSave();
  Object.assign(s, patch);
  await post("/api/compare/update", {id: s.id, ...patch}).catch(() => {});
  await loadSets();
  L2.tab = statusOf(s);
  drawSets();
  drawMain();
}

function drawMain() {
  const main = $("cmpMain");
  if (!main) return;
  const s = C.cur;
  if (!s) {
    main.replaceChildren(h("div.cnone", {}, h("p", {"data-i18n": "cmp.empty"}),
      h("a.pill", {href: "#library", "data-i18n": "cmp.goLibrary"})));
    translate(main);
    return;
  }
  const d = diffs(s.items.filter(it => !it.missing));
  C.view = s.items.length < 2 && viewOf(s) === "slider" ? "side" : viewOf(s);
  // 名稱：點標題直接改
  const nameIn = h("input.cname", {type: "text", value: s.name, "data-i18n-placeholder": "cmp.untitled", "data-i18n-aria": "cmp.name",
                                   "data-i18n-title": "cmp.renameHint",
                                   on: {change: e => { s.name = e.target.value.trim(); save(); },
                                        input: e => fitName(e.currentTarget),
                                        keydown: e => { if (e.key === "Enter") e.currentTarget.blur(); }}});
  // 在比較頁直接加圖（使用者筆記）
  const addBtn = h("button.pill", {type: "button", "data-i18n": "cmp.addImages", "data-i18n-title": "cmp.addImagesHint",
                                   on: {click: () => openPicker({
                                     title: t("cmp.addTo", {name: s.name || t("cmp.untitled")}),
                                     have: s.items.map(it => it.img?.id).filter(Boolean),
                                     onAdd: async ids => {
                                       await flushSave();            // 還沒存的名稱、勝出先存（加完會整組重讀）
                                       const d = await post("/api/compare/add", {id: s.id, ids}).catch(e => ({error: e.message}));
                                       await loadSets();
                                       await loadSet(s.id);
                                       drawSets();
                                       drawMain();
                                       const m = $("cmpMsg");
                                       if (m) { m.textContent = d.error || t("cmp.added", {n: d.added, count: d.count}); m.dataset.kind = d.error ? "err" : "ok"; }
                                     }})}});
  const loupe = h("button.chip", {type: "button", "aria-pressed": String(C.loupe),
                                  on: {click: e => { C.loupe = !C.loupe; e.currentTarget.setAttribute("aria-pressed", String(C.loupe)); resetZoom(); }}},
    t("cmp.loupe"));
  const del = h("button.ghost", {type: "button", "data-i18n": "cmp.delete", on: {click: e => delSet(e.currentTarget)}});
  const doneBtn = h("button.ghost", {type: "button", "data-i18n": s.done ? "cmp.markActive" : "cmp.markDone",
                                     on: {click: () => setStatus({done: !s.done})}});
  const archBtn = h("button.ghost", {type: "button", "data-i18n": s.archived ? "cmp.unarchive" : "cmp.archive",
                                     on: {click: () => setStatus({archived: !s.archived})}});
  // replaceChildren 不會略過 null（會變成一段「null」文字），所以先濾掉
  main.replaceChildren(...[
    h("div.chd", {},
      nameIn,
      h("span.dim.mono", {text: t("cmp.count", {n: s.items.length})}),
      addBtn,
      h("span.sp"),
      h("div.seg", {role: "group"}, ["side", "grid", "slider", "cup"].map(v => h("button", {
        type: "button", "aria-pressed": String(C.view === v), "data-i18n": `cmp.view.${v}`,
        disabled: v === "slider" && s.items.length < 2,
        on: {click: () => { setPref({cmpview: v}); drawMain(); }}}))),
      C.view !== "slider" && C.view !== "cup" ? loupe : null,
      h("button.pill.primary", {type: "button", "data-i18n": "cmp.export", disabled: !s.items.some(it => !it.missing),
                                on: {click: e => exportImage(e.currentTarget)}}),
      doneBtn, archBtn, del,
      h("span.msg.cmsg", {id: "cmpMsg", role: "status"})),
    s.items.length > MANY ? h("p.cmany", {text: t("cmp.many", {n: s.items.length, max: MANY})}) : null,
    // 有圖但全部找不到：滑桿、淘汰賽這時是空的，說一下為什麼
    s.items.length && !s.items.some(it => !it.missing) ? h("p.cmany", {"data-i18n": "cmp.allMissing"}) : null,
    s.items.length ? (C.view === "slider" ? sliderView(s) : C.view === "grid" ? gridView(s, d)
      : C.view === "cup" ? cupView(s) : sideView(s, d))
      : h("p.dim", {"data-i18n": "cmp.noItems"}),
    s.items.length > 1 && C.view !== "cup" ? diffTable(s, d) : null].filter(Boolean));
  translate(main);
  fitName(nameIn);
}

// 並排：每張同高，底下列出「這張跟別張哪裡不一樣」
function sideView(s, d) {
  const ok = s.items.filter(it => !it.missing);
  const strip = h("div.cstrip", {});
  s.items.forEach((it, i) => {
    const k = ok.indexOf(it);
    const im = it.img;
    const fig = it.missing ? h("div.cmiss", {text: t("cmp.missing")})
      : h("div.cpic", {},
          h("img", {src: `/api/thumb?id=${im.id}&v=${encodeURIComponent(im.v)}`, alt: "", "data-full": `/api/full?id=${im.id}&v=${encodeURIComponent(im.v)}`,
                    style: im.width && im.height ? `aspect-ratio:${im.width} / ${im.height}` : null}));
    if (!it.missing) {
      const full = new Image();
      full.onload = () => { const img = fig.querySelector("img"); if (img) img.src = full.src; };
      full.src = `/api/full?id=${im.id}&v=${encodeURIComponent(im.v)}`;
    }
    const col = h("figure.ccol", {"data-win": it.winner ? "1" : null},
      h("div.cframe", {}, fig,
        h("span.clab", {text: label(i)}),
        it.winner ? h("span.tape.win", {"data-i18n": "cmp.winner"}) : null),
      h("figcaption", {},
        h("div.cchips", {}, (k >= 0 ? d.per[k] : []).map(x => h("span.cchip", {text: x}))),
        actions(s, it, i)));
    strip.append(col);
  });
  loupeOn(strip);
  return strip;
}

// 每張圖的動作：勝出、看大圖、存成 Recipe、左右移、拿掉
function actions(s, it, i) {
  return h("div.cacts", {},
    h("button.mini.tog", {type: "button", "aria-pressed": String(it.winner), "data-i18n": "cmp.winner",
                          on: {click: () => { it.winner = !it.winner; save(); drawMain(); }}}),
    it.missing ? null : h("button.mini", {type: "button", "data-i18n": "cmp.view", on: {click: () => viewAt(it)}}),
    it.missing ? null : h("button.mini", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false",
                                          "data-i18n": "rcp.saveShort", "data-i18n-title": "rcp.saveImageHint",
                                          on: {click: e => {
                                            const btn = e.currentTarget;
                                            saveFromImagePop(btn, it.img, () => { btn.textContent = t("rcp.savedShort"); btn.disabled = true; });
                                          }}}),
    h("button.mini", {type: "button", "data-i18n-aria": "cmp.left", "data-i18n-title": "cmp.left", disabled: i === 0,
                      on: {click: () => move(i, -1)}}, "←"),
    h("button.mini", {type: "button", "data-i18n-aria": "cmp.right", "data-i18n-title": "cmp.right", disabled: i === s.items.length - 1,
                      on: {click: () => move(i, 1)}}, "→"),
    h("button.mini", {type: "button", "data-i18n": "cmp.remove", on: {click: () => { s.items.splice(i, 1); save(); drawMain(); }}}));
}

// 同步放大鏡：游標在哪張圖的哪個位置，每張都放大同一個位置（並排與格狀共用）。
// 格狀平常用縮圖，開放大鏡時才換成原圖，不然放大後是糊的
function loupeOn(box) {
  box.addEventListener("mousemove", e => {
    if (!C.loupe) return;
    const img = e.target.closest(".cpic")?.querySelector("img");
    if (!img) return;
    const r = img.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width * 100, fy = (e.clientY - r.top) / r.height * 100;
    box.querySelectorAll(".cpic img").forEach(x => {
      if (x.dataset.full && x.src !== x.dataset.full && !x.dataset.fullAsked) { x.dataset.fullAsked = "1"; x.src = x.dataset.full; }
      x.style.transformOrigin = `${fx}% ${fy}%`;
      x.style.transform = `scale(${C.zoom})`;
    });
  });
  box.addEventListener("mouseleave", resetZoom);
  box.addEventListener("wheel", e => {
    if (!C.loupe || !e.target.closest(".cpic")) return;
    e.preventDefault();
    C.zoom = Math.max(1.5, Math.min(8, C.zoom * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
    box.querySelectorAll(".cpic img").forEach(x => { x.style.transform = `scale(${C.zoom})`; });
  }, {passive: false});
}

function resetZoom() {
  document.querySelectorAll("#cmpMain .cpic img").forEach(x => { x.style.transform = ""; });
}

// ── 格狀：多排排列。只差一兩個參數時排成 XY 對照表，差異變成表頭 ──
// 每張底下只放最關鍵的兩個差異；游標停在圖上或點一下，下面的詳細列顯示完整差異與動作。
function gridView(s, d) {
  const ok = s.items.filter(it => !it.missing);
  const dims = dimsOf(ok);
  const lay = pref("cmpxy", true) ? xyLayout(dims, ok.length, {swap: C.swap}) : null;
  const canXy = !!xyLayout(dims, ok.length);
  const size = GRID_SIZES[pref("cmpsize", "m")] || GRID_SIZES.m;
  if (!s.items.some(it => it.path === C.active)) C.active = s.items[0]?.path || "";
  const detail = h("div.cdetail", {"aria-live": "polite"});
  const dimOf = key => dims.find(x => x.key === key);
  const head = (key, v) => `${t(dimOf(key).lab)} ${v || "—"}`;

  const pick = it => {
    C.active = it.path;
    box.querySelectorAll(".gcard").forEach(x => x.setAttribute("aria-selected", String(x.dataset.path === it.path)));
    drawDetail(s, d, detail);
  };
  const card = it => {
    const i = s.items.indexOf(it), k = ok.indexOf(it), im = it.img;
    // XY 模式下軸上的差異已經在表頭，格子裡只寫其餘的（通常是隨機 seed）
    const chips = it.missing ? [] : lay ? lay.noise.map(key => head(key, dimOf(key).vals[k])) : d.per[k] || [];
    const shown = chips.slice(0, 2), more = chips.length - shown.length;
    const el = h("figure.gcard", {"data-win": it.winner ? "1" : null, "data-path": it.path, tabindex: "0",
                                  "aria-selected": String(it.path === C.active), title: im?.filename || ""},
      h("div.cframe", {},
        it.missing ? h("div.cmiss", {text: t("cmp.missing")})
          : h("div.cpic", {}, h("img", {src: `/api/thumb?id=${im.id}&v=${encodeURIComponent(im.v)}`, alt: "", loading: "lazy",
                                        "data-full": `/api/full?id=${im.id}&v=${encodeURIComponent(im.v)}`,
                                        style: im.width && im.height ? `aspect-ratio:${im.width} / ${im.height}` : null})),
        h("span.clab", {text: label(i)}),
        it.winner ? h("span.tape.win", {"data-i18n": "cmp.winner"}) : null,
        removeBtn(s, it)),
      chips.length ? h("figcaption.gchips", {},
        shown.map(x => h("span.cchip", {text: x})),
        more > 0 ? h("span.cchip.more", {text: t("cmp.moreChips", {n: more})}) : null) : null);
    el.addEventListener("mouseenter", () => pick(it));
    el.addEventListener("focus", () => pick(it));
    el.addEventListener("click", () => pick(it));
    el.addEventListener("dblclick", () => { if (!it.missing) viewAt(it); });
    el.addEventListener("keydown", e => { if (e.key === "Enter" && !it.missing) { e.preventDefault(); viewAt(it); } });
    return el;
  };

  let box;
  if (lay) {
    const cols = lay.x.values.length, rows = lay.y ? lay.y.values.length : 1;
    box = h("div.cxy", {style: `--cw:${size}px;grid-template-columns:${lay.y ? "max-content " : ""}repeat(${cols}, var(--cw))`},
      lay.y ? h("div.cxcorner", {}, h("span", {text: `${t(dimOf(lay.x.key).lab)} →`}), h("span", {text: `${t(dimOf(lay.y.key).lab)} ↓`})) : null,
      lay.x.values.map(v => h("div.cxhead", {text: head(lay.x.key, v), title: head(lay.x.key, v)})),
      Array.from({length: rows}, (_, yi) => [
        lay.y ? h("div.cyhead", {text: head(lay.y.key, lay.y.values[yi]), title: head(lay.y.key, lay.y.values[yi])}) : null,
        Array.from({length: cols}, (_, xi) => h("div.cxcell", {},
          lay.cells[yi * cols + xi].map(n => card(ok[n])))),
      ]).flat(2));
  } else {
    box = h("div.cgrid", {style: `--cw:${size}px`}, s.items.map(card));
  }
  loupeOn(box);
  const tools = h("div.ctools", {},
    h("button.chip", {type: "button", "aria-pressed": String(!!lay), disabled: !canXy,
                      title: canXy ? null : t("cmp.xyOff"),
                      on: {click: () => { setPref({cmpxy: !pref("cmpxy", true)}); drawMain(); }}}, t("cmp.xy")),
    lay?.y ? h("button.chip", {type: "button", "data-i18n": "cmp.swap", on: {click: () => { C.swap = !C.swap; drawMain(); }}}) : null,
    lay ? h("span.dim", {text: lay.noise.length ? t("cmp.xyNoise", {what: lay.noise.map(k => t(dimOf(k).lab)).join("、")}) : ""}) : null,
    h("span.sp"),
    h("div.seg", {role: "group", "data-i18n-aria": "cmp.size"}, Object.keys(GRID_SIZES).map(k => h("button", {
      type: "button", "aria-pressed": String(pref("cmpsize", "m") === k), "data-i18n": `wall.size.${k}`,
      on: {click: () => { setPref({cmpsize: k}); drawMain(); }}}))));
  drawDetail(s, d, detail);
  return h("div.cgridwrap", {}, tools, h("div.cgscroll", {}, box), detail);
}

// 格狀的詳細列：目前選到的那張的完整差異與動作
function drawDetail(s, d, box) {
  const it = s.items.find(x => x.path === C.active);
  if (!it) { box.replaceChildren(h("p.dim", {"data-i18n": "cmp.detailHint"})); translate(box); return; }
  const ok = s.items.filter(x => !x.missing), i = s.items.indexOf(it), k = ok.indexOf(it);
  box.replaceChildren(
    h("div.cdhd", {},
      h("b.clab.static", {text: label(i)}),
      h("span.dim", {text: it.missing ? t("cmp.missing") : it.img.filename, title: it.path}),
      h("span.sp"),
      actions(s, it, i)),
    h("div.cchips", {}, (k >= 0 ? d.per[k] : []).map(x => h("span.cchip", {text: x}))));
  translate(box);
}

// 名稱框跟著字的長度（後面的張數才不會離很遠）。Firefox 還沒有 field-sizing，用一個看不見的 span 量
function fitName(input) {
  if (!input?.isConnected) return;
  const m = h("span.cname.cmeasure", {text: input.value || input.placeholder || " "});
  input.after(m);
  input.style.width = `${Math.ceil(m.getBoundingClientRect().width / (input.getBoundingClientRect().width / input.offsetWidth || 1)) + 4}px`;
  m.remove();
}

// 卡片右上角的 ✕：從這組拿掉（圖本身不動）。不觸發卡片的點一下（選取、標晉級）
function removeBtn(s, it) {
  return h("button.cdel", {type: "button", "data-i18n-aria": "cmp.remove", "data-i18n-title": "cmp.removeHint",
                           on: {click: e => {
                             e.stopPropagation();
                             const i = s.items.indexOf(it);
                             if (i < 0) return;
                             s.items.splice(i, 1);
                             save();
                             drawMain();
                           }, dblclick: e => e.stopPropagation(), keydown: e => e.stopPropagation()}}, "✕");
}

function move(i, dir) {
  const a = C.cur.items;
  [a[i], a[i + dir]] = [a[i + dir], a[i]];
  save();
  drawMain();
}

function viewAt(it) {
  const ok = C.cur.items.filter(x => !x.missing);
  const pseudo = {items: ok.map(x => ({id: x.img.id, v: x.img.v, width: x.img.width, height: x.img.height, filename: x.img.filename})),
                  total: ok.length, offset: ok.length, loadMore: async () => {}, nodes: new Map()};
  openViewer(pseudo, it.img.id);
}

// ── 淘汰賽 ──
// 上面是晉級路線（點前面的輪次可以回去重選），中間是這一輪還在的圖：點一下標晉級、點兩下看大圖，
// 數字鍵 1～9 也能標、Enter 下一輪。剩兩張時上面多一個滑桿對照。覺得夠了就「就選這些」
function cupView(s) {
  const live = alive(s), R = roundOf(s);
  const size = GRID_SIZES[pref("cmpsize", "m")] || GRID_SIZES.m;
  const redraw = () => { save(); drawMain(); };
  const steps = route(s);
  const path = h("div.croute", {}, ...steps.flatMap(({r, n}, i) => [
    i ? h("span.dim", {text: "→"}) : null,
    h("button.chip", {type: "button", "aria-pressed": String(r === R && !s.done), disabled: r === R && !s.done,
                      title: r < R || s.done ? t("cmp.cup.back") : null,
                      on: {click: () => { back(s, r); redraw(); }}}, t("cmp.cup.round", {r, n}))]).filter(Boolean),
    s.done ? [h("span.dim", {text: "→"}), h("b.cwon", {text: t("cmp.cup.won", {n: s.items.filter(it => it.winner).length})})] : null);
  const card = (it, k) => {
    const im = it.img, i = s.items.indexOf(it);
    const el = h("figure.gcard.cupcard", {"data-adv": it.adv ? "1" : null, "data-win": s.done && it.winner ? "1" : null, tabindex: "0",
                                          title: im.filename},
      h("div.cframe", {},
        h("div.cpic", {}, h("img", {src: `/api/thumb?id=${im.id}&v=${encodeURIComponent(im.v)}`, alt: "", loading: "lazy",
                                    "data-full": `/api/full?id=${im.id}&v=${encodeURIComponent(im.v)}`,
                                    style: im.width && im.height ? `aspect-ratio:${im.width} / ${im.height}` : null})),
        h("span.clab", {text: k < 9 ? `${k + 1}` : label(i)}),
        s.done ? (it.winner ? h("span.tape.win", {"data-i18n": "cmp.winner"}) : null)
          : it.adv ? h("span.tape.win", {"data-i18n": "cmp.cup.adv"}) : null,
        removeBtn(s, it)));
    if (!s.done) {
      el.addEventListener("click", () => toggleAdv(it));
      el.addEventListener("keydown", e => { if (e.key === " ") { e.preventDefault(); toggleAdv(it); } });
    }
    el.addEventListener("dblclick", () => viewAt(it));
    return el;
  };
  const shown = s.done ? s.items.filter(it => it.winner && !it.missing) : live;
  const grid = h("div.cgrid.cupgrid", {style: `--cw:${size}px`}, shown.map(card));
  loupeOn(grid);
  const picked = live.filter(it => it.adv).length;
  const bar = s.done
    ? h("div.ccupbar", {},
      h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "cmp.cup.reopen", on: {click: () => { back(s, R); redraw(); }}}),
      h("button.pill", {type: "button", "data-i18n": "cmp.cup.restart", on: {click: () => { restart(s); redraw(); }}}))
    : h("div.ccupbar", {},
      h("b", {id: "cupPicked", text: t("cmp.cup.picked", {k: picked, n: live.length})}),
      h("span.dim", {text: t("cmp.cup.keys")}),
      h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "cmp.cup.finish",
                        on: {click: () => { finish(s); redraw(); }}}),
      h("button.pill.primary", {type: "button", id: "cupNext", "data-i18n": "cmp.cup.next", disabled: !canAdvance(s),
                                on: {click: () => { if (advance(s)) redraw(); }}}));
  const sizes = h("div.seg", {role: "group", "data-i18n-aria": "cmp.size"}, Object.keys(GRID_SIZES).map(k => h("button", {
    type: "button", "aria-pressed": String(pref("cmpsize", "m") === k), "data-i18n": `wall.size.${k}`,
    on: {click: () => { setPref({cmpsize: k}); drawMain(); }}})));
  return h("div.cgridwrap.cupwrap", {},
    h("div.ctools", {}, path, h("span.sp"), sizes),
    bar,
    !s.done && live.length === 2 ? duel(live[0], live[1]) : null,
    h("div.cgscroll", {}, grid));
}

// 標晉級：只改那張卡片與上面的計數，不整頁重畫（格子一多，重畫會跳回最上面）
function toggleAdv(it) {
  const s = C.cur;
  it.adv = !it.adv;
  save();
  const k = alive(s).indexOf(it);
  const el = [...document.querySelectorAll(".cupgrid .cupcard")][k];
  if (el) {
    el.toggleAttribute("data-adv", !!it.adv);
    el.querySelector(".tape")?.remove();
    if (it.adv) {
      el.querySelector(".cframe").append(h("span.tape.win", {text: t("cmp.cup.adv")}));
    }
  }
  const live = alive(s), n = $("cupPicked"), next = $("cupNext");
  if (n) n.textContent = t("cmp.cup.picked", {k: live.filter(x => x.adv).length, n: live.length});
  if (next) next.disabled = !canAdvance(s);
}

// 剩兩張：左右拖曳對照，底下的兩張照樣點一下標晉級
function duel(a, b) {
  const src = it => `/api/full?id=${it.img.id}&v=${encodeURIComponent(it.img.v)}`;
  const ar = a.img.width && a.img.height ? `${a.img.width} / ${a.img.height}` : "3 / 4";
  const stage = h("div.ba.cba", {style: `aspect-ratio:${ar}`},
    h("img.layer.after", {src: src(b), alt: ""}),
    h("img.layer.before", {src: src(a), alt: ""}),
    h("span.lab", {style: "left:10px", text: "1"}), h("span.lab", {style: "right:10px", text: "2"}),
    h("div.line"), h("div.knob", {text: "⇆"}),
    h("input", {type: "range", min: "0", max: "100", value: "50", "data-i18n-aria": "cmp.view.slider",
                on: {input: e => stage.style.setProperty("--x", `${e.target.value}%`)}}));
  return h("div.cduel", {}, stage);
}

document.addEventListener("keydown", e => {
  if (location.hash !== "#compare" || C.view !== "cup" || !C.cur || C.cur.done || e.defaultPrevented) return;
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.("input, textarea, select, [contenteditable]")) return;
  if (document.body.classList.contains("viewing")) return;
  const live = alive(C.cur);
  if (/^[1-9]$/.test(e.key) && live[Number(e.key) - 1]) {
    e.preventDefault();
    toggleAdv(live[Number(e.key) - 1]);
  } else if (e.key === "Enter" && !e.target.closest?.("button, a, figure") && advance(C.cur)) {
    e.preventDefault();
    save();
    drawMain();
  }
});

// 兩張滑桿：挑兩張左右拖曳對照
function sliderView(s) {
  const ok = s.items.map((it, i) => [it, i]).filter(([it]) => !it.missing);
  const pick = (key, val) => h("select.chip", {"data-i18n-aria": key, on: {change: e => { C[key] = Number(e.target.value); drawMain(); }}},
    ok.map(([it, i]) => h("option", {value: String(i), text: `${label(i)} · ${it.img.filename}`, selected: i === val})));
  const A = s.items[C.pickA], B = s.items[C.pickB];
  const src = it => (it && !it.missing ? `/api/full?id=${it.img.id}&v=${encodeURIComponent(it.img.v)}` : "");
  const ar = A?.img?.width && A?.img?.height ? `${A.img.width} / ${A.img.height}` : "3 / 4";
  const stage = h("div.ba.cba", {style: `aspect-ratio:${ar}`},
    h("img.layer.after", {src: src(B), alt: ""}),
    h("img.layer.before", {src: src(A), alt: ""}),
    h("span.lab", {style: "left:10px", text: label(C.pickA)}), h("span.lab", {style: "right:10px", text: label(C.pickB)}),
    h("div.line"), h("div.knob", {text: "⇆"}),
    h("input", {type: "range", min: "0", max: "100", value: "50", "data-i18n-aria": "cmp.view.slider",
                on: {input: e => stage.style.setProperty("--x", `${e.target.value}%`)}}));
  return h("div.cslider", {},
    h("div.cpick", {}, pick("pickA", C.pickA), h("span.dim", {text: "↔"}), pick("pickB", C.pickB)),
    h("div.cstage", {}, stage));
}

function diffTable(s, d) {
  const ok = s.items.filter(it => !it.missing);
  // 預設收起來：每張底下已經寫了差異，表格是要逐項對照時才打開
  return h("details.cdiff", {open: C.diffOpen, on: {toggle: e => { C.diffOpen = e.currentTarget.open; }}},
    h("summary", {}, t("cmp.diffSummary", {d: d.diff.length, s: d.same.length})),
    d.diff.length ? h("div.ctable", {},
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}), ok.map(it => h("th", {text: label(s.items.indexOf(it))})))),
        h("tbody", {}, d.diff.map(([lab, vals]) => h("tr", {}, h("th", {"data-i18n": lab}), vals.map(v => h("td", {text: v}))))))) : null,
    d.same.length ? h("p.csame", {}, h("b", {"data-i18n": "cmp.same"}), " ",
      d.same.map(([lab, v], i) => [i ? " · " : "", h("span", {"data-i18n": lab}), ` ${v}`]).flat()) : null);
}

async function delSet(btn) {
  if (!C.delArm) {
    btn.textContent = t("cmp.deleteSure");
    btn.classList.add("armed");
    C.delArm = setTimeout(() => { C.delArm = null; btn.textContent = t("cmp.delete"); btn.classList.remove("armed"); }, 4000);
    return;
  }
  clearTimeout(C.delArm);
  C.delArm = null;
  await post("/api/compare/delete", {id: C.cur.id}).catch(() => {});
  C.curId = "";
  await showCompare();
}

// ── 匯出比較圖 ──
// 用 canvas 把整組排成一張。格狀而且開著 XY 對照表時，照畫面上的表匯出：最上面是欄的表頭、左邊是列的表頭，
// 每格放那一格的圖（只差 seed 這類的好幾張就並排），圖底下寫代號與表頭以外的差異。
// 其他排法：每格一張圖（等寬），底下是代號與差異。兩種最上面都是組名與共同參數。
async function exportImage(btn) {
  const s = C.cur, ok = s.items.filter(it => !it.missing);
  if (!ok.length) return;
  btn.disabled = true;
  btn.textContent = t("cmp.exporting");
  const msg = $("cmpMsg");
  if (msg) { msg.textContent = ""; msg.dataset.kind = ""; }
  try {
    const d = diffs(ok);
    const dims = dimsOf(ok);
    const lay = C.view === "grid" && pref("cmpxy", true) ? xyLayout(dims, ok.length, {swap: C.swap}) : null;
    const css = getComputedStyle(document.documentElement);
    const col = k => css.getPropertyValue(k).trim() || "#888";
    const title = "\"LXGW WenKai TC\", \"Microsoft JhengHei UI\", sans-serif";
    const body = "\"Microsoft JhengHei UI\", \"PingFang TC\", sans-serif";
    await document.fonts.ready;
    const imgs = await Promise.all(ok.map(it => new Promise(res => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => res(null);
      im.src = `/api/full?id=${it.img.id}&v=${encodeURIComponent(it.img.v)}`;
    })));
    const cv = document.createElement("canvas");
    const ctx = cv.getContext("2d");
    // 換行以「一項」為單位（例如「seed 123」「+1girl」），一項本身太長才逐字切
    const SEP = "  ·  ";
    const wrap = (parts, font, maxW) => {
      ctx.font = font;
      const lines = [];
      let line = "";
      const fits = x => ctx.measureText(x).width <= maxW;
      for (const part of parts) {
        const next = line ? line + SEP + part : part;
        if (fits(next)) { line = next; continue; }
        if (line) lines.push(line);
        line = "";
        for (const ch of part) {
          if (!fits(line + ch) && line) { lines.push(line); line = ch; } else line += ch;
        }
      }
      if (line) lines.push(line);
      return lines;
    };
    const capFont = `22px ${body}`, headFont = `700 24px ${body}`;
    const W = 640, PAD = 32, GAP = 24;
    const cellH = Math.round(W * Math.max(...ok.map(it => (it.img.height || 4) / (it.img.width || 3))));
    const dimOf = key => dims.find(x => x.key === key);
    const head = (key, v) => `${t(dimOf(key).lab)} ${v || "—"}`;

    // 一張圖：底色、圖、代號、勝出、底下的字
    const drawCell = (it, im, x, y, lines, capH) => {
      ctx.fillStyle = col("--p1");
      ctx.fillRect(x, y, W, cellH + capH);
      if (im) {
        const k = Math.min(W / im.naturalWidth, cellH / im.naturalHeight);
        const w = im.naturalWidth * k, hh = im.naturalHeight * k;
        ctx.drawImage(im, x + (W - w) / 2, y + (cellH - hh) / 2, w, hh);
      }
      ctx.fillStyle = col("--tx");
      ctx.font = `700 30px ${title}`;
      ctx.fillText(label(s.items.indexOf(it)), x + 16, y + cellH + 14);
      if (it.winner) {
        // 勝出標在圖的右上角，不跟底下的說明搶位置
        ctx.font = `700 24px ${title}`;
        const tag = `★ ${t("cmp.winner")}`, tw = ctx.measureText(tag).width + 28;
        ctx.fillStyle = col("--ac");
        ctx.fillRect(x + W - tw - 12, y + 12, tw, 40);
        ctx.fillStyle = col("--ink");
        ctx.fillText(tag, x + W - tw + 2, y + 20);
      }
      ctx.fillStyle = col("--dm");
      ctx.font = capFont;
      lines.forEach((l, j) => ctx.fillText(l, x + 60, y + cellH + 18 + j * 30));
    };
    // 最上面：組名、共同參數（XY 時再加一句「每格不同的」）
    const drawTitle = (lines) => {
      ctx.fillStyle = col("--tx");
      ctx.font = `700 40px ${title}`;
      ctx.textBaseline = "top";
      ctx.fillText(s.name || t("cmp.untitled"), PAD, PAD);
      ctx.font = capFont;
      ctx.fillStyle = col("--dm");
      lines.forEach((l, i) => ctx.fillText(l, PAD, PAD + 62 + i * 30));
    };
    // 版面照原尺寸算；超過瀏覽器 canvas 的上限就整張等比例縮小再畫（字也一起縮）
    const begin = (fullW, fullH) => {
      const k = canvasScale(fullW, fullH);
      cv.width = Math.floor(fullW * k);
      cv.height = Math.floor(fullH * k);
      ctx.setTransform(k, 0, 0, k, 0, 0);      // 改 canvas 大小會重設狀態，所以放在這之後
      ctx.fillStyle = col("--bg");
      ctx.fillRect(0, 0, cv.width, cv.height);
      return k;
    };

    let k;
    if (lay) {
      // XY 對照表：格子裡只寫表頭以外的差異（通常是隨機 seed）
      const caps = ok.map((it, n) => wrap(lay.noise.map(key => head(key, dimOf(key).vals[n])), capFont, W - 60).slice(0, 4));
      const capH = 30 + Math.max(1, ...caps.map(c => c.length)) * 30;
      const HEAD_W = 300;
      const probe = xyExportLayout(lay, {cellW: W, cellH, capH, gap: GAP, pad: PAD, titleH: 0, headH: 0, headW: HEAD_W});
      const colHeads = probe.cols.map((c, xi) => wrap([head(lay.x.key, lay.x.values[xi])], headFont, c.w - 24).slice(0, 3));
      const headH = 24 + Math.max(...colHeads.map(c => c.length)) * 32;
      const rowHeads = lay.y ? lay.y.values.map(v => wrap([head(lay.y.key, v)], headFont, HEAD_W - 24).slice(0, 8)) : [];
      const sameParts = d.same.map(([lab, v]) => `${t(lab)} ${v}`);
      if (lay.noise.length) sameParts.push(t("cmp.xyNoise", {what: lay.noise.map(key => t(dimOf(key).lab)).join("、")}));
      const titleLines = wrap(sameParts, capFont, probe.W - PAD * 2);
      const titleH = 70 + titleLines.length * 30 + 16;
      const L = xyExportLayout(lay, {cellW: W, cellH, capH, gap: GAP, pad: PAD, titleH, headH, headW: HEAD_W});
      k = begin(L.W, L.H);
      drawTitle(titleLines);
      // 左上角：哪一軸是欄、哪一軸是列
      if (lay.y) {
        ctx.fillStyle = col("--dm");
        ctx.font = headFont;
        ctx.fillText(`${t(dimOf(lay.x.key).lab)} →`, PAD + 12, L.headY + 12);
        ctx.fillText(`${t(dimOf(lay.y.key).lab)} ↓`, PAD + 12, L.headY + 12 + 32);
      }
      ctx.font = headFont;
      L.cols.forEach((c, xi) => {
        ctx.fillStyle = col("--p2");
        ctx.fillRect(c.x, L.headY, c.w, headH);
        ctx.fillStyle = col("--tx");
        colHeads[xi].forEach((l, j) => ctx.fillText(l, c.x + 12, L.headY + 12 + j * 32));
      });
      if (lay.y) L.rows.forEach((y, yi) => {
        ctx.fillStyle = col("--p2");
        ctx.fillRect(PAD, y, HEAD_W, cellH + capH);
        ctx.fillStyle = col("--tx");
        rowHeads[yi].forEach((l, j) => ctx.fillText(l, PAD + 12, y + 12 + j * 32));
      });
      L.items.forEach(({n, x, y}) => drawCell(ok[n], imgs[n], x, y, caps[n], capH));
    } else {
      const n = ok.length, cols = n <= 4 ? n : Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      const caps = d.per.map(p => wrap(p, capFont, W - 60).slice(0, 6));
      const capH = 30 + Math.max(1, ...caps.map(c => c.length)) * 30;
      const headLines = wrap(d.same.map(([lab, v]) => `${t(lab)} ${v}`), capFont, cols * W + (cols - 1) * GAP);
      const headH = 70 + headLines.length * 30 + 16;
      const fullW = PAD * 2 + cols * W + (cols - 1) * GAP;
      const fullH = PAD * 2 + headH + rows * (cellH + capH) + (rows - 1) * GAP;
      k = begin(fullW, fullH);
      drawTitle(headLines);
      ok.forEach((it, i) => drawCell(it, imgs[i], PAD + (i % cols) * (W + GAP), PAD + headH + Math.floor(i / cols) * (cellH + capH + GAP),
                                     caps[i], capH));
    }
    const blob = await new Promise(res => cv.toBlob(res, "image/png"));
    // 還是太大（記憶體不夠之類）時 toBlob 會給 null，不說的話按了等於沒反應
    if (!blob) throw new Error(t("cmp.exportTooBig", {w: cv.width, h: cv.height}));
    const a = h("a", {href: URL.createObjectURL(blob),
                      download: `compare-${(s.name || s.id).replace(/[\\/:*?"<>|]+/g, "_")}${lay ? "-xy" : ""}.png`});
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    if (msg) {
      const what = lay ? t("cmp.exportedXy") + " · " : "";
      msg.textContent = what + (k < 1 ? t("cmp.exportScaled", {pct: Math.round(k * 100), w: cv.width, h: cv.height})
                                       : t("cmp.exported", {w: cv.width, h: cv.height}));
      msg.dataset.kind = k < 1 ? "warn" : "ok";
    }
  } catch (e) {
    if (msg) { msg.textContent = t("cmp.exportErr", {msg: e.message || e}); msg.dataset.kind = "err"; }
  } finally {
    btn.disabled = false;
    btn.textContent = t("cmp.export");
  }
}
