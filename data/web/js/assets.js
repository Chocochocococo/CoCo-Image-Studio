// 資源庫：模型、LoRA、VAE、文字編碼器、Embeddings。從生成台的資源格打開。
//
// 左邊是標題、分頁、搜尋與篩選（使用者筆記 #4：放上面太佔高度），中間是卡片牆，右邊是詳細欄。
// 點一張卡就是「用這個」：模型、VAE、編碼器直接換；LoRA 加進提示詞（再點一次拿掉），
// 卡片上可以直接調權重。提示詞是 LoRA 唯一的真相，這裡只是改那串文字。
// 右鍵（或 Shift+Enter）只選起來看詳細、不套用；正在用的另外排成牆上方的一列小圖，
// 牆本身的順序不跟著變（筆記 #3：以前用了就跳到第一格，要改資料得先找回來）。
// 右邊的詳細欄：大圖（可以換封面，筆記 #9）、觸發詞（點了加進提示詞）、個人備註、收藏、在圖庫看用過的圖；
// 模型另外有「品質標籤與預設負面」（生成時自動加上，見 server/presets.py）。
// Embeddings 跟 A1111 一樣是寫名字：點卡片加進生成台正在編輯的那一邊（正面或負面），再點一次拿掉。
// 放錯分頁的檔案可以在詳細欄改分類（server/resources.py 的 RES_KIND）。
// Recipe（recipelib.js）與標籤（taglib.js，你的標籤庫）不是檔案，工具列、牆、詳細欄都交給它們畫。
// 多視窗（sync.js）：⧉ 把資源庫開成獨立視窗（網址 #assets-種類，整頁模式），放在另一個螢幕常駐；
// 在那裡點卡片一樣改生成台的設定（經 generate.js 存檔、廣播給主視窗）。

import {get, post} from "./api.js";
import {$, h, stem} from "./dom.js";
import {editingField, genSettings, refreshPreset, reloadSettings, setField, setPromptText} from "./generate.js";
import {lang, t, translate} from "./i18n.js";
import {showInLibrary} from "./library.js";
import {ARCH_LABEL, addEmb, addLora, addTrigger, fmtWeight, hasEmb, hasLora, hasTrigger, loraTags, removeEmb, removeLora,
        setLoraWeight} from "./options.js";
import {closePop, openPop} from "./pop.js";
import {multiEnabled, on, openWindow, send} from "./sync.js";
import {info} from "./hint.js";
import {R as RCP} from "./recipes.js";
import {bindRecipeTab, loadRecipeTab, markRecipeSel, recipeDetail, recipeGrid, recipeInUse, recipeTools} from "./recipelib.js";
import {bindTagTab, loadTagTab, tagDetail, tagGrid, tagInUse, tagTools} from "./taglib.js";
import {keyedDelay} from "./inflight.js";

const KINDS = ["model", "lora", "vae", "te", "embedding"];
const TABS = [...KINDS, "recipe", "tags"];
const isRecipe = () => A.kind === "recipe";
const isTags = () => A.kind === "tags";
const SIDES = ["PROMPT", "NEGATIVE"];
const SORTS = ["name", "recent", "uses", "size"];
// 架構代碼 → 顯示名稱。sd＝SD 系列通用（VAE、CLIP），qwenimage＝Anima 與 Krea 2 共用的 VAE，
// flux2＝FLUX.2 klein 4B、9B 共用的 VAE，flux1＝FLUX.1 的 LoRA（dev、schnell 共用），clipl＝CLIP-L（SD 系列與 FLUX.1），
// t5＝T5-XXL（FLUX.1 的文字編碼器），partial＝CoCo 還不能載入的單檔
const ARCH_NAME = {...ARCH_LABEL, sd: "SD", qwenimage: "Qwen Image", flux2: "FLUX.2", flux1: "FLUX.1", clipl: "CLIP-L", t5: "T5",
                   other: "other", partial: "partial"};
const FLUX1 = ["flux1", "flux1s", "kontext"];

const A = {el: null, page: false, ui: null, kind: "model", data: {}, q: "", onlyOk: true, fav: false, folder: "", sort: "name",
           sel: null, back: null, presets: null};
// 筆記與品質標籤：停 0.6 秒沒再打字才存，每個資源各排各的、分類與名稱在打字的當下就記住
// （以前共用一個計時器、送出時才讀目前的分類：改了模型的筆記馬上切到 LoRA，會存成同名 LoRA 的筆記；改另一個會取消前一個）
const later = keyedDelay(600);

const low = s => String(s || "").toLowerCase();
const samePath = (a, b) => !!a && !!b && low(a).replace(/\//g, "\\") === low(b).replace(/\//g, "\\");
const archNow = () => genSettings()?.MODEL_ARCH || "";

// 跟 server/assets.py 的 compatible() 同一條規則
function compatible(kind, a) {
  const m = archNow();
  if (!a || !m || kind === "model") return true;
  if (a === "sd" || a === "clipl") return m === "sdxl" || m === "sd15";
  if (a === "flux1" || a === "t5") return FLUX1.includes(m);          // FLUX.1 的 LoRA 大多 dev、schnell 都能套
  if (a === "qwenimage") return m === "anima" || m === "krea2";
  if (a === "flux2") return m === "klein4" || m === "klein9";
  if (a === "zimage" && kind === "te") return m === "zimage" || m === "klein4";   // Qwen3-4B 兩個共用
  if (a === "zimage" && kind === "vae") return m === "zimage" || FLUX1.includes(m);  // ae.safetensors：Z-Image 與 FLUX.1 共用
  return a === m;
}

// ── 開關 ──
export function initAssets(ui) {
  A.ui = ui;
}

export async function openAssets(kind = "model") {
  closePop();
  A.back = document.activeElement;
  if (!A.el) build();
  A.el.hidden = false;
  document.body.classList.add("viewing");
  A.data = {};
  await switchKind(TABS.includes(kind) ? kind : "model", true);
  $("assetSearch")?.focus();
}

export function closeAssets() {
  later.flush();                       // 排著還沒存的筆記先送（獨立視窗馬上就關了，送的時候用 keepalive）
  if (A.page) { window.close(); return; }
  if (!A.el || A.el.hidden) return;
  A.el.hidden = true;
  document.body.classList.remove("viewing");
  A.back?.focus?.();
}

const isOpen = () => !!A.el && !A.el.hidden;

// ── 獨立視窗（整頁模式）──
// 每 2 秒說一次「我還開著」，主視窗點資源格時就叫這裡換分頁，不再開彈出視窗
export async function assetsPage(kind = "model") {
  A.page = true;
  build();
  document.title = `${t("assets.title")} · ${t("app.title")}`;
  const here = () => send("assets:here");
  here();
  setInterval(here, 2000);
  on("assets:ping", here);
  on("assets:show", ({kind: k}) => { if (TABS.includes(k) && k !== A.kind) switchKind(k); });
  addEventListener("pagehide", () => { later.flush(); send("assets:bye"); });
  await switchKind(TABS.includes(kind) ? kind : "model", true);
}

// 別的視窗改了設定：reload＝伺服器上的設定變了（模型、資料夾…），清單重抓；其他只重畫使用中的標記
export async function refreshAssets(why) {
  if (!isOpen()) return;
  if (why === "reload") {
    const keep = A.sel?.path;
    A.data = {};
    await switchKind(A.kind, true);
    const it = keep && (A.data[A.kind] || []).find(x => x.path === keep);
    if (it) select(it);
  } else {
    drawTools();
    drawAll();
  }
}

function popOut() {
  openWindow(`/#assets-${A.kind}`, "coco-assets");
  closeAssets();
}

function build() {
  A.el = h(A.page ? "div.assets.apage" : "div.assets",
           A.page ? {} : {role: "dialog", "aria-modal": "true", "aria-labelledby": "assetTitle", hidden: true},
    h("div.apanel.alib", {},
      h("aside.aside", {},
        h("h2.title", {id: "assetTitle", "data-i18n": "assets.title"}),
        h("div.atabs", {role: "tablist", "aria-orientation": "vertical", id: "assetTabs"}),
        h("div.atools", {id: "assetTools"})),
      h("div.amain", {},
        h("div.ainuse", {id: "assetInUse", hidden: true}),
        h("div.agrid", {id: "assetGrid", role: "listbox", "data-i18n-aria": "assets.title"})),
      h("aside.adetail", {id: "assetDetail"}),
      h("div.acorner", {},
        !A.page && multiEnabled(A.ui?.multiwin)
          ? h("button.round", {type: "button", "data-i18n-aria": "assets.popout", "data-i18n-title": "assets.popout",
                               on: {click: popOut}}, "⧉") : null,
        h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close",
                           on: {click: closeAssets}}, "✕"))));
  if (!A.page) A.el.addEventListener("click", e => { if (e.target === A.el) closeAssets(); });
  if (A.page) $("app").replaceChildren(A.el); else document.body.append(A.el);
  translate(A.el);
}

// 多視窗的開關改了：⧉ 要跟著出現或消失（下次打開時重建）
export function resetAssetsFrame() {
  if (A.page || !A.el || isOpen()) return;
  A.el.remove();
  A.el = null;
}

async function switchKind(kind, refresh = false) {
  A.kind = kind;
  $("assetGrid")?.classList.remove("tlgrid", "tlinbox");          // 標籤分頁的小卡片排法
  A.folder = "";
  A.sel = null;
  if (kind === "recipe") {
    bindRecipeTab(what => {
      if (!isRecipe()) return;
      if (what === "grid") recipeGrid($("assetGrid"));
      else if (what === "sel") { markRecipeSel(); recipeDetail($("assetDetail")); }
      else { drawTabs(); recipeInUse($("assetInUse")); recipeGrid($("assetGrid")); recipeDetail($("assetDetail")); }
    });
    A.data.recipe = await loadRecipeTab();
    drawTabs();
    drawTools();
    drawAll();
    return;
  }
  if (kind === "tags") {
    bindTagTab(what => {
      if (!isTags()) return;
      if (what === "grid") tagGrid($("assetGrid"));
      else if (what === "sel") { tagGrid($("assetGrid")); tagDetail($("assetDetail")); }
      else if (what === "tools") tagTools($("assetTools"));
      else if (what === "loading") $("assetGrid").replaceChildren(h("p.dim", {text: t("wall.loading")}));
      else { drawTabs(); tagTools($("assetTools")); tagInUse($("assetInUse")); tagGrid($("assetGrid")); tagDetail($("assetDetail")); }
    });
    drawTabs();
    $("assetGrid").replaceChildren(h("p.dim", {text: t("wall.loading")}));
    const d = await loadTagTab();
    A.data.tags = [...d.tags, ...d.groups];
    drawTabs();
    drawTools();
    drawAll();
    return;
  }
  if (refresh || !A.data[kind]) {
    drawTabs();
    $("assetGrid").replaceChildren(h("p.dim", {text: t("wall.loading")}));
    try {
      const d = await get(`/api/assets?kind=${kind}`);
      if (d.error) throw new Error(d.error);
      A.data[kind] = d.items;
    } catch (e) {
      $("assetGrid").replaceChildren(h("p.dim", {text: t("assets.err", {msg: e.message || e})}));
      return;
    }
  }
  A.sel = (active()[0] && A.data[kind].find(it => isActive(it))) || null;
  drawTabs();
  drawTools();
  drawAll();
}

function drawAll() {
  if (isTags()) {
    tagInUse($("assetInUse"));
    tagGrid($("assetGrid"));
    tagDetail($("assetDetail"));
    return;
  }
  if (isRecipe()) {
    recipeInUse($("assetInUse"));
    recipeGrid($("assetGrid"));
    recipeDetail($("assetDetail"));
    return;
  }
  drawInUse();
  drawGrid();
  drawDetail();
}

// ── 目前在用的 ──
function active() {
  const s = genSettings() || {}, a = archNow();
  if (A.kind === "model") return [s.MODEL].filter(Boolean);
  if (A.kind === "lora") return loraTags(s.PROMPT).map(x => x.name);
  if (A.kind === "embedding") return (A.data.embedding || []).filter(isActive).map(it => it.name);
  const key = A.kind === "vae" ? "VAE_OVERRIDE" : "TE_OVERRIDE";
  return [(s[key] || {})[a]].filter(Boolean);
}

// Embedding 寫在正面或負面的哪一邊（都沒有是空陣列）
function embSides(it) {
  const s = genSettings() || {};
  return SIDES.filter(k => hasEmb(s[k], it.name));
}

function isActive(it) {
  if (A.kind === "lora") return hasLora(genSettings()?.PROMPT, it.name);
  if (A.kind === "embedding") return embSides(it).length > 0;
  return active().some(p => samePath(p, it.path));
}

function loraWeight(it) {
  const tag = loraTags(genSettings()?.PROMPT).find(x => low(x.name) === low(it.name));
  return tag ? tag.weight : 1;
}

// ── 上方：分頁與工具列 ──
function drawTabs() {
  const tabs = $("assetTabs");
  tabs.replaceChildren(...TABS.map(k => h("button", {
    type: "button", role: "tab", "aria-selected": String(k === A.kind), "aria-pressed": String(k === A.kind),
    on: {click: () => switchKind(k)}},
    h("span", {text: t(`assets.kind.${k}`)}),
    (k === "recipe" ? RCP.list : A.data[k]) ? h("span.c.mono", {text: (k === "recipe" ? RCP.list : A.data[k]).length}) : null)));
}

function drawTools() {
  if (isRecipe()) { recipeTools($("assetTools")); return; }
  if (isTags()) { tagTools($("assetTools")); return; }
  const items = A.data[A.kind] || [];
  const folders = [...new Set(items.map(it => it.rel).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  const arch = ARCH_NAME[archNow()] || archNow() || "?";
  const box = $("assetTools");
  box.replaceChildren(...[
    h("input.find", {id: "assetSearch", type: "search", value: A.q, "data-i18n-placeholder": "assets.search",
                     "data-i18n-aria": "assets.search", autocomplete: "off", spellcheck: "false",
                     on: {input: e => { A.q = e.target.value; drawGrid(); }}}),
    A.kind !== "model" ? h("button.chip", {type: "button", "aria-pressed": String(A.onlyOk),
                                          on: {click: e => { A.onlyOk = !A.onlyOk; e.currentTarget.setAttribute("aria-pressed", String(A.onlyOk)); drawGrid(); }}},
      t("assets.onlyOk", {arch})) : null,
    h("button.chip", {type: "button", "aria-pressed": String(A.fav),
                      on: {click: e => { A.fav = !A.fav; e.currentTarget.setAttribute("aria-pressed", String(A.fav)); drawGrid(); }}},
      `★ ${t("assets.fav")}`),
    folders.length ? h("select.chip", {"data-i18n-aria": "lib.folder", on: {change: e => { A.folder = e.target.value; drawGrid(); }}},
      h("option", {value: "", text: t("assets.allFolders")}),
      folders.map(f => h("option", {value: f, text: f, selected: f === A.folder}))) : null,
    h("label.asort", {},
      h("span", {"data-i18n": "lib.sort"}),
      h("select.chip", {"data-i18n-aria": "lib.sort", on: {change: e => { A.sort = e.target.value; drawGrid(); }}},
        SORTS.map(k => h("option", {value: k, text: t(`assets.sort.${k}`), selected: A.sort === k})))),
  ].filter(Boolean));
  translate(box);
}

// ── 卡片 ──
function visible() {
  const q = low(A.q).trim();
  const list = (A.data[A.kind] || []).filter(it =>
    (!q || low(it.name).includes(q) || low(it.rel).includes(q) || it.triggers.some(x => low(x).includes(q)) || low(it.note).includes(q))
    && (!A.fav || it.fav) && (!A.folder || it.rel === A.folder)
    && (!A.onlyOk || A.kind === "model" || compatible(A.kind, it.arch) || isActive(it)));
  const by = {
    name: (a, b) => a.name.localeCompare(b.name, lang()),
    recent: (a, b) => (b.last || 0) - (a.last || 0) || a.name.localeCompare(b.name, lang()),
    uses: (a, b) => (b.uses || 0) - (a.uses || 0) || a.name.localeCompare(b.name, lang()),
    size: (a, b) => (b.size || 0) - (a.size || 0),
  }[A.sort];
  // 使用中的不再排到最前面（上方另有一列小圖），牆的順序才不會一點就跳
  return list.sort(by);
}

function picOf(it, big = false) {
  const glyph = () => h("span.aglyph", {text: ARCH_NAME[it.arch] || "—"});
  const img = src => {
    const el = h("img", {src, alt: "", loading: "lazy", decoding: "async"});
    el.addEventListener("error", () => el.replaceWith(glyph()), {once: true});
    return el;
  };
  if (it.cover) {
    return img(`/api/res/cover?kind=${A.kind}&path=${encodeURIComponent(it.path)}${big ? "" : "&thumb=1"}` +
               (it.coverv ? `&v=${encodeURIComponent(it.coverv)}` : ""));
  }
  if (it.sample) {
    return h("span.asample", {}, img(`/api/thumb?id=${it.sample.id}&v=${encodeURIComponent(it.sample.v)}`),
      big ? null : h("span.abadge", {"data-i18n": "assets.fromGallery"}));
  }
  return glyph();
}

function card(it) {
  const on = isActive(it), ok = compatible(A.kind, it.arch);
  const el = h("div.acard", {role: "option", tabindex: "0", "aria-selected": String(on),
                             "data-key": it.path, "data-off": ok ? null : "1", "data-sel": A.sel === it ? "1" : null,
                             title: `${it.name}\n${t("assets.how")}`,
                             on: {click: e => { if (!e.target.closest(".afav, .aweight")) use(it); },
                                  contextmenu: e => { e.preventDefault(); select(it); },
                                  keydown: e => {
                                    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); select(it); }
                                    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); use(it); }
                                  }}},
    h("div.acov", {}, picOf(it), on ? h("span.tape", {"data-i18n": "assets.inUse"}) : null,
      h("button.afav", {type: "button", "aria-pressed": String(!!it.fav), "data-i18n-aria": "assets.fav", "data-i18n-title": "assets.fav",
                        on: {click: () => setFav(it, !it.fav)}}, "★")),
    h("div.ainfo", {},
      h("b.aname", {text: it.name}),
      h("div.ameta", {},
        it.arch ? h("span.arch", {"data-arch": it.arch, text: it.arch === "partial" ? t("assets.partial") : (ARCH_NAME[it.arch] || it.arch)}) : null,
        !ok ? h("span.bad", {"data-i18n": "assets.incompatible"}) : null,
        it.uses ? h("span", {text: t("assets.uses", {n: it.uses.toLocaleString(lang())})}) : null,
        it.size ? h("span.mono", {text: `${it.size} GB`}) : null)),
    on && A.kind === "lora" ? weightRow(it) : null,
    on && A.kind === "embedding" ? h("div.aweight.aemb", {}, h("span", {text: embSides(it).map(k => t(`assets.side.${k}`)).join("＋")})) : null);
  return el;
}

// 內建選項：VAE、編碼器可以選「模型內建」（清掉這個架構的覆寫）
function builtinCard() {
  const on = !active().length;
  return h("div.acard.builtin", {role: "option", tabindex: "0", "aria-selected": String(on),
                                 on: {click: () => useBuiltin(), keydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); useBuiltin(); } }}},
    h("div.acov", {}, h("span.aglyph", {"data-i18n": "gen.slot.builtin"}), on ? h("span.tape", {"data-i18n": "assets.inUse"}) : null),
    h("div.ainfo", {}, h("b.aname", {"data-i18n": "gen.slot.builtin"}), h("div.ameta", {}, h("span", {"data-i18n": "assets.builtinHint"}))));
}

function weightRow(it) {
  const w = loraWeight(it);
  const out = h("span.mono.wv", {text: fmtWeight(w)});
  return h("div.aweight", {},
    h("input", {type: "range", min: "-1", max: "2", step: "0.05", value: String(w), "data-i18n-aria": "assets.weight",
                on: {input: e => { out.textContent = fmtWeight(e.target.value); },
                     change: e => setPromptText(setLoraWeight(genSettings().PROMPT, it.name, e.target.value))}}),
    out);
}

// ── 使用中：牆上方的一列小圖。點一下看它的詳細（LoRA 的 × 是拿掉） ──
function drawInUse() {
  const box = $("assetInUse");
  if (!box) return;
  if (isRecipe()) { recipeInUse(box); return; }
  if (isTags()) { tagInUse(box); return; }
  const items = (A.data[A.kind] || []).filter(isActive);
  const builtin = (A.kind === "vae" || A.kind === "te") && !active().length;
  box.hidden = !items.length && !builtin;
  box.replaceChildren(...(box.hidden ? [] : [
    h("span.k", {"data-i18n": "assets.inUse"}),
    ...(builtin ? [h("div.amini", {}, h("span.mpic", {}, h("span.aglyph", {text: "—"})),
                     h("span.mname", {"data-i18n": "gen.slot.builtin"}))] : items.map(mini))]));
  translate(box);
}

function mini(it) {
  const lora = A.kind === "lora" || A.kind === "embedding";
  return h("div.amini", {role: "button", tabindex: "0", title: it.name, "data-key": it.path, "data-sel": A.sel === it ? "1" : null,
                         on: {click: e => { if (!e.target.closest(".mx")) select(it, true); },
                              keydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(it, true); } }}},
    h("span.mpic", {}, picOf(it)),
    h("span.mname", {text: it.name}),
    A.kind === "lora" ? h("span.mw.mono", {text: fmtWeight(loraWeight(it))}) : null,
    A.kind === "embedding" ? h("span.mw", {text: embSides(it).map(k => t(`assets.side.${k}`)).join("＋")}) : null,
    lora ? h("button.mx", {type: "button", title: t("assets.remove"), "aria-label": t("assets.remove"),
                           on: {click: () => (A.kind === "embedding" ? embToggle(it, null) : use(it))}}, "×") : null);
}

// 只選起來看詳細，不套用（右鍵、Shift+Enter、點使用中的小圖）
function select(it, reveal = false) {
  A.sel = it;
  for (const el of document.querySelectorAll("#assetGrid .acard[data-key], #assetInUse .amini")) {
    const on = el.dataset.key === it.path;
    el.toggleAttribute("data-sel", on);
    if (on && reveal && el.classList.contains("acard")) el.scrollIntoView({block: "nearest", behavior: "smooth"});
  }
  drawDetail();
}

function drawGrid() {
  if (isRecipe()) { recipeGrid($("assetGrid")); return; }
  if (isTags()) { tagGrid($("assetGrid")); return; }
  const grid = $("assetGrid");
  const list = visible();
  const extra = A.kind === "vae" || A.kind === "te" ? [builtinCard()] : [];
  grid.replaceChildren(...extra, ...list.map(card));
  if (!list.length) grid.append(h("p.dim.aempty", {text: t(A.q || A.fav || A.folder ? "assets.noMatch" : "assets.empty")}));
  translate(grid);
}

// ── 用這個 ──
async function use(it) {
  A.sel = it;
  const kind = A.kind, s = genSettings();
  try {
    if (kind === "lora") {
      const on = isActive(it);
      setPromptText(on ? removeLora(s.PROMPT, it.name) : addLora(s.PROMPT, it.name, 1));
      if (!on) touch(it);
    } else if (kind === "embedding") {
      // 用了就拿掉（兩邊都拿）；沒用就加進生成台正在編輯的那一邊
      embToggle(it, isActive(it) ? null : editingField());
    } else if (kind === "model") {
      if (!samePath(s.MODEL, it.path)) {
        await post("/api/res/pick", {kind: "model", path: it.path});
        touch(it);
        await reloadSettings();
      }
    } else {
      const key = kind === "vae" ? "VAE_OVERRIDE" : "TE_OVERRIDE";
      const arch = archNow();
      if (!arch) return;
      await post("/api/settings", {[key]: {[arch]: it.path}});
      touch(it);
      await reloadSettings();
    }
  } finally {
    drawTools();
    drawAll();
  }
}

// side：加進 PROMPT / NEGATIVE；null＝兩邊都拿掉
function embToggle(it, side) {
  const s = genSettings();
  if (!s) return;
  if (side) {
    setField(side, addEmb(s[side], it.name));
    touch(it);
  } else {
    for (const k of SIDES) if (hasEmb(s[k], it.name)) setField(k, removeEmb(s[k], it.name));
  }
  drawTools();
  drawAll();
}

async function useBuiltin() {
  const key = A.kind === "vae" ? "VAE_OVERRIDE" : "TE_OVERRIDE";
  const arch = archNow();
  if (!arch) return;
  await post("/api/settings", {[key]: {[arch]: ""}});
  await reloadSettings();
  A.sel = null;
  drawAll();
}

function touch(it) {
  post("/api/assets/meta", {kind: A.kind, name: it.name, touch: true})
    .then(d => { if (d.meta) { it.last = d.meta.last || it.last; it.picks = d.meta.picks || it.picks; } })
    .catch(() => { /* 最近用過沒記到不影響使用 */ });
}

async function setFav(it, fav) {
  it.fav = fav;
  drawGrid();
  drawInUse();
  if (A.sel === it) drawDetail();
  await post("/api/assets/meta", {kind: A.kind, name: it.name, fav}).catch(() => {});
}

// ── 右邊的詳細欄 ──
function drawDetail() {
  if (isRecipe()) { recipeDetail($("assetDetail")); return; }
  if (isTags()) { tagDetail($("assetDetail")); return; }
  const box = $("assetDetail");
  const it = A.sel;
  if (!it) {
    box.replaceChildren(h("p.dim", {"data-i18n": A.kind === "lora" ? "assets.hintLora" : A.kind === "embedding" ? "assets.hintEmb" : "assets.hint"}));
    translate(box);
    return;
  }
  const on = isActive(it);
  const noteBox = h("textarea.anote", {rows: "4", value: it.note || "", "data-i18n-placeholder": "assets.notePh",
                                       "data-i18n-aria": "assets.note",
                                       on: {input: e => {
                                         it.note = e.target.value;
                                         const kind = A.kind, name = it.name;
                                         later.put(`note:${kind}:${name}`, () =>
                                           post("/api/assets/meta", {kind, name, note: it.note}, {keepalive: true}).catch(() => {}));
                                       }}});
  const prompt = genSettings()?.PROMPT || "";
  box.replaceChildren(...[
    h("div.adpic", {}, picOf(it, true),
      h("button.pill.acovbtn", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "assets.cover",
                                on: {click: e => openPop("popCover", e.currentTarget, () => coverPop(it))}})),
    h("h3.title", {text: it.name}),
    h("p.dim.apath", {text: it.path}),
    h("div.ameta", {},
      it.arch ? h("span.arch", {"data-arch": it.arch, text: it.arch === "partial" ? t("assets.partial") : (ARCH_NAME[it.arch] || it.arch)}) : null,
      !compatible(A.kind, it.arch) ? h("span.bad", {text: t("assets.incompatibleWith", {arch: ARCH_NAME[archNow()] || archNow()})}) : null,
      it.size ? h("span.mono", {text: `${it.size} GB`}) : null),
    h("div.vacts", {},
      A.kind === "embedding"
        ? SIDES.map(k => {
          const here = embSides(it).includes(k);
          return h(`button.pill${k === "PROMPT" ? ".primary" : ""}`, {type: "button", on: {click: () => {
            const s = genSettings();
            if (here) { setField(k, removeEmb(s[k], it.name)); drawAll(); } else embToggle(it, k);
          }}}, t(here ? `assets.emb.remove.${k}` : `assets.emb.add.${k}`));
        })
        : h("button.pill.primary", {type: "button", on: {click: () => use(it)}},
          t(A.kind === "lora" ? (on ? "assets.remove" : "assets.add") : (on ? "assets.inUse" : "assets.use"))),
      h("button.pill", {type: "button", "aria-pressed": String(!!it.fav), on: {click: () => setFav(it, !it.fav)}},
        `${it.fav ? "★" : "☆"} ${t("assets.fav")}`),
      (A.kind === "lora" || A.kind === "model") && it.uses
        ? h("button.pill", {type: "button", on: {click: () => { closeAssets(); showInLibrary(A.kind, it.key); }}},
            t("assets.seeImages", {n: it.uses.toLocaleString(lang())}))
        : null),
    trigSection(it, prompt),
    A.kind === "model" ? presetSection(it) : null,
    h("section.vsec", {}, h("h3", {"data-i18n": "assets.note"}), noteBox),
    kindSection(it),
    it.last ? h("p.dim.small", {text: t("assets.lastUsed", {when: new Date(it.last * 1000).toLocaleString(lang(), {dateStyle: "medium", timeStyle: "short"})})}) : null,
  ].filter(Boolean));
  translate(box);
}

// ── 觸發詞：點了加進提示詞；可以自己改（一行一個），改過的記在 asset_notes，模型旁的資料檔不動 ──
function trigSection(it, prompt) {
  const sec = h("section.vsec.atrigsec");
  const save = async body => {
    const d = await post("/api/assets/meta", {kind: A.kind, name: it.name, ...body}).catch(e => ({error: e.message}));
    if (d.error) return;
    it.triggers = body.resetTriggers ? (it.fileTriggers || []) : body.triggers;
    it.trigEdited = !body.resetTriggers;
    view();
  };
  const view = () => {
    sec.replaceChildren(
      h("div.vsh", {}, h("h3", {"data-i18n": "assets.triggers"}),
        it.trigEdited ? h("span.dim.small", {"data-i18n": "assets.trigEdited"}) : null,
        h("span.sp"),
        h("button.mini", {type: "button", "data-i18n": "assets.trigEdit", on: {click: edit}})),
      it.triggers.length ? h("div.atrig", {}, it.triggers.map(tr => h("button.chip", {
        type: "button", title: t("assets.addTrigger"), "aria-pressed": String(hasTrigger(prompt, tr)),
        on: {click: e => {
          const p = genSettings().PROMPT || "";
          if (hasTrigger(p, tr)) return;
          setPromptText(addTrigger(p, tr));
          e.currentTarget.setAttribute("aria-pressed", "true");
        }}}, tr)))
        : h("p.dim.small", {"data-i18n": "assets.trigNone"}));
    translate(sec);
  };
  const edit = () => {
    const ta = h("textarea.anote", {rows: String(Math.max(3, it.triggers.length + 1)), value: it.triggers.join("\n"),
                                    spellcheck: "false", "data-i18n-placeholder": "assets.trigPh", "data-i18n-aria": "assets.triggers"});
    sec.replaceChildren(
      h("div.vsh", {}, h("h3", {"data-i18n": "assets.triggers"})),
      ta,
      h("div.vacts", {},
        h("button.pill.primary", {type: "button", "data-i18n": "assets.trigSave",
                                  on: {click: () => save({triggers: ta.value.split(/\n/).map(x => x.trim()).filter(Boolean)})}}),
        h("button.pill", {type: "button", "data-i18n": "fb.cancel", on: {click: view}}),
        it.trigEdited ? h("button.pill", {type: "button", "data-i18n": "assets.trigReset", on: {click: () => save({resetTriggers: true})}})
          : null));
    translate(sec);
    ta.focus();
  };
  view();
  return sec;
}

// ── 改分類：放錯分頁的檔案（例如 LoRA 資料夾裡混了一個 embedding）──
function kindSection(it) {
  const pick = h("select.chip", {"data-i18n-aria": "assets.kindMove", on: {change: async e => {
    const kind = e.target.value;
    const d = await post("/api/res/kind", {path: it.path, kind: kind === A.kind && !it.moved ? "" : kind})
      .catch(err => ({error: err.message}));
    if (d.error) { msg.textContent = d.error; return; }
    A.data = {};                         // 兩個分頁的清單都變了
    await switchKind(A.kind, true);
  }}}, KINDS.map(k => h("option", {value: k, text: t(`assets.kind.${k}`), selected: k === A.kind})));
  const msg = h("p.hint", {role: "status"});
  const [btn, hint] = info("assets.kindHint");
  return h("section.vsec.akind", {},
    h("h3", {}, t("assets.kindMove"), btn),
    hint,
    h("div.row", {}, pick,
      it.moved ? h("button.mini", {type: "button", "data-i18n": "assets.kindReset", on: {click: async () => {
        await post("/api/res/kind", {path: it.path, kind: ""}).catch(() => {});
        A.data = {};
        await switchKind(A.kind, true);
      }}}) : null),
    it.moved ? h("p.hint", {"data-i18n": "assets.kindMoved"}) : null,
    msg);
}

// ── 自訂封面（筆記 #9）──
// 用過它的圖挑一張，或上傳一張（先在瀏覽器縮到 1024 再送）；存在 asset_notes，模型資料夾不動
function coverPop(it) {
  const kind = A.kind;
  const list = h("div.acovlist", {}, h("p.dim", {"data-i18n": "wall.loading"}));
  const file = h("input", {type: "file", accept: "image/*", hidden: true, on: {change: e => upload(e.target.files?.[0])}});
  const msg = h("p.hint.acovmsg", {role: "status"});
  const done = async body => {
    const d = await post("/api/assets/cover", {kind, name: it.name, ...body}).catch(e => ({error: e.message}));
    if (d.error) { msg.textContent = d.error; return; }
    it.custom = !body.clear;
    it.cover = it.custom || !!it.sidecar;
    it.coverv = it.custom ? String(Date.now()) : "";
    closePop();
    drawAll();
  };
  const upload = async f => {
    if (!f) return;
    msg.textContent = t("assets.coverUploading");
    try { await done({data: await shrink(f)}); } catch { msg.textContent = t("assets.coverBad"); }
  };
  if (kind === "model" || kind === "lora") {
    get(`/api/assets/images?kind=${kind}&key=${encodeURIComponent(it.key)}`).then(d => {
      const items = d.items || [];
      list.replaceChildren(...(items.length ? items.map(x => h("button.acovpick", {
        type: "button", on: {click: () => done({id: x.id})}},
        h("img", {src: `/api/thumb?id=${x.id}&v=${encodeURIComponent(x.v)}`, alt: "", loading: "lazy"})))
        : [h("p.dim", {"data-i18n": "assets.coverNone"})]));
      translate(list);
    }).catch(() => list.replaceChildren(h("p.dim", {"data-i18n": "assets.coverNone"})));
  } else {
    list.replaceChildren(h("p.dim", {"data-i18n": "assets.coverOnlyUpload"}));
  }
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "assets.cover"})),
    list,
    h("div.row.acovacts", {},
      h("button.pill", {type: "button", "data-i18n": "assets.coverUpload", on: {click: () => file.click()}}),
      it.custom ? h("button.pill", {type: "button", "data-i18n": "assets.coverReset", on: {click: () => done({clear: true})}}) : null,
      file),
    msg].filter(Boolean);
}

// 圖片檔 → 最長邊 1024 的 WebP data URL
function shrink(f) {
  return new Promise((ok, bad) => {
    const url = URL.createObjectURL(f), img = new Image();
    img.onload = () => {
      const k = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
      const c = h("canvas", {width: String(Math.round(img.naturalWidth * k)), height: String(Math.round(img.naturalHeight * k))});
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      ok(c.toDataURL("image/webp", 0.9));
    };
    img.onerror = () => { URL.revokeObjectURL(url); bad(new Error("image")); };
    img.src = url;
  });
}

// ── 品質標籤與預設負面 ──
// 「用架構預設」時改的是那個架構所有模型共用的；「自訂」只給這個模型；「不加」就不加。
// 模型用 res_stem 對（小寫、不含資料夾與副檔名），跟 server/presets.py 一樣
const presetKey = it => stem(it.path).toLowerCase();
// 只拿來複製、不能直接改：以前沒有預設的架構都直接改這一個，Z-Image 打的字跑到 Krea 2、存進別的架構（R66）
const EMPTY = {quality: "", negative: "", position: "before"};

async function savePreset(body, redraw = false) {
  const d = await post("/api/presets", body, {keepalive: true}).catch(e => ({error: e.message}));      // 關視窗前 flush 的也送得到
  if (d.error) return;
  A.presets = {arch: d.arch, model: d.model};
  refreshPreset();                     // 生成台開關滑過時顯示的內容跟著變
  if (redraw) drawDetail();
}

function presetSection(it) {
  if (!A.presets) {
    get("/api/presets").then(d => { A.presets = d; if (A.sel === it) drawDetail(); }).catch(() => {});
    return h("section.vsec", {}, h("p.dim", {"data-i18n": "wall.loading"}));
  }
  const arch = it.arch && it.arch !== "partial" ? it.arch : "";
  const own = A.presets.model[presetKey(it)];
  const mode = own ? own.mode : "inherit";
  // 還沒有預設的架構各自建一份放進 A.presets：存好之前重畫也還看得到剛打的字
  const cur = mode === "custom" ? own : (A.presets.arch[arch] ||= {...EMPTY});
  const archName = ARCH_NAME[arch] || arch;
  // 改文字：存到自訂（這個模型）或架構預設（這個架構全部）
  const edit = patch => {
    Object.assign(cur, patch);
    later.put(mode === "custom" ? `preset:model:${it.path}` : `preset:arch:${arch}`, () => savePreset(mode === "custom"
      ? {scope: "model", key: it.path, mode: "custom", ...cur} : {scope: "arch", key: arch, ...cur}));
  };
  const field = (key, rows, label) => h("label.aprow", {},
    h("span.k", {"data-i18n": label}),
    h("textarea.anote", {rows: String(rows), value: cur[key] || "", spellcheck: "false", "data-i18n-aria": label,
                         on: {input: e => edit({[key]: e.target.value})}}));
  const pick = m => {
    if (m === mode) return;
    savePreset(m === "custom" ? {scope: "model", key: it.path, mode: "custom", ...(A.presets.arch[arch] || EMPTY)}
                              : {scope: "model", key: it.path, mode: m}, true);
  };
  const [btn, hint] = info("assets.presetHint");
  return h("section.vsec.apreset", {},
    h("h3", {}, t("assets.preset"), btn),
    hint,
    h("div.seg", {role: "group", "data-i18n-aria": "assets.preset"},
      h("button", {type: "button", "aria-pressed": String(mode === "inherit"), disabled: !arch,
                   on: {click: () => pick("inherit")}}, arch ? t("assets.preset.inherit", {arch: archName}) : t("assets.preset.noArch")),
      h("button", {type: "button", "aria-pressed": String(mode === "custom"), "data-i18n": "assets.preset.custom", on: {click: () => pick("custom")}}),
      h("button", {type: "button", "aria-pressed": String(mode === "none"), "data-i18n": "assets.preset.none", on: {click: () => pick("none")}})),
    mode === "none" ? h("p.dim.small", {"data-i18n": "assets.preset.noneHint"})
      : !arch && mode === "inherit" ? h("p.dim.small", {"data-i18n": "assets.preset.noArchHint"})
      : [
        mode === "inherit" ? h("p.dim.small", {text: t("assets.preset.shared", {arch: archName})}) : null,
        field("quality", 2, "assets.preset.quality"),
        h("div.seg.aprow", {role: "group", "data-i18n-aria": "assets.preset.where"}, ["before", "after"].map(pos => h("button", {
          type: "button", "aria-pressed": String((cur.position || "before") === pos), "data-i18n": `assets.preset.${pos}`,
          on: {click: e => {
            e.currentTarget.parentElement.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
            edit({position: pos});
          }}}))),
        field("negative", 3, "assets.preset.negative"),
      ]);
}

document.addEventListener("keydown", e => {
  if (A.page || !isOpen() || e.defaultPrevented) return;
  if (e.key === "Escape") { e.preventDefault(); closeAssets(); }
});
