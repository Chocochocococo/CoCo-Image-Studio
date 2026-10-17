// Recipe：可重複使用的組合——模型、LoRA、VAE、編碼器、取樣設定、尺寸、負面提示詞，以及（存的時候勾了才有）提示詞。
// 有提示詞的 Recipe 套用時整段換掉；沒有的只把提示詞裡的 <lora:…> 換成 Recipe 的，其他文字不動。
// （使用者筆記：從圖庫存成 Recipe 要連提示詞、VAE、編碼器一起存）
//
// 資料格式（跟 server/recipes.py 的 clean() 一致）：
//   {model, arch, vae, te, loras: [{name, weight}], sampler, schedule, steps, cfg, clip_skip, shift,
//    width, height, negative, prompt}
//   vae / te 是檔案路徑；"builtin" 代表用模型內建的；"fp16fix" 是 SDXL 的修正版 VAE；沒有這個鍵代表「不動」。
//   prompt 不含外層的 <lora:…>（LoRA 另外記在 loras）

import {get, post} from "./api.js";
import {$, h, stem} from "./dom.js";
import {applyParams, genSettings} from "./generate.js";
import {lang, t, translate} from "./i18n.js";
import {addLora, fmtWeight, loraTags, mapSampler, shiftKey, stripTopLoras} from "./options.js";
import {closePop, openPop} from "./pop.js";

export const R = {list: [], ui: null, hooks: {},
                  undo: null};          // 套用前後的設定（這次開著 CoCo 期間）：取消套用時放回去
const low = s => String(s || "").toLowerCase();
const samePath = (a, b) => !!a && !!b && low(a).replace(/\//g, "\\") === low(b).replace(/\//g, "\\");

export async function loadRecipes() {
  try { R.list = (await get("/api/recipes")).recipes || []; } catch { R.list = []; }
  return R.list;
}

export function initRecipes(ui, hooks = {}) {
  R.ui = ui;
  R.hooks = hooks;
  return loadRecipes();
}

// ── 目前生成台的設定 → Recipe ──
export function currentRecipe() {
  const s = genSettings() || {}, a = s.MODEL_ARCH || "";
  const part = key => (s[key] || {})[a] || "builtin";
  return {
    model: s.MODEL || "", arch: a, vae: part("VAE_OVERRIDE"), te: part("TE_OVERRIDE"),
    loras: loraTags(s.PROMPT).map(x => ({name: x.name, weight: x.weight})),
    sampler: s.SAMPLER, schedule: s.SCHEDULE, steps: s.STEPS, cfg: s.CFG,
    clip_skip: s.CLIP_SKIP, shift: s[shiftKey(a)], width: s.WIDTH, height: s.HEIGHT, negative: s.NEGATIVE || "",
    prompt: stripTopLoras(s.PROMPT || ""),
  };
}

// 不存提示詞（存檔時沒勾「包含提示詞」）
const withPrompt = (data, on) => {
  if (on) return data;
  const {prompt, ...rest} = data;
  return rest;
};

// 圖的參數（/api/detail 或比較組裡的 img）→ Recipe。模型此時只有名字，存檔時再找檔案。
// VAE、編碼器：CoCo 存的圖記在 extra.parts（路徑、fp16fix 或 builtin）；Forge 的 Anima 圖記在 Module 1 / 2（檔名）
export function recipeFromImage(d) {
  const extra = d.extra || {};
  const m = mapSampler(d.sampler, d.scheduler);
  const shift = Number(extra.Shift ?? extra.shift);
  const out = {
    model: d.model || "", loras: (d.loras || []).map(l => ({name: stem(l.file), weight: Number(l.weight) || 1})),
    steps: d.steps, cfg: d.cfg, clip_skip: d.clip_skip, width: d.width, height: d.height, negative: d.extra?.user_negative ?? d.negative ?? "",
  };
  if (m.sampler) out.sampler = m.sampler;
  if (m.schedule) out.schedule = m.schedule;
  if (Number.isFinite(shift) && shift > 0) out.shift = shift;
  // Forge 的 Anima 圖會記 VAE 與文字編碼器（Module 1 / Module 2）
  if (extra["Module 1"]) out.vae = String(extra["Module 1"]);
  if (extra["Module 2"]) out.te = String(extra["Module 2"]);
  if (extra.parts?.vae) out.vae = String(extra.parts.vae);
  if (extra.parts?.te) out.te = String(extra.parts.te);
  // 提示詞：開著品質標籤生成的圖用使用者自己寫的那份；外層的 LoRA 另外記在 loras
  const p = stripTopLoras(extra.user_prompt ?? d.prompt ?? "");
  if (p.trim()) out.prompt = p;
  for (const k of Object.keys(out)) if (out[k] == null || out[k] === "") delete out[k];
  return out;
}

// 名字 → 檔案路徑（已經是路徑的就照用）
async function resolve(kind, name) {
  if (!name || name === "builtin" || name === "fp16fix" || /[\\/]/.test(name)) return name;
  try {
    const items = (await get(`/api/res?kind=${kind}`)).items || [];
    const hit = items.find(x => low(x.name) === low(stem(name)));
    return hit ? hit.path : null;
  } catch { return null; }
}

// ── 摘要與比較 ──
export function summary(d) {
  const parts = [];
  if (d.model) parts.push(stem(d.model));
  if (d.loras?.length) parts.push(t("rcp.loraN", {n: d.loras.length}));
  if (d.sampler) parts.push([d.sampler, d.schedule].filter(Boolean).join(" · "));
  if (d.steps) parts.push(t("rcp.steps", {n: d.steps}));
  if (d.cfg != null) parts.push(`CFG ${d.cfg}`);
  if (d.width && d.height) parts.push(`${d.width}×${d.height}`);
  if (d.prompt) parts.push(t("rcp.hasPrompt"));
  return parts.join(" · ");
}

const loraKey = ls => (ls || []).map(x => `${low(x.name)}:${fmtWeight(x.weight)}`).sort().join("|");
const normPrompt = p => String(p ?? "").replace(/\s+/g, " ").replace(/\s*,\s*/g, ", ").trim();

// 目前生成台跟這個 Recipe 有沒有不一樣（只看 Recipe 有記的欄位）
export function isModified(r) {
  if (!r) return false;
  const d = r.data || {}, c = currentRecipe();
  for (const [k, v] of Object.entries(d)) {
    if (k === "arch") continue;
    // 有存提示詞的 Recipe（存檔時勾「包含提示詞」）：重新套用會把提示詞換回去，所以改了也算「已修改」（使用者回報兩邊不一致）。
    // 比的是 LoRA 以外的文字，空白不算
    if (k === "prompt") { if (normPrompt(v) !== normPrompt(c.prompt)) return true; continue; }
    if (k === "loras") { if (loraKey(v) !== loraKey(c.loras)) return true; continue; }
    if (k === "model" || k === "vae" || k === "te") {
      const cv = c[k];
      if (v === "builtin" ? cv !== "builtin" : !(samePath(v, cv) || low(stem(v)) === low(stem(cv)))) return true;
      continue;
    }
    if (typeof v === "number") { if (Math.abs(Number(c[k]) - v) > 1e-6) return true; continue; }
    if (String(c[k] ?? "") !== String(v)) return true;
  }
  return false;
}

// ── 套用 ──
// 套用會改到的設定：取消套用時比對、放回去（模型與零件要另外存，見 unapplyRecipe）
const UNDO_KEYS = ["MODEL", "SAMPLER", "SCHEDULE", "STEPS", "CFG", "CLIP_SKIP", "ANIMA_SHIFT", "ZIMAGE_SHIFT", "KREA2_SHIFT", "QWEN21_SHIFT", "FLUX2_SHIFT", "FLUX1_SHIFT", "FLUX1S_SHIFT", "WIDTH", "HEIGHT", "NEGATIVE", "PROMPT"];
const PART_KEYS = ["VAE_OVERRIDE", "TE_OVERRIDE"];
const snapOf = () => {
  const s = genSettings() || {};
  return Object.fromEntries([...UNDO_KEYS, ...PART_KEYS].map(k => [k, JSON.parse(JSON.stringify(s[k] ?? null))]));
};
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export async function applyRecipe(data, {id = "", name = ""} = {}) {
  const notes = [], s = genSettings() || {};
  const before = snapOf();
  if (data.model) {
    const path = await resolve("model", data.model);
    if (!path) notes.push(t("view.noModel", {name: stem(data.model)}));
    else if (!samePath(path, s.MODEL)) await post("/api/res/pick", {kind: "model", path});
  }
  // 模型可能剛換過，架構要重新問
  const arch = data.arch || (await get("/api/settings").catch(() => ({}))).MODEL_ARCH || s.MODEL_ARCH;
  for (const [kind, key] of [["vae", "VAE_OVERRIDE"], ["te", "TE_OVERRIDE"]]) {
    if (!data[kind] || !arch) continue;
    const path = data[kind] === "builtin" ? "" : await resolve(kind, data[kind]);
    if (path == null) { notes.push(t("rcp.noPart", {name: stem(data[kind])})); continue; }
    await post("/api/settings", {[key]: {[arch]: path}});
  }
  const patch = {};
  const map = {sampler: "SAMPLER", schedule: "SCHEDULE", steps: "STEPS", cfg: "CFG", clip_skip: "CLIP_SKIP",
               shift: shiftKey(arch), width: "WIDTH", height: "HEIGHT", negative: "NEGATIVE"};
  for (const [k, key] of Object.entries(map)) if (data[k] != null) patch[key] = data[k];
  if (data.loras || data.prompt != null) {
    // 重組會帶組好的提示詞；Recipe 不存提示詞，就用生成台現在的，只換 LoRA 標籤
    let p = data.prompt != null ? data.prompt : (genSettings()?.PROMPT || "");
    p = stripTopLoras(p);                     // 動態語法裡輪流用的 LoRA 留著，addLora 也不會再加一次
    for (const x of data.loras || []) p = addLora(p, x.name, x.weight);
    patch.PROMPT = p;
    // 找不到檔案的 LoRA 先講（例如改過名的），不要等生成時才發現
    try {
      const have = new Set(((await get("/api/res?kind=lora")).items || []).map(x => low(x.name)));
      const missing = (data.loras || []).map(x => x.name).filter(n => !have.has(low(stem(n))));
      if (missing.length) notes.push(t("view.noLora", {names: missing.join("、")}));
    } catch { /* 清單讀不到就不檢查 */ }
  }
  // 套用某個 Recipe 就記住它；套用重組出來的組合就不屬於任何 Recipe 了
  if (id) post("/api/recipes/use", {id}).catch(() => {});
  R.ui.recipe = id || "";
  post("/api/ui", {recipe: R.ui.recipe}).catch(() => {});
  await applyParams(patch, {text: [t("rcp.applied", {name: name || t("rcp.combo")}), ...notes].join(" · "),
                            kind: notes.length ? "warn" : "ok"});
  R.undo = id ? {id, before, after: snapOf()} : null;
  await loadRecipes();
  refreshBar();
}

// 取消套用（使用者筆記：套用了就沒辦法取消）：套用之後沒改過設定，就放回套用前的設定；
// 改過的話只是不再算「用這個 Recipe」，目前的設定照留，免得把後來的修改蓋掉
export async function unapplyRecipe() {
  const r = current();
  if (!r) return;
  const name = r.name || t("rcp.untitled");
  const u = R.undo?.id === r.id ? R.undo : null, now = snapOf();
  const back = u && same(now, u.after);
  R.ui.recipe = "";
  R.undo = null;
  post("/api/ui", {recipe: ""}).catch(() => {});
  if (back) {
    const b = u.before;
    if (b.MODEL && !samePath(b.MODEL, now.MODEL)) await post("/api/res/pick", {kind: "model", path: b.MODEL});
    // VAE／編碼器是一個架構一格、存檔時是合併：每一格都要明確寫回（套用前沒有的那格寫空的＝模型自帶）
    for (const k of PART_KEYS) {
      if (same(b[k], now[k])) continue;
      const archs = new Set([...Object.keys(b[k] || {}), ...Object.keys(now[k] || {})]);
      await post("/api/settings", {[k]: Object.fromEntries([...archs].map(a => [a, (b[k] || {})[a] ?? ""]))});
    }
    const patch = {};
    for (const k of UNDO_KEYS) if (k !== "MODEL" && !same(b[k], now[k])) patch[k] = b[k];
    await applyParams(patch, {text: t("rcp.unapplied", {name}), kind: "ok"});
  } else {
    await applyParams({}, {text: t("rcp.detached", {name}), kind: "ok"});
  }
  await loadRecipes();
  refreshBar();
}

// ── 生成台上的 Recipe 按鈕與膠囊列 ──
const current = () => R.list.find(r => r.id === R.ui?.recipe) || null;

export function recipeButton() {
  const r = current();
  return h("button.recipe", {type: "button", id: "btnRecipe", "aria-haspopup": "dialog", "aria-expanded": "false",
                             on: {click: e => openPop("popRecipe", e.currentTarget, recipePop)}},
    h("span.k", {text: "Recipe"}),
    h("b", {text: r ? r.name || t("rcp.untitled") : t("rcp.none")}),
    h("span.mod", {id: "recipeMod", "data-i18n": "rcp.modified", hidden: !isModified(r)}),
    " ▾");
}

export function recipePills() {
  // 取最常用的 8 個，但照名字排：點了不會跳到第一個（使用者筆記）
  const list = [...R.list].sort((a, b) => (b.used || 0) - (a.used || 0) || b.updated - a.updated).slice(0, 8)
    .sort((a, b) => (a.name || "").localeCompare(b.name || "", lang()));
  if (!list.length) return null;
  const cur = current();
  return h("div.rpills", {id: "recipePills", role: "group", "data-i18n-aria": "rcp.quick"},
    // 已經套用的那顆再點一次＝取消套用
    list.map(r => h("button.pill", {type: "button", "aria-pressed": String(cur?.id === r.id),
                                    title: cur?.id === r.id ? t("rcp.pillOff") : summary(r.data),
                                    on: {click: () => (cur?.id === r.id ? unapplyRecipe() : applyRecipe(r.data, {id: r.id, name: r.name}))}},
      r.name || t("rcp.untitled"))));
}

// 設定一改就更新「已修改」；套用或存檔後整條重畫
export function refreshBar() {
  const btn = $("btnRecipe");
  if (btn) { const nb = recipeButton(); btn.replaceWith(nb); translate(nb); }
  const pills = $("recipePills"), np = recipePills();
  if (pills) { if (np) { pills.replaceWith(np); translate(np); } else pills.remove(); }
  else if (np && $("slots")) { $("slots").after(np); translate(np); }   // 第一個 Recipe 剛存好時還沒有這一列
}

export function markModified() {
  const m = $("recipeMod");
  if (m) m.hidden = !isModified(current());
}

// ── 小視窗：清單、套用、更新、改名、刪除、另存 ──
function recipePop() {
  const cur = current();
  const list = h("div.rlist", {}, R.list.length ? R.list.map(r => recipeRow(r, cur)) : [h("p.dim", {"data-i18n": "rcp.empty"})]);
  const name = h("input.find", {type: "text", "data-i18n-placeholder": "rcp.newName", "data-i18n-aria": "rcp.newName"});
  const withP = promptBox();
  return [
    h("div.hd", {}, h("h3.title", {}, "Recipe"),
      R.hooks.openAssets ? h("button.pill", {type: "button", "data-i18n": "rcp.manage", on: {click: () => { closePop(); R.hooks.openAssets("recipe"); }}}) : null,
      h("a.pill", {href: "#recombine", on: {click: () => closePop()}}, t("rcp.toRecombine"))),
    list,
    h("div.row", {}, h("span.lbl", {"data-i18n": "rcp.saveCurrent"}),
      h("div.newset", {}, name, h("button.pill.primary", {type: "button", "data-i18n": "rcp.save", on: {click: async () => {
        const id = await saveRecipe(name.value.trim() || t("rcp.untitled"), withPrompt(currentRecipe(), withP.checked), R.hooks.coverId?.());
        R.ui.recipe = id;
        post("/api/ui", {recipe: id}).catch(() => {});
        await loadRecipes();
        closePop();
        refreshBar();
      }}}))),
    h("label.rcpprompt", {}, withP, h("span", {"data-i18n": "rcp.withPrompt"}))];
}

// 「包含提示詞」的勾選框（預設勾）
function promptBox(on = true) {
  return h("input", {type: "checkbox", checked: on});
}

function recipeRow(r, cur) {
  const isCur = cur?.id === r.id;
  let armed = null;
  const del = h("button.mini", {type: "button", "data-i18n": "rcp.delete", on: {click: async e => {
    const b = e.currentTarget;
    if (!armed) {
      b.textContent = t("rcp.deleteSure");
      b.classList.add("armed");
      armed = setTimeout(() => { armed = null; b.textContent = t("rcp.delete"); b.classList.remove("armed"); }, 4000);
      return;
    }
    clearTimeout(armed);
    await post("/api/recipes/delete", {id: r.id});
    if (R.ui.recipe === r.id) { R.ui.recipe = ""; post("/api/ui", {recipe: ""}).catch(() => {}); }
    await loadRecipes();
    b.closest(".rrow").remove();
    refreshBar();
  }}});
  return h("div.rrow", {"data-cur": isCur ? "1" : null},
    h("button.rmain", {type: "button", on: {click: () => { closePop(); applyRecipe(r.data, {id: r.id, name: r.name}); }}},
      r.cover ? h("img", {src: `/api/thumb?id=${r.cover.id}&v=${encodeURIComponent(r.cover.v)}`, alt: ""}) : h("span.rglyph", {text: "R"}),
      h("span.rtxt", {}, h("b", {text: r.name || t("rcp.untitled")}), h("span.dim", {text: summary(r.data)}))),
    h("div.racts", {},
      isCur ? h("button.mini", {type: "button", "data-i18n": "rcp.unapply", "data-i18n-title": "rcp.unapplyHint",
                                on: {click: () => { closePop(); unapplyRecipe(); }}}) : null,
      isCur ? h("button.mini", {type: "button", "data-i18n": "rcp.update", "data-i18n-title": "rcp.updateHint", on: {click: async e => {
        await post("/api/recipes/save", {id: r.id, data: withPrompt(currentRecipe(), r.data?.prompt != null)});
        await loadRecipes();
        e.currentTarget.textContent = t("rcp.updated");
        refreshBar();
      }}}) : null,
      h("button.mini", {type: "button", "data-i18n": "rcp.rename", on: {click: e => {
        const row = e.currentTarget.closest(".rrow"), b = row.querySelector(".rtxt b");
        const inp = h("input.find", {type: "text", value: r.name, on: {
          keydown: ev => { if (ev.key === "Enter") ev.currentTarget.blur(); if (ev.key === "Escape") { ev.stopPropagation(); ev.preventDefault(); ev.currentTarget.value = r.name; ev.currentTarget.blur(); } },
          blur: async ev => {
            const v = ev.currentTarget.value.trim();
            if (v && v !== r.name) { r.name = v; await post("/api/recipes/save", {id: r.id, name: v}); await loadRecipes(); refreshBar(); }
            b.textContent = r.name;
            ev.currentTarget.replaceWith(b);
          }}});
        b.replaceWith(inp);
        inp.focus();
        inp.select();
      }}}),
      del));
}

export async function saveRecipe(name, data, coverId) {
  const d = await post("/api/recipes/save", {name, data, cover: coverId ?? null});
  return d.id;
}

// 從一張圖存成 Recipe（看大圖、比較模式用）
export function saveFromImagePop(anchor, img, onDone) {
  openPop("popSaveRecipe", anchor, () => {
    const name = h("input.find", {type: "text", value: stem(img.filename || ""), "data-i18n-aria": "rcp.newName"});
    const full = recipeFromImage(img);
    const withP = promptBox();
    const sum = h("p.dim.small", {text: summary(full)});
    withP.addEventListener("change", () => { sum.textContent = summary(withPrompt(full, withP.checked)); });
    return [
      h("div.hd", {}, h("h3.title", {"data-i18n": "rcp.fromImage"})),
      sum,
      full.prompt ? h("label.rcpprompt", {}, withP, h("span", {"data-i18n": "rcp.withPrompt"})) : null,
      h("div.newset", {}, name, h("button.pill.primary", {type: "button", "data-i18n": "rcp.save", on: {click: async () => {
        const data = withPrompt({...full}, withP.checked);
        // 先把名字換成檔案路徑，之後套用就不必再找
        for (const k of ["model", "vae", "te"]) if (data[k]) data[k] = (await resolve(k, data[k])) || data[k];
        const id = await saveRecipe(name.value.trim() || t("rcp.untitled"), data, img.id);
        await loadRecipes();
        closePop();
        onDone?.(id);
      }}}))];
  });
}

