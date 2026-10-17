// 重組（工作室的 04）：把幾個來源排成欄——目前的生成台、Recipe、比較組裡的勝出圖——
// 每一列是一個部分（模型、VAE、編碼器、取樣設定、尺寸、負面提示詞），點格子決定這部分用哪一欄的；
// LoRA 可以跨來源混搭、各自調權重。組好之後套用到生成台，或存成新的 Recipe。
// 提示詞拆成標籤、依標籤庫的分類分組（promptmix.js），每一組挑一個或好幾個來源的（例如頭髮拿 A、服裝拿 B）。
// 只有生成台與比較組的圖有提示詞；Recipe 本來就不存提示詞，存成 Recipe 時提示詞也不會存進去。

import {get, post} from "./api.js";
import {tagGrouper} from "./chips.js";
import {genSettings} from "./generate.js";
import {GROUPS, mix, splitByGroup} from "./promptmix.js";
import {parsePrompt} from "./prompttext.js";
import {$, h, stem} from "./dom.js";
import {t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {fmtWeight} from "./options.js";
import {R, applyRecipe, currentRecipe, loadRecipes, recipeFromImage, saveRecipe, summary} from "./recipes.js";

const MAX_COLS = 5;
// 單選的列：鍵、顯示文字、這列包含的欄位、怎麼顯示
const ROWS = [
  ["model", "rcb.row.model", ["model", "arch"], d => (d.model ? stem(d.model) : null)],
  ["vae", "rcb.row.vae", ["vae"], d => part(d.vae)],
  ["te", "rcb.row.te", ["te"], d => part(d.te)],
  ["sampling", "rcb.row.sampling", ["sampler", "schedule", "steps", "cfg", "clip_skip", "shift"],
   d => (d.sampler || d.steps ? [d.sampler, d.schedule, d.steps && t("rcp.steps", {n: d.steps}), d.cfg != null && `CFG ${d.cfg}`]
     .filter(Boolean).join(" · ") : null)],
  ["size", "rcb.row.size", ["width", "height"], d => (d.width && d.height ? `${d.width}×${d.height}` : null)],
  ["negative", "rcb.row.negative", ["negative"], d => d.negative || null],
];
const part = v => (!v ? null : v === "builtin" ? t("gen.slot.builtin") : stem(v));

const X = {cols: [], pick: {}, loras: [], winners: null, saveName: "", loraArch: null,
           ppick: {},                   // 提示詞：{組: [來源序號…]}
           known: new Set(), groupOf: () => ""};   // 查過分類的標籤、查表函式

// ── 來源 ──
function curCol() {
  return {key: "cur", name: t("rcb.current"), data: {...currentRecipe(), prompt: genSettings()?.PROMPT || ""}, img: null};
}

async function loadWinners() {
  X.winners = [];
  try {
    const sets = (await get("/api/compare/sets")).sets || [];
    for (const s of sets.filter(x => x.winners)) {
      const d = await get(`/api/compare/set?id=${encodeURIComponent(s.id)}`);
      for (const it of d.items || []) {
        // 提示詞：開著品質標籤生成的圖用使用者自己寫的那份
        if (it.winner && it.img) X.winners.push({key: `w:${it.path}`, name: it.img.filename, set: s.name || t("cmp.untitled"),
                                                  data: {...recipeFromImage(it.img), prompt: it.img.extra?.user_prompt ?? it.img.prompt ?? ""},
                                                  img: it.img});
      }
    }
  } catch { /* 比較組讀不到就只剩 Recipe */ }
}

// LoRA 名稱 → 架構（資源庫從檔頭判斷的），拿來標出跟選的模型不相容的 LoRA
// 模型也一樣：從圖存的 Recipe 沒有記架構，要靠模型名字查
async function loadLoraArch() {
  X.loraArch = new Map();
  X.modelArch = new Map();
  try {
    const [l, m] = await Promise.all([get("/api/assets?kind=lora"), get("/api/assets?kind=model")]);
    for (const it of l.items || []) X.loraArch.set(it.name.toLowerCase(), it.arch);
    for (const it of m.items || []) X.modelArch.set(it.name.toLowerCase(), it.arch);
  } catch { /* 讀不到就不標 */ }
}

export async function showRecombine() {
  await Promise.all([loadRecipes(), X.winners ? null : loadWinners(), X.loraArch ? null : loadLoraArch()]);
  // 第一欄永遠是目前的生成台（每次進來都重讀，生成台可能改過）
  X.cols = [curCol(), ...X.cols.filter(c => c.key !== "cur")];
  if (!Object.keys(X.pick).length) for (const [k] of ROWS) X.pick[k] = 0;
  if (!Object.keys(X.ppick).length) for (const g of GROUPS) X.ppick[g] = [0];
  if (!X.loras.length) X.loras = X.cols[0].data.loras.map(l => ({name: l.name, weight: l.weight, from: 0}));
  draw();
  classify();
}

// 提示詞的標籤要查過分類才分得出組：還沒查到的先歸「其他」，查完重畫
async function classify() {
  const names = [];
  for (const c of X.cols) {
    for (const it of parsePrompt(c.data.prompt || "")) if (it.kind === "tag" && !X.known.has(it.text)) names.push(it.text);
  }
  if (!names.length) return;
  for (const n of names) X.known.add(n);
  try { X.groupOf = await tagGrouper([...X.known]); } catch { return; }
  draw();
}

const parts = () => X.cols.map(c => (c.data.prompt ? splitByGroup(c.data.prompt, X.groupOf) : null));

export function recombineScreen() {
  return h("div.cmp.rcb", {},
    h("aside.card.csets", {"aria-labelledby": "rcbSrc"},
      h("h2.title", {id: "rcbSrc", "data-i18n": "rcb.sources"}),
      h("div.rsrc", {id: "rcbSources"})),
    h("section.card.cmain", {id: "rcbMain"}));
}

function toggleCol(src) {
  const i = X.cols.findIndex(c => c.key === src.key);
  if (i > 0) {
    X.cols.splice(i, 1);
    // 指到這欄或後面的選擇往前挪；指到被拿掉那欄的改回第一欄
    for (const k of Object.keys(X.pick)) X.pick[k] = X.pick[k] === i ? 0 : X.pick[k] > i ? X.pick[k] - 1 : X.pick[k];
    for (const g of Object.keys(X.ppick)) X.ppick[g] = X.ppick[g].filter(x => x !== i).map(x => (x > i ? x - 1 : x));
    X.loras = X.loras.filter(l => l.from !== i).map(l => ({...l, from: l.from > i ? l.from - 1 : l.from}));
  } else if (X.cols.length < MAX_COLS) {
    X.cols.push(src);
  }
  draw();
  classify();
}

function srcCard(src, sub) {
  const on = X.cols.some(c => c.key === src.key);
  return h("button.rsrcard", {type: "button", "aria-pressed": String(on), disabled: !on && X.cols.length >= MAX_COLS,
                              on: {click: () => toggleCol(src)}},
    src.cover ? h("img", {src: `/api/thumb?id=${src.cover.id}&v=${encodeURIComponent(src.cover.v)}`, alt: ""})
      : src.img ? h("img", {src: `/api/thumb?id=${src.img.id}&v=${encodeURIComponent(src.img.v)}`, alt: ""})
      : h("span.rglyph", {text: "R"}),
    h("span.rtxt", {}, h("b", {text: src.name}), h("span.dim", {text: sub})));
}

function drawSources() {
  const box = $("rcbSources");
  if (!box) return;
  const recipes = R.list.map(r => ({key: `r:${r.id}`, name: r.name || t("rcp.untitled"), data: r.data, cover: r.cover}));
  const [btn, hint] = info("rcb.pickHint", {n: MAX_COLS - 1});
  box.replaceChildren(
    h("h3", {}, t("rcb.recipes"), btn),
    hint,
    recipes.length ? h("div.rsrclist", {}, recipes.map(r => srcCard(r, summary(r.data))))
      : h("p.dim", {"data-i18n": "rcp.empty"}),
    h("h3", {"data-i18n": "rcb.winners"}),
    X.winners?.length ? h("div.rsrclist", {}, X.winners.map(w => srcCard(w, `${w.set} · ${summary(w.data)}`)))
      : h("p.dim", {"data-i18n": "rcb.noWinners"}));
  translate(box);
}

// ── 組合 ──
function assemble() {
  const out = {};
  for (const [key, , fields] of ROWS) {
    const d = X.cols[X.pick[key]]?.data || {};
    for (const f of fields) if (d[f] != null && d[f] !== "") out[f] = d[f];
  }
  out.loras = X.loras.filter(l => l.from != null).map(l => ({name: l.name, weight: Number(l.weight)}));
  // 有提示詞的來源才組提示詞（只有 Recipe 的話就不動生成台的提示詞）
  if (X.cols.some(c => c.data.prompt)) out.prompt = mix(parts(), X.ppick);
  return out;
}

// ── 提示詞：一組一列，每一格是那個來源在這一組的標籤，點了加進或拿掉（可以複選）──
function promptRows(cols) {
  const ps = parts();
  if (!ps.some(Boolean)) return [];
  const label = g => t(g === "other" ? "rcb.cat.other" : `chip.cat.${g}`);
  const rows = GROUPS.filter(g => ps.some(p => p?.[g]?.length)).map(g => {
    const picked = X.ppick[g] || [];
    const res = mix(ps, {[g]: picked});
    return h("tr.prow", {},
      h("th", {text: label(g)}),
      cols.map((c, i) => {
        const tags = ps[i]?.[g];
        return h("td", {}, tags?.length
          ? h("button.cell.ptags", {type: "button", "aria-pressed": String(picked.includes(i)), title: tags.join(", "),
                                    on: {click: () => {
                                      X.ppick[g] = picked.includes(i) ? picked.filter(x => x !== i) : [...picked, i];
                                      draw();
                                    }}}, tags.join(", "))
          : h("span.dim", {text: "—"}));
      }),
      h("td.rres", {}, res ? h("span.ptags", {text: res, title: res}) : h("span.dim", {"data-i18n": "rcb.off"})));
  });
  const [btn, hint] = info("rcb.promptHint");
  return [
    h("tr.lhead", {}, h("th", {colspan: String(cols.length + 2)}, h("span", {}, t("rcb.prompt"), btn), hint)),
    ...rows];
}

function coverId() {
  // 封面：模型那列選的那欄如果是一張圖就用它，不然用第一張有圖的欄
  const c = X.cols[X.pick.model];
  return c?.img?.id ?? c?.cover?.id ?? X.cols.find(x => x.img)?.img.id ?? X.cols.find(x => x.cover)?.cover.id ?? null;
}

// Recipe 改名（使用者筆記：重組頁更直觀）：欄的標題點一下直接改，跟比較組的標題一樣
async function renameRecipe(col, name) {
  const id = col.key.slice(2);
  if (!name || name === col.name) return;
  col.name = name;
  await post("/api/recipes/save", {id, name});
  await loadRecipes();
  draw();
}

function drawMatrix() {
  const cols = X.cols;
  const head = h("tr", {}, h("th"), cols.map((c, i) => h("th.rcol", {},
    c.key.startsWith("r:")
      ? h("input.rname.rnamein", {type: "text", value: c.name, "data-i18n-aria": "rcp.rename", "data-i18n-title": "rcb.renameHint",
                                   on: {change: e => renameRecipe(c, e.target.value.trim()),
                                        keydown: e => {
                                          if (e.key === "Enter") e.currentTarget.blur();
                                          if (e.key === "Escape") { e.currentTarget.value = c.name; e.currentTarget.blur(); }
                                        }}})
      : h("span.rname", {text: c.name, title: c.name}),
    i > 0 ? h("button.mini", {type: "button", "data-i18n-aria": "cmp.remove", title: t("cmp.remove"),
                               on: {click: () => toggleCol(c)}}, "✕") : null)),
    h("th.rres", {"data-i18n": "rcb.result"}));
  const res = assemble();
  const rows = ROWS.map(([key, lab, , show]) => {
    const vals = cols.map(c => show(c.data || {}));
    const same = new Set(vals.filter(v => v != null)).size <= 1;
    return h("tr", {"data-same": same ? "1" : null},
      h("th", {"data-i18n": lab}),
      vals.map((v, i) => h("td", {},
        v == null ? h("span.dim", {text: "—"})
          : h("button.cell", {type: "button", "aria-pressed": String(X.pick[key] === i), title: v,
                              on: {click: () => { X.pick[key] = i; draw(); }}}, v))),
      h("td.rres", {text: show(res) ?? t("rcb.keep"), title: show(res) ?? ""}));
  });
  // LoRA：所有來源出現過的都列出來，一個 LoRA 一列
  const names = [];
  for (const c of cols) for (const l of c.data.loras || []) if (!names.some(n => n.toLowerCase() === l.name.toLowerCase())) names.push(l.name);
  for (const l of X.loras) if (!names.some(n => n.toLowerCase() === l.name.toLowerCase())) names.push(l.name);
  const md = X.cols[X.pick.model]?.data || {};
  const modelArch = md.arch || (md.model && X.modelArch?.get(stem(md.model).toLowerCase())) || "";
  const loraRows = names.map(name => {
    const mine = X.loras.find(l => l.name.toLowerCase() === name.toLowerCase());
    const la = X.loraArch?.get(name.toLowerCase()) || "";
    const bad = mine && mine.from != null && modelArch && la && la !== modelArch;
    return h("tr.lrow", {},
      h("th", {text: name, title: name}),
      cols.map((c, i) => {
        const l = (c.data.loras || []).find(x => x.name.toLowerCase() === name.toLowerCase());
        return h("td", {}, l ? h("button.cell", {type: "button", "aria-pressed": String(mine?.from === i),
                                                 on: {click: () => {
                                                   if (mine?.from === i) mine.from = null;
                                                   else if (mine) Object.assign(mine, {from: i, weight: l.weight});
                                                   else X.loras.push({name: l.name, weight: l.weight, from: i});
                                                   draw();
                                                 }}}, fmtWeight(l.weight)) : h("span.dim", {text: "—"}));
      }),
      h("td.rres", {}, mine && mine.from != null
        ? [h("input.rw", {type: "number", step: "0.05", min: "-2", max: "3", value: fmtWeight(mine.weight), "data-i18n-aria": "assets.weight",
                          on: {change: e => { mine.weight = Number(e.target.value) || 0; }}}),
           bad ? h("span.bad", {text: ` ${t("assets.incompatible")}`, title: t("assets.incompatibleWith", {arch: modelArch})}) : null]
        : h("span.dim", {"data-i18n": "rcb.off"})));
  });
  return h("div.ctable.rtable", {},
    h("table", {},
      h("thead", {}, head),
      h("tbody", {}, rows,
        promptRows(cols),
        h("tr.lhead", {}, h("th", {colspan: String(cols.length + 2)}, h("span", {"data-i18n": "rcb.loras"}),
          h("span.dim", {"data-i18n": "rcb.lorasHint"}))),
        loraRows.length ? loraRows : h("tr", {}, h("td.dim", {colspan: String(cols.length + 2), "data-i18n": "rcb.noLoras"})))));
}

function draw() {
  drawSources();
  const main = $("rcbMain");
  if (!main) return;
  const name = h("input.find", {type: "text", value: X.saveName, "data-i18n-placeholder": "rcp.newName", "data-i18n-aria": "rcp.newName",
                                on: {input: e => { X.saveName = e.target.value; }}});
  const msg = h("span.msg", {id: "rcbMsg", role: "status"});
  // 整頁重畫：表格框換新的，捲動位置要自己留住（使用者回報：點一個提示詞就捲回最上面）
  const old = main.querySelector(".rtable");
  const keep = old ? [old.scrollTop, old.scrollLeft] : null, keepMain = main.scrollTop;
  main.replaceChildren(
    h("div.chd", {},
      h("h2.title", {"data-i18n": "nav.recombine"}),
      h("span.dim", {"data-i18n": "rcb.intro"}),
      h("span.sp"),
      h("button.pill.primary", {type: "button", "data-i18n": "rcb.apply", on: {click: async () => {
        await applyRecipe(assemble(), {name: t("rcp.combo")});
        location.hash = "#generate";
      }}})),
    drawMatrix(),
    promptPreview(),
    h("div.rsave", {},
      h("span.lbl", {"data-i18n": "rcb.saveAs"}), name,
      h("button.pill", {type: "button", "data-i18n": "rcp.save", on: {click: async () => {
        const id = await saveRecipe(X.saveName.trim() || t("rcp.untitled"), assemble(), coverId());
        await loadRecipes();
        msg.textContent = t("rcb.saved");
        msg.dataset.kind = "ok";
        drawSources();
        return id;
      }}}), msg));
  translate(main);
  main.scrollTop = keepMain;
  const now = main.querySelector(".rtable");
  if (now && keep) [now.scrollTop, now.scrollLeft] = keep;
}

function promptPreview() {
  if (!X.cols.some(c => c.data.prompt)) return null;
  const text = mix(parts(), X.ppick);
  return h("div.rprompt", {},
    h("span.lbl", {"data-i18n": "rcb.promptOut"}),
    h("pre.rptext", {text: text || t("rcb.promptEmpty")}));
}

// 從比較模式標了勝出之後進來：重新讀勝出清單
export function resetWinners() { X.winners = null; }
