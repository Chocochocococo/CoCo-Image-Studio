// 資源庫的「Recipe」分頁（使用者筆記：Recipe 越來越多要能管理）。跟其他分頁一樣：
// 點卡片就是套用（已經套用的再點一次是取消）、右鍵只看詳細；套用中的排在牆上方的「使用中」一列，
// 牆的順序打開分頁時排好就不動（以前套用了就跳到第一格）；
// 詳細欄可以點名字改名、用目前生成台的設定更新、複製、刪除（按兩次）。
// 資料就是 recipes.js 的 R.list；assets.js 在「Recipe」分頁時把工具列、牆、詳細欄交給這裡畫

import {post} from "./api.js";
import {h} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {R, applyRecipe, currentRecipe, loadRecipes, saveRecipe, summary, unapplyRecipe} from "./recipes.js";

const SORTS = ["name", "used", "updated"];
const Q = {q: "", sort: "name", sel: null, arm: null,
           order: []};                  // 牆的順序（Recipe 的 id）：打開分頁、換排序時才重排
let redraw = () => {};

const low = s => String(s || "").toLowerCase();
const nameOf = r => r.name || t("rcp.untitled");
const cur = () => R.ui?.recipe || "";

// assets.js 給的「整個分頁重畫」
export function bindRecipeTab(fn) { redraw = fn; }

export async function loadRecipeTab() {
  await loadRecipes();
  if (Q.sel && !R.list.some(r => r.id === Q.sel)) Q.sel = null;
  if (!Q.sel) Q.sel = R.list.find(r => r.id === cur())?.id || null;
  sortNow();
  return R.list;
}

function sortNow() {
  const by = {
    used: (a, b) => (b.used || 0) - (a.used || 0) || b.updated - a.updated,
    updated: (a, b) => b.updated - a.updated,
    name: (a, b) => nameOf(a).localeCompare(nameOf(b), lang()),
  }[Q.sort];
  Q.order = [...R.list].sort(by).map(r => r.id);
}

export function recipeTools(box) {
  box.replaceChildren(
    h("input.find", {id: "assetSearch", type: "search", value: Q.q, "data-i18n-placeholder": "rcl.search", "data-i18n-aria": "rcl.search",
                     autocomplete: "off", spellcheck: "false", on: {input: e => { Q.q = e.target.value; redraw("grid"); }}}),
    h("label.asort", {},
      h("span", {"data-i18n": "lib.sort"}),
      h("select.chip", {"data-i18n-aria": "lib.sort", on: {change: e => { Q.sort = e.target.value; sortNow(); redraw("grid"); }}},
        SORTS.map(k => h("option", {value: k, text: t(`rcl.sort.${k}`), selected: Q.sort === k})))));
  translate(box);
}

// 照排好的順序；後來才出現的（例如剛複製的）接在最後
function visible() {
  const q = low(Q.q).trim();
  const at = new Map(Q.order.map((id, i) => [id, i]));
  return R.list.filter(r => !q || low(r.name).includes(q) || low(summary(r.data)).includes(q) || low(r.data?.prompt).includes(q))
    .sort((a, b) => (at.get(a.id) ?? 1e9) - (at.get(b.id) ?? 1e9) || b.updated - a.updated);
}

const pic = r => (r.cover ? h("img", {src: `/api/thumb?id=${r.cover.id}&v=${encodeURIComponent(r.cover.v)}`, alt: "", loading: "lazy"})
  : h("span.aglyph", {text: "R"}));

// 點卡片：沒套用就套用，已經套用的就取消
async function apply(r) {
  Q.sel = r.id;
  if (r.id === cur()) await unapplyRecipe();
  else await applyRecipe(r.data, {id: r.id, name: r.name});
  redraw("all");
}

// 牆上方的「使用中」：套用中的那個 Recipe，點一下看詳細、× 取消套用（跟其他分頁一樣）
export function recipeInUse(box) {
  const r = R.list.find(x => x.id === cur());
  box.hidden = !r;
  if (!r) { box.replaceChildren(); return; }
  box.replaceChildren(
    h("span.k", {"data-i18n": "assets.inUse"}),
    h("div.amini", {role: "button", tabindex: "0", title: nameOf(r), "data-key": r.id, "data-sel": Q.sel === r.id ? "1" : null,
                    on: {click: e => { if (!e.target.closest(".mx")) { Q.sel = r.id; redraw("sel"); } },
                         keydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); Q.sel = r.id; redraw("sel"); } }}},
      h("span.mpic", {}, pic(r)),
      h("span.mname", {text: nameOf(r)}),
      h("button.mx", {type: "button", title: t("rcp.unapply"), "aria-label": t("rcp.unapply"),
                      on: {click: async () => { await unapplyRecipe(); redraw("all"); }}}, "×")));
  translate(box);
}

export function recipeGrid(grid) {
  const list = visible();
  grid.replaceChildren(...list.map(r => h("div.acard", {
    role: "option", tabindex: "0", "aria-selected": String(r.id === cur()), "data-key": r.id, "data-sel": Q.sel === r.id ? "1" : null,
    title: `${nameOf(r)}\n${t("rcl.how")}`,
    on: {click: () => apply(r),
         contextmenu: e => { e.preventDefault(); Q.sel = r.id; redraw("sel"); },
         keydown: e => {
           if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); Q.sel = r.id; redraw("sel"); }
           else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); apply(r); }
         }}},
    h("div.acov", {}, pic(r), r.id === cur() ? h("span.tape", {"data-i18n": "assets.inUse"}) : null),
    h("div.ainfo", {}, h("b.aname", {text: nameOf(r)}), h("div.ameta", {}, h("span", {text: summary(r.data)}))))));
  if (!list.length) grid.append(h("p.dim.aempty", {text: t(Q.q ? "assets.noMatch" : "rcp.empty")}));
  translate(grid);
}

export function markRecipeSel() {
  for (const el of document.querySelectorAll("#assetGrid .acard[data-key], #assetInUse .amini[data-key]")) {
    el.toggleAttribute("data-sel", el.dataset.key === Q.sel);
  }
}

async function rename(r, name) {
  if (!name || name === r.name) return;
  r.name = name;
  await post("/api/recipes/save", {id: r.id, name});
  await loadRecipes();
  redraw("all");
}

export function recipeDetail(box) {
  const r = R.list.find(x => x.id === Q.sel);
  if (!r) { box.replaceChildren(h("p.dim", {"data-i18n": "rcl.hint"})); translate(box); return; }
  const on = r.id === cur(), d = r.data || {};
  const del = h("button.pill", {type: "button", "data-i18n": "rcp.delete", on: {click: async e => {
    const b = e.currentTarget;
    if (!Q.arm) {
      b.textContent = t("rcp.deleteSure");
      b.classList.add("armed");
      Q.arm = setTimeout(() => { Q.arm = null; b.textContent = t("rcp.delete"); b.classList.remove("armed"); }, 4000);
      return;
    }
    clearTimeout(Q.arm);
    Q.arm = null;
    await post("/api/recipes/delete", {id: r.id});
    if (R.ui.recipe === r.id) { R.ui.recipe = ""; post("/api/ui", {recipe: ""}).catch(() => {}); }
    Q.sel = null;
    await loadRecipes();
    redraw("all");
  }}});
  box.replaceChildren(...[
    h("div.adpic", {}, pic(r)),
    h("input.rclname", {type: "text", value: r.name || "", "data-i18n-placeholder": "rcp.untitled", "data-i18n-aria": "rcp.rename",
                        "data-i18n-title": "rcb.renameHint",
                        on: {change: e => rename(r, e.target.value.trim()),
                             keydown: e => {
                               if (e.key === "Enter") e.currentTarget.blur();
                               if (e.key === "Escape") { e.stopPropagation(); e.currentTarget.value = r.name || ""; e.currentTarget.blur(); }
                             }}}),
    h("p.dim.small", {text: summary(d)}),
    h("div.vacts", {},
      h("button.pill.primary", {type: "button", on: {click: () => apply(r)}}, t(on ? "rcp.unapply" : "rcl.apply")),
      h("button.pill", {type: "button", "data-i18n": "rcl.update", "data-i18n-title": "rcp.updateHint", on: {click: async e => {
        const data = currentRecipe();
        if (d.prompt == null) delete data.prompt;                 // 原本沒存提示詞的照樣不存
        await post("/api/recipes/save", {id: r.id, data});
        await loadRecipes();
        e.currentTarget.textContent = t("rcp.updated");
        redraw("grid");
      }}}),
      h("button.pill", {type: "button", "data-i18n": "rcl.copy", on: {click: async () => {
        Q.sel = await saveRecipe(t("rcl.copyName", {name: nameOf(r)}), d, r.cover?.id ?? null);
        await loadRecipes();
        redraw("all");
      }}}),
      del),
    d.prompt ? h("section.vsec", {}, h("h3", {"data-i18n": "gen.prompt"}), h("pre.rclprompt", {text: d.prompt})) : null,
    d.negative ? h("section.vsec", {}, h("h3", {"data-i18n": "gen.negative"}), h("pre.rclprompt", {text: d.negative})) : null,
    r.used ? h("p.dim.small", {text: t("assets.lastUsed", {when: new Date(r.used * 1000).toLocaleString(lang(), {dateStyle: "medium", timeStyle: "short"})})}) : null,
  ].filter(Boolean));
  translate(box);
}
