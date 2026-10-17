// 資源庫的「標籤」分頁：你的標籤庫（使用者筆記 #1、#2；後端 server/usertags.py）。
// 四個檢視：
//   我的標籤  內建庫沒有的詞（角色名、LoRA 觸發詞、自創的詞），可以設分類（標籤模式的顏色）與中文
//   組合      一串標籤存成一組，點一下整組加進提示詞（萬用字元是每次抽一個，這是整組一起用）
//   收藏      內建庫的標籤加星號：自動完成排前面
//   待確認    圖庫裡出現、內建庫沒有的詞，依出現張數排序；收下才算，略過的不再出現
// 跟其他分頁一樣：點卡片加進生成台正在編輯的那一邊（正面或負面），再點一次拿掉；右鍵只看詳細、編輯。
// assets.js 在「標籤」分頁時把工具列、牆、詳細欄交給這裡畫

import {get, post} from "./api.js";
import {h} from "./dom.js";
import {editingField, genSettings, setField} from "./generate.js";
import {lang, t, translate} from "./i18n.js";
import {searchLibrary} from "./library.js";
import {addTags, dropTags, hasTag, parseChip, splitTags, tagOut} from "./prompttext.js";

const VIEWS = ["mine", "group", "fav", "inbox"];
// 分類＝標籤模式的十種顏色（chips.js 的 CAT_GROUP），存的是內建庫的分類名
const CATS = [["subject", "subject"], ["hair", "hair"], ["eyes", "eyes"], ["body", "appearance"], ["wear", "clothing"],
              ["face", "expression"], ["pose", "pose"], ["scene", "scene"], ["meta", "meta"], ["who", "character"]];
const GROUP_OF = {subject: "subject", species: "subject", age: "subject", profession: "subject", hair: "hair", eyes: "eyes",
                  appearance: "body", clothing: "wear", accessory: "wear", expression: "face", mood: "face", pose: "pose",
                  action: "pose", scene: "scene", lighting: "scene", weather_time: "scene", composition: "scene", effect: "scene",
                  object: "scene", quality: "meta", meta: "meta", negative: "meta", artist: "who", character: "who",
                  copyright: "who", lore: "who", contributor: "who"};
const SORTS = ["name", "uses", "added"];

const Q = {view: "mine", q: "", cat: "", folder: "", sort: "name", sel: null, data: {tags: [], groups: [], ignored: 0},
           inbox: null, inboxSel: new Set(), sentences: false, showIgnored: false, ignoredList: [], arm: null};
let redraw = () => {};

const low = s => String(s || "").toLowerCase();
const show = name => tagOut(name);                                  // 標籤庫的寫法 → 提示詞的寫法（括號要跳脫）
const label = name => show(name).replace(/\\([()])/g, "$1");          // 畫面上顯示的（不要跳脫的反斜線）
const side = () => editingField();
const prompt = () => genSettings()?.[side()] || "";
const catOf = it => it.cat || it.base_cat || "";
const zhOf = it => it.zh || it.base_zh || "";

export function bindTagTab(fn) { redraw = fn; }

export async function loadTagTab() {
  Q.data = await get("/api/mytags").catch(() => ({tags: [], groups: [], ignored: 0}));
  Q.inbox = null;
  if (Q.view === "inbox") await loadInbox();
  return Q.data;
}

async function loadInbox() {
  Q.inbox = await get("/api/mytags/candidates").catch(() => ({items: [], total: 0}));
  const names = new Set(Q.inbox.items.map(x => x.name));
  for (const n of [...Q.inboxSel]) if (!names.has(n)) Q.inboxSel.delete(n);
}

const tags = own => Q.data.tags.filter(x => own ? x.own : !x.own && x.fav);
const folders = () => [...new Set(Q.data.groups.map(g => g.folder).filter(Boolean))].sort((a, b) => a.localeCompare(b, lang()));

// ── 左邊：檢視、搜尋、篩選 ──
export function tagTools(box) {
  const count = {mine: tags(true).length, group: Q.data.groups.length, fav: tags(false).length,
                 inbox: Q.inbox ? Q.inbox.total : null};
  const viewBtn = v => h("button.chip", {type: "button", "aria-pressed": String(Q.view === v), on: {click: () => switchView(v)}},
    h("span", {"data-i18n": `tlib.view.${v}`}), count[v] != null ? h("span.c", {text: ` ${count[v]}`}) : null);
  const search = h("input.find", {id: "assetSearch", type: "search", value: Q.q, "data-i18n-placeholder": "tlib.search",
                                  "data-i18n-aria": "tlib.search", autocomplete: "off", spellcheck: "false",
                                  on: {input: e => { Q.q = e.target.value; redraw("grid"); }}});
  const isTags = Q.view === "mine" || Q.view === "fav";
  box.replaceChildren(...[
    h("div.tlviews", {role: "group", "data-i18n-aria": "tlib.views"}, VIEWS.map(viewBtn)),
    search,
    isTags ? h("label.asort", {}, h("span", {"data-i18n": "tlib.cat"}),
      h("select.chip", {"data-i18n-aria": "tlib.cat", on: {change: e => { Q.cat = e.target.value; redraw("grid"); }}},
        h("option", {value: "", text: t("tlib.catAll"), selected: !Q.cat}),
        CATS.map(([g]) => h("option", {value: g, text: t(`chip.cat.${g}`), selected: Q.cat === g})),
        h("option", {value: "-", text: t("tlib.catNone"), selected: Q.cat === "-"}))) : null,
    Q.view === "group" && folders().length ? h("label.asort", {}, h("span", {"data-i18n": "tlib.folder"}),
      h("select.chip", {"data-i18n-aria": "tlib.folder", on: {change: e => { Q.folder = e.target.value; redraw("grid"); }}},
        h("option", {value: "", text: t("tlib.folderAll"), selected: !Q.folder}),
        folders().map(f => h("option", {value: f, text: f, selected: Q.folder === f})))) : null,
    Q.view !== "inbox" ? h("label.asort", {}, h("span", {"data-i18n": "lib.sort"}),
      h("select.chip", {"data-i18n-aria": "lib.sort", on: {change: e => { Q.sort = e.target.value; redraw("grid"); }}},
        SORTS.map(k => h("option", {value: k, text: t(`tlib.sort.${k}`), selected: Q.sort === k})))) : null,
    Q.view !== "inbox" ? h("button.pill", {type: "button", on: {click: () => { Q.sel = {new: Q.view === "group" ? "group" : "mine"}; redraw("sel"); }}},
      t(Q.view === "group" ? "tlib.newGroup" : "tlib.newTag")) : null,
    Q.view === "fav" ? h("p.dim.small", {"data-i18n": "tlib.favHow"}) : null,
    Q.view === "inbox" ? h("label.tlcheck", {}, h("input", {type: "checkbox", checked: Q.sentences,
      on: {change: e => { Q.sentences = e.target.checked; redraw("grid"); }}}), h("span", {"data-i18n": "tlib.sentences"})) : null,
    Q.view === "inbox" && Q.data.ignored ? h("label.tlcheck", {}, h("input", {type: "checkbox", checked: Q.showIgnored,
      on: {change: async e => { Q.showIgnored = e.target.checked; if (Q.showIgnored) Q.ignoredList = (await get("/api/mytags/ignored")).items || []; redraw("all"); }}}),
      h("span", {text: t("tlib.showIgnored", {n: Q.data.ignored})})) : null,
  ].filter(Boolean));
  translate(box);
}

async function switchView(v) {
  Q.view = v;
  Q.q = "";
  Q.sel = null;
  if (v === "inbox" && !Q.inbox) {
    redraw("tools");
    redraw("loading");
    await loadInbox();
  }
  redraw("all");
}

// ── 牆 ──
function sorted(list, nameOf) {
  const by = {
    name: (a, b) => nameOf(a).localeCompare(nameOf(b), lang()),
    uses: (a, b) => (b.uses || 0) - (a.uses || 0) || (b.used || 0) - (a.used || 0),
    added: (a, b) => (b.added || 0) - (a.added || 0),
  }[Q.sort];
  return [...list].sort((a, b) => (b.fav || 0) - (a.fav || 0) || by(a, b));
}

function visibleTags() {
  const q = low(Q.q).trim().replace(/ /g, "_");
  return sorted(tags(Q.view === "mine").filter(it =>
    (!q || low(it.name).includes(q) || low(zhOf(it)).includes(low(Q.q).trim()) || low(it.note).includes(low(Q.q).trim()))
    && (!Q.cat || (Q.cat === "-" ? !GROUP_OF[catOf(it)] : GROUP_OF[catOf(it)] === Q.cat))), it => it.name);
}

function visibleGroups() {
  const q = low(Q.q).trim();
  return sorted(Q.data.groups.filter(g => (!q || low(g.name).includes(q) || low(g.text).includes(q) || low(g.folder).includes(q))
    && (!Q.folder || g.folder === Q.folder)), g => g.name);
}

const tagOn = it => hasTag(prompt(), show(it.name));
const groupParts = g => splitTags(g.text);
const groupOn = g => { const p = prompt(); const parts = groupParts(g); return parts.length > 0 && parts.every(x => hasTag(p, x)); };

function tagCard(it) {
  const on = tagOn(it), sel = Q.sel?.tag === it.name;
  return h("div.tlcard", {role: "option", tabindex: "0", "aria-selected": String(on), "data-sel": sel ? "1" : null,
                         title: `${label(it.name)}${zhOf(it) ? `\n${zhOf(it)}` : ""}\n${t("tlib.how")}`,
                         on: {click: () => toggleTag(it), contextmenu: e => { e.preventDefault(); pick({tag: it.name}); },
                              keydown: e => keyed(e, () => toggleTag(it), () => pick({tag: it.name}))}},
    h("span.dot", {"data-c": GROUP_OF[catOf(it)] || null}),
    h("span.tlname", {text: label(it.name)}),
    zhOf(it) ? h("span.tlzh", {text: zhOf(it)}) : null,
    it.fav && it.own ? h("span.tlstar", {text: "★"}) : null);
}

function groupCard(g) {
  const on = groupOn(g), sel = Q.sel?.group === g.id;
  return h("div.acard.tlgroup", {role: "option", tabindex: "0", "aria-selected": String(on), "data-sel": sel ? "1" : null,
                                title: `${g.name}\n${t("tlib.howGroup")}`,
                                on: {click: () => toggleGroup(g), contextmenu: e => { e.preventDefault(); pick({group: g.id}); },
                                     keydown: e => keyed(e, () => toggleGroup(g), () => pick({group: g.id}))}},
    h("div.ainfo", {},
      h("b.aname", {text: `${g.fav ? "★ " : ""}${g.name}`}),
      g.folder ? h("span.tlfolder", {text: g.folder}) : null,
      h("span.tltext", {text: g.text})),
    on ? h("span.tape", {"data-i18n": "assets.inUse"}) : null);
}

function keyed(e, use, look) {
  if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); look(); }
  else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); use(); }
}

function pick(sel) {
  Q.sel = sel;
  Q.arm = null;
  redraw("sel");
}

export function tagGrid(grid) {
  if (Q.view === "inbox") { inboxGrid(grid); return; }
  const list = Q.view === "group" ? visibleGroups() : visibleTags();
  grid.classList.toggle("tlgrid", Q.view !== "group");
  grid.classList.remove("tlinbox");
  grid.replaceChildren(...list.map(Q.view === "group" ? groupCard : tagCard));
  // 搜尋時也找內建標籤庫（中文、英文都行）：你的標籤下面另一區，已經收藏或是你的標籤的不重複列
  let base = [];
  if ((Q.view === "mine" || Q.view === "fav") && Q.q.trim()) {
    loadBase();
    const have = new Set(Q.data.tags.filter(x => x.own || x.fav).map(x => x.name));
    base = BASE.q === Q.q.trim() ? BASE.items.filter(x => !have.has(x.name)) : [];
    if (base.length) grid.append(h("p.tlsec", {text: t("tlib.base", {n: base.length})}), ...base.map(baseCard));
  }
  if (!list.length && !base.length) grid.append(h("p.dim.aempty", {text: t(Q.q || Q.cat || Q.folder ? "assets.noMatch" : `tlib.empty.${Q.view}`)}));
  translate(grid);
}

// 內建標籤庫的搜尋結果（伺服器 /api/tags，跟提示詞的自動完成同一套：有中文字就用中文找）。打字停 0.2 秒才查
const BASE = {q: "", items: [], timer: null};

function loadBase() {
  const q = Q.q.trim();
  if (BASE.q === q) return;
  clearTimeout(BASE.timer);
  BASE.timer = setTimeout(async () => {
    const d = await get(`/api/tags?q=${encodeURIComponent(q)}&limit=40`).catch(() => ({items: []}));
    if (Q.q.trim() !== q) return;
    BASE.q = q;
    BASE.items = (d.items || []).filter(x => !x.mine);
    redraw("grid");
  }, 200);
}

// 跟你的標籤一樣：點一下加進提示詞、再點一次拿掉；☆ 加進收藏（自動完成排前面）
function baseCard(x) {
  const it = {name: x.name, zh: x.zh || "", cat: x.cat || ""};
  return h("div.tlcard", {role: "option", tabindex: "0", "aria-selected": String(tagOn(it)),
                         title: `${label(it.name)}${it.zh ? `\n${it.zh}` : ""}\n${t("tlib.how")}`,
                         on: {click: () => toggleTag(it), keydown: e => keyed(e, () => toggleTag(it), () => {})}},
    h("span.dot", {"data-c": GROUP_OF[it.cat] || null}),
    h("span.tlname", {text: label(it.name)}),
    it.zh ? h("span.tlzh", {text: it.zh}) : null,
    h("button.tlfav", {type: "button", text: "☆", title: t("tlib.baseFav"), "aria-label": t("tlib.baseFav"),
                       on: {click: async e => { e.stopPropagation(); await saveTag({name: it.name, fav: true}); redraw("all"); }}}));
}

function inboxGrid(grid) {
  grid.classList.add("tlgrid", "tlinbox");
  if (!Q.inbox) { grid.replaceChildren(h("p.dim", {text: t("wall.loading")})); return; }
  if (Q.showIgnored) {
    grid.replaceChildren(...Q.ignoredList.map(n => h("div.tlcard.tlignored", {title: t("tlib.unignoreHint"),
      role: "button", tabindex: "0", on: {click: () => unignore(n), keydown: e => keyed(e, () => unignore(n), () => {})}},
      h("span.tlname", {text: label(n)}), h("span.tlzh", {"data-i18n": "tlib.unignore"}))));
    if (!Q.ignoredList.length) grid.append(h("p.dim.aempty", {"data-i18n": "tlib.noIgnored"}));
    translate(grid);
    return;
  }
  const q = low(Q.q).trim().replace(/ /g, "_");
  const list = Q.inbox.items.filter(x => (Q.sentences || x.words < 5) && (!q || x.name.includes(q)));
  grid.replaceChildren(...list.map(x => {
    const on = Q.inboxSel.has(x.name);
    return h("label.tlcand", {"data-on": on ? "1" : null, title: t("tlib.inboxHow")},
      h("input", {type: "checkbox", checked: on, on: {change: e => {
        if (e.target.checked) Q.inboxSel.add(x.name); else Q.inboxSel.delete(x.name);
        e.target.closest(".tlcand").toggleAttribute("data-on", e.target.checked);
        redraw("sel");
      }}}),
      h("img", {src: `/api/thumb?id=${x.sample.id}&v=${encodeURIComponent(x.sample.v)}`, alt: "", loading: "lazy"}),
      h("span.tlname", {text: label(x.name)}),
      h("span.tlcount", {text: t("tlib.images", {n: x.images})}));
  }));
  if (!list.length) grid.append(h("p.dim.aempty", {text: t(Q.q ? "assets.noMatch" : "tlib.empty.inbox")}));
  translate(grid);
}

// ── 加進提示詞／拿掉 ──
function edit(fn) {
  const k = side();
  const before = prompt();
  const after = fn(before);
  if (after !== before) setField(k, after);
}

function toggleTag(it) {
  const name = show(it.name);
  if (tagOn(it)) edit(p => dropTags(p, [name], true).prompt);
  else {
    edit(p => addTags(p, [name]));
    post("/api/mytags/used", {names: [it.name]}).catch(() => {});
    it.uses = (it.uses || 0) + 1;
  }
  redraw("grid");
  if (Q.sel?.tag === it.name) redraw("sel");
}

function toggleGroup(g) {
  const parts = groupParts(g);
  if (groupOn(g)) {
    edit(p => dropTags(p, parts.map(x => (parseChip(x).kind === "tag" ? parseChip(x).text : x)), true).prompt);
  } else {
    edit(p => { const add = parts.filter(x => !hasTag(p, x)); return add.length ? appendRaw(p, add) : p; });
    post("/api/mytags/used", {group: g.id}).catch(() => {});
    g.uses = (g.uses || 0) + 1;
  }
  redraw("grid");
  if (Q.sel?.group === g.id) redraw("sel");
}

// 組合照原本的寫法加（權重、LoRA 都留著）；addTags 會把權重當成不同的字，所以先篩掉已經有的再接上
function appendRaw(text, parts) {
  const s = String(text || "");
  const trail = /,\s*$/.test(s);
  const base = s.replace(/[\s,]*$/, "");
  return `${base}${base ? (/\n/.test(base) ? ",\n" : ", ") : ""}${parts.join(", ")}${trail ? "," : ""}`;
}

// ── 詳細欄 ──
export function tagDetail(box) {
  const s = Q.sel;
  let kids;
  if (Q.view === "inbox") kids = inboxDetail();
  else if (s?.new === "group" || s?.group != null) kids = groupDetail(s.group != null ? Q.data.groups.find(g => g.id === s.group) : null);
  else if (s?.new === "mine") kids = newTagDetail();
  else if (s?.tag) {
    const it = Q.data.tags.find(x => x.name === s.tag);
    kids = it ? tagEditor(it) : [h("p.dim", {"data-i18n": `tlib.hint.${Q.view}`})];
  } else kids = [h("p.dim", {"data-i18n": `tlib.hint.${Q.view}`})];
  box.replaceChildren(...kids.filter(Boolean));
  translate(box);
}

function catSelect(value, onChange) {
  const grp = GROUP_OF[value] || "";
  return h("select.chip", {"data-i18n-aria": "tlib.cat", on: {change: e => onChange(CATS.find(([g]) => g === e.target.value)?.[1] || "")}},
    h("option", {value: "", text: t("tlib.catNone"), selected: !grp}),
    CATS.map(([g]) => h("option", {value: g, text: t(`chip.cat.${g}`), selected: grp === g})));
}

async function saveTag(data) {
  const r = await post("/api/mytags/tag", data);
  if (r.error) return r;
  await loadTagTab();
  return r;
}

function deleteButton(label, sure, run) {
  return h("button.pill", {type: "button", "data-i18n": label, on: {click: async e => {
    const b = e.currentTarget;
    if (!Q.arm) {
      b.textContent = t(sure);
      b.classList.add("armed");
      Q.arm = setTimeout(() => { Q.arm = null; b.textContent = t(label); b.classList.remove("armed"); }, 4000);
      return;
    }
    clearTimeout(Q.arm);
    Q.arm = null;
    await run();
  }}});
}

function tagEditor(it) {
  const on = tagOn(it);
  const field = (key, value, area = false) => h(area ? "textarea.tlnote" : "input.tlfield", {
    type: area ? null : "text", value, "data-i18n-placeholder": `tlib.${key}Ph`, "data-i18n-aria": `tlib.${key}`,
    on: {change: async e => { await saveTag({name: it.name, [key]: e.target.value}); redraw("grid"); }}});
  return [
    h("div.tltitle", {}, h("span.dot", {"data-c": GROUP_OF[catOf(it)] || null}), h("b", {text: label(it.name)})),
    h("p.dim.small", {text: it.own ? t("tlib.isOwn") : t("tlib.isBase", {n: (it.posts || 0).toLocaleString(lang())})}),
    h("div.vacts", {},
      h("button.pill.primary", {type: "button", on: {click: () => toggleTag(it)}}, t(on ? "tlib.remove" : "tlib.add")),
      h("button.pill", {type: "button", "aria-pressed": String(!!it.fav), on: {click: async () => {
        const r = await saveTag({name: it.name, fav: !it.fav});
        if (r.removed) Q.sel = null;
        redraw("all");
      }}}, t(it.fav ? "tlib.unfav" : "tlib.fav")),
      h("button.pill", {type: "button", "data-i18n": "tlib.findImages", on: {click: () => searchLibrary(label(it.name))}})),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.zh"}), field("zh", it.zh || "")),
    !it.own && it.base_zh ? h("p.dim.small", {text: t("tlib.baseZh", {zh: it.base_zh})}) : null,
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.cat"}),
      catSelect(catOf(it), async cat => { await saveTag({name: it.name, cat}); redraw("all"); })),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.note"}), field("note", it.note || "", true)),
    it.own ? deleteButton("tlib.delete", "tlib.deleteSure", async () => {
      await post("/api/mytags/tag/delete", {name: it.name});
      Q.sel = null;
      await loadTagTab();
      redraw("all");
    }) : null,
  ];
}

// 新增：打字時就比對內建庫（同名的變收藏、別名的指出正名）
function newTagDetail() {
  const name = h("input.tlfield", {type: "text", "data-i18n-placeholder": "tlib.namePh", "data-i18n-aria": "tlib.name",
                                  autocomplete: "off", spellcheck: "false"});
  const zh = h("input.tlfield", {type: "text", "data-i18n-placeholder": "tlib.zhPh", "data-i18n-aria": "tlib.zh"});
  let cat = "";
  const note = h("p.small.tlcheckmsg");
  const go = h("button.pill.primary", {type: "button", "data-i18n": "tlib.addTag", disabled: true});
  let timer = null, last = null;
  const check = async () => {
    const v = name.value.trim();
    if (!v) { note.textContent = ""; go.disabled = true; return; }
    const r = await get(`/api/mytags/check?name=${encodeURIComponent(v)}`).catch(() => null);
    if (!r || name.value.trim() !== v) return;
    last = r;
    go.disabled = r.mine;
    note.replaceChildren();
    if (r.mine) note.textContent = t("tlib.already");
    else if (r.alias) {
      note.append(t("tlib.isAlias", {name: label(r.base)}), " ",
        h("button.pill", {type: "button", text: t("tlib.favInstead", {name: label(r.base)}), on: {click: async () => {
          await saveTag({name: r.base, fav: true});
          Q.view = "fav";
          Q.sel = {tag: r.base};
          redraw("all");
        }}}));
      go.disabled = true;
    } else if (r.base) note.textContent = t("tlib.inBase");
    else note.textContent = t("tlib.newOk");
  };
  name.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(check, 250); });
  name.addEventListener("keydown", e => { if (e.key === "Enter" && !go.disabled) go.click(); });
  go.addEventListener("click", async () => {
    const r = await saveTag({name: name.value, zh: zh.value, cat, fav: last?.base ? true : undefined});
    if (r.error) { note.textContent = r.error === "alias" ? t("tlib.isAlias", {name: label(r.alias_of)}) : r.error; return; }
    if (!r.own) Q.view = "fav";
    Q.sel = {tag: r.name};
    redraw("all");
  });
  setTimeout(() => name.focus(), 0);
  return [
    h("h3.tltitle", {"data-i18n": "tlib.newTag"}),
    h("p.dim.small", {"data-i18n": "tlib.newHint"}),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.name"}), name),
    note,
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.zh"}), zh),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.cat"}), catSelect("", v => { cat = v; })),
    h("div.vacts", {}, go),
  ];
}

function groupDetail(g) {
  const name = h("input.tlfield", {type: "text", value: g?.name || "", "data-i18n-placeholder": "tlib.groupNamePh", "data-i18n-aria": "tlib.groupName"});
  const listId = "tlibFolders";
  const folder = h("input.tlfield", {type: "text", value: g?.folder || "", list: listId, "data-i18n-placeholder": "tlib.folderPh",
                                    "data-i18n-aria": "tlib.folder"});
  const text = h("textarea.tltextarea", {value: g?.text || "", "data-i18n-placeholder": "tlib.textPh", "data-i18n-aria": "tlib.text",
                                        spellcheck: "false"});
  const note = h("textarea.tlnote", {value: g?.note || "", "data-i18n-placeholder": "tlib.notePh", "data-i18n-aria": "tlib.note"});
  const msg = h("p.small.tlcheckmsg");
  const save = async () => {
    const r = await post("/api/mytags/group", {id: g?.id, name: name.value, folder: folder.value, text: text.value, note: note.value,
                                               fav: g?.fav || 0});
    if (r.error) { msg.textContent = r.error; return false; }
    await loadTagTab();
    Q.sel = {group: r.id};
    redraw("all");
    return true;
  };
  return [
    h("h3.tltitle", {text: g ? g.name : t("tlib.newGroup")}),
    g ? null : h("p.dim.small", {"data-i18n": "tlib.groupHint"}),
    g ? h("div.vacts", {},
      h("button.pill.primary", {type: "button", on: {click: () => toggleGroup(g)}}, t(groupOn(g) ? "tlib.remove" : "tlib.add")),
      h("button.pill", {type: "button", "aria-pressed": String(!!g.fav), on: {click: async () => {
        await post("/api/mytags/group", {...g, fav: !g.fav});
        await loadTagTab();
        redraw("all");
      }}}, t(g.fav ? "tlib.unfav" : "tlib.fav"))) : null,
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.groupName"}), name),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.folder"}), folder),
    h("datalist", {id: listId}, folders().map(f => h("option", {value: f}))),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.text"}), text),
    h("button.pill", {type: "button", "data-i18n": "tlib.fromPrompt", "data-i18n-title": "tlib.fromPromptHint",
                      on: {click: () => { text.value = prompt(); }}}),
    h("label.tllbl", {}, h("span", {"data-i18n": "tlib.note"}), note),
    msg,
    h("div.vacts", {},
      h("button.pill.primary", {type: "button", "data-i18n": g ? "tlib.save" : "tlib.addGroup", on: {click: save}}),
      g ? deleteButton("tlib.delete", "tlib.deleteSure", async () => {
        await post("/api/mytags/group/delete", {id: g.id});
        Q.sel = null;
        await loadTagTab();
        redraw("all");
      }) : null),
  ];
}

function inboxDetail() {
  if (Q.showIgnored) return [h("p.dim", {"data-i18n": "tlib.ignoredHint"})];
  const n = Q.inboxSel.size;
  let cat = "";
  const names = [...Q.inboxSel];
  const visible = () => (Q.inbox?.items || []).filter(x => (Q.sentences || x.words < 5)).map(x => x.name);
  const done = async () => { Q.inboxSel.clear(); await loadTagTab(); await loadInbox(); redraw("all"); };
  return [
    h("p.dim.small", {"data-i18n": "tlib.hint.inbox"}),
    Q.inbox ? h("p.small", {text: t("tlib.inboxTotal", {n: Q.inbox.total})}) : null,
    h("div.vacts", {},
      h("button.pill", {type: "button", "data-i18n": "tlib.selectShown", on: {click: () => { for (const x of visible()) Q.inboxSel.add(x); redraw("all"); }}}),
      n ? h("button.pill", {type: "button", "data-i18n": "tlib.selectNone", on: {click: () => { Q.inboxSel.clear(); redraw("all"); }}}) : null),
    h("p.small", {text: t("tlib.selected", {n})}),
    n ? h("label.tllbl", {}, h("span", {"data-i18n": "tlib.catForAll"}), catSelect("", v => { cat = v; })) : null,
    n ? h("div.vacts", {},
      h("button.pill.primary", {type: "button", "data-i18n": "tlib.accept", on: {click: async () => {
        await post("/api/mytags/accept", {names, cat});
        await done();
      }}}),
      h("button.pill", {type: "button", "data-i18n": "tlib.ignore", "data-i18n-title": "tlib.ignoreHint", on: {click: async () => {
        await post("/api/mytags/ignore", {names});
        await done();
      }}})) : null,
  ];
}

async function unignore(name) {
  await post("/api/mytags/ignore", {names: [name], undo: true});
  Q.ignoredList = Q.ignoredList.filter(n => n !== name);
  Q.data.ignored = Math.max(0, (Q.data.ignored || 1) - 1);
  Q.inbox = null;
  if (!Q.ignoredList.length) { Q.showIgnored = false; await loadInbox(); }
  redraw("all");
}

// 牆上方的「使用中」一列：標籤分頁不用（加進提示詞的標籤在牆上直接標出來）
export function tagInUse(box) {
  box.hidden = true;
  box.replaceChildren();
}
