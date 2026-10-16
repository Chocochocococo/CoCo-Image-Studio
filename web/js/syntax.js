// 動態語法：預覽提示詞會展開成哪些組合、編輯萬用字元檔、編輯變數。
// 從提示詞編輯區上方的「語法」打開。萬用字元與變數可以一鍵插進提示詞游標的位置。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {genSettings, insertIntoPrompt} from "./generate.js";
import {lang, t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {closePop} from "./pop.js";
import {newest} from "./inflight.js";

const Y = {el: null, tab: "preview", back: null, wc: [], wcFile: "", vars: [], vfilter: "", editVar: null, delArm: null};
const TABS = ["preview", "wildcards", "vars"];
const loadTicket = newest();          // 萬用字元換檔：只收最後一次點的那個檔的內容

export function openSyntax(tab = "preview") {
  closePop();
  Y.tab = TABS.includes(tab) ? tab : "preview";
  Y.back = document.activeElement;
  Y.el?.remove();
  Y.el = h("div.assets.syntax", {role: "dialog", "aria-modal": "true", "aria-labelledby": "synTitle"});
  Y.el.addEventListener("click", e => { if (e.target === Y.el) closeSyntax(); });
  document.body.append(Y.el);
  document.body.classList.add("viewing");
  draw();
}

export function closeSyntax() {
  if (!Y.el) return;
  Y.el.remove();
  Y.el = null;
  document.body.classList.remove("viewing");
  Y.back?.focus?.();
}

function draw() {
  Y.el.replaceChildren(h("div.apanel.ypanel", {},
    h("div.ahd", {},
      h("h2.title", {id: "synTitle", "data-i18n": "syn.title"}),
      h("div.seg", {role: "tablist"}, TABS.map(k => h("button", {
        type: "button", role: "tab", "aria-selected": String(Y.tab === k), "aria-pressed": String(Y.tab === k),
        "data-i18n": `syn.tab.${k}`, on: {click: () => { Y.tab = k; draw(); }}}))),
      h("span.sp"),
      h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: closeSyntax}}, "✕")),
    h("div.ybody", {id: "synBody"})));
  translate(Y.el);
  ({preview: drawPreview, wildcards: drawWildcards, vars: drawVars})[Y.tab]();
}

// ── 預覽 ──
async function drawPreview() {
  const body = $("synBody"), s = genSettings() || {};
  const lock = !!(s.TPL_LOCK_RANDOM ?? true);
  const out = h("div.ylist", {}, h("p.dim", {"data-i18n": "wall.loading"}));
  const summary = h("p.ysum");
  const run = async () => {
    const prompt = genSettings()?.PROMPT || "";
    const d = await post("/api/expand", {template: prompt, count: genSettings()?.BATCH || 1, seed: "",
                                         lock: !!(genSettings()?.TPL_LOCK_RANDOM ?? true)}).catch(e => ({error: e.message}));
    if (d.error) { summary.textContent = t("syn.err", {msg: d.error}); summary.dataset.kind = "err"; out.replaceChildren(); return; }
    summary.dataset.kind = d.missing?.length ? "warn" : "";
    summary.textContent = [t("syn.count", {n: d.items.length.toLocaleString(lang()), combos: d.combos, per: d.count}),
                           d.missing?.length ? t("syn.missing", {names: d.missing.join("、")}) : ""].filter(Boolean).join(" · ");
    const shown = d.items.slice(0, 300);
    out.replaceChildren(...[...shown.map((it, i) => h("div.yitem", {}, h("b.mono", {text: String(i + 1)}), h("span", {text: it.prompt}))),
      d.items.length > shown.length ? h("p.dim", {text: t("syn.more", {n: d.items.length - shown.length})}) : null].filter(Boolean));
  };
  body.replaceChildren(
    h("div.yhelp", {},
      h("ul", {}, [["[[A|B]]", "syn.h.cart"], ["{A|B}", "syn.h.rand"], ["{3::A|B}", "syn.h.weight"], ["{2$$A|B|C}", "syn.h.pick"],
                   ["__name__", "syn.h.wc"], ["${name}", "syn.h.var"], ["#", "syn.h.comment"]]
        .map(([code, key]) => h("li", {}, h("code", {text: code}), " ", h("span", {"data-i18n": key}))))),
    h("div.ytools", {},
      h("button.switch", {type: "button", role: "switch", "aria-checked": String(lock), "aria-labelledby": "synLockLbl",
                          on: {click: async e => {
                            const v = e.currentTarget.getAttribute("aria-checked") !== "true";
                            e.currentTarget.setAttribute("aria-checked", String(v));
                            genSettings().TPL_LOCK_RANDOM = v;
                            await post("/api/settings", {TPL_LOCK_RANDOM: v}).catch(() => {});
                            run();
                          }}}, h("i")),
      ...(([btn, hint]) => [h("span", {}, h("b", {id: "synLockLbl", "data-i18n": "syn.lock"}), btn), hint])(info("syn.lockHint")),
      h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "syn.reroll", on: {click: run}})),
    summary, out);
  translate(body);
  run();
}

// ── 萬用字元 ──
async function drawWildcards() {
  const body = $("synBody");
  const d = await get("/api/wildcards").catch(() => ({files: [], dir: ""}));
  Y.wc = d.files || [];
  if (!Y.wcFile && Y.wc.length) Y.wcFile = Y.wc[0].file;
  const text = h("textarea.yedit", {spellcheck: "false", "data-i18n-aria": "syn.tab.wildcards"});
  const msg = h("span.msg", {role: "status"});
  const nameOf = f => f.replace(/\.txt$/i, "");
  // 換檔：內容回來才把檔名、內容一起換上去，讀好之前不能儲存。先點 A 又點 B、A 比較晚回來時不收 A 的
  // （以前檔名先換成 B、內容後來被 A 蓋掉，按儲存就把 A 的內容存進 B；讀不到時內容是空的，按儲存會清空檔案）
  const load = async file => {
    const still = loadTicket();
    saveBtn.disabled = true;
    list.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.file === file)));
    const r = await get(`/api/wildcard?file=${encodeURIComponent(file)}`).catch(e => ({error: e.message}));
    if (!still()) return;
    Y.wcFile = file;
    title.textContent = `__${nameOf(file)}__`;
    if (r.error || typeof r.text !== "string") {
      text.value = "";
      msg.textContent = t("syn.wcLoadErr", {msg: r.error || "—"});
      msg.dataset.kind = "err";
      return;
    }
    text.value = r.text;
    saveBtn.disabled = false;
  };
  const title = h("code.ytitle");
  const list = h("div.plist.ywc", {}, Y.wc.map(f => h("button", {type: "button", "data-file": f.file, "aria-pressed": String(f.file === Y.wcFile),
                                                                on: {click: () => load(f.file)}},
    h("span", {text: f.name}), h("span.c.mono", {text: String(f.lines)}))));
  const saveBtn = h("button.pill.primary", {type: "button", "data-i18n": "set.save", disabled: true, on: {click: async () => {
    if (!Y.wcFile || saveBtn.disabled) return;
    const r = await post("/api/wildcard", {file: Y.wcFile, text: text.value}).catch(e => ({error: e.message}));
    msg.textContent = r.error ? r.error : t("set.saved");
    msg.dataset.kind = r.error ? "err" : "ok";
    if (!r.error) {
      const n = text.value.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith("#")).length;
      const b = list.querySelector(`button[data-file="${CSS.escape(Y.wcFile)}"] .c`);
      if (b) b.textContent = String(n);
    }
  }}});
  const newName = h("input.find", {type: "text", "data-i18n-placeholder": "syn.newWc", "data-i18n-aria": "syn.newWc", spellcheck: "false"});
  body.replaceChildren(
    h("div.ysplit", {},
      h("div.yside", {},
        ...(([btn, hint]) => [h("div.yinfo", {}, h("b", {"data-i18n": "syn.tab.wildcards"}), btn), hint])(
          info("syn.wcHint", {dir: d.dir || "wildcards"})),
        list,
        h("div.newset", {}, newName, h("button.pill", {type: "button", "data-i18n": "fb.mkdir", on: {click: async () => {
          const n = newName.value.trim().replace(/\\/g, "/").replace(/\.txt$/i, "");
          if (!n) return;
          // create：已經有同名的檔不會被清空（後端排他建立，以前直接覆寫成空的）
          const r = await post("/api/wildcard", {file: `${n}.txt`, text: "", create: true}).catch(e => ({error: e.message}));
          if (r.error) { msg.textContent = r.exists ? t("syn.wcExists", {name: n}) : r.error; msg.dataset.kind = "err"; return; }
          Y.wcFile = r.file;
          drawWildcards();
        }}}))),
      h("div.ymain", {},
        h("div.yhd", {}, title, h("span.sp"),
          h("button.pill", {type: "button", "data-i18n": "syn.insert", on: {click: () => {
            if (Y.wcFile) { insertIntoPrompt(`__${nameOf(Y.wcFile)}__`); msg.textContent = t("syn.inserted"); msg.dataset.kind = "ok"; }
          }}})),
        text,
        h("div.yhd", {}, msg, h("span.sp"), saveBtn))));
  translate(body);
  if (Y.wcFile) load(Y.wcFile);
  else text.placeholder = t("syn.noWc");
}

// ── 變數 ──
async function drawVars() {
  const body = $("synBody");
  const d = await get("/api/vars").catch(() => ({vars: []}));
  Y.vars = d.vars || [];
  const nameIn = h("input.find", {type: "text", "data-i18n-placeholder": "syn.varName", "data-i18n-aria": "syn.varName", spellcheck: "false"});
  const valueIn = h("textarea.yedit.small", {rows: "3", spellcheck: "false", "data-i18n-placeholder": "syn.varValue", "data-i18n-aria": "syn.varValue"});
  const msg = h("span.msg", {role: "status"});
  const filter = h("input.find", {type: "search", value: Y.vfilter, "data-i18n-placeholder": "lib.find", "data-i18n-aria": "lib.find",
                                  on: {input: e => { Y.vfilter = e.target.value; drawList(); }}});
  const list = h("div.yvars");
  const act = async payload => {
    const r = await post("/api/vars", payload).catch(e => ({error: e.message}));
    if (r.error) { msg.textContent = r.error; msg.dataset.kind = "err"; return false; }
    Y.vars = r.vars;
    drawList();
    return true;
  };
  const edit = v => {
    Y.editVar = v?.name || null;
    nameIn.value = v?.name || "";
    valueIn.value = v?.value || "";
    saveBtn.textContent = t(Y.editVar ? "syn.update" : "syn.add");
    nameIn.focus();
  };
  const saveBtn = h("button.pill.primary", {type: "button", on: {click: async () => {
    const ok = await act(Y.editVar ? {action: "update", old_name: Y.editVar, name: nameIn.value, value: valueIn.value}
                                   : {action: "add", name: nameIn.value, value: valueIn.value});
    if (ok) { msg.textContent = t("set.saved"); msg.dataset.kind = "ok"; edit(null); }
  }}}, t("syn.add"));
  const drawList = () => {
    const q = Y.vfilter.trim().toLowerCase();
    const shown = Y.vars.filter(v => !q || v.name.toLowerCase().includes(q) || v.value.toLowerCase().includes(q));
    list.replaceChildren(...(shown.length ? shown.map(v => {
      let armed = null;
      return h("div.yvar", {},
        h("button.yvmain", {type: "button", title: t("syn.editHint"), on: {click: () => edit(v)}},
          h("code", {text: `\${${v.name}}`}), h("span", {text: v.value})),
        h("button.mini", {type: "button", "data-i18n": "syn.insert", on: {click: () => {
          insertIntoPrompt(`\${${v.name}}`); msg.textContent = t("syn.inserted"); msg.dataset.kind = "ok";
        }}}),
        h("button.mini", {type: "button", "data-i18n": "rcp.delete", on: {click: async e => {
          const b = e.currentTarget;
          if (!armed) {
            b.textContent = t("rcp.deleteSure"); b.classList.add("armed");
            armed = setTimeout(() => { armed = null; b.textContent = t("rcp.delete"); b.classList.remove("armed"); }, 4000);
            return;
          }
          clearTimeout(armed);
          await act({action: "delete", name: v.name});
          if (Y.editVar === v.name) edit(null);
        }}}));
    }) : [h("p.dim", {text: t(Y.vars.length ? "lib.noMatch" : "syn.noVars")})]));
    translate(list);
  };
  body.replaceChildren(
    ...(([btn, hint]) => [h("div.yinfo", {}, h("b", {"data-i18n": "syn.tab.vars"}), btn), hint])(
      info("syn.varHint", {ex: `\${${t("syn.varName").toLowerCase()}}`})),
    h("div.yform", {}, nameIn, valueIn,
      h("div.yhd", {}, msg, h("span.sp"),
        h("button.pill", {type: "button", "data-i18n": "syn.clear", on: {click: () => edit(null)}}), saveBtn)),
    filter, list);
  translate(body);
  drawList();
}

document.addEventListener("keydown", e => {
  if (Y.el && e.key === "Escape" && !e.defaultPrevented) { e.preventDefault(); closeSyntax(); }
});
