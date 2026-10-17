// 生成頁的控制台：資源格、提示詞、參數、生成鈕與進度。
//
// 參數存在 settings.json（POST /api/settings，逐欄合併）。模型、LoRA、VAE 在這裡只顯示，換的功能在資源庫做。
//
// 多視窗（sync.js）：每個視窗都有自己的一份設定（G.s）。存檔後廣播改了哪些欄位，別的視窗照著改；
// 伺服器上的設定被改了（換模型等）就廣播 reload。視窗失去焦點時馬上存，切到另一個螢幕時那邊拿到的是最新的。

import {tokenLabel, wrapSyntax} from "./prompttext.js";
import {get, post} from "./api.js";
import {t, translate} from "./i18n.js";
import {openControlNet, refThumb} from "./cnpanel.js";
import {$, DIE, REUSE, h, icon, stem} from "./dom.js";
import {closePop, openPop, openedPop} from "./pop.js";
import {on, send} from "./sync.js";
import {putBack} from "./inflight.js";
import {info} from "./hint.js";
import {ChipBox, attachAC} from "./chips.js";
import {INSTALL, litePopBody} from "./install.js";
import {ARCH_LABEL, AUTO_ON_OFF, COND_MODES, DYN_SYNTAX, SAMPLER_GROUPS, SCHEDULES, SIZES,
        VRAM_MODES, loraTags, FLOW_ARCHS, shiftKey} from "./options.js";

const G = {
  s: null,          // /api/settings
  err: "",          // 設定讀不到時的原因
  tab: "PROMPT",    // 編輯區目前的分頁：PROMPT / NEGATIVE
  caret: null,      // 提示詞游標最後停的位置（插入萬用字元、變數用）
  chips: null,      // 標籤模式的 ChipBox（每次重畫編輯區重建）
  preset: null,     // 目前模型的品質標籤與預設負面（/api/presets/resolve）
  logTab: "table",  // 紀錄小視窗目前的分頁：table（進度表）／log（訊息）
  state: {},        // /api/gen/state
  last: undefined,  // 看過的最新一張；undefined＝還沒輪詢過，第一次不算「新圖」
  tok: null,        // 最近一次 token 計數
  dirty: {},        // 還沒存的欄位
  busy: false,      // 送出中，避免連按
  hooks: {},
};

// 後端進度表的狀態代碼（server/worker_client.py），這幾種代表那張已經結束
const FINISHED = new Set(["done", "failed", "aborted", "cancelled", "skipped"]);

// ── 設定的讀寫 ───────────────────────────────────
let saveTimer = null;
const RETRY_MS = 5000;           // 存不進去（連不上）時過多久再試

function set(key, value) {
  G.s[key] = value;
  G.dirty[key] = value;
  // settings.json 還有舊介面留下的 TEMPLATE 欄位，PROMPT 空白時會拿它來填，兩個要一致
  if (key === "PROMPT") G.dirty.TEMPLATE = value;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 600);
  G.hooks.onChange?.();          // Recipe 的「已修改」
}

async function flush(keepalive = false) {
  clearTimeout(saveTimer);
  const data = G.dirty;
  if (!Object.keys(data).length) return;
  G.dirty = {};
  try {
    const d = await post("/api/settings", data, {keepalive});
    if (d.error) say(t("gen.err.save", {msg: d.error}), "err");
    else send("settings", {patch: data});
  } catch (e) {
    // 連不上：送出去的放回待存、過一會兒再試（切視窗、關分頁時也會再送）；期間又改的以新的為準。
    // 以前直接丟掉，恢復連線後也不會再送，重開只剩舊的提示詞（Codex 第三輪審查）
    putBack(G.dirty, data, G.s, {TEMPLATE: "PROMPT"});
    if (!keepalive) {
      say(t("gen.err.save", {msg: e.message || e}), "err");
      clearTimeout(saveTimer);
      saveTimer = setTimeout(flush, RETRY_MS);
    }
  }
}
// 關分頁時把還沒存的送出去；切到別的視窗（另一個螢幕）時也馬上存
addEventListener("pagehide", () => flush(true));
addEventListener("blur", () => flush());

// ── 別的視窗改的 ──
// 提示詞只換文字（不整個重畫，游標、捲動位置都還在）；其他欄位重畫控制台
const TEXT_KEYS = ["PROMPT", "NEGATIVE", "TEMPLATE"];
function applyRemote(patch) {
  if (!G.s || !patch) return;
  const keys = Object.keys(patch).filter(k => JSON.stringify(G.s[k]) !== JSON.stringify(patch[k]));
  if (!keys.length) return;
  for (const k of keys) {
    G.s[k] = patch[k];
    delete G.dirty[k];                    // 同一欄以後改的為準
  }
  if (keys.every(k => TEXT_KEYS.includes(k))) {
    for (const k of ["PROMPT", "NEGATIVE"]) if (keys.includes(k)) showField(k);
  } else {
    redraw();
  }
  G.hooks.onChange?.();
  G.hooks.onRemote?.("settings");
}
on("settings", ({patch}) => applyRemote(patch));
on("reload", async () => { await reloadSettings(true); G.hooks.onRemote?.("reload"); });
on("tab", ({key}) => { if (key === "PROMPT" || key === "NEGATIVE") G.lastTab = key; });

// 資源類的欄位只在這裡顯示、不在這裡改；外面（狀態列輪詢）拿到新設定時更新
const READONLY = ["MODEL", "MODEL_ARCH", "LORAS", "VAE_OVERRIDE", "TE_OVERRIDE"];
// 生成台用不到、只是跟著更新讓別處讀的（圖庫的「直接刪除圖片」、看大圖的兩個 LoRA 選項），改了不用重畫生成台
export const PASSIVE = ["HARD_DELETE", "SEND_KEEP_LORAS", "COPY_WITH_LORAS"];

export function syncSettings(s) {
  if (!G.s || !s || s.error) return;
  for (const k of PASSIVE) G.s[k] = s[k];
  const archChanged = s.MODEL_ARCH !== G.s.MODEL_ARCH;
  // 換了架構：伺服器換成那個架構的步數、CFG、取樣器、排程器（server/settings_api.arch_switch），生成台跟著換
  if (archChanged) for (const k of ["STEPS", "CFG", "SAMPLER", "SCHEDULE"]) if (s[k] != null) G.s[k] = s[k];
  if (s.MODEL !== G.s.MODEL) refreshPreset();
  const changed = READONLY.some(k => JSON.stringify(s[k]) !== JSON.stringify(G.s[k]));
  for (const k of READONLY) G.s[k] = s[k];
  if (archChanged) redraw();
  else if (changed) drawSlots();
  if (changed) G.hooks.onChange?.();
}

document.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
});

// ── 訊息 ───────────────────────────────────────
let sayTimer = null;
function say(text, kind = "") {
  const el = $("genMsg");
  if (!el) return;
  el.textContent = text;
  el.dataset.kind = kind;
  clearTimeout(sayTimer);
  if (kind !== "err") sayTimer = setTimeout(() => { el.textContent = ""; }, 5000);
}

// ⧉ 另開的視窗按了「送到生成區」（viewer.js 的 sendToGen）：參數已經存好、也同步過了，這裡只顯示那段訊息。
// 生成台正開著就馬上顯示；還沒切過來就等它畫出來（跟 applyParams 的 notice 同一條路）
addEventListener("coco:notice", e => {
  G.notice = e.detail || null;
  if (G.notice && $("genMsg")) { say(G.notice.text, G.notice.kind); G.notice = null; }
});

// ── 資源格 ─────────────────────────────────────
function arch() { return G.s?.MODEL_ARCH || ""; }

function cover(kind, path, fallback, name = "") {
  if (!path && !name) return fallback;
  const q = path ? `path=${encodeURIComponent(path)}` : `name=${encodeURIComponent(name)}`;
  const img = h("img", {src: `/api/res/cover?kind=${kind}&${q}&thumb=1`, alt: "",
                        loading: "lazy", decoding: "async"});
  img.addEventListener("error", () => img.replaceWith(fallback), {once: true});
  return img;
}

// 點資源格打開資源庫的那一頁（main.js 經 hooks.openAssets 接上 assets.js）
function slot(kind, keyLabel, value, pic, extra) {
  return h("button.slot", {type: "button", "aria-haspopup": "dialog", "data-bad": extra?.bad ? "" : null,
                           ...(extra?.bad ? {title: extra.bad} : {"data-i18n-title": "gen.slot.open"}),
                           on: {click: () => G.hooks.openAssets?.(kind)}},
    h("div.pic", {}, pic, extra?.badge ? h("span.badge", {text: extra.badge}) : null,
      extra?.bad ? h("span.badbadge", {text: t("gen.slot.bad")}) : null),
    h("span.k", keyLabel),
    h("span.v", {text: value, title: value}),
    extra?.strength != null
      ? h("span.str", {}, h("i", {style: `width:${Math.min(100, Math.abs(extra.strength) / 2 * 100)}%`}))
      : null);
}

function glyph(text) { return h("span.glyph", {text}); }

// LoRA 跟主模型的架構合不合（資源庫算好的 ok，/api/assets）：不合的照樣能生，但那個 LoRA 會被跳過——
// 以前只寫在生成紀錄裡，很容易沒注意到（使用者選的：資源格與標籤標出來，不擋生成）。架構換了才重問
const LC = {arch: null, ok: {}};
async function loraCompat() {
  const a = arch();
  if (LC.arch === a) return;
  LC.arch = a;
  const d = await get("/api/assets?kind=lora").catch(() => null);
  if (!d?.items || LC.arch !== a) return;
  LC.ok = Object.fromEntries(d.items.map(it => [it.name.toLowerCase(), it.ok]));
  drawSlots();
  if (chipMode() && G.tab === "PROMPT") G.chips?.draw();
}
const loraBad = name => LC.ok[String(name || "").toLowerCase()] === false;

function slots() {
  const s = G.s, a = arch();
  const archName = ARCH_LABEL[a] || a || "?";
  // LoRA 以提示詞裡的 <lora:…> 為準（舊介面也是），卡片選的 LORAS 也算
  const tags = [...loraTags(s.PROMPT), ...(s.LORAS || []).filter(l => l.path)
    .map(l => ({name: stem(l.path), path: l.path, weight: Number(l.weight) || 1}))];
  const first = tags[0];
  loraCompat();
  const bad = tags.filter(x => loraBad(x.name)).map(x => x.name);
  const part = key => {
    const v = (s[key] || {})[a];
    return v ? {name: stem(v), path: v} : null;
  };
  const vae = part("VAE_OVERRIDE"), te = part("TE_OVERRIDE");
  // 拆分式模型（flow 架構）沒有內建的 VAE、文字編碼器：沒選時生成會自動找資源庫裡相符的（server/assets.fill_parts）
  const none = t(FLOW_ARCHS.includes(a) ? "gen.slot.autoPart" : "gen.slot.builtin");
  return h("div.g5.slots", {id: "slots"},
    slot("model", {"data-i18n": "gen.slot.model"}, stem(s.MODEL) || "—", cover("model", s.MODEL, glyph(archName))),
    slot("lora", first ? {text: t("gen.slot.loraW", {w: first.weight.toFixed(2)})} : {"data-i18n": "gen.slot.lora"},
         first ? first.name : t("gen.slot.none"),
         // 第一個 LoRA 的封面（好幾個時右上角標 +N）；沒有封面就顯示 L
         first ? cover("lora", first.path || "", glyph("L"), first.path ? "" : first.name) : glyph("—"),
         {badge: tags.length > 1 ? t("gen.slot.more", {n: tags.length - 1}) : null,
          strength: first ? first.weight : null, bad: bad.length ? t("gen.slot.loraBad", {names: bad.join("、")}) : ""}),
    slot("vae", {text: "VAE"}, vae ? vae.name : none,
         vae ? cover("vae", vae.path, glyph("VAE")) : glyph("—")),
    slot("te", {"data-i18n": "gen.slot.te"}, te ? te.name : none,
         te ? cover("te", te.path, glyph("TE")) : glyph("—")),
    cnSlot());
}

// 參考圖（ControlNet，cnpanel.js）：第一張開著的參考圖、種類；好幾張時右上角標 +N
function cnSlot() {
  const all = (G.s.CONTROLNETS || []).filter(u => u.src), on = all.filter(u => !u.off), first = on[0];
  return h("button.slot", {type: "button", "data-i18n-title": "cn.slotHint", "aria-haspopup": "dialog", "data-off": all.length && !on.length ? "" : null,
                           on: {click: () => openControlNet()}},
    h("div.pic", {}, first ? h("img", {src: refThumb(first.src), alt: "", loading: "lazy"}) : glyph("CN"),
      on.length > 1 ? h("span.badge", {text: t("gen.slot.more", {n: on.length - 1})}) : null),
    h("span.k", {"data-i18n": "cn.slot"}),
    h("span.v", {text: first ? t(`cn.type.${first.type || "pose"}`) : t(all.length ? "cn.allOff" : "gen.slot.none")}));
}

function drawSlots() {
  const old = $("slots");
  if (!old) return;
  const neu = slots();
  old.replaceWith(neu);
  translate(neu);
}

// ── 提示詞編輯區 ─────────────────────────────────
const chipMode = () => !!G.hooks.ui?.chips;

function editor() {
  const tab = (key, label) => h("button.tb", {
    type: "button", role: "tab", id: `tab_${key}`, "aria-selected": String(G.tab === key),
    "aria-controls": "genText", "data-i18n": label, on: {click: () => switchTab(key)}});
  const ta = h("textarea.edtext", {id: "genText", spellcheck: "false", role: "tabpanel", hidden: chipMode(),
                                   "aria-labelledby": `tab_${G.tab}`, value: G.s[G.tab] || "",
                                   on: {input: onText, keyup: keepCaret, mouseup: keepCaret, blur: keepCaret}});
  attachAC(ta, {mode: "text"});
  G.chips = new ChipBox({read: () => G.s[G.tab] || "", write: text => { ta.value = text; edited(text); },
                         label: t("gen.chips"), loraBad,
                         readOff: () => G.s[`${G.tab}_OFF`] || [], writeOff: list => set(`${G.tab}_OFF`, list)});
  G.chips.el.id = "genChips";
  G.chips.el.hidden = !chipMode();
  if (chipMode()) G.chips.refresh();
  return h("div.ed", {id: "genEd"},
    h("div.edtop", {role: "tablist"},
      tab("PROMPT", "gen.prompt"), tab("NEGATIVE", "gen.negative"),
      h("span.sp"),
      h("button.mini.tog", {type: "button", id: "presetBtn", "aria-pressed": String(presetOn()), on: {click: togglePreset}},
        t(presetOn() ? "gen.preset" : "gen.presetOff")),
      h("button.mini.tog", {type: "button", id: "chipBtn", "aria-pressed": String(chipMode()), "data-i18n": "gen.chips",
                            "data-i18n-title": "gen.chipsHint", on: {click: toggleChips}})),
    ta, G.chips.el,
    // 有動態語法時的提醒：以前擠在頂端列被切成「動態語法…」（使用者回報），放到編輯區下面自己一行
    h("p.dyn", {id: "dynHint", "data-i18n": "gen.dynamic", hidden: !DYN_SYNTAX.test(G.s.PROMPT || "")}),
    // 底下一排：左邊 token 數，右邊動態語法的三顆與放大（頂端列放不下會變兩行，使用者要的排法）
    h("div.edfoot", {},
      h("span.tokn", {id: "tokLine", "aria-live": "polite"}),
      h("span.sp"),
      // 選起來的字包成 [[a|b]]／{a|b}（純文字模式；使用者要求的動態語法輸入助手）
      h("button.mini.wrapbtn", {type: "button", "data-i18n-title": "gen.wrapCart", "data-i18n-aria": "gen.wrapCart",
                                on: {click: () => wrapSel("cart")}}, "[[ ]]"),
      h("button.mini.wrapbtn", {type: "button", "data-i18n-title": "gen.wrapRand", "data-i18n-aria": "gen.wrapRand",
                                on: {click: () => wrapSel("rand")}}, "{ }"),
      h("button.mini", {type: "button", id: "synBtn", "data-i18n": "gen.syntax", "data-i18n-title": "gen.syntaxHint",
                        on: {click: () => G.hooks.openSyntax?.()}}),
      h("button.mini.edbigbtn", {type: "button", id: "bigBtn", "aria-pressed": "false", "data-i18n-title": "gen.bigEdit",
                                 "data-i18n-aria": "gen.bigEdit", on: {click: () => (BIG.el ? closeBig() : openBig())}}, "⤢")));
}

function wrapSel(kind) {
  const ta = $("genText");
  if (!ta || ta.hidden) return;
  const a = ta.selectionStart, b = ta.selectionEnd;
  const out = wrapSyntax(ta.value.slice(a, b), kind);
  ta.value = ta.value.slice(0, a) + out + ta.value.slice(b);
  const caret = a === b ? a + (kind === "cart" ? 2 : 1) : a + out.length;      // 沒選字：游標放進括號裡
  ta.focus();
  ta.setSelectionRange(caret, caret);
  ta.dispatchEvent(new Event("input", {bubbles: true}));
}

// ── 放大編輯（使用者筆記 #13）──
// 長提示詞在側欄裡太擠：把整個編輯區搬進一個大視窗，關掉再搬回原位。
// 搬的是同一個元素，分頁、標籤模式、自動完成、token 數都照常運作
const BIG = {el: null, home: null, back: null};

function openBig() {
  const ed = $("genEd");
  if (!ed || BIG.el) return;
  closePop();
  BIG.back = document.activeElement;
  BIG.home = document.createComment("genEd");
  ed.replaceWith(BIG.home);
  BIG.el = h("div.edbig", {role: "dialog", "aria-modal": "true", "aria-label": t("gen.bigEdit")}, ed);
  BIG.el.addEventListener("click", e => { if (e.target === BIG.el) closeBig(); });
  document.body.append(BIG.el);
  document.body.classList.add("viewing");
  ed.classList.add("big");
  syncBigBtn();
  (chipMode() ? G.chips.input : $("genText"))?.focus();
}

function closeBig() {
  if (!BIG.el) return;
  const ed = $("genEd");
  if (ed) {
    ed.classList.remove("big");
    if (BIG.home?.isConnected) BIG.home.replaceWith(ed);
  }
  BIG.el.remove();
  BIG.el = BIG.home = null;
  document.body.classList.remove("viewing");
  syncBigBtn();
  (BIG.back?.isConnected ? BIG.back : $("genText"))?.focus?.();
}

function syncBigBtn() {
  const b = $("bigBtn");
  if (!b) return;
  b.setAttribute("aria-pressed", String(!!BIG.el));
  b.title = t(BIG.el ? "gen.bigEditClose" : "gen.bigEdit");
  b.setAttribute("aria-label", b.title);
}

document.addEventListener("keydown", e => {
  if (BIG.el && e.key === "Escape" && !e.defaultPrevented) { e.preventDefault(); closeBig(); }
});

// ── 品質標籤與預設負面（server/presets.py） ──
// 生成時自動加在提示詞後面（或前面），提示詞框裡看不到；開關存在設定裡，關掉會記住
const presetOn = () => G.s?.QUALITY_PRESET !== false;

export async function refreshPreset() {
  try { G.preset = await get("/api/presets/resolve"); } catch { G.preset = null; }
  drawPresetBtn();
  tokQueue();
}

function drawPresetBtn() {
  const b = $("presetBtn");
  if (!b) return;
  const on = presetOn(), p = G.preset;
  b.setAttribute("aria-pressed", String(on));
  b.textContent = t(on ? "gen.preset" : "gen.presetOff");
  // 滑過看得到實際會加什麼；這個模型沒設就說去哪裡設
  const where = t(p?.position === "before" ? "gen.presetBefore" : "gen.presetAfter");
  b.title = !p || p.source === "none" ? t("gen.presetNone")
    : [t(on ? "gen.presetOnHint" : "gen.presetOffHint", {where}),
       p.quality ? t("gen.presetQ", {tags: p.quality, where}) : "",
       p.negative ? t("gen.presetN", {tags: p.negative}) : "",
       t(p.source === "model" ? "gen.presetFromModel" : "gen.presetFromArch")].filter(Boolean).join("\n");
}

function togglePreset() {
  set("QUALITY_PRESET", !presetOn());
  drawPresetBtn();
  tokQueue();
}

// 標籤模式 ↔ 純文字：只是換一種畫法，文字一直是同一份
function toggleChips() {
  const ui = G.hooks.ui;
  if (!ui) return;
  ui.chips = !ui.chips;
  $("chipBtn").setAttribute("aria-pressed", String(ui.chips));
  $("genText").hidden = ui.chips;
  G.chips.el.hidden = !ui.chips;
  if (ui.chips) { G.chips.refresh(); G.chips.input.focus(); }
  else $("genText").focus();
  post("/api/ui", {chips: ui.chips}).catch(() => { /* 只影響下次開啟 */ });
}

function switchTab(key) {
  G.tab = G.lastTab = key;
  send("tab", {key});
  for (const k of ["PROMPT", "NEGATIVE"]) $(`tab_${k}`)?.setAttribute("aria-selected", String(k === key));
  const ta = $("genText");
  ta.value = G.s[key] || "";
  ta.setAttribute("aria-labelledby", `tab_${key}`);
  if (chipMode()) { G.chips.refresh(); G.chips.input.focus(); }
  else ta.focus();
  drawTokens();
}

const keepCaret = e => { if (G.tab === "PROMPT") G.caret = e.target.selectionStart; };

function onText(e) {
  keepCaret(e);
  edited(e.target.value);
}

// 目前分頁的文字改了（打字或標籤模式）：存檔、更新動態語法提示、資源格、token 數
function edited(text) {
  set(G.tab, text);
  if (G.tab === "PROMPT") {
    $("dynHint").hidden = !DYN_SYNTAX.test(text);
    drawSlots();                     // <lora:…> 可能改了
  }
  tokQueue();
}

// token 計數一定要問 worker：SDXL 兩顆 CLIP、Anima 走 Qwen，上限跟詞彙表都不同。
// worker 是串行的，同時只送一個，打字很快時中間的請求併掉、只補跑最後一次。
let tokTimer = null, tokBusy = false, tokAgain = false;
function tokQueue() { clearTimeout(tokTimer); tokTimer = setTimeout(tokRun, 400); }

async function tokRun() {
  if (tokBusy) { tokAgain = true; return; }
  tokBusy = true;
  try {
    G.tok = await post("/api/gen/tokens", {prompt: G.s.PROMPT || "", negative: G.s.NEGATIVE || "", preset: presetOn()});
    drawTokens();
  } catch { /* 計數失敗不擋任何事 */ }
  finally {
    tokBusy = false;
    if (tokAgain) { tokAgain = false; tokRun(); }
  }
}

// token 數只顯示「54/77」：native 超過上限變色（會直接截斷），a1111 的上限跟著段數長（每 75 個一段、不截斷、不變色）。
// 說明（沒超過、截掉幾個、幾段、估計值、數的是模板）放在滑鼠停上去的提示（使用者要的：不要一行警告文字）
function drawTokens() {
  const box = $("tokLine");
  if (!box) return;
  box.textContent = "";
  box.removeAttribute("data-kind");
  box.title = "";
  const d = G.tok;
  if (!d) return;
  if (!d.ready) {                    // 工作行程沒在跑、這種模型不用 CLIP（Anima 這類沒有 77 的上限）
    box.textContent = "—";
    box.dataset.kind = "dim";
    box.title = d.why ? t("gen.tok.why", {why: d.why}) : "";
    return;
  }
  // 欄位名是後端的「正面」「正面（第二組）」「負面」…；SDXL 有兩組 tokenizer，取比較接近上限的那組
  const side = G.tab === "PROMPT" ? "正面" : "負面";
  const fs = (d.fields || []).filter(f => f.name.startsWith(side));
  if (!fs.length) return;
  const f = fs.reduce((a, b) => (b.n - b.limit > a.n - a.limit ? b : a));
  // native 超過上限會直接截斷而且不會有任何訊息，所以要在打字的當下就看得到；a1111 分段、不截斷
  const L = tokenLabel(f, d.mode);
  if (L.over) box.dataset.kind = "err";
  box.textContent = `${d.estimated ? "≈" : ""}${L.n}/${L.cap}`;
  box.title = d.mode === "native" ? t(L.over ? "gen.tok.cut" : "gen.tok.fits", {n: L.over}) : "";
}

// ── 參數格 ─────────────────────────────────────
const fmt = v => (v == null || v === "" ? "" : String(v));

function toNum(raw, {min, max, step, old}) {
  let v = Number(raw);
  if (raw === "" || !Number.isFinite(v)) return old;
  if (min != null) v = Math.max(min, v);
  if (max != null) v = Math.min(max, v);
  if (step >= 1) v = Math.round(v / step) * step;
  else v = Math.round(v * 100) / 100;
  return v;
}

function numInput(key, o) {
  // floor：這個架構實際的下限（SDXL 的 CLIP skip 是 2），存的值比它小時顯示實際會用的
  const shown = o.floor && Number(G.s[key]) < o.floor ? o.floor : G.s[key];
  return h("input.v", {
    id: `f_${key}`, type: "number", inputmode: o.step >= 1 ? "numeric" : "decimal",
    min: o.min, max: o.max, step: o.step, value: fmt(shown), "data-i18n-aria": o.aria,
    on: {change: e => {
      const v = toNum(e.target.value, {...o, old: G.s[key]});
      e.target.value = fmt(v);
      set(key, v);
      if (key === "WIDTH" || key === "HEIGHT") drawAspect();
    }}});
}

function numCell(key, label, o, cls = "") {
  return h(`label.cell${cls}`, {for: `f_${key}`, "data-i18n-title": o.hint || null},
    h("span.k", {"data-i18n": label}), numInput(key, o));
}

function selectCell(key, label, cls, options) {
  const sel = h("select.v", {id: `f_${key}`, on: {change: e => {
    set(key, e.target.value);
    if (key === "SAMPLER" || key === "SCHEDULE") drawSlots();
  }}}, options);
  sel.value = G.s[key] ?? "";
  return h(`label.cell${cls}`, {for: `f_${key}`}, h("span.k", {"data-i18n": label}), sel);
}

function samplerOptions() {
  const a = arch(), cur = G.s.SAMPLER;
  const all = SAMPLER_GROUPS.flatMap(([, names]) => names);
  return [
    // 舊設定或別名（例如 "Euler a kdiff"）不在表上也要顯示得出來
    cur && !all.includes(cur) ? h("option", {value: cur, text: cur}) : null,
    ...SAMPLER_GROUPS.map(([g, names]) => {
      const flowOnly = g === "anima" || g === "animaLab";       // flow 架構（Anima、Z-Image、Krea 2、Qwen Image 2.1）才能用
      return h("optgroup", {label: t(`gen.sg.${g}`)},
        names.map(n => h("option", {value: n, text: n, disabled: flowOnly && a && !FLOW_ARCHS.includes(a) && n !== cur})));
    })];
}

// 尺寸格左邊的小長方形，比例跟著寬高變
function drawAspect() {
  const i = $("aspectIcon");
  if (!i) return;
  const w = G.s.WIDTH || 1, ht = G.s.HEIGHT || 1, k = 16 / Math.max(w, ht);
  i.style.width = `${Math.max(6, Math.round(w * k))}px`;
  i.style.height = `${Math.max(6, Math.round(ht * k))}px`;
}

function sizeCell() {
  const btn = h("button.ar", {type: "button", id: "btnSize", "aria-haspopup": "dialog", "aria-expanded": "false",
                              "data-i18n-aria": "gen.size.pick", "data-i18n-title": "gen.size.pick",
                              on: {click: e => openPop("popSize", e.currentTarget, sizePop)}},
    h("i", {id: "aspectIcon"}));
  return h("div.cell.w2.size", {},
    h("span.k", {"data-i18n": "gen.size"}),
    h("div.v", {}, btn,
      numInput("WIDTH", {...SIZE_LIMIT, aria: "gen.width"}),
      h("span.x", {text: "×", "aria-hidden": "true"}),
      numInput("HEIGHT", {...SIZE_LIMIT, aria: "gen.height"})));
}

// 寬高的範圍與倍數（跟伺服器 server/refine.snap 一樣）：每種模型都要 16 的倍數，太小太大都畫不好
const SIZE_LIMIT = {min: 256, max: 4096, step: 16};
const snapSize = v => toNum(v, {...SIZE_LIMIT, old: 1024});

// 使用者自己存的常用尺寸（介面偏好 sizes：[[寬, 高], …]，照存的方向顯示）
const mySizes = () => (Array.isArray(G.hooks.ui?.sizes) ? G.hooks.ui.sizes : [])
  .filter(x => Array.isArray(x) && x.length === 2).map(([w, ht]) => [snapSize(w), snapSize(ht)]);
function saveSizes(list) {
  const ui = G.hooks.ui;
  if (ui) ui.sizes = list;
  post("/api/ui", {sizes: list}).catch(() => { /* 只影響下次開啟 */ });
}
const gcd = (a, b) => (b ? gcd(b, a % b) : a);
// 比例的名字：約得乾淨的直接寫（2:3），約不乾淨的找最接近的常見比例加 ≈
function ratioName(w, ht) {
  const g = gcd(w, ht);
  if (w / g <= 21 && ht / g <= 21) return `${w / g}:${ht / g}`;
  const common = [[1, 1], [2, 3], [3, 4], [4, 5], [5, 7], [7, 9], [9, 16], [9, 21]].flatMap(([a, b]) => [[a, b], [b, a]]);
  const [a, b] = common.reduce((best, c) => (Math.abs(c[0] / c[1] - w / ht) < Math.abs(best[0] / best[1] - w / ht) ? c : best));
  return `≈${a}:${b}`;
}

function setSize(w, ht) {
  set("WIDTH", w); set("HEIGHT", ht);
  $("f_WIDTH").value = w; $("f_HEIGHT").value = ht;
  drawAspect();
}

function sizePop() {
  const land = (G.s.WIDTH || 0) > (G.s.HEIGHT || 0);
  const card = (ratio, w, ht, del) => {
    const k = 28 / Math.max(w, ht);
    const on = w === G.s.WIDTH && ht === G.s.HEIGHT;
    return h("div.szwrap", {},
      h("button.sz", {type: "button", "aria-pressed": String(on),
                      on: {click: () => { setSize(w, ht); closePop()?.focus(); }}},
        h("span.shape", {}, h("i", {style: `width:${Math.round(w * k)}px;height:${Math.round(ht * k)}px`})),
        h("span.r", {text: ratio}),
        h("span.d.mono", {text: `${w}×${ht}`})),
      del ? h("button.szdel", {type: "button", "data-i18n-aria": "gen.size.del", "data-i18n-title": "gen.size.del", on: {click: del}}, "✕") : null);
  };
  // 內建的只列直的，現在是橫的就整組轉成橫的
  const builtin = SIZES.map(([ratio, a, b]) => {
    const [w, ht] = land && a !== b ? [b, a] : [a, b];
    return card(land && a !== b ? ratio.replace(/(\d+):(\d+)/, "$2:$1") : ratio, w, ht);
  });
  const mine = mySizes();
  const cur = [snapSize(G.s.WIDTH), snapSize(G.s.HEIGHT)];
  const saved = mine.some(([w, ht]) => w === cur[0] && ht === cur[1]) || SIZES.some(([, a, b]) => (a === cur[0] && b === cur[1]) || (a === cur[1] && b === cur[0]));
  const redraw = () => { closePop(); openPop("popSize", $("btnSize"), sizePop); };
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "gen.size.pick"}),
      h("button.pill", {type: "button", "data-i18n": "gen.size.swap",
                        on: {click: () => { setSize(G.s.HEIGHT, G.s.WIDTH); closePop()?.focus(); }}})),
    h("div.sizes", {}, builtin),
    h("div.hd", {}, h("h4", {"data-i18n": "gen.size.mine"}), h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "gen.size.add", disabled: saved,
                        "data-i18n-title": saved ? "gen.size.added" : null,
                        on: {click: () => { saveSizes([...mine, cur]); redraw(); }}})),
    mine.length
      ? h("div.sizes", {}, mine.map(([w, ht], i) => card(ratioName(w, ht), w, ht, () => { saveSizes(mine.filter((_, j) => j !== i)); redraw(); })))
      : null,
    h("p.hint", {text: t("gen.size.limit", SIZE_LIMIT)})];
}

function seedCell() {
  const seed = Number(G.s.SEED);
  const input = h("input.v", {id: "f_SEED", type: "number", inputmode: "numeric", min: -1, step: 1,
                              value: seed >= 0 ? String(seed) : "", "data-i18n-placeholder": "gen.seed.random",
                              on: {change: e => {
                                const v = e.target.value === "" ? -1 : toNum(e.target.value, {min: -1, step: 1, old: G.s.SEED});
                                e.target.value = v >= 0 ? String(v) : "";
                                set("SEED", v);
                              }}});
  return h("div.cell.w3.seed", {},
    h("label.k", {for: "f_SEED", "data-i18n": "gen.seed"}),
    h("div.v", {}, input,
      h("button.mini", {type: "button", "data-i18n-title": "gen.seed.dice", "data-i18n-aria": "gen.seed.dice",
                        on: {click: () => { set("SEED", -1); $("f_SEED").value = ""; }}}, icon(DIE, 13)),
      h("button.mini", {type: "button", id: "btnLastSeed", on: {click: () => {
        const v = lastSeed();
        if (v == null) return;
        set("SEED", v);
        $("f_SEED").value = String(v);
      }}}, icon(REUSE, 13)),
      h("button.mini.tog", {type: "button", id: "btnFix", "aria-pressed": String(!!G.s.FIX_SEED),
                            "data-i18n": "gen.seed.fix", "data-i18n-title": "gen.seed.fixHint",
                            on: {click: e => {
                              set("FIX_SEED", !G.s.FIX_SEED);
                              e.currentTarget.setAttribute("aria-pressed", String(G.s.FIX_SEED));
                            }}})));
}

// 最近生成完成的那一張的種子：進度表（這次開著的工作台）裡最後一個完成的
function lastSeed() {
  const T = G.state?.table || [];
  for (let i = T.length - 1; i >= 0; i--) {
    if (T[i].status === "done" && T[i].seed != null && T[i].seed >= 0) return Number(T[i].seed);
  }
  return null;
}

function drawLastSeed() {
  const b = $("btnLastSeed");
  if (!b) return;
  const v = lastSeed();
  b.disabled = v == null;
  b.title = v == null ? t("gen.seed.lastNone") : t("gen.seed.last", {seed: v});
  b.setAttribute("aria-label", b.title);
}

function seg(key, values, labelPrefix) {
  return h("div.seg", {role: "group"}, values.map(v => h("button", {
    type: "button", "aria-pressed": String(G.s[key] === v), "data-i18n": `${labelPrefix}.${v}`,
    on: {click: e => {
      set(key, v);
      e.currentTarget.parentElement.querySelectorAll("button")
        .forEach(b => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
      drawAdv();
    }}})));
}

// 進階格上直接寫出目前的 VRAM 策略，不用打開也看得到
function drawAdv() {
  const v = $("advValue");
  if (v) v.textContent = t(`gen.vram.${G.s.VRAM_MODE || "balanced"}`);
}

function advPop() {
  const row = (label, control, hintKey) => {
    const [btn, hint] = hintKey ? info(hintKey) : [null, null];
    return h("div.row", {}, h("span.lbl", {}, t(label), btn), control, hint);
  };
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "gen.adv.title"})),
    row("gen.vram", seg("VRAM_MODE", VRAM_MODES, "gen.vram")),
    row("gen.cond", seg("COND_MODE", COND_MODES, "gen.cond"), "gen.cond.hint"),
    row("gen.tile", seg("TILE_VAE", AUTO_ON_OFF, "gen.opt"), "gen.tile.hint"),
    row("gen.lowDecode", seg("LOW_DECODE", AUTO_ON_OFF, "gen.opt"))];
}

const SHIFT_HINT = {anima: "gen.shift.hint", zimage: "gen.shift.hintZ", krea2: "gen.shift.hintK", qwen21: "gen.shift.hintQ",
                    klein4: "gen.shift.hintF", klein9: "gen.shift.hintF", flux1: "gen.shift.hintF1", flux1s: "gen.shift.hintF1s",
                    kontext: "gen.shift.hintF1"};

function params() {
  const a = arch();
  // flow 架構不用 CLIP，CLIP skip 換成它們才有的 shift
  const fifth = FLOW_ARCHS.includes(a)
    ? numCell(shiftKey(a), "gen.shift", {min: 0.1, max: 20, step: 0.1, hint: SHIFT_HINT[a]})
    // SDXL 的兩顆文字編碼器本來就用倒數第二層，1 跟 2 一樣（Forge 也是最低 2，見 pipeline._effective_clip_skip）
    : numCell("CLIP_SKIP", "gen.clipSkip", a === "sdxl" ? {min: 2, floor: 2, max: 12, step: 1, hint: "gen.clipSkip.sdxlHint"}
                                                        : {min: 1, max: 12, step: 1});
  return h("div.g5.params", {id: "params"},
    sizeCell(),
    numCell("STEPS", "gen.steps", {min: 1, max: 200, step: 1}),
    // FLUX.1 dev、Kontext 的 CFG 欄是官方的 guidance（core/flux1.py）
    numCell("CFG", "gen.cfg", {min: 0, max: 30, step: 0.5, hint: a === "flux1" || a === "kontext" ? "gen.cfg.hintF1" : null}),
    fifth,
    selectCell("SAMPLER", "gen.sampler", ".w3", samplerOptions()),
    selectCell("SCHEDULE", "gen.schedule", ".w2", SCHEDULES.map(n => h("option", {value: n, text: n}))),
    seedCell(),
    h("button.cell.w2.adv", {type: "button", id: "btnAdv", "aria-haspopup": "dialog", "aria-expanded": "false",
                             on: {click: e => openPop("popAdv", e.currentTarget, advPop)}},
      h("span.k", {"data-i18n": "gen.adv"}),
      h("span.v", {}, h("span", {id: "advValue"}), h("span.more", {text: "⋯", "aria-hidden": "true"}))));
}

// ── 生成鈕與進度 ─────────────────────────────────
function goBar() {
  const step = d => () => {
    const v = Math.max(1, Math.min(100, (Number(G.s.BATCH) || 1) + d));
    set("BATCH", v);
    $("f_BATCH").value = v;
  };
  return h("div.go", {},
    // 輕量安裝：看起來是鎖住的，按了在旁邊說明要完整安裝（生成台其他東西照常能用）
    h("button.gobtn", {type: "button", id: "btnGo", "data-lite": INSTALL.lite ? "" : null,
                       "aria-haspopup": INSTALL.lite ? "dialog" : null, on: {click: run}},
      h("span.l", {id: "goLabel", "data-i18n": "gen.go"}),
      h("kbd", {text: "Ctrl ⏎"})),
    h("div.cnt", {role: "group", "data-i18n-aria": "gen.batch", "data-i18n-title": "gen.batch"},
      h("button", {type: "button", "data-i18n-aria": "gen.batch.less", on: {click: step(-1)}}, "−"),
      numInput("BATCH", {min: 1, max: 100, step: 1, aria: "gen.batch"}),
      h("button", {type: "button", "data-i18n-aria": "gen.batch.more", on: {click: step(1)}}, "＋")),
    // 兩個小按鈕上下疊：加入排程（排程彈窗在標題列的「排程」）、生成後自動修臉（像 ADetailer；
    // 修改幅度、要不要留沒修過的在 ⚙ 設定 → 生成）
    h("div.gosub", {},
      h("button.gotog", {type: "button", id: "btnQueueAdd", "data-i18n-title": "gen.q.addHint", on: {click: () => queueAdd()}},
        h("span", {"data-i18n": "gen.q.add"})),
      h("button.gotog", {type: "button", id: "btnFace", "aria-pressed": String(!!G.s.AUTO_FACE),
                         "data-i18n-title": "gen.autoFaceHint", on: {click: e => {
                           set("AUTO_FACE", !G.s.AUTO_FACE);
                           e.currentTarget.setAttribute("aria-pressed", String(!!G.s.AUTO_FACE));
                         }}},
        h("span", {"data-i18n": "gen.autoFace"}))));
}

function runBox() {
  return h("div.run", {},
    h("div.track", {}, h("i", {id: "runBar"})),
    h("div.runl", {id: "runLine"}),
    h("div.runl.sub", {},
      h("span.msg", {id: "genMsg", role: "status"}),
      h("span.sp"),
      h("button.pill", {type: "button", id: "btnCancel", hidden: true, "data-i18n": "gen.cancel", on: {click: cancel}})));
}

// ── 生完送去比較（使用者筆記）──
// 開關在 ⚙ 設定 → 生成（setup.js 的 UI_BOOLS）。開關開著、一批生了 2 張以上：整批結束、重新掃描完，建一個比較組，訊息欄給「去淘汰賽」。
// 好幾個視窗都會偵測到這批生完了，用第一張的路徑當識別，伺服器同一批只建一組
async function autoCompare(rows, scanned) {
  if (!G.hooks.ui?.autocmp) return;
  if (rows.some(r => r.kind)) return;    // 成對圖輔助做的（server/pairmaker.py）：圖已經搬到資料集，不進比較組
  const paths = rows.filter(r => r.status === "done" && r.path).map(r => r.path);
  if (paths.length < 2) return;
  await scanned;                         // 圖庫裡要先有這些圖
  const when = new Date().toLocaleString(undefined, {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false});
  const d = await post("/api/compare/add", {paths, auto: paths[0], name: t("gen.autoCmpName", {when, n: paths.length})})
    .catch(e => ({error: e.message}));
  const el = $("genMsg");
  if (!el) return;
  if (d.error) { say(t("gen.autoCmpErr", {msg: d.error}), "err"); return; }
  clearTimeout(sayTimer);
  el.dataset.kind = "ok";
  el.replaceChildren(t("gen.autoCmpDone", {n: d.count}), " ",
    h("a", {href: "#compare", on: {click: e => { e.preventDefault(); G.hooks.openCup?.(d.id); }}}, t("gen.autoCmpGo")));
}

function batchRows(s) {
  const T = s.table || [];
  if (!T.length) return [];
  const b = Math.max(...T.map(r => r.b ?? 0));
  return T.filter(r => (r.b ?? 0) === b);
}

function drawRun() {
  const line = $("runLine");
  if (!line) return;
  const s = G.state || {}, rows = batchRows(s), running = !!s.running;
  const go = $("btnGo");
  go.disabled = running || G.busy;
  $("goLabel").textContent = t(running ? "gen.going" : "gen.go");
  $("btnCancel").hidden = !running;
  drawLastSeed();
  line.replaceChildren();
  drawQueueBtn();
  const ql = queueLine();
  const bar = $("runBar");
  const secs = rows.filter(r => r.status === "done" && r.secs != null).map(r => Number(r.secs));
  const avg = secs.length ? secs.reduce((a, b) => a + b, 0) / secs.length : null;
  if (running) {
    const done = rows.filter(r => FINISHED.has(r.status)).length, n = Math.max(rows.length, 1);
    const frac = s.total ? s.step / s.total : 0;
    bar.style.width = `${Math.round((done + frac) / n * 100)}%`;
    if (!s.total) line.append(h("span", {text: t("gen.run.loading")}));
    else {
      line.append(h("span", {text: t("gen.run.image", {i: Math.min(done + 1, n), n})}),
                  h("span", {text: t("gen.run.steps", {step: s.step, total: s.total})}));
      if (avg) line.append(h("span.eta", {text: t("gen.run.eta", {s: Math.max(1, Math.round((n - done - frac) * avg))})}));
    }
  } else if (rows.length) {
    const ok = rows.filter(r => r.status === "done").length, skip = rows.filter(r => r.status === "skipped").length;
    const bad = rows.length - ok - skip;
    bar.style.width = ok ? "100%" : "0";
    line.append(h("span", {text: t("gen.run.done", {n: ok, s: avg ? avg.toFixed(1) : "–"})}));
    if (skip) line.append(h("span", {text: t("gen.run.skipped", {n: skip})}));
    if (bad) line.append(h("span.bad", {text: t("gen.run.failed", {n: bad})}));
    // 最後一行是錯誤就顯示出來，不必打開紀錄才看得到
    const lastLog = (s.log || []).at(-1);
    if (lastLog?.k === "err") line.append(h("span.bad", {text: lastLog.m, title: lastLog.m}));
  } else {
    bar.style.width = "0";
  }
  if (ql) line.append(h("span.qline", {text: ql}));
  const log = openedPop("popLog");
  if (log) fillLogPop(log);
  if (openedPop("popQueue")) refreshQueuePop();
}

// ── 動作 ───────────────────────────────────────
// 送出一批生成。over 蓋掉生成台的設定（精修頁用它帶底圖、強度、自己的提示詞）。
// 回傳 {queued, warn} 或 {error}（已翻好的訊息），不自己顯示——生成台與精修頁各有各的訊息欄。
// 這次要送出的內容：生成台的設定，over 蓋掉其中幾項。回傳 {body} 或 {error}（已翻好的訊息）
async function buildBody(over = {}) {
  const s = G.s, prompt = String(over.prompt ?? s.PROMPT ?? "").trim();
  if (!prompt) return {error: t("gen.err.empty")};
  const body = {
    prompt, negative: s.NEGATIVE || "", width: s.WIDTH, height: s.HEIGHT, steps: s.STEPS, cfg: s.CFG,
    seed: s.SEED, fix_seed: !!s.FIX_SEED, batch: s.BATCH || 1, sampler: s.SAMPLER, schedule: s.SCHEDULE,
    shift: s[shiftKey(arch())], clip_skip: s.CLIP_SKIP, cond_mode: s.COND_MODE, vram_mode: s.VRAM_MODE,
    low_decode: s.LOW_DECODE, tile_vae: s.TILE_VAE, preset: presetOn(),
    controlnets: (s.CONTROLNETS || []).filter(u => u.src && !u.off),
    ...over, prompt,
  };
  // 有動態語法就先展開。張數＝每種組合各生幾張（[[A|B]] 會相乘），跟舊介面的批次生成一樣
  if (DYN_SYNTAX.test(prompt)) {
    const d = await post("/api/expand", {template: prompt, count: body.batch || 1, seed: "",
                                         lock: s.TPL_LOCK_RANDOM !== false});
    if (d.error) return {error: t("gen.err.syntax", {msg: d.error})};
    if (!d.items?.length) return {error: t("gen.err.noCombo")};
    body.prompts = d.items.map(i => i.prompt);
    body.batch = 1;
  }
  return {body};
}

export async function submit(over = {}) {
  if (!G.s) return {error: t("gen.err.run", {msg: G.err || "settings"})};
  if (G.state?.running || G.busy) return {error: t("gen.busyNow")};
  G.busy = true;
  drawRun();
  try {
    await flush();
    const b = await buildBody(over);
    if (b.error) return b;
    const d = await post("/api/gen/run", b.body);
    if (d.error) return {error: t("gen.err.run", {msg: d.error})};
    poll();
    return d;
  } catch (e) {
    return {error: t("gen.err.run", {msg: e.message || e})};
  } finally {
    G.busy = false;
    drawRun();
  }
}

// 批次精修（refinebatch.js）：每張圖各自的送出內容，生成台的設定當底（跟 submit 一樣），伺服器一張一張排進佇列
export async function submitBatch(items) {
  if (!G.s) return {error: t("gen.err.run", {msg: G.err || "settings"})};
  if (G.state?.running || G.busy) return {error: t("gen.busyNow")};
  G.busy = true;
  drawRun();
  try {
    await flush();
    const s = G.s;
    const base = {negative: s.NEGATIVE || "", steps: s.STEPS, cfg: s.CFG, sampler: s.SAMPLER, schedule: s.SCHEDULE,
                  shift: s[shiftKey(arch())], clip_skip: s.CLIP_SKIP, cond_mode: s.COND_MODE, vram_mode: s.VRAM_MODE,
                  low_decode: s.LOW_DECODE, tile_vae: s.TILE_VAE, preset: presetOn(), batch: 1, fix_seed: false, seed: -1};
    const d = await post("/api/refine/batch", {items: items.map(it => ({...base, ...it}))});
    if (d.error) return {error: d.error, skipped: d.skipped};
    poll();
    return d;
  } catch (e) {
    return {error: e.message || String(e)};
  } finally {
    G.busy = false;
    drawRun();
  }
}

// ── 排程生成（使用者筆記 #7，server/jobqueue.py）──
// 「加入排程」把生成台當下的完整設定記成一項（模型、LoRA、VAE、提示詞、參數、張數；動態語法先展開），
// 之後換模型、改提示詞都不影響已經排好的。排程彈窗裡看清單、調順序、開始／暫停、選冷卻方式
async function snapshot(over = {}) {
  const b = await buildBody(over);
  if (b.error) return b;
  const s = G.s;
  return {body: {...b.body, model: s.MODEL || "", loras: (s.LORAS || []).map(l => [l.path, l.weight]),
                 vae_path: s.VAE_OVERRIDE ?? "", te_path: s.TE_OVERRIDE ?? "", auto_face: !!s.AUTO_FACE}};
}

async function queueAdd(prompts = null) {
  if (!G.s) return;
  await flush();
  const items = [];
  for (const p of prompts || [null]) {
    const r = await snapshot(p == null ? {} : {prompt: p});
    if (r.error) { say(r.error, "err"); return; }
    items.push(r.body);
  }
  const d = await post("/api/queue/add", {items}).catch(e => ({error: e.message}));
  if (d.error || !d.added) { say(t("gen.q.addErr", {msg: d.error || (d.errors || []).join("；")}), "err"); return; }
  say(t("gen.q.added", {n: d.added}) + (d.errors?.length ? ` ${d.errors.join("；")}` : ""), d.errors?.length ? "warn" : "ok");
  poll();
  refreshQueuePop();
}

const qrun = () => G.state?.queue_run || {};

// 排程現在在做什麼（進度列與彈窗共用）：休息中、等降溫、暫停中
function queueLine(q = qrun()) {
  if (q.state === "cooling") return t("gen.q.resting", {s: Math.max(0, Math.ceil((q.until || 0) - Date.now() / 1000))});
  if (q.state === "hot") return t("gen.q.hot", {t: q.temp ?? "?", target: q.target});
  if (q.state === "running") return t(q.pausing ? "gen.q.pausing" : "gen.q.running", {n: q.left});
  return q.msg || "";
}

function drawQueueBtn() {
  const b = $("btnQueue");
  if (!b) return;
  const q = qrun();
  b.querySelector(".qn").textContent = q.waiting ? String(q.waiting) : "";
  b.toggleAttribute("data-on", q.state && q.state !== "idle");
}

const QSTATUS = {waiting: "gen.q.st.waiting", running: "gen.q.st.running", done: "gen.q.st.done", failed: "gen.q.st.failed"};

function queuePop() {
  const list = h("div.qlist", {id: "qList"}), head = h("div.qhead", {id: "qHead"}), cool = h("div.qcool", {id: "qCool"});
  const lines = h("textarea.rftext", {rows: "4", spellcheck: "false", "data-i18n-placeholder": "gen.q.linesPh", "data-i18n-aria": "gen.q.lines"});
  queueMicrotask(() => refreshQueuePop(true));
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "gen.q.title"})),
    head, cool, list,
    h("details.qbulk", {},
      h("summary", {"data-i18n": "gen.q.lines"}),
      h("p.hint", {"data-i18n": "gen.q.linesHint"}),
      lines,
      h("button.pill", {type: "button", "data-i18n": "gen.q.addLines", on: {click: async () => {
        const ps = lines.value.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
        if (!ps.length) return;
        await queueAdd(ps);
        lines.value = "";
      }}}))];
}

let qFetch = 0;
async function refreshQueuePop(full = false) {
  const pop = openedPop("popQueue");
  if (!pop) return;
  const seq = ++qFetch;
  const d = await get("/api/queue").catch(() => null);
  if (!d || seq !== qFetch || !openedPop("popQueue")) return;
  const q = d.run, running = q.state !== "idle";
  const act = (url, body = {}) => async () => { await post(url, body).catch(() => null); poll(); refreshQueuePop(); };
  const finished = d.items.some(it => it.status === "done" || it.status === "failed");
  $("qHead").replaceChildren(...[
    running ? h("button.pill", {type: "button", "data-i18n": q.pausing ? "gen.q.resume" : "gen.q.pause",
                                on: {click: act(q.pausing ? "/api/queue/start" : "/api/queue/pause")}})
      : h("button.pill.primary", {type: "button", "data-i18n": "gen.q.start", disabled: !d.items.some(it => it.status === "waiting"),
                                  on: {click: act("/api/queue/start")}}),
    h("span.qstate", {text: queueLine(q) || t(d.items.length ? "gen.q.idle" : "gen.q.empty", {n: q.left})}),
    h("span.sp"),
    finished ? h("button.mini", {type: "button", "data-i18n": "gen.q.clear", on: {click: act("/api/queue/clear")}}) : null].filter(Boolean));
  if (full || !$("qCool").childElementCount) drawQueueCool(d.cool);
  const n = d.items.length;
  $("qList").replaceChildren(...(n ? d.items.map((it, i) => h("div.qitem", {"data-status": it.status},
    h("span.qi.mono", {text: String(i + 1)}),
    h("div.qtxt", {},
      h("b", {text: it.label || "—", title: it.label}),
      h("span.dim", {text: `${it.model} · ${it.size?.[0] ?? "?"}×${it.size?.[1] ?? "?"} · ${t("gen.q.count", {done: it.done, total: it.total})}`
                     + (it.failed ? ` · ${t("gen.q.failedN", {n: it.failed})}` : "")})),
    h("span.qst", {"data-i18n": QSTATUS[it.status] || QSTATUS.waiting}),
    it.status === "failed" ? h("button.mini", {type: "button", "data-i18n": "gen.q.retry",
                                               on: {click: act("/api/queue/retry", {id: it.id})}}) : null,
    h("button.mini", {type: "button", "data-i18n-aria": "gen.q.up", "data-i18n-title": "gen.q.up", disabled: i === 0,
                      on: {click: act("/api/queue/move", {id: it.id, to: i - 1})}}, "↑"),
    h("button.mini", {type: "button", "data-i18n-aria": "gen.q.down", "data-i18n-title": "gen.q.down", disabled: i === n - 1,
                      on: {click: act("/api/queue/move", {id: it.id, to: i + 1})}}, "↓"),
    h("button.mini", {type: "button", "data-i18n-aria": "gen.q.remove", "data-i18n-title": "gen.q.remove", disabled: it.status === "running",
                      on: {click: act("/api/queue/remove", {id: it.id})}}, "✕")))
    : []));
  translate(pop);
}

// 冷卻方式：不休息、每幾張休息幾秒、等溫度降到幾度（改了馬上存）
function drawQueueCool(c) {
  const box = $("qCool");
  if (!box) return;
  const save = patch => { Object.assign(c, patch); post("/api/queue/cool", c).catch(() => null); drawQueueCool(c); };
  const num = (key, min, max) => h("input.find.qnum", {type: "number", min: String(min), max: String(max), step: "1", value: String(c[key]),
                                                      "data-i18n-aria": `gen.q.cool.${key}`,
                                                      on: {change: e => save({[key]: Math.max(min, Math.min(max, Math.round(Number(e.target.value) || min)))})}});
  box.replaceChildren(...[
    h("div.rflab", {}, h("b", {"data-i18n": "gen.q.cool"}), h("span.sp"),
      h("div.seg", {role: "group", "data-i18n-aria": "gen.q.cool"}, ["off", "count", "temp"].map(k => h("button", {
        type: "button", "aria-pressed": String(c.mode === k), "data-i18n": `gen.q.cool.${k}`, on: {click: () => save({mode: k})}})))),
    c.mode === "count" ? h("p.qcoolrow", {}, t("gen.q.cool.every1"), num("every", 1, 1000), t("gen.q.cool.every2"), num("rest", 1, 3600), t("gen.q.cool.every3"))
      : c.mode === "temp" ? h("p.qcoolrow", {}, t("gen.q.cool.temp1"), num("temp", 40, 95), t("gen.q.cool.temp2"))
      : null].filter(Boolean));
  translate(box);
}

export async function run() {
  if (INSTALL.lite) { const go = $("btnGo"); if (go) openPop("popLite", go, litePopBody); return; }      // Ctrl+Enter 也一樣
  const d = await submit();
  if (d.error) { say(d.error, "err"); return; }
  say(d.warn ? t("gen.queuedWarn", {n: d.queued, warn: d.warn}) : t("gen.queued", {n: d.queued}),
      d.warn ? "warn" : "ok");
}

export async function cancel() {
  try {
    const d = await post("/api/gen/cancel", {});
    say(t("gen.cancelled", {n: d.dropped ?? 0}));
  } catch { /* 下一次輪詢會反映真正的狀態 */ }
}

async function unload() {
  try {
    await post("/api/gen/stop", {});
    say(t("gen.unloaded"), "ok");
    poll();
  } catch { /* 同上 */ }
}

// 後端進度表的狀態代碼 → 顯示文字與樣式
const STATUS = {done: ["gen.st.done", "ok"], failed: ["gen.st.failed", "bad"], aborted: ["gen.st.aborted", "bad"],
                cancelled: ["gen.st.cancelled", "bad"], running: ["gen.st.running", "now"], waiting: ["gen.st.waiting", ""],
                skipped: ["gen.st.skipped", ""]};

function fillLogPop(pop) {
  pop.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.tab === G.logTab)));
  const log = pop.querySelector(".log"), table = pop.querySelector(".gtable");
  log.hidden = G.logTab !== "log";
  table.hidden = G.logTab !== "table";
  if (G.logTab === "log") fillLog(log);
  else fillTable(table);
}

// 進度表：這次開啟 CoCo 以來每一張的狀態（後端只留最近 300 列），一批一段
function fillTable(box) {
  const T = G.state?.table || [];
  const sig = JSON.stringify(T);
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig;
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  if (!T.length) { box.replaceChildren(h("p.dim", {text: t("gen.table.empty")})); return; }
  const out = [], batches = [...new Set(T.map(r => r.b ?? 0))];
  const total = T.filter(r => r.status === "done").length;
  out.push(h("p.gsum", {text: t("gen.table.total", {n: total})}));
  for (const b of batches) {
    const rows = T.filter(r => (r.b ?? 0) === b);
    out.push(h("div.gbh", {text: t("gen.table.batch", {b, ok: rows.filter(r => r.status === "done").length, n: rows.length})}));
    for (const r of rows) {
      const [key, kind] = STATUS[r.status] || ["", ""];
      const pct = r.total ? Math.round(r.step / r.total * 100) : 0;
      const right = r.status === "running"
        ? [h("span.bar", {}, h("i", {style: `width:${pct}%`})), h("span.m", {text: `${r.step}/${r.total}`})]
        : [h("span.m", {text: [r.secs != null ? `${r.secs}s` : "", r.seed != null ? `seed ${r.seed}` : ""].filter(Boolean).join(" · ")})];
      const kids = [h("span.n", {text: String(r.n ?? r.i + 1)}), h("span.s", {text: key ? t(key) : r.status}),
                    h("span.p", {text: r.prompt || "", title: r.prompt || ""}), ...right];
      // 完成的那張可以直接看大圖（在牆上找得到的話）
      out.push(r.status === "done" && r.file
        ? h("button.tr", {type: "button", "data-kind": kind, title: t("gen.table.open"),
                          on: {click: () => { closePop(); G.hooks.openFile?.(r.file); }}}, kids)
        : h("div.tr", {"data-kind": kind}, kids));
    }
  }
  box.replaceChildren(...out);
  const now = box.querySelector('.tr[data-kind="now"]');
  if (now) now.scrollIntoView({block: "nearest"});
  else if (atBottom || !box.dataset.seen) box.scrollTop = box.scrollHeight;
  box.dataset.seen = "1";
}

function fillLog(box) {
  if (!box) return;
  const L = G.state?.log || [];
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.replaceChildren(...(L.length ? L.map(l => h("p", {"data-kind": l.k},
    h("b.mono", {text: l.t}), h("span", {text: l.m}))) : [h("p.dim", {text: t("gen.log.empty")})]));
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// ── 輪詢 ───────────────────────────────────────
let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const was = !!G.state?.running;
    const s = await get("/api/gen/state");
    G.state = s;
    drawRun();
    G.hooks.progress?.(s, G.s);
    const fresh = G.last !== undefined && s.last && s.last !== G.last;
    G.last = s.last ?? null;
    // 每出一張就更新圖庫；整批結束再補一次，免得重掃撞在一起漏掉最後一張
    const scanned = fresh || (was && !s.running) ? newImages() : null;
    if (was && !s.running) autoCompare(batchRows(s), scanned);
  } catch { /* 伺服器暫時連不上就下次再試 */ }
  // 生成中、排程在跑（休息倒數、等降溫）時輪詢快一點
  pollTimer = setTimeout(poll, G.state?.running || (qrun().state || "idle") !== "idle" ? 700 : 3000);
}

async function newImages() {
  try { await post("/api/reindex", {}); } catch { /* 沒掃到就等下一張 */ }
  G.hooks.newImages?.();
}

// ── 組起來 ─────────────────────────────────────
function redraw() {
  const old = $("rail");
  if (!old) return;
  closePop();
  closeBig();                    // 編輯區要跟著重建，先搬回原位
  const neu = railCard();
  old.replaceWith(neu);
  translate(neu);
}

export function railCard() {
  if (!G.s) {
    return h("aside.card.rail", {id: "rail", "aria-labelledby": "railTitle"},
      h("h2.title", {id: "railTitle", "data-i18n": "rail.title"}),
      h("p.dim", {text: t("gen.err.load", {msg: G.err})}));
  }
  const a = arch();
  const el = h("aside.card.rail", {id: "rail", "aria-labelledby": "railTitle"},
    h("div.rhd", {},
      h("h2.title", {id: "railTitle", "data-i18n": "rail.title"}),
      a ? h("span.tag", {text: ARCH_LABEL[a] || a}) : null,
      G.hooks.recipeButton?.() ?? null,
      h("span.sp"),
      h("button.ghost.qbtn", {type: "button", id: "btnQueue", "aria-haspopup": "dialog", "aria-expanded": "false",
                              on: {click: e => openPop("popQueue", e.currentTarget, queuePop)}},
        h("span", {"data-i18n": "gen.q.title"}), h("span.qn.mono")),
      h("button.ghost", {type: "button", id: "btnLog", "aria-haspopup": "dialog", "aria-expanded": "false",
                         "data-i18n": "gen.log",
                         on: {click: e => openPop("popLog", e.currentTarget, () => {
                           const box = h("div.log", {hidden: true}), table = h("div.gtable", {hidden: true});
                           const tab = k => h("button", {type: "button", "data-tab": k, "data-i18n": `gen.tab.${k}`,
                                                         "aria-pressed": String(G.logTab === k),
                                                         on: {click: e => { G.logTab = k; fillLogPop(e.currentTarget.closest(".pp")); }}});
                           queueMicrotask(() => { const pop = openedPop("popLog"); if (pop) fillLogPop(pop); });
                           // 卸載模型放在紀錄小視窗裡，標題列才放得下 Recipe
                           return [h("div.hd", {}, h("h3.title", {"data-i18n": "gen.log.title"}),
                             h("div.seg", {}, tab("table"), tab("log")),
                             h("button.pill", {type: "button", "data-i18n": "gen.unload", "data-i18n-title": "gen.unload.hint",
                                               on: {click: () => { closePop(); unload(); }}})), table, box];
                         })}})),
    slots(), G.hooks.recipePills?.() ?? null, editor(), params(), goBar(), runBox());
  queueMicrotask(() => {
    drawAspect(); drawTokens(); drawRun(); drawAdv();
    if (G.notice) { say(G.notice.text, G.notice.kind); G.notice = null; }
  });
  return el;
}

export {closePop};

// ── 給資源庫用的 ──
export const genSettings = () => G.s;
// 別的模組改生成台的設定（ControlNet 的參考圖視窗）：跟生成台自己改一樣會存檔、同步到別的視窗、重畫資源格
export function setGen(key, value) {
  set(key, value);
  drawSlots();
}

// 換掉提示詞（資源庫加減 LoRA、改權重）：存檔、更新編輯區、資源格與 token 數
export function setPromptText(text) {
  setField("PROMPT", text);
}

// 改正面（PROMPT）或負面（NEGATIVE）提示詞，編輯區正開著那一邊就跟著更新（資源庫的 Embeddings 用）
export function setField(key, text) {
  if (!G.s || (key !== "PROMPT" && key !== "NEGATIVE")) return;
  set(key, text);
  showField(key);
}

// 畫面跟上 G.s 裡的提示詞（自己改的、別的視窗改的都走這裡）
function showField(key) {
  const text = G.s[key] || "";
  const ta = $("genText");
  if (ta && G.tab === key && ta.value !== text) ta.value = text;
  if (G.tab === key && chipMode()) G.chips?.refresh();
  if (key === "PROMPT") {
    const dyn = $("dynHint");
    if (dyn) dyn.hidden = !DYN_SYNTAX.test(text);
    drawSlots();
  }
  tokQueue();
}

// 編輯區目前開著哪一邊：PROMPT / NEGATIVE。資源庫開在另一個視窗時，看的是最後切換的那個視窗
export const editingField = () => G.lastTab || G.tab;

// 把萬用字元或變數插進提示詞：插在上次游標停的地方，沒有就接在最後，前後補逗號
export function insertIntoPrompt(piece) {
  if (!G.s) return;
  const text = G.s.PROMPT || "";
  const at = Math.min(G.caret ?? text.length, text.length);
  const before = text.slice(0, at).replace(/\s+$/, ""), after = text.slice(at).replace(/^\s+/, "");
  const left = before && !/[,(\[{|]$/.test(before) ? `${before}, ` : before ? `${before} ` : "";
  const right = after && !/^[,)\]}|]/.test(after) ? `, ${after}` : after ? ` ${after}` : "";
  G.caret = left.length + piece.length;
  setPromptText(left + piece + right);
}

// 模型、VAE、編碼器在別處換過了：重讀設定，架構變了就整個控制台重畫
// quiet：別的視窗通知的，不用再廣播出去
export async function reloadSettings(quiet = false) {
  try {
    const s = await get("/api/settings");
    if (!s.error) syncSettings(s);
    if (!quiet) send("reload");
    return s;
  } catch { return null; }
}

// 從別的地方整組帶入參數（圖庫的「送到生成區」）。先存檔，再重讀設定——
// 模型可能剛換過，架構跟著變，參數格（CLIP skip／shift）要重畫。
// notice：控制台下次畫出來時顯示的訊息（通常這時候還在圖庫空間）。
export async function applyParams(patch, notice) {
  if (!G.s) return;
  Object.assign(G.s, patch);
  Object.assign(G.dirty, patch);
  if ("PROMPT" in patch) G.dirty.TEMPLATE = patch.PROMPT;
  G.tab = "PROMPT";
  await flush();
  try {
    const s = await get("/api/settings");
    if (!s.error) for (const k of READONLY) G.s[k] = s[k];
  } catch { /* 下一次狀態輪詢會補上 */ }
  G.notice = notice || null;
  redraw();
  tokQueue();
  send("reload");                // 模型、零件可能也在伺服器上換了，別的視窗重讀
}

export async function initGenerate(hooks) {
  G.hooks = hooks || {};
  try {
    const s = await get("/api/settings");
    if (s.error) throw new Error(s.error);
    // 舊版設定檔只有 TEMPLATE 的話拿來用
    if (!s.PROMPT && s.TEMPLATE) s.PROMPT = s.TEMPLATE;
    G.s = s;
  } catch (e) {
    G.err = e.message || String(e);
  }
  poll();
  if (G.s) { tokRun(); refreshPreset(); }
}
