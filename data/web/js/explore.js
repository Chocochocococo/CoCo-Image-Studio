// 探索（工作室的 02，server/explore.py）：找相似的圖、用文字找圖（可以打中文）。
// 範圍：CoCo 的圖庫，加上參考圖來源（⚙ 設定 → 參考圖：資料夾或 Eagle 的檔案庫）。
// 點結果就以那張繼續找，上面留著一路找過來的足跡，可以點回去。
// 看圖的模型（SigLIP2）不內建，第一次按了才下載；索引進來這頁時自動更新（只算新的、改過的）

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {partPct} from "./modeldl.js";
import {openRefine} from "./refine.js";
import {openViewer} from "./viewer.js";

const X = {ui: null, status: null, query: null, trail: [], items: [], busy: false, msg: "", timer: 0};
const SCOPES = ["all", "lib", "ref"];
// Pinterest 搜尋時拿掉的：品質詞、評分、年份這類跟畫面內容無關的標籤
const META = /^(masterpiece|best quality|amazing quality|high quality|very aesthetic|absurdres|highres|newest|safe|sensitive|nsfw|explicit|score_\d+(_up)?|year \d+|(very )?awa|lazyhand)$/i;
// 每張都有、在 Pinterest 上找不到東西的通用標籤：人數、構圖、視角
const GENERIC = /^(\d+(girl|boy|other)s?|solo|male focus|female focus|multiple (girls|boys)|full body|upper body|cowboy shot|portrait|close-?up|looking at viewer|simple background|white background|standing|sitting|from (side|above|below|behind)|three-quarter view|depth of field|bokeh)$/i;

export function initExplore(ui) { X.ui = ui; }

const saveUi = patch => { Object.assign(X.ui, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const scope = () => (SCOPES.includes(X.ui?.xscope) ? X.ui.xscope : "all");
const thumbOf = it => (it.kind === "ref" ? `/api/explore/ref/thumb?id=${it.id}` : `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v || "")}`);

// 從別的地方（圖庫、看大圖）點「找相似」
export function openExplore(item) {
  X.trail = [];
  X.query = item;
  X.items = [];
  if (location.hash !== "#explore") location.hash = "#explore";
  else showExplore();
}

export function exploreScreen() {
  return h("div.rfm.xm", {},
    h("aside.card.rail.rfpanel.xpanel", {id: "xPanel", "aria-labelledby": "xTitle"}),
    h("section.card.rfstage.xstage", {id: "xStage", "aria-live": "polite"}));
}

export async function showExplore() {
  if (!$("xPanel")) return;
  await refreshStatus();
  const S = X.status;
  // 進來就更新索引（只算新的、改過的；沒有要算的很快就結束）
  if (S?.model.ready && !S.index.running && Date.now() / 1000 - (S.index.finished || 0) > 30) {
    await post("/api/explore/index", {}).catch(() => null);
    await refreshStatus();
  }
  drawPanel();
  if (X.query && !X.items.length) await search();
  else drawStage();
}

async function refreshStatus() {
  X.status = await get("/api/explore/status").catch(() => null);
  clearTimeout(X.timer);
  const S = X.status;
  if (S && (S.model.download?.running || S.index.running)) {
    X.timer = setTimeout(async () => {
      const wasIdx = X.status?.index.running;
      await refreshStatus();
      if (!$("xPanel")) return;
      drawPanel();
      // 下載完：開始建索引；索引建完：有查詢就重查（結果會多出剛算好的）
      if (X.status?.model.ready && !X.status.index.running && !X.status.index.finished) {
        await post("/api/explore/index", {}).catch(() => null);
        await refreshStatus();
        drawPanel();
      } else if (wasIdx && !X.status?.index.running && X.query) search();
    }, 1000);
  }
}

// ── 左欄 ──
function drawPanel() {
  const box = $("xPanel");
  if (!box) return;
  const S = X.status;
  const q = X.query;
  const text = h("input.find.xtext", {type: "search", value: q?.text || "", spellcheck: "false", "data-i18n-placeholder": "x.textPh",
                                      "data-i18n-aria": "x.textPh",
                                      on: {keydown: e => { if (e.key === "Enter" && e.target.value.trim()) setQuery({text: e.target.value.trim()}); }}});
  box.replaceChildren(...[
    h("div.rhd", {}, h("h2.title", {id: "xTitle", "data-i18n": "nav.explore"})),
    S && !S.model.ready ? modelCard(S.model) : null,
    S?.model.ready ? indexLine(S) : null,
    S?.model.ready ? h("div.rfsec", {}, h("div.rflab", {}, h("b", {"data-i18n": "x.byText"})), text) : null,
    q ? querySection(q) : (S?.model.ready ? h("p.hint", {"data-i18n": "x.howTo"}) : null),
    S?.model.ready ? h("div.rfsec", {},
      h("div.rflab", {}, h("b", {"data-i18n": "x.scope"}), h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "x.scope"}, SCOPES.map(k => h("button", {
          type: "button", "aria-pressed": String(scope() === k), "data-i18n": `x.scope.${k}`,
          on: {click: () => { saveUi({xscope: k}); drawPanel(); if (X.query) search(); }}})))),
      S.index.refs ? null : h("p.hint", {"data-i18n": "x.noRefs"})) : null,
    X.trail.length ? trailSection() : null,
  ].filter(Boolean));
  translate(box);
}

function modelCard(m) {
  const dl = m.download;
  const pct = dl?.running ? Math.round(dl.done / Math.max(1, dl.total) * 100) : 0;
  return h("div.rfsec.xmodel", {},
    h("p", {"data-i18n": "x.needModel"}),
    h("p.hint", {text: t("x.modelSize", {mb: Math.round(m.bytes / 1e6)})}),
    dl?.running
      ? h("div.xbar", {}, h("i", {style: `width:${pct}%`}), h("span.mono", {text: `${Math.round(dl.done / 1e6)} / ${Math.round(dl.total / 1e6)} MB`}))
      : null,
    dl?.error ? h("p.hint.warn", {text: dl.error})
      : m.part && !dl?.running ? h("p.hint", {text: t("dl.partial", {pct: partPct(m.part, m.bytes)})}) : null,
    h("div.rflab", {},
      dl?.running
        ? h("button.pill", {type: "button", "data-i18n": "x.cancel", on: {click: async () => { await post("/api/explore/model/cancel", {}); }}})
        : h("button.pill.primary", {type: "button", "data-i18n": "x.download", on: {click: async () => {
          await post("/api/explore/model/download", {}).catch(() => null);
          await refreshStatus();
          drawPanel();
        }}})));
}

function indexLine(S) {
  const I = S.index;
  const pct = I.total ? Math.round(I.done / I.total * 100) : 0;
  return h("div.rfsec.xidx", {},
    h("div.rflab", {},
      h("span.dim", {text: I.running
        ? (I.phase === "scan" ? t("x.scanning") : t("x.indexing", {done: I.done, total: I.total}))
        : t("x.indexed", {n: I.vecs, refs: I.refs})}),
      h("span.sp"),
      I.running ? null : h("button.mini", {type: "button", "data-i18n": "x.reindex", on: {click: async () => {
        await post("/api/explore/index", {}).catch(() => null);
        await refreshStatus();
        drawPanel();
      }}})),
    I.running && I.total ? h("div.xbar", {}, h("i", {style: `width:${pct}%`})) : null,
    I.error ? h("p.hint.warn", {text: I.error}) : null,
    ...(I.warn || []).map(w => h("p.hint.warn", {text: w})));
}

// 現在在找什麼：圖（縮圖＋動作）或文字
function querySection(q) {
  if (q.text) return h("div.rfsec", {}, h("div.rflab", {}, h("b", {"data-i18n": "x.query"})), h("p.xqtext", {text: q.text}));
  return h("div.rfsec.xquery", {},
    h("div.rflab", {}, h("b", {"data-i18n": "x.query"})),
    h("img.xqimg", {src: thumbOf(q), alt: "", on: {click: () => view([q], q)}}),
    h("p.dim.small", {text: q.kind === "ref" ? q.name || "" : q.filename || ""}),
    actions(q));
}

function actions(it) {
  return h("div.xacts", {}, ...[
    h("a.pill", {href: pinterestUrl(it), target: "_blank", rel: "noopener", "data-i18n": "x.pinterest", "data-i18n-title": "x.pinterestHint"}),
    it.kind === "lib" ? h("button.pill", {type: "button", "data-i18n": "lib.refine", on: {click: () => openRefine(it.id)}}) : null,
    it.kind === "ref" && it.eagle ? h("a.pill", {href: `eagle://item/${it.eagle}`, "data-i18n": "x.openEagle"}) : null,
    it.kind === "ref" && it.url ? h("a.pill", {href: it.url, target: "_blank", rel: "noopener", "data-i18n": "x.source"}) : null,
  ].filter(Boolean));
}

function trailSection() {
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "x.trail"}), h("span.sp"),
      h("button.mini", {type: "button", "data-i18n": "x.clearTrail", on: {click: () => { X.trail = []; drawPanel(); }}})),
    h("div.xtrail", {}, X.trail.map((q, i) => h("button.xcrumb", {type: "button", title: q.text || q.name || q.filename || "",
      on: {click: () => { X.trail = X.trail.slice(0, i); setQuery(q, false); }}},
      q.text ? h("span", {text: q.text.slice(0, 12)}) : h("img", {src: thumbOf(q), alt: ""})))));
}

// Pinterest 的搜尋：圖庫的圖用提示詞裡跟畫面有關的前幾個標籤，參考圖用 Eagle 的標籤（沒有就用名字）
export function pinterestQuery(it) {
  if (it.text) return it.text;
  if (it.kind === "ref") return (it.tags?.length ? it.tags.slice(0, 6) : [it.name || ""]).join(" ");
  const tags = String(it.prompt || "").replace(/<[^>]*>/g, "").split(/[,\n]+/)
    .map(s => s.replace(/[()[\]{}]/g, "").replace(/:\s*[\d.]+\s*$/, "").replace(/_/g, " ").trim())
    .filter(s => s && !META.test(s) && !GENERIC.test(s));
  return tags.slice(0, 6).join(" ");
}
const pinterestUrl = it => `https://www.pinterest.com/search/pins/?q=${encodeURIComponent(pinterestQuery(it))}`;

// ── 查詢 ──
function setQuery(q, remember = true) {
  if (remember && X.query) X.trail = [...X.trail, X.query].slice(-12);
  X.query = q;
  X.items = [];
  drawPanel();
  search();
}

async function search() {
  const q = X.query;
  if (!q) return;
  X.busy = true;
  X.msg = "";
  drawStage();
  const body = {scope: scope(), limit: 200, ...(q.text ? {text: q.text} : q.kind === "ref" ? {key: q.key} : {id: q.id})};
  const d = await post("/api/explore/search", body).catch(e => ({error: e.message}));
  if (X.query !== q) return;                      // 等的時候又換了
  X.busy = false;
  X.items = d.items || [];
  X.msg = d.error || "";
  // 從圖庫點進來的：補上提示詞（Pinterest 搜尋要用）
  if (q.kind === "lib" && q.prompt == null) {
    const r = await get(`/api/refine/items?ids=${q.id}`).catch(() => null);
    const src = r?.items?.[0];
    if (src) Object.assign(q, {prompt: src.user_prompt ?? src.prompt ?? "", filename: src.filename, v: q.v || src.v});
    drawPanel();
  }
  drawStage();
}

// ── 右邊：結果 ──
function drawStage() {
  const box = $("xStage");
  if (!box) return;
  const q = X.query;
  const head = h("div.rfhd", {},
    h("span.dim", {text: !q ? t("x.stageEmpty") : X.busy ? t("x.searching") : X.msg ? X.msg
      : t(q.text ? "x.foundText" : "x.found", {n: X.items.length})}));
  // 點圖：用看大圖一張張看（←／→ 換張，參數、送到生成區、精修…都在那裡）；右上角的按鈕才是「以這張繼續找」
  const nodes = new Map();
  const grid = h("div.xgrid", {}, X.items.map(it => h("div.xcard", {"data-kind": it.kind},
    h("button.xpic", {type: "button", title: t("x.preview"), on: {click: () => view(X.items, it, nodes)}},
      h("img", {src: thumbOf(it), alt: "", loading: "lazy", decoding: "async"})),
    // 相似度只在以圖找圖時有意義（文字跟圖的分數本來就低，數字反而誤導）
    q?.text ? null : h("span.xsim.mono", {text: `${Math.round(it.sim * 100)}`}),
    it.kind === "ref" ? h("span.xtag", {"data-i18n": "x.ref"}) : null,
    h("button.xzoom", {type: "button", "data-i18n-aria": "x.again", "data-i18n-title": "x.again", on: {click: () => setQuery(it)}}, "⌕"))));
  box.replaceChildren(head, grid);
  for (const el of grid.querySelectorAll(".xpic")) {
    const i = [...grid.children].indexOf(el.parentElement);
    nodes.set(vid(X.items[i]), el);
  }
  translate(box);
}

// 看大圖：探索找到的圖排成一串（參考圖的 id 前面加 r，跟圖庫的 id 分開）
const vid = it => (it.kind === "ref" ? `r${it.id}` : it.id);
function view(list, it, nodes = new Map()) {
  const items = list.map(x => ({...x, id: vid(x), rid: x.kind === "ref" ? x.id : undefined}));
  openViewer({items, total: items.length, offset: items.length, loadMore: async () => {}, nodes}, vid(it));
}
