// 訓練空間的 01 資料集（打標台，server/dataset.py）：從舊的獨立打標台搬進來的。
// 左欄：資料夾、打標籤的模型、觸發詞、門檻與清理規則、開始／取消、進度與紀錄。
// 右邊兩頁：「審核」（大圖、標籤用生成台的標籤模式改、描述另外一格、縮圖格多選做批次）與
// 「標籤統計」（整個資料集每個標籤幾張，可以整批拿掉、換成別的、全部加上一個）。
// 審核時要看得到整張圖又不能太小（使用者回饋）：圖獨佔中間一整欄、從上到下；標籤、描述、縮圖格在右欄；
// 左欄可以收起來（介面偏好 dsfold），收起來只留展開、打標籤與進度；點圖切換原始大小（看手指、配件）。
// WD14 的資料夾設定一次就好，平常在 ⚙ 設定 → 資源資料夾改；這頁只在還沒有模型時出現下載卡。
// 寫描述用 JoyCaption（使用者的 GGUF，資料夾在 ⚙ 設定 → 資源資料夾）：描述長度、把標籤當參考、提示詞（{length} 換成長度）。
// 標籤與描述由伺服器分開給、存的時候由伺服器用設定的分隔接回去。描述的中文送 Google 翻譯（按了才送）。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {ChipBox} from "./chips.js";
import {info} from "./hint.js";
import {partPct, size} from "./modeldl.js";
import {pickFolder} from "./setup.js";
import {useTrainConfig} from "./train.js";
import {keepUnsaved, serial} from "./inflight.js";

const D = {ui: null, dir: "", items: [], idx: 0, sel: new Set(), last: -1, opts: null, model: null,
           job: null, poll: 0, view: "review", stats: null, timer: 0, chips: null, dlArm: false, trashArm: false,
           trans: null, msg: "", busy: false, zoom: false, info: null};
const OPT_NUM = ["general", "character"];
const OPT_BOOL = ["keep_character", "underscore", "escape", "dedupe", "overwrite", "use_tags"];
const OPT_TEXT = ["remove_prefixes", "strip_labels", "separator"];

export function initDataset(ui) { D.ui = ui; }

const saveUi = patch => { Object.assign(D.ui, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const enc = encodeURIComponent;
const thumbUrl = it => `/api/dataset/thumb?dir=${enc(D.dir)}&path=${enc(it.path)}&v=${it.mtime}`;
const fullUrl = it => `/api/dataset/full?dir=${enc(D.dir)}&path=${enc(it.path)}&v=${it.mtime}`;
const cur = () => D.items[D.idx];

// 從圖庫「加進資料集」之後過來
export function openDataset(dir) {
  if (dir) saveUi({dsdir: dir});
  if (location.hash !== "#dataset") location.hash = "#dataset";
  else showDataset();
}

export function datasetScreen() {
  return h("div.rfm.dsm", {"data-fold": D.ui?.dsfold ? "1" : null},
    h("aside.card.rail.rfpanel.dspanel", {id: "dsPanel", "aria-labelledby": "dsTitle"}),
    h("section.card.rfstage.dsstage", {id: "dsStage"}));
}

export async function showDataset() {
  if (!$("dsPanel")) return;
  const d = await get("/api/dataset/opts").catch(() => null);
  if (d) Object.assign(D, {opts: d.opts, model: d.model});
  if (D.ui.dsdir && D.ui.dsdir !== D.dir) await openDir(D.ui.dsdir, true);
  else if (D.dir) await reload();
  drawPanel();
  drawStage();
  pollJob();
}

async function openDir(dir, quiet = false) {
  // 換資料集之前：這一個還沒存的全部先存。存不進去就留在這個（清單換掉，剛改的標籤與描述就沒了，R65）
  const err = await saveNow();
  if (err) {
    say(t("ds.unsavedStay", {error: err}), "err");
    if (D.ui.dsdir !== D.dir) saveUi({dsdir: D.dir});
    return false;
  }
  const d = await get(`/api/dataset/open?dir=${enc(dir)}`).catch(e => ({error: e.message}));
  if (d.error) {
    if (!quiet) say(d.error, "err");
    D.dir = ""; D.items = [];
    return false;
  }
  Object.assign(D, {dir: d.dir, items: d.items, idx: 0, sel: new Set(), last: -1, stats: null, trigger: d.trigger, trans: null, msg: ""});
  if (D.ui.dsdir !== d.dir) saveUi({dsdir: d.dir});
  D.info = await get(`/api/dataset/info?dir=${enc(d.dir)}`).catch(() => null);
  return true;
}

// 存標籤與重新讀一次排成一列（inflight.serial）：讀的時候不會有存到一半的，存的時候不會被讀回來的舊資料蓋掉
const io = serial();

// 重新讀一次（打完標籤、整批修改之後）：停在原本那張。
// 還沒存的改動（停 0.7 秒才存）照圖片路徑蓋回去、等一下照常存——以前背景工作做完時輪詢叫這裡，
// 剛改還沒存的標籤連同 dirty 一起被換掉（Codex 第三輪審查）
function reload() {
  return io(reloadOnce);
}

async function reloadOnce() {
  const keep = cur()?.path;
  const d = await get(`/api/dataset/open?dir=${enc(D.dir)}`).catch(() => null);
  if (!d || d.error) return;
  D.items = keepUnsaved(d.items, D.items);
  if (D.items.some(x => x.dirty)) { clearTimeout(D.timer); D.timer = setTimeout(saveNow, 700); }
  D.trigger = d.trigger;
  // 資料集設定與訓練紀錄也重讀：訓練完回到同一個資料集，紀錄才看得到（使用者回報）
  D.info = await get(`/api/dataset/info?dir=${enc(D.dir)}`).catch(() => D.info);
  const i = keep ? D.items.findIndex(x => x.path === keep) : -1;
  D.idx = i >= 0 ? i : Math.min(D.idx, Math.max(0, D.items.length - 1));
  const paths = new Set(D.items.map(x => x.path));
  D.sel = new Set([...D.sel].filter(p => paths.has(p)));
}

function say(msg, kind = "") {
  D.msg = msg;
  const el = $("dsMsg");
  if (el) { el.textContent = msg; el.dataset.kind = kind; }
}

// ── 左欄 ──────────────────────────────────────────
function drawPanel() {
  const box = $("dsPanel");
  if (!box) return;
  if (D.ui.dsfold) { drawFolded(box); return; }
  const [howBtn, howHint] = info("ds.howTo");
  box.replaceChildren(...[
    h("div.rhd", {}, h("h2.title", {id: "dsTitle", "data-i18n": "nav.dataset"}), howBtn, h("span.sp"),
      h("button.mini", {type: "button", "data-i18n-aria": "ds.fold", "data-i18n-title": "ds.fold", on: {click: () => fold(true)}}, "‹")),
    howHint,
    datasetHead(),
    D.dir ? triggerSection() : null,
    D.dir ? settingsSection() : null,
    modelSection(),
    D.dir && D.opts ? thresholdSection() : null,
    D.dir && D.opts ? rulesSection() : null,
    D.dir && D.opts ? descSection() : null,
    D.dir ? runSection() : null,
  ].filter(Boolean));
  translate(box);
}

// 收起來的左欄：展開、打標籤（或取消）、進度
function drawFolded(box) {
  const j = D.job;
  const running = j?.running && j.dir === D.dir;
  box.replaceChildren(...[
    h("button.mini", {type: "button", "data-i18n-aria": "ds.unfold", "data-i18n-title": "ds.unfold", on: {click: () => fold(false)}}, "›"),
    h("h2.title.dsvtitle", {id: "dsTitle", "data-i18n": "nav.dataset"}),
    D.dir ? (running
      ? h("button.mini", {type: "button", "data-i18n": "cf.cancel", on: {click: () => post("/api/dataset/cancel", {}).then(pollJob)}})
      : h("button.mini.tog", {type: "button", "aria-pressed": "true", "data-i18n": "ds.run", disabled: !D.model?.dir || !D.items.length,
                              on: {click: () => runTag([])}})) : null,
    D.dir && !running ? h("button.mini.tog", {type: "button", "aria-pressed": "true", "data-i18n": "ds.runDesc",
                                              disabled: !jcReady() || !D.items.length, on: {click: () => runDesc([])}}) : null,
    running ? h("span.mono.small", {text: `${Math.floor(j.done / Math.max(1, j.total) * 100)}%`}) : null,
  ].filter(Boolean));
  translate(box);
}

function fold(on) {
  saveUi({dsfold: on});
  const m = document.querySelector(".dsm");
  if (m) m.toggleAttribute("data-fold", on);
  drawPanel();
}

// 目前的資料集：名字、位置、張數；換資料集回 01 資料集的清單
function datasetHead() {
  if (!D.dir) return h("div.rfsec", {}, h("p.hint", {"data-i18n": "ds.pickFirst"}),
    h("a.pill", {href: "#datasets", "data-i18n": "ds.toList"}));
  return h("div.rfsec.dshead1", {},
    h("div.rflab", {}, h("b.dsdsname", {text: D.dir.split(/[\\/]/).pop(), title: D.dir}), h("span.sp"),
      h("a.mini.pill", {href: "#datasets", "data-i18n": "ds.toList"})),
    h("span.small.dim.dspath", {text: D.dir}),
    h("p.hint", {text: t("ds.count", {n: D.items.length, tagged: D.items.filter(i => i.has_txt).length})}));
}

// 這個資料集自己的設定：每張重複幾次（訓練頁選這個資料集時帶入）、備註、訓練紀錄（訓練完自動記）
function settingsSection() {
  const inf = D.info || {};
  const upd = async patch => {
    const d = await post("/api/dataset/library", {op: "update", dir: D.dir, ...patch}).catch(e => ({error: e.message}));
    if (d.error) say(d.error, "err");
    else D.info = {...inf, ...d};
  };
  const runs = inf.runs || [];
  return h("div.rfsec", {},
    h("label.dsfield", {}, h("span", {"data-i18n": "ds.repeats"}),
      h("input.find", {type: "number", min: "1", max: "100", value: inf.repeats ?? "", "data-i18n-placeholder": "ds.repeatsPh",
                       on: {change: e => upd({repeats: e.target.value === "" ? null : Number(e.target.value)})}})),
    h("label.dsfield", {}, h("span", {"data-i18n": "ds.note"}),
      h("textarea.find.trprompts", {rows: "2", value: inf.note || "", spellcheck: "false", "data-i18n-placeholder": "ds.notePh",
                                    on: {change: e => upd({note: e.target.value})}})),
    runs.length ? h("details.dsruns", {},
      h("summary.small", {text: t("ds.runsTitle", {n: runs.length})}),
      ...runs.map(r => h("div.small.dsrun", {title: `${r.dir || ""}\\${r.file || ""}`},
        h("b", {text: r.name}), h("span.dim", {text: ` · ${r.date || ""} · ${r.model || ""}`}),
        h("span.dim", {text: ` · ${r.steps || 0} ${t("ds.steps")}`}),
        r.stopped ? h("span.warn", {text: ` · ${t("ds.stopped")}`}) : null,
        // 存了那次的整套設定（之前的紀錄沒有）：帶到 03 訓練再訓練一次
        r.cfg ? h("button.mini", {type: "button", "data-i18n": "ds.useCfg", on: {click: () => useTrainConfig(r.cfg)}}) : null))) : null);
}

function modelSection() {
  const m = D.model;
  if (!m) return null;
  if (m.dir) return h("p.hint.dsmodel", {title: m.dir, text: t("ds.modelIn", {name: m.dir.split(/[\\/]/).pop()})});
  const dl = m.download;
  const pct = dl?.running ? Math.floor(dl.done / Math.max(1, dl.total) * 100) : 0;
  const rest = m.part ? t("dl.sureRest", {size: size(Math.max(0, m.bytes - m.part))}) : t("dl.sure", {size: size(m.bytes)});
  return h("div.rfsec.xmodel", {},
    h("p", {"data-i18n": "ds.needModel"}),
    h("p.hint", {}, `${m.name} · ${size(m.bytes)} · `, h("a", {href: m.page, target: "_blank", rel: "noopener", text: m.license})),
    dl?.running ? h("div.xbar", {}, h("i", {style: `width:${pct}%`}), h("span.mono", {text: `${pct}%`})) : null,
    dl?.error ? h("p.hint.warn", {text: dl.error})
      : m.part && !dl?.running ? h("p.hint", {text: t("dl.partial", {pct: partPct(m.part, m.bytes)})}) : null,
    h("div.rflab", {},
      dl?.running
        ? h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: () => post("/api/dataset/model/cancel", {}).then(refreshModel)}})
        : h("button.pill.primary", {type: "button", text: D.dlArm ? rest : t("dl.download"), on: {click: async () => {
          if (!D.dlArm) { D.dlArm = true; drawPanel(); return; }
          D.dlArm = false;
          await post("/api/dataset/model/download", {}).catch(() => null);
          refreshModel();
        }}}),
      h("button.pill", {type: "button", "data-i18n": "ds.modelPick", on: {click: pickModel}})),
    h("p.hint", {text: t("ds.modelTo", {dir: m.home})}));
}

// 自己已經有的 WD14：指定資料夾（裡面要有 model.onnx 或 model.safetensors，加上 selected_tags.csv）
async function pickModel() {
  const got = await pickFolder(D.model?.own || D.model?.dir || "");
  if (!got) return;
  await post("/api/settings", {TAGGER_DIR: got}).catch(() => null);
  await refreshModel();
  if (!D.model?.dir) say(t("ds.modelBad"), "err");
}

async function refreshModel() {
  clearTimeout(D.mpoll);
  const d = await get("/api/dataset/opts").catch(() => null);
  if (d) D.model = d.model;
  drawPanel();
  if (D.model?.download?.running) D.mpoll = setTimeout(refreshModel, 1000);
}

function triggerSection() {
  const [btn, hint] = info("ds.triggerHint");
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ds.trigger"}), btn),
    hint,
    h("input.find", {type: "text", value: D.trigger || "", spellcheck: "false", "data-i18n-placeholder": "ds.triggerPh",
                     "data-i18n-aria": "ds.trigger",
                     on: {change: async e => {
                       const d = await post("/api/dataset/trigger", {dir: D.dir, trigger: e.target.value}).catch(() => null);
                       if (d && !d.error) D.trigger = d.trigger;
                     }}}));
}

function optInput(key, attrs = {}) {
  const v = D.opts[key];
  if (OPT_BOOL.includes(key)) {
    return h("label.dschk", {}, h("input", {type: "checkbox", checked: v, on: {change: e => setOpt(key, e.target.checked)}}),
      h("span", {"data-i18n": `ds.opt.${key}`}));
  }
  return h("label.dsfield", {}, h("span", {"data-i18n": `ds.opt.${key}`}),
    h("input.find", {type: OPT_NUM.includes(key) ? "number" : "text", value: String(v), spellcheck: "false", ...attrs,
                     on: {change: e => setOpt(key, OPT_NUM.includes(key) ? Number(e.target.value) : e.target.value)}}));
}

async function setOpt(key, value) {
  const d = await post("/api/dataset/opts", {opts: {[key]: value}}).catch(() => null);
  if (d?.opts) D.opts = d.opts;
  if (["keep_character", "general", "nl_length"].includes(key)) drawPanel();
}

function thresholdSection() {
  const [btn, hint] = info("ds.thresholdHint");
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ds.threshold"}), btn),
    hint,
    h("div.dsgrid2", {}, optInput("general", {min: "0", max: "1", step: "0.05"}),
      D.opts.keep_character ? optInput("character", {min: "0", max: "1", step: "0.05"}) : null),
    optInput("keep_character"));
}

function rulesSection() {
  return h("details.rfsec.dsrules", {},
    h("summary", {"data-i18n": "ds.rules"}),
    h("div.dsgrid2", {}, optInput("remove_prefixes"), optInput("strip_labels")),
    optInput("separator"),
    optInput("underscore"), optInput("escape"), optInput("dedupe"));
}

// 寫描述：模型（JoyCaption 的 GGUF，在設定裡指定）、長度、把標籤當參考、提示詞
function descSection() {
  const jc = D.model?.jc || {};
  const [btn, hint] = info("ds.promptHint");
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ds.descSec"})),
    jc.main && jc.mmproj
      ? h("p.hint.dsmodel", {title: `${jc.main}\n${jc.mmproj}`, text: t("ds.jcIn", {name: jc.main.split(/[\\/]/).pop()})})
      : h("p.hint", {text: jc.error || t("ds.needJc")}),
    h("div.rflab", {}, h("span.small", {"data-i18n": "ds.length"}), h("span.sp"),
      h("div.seg", {role: "group", "data-i18n-aria": "ds.length"}, ["short", "medium", "long"].map(k => h("button", {
        type: "button", "aria-pressed": String(D.opts.nl_length === k), "data-i18n": `ds.len.${k}`,
        on: {click: () => setOpt("nl_length", k)}})))),
    optInput("use_tags"),
    h("details.dsrules", {},
      h("summary", {}, t("ds.prompt"), " ", btn),
      hint,
      h("textarea.rftext.dsprompt", {rows: "5", spellcheck: "false", "data-i18n-aria": "ds.prompt", value: D.opts.nl_prompt,
                                     on: {change: e => setOpt("nl_prompt", e.target.value)}})));
}

function runSection() {
  const j = D.job;
  const running = j?.running && j.dir === D.dir;
  const pct = running ? Math.floor(j.done / Math.max(1, j.total) * 100) : 0;
  const log = (j?.dir === D.dir ? j.log : []) || [];
  return h("div.rfsec", {},
    optInput("overwrite"),
    h("div.rflab", {},
      running
        ? h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: () => post("/api/dataset/cancel", {}).then(pollJob)}})
        : h("button.pill.primary", {type: "button", "data-i18n": "ds.run", disabled: !D.model?.dir || !D.items.length,
                                    on: {click: () => runTag([])}}),
      running ? null : h("button.pill.primary", {type: "button", "data-i18n": "ds.runDesc", disabled: !jcReady() || !D.items.length,
                                                 on: {click: () => runDesc([])}}),
      h("span.dim.small", {text: running ? `${t(j.kind === "desc" ? "ds.runDesc" : "ds.run")} ${j.done} / ${j.total}` : ""})),
    running ? h("div.xbar", {}, h("i", {style: `width:${pct}%`})) : null,
    log.length ? h("div.dslog", {}, log.slice(-6).map(x => h("div", {"data-kind": x.k, text: `${x.t}  ${x.m}`}))) : null);
}

const jcReady = () => !!(D.model?.jc?.main && D.model?.jc?.mmproj);

async function runDesc(targets) {
  await saveNow();
  const d = await post("/api/dataset/caption", {dir: D.dir, targets}).catch(e => ({error: e.message}));
  if (d.error) { say(d.error, "err"); return; }
  say("");
  pollJob();
}

async function runTag(targets) {
  await saveNow();
  const d = await post("/api/dataset/tag", {dir: D.dir, targets}).catch(e => ({error: e.message}));
  if (d.error) { say(d.error, "err"); return; }
  say("");
  pollJob();
}

async function pollJob() {
  clearTimeout(D.poll);
  const was = D.job?.running;
  const j = await get("/api/dataset/state").catch(() => null);
  if (!j || !$("dsPanel")) return;
  D.job = j;
  if (j.running) D.poll = setTimeout(pollJob, 700);
  if (was && !j.running && j.dir === D.dir) {           // 打完了：重新讀標籤
    await reload();
    D.stats = null;
    drawStage();
  }
  if (was || j.running) drawPanel();
}

// ── 右邊 ──────────────────────────────────────────
function drawStage() {
  const box = $("dsStage");
  if (!box) return;
  if (!D.dir) {
    box.replaceChildren(h("p.hint.dsempty", {"data-i18n": "ds.noFolder"}));
    translate(box);
    return;
  }
  const tabs = h("div.seg", {role: "tablist"}, ["review", "stats"].map(v => h("button", {
    type: "button", role: "tab", "aria-selected": String(D.view === v), "aria-pressed": String(D.view === v), "data-i18n": `ds.view.${v}`,
    on: {click: async () => { await saveNow(); D.view = v; drawStage(); }}})));
  const head = h("div.rhd.dshead", {}, tabs, h("span.sp"), h("span.msg", {id: "dsMsg", role: "status", text: D.msg}));
  box.replaceChildren(head, D.view === "stats" ? statsView() : reviewView());
  translate(box);
  if (D.view === "stats" && !D.stats) loadStats();
  if (D.view === "review") D.chips?.refresh();
}

// 審核：大圖＋編輯在上，縮圖格在下
function reviewView() {
  if (!D.items.length) return h("p.hint.dsempty", {"data-i18n": "ds.noImages"});
  const it = cur();
  D.chips = new ChipBox({read: () => cur()?.tags || "", write: text => { cur().tags = text; dirty("tags"); }, label: t("ds.tags")});
  const desc = h("textarea.rftext.dsdesc", {rows: "3", spellcheck: "false", value: it.desc || "", "data-i18n-placeholder": "ds.descPh",
                                            "data-i18n-aria": "ds.desc",
                                            on: {input: e => { cur().desc = e.target.value; dirty("desc"); }}});
  desc.value = it.desc || "";
  const auto = !!D.ui.dstrans;
  const big = h("div.dsbig", {"data-zoom": D.zoom ? "1" : null, "data-i18n-title": "ds.zoom",
                              on: {click: e => toggleZoom(e, big)}}, h("img", {src: fullUrl(it), alt: it.name}));
  const edit = h("div.dsedit", {},
      h("div.rflab", {},
        h("button.mini", {type: "button", "data-i18n-aria": "ds.prev", on: {click: () => jump(-1)}}, "←"),
        h("b.dsname", {text: it.rel, title: it.path}),
        h("span.dim.small", {text: `${D.idx + 1} / ${D.items.length}`}),
        h("button.mini", {type: "button", "data-i18n-aria": "ds.next", on: {click: () => jump(1)}}, "→"),
        h("span.sp"), h("span.dim.small.dssaved", {id: "dsSaved"}), coverBtn(it)),
      h("div.rflab", {}, h("b", {"data-i18n": "ds.tags"})),
      h("div.dschips", {}, D.chips.el),
      h("div.rflab", {}, h("b", {"data-i18n": "ds.desc"}), h("span.sp"),
        h("label.dschk", {}, h("input", {type: "checkbox", checked: auto, on: {change: e => { saveUi({dstrans: e.target.checked}); if (e.target.checked) doTrans(); }}}),
          h("span", {"data-i18n": "ds.autoTrans"})),
        h("button.mini", {type: "button", "data-i18n": "ds.translate", on: {click: doTrans}})),
      desc,
      h("div.dstrans", {id: "dsTrans"}, D.trans && D.trans.path === it.path ? transView(D.trans) : null),
      h("p.hint", {"data-i18n": "ds.keys"}));
  if (auto && it.desc && D.trans?.path !== it.path) setTimeout(doTrans, 0);
  return h("div.dsreview", {}, big, h("div.dsside", {}, edit, selBar(), grid()));
}

// 這張是不是封面（沒選封面時第一張就是）；按了設成封面，01 資料集的卡片換成這張
function coverBtn(it) {
  const isCover = D.info?.cover ? D.info.cover.toLowerCase() === it.path.toLowerCase() : D.idx === 0;
  if (isCover) return h("span.small.dim", {"data-i18n": "ds.isCover"});
  return h("button.mini", {type: "button", "data-i18n": "ds.setCover", on: {click: async () => {
    const d = await post("/api/dataset/library", {op: "update", dir: D.dir, cover: it.path}).catch(e => ({error: e.message}));
    if (d.error) { say(d.error, "err"); return; }
    D.info = {...(D.info || {}), cover: d.cover};
    drawStage();
  }}});
}

// 點圖：切換原始大小，放大時捲到點的那個位置。
// 圖是 contain 塞在整格裡（img 的範圍＝整格），點的位置要扣掉兩邊空白才是圖上的比例
function toggleZoom(e, box) {
  const img = box.querySelector("img");
  const r = box.getBoundingClientRect();
  const bw = box.clientWidth, bh = box.clientHeight, nw = img.naturalWidth || 1, nh = img.naturalHeight || 1;
  const s = Math.min(bw / nw, bh / nh), dw = nw * s, dh = nh * s;
  const fx = Math.min(1, Math.max(0, ((e.clientX - r.left) / r.width * bw - (bw - dw) / 2) / dw));
  const fy = Math.min(1, Math.max(0, ((e.clientY - r.top) / r.height * bh - (bh - dh) / 2) / dh));
  D.zoom = !D.zoom;
  box.toggleAttribute("data-zoom", D.zoom);
  if (D.zoom) {
    box.scrollLeft = fx * img.offsetWidth - box.clientWidth / 2;
    box.scrollTop = fy * img.offsetHeight - box.clientHeight / 2;
  }
}

function selBar() {
  const n = D.sel.size;
  const pick = fn => () => { D.sel = new Set(D.items.filter(fn).map(i => i.path)); drawStage(); };
  return h("div.dsselbar", {},
    h("span.small", {text: t("ds.selected", {n, total: D.items.length})}),
    h("button.mini", {type: "button", "data-i18n": "ds.selAll", on: {click: pick(() => true)}}),
    h("button.mini", {type: "button", "data-i18n": "ds.selNone", on: {click: pick(() => false)}}),
    h("button.mini", {type: "button", "data-i18n": "ds.selInvert", on: {click: pick(i => !D.sel.has(i.path))}}),
    h("button.mini", {type: "button", "data-i18n": "ds.selUntagged", on: {click: pick(i => !i.has_txt)}}),
    h("span.sp"),
    h("button.mini", {type: "button", "data-i18n": "ds.redesc", disabled: !n || !jcReady() || D.job?.running,
                      on: {click: () => runDesc([...D.sel])}}),
    h("button.mini", {type: "button", "data-i18n": "ds.retag", disabled: !n || !D.model?.dir || D.job?.running,
                      on: {click: () => runTag([...D.sel])}}),
    h("button.mini.dsdanger", {type: "button", disabled: !n, text: D.trashArm ? t("ds.trashSure", {n}) : t("ds.trash"),
                               on: {click: trashSel}}));
}

async function trashSel() {
  if (!D.trashArm) { D.trashArm = true; drawStage(); setTimeout(() => { D.trashArm = false; }, 4000); return; }
  D.trashArm = false;
  await saveNow();
  const d = await post("/api/dataset/trash", {dir: D.dir, targets: [...D.sel]}).catch(e => ({error: e.message}));
  if (d.error) { say(d.error, "err"); return; }
  D.sel = new Set();
  await reload();
  D.stats = null;
  say(t("ds.trashed", {n: d.moved}), "ok");
  drawPanel();
  drawStage();
}

function grid() {
  return h("div.dscells", {role: "listbox", "aria-multiselectable": "true", "data-i18n-aria": "ds.grid"},
    D.items.map((it, i) => h("div.dscell", {
      role: "option", "aria-selected": String(D.sel.has(it.path)), "data-cur": i === D.idx ? "1" : null, title: it.rel,
      on: {click: e => cellClick(e, i)}},
    h("img", {src: thumbUrl(it), alt: "", loading: "lazy"}),
    h("span.ck", {text: "✓"}),
    it.has_txt ? null : h("span.dsnotag", {"data-i18n-title": "ds.untagged"}))));
}

function cellClick(e, i) {
  const p = D.items[i].path;
  if (e.shiftKey && D.last >= 0) {
    const [a, b] = [Math.min(D.last, i), Math.max(D.last, i)];
    for (let k = a; k <= b; k++) D.sel.add(D.items[k].path);
  } else if (e.ctrlKey || e.metaKey || e.target.closest(".ck")) {
    if (D.sel.has(p)) D.sel.delete(p); else D.sel.add(p);
    D.last = i;
  } else {
    D.last = i;
    go(i);
    return;
  }
  drawStage();
}

// 標籤或描述改了：停 0.7 秒沒再改就存。dirty 記改了哪幾個欄位（重新讀的時候只蓋回這幾個，見 reloadOnce）
function dirty(field) {
  const it = cur();
  if (it) it.dirty = {...(it.dirty || {}), [field]: true};
  const s = $("dsSaved");
  if (s) s.textContent = "";
  clearTimeout(D.timer);
  D.timer = setTimeout(saveNow, 700);
}

function saveNow() {
  clearTimeout(D.timer);
  return io(saveOnce);
}

// 還沒存的全部存：之前存失敗、又換到下一張改了，就不只一筆（以前只存第一筆）。回傳存不進去的原因，全部存好是 ""
async function saveOnce() {
  let err = "";
  for (const it of D.items.filter(x => x.dirty)) {
    const was = it.dirty;
    it.dirty = false;
    const d = await post("/api/dataset/save", {dir: D.dir, path: it.path, tags: it.tags, desc: it.desc}).catch(e => ({error: e.message}));
    if (d.error) { it.dirty = {...was, ...(it.dirty || {})}; say(d.error, "err"); err = d.error; continue; }
    it.text = d.text;
    if (!it.has_txt) {
      it.has_txt = true;
      const cell = document.querySelectorAll(".dscell")[D.items.indexOf(it)];
      cell?.querySelector(".dsnotag")?.remove();
    }
    D.stats = null;
    const s = $("dsSaved");
    if (s && it === cur()) s.textContent = t("ds.saved");
  }
  return err;
}

async function go(i) {
  if (!D.items.length) return;
  await saveNow();
  D.idx = (i + D.items.length) % D.items.length;
  D.zoom = false;
  drawStage();
  document.querySelector(".dscell[data-cur]")?.scrollIntoView({block: "nearest"});
}
const jump = d => go(D.idx + d);

// 描述的中文：按了才送 Google 翻譯（或打開換圖自動翻）
async function doTrans() {
  const it = cur();
  if (!it?.desc?.trim()) { D.trans = null; showTrans(); return; }
  D.trans = {path: it.path, busy: true};
  showTrans();
  const d = await post("/api/dataset/translate", {segments: [it.desc], target: "zh-TW"}).catch(e => ({error: e.message}));
  if (cur()?.path !== it.path) return;
  D.trans = {path: it.path, text: d.error ? "" : (d.lines || []).join(" "), error: d.error || ""};
  showTrans();
}

function showTrans() {
  const box = $("dsTrans");
  if (box) box.replaceChildren(...(D.trans && D.trans.path === cur()?.path ? [transView(D.trans)] : []));
}

const transView = tr => tr.busy ? h("p.hint", {text: t("ds.translating")})
  : tr.error ? h("p.hint.warn", {text: tr.error}) : h("p.dstranstext", {text: tr.text});

// 標籤統計
async function loadStats() {
  const d = await get(`/api/dataset/stats?dir=${enc(D.dir)}`).catch(e => ({error: e.message}));
  D.stats = d.error ? {error: d.error, tags: []} : d;
  if (D.view === "stats") drawStage();
}

function statsView() {
  const S = D.stats;
  if (!S) return h("p.hint", {"data-i18n": "dl.loading"});
  if (S.error) return h("p.hint.warn", {text: S.error});
  const add = h("input.find", {type: "text", spellcheck: "false", "data-i18n-placeholder": "ds.addPh", "data-i18n-aria": "ds.addPh"});
  const [btn, hint] = info("ds.statsHint");
  return h("div.dsstats", {},
    h("div.rflab", {}, h("span.small", {text: t("ds.statsCount", {n: S.tags.length, images: S.images})}), btn),
    hint,
    h("div.dsrow", {}, add, h("button.pill", {type: "button", "data-i18n": "ds.addAll", on: {click: () => {
      if (add.value.trim()) bulk("add", add.value.trim());
    }}})),
    h("div.dstags", {}, S.tags.map(row => statRow(row))));
}

function statRow(row) {
  const el = h("div.dstag", {});
  const plain = () => el.replaceChildren(...[
    h("span.dstagname", {text: row.tag}),
    row.trigger ? h("span.dim.small", {"data-i18n": "ds.isTrigger"}) : null,
    h("span.sp"),
    h("span.mono.small", {text: String(row.count)}),
    h("button.mini", {type: "button", "data-i18n": "ds.replace", on: {click: editing}}),
    h("button.mini.dsdanger", {type: "button", "data-i18n": "ds.remove", on: {click: e => {
      if (e.currentTarget.dataset.arm) { bulk("remove", row.tag); return; }
      e.currentTarget.dataset.arm = "1";
      e.currentTarget.textContent = t("ds.removeSure", {n: row.count});
    }}})].filter(Boolean));
  const editing = () => {
    const to = h("input.find", {type: "text", value: row.tag, spellcheck: "false", "data-i18n-aria": "ds.replaceTo",
                                on: {keydown: e => { if (e.key === "Enter") ok(); else if (e.key === "Escape") { plain(); translate(el); } }}});
    const ok = () => { if (to.value.trim() && to.value.trim() !== row.tag) bulk("replace", row.tag, to.value.trim()); };
    el.replaceChildren(h("span.dstagname", {text: row.tag}), h("span", {text: "→"}), to,
      h("button.mini", {type: "button", "data-i18n": "ds.ok", on: {click: ok}}),
      h("button.mini", {type: "button", "data-i18n": "cf.cancel", on: {click: () => { plain(); translate(el); }}}));
    translate(el);
    to.focus();
    to.select();
  };
  plain();
  return el;
}

async function bulk(op, tag, to = "") {
  await saveNow();
  const d = await post("/api/dataset/bulk", {dir: D.dir, op, tag, to}).catch(e => ({error: e.message}));
  if (d.error) { say(d.error, "err"); return; }
  say(t("ds.bulkDone", {n: d.changed}), "ok");
  await reload();
  D.stats = null;
  drawPanel();
  drawStage();
}

// ← → 換圖（在標籤或描述裡打字時不算）；Ctrl+Enter 存好跳下一張
document.addEventListener("keydown", e => {
  if (!$("dsStage") || D.view !== "review" || !D.items.length || e.defaultPrevented) return;
  if (document.querySelector(".fbrowse, .pop:not([hidden])")) return;
  const typing = e.target.closest?.("input, textarea, [contenteditable]");
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); jump(1); return; }
  if (typing) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); jump(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); jump(1); }
});
