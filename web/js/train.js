// 訓練空間的 03 訓練（server/train.py、core/sdxl_train.py、core/anima_train.py）：SDXL 與 Anima 的 LoRA。
// 最上面的「設定檔」：整套設定取名字存起來、從清單載入（server/train.py 的 save_config）；
// 02 打標的訓練紀錄按「用這組設定」也會帶著那次的設定過來（useTrainConfig）。
// 左欄：資料集、底模（SDXL、Anima 分兩組）、預設（SDXL：使用者 OneTrainer 的 200MB／400MB／一張圖；
// Anima：Prodigy／AdamW）、名稱與存放位置、
// 重複次數、epoch、rank／alpha、從第幾個 epoch 開始存、文字編碼器一起訓練、翻轉、打亂標籤；開始前估步數與時間。
// 最上面切換「LoRA／Slider」：Slider（core/slider_train.py）填中性、正向、反向、錨點四則提示詞，底圖自動畫或選資料集，
// 步數、rank／alpha、強度。
// 右邊（整欄可以捲）：進度、loss、學習率、速度、剩幾分鐘、試畫（每則提示詞一列，可以到比較組挑；Slider 是每存一次一排 −2～+2）、
// 存好的檔、紀錄。
// 每秒問一次進度。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {pickFolder} from "./setup.js";
import {openCompare} from "./compare.js";

// figs：試畫的縮圖元素（路徑 → figure），每秒重畫時沿用，圖才不會閃
const T = {ui: null, opts: null, cfg: null, st: null, poll: 0, prev: null, prevTimer: 0, msg: "", figs: new Map()};
const NUMS = [["repeats", 1, 100, 1], ["epochs", 1, 200, 1], ["rank", 1, 256, 1], ["alpha", 0.1, 256, 0.5], ["save_skip", 0, 200, 1]];
const PRESET_KEYS = {sdxl: ["200mb", "400mb", "single"], anima: ["prodigy", "adamw"]};
const SL_NUMS = [["sl_steps", 50, 5000, 50], ["sl_guidance", 0.5, 10, 0.5], ["sl_rank", 1, 128, 1], ["sl_alpha", 0.1, 128, 0.5]];
// 成對圖片：沒有「強度」（方向直接是兩張圖的差別）；步數另外記（學得快，預設 200，server/train.py 的 sl_pair_steps）
const SL_PAIR_NUMS = SL_NUMS.filter(([k]) => k !== "sl_guidance").map(x => x[0] === "sl_steps" ? ["sl_pair_steps", ...x.slice(1)] : x);
// 成對圖輔助（server/pairmaker.py）：原圖資料夾、Qwen 模型、做法、兩邊的修改指示、估計、進度
// 圖從哪來：folder＝資料夾裡的圖；auto＝先用 Slider 的底模照提示詞畫底圖再改（底模自己的畫風，不會漏資料集的畫風）
const PM = {open: false, src_mode: "auto", src: "", model: "", mode: "both", pos_text: "", neg_text: "",
            prompts: "", n_base: 12, name: "", root: "", est: null, st: null, timer: 0, poll: 0};
const PM_SRC = ["auto", "folder"];
const PM_MODES = ["both", "orig_neg", "orig_pos"];
const SL_MULTS = [-2, -1, 0, 1, 2];        // 試畫一排的強度（0 是開始前那張不掛 LoRA 的）
const ARCH_LABEL = {sdxl: "SDXL", anima: "Anima"};
const archOf = path => T.opts.models.find(m => m.path === path)?.arch || "sdxl";

// 換了不同架構的底模：套用那個架構的第一個預設（rank、epoch 這些兩邊差很多）
function pickModel(path) {
  const before = archOf(T.cfg.model);
  T.cfg.model = path;
  const arch = archOf(path);
  if (arch !== before) {
    const k = PRESET_KEYS[arch][0];
    Object.assign(T.cfg, T.opts.presets[arch][k], {preset: k});
  }
  set("model", path, true);
}

export function initTrain(ui) { T.ui = ui; }

export function trainScreen() {
  return h("div.rfm.trm", {},
    h("aside.card.rail.rfpanel.dspanel", {id: "trPanel", "aria-labelledby": "trTitle"}),
    h("section.card.rfstage.dsstage", {id: "trStage", "aria-live": "polite"}));
}

export async function showTrain() {
  if (!$("trPanel")) return;
  T.opts = await get("/api/train/options").catch(e => ({error: e.message}));
  if (T.opts.error) { $("trPanel").replaceChildren(h("p.hint.warn", {text: T.opts.error})); return; }
  T.st = T.opts.state;
  if (!T.cfg) {
    T.cfg = {...T.opts.last};
    T.cfg.dataset ||= T.ui.dsdir || T.opts.datasets[0] || "";
  }
  // 剛在 02 打標開的資料集：直接拿來訓練
  if (!T.st?.running && T.ui.dsdir && T.ui.dsdir !== T.dsSeen) {
    T.dsSeen = T.ui.dsdir;
    T.cfg.dataset = T.ui.dsdir;
  }
  // 02 打標的訓練紀錄按了「用這組設定」：整套換成那次的（資料集也是那次的，重複次數寫回資料集）；
  // 不然每張重複幾次照資料集現在的設定（使用者回報：在 02 改了，回 03 還是舊的）
  if (T.pending && !T.st?.running) {
    Object.assign(T.cfg, T.pending);
    T.cfgMsg = t("tr.cfgLoaded", {name: T.pending.name || ""});
    saveRepeats();
  } else if (!T.st?.running) {
    useRepeats();
  }
  T.pending = null;
  if (!T.cfg.model) T.cfg.model = T.opts.models[0]?.path || "";
  if (!T.cfg.out_root) T.cfg.out_root = T.opts.lora_root;
  drawPanel();
  drawStage();
  preview();
  if (T.st.running) poll();
}

// 02 打標的訓練紀錄：用那次的設定再訓練一次
export function useTrainConfig(cfg) {
  T.pending = {...cfg};
  location.hash = "#train";
}

// 每張重複幾次就是資料集的設定（02 打標的資料集設定）：資料集有設定就照它的，這裡改了寫回資料集，兩邊永遠同一個數字
function useRepeats() {
  const r = T.opts?.ds_repeats?.[T.cfg.dataset];
  if (r) T.cfg.repeats = r;
}

function saveRepeats() {
  const ds = T.cfg.dataset, r = Number(T.cfg.repeats);
  if (!ds || !r || !T.opts || T.opts.ds_repeats?.[ds] === r) return;
  (T.opts.ds_repeats ||= {})[ds] = r;
  post("/api/dataset/library", {op: "update", dir: ds, repeats: r}).catch(() => { /* 寫不回去只影響下次打開 */ });
}

// ── 左欄 ──
function field(label, input, hintKey) {
  const [btn, hint] = hintKey ? info(hintKey) : [null, null];
  return h("div.rfsec", {}, h("div.rflab", {}, h("b", {"data-i18n": label}), btn), hint, input);
}

function set(k, v, redraw = false) {
  T.cfg[k] = v;
  if (redraw) drawPanel();
  clearTimeout(T.prevTimer);
  T.prevTimer = setTimeout(preview, 300);
}

function drawPanel() {
  const box = $("trPanel");
  if (!box) return;
  const c = T.cfg, running = T.st?.running, arch = archOf(c.model), slider = c.kind === "slider";
  const [howBtn, howHint] = info(slider ? "tr.sl.howTo" : "tr.howTo");
  const kinds = h("div.seg", {id: "trKind", role: "group", "data-i18n-aria": "tr.kind"}, ["lora", "slider"].map(k => h("button", {
    type: "button", disabled: running, "aria-pressed": String((c.kind || "lora") === k), "data-i18n": `tr.kind.${k}`,
    on: {click: () => set("kind", k, true)}})));
  const datasets = [...new Set([c.dataset, ...T.opts.datasets].filter(Boolean))];
  const ds = h("select.find", {"data-i18n-aria": "tr.dataset", disabled: running,
                              on: {change: e => { T.cfg.dataset = e.target.value; useRepeats(); set("dataset", e.target.value, true); }}},
    datasets.map(d => h("option", {value: d, selected: d === c.dataset, text: d.split(/[\\/]/).pop()})));
  const model = h("select.find", {"data-i18n-aria": "tr.model", disabled: running, on: {change: e => pickModel(e.target.value)}},
    T.opts.models.length ? Object.keys(ARCH_LABEL).map(a => {
      const list = T.opts.models.filter(m => m.arch === a);
      return list.length ? h("optgroup", {label: ARCH_LABEL[a]},
        list.map(m => h("option", {value: m.path, selected: m.path === c.model, text: m.name}))) : null;
    }) : [h("option", {value: "", text: t("tr.noModel")})]);
  const presets = h("div.seg", {id: "trPresets", role: "group", "data-i18n-aria": "tr.preset"}, PRESET_KEYS[arch].map(k => h("button", {
    type: "button", disabled: running, "aria-pressed": String(c.preset === k), "data-i18n": `tr.preset.${k}`,
    on: {click: () => { Object.assign(c, T.opts.presets[arch][k], {preset: k}); drawPanel(); clearTimeout(T.prevTimer); preview(); }}})));
  const numGrid = (list, after) => h("div.dsgrid2", {}, list.map(([k, lo, hi, step]) => h("label.dsfield", {},
    h("span", {"data-i18n": `tr.${k}`}),
    h("input.find", {type: "number", min: String(lo), max: String(hi), step: String(step), value: String(c[k] ?? ""), disabled: running,
                     on: {change: e => { set(k, Number(e.target.value)); if (k === "repeats") saveRepeats(); if (after) after(); }}}))));
  const chk = (k, lab) => h("label.dschk", {}, h("input", {type: "checkbox", checked: !!c[k], disabled: running,
                                                              on: {change: e => set(k, e.target.checked, k === "shuffle_tags" || k === "sample")}}),
    h("span", {"data-i18n": lab}));
  const chkHint = (k, lab, hintKey) => {
    const [btn, hint] = info(hintKey);
    return h("div", {}, h("div.rflab", {}, chk(k, lab), btn), hint);
  };
  const text = (k, rows) => h(rows ? "textarea.find.trprompts" : "input.find", {
    ...(rows ? {rows: String(rows)} : {type: "text"}), value: c[k] || "", spellcheck: "false", disabled: running,
    "data-i18n-aria": `tr.${k}`, on: {input: e => set(k, e.target.value)}});
  const nameBlock = h("div.rfsec", {}, h("div.rflab", {}, h("b", {"data-i18n": "tr.name"})),
    h("input.find", {type: "text", value: c.name || "", spellcheck: "false", "data-i18n-placeholder": "tr.namePh", disabled: running,
                     "data-i18n-aria": "tr.name", on: {input: e => set("name", e.target.value)}}),
    h("div.dsrow", {}, h("input.find", {type: "text", value: c.out_root || "", spellcheck: "false", "data-i18n-aria": "tr.outRoot",
                                        disabled: running, on: {change: e => set("out_root", e.target.value.trim())}}),
      h("button.pill", {type: "button", "data-i18n": "set.browse", disabled: running, on: {click: async () => {
        const got = await pickFolder(c.out_root || T.opts.lora_root);
        if (got) set("out_root", got, true);
      }}})));
  // Slider 的底圖：自動畫（照中性提示詞）或資料集
  const base = h("select.find", {"data-i18n-aria": "tr.sl_dataset", disabled: running, on: {change: e => set("sl_dataset", e.target.value, true)}},
    h("option", {value: "", selected: !c.sl_dataset, text: t("tr.sl.auto")}),
    [...new Set([c.sl_dataset, ...T.opts.datasets].filter(Boolean))].map(d =>
      h("option", {value: d, selected: d === c.sl_dataset, text: d.split(/[\\/]/).pop()})));
  // 試畫的提示詞：一則一格（使用者回饋：一格多行看不出來可以寫好幾則）；存的時候還是一行一則
  const promptBoxes = () => {
    const lines = String(c.sample_prompts || "").split("\n");
    const boxes = Array.from({length: T.opts.max_samples || 4}, (_, i) => h("input.find", {
      type: "text", id: i ? null : "trPrompts", value: lines[i] || "", spellcheck: "false", disabled: running,
      placeholder: i ? t("tr.promptMore", {n: i + 1}) : T.prev?.default_prompt || "", "aria-label": t("tr.promptN", {n: i + 1}),
      on: {input: () => set("sample_prompts", boxes.map(b => b.value.replace(/[\r\n]+/g, " ")).join("\n"))}}));
    return h("div.trpromptlist", {}, boxes);
  };
  // Slider 的中性提示詞：一則一格（長的提示詞也放得下，Enter 不換行）；存成一行一則
  const neutralBoxes = () => {
    const lines = String(c.sl_neutral || "").split("\n");
    const boxes = Array.from({length: T.opts.max_neutral || 4}, (_, i) => h("textarea.find.trprompts", {
      rows: "2", value: lines[i] || "", spellcheck: "false", disabled: running,
      placeholder: i ? t("tr.sl.neutralMore", {n: i + 1}) : "1girl, solo", "aria-label": t("tr.promptN", {n: i + 1}),
      on: {keydown: e => { if (e.key === "Enter") e.preventDefault(); },
           input: () => set("sl_neutral", boxes.map(b => b.value.replace(/[\r\n]+/g, " ")).join("\n"))}}));
    return h("div.trpromptlist", {}, boxes);
  };
  const head =[h("div.rhd", {}, h("h2.title", {id: "trTitle", "data-i18n": "nav.train"}), howBtn), howHint, kinds, savedRow(running),
               progressRow(running)];
  const pairs = c.sl_mode === "pairs";
  // Slider 的兩種教法：用詞（底模對正向、反向詞的理解）、用成對圖片（同一張圖的正向版與反向版）
  const modeSeg = h("div.seg", {id: "trSlMode", role: "group", "data-i18n-aria": "tr.sl_mode"}, ["words", "pairs"].map(k => h("button", {
    type: "button", disabled: running, "aria-pressed": String((c.sl_mode || "words") === k), "data-i18n": `tr.sl.mode.${k}`,
    on: {click: () => set("sl_mode", k, true)}})));
  const dirSel = k => h("select.find", {"data-i18n-aria": `tr.${k}`, disabled: running, on: {change: e => set(k, e.target.value, true)}},
    h("option", {value: "", selected: !c[k], text: t("tr.sl.pickDir")}),
    [...new Set([c[k], ...T.opts.datasets].filter(Boolean))].map(d =>
      h("option", {value: d, selected: d === c[k], text: d.split(/[\\/]/).slice(-2).join("\\")})));      // 正向、反向的資料夾名一樣：帶上一層
  const body = slider && pairs ? [
    field("tr.model", model, "tr.modelHint"),
    field("tr.sl_mode", modeSeg, "tr.sl_modeHint"),
    field("tr.sl_pos_dir", dirSel("sl_pos_dir")),
    field("tr.sl_neg_dir", dirSel("sl_neg_dir"), "tr.sl_pairsHint"),
    pairMaker(running),
    field("tr.sl_pair_prompt", text("sl_pair_prompt", 2), "tr.sl_pair_promptHint"),
    nameBlock,
    numGrid(SL_PAIR_NUMS),
    chkHint("sl_quality", "tr.sl_quality", "tr.sl_qualityHint"),
    chkHint("sl_sfw", "tr.sl_sfw", "tr.sl_sfwHint"),
  ] : slider ? [
    field("tr.model", model, "tr.modelHint"),
    field("tr.sl_mode", modeSeg, "tr.sl_modeHint"),
    field("tr.sl_neutral", neutralBoxes(), "tr.sl_neutralHint"),
    field("tr.sl_positive", text("sl_positive", 2), "tr.sl_positiveHint"),
    field("tr.sl_negative", text("sl_negative", 2)),
    field("tr.sl_anchor", text("sl_anchor"), "tr.sl_anchorHint"),
    nameBlock,
    field("tr.sl_dataset", base, "tr.sl_datasetHint"),
    numGrid(c.sl_dataset ? SL_NUMS : [...SL_NUMS, ["sl_n_base", 4, 40, 1]]),
    // 底圖與試畫要不要另外加標籤：使用者自己決定（預設勾，看得到、可以取消）
    chkHint("sl_quality", "tr.sl_quality", "tr.sl_qualityHint"),
    chkHint("sl_sfw", "tr.sl_sfw", "tr.sl_sfwHint"),
  ] : [
    field("tr.dataset", ds),
    field("tr.model", model, "tr.modelHint"),
    field("tr.preset", presets, arch === "anima" ? "tr.presetHintAnima" : "tr.presetHint"),
    nameBlock,
    numGrid(NUMS, () => { c.preset = ""; drawPresetState(); }),
    // Anima 照使用者的設定只訓練 DiT、不翻轉、不打亂標籤（標籤的條件先算好）
    arch === "sdxl" ? chk("train_te", "tr.train_te") : null,
    arch === "sdxl" ? chk("flip", "tr.flip") : null,
    arch === "sdxl" ? chk("shuffle_tags", "tr.shuffle_tags") : null,
    arch === "sdxl" && c.shuffle_tags ? h("label.dsfield", {}, h("span", {"data-i18n": "tr.keep_tags"}),
      h("input.find", {type: "number", min: "0", max: "20", value: String(c.keep_tags ?? 1), disabled: running,
                       on: {change: e => set("keep_tags", Number(e.target.value))}})) : null,
  ];
  box.replaceChildren(...[
    ...head, ...body,
    chk("sample", "tr.sample"),
    c.sample ? field("tr.samplePrompts", promptBoxes(), slider ? "tr.sl.sampleHint" : "tr.samplePromptsHint") : null,
    chkHint("keep_state", "tr.keepState", "tr.keepStateHint"),
    h("div.rfsec", {}, h("p.hint", {id: "trPreview"}),
      h("div.rflab", {},
        running ? h("button.pill", {type: "button", "data-i18n": "tr.stop", on: {click: stop}})
          : h("button.pill.primary", {type: "button", "data-i18n": "tr.start", on: {click: start}}),
        h("span.msg", {id: "trMsg", role: "status", text: T.msg}))),
  ].filter(Boolean));
  translate(box);
  showPreview();
}

// 設定檔：自己取名字存起來、要的時候載入（「上次的設定」只有一份，按開始訓練時自動記）
function savedRow(running) {
  const list = T.opts.saved || [];
  const cur = list.some(x => x.name === T.cfgName) ? T.cfgName : "";
  const pick = h("select.find", {"data-i18n-aria": "tr.cfgLoad", disabled: running || !list.length, on: {change: e => {
    const it = list.find(x => x.name === e.target.value);
    if (!it) return;
    Object.assign(T.cfg, it.cfg);
    saveRepeats();
    Object.assign(T, {cfgName: it.name, delArm: false, cfgMsg: t("tr.cfgLoaded", {name: it.name})});
    drawPanel();
    clearTimeout(T.prevTimer);
    preview();
  }}},
    h("option", {value: "", selected: !cur, text: t(list.length ? "tr.cfgLoad" : "tr.cfgNone")}),
    list.map(x => h("option", {value: x.name, selected: x.name === cur, text: `${x.name} · ${x.date || ""}`})));
  const del = cur ? h("button.pill", {type: "button", disabled: running, text: t(T.delArm ? "tr.cfgDeleteSure" : "tr.cfgDelete"),
    on: {click: async () => {
      if (!T.delArm) { T.delArm = true; drawPanel(); return; }
      const d = await post("/api/train/saved", {op: "delete", name: cur}).catch(e => ({error: e.message}));
      Object.assign(T, {delArm: false, cfgName: "", cfgMsg: d.error || ""});
      if (d.saved) T.opts.saved = d.saved;
      drawPanel();
    }}}) : null;
  const name = h("input.find", {type: "text", value: cur || T.cfg.name || "", spellcheck: "false", "data-i18n-placeholder": "tr.cfgNamePh",
                                "data-i18n-aria": "tr.cfgNamePh"});
  const save = async () => {
    const d = await post("/api/train/saved", {op: "save", name: name.value, cfg: T.cfg}).catch(e => ({error: e.message}));
    if (d.error) { T.cfgMsg = d.error; drawPanel(); return; }
    Object.assign(T, {saving: false, cfgName: name.value.trim(), cfgMsg: t("tr.cfgSaved", {name: name.value.trim()})});
    T.opts.saved = d.saved;
    drawPanel();
  };
  name.addEventListener("keydown", e => { if (e.key === "Enter") save(); });
  const [btn, hint] = info("tr.cfgHint");
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "tr.cfgFile"}), btn), hint,
    h("div.dsrow", {}, pick,
      h("button.pill", {type: "button", disabled: running, "data-i18n": "tr.cfgSave",
                        on: {click: () => { T.saving = !T.saving; drawPanel(); if (T.saving) $("trCfgName")?.focus(); }}}), del),
    T.saving ? h("div.dsrow", {}, Object.assign(name, {id: "trCfgName"}),
      h("button.pill.primary", {type: "button", "data-i18n": "tr.cfgSaveGo", on: {click: save}})) : null,
    T.cfgMsg ? h("span.msg.small", {role: "status", text: T.cfgMsg}) : null);
}

// 暫停的訓練（按停止時存的進度；勾了「練完也存進度」的練完也有）：接著練、加練、丟掉（server/train.py 的 resume）
function progressRow(running) {
  const list = T.opts.progress || [];
  if (!list.length) return null;
  const [btn, hint] = info("tr.progressHint");
  return h("div.rfsec", {id: "trProgress"},
    h("div.rflab", {}, h("b", {"data-i18n": "tr.progress"}), btn), hint,
    list.map(p => {
      const slider = p.kind === "slider";
      const more = h("input.find.trmore", {type: "number", min: p.done ? "1" : "0", max: slider ? "5000" : "200", step: slider ? "50" : "1",
                                           value: String(p.done ? (slider ? 200 : 2) : 0), disabled: running,
                                           "aria-label": t(slider ? "tr.moreSteps" : "tr.moreEpochs")});
      const arm = T.dropArm === p.dir;
      return h("div.trprog", {},
        h("div.trprogname", {}, h("b", {text: p.name}),
          h("span.small", {text: `${slider ? "Slider" : ARCH_LABEL[p.arch] || ""} · ${p.date || ""} · ${p.mb} MB`})),
        h("span.small", {text: p.done ? t("tr.progDone", {step: p.step}) : t("tr.progAt", {step: p.step, total: p.total})}),
        h("label.trmorelab", {}, h("span", {text: t(slider ? "tr.moreSteps" : "tr.moreEpochs")}), more),
        h("button.pill.primary", {type: "button", disabled: running, text: t(p.done ? "tr.extend" : "tr.resume"),
                                  on: {click: () => resumeGo(p, Number(more.value) || 0)}}),
        h("button.pill", {type: "button", disabled: running, text: t(arm ? "tr.progDropSure" : "tr.progDrop"),
                          on: {click: async () => {
                            if (!arm) { T.dropArm = p.dir; drawPanel(); return; }
                            const d = await post("/api/train/progress", {op: "discard", dir: p.dir}).catch(e => ({error: e.message}));
                            T.dropArm = "";
                            if (d.progress) T.opts.progress = d.progress;
                            T.progMsg = d.error || "";
                            drawPanel();
                          }}}));
    }),
    T.progMsg ? h("span.msg.small", {role: "status", "data-kind": "err", text: T.progMsg}) : null);
}

async function resumeGo(p, more) {
  const d = await post("/api/train/resume", {dir: p.dir, more, keep_state: !!T.cfg.keep_state}).catch(e => ({error: e.message}));
  T.progMsg = d.error || "";          // 錯誤顯示在這一區（開始鈕在最下面，看不到）
  if (d.error) { drawPanel(); return; }
  T.msg = "";
  T.figs.clear();
  T.st = {...(T.st || {}), running: true};
  drawPanel();
  poll();
}

// 成對圖輔助：用 Qwen 修圖把一個資料夾的圖改成正向版與反向版，做好自動填進上面的兩個資料夾
function pairMaker(running) {
  const st = PM.st, busy = running || st?.running;
  PM.src ||= T.cfg.dataset || T.opts.datasets[0] || "";
  // 預設官方的 Qwen Image 2.1（檔名有 qwen_image 的），沒有才用第一個（別人微調的不一定只改要改的地方）
  PM.model ||= (T.opts.qwen_models?.find(m => /qwen[_-]?image/i.test(m.name)) || T.opts.qwen_models?.[0])?.path || "";
  PM.prompts ||= T.cfg.sl_neutral || "1girl, solo";
  PM.name ||= T.cfg.name ? `${T.cfg.name}_pairs` : "";
  PM.root ||= T.opts.pairs_root || "";
  const auto = PM.src_mode === "auto";
  const upd = (k, v, redraw = false) => { PM[k] = v; if (redraw) drawPanel(); clearTimeout(PM.timer); PM.timer = setTimeout(pmEstimate, 400); };
  const src = h("select.find", {"data-i18n-aria": "tr.pm.src", disabled: busy, on: {change: e => upd("src", e.target.value, true)}},
    [...new Set([PM.src, ...T.opts.datasets].filter(Boolean))].map(d =>
      h("option", {value: d, selected: d === PM.src, text: d.split(/[\\/]/).pop()})));
  const models = T.opts.qwen_models || [];
  const model = models.length ? h("select.find", {"data-i18n-aria": "tr.pm.model", disabled: busy, on: {change: e => upd("model", e.target.value)}},
    models.map(m => h("option", {value: m.path, selected: m.path === PM.model, text: m.name}))) : h("p.hint.warn", {"data-i18n": "tr.pm.noQwen"});
  const mode = h("div.seg", {role: "group", "data-i18n-aria": "tr.pm.mode"}, PM_MODES.map(k => h("button", {
    type: "button", disabled: busy, "aria-pressed": String(PM.mode === k), "data-i18n": `tr.pm.mode.${k}`, on: {click: () => upd("mode", k, true)}})));
  const box = k => h("textarea.find.trprompts", {rows: "2", value: PM[k], spellcheck: "false", disabled: busy,
    "data-i18n-placeholder": `tr.pm.${k}Ph`, "data-i18n-aria": `tr.pm.${k}`, on: {input: e => upd(k, e.target.value)}});
  const srcSeg = h("div.seg", {role: "group", "data-i18n-aria": "tr.pm.srcMode"}, PM_SRC.map(k => h("button", {
    type: "button", disabled: busy, "aria-pressed": String(PM.src_mode === k), "data-i18n": `tr.pm.srcMode.${k}`,
    on: {click: () => upd("src_mode", k, true)}})));
  const prompts = h("textarea.find.trprompts", {rows: "3", value: PM.prompts, spellcheck: "false", disabled: busy,
    "data-i18n-aria": "tr.pm.prompts", on: {input: e => upd("prompts", e.target.value)}});
  const nBase = h("input.find.trmore", {type: "number", min: "4", max: "40", step: "1", value: String(PM.n_base), disabled: busy,
    "data-i18n-aria": "tr.pm.n_base", on: {change: e => upd("n_base", Number(e.target.value) || 12)}});
  const where = h("div.dsrow", {},
    h("input.find", {type: "text", value: PM.name, spellcheck: "false", disabled: busy, "data-i18n-placeholder": "tr.pm.namePh",
                     "data-i18n-aria": "tr.pm.name", on: {input: e => upd("name", e.target.value)}}));
  const root = h("div.dsrow", {},
    h("input.find", {type: "text", value: PM.root, spellcheck: "false", disabled: busy, "data-i18n-aria": "tr.pm.root",
                     on: {change: e => upd("root", e.target.value.trim())}}),
    h("button.pill", {type: "button", "data-i18n": "set.browse", disabled: busy, on: {click: async () => {
      const got = await pickFolder(PM.root || T.opts.pairs_root);
      if (got) upd("root", got, true);
    }}}));
  const [note, err] = pmNote();
  const [btn, hint] = info("tr.pm.hint");
  return h("details.trpm", {open: PM.open || !!st?.running, on: {toggle: ev => { PM.open = ev.currentTarget.open; if (PM.open && !PM.est) pmEstimate(); }}},
    h("summary", {}, h("b", {"data-i18n": "tr.pm.title"})),
    h("div.rflab", {}, h("span.small.dim", {"data-i18n": "tr.pm.lead"}), btn), hint,
    field("tr.pm.srcMode", srcSeg, "tr.pm.srcModeHint"),
    ...(auto ? [field("tr.pm.prompts", prompts, "tr.pm.promptsHint"),
                h("label.trmorelab", {}, h("span", {"data-i18n": "tr.pm.n_base"}), nBase),
                field("tr.pm.name", where), field("tr.pm.root", root)]
      : [field("tr.pm.src", src)]),
    field("tr.pm.model", model),
    field("tr.pm.mode", mode, "tr.pm.modeHint"),
    PM.mode !== "orig_pos" ? field("tr.pm.pos_text", box("pos_text")) : null,
    PM.mode !== "orig_neg" ? field("tr.pm.neg_text", box("neg_text")) : null,
    h("div.rflab", {},
      st?.running ? h("button.pill", {type: "button", "data-i18n": "tr.pm.stop", on: {click: pmStop}})
        : h("button.pill.primary", {type: "button", disabled: running || !models.length, "data-i18n": "tr.pm.start", on: {click: pmStart}}),
      h("span.msg.small", {id: "trPmMsg", role: "status", "data-kind": err ? "err" : null, text: note})));
}

function pmNote() {
  const e = PM.est, st = PM.st;
  const note = st?.running ? t(st.phase === "base" ? "tr.pm.runningBase" : "tr.pm.running", {done: st.done + st.failed, total: st.total})
    : st?.error ? st.error : st?.msg ? `${st.msg}（${t("tr.pm.filled")}）` : e?.error ? e.error
    : e ? t(e.n_base ? "tr.pm.estimateBase" : e.n ? "tr.pm.estimate" : "tr.pm.allDone", {base: e.n_base, n: e.n, skipped: e.skipped, min: e.minutes}) : "";
  return [note, !st?.running && !!e?.error];
}

const pmBody = () => ({src_mode: PM.src_mode, src: PM.src, model: PM.model, mode: PM.mode, pos_text: PM.pos_text, neg_text: PM.neg_text,
                        seed: T.cfg.seed, prompts: PM.prompts, n_base: PM.n_base, name: PM.name, root: PM.root,
                        base_model: T.cfg.model, quality: T.cfg.sl_quality !== false, sfw: T.cfg.sl_sfw !== false});

async function pmEstimate() {
  PM.est = await post("/api/pairs/estimate", pmBody()).catch(err => ({error: err.message}));
  if (PM.st && !PM.st.running) PM.st = null;          // 改了設定：顯示新的估計，不再是上次做完的訊息
  const m = $("trPmMsg");                              // 只換這一行：整個重畫的話正在打字的框會失去焦點
  if (!m) return;
  const [note, err] = pmNote();
  m.textContent = note;
  if (err) m.dataset.kind = "err"; else m.removeAttribute("data-kind");
}

async function pmStart() {
  const d = await post("/api/pairs/start", pmBody()).catch(err => ({error: err.message}));
  if (d.error) { PM.est = {error: d.error}; drawPanel(); return; }
  PM.st = d;
  PM.open = true;
  drawPanel();
  pmPoll();
}

async function pmStop() {
  await post("/api/pairs/cancel", {}).catch(() => null);
  pmPoll();
}

// 做的時候每 1.5 秒問一次；做完把兩個資料夾填進上面（資料集清單也重讀，新的資料夾才選得到）
async function pmPoll() {
  clearTimeout(PM.poll);
  const s = await get("/api/pairs/state").catch(() => null);
  if (!s) return;
  const was = PM.st?.running;
  PM.st = s;
  if (s.running) PM.poll = setTimeout(pmPoll, 1500);
  else if (was || s.pos_dir) {
    const o = await get("/api/train/options").catch(() => null);
    if (o?.datasets) T.opts.datasets = o.datasets;
    if (s.pos_dir && s.neg_dir) Object.assign(T.cfg, {sl_pos_dir: s.pos_dir, sl_neg_dir: s.neg_dir});
    PM.est = null;
    preview();
  }
  if ($("trPanel")) drawPanel();
}

function drawPresetState() {
  const keys = PRESET_KEYS[archOf(T.cfg.model)];
  document.querySelectorAll("#trPresets button").forEach((b, i) => b.setAttribute("aria-pressed", String(T.cfg.preset === keys[i])));
}

async function preview() {
  if (!T.cfg || T.st?.running) return;
  T.prev = await post("/api/train/preview", T.cfg).catch(e => ({error: e.message}));
  showPreview();
}

function showPreview() {
  const el = $("trPreview");
  if (!el || T.st?.running) { if (el) el.textContent = ""; return; }
  const p = T.prev;
  el.dataset.kind = p?.error ? "err" : "";
  el.textContent = !p ? "" : p.error ? p.error
    : T.cfg.kind === "slider" ? t(T.cfg.sl_mode === "pairs" ? "tr.sl.estimatePairs" : p.base ? "tr.sl.estimateBase" : "tr.sl.estimate",
                                  {base: p.base, n: p.n_images, steps: p.steps, samples: p.samples, min: p.minutes, dir: p.out_dir})
    : t(p.samples ? "tr.estimateSamples" : "tr.estimate", {n: p.n_images, steps: p.steps, samples: p.samples, min: p.minutes, dir: p.out_dir});
  const ta = $("trPrompts");
  if (ta) ta.placeholder = p?.default_prompt || "";      // 空白時用的提示詞（資料集第一張圖的標籤；Slider 是中性提示詞）
}

async function start() {
  const d = await post("/api/train/start", T.cfg).catch(e => ({error: e.message}));
  if (d.error) { T.msg = d.error; const m = $("trMsg"); if (m) { m.textContent = d.error; m.dataset.kind = "err"; } return; }
  T.msg = "";
  T.figs.clear();
  T.st = {...(T.st || {}), running: true};
  drawPanel();
  poll();
}

async function stop() {
  await post("/api/train/stop", {}).catch(() => null);
  poll();
}

async function poll() {
  clearTimeout(T.poll);
  const was = T.st?.running;
  const s = await get("/api/train/state").catch(() => null);
  if (!s || !$("trStage")) return;
  T.st = s;
  // 試畫一張張進圖庫：生成頁與圖庫的牆補上（不然要生一張圖才出現，使用者回報；main.js 接，同 library.js 的 galleryChanged）
  const n = (s.samples || []).length;
  if (T.nSamples != null && n > T.nSamples) dispatchEvent(new CustomEvent("coco:gallery", {detail: {kind: "new"}}));
  T.nSamples = n;
  drawStage();
  if (s.running) T.poll = setTimeout(poll, 1000);
  else if (was) {
    // 停下來或練完：進度的清單可能多了一份（或接續練完、用掉了一份）
    const p = await post("/api/train/progress", {}).catch(() => null);
    if (p?.progress) T.opts.progress = p.progress;
    drawPanel();
    preview();
  }
}

// ── 右邊 ──
const mmss = sec => {
  const s = Math.round(sec), hh = Math.floor(s / 3600), mm = Math.floor(s % 3600 / 60), ss = String(s % 60).padStart(2, "0");
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
};

function drawStage() {
  const box = $("trStage");
  if (!box) return;
  const s = T.st;
  if (!s || (!s.running && !s.phase)) {
    box.replaceChildren(h("p.hint.dsempty", {"data-i18n": "tr.idle"}));
    translate(box);
    return;
  }
  const pct = s.total ? Math.floor(s.step / s.total * 100) : 0;
  const slider = s.kind === "slider";
  const phase = s.phase === "cache" ? t("tr.phase.cache", {done: s.cache[0], total: s.cache[1]})
    : s.phase === "base" ? t("tr.phase.base", {done: s.cache[0], total: s.cache[1]})
    : s.phase === "load" ? t("tr.phase.load")
    : s.phase === "done" ? t("tr.phase.done")
    : s.phase === "sample" ? t("tr.phase.sample")
    : s.phase === "error" ? t("tr.phase.error", {msg: s.error})
    : slider ? t("tr.phase.slider", {step: s.step, total: s.total})
    : t("tr.phase.train", {epoch: s.epoch, epochs: s.epochs, step: s.step, total: s.total});
  box.replaceChildren(...[
    h("div.rhd", {}, h("b.trname", {text: s.name}), h("span.sp"), h("span.small.dim", {text: phase})),
    h("div.xbar.trbar", {}, h("i", {style: `width:${pct}%`})),
    ["train", "sample", "done"].includes(s.phase) ? h("div.trstats", {},
      stat("tr.loss", s.loss.length ? s.loss[s.loss.length - 1][1].toFixed(4) : "—"),
      stat("tr.lr", s.lr ? s.lr.toExponential(2) : "—"),
      stat("tr.speed", s.sec ? t("tr.secPerStep", {s: s.sec.toFixed(2)}) : "—"),
      stat("tr.eta", s.running && s.eta ? mmss(s.eta) : "—")) : null,
    ...sampleRows(s),
    s.saved.length ? h("div.rfsec", {}, h("b", {"data-i18n": "tr.saved"}),
      h("div.trsaved", {}, s.saved.map(f => h("div.mono.small", {title: f.path,
        text: t(slider ? "tr.savedStep" : "tr.savedItem", {file: f.path.split(/[\\/]/).pop(), epoch: f.epoch, step: f.step})})))) : null,
    h("div.dslog.trlog", {}, s.log.slice(-12).map(x => h("div", {"data-kind": x.k, text: `${x.t}  ${x.m}`}))),
  ].filter(Boolean));
  translate(box);
}

// 試畫：每則提示詞一列（不掛 LoRA、epoch 5、6…），按鈕打開那則的比較組（挑哪個 epoch 最好可以用淘汰賽）。
// Slider：每則提示詞底下每存一次一排 −2、−1、0、+1、+2（0 是開始前那張不掛 LoRA 的，每排都放）
function sampleRows(s) {
  const by = new Map();
  for (const it of s.samples || []) {
    if (!by.has(it.index)) by.set(it.index, []);
    by.get(it.index).push(it);
  }
  return [...by.keys()].sort((a, b) => a - b).map(i => {
    const sid = s.sets?.[String(i)];
    const list = by.get(i);
    const grid = s.kind === "slider" ? sliderGrid(list)
      : h("div.trsamples", {}, list.map(it => sampleFig(it, it.path, it.epoch ? `epoch ${it.epoch}` : t("tr.sampleBase"))));
    return h("div.rfsec", {},
      h("div.rflab", {}, h("b", {text: by.size > 1 ? `${t("tr.samples")} ${i + 1}` : t("tr.samples")}), h("span.sp"),
        sid ? h("button.mini", {type: "button", "data-i18n": "tr.openCompare", on: {click: () => openCompare(sid)}}) : null),
      grid);
  });
}

function sliderGrid(list) {
  const base = list.find(it => !it.epoch);
  const steps = [...new Set(list.filter(it => it.epoch).map(it => it.epoch))].sort((a, b) => a - b);
  if (!steps.length) return h("div.trsamples.trslider", {}, base ? sampleFig(base, base.path, "0") : null);
  return h("div.trslrows", {}, steps.map(st => h("div.trslrow", {},
    h("span.small.dim.trslstep", {text: t("tr.sl.stepLabel", {step: st})}),
    h("div.trsamples.trslider", {}, SL_MULTS.map(m => {
      const it = m === 0 ? base : list.find(x => x.epoch === st && x.mult === m);
      const label = m > 0 ? `+${m}` : String(m);
      return it ? sampleFig(it, `${it.path}@${st}`, label) : h("figure.trsample", {}, h("div.trsmiss", {text: "…"}),
                                                              h("figcaption.small.dim", {text: label}));
    })))));
}

// key：同一張圖可以出現在好幾排（Slider 的 0），每排各一個元素
function sampleFig(it, key, label) {
  let f = T.figs.get(key);
  if (!f) {
    f = h("figure.trsample", {},
      it.id ? h("a", {href: `/api/full?id=${it.id}&v=${encodeURIComponent(it.v || "")}`, target: "_blank", rel: "noopener", title: label},
        h("img", {src: `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v || "")}`, alt: label, loading: "lazy"}))
        : h("div.trsmiss", {text: "—"}),
      h("figcaption.small.dim", {text: label}));
    T.figs.set(key, f);
  }
  return f;
}

const stat = (k, v) => h("div.trstat", {}, h("span.small.dim", {"data-i18n": k}), h("b.mono", {text: v}));
