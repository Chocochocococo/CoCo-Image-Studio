// 精修（工作室的 05）：拿一張圖重新畫。整張重繪（img2img）、局部重繪（在圖上塗遮罩，
// 只重畫塗到的地方）、移除內容（局部重繪的固定設定：先抹掉、整張，填要移除的東西）、
// 修臉（自動找臉，每張臉可以另外加提示詞，ROADMAP C+.2）；放大之後接在同一頁（C++）。
// 用指示修圖：Qwen Image 2.1 或 FLUX.2 klein 拿這張圖（與其他參考圖）當條件，照一句指示從頭畫一張
// （core/qwen21.edit_embeds、core/flux2.edit_images）。
//
// 左邊選底圖、修改幅度、提示詞；右邊是前後對照滑桿（局部重繪時是畫遮罩的地方）與版本列。
// 版本不另外存：img2img 的輸出在中繼資料記著底圖路徑，/api/refine/versions 直接從圖庫查。
// 模型、取樣器、步數、CFG 預設照原圖的（使用者筆記：用別的模型修，畫風會對不上）；原圖沒記模型、
// 或模型檔找不到時用生成台的（options.js 的 sourceParams）。左欄底下可以切成一律用生成台的。

import {openPicker} from "./picker.js";
import {get, getBlob, post} from "./api.js";
import {addToCompare} from "./compare.js";
import {confirmDialog} from "./confirm.js";
import {$, h, stem} from "./dom.js";
import {cancel, genSettings, submit} from "./generate.js";
import {t, translate} from "./i18n.js";
import {info as hintInfo} from "./hint.js";
import {ARCH_LABEL, EDIT_ARCHS, SAMPLER_GROUPS, addLora, editModelOf, editOpts as editOptsOf, editPatch, editSize,
        sourceParams} from "./options.js";
import {openEditor} from "./editor.js";
import {partPct} from "./modeldl.js";
import {addTags, dropTags} from "./prompttext.js";
import {applySelection, floodSelect, grow, invert, nearFirst, tracePath} from "./maskops.js";
import {FIT, clampPan, panBy, pixelPercent, zoomAt} from "./zoom.js";
import {newest} from "./inflight.js";

// 修改幅度三段（SPEC §6.6：不出現 denoise strength 這種術語，進階才看數值）。
// 局部重繪通常是要換掉東西，同一段要比整張重繪強
export const LEVELS = {
  whole: [["light", 0.3], ["mid", 0.5], ["strong", 0.7]],
  inpaint: [["light", 0.4], ["mid", 0.65], ["strong", 0.9]],
  // 移除內容：先抹掉之後幅度太小會留下糊糊的一塊，三段都偏高
  remove: [["light", 0.75], ["mid", 0.85], ["strong", 1]],
  // 修臉：ADetailer 的預設是 0.4
  face: [["light", 0.3], ["mid", 0.4], ["strong", 0.55]],
  // 高清化的補細節：太高會長出原圖沒有的東西（每塊各自照提示詞畫），三段都偏低
  upscale: [["light", 0.2], ["mid", 0.3], ["strong", 0.4]],
  // 擴圖：新的地方起點只是邊緣顏色延伸開的色塊，幅度太小會留下糊糊的一片，三段都偏高
  outpaint: [["light", 0.75], ["mid", 0.9], ["strong", 1]],
};
const STRENGTH_KEY = {whole: ["rstrength", 0.5], inpaint: ["rinstrength", 0.65], remove: ["rrmstrength", 0.85],
                      face: ["rfstrength", 0.4], upscale: ["rustrength", 0.3], outpaint: ["rostrength", 0.9]};
const MODES = ["whole", "edit", "inpaint", "remove", "outpaint", "face", "rembg", "upscale"];
const READY = ["whole", "edit", "inpaint", "remove", "outpaint", "face", "rembg", "upscale"];
// 修圖的取樣器：常用的那組。排程器不給選：σ 由 CoCo 照官方算好（core/qwen21.sigmas、core/flux2.sigmas），選了也一樣
const EDIT_SAMPLERS = SAMPLER_GROUPS[0][1];
// 去背的模型（server/rembg.py 的清單）：使用者按了才下載
const RB_MODELS = ["isnet-anime"];
const MASKED = ["inpaint", "remove"];    // 要畫遮罩的方式
const BRUSH = [4, 400];                  // 筆刷大小（底圖的像素）
const TINT = [255, 64, 128];
const MASK_TINT = `rgba(${TINT.join(", ")}, 1)`;
// 遮罩工具（使用者筆記：除了筆刷，還要方形、圓形、套索、多邊形、魔術棒）。每種都可以塗或擦（rerase）。
// 塗色（color）只有局部重繪有：塗上的顏色當重畫的起點（A1111 的 Inpaint sketch），同時也塗進遮罩
const TOOLS = ["brush", "color", "rect", "ellipse", "lasso", "poly", "wand"];
const TOOL_KEY = {b: "brush", c: "color", r: "rect", o: "ellipse", l: "lasso", p: "poly", w: "wand"};
const TOL = [0, 128];                    // 魔術棒的容許度（每個色版差多少以內算相近）
const GROW = 8;                          // 擴張／收縮一次幾像素（底圖的像素）

const R = {ui: null, id: 0, data: null, pick: 0, prompt: null, negative: null, seed: "random", count: 1,
           busy: false, running: false, left: 0, x: 50, view: "mask",
           mask: null,                   // {w, h, canvas, color（塗色）, shape（形狀預覽）, undo: [], empty, poly（畫到一半的多邊形）, colored}
           pix: null,                    // 魔術棒、吸色用的底圖像素：{id, data}
           zoom: {...FIT},               // 放大檢視（zoom.js）；遮罩與前後對照共用，切換時停在同一個地方
           space: false,                 // 按著空白鍵：拖曳是移動畫面，不是塗
           zoomApply: null,              // 目前這個舞台的「套用放大」（重畫舞台時換新的）
           // 修臉：偵測結果。key＝底圖 id｜偵測模型；list 每張 {box（原圖座標）, conf, on（要不要修）, prompt（另外加的）}
           faces: null,
           srcP: null,                   // 照原圖的模型與參數：{id, ok, over, name, why}（id＝底圖）
           rmText: "",                   // 移除內容：要移除的東西（英文標籤，逗號分開）
           editText: "",                 // 用指示修圖：要怎麼改（一句話）
           editRefs: [],                 // 用指示修圖的其他參考圖（圖庫的 id，圖2、圖3）；換一張來修時照留
           em: null};                    // 用指示修圖能用的模型（Qwen Image 2.1、FLUX.2 klein）：{list} 或 {loading}

export function initRefine(ui) {
  R.ui = ui;
  R.id = Number(ui.rsrc) || 0;
  // 以前橡皮擦是一種工具；現在每種工具都能擦
  if (ui.rtool === "erase") saveUi({rtool: "brush", rerase: true});
}

const mode = () => (READY.includes(R.ui?.rmode) ? R.ui.rmode : "whole");
const masked = (m = mode()) => MASKED.includes(m);
const levels = () => LEVELS[mode()] || [];               // 去背沒有修改幅度
const strength = () => {
  const [k, def] = STRENGTH_KEY[mode()] || ["", 1];
  return Math.min(1, Math.max(0.05, Number(R.ui?.[k]) || def));
};
const setStrength = v => saveUi({[STRENGTH_KEY[mode()][0]]: v});
const level = v => levels().find(([, x]) => Math.abs(x - v) < 0.001)?.[0] || "custom";
const brush = () => Math.min(BRUSH[1], Math.max(BRUSH[0], Number(R.ui?.rbrush) || 48));
const area = () => (R.ui?.rarea === "masked" ? "masked" : "whole");
// 擴圖、局部重繪、移除內容可以加 union ControlNet 的「重繪」讓接縫更自然（要先下載 union 模型）
const cnOn = () => !!R.ui?.rcnrepaint;
function cnSection(md) {
  if (!["outpaint", "inpaint", "remove"].includes(md)) return null;
  if (R.cnReady == null) {
    R.cnReady = false;
    get("/api/controlnet/status").then(d => { R.cnReady = (d.models || []).some(m => m.arch === "sdxl-union"); drawPanel(); }).catch(() => {});
  }
  return h("div.cell.swcell.rfcn", {},
    h("span.swtext", {}, h("b", {id: "rfCnLbl", "data-i18n": "ref.cn"}), h("span.hint", {"data-i18n": R.cnReady ? "ref.cnHint" : "ref.cnNeed"})),
    h("button.switch", {type: "button", role: "switch", "aria-labelledby": "rfCnLbl", disabled: !R.cnReady,
                        "aria-checked": String(R.cnReady && cnOn()), on: {click: () => { saveUi({rcnrepaint: !cnOn()}); drawPanel(); }}}, h("i")));
}
const fillMode = () => (R.ui?.rfill === "fill" ? "fill" : "original");
const toolList = () => (mode() === "inpaint" ? TOOLS : TOOLS.filter(k => k !== "color"));
const tool = () => (toolList().includes(R.ui?.rtool) ? R.ui.rtool : "brush");
const tolerance = () => Math.min(TOL[1], Math.max(TOL[0], Number(R.ui?.rtol ?? 32)));
const paintColor = () => (/^#[0-9a-f]{6}$/i.test(R.ui?.rcolor || "") ? R.ui.rcolor : "#d04848");
const brushLike = tl => tl === "brush" || tl === "color";
const erasing = () => !!R.ui?.rerase;
const blur = () => Math.min(32, Math.max(0, Number(R.ui?.rblur ?? 8)));
const thumb = it => `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v)}`;
const full = it => `/api/full?id=${it.id}&v=${encodeURIComponent(it.v)}`;

function saveUi(patch) {
  Object.assign(R.ui, patch);
  post("/api/ui", patch).catch(() => { /* 只影響下次開啟 */ });
}

// 別的視窗（生成台）改了設定：左欄底下「使用中的模型、取樣」跟著更新。正在送出時不動
export function refineSettingsChanged() {
  if ($("rfPanel") && R.data && !R.busy) drawPanel();
}

// 從圖庫、看大圖進來
export function openRefine(id) {
  if (Number(id) !== R.id) {
    R.id = Number(id); R.prompt = R.negative = null; R.pick = 0; R.view = "mask"; R.zoom = {...FIT}; R.faces = null; R.srcP = null;
    R.rmText = "";
    saveUi({rsrc: R.id});
  }
  if (location.hash !== "#refine") location.hash = "#refine";
  else showRefine();
}

// 換一張：跳出挑圖的視窗，點一張就換（使用者回報：以前回到圖庫，之前的勾選還在，選好還要再按「送到精修」）
function pickSource() {
  openPicker({title: t("ref.pickTitle"), have: R.id ? [R.id] : [], single: true, onAdd: ids => openRefine(ids[0])});
}

export function refineScreen() {
  return h("div.rfm", {},
    h("aside.card.rail.rfpanel", {id: "rfPanel", "aria-labelledby": "rfTitle"}),
    h("section.card.rfstage", {id: "rfStage", "aria-live": "polite"}));
}

// 底圖的版本清單：只收最後一次、而且還是目前這張的回覆——快速開 A 再開 B、A 比較晚回來時，
// 以前畫面與 R.data.source 被換回 A，之後精修送出的是 A（Codex 第三輪審查）
const versionsTicket = newest();
const newImagesTicket = newest();

export async function showRefine() {
  if (!$("rfPanel")) return;
  R.data = null;
  const id = R.id, still = versionsTicket();
  if (id) {
    const d = await get(`/api/refine/versions?id=${id}`).catch(e => ({error: e.message}));
    if (!still() || id !== R.id) return;            // 已經又開了別張：這個回覆不要了（那張的 showRefine 會畫）
    if (!d.error) R.data = d;
    else { R.id = 0; saveUi({rsrc: 0}); }
  }
  if (R.data && !R.data.items.some(it => it.id === R.pick)) R.pick = R.data.items[0]?.id || 0;
  drawPanel();
  drawStage();
}

// 生成台輪詢到新圖時：重查版本，自動選最新那張
export async function refineNewImages() {
  if (!$("rfStage") || !R.id) return;
  const id = R.id, still = newImagesTicket();
  const before = new Set((R.data?.items || []).map(it => it.id));
  const d = await get(`/api/refine/versions?id=${id}`).catch(() => null);
  // 同上；這張還沒載好（showRefine 還在讀）也不動，交給 showRefine 畫
  if (!still() || id !== R.id || !R.data || !d || d.error) return;
  R.data = d;
  const fresh = d.items.find(it => !before.has(it.id));
  if (fresh) { R.pick = fresh.id; R.view = "compare"; }
  drawStage();
}

// 生成台的輪詢結果（每次都會來）：更新進度、生成中的空格
export function refineProgress(st) {
  if (!$("rfPanel")) return;
  const was = R.running;
  R.running = !!st.running;
  const rows = (st.table || []).filter(r => r.b === Math.max(0, ...(st.table || []).map(x => x.b ?? 0)));
  R.left = R.running ? rows.filter(r => r.status === "waiting" || r.status === "running").length : 0;
  const bar = $("rfBar"), line = $("rfLine"), stop = $("rfStop");
  if (bar) bar.style.width = R.running && st.total ? `${Math.round(st.step / st.total * 100)}%` : "0";
  if (line) line.textContent = R.running ? (st.total ? t("ref.running", {step: st.step, total: st.total, left: R.left}) : t("gen.run.loading")) : "";
  if (stop) stop.hidden = !R.running;
  syncGo();
  if (was !== R.running) drawStrip();
}

// 重繪鈕：生成中、沒有底圖、局部重繪卻還沒塗遮罩、修臉卻一張臉都沒勾時不能按
function syncGo() {
  const go = $("rfGo"), hint = $("rfGoHint");
  const needMask = masked() && (!R.mask || R.mask.empty);
  const needFace = mode() === "face" && !R.faces?.list.some(f => f.on);
  const needModel = mode() === "rembg" && !rbModel();
  const needEdit = mode() === "edit" && (!editModel() || !R.editText.trim());
  const needUp = upNoop();
  if (go) go.disabled = R.running || R.busy || !R.data || needMask || needFace || needModel || needEdit || needUp;
  if (hint) {
    hint.hidden = !needMask && !needFace && !needModel && !needEdit && !needUp;
    hint.textContent = t(needEdit ? (editModel() ? "ref.edit.needText" : "ref.edit.noModel")
      : needUp ? "ref.up.needChange" : needModel ? "ref.rb.needModel" : needFace ? "ref.needFace" : "ref.needMask");
  }
}

// ── 修臉：找臉（server 交給 worker 跑 YOLO，core/faces.py）──
async function detectFaces(model = R.ui?.rfmodel || "") {
  const src = R.data?.source;
  if (!src) return;
  const key = `${src.id}|${model}`;
  const mine = facesFor(src) ? R.faces.list.filter(f => f.manual) : [];   // 自己框的：重新找臉、換偵測模型都留著
  R.faces = {key, status: "loading", list: mine, models: R.faces?.models || [], model};
  drawPanel();
  drawStage();
  const d = await post("/api/refine/detect", {id: src.id, model}).catch(e => ({error: e.message}));
  if (R.faces?.key !== key) return;                     // 這段時間換了底圖或偵測模型
  R.faces = d.error
    ? {key, status: "error", msg: d.error, list: mine, models: d.models || [], model}
    : {key, status: "ok", list: [...d.faces.map(f => ({...f, on: true, prompt: ""})), ...mine], size: d.size, models: d.models, model: d.model};
  drawPanel();
  drawStage();
}

const facesFor = src => R.faces && R.faces.key.startsWith(`${src.id}|`);

function faceSection() {
  const F = R.faces;
  const [btn, hint] = info("faces", "ref.facesHint");
  const models = F?.models || [];
  const blurNum = h("span.mono.rfnum", {text: `${blur()}px`});
  const status = !F || F.status === "loading" ? h("p.dim.small", {"data-i18n": "ref.detecting"})
    : F.status === "error" ? h("p.hint.warn", {text: models.length ? F.msg : t("ref.noDetector")})
    : !F.list.some(f => !f.manual) ? h("p.dim.small", {"data-i18n": "ref.noFaces"})
    : null;
  const list = F?.list.length ? h("div.rffacelist", {}, F.list.map((f, i) => h("div.rfface", {"data-on": f.on ? "1" : null},
      h("label", {},
        h("input", {type: "checkbox", checked: f.on, on: {change: e => { f.on = e.target.checked; drawStage(); syncGo(); markFaces(); }}}),
        h("b", {text: t("ref.face", {n: i + 1})}),
        f.manual ? h("span.dim", {"data-i18n": "ref.faceManual"}) : h("span.dim.mono", {text: `${Math.round(f.conf * 100)}%`}),
        f.manual ? h("button.mini.rfdel", {type: "button", "data-i18n-title": "ref.faceDel", "data-i18n-aria": "ref.faceDel",
                                           on: {click: e => { e.preventDefault(); removeFace(f); }}}, "✕") : null),
      h("input.find", {type: "text", value: f.prompt, spellcheck: "false", "data-i18n-placeholder": "ref.facePh",
                       "data-i18n-aria": "ref.facePh", on: {input: e => { f.prompt = e.target.value; }}})))) : null;
  const body = h("div.rffacebody", {}, ...[status, list, h("p.hint", {"data-i18n": "ref.faceDraw"})].filter(Boolean));
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ref.faces"}), btn, h("span.sp"),
      models.length > 1 ? h("select.chip", {"data-i18n-aria": "ref.detector", on: {change: e => { saveUi({rfmodel: e.target.value}); detectFaces(e.target.value); }}},
        models.map(m => h("option", {value: m, text: m, selected: m === F?.model}))) : null,
      h("button.mini", {type: "button", "data-i18n": "ref.redetect", disabled: F?.status === "loading", on: {click: () => detectFaces()}})),
    hint,
    body,
    h("details.rfexact", {},
      h("summary", {}, t("ref.blur"), " ", blurNum),
      h("input", {type: "range", min: "0", max: "32", step: "1", value: String(blur()), "data-i18n-aria": "ref.blur",
                  on: {input: e => { saveUi({rblur: Number(e.target.value)}); blurNum.textContent = `${blur()}px`; }}}),
      h("p.hint", {"data-i18n": "ref.blurHint"})));
}

// ── 移除內容（使用者筆記：新手不知道移除東西要怎麼設定）──
// 局部重繪的固定設定：起點先抹掉、範圍整張（看得到整張構圖，提示詞也對得上——遮罩附近只看一小塊，
// 整段提示詞會讓它在那塊裡畫出人或手）。填了要移除的東西，送出時加進負面提示詞、
// 從提示詞拿掉提到它的標籤（提示詞框裡的不動，只改送出的那份）
const rmPhrases = () => R.rmText.split(",").map(s => s.trim()).filter(Boolean);

// 送出的提示詞與負面提示詞（移除內容才改）
function sentPrompts(prompt, negative) {
  if (mode() !== "remove" || !rmPhrases().length) return {prompt, negative};
  return {prompt: dropTags(prompt, rmPhrases()).prompt, negative: addTags(negative, rmPhrases())};
}

function removeSection() {
  const [btn, hint] = info("remove", "ref.rm.hint");
  const drops = () => (rmPhrases().length ? dropTags(R.prompt ?? "", rmPhrases()).removed : []);
  const line = h("p.hint.rfrmdrop", {});
  const sync = () => {
    const d = drops();
    line.hidden = !d.length;
    line.textContent = d.length ? t("ref.rm.dropped", {n: d.length, tags: d.join(", ")}) : "";
  };
  sync();
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ref.rm"}), btn),
    hint,
    h("input.find", {type: "text", value: R.rmText, spellcheck: "false", "data-i18n-placeholder": "ref.rm.ph",
                     "data-i18n-aria": "ref.rm.ph", on: {input: e => { R.rmText = e.target.value; sync(); }}}),
    line);
}

// ── 擴圖（pipeline._outpaint）──
// 畫布往四邊加大，新的地方交給模型補。兩種指定法：選目標比例（原圖放在哪一邊），或自己填四邊各擴幾像素
const OUT_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];
const OUT_MAX = {side: 2048, total: 4096, px: 1536 * 1024};   // 每邊上限、加大後的長邊上限（server/refine）、超過就先縮小畫
const oset = () => ({how: "ratio", ratio: "16:9", anchor: "center", sides: {left: 0, top: 0, right: 256, bottom: 0}, ...(R.ui?.rout || {})});
const setO = patch => saveUi({rout: {...oset(), ...patch}});
const even8 = v => Math.max(0, Math.round(v / 8) * 8);

// 這次要擴的四邊：{ok, sides, W, H, why}
function outPlan(src) {
  const o = oset(), w = src.width, hgt = src.height;
  const sides = {left: 0, top: 0, right: 0, bottom: 0};
  if (o.how === "custom") {
    for (const k of Object.keys(sides)) sides[k] = Math.min(OUT_MAX.side, even8(Number(o.sides?.[k]) || 0));
  } else {
    const [a, b] = o.ratio.split(":").map(Number), r = a / b;
    // 原圖靠哪一邊：start＝靠左／靠上（往右／往下擴），end 反過來，center 兩邊平分
    const split = add => (o.anchor === "start" ? [0, add] : o.anchor === "end" ? [add, 0] : [even8(add / 2), add - even8(add / 2)]);
    if (w / hgt < r - 1e-3) [sides.left, sides.right] = split(even8(hgt * r - w));
    else if (w / hgt > r + 1e-3) [sides.top, sides.bottom] = split(even8(w / r - hgt));
  }
  const W = w + sides.left + sides.right, H = hgt + sides.top + sides.bottom;
  const why = !Object.values(sides).some(Boolean) ? (o.how === "custom" ? "ref.out.none" : "ref.out.same")
    : Math.max(W, H) > OUT_MAX.total ? "ref.out.tooBig" : "";
  return {ok: !why, sides, W, H, why};
}
const outSize = src => { const p = outPlan(src); return p.ok ? {w: p.W, h: p.H} : null; };

function outpaintSection(src) {
  const o = oset(), p = outPlan(src);
  const redraw = () => { R.view = "outpaint"; drawPanel(); drawStage(); };    // 改了範圍就切回預覽，看得到新的畫布
  const seg = (aria, keys, on, label, click) => h("div.seg", {role: "group", "data-i18n-aria": aria}, keys.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), ...label(k), on: {click: () => { click(k); redraw(); }}})));
  const sideNum = k => h("label.rfoutnum", {},
    h("span.dim", {"data-i18n": `ref.out.${k}`}),
    h("input.find", {type: "number", min: "0", max: String(OUT_MAX.side), step: "64", value: String(o.sides?.[k] || 0),
                     "data-i18n-aria": `ref.out.${k}`,
                     on: {change: e => { setO({sides: {...o.sides, [k]: Math.min(OUT_MAX.side, even8(Number(e.target.value) || 0))}}); redraw(); }}}));
  // 原圖放哪：比例比原圖寬就是左右，比原圖高就是上下
  const [a, b] = o.ratio.split(":").map(Number);
  const wide = a / b > src.width / src.height;
  return h("div.rfsec", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ref.out.how"}), h("span.sp"),
      seg("ref.out.how", ["ratio", "custom"], k => o.how === k, k => ({"data-i18n": `ref.out.${k}`}), k => setO({how: k}))),
    o.how === "ratio" ? seg("ref.out.ratio", OUT_RATIOS, k => o.ratio === k, k => ({text: k}), k => setO({ratio: k})) : null,
    o.how === "ratio" ? h("div.rflab", {}, h("span.dim", {"data-i18n": "ref.out.anchor"}), h("span.sp"),
      seg("ref.out.anchor", ["start", "center", "end"], k => o.anchor === k, k => ({"data-i18n": `ref.out.${wide ? "h" : "v"}.${k}`}),
          k => setO({anchor: k}))) : null,
    o.how === "custom" ? h("div.rfoutsides", {}, ["top", "bottom", "left", "right"].map(sideNum)) : null,
    h("p.hint", {text: p.why ? t(p.why, {max: OUT_MAX.total}) : t("ref.up.size", {w: src.width, h: src.height, tw: p.W, th: p.H})}),
    p.ok && p.W * p.H > OUT_MAX.px ? h("p.hint", {"data-i18n": "ref.out.bigHint"}) : null);
}

// ── 修臉：自己框（使用者筆記：找不到的臉、想多修的地方，在圖上拖曳加一個框）──
// 座標跟找臉的一樣用原圖的像素；跟找到的臉一起送，伺服器那邊一樣檢查（refine.clean_faces）
const FACE_MIN = 8;                   // 太小的框（其實是點了一下）不算

function addFace(src, box) {
  if (!facesFor(src)) R.faces = {key: `${src.id}|${R.ui?.rfmodel || ""}`, status: "ok", list: [], models: [], model: R.ui?.rfmodel || ""};
  R.faces.list.push({box, on: true, prompt: "", manual: true});
  drawPanel();
  drawStage();
}

function removeFace(f) {
  if (!R.faces) return;
  R.faces.list = R.faces.list.filter(x => x !== f);
  drawPanel();
  drawStage();
}

// 在修臉的圖上拖曳：畫一個框（點在既有的框上是切換修不修，不會開始畫）。平移（空白鍵、滾輪鍵）由 wireZoom 先攔下
function wireFaceDraw(layer, src, size) {
  const [fw, fh] = size;
  const img = layer.querySelector("img");
  const at = e => {
    const r = img.getBoundingClientRect();      // 含放大的 transform
    return [Math.max(0, Math.min(fw, (e.clientX - r.left) / r.width * fw)),
            Math.max(0, Math.min(fh, (e.clientY - r.top) / r.height * fh))];
  };
  const pct = (v, of) => `${(v / of * 100).toFixed(3)}%`;
  let start = null, draft = null;
  const rect = (a, b) => [Math.round(Math.min(a[0], b[0])), Math.round(Math.min(a[1], b[1])),
                          Math.round(Math.max(a[0], b[0])), Math.round(Math.max(a[1], b[1]))];
  layer.addEventListener("pointerdown", e => {
    if (e.button !== 0 || R.space || e.target.closest(".rfbox")) return;
    e.preventDefault();
    start = at(e);
    draft = h("div.rfbox.rfdraft", {"aria-hidden": "true"});
    layer.append(draft);
    try { layer.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
  });
  layer.addEventListener("pointermove", e => {
    if (!start) return;
    const [x0, y0, x1, y1] = rect(start, at(e));
    Object.assign(draft.style, {left: pct(x0, fw), top: pct(y0, fh), width: pct(x1 - x0, fw), height: pct(y1 - y0, fh)});
  });
  const end = e => {
    if (!start) return;
    const box = rect(start, at(e));
    start = null;
    draft?.remove();
    draft = null;
    if (e.type === "pointerup" && box[2] - box[0] >= FACE_MIN && box[3] - box[1] >= FACE_MIN) addFace(src, box);
  };
  layer.addEventListener("pointerup", end);
  layer.addEventListener("pointercancel", end);
}

// 勾選與框的樣子同步（不整個重畫，打到一半的提示詞才不會不見）
function markFaces() {
  const list = R.faces?.list || [];
  document.querySelectorAll(".rfface").forEach((el, i) => {
    el.toggleAttribute("data-on", !!list[i]?.on);
    const cb = el.querySelector("input[type=checkbox]");
    if (cb) cb.checked = !!list[i]?.on;
  });
}

// ── 遮罩 ──
// 一張跟底圖一樣大的 canvas：塗到的地方不透明（顯示時整層半透明疊在圖上），
// 匯出成 PNG 時 alpha 就是遮罩（core/inpaint.load_mask 看 alpha）。
// 換底圖時大小一樣就留著——從某一版繼續精修時通常還是改同一塊
function ensureMask(src) {
  if (R.mask && R.mask.w === src.width && R.mask.h === src.height) return R.mask;
  const canvas = h("canvas.rfmask", {width: String(src.width), height: String(src.height), "data-i18n-aria": "ref.maskCanvas",
                                     "data-tool": tool()});
  // 塗色那層（不接滑鼠，事件都給遮罩那層）：顏色不透明地蓋在圖上，範圍永遠在遮罩裡
  const color = h("canvas.rfcolor", {width: String(src.width), height: String(src.height), "aria-hidden": "true"});
  // 拖曳中的方形、橢圓、套索與畫到一半的多邊形畫在這層，放開才塗進遮罩
  const shape = h("canvas.rfshape", {width: String(src.width), height: String(src.height), "aria-hidden": "true"});
  R.mask = {w: src.width, h: src.height, canvas, color, shape, undo: [], empty: true, poly: null, colored: false};
  wireMask(R.mask);
  return R.mask;
}

function maskEmpty(m) {
  const d = m.canvas.getContext("2d").getImageData(0, 0, m.w, m.h).data;
  for (let i = 3; i < d.length; i += 16) if (d[i]) return false;     // 隔幾個像素看就夠了
  return true;
}

function layerEmpty(canvas) {
  const d = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < d.length; i += 16) if (d[i]) return false;
  return true;
}

// 復原的紀錄：遮罩，塗過色的話連塗色那層一起記（沒塗過就不佔記憶體）
function snapshot(m) {
  m.undo.push({mask: m.canvas.getContext("2d").getImageData(0, 0, m.w, m.h),
               color: m.colored ? m.color.getContext("2d").getImageData(0, 0, m.w, m.h) : null});
  if (m.undo.length > 20) m.undo.shift();
}

// 塗色只留在遮罩裡：擦掉、反轉、收縮遮罩時，那裡的顏色跟著不見
function clipColor(m) {
  if (!m.colored) return;
  const ctx = m.color.getContext("2d");
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(m.canvas, 0, 0);
  ctx.globalCompositeOperation = "source-over";
}

// 遮罩改完之後：塗色跟著裁、更新「有沒有塗」
function maskChanged(m) {
  clipColor(m);
  m.empty = maskEmpty(m);
  syncGo();
}

function undoMask() {
  const m = R.mask;
  if (m?.poly) {                         // 畫到一半的多邊形：先一點一點退回去
    m.poly.pts.pop();
    if (!m.poly.pts.length) m.poly = null;
    drawShape(m);
    return;
  }
  if (!m?.undo.length) return;
  const u = m.undo.pop();
  m.canvas.getContext("2d").putImageData(u.mask, 0, 0);
  if (u.color) m.color.getContext("2d").putImageData(u.color, 0, 0);
  else if (m.colored) m.color.getContext("2d").clearRect(0, 0, m.w, m.h);
  m.empty = maskEmpty(m);
  syncGo();
}

function clearMask() {
  const m = R.mask;
  if (!m) return;
  snapshot(m);
  m.canvas.getContext("2d").clearRect(0, 0, m.w, m.h);
  m.color.getContext("2d").clearRect(0, 0, m.w, m.h);
  m.empty = true;
  syncGo();
}

// 反轉、擴張、收縮：整張遮罩一起算（maskops.js），可以復原
function adjustMask(fn) {
  const m = R.mask;
  if (!m) return;
  cancelPoly();
  snapshot(m);
  const ctx = m.canvas.getContext("2d");
  const img = ctx.getImageData(0, 0, m.w, m.h);
  fn(img.data, m.w, m.h);
  ctx.putImageData(img, 0, 0);
  maskChanged(m);
}
const invertMask = () => adjustMask(d => invert(d, TINT));
const growMask = r => adjustMask((d, w, h) => grow(d, w, h, r, TINT));

// 形狀塗進遮罩（或從遮罩擦掉）
function commitShape(m, s) {
  const ctx = m.canvas.getContext("2d");
  const probe = {beginPath() {}, rect() {}, ellipse() {}, moveTo() {}, lineTo() {}, closePath() {}};
  if (!tracePath(probe, s)) return;                  // 圍不出範圍（只點了一下）：不留復原的紀錄
  snapshot(m);
  ctx.globalCompositeOperation = s.erase ? "destination-out" : "source-over";
  ctx.fillStyle = MASK_TINT;
  tracePath(ctx, s);
  ctx.fill();
  maskChanged(m);
}

// ── 魔術棒、吸色：要底圖的像素（第一次用時載入，跟遮罩一樣大）──
async function sourcePixels(src) {
  if (R.pix?.id === src.id) return R.pix.data;
  // 不用 Image.decode()：分頁不在前景時它可能一直不回來
  const img = await createImageBitmap(await getBlob(full(src)));
  const c = document.createElement("canvas");
  c.width = src.width; c.height = src.height;
  const ctx = c.getContext("2d", {willReadFrequently: true});
  ctx.drawImage(img, 0, 0, src.width, src.height);
  R.pix = {id: src.id, data: ctx.getImageData(0, 0, src.width, src.height).data};
  return R.pix.data;
}

// 魔術棒：點一下選顏色相近、相連的區域；按住 Shift 選整張圖裡相近的顏色
async function wandAt(m, p, erase, all) {
  const src = R.data?.source;
  if (!src || src.width !== m.w || src.height !== m.h) return;
  let px;
  try { px = await sourcePixels(src); } catch { return; }
  const sel = floodSelect(px, m.w, m.h, p[0], p[1], tolerance(), !all);
  snapshot(m);
  const ctx = m.canvas.getContext("2d");
  const img = ctx.getImageData(0, 0, m.w, m.h);
  applySelection(img.data, sel, erase, TINT);
  ctx.putImageData(img, 0, 0);
  maskChanged(m);
}

// 吸色（塗色時 Ctrl＋點一下）：塗過的顏色優先，沒塗過就取底圖的
async function pickColor(m, p) {
  const [x, y] = [Math.floor(p[0]), Math.floor(p[1])];
  const c = m.color.getContext("2d").getImageData(x, y, 1, 1).data;
  let rgb = c[3] ? [c[0], c[1], c[2]] : null;
  if (!rgb) {
    try {
      const px = await sourcePixels(R.data.source), i = (y * m.w + x) * 4;
      rgb = [px[i], px[i + 1], px[i + 2]];
    } catch { return; }
  }
  setColor(`#${rgb.map(v => v.toString(16).padStart(2, "0")).join("")}`);
}

// 預覽：虛線外框，塗的時候裡面淡淡上色。線寬換算成畫面上固定的粗細（含放大檢視）
function drawShape(m, s = m.drag || (m.poly && {...m.poly, pts: m.poly.hover ? [...m.poly.pts, m.poly.hover] : m.poly.pts})) {
  const ctx = m.shape.getContext("2d");
  ctx.clearRect(0, 0, m.w, m.h);
  if (!s) return;
  const px = m.w / (m.shape.getBoundingClientRect().width || m.w);
  const open = s.tool === "poly" && s.pts.length < 3;
  if (open) {                                         // 還圍不出範圍：只畫線
    ctx.beginPath();
    s.pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  } else if (!tracePath(ctx, s)) return;
  if (!s.erase && !open) { ctx.fillStyle = "rgba(255, 64, 128, .35)"; ctx.fill(); }
  ctx.setLineDash([]);
  ctx.lineWidth = 3 * px;
  ctx.strokeStyle = "rgba(0, 0, 0, .6)";
  ctx.stroke();
  ctx.setLineDash([6 * px, 4 * px]);
  ctx.lineWidth = 1.5 * px;
  ctx.strokeStyle = "#fff";
  ctx.stroke();
  if (s.tool === "poly" && s.pts.length) {            // 第一個點畫大一點：點回這裡就收起來
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(...s.pts[0], 5 * px, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.stroke();
  }
}

function closePoly() {
  const m = R.mask;
  if (!m?.poly) return;
  const s = {tool: "poly", pts: m.poly.pts, erase: m.poly.erase};
  m.poly = null;
  drawShape(m);
  commitShape(m, s);
}

function cancelPoly() {
  const m = R.mask;
  if (!m?.poly) return;
  m.poly = null;
  drawShape(m);
}

function wireMask(m) {
  const c = m.canvas;
  let last = null, lastErase = false, lastColor = null;
  // 螢幕座標 → 底圖像素。getBoundingClientRect 與 clientX 都是畫面上的像素，
  // 介面大小（CSS zoom）兩邊一起算進去，比例不受影響
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * m.w / r.width, (e.clientY - r.top) * m.h / r.height];
  };
  const stroke = (a, b) => {
    const ctx = c.getContext("2d");
    ctx.globalCompositeOperation = lastErase ? "destination-out" : "source-over";
    ctx.strokeStyle = ctx.fillStyle = MASK_TINT;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = brush();
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
    if (lastColor && !lastErase) {         // 塗色：同一筆也畫在塗色那層
      const cc = m.color.getContext("2d");
      cc.globalCompositeOperation = "source-over";
      cc.strokeStyle = lastColor;
      cc.lineCap = cc.lineJoin = "round";
      cc.lineWidth = brush();
      cc.beginPath();
      cc.moveTo(...a);
      cc.lineTo(...b);
      cc.stroke();
    }
  };
  c.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    e.preventDefault();
    const p = at(e), erase = erasing() !== e.altKey;           // 按著 Alt：這一筆反過來
    const tl = tool();
    if (tl === "poly") {
      // 多邊形不用拖：一點一點點下去；點回第一個點附近或雙擊就收起來
      const tol = 10 * m.w / (c.getBoundingClientRect().width || m.w);
      if (!m.poly) m.poly = {tool: "poly", pts: [p], erase, hover: null};
      else if (e.detail >= 2 || nearFirst(m.poly.pts, p, tol)) { closePoly(); return; }
      else m.poly.pts.push(p);
      drawShape(m);
      return;
    }
    if (tl === "wand") { wandAt(m, p, erase, e.shiftKey); return; }
    if (tl === "color" && (e.ctrlKey || e.metaKey)) { pickColor(m, p); return; }
    try { c.setPointerCapture(e.pointerId); } catch { /* 拖出畫布外時就收不到，不影響塗 */ }
    if (brushLike(tl)) {
      lastColor = tl === "color" ? paintColor() : null;
      if (lastColor && !erase) m.colored = true;
      snapshot(m);
      last = p;
      lastErase = erase;
      stroke(last, last);
      return;
    }
    m.drag = {tool: tl, a: p, b: p, pts: [p], square: e.shiftKey, erase};
    drawShape(m);
  });
  c.addEventListener("pointermove", e => {
    moveCursor(e);
    const p = at(e);
    if (m.poly) { m.poly.hover = p; drawShape(m); }
    if (m.drag) {
      m.drag.b = p;
      m.drag.square = e.shiftKey;
      if (m.drag.tool === "lasso") m.drag.pts.push(p);
      drawShape(m);
      return;
    }
    if (!last) return;
    stroke(last, p);
    last = p;
  });
  const end = () => {
    if (m.drag) {
      const s = m.drag;
      m.drag = null;
      drawShape(m);
      commitShape(m, s);
      return;
    }
    if (!last) return;
    last = null;
    maskChanged(m);
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => {
    const cur = $("rfCursor");
    if (cur) cur.hidden = true;
    if (m.poly) { m.poly.hover = null; drawShape(m); }
  });
  // Shift＋滾輪調筆刷大小（滾輪本身是放大檢視，見 wireZoom）。
  // 按著 Shift 時有些系統把滾動換成水平的 deltaX
  c.addEventListener("wheel", e => {
    if (!e.shiftKey) return;
    e.preventDefault();
    e.stopPropagation();
    const d = e.deltaY || e.deltaX;
    if (d) setBrush(brush() * (d < 0 ? 1.15 : 1 / 1.15));
    moveCursor(e);
  }, {passive: false});
}

// 筆刷的圓圈跟著游標，大小是畫面上實際會塗到的範圍（含放大檢視）。
// 圓圈放在沒有放大的那一層：canvas 的大小含放大倍率，外框的大小只含介面大小（CSS zoom）
function moveCursor(e) {
  const cur = $("rfCursor"), m = R.mask;
  if (!cur || !m) return;
  if (!brushLike(tool())) { cur.hidden = true; return; }    // 形狀工具、魔術棒用十字游標（CSS）
  const box = cur.parentElement, br = box.getBoundingClientRect(), r = m.canvas.getBoundingClientRect();
  const z = br.width / box.offsetWidth || 1;          // 介面大小的倍率：style 要用放大前的像素
  const d = brush() * r.width / m.w / z;
  Object.assign(cur.style, {width: `${d}px`, height: `${d}px`,
                            left: `${(e.clientX - br.left) / z}px`, top: `${(e.clientY - br.top) / z}px`});
  cur.hidden = false;
}

// ── 放大檢視（使用者筆記 #1）──
// 滾輪以游標為中心放大縮小；空白鍵＋拖曳、或按住滾輪拖曳移動畫面；0 回到符合視窗。
// 整個檢視區都是畫布（舞台佔滿 .rfview）：圖平常置中在 .rfz 的位置，放大後可以延伸到整個區域，
// 窄的直圖放大時左右不會被切掉。只放大圖那一層（.rfz），前後對照的分隔線、標籤、筆刷圓圈不跟著變大
function wireZoom(view, stage, src) {
  const layers = [...stage.querySelectorAll(".rfz")];
  const img = layers[0];                 // 符合視窗時圖的位置（offset* 不含 transform）
  const frame = () => ({L: img.offsetLeft, T: img.offsetTop, w: img.offsetWidth, h: img.offsetHeight,
                        W: stage.offsetWidth, H: stage.offsetHeight});
  const uiz = () => stage.getBoundingClientRect().width / stage.offsetWidth || 1;
  // 滑鼠位置換成「相對於符合視窗時圖的左上角」（zoom.js 的座標）
  const local = e => {
    const r = stage.getBoundingClientRect(), z = uiz();
    return [(e.clientX - r.left) / z - img.offsetLeft, (e.clientY - r.top) / z - img.offsetTop];
  };
  const pct = h("button.rfzpct.mono", {type: "button", "data-i18n-title": "ref.zoomFit", on: {click: () => set({...FIT})}});
  const set = z => {
    R.zoom = clampPan(z, frame());
    const {k, x, y} = R.zoom;
    for (const l of layers) l.style.transform = k > 1 ? `translate(${x}px, ${y}px) scale(${k})` : "";
    view.toggleAttribute("data-zoomed", k > 1);
    pct.textContent = `${pixelPercent(R.zoom, img.offsetWidth * uiz(), src.width)}%`;
  };
  // 按鈕：以畫布中間為中心
  const step = factor => { const f = frame(); set(zoomAt(R.zoom, factor, f.W / 2 - f.L, f.H / 2 - f.T, f)); };
  view.append(h("div.rfzctl", {role: "group", "data-i18n-aria": "ref.zoom"},
    h("button", {type: "button", "data-i18n-aria": "ref.zoomOut", "data-i18n-title": "ref.zoomOut", on: {click: () => step(1 / 1.5)}}, "−"),
    pct,
    h("button", {type: "button", "data-i18n-aria": "ref.zoomIn", "data-i18n-title": "ref.zoomIn", on: {click: () => step(1.5)}}, "+")));
  view.addEventListener("wheel", e => {
    e.preventDefault();
    const [px, py] = local(e);
    set(zoomAt(R.zoom, e.deltaY < 0 ? 1.25 : 1 / 1.25, px, py, frame()));
    moveCursor(e);
  }, {passive: false});
  // 移動畫面：捕獲階段先攔下來，canvas 才不會跟著塗、對照的分隔線才不會跟著跑
  let pan = null;
  view.addEventListener("pointerdown", e => {
    if (!(e.button === 1 || (e.button === 0 && R.space))) return;
    e.preventDefault();
    e.stopPropagation();
    if (R.zoom.k <= 1) return;
    pan = {x: e.clientX, y: e.clientY, z: R.zoom};
    try { view.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    view.classList.add("panning");
  }, true);
  view.addEventListener("mousedown", e => { if (e.button === 1) e.preventDefault(); });   // 不要跳出瀏覽器的自動捲動
  view.addEventListener("pointermove", e => {
    if (!pan) return;
    const z = uiz();
    set(panBy(pan.z, (e.clientX - pan.x) / z, (e.clientY - pan.y) / z, frame()));
  });
  const end = () => { pan = null; view.classList.remove("panning"); };
  view.addEventListener("pointerup", end);
  view.addEventListener("pointercancel", end);
  R.zoomApply = set;
  set(R.zoom);
  translate(view);
}
addEventListener("resize", () => { if ($("rfStage")) R.zoomApply?.(R.zoom); });

function setBrush(v) {
  const b = Math.round(Math.min(BRUSH[1], Math.max(BRUSH[0], v)));
  saveUi({rbrush: b});
  const sl = $("rfBrush"), num = $("rfBrushNum");
  if (sl) sl.value = String(b);
  if (num) num.textContent = `${b}px`;
}

function setTool(tl) {
  if (tl !== "poly") cancelPoly();
  saveUi({rtool: tl});
  document.querySelectorAll("#rfTools button[data-tool]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.tool === tl)));
  if (R.mask) R.mask.canvas.dataset.tool = tl;
  toolOpts(tl);
}

// 工具列上只顯示目前這個工具用得到的設定：筆刷大小（畫筆、塗色）、顏色（塗色）、容許度（魔術棒）
function toolOpts(tl = tool()) {
  const show = {rfBrushSize: brushLike(tl), rfColorBox: tl === "color", rfTolBox: tl === "wand"};
  for (const [id, on] of Object.entries(show)) { const el = $(id); if (el) el.hidden = !on; }
}

function setColor(v) {
  saveUi({rcolor: v});
  const el = $("rfColor");
  if (el) el.value = v;
}

function setTolerance(v) {
  const x = Math.round(Math.min(TOL[1], Math.max(TOL[0], v)));
  saveUi({rtol: x});
  const num = $("rfTolNum");
  if (num) num.textContent = String(x);
}

function setErase(on) {
  saveUi({rerase: on});
  document.querySelectorAll("#rfTools button[data-erase]").forEach(b => b.setAttribute("aria-pressed", String((b.dataset.erase === "erase") === on)));
}

// ── 左邊：底圖、方式、修改幅度、提示詞、送出 ──
// 說明文字平常收起來，點欄位名稱旁的 ⓘ 才展開（使用者筆記 #12：左欄太長，重繪鈕被擠到看不見）。
// 修改幅度的說明會跟著選的那一段換，展開與否用欄位 id 記
const info = (id, key) => hintInfo(key, undefined, `ref.${id}`);

function drawPanel() {
  const box = $("rfPanel");
  const src = R.data?.source;
  if (!src) {
    box.replaceChildren(
      h("div.rhd", {}, h("h2.title", {id: "rfTitle", "data-i18n": "nav.refine"})),
      h("div.rfempty", {},
        h("p", {"data-i18n": "ref.empty"}),
        h("button.pill.primary", {type: "button", "data-i18n": "ref.pickInLibrary", on: {click: pickSource}})));
    translate(box);
    return;
  }
  const s = genSettings() || {};
  // 開著品質標籤生成的圖，預填使用者自己寫的（自動加的重繪時會再加，而且不會重複）
  const srcPrompt = withLoras(src), srcNeg = src.user_negative ?? src.negative ?? "";
  if (R.prompt == null) R.prompt = srcPrompt;
  if (R.negative == null) R.negative = srcNeg;
  const md = mode(), v = strength(), lv = level(v);
  const prompt = h("textarea.rftext", {rows: "4", spellcheck: "false", value: R.prompt, "data-i18n-aria": "gen.prompt",
                                       on: {input: e => { R.prompt = e.target.value; }}});
  const neg = h("textarea.rftext", {rows: "3", spellcheck: "false", value: R.negative, "data-i18n-aria": "gen.negative",
                                    on: {input: e => { R.negative = e.target.value; }}});
  const num = h("span.mono.rfnum", {text: v.toFixed(2)});
  const range = h("input", {type: "range", min: "0.05", max: "1", step: "0.05", value: String(v), "data-i18n-aria": "ref.exact",
                            on: {input: e => { setStrength(Number(e.target.value)); num.textContent = strength().toFixed(2); markLevel(); },
                                 change: () => drawPanel()}});
  const markLevel = () => box.querySelectorAll(".rflevel button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.k === level(strength()))));
  const blurNum = h("span.mono.rfnum", {text: `${blur()}px`});
  const top = [
    h("div.rhd", {},
      h("h2.title", {id: "rfTitle", "data-i18n": "nav.refine"}),
      h("span.tag", {text: md === "rembg" || md === "edit" || (md === "upscale" && !upDetail()) ? t(`ref.mode.${md}`) : "img2img"})),
    h("div.rfsrc", {},
      h("img", {src: thumb(src), alt: ""}),
      h("div.rfsrcinfo", {},
        h("b", {text: src.filename, title: src.filename}),
        h("span.dim", {text: `${src.width} × ${src.height}`}),
        h("div.rfsrcact", {},
          R.data.parent ? h("button.mini", {type: "button", "data-i18n": "ref.parent", "data-i18n-title": "ref.parentHint",
                                            on: {click: () => openRefine(R.data.parent.id)}}) : null,
          h("button.mini", {type: "button", "data-i18n": "ref.change", "data-i18n-title": "ref.changeHint", on: {click: pickSource}})))),
    h("div.seg.rfmode", {role: "group", "data-i18n-aria": "ref.mode"}, MODES.map(m => h("button", {
      type: "button", "aria-pressed": String(m === md), "aria-disabled": READY.includes(m) ? null : "true",
      "data-i18n": `ref.mode.${m}`, "data-i18n-title": READY.includes(m) ? `ref.mode.${m}Hint` : "lib.soon",
      on: {click: () => {
        if (!READY.includes(m) || m === md) return;
        saveUi({rmode: m});
        if (masked(m)) R.view = R.mask && !R.mask.empty && R.pick ? "compare" : "mask";
        if (m === "face") R.view = "faces";
        if (m === "outpaint") R.view = "outpaint";
        if (m === "rembg" || m === "upscale" || m === "edit") R.view = "compare";
        drawPanel();
        drawStage();
      }}})))];
  if (md === "rembg") {
    box.replaceChildren(...top, ...rembgPanel());
    translate(box);
    syncGo();
    return;
  }
  if (md === "upscale" && !upDetail()) {
    box.replaceChildren(...top, ...upscalePanel(src), upscaleFoot(src));
    translate(box);
    syncGo();
    return;
  }
  if (md === "edit") {
    box.replaceChildren(...top, ...editPanel(src), seedRow(src), goFoot("ref.edit.go", editSize(src.width, src.height, editOpts().size)));
    translate(box);
    syncGo();
    if (!R.em) loadEditModels();
    return;
  }
  const goSize = md === "upscale" ? upTarget(src) : md === "outpaint" ? outSize(src) : {w: snap(src.width), h: snap(src.height)};
  box.replaceChildren(...[...top,          // 整張重繪時範圍那段是 null，replaceChildren 不會略過
    ...(md === "upscale" ? upscalePanel(src) : []),
    md === "face" ? faceSection() : null,
    md === "outpaint" ? outpaintSection(src) : null,
    md === "remove" ? removeSection() : null,
    cnSection(md),
    md === "inpaint" ? h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "ref.area"}), btn, h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ref.area"}, ["whole", "masked"].map(k => h("button", {
          type: "button", "aria-pressed": String(area() === k), "data-i18n": `ref.area.${k}`,
          on: {click: () => { saveUi({rarea: k}); drawPanel(); }}})))),
      hint])(info("area", `ref.area.${area()}Hint`)),
      h("details.rfexact", {},
        h("summary", {}, t("ref.blur"), " ", blurNum),
        h("input", {type: "range", min: "0", max: "32", step: "1", value: String(blur()), "data-i18n-aria": "ref.blur",
                    on: {input: e => { saveUi({rblur: Number(e.target.value)}); blurNum.textContent = `${blur()}px`; }}}),
        h("p.hint", {"data-i18n": "ref.blurHint"}))) : null,
    md === "remove" ? h("details.rfexact", {},
      h("summary", {}, t("ref.blur"), " ", blurNum),
      h("input", {type: "range", min: "0", max: "32", step: "1", value: String(blur()), "data-i18n-aria": "ref.blur",
                  on: {input: e => { saveUi({rblur: Number(e.target.value)}); blurNum.textContent = `${blur()}px`; }}}),
      h("p.hint", {"data-i18n": "ref.blurHint"})) : null,
    md === "inpaint" ? h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "ref.fill"}), btn, h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ref.fill"}, ["original", "fill"].map(k => h("button", {
          type: "button", "aria-pressed": String(fillMode() === k), "data-i18n": `ref.fill.${k}`,
          on: {click: () => { saveUi({rfill: k}); drawPanel(); }}})))),
      hint])(info("fill", `ref.fill.${fillMode()}Hint`))) : null,
    h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "ref.amount"}), btn, h("span.sp"), num),
      h("div.seg.rflevel", {role: "group", "data-i18n-aria": "ref.amount"}, levels().map(([k, x]) => h("button", {
        type: "button", "data-k": k, "aria-pressed": String(lv === k), "data-i18n": `ref.level.${k}`,
        on: {click: () => { setStrength(x); drawPanel(); }}}))),
      hint])(info("amount", lv === "custom" ? "ref.level.customHint" : md === "inpaint" ? `ref.ilevel.${lv}Hint`
        : md === "remove" ? `ref.rlevel.${lv}Hint` : md === "face" ? `ref.flevel.${lv}Hint`
        : md === "upscale" ? `ref.ulevel.${lv}Hint` : md === "outpaint" ? `ref.olevel.${lv}Hint` : `ref.level.${lv}Hint`)),
      h("details.rfexact", {open: lv === "custom"}, h("summary", {"data-i18n": "ref.exact"}), range)),
    h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "gen.prompt"}), masked(md) || md === "outpaint" ? btn : null, h("span.sp"),
        h("button.mini", {type: "button", "data-i18n": "ref.useSource", on: {click: () => { R.prompt = srcPrompt; R.negative = srcNeg; drawPanel(); }}}),
        h("button.mini", {type: "button", "data-i18n": "ref.useGen", on: {click: () => { R.prompt = s.PROMPT || ""; R.negative = s.NEGATIVE || ""; drawPanel(); }}})),
      masked(md) || md === "outpaint" ? hint : null])(info("prompt", md === "remove" ? "ref.promptHintRemove"
        : md === "outpaint" ? "ref.promptHintOutpaint" : "ref.promptHintInpaint")),
      prompt,
      h("details", {}, h("summary", {"data-i18n": "gen.negative"}), neg)),
    seedRow(src),
    usingLine(s),
    goFoot(md === "upscale" ? "ref.up.go" : "ref.go", goSize)].filter(Boolean));
  translate(box);
  syncGo();
  // 換到修臉、或換了底圖：找這張的臉（第一次載入偵測模型約 2 秒，之後幾十毫秒）
  if (md === "face" && !facesFor(src)) detectFaces();
  if (R.srcP?.id !== src.id) loadSrcParams(src);
}

// seed（隨機／照原圖）與張數
function seedRow(src) {
  return h("div.rfrow", {},
    h("div.seg", {role: "group", "data-i18n-aria": "ref.seed"}, ["random", "source"].map(k => h("button", {
      type: "button", "aria-pressed": String(R.seed === k), "data-i18n": `ref.seed.${k}`,
      "data-i18n-title": `ref.seed.${k}Hint`, disabled: k === "source" && src.seed == null,
      on: {click: () => { R.seed = k; drawPanel(); }}}))),
    h("span.sp"),
    h("div.cnt", {role: "group", "data-i18n-aria": "gen.batch"},
      h("button", {type: "button", "data-i18n-aria": "gen.batch.less", on: {click: () => { R.count = Math.max(1, R.count - 1); drawPanel(); }}}, "−"),
      h("span.mono.rfcount", {text: String(R.count)}),
      h("button", {type: "button", "data-i18n-aria": "gen.batch.more", on: {click: () => { R.count = Math.min(8, R.count + 1); drawPanel(); }}}, "＋")));
}

// 送出鈕、訊息、進度固定在左欄底部：上面再長，按鈕都看得到
function goFoot(label, goSize) {
  return h("div.rffoot", {},
    h("div.go", {},
      h("button.gobtn", {type: "button", id: "rfGo", on: {click: go}},
        h("span.l", {"data-i18n": label}),
        h("kbd", {text: goSize ? t("ref.goSize", goSize) : "—"}))),
    h("p.hint.rfgohint", {id: "rfGoHint", hidden: true, "data-i18n": "ref.needMask"}),
    h("div.run", {},
      h("div.track", {}, h("i", {id: "rfBar"})),
      h("div.runl", {}, h("span", {id: "rfLine"}), h("span.sp"),
        h("button.pill", {type: "button", id: "rfStop", hidden: !R.running, "data-i18n": "gen.cancel", on: {click: cancel}})),
      h("div.runl", {}, h("span.msg", {id: "rfMsg", role: "status"}))));
}

// ── 用指示修圖（Qwen Image 2.1、FLUX.2 klein）──
// 模型：修圖面板選過的；沒選過時生成台正在用的能修圖就用它，不然用資源庫裡第一個（options.editModelOf）。
// 參數在修圖面板選（記在介面偏好的 redit，步數、CFG、取樣器照模型家族各記各的，options.editPatch），
// 沒選過用官方建議；LoRA、品質標籤不帶（那是別的模型的）
async function loadEditModels() {
  R.em = {loading: true, list: []};
  let items = [];
  try { items = (await get("/api/assets?kind=model")).items || []; } catch { /* 讀不到就當成沒有 */ }
  R.em = {list: items.filter(it => EDIT_ARCHS.includes(it.arch))};
  if (mode() === "edit" && $("rfPanel")) drawPanel();
}

const editModel = () => editModelOf(R.em?.list, R.ui?.redit, genSettings());
const editOpts = () => editOptsOf(R.ui?.redit, editModel()?.arch, genSettings());

function setEdit(patch) {
  saveUi({redit: editPatch(R.ui?.redit, editModel()?.arch, patch)});
  drawPanel();
}

function editParams() {
  const o = editOpts();
  return {model: editModel()?.path, steps: o.steps, cfg: o.cfg, sampler: o.sampler, schedule: o.schedule, shift: o.shift};
}

function editPanel(src) {
  const text = h("textarea.rftext", {rows: "4", spellcheck: "false", value: R.editText, "data-i18n-aria": "ref.edit.what",
                                     "data-i18n-placeholder": "ref.edit.placeholder",
                                     on: {input: e => { R.editText = e.target.value; syncGo(); }}});
  const what = h("div.rfsec", {}, ...(([btn, hint]) => [
    h("div.rflab", {}, h("b", {"data-i18n": "ref.edit.what"}), btn), hint])(info("edit", "ref.edit.hint")),
    text);
  // 其他參考圖：正在修的這張是圖1，這裡的是圖2、圖3（使用者要的：多張條件圖，例如把別張的衣服、背景帶進來）
  const refs = h("div.rfsec", {}, ...(([btn, hint]) => [
    h("div.rflab", {}, h("b", {"data-i18n": "ref.edit.refs"}), btn), hint])(info("editr", "ref.edit.refsHint")),
    h("div.rfrefs", {},
      R.editRefs.map((id, i) => h("figure.rfref", {},
        h("img", {src: `/api/thumb?id=${encodeURIComponent(id)}`, alt: t("ref.edit.refN", {n: i + 2}), loading: "lazy"}),
        h("figcaption", {text: t("ref.edit.refN", {n: i + 2})}),
        h("button.rfrefx", {type: "button", title: t("ref.edit.refRemove"), "aria-label": t("ref.edit.refRemove"),
                            on: {click: () => { R.editRefs.splice(i, 1); drawPanel(); }}}, "×"))),
      R.editRefs.length < EDIT_REFS_MAX ? h("button.rfrefadd", {type: "button", "data-i18n": "ref.edit.refAdd", on: {click: pickRef}}) : null));
  const m = editModel();
  if (!m) return [what, refs, h("p.hint.rfusing", {"data-i18n": R.em?.loading ? "ref.edit.loading" : "ref.edit.noModel"})];
  const o = editOpts(), list = R.em?.list || [];
  // 模型清單寫出是哪一種（Qwen 與 klein 的步數差很多，選了參數跟著換）
  const arch = Object.fromEntries(list.map(x => [x.path, x.arch]));
  const label = v => `${stem(v)} · ${ARCH_LABEL[arch[v] || m.arch] || ""}`;
  const row = (key, ...kids) => h("div.rflab", {}, h("span.dim", {"data-i18n": `ref.edit.${key}`}), h("span.sp"), ...kids);
  const num = (key, min, max, step) => h("input.v.btnum", {
    type: "number", min: String(min), max: String(max), step: String(step), value: String(o[key]), "data-i18n-aria": `ref.edit.${key}`,
    on: {change: e => { const v = Number(e.target.value); setEdit({[key]: Number.isFinite(v) && v > 0 ? Math.min(max, Math.max(min, step < 1 ? v : Math.round(v))) : o[key]}); }}});
  const pick = (key, values, cur, label = v => v) => h("select.rfrbsel", {"data-i18n-aria": `ref.edit.${key}`,
    on: {change: e => setEdit({[key]: e.target.value})}},
    ...(values.includes(cur) ? values : [cur, ...values]).map(v => h("option", {value: v, selected: v === cur ? "" : null, text: label(v)})));
  const sizes = ["src", "mp1"].map(k => [k, editSize(src.width, src.height, k)]);
  const same = sizes[0][1].w * sizes[0][1].h <= sizes[1][1].w * sizes[1][1].h;      // 原圖本來就不大：兩個一樣
  return [what, refs,
    h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "ref.edit.params"}), btn), hint])(info("editp", "ref.edit.paramsHint")),
      row("model", list.length > 1 ? pick("model", list.map(x => x.path), m.path, label) : h("span.rfnum", {text: label(m.path)})),
      row("size", h("div.seg", {role: "group", "data-i18n-aria": "ref.edit.size"}, sizes.map(([k, z]) => h("button", {
        type: "button", "aria-pressed": String(o.size === k || (same && k === "src")), disabled: same && k === "mp1",
        "data-i18n": `ref.edit.size.${k}`, title: `${z.w} × ${z.h}`, on: {click: () => setEdit({size: k})}})))),
      row("steps", num("steps", 1, 100, 1)),
      row("cfg", num("cfg", 1, 10, 0.5)),
      row("sampler", pick("sampler", EDIT_SAMPLERS, o.sampler)))];
}

const EDIT_REFS_MAX = 2;          // 同 server/refine.EDIT_REFS_MAX

function pickRef() {
  openPicker({title: t("ref.edit.refPick"), have: [R.id, ...R.editRefs], single: true,
              onAdd: ids => { if (ids[0] && R.editRefs.length < EDIT_REFS_MAX) R.editRefs.push(ids[0]); drawPanel(); }});
}

async function goEdit(src) {
  R.busy = true;
  $("rfGo").disabled = true;
  const msg = $("rfMsg");
  msg.textContent = "";
  const z = editSize(src.width, src.height, editOpts().size);
  const d = await submit({prompt: R.editText.trim(), negative: "", init_id: src.id, edit: true, batch: R.count,
                          width: z.w, height: z.h, seed: R.seed === "source" && src.seed != null ? src.seed : -1,
                          fix_seed: false, loras: [], preset: false, controlnets: [], edit_refs: [...R.editRefs], ...editParams()});
  R.busy = false;
  msg.textContent = d.error || (d.warn ? t("gen.queuedWarn", {n: d.queued, warn: d.warn}) : t("gen.queued", {n: d.queued}));
  msg.dataset.kind = d.error ? "err" : d.warn ? "warn" : "ok";
  syncGo();
}

// ── 照原圖的模型與參數 ──
const useSrc = () => R.ui?.rsrcp !== false;

async function loadSrcParams(src) {
  if (R.srcP?.id === src.id) return;
  R.srcP = {id: src.id, loading: true, over: {}};
  let models = [];
  try { models = (await get("/api/res?kind=model")).items || []; } catch { /* 讀不到清單就當成找不到 */ }
  R.srcP = {id: src.id, ...sourceParams(src, models)};
  if (R.data?.source?.id === src.id) drawPanel();
}

// 這次送出要蓋掉生成台哪些參數（照原圖、而且找得到原圖的模型時才有）
const srcOver = () => (useSrc() && R.srcP?.ok && R.srcP.id === R.data?.source?.id ? R.srcP.over : {});

// 左欄底下那行：這次用的是哪一組設定（可以切換）；退回生成台的時候說原因
function usingLine(s) {
  const P = R.srcP, on = useSrc(), src = on && P?.ok;
  const eff = {...s, ...(src ? {MODEL: P.over.model, SAMPLER: P.over.sampler ?? s.SAMPLER, SCHEDULE: P.over.schedule ?? s.SCHEDULE,
                                  STEPS: P.over.steps ?? s.STEPS, CFG: P.over.cfg ?? s.CFG} : {})};
  const what = {model: stem(eff.MODEL || "") || "—", sampler: [eff.SAMPLER, eff.SCHEDULE].filter(Boolean).join(" · "),
                steps: eff.STEPS ?? "—", cfg: eff.CFG ?? "—"};
  const why = on && P && !P.loading && !P.ok ? t(`ref.src.${P.why}`, {name: P.name || ""}) : "";
  return h("div.rfusing", {},
    h("div.seg", {role: "group", "data-i18n-aria": "ref.src.which"}, ["src", "gen"].map(k => h("button", {
      type: "button", "aria-pressed": String(on === (k === "src")), "data-i18n": `ref.src.${k}`,
      on: {click: () => { saveUi({rsrcp: k === "src"}); drawPanel(); }}}))),
    h("p.hint", {},
      t(src ? "ref.src.using" : "ref.using", what), " ",
      src ? null : h("a", {href: "#generate", "data-i18n": "ref.changeInGen"}),
      why ? h("span.rfwhy", {text: ` ${why}`}) : null,
      src && P.over.model && stem(P.over.model).toLowerCase() !== stem(s.MODEL || "").toLowerCase()
        ? h("span.dim", {text: ` ${t("ref.src.load")}`}) : null));
}

// 底圖的提示詞。CoCo 存圖時把 <lora:…> 拿掉、另外記在 loras，這裡補回去（已經在提示詞裡的不重複加），
// 重畫才會用同一組 LoRA。開著品質標籤生成的圖用使用者自己寫的那份（自動加的重繪時會再加）
export function withLoras(src) {
  let p = src.user_prompt ?? src.prompt ?? "";
  for (const l of src.loras || []) {
    const name = stem(l.file);
    if (name) p = addLora(p, name, Number(l.weight) || 1);
  }
  return p;
}

// 版本跟底圖的形狀一樣（前後對照的滑桿疊得起來）：比例差不到 2%。
// 用指示修圖縮到約 100 萬像素時寬高各自取整，比例會差一點點；擴圖改的比例遠大於這個
const sameShape = (a, b) => Math.abs((a.width / a.height) / (b.width / b.height) - 1) <= 0.02;

// 跟伺服器一樣對齊 16（server/refine.snap），按鈕上顯示的尺寸才會對
const snap = v => Math.max(256, Math.min(4096, Math.round(Number(v) / 16) * 16));

async function go() {
  const src = R.data?.source;
  if (!src || R.busy) return;
  if (mode() === "upscale" && !upDetail()) { goUpscale(); return; }
  if (mode() === "edit") { if (editModel() && R.editText.trim()) goEdit(src); return; }
  const inpaint = masked(), face = mode() === "face", remove = mode() === "remove", hires = mode() === "upscale";
  const outp = mode() === "outpaint" ? outPlan(src) : null;
  if (hires && !upTarget(src)) return;
  if (outp && !outp.ok) return;
  if (inpaint && (!R.mask || R.mask.empty)) { syncGo(); return; }
  const picked = face ? (R.faces?.list || []).filter(f => f.on) : [];
  if (face && !picked.length) { syncGo(); return; }
  R.busy = true;
  $("rfGo").disabled = true;
  const msg = $("rfMsg");
  msg.textContent = "";
  const d = await submit({
    ...sentPrompts(R.prompt ?? withLoras(src), R.negative ?? ""), init_id: src.id, strength: strength(),
    batch: R.count, width: null, height: null,
    seed: R.seed === "source" && src.seed != null ? src.seed : -1, fix_seed: false,
    ...(inpaint ? {mask: R.mask.canvas.toDataURL("image/png"), inpaint_area: remove ? "whole" : area(),
                   inpaint_fill: remove ? "fill" : fillMode(), mask_blur: blur()} : {}),
    ...(inpaint && !remove && R.mask.colored && !layerEmpty(R.mask.color) ? {sketch: R.mask.color.toDataURL("image/png")} : {}),
    ...(face ? {faces: picked.map(f => ({box: f.box, prompt: f.prompt})), face_model: R.faces.model, mask_blur: blur()} : {}),
    ...(hires ? {upscale: {model: upModel(), scale: upScale()}, mask_blur: 32} : {}),
    ...(outp ? {outpaint: outp.sides, mask_blur: 16} : {}),
    ...((outp || inpaint) && R.cnReady && cnOn() ? {cn_repaint: true} : {}),
    ...srcOver(),
  });
  R.busy = false;
  msg.textContent = d.error || (d.warn ? t("gen.queuedWarn", {n: d.queued, warn: d.warn}) : t("gen.queued", {n: d.queued}));
  msg.dataset.kind = d.error ? "err" : d.warn ? "warn" : "ok";
  syncGo();
}

// ── 去背（server/rembg.py、core/bgremove.py）──
// CoCo 認的是架構不是檔案：models/rembg（或模型總資料夾的 rembg）裡認得的 .safetensors 都能選來用。
// 可以下載的模型列在上面，按了才下載（進度、可以取消）；自動下載不了時可以自己下載放進資料夾。
// 結果是透明背景的 PNG，存成這張的新版本
const rbEdge = () => (R.ui?.rbedge === "sharp" ? "sharp" : "soft");
// 選的模型：記得的那個還在就用它，不然用第一個
const rbModel = () => {
  const list = R.rb?.models || [];
  return list.find(m => m.id === R.ui?.rbmodel) || list[0];
};
// 前處理：使用者換過就照換的（每個模型各自記），不然照 CoCo 猜的
const rbPrep = m => (m.arch === "isnet" && (R.ui?.rbprep?.[m.id] === "anime" || R.ui?.rbprep?.[m.id] === "dis") ? R.ui.rbprep[m.id] : m.prep);
const mb = n => `${Math.round(n / 1e6)} MB`;               // 跟 Hugging Face 頁面寫的一樣（1 MB＝100 萬位元組）
const rbName = m => (m.known ? t(`ref.rb.${m.known}`) : m.name);

async function loadRembg() {
  const d = await get("/api/rembg/status").catch(() => null);
  if (!d) return;
  const was = (R.rb?.catalog || []).some(m => m.download?.running);
  R.rb = d;
  const now = d.catalog.some(m => m.download?.running);
  clearTimeout(R.rbTimer);
  if (now) R.rbTimer = setTimeout(loadRembg, 700);
  if (mode() !== "rembg" || !$("rfPanel")) return;
  // 下載中只更新進度（不要整個重畫，展開的說明才不會收起來）；開始、結束時整個重畫
  if (now && was) {
    for (const m of d.catalog) {
      const dl = m.download;
      const bar = $(`rfRbBar-${m.key}`), txt = $(`rfRbTxt-${m.key}`);
      if (bar && dl) bar.style.width = `${Math.round(dl.done / Math.max(1, dl.total) * 100)}%`;
      if (txt && dl) txt.textContent = t("ref.rb.downloading", {done: mb(dl.done), total: mb(dl.total)});
    }
    return;
  }
  drawPanel();
}

async function rbAction(path, key) {
  const d = await post(path, {key}).catch(e => ({error: e.message}));
  if (d.error) { const m = $("rfMsg"); if (m) { m.textContent = d.error; m.dataset.kind = "err"; } }
  await loadRembg();
}

function copyText(text, btn) {
  navigator.clipboard?.writeText(text).then(() => { btn.textContent = t("ref.rb.copied"); }).catch(() => {});
}

const copyRow = text => h("div.rfrbcopy", {}, h("code", {text}),
  h("button.mini", {type: "button", "data-i18n": "ref.rb.copy", on: {click: e => copyText(text, e.currentTarget)}}));

// 可以下載的模型：還沒有的才顯示下載鈕
function catalogCard(m) {
  const dl = m.download, running = dl?.running;
  const pct = running ? Math.round(dl.done / Math.max(1, dl.total) * 100) : 0;
  return h("div.rfrb", {},
    h("div.rflab", {}, h("b", {"data-i18n": `ref.rb.${m.key}`}), h("span.sp"),
      h("span.tag", {"data-i18n": m.installed ? "ref.rb.installed" : "ref.rb.notInstalled"})),
    h("p.hint", {"data-i18n": `ref.rb.${m.key}Hint`}),
    h("p.hint.rfrbmeta", {}, `${mb(m.bytes)} · ${m.license} · `, h("a", {href: m.page, target: "_blank", rel: "noopener", "data-i18n": "ref.rb.source"})),
    running ? h("div.run", {},
      h("div.track", {}, h("i", {id: `rfRbBar-${m.key}`, style: `width:${pct}%`})),
      h("div.runl", {}, h("span", {id: `rfRbTxt-${m.key}`, text: t("ref.rb.downloading", {done: mb(dl.done), total: mb(dl.total)})}),
        h("span.sp"), h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: () => rbAction("/api/rembg/cancel", m.key)}}))) : null,
    !m.installed && !running && m.part && !dl?.error ? h("p.hint", {text: t("dl.partial", {pct: partPct(m.part, m.bytes)})}) : null,
    !m.installed && !running ? h("button.pill.primary", {type: "button",
                                                         text: m.part ? t("ref.rb.downloadRest", {size: mb(m.bytes - m.part)}) : t("ref.rb.download", {size: mb(m.bytes)}),
                                                         on: {click: () => rbAction("/api/rembg/download", m.key)}}) : null,
    dl?.error && !running ? h("p.hint.rferr", {text: dl.error}) : null,
    // 自動下載不了：自己下載、放進資料夾（檔名不用改）
    !m.installed ? h("details.rfexact", {open: !!dl?.error},
      h("summary", {"data-i18n": "ref.rb.manual"}),
      h("p.hint", {"data-i18n": "ref.rb.manualHint"}),
      copyRow(m.url), copyRow(R.rb.dir),
      h("button.mini", {type: "button", "data-i18n": "ref.rb.recheck", on: {click: loadRembg}})) : null);
}

// 選好的模型：架構、前處理、檔案位置、刪除
function modelCard(m) {
  const prep = rbPrep(m);
  return h("div.rfrb", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ref.rb.model"}), h("span.sp"),
      R.rb.models.length > 1
        ? h("select.rfrbsel", {"data-i18n-aria": "ref.rb.model", on: {change: e => { saveUi({rbmodel: e.target.value}); drawPanel(); }}},
          R.rb.models.map(x => h("option", {value: x.id, selected: x.id === m.id ? "" : null, text: rbName(x)})))
        : h("span", {text: rbName(m)})),
    h("p.hint.rfrbmeta", {text: `${t(`ref.rb.arch.${m.arch}`)}${m.variant ? `（${m.variant}）` : ""} · ${mb(m.bytes)}`}),
    // 前處理只有 IS-Net 要選（不同來源的做法不同）；BiRefNet 只有官方那一種
    m.arch === "isnet" ? h("div.rflab", {}, h("span.dim", {"data-i18n": "ref.rb.prep"}), h("span.sp"),
      h("div.seg", {role: "group", "data-i18n-aria": "ref.rb.prep"}, ["anime", "dis"].map(k => h("button", {
        type: "button", "aria-pressed": String(prep === k), "data-i18n": `ref.rb.prep.${k}`, "data-i18n-title": `ref.rb.prep.${k}Hint`,
        on: {click: () => { saveUi({rbprep: {...(R.ui?.rbprep || {}), [m.id]: k}}); drawPanel(); }}})))) : null,
    h("details.rfexact", {},
      h("summary", {"data-i18n": "ref.rb.manage"}),
      h("p.hint.mono", {text: m.path}),
      m.own ? h("button.mini.danger", {type: "button", text: t("ref.rb.remove", {size: mb(m.bytes)}), on: {click: async () => {
        const ok = await confirmDialog({title: t("ref.rb.removeTitle"), body: [h("p", {text: t("ref.rb.removeBody", {name: m.id})})], ok: t("ref.rb.removeOk")});
        if (ok) rbAction("/api/rembg/remove", m.id);
      }}}) : h("p.hint", {"data-i18n": "ref.rb.notOwn"})));
}

// 自己的模型：放哪、認得哪些架構、哪些檔案認不得（為什麼）
function ownModels() {
  const sk = R.rb.skipped || [];
  return h("details.rfexact.rfrbown", {open: sk.length > 0},
    h("summary", {}, t("ref.rb.own"), sk.length ? h("span.tag", {text: t("ref.rb.skippedN", {n: sk.length})}) : null),
    h("p.hint", {text: t("ref.rb.ownHint", {arches: R.rb.arches.map(a => t(`ref.rb.arch.${a}`)).join("、")})}),
    ...R.rb.dirs.map(copyRow),
    sk.length ? h("ul.rfrbskip", {}, sk.map(x => h("li", {}, h("b", {text: x.name}), " ", h("span.dim", {text: t(`ref.rb.why.${x.why}`)})))) : null,
    h("button.mini", {type: "button", "data-i18n": "ref.rb.recheck", on: {click: loadRembg}}));
}

function rembgPanel() {
  if (!R.rb) { loadRembg(); return [h("p.dim", {"data-i18n": "wall.loading"})]; }
  const m = rbModel();
  return [
    h("div.rfsec", {}, h("p.hint", {"data-i18n": "ref.rb.intro"})),
    m ? modelCard(m) : null,
    ...R.rb.catalog.filter(c => !c.installed).map(catalogCard),
    ownModels(),
    h("div.rfsec", {}, ...(([btn, hint]) => [
      h("div.rflab", {}, h("b", {"data-i18n": "ref.rb.edge"}), btn, h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ref.rb.edge"}, ["soft", "sharp"].map(k => h("button", {
          type: "button", "aria-pressed": String(rbEdge() === k), "data-i18n": `ref.rb.edge.${k}`,
          on: {click: () => { saveUi({rbedge: k}); drawPanel(); }}})))),
      hint])(info("rbedge", `ref.rb.edge.${rbEdge()}Hint`))),
    h("div.rffoot", {},
      h("div.go", {}, h("button.gobtn", {type: "button", id: "rfGo", on: {click: goRembg}}, h("span.l", {"data-i18n": "ref.rb.go"}))),
      h("p.hint.rfgohint", {id: "rfGoHint", hidden: true}),
      h("div.run", {}, h("div.runl", {}, h("span.msg", {id: "rfMsg", role: "status"}))))].filter(Boolean);
}

async function goRembg() {
  const src = R.data?.source, m = rbModel();
  if (!src || !m || R.busy) return;
  R.busy = true;
  syncGo();
  const msg = $("rfMsg");
  if (msg) { msg.textContent = t("ref.rb.working"); msg.dataset.kind = ""; }
  const d = await post("/api/refine/rembg", {id: src.id, model: m.id, edge: rbEdge(), prep: rbPrep(m)}).catch(e => ({error: e.message}));
  R.busy = false;
  if (d.error) {
    if (msg) { msg.textContent = d.error; msg.dataset.kind = "err"; }
    loadRembg();
    syncGo();
    return;
  }
  const v = await get(`/api/refine/versions?id=${R.id}`).catch(() => null);
  if (v && !v.error) R.data = v;
  if (d.id) { R.pick = d.id; R.view = "compare"; }
  drawStage();
  const m2 = $("rfMsg");
  if (m2) { m2.textContent = t("ref.rb.done"); m2.dataset.kind = "ok"; }
  syncGo();
}

// ── 高清化（server/upscalers.py、core/upscale.py、pipeline._hires）──
// 放大模型（模型總資料夾的 ESRGAN 之類）或一般放大，倍數 1～4；補細節：放大後分塊用低修改幅度重畫。
// 1 倍是大小不變（補細節在放大的畫布上做再縮回，pipeline._hires）；只放大又沒選模型時圖不會變，按鈕停用
const upDetail = () => R.ui?.rupdetail !== false;
const upNoop = () => mode() === "upscale" && !upDetail() && upScale() === 1 && !upModel();
const upScale = () => (R.up?.scales || [2]).includes(Number(R.ui?.rupscale)) ? Number(R.ui.rupscale) : 2;
const upModel = () => {
  const names = (R.up?.items || []).map(m => m.name);
  const want = R.ui?.rupmodel;
  if (want === "") return "";                            // 選了「一般放大」
  return names.includes(want) ? want : names[0] || "";
};
const upSize = (src, s) => ({w: Math.round(src.width * s / 16) * 16, h: Math.round(src.height * s / 16) * 16});
const upTarget = src => {
  const z = upSize(src, upScale());
  return Math.max(z.w, z.h) <= (R.up?.max || 4096) ? z : null;
};

async function loadUpscalers() {
  const d = await get("/api/refine/upscalers").catch(() => null);
  R.up = d || {items: [], scales: [1, 1.5, 2, 3, 4], max: 4096};
  if (mode() === "upscale") drawPanel();
}

function upscalePanel(src) {
  if (!R.up) { loadUpscalers(); return [h("p.dim", {"data-i18n": "wall.loading"})]; }
  const s = upScale(), z = upTarget(src);
  return [
    h("div.rfsec", {},
      h("div.rflab", {}, h("b", {"data-i18n": "ref.up.model"}), h("span.sp"),
        h("select.rfrbsel", {"data-i18n-aria": "ref.up.model", on: {change: e => { saveUi({rupmodel: e.target.value}); drawPanel(); }}},
          ...R.up.items.map(m => h("option", {value: m.name, selected: upModel() === m.name ? "" : null, text: m.name})),
          h("option", {value: "", selected: upModel() === "" ? "" : null, "data-i18n": "ref.up.none"}))),
      R.up.items.length ? null : h("p.hint", {"data-i18n": "ref.up.noModels"})),
    h("div.rfsec", {},
      h("div.rflab", {}, h("b", {"data-i18n": "ref.up.scale"}), h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ref.up.scale"}, R.up.scales.map(k => {
          const zz = upSize(src, k), big = Math.max(zz.w, zz.h) > R.up.max;
          return h("button", {type: "button", "aria-pressed": String(s === k), disabled: big, text: `${k}×`,
                              title: big ? t("ref.up.tooBig", {max: R.up.max}) : `${zz.w} × ${zz.h}`,
                              on: {click: () => { saveUi({rupscale: k}); drawPanel(); }}});
        }))),
      h("p.hint", {text: !z ? t("ref.up.tooBig", {max: R.up.max})
        : s === 1 ? t("ref.up.same", {w: z.w, h: z.h}) : t("ref.up.size", {w: src.width, h: src.height, tw: z.w, th: z.h})})),
    h("div.cell.swcell", {},
      h("span.swtext", {}, h("b", {id: "rfUpDetailLbl", "data-i18n": "ref.up.detail"}),
        h("span.hint", {"data-i18n": upDetail() ? "ref.up.detailOn" : "ref.up.detailOff"})),
      h("button.switch", {type: "button", role: "switch", "aria-labelledby": "rfUpDetailLbl", "aria-checked": String(upDetail()),
                          on: {click: () => { saveUi({rupdetail: !upDetail()}); drawPanel(); }}}, h("i")))];
}

// 只放大：按鈕在左欄底部（跟重繪鈕同一個位置）
function upscaleFoot(src) {
  const z = upTarget(src);
  return h("div.rffoot", {},
    h("div.go", {}, h("button.gobtn", {type: "button", id: "rfGo", on: {click: goUpscale}},
      h("span.l", {"data-i18n": "ref.up.goPlain"}), h("kbd", {text: z ? t("ref.goSize", z) : "—"}))),
    h("p.hint.rfgohint", {id: "rfGoHint", hidden: true}),
    h("div.run", {}, h("div.runl", {}, h("span.msg", {id: "rfMsg", role: "status"}))));
}

async function goUpscale() {
  const src = R.data?.source;
  if (!src || R.busy || !upTarget(src) || upNoop()) return;
  R.busy = true;
  syncGo();
  const msg = $("rfMsg");
  if (msg) { msg.textContent = t("ref.up.working"); msg.dataset.kind = ""; }
  const d = await post("/api/refine/upscale", {id: src.id, model: upModel(), scale: upScale()}).catch(e => ({error: e.message}));
  R.busy = false;
  if (d.error) {
    if (msg) { msg.textContent = d.error; msg.dataset.kind = "err"; }
    syncGo();
    return;
  }
  const v = await get(`/api/refine/versions?id=${R.id}`).catch(() => null);
  if (v && !v.error) R.data = v;
  if (d.id) { R.pick = d.id; R.view = "compare"; }
  drawStage();
  const m2 = $("rfMsg");
  if (m2) { m2.textContent = t("ref.up.done"); m2.dataset.kind = "ok"; }
  syncGo();
}

// 版本的說明：重繪的寫修改幅度與 seed；去背的沒有這兩個
const pickInfo = it => (it.rembg ? t("ref.rb.pickInfo")
  : it.edit ? t("ref.edit.pickInfo", {text: String(it.user_prompt ?? it.prompt ?? "").slice(0, 80), seed: it.seed ?? "—"})
  : it.upscale && !it.upscale.detail ? t(Number(it.upscale.scale) === 1 ? "ref.up.pickInfoSame" : "ref.up.pickInfo", {s: it.upscale.scale})
  : t("ref.pickInfo", {s: Number(it.strength ?? 0).toFixed(2), seed: it.seed ?? "—"}));

// ── 右邊：前後對照（或畫遮罩）、版本列 ──
function drawStage() {
  const box = $("rfStage");
  const src = R.data?.source;
  if (!src) { box.replaceChildren(h("div.rfnone", {}, h("p.dim", {"data-i18n": "ref.stageEmpty"}))); translate(box); return; }
  const pick = R.data.items.find(it => it.id === R.pick);
  const ar = `--ar:${(src.width / src.height || 0.75).toFixed(5)}`;
  const inpaint = masked();
  const editing = inpaint && (R.view === "mask" || !pick);
  // 修臉：還沒有結果、或切回「看偵測到的臉」時，圖上框出每張臉（點框決定修不修）
  const faceView = mode() === "face" && (R.view !== "compare" || !pick);
  // 擴圖：還沒有結果、或切回「看擴的範圍」時，畫出加大後的畫布（格子是要補的地方）
  const outView = mode() === "outpaint" && (R.view !== "compare" || !pick);
  let stage, tools = null;
  if (outView) {
    const p = outPlan(src), W = p.W, H = p.H;
    const pct = (v, of) => `${(v / of * 100).toFixed(3)}%`;
    stage = h("div.rfba.rfsolo.rfoutpv", {style: `--ar:${(W / H).toFixed(5)}`},
      h("div.rfz.rfcheck", {}, h("img.rfoutsrc", {src: full(src), alt: "", draggable: "false",
        style: `left:${pct(p.sides.left, W)};top:${pct(p.sides.top, H)};width:${pct(src.width, W)};height:${pct(src.height, H)}`})));
  } else if (faceView) {
    const F = facesFor(src) ? R.faces : null;
    const [fw, fh] = F?.size || [src.width, src.height];
    const pct = (v, of) => `${(v / of * 100).toFixed(3)}%`;
    stage = h("div.rfba.rfsolo.rffaces", {style: ar},
      h("div.rfz", {}, h("img", {src: full(src), alt: "", draggable: "false"}),
        (F?.list || []).map((f, i) => h("button.rfbox", {
          type: "button", "aria-pressed": String(f.on), title: t("ref.face", {n: i + 1}), "data-manual": f.manual ? "" : null,
          style: `left:${pct(f.box[0], fw)};top:${pct(f.box[1], fh)};width:${pct(f.box[2] - f.box[0], fw)};height:${pct(f.box[3] - f.box[1], fh)}`,
          on: {click: e => {
            f.on = !f.on;
            e.currentTarget.setAttribute("aria-pressed", String(f.on));
            markFaces();
            syncGo();
          }}}, h("span", {text: String(i + 1)})))));
    wireFaceDraw(stage.querySelector(".rfz"), src, [fw, fh]);
  } else if (editing) {
    const m = ensureMask(src);
    stage = h("div.rfba.rfpaint", {style: ar},
      h("div.rfz", {}, h("img", {src: full(src), alt: "", draggable: "false"}), m.canvas,
        mode() === "inpaint" ? m.color : null, m.shape),
      h("div.rfcursor", {id: "rfCursor", hidden: true}));
    tools = h("div.rftools", {id: "rfTools", role: "toolbar", "data-i18n-aria": "ref.maskTools"},
      h("div.seg", {role: "group", "data-i18n-aria": "ref.tool"}, toolList().map(k => h("button", {
        type: "button", "data-tool": k, "aria-pressed": String(tool() === k), "data-i18n": `ref.tool.${k}`,
        "data-i18n-title": `ref.tool.${k}Key`, on: {click: () => setTool(k)}}))),
      h("div.seg", {role: "group", "data-i18n-aria": "ref.paintMode"}, ["paint", "erase"].map(k => h("button", {
        type: "button", "data-erase": k, "aria-pressed": String(erasing() === (k === "erase")), "data-i18n": `ref.tool.${k}`,
        "data-i18n-title": "ref.tool.eraseKey", on: {click: () => setErase(k === "erase")}}))),
      h("label.rfbrush", {id: "rfBrushSize", hidden: !brushLike(tool())},
        h("span", {"data-i18n": "ref.brush"}),
        h("input", {type: "range", id: "rfBrush", min: String(BRUSH[0]), max: String(BRUSH[1]), value: String(brush()),
                    "data-i18n-aria": "ref.brush", on: {input: e => setBrush(Number(e.target.value))}}),
        h("span.mono", {id: "rfBrushNum", text: `${brush()}px`})),
      h("label.rfbrush", {id: "rfColorBox", hidden: tool() !== "color", "data-i18n-title": "ref.colorHint"},
        h("span", {"data-i18n": "ref.color"}),
        h("input.rfswatch", {type: "color", id: "rfColor", value: paintColor(), "data-i18n-aria": "ref.color",
                             on: {input: e => setColor(e.target.value)}})),
      h("label.rfbrush", {id: "rfTolBox", hidden: tool() !== "wand", "data-i18n-title": "ref.tolHint"},
        h("span", {"data-i18n": "ref.tol"}),
        h("input", {type: "range", min: String(TOL[0]), max: String(TOL[1]), value: String(tolerance()), "data-i18n-aria": "ref.tol",
                    on: {input: e => setTolerance(Number(e.target.value))}}),
        h("span.mono", {id: "rfTolNum", text: String(tolerance())})),
      h("div.rfadj", {},
        h("button.pill", {type: "button", "data-i18n": "ref.invert", "data-i18n-title": "ref.invertHint", on: {click: invertMask}}),
        h("button.pill", {type: "button", "data-i18n": "ref.grow", "data-i18n-title": "ref.growHint", on: {click: () => growMask(GROW)}}),
        h("button.pill", {type: "button", "data-i18n": "ref.shrink", "data-i18n-title": "ref.shrinkHint", on: {click: () => growMask(-GROW)}})),
      h("button.pill", {type: "button", "data-i18n": "ref.undo", "data-i18n-title": "ref.undoKey", on: {click: undoMask}}),
      h("button.pill", {type: "button", "data-i18n": "ref.clear", on: {click: clearMask}}),
      h("span.sp"),
      pick ? h("button.pill", {type: "button", "data-i18n": "ref.seeResult", on: {click: () => { R.view = "compare"; drawStage(); }}}) : null);
  } else if (pick && !sameShape(pick, src)) {
    // 形狀跟原圖不一樣（擴圖出來的）：前後對照的滑桿疊不起來，單獨顯示
    stage = h("div.rfba.rfsolo", {style: `--ar:${(pick.width / pick.height).toFixed(5)}`},
      h("div.rfz.rfcheck", {}, h("img", {src: full(pick), alt: ""})));
  } else if (pick) {
    stage = h("div.ba.rfba", {style: `${ar};--x:${R.x}%`},
      h("div.rfz.rfcheck", {}, h("img", {src: full(pick), alt: ""})),
      h("div.layer.before", {}, h("div.rfz", {}, h("img", {src: full(src), alt: ""}))),
      h("span.lab", {style: "left:10px", "data-i18n": "ref.before"}),
      h("span.lab", {style: "right:10px", "data-i18n": "ref.after"}),
      h("div.line"), h("div.knob", {text: "⇆"}),
      h("input", {type: "range", min: "0", max: "100", step: "0.5", value: String(R.x), "data-i18n-aria": "ref.divider",
                  on: {input: e => { R.x = Number(e.target.value); e.target.parentElement.style.setProperty("--x", `${R.x}%`); }}}));
  } else {
    stage = h("div.rfba.rfsolo", {style: ar}, h("div.rfz", {}, h("img", {src: full(src), alt: ""})));
  }
  const [maskBtn, maskHint] = info("mask", "ref.maskHint");
  box.replaceChildren(...[          // 前後對照時沒有遮罩工具列（tools 是 null）
    h("div.rfhd", {},
      faceView ? h("span.dim", {"data-i18n": "ref.faceHd"})
        : outView ? h("span.dim", {"data-i18n": "ref.out.hd"})
        : editing ? h("span.dim", {}, t("ref.maskShort"), " ", maskBtn)
        : pick ? h("span.dim", {text: pickInfo(pick)})
        : h("span.dim", {"data-i18n": mode() === "edit" ? "ref.edit.noVersions" : "ref.noVersions"}),
      h("span.sp"),
      inpaint && !editing ? h("button.pill", {type: "button", "data-i18n": "ref.editMask", on: {click: () => { R.view = "mask"; drawStage(); }}}) : null,
      mode() === "face" && !faceView ? h("button.pill", {type: "button", "data-i18n": "ref.faceView", on: {click: () => { R.view = "faces"; drawStage(); }}}) : null,
      faceView && pick ? h("button.pill", {type: "button", "data-i18n": "ref.seeResult", on: {click: () => { R.view = "compare"; drawStage(); }}}) : null,
      mode() === "outpaint" && !outView ? h("button.pill", {type: "button", "data-i18n": "ref.out.view", on: {click: () => { R.view = "outpaint"; drawStage(); }}}) : null,
      outView && pick ? h("button.pill", {type: "button", "data-i18n": "ref.seeResult", on: {click: () => { R.view = "compare"; drawStage(); }}}) : null,
      pick && !editing && !faceView && !outView ? h("button.pill", {type: "button", "data-i18n": "ref.continue", "data-i18n-title": "ref.continueHint",
                                           on: {click: () => openRefine(pick.id)}}) : null,
      pick && !editing && !faceView && !outView ? h("button.pill", {type: "button", "data-i18n": "lib.edit", "data-i18n-title": "ed.fromHint",
                                           on: {click: () => openEditor(pick.id)}}) : null,
      pick && !editing && !faceView && !outView ? h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.toCompare",
                                           on: {click: e => addToCompare([src.id, pick.id], e.currentTarget, r => {
                                             const m = $("rfMsg");
                                             if (m) { m.textContent = r.error || t("cmp.added", {n: r.added, count: r.count}); m.dataset.kind = r.error ? "err" : "ok"; }
                                           })}}) : null),
    editing ? maskHint : null,
    tools,
    h("div.rfview", {id: "rfView", "data-space": R.space ? "" : null}, stage),
    h("div.rfstrip", {id: "rfStrip", role: "listbox", "data-i18n-aria": "ref.versions"})].filter(Boolean));
  translate(box);
  wireZoom($("rfView"), stage, src);
  drawStrip();
  syncGo();
}

function drawStrip() {
  const box = $("rfStrip");
  if (!box || !R.data) return;
  const src = R.data.source;
  const label = it => (it.rembg ? t("ref.rembgShort")
    : it.edit ? t("ref.edit.short")
    : it.upscale ? `${t("ref.up.short", {s: it.upscale.scale})}${it.upscale.detail ? ` ${Number(it.strength ?? 0).toFixed(2)}` : ""}`
    : `${it.outpaint ? `${t("ref.out.short")} ` : it.faces ? `${t("ref.faceShort")} ` : it.inpaint ? `${t("ref.inpaintShort")} ` : ""}${Number(it.strength ?? 0).toFixed(2)}`);
  const card = (it, isSrc) => h("button.rfv", {
    type: "button", role: "option", "aria-selected": String(isSrc ? !R.pick : it.id === R.pick),
    title: isSrc ? t("ref.before") : pickInfo(it),
    on: {click: () => { R.pick = isSrc ? 0 : it.id; R.view = isSrc ? "mask" : "compare"; drawStage(); }}},
    h("img", {src: thumb(it), alt: "", loading: "lazy"}),
    h("span", {text: isSrc ? t("ref.before") : label(it)}));
  const pending = R.running ? Array.from({length: Math.min(R.left, 8)}, () => h("div.rfv.dev", {}, h("span", {"data-i18n": "ref.developing"}))) : [];
  box.replaceChildren(card(src, true), ...pending, ...R.data.items.map(it => card(it, false)));
  translate(box);
}

document.addEventListener("keydown", e => {
  if (location.hash !== "#refine" || e.defaultPrevented) return;
  // Ctrl+Enter 在精修頁是「重繪」（生成台自己那組在 generate.js）
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    e.stopImmediatePropagation();
    if (mode() === "rembg") goRembg(); else if (mode() === "upscale" && !upDetail()) goUpscale(); else go();
    return;
  }
  const typing = e.target.closest?.("input:not([type=range]), textarea, select, [contenteditable]");
  if (!$("rfView") || typing) return;
  if (e.key === " ") {
    e.preventDefault();                  // 不捲頁、也不要按到剛剛點過的按鈕
    if (!R.space) { R.space = true; $("rfView").toggleAttribute("data-space", true); }
    return;
  }
  if (e.key === "0" && !e.ctrlKey && !e.metaKey && !e.altKey) { R.zoomApply?.({...FIT}); return; }
  // 畫遮罩的快捷鍵
  if (!$("rfTools")) return;
  const k = e.key.toLowerCase();
  if (k === "z" && (e.ctrlKey || e.metaKey) && !e.shiftKey) { e.preventDefault(); undoMask(); }
  else if (e.ctrlKey || e.metaKey || e.altKey) return;
  else if (R.mask?.poly && e.key === "Enter") { e.preventDefault(); closePoly(); }
  else if (R.mask?.poly && e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); cancelPoly(); }
  else if (R.mask?.poly && e.key === "Backspace") { e.preventDefault(); undoMask(); }
  else if (k === "[" || k === "]") { e.preventDefault(); setBrush(brush() * (k === "]" ? 1.25 : 0.8)); }
  else if (TOOL_KEY[k] && toolList().includes(TOOL_KEY[k])) setTool(TOOL_KEY[k]);
  else if (k === "e") setErase(!erasing());
  else if (k === "i") invertMask();
}, true);

const spaceUp = () => {
  if (!R.space) return;
  R.space = false;
  $("rfView")?.removeAttribute("data-space");
};
document.addEventListener("keyup", e => {
  if (e.key !== " " || !R.space) return;
  e.preventDefault();
  spaceUp();
}, true);
addEventListener("blur", spaceUp);
