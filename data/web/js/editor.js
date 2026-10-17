// 06 輸出：圖層編輯器（使用者筆記：浮水印、裁切、翻轉、旋轉、輸出尺寸與檔案類型、滴管、筆刷…
// 基本功都在這裡做好，不必另外開 Krita、GIMP）。第 1 輪：圖層、復原、裁切、翻轉、旋轉、輸出；
// 第 2 輪：筆刷、橡皮擦、滴管（Alt＋點一下）、模糊與銳利筆刷。
// 第 3 輪：選取範圍（方形、橢圓、套索、多邊形、魔術棒；加入、減去、反選、擴張收縮、羽化）、
// 選取範圍的刪除／填滿／拷貝成新圖層／裁切、變形（移動、縮放、旋轉、翻轉）、複製貼上（也能貼外面的圖）。
// 有選取範圍時，筆刷、橡皮擦、模糊、銳利都只畫得到選取的地方。
// 第 4 輪：調整（亮度、對比、飽和度、色相；即時預覽、套用才改，有選取時只調選取的地方）、
// 仿製印章（Alt＋點一下設定取樣點；可以取所有圖層畫在空白圖層上；對齊／每筆從頭）。
// 第 5 輪：文字圖層（點畫面新增、點文字再改、拖著移；字型、大小、顏色、粗斜體、對齊、描邊、陰影；
// 用筆刷、變形等改過像素就變成一般圖層）、浮水印（文字或 logo、九宮格或平鋪，做成一層「浮水印」）。
// 批次輸出在 batch.js（圖庫選好幾張一次輸出）。
// 之後：圖層可以「鎖定透明」（只改顏色、不改透明度，方便只幫有東西的地方上色）；
// 工作檔可以存成 .kra（Krita 打得開），也能從 .kra 開回來接著編輯（server/kra.py）。
//
// 圖層：一層一張 canvas（跟底圖一樣大），由下往上疊；每層有名字、顯示／隱藏、不透明度。
// 新增空白圖層、複製圖層（包括複製底圖）、刪除、上下移、向下合併——改壞了刪掉那層就好。
// 整張的動作（裁切、翻轉、旋轉）每層都換成新的 canvas，舊的留著給復原用，不必另外複製像素。
//
// 工作檔（server/edit.py）：改了就自動存（只送換過的圖層），關掉再開同一張圖時問要不要接著做。
// 輸出：看得到的圖層合併成一張，照設定的尺寸、檔案類型存成新檔；存進圖庫一律保留生成參數，
// 匯出到別的資料夾可以選要不要保留。

import {get, getBlob, post} from "./api.js";
import {$, h, icon} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {confirmDialog} from "./confirm.js";
import {pickFolder} from "./setup.js";
import {FIT, clampPan, panBy, pixelPercent, zoomAt} from "./zoom.js";
import {saveQueue} from "./inflight.js";
import {HANDLES, RATIOS, alphaBox, cropFromDrag, dabs, edgePixels, gridOf, hitCrop, inPoly, lockSize, moveCrop, nearPoint, quadOf,
        resizeCrop, roundCrop, sharpen, shiftPts, stamp, turnPts, xfCorners, xfDrag, xfHit, xfMap, adjustMatrix, adjustPixels} from "./editops.js";
import {meshSupported, renderMesh} from "./meshgl.js";
import {FONTS, drawText} from "./textdraw.js";
import {WM_POS} from "./editops.js";
import {clearLogo, drawWatermark, loadLogo, logoNow, logoVer, setLogo, wmReady, wmset} from "./watermark.js";
import {floodSelect, grow, invert, nearFirst, tracePath} from "./maskops.js";

const HIST_MAX = 40;
const FORMATS = ["png", "jpg", "webp"];
// 工具：畫的都畫在選中的那層上；一筆一步復原
const PAINT = ["brush", "eraser", "blur", "sharpen", "clone", "picker"];
const TOOLS = ["view", "crop", "select", "transform", "adjust", "text", ...PAINT];
const TOOL_KEY = {v: "view", h: "view", c: "crop", m: "select", t: "transform", a: "adjust", x: "text", b: "brush", e: "eraser", r: "blur",
                  s: "sharpen", k: "clone", i: "picker"};
// 調整的項目：[鍵, 最小, 最大, 單位]
const ADJ = [["bri", -100, 100, ""], ["con", -100, 100, ""], ["sat", -100, 100, ""], ["hue", -180, 180, "°"]];
// 選取的形狀（跟精修畫遮罩的一樣）、選取的方式（新的、加入、減去）
const SHAPES = ["rect", "ellipse", "lasso", "poly", "wand"];
const SEL_MODES = ["new", "add", "sub"];
const WHITE = [255, 255, 255];
const SIZE = [1, 500];
// 筆刷設定（介面偏好 ebrush／efx，下次打開還在）：顏色、大小（底圖的像素）、不透明度、硬度（%）；模糊半徑、銳利強度
const bset = () => ({color: "#d04848", size: 24, opacity: 100, hard: 80, pSize: true, pOpacity: false, ...(E.ui?.ebrush || {})});
const fxset = () => ({blur: 6, sharp: 60, ...(E.ui?.efx || {})});
// 選取的設定（介面偏好 esel）：形狀、魔術棒的容許度／只選相連的／看所有圖層還是這一層、擴張收縮羽化的幅度
const selset = () => ({shape: "rect", tol: 32, contig: true, sample: "all", amt: 4, ...(E.ui?.esel || {})});
// 仿製印章（介面偏好 eclone）：對齊（每一筆跟第一筆保持同樣的距離）、取樣看所有圖層還是這一層
const cloneset = () => ({aligned: true, sample: "all", ...(E.ui?.eclone || {})});
// 文字（介面偏好 etext：上次用的樣式，新的文字照這個）
const TEXT_STYLE = ["font", "size", "color", "bold", "italic", "align", "stroke", "scolor", "shadow"];
const tset = () => ({font: FONTS[0], size: 64, color: "#ffffff", bold: false, italic: false, align: "left", stroke: 0, scolor: "#000000",
                     shadow: false, ...(E.ui?.etext || {})});
const E = {ui: null, src: null, pid: "", w: 0, h: 0, layers: [], active: "", hist: [], fut: [],
           tool: "view", crop: null, ratio: "free", zoom: {...FIT}, zoomApply: null, space: false,
           onDisk: new Map(),              // 圖層 id → 伺服器上存的是哪一張 canvas（換過的才要再送）
           saveTimer: null, saving: false, savedAt: 0, out: null, lock: true, msg: null, loadSeq: 0, rev: 0, pending: false,
           sel: null,                      // 選取範圍：跟圖一樣大的 canvas（看透明度，羽化過是半透明）；null＝沒選取
           selMode: "new", sdrag: null, poly: null, xf: null, clip: null, ants: null, antPhase: 0,
           adj: null,                      // 調整中：預覽那層、拉桿的值
           clone: {src: null, off: null, mark: null},    // 仿製印章：取樣點、目前的位移（對齊時一直沿用）、畫面上取樣圈的位置
           txt: null,                      // 正在改的文字圖層（這一段編輯結束才記一步復原）
           fonts: null,                    // 電腦裡的字型（按「列出電腦裡的字型」之後才有）
           autoResume: false};             // 剛從 .kra 開回來：下一次載入直接接著做，不再問

export function initEditor(ui) { E.ui = ui; }

const saveUi = patch => { Object.assign(E.ui, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const newId = () => Array.from(crypto.getRandomValues(new Uint8Array(6)), b => b.toString(16).padStart(2, "0")).join("");
const canvasOf = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
const cur = () => E.layers.find(L => L.id === E.active) || E.layers[E.layers.length - 1];
const fmt = () => (FORMATS.includes(E.ui?.eformat) ? E.ui.eformat : "png");
const toLibrary = () => E.ui?.edest !== "folder";

// ── 打開 ──
export function openEditor(id) {
  saveUi({esrc: Number(id)});
  if (location.hash !== "#output") location.hash = "#output";
  else showEditor();
}

export function editorScreen() {
  return h("div.rfm.edm", {},
    h("aside.card.rail.rfpanel.edpanel", {id: "edPanel", "aria-labelledby": "edTitle"}),
    h("section.card.rfstage.edstage", {id: "edStage"}));
}

export async function showEditor() {
  if (!$("edPanel")) return;
  const id = Number(E.ui?.esrc) || 0;
  if (!id) { E.src = null; await drawEmpty(); return; }
  if (E.src?.id === id && E.layers.length) { drawPanel(); drawStage(); return; }   // 從別頁回來：接著編
  await load(id);
}

async function bitmapCanvas(url, w, h) {
  const bmp = await createImageBitmap(await getBlob(url));
  const c = canvasOf(w || bmp.width, h || bmp.height);
  c.getContext("2d").drawImage(bmp, 0, 0);
  return c;
}

// seq：載入到一半又換了一張（例如還在問要不要接著做時又開了別張），舊的那次就不要再動畫面與狀態。
// 開的是哪張（E.src）、圖層、工作檔 id 在最後一起換，自動存檔才不會把這張的圖層存進別張的工作檔
async function load(id) {
  const seq = ++E.loadSeq;
  const stale = () => seq !== E.loadSeq;
  // 上一張還有沒存的改動（或正在存）：等到最新的都存好再換；存不了就留在這張
  if (!await settleBeforeLeaving()) return;
  if (stale()) return;
  const box = $("edStage");
  box.replaceChildren(h("p.dim.ednone", {text: t("wall.loading")}));
  const d = await get(`/api/detail?id=${id}`).catch(() => null);
  if (stale()) return;
  if (!d || d.error || !d.exists) { saveUi({esrc: 0}); E.src = null; E.layers = []; await drawEmpty(); return; }
  const pj = (await get(`/api/edit/project?src=${id}`).catch(() => ({}))).project;
  if (stale()) return;
  let resume = !!pj;
  const auto = E.autoResume;
  E.autoResume = false;
  if (pj && !auto) {
    // 有工作檔：預設接著做（Esc、點外面也是接著做）；「重新開始」才捨棄
    const restart = await confirmDialog({
      title: t("ed.resumeTitle"),
      body: [h("p", {text: t("ed.resumeBody", {n: pj.layers.length, when: new Date(pj.updated * 1000).toLocaleString(lang() || undefined)})})],
      ok: t("ed.restart"), cancel: t("ed.resume")});
    if (stale()) return;
    if (restart) { await post("/api/edit/discard", {id: pj.id}).catch(() => {}); resume = false; }
  }
  let state;
  try {
    if (resume) {
      const onDisk = new Map();
      const layers = await Promise.all(pj.layers.map(async L => {
        const canvas = await bitmapCanvas(`/api/edit/layer?id=${pj.id}&layer=${L.id}&v=${pj.updated}`, pj.w, pj.h);
        onDisk.set(L.id, canvas);
        return {...L, canvas};
      }));
      state = {pid: pj.id, w: pj.w, h: pj.h, layers, onDisk,
               active: layers.some(L => L.id === pj.active) ? pj.active : layers[layers.length - 1].id, savedAt: pj.updated};
    } else {
      const base = await bitmapCanvas(`/api/full?id=${id}&v=${encodeURIComponent(d.mtime)}`);
      const L = {id: newId(), name: t("ed.base"), canvas: base, visible: true, opacity: 1};
      state = {pid: "", w: base.width, h: base.height, layers: [L], onDisk: new Map(), active: L.id, savedAt: 0};
    }
  } catch (e) {
    if (!stale()) box.replaceChildren(h("p.dim.ednone", {text: t("ed.loadErr", {msg: e.message || e})}));
    return;
  }
  if (stale()) return;
  // 載入的這段時間左欄還在：又改了上一張（例如圖層的不透明度）就再存一次
  if (!await settleBeforeLeaving() || stale()) return;
  clearTimeout(E.saveTimer);
  Object.assign(E, state, {src: d, hist: [], fut: [], crop: null, tool: "view", zoom: {...FIT}, msg: null, pending: false,
                           sel: null, sdrag: null, poly: null, xf: null, ants: null, adj: null, clone: {src: null, off: null, mark: null},
                           txt: null});
  E.out = {w: E.w, h: E.h, w0: E.w, h0: E.h};
  drawPanel();
  drawStage();
  loadLogo().then(() => drawPanel()).catch(() => {});
}

// 沒有打開圖：說明怎麼開始，列出未完成的編輯
async function drawEmpty() {
  const panel = $("edPanel"), stage = $("edStage");
  const list = (await get("/api/edit/projects").catch(() => ({}))).projects || [];
  panel.replaceChildren(
    h("div.rhd", {}, h("h2.title", {id: "edTitle", "data-i18n": "nav.output"})),
    h("div.rfempty", {}, h("p", {"data-i18n": "ed.empty"}), h("a.pill.primary", {href: "#library", "data-i18n": "ref.pickInLibrary"}),
      kraOpenButton("button.pill")));
  stage.replaceChildren(list.length
    ? h("div.edprojects", {},
      h("h3", {"data-i18n": "ed.unfinished"}),
      h("div.tgrid", {}, list.map(p => h("button.tcard", {type: "button", disabled: !p.src_id, title: p.filename,
                                                           on: {click: () => openEditor(p.src_id)}},
        p.src_id ? h("img", {src: `/api/thumb?id=${p.src_id}&v=${encodeURIComponent(p.v)}`, alt: "", loading: "lazy"}) : h("span.aglyph", {text: "—"}),
        h("span.tmeta", {}, h("b", {text: p.filename}),
          h("span.dim", {text: p.src_id ? t("ed.layersN", {n: p.layers}) : t("ed.srcGone")}),
          h("span.dim.mono", {text: new Date(p.updated * 1000).toLocaleString(lang() || undefined)}))))))
    : h("div.rfnone", {}, h("p.dim", {"data-i18n": "ed.noUnfinished"})));
  translate(panel);
  translate(stage);
}

// ── 復原 ──
// 一步＝整個圖層狀態的淺拷貝（canvas 是參照）。會改像素的動作一律換新的 canvas，舊的就是復原用的那份。
// 選取範圍也一樣（改選取＝換一張新的），所以選取也能復原
const snap = () => ({w: E.w, h: E.h, active: E.active, layers: E.layers.map(L => ({...L})), sel: E.sel});

function restore(s) {
  E.w = s.w; E.h = s.h; E.active = s.active;
  E.layers = s.layers.map(L => ({...L}));
  E.sel = s.sel || null;
}

function commit(fn) {
  if (E.xf) applyXf();                               // 變形、調整、打字到一半去做別的：先套用
  if (E.adj) applyAdj();
  if (E.txt) endText(false);
  const before = snap();
  fn();
  E.hist.push(before);
  if (E.hist.length > HIST_MAX) E.hist.shift();
  E.fut = [];
  changed();
}

function undo() {
  if (E.xf) { cancelXf(); return; }                  // 變形、調整到一半：復原＝取消這次
  if (E.adj) { cancelAdj(); return; }
  if (E.txt) endText(false);
  if (E.poly) { E.poly = null; drawStage(); return; }
  if (!E.hist.length) return;
  E.fut.push(snap());
  restore(E.hist.pop());
  changed();
}

function redo() {
  if (!E.fut.length || E.xf || E.adj) return;
  E.hist.push(snap());
  restore(E.fut.pop());
  changed();
}

// 改了之後：重畫、尺寸變了輸出尺寸跟著變、排自動存檔
function changed(sizeChanged = false) {
  if (sizeChanged || !E.out || E.out.w0 !== E.w || E.out.h0 !== E.h) E.out = {w: E.w, h: E.h, w0: E.w, h0: E.h};
  drawStage();
  drawPanel();
  clearTimeout(E.saveTimer);
  E.pending = true;
  E.rev++;
  E.saveTimer = setTimeout(autoSave, 1200);
}

// ── 工作檔 ──
function pngOf(canvas) {
  return new Promise(res => canvas.toBlob(b => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(b);
  }, "image/png"));
}

// 存的是開始存的那一刻的樣子（圖層清單先拍下來）；存的過程中又改了，存完再存一次。
// 同一時間只送一次（inflight.saveQueue）：計時器到了正在送就不重疊；換圖、開 .kra、存 .kra 之前用 SAVE.flush()
// 等到最新的都存好——以前存檔還在送時換圖，saveProject 直接返回，換圖前那一筆沒存到（Codex 第三輪審查）
const SAVE = saveQueue(saveOnce, () => E.pending);
const saveProject = () => (E.src ? SAVE.now() : Promise.resolve(true));
const autoSave = () => { if (E.pending) saveProject(); };

async function saveOnce() {
  if (!E.src) return true;
  E.saving = true;
  syncSaved();
  const src = E.src, rev = E.rev, list = E.layers.slice();
  const body = {id: E.pid, src_id: src.id, w: E.w, h: E.h, active: E.active};
  body.layers = await Promise.all(list.map(async L => ({
    id: L.id, name: L.name, visible: L.visible, opacity: L.opacity, ...(L.text ? {text: L.text} : {}), ...(L.wm ? {wm: true} : {}), ...(L.lock ? {lock: true} : {}),
    ...(E.onDisk.get(L.id) === L.canvas ? {} : {png: await pngOf(L.canvas)})})));
  const d = await post("/api/edit/save", body).catch(e => ({error: e.message}));
  E.saving = false;
  if (E.src !== src) return !d.error;                // 換圖會先 flush，照理不會走到這裡
  if (!d.error) {
    E.pid = d.id;
    E.onDisk = new Map(list.map(L => [L.id, L.canvas]));
    E.savedAt = Date.now() / 1000;
    E.pending = E.rev !== rev;
    if (E.pending) { clearTimeout(E.saveTimer); E.saveTimer = setTimeout(autoSave, 600); }
  }
  syncSaved(d.error);
  E.saveErr = d.error || "";
  return !d.error;
}

// 換圖、開 .kra 之前：手上這張先存到最新；存不了就留在這張（不能把沒存的編輯丟掉），回傳 false
async function settleBeforeLeaving() {
  if (!E.src || (!E.pending && !SAVE.running())) return true;
  clearTimeout(E.saveTimer);
  if (await SAVE.flush()) return true;
  saveUi({esrc: E.src.id});                          // 打開的還是這張
  note(t("ed.keepUnsaved", {msg: E.saveErr || "—"}), "err");
  drawPanel();
  drawStage();
  return false;
}

function syncSaved(err) {
  const el = $("edSaved");
  if (!el) return;
  el.textContent = err ? t("ed.saveErr", {msg: err}) : E.saving ? t("ed.saving")
    : E.savedAt ? t("ed.saved", {when: new Date(E.savedAt * 1000).toLocaleTimeString(lang() || undefined, {hour: "2-digit", minute: "2-digit"})}) : "";
  el.dataset.kind = err ? "err" : "";
}

// ── 整張的動作：每層換成新的 canvas ──
function eachLayer(w, h, draw) {
  E.sel = null;
  E.layers = E.layers.map(L => {
    const c = canvasOf(w, h), ctx = c.getContext("2d");
    draw(ctx, L.canvas);
    return {...L, canvas: c, text: undefined};
  });
  E.w = w; E.h = h;
}

function flip(horizontal) {
  commit(() => eachLayer(E.w, E.h, (ctx, src) => {
    if (horizontal) { ctx.translate(E.w, 0); ctx.scale(-1, 1); } else { ctx.translate(0, E.h); ctx.scale(1, -1); }
    ctx.drawImage(src, 0, 0);
  }));
}

function rotate(cw) {
  const w = E.h, h = E.w;
  commit(() => eachLayer(w, h, (ctx, src) => {
    if (cw) { ctx.translate(w, 0); ctx.rotate(Math.PI / 2); } else { ctx.translate(0, h); ctx.rotate(-Math.PI / 2); }
    ctx.drawImage(src, 0, 0);
  }));
  E.zoom = {...FIT};
}

function applyCrop() {
  if (!E.crop || E.crop.w < 1 || E.crop.h < 1) return;
  const r = roundCrop(E.crop, E.w, E.h);
  E.crop = null;
  E.tool = "view";
  commit(() => eachLayer(r.w, r.h, (ctx, src) => ctx.drawImage(src, -r.x, -r.y)));
  E.zoom = {...FIT};
}

// ── 圖層 ──
function addLayer(dup) {
  const L0 = cur(), i = E.layers.indexOf(L0);
  const canvas = canvasOf(E.w, E.h);
  if (dup) canvas.getContext("2d").drawImage(L0.canvas, 0, 0);
  const n = E.layers.length + 1;
  const L = {id: newId(), name: dup ? t("ed.copyOf", {name: L0.name}) : t("ed.layerN", {n}), canvas, visible: true,
             opacity: dup ? L0.opacity : 1, ...(dup && L0.text ? {text: {...L0.text}} : {}), ...(dup && L0.lock ? {lock: true} : {})};
  commit(() => { E.layers.splice(i + 1, 0, L); E.active = L.id; });
}

function deleteLayer() {
  if (E.layers.length < 2) return;
  const i = E.layers.indexOf(cur());
  commit(() => {
    E.layers.splice(i, 1);
    E.active = E.layers[Math.max(0, i - 1)].id;
  });
}

function moveLayer(dir) {
  const i = E.layers.indexOf(cur()), j = i + dir;
  if (j < 0 || j >= E.layers.length) return;
  commit(() => { [E.layers[i], E.layers[j]] = [E.layers[j], E.layers[i]]; });
}

// 向下合併：目前這層畫到下面那層上（各自的不透明度算進去），合併後的不透明度是 100%
function mergeDown() {
  const i = E.layers.indexOf(cur());
  if (i < 1) return;
  const up = E.layers[i], down = E.layers[i - 1];
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  if (down.visible) { ctx.globalAlpha = down.opacity; ctx.drawImage(down.canvas, 0, 0); }
  if (up.visible) { ctx.globalAlpha = up.opacity; ctx.drawImage(up.canvas, 0, 0); }
  commit(() => {
    E.layers.splice(i - 1, 2, {...down, canvas: c, opacity: 1, visible: true, text: undefined, wm: undefined});
    E.active = down.id;
  });
}

function setLayer(L, patch) {
  commit(() => { E.layers = E.layers.map(x => (x.id === L.id ? {...x, ...patch} : x)); });
}

// 看得到的圖層合併成一張
function flatten() {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  for (const L of E.layers) {
    if (!L.visible) continue;
    ctx.globalAlpha = L.opacity;
    ctx.drawImage(L.canvas, 0, 0);
  }
  return c;
}

// ── 右邊：工具列、畫布 ──
function drawStage() {
  const box = $("edStage");
  if (!box || !E.src) return;
  const layerEls = E.layers.map(L => {
    const c = E.xf?.lid === L.id ? E.xf.view : E.adj?.lid === L.id ? E.adj.view : L.canvas;
    c.className = "edlayer";
    c.hidden = !L.visible;
    c.style.opacity = String(L.opacity);
    return c;
  });
  const overlay = h("canvas.edover", {width: String(E.w), height: String(E.h), "data-tool": E.tool});
  const stage = h("div.rfba.rfsolo.edba", {style: `--ar:${(E.w / E.h).toFixed(5)}`},
    h("div.rfz.edz", {}, ...layerEls, h("canvas.edants", {width: "1", height: "1"}), overlay));
  const crop = E.tool === "crop";
  const tools = h("div.rftools.edtools", {role: "toolbar", "data-i18n-aria": "ed.tools"},
    h("div.seg", {role: "group", "data-i18n-aria": "ref.tool"}, TOOLS.map(k => h("button", {
      type: "button", "aria-pressed": String(E.tool === k), "data-i18n": `ed.tool.${k}`, "data-i18n-title": `ed.tool.${k}Key`,
      on: {click: () => setTool(k)}}))),
    h("span.edsep"),
    // 整張的旋轉、翻轉：只放符號（工具一多就排不下），說明在滑鼠停上去的提示裡
    ...[["↺", "ed.rotL", () => rotate(false)], ["↻", "ed.rotR", () => rotate(true)], ["⇋", "ed.flipH", () => flip(true)], ["⇅", "ed.flipV", () => flip(false)]]
      .map(([sym, key, fn]) => h("button.pill.edicon", {type: "button", text: sym, "data-i18n-title": key, "data-i18n-aria": key, on: {click: fn}})),
    h("span.sp"),
    h("button.pill", {type: "button", "data-i18n": "ed.undo", "data-i18n-title": "ed.undoKey", disabled: !E.hist.length, on: {click: undo}}),
    h("button.pill", {type: "button", "data-i18n": "ed.redo", "data-i18n-title": "ed.redoKey", disabled: !E.fut.length, on: {click: redo}}));
  const cropBar = crop ? h("div.rftools.edcropbar", {},
    h("div.seg", {role: "group", "data-i18n-aria": "ed.ratio"}, RATIOS.map(([k]) => h("button", {
      type: "button", "aria-pressed": String(E.ratio === k), "data-i18n": `ed.ratio.${k}`,
      on: {click: () => { E.ratio = k; if (E.crop) E.crop = fitRatio(E.crop); drawStage(); }}}))),
    h("span.sp"),
    h("span.dim.mono", {id: "edCropSize", text: E.crop ? `${Math.round(E.crop.w)} × ${Math.round(E.crop.h)}` : ""}),
    h("button.pill", {type: "button", "data-i18n": "ed.cropCancel", on: {click: () => { E.crop = null; setTool("view"); }}}),
    h("button.pill.primary", {type: "button", "data-i18n": "ed.cropApply", "data-i18n-title": "ed.cropApplyKey", disabled: !E.crop,
                              on: {click: applyCrop}})) : null;
  box.replaceChildren(...[
    tools, cropBar, PAINT.includes(E.tool) ? paintBar() : null,
    ...(E.tool === "select" ? selBars() : []), E.tool === "transform" && E.xf ? xfBar() : null,
    E.tool === "adjust" && E.adj ? adjBar() : null,
    E.tool === "text" ? textBar() : null,
    h("div.rfview.edview", {id: "edView", "data-space": E.space ? "" : null}, stage,
      h("div.rfcursor", {id: "edCursor", hidden: true}),
      h("canvas.edloupe", {id: "edLoupe", width: String(LOUPE_N * LOUPE_Z), height: String(LOUPE_N * LOUPE_Z + LOUPE_BAR), hidden: true}))].filter(Boolean));
  translate(box);
  wireZoom($("edView"), stage);
  wireOverlay(overlay);
  wirePaint(overlay);
  wireSelect(overlay);
  wireXf(overlay);
  wireText(overlay);
  drawOverlay(overlay);
  drawAnts();
}

// ── 筆刷的設定列 ──
function setBrush(patch, persist = true) {
  E.ui.ebrush = {...bset(), ...patch};
  if (persist) saveUi({ebrush: E.ui.ebrush});
}
function setFx(patch, persist = true) {
  E.ui.efx = {...fxset(), ...patch};
  if (persist) saveUi({efx: E.ui.efx});
}

// 拉桿：拖的時候只改數字，放開才存
function slider(lab, min, max, val, unit, set, id) {
  const n = h("span.mono.rfnum", {text: `${val}${unit}`, id: id ? `${id}Num` : null});
  return h("label.rfbrush", {},
    h("span", {"data-i18n": lab}),
    h("input", {type: "range", min: String(min), max: String(max), step: "1", value: String(val), id, "data-i18n-aria": lab,
                on: {input: e => { n.textContent = `${e.target.value}${unit}`; set(Number(e.target.value), false); },
                     change: e => set(Number(e.target.value), true)}}),
    n);
}

function paintBar() {
  const b = bset(), f = fxset(), tl = E.tool;
  const swatch = h("input.rfswatch", {type: "color", id: "edColor", value: b.color, "data-i18n-aria": "ref.color", "data-i18n-title": "ed.colorHint",
                                      on: {input: e => setBrush({color: e.target.value}, false), change: e => setBrush({color: e.target.value})}});
  return h("div.rftools.edpaintbar", {},
    tl === "brush" || tl === "picker" ? h("label.rfbrush", {}, h("span", {"data-i18n": "ref.color"}), swatch) : null,
    tl !== "picker" ? slider("ed.size", SIZE[0], SIZE[1], b.size, "px", (v, p) => setBrush({size: v}, p), "edSize") : null,
    ["brush", "eraser", "clone"].includes(tl) ? slider("ed.bopacity", 1, 100, b.opacity, "%", (v, p) => setBrush({opacity: v}, p)) : null,
    ["brush", "eraser", "clone"].includes(tl) ? slider("ed.hard", 0, 100, b.hard, "%", (v, p) => setBrush({hard: v}, p)) : null,
    tl === "clone" ? h("button.pill.edtoggle", {type: "button", "aria-pressed": String(!!cloneset().aligned), "data-i18n": "ed.aligned",
                                                 "data-i18n-title": "ed.alignedHint", on: {click: () => {
      E.ui.eclone = {...cloneset(), aligned: !cloneset().aligned};
      saveUi({eclone: E.ui.eclone});
      E.clone.off = null;
      drawStage();
    }}}) : null,
    tl === "clone" ? h("div.seg", {role: "group", "data-i18n-aria": "ed.cloneSample"}, ["all", "layer"].map(k => h("button", {
      type: "button", "aria-pressed": String(cloneset().sample === k), "data-i18n": `ed.sample.${k}`, "data-i18n-title": `ed.cloneSample.${k}`,
      on: {click: () => { E.ui.eclone = {...cloneset(), sample: k}; saveUi({eclone: E.ui.eclone}); drawStage(); }}}))) : null,
    tl === "blur" ? slider("ed.blurR", 1, 40, f.blur, "px", (v, p) => setFx({blur: v}, p)) : null,
    tl === "sharpen" ? slider("ed.sharpAmt", 1, 100, f.sharp, "%", (v, p) => setFx({sharp: v}, p)) : null,
    tl !== "picker" ? h("span.rfbrush.edpress", {"data-i18n-title": "ed.pressHint"},
      h("span", {"data-i18n": "ed.press"}),
      ...[["pSize", "ed.pSize"], ["pOpacity", "ed.pOpacity"]].map(([k, key]) => h("button.pill", {
        type: "button", "aria-pressed": String(!!b[k]), "data-i18n": key,
        on: {click: e => { setBrush({[k]: !bset()[k]}); e.currentTarget.setAttribute("aria-pressed", String(!!bset()[k])); }}}))) : null,
    h("span.sp"),
    h("span.dim", {"data-i18n": tl === "picker" ? "ed.pickerHint" : tl === "clone" ? (E.clone.src ? "ed.cloneHint" : "ed.cloneNeedSrc") : "ed.paintHint"}));
}

function setSize(v) {
  const s = Math.round(Math.min(SIZE[1], Math.max(SIZE[0], v)));
  setBrush({size: s});
  const sl = $("edSize"), n = $("edSizeNum");
  if (sl) sl.value = String(s);
  if (n) n.textContent = `${s}px`;
}

// ── 畫 ──
// 一筆開始時選中那層換一張新的 canvas（舊的留給復原），筆觸先畫在一張遮罩上，每動一下重新合成：
// 原本那層＋遮罩×（顏色／擦掉／模糊或銳利過的那一份）。不透明度是整筆一起算，同一筆來回塗不會越疊越深
function wirePaint(c) {
  if (!PAINT.includes(E.tool)) return;
  let st = null;
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * E.w / r.width, (e.clientY - r.top) * E.h / r.height];
  };
  c.addEventListener("pointerdown", e => {
    if (e.button !== 0 || E.space) return;
    e.preventDefault();
    const p = at(e);
    const penEraser = e.pointerType === "pen" && (e.buttons & 32);     // 繪圖筆翻過來用筆尾：當橡皮擦
    const cloning = E.tool === "clone" && !penEraser;
    if (cloning && e.altKey) { setCloneSrc(p); return; }              // 仿製印章：Alt＋點一下＝取樣點（跟 Photoshop 一樣）
    if (!penEraser && (E.tool === "picker" || e.altKey)) { pickAt(p); loupe(e, p); return; }
    const L = cur();
    if (!L.visible) { note(t("ed.hiddenLayer"), "warn"); return; }
    if (cloning && !E.clone.src) { note(t("ed.cloneNeedSrc"), "warn"); return; }
    if (L.lock && (penEraser || E.tool === "eraser")) { note(t("ed.lockedErase"), "warn"); return; }
    // 位移＝這一筆的起點－取樣點。對齊：第一筆之後一直沿用；不對齊：每一筆都從取樣點開始
    if (cloning && (!E.clone.off || !cloneset().aligned)) E.clone.off = [p[0] - E.clone.src[0], p[1] - E.clone.src[1]];
    try { c.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    st = beginStroke(L, penEraser ? "eraser" : E.tool);
    dabTo(st, p, e);
    flush(st);
  });
  c.addEventListener("pointermove", e => {
    moveCursor(e);
    if (E.tool === "clone") cloneMark(c, at(e));
    if (!st) loupe(e, at(e));
    if (!st) return;
    // 繪圖板一格畫面之間回報的點很多，全部用上線條才順
    const evs = e.getCoalescedEvents?.() ?? [];
    for (const ev of evs.length ? evs : [e]) dabTo(st, at(ev), ev);
    flush(st);
  });
  c.addEventListener("contextmenu", e => e.preventDefault());           // 筆長按、筆身按鈕不要跳出選單
  const end = () => { if (st) { endStroke(st); st = null; } };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => { const cur0 = $("edCursor"); if (cur0) cur0.hidden = true; hideLoupe(); });
  // Shift＋滾輪調大小（滾輪本身是放大檢視）
  c.addEventListener("wheel", e => {
    if (!e.shiftKey || E.tool === "picker") return;
    e.preventDefault();
    e.stopPropagation();
    const d = e.deltaY || e.deltaX;
    if (d) setSize(bset().size * (d < 0 ? 1.15 : 1 / 1.15));
    moveCursor(e);
  }, {passive: false});
}

// 筆刷的圓圈跟著游標，大小是畫面上實際會塗到的範圍（含放大檢視）
function moveCursor(e) {
  const el = $("edCursor"), ov = document.querySelector(".edover");
  if (!el || !ov) return;
  if (E.tool === "picker") { el.hidden = true; return; }
  const box = el.parentElement, br = box.getBoundingClientRect(), r = ov.getBoundingClientRect();
  const z = br.width / box.offsetWidth || 1;
  const d = bset().size * r.width / E.w / z;
  Object.assign(el.style, {width: `${d}px`, height: `${d}px`, left: `${(e.clientX - br.left) / z}px`, top: `${(e.clientY - br.top) / z}px`});
  el.hidden = false;
}

function note(text, kind = "ok") {
  E.msg = {text, kind};
  const m = $("edMsg");
  if (m) { m.textContent = text; m.dataset.kind = kind; }
}

// 模糊過、銳利過的整層（一筆開始時算一次，筆刷塗到的地方才換成這份）
function blurred(src, r) {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.filter = `blur(${r}px)`;
  ctx.drawImage(src, 0, 0);
  return c;
}

function sharpened(src, amount) {
  const a = src.getContext("2d").getImageData(0, 0, E.w, E.h);
  const b = blurred(src, 2).getContext("2d").getImageData(0, 0, E.w, E.h);
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  const out = ctx.createImageData(E.w, E.h);
  sharpen(a.data, b.data, amount, out.data);
  ctx.putImageData(out, 0, 0);
  return c;
}

function beginStroke(L, tool = E.tool) {
  const before = snap();
  const orig = L.canvas, nc = canvasOf(E.w, E.h);
  nc.className = "edlayer";
  nc.style.opacity = orig.style.opacity;
  nc.getContext("2d").drawImage(orig, 0, 0);
  orig.replaceWith(nc);                      // 畫面上換成新的那張
  E.layers = E.layers.map(x => (x.id === L.id ? {...x, canvas: nc, text: undefined} : x));   // 畫過就不再是文字圖層
  const f = fxset(), b = bset();
  const fx = tool === "blur" ? blurred(orig, f.blur) : tool === "sharpen" ? sharpened(orig, f.sharp / 50)
    : tool === "clone" ? cloneSource(orig) : null;
  const mask = canvasOf(E.w, E.h), md = new ImageData(E.w, E.h);
  return {tool, orig, nc, fx, mask, md, dirty: null, tmp: canvasOf(E.w, E.h), b, last: null, lastPr: null, carry: 0, before, sel: E.sel,
          lock: !!L.lock,
          dyn: {size: !!b.pSize, opacity: !!b.pOpacity}};
}

// 繪圖筆的壓力 0～1；滑鼠、觸控當作 1
const pressureOf = e => (e.pointerType === "pen" && e.pressure > 0 ? e.pressure : 1);

function dabTo(st, p, e) {
  const pr = pressureOf(e), p0 = st.lastPr ?? pr;
  const radius = q => Math.max(0.5, st.b.size * (st.dyn.size ? Math.max(0.1, q) : 1) / 2);
  // 模糊、銳利用軟邊，接縫才不明顯
  const hard = ["brush", "eraser", "clone"].includes(st.tool) ? st.b.hard / 100 : 0.3;
  const seg = st.last ? dabs(st.last, p, Math.max(1, radius(pr) * 0.3), st.carry) : {pts: [p], carry: 0};
  const len = st.last ? Math.hypot(p[0] - st.last[0], p[1] - st.last[1]) : 0;
  for (const q of seg.pts) {
    // 兩個回報點之間的筆壓照距離內插，粗細、濃淡才是漸變的
    const qp = len ? p0 + (pr - p0) * Math.hypot(q[0] - st.last[0], q[1] - st.last[1]) / len : pr;
    const box = stamp(st.md.data, E.w, E.h, q[0], q[1], radius(qp), hard, st.dyn.opacity ? qp : 1);
    if (box) st.dirty = st.dirty ? [Math.min(st.dirty[0], box[0]), Math.min(st.dirty[1], box[1]),
                                     Math.max(st.dirty[2], box[2]), Math.max(st.dirty[3], box[3])] : box;
  }
  st.carry = seg.carry;
  st.last = p;
  st.lastPr = pr;
}

// 遮罩改到的那塊寫回 canvas，再重新合成一次
function flush(st) {
  if (!st.dirty) return;
  const [x0, y0, x1, y1] = st.dirty, m = st.mask.getContext("2d");
  m.putImageData(st.md, 0, 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  if (st.sel) {                                      // 有選取範圍：選取外面的筆觸切掉
    m.save();
    m.beginPath();
    m.rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    m.clip();
    m.globalCompositeOperation = "destination-in";
    m.drawImage(st.sel, 0, 0);
    m.restore();
  }
  st.dirty = null;
  recompose(st);
}

function recompose(st) {
  const ctx = st.nc.getContext("2d"), t0 = st.tmp.getContext("2d");
  ctx.globalCompositeOperation = "copy";
  ctx.globalAlpha = 1;
  ctx.drawImage(st.orig, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  if (st.tool === "eraser") {
    ctx.globalCompositeOperation = "destination-out";
    ctx.globalAlpha = st.b.opacity / 100;
    ctx.drawImage(st.mask, 0, 0);
  } else {
    // 塗色：遮罩換成顏色；模糊／銳利：那一份只留遮罩塗到的地方
    t0.globalCompositeOperation = "copy";
    if (st.tool === "brush") {
      t0.drawImage(st.mask, 0, 0);
      t0.globalCompositeOperation = "source-in";
      t0.fillStyle = st.b.color;
      t0.fillRect(0, 0, E.w, E.h);
    } else {
      t0.drawImage(st.fx, 0, 0);
      t0.globalCompositeOperation = "destination-in";
      t0.drawImage(st.mask, 0, 0);
    }
    t0.globalCompositeOperation = "source-over";
    ctx.globalAlpha = st.tool === "brush" || st.tool === "clone" ? st.b.opacity / 100 : 1;
    if (st.lock) ctx.globalCompositeOperation = "source-atop";   // 鎖定透明：只改顏色，透明度照原本的
    ctx.drawImage(st.tmp, 0, 0);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

function endStroke(st) {
  E.hist.push(st.before);
  if (E.hist.length > HIST_MAX) E.hist.shift();
  E.fut = [];
  changed();
}

// ── 滴管的放大鏡 ──
// 用滴管（或筆刷時按住 Alt）時，游標旁邊放大附近 11×11 個像素，中間框起來的那格就是點下去會吸到的顏色。
// 下面左半是那格的顏色（＋色碼），右半是現在筆刷的顏色，方便比較
const LOUPE_N = 11, LOUPE_Z = 12, LOUPE_BAR = 24;
let loupeBuf = null;

function hideLoupe() {
  const el = $("edLoupe");
  if (el) el.hidden = true;
}

function loupe(e, p) {
  const el = $("edLoupe");
  if (!el) return;
  const px = Math.floor(p[0]), py = Math.floor(p[1]), r = (LOUPE_N - 1) / 2;
  if (!(E.tool === "picker" || (e.altKey && PAINT.includes(E.tool) && E.tool !== "clone")) || px < 0 || py < 0 || px >= E.w || py >= E.h) { el.hidden = true; return; }
  loupeBuf ||= canvasOf(LOUPE_N, LOUPE_N);
  const sc = loupeBuf.getContext("2d", {willReadFrequently: true});
  sc.clearRect(0, 0, LOUPE_N, LOUPE_N);
  for (const L of E.layers) {                        // 跟吸色一樣：看得到的圖層合起來
    if (!L.visible) continue;
    sc.globalAlpha = L.opacity;
    sc.drawImage(L.canvas, px - r, py - r, LOUPE_N, LOUPE_N, 0, 0, LOUPE_N, LOUPE_N);
  }
  sc.globalAlpha = 1;
  const d = sc.getImageData(r, r, 1, 1).data;
  const S = LOUPE_N * LOUPE_Z, ctx = el.getContext("2d");
  for (let y = 0; y < S; y += 6) {                    // 透明的地方看得出來：棋盤格
    for (let x = 0; x < S; x += 6) {
      ctx.fillStyle = ((x + y) / 6) % 2 ? "#c8c8c8" : "#f0f0f0";
      ctx.fillRect(x, y, 6, 6);
    }
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(loupeBuf, 0, 0, S, S);
  ctx.strokeStyle = "rgba(0, 0, 0, .12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 1; k < LOUPE_N; k++) {
    ctx.moveTo(k * LOUPE_Z + 0.5, 0); ctx.lineTo(k * LOUPE_Z + 0.5, S);
    ctx.moveTo(0, k * LOUPE_Z + 0.5); ctx.lineTo(S, k * LOUPE_Z + 0.5);
  }
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#000";
  ctx.strokeRect(r * LOUPE_Z - 1, r * LOUPE_Z - 1, LOUPE_Z + 2, LOUPE_Z + 2);
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(r * LOUPE_Z + 1, r * LOUPE_Z + 1, LOUPE_Z - 2, LOUPE_Z - 2);
  // 下面：吸到的顏色（全透明＝吸不到）｜現在的筆刷顏色
  const hex = d[3] ? `#${[d[0], d[1], d[2]].map(v => v.toString(16).padStart(2, "0")).join("")}` : "";
  ctx.fillStyle = hex || "#808080";
  ctx.fillRect(0, S, S / 2, LOUPE_BAR);
  ctx.fillStyle = bset().color;
  ctx.fillRect(S / 2, S, S / 2, LOUPE_BAR);
  const light = d[3] && (0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]) > 140;
  ctx.fillStyle = light ? "#000" : "#fff";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  ctx.fillText(hex || "—", 6, S + LOUPE_BAR / 2);
  // 放在游標右下；碰到邊就換到另一邊（藏起來時量不到大小，用畫布的大小算）
  const box = el.parentElement, br = box.getBoundingClientRect(), z = br.width / box.offsetWidth || 1;
  const cx = (e.clientX - br.left) / z, cy = (e.clientY - br.top) / z, gap = 20;
  const lx = cx + gap + S > box.offsetWidth ? cx - gap - S : cx + gap;
  const ly = cy + gap + (S + LOUPE_BAR) > box.offsetHeight ? cy - gap - (S + LOUPE_BAR) : cy + gap;
  el.style.left = `${Math.max(0, lx)}px`;
  el.style.top = `${Math.max(0, ly)}px`;
  el.hidden = false;
}

// 滴管：取看得到的圖層合起來的顏色（跟 Krita 預設一樣），完全透明的地方不取
function pickAt([x, y]) {
  const c = canvasOf(1, 1), ctx = c.getContext("2d", {willReadFrequently: true});
  const px = Math.floor(x), py = Math.floor(y);
  if (px < 0 || py < 0 || px >= E.w || py >= E.h) return;
  for (const L of E.layers) {
    if (!L.visible) continue;
    ctx.globalAlpha = L.opacity;
    ctx.drawImage(L.canvas, px, py, 1, 1, 0, 0, 1, 1);
  }
  const d = ctx.getImageData(0, 0, 1, 1).data;
  if (!d[3]) return;
  const hex = `#${[d[0], d[1], d[2]].map(v => v.toString(16).padStart(2, "0")).join("")}`;
  setBrush({color: hex});
  const sw = $("edColor");
  if (sw) sw.value = hex;
}

function setTool(k) {
  if (!TOOLS.includes(k)) return;
  if (E.xf && k !== "transform") applyXf();          // 變形、調整中換別的工具：先套用
  if (E.adj && k !== "adjust") applyAdj();
  if (E.txt && k !== "text") endText(false);
  if (k === "transform" && !E.xf && !startXf()) { drawStage(); return; }
  if (k === "adjust" && !E.adj && !startAdj()) { drawStage(); return; }
  if (k !== "select") E.poly = null;
  E.tool = k;
  if (k !== "crop") E.crop = null;
  drawStage();
}

const ratioNow = () => {
  const r = RATIOS.find(([k]) => k === E.ratio)?.[1] ?? 0;
  return r < 0 ? E.w / E.h : r;
};

// 換比例時：保持中心，照新比例縮到放得進原本的框
function fitRatio(r) {
  const ratio = ratioNow();
  if (!(ratio > 0)) return r;
  const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
  let w = r.w, hh = w / ratio;
  if (hh > r.h) { hh = r.h; w = hh * ratio; }
  return {x: cx - w / 2, y: cy - hh / 2, w, h: hh};
}

// 裁切框：拉出新框、拖著移、拉邊角改大小
function wireOverlay(c) {
  if (E.tool !== "crop") return;
  let drag = null;
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * E.w / r.width, (e.clientY - r.top) * E.h / r.height];
  };
  const tol = () => 10 * E.w / (c.getBoundingClientRect().width || E.w);
  c.addEventListener("pointerdown", e => {
    if (e.button !== 0 || E.space) return;
    e.preventDefault();
    try { c.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    const p = at(e), hit = hitCrop(E.crop, p, tol());
    drag = hit ? {mode: hit, start: p, from: {...E.crop}} : {mode: "new", start: p};
  });
  c.addEventListener("pointermove", e => {
    const p = at(e);
    if (!drag) { c.dataset.hit = hitCrop(E.crop, p, tol()) || ""; return; }
    if (drag.mode === "new") E.crop = cropFromDrag(drag.start, p, ratioNow(), E.w, E.h);
    else if (drag.mode === "move") E.crop = moveCrop(drag.from, p[0] - drag.start[0], p[1] - drag.start[1], E.w, E.h);
    else E.crop = resizeCrop(drag.from, drag.mode, p, ratioNow(), E.w, E.h);
    drawOverlay(c);
    const sz = $("edCropSize");
    if (sz) sz.textContent = `${Math.round(E.crop.w)} × ${Math.round(E.crop.h)}`;
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    if (E.crop && (E.crop.w < 4 || E.crop.h < 4)) E.crop = null;
    drawStage();
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
}

// 畫布上面那層：裁切框、選取中的形狀、變形框（線寬換成畫面上固定的粗細）
function drawOverlay(c) {
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, E.w, E.h);
  const px = E.w / (c.getBoundingClientRect().width || E.w);
  if (E.tool === "crop" && E.crop) drawCropBox(ctx, E.crop, px);
  else if (E.tool === "select") drawSelShape(ctx, px);
  else if (E.tool === "transform" && E.xf) drawXfBox(ctx, E.xf, px);
  else if (E.tool === "clone") drawCloneMark(ctx, px);
  else if (E.tool === "text") drawTextBoxes(ctx, px);
}

// 框外變暗、框線、三等分線、八個把手
function drawCropBox(ctx, r, px) {
  ctx.fillStyle = "rgba(0, 0, 0, .55)";
  ctx.fillRect(0, 0, E.w, E.h);
  ctx.clearRect(r.x, r.y, r.w, r.h);
  ctx.strokeStyle = "rgba(255, 255, 255, .45)";
  ctx.lineWidth = px;
  ctx.beginPath();
  for (const f of [1 / 3, 2 / 3]) {
    ctx.moveTo(r.x + r.w * f, r.y); ctx.lineTo(r.x + r.w * f, r.y + r.h);
    ctx.moveTo(r.x, r.y + r.h * f); ctx.lineTo(r.x + r.w, r.y + r.h * f);
  }
  ctx.stroke();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2 * px;
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  const s = 9 * px;
  ctx.fillStyle = "#fff";
  for (const [x, y] of [[r.x, r.y], [r.x + r.w / 2, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h / 2], [r.x + r.w, r.y + r.h / 2],
                        [r.x, r.y + r.h], [r.x + r.w / 2, r.y + r.h], [r.x + r.w, r.y + r.h]]) {
    ctx.fillRect(x - s / 2, y - s / 2, s, s);
  }
}

// 放大檢視：滾輪以游標為中心縮放、空白鍵＋拖曳或按住滾輪移動、「檢視」工具直接拖；0 回到符合視窗
function wireZoom(view, stage) {
  const z0 = stage.querySelector(".rfz");
  const frame = () => ({L: z0.offsetLeft, T: z0.offsetTop, w: z0.offsetWidth, h: z0.offsetHeight, W: stage.offsetWidth, H: stage.offsetHeight});
  const uiz = () => stage.getBoundingClientRect().width / stage.offsetWidth || 1;
  const local = e => {
    const r = stage.getBoundingClientRect(), z = uiz();
    return [(e.clientX - r.left) / z - z0.offsetLeft, (e.clientY - r.top) / z - z0.offsetTop];
  };
  const pct = h("button.rfzpct.mono", {type: "button", "data-i18n-title": "ref.zoomFit", on: {click: () => set({...FIT})}});
  const set = z => {
    E.zoom = clampPan(z, frame());
    const {k, x, y} = E.zoom;
    z0.style.transform = k > 1 ? `translate(${x}px, ${y}px) scale(${k})` : "";
    view.toggleAttribute("data-zoomed", k > 1);
    pct.textContent = `${pixelPercent(E.zoom, z0.offsetWidth * uiz(), E.w)}%`;
    const ov = stage.querySelector(".edover");
    if (ov) drawOverlay(ov);
    drawAnts();
  };
  const step = f => { const fr = frame(); set(zoomAt(E.zoom, f, fr.W / 2 - fr.L, fr.H / 2 - fr.T, fr)); };
  view.append(h("div.rfzctl", {role: "group", "data-i18n-aria": "ref.zoom"},
    h("button", {type: "button", "data-i18n-aria": "ref.zoomOut", "data-i18n-title": "ref.zoomOut", on: {click: () => step(1 / 1.5)}}, "−"),
    pct,
    h("button", {type: "button", "data-i18n-aria": "ref.zoomIn", "data-i18n-title": "ref.zoomIn", on: {click: () => step(1.5)}}, "+")));
  view.addEventListener("wheel", e => {
    e.preventDefault();
    const [x, y] = local(e);
    set(zoomAt(E.zoom, e.deltaY < 0 ? 1.25 : 1 / 1.25, x, y, frame()));
  }, {passive: false});
  let pan = null;
  view.addEventListener("pointerdown", e => {
    const viewDrag = e.button === 0 && E.tool === "view";
    if (!(e.button === 1 || (e.button === 0 && E.space) || viewDrag)) return;
    e.preventDefault();
    e.stopPropagation();
    if (E.zoom.k <= 1) return;
    pan = {x: e.clientX, y: e.clientY, z: E.zoom};
    try { view.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    view.classList.add("panning");
  }, true);
  view.addEventListener("mousedown", e => { if (e.button === 1) e.preventDefault(); });
  view.addEventListener("pointermove", e => {
    if (!pan) return;
    const z = uiz();
    set(panBy(pan.z, (e.clientX - pan.x) / z, (e.clientY - pan.y) / z, frame()));
  });
  const end = () => { pan = null; view.classList.remove("panning"); };
  view.addEventListener("pointerup", end);
  view.addEventListener("pointercancel", end);
  E.zoomApply = set;
  set(E.zoom);
  translate(view);
}
addEventListener("resize", () => { if ($("edStage")) E.zoomApply?.(E.zoom); });

// ── 選取範圍（第 3 輪）──
// 選取範圍是一張跟圖一樣大的 canvas，看透明度（白色）；改選取一律換一張新的，舊的留給復原
const pixelsOf = c => c.getContext("2d", {willReadFrequently: true}).getImageData(0, 0, c.width, c.height);
const selBox = (sel = E.sel) => (sel ? alphaBox(pixelsOf(sel).data, E.w, E.h) : null);
const selOrNull = c => (c && selBox(c) ? c : null);   // 全空的選取＝沒選取

function setSelOpt(patch, persist = true) {
  E.ui.esel = {...selset(), ...patch};
  if (persist) saveUi({esel: E.ui.esel});
}

// 改選取：記一步復原。不改像素，不用存工作檔
function setSel(next) {
  if (E.xf) applyXf();
  if (next === E.sel) return;
  E.hist.push(snap());
  if (E.hist.length > HIST_MAX) E.hist.shift();
  E.fut = [];
  E.sel = next;
  E.poly = null;
  drawStage();
}

// 0／1 的陣列（魔術棒、擴張收縮的結果）或 ImageData → 選取用的 canvas
function selFromImage(img) {
  const c = canvasOf(E.w, E.h);
  c.getContext("2d").putImageData(img, 0, 0);
  return c;
}

function shapeSel(s) {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  if (!tracePath(ctx, s)) return null;
  ctx.fillStyle = "#fff";
  ctx.fill();
  return c;
}

// 新的形狀跟原本的選取合起來：new 換掉、add 加進去、sub 從原本的扣掉
function combineSel(shape, mode) {
  if (mode === "new") return shape;
  if (!E.sel) return mode === "add" ? shape : null;
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.drawImage(E.sel, 0, 0);
  ctx.globalCompositeOperation = mode === "sub" ? "destination-out" : "source-over";
  ctx.drawImage(shape, 0, 0);
  return c;
}

function selectAll() {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, E.w, E.h);
  setSel(c);
}

// 反選、擴張、收縮：沿用精修遮罩的計算（maskops）
function adjustSel(fn) {
  if (!E.sel) return;
  const img = pixelsOf(E.sel);
  fn(img.data, E.w, E.h);
  setSel(selOrNull(selFromImage(img)));
}
const invertSel = () => adjustSel(d => invert(d, WHITE));
const growSel = r => adjustSel((d, w, h) => grow(d, w, h, r, WHITE));

// 羽化：邊緣變柔，選取的邊緣是半透明（刪除、填滿、筆刷都是漸漸變淡）
function featherSel(r) {
  if (!E.sel) return;
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.filter = `blur(${Math.max(0.5, r / 2)}px)`;
  ctx.drawImage(E.sel, 0, 0);
  setSel(c);
}

// 魔術棒：預設看所有看得到的圖層合起來的樣子，也可以只看這一層
function wandAt(p, mode) {
  const s = selset();
  const px = pixelsOf(s.sample === "layer" ? cur().canvas : flatten()).data;
  const on = floodSelect(px, E.w, E.h, p[0], p[1], s.tol, s.contig);
  const img = new ImageData(E.w, E.h);
  for (let i = 0; i < on.length; i++) {
    if (!on[i]) continue;
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = img.data[i * 4 + 3] = 255;
  }
  setSel(selOrNull(combineSel(selFromImage(img), mode)));
}

function closePoly() {
  const s = E.poly;
  if (!s) return;
  E.poly = null;
  const shape = shapeSel(s);
  if (shape) setSel(selOrNull(combineSel(shape, s.mode)));
  else drawStage();
}

function wireSelect(c) {
  if (E.tool !== "select") return;
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * E.w / r.width, (e.clientY - r.top) * E.h / r.height];
  };
  c.addEventListener("pointerdown", e => {
    if (e.button !== 0 || E.space) return;
    e.preventDefault();
    const p = at(e), s = selset();
    // 按著 Shift 開始＝加入、按著 Alt＝減去（跟 Photoshop、Krita 一樣），不然照設定列選的方式
    const mode = e.shiftKey ? "add" : e.altKey ? "sub" : E.selMode;
    if (s.shape === "poly") {
      const tol = 10 * E.w / (c.getBoundingClientRect().width || E.w);
      if (!E.poly) E.poly = {tool: "poly", pts: [p], mode, hover: null};
      else if (e.detail >= 2 || nearFirst(E.poly.pts, p, tol)) { closePoly(); return; }
      else E.poly.pts.push(p);
      drawOverlay(c);
      return;
    }
    if (s.shape === "wand") { wandAt(p, mode); return; }
    try { c.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    E.sdrag = {tool: s.shape, a: p, b: p, pts: [p], square: false, mode, shiftAtStart: e.shiftKey};
    drawOverlay(c);
  });
  c.addEventListener("pointermove", e => {
    const p = at(e);
    if (E.poly) { E.poly.hover = p; drawOverlay(c); }
    if (!E.sdrag) return;
    E.sdrag.b = p;
    E.sdrag.square = e.shiftKey && !E.sdrag.shiftAtStart;   // 拖到一半才按 Shift：正方形、正圓
    if (E.sdrag.tool === "lasso") E.sdrag.pts.push(p);
    drawOverlay(c);
  });
  const end = () => {
    const s = E.sdrag;
    if (!s) return;
    E.sdrag = null;
    const shape = shapeSel(s);
    // 只點一下沒拖：新選取時＝取消選取
    if (!shape) { if (s.mode === "new" && E.sel) setSel(null); else drawOverlay(c); return; }
    setSel(selOrNull(combineSel(shape, s.mode)));
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => { if (E.poly) { E.poly.hover = null; drawOverlay(c); } });
}

// 選取中的形狀：虛線外框（減去時不上色）；多邊形的第一個點畫大一點，點回這裡就收起來
function drawSelShape(ctx, px) {
  const s = E.sdrag || (E.poly && {...E.poly, pts: E.poly.hover ? [...E.poly.pts, E.poly.hover] : E.poly.pts});
  if (!s) return;
  const open = s.tool === "poly" && s.pts.length < 3;
  if (open) {
    ctx.beginPath();
    s.pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  } else if (!tracePath(ctx, s)) return;
  if (s.mode !== "sub" && !open) { ctx.fillStyle = "rgba(80, 160, 255, .2)"; ctx.fill(); }
  ctx.setLineDash([]);
  ctx.lineWidth = 3 * px;
  ctx.strokeStyle = "rgba(0, 0, 0, .6)";
  ctx.stroke();
  ctx.setLineDash([6 * px, 4 * px]);
  ctx.lineWidth = 1.5 * px;
  ctx.strokeStyle = "#fff";
  ctx.stroke();
  ctx.setLineDash([]);
  if (s.tool === "poly" && s.pts.length) {
    ctx.beginPath();
    ctx.arc(...s.pts[0], 5 * px, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.stroke();
  }
}

// 螞蟻線：選取範圍的邊緣黑白相間、慢慢流動。解析度跟著畫面（放大檢視時重算，最多到圖的像素），
// 邊緣的像素只在選取改了、解析度變了時才重找
function drawAnts() {
  const c = document.querySelector(".edants");
  if (!c) return;
  if (!E.sel || E.xf) { E.ants = null; c.width = c.height = 1; return; }
  const box = c.parentElement;
  const W = Math.max(1, Math.min(E.w, Math.round((box.offsetWidth || E.w) * Math.max(1, E.zoom.k))));   // 一格＝畫面上一個像素，線才看得清楚
  const H = Math.max(1, Math.round(W * E.h / E.w));
  if (!E.ants || E.ants.sel !== E.sel || E.ants.W !== W) {
    const tmp = canvasOf(W, H);
    tmp.getContext("2d").drawImage(E.sel, 0, 0, W, H);
    const d = pixelsOf(tmp).data, on = new Uint8Array(W * H);
    for (let i = 0; i < on.length; i++) on[i] = d[i * 4 + 3] >= 128 ? 1 : 0;
    E.ants = {sel: E.sel, W, H, idx: edgePixels(on, W, H), img: new ImageData(W, H)};
  }
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  paintAnts(c);
}

function paintAnts(c) {
  const a = E.ants;
  if (!a) return;
  const d = a.img.data;
  for (const p of a.idx) {
    const x = p % a.W, y = (p - x) / a.W, v = ((x + y + E.antPhase) >> 2) & 1 ? 255 : 0, i = p * 4;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  c.getContext("2d").putImageData(a.img, 0, 0);
}

setInterval(() => {
  if (!E.ants || document.hidden) return;
  const c = document.querySelector(".edants");
  if (!c?.isConnected) return;
  E.antPhase = (E.antPhase + 1) % 8;
  paintAnts(c);
}, 160);

// ── 選取範圍裡的像素 ──
// 換掉某層的像素（文字圖層改過像素就變成一般圖層，不能再改字）
const replaceCanvas = (id, c) => { E.layers = E.layers.map(x => (x.id === id ? {...x, canvas: c, text: undefined} : x)); };

function needSel() {
  if (E.sel) return true;
  note(t("ed.needSel"), "warn");
  return false;
}

// 這層在選取範圍裡的部分（沒選取＝整層）／扣掉選取範圍剩下的
function layerPart(L, inside) {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.drawImage(L.canvas, 0, 0);
  if (E.sel) {
    ctx.globalCompositeOperation = inside ? "destination-in" : "destination-out";
    ctx.drawImage(E.sel, 0, 0);
  }
  return c;
}

function clearSel() {
  if (!needSel()) return;
  if (cur().lock) { note(t("ed.lockedErase"), "warn"); return; }
  const L = cur(), c = layerPart(L, false);
  commit(() => replaceCanvas(L.id, c));
}

// 填滿：用筆刷的顏色
function fillSel() {
  if (!needSel()) return;
  const L = cur(), tmp = canvasOf(E.w, E.h), t0 = tmp.getContext("2d");
  t0.drawImage(E.sel, 0, 0);
  t0.globalCompositeOperation = "source-in";
  t0.fillStyle = bset().color;
  t0.fillRect(0, 0, E.w, E.h);
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.drawImage(L.canvas, 0, 0);
  if (L.lock) ctx.globalCompositeOperation = "source-atop";     // 鎖定透明：只填有東西的地方
  ctx.drawImage(tmp, 0, 0);
  commit(() => replaceCanvas(L.id, c));
}

// 拷貝（cut：剪下）成新圖層，放在這層上面
function selToLayer(cut) {
  if (!needSel()) return;
  const L = cur(), part = layerPart(L, true), rest = cut ? layerPart(L, false) : null;
  const N = {id: newId(), name: t(cut ? "ed.cutName" : "ed.copyName", {name: L.name}), canvas: part, visible: true, opacity: L.opacity};
  commit(() => {
    if (rest) replaceCanvas(L.id, rest);
    E.layers.splice(E.layers.findIndex(x => x.id === L.id) + 1, 0, N);
    E.active = N.id;
  });
}

function cropToSel() {
  const r = needSel() && selBox();
  if (!r) return;
  E.crop = r;
  applyCrop();
}

// ── 複製、貼上 ──
// Ctrl+C／X：這層選取範圍裡的部分（沒選取＝整層）。同時放進系統剪貼簿，可以貼到別的程式
function copySel(cut) {
  const L = cur(), r = E.sel ? selBox() : alphaBox(pixelsOf(L.canvas).data, E.w, E.h);
  if (!r) return;
  const c = canvasOf(r.w, r.h);
  c.getContext("2d").drawImage(layerPart(L, true), -r.x, -r.y);
  E.clip = {canvas: c, x: r.x, y: r.y};
  try {
    c.toBlob(b => {
      if (b && navigator.clipboard?.write && window.ClipboardItem) navigator.clipboard.write([new ClipboardItem({"image/png": b})]).catch(() => {});
    }, "image/png");
  } catch { /* 系統剪貼簿不給用：編輯器裡面照樣可以貼 */ }
  if (cut && E.sel) clearSel();
  note(t(cut ? "ed.cutDone" : "ed.copied"));
}

// 貼上成新圖層：自己剛複製的（大小一樣）貼回原本的位置；外面的圖放在中間，比圖大就縮到放得下
async function pasteImage(file, external = false) {
  let bmp = null;
  if (file) { try { bmp = await createImageBitmap(file); } catch { bmp = null; } }
  const own = E.clip && !external && (!bmp || (bmp.width === E.clip.canvas.width && bmp.height === E.clip.canvas.height));
  if (own) { addPasted(E.clip.canvas, E.clip.x, E.clip.y, 1); return; }
  if (!bmp) return;
  const s = Math.min(1, E.w / bmp.width, E.h / bmp.height);
  addPasted(bmp, (E.w - bmp.width * s) / 2, (E.h - bmp.height * s) / 2, s);
}

function addPasted(img, x, y, s) {
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, Math.round(x), Math.round(y), img.width * s, img.height * s);
  const L0 = cur(), N = {id: newId(), name: t("ed.pasted"), canvas: c, visible: true, opacity: 1};
  commit(() => {
    E.layers.splice(E.layers.indexOf(L0) + 1, 0, N);
    E.active = N.id;
  });
}

const imageFile = list => [...(list || [])].find(f => f.type.startsWith("image/"));
document.addEventListener("paste", e => {
  if (location.hash !== "#output" || !E.src || typing(e.target)) return;
  const file = imageFile(e.clipboardData?.files);
  if (!file && !E.clip) return;
  e.preventDefault();
  pasteImage(file);
});
// 把圖拖進畫布：一樣變成新圖層
document.addEventListener("dragover", e => {
  if (location.hash === "#output" && E.src && e.target.closest?.("#edView") && e.dataTransfer?.types?.includes("Files")) e.preventDefault();
});
document.addEventListener("drop", e => {
  if (location.hash !== "#output" || !E.src || !e.target.closest?.("#edView")) return;
  const file = imageFile(e.dataTransfer?.files);
  if (!file) return;
  e.preventDefault();
  pasteImage(file, true);
});

// ── 選取的設定列 ──
function selBars() {
  const s = selset();
  const seg = (aria, keys, on, label, click) => h("div.seg", {role: "group", "data-i18n-aria": aria}, keys.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), "data-i18n": label(k), "data-i18n-title": `${label(k)}Key`, on: {click: () => click(k)}})));
  const act = (key, fn, ok = true) => h("button.pill", {type: "button", "data-i18n": key, "data-i18n-title": `${key}Key`, disabled: !ok, on: {click: fn}});
  const wand = s.shape === "wand";
  const amt = h("input.v.edamt", {type: "number", min: "1", max: "200", step: "1", value: String(s.amt), "data-i18n-aria": "ed.selAmt",
                                   on: {change: e => setSelOpt({amt: Math.max(1, Math.min(200, Math.round(Number(e.target.value) || 1)))})}});
  const has = !!E.sel;
  return [
    h("div.rftools.edselbar", {},
      seg("ed.selShape", SHAPES, k => s.shape === k, k => `ed.shape.${k}`, k => { E.poly = null; setSelOpt({shape: k}); drawStage(); }),
      seg("ed.selMode", SEL_MODES, k => E.selMode === k, k => `ed.sel.${k}`, k => { E.selMode = k; drawStage(); }),
      wand ? slider("ref.tol", 0, 128, s.tol, "", (v, p) => setSelOpt({tol: v}, p)) : null,
      wand ? h("button.pill.edtoggle", {type: "button", "aria-pressed": String(!!s.contig), "data-i18n": "ed.contig", "data-i18n-title": "ed.contigHint",
                                         on: {click: () => { setSelOpt({contig: !selset().contig}); drawStage(); }}}) : null,
      wand ? seg("ed.sample", ["all", "layer"], k => s.sample === k, k => `ed.sample.${k}`, k => { setSelOpt({sample: k}); drawStage(); }) : null,
      h("span.sp"),
      h("span.dim", {"data-i18n": s.shape === "poly" ? "ed.polyHint" : wand ? "ed.wandHint" : "ed.selHint"})),
    h("div.rftools.edselops", {},
      act("ed.selAll", selectAll),
      act("ed.deselect", () => setSel(null), has),
      act("ed.selInvert", invertSel, has),
      h("span.edsep"),
      h("label.rfbrush", {}, h("span", {"data-i18n": "ed.selAmt"}), amt, h("span.dim", {text: "px"})),
      act("ed.selGrow", () => growSel(selset().amt), has),
      act("ed.selShrink", () => growSel(-selset().amt), has),
      act("ed.selFeather", () => featherSel(selset().amt), has),
      h("span.edsep"),
      act("ed.selClear", clearSel, has),
      act("ed.selFill", fillSel, has),
      act("ed.selCopy", () => selToLayer(false), has),
      act("ed.selCut", () => selToLayer(true), has),
      act("ed.selCrop", cropToSel, has),
      act("ed.selTransform", () => setTool("transform")))];
}

// ── 變形（第 3 輪）──
// 有選取範圍：選到的那塊拿起來變形（原本的位置變透明）；沒選取：整層有內容的部分。
// 變形中那層在畫面上換成預覽，按「套用」（Enter）才真的改，「取消」（Esc）或 Ctrl+Z 放棄。
// 三種模式：一般（移動、縮放、旋轉、翻轉）、扭曲（四個角各自拖；拖邊＝歪斜）、彎曲（拖網格點）。
// 往後換（一般→扭曲→彎曲）接著現在的樣子；往回換先把目前的樣子固定下來，再從那裡開始
const XF_MODES = ["free", "distort", "warp"];
const GRID_NS = [3, 4, 5];                            // 彎曲的網格：幾個點（2×2、3×3、4×4 格）

function startXf() {
  const L = cur();
  if (!L.visible) { note(t("ed.hiddenLayer"), "warn"); return false; }
  const r = E.sel ? selBox() : alphaBox(pixelsOf(L.canvas).data, E.w, E.h);
  if (!r) { note(t(E.sel ? "ed.needSel" : "ed.emptyLayer"), "warn"); return false; }
  const crop = src => {
    const c = canvasOf(r.w, r.h);
    c.getContext("2d").drawImage(src, -r.x, -r.y);
    return c;
  };
  const t0 = {cx: r.x + r.w / 2, cy: r.y + r.h / 2, w: r.w, h: r.h, rot: 0, fx: false, fy: false};
  const float = crop(layerPart(L, true)), selFloat = E.sel ? crop(E.sel) : null;
  E.xf = {lid: L.id, float, selFloat, hole: E.sel ? layerPart(L, false) : canvasOf(E.w, E.h),
          mode: "free", t: t0, t0, q: null, g: null, gridN: 4, touched: false, orig: {float, selFloat},
          view: canvasOf(E.w, E.h), before: snap(), prev: E.tool === "transform" ? "view" : E.tool};
  E.xf.view.style.opacity = String(L.opacity);
  renderXf();
  return true;
}

function drawXfTo(ctx, src, t) {
  ctx.save();
  ctx.translate(t.cx, t.cy);
  ctx.rotate(t.rot);
  ctx.scale(t.fx ? -1 : 1, t.fy ? -1 : 1);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, -t.w / 2, -t.h / 2, t.w, t.h);
  ctx.restore();
}

// 照目前的模式把 src 畫上去；(ox, oy) 是畫布左上角在圖上的位置（固定下來時用），W×H 是畫布大小
function drawMapped(ctx, src, x, W = E.w, H = E.h, ox = 0, oy = 0) {
  if (x.mode === "free") {
    ctx.save();
    ctx.translate(-ox, -oy);
    drawXfTo(ctx, src, x.t);
    ctx.restore();
    return;
  }
  const map = xfMap(x);
  const out = renderMesh(src, W, H, (u, v) => { const [px, py] = map(u, v); return [px - ox, py - oy]; }, x.mode === "warp" ? 48 : 32);
  if (out) ctx.drawImage(out, 0, 0);
}

function renderXf() {
  const x = E.xf, ctx = x.view.getContext("2d");
  ctx.clearRect(0, 0, E.w, E.h);
  ctx.drawImage(x.hole, 0, 0);
  drawMapped(ctx, x.float, x);
}

// 往回換模式前：把目前的樣子畫成新的一塊（範圍是變形後的外框），從一般模式重新開始
function bakeXf() {
  const x = E.xf, map = xfMap(x), xs = [], ys = [];
  for (let j = 0; j <= 16; j++) for (let i = 0; i <= 16; i++) { const [px, py] = map(i / 16, j / 16); xs.push(px); ys.push(py); }
  const x0 = Math.floor(Math.max(-E.w, Math.min(...xs))), y0 = Math.floor(Math.max(-E.h, Math.min(...ys)));
  const x1 = Math.ceil(Math.min(2 * E.w, Math.max(...xs))), y1 = Math.ceil(Math.min(2 * E.h, Math.max(...ys)));
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const bake = src => {
    const c = canvasOf(w, h);
    drawMapped(c.getContext("2d"), src, x, w, h, x0, y0);
    return c;
  };
  x.float = bake(x.float);
  if (x.selFloat) x.selFloat = bake(x.selFloat);
  x.t = {cx: x0 + w / 2, cy: y0 + h / 2, w, h, rot: 0, fx: false, fy: false};
  x.mode = "free";
}

function setXfMode(m) {
  const x = E.xf;
  if (!x || x.mode === m || (m !== "free" && !meshSupported())) return;
  if (XF_MODES.indexOf(m) < XF_MODES.indexOf(x.mode)) bakeXf();
  const map = xfMap(x);
  if (m === "distort") x.q = quadOf(map);
  if (m === "warp") x.g = gridOf(map, x.gridN);
  x.mode = m;
  renderXf();
  drawStage();
}

function setGridN(n) {
  const x = E.xf;
  if (!x) return;
  x.gridN = n;
  if (x.mode === "warp") x.g = gridOf(xfMap(x), n);   // 換網格密度：照現在的彎曲重新取點
  renderXf();
  drawStage();
}

// 還原：回到剛開始變形的樣子（模式不變）
function resetXf() {
  const x = E.xf;
  Object.assign(x, {float: x.orig.float, selFloat: x.orig.selFloat, t: x.t0, touched: false});
  const mode = x.mode;
  x.mode = "free";
  if (mode === "distort") x.q = quadOf(xfMap(x));
  if (mode === "warp") x.g = gridOf(xfMap(x), x.gridN);
  x.mode = mode;
  renderXf();
  drawStage();
}

// 套用：那層換成變形後的樣子，選取範圍跟著變形；沒動過就當沒發生。next：之後換到哪個工具
function applyXf(next) {
  const x = E.xf;
  if (!x) return;
  E.xf = null;
  E.tool = next || x.prev;
  if (!x.touched) { drawStage(); return; }
  const c = canvasOf(E.w, E.h), ctx = c.getContext("2d");
  ctx.drawImage(x.hole, 0, 0);
  drawMapped(ctx, x.float, x);
  let sel = E.sel;
  if (x.selFloat) {
    const s = canvasOf(E.w, E.h);
    drawMapped(s.getContext("2d"), x.selFloat, x);
    sel = selOrNull(s);
  }
  E.hist.push(x.before);
  if (E.hist.length > HIST_MAX) E.hist.shift();
  E.fut = [];
  replaceCanvas(x.lid, c);
  E.sel = sel;
  changed();
}

function cancelXf() {
  const x = E.xf;
  if (!x) return;
  E.xf = null;
  E.tool = x.prev;
  drawStage();
}

function setXf(patch) {
  E.xf.t = {...E.xf.t, ...patch};
  E.xf.touched = true;
  renderXf();
  drawStage();
}

const deg = r => Math.round(r * 1800 / Math.PI) / 10;

function syncXfNums() {
  const tt = E.xf?.t;
  if (!tt || E.xf.mode !== "free") return;
  for (const [id, v] of [["edXfW", Math.round(tt.w)], ["edXfH", Math.round(tt.h)], ["edXfA", deg(tt.rot)]]) {
    const el = $(id);
    if (el && document.activeElement !== el) el.value = String(v);
  }
}

function xfBar() {
  const x = E.xf, tt = x.t, gl = meshSupported();
  const num = (id, key, v, min, max, set) => h("label.rfbrush", {},
    h("span", {"data-i18n": key}),
    h("input.v.edamt", {type: "number", id, min: String(min), max: String(max), step: "1", value: String(v), "data-i18n-aria": key,
                        on: {change: e => { const n = Number(e.target.value); if (Number.isFinite(n)) set(n); }}}));
  const free = x.mode === "free";
  return h("div.rftools.edxfbar", {},
    h("div.seg", {role: "group", "data-i18n-aria": "ed.xfMode"}, XF_MODES.map(k => h("button", {
      type: "button", "aria-pressed": String(x.mode === k), "data-i18n": `ed.xf.${k}`,
      "data-i18n-title": k === "free" || gl ? `ed.xf.${k}Key` : "ed.xfNoGl", disabled: k !== "free" && !gl,
      on: {click: () => setXfMode(k)}}))),
    free ? num("edXfW", "ed.width", Math.round(tt.w), 1, 16384, v => setXf({w: Math.max(1, v)})) : null,
    free ? num("edXfH", "ed.height", Math.round(tt.h), 1, 16384, v => setXf({h: Math.max(1, v)})) : null,
    free ? num("edXfA", "ed.angle", deg(tt.rot), -180, 180, v => setXf({rot: v * Math.PI / 180})) : null,
    free ? h("button.pill", {type: "button", "data-i18n": "ed.rotL", on: {click: () => setXf({rot: E.xf.t.rot - Math.PI / 2})}}) : null,
    free ? h("button.pill", {type: "button", "data-i18n": "ed.rotR", on: {click: () => setXf({rot: E.xf.t.rot + Math.PI / 2})}}) : null,
    free ? h("button.pill", {type: "button", "data-i18n": "ed.flipH", on: {click: () => setXf({fx: !E.xf.t.fx})}}) : null,
    free ? h("button.pill", {type: "button", "data-i18n": "ed.flipV", on: {click: () => setXf({fy: !E.xf.t.fy})}}) : null,
    x.mode === "warp" ? h("div.seg", {role: "group", "data-i18n-aria": "ed.grid"}, GRID_NS.map(n => h("button", {
      type: "button", "aria-pressed": String(x.gridN === n), text: `${n - 1}×${n - 1}`, "data-i18n-title": "ed.gridHint",
      on: {click: () => setGridN(n)}}))) : null,
    h("button.pill", {type: "button", "data-i18n": "ed.xfReset", on: {click: resetXf}}),
    h("span.edsep"),
    h("button.pill", {type: "button", "data-i18n": "ed.cropCancel", "data-i18n-title": "ed.xfCancelKey", on: {click: cancelXf}}),
    h("button.pill.primary", {type: "button", "data-i18n": "ed.xfApply", "data-i18n-title": "ed.cropApplyKey", on: {click: () => applyXf()}}),
    h("span.sp"),
    h("span.dim", {"data-i18n": `ed.xfHint.${x.mode}`}));
}

// 扭曲的四邊中點（拖它＝歪斜：那條邊的兩個角一起動）
const quadMids = q => q.map((p, i) => { const n = q[(i + 1) % 4]; return [(p[0] + n[0]) / 2, (p[1] + n[1]) / 2]; });
const centroid = pts => [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];

// 彎曲的外框（沿著邊取點），判斷游標在不在裡面、畫框線用
function warpOutline(map) {
  const out = [], S = 24;
  for (let i = 0; i < S; i++) out.push(map(i / S, 0));
  for (let i = 0; i < S; i++) out.push(map(1, i / S));
  for (let i = S; i > 0; i--) out.push(map(i / S, 1));
  for (let i = S; i > 0; i--) out.push(map(0, i / S));
  return out;
}

// 扭曲、彎曲：按下的地方是哪種拖法。控制點（角、網格點）、邊的中點（扭曲）、裡面＝整個移動、外面＝整個旋轉
function meshHit(x, p, tol) {
  if (x.mode === "distort") {
    const i = nearPoint(x.q, p, tol);
    if (i >= 0) return {kind: "pt", i};
    const j = nearPoint(quadMids(x.q), p, tol);
    if (j >= 0) return {kind: "edge", i: j};
    return {kind: inPoly(x.q, p) ? "move" : "rotate"};
  }
  const i = nearPoint(x.g.pts, p, tol);
  if (i >= 0) return {kind: "pt", i};
  return {kind: inPoly(warpOutline(xfMap(x)), p) ? "move" : "rotate"};
}

function meshDrag(x, d, p, shift) {
  const pts0 = d.pts0, dx = p[0] - d.start[0], dy = p[1] - d.start[1];
  let pts;
  if (d.kind === "pt") pts = pts0.map((q, i) => (i === d.i ? [q[0] + dx, q[1] + dy] : q));
  else if (d.kind === "edge") pts = pts0.map((q, i) => (i === d.i || i === (d.i + 1) % 4 ? [q[0] + dx, q[1] + dy] : q));
  else if (d.kind === "move") pts = shiftPts(pts0, dx, dy);
  else {
    const c = centroid(pts0);
    let a = Math.atan2(p[1] - c[1], p[0] - c[0]) - Math.atan2(d.start[1] - c[1], d.start[0] - c[0]);
    if (shift) a = Math.round(a / (Math.PI / 12)) * (Math.PI / 12);
    pts = turnPts(pts0, c, a);
  }
  if (x.mode === "distort") x.q = pts;
  else x.g = {...x.g, pts};
}

function wireXf(c) {
  if (E.tool !== "transform" || !E.xf) return;
  let drag = null;
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * E.w / r.width, (e.clientY - r.top) * E.h / r.height];
  };
  const tol = () => 10 * E.w / (c.getBoundingClientRect().width || E.w);
  c.addEventListener("pointerdown", e => {
    const x = E.xf;
    if (e.button !== 0 || E.space || !x) return;
    e.preventDefault();
    try { c.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
    const p = at(e);
    if (x.mode === "free") drag = {mode: xfHit(x.t, p, tol()), start: p, t0: x.t};
    else drag = {...meshHit(x, p, tol()), start: p, pts0: x.mode === "distort" ? x.q : x.g.pts};
  });
  c.addEventListener("pointermove", e => {
    const x = E.xf;
    if (!x) return;
    const p = at(e);
    if (!drag) {
      c.dataset.hit = x.mode === "free" ? xfHit(x.t, p, tol()) : meshHit(x, p, tol()).kind;
      return;
    }
    // 一般模式：角預設照比例，按住 Shift 自由；旋轉時按住 Shift 每 15°
    if (x.mode === "free") x.t = xfDrag(drag.mode, drag.t0, drag.start, p, {free: e.shiftKey, snap: e.shiftKey});
    else meshDrag(x, drag, p, e.shiftKey);
    x.touched = true;
    renderXf();
    drawOverlay(c);
    syncXfNums();
  });
  const end = () => { drag = null; };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
}

// 變形框：一般＝外框（跟著旋轉）＋八個把手；扭曲＝四邊形＋四個角（方的）與邊的中點（圓的）；
// 彎曲＝彎過的網格線＋網格點
function drawXfBox(ctx, x, px) {
  const line = pts => {
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  };
  const stroke = () => {
    ctx.lineWidth = 3 * px;
    ctx.strokeStyle = "rgba(0, 0, 0, .6)";
    ctx.stroke();
    ctx.lineWidth = 1.5 * px;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  };
  const s = 9 * px;
  const square = ([hx, hy]) => {
    ctx.fillStyle = "#fff";
    ctx.fillRect(hx - s / 2, hy - s / 2, s, s);
    ctx.lineWidth = px;
    ctx.strokeStyle = "rgba(0, 0, 0, .6)";
    ctx.strokeRect(hx - s / 2, hy - s / 2, s, s);
  };
  const dot = ([hx, hy], r = s / 2) => {
    ctx.beginPath();
    ctx.arc(hx, hy, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.lineWidth = px;
    ctx.strokeStyle = "rgba(0, 0, 0, .6)";
    ctx.stroke();
  };
  if (x.mode === "distort") {
    line(x.q);
    ctx.closePath();
    stroke();
    quadMids(x.q).forEach(p => dot(p, s * 0.4));
    x.q.forEach(square);
    return;
  }
  if (x.mode === "warp") {
    const map = xfMap(x), n = x.g.n, S = 24;
    for (let k = 0; k < n; k++) {
      line(Array.from({length: S + 1}, (_, i) => map(i / S, k / (n - 1))));
      stroke();
      line(Array.from({length: S + 1}, (_, i) => map(k / (n - 1), i / S)));
      stroke();
    }
    x.g.pts.forEach(p => dot(p));
    return;
  }
  const tt = x.t;
  line(xfCorners(tt));
  ctx.closePath();
  stroke();
  const cos = Math.cos(tt.rot), sin = Math.sin(tt.rot);
  for (const [dx, dy] of Object.values(HANDLES)) {
    const u = dx * tt.w / 2, v = dy * tt.h / 2;
    square([tt.cx + u * cos - v * sin, tt.cy + u * sin + v * cos]);
  }
}

// ── 調整（第 4 輪）──
// 選中的那層（有選取時只調選到的地方）照拉桿的值算一份預覽，畫面上那層暫時換成預覽；
// 「套用」（Enter）才真的改，「取消」（Esc）或 Ctrl+Z 放棄。全部歸零＝沒改
function startAdj() {
  const L = cur();
  if (!L.visible) { note(t("ed.hiddenLayer"), "warn"); return false; }
  const view = canvasOf(E.w, E.h);
  view.getContext("2d").drawImage(L.canvas, 0, 0);
  view.style.opacity = String(L.opacity);
  E.adj = {lid: L.id, src: pixelsOf(L.canvas), mask: E.sel ? pixelsOf(E.sel).data : null, out: new ImageData(E.w, E.h),
           view, v: {bri: 0, con: 0, sat: 0, hue: 0}, timer: 0, before: snap(), prev: E.tool === "adjust" ? "view" : E.tool};
  return true;
}

const adjZero = a => ADJ.every(([k]) => !a.v[k]);

function renderAdj() {
  const a = E.adj;
  if (!a) return;
  clearTimeout(a.timer);
  a.timer = 0;
  adjustPixels(a.src.data, adjustMatrix(a.v), a.out.data, a.mask);
  a.view.getContext("2d").putImageData(a.out, 0, 0);
}

// 拖拉桿時：這一輪事件跑完才重算一次（2MP 的圖算一次幾十毫秒，不要每個事件都算）
function scheduleAdj() {
  if (E.adj && !E.adj.timer) E.adj.timer = setTimeout(renderAdj, 0);
}

function applyAdj() {
  const a = E.adj;
  if (!a) return;
  if (adjZero(a)) { cancelAdj(); return; }
  renderAdj();                                       // 拖到一半就套用：先把最後的值算完
  E.adj = null;
  E.tool = a.prev;
  E.hist.push(a.before);
  if (E.hist.length > HIST_MAX) E.hist.shift();
  E.fut = [];
  replaceCanvas(a.lid, a.view);
  changed();
}

function cancelAdj() {
  const a = E.adj;
  if (!a) return;
  clearTimeout(a.timer);
  E.adj = null;
  E.tool = a.prev;
  drawStage();
}

function adjBar() {
  const a = E.adj;
  return h("div.rftools.edadjbar", {},
    ...ADJ.map(([k, min, max, unit]) => slider(`ed.adj.${k}`, min, max, a.v[k], unit, v => { a.v[k] = v; scheduleAdj(); })),
    h("button.pill", {type: "button", "data-i18n": "ed.xfReset", disabled: adjZero(a),
                      on: {click: () => { ADJ.forEach(([k]) => { a.v[k] = 0; }); renderAdj(); drawStage(); }}}),
    h("span.edsep"),
    h("button.pill", {type: "button", "data-i18n": "ed.cropCancel", "data-i18n-title": "ed.xfCancelKey", on: {click: cancelAdj}}),
    h("button.pill.primary", {type: "button", "data-i18n": "ed.xfApply", "data-i18n-title": "ed.cropApplyKey", on: {click: applyAdj}}),
    h("span.sp"),
    h("span.dim", {"data-i18n": E.sel ? "ed.adjHintSel" : "ed.adjHint"}));
}

// ── 仿製印章（第 4 輪）──
function setCloneSrc(p) {
  E.clone = {src: p, off: null, mark: p};
  note(t("ed.cloneSrcSet"));
  drawStage();
}

// 這一筆要蓋上去的內容：取樣的來源（所有看得到的圖層合起來，或這一層畫之前的樣子）整張平移「位移」那麼多
function cloneSource(orig) {
  const src = cloneset().sample === "layer" ? orig : flatten();
  const c = canvasOf(E.w, E.h);
  c.getContext("2d").drawImage(src, E.clone.off[0], E.clone.off[1]);
  return c;
}

// 取樣圈跟著游標：位移已經定了就在「游標－位移」，還沒定就在取樣點
function cloneMark(c, p) {
  if (!E.clone.src) return;
  const off = cloneset().aligned ? E.clone.off : null;
  E.clone.mark = off ? [p[0] - off[0], p[1] - off[1]] : E.clone.src;
  drawOverlay(c);
}

function drawCloneMark(ctx, px) {
  const m = E.clone.mark || E.clone.src;
  if (!m) return;
  const r = Math.max(4 * px, bset().size / 2), arm = 6 * px;
  for (const [w, color] of [[3 * px, "rgba(0, 0, 0, .6)"], [1.5 * px, "#fff"]]) {
    ctx.lineWidth = w;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(m[0], m[1], r, 0, Math.PI * 2);
    ctx.moveTo(m[0] - arm, m[1]); ctx.lineTo(m[0] + arm, m[1]);
    ctx.moveTo(m[0], m[1] - arm); ctx.lineTo(m[0], m[1] + arm);
    ctx.stroke();
  }
}

// ── 文字（第 5 輪）──
// 文字圖層＝一層 canvas＋文字的設定（L.text）。點畫面空白處新增、點已有的文字開始改、拖著移動。
// 打字、改樣式直接畫在這層上（這一段開始時換一張新的 canvas，舊的留給復原），
// 結束這一段（點別的地方、換工具、按「完成」或 Esc）才記一步復原。字刪光＝這層不要了
const firstLine = str => String(str || "").split("\n")[0].trim().slice(0, 24);

function textAt(p) {
  for (let i = E.layers.length - 1; i >= 0; i--) {
    const L = E.layers[i], b = L.text?.box;
    if (L.visible && b && p[0] >= b.x && p[0] <= b.x + b.w && p[1] >= b.y && p[1] <= b.y + b.h) return L;
  }
  return null;
}

function paintText(L) {
  const ctx = L.canvas.getContext("2d");
  ctx.clearRect(0, 0, E.w, E.h);
  L.text = {...L.text, box: drawText(ctx, L.text)};
}

function startText(L, before = snap(), created = false) {
  if (!created) {
    const nc = canvasOf(E.w, E.h);
    nc.className = "edlayer";
    nc.style.opacity = String(L.opacity);
    nc.getContext("2d").drawImage(L.canvas, 0, 0);
    if (L.canvas.isConnected) L.canvas.replaceWith(nc);
    E.layers = E.layers.map(x => (x.id === L.id ? {...x, canvas: nc} : x));
  }
  E.active = L.id;
  E.txt = {lid: L.id, before, created, start: JSON.stringify(L.text), autoName: L.name === t("ed.textLayer") || L.name === firstLine(L.text.str)};
}

function newText(p) {
  const before = snap();
  const T = {...tset(), str: t("ed.textDefault"), x: Math.round(p[0]), y: Math.round(p[1])};
  const L = {id: newId(), name: firstLine(T.str), canvas: canvasOf(E.w, E.h), visible: true, opacity: 1, text: T};
  paintText(L);
  E.layers.splice(E.layers.indexOf(cur()) + 1, 0, L);
  startText(L, before, true);
  textChanged();
  drawStage();
  drawPanel();
  const ta = $("edText");
  if (ta) { ta.focus(); ta.select(); }
}

const txtLayer = () => E.layers.find(x => x.id === E.txt?.lid);

// 改了文字或樣式：重畫這層、排自動存檔（這層的像素換了，要再送）
function textChanged() {
  const L = txtLayer();
  if (!L) return;
  E.onDisk.delete(L.id);
  E.pending = true;
  E.rev++;
  clearTimeout(E.saveTimer);
  E.saveTimer = setTimeout(autoSave, 1200);
  const ov = document.querySelector(".edover");
  if (ov) drawOverlay(ov);
}

// 樣式：在改的文字就改那段文字，沒有就改「下一個新文字」的預設
function setT(patch, redraw = true) {
  const L = txtLayer();
  if (L) {
    L.text = {...L.text, ...patch};
    paintText(L);
    textChanged();
  }
  const style = Object.fromEntries(Object.entries(patch).filter(([k]) => TEXT_STYLE.includes(k)));
  if (Object.keys(style).length) { E.ui.etext = {...tset(), ...style}; if (redraw) saveUi({etext: E.ui.etext}); }
  if (redraw) drawStage();
}

// 結束這一段：有改才記一步復原；字刪光＝不要這層（新加的就當沒加過）
function endText(redraw = true) {
  const s = E.txt;
  if (!s) return;
  E.txt = null;
  const L = E.layers.find(x => x.id === s.lid);
  if (!L) return;
  if (!String(L.text?.str || "").trim()) {
    if (s.created) { restore(s.before); changed(); return; }
    E.hist.push(s.before);
    E.fut = [];
    E.layers = E.layers.filter(x => x.id !== L.id);
    E.active = (E.layers[E.layers.length - 1] || {}).id;
    changed();
    return;
  }
  if (s.autoName) L.name = firstLine(L.text.str) || t("ed.textLayer");
  if (s.created || JSON.stringify(L.text) !== s.start) {
    E.hist.push(s.before);
    if (E.hist.length > HIST_MAX) E.hist.shift();
    E.fut = [];
    changed();
  } else if (redraw) drawStage();
}

function wireText(c) {
  if (E.tool !== "text") return;
  let drag = null;
  const at = e => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * E.w / r.width, (e.clientY - r.top) * E.h / r.height];
  };
  c.addEventListener("pointerdown", e => {
    if (e.button !== 0 || E.space) return;
    e.preventDefault();
    const p = at(e), hit = textAt(p);
    if (hit && E.txt?.lid === hit.id) {                // 拖著正在改的文字：移動
      try { c.setPointerCapture(e.pointerId); } catch { /* 合成事件沒有真的指標 */ }
      drag = {start: p, x: hit.text.x, y: hit.text.y};
      return;
    }
    endText(false);
    if (hit) { startText(hit); drawStage(); drawPanel(); $("edText")?.focus(); return; }
    newText(p);
  });
  c.addEventListener("pointermove", e => {
    const p = at(e);
    if (!drag) { c.dataset.hit = textAt(p) ? "move" : ""; return; }
    setT({x: Math.round(drag.x + p[0] - drag.start[0]), y: Math.round(drag.y + p[1] - drag.start[1])}, false);
  });
  const end = () => { drag = null; };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
}

// 每個文字圖層畫一個虛線框，正在改的那個是實線
function drawTextBoxes(ctx, px) {
  for (const L of E.layers) {
    const b = L.text?.box;
    if (!b || !L.visible) continue;
    const on = E.txt?.lid === L.id;
    ctx.setLineDash(on ? [] : [5 * px, 4 * px]);
    for (const [w, color] of [[3 * px, "rgba(0, 0, 0, .55)"], [1.5 * px, on ? "#fff" : "rgba(255, 255, 255, .8)"]]) {
      ctx.lineWidth = w;
      ctx.strokeStyle = color;
      ctx.strokeRect(b.x - 4 * px, b.y - 4 * px, b.w + 8 * px, b.h + 8 * px);
    }
  }
  ctx.setLineDash([]);
}

// 電腦裡的字型（Chrome／Edge 支援，第一次會問要不要允許）
async function listFonts() {
  try {
    const all = await window.queryLocalFonts();
    E.fonts = [...new Set(all.map(f => f.family))].sort((a, b) => a.localeCompare(b));
    note(t("ed.fontsListed", {n: E.fonts.length}));
  } catch (e) {
    note(t("ed.fontsErr", {msg: e.message || e}), "warn");
  }
  drawStage();
}

function textBar() {
  const L = txtLayer(), T = L ? L.text : tset();
  const toggle = (key, k, on) => h("button.pill.edtoggle", {type: "button", "aria-pressed": String(!!on), "data-i18n": key,
                                                           on: {click: () => setT({[k]: !on})}});
  const num = (key, k, v, min, max) => h("label.rfbrush", {}, h("span", {"data-i18n": key}),
    h("input.v.edamt", {type: "number", min: String(min), max: String(max), step: "1", value: String(v), "data-i18n-aria": key,
                        on: {change: e => setT({[k]: Math.max(min, Math.min(max, Math.round(Number(e.target.value) || min)))})}}));
  const swatch = (key, k, v) => h("label.rfbrush", {}, h("span", {"data-i18n": key}),
    h("input.rfswatch", {type: "color", value: v, "data-i18n-aria": key,
                         on: {input: e => setT({[k]: e.target.value}, false), change: e => setT({[k]: e.target.value})}}));
  return h("div.rftools.edtextbar", {},
    L ? h("textarea.edtext", {id: "edText", rows: "2", spellcheck: "false", "data-i18n-aria": "ed.textInput", value: T.str,
                              on: {input: e => setT({str: e.target.value}, false)}}) : null,
    h("label.rfbrush", {}, h("span", {"data-i18n": "ed.font"}),
      h("input.v.edfont", {type: "text", list: "edFonts", value: T.font, spellcheck: "false", "data-i18n-aria": "ed.font",
                           on: {change: e => setT({font: e.target.value.trim() || FONTS[0]})}}),
      h("datalist", {id: "edFonts"}, (E.fonts || FONTS).map(f => h("option", {value: f})))),
    window.queryLocalFonts && !E.fonts ? h("button.pill", {type: "button", "data-i18n": "ed.listFonts", "data-i18n-title": "ed.listFontsHint",
                                                           on: {click: listFonts}}) : null,
    num("ed.fontSize", "size", T.size, 4, 2000),
    swatch("ref.color", "color", T.color),
    toggle("ed.bold", "bold", T.bold),
    toggle("ed.italic", "italic", T.italic),
    h("div.seg", {role: "group", "data-i18n-aria": "ed.align"}, ["left", "center", "right"].map(k => h("button", {
      type: "button", "aria-pressed": String((T.align || "left") === k), "data-i18n": `ed.align.${k}`, on: {click: () => setT({align: k})}}))),
    num("ed.stroke", "stroke", T.stroke || 0, 0, 200),
    T.stroke ? swatch("ed.strokeColor", "scolor", T.scolor || "#000000") : null,
    toggle("ed.shadow", "shadow", T.shadow),
    L ? h("button.pill.primary", {type: "button", "data-i18n": "ed.textDone", "data-i18n-title": "ed.textDoneKey", on: {click: () => endText(true)}}) : null,
    h("span.sp"),
    h("span.dim", {"data-i18n": L ? "ed.textEditHint" : "ed.textHint"}));
}

// ── 浮水印（第 5 輪）──
// 設定記在介面偏好（每張圖、批次輸出共用）；「加到這張圖」做成最上面一層「浮水印」，再按一次是更新那層
function setWm(patch, redraw = true) {
  E.ui.ewm = {...wmset(E.ui), ...patch};
  saveUi({ewm: E.ui.ewm});
  if (redraw) drawPanel();
}

function applyWm() {
  const wm = wmset(E.ui);
  if (!wmReady(wm, logoNow())) { note(t("ed.wmNotReady"), "warn"); return; }
  const c = canvasOf(E.w, E.h);
  drawWatermark(c.getContext("2d"), E.w, E.h, wm, logoNow());
  const old = E.layers.find(L => L.wm);
  commit(() => {
    if (old) E.layers = E.layers.map(x => (x.id === old.id ? {...x, canvas: c, opacity: wm.opacity / 100, visible: true} : x));
    else {
      const L = {id: newId(), name: t("ed.wmLayer"), canvas: c, visible: true, opacity: wm.opacity / 100, wm: true};
      E.layers.push(L);
      E.active = L.id;
    }
  });
  note(t(old ? "ed.wmUpdated" : "ed.wmAdded"));
}

function wmSection() {
  const wm = wmset(E.ui), img = wm.type === "image", logo = logoNow(), has = E.layers.some(L => L.wm);
  const [btn, hint] = info("ed.wmInfo");
  const pick = h("input", {type: "file", accept: "image/*", hidden: true, on: {change: async e => {
    const f = e.target.files?.[0];
    if (!f) return;
    try { await setLogo(f); } catch (err) { note(t("ed.loadErr", {msg: err.message || err}), "err"); }
    drawPanel();
  }}});
  const sl = (key, k, min, max, unit) => h("div.edlrow", {}, h("span.dim", {"data-i18n": key}),
    h("input", {type: "range", min: String(min), max: String(max), step: "1", value: String(wm[k]), "data-i18n-aria": key,
                on: {input: e => { e.target.nextElementSibling.textContent = `${e.target.value}${unit}`; },
                     change: e => setWm({[k]: Number(e.target.value)})}}),
    h("span.mono.rfnum", {text: `${wm[k]}${unit}`}));
  return h("section.rfsec.edwm", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ed.wm"}), btn, h("span.sp"),
      h("div.seg", {role: "group", "data-i18n-aria": "ed.wmType"}, ["text", "image"].map(k => h("button", {
        type: "button", "aria-pressed": String(wm.type === k), "data-i18n": `ed.wmType.${k}`, on: {click: () => setWm({type: k})}})))),
    hint,
    img ? h("div.edwmlogo", {},
      logo ? h("img", {src: `/api/edit/wmlogo?v=${logoVer()}`, alt: ""}) : h("span.dim", {"data-i18n": "ed.wmNoLogo"}),
      h("span.sp"),
      pick,
      h("button.mini", {type: "button", "data-i18n": logo ? "ed.wmChange" : "ed.wmPick", on: {click: () => pick.click()}}),
      logo ? h("button.mini", {type: "button", "data-i18n": "ed.wmRemove", on: {click: async () => { await clearLogo(); drawPanel(); }}}) : null)
      : h("div.edwmtext", {},
        h("input.find", {type: "text", value: wm.text, "data-i18n-placeholder": "ed.wmTextPh", "data-i18n-aria": "ed.wmText",
                         on: {change: e => setWm({text: e.target.value})}}),
        h("input.rfswatch", {type: "color", value: wm.color, "data-i18n-aria": "ref.color", on: {change: e => setWm({color: e.target.value})}}),
        h("button.mini.edtoggle", {type: "button", "aria-pressed": String(!!wm.bold), "data-i18n": "ed.bold", on: {click: () => setWm({bold: !wm.bold})}}),
        h("button.mini.edtoggle", {type: "button", "aria-pressed": String(!!wm.shadow), "data-i18n": "ed.shadow", on: {click: () => setWm({shadow: !wm.shadow})}})),
    img ? null : h("div.edlrow", {}, h("span.dim", {"data-i18n": "ed.font"}),
      h("input.v.edfont", {type: "text", list: "edWmFonts", value: wm.font, spellcheck: "false", "data-i18n-aria": "ed.font",
                           on: {change: e => setWm({font: e.target.value.trim() || FONTS[0]})}}),
      h("datalist", {id: "edWmFonts"}, (E.fonts || FONTS).map(f => h("option", {value: f})))),
    h("div.edwmpos", {},
      h("div.edgrid9", {role: "group", "data-i18n-aria": "ed.wmPos"}, WM_POS.map(k => h("button", {
        type: "button", "aria-pressed": String(wm.pos === k), "data-i18n-aria": `ed.wmPos.${k}`, "data-i18n-title": `ed.wmPos.${k}`,
        on: {click: () => setWm({pos: k})}}))),
      h("button.mini.edtoggle", {type: "button", "aria-pressed": String(wm.pos === "tile"), "data-i18n": "ed.wmTile", "data-i18n-title": "ed.wmTileHint",
                                 on: {click: () => setWm({pos: wm.pos === "tile" ? "br" : "tile"})}})),
    sl("ed.wmSize", "size", 2, 100, "%"),
    sl("ed.bopacity", "opacity", 5, 100, "%"),
    wm.pos === "tile" ? sl("ed.angle", "angle", -90, 90, "°") : sl("ed.wmMargin", "margin", 0, 20, "%"),
    h("div.edlbtns", {},
      h("button.mini", {type: "button", "data-i18n": has ? "ed.wmUpdate" : "ed.wmApply", disabled: !wmReady(wm, logo), on: {click: applyWm}})));
}

// ── 編輯檔（.kra）──
// 存成 .kra：先把手上在做的（變形、調整、打字）結束、存好工作檔，再請伺服器照工作檔寫成 .kra。
// 開 .kra：整個檔案送給伺服器，它建好工作檔（CoCo 存的接回原本那張圖，別的程式存的先把合併的圖存進圖庫），
// 這裡再打開那張圖、直接接著做
const LOCK = [["rect", {x: 3.5, y: 7, width: 9, height: 7, rx: 1.5, fill: "currentColor"}],
              ["path", {d: "M5.5 7V5a2.5 2.5 0 0 1 5 0v2", fill: "none", stroke: "currentColor", "stroke-width": 1.5}]];

async function saveKra() {
  if (E.xf) applyXf();
  if (E.adj) applyAdj();
  if (E.txt) endText(false);
  const folder = await pickFolder(E.ui?.ekradir || E.ui?.efolder || "");
  if (!folder) return;
  saveUi({ekradir: folder});
  clearTimeout(E.saveTimer);
  if (!E.pid && !SAVE.running()) await saveProject();     // 還沒存過（沒改任何東西）也要先有工作檔
  if (!await SAVE.flush() || !E.pid) { note(t("ed.saveErr", {msg: E.saveErr || "—"}), "err"); return; }
  note(t("ed.kraSaving"));
  const d = await post("/api/edit/kra/export", {id: E.pid, folder}).catch(e => ({error: e.message}));
  note(d.error ? t("ed.kraErr", {msg: d.error}) : t("ed.kraSaved", {path: d.path}), d.error ? "err" : "ok");
}

function readDataUrl(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = () => rej(r.error);
    r.readAsDataURL(file);
  });
}

async function openKra(file) {
  const say = (text, kind = "ok") => { if (E.src) note(text, kind); else { const m = $("edEmptyMsg"); if (m) { m.textContent = text; m.dataset.kind = kind; } } };
  // 手上這張還有沒存的：先存到最新（開回來的 .kra 可能就是這張，之後不能再存舊的蓋過去）；存不了就不開
  if (!await settleBeforeLeaving()) return;
  say(t("ed.kraOpening"));
  let data;
  try { data = await readDataUrl(file); } catch (e) { say(t("ed.kraErr", {msg: e.message || e}), "err"); return; }
  const send = replace => post("/api/edit/kra/import", {name: file.name, data, replace}).catch(e => ({error: e.message}));
  let d = await send(false);
  if (d.exists) {
    // 那張圖已經有編輯到一半的工作檔：問要不要換成這個 .kra
    const ok = await confirmDialog({title: t("ed.kraReplaceTitle"), body: [h("p", {text: t(d.imported ? "ed.kraReplaceImported" : "ed.kraReplaceBody", {name: d.filename})})], ok: t("ed.kraReplaceOk")});
    if (!ok) { say(""); return; }
    d = await send(true);
  }
  if (d.error) { say(t("ed.kraErr", {msg: d.error}), "err"); return; }
  E.autoResume = true;
  E.src = null;                                      // 同一張也要重新載入
  E.layers = [];
  E.pending = false;
  openEditor(d.src_id);
  if (d.skipped?.length) setTimeout(() => note(t("ed.kraSkipped", {n: d.skipped.length, names: d.skipped.slice(0, 5).join("、")}), "warn"), 1500);
}

function kraOpenButton(sel = "button.mini") {
  const pick = h("input", {type: "file", accept: ".kra", hidden: true, on: {change: e => { const f = e.target.files?.[0]; if (f) openKra(f); e.target.value = ""; }}});
  return h("span.edkraopen", {}, pick,
    h(sel, {type: "button", "data-i18n": "ed.kraOpen", "data-i18n-title": "ed.kraOpenHint", on: {click: () => pick.click()}}),
    E.src ? null : h("p.hint.edmsg", {id: "edEmptyMsg", role: "status"}));
}

function kraSection() {
  const [btn, hint] = info("ed.kraInfo");
  return h("section.rfsec.edkra", {},
    h("div.rflab", {}, h("b", {"data-i18n": "ed.kra"}), btn),
    hint,
    h("div.edlbtns", {},
      h("button.mini", {type: "button", "data-i18n": "ed.kraSave", "data-i18n-title": "ed.kraSaveHint", on: {click: saveKra}}),
      kraOpenButton()));
}

// ── 左邊：圖層、輸出 ──
function drawPanel() {
  const box = $("edPanel");
  if (!box || !E.src) return;
  const [layBtn, layHint] = info("ed.layersHint");
  const L0 = cur();
  const rows = [...E.layers].reverse().map(L => {
    const thumb = canvasOf(44, Math.max(1, Math.round(44 * E.h / E.w)));
    thumb.getContext("2d").drawImage(L.canvas, 0, 0, thumb.width, thumb.height);
    const name = h("span.edlname", {text: L.name, title: t("ed.renameHint")});
    const row = h("div.edlayerrow", {role: "option", tabindex: "0", "aria-selected": String(L.id === L0.id), "data-hidden": L.visible ? null : "1",
                                     on: {click: e => { if (e.target.closest("button, input")) return; if (E.active !== L.id) { E.active = L.id; drawPanel(); } },
                                          keydown: e => { if (e.key === "Enter" && !e.target.closest("input")) { E.active = L.id; drawPanel(); } }}},
      h("button.edeye", {type: "button", "aria-pressed": String(L.visible), "data-i18n-aria": "ed.visible", "data-i18n-title": "ed.visible",
                         on: {click: () => setLayer(L, {visible: !L.visible})}}, L.visible ? "◉" : "○"),
      h("span.edthumb", {}, thumb),
      L.text ? h("span.edbadge", {text: "T", "data-i18n-title": "ed.textLayerHint"}) : L.wm ? h("span.edbadge", {text: "©", "data-i18n-title": "ed.wmLayerHint"}) : null,
      name,
      L.lock ? h("span.edbadge.edlockb", {"data-i18n-title": "ed.lockAlphaOn"}, icon(LOCK, 12)) : null,
      L.opacity < 1 ? h("span.dim.mono.edop", {text: `${Math.round(L.opacity * 100)}%`}) : null);
    // 點選中那層的名字：直接改名
    name.addEventListener("click", () => {
      if (L.id !== E.active) return;
      const inp = h("input.edlnamein", {type: "text", value: L.name, "data-i18n-aria": "ed.rename", on: {
        keydown: ev => { if (ev.key === "Enter") ev.currentTarget.blur(); if (ev.key === "Escape") { ev.currentTarget.value = L.name; ev.currentTarget.blur(); } },
        blur: ev => { const v = ev.currentTarget.value.trim(); if (v && v !== L.name) setLayer(L, {name: v}); else drawPanel(); }}});
      name.replaceWith(inp);
      inp.focus();
      inp.select();
    });
    return row;
  });
  let opBefore = null;
  const opNum = h("span.mono.rfnum", {text: `${Math.round(L0.opacity * 100)}%`});
  const out = E.out || {w: E.w, h: E.h};
  const sizeIn = which => h("input.v", {type: "number", min: "1", max: "8192", step: "1", value: String(which === "w" ? out.w : out.h),
    on: {change: e => {
      const v = Math.max(1, Math.min(8192, Math.round(Number(e.target.value) || 1)));
      E.out = {...out, ...(E.lock ? lockSize(E.w, E.h, which, v) : {[which]: v})};
      drawPanel();
    }}});
  const q = Number(E.ui?.equality) || 92;
  const qNum = h("span.mono.rfnum", {text: String(q)});
  const [outBtn, outHint] = info("ed.outHint");
  box.replaceChildren(...[
    h("div.rhd", {},
      h("h2.title", {id: "edTitle", "data-i18n": "nav.output"}),
      h("span.sp"),
      h("button.mini", {type: "button", "data-i18n": "ed.close", "data-i18n-title": "ed.closeHint", on: {click: closeEdit}})),
    h("div.rfsrc", {},
      h("img", {src: `/api/thumb?id=${E.src.id}`, alt: ""}),
      h("div.rfsrcinfo", {},
        h("b", {text: E.src.filename, title: E.src.filename}),
        h("span.dim", {text: `${E.w} × ${E.h}`}),
        h("span.dim.small", {id: "edSaved", role: "status"}))),
    h("section.rfsec.edlayers", {},
      h("div.rflab", {}, h("b", {"data-i18n": "ed.layers"}), layBtn),
      layHint,
      h("div.edlayerlist", {role: "listbox", "data-i18n-aria": "ed.layers"}, rows),
      h("div.edlrow", {},
        h("span.dim", {"data-i18n": "ed.opacity"}),
        h("input", {type: "range", min: "0", max: "100", step: "1", value: String(Math.round(L0.opacity * 100)), "data-i18n-aria": "ed.opacity",
                    on: {pointerdown: () => { opBefore = snap(); }, focus: () => { opBefore = snap(); },
                         input: e => {
                           const v = Number(e.target.value) / 100;
                           L0.opacity = v;
                           L0.canvas.style.opacity = String(v);
                           opNum.textContent = `${Math.round(v * 100)}%`;
                         },
                         change: () => {
                           // 拖完才算一步（拖的過程只改畫面）
                           const before = opBefore || snap();
                           opBefore = null;
                           E.hist.push(before);
                           E.fut = [];
                           changed();
                         }}}),
        opNum),
      h("div.edlbtns", {},
        h("button.mini", {type: "button", "data-i18n": "ed.addLayer", on: {click: () => addLayer(false)}}),
        h("button.mini", {type: "button", "data-i18n": "ed.dupLayer", "data-i18n-title": "ed.dupLayerHint", on: {click: () => addLayer(true)}}),
        h("button.mini", {type: "button", "data-i18n": "ed.delLayer", disabled: E.layers.length < 2, on: {click: deleteLayer}}),
        h("button.mini", {type: "button", "data-i18n-aria": "ed.up", "data-i18n-title": "ed.up", disabled: E.layers.indexOf(L0) === E.layers.length - 1,
                          on: {click: () => moveLayer(1)}}, "↑"),
        h("button.mini", {type: "button", "data-i18n-aria": "ed.down", "data-i18n-title": "ed.down", disabled: E.layers.indexOf(L0) === 0,
                          on: {click: () => moveLayer(-1)}}, "↓"),
        h("button.mini", {type: "button", "data-i18n": "ed.merge", "data-i18n-title": "ed.mergeHint", disabled: E.layers.indexOf(L0) === 0,
                          on: {click: mergeDown}}),
        h("button.mini.edtoggle", {type: "button", "aria-pressed": String(!!L0.lock), "data-i18n": "ed.lockAlpha", "data-i18n-title": "ed.lockAlphaHint",
                                   on: {click: () => setLayer(L0, {lock: !L0.lock})}}))),
    wmSection(),
    h("section.rfsec.edout", {},
      h("div.rflab", {}, h("b", {"data-i18n": "ed.output"}), outBtn),
      outHint,
      h("div.edsize", {},
        h("label.cell", {}, h("span.k", {"data-i18n": "ed.width"}), sizeIn("w")),
        h("button.edlock", {type: "button", "aria-pressed": String(E.lock), "data-i18n-aria": "ed.lock", "data-i18n-title": "ed.lock",
                            on: {click: () => { E.lock = !E.lock; drawPanel(); }}}, E.lock ? "🔗" : "⛓"),
        h("label.cell", {}, h("span.k", {"data-i18n": "ed.height"}), sizeIn("h")),
        h("button.mini", {type: "button", "data-i18n": "ed.sizeReset", disabled: out.w === E.w && out.h === E.h,
                          on: {click: () => { E.out = {w: E.w, h: E.h, w0: E.w, h0: E.h}; drawPanel(); }}})),
      h("div.rflab", {}, h("span.dim", {"data-i18n": "ed.format"}), h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ed.format"}, FORMATS.map(k => h("button", {
          type: "button", "aria-pressed": String(fmt() === k), text: k.toUpperCase(),
          on: {click: () => { saveUi({eformat: k}); drawPanel(); }}})))),
      fmt() !== "png" ? h("div.edlrow", {},
        h("span.dim", {"data-i18n": "ed.quality"}),
        h("input", {type: "range", min: "50", max: "100", step: "1", value: String(q), "data-i18n-aria": "ed.quality",
                    on: {input: e => { qNum.textContent = e.target.value; }, change: e => saveUi({equality: Number(e.target.value)})}}),
        qNum) : null,
      h("div.rflab", {}, h("span.dim", {"data-i18n": "ed.dest"}), h("span.sp"),
        h("div.seg", {role: "group", "data-i18n-aria": "ed.dest"}, ["library", "folder"].map(k => h("button", {
          type: "button", "aria-pressed": String((k === "library") === toLibrary()), "data-i18n": `ed.dest.${k}`,
          on: {click: () => { saveUi({edest: k}); drawPanel(); }}})))),
      toLibrary() ? null : h("div.newset", {},
        h("input.find", {type: "text", value: E.ui?.efolder || "", spellcheck: "false", "data-i18n-placeholder": "lib.dest", "data-i18n-aria": "lib.dest",
                         on: {change: e => saveUi({efolder: e.target.value.trim()})}}),
        h("button.pill", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
          const got = await pickFolder(E.ui?.efolder || "");
          if (got) { saveUi({efolder: got}); drawPanel(); }
        }}})),
      // 生成參數：存進圖庫一律保留；匯出到資料夾自己選（使用者筆記）
      h("div.cell.swcell.edkeep", {},
        h("span.swtext", {}, h("b", {id: "edKeepLbl", "data-i18n": "ed.keep"}),
          h("span.hint", {"data-i18n": toLibrary() ? "ed.keepLibrary" : "ed.keepHint"})),
        h("button.switch", {type: "button", role: "switch", "aria-labelledby": "edKeepLbl", disabled: toLibrary(),
                            "aria-checked": String(toLibrary() || E.ui?.ekeep !== false),
                            on: {click: () => { saveUi({ekeep: E.ui?.ekeep === false}); drawPanel(); }}}, h("i")))),
    kraSection(),
    h("div.rffoot", {},
      h("div.go", {}, h("button.gobtn", {type: "button", id: "edGo", on: {click: doExport}},
        h("span.l", {"data-i18n": toLibrary() ? "ed.saveLibrary" : "ed.export"}),
        h("kbd", {text: `${out.w} × ${out.h} · ${fmt().toUpperCase()}`}))),
      h("p.hint.edmsg", {id: "edMsg", role: "status", "data-kind": E.msg?.kind || null, text: E.msg?.text || ""}))].filter(Boolean));
  translate(box);
  syncSaved();
}

async function doExport() {
  const btn = $("edGo");
  if (!E.src || btn?.disabled) return;
  if (!toLibrary() && !(E.ui?.efolder || "").trim()) { E.msg = {kind: "err", text: t("ed.needFolder")}; drawPanel(); return; }
  btn.disabled = true;
  const out = E.out || {w: E.w, h: E.h};
  const d = await post("/api/edit/export", {
    src_id: E.src.id, png: await pngOf(flatten()), format: fmt(), quality: Number(E.ui?.equality) || 92,
    width: out.w, height: out.h, dest: toLibrary() ? "library" : "folder", folder: E.ui?.efolder || "",
    keep_meta: E.ui?.ekeep !== false}).catch(e => ({error: e.message}));
  E.msg = d.error ? {kind: "err", text: d.error} : {kind: "ok", text: t(toLibrary() ? "ed.savedLibrary" : "ed.exported", {name: d.path.split(/[\\/]/).pop()})};
  drawPanel();
}

// 結束編輯：捨棄工作檔（圖層都不留），回到空白頁。已經輸出的圖不受影響
async function closeEdit() {
  const ok = await confirmDialog({title: t("ed.closeTitle"), body: [h("p", {text: t("ed.closeBody")})], ok: t("ed.closeOk")});
  if (!ok) return;
  clearTimeout(E.saveTimer);
  if (E.pid) await post("/api/edit/discard", {id: E.pid}).catch(() => {});
  E.src = null; E.layers = []; E.pid = ""; E.hist = []; E.fut = []; E.pending = false;
  saveUi({esrc: 0});
  await drawEmpty();
}

// ── 快捷鍵 ──
// 只有在打字的地方（圖層名稱、數字欄）才讓給輸入框；調完滑桿、按完按鈕焦點還停在那裡，快捷鍵照樣要能用
const NOT_TYPING = ["range", "checkbox", "radio", "color", "button"];
const typing = el => !!el?.closest?.("textarea, select, [contenteditable]") || (el?.tagName === "INPUT" && !NOT_TYPING.includes(el.type));
document.addEventListener("keydown", e => {
  if (location.hash !== "#output" || !E.src || e.defaultPrevented || document.body.classList.contains("viewing")) return;
  if (e.target.id === "edText" && e.key === "Escape") { e.preventDefault(); endText(true); return; }
  if (typing(e.target)) return;
  const k = e.key.toLowerCase(), mod = e.ctrlKey || e.metaKey;
  if (mod && k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
  else if (mod && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); redo(); }
  else if (e.key === "Alt" && (PAINT.includes(E.tool) || E.tool === "select")) e.preventDefault();  // Alt＋點是吸色／減去，不要讓瀏覽器把焦點移到選單
  else if (mod && k === "a") { e.preventDefault(); selectAll(); }
  else if (mod && k === "d") { e.preventDefault(); if (E.sel) setSel(null); }
  else if (mod && k === "i") { e.preventDefault(); invertSel(); }
  else if (mod && k === "j") { e.preventDefault(); selToLayer(e.shiftKey); }
  else if (mod && (k === "c" || k === "x")) { e.preventDefault(); copySel(k === "x"); }
  else if (mod || e.altKey) return;                  // Ctrl+V 交給 paste 事件（才拿得到外面複製的圖）
  else if (E.xf && e.key === "Enter") { e.preventDefault(); applyXf(); }
  else if (E.adj && e.key === "Enter") { e.preventDefault(); applyAdj(); }
  else if (E.adj && e.key === "Escape") { e.preventDefault(); cancelAdj(); }
  else if (E.xf && e.key === "Escape") { e.preventDefault(); cancelXf(); }
  else if (E.poly && e.key === "Enter") { e.preventDefault(); closePoly(); }
  else if (E.poly && e.key === "Escape") { e.preventDefault(); E.poly = null; drawStage(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && E.sel) { e.preventDefault(); clearSel(); }
  else if (e.key === " ") { e.preventDefault(); if (!E.space) { E.space = true; $("edView")?.toggleAttribute("data-space", true); } }
  else if (k === "0") E.zoomApply?.({...FIT});
  else if (TOOL_KEY[k]) setTool(TOOL_KEY[k]);
  else if ((k === "[" || k === "]") && PAINT.includes(E.tool)) { e.preventDefault(); setSize(bset().size * (k === "]" ? 1.25 : 0.8)); }
  else if (E.tool === "crop" && e.key === "Enter") { e.preventDefault(); applyCrop(); }
  else if (E.tool === "crop" && e.key === "Escape") { e.preventDefault(); E.crop = null; setTool("view"); }
}, true);
document.addEventListener("keyup", e => {
  if (e.key === "Alt" && E.tool !== "picker") hideLoupe();
  if (e.key !== " " || !E.space) return;
  E.space = false;
  $("edView")?.removeAttribute("data-space");
}, true);
