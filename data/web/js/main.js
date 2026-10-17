// 新介面的進入點：頂端列（空間切換、步驟）、外觀與語言設定，以及兩個空間的畫面：
//   工作室（#generate，預設）：生成控制台＋圖庫牆
//   圖庫（#library）：整面圖牆、搜尋與篩選，見 library.js
//
// 多視窗（sync.js，使用者筆記 #10）：#assets-種類 是資源庫的獨立視窗（整頁只有資源庫）；
// 頂端列的 ⧉ 選單把資源庫或某一頁開成新視窗（例如生成在一個螢幕、精修在另一個）。
// ⧉ 預設偵測到第二個螢幕才出現，外觀設定裡可以手動開關（ui.multiwin）。
//
// 一律用 DOM API 組畫面（textContent），不把字串拼進 innerHTML——
// 模型名稱、提示詞都來自資料庫，拼進 HTML 就是 XSS（見 REVIEW_LOG R37）。

import {get, post} from "./api.js";
import {assetsPage, initAssets, openAssets, refreshAssets, resetAssetsFrame} from "./assets.js";
import {openSetup} from "./setup.js";
import {compareScreen, initCompare, openCup, showCompare} from "./compare.js";
import {initRecipes, markModified, recipeButton, recipePills} from "./recipes.js";
import {recombineScreen, resetWinners, showRecombine} from "./recombine.js";
import {$, h, stem} from "./dom.js";
import {PASSIVE, initGenerate, railCard, syncSettings} from "./generate.js";
import {LANGS, initI18n, t, translate} from "./i18n.js";
import {ARCH_LABEL} from "./options.js";
import {afterRender, initLibrary, libraryNewImages, libraryScreen, refreshLibrary, refreshTray, searchBox, showLibrary, trayFor} from "./library.js";
import {Filters, facets, loadFacets} from "./filters.js";
import {info} from "./hint.js";
import {closePop, openPop} from "./pop.js";
import {initRefine, refineNewImages, refineProgress, refineScreen, refineSettingsChanged, showRefine} from "./refine.js";
import {editorScreen, initEditor, showEditor} from "./editor.js";
import {initBatch} from "./batch.js";
import {initCollage} from "./collage.js";
import {exploreScreen, initExplore, showExplore} from "./explore.js";
import {datasetScreen, initDataset, showDataset} from "./dataset.js";
import {datasetsScreen, showDatasets} from "./datasets.js";
import {initTrain, showTrain, trainScreen} from "./train.js";
import {initRefineBatch} from "./refinebatch.js";
import {INSTALL, litePreview, loadInstall, previewOnly} from "./install.js";
import {MULTI_MODES, assetsWindowOpen, multiEnabled, on, openWindow, send} from "./sync.js";
import {openSyntax} from "./syntax.js";
import {DEFAULT_THEME, THEMES, applyTheme, swatches} from "./theme.js";
import {initViewer, openViewer} from "./viewer.js";
import {Wall} from "./wall.js";

const ui = {lang: "", theme: DEFAULT_THEME, mode: "dark", scale: 1, multiwin: "auto"};
// 資源庫的獨立視窗：網址 #assets 或 #assets-lora 這種
const POP_ASSETS = /^#assets(-\w+)?$/.test(location.hash);
const SCALES = [0.9, 1, 1.15, 1.3, 1.5];

// 介面大小：整頁等比例放大（跟瀏覽器的 Ctrl＋ 一樣），格子和文字一起變大才不會擠爆
function applyScale() {
  const s = SCALES.includes(Number(ui.scale)) ? Number(ui.scale) : 1;
  document.documentElement.style.zoom = s === 1 ? "" : String(s);
}
// 網址決定畫面：#library → 圖庫空間；#datasets、#dataset、#train → 訓練空間的資料集、打標、訓練；#compare → 工作室的比較；
// 其他 → 工作室的生成
const route = () => (location.hash === "#library" ? ["library", ""]
  : location.hash === "#datasets" ? ["train", "datasets"]
  : location.hash === "#dataset" ? ["train", "dataset"]
  : location.hash === "#train" ? ["train", "train"]
  : ["studio", ({"#explore": "explore", "#compare": "compare", "#recombine": "recombine", "#refine": "refine", "#output": "output"})[location.hash] || "generate"]);
let [space, step] = route();

// 工作室右側的牆：排法與大小存在 ui 偏好的 gorder / genw（沿用舊介面生成頁圖庫的鍵名）
export const STUDIO_SIZES = [["s", 130], ["m", 190], ["l", 260], ["x", 340]];
let studioWall = null;
let libWall = null;
// 啟動完成（兩道牆、各頁都建好）才畫。之前來的重畫——螢幕的 change（接上第二個螢幕；內建瀏覽器開頁時也會觸發）、
// 別的視窗改外觀、換網址——都不用畫：start() 最後會照當下的網址與設定畫一次。
// 以前在 start() 還在等的時候就畫，studioWall 還是 null，主控台噴 studioWallCard 的錯（2026-10-13 實測時看到）
let booted = false;

// ── 頂端列 ─────────────────────────────────────────
// 第一層是空間，第二層是那個空間自己的步驟（圖庫沒有步驟，中間放搜尋框）
// 工作室的連結指到最後開的那一步（ui.laststep），從圖庫回來不會都跳回生成頁；訓練也是（ui.trainstep）
const SPACES = [["studio", "#generate"], ["library", "#library"], ["train", "#datasets"]];
const STUDIO_STEPS = ["generate", "explore", "compare", "recombine", "refine", "output"];
const STEPS = [["generate", "01"], ["explore", "02"], ["compare", "03"], ["recombine", "04"], ["refine", "05"], ["output", "06"]];
const READY = ["generate", "explore", "compare", "recombine", "refine", "output"];     // 做好的步驟；其他先灰掉
// 訓練空間的步驟：資料集（清單，像 OneTrainer 的 concept）→ 打標（打標台）→ 訓練
const TRAIN_STEPS = [["datasets", "01"], ["dataset", "02"], ["train", "03"]];
const TRAIN_READY = ["datasets", "dataset", "train"];

function bar() {
  const spaces = h("nav.spaces", {"data-i18n-aria": "space.aria"},
    SPACES.map(([key, href]) => h("a", {
      href: key === "studio" && STUDIO_STEPS.includes(ui.laststep) && !previewOnly(key, ui.laststep) ? `#${ui.laststep}`
        : key === "studio" && INSTALL.lite ? "#explore"          // 輕量安裝：工作室從探索開始（精修只能看）
        : key === "train" && TRAIN_READY.includes(ui.trainstep) ? `#${ui.trainstep}` : href,
      "aria-current": key === space ? "page" : null, "aria-disabled": href ? null : "true",
      "data-i18n-title": href ? null : "space.soon",
    }, h("span", {"data-i18n": `space.${key}`}))));
  const ready = space === "train" ? TRAIN_READY : READY;
  const center = space === "library" ? searchBox() : h("nav.flow", {"data-i18n-aria": "nav.aria"},
    (space === "train" ? TRAIN_STEPS : STEPS).map(([key, n]) => h("a", {
      href: ready.includes(key) ? `#${key}` : null,
      "aria-current": key === step ? "page" : null,
      "aria-disabled": ready.includes(key) ? null : "true",
      "data-i18n-title": ready.includes(key) ? null : "space.soon",
    }, h("span.n", {text: n}), h("span", {"data-i18n": `nav.${key}`}))));
  return h("header.bar", {},
    h("div.logo", {}, h("b", {text: "CoCo"}), h("i"), h("span", {text: "STUDIO"})),
    spaces,
    h("span.sp"), center, h("span.sp"),
    h("div.stat", {id: "stModel"}, h("span.dot"), h("b", {"data-i18n": "status.noModel"})),
    h("div.stat", {id: "stEngine"}, h("span.dot"), h("span", {"data-i18n": "status.engineOff"})),
    multiEnabled(ui.multiwin)
      ? h("button.round", {id: "btnPop", type: "button", "data-i18n-aria": "nav.popout", "data-i18n-title": "nav.popout",
                           "aria-haspopup": "dialog", "aria-expanded": "false",
                           on: {click: e => openPop("popWin", e.currentTarget, popMenu)}}, "⧉") : null,
    h("button.round", {id: "btnSetup", type: "button", "data-i18n-aria": "set.title", "data-i18n-title": "set.title",
                       "aria-haspopup": "dialog", on: {click: () => openSetup(setupSaved)}}, "⚙"),
    h("button.round", {id: "btnSettings", type: "button", "data-i18n-aria": "settings.open",
                       "data-i18n-title": "settings.open", "aria-haspopup": "dialog",
                       on: {click: toggleSettings}}, "◐"));
}

// ⧉ 選單：資源庫或某一頁開成新視窗。同一個名字再開一次是同一個視窗
const POP_PAGES = [["assets", "/#assets-model", null], ["generate", "/#generate", "01"], ["compare", "/#compare", "03"],
                   ["recombine", "/#recombine", "04"], ["refine", "/#refine", "05"], ["library", "/#library", null]];
function popMenu() {
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "nav.popout"})),
    h("div.popmenu", {}, POP_PAGES.map(([key, url, n]) => h("button", {type: "button", on: {click: () => {
      closePop();
      openWindow(url, `coco-${key}`);
    }}}, h("span.n", {text: n || "·"}), h("span", {"data-i18n": key === "assets" ? "assets.title" : key === "library" ? "space.library" : `nav.${key}`}))))];
}

async function refreshStatus() {
  try {
    const [s, g] = await Promise.all([get("/api/settings"), get("/api/gen/state")]);
    syncSettings(s);                                  // 別的分頁換了模型，這邊也要跟上
    const model = stem(s.MODEL);
    const m = $("stModel");
    m.querySelector(".dot").classList.toggle("on", !!model);
    const mb = m.querySelector("b");
    const arch = ARCH_LABEL[s.MODEL_ARCH] || s.MODEL_ARCH;
    if (model) { mb.removeAttribute("data-i18n"); mb.textContent = arch ? `${model} · ${arch}` : model; }
    const e = $("stEngine");
    // 輕量安裝沒有生成引擎：子行程只拿來數 token，說「已啟動」會讓人以為可以生成
    e.querySelector(".dot").classList.toggle("on", !!g.alive && !INSTALL.lite);
    e.querySelector("span:last-child").dataset.i18n = INSTALL.lite ? "set.install.lite" : g.alive ? "status.engineOn" : "status.engineOff";
    translate(e);
    // 頂端列排不下時這兩格會用「…」收尾：滑過時看完整的
    m.title = mb.textContent;
    e.title = e.textContent.trim();
  } catch { /* 狀態列失敗不影響其他部分 */ }
}

// ── 外觀與語言 ─────────────────────────────────────
function settingsPanel() {
  const themes = h("div.themes", {},
    Object.keys(THEMES).map(key => h("button.th", {
      type: "button", "aria-pressed": String(ui.theme === key), "data-theme-key": key, "data-i18n-title": `theme.${key}.note`,
      on: {click: () => change({theme: key})},
    },
      h("span.nm", {}, h("b", {"data-i18n": `theme.${key}.name`}),
        key === DEFAULT_THEME ? h("small", {"data-i18n": "theme.default"}) : null),
      h("span.dots", {}, swatches(key, ui.mode).map(c => { const i = h("i"); i.style.background = c; return i; })))));
  const [multiBtn, multiHint] = info("settings.multiHint");
  const mode = h("div.seg", {}, ["dark", "light"].map(m => h("button", {
    type: "button", "aria-pressed": String(ui.mode === m), "data-i18n": `settings.mode.${m}`,
    on: {click: () => change({mode: m})}})));
  const langs = h("div.seg", {},
    h("button", {type: "button", "aria-pressed": String(!ui.lang), "data-i18n": "settings.lang.auto",
                 on: {click: () => change({lang: ""})}}),
    LANGS.map(([k, name]) => h("button", {type: "button", "aria-pressed": String(ui.lang === k), text: name,
                                         lang: k, on: {click: () => change({lang: k})}})));
  return h("div.pop", {id: "settings", role: "dialog", "aria-modal": "false", "aria-labelledby": "setTitle", hidden: true},
    h("div.hd", {}, h("h2.title", {id: "setTitle", "data-i18n": "settings.title"}),
      h("span.toast", {id: "setToast"}),
      h("button.round", {type: "button", "data-i18n-aria": "settings.close", on: {click: toggleSettings}}, "✕")),
    h("div.row", {}, h("span.lbl", {"data-i18n": "settings.theme"}), themes),
    h("div.row", {}, h("span.lbl", {"data-i18n": "settings.mode"}), mode),
    h("div.row", {}, h("span.lbl", {"data-i18n": "settings.lang"}), langs),
    h("div.row", {}, h("span.lbl", {"data-i18n": "settings.scale"}),
      h("div.seg", {}, SCALES.map(v => h("button", {
        type: "button", "aria-pressed": String(Number(ui.scale) === v), text: `${Math.round(v * 100)}%`,
        on: {click: () => change({scale: v})}})))),
    h("div.row", {}, h("span.lbl", {}, t("settings.multi"), multiBtn),
      h("div.seg", {}, MULTI_MODES.map(m => h("button", {
        type: "button", "aria-pressed": String(ui.multiwin === m), "data-i18n": `settings.multi.${m}`,
        on: {click: () => change({multiwin: m})}}))),
      multiHint));
}

function toggleSettings() {
  const p = $("settings");
  p.hidden = !p.hidden;
  if (!p.hidden) p.querySelector("[aria-pressed='true']")?.focus();
}

// 外觀偏好：自己改的要存、廣播；別的視窗改的只套用（remote）
async function change(patch, remote = false) {
  const langChanged = "lang" in patch && patch.lang !== ui.lang;
  Object.assign(ui, patch);
  Object.assign(ui, applyTheme(ui.theme, ui.mode));
  applyScale();
  if (langChanged) await initI18n(ui.lang);
  if ("multiwin" in patch) resetAssetsFrame();
  if (POP_ASSETS) { if (langChanged) translate(document.body); return; }
  if (!booted) return;                    // 還在啟動（設定視窗還沒畫出來）：start() 最後會照新的 ui 畫
  const open = !$("settings").hidden;
  render();
  if (open) toggleSettings();
  if (remote) return;
  send("ui", {patch});
  try {
    await post("/api/ui", {lang: ui.lang, theme: ui.theme, mode: ui.mode, scale: ui.scale, multiwin: ui.multiwin});
    const toast = $("setToast");
    toast.textContent = t("settings.saved");
    setTimeout(() => { toast.textContent = ""; }, 1500);
  } catch { /* 存不進去也不影響這次的顯示 */ }
}

// ── 工作室的圖庫牆 ─────────────────────────────────
// 牆可以多選（Ctrl／Shift＋點、角落的勾），選了底下出現跟圖庫空間一樣的動作列
// 篩選（使用者要求 2026-10-10）：搜尋提示詞、模型、LoRA、資料夾、評分。開著篩選時，這次開 CoCo 之後生成的圖
// 照樣列在最前面（keep_new，使用者選的：生成時一定看得到結果）。
// 收成「排法」旁邊的一顆「篩選」鈕，點了開小視窗（使用者要求：跟排法一樣的下拉選單），一種篩選一列
const GF = new Filters({id: "gen", keepNew: true, menu: true, build: studioMenu});
const GEN_KINDS = [["model", "lib.model"], ["lora", "lib.lora"], ["folder", "lib.folder"], ["rating", "lib.rating"]];
function studioMenu(F) {
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "lib.filters"}), F.clearButton()),
    h("div.search", {role: "search"}, F.searchInput("genSearch")),
    h("div.fgrid", {}, GEN_KINDS.flatMap(([kind, label]) => [h("span.lbl", {"data-i18n": label}), F.dropChip(kind, label, true)]))];
}

// 篩選清單（模型、LoRA、資料夾）跟圖庫空間共用；還沒有或圖庫變過就先抓好，打開篩選時就有
function studioFilters() {
  if (!facets()) loadFacets().then(() => GF.draw()).catch(() => { /* 搜尋照樣能用 */ });
}

function studioWallCard() {
  const w = studioWall;
  return h("div.libwall", {},
    h("section.card.wall", {"aria-labelledby": "wallTitle"},
      h("div.whd", {}, h("h2.title", {id: "wallTitle", "data-i18n": "wall.title"}),
        h("span.dim.mono", {id: "wallCount", text: w.total ? w.countText() : ""}),
        h("span.dim", {id: "libNote", role: "status"}),
        h("span.sp"),
        GF.menuButton(),
        w.viewButton()),
      w.box, w.end, w.top),
    h("div.tray", {id: "tray", hidden: true, role: "region", "data-i18n-aria": "lib.trayAria"}));
}

// 設定存好了：掃描資料夾改了就重新載入兩道圖牆
// uiPatch：設定視窗裡改的介面偏好（生成後送去比較）；別的視窗也跟著換
function setupSaved(keys, d, uiPatch) {
  if (uiPatch) { Object.assign(ui, uiPatch); send("ui", {patch: uiPatch}); }
  if (keys.includes("SCAN_DIRS")) { studioWall.reset(); refreshLibrary(); }
  // 生成台用不到、別處讀的開關（直接刪除、看大圖的 LoRA 選項）：存好馬上更新，不用等輪詢
  if (keys.some(k => PASSIVE.includes(k)) && d?.settings) syncSettings(d.settings);
  if (keys.includes("HARD_DELETE")) refreshTray();
  // 參考圖來源改了：探索的索引跟著更新（只算新加的）
  if (keys.includes("REF_SOURCES")) post("/api/explore/index", {}).catch(() => { /* 進探索頁時也會更新 */ });
}

function newImages() {
  studioWall.refreshNewest();
  libraryNewImages();
  refineNewImages();
}

// 圖庫變了（library.js 的 galleryChanged）：重新掃描、搬走、還原 → 工作室的牆整個重讀；
// 新的圖（訓練試畫）→ 兩道牆都補上最新的
addEventListener("coco:gallery", e => {
  if (!studioWall) return;
  if (e.detail?.kind === "new") { studioWall.refreshNewest(); libraryNewImages(); }
  else studioWall.reset();
});

// 步驟列放不下（介面放大到 130%、150%，或視窗窄）：其他步驟只顯示編號、滑過看名稱，目前那一步照樣顯示名稱
// （使用者回報：工作室的「06 輸出」被擠出畫面，只能在上下邊緣用滾輪移）
function fitFlow() {
  const nav = document.querySelector(".bar .flow");
  if (!nav) return;
  nav.removeAttribute("data-compact");
  const tight = nav.scrollWidth > nav.clientWidth + 1;
  nav.toggleAttribute("data-compact", tight);
  for (const a of nav.querySelectorAll("a")) {
    if (a.getAttribute("aria-disabled") === "true") continue;
    if (tight) a.title = a.querySelector("span:not(.n)")?.textContent || "";
    else a.removeAttribute("title");
  }
}
let fitRaf = 0;
addEventListener("resize", () => { cancelAnimationFrame(fitRaf); fitRaf = requestAnimationFrame(fitFlow); });

// ── 組起來 ─────────────────────────────────────────
// 換頁前記下兩道牆捲到哪（畫面整個重建，負責捲動的卡片會換新的）
function render() {
  if (!booted) return;
  studioWall?.saveScroll();
  libWall?.saveScroll();
  // 工作室最後開的是哪一步：頂端的「工作室」連結下次就指到這裡（存起來，重開也記得）
  // 輕量安裝時只能看的頁（精修、訓練）不記，下次不會一進來就是遮罩
  const lite = previewOnly(space, step);
  if (space === "studio" && STUDIO_STEPS.includes(step) && ui.laststep !== step && !lite) {
    ui.laststep = step;
    post("/api/ui", {laststep: step}).catch(() => { /* 只影響下次 */ });
  }
  // 訓練空間也一樣（使用者回報：切到工作室或圖庫再回來，都跳回資料集）
  if (space === "train" && TRAIN_READY.includes(step) && ui.trainstep !== step && !lite) {
    ui.trainstep = step;
    post("/api/ui", {trainstep: step}).catch(() => { /* 只影響下次 */ });
  }
  closePop();
  const app = $("app");
  // 輕量安裝：精修與訓練照常畫出來，蓋上遮罩＋說明卡（litePreview）；生成頁照常（只有生成鈕要完整安裝，generate.js）
  const screen = space === "library" ? libraryScreen()
    : space === "train" ? (step === "train" ? trainScreen() : step === "datasets" ? datasetsScreen() : datasetScreen())
    : step === "explore" ? exploreScreen()
    : step === "compare" ? compareScreen()
    : step === "recombine" ? recombineScreen()
    : step === "refine" ? refineScreen()
    : step === "output" ? editorScreen()
    : h("main.main", {}, railCard(), studioWallCard());
  const main = lite ? litePreview(screen) : screen;
  app.replaceChildren(h("div.shell", {"data-space": space},
    bar(),
    main,
    settingsPanel()));
  translate(app);
  fitFlow();
  if (space === "library") { afterRender(); showLibrary(); libWall?.restoreScroll(); }
  else if (space === "train") { if (step === "train") showTrain(); else if (step === "datasets") showDatasets(); else showDataset(); }
  else if (step === "explore") showExplore();
  else if (step === "compare") showCompare();
  else if (step === "recombine") { resetWinners(); showRecombine(); }
  else if (step === "refine") showRefine();
  else if (step === "output") showEditor();
  else { studioFilters(); studioWall.refit(); trayFor(studioWall)(studioWall.selected()); studioWall.restoreScroll(); }
  document.title = `${t(space === "library" ? "space.library" : `nav.${step}`)} · ${t("app.title")}`;
  refreshStatus();
}

on("ui", ({patch}) => { if (patch) change(patch, true); });
// 接上／拔掉第二個螢幕：自動模式的 ⧉ 跟著出現或消失
screen.addEventListener?.("change", () => { if (ui.multiwin === "auto" && !POP_ASSETS) render(); });

// 點生成台的資源格：資源庫已經開在另一個視窗的話，叫它換到那一頁就好
function showAssets(kind) {
  if (assetsWindowOpen()) send("assets:show", {kind});
  else openAssets(kind);
}

// 別的視窗改了設定：資源庫、精修頁的「使用中的模型」跟著更新
function remoteChanged(why) {
  refreshAssets(why);
  refineSettingsChanged();
  refreshTray();                       // 別的視窗開關了「直接刪除圖片」
}

addEventListener("hashchange", () => {
  if (POP_ASSETS) return;
  if (location.hash === "#settings") { toggleSettings(); return; }
  const [ns, nstep] = route();
  if (ns !== space || nstep !== step) { [space, step] = [ns, nstep]; render(); }
});

document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    const p = $("settings");
    if (p && !p.hidden) { p.hidden = true; $("btnSettings")?.focus(); }
  }
});
document.addEventListener("click", e => {
  const p = $("settings");
  if (p && !p.hidden && !p.contains(e.target) && !e.target.closest("#btnSettings")) p.hidden = true;
});

(async function start() {
  try { Object.assign(ui, await get("/api/ui")); } catch { /* 用預設值 */ }
  Object.assign(ui, applyTheme(ui.theme, ui.mode));
  applyScale();
  await initI18n(ui.lang);
  await loadInstall();
  // 輕量安裝沒指定頁面時從圖庫開始（預設的生成頁要完整安裝）
  if (INSTALL.lite && !location.hash && !POP_ASSETS) { history.replaceState(null, "", "#library"); [space, step] = route(); }
  initAssets(ui);
  if (POP_ASSETS) {
    // 資源庫的獨立視窗：只需要生成台的設定（genSettings）與存檔，不畫控制台
    await initGenerate({ui, onRemote: remoteChanged});
    await initRecipes(ui);                 // Recipe 分頁要能套用
    await assetsPage(location.hash.split("-")[1] || "model");
    return;
  }
  studioWall = new Wall({name: "studio", ui, orderKey: "gorder", sizeKey: "genw", zoomKey: "genzoom", sizes: STUDIO_SIZES,
                         selectable: "mod", onSelect: sel => trayFor(studioWall)(sel),
                         onCount: () => { const c = $("wallCount"); if (c) c.textContent = studioWall.countText(); },
                         onOpen: it => openViewer(studioWall, it.id), query: () => GF.query(),
                         emptyKey: () => (GF.count() ? "lib.empty" : "wall.empty")});
  GF.wall = studioWall;
  libWall = initLibrary(ui);
  initViewer(ui);
  initRefine(ui);
  initEditor(ui);
  initBatch(ui);
  initCollage(ui);
  initExplore(ui);
  initDataset(ui);
  initTrain(ui);
  initRefineBatch(ui, newImages);
  initCompare(ui);
  // Recipe 的封面預設用最新生成的那張
  await initRecipes(ui, {coverId: () => studioWall.items[0]?.id ?? null, openAssets: showAssets});
  await initGenerate({newImages, progress: (st, s) => { studioWall.setDev(st, s); refineProgress(st); }, openAssets: showAssets,
                      openCup,
                      onRemote: remoteChanged,
                      recipeButton, recipePills, onChange: markModified, openSyntax, ui,
                      openFile: name => {
                        const it = studioWall.items.find(i => i.filename === name);
                        if (it) openViewer(studioWall, it.id);
                      }});
  booted = true;
  render();
  studioWall.loadMore();
  setInterval(refreshStatus, 5000);
  if (location.hash === "#settings") toggleSettings();   // 可以直接用連結打開設定
})();
