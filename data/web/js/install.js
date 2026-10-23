// 安裝方式（server/install.py）：輕量安裝沒有生成引擎。完整安裝時什麼都不變。
// - 生成頁照常能用（提示詞、標籤、資源庫、Recipe）：在 CoCo 編好提示詞再拿去別的地方生成，只有生成鈕要完整安裝，
//   按了在旁邊說明（使用者要求，2026-10-16）
// - 精修、訓練：畫面照常畫出來當預覽，上面蓋一層半透明遮罩＋說明卡＋「改為完整安裝」（同上；原本整頁只有說明卡）
// 文字用 t() 直接填：換語言時整頁重畫（main.js 的 render），設定視窗也是重畫

import {get, post} from "./api.js";
import {h} from "./dom.js";
import {t} from "./i18n.js";

export const INSTALL = {lite: false, pending: false, gpu: true};
const FULL_STEPS = ["generate", "refine"];          // 工作室裡要完整安裝的步驟；訓練空間整個都要

export async function loadInstall() {
  try { Object.assign(INSTALL, await get("/api/install")); } catch { /* 讀不到就照完整安裝畫 */ }
  return INSTALL;
}

// 這一頁的主要功能要完整安裝
export function needsFull(space, step) {
  return INSTALL.lite && (space === "train" || (space === "studio" && FULL_STEPS.includes(step)));
}

// 這一頁只能看（蓋遮罩）：要完整安裝的頁裡，生成頁除外
export function previewOnly(space, step) {
  return needsFull(space, step) && !(space === "studio" && step === "generate");
}

// 說明下面那一塊：按鈕、已設定（下次開啟補裝）、沒有 NVIDIA 顯示卡。按了之後就地換掉
function action() {
  if (INSTALL.pending) return h("p.hint", {text: t("lite.pending")});
  if (!INSTALL.gpu) return h("p.hint.warn", {text: t("lite.noGpu")});
  const box = h("div.liteact");
  const btn = h("button.pill.primary", {type: "button", text: t("lite.toFull"), on: {click: async () => {
    btn.disabled = true;
    const r = await post("/api/install/full", {}).catch(() => null);
    if (r && !r.error) Object.assign(INSTALL, r);
    box.replaceChildren(r?.error ? h("p.hint.warn", {text: r.error}) : !r ? btn : action());
    btn.disabled = false;
  }}});
  box.append(btn);
  return box;
}

// 精修、訓練在輕量時蓋在畫面上的那張卡
export function liteCard() {
  return h("section.card.litecard", {"aria-labelledby": "liteTitle"},
    h("h2.title", {id: "liteTitle", text: t("lite.title")}),
    h("p", {text: t("lite.body")}),
    action());
}

// 精修、訓練在輕量時：畫面照常畫（不能操作），上面蓋遮罩＋說明卡
export function litePreview(screen) {
  screen.inert = true;
  return h("div.litepv", {}, screen, h("div.liteveil", {}, liteCard()));
}

// 生成鈕在輕量時旁邊那個小視窗的內容（generate.js 開；生成台的其他東西照常能用）。
// 不在這裡 import pop.js：它一載入就要用到 document，node 的測試只載入這個檔
export function litePopBody() {
  return [h("h3.title", {text: t("lite.genTitle")}), h("p.litep", {text: t("lite.genBody")}), action()];
}

// ⚙ 設定 →「其他」：目前的安裝方式；輕量時加上改為完整的按鈕
export function installBox() {
  if (!INSTALL.lite) return h("p", {text: t("set.install.full")});
  return h("div.litebox", {}, h("p", {}, h("b", {text: t("set.install.lite")})), h("p.hint", {text: t("lite.body")}), action());
}

// ⚙ 設定 →「其他」：桌面與開始功能表上開這份 CoCo 的捷徑（server/shortcut.py）。不再用 CoCo 時先移除，再刪資料夾。
// 查捷徑要開一次 PowerShell（約 0.7 秒），所以先畫空的、查到再換
export function shortcutBox() {
  const box = h("div.scbox", {});
  const show = r => box.replaceChildren(
    !r ? h("p.hint.warn", {text: t("set.scFail")}) :
    r.error ? h("p.hint.warn", {text: r.error}) :
    r.removed ? h("p.hint", {text: t("set.scDone")}) :
    !r.found ? h("p.hint", {text: t("set.scNone")}) :
    h("button.mini", {type: "button", text: t("set.scRemove"), on: {click: async e => {
      e.currentTarget.disabled = true;
      show(await post("/api/shortcut/remove", {}).catch(() => null));
    }}}));
  get("/api/shortcut").then(show, () => show(null));
  return box;
}
