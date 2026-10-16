// 安裝方式（server/install.py）：輕量安裝沒有生成引擎，生成、精修、訓練這幾頁換成說明＋「改為完整安裝」
// （頁面留著，讓人知道有這些功能；使用者決定的，2026-10-15）。完整安裝時什麼都不變。
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

export function needsFull(space, step) {
  return INSTALL.lite && (space === "train" || (space === "studio" && FULL_STEPS.includes(step)));
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

// 生成、精修、訓練在輕量時的那張卡
export function liteCard() {
  return h("section.card.litecard", {"aria-labelledby": "liteTitle"},
    h("h2.title", {id: "liteTitle", text: t("lite.title")}),
    h("p", {text: t("lite.body")}),
    action());
}

// ⚙ 設定 →「其他」：目前的安裝方式；輕量時加上改為完整的按鈕
export function installBox() {
  if (!INSTALL.lite) return h("p", {text: t("set.install.full")});
  return h("div.litebox", {}, h("p", {}, h("b", {text: t("set.install.lite")})), h("p.hint", {text: t("lite.body")}), action());
}
