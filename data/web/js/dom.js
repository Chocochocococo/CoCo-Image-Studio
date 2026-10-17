// 組畫面用的小工具。
//
// 一律用 DOM API（textContent），不把字串拼進 innerHTML——
// 模型名稱、提示詞都來自資料庫或使用者，拼進 HTML 就是 XSS（見 REVIEW_LOG R37）。

// h("div.card", {id:"x", "data-i18n":"k", on:{click:fn}}, child, "文字")
export function h(sel, attrs, ...kids) {
  const [tag, ...cls] = sel.split(".");
  const el = document.createElement(tag || "div");
  if (cls.length) el.className = cls.join(" ");
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "text") el.textContent = v;
    else if (k === "value") el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k);
  return el;
}

export const $ = id => document.getElementById(id);

// 介面大小（整頁 CSS zoom）。要照游標或元素位置擺放東西時要換算：
// getBoundingClientRect、clientX、innerWidth 是畫面上的像素；offsetWidth 與寫進
// style.left / top 的值是放大前的像素（畫出來會再乘上倍率）。
export const zoom = () => parseFloat(document.documentElement.style.zoom) || 1;

// 小圖示用 SVG 畫，不靠字型裡的符號——⚄ 這類字元很多字型沒有，會變成方框。
// shapes：[["rect", {x,y,…}], ["circle", {…}]]，顏色一律跟著文字（currentColor）
export function icon(shapes, size = 14) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  for (const [k, v] of Object.entries({viewBox: "0 0 16 16", width: size, height: size, "aria-hidden": "true"})) {
    svg.setAttribute(k, v);
  }
  for (const [tag, attrs] of shapes) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    svg.append(el);
  }
  return svg;
}

// 繞回來的箭頭：「用上一張的」
export const REUSE = [
  ["path", {d: "M13.2 8.6A5.3 5.3 0 1 1 11.6 4", fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round"}],
  ["path", {d: "M12.3 1.4v3.2H9.1", fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round"}],
];

export const DIE = [
  ["rect", {x: 1.5, y: 1.5, width: 13, height: 13, rx: 3.5, fill: "none", stroke: "currentColor", "stroke-width": 1.5}],
  ["circle", {cx: 5, cy: 5, r: 1.3, fill: "currentColor"}],
  ["circle", {cx: 8, cy: 8, r: 1.3, fill: "currentColor"}],
  ["circle", {cx: 11, cy: 11, r: 1.3, fill: "currentColor"}],
];

// 路徑取檔名、去掉副檔名："C:\\a\\b.safetensors" → "b"
export function stem(path) {
  return String(path || "").split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin|gguf)$/i, "");
}

// 比較組的代號：0 → A、25 → Z、26 → AA、701 → ZZ、702 → AAA…（跟試算表的欄名一樣，沒有上限）
export function seqLabel(i) {
  let s = "";
  for (let n = Math.floor(i) + 1; n > 0; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s;
  return s;
}

// 一張 w×h 的 canvas 要縮成幾倍才畫得出來。瀏覽器對單張 canvas 的邊長與面積有上限
// （Chrome 邊長 32767、面積約 2.7 億像素，其他瀏覽器更小），超過時 toBlob 會拿到 null。
// 這裡取保守值，所有瀏覽器都安全；回傳 1 表示不必縮。
export const CANVAS_MAX_SIDE = 16384, CANVAS_MAX_AREA = 120e6;
export function canvasScale(w, h) {
  return Math.min(1, CANVAS_MAX_SIDE / w, CANVAS_MAX_SIDE / h, Math.sqrt(CANVAS_MAX_AREA / (w * h)));
}
