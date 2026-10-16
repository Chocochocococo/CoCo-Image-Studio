// 放大檢視的計算（精修頁的圖）。不碰 DOM，測試直接用 node 跑。
//
// 圖在「符合視窗」（k = 1）時的位置是版面排出來的；放大只加一層
// transform: translate(x, y) scale(k)，原點在圖的左上角。
// frame：{L, T, w, h, W, H}——圖在 k = 1 時相對於檢視框的左上角 (L, T)、圖的寬高 (w, h)、檢視框的寬高 (W, H)。
// 滑鼠位置 (px, py) 一律是相對於圖在 k = 1 時的左上角（CSS 像素、未縮放）。

export const ZMIN = 1;
export const ZMAX = 8;
export const FIT = Object.freeze({k: 1, x: 0, y: 0});

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// 拖曳的範圍：圖比框大時不能露出空白（圖的邊不能跑進框裡），比框小時不能拖出框
function range(offset, size, k, box) {
  const a = -offset, b = box - offset - k * size;
  return [Math.min(a, b), Math.max(a, b)];
}

export function clampPan(z, f) {
  if (!(z.k > ZMIN)) return {...FIT};
  const [x0, x1] = range(f.L, f.w, z.k, f.W), [y0, y1] = range(f.T, f.h, z.k, f.H);
  return {k: z.k, x: clamp(z.x, x0, x1), y: clamp(z.y, y0, y1)};
}

// 以 (px, py) 為中心縮放：游標底下的那一點縮放前後停在畫面上同一個位置
export function zoomAt(z, factor, px, py, f) {
  const k = clamp(z.k * factor, ZMIN, ZMAX);
  if (k === ZMIN) return {...FIT};
  const ux = (px - z.x) / z.k, uy = (py - z.y) / z.k;       // 圖上那一點（未縮放的座標）
  return clampPan({k, x: px - ux * k, y: py - uy * k}, f);
}

export function panBy(z, dx, dy, f) {
  return clampPan({k: z.k, x: z.x + dx, y: z.y + dy}, f);
}

// 放大比例換成「原圖像素」的百分比：100% 是原圖一個像素對一個 CSS 像素
export function pixelPercent(z, shownWidth, imageWidth) {
  return Math.round(z.k * shownWidth / imageWidth * 100);
}
