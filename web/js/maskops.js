// 遮罩的計算（不碰 DOM，測試直接用 node 跑）：反轉、擴張／收縮、魔術棒、方形與橢圓的範圍、多邊形收起來的判斷。
//
// 遮罩是 canvas 的 RGBA 像素陣列（ImageData.data）：alpha ≥ 128 算塗到，
// 跟 core/inpaint.load_mask 同一條線。顏色只是顯示用，一律塗成 tint（[r, g, b]）

export const ON = 128;

// 反轉：塗到的變沒塗到，反過來也是。半透明的邊（筆刷的反鋸齒）一起反，邊緣還是平滑的
export function invert(data, tint) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = tint[0]; data[i + 1] = tint[1]; data[i + 2] = tint[2];
    data[i + 3] = 255 - data[i + 3];
  }
  return data;
}

// 擴張（r > 0）／收縮（r < 0）r 像素，圓角（用距離算，不是方塊）。
// 擴張：離塗到的地方不超過 r 的都塗上；收縮：離沒塗到的地方不超過 r 的都擦掉（＝把沒塗到的擴張）。
// 圖的外面不算沒塗到：貼齊邊緣的遮罩收縮時，貼邊那一側不會內縮
export function grow(data, w, h, r, tint) {
  if (!r) return data;
  const shrink = r < 0, lim = r * r, n = w * h;
  const on = new Uint8Array(n);
  for (let p = 0; p < n; p++) on[p] = (data[p * 4 + 3] >= ON) !== shrink ? 1 : 0;
  const d2 = sqDist(on, w, h);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const near = d2[p] <= lim;
    data[i] = tint[0]; data[i + 1] = tint[1]; data[i + 2] = tint[2];
    data[i + 3] = near !== shrink ? 255 : 0;
  }
  return data;
}

// 每個像素到最近的 on 像素的距離平方（Felzenszwalb & Huttenlocher 的距離轉換：先直的一遍、再橫的一遍，
// 跟半徑無關都是線性時間，2MP 的圖幾十毫秒）
export function sqDist(on, w, h) {
  const INF = 1e20, n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  const g = new Float64Array(w * h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = on[y * w + x] ? 0 : INF;
    dt1(f, h, d, v, z);
    for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    dt1(f, w, d, v, z);
    for (let x = 0; x < w; x++) g[y * w + x] = d[x];
  }
  return g;
}

// 一維的距離轉換：d[q] = min over p of (q - p)² + f[p]（拋物線的下包絡）
function dt1(f, n, d, v, z) {
  let k = 0;
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  const cross = (q, p) => ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p);
  for (let q = 1; q < n; q++) {
    let s = cross(q, v[k]);
    while (s <= z[k]) { k--; s = cross(q, v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) ** 2 + f[v[k]];
  }
}

// 魔術棒：從 (x, y) 選取顏色相近的區域，回傳 0/1 的 Uint8Array（每個像素一格）。
// px 是底圖的 RGBA；相近＝每個色版跟起點的差都不超過 tol（0～255）。
// contiguous：只選跟起點相連的（上下左右）；false 是整張圖裡相近的都選
export function floodSelect(px, w, h, x, y, tol, contiguous = true) {
  x = Math.floor(x); y = Math.floor(y);
  const n = w * h, out = new Uint8Array(n);
  if (x < 0 || y < 0 || x >= w || y >= h) return out;
  const i0 = (y * w + x) * 4, r0 = px[i0], g0 = px[i0 + 1], b0 = px[i0 + 2];
  const near = p => {
    const i = p * 4;
    return Math.abs(px[i] - r0) <= tol && Math.abs(px[i + 1] - g0) <= tol && Math.abs(px[i + 2] - b0) <= tol;
  };
  if (!contiguous) {
    for (let p = 0; p < n; p++) if (near(p)) out[p] = 1;
    return out;
  }
  // 放進堆疊時就標記，每個像素最多進一次，堆疊用固定大小的陣列
  const stack = new Int32Array(n);
  let top = 0;
  const push = p => { out[p] = 1; stack[top++] = p; };
  push(y * w + x);
  while (top) {
    const p = stack[--top], px0 = p % w;
    if (px0 > 0 && !out[p - 1] && near(p - 1)) push(p - 1);
    if (px0 < w - 1 && !out[p + 1] && near(p + 1)) push(p + 1);
    if (p >= w && !out[p - w] && near(p - w)) push(p - w);
    if (p < n - w && !out[p + w] && near(p + w)) push(p + w);
  }
  return out;
}

// 選取的範圍塗進遮罩（erase：擦掉）
export function applySelection(data, sel, erase, tint) {
  for (let p = 0, i = 0; p < sel.length; p++, i += 4) {
    if (!sel[p]) continue;
    data[i] = tint[0]; data[i + 1] = tint[1]; data[i + 2] = tint[2];
    data[i + 3] = erase ? 0 : 255;
  }
  return data;
}

// 方形、橢圓：從按下的點 a 拖到 b 的外框；square（按著 Shift）固定成正方形、正圓，往拖的方向長
export function shapeBox(a, b, square = false) {
  let dx = b[0] - a[0], dy = b[1] - a[1];
  if (square) {
    const s = Math.max(Math.abs(dx), Math.abs(dy));
    dx = (dx < 0 ? -1 : 1) * s;
    dy = (dy < 0 ? -1 : 1) * s;
  }
  return {x: Math.min(a[0], a[0] + dx), y: Math.min(a[1], a[1] + dy), w: Math.abs(dx), h: Math.abs(dy)};
}

// 多邊形：點回第一個點附近（tol 是底圖的像素）就收起來；至少要三個點才圍得出範圍
export function nearFirst(pts, p, tol) {
  return pts.length >= 3 && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) <= tol;
}

// 在 ctx 上描出形狀的路徑（塗進遮罩與畫預覽共用）。圍不出範圍的回 false
// shape：{tool: "rect" / "ellipse"，a, b, square} 或 {tool: "lasso" / "poly"，pts}
export function tracePath(ctx, shape) {
  if (shape.tool === "rect" || shape.tool === "ellipse") {
    const r = shapeBox(shape.a, shape.b, shape.square);
    if (r.w < 1 || r.h < 1) return false;
    ctx.beginPath();
    if (shape.tool === "rect") ctx.rect(r.x, r.y, r.w, r.h);
    else ctx.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2);
    return true;
  }
  const pts = shape.pts || [];
  if (pts.length < 3) return false;
  ctx.beginPath();
  ctx.moveTo(...pts[0]);
  for (const p of pts.slice(1)) ctx.lineTo(...p);
  ctx.closePath();
  return true;
}
