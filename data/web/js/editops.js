// 06 輸出的計算（不碰 DOM，測試直接用 node 跑）：裁切框（拉出來、拖著移、拉邊角改大小，比例固定時照比例）、
// 輸出尺寸鎖比例、筆刷的圓點、選取範圍的外框與螞蟻線、變形框的把手與拖曳。座標都是圖的像素

// 裁切比例：[鍵, 寬／高]；free＝自由、orig＝原圖比例（呼叫時換成實際的數字）
export const RATIOS = [["free", 0], ["orig", -1], ["1:1", 1], ["4:5", 4 / 5], ["3:4", 3 / 4], ["2:3", 2 / 3],
                       ["9:16", 9 / 16], ["16:9", 16 / 9], ["3:2", 3 / 2]];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// 從 a 拉到 b 的框：比例固定時照寬度算高度（往拖的方向長），整個夾在圖裡面
export function cropFromDrag(a, b, ratio, W, H) {
  let dx = b[0] - a[0], dy = b[1] - a[1];
  if (ratio > 0) {
    const sx = dx < 0 ? -1 : 1, sy = dy < 0 ? -1 : 1;
    // 游標到哪都能拉：取寬、高兩邊換算後比較大的那個，再縮到放得進圖裡
    let w = Math.max(Math.abs(dx), Math.abs(dy) * ratio);
    const maxW = sx > 0 ? W - a[0] : a[0], maxH = sy > 0 ? H - a[1] : a[1];
    w = Math.min(w, maxW, maxH * ratio);
    dx = sx * w;
    dy = sy * w / ratio;
  }
  const x0 = clamp(Math.min(a[0], a[0] + dx), 0, W), y0 = clamp(Math.min(a[1], a[1] + dy), 0, H);
  const x1 = clamp(Math.max(a[0], a[0] + dx), 0, W), y1 = clamp(Math.max(a[1], a[1] + dy), 0, H);
  return {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
}

// 游標在框的哪裡：邊角（n s e w ne nw se sw）、裡面（move），都不是回 null。tol 是圖的像素
export function hitCrop(r, p, tol) {
  if (!r) return null;
  const [x, y] = p, L = Math.abs(x - r.x) <= tol, R = Math.abs(x - (r.x + r.w)) <= tol;
  const T = Math.abs(y - r.y) <= tol, B = Math.abs(y - (r.y + r.h)) <= tol;
  const inX = x >= r.x - tol && x <= r.x + r.w + tol, inY = y >= r.y - tol && y <= r.y + r.h + tol;
  if (!inX || !inY) return null;
  const v = T ? "n" : B ? "s" : "", hz = L ? "w" : R ? "e" : "";
  if (v || hz) return v + hz;
  return "move";
}

// 拖著移：整個框不能跑出圖
export function moveCrop(r, dx, dy, W, H) {
  return {...r, x: clamp(r.x + dx, 0, W - r.w), y: clamp(r.y + dy, 0, H - r.h)};
}

// 拉邊或角改大小：對面的邊（角）不動；比例固定時照比例（拉邊的話另一邊從中間長），最小 8 像素
export function resizeCrop(r, handle, p, ratio, W, H) {
  const MIN = 8;
  let x0 = r.x, y0 = r.y, x1 = r.x + r.w, y1 = r.y + r.h;
  const [px, py] = [clamp(p[0], 0, W), clamp(p[1], 0, H)];
  if (handle.includes("w")) x0 = Math.min(px, x1 - MIN);
  if (handle.includes("e")) x1 = Math.max(px, x0 + MIN);
  if (handle.includes("n")) y0 = Math.min(py, y1 - MIN);
  if (handle.includes("s")) y1 = Math.max(py, y0 + MIN);
  if (ratio > 0) {
    let w = x1 - x0, h = y1 - y0;
    const corner = handle.length === 2;
    if (corner) {
      // 角：取比較大的那個方向，對角不動
      if (w / h > ratio) h = w / ratio; else w = h * ratio;
      const ax = handle.includes("w") ? x1 : x0, ay = handle.includes("n") ? y1 : y0;
      const maxW = handle.includes("w") ? ax : W - ax, maxH = handle.includes("n") ? ay : H - ay;
      w = Math.min(w, maxW, maxH * ratio);
      h = w / ratio;
      x0 = handle.includes("w") ? ax - w : ax;
      y0 = handle.includes("n") ? ay - h : ay;
      x1 = x0 + w;
      y1 = y0 + h;
    } else if (handle === "e" || handle === "w") {
      const cy = (r.y + r.y + r.h) / 2;
      h = w / ratio;
      h = Math.min(h, 2 * Math.min(cy, H - cy));
      w = h * ratio;
      if (handle === "w") x0 = x1 - w; else x1 = x0 + w;
      y0 = cy - h / 2;
      y1 = cy + h / 2;
    } else {
      const cx = (r.x + r.x + r.w) / 2;
      w = h * ratio;
      w = Math.min(w, 2 * Math.min(cx, W - cx));
      h = w / ratio;
      if (handle === "n") y0 = y1 - h; else y1 = y0 + h;
      x0 = cx - w / 2;
      x1 = cx + w / 2;
    }
  }
  return {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
}

// 裁切框換成整數像素（套用時用）
export function roundCrop(r, W, H) {
  const x = clamp(Math.round(r.x), 0, W - 1), y = clamp(Math.round(r.y), 0, H - 1);
  return {x, y, w: clamp(Math.round(r.w), 1, W - x), h: clamp(Math.round(r.h), 1, H - y)};
}

// ── 筆刷（第 2 輪）──
// 從 a 畫到 b 要蓋幾個點（筆觸是一連串圓點）：間距 step，carry 是上一段剩下沒走完的距離，
// 回傳 {pts, carry}。快速移動時點也會等距補滿，慢慢移動時不會在同一個地方疊太多
export function dabs(a, b, step, carry = 0) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
  const s = Math.max(0.5, step), pts = [];
  let d = s - carry;
  while (d <= len) {
    pts.push([a[0] + dx * d / len, a[1] + dy * d / len]);
    d += s;
  }
  return {pts, carry: len - (d - s)};
}

// 銳利化（unsharp mask）：原圖＋(原圖－模糊)×amount，每個色版夾在 0～255；透明度不動。
// src、blur、out 都是 RGBA 的陣列（ImageData.data）
export function sharpen(src, blur, amount, out = new Uint8ClampedArray(src.length)) {
  for (let i = 0; i < src.length; i += 4) {
    out[i] = src[i] + (src[i] - blur[i]) * amount;
    out[i + 1] = src[i + 1] + (src[i + 1] - blur[i + 1]) * amount;
    out[i + 2] = src[i + 2] + (src[i + 2] - blur[i + 2]) * amount;
    out[i + 3] = src[i + 3];
  }
  return out;
}

// 在遮罩上蓋一個圓點（data 是 RGBA 陣列，只用透明度那格）。重疊的地方取比較大的值，不是一直疊上去：
// 同一筆來回塗不會越來越深，筆壓輕的那段也不會被多蓋幾次就變實心。
// hard 0～1：1＝硬邊，越小邊緣越軟（從 r×hard 開始淡出）；alpha 0～1 是這一點的濃度。
// 回傳改到的範圍 [x0, y0, x1, y1]（含），完全在圖外就是 null
export function stamp(data, w, h, cx, cy, r, hard, alpha = 1) {
  const x0 = Math.max(0, Math.floor(cx - r - 1)), x1 = Math.min(w - 1, Math.ceil(cx + r + 1));
  const y0 = Math.max(0, Math.floor(cy - r - 1)), y1 = Math.min(h - 1, Math.ceil(cy + r + 1));
  if (x0 > x1 || y0 > y1) return null;
  const soft = r * (1 - Math.min(1, Math.max(0, hard)));
  const top = Math.min(1, Math.max(0, alpha)) * 255;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
      let a = Math.min(1, r + 0.5 - d);                // 邊緣抗鋸齒
      if (soft > 0.5) a = Math.min(a, (r - d) / soft);
      if (a <= 0) continue;
      const v = Math.round(a * top), i = (y * w + x) * 4 + 3;
      if (v > data[i]) data[i] = v;
    }
  }
  return [x0, y0, x1, y1];
}

// 輸出尺寸：改了寬（或高），鎖比例時另一邊跟著算
export function lockSize(w0, h0, which, v) {
  const n = Math.max(1, Math.round(Number(v) || 0));
  return which === "w" ? {w: n, h: Math.max(1, Math.round(n * h0 / w0))} : {w: Math.max(1, Math.round(n * w0 / h0)), h: n};
}

// ── 選取範圍、變形（第 3 輪）──
// 遮罩（RGBA，看透明度）裡有東西的範圍 {x, y, w, h}；min 以上才算（選取範圍用 1，輸出內容的範圍也是 1）。全空回 null
export function alphaBox(data, w, h, min = 1) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    let row = false;
    for (let x = 0, i = y * w * 4 + 3; x < w; x++, i += 4) {
      if (data[i] < min) continue;
      row = true;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
    }
    if (row) { if (y < y0) y0 = y; y1 = y; }
  }
  return x1 < 0 ? null : {x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1};
}

// 螞蟻線要畫的像素：選到的（on）裡面，上下左右有一邊沒選到或是圖的邊。回傳像素的索引
export function edgePixels(on, w, h) {
  const out = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (!on[p]) continue;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !on[p - 1] || !on[p + 1] || !on[p - w] || !on[p + w]) out.push(p);
    }
  }
  return Int32Array.from(out);
}

// 變形的狀態 t = {cx, cy, w, h, rot（弧度）, fx, fy}：中心、目前的寬高（正數）、旋轉、翻轉
// 八個把手：方向 [dx, dy]（-1 左／上、1 右／下）
export const HANDLES = {nw: [-1, -1], n: [0, -1], ne: [1, -1], e: [1, 0], se: [1, 1], s: [0, 1], sw: [-1, 1], w: [-1, 0]};

const rot = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

// 圖上的點 → 框自己的座標（以中心為原點、轉回沒旋轉的方向）
export function xfLocal(t, p) {
  return rot([p[0] - t.cx, p[1] - t.cy], -t.rot);
}

// 框的四個角（圖上的座標）：左上、右上、右下、左下
export function xfCorners(t) {
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([dx, dy]) => {
    const [x, y] = rot([dx * t.w / 2, dy * t.h / 2], t.rot);
    return [t.cx + x, t.cy + y];
  });
}

// 游標在框的哪裡：把手（nw…w）、框裡面（move）、框外面（rotate）。tol 是圖的像素
export function xfHit(t, p, tol) {
  const [u, v] = xfLocal(t, p), hw = t.w / 2, hh = t.h / 2;
  for (const [k, [dx, dy]] of Object.entries(HANDLES)) {
    if (Math.abs(u - dx * hw) <= tol && Math.abs(v - dy * hh) <= tol) return k;
  }
  return Math.abs(u) <= hw && Math.abs(v) <= hh ? "move" : "rotate";
}

// 拖曳後的新狀態。mode：move、rotate、把手；從 start 拖到 p。
// 角的把手預設照比例（free＝true 時自由），對角固定不動；邊的把手只改一邊、對邊固定。
// rotate 時 snap＝true 每 15° 一格
export function xfDrag(mode, t0, start, p, {free = false, snap = false} = {}) {
  if (mode === "move") return {...t0, cx: t0.cx + p[0] - start[0], cy: t0.cy + p[1] - start[1]};
  if (mode === "rotate") {
    let a = t0.rot + Math.atan2(p[1] - t0.cy, p[0] - t0.cx) - Math.atan2(start[1] - t0.cy, start[0] - t0.cx);
    if (snap) a = Math.round(a / (Math.PI / 12)) * (Math.PI / 12);
    a = Math.atan2(Math.sin(a), Math.cos(a));                    // 收在 -180°～180°
    return {...t0, rot: a};
  }
  const [dx, dy] = HANDLES[mode];
  const anchor = [-dx * t0.w / 2, -dy * t0.h / 2];               // 對角（或對邊）在框座標裡的位置
  const [u, v] = xfLocal(t0, p);
  let w = dx ? Math.max(1, (u - anchor[0]) * dx) : t0.w;
  let h = dy ? Math.max(1, (v - anchor[1]) * dy) : t0.h;
  if (dx && dy && !free) {
    const s = Math.max(w / t0.w, h / t0.h);
    w = Math.max(1, t0.w * s);
    h = Math.max(1, t0.h * s);
  }
  const c = [dx ? anchor[0] + dx * w / 2 : 0, dy ? anchor[1] + dy * h / 2 : 0];
  const [ox, oy] = rot(c, t0.rot);
  return {...t0, w, h, cx: t0.cx + ox, cy: t0.cy + oy};
}

// ── 扭曲、彎曲（變形的另外兩種模式）──
// 變形一律看成「原圖上的 (u, v)（0～1）→ 圖上的哪一點」。一般模式是平移旋轉縮放翻轉，
// 扭曲是四個角各自拖（投影變換，也能做出透視、歪斜），彎曲是 n×n 個網格點（點跟點之間用平滑的曲線接起來）

// 一般模式：t 的對應
export function freeMap(t, u, v) {
  const x = (u - 0.5) * t.w * (t.fx ? -1 : 1), y = (v - 0.5) * t.h * (t.fy ? -1 : 1);
  const [ox, oy] = rot([x, y], t.rot);
  return [t.cx + ox, t.cy + oy];
}

// 扭曲：正方形 (0,0)(1,0)(1,1)(0,1) → 四邊形 q[0..3] 的投影變換（Heckbert 的公式）
export function quadMap(q) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  let g = 0, h = 0;
  const den = dx1 * dy2 - dx2 * dy1;
  if ((dx3 || dy3) && den) {
    g = (dx3 * dy2 - dx2 * dy3) / den;
    h = (dx1 * dy3 - dx3 * dy1) / den;
  }
  const a = x1 - x0 + g * x1, b = x3 - x0 + h * x3, d = y1 - y0 + g * y1, e = y3 - y0 + h * y3;
  return (u, v) => {
    const w = g * u + h * v + 1;
    return [(a * u + b * v + x0) / w, (d * u + e * v + y0) / w];
  };
}

// 彎曲：網格 {n, pts}（pts[j * n + i]，i 是橫的、j 是直的）。點跟點之間用 Catmull-Rom 曲線接，
// 會通過每個網格點；邊外面照最外兩格的方向延伸，所以沒拖過的網格就是原樣
export function gridMap(g) {
  const {n, pts} = g;
  const at = (i, j) => {
    if (i < 0) return lerp2(at(0, j), at(1, j), -1);
    if (i > n - 1) return lerp2(at(n - 1, j), at(n - 2, j), -1);
    if (j < 0) return lerp2(at(i, 0), at(i, 1), -1);
    if (j > n - 1) return lerp2(at(i, n - 1), at(i, n - 2), -1);
    return pts[j * n + i];
  };
  return (u, v) => {
    const su = Math.min(n - 1, Math.max(0, u * (n - 1))), sv = Math.min(n - 1, Math.max(0, v * (n - 1)));
    const i = Math.min(n - 2, Math.floor(su)), j = Math.min(n - 2, Math.floor(sv));
    const rows = [-1, 0, 1, 2].map(dj => cr([-1, 0, 1, 2].map(di => at(i + di, j + dj)), su - i));
    return cr(rows, sv - j);
  };
}
const lerp2 = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];   // k = -1：a 往 b 的反方向延伸一格
function cr([p0, p1, p2, p3], f) {
  const f2 = f * f, f3 = f2 * f;
  return [0, 1].map(k => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * f + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * f2
                                + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * f3));
}

// 目前模式的對應函式；x = {mode, t, q, g}
export function xfMap(x) {
  if (x.mode === "distort") return quadMap(x.q);
  if (x.mode === "warp") return gridMap(x.g);
  return (u, v) => freeMap(x.t, u, v);
}

// 換模式時接著現在的樣子：四個角、或 n×n 的網格點都從目前的對應取
export const quadOf = map => [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => map(u, v));
export function gridOf(map, n) {
  const pts = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pts.push(map(i / (n - 1), j / (n - 1)));
  return {n, pts};
}

// 最近的控制點（tol 以內），沒有回 -1
export function nearPoint(pts, p, tol) {
  let best = -1, bd = tol;
  pts.forEach((q, i) => {
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d <= bd) { bd = d; best = i; }
  });
  return best;
}

// 點在不在多邊形裡面（ray casting）
export function inPoly(poly, [x, y]) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// 一組點繞 c 轉 a（弧度）、平移 (dx, dy)
export const turnPts = (pts, c, a) => pts.map(p => { const [x, y] = rot([p[0] - c[0], p[1] - c[1]], a); return [c[0] + x, c[1] + y]; });
export const shiftPts = (pts, dx, dy) => pts.map(p => [p[0] + dx, p[1] + dy]);

// ── 調整（第 4 輪）──
// 亮度、對比、飽和度（-100～100）、色相（-180～180 度）合成一個顏色矩陣：out = m·(r, g, b) + d，一次算完。
// 亮度：加減最多 127；對比：以 128 為中心放大縮小（-100＝全灰、100＝3 倍）；
// 飽和度：跟灰階版本混合（-100＝灰階、100＝2 倍）；色相：保持亮度的色相旋轉（跟 CSS hue-rotate 同一組係數）
const LUM = [0.213, 0.715, 0.072];

export function adjustMatrix({bri = 0, con = 0, sat = 0, hue = 0} = {}) {
  const k = con >= 0 ? 1 + con / 50 : 1 + con / 100;
  const s = 1 + sat / 100;
  const S = [0, 1, 2].map(r => [0, 1, 2].map(c => LUM[c] * (1 - s) + (r === c ? s : 0)));
  const a = hue * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a);
  const H = [
    [0.213 + cs * 0.787 - sn * 0.213, 0.715 - cs * 0.715 - sn * 0.715, 0.072 - cs * 0.072 + sn * 0.928],
    [0.213 - cs * 0.213 + sn * 0.143, 0.715 + cs * 0.285 + sn * 0.140, 0.072 - cs * 0.072 - sn * 0.283],
    [0.213 - cs * 0.213 - sn * 0.787, 0.715 - cs * 0.715 + sn * 0.715, 0.072 + cs * 0.928 + sn * 0.072]];
  // 先亮度、再對比：k·(v + b - 128) + 128。飽和度、色相的矩陣每一列加起來都是 1，灰色的偏移量不受影響
  const m = [0, 1, 2].map(r => [0, 1, 2].map(c => k * (H[r][0] * S[0][c] + H[r][1] * S[1][c] + H[r][2] * S[2][c])));
  return {m, d: k * bri * 1.275 + 128 * (1 - k)};
}

// 套到像素上（RGBA 陣列，透明度不動）。mask（選取範圍的 RGBA）有的話只改選到的地方，半透明的邊照比例混合
export function adjustPixels(src, {m, d}, out = new Uint8ClampedArray(src.length), mask = null) {
  const [[a, b, c], [e, f, g], [i, j, k]] = m;
  for (let p = 0; p < src.length; p += 4) {
    const r = src[p], gr = src[p + 1], bl = src[p + 2];
    let nr = a * r + b * gr + c * bl + d, ng = e * r + f * gr + g * bl + d, nb = i * r + j * gr + k * bl + d;
    if (mask) {
      const w = mask[p + 3] / 255;
      nr = r + (nr - r) * w; ng = gr + (ng - gr) * w; nb = bl + (nb - bl) * w;
    }
    out[p] = nr; out[p + 1] = ng; out[p + 2] = nb; out[p + 3] = src[p + 3];
  }
  return out;
}

// ── 浮水印（第 5 輪）──
// 放在哪：九宮格（tl t tr l c r bl b br）或 tile（平鋪，整片斜著排、隔一排錯開半格）。
// w×h 是一個浮水印的大小、margin 是離邊的距離（九宮格才用）、angle 是平鋪的角度（度）。
// 回傳要蓋的位置 [{x, y, rot}]（中心點、旋轉的弧度）
export const WM_POS = ["tl", "t", "tr", "l", "c", "r", "bl", "b", "br"];

export function wmSpots(W, H, w, h, pos, margin = 0, angle = 0) {
  if (pos === "tile") {
    const a = angle * Math.PI / 180, sx = w * 1.5, sy = Math.max(h * 3, w * 0.6), D = Math.hypot(W, H) / 2 + Math.max(w, h);
    const out = [];
    for (let j = -Math.ceil(D / sy); j <= Math.ceil(D / sy); j++) {
      for (let i = -Math.ceil(D / sx) - 1; i <= Math.ceil(D / sx) + 1; i++) {
        const [px, py] = rot([i * sx + (j % 2 ? sx / 2 : 0), j * sy], a);
        const x = W / 2 + px, y = H / 2 + py;
        if (x > -w && x < W + w && y > -h && y < H + h) out.push({x, y, rot: a});
      }
    }
    return out;
  }
  const p = WM_POS.includes(pos) ? pos : "br";
  const x = p.endsWith("l") ? margin + w / 2 : p.endsWith("r") ? W - margin - w / 2 : W / 2;
  const y = p.startsWith("t") ? margin + h / 2 : p.startsWith("b") ? H - margin - h / 2 : H / 2;
  return [{x, y, rot: 0}];
}

// 批次輸出的尺寸：b = {size: "orig" / "long" / "pct", long, pct}。長邊 n 像素、或縮放 %；最大 8192
export function outSize(w, h, b) {
  const s = b.size === "long" ? Math.max(1, b.long) / Math.max(w, h) : b.size === "pct" ? Math.max(1, b.pct) / 100 : 1;
  const k = Math.min(s, 8192 / Math.max(w, h));
  return {w: Math.max(1, Math.round(w * k)), h: Math.max(1, Math.round(h * k))};
}
