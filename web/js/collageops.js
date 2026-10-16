// 拼接的排版計算（collage.js 用；不碰 DOM，node 測得到）。
// 輸入每張圖的原尺寸，算出畫布大小與每張圖畫在哪（x, y, w, h）、取原圖的哪一塊（sx, sy, sw, sh）。
// 單位是「原尺寸的像素」：輸出時整張再乘一個倍數（太大自動縮到 MAX_SIDE 以內）。

export const MAX_SIDE = 8192;          // 跟 /api/edit/export 的上限一樣
export const MAX_ITEMS = 40;           // 一次最多拼幾張（每張都要整張讀進瀏覽器）
export const MODES = ["grid", "col", "row", "free"];
export const CELLS = ["first", "1:1", "2:3", "3:4", "4:3", "3:2", "16:9", "9:16"];

const ratioOf = (cell, first) => {
  if (cell === "first" || !cell) return first.w / first.h;
  const [a, b] = String(cell).split(":").map(Number);
  return a > 0 && b > 0 ? a / b : first.w / first.h;
};

// 整張原圖放進去：沒有裁切
const whole = (s, x, y, w, h) => ({x, y, w, h, sx: 0, sy: 0, sw: s.w, sh: s.h});

// cover：裁成格子的比例（取中間）、填滿格子；contain：整張放進格子、置中，旁邊留背景
function fitInto(s, x, y, w, h, fit) {
  const r = w / h, ir = s.w / s.h;
  if (fit === "contain") {
    const k = Math.min(w / s.w, h / s.h);
    const dw = s.w * k, dh = s.h * k;
    return whole(s, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
  }
  if (ir > r) {                                   // 原圖比較寬：左右各裁一點
    const sw = s.h * r;
    return {x, y, w, h, sx: (s.w - sw) / 2, sy: 0, sw, sh: s.h};
  }
  const sh = s.w / r;
  return {x, y, w, h, sx: 0, sy: (s.h - sh) / 2, sw: s.w, sh};
}

// 格狀：欄數（0＝自動，接近正方形）、格子比例（照第一張或固定比例）、填法；最後一列不滿時置中
export function gridLayout(sizes, {cols = 0, cell = "first", fit = "cover", gap = 0, pad = 0} = {}) {
  const n = sizes.length;
  if (!n) return {W: 1, H: 1, boxes: []};
  const c = Math.max(1, Math.min(n, Math.round(cols) || Math.ceil(Math.sqrt(n))));
  const rows = Math.ceil(n / c);
  const first = sizes[0], r = ratioOf(cell, first);
  // 格子大小：跟第一張差不多的面積（照第一張的比例時就是第一張的尺寸）
  const area = first.w * first.h;
  const cw = Math.max(1, Math.round(Math.sqrt(area * r))), ch = Math.max(1, Math.round(cw / r));
  const W = pad * 2 + c * cw + (c - 1) * gap, H = pad * 2 + rows * ch + (rows - 1) * gap;
  const boxes = sizes.map((s, i) => {
    const row = Math.floor(i / c), inRow = row === rows - 1 ? n - row * c : c;
    const off = (c - inRow) * (cw + gap) / 2;
    return fitInto(s, pad + off + (i % c) * (cw + gap), pad + row * (ch + gap), cw, ch, fit);
  });
  return {W, H, boxes};
}

// 直排：每張縮成一樣寬（最窄那張的寬度，不放大），由上往下接；橫排：一樣高，由左往右接
export function stripLayout(sizes, {mode = "col", gap = 0, pad = 0} = {}) {
  const n = sizes.length;
  if (!n) return {W: 1, H: 1, boxes: []};
  const boxes = [];
  if (mode === "row") {
    const hh = Math.min(...sizes.map(s => s.h));
    let x = pad;
    for (const s of sizes) {
      const w = Math.max(1, Math.round(s.w * hh / s.h));
      boxes.push(whole(s, x, pad, w, hh));
      x += w + gap;
    }
    return {W: x - gap + pad, H: hh + pad * 2, boxes};
  }
  const ww = Math.min(...sizes.map(s => s.w));
  let y = pad;
  for (const s of sizes) {
    const h = Math.max(1, Math.round(s.h * ww / s.w));
    boxes.push(whole(s, pad, y, ww, h));
    y += h + gap;
  }
  return {W: ww + pad * 2, H: y - gap + pad, boxes};
}

export function layout(sizes, o = {}) {
  if (o.mode === "col" || o.mode === "row") return stripLayout(sizes, o);
  return gridLayout(sizes, o);
}

// 自由排：從格狀（格子照第一張、整張放進）開始，之後每張自己拖；畫布大小也可以自己改
export function freeStart(sizes, o = {}) {
  const g = gridLayout(sizes, {...o, cell: "first", fit: "contain"});
  return {W: Math.round(g.W), H: Math.round(g.H), boxes: g.boxes.map(b => ({x: b.x, y: b.y, w: b.w, h: b.h}))};
}

// 自由排的畫布貼合內容：所有圖的外框再加留白，圖跟著平移
export function fitFree(boxes, pad = 0) {
  if (!boxes.length) return {W: 1, H: 1, boxes};
  const x0 = Math.min(...boxes.map(b => b.x)), y0 = Math.min(...boxes.map(b => b.y));
  const x1 = Math.max(...boxes.map(b => b.x + b.w)), y1 = Math.max(...boxes.map(b => b.y + b.h));
  return {W: Math.max(1, Math.round(x1 - x0 + pad * 2)), H: Math.max(1, Math.round(y1 - y0 + pad * 2)),
          boxes: boxes.map(b => ({...b, x: b.x - x0 + pad, y: b.y - y0 + pad}))};
}

// 自由排：點到哪一張（後面的疊在上面，先找上面的）；靠近右下角是「縮放」
export function hitBox(boxes, [x, y], grip) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (Math.abs(x - (b.x + b.w)) <= grip && Math.abs(y - (b.y + b.h)) <= grip) return {i, part: "size"};
    if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) return {i, part: "move"};
  }
  return null;
}

// 拖右下角縮放：左上角不動、比例不變，最小 16 像素
export function resizeBox(b, [x, y]) {
  const r = b.w / b.h;
  const w = Math.max(16, Math.max(x - b.x, (y - b.y) * r));
  return {...b, w, h: w / r};
}

// 輸出的倍數：size = "orig"（原尺寸）或 "long"（長邊 long 像素）；一律不超過 MAX_SIDE
export function outScale(W, H, {size = "orig", long = 4096} = {}) {
  const side = Math.max(W, H);
  const want = size === "long" ? Math.max(16, Number(long) || 4096) / side : 1;
  return Math.min(want, MAX_SIDE / side);
}

// 換順序：把第 from 張移到第 to 張的位置
export function moveItem(list, from, to) {
  const out = list.slice();
  if (from < 0 || from >= out.length) return out;
  const [it] = out.splice(from, 1);
  out.splice(Math.max(0, Math.min(out.length, to)), 0, it);
  return out;
}
