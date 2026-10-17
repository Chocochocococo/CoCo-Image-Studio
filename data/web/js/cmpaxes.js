// 比較模式的 XY 對照表：一組圖之間只差一兩個參數時，把那一兩個參數當成欄與列，
// 差異直接變成表頭，每張圖底下就不用再寫（Forge 的 X/Y plot 那種排法）。
// 這個模組不碰 DOM，測試直接用 node 跑。
//
// dims：每個「有差異的項目」一筆 {key, vals}，vals 是每張圖在這一項的值（字串，沒有就是 ""）。
// 由 compare.js 從圖片參數整理出來（取樣器、CFG、LoRA、提示詞多出來的標籤…）。

export const MAX_SIDE = 10;       // 一軸最多幾個值；再多就不像表了，改用格狀
export const MIN_FILL = 0.5;      // 表格至少要填滿一半；太空的話格狀比較好讀

const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: "base"});

// 軸上的值照數字大小排（CFG 5 < 7 < 12、LoRA 0.6 < 0.8），不是照字串
export function sortValues(vals) {
  return [...new Set(vals)].sort((a, b) => (a === "" ? -1 : b === "" ? 1 : collator.compare(a, b)));
}

// 回傳 null（排不成表）或
//   {x: {key, values}, y: {key, values} | null, noise: [key…],
//    cells: [[圖片索引…] 依 y 再依 x 攤平，長度 = x.values × (y ? y.values : 1)]}
// swap 為真時欄列對調。
// dims 的 blankUnknown 為真時，空值表示「沒記錄」（舊圖沒有排程欄位之類），
// 不算一種差異：只有空值跟一種值的話，這一項當作沒在變。
// 提示詞多出來的標籤不設這個：那裡的空值是「沒有多出來的標籤」，是真的差異。
const distinct = d => new Set(d.blankUnknown ? d.vals.filter(v => v !== "") : d.vals).size;

export function xyLayout(dims, n, {swap = false} = {}) {
  const varying = dims.filter(d => distinct(d) > 1);
  if (n < 3 || !varying.length) return null;          // 兩張用並排就好，排成表一半是空的
  // 每張都不一樣的項目（例如隨機 seed）當雜訊：不拿來當軸，只寫在格子裡。
  // 但它是唯一的差異時（例如只比 seed）就用它當軸。
  const noisy = varying.filter(d => n > 2 && new Set(d.vals).size === n);
  let axes = varying.length > noisy.length ? varying.filter(d => !noisy.includes(d)) : varying;
  if (axes.length > 2) return null;
  axes = axes.map(d => ({key: d.key, vals: d.vals, values: sortValues(d.vals)}));
  if (axes.some(a => a.values.length > MAX_SIDE)) return null;

  // 欄放值比較多的那一軸：圖多半是直的，畫面是橫的
  if (axes.length === 2 && axes[1].values.length > axes[0].values.length) axes.reverse();
  if (swap && axes.length === 2) axes.reverse();
  const [x, y] = axes;
  const cols = x.values.length, rows = y ? y.values.length : 1;
  const cells = Array.from({length: cols * rows}, () => []);
  for (let i = 0; i < n; i++) {
    const xi = x.values.indexOf(x.vals[i]), yi = y ? y.values.indexOf(y.vals[i]) : 0;
    cells[yi * cols + xi].push(i);
  }
  if (y && cells.filter(c => c.length).length < cols * rows * MIN_FILL) return null;
  const strip = a => (a ? {key: a.key, values: a.values} : null);
  return {x: strip(x), y: strip(y), noise: varying.filter(d => !axes.some(a => a.key === d.key)).map(d => d.key), cells};
}

// XY 對照表匯出成一張圖的版面（compare.js 的匯出用；這裡只算位置，不碰 DOM）。
// 欄寬照這一欄裡最多幾張並排（同一格有好幾張，例如只差 seed 的，橫著排）。
// o：{cellW, cellH（一張圖的框）, capH（圖底下的字）, gap, pad, titleH（最上面組名與共同參數）,
//     headH（欄的表頭高）, headW（列的表頭寬，沒有 Y 軸時不佔位）}
// 回傳 {W, H, cols: [{x, w}], rows: [y], headY, items: [{n（ok 裡第幾張）, x, y}]}
export function xyExportLayout(lay, o) {
  const cols = lay.x.values.length, rows = lay.y ? lay.y.values.length : 1;
  const inner = Math.round(o.gap / 2);                 // 同一格裡圖跟圖的間隔
  const per = Array.from({length: cols}, (_, xi) =>
    Math.max(1, ...Array.from({length: rows}, (_, yi) => lay.cells[yi * cols + xi].length)));
  const left = o.pad + (lay.y ? o.headW + o.gap : 0);
  const out = [];
  let x = left;
  for (let xi = 0; xi < cols; xi++) {
    const w = per[xi] * o.cellW + (per[xi] - 1) * inner;
    out.push({x, w});
    x += w + o.gap;
  }
  const headY = o.pad + o.titleH;
  const top = headY + o.headH + o.gap;
  const rowY = Array.from({length: rows}, (_, yi) => top + yi * (o.cellH + o.capH + o.gap));
  const items = [];
  lay.cells.forEach((cell, k) => {
    const xi = k % cols, yi = Math.floor(k / cols);
    cell.forEach((n, j) => items.push({n, x: out[xi].x + j * (o.cellW + inner), y: rowY[yi]}));
  });
  return {W: x - o.gap + o.pad, H: rowY[rows - 1] + o.cellH + o.capH + o.pad, cols: out, rows: rowY, headY, items};
}
