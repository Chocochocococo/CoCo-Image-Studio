// 淘汰賽（比較頁的一種排法，使用者筆記）：一組圖一輪一輪挑晉級的，直到覺得夠了。不碰 DOM，測試直接用 node 跑。
//
// 比較組 s：round＝目前第幾輪（從 1 開始）、done＝已經選好了。
// 每張 it：reach＝打到第幾輪（沒寫是 1）、adv＝這一輪標了晉級。
// 「就選這些」把結果寫進原本的 winner（勝出），重組頁、匯出比較圖照樣看得懂。

export const roundOf = s => Math.max(1, Number(s.round) || 1);
const reachOf = it => Math.max(1, Number(it.reach) || 1);

// 這一輪還在比的（檔案不見了的不算）
export function alive(s) {
  const r = roundOf(s);
  return s.items.filter(it => !it.missing && reachOf(it) >= r);
}

// 晉級路線：[{r: 輪次, n: 那一輪有幾張}]
export function route(s) {
  const out = [];
  for (let r = 1; r <= roundOf(s); r++) out.push({r, n: s.items.filter(it => !it.missing && reachOf(it) >= r).length});
  return out;
}

// 下一輪：標了晉級的進下一輪。要有標、又不是全部都標才能進（全部都標等於沒挑）
export function canAdvance(s) {
  const live = alive(s), k = live.filter(it => it.adv).length;
  return !s.done && k > 0 && k < live.length;
}

export function advance(s) {
  if (!canAdvance(s)) return false;
  const r = roundOf(s);
  for (const it of alive(s)) {
    if (it.adv) it.reach = r + 1;
    it.adv = false;
  }
  s.round = r + 1;
  return true;
}

// 回到第 r 輪重選：那一輪晉級過的先標好（上次的選擇還在，改幾張就好）
export function back(s, r) {
  r = Math.max(1, Math.min(roundOf(s), r));
  for (const it of s.items) {
    const was = reachOf(it);
    it.adv = was > r;
    if (was > r) it.reach = r;
  }
  s.round = r;
  s.done = false;
}

// 就選這些：這一輪標了晉級的是勝出；一張都沒標的話，這一輪還在的全部算勝出
export function finish(s) {
  const live = alive(s);
  const picked = live.some(it => it.adv) ? live.filter(it => it.adv) : live;
  const win = new Set(picked);
  for (const it of s.items) {
    it.winner = win.has(it);
    it.adv = false;
  }
  s.done = true;
  return picked.length;
}

// 重新開始：全部回到第一輪（勝出標記留著，下次「就選這些」才換掉）
export function restart(s) {
  for (const it of s.items) { it.reach = 1; it.adv = false; }
  s.round = 1;
  s.done = false;
}
