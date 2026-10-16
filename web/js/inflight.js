// 非同步請求的先來後到與存檔排隊（Codex 第三輪審查找到的幾個都是這一類）：
// 快速連點時先送的請求比較晚回來、蓋掉後來選的；存檔還在送就換頁，後面的改動沒存到；
// 幾個東西共用一個延遲計時器，改了 B 把 A 的存檔取消掉。不碰畫面，tests/test_inflight.py 用 node 直接測

// 只收最後一次的回覆：const ticket = newest(); …
//   const still = ticket(); const r = await get(…); if (!still()) return;   // 已經又點了別的：這個回覆不要了
export function newest() {
  let seq = 0;
  return () => {
    const mine = ++seq;
    return () => mine === seq;
  };
}

// 排成一列做：上一個做完（成功或失敗）才開始下一個；回傳這一個的結果。
// 例：存標籤與重新讀一次不能交錯（讀回來的蓋掉還沒存的，或存到一半被讀回來的舊資料蓋掉）
export function serial() {
  let tail = Promise.resolve();
  return fn => {
    const run = tail.then(fn, fn);
    tail = run.catch(() => {});
    return run;
  };
}

// 重新讀回來的清單（fresh）蓋上本機還沒存的欄位：local 裡 dirty 是 {欄位: true} 的那幾筆照 key 對回去，
// 只換改過的欄位、dirty 留著（等一下照常存）；其他欄位照新讀的（例如背景剛寫好的描述）
export function keepUnsaved(fresh, local, key = "path") {
  const mine = new Map(local.filter(x => x.dirty).map(x => [x[key], x]));
  return fresh.map(x => {
    const m = mine.get(x[key]);
    if (!m) return x;
    const out = {...x, dirty: {...m.dirty}};
    for (const f of Object.keys(m.dirty)) out[f] = m[f];
    return out;
  });
}

// 送失敗（連不上）：送出去的欄位（sent）放回待存（pending），但期間又改過的以新的為準——
// 已經又在 pending 裡（又改了），或現在的值（now）已經不一樣（別的視窗改了）的就不放回。
// alias：這個欄位要拿哪個欄位的現在值來比（生成台的 TEMPLATE 跟著 PROMPT 存）
export function putBack(pending, sent, now, alias = {}) {
  for (const [k, v] of Object.entries(sent)) {
    if (k in pending) continue;
    if (JSON.stringify(now[alias[k] || k]) !== JSON.stringify(v)) continue;
    pending[k] = v;
  }
  return pending;
}

// 每個 key 各自延遲：同一個 key 再排會取消它自己的前一次，不同 key 互不影響。
// 要送的東西（例如資源的分類與名稱）在排的當下就放進 fn 裡，之後切到別的地方才送也不會送錯
export function keyedDelay(ms) {
  const timers = new Map();
  return {
    put(key, fn) {
      clearTimeout(timers.get(key)?.id);
      const id = setTimeout(() => { timers.delete(key); fn(); }, ms);
      timers.set(key, {id, fn});
    },
    // 排著的全部馬上做（例如關掉視窗之前）
    flush() {
      const all = [...timers.values()];
      timers.clear();
      for (const t of all) { clearTimeout(t.id); t.fn(); }
    },
    size: () => timers.size,
  };
}

// 自動存檔的排隊：saveOnce() 送一次、回傳成不成功；dirty() 是還有沒有沒存的改動。
//   now()：計時器到了。正在送就不重疊（回傳正在送的那次）
//   flush()：換圖、輸出之前。等正在送的送完，還有沒存的就再送，直到最新的都存好；存失敗回 false（呼叫的人不要丟掉現在的編輯）
export function saveQueue(saveOnce, dirty) {
  let run = null;
  const once = () => (run ??= Promise.resolve().then(saveOnce).finally(() => { run = null; }));
  return {
    now: once,
    running: () => !!run,
    async flush() {
      for (let i = 0; i < 20; i++) {
        if (run) { await run.catch(() => false); continue; }
        if (!dirty()) return true;
        if (await once().catch(() => false) === false) return false;
      }
      return !dirty();
    },
  };
}
