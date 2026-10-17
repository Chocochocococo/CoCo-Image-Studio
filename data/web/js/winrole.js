// 這個視窗是不是 ⧉ 開的新視窗（sync.js 的 openWindow，名字是 coco-…）。是的話回傳開它的主視窗；
// 不是、主視窗已經關了、或讀不到主視窗（不是 CoCo 開的）時回傳 null。
// 生成頁本身開成新視窗（coco-generate）不算：送到生成區照舊在自己這裡切。
// 獨立出來是為了能用 node 測（sync.js 一載入就會開 BroadcastChannel）。
export function popoutOpener(w = globalThis) {
  try {
    const o = w.opener;
    const name = String(w.name || "");
    if (!o || o.closed || !name.startsWith("coco-") || name === "coco-generate") return null;
    void o.location.hash;          // 不同來源的視窗讀不到，會丟錯
    return o;
  } catch {
    return null;
  }
}
