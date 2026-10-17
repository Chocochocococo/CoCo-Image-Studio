// 小視窗（pop.js）要不要因為這一下點擊而關掉。獨立出來是為了能用 node 測（pop.js 一載入就要用到 document）。
//
// 點在小視窗或它的按鈕上：不關。
// 點在蓋在上面的另一個對話框裡（例如「搬移／複製…」裡按「瀏覽…」開的選資料夾）：不關。以前會關，
// 選好的資料夾填進已經不在畫面上的欄位，按了像沒反應、圖也沒搬（2026-10-10 使用者回報）。
// 小視窗本身就開在某個對話框裡（觸發按鈕在那個對話框裡）時，點那個對話框的其他地方照樣關。
export const MODAL = '[aria-modal="true"]';

export function closesPop(target, pop, anchor) {
  if (pop.contains(target) || anchor.contains(target)) return false;
  const dlg = target.closest?.(MODAL);
  return !(dlg && !dlg.contains(anchor));
}
