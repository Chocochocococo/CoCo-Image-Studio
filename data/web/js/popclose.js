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

// ◐ 外觀與語言（main.js 的 #settings）要不要因為這一下點擊而關掉。
// 點配色、深淺、介面大小時 change() 會同步重畫整個畫面、再把視窗打開；這次點擊冒泡到 document 時，
// e.target 是已經被換掉的舊按鈕，新的視窗不包含它。那不是點到外面，不關
// （2026-10-21 錄教學影片時發現：每點一個配色視窗就關掉，沒辦法連續試）。
export function closesSettings(target, panel) {
  if (!target.isConnected) return false;
  return !panel.contains(target) && !target.closest("#btnSettings");
}
