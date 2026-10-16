// 小視窗（尺寸、進階、紀錄、圖庫排法…）。同一時間只開一個。
//
// 貼在觸發按鈕旁邊；下面放不下就翻到上面。點外面、按 Esc、視窗縮放都會關。

import {translate} from "./i18n.js";
import {h, zoom} from "./dom.js";
import {closesPop} from "./popclose.js";

let cur = null;   // {el, anchor}

export function openPop(id, anchor, build) {
  const again = cur?.el.id === id;
  closePop();
  if (again) return;                       // 再按一次同一顆＝關掉
  const el = h("div.pp", {id, role: "dialog", "aria-modal": "false"}, build());
  document.body.append(el);
  translate(el);
  place(el, anchor);
  anchor.setAttribute("aria-expanded", "true");
  cur = {el, anchor};
  el.querySelector("[aria-pressed='true'], button, input, select")?.focus();
}

function place(el, anchor) {
  // 先全部換成畫面上的像素來算，寫回 style 時再除以介面大小（見 dom.js 的 zoom）
  const z = zoom(), r = anchor.getBoundingClientRect();
  const w = el.offsetWidth * z, ht = el.offsetHeight * z;
  const left = Math.max(16, Math.min(r.left, innerWidth - w - 16));
  let top = r.bottom + 8;
  if (top + ht > innerHeight - 16) top = Math.max(16, r.top - ht - 8);
  el.style.left = `${left / z}px`;
  el.style.top = `${top / z}px`;
}

// 回傳觸發按鈕，呼叫端可以把焦點還給它
export function closePop() {
  if (!cur) return null;
  const {el, anchor} = cur;
  cur = null;
  el.remove();
  anchor.setAttribute("aria-expanded", "false");
  return anchor;
}

// 目前開著的小視窗（id 相符才回傳）
export function openedPop(id) {
  return cur?.el.id === id ? cur.el : null;
}

document.addEventListener("keydown", e => {
  // 標成已處理，後面的 Esc 處理（例如圖庫的取消選取）就不會同時觸發
  if (e.key === "Escape" && cur) { e.preventDefault(); closePop()?.focus(); }
});
document.addEventListener("pointerdown", e => {
  if (cur && closesPop(e.target, cur.el, cur.anchor)) closePop();
});
addEventListener("resize", () => closePop());
