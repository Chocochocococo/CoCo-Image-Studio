// 說明文收在 ⓘ 裡：平常只看得到一個小小的 i，點了才展開（使用者筆記：介面上說明文太多、佔位置）。
// 回傳 [按鈕, 說明段落]，呼叫端把按鈕放在標題旁、段落放在標題下。
// 展開過的在這次開著的頁面裡記住，同一個地方重畫時保持展開（id：說明內容會換、位置不變時用同一個 id）。

import {h} from "./dom.js";
import {t} from "./i18n.js";

const OPEN = new Set();

export function info(key, vars, id = key) {
  const hint = h("p.hint.infotext", {text: t(key, vars), hidden: !OPEN.has(id)});
  const btn = h("button.info", {type: "button", "aria-expanded": String(OPEN.has(id)), title: t("ui.info"), "aria-label": t("ui.info"),
                                on: {click: e => {
                                  e.stopPropagation();
                                  const on = hint.hidden;
                                  hint.hidden = !on;
                                  btn.setAttribute("aria-expanded", String(on));
                                  if (on) OPEN.add(id); else OPEN.delete(id);
                                }}}, "i");
  return [btn, hint];
}
