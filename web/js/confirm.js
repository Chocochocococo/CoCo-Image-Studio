// 確認視窗：蓋在畫面中間，「取消」與「繼續」兩個按鈕，回傳 Promise<boolean>。
// 使用者筆記：提醒視窗要能直接繼續，不該關掉提醒之後還要再按一次原本的按鈕。
// Esc、點外面＝取消；一開始焦點在「取消」上，免得 Enter 一按就刪了

import {h} from "./dom.js";
import {t, translate} from "./i18n.js";

export function confirmDialog({title, body = [], ok, cancel, danger = true}) {
  return new Promise(resolve => {
    const back = document.activeElement;
    // 從看大圖叫出來的（丟進回收桶）：看大圖還開著，關掉確認視窗時不能把「疊層開著」的標記一起拿掉
    const under = document.body.classList.contains("viewing");
    const el = h("div.assets.confirm", {role: "alertdialog", "aria-modal": "true", "aria-labelledby": "cfTitle", "aria-describedby": "cfBody"});
    const done = v => {
      removeEventListener("keydown", onKey, true);
      el.remove();
      if (!under) document.body.classList.remove("viewing");
      back?.focus?.();
      resolve(v);
    };
    const onKey = e => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); }
    };
    const no = h("button.pill", {type: "button", text: cancel || t("cf.cancel"), on: {click: () => done(false)}});
    const yes = h(danger ? "button.pill.danger" : "button.pill.primary", {type: "button", text: ok, on: {click: () => done(true)}});
    el.addEventListener("click", e => { if (e.target === el) done(false); });
    el.append(h("div.apanel.cfpanel", {},
      h("h2.title", {id: "cfTitle", text: title}),
      h("div.cfbody", {id: "cfBody"}, ...[].concat(body).filter(Boolean)),
      h("div.sfoot", {}, h("span.sp"), no, yes)));
    translate(el);
    addEventListener("keydown", onKey, true);
    document.body.append(el);
    document.body.classList.add("viewing");
    no.focus();
  });
}
