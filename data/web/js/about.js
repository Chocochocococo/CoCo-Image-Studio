// 關於（⚙ 設定 → 關於）：程式名稱與版本號、簡短介紹、官網、GitHub、贊助。
// 版本號在 data\VERSION，伺服器送首頁時填進 <meta name="coco-version">（server/handler.py）；沒有就不顯示。
// 對外連結集中在這裡（README 與 .github/FUNDING.yml 的連結要跟這裡一樣，tests/test_about.py 檢查）。
// 空字串＝還沒有，不顯示

import {h} from "./dom.js";

export const LINKS = {
  site: "https://studio.cocos-homes.com",                     // 官網與使用手冊（2026-10-18 起）
  github: "https://github.com/Chocochocococo/CoCo-Image-Studio",  // 公開版；下載在 Releases
  ecpay: "https://p.ecpay.com.tw/D6622EF",       // 綠界：台灣（信用卡、ATM、超商）
  kofi: "https://ko-fi.com/chocochocococo",       // Ko-fi：海外（信用卡、PayPal）
};
const SPONSORS = [["ecpay", "about.ecpay", "about.ecpayHint"], ["kofi", "about.kofi", "about.kofiHint"]];

// 外部網站一律開新分頁（CoCo 這一頁留著）
const out = (url, cls, key) => h(`a.${cls}`, {href: url, target: "_blank", rel: "noopener noreferrer"},
  h("span", {"data-i18n": key}), " ↗");

export function aboutBody() {
  const links = [["site", "about.site"], ["github", "about.github"]].filter(([k]) => LINKS[k]);
  const ver = globalThis.document?.querySelector?.('meta[name="coco-version"]')?.content || "";   // 測試在 node 跑，document 是假的
  return [
    h("section.ssec", {},
      h("h3", {}, "CoCo Image Studio", ver ? h("span.aboutver", {text: `v${ver}`}) : null),
      h("p.aboutp", {"data-i18n": "about.intro"}),
      links.length ? h("div.aboutlinks", {}, links.map(([k, key]) => out(LINKS[k], "pill", key))) : null),
    h("section.ssec", {},
      h("h3", {"data-i18n": "about.support"}),
      h("p.hint", {"data-i18n": "about.supportHint"}),
      h("div.aboutsp", {}, SPONSORS.filter(([k]) => LINKS[k]).map(([k, name, hint]) => h("div.spcard", {},
        h("b", {"data-i18n": name}),
        h("span.hint", {"data-i18n": hint}),
        out(LINKS[k], "pill.primary", "about.go")))))];
}
