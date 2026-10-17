// 配色主題：每組都有深色與淺色。
//
// 底與卡片刻意壓低彩度，顏色集中在 ac（主要按鈕）、sg（次要）、tg（提示詞
// 標籤）、wn（權重與提醒）。生圖工具的底色會影響人判斷作品的色調，所以不開放
// 任意自訂顏色；每一組的文字與按鈕對比都是調過的。
//
// 名稱格式：兩個中文字＋一個英文字，程式代號用英文小寫。顯示名稱在 locales/。

export const THEMES = {
  cherry: {   // 參考：影色のさくらんぼ。幾乎黑白，最不干擾作品顏色；預設
    dark:  {bg:"#141414",p1:"#1d1c1b",p2:"#262422",p3:"#36332f",tx:"#e7e3e0",dm:"#ac9e9b",ft:"#6f6563",ac:"#cdb6b3",ink:"#1d1817",sg:"#948683",tg:"#d6c7c3",wn:"#dcc2a6",tape:"rgba(231,227,224,.5)",tape2:"rgba(172,158,155,.72)"},
    light: {bg:"#e7e3e0",p1:"#f6f4f2",p2:"#eeeae7",p3:"#ddd7d3",tx:"#1c1b1a",dm:"#695f5d",ft:"#a39896",ac:"#36332f",ink:"#efebe8",sg:"#695f5d",tg:"#7a5f5a",wn:"#8a5a4a",tape:"rgba(172,158,155,.72)",tape2:"rgba(105,95,93,.55)"},
  },
  lamplight: {   // 參考：暮れ際の灯影
    dark:  {bg:"#202528",p1:"#282e32",p2:"#30373c",p3:"#3a4349",tx:"#ede7db",dm:"#a8b2b4",ft:"#6f7c81",ac:"#eca93a",ink:"#281a04",sg:"#8fa7ae",tg:"#a9c3ca",wn:"#d9c7a8",tape:"rgba(236,169,58,.5)",tape2:"rgba(188,172,149,.72)"},
    light: {bg:"#e6e1d8",p1:"#f6f3ee",p2:"#ece7df",p3:"#dcd5c9",tx:"#2d3337",dm:"#5b6a70",ft:"#97a1a2",ac:"#2d4656",ink:"#f6f1e6",sg:"#657a80",tg:"#35667a",wn:"#b0741a",tape:"rgba(236,169,58,.55)",tape2:"rgba(188,172,149,.85)"},
  },
  moss: {   // 參考：Green Color Palettes
    dark:  {bg:"#191c17",p1:"#20241e",p2:"#282d25",p3:"#333a2f",tx:"#e9e6dc",dm:"#a7ac9c",ft:"#6f7667",ac:"#a3b79f",ink:"#18230f",sg:"#c8c2b4",tg:"#b3c6ae",wn:"#d8c894",tape:"rgba(200,194,180,.6)",tape2:"rgba(138,157,135,.72)"},
    light: {bg:"#e6e3da",p1:"#f7f5ef",p2:"#eeebe3",p3:"#dfdbcf",tx:"#26301f",dm:"#5d6656",ft:"#99a08f",ac:"#3a533a",ink:"#f4f1e8",sg:"#6f8a6b",tg:"#4c6b48",wn:"#8a6a2a",tape:"rgba(200,194,180,.88)",tape2:"rgba(138,157,135,.8)"},
  },
  cocoa: {   // 參考：堅果黑巧
    dark:  {bg:"#1b1714",p1:"#241f1b",p2:"#2d2722",p3:"#39312b",tx:"#f3e4d4",dm:"#bba998",ft:"#86776a",ac:"#d7c2ad",ink:"#2a1f16",sg:"#a5907d",tg:"#e2c8ad",wn:"#f0b98a",tape:"rgba(243,222,203,.55)",tape2:"rgba(165,144,125,.72)"},
    light: {bg:"#ece3d8",p1:"#fbf7f2",p2:"#f3ebe1",p3:"#e4d7c9",tx:"#3a2e24",dm:"#76675a",ft:"#ab9c8e",ac:"#7a6655",ink:"#fdf7f0",sg:"#a5907d",tg:"#8a5f3f",wn:"#a0663a",tape:"rgba(199,182,166,.85)",tape2:"rgba(243,222,203,.95)"},
  },
  brick: {   // 參考：石街の温度
    dark:  {bg:"#1f1f25",p1:"#28282f",p2:"#303038",p3:"#3b3b44",tx:"#eceaec",dm:"#abb4bf",ft:"#707683",ac:"#c96a62",ink:"#220f0d",sg:"#abb7c2",tg:"#c3cdd7",wn:"#d9a07a",tape:"rgba(194,198,204,.5)",tape2:"rgba(171,183,194,.66)"},
    light: {bg:"#e2e4e8",p1:"#f5f6f8",p2:"#ebedf0",p3:"#dadde2",tx:"#2b2b32",dm:"#5c5a62",ft:"#9ba1a9",ac:"#7e3a36",ink:"#fbf2f1",sg:"#654748",tg:"#4a6478",wn:"#9a4a2f",tape:"rgba(194,198,204,.9)",tape2:"rgba(171,183,194,.85)"},
  },
  rose: {   // 草稿 v4 原本的「夜讀」
    dark:  {bg:"#171615",p1:"#201e1c",p2:"#292725",p3:"#34312e",tx:"#ede7df",dm:"#a89f95",ft:"#766e65",ac:"#e0a3ae",ink:"#2a1318",sg:"#a9bf9f",tg:"#b9cdb2",wn:"#e3c088",tape:"rgba(224,200,160,.62)",tape2:"rgba(169,191,159,.6)"},
    light: {bg:"#e9e7e1",p1:"#fbfaf7",p2:"#f2f0eb",p3:"#e5e2da",tx:"#2a2623",dm:"#6f6760",ft:"#a39a91",ac:"#b25d72",ink:"#ffffff",sg:"#5f7f58",tg:"#4f7449",wn:"#9a6b1f",tape:"rgba(214,190,140,.7)",tape2:"rgba(150,180,140,.7)"},
  },
};

export const DEFAULT_THEME = "cherry";
export const MODES = ["dark", "light"];

// 提醒色（wn）當底時的字色：深色主題的 wn 偏亮、淺色的偏暗，還有 lamplight 淺色那種中間亮度的橘，
// 所以不從主題裡挑，直接看黑、白哪個對比高（WCAG 的相對亮度）
export function inkOn(hex) {
  const lum = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255)
    .map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
    .reduce((s, c, i) => s + c * [0.2126, 0.7152, 0.0722][i], 0);
  const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  const L = lum(hex), dark = "#1a1a1a", light = "#ffffff";
  return ratio(L, lum(dark)) >= ratio(L, lum(light)) ? dark : light;
}

// 從色票表產生每一組的 CSS 變數，一次注入。元件只用變數，不寫死顏色。
function css() {
  let out = "";
  for (const [key, modes] of Object.entries(THEMES)) {
    for (const mode of MODES) {
      const v = modes[mode], dark = mode === "dark";
      out += `:root[data-theme="${key}"][data-mode="${mode}"]{color-scheme:${mode};`
        + `--bg:${v.bg};--p1:${v.p1};--p2:${v.p2};--p3:${v.p3};`
        + `--ln:${dark ? "rgba(255,255,255,.07)" : "rgba(0,0,0,.08)"};`
        + `--tx:${v.tx};--dm:${v.dm};--ft:${v.ft};--ac:${v.ac};--ac-ink:${v.ink};`
        + `--ac-soft:color-mix(in srgb,${v.ac} 16%,transparent);`
        + `--sg:${v.sg};--sg-soft:color-mix(in srgb,${v.sg} 15%,transparent);`
        + `--tg:${v.tg};--wn:${v.wn};--wn-ink:${inkOn(v.wn)};--tape:${v.tape};--tape2:${v.tape2};`
        + `--veil:${dark ? "rgba(8,8,8,.64)" : "rgba(236,234,228,.72)"};`
        + `--sh:${dark ? "0 24px 60px rgba(0,0,0,.5)" : "0 20px 50px rgba(40,30,20,.16)"}}\n`;
    }
  }
  return out;
}

let injected = false;
export function applyTheme(theme, mode) {
  if (!injected) {
    const st = document.createElement("style");
    st.id = "coco-themes";
    st.textContent = css();
    document.head.prepend(st);
    injected = true;
  }
  const t = THEMES[theme] ? theme : DEFAULT_THEME;
  const m = MODES.includes(mode) ? mode : "dark";
  document.documentElement.dataset.theme = t;
  document.documentElement.dataset.mode = m;
  return {theme: t, mode: m};
}

// 主題卡片上的小色點：每組取底、卡片、主色、次色、文字
export function swatches(theme, mode) {
  const v = (THEMES[theme] || THEMES[DEFAULT_THEME])[mode] || THEMES[DEFAULT_THEME].dark;
  return [v.bg, v.p2, v.ac, v.sg, v.tx];
}
