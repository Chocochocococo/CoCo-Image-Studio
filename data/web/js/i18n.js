// 介面文字。繁體中文是原文，其他語言缺字時退回繁體中文，再缺就顯示鍵名，
// 讓漏翻的地方一眼看得出來，而不是變成空白。
//
// 技術名詞（LoRA、VAE、CFG、Recipe、取樣器名稱、token）兩種語言都保留英文。
// 用法：t("nav.generate")、t("wall.count", {n: 12})；
// HTML 裡用 data-i18n="鍵" / data-i18n-title / data-i18n-placeholder。
//
// 單複數：n 是 1、而目前語言的語系檔有「鍵.one」就改用它（英文 "1 winner"／"3 winners"）。
// 中文不分單複數，zh-TW.json 的 .one 跟原鍵同字（兩份語系檔的鍵要一樣）；後端 core/i18n.py 的 tr() 同規則。

export const LANGS = [["zh-TW", "繁體中文"], ["en", "English"]];
const SOURCE = "zh-TW";

let dict = {}, fallback = {}, current = SOURCE;

async function load(lang) {
  const r = await fetch(`locales/${lang}.json`);
  if (!r.ok) throw new Error(`locale ${lang}: ${r.status}`);
  return r.json();
}

// 偏好空字串＝跟著瀏覽器；瀏覽器是中文就用繁中，其餘一律英文
export function pickLang(pref) {
  if (pref && LANGS.some(([k]) => k === pref)) return pref;
  const nav = (navigator.language || "").toLowerCase();
  return nav.startsWith("zh") ? "zh-TW" : "en";
}

export async function initI18n(pref) {
  current = pickLang(pref);
  fallback = await load(SOURCE);
  dict = current === SOURCE ? fallback : await load(current).catch(() => ({}));
  document.documentElement.lang = current;
  return current;
}

export function lang() { return current; }

export function t(key, params) {
  // 只看目前語言的 .one：英文缺 .one 時用英文的複數，不要退回成繁中的字
  if (params && String(params.n) === "1" && dict[`${key}.one`] != null) key = `${key}.one`;
  const s = dict[key] ?? fallback[key] ?? key;
  if (!params) return s;
  return s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}

export function translate(root = document) {
  root.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  root.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  root.querySelectorAll("[data-i18n-aria]").forEach(el => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); });
}
