// 圖庫篩選的條件與查詢字串（沒有畫面，可以用 node 測）。畫面在 filters.js。
// 名稱跟 /api/list 的參數一致（server/library.py build_query）。
// 來源、模型、LoRA、資料夾可以多選（網址重複帶同一個參數）：
// 模型、資料夾、來源是「任一個」；LoRA 預設「全部都要有」，可以切成「任一個」。
export const MULTI = ["root", "model", "lora", "folder"];
// 評分：勾哪幾個星級就是哪幾個（可以複選），0＝未評分
export const RATINGS = ["5", "4", "3", "2", "1", "0"];
export const LORA_MODES = ["all", "any"];
// win：只看比較組裡標了勝出的圖（使用者筆記：勝出的圖越來越多，要能在圖庫一起看）
export const blank = () => ({q: "", root: [], model: [], lora: [], loraMode: "all", folder: [], rating: [],
                             orient: "", win: false, sort: "new"});

// 有沒有開篩選（搜尋另外算：搜尋框清掉就好，「清除篩選」不動它）
export const activeOf = f => MULTI.some(k => f[k].length) || f.rating.length > 0 || !!f.orient || f.win;

// 現在的時間（秒），跟圖檔的修改時間（gallery.db 的 mtime）同一種單位
export const nowSec = () => Date.now() / 1000;

// 開了幾個條件（收成一顆「篩選」鈕時寫在鈕上）：搜尋、每一種篩選各算一個
export const countOf = f => MULTI.filter(k => f[k].length).length + Number(f.rating.length > 0) + Number(!!f.orient)
  + Number(f.win) + Number(!!f.q);

// keepNew：生成頁的牆——開著篩選或搜尋時，since（最後一次改篩選的時間，秒）之後才生成的圖照樣列出（使用者選的，2026-10-10）。
// 從改篩選的時間算：取消再重新篩選時，之前的新圖都當成舊圖照篩選（使用者回報）
export function queryOf(f, {keepNew = false, since = 0} = {}) {
  const p = new URLSearchParams();
  if (f.q) p.set("q", f.q);
  for (const k of MULTI) for (const v of f[k]) p.append(k, v);
  if (f.lora.length > 1 && f.loraMode === "any") p.set("lora_mode", "any");
  for (const v of f.rating) p.append("rating", v);
  if (f.orient) p.set("orient", f.orient);
  if (f.win) p.set("win", "1");
  if (f.sort !== "new") p.set("sort", f.sort);
  if (keepNew && since > 0 && (activeOf(f) || f.q)) p.set("keep_new", since.toFixed(3));
  return p.toString();
}
