// 呼叫本機後端。所有 POST 都必須帶這次啟動的 token（見 server/security.py），
// 少了會被 403 擋下。token 由伺服器在回應 index.html 時注入到 window.COCO_TOKEN。

import {lang} from "./i18n.js";

let TOKEN = window.COCO_TOKEN || "";

// CoCo 重新啟動後 token 會換。這一頁是重開前打開的，POST 就會被 403 擋下，使用者只看到「forbidden」
// （2026-10-06：圖庫「加進資料集」沒複製，資料夾卻建好了，以為功能壞了）。重新讀一次首頁拿新的 token、再送一次。
// 別的網站讀不到 CoCo 的首頁（瀏覽器的同源政策），所以不會削弱 token 的防護
async function freshToken() {
  const html = await (await fetch("/", {cache: "no-store"})).text();
  const m = html.match(/window\.COCO_TOKEN = ("(?:[^"\\]|\\.)*")/);
  const got = m ? JSON.parse(m[1]) : "";
  if (!got || got === TOKEN) return false;
  TOKEN = got;
  window.COCO_TOKEN = got;
  return true;
}

// 後端的錯誤與狀態訊息照介面目前的語言產生（core/i18n.py）
const langHeader = () => ({"X-CoCo-Lang": lang()});

export async function get(path) {
  const r = await fetch(path, {cache: "no-store", headers: langHeader()});
  return r.json();
}

// 圖片之類的二進位內容（精修的魔術棒要讀底圖的像素）
export async function getBlob(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.blob();
}

// keepalive：關分頁時送出的請求也要送到（例如還沒存的設定）
export async function post(path, body, {keepalive = false} = {}) {
  const send = () => fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-CoCo-Token": TOKEN, ...langHeader()},
    body: JSON.stringify(body ?? {}),
    keepalive,
  });
  let r = await send();
  if (r.status === 403 && await freshToken().catch(() => false)) r = await send();
  return r.json();
}
