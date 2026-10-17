// 多視窗（使用者筆記 #10）：同一個瀏覽器裡開好幾個 CoCo 視窗時互相通知，例如資源庫開在另一個螢幕、
// 生成頁與精修頁各佔一個螢幕。用 BroadcastChannel：只在同一個網址（來源）的視窗之間傳，不經過伺服器。
//
// 訊息（type → data）：
//   settings  {patch}   某個視窗存了設定欄位（generate.js 的 flush），其他視窗照著改
//   reload    {}        伺服器上的設定被改了（換模型、VAE、⚙ 設定），其他視窗重讀
//   tab       {key}     生成台的提示詞編輯區切到正面／負面（資源庫的 Embeddings 要加到哪一邊）
//   ui        {patch}   外觀偏好（主題、深淺、語言、介面大小、多視窗）
//   assets:*            資源庫獨立視窗：here（我還開著）、bye（關了）、ping（誰在？）、show {kind}（切到這一頁）

const ch = typeof BroadcastChannel === "function" ? new BroadcastChannel("coco") : null;
const handlers = new Map();

export function send(type, data = {}) {
  try { ch?.postMessage({type, data}); } catch { /* 收不到的視窗就等它自己輪詢 */ }
}

export function on(type, fn) {
  if (!handlers.has(type)) handlers.set(type, []);
  handlers.get(type).push(fn);
}

ch?.addEventListener("message", e => {
  for (const fn of handlers.get(e.data?.type) || []) {
    try { fn(e.data.data || {}); } catch (err) { console.error(err); }
  }
});

// ── 資源庫的獨立視窗還在不在 ──
// 它每 2 秒說一次「我還開著」，關掉時說「bye」；當掉沒說的話 6 秒後也當成關了
const ALIVE_MS = 6000;
let assetsSeen = 0;
on("assets:here", () => { assetsSeen = Date.now(); });
on("assets:bye", () => { assetsSeen = 0; });
export const assetsWindowOpen = () => Date.now() - assetsSeen < ALIVE_MS;
send("assets:ping");                       // 剛打開的視窗先問一次，不用等 2 秒

// ── ⧉（開到新視窗）要不要出現 ──
// auto：偵測到第二個螢幕才出現（screen.isExtended，只有 Chrome／Edge 有；Firefox、Brave 當成一個螢幕）；
// on / off：手動
export const MULTI_MODES = ["auto", "on", "off"];
export function multiEnabled(pref, scr = globalThis.screen) {
  if (pref === "on") return true;
  if (pref === "off") return false;
  return !!scr?.isExtended;
}

// 開新視窗。同一個 name 再開一次是同一個視窗（換到那一頁、不會多開）。
// 允許過「視窗管理」權限的話直接放到另一個螢幕、佔滿它；還沒問過就順便問，下次就能直接放過去
export async function openWindow(url, name, {width = 1500, height = 950} = {}) {
  let feat = `popup,width=${width},height=${height}`;
  try {
    if (typeof getScreenDetails === "function" && screen.isExtended) {
      const st = await navigator.permissions.query({name: "window-management"}).catch(() => null);
      if (st?.state === "granted") {
        const sd = await getScreenDetails();
        const other = sd.screens.find(s => s !== sd.currentScreen);
        if (other) feat = `popup,left=${other.availLeft},top=${other.availTop},width=${other.availWidth},height=${other.availHeight}`;
      } else if (st?.state === "prompt") {
        getScreenDetails().catch(() => { /* 不給也沒關係，就是一般的新視窗 */ });
      }
    }
  } catch { /* 不支援就是一般的新視窗 */ }
  return window.open(url, name, feat);
}
