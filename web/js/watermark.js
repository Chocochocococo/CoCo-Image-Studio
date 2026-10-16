// 浮水印：文字或圖片（logo），放在九宮格的某一格或整片平鋪。06 輸出的「浮水印」與批次輸出共用。
// 設定記在介面偏好 ewm；logo 存在伺服器（edits/watermark.png），換電腦以外都一直在。
// 大小＝一個浮水印的寬度佔圖寬的 %；離邊＝短邊的 %。透明度不在這裡畫：
// 編輯器裡當成浮水印圖層的不透明度（之後還能調），批次輸出時外面設 globalAlpha

import {get, getBlob, post} from "./api.js";
import {drawText, measureText} from "./textdraw.js";
import {wmSpots} from "./editops.js";

export const wmset = ui => ({type: "text", text: "", font: "Microsoft JhengHei", color: "#ffffff", bold: true, shadow: true,
                            size: 25, opacity: 60, pos: "br", margin: 3, angle: -30, ...(ui?.ewm || {})});

let logo = {v: -1, bmp: null};

// 載入 logo（有換過才重新抓）；沒有 logo 回 null
export async function loadLogo() {
  const v = (await get("/api/edit/wminfo").catch(() => ({v: 0}))).v || 0;
  if (v === logo.v) return logo.bmp;
  logo = {v, bmp: v ? await createImageBitmap(await getBlob(`/api/edit/wmlogo?v=${v}`)).catch(() => null) : null};
  return logo.bmp;
}
export const logoNow = () => logo.bmp;
export const logoVer = () => logo.v;

// 換 logo：任何格式的圖都先畫成 PNG 再送
export async function setLogo(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement("canvas");
  c.width = bmp.width; c.height = bmp.height;
  c.getContext("2d").drawImage(bmp, 0, 0);
  const png = c.toDataURL("image/png");
  const d = await post("/api/edit/wmlogo", {png});
  if (d.error) throw new Error(d.error);
  return loadLogo();
}
export async function clearLogo() {
  await post("/api/edit/wmlogo", {});
  return loadLogo();
}

// 設好了沒（文字有字、或圖片模式有 logo）
export const wmReady = (wm, bmp = logo.bmp) => (wm.type === "image" ? !!bmp : !!String(wm.text || "").trim());

export function drawWatermark(ctx, W, H, wm, bmp = logo.bmp) {
  const width = Math.max(1, W * wm.size / 100), margin = Math.min(W, H) * wm.margin / 100;
  let w, h, stamp;
  if (wm.type === "image") {
    if (!bmp) return;
    w = width;
    h = width * bmp.height / bmp.width;
    stamp = () => ctx.drawImage(bmp, -w / 2, -h / 2, w, h);
  } else {
    if (!String(wm.text || "").trim()) return;
    const T0 = {str: wm.text, font: wm.font, size: 100, bold: wm.bold, color: wm.color, align: "center", shadow: wm.shadow};
    const T = {...T0, size: 100 * width / measureText(ctx, T0).w};    // 字的大小照寬度換算
    const m = measureText(ctx, T);
    w = m.w;
    h = m.h;
    stamp = () => drawText(ctx, T, 0, -h / 2);
  }
  for (const s of wmSpots(W, H, w, h, wm.pos, margin, wm.angle)) {
    ctx.save();
    ctx.translate(s.x, s.y);
    ctx.rotate(s.rot);
    ctx.imageSmoothingQuality = "high";
    stamp();
    ctx.restore();
  }
}
