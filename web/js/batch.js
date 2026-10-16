// 批次輸出（06 輸出第 5 輪）：圖庫選好幾張，一次照同樣的設定輸出——尺寸（原尺寸／長邊幾像素／縮放 %）、
// 檔案類型與品質、存到圖庫或資料夾、要不要保留生成參數（使用者筆記：不一定要批次才能清參數，
// 06 輸出裡單張也能選；存進圖庫一律保留）、要不要加浮水印（用 06 輸出設好的那組）。
// 一張一張在瀏覽器裡畫（加浮水印），再交給 /api/edit/export 縮放、轉檔、寫參數——跟單張輸出同一條路

import {getBlob, post} from "./api.js";
import {h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {pickFolder} from "./setup.js";
import {drawWatermark, loadLogo, wmReady, wmset} from "./watermark.js";
import {outSize} from "./editops.js";

const FORMATS = ["png", "jpg", "webp"];
const SIZES = ["orig", "long", "pct"];
let UI = null;

export function initBatch(ui) { UI = ui; }

const saveUi = patch => { Object.assign(UI, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const bset = () => ({size: "orig", long: 2048, pct: 50, wm: false, ...(UI?.ebatch || {})});
const setB = patch => saveUi({ebatch: {...bset(), ...patch}});

function pngOf(canvas) {
  return new Promise(res => canvas.toBlob(b => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(b);
  }, "image/png"));
}

export async function openBatch(items) {
  if (!items.length) return;
  const logo = await loadLogo().catch(() => null);
  const back = document.activeElement;
  const el = h("div.assets.confirm", {role: "dialog", "aria-modal": "true", "aria-labelledby": "btTitle"});
  let running = false, cancel = false;
  const close = () => {
    if (running) { cancel = true; return; }
    removeEventListener("keydown", onKey, true);
    el.remove();
    document.body.classList.remove("viewing");
    back?.focus?.();
  };
  const onKey = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); } };
  const panel = h("div.apanel.cfpanel.edbatch");
  const fmt = () => (FORMATS.includes(UI?.eformat) ? UI.eformat : "png");
  const toLib = () => UI?.edest !== "folder";
  const seg = (aria, keys, on, label, click) => h("div.seg", {role: "group", "data-i18n-aria": aria}, keys.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), ...label(k), on: {click: () => { click(k); draw(); }}})));
  const row = (key, ...kids) => h("div.rflab.btrow", {}, h("span.dim", {"data-i18n": key}), h("span.sp"), ...kids);
  const msg = h("p.hint.edmsg", {role: "status"});
  const go = h("button.pill.primary", {type: "button", text: t("bt.start", {n: items.length})});
  const stop = h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: close}});

  function draw() {
    const b = bset(), wm = wmset(UI), ready = wmReady(wm, logo), q = Number(UI?.equality) || 92;
    const num = (key, val, min, max) => h("input.v.btnum", {type: "number", min: String(min), max: String(max), step: "1", value: String(val),
                                                            "data-i18n-aria": "bt.size", on: {change: e => setB({[key]: Math.max(min, Math.min(max, Math.round(Number(e.target.value) || min)))})}});
    panel.replaceChildren(...[
      h("h2.title", {id: "btTitle", text: t("bt.title", {n: items.length})}),
      row("bt.size", seg("bt.size", SIZES, k => b.size === k, k => ({"data-i18n": `bt.size.${k}`}), k => setB({size: k})),
        b.size === "long" ? num("long", b.long, 16, 8192) : null, b.size === "long" ? h("span.dim", {text: "px"}) : null,
        b.size === "pct" ? num("pct", b.pct, 1, 400) : null, b.size === "pct" ? h("span.dim", {text: "%"}) : null),
      row("ed.format", seg("ed.format", FORMATS, k => fmt() === k, k => ({text: k.toUpperCase()}), k => saveUi({eformat: k}))),
      fmt() !== "png" ? row("ed.quality", h("input", {type: "range", min: "50", max: "100", step: "1", value: String(q), "data-i18n-aria": "ed.quality",
                                                      on: {change: e => { saveUi({equality: Number(e.target.value)}); draw(); }}}),
                           h("span.mono.rfnum", {text: String(q)})) : null,
      row("ed.dest", seg("ed.dest", ["library", "folder"], k => (k === "library") === toLib(), k => ({"data-i18n": `ed.dest.${k}`}), k => saveUi({edest: k}))),
      toLib() ? null : h("div.newset", {},
        h("input.find", {type: "text", value: UI?.efolder || "", spellcheck: "false", "data-i18n-placeholder": "lib.dest", "data-i18n-aria": "lib.dest",
                         on: {change: e => saveUi({efolder: e.target.value.trim()})}}),
        h("button.pill", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
          const got = await pickFolder(UI?.efolder || "");
          if (got) { saveUi({efolder: got}); draw(); }
        }}})),
      h("div.cell.swcell", {},
        h("span.swtext", {}, h("b", {id: "btKeepLbl", "data-i18n": "ed.keep"}), h("span.hint", {"data-i18n": toLib() ? "ed.keepLibrary" : "ed.keepHint"})),
        h("button.switch", {type: "button", role: "switch", "aria-labelledby": "btKeepLbl", disabled: toLib(), "aria-checked": String(toLib() || UI?.ekeep !== false),
                            on: {click: () => { saveUi({ekeep: UI?.ekeep === false}); draw(); }}}, h("i"))),
      h("div.cell.swcell", {},
        h("span.swtext", {}, h("b", {id: "btWmLbl", "data-i18n": "bt.wm"}), h("span.hint", {"data-i18n": ready ? "bt.wmHint" : "bt.wmNone"})),
        h("button.switch", {type: "button", role: "switch", "aria-labelledby": "btWmLbl", disabled: !ready, "aria-checked": String(ready && b.wm),
                            on: {click: () => { setB({wm: !b.wm}); draw(); }}}, h("i"))),      // 存了要重畫，開關才會馬上換（使用者回報）
      msg,
      h("div.sfoot", {}, h("span.sp"), stop, go)].filter(Boolean));
    translate(panel);
    if (running) for (const b2 of panel.querySelectorAll("button, input")) b2.disabled = true;   // 跑的時候設定都不能動
    go.disabled = running;
    stop.disabled = false;
  }

  go.addEventListener("click", async () => {
    if (running) return;
    if (!toLib() && !(UI?.efolder || "").trim()) { msg.dataset.kind = "err"; msg.textContent = t("ed.needFolder"); return; }
    running = true;
    cancel = false;
    stop.dataset.i18n = "bt.stop";
    draw();
    const b = bset(), wm = wmset(UI), useWm = b.wm && wmReady(wm, logo), errs = [];
    let done = 0;
    for (const it of items) {
      if (cancel) break;
      msg.dataset.kind = "";
      msg.textContent = t("bt.progress", {i: done + 1, n: items.length});
      try {
        const bmp = await createImageBitmap(await getBlob(`/api/full?id=${it.id}&v=${encodeURIComponent(it.v || "")}`));
        const c = document.createElement("canvas");
        c.width = bmp.width; c.height = bmp.height;
        const ctx = c.getContext("2d");
        ctx.drawImage(bmp, 0, 0);
        if (useWm) {
          ctx.globalAlpha = wm.opacity / 100;
          drawWatermark(ctx, c.width, c.height, wm, logo);
          ctx.globalAlpha = 1;
        }
        const size = outSize(c.width, c.height, b);
        const d = await post("/api/edit/export", {
          src_id: it.id, png: await pngOf(c), format: fmt(), quality: Number(UI?.equality) || 92, width: size.w, height: size.h,
          dest: toLib() ? "library" : "folder", folder: UI?.efolder || "", keep_meta: UI?.ekeep !== false});
        if (d.error) throw new Error(d.error);
        done++;
      } catch (e) {
        errs.push(`${it.filename || it.id}：${e.message || e}`);
      }
    }
    running = false;
    stop.dataset.i18n = "bt.close";
    draw();
    msg.dataset.kind = errs.length ? "err" : "ok";
    msg.replaceChildren(t(cancel ? "bt.stopped" : "bt.done", {n: done}), ...errs.slice(0, 5).map(x => h("span.bterr", {text: x})));
  });

  draw();
  el.addEventListener("click", e => { if (e.target === el && !running) close(); });
  el.append(panel);
  addEventListener("keydown", onKey, true);
  document.body.append(el);
  document.body.classList.add("viewing");
  go.focus();
}
