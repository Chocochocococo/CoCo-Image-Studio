// 拼接（06 輸出）：圖庫選好幾張拼成一張。排法：格狀（欄數、格子比例、裁切填滿或完整放進）、
// 直排（一樣寬，由上往下）、橫排（一樣高，由左往右）、自由排（每張自己拖位置、拖右下角縮放）。
// 間距、外框留白、圓角、背景（顏色或透明）；下面的縮圖列拖曳換順序（自由排是換上下層）。
// 輸出跟單張編輯同一條路（/api/edit/export，沒有單一來源圖，所以不帶生成參數）：
// 尺寸、檔案類型、存到圖庫或資料夾、加不加浮水印。存進圖庫後可以按「編輯這張」接著加字。
// 排版的計算在 collageops.js。

import {getBlob, post} from "./api.js";
import {h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {pickFolder} from "./setup.js";
import {drawWatermark, loadLogo, wmReady, wmset} from "./watermark.js";
import {openEditor} from "./editor.js";
import {CELLS, MAX_ITEMS, MODES, fitFree, freeStart, hitBox, layout, moveItem, outScale, resizeBox} from "./collageops.js";

const FORMATS = ["png", "jpg", "webp"];
const PREVIEW = 1600;              // 預覽最多畫這麼大（長邊），拖曳時才順
let UI = null;

export function initCollage(ui) { UI = ui; }

const saveUi = patch => { Object.assign(UI, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const cset = () => ({mode: "grid", cols: 0, cell: "first", fit: "cover", gap: 16, pad: 16, radius: 0, bg: "#ffffff", clear: false,
                     size: "orig", long: 4096, wm: false, ...(UI?.collage || {})});
const setC = patch => saveUi({collage: {...cset(), ...patch}});

function pngOf(canvas) {
  return new Promise(res => canvas.toBlob(b => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(b);
  }, "image/png"));
}

export async function openCollage(picked) {
  const items = picked.slice(0, MAX_ITEMS);
  if (items.length < 2) return;
  const C = {items, bmps: new Map(), free: null, sel: -1, drag: null, raf: 0, logo: null, busy: false};
  const back = document.activeElement;
  const el = h("div.assets", {role: "dialog", "aria-modal": "true", "aria-labelledby": "clgTitle"});
  const close = () => {
    if (C.busy) return;
    removeEventListener("keydown", onKey, true);
    ro.disconnect();
    el.remove();
    C.bmps.forEach(b => b.close?.());
    document.body.classList.remove("viewing");
    back?.focus?.();
  };
  const onKey = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); } };
  const side = h("div.clgside");
  const preview = h("canvas.clgcanvas", {"data-i18n-aria": "clg.preview", role: "img"});
  const stage = h("div.clgstage", {}, h("p.dim", {text: t("clg.loading", {i: 0, n: items.length})}));
  const strip = h("div.clgstrip", {role: "list", "data-i18n-aria": "clg.order"});
  const msg = h("p.hint.edmsg", {role: "status"});
  const go = h("button.pill.primary", {type: "button", "data-i18n": "clg.go", disabled: true});

  const sizes = () => C.items.map(it => C.bmps.get(it.id)).map(b => ({w: b.width, h: b.height}));
  // 目前的排版：自由排用自己拖過的位置（張數變了就重排）
  const current = () => {
    const s = cset(), sz = sizes();
    if (s.mode !== "free") return layout(sz, s);
    if (!C.free || C.free.boxes.length !== sz.length) C.free = freeStart(sz, s);
    return {W: C.free.W, H: C.free.H, boxes: C.free.boxes.map((b, i) => ({...b, sx: 0, sy: 0, sw: sz[i].w, sh: sz[i].h}))};
  };

  function render(canvas, L, k, useWm) {
    const s = cset();
    canvas.width = Math.max(1, Math.round(L.W * k));
    canvas.height = Math.max(1, Math.round(L.H * k));
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!s.clear) { ctx.fillStyle = s.bg; ctx.fillRect(0, 0, canvas.width, canvas.height); }
    ctx.imageSmoothingQuality = "high";
    L.boxes.forEach((b, i) => {
      const bmp = C.bmps.get(C.items[i].id);
      if (!bmp) return;
      ctx.save();
      if (s.radius > 0) {
        ctx.beginPath();
        ctx.roundRect(b.x * k, b.y * k, b.w * k, b.h * k, Math.min(s.radius * k, b.w * k / 2, b.h * k / 2));
        ctx.clip();
      }
      ctx.drawImage(bmp, b.sx, b.sy, b.sw, b.sh, b.x * k, b.y * k, b.w * k, b.h * k);
      ctx.restore();
    });
    if (useWm) {
      const wm = wmset(UI);
      ctx.globalAlpha = wm.opacity / 100;
      drawWatermark(ctx, canvas.width, canvas.height, wm, C.logo);
      ctx.globalAlpha = 1;
    }
  }

  const useWm = () => cset().wm && wmReady(wmset(UI), C.logo);
  // 拖曳時一個畫面只畫一次；其他時候直接畫
  const schedule = () => { if (!C.raf) C.raf = requestAnimationFrame(() => { C.raf = 0; drawPreview(); }); };
  function drawPreview() {
    const L = current(), k = Math.min(1, PREVIEW / Math.max(L.W, L.H));
    render(preview, L, k, useWm());
    fitPreview(L);
    // 自由排：選中的那張畫外框與右下角的縮放點
    if (cset().mode === "free" && L.boxes[C.sel]) {
      const b = L.boxes[C.sel], ctx = preview.getContext("2d");
      const px = Math.max(1, L.W * k / (preview.getBoundingClientRect().width || 1));
      const ac = getComputedStyle(el).getPropertyValue("--ac").trim() || "#e27497";
      ctx.save();
      ctx.lineWidth = 2 * px;
      ctx.strokeStyle = ac;
      ctx.strokeRect(b.x * k, b.y * k, b.w * k, b.h * k);
      ctx.fillStyle = ac;
      const g = 7 * px;
      ctx.fillRect(b.x * k + b.w * k - g, b.y * k + b.h * k - g, g * 2, g * 2);
      ctx.restore();
    }
    const k2 = outScale(L.W, L.H, cset());
    outInfo.textContent = t("clg.outSize", {w: Math.round(L.W * k2), h: Math.round(L.H * k2)});
  }
  const outInfo = h("span.dim.mono.small");
  // 預覽縮到剛好放進預覽區（CSS 的 max-height: 100% 在 grid 裡會失效，所以自己算）
  const fitPreview = (L = current()) => {
    const cs = getComputedStyle(stage);
    const sw = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const sh = stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const k = Math.max(0.01, Math.min(sw / L.W, sh / L.H));
    preview.style.width = `${Math.max(1, Math.floor(L.W * k))}px`;
    preview.style.height = `${Math.max(1, Math.floor(L.H * k))}px`;
  };
  const ro = new ResizeObserver(() => { if (preview.isConnected && C.items.every(it => C.bmps.has(it.id))) fitPreview(); });
  ro.observe(stage);

  // ── 自由排：拖位置、拖右下角縮放 ──
  const toLayout = e => {
    const r = preview.getBoundingClientRect(), L = current();
    return {p: [(e.clientX - r.left) / r.width * L.W, (e.clientY - r.top) / r.height * L.H], unit: L.W / r.width};
  };
  preview.addEventListener("pointerdown", e => {
    if (cset().mode !== "free" || !C.free) return;
    const {p, unit} = toLayout(e);
    const hit = hitBox(C.free.boxes, p, 12 * unit);
    C.sel = hit ? hit.i : -1;
    if (hit) {
      C.drag = {...hit, start: p, box: {...C.free.boxes[hit.i]}};
      try { preview.setPointerCapture(e.pointerId); } catch { /* 沒有實體指標（例如程式送的事件） */ }
    }
    drawSide();
    drawPreview();
  });
  preview.addEventListener("pointermove", e => {
    if (cset().mode !== "free" || !C.free) return;
    const {p, unit} = toLayout(e);
    if (!C.drag) {
      const hit = hitBox(C.free.boxes, p, 12 * unit);
      preview.style.cursor = hit ? (hit.part === "size" ? "nwse-resize" : "move") : "";
      return;
    }
    const d = C.drag, b = d.box;
    C.free.boxes[d.i] = d.part === "size" ? resizeBox(b, p) : {...b, x: b.x + p[0] - d.start[0], y: b.y + p[1] - d.start[1]};
    schedule();
  });
  const endDrag = () => { C.drag = null; };
  preview.addEventListener("pointerup", endDrag);
  preview.addEventListener("pointercancel", endDrag);

  // 換順序（自由排的位置跟著那張圖走）
  function reorder(from, to) {
    if (from === to) return;
    if (C.free) C.free = {...C.free, boxes: moveItem(C.free.boxes, from, to)};
    if (C.sel === from) C.sel = to;
    C.items = moveItem(C.items, from, to);
    drawStrip();
    drawPreview();
  }
  function remove(i) {
    if (C.items.length <= 2) return;
    if (C.free) C.free = {...C.free, boxes: C.free.boxes.filter((_, j) => j !== i)};
    C.items = C.items.filter((_, j) => j !== i);
    C.sel = -1;
    drawStrip();
    drawSide();
    drawPreview();
  }

  function drawStrip() {
    let from = -1;
    strip.replaceChildren(...C.items.map((it, i) => h("div.clgthumb", {role: "listitem", draggable: "true", tabindex: "0",
      "aria-label": t("clg.item", {i: i + 1, name: it.filename || ""}), "data-sel": cset().mode === "free" && C.sel === i ? "" : null,
      on: {
        dragstart: e => { from = i; e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", String(i)); },
        dragover: e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; },
        drop: e => { e.preventDefault(); reorder(from >= 0 ? from : Number(e.dataTransfer.getData("text/plain")), i); },
        click: () => { if (cset().mode === "free") { C.sel = i; drawStrip(); drawSide(); drawPreview(); } },
        keydown: e => {
          if (e.key === "ArrowLeft" && i > 0) { e.preventDefault(); reorder(i, i - 1); strip.children[i - 1]?.focus(); }
          if (e.key === "ArrowRight" && i < C.items.length - 1) { e.preventDefault(); reorder(i, i + 1); strip.children[i + 1]?.focus(); }
          if (e.key === "Delete") { e.preventDefault(); remove(i); }
        }}},
      h("img", {src: `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v || "")}`, alt: "", draggable: "false"}),
      h("span.clgnum", {text: String(i + 1)}),
      C.items.length > 2 ? h("button.mini.clgdel", {type: "button", "data-i18n-aria": "clg.remove", "data-i18n-title": "clg.remove",
                                                    on: {click: e => { e.stopPropagation(); remove(i); }}}, "✕") : null)));
    translate(strip);
  }

  // ── 左邊的設定 ──
  const seg = (aria, keys, on, label, click) => h("div.seg", {role: "group", "data-i18n-aria": aria}, keys.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), ...label(k), on: {click: () => { click(k); drawSide(); drawPreview(); }}})));
  const row = (key, ...kids) => h("div.rflab.btrow", {}, h("span.dim", {"data-i18n": key}), h("span.sp"), ...kids);
  const num = (key, val, min, max, set) => h("input.v.btnum", {type: "number", min: String(min), max: String(max), step: "1", value: String(val),
    "data-i18n-aria": key, on: {change: e => {
      const v = Math.max(min, Math.min(max, Math.round(Number(e.target.value) || 0)));
      e.target.value = String(v);
      set(v);
      drawPreview();
    }}});
  const fmt = () => (FORMATS.includes(UI?.eformat) ? UI.eformat : "png");
  const toLib = () => UI?.edest !== "folder";

  function drawSide() {
    const s = cset(), n = C.items.length, free = s.mode === "free";
    const ready = wmReady(wmset(UI), C.logo), q = Number(UI?.equality) || 92;
    if (free && C.items.length && C.items.every(it => C.bmps.has(it.id))) current();   // 剛切到自由排：先排好，畫布大小才有數字
    const selected = free && C.free?.boxes[C.sel];
    side.replaceChildren(...[
      h("div.ahd", {}, h("h2.title", {id: "clgTitle", text: t("clg.title", {n})})),
      picked.length > MAX_ITEMS ? h("p.hint.warn", {text: t("clg.tooMany", {max: MAX_ITEMS})}) : null,
      h("div.clgsec", {},
        row("clg.mode", seg("clg.mode", MODES, k => s.mode === k, k => ({"data-i18n": `clg.mode.${k}`}), k => setC({mode: k}))),
        h("p.hint", {"data-i18n": `clg.modeHint.${s.mode}`}),
        s.mode === "grid" ? row("clg.cols", h("select.chip", {"data-i18n-aria": "clg.cols", on: {change: e => { setC({cols: Number(e.target.value)}); drawPreview(); }}},
          [0, ...Array.from({length: Math.min(n, 12)}, (_, i) => i + 1)].map(c => h("option", {value: String(c), selected: (s.cols || 0) === c,
            text: c ? String(c) : t("clg.auto")})))) : null,
        s.mode === "grid" ? row("clg.cell", h("select.chip", {"data-i18n-aria": "clg.cell", on: {change: e => { setC({cell: e.target.value}); drawPreview(); }}},
          CELLS.map(c => h("option", {value: c, selected: s.cell === c, text: c === "first" ? t("clg.cellFirst") : c})))) : null,
        s.mode === "grid" ? row("clg.fit", seg("clg.fit", ["cover", "contain"], k => s.fit === k, k => ({"data-i18n": `clg.fit.${k}`}), k => setC({fit: k}))) : null,
        free ? row("clg.canvas", num("clg.canvasW", Math.round(C.free?.W || 1), 16, 16384, v => { C.free = {...C.free, W: v}; }),
                   h("span.dim", {text: "×"}), num("clg.canvasH", Math.round(C.free?.H || 1), 16, 16384, v => { C.free = {...C.free, H: v}; })) : null,
        free ? h("div.drow", {},
          h("button.pill", {type: "button", "data-i18n": "clg.fitCanvas", "data-i18n-title": "clg.fitCanvasHint", on: {click: () => {
            C.free = fitFree(C.free.boxes, s.pad); drawSide(); drawPreview();
          }}}),
          h("button.pill", {type: "button", "data-i18n": "clg.rearrange", on: {click: () => {
            C.free = freeStart(sizes(), s); C.sel = -1; drawSide(); drawStrip(); drawPreview();
          }}})) : null,
        free ? (selected
          ? h("div.drow", {},
            h("span.dim.small", {text: t("clg.selected", {i: C.sel + 1})}), h("span.sp"),
            h("button.mini", {type: "button", "data-i18n": "clg.toTop", on: {click: () => reorder(C.sel, C.items.length - 1)}}),
            h("button.mini", {type: "button", "data-i18n": "clg.toBottom", on: {click: () => reorder(C.sel, 0)}}))
          : h("p.dim.small", {"data-i18n": "clg.freeHint"})) : null),
      h("div.clgsec", {},
        free ? null : row("clg.gap", num("clg.gap", s.gap, 0, 400, v => setC({gap: v})), h("span.dim", {text: "px"})),
        row("clg.pad", num("clg.pad", s.pad, 0, 400, v => setC({pad: v})), h("span.dim", {text: "px"})),
        row("clg.radius", num("clg.radius", s.radius, 0, 400, v => setC({radius: v})), h("span.dim", {text: "px"})),
        row("clg.bg",
          h("input.clgcolor", {type: "color", value: s.bg, disabled: s.clear, "data-i18n-aria": "clg.bg",
                               on: {input: e => { setC({bg: e.target.value}); drawPreview(); }}}),
          h("label.egcheck", {}, h("input", {type: "checkbox", checked: s.clear, on: {change: e => { setC({clear: e.target.checked}); drawSide(); drawPreview(); }}}),
            h("span", {"data-i18n": "clg.clear"})))),
      h("div.clgsec", {},
        h("h3.clgh", {"data-i18n": "clg.output"}),
        row("bt.size", seg("bt.size", ["orig", "long"], k => s.size === k, k => ({"data-i18n": `bt.size.${k}`}), k => setC({size: k})),
          s.size === "long" ? num("bt.size", s.long, 16, 8192, v => setC({long: v})) : null, s.size === "long" ? h("span.dim", {text: "px"}) : null),
        h("div.rflab", {}, h("span.sp"), outInfo),
        row("ed.format", seg("ed.format", FORMATS, k => fmt() === k, k => ({text: k.toUpperCase()}), k => saveUi({eformat: k}))),
        fmt() !== "png" ? row("ed.quality", h("input", {type: "range", min: "50", max: "100", step: "1", value: String(q), "data-i18n-aria": "ed.quality",
                                                        on: {change: e => { saveUi({equality: Number(e.target.value)}); drawSide(); }}}),
                             h("span.mono.rfnum", {text: String(q)})) : null,
        fmt() === "jpg" && s.clear ? h("p.hint", {"data-i18n": "clg.jpgClear"}) : null,
        row("ed.dest", seg("ed.dest", ["library", "folder"], k => (k === "library") === toLib(), k => ({"data-i18n": `ed.dest.${k}`}), k => saveUi({edest: k}))),
        toLib() ? null : h("div.newset", {},
          h("input.find", {type: "text", value: UI?.efolder || "", spellcheck: "false", "data-i18n-placeholder": "lib.dest", "data-i18n-aria": "lib.dest",
                           on: {change: e => saveUi({efolder: e.target.value.trim()})}}),
          h("button.pill", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
            const got = await pickFolder(UI?.efolder || "");
            if (got) { saveUi({efolder: got}); drawSide(); }
          }}})),
        h("div.cell.swcell", {},
          h("span.swtext", {}, h("b", {id: "clgWmLbl", "data-i18n": "bt.wm"}), h("span.hint", {"data-i18n": ready ? "bt.wmHint" : "bt.wmNone"})),
          h("button.switch", {type: "button", role: "switch", "aria-labelledby": "clgWmLbl", disabled: !ready, "aria-checked": String(ready && s.wm),
                              on: {click: () => { setC({wm: !s.wm}); drawSide(); drawPreview(); }}}, h("i")))),
      msg,
      h("div.sfoot", {}, h("span.sp"), h("button.pill", {type: "button", "data-i18n": "bt.close", on: {click: close}}), go),
    ].filter(Boolean));
    translate(side);
    go.disabled = C.busy || C.bmps.size < C.items.length;
  }

  go.addEventListener("click", async () => {
    if (C.busy) return;
    if (!toLib() && !(UI?.efolder || "").trim()) { msg.dataset.kind = "err"; msg.textContent = t("ed.needFolder"); return; }
    C.busy = true;
    go.disabled = true;
    msg.dataset.kind = "";
    msg.textContent = t("clg.saving");
    try {
      const L = current(), k = outScale(L.W, L.H, cset());
      const c = document.createElement("canvas");
      render(c, L, k, useWm());
      const d = await post("/api/edit/export", {
        collage: C.items.map(it => it.id), png: await pngOf(c), format: fmt(), quality: Number(UI?.equality) || 92,
        width: c.width, height: c.height, dest: toLib() ? "library" : "folder", folder: UI?.efolder || ""});
      if (d.error) throw new Error(d.error);
      msg.dataset.kind = "ok";
      msg.replaceChildren(t(d.id ? "clg.savedLib" : "clg.saved", {path: d.path}), " ",
        d.id ? h("button.mini", {type: "button", "data-i18n": "clg.edit", on: {click: () => { C.busy = false; close(); openEditor(d.id); }}}) : "");
      translate(msg);
    } catch (e) {
      msg.dataset.kind = "err";
      msg.textContent = t("clg.err", {msg: e.message || e});
    } finally {
      C.busy = false;
      go.disabled = false;
    }
  });

  el.append(h("div.apanel.clgpanel", {}, side, h("div.clgmain", {}, stage, strip)));
  el.addEventListener("click", e => { if (e.target === el) close(); });
  addEventListener("keydown", onKey, true);
  document.body.append(el);
  document.body.classList.add("viewing");
  drawSide();
  drawStrip();

  // 讀圖：一次讀 4 張
  C.logo = await loadLogo().catch(() => null);
  let done = 0, failed = [];
  const queue = items.slice();
  await Promise.all(Array.from({length: 4}, async () => {
    while (queue.length) {
      const it = queue.shift();
      try {
        C.bmps.set(it.id, await createImageBitmap(await getBlob(`/api/full?id=${it.id}&v=${encodeURIComponent(it.v || "")}`)));
      } catch {
        failed.push(it);
      }
      done++;
      stage.firstChild.textContent = t("clg.loading", {i: done, n: items.length});
    }
  }));
  if (!el.isConnected) { C.bmps.forEach(b => b.close?.()); return; }
  if (failed.length) C.items = C.items.filter(it => C.bmps.has(it.id));
  if (C.items.length < 2) { stage.replaceChildren(h("p.hint.warn", {text: t("clg.loadErr")})); return; }
  if (failed.length) { msg.dataset.kind = "warn"; msg.textContent = t("clg.someFailed", {n: failed.length}); }
  stage.replaceChildren(preview);
  drawSide();
  drawStrip();
  drawPreview();
}
