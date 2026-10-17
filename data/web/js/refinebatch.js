// 批次精修（使用者筆記：不用填提示詞的精修，一次處理好幾張）：圖庫選好幾張 →「精修」→ 選方式
//   修臉：每張自動找臉、找到的都修（沒有臉的略過）；高清化：放大，可以一起補細節；去背
// 修臉與補細節要重畫：每張照原圖的提示詞、模型與參數（跟精修頁一樣，也可以切成照生成台），
// 一起排進生成佇列（/api/refine/batch，原圖的模型不一樣時一組做完換下一組），進度在生成台看。
// 只放大、去背不重畫：在這裡一張一張做（跟精修頁同一個 API），可以中途停。
// 結果都存成各張圖的新版本，精修頁的版本列看得到。設定跟精修頁共用同一組（改這裡，精修頁也跟著變）

import {get, post} from "./api.js";
import {h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {LEVELS, withLoras} from "./refine.js";
import {sourceParams} from "./options.js";
import {submitBatch} from "./generate.js";

const MODES = ["face", "upscale", "rembg"];
let UI = null, onNew = null;

export function initRefineBatch(ui, newImages) { UI = ui; onNew = newImages; }

const saveUi = patch => { Object.assign(UI, patch); post("/api/ui", patch).catch(() => { /* 只影響下次 */ }); };
const mode = () => (MODES.includes(UI?.rbatch) ? UI.rbatch : "face");
const useSrc = () => UI?.rsrcp !== false;
const faceStrength = () => Math.min(1, Math.max(0.05, Number(UI?.rfstrength) || 0.4));
const upStrength = () => Math.min(1, Math.max(0.05, Number(UI?.rustrength) || 0.3));
const upDetail = () => UI?.rupdetail !== false;
const rbEdge = () => (UI?.rbedge === "sharp" ? "sharp" : "soft");
const snap = v => Math.round(Number(v) / 16) * 16;

export async function openRefineBatch(items) {
  if (!items.length) return;
  const back = document.activeElement;
  const el = h("div.assets.confirm", {role: "dialog", "aria-modal": "true", "aria-labelledby": "rbtTitle"});
  const panel = h("div.apanel.cfpanel.edbatch");
  const msg = h("p.hint.edmsg", {role: "status"});
  const go = h("button.pill.primary", {type: "button"});
  const stop = h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: () => close()}});
  let running = false, cancel = false, finished = false;
  // 放大模型、去背模型的清單（跟精修頁同一個 API）
  const [ups, rb] = await Promise.all([get("/api/refine/upscalers").catch(() => null), get("/api/rembg/status").catch(() => null)]);
  const U = ups || {items: [], scales: [1, 1.5, 2, 3, 4], max: 4096};
  const rbModels = rb?.models || [];

  const upScale = () => (U.scales.includes(Number(UI?.rupscale)) ? Number(UI.rupscale) : 2);
  const upModel = () => {
    const names = U.items.map(m => m.name);
    if (UI?.rupmodel === "") return "";
    return names.includes(UI?.rupmodel) ? UI.rupmodel : names[0] || "";
  };
  const rbModel = () => rbModels.find(m => m.id === UI?.rbmodel) || rbModels[0];
  const rbPrep = m => (m.arch === "isnet" && ["anime", "dis"].includes(UI?.rbprep?.[m.id]) ? UI.rbprep[m.id] : m.prep);
  // 放大後超過上限的圖：這些會略過
  const tooBig = () => items.filter(it => Math.max(snap(it.width * upScale()), snap(it.height * upScale())) > U.max);
  const redraws = () => mode() === "face" || (mode() === "upscale" && upDetail());
  // 1 倍只放大又沒選放大模型：圖不會變（跟精修頁一樣停用）
  const upNoop = () => mode() === "upscale" && !upDetail() && upScale() === 1 && !upModel();

  function close() {
    if (running) { cancel = true; return; }
    removeEventListener("keydown", onKey, true);
    el.remove();
    document.body.classList.remove("viewing");
    back?.focus?.();
  }
  const onKey = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); } };

  const seg = (aria, list, on, label, click) => h("div.seg", {role: "group", "data-i18n-aria": aria}, list.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), ...label(k), on: {click: () => { click(k); draw(); }}})));
  const row = (key, ...kids) => h("div.rflab.btrow", {}, h("span.dim", {"data-i18n": key}), h("span.sp"), ...kids);
  const levelRow = (lv, cur, key) => row("rbt.strength", seg("rbt.strength", lv.map(([k]) => k), k => Math.abs(lv.find(x => x[0] === k)[1] - cur) < 0.001,
    k => ({"data-i18n": `ref.level.${k}`}), k => saveUi({[key]: lv.find(x => x[0] === k)[1]})));

  function body() {
    const md = mode();
    if (md === "face") return [
      levelRow(LEVELS.face, faceStrength(), "rfstrength"),
      h("p.hint", {"data-i18n": "rbt.faceHint"})];
    if (md === "upscale") {
      const big = tooBig();
      return [
        row("ref.up.model", h("select.rfrbsel", {"data-i18n-aria": "ref.up.model", on: {change: e => { saveUi({rupmodel: e.target.value}); draw(); }}},
          ...U.items.map(m => h("option", {value: m.name, selected: upModel() === m.name ? "" : null, text: m.name})),
          h("option", {value: "", selected: upModel() === "" ? "" : null, "data-i18n": "ref.up.none"}))),
        row("ref.up.scale", seg("ref.up.scale", U.scales, k => upScale() === k, k => ({text: `${k}×`}), k => saveUi({rupscale: k}))),
        big.length ? h("p.hint", {text: t("rbt.tooBig", {n: big.length, max: U.max})}) : null,
        upNoop() ? h("p.hint", {"data-i18n": "ref.up.needChange"}) : null,
        h("div.cell.swcell", {},
          h("span.swtext", {}, h("b", {id: "rbtDetailLbl", "data-i18n": "ref.up.detail"}),
            h("span.hint", {"data-i18n": upDetail() ? "ref.up.detailOn" : "ref.up.detailOff"})),
          h("button.switch", {type: "button", role: "switch", "aria-labelledby": "rbtDetailLbl", "aria-checked": String(upDetail()),
                              on: {click: () => { saveUi({rupdetail: !upDetail()}); draw(); }}}, h("i"))),
        upDetail() ? levelRow(LEVELS.upscale, upStrength(), "rustrength") : null];
    }
    const m = rbModel();
    if (!m) return [h("p.hint", {"data-i18n": "rbt.noRembg"})];
    return [
      row("ref.rb.model", h("select.rfrbsel", {"data-i18n-aria": "ref.rb.model", on: {change: e => { saveUi({rbmodel: e.target.value}); draw(); }}},
        rbModels.map(x => h("option", {value: x.id, selected: x.id === m.id ? "" : null, text: x.known ? t(`ref.rb.${x.known}`) : x.name})))),
      row("ref.rb.edge", seg("ref.rb.edge", ["soft", "sharp"], k => rbEdge() === k, k => ({"data-i18n": `ref.rb.edge.${k}`}), k => saveUi({rbedge: k})))];
  }

  function draw() {
    const md = mode();
    const ready = md !== "rembg" || !!rbModel();
    panel.replaceChildren(...[
      h("h2.title", {id: "rbtTitle", text: t("rbt.title", {n: items.length})}),
      row("rbt.mode", seg("rbt.mode", MODES, k => md === k, k => ({"data-i18n": `ref.mode.${k}`}), k => saveUi({rbatch: k}))),
      ...body(),
      redraws() ? row("ref.src.which", seg("ref.src.which", ["src", "gen"], k => useSrc() === (k === "src"), k => ({"data-i18n": `ref.src.${k}`}),
                                           k => saveUi({rsrcp: k === "src"}))) : null,
      redraws() ? h("p.hint", {"data-i18n": "rbt.queueHint"}) : null,
      msg,
      h("div.sfoot", {}, h("span.sp"), stop, go)].filter(Boolean));
    go.textContent = t(redraws() ? "rbt.queue" : "rbt.start", {n: items.length});
    translate(panel);
    if (running || finished) for (const b of panel.querySelectorAll("button, select")) if (b !== stop) b.disabled = true;
    go.disabled = running || finished || !ready || upNoop();
    stop.disabled = false;
  }

  // 修臉、補細節：每張的送出內容跟精修頁一樣（照原圖的提示詞含 LoRA、模型與參數），一起排進佇列
  async function queue() {
    const ids = items.map(it => it.id).join(",");
    const [d, res] = await Promise.all([get(`/api/refine/items?ids=${ids}`), useSrc() ? get("/api/res?kind=model").catch(() => ({items: []})) : null]);
    const face = mode() === "face";
    const sends = (d.items || []).map(src => ({
      prompt: withLoras(src), negative: src.user_negative ?? src.negative ?? "", init_id: src.id, width: null, height: null,
      ...(face ? {strength: faceStrength(), faces: "auto", mask_blur: 8}
               : {strength: upStrength(), upscale: {model: upModel(), scale: upScale()}, mask_blur: 32}),
      ...(useSrc() ? (sourceParams(src, res?.items || []).over || {}) : {}),
    }));
    const r = await submitBatch(sends);
    const names = new Map(items.map(it => [it.id, it.filename || String(it.id)]));
    const skipped = (r.skipped || []).map(s => `${names.get(Number(s.id)) || s.id}：${s.error}`);
    if (r.error) { msg.dataset.kind = "err"; msg.replaceChildren(r.error, ...skipped.slice(0, 5).map(x => h("span.bterr", {text: x}))); return false; }
    msg.dataset.kind = skipped.length || r.warn ? "warn" : "ok";
    msg.replaceChildren(t("rbt.queued", {n: r.queued}), " ", h("a", {href: "#generate", "data-i18n": "rbt.seeProgress", on: {click: () => close()}}),
      r.warn ? h("span.bterr", {text: r.warn}) : "", ...skipped.slice(0, 5).map(x => h("span.bterr", {text: x})));
    translate(msg);
    return true;
  }

  // 只放大、去背：一張一張做，可以中途停
  async function direct() {
    const errs = [];
    let done = 0;
    const md = mode(), m = rbModel();
    for (const it of items) {
      if (cancel) break;
      msg.dataset.kind = "";
      msg.textContent = t("rbt.progress", {i: done + errs.length + 1, n: items.length});
      try {
        const d = md === "rembg"
          ? await post("/api/refine/rembg", {id: it.id, model: m.id, edge: rbEdge(), prep: rbPrep(m)})
          : await post("/api/refine/upscale", {id: it.id, model: upModel(), scale: upScale()});
        if (d.error) throw new Error(d.error);
        done++;
      } catch (e) {
        errs.push(`${it.filename || it.id}：${e.message || e}`);
      }
    }
    onNew?.();
    msg.dataset.kind = errs.length ? "err" : "ok";
    msg.replaceChildren(t(cancel ? "rbt.stopped" : "rbt.done", {n: done}), ...errs.slice(0, 5).map(x => h("span.bterr", {text: x})));
    return true;
  }

  go.addEventListener("click", async () => {
    if (running || finished) return;
    running = true;
    cancel = false;
    const viaQueue = redraws();
    stop.dataset.i18n = viaQueue ? "bt.close" : "bt.stop";
    msg.dataset.kind = "";
    msg.textContent = t(viaQueue ? "rbt.preparing" : "rbt.progress", {i: 1, n: items.length});
    draw();
    let ok = false;
    try {
      ok = await (viaQueue ? queue() : direct());
    } catch (e) {
      msg.dataset.kind = "err";
      msg.textContent = e.message || String(e);
    }
    running = false;
    finished = ok;
    stop.dataset.i18n = "bt.close";
    draw();
  });

  draw();
  el.addEventListener("click", e => { if (e.target === el && !running) close(); });
  el.append(panel);
  addEventListener("keydown", onKey, true);
  document.body.append(el);
  document.body.classList.add("viewing");
  go.focus();
}
