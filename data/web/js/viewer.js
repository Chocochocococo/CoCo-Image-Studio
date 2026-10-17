// 看大圖：蓋在畫面上的暫時視窗（不是固定的側欄）。左邊大圖、右邊參數與動作。
// ←／→ 換張（到最後一張會自動載入下一批），Esc 關閉，點圖在「符合視窗」與「原始大小」之間切換。

import {get, post} from "./api.js";
import {$, h, stem} from "./dom.js";
import {applyParams, genSettings} from "./generate.js";
import {lang, t, translate} from "./i18n.js";
import {addToCompare, openCompare} from "./compare.js";
import {openRefine} from "./refine.js";
import {openExplore, pinterestQuery} from "./explore.js";
import {openEditor} from "./editor.js";
import {saveFromImagePop} from "./recipes.js";
import {closePop, openPop} from "./pop.js";
import {addReference} from "./cnpanel.js";
import {fmtWeight as fmtW, keepLoras, mapSampler, renameLora, shiftKey, withLoras} from "./options.js";
import {popoutOpener} from "./winrole.js";
import {Wall} from "./wall.js";
import {movePop, moveText, removeFiles, trashClick, trashLabel, trashText} from "./fileacts.js";
import {chipView} from "./chips.js";

const V = {wall: null, idx: -1, d: null, el: null, back: null, seq: 0, ui: null};

// 介面偏好（main.js 啟動時給）：提示詞要不要用標籤模式顯示（ui.vchips）
export function initViewer(ui) { V.ui = ui; }

// 圖的網址：圖庫的圖用 id；探索找到的參考圖（kind: "ref"，rid＝參考圖的 id）走探索的 API
const urls = it => (it.kind === "ref"
  ? {thumb: `/api/explore/ref/thumb?id=${it.rid}`, full: `/api/explore/ref/full?id=${it.rid}`, detail: `/api/explore/ref/detail?id=${it.rid}`}
  : {thumb: `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v)}`, full: `/api/full?id=${it.id}&v=${encodeURIComponent(it.v)}`,
     detail: `/api/detail?id=${it.id}`});

// 參考圖要精修、編輯、放進比較：先加進圖庫（複製一份到 outputs/imports，同一張只複製一次），再用圖庫的 id
async function libId(d, btn) {
  if (!d.ref) return d.id;
  if (d.libId) return d.libId;
  const label = btn?.textContent;
  if (btn) { btn.disabled = true; btn.textContent = t("view.importing"); }
  const r = await post("/api/explore/ref/import", {id: d.ref.rid}).catch(e => ({error: e.message}));
  if (btn) { btn.disabled = false; btn.textContent = label; }
  if (r.error) { $("vMsg").textContent = r.error; return null; }
  d.libId = r.id;
  return r.id;
}


export function openViewer(wall, id) {
  closePop();
  V.wall = wall;
  V.back = document.activeElement;
  if (!V.el) build();
  V.el.hidden = false;
  document.body.classList.add("viewing");
  show(wall.items.findIndex(i => i.id === id));
}

export function closeViewer() {
  if (!V.el || V.el.hidden) return;
  V.el.hidden = true;
  document.body.classList.remove("viewing");
  // 焦點還給原本那格圖（換過張就還給現在這張）
  const it = V.wall?.items[V.idx];
  const node = it && V.wall.nodes.get(it.id);
  (node?.isConnected ? node : V.back)?.focus?.();
}

export const viewerOpen = () => !!V.el && !V.el.hidden;

function build() {
  V.el = h("div.viewer", {role: "dialog", "aria-modal": "true", "aria-labelledby": "vName", hidden: true},
    // 捲動只發生在 vscroll 裡；箭頭、張數、提示放在外層 vstage，原始大小時不會跟著捲走
    h("div.vstage", {id: "vStage"},
      h("div.vscroll", {id: "vScroll"}, h("img", {id: "vImg", alt: "", on: {click: toggleFit}})),
      h("p.vmsg", {id: "vMsg"}),
      h("button.vnav.prev", {type: "button", "data-i18n-aria": "view.prev", "data-i18n-title": "view.prev", on: {click: () => step(-1)}}, "‹"),
      h("button.vnav.next", {type: "button", "data-i18n-aria": "view.next", "data-i18n-title": "view.next", on: {click: () => step(1)}}, "›"),
      h("span.vpos.mono", {id: "vPos"})),
    h("aside.vpanel", {id: "vPanel"}));
  // 點背景（圖以外的地方）關閉
  V.el.addEventListener("click", e => {
    if (e.target === V.el || e.target.id === "vStage" || e.target.id === "vScroll") closeViewer();
  });
  document.body.append(V.el);
  translate(V.el);
}

async function step(d) {
  const w = V.wall;
  let i = V.idx + d;
  if (i >= w.items.length && w.offset < w.total) {
    await w.loadMore();
  }
  if (i < 0 || i >= w.items.length) return;
  show(i);
}

async function show(i) {
  const w = V.wall, it = w.items[i];
  if (!it) return;
  V.idx = i;
  const seq = ++V.seq;
  const img = $("vImg");
  V.actual = false;
  $("vMsg").textContent = "";
  // 先放縮圖（已經在快取裡），原圖載好再換上。尺寸用原圖的，縮圖先被放大顯示、不會跳動
  const u = urls(it);
  img.src = u.thumb;
  fit();
  const full = new Image();
  full.onload = () => { if (seq === V.seq) img.src = full.src; };
  full.src = u.full;
  $("vPos").textContent = `${i + 1} / ${w.total.toLocaleString(lang())}`;
  V.el.querySelector(".prev").disabled = i === 0;
  V.el.querySelector(".next").disabled = i >= w.total - 1;
  const panel = $("vPanel");
  panel.replaceChildren(h("p.dim", {"data-i18n": "view.loading"}));
  translate(panel);
  try {
    const d = await get(u.detail);
    if (seq !== V.seq) return;
    if (d.error) throw new Error(d.error);
    V.d = d;
    if (!d.exists) $("vMsg").textContent = t("view.missing");
    panel.replaceChildren(...detailPanel(d, it).filter(Boolean));   // 沒有的區塊是 null，不能直接放進去
    translate(panel);
    panel.querySelector(".vclose")?.focus();
  } catch (e) {
    if (seq === V.seq) panel.replaceChildren(h("p.dim", {text: t("view.err", {msg: e.message || e})}));
  }
  // 預先載下一張的原圖，按 → 時比較快
  const next = w.items[i + 1];
  if (next) new Image().src = urls(next).full;
}

// 依可見範圍算圖要多大。不靠 CSS 的 max-height: 100%——那要父層高度是「確定的」
// 才會生效，放在 grid 裡常常失效，圖就用原始尺寸撐開，下半部被裁掉。
// 符合視窗：整張放得下、不超過原圖大小；原始大小：用原圖像素，超出的部分在 vscroll 裡捲。
function fit() {
  const it = V.wall?.items[V.idx], img = $("vImg"), box = $("vScroll");
  if (!it || !img) return;
  const iw = it.width || img.naturalWidth || 1, ih = it.height || img.naturalHeight || 1;
  const k = V.actual ? 1 : Math.min(1, box.clientWidth / iw, box.clientHeight / ih);
  img.style.width = `${Math.round(iw * k)}px`;
  img.style.height = `${Math.round(ih * k)}px`;
  $("vStage").classList.toggle("actual", !!V.actual);
}

function toggleFit(e) {
  const img = $("vImg"), box = $("vScroll");
  // 放大時把點的那一點捲到視窗中間
  const r = img.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  V.actual = !V.actual;
  fit();
  if (V.actual) {
    box.scrollLeft = fx * img.offsetWidth - box.clientWidth / 2;
    box.scrollTop = fy * img.offsetHeight - box.clientHeight / 2;
  }
}
addEventListener("resize", () => { if (viewerOpen()) fit(); });

// ── 右邊的參數欄 ──
// 參數都是唯讀的，最常做的事是複製：每一格點一下就複製那一格的內容（使用者筆記）。
// 有選取文字時不複製整格，才能只選一段自己複製。text 可以是函式：按下去時才決定（例如照設定要不要連 LoRA）
export function copyable(el, text) {
  if (text == null || text === "") return el;
  el.classList.add("cp");
  el.tabIndex = 0;
  el.setAttribute("role", "button");
  el.title = [el.title, t("view.clickCopy")].filter(Boolean).join("\n");
  const go = async () => {
    try { await navigator.clipboard.writeText(String(typeof text === "function" ? text() : text)); el.dataset.copied = t("view.copied"); }
    catch { el.dataset.copied = t("view.copyFail"); }
    clearTimeout(el.cpTimer);
    el.cpTimer = setTimeout(() => { delete el.dataset.copied; }, 1200);
  };
  el.addEventListener("click", () => { if (!String(getSelection() || "").trim()) go(); });
  el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  return el;
}

// 提示詞的「標籤」開關：記在介面偏好（下次打開看大圖照舊），整個右欄照同一張重畫
function toggleChips(it) {
  if (!V.ui || !V.d) return;
  V.ui.vchips = !V.ui.vchips;
  post("/api/ui", {vchips: V.ui.vchips}).catch(() => { /* 只影響下次 */ });
  const panel = $("vPanel"), top = panel.scrollTop;
  panel.replaceChildren(...detailPanel(V.d, it).filter(Boolean));
  translate(panel);
  panel.scrollTop = top;
  panel.querySelector(".vchipbtn")?.focus();
}

function detailPanel(d, it) {
  const extra = d.extra || {};
  const shift = extra.Shift ?? extra.shift;
  const size = d.filesize ? `${(d.filesize / 2 ** 20).toFixed(1)} MB` : "";
  const date = d.mtime ? new Date(d.mtime * 1000).toLocaleString(lang(), {dateStyle: "medium", timeStyle: "short"}) : "";
  const kv = (label, value, raw) => value == null || value === "" ? null
    : copyable(h("div.kv", {}, h("span.k", raw ? {text: label} : {"data-i18n": label}), h("span.v", {text: String(value)})), value);
  const stars = h("div.vstars", {role: "group", "data-i18n-aria": "view.rating"}, [1, 2, 3, 4, 5].map(r => h("button", {
    type: "button", "aria-pressed": String((d.rating || 0) >= r), title: t("lib.stars", {n: r}),
    on: {click: () => rate(d, it, d.rating === r ? 0 : r)}}, "★")));
  // img2img 的底圖與強度另外排成一行，不在「其他參數」裡重複列（init 是完整路徑，很長）
  const strength = extra.strength ?? extra["Denoising strength"];
  const from = extra.init ? String(extra.init).split(/[\\/]/).pop() : "";
  // 品質標籤（server/presets.py）：開關與自動加上的部分另外排，使用者寫的那份只給送到生成區用
  const others = Object.entries(extra).filter(([k]) =>
    !/^(shift|conditioning|init|strength|denoising strength|preset|preset_quality|preset_negative|user_prompt|user_negative)$/i.test(k));
  const added = [extra.preset_quality, extra.preset_negative].some(Boolean);
  const ref = d.ref;                       // 探索找到的參考圖（不在圖庫裡）
  const params = !!(d.prompt || d.model);
  const lib = fn => async e => { const id = await libId(d, e.currentTarget); if (id != null) fn(id, e.currentTarget); };
  // 搬移、丟進回收桶：只有從兩道圖庫的牆（工作室、圖庫空間）打開時有——比較、探索的清單不是圖庫的牆，丟掉之後更新不了
  // （使用者回報 2026-10-10：看大圖沒有這兩個）
  const files = !ref && V.wall instanceof Wall;
  // 提示詞用標籤模式顯示：一個標籤一格、有中文翻譯，點一下複製那個標籤；「複製全部」複製整段
  const chips = !!V.ui?.vchips;
  // 設定開了「複製提示詞時連 LoRA 一起複製」：補上這張圖的 <lora:…>（CoCo 存的提示詞裡沒有）
  const promptCopy = () => genSettings()?.COPY_WITH_LORAS === true ? withLoras(d.prompt, imageLoras(d)) : d.prompt;
  return [
    h("div.vhd", {},
      h("h2.title", {id: "vName", text: d.filename, title: d.filename}),
      h("button.round.vclose", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close",
                                on: {click: closeViewer}}, "✕")),
    h("p.dim.vmeta", {text: [`${d.width} × ${d.height}`, size, date].filter(Boolean).join(" · ")}),
    ref ? null : stars,
    h("div.vacts", {}, ...[
      !ref || params ? h("button.pill.primary", {type: "button", "data-i18n": "view.send",
                                                 on: {click: e => sendToGen(d, e.currentTarget)}}) : null,
      // 只送正、負面提示詞（使用者要求 2026-10-12）：模型、尺寸、seed 這些不動
      d.prompt ? h("button.pill", {type: "button", "data-i18n": "view.sendPrompt",
                                   on: {click: e => sendToGen(d, e.currentTarget, true)}}) : null,
      ref ? null : h("button.pill", {type: "button", "data-i18n": "lib.reveal", on: {click: () => post("/api/reveal", {id: d.id})}}),
      h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.toCompare",
                        "data-i18n-title": ref ? "view.importHint" : null,
                        on: {click: lib((id, btn) => {
                          addToCompare([id], btn, r => {
                            if (r.error) return;
                            // 換成「去比較」：關掉大圖、切到比較頁
                            btn.replaceWith(h("button.pill", {type: "button", on: {click: () => { closeViewer(); openCompare(r.id); }}},
                              t("cmp.addedGo")));
                          });
                        })}}),
      params ? h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false",
                                              "data-i18n": "rcp.saveImage", "data-i18n-title": "rcp.saveImageHint",
                                              on: {click: e => {
                                                const btn = e.currentTarget;
                                                saveFromImagePop(btn, d, () => { btn.textContent = t("rcp.savedShort"); btn.disabled = true; });
                                              }}}) : null,
      // ControlNet：這張當生成台的參考圖（探索的參考圖不用先加進圖庫）
      h("button.pill", {type: "button", "data-i18n": "cn.asRef", "data-i18n-title": "cn.asRefHint", on: {click: () => {
        const src = ref ? {kind: "ref", rid: ref.rid} : {kind: "lib", id: d.id, v: d.mtime};
        closeViewer();
        location.hash = "#generate";
        addReference(src);
      }}}),
      h("button.pill", {type: "button", "data-i18n": "lib.refine", "data-i18n-title": ref ? "view.importHint" : "ref.fromViewerHint",
                        on: {click: lib(id => { closeViewer(); openRefine(id); })}}),
      h("button.pill", {type: "button", "data-i18n": "lib.edit", "data-i18n-title": ref ? "view.importHint" : "ed.fromHint",
                        on: {click: lib(id => { closeViewer(); openEditor(id); })}}),
      h("button.pill", {type: "button", "data-i18n": "lib.similar", "data-i18n-title": "x.fromHint",
                        on: {click: () => {
                          closeViewer();
                          openExplore(ref ? {kind: "ref", key: ref.key, id: ref.rid, name: ref.name, tags: ref.tags, url: ref.url, eagle: ref.eagle}
                                          : {kind: "lib", id: d.id, filename: d.filename, v: d.v, prompt: d.extra?.user_prompt ?? d.prompt});
                        }}}),
      h("a.pill", {href: `https://www.pinterest.com/search/pins/?q=${encodeURIComponent(pinterestQuery(
        ref ? {kind: "ref", tags: ref.tags, name: ref.name} : {kind: "lib", prompt: d.extra?.user_prompt ?? d.prompt}))}`,
                   target: "_blank", rel: "noopener", "data-i18n": "x.pinterest", "data-i18n-title": "x.pinterestHint"}),
      ref?.eagle ? h("a.pill", {href: `eagle://item/${ref.eagle}`, "data-i18n": "x.openEagle"})
        : h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "eg.save",
                            "data-i18n-title": "eg.saveHint", on: {click: e => openPop("popEagle", e.currentTarget, () => eaglePop(d))}}),
      ref?.url ? h("a.pill", {href: ref.url, target: "_blank", rel: "noopener", "data-i18n": "x.source"}) : null,
      files ? h("button.pill", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.move",
                                on: {click: e => openPop("popMove", e.currentTarget, () => movePop([d.id], V.wall.ui, moved))}}) : null,
      files ? h("button.pill.vtrash", {type: "button", "data-i18n": trashLabel(),
                                       on: {click: e => trashClick(e.currentTarget, [d.id], trashed)}}) : null,
    ].filter(Boolean)),
    h("p.vnote", {id: "vNote", role: "status"}),
    // 參考圖：Eagle 的標籤（點一下複製）
    ref?.tags?.length ? h("section.vsec", {},
      h("div.vsh", {}, h("h3", {"data-i18n": "view.tags"})),
      h("div.vres", {}, ref.tags.map(tg => copyable(h("span.rchip", {text: tg}), tg)))) : null,
    h("section.vsec", {},
      h("div.vres", {},
        d.model ? copyable(h("span.rchip", {title: d.model}, h("b", {"data-i18n": "view.model"}), stem(d.model)), stem(d.model)) : null,
        (d.loras || []).map(l => copyable(h("span.rchip.l", {title: l.file}, h("b", {text: "LoRA"}), `${stem(l.file)} · ${fmtW(l.weight)}`),
                                          `<lora:${stem(l.file)}:${fmtW(l.weight)}>`)))),
    d.prompt ? h("section.vsec", {},
      h("div.vsh", {}, h("h3", {"data-i18n": "view.prompt"}), h("span.sp"),
        chips ? copyable(h("span.mini.vcopy", {"data-i18n": "view.copyAll"}), promptCopy) : null,
        h("button.mini.tog.vchipbtn", {type: "button", "aria-pressed": String(chips), "data-i18n": "view.chips",
                                   "data-i18n-title": "view.chipsHint", on: {click: () => toggleChips(it)}})),
      chips ? chipView(d.prompt, copyable) : copyable(h("p.vtext", {text: d.prompt}), promptCopy),
      added ? h("p.vadded", {},
        h("b", {"data-i18n": "view.presetAdded"}), " ",
        [extra.preset_quality && t("view.presetQ", {tags: extra.preset_quality}),
         extra.preset_negative && t("view.presetN", {tags: extra.preset_negative})].filter(Boolean).join(" · ")) : null)
      : h("p.dim", {"data-i18n": "view.noParams"}),
    d.negative ? h("details.vsec", {},
      h("summary", {}, h("h3", {"data-i18n": "view.negative"})),
      chips ? chipView(d.negative, copyable) : copyable(h("p.vtext", {text: d.negative}), d.negative)) : null,
    h("section.vsec.vgrid", {},
      kv("view.steps", d.steps), kv("CFG", d.cfg, true),
      kv("view.sampler", d.sampler), kv("view.schedule", d.scheduler),
      kv("view.seed", d.seed), kv("CLIP skip", d.clip_skip, true),
      kv("shift", shift, true),
      kv("view.refinedFrom", from || null), kv("ref.amount", strength != null ? Number(strength).toFixed(2) : null),
      extra.preset ? kv("view.preset", t(extra.preset === "on" ? "view.presetOn" : "view.presetOff")) : null),
    others.length ? h("details.vsec", {},
      h("summary", {}, h("h3", {text: t("view.extra", {n: others.length})})),
      h("div.vgrid.vextra", {}, others.map(([k, v]) => kv(k, typeof v === "string" ? v : JSON.stringify(v), true)))) : null,
    copyable(h("p.vpath", {text: d.path}), d.path)];
}

// ── 搬移、丟進回收桶（fileacts.js，跟動作列同一套）──
// 結果寫在動作下面；篩選清單與另一道牆的更新交給 library.js（coco:files 事件）
function vnote(text, kind) {
  const n = $("vNote");
  if (!n) return;
  n.textContent = text;
  n.dataset.kind = kind;
}

// 搬走：這張還在圖庫裡（路徑換了），重讀參數欄
async function moved(d, mode) {
  if (!d.error && mode === "move" && d.done) {
    dispatchEvent(new CustomEvent("coco:files", {detail: {wall: V.wall}}));
    await show(V.idx);
  }
  vnote(moveText(d, mode), d.error || d.failed ? "err" : "ok");
}

// 丟掉：從牆上拿掉、換成下一張（最後一張就換前一張），牆空了就關掉
async function trashed(ids, mode) {
  const d = await removeFiles(ids, mode);
  if (d.error || !d.done) { vnote(trashText(d, mode), "err"); return; }
  const w = V.wall, i = V.idx;
  w.remove(ids);
  dispatchEvent(new CustomEvent("coco:files", {detail: {wall: w}}));
  if (i >= w.items.length && w.offset < w.total) await w.loadMore();
  if (!w.items.length) {
    closeViewer();
    const c = $("libNote");             // 牆標題旁（兩道牆都有）
    if (c) c.textContent = trashText(d, mode);
    return;
  }
  await show(Math.min(i, w.items.length - 1));
  vnote(trashText(d, mode), d.failed ? "err" : "ok");
}

// ── 存回 Eagle（server/eagle.py）──
// 存進 Eagle「目前開著的檔案庫」（Eagle 的 API 只能這樣），所以先說是哪一個；資料夾、標籤、要不要把生成參數寫進註解，
// 每個檔案庫各自記上次選的資料夾。存過的再存先問一次
function eaglePop(d) {
  const box = h("div.egbox", {}, h("p.dim", {"data-i18n": "view.loading"}));
  const fill = async () => {
    const [s, ui, xs, cfg] = await Promise.all([get("/api/eagle/status").catch(() => ({running: false})), get("/api/ui").catch(() => ({})),
                                                get("/api/explore/status").catch(() => ({})), get("/api/settings").catch(() => ({}))]);
    if (!s.running || s.error) {
      box.replaceChildren(h("p.hint.warn", {text: s.error || t("eg.notRunning")}),
        h("button.pill", {type: "button", "data-i18n": "eg.retry", on: {click: () => { box.replaceChildren(h("p.dim", {"data-i18n": "view.loading"})); translate(box); fill(); }}}));
      translate(box);
      return;
    }
    const lib = s.library;
    // 資料夾可能有好幾千個（實測 8,409）：不用下拉選單，改成打字搜尋。還沒打字時先列常用的——
    // 上次存的、參考圖來源勾的那幾個（跟它們底下一層）
    const flat = [], byId = new Map();
    const walk = (fs, path, parent) => fs.forEach(f => {
      const it = {id: f.id, name: f.name, path: path ? `${path} / ${f.name}` : f.name, parent, kids: f.children.map(c => c.id)};
      flat.push(it);
      byId.set(f.id, it);
      walk(f.children, it.path, f.id);
    });
    walk(s.folders, "", null);
    const src = (xs.sources || []).find(x => x.kind === "eagle" && x.path.toLowerCase() === String(lib.path).toLowerCase());
    const common = [...new Set([(ui.eaglefolder || {})[lib.path], ...(src?.folders || []).flatMap(id => [id, ...(byId.get(id)?.kids || [])])]
      .filter(id => id && byId.has(id)))].slice(0, 40).map(id => byId.get(id));
    let chosen = byId.get((ui.eaglefolder || {})[lib.path]) || null;
    const shown = h("b.egchosen");
    const find = h("input.find", {type: "search", spellcheck: "false", "data-i18n-placeholder": "eg.findPh", "data-i18n-aria": "eg.findPh",
                                  on: {input: () => list()}});
    const results = h("div.eglist", {role: "listbox", "data-i18n-aria": "eg.folder"});
    const showChosen = () => { shown.textContent = chosen ? chosen.path : t("eg.noFolder"); };
    const list = () => {
      const q = find.value.trim().toLowerCase();
      const hits = q ? flat.filter(f => f.name.toLowerCase().includes(q)).slice(0, 40) : common;
      results.replaceChildren(...[
        q ? null : h("button.egitem", {type: "button", "aria-selected": String(!chosen), on: {click: () => { chosen = null; showChosen(); list(); }}},
          h("span", {"data-i18n": "eg.noFolder"})),
        ...hits.map(f => h("button.egitem", {type: "button", "aria-selected": String(chosen?.id === f.id), title: f.path,
          on: {click: () => { chosen = f; showChosen(); list(); }}}, h("span", {text: f.name}), h("span.dim", {text: f.path}))),
        q && !hits.length ? h("p.dim.small", {"data-i18n": "eg.noMatch"}) : null,
      ].filter(Boolean));
      translate(results);
    };
    showChosen();
    list();
    // 標籤：先放 ⚙ 設定裡的預設標籤；標籤組合點一下加進去，再點一下拿掉
    const split = v => v.split(/[,，]/).map(x => x.trim()).filter(Boolean);
    const tags = h("input.find", {type: "text", value: cfg.EAGLE_TAGS ?? "CoCo", spellcheck: "false", "data-i18n-placeholder": "eg.tagsPh",
                                  "data-i18n-aria": "eg.tags", on: {input: () => markSets()}});
    const sets = cfg.EAGLE_TAG_SETS || [];
    const has = () => new Set(split(tags.value).map(x => x.toLowerCase()));
    const setBtns = sets.map(set => h("button.mini.tog", {type: "button", "aria-pressed": "false", title: set.tags.join(", "),
      text: set.name || set.tags.join(", "), on: {click: () => {
        const cur = split(tags.value), low = has(), mine = new Set(set.tags.map(x => x.toLowerCase()));
        tags.value = (set.tags.every(x => low.has(x.toLowerCase())) ? cur.filter(x => !mine.has(x.toLowerCase()))
          : [...cur, ...set.tags.filter(x => !low.has(x.toLowerCase()))]).join(", ");
        markSets();
      }}}));
    const markSets = () => { const low = has(); setBtns.forEach((b, i) => b.setAttribute("aria-pressed", String(sets[i].tags.every(x => low.has(x.toLowerCase()))))); };
    markSets();
    const note = h("input", {type: "checkbox", checked: ui.eaglenote !== false});
    const ptags = h("input", {type: "checkbox", checked: !!cfg.EAGLE_PROMPT_TAGS});
    const msg = h("p.hint.egmsg", {role: "status"});
    const go = async (force = false) => {
      const patch = {eaglefolder: {...(ui.eaglefolder || {}), [lib.path]: chosen?.id || ""}, eaglenote: note.checked};
      post("/api/ui", patch).catch(() => { /* 只影響下次 */ });
      msg.dataset.kind = "";
      msg.textContent = t("eg.saving");
      const r = await post("/api/eagle/save", {...(d.ref ? {rid: d.ref.rid} : {id: d.id}), folder: chosen?.id || "",
                                              tags: split(tags.value), promptTags: !!d.prompt && ptags.checked,
                                              annotate: note.checked, force}).catch(e => ({error: e.message}));
      if (r.already) {
        msg.dataset.kind = "warn";
        msg.replaceChildren(t("eg.already", {when: new Date(r.already.time * 1000).toLocaleString(lang(), {dateStyle: "medium", timeStyle: "short"})}), " ",
          h("button.mini", {type: "button", "data-i18n": "eg.again", on: {click: () => go(true)}}));
        translate(msg);
        return;
      }
      msg.dataset.kind = r.error ? "err" : "ok";
      msg.textContent = r.error || t("eg.saved", {name: r.library || lib.name});
    };
    box.replaceChildren(...[
      h("p", {text: t("eg.into", {name: lib.name})}),
      h("div.egrow", {}, h("span.dim", {"data-i18n": "eg.folder"}), shown),
      find,
      results,
      h("label.egrow", {}, h("span.dim", {"data-i18n": "eg.tags"}), tags),
      sets.length ? h("div.egrow", {}, h("span.dim", {"data-i18n": "eg.sets"}), h("div.egsets", {}, setBtns))
        : h("p.dim.small", {"data-i18n": "eg.setsNone"}),
      d.prompt ? h("label.egcheck", {}, ptags, h("span", {"data-i18n": "eg.promptTags"})) : null,
      h("label.egcheck", {}, note, h("span", {"data-i18n": "eg.note"})),
      h("button.pill.primary", {type: "button", "data-i18n": "eg.go", on: {click: () => go()}}),
      msg].filter(Boolean));
    translate(box);
  };
  fill();
  return [h("div.hd", {}, h("h3.title", {"data-i18n": "eg.title"})), box];
}

async function rate(d, it, r) {
  const res = await post("/api/rate", {id: d.id, rating: r}).catch(e => ({error: e.message}));
  if (res.error) return;
  d.rating = r;
  it.rating = r;
  $("vPanel").querySelectorAll(".vstars button").forEach((b, i) => b.setAttribute("aria-pressed", String(r >= i + 1)));
}

// ── 送到生成區 ──
// 舊版漏了三件事，這裡補上（使用者筆記第 8 點）：
//   1. CoCo 存圖時會把 <lora:…> 從提示詞拿掉，只記在 loras 欄位，要補回提示詞
//   2. 模型：圖是用哪個模型生的，找得到檔案就換過去
//   3. 取樣器與排程器的名稱要轉成這裡的（Forge 寫 ER SDE、Automatic）
// only：「只送提示詞」（使用者要求 2026-10-12）——只帶正、負面提示詞與 LoRA，模型與參數不動；
// 設定開了「只送提示詞時保留生成台的 LoRA」時，LoRA 用生成台提示詞裡的、不帶這張圖的
const imageLoras = d => (d.loras || []).map(l => ({name: stem(l.file), weight: l.weight}));
const key = s => stem(s).toLowerCase();

async function sendToGen(d, btn, only = false) {
  const notes = [];        // 沒帶到的東西（黃字提醒）
  const info = [];         // 順便做了的事（例如換了模型）
  // 開著品質標籤生成的：只帶使用者自己寫的，自動加上的那些生成時會再加，不然會一直疊上去
  const ex = d.extra || {};
  const own = ex.user_prompt ?? d.prompt ?? "";
  const keep = only && genSettings()?.SEND_KEEP_LORAS === true;
  const patch = {PROMPT: keep ? keepLoras(own, genSettings()?.PROMPT) : withLoras(own, imageLoras(d)),   // 已經在提示詞裡的不會重複加
                 NEGATIVE: ex.user_negative ?? d.negative ?? ""};
  if (keep) info.push(t("view.keptLoras"));
  else await checkLoras(d, patch, info, notes);
  if (!only) await withParams(d, patch, info, notes);
  await deliver(patch, {text: [t(only ? "view.sentPrompt" : "view.sent"), ...info, ...notes].join(" · "), kind: notes.length ? "warn" : "ok"}, btn);
}

// 這張圖的 LoRA 找不到檔案的先講；名字找不到但記了雜湊值的（改過名），換成新名字
async function checkLoras(d, patch, info, notes) {
  const ex = d.extra || {};
  try {
    const loras = await get("/api/res?kind=lora");
    const have = new Set((loras.items || []).map(x => key(x.name)));
    let missing = (d.loras || []).map(l => stem(l.file)).filter(n => n && !have.has(n.toLowerCase()));
    // 名字找不到：用圖裡記的雜湊值找改過名的（Forge 的 Lora hashes、CoCo 存在 loras 裡的 hash），找到就換成新名字
    const hashes = {};
    for (const m of String(ex["Lora hashes"] || "").matchAll(/([^:,]+):\s*([0-9a-f]{10,64})/gi)) hashes[m[1].trim()] = m[2];
    for (const l of d.loras || []) if (l.hash) hashes[stem(l.file)] = l.hash;
    const ask = Object.fromEntries(missing.filter(n => hashes[n]).map(n => [n, hashes[n]]));
    if (Object.keys(ask).length) {
      const r = await post("/api/lora/resolve", {hashes: ask}).catch(() => ({}));
      for (const [old, now] of Object.entries(r.found || {})) {
        patch.PROMPT = renameLora(patch.PROMPT, old, now);
        info.push(t("view.loraRenamed", {old, now}));
      }
      missing = missing.filter(n => !(r.found || {})[n]);
      if (r.partial && missing.length) notes.push(t("view.loraStillChecking"));
    }
    if (missing.length) notes.push(t("view.noLora", {names: missing.join("、")}));
  } catch { /* 找不到資源清單時照樣帶入參數，生成時後端會再提醒 */ }
}

// 送到生成區的其他參數：尺寸、步數、CFG、seed、取樣器、模型、shift、NGMS 與加權方式、ControlNet
async function withParams(d, patch, info, notes) {
  const ex = d.extra || {};
  for (const [k, v] of [["WIDTH", d.width], ["HEIGHT", d.height], ["STEPS", d.steps], ["CFG", d.cfg],
                        ["SEED", d.seed], ["CLIP_SKIP", d.clip_skip]]) {
    if (v != null && v !== "") patch[k] = Number(v);
  }
  const shift = Number(ex.Shift ?? ex.shift);
  const m = mapSampler(d.sampler, d.scheduler);
  if (m.sampler) patch.SAMPLER = m.sampler; else if (d.sampler) notes.push(t("view.noSampler", {name: d.sampler}));
  if (m.schedule) patch.SCHEDULE = m.schedule;
  try {
    if (d.model) {
      const [models, cur] = await Promise.all([get("/api/res?kind=model"), get("/api/settings")]);
      const hit = (models.items || []).find(x => key(x.name) === key(d.model) || key(x.path) === key(d.model));
      if (!hit) notes.push(t("view.noModel", {name: stem(d.model)}));
      else if (key(hit.path) !== key(cur.MODEL)) {
        await post("/api/res/pick", {kind: "model", path: hit.path});
        info.push(t("view.modelSwitched", {name: hit.name}));
      }
    }
  } catch { /* 找不到模型清單時照樣帶入參數 */ }
  // shift 一個架構記一格：模型可能剛換過，照換完的架構放（Anima 與 Z-Image 的官方值不同）
  if (Number.isFinite(shift) && shift > 0) {
    const arch = (await get("/api/settings").catch(() => ({}))).MODEL_ARCH;
    patch[shiftKey(arch)] = shift;
  }
  // NGMS 與加權方式（只影響 SDXL／SD 1.5）：照這張圖的設定切換，不然重畫出來會不一樣。CoCo 與 Forge 都照 Forge 的寫法記：
  // 參數裡沒有 NGMS 就是關；加權方式只在提示詞有加權括號時有差，沒寫 Emphasis 就是預設（拉回整體強度）
  try {
    const st = await get("/api/settings");
    if (["sdxl", "sd15"].includes(st.MODEL_ARCH)) {
      const ngms = Math.max(0, Number(ex.NGMS) || 0);
      if (ngms !== (Number(st.NGMS) || 0)) {
        patch.NGMS = ngms;
        info.push(ngms ? t("view.ngmsSet", {v: ngms}) : t("view.ngmsOff"));
      }
      if (/[([]/.test(`${d.prompt || ""}${d.negative || ""}`)) {
        const norm = ex.Emphasis !== "No norm";
        if (norm !== (st.EMPHASIS_NORM !== false)) {
          patch.EMPHASIS_NORM = norm;
          info.push(t(norm ? "view.emphNorm" : "view.emphNoNorm"));
        }
      }
    }
  } catch { /* 讀不到設定就不動這兩個 */ }

  // ControlNet：圖裡記了參考圖就一起帶回（模型或參考圖找不到的列出來）；這張沒用參考圖，生成台開著的先關掉，
  // 不然送過去畫出來會跟這張不一樣
  try {
    const cn = (d.extra || {}).controlnet;
    const cur = (await get("/api/settings")).CONTROLNETS || [];
    if (Array.isArray(cn) && cn.some(u => u.type !== "repaint")) {
      const r = await post("/api/controlnet/restore", {units: cn});
      if (r.units?.length) {
        patch.CONTROLNETS = r.units;
        info.push(t("view.cnBack", {n: r.units.length}));
      }
      for (const [kind, name] of r.missing || []) notes.push(t(kind === "model" ? "view.cnNoModel" : "view.cnNoImage", {name}));
    } else if (cur.some(u => u.src && !u.off)) {
      patch.CONTROLNETS = cur.map(u => ({...u, off: true}));
      info.push(t("view.cnOff"));
    }
  } catch { /* 參考圖帶不回來不影響其他參數 */ }
}

async function deliver(patch, notice, btn) {
  // ⧉ 另開的視窗（圖庫、比較…）：自己留在這一頁，主視窗切到生成頁顯示結果
  // （使用者回饋 2026-10-10：另開的視窗跳去生成頁，開新視窗就沒意義了）。參數由 applyParams 存好並通知其他視窗
  const main = popoutOpener();
  await applyParams(patch, main ? null : notice);
  if (main) {
    main.dispatchEvent(new main.CustomEvent("coco:notice", {detail: notice}));
    main.location.hash = "#generate";
    if (btn) { btn.textContent = t("view.sentMain"); btn.disabled = true; }
    return;
  }
  closeViewer();
  location.hash = "#generate";
}

document.addEventListener("keydown", e => {
  if (!viewerOpen() || e.defaultPrevented) return;
  const typing = e.target instanceof Element && e.target.closest("input, textarea");
  if (e.key === "Escape") { e.preventDefault(); closeViewer(); }
  else if (e.key === "ArrowLeft" && !typing) { e.preventDefault(); step(-1); }
  else if (e.key === "ArrowRight" && !typing) { e.preventDefault(); step(1); }
  else if (e.key === "Tab") {
    // 焦點留在視窗裡
    const f = [...V.el.querySelectorAll("button:not([disabled]), summary, [href], input")].filter(x => x.offsetParent);
    if (!f.length) return;
    const [a, b] = [f[0], f[f.length - 1]];
    if (e.shiftKey && document.activeElement === a) { e.preventDefault(); b.focus(); }
    else if (!e.shiftKey && document.activeElement === b) { e.preventDefault(); a.focus(); }
  }
});
