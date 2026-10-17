// 參考圖（ControlNet）：生成台資源格的「參考圖」打開的視窗。最多 3 張參考圖，每張可以選：
//   來源：圖庫的圖、探索的參考圖（看大圖的「當參考圖」）、從電腦選（拖進來也行）
//   控制的種類：姿勢、深度、線稿、Canny 輪廓、軟邊緣、Tile；參考圖是一般的圖（自動抽出姿勢、算深度…）還是已經是控制圖
//   模型：union／promax 什麼種類都能用；一般的 ControlNet 只管一種（從檔名猜，猜不到的都列）
//   強度、作用的步數範圍（從幾 % 到幾 %）、尺寸不同時裁切或整張放進
// 「看處理結果」照目前的生成大小做一次前處理給你看（抽到幾個人）。下面是模型：可以下載的（union、DWPose、深度）
// 與模型總資料夾裡找到的。設定存在 settings.json 的 CONTROLNETS（生成台的設定，排程的快照也會帶上）。
// 伺服器那邊在 server/controlnet.py，生成在 core/controlnet_run.py。

import {get, post} from "./api.js";
import {h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {partPct} from "./modeldl.js";
import {genSettings, setGen} from "./generate.js";
import {openPicker} from "./picker.js";

const TYPES = ["pose", "depth", "lineart", "canny", "softedge", "tile"];
const AUTO = {pose: "dwpose", depth: "depth", lineart: "lineart", canny: "canny", softedge: "softedge", tile: "tile"};
const NEED = {dwpose: "dwpose", depth: "depth"};
const MAX = 3;
const CP = {el: null, st: null, sel: 0, prev: new Map(), poll: 0, back: null, msg: ""};

const units = () => (genSettings()?.CONTROLNETS || []).map(u => ({...u}));
const save = list => { setGen("CONTROLNETS", list); };
const blank = () => ({src: null, model: defaultModel("pose"), type: "pose", prep: "dwpose", scale: 1, start: 0, end: 1, resize: "crop",
                      low: 100, high: 200, off: false});

// 參考圖的縮圖網址
export function refThumb(src) {
  if (!src) return "";
  if (src.kind === "lib") return `/api/thumb?id=${encodeURIComponent(src.id)}${src.v ? `&v=${encodeURIComponent(src.v)}` : ""}`;
  if (src.kind === "ref") return `/api/explore/ref/thumb?id=${encodeURIComponent(String(src.rid).replace(/^r/, ""))}`;
  if (src.kind === "upload") return `/api/controlnet/upload?name=${encodeURIComponent(src.name)}`;
  return "";
}

// 能用在這個種類的模型：union 都能用；一般的只管一種（猜不到種類的也列，讓使用者自己決定）
const fits = (m, type) => m.ok && (m.arch === "sdxl-union" || !m.type || m.type === type);
// 預設先用那個種類專用的，再來是 CoCo 自己下載的 union、最後是能用的。
// 2026-10-08 實測（Illustrious 系的模型、同一張姿勢圖與 seed）：union promax 的姿勢完全沒作用（試過模式 0～5、線條加粗都一樣，
// 輪廓倒是照得很好），Illustrious 與 NoobAI 專用的姿勢模型都照得很準。10/7 曾改成 union 優先（以為專用的是問題），是錯的
function defaultModel(type, cur = "") {
  const ms = CP.st?.models || [];
  if (ms.some(m => m.id === cur && fits(m, type))) return cur;
  return (ms.find(m => m.ok && m.arch !== "sdxl-union" && m.type === type) || ms.find(m => m.ok && m.own && m.arch === "sdxl-union")
          || ms.find(m => fits(m, type)))?.id || "";
}
const modelOf = id => (CP.st?.models || []).find(m => m.id === id);
const modelName = m => m ? (m.label || m.name) : "";

// 從別的地方（看大圖的「當參考圖」）加一張：有空的格子就放進去，滿了換掉選中的那張
export async function addReference(src) {
  CP.st = CP.st || await get("/api/controlnet/status").catch(() => null);
  const list = units();
  let i = list.findIndex(u => !u.src);
  if (i < 0 && list.length < MAX) { list.push(blank()); i = list.length - 1; }
  if (i < 0) i = Math.min(CP.sel, list.length - 1);
  list[i] = {...list[i], src, off: false};
  if (!list[i].model) list[i].model = defaultModel(list[i].type);
  save(list);
  CP.prev.delete(i);
  openControlNet(i);
}

export async function openControlNet(focus = null) {
  CP.back = document.activeElement;
  if (focus != null) CP.sel = focus;
  if (!CP.el) {
    CP.el = h("div.assets", {role: "dialog", "aria-modal": "true", "aria-labelledby": "cnTitle"});
    CP.el.addEventListener("click", e => { if (e.target === CP.el) close(); });
    document.body.append(CP.el);
    document.body.classList.add("viewing");
    addEventListener("keydown", onKey, true);
  }
  CP.st = await get("/api/controlnet/status").catch(e => ({error: e.message}));
  if (!units().length) save([blank()]);
  draw();
}

function close() {
  if (!CP.el) return;
  clearTimeout(CP.poll);
  removeEventListener("keydown", onKey, true);
  CP.el.remove();
  CP.el = null;
  document.body.classList.remove("viewing");
  CP.back?.focus?.();
}
const onKey = e => { if (e.key === "Escape" && !e.defaultPrevented) { e.preventDefault(); e.stopPropagation(); close(); } };

function update(i, patch, keepPreview = false) {
  const list = units();
  list[i] = {...list[i], ...patch};
  save(list);
  if (!keepPreview) CP.prev.delete(i);
  draw();
}

// ── 畫 ──
function draw() {
  if (!CP.el) return;
  const list = units(), st = CP.st || {};
  CP.sel = Math.max(0, Math.min(CP.sel, list.length - 1));
  // 還沒選模型、或選的模型不適合這個種類（例如剛下載好 union）：換成合適的
  let fixed = false;
  for (const u of list) {
    const d = defaultModel(u.type, u.model);
    if (d && d !== u.model) { u.model = d; fixed = true; }
  }
  if (fixed) save(list);
  const arch = genSettings()?.MODEL_ARCH;
  CP.el.replaceChildren(h("div.apanel.cnpanel", {},
    h("div.ahd", {},
      h("h2.title", {id: "cnTitle", "data-i18n": "cn.title"}),
      ...info("cn.hint"),
      h("span.sp"),
      h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: close}}, "✕")),
    arch && arch !== "sdxl" ? h("p.hint.warn", {"data-i18n": "cn.notSdxl"}) : null,
    st.error ? h("p.hint.warn", {text: st.error}) : null,
    h("div.cnbody", {},
      h("div.cnside", {},
        h("div.cnlist", {role: "list"}, list.map((u, i) => unitCard(u, i)),
          list.length < MAX ? h("button.pill.cnadd", {type: "button", "data-i18n": "cn.add",
                                                      on: {click: () => { save([...units(), blank()]); CP.sel = units().length - 1; draw(); }}}) : null),
        modelsBox()),
      list[CP.sel] ? unitEditor(list[CP.sel], CP.sel) : null)));
  translate(CP.el);
}

function unitCard(u, i) {
  const m = modelOf(u.model);
  return h("div.cncard", {role: "listitem", "aria-current": i === CP.sel ? "true" : null, "data-off": u.off ? "" : null,
                          on: {click: e => { if (!e.target.closest("button")) { CP.sel = i; draw(); } }}},
    h("div.cnthumb", {}, u.src ? h("img", {src: refThumb(u.src), alt: ""}) : h("span.dim", {text: "＋"})),
    h("div.cnmeta", {},
      h("b", {text: `${i + 1}. ${t(`cn.type.${u.type || "pose"}`)}`}),
      h("span.dim.small", {text: u.src ? (modelName(m) || t("cn.noModel")) : t("cn.noImage")})),
    h("button.switch.cnon", {type: "button", role: "switch", "aria-checked": String(!u.off), "data-i18n-aria": "cn.use",
                             "data-i18n-title": "cn.use", on: {click: () => update(i, {off: !u.off}, true)}}, h("i")),
    h("button.mini", {type: "button", "data-i18n-aria": "cn.remove", "data-i18n-title": "cn.remove", on: {click: () => {
      const list = units().filter((_, j) => j !== i);
      CP.prev.clear();
      save(list);
      CP.sel = Math.min(CP.sel, Math.max(0, list.length - 1));
      draw();
    }}}, "✕"));
}

function seg(aria, keys, on, label, click) {
  return h("div.seg", {role: "group", "data-i18n-aria": aria}, keys.map(k => h("button", {
    type: "button", "aria-pressed": String(on(k)), ...label(k), on: {click: () => click(k)}})));
}
function slider(label, val, min, max, step, fmt, set) {
  const out = h("span.mono.rfnum", {text: fmt(val)});
  return h("label.rflab.cnrow", {}, h("span.dim", {"data-i18n": label}), h("span.sp"),
    h("input", {type: "range", min: String(min), max: String(max), step: String(step), value: String(val), "data-i18n-aria": label,
                on: {input: e => { out.textContent = fmt(Number(e.target.value)); }, change: e => set(Number(e.target.value))}}),
    out);
}

function unitEditor(u, i) {
  const st = CP.st || {}, m = modelOf(u.model);
  const prepNeed = NEED[u.prep], prepPkg = (st.catalog || []).find(c => c.key === prepNeed);
  const choices = (st.models || []).filter(x => fits(x, u.type));
  const file = h("input", {type: "file", accept: "image/png,image/jpeg,image/webp", hidden: true,
                           on: {change: e => { if (e.target.files[0]) upload(e.target.files[0], i); e.target.value = ""; }}});
  const prev = CP.prev.get(i);
  const stage = h("div.cnimgs", {on: {
    dragover: e => { e.preventDefault(); e.currentTarget.dataset.drop = ""; },
    dragleave: e => { delete e.currentTarget.dataset.drop; },
    drop: e => { e.preventDefault(); delete e.currentTarget.dataset.drop; const f = e.dataTransfer.files?.[0]; if (f) upload(f, i); }}},
    h("figure.cnfig", {}, u.src ? h("img", {src: refThumb(u.src), alt: ""}) : h("p.dim.small", {"data-i18n": "cn.dropHere"}),
      h("figcaption.dim.small", {"data-i18n": "cn.source"})),
    h("figure.cnfig", {}, prev?.png ? h("img", {src: prev.png, alt: ""}) : h("p.dim.small", {"data-i18n": prev?.loading ? "cn.working" : "cn.noPreview"}),
      h("figcaption.dim.small", {}, t("cn.result"), prev?.people != null ? ` · ${t("cn.people", {n: prev.people})}` : "")));
  return h("section.cnedit", {"aria-label": t("cn.unit", {n: i + 1})}, ...[
    stage,
    h("div.drow", {},
      h("button.pill", {type: "button", "data-i18n": "cn.fromLibrary", on: {click: () => openPicker({title: t("cn.pickTitle"), onAdd: ids => {
        update(i, {src: {kind: "lib", id: ids[0]}});
      }})}}),
      h("button.pill", {type: "button", "data-i18n": "cn.fromFile", on: {click: () => file.click()}}), file,
      h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "cn.preview", disabled: !u.src, on: {click: () => preview(i)}})),
    h("p.hint", {"data-i18n": "cn.fromExplore"}),
    h("div.rflab.cnrow", {}, h("span.dim", {"data-i18n": "cn.type"}), h("span.sp"),
      seg("cn.type", TYPES, k => u.type === k, k => ({"data-i18n": `cn.type.${k}`}), k => {
        const prep = u.prep === "none" ? "none" : AUTO[k];
        update(i, {type: k, prep: k === "tile" ? "tile" : prep, model: defaultModel(k, u.model)});
      })),
    u.type === "tile" ? null : h("div.rflab.cnrow", {}, h("span.dim", {"data-i18n": "cn.input"}), h("span.sp"),
      seg("cn.input", ["auto", "none"], k => (k === "none") === (u.prep === "none"), k => ({"data-i18n": `cn.input.${k}.${u.type}`}),
          k => update(i, {prep: k === "none" ? "none" : AUTO[u.type]}))),
    prepNeed && prepPkg && !prepPkg.installed ? h("p.hint.warn", {}, t(`cn.needPkg.${prepNeed}`), " ",
      h("button.mini", {type: "button", "data-i18n": "cn.download", on: {click: () => download(prepNeed)}})) : null,
    h("p.hint", {"data-i18n": `cn.typeHint.${u.type}`}),
    h("label.rflab.cnrow", {}, h("span.dim", {"data-i18n": "cn.model"}), h("span.sp"),
      choices.length ? h("select.chip.cnsel", {"data-i18n-aria": "cn.model", on: {change: e => update(i, {model: e.target.value}, true)}},
        choices.map(x => h("option", {value: x.id, selected: x.id === u.model,
                                      text: `${modelName(x)}${x.arch === "sdxl-union" ? " · union" : ""}`})))
        : h("span.hint.warn", {"data-i18n": "cn.noModelFor"})),
    m && !fits(m, u.type) ? h("p.hint.warn", {"data-i18n": "cn.modelMismatch"}) : null,
    // union 的姿勢在動漫底模上常常沒作用（見 defaultModel 的實測）：提醒、有專用的就說可以換
    m && m.arch === "sdxl-union" && u.type === "pose" ? h("p.hint.warn", {"data-i18n": "cn.unionPose"}) : null,
    slider("cn.scale", u.scale ?? 1, 0, 2, 0.05, v => v.toFixed(2), v => update(i, {scale: v}, true)),
    slider("cn.start", u.start ?? 0, 0, 1, 0.05, v => `${Math.round(v * 100)}%`, v => update(i, {start: Math.min(v, (u.end ?? 1) - 0.05)}, true)),
    slider("cn.end", u.end ?? 1, 0, 1, 0.05, v => `${Math.round(v * 100)}%`, v => update(i, {end: Math.max(v, (u.start ?? 0) + 0.05)}, true)),
    h("div.rflab.cnrow", {}, h("span.dim", {"data-i18n": "cn.resize"}), h("span.sp"),
      seg("cn.resize", ["crop", "fit"], k => (u.resize || "crop") === k, k => ({"data-i18n": `cn.resize.${k}`}), k => update(i, {resize: k}))),
    u.type === "canny" && u.prep !== "none" ? slider("cn.low", u.low ?? 100, 1, 254, 1, v => String(v), v => update(i, {low: v})) : null,
    u.type === "canny" && u.prep !== "none" ? slider("cn.high", u.high ?? 200, 2, 255, 1, v => String(v), v => update(i, {high: v})) : null,
    CP.msg ? h("p.hint.warn", {text: CP.msg}) : null,
  ].filter(Boolean));
}

async function upload(f, i) {
  CP.msg = "";
  const data = await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f); });
  const d = await post("/api/controlnet/upload", {png: data}).catch(e => ({error: e.message}));
  if (d.error) { CP.msg = d.error; draw(); return; }
  update(i, {src: {kind: "upload", name: d.name}});
}

async function preview(i) {
  const u = units()[i], s = genSettings() || {};
  if (!u?.src) return;
  CP.msg = "";
  CP.prev.set(i, {loading: true});
  draw();
  const d = await post("/api/controlnet/preview", {src: u.src, type: u.type, prep: u.prep, resize: u.resize, low: u.low, high: u.high,
                                                  width: s.WIDTH, height: s.HEIGHT}).catch(e => ({error: e.message}));
  if (d.error) { CP.prev.delete(i); CP.msg = d.error; } else CP.prev.set(i, d);
  draw();
}

// ── 模型：可以下載的＋找到的 ──
function modelsBox() {
  const st = CP.st || {};
  const cat = st.catalog || [];
  const mine = (st.models || []).filter(m => !m.own);
  // 還沒有裝任何一個、下載中、上次下載到一半（要看得到「上次下載到 N%」）時展開
  return h("details.cnmodels", {open: !cat.some(c => c.installed) || cat.some(c => c.download?.running || (c.part && !c.installed))},
    h("summary", {"data-i18n": "cn.models"}),
    h("div.cnpkgs", {}, cat.map(c => {
      const dl = c.download || {};
      const pct = dl.running && dl.total ? Math.round(dl.done / dl.total * 100) : null;
      return h("div.cnpkg", {},
        h("div.cnpkgtx", {}, h("b", {"data-i18n": `cn.pkg.${c.key}`}),
          h("span.dim.small", {text: `${t(`cn.pkgHint.${c.key}`)} · ${(c.bytes / 2 ** 20 >= 1024 ? `${(c.bytes / 2 ** 30).toFixed(1)} GB` : `${Math.round(c.bytes / 2 ** 20)} MB`)} · ${c.license}`}),
          dl.error ? h("span.hint.warn", {text: dl.error})
            : c.part && !dl.running && !c.installed ? h("span.dim.small", {text: t("dl.partial", {pct: partPct(c.part, c.bytes)})}) : null),
        c.installed ? h("span.dim.small", {"data-i18n": "cn.installed"})
          : dl.running ? h("span.cnprog", {}, h("span.mono.small", {text: `${pct ?? 0}%`}),
                           h("button.mini", {type: "button", "data-i18n": "cf.cancel", on: {click: () => post("/api/controlnet/cancel", {key: c.key}).then(refresh)}}))
          : h("button.mini", {type: "button", "data-i18n": "cn.download", on: {click: () => download(c.key)}}),
        c.installed ? h("button.mini", {type: "button", "data-i18n": "cn.delete", on: {click: e => {
          const b = e.currentTarget;
          if (b.dataset.arm) post("/api/controlnet/remove", {key: c.key}).then(refresh);
          else { b.dataset.arm = "1"; b.textContent = t("cn.deleteSure"); }
        }}}) : null);
    })),
    mine.length ? h("div.cnmine", {}, h("p.dim.small", {"data-i18n": "cn.yours"}), mine.map(m => h("div.cnmodel", {title: m.path},
      h("span", {text: modelName(m)}),
      h("span.tag", {text: m.ok ? (m.arch === "sdxl-union" ? "union" : `SDXL · ${m.type ? t(`cn.type.${m.type}`) : "?"}`) : t("cn.sd15")})))) : null,
    (st.skipped || []).length ? h("p.dim.small", {text: t("cn.skipped", {names: st.skipped.map(s => s.name).join("、")})}) : null,
    h("p.dim.small", {text: t("cn.folders", {dirs: (st.dirs || []).join("；")})}));
}

async function download(key) {
  await post("/api/controlnet/download", {key}).catch(() => null);
  refresh();
}

async function refresh() {
  clearTimeout(CP.poll);
  if (!CP.el) return;
  CP.st = await get("/api/controlnet/status").catch(e => ({error: e.message}));
  draw();
  if ((CP.st.catalog || []).some(c => c.download?.running)) CP.poll = setTimeout(refresh, 1000);
}
