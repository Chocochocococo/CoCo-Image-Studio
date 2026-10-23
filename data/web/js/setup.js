// 設定：圖庫要掃的資料夾、資源資料夾、縮圖寬度等，以及挑資料夾用的瀏覽器。存進 settings.json。
// 內容多了，跟資源庫一樣左邊分頁（使用者筆記）：圖庫、資源資料夾、其他。存檔一次存全部分頁改過的
//
// 資源資料夾（使用者筆記 #5）：指定一個「模型總資料夾」（例如 StabilityMatrix 的 Data\Models），
// 底下的子資料夾依名稱自動分到模型、LoRA、VAE…（server/resources.py 的 FOLDER_KINDS），
// 認錯的在表格裡改（MODELS_ROOT_MAP）；各類自己的資料夾清單照樣有效，當成額外的資料夾。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {closePop} from "./pop.js";
import {send} from "./sync.js";
import {info} from "./hint.js";
import {downloadsBox} from "./modeldl.js";
import {installBox, shortcutBox} from "./install.js";
import {aboutBody} from "./about.js";

// 設定鍵 → 顯示文字
const DIRS = [["SCAN_DIRS", "set.scan", "set.scanHint"], ["MODEL_DIRS", "set.models"], ["LORA_DIRS", "set.loras"],
              ["VAE_DIRS", "set.vaes"], ["TE_DIRS", "set.tes"], ["EMBEDDING_DIRS", "set.embeddings"],
              ["DETECTOR_DIRS", "set.detectors"]];
const ROOT_KEYS = ["MODELS_ROOT", "MODELS_ROOT_MAP"];
const NUMS = [["THUMB_WIDTH", "set.thumbW", 120, 1024], ["PAGE_SIZE", "set.pageSize", 20, 500],
              ["GALLERY_PORT", "set.port", 1024, 65535]];
// 開關（不用重開）
const BOOLS = [["HARD_DELETE", "set.hardDelete", "set.hardDeleteHint"]];
// 看大圖：只送提示詞時保留生成台的 LoRA、複製提示詞時連 LoRA 一起複製（使用者要求 2026-10-12）
const VIEW_BOOLS = [["SEND_KEEP_LORAS", "set.keepLoras", "set.keepLorasHint"], ["COPY_WITH_LORAS", "set.copyLoras", "set.copyLorasHint"]];
// 生成後自動修臉（開關在生成鈕旁邊）：要不要留沒修過的、修改幅度（跟精修的修臉三段一樣）
const FACE_BOOLS = [["AUTO_FACE_KEEP", "set.faceKeep", "set.faceKeepHint"]];
const FACE_LEVELS = [["light", 0.3], ["mid", 0.4], ["strong", 0.55]];
// 畫質：模型自帶的 SDXL VAE 用 32 位元（預設關；換了下次生成會重新載入模型）
const QUALITY_BOOLS = [["RNG_CPU", "set.rngCpu", "set.rngCpuHint"], ["VAE_FP32", "set.vaeFp32", "set.vaeFp32Hint"],
                       ["FP8_WEIGHTS", "set.fp8Weights", "set.fp8WeightsHint"], ["EMPHASIS_NORM", "set.emphasisNorm", "set.emphasisNormHint"]];
// NGMS（0＝關）：小數，範圍跟 core/settings.RANGES 一樣（tests/test_limits.py 對得上）
const NGMS_RANGE = {min: 0, max: 15};

// 存回 Eagle：預設標籤、標籤組合（存的時候點一下套用）、正面提示詞存成標籤（存的小視窗裡預設勾不勾）
const EAGLE_BOOLS = [["EAGLE_PROMPT_TAGS", "set.egPrompt", "set.egPromptHint"]];
const EAGLE_KEYS = ["EAGLE_TAGS", "EAGLE_TAG_SETS", "EAGLE_SKIP_TAGS"];

// 存檔名稱（core/naming.py）：生成的圖、06 輸出的編輯、拼接各一個格式 [設定鍵, 種類, 名稱, 預設]
const NAME_KINDS = [["SAVE_NAME", "gen", "set.nameGen", "{date}/{seq}-{seed}-{width}x{height}"],
                    ["EDIT_NAME", "edit", "set.nameEdit", "{date}/{name}_edit"],
                    ["COLLAGE_NAME", "collage", "set.nameCollage", "{date}/collage-{date}-{time}"]];
const NAME_KEYS = NAME_KINDS.map(k => k[0]);

// 介面偏好（gallery_ui.json）裡的開關：一樣在這裡改、按存檔才存（使用者筆記：生成後送去比較要好找）
const UI_BOOLS = [["autocmp", "set.autoCmp", "set.autoCmpHint"]];
// 介面偏好裡的選項：自動建的比較組幾天沒打開就封存（0＝不封存）
const UI_SELECTS = [["cmpautoarch", "set.autoArch", "set.autoArchHint", [0, 3, 7, 14, 30]]];

const S = {el: null, orig: null, cur: null, uiOrig: {}, uiCur: {}, back: null, onSaved: null, folders: null, kinds: [], rootSeq: 0,
           tab: "library"};

// 分頁與各自管的設定鍵（改了沒存的分頁標一個點；介面偏好的鍵前面加 ui:）
const TABS = [["library", "set.tab.library", ["OUTPUT_DIR", "SCAN_DIRS", ...VIEW_BOOLS.map(b => b[0]), "HARD_DELETE"]],
              ["gen", "set.tab.gen", [...NAME_KEYS, ...FACE_BOOLS.map(b => b[0]), "AUTO_FACE_STRENGTH", ...QUALITY_BOOLS.map(b => b[0]), "NGMS", ...[...UI_BOOLS, ...UI_SELECTS].map(b => `ui:${b[0]}`)]],
              ["res", "set.tab.res", [...ROOT_KEYS, ...DIRS.slice(1).map(d => d[0]), "TAGGER_DIR", "JOYCAPTION_DIR"]],
              ["refs", "set.tab.refs", ["REF_SOURCES"]],
              ["eagle", "set.tab.eagle", [...EAGLE_KEYS, ...EAGLE_BOOLS.map(b => b[0])]],
              ["other", "set.tab.other", NUMS.map(n => n[0])],
              ["about", "set.tab.about", []]];        // 介紹、官網、贊助（about.js），沒有設定

export async function openSetup(onSaved) {
  closePop();
  S.onSaved = onSaved;
  S.back = document.activeElement;
  const [s, ui] = await Promise.all([get("/api/settings"), get("/api/ui").catch(() => ({}))]);
  S.orig = JSON.parse(JSON.stringify(s));
  S.cur = JSON.parse(JSON.stringify(s));
  S.uiOrig = Object.fromEntries([...UI_BOOLS.map(([k]) => [k, !!ui[k]]), ...UI_SELECTS.map(([k]) => [k, Number(ui[k]) || 0])]);
  S.uiCur = {...S.uiOrig};
  S.el?.remove();
  S.el = h("div.assets.setup", {role: "dialog", "aria-modal": "true", "aria-labelledby": "setupTitle"});
  S.el.addEventListener("click", e => { if (e.target === S.el) closeSetup(); });
  document.body.append(S.el);
  document.body.classList.add("viewing");
  draw();
}

export function closeSetup() {
  if (!S.el) return;
  S.el.remove();
  S.el = null;
  document.body.classList.remove("viewing");
  S.back?.focus?.();
}

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const ALL_BOOLS = [...BOOLS, ...VIEW_BOOLS, ...FACE_BOOLS, ...QUALITY_BOOLS, ...EAGLE_BOOLS];
const changedKeys = () => ["OUTPUT_DIR", "TAGGER_DIR", "JOYCAPTION_DIR", ...DIRS.map(d => d[0]), ...ROOT_KEYS, ...NUMS.map(n => n[0]), ...ALL_BOOLS.map(b => b[0]), "AUTO_FACE_STRENGTH", "NGMS",
                           "REF_SOURCES", ...NAME_KEYS, ...EAGLE_KEYS]
  .filter(k => !same(S.orig[k] ?? null, S.cur[k] ?? null) && !(ALL_BOOLS.some(b => b[0] === k) && !S.orig[k] === !S.cur[k]));
const changedUi = () => [...UI_BOOLS.map(b => b[0]).filter(k => !!S.uiOrig[k] !== !!S.uiCur[k]),
                         ...UI_SELECTS.map(b => b[0]).filter(k => S.uiOrig[k] !== S.uiCur[k])];

// ── CoCo 自己的模型資料夾（server/modelhome.py）：一定會讀，每一格寫用途、幾個檔、可以打開 ──
function homeBox() {
  const box = h("div.sroot");
  const open = sub => post("/api/res/open_home", sub ? {sub} : {}).catch(() => { /* 打不開就算了 */ });
  get("/api/res/home").then(d => {
    box.replaceChildren(
      h("div.drow", {}, h("span.mono.small.shome", {text: d.home, title: d.home}),
        h("button.mini", {type: "button", "data-i18n": "set.homeOpen", on: {click: () => open("")}})),
      h("div.stable", {role: "table"}, d.folders.map(f => h("div.strow.shrow", {role: "row", "data-empty": f.files ? null : "1"},
        h("span.shname", {role: "cell", title: f.path}, h("b.mono", {text: f.name}), h("span.dim", {"data-i18n": `set.home.${f.name}`})),
        h("span.sfiles.mono", {role: "cell", text: t("set.rootFiles", {n: f.files})}),
        h("button.mini", {type: "button", role: "cell", "data-i18n": "set.homeOpen", on: {click: () => open(f.name)}})))));
    translate(box);
  }).catch(() => { /* 讀不到就不顯示 */ });
  return box;
}

// ── 模型總資料夾：路徑＋子資料夾的分類表 ──
function rootBox() {
  const box = h("div.sroot");
  const draw = () => {
    const root = S.cur.MODELS_ROOT || "";
    const map = S.cur.MODELS_ROOT_MAP || {};
    const kindOf = f => map[f.name] ?? map[f.name.toLowerCase()] ?? f.auto;
    box.replaceChildren(...[
      h("div.drow", {},
        h("input.find", {type: "text", value: root, spellcheck: "false", "data-i18n-placeholder": "set.rootPh", "data-i18n-aria": "set.root",
                         on: {change: e => setRoot(e.target.value.trim())}}),
        h("button.mini", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
          const got = await pickFolder(root);
          if (got) setRoot(got);
        }}}),
        root ? h("button.mini", {type: "button", "data-i18n": "set.remove", on: {click: () => setRoot("")}}) : null),
      !root ? h("p.dim.small", {"data-i18n": "set.rootNone"})
        : S.folders == null ? h("p.dim.small", {"data-i18n": "wall.loading"})
        : !S.folders.length ? h("p.dim.small", {"data-i18n": "set.rootEmpty"})
        : h("div.stable", {role: "table"}, S.folders.map(f => h("div.strow", {role: "row", "data-dup": f.dup ? "1" : null,
                                                                               "data-empty": f.files ? null : "1",
                                                                               "data-off": !kindOf(f) || kindOf(f) === "skip" ? "1" : null},
          h("span.sname", {role: "cell", text: f.name, title: f.path}),
          h("span.sfiles.mono", {role: "cell", text: t("set.rootFiles", {n: f.files ?? 0})}),
          f.dup ? h("span.dim.small", {role: "cell", "data-i18n": "set.rootDup"})
            : h("select.chip", {role: "cell", "data-i18n-aria": "set.rootKind", on: {change: e => {
              const m = {...(S.cur.MODELS_ROOT_MAP || {})};
              delete m[f.name];
              delete m[f.name.toLowerCase()];
              if (e.target.value !== (f.auto || "skip")) m[f.name] = e.target.value;   // 跟自動判斷一樣就不必記
              S.cur.MODELS_ROOT_MAP = m;
              draw();
              markDirty();
            }}}, S.kinds.map(k => h("option", {value: k, text: t(`set.kind.${k}`), selected: (kindOf(f) || "skip") === k})))))),
    ].filter(Boolean));
    translate(box);
  };
  const setRoot = async v => {
    S.cur.MODELS_ROOT = v;
    S.folders = v ? null : [];
    draw();
    markDirty();
    if (!v) return;
    const seq = ++S.rootSeq;
    const d = await get(`/api/res/folders?root=${encodeURIComponent(v)}`).catch(() => ({folders: []}));
    if (seq !== S.rootSeq) return;
    S.folders = d.folders || [];
    S.kinds = d.kinds || S.kinds;
    draw();
  };
  S.cur.MODELS_ROOT_MAP = {...(S.cur.MODELS_ROOT_MAP || {})};
  setRoot(S.cur.MODELS_ROOT || "");
  return box;
}

function dirList(key) {
  const list = S.cur[key] = (S.cur[key] || []).slice();
  const box = h("div.dlist");
  const redraw = () => { box.replaceChildren(...rows()); translate(box); markDirty(); };
  const rows = () => [
    ...list.map((p, i) => h("div.drow", {},
      h("input.find", {type: "text", value: p, spellcheck: "false", "data-i18n-aria": "set.path",
                       on: {change: e => { list[i] = e.target.value.trim(); S.cur[key] = list.filter(Boolean); markDirty(); }}}),
      h("button.mini", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
        const got = await pickFolder(list[i]);
        if (got) { list[i] = got; S.cur[key] = list.filter(Boolean); redraw(); }
      }}}),
      h("button.mini", {type: "button", "data-i18n": "set.remove", on: {click: () => {
        list.splice(i, 1); S.cur[key] = list.filter(Boolean); redraw();
      }}}))),
    list.length ? null : h("p.dim.small", {"data-i18n": "set.noDirs"}),
    h("button.pill", {type: "button", "data-i18n": "set.add", on: {click: async () => {
      const got = await pickFolder(list[list.length - 1] || "");
      if (got && !list.some(x => x.toLowerCase() === got.toLowerCase())) { list.push(got); S.cur[key] = list.filter(Boolean); redraw(); }
    }}})].filter(Boolean);
  box.append(...rows());
  return box;
}

// ── 輸出資料夾：生成、精修、06 輸出存圖的地方（空的＝CoCo 資料夾裡的 outputs）──
function outputBox() {
  const def = S.cur.OUTPUT_DIR_DEFAULT || "";
  const input = h("input.find", {type: "text", value: S.cur.OUTPUT_DIR || def, spellcheck: "false", "data-i18n-aria": "set.output",
                                 on: {change: e => { S.cur.OUTPUT_DIR = e.target.value.trim() || def; e.target.value = S.cur.OUTPUT_DIR; markDirty(); }}});
  return h("div.dlist", {},
    h("div.drow", {}, input,
      h("button.mini", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
        const got = await pickFolder(input.value);
        if (got) { input.value = got; S.cur.OUTPUT_DIR = got; markDirty(); }
      }}}),
      h("button.mini", {type: "button", "data-i18n": "set.nameReset", on: {click: () => { input.value = def; S.cur.OUTPUT_DIR = def; markDirty(); }}})));
}

// ── 打標台的模型資料夾（訓練 → 資料集）：WD14、JoyCaption。設定一次就好，所以放這裡不放在資料集頁 ──
function taggerBox(key = "TAGGER_DIR", lab = "set.tagger", ph = "set.taggerPh") {
  const input = h("input.find", {type: "text", value: S.cur[key] || "", spellcheck: "false", "data-i18n-placeholder": ph,
                                 "data-i18n-aria": lab,
                                 on: {change: e => { S.cur[key] = e.target.value.trim(); markDirty(); }}});
  return h("div.dlist", {},
    h("div.drow", {}, input,
      h("button.mini", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
        const got = await pickFolder(input.value);
        if (got) { input.value = got; S.cur[key] = got; markDirty(); }
      }}}),
      h("button.mini", {type: "button", "data-i18n": "set.remove", on: {click: () => { input.value = ""; S.cur[key] = ""; markDirty(); }}})));
}

// 小標題＋ⓘ，說明在底下（點了才展開）
function head(tag, label, hintKey) {
  const [btn, hint] = info(hintKey);
  return [h(tag, {}, t(label), btn), hint];
}

// 開關：一張卡片，左邊名稱與說明、右邊開關（跟上面的數字格同一種外觀）
// store：S.cur（settings.json）或 S.uiCur（介面偏好）
// NGMS：雜訊低於這個值的步驟每隔一步只算正面提示詞（core/pipeline._ngms）；0＝關
function ngmsCell() {
  return h("div.cell.swcell", {},
    h("span.swtext", {}, h("b", {id: "num-NGMS", "data-i18n": "set.ngms"}), h("span.hint", {"data-i18n": "set.ngmsHint"})),
    h("input.snum", {type: "number", min: String(NGMS_RANGE.min), max: String(NGMS_RANGE.max), step: "0.1",
                     value: String(Number(S.cur.NGMS) || 0), "aria-labelledby": "num-NGMS",
                     on: {change: e => {
                       const v = Number(e.target.value);
                       S.cur.NGMS = Number.isFinite(v) ? Math.max(NGMS_RANGE.min, Math.min(NGMS_RANGE.max, Math.round(v * 100) / 100)) : 0;
                       e.target.value = String(S.cur.NGMS);
                       markDirty();
                     }}}));
}

function switchCard([key, lab, hint], store = S.cur) {
  return h("div.cell.swcell", {},
    h("span.swtext", {}, h("b", {id: `sw-${key}`, "data-i18n": lab}), h("span.hint", {"data-i18n": hint})),
    h("button.switch", {type: "button", role: "switch", "aria-checked": String(!!store[key]), "aria-labelledby": `sw-${key}`,
                        on: {click: e => {
                          store[key] = !store[key];
                          e.currentTarget.setAttribute("aria-checked", String(!!store[key]));
                          markDirty();
                        }}}, h("i")));
}

// ── 參考圖來源（探索用，server/explore.py）：資料夾，或 Eagle 的檔案庫（勾要用的 Eagle 資料夾，子資料夾一起算）──
const EAGLE = {};          // 檔案庫路徑 → 資料夾樹（讀過就記著）

function refsBox() {
  const box = h("div.srefs");
  const list = () => (S.cur.REF_SOURCES = Array.isArray(S.cur.REF_SOURCES) ? S.cur.REF_SOURCES : []);
  const redraw = () => { box.replaceChildren(...rows()); translate(box); markDirty(); };
  const add = src => { if (!list().some(s => s.path.toLowerCase() === src.path.toLowerCase())) list().push(src); redraw(); };
  const rows = () => [
    ...list().map((src, i) => h("div.srefrow", {},
      h("div.drow", {},
        h("span.tag", {"data-i18n": src.kind === "eagle" ? "set.refs.eagle" : "set.refs.folder"}),
        h("span.mono.srefpath", {text: src.path, title: src.path}),
        h("button.mini", {type: "button", "data-i18n-aria": "set.remove", "data-i18n-title": "set.remove",
                          on: {click: () => { list().splice(i, 1); redraw(); }}}, "✕")),
      src.kind === "eagle" ? eagleFolders(src) : null)),
    list().length ? null : h("p.dim.small", {"data-i18n": "set.refs.none"}),
    h("div.drow", {},
      h("button.pill", {type: "button", "data-i18n": "set.refs.addFolder", on: {click: async () => {
        const got = await pickFolder("");
        if (got) add({kind: "folder", path: got});
      }}}),
      h("button.pill", {type: "button", "data-i18n": "set.refs.addEagle", on: {click: async () => {
        const got = await pickFolder("");
        if (!got) return;
        const d = await get(`/api/explore/eagle?path=${encodeURIComponent(got)}`).catch(e => ({error: e.message}));
        if (d.error) { box.querySelector(".srefmsg").textContent = d.error; return; }
        EAGLE[d.path] = d;
        add({kind: "eagle", path: d.path, folders: []});
      }}})),
    h("p.hint.warn.srefmsg", {})].filter(Boolean);
  box.append(...rows());
  return box;
}

// Eagle 的資料夾：第一次打開這一頁時去讀。資料夾可能有好幾千個（實測 8,409）：
//   上面列出勾了的（完整路徑，✕ 拿掉）；打字搜尋名字直接勾；不打字時是可以收合的樹，
//   一開始只展開勾了的那幾個所在的地方（不一次畫幾千列）。上層勾了，子資料夾一起算：顯示成勾起來、不能單獨取消
const EAGLE_HITS = 60;

function eagleFolders(src) {
  src.folders = src.folders || [];
  const box = h("details.seagle", {open: !src.folders.length},
    h("summary", {}, t(src.folders.length ? "set.refs.picked" : "set.refs.pickAll", {n: src.folders.length})),
    h("p.hint", {"data-i18n": "set.refs.eagleHint"}));
  const fill = d => {
    if (d.error) { box.append(h("p.hint.warn", {text: d.error})); return; }
    const byId = new Map();
    const walk = (fs, path, parents) => fs.forEach(f => {
      const it = {id: f.id, name: f.name, path: path ? `${path} / ${f.name}` : f.name, parents};
      byId.set(f.id, it);
      walk(f.children, it.path, [...parents, f.id]);
    });
    walk(d.folders, "", []);
    const open = new Set(src.folders.flatMap(id => byId.get(id)?.parents || []));
    let q = "";
    const chosen = h("div.echosen");
    const body = h("div.ebody");
    const find = h("input.find.efind", {type: "search", spellcheck: "false", "data-i18n-placeholder": "set.refs.findPh",
                                        "data-i18n-aria": "set.refs.findPh", on: {input: e => { q = e.target.value.trim().toLowerCase(); draw(); }}});
    const inherited = it => it.parents.some(id => src.folders.includes(id));
    const toggle = (id, on) => {
      src.folders = on ? [...new Set([...src.folders, id])] : src.folders.filter(x => x !== id);
      markDirty();
      draw();
    };
    const check = it => {
      const inh = inherited(it);
      return h("label", {"data-inherited": inh ? "" : null, title: inh ? t("set.refs.inherited") : it.path},
        h("input", {type: "checkbox", checked: inh || src.folders.includes(it.id), disabled: inh,
                    on: {change: e => toggle(it.id, e.target.checked)}}),
        h("span", {text: it.name}));
    };
    const tree = fs => h("ul.etree", {}, fs.map(f => {
      const isOpen = open.has(f.id), has = f.children.length > 0;
      return h("li", {},
        h("div.erow", {},
          has ? h("button.etog", {type: "button", "aria-expanded": String(isOpen), "aria-label": t(isOpen ? "set.refs.collapse" : "set.refs.expand"),
                                  on: {click: () => { if (isOpen) open.delete(f.id); else open.add(f.id); draw(); }}}, isOpen ? "▾" : "▸")
            : h("span.etog"),
          check(byId.get(f.id))),
        has && isOpen ? tree(f.children) : null);
    }));
    const draw = () => {
      const top = body.scrollTop;
      chosen.replaceChildren(...src.folders.map(id => byId.get(id)).filter(Boolean).map(it => h("span.echip", {title: it.path},
        h("span", {text: it.path}),
        h("button.mini", {type: "button", "data-i18n-aria": "set.remove", "data-i18n-title": "set.remove", on: {click: () => toggle(it.id, false)}}, "✕"))));
      if (q) {
        const all = [...byId.values()].filter(it => it.name.toLowerCase().includes(q));
        body.replaceChildren(...(all.length ? [
          h("ul.etree.eflat", {}, all.slice(0, EAGLE_HITS).map(it => h("li", {}, check(it), h("span.dim.small", {text: it.path})))),
          all.length > EAGLE_HITS ? h("p.dim.small", {text: t("set.refs.more", {n: all.length - EAGLE_HITS})}) : null,
        ].filter(Boolean) : [h("p.dim.small", {"data-i18n": "eg.noMatch"})]));
      } else {
        body.replaceChildren(tree(d.folders));
      }
      body.scrollTop = top;
      box.querySelector("summary").textContent = t(src.folders.length ? "set.refs.picked" : "set.refs.pickAll", {n: src.folders.length});
      translate(box);
    };
    box.append(chosen, find, body);
    draw();
  };
  if (EAGLE[src.path]) fill(EAGLE[src.path]);
  else get(`/api/explore/eagle?path=${encodeURIComponent(src.path)}`).then(d => { if (!d.error) EAGLE[src.path] = d; fill(d); })
    .catch(e => fill({error: e.message}));
  return box;
}

// ── 存檔名稱：三種（生成的圖、06 輸出的編輯、拼接）各一個格式；變數列共用，點了插進上次點過的那一格。
// 每一格下面顯示下一張會存成什麼（server 照目前的生成設定算）──
const NAME_VARS = ["date", "time", "seq", "seed", "width", "height", "model", "sampler", "steps", "cfg", "prompt", "loras", "name",
                   "year", "month", "day"];

function nameRow([key, kind, label, def], onFocus) {
  const input = h("input.find.snamein", {type: "text", value: S.cur[key] || def, spellcheck: "false", "aria-label": t(label)});
  const shown = h("code.snameout");
  const warn = h("p.hint.warn.snamewarn", {role: "status"});
  let seq = 0, timer = 0;
  const preview = async () => {
    const mine = ++seq;
    const d = await get(`/api/naming/preview?kind=${kind}&pattern=${encodeURIComponent(input.value)}`).catch(() => null);
    if (mine !== seq || !d) return;
    shown.textContent = d.path;
    warn.textContent = [d.unknown?.length ? t("set.nameUnknown", {vars: d.unknown.map(v => `{${v}}`).join(" ")}) : "",
                        d.noUnique ? t("set.nameNoUnique") : ""].filter(Boolean).join(" ");
  };
  const set = v => {
    S.cur[key] = v;
    markDirty();
    clearTimeout(timer);
    timer = setTimeout(preview, 200);
  };
  const row = {
    // 點變數：插在游標的位置
    insert: v => {
      const a = input.selectionStart ?? input.value.length, b = input.selectionEnd ?? a;
      input.value = input.value.slice(0, a) + v + input.value.slice(b);
      input.focus();
      input.setSelectionRange(a + v.length, a + v.length);
      set(input.value);
    }};
  input.addEventListener("input", () => set(input.value));
  input.addEventListener("focus", () => onFocus(row));
  preview();
  row.el = h("div.snamerow", {},
    h("b.small", {"data-i18n": label}),
    h("div.drow", {}, input,
      h("button.mini", {type: "button", "data-i18n": "set.nameReset", on: {click: () => { input.value = def; set(def); }}})),
    h("p.small", {}, h("span.dim", {"data-i18n": "set.nameNext"}), " ", shown),
    warn);
  return row;
}

function nameBox() {
  let target = null;
  const rows = NAME_KINDS.map(k => nameRow(k, r => { target = r; }));
  target = rows[0];
  return h("div.sname", {},
    h("div.snamevars", {role: "group", "data-i18n-aria": "set.nameVars"}, NAME_VARS.map(v => h("button.mini", {type: "button",
      title: t(`set.nameVar.${v}`), on: {mousedown: e => e.preventDefault(), click: () => target.insert(`{${v}}`)}},
      h("span.mono", {text: `{${v}}`}), " ", h("span.dim", {text: t(`set.nameVar.${v}`)})))),
    h("p.dim.small", {"data-i18n": "set.nameVarsHint"}),
    ...rows.map(r => r.el));
}

// ── 存回 Eagle 的標籤組合：名稱＋標籤，一列一組 ──
const splitTags = v => v.split(/[,，]/).map(x => x.trim()).filter(Boolean);

function tagSetsBox() {
  const list = S.cur.EAGLE_TAG_SETS = (S.cur.EAGLE_TAG_SETS || []).map(s => ({name: s.name || "", tags: [...(s.tags || [])]}));
  const box = h("div.dlist");
  const redraw = () => { box.replaceChildren(...rows()); translate(box); markDirty(); };
  const rows = () => [
    ...list.map((s, i) => h("div.drow.egset", {},
      h("input.find.egname", {type: "text", value: s.name, "data-i18n-placeholder": "set.egSetName", "data-i18n-aria": "set.egSetName",
                               on: {input: e => { s.name = e.target.value; markDirty(); }}}),
      h("input.find", {type: "text", value: s.tags.join(", "), spellcheck: "false", "data-i18n-placeholder": "set.egSetTags",
                       "data-i18n-aria": "set.egSetTags", on: {input: e => { s.tags = splitTags(e.target.value); markDirty(); }}}),
      h("button.mini", {type: "button", "data-i18n-aria": "set.remove", "data-i18n-title": "set.remove",
                        on: {click: () => { list.splice(i, 1); redraw(); }}}, "✕"))),
    list.length ? null : h("p.dim.small", {"data-i18n": "set.egNoSets"}),
    h("button.pill", {type: "button", "data-i18n": "set.egAddSet", on: {click: () => {
      list.push({name: "", tags: []});
      redraw();
      [...box.querySelectorAll(".egname")].pop()?.focus();
    }}})].filter(Boolean);
  box.append(...rows());
  return box;
}

function tabBody(tab) {
  if (tab === "gen") return [
    h("section.ssec", {}, ...head("h3", "set.saveName", "set.saveNameHint"), nameBox()),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.autoFace"}), FACE_BOOLS.map(b => switchCard(b)),
      h("div.cell.swcell", {},
        h("span.swtext", {}, h("b", {id: "seg-faceStrength", "data-i18n": "set.faceStrength"}), h("span.hint", {"data-i18n": "set.faceStrengthHint"})),
        h("div.seg", {role: "group", "aria-labelledby": "seg-faceStrength"}, FACE_LEVELS.map(([k, v]) => h("button", {
          type: "button", "aria-pressed": String(Math.abs((Number(S.cur.AUTO_FACE_STRENGTH) || 0.4) - v) < 0.001), "data-i18n": `ref.level.${k}`,
          on: {click: e => {
            S.cur.AUTO_FACE_STRENGTH = v;
            e.currentTarget.parentElement.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
            markDirty();
          }}}))))),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.quality"}), QUALITY_BOOLS.map(b => switchCard(b)), ngmsCell()),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.compare"}), UI_BOOLS.map(b => switchCard(b, S.uiCur)),
      UI_SELECTS.map(([key, lab, hint, opts]) => h("div.cell.swcell", {},
        h("span.swtext", {}, h("b", {id: `sel-${key}`, "data-i18n": lab}), h("span.hint", {"data-i18n": hint})),
        h("select.chip.sselect", {"aria-labelledby": `sel-${key}`, on: {change: e => { S.uiCur[key] = Number(e.target.value); markDirty(); }}},
          opts.map(v => h("option", {value: String(v), selected: S.uiCur[key] === v, text: v ? t("set.days", {n: v}) : t("set.never")}))))))];
  if (tab === "eagle") return [
    h("section.ssec", {}, ...head("h3", "set.egTags", "set.egTagsHint"),
      h("div.drow", {}, h("input.find", {type: "text", value: S.cur.EAGLE_TAGS ?? "", spellcheck: "false", "data-i18n-placeholder": "eg.tagsPh",
                                         "data-i18n-aria": "set.egTags", on: {input: e => { S.cur.EAGLE_TAGS = e.target.value; markDirty(); }}}))),
    h("section.ssec", {}, ...head("h3", "set.egSets", "set.egSetsHint"), tagSetsBox()),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.egPromptSec"}), EAGLE_BOOLS.map(b => switchCard(b)),
      ...head("h4", "set.egSkip", "set.egSkipHint"),
      h("textarea.find.egskip", {rows: "3", spellcheck: "false", "data-i18n-placeholder": "set.egSkipPh", "data-i18n-aria": "set.egSkip",
                                 value: S.cur.EAGLE_SKIP_TAGS ?? "", on: {input: e => { S.cur.EAGLE_SKIP_TAGS = e.target.value; markDirty(); }}}))];
  if (tab === "refs") return [
    h("section.ssec", {}, ...head("h3", "set.refs", "set.refsHint"), refsBox())];
  if (tab === "res") return [
    h("section.ssec", {}, ...head("h3", "set.home", "set.homeHint"), homeBox()),
    h("section.ssec", {}, ...head("h3", "set.root", "set.rootHint"), rootBox()),
    h("section.ssec", {}, ...head("h3", "set.dl", "set.dlHint"), downloadsBox()),
    h("section.ssec", {}, ...head("h3", "set.tagger", "set.taggerHint"), taggerBox()),
    h("section.ssec", {}, ...head("h3", "set.jc", "set.jcHint"), taggerBox("JOYCAPTION_DIR", "set.jc", "set.jcPh")),
    h("section.ssec", {}, ...head("h3", "set.extraDirs", "set.resHint"),
      DIRS.slice(1).map(([key, lab]) => h("div.ssub", {}, h("h4", {"data-i18n": lab}), dirList(key))))];
  if (tab === "about") return aboutBody();
  if (tab === "other") return [
    h("section.ssec", {},
      ...head("h3", "set.display", "set.restartHint"),
      h("div.snums", {}, NUMS.map(([key, lab, min, max]) => h("label.cell", {},
        h("span.k", {"data-i18n": lab}),
        h("input.v", {type: "number", min: String(min), max: String(max), step: "1", value: S.cur[key] ?? "",
                      on: {change: e => {
                        const v = Math.round(Number(e.target.value));
                        if (Number.isFinite(v)) S.cur[key] = Math.max(min, Math.min(max, v));
                        e.target.value = S.cur[key];
                        markDirty();
                      }}}))))),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.install"}), installBox()),
    h("section.ssec", {}, ...head("h3", "set.shortcut", "set.shortcutHint"), shortcutBox())];
  return [
    h("section.ssec", {}, ...head("h3", "set.output", "set.outputHint"), outputBox()),
    h("section.ssec", {}, ...head("h3", "set.scan", "set.scanHint"), dirList("SCAN_DIRS")),
    h("section.ssec", {}, h("h3", {"data-i18n": "set.viewer"}), VIEW_BOOLS.map(b => switchCard(b))),
    // switchCard 不能直接交給 map：map 的索引 0 會變成 store，開關永遠顯示關、點了也存不進去
    h("section.ssec", {}, h("h3", {"data-i18n": "set.delete"}), BOOLS.map(b => switchCard(b)))];
}

function draw() {
  const msg = h("span.msg", {id: "setupMsg", role: "status"});
  const label = TABS.find(([k]) => k === S.tab)?.[1] || TABS[0][1];
  S.el.replaceChildren(h("div.apanel.spanel", {},
    h("aside.aside", {},
      h("h2.title", {id: "setupTitle", "data-i18n": "set.title"}),
      h("nav.atabs", {id: "setupTabs", "data-i18n-aria": "set.tabs"}, TABS.map(([k, lab]) => h("button", {
        type: "button", "data-tab": k, "aria-pressed": String(S.tab === k),
        on: {click: () => { if (S.tab !== k) { S.tab = k; draw(); } }}},
        h("span", {"data-i18n": lab}), h("span.c.sdot", {"aria-hidden": "true"}))))),
    h("div.smain", {},
      h("div.ahd", {},
        h("h3.title", {"data-i18n": label}),
        h("span.sp"),
        h("button.round", {type: "button", "data-i18n-aria": "view.close", "data-i18n-title": "view.close", on: {click: closeSetup}}, "✕")),
      h("div.sbody", {}, ...tabBody(S.tab)),
      h("div.sfoot", {},
        msg, h("span.sp"),
        h("button.pill", {type: "button", "data-i18n": "set.revert", on: {click: () => {
          S.cur = JSON.parse(JSON.stringify(S.orig)); S.uiCur = {...S.uiOrig}; draw();
        }}}),
        h("button.pill.primary", {type: "button", id: "setupSave", "data-i18n": "set.save", on: {click: save}})))));
  translate(S.el);
  markDirty();
}

function markDirty() {
  const keys = [...changedKeys(), ...changedUi().map(k => `ui:${k}`)];
  const b = $("setupSave");
  if (b) b.disabled = !keys.length;
  // 改了沒存的分頁標一個點
  for (const [k, , own] of TABS) {
    const dot = S.el?.querySelector(`#setupTabs [data-tab="${k}"] .sdot`);
    if (dot) dot.textContent = own.some(x => keys.includes(x)) ? "●" : "";
  }
}

async function save() {
  const keys = changedKeys(), uiKeys = changedUi();
  if (!keys.length && !uiKeys.length) return;
  const btn = $("setupSave"), msg = $("setupMsg");
  btn.disabled = true;
  msg.textContent = keys.includes("SCAN_DIRS") ? t("set.scanning") : t("set.saving");
  msg.dataset.kind = "";
  const data = {};
  for (const k of keys) data[k] = S.cur[k];
  try {
    let d = {};
    if (keys.length) {
      d = await post("/api/settings", data);
      if (d.error) throw new Error(d.error);
      send("reload");                      // 別的視窗（例如另一個螢幕的資源庫）重讀
      S.orig = JSON.parse(JSON.stringify(d.settings || S.cur));
      S.cur = JSON.parse(JSON.stringify(S.orig));
    }
    let uiPatch = null;
    if (uiKeys.length) {
      uiPatch = Object.fromEntries(uiKeys.map(k => [k, UI_SELECTS.some(b => b[0] === k) ? S.uiCur[k] : !!S.uiCur[k]]));
      const r = await post("/api/ui", uiPatch);
      if (r?.error) throw new Error(r.error);
      S.uiOrig = {...S.uiCur};
    }
    const parts = [t("set.saved")];
    if (d.reindexed && d.index) parts.push(t("set.reindexed", {msg: d.index}));
    if (d.index_error) parts.push(t("set.indexErr", {msg: d.index_error}));
    if (keys.some(k => NUMS.some(n => n[0] === k))) parts.push(t("set.needRestart"));
    draw();
    const m = $("setupMsg");
    m.textContent = parts.join(" · ");
    m.dataset.kind = d.index_error ? "err" : "ok";
    S.onSaved?.(keys, d, uiPatch);
  } catch (e) {
    msg.textContent = t("set.err", {msg: e.message || e});
    msg.dataset.kind = "err";
    btn.disabled = false;
  }
}

// ── 資料夾瀏覽器 ──
// 回傳選到的路徑；取消回傳 null。空路徑＝磁碟機與常用位置
const ROOT_NAMES = {"桌面": "fb.desktop", "文件": "fb.documents", "圖片": "fb.pictures", "下載": "fb.downloads",
                    "專案資料夾": "fb.project", "家目錄": "fb.home"};

export function pickFolder(start = "") {
  return new Promise(resolve => {
    const el = h("div.fbrowse", {role: "dialog", "aria-modal": "true", "aria-labelledby": "fbTitle"});
    const pathIn = h("input.find", {type: "text", spellcheck: "false", "data-i18n-aria": "set.path"});
    const list = h("div.fblist", {role: "listbox", "data-i18n-aria": "fb.title"});
    const newName = h("input.find", {type: "text", "data-i18n-placeholder": "fb.newName", "data-i18n-aria": "fb.newName"});
    const msg = h("span.msg", {role: "status"});
    let cur = "";
    const done = v => { el.remove(); removeEventListener("keydown", onKey, true); resolve(v); };
    const onKey = e => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); } };
    const go = async p => {
      const d = await get(`/api/browse?path=${encodeURIComponent(p || "")}`).catch(err => ({error: err.message}));
      if (d.error) {
        // 起點不存在（例如資料夾被搬走了）就改從磁碟機清單開始
        if (!list.childElementCount && p) { go(""); return; }
        msg.textContent = d.error; msg.dataset.kind = "err"; return;
      }
      msg.textContent = "";
      cur = d.path || "";
      pathIn.value = cur;
      up.disabled = !cur;
      pick.disabled = !cur;
      list.replaceChildren(...(d.dirs.length ? d.dirs.map(x => h("button.fbitem", {type: "button", title: x.path,
        on: {click: () => go(x.path)}}, ROOT_NAMES[x.name] && !cur ? t(ROOT_NAMES[x.name]) : x.name))
        : [h("p.dim.small", {"data-i18n": "fb.empty"})]));
      translate(list);
      parentPath = d.parent;
    };
    let parentPath = "";
    const up = h("button.pill", {type: "button", "data-i18n": "fb.up", on: {click: () => go(parentPath || "")}});
    const pick = h("button.pill.primary", {type: "button", "data-i18n": "fb.pick", on: {click: () => done(cur || null)}});
    el.append(h("div.fbbox", {},
      h("div.ahd", {}, h("h2.title", {id: "fbTitle", "data-i18n": "fb.title"}), h("span.sp"),
        h("button.round", {type: "button", "data-i18n-aria": "view.close", on: {click: () => done(null)}}, "✕")),
      h("div.fbtop", {}, up, pathIn,
        h("button.pill", {type: "button", "data-i18n": "fb.go", on: {click: () => go(pathIn.value.trim())}})),
      list,
      h("div.fbfoot", {}, newName,
        h("button.pill", {type: "button", "data-i18n": "fb.mkdir", on: {click: async () => {
          const name = newName.value.trim();
          if (!cur || !name) return;
          const d = await post("/api/mkdir", {parent: cur, name}).catch(err => ({error: err.message}));
          if (d.error) { msg.textContent = d.error; msg.dataset.kind = "err"; return; }
          newName.value = "";
          go(d.path);
        }}}),
        msg, h("span.sp"),
        h("button.pill", {type: "button", "data-i18n": "fb.cancel", on: {click: () => done(null)}}),
        pick)));
    pathIn.addEventListener("keydown", e => { if (e.key === "Enter") go(pathIn.value.trim()); });
    el.addEventListener("click", e => { if (e.target === el) done(null); });
    addEventListener("keydown", onKey, true);
    document.body.append(el);
    translate(el);
    go(start);
  });
}

document.addEventListener("keydown", e => {
  if (S.el && e.key === "Escape" && !e.defaultPrevented) { e.preventDefault(); closeSetup(); }
});
