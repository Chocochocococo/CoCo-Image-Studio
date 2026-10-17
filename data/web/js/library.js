// 圖庫空間：整面都是圖。搜尋在頂端列中間，篩選在下面一列；
// 選了圖，底部的動作列可以評分、搬移、丟進回收桶（比較、找相似、精修之後接上）。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {lang, t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {closePop, openPop} from "./pop.js";
import {MANY, addToCompare, openCompare, openSlider} from "./compare.js";
import {openRefine} from "./refine.js";
import {openEditor} from "./editor.js";
import {openBatch} from "./batch.js";
import {openCollage} from "./collage.js";
import {openRefineBatch} from "./refinebatch.js";
import {openExplore} from "./explore.js";
import {openDataset} from "./dataset.js";
import {openTrash} from "./trash.js";
import {pickFolder} from "./setup.js";
import {openViewer} from "./viewer.js";
import {Wall} from "./wall.js";
import {Filters, dropFacets, facets, lastPart, loadFacets as fetchFacets} from "./filters.js";
import {movePop, moveText, removeFiles, trashClick, trashLabel, trashText} from "./fileacts.js";

// tw：動作列現在作用在哪道牆（圖庫空間的，或工作室右側的）。兩個畫面不會同時存在，
// 所以動作列、訊息欄沿用同一組 id（#tray、#trayMsg、#libNote），只要記住是哪道牆在選
const L = {wall: null, tw: null, ui: null, msgTimer: null, noteTimer: null, stale: false};
// 圖庫空間的篩選列（filters.js；工作室的牆有自己的一份，main.js）
const LF = new Filters({id: "lib", build: libraryRow});

// 圖庫空間的排法與大小存在 ui 偏好的 order / colw（沿用舊介面圖庫分頁的鍵名）
export const LIBRARY_SIZES = [["s", 200], ["m", 260], ["l", 340], ["x", 440]];
const ORIENTS = ["", "portrait", "landscape", "square"];

export function initLibrary(ui) {
  L.ui = ui;
  L.wall = new Wall({
    name: "lib", ui, orderKey: "order", sizeKey: "colw", zoomKey: "libzoom", sizes: LIBRARY_SIZES,
    onOpen: it => openViewer(L.wall, it.id),
    query: () => LF.query(), selectable: true, emptyKey: "lib.empty",
    onSelect: sel => drawTray(sel, L.wall), onCount: () => { const c = $("libCount"); if (c) c.textContent = L.wall.countText(); },
  });
  LF.wall = L.wall;
  return L.wall;
}

async function loadFacets() {
  try {
    await fetchFacets();
    LF.draw();
  } catch { /* 篩選清單拿不到時，搜尋與排序還是能用 */ }
}

// 第一次進圖庫才載入；之後回來保留原本的捲動位置與選取
export function showLibrary() {
  if (!facets()) loadFacets();
  if (L.stale || (!L.wall.items.length && !L.wall.loading)) { L.stale = false; L.wall.reset(); }
  else L.wall.refit();
}

// 從別處跳過來看「用了這個模型／LoRA 的圖」：換成只有這個篩選，切到圖庫
export function showInLibrary(kind, key) {
  LF.only({[kind]: [key]});
  if (location.hash === "#library") { LF.draw(); L.wall.reset(); }
  else { L.stale = true; location.hash = "#library"; }
}

// 用文字找圖（資源庫標籤分頁的「在圖庫找用過的圖」）
export function searchLibrary(q) {
  LF.only({q: String(q || "")});
  const box = document.getElementById("libSearch");
  if (box) box.value = LF.f.q;
  if (location.hash === "#library") { LF.draw(); L.wall.reset(); }
  else { L.stale = true; location.hash = "#library"; }
}

// 生成了新圖：沒有篩選、照最新排序時才補到最前面，其他情況等使用者自己重新整理
// 掃描資料夾換了：清單與圖牆都重來（在圖庫空間就馬上重載，不在就等下次進來）
export function refreshLibrary() {
  dropFacets();
  if (location.hash === "#library") { loadFacets(); L.wall.reset(); } else L.stale = true;
}

export function libraryNewImages() {
  if (!L.wall?.items.length) return;
  if (!LF.active() && !LF.f.q && LF.f.sort === "new") L.wall.refreshNewest();
  dropFacets();
}

// ── 頂端列的搜尋框 ──
export function searchBox() {
  return h("div.search", {role: "search"}, LF.searchInput("libSearch"));
}

// ── 篩選列 ──
// 圖庫空間的篩選列：來源、模型、LoRA、資料夾、評分、勝出、方向、清除，右邊排序
function libraryRow(F) {
  const fc = facets(), roots = fc?.roots || [];
  const f = F.f;
  return [
    h("div.seg.src", {role: "group", "data-i18n-aria": "lib.source", "data-i18n-title": "lib.multiHint"},
      h("button", {type: "button", "aria-pressed": String(!f.root.length), on: {click: () => F.set({root: []})}},
        t("lib.all"), fc ? h("span.c.mono", {text: fc.total.toLocaleString(lang())}) : null),
      roots.map(r => h("button", {type: "button", "aria-pressed": String(f.root.includes(r.v)), title: r.v,
                                   on: {click: () => {
                                     const next = f.root.includes(r.v) ? f.root.filter(x => x !== r.v) : [...f.root, r.v];
                                     F.set({root: next});
                                   }}},
        lastPart(r.v), h("span.c.mono", {text: r.c.toLocaleString(lang())})))),
    F.dropChip("model", "lib.model"),
    F.dropChip("lora", "lib.lora"),
    F.dropChip("folder", "lib.folder"),
    F.dropChip("rating", "lib.rating"),
    h("button.wintog", {type: "button", "aria-pressed": String(!!f.win),
                             on: {click: () => F.set({win: !f.win})}},
      t("lib.winners"), fc ? h("span.c.mono", {text: String(fc.winners ?? 0)}) : null),
    h("div.seg", {role: "group", "data-i18n-aria": "lib.orient"}, ORIENTS.map(o => h("button", {
      type: "button", "aria-pressed": String(f.orient === o), "data-i18n": `lib.orient.${o || "any"}`,
      on: {click: () => F.set({orient: o})}}))),
    F.clearButton(),
    h("span.sp"),
    F.dropChip("sort", "lib.sort")];
}

// ── 選取後的動作列 ──
// 訊息可以是文字或元素（例如「已放進比較 · 去看」帶一個連結）
function say(text, kind = "") {
  const m = $("trayMsg");
  if (!m) return;
  m.replaceChildren(text);
  m.dataset.kind = kind;
  clearTimeout(L.msgTimer);
  L.msgTimer = setTimeout(() => { m.textContent = ""; }, 6000);
}

// 工作室的牆也用同一個動作列（使用者筆記：生成區的圖庫只能一張一張送去比較）
export const trayFor = wall => sel => drawTray(sel, wall);

function drawTray(sel, wall = L.wall) {
  L.tw = wall;
  const tray = $("tray");
  if (!tray) return;
  tray.hidden = !sel.length;
  if (!sel.length) return;
  const trashBtn = h("button.b.danger", {type: "button", id: "btnTrash", "data-i18n": trashLabel(),
                                           on: {click: e => trashClick(e.currentTarget, L.tw.sel.slice(), runTrash)}});
  tray.replaceChildren(...[
    h("div.ths", {}, sel.slice(0, 3).map(it => h("img", {src: `/api/thumb?id=${it.id}&v=${encodeURIComponent(it.v)}`, alt: ""}))),
    h("span.txt", {text: t("lib.selected", {n: sel.length})}),
    h("button.b", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.toCompare",
                   on: {click: e => addToCompare(sel.map(it => it.id), e.currentTarget, d => {
                     if (d.error) { say(t("lib.err", {msg: d.error}), "err"); return; }
                     const many = d.count > MANY;
                     say(h("span", {}, t("cmp.added", {n: d.added, count: d.count}), many ? t("cmp.manyShort", {max: MANY}) : "", " ",
                       h("a", {href: "#compare", on: {click: e2 => { e2.preventDefault(); openCompare(d.id); }}}, t("cmp.go"))), many ? "warn" : "ok");
                   })}}),
    // 剛好兩張：直接拖滑桿看差異，不必先建比較組
    sel.length === 2 ? h("button.b", {type: "button", "data-i18n": "lib.slider",
                                      on: {click: () => openSlider(sel)}}) : null,
    // 找相似：一張開探索頁
    sel.length === 1 ? h("button.b", {type: "button", "data-i18n": "lib.similar", "data-i18n-title": "x.fromHint",
                                      on: {click: () => openExplore({kind: "lib", id: sel[0].id, filename: sel[0].filename, v: sel[0].v})}})
      : h("button.b", {type: "button", "aria-disabled": "true", "data-i18n": "lib.similar", "data-i18n-title": "ref.pickOne"}),
    // 精修：一張開精修頁；好幾張是批次精修（修臉、高清化、去背這些不用填提示詞的）
    sel.length === 1 ? h("button.b", {type: "button", "data-i18n": "lib.refine", "data-i18n-title": "ref.fromViewerHint",
                                      on: {click: () => openRefine(sel[0].id)}})
      : h("button.b", {type: "button", "aria-haspopup": "dialog", "data-i18n": "lib.refineBatch", "data-i18n-title": "rbt.hint",
                       on: {click: () => openRefineBatch(sel)}}),
    // 06 輸出：一次編輯一張
    sel.length === 1 ? h("button.b", {type: "button", "data-i18n": "lib.edit", "data-i18n-title": "ed.fromHint",
                                      on: {click: () => openEditor(sel[0].id)}})
      : h("button.b", {type: "button", "aria-disabled": "true", "data-i18n": "lib.edit", "data-i18n-title": "ref.pickOne"}),
    // 批次輸出：選幾張都行，照同樣的設定一次輸出
    h("button.b", {type: "button", "aria-haspopup": "dialog", "data-i18n": "lib.batch", "data-i18n-title": "bt.hint",
                   on: {click: () => openBatch(sel)}}),
    // 拼接：兩張以上拼成一張
    sel.length >= 2 ? h("button.b", {type: "button", "aria-haspopup": "dialog", "data-i18n": "lib.collage", "data-i18n-title": "clg.hint",
                                     on: {click: () => openCollage(sel)}})
      : h("button.b", {type: "button", "aria-disabled": "true", "data-i18n": "lib.collage", "data-i18n-title": "clg.pickTwo"}),
    // 加進資料集：複製到訓練用的資料夾（訓練空間的 01 資料集打標籤）
    h("button.b", {type: "button", "aria-haspopup": "dialog", "data-i18n": "lib.toDataset", on: {click: openDatasetDialog}}),
    h("button.b", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.rate",
                   on: {click: e => openPop("popRate", e.currentTarget, ratePop)}}),
    h("button.b", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n": "lib.move",
                   on: {click: e => openPop("popMove", e.currentTarget, () => movePop(L.tw.sel.slice(), L.ui, moved))}}),
    sel.length === 1 ? h("button.b", {type: "button", "data-i18n": "lib.reveal",
                                       on: {click: () => post("/api/reveal", {id: sel[0].id})}}) : null,
    trashBtn,
    h("button.b", {type: "button", "data-i18n": "lib.deselect", on: {click: () => L.tw.clearSelection()}}),
    h("span.msg", {id: "trayMsg", role: "status"}),
  ].filter(Boolean));
  translate(tray);
}

function ratePop() {
  const ids = L.tw.sel.slice();
  return [
    h("div.hd", {}, h("h3.title", {text: t("lib.rateTitle", {n: ids.length})})),
    h("div.stars", {}, [0, 1, 2, 3, 4, 5].map(r => h("button", {
      type: "button", title: r ? t("lib.stars", {n: r}) : t("lib.noStars"),
      on: {click: async () => {
        closePop();
        const d = await post("/api/rate_bulk", {ids, rating: r}).catch(e => ({error: e.message}));
        if (d.error) { say(t("lib.err", {msg: d.error}), "err"); return; }
        for (const w of [L.tw, L.wall]) for (const it of w.items) if (ids.includes(it.id)) it.rating = r;
        say(r ? t("lib.rated", {n: ids.length, r}) : t("lib.unrated", {n: ids.length}), "ok");
      }}}, r ? "★".repeat(r) : "☆")))];
}

// 搬移／複製之後：結果寫在牆的標題旁——動作列在選取清掉後會收起來，寫在裡面會看不到
// （使用者回報 2026-10-10：生成頁搬移看起來沒反應；搬到圖庫裡的資料夾時圖照樣在牆上，更看不出搬了沒）
function moved(d, mode) {
  note(moveText(d, mode), d.error || d.failed ? "err" : "ok");
  // 搬走的圖可能不再符合目前的篩選，重新載入；複製的新檔要重新掃描才會出現
  if (!d.error && mode === "move") { filesChanged(L.tw); L.tw.reset(); }
}

// 牆標題旁的訊息（動作列收起來也看得到）
function note(text, kind = "") {
  const c = $("libNote");
  if (!c) return;
  c.textContent = text;
  c.dataset.kind = kind;
  clearTimeout(L.noteTimer);
  L.noteTimer = setTimeout(() => { c.textContent = ""; }, kind === "err" ? 12000 : 6000);
}

// 加進資料集：蓋在畫面中間（使用者回饋：從動作列往上長的小視窗，資料集一多就超出畫面）。
// 上面是資料集清單（01 資料集的那些，有封面，多了在框裡捲），下面可以直接打路徑或瀏覽（不存在會建）；圖複製過去、原圖不動
function openDatasetDialog() {
  const ids = L.tw.sel.slice();
  const back = document.activeElement;
  const el = h("div.assets.confirm", {role: "dialog", "aria-modal": "true", "aria-labelledby": "tdsTitle"});
  const close = () => {
    removeEventListener("keydown", onKey, true);
    el.remove();
    document.body.classList.remove("viewing");
    back?.focus?.();
  };
  // 「瀏覽…」開的資料夾視窗蓋在上面時，Esc 是關它的
  const onKey = e => {
    if (e.key === "Escape" && document.body.lastElementChild === el) { e.preventDefault(); e.stopPropagation(); close(); }
  };
  const input = h("input.find", {type: "text", "data-i18n-placeholder": "lib.dest", "data-i18n-aria": "lib.dest", spellcheck: "false",
                                 on: {input: () => mark()}});
  const list = h("div.tdslist", {role: "list"});
  const mark = () => {
    for (const b of list.children) b.setAttribute("aria-pressed", String(b.dataset.path === input.value.trim()));
  };
  get("/api/dataset/library").then(d => {
    const items = (d.items || []).filter(it => !it.missing);
    if (!items.length) { list.replaceChildren(h("p.dim.small", {text: t("lib.noDatasets")})); return; }
    if (!input.value) input.value = items[0].path;
    list.replaceChildren(...items.map(it => h("button.tdsitem", {type: "button", title: it.path, "data-path": it.path,
      on: {click: () => { input.value = it.path; mark(); }, dblclick: () => run.click()}},
      h("span.tdscover", {}, it.cover
        ? h("img", {src: `/api/dataset/thumb?dir=${encodeURIComponent(it.path)}&path=${encodeURIComponent(it.cover)}&v=${it.cover_v}`,
                    alt: "", loading: "lazy"}) : null),
      h("span.tdsname", {}, h("b", {text: it.name}), h("span.small.dim", {text: t("dsl.count", {n: it.n, tagged: it.tagged})})),
      h("span.c.small.dim", {text: it.path}))));
    mark();
  }).catch(() => list.replaceChildren());
  const run = h("button.pill.primary", {type: "button", text: t("lib.doDataset", {n: ids.length}), on: {click: async () => {
    const dir = input.value.trim();
    if (!dir) { input.focus(); return; }
    close();
    const d = await post("/api/dataset/from_gallery", {ids, dir}).catch(e => ({error: e.message}));
    if (d.error) { say(t("lib.err", {msg: d.error}), "err"); return; }
    say(h("span", {}, t("lib.toDatasetDone", {n: d.copied, dest: lastPart(d.dir)}), " ",
      h("a", {href: "#dataset", on: {click: e2 => { e2.preventDefault(); openDataset(d.dir); }}}, t("lib.openDataset"))), "ok");
  }}});
  const [hintBtn, hint] = info("lib.toDatasetHint");
  el.addEventListener("click", e => { if (e.target === el) close(); });
  el.append(h("div.apanel.cfpanel.tdspanel", {},
    h("div.rhd", {}, h("h2.title", {id: "tdsTitle", text: t("lib.toDatasetN", {n: ids.length})}), hintBtn),
    hint,
    list,
    h("div.newset", {}, input, h("button.pill", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
      const got = await pickFolder(input.value.trim());
      if (got) { input.value = got; mark(); input.focus(); }
    }}})),
    h("div.sfoot", {}, h("span.sp"),
      h("button.pill", {type: "button", "data-i18n": "cf.cancel", on: {click: close}}), run)));
  translate(el);
  addEventListener("keydown", onKey, true);
  closePop();
  document.body.append(el);
  document.body.classList.add("viewing");
  run.focus();
}

// 設定改了：動作列開著的話重畫（按鈕在「丟進回收桶」與「刪除」之間換）
export function refreshTray() {
  const tray = $("tray");
  if (L.tw && tray && !tray.hidden) drawTray(L.tw.selected(), L.tw);
}

// 丟進回收桶或刪除（要按兩次、被用到的先問，見 fileacts.js）
async function runTrash(ids, mode) {
  const d = await removeFiles(ids, mode);
  note(trashText(d, mode), d.error || d.failed ? "err" : "ok");
  if (d.error) return;
  L.tw.remove(ids);
  filesChanged(L.tw);
}

// 從回收桶還原了：圖回到圖庫，重新載入
function trashRestored() {
  dropFacets();
  if (location.hash === "#library") { loadFacets(); L.wall.reset(); }
  else L.stale = true;
  galleryChanged();
}

// 圖庫變了（重新掃描、還原、在圖庫空間搬走或丟掉）：工作室的牆也要跟著換（main.js 接；使用者回報：
// 圖庫搬走的圖在生成頁還在，點了說找不到）。從工作室的牆做的，那道牆自己已經更新了
export function galleryChanged(kind = "changed") {
  dispatchEvent(new CustomEvent("coco:gallery", {detail: {kind}}));
}

// 搬走、丟掉之後（動作列或看大圖，wall＝在哪道牆做的）：篩選清單重抓；另一道牆跟著換——在圖庫空間做的叫工作室的牆重讀，
// 在工作室做的讓圖庫空間下次進去時重新載入。看大圖用事件通知（viewer → library 不直接 import，免得繞一圈）
function filesChanged(wall) {
  dropFacets();
  if (wall === L.wall) { loadFacets(); galleryChanged(); }
  else L.stale = true;
}
addEventListener("coco:files", e => filesChanged(e.detail?.wall));

async function rescan(e) {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = t("lib.scanning");
  const d = await post("/api/reindex", {}).catch(err => ({error: err.message}));
  btn.disabled = false;
  btn.textContent = t("lib.rescan");
  const c = $("libNote");
  if (c) c.textContent = d.error ? t("lib.err", {msg: d.error}) : d.msg || "";
  dropFacets();
  loadFacets();
  L.wall.reset();
  galleryChanged();
}

// ── 組起來 ──
export function libraryScreen() {
  const w = L.wall;
  return h("div.lib", {},
    h("div.filters", {id: "libFilters", role: "toolbar", "data-i18n-aria": "lib.filters"}),
    h("div.libwall", {},
      h("section.card.wall", {"aria-labelledby": "libTitle"},
        h("div.whd", {},
          h("h2.title", {id: "libTitle", "data-i18n": "space.library"}),
          h("span.dim.mono", {id: "libCount", text: w.total ? w.countText() : ""}),
          h("span.dim", {id: "libNote", role: "status"}),
          h("span.sp"),
          // 排法、大小、重新掃描、回收桶收進一顆（使用者選的：頂端只留搜尋與排序）；跟牆的排法小視窗同一個 id，換排法時亮的那格會跟著換
          h("button.round", {type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", "data-i18n-aria": "lib.more",
                             "data-i18n-title": "lib.more", on: {click: e => openPop(`popView-${w.name}`, e.currentTarget, morePop)}}, "⋯")),
        w.box, w.end, w.top),
      h("div.tray", {id: "tray", hidden: true, role: "region", "data-i18n-aria": "lib.trayAria"})));
}

function morePop() {
  return [...L.wall.viewPop(),
    h("div.row", {}, h("span.lbl", {"data-i18n": "lib.manage"}),
      h("div.rowbtns", {},
        h("button.pill", {type: "button", "data-i18n": "lib.rescan", on: {click: rescan}}),
        h("button.pill", {type: "button", "aria-haspopup": "dialog", "data-i18n": "trash.title", "data-i18n-title": "trash.open",
                          on: {click: () => { closePop(); openTrash(trashRestored); }}})))];
}

// 畫面組好之後呼叫：篩選列與動作列要依目前狀態填內容
export function afterRender() {
  LF.draw();
  drawTray(L.wall.selected());
}

document.addEventListener("keydown", e => {
  // Esc：沒有開著的小視窗時，取消選取
  if (e.key === "Escape" && !e.defaultPrevented && L.tw?.sel.length && $("tray")) L.tw.clearSelection();
  // 以下只在圖庫空間、沒有在打字、沒有開著看大圖之類的疊層時
  if (e.defaultPrevented || location.hash !== "#library" || !L.wall) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable]") || document.body.classList.contains("viewing")) return;
  if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    $("libSearch")?.focus();
    $("libSearch")?.select();
  } else if (e.key.toLowerCase() === "a" && (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey) {
    e.preventDefault();                  // 不要選到整頁的文字
    L.wall.selectAll();
    const note = $("libNote");
    if (note) {
      const msg = note.textContent = t("lib.selectAllNote", {n: L.wall.sel.length.toLocaleString(lang())});
      setTimeout(() => { if (note.textContent === msg) note.textContent = ""; }, 4000);
    }
  }
});
