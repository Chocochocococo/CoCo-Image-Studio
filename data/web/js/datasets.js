// 訓練空間的 01 資料集（server/dataset.py 的 library，像 OneTrainer 的 concept 清單）：每個資料集一張卡片——
// 封面（沒選就是第一張）、名字、張數、有標籤的張數、觸發詞、每張重複幾次、訓練過幾次、備註。
// 點卡片 → 02 打標；「＋ 新增資料集」選資料夾（可以新建）；× 從清單移除（按兩次，資料夾裡的檔案不動）。

import {get, post} from "./api.js";
import {$, h} from "./dom.js";
import {t, translate} from "./i18n.js";
import {info} from "./hint.js";
import {pickFolder} from "./setup.js";
import {openDataset} from "./dataset.js";

const L = {items: [], armed: "", armTimer: 0, msg: ""};
const enc = encodeURIComponent;

export function datasetsScreen() {
  return h("main.dslib", {}, h("section.card.dslibcard", {id: "dslPanel", "aria-labelledby": "dslTitle"}));
}

export async function showDatasets() {
  if (!$("dslPanel")) return;
  const d = await get("/api/dataset/library").catch(e => ({error: e.message}));
  L.items = d.items || [];
  L.msg = d.error || "";
  draw();
}

function draw() {
  const box = $("dslPanel");
  if (!box) return;
  const [howBtn, howHint] = info("dsl.howTo");
  box.replaceChildren(...[         // replaceChildren 不會略過 null（會變成「null」字樣）
    h("div.rhd", {}, h("h2.title", {id: "dslTitle", "data-i18n": "nav.datasets"}), howBtn, h("span.sp"),
      h("span.msg", {role: "status", "data-kind": L.msg ? "err" : null, text: L.msg})),
    howHint,
    L.items.length ? null : h("p.hint", {"data-i18n": "dsl.none"}),
    h("div.dslgrid", {role: "list"}, ...L.items.map(card), addCard())].filter(Boolean));
  translate(box);
}

function card(it) {
  const open = () => { if (!it.missing) openDataset(it.path); };
  const cover = it.cover
    ? h("img", {src: `/api/dataset/thumb?dir=${enc(it.path)}&path=${enc(it.cover)}&v=${it.cover_v}`, alt: "", loading: "lazy"})
    : h("span.dim.small", {"data-i18n": it.missing ? "dsl.missing" : "dsl.noImages"});
  const last = it.runs?.[0];
  const armed = L.armed === it.path;
  return h("div.dslcard", {role: "listitem", "data-missing": it.missing ? "1" : null},
    h("button.dslopen", {type: "button", title: it.path, disabled: it.missing, on: {click: open}},
      h("div.dslcover", {}, cover),
      h("div.dslbody", {},
        h("b.dslname", {text: it.name}),
        it.missing ? h("span.small.warn", {"data-i18n": "dsl.missing"})
          : h("span.small.dim", {text: t("dsl.count", {n: it.n, tagged: it.tagged})}),
        it.trigger ? h("span.small.mono.dsltrig", {text: it.trigger}) : null,
        it.repeats ? h("span.small.dim", {text: t("dsl.repeats", {n: it.repeats})}) : null,
        last ? h("span.small.dim", {text: t("dsl.runs", {n: it.n_runs, name: last.name, date: String(last.date || "").slice(0, 10)})}) : null,
        it.note ? h("span.small.dim.dslnote", {text: it.note}) : null)),
    h("button.mini.dslrm", {type: "button", "data-armed": armed ? "1" : null, "data-i18n-title": "dsl.remove",
                            "aria-label": armed ? t("dsl.removeSure") : t("dsl.remove"),
                            on: {click: () => remove(it)}}, armed ? t("dsl.removeSure") : "×"));
}

// 從清單移除：按兩次（第一次變成「確定移除？」，4 秒內再按才移除）
async function remove(it) {
  clearTimeout(L.armTimer);
  if (L.armed !== it.path) {
    L.armed = it.path;
    L.armTimer = setTimeout(() => { L.armed = ""; draw(); }, 4000);
    draw();
    return;
  }
  L.armed = "";
  const d = await post("/api/dataset/library", {op: "remove", dir: it.path}).catch(e => ({error: e.message}));
  if (d.error) L.msg = d.error;
  await showDatasets();
}

function addCard() {
  return h("button.dslcard.dsladd", {type: "button", on: {click: async () => {
    const got = await pickFolder("");
    if (!got) return;
    const d = await post("/api/dataset/library", {op: "add", dir: got}).catch(e => ({error: e.message}));
    if (d.error) { L.msg = d.error; draw(); return; }
    openDataset(d.dir);          // 加好直接去打標
  }}}, h("span.dsladdplus", {text: "＋"}), h("span", {"data-i18n": "dsl.add"}));
}
