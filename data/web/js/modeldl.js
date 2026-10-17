// 設定 → 資源資料夾 → 下載模型：試用模型（Anima Aesthetic）與拆分式模型缺少的 VAE、文字編碼器（server/modeldl.py）。
// 按了才下載；第一次按顯示大小請使用者再按一次確定。下載中每秒更新進度，可以取消
import {get, post} from "./api.js";
import {h} from "./dom.js";
import {t, translate} from "./i18n.js";

const D = {el: null, st: null, poll: 0, arm: ""};

export function downloadsBox() {
  D.el = h("div.dlbox", {});
  D.arm = "";
  D.st = null;                                          // 重新打開：上次的狀態不算，refresh 才會去讀
  refresh();
  return D.el;
}

// 上次下載到幾 %（下載了一點點也不說 0%）
export const partPct = (part, total) => Math.max(1, Math.floor(part / Math.max(1, total) * 100));

export const size = b => b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 2 ** 20))} MB`;

async function refresh() {
  clearTimeout(D.poll);
  if (D.st && !D.el?.isConnected) return;               // 設定關掉、換分頁了就不再更新
  D.st = await get("/api/modeldl/status").catch(e => ({error: e.message}));
  draw();
  const all = [D.st.trial?.download, ...(D.st.parts || []).map(p => p.download)];
  if (all.some(d => d?.running)) D.poll = setTimeout(refresh, 1000);
}

async function start(job) {
  D.arm = "";
  await post("/api/modeldl/download", {job}).catch(() => null);
  refresh();
}

// 下載按鈕：第一次按變成「確定下載 X？」（上次下載到一半的話是剩下的大小），再按才開始；下載中顯示百分比與取消
function action(job, bytes, dl, done, part = 0) {
  if (done) return h("span.dim.small", {"data-i18n": "dl.have"});
  if (dl?.running) {
    const pct = dl.total ? Math.floor(dl.done / dl.total * 100) : 0;
    return h("span.cnprog", {}, h("span.mono.small", {text: `${pct}%`}),
      h("button.mini", {type: "button", "data-i18n": "cf.cancel", on: {click: () => post("/api/modeldl/cancel", {job}).then(refresh)}}));
  }
  const sure = part ? t("dl.sureRest", {size: size(Math.max(0, bytes - part))}) : t("dl.sure", {size: size(bytes)});
  return h("button.mini.dlgo", {type: "button", text: D.arm === job ? sure : t("dl.download"),
                           on: {click: () => { if (D.arm === job) start(job); else { D.arm = job; draw(); } }}});
}

const license = f => h("a.small", {href: f.license_url, target: "_blank", rel: "noopener", text: f.license});

// 錯誤訊息；沒有錯誤但上次下載到一半（關掉 CoCo 再打開）時說下載到哪裡（錯誤訊息本身已經說會接著下載）
function note(dl, bytes, part) {
  if (dl.error) return h("span.hint.warn", {text: dl.error});
  if (part && !dl.running) return h("span.dim.small", {text: t("dl.partial", {pct: partPct(part, bytes)})});
  return null;
}

function trialRow(tr) {
  const dl = tr.download || {};
  const done = tr.model.have && !tr.files.length;
  const part = tr.files.reduce((n, f) => n + f.part, 0);
  return h("div.cnpkg", {},
    h("div.cnpkgtx", {},
      h("b", {"data-i18n": "dl.trial"}),
      h("span.dim.small", {"data-i18n": "dl.trialHint"}),
      h("span.dim.small", {}, done ? t("dl.trialDone") : t("dl.need", {files: tr.files.map(f => f.name).join("、"), size: size(tr.bytes)}),
        " · ", license(tr.model), " · ", h("a.small", {href: tr.model.page, target: "_blank", rel: "noopener", "data-i18n": "dl.page"})),
      note(dl, tr.bytes, part),
      dl.running && dl.file ? h("span.dim.small", {text: dl.file}) : null),
    action("trial", tr.bytes, dl, done, part));
}

function partCell(p) {
  if (p.found.length) return h("div.dlpart", {}, h("span.dim.small", {text: t("dl.found", {label: p.label, names: p.found.join("、")})}));
  const f = p.file, dl = p.download || {};
  return h("div.dlpart.miss", {},                       // 左邊說明與訊息上下排，右邊按鈕（訊息長也不會把按鈕擠到下一行）
    h("div.dltx", {},
      h("span.small", {}, t("dl.missing", {label: p.label}), " ", h("b", {text: f.name}), ` · ${size(f.bytes)} · `, license(f)),
      note(dl, f.bytes, f.part)),
    action(f.key, f.bytes, dl, f.have, f.part));
}

function draw() {
  if (!D.el) return;
  const st = D.st || {};
  if (st.error) { D.el.replaceChildren(h("p.hint.warn", {text: st.error})); return; }
  if (!st.trial) { D.el.replaceChildren(h("p.dim.small", {"data-i18n": "dl.loading"})); translate(D.el); return; }
  const archs = [...new Set(st.parts.map(p => p.arch))];
  D.el.replaceChildren(
    trialRow(st.trial),
    h("h4", {"data-i18n": "dl.parts"}),
    h("p.dim.small", {"data-i18n": "dl.partsHint"}),
    h("div.dlparts", {}, archs.map(a => {
      const ps = st.parts.filter(p => p.arch === a);
      return h("div.cnpkg.dlarch", {}, h("b.dlname", {text: ps[0].title}), h("div.dlcells", {}, ps.map(partCell)));
    })),
    h("p.dim.small", {text: t("dl.to", st.dirs)}));
  translate(D.el);
}
