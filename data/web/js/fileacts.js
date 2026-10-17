// 搬移／複製、丟進回收桶（或刪除）：圖庫的動作列與看大圖共用（使用者回報 2026-10-10：看大圖沒有這兩個）。
// 這裡只負責小視窗、確認與送出；結果怎麼顯示、牆怎麼更新由呼叫的人決定（動作列在選取清掉後會收起來，
// 訊息要放在收不起來的地方）。

import {post} from "./api.js";
import {h} from "./dom.js";
import {t} from "./i18n.js";
import {info} from "./hint.js";
import {closePop} from "./pop.js";
import {confirmDialog} from "./confirm.js";
import {pickFolder} from "./setup.js";
import {genSettings} from "./generate.js";
import {lastPart} from "./filters.js";

// ⚙ 設定 → 其他 → 直接刪除圖片：開了之後「丟進回收桶」變成「刪除」（檔案直接刪掉，無法復原）
export const hardDelete = () => genSettings()?.HARD_DELETE === true;

// ── 搬移／複製 ──
// ui：介面偏好（dests＝最近用過的目的地，伺服器那邊也會記）；done(d, mode)：伺服器回來之後（失敗時 d.error）
export function movePop(ids, ui, done) {
  let mode = "move";
  const input = h("input.find", {id: "moveDest", type: "text", "data-i18n-placeholder": "lib.dest", "data-i18n-aria": "lib.dest",
                                 spellcheck: "false"});
  const run = h("button.pill.primary", {type: "button"});
  const label = () => { run.textContent = t(mode === "move" ? "lib.doMove" : "lib.doCopy", {n: ids.length}); };
  label();
  run.addEventListener("click", async () => {
    const dest = input.value.trim();
    if (!dest) { input.focus(); return; }
    closePop();
    const d = await post("/api/files", {mode, ids, dest}).catch(e => ({error: e.message}));
    if (!d.error) ui.dests = [d.dest, ...(ui.dests || []).filter(x => x !== d.dest)].slice(0, 8);
    done(d, mode);
  });
  // 打完路徑習慣按 Enter：當成按了「搬移」（以前沒反應，看起來像搬了）
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); run.click(); } });
  return [
    h("div.hd", {}, h("h3.title", {"data-i18n": "lib.moveTitle"})),
    h("div.seg", {role: "group"}, ["move", "copy"].map(m => h("button", {
      type: "button", "aria-pressed": String(m === mode), "data-i18n": `lib.mode.${m}`,
      on: {click: e => {
        mode = m;
        e.currentTarget.parentElement.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
        label();
      }}}))),
    h("div.newset", {}, input, h("button.pill", {type: "button", "data-i18n": "set.browse", on: {click: async () => {
      const got = await pickFolder(input.value.trim() || (ui.dests || [])[0] || "");
      if (got) { input.value = got; input.focus(); }
    }}})),
    (ui.dests || []).length ? h("div.row", {}, h("span.lbl", {"data-i18n": "lib.recent"}),
      h("div.plist", {}, ui.dests.map(d => h("button", {type: "button", title: d, on: {click: () => { input.value = d; input.focus(); }}},
        h("span", {text: lastPart(d)}), h("span.c", {text: d}))))) : null,
    ...(([btn, hint]) => [h("div.infoline", {}, run, btn), hint])(info("lib.moveHint"))];
}

// 結果的一句話。失敗的附上第一個原因（例如檔案正被別的程式開著），不然只看到「N 張失敗」不知道怎麼辦
export function moveText(d, mode) {
  if (d.error) return t("lib.err", {msg: d.error});
  return t(mode === "move" ? "lib.moved" : "lib.copied", {n: d.done, dest: lastPart(d.dest)}) + failText(d);
}

const failText = d => (d.failed ? " · " + t("lib.failed", {n: d.failed}) + (d.errors?.[0] ? `（${d.errors[0]}）` : "") : "");

// ── 丟進回收桶／刪除 ──
// 要按兩次：第一次變成「確定？」，4 秒內再按才執行。照第一次按下時按鈕上寫的做：中間設定改了也不會變成另一種。
// 選的圖有被用到（比較組、重組的來源、封面、精修的底圖，server/usage.py）：改跳確認視窗，
// 按「繼續」就直接做，不必再按一次（使用者筆記）。run(ids, mode)：真的去做
const ARMED = new WeakMap();
export const trashLabel = () => (hardDelete() ? "lib.delete" : "lib.trash");

export async function trashClick(btn, ids, run) {
  const armed = ARMED.get(btn);
  if (armed) {
    clearTimeout(armed.timer);
    ARMED.delete(btn);
    disarm(btn, armed.mode);
    await run(ids, armed.mode);
    return;
  }
  const mode = hardDelete() ? "delete" : "trash";
  const u = await post("/api/files/usage", {ids}).catch(() => ({items: []}));
  if (u.items?.length) {
    if (await confirmUse(u.items, mode, ids.length)) await run(ids, mode);
    return;
  }
  btn.textContent = t(mode === "delete" ? "lib.deleteSure" : "lib.trashSure", {n: ids.length});
  btn.classList.add("armed");
  ARMED.set(btn, {mode, timer: setTimeout(() => { ARMED.delete(btn); disarm(btn, mode); }, 4000)});
}

function disarm(btn, mode) {
  btn.textContent = t(mode === "delete" ? "lib.delete" : "lib.trash");
  btn.classList.remove("armed");
}

// 被用到的圖：列出在哪裡用到、丟掉之後會怎樣，取消或繼續
function confirmUse(items, mode, total) {
  const del = mode === "delete";
  const what = u => u.kind === "compare" ? t(u.winner ? "use.compareWin" : "use.compare", {name: u.name || t("cmp.untitled")})
    : u.kind === "recipe" ? t("use.recipe", {name: u.name || t("rcp.untitled")})
    : u.kind === "cover" ? t("use.cover", {kind: t(`use.kind.${u.res}`), name: u.name})
    : t("use.refine");
  return confirmDialog({
    title: t("use.title", {n: items.length}),
    body: [
      h("ul.uselist", {}, items.map(it => h("li", {},
        h("img", {src: `/api/thumb?id=${it.id}`, alt: ""}),
        h("span", {}, h("b", {text: it.filename}), h("span.dim", {text: it.uses.map(what).join("、")}))))),
      h("p.hint", {text: t(del ? "use.afterDelete" : "use.afterTrash")})],
    ok: t(del ? "use.goDelete" : "use.goTrash", {n: total}),
  });
}

export function removeFiles(ids, mode) {
  return post("/api/files", {mode, ids}).catch(e => ({error: e.message}));
}

export function trashText(d, mode) {
  if (d.error) return t("lib.err", {msg: d.error});
  return t(mode === "delete" ? "lib.deleted" : "lib.trashed", {n: d.done}) + failText(d)
    + (d.cleaned?.items ? " · " + t(d.cleaned.sets ? "use.cleanedSets" : "use.cleaned", {n: d.cleaned.items, sets: d.cleaned.sets}) : "");
}
