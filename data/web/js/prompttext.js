// 提示詞 ↔ 標籤清單。標籤模式只是提示詞的另一種畫法，真正存的還是那段文字，
// 所以兩個方向都要能來回轉而不走樣。這個模組不碰 DOM，測試直接用 node 跑。
//
// 標籤的種類（kind）：
//   tag      一般標籤，可以有權重：(x:1.2)、((x))、[x]
//   lora     <lora:名稱:權重>
//   cart     [[a|b|c]] 每個都跑一次，畫成「N 選 1」的一組
//   rand     {a|b|c} 隨機挑一個
//   tpl      其他動態語法（__萬用字元__、${變數}、{2$$a|b}…），只能整段改字
//   comment  # 到行尾是註解，寫回去時自己佔一行

const OPEN = "{([<", CLOSE = "})]>";

// 依巢狀深度切逗號：{a,b}、(x:1.2)、<lora:…> 裡面的逗號不是分隔
export function splitTags(text) {
  const out = [];
  let buf = "", d = 0;
  for (const c of text || "") {
    if (OPEN.includes(c)) d++;
    else if (CLOSE.includes(c)) d = Math.max(0, d - 1);
    if ((c === "," || c === "\n") && d === 0) { out.push(buf); buf = ""; }
    else buf += c;
  }
  out.push(buf);
  return out.map(s => s.trim()).filter(Boolean);
}

// 依 | 拆選項，同樣要顧巢狀（[[ ]] 算一層）
export function splitOpts(text) {
  const out = [];
  let buf = "", d = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (text.startsWith("[[", i)) { d++; buf += "[["; i++; continue; }
    if (text.startsWith("]]", i)) { d = Math.max(0, d - 1); buf += "]]"; i++; continue; }
    if (c === "{" || c === "(" || c === "<") d++;
    else if (c === "}" || c === ")" || c === ">") d = Math.max(0, d - 1);
    if (c === "|" && d === 0) { out.push(buf.trim()); buf = ""; }
    else buf += c;
  }
  out.push(buf.trim());
  return out;
}

const round = (w, n = 1000) => Math.round(w * n) / n;

export function parseChip(seg) {
  seg = seg.trim();
  if (seg.startsWith("#")) return {kind: "comment", text: seg};
  if (/^<lora:[^>]*>$/i.test(seg)) return {kind: "lora", text: seg};
  // [[a|b]]：選項互斥、不會同時出現，所以畫成一組，並排會讓人以為都生效
  let m = seg.match(/^\[\[([\s\S]*)\]\]$/);
  if (m && splitOpts(m[1]).length > 1) return {kind: "cart", opts: splitOpts(m[1])};
  // {a|b}：{2$$a|b}、{3-5$$…}、{set:…}、{get:…} 是別的語法，不拆
  m = seg.match(/^\{([\s\S]*)\}$/);
  if (m && splitOpts(m[1]).length > 1 && !/^\s*(set:|get:|\d+\s*(-\s*\d+)?\s*\$\$)/.test(m[1]))
    return {kind: "rand", opts: splitOpts(m[1])};
  if (/\{|\[\[|__|\$\{|<|\|/.test(seg)) return {kind: "tpl", text: seg};
  // A1111 的權重：( ) ×1.1、[ ] ÷1.1、(x:N) 直接指定，可以巢狀相乘。\( 是跳脫的括號，不算
  let s = seg, w = 1;
  for (;;) {
    m = s.match(/^\(([\s\S]*):\s*(-?[\d.]+)\s*\)$/);
    if (m && !m[1].endsWith("\\") && balanced(m[1])) {
      const n = parseFloat(m[2]);              // (x:0) 是明確的 0，不能當成沒寫（以前 || 1 把它變成 1，寫回去成了 x）
      w *= Number.isFinite(n) ? n : 1;
      s = m[1].trim();
      continue;
    }
    if (s.length > 1 && s[0] === "(" && s.endsWith(")") && !s.endsWith("\\)") && balanced(s.slice(1, -1))) { w *= 1.1; s = s.slice(1, -1).trim(); continue; }
    if (s.length > 1 && s[0] === "[" && s.endsWith("]") && !s.endsWith("\\]") && balanced(s.slice(1, -1))) { w /= 1.1; s = s.slice(1, -1).trim(); continue; }
    break;
  }
  return {kind: "tag", text: s, weight: round(w)};
}

// (a) (b) 這種兩組並排的不能剝外層括號
function balanced(s) {
  let d = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "\\") { i++; continue; }
    if (s[i] === "(" || s[i] === "[") d++;
    else if (s[i] === ")" || s[i] === "]") { d--; if (d < 0) return false; }
  }
  return d === 0;
}

export function chipText(c) {
  if (c.kind === "cart") return `[[${c.opts.join("|")}]]`;
  if (c.kind === "rand") return `{${c.opts.join("|")}}`;
  if (c.kind !== "tag") return c.text;
  const w = round(c.weight, 100);
  return Math.abs(w - 1) < 0.005 ? c.text : `(${c.text}:${w})`;
}

// 一整段提示詞 → 標籤清單。註解（# 到行尾）先切出來，其餘照逗號切
export function parsePrompt(text) {
  const out = [];
  for (const line of (text || "").split(/\r?\n/)) {
    const at = commentAt(line);
    out.push(...splitTags(at < 0 ? line : line.slice(0, at)).map(parseChip));
    if (at >= 0 && line.slice(at).trim()) out.push({kind: "comment", text: line.slice(at).trim()});
  }
  return out;
}

// 跟後端（core/template.py 的 _re_comment）一樣：# 到行尾都是註解，不管在哪裡
const commentAt = line => line.indexOf("#");

// 標籤清單 → 提示詞。註解後面一定要換行，不然接在後面的標籤也會被當成註解
export function joinPrompt(list) {
  let out = "";
  for (const c of list) {
    const s = chipText(c);
    if (!s) continue;
    if (c.kind === "comment") out += (out && !out.endsWith("\n") ? "\n" : "") + s + "\n";
    else out += (out && !out.endsWith("\n") ? ", " : "") + s;
  }
  return out.replace(/\n$/, "");
}

// 標籤庫的寫法（blue_eyes、gap_(touhou)）→ 提示詞的寫法。
// 括號一定要跳脫，不然 (touhou) 會被當成權重 1.1。score_ 開頭的模型認底線版，不換。
// 提示詞框的自動完成：游標所在的那個字（逗號、括號、| 之後到游標，加上游標後面同一個字的其餘部分）。
// 中文的逗號、頓號也是分隔；中文一個字就查（打中文找英文標籤，伺服器 prompts._zh_search），英文要兩個字。
// 回傳 {start, end, q} 或 null（註解、<lora:…> 裡、太短、數字）
const CJK = /[\u3400-\u9fff\uf900-\ufaff]/;

export function tokenAt(value, pos) {
  const before = value.slice(0, pos);
  const line = before.slice(before.lastIndexOf("\n") + 1);
  if (line.includes("#")) return null;                                   // 註解裡
  if (before.lastIndexOf("<") > before.lastIndexOf(">")) return null;    // <lora:…> 裡
  const seg = s => s.match(/([^,，、|{}[\]()<>:]*)$/)[1];
  let from = line.length - seg(line).length;
  // 緊貼在字後面的 ( 是名字的一部分（gap_(touhou)、gap \(touhou\)），不是權重的括號
  while (from > 1 && line[from - 1] === "(" && /[\w\\]/.test(line[from - 2])) from -= 1 + seg(line.slice(0, from - 1)).length;
  const raw = line.slice(from);
  const lead = raw.length - raw.trimStart().length;
  const word = raw.trimStart();
  if ((word.length < 2 && !CJK.test(word)) || /^(__|\$|[\d.]+$)/.test(word)) return null;
  const tail = value.slice(pos).match(/^[^,，、|{}[\]()<>:\n]*/)[0];    // 游標後面同一個字的其餘部分
  return {start: pos - raw.length + lead, end: pos + tail.length, q: (word + tail.trimEnd()).replace(/\\/g, "")};
}

// 自動完成選了標籤：換掉那個字，後面不是分隔符就補「, 」。
// 打中文時旁邊的「，」「、」換成英文逗號（提示詞裡不留中文標點）。回傳 {value, caret}
export function insertTag(value, tok, text) {
  const after = value.slice(tok.end);
  const suffix = /^\s*[,，、)\]}|:>]/.test(after) || /^\s*\n/.test(after) ? "" : ", ";
  const rest = suffix ? after.replace(/^\s+/, "") : after.replace(/^\s*[，、]\s*/, ", ");
  const head = value.slice(0, tok.start).replace(/[，、]\s*$/, ", ");
  return {value: head + text + suffix + rest, caret: head.length + text.length + suffix.length};
}

export function tagOut(name) {
  const s = /^score_/i.test(name) ? name : name.replace(/_/g, " ");
  return s.replace(/([()])/g, "\\$1");
}

// ── 精修的「一鍵移除」：從提示詞拿掉提到某樣東西的標籤、加進負面提示詞 ──
// 不經過 parsePrompt／joinPrompt：那樣會把使用者分好的行、逗號後面有沒有空白都改掉。
// 這裡一行一行、照最外層的逗號切開，只拿掉要拿的那幾段，其他原封不動

// 比對用的寫法：小寫、底線當空白、權重括號拿掉
const plain = s => String(s).toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ").trim();

// 某個標籤有沒有提到 phrase（整個字詞比對：bottle 對得到 perfume bottle，對不到 bottles 以外的 bottlecap）
const mentions = (tag, phrase) => {
  const esc = plain(phrase).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-z0-9])${esc}([^a-z0-9]|$)`).test(plain(tag));
};

// 最外層的逗號切開，每段保留原本的樣子（含前後空白），接回去就跟原本一樣
function rawParts(line) {
  const out = [];
  let buf = "", d = 0;
  for (const c of line) {
    if (OPEN.includes(c)) d++;
    else if (CLOSE.includes(c)) d = Math.max(0, d - 1);
    if (c === "," && d === 0) { out.push(buf); buf = ""; }
    else buf += c;
  }
  out.push(buf);
  return out;
}

// 回傳 {prompt, removed: [拿掉的標籤（原本的寫法）]}。只拿一般標籤；LoRA、動態語法、註解不動。
// 整行都拿光了（只剩逗號）的那一行也拿掉。
// exact：整個標籤相同才拿（資源庫的標籤分頁「再點一次拿掉」：拿掉 red eyes 不能連 dark red eyes 一起拿），
// 這時 LoRA 這類非一般標籤的整段寫法相同也拿
export function dropTags(prompt, phrases, exact = false) {
  const want = (phrases || []).map(plain).filter(Boolean);
  const removed = [];
  if (!want.length) return {prompt: String(prompt || ""), removed};
  const lines = [];
  for (const line of String(prompt || "").split(/\r?\n/)) {
    const at = commentAt(line);
    const body = at < 0 ? line : line.slice(0, at), tail = at < 0 ? "" : line.slice(at);
    const parts = rawParts(body);
    const keep = parts.filter(p => {
      const c = p.trim() ? parseChip(p) : null;
      const hit = !c ? false : exact ? want.includes(plain(c.kind === "tag" ? c.text : p))
        : c.kind === "tag" && want.some(w => mentions(c.text, w));
      if (hit) { removed.push(p.trim()); return false; }
      return true;
    });
    if (keep.length === parts.length) { lines.push(line); continue; }
    const text = keep.join(",").replace(/^[\s,]+/, "");
    if (text.replace(/[\s,]/g, "") || tail) lines.push(text + tail);
  }
  return {prompt: lines.join("\n"), removed};
}

// 加到最後面（已經有的不重複加）。原本結尾的逗號照留
export function addTags(text, tags) {
  const s = String(text || "");
  const have = parsePrompt(s).filter(c => c.kind === "tag").map(c => plain(c.text));
  const add = [...new Set((tags || []).map(t => t.trim()).filter(t => t && !have.includes(plain(t))))];
  if (!add.length) return s;
  const trail = /,\s*$/.test(s);
  const base = s.replace(/[\s,]*$/, "");
  return `${base}${base ? (/\n/.test(base) ? ",\n" : ", ") : ""}${add.join(", ")}${trail ? "," : ""}`;
}

// 提示詞裡有沒有這個標籤（整個標籤相同、不管權重；LoRA 這類照整段寫法比）
export function hasTag(text, tag) {
  const want = plain(parseChip(tag).kind === "tag" ? parseChip(tag).text : tag);
  return parsePrompt(text).some(c => plain(c.kind === "tag" ? c.text : chipText(c)) === want);
}

// [[ ]] 會相乘，先讓人看到總數
export function cartTotal(list) {
  const carts = list.filter(c => c.kind === "cart");
  return {parts: carts.map(c => c.opts.length), total: carts.reduce((a, c) => a * c.opts.length, 1)};
}


// ── 暫時關掉的標籤（標籤模式的 ⊘）─────────────────────
// 關掉的不在提示詞裡（純文字模式看不到、不會生成），另外記 [{text, after}]：after＝它前面那個開著的標籤（或組）寫成提示詞的樣子

const itemText = x => joinPrompt([x]).trim();

// 標籤清單裡關掉的（off）→ 要記的 [{text, after}]
export function offOf(list) {
  const out = [];
  let prev = "";
  for (const c of list) {
    if (c.off) out.push({text: itemText(c), after: prev});
    else prev = itemText(c);
  }
  return out;
}

// 開著的標籤清單＋記的關掉的 → 放回原位的完整清單（關掉的標 off）。跟在後面的那個標籤不見了（純文字模式改過）就放到最後；
// after 空白的放最前面；同一個位置好幾個照原本的順序
export function withOff(list, off) {
  const out = list.slice();
  for (const m of off || []) {
    const items = parsePrompt(m.text || "");
    if (!items.length) continue;
    for (const x of items) x.off = true;
    let at = m.after ? out.findIndex(x => !x.off && itemText(x) === m.after) : -1;
    at = m.after && at < 0 ? out.length : at + 1;
    while (out[at]?.off) at++;
    out.splice(at, 0, ...items);
  }
  return out;
}


// ── 動態語法輸入助手（使用者要求：以前要打開語法視窗記下寫法、關掉再回來打）─────────
// 游標前面正在打萬用字元（__名字，前面的 __ 都成對才算新開的）或變數（$名字、${名字）→ {kind, start, end, q}；不是回 null。
// end 包含游標後面同一個名字的其餘部分與收尾（__ 或 }），換掉時不會留半截
export function syntaxAt(value, pos) {
  const before = value.slice(0, pos);
  const line = before.slice(before.lastIndexOf("\n") + 1);
  if (line.includes("#")) return null;                                   // 註解裡
  const w = line.match(/__([A-Za-z0-9_\-/. ]*)$/);
  if (w && !w[1].includes("__") && (line.slice(0, line.length - w[0].length).split("__").length - 1) % 2 === 0) {
    const tail = value.slice(pos).match(/^[A-Za-z0-9_\-/.]*(?:__)?/)[0];
    return {kind: "wild", start: pos - w[0].length, end: pos + tail.length, q: w[1].trim()};
  }
  const v = line.match(/\$\{?([A-Za-z0-9_-]*)$/);
  if (v) {
    const tail = value.slice(pos).match(/^[A-Za-z0-9_-]*\}?/)[0];
    return {kind: "var", start: pos - v[0].length, end: pos + tail.length, q: v[1]};
  }
  return null;
}

// 選起來的字包成 [[a|b]]（每個各生一組）或 {a|b}（隨機一個）：逗號、頓號分開的變成選項；沒選字就放一個空的
export function wrapSyntax(text, kind) {
  const opts = String(text || "").split(/[,，、|]/).map(s => s.trim()).filter(Boolean);
  const body = opts.length ? opts.join("|") : "|";
  return kind === "cart" ? `[[${body}]]` : `{${body}}`;
}

// 提示詞的 token 數怎麼顯示（生成台底下那排的「54/77」）：f 是後端數的那一組 {n, limit, over}，mode 是 native 或 a1111。
// native 超過上限會直接截斷（over＞0 要變色）；a1111 每 75 個一段、不截斷，上限跟著段數長（1 段 77、2 段 152）
export function tokenLabel(f, mode) {
  if (mode === "native") return {n: f.n, cap: f.limit, over: Math.max(0, f.over || 0), chunks: 1};
  const per = Math.max(1, f.limit - 2), chunks = Math.max(1, Math.ceil(Math.max(0, f.n - 2) / per));
  return {n: f.n, cap: chunks * per + 2, over: 0, chunks};
}
