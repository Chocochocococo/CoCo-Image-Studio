// 生成參數的選項表。名稱是專有名詞，兩種語言都不翻。
//
// 取樣器與排程器的名字必須跟 core/pipeline.py 的 SAMPLERS / SCHEDULES /
// FLOW_ONLY 一致，tests/test_next.py 會比對，後端加了新的這裡沒加就會失敗。

// core/pipeline.py detect_arch() 的回傳值 → 顯示名稱
export const ARCH_LABEL = {sdxl: "SDXL", anima: "Anima", zimage: "Z-Image", krea2: "Krea 2", qwen21: "Qwen Image 2.1",
                           klein4: "FLUX.2 klein 4B", klein9: "FLUX.2 klein 9B", flux1: "FLUX.1 dev", flux1s: "FLUX.1 schnell",
                           kontext: "FLUX.1 Kontext",
                           sd15: "SD 1.5"};

// flow matching 的架構：「只支援 Anima」那兩組取樣器也能用、有 shift、沒有 CLIP skip
export const FLOW_ARCHS = ["anima", "zimage", "krea2", "qwen21", "klein4", "klein9", "flux1", "flux1s", "kontext"];
// shift 一個架構記一格（Anima 3.0、Z-Image 6.0、Krea 2 3.16、Qwen Image 2.1 2.0、FLUX.2 klein 9.89（4B、9B 共用）、
// FLUX.1 dev 3.16、schnell 1，伺服器 handler.SHIFT_FIELDS 同一張表）
const SHIFT_KEYS = {zimage: "ZIMAGE_SHIFT", krea2: "KREA2_SHIFT", qwen21: "QWEN21_SHIFT", klein4: "FLUX2_SHIFT", klein9: "FLUX2_SHIFT",
                    flux1: "FLUX1_SHIFT", flux1s: "FLUX1S_SHIFT", kontext: "FLUX1_SHIFT"};
export const shiftKey = arch => SHIFT_KEYS[arch] || "ANIMA_SHIFT";

// 分組只是為了好找：舊介面一整排二十幾個，常用的被埋在中間。
// anima 組只支援 flow-match 模型（Anima、Z-Image、Krea 2、Qwen Image 2.1），其他架構選了後端會略過。
export const SAMPLER_GROUPS = [
  ["common", ["Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ SDE", "DPM++ 3M SDE", "UniPC", "DDIM"]],
  ["other", ["LMS", "Heun", "DPM2", "DPM2 a", "DEIS", "SA-Solver"]],
  ["distilled", ["LCM", "TCD"]],
  ["anima", ["ER-SDE"]],
  // 實驗用，出圖不一定比較好。Anima 的 Euler a 跟 Forge／ComfyUI 一樣；
  // renoise＝每步把雜訊換新（CoCo 以前的 Euler a），數字＝換掉的比例，late＝最後 1/4 步不加新雜訊、late-half＝最後一半
  ["animaLab", ["Euler a 0.5", "Euler a CoCo", "Euler a CoCo 0.5",
                "Euler a renoise", "Euler a renoise 0.25", "Euler a renoise 0.5",
                "Euler a renoise late", "Euler a renoise 0.5 late", "Euler a renoise late-half",
                "Euler Churn", "ER-SDE 1", "ER-ODE", "ER-SDE ve"]],
];

export const SCHEDULES = ["Normal", "Karras", "Exponential", "Beta", "SGM Uniform",
                          "Flow", "Lu Lambdas", "Linspace", "Leading"];

// 常用尺寸，都是 64 的倍數。只列直的，橫的用「互換」得到。
// 前四組約 100 萬像素（SDXL 訓練的大小），後兩組約 140～160 萬，Anima 常用。
export const SIZES = [
  ["1:1", 1024, 1024],
  ["7:9", 896, 1152],
  ["≈2:3", 832, 1216],
  ["9:16", 768, 1344],
  ["≈3:4", 1024, 1344],
  ["2:3", 1024, 1536],
];

export const VRAM_MODES = ["speed", "balanced", "low"];
export const COND_MODES = ["auto", "a1111", "native", "both"];
export const AUTO_ON_OFF = ["auto", "on", "off"];

// ── 圖片中繼資料裡的名稱 → 這裡的名稱（「送到生成區」用） ──
// 別名：跟 core/anima.py 的 _ALIASES 一致；舊合併名：跟 core/pipeline.py 的 LEGACY 一致
const SAMPLER_ALIASES = {
  "euler a kdiff": "Euler a CoCo", "euler a kdiff eta 0.5": "Euler a CoCo 0.5", "er-sde ode": "ER-ODE",
  "er sde": "ER-SDE",                    // Forge / ComfyUI 的寫法
  // 2026-10-03 改名：Euler a RF 就是現在 Anima 的 Euler a；以前 CoCo 的 Euler a 那一系改叫 renoise
  "euler a rf": "Euler a", "euler a rf 0.5": "Euler a 0.5", "euler a rf eta 0.5": "Euler a 0.5",
  "euler a eta 0.25": "Euler a renoise 0.25", "euler a eta 0.5": "Euler a renoise 0.5",
  "euler a late-off": "Euler a renoise late", "euler a late-off 0.5": "Euler a renoise late-half",
  "euler a eta 0.5 late-off": "Euler a renoise 0.5 late",
  // CoCo 更早以前存 Anima 圖時寫的是實際的排程器（core/anima.py 的 _coco_sched_name）
  "flowmatch euler": "Euler", "flowmatch euler a": "Euler a renoise", "flowmatch euler sde": "Euler a renoise",
  "flowmatch heun": "Heun",
};
const LEGACY = {
  "DPM++ 2M Karras": ["DPM++ 2M", "Karras"],
  "DPM++ 2M SDE Karras": ["DPM++ 2M SDE", "Karras"],
  "DPM++ SDE Karras": ["DPM++ SDE", "Karras"],
};
const SCHEDULE_ALIASES = {"automatic": "Normal"};   // Forge：交給取樣器自己決定，就是 Normal

// 回傳 {sampler, schedule}，認不得的是 null（沿用目前的設定）
export function mapSampler(sampler, schedule) {
  let s = String(sampler || "").trim(), k = String(schedule || "").trim();
  if (LEGACY[s]) [s, k] = [LEGACY[s][0], k || LEGACY[s][1]];
  // CoCo 以前存 Anima 圖：排程併在取樣器後面（「Euler a RF + Karras」），旁邊的 Schedule type 一律 Normal
  const plus = s.match(/^(.+?)\s*\+\s*(Karras|Exponential|Beta)$/i);
  if (plus) [s, k] = [plus[1], plus[2]];
  const all = SAMPLER_GROUPS.flatMap(([, names]) => names);
  const byLow = new Map(all.map(n => [n.toLowerCase(), n]));
  const low = s.toLowerCase();
  const outS = byLow.get(low) || SAMPLER_ALIASES[low] || null;
  const outK = SCHEDULES.find(n => n.toLowerCase() === k.toLowerCase()) || SCHEDULE_ALIASES[k.toLowerCase()] || null;
  return {sampler: outS, schedule: outK};
}

// ── 精修照原圖的模型與參數（使用者筆記：修臉要用原圖的模型，不然畫風對不上）──
// src：精修頁的底圖（server/refine._item）；models：資源清單（/api/res?kind=model 的 items）。
// ── 精修的「用指示修圖」：Qwen Image 2.1、FLUX.2 klein 或 FLUX.1 Kontext（server/refine.EDIT_ARCHS 同一張表）──
export const EDIT_ARCHS = ["qwen21", "klein4", "klein9", "kontext"];
// 參數照模型家族各記各的（官方建議差很多）：klein 4B、9B 一組。沒改過用官方建議
// （Qwen Image 2.1 40 步；klein 蒸餾版 4 步，Base 版官方 50 步、CFG 4，改了會記住；Kontext 28 步、guidance 2.5）
const EDIT_DEFAULTS = {qwen21: {steps: 40, cfg: 1, sampler: "Euler"}, klein: {steps: 4, cfg: 1, sampler: "Euler"},
                       kontext: {steps: 28, cfg: 2.5, sampler: "Euler"}};
const EDIT_PARAMS = ["steps", "cfg", "sampler"];
export const editFamily = arch => arch === "qwen21" || arch === "kontext" ? arch : "klein";

// 修圖面板目前的設定。redit：介面偏好（{model, size, by: {qwen21: {...}, klein: {...}}}）；以前只有 Qwen 時
// 參數直接放在 redit 底下，照舊算 Qwen 的。shift 用生成台那個架構的（gen：生成台的設定）
export function editOpts(redit, arch, gen) {
  const r = redit || {}, fam = editFamily(arch);
  const legacy = fam === "qwen21" ? Object.fromEntries(EDIT_PARAMS.filter(k => r[k] != null).map(k => [k, r[k]])) : {};
  const shiftKeyName = shiftKey(arch);
  return {...EDIT_DEFAULTS[fam], schedule: "Normal", size: r.size || "src", shift: gen?.[shiftKeyName] ?? null,
          ...((r.by || {})[fam] ?? legacy)};
}

// 改一個設定之後的 redit：模型、大小大家共用，步數、CFG、取樣器記在這個模型家族底下
export function editPatch(redit, arch, patch) {
  const r = {...(redit || {})}, fam = editFamily(arch);
  const own = {}, mine = {};
  for (const [k, v] of Object.entries(patch)) (EDIT_PARAMS.includes(k) ? mine : own)[k] = v;
  if (Object.keys(mine).length) {
    const cur = editOpts(r, arch, null);
    r.by = {...(r.by || {}), [fam]: {...Object.fromEntries(EDIT_PARAMS.map(k => [k, cur[k]])), ...mine}};
  }
  return {...r, ...own};
}

// 修圖用哪個模型：面板選過的；沒選過時生成台正在用的能修圖就用它，不然 Qwen Image 2.1 優先
// （以前只有它，沒選過的人照舊用它；清單照名字排，klein 會排在前面），再來清單裡第一個。回傳 {path, arch} 或 null
export function editModelOf(list, redit, gen) {
  const l = list || [];
  const it = l.find(x => x.path === redit?.model)
    || (EDIT_ARCHS.includes(gen?.MODEL_ARCH) && gen?.MODEL ? {path: gen.MODEL, arch: gen.MODEL_ARCH}
        : l.find(x => x.arch === "qwen21") || l[0]);
  return it ? {path: it.path, arch: it.arch} : null;
}

// 用指示修圖出來的大小：照原圖（src），或縮到約 100 萬像素（mp1，1024²，官方預設；比較快）。
// 照原圖對齊 16（跟其他精修方式一樣，伺服器 refine.snap）；縮小的對齊 32（Qwen Image 2.1 要的），不放大
export function editSize(w, h, size) {
  const r = size === "mp1" ? Math.min(1, Math.sqrt(1024 * 1024 / (w * h))) : 1;
  const step = r < 1 ? 32 : 16;
  const fit = v => Math.max(256, Math.min(4096, Math.round(v * r / step) * step));
  return {w: fit(w), h: fit(h)};
}

// 找得到原圖的模型：模型與參數都照原圖，原圖沒記的單一欄位才留給生成台（不蓋掉）。
// 原圖沒記模型、或模型檔已經找不到：整組用生成台的——不混用，Anima 的 CFG、取樣器拿去給 SDXL 反而壞。
// 回傳 {ok, over: 要蓋掉的參數, name: 模型名稱, why: 用不了的原因（"noMeta" / "noModel"）}
export function sourceParams(src, models) {
  const key = s => String(s || "").replace(/\\/g, "/").split("/").pop().replace(/\.(safetensors|ckpt|pt|sft)$/i, "").toLowerCase();
  if (!src?.model) return {ok: false, over: {}, why: "noMeta"};
  const hit = (models || []).find(x => key(x.name) === key(src.model) || key(x.path) === key(src.model));
  if (!hit) return {ok: false, over: {}, why: "noModel", name: key(src.model) && String(src.model).replace(/\\/g, "/").split("/").pop()};
  const over = {model: hit.path};
  const m = mapSampler(src.sampler, src.scheduler);
  if (m.sampler) over.sampler = m.sampler;
  if (m.schedule) over.schedule = m.schedule;
  const num = v => (v === null || v === undefined || v === "" || !Number.isFinite(Number(v)) ? null : Number(v));
  if (num(src.steps) > 0) over.steps = num(src.steps);
  if (num(src.cfg) != null) over.cfg = num(src.cfg);
  if (num(src.clip_skip) > 0) over.clip_skip = num(src.clip_skip);
  if (num(src.shift) > 0) over.shift = num(src.shift);
  return {ok: true, over, name: hit.name};
}

// 跟 server/prompts.py 的 _LORA_TAG、_DYN_SYNTAX 同一條規則
export const LORA_TAG = /<lora:([^:>]+?)(?::([-\d.]+))?>/gi;
export const DYN_SYNTAX = /__[^_]+__|\{[^}]*\}|\[\[/;

// ── 提示詞裡的 <lora:名字:權重>（提示詞是唯一的真相，資源庫只是改這串文字） ──
const escRe = s => String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const tagRe = name => new RegExp(`<lora:${escRe(name)}(?::([-\\d.]+))?>`, "gi");
export const fmtWeight = w => String(Math.round(Number(w) * 100) / 100);

export function hasLora(prompt, name) {
  return tagRe(name).test(String(prompt || ""));
}

export function addLora(prompt, name, weight = 1) {
  const p = String(prompt || "");
  if (hasLora(p, name)) return p;
  const base = p.replace(/[\s,]*$/, "");
  return `${base}${base ? ", " : ""}<lora:${name}:${fmtWeight(weight)}>`;
}

// 提示詞裡的 <lora:舊名…> 換成新名字（權重不動）
export function renameLora(prompt, old, now) {
  return String(prompt || "").replace(new RegExp(`<lora:${escRe(old)}(?=[:>])`, "gi"), `<lora:${now}`);
}

// 拿掉標籤，連同它前面多出來的逗號與空白，不留下「, ,」
export function removeLora(prompt, name) {
  return String(prompt || "")
    .replace(new RegExp(`[ \\t]*,?[ \\t]*<lora:${escRe(name)}(?::[-\\d.]+)?>`, "gi"), "")
    .replace(/^[\s,]+/, "");
}

// 拿掉外層的 LoRA 標籤（套用 Recipe、重組時換 LoRA 用）。動態語法（[[a|b]]、{a|b}）裡的是輪流用的，
// 不能拿掉，也不能搬到外面同時套用——以前會把 [[<lora:A>|<lora:B>]] 拆成 [[|]]
export function stripTopLoras(prompt) {
  const p = String(prompt || "");
  const spans = [...p.matchAll(/\[\[[\s\S]*?\]\]|\{[^{}]*\}/g)].map(m => [m.index, m.index + m[0].length]);
  const inGroup = i => spans.some(([a, b]) => i >= a && i < b);
  let out = "", last = 0;
  for (const m of p.matchAll(/[ \t]*,?[ \t]*<lora:[^>]*>/gi)) {
    if (inGroup(m.index + m[0].indexOf("<"))) continue;
    out += p.slice(last, m.index);
    last = m.index + m[0].length;
  }
  return (out + p.slice(last)).replace(/^[\s,]+/, "");
}

// 圖的提示詞補上它的 LoRA（看大圖的送到生成區、只送提示詞、複製提示詞）：CoCo 存圖時把 <lora:…> 從提示詞拿掉、
// 另外記在 loras；Forge 的本來就在提示詞裡，已經有的不重複加。loras：[{name, weight}]
export function withLoras(prompt, loras) {
  let p = String(prompt || "");
  for (const l of loras || []) if (l.name) p = addLora(p, l.name, l.weight);
  return p;
}

// 只送提示詞、保留生成台的 LoRA（使用者要求 2026-10-12）：圖的提示詞外層的 LoRA 拿掉，換成生成台提示詞外層的。
// 動態語法裡輪流用的不動：圖的留著，生成台的不搬過來
export function keepLoras(prompt, genPrompt) {
  const g = String(genPrompt || "");
  const inner = stripTopLoras(g);
  let p = stripTopLoras(prompt);
  for (const l of loraTags(g)) if (!hasLora(inner, l.name)) p = addLora(p, l.name, l.weight);
  return p;
}

export function setLoraWeight(prompt, name, weight) {
  return String(prompt || "").replace(tagRe(name), `<lora:${name}:${fmtWeight(weight)}>`);
}

// ── Embeddings：提示詞裡寫檔名就套用（跟 core/embeddings.py 的 used() 同一條規則：
// 不分大小寫、可以寫成 embedding:名字、前後不能接著字母數字底線連字號） ──
const embRe = (name, flags = "i") =>
  new RegExp(`(?<![\\w\\-.:])(?:embedding:)?${escRe(name)}(?:\\.(?:safetensors|pt|bin))?(?![\\w\\-])`, flags);

export function hasEmb(prompt, name) {
  return embRe(name).test(String(prompt || ""));
}

export function addEmb(prompt, name) {
  const p = String(prompt || "");
  if (hasEmb(p, name)) return p;
  const base = p.replace(/[\s,]*$/, "");
  return `${base}${base ? ", " : ""}${name}`;
}

// 拿掉名字（加了權重的「(名字:1.2)」整個拿掉），連同前面多出來的逗號
export function removeEmb(prompt, name) {
  const n = embRe(name).source;
  return String(prompt || "")
    .replace(new RegExp(`[ \\t]*,?[ \\t]*\\(\\s*${n}\\s*(?::\\s*[-\\d.]+\\s*)?\\)`, "gi"), "")
    .replace(new RegExp(`[ \\t]*,?[ \\t]*${n}`, "gi"), "")
    .replace(/^[\s,]+/, "");
}

// ── LoRA 觸發詞（資源庫的觸發詞按鈕）：不分大小寫、前後不能接著字母數字底線連字號（girl 不算 1girl 裡有）。
// <lora:…> 標籤裡的不算：觸發詞常常就是 LoRA 的檔名，以前直接找子字串，套用中的 LoRA 讓它的觸發詞永遠加不進去 ──
const trigRe = word => new RegExp(`(?<![\\p{L}\\p{N}_\\-])${escRe(word)}(?![\\p{L}\\p{N}_\\-])`, "iu");

export function hasTrigger(prompt, word) {
  const w = String(word || "").trim();
  return !!w && trigRe(w).test(String(prompt || "").replace(/<[^<>]*>/g, " "));
}

export function addTrigger(prompt, word) {
  const p = String(prompt || "");
  if (hasTrigger(p, word)) return p;
  const base = p.replace(/[\s,]*$/, "");
  return `${base}${base ? ", " : ""}${String(word).trim()}`;
}

export function loraTags(prompt) {
  const out = [];
  for (const m of String(prompt || "").matchAll(LORA_TAG)) {
    const name = m[1].trim();
    const w = m[2] ? Number(m[2]) : 1;
    if (name) out.push({name, weight: Number.isFinite(w) ? w : 1});
  }
  return out;
}
