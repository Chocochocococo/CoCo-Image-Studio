// 重組的提示詞（使用者筆記：提示詞要能拆開來重組，不是只能整段用）。不碰 DOM，測試直接用 node 跑。
//
// 每個來源的提示詞拆成標籤（prompttext.js），依標籤庫的分類分組（品質、人物、頭髮、服裝…）；
// 每一組可以挑一個或好幾個來源，組回去時照組的順序、組裡照來源的順序，重複的標籤只留第一個。
// LoRA 標籤不在這裡（重組的 LoRA 列另外處理），註解不帶。

import {chipText, parsePrompt} from "./prompttext.js";

// 組的順序＝組出來的提示詞的順序：品質詞放最前面，再來是人物、外觀、服裝、動作、場景
export const GROUPS = ["meta", "subject", "who", "hair", "eyes", "body", "face", "wear", "pose", "scene", "other"];

// 提示詞 → {組: [標籤文字（含權重的寫法）…]}。groupOf(標籤) 回傳組名，認不得的（""）歸到 other；
// 動態語法（[[a|b]]、{a|b}、__x__…）也歸到 other
export function splitByGroup(prompt, groupOf) {
  const out = {};
  for (const c of parsePrompt(prompt || "")) {
    if (c.kind === "lora" || c.kind === "comment") continue;
    const text = chipText(c);
    if (!text) continue;
    const g = c.kind === "tag" ? (GROUPS.includes(groupOf(c.text)) ? groupOf(c.text) : "other") : "other";
    (out[g] = out[g] || []).push(text);
  }
  return out;
}

// 標籤的比對鍵：不分大小寫、權重不算（(smile:1.2) 跟 smile 是同一個）
const keyOf = text => String(text).replace(/^\((.*):[-\d.]+\)$/, "$1").replace(/^[([]+|[)\]]+$/g, "").trim().toLowerCase();

// parts：每個來源的 splitByGroup 結果（沒有提示詞的來源是 null）；picks：{組: [來源序號…]}
// 回傳組好的提示詞：一組一行
export function mix(parts, picks) {
  const lines = [];
  for (const g of GROUPS) {
    const seen = new Set(), tags = [];
    for (const i of [...(picks[g] || [])].sort((a, b) => a - b)) {
      for (const tag of parts[i]?.[g] || []) {
        const k = keyOf(tag);
        if (!seen.has(k)) { seen.add(k); tags.push(tag); }
      }
    }
    if (tags.length) lines.push(tags.join(", "));
  }
  return lines.join(",\n");
}
