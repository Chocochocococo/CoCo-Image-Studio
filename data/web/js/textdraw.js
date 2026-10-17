// 文字的排版與繪製：文字圖層、浮水印共用。T = {str, font, size, bold, italic, color, align, stroke, scolor, shadow}
// 多行照換行切開，行高是字的 1.25 倍；(x, y) 的 x 照對齊方式是左邊／中間／右邊，y 是第一行的頂端

// 常見的字型（Windows 內建＋常見的中日文字型）；輸入框也能直接打電腦裡任何字型的名字
export const FONTS = ["Microsoft JhengHei", "PMingLiU", "DFKai-SB", "Microsoft YaHei", "Yu Gothic", "Meiryo", "Noto Sans TC",
                      "Arial", "Segoe UI", "Georgia", "Times New Roman", "Impact", "Consolas", "Comic Sans MS"];

export const fontCss = T => `${T.italic ? "italic " : ""}${T.bold ? 700 : 400} ${Math.max(1, T.size)}px "${T.font || FONTS[0]}", "Microsoft JhengHei", sans-serif`;

export function measureText(ctx, T) {
  ctx.save();
  ctx.font = fontCss(T);
  const lines = String(T.str ?? "").split("\n"), lh = T.size * 1.25;
  const w = Math.max(1, ...lines.map(l => ctx.measureText(l).width));
  ctx.restore();
  return {lines, lh, w, h: lines.length * lh};
}

// 畫出來，回傳佔的範圍 {x, y, w, h}（含描邊）。有描邊時陰影只跟著描邊畫一次，字本身不再疊一層陰影
export function drawText(ctx, T, x = T.x, y = T.y) {
  const m = measureText(ctx, T), sw = Math.max(0, Number(T.stroke) || 0);
  ctx.save();
  ctx.font = fontCss(T);
  ctx.textAlign = T.align || "left";
  ctx.textBaseline = "top";
  const shadow = on => {
    ctx.shadowColor = on ? "rgba(0, 0, 0, .55)" : "transparent";
    ctx.shadowBlur = on ? T.size * 0.12 : 0;
    ctx.shadowOffsetX = ctx.shadowOffsetY = on ? T.size * 0.05 : 0;
  };
  m.lines.forEach((line, i) => {
    const ly = y + i * m.lh + (m.lh - T.size) / 2;
    if (sw) {
      shadow(!!T.shadow);
      ctx.lineJoin = "round";
      ctx.lineWidth = sw * 2;
      ctx.strokeStyle = T.scolor || "#000";
      ctx.strokeText(line, x, ly);
      shadow(false);
    } else shadow(!!T.shadow);
    ctx.fillStyle = T.color || "#fff";
    ctx.fillText(line, x, ly);
  });
  ctx.restore();
  const left = T.align === "center" ? x - m.w / 2 : T.align === "right" ? x - m.w : x;
  return {x: left - sw, y: y - sw, w: m.w + 2 * sw, h: m.h + 2 * sw};
}
