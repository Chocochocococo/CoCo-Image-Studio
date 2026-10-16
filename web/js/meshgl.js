// 扭曲、彎曲的繪製：把圖切成 N×N 的小格子，每個格子的四個角照對應函式搬到新的位置，用 WebGL2 一次畫出來。
// Canvas 2D 只會做平移旋轉縮放，四角各自拖、彎曲這種變形要靠 GPU 貼圖才不會有接縫、也夠快。
// 用 mipmap 縮小時不會出現鋸齒；邊緣用多重取樣抗鋸齒。圖都是預乘透明度，半透明的邊不會發黑

let gl = null, cv = null, prog = null, buf = null, tex = null, tried = false;

const VS = `#version 300 es
in vec2 p; in vec2 uv; out vec2 v;
void main() { v = uv; gl_Position = vec4(p, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec2 v; uniform sampler2D s; out vec4 o;
void main() { o = texture(s, v); }`;

function init() {
  tried = true;
  cv = document.createElement("canvas");
  gl = cv.getContext("webgl2", {premultipliedAlpha: true, preserveDrawingBuffer: true, antialias: true, alpha: true});
  if (!gl) return false;
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { gl = null; return false; }
  buf = gl.createBuffer();
  tex = gl.createTexture();
  return true;
}

// 這台電腦能不能用（不能的話扭曲、彎曲的按鈕停用）
export function meshSupported() {
  return !!gl || (!tried && init());
}

// 把 src 照 map（(u, v) → 圖上的點）畫到 W×H 的畫布上，回傳那張畫布（要馬上 drawImage 拿走，下次會重畫）
export function renderMesh(src, W, H, map, N = 32) {
  if (!meshSupported()) return null;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  gl.viewport(0, 0, W, H);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // 格點先算好：圖上的像素 → WebGL 的座標（-1～1，上下顛倒）
  const P = [];
  for (let j = 0; j <= N; j++) {
    for (let i = 0; i <= N; i++) {
      const [x, y] = map(i / N, j / N);
      P.push([x / W * 2 - 1, 1 - y / H * 2, i / N, j / N]);
    }
  }
  const data = new Float32Array(N * N * 6 * 4);
  let k = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const a = P[j * (N + 1) + i], b = P[j * (N + 1) + i + 1], c = P[(j + 1) * (N + 1) + i + 1], d = P[(j + 1) * (N + 1) + i];
      for (const q of [a, b, c, a, c, d]) { data.set(q, k); k += 4; }
    }
  }
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
  const lp = gl.getAttribLocation(prog, "p"), lu = gl.getAttribLocation(prog, "uv");
  gl.enableVertexAttribArray(lp);
  gl.vertexAttribPointer(lp, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(lu);
  gl.vertexAttribPointer(lu, 2, gl.FLOAT, false, 16, 8);
  gl.disable(gl.BLEND);
  gl.drawArrays(gl.TRIANGLES, 0, N * N * 6);
  return cv;
}
