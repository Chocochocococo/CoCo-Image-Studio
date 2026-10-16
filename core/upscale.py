# -*- coding: utf-8 -*-
"""
upscale.py —— 高清化的第一步：用放大模型把圖放大（在工作行程裡跑）

支援 ESRGAN 架構（RRDBNet）：4x-UltraSharp、4x-AnimeSharp、Remacri、NMKD-Siax、RealESRGAN_x4plus(_anime_6B)…
這類模型 A1111／Forge／ComfyUI 都在用。兩種權重寫法都認得：
  舊的 ESRGAN（model.0、model.1.sub.N.RDB1.conv1.0…）、新的 BasicSR／Real-ESRGAN（conv_first、body.N.rdb1.conv1…，
  外面可能包一層 params_ema）。區塊數、通道數、放大倍數從權重的形狀推算。
.pth 用 torch.load(weights_only=True) 讀：只收權重數字，檔案裡夾帶的程式不會執行；.safetensors 照常讀。

大圖分塊跑（每塊 256、四周多留 32 當上下文），顯示卡記憶體不會隨圖變大而爆掉：
實測 4x 模型每塊 512 時峰值約 3 GB、256 時 1～2 GB，速度差不多。用完 release() 卸載、把快取還給系統。
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import i18n

TILE = 256
PAD = 32
_MODELS = {}


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN／Real-ESRGAN 的生成器。ups＝放大幾次 2 倍；unshuffle＝輸入先縮小幾倍（Real-ESRGAN 的 x2／x1 版本）"""

    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, gc=32, ups=2, unshuffle=1):
        super().__init__()
        self.ups, self.unshuffle = ups, unshuffle
        self.conv_first = nn.Conv2d(in_ch * unshuffle * unshuffle, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        for i in range(ups):
            setattr(self, f"conv_up{i + 1}", nn.Conv2d(nf, nf, 3, 1, 1))
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    @property
    def scale(self):
        return 2 ** self.ups // self.unshuffle

    def forward(self, x):
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        for i in range(self.ups):
            feat = self.lrelu(getattr(self, f"conv_up{i + 1}")(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def convert_old(sd):
    """舊的 ESRGAN 命名 → 新的（BasicSR）命名。舊的：model.0＝conv_first、model.1.sub.N＝RRDB（最後一個是 conv_body）、
    model.(≥2) 的卷積照順序是放大層…、倒數第二是 conv_hr、最後是 conv_last"""
    out = {}
    tail = sorted({int(m.group(1)) for k in sd for m in [re.match(r"model\.(\d+)\.weight$", k)] if m and int(m.group(1)) >= 2})
    names = {n: f"conv_up{i + 1}" for i, n in enumerate(tail[:-2])}
    names.update({tail[-2]: "conv_hr", tail[-1]: "conv_last"} if len(tail) >= 2 else {})
    nb = max((int(m.group(1)) for k in sd for m in [re.match(r"model\.1\.sub\.(\d+)\.", k)] if m), default=-1)
    for k, v in sd.items():
        if m := re.match(r"model\.0\.(weight|bias)$", k):
            out[f"conv_first.{m.group(1)}"] = v
        elif m := re.match(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)$", k):
            out[f"body.{m.group(1)}.rdb{m.group(2)}.conv{m.group(3)}.{m.group(4)}"] = v
        elif m := re.match(r"model\.1\.sub\.(\d+)\.(weight|bias)$", k):
            if int(m.group(1)) == nb:
                out[f"conv_body.{m.group(2)}"] = v
        elif m := re.match(r"model\.(\d+)\.(weight|bias)$", k):
            if int(m.group(1)) in names:
                out[f"{names[int(m.group(1))]}.{m.group(2)}"] = v
            else:
                out[k] = v                       # 認不得的留著，嚴格載入時會報錯
        else:
            out[k] = v
    return out


def read_weights(path):
    """權重檔 → {名字: tensor}。.pth 只收權重數字（weights_only），外面包一層 params_ema 之類的拆掉"""
    p = str(path)
    if p.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(p)
    else:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params", "state_dict", "model"):
        if isinstance(sd, dict) and isinstance(sd.get(k), dict):
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        raise ValueError(i18n.tr("eng.up.unreadable"))
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    if "model.0.weight" in sd:
        sd = convert_old(sd)
    return sd


def build(sd):
    """從權重的形狀推出 RRDBNet 的設定並建出來；不是 ESRGAN 架構回 None"""
    if "conv_first.weight" not in sd or "body.0.rdb1.conv1.weight" not in sd:
        return None
    nf, cin = sd["conv_first.weight"].shape[:2]
    gc = sd["body.0.rdb1.conv1.weight"].shape[0]
    nb = 1 + max(int(m.group(1)) for k in sd for m in [re.match(r"body\.(\d+)\.", k)] if m)
    ups = sum(1 for k in sd if re.match(r"conv_up\d+\.weight$", k))
    out_ch = sd["conv_last.weight"].shape[0]
    un = int(round(math.sqrt(cin / out_ch))) if cin > out_ch else 1
    return RRDBNet(in_ch=out_ch, out_ch=out_ch, nf=nf, nb=nb, gc=gc, ups=ups, unshuffle=un)


def load(path, device=None):
    key = str(path)
    if key in _MODELS:
        return _MODELS[key]
    sd = read_weights(path)
    model = build(sd)
    if model is None:
        raise ValueError(i18n.tr("eng.up.arch"))
    model.load_state_dict(sd, strict=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    if dev != "cpu":
        model.half()
    _MODELS.clear()                          # 一次只留一個放大模型（每個約 60 MB，換模型時放掉舊的）
    _MODELS[key] = model
    return model


@torch.inference_mode()
def run(image, model_path, tile=TILE, pad=PAD):
    """圖 → 放大後的圖（模型本身的倍數，例如 4 倍）。分塊跑，每塊四周多算 pad 再切掉，接縫看不出來"""
    import numpy as np
    model = load(model_path)
    p = next(model.parameters())
    rgb = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
    x = torch.from_numpy(np.asarray(rgb, dtype="float32") / 255.0).permute(2, 0, 1)[None]
    _, _, H, W = x.shape
    s = model.scale
    out = torch.zeros(1, 3, H * s, W * s)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            py0, px0, py1, px1 = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad), min(W, x1 + pad)
            piece = model(x[:, :, py0:py1, px0:px1].to(p.device, p.dtype)).float().cpu()
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = piece[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
    arr = (out[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def release():
    """用完就卸載放大模型、把 PyTorch 留著的快取還給系統（接下來補細節要用顯示卡；工作管理員看到的佔用也會降下來）"""
    _MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def upscale_to(image, size, model_path=None):
    """放大到指定大小：有模型就先用模型放大，再用 Lanczos 調到剛好的大小；沒有模型就直接 Lanczos。
    透明背景的圖（例如去背過的）：顏色照放大，透明度用 Lanczos 放大後接回去，背景還是透明的"""
    im = image if isinstance(image, Image.Image) else Image.open(image)
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        a = im.convert("RGBA").getchannel("A")
        alpha = a if a.getextrema()[0] < 255 else None
    rgb = im.convert("RGB")
    if model_path:
        rgb = run(rgb, model_path)
    size = tuple(size)
    if rgb.size != size:
        rgb = rgb.resize(size, Image.LANCZOS)
    if alpha is not None:
        rgb.putalpha(alpha.resize(size, Image.LANCZOS))
    return rgb


# ── 補細節：放大後切成重疊的小塊，每塊用局部重繪的「只重畫遮罩附近」重畫（pipeline._hires）──
GEN_TILE = 832             # 一塊的大小：加上四周的上下文（32）與遮罩柔化往外擴的部分（約 57），剛好不超過 1024（SDXL、Anima 都是 1MP 級，不必縮小）
GEN_OVERLAP = 128          # 相鄰兩塊重疊多少，接縫靠遮罩柔化蓋過去


def tile_boxes(W, H, tile=GEN_TILE, overlap=GEN_OVERLAP):
    """把 W×H 切成蓋滿整張的小塊 [(x0, y0, x1, y1)]，平均分布、相鄰至少重疊 overlap；比 tile 小的邊就一整塊"""
    def axis(n):
        if n <= tile:
            return [(0, n)]
        k = math.ceil((n - overlap) / (tile - overlap))
        step = (n - tile) / (k - 1)
        return [(round(i * step), round(i * step) + tile) for i in range(k)]
    return [(x0, y0, x1, y1) for (y0, y1) in axis(H) for (x0, x1) in axis(W)]


# 1 倍（大小不變、只補細節）：在原尺寸分塊重畫，模型看到的跟原圖一樣密，細節反而變少；
# 先放大到約這麼多像素補細節、再縮回原尺寸，細節才會多。實測 832×1216（SDXL、修改幅度 0.3）細節量（Laplacian）以原圖為 100：
# 原尺寸分塊 87、整張重繪 89、放大 2 倍補完再縮回 111（見 DEVLOG）
DETAIL_PX = 2048 * 2048
DETAIL_MAX_SIDE = 4096     # 跟精修的上限一樣（server/upscalers.MAX_SIDE）


def detail_size(w, h):
    """1 倍補細節時在多大的畫布上重畫（對齊 16）：1MP 的圖放大 2 倍；本來就大的圖不必再放大（最少 1 倍、最多 2 倍）"""
    k = max(1.0, min(2.0, (DETAIL_PX / (w * h)) ** 0.5, DETAIL_MAX_SIDE / max(w, h)))
    return round(w * k / 16) * 16, round(h * k / 16) * 16
