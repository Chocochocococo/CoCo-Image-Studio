# -*- coding: utf-8 -*-
"""
bgremove.py —— 去背（精修的「去背」方式），在工作行程裡跑

CoCo 認的是架構不是檔案：目前是 IS-Net（DIS 的官方實作，anime-segmentation 也是它）與
BiRefNet（Swin 主幹，core/birefnet.py；官方的 BiRefNet、lite、動漫的 ToonOut 都是），
這兩種架構的 safetensors 權重都能用（server/rembg.py 讀檔頭判斷）。可以下載的是
SkyTNT 的 anime-segmentation（動漫角色專用，Apache-2.0）；模型不內建、不自動下載。

IS-Net 的結構照 DIS 官方實作（xuebinqin/DIS）寫在這裡，不從網路抓程式碼來執行。
權重名字前面的 "net." 之類自動拿掉（anime-segmentation 訓練時另外有一個 gt_encoder，推論用不到）。

前處理有兩種（同一個架構、不同來源的訓練方式不同，用錯效果會變差）：
  anime：跟 anime-segmentation 的 inference.py 一樣——RGB／255、長邊縮到 1024、補成正方形（置中）
  dis  ：跟 DIS 官方一樣——RGB／255 再減 0.5、直接縮成 1024×1024
都取第一個輸出套 sigmoid，再縮回原圖大小。結果是 0～1 的遮罩（1＝前景）。
BiRefNet 只有一種（官方的）：直接縮成 1024×1024、ImageNet 的平均與標準差標準化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import i18n

SIZE = 1024                    # anime-segmentation 推論預設的大小
_MODELS = {}                   # 路徑 → 載好的模型（第一次約 1 秒，之後直接用；只留最後用的一個）


def _up(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    """RSU-L（L＝7、6、5、4）：一個小 U-Net。depth＝往下縮幾次再加一層空洞卷積"""

    def __init__(self, depth, in_ch, mid_ch, out_ch):
        super().__init__()
        self.depth = depth
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        for i in range(1, depth - 1):
            setattr(self, f"pool{i}", nn.MaxPool2d(2, stride=2, ceil_mode=True))
            setattr(self, f"rebnconv{i + 1}", REBNCONV(mid_ch, mid_ch, dirate=1))
        setattr(self, f"rebnconv{depth}", REBNCONV(mid_ch, mid_ch, dirate=2))
        for i in range(depth - 1, 1, -1):
            setattr(self, f"rebnconv{i}d", REBNCONV(mid_ch * 2, mid_ch, dirate=1))
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hs = [self.rebnconv1(hxin)]
        hx = hs[0]
        for i in range(1, self.depth - 1):
            hx = getattr(self, f"pool{i}")(hx)
            hx = getattr(self, f"rebnconv{i + 1}")(hx)
            hs.append(hx)
        d = getattr(self, f"rebnconv{self.depth}")(hs[-1])
        for i in range(self.depth - 1, 0, -1):
            d = getattr(self, f"rebnconv{i}d")(torch.cat((d, hs[i - 1]), 1))
            if i > 1:
                d = _up(d, hs[i - 2])
        return d + hxin


class RSU4F(nn.Module):
    """RSU-4F：不縮小，用空洞卷積（1、2、4、8）看更大的範圍"""

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        return self.rebnconv1d(torch.cat((hx2d, hx1), 1)) + hxin


class ISNetDIS(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)      # 官方實作有定義，但沒接進 stage1
        self.stage1 = RSU(7, 64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU(4, 1024, 128, 256)
        self.stage3d = RSU(5, 512, 64, 128)
        self.stage2d = RSU(6, 256, 32, 64)
        self.stage1d = RSU(7, 128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        """只回傳最細的那個輸出（d1，還沒套 sigmoid）；其他五個是訓練時用的"""
        hxin = self.conv_in(x)
        hx1 = self.stage1(hxin)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))
        hx5d = self.stage5d(torch.cat((_up(hx6, hx5), hx5), 1))
        hx4d = self.stage4d(torch.cat((_up(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_up(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_up(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_up(hx2d, hx1), hx1), 1))
        return _up(self.side1(hx1d), x)


PREFIXES = ("", "net.", "model.", "module.")    # 權重名字前面可能有的（anime-segmentation 是 net.）


IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # BiRefNet 的標準化


def load(path, device=None, arch="isnet"):
    """safetensors 權重 → 模型（照路徑快取）。名字前面的 net. 之類自動拿掉（訓練用的其他部分略過）；
    名字對不上、少了或多了任何一個都直接報錯，不會默默載錯"""
    from safetensors.torch import load_file
    if arch not in ("isnet", "birefnet"):
        raise ValueError(i18n.tr("eng.t.badArch", arch=arch))
    key = str(path)
    if key in _MODELS:
        return _MODELS[key]
    sd = load_file(key)
    probe = "conv_in.weight" if arch == "isnet" else "bb.patch_embed.proj.weight"
    pre = next((p for p in PREFIXES if p + probe in sd), "")
    net = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if arch == "isnet":
        model = ISNetDIS()
    else:
        import birefnet
        cfg = birefnet.config_from({k: list(v.shape) for k, v in net.items()})
        if not cfg:
            raise ValueError(i18n.tr("eng.rb.backbone"))
        model = birefnet.BiRefNet(**cfg)
        # 相對位置索引是算出來的固定值：權重裡沒存的話用算的
        own = model.state_dict()
        for k, v in own.items():
            if k.endswith("relative_position_index") and k not in net:
                net[k] = v
    model.load_state_dict(net, strict=True)
    _MODELS.clear()                 # 一次只留一個：以前每個用過的都留在顯示卡上，換幾個模型就越佔越多
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    _MODELS[key] = model
    return model


def release():
    """放掉載好的去背模型（要載入生成模型時讓出顯示卡；下次去背再載，約 1 秒）"""
    _MODELS.clear()


@torch.inference_mode()
def predict(image, model_path, size=SIZE, arch="isnet", prep="anime"):
    """圖（路徑或 PIL）→ 遮罩（PIL 的 L，跟原圖一樣大；255＝前景）。
    prep＝anime：anime-segmentation 的做法（0～1，長邊縮到 size、補成正方形）；
    prep＝dis：DIS 官方的做法（0～1 再減 0.5，直接縮成 size×size）"""
    import numpy as np
    im = image if isinstance(image, Image.Image) else Image.open(image)
    rgb = im.convert("RGB")
    w0, h0 = rgb.size
    model = load(model_path, arch=arch)
    dev = next(model.parameters()).device
    if arch == "birefnet":
        # 官方的做法：直接縮成 size×size、ImageNet 標準化，取最後一個輸出套 sigmoid
        x = torch.from_numpy(np.asarray(rgb.resize((size, size), Image.BILINEAR), dtype="float32") / 255.0).permute(2, 0, 1)
        mean, std = (torch.tensor(v).view(3, 1, 1) for v in IMAGENET)
        x = ((x - mean) / std)[None].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pred = model(x)
        pred = F.interpolate(pred.float().sigmoid(), size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
        return Image.fromarray((pred.clamp(0, 1) * 255).round().byte().cpu().numpy(), "L")
    if prep == "dis":
        h, w, ph, pw = size, size, 0, 0
    else:
        h, w = (size, max(1, round(size * w0 / h0))) if h0 > w0 else (max(1, round(size * h0 / w0)), size)
        ph, pw = size - h, size - w
    x = torch.zeros(1, 3, size, size, device=dev)
    small = torch.from_numpy(np.asarray(rgb.resize((w, h), Image.BILINEAR), dtype="float32") / 255.0)
    if prep == "dis":
        small = small - 0.5
    x[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = small.permute(2, 0, 1).to(dev)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        pred = model(x)
    pred = pred.float().sigmoid()[0, 0, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    pred = F.interpolate(pred[None, None], size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
    return Image.fromarray((pred.clamp(0, 1) * 255).round().byte().cpu().numpy(), "L")
