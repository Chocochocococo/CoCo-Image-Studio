# -*- coding: utf-8 -*-
"""
yolo.py —— 自己跑 YOLOv8 的找物體外框（修臉的找臉），不靠 ultralytics

ultralytics 是 AGPL-3.0，跟 CoCo「公開原始碼、禁止商用」的方向不相容（儀表板「分享前必做」）。
模型檔照舊用 ADetailer 那些（face_yolov8n/s.pt、hand_yolov8n.pt，分割模型 person_yolov8*-seg.pt
也行，只用它找外框的部分），這裡自己讀檔、照檔案裡記的網路結構組起來、用 PyTorch 跑。

**讀檔。** .pt 是 pickle，裡面的類別是 ultralytics 的。不 import ultralytics，改用自己的 Unpickler：
ultralytics 與 torch.nn 的模組一律換成空殼（只要權重），其他只放行 PyTorch 重建張量需要的幾樣，
認不得的就拒絕載入——一般的 torch.load 會照檔案說的執行任何東西，這裡不會。
torch 讀檔時用到 pickle 的地方都換成這個 Unpickler；只收 zip 格式，舊格式與 TorchScript 另有入口，直接擋掉。

**網路結構。** 檔案裡有一份描述（model.yaml）：每一層從哪幾層接進來、重複幾次、是哪種區塊、參數，
加上 n／s／m… 各自的深度與寬度倍數。這裡照那份描述組網路，支援 YOLOv8 用到的區塊：
Conv、C2f、SPPF、Upsample、Concat、Detect（分割模型的 Segment 只用其中的 Detect 部分）。

**前後處理。** 跟 ADetailer 拿到的結果一致：長邊縮到 640、補邊到 32 的倍數（補 114 灰）、
信心門檻、IoU 0.7 的 NMS，外框換回原圖座標。
"""

import math
import pickle
import types

import numpy as np
import torch
from torch import nn

import i18n

IMG_SIZE = 640
IOU = 0.7
MAX_DET = 300
_PAD = 114


# ---------------------------------------------------------------- 讀檔

class _Shell(nn.Module):
    """ultralytics 與 torch.nn 模組的空殼：pickle 還原時塞進 __dict__，之後只取 state_dict"""


class _Bag:
    """ultralytics 其他物件（訓練參數之類）的空殼"""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _safe_getattr(obj, name):
    """檔案裡的 getattr 只拿來取 torch 的型別（torch.float16 之類），或空殼上的方法
    （訓練時把 forward 換成合併 BN 的版本，存成「某模組的某方法」）——後者不會被呼叫，給 None"""
    if isinstance(obj, (_Shell, _Bag)) or obj in (_Shell, _Bag):
        return None
    value = getattr(obj, name)
    if isinstance(value, (torch.dtype, torch.device)):
        return value
    raise pickle.UnpicklingError(i18n.tr("eng.yolo.get", name=name))


_ALLOWED = {
    ("collections", "OrderedDict"), ("torch", "Size"), ("torch", "device"),
    ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
    ("builtins", "set"), ("__builtin__", "set"),
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ultralytics."):
            return _Shell if ".nn." in module and name[:1].isupper() else _Bag
        if module.startswith("torch.nn.modules."):
            return _Shell
        if (module, name) in (("builtins", "getattr"), ("__builtin__", "getattr")):
            return _safe_getattr
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(i18n.tr("eng.yolo.unknown", name=f"{module}.{name}"))


def _load(file, **kw):
    """torch 讀舊格式（不是 zip 的 .pt）時直接呼叫 pickle_module.load 讀檔頭：這裡也要走白名單，
    不然檔頭那段就照檔案執行了（之後才報檔案壞掉也來不及）"""
    return _Unpickler(file, **kw).load()


_PICKLE = types.ModuleType("coco_yolo_pickle")
_PICKLE.Unpickler = _Unpickler
_PICKLE.load = _load
_PICKLE.__name__ = "pickle"


def _check_format(path):
    """只收 torch.save 的 zip 格式（ADetailer 的 YOLOv8 模型都是）。舊格式與 TorchScript 擋掉：
    TorchScript 的 zip（裡面有 constants.pkl）torch.load 會轉給 torch.jit.load，不經過上面的白名單"""
    import zipfile
    with open(path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":              # torch 判斷 zip 格式也是看開頭這 4 個位元組
            raise ValueError(i18n.tr("eng.yolo.notYolo"))
    with zipfile.ZipFile(path) as z:
        if any(n == "constants.pkl" or n.endswith("/constants.pkl") for n in z.namelist()):
            raise ValueError(i18n.tr("eng.yolo.torchscript"))


def read_checkpoint(path):
    """回傳 (網路結構描述, 權重 state_dict)"""
    _check_format(path)
    ck = torch.load(str(path), map_location="cpu", pickle_module=_PICKLE, weights_only=False)
    net = ck.get("model") if isinstance(ck, dict) else None
    if net is None:
        net = ck.get("ema") if isinstance(ck, dict) else None
    spec = getattr(net, "yaml", None)
    if not isinstance(spec, dict):
        raise ValueError(i18n.tr("eng.yolo.noStruct"))
    return spec, net.state_dict()


# ---------------------------------------------------------------- 區塊

class ConvBlock(nn.Module):
    """卷積 → BatchNorm → SiLU（載入後把 BN 併進卷積）"""

    def __init__(self, c_in, c_out, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()
        self.merged = None

    def forward(self, x):
        if self.merged is not None:
            return self.act(self.merged(x))
        return self.act(self.bn(self.conv(x)))

    def merge_bn(self):
        bn, conv = self.bn, self.conv
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        merged = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
        merged.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
        merged.bias.data = bn.bias.data - bn.running_mean * scale
        self.merged = merged.to(conv.weight.device)


class Residual(nn.Module):
    """兩個 3×3 卷積，輸入輸出同寬時加回輸入"""

    def __init__(self, c, add):
        super().__init__()
        self.cv1 = ConvBlock(c, c, 3)
        self.cv2 = ConvBlock(c, c, 3)
        self.add = add

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class SplitStack(nn.Module):
    """C2f：先 1×1 卷積後對半分，後半依序過 n 個 Residual，每一段的輸出全部接起來再 1×1 卷積"""

    def __init__(self, c_in, c_out, n=1, shortcut=False):
        super().__init__()
        self.half = int(c_out * 0.5)
        self.cv1 = ConvBlock(c_in, 2 * self.half, 1)
        self.cv2 = ConvBlock((2 + n) * self.half, c_out, 1)
        self.m = nn.ModuleList(Residual(self.half, shortcut) for _ in range(n))

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, 1))


class PoolPyramid(nn.Module):
    """SPPF：同一個 k×k 最大池化連做三次，四段接起來"""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        mid = c_in // 2
        self.cv1 = ConvBlock(c_in, mid, 1)
        self.cv2 = ConvBlock(mid * 4, c_out, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        outs = [self.cv1(x)]
        for _ in range(3):
            outs.append(self.m(outs[-1]))
        return self.cv2(torch.cat(outs, 1))


class Join(nn.Module):
    def forward(self, xs):
        return torch.cat(xs, 1)


class _Bins(nn.Module):
    """外框四邊的距離是 16 格的機率分布，取期望值（權重固定 0..15）"""

    def __init__(self, bins):
        super().__init__()
        self.conv = nn.Conv2d(bins, 1, 1, bias=False)


class BoxHead(nn.Module):
    """Detect：每個尺度各一組「外框」與「類別」分支，輸出 (批次, 4+類別數, 格點數)，外框是原圖像素的 xyxy"""

    BINS = 16

    def __init__(self, nc, chans, strides):
        super().__init__()
        self.nc = nc
        self.strides = strides
        c_box = max(16, chans[0] // 4, self.BINS * 4)
        c_cls = max(chans[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(ConvBlock(c, c_box, 3), ConvBlock(c_box, c_box, 3),
                                               nn.Conv2d(c_box, 4 * self.BINS, 1)) for c in chans)
        self.cv3 = nn.ModuleList(nn.Sequential(ConvBlock(c, c_cls, 3), ConvBlock(c_cls, c_cls, 3),
                                               nn.Conv2d(c_cls, nc, 1)) for c in chans)
        self.dfl = _Bins(self.BINS)

    def forward(self, feats):
        b = feats[0].shape[0]
        boxes, scores, centers, steps = [], [], [], []
        for i, f in enumerate(feats):
            h, w = f.shape[-2:]
            boxes.append(self.cv2[i](f).view(b, 4 * self.BINS, -1))
            scores.append(self.cv3[i](f).view(b, self.nc, -1))
            ys, xs = torch.meshgrid(torch.arange(h, device=f.device, dtype=f.dtype) + 0.5,
                                    torch.arange(w, device=f.device, dtype=f.dtype) + 0.5, indexing="ij")
            centers.append(torch.stack((xs, ys), -1).view(-1, 2))
            steps.append(torch.full((h * w, 1), float(self.strides[i]), device=f.device, dtype=f.dtype))
        box = torch.cat(boxes, 2)
        center = torch.cat(centers).T.unsqueeze(0)          # (1, 2, 格點)
        step = torch.cat(steps).T.unsqueeze(0)
        n = box.shape[-1]
        prob = box.view(b, 4, self.BINS, n).transpose(2, 1).softmax(1)
        dist = self.dfl.conv(prob).view(b, 4, n)            # 中心到左、上、右、下的距離（格）
        xyxy = torch.cat((center - dist[:, :2], center + dist[:, 2:]), 1) * step
        return torch.cat((xyxy, torch.cat(scores, 2).sigmoid()), 1)


# ---------------------------------------------------------------- 組網路

def _round8(x):
    return int(math.ceil(x / 8) * 8)


class Detector(nn.Module):
    """照模型檔的結構描述組起來的 YOLOv8"""

    def __init__(self, spec):
        super().__init__()
        nc = int(spec["nc"])
        scale = spec.get("scale")
        if spec.get("scales") and scale in spec["scales"]:
            depth, width, max_ch = spec["scales"][scale]
        else:
            depth, width, max_ch = spec.get("depth_multiple", 1.0), spec.get("width_multiple", 1.0), 1024
        layers, sources, chans = [], [], []
        strides, down = [], 1
        for i, (src, repeat, kind, args) in enumerate(spec["backbone"] + spec["head"]):
            repeat = max(round(repeat * depth), 1) if repeat > 1 else repeat
            c_prev = chans[src] if isinstance(src, int) and chans else 3
            if kind in ("Conv", "C2f", "SPPF"):
                c_out = args[0] if args[0] == nc else _round8(min(args[0], max_ch) * width)
                if kind == "Conv":
                    k, s = (args[1] if len(args) > 1 else 1), (args[2] if len(args) > 2 else 1)
                    layer = ConvBlock(c_prev, c_out, k, s)
                elif kind == "C2f":
                    layer = SplitStack(c_prev, c_out, repeat, bool(args[1]) if len(args) > 1 else False)
                else:
                    layer = PoolPyramid(c_prev, c_out, args[1] if len(args) > 1 else 5)
            elif kind == "nn.Upsample":
                layer, c_out = nn.Upsample(scale_factor=args[1], mode=args[2]), c_prev
            elif kind == "Concat":
                layer, c_out = Join(), sum(chans[j] for j in src)
            elif kind in ("Detect", "Segment"):
                ins = [chans[j] for j in src]
                layer, c_out = BoxHead(nc, ins, [8 * 2 ** k for k in range(len(ins))]), 0
            else:
                raise ValueError(i18n.tr("eng.yolo.block", kind=kind))
            layers.append(layer)
            sources.append(src)
            chans.append(c_out)
        self.model = nn.ModuleList(layers)
        self.sources = sources
        self.names = spec.get("names")

    def forward(self, x):
        outs = []
        for layer, src in zip(self.model, self.sources):
            if isinstance(src, int):
                inp = outs[src] if src != -1 or outs else x
            else:
                inp = [outs[j] for j in src]
            x = layer(inp)
            outs.append(x)
        return x


def load(path, device=None):
    spec, weights = read_checkpoint(path)
    net = Detector(spec)
    head = net.model[-1]
    own = set(net.state_dict())
    # 分割模型多的是遮罩分支（proto、cv4），不用；DFL 的固定權重檔案裡有就照檔案
    extra = [k for k in weights if k not in own and not k.split(".")[2:3] in (["proto"], ["cv4"])]
    if extra:
        raise ValueError(i18n.tr("eng.yolo.extra", k=extra[0]))
    weights = {k: v.float() for k, v in weights.items() if k in own and not k.endswith("num_batches_tracked")}
    missing = [k for k in own if k not in weights and not k.endswith("num_batches_tracked")]
    if missing:
        raise ValueError(i18n.tr("eng.yolo.missing", k=missing[0]))
    net.load_state_dict(weights, strict=False)
    if "model.%d.dfl.conv.weight" % (len(net.model) - 1) not in weights:
        head.dfl.conv.weight.data = torch.arange(head.BINS, dtype=torch.float32).view(1, -1, 1, 1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    for m in net.modules():
        if isinstance(m, ConvBlock):
            m.merge_bn()
    net.device = device
    return net


# ---------------------------------------------------------------- 前後處理

def _letterbox(rgb):
    """長邊縮到 640、短邊補到 32 的倍數（兩邊各半、補 114 灰）。回傳 (圖, 縮放比, (左, 上))"""
    import cv2
    h, w = rgb.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_w, pad_h = (IMG_SIZE - nw) % 32 / 2, (IMG_SIZE - nh) % 32 / 2
    if (w, h) != (nw, nh):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(_PAD, _PAD, _PAD))
    return rgb, r, (left, top)


@torch.no_grad()
def find(net, image, conf):
    """PIL 圖 → [(x0, y0, x1, y1, 信心), …]，原圖座標（浮點）"""
    import torchvision
    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    boxed, r, (left, top) = _letterbox(rgb)
    x = torch.from_numpy(np.ascontiguousarray(boxed.transpose(2, 0, 1))).to(net.device).float().unsqueeze(0) / 255
    out = net(x)[0]                                   # (4+類別, 格點)
    score, cls = out[4:].max(0)
    keep = score > conf
    boxes, score, cls = out[:4, keep].T, score[keep], cls[keep]
    if boxes.numel():
        keep = torchvision.ops.batched_nms(boxes, score, cls, IOU)[:MAX_DET]
        boxes, score = boxes[keep], score[keep]
    # 換回原圖座標：減掉實際補的左、上邊（不要重算：896×1344 這類尺寸重算會差一格）、除以縮放比、夾在圖內
    h0, w0 = rgb.shape[:2]
    boxes = boxes.clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clamp(0, w0)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clamp(0, h0)
    return [(*b, s) for b, s in zip(boxes.cpu().tolist(), score.cpu().tolist())]
