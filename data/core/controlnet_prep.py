# -*- coding: utf-8 -*-
"""ControlNet 的前處理：參考圖 → ControlNet 吃的控制圖（工作行程裡跑）。

  dwpose    抽姿勢：YOLOX 找人、DWPose（RTMPose 架構，COCO-WholeBody 133 點）找身體、手、臉的點，
            畫成 OpenPose 的彩色骨架（黑底）。兩個都是 ONNX，用 opencv 的 dnn 在 CPU 跑，不用另外裝套件
  depth     深度：Depth Anything V2（transformers），近白遠黑
  canny     Canny 輪廓（opencv），白線黑底
  lineart   線稿：XDoG（高斯差分加柔化門檻），白線黑底；不用模型，適合動漫圖的乾淨線條
  softedge  軟邊緣：多個尺度的梯度強度疊起來、柔化，白線黑底；不用模型
  tile      原圖（ControlNet Tile 照原圖的顏色與構圖）
  none      參考圖已經是姿勢圖／線稿／深度圖，直接用

模型都只在用到時載入、留在系統記憶體（DWPose 約 350 MB，深度約 100 MB），閒著不佔顯示卡；
閒置一陣子跟 ControlNet 一起卸載（controlnet_run 的倒數）。
骨架的畫法（點的順序、顏色、粗細）照 OpenPose 的慣例，ControlNet 的姿勢模型都是用這種圖訓練的。
"""

import math
import threading
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()
_NETS = {}                 # 模型檔 → 載好的東西

# ── 共用 ──
def _rgb(img):
    """PIL → HxWx3 uint8 RGB"""
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _to_pil(arr):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))


# ── Canny／線稿／軟邊緣（不用模型）──
def canny(img, low=100, high=200):
    import cv2
    g = cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, int(low), int(high))
    return _to_pil(np.stack([e] * 3, -1))


def lineart(img, sigma=0.8, k=1.6, p=20.0, eps=-0.02, phi=10.0):
    """XDoG：兩個高斯模糊相減（加強過的），再用柔化的門檻變成線；回傳白線黑底"""
    import cv2
    g = cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    s = max(0.6, sigma * max(g.shape) / 1024)        # 大圖的線也一樣粗細
    g1 = cv2.GaussianBlur(g, (0, 0), s)
    g2 = cv2.GaussianBlur(g, (0, 0), s * k)
    d = (1 + p) * g1 - p * g2
    x = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    line = (1.0 - np.clip(x, 0, 1)) * 255.0
    return _to_pil(np.stack([line] * 3, -1))


def softedge(img):
    """幾個尺度的梯度強度平均、正規化、柔化：像 HED 那種有粗細、比較軟的邊"""
    import cv2
    g = cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    base = max(1.0, max(g.shape) / 1024)
    acc = np.zeros_like(g)
    for s in (1.0, 2.0, 4.0):
        b = cv2.GaussianBlur(g, (0, 0), s * base)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        acc += np.sqrt(gx * gx + gy * gy)
    hi = np.percentile(acc, 99.5) or 1.0
    e = np.clip(acc / hi, 0, 1) ** 0.8
    e = cv2.GaussianBlur(e, (0, 0), 0.8 * base)
    return _to_pil(np.stack([e * 255.0] * 3, -1))


# ── 深度：Depth Anything V2 ──
def depth(img, files):
    import torch
    from transformers import DepthAnythingForDepthEstimation
    folder = str(Path(files["model.safetensors"]).parent)
    with _LOCK:
        m = _NETS.get(folder)
        if m is None:
            m = DepthAnythingForDepthEstimation.from_pretrained(folder, use_safetensors=True, local_files_only=True).eval()
            _NETS[folder] = m
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rgb = _rgb(img)
    h, w = rgb.shape[:2]
    # 跟官方的前處理一樣：短邊縮到 518（14 的倍數、保持比例），ImageNet 標準化
    k = 518 / min(h, w)
    th, tw = max(14, round(h * k / 14) * 14), max(14, round(w * k / 14) * 14)
    import cv2
    x = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    try:
        m.to(dev)
        with torch.inference_mode():
            pred = m(pixel_values=t.to(dev)).predicted_depth[0].float().cpu().numpy()
    finally:
        m.to("cpu")
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(pred, 1), np.percentile(pred, 99)
    d = np.clip((pred - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0      # 近白遠黑（Depth Anything 輸出的是越近越大）
    return _to_pil(np.stack([d] * 3, -1))


# ── 姿勢：YOLOX＋DWPose（ONNX，opencv dnn）──
def _net(path):
    import cv2
    with _LOCK:
        if path not in _NETS:
            _NETS[path] = cv2.dnn.readNetFromONNX(path)
        return _NETS[path]


def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def find_people(bgr, model, score=0.3):
    """YOLOX：畫面裡的人 → [[x1, y1, x2, y2], …]（原圖座標）。
    偵測器是用真人照片訓練的，動漫圖常常認不出是「人」（實測跳舞的女孩被當成風箏，分數 0.8）：
    找不到人時改用分數夠高的其他物體框，再找不到就整張圖當一個框；之後由 dwpose() 照關節分數篩掉不是人的"""
    import cv2
    h, w = bgr.shape[:2]
    r = min(640 / h, 640 / w)
    pad = np.full((640, 640, 3), 114, np.uint8)
    pad[:int(h * r), :int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    net = _net(model)
    net.setInput(pad.transpose(2, 0, 1)[None].astype(np.float32))
    out = net.forward()[0]                                   # 8400 × 85：每格的框、物體分數、80 類分數
    grids, strides = [], []
    for s in (8, 16, 32):
        n = 640 // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    grids, strides = np.concatenate(grids), np.concatenate(strides)
    xy = (out[:, :2] + grids) * strides
    wh = np.exp(out[:, 2:4]) * strides
    whole = np.array([0, 0, w, h], np.float32)
    conf = out[:, 4] * out[:, 5]                             # 第 0 類是人
    m = conf > score
    fallback = not m.any()
    if fallback:
        conf = out[:, 4] * out[:, 5:].max(1)                 # 備援：任何東西，再加上整張圖
        m = conf > score
    if not m.any():
        return [whole]
    xy, wh, conf = xy[m], wh[m], conf[m]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
    keep = _nms(boxes, conf, 0.45)
    boxes = boxes[keep]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    found = [b for b in boxes if (b[2] - b[0]) > 8 and (b[3] - b[1]) > 8]
    return found + [whole] if fallback else found


def _affine(center, scale, out_w, out_h, inverse=False):
    import cv2
    src = np.array([center, center + [0, -scale[1] / 2], center + [-scale[0] / 2, -scale[1] / 2]], np.float32)
    dst = np.array([[out_w / 2, out_h / 2], [out_w / 2, 0], [0, 0]], np.float32)
    return cv2.getAffineTransform(dst, src) if inverse else cv2.getAffineTransform(src, dst)


def body_points(rgb, box, model):
    """DWPose：一個人的框 → 133 個點 (x, y) 與分數"""
    import cv2
    W, H = 288, 384
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale = np.array([(x2 - x1) * 1.25, (y2 - y1) * 1.25], np.float32)
    if scale[0] / scale[1] > W / H:                          # 補成模型要的比例
        scale[1] = scale[0] * H / W
    else:
        scale[0] = scale[1] * W / H
    crop = cv2.warpAffine(rgb, _affine(center, scale, W, H), (W, H), flags=cv2.INTER_LINEAR)
    x = (crop.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    net = _net(model)
    net.setInput(x.transpose(2, 0, 1)[None].astype(np.float32))
    sx, sy = net.forward(["simcc_x", "simcc_y"])
    sx, sy = sx[0], sy[0]
    px, py = sx.argmax(1) / 2.0, sy.argmax(1) / 2.0           # SimCC 的格子是像素的兩倍細
    score = np.minimum(sx.max(1), sy.max(1))
    pts = np.stack([px, py], 1)
    pts = pts / [W, H] * scale + center - scale / 2
    return pts, score


# OpenPose 的 18 點身體：COCO-WholeBody 的前 17 點補上脖子（兩肩中點），再換成 OpenPose 的順序
_MM = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
_OP = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
_LIMBS = [(2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
          (2, 1), (1, 15), (15, 17), (1, 16), (16, 18)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def to_openpose(pts, score):
    """133 點 → (身體 18 點, 身體分數, 左手 21, 右手 21, 臉 68 及各自的分數)"""
    neck = (pts[5] + pts[6]) / 2
    nscore = min(score[5], score[6])
    p = np.insert(pts, 17, neck, 0)
    s = np.insert(score, 17, nscore, 0)
    bp, bs = p.copy(), s.copy()
    bp[_OP], bs[_OP] = p[_MM], s[_MM]
    return {"body": (bp[:18], bs[:18]), "face": (p[24:92], s[24:92]),
            "lhand": (p[92:113], s[92:113]), "rhand": (p[113:134], s[113:134])}


def draw_pose(people, w, h, thr=0.3, hands=True, face=True):
    """OpenPose 的畫法：黑底、身體 17 條彩色橢圓棒（半透明疊）、關節圓點；手 20 條、臉白點"""
    import colorsys
    import cv2
    canvas = np.zeros((h, w, 3), np.uint8)
    unit = max(1.0, max(w, h) / 1024)
    stick = max(2, int(4 * unit))
    for p in people:
        pts, sc = p["body"]
        for i, (a, b) in enumerate(_LIMBS):
            a, b = a - 1, b - 1
            if sc[a] < thr or sc[b] < thr:
                continue
            (x1, y1), (x2, y2) = pts[a], pts[b]
            length = math.hypot(x1 - x2, y1 - y2)
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            poly = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2), stick), int(angle), 0, 360, 1)
            layer = canvas.copy()
            cv2.fillConvexPoly(layer, poly, _COLORS[i])
            canvas = cv2.addWeighted(canvas, 0.4, layer, 0.6, 0)
        for i in range(18):
            if sc[i] >= thr:
                cv2.circle(canvas, (int(pts[i][0]), int(pts[i][1])), stick, _COLORS[i], -1)
    for p in people:
        if hands:
            for key in ("lhand", "rhand"):
                pts, sc = p[key]
                for i, (a, b) in enumerate(_HAND):
                    if sc[a] < thr or sc[b] < thr:
                        continue
                    rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(_HAND), 1.0, 1.0))
                    cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), rgb, max(1, int(2 * unit)))
                for i in range(21):
                    if sc[i] >= thr:
                        cv2.circle(canvas, tuple(int(v) for v in pts[i]), max(1, int(3 * unit)), (0, 0, 255), -1)
        if face:
            pts, sc = p["face"]
            for i in range(len(pts)):
                if sc[i] >= thr:
                    cv2.circle(canvas, tuple(int(v) for v in pts[i]), max(1, int(2 * unit)), (255, 255, 255), -1)
    return canvas


def dwpose(img, files, hands=True, face=True):
    """參考圖 → 姿勢圖（跟參考圖一樣大）。找不到人回全黑的圖（ControlNet 就不會限制姿勢）"""
    import cv2
    rgb = _rgb(img)
    h, w = rgb.shape[:2]
    boxes = find_people(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), files["yolox_l.onnx"])[:10]
    people, kept = [], []
    for b in boxes:
        pts, sc = body_points(rgb, b, files["dw-ll_ucoco_384.onnx"])
        conf = np.sort(sc[:17])[-8:].mean()                 # 至少大半個身體的點有把握，才算是人
        if conf < 0.35:
            continue
        # 備援時同一個人可能被物體框和整張圖各找到一次：身體的中心很近的只留一個
        c = pts[:17][sc[:17] >= 0.3].mean(0)
        if any(np.hypot(*(c - c2)) < 0.05 * max(w, h) for c2 in kept):
            continue
        kept.append(c)
        people.append(to_openpose(pts, sc))
    return _to_pil(draw_pose(people, w, h, hands=hands, face=face)), len(people)


# ── 入口 ──
def run(img, prep, files=None, low=100, high=200):
    """參考圖（PIL）→ (控制圖 PIL, 說明 dict)"""
    files = files or {}
    if prep == "dwpose":
        out, n = dwpose(img, files)
        return out, {"people": n}
    if prep == "depth":
        return depth(img, files), {}
    if prep == "canny":
        return canny(img, low, high), {}
    if prep == "lineart":
        return lineart(img), {}
    if prep == "softedge":
        return softedge(img), {}
    return img.convert("RGB"), {}                           # tile、none：原圖


def loaded():
    return bool(_NETS)


def release():
    """模型都放掉（閒置時，controlnet_run 的倒數呼叫）"""
    with _LOCK:
        _NETS.clear()
