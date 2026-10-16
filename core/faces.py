# -*- coding: utf-8 -*-
"""
faces.py —— 修臉（ROADMAP C+.2）

找臉用 ADetailer 的 YOLO 模型（face_yolov8n/s.pt，StabilityMatrix 放在 Models\AfterDetailer），
修則是局部重繪的「只重畫遮罩附近」：每張臉的外框當遮罩，裁切、放大到 1MP 級重畫、貼回
（pipeline._face_fix），所以臉以外的像素逐位元不變。多張臉一張接一張修在同一張圖上。

模型由 CoCo 自己讀、自己跑（core/yolo.py），不用 ultralytics（AGPL-3.0，2026-10-04 拿掉）。
跟 ultralytics 比過圖庫 420 張 × 4 個模型：找到的外框數一樣（只有信心剛好卡在門檻上的一張臉差一個），
座標差不到 0.6 像素。
"""

from PIL import Image, ImageDraw

CONF = 0.3                 # 信心門檻，跟 ADetailer 的預設一樣
_MODELS = {}               # 路徑 → 載好的網路


def _net(path):
    if path not in _MODELS:
        try:
            from core import yolo
        except ImportError:
            import yolo
        _MODELS[path] = yolo.load(path)
    return _MODELS[path]


def detect(image, model_path, conf=CONF):
    """圖（路徑或 PIL）→ {"size": [寬, 高], "faces": [{"box": [x0, y0, x1, y1], "conf": 信心}, …]}

    座標是原圖的像素；由左到右排（多人圖的「第 1 張臉」就是最左邊那張）"""
    im = image if isinstance(image, Image.Image) else Image.open(image)
    im = im.convert("RGB")
    try:
        from core import yolo
    except ImportError:
        import yolo
    faces = []
    for x0, y0, x1, y1, c in yolo.find(_net(str(model_path)), im, conf):
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        faces.append({"box": [max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1)], "conf": round(float(c), 3)})
    faces.sort(key=lambda f: (f["box"][0], f["box"][1]))
    return {"size": [im.width, im.height], "faces": faces}


def scale_box(box, src_size, size):
    """偵測時的原圖座標 → 生成時的尺寸（精修的寬高對齊 16，可能跟原圖差幾個像素）"""
    sx, sy = size[0] / max(1, src_size[0]), size[1] / max(1, src_size[1])
    x0, y0, x1, y1 = box
    return [int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))]


def box_mask(size, box):
    """外框 → 遮罩（白＝要重畫）。邊緣的柔化交給局部重繪的 mask_blur"""
    m = Image.new("L", tuple(size), 0)
    x0, y0, x1, y1 = box
    if x1 > x0 and y1 > y0:
        ImageDraw.Draw(m).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return m
