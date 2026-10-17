# -*- coding: utf-8 -*-
"""
inpaint.py —— 局部重繪（ROADMAP C+.1）

只重畫遮罩裡（白色）的部分，外面**逐位元不變**。兩層保護：

  1. 去噪期間：每一步結束後，把遮罩外的 latent 換回「底圖在下一個 σ 的加噪版本」
     （SDXL 由 diffusers 的 Inpaint 管線做，Anima 由 anima._mask_steps 做），
     模型看得到完整的畫面，遮罩裡畫出來的東西才接得上外面
  2. 解碼之後：遮罩外直接貼回底圖原本的像素，邊緣用柔化過的遮罩混合。
     VAE 編碼再解碼本身就會讓像素飄一點，只靠第 1 層做不到「逐位元不變」

範圍（area）：
  whole   整張圖都丟進去，遮罩只決定哪裡要改
  masked  只裁遮罩附近（加一圈留白）放大到模型的原生解析度重畫，再縮回去貼上。
          臉、手這種小地方的細節會多很多；之後的修臉（C+.2）也用這個

起點（fill）：
  original  遮罩裡照原圖加噪再畫，構圖、顏色會延續（預設）
  fill      先用周圍的顏色把遮罩裡糊掉再畫（A1111 的 Masked content: fill），移除路人、雜物用

塗色（sketch，A1111 的 Inpaint sketch）：精修頁在圖上塗的顏色先蓋在底圖上當起點，
  例如把衣服塗紅再重畫。只在遮罩裡有用；搭配先抹掉時，塗了顏色的地方不抹
"""

from PIL import Image, ImageChops, ImageFilter, ImageOps

MASK_BLUR = 8          # 邊緣柔化（像素）；A1111 預設 4，高解析的圖 8 比較不會看到接縫
MASK_PADDING = 32      # masked 範圍在遮罩外多留幾像素當上下文
TARGET_PX = 1024 * 1024   # masked 範圍放大到大約這麼多像素（SDXL、Anima 都是 1MP 級）
MAX_UPSCALE = 4.0      # 太小的區域最多放大幾倍，再大只是在放大雜訊
STEP = 16              # 寬高對齊（Anima 要 16 的倍數）


def load_mask(src, size):
    """遮罩檔 → 黑白的 L 圖，大小跟底圖一樣。白＝要重畫。
    透明 PNG 看 alpha（畫布匯出的常是這種），其他看亮度"""
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        m = im.convert("RGBA").getchannel("A")
    else:
        m = im.convert("L")
    if m.size != tuple(size):
        m = m.resize(tuple(size), Image.NEAREST)
    return m.point(lambda v: 255 if v >= 128 else 0)


def feather(mask, blur=MASK_BLUR):
    """柔化邊緣。往外擴一點再模糊，遮罩的邊一定落在重畫過的範圍裡"""
    blur = max(0, int(blur))
    if not blur:
        return mask.copy()
    return mask.filter(ImageFilter.MaxFilter(blur // 2 * 2 + 1)).filter(ImageFilter.GaussianBlur(blur / 2))


def _snap(v):
    return max(STEP * 4, int(round(v / STEP)) * STEP)


def plan(size, soft, area="whole", padding=MASK_PADDING, target_px=TARGET_PX, max_up=MAX_UPSCALE):
    """決定裁哪一塊、放大到多大。回傳 {box: (x0, y0, x1, y1), work: (w, h)}；遮罩是空的回 None

    whole：整張，工作大小就是原圖大小（已經是 16 的倍數，見 server/refine.snap）
    masked：遮罩的外框加 padding，縮放到大約 target_px 像素（最多放大 max_up 倍；修臉的臉常常很小，放寬到 8）
    """
    bbox = soft.getbbox()
    if not bbox:
        return None
    W, H = size
    if area != "masked":
        return {"box": (0, 0, W, H), "work": (W, H)}
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(W, x1 + padding), min(H, y1 + padding)
    bw, bh = x1 - x0, y1 - y0
    scale = min(max_up, (target_px / (bw * bh)) ** 0.5)
    return {"box": (x0, y0, x1, y1), "work": (_snap(bw * scale), _snap(bh * scale))}


def crop_inputs(full, soft, p):
    """依 plan 裁出要重畫的底圖與遮罩，縮放到工作大小"""
    box, work = p["box"], p["work"]
    img = full.crop(box).resize(work, Image.LANCZOS)
    m = soft.crop(box).resize(work, Image.BILINEAR)
    return img, m


def fill(img, mask):
    """遮罩裡換成周圍顏色糊開的樣子（跟 A1111 masking.fill 同一個做法）：
    遮罩外的像素由大到小一層層模糊疊上去，最後一層半徑 0 是原圖本身，所以遮罩外不變。
    用 premultiplied 的 RGBa 模糊，透明的遮罩區才不會把黑色混進來"""
    base = Image.new("RGBa", img.size)
    base.paste(img.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))
    out = Image.new("RGBA", img.size)
    for radius, repeats in ((256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)):
        blurred = (base.filter(ImageFilter.GaussianBlur(radius)) if radius else base).convert("RGBA")
        for _ in range(repeats):
            out.alpha_composite(blurred)
    return out.convert("RGB")


def load_sketch(src, size):
    """塗色檔 → RGBA，大小跟底圖一樣"""
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im.load()
    im = im.convert("RGBA")
    return im if im.size == tuple(size) else im.resize(tuple(size), Image.NEAREST)


def apply_sketch(full, sketch, soft):
    """把塗的顏色蓋在底圖上。只蓋遮罩裡的：遮罩外之後會貼回原圖，蓋了也只會讓模型看到不存在的東西"""
    alpha = ImageChops.darker(sketch.getchannel("A"), soft)
    return Image.composite(sketch.convert("RGB"), full, alpha)


def unsketched(soft, sketch):
    """先抹掉的範圍：遮罩裡、沒塗顏色的地方（塗了顏色的就是使用者要的起點，不抹）"""
    return ImageChops.subtract(soft, sketch.getchannel("A"))


def paste_back(full, gen, soft, p):
    """重畫好的（工作大小）縮回原本那一塊，用柔化遮罩跟底圖混合後貼回。
    遮罩值 0 的地方就是底圖的原像素，一個位元都不差"""
    box = p["box"]
    size = (box[2] - box[0], box[3] - box[1])
    gen = gen.convert("RGB")
    if gen.size != size:
        gen = gen.resize(size, Image.LANCZOS)
    region = Image.composite(gen, full.crop(box), soft.crop(box))
    out = full.copy()
    out.paste(region, box[:2])
    return out


def latent_mask(mask, lat_w, lat_h):
    """工作大小的柔化遮罩 → latent 大小的 0/1（float 陣列）。
    只要格子裡有一點要重畫就算整格：柔化的邊要落在重畫過的範圍裡"""
    import numpy as np
    small = mask.resize((lat_w, lat_h), Image.BOX)
    return (np.asarray(small, dtype=np.float32) > 0.5).astype(np.float32)
