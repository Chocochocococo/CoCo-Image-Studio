# -*- coding: utf-8 -*-
"""
birefnet.py —— BiRefNet 架構（去背用），在工作行程裡跑

照官方實作（ZhengPeng7/BiRefNet，MIT；Hugging Face 上的 birefnet.py）的推論部分改寫，
不從網路抓程式碼來執行，也不需要 timm、kornia、einops。設定跟官方釋出的權重一樣：
  主幹 Swin Transformer（T／S／B／L，從權重的形狀判斷）、同一張圖原尺寸與一半大小各跑一次再接起來（mul_scl_ipt=cat）、
  最深那層接上前三層（cxt=3）、squeeze 一個 BasicDecBlk、解碼器的注意力是可變形卷積的 ASPP、
  每層解碼前接上原圖切成的小塊（dec_ipt_split）、梯度注意力（out_ref）。
訓練才用到的分支（多尺度監督、梯度預測）權重裡有，這裡也留著層（嚴格載入），推論不會跑到。

推論：RGB 縮成 1024×1024、ImageNet 的平均與標準差標準化，取最後一個輸出套 sigmoid。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(x, ref_or_size):
    size = ref_or_size if isinstance(ref_or_size, (tuple, list, torch.Size)) else ref_or_size.shape[2:]
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


# ── Swin Transformer（主幹）──
class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), heads))
        coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        n = self.ws * self.ws
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        bias = bias.unsqueeze(0).to(q.dtype)                    # 1, nH, N, N
        if mask is not None:                                    # 位移過的視窗：不同區塊之間不互相看
            nW = mask.shape[0]
            bias = (bias.unsqueeze(1) + mask.to(q.dtype).unsqueeze(1).unsqueeze(0)).expand(B_ // nW, nW, -1, N, N).reshape(B_, self.heads, N, N)
        # 跟官方一樣是 softmax(q·k×scale＋bias)·v，用 PyTorch 的 SDPA 算比較省記憶體
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B_, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)
        self.H = self.W = None

    def forward(self, x, mask):
        B, L, C = x.shape
        H, W = self.H, self.W
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r, pad_b = (self.ws - W % self.ws) % self.ws, (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        win = self.attn(win, mask=mask if self.shift > 0 else None).view(-1, self.ws, self.ws, C)
        x = window_reverse(win, self.ws, Hp, Wp)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        if pad_r or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class SwinLayer(nn.Module):
    def __init__(self, dim, depth, heads, ws, downsample):
        super().__init__()
        self.ws, self.shift = ws, ws // 2
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        ws, sh = self.ws, self.shift
        Hp, Wp = math.ceil(H / ws) * ws, math.ceil(W / ws) * ws
        img = torch.zeros((1, Hp, Wp, 1), device=x.device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -sh), slice(-sh, None)):
            for wsl in (slice(0, -ws), slice(-ws, -sh), slice(-sh, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img, ws).view(-1, ws * ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0).to(x.dtype)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, mask)
        if self.downsample is not None:
            return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
        return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        _, _, H, W = x.shape
        if W % 4:
            x = F.pad(x, (0, 4 - W % 4))
        if H % 4:
            x = F.pad(x, (0, 0, 0, 4 - H % 4))
        x = self.proj(x)
        Wh, Ww = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(-1, self.dim, Wh, Ww)


class SwinTransformer(nn.Module):
    def __init__(self, dim, depths, heads, ws):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.layers = nn.ModuleList([SwinLayer(dim * 2 ** i, depths[i], heads[i], ws, i < len(depths) - 1) for i in range(len(depths))])
        self.num_features = [dim * 2 ** i for i in range(len(depths))]
        for i, n in enumerate(self.num_features):
            self.add_module(f"norm{i}", nn.LayerNorm(n))

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            x_out = getattr(self, f"norm{i}")(x_out)
            outs.append(x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)


# ── 解碼器 ──
class DeformableConv2d(nn.Module):
    def __init__(self, cin, cout, k, padding):
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(cin, 2 * k * k, k, 1, padding, bias=True)
        self.modulator_conv = nn.Conv2d(cin, k * k, k, 1, padding, bias=True)
        self.regular_conv = nn.Conv2d(cin, cout, k, 1, padding, bias=False)

    def forward(self, x):
        from torchvision.ops import deform_conv2d
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                             padding=self.padding, mask=modulator, stride=(1, 1))


class _ASPPDeform(nn.Module):
    def __init__(self, cin, planes, k, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(cin, planes, k, padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, cin, sizes=(1, 3, 7)):
        super().__init__()
        mid = 256
        self.aspp1 = _ASPPDeform(cin, mid, 1, 0)
        self.aspp_deforms = nn.ModuleList([_ASPPDeform(cin, mid, k, k // 2) for k in sizes])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(cin, mid, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid * (2 + len(sizes)), cin, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x5 = _up(self.global_avg_pool(x), x1)
        x = torch.cat((x1, *[m(x) for m in self.aspp_deforms], x5), dim=1)
        return self.relu(self.bn1(self.conv1(x)))


class BasicDecBlk(nn.Module):
    def __init__(self, cin, cout, inter=64):
        super().__init__()
        self.conv_in = nn.Conv2d(cin, inter, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(inter)
        self.conv_out = nn.Conv2d(inter, cout, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter)
        self.bn_out = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        return self.bn_out(self.conv_out(self.dec_att(x)))


class BasicLatBlk(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, cin, cout, inter=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, inter, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter, cout, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def patches_to_channels(x, ref):
    """原圖切成 ref 大小的小塊、疊到通道上（官方的 image2patches：b c (hg h) (wg w) -> b (c hg wg) h w）"""
    b, c, H, W = x.shape
    hg, wg = H // ref.shape[-2], W // ref.shape[-1]
    h, w = H // hg, W // wg
    return x.view(b, c, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg, h, w)


def _gdt(cin, n=16):
    return nn.Sequential(nn.Conv2d(cin, n, 3, 1, 1), nn.BatchNorm2d(n), nn.ReLU(inplace=True))


class Decoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ipt_blk5 = SimpleConvs(2 ** 10 * 3, ch[0] // 8)
        self.ipt_blk4 = SimpleConvs(2 ** 8 * 3, ch[0] // 8)
        self.ipt_blk3 = SimpleConvs(2 ** 6 * 3, ch[1] // 8)
        self.ipt_blk2 = SimpleConvs(2 ** 4 * 3, ch[2] // 8)
        self.ipt_blk1 = SimpleConvs(2 ** 0 * 3, ch[3] // 8)
        self.decoder_block4 = BasicDecBlk(ch[0] + ch[0] // 8, ch[1])
        self.decoder_block3 = BasicDecBlk(ch[1] + ch[0] // 8, ch[2])
        self.decoder_block2 = BasicDecBlk(ch[2] + ch[1] // 8, ch[3])
        self.decoder_block1 = BasicDecBlk(ch[3] + ch[2] // 8, ch[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(ch[3] // 2 + ch[3] // 8, 1, 1, 1, 0))
        self.lateral_block4 = BasicLatBlk(ch[1], ch[1])
        self.lateral_block3 = BasicLatBlk(ch[2], ch[2])
        self.lateral_block2 = BasicLatBlk(ch[3], ch[3])
        # 訓練用（多尺度監督、梯度預測）：推論不跑，留著是為了嚴格載入
        self.conv_ms_spvn_4 = nn.Conv2d(ch[1], 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(ch[2], 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(ch[3], 1, 1, 1, 0)
        self.gdt_convs_4, self.gdt_convs_3, self.gdt_convs_2 = _gdt(ch[1]), _gdt(ch[2]), _gdt(ch[3])
        for i in (4, 3, 2):
            setattr(self, f"gdt_convs_pred_{i}", nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0)))
            setattr(self, f"gdt_convs_attn_{i}", nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0)))

    def forward(self, x, x1, x2, x3, x4):
        def ipt(blk, ref, size):
            return blk(_up(patches_to_channels(x, ref), size))
        x4 = torch.cat((x4, ipt(self.ipt_blk5, x4, x4.shape[2:])), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p3 = _up(p4, x3) + self.lateral_block4(x3)
        _p3 = torch.cat((_p3, ipt(self.ipt_blk4, _p3, x3.shape[2:])), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p2 = _up(p3, x2) + self.lateral_block3(x2)
        _p2 = torch.cat((_p2, ipt(self.ipt_blk3, _p2, x2.shape[2:])), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p1 = _up(p2, x1) + self.lateral_block2(x1)
        _p1 = torch.cat((_p1, ipt(self.ipt_blk2, _p1, x1.shape[2:])), 1)
        _p1 = _up(self.decoder_block1(_p1), x)
        _p1 = torch.cat((_p1, ipt(self.ipt_blk1, _p1, x.shape[2:])), 1)
        return self.conv_out1(_p1)


class BiRefNet(nn.Module):
    def __init__(self, dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), ws=7):
        super().__init__()
        self.bb = SwinTransformer(dim, list(depths), list(heads), ws)
        ch = [dim * 8 * 2, dim * 4 * 2, dim * 2 * 2, dim * 2]        # 原尺寸與一半大小的特徵接在一起，所以 ×2
        self.cxt = ch[1:][::-1]                                      # 最深那層接上前三層
        self.squeeze_module = nn.Sequential(BasicDecBlk(ch[0] + sum(self.cxt), ch[0]))
        self.decoder = Decoder(ch)

    def forward(self, x):
        x1, x2, x3, x4 = self.bb(x)
        H, W = x.shape[2:]
        y1, y2, y3, y4 = self.bb(F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True))
        x1, x2, x3, x4 = (torch.cat([a, _up(b, a)], dim=1) for a, b in ((x1, y1), (x2, y2), (x3, y3), (x4, y4)))
        x4 = torch.cat((_up(x1, x4), _up(x2, x4), _up(x3, x4), x4), dim=1)
        x4 = self.squeeze_module(x4)
        return self.decoder(x, x1, x2, x3, x4)


def config_from(shapes):
    """權重的名字與形狀（不含前綴）→ Swin 主幹的設定 {dim, depths, heads, ws}；看不懂回 None"""
    pe = shapes.get("bb.patch_embed.proj.weight")
    if not pe or list(pe[1:]) != [3, 4, 4]:
        return None
    depths, heads, ws = [], [], None
    for i in range(4):
        n = 0
        while f"bb.layers.{i}.blocks.{n}.norm1.weight" in shapes:
            n += 1
        tab = shapes.get(f"bb.layers.{i}.blocks.0.attn.relative_position_bias_table")
        if not n or not tab:
            return None
        side = int(round(math.sqrt(tab[0])))
        ws = (side + 1) // 2
        depths.append(n)
        heads.append(tab[1])
    return {"dim": pe[0], "depths": depths, "heads": heads, "ws": ws}
