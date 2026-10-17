# -*- coding: utf-8 -*-
"""
gguf.py —— 讀 GGUF 檔（llama.cpp 的模型格式），不需要 gguf 套件

打標台的 JoyCaption 直接用使用者手上的 GGUF（主體＋看圖的 mmproj），不用開 llama-server。
格式（v2／v3）：開頭 "GGUF"、版本、張量數、設定數；設定是「鍵、型別、值」；接著每個張量的名字、維度、型別、
在資料區的位置；資料區從對齊（general.alignment，預設 32）的地方開始。

**維度順序跟 PyTorch 相反**：GGUF 記的第一維是最內層（變化最快的那維），所以 nn.Linear 的 (out, in)
在 GGUF 裡寫成 [in, out]。`shape()` 回 PyTorch 的順序。

讀權重時用 mmap，張量只在要用時才從檔案讀（`raw()` 回指向檔案的 numpy 陣列，不複製）。
"""

import mmap
import struct
from pathlib import Path

import numpy as np

import i18n

# ggml 的張量型別 → (每一塊幾個值, 每一塊幾個位元組)
F32, F16, Q8_0, BF16 = 0, 1, 8, 30
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), BF16: (1, 2)}
TYPE_NAME = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}

# 設定值的型別
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_NP = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32, 7: np.bool_,
       10: np.uint64, 11: np.int64, 12: np.float64}


class GGUF:
    """一個 GGUF 檔：meta（設定）、tensors（名字 → (PyTorch 順序的形狀, 型別, 檔案裡的位置, 位元組數)）"""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        b = self._mm
        if b[:4] != b"GGUF":
            raise ValueError(i18n.tr("eng.gguf.not", name=self.path.name))
        self.version, n_tensors, n_kv = struct.unpack_from("<IQQ", b, 4)
        if self.version not in (2, 3):
            raise ValueError(i18n.tr("eng.gguf.version", v=self.version))
        pos = 24
        self.meta = {}
        for _ in range(n_kv):
            key, pos = self._str(pos)
            typ, = struct.unpack_from("<I", b, pos)
            val, pos = self._value(typ, pos + 4)
            self.meta[key] = val
        infos = []
        for _ in range(n_tensors):
            name, pos = self._str(pos)
            nd, = struct.unpack_from("<I", b, pos)
            dims = struct.unpack_from(f"<{nd}Q", b, pos + 4)
            typ, off = struct.unpack_from("<IQ", b, pos + 4 + 8 * nd)
            pos += 4 + 8 * nd + 12
            infos.append((name, dims, typ, off))
        align = int(self.meta.get("general.alignment", 32))
        data = (pos + align - 1) // align * align
        self.tensors = {}
        for name, dims, typ, off in infos:
            n = int(np.prod(dims))
            per, nbytes = BLOCK.get(typ, (None, None))
            size = n // per * nbytes if per else None
            self.tensors[name] = (tuple(int(d) for d in reversed(dims)), typ, data + off, size)

    def _str(self, pos):
        n, = struct.unpack_from("<Q", self._mm, pos)
        return bytes(self._mm[pos + 8:pos + 8 + n]).decode("utf-8", "replace"), pos + 8 + n

    def _value(self, typ, pos):
        if typ == 8:
            return self._str(pos)
        if typ == 9:
            sub, n = struct.unpack_from("<IQ", self._mm, pos)
            pos += 12
            if sub in _NP:                                  # 數字陣列一次讀完
                arr = np.frombuffer(self._mm, _NP[sub], count=n, offset=pos).copy()
                return arr, pos + arr.nbytes
            out = []
            for _ in range(n):
                v, pos = self._value(sub, pos)
                out.append(v)
            return out, pos
        fmt = _SCALAR.get(typ)
        if fmt is None:
            raise ValueError(i18n.tr("eng.gguf.metaType", t=typ))
        return struct.unpack_from(fmt, self._mm, pos)[0], pos + struct.calcsize(fmt)

    def shape(self, name):
        return self.tensors[name][0]

    def type_of(self, name):
        return self.tensors[name][1]

    def raw(self, name):
        """張量在檔案裡的原始位元組（uint8，指向 mmap，不複製）"""
        shape, typ, off, size = self.tensors[name]
        if size is None:
            raise ValueError(i18n.tr("eng.gguf.type", name=name, t=typ))
        return np.frombuffer(self._mm, np.uint8, count=size, offset=off)

    def array(self, name):
        """F32／F16／BF16 張量 → numpy（PyTorch 順序的形狀，複製一份）；Q8_0 → 還原成 float32"""
        shape, typ, _, _ = self.tensors[name]
        r = self.raw(name)
        if typ == F32:
            return r.view(np.float32).reshape(shape).copy()
        if typ == F16:
            return r.view(np.float16).reshape(shape).copy()
        if typ == BF16:
            return (r.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if typ == Q8_0:
            q = q8_0_split(r)
            return (q[1].astype(np.float32) * q[0].astype(np.float32)[:, None]).reshape(shape)
        raise ValueError(i18n.tr("eng.gguf.type", name=name, t=typ))

    def close(self):
        try:
            self._mm.close()
        except BufferError:
            pass                      # 還有陣列指著檔案：等它們被回收
        self._fh.close()


def q8_0_split(raw):
    """Q8_0 的原始位元組 → (每塊的比例 float16 (n_blocks,), 值 int8 (n_blocks, 32))，都指向原本的記憶體

    一塊 34 位元組：前 2 個是 fp16 的比例 d，後 32 個是 int8 的值；還原＝d × q"""
    blocks = raw.reshape(-1, 34)
    return blocks[:, :2].copy().view(np.float16).reshape(-1), blocks[:, 2:].view(np.int8)
