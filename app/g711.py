"""Telephone audio conversion (pure numpy, works on every Python version).

Phone networks carry 8 kHz G.711 mu-law. Our engine speaks 16 kHz PCM in and 24 kHz PCM out, so we convert:
    phone -> engine :  mu-law 8 kHz  ->  PCM16 8 kHz  ->  PCM16 16 kHz   (linear interpolation)
    engine -> phone :  PCM16 24 kHz  ->  PCM16 8 kHz  ->  mu-law 8 kHz   (average of 3 samples = simple low-pass)
"""
from __future__ import annotations

import numpy as np

_BIAS = 0x84
_CLIP = 32635


def _decode_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sample = (((u & 0x0F) << 3) + _BIAS) << ((u >> 4) & 7)
        sample -= _BIAS
        t[i] = -sample if (u & 0x80) else sample
    return t


_DEC = _decode_table()


def ulaw_to_pcm(data: bytes) -> np.ndarray:
    return _DEC[np.frombuffer(data, dtype=np.uint8)]


def pcm_to_ulaw(pcm: np.ndarray) -> bytes:
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    exp = np.frexp(x.astype(np.float64))[1].astype(np.int32) - 8      # bit length - 8  (0..7)
    mant = (x >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


class Upsampler8to16:
    """8 kHz -> 16 kHz by linear interpolation; keeps one sample of state so chunk edges stay smooth."""

    def __init__(self) -> None:
        self.last = 0.0

    def __call__(self, pcm8: np.ndarray) -> np.ndarray:
        if pcm8.size == 0:
            return pcm8.astype(np.int16)
        x = pcm8.astype(np.float32)
        prev = np.concatenate(([self.last], x[:-1]))
        out = np.empty(x.size * 2, dtype=np.float32)
        out[0::2] = (prev + x) / 2     # midpoint between previous and current sample
        out[1::2] = x
        self.last = float(x[-1])
        return np.clip(out, -32768, 32767).astype(np.int16)


def down_24k_to_8k(pcm24: np.ndarray) -> np.ndarray:
    n = pcm24.size // 3 * 3
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    return pcm24[:n].astype(np.float32).reshape(-1, 3).mean(axis=1).astype(np.int16)


def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2))) if pcm.size else 0.0
