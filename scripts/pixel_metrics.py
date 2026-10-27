"""Objective pixel checks for patent-drawing style. Pure NumPy, no GPU.

Real patent figures are black lines on pure white. These three numbers capture
how close a generated image is to that:

  color_leak    share of pixels with visible color (saturation > 0.15 and not near-black).
                Should be ~0 for a good result.
  border_white  share of near-white pixels in a strip around the edge.
                Should be ~1: patent figures float on a white page.
  gray_share    share of mid-gray pixels (luminance 40-215).
                Line art is mostly pure black or pure white, so lower is better.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _rgb(img: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an RGB image")
    return arr / 255.0


def color_leak(img, sat_min: float = 0.15, val_min: float = 0.2) -> float:
    a = _rgb(img)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(((sat > sat_min) & (mx > val_min)).mean())


def border_white(img, frac: float = 0.05, thresh: float = 240 / 255) -> float:
    a = _rgb(img)
    h, w = a.shape[:2]
    b = max(1, int(round(min(h, w) * frac)))
    lum = a.mean(axis=2)
    strip = np.concatenate([lum[:b].ravel(), lum[-b:].ravel(), lum[b:-b, :b].ravel(), lum[b:-b, -b:].ravel()])
    return float((strip >= thresh).mean())


def gray_share(img, lo: int = 40, hi: int = 215) -> float:
    lum = _rgb(img).mean(axis=2) * 255
    return float(((lum > lo) & (lum < hi)).mean())


def all_metrics(img) -> dict[str, float]:
    return {"color_leak": color_leak(img), "border_white": border_white(img), "gray_share": gray_share(img)}
