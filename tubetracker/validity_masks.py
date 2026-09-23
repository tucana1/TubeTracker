"""Spatial validity masks for supervision repair (P2 prep, Qt-free).

The H225 finding: ~4 accepted tips per labeled tip inside dataset
frames receive background targets under full-channel loss. These
builders produce per-patch pixel states for the matched ablation:

  positive: reviewed label discs (human or weak, caller-weighted)
  ignore:   unreviewed accepted-tip discs (unknown, zero gradient)
  background: everything else

Weak (tracker-accepted, unlabeled) tips are IGNORed here, not promoted
to positives — P2 tests human-only and human-plus-weak separately with
explicit weights.
"""

from __future__ import annotations

import math

import numpy as np


def disc_mask(height: int, width: int, x: float, y: float,
              radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius


def rect_mask(height: int, width: int, x0: float, y0: float,
              x1: float, y1: float) -> "np.ndarray":
    """Boolean mask for a native-coordinate rectangle (verified negative)."""
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    lo_x, hi_x = (min(x0, x1), max(x0, x1))
    lo_y, hi_y = (min(y0, y1), max(y0, y1))
    return ((xx >= lo_x) & (xx < hi_x) & (yy >= lo_y) & (yy < hi_y))


def neg_region_mask(height: int, width: int,
                    boxes: list) -> "np.ndarray":
    """Union mask over verified-negative boxes [(x0,y0,x1,y1), ...]."""
    import numpy as np

    out = np.zeros((height, width), dtype=bool)
    for b in boxes or []:
        try:
            out |= rect_mask(height, width, *b)
        except (TypeError, ValueError):
            continue
    return out


def build_validity(
    height: int,
    width: int,
    positive_xy: list[tuple[float, float]],
    weak_xy: list[tuple[float, float]],
    pos_radius: float = 6.0,
    weak_radius: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive_mask, ignore_mask) boolean arrays.

    Ignore discs cover weak tips EXCEPT where they overlap positives
    (a reviewed label always wins its pixels).
    """
    pos = np.zeros((height, width), dtype=bool)
    for x, y in positive_xy:
        pos |= disc_mask(height, width, x, y, pos_radius)
    ign = np.zeros((height, width), dtype=bool)
    for x, y in weak_xy:
        ign |= disc_mask(height, width, x, y, weak_radius)
    ign &= ~pos
    return pos, ign


def masked_targets(
    height: int,
    width: int,
    positive_xy: list[tuple[float, float]],
    weak_xy: list[tuple[float, float]],
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian tip targets + validity mask (1 = supervised, 0 = ignore)."""
    target = np.zeros((height, width), dtype=np.float32)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    for x, y in positive_xy:
        cx, cy = int(round(x)), int(round(y))
        l, r = max(0, cx - radius), min(width, cx + radius + 1)
        t, b = max(0, cy - radius), min(height, cy + radius + 1)
        if l >= r or t >= b:
            continue
        yg, xg = np.ogrid[t:b, l:r]
        target[t:b, l:r] = np.maximum(
            target[t:b, l:r],
            np.exp(-((xg - x) ** 2 + (yg - y) ** 2) / (2 * sigma * sigma)),
        )
    pos, ign = build_validity(height, width, positive_xy, weak_xy)
    valid = ~(ign & (target < 0.5))
    return target.astype(np.float32), valid
