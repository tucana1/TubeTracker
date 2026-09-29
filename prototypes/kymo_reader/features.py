"""Network inputs and targets from stored kymograph samples.

Dynamic input (per bin t and arc s), 35 channels: I_t - B on the turned path (17 lateral offsets),
the same on the unturned path (17; equal to the first block when no rotation track was given), and
the "sample inside the movie" flag. Static input (per arc s), 36 channels: E - B and B (17 lateral
offsets each), the path point's distance from its grain's rim, and its distance to the nearest other
grain's rim. Grey levels are divided by 10 (dynamic, E - B) or 20 (B) and clipped.
"""

from __future__ import annotations

import numpy as np

from . import store

C_DYN, C_STAT = 35, 36
W = 17


def inputs(sm: dict, t0: int = 0, t1: int | None = None, s0: int = 0, s1: int | None = None,
           gain: float = 1.0, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(dyn (35, T, S), stat (36, S)) float32 for one stored sample, cropped to [t0, t1) x [s0, s1)."""
    d = store.dequant(sm["dyn"][t0:t1, s0:s1])                       # (T, S, W)
    d0 = store.dequant(sm["dyn0"][t0:t1, s0:s1]) if "dyn0" in sm else d
    T, S, w = d.shape
    lat = slice(None, None, -1) if flip else slice(None)
    dyn = np.empty((2 * w + 1, T, S), np.float32)
    dyn[:w] = np.clip(np.moveaxis(d[:, :, lat], 2, 0) * (gain / 10.0), -6.0, 6.0)
    dyn[w:2 * w] = np.clip(np.moveaxis(d0[:, :, lat], 2, 0) * (gain / 10.0), -6.0, 6.0)
    dyn[2 * w] = sm["valid"][t0:t1, s0:s1].astype(np.float32)
    stat = np.empty((2 * w + 2, S), np.float32)
    stat[:w] = np.clip(sm["eb"][s0:s1, lat].astype(np.float32).T * (gain / 10.0), -6.0, 6.0)
    stat[w:2 * w] = np.clip(sm["bb"][s0:s1, lat].astype(np.float32).T / 20.0, -6.0, 6.0)
    stat[2 * w] = np.clip(sm["rim"][s0:s1].astype(np.float32), -10.0, 30.0) / 10.0
    stat[2 * w + 1] = np.clip(sm["other"][s0:s1].astype(np.float32).min(axis=1), -5.0, 30.0) / 10.0
    return dyn, stat


def target(sm: dict, t0: int = 0, t1: int | None = None, s0: int = 0, s1: int | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """(target, mask) over the crop: Y(t, s) = "arc s is covered by bin t" (grain body, or tube up to
    its front). Synthetic: exact, Y = [s < L(t)] with s = 0 where the drawn tube starts (the census
    circle), except that where the grain's visible edge lies (s in [-4, 2]) nothing is asserted while
    the tube is shorter than 2 px: where a length starts is the annotator's convention, which only
    the real labels teach. Real: the partial labels."""
    if "target" in sm:
        L = sm["target"][t0:t1].astype(np.float32)
        s = sm["s"][s0:s1].astype(np.float32)
        y = (s[None, :] < L[:, None]).astype(np.float32)
        m = np.ones_like(y)
        m[(L[:, None] < 2.0) & (s[None, :] >= -4.0) & (s[None, :] <= 2.0)] = 0.0
        return y, m
    return sm["lab_y"][t0:t1, s0:s1].astype(np.float32), sm["lab_m"][t0:t1, s0:s1].astype(np.float32)
