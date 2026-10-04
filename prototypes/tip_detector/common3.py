"""Version 3 of the tip detector: inputs with several lags (prototypes/tip_detector/README.md, version 3).

Channels at bin b (registered, reference coordinates; M(k) = mean of bins k-1..k+1 with k clamped to
[ref_start + 1, n_bins - 2], so every window has three bins; E = M(ref_start + 1), the reference bins):
  A    M(b) less its local median                           / 20 grey levels   (as version 2)
  D6   M(b) - M(b-6), less its local median                 / 8                (as version 2)
  D12  M(b) - M(b-12), less its local median                / 8                (new: a slow tip still shows)
  D24  M(b) - M(b-24), less its local median                / 8                (new: the recent growth as a streak)
  C    M(b) - E, less its local median                      / 20               (as version 2)
Where b - lag falls before the reference bins the window is clamped there (then D_lag = C before the median).
Inputs are clipped to +/- CLIP units in training and inference alike (samples are stored as int8, 1/24 unit a step).
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import MARGIN, Movie, medbg

CHANNELS3 = ("A", "D6", "D12", "D24", "C")
LAGS = {"D6": 6, "D12": 12, "D24": 24}
SCALE3 = {"A": 20.0, "D6": 8.0, "D12": 8.0, "D24": 8.0, "C": 20.0}
Q = 24.0                 # int8 steps per input unit
CLIP = 127.0 / Q         # +/- 5.29 units


class Movie3(Movie):
    """A Movie with the version 3 inputs (any subset of CHANNELS3, in that order)."""

    def M(self, k: int, cx: float, cy: float, H: int) -> np.ndarray:
        k = int(min(max(k, self.rs + 1), self.nb - 2))
        return self._mean(k - 1, k + 1, cx, cy, H)

    def raw3(self, b: int, cx: float, cy: float, half: int, channels=CHANNELS3) -> np.ndarray:
        """(len(channels), 2h, 2h) unscaled inputs, local medians removed on the crop + MARGIN, margin cut off."""
        H = half + MARGIN
        fill = lambda v: np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0)
        Mb = fill(self.M(b, cx, cy, H))
        out = []
        for c in channels:
            if c == "A":
                v = Mb - medbg(Mb)
            elif c == "C":
                v = Mb - fill(self.M(self.rs + 1, cx, cy, H))
                v = v - medbg(v)
            else:
                v = Mb - fill(self.M(b - LAGS[c], cx, cy, H))
                v = v - medbg(v)
            out.append(v[MARGIN:-MARGIN, MARGIN:-MARGIN])
        return np.stack(out).astype(np.float32)

    def inputs3(self, b: int, cx: float, cy: float, half: int, channels=CHANNELS3) -> np.ndarray:
        return scale3(self.raw3(b, cx, cy, half, channels), channels)


def scale3(raw: np.ndarray, channels=CHANNELS3) -> np.ndarray:
    s = np.array([SCALE3[c] for c in channels], np.float32)
    return np.clip(raw / s[:, None, None], -CLIP, CLIP).astype(np.float32)


def full_frame_inputs(bins, meta, b: int, channels=CHANNELS3, reg=None) -> np.ndarray:
    """The version 3 inputs of bin b on the whole registered frame (as Movie3.raw3 on a crop covering the frame:
    registration by the cache's shifts, edge pixels replicated). ``reg``: a dict caching registered bins."""
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    H, W = bins.shape[1:]
    reg = {} if reg is None else reg

    def R(k):
        if k not in reg:
            dx, dy = shifts[k]
            reg[k] = cv2.warpAffine(np.asarray(bins[k], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return reg[k]

    def M(k):
        k = int(min(max(k, rs + 1), nb - 2))
        return np.mean([R(j) for j in (k - 1, k, k + 1)], axis=0)

    Mb = M(b)
    out = []
    for c in channels:
        if c == "A":
            v = Mb - medbg(Mb)
        elif c == "C":
            v = Mb - M(rs + 1)
            v = v - medbg(v)
        else:
            v = Mb - M(b - LAGS[c])
            v = v - medbg(v)
        out.append(v)
    return scale3(np.stack(out).astype(np.float32), channels)
