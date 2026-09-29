"""From P(arc s is covered by bin t) to a growth front, a length, an onset and tips.

"Covered" = grain body or tube up to its front. The front index f(t) (number of arc samples
covered, from the kymograph's first sample 8 px inside the grain) maximises the summed
log-likelihood sum_t [sum_{i < f(t)} log p + sum_{i >= f(t)} log(1 - p)], i.e. the cumulative
logit up to f(t), under 0 <= f(t) - f(t-1) <= vmax (a tube does not shrink and grows at most
vmax px per bin). The front's arc position is F(t) = s[f(t)] - 0.5 (between the last covered and
the first uncovered 1 px sample); the length is F(t) minus where the length starts (the start
``a``: the annotator's first click on a human route; estimated on other paths, see ``start``).
"""

from __future__ import annotations

import math

import numpy as np


def dp_front(evidence: np.ndarray, vmax: int) -> np.ndarray:
    """Non-decreasing front per bin maximising summed evidence behind it (as sparsetrack.analyze.dp_front)."""
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
    idx = np.arange(n + 1)
    for t in range(1, n_bins):
        pad = np.concatenate([np.full(vmax, -np.inf), score])
        windows = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + windows[idx, j]
        back[t] = idx - (vmax - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def fronts(logits: np.ndarray, s: np.ndarray, vmax: float = 4.0, clip: float = 8.0) -> np.ndarray:
    """Front arc position F(t) per bin."""
    ev = np.clip(logits.astype(np.float64), -clip, clip)
    f = dp_front(ev, max(1, int(round(vmax))))
    return float(s[0]) + f - 0.5


def edge_from_bb(bb: np.ndarray, s: np.ndarray, lo: float = -7.0, hi: float = 5.0, half_w: int = 3) -> float:
    """The grain's visible edge along the path: the steepest step of the before image (median over the
    central lateral offsets), 7 px inside to 5 px outside the path's first point (the idea of
    sparsetrack.analyze.exit_edge, along the path instead of the radius)."""
    W = bb.shape[1]
    prof = np.median(bb[:, W // 2 - half_w: W // 2 + half_w + 1].astype(np.float64), axis=1)
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    g = np.gradient(prof, s)
    win = (s > lo) & (s < hi)
    if not win.any():
        return 0.0
    return float(s[win][np.argmax(np.abs(g[win]))])


def start(F: np.ndarray, mode: str, bb: np.ndarray | None = None, s: np.ndarray | None = None,
          rest_bins: int = 3) -> float:
    """Where the length starts on a path whose first point is not a human click: ``census`` (the path's
    first point), ``rest`` (where the reader's front sits in the first bins, before any tube) or
    ``edge`` (the before image's steepest step, ``edge_from_bb``)."""
    if mode == "census":
        return 0.0
    if mode == "rest":
        return float(np.min(F[:rest_bins]))
    if mode == "edge":
        return edge_from_bb(bb, s)
    raise ValueError(mode)


def calls(L: np.ndarray, frames: list, onset_px: float = 2.0, min_len: float = 8.0) -> dict:
    """SparseTrack-style germination call from a length series (observed bins)."""
    L = np.asarray(L, float).copy()
    above = np.nonzero(L >= onset_px)[0]
    if not len(above) or L[-1] < min_len:
        return {"status": "no_emergence_by_end", "onset_frame": None, "onset_interval": None, "L": np.zeros_like(L),
                "onset_bin": None}
    b = int(above[0])
    L[:b] = 0.0
    if b == 0:
        return {"status": "emerged_at_start", "onset_frame": frames[0], "onset_interval": None, "L": L, "onset_bin": 0}
    return {"status": "emerged_within", "onset_frame": frames[b], "onset_interval": [frames[b - 1], frames[b]],
            "L": L, "onset_bin": b}


def tips(F: np.ndarray, pts: np.ndarray, s: np.ndarray, rot: np.ndarray | None, pivot) -> list:
    """Path point at the front's arc position F(t) per bin, turned like the path was read."""
    out = []
    S = len(pts)
    for t, f in enumerate(F):
        q = pts[int(np.clip(round(f - float(s[0])), 0, S - 1))].astype(np.float64)
        if rot is not None and abs(float(rot[t])) > 1e-9:
            a = math.radians(float(rot[t]))
            p0 = np.asarray(pivot, np.float64)
            d = q - p0
            q = p0 + np.array([math.cos(a) * d[0] - math.sin(a) * d[1], math.sin(a) * d[0] + math.cos(a) * d[1]])
        out.append([round(float(q[0]), 2), round(float(q[1]), 2)])
    return out
