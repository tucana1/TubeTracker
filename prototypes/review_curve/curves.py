"""Reviewed growth curves: the model's lengths corrected by a person's checked lengths (``anchors``), candidates for
``sparsetrack.review.reviewed_curve``.

Every candidate starts from the model's curve made monotone (its running maximum), zero before the review's onset
``fv``, and ends through the anchors (their running maximum: a curve that never shrinks cannot pass below an earlier
anchor), never shrinking, inside the band the anchors allow (before an anchor at most its length, after it at least).

- ``stretch`` (the current app, W = None): between two anchors the model's growth rescaled to meet them; after the
  last anchor the model's growth added. With a window W only the last W bins before each anchor are rescaled.
- ``add``: the model's own lengths plus a correction: at each anchor the model's error there, d = L - m; it ramps in
  linearly over the W bins before the anchor and, after it, is carried to the next anchor (``after="carry"``),
  fades over W bins (``"decay"``) or is dropped (``"none"``). ``stall``: where the model is short at an anchor and
  had stopped growing before it (grown <= ``delta`` px since bin s), the ramp starts at s: the tube went on growing
  while the model lost its tip.
- ``onset="shift"``: when the review's onset differs from the model's (the first bin it read a tube), the model's
  curve is moved in time to start at the review's onset (not rescaled).
"""
from __future__ import annotations

import numpy as np


def model_onset(m: np.ndarray, eps: float = 0.0) -> int | None:
    nz = np.flatnonzero(m > eps)
    return int(nz[0]) if len(nz) else None


def stall_start(m: np.ndarray, b: int, lo: int, delta: float) -> int:
    """The first bin s >= lo from which the curve grew at most ``delta`` px up to bin b."""
    s = b
    while s > lo and m[b] - m[s - 1] <= delta:
        s -= 1
    return s


def base_curve(model_px, fv: int, onset: str = "zero") -> np.ndarray:
    m = np.maximum.accumulate(np.nan_to_num(np.asarray(model_px, float)))
    n = len(m)
    if onset == "shift":
        fm = model_onset(m)
        if fm is not None and fm != fv:
            idx = np.clip(np.arange(n) + (fm - fv), 0, n - 1)
            m = m[idx]
    out = m.copy()
    if onset == "down" and fv > 0 and m[fv - 1] > 0:  # the review's onset is later: what the model read by then is not
        out = np.maximum(m - m[fv - 1], 0.0)          # this tube (its curve moved down to start from zero there)
    out[:fv] = 0.0
    return out


def clean_anchors(anchors, fv: int, n: int) -> list[tuple[int, float]]:
    pts = sorted((int(b), float(L)) for b, L in anchors if fv <= int(b) < n)
    out, run = [], 0.0
    for b, L in pts:
        run = max(run, L)
        if out and out[-1][0] == b:
            out[-1] = (b, run)
        else:
            out.append((b, run))
    return out


def band(y: np.ndarray, pts: list, fv: int) -> np.ndarray:
    """Clip into the band the anchors allow, through them, zero before the onset, never shrinking."""
    y = y.copy()
    prev_b, prev_L = fv - 1, 0.0
    for b, L in pts:
        seg = slice(max(prev_b + 1, 0), b)
        y[seg] = np.clip(y[seg], prev_L, L)
        y[b] = L
        prev_b, prev_L = b, L
    y[prev_b + 1:] = np.maximum(y[prev_b + 1:], prev_L)
    y[:fv] = 0.0
    return np.maximum.accumulate(np.maximum(y, 0.0))


def stall_applies(rule, dk: float, L: float, mb: float) -> bool:
    """Whether the model's shortfall dk at an anchor of length L (model mb) is treated as a stall: ``True``/"any"
    any shortfall; "tol" only beyond the scoring tolerance max(2 px, 10%); "zero" only where the model read no tube
    by then."""
    if not rule or dk <= 0:
        return False
    if rule == "tol":
        return dk > max(2.0, 0.1 * L)
    if rule == "zero":
        return mb <= 0.0
    return True


def curve(model_px, fv: int | None, anchors, family: str = "add", W: float | None = 20.0, after: str = "carry",
          stall=False, delta: float = 2.0, onset: str = "zero", Wa: float | None = None) -> np.ndarray:
    n = len(model_px)
    if fv is None or fv >= n:
        return np.zeros(n)
    fv = max(int(fv), 0)
    m = base_curve(model_px, fv, onset)
    pts = clean_anchors(anchors, fv, n)
    if not pts:
        return np.maximum.accumulate(m)
    if family == "stretch":
        return band(stretch(m, fv, pts, W), pts, fv)
    d = [L - m[b] for b, L in pts]
    starts = []  # where each anchor's correction starts ramping in
    prev = fv - 1
    for (b, L), dk in zip(pts, d):
        w = (b - prev) if W is None else min(W, b - prev)
        s = b - w
        if stall_applies(stall, dk, L, m[b]):
            s = min(s, stall_start(m, b, max(prev, 0), delta))
        starts.append(max(s, prev))
        prev = b
    Wa = W if Wa is None else Wa
    c = np.zeros(n)

    def after_value(k: int, bb: np.ndarray) -> np.ndarray:
        if after == "carry":
            return np.full(len(bb), d[k])
        if after == "decay" and Wa:
            return d[k] * np.clip(1.0 - (bb - pts[k][0]) / Wa, 0.0, 1.0)
        return np.zeros(len(bb))

    prev_b = fv - 1
    for k, (b, L) in enumerate(pts):
        seg = np.arange(max(prev_b + 1, 0), b + 1)
        base = after_value(k - 1, seg.astype(float)) if k > 0 else np.zeros(len(seg))
        s = starts[k]
        r = np.clip((seg - s) / max(b - s, 1e-9), 0.0, 1.0) if b > s else (seg >= b).astype(float)
        c[seg] = (1.0 - r) * base + r * d[k]
        prev_b = b
    tail = np.arange(prev_b + 1, n)
    if len(tail):
        c[tail] = after_value(len(pts) - 1, tail.astype(float))
    return band(m + c, pts, fv)


def rescaled_curve(model_px, fv: int | None, anchors: list[tuple[int, float]]) -> np.ndarray:
    """The app's reviewed curve until 3 Oct 2026 (``sparsetrack.review.reviewed_curve`` then, verbatim; M1 here and
    in prototypes/review_fill): zero before the first visible bin ``fv``, through the checked lengths, between them
    the model's growth rescaled to reach each; after the last the model's growth added; never shrinks."""
    m = np.maximum.accumulate(np.nan_to_num(np.asarray(model_px, float)))
    n = len(m)
    out = np.zeros(n)
    if fv is None or fv >= n:
        return out
    pts = sorted((b, float(L)) for b, L in anchors if fv <= b < n)
    if fv > 0:
        pts = [(fv - 1, 0.0)] + pts
    elif pts:  # there from the start: the model's shape up to the first checked length
        b1, L1 = pts[0]
        pts = [(0, m[0] * L1 / m[b1] if m[b1] > 0 else L1)] + pts
    else:
        pts = [(0, m[0])]
    for (a, La), (b, Lb) in zip(pts, pts[1:]):
        seg = np.arange(a, b + 1)
        dm = m[b] - m[a]
        frac = (m[seg] - m[a]) / dm if dm > 1e-6 else (seg - a) / max(b - a, 1)
        out[seg] = La + frac * (Lb - La)
    b_last, L_last = pts[-1]
    out[b_last:] = L_last + (m[b_last:] - m[b_last])
    out[:fv] = 0.0
    return np.maximum.accumulate(np.maximum(out, 0.0))


def stretch(m: np.ndarray, fv: int, pts: list, W: float | None) -> np.ndarray:
    """The model's growth rescaled between anchors (the current app with W None), only over the W bins before each
    anchor with a window; after the last anchor the model's growth added."""
    n = len(m)
    y = m.copy()
    prev_b, prev_L = fv - 1, 0.0
    for b, L in pts:
        a = prev_b if W is None else max(prev_b, int(b - W))
        La = prev_L if a == prev_b else min(max(m[a], prev_L), L)
        seg = np.arange(max(a, 0), b + 1)
        dm = m[b] - m[max(a, 0)]
        frac = (m[seg] - m[max(a, 0)]) / dm if dm > 1e-6 else (seg - a) / max(b - a, 1)
        y[seg] = La + frac * (L - La)
        prev_b, prev_L = b, L
    y[prev_b:] = prev_L + (m[prev_b:] - m[prev_b])
    return y
