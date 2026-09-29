"""Kymographs along a tube path, in the grain-following view frame.

A path is a polyline in reference ("view") coordinates from the tube's exit on the grain rim
outward (SparseTrack's ``path`` or a human trace's ``path_xy_view``). It is resampled at 1 px
arc steps, extended ``extend`` px straight past its end and ``back`` px straight back into the
grain (arc s < 0; the annotator starts a trace at the grain's visible edge, which lies up to a few
px inside the census circle, so the reader must see the edge region). For every observed bin ``b`` the
registered bin image is sampled at ``point + w * normal`` for lateral offsets ``w`` (the view
frame follows the grain's own drift, ``follow``, exactly as the labelling tool and SparseTrack
do; ``Renderer.crop`` adds the field registration). Optionally the path is also turned per bin
about its exit by SparseTrack's rotation track (``sparsetrack.report.turned_path``).

Stored per kymograph (grey levels, float16):

- ``dyn``  (T, S, W): I_t - B sampled on the path as it is at bin t (turned, if a track is given)
- ``dyn0`` (T, S, W): the same on the unturned path (only kept when a track is given)
- ``eb``   (S, W): E - B on the unturned path (the end state; E = mean of the last full bins)
- ``bb``   (S, W): B minus its median over the samples (grain rim and field structure)
- ``rim``  (S,): distance of each path point from the grain centre minus its radius
- ``other`` (S, W): distance to the nearest other grain's rim (clipped at 30 px)
- ``valid`` (T, S, W) bool as uint8: the sample's source lies inside the movie frame

B is the mean of the first three observed bins (SparseTrack's "before" image), sampled at the
same positions as the bin being read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

LATERAL = np.arange(-8.0, 8.0 + 1e-9, 1.0)   # 17 lateral offsets, px
EXTEND = 20.0                                   # straight extension past the path end, px
BACK = 8.0                                      # straight extension back into the grain, px
STEP = 1.0                                      # arc step, px
FOLLOW_HALF = 60                                # as sparsetrack.bench.server
FOLLOW_MAX_STEP = 10.0


@dataclass
class Path1:
    pts: np.ndarray       # (S, 2) reference/view coordinates
    s: np.ndarray         # (S,) arc position from the path's first point, px (negative: back into the grain)
    normal: np.ndarray    # (S, 2) unit normals
    n_path: int           # points on the original polyline
    length: float         # polyline length of the original path
    i0: int = 0           # index of arc 0 (the path's first point)


def resample(path_xy, step: float = STEP, extend: float = EXTEND, smooth_px: float = 2.0,
             back: float = 0.0) -> Path1 | None:
    """Uniform arc resampling of a polyline, straight extensions past its end and ``back`` px before
    its start, and smoothed normals."""
    p = np.asarray(path_xy, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return None
    keep = np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-6]
    p = p[keep]
    if len(p) < 2:
        return None
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    total = float(cum[-1])
    s = np.arange(0.0, total + 1e-9, step)
    pts = np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], 1)
    # extension direction: the last ~6 px of the polyline
    tail = p[-1] - np.array([np.interp(max(total - 6.0, 0.0), cum, p[:, 0]), np.interp(max(total - 6.0, 0.0), cum, p[:, 1])])
    if np.linalg.norm(tail) < 1e-6:
        tail = p[-1] - p[0]
    tail = tail / (np.linalg.norm(tail) + 1e-12)
    n_ext = int(round(extend / step))
    ext = pts[-1][None] + tail[None] * (np.arange(1, n_ext + 1) * step)[:, None]
    head = np.array([np.interp(min(6.0, total), cum, p[:, 0]), np.interp(min(6.0, total), cum, p[:, 1])]) - p[0]
    head = head / (np.linalg.norm(head) + 1e-12)
    n_back = int(round(back / step))
    pre = p[0][None] - head[None] * (np.arange(n_back, 0, -1) * step)[:, None]
    allp = np.vstack([pre, pts, ext])
    ss = np.r_[-np.arange(n_back, 0, -1) * step, s, s[-1] + np.arange(1, n_ext + 1) * step]
    # tangents of a smoothed copy (clicked polylines have corners)
    k = max(1, int(round(smooth_px / step)))
    padded = np.vstack([np.repeat(allp[:1], k, 0), allp, np.repeat(allp[-1:], k, 0)])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(padded[:, i], ker, "valid") for i in range(2)], 1)
    tang = np.gradient(sm, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    return Path1(allp, ss, normal, len(pts), total, n_back)


def follow_offsets(renderer, g: dict) -> np.ndarray:
    """(n_bins, 2) grain-following offsets: the same computation as ``Bench.follow`` (which needs a
    labels file); zero-drift fallback when the track is implausible."""
    from sparsetrack.analyze import local_shifts, plausible_drift
    rs, half, n_bins = renderer.ref_start, FOLLOW_HALF, renderer.n_bins
    crops = np.stack([renderer.crop(b, g["x"], g["y"], half) for b in range(rs, n_bins)])
    if np.isnan(crops).any():
        crops = np.nan_to_num(crops, nan=float(np.nanmedian(crops)))
    ls = local_shifts(crops, half - 0.5, g["r"], 12.0, 3)
    if not plausible_drift(ls, FOLLOW_MAX_STEP):
        ls = np.zeros_like(ls)
    return np.vstack([np.repeat(ls[:1], rs, axis=0), ls])


def turn(q: np.ndarray, pivot: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-9:
        return q
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (q - pivot) @ r.T + pivot


def _remap(img: np.ndarray, q: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Sample ``img`` (a crop whose pixel j is centred on reference x0 + j + 0.5) at reference points q."""
    mx = (q[..., 0] - x0 - 0.5).astype(np.float32)
    my = (q[..., 1] - y0 - 0.5).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def extract(renderer, g: dict, path: Path1, follow: np.ndarray, others: list[dict] | None = None,
            rot_deg: np.ndarray | None = None, pivot=None, lateral: np.ndarray = LATERAL,
            ref_bins: int = 3, late_bins: int = 3) -> dict:
    """Kymograph of one grain along ``path`` for every observed bin (``renderer.ref_start`` on).

    ``rot_deg`` (T,) turns the path about ``pivot`` (default its first point) per observed bin.
    """
    rs, nb = renderer.ref_start, renderer.n_bins
    T = nb - rs
    S, W = len(path.pts), len(lateral)
    base = path.pts[:, None, :] + lateral[None, :, None] * path.normal[:, None, :]   # (S, W, 2)
    pivot = np.asarray(path.pts[path.i0] if pivot is None else pivot, np.float64)
    turned = rot_deg is not None and np.any(np.abs(np.asarray(rot_deg)) > 1e-6)
    qs = [turn(base, pivot, float(rot_deg[t])) for t in range(T)] if turned else None
    # the crop must hold the unturned path and every bin's turned path
    allq = np.concatenate([base.reshape(-1, 2)] + ([q.reshape(-1, 2) for q in qs] if turned else []))
    lo, hi = allq.min(axis=0) - 3.0, allq.max(axis=0) + 3.0
    cx, cy = (lo + hi) / 2.0
    half = int(math.ceil(max(hi - lo) / 2.0)) + 2
    x0, y0 = cx - half, cy - half

    def crop(b):
        c = renderer.crop(b, cx, cy, half, follow)
        return np.nan_to_num(c, nan=float(np.nanmedian(c)) if np.isfinite(c).any() else 0.0).astype(np.float32)

    early = np.mean([crop(b) for b in range(rs, rs + ref_bins)], axis=0)
    late_idx = list(range(nb - 1 - late_bins, nb - 1))
    late = np.mean([crop(b) for b in late_idx], axis=0)
    B0 = _remap(early, base, x0, y0)
    E0 = _remap(late, base, x0, y0)
    dyn = np.empty((T, S, W), np.float16)
    dyn0 = np.empty((T, S, W), np.float16) if turned else None
    valid = np.empty((T, S, W), np.uint8)
    h, w = renderer.height, renderer.width
    for t in range(T):
        b = rs + t
        img = crop(b)
        q = qs[t] if turned else base
        dyn[t] = (_remap(img, q, x0, y0) - (_remap(early, q, x0, y0) if turned else B0)).astype(np.float16)
        if turned:
            dyn0[t] = (_remap(img, base, x0, y0) - B0).astype(np.float16)
        # source position of each sample in bin b (reference + follow + field shift) inside the frame?
        src = q + follow[b][None, None, :] + renderer.shifts[b][None, None, :]
        valid[t] = ((src[..., 0] >= 0.5) & (src[..., 0] <= w - 0.5) & (src[..., 1] >= 0.5) & (src[..., 1] <= h - 0.5))
    rim = np.hypot(path.pts[:, 0] - g["x"], path.pts[:, 1] - g["y"]) - g["r"]
    other = np.full((S, W), 30.0, np.float32)
    for o in others or []:
        if o.get("id") == g.get("id"):
            continue
        d = np.hypot(base[..., 0] - o["x"], base[..., 1] - o["y"]) - o["r"]
        other = np.minimum(other, np.maximum(d, -5.0))
    out = {"dyn": dyn, "eb": (E0 - B0).astype(np.float16), "bb": (B0 - np.median(B0)).astype(np.float16),
           "rim": rim.astype(np.float32), "other": other.astype(np.float16), "valid": valid,
           "s": path.s.astype(np.float32), "n_path": path.n_path, "path_len": path.length, "i0": path.i0,
           "pts": path.pts.astype(np.float32), "normal": path.normal.astype(np.float32),
           "rs": rs, "turned": bool(turned)}
    if turned:
        out["dyn0"] = dyn0
        out["rot_deg"] = np.asarray(rot_deg, np.float32)
        out["pivot"] = pivot.astype(np.float32)
    return out


def noise_sigma(dyn: np.ndarray) -> float:
    """Per-bin noise of a kymograph: robust sd of consecutive-bin differences / sqrt(2)."""
    d = np.diff(dyn[:, :, dyn.shape[2] // 2 - 2: dyn.shape[2] // 2 + 3].astype(np.float32), axis=0).ravel()
    d = d[np.isfinite(d)]
    if not d.size:
        return 1.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))
