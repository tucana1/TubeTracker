"""Final state first, then backward: carry a tube's route through the movie with composed DIS optical flow, read the
learned tube maps along it into a kymograph K[bin, material arc length], and read the tip with a monotone front.

Routes are carried as polylines (never images: dense warping of whole maps absorbs growth, 29 Sep). Flow: OpenCV DIS
(PRESET_MEDIUM) between 3-bin means of the registered frames, uint8 over the grain crop's 1-99 percentile window,
composed in ``STEP_BINS`` steps (runs/lab_checks_2026-09-29/short_interval/flowcheck.py); bins between two steps are
interpolated linearly. Everything is computed on a crop round the route (bounding box + ``MARGIN`` px); nothing
full-frame is written.

Coordinates: reference coordinates (labels' ``path_xy_ref``; the tube-map cache is in reference coordinates with zero
shifts). A crop of half-size ``half`` centred on integer reference (cx, cy) has pixel j covering reference
[cx - half + j, cx - half + j + 1), so crop coordinate u = x_ref - (cx - half) - 0.5 (pixel centres at integers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sparsetrack import stack  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402
from sparsetrack.routes import normals  # noqa: E402

OUT = REPO / "runs/research/carry_front"
PROB = "prob_tubes_bn_real_ld_m2"
STEP_BINS = 10      # composed flow steps (bins)
MARGIN = 60         # crop margin round the route (px)
EXTEND_PX = 40.0    # the route carried on straight past the final apex, so the front is not capped at the answer
BANDS = (1.0, 2.0, 3.0)  # half-widths of the normal band the maps are maxed over (px)

cv2.setNumThreads(2)


def dense(path, step: float = 1.0) -> np.ndarray:
    p = np.asarray(path, float).reshape(-1, 2)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if len(p) < 2 or s[-1] <= 0:
        return p[:1].copy()
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - ss[-1] > 0.25:
        ss = np.append(ss, s[-1])
    return np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], 1)


def extend(pts: np.ndarray, px: float, back: float = 6.0) -> np.ndarray:
    """The dense route carried on straight along its last ``back`` px direction by ``px`` (1 px steps)."""
    if len(pts) < 2 or px <= 0:
        return pts
    j = max(0, len(pts) - 1 - int(round(back)))
    d = pts[-1] - pts[j]
    n = float(np.hypot(*d))
    if n < 1e-6:
        d, n = pts[-1] - pts[0], float(np.hypot(*(pts[-1] - pts[0]))) or 1.0
    d = d / n
    k = np.arange(1, int(px) + 1, dtype=float)[:, None]
    return np.vstack([pts, pts[-1] + k * d])


def arclen(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


class Movie:
    def __init__(self, name: str):
        self.name = name
        bins, meta = stack.load(REPO / f"runs/sparsetrack/{name}")
        self.R = Renderer(bins, meta)
        self.meta = meta
        self.rs = int(meta.get("ref_start", 0))
        self.nb = int(meta["n_bins"])
        self.fpb = int(meta["frames_per_bin"])
        self.P = np.load(REPO / f"runs/sparsetrack/{name}/{PROB}/bins.npy", mmap_mode="r")
        self.labels = json.loads((REPO / f"benchmark/labels/{name}_v1.json").read_text())
        self.H, self.W = self.P.shape[1:]

    def scored_grains(self) -> list[str]:
        """Grains evaluate.score scores (not excluded, isolated) with at least one FULL trace."""
        out = []
        for gid, g in sorted(self.labels["grains"].items()):
            if g.get("excluded") or not g.get("isolated", True):
                continue
            tr = (self.labels["labels"].get(gid) or {}).get("traces") or {}
            if any(t["state"] == "full" for t in tr.values()):
                out.append(gid)
        return out

    def full_traces(self, gid: str) -> list[tuple[int, dict]]:
        tr = (self.labels["labels"].get(gid) or {}).get("traces") or {}
        return sorted(((int(b), t) for b, t in tr.items() if t["state"] == "full"), key=lambda x: x[0])


class Crop:
    """A grain's crop window, its 3-bin mean frames (lazily, for the flow) and its tube maps."""

    def __init__(self, mv: Movie, pts_ref: list[np.ndarray], grain: dict):
        allp = np.vstack(pts_ref + [np.array([[grain["x"], grain["y"]]])])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        self.cx, self.cy = int(round((lo[0] + hi[0]) / 2)), int(round((lo[1] + hi[1]) / 2))
        self.half = int(max(100, np.ceil(max(hi - lo) / 2 + MARGIN)))
        self.mv = mv
        self._mean: dict[int, np.ndarray] = {}
        self._flow: dict[tuple, np.ndarray] = {}
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        a = self.mean3(mv.rs + 1)
        b = self.mean3(mv.nb - 2)
        self.lo, self.hi = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1, 99])

    def to_crop(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float) - [self.cx - self.half + 0.5, self.cy - self.half + 0.5]

    def to_ref(self, uv: np.ndarray) -> np.ndarray:
        return np.asarray(uv, float) + [self.cx - self.half + 0.5, self.cy - self.half + 0.5]

    def mean3(self, b: int) -> np.ndarray:
        mv = self.mv
        b = min(max(int(b), mv.rs), mv.nb - 1)
        if b not in self._mean:
            bs = [x for x in (b - 1, b, b + 1) if mv.rs <= x < mv.nb]
            img = np.mean([mv.R.crop(x, self.cx, self.cy, self.half) for x in bs], axis=0)
            med = np.nanmedian(img) if np.isfinite(img).any() else 0.0
            self._mean[b] = np.nan_to_num(img, nan=med).astype(np.float32)
        return self._mean[b]

    def u8(self, img: np.ndarray) -> np.ndarray:
        return np.clip((img - self.lo) / max(self.hi - self.lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def flow(self, b0: int, b1: int) -> np.ndarray:
        """DIS flow u with I_b0(x) ~ I_b1(x + u(x)): a point at x in bin b0 lies at x + u(x) in bin b1 (cached per
        crop; ``clear`` frees them)."""
        if (b0, b1) not in self._flow:
            self._flow[(b0, b1)] = self._dis.calc(self.u8(self.mean3(b0)), self.u8(self.mean3(b1)), None)
        return self._flow[(b0, b1)]

    def clear(self):
        self._flow.clear()

    def pmap(self, b: int) -> np.ndarray:
        """The tube map (P in 0..1, float32) of bin b over the crop (pixel-aligned: integer centre, zero shifts)."""
        x0, y0 = self.cx - self.half, self.cy - self.half
        x1, y1 = x0 + 2 * self.half, y0 + 2 * self.half
        out = np.zeros((2 * self.half, 2 * self.half), np.float32)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(self.mv.W, x1), min(self.mv.H, y1)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(self.mv.P[b, sy0:sy1, sx0:sx1], np.float32) / 250.0
        return out


def sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear samples of a 2D (or HxWx2) image at crop coordinates uv (N, 2)."""
    x = np.asarray(uv[:, 0], np.float32)[None]
    y = np.asarray(uv[:, 1], np.float32)[None]
    if img.ndim == 3:
        return np.stack([cv2.remap(img[..., k], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
                         for k in range(img.shape[2])], 1)
    return cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]


def carry(crop: Crop, uv: np.ndarray, b_from: int, b_to: int, step: int = STEP_BINS) -> dict[int, np.ndarray]:
    """Points ``uv`` (crop coordinates, as they lie in bin b_from) carried to every bin from b_from to b_to inclusive,
    composing DIS flow over ``step``-bin steps and interpolating linearly between the steps."""
    out = {b_from: uv.copy()}
    if b_to == b_from:
        return out
    d = 1 if b_to > b_from else -1
    keys = [b_from] + list(range(b_from + d * step, b_to, d * step)) + [b_to]
    keys = [k for i, k in enumerate(keys) if i == 0 or k != keys[i - 1]]
    cur = uv.copy()
    for k0, k1 in zip(keys, keys[1:]):
        fl = crop.flow(k0, k1)
        nxt = cur + sample(fl, cur)
        for b in range(k0 + d, k1 + d, d):
            a = (b - k0) / (k1 - k0)
            out[b] = (1 - a) * cur + a * nxt
        cur = nxt
    return out


def _running_median(v: np.ndarray, k: int) -> np.ndarray:
    if len(v) < 3 or k < 3:
        return v
    pad = np.pad(v, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


def _gauss_along(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(v) < 3:
        return v
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, "valid")


def resample_at(curve: np.ndarray, s_target: np.ndarray) -> np.ndarray:
    """Points of ``curve`` at arc lengths ``s_target`` from its first point (carried on straight past its end)."""
    s = arclen(curve)
    if s[-1] <= 0:
        return np.repeat(curve[:1], len(s_target), axis=0)
    out = np.stack([np.interp(s_target, s, curve[:, 0]), np.interp(s_target, s, curve[:, 1])], 1)
    beyond = s_target > s[-1]
    if beyond.any():
        j = max(0, int(np.searchsorted(s, s[-1] - 5.0)) - 1)
        d = curve[-1] - curve[j]
        d = d / max(float(np.hypot(*d)), 1e-6)
        out[beyond] = curve[-1] + (s_target[beyond] - s[-1])[:, None] * d
    return out


def carry_inext(crop: Crop, uv: np.ndarray, b_from: int, b_to: int, step: int = STEP_BINS, base_px: float = 15.0,
                med_k: int = 9, sigma: float = 5.0, fb_px: float = 1.5, max_step_px: float = 12.0) -> dict[int, np.ndarray]:
    """As ``carry``, for a tube: each step moves the route by the flow's median over its first ``base_px`` px (the
    grain and the tube's base) plus the sideways (normal) part of the rest, robustly smoothed along the tube, and keeps
    the route's material spacing from its base (a tube does not stretch). The flow's component ALONG the tube is
    dropped: carried backward, it is the tube's growth played in reverse, which pulls the route beyond the tip back
    onto the tube (the dense-warp failure of 29 Sep). Only points passing a forward-backward flow check count; a step
    moves at most ``max_step_px``."""
    S = arclen(uv)
    out = {b_from: uv.copy()}
    if b_to == b_from:
        return out
    d = 1 if b_to > b_from else -1
    keys = [b_from] + list(range(b_from + d * step, b_to, d * step)) + [b_to]
    keys = [k for i, k in enumerate(keys) if i == 0 or k != keys[i - 1]]
    cur = uv.copy()
    base = S <= base_px
    near = S <= 50.0
    for k0, k1 in zip(keys, keys[1:]):
        u = sample(crop.flow(k0, k1), cur)
        # forward-backward check: the flow back from where a point lands must bring it home (DIS fails on a grain
        # whose look changes, on blur and on featureless patches; one failed step once threw a route 100 px)
        err = np.hypot(*(u + sample(crop.flow(k1, k0), cur + u)).T)
        ok = err <= fb_px
        for sel in (base & ok, near & ok, ok):
            if sel.sum() >= 3:
                T = np.median(u[sel], axis=0)
                break
        else:
            T = np.zeros(2)
        tn = float(np.hypot(*T))
        if tn > max_step_px:
            T = T * (max_step_px / tn)
        n = normals(cur)
        delta = np.einsum("ij,ij->i", u - T, n)
        if ok.sum() >= 2:
            idx = np.arange(len(delta))
            delta = np.interp(idx, idx[ok], delta[ok])
        else:
            delta = np.zeros_like(delta)
        delta = np.clip(_gauss_along(_running_median(delta, med_k), sigma), -max_step_px, max_step_px)
        moved = cur + T + delta[:, None] * n
        nxt = resample_at(moved, S)
        for b in range(k0 + d, k1 + d, d):
            a = (b - k0) / (k1 - k0)
            out[b] = (1 - a) * cur + a * nxt
        cur = nxt
    return out


def kymograph(crop: Crop, routes: dict[str, np.ndarray], bins: range, bands=BANDS, step: float = 0.5) -> dict:
    """K[variant][band] (n_bins, N): the tube map maxed over the normal band of half-width ``band`` round each route
    point, per bin. ``routes[variant]`` is (n_bins, N, 2) in crop coordinates (row i = bins[i])."""
    out = {v: {w: np.zeros(r.shape[:2], np.float32) for w in bands} for v, r in routes.items()}
    wmax = max(bands)
    offs = np.arange(-wmax, wmax + 1e-9, step)
    for i, b in enumerate(bins):
        pm = crop.pmap(b)
        for v, r in routes.items():
            pts = r[i]
            n = normals(pts)
            qx = (pts[:, 0][None, :] + offs[:, None] * n[:, 0][None, :]).astype(np.float32)
            qy = (pts[:, 1][None, :] + offs[:, None] * n[:, 1][None, :]).astype(np.float32)
            vals = cv2.remap(pm, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for w in bands:
                sel = np.abs(offs) <= w + 1e-9
                out[v][w][i] = vals[sel].max(axis=0)
    return out


def dp_bounded(evidence: np.ndarray, vmax: int, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> np.ndarray:
    """Non-decreasing front index per bin (0 <= f[t] - f[t-1] <= vmax) maximising the summed evidence behind it
    (the front at f claims points [0, f)); optional per-bin bounds lo[t] <= f[t] <= hi[t]. As sparsetrack.analyze.
    dp_front, with bounds."""
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    idx = np.arange(n + 1)
    if lo is not None or hi is not None:
        lo_ = np.zeros(n_bins, int) if lo is None else np.asarray(lo, int)
        hi_ = np.full(n_bins, n, int) if hi is None else np.asarray(hi, int)
        gain = np.where((idx[None] >= lo_[:, None]) & (idx[None] <= hi_[:, None]), gain, -1e18)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
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


def arrival_bins(K: np.ndarray, theta: float, persist: int = 10, frac: float = 0.7) -> np.ndarray:
    """Per route point, the first bin from which the map stays marked there (K >= theta in >= ``frac`` of the next
    ``persist`` bins); n_bins where it never does."""
    n_bins, n = K.shape
    on = (K >= theta).astype(np.float32)
    cs = np.concatenate([np.zeros((1, n), np.float32), np.cumsum(on, axis=0)], axis=0)
    hi = np.minimum(np.arange(n_bins) + persist, n_bins)
    share = (cs[hi] - cs[np.arange(n_bins)]) / (hi - np.arange(n_bins))[:, None]
    ok = share >= frac
    return np.where(ok.any(axis=0), np.argmax(ok, axis=0), n_bins)


def dp_arrival(evidence: np.ndarray, vmax: int, arrive: np.ndarray, mu: float, delta: int, exempt: int = 0,
               lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> np.ndarray:
    """dp_bounded plus ownership by arrival: a front passing point s at bin b pays mu per bin by which the map had
    marked s more than ``delta`` bins before (material that was there before the tip came is not this tube's growth:
    a crossing or neighbouring tube on the route's future part, the flood's rule in kymograph form). Points before
    ``exempt`` (the rim) pay nothing."""
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    idx = np.arange(n + 1)
    if lo is not None or hi is not None:
        lo_ = np.zeros(n_bins, int) if lo is None else np.asarray(lo, int)
        hi_ = np.full(n_bins, n, int) if hi is None else np.asarray(hi, int)
        gain = np.where((idx[None] >= lo_[:, None]) & (idx[None] <= hi_[:, None]), gain, -1e18)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
    arr = np.asarray(arrive, float)
    for t in range(1, n_bins):
        phi = -mu * np.maximum(0.0, t - arr - delta)
        phi[:exempt] = 0.0
        C = np.concatenate([[0.0], np.cumsum(phi)])
        pad = np.concatenate([np.full(vmax, -np.inf), score - C])
        windows = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + C + windows[idx, j]
        back[t] = idx - (vmax - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def _waived(old: np.ndarray, fresh: np.ndarray, gap: int) -> np.ndarray:
    """Per point of a route: old there now, and fresh material within ``gap`` points beyond the end of its old run."""
    n = len(old)
    idx = np.arange(n)
    nxt = np.minimum.accumulate(np.where(~old, idx, n)[::-1])[::-1]  # first point at or after s that is not old
    cf = np.concatenate([[0], np.cumsum(fresh)])
    lo, hi = np.clip(nxt, 0, n), np.clip(nxt + gap, 0, n)
    return old & ((cf[hi] - cf[lo]) > 0)


def _long_voids(on: np.ndarray, hole: int) -> np.ndarray:
    """Points in unmarked runs longer than ``hole`` points."""
    n = len(on)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    d = np.diff(np.concatenate([[1], on.astype(np.int8), [1]]))
    for a, b in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
        if b - a > hole:
            out[a:b] = True
    return out


def dp_recency(evidence: np.ndarray, vmax: int, arrive: np.ndarray, mu: float, delta: int, wait: int, gap: int = 4,
               exempt: int = 0, lo: np.ndarray | None = None, hi: np.ndarray | None = None, void: float = 0.0,
               marked: np.ndarray | None = None, hole: int = 3) -> np.ndarray:
    """dp_bounded with a SOFT recency term: a front advancing at bin t through route point s whose map mark had
    arrived more than ``delta`` bins earlier (old material: a crossing or neighbouring tube on the route's future
    part) pays mu x (bins left) x min(1, (t - arrival - delta) / delta), i.e. up to mu of what claiming it for the
    rest of the movie could earn per unit of evidence; waived when fresh material (arrived no earlier than t - delta
    and no later than t + ``wait``) lies within ``gap`` points beyond the end of that old run: a tube growing across
    an older one keeps laying down new material beyond it. Points before ``exempt`` (the rim) pay nothing.
    The front can dodge that by passing old material early, racing ahead through unmarked route to meet it as it
    arrives; ``void`` > 0 also charges void x (bins left) for advancing through a point that lies, at that bin, in an
    unmarked run longer than ``hole`` points (``marked``: K >= theta, n_bins x n)."""
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    idx = np.arange(n + 1)
    if lo is not None or hi is not None:
        lo_ = np.zeros(n_bins, int) if lo is None else np.asarray(lo, int)
        hi_ = np.full(n_bins, n, int) if hi is None else np.asarray(hi, int)
        gain = np.where((idx[None] >= lo_[:, None]) & (idx[None] <= hi_[:, None]), gain, -1e18)
    back = np.zeros((n_bins, n + 1), np.int32)
    arr = np.asarray(arrive, float)
    seen = arr < n_bins

    def passing_cost(t):  # cumulative cost of advancing the front through points [0, f) at bin t
        lag = t - arr - delta
        old = seen & (lag > 0)
        fresh = seen & (arr >= t - delta) & (arr <= t + wait)
        w = np.minimum(1.0, np.maximum(lag, 0.0) / max(delta, 1))
        phi = np.where(old & ~_waived(old, fresh, gap), -mu * (n_bins - t) * w, 0.0)
        if void > 0 and marked is not None:
            phi = phi - np.where(_long_voids(marked[t], hole), void * (n_bins - t), 0.0)
        phi[:exempt] = 0.0
        return np.concatenate([[0.0], np.cumsum(phi)])

    score = gain[0] + passing_cost(0)  # the first bin's front passes its points then too
    for t in range(1, n_bins):
        C = passing_cost(t)
        pad = np.concatenate([np.full(vmax, -np.inf), score - C])
        windows = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + C + windows[idx, j]
        back[t] = idx - (vmax - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front
