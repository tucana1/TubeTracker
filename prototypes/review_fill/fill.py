"""Fill a grain's whole growth curve from the image, anchored on the tube a person traced at one or two times.

Each trace (exit to apex, reference coordinates, at its bin) becomes a route: the trace itself, carried on along the
model's route at that bin where the apex lies on it, then straight on ``EXTEND_PX``. The route is carried through
the movie with composed DIS optical flow as a tube (the flow's median over its base plus its sideways part along the
tube, material spacing kept: ``carry_tube``, after prototypes/carry_front/carry.py's ``carry_inext``, 3 Oct 2026),
the learned tube map is read along it into a kymograph K[bin, arc length] (max over a band across the route), and
the tip is read as the globally best monotone front (dynamic programming) that passes through every traced length,
with a speed cap set from the traced lengths themselves.

Between two traces the later trace's route is used beyond the earlier trace's length, joined onto the earlier
trace's carried apex near the earlier bin and left where the later trace put it near the later bin (``join``).

Everything works on a crop round the routes; flows between grid bins (every ``STEP_BINS``) are cached per crop,
so several traces of one grain share them. Coordinates: reference coordinates (the tube maps are in reference
coordinates); a crop of half-size ``half`` centred on integer (cx, cy) has pixel j covering [cx - half + j,
cx - half + j + 1), so crop coordinate u = x_ref - (cx - half) - 0.5.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from sparsetrack.routes import normals

STEP_BINS = 10      # composed flow steps (bins), aligned to a grid so traces of one grain share the flows
MARGIN = 60         # crop margin round the routes (px)
MAX_HALF = 320      # largest crop half-size (memory)
EXTEND_PX = 40.0    # a route is carried on straight past its end, so the front is not capped at it
JOIN_TOL_PX = 4.0   # a trace's apex this close to the model's route continues along it
P_SCALE = 250.0     # tube maps are uint8 P x P_SCALE (sparsetrack.learned)


# ----------------------------------------------------------------------------- polylines
def arclen(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) if len(p) else np.zeros(0)


def dense(path, step: float = 1.0) -> np.ndarray:
    """The polyline every ``step`` px of arc (its end kept)."""
    p = np.asarray(path, float).reshape(-1, 2)
    s = arclen(p)
    if len(p) < 2 or s[-1] <= 0:
        return p[:1].copy()
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - ss[-1] > 0.25:
        ss = np.append(ss, s[-1])
    return np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], 1)


def point_at(pts: np.ndarray, s_target) -> np.ndarray:
    """Points at arc lengths ``s_target`` (carried on straight along the last 5 px past the end)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    st = np.atleast_1d(np.asarray(s_target, float))
    s = arclen(p)
    if len(p) < 2 or s[-1] <= 0:
        return np.repeat(p[:1], len(st), axis=0)
    out = np.stack([np.interp(st, s, p[:, 0]), np.interp(st, s, p[:, 1])], 1)
    beyond = st > s[-1]
    if beyond.any():
        j = max(0, int(np.searchsorted(s, s[-1] - 5.0)) - 1)
        d = p[-1] - p[j]
        d = d / max(float(np.hypot(*d)), 1e-6)
        out[beyond] = p[-1] + (st[beyond] - s[-1])[:, None] * d
    return out


def extend(pts: np.ndarray, px: float, back: float = 6.0) -> np.ndarray:
    """The 1 px route carried on straight along its last ``back`` px direction by ``px`` (1 px steps)."""
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


def project(route: np.ndarray, p) -> tuple[float, float]:
    """(arc length along ``route`` of the point nearest ``p``, the distance to it)."""
    r = np.asarray(route, float).reshape(-1, 2)
    if len(r) < 2:
        return 0.0, float(np.hypot(*(r[0] - p))) if len(r) else math.inf
    a, b = r[:-1], r[1:]
    d = b - a
    seg2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((np.asarray(p, float) - a) * d).sum(1) / seg2, 0.0, 1.0)
    q = a + t[:, None] * d
    dist = np.hypot(*(q - p).T)
    k = int(np.argmin(dist))
    return float(arclen(r)[k] + t[k] * math.sqrt(seg2[k])), float(dist[k])


def anchor_route(trace_ref, model_ref=None, extend_px: float = EXTEND_PX, join_tol: float = JOIN_TOL_PX) -> tuple:
    """A traced tube as a route for the fill: the trace (1 px steps, exit first), carried on along the model's route
    at that bin (``model_ref``, reference coordinates) when the trace's apex lies within ``join_tol`` of it, then
    straight on by ``extend_px``. Returns (route (N, 2), traced length, whether the model's route was joined)."""
    Y = dense(trace_ref, 1.0)
    L = float(arclen(np.asarray(trace_ref, float))[-1])
    joined = False
    if model_ref is not None and len(model_ref) >= 2 and len(Y) >= 2:
        M = dense(model_ref, 1.0)
        sM = arclen(M)
        sp, d = project(M, Y[-1])
        if d <= join_tol and sp < sM[-1] - 2.0:
            tail = M[sM > sp + 1.0] + (Y[-1] - point_at(M, sp)[0])
            Y = dense(np.vstack([Y, tail]), 1.0)
            joined = True
    return extend(Y, extend_px), L, joined


# ----------------------------------------------------------------------------- images and flow
def sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear samples of a 2D (or HxWx2) image at crop coordinates uv (N, 2)."""
    x = np.asarray(uv[:, 0], np.float32)[None]
    y = np.asarray(uv[:, 1], np.float32)[None]
    if img.ndim == 3:
        return np.stack([cv2.remap(np.ascontiguousarray(img[..., k], np.float32), x, y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)[0] for k in range(img.shape[2])], 1)
    return cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]


class GrainFlow:
    """A grain's crop window: its 3-bin mean frames and DIS flows (cached, so several routes share them) and its tube
    maps. ``renderer``: sparsetrack.render.Renderer on the movie's cache; ``prob``: the tube-probability bins (uint8,
    reference coordinates, memory-mapped is fine)."""

    def __init__(self, renderer, prob, pts_ref: list, centre, margin: int = MARGIN, max_half: int = MAX_HALF):
        allp = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in pts_ref] + [np.asarray(centre, float)[None]])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        self.cx, self.cy = int(round((lo[0] + hi[0]) / 2)), int(round((lo[1] + hi[1]) / 2))
        self.half = int(min(max_half, max(100, math.ceil(max(hi - lo) / 2 + margin))))
        self.R = renderer
        self.P = prob
        self.rs = int(renderer.ref_start)
        self.nb = int(renderer.n_bins)
        self.H, self.W = prob.shape[1:]
        self._mean: dict[int, np.ndarray] = {}
        self._flow: dict[tuple, np.ndarray] = {}
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        a, b = self.mean3(self.rs + 1), self.mean3(self.nb - 2)
        self.lo, self.hi = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1, 99])

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.cx - self.half + 0.5, self.cy - self.half + 0.5])

    def to_crop(self, xy) -> np.ndarray:
        return np.asarray(xy, float) - self.origin

    def to_ref(self, uv) -> np.ndarray:
        return np.asarray(uv, float) + self.origin

    def mean3(self, b: int) -> np.ndarray:
        b = min(max(int(b), self.rs), self.nb - 1)
        if b not in self._mean:
            bs = [x for x in (b - 1, b, b + 1) if self.rs <= x < self.nb]
            img = np.mean([self.R.crop(x, self.cx, self.cy, self.half) for x in bs], axis=0)
            med = np.nanmedian(img) if np.isfinite(img).any() else 0.0
            self._mean[b] = np.nan_to_num(img, nan=med).astype(np.float32)
        return self._mean[b]

    def u8(self, img: np.ndarray) -> np.ndarray:
        return np.clip((img - self.lo) / max(self.hi - self.lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def flow(self, b0: int, b1: int) -> np.ndarray:
        """DIS flow u with I_b0(x) ~ I_b1(x + u(x)): a point at x in bin b0 lies at x + u(x) in bin b1."""
        if (b0, b1) not in self._flow:
            f = self._dis.calc(self.u8(self.mean3(b0)), self.u8(self.mean3(b1)), None)
            self._flow[(b0, b1)] = f.astype(np.float16)
        return self._flow[(b0, b1)]

    def clear(self) -> None:
        self._flow.clear()
        self._mean.clear()

    def pmap(self, b: int) -> np.ndarray:
        """The tube map (P in 0..1, float32) of bin b over the crop."""
        x0, y0 = self.cx - self.half, self.cy - self.half
        x1, y1 = x0 + 2 * self.half, y0 + 2 * self.half
        out = np.zeros((2 * self.half, 2 * self.half), np.float32)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(self.P[b, sy0:sy1, sx0:sx1], np.float32) / P_SCALE
        return out


# ----------------------------------------------------------------------------- carrying a tube
def grid_keys(b_from: int, b_to: int, step: int = STEP_BINS) -> list[int]:
    """The flow steps from ``b_from`` to ``b_to``: the start, every grid bin (multiple of ``step``) strictly
    between, the end."""
    if b_to == b_from:
        return [b_from]
    if b_to > b_from:
        mid = list(range((b_from // step + 1) * step, b_to, step))
    else:
        mid = list(range(((b_from - 1) // step) * step, b_to, -step))
    return [b_from] + mid + [b_to]


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


def carry_tube(gf: GrainFlow, uv: np.ndarray, b_from: int, b_to: int, step: int = STEP_BINS, base_px: float = 15.0,
               med_k: int = 9, sigma: float = 5.0, fb_px: float = 1.5, max_step_px: float = 12.0) -> dict:
    """Route ``uv`` (crop coordinates, as it lies at ``b_from``) carried to every bin up to ``b_to``: each flow step
    moves it by the flow's median over its first ``base_px`` px (grain and tube base) plus the sideways (normal) part
    of the rest, robustly smoothed along the tube, keeping its material spacing; only points passing a
    forward-backward check count; a step moves at most ``max_step_px``. Linear between steps. The flow along the
    tube is dropped: carried backward it is growth played in reverse (carry_front, 3 Oct 2026)."""
    S = arclen(uv)
    out = {b_from: uv.copy()}
    keys = grid_keys(b_from, b_to, step)
    if len(keys) < 2:
        return out
    d = 1 if b_to > b_from else -1
    cur = uv.copy()
    base, near = S <= base_px, S <= 50.0
    for k0, k1 in zip(keys, keys[1:]):
        u = sample(gf.flow(k0, k1), cur)
        err = np.hypot(*(u + sample(gf.flow(k1, k0), cur + u)).T)
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
        nxt = point_at(cur + T + delta[:, None] * n, S)
        for b in range(k0 + d, k1 + d, d):
            a = (b - k0) / (k1 - k0)
            out[b] = (1 - a) * cur + a * nxt
        cur = nxt
    return out


def carry_all(gf: GrainFlow, route_ref: np.ndarray, b_at: int, **kw) -> np.ndarray:
    """A route (reference coordinates, as it lies at bin ``b_at``) at every bin from the reference start: (n, N, 2)
    reference coordinates, row i = bin rs + i."""
    b_at = min(max(int(b_at), gf.rs), gf.nb - 1)
    uv = gf.to_crop(route_ref)
    car = carry_tube(gf, uv, b_at, gf.rs, **kw)
    car.update(carry_tube(gf, uv, b_at, gf.nb - 1, **kw))
    return gf.to_ref(np.stack([car[b] for b in range(gf.rs, gf.nb)]))


def join(R_early: np.ndarray, R_late: np.ndarray, i_early: int, i_late: int, L_early: float) -> np.ndarray:
    """One route per bin from two traces' carried routes (n, N_e, 2) and (n, N_l, 2) (rows by series index; the
    traces at rows ``i_early`` < ``i_late``): up to ``i_early`` the early trace's; from ``i_late`` the late trace's;
    between, the late trace's route with its part beyond the early trace's length moved onto the early trace's
    carried apex, fully at ``i_early`` and less and less towards ``i_late``. Length N_l."""
    n, N = R_late.shape[:2]
    out = R_late.copy()
    Ne = R_early.shape[1]
    k = min(N, Ne)
    out[: i_early + 1, :k] = R_early[: i_early + 1, :k]
    j = int(min(max(round(L_early), 0), N - 1, Ne - 1))
    for i in range(i_early + 1, i_late):
        w = (i_late - i) / max(i_late - i_early, 1)
        off = R_early[i, j] - R_late[i, j]
        out[i, :j + 1] = R_early[i, :j + 1]
        out[i, j + 1:] = R_late[i, j + 1:] + w * off
    return out


def fit_length(route: np.ndarray, N: int) -> np.ndarray:
    """A 1 px route cut or carried on straight to exactly ``N`` points."""
    if len(route) >= N:
        return route[:N]
    return extend(route, N - len(route))[:N]


def forward_join(R: np.ndarray, i_from: int, L: float, model_routes: list, tol: float = 6.0,
                 extend_px: float = EXTEND_PX, max_n: int = 800) -> np.ndarray:
    """After row ``i_from`` (a trace of length ``L``), each row's route beyond the trace's carried apex follows the
    model's route at that bin (``model_routes[i]``, reference coordinates, or None) when the apex lies within
    ``tol`` px of it: the tube's future route as the model read it then, instead of the route the trace's bin had,
    carried on. (n, N, 2), N the longest such route (at most ``max_n``)."""
    n, N0 = R.shape[:2]
    j = int(min(max(round(L), 1), N0 - 1))
    rows = []
    for i in range(n):
        route = R[i]
        M = model_routes[i] if i > i_from else None
        if M is not None and len(M) >= 2:
            M = dense(M, 1.0)
            sM = arclen(M)
            apex = R[i, j]
            sp, d = project(M, apex)
            if d <= tol and sp < sM[-1] - 2.0:
                tail = M[sM > sp + 1.0] + (apex - point_at(M, sp)[0])
                route = extend(dense(np.vstack([R[i, :j + 1], tail]), 1.0), extend_px)
        rows.append(route)
    N = int(min(max_n, max(len(r) for r in rows)))
    return np.stack([fit_length(r, N) for r in rows])


def kymograph(gf: GrainFlow, routes_ref: np.ndarray, ws=(1.0, 2.0), step: float = 0.5,
              rows: range | None = None) -> dict:
    """K[w] (n, N): the tube map maxed over the band of half-width w across each route point, per bin (row i = bin
    rs + i)."""
    n, N = routes_ref.shape[:2]
    out = {w: np.zeros((n, N), np.float32) for w in ws}
    wmax = max(ws)
    offs = np.arange(-wmax, wmax + 1e-9, step)
    for i in (rows if rows is not None else range(n)):
        pm = gf.pmap(gf.rs + i)
        pts = gf.to_crop(routes_ref[i])
        nn = normals(pts)
        qx = (pts[:, 0][None, :] + offs[:, None] * nn[:, 0][None, :]).astype(np.float32)
        qy = (pts[:, 1][None, :] + offs[:, None] * nn[:, 1][None, :]).astype(np.float32)
        vals = cv2.remap(pm, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for w in ws:
            out[w][i] = vals[np.abs(offs) <= w + 1e-9].max(axis=0)
    return out


# ----------------------------------------------------------------------------- reading the front
def dp_front(evidence: np.ndarray, vmax, lo: np.ndarray | None = None, hi: np.ndarray | None = None,
             unary: np.ndarray | None = None) -> np.ndarray:
    """Non-decreasing front index per bin (0 <= f[t] - f[t-1] <= vmax[t]) maximising the summed evidence behind it
    (the front at f claims points [0, f)), with per-bin bounds lo[t] <= f[t] <= hi[t] (sparsetrack.analyze.dp_front,
    with bounds and a per-bin speed cap); ``unary`` (n_bins, n + 1) is added to the score of each front position."""
    n_bins, n = evidence.shape
    vm = np.broadcast_to(np.asarray(vmax, int), (n_bins,))
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    if unary is not None:
        gain = gain + unary
    idx = np.arange(n + 1)
    if lo is not None or hi is not None:
        lo_ = np.zeros(n_bins, int) if lo is None else np.asarray(lo, int)
        hi_ = np.full(n_bins, n, int) if hi is None else np.asarray(hi, int)
        gain = np.where((idx[None] >= lo_[:, None]) & (idx[None] <= hi_[:, None]), gain, -1e18)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
    for t in range(1, n_bins):
        v = max(int(vm[t]), 0)
        pad = np.concatenate([np.full(v, -np.inf), score])
        windows = np.lib.stride_tricks.sliding_window_view(pad, v + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + windows[idx, j]
        back[t] = idx - (v - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def speed_caps(n: int, anchors: list, start: int, k_v: float, v_floor: float, c: float) -> np.ndarray:
    """Per-bin speed caps (px per bin) from the traced lengths: per stretch between traces (and from ``start``, the
    earliest bin the tube may have started at, to the first) ``k_v`` x the mean speed the traces need there, at least
    ``v_floor`` and never too slow to reach the next trace; after the last trace, the last stretch's cap."""
    vm = np.full(n, max(1, int(math.ceil(v_floor))), int)
    pts = [(int(i), float(L) + c) for i, L in sorted(anchors)]
    prev_i, prev_L = min(start, pts[0][0]) - 1, 0.0
    cap = vm[0]
    for i, L in pts:
        need = max(L - prev_L, 0.0) / max(i - prev_i, 1)
        cap = max(int(math.ceil(v_floor)), int(math.ceil(k_v * need)), int(math.ceil(need)) + 1)
        vm[max(prev_i + 1, 0): i + 1] = cap
        prev_i, prev_L = i, L
    vm[prev_i + 1:] = cap
    return vm


def read_lengths(K: np.ndarray, anchors: list, theta: float, skip: float = 0.0, c: float = 0.0, k_v: float = 2.0,
                 v_floor: float = 1.0, onset: tuple | None = None, start: int | None = None,
                 prior: np.ndarray | None = None, lam: float = 0.0, lower: list | None = None,
                 zero: list | None = None) -> np.ndarray:
    """Lengths at every bin (row) of kymograph ``K`` (n, N; 1 px of arc per column) through the traced lengths
    ``anchors`` [(row, px)]: the best monotone front on K - theta (the first ``skip`` px neutral) that passes through
    each traced length (plus the tip offset ``c``), length = arc to the front less c. ``onset`` (last absent row,
    first visible row): no tube up to the first, at least 1 px from the second. ``start``: the earliest row the tube
    is taken to have started at for the speed caps (default the onset's first visible row, else row 0). ``prior``
    (n,) lengths and ``lam``: each px the length lies from the prior costs ``lam``. ``lower`` [(row, px)]: at least
    that long there (a partly traced tube); ``zero`` [row]: no tube there. Traced lengths that shrink are raised to the
    longest before them (a front cannot shrink)."""
    n, N = K.shape
    E = K.astype(np.float64) - theta
    ns = int(math.ceil(skip))
    if ns > 0:
        E[:, :ns] = 0.0
    lo, hi = np.zeros(n, int), np.full(n, N, int)
    F = lambda L: int(min(N, max(1, round(L + c) + 1)))
    anchors = sorted((int(i), float(L)) for i, L in anchors if 0 <= int(i) < n)
    run_max = 0.0
    fixed = []
    for i, L in anchors:
        run_max = max(run_max, L)
        fixed.append((i, run_max))
        lo[i] = hi[i] = F(run_max) if run_max > 0 else 0
    for i, L in lower or []:
        if 0 <= int(i) < n and L > 0:
            lo[int(i)] = max(lo[int(i)], F(L))
    for i in zero or []:
        if 0 <= int(i) < n:
            hi[: int(i) + 1] = 0
    lo = np.minimum(lo, hi)
    if onset is not None:
        la, fv = onset
        if la is not None and la >= 0:
            hi[: la + 1] = np.minimum(hi[: la + 1], 0)
        if fv is not None:
            lo[max(fv, 0):] = np.maximum(lo[max(fv, 0):], F(1.0))
        lo = np.minimum(lo, hi)
    if start is None:
        start = onset[1] if onset is not None and onset[1] is not None else 0
    vm = speed_caps(n, fixed, int(start), k_v, v_floor, c) if fixed else np.full(n, max(1, int(v_floor)))
    if onset is not None and onset[1] is not None and 0 <= onset[1] < n:
        vm[onset[1]] = max(vm[onset[1]], F(1.0))  # the tube may appear at once at its first visible bin
    unary = None
    if prior is not None and lam > 0:
        Lf = np.maximum(np.arange(N + 1, dtype=float) - 1.0 - c, 0.0)  # length at front f
        Lf[0] = 0.0
        unary = -lam * np.abs(Lf[None, :] - np.asarray(prior, float)[:, None])
    f = dp_front(E, vm, lo, hi, unary)
    if np.any((f < lo) | (f > hi)):  # the bounds could not all be met within the caps
        f = dp_front(E, np.full(n, N), lo, hi, unary)
    S = np.arange(N, dtype=float)
    L = np.where(f >= 1, S[np.maximum(f - 1, 0)] - c, 0.0)
    L = np.maximum(L, 0.0)
    for i, La in anchors:
        L[i] = La
    return L
