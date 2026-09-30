"""A stalled flood carried on by a tip tracker (``Params.flood_tip_track``, off by default).

A tube grows only at its tip. In the difference of 3-bin means a few bins apart (the short-interval difference) the
growing apex shows as new material, also where the learned tube map has lost the tube: movie 2's faint, light-cored
tubes fade from the map as they grow long, and the flood (sparsetrack/learned.py), which reads the map, stops there.

The tracker follows the tip bin by bin through the short-interval difference: a beam search over moves in a forward
fan (up to the movie's growth cap per bin, turns of at most 40 degrees), pausing when nothing new appears; it never
steps onto a grain. Lengths are re-traced at each bin from the tube's exit to the tracked tip along that bin's tube
probability, inside a corridor round the track and the flood's own tube, rather than by the track's arc length (the
tip jitters from bin to bin).

Where the flood stopped growing and the tracker kept going, what decides? A continuation is kept only when
- the tracker was on the flood's tube when the flood stopped (its tip within ``tt_agree_px`` of the flood's),
- it adds at least ``tt_min_gain_px`` by the last bin,
- and its path was laid down the way a growing tip lays a tube down: the change from "before" came to stay along at
  least ``tt_support`` of it, later the farther out (rank correlation at least ``cont_order``, the tip
  continuation's test in analyze.py), and no stretch longer than 3 bins of growth at the speed cap came at once.
The tracker is started twice: from the flood's exit where the flood first claimed its tube, and from the flood's
tip where it stopped.

Measured 29 Sep 2026 (see the Params comment in analyze.py). On movie 2 the growing apex is a weak landmark: at the
annotator's growing traces it beats every other local peak of the short-interval difference within 50 px in 16 of
60 (ld 68 of 113) and beats its own tube's body 8+ px behind it in 27 of 42 (a tube that moves lights up all along).
Started at the flood's onset, the tracker alone reads 17 of 50 FULL m2 traces of flood-read grains against the
flood's 22; it loses tips at illumination blobs and onto neighbours' tubes, and after a tube stops it drifts on
noise. Its continuations on m2 that looked right are not kept: g082 at bin 244 (51 of 53 px, tip 4 px) ran over
change that came to stay along only 7% of its path, and g092 at bin 244 (159 of 153 px) was found from float16
copies of the frames but not from the float32 frames themselves.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class Frames:
    """One grain's registered image crops (the flood's frame: cropped at ``off_abs``, warped by ``resid``), made on
    demand and kept for a few bins (the tracker walks forward through the movie)."""

    def __init__(self, renderer, gx: float, gy: float, half: int, rs: int, n: int, off_abs, resid, keep: int = 24):
        self.r, self.gx, self.gy, self.half, self.rs, self.n = renderer, gx, gy, half, rs, n
        self.off_abs, self.resid, self.keep = off_abs, np.asarray(resid, float), keep
        self._cache: dict = {}

    def __len__(self):
        return self.n

    def __call__(self, t: int) -> np.ndarray:
        if t not in self._cache:
            c = np.nan_to_num(self.r.crop(self.rs + t, self.gx, self.gy, self.half, self.off_abs)).astype(np.float32)
            dx, dy = self.resid[t]
            self._cache[t] = cv2.warpAffine(c, np.float32([[1, 0, -dx], [0, 1, -dy]]), (2 * self.half, 2 * self.half),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if len(self._cache) > self.keep:
                for k in sorted(self._cache, key=lambda b: abs(b - t))[self.keep:]:
                    del self._cache[k]
        return self._cache[t]

    def mean3(self, t: int) -> np.ndarray:
        return np.mean([self(j) for j in range(max(t - 1, 0), min(t + 2, self.n))], axis=0)


def short_z(frames, t: int, k: int, valid: np.ndarray, sigma: float = 1.0) -> np.ndarray | None:
    """|short-interval difference| at bin t (3-bin means k bins apart, lightly blurred) in robust units over the
    ``valid`` pixels, 3x3 max-filtered; None before bin k + 1."""
    if t < k + 1:
        return None
    d = np.abs(cv2.GaussianBlur(frames.mean3(t) - frames.mean3(t - k), (0, 0), sigma))
    v = d[valid]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-3
    return cv2.dilate(((d - med) / mad).astype(np.float32), np.ones((3, 3), np.uint8))


def track_tip(frames, valid: np.ndarray, blocked: np.ndarray, t0: int, x0: float, y0: float, th0: float,
              vmax: float = 4.5, k: int = 6, floor: float = 2.5, ahead_w: float = 0.5, ahead_px: float = 3.0,
              turn_cost: float = 0.4, beam: int = 40, turns=(-40, -20, -10, 0, 10, 20, 40)) -> np.ndarray:
    """The tip's position per bin from bin ``t0`` at (``x0``, ``y0``) heading ``th0`` (radians), (n, 2) (x, y), NaN
    before t0. Per bin the tip pauses (gain 0) or steps forward (0.5 px up to ``vmax``, turning by one of ``turns``
    degrees), gaining (z - floor) + ahead_w (z - z ``ahead_px`` further on) - ``turn_cost`` per 45 degrees, z the
    short-interval evidence where it lands (``short_z``): a move is worth it only onto new material with nothing
    new beyond it. The best-scoring beam's history is returned."""
    n = len(frames)
    steps = [s for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0) if s <= vmax + 1e-9]
    mv = [(0.0, 0.0)] + [(s, math.radians(a)) for s in steps for a in turns]
    D = np.array([m[0] for m in mv])
    T = np.array([m[1] for m in mv])
    h, w = blocked.shape
    S = np.zeros(1)
    X, Y, TH = np.array([x0], float), np.array([y0], float), np.array([th0], float)
    hist = []
    for t in range(t0 + 1, n):
        Z = short_z(frames, t, k, valid)
        if Z is None:
            continue
        th2 = TH[:, None] + T[None, :]
        x2 = X[:, None] + D[None, :] * np.cos(th2)
        y2 = Y[:, None] + D[None, :] * np.sin(th2)
        xi, yi = np.round(x2).astype(int), np.round(y2).astype(int)
        ok = (xi >= 1) & (xi < w - 1) & (yi >= 1) & (yi < h - 1)
        xi, yi = np.clip(xi, 1, w - 2), np.clip(yi, 1, h - 2)
        ok &= ~blocked[yi, xi] | (D[None, :] == 0)  # never onto a grain
        z = Z[yi, xi]
        xa = np.clip(np.round(x2 + ahead_px * np.cos(th2)).astype(int), 0, w - 1)
        ya = np.clip(np.round(y2 + ahead_px * np.sin(th2)).astype(int), 0, h - 1)
        gain = np.where(D[None, :] > 0, (z - floor) + ahead_w * (z - Z[ya, xa])
                        - turn_cost * np.abs(T[None, :]) / math.radians(45), 0.0)
        tot = np.where(ok, S[:, None] + gain, -np.inf)
        flat = np.argsort(-tot, axis=None)
        keys = (np.round(x2 * 2).astype(np.int64) * 100000 + np.round(y2 * 2).astype(np.int64)).ravel()[flat]
        _, first = np.unique(keys, return_index=True)  # one beam per half-pixel place, its best history
        pick = flat[np.sort(first)]
        pick = pick[np.isfinite(tot.ravel()[pick])]
        if not len(pick):
            break
        pick = pick[np.argsort(-tot.ravel()[pick])][:beam]
        bi, mi = np.unravel_index(pick, tot.shape)
        S, X, Y = tot[bi, mi], x2[bi, mi], y2[bi, mi]
        TH = np.where(D[mi] > 0, th2[bi, mi], TH[bi])
        hist.append((t, X.copy(), Y.copy(), bi.copy()))
    out = np.full((n, 2), np.nan)
    out[t0] = (x0, y0)
    if hist:
        j = int(np.argmax(S))
        for t, xs, ys, par in reversed(hist):
            out[t] = (xs[j], ys[j])
            j = int(par[j])
    return out


def retrace(prob: np.ndarray, blocked: np.ndarray, start, end, corridor: np.ndarray | None = None,
            eps: float = 0.05) -> float:
    """Length of the cheapest path from ``start`` to ``end`` ((x, y) px) on cost 1 / (P + eps) (P in [0, 1]), inside
    ``corridor`` and never through ``blocked`` (the two ends excepted), after a 5-point moving average; NaN if none."""
    from skimage.graph import MCP_Geometric
    h, w = prob.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    ex, ey = int(round(end[0])), int(round(end[1]))
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return float("nan")
    pad = 20
    x0, x1 = max(min(sx, ex) - pad, 0), min(max(sx, ex) + pad + 1, w)
    y0, y1 = max(min(sy, ey) - pad, 0), min(max(sy, ey) + pad + 1, h)
    if corridor is not None:
        ys, xs = np.nonzero(corridor)
        if len(xs):
            x0, x1 = max(min(x0, xs.min()), 0), min(max(x1, xs.max() + 1), w)
            y0, y1 = max(min(y0, ys.min()), 0), min(max(y1, ys.max() + 1), h)
    bad = blocked[y0:y1, x0:x1].copy()
    if corridor is not None:
        bad |= ~corridor[y0:y1, x0:x1]
    bad[sy - y0, sx - x0] = bad[ey - y0, ex - x0] = False
    m = MCP_Geometric(np.where(bad, np.inf, 1.0 / (prob[y0:y1, x0:x1] + eps)), fully_connected=True)
    m.find_costs([(sy - y0, sx - x0)], [(ey - y0, ex - x0)])
    try:
        path = np.array(m.traceback((ey - y0, ex - x0)), float)[:, ::-1] + [x0, y0]
    except ValueError:
        return float("nan")
    if len(path) > 7:
        kk = np.ones(5) / 5
        path = np.stack([np.convolve(np.pad(path[:, i], 2, mode="edge"), kk, "valid") for i in range(2)], axis=1)
    return float(np.sum(np.hypot(*np.diff(path, axis=0).T))) if len(path) > 1 else 0.0


def _hp(img: np.ndarray, sigma: float = 1.0, hp: float = 8.0) -> np.ndarray:
    """|img minus its 8 px blur|, lightly blurred: fine structure, without illumination drifts."""
    return np.abs(cv2.GaussianBlur(img - cv2.GaussianBlur(img, (0, 0), hp), (0, 0), sigma))


def arrivals_along(frames, pts: np.ndarray, blocked: np.ndarray, hold: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Along the path ``pts`` ((x, y), in order): each point's distance along it, and the bin the change from
    "before" arrived there to stay (above 5 robust sigmas of the end-state change, in ``hold`` of the remaining bins;
    the earliest within 1 px; len(frames) where it never did)."""
    n = len(frames)
    early = frames.mean3(1)
    late = _hp(frames.mean3(n - 3) - early)
    v = late[~blocked]
    thr = float(np.median(v) + 5 * 1.4826 * np.median(np.abs(v - np.median(v))))
    h, w = blocked.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    m = 30  # the blur's reach: the window round the path gives the whole crop's values there
    x0, x1 = max(xi.min() - m, 0), min(xi.max() + m + 1, w)
    y0, y1 = max(yi.min() - m, 0), min(yi.max() + m + 1, h)
    above = np.stack([_hp(frames.mean3(t)[y0:y1, x0:x1] - early[y0:y1, x0:x1]) > thr for t in range(n)])
    rest = np.cumsum(above[::-1], axis=0)[::-1]
    ok = above & (rest >= hold * np.arange(n, 0, -1)[:, None, None])
    arr = np.where(ok.any(axis=0), ok.argmax(axis=0), n)
    a = np.array([arr[max(y - y0 - 1, 0):y - y0 + 2, max(x - x0 - 1, 0):x - x0 + 2].min() for x, y in zip(xi, yi)],
                 float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]), a


def arrival_order(s: np.ndarray, a: np.ndarray, n: int) -> float:
    """Rank correlation of distance along a path with arrival bin (``arrivals_along``) over the points that changed to
    stay; NaN with fewer than 4 of them."""
    from scipy.stats import rankdata
    keep = a < n
    if keep.sum() < 4:
        return float("nan")
    rx, ry = rankdata(s[keep]), rankdata(a[keep])
    return 0.0 if np.std(rx) == 0 or np.std(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1])


def longest_block(s: np.ndarray, a: np.ndarray, n: int, bins: int = 2) -> float:
    """The longest stretch of a path (px, over its points that changed to stay, in order) whose change arrived within
    ``bins`` bins: a tube lays itself down at most the growth cap per bin, so a long stretch arriving at once is
    material that came from elsewhere (a moved or crossing tube, a landing piece)."""
    keep = np.nonzero(a < n)[0]
    best, i = 0.0, 0
    for j in range(len(keep)):
        while a[keep[i:j + 1]].max() - a[keep[i:j + 1]].min() > bins:
            i += 1
        best = max(best, float(s[keep[j]] - s[keep[i]]))
    return best


def stall_bin(length: np.ndarray, grow_px: float = 1.0, window: int = 10) -> int | None:
    """The last bin by which the flood's length grew by more than ``grow_px`` over ``window`` bins (None: never)."""
    g = [t for t in range(window, len(length)) if length[t] - length[t - window] > grow_px]
    return g[-1] if g else None


def continue_flood(frames, pstack: np.ndarray, blocked: np.ndarray, fl: dict, length: np.ndarray,
                   tips: np.ndarray, line: list, onset: int, vmax: float, p, diag: list | None = None) -> dict | None:
    """The flood's reading carried on by the tracker where it stopped (see the module docstring), or None.

    ``length``: the flood's length per bin; ``tips``: its tip per bin, (x, y) crop px; ``line``: its drawn
    centreline, (y, x), exit first; ``onset``: its onset bin. Returns {"length", "tips" (x, y; NaN where the flood's
    tip stands), "start" (the bin the flood stopped), "gain" (px at the last bin), "mode" ("onset" or "stall")}.
    ``diag``, if given, collects each candidate's tests."""
    n = len(frames)
    ts = stall_bin(length)
    if ts is None or ts > n - p.tt_stall_bins or onset is None or len(line) < 2:
        return None
    valid = ~blocked
    ex = np.array([line[0][1], line[0][0]], float)
    nxt = np.array([line[min(3, len(line) - 1)][1], line[min(3, len(line) - 1)][0]], float)
    starts = [("onset", onset, ex, math.atan2(*(nxt - ex)[::-1]) if np.hypot(*(nxt - ex)) > 1 else
               math.atan2(ex[1] - (frames.half - 0.5), ex[0] - (frames.half - 0.5)))]
    back = tips[max(onset, ts - 10)]
    head = (math.atan2(*(tips[ts] - back)[::-1]) if np.hypot(*(tips[ts] - back)) > 2 else
            math.atan2(line[-1][0] - line[-2][0], line[-1][1] - line[-2][1]))
    starts.append(("stall", ts, tips[ts].astype(float), head))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # the corridor: 15 px round track and tube

    def corridor(tr, t0, i):
        m = np.zeros(blocked.shape, np.uint8)
        pts = tr[t0:i + 1]
        for q in pts[np.isfinite(pts[:, 0])][::2]:
            cv2.circle(m, (int(round(q[0])), int(round(q[1]))), 1, 1, -1)
        m |= (fl["tube"] & (fl["t_in"] <= i) & (fl["t_in"] >= 0)).astype(np.uint8)
        cv2.circle(m, (int(round(ex[0])), int(round(ex[1]))), 3, 1, -1)
        return cv2.dilate(m, kernel) > 0

    for mode, t0, (x0, y0), th0 in starts:
        tr = track_tip(frames, valid, blocked, int(t0), float(x0), float(y0), th0, vmax=vmax, k=p.tt_bins,
                       floor=p.tt_floor)
        if mode == "stall":
            tr[:t0] = tips[:t0]
        rec = {"mode": mode, "agree_px": float(np.hypot(*(tr[ts] - tips[ts]))) if np.isfinite(tr[ts, 0]) else None}
        if diag is not None:
            diag.append(rec)
        if not np.isfinite(tr[ts, 0]) or np.hypot(*(tr[ts] - tips[ts])) > p.tt_agree_px:
            continue  # the tracker was not on the flood's tube when the flood stopped
        last = n - 1
        if not np.isfinite(tr[last, 0]):
            continue
        gain = retrace(pstack[last] / 250.0, blocked, ex, tr[last], corridor(tr, t0, last)) - length[last]
        rec["gain_px"] = float(gain)
        if not gain >= p.tt_min_gain_px:
            continue
        seg = tr[ts:][np.isfinite(tr[ts:, 0])]
        path = [seg[0]]
        for q in seg[1:]:
            if math.hypot(*(q - path[-1])) >= 1.0:
                path.append(q)
        if len(path) < 4:
            continue
        s, a = arrivals_along(frames, np.array(path), blocked)
        rec.update(order=arrival_order(s, a, n), support=float(np.mean(a < n)), block_px=longest_block(s, a, n),
                   path_px=float(s[-1]))
        if not rec["order"] >= p.cont_order or rec["support"] < p.tt_support:
            continue  # not laid down tip first, or mostly over nothing that stayed (the tip wandered)
        if rec["block_px"] > 3.0 * vmax + 3.0:
            continue  # a stretch longer than 3 bins of growth at the cap (plus blur) came at once: not its tip's
        new = np.array(length, float)
        tip_out = np.full((n, 2), np.nan)
        for i in range(ts, n):
            if not np.isfinite(tr[i, 0]):
                continue
            r = retrace(pstack[i] / 250.0, blocked, ex, tr[i], corridor(tr, t0, i))
            if np.isfinite(r) and r > new[i]:
                new[i] = r
                tip_out[i] = tr[i]
        new[ts:] = np.maximum.accumulate(new[ts:])
        return {"length": new, "tips": tip_out, "start": ts, "gain": float(gain), "mode": mode}
    return None
