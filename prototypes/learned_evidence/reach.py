"""Decoder v2, the per-bin decoder: read the tube where it is in every bin, then make its length monotone.

SparseTrack traces one centreline on the end-state change map and reads every bin along that
path (rotated rigidly). With a tube-probability map for every bin that is no longer necessary:

1. per grain, follow its drift (SparseTrack's ``local_shifts`` on the image cache);
2. per bin, keep P > ``thr`` outside grain bodies, take the connected region attached to the
   grain's rim, and split it from regions reaching other grains' rims by geodesic ownership
   (SparseTrack's ``geodesic_owner``);
3. the bin's raw length is the geodesic length along that region's medial axis, from its end
   nearest the grain (plus that end's distance from the rim), plus 1 px;
4. over bins, the length is the monotone curve (growth <= ``vmax`` px per bin) closest to the raw
   lengths in L1, so single-bin misses and flickers do not move it; onset is where it passes
   ``onset_px``.

Writes predictions in SparseTrack's schema, so ``sparsetrack.evaluate.score`` scores them.

Options for real footage. The first three, and ``analyze(ghosts=True)``, are the defaults since 25 Sep 2026: on
five held-out synthetic movies they changed lengths within tolerance by +12 (95% CI -12 to +39) and onsets by 0, and
on the real sample movie they fixed what an audit by eye found (edge grains, grains that left, ghost discs, a fit
dragged down by failed readings). ``edge_mask="movie", gone="flag", fit="l1"`` with ``ghosts=False`` reproduce the
earlier frozen decoder. The others were tried and are off (``track``, ``burst_run``, ``abrupt``, ``owner``,
``continuity``):

* ``edge_mask="bin"``: a pixel whose source lies outside the movie frame is blocked in that bin only
  (``frame_outside``), not in every bin as soon as it leaves the frame in any bin: a grain drifting
  towards an edge keeps its tube in the bins where it is visible.
* ``gone="hold"``: bins where the grain is no longer at its tracked position (its disc has lost its
  contrast for a run of bins, ``gone_bins``) are not readings: the growth fit skips them, so what is read
  at an empty spot (a tube passing by, a neighbour's) is not the grain's. They are returned under
  ``gone_bins`` and flagged: ``no_grain_after:`` when the grain does not come back, ``gone:first-last`` (frames)
  for a spell away that ends.
* ``track="reacquire"``: when the grain is gone from the tracked position, look for it again nearby
  (``reacquire``) and follow it from there: grains that jump or are dragged tens of px.
* ``fit="grown"``: the growth fit may not drop below the lengths the tube has been *grown* to
  (``grown_floor``): tip growth means a tube read this long after a steady climb is at least this long
  later, so a long run of failed short readings (a region split at a crossing, a half-missed tube) no
  longer drags the whole curve down.
* ``burst_run``: a burst must start a run of at least this many consecutive collapsed readings
  (``burst_cut``); ``abrupt``: bins just after an abrupt field-wide change cannot start a burst
  (``abrupt_bins`` finds them).
* ``owner="passby"``: another grain's ring that the region only passes tangentially does not split the
  region (``passby``): a tube growing past a neighbour is not cut at the neighbour's ring.
* ``continuity="path"``: tip growth (``TipContinuity``): a reading longer than the tube's accepted path stands only
  if it continues that path at the tip; a tube crossing or touching it, one lying across the tip's way before the tip
  got there, a neighbour's tube at the rim make the farthest point jump off the grain's own tube, and such a bin is
  read along the path instead (or held). Development numbers (26 Sep 2026): +2 lengths in tolerance on the thin
  development movies (95% CI -3 to +9), +3 on the thick one, the faint and wide ones unchanged; on the real sample
  movie a tie with the shipped network (one end length fixed, one broken) but most foreign over-reads removed with
  a network that sees thick tubes (end lengths within 25% of an audit by eye: 16 -> 20 of 34). Off by default: the
  rule fixed beforehand asked for more fixes than breaks on the real movie with the shipped network.
* ``analyze(..., ghosts=True)``: census discs that are soft and faint in every early window (out-of-focus
  ghosts, smears of grains still landing) are not analysed and not treated as grains (``census_ghosts``); the
  review pre-fill excludes them as "not_a_grain", so a person can re-include one.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

import sparsetrack.analyze as A
from sparsetrack import grains as census_mod
from sparsetrack import stack
from sparsetrack.evaluate import PRED_SCHEMA
from sparsetrack.render import Renderer


# ----------------------------------------------------------------------------- growth fits
def monotone_l1(raw: np.ndarray, vmax: float, step: float = 0.5) -> np.ndarray:
    """Non-decreasing sequence with increments <= vmax minimising sum |x - raw| (grid DP)."""
    levels = np.arange(0.0, float(raw.max()) + 2 * step, step)
    k = max(1, int(round(vmax / step)))
    n, m = len(raw), len(levels)
    cost = np.abs(levels[None, :] - raw[:, None])
    best = cost[0].copy()
    back = np.zeros((n, m), np.int32)
    idx = np.arange(m)
    for t in range(1, n):
        pad = np.concatenate([np.full(k, np.inf), best])
        win = np.lib.stride_tricks.sliding_window_view(pad, k + 1)  # win[j] covers levels j-k..j
        j = np.argmin(win, axis=1)
        best = cost[t] + win[idx, j]
        back[t] = idx - (k - j)
    out = np.zeros(n, np.int32)
    out[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return levels[out]


def monotone_fit(raw: np.ndarray, vmax: float, step: float = 0.5, weight: np.ndarray | None = None,
                 floor: np.ndarray | None = None, floor_w: float = 0.0) -> np.ndarray:
    """Non-decreasing sequence x with increments <= vmax minimising
    sum_t weight_t |x_t - raw_t| + floor_w sum_t (floor_t - x_t)+ (grid DP; NaN readings cost nothing). Without
    weights and floor it is ``monotone_l1`` (same grid, window and tie order)."""
    raw = np.asarray(raw, float)
    ok = np.isfinite(raw)
    r = np.where(ok, raw, 0.0)
    w = np.where(ok, 1.0 if weight is None else np.asarray(weight, float), 0.0)
    hi = float(r.max()) if floor is None else max(float(r.max()), float(np.max(floor)))
    levels = np.arange(0.0, hi + 2 * step, step)
    k = max(1, int(round(vmax / step)))
    n, m = len(r), len(levels)
    cost = w[:, None] * np.abs(levels[None, :] - r[:, None])
    if floor is not None and floor_w > 0:
        cost = cost + floor_w * np.maximum(np.asarray(floor, float)[:, None] - levels[None, :], 0.0)
    best = cost[0].copy()
    back = np.zeros((n, m), np.int32)
    idx = np.arange(m)
    for t in range(1, n):
        pad = np.concatenate([np.full(k, np.inf), best])
        win = np.lib.stride_tricks.sliding_window_view(pad, k + 1)  # win[j] covers levels j-k..j
        j = np.argmin(win, axis=1)
        best = cost[t] + win[idx, j]
        back[t] = idx - (k - j)
    out = np.zeros(n, np.int32)
    out[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return levels[out]


def grown_floor(raw: np.ndarray, vmax: float, start_px: float = 10.0, tol: float = 3.0, gap: int = 3,
                k: int = 3, frac: float = 0.9, min_px: float = 16.0, span: int = 3) -> np.ndarray:
    """Lower bound on the length from readings the tube was *grown* to (NaN = no reading).

    A reading is grown when a chain of readings leads up to it from a short one (<= ``start_px``), each rising at
    most ``vmax`` px per bin (+``tol``) over at most ``gap`` bins. Tip growth means a tube read this long is at
    least this long later, apart from a burst; a foreign tube joining the region or a tracker landing on another
    tube shows up as a sudden jump, which no chain supports. The floor at bin t is ``frac`` x the ``k``-th longest
    grown reading up to t (``k`` readings agree), counting readings of at least ``min_px`` whose chain spans at
    least ``span`` bins."""
    raw = np.asarray(raw, float)
    n = len(raw)
    ok = np.isfinite(raw)
    grown = np.zeros(n, bool)
    first = np.full(n, -1)
    for t in range(n):
        if not ok[t]:
            continue
        if raw[t] <= start_px + tol:
            grown[t], first[t] = True, t
            continue
        for s in range(t - 1, max(-1, t - gap - 1), -1):
            if grown[s] and raw[t] <= raw[s] + vmax * (t - s) + tol:
                grown[t], first[t] = True, first[s]
                break
    floor = np.zeros(n)
    vals: list[float] = []
    for t in range(n):
        if grown[t] and raw[t] >= min_px and t - first[t] >= span:
            vals = sorted(vals + [float(raw[t])], reverse=True)[:k]
        if len(vals) >= k:
            floor[t] = frac * vals[k - 1]
        if t:
            floor[t] = max(floor[t], floor[t - 1])
    return floor


def burst_cut(raw: np.ndarray, min_px: float = 8.0, frac: float = 0.3, hold: float = 0.9,
              min_bins: int = 5, run: int = 1, skip: np.ndarray | None = None) -> int | None:
    """First bin of a burst: the tube had been read at ``min_px`` or more, and from this bin on its
    reading stays below ``frac`` of the longest reading so far in at least ``hold`` of the remaining
    bins (at least ``min_bins`` of them). A burst tube leaves nothing to trace, so its reading collapses
    for good; a monotone fit over the whole movie would instead pull its growth down.

    NaN readings (grain gone) are not counted, neither as collapsed readings nor towards ``min_bins``. With
    ``run`` > 1 the burst must start a run of that many consecutive collapsed readings (one low reading
    followed by long ones is a failed reading, not a burst); bins flagged in ``skip`` (just after an abrupt
    field-wide change) cannot start one."""
    raw = np.asarray(raw, float)
    ok = np.isfinite(raw)
    r = np.where(ok, raw, np.nan)
    peak = np.maximum.accumulate(np.where(ok, raw, 0.0))
    for b in range(1, raw.size - min_bins + 1):
        if not ok[b] or (skip is not None and skip[b]):
            continue
        lim = frac * peak[b - 1]
        if peak[b - 1] >= min_px and r[b] < lim:
            rest = r[b:][ok[b:]]
            if len(rest) < min_bins:
                continue
            if run > 1:
                head = rest[:run]
                if len(head) < run or np.any(head >= lim):
                    continue
            if np.mean(rest < lim) >= hold:
                return b
    return None


def fit_lengths(raw: np.ndarray, vmax: float, burst: bool, min_tube_px: float = 8.0, fit: str = "l1",
                burst_run: int = 1, skip_burst: np.ndarray | None = None, **fkw) -> tuple[np.ndarray, int | None]:
    """Growth curve over the raw readings (NaN = not a reading) and the burst bin or None.

    ``fit="l1"``: monotone L1 up to a burst, then held (the frozen decoder). ``fit="grown"``: the same, but the
    curve may not drop below ``grown_floor`` (soft: ``floor_w`` per px and bin)."""
    raw = np.asarray(raw, float)
    ok = np.isfinite(raw)
    if not ok.any() or np.nanmax(raw) <= 0:
        return np.zeros(raw.size), None
    cut = burst_cut(raw, min_px=min_tube_px, run=burst_run, skip=skip_burst) if burst else None
    end = raw.size if cut is None else cut
    r = raw[:end]
    if fit == "l1":
        head = monotone_fit(r, vmax)
    elif fit == "grown":
        gkw = {k: v for k, v in fkw.items() if k in ("start_px", "tol", "gap", "k", "frac", "min_px", "span")}
        head = monotone_fit(r, vmax, floor=grown_floor(r, vmax, **gkw), floor_w=fkw.get("floor_w", 5.0))
    else:
        raise ValueError(f"unknown fit {fit!r}")
    if not np.isfinite(r).any():
        head = np.zeros(end)
    out = np.concatenate([head, np.full(raw.size - end, head[-1] if end else 0.0)]) if cut is not None else head
    return out, cut


# ----------------------------------------------------------------------------- frame edge (change 1)
def frame_outside(width: int, height: int, gx: float, gy: float, half: int, shift: np.ndarray,
                  margin: float = 1.0) -> np.ndarray:
    """Grain-frame pixels whose source lies outside the movie frame (within ``margin`` px of its edge) in one bin,
    given that bin's total shift (global registration + the grain's own drift)."""
    j = np.arange(2 * half, dtype=np.float64) + 0.5
    xs, ys = gx - half + j + shift[0], gy - half + j + shift[1]
    return ((ys < margin) | (ys >= height - margin))[:, None] | ((xs < margin) | (xs >= width - margin))[None, :]


# ----------------------------------------------------------------------------- grain presence (change 2)
def disc_contrast(img: np.ndarray, ls: np.ndarray, centre: float, gr: float, rg: np.ndarray) -> np.ndarray:
    """Per bin, the ground round the grain minus its inner disc, on the image registered on the tracked grain
    (as ``grain_gone``)."""
    inner, back = rg < 0.7 * gr, (rg >= gr + 6.0) & (rg <= gr + 14.0)
    out = np.empty(len(img))
    for i, (im, (dx, dy)) in enumerate(zip(img, ls)):
        w = cv2.warpAffine(im.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), im.shape[::-1],
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[i] = float(np.median(w[back])) - float(w[inner].mean())
    return out


def gone_bins(contrast: np.ndarray, frac: float = 0.5, run: int = 5, ref_bins: int = 10,
              min_ref: float = 5.0, testable: np.ndarray | None = None) -> np.ndarray:
    """Bins where the grain is not at its tracked position: its disc contrast is below ``frac`` of its early
    level (median of the first ``ref_bins``) for a run of ``run`` bins or more. A return shorter than ``run`` bins
    after such a run (up to the next run or the end of the movie) does not count. A grain without early contrast
    (under ``min_ref`` grey levels: a smear, not a grain) is never called gone, nor is it in bins where its disc and
    ground ring are not wholly inside the movie frame (``testable`` False: the contrast test cannot tell there)."""
    c = np.asarray(contrast, float)
    ref = float(np.median(c[:ref_bins]))
    out = np.zeros(len(c), bool)
    if ref < min_ref:
        return out
    low = c < frac * ref
    if testable is not None:
        low &= np.asarray(testable, bool)
    i = 0
    while i < len(c):
        if low[i]:
            j = i
            while j < len(c) and low[j]:
                j += 1
            if j - i >= run:
                out[i:j] = True
            i = j
        else:
            i += 1
    # short returns after a gone run are not returns
    i = 0
    while i < len(c):
        if out[i]:
            j = i
            while j < len(c) and out[j]:
                j += 1
            k = j
            while k < len(c) and not out[k]:
                k += 1
            if j < len(c) and k - j < run:
                out[j:k] = True
            i = k if k > j else j
        else:
            i += 1
    return out


def _hp(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    img = img.astype(np.float32)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def reacquire(img: np.ndarray, ls: np.ndarray, contrast: np.ndarray, gone: np.ndarray, centre: float, gr: float,
              rg: np.ndarray, search: float = 60.0, ncc_min: float = 0.6, frac: float = 0.5, pad: float = 12.0,
              ref_bins: int = 10, max_search: float = 90.0
              ) -> tuple[np.ndarray, np.ndarray]:
    """Follow a grain that ``local_shifts`` lost (it jumped or was dragged beyond its window).

    In each gone bin, the grain is looked for within ``search`` px of the last position where it was seen (2 px more
    per bin since, up to ``max_search``): normalised cross-correlation of its own look in the last good bins (disc
    and rim, high-passed), at the best match whose disc contrast is back to ``frac`` of its early level and whose
    NCC reaches ``ncc_min``. From there it is followed bin by bin (the same template, +-``pad``/2 px per bin) until
    it is lost again. Returns the new shifts and gone mask (bins where the grain was not re-found stay gone)."""
    T, H, W = img.shape
    c = int(round(centre))
    rt = int(np.ceil(gr + 3))
    yy, xx = np.mgrid[-rt:rt + 1, -rt:rt + 1]
    mask = (np.hypot(xx, yy) <= gr + 3).astype(np.float32)
    ref = float(np.median(contrast[:ref_bins]))
    inner, back = rg < 0.7 * gr, (rg >= gr + 6.0) & (rg <= gr + 14.0)
    ls, gone = ls.copy(), gone.copy()
    hp = None

    def template(t_end):
        good = [t for t in range(t_end) if not gone[t]][-3:]
        if not good:
            return None
        acc = []
        for t in good:
            m = np.float32([[1, 0, -ls[t, 0]], [0, 1, -ls[t, 1]]])
            acc.append(cv2.warpAffine(hp[t], m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        return np.ascontiguousarray(np.mean(acc, axis=0)[c - rt:c + rt + 1, c - rt:c + rt + 1])

    def contrast_at(t, pos):
        m = np.float32([[1, 0, -pos[0]], [0, 1, -pos[1]]])
        w = cv2.warpAffine(img[t].astype(np.float32), m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return float(np.median(w[back])) - float(w[inner].mean())

    def match(t, tmpl, pos, s):
        ox, oy = int(round(pos[0])), int(round(pos[1]))
        s = int(s)
        x0, x1, y0, y1 = c + ox - rt - s, c + ox + rt + s + 1, c + oy - rt - s, c + oy + rt + s + 1
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1c - x0c < 2 * rt + 2 or y1c - y0c < 2 * rt + 2:
            return []
        res = cv2.matchTemplate(np.ascontiguousarray(hp[t][y0c:y1c, x0c:x1c]), tmpl, cv2.TM_CCORR_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=-1.0)
        # local maxima, best first
        dil = cv2.dilate(res, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero((res >= dil - 1e-6) & (res >= ncc_min))
        cand = [(float(res[y, x]), np.array([x0c - c + rt + x, y0c - c + rt + y], float)) for y, x in zip(ys, xs)]
        return sorted(cand, key=lambda q: -q[0])

    t = 0
    while t < T:
        if not gone[t]:
            t += 1
            continue
        if hp is None:
            hp = [_hp(im) for im in img]
        tmpl = template(t)
        good = np.nonzero(~gone[:t])[0]
        if tmpl is None or not len(good):
            t += 1
            continue
        last = ls[good[-1]].copy()  # the last position where the grain was seen
        radius = min(max_search, search + 2.0 * (t - good[-1] - 1))  # it may have moved on since
        found = None
        for q, pos in match(t, tmpl, last, radius):
            if contrast_at(t, pos) >= frac * ref:
                found = pos
                break
        if found is None:
            t += 1
            continue
        pos = found
        u = t
        while u < T:  # follow from the re-found position until it is lost again
            if u > t:
                cands = match(u, tmpl, pos, pad / 2)
                cands = [p for q, p in cands[:3] if contrast_at(u, p) >= frac * ref]
                if not cands:
                    break
                pos = cands[0]
            ls[u], gone[u] = pos, False
            u += 1
        t = u
    return ls, gone


# ----------------------------------------------------------------------------- abrupt changes (change 4)
def frame_change(R_img: Renderer, stride: int = 4, margin: int = 16) -> np.ndarray:
    """Field-wide change from the previous bin (registered frames, median + mean |difference|)."""
    nb = R_img.n_bins
    H, W = R_img.height, R_img.width
    prev, d = None, np.zeros(nb)
    for b in range(nb):
        dx, dy = R_img.shifts[b]
        f = cv2.warpAffine(np.asarray(R_img.bins[b], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[margin:-margin:stride, margin:-margin:stride]
        if prev is not None:
            a = np.abs(f - prev)
            d[b] = float(np.median(a)) + float(np.mean(a))
        prev = f
    return d


def abrupt_bins(d: np.ndarray, factor: float = 2.5, local: float = 2.0, window: int = 10) -> list[int]:
    """Bins whose field-wide change from the previous bin is ``factor`` x the running median around it and
    ``local`` x the change of the two bins before (a jump, not a busy stretch)."""
    out = []
    for b in range(2, len(d)):
        lo, hi = max(1, b - window), min(len(d), b + window + 1)
        med = float(np.median(d[lo:hi]))
        if med > 0 and d[b] >= factor * med and d[b] >= local * max(d[b - 1], d[b - 2]):
            out.append(b)
    return out


# ----------------------------------------------------------------------------- ghost discs (change 3)
def _disc_sharpness(ref: np.ndarray, x: float, y: float, r: float) -> tuple[float, float]:
    """(rim slope, focus): the steepest radial step near the rim (grey levels per px, median over angles) and the
    RMS of the Laplacian over the disc and rim."""
    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rads = np.arange(max(1.0, r - 10), r + 12.01, 0.5)
    mx = (x + rads[:, None] * np.cos(phis)[None]).astype(np.float32)
    my = (y + rads[:, None] * np.sin(phis)[None]).astype(np.float32)
    prof = cv2.remap(ref.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(np.diff(prof, axis=0)) / 0.5
    near = (rads[:-1] >= r - 4) & (rads[:-1] <= r + 4)
    slope = float(np.median(d[near].max(axis=0)))
    h, w = ref.shape
    y0, y1 = max(0, int(y - r - 4)), min(h, int(y + r + 5))
    x0, x1 = max(0, int(x - r - 4)), min(w, int(x + r + 5))
    if y1 - y0 < 3 or x1 - x0 < 3:
        return slope, 0.0
    sub = cv2.GaussianBlur(ref[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4].astype(np.float32), (0, 0), 1.0)
    lap = cv2.Laplacian(sub, cv2.CV_32F)[y0 - max(0, y0 - 4):, x0 - max(0, x0 - 4):][:y1 - y0, :x1 - x0]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = np.hypot(xx - x, yy - y) <= r + 3
    return slope, float(np.sqrt(np.mean(lap[m] ** 2))) if m.any() else 0.0


def census_ghosts(R_img: Renderer, census: list[dict], starts: tuple = (0, 4, 8, 12, 16, 20),
                  slope_max: float = 0.3, focus_max: float = 0.65) -> dict[str, tuple[float, float]]:
    """Census discs that are not grains: soft and faint in every early window. In each window (mean of 3 bins from
    ``ref_start`` + s) the disc's rim slope and focus are taken relative to the median of the included discs (not
    ``excluded``) on the first window; in later windows the disc is first re-fitted within 8 px (``grains.rim_fit``:
    grains still landing). A disc whose sharpest window stays below ``slope_max`` and ``focus_max`` is out of focus
    (a ghost) or a smear left by a grain that was still landing; an in-focus grain has a sharp dark rim (on the
    sample movie the two ghost discs read at most 0.13 and 0.23 of the median slope; the softest grains' medians over
    the windows are 0.54 and 0.59). The first window is at the census position by construction, so a grain that
    moves beyond the re-fit's reach early on is still judged sharp there. Every disc given is judged, excluded ones
    too (an annotator's "out of focus" disc must not block its neighbours' tubes either). Returns {id: (slope,
    focus)} of the discs judged not grains."""
    from sparsetrack.cli import registered_mean
    rs, nb = R_img.ref_start, R_img.n_bins
    wins = [registered_mean(R_img.bins, R_img.meta["shifts"], [rs + s, rs + s + 1, rs + s + 2])
            for s in starts if rs + s + 2 < nb]
    included = np.array([not g.get("excluded") for g in census], bool)
    if not wins or included.sum() < 5:
        return {}
    feats = np.zeros((len(census), len(wins), 2))
    for j, w in enumerate(wins):
        for i, g in enumerate(census):
            dx, dy, _ = census_mod.rim_fit(w, g["x"], g["y"], g["r"]) if j else (0.0, 0.0, 0.0)
            feats[i, j] = _disc_sharpness(w, g["x"] + dx, g["y"] + dy, g["r"])
    med = np.median(feats[included, 0, :], axis=0)
    rel = np.max(feats / np.maximum(med[None, None, :], 1e-6), axis=1)
    return {g["id"]: (round(float(rel[i, 0]), 3), round(float(rel[i, 1]), 3)) for i, g in enumerate(census)
            if rel[i, 0] < slope_max and rel[i, 1] < focus_max}


# ----------------------------------------------------------------------------- pass-by rivals (change 6)
def passby(comp: np.ndarray, ring: np.ndarray, ocx: float, ocy: float, radius: float = 8.0,
           max_deg: float = 60.0) -> bool:
    """True when ``comp`` only passes another grain's ring tangentially: at every contact with the ring, the
    region's main axis nearby makes more than ``max_deg`` with the direction to that grain's centre (a tube attached
    to the grain leaves its rim roughly radially; one growing past it runs along the rim). A contact without a
    clear direction (a blob) counts as attached."""
    touch = (comp & ring).astype(np.uint8)
    n, lab = cv2.connectedComponents(touch, connectivity=8)
    if n <= 1:
        return False
    h, w = comp.shape
    for k in range(1, n):
        cy, cx = np.nonzero(lab == k)
        mx, my = float(cx.mean()), float(cy.mean())
        y0, y1 = max(0, int(my - radius)), min(h, int(my + radius) + 1)
        x0, x1 = max(0, int(mx - radius)), min(w, int(mx + radius) + 1)
        sy, sx = np.nonzero(comp[y0:y1, x0:x1])
        sx, sy = sx + x0, sy + y0
        keep = np.hypot(sx - mx, sy - my) <= radius
        sx, sy = sx[keep], sy[keep]
        if len(sx) < 6:
            return False
        evals, evecs = np.linalg.eigh(np.cov(np.stack([sx - sx.mean(), sy - sy.mean()])))
        if evals[1] < 2.0 * max(evals[0], 1e-6):  # no clear direction
            return False
        radial = np.array([mx - ocx, my - ocy])
        radial /= max(float(np.hypot(*radial)), 1e-9)
        if np.degrees(np.arccos(min(1.0, abs(float(evecs[:, 1] @ radial))))) <= max_deg:
            return False  # this contact leaves the ring radially: attached there
    return True


# ----------------------------------------------------------------------------- the decoder
def smooth_length(pts: np.ndarray, eps: float = 1.0) -> float:
    """A pixel path's length as a smooth curve: Douglas-Peucker to ``eps`` px, then the polyline's length.
    Through pixel centres (diagonal steps sqrt 2) a straight tube reads up to 8% long, 5% on average,
    where it runs between the axes and the diagonals; a traced polyline does not."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) == 2 else 0.0
    keep, stack = {0, len(pts) - 1}, [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        t = pts[b] - pts[a]
        dev = np.abs((pts[a + 1:b, 0] - pts[a, 0]) * t[1] - (pts[a + 1:b, 1] - pts[a, 1]) * t[0]) / (np.hypot(*t) + 1e-9)
        k = a + 1 + int(np.argmax(dev))
        if dev[k - a - 1] > eps:
            keep.add(k)
            stack += [(a, k), (k, b)]
    q = pts[sorted(keep)]
    return float(np.sum(np.hypot(*np.diff(q, axis=0).T)))


def grain_gone(img: np.ndarray, ls: np.ndarray, centre: float, gr: float, rg: np.ndarray, run: int = 5) -> int | None:
    """First bin from which the grain is no longer where it was, for ``run`` bins or more: its inner disc
    (registered on the grain) has lost more than half its early contrast against the ground round it (the
    contrast, so that the whole movie brightening or dimming does not count)."""
    inner, back = rg < 0.7 * gr, (rg >= gr + 6.0) & (rg <= gr + 14.0)
    disc = np.empty(len(img))
    ground = np.empty(len(img))
    for i, (im, (dx, dy)) in enumerate(zip(img, ls)):
        w = cv2.warpAffine(im.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), im.shape[::-1],
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        disc[i], ground[i] = float(w[inner].mean()), float(np.median(w[back]))
    c0 = float((ground[:3] - disc[:3]).mean())
    if abs(c0) < 1e-6:
        return None
    away = (ground - disc) / c0 < 0.5
    for i in range(len(away) - run + 1):
        if away[i:i + run].all():
            return i
    return None


# ----------------------------------------------------------------------------- tip-growth continuity (change 7)
_NEVER = np.float32(1e9)  # a pixel never persistently above the threshold


def _arclen(pts: np.ndarray) -> np.ndarray:
    """Cumulative length along an (n, 2) point chain."""
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))] if len(pts) > 1 else np.zeros(len(pts))


def _rotate_about(pts: np.ndarray, pivot: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    d = pts - pivot
    return pivot + np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]


def _chord(P: np.ndarray, arc: np.ndarray, s0: float, s1: float) -> np.ndarray | None:
    """Unit chord of chain ``P`` from arclength ``s0`` to ``s1`` (None where that is not on the chain)."""
    if s0 < 0 or s1 > arc[-1] or s1 - s0 < 3.0:
        return None
    v = P[int(np.searchsorted(arc, s1))] - P[int(np.searchsorted(arc, s0))]
    n = float(np.hypot(*v))
    return v / n if n > 1e-6 else None


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between unit vectors."""
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _turned(u: np.ndarray, earlier: np.ndarray | None) -> np.ndarray:
    """``u`` turned on by the angle from ``earlier`` to ``u`` (a chain turning steadily keeps turning)."""
    if earlier is None:
        return u
    return _rotate_about(u[None, :], np.zeros(2), float(np.degrees(np.arctan2(
        earlier[0] * u[1] - earlier[1] * u[0], earlier @ u))))[0]


def fill_small_holes(region: np.ndarray, max_r: float = 4.0) -> np.ndarray:
    """``region`` with its enclosed holes filled where they are thin (inscribed radius <= ``max_r`` px): the hollow
    middle of a wide, double-walled tube, not the eye of a coil."""
    from scipy import ndimage
    inv = (~region).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    if n <= 2:
        return region
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ids = np.arange(1, n)
    rmax = np.asarray(ndimage.maximum(cv2.distanceTransform(inv, cv2.DIST_L2, 3), lab, ids))
    fill = ids[(rmax <= max_r) & ~np.isin(ids, border)]
    return region | np.isin(lab, fill) if len(fill) else region


def drawn_path(region: np.ndarray, path: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """What a review picture draws for a bin that continuity re-read or held: the path it measured along ((x, y) crop
    pixels) and the part of the region within the tube's half-width of it, so a foreign tube the reading left out is
    not outlined; a held bin (``path`` None) draws nothing."""
    if path is None or len(path) < 2:
        return None, None
    axis = np.zeros(region.shape, np.uint8)
    cv2.polylines(axis, [np.round(np.asarray(path, float)).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    axis = axis.astype(bool)
    on = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)[axis & region]
    half = float(np.percentile(on, 90)) if len(on) else 2.0
    near = cv2.distanceTransform((~axis).astype(np.uint8), cv2.DIST_L2, 3) <= half + 1.0
    return region & near, axis


class TipContinuity:
    """Tip growth for one grain's per-bin readings (``reach_grain(continuity="path")``): a reading longer than the
    tube's accepted path stands only if it continues that path at the tip; otherwise the bin is read along the path.

    A tube grows only at its tip, so in the grain's registered frame its medial axis at bin t is the axis of an earlier
    bin, swayed or turned a little, extended at the apex. ``A`` is the accepted path (rim exit -> apex); it is first
    taken from a farthest-point path that grew steadily from a short one (``start_px``..``init_max``, the same exit
    each bin, by ``init_grow`` px to ``min_len`` px, followed ``k_init`` bins), so a tube first read long is never
    locked on. Until then, and whenever a bin's default (farthest-point) reading is at most ``adv_px`` px (or 5%)
    longer than ``A``'s, the default reading stands. A longer one stands only if its path continues ``A``
    (``_continues``): from ``A``'s exit, within ``lat`` px of ``A`` (plus ``lat_s`` px per px along it: sway; at
    least the tube's half-width) for all but ``off_frac`` of ``A``'s length, then on beyond ``A``'s apex by at most
    ``vmax`` px per bin since the apex last advanced plus ``tol_ext``, without turning off by more than ``max_turn``
    degrees beyond what its own turning predicts (a tube it meets branches off at once; a coil turns steadily), and
    through pixels that were not tube before ``A``'s newest part was (first above the threshold for ``birth_k`` bins
    at most ``slack`` bins before the latest first-bin of ``A``'s last ``back_px``): a tube lying there before this
    tube's tip got there is not this tube. Its path then becomes ``A``.

    Otherwise the bin is read in two stages on the medial axis, thin holes filled first (``fill_small_holes``), with
    ``A`` swung about its exit (up to ``swing_deg`` per bin) or turned about the grain centre (``align_deg``) as best
    lays it on the region (``_along``): (1) from the skeleton pixels at ``A``'s start, along its corridor: a branch
    leaving ``A`` mid-way (a tube crossing or touching it, a neighbour's tube at the rim) is not read; (2) if the region
    covers ``A`` to within ``cov_tol`` px of its apex, on beyond the apex through the same tests, ending on a new pixel
    (older ones near the apex may be crossed: a tube it grows over); the apex advances with it. A bin whose region does
    not cover ``A`` is not a reading (NaN, held like a gone bin).

    An early error is not kept for good: after ``k_switch`` rejected bins whose paths are steady (the same exit, each
    within tip growth of the last; bins whose default reading stands may come between them, as when fragmentary
    evidence reads the tube short in some bins), such a path replaces ``A`` when ``A`` was not covered all along (a
    tube turned beyond the alignment) or when the path grew by ``init_grow`` px or more from a short one along another
    way (``A`` was a stub at the rim while the tube grew elsewhere); a steady path that does not grow (a tube crossing
    this one) never does."""

    def __init__(self, shape: tuple, centre: float, gr: float, rg: np.ndarray, own_ring: np.ndarray, vmax: float,
                 lat: float = 4.0, lat_s: float = 0.08, lat_bin: float = 0.5, hw_k: float = 1.0, off_frac: float = 0.25,
                 exit_tol: float = 8.0, start_px: float = 3.0, init_max: float | None = None, min_len: float = 6.0,
                 init_grow: float = 4.0, k_init: int = 2, adv_px: float = 6.0, tol_ext: float = 4.0,
                 cov_tol: float = 8.0, max_turn: float = 70.0, birth: bool = True, birth_k: int = 2, slack: int = 3,
                 back_px: float = 8.0, k_switch: int = 4, align_deg: float = 3.0, align_cap: float = 30.0,
                 swing_deg: float = 12.0, swing_cap: float = 90.0, fill_r: float = 4.0):
        self.shape, self.centre, self.gr, self.rg, self.own_ring, self.vmax = shape, centre, gr, rg, own_ring, vmax
        self.lat0, self.lat_s, self.lat_bin, self.hw_k, self.off_frac = lat, lat_s, lat_bin, hw_k, off_frac
        self.exit_tol, self.start_px, self.min_len = exit_tol, start_px, min_len
        self.init_grow, self.k_init = init_grow, k_init
        self.init_max = max(12.0, 2.5 * vmax) if init_max is None else init_max
        self.adv_px, self.tol_ext, self.cov_tol, self.max_turn = adv_px, tol_ext, cov_tol, max_turn
        self.use_birth, self.birth_k, self.slack, self.back_px = birth, birth_k, slack, back_px
        self.k_switch, self.align_deg, self.align_cap = k_switch, align_deg, align_cap
        self.swing_deg, self.swing_cap, self.fill_r = swing_deg, swing_cap, fill_r
        self.birth = np.full(shape, _NEVER, np.float32)  # per pixel: the first bin of its first persistent run
        self.run = np.zeros(shape, np.int16)
        self.A: np.ndarray | None = None  # the accepted path, (x, y) crop pixels from the exit to the apex
        self.L_A = 0.0  # the reading it was accepted with
        self.t_apex = self.t_geom = 0  # bins its apex last advanced, its geometry was last laid on a region
        self.C: np.ndarray | None = None  # before A: the farthest-point path while it grows steadily
        self.C_t, self.C_run, self.C_L0 = 0, 0, 0.0
        self.J: np.ndarray | None = None  # rejected paths while steady: a candidate to replace A
        self.J_t, self.J_run, self.J_L0, self.J_lost = 0, 0, 0.0, 0
        self.reread: list[int] = []
        self.held: list[int] = []
        self.switched: list[int] = []

    def observe(self, i: int, above: np.ndarray) -> None:
        """Bin ``i``'s thresholded evidence (registered on the grain): first bins of persistent tube per pixel."""
        self.run = np.where(above, self.run + 1, 0).astype(np.int16)
        new = (self.run == self.birth_k) & (self.birth == _NEVER)
        self.birth[new] = i - self.birth_k + 1

    def read(self, i: int, sk: np.ndarray, region: np.ndarray, end: float, raw: float,
             route: np.ndarray | None) -> tuple[float, np.ndarray | None]:
        """Bin ``i``'s reading and path, given its region, medial axis and default reading ``raw`` along ``route``
        ((x, y) crop pixels, rim to the farthest point)."""
        if self.A is None:
            self._grow(i, raw, route)
            if self.C is not None and self.C_run >= self.k_init:
                L = float(_arclen(self.C)[-1])
                if L >= self.min_len and L >= self.C_L0 + self.init_grow:
                    self.A, self.L_A, self.t_apex, self.t_geom = self._trim(self.C), raw, i, i
            return raw, route
        if raw <= self.L_A + max(self.adv_px, 0.05 * self.L_A):
            self._refresh(i, route)  # (rejected paths stay followed: a flickering region reads short in between)
            return raw, route
        if self.fill_r > 0:
            filled = fill_small_holes(region, self.fill_r)
            if filled.sum() > region.sum():
                region = filled
                sk = skeletonize(region)
                sk = sk if sk.sum() >= 2 else region
        A = self._align(self.A, region, i)
        arcA = _arclen(A)
        lat = self._lat(i, A, region)
        if self._continues(i, route, A, arcA, lat):  # it grew on at the tip (or caught up): its path is A
            self.A, self.L_A, self.t_apex, self.t_geom = self._trim(route), raw, i, i
            self.J, self.J_run = None, 0
            return raw, route
        self._track(i, route)
        reading, path, covered = self._along(i, sk, A, arcA, lat, end)
        if self._switch(i, raw, route, covered):
            self.switched.append(i)
            return raw, route
        if covered:
            self.reread.append(i)
            return reading, path
        self.held.append(i)
        return np.nan, None

    # -- helpers
    def _refresh(self, i: int, route: np.ndarray | None) -> None:
        """A standing reading whose path lies along ``A`` lays ``A`` on the region (the alignment's search span counts
        from there); ``A`` takes on its geometry when about as long (sway), never a longer one: ``A`` grows only by
        passing ``_continues`` or stage 2."""
        if route is None or len(route) < 2 or not self._lies_along(route, self.A):
            return
        LA, L = float(_arclen(self.A)[-1]), float(_arclen(route)[-1])
        if LA - 1.0 <= L <= LA + 1.0:
            self.A = self._trim(route)
        self.t_geom = i

    def _lies_along(self, route: np.ndarray, A: np.ndarray, lat: float | None = None, upto: int | None = None) -> bool:
        """Does ``route`` (its first ``upto`` points) lie within the corridor of ``A`` for all but ``off_frac`` of
        ``A``'s length (the base, where the axis meets the rim, is free)?"""
        arcA = _arclen(A)
        k = len(route) if upto is None else upto
        d, idx = cKDTree(A).query(route[:k])
        rim = np.hypot(*(route[:k] - self.centre).T) <= self.gr + 6.0
        seg = np.r_[0.0, np.hypot(*np.diff(route[:k], axis=0).T)]
        off = (d > (self.lat0 if lat is None else lat) + self.lat_s * arcA[idx]) & ~rim
        return float(np.sum(seg[off])) <= self.off_frac * max(float(arcA[-1]), 1.0)

    def _lat(self, i: int, A: np.ndarray, region: np.ndarray) -> float:
        lat = self.lat0 + min(self.lat_bin * max(i - self.t_geom, 0), 12.0)
        if self.hw_k > 0:  # a wide tube's medial axis wanders across its width
            q = np.round(A).astype(int)
            q[:, 0] = np.clip(q[:, 0], 0, self.shape[1] - 1)
            q[:, 1] = np.clip(q[:, 1], 0, self.shape[0] - 1)
            hw = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)[q[:, 1], q[:, 0]]
            hw = hw[hw > 0]
            if len(hw):
                lat = max(lat, min(self.hw_k * float(np.median(hw)), 16.0))
        return lat

    def _align(self, A: np.ndarray, region: np.ndarray, i: int) -> np.ndarray:
        """``A`` swung about its exit (a tube swaying on its base) or turned about the grain centre (the grain
        turning), whichever lays it best on the region, within ``swing_deg`` / ``align_deg`` per bin since ``A`` was
        last laid on a region."""
        if len(A) < 3 or (self.align_deg <= 0 and self.swing_deg <= 0):
            return A
        dt = max(i - self.t_geom, 1)
        d = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
        best, best_s = A, None
        for pivot, per, cap in ((A[0], self.swing_deg, self.swing_cap),
                                (np.array([self.centre, self.centre]), self.align_deg, self.align_cap)):
            span = np.floor(min(per * dt, cap))
            w = np.hypot(*(A - pivot).T)
            for deg in np.r_[0.0, np.arange(-span, span + 1e-9, 1.0)]:
                P = _rotate_about(A, pivot, deg) if deg else A
                q = np.round(P).astype(int)
                inb = (q[:, 0] >= 0) & (q[:, 0] < self.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < self.shape[0])
                v = np.full(len(q), 8.0)
                v[inb] = np.minimum(d[q[inb, 1], q[inb, 0]], 8.0)
                sc = float(np.sum(w * v) / max(float(w.sum()), 1e-9)) + 0.02 * abs(deg)
                if best_s is None or sc < best_s - 1e-9:
                    best, best_s = P, sc
        return best

    @staticmethod
    def _trim(A: np.ndarray) -> np.ndarray:
        """``A`` without a hook at its end (the medial axis of a blunt tip forks): cut where its last 15 px are
        farthest from the point 15 px back."""
        arc = _arclen(A)
        if len(A) < 3 or arc[-1] < 10.0:
            return A
        j0 = int(np.searchsorted(arc, arc[-1] - 15.0))
        k = j0 + int(np.argmax(np.hypot(*(A[j0:] - A[j0]).T)))
        return A[:k + 1] if k >= 1 else A

    def _b_ref(self, A: np.ndarray, arcA: np.ndarray) -> float:
        """The first bin of this tube's newest part: the latest first-bin among ``A``'s last ``back_px``."""
        q = np.round(A[arcA >= arcA[-1] - self.back_px]).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, self.shape[1] - 1)
        q[:, 1] = np.clip(q[:, 1], 0, self.shape[0] - 1)
        b = cv2.erode(self.birth, np.ones((3, 3), np.uint8))[q[:, 1], q[:, 0]]
        b = b[b < _NEVER]
        return float(b.max()) if len(b) else float(self.t_apex)

    def _old(self, A: np.ndarray, arcA: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Pixels that were tube before this tube's newest part was (first-bins, min over 3 x 3)."""
        if not self.use_birth:
            return np.zeros(len(xs), bool)
        return cv2.erode(self.birth, np.ones((3, 3), np.uint8))[ys, xs] < self._b_ref(A, arcA) - self.slack

    def _forward(self, route: np.ndarray, k_apex: int) -> bool:
        """Does the path go on past ``A``'s apex (its point ``k_apex``) as it came: its direction after (chord 3..17 px)
        within ``max_turn`` of its direction before (20..3 px), or of that direction turned on as it was turning (a
        coil turns steadily; a meandering tube turns back)?"""
        arc = _arclen(route)
        sa = float(arc[k_apex])
        after = _chord(route, arc, sa + 3.0, min(sa + 17.0, float(arc[-1])))
        before = _chord(route, arc, sa - 20.0, sa - 3.0)
        if after is None or before is None or float(arc[-1]) - sa < 6.0:
            return True
        pred = _turned(before, _chord(route, arc, sa - 40.0, sa - 23.0))
        return min(_angle(after, before), _angle(after, pred)) <= self.max_turn

    def _continues(self, i: int, route: np.ndarray | None, A: np.ndarray, arcA: np.ndarray, lat: float) -> bool:
        """Does the default path continue ``A`` (see the class docstring)?"""
        if route is None or len(route) < 2 or float(np.hypot(*(route[0] - A[0]))) > 2.0 * self.exit_tol:
            return False
        LA = float(arcA[-1])
        d, idx = cKDTree(A).query(route)
        at = np.flatnonzero(arcA[idx] >= LA - self.cov_tol)
        k = int(at[0]) if len(at) else len(route)  # where it comes near A's apex
        if not self._lies_along(route, A, lat, k):
            return False  # it leaves A before A's apex: a branch
        if k >= len(route):
            return False  # longer than A, yet never near A's apex: it branched off
        arc = _arclen(route)
        ext = float(arc[-1] - arc[k]) - (LA - float(arcA[idx[k]]))
        if ext <= self.adv_px:
            return True
        if ext > self.vmax * max(i - self.t_apex, 1) + self.tol_ext:
            return False  # a sudden extension far beyond the apex
        k_apex = k + int(np.argmin(np.hypot(*(route[k:] - A[-1]).T)))
        if self.max_turn < 180 and not self._forward(route, k_apex):
            return False  # it turns off at the apex: a tube it met, not its own tip
        q = np.round(route[k:]).astype(int)
        out = (d[k:] > lat) & (np.hypot(*(route[k:] - A[-1]).T) > max(self.cov_tol, 2.0 * lat + 2.0))
        return int(np.sum(self._old(A, arcA, q[out, 0], q[out, 1]))) <= 2

    def _seeds(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Seed pixels of ``mask`` nearest the grain (as the default reading) and their offset from the rim."""
        if not mask.any():
            return mask, 0.0
        near = mask & self.own_ring if (mask & self.own_ring).any() else mask
        rmin = float(self.rg[near].min())
        return mask & (self.rg <= rmin + 0.5), max(0.0, rmin - self.gr)

    def _along(self, i: int, sk: np.ndarray, A: np.ndarray, arcA: np.ndarray, lat: float,
               end: float) -> tuple[float | None, np.ndarray | None, bool]:
        """Stages 1 and 2 on (aligned) ``A``: the reading along ``A`` and on beyond its apex, and whether the region
        covers ``A``; ``A`` follows (its apex advances with the extension)."""
        LA = float(arcA[-1])
        tree = cKDTree(A)
        ys, xs = np.nonzero(sk)
        if not len(ys):
            return None, None, False
        d, idx = tree.query(np.c_[xs, ys])
        inc = d <= lat + self.lat_s * arcA[idx]
        s1 = np.zeros_like(sk)
        s1[ys[inc], xs[inc]] = True
        first = inc & (arcA[idx] <= min(self.exit_tol, 0.5 * LA)) & (
            np.hypot(xs - A[0][0], ys - A[0][1]) <= 2.0 * self.exit_tol)
        start = np.zeros_like(sk)
        start[ys[first], xs[first]] = True
        seeds, offset = self._seeds(start)
        if not seeds.any():
            return None, None, False
        mcp = MCP_Geometric(np.where(s1, 1.0, np.inf))
        cum, _ = mcp.find_costs(list(zip(*np.nonzero(seeds))))
        ys, xs = np.nonzero(s1 & np.isfinite(cum))
        _, idx = tree.query(np.c_[xs, ys])
        sA = arcA[idx]
        top = sA >= sA.max() - 0.5
        j = np.flatnonzero(top)[int(np.argmin(cum[ys[top], xs[top]]))]
        q = (int(ys[j]), int(xs[j]))  # the reachable corridor pixel farthest along A
        covered = bool(sA.max() >= LA - self.cov_tol) and float(cum[q]) >= 0.5 * (float(sA.max()) - self.exit_tol)
        path = np.asarray(mcp.traceback(q), float)[:, ::-1]
        ext, tail = self._extend(i, sk, A, arcA, lat, q) if covered else (0.0, None)
        if tail is not None:
            path = np.vstack([path, tail[1:]])
        if covered:
            if tail is not None and _arclen(path)[-1] > LA + self.adv_px:
                self.A, self.L_A, self.t_apex = self._trim(path), max(0.0, float(cum[q]) + ext + offset + end), i
            else:
                self.A = A
            self.t_geom = i
        return max(0.0, float(cum[q]) + ext + offset + end), path, covered

    def _extend(self, i: int, sk: np.ndarray, A: np.ndarray, arcA: np.ndarray, lat: float,
                q: tuple) -> tuple[float, np.ndarray | None]:
        """Stage 2: from ``q`` on beyond ``A``'s apex (never back alongside ``A``, leaving forwards), ending on a new
        pixel; older ones near the apex may be crossed."""
        LA = float(arcA[-1])
        ys, xs = np.nonzero(sk)
        d, idx = cKDTree(A).query(np.c_[xs, ys])
        geo = (d > 2.0 * lat + 2.0) | (arcA[idx] >= LA - 2.0)  # beyond the apex, or well away from A
        u0 = _chord(A, arcA, LA - 20.0, LA - 3.0)
        if self.max_turn < 180 and u0 is not None:  # leave within max_turn of A's direction, or of it turned on
            u1 = _turned(u0, _chord(A, arcA, LA - 40.0, LA - 23.0))
            v = np.c_[xs, ys] - np.array([q[1], q[0]], float)
            r = np.hypot(*v.T)
            c = np.cos(np.radians(self.max_turn)) * r
            geo &= (r < 3.0) | (r > 17.0) | ((v @ u0) >= c) | ((v @ u1) >= c)
        new = ~self._old(A, arcA, xs, ys)
        r0 = max(self.cov_tol, 2.0 * lat + 2.0)
        near = (np.hypot(xs - A[-1][0], ys - A[-1][1]) <= r0) | (np.hypot(xs - q[1], ys - q[0]) <= r0)
        dom = np.zeros_like(sk)
        dom[ys[geo & (new | near)], xs[geo & (new | near)]] = True
        dom[q] = True
        end_ok = np.zeros_like(sk)
        end_ok[ys[geo & new], xs[geo & new]] = True
        mcp = MCP_Geometric(np.where(dom, 1.0, np.inf))
        cum, _ = mcp.find_costs([q])
        ok = end_ok & np.isfinite(cum) & (cum <= self.vmax * max(i - self.t_apex, 1) + self.tol_ext)
        if not ok.any():
            return 0.0, None
        far = np.unravel_index(int(np.argmax(np.where(ok, cum, -1.0))), cum.shape)
        if cum[far] <= 0:
            return 0.0, None
        tail = np.asarray(mcp.traceback(far), float)[:, ::-1]
        # a loop at the tip: the farthest pixel along it can lie on the way back; end the tail where it is farthest
        # from the apex, so A never hooks back on itself
        tail = tail[:int(np.argmax(np.hypot(*(tail - A[-1]).T))) + 1]
        return (float(_arclen(tail)[-1]), tail) if len(tail) >= 2 else (0.0, None)

    def _grow(self, i: int, raw: float, route: np.ndarray | None) -> None:
        """Before ``A``: follow the farthest-point path while it grows steadily from a short one (the same exit, each
        no longer than tip growth allows from the last); a long path appearing at once cannot start it."""
        if route is None or raw < self.start_px:
            return  # a miss neither ends nor extends it
        L = float(_arclen(route)[-1])
        if self.C is not None:
            LC = float(_arclen(self.C)[-1])
            if (float(np.hypot(*(route[0] - self.C[0]))) <= 1.5 * self.exit_tol
                    and L <= LC + self.vmax * (i - self.C_t) + self.tol_ext):
                if L >= LC - self.adv_px:
                    self.C, self.C_t = route, i
                self.C_run += 1
                return
            self.C = None
        if L <= self.init_max:
            self.C, self.C_t, self.C_run, self.C_L0 = route, i, 1, L

    def _track(self, i: int, route: np.ndarray | None) -> None:
        """Follow the rejected paths while they are steady: the same exit, each within tip growth of the last."""
        if route is None:
            return
        L = float(_arclen(route)[-1])
        if self.J is not None:
            LJ = float(_arclen(self.J)[-1])
            if (float(np.hypot(*(route[0] - self.J[0]))) <= 1.5 * self.exit_tol
                    and 0.85 * LJ - self.adv_px <= L <= LJ + self.vmax * (i - self.J_t) + self.tol_ext):
                self.J, self.J_t, self.J_run = route, i, self.J_run + 1
                return
        self.J, self.J_t, self.J_run, self.J_L0 = route, i, 1, L

    def _switch(self, i: int, raw: float, route: np.ndarray | None, covered: bool) -> bool:
        """After ``k_switch`` steady rejected paths, one replaces ``A`` if ``A`` was not covered all along, or if it grew
        from a short one along another way than ``A``."""
        if route is None or self.J is None:
            return False
        self.J_lost = self.J_lost + 1 if not covered and self.J_run > 1 else (0 if covered else 1)
        if self.J_run < self.k_switch:
            return False
        LA, LJ = float(_arclen(self.A)[-1]), float(_arclen(self.J)[-1])
        grew = (self.J_L0 <= max(self.init_max, LA + 2.0 * self.adv_px) and LJ >= self.J_L0 + self.init_grow
                and not self._lies_along(self.J, self.A))
        if self.J_lost >= self.k_switch or grew:
            self.A, self.L_A, self.t_apex, self.t_geom = self._trim(route), raw, i, i
            self.J, self.J_run, self.J_lost = None, 0, 0
            return True
        return False


def reach_grain(RP: Renderer, R_img: Renderer, meta: dict, grain: dict, others: list[dict], thr: float = 0.5,
                scale: float = 16.0, half: int = 150, vmax: float = 4.0, onset_px: float = 2.0,
                min_tube_px: float = 8.0, rim_band: float = 5.0, seed: str = "skeleton", end_px: float = 1.0,
                tip: str = "const", big: int | None = None, burst: bool = False, keep: tuple = (),
                paths: tuple = (), length: str = "path", edge_mask: str = "bin", edge_margin: float = 1.0,
                gone: str = "hold", track: str = "local", fit: str = "grown", fit_kw: dict | None = None,
                burst_run: int = 1, abrupt: tuple = (), owner: str = "geodesic", continuity: str = "off",
                cont_kw: dict | None = None, _ls=None) -> dict:
    """Length per bin, then a monotone fit. ``seed`` chooses how the bin's length is read:
    ``"skeleton"`` (the default, frozen on the development seed): geodesic length along the region's
    medial axis from its pixels nearest the grain centre, plus their distance from the rim, so a
    wide tube's half-width does not count; ``"nearest"``: the same over the whole region;
    ``"rim"`` (the first sketch): from every region pixel within 1.5 px of the rim, which reads
    ~2 px short. ``end_px`` is added to every non-zero reach: the skeleton stops short of the
    tube's end. With ``big`` set, a grain whose region reaches the crop's edge in any bin is read
    again with ``half = big`` (movies longer than the dev movie; ``pipeline.py`` sets 300).
    ``tip="dt"`` adds the region's half-width at the end of the axis instead: it helped with exact
    truth masks and hurt with learned evidence on fresh held-out movies, so it is off. With
    ``burst``, a reading that collapses for good (``burst_cut``) ends the growth fit there: a burst
    tube leaves nothing to trace (``pipeline.py`` sets it; its burst frame is not a measurement).
    ``keep`` lists bins (counted from the first analysed bin) whose registered image, region and
    medial axis are returned under ``_views`` for review pictures (``review.py``). ``paths`` lists
    bins whose reading is returned as a path under ``_paths``: from the grain's rim along the medial
    axis to its far end, in reference coordinates at that bin (the grain's own drift included, as
    the labelling tool's ``path_xy_ref``), for pre-filling review labels (``prefill.py``).
    ``length="smooth"`` measures the medial axis as a smooth curve (``smooth_length``) instead of a path
    through pixel centres, which reads up to 8% long where a tube runs between the axes and the
    diagonals. Unbiased, it still put fewer lengths in tolerance on synthetic development movies
    (1364 against 1368 of 1792; 67 against 80 of the 100 longest tubes): the staircase's excess
    brings some of the under-reads (a region stopping short of the tip) inside tolerance. So the
    default stays "path".

    Options for real footage (module docstring; ``edge_mask="movie", gone="flag", fit="l1"`` reproduce the
    earlier frozen decoder): ``edge_mask``
    ("movie" | "bin") and ``edge_margin`` (px); ``gone`` ("flag" | "hold"); ``track`` ("local" | "reacquire");
    ``fit`` ("l1" | "grown", ``fit_kw`` for ``grown_floor``); ``burst_run``; ``abrupt`` (analysed-bin indices just
    after an abrupt change); ``owner`` ("geodesic" | "passby"); ``continuity`` ("off" | "path", ``cont_kw`` for
    ``TipContinuity``; it reads along the medial axis, so it needs the default ``seed``, ``tip`` and ``length``: the
    bins it re-read along the tube's path and those it held are returned under ``continuity``). With ``gone="hold"``
    (or ``track="reacquire"``) the bins where the grain is gone are returned under ``gone_bins``.

    With learned evidence it is ahead of SparseTrack's decoder on synthetic movies: +84 lengths in
    tolerance on seven development movies, +12 on eight held-out and +44 on four untouched test
    movies. No label-free per-grain switch between the two decoders did better than this alone."""
    fpb, rs, nb = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0)), int(meta["n_bins"])
    gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
    centre = half - 0.5
    img = np.stack([R_img.crop(b, gx, gy, half) for b in range(rs, nb)])
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - centre, yy - centre)
    ls = A.local_shifts(img, centre, gr, 12.0, 3) if _ls is None else _ls
    absent = np.zeros(nb - rs, bool)
    abrupt_mask = np.zeros(nb - rs, bool)
    for b in abrupt:
        for d in range(0, 3):  # the jump bin and the two after it
            if 0 <= b + d < nb - rs:
                abrupt_mask[b + d] = True
    con = disc_contrast(img, ls, centre, gr, rg)  # also a diagnostic ("gone_bins") when not used
    at = R_img.shifts[rs:nb] + ls + [gx, gy]  # the grain's position in the movie frame, per bin
    inside = ((at[:, 0] >= gr + 14) & (at[:, 0] < R_img.width - gr - 14) &
              (at[:, 1] >= gr + 14) & (at[:, 1] < R_img.height - gr - 14))
    absent = gone_bins(con, testable=inside)
    if track == "reacquire" and absent.any():
        ls, absent = reacquire(img, ls, con, absent, centre, gr, rg)
    blocked = rg < gr - 1.0
    # pixels whose source leaves the movie in any bin are not evidence (as in SparseTrack): near an
    # edge, the padded border shows up as straight streaks that the network can take for tubes
    shifts = R_img.shifts[rs:nb] + ls
    ref_x, ref_y = gx - half + xx + 0.5, gy - half + yy + 0.5
    if edge_mask == "movie":
        blocked |= ~((ref_x + shifts[:, 0].min() >= 1) & (ref_x + shifts[:, 0].max() < R_img.width - 1) &
                     (ref_y + shifts[:, 1].min() >= 1) & (ref_y + shifts[:, 1].max() < R_img.height - 1))
    rings, ring_c = [], []
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 10 < ox < 2 * half + o["r"] + 10 and -o["r"] - 10 < oy < 2 * half + o["r"] + 10:
            d = np.hypot(xx - ox, yy - oy)
            blocked |= d < o["r"] + 1.0
            rings.append((d >= o["r"] + 1.0) & (d <= o["r"] + rim_band))
            ring_c.append((ox, oy))
    own_ring = (rg >= gr - 1.0) & (rg <= gr + rim_band)
    rim = (rg >= gr - 1.0) & (rg <= gr + 1.5)
    raw = np.zeros(nb - rs)
    cont = None
    if continuity == "path":
        if seed != "skeleton" or tip != "const" or length != "path":
            raise ValueError("continuity='path' reads along the medial axis: seed='skeleton', tip='const', length='path'")
        cont = TipContinuity((2 * half, 2 * half), centre, gr, rg, own_ring, vmax, **(cont_kw or {}))
    elif continuity != "off":
        raise ValueError(f"unknown continuity {continuity!r}")
    width = np.full(nb - rs, np.nan)  # region area per px of length: the tube's full width
    edge = False  # the tube reaches the crop's edge: read the grain again with a bigger crop
    views, routes = {}, {}
    for i, b in enumerate(range(rs, nb)):
        p = RP.crop(b, gx, gy, half) / scale
        dx, dy = ls[i]
        shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
        p = cv2.warpAffine(np.nan_to_num(p).astype(np.float32), shift,
                           (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if i in keep:
            views[i] = [cv2.warpAffine(img[i].astype(np.float32), shift, (2 * half, 2 * half),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), None, None]
        blk = blocked
        if edge_mask == "bin":
            blk = blocked | frame_outside(R_img.width, R_img.height, gx, gy, half, shifts[i], edge_margin)
        m = (p > thr) & ~blk
        if cont is not None:
            cont.observe(i, m)
        if not (m & own_ring).any():
            continue
        _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[m & own_ring])
        comp = np.isin(lab, ids[ids > 0])
        rivals = [(r, c) for r, c in zip(rings, ring_c) if (comp & r).any()]
        if owner == "passby":
            rivals = [(r, c) for r, c in rivals if not passby(comp, r, c[0], c[1])]
        if rivals:
            comp = A.geodesic_owner(comp, [own_ring] + [r for r, _ in rivals]) == 0
        k = 3 + int(np.ceil(np.abs(ls).max()))  # the registration shift leaves an empty band at the border
        edge = edge or bool(comp[:k].any() or comp[-k:].any() or comp[:, :k].any() or comp[:, -k:].any())
        region = comp
        if seed == "skeleton":  # along the medial axis: a wide tube's half-width does not count
            sk = skeletonize(comp)
            comp = sk if sk.sum() >= 2 else comp
        if i in views:
            views[i][1], views[i][2] = region, comp
        seeds = comp & rim if seed == "rim" else np.zeros_like(comp)
        offset = 0.0
        if not seeds.any():  # nearest pixels (or a faint base that starts beyond the rim)
            near = comp & own_ring if (comp & own_ring).any() else comp
            rmin = float(rg[near].min())
            seeds = comp & (rg <= rmin + (1.0 if seed == "rim" else 0.5))
            offset = max(0.0, rmin - gr)
        if not seeds.any():
            continue
        mcp = MCP_Geometric(np.where(comp, 1.0, np.inf))
        cum, _ = mcp.find_costs(list(zip(*np.nonzero(seeds))))
        ok = comp & np.isfinite(cum)
        if not ok.any():
            continue
        far = np.unravel_index(int(np.argmax(np.where(ok, cum, -1.0))), cum.shape)
        if i in paths or length == "smooth" or cont is not None:
            route = np.asarray(mcp.traceback(far), float)[:, ::-1]  # crop cols, rows
        if i in paths:  # the exit on the rim, then the medial axis to the far end (-> reference x, y)
            ey, ex = np.nonzero(region & own_ring)
            v = (np.array([ex.mean(), ey.mean()]) if len(ex) else route[0]) - centre
            rim_pt = centre + v * gr / max(float(np.hypot(*v)), 1e-9)
            j = int(np.argmin(np.hypot(*(route - rim_pt).T)))
            routes[i] = np.vstack([rim_pt, route[j:]]) + [gx - half + dx + 0.5, gy - half + dy + 0.5]
        end = end_px
        if tip == "dt":  # the medial axis stops about a half-width short of the tube's end
            end += float(cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)[far])
        along = smooth_length(route) if length == "smooth" else float(cum[far])
        raw[i] = max(0.0, along + offset + end)  # a negative end_px must not make lengths negative
        width[i] = float(region.sum()) / max(float(cum[far]) + 1.0, 1.0)
        if cont is not None and not absent[i]:  # tip growth: a reading that jumps off the tube's path is re-read
            raw[i], cpath = cont.read(i, comp, region, end, raw[i], route)
            if i in views and cpath is not route:  # the picture shows what was measured, not the region left out
                views[i][1], views[i][2] = drawn_path(region, cpath)
            if i in routes and cpath is None:  # held: the farthest point's path is not the grain's
                del routes[i]
            elif i in routes and cpath is not route:  # its exit: where the path itself leaves the rim
                v = cpath[0] - centre
                rim_pt = centre + v * gr / max(float(np.hypot(*v)), 1e-9)
                routes[i] = np.vstack([rim_pt, cpath]) + [gx - half + dx + 0.5, gy - half + dy + 0.5]
    if edge and big and half < big:
        res = reach_grain(RP, R_img, meta, grain, others, thr=thr, scale=scale, half=big, vmax=vmax,
                          onset_px=onset_px, min_tube_px=min_tube_px, rim_band=rim_band, seed=seed, end_px=end_px,
                          tip=tip, big=big, burst=burst, keep=keep, paths=paths, length=length, edge_mask=edge_mask,
                          edge_margin=edge_margin, gone=gone, track=track, fit=fit, fit_kw=fit_kw,
                          burst_run=burst_run, abrupt=abrupt, owner=owner, continuity=continuity, cont_kw=cont_kw)
        res["flags"].append(f"crop_grown:{big}")
        return res
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    readings = np.where(absent, np.nan, raw) if gone == "hold" else raw
    fit_len, cut = fit_lengths(readings, vmax, burst, min_tube_px=min_tube_px, fit=fit, burst_run=burst_run,
                               skip_burst=abrupt_mask if len(abrupt) else None, **(fit_kw or {}))
    fit_ = fit_len
    on = np.nonzero(fit_ >= onset_px)[0]
    flags = []
    if fit_[-1] < min_tube_px or not len(on):  # (never reaching a calibrated onset_px > min_tube_px: no onset)
        status, onset, fit_ = "no_emergence_by_end", None, np.zeros_like(fit_)
    elif on[0] == 0:
        status, onset = "emerged_at_start", frames[0]
    else:
        status, onset = "emerged_within", frames[int(on[0])]
    interval = None if onset is None or on[0] == 0 else [frames[int(on[0]) - 1], onset]
    if cut is not None and status != "no_emergence_by_end":
        flags.append(f"burst_after:{frames[cut - 1]}")
    # review hints (the readings are unchanged): the grain has left its place, so what is read there is
    # not its tube; or the reading keeps jumping off the fit, as when a crossing tube takes over the region
    if gone == "hold" or track == "reacquire":
        edges = np.flatnonzero(np.diff(np.r_[0, absent.astype(np.int8), 0]))
        for a, z in zip(edges[::2], edges[1::2]):  # runs [a, z) of bins where the grain is not at its place
            flags.append(f"no_grain_after:{frames[a]}" if z == absent.size else f"gone:{frames[a]}-{frames[z - 1]}")
    else:
        g_ = grain_gone(img, ls, centre, gr, rg)
        if g_ is not None:
            flags.append(f"no_grain_after:{frames[g_]}")
    if status != "no_emergence_by_end":
        live = ((fit_ >= min_tube_px / 2) & (np.arange(raw.size) < (cut if cut is not None else raw.size))
                & np.isfinite(raw))
        off = live & (np.abs(raw - fit_) > np.maximum(10.0, 0.3 * fit_))
        if live.sum() >= 5 and off.sum() >= 0.3 * live.sum():
            flags.append(f"unsteady:{int(off.sum())}/{int(live.sum())}")
    return {"id": grain["id"], "x": gx, "y": gy, "r": gr, "status": status, "onset_frame": onset,
            "onset_interval": interval, "final_length_px": round(float(fit_[-1]), 2),
            "length": {"frames": frames, "px": [round(float(v), 2) for v in fit_]},
            "raw_reach_px": [round(float(v), 2) if np.isfinite(v) else None for v in raw], "flags": flags,
            "burst_frame": frames[cut] if cut is not None and status != "no_emergence_by_end" else None,
            "width_px": (round(float(np.nanmedian(width[raw >= min_tube_px])), 2)
                         if np.any((raw >= min_tube_px) & np.isfinite(width)) else None),
            **({"gone_bins": [int(i) for i in np.nonzero(absent)[0]]}
               if absent.any() and (gone == "hold" or track == "reacquire") else {}),
            **({"_track": ls.round(2).tolist()} if track == "reacquire" else {}),
            **({"_views": views, "_centre": centre} if keep else {}),
            **({"_paths": {i: routes[i].round(2).tolist() for i in routes}} if paths else {}),
            **({"continuity": {"reread_bins": cont.reread, "held_bins": cont.held, "switched_at": cont.switched}}
               if cont is not None else {})}


def analyze(pcache: str | Path, image_cache: str | Path, grains_path: str | Path | None = None, log=print,
            ghosts: bool = True, abrupt: bool = False, **kw) -> dict:
    """Decode every census grain. ``ghosts``: census discs judged not grains (``census_ghosts``, excluded discs
    included) are neither analysed nor treated as other grains' discs; they are listed under "ghosts" (the review
    pre-fill excludes them as "not_a_grain", to be re-included there if one is a grain). ``abrupt``: bins just after
    an abrupt field-wide change (``abrupt_bins``) cannot start a burst."""
    import json
    bins_p, meta = stack.load(pcache)
    RP = Renderer(bins_p, meta)
    R_img = Renderer(*stack.load(image_cache))
    src = Path(grains_path) if grains_path else Path(pcache) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    extra = {}
    if ghosts:
        judged = census_ghosts(R_img, [g for g in census if g.get("exclude_reason") != "not_a_grain"])
        census = [dict(g, excluded=True, exclude_reason="not_a_grain", ghost=judged[g["id"]]) if g["id"] in judged
                  else g for g in census]
        extra["ghosts"] = judged
        if judged:
            log(f"census discs judged not grains (rim slope, focus vs the census median): {judged}")
    if abrupt:
        rs = int(meta.get("ref_start", 0))
        jumps = abrupt_bins(frame_change(R_img))
        kw["abrupt"] = tuple(b - rs for b in jumps if b >= rs)
        extra["abrupt_bins"] = jumps
    grains = [g for g in census if not g.get("excluded")]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    started, out = time.time(), []
    for g in grains:
        others = [o for o in physical if o["id"] != g["id"]]
        out.append(reach_grain(RP, R_img, meta, g, others, **kw))
    log(f"reach decoder: {len(out)} grains in {time.time() - started:.0f} s")
    return {"schema": PRED_SCHEMA, "method": "reach decoder (learned-evidence prototype)",
            "frames_per_bin": meta["frames_per_bin"], "grains": out, **extra}
