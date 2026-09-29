"""Following a grain through the whole movie, and saying when it can no longer be followed.

The analysis reads each grain in a crop around its census position; ``analyze.local_shifts`` follows it
bin to bin by phase correlation and ``analyze.checked_drift`` throws the whole track away (zero drift for
the movie) when it jumps or wanders. That fails both ways: a real mover is read at its old place, and a
grain that bursts, is carried off or leaves the field is read long after it is gone.

``follow`` tracks one grain by its own look, bin by bin, with a bank of templates of the grain:

- each bin, the normalised cross-correlation (NCC) of every template in the bank with the grain's
  neighbourhood (a disc a little larger than the grain, high-passed) is taken within ``step_px`` of the
  last position (more after missed bins), never on another census grain's disc;
- the best match is accepted when it scores at least ``min_score``; otherwise the bin is missed and the
  grain coasts at its last position (a passing blob, a tube crossing the grain, a focus flicker);
- the bank learns the grain's changing look: when an accepted match scores below ``learn_below``, the
  grain's current appearance (at the accepted position) joins the bank, so a grain that darkens, empties
  or gets its own tube is still recognised; the reference template always stays in the bank;
- after more than ``max_gap`` missed bins in a row the grain is lost, from the first of them; a grain whose
  disc leaves the movie frame is lost too.

Missed bins inside the track are filled by linear interpolation between accepted neighbours, and single-bin
excursions are removed with a running median. The result is a coarse (0.25-px) track; ``analyze`` refines
it to the phase-correlation precision it used before with a correlation window centred on this track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FollowConfig:
    pad_px: float = 2.0          # template: the grain's disc + this margin (px; more takes in the tube's base,
                                 # which stays behind when the grain moves off it)
    hp_sigma: float = 3.0        # high-pass before matching (px)
    step_px: int = 8             # search radius per bin (px; m2's g048 is swept off at ~8 px per bin); grows with
                                 # missed bins...
    max_search_px: int = 30      # ...up to this
    min_score: float = 0.5       # NCC below this: the bin is missed
    learn_below: float = 0.85    # an accepted match scoring below this teaches the bank the grain's new look
    learn_min: float = 0.6       # ...if it scores at least this
    learn_every: int = 4         # bins between two lessons
    bank_size: int = 6           # templates kept besides the reference
    max_gap: int = 8             # more missed bins in a row than this: lost (from the first of them)
    exclusion: float = 0.8       # another grain's disc: centres closer than this x (r1 + r2) are not ours
    median_bins: int = 5         # running median over accepted positions (single-bin excursions)


def _hp(img: np.ndarray, sigma: float) -> np.ndarray:
    img = img.astype(np.float32)
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=float(np.nanmedian(img)) if np.isfinite(img).any() else 0.0)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def _subpix(res: np.ndarray, y: int, x: int) -> tuple[float, float]:
    def par(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dy = par(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    dx = par(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def follow(renderer, gx: float, gy: float, gr: float, first: int, last: int, others=(), cfg: FollowConfig | None = None
           ) -> dict:
    """Track the grain at reference (gx, gy), radius gr, from bin ``first`` (its reference bins start there) to
    bin ``last`` (exclusive). ``others``: the other census grains, (x, y, r) in reference coordinates.

    Returns {"xy": (n, 2) offsets (dx, dy) from (gx, gy) per bin from ``first`` (NaN from the loss on),
    "score": (n,) the match score (NaN where missed), "accepted": (n,) bool, "lost_from": the first lost bin
    (absolute index) or None, "lost_reason": None | "gap" | "frame"}."""
    cfg = cfg or FollowConfig()
    n = last - first
    w = int(math.ceil(gr + cfg.pad_px))
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    mask = (np.hypot(xx, yy) <= gr + cfg.pad_px).astype(np.float32)
    margin = int(math.ceil(3 * cfg.hp_sigma))
    # other grains' discs, relative to the census position: (dx, dy, exclusion radius); a clump partner closer
    # than the exclusion distance keeps a smaller disc, so that the grain's own place is never excluded
    near = []
    for ox, oy, orr in others:
        d = math.hypot(ox - gx, oy - gy)
        if 0 < d < gr + orr + 2 * cfg.max_search_px + 10:
            near.append((float(ox) - gx, float(oy) - gy, min(cfg.exclusion * (gr + orr), 0.9 * d)))
    H, W = renderer.height, renderer.width

    def patch(b, px, py, R):
        """High-passed neighbourhood of (gx + px, gy + py) at bin b: a (2 (w + R) + 1) square whose centre pixel is
        centred on that point, and the fraction of it outside the movie frame."""
        half = w + R + margin + 1
        # crop pixel j covers reference x in [cx - half + j, cx - half + j + 1): centred on cx + 0.5 - half + j,
        # so with cx = x + 0.5 pixel half - 1 is centred on x
        c = renderer.crop(b, gx + px + 0.5, gy + py + 0.5, half)
        sl = slice(margin, margin + 2 * (w + R) + 1)
        return _hp(c, cfg.hp_sigma)[sl, sl], float(np.isnan(c[sl, sl]).mean())

    # reference template: the mean of the first three bins (the census reference) at the census position
    ref = np.mean([patch(b, 0.0, 0.0, 0)[0] for b in range(first, min(first + 3, last))], axis=0)
    bank = [ref]
    xy = np.full((n, 2), np.nan)
    score = np.full(n, np.nan)
    accepted = np.zeros(n, bool)
    pos = np.zeros(2)
    misses, last_learn = 0, 0
    lost_from, reason = None, None
    recent: list[np.ndarray] = []
    for t in range(n):
        b = first + t
        # the grain's centre has left the movie frame: nothing to follow there
        cx, cy = gx + pos[0] + renderer.shifts[b][0], gy + pos[1] + renderer.shifts[b][1]
        if cx < 0 or cy < 0 or cx >= W or cy >= H:
            lost_from, reason = b - misses, "frame"
            break
        R = int(min(cfg.step_px * (1 + misses), cfg.max_search_px))
        base = np.round(pos).astype(int)
        img, out_frac = patch(b, float(base[0]), float(base[1]), R)
        res = np.max([cv2.matchTemplate(img, tp, cv2.TM_CCORR_NORMED, mask=mask) for tp in bank], axis=0)
        res = np.nan_to_num(res, nan=-1.0)
        # offsets of the map cells from the census position
        off = np.arange(-R, R + 1)
        ox, oy = base[0] + off[None, :], base[1] + off[:, None]
        for ax_, ay_, ex in near:
            res[np.hypot(ox - ax_, oy - ay_) < ex] = -1.0
        res[np.hypot(off[None, :], off[:, None]) > R + 0.5] = -1.0
        i, j = np.unravel_index(int(np.argmax(res)), res.shape)
        s = float(res[i, j])
        if s >= cfg.min_score and out_frac < 0.5:
            dx, dy = _subpix(res, i, j)
            pos = np.array([base[0] + off[j] + dx, base[1] + off[i] + dy])
            xy[t], score[t], accepted[t] = pos, s, True
            misses = 0
            recent.append(img[i:i + 2 * w + 1, j:j + 2 * w + 1].copy())  # the matched neighbourhood
            recent = recent[-3:]
            if cfg.learn_min <= s < cfg.learn_below and t - last_learn >= cfg.learn_every and len(recent) == 3:
                bank.append(np.mean(recent, axis=0))
                bank = [bank[0]] + bank[1:][-cfg.bank_size:]
                last_learn = t
        else:
            misses += 1
            recent = []
            if misses > cfg.max_gap:
                lost_from, reason = b - misses + 1, "gap"
                break
    nt = n if lost_from is None else lost_from - first
    if lost_from is None and misses:
        nt = n  # missed bins at the very end: the grain coasts at its last place
    good = np.flatnonzero(accepted[:nt])
    track = np.full((n, 2), np.nan)
    if len(good):
        for k in range(2):
            v = xy[good, k]
            if cfg.median_bins > 1 and len(v) >= cfg.median_bins:
                from scipy.ndimage import median_filter
                v = median_filter(v, size=cfg.median_bins, mode="nearest")
            track[:nt, k] = np.interp(np.arange(nt), good, v)
    return {"xy": track, "score": score, "accepted": accepted, "lost_from": lost_from, "lost_reason": reason,
            "bank": len(bank)}
