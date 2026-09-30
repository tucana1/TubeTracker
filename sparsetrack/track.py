"""Following a grain through the whole movie, and saying when it can no longer be followed.

The analysis reads each grain in a crop around its census position; ``analyze.local_shifts`` follows it
bin to bin by phase correlation and ``analyze.checked_drift`` throws the whole track away (zero drift for
the movie) when it jumps or wanders. That fails both ways: a real mover is read at its old place, and a
grain that bursts, is carried off or leaves the field is read long after it is gone.

``follow`` tracks one grain by its own look, bin by bin, with a bank of templates of the grain:

- each bin, the normalised cross-correlation (NCC) of every template in the bank with the grain's
  neighbourhood (its disc and rim, high-passed) is taken within ``step_px`` of the last position (more after
  missed bins), never on the disc of another census grain that is still at its place;
- the best match is accepted when it scores at least ``min_score`` (``reacquire_score`` after missed bins);
  otherwise the bin is missed and the grain coasts at its last position (a passing blob, a tube crossing the
  grain, a focus flicker);
- the bank learns the grain's changing look: when an accepted match scores below ``learn_below``, the
  grain's current appearance (at the accepted position) joins the bank, so a grain that darkens, empties
  or gets its own tube is still recognised; the reference template always stays in the bank;
- after more than ``max_gap`` missed bins in a row the grain is lost, from the first of them (a burst, a grain
  swept off faster than the search reaches); a grain whose centre leaves the movie frame is lost too.

Re-finding a knocked grain (``refind``, 30 Sep 2026). Movie 1's grains get knocked: g027 jumps ~30 px in one bin and
turns ~70 degrees with its tube (and, seen from another side, looks darker and smaller), g030 tumbles in place, g056 is
nudged a few px and turns. None of them matches the bank any more, and 0.7.0 lost them for good. From the first missed
bin, the grain's last look - its disc and ``refind_pad_px`` around it (the base of its own tube, which it carries
along), the mean of its last accepted bins - is searched for within ``refind_px``, turned every ``refind_step_deg``.
The grain is taken to be at the best match when that match

- scores at least ``refind_score``, and ``refind_margin`` more than the best match anywhere else (more than
  ``refind_sep_px`` from it): after a knock the grain's look has changed, so its match is weak, but nothing else in
  view looks as much like the grain did (a neighbour is not searched while it is still at its place);
- and the next missed bin finds it at the same place (``refind_agree_px``) and turn (``refind_agree_deg``);
- and it is the grain itself (``_identity``), not only what was round it: its own disc, turned, matches there
  (``refind_disc``), with a rim all round (``ring_contrast``, ``refind_ring`` of its own rim in the reference bins, any
  radius from 0.6 to 1.2 of its own: m1 g027 looks smaller once turned over), and after a jump further than its
  radius, the place did not already look like that in the last bin the grain was followed (``refind_old``). Without
  these, synthetic grains that vanished were "found" at their empty place (their surroundings still match), at their
  tube's base, or in a neighbour that was already there (m2 field: 3 of 9 vanishing grains; true re-finds score disc
  >= 0.52, rim >= 0.46, before <= 0.22; those false ones rim <= 0.14 or before >= 0.83).

The grain then carries its turn (``angle``, deg, clockwise in the image, from its look in the reference bins): the bank
is matched turned by it (its templates are kept in the reference orientation), and the grain's look at the two bins
that found it joins the bank. A disc of the grain's size found by its rim alone (without the look) was tried first and
dropped: in m1's crowds it takes tube segments and junctions for the grain (g027, g060).

Missed bins inside the track are filled by linear interpolation between accepted neighbours, and single-bin
excursions are removed with a running median. ``analyze.followed_drift`` keeps the phase-correlation track
wherever it stays within a few px of this one (so readings do not change there) and uses this one elsewhere.

Linking per-bin census detections (Hough circles, then laptrack) was tried first (29 Sep 2026): the detector
misses a grain for tens of bins once its look changes (m2 g054 once its tube grew) and the links jump onto
neighbours in crowds (m2 g048), so the grain's own look is followed instead.
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
    max_search_px: int = 40      # ...up to this (a push of 33 px over 2 bins is found again)
    min_score: float = 0.5       # NCC below this: the bin is missed
    reacquire_score: float = 0.6  # after two or more missed bins in a row a match must score this much
    learn_below: float = 0.85    # an accepted match scoring below this teaches the bank the grain's new look
    learn_min: float = 0.6       # ...if it scores at least this
    learn_every: int = 4         # bins between two lessons
    bank_size: int = 6           # templates kept besides the reference
    max_gap: int = 8             # more missed bins in a row than this: lost (from the first of them)
    exclusion: float = 0.8       # another grain's disc: centres closer than this x (r1 + r2) are not ours
    median_bins: int = 5         # running median over accepted positions (single-bin excursions)
    # re-finding a knocked grain (module docstring; off: 0.7.0's rules)
    refind: bool = False
    refind_pad_px: float = 10.0  # the grain's last look: its disc and this much around it
    refind_px: int = 40          # searched this far from its last place, from the first missed bin
    refind_step_deg: float = 10.0  # turned every this many degrees
    refind_score: float = 0.45   # the best match scores at least this...
    refind_margin: float = 0.08  # ...and this much more than the best match more than refind_sep_px from it...
    refind_sep_px: float = 6.0
    refind_agree_px: float = 3.0   # ...and the next missed bin finds it within this...
    refind_agree_deg: float = 20.0  # ...turned within this...
    refind_disc: float = 0.4     # ...and the grain's own disc (the bank, turned) matches there at least this...
    refind_ring: float = 0.3     # ...with a rim round it of this share of its own (ring_contrast)...
    refind_old: float = 0.6      # ...and after a jump beyond its radius, the place did not look like that already


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


def turned(t: np.ndarray, deg: float) -> np.ndarray:
    """A square template turned by ``deg`` degrees about its centre pixel, clockwise in the image (x right, y down:
    (x, y) -> (x cos - y sin, x sin + y cos))."""
    if deg == 0:
        return t
    c = (t.shape[1] - 1) / 2.0
    m = cv2.getRotationMatrix2D((c, c), -float(deg), 1.0)
    return cv2.warpAffine(t.astype(np.float32), m, (t.shape[1], t.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _wrap(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def follow(renderer, gx: float, gy: float, gr: float, first: int, last: int, others=(), cfg: FollowConfig | None = None
           ) -> dict:
    """Track the grain at reference (gx, gy), radius gr, from bin ``first`` (its reference bins start there) to
    bin ``last`` (exclusive). ``others``: the other census grains, (x, y, r) in reference coordinates.

    Returns {"xy": (n, 2) offsets (dx, dy) from (gx, gy) per bin from ``first`` (NaN from the loss on),
    "score": (n,) the match score (NaN where missed), "accepted": (n,) bool, "lost_from": the first lost bin
    (absolute index) or None, "lost_reason": None | "gap" | "frame", "angle": (n,) the grain's turn (deg) where
    accepted, "refound": [{"bin", "score", "margin", "moved_px", "turned_deg", "disc", "ring", "before"}]: bins where a
    missed grain was found again by its last look (``refind``), "rejected": the same for places its last look was
    found twice that were not the grain (``_identity``)}."""
    cfg = cfg or FollowConfig()
    n = last - first
    w = int(math.ceil(gr + cfg.pad_px))
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    mask = (np.hypot(xx, yy) <= gr + cfg.pad_px).astype(np.float32)
    margin = int(math.ceil(3 * cfg.hp_sigma))
    reach = max(cfg.max_search_px, cfg.refind_px if cfg.refind else 0)
    # other grains' discs, relative to the census position: [dx, dy, exclusion radius, radius, template]; a clump
    # partner closer than the exclusion distance keeps a smaller disc, so that the grain's own place is never excluded
    near = []
    for ox, oy, orr in others:
        d = math.hypot(ox - gx, oy - gy)
        if 0 < d < gr + orr + 2 * reach + 10:
            near.append([float(ox) - gx, float(oy) - gy, min(cfg.exclusion * (gr + orr), 0.9 * d), float(orr), None])
    H, W = renderer.height, renderer.width

    def still_there(b, nb_) -> bool:
        """Is a neighbour still at its census place (its reference look within 3 px of it)? One that has moved off
        or burst no longer keeps the grain out of its place."""
        dx_, dy_, _, orr, tmpl = nb_
        wn = int(math.ceil(orr + cfg.pad_px))
        yy_, xx_ = np.mgrid[-wn:wn + 1, -wn:wn + 1]
        mk = (np.hypot(xx_, yy_) <= orr + cfg.pad_px).astype(np.float32)

        def around(bb, R):
            half = wn + R + margin + 1
            c = renderer.crop(bb, gx + dx_ + 0.5, gy + dy_ + 0.5, half)
            sl = slice(margin, margin + 2 * (wn + R) + 1)
            return _hp(c, cfg.hp_sigma)[sl, sl]

        if tmpl is None:
            nb_[4] = tmpl = np.mean([around(bb, 0) for bb in range(first, min(first + 3, last))], axis=0)
        res = cv2.matchTemplate(around(b, 3), tmpl, cv2.TM_CCORR_NORMED, mask=mk)
        return float(np.nan_to_num(res, nan=-1.0).max()) >= cfg.min_score

    there: dict = {}  # (bin, neighbour index) -> still_there, asked once per bin

    def excluded(b, base, R):
        """Map cells (offsets base + [-R, R]) that are not the grain's: outside the search circle, or on the disc of
        a neighbour still at its place."""
        off = np.arange(-R, R + 1)
        ox, oy = base[0] + off[None, :], base[1] + off[:, None]
        ex_ = np.hypot(off[None, :], off[:, None]) > R + 0.5
        for k, nb_ in enumerate(near):
            ax_, ay_, ex = nb_[:3]
            inside = np.hypot(ox - ax_, oy - ay_) < ex
            if inside.any():
                if (b, k) not in there:
                    there[(b, k)] = still_there(b, nb_)
                if there[(b, k)]:
                    ex_ |= inside
        return ex_

    def patch(b, px, py, R, ww=w):
        """High-passed neighbourhood of (gx + px, gy + py) at bin b: a (2 (ww + R) + 1) square whose centre pixel is
        centred on that point, and the fraction of it outside the movie frame."""
        half = ww + R + margin + 1
        # crop pixel j covers reference x in [cx - half + j, cx - half + j + 1): centred on cx + 0.5 - half + j,
        # so with cx = x + 0.5 pixel half - 1 is centred on x
        c = renderer.crop(b, gx + px + 0.5, gy + py + 0.5, half)
        sl = slice(margin, margin + 2 * (ww + R) + 1)
        return _hp(c, cfg.hp_sigma)[sl, sl], float(np.isnan(c[sl, sl]).mean())

    ring_ref: list = []  # the grain's own rim in its reference bins (ring_contrast), measured once when needed

    def ring(b, px, py):
        """The rim at (gx + px, gy + py) in bin b, as a share of the grain's own in its reference bins."""
        if not ring_ref:
            ring_ref.append(abs(ring_contrast(renderer, first, gx, gy, gr, avg=min(3, last - first))))
        return ring_contrast(renderer, b, gx + px, gy + py, gr) / max(ring_ref[0], 1.0)

    # the grain's last look for re-finding: its disc and refind_pad_px around it
    wk = int(math.ceil(gr + cfg.refind_pad_px))
    yk, xk = np.mgrid[-wk:wk + 1, -wk:wk + 1]
    mask_k = (np.hypot(xk, yk) <= gr + cfg.refind_pad_px).astype(np.float32)

    # reference template: the mean of the first three bins (the census reference) at the census position
    ref = np.mean([patch(b, 0.0, 0.0, 0)[0] for b in range(first, min(first + 3, last))], axis=0)
    bank = [ref]
    xy = np.full((n, 2), np.nan)
    score = np.full(n, np.nan)
    accepted = np.zeros(n, bool)
    angle = np.full(n, np.nan)
    pos = np.zeros(2)
    theta = 0.0                         # the grain's turn since the reference bins (deg)
    turned_bank: tuple = (None, None)   # (key, the bank turned by theta)
    misses, last_learn = 0, 0
    lost_from, reason = None, None
    recent: list[np.ndarray] = []
    refound: list[dict] = []
    knock = None                        # while missed: {"turns": [(deg, template)], "cand": last bin's candidate}
    last_ok = 0                         # the last accepted bin (index from first)
    rejected: list[dict] = []           # candidates found twice that were not the grain (_identity)
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
        if theta == 0:
            cur = bank
        else:
            key = (theta, len(bank), id(bank[-1]))
            if turned_bank[0] != key:
                turned_bank = (key, [turned(tp, theta) for tp in bank])
            cur = turned_bank[1]
        res = np.max([cv2.matchTemplate(img, tp, cv2.TM_CCORR_NORMED, mask=mask) for tp in cur], axis=0)
        res = np.nan_to_num(res, nan=-1.0)
        # offsets of the map cells from the census position
        off = np.arange(-R, R + 1)
        res[excluded(b, base, R)] = -1.0
        i, j = np.unravel_index(int(np.argmax(res)), res.shape)
        s = float(res[i, j])
        if s >= (cfg.min_score if misses < 2 else cfg.reacquire_score) and out_frac < 0.5:
            dx, dy = _subpix(res, i, j)
            pos = np.array([base[0] + off[j] + dx, base[1] + off[i] + dy])
            xy[t], score[t], accepted[t], angle[t] = pos, s, True, theta
            misses, knock, last_ok = 0, None, t
            look = img[i:i + 2 * w + 1, j:j + 2 * w + 1].copy()  # the matched neighbourhood
            recent.append(turned(look, -theta) if theta else look)  # kept in the reference orientation
            recent = recent[-3:]
            if cfg.learn_min <= s < cfg.learn_below and t - last_learn >= cfg.learn_every and len(recent) == 3:
                bank.append(np.mean(recent, axis=0))
                bank = [bank[0]] + bank[1:][-cfg.bank_size:]
                last_learn = t
            continue
        if cfg.refind and out_frac < 0.5:
            if knock is None:
                knock = _last_look(patch, first, xy, accepted, t, wk, mask_k, cfg)
            found = _search_look(patch, excluded, b, base, pos, wk, mask_k, knock, cfg) if knock else None
            prev = knock["cand"] if knock else None
            if knock:
                knock["cand"] = found
            agree = (found is not None and prev is not None and prev["bin"] == b - 1
                     and math.hypot(found["x"] - prev["x"], found["y"] - prev["y"]) <= cfg.refind_agree_px
                     and abs(_wrap(found["deg"] - prev["deg"])) <= cfg.refind_agree_deg)
            ident = (_identity(patch, mask, bank, _wrap(theta + found["deg"]), found, pos, gr, first + last_ok, cfg,
                               ring) if agree else None)
            if ident is not None and not ident["ok"]:
                rejected.append({"bin": int(b), "score": round(found["score"], 3), "margin": round(found["margin"], 3),
                                 "moved_px": round(math.hypot(found["x"] - pos[0], found["y"] - pos[1]), 1),
                                 "turned_deg": round(found["deg"], 1), **{k: v for k, v in ident.items() if k != "ok"}})
            if ident is not None and ident["ok"]:
                moved = math.hypot(found["x"] - pos[0], found["y"] - pos[1])
                theta = _wrap(theta + found["deg"])
                for c_ in (prev, found):  # both bins that found it
                    tc = c_["bin"] - first
                    xy[tc], score[tc], accepted[tc], angle[tc] = (c_["x"], c_["y"]), c_["score"], True, theta
                pos = np.array([found["x"], found["y"]])
                looks = [turned(patch(c_["bin"], c_["x"], c_["y"], 0)[0], -theta) for c_ in (prev, found)]
                bank.append(np.mean(looks, axis=0))  # the grain's look now, in the reference orientation
                bank = [bank[0]] + bank[1:][-cfg.bank_size:]
                recent, last_learn = looks[-1:], t
                refound.append({"bin": int(prev["bin"]), "score": round(found["score"], 3),
                                "margin": round(found["margin"], 3), "moved_px": round(moved, 1),
                                "turned_deg": round(found["deg"], 1),
                                **{k: v for k, v in ident.items() if k != "ok"}})
                misses, knock, last_ok = 0, None, t
                continue
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
            "bank": len(bank), "angle": angle, "refound": refound, "rejected": rejected}


def _last_look(patch, first, xy, accepted, t, wk, mask_k, cfg: FollowConfig) -> dict | None:
    """The grain's look (disc and surroundings, as it was turned then) over its last accepted bins before bin index
    ``t`` (up to three in a row), turned every ``refind_step_deg``."""
    last = []
    k = t - 1
    while k >= 0 and accepted[k] and len(last) < 3:
        last.append(k)
        k -= 1
    if not last:
        return None
    look = np.mean([patch(first + kk, float(xy[kk][0]), float(xy[kk][1]), 0, wk)[0] for kk in last], axis=0)
    look = look.astype(np.float32) * mask_k
    return {"turns": [(float(a), turned(look, a)) for a in np.arange(0.0, 360.0 - 1e-6, cfg.refind_step_deg)],
            "cand": None}


def _identity(patch, mask, bank, theta, cand, pos, gr, b_ok, cfg: FollowConfig, ring) -> dict:
    """Is the grain itself where its last look was found (``cand``, turned to ``theta``)? Its disc, turned, must be
    there (``refind_disc``), with a rim round it (``refind_ring`` of the grain's own in the reference bins, by
    ``ring``): not only its surroundings, which stay when the grain bursts or is swept off, or a tube's base. After a
    jump further than its radius, what is there now must not have been there already in the last bin the grain was
    followed in (``refind_old``: a neighbour, or a grain the census missed). Returns {"ok", "disc", "ring",
    "before"}."""
    here, _ = patch(cand["bin"], cand["x"], cand["y"], 2)
    tps = [turned(tp, theta) for tp in _disc_templates(bank)]
    disc = float(np.nan_to_num(np.max([cv2.matchTemplate(here, tp, cv2.TM_CCORR_NORMED, mask=mask) for tp in tps]),
                               nan=-1.0))
    rim = abs(ring(cand["bin"], cand["x"], cand["y"]))  # a grain turned over can show a light rim
    before = -1.0
    if math.hypot(cand["x"] - pos[0], cand["y"] - pos[1]) > gr:
        now = here[2:-2, 2:-2]
        was, _ = patch(b_ok, cand["x"], cand["y"], 2)
        before = float(np.nan_to_num(cv2.matchTemplate(was, now, cv2.TM_CCORR_NORMED, mask=mask).max(), nan=-1.0))
    ok = disc >= cfg.refind_disc and rim >= cfg.refind_ring and before < cfg.refind_old
    return {"ok": ok, "disc": round(disc, 3), "ring": round(rim, 2), "before": round(before, 3)}


def ring_contrast(renderer, b: int, x: float, y: float, gr: float, search: int = 2, avg: int = 1,
                  fracs=tuple(np.arange(0.6, 1.201, 0.05)), gap: float = 3.0) -> float:
    """A grain's rim at reference (x, y) in bin ``b`` (the mean of ``avg`` bins from there): the rim against just
    inside and outside it, taken round the whole circle (the 25th percentile of its size over 48 directions, so a tube
    or a crescent does not make a rim), the best over centres within ``search`` px and radii 0.6-1.2 x ``gr`` (a
    grain that turns over can look smaller: m1 g027 x0.65). Grey levels, signed like the rim (dark: positive)."""
    half = int(math.ceil(1.2 * gr + gap + search + 3))
    img = np.mean([renderer.crop(bb, x + 0.5, y + 0.5, half) for bb in range(b, b + avg)], axis=0)
    img = np.nan_to_num(img, nan=float(np.nanmedian(img)) if np.isfinite(img).any() else 0.0).astype(np.float32)
    phis = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    best = 0.0
    for dx in range(-search, search + 1):
        for dy in range(-search, search + 1):
            cx, cy = half - 1 + dx, half - 1 + dy
            for f in fracs:
                rad = np.array([f * gr - gap, f * gr, f * gr + gap])[:, None]
                xs = (cx + rad * np.cos(phis)[None]).astype(np.float32)
                ys = (cy + rad * np.sin(phis)[None]).astype(np.float32)
                pr = cv2.remap(img, xs, ys, cv2.INTER_LINEAR)
                c = 0.5 * (pr[0] + pr[2]) - pr[1]
                s = float(np.percentile(np.abs(c), 25)) * (1.0 if np.median(c) >= 0 else -1.0)
                if abs(s) > abs(best):
                    best = s
    return best


def _disc_templates(bank: list[np.ndarray]) -> list[np.ndarray]:
    """The reference and the latest learned look."""
    return [bank[0]] if len(bank) == 1 else [bank[0], bank[-1]]


def _search_look(patch, excluded, b, base, pos, wk, mask_k, knock, cfg: FollowConfig) -> dict | None:
    """The best match of the grain's last look, turned, within ``refind_px`` of its last place in bin ``b``, if it
    stands out (module docstring): {"bin", "x", "y", "deg", "score", "margin"} or None."""
    R = int(cfg.refind_px)
    img, _ = patch(b, float(base[0]), float(base[1]), R, wk)
    maps = np.stack([np.nan_to_num(cv2.matchTemplate(img, tp, cv2.TM_CCORR_NORMED, mask=mask_k), nan=-1.0)
                     for _, tp in knock["turns"]])
    best = maps.max(axis=0)
    best[excluded(b, base, R)] = -1.0
    i, j = np.unravel_index(int(np.argmax(best)), best.shape)
    s = float(best[i, j])
    ii, jj = np.mgrid[0:best.shape[0], 0:best.shape[1]]
    other = float(np.max(np.where(np.hypot(ii - i, jj - j) > cfg.refind_sep_px, best, -1.0)))
    if s < cfg.refind_score or s - other < cfg.refind_margin:
        return None
    dx, dy = _subpix(best, i, j)
    return {"bin": int(b), "x": float(base[0] + j - R + dx), "y": float(base[1] + i - R + dy),
            "deg": _wrap(knock["turns"][int(np.argmax(maps[:, i, j]))][0]), "score": s, "margin": s - other}
