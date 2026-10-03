"""Temporal propagation of human traces: training crops at the bins between and after a grain's traces, and before
its onset.

    python -m prototypes.tube_adapt.propagate OUT.npz --grains g003 g005 ... [--movie m1] [--every 3]

Routes are carried between bins with DIS optical flow (OpenCV PRESET_MEDIUM) composed in 10-bin steps on 3-bin-mean
registered frames, uint8 over the 1-99 percentile window of the grain's region
(``runs/lab_checks_2026-09-29/short_interval/flowcheck.py``: annotator routes carried to median ~1.3 px). Per grain:

- **between consecutive traces** b1 < b2 (both FULL or PARTIAL with a route; skipped where the tube got shorter by
  more than max(5 px, 15%): a burst or a break): the b1 tube is tube at every bin in between. Its route at bin b is
  the b1 route carried forward blended (by time) with the first L(b1) px of the b2 route carried back, each anchored
  at its own trace; bins where the two disagree by more than ``max_dev`` px (median closest-point distance) are
  skipped. Unscored: the rest of the carried-back b2 route (the tube grows into it somewhere in between) within
  ``ext_margin`` px, a disc of ``tip_blind`` px round the carried apex, and everything ``body_px``..``gap_px`` px
  from the route; scored background: ``gap_px``..``band_px`` px from it (only where neither trace touches anything);
- **after the last trace**: the traced tube stays (no burst is labelled in these movies), carried forward while the
  flow's forward-backward error stays small (median <= ``fb_max`` px per step); where a later bin was answered
  "unsure" or "no tube", at most ``after_cap`` bins and never up to it;
- **before the onset bracket** (bins up to ``last_absent_bin`` - 1): no tube at the exit - the grain's first traced
  route placed where the grain is at that bin (the labelling tool's grain offsets) is scored background, with the
  background band and the grain's inside, as ``prototypes.tube_net.realdata``'s negatives.

The grain's inside (at its place at that bin) is background, except on traces that start over the grain (a face-on
pore; ``over_grain``), where it is unscored. Bins within 2 of a trace are left to the traced crops
(``realdata.py``: the trace at +/- 1, 2 bins). Each crop stores its grain, bin and kind, so subsets of grains can be
drawn from one shard.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from prototypes.learned_flood.data import CacheView
from prototypes.tube_adapt.common import MOVIES, REPO, labels, offsets

STEP = 10


def dense(path, step: float = 1.0) -> np.ndarray:
    p = np.asarray(path, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    ss = np.append(np.arange(0, s[-1], step), s[-1])
    return np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], 1)


def arclen(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def resample(p: np.ndarray, n: int) -> np.ndarray:
    s = arclen(p)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def closest(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric median closest-point distance between two point sets."""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return float(max(np.median(d.min(axis=1)), np.median(d.min(axis=0))))


class Region:
    """3-bin-mean registered frames of one region of the movie as uint8 for DIS, cached."""

    def __init__(self, R, cx: float, cy: float, half: int, window_bins: list[int]):
        self.R, self.cx, self.cy, self.half = R, float(cx), float(cy), int(half)
        self._f: dict[int, np.ndarray] = {}
        vals = np.concatenate([self.mean3(b).ravel() for b in window_bins])
        self.lo, self.hi = (float(v) for v in np.nanpercentile(vals, [1, 99]))
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def mean3(self, b: int) -> np.ndarray:
        if b not in self._f:
            bs = [x for x in (b - 1, b, b + 1) if 0 <= x < self.R.n_bins]
            self._f[b] = np.mean([self.R.crop(x, self.cx, self.cy, self.half) for x in bs], axis=0)
        return self._f[b]

    def u8(self, b: int) -> np.ndarray:
        img = np.nan_to_num(self.mean3(b), nan=self.lo)
        return np.clip((img - self.lo) / (self.hi - self.lo) * 255, 0, 255).astype(np.uint8)

    def to_crop(self, p: np.ndarray) -> np.ndarray:
        return p - [self.cx - self.half, self.cy - self.half] - 0.5

    def to_ref(self, q: np.ndarray) -> np.ndarray:
        return q + [self.cx - self.half, self.cy - self.half] + 0.5

    def flow(self, a: int, c: int) -> np.ndarray:
        return self.dis.calc(self.u8(a), self.u8(c), None)


def _sample(flow: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    return np.stack([cv2.remap(flow[..., k], q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)[0] for k in (0, 1)], 1)


def carry(reg: Region, pts: np.ndarray, b0: int, b1: int, step: int = STEP) -> tuple[dict, dict]:
    """Points (reference coordinates at b0) carried to b1 in ``step``-bin steps: {bin: points} at every step bin and
    {bin: median forward-backward error of the step ending there (px)}."""
    sgn = 1 if b1 > b0 else -1
    seq = list(range(b0, b1, sgn * step)) + [b1]
    q = reg.to_crop(np.asarray(pts, float))
    out, fb = {b0: np.asarray(pts, float).copy()}, {}
    for a, c in zip(seq, seq[1:]):
        fwd, bwd = reg.flow(a, c), reg.flow(c, a)
        q2 = q + _sample(fwd, q)
        back = q2 + _sample(bwd, q2)
        fb[c] = float(np.median(np.hypot(*(back - q).T)))
        q = q2
        out[c] = reg.to_ref(q)
    return out, fb


def at(carried: dict, b: int) -> np.ndarray:
    """Carried points at bin ``b``, linear in time between the step bins either side."""
    ks = sorted(carried)
    if b <= ks[0]:
        return carried[ks[0]]
    if b >= ks[-1]:
        return carried[ks[-1]]
    j = int(np.searchsorted(ks, b))
    a, c = ks[j - 1], ks[j]
    w = (b - a) / (c - a)
    return (1 - w) * carried[a] + w * carried[c]


def _polyline_dist(q: np.ndarray, size: int) -> np.ndarray:
    line = np.zeros((size, size), np.uint8)
    cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
    return cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)


def targets(route: np.ndarray, cx: float, cy: float, half: int, gxy, gr: float, *, neg: bool, band: bool,
            body_px: float, gap_px: float, band_px: float, blind_discs=(), blind_lines=(), ext_margin: float = 8.0,
            inside_unscored: bool = False):
    """(body, w): body within ``body_px`` of the route (none on negatives), scored background ``gap_px``..
    ``band_px`` from it (if ``band``), unscored discs (x, y, r) and polylines (within ``ext_margin``), the grain's
    inside background (or unscored)."""
    size = 2 * half
    off = np.array([cx - half, cy - half]) + 0.5
    d = _polyline_dist(route - off, size)
    if neg:
        body = np.zeros((size, size), bool)
        w = (d <= gap_px) | (band & (d >= gap_px) & (d <= band_px))
    else:
        body = d <= body_px
        w = body | (band & (d >= gap_px) & (d <= band_px))
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    for bx, by, rad in blind_discs:
        w &= np.hypot(jj - (bx - off[0]), ii - (by - off[1])) > rad
    for ln in blind_lines:
        if len(ln) >= 2:
            w &= _polyline_dist(np.asarray(ln) - off, size) > ext_margin
    inside = np.hypot(jj - (gxy[0] - off[0]), ii - (gxy[1] - off[1])) <= gr - 1
    if inside_unscored:
        w &= ~inside
    else:
        w |= inside
    body &= ~inside
    return body.astype(np.uint8), w.astype(np.uint8), (d <= 1.0) & ~inside


def valid_until(fb: dict, b0: int, fb_max: float) -> int:
    """The last step bin a chain started at ``b0`` reaches before its first step with a forward-backward error above
    ``fb_max`` px (``b0`` itself if the first step is already bad)."""
    last = b0
    for c in sorted(fb, key=lambda c: abs(c - b0)):
        if fb[c] > fb_max:
            break
        last = c
    return last


def build(out: str | Path, grains: list[str], movie: str = "m1", half: int = 48, every: int = 3, seed: int = 0,
          body_px: float = 2.0, gap_px: float = 8.0, band_px: float = 16.0, tip_blind: float = 10.0,
          partial_blind: float = 18.0, ext_margin: float = 8.0, max_dev: float = 3.0, fb_max: float = 1.5,
          one_sided: int = 30, after_cap: int = 35, neg_every: int = 2, over_grain: float = 0.6, log=print) -> Path:
    L = labels(movie)
    view = CacheView(REPO / MOVIES[movie][0])
    R = view.r
    fol = offsets(movie) if movie == "m1" else {}
    rng = np.random.default_rng(seed)
    lo, hi = view.rs + 3, view.n_bins - 2
    xs, bodies, ws, skels, info, gids, kinds = [], [], [], [], [], [], []
    stats = {"between": 0, "after": 0, "negative": 0, "bins_both": 0, "bins_fwd_only": 0, "bins_bwd_only": 0,
             "skipped_dev": 0, "skipped_unreliable": 0, "skipped_shrank": 0, "after_stopped_fb": 0,
             "skipped_nan": 0, "intervals": 0}
    devs, fbs = [], []

    def add(b, cx, cy, route, gxy, gr, kind, gid, **kw):
        cx = float(np.clip(round(cx), half + 4, R.width - half - 4))
        cy = float(np.clip(round(cy), half + 4, R.height - half - 4))
        x = view.sample(b, cx, cy, half)
        if not np.isfinite(x).all():
            stats["skipped_nan"] += 1
            return
        body, w, sk = targets(route, cx, cy, half, gxy, gr, body_px=body_px, gap_px=gap_px, band_px=band_px,
                              ext_margin=ext_margin, **kw)
        xs.append(x.astype(np.float16)); bodies.append(body); ws.append(w)
        skels.append((sk & (body > 0)).astype(np.uint8))
        info.append((b, cx, cy, int(kw.get("neg", False))))
        gids.append(gid)
        kinds.append(kind)
        stats[kind] += 1

    def jit():
        return rng.uniform(-half / 3, half / 3)

    def along(route):
        s = arclen(route)
        u = rng.uniform(0, s[-1])
        return np.interp(u, s, route[:, 0]), np.interp(u, s, route[:, 1])

    for gid in grains:
        g, lab = L["grains"][gid], L["labels"].get(gid) or {}
        tr = sorted(((int(b), t) for b, t in (lab.get("traces") or {}).items()
                     if t["state"] in ("full", "partial") and len(t.get("path_xy_ref") or []) >= 2), key=lambda x: x[0])
        if not tr:
            continue
        answered = sorted(int(b) for b in (lab.get("traces") or {}))
        centre = lambda t: np.array([[g["x"] + (t.get("view_offset") or [0, 0])[0],
                                      g["y"] + (t.get("view_offset") or [0, 0])[1]]])  # the grain where the annotator saw it
        allp = np.concatenate([np.asarray(t["path_xy_ref"], float) for _, t in tr])
        cx, cy = (allp.min(axis=0) + allp.max(axis=0)) / 2
        hw = int(min(400, max(96, np.max(np.abs(allp - [cx, cy])) + 64)))
        reg = Region(R, cx, cy, hw, [b for b, _ in tr])
        over = lambda t: bool(over_grain) and np.hypot(*(np.asarray(t["path_xy_ref"][0]) - centre(t)[0])) < \
            over_grain * g["r"]

        def emit(b, route, gc, kind, t_state, band, ov, ext=None, start=0):
            blind = [(route[-1][0], route[-1][1], (tip_blind if kind == "between" else max(tip_blind, band_px + 4.0))
                      if t_state == "full" else partial_blind)]
            kw = dict(neg=False, band=band, blind_discs=blind, blind_lines=[ext] if ext is not None and len(ext) >= 2
                      else [], inside_unscored=ov)
            add(b, *(np.array(along(route)) + [jit(), jit()]), route, gc, g["r"], kind, gid, **kw)
            if (b - start) // every % 2 == 0:
                add(b, route[0][0] + jit(), route[0][1] + jit(), route, gc, g["r"], kind, gid, **kw)

        # 1) between consecutive traces
        for (b1, t1), (b2, t2) in zip(tr, tr[1:]):
            r1, r2 = dense(t1["path_xy_ref"]), dense(t2["path_xy_ref"])
            l1, l2 = arclen(r1)[-1], arclen(r2)[-1]
            if l2 < l1 - max(5.0, 0.15 * l1):
                stats["skipped_shrank"] += 1
                continue
            stats["intervals"] += 1
            fwd, fb_f = carry(reg, np.vstack([r1, centre(t1)]), b1, b2)
            bwd, fb_b = carry(reg, np.vstack([r2, centre(t2)]), b2, b1)
            fbs.extend(list(fb_f.values()) + list(fb_b.values()))
            vf, vb = valid_until(fb_f, b1, fb_max), valid_until(fb_b, b2, fb_max)
            n1 = int(np.searchsorted(arclen(r2), l1, side="right"))  # r2's first L(b1) px
            band = not t1.get("contact") and not t2.get("contact")
            ov = over(t1)
            for b in range(b1 + 3, b2 - 2, every):
                if not lo <= b <= hi:
                    continue
                okf, okb = b <= vf, b >= vb
                f, k = at(fwd, b), at(bwd, b)
                fr, fc = f[:-1], f[-1]
                kr, kc = k[:-1], k[-1]
                kp, ext = kr[:max(n1, 2)], kr[max(n1, 2) - 1:]
                n = max(int(round(l1)) + 1, 2)
                if okf and okb:
                    dev = closest(fr, kp)
                    devs.append(dev)
                    if dev > max_dev:
                        stats["skipped_dev"] += 1
                        continue
                    w = (b - b1) / (b2 - b1)
                    route, gc = (1 - w) * resample(fr, n) + w * resample(kp, n), (1 - w) * fc + w * kc
                    stats["bins_both"] += 1
                elif okf and b - b1 <= one_sided:
                    route, gc, ext = resample(fr, n), fc, (ext if okb else None)
                    stats["bins_fwd_only"] += 1
                elif okb and b2 - b <= one_sided:
                    route, gc = resample(kp, n), kc
                    stats["bins_bwd_only"] += 1
                else:
                    stats["skipped_unreliable"] += 1
                    continue
                if ext is None:  # the tube's growth since b1 is not known: a wider unscored disc at the apex
                    t_state = "partial"
                else:
                    t_state = t1["state"]
                emit(b, route, gc, "between", t_state, band, ov, ext, b1)
        # 2) after the last trace
        bl, tl = tr[-1]
        later = [a for a in answered if a > bl]
        end = hi if not later else min(hi, later[0] - 1, bl + after_cap)
        if end >= bl + 3:
            rl = dense(tl["path_xy_ref"])
            fwd, fb_f = carry(reg, np.vstack([rl, centre(tl)]), bl, end)
            fbs.extend(fb_f.values())
            stop = valid_until(fb_f, bl, fb_max)
            stats["after_stopped_fb"] += int(stop < end)
            for b in range(bl + 3, stop + 1, every):
                f = at(fwd, b)
                emit(b, f[:-1], f[-1], "after", tl["state"], not tl.get("contact"), over(tl), None, bl)
        # 3) before the onset bracket: no tube at the exit
        on = lab.get("onset") or {}
        if on.get("verdict") == "emerged_within":
            b0, t0 = tr[0]
            p0 = np.asarray(t0["path_xy_ref"], float)
            first = lo
            bwd, fb_b = carry(reg, np.vstack([p0, centre(t0)]), b0, first)
            vb = valid_until(fb_b, b0, fb_max)
            for nb_ in range(first, int(on["last_absent_bin"]), neg_every):
                if nb_ >= vb:
                    k = at(bwd, nb_)
                    path, gc = k[:-1], k[-1]
                else:  # the flow lost it: where the labelling tool's grain offsets put it (realdata's negatives)
                    sh = (fol[gid][nb_] - fol[gid][b0]) if gid in fol else np.zeros(2)
                    path, gc = p0 + sh, centre(t0)[0] + sh
                add(nb_, path[0][0] + jit(), path[0][1] + jit(), path, gc, g["r"], "negative", gid,
                    neg=True, band=not t0.get("contact"), inside_unscored=over(t0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    B, W = np.stack(bodies), np.stack(ws)
    np.savez_compressed(out, x=np.stack(xs), body=B, tip=np.zeros(B.shape, np.float16), w=W, skel=np.stack(skels),
                        info=np.array(info, np.float32), grain=np.array(gids), kind=np.array(kinds), movie=movie)
    devs, fbs = np.array(devs + [np.nan]), np.array(fbs)
    log(f"{out.name}: {len(xs)} crops {stats}; scored {100 * W.mean():.1f}%, tube among them "
        f"{100 * B.sum() / max(W.sum(), 1):.1f}%; forward/backward routes (both reliable): median "
        f"{np.nanmedian(devs):.2f} px, p90 {np.nanpercentile(devs, 90):.2f}; flow FB error per step median "
        f"{np.median(fbs):.2f}, p90 {np.percentile(fbs, 90):.2f}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--grains", nargs="+", required=True)
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    build(a.out, a.grains, a.movie, every=a.every)
