"""Training samples from the human traces (one npz per movie, inputs computed once).

    python -m prototypes.tip_detector.build ld m2 m1

Extractions of 192 px (rotated and cut to 128 px in training) at a trace's bin:
- FULL trace: centred on the grain (+/- 24 px), on the apex (+/- 40 px) and on a random point of the trace (+/- 30 px);
- PARTIAL trace: on the grain and on a random point of the trace (no apex: the tube goes on past the traced end);
- no_tube / burst: on the grain (no tip anywhere near it);
- before the onset (verdict emerged_within): bins last_absent_bin - (1, 3, 6, 12, 24), on the grain where it is at
  that bin (the labelling tool's grain following).
Targets (built from every labelled grain's trace at that bin that falls in the extraction):
- apexes: the last point of each FULL trace (a Gaussian at training time);
- scored region (loss only there; elsewhere an unlabelled tube's tip may lie): the grain's disc r + 25 px and the band
  25 px round each traced path; before the onset the grain's disc r + 40 px; less blind zones: 40 px round the end of
  a PARTIAL trace, the disc r + 40 and band round an UNSURE trace;
- body (weak second head): within 2.5 px of each FULL or PARTIAL trace.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .common import OUT, Movie

EXT = 96          # extraction half (192 px)
BAND = 25.0
NEG_BACK = (1, 3, 6, 12, 24)
KINDS = ("full_grain", "full_apex", "full_path", "partial", "no_tube", "pre_onset")


def seg_dist(px: np.ndarray, py: np.ndarray, path: np.ndarray) -> np.ndarray:
    if len(path) == 1:
        return np.hypot(px - path[0, 0], py - path[0, 1])
    d = np.full(px.shape, np.inf, np.float32)
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * vx + (py - y0) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(px - (x0 + t * vx), py - (y0 + t * vy)))
    return d


def annotate(mv: Movie, b: int, cx: float, cy: float, half: int, neg_disc=None):
    """Scored region, body target and apexes (crop pixel-centre coordinates: pixel j's centre is at j) for an
    extraction of bin b centred on reference (cx, cy)."""
    S = 2 * half
    x0, y0 = cx - half + 0.5, cy - half + 0.5  # reference coordinate of crop pixel 0's centre
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    scored = np.zeros((S, S), bool)
    blind = np.zeros((S, S), bool)
    body = np.zeros((S, S), bool)
    apexes = []
    for gid, lab in mv.labels.items():
        t = (lab.get("traces") or {}).get(str(b))
        if t is None:
            continue
        gx, gy, r = mv.grain_at(gid, b, t)
        path = np.asarray(t.get("path_xy_ref") or [], float).reshape(-1, 2)
        far = np.hypot(gx - cx, gy - cy) > half * 1.5 + r + 40 + (t.get("length_px") or 0)
        if far:
            continue
        gd = np.hypot(jj - (gx - x0), ii - (gy - y0))
        q = path - [x0, y0] if len(path) else path
        st = t["state"]
        if st == "full" and len(q) >= 2:
            d = seg_dist(jj, ii, q)
            scored |= (gd <= r + BAND) | (d <= BAND)
            body |= d <= 2.5
            apexes.append(q[-1])
        elif st == "partial" and len(q) >= 2:
            d = seg_dist(jj, ii, q)
            scored |= (gd <= r + BAND) | (d <= BAND)
            body |= d <= 2.5
            blind |= np.hypot(jj - q[-1][0], ii - q[-1][1]) <= 40
        elif st in ("no_tube", "burst"):
            scored |= gd <= r + BAND
        else:  # unsure (or a FULL/PARTIAL without a path)
            blind |= gd <= r + 40
            if len(q) >= 2:
                blind |= seg_dist(jj, ii, q) <= BAND
                blind |= np.hypot(jj - q[-1][0], ii - q[-1][1]) <= 40
    if neg_disc is not None:
        gx, gy, rad = neg_disc
        scored |= np.hypot(jj - (gx - x0), ii - (gy - y0)) <= rad
    scored &= ~blind
    return scored, body, apexes, blind


def build(name: str, seed: int = 0, log=print):
    mv = Movie(name)
    rng = np.random.default_rng(seed)
    W, H = mv.R.width, mv.R.height
    X, SC, BO, BL, AP, INFO = [], [], [], [], [], []
    gids = sorted(mv.labels)
    t0 = time.time()

    def add(b, cx, cy, kind, gid, neg_disc=None):
        cx = float(np.clip(round(cx), EXT + 4, W - EXT - 4))
        cy = float(np.clip(round(cy), EXT + 4, H - EXT - 4))
        x = mv.inputs(b, cx, cy, EXT)
        sc, bo, ap, bl = annotate(mv, b, cx, cy, EXT, neg_disc)
        a = np.full((4, 2), np.nan, np.float32)
        for k, p in enumerate(ap[:4]):
            a[k] = p
        X.append(x.astype(np.float16)); SC.append(sc); BO.append(bo); BL.append(bl); AP.append(a)
        INFO.append((gids.index(gid), b, KINDS.index(kind), cx, cy))

    for gid in gids:
        lab = mv.labels[gid]
        for b, t in mv.traces(gid, ("full", "partial", "no_tube", "burst")):
            if not mv.lo <= b <= mv.hi:
                continue
            gx, gy, r = mv.grain_at(gid, b, t)
            j = lambda s: rng.uniform(-s, s)
            if t["state"] in ("no_tube", "burst"):
                add(b, gx + j(24), gy + j(24), "no_tube", gid)
                continue
            path = np.asarray(t["path_xy_ref"], float)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
            u = rng.uniform(0, s[-1])
            pp = (np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1]))
            if t["state"] == "full":
                add(b, gx + j(24), gy + j(24), "full_grain", gid)
                add(b, path[-1][0] + j(40), path[-1][1] + j(40), "full_apex", gid)
                add(b, pp[0] + j(30), pp[1] + j(30), "full_path", gid)
            else:
                add(b, gx + j(24), gy + j(24), "partial", gid)
                add(b, pp[0] + j(30), pp[1] + j(30), "partial", gid)
        on = lab.get("onset") or {}
        if on.get("verdict") == "emerged_within":
            for k in NEG_BACK:
                b = int(on["last_absent_bin"]) - k
                if b < mv.lo:
                    continue
                gx, gy, r = mv.grain_at(gid, b)
                add(b, gx + rng.uniform(-24, 24), gy + rng.uniform(-24, 24), "pre_onset", gid, (gx, gy, r + 40))
    info = np.array(INFO, np.float32)
    out = OUT / f"samples_{name}.npz"
    np.savez_compressed(out, x=np.stack(X), scored=np.packbits(np.stack(SC), axis=-1),
                        body=np.packbits(np.stack(BO), axis=-1), blind=np.packbits(np.stack(BL), axis=-1),
                        apex=np.stack(AP), info=info,
                        gids=np.array(gids), scale=mv.scale)
    kinds = {k: int((info[:, 2] == i).sum()) for i, k in enumerate(KINDS)}
    n_ap = int(np.isfinite(np.stack(AP)[:, :, 0]).sum())
    log(f"{name}: {len(X)} extractions {kinds}, apexes {n_ap}, scales {np.round(mv.scale, 2)}, "
        f"{time.time() - t0:.0f} s -> {out.name} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        build(m)
