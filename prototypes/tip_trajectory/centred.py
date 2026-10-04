"""Where the body runs (post-pass): the cheapest route hugs the inside of a curving tube, the annotator traces its
middle; on a tube that turns by an angle a, the two differ by about half the tube's width x a (~11 px for a U-turn).
For each chosen reading, the body is moved onto the middle of the map's band across it (``learned._band_centres``,
median-filtered along the body, at most ``MAX_SHIFT`` px) and ``w_mid`` x (centred length - raw length) is added.

    python -m prototypes.tip_trajectory.centred ld --w 0.5 1.0
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from sparsetrack.learned import _band_centres, _running_median

from . import dp
from .cands import resample
from .common import OUT, PROB, baseline, cache_dir, labels

MAX_SHIFT, REACH, SMOOTH = 4.0, 8.0, 3


def normals(q: np.ndarray) -> np.ndarray:
    t = np.gradient(q, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    return np.stack([-t[:, 1], t[:, 0]], 1)


def mid_corrections(doc: dict, movie: str, choices: dict, prob_dir=None) -> dict:
    from pathlib import Path
    prob = np.load(Path(prob_dir) / "bins.npy" if prob_dir else cache_dir(movie) / PROB / "bins.npy", mmap_mode="r")
    offs = np.arange(-REACH, REACH + 1e-6, 0.5)
    out = {}
    for gid, G in doc["grains"].items():
        res = {}
        for i, k in enumerate(choices[gid]):
            if k < 0:
                continue
            b = doc["rs"] + i
            body = G["bins"][b]["bodies"][k] / 10.0 + G["drift"][b]  # reference coordinates
            if len(body) < 4:
                res[b] = 0.0
                continue
            q, s = resample(body, 1.0)
            n = normals(q)
            pts = (q[:, None, :] + offs[None, :, None] * n[:, None, :] - 0.5).astype(np.float32)  # -> pixel centres
            P = np.asarray(prob[b], np.float32) / 250.0
            prof = cv2.remap(P, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            c = _band_centres(prof, offs, 0.35, 14.0)
            ok = np.isfinite(c)
            if ok.sum() < 3:
                res[b] = 0.0
                continue
            idx = np.arange(len(c))
            c = np.interp(idx, idx[ok], c[ok])
            c = np.clip(_running_median(c, SMOOTH), -MAX_SHIFT, MAX_SHIFT)
            qc = q + c[:, None] * n
            res[b] = float(np.sum(np.hypot(*np.diff(qc, axis=0).T)) - s[-1])
        out[gid] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--w", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--cands", default="cands_{movie}.pkl")
    a = ap.parse_args(argv)
    p = {**dp.DEFAULT, **json.loads((OUT / "tune_ld_c.json").read_text())["best"]}
    doc = dp.load(a.movie, a.cands.format(movie=a.movie))
    q = dp.with_speed(doc, dp.with_scale(doc, p))
    choices = {gid: dp.viterbi(G, doc["rs"], doc["nb"], q)[0] for gid, G in doc["grains"].items()}
    corr = mid_corrections(doc, a.movie, choices)
    v = np.array([x for g in corr.values() for x in g.values()])
    print(f"{a.movie}: centred minus raw body length: median {np.median(v):+.2f} px, p10 {np.percentile(v, 10):+.2f}, "
          f"p90 {np.percentile(v, 90):+.2f}", flush=True)
    lab, base = labels(a.movie), baseline(a.movie)
    pg_b = dp.per_grain(dp.score(lab, base))
    ref = None
    for w in [0.0] + list(a.w):
        dp.BACK = {gid: {b: w * x for b, x in g.items()} for gid, g in corr.items()} if w else None
        ev = dp.evaluate(a.movie, doc, p)
        pg = dp.per_grain(ev["rep"])
        ref = pg if ref is None else ref
        pb, pr = dp.paired(pg_b, pg), dp.paired(ref, pg)
        print(f"  w_mid {w}: lengths {ev['lengths']}/{ev['n']} l&t {ev['lt']} | vs 0.8.8 len {pb[1][0]:+d} "
              f"[{pb[1][1]:+.0f}, {pb[1][2]:+.0f}] | vs w 0 len {pr[1][0]:+d} [{pr[1][1]:+.0f}, {pr[1][2]:+.0f}] "
              f"l&t {pr[2][0]:+d}", flush=True)
    dp.BACK = None


if __name__ == "__main__":
    main()
