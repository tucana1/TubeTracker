"""Per grain and bin, the detector's peaks in the band just outside the grain's visible edge (r - 2 .. r + 25 px from
where the grain is at that bin): the rim response the onset rules read (onsets.py). Also every peak within r + 60 px
for the young-tube study (young.py). Writes OUT/rim_<movie>.json.

    python -m prototypes.tip_track.rim ld m2 m1
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import maximum_filter

from sparsetrack import stack

from .common import DATA, HALF, MOVIES, OUT, baseline, labels, load_maps, positions

BAND_IN, BAND_OUT = -2.0, 25.0   # px from the census radius
MIN_V = 0.03                     # peaks below this are not kept


def peaks_of(m: np.ndarray, min_v: float = MIN_V) -> list[tuple[int, int, float]]:
    pk = (m == maximum_filter(m, size=9, mode="constant", cval=-1.0)) & (m >= min_v)
    ys, xs = np.nonzero(pk)
    out = []
    for o in np.argsort(-m[ys, xs], kind="stable"):
        x, y = int(xs[o]), int(ys[o])
        if all((x - a) ** 2 + (y - c) ** 2 >= 16 for a, c, _ in out):
            out.append((x, y, float(m[y, x])))
    return out


def run(movie: str, log=print):
    _, meta = stack.load(DATA / MOVIES[movie][0])
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    lo, hi = rs + 7, nb - 2
    pred = {g["id"]: g for g in baseline(movie)["grains"]}
    L = labels(movie)
    maps = load_maps(movie)
    out = {"movie": movie, "rs": rs, "nb": nb, "lo": lo, "hi": hi, "fpb": int(meta["frames_per_bin"]), "grains": {}}
    for gid, (mp, cen) in maps.items():
        if gid in pred:
            res = pred[gid]
            pos, r, kind = positions(res, rs, nb), float(res["r"]), "graded"
        else:
            g = L["grains"][gid]
            pos, r, kind = np.tile([g["x"], g["y"]], (nb, 1)).astype(float), float(g["r"]), "debris"
        rows = []
        for b in range(lo, hi + 1):
            m = mp[b].astype(np.float32) / 250.0
            x0, y0 = cen[b][0] - HALF + 0.5, cen[b][1] - HALF + 0.5
            pk = []
            for x, y, v in peaks_of(m):
                X, Y = x0 + x, y0 + y
                d = float(np.hypot(X - pos[b][0], Y - pos[b][1]))
                if d <= r + 60:
                    pk.append([round(v, 3), round(float(X), 2), round(float(Y), 2), round(d - r, 2)])
            rows.append(pk[:12])
        out["grains"][gid] = {"kind": kind, "r": r, "pos": np.round(pos, 2).tolist(), "peaks": rows}
    (OUT / f"rim_{movie}.json").write_text(json.dumps(out))
    log(f"{movie}: {len(out['grains'])} grains -> rim_{movie}.json")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        run(m)
