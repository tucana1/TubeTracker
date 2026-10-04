"""A first reading as the guide of a second pass (cands.py --guide): per grain, the reading's emergence angle (where the
tube left the rim when it was young: the circular mean of its tip's angle round the grain over its first readings of
at most ``YOUNG_PX``) and, per bin it chose a candidate, that tip and body (grain frame).

    python -m prototypes.tip_trajectory.guide ld m2 m1 --setting tune_ld_c.json [--cands cands_{movie}.pkl]
Output: OUT/guide_<movie>.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np

from . import dp
from .common import OUT

YOUNG_PX = 12.0
N_YOUNG = 4


def guide(doc: dict, p: dict) -> dict:
    p = dp.with_speed(doc, dp.with_scale(doc, p))
    rs, nb = doc["rs"], doc["nb"]
    out = {}
    for gid, G in doc["grains"].items():
        choice, L = dp.viterbi(G, rs, nb, p)
        tips, bodies, young, first = {}, {}, [], None
        for i, k in enumerate(choice):
            if k < 0:
                continue
            B = G["bins"][rs + i]
            t = B["tips"][k].astype(float)
            tips[rs + i] = (float(t[0]), float(t[1]))
            bodies[rs + i] = B["bodies"][k] / 10.0
            a = math.atan2(t[1] - G["y"], t[0] - G["x"])
            if first is None:
                first = float(B["F"][k, dp.FI["theta"]])
            if L[i] <= YOUNG_PX and len(young) < N_YOUNG:
                young.append(a)
        theta0 = (float(np.angle(np.mean(np.exp(1j * np.array(young))))) if young else first)
        out[gid] = {"theta0": theta0, "tips": tips, "bodies": bodies}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movies", nargs="+")
    ap.add_argument("--setting", default="tune_ld_c.json")
    ap.add_argument("--cands", default="cands_{movie}.pkl")
    ap.add_argument("--out", default="guide_{movie}.pkl")
    a = ap.parse_args(argv)
    p = {**dp.DEFAULT, **json.loads((OUT / a.setting).read_text())["best"]}
    for m in a.movies:
        doc = dp.load(m, a.cands.format(movie=m))
        g = guide(doc, p)
        out = a.out.format(movie=m)
        with open(OUT / out, "wb") as fh:
            pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n = sum(1 for v in g.values() if v["theta0"] is not None)
        print(f"{m}: {len(g)} grains, {n} with an emergence angle -> {out}", flush=True)


if __name__ == "__main__":
    main()
