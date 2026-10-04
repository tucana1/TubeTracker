"""The tip detector's maps round every labelled grain (and the census debris) of a movie, every bin, with that movie's
held-out fold model, where the 0.8.8 reading put the grain at each bin (census + its drift): what the tracker would
compute. Stored as uint8 (P x 250) in OUT/maps_<movie>.npz (temporary; ~5 MB per grain).

    python -m prototypes.tip_track.maps ld m2 m1 [--only g001 g002]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.tipdet import GrainTipMaps, load_detector

from .common import DATA, HALF, MOVIES, OUT, baseline, labels, model_path, positions


def run(movie: str, only=None, log=print):
    bins, meta = stack.load(DATA / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    net = load_detector(model_path(movie))
    pred = baseline(movie)
    L = labels(movie)
    items = [(g["id"], positions(g, rs, nb), "graded") for g in pred["grains"]]
    for gid, g in L["grains"].items():  # census debris (excluded by the annotator as not a grain): static
        if g.get("excluded") and g.get("exclude_reason") == "not_a_grain":
            items.append((gid, np.tile([g["x"], g["y"]], (nb, 1)).astype(float), "debris"))
    if only:
        items = [it for it in items if it[0] in only]
    out = {}
    t0 = time.time()
    for gid, pos, kind in items:
        t1 = time.time()
        gm = GrainTipMaps(R, net, pos, half=HALF)
        out[f"{gid}_maps"] = np.clip(np.round(gm.maps.astype(np.float32) * 250), 0, 250).astype(np.uint8)
        out[f"{gid}_centre"] = gm.centre
        log(f"  {movie} {gid} ({kind}): {time.time() - t1:.1f} s, max {float(gm.maps.max()):.2f}")
    out["_kinds"] = np.array([f"{gid}:{kind}" for gid, _, kind in items])
    np.savez(OUT / f"maps_{movie}.npz", **out)
    log(f"{movie}: {len(items)} grains in {time.time() - t0:.0f} s -> maps_{movie}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movies", nargs="+")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for m in a.movies:
        run(m, a.only)
