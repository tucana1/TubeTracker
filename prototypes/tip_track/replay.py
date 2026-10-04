"""The tracker's own code (sparsetrack.tipdet.apply) on the stored 0.8.8 readings and stored detector maps: what
analyze() would produce with the tipdet options, without re-running the readers. Scores against 0.8.8.

    python -m prototypes.tip_track.replay --onset later --young
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace

import numpy as np

from sparsetrack import stack
from sparsetrack.analyze import Params, exit_edge
from sparsetrack.render import Renderer
from sparsetrack.tipdet import GrainTipMaps, apply, grain_positions

from .common import DATA, HALF, MOVIES, baseline, load_maps
from .young import compare


def stored_maps(mp: np.ndarray, cen: np.ndarray, rs: int, nb: int) -> GrainTipMaps:
    gm = object.__new__(GrainTipMaps)
    gm.half, gm.lo, gm.hi = HALF, rs + 7, nb - 2
    gm.maps, gm.centre = mp.astype(np.float32) / 250.0, cen
    return gm


def run(movie: str, p: Params, log=print) -> dict:
    bins, meta = stack.load(DATA / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    maps = load_maps(movie)
    doc = copy.deepcopy(baseline(movie))
    moved = young = 0
    for res in doc["grains"]:
        if res["id"] not in maps:
            continue
        pos = grain_positions(res, rs, nb)
        gm = stored_maps(*maps[res["id"]], rs, nb)
        early = np.mean([np.nan_to_num(R.crop(k, float(pos[rs][0]), float(pos[rs][1]), 64)) for k in range(rs, rs + 3)],
                        axis=0)
        edge = lambda th, early=early, r=float(res["r"]): exit_edge(  # noqa: E731
            early, 63.5, r, math.radians(5.0 * (int(round(math.degrees(th) % 360 / 5.0)) % 72)))
        s = apply(res, gm, pos, edge, meta, p)
        moved += bool(s["moved_onset_bins"])
        young += s["young_bins"] > 0
    log(f"{movie}: onsets moved on {moved} grains, young lengths on {young}")
    return compare(movie, doc, log, tag="replay")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset", default="later")
    ap.add_argument("--young", action="store_true")
    ap.add_argument("--late", type=int, default=10)
    ap.add_argument("movies", nargs="*", default=["ld", "m2", "m1"])
    a = ap.parse_args()
    prm = replace(Params(), tipdet_model="x", tipdet_onset=a.onset, tipdet_young=a.young, tipdet_late_bins=a.late)
    out = [run(m, prm) for m in a.movies]
    print(json.dumps([{k: v for k, v in r.items() if k in ("movie", "all", "young", "onset")} for r in out], default=str))
