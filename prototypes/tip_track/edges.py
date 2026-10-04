"""Per grain, its visible edge round the rim (analyze.exit_edge on the before image, every 5 degrees), as an offset
from the census radius: where an annotator starts a trace. Writes OUT/edges_<movie>.json.

    python -m prototypes.tip_track.edges ld m2 m1
"""
from __future__ import annotations

import json
import sys

import numpy as np

from sparsetrack import stack
from sparsetrack.analyze import exit_edge
from sparsetrack.render import Renderer

from .common import DATA, MOVIES, OUT


def run(movie: str, log=print):
    bins, meta = stack.load(DATA / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs = int(meta.get("ref_start", 0))
    rim = json.loads((OUT / f"rim_{movie}.json").read_text())
    out = {}
    for gid, G in rim["grains"].items():
        x, y = G["pos"][rs]
        half = 64
        early = np.mean([np.nan_to_num(R.crop(k, x, y, half)) for k in range(rs, rs + 3)], axis=0)
        out[gid] = [round(exit_edge(early, half - 0.5, G["r"], np.deg2rad(a)), 2) for a in range(0, 360, 5)]
    (OUT / f"edges_{movie}.json").write_text(json.dumps(out))
    log(f"{movie}: {len(out)} grains -> edges_{movie}.json")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        run(m)
