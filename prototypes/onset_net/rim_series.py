"""The tip detector's rim response for the same grains, to compare its onset rules with the onset network: the
detector's maps round every labelled grain and the census debris, every bin, with that movie's held-out fold model,
where the 0.8.8 reading put the grain (as prototypes/tip_track/maps.py), reduced at once to the peaks within r + 60 px
(as prototypes/tip_track/rim.py; the maps themselves are not kept). Writes OUT/tiptrack/rim_<movie>.json in
tip_track's format, so its onset rules run on it with TT_TIPTRACK_OUT=OUT/tiptrack.

    python -m prototypes.onset_net.rim_series ld m2 m1
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.tipdet import GrainTipMaps, load_detector

from prototypes.tip_track.rim import peaks_of

from .common import FOLD_TIPDET, MOVIES, OUT, REPO, baseline, labels, positions

HALF = 64
TD = OUT / "tiptrack"
TD.mkdir(exist_ok=True)


def run(movie: str, log=print):
    t0 = time.time()
    bins, meta = stack.load(REPO / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs, nb, fpb = int(meta.get("ref_start", 0)), int(meta["n_bins"]), int(meta["frames_per_bin"])
    lo, hi = rs + 7, nb - 2
    net = load_detector(REPO / "runs/research/tip_detector" / FOLD_TIPDET[movie])
    L = labels(movie)
    items = [(g["id"], positions(g, rs, nb), float(g["r"]), "graded") for g in baseline(movie)["grains"]]
    for gid, g in L["grains"].items():
        if g.get("excluded") and g.get("exclude_reason") == "not_a_grain":
            items.append((gid, np.tile([g["x"], g["y"]], (nb, 1)).astype(float), float(g["r"]), "debris"))
    out = {"movie": movie, "rs": rs, "nb": nb, "lo": lo, "hi": hi, "fpb": fpb, "grains": {}}
    for gid, pos, r, kind in items:
        t1 = time.time()
        gm = GrainTipMaps(R, net, pos, half=HALF)
        rows = []
        for b in range(lo, hi + 1):
            m = gm.maps[b].astype(np.float32)
            m = np.round(m * 250) / 250.0  # tip_track stored the maps as uint8 (P x 250)
            x0, y0 = gm.centre[b][0] - HALF + 0.5, gm.centre[b][1] - HALF + 0.5
            pk = []
            for x, y, v in peaks_of(m):
                X, Y = x0 + x, y0 + y
                d = float(np.hypot(X - pos[b][0], Y - pos[b][1]))
                if d <= r + 60:
                    pk.append([round(v, 3), round(float(X), 2), round(float(Y), 2), round(d - r, 2)])
            rows.append(pk[:12])
        out["grains"][gid] = {"kind": kind, "r": r, "pos": np.round(pos, 2).tolist(), "peaks": rows}
        log(f"  {movie} {gid} ({kind}) {time.time() - t1:.1f} s", flush=True)
    (TD / f"rim_{movie}.json").write_text(json.dumps(out))
    log(f"{movie}: {len(items)} grains in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        run(m)
