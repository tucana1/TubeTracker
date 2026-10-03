"""Per-bin grain offsets (the labelling tool's own grain following, ``Bench.follow``) for every labelled grain, saved
once per movie: where each grain is at bins without a trace (before the onset; for the linking check).

    python -m prototypes.tip_detector.offsets ld m2 m1
"""
from __future__ import annotations

import sys
import time

import numpy as np

from prototypes.tube_net.realdata import grain_offsets

from .common import OUT


def main(movies):
    for m in movies:
        t0 = time.time()
        fol = grain_offsets(m)
        np.savez_compressed(OUT / f"offsets_{m}.npz", **{g: np.asarray(v, np.float32) for g, v in fol.items()})
        mv = [float(np.max(np.hypot(*np.asarray(v).T))) for v in fol.values()]
        print(f"{m}: {len(fol)} grains in {time.time() - t0:.0f} s; max move median {np.median(mv):.1f} px")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ld", "m2", "m1"])
