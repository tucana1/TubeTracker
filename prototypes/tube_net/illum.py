"""How far each movie's local background departs from its frame median, in network input units (20 grey levels).

    python -m prototypes.tube_net.illum [CACHE ...]      # default: the dev movie, movie 2, movie 1 (no labels used)

The network's input is (image - median of the "before" image) / 20; every training crop (96 px) was normalised by
its own median, with offsets of at most +/-0.2 units added in training. Here: the before image's median over ~96 px
round every pixel (``learned.local_background``) minus the frame's median.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sparsetrack import learned, stack

REPO = Path(__file__).resolve().parents[2]


def offsets(cache: str | Path, px: int = 96) -> np.ndarray:
    src, meta = stack.load(cache)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs = int(meta.get("ref_start", 0))
    early = np.mean([learned._registered(src, shifts, b) for b in range(rs, rs + 3)], axis=0)
    return (learned.local_background(early, px) - float(np.nanmedian(early))) / learned.IN_SCALE


if __name__ == "__main__":
    for c in sys.argv[1:] or ["runs/sparsetrack/ld", "runs/sparsetrack/m2", "runs/sparsetrack/m1"]:
        d = offsets(REPO / c)
        p = np.percentile(d, [1, 10, 50, 90, 99])
        print(f"{c}: percentiles 1/10/50/90/99 " + " ".join(f"{v:+.2f}" for v in p) +
              f"; |offset| > 0.2: {100 * np.mean(np.abs(d) > 0.2):.0f}%, > 0.5: {100 * np.mean(np.abs(d) > 0.5):.0f}%")
