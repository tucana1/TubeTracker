"""How many of a run's FULL-trace length hits sit on grains whose tube it started more than ``early`` bins before the
human's first visible bin (a start on the grain's own rim, not on the tube): hits by coincidence of a stalled
false start, rather than reads of the tube. An explanation of the paired scores, not a scoring rule.

    python -m prototypes.tube_net.falsestart MOVIE NAME=DUMP.json [NAME=DUMP.json ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .onsets import errors


def split(movie: str, dump: dict, early: int = 10) -> tuple[int, int, int]:
    err = errors(movie, dump)
    hits = [(gid, abs(e) <= max(2.0, 0.1 * h)) for e, h, gid, _ in dump[movie]["errs"]]
    fs = {g for g, e in err.items() if np.isfinite(e) and e < -early}
    n_hit = sum(h for _, h in hits)
    n_fs = sum(h for g, h in hits if g in fs)
    return n_hit, n_fs, len(fs)


if __name__ == "__main__":
    movie = sys.argv[1]
    for kv in sys.argv[2:]:
        name, path = kv.split("=", 1)
        n, f, g = split(movie, json.load(open(path)))
        print(f"{name:14s} length hits {n:3d}: {f:2d} of them on the {g} grains started > 10 bins early, "
              f"{n - f:3d} on the others")
