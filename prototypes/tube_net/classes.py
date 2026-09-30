"""FULL-trace length hits by trace length (young <= 8 px, 8-60 px, long > 60 px) for bench dumps, paired against a
baseline dump with a bootstrap over grains.

    python -m prototypes.tube_net.classes MOVIE BASELINE.json NAME=DUMP.json [NAME=DUMP.json ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

CLASSES = (("young<=8", 0, 8), ("8-60", 8, 60), (">60", 60, 1e9))


def hits(dump: dict, movie: str) -> dict:
    out = {}
    for e, h, gid, fr in dump[movie]["errs"]:
        out[(gid, fr)] = (abs(e) <= max(2.0, 0.1 * h), h)
    return out


def table(movie: str, base: dict, runs: dict[str, dict], n_boot: int = 4000) -> str:
    hb = hits(base, movie)
    rng = np.random.default_rng(0)
    lines = [f"{'':28s}" + "".join(f"{c:>26s}" for c, _, _ in CLASSES) + f"{'all':>26s}"]
    for name, d in [("baseline", base)] + list(runs.items()):
        h = hits(d, movie)
        cells = []
        for c, lo, hi in CLASSES + (("all", -1, 1e9),):
            keys = [k for k, (_, hl) in hb.items() if lo < hl <= hi]
            n = sum(h[k][0] for k in keys if k in h)
            if name == "baseline":
                cells.append(f"{n:>3d}/{len(keys):<3d}{'':19s}")
                continue
            grains = sorted({k[0] for k in keys})
            diff = {g: sum(int(h[k][0]) - int(hb[k][0]) for k in keys if k[0] == g and k in h) for g in grains}
            d_arr = np.array([diff[g] for g in grains])
            idx = rng.integers(0, len(grains), (n_boot, len(grains)))
            lo_ci, hi_ci = np.percentile(d_arr[idx].sum(axis=1), [2.5, 97.5])
            cells.append(f"{n:>3d}/{len(keys):<3d} {d_arr.sum():+3d} ({lo_ci:+.0f}..{hi_ci:+.0f})    ")
        lines.append(f"{name:28s}" + "".join(f"{c:>26s}" for c in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    movie, base = sys.argv[1], json.load(open(sys.argv[2]))
    runs = {kv.split("=", 1)[0]: json.load(open(kv.split("=", 1)[1])) for kv in sys.argv[3:]}
    print(table(movie, base, runs))
