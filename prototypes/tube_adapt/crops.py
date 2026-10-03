"""Training crops of movie 1's training grains for one fold: the traced crops (``prototypes.tube_net.realdata``
version 3 with movie 1's options: ``--flat-cap --over-grain 0.6 --follow``, only these grains) and the propagated
crops (``propagate.py``).

    python -m prototypes.tube_adapt.crops k2f0 [k2f1 k3f0 ...]   # k<K>f<i>: K folds, fold i held out
    python -m prototypes.tube_adapt.crops --grains g009 g011 --name sub5_k2f0 --trace-only

Shards go to runs/research/tube_adapt/shards/{trace,prop}_<name>.npz.
"""

from __future__ import annotations

import argparse
import re

from prototypes.tube_adapt.common import OUT, folds, offsets, write_subset


def fold_grains(name: str) -> tuple[list[str], list[str]]:
    """(training grains, held-out grains) of fold ``k<K>f<i>``."""
    m = re.fullmatch(r"k(\d)f(\d)", name)
    k, i = int(m.group(1)), int(m.group(2))
    fs = folds(k)
    return sorted(g for j, f in enumerate(fs) if j != i for g in f), fs[i]


def build_trace(grains: list[str], out, log=print):
    from prototypes.tube_net import realdata
    lab = write_subset("m1", grains, OUT / "labels" / f"{out.stem}.json")
    off = offsets("m1")
    realdata.REAL = {**realdata.REAL, "m1": (realdata.REAL["m1"][0], str(lab))}
    realdata.grain_offsets = lambda movie: {g: off[g] for g in grains if g in off}
    return realdata.build("m1", out, flat_cap=True, over_grain=0.6, follow=True, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="*")
    ap.add_argument("--grains", nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--trace-only", action="store_true")
    a = ap.parse_args(argv)
    jobs = [(f, fold_grains(f)[0]) for f in a.folds] + ([(a.name, a.grains)] if a.grains else [])
    for name, grains in jobs:
        print(f"{name}: training grains {' '.join(grains)}", flush=True)
        build_trace(grains, OUT / "shards" / f"trace_{name}.npz")
        if not a.trace_only:
            from prototypes.tube_adapt.propagate import build
            build(OUT / "shards" / f"prop_{name}.npz", grains)


if __name__ == "__main__":
    main()
