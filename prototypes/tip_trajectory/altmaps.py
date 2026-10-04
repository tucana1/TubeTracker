"""Tube maps of a movie from another tube network (e.g. the label-free self-trained m1 network of prototypes/self_train),
built as learned.prob_cache does but into the scratch folder (OUT/prob_<model stem>_<movie>), for a secondary variant
of the reader: ``TT_GT_PROB=<that dir> python -m prototypes.tip_trajectory.cands m1``.

    python -m prototypes.tip_trajectory.altmaps m1 /path/to/model.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from sparsetrack import learned, stack

from .common import OUT, cache_dir


def main(movie: str, model: str) -> Path:
    bins, meta = stack.load(cache_dir(movie))
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    net = learned.load_model(model)
    early = np.mean([learned._registered(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([learned._registered(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    out = OUT / f"prob_{Path(model).stem}_{movie}"
    out.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out / "bins.npy", mode="w+", dtype=np.uint8, shape=bins.shape)
    t0 = time.time()
    for b in range(nb):
        arr[b] = np.round(learned.tube_probability(net, learned._registered(bins, shifts, b), early, late)
                          * learned.P_SCALE).astype(np.uint8)
    arr.flush()
    print(f"{movie}: {nb} bins in {time.time() - t0:.0f} s -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
