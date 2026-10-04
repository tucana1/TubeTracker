"""Grain-centred crops of every bin for every labelled grain (and the census debris) of a movie, once.

    python -m prototypes.onset_net.build ld m2 m1

Each grain's crop at bin b is centred on where the grain is at bin b: census + the labelling tool's own per-bin
grain following (prototypes/tip_detector/offsets.py, `Bench.follow`, what the annotator saw); debris (excluded as
'not a grain') at its census place. 80 px, registered, float16, one memmap per movie:
OUT/crops_<movie>.npy (n_items, n_bins, 80, 80) and OUT/index_<movie>.json (per item: id, kind, r, verdict,
bracket, isolated, labels per bin, positions).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer

from .common import HALF, MOVIES, OUT, REPO, bin_labels, extract, labels

OFFSETS = REPO / "runs/research/tip_detector"


def items_of(movie: str, L: dict, nb: int) -> list[dict]:
    off = dict(np.load(OFFSETS / f"offsets_{movie}.npz"))
    items = []
    for gid, g in sorted(L["grains"].items()):
        lab = L["labels"].get(gid)
        if g.get("excluded"):
            if g.get("exclude_reason") == "not_a_grain":
                items.append({"id": gid, "kind": "debris", "r": float(g["r"]), "verdict": "debris", "la": None,
                              "fv": None, "isolated": bool(g.get("isolated", True)),
                              "pos": np.tile([g["x"], g["y"]], (nb, 1)).astype(float)})
            continue
        on = (lab or {}).get("onset") or {}
        v = on.get("verdict")
        if v not in ("emerged_within", "emerged_at_start", "no_emergence_by_end"):
            continue
        o = off.get(gid, np.zeros((nb, 2)))
        items.append({"id": gid, "kind": "grain", "r": float(g["r"]), "verdict": v, "la": on.get("last_absent_bin"),
                      "fv": on.get("first_visible_bin"), "isolated": bool(g.get("isolated", True)),
                      "pos": np.array([g["x"], g["y"]], float)[None] + np.asarray(o, float)})
    return items


def run(movie: str, log=print):
    t0 = time.time()
    bins, meta = stack.load(REPO / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs, nb, fpb = int(meta.get("ref_start", 0)), int(meta["n_bins"]), int(meta["frames_per_bin"])
    L = labels(movie)
    items = items_of(movie, L, nb)
    S = 2 * HALF
    arr = np.lib.format.open_memmap(OUT / f"crops_{movie}.npy", mode="w+", dtype=np.float16,
                                    shape=(len(items), nb, S, S))
    for i, it in enumerate(items):
        arr[i] = extract(R, it["pos"])
        it["y"] = bin_labels(it["verdict"], it["la"], it["fv"], nb).tolist()
        it["pos"] = np.round(it["pos"], 2).tolist()
    arr.flush()
    del arr
    idx = {"movie": movie, "rs": rs, "nb": nb, "fpb": fpb, "half": HALF, "items": items}
    (OUT / f"index_{movie}.json").write_text(json.dumps(idx))
    kinds = {}
    for it in items:
        kinds[it["verdict"]] = kinds.get(it["verdict"], 0) + 1
    log(f"{movie}: {len(items)} items {kinds} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        run(m)
