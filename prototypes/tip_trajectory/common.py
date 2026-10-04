"""Shared pieces of the global tip-trajectory study: movies, labels, the 0.8.8 baseline, detector fold models.

Data (caches, tube maps, detector models) live in the main checkout (``DATA``); outputs go to ``OUT`` (scratch).
Coordinates: reference (x, y) as everywhere in SparseTrack; a full-frame map pixel (i, j) covers reference
[j, j + 1) x [i, i + 1), its centre at (j + 0.5, i + 0.5).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("TT_DATA", "/Users/joshjiang/Documents/TubeTracker"))
OUT = Path(os.environ.get("TT_GT_OUT", "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/"
                          "eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/gt"))
BASE = Path(os.environ.get("TT_BASE088", "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/"
                           "eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/bt/base088"))
MOVIES = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
LABELS = {m: REPO / f"benchmark/labels/{m}_v1.json" for m in MOVIES}
# each movie's held-out tip-detector fold (named by its training movies)
FOLD = {"ld": "tip2_m2m1.pt", "m2": "tip2_ldm1.pt", "m1": "tip2_ldm2.pt"}
DET_DIR = DATA / "runs/research/tip_detector"
PROB = "prob_tubes_bn_real_ld_m2"  # the shipped tube maps (0.8.0 on)


def cache_dir(movie: str) -> Path:
    return DATA / MOVIES[movie]


def labels(movie: str) -> dict:
    return json.loads(LABELS[movie].read_text())


def baseline(movie: str) -> dict:
    return json.loads((BASE / f"{movie}_real_0" / "predictions.json").read_text())


def scored_grains(lab: dict) -> list[str]:
    """The grains the scorer scores (not excluded, isolated)."""
    return sorted(g for g, v in lab["grains"].items() if not v.get("excluded") and v.get("isolated", True))


def drift_per_bin(res: dict, rs: int, nb: int) -> np.ndarray:
    """(nb, 2) drift (x, y) of a reading's grain at every bin (zero where the reading gives none; held before the
    first and over unknown bins)."""
    d = np.zeros((nb, 2))
    if res.get("drift"):
        xy = np.asarray(res["drift"]["xy"], float)
        fin = np.isfinite(xy).all(axis=1)
        if fin.any():
            idx = np.arange(len(xy))
            xy = np.stack([np.interp(idx, idx[fin], xy[fin, k]) for k in (0, 1)], axis=1)
            d[rs:rs + len(xy)] = xy[: nb - rs]
            d[:rs] = xy[0]
    return d


def length_class(L: float) -> str:
    return "young" if L < 15 else ("long" if L >= 60 else "mid")
