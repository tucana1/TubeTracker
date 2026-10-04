"""Shared pieces of the tip-detector-in-the-tracker study: movies, fold models, baseline readings, stored maps."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("TT_DATA", "/Users/joshjiang/Documents/TubeTracker"))  # runs/ lives in the main checkout
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
# each movie's held-out fold model (named by its training movies)
FOLD = {"ld": "tip2_m2m1.pt", "m2": "tip2_ldm1.pt", "m1": "tip2_ldm2.pt"}
MODEL_DIR = DATA / "runs/research/tip_detector"
OUT = Path(os.environ.get("TT_TIPTRACK_OUT", "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/"
                          "eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/td"))  # stored maps (temporary)
BASE = Path(os.environ.get("TT_BASE088", "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/"
                           "eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/bt/base088"))
HALF = 64


def model_path(movie: str) -> Path:
    return MODEL_DIR / FOLD[movie]


def labels(movie: str) -> dict:
    return json.loads((REPO / MOVIES[movie][1]).read_text())


def baseline(movie: str) -> dict:
    return json.loads((BASE / f"{movie}_real_0" / "predictions.json").read_text())


def positions(res: dict, rs: int, nb: int) -> np.ndarray:
    """(nb, 2) reference position of a reading's grain per bin: its place + its drift (held before the start)."""
    pos = np.tile(np.array([res["x"], res["y"]], float), (nb, 1))
    if res.get("drift"):
        d = np.asarray(res["drift"]["xy"], float)
        d = np.nan_to_num(d)
        pos[rs:rs + len(d)] += d
        pos[:rs] += d[0]
    return pos


def load_maps(movie: str) -> dict:
    """{gid: (maps uint8 (nb, S, S) = P x 250, centre (nb, 2))}."""
    z = np.load(OUT / f"maps_{movie}.npz")
    out = {}
    for k in z.files:
        if k.endswith("_maps"):
            g = k[:-5]
            out[g] = (z[k], z[g + "_centre"])
    return out
