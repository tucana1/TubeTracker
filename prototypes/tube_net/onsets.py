"""Onset errors in bins (model onset bin minus the human's first visible bin) per timed grain, for bench runs.

    python -m prototypes.tube_net.onsets MOVIE NAME=DUMP.json [NAME=DUMP.json ...]

Beyond the bench's hit count (within 2 bins): the median absolute error, how many within 10 bins, and how many
start more than 10 bins early (a false start on the grain's rim) or late / never.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json"}


def errors(movie: str, dump: dict) -> dict[str, float]:
    L = json.loads((REPO / LABELS[movie]).read_text())
    pred = {g["id"]: g for g in json.loads(Path(dump[movie]["pred"]).read_text())["grains"]}
    fpb = int(L.get("frames_per_bin", 300))
    out = {}
    for gid, lab in L["labels"].items():
        on = lab.get("onset") or {}
        if L["grains"][gid].get("excluded") or on.get("verdict") != "emerged_within" or gid not in pred:
            continue
        g = pred[gid]
        out[gid] = (g["onset_frame"] // fpb - on["first_visible_bin"]) if g.get("onset_frame") is not None else np.inf
    return out


def summary(err: dict[str, float]) -> str:
    e = np.array(list(err.values()), float)
    fin = np.isfinite(e)
    return (f"n {len(e)}: within 2 bins {int(np.sum(np.abs(e) <= 2))}, within 10 {int(np.sum(np.abs(e) <= 10))}, "
            f">10 early {int(np.sum(e < -10))}, >10 late {int(np.sum(e[fin] > 10))}, none {int(np.sum(~fin))}; "
            f"median |error| {np.median(np.abs(e)):.1f} bins")


if __name__ == "__main__":
    movie = sys.argv[1]
    runs = {kv.split("=", 1)[0]: json.load(open(kv.split("=", 1)[1])) for kv in sys.argv[2:]}
    errs = {n: errors(movie, d) for n, d in runs.items()}
    for n, e in errs.items():
        print(f"{n:16s} {summary(e)}")
    gids = sorted(set().union(*errs.values()))
    print("per grain (bins): " + "; ".join(f"{g} " + "/".join(f"{errs[n].get(g, np.nan):+.0f}" for n in errs)
                                          for g in gids))
