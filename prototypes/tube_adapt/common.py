"""Shared pieces of the per-movie adaptation experiment: paths, grain folds of movie 1, the labelling tool's grain
offsets (cached), labels restricted to a set of grains, and the baseline network and predictions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/research/tube_adapt"
BASE_NET = REPO / "sparsetrack/models/tubes_bn_real_ld_m2.pt"  # 0.8.x network: ld + m2 traces, never saw movie 1
SCRATCH = Path("/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/"
               "scratchpad")
BASE_DUMP = SCRATCH / "bt/base088.json"  # 0.8.8 defaults on ld, m2, m1 (per-grain hits; synth_bench --dump-real format)
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
# replay data: 0.8.0's own training sets (prototypes/tube_net/loo3.py): its ld + m2 trace crops and one shard of each
# synthetic family (memory: the laptop has 8 GB)
V5 = "runs/learned_flood/shards/train_v5m2s10.npz"
V6 = "runs/synth_v6/shards/train_v6m1_s50.npz"
REAL_LDM2 = "runs/tube_net/shards/real3_ld.npz,runs/tube_net/shards/real3_m2.npz"


def labels(movie: str) -> dict:
    return json.loads((REPO / MOVIES[movie][1]).read_text())


def path_len(path) -> float:
    p = np.asarray(path, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) >= 2 else 0.0


def grain_table(L: dict) -> list[dict]:
    """Per labelled grain: scored FULL traces (not touching), long ones (>= 50 px), young ones (<= 8 px), longest
    traced tube (full or partial)."""
    rows = []
    for gid, lab in sorted(L["labels"].items()):
        if L["grains"][gid].get("excluded"):
            continue
        tr = lab.get("traces") or {}
        full = [t["length_px"] for t in tr.values() if t["state"] == "full" and not t.get("contact")]
        anyp = [t["length_px"] for t in tr.values() if t["state"] in ("full", "partial")
                and len(t.get("path_xy_ref") or []) >= 2]
        rows.append({"grain": gid, "full": len(full), "long": sum(x >= 50 for x in full),
                     "young": sum(x <= 8 for x in full), "longest": max(anyp) if anyp else 0.0,
                     "traces": len(anyp)})
    return rows


def folds(k: int, seed: int = 20261003, tries: int = 20000) -> list[list[str]]:
    """``k`` folds of movie 1's labelled grains, by grain, balanced on label counts only (scored FULL traces, long
    traces, young traces, traced tubes, longest tube): the best of ``tries`` random splits of equal size."""
    rows = grain_table(labels("m1"))
    ids = [r["grain"] for r in rows]
    feats = np.array([[r["full"], r["long"], r["young"], r["traces"], r["longest"] / 100.0] for r in rows], float)
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(tries):
        perm = rng.permutation(len(ids))
        parts = [perm[i::k] for i in range(k)]
        sums = np.array([feats[p].sum(axis=0) for p in parts])
        cost = float(((sums - sums.mean(axis=0)) ** 2 / (feats.var(axis=0) + 1e-9)).sum())
        if cost < best_cost:
            best, best_cost = parts, cost
    return [sorted(ids[i] for i in p) for p in best]


def subset(L: dict, grains) -> dict:
    """The labels with only ``grains``' labels kept (the census of grains is kept whole: neighbours stay obstacles)."""
    out = copy.deepcopy(L)
    out["labels"] = {g: v for g, v in L["labels"].items() if g in set(grains)}
    out.pop("retest", None)
    return out


def write_subset(movie: str, grains, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset(labels(movie), grains)))
    return path


def offsets(movie: str = "m1") -> dict:
    """grain -> (n_bins, 2) offsets from its census place, the labelling tool's own (``Bench.follow``, which gave
    every trace's ``view_offset``; ``prototypes.tube_net.realdata.grain_offsets``), cached in OUT."""
    f = OUT / f"offsets_{movie}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    from prototypes.tube_net.realdata import grain_offsets
    off = grain_offsets(movie)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **off)
    return off


if __name__ == "__main__":
    for k in (2, 3):
        fs = folds(k)
        rows = {r["grain"]: r for r in grain_table(labels("m1"))}
        for i, f in enumerate(fs):
            s = {key: sum(rows[g][key] for g in f) for key in ("full", "long", "young", "traces")}
            print(f"k={k} fold {i}: {len(f)} grains {s} longest-sum {sum(rows[g]['longest'] for g in f):.0f}: "
                  f"{' '.join(f)}")


def nested_subsets(train: list[str], sizes=(5, 10), seed: int = 20261003, tries: int = 5000) -> dict[int, list[str]]:
    """Nested subsets of a fold's training grains (5 within 10 within all), each as close as random search finds to
    its share of the fold's label counts (the learning curve's corrected grains)."""
    rows = {r["grain"]: r for r in grain_table(labels("m1"))}
    feats = {g: np.array([rows[g]["full"], rows[g]["long"], rows[g]["young"], rows[g]["traces"],
                          rows[g]["longest"] / 100.0], float) for g in train}
    total = sum(feats.values())
    var = np.var(np.array(list(feats.values())), axis=0) + 1e-9
    rng = np.random.default_rng(seed)
    out, pool = {}, list(train)
    for n in sorted(sizes, reverse=True):
        best, bc = None, np.inf
        for _ in range(tries):
            pick = list(rng.choice(pool, size=n, replace=False))
            c = float((((sum(feats[g] for g in pick)) - total * n / len(train)) ** 2 / var).sum())
            if c < bc:
                best, bc = pick, c
        out[n] = sorted(str(g) for g in best)
        pool = out[n]
    return out
