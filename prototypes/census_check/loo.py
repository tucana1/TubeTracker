"""The census check judged leave one movie out: fitted on two movies' judgements, tested on the third.

    python -m prototypes.census_check.loo [--json runs/census_check/loo.json]

Grains: each movie's judged census grains that are isolated and away from the edge (the grains the product's
population statistics use, and the ones m2's and m1's random samples were drawn from): ld 32 (4 excluded), m2 30 (6),
m1 41 (11). The census's own clumps (ld g018-g020, g023) are already outside the product's population.

Variants (all fixed before the held-out results were seen; every one is reported): logistic regression (L2, C=1,
balanced class weights) on three feature sets - ALL (the 14 features of ``sparsetrack.census_check``), SMALL
(disc_dark, stay, mass_area, outer_dark, ring_rel: one feature per kind of false grain seen) and TWO (disc_dark,
stay) - each with features standardised over the training grains ("global") or per movie over its whole census
(median / IQR of every census grain, labels unused: "movie"). The flagging threshold is the one that maximises F1 on
the two training movies. Per held-out movie: AUROC, flagged, precision, recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sparsetrack.census_check import FEATURES

REPO = Path(__file__).resolve().parents[2]
SETS = {"ALL": FEATURES, "SMALL": ("disc_dark", "stay", "mass_area", "outer_dark", "ring_rel"),
        "TWO": ("disc_dark", "stay")}
MOVIES = ("ld", "m2", "m1")


def movie_scaled(rows: list[dict], feats) -> dict:
    """(movie, id) -> features standardised by the movie's whole census (median / IQR; no labels used)."""
    out = {}
    for mv in MOVIES:
        s = [r for r in rows if r["movie"] == mv]
        X = np.array([[r[f] for f in feats] for r in s], float)
        med = np.median(X, axis=0)
        iqr = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1e-3)
        for r, x in zip(s, (X - med) / iqr):
            out[(mv, r["id"])] = x
    return out


def population(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["judged"] and r["isolated"] and not r["border"]]


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(p):
        f = p >= t
        tp = float((f & y).sum())
        f1 = 2 * tp / max(float(f.sum() + y.sum()), 1.0)
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def fit(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit((X - mu) / sd, y)
    return clf, mu, sd


def run(rows: list[dict], feats, scaling: str) -> dict:
    pop = population(rows)
    ms = movie_scaled(rows, feats) if scaling == "movie" else None
    X = np.array([ms[(r["movie"], r["id"])] if ms else [r[f] for f in feats] for r in pop], float)
    y = np.array([r["excluded"] for r in pop])
    mv = np.array([r["movie"] for r in pop])
    res = {}
    for held in MOVIES:
        tr, te = mv != held, mv == held
        clf, mu, sd = fit(X[tr], y[tr])
        ptr = clf.predict_proba((X[tr] - mu) / sd)[:, 1]
        thr = best_threshold(ptr, y[tr])
        p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        f = p >= thr
        tp = int((f & y[te]).sum())
        ids = [r["id"] for r, t in zip(pop, te) if t]
        res[held] = {"n": int(te.sum()), "excluded": int(y[te].sum()), "auroc": float(roc_auc_score(y[te], p)),
                     "flagged": int(f.sum()), "tp": tp, "precision": tp / max(int(f.sum()), 1),
                     "recall": tp / max(int(y[te].sum()), 1), "threshold": thr,
                     "flagged_ids": [i for i, x in zip(ids, f) if x], "p": dict(zip(ids, map(float, p)))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "runs/census_check/loo.json"))
    a = ap.parse_args(argv)
    rows = json.loads((REPO / "runs/census_check/features.json").read_text())
    out = {}
    for name, feats in SETS.items():
        for scaling in ("global", "movie"):
            r = run(rows, feats, scaling)
            out[f"{name}/{scaling}"] = r
            print(f"{name:5s} {scaling:6s} " + " | ".join(
                f"{mv}: AUROC {r[mv]['auroc']:.2f}, flags {r[mv]['flagged']} ({r[mv]['tp']} right) of {r[mv]['n']} "
                f"(excluded {r[mv]['excluded']}): precision {r[mv]['precision']:.2f} recall {r[mv]['recall']:.2f}"
                for mv in MOVIES))
    Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
