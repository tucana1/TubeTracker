"""Fit the census check on all three movies' judgements (the variant chosen leave one movie out: loo.py) and print
``sparsetrack.census_check.MODEL``.

    python -m prototypes.census_check.fit [--set SMALL] [--scaling global]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prototypes.census_check.loo import SETS, best_threshold, fit, population

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="SMALL", choices=tuple(SETS))
    a = ap.parse_args(argv)
    feats = SETS[a.set]
    pop = population(json.loads((REPO / "runs/census_check/features.json").read_text()))
    X = np.array([[r[f] for f in feats] for r in pop], float)
    y = np.array([r["excluded"] for r in pop])
    clf, mu, sd = fit(X, y)
    p = clf.predict_proba((X - mu) / sd)[:, 1]
    thr = best_threshold(p, y)
    model = {"features": list(feats), "coef": [round(float(c), 4) for c in clf.coef_[0]],
             "intercept": round(float(clf.intercept_[0]), 4), "mean": [round(float(v), 4) for v in mu],
             "scale": [round(float(v), 4) for v in sd], "threshold": round(float(thr), 4),
             "fitted_on": f"{len(y)} judged census-isolated grains of ld, m2, m1 ({int(y.sum())} excluded)"}
    f = p >= thr
    print(f"in-sample: flags {int(f.sum())}, right {int((f & y).sum())} of {int(y.sum())} excluded")
    print(json.dumps(model, indent=1))


if __name__ == "__main__":
    main()
