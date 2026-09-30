"""FULL traces whose hit changed between two bench dumps (length, and length and tip).

    python -m prototypes.flood_rules.traces MOVIE BASE.json NEW.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def rows(movie: str, d: dict) -> dict:
    pred = {g["id"]: g for g in json.loads(Path(d[movie]["pred"]).read_text())["grains"]}
    from sparsetrack.evaluate import score
    labels = json.loads((REPO / {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json"}[movie]).read_text())
    rep = score(labels, {"grains": list(pred.values())})
    out = {}
    for r in rep["rows"]:
        for f in r.get("full", []):
            tol = max(2.0, 0.1 * f["human"])
            out[(r["grain"], f["frame"])] = (f["human"], f["pred"], abs(f["error"]) <= tol,
                                             abs(f["error"]) <= tol and f.get("tip_error", 1e9) <= max(5.0, 0.1 * f["human"]))
    return out


if __name__ == "__main__":
    movie = sys.argv[1]
    a, b = (rows(movie, json.loads(Path(p).read_text())) for p in sys.argv[2:4])
    fpb = 300
    for k in sorted(a):
        ha, pa, la, ta = a[k]
        _, pb, lb, tb = b[k]
        if (la, ta) != (lb, tb):
            print(f"{k[0]}@{k[1] // fpb:3d} human {ha:6.1f}: {pa:7.1f} {'L' if la else '-'}{'T' if ta else '-'} -> "
                  f"{pb:7.1f} {'L' if lb else '-'}{'T' if tb else '-'}")
