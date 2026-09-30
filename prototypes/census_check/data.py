"""Census-check features of every census grain of the three labelled movies, with the annotator's judgement.

    python -m prototypes.census_check.data            # -> runs/census_check/features.json

Per grain: ``sparsetrack.census_check.features`` (reads each cache; writes nothing there), the census's own layout
(isolated, border, clump size) and the annotator's judgement from benchmark/labels/<movie>_v1.json: ``judged``
(the annotator saw it: every ld grain; in m2 and m1 the random sample of isolated grains away from the edge and its
replacements - "not_sampled" grains were never judged), ``excluded`` and the reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from sparsetrack import census_check

REPO = Path(__file__).resolve().parents[2]
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
OUT = REPO / "runs/census_check/features.json"


def rows(movie: str) -> list[dict]:
    cache, labels = (REPO / p for p in MOVIES[movie])
    L = json.loads(labels.read_text())
    census = json.loads((cache / "grains.json").read_text())
    feats = census_check.features(cache, census)
    out = []
    for g in census["grains"]:
        lg = L["grains"].get(g["id"], {})
        reason = lg.get("exclude_reason") if lg.get("excluded") else None
        out.append({"movie": movie, "id": g["id"], "x": g["x"], "y": g["y"], "r": g["r"],
                    "isolated": bool(g.get("isolated")), "border": bool(g.get("border")),
                    "clump_size": int(g.get("clump_size", 1)), "judged": reason != "not_sampled",
                    "excluded": reason not in (None, "not_sampled"), "reason": reason,
                    "onset": ((L["labels"].get(g["id"]) or {}).get("onset") or {}).get("verdict"),
                    **feats[g["id"]]})
    return out


def load() -> list[dict]:
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    allrows = [r for mv in MOVIES for r in rows(mv)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(allrows, indent=0))
    j = [r for r in allrows if r["judged"]]
    print(f"{len(allrows)} census grains, {len(j)} judged, {sum(r['excluded'] for r in j)} excluded -> {OUT}")
