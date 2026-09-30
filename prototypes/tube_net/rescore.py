"""Score a bench dump's predictions again on another labels file (same analysis, e.g. movie 1's secondary scoring).

    python -m prototypes.tube_net.rescore DUMP.json MOVIE=LABELS.json [...] --out NEW_DUMP.json

The dump is ``bench.py`` / ``scripts/synth_bench.py --dump-real`` format; each movie's ``pred`` path is scored with
``synth_bench.score_real``'s rules against the given labels file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def rescore(entry: dict, labels: str | Path) -> dict:
    import synth_bench as sb
    from sparsetrack.evaluate import load, score
    pred = json.loads(Path(entry["pred"]).read_text())
    rep = score(load(labels), pred)
    grains = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        grains[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                              "len_hit": sum(sb._within(f["error"], f["human"]) for f in full), "len_n": len(full),
                              "both_hit": sum(sb._within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                              <= max(5.0, 0.1 * f["human"]) for f in full)}
    return {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
            "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
            "len_med": rep["length_full"]["median_abs_error"], "len_bias": rep["length_full"]["bias"],
            "both": rep["tips"]["length_and_tip"],
            "errs": [(f["error"], f["human"], r["grain"], f["frame"]) for r in rep["rows"] for f in r.get("full", [])],
            "grains": grains, "pred": entry["pred"], "labels": str(labels),
            "population": rep.get("population"), "growth": {k: v for k, v in (rep.get("growth") or {}).items()
                                                            if not isinstance(v, (list, dict))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("labels", nargs="+", help="MOVIE=LABELS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = json.loads(Path(a.dump).read_text())
    for kv in a.labels:
        mv, lab = kv.split("=", 1)
        d[mv] = rescore(d[mv], lab)
        g = d[mv]
        print(f"{mv} on {Path(lab).name}: onset {g['on_hit']}/{g['on_n']}, len {g['len_hit']}/{g['len_n']}, "
              f"with tip {g['both']}")
    Path(a.out).write_text(json.dumps(d, default=float))


if __name__ == "__main__":
    main()
