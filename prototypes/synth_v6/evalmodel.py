"""Probability caches for a model on both labelled movies (next to their caches, as SparseTrack builds them)
and the pixel check against the human traces, side by side with the shipped model.

    python -m prototypes.synth_v6.evalmodel runs/synth_v6/tubes_synth_v6ft.pt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sparsetrack import learned

from .recall import fmt, measure, summarise

REPO = Path(__file__).resolve().parents[2]
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json")}


def main(model: str) -> None:
    stem = Path(model).stem
    for mv, (cache, labels) in MOVIES.items():
        t0 = time.time()
        pc = learned.prob_cache(REPO / cache, REPO / model, log=print)
        rows = measure(pc, REPO / labels)
        s = summarise(rows)
        out = REPO / f"runs/synth_v6/recall_{mv}_{stem}.json"
        out.write_text(json.dumps({"summary": s, "rows": rows}))
        base = json.loads((REPO / f"runs/synth_v6/recall_{mv}_tubes_synth_v1.json").read_text())["summary"]
        print(f"{mv} ({time.time() - t0:.0f} s)\n  tubes_synth_v1: {fmt('', base)}\n  {stem}: {fmt('', s)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
