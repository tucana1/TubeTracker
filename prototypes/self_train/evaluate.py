"""Judge a self-trained network on all of a movie's labelled grains (no labels were used to train it), maps in memory:
``prototypes/tube_adapt/evaluate.py``'s pixel check and end to end (SparseTrack's defaults, only the network
changed), outputs in runs/research/self_train/eval/.

    python -m prototypes.self_train.evaluate pix MODEL.pt --movie m1 --out OUT.json
    python -m prototypes.self_train.evaluate e2e MODEL.pt --movie m1 --out OUT.json --tag NAME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototypes.self_train.select import OUT
from prototypes.tube_adapt import evaluate as ev
from prototypes.tube_adapt.common import SCRATCH, labels

ev.OUT = OUT  # label subsets written here, not in tube_adapt's folder
ev.SCRATCH = SCRATCH / "self_train"


def grains_of(movie: str) -> list[str]:
    L = labels(movie)
    return sorted(g for g in L["labels"] if not L["grains"][g].get("excluded"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pix", "e2e"))
    ap.add_argument("model")
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    g = grains_of(a.movie)
    if a.what == "pix":
        res = ev.pix(a.model, a.movie, g)
        from prototypes.tube_net.pixels import line
        print(f"{Path(a.model).stem} on {a.movie} ({len(res['grains'])} grains): {line(res['summary'])}", flush=True)
    else:
        res = ev.e2e(a.model, g, a.tag or Path(a.out).stem, a.movie)
        print(f"{Path(a.model).stem} on {a.movie} ({len(g)} grains): onsets {res['on_hit']}/{res['on_n']}, "
              f"lengths {res['len_hit']}/{res['len_n']}, length and tip {res['both']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
