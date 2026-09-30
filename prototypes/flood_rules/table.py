"""One line per bench dump (``bench serve``'s QUEUE/done/NAME.json, or a synth_bench --dump-real file), paired against
one or two baselines (95% bootstrap intervals over grains).

    python -m prototypes.flood_rules.table MOVIE BASE1.json [--vs BASE2.json] DUMP.json [DUMP.json ...]

"acc" = length hits on grains whose flood started more than 10 bins before the annotator's last-absent bin: by its
first claim / by its reported onset (after the onset look-back); "-" where the dump does not record the flood's own
reading (synth_bench dumps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .bench import paired
from .sweep import LABELS, REPO, accidental


def row(movie: str, d: dict, bases: list[dict], labels: dict) -> str:
    r = d[movie]
    g = r.get("growth") or {}
    acc = "  -/- " if "own" not in r else "{:>3d}/{:<2d}".format(*accidental(movie, r, labels)[:2])
    txt = (f"len {r['len_hit']:3d}/{r['len_n']} len&tip {r['both']:3d} on {r['on_hit']:2d}/{r['on_n']:2d} "
           f"acc {acc} rate {g.get('rate_within', '-')}/{g.get('grains_with_rate', '-')} "
           f"r {g.get('rate_pearson') or float('nan'):.2f} bias {r['len_bias']:+6.2f}")
    for b in bases:
        cells = []
        for key in ("len_hit", "both_hit", "onset_hit"):
            dd, lo, hi = paired(b[movie], r, key)
            cells.append(f"{dd:+3d} ({lo:+.0f}..{hi:+.0f})")
        txt += " | " + " ".join(cells)
    return txt


if __name__ == "__main__":
    args = sys.argv[1:]
    movie = args.pop(0)
    labels = json.loads((REPO / LABELS[movie]).read_text())
    bases = [json.loads(Path(args.pop(0)).read_text())]
    if args and args[0] == "--vs":
        args.pop(0)
        bases.append(json.loads(Path(args.pop(0)).read_text()))
    print(f"{'':28s} {'':80s} | vs base: lengths, len&tip, onsets" + (" | vs 2nd base" if len(bases) > 1 else ""))
    for f in args:
        d = json.loads(Path(f).read_text())
        if movie in d:
            print(f"{Path(f).stem:28s} {row(movie, d, bases, labels)}")
