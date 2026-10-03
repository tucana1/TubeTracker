"""Apply an O1 run's tuned settings, unchanged, to kymographs built another way (e.g. a carry sensitivity check):

    python -m prototypes.carry_front.apply --params tune_ld_auto ld/b --root runs/research/carry_front/sens_s5c30 \
        --movies m1 --variants b c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototypes.carry_front.carry import OUT
from prototypes.carry_front.o1 import baseline, load, paired, per_grain, predict, reasons, score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs=2, metavar=("TAG", "MOVIE/VARIANT"), required=True,
                    help="whose tuned settings: an O1 tag and the summary key of the tuning movie, e.g. tune_ld ld/b")
    ap.add_argument("--root", required=True)
    ap.add_argument("--movies", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["b"])
    a = ap.parse_args(argv)
    prm = json.loads((OUT / "o1" / a.params[0] / "summary.json").read_text())[a.params[1]]["params"]
    print(f"settings from {a.params[0]} {a.params[1]}: {prm}")
    for m in a.movies:
        mv, gs = load(m, Path(a.root))
        for v in a.variants:
            pred, rows = predict(mv, gs, v, prm, False, f"carry_front {v} ({a.root})")
            rep, base = score(mv.labels, pred), score(mv.labels, baseline(m))
            pc = paired(per_grain(base), per_grain(rep))
            cov = [r["cover"] for rr in rows.values() for r in rr if r["cover"] is not None]
            print(f"  {m}/{v}: vmax {pred.get('vmax_px_per_bin', prm['vmax'])} lengths "
                  f"{rep['length_full']['within_tolerance']}/{rep['length_full']['n']} (0.8.8 "
                  f"{base['length_full']['within_tolerance']}) d {pc['lengths'][0]:+d} [{pc['lengths'][1]:+.0f},"
                  f"{pc['lengths'][2]:+.0f}] | len&tip {rep['tips']['length_and_tip']} | {reasons(rows)} | traces with "
                  f"route <= 3 px {sum(c <= 3 for c in cov)}/{len(cov)}")


if __name__ == "__main__":
    main()
