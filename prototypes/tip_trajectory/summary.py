"""Tables for a fixed setting of the global tip-trajectory reader (from tune.py): per movie, against SparseTrack 0.8.8
on the same grains (paired 95% bootstrap), by length class, and why traces are missed (failures.py), for the reader on
every scored grain and on 0.8.8's flood-read grains only; writes the predictions for scripts/compare_predictions.py.

    python -m prototypes.tip_trajectory.summary --setting OUT/tune_ld_b.json --movies ld m2 m1 --tag ld_b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import dp
from .common import OUT, baseline, labels, scored_grains
from .failures import classify, table
from .tune import fmt, report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, help="tune_<tag>.json (its 'best') or a JSON dict")
    ap.add_argument("--movies", nargs="+", default=["ld", "m2", "m1"])
    ap.add_argument("--tag", default="s")
    a = ap.parse_args(argv)
    s = Path(a.setting)
    p = {**dp.DEFAULT, **(json.loads(s.read_text())["best"] if s.exists() else json.loads(a.setting))}
    out = {"setting": p, "movies": {}}
    for m in a.movies:
        doc = dp.load(m)
        lab, base = labels(m), baseline(m)
        res = {}
        for of in (False, True):
            r = report(m, doc, p, of)
            print(fmt(r), flush=True)
            res["flood_only" if of else "all"] = r
        ev = dp.evaluate(m, doc, p)
        gids = scored_grains(lab)
        res["why_new"] = table(classify(lab, ev["pred"], gids, doc["rs"], doc["nb"], doc["fpb"]))
        res["why_base"] = table(classify(lab, base, gids, doc["rs"], doc["nb"], doc["fpb"]))
        print("  why (new):", res["why_new"]["all"], " moved:", res["why_new"].get("moved"))
        print("  why (0.8.8):", res["why_base"]["all"], " moved:", res["why_base"].get("moved"))
        for c in ("young", "mid", "long"):
            print(f"  {c}: new {res['why_new'].get(c)} | 0.8.8 {res['why_base'].get(c)}")
        ev["pred"]["method"] = f"tip-trajectory DP ({a.tag})"
        (OUT / f"pred_{a.tag}_{m}.json").write_text(json.dumps(ev["pred"]))
        out["movies"][m] = res
    (OUT / f"summary_{a.tag}.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
