"""End-to-end check of a SparseTrack variant on one labelled movie, reading only the scored grains (the readings of
the other grains do not enter the scores; the speed-cap probe still samples the whole census, so a grain's reading is
the one a whole-movie run gives), scored and paired against the 0.8.8 baseline over grains (95% bootstrap).

    python -m prototypes.tip_trajectory.bench m2 --set tiptraj=flood tiptraj_det_dir=<dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from sparsetrack.analyze import Params, analyze
from sparsetrack.evaluate import score

from .common import REPO, baseline, cache_dir, labels, scored_grains
from .dp import paired, per_grain

sys.path.insert(0, str(REPO / "scripts"))
from synth_bench import parse_value  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = {f.name for f in fields(Params)}
    kw = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown Params field {k}")
        kw[k] = parse_value(v)
    p = Params(**kw)
    lab = labels(a.movie)
    ids = scored_grains(lab)
    pred = analyze(cache_dir(a.movie), Path(a.out), grains_path=REPO / f"benchmark/labels/{a.movie}_v1.json", params=p,
                   only=ids, log=lambda *x: None)
    rep, b_rep = score(lab, pred), score(lab, baseline(a.movie))
    pr = paired(per_grain(b_rep), per_grain(rep))
    txt = (f"{a.movie} {' '.join(a.set)}: onsets {rep['onset']['hits']}/{rep['onset']['n_timed']} "
           f"(0.8.8 {b_rep['onset']['hits']}/{b_rep['onset']['n_timed']}; {pr[0][0]:+d}, {pr[0][1]:+.0f}..{pr[0][2]:+.0f}), "
           f"lengths {rep['length_full']['within_tolerance']}/{rep['length_full']['n']} "
           f"(0.8.8 {b_rep['length_full']['within_tolerance']}; {pr[1][0]:+d}, {pr[1][1]:+.0f}..{pr[1][2]:+.0f}), "
           f"length and tip {rep['tips']['length_and_tip']} (0.8.8 {b_rep['tips']['length_and_tip']}; "
           f"{pr[2][0]:+d}, {pr[2][1]:+.0f}..{pr[2][2]:+.0f})")
    print(txt, flush=True)
    Path(a.out, "bench.json").write_text(json.dumps({"set": a.set, "summary": txt, "paired": pr}))


if __name__ == "__main__":
    main()
