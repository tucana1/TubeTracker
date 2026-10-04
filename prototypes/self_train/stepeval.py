"""Judge the self-training step (sparsetrack/selftrain.py) end to end on a labelled movie: the tip-trajectory reader with
the 0.9.0 candidate's settings (tiptraj="flood", weights_ld_v3.json, tiptraj_mid) and a held-out tip-detector fold, on
the maps of a given tube network (``Params.model``: the hybrid's flood reads them too), reading the movie's scored
grains as prototypes/tip_trajectory/bench.py does (census = the labels file, as scripts/score_heldout.sh runs it).

    python -m prototypes.self_train.stepeval read m1 --model NET.pt --cache CACHE --det DET_DIR --out OUT_DIR [--drop-maps]
    python -m prototypes.self_train.stepeval table m1 --pred NAME=PRED.json ... --vs NAME ... [--md OUT.md]

``table``: onsets, lengths and length-and-tip of each reading on the movie's scored grains, and each against the
``--vs`` readings paired over grains (95% bootstrap, prototypes/tip_trajectory/dp.paired).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from sparsetrack.analyze import Params, analyze
from sparsetrack.evaluate import score

from prototypes.tip_trajectory.common import LABELS, labels, scored_grains
from prototypes.tip_trajectory.dp import paired, per_grain

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "prototypes/tip_trajectory/weights_ld_v3.json"


def read(movie: str, model: str, cache: str, det: str, out: str, drop_maps: bool = False) -> dict:
    lab = labels(movie)
    t0 = time.time()
    p = Params(model=model, tiptraj="flood", tiptraj_det=det, tiptraj_weights=str(WEIGHTS), tiptraj_mid=True)
    pred = analyze(cache, Path(out), grains_path=LABELS[movie], params=p, only=scored_grains(lab), log=lambda *a: None)
    if drop_maps:  # the probability movie (~440 MB) is not kept
        maps = Path(cache) / f"prob_{Path(model).stem}"
        if maps.exists() and not maps.is_symlink():
            shutil.rmtree(maps)
    rep = score(lab, pred)
    print(f"{movie} {Path(model).name}: onsets {rep['onset']['hits']}/{rep['onset']['n_timed']}, lengths "
          f"{rep['length_full']['within_tolerance']}/{rep['length_full']['n']}, length and tip "
          f"{rep['tips']['length_and_tip']} ({time.time() - t0:.0f} s)", flush=True)
    return pred


def table(movie: str, preds: dict[str, str], vs: list[str]) -> str:
    lab = labels(movie)
    reps = {k: score(lab, json.loads(Path(v).read_text())) for k, v in preds.items()}
    pg = {k: per_grain(r) for k, r in reps.items()}
    head = "| reading | onsets | lengths in tolerance | length and tip |" + "".join(
        f" vs {v}: onsets; lengths; l&t |" for v in vs)
    lines = [head, "|" + "---|" * (4 + len(vs))]
    for k, r in reps.items():
        row = (f"| {k} | {r['onset']['hits']}/{r['onset']['n_timed']} | {r['length_full']['within_tolerance']}/"
               f"{r['length_full']['n']} | {r['tips']['length_and_tip']} |")
        for v in vs:
            if v == k:
                row += " - |"
                continue
            pr = paired(pg[v], pg[k])
            row += " " + "; ".join(f"{d:+d} [{lo:+.0f}, {hi:+.0f}]" for d, lo, hi in pr) + " |"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "table"))
    ap.add_argument("movie")
    ap.add_argument("--model")
    ap.add_argument("--cache")
    ap.add_argument("--det")
    ap.add_argument("--out")
    ap.add_argument("--drop-maps", action="store_true")
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--vs", nargs="*", default=[])
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    if a.what == "read":
        read(a.movie, a.model, a.cache, a.det, a.out, a.drop_maps)
    else:
        txt = table(a.movie, dict(kv.split("=", 1) for kv in a.pred), a.vs)
        print(txt)
        if a.md:
            Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
