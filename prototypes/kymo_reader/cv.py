"""Within-movie cross-validation: does the reader beat st053 when it has labels from the same movie?

    python -m prototypes.kymo_reader.cv ld [--folds 2]

The movie's included grains are split at random (seed 0) into folds. For each fold, the pilot is
fine-tuned exactly like B/C (synthetic pool + the OTHER folds' real labels, 500 steps of the 2500-step
schedule), its settings (start, onset source, thresholds) are chosen on the other folds' labels, and it
reads the held-out fold. Every grain is read once by a model that never saw its labels; the merged
predictions are scored against st053 and SparseTrack-on-the-human-route with the paired bootstrap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import store
from .model import load

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/kymo_reader"
SYNTH = [f"synth_synthv5_s{s}" for s in (0, 1, 2, 5, 6, 7)] + ["synth_synthv5m2_s10"]
VAL = ["synth_synthv5_s3", "synth_synthv5m2_s20"]


def main(argv=None):
    from sparsetrack.evaluate import load as jload, score
    ap = argparse.ArgumentParser()
    ap.add_argument("movie", choices=("ld", "m2"))
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    labels = jload(E.LABELS[a.movie])
    base = jload(E.ST053[a.movie])
    frames = base["grains"][0]["length"]["frames"]
    by_id = {g["id"]: g for g in base["grains"]}
    grains = sorted(gid for gid, g in labels["grains"].items() if not g.get("excluded"))
    perm = list(np.random.default_rng(a.seed).permutation(grains))
    fold_of = {g: i % a.folds for i, g in enumerate(perm)}
    samples = store.load(OUT / "data" / f"real_{a.movie}.npz")
    reads = {"st": {}, "route": {}}
    info = {"folds": {}}
    for k in range(a.folds):
        train_g = [g for g in grains if fold_of[g] != k]
        test_g = [g for g in grains if fold_of[g] == k]
        out = OUT / "models" / f"cv_{a.movie}_f{k}.pt"
        if not out.exists():
            cmd = [sys.executable, "-u", "-m", "prototypes.kymo_reader.train", "--out", str(out),
                   "--init", str(OUT / "models/pilot_front.pt"),
                   "--synth", *[str(OUT / "data" / f"{n}.npz") for n in SYNTH],
                   "--val", *[str(OUT / "data" / f"{n}.npz") for n in VAL],
                   "--steps", "2500", "--stop-after", "500", "--val-every", "500", "--lr", "1e-3", "--front", "1.0",
                   "--real", a.movie, "--p-real", "0.35", "--real-grains", *train_g]
            with open(OUT / "logs" / f"train_cv_{a.movie}_f{k}.log", "w") as fh:
                subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
        net = load(out, dev)
        ch = E.choose_on_movie(net, a.movie, dev, grains=train_g)
        info["folds"][k] = {"train": train_g, "test": test_g, "choice": ch}
        print(f"fold {k}: {len(train_g)} training grains, {len(test_g)} held out; chosen on the training grains: {ch}",
              flush=True)
        for sm in samples:
            gid, v = sm["info"]["grain"], sm["info"]["variant"]
            if gid not in test_g:
                continue
            keep = by_id.get(gid) if (ch["onset_source"] == "st053" and v == "st") else None
            reads[v][gid] = E.read_sample(net, sm, dev, frames, ch["start"], 4.0, ch["onset_px"], ch["min_len"],
                                          keep_call=keep)
    rep_base = score(labels, base)
    hits_base = E.grain_hits(rep_base)
    res = {"movie": a.movie, "st053": E.summary(rep_base), **info}
    preds = {f"cv_st": E.write_pred(base, reads["st"], f"kymo_reader within-movie CV ({a.movie}) on st053 paths"),
             f"cv_route": E.write_pred(base, reads["route"], f"kymo_reader within-movie CV ({a.movie}) on human routes"),
             "st053_route": E.write_pred(base, E.st_oracle_reads(samples, frames), "st053 on human routes")}
    hits = {}
    for name, pred in preds.items():
        rep = score(labels, pred)
        hits[name] = E.grain_hits(rep)
        if name != "st053_route":
            (OUT / "results" / f"{name}_{a.movie}.json").write_text(json.dumps(pred))
            rows = E.trace_rows(rep)
            import csv
            with open(OUT / "results" / f"{name}_{a.movie}_traces.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        res[name] = {**E.summary(rep), "vs_st053": E.paired(hits_base, hits[name])}
    res["cv_route"]["vs_st053_route"] = E.paired(hits["st053_route"], hits["cv_route"])
    (OUT / "results" / f"cv_{a.movie}_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(E.fmt_row("st053", res["st053"]))
    for k in ("cv_st", "cv_route", "st053_route"):
        print(E.fmt_row(k, res[k]))
    v = res["cv_route"]["vs_st053_route"]
    print("  same human routes, CV kymo vs SparseTrack: " + ", ".join(
        f"{k} {v[k]['delta']:+d} [{v[k]['ci'][0]:+.0f},{v[k]['ci'][1]:+.0f}]" for k in ("onset_hit", "len_hit", "both_hit")))


if __name__ == "__main__":
    main()
