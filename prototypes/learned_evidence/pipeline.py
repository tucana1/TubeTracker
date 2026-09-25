"""One command: train learned evidence on synthetic movies built from a real field, then score
SparseTrack with and without it on that real movie's human benchmark, and the learned evidence
read by the per-bin decoder (``reach.py``) as well.

    .venv/bin/python -m prototypes.learned_evidence.pipeline \
        --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld

Every step is cached in ``--work`` (re-running skips what exists). Synthetic caches (440 MB each)
are deleted once their training shard is written, unless ``--keep-caches``. Train on the dev
movie's field only: the held-out movie 2 must never supply training data, and its labels are
scored once per frozen model (``--heldout-once``). Both SparseTrack runs read a grain again at
+/-300 px when its path reaches the edge of the +/-150 px crop (``evaluate.adaptive_crop``;
``--fixed-crop`` keeps SparseTrack as is).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np

from sparsetrack import stack
from sparsetrack.cli import write_census
from sparsetrack.evaluate import load, score
from sparsetrack.synth import make_movie

from . import calibrate, data, evaluate, reach, review, train
from .model import load as load_model

# ten movies (model v2): on held-out synthetic seeds, +35 lengths in tolerance over five (95% CI +6 to +70)
DEFAULT_MOVIES = ("v5:0", "v5:1", "v5:2", "v5:9", "v5:10", "v5:11", "v2:0", "v2:1", "v3:0", "v3:2")


def ensure_synthetic(field: Path, work: Path, spec: str, keep_caches: bool, log=print) -> Path:
    """Movie -> cache -> training shard for one synthetic movie; returns the shard path."""
    pr, seed = spec.split(":")
    seed = int(seed)
    name = f"synth_{pr}_s{seed}"
    shard = work / "shards" / f"train_{pr}_s{seed}.npz"
    if shard.exists():
        return shard
    movie = work / "synth" / f"{name}.mp4"
    if not movie.exists():
        make_movie(field, work / "synth", data.synth_config(pr, seed), name=name, log=log)
    cache = work / "synth" / f"{pr}s{seed}_cache"
    if not (cache / "grains.json").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)
    data.build(field, cache, pr, seed, shard, log=log)
    if not keep_caches:
        shutil.rmtree(cache)
    return shard


def write_per_grain(work: Path, runs: dict, seconds: float, um_per_px: float | None = None,
                    s_per_frame: float | None = None) -> None:
    """Without labels: one row per grain with each run's status, onset interval and final length
    (also in um and minutes when the pixel size and frame interval are given)."""
    import csv
    ids = [g["id"] for g in runs["sparsetrack"]["grains"]]
    by_run = {name: {g["id"]: g for g in pred["grains"]} for name, pred in runs.items()}
    with open(work / "per_grain.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["status", "onset_after", "onset_by", "final_length_px"]
        cols += ["onset_by_min"] if s_per_frame else []
        cols += ["final_length_um"] if um_per_px else []
        w.writerow(["grain", "x", "y"] + [f"{name}_{col}" for name in runs for col in cols] + ["perbin_burst_frame"])
        for gid in ids:
            g0 = by_run["sparsetrack"][gid]
            row = [gid, g0.get("x"), g0.get("y")]
            for name in runs:
                g = by_run[name].get(gid, {})
                iv = g.get("onset_interval") or [None, None]
                row += [g.get("status"), iv[0], iv[1], g.get("final_length_px")]
                if s_per_frame:
                    row.append(round(iv[1] * s_per_frame / 60.0, 2) if iv[1] is not None else None)
                if um_per_px:
                    row.append(round(float(g.get("final_length_px") or 0.0) * um_per_px, 2))
            w.writerow(row + [by_run["perbin"].get(gid, {}).get("burst_frame")])
    lines = [f"{len(ids)} grains ({seconds:.0f} s); per-grain results in {work / 'per_grain.csv'}"]
    for name, pred in runs.items():
        st = [g.get("status") for g in pred["grains"]]
        grew = [g.get("final_length_px") or 0.0 for g in pred["grains"] if g.get("status") == "emerged_within"]
        lines.append(f"{name:12s} emerged within the movie {st.count('emerged_within'):3d}, at start "
                     f"{st.count('emerged_at_start'):3d}, none by the end {st.count('no_emergence_by_end'):3d}; "
                     f"median final length {float(np.median(grew)) if grew else 0.0:.1f} px")
    print("\n".join(lines))
    (work / "report.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="prepared cache of the real movie (e.g. runs/sparsetrack/ld)")
    ap.add_argument("--labels", default=None,
                    help="human benchmark labels for that movie; without them the three runs are written "
                         "(predictions and per_grain.csv) but not scored")
    ap.add_argument("--work", required=True)
    ap.add_argument("--synthetic", nargs="+", default=list(DEFAULT_MOVIES), help="PRESET:SEED synthetic movies")
    ap.add_argument("--train-field", default=None,
                    help="cache whose field the synthetic movies are built on (default --field; use the dev "
                         "movie's cache when scoring a held-out movie)")
    ap.add_argument("--model", default=None, help="use this trained model instead of training one")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--keep-caches", action="store_true")
    ap.add_argument("--heldout-once", action="store_true", help="required to score labels whose name contains m2")
    ap.add_argument("--um-per-px", type=float, default=None, help="pixel size, for lengths in um in per_grain.csv")
    ap.add_argument("--s-per-frame", type=float, default=None, help="frame interval, for onsets in minutes")
    ap.add_argument("--no-burst", action="store_true",
                    help="per-bin decoder: fit growth over the whole movie even where a tube's reading collapses "
                         "for good (by default that is read as a burst: growth is fitted up to it)")
    ap.add_argument("--decoder", default=None,
                    help="per-bin decoder settings fitted on the dev movie's traces by calibrate.py (decoder.json)")
    ap.add_argument("--fixed-crop", action="store_true",
                    help="keep SparseTrack's fixed +/-150 px grain crop even where a path runs into its edge "
                         "(by default such grains are read again at +/-300 px, in both runs)")
    args = ap.parse_args(argv)
    field, work = Path(args.field), Path(args.work)
    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None and "m2" in labels_path.name and not args.heldout_once:
        raise SystemExit("movie 2 is the held-out benchmark: score it once per frozen model (--heldout-once), "
                         "with --model trained on the dev field (--train-field runs/sparsetrack/ld)")
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_path = Path(args.model) if args.model else work / "unet.pt"
    if not model_path.exists():
        train_field = Path(args.train_field) if args.train_field else field
        shards = [str(ensure_synthetic(train_field, work, spec, args.keep_caches)) for spec in args.synthetic]
        train.main(["--shards", *shards, "--out", str(model_path), "--steps", str(args.steps)])
    net = load_model(str(model_path))
    pcache = evaluate.prob_cache(field, net, work / f"prob_{field.name}")
    with contextlib.nullcontext() if args.fixed_crop else evaluate.adaptive_crop():
        base = evaluate.run_baseline(field, work, grains_path=labels_path)
        learned = evaluate.run_on_prob_cache(pcache, field, work, "learned", grains_path=labels_path)
    # the same learned evidence read by the per-bin decoder (reach.py) instead of SparseTrack's
    kw = dict(big=None if args.fixed_crop else 300, burst=not args.no_burst,
              vmax=float(learned.get("params", {}).get("vmax_px", 4.0)))  # SparseTrack's speed cap
    kw.update(calibrate.decoder_settings(args.decoder, model_path))
    perbin = reach.analyze(pcache, field, grains_path=labels_path, log=lambda *a: None, **kw)
    perbin["decoder"] = {k: v for k, v in kw.items() if k != "vmax"} | {"vmax": kw["vmax"], "from": args.decoder}
    (work / "perbin").mkdir(exist_ok=True)
    (work / "perbin" / "predictions.json").write_text(json.dumps(perbin))
    # what the lab reviews and reports: pictures on the movie itself, the population curve, growth curves
    from sparsetrack.report import write_growth_curves, write_population
    review.write_review(pcache, field, perbin, work / "perbin", grains_path=labels_path, **kw)
    write_population(perbin, work / "perbin")
    write_growth_curves(perbin, work / "perbin", [g["id"] for g in perbin["grains"] if g.get("status") == "emerged_within"])
    if labels_path is None:
        write_per_grain(work, {"sparsetrack": base, "learned": learned, "perbin": perbin}, time.time() - started,
                        um_per_px=args.um_per_px, s_per_frame=args.s_per_frame)
        return
    labels = load(labels_path)
    rb, rl, rp = score(labels, base), score(labels, learned), score(labels, perbin)
    tol = rb["onset"]["tolerance_frames"]
    pb = evaluate.paired_bootstrap(rl, rb, tol)
    pp = evaluate.paired_bootstrap(rp, rl, tol)

    def diff(name, p):
        return (f"{name} over {p['grains']} grains: onset {p['onset_diff']:+.0f} "
                f"[{p['onset_ci'][0]:+.0f}, {p['onset_ci'][1]:+.0f}], lengths {p['length_diff']:+.0f} "
                f"[{p['length_ci'][0]:+.0f}, {p['length_ci'][1]:+.0f}] (95% paired bootstrap over grains)")

    lines = [f"{labels_path.name}: {rb['grains_scored']} grains scored ({time.time() - started:.0f} s)",
             evaluate.e2e_summary("baseline", rb), evaluate.e2e_summary("learned", rl),
             evaluate.e2e_summary("per-bin", rp),
             diff("learned - baseline", pb), diff("per-bin - learned", pp)]
    print("\n".join(lines))
    (work / "report.txt").write_text("\n".join(lines) + "\n")
    (work / "scores.json").write_text(json.dumps({"baseline": rb, "learned": rl, "perbin": rp, "paired": pb,
                                                  "paired_perbin_learned": pp}, default=str, indent=1))


if __name__ == "__main__":
    main()
