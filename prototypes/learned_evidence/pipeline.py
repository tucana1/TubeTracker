"""One command: train learned evidence on synthetic movies built from a real field, then score
SparseTrack with and without it on that real movie's human benchmark.

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

from sparsetrack import stack
from sparsetrack.cli import write_census
from sparsetrack.evaluate import load, score
from sparsetrack.synth import make_movie, preset

from . import data, evaluate, train
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
        make_movie(field, work / "synth", preset(pr, seed=seed), name=name, log=log)
    cache = work / "synth" / f"{pr}s{seed}_cache"
    if not (cache / "grains.json").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)
    data.build(field, cache, pr, seed, shard, log=log)
    if not keep_caches:
        shutil.rmtree(cache)
    return shard


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="prepared cache of the real movie (e.g. runs/sparsetrack/ld)")
    ap.add_argument("--labels", required=True, help="human benchmark labels for that movie")
    ap.add_argument("--work", required=True)
    ap.add_argument("--synthetic", nargs="+", default=list(DEFAULT_MOVIES), help="PRESET:SEED synthetic movies")
    ap.add_argument("--train-field", default=None,
                    help="cache whose field the synthetic movies are built on (default --field; use the dev "
                         "movie's cache when scoring a held-out movie)")
    ap.add_argument("--model", default=None, help="use this trained model instead of training one")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--keep-caches", action="store_true")
    ap.add_argument("--heldout-once", action="store_true", help="required to score labels whose name contains m2")
    ap.add_argument("--fixed-crop", action="store_true",
                    help="keep SparseTrack's fixed +/-150 px grain crop even where a path runs into its edge "
                         "(by default such grains are read again at +/-300 px, in both runs)")
    args = ap.parse_args(argv)
    field, work, labels_path = Path(args.field), Path(args.work), Path(args.labels)
    if "m2" in labels_path.name and not args.heldout_once:
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
    labels = load(labels_path)
    with contextlib.nullcontext() if args.fixed_crop else evaluate.adaptive_crop():
        base = evaluate.run_baseline(field, work, grains_path=labels_path)
        learned = evaluate.run_on_prob_cache(pcache, field, work, "learned", grains_path=labels_path)
    rb, rl = score(labels, base), score(labels, learned)
    tol = rb["onset"]["tolerance_frames"]
    pb = evaluate.paired_bootstrap(rl, rb, tol)
    lines = [f"{labels_path.name}: {rb['grains_scored']} grains scored ({time.time() - started:.0f} s)",
             evaluate.e2e_summary("baseline", rb), evaluate.e2e_summary("learned", rl),
             (f"learned - baseline over {pb['grains']} grains: onset {pb['onset_diff']:+.0f} "
              f"[{pb['onset_ci'][0]:+.0f}, {pb['onset_ci'][1]:+.0f}], lengths {pb['length_diff']:+.0f} "
              f"[{pb['length_ci'][0]:+.0f}, {pb['length_ci'][1]:+.0f}] (95% paired bootstrap over grains)")]
    print("\n".join(lines))
    (work / "report.txt").write_text("\n".join(lines) + "\n")
    (work / "scores.json").write_text(json.dumps({"baseline": rb, "learned": rl, "paired": pb}, default=str, indent=1))


if __name__ == "__main__":
    main()
