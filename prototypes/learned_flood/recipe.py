"""How sparsetrack/models/tubes_synth_v1.pt was made (27 Sep 2026), step by step.

    python -m prototypes.learned_flood.recipe movies      # ~7 min per movie-2-field movie
    python -m prototypes.learned_flood.recipe shards      # bins each movie, cuts a shard, drops the bins
    python -m prototypes.learned_flood.train --shards "runs/learned_flood/shards/train_*.npz" \
        --val runs/learned_flood/shards/val_v5s3.npz --out runs/learned_flood/unet_d_syn.pt --steps 8000

Training data: synthetic v5 movies on the dev movie's field (seeds 0-2, the existing
runs/sparsetrack/synth/synthv5_s*.mp4; seed 3 for validation) and on movie 2's field (seeds
10-12, sized like movie 2; seeds 11-12 with movie 2's mostly light-cored, wider tubes).
No human labels. The checkpoint's weights were then copied to sparsetrack/models/.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SYN = REPO / "runs/sparsetrack/synth"
SHARDS = REPO / "runs/learned_flood/shards"
M2 = dict(n_frames=351 * 25, onset_bins=(10.0, 250.0), rate_px_per_bin=(0.15, 1.5), max_length=280.0)
LOOK = dict(p_bright_core=0.8, width=(1.0, 1.6))  # movie 2's tubes: mostly light-cored, wider
# (field cache, seed, config overrides, shard name, bins per shard)
PLAN = [("runs/sparsetrack/ld", s, {}, f"train_v5s{s}", 60) for s in (0, 1, 2)] + \
       [("runs/sparsetrack/ld", 3, {}, "val_v5s3", 60)] + \
       [("runs/sparsetrack/m2", s, {**M2, **(LOOK if s >= 11 else {})}, f"train_v5m2s{s}", 90) for s in (10, 11, 12)]


def movie_path(field: str, seed: int) -> Path:
    return SYN / (f"synthv5_s{seed}.mp4" if field.endswith("/ld") else f"synthv5m2_s{seed}.mp4")


def movies() -> None:
    from sparsetrack.synth import make_movie, preset
    for field, seed, over, _, _ in PLAN:
        if not movie_path(field, seed).exists():
            make_movie(REPO / field, SYN, preset("v5", seed=seed, **over), name=movie_path(field, seed).stem)


def shards() -> None:
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    from .data import build
    for field, seed, over, name, n_bins in PLAN:
        out = SHARDS / f"{name}.npz"
        if out.exists():
            continue
        cache = SYN / f"{movie_path(field, seed).stem}_cache"
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(movie_path(field, seed), cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)
        build(REPO / field, cache, "v5", seed, out, n_bins=n_bins, **over)
        shutil.rmtree(cache)  # 440-920 MB each


if __name__ == "__main__":
    {"movies": movies, "shards": shards}[sys.argv[1]]()
