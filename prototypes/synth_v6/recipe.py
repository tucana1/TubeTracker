"""How the v6 networks were made (29 Sep 2026), step by step.

    python -m prototypes.synth_v6.recipe shards [NAME ...]   # renders only the bins each shard uses (shards.py)
    python -m prototypes.synth_v6.train6 ...                 # see README.md for the exact training commands

Training data: synthetic v6 movies (v5 + tip bulbs and rounded tips + the grain body's own change) on the dev
movie's field (seeds 30-31; 33 for validation), on movie 2's field sized like movie 2 (seeds 40-43; 41-42 with
movie 2's mostly light-cored, wider tubes) and on movie 1's field (unlabelled; seed 50, sized like movie 2). New
seeds, so no scene repeats a v5 training scene. 15% of the crops are centred on a young tube (<= 12 px), the case
the first network missed. No human labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARDS = REPO / "runs/synth_v6/shards"
M2 = dict(n_frames=351 * 25, onset_bins=(10.0, 250.0), rate_px_per_bin=(0.15, 1.5), max_length=280.0)
LOOK = dict(p_bright_core=0.8, width=(1.0, 1.6))
YOUNG = 0.15
# Round 1 was built before the grain's own change was calibrated against the real movies (it was ~2x too weak:
# prototypes/synth_v6/README.md); its shards keep those settings as explicit overrides, so they rebuild exactly.
R1 = dict(focus_px=2.0, focus_max_px=2.0, focus_bins=8.0, focus_grain_px=0.8, jitter_px=(0.2, 3.0),
          body_lead_bins=(-2.0, 4.0))
# (field cache, seed, config overrides, shard name, bins per shard, share of crops on young tubes), in the order
# they were built, one at a time (the laptop was shared).
PLAN = [("runs/sparsetrack/m2", 40, {**M2, **R1}, "train_v6m2_s40", 90, YOUNG),                  # round 1
        ("runs/sparsetrack/m2", 41, {**M2, **LOOK, **R1}, "train_v6m2_s41", 90, YOUNG),
        ("runs/sparsetrack/ld", 30, R1, "train_v6ld_s30", 60, YOUNG),
        ("runs/sparsetrack/ld", 33, R1, "val_v6ld_s33", 30, 0.0),
        ("runs/sparsetrack/m2", 42, {**M2, **LOOK}, "train_v6m2_s42", 90, YOUNG),                  # round 2: final v6
        ("runs/sparsetrack/m2", 43, M2, "train_v6m2_s43", 90, YOUNG),
        ("runs/sparsetrack/ld", 31, {}, "train_v6ld_s31", 60, YOUNG),
        ("runs/sparsetrack/m1", 50, M2, "train_v6m1_s50", 90, YOUNG)]


def one(item) -> str:
    from .shards import build
    field, seed, over, name, n_bins, young = item
    out = SHARDS / f"{name}.npz"
    if not out.exists():
        build(REPO / field, "v6", seed, out, n_bins_pick=n_bins, young_frac=young, work=SHARDS / "work",
              log=lambda *a: print(*a, flush=True), **over)
    return str(out)


def shards(names: list[str] | None = None) -> None:
    """Build the planned shards (or only ``names``) one after another, skipping finished ones."""
    for item in PLAN:
        if (names and item[3] not in names) or (SHARDS / f"{item[3]}.npz").exists():
            continue
        print("done", one(item), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "shards":
        shards(sys.argv[2:] or None)
