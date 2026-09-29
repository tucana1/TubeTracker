"""Render, prepare and shard new training movies (preset v5 at a lower tube amplitude) on the sample field, the way
pipeline.ensure_synthetic does (25 frames per bin, 3 reference bins, census; data.build's shard); the image cache and
the movie are deleted once the shard is written, the truth is kept. Seeds >= 50 only (never a development or held-out
seed). Every shard's seed and full config is appended to records.jsonl.

    python gen.py LABEL SEED [SEED ...]      LABEL: v5faint (amplitude 0.35-0.7) or v5mid (0.5-1.0)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402

cv2.setNumThreads(1)
sys.path.insert(0, "/home/user/TubeTracker")
import numpy as np  # noqa: E402

from prototypes.learned_evidence import data  # noqa: E402
from sparsetrack import stack  # noqa: E402
from sparsetrack.cli import write_census  # noqa: E402
from sparsetrack.synth import make_movie  # noqa: E402

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
FIELD = Path("/home/user/TubeTracker/runs/sparsetrack/sample_movie/cache")
# name -> (base preset, SynthConfig overrides), registered beside data.EXTRA_PRESETS' v5w in this process only, so
# data.build's Scene(field, data.synth_config(name, seed)) renders the same geometry as the movie
CONFIGS = {"v5faint": ("v5", {"amplitude": (0.35, 0.7)}),   # the faint development movies' range
           "v5mid": ("v5", {"amplitude": (0.5, 1.0)})}      # bridges faint (<= 0.7) and v5 (>= 0.9)


def ensure(label: str, seed: int) -> Path:
    if seed < 50:
        raise SystemExit(f"seed {seed}: new training movies use seeds >= 50 only")
    data.EXTRA_PRESETS.update(CONFIGS)
    cfg = data.synth_config(label, seed)
    name = f"synth_{label}_s{seed}"
    shard = ME / "shards" / f"train_{label}_s{seed}.npz"
    if shard.exists():
        return shard
    t0 = time.time()
    synth = ME / "synth"
    movie, truth = synth / f"{name}.mp4", synth / f"{name}_truth.json"
    if not (movie.exists() and truth.exists()):  # the truth is written after the movie: its marker of completion
        make_movie(FIELD, synth, cfg, name=name, log=lambda *a: None)
    t1 = time.time()
    cache = synth / f"{label}s{seed}_cache"
    if not (cache / "grains.json").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)
    t2 = time.time()
    data.build(FIELD, cache, label, seed, shard)
    t3 = time.time()
    shutil.rmtree(cache)
    movie.unlink()
    d = np.load(shard)
    tr = json.loads(truth.read_text())
    rec = {"label": label, "seed": seed, "base_preset": CONFIGS[label][0], "overrides": CONFIGS[label][1],
           "config": asdict(cfg), "field": str(FIELD), "shard": str(shard), "samples": int(len(d["x"])),
           "tube_pixel_pct": round(100 * float(np.mean(d["body"])), 2),
           "emerged_within": sum(v["onset"].get("verdict") == "emerged_within" for v in tr["labels"].values()),
           "grains": len(tr["labels"]),
           "prepare": "stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0); write_census(cache, 3, False)",
           "shard_fn": "data.build(FIELD, cache, label, seed, shard) defaults: n_bins 70, crops_per_bin 16, half 48, "
                       "pos_frac 0.65, rng_seed 0",
           "seconds": {"render": round(t1 - t0), "prepare": round(t2 - t1), "shard": round(t3 - t2)}}
    with open(ME / "records.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return shard


if __name__ == "__main__":
    lab = sys.argv[1]
    for s in [int(v) for v in sys.argv[2:]]:
        t = time.time()
        p = ensure(lab, s)
        print(f"GEN_DONE {lab} {s} {p} in {time.time() - t:.0f}s", flush=True)
