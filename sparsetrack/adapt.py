"""Adapt the tube network to a new movie, without labels.

Synthetic movies are drawn on the movie's own field (its real pre-germination background and grains,
x264-encoded like the real movies, with exact truth), sized like it, and the shipped network
(``learned.MODEL``) is fine-tuned on them. The analysis of that movie then reads its crowded and noisy
grains with the adapted network.

Why: the network was trained on synthetic movies of the two benchmark fields only. On movie 2, synthetic
movies built on its own field lifted the share of traced tube the network found from 43% to 68%, and its
lengths within tolerance from 9 to 23 of 54 (27 Sep 2026, no labels involved). A new movie's field
(grain density, illumination, optics, tube looks) can differ as much.

    python -m sparsetrack adapt MOVIE      # 30-60 min; then `sparsetrack run MOVIE` uses the adapted network

Needs torch and the repository's ``prototypes/learned_flood`` (shard builder and trainer).
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np

LOOK = dict(p_bright_core=0.8, width=(1.0, 1.6))  # light-cored, wider tubes (movie 2's): every other movie
MODEL_NAME = "tubes_adapted.pt"


def sized_like(meta: dict) -> dict:
    """Synthetic movie settings for a real movie of ``meta["n_bins"]`` bins: as long, onsets over its first
    70%, tubes that can grow as long as its bins allow (160 px over the dev movie's 176 bins, 280 over movie 2's
    351)."""
    nb = int(meta["n_bins"])
    return dict(n_frames=nb * 25, onset_bins=(6.0, round(0.7 * nb, 1)), rate_px_per_bin=(0.15, 1.5),
                max_length=float(np.clip(0.8 * nb, 120.0, 300.0)))


def adapted_model(folder: str | Path) -> Path | None:
    """The adapted network in a movie's run folder, if ``adapt`` has made one."""
    p = Path(folder) / "adapt" / MODEL_NAME
    return p if p.exists() else None


def adapt(field: str | Path, work: str | Path, seeds: tuple[int, ...] = (101, 102), steps: int = 3000,
          init: str | Path | None = None, lr: float = 1e-3, n_bins: int = 90, log=print) -> Path:
    """Synthetic movies on ``field`` (a prepared cache) -> training shards -> the network fine-tuned from
    ``init`` (default: the shipped one) -> ``work/tubes_adapted.pt``. Finished steps are skipped when run again."""
    try:
        from prototypes.learned_flood import train
        from prototypes.learned_flood.data import build
    except ImportError as e:  # the shard builder and trainer live in the repository, next to the package
        raise SystemExit(f"adapt needs torch and the repository's prototypes/learned_flood ({e})")
    from . import learned, stack
    from .cli import write_census
    from .synth import make_movie, preset

    field, work = Path(field), Path(work)
    meta = json.loads((field / "meta.json").read_text())
    size = sized_like(meta)
    started = time.time()
    shards = []
    for i, seed in enumerate(seeds):
        over = {**size, **(LOOK if i % 2 else {})}
        shard = work / "shards" / f"synth_s{seed}.npz"
        if not shard.exists():
            movie = work / "synth" / f"synth_s{seed}.mp4"
            if not (work / "synth" / f"{movie.stem}_truth.json").exists():  # written last: the movie is complete
                log(f"synthetic movie {i + 1} of {len(seeds)} on this movie's field (seed {seed}) ...")
                make_movie(field, work / "synth", preset("v5", seed=seed, **over), name=movie.stem, log=lambda *a: None)
            cache = work / "synth" / f"synth_s{seed}_cache"
            with contextlib.redirect_stdout(io.StringIO()):
                stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
                write_census(cache, 3, False)
            build(field, cache, "v5", seed, shard, n_bins=n_bins, log=log, **over)
            shutil.rmtree(cache)  # 0.4-0.9 GB; the shard holds what training needs
        shards.append(str(shard))
    out = work / MODEL_NAME
    tmp = work / (MODEL_NAME + ".training")
    init = Path(init) if init else learned.MODEL
    log(f"fine-tuning {init.name} on {len(shards)} synthetic movies of this field ({steps} steps) ...")
    train.main(["--shards", *shards, "--out", str(tmp), "--steps", str(steps), "--init", str(init),
                "--lr", str(lr)])
    ck = __import__("torch").load(str(tmp), map_location="cpu", weights_only=False)
    ck["adapted"] = {"field": str(field), "seeds": list(seeds), "steps": steps, "init": str(init),
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    __import__("torch").save(ck, str(out))
    tmp.unlink()
    log(f"adapted network: {out} ({(time.time() - started) / 60:.0f} min)")
    return out
