"""Synthetic movies with moving grains whose true tracks are known (for sparsetrack/track.py).

``sparsetrack.synth`` already moves some grains rigidly with their tubes (``p_move``, a smooth random drift
scaled to ``move_px``) and lands others in the first bins (``p_arrive``). Here, on top of that and without
changing ``synth.py``:

- ``push_frac`` of the moving grains are pushed instead of drifting: 1-3 sudden moves of 1-2 bins each (as on
  movie 2, where grains are swept off at up to ~8 px per bin), within the same reach;
- ``vanish_frac`` of the moving grains vanish (burst, or carried out of view) at a random bin: their sprite is
  no longer drawn (the field under it was already filled in), so the truth says from which bin they are lost;
- the truth file gets each moving grain's track: its displacement per bin (the mean over the bin's frames),
  and its vanishing bin.

    python -m prototypes.grain_tracking.synth_moves runs/sparsetrack/m2 --seed 31 --out runs/grain_tracking/synth
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from sparsetrack.synth import X264, Scene, preset
from sparsetrack.video import FFMPEG

MOVES = dict(p_move=0.5, move_px=(15.0, 60.0), p_anchor=0.1, p_arrive=0.05)


def _pushes(rng, n: int, fpb: int, reach: np.ndarray) -> np.ndarray:
    """A trajectory of 1-3 sudden pushes (each over 1-2 bins) after bin 5, fitting within ``reach`` (per axis)."""
    k = int(rng.integers(1, 4))
    starts = np.sort(rng.uniform(5 * fpb, 0.85 * n, k)).astype(int)
    traj = np.zeros((n, 2))
    pos = np.zeros(2)
    for s in starts:
        a = rng.uniform(0, 2 * np.pi)
        step = rng.uniform(0.4, 1.0) * np.array([math.cos(a), math.sin(a)]) * reach
        new = np.clip(pos + step, -reach, reach)
        dur = int(rng.uniform(1.0, 2.0) * fpb)
        ramp = np.clip((np.arange(n) - s) / max(dur, 1), 0.0, 1.0)[:, None]
        traj = traj + ramp * (new - pos)
        pos = new
    return traj


def make(field: str, seed: int, out_dir: str | Path, name: str | None = None, push_frac: float = 0.4,
         vanish_frac: float = 0.25, log=print, **over) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"moves_{Path(field).name}_s{seed}"
    cfg = preset("v5", seed=seed, **{**MOVES, **over})
    started = time.time()
    scene = Scene(field, cfg)
    rng = np.random.default_rng(10_000 + seed)
    fpb, n = cfg.frames_per_bin, cfg.n_frames
    movers = [gid for gid in scene.moves if gid not in scene.anchored and gid not in scene.arriving]
    kinds, vanish = {}, {}
    for gid in movers:
        traj = scene.moves[gid]
        if rng.random() < push_frac:
            traj[:] = _pushes(rng, n, fpb, np.abs(traj).max(axis=0))  # in place: the tube and the sprite share it
            kinds[gid] = "push"
        else:
            kinds[gid] = "drift"
        if gid in scene.sprites and rng.random() < vanish_frac:
            vanish[gid] = int(rng.uniform(20 * fpb, 0.9 * n))
    render_one = scene.render

    def render(k):
        hidden = {gid: scene.sprites.pop(gid) for gid, f in vanish.items() if k >= f and gid in scene.sprites}
        try:
            return render_one(k)
        finally:
            scene.sprites.update(hidden)

    movie = out_dir / f"{name}.mp4"
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{scene.w}x{scene.h}",
           "-r", "14/12", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-b:v", "12000k", "-x264-params", X264,
           "-pix_fmt", "yuv420p", str(movie)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k in range(n):
            proc.stdin.write(render(k).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    truth = scene.truth(name)
    n_bins = -(-n // fpb)
    tracks = {}
    for gid, traj in scene.moves.items():
        per_bin = np.array([traj[b * fpb:(b + 1) * fpb].mean(axis=0) for b in range(n_bins)])
        tracks[gid] = {"xy": np.round(per_bin, 3).tolist(), "kind": kinds.get(gid, "anchored" if gid in scene.anchored
                                                                                else "arriving"),
                       "vanish_bin": vanish[gid] // fpb if gid in vanish else None}
    truth["grain_tracks"] = {"about": "per census grain of the source field (census_id): its displacement (dx, dy) "
                                      "per bin, the mean over the bin's frames; vanish_bin: its sprite is not drawn "
                                      "from the frame in that bin on", "tracks": tracks,
                             "movers": {"push_frac": push_frac, "vanish_frac": vanish_frac, **MOVES, **over}}
    truth_path = out_dir / f"{name}_truth.json"
    truth_path.write_text(json.dumps(truth))
    log(f"{name}: {len(scene.tubes)} tubes, {len(movers)} moving grains ({sum(k == 'push' for k in kinds.values())} "
        f"pushed, {len(vanish)} vanish), {len(scene.anchored)} anchored, {len(scene.arriving)} landing; "
        f"{time.time() - started:.0f} s -> {movie}")
    return movie, truth_path


def prepare(movie: Path, cache: Path) -> Path:
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    if not (cache / "grains.json").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("field")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="runs/grain_tracking/synth")
    ap.add_argument("--push-frac", type=float, default=0.4)
    ap.add_argument("--vanish-frac", type=float, default=0.25)
    ap.add_argument("--n-frames", type=int, default=4382)
    args = ap.parse_args()
    movie, truth = make(args.field, args.seed, args.out, push_frac=args.push_frac, vanish_frac=args.vanish_frac,
                        n_frames=args.n_frames)
    prepare(movie, Path(args.out) / f"{movie.stem}_cache")
