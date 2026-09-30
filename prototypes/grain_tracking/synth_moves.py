"""Synthetic movies with moving grains whose true tracks are known (for sparsetrack/track.py).

``sparsetrack.synth`` already moves some grains rigidly with their tubes (``p_move``, a smooth random drift
scaled to ``move_px``) and lands others in the first bins (``p_arrive``). Here, on top of that and without
changing ``synth.py``:

- ``push_frac`` of the moving grains are pushed instead of drifting: 1-3 sudden moves of 1-2 bins each (as on
  movie 2, where grains are swept off at up to ~8 px per bin), within the same reach;
- ``vanish_frac`` of the moving grains vanish (burst, or carried out of view) at a random bin: their sprite is
  no longer drawn (the field under it was already filled in), so the truth says from which bin they are lost;
- ``knock_frac`` of the moving grains are knocked instead (30 Sep 2026, as m1 g027, g030, g056): 1-2 sudden jumps of
  0-40 px, each within one bin, turning the grain and its tube by 30-150 degrees either way; ``flip_frac`` of the
  knocked grains also show another side from their first knock on (their look mirrored), as m1 g027 did. Drawn
  from their own random stream, so a movie without knocks is the one made before;
- the truth file gets each moving grain's track: its displacement per bin (the mean over the bin's frames),
  its turn per bin (``rot``, deg, clockwise in the image), and its vanishing bin.

    python -m prototypes.grain_tracking.synth_moves runs/sparsetrack/m2 --seed 31 --out runs/grain_tracking/synth
    python -m prototypes.grain_tracking.synth_moves runs/sparsetrack/m2 --seed 33 --knock-frac 0.5 --flip-frac 0.5
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

import cv2
import numpy as np
from scipy.spatial import cKDTree

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


def _knocks(rng, n: int, fpb: int) -> tuple[np.ndarray, np.ndarray]:
    """1-2 knocks after bin 8: each a jump of 0-40 px and a turn of 30-150 degrees either way, within one bin.
    Returns the displacement (n, 2) and the turn (n,) per frame, both 0 before the first knock."""
    k = int(rng.integers(1, 3))
    starts = np.sort(rng.uniform(8 * fpb, 0.85 * n, k)).astype(int)
    traj, rot = np.zeros((n, 2)), np.zeros(n)
    for s in starts:
        a = rng.uniform(0, 2 * np.pi)
        jump = rng.uniform(0.0, 40.0) * np.array([math.cos(a), math.sin(a)])
        turn = rng.uniform(30.0, 150.0) * rng.choice([-1.0, 1.0])
        ramp = np.clip((np.arange(n) - s) / max(int(rng.uniform(0.1, 1.0) * fpb), 1), 0.0, 1.0)
        traj = traj + ramp[:, None] * jump
        rot = rot + ramp * turn
    return traj, rot


def _tube_map(scene: Scene, t) -> tuple:
    """A tube's lookup maps (as Scene builds them) over the disc it sweeps round its grain, wherever the grain goes."""
    pad = 8
    gx, gy = t.grain["x"], t.grain["y"]
    reach = float(np.max(np.hypot(t.path[:, 0] - gx, t.path[:, 1] - gy))) + pad
    lo = np.floor([gx - reach, gy - reach]).astype(int)
    hi = np.ceil([gx + reach, gy + reach]).astype(int)
    if t.move is not None:
        m = int(math.ceil(np.abs(t.move).max())) + 2
        lo, hi = lo - m, hi + m
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [scene.w - 1, scene.h - 1])
    yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].astype(np.float32)
    d, idx = cKDTree(t.path).query(np.stack([xx.ravel(), yy.ravel()], 1))
    return lo, hi, xx, yy, (idx * 0.25).reshape(xx.shape).astype(np.float32), d.reshape(xx.shape).astype(np.float32)


def _turned_sprite(sprite: tuple, g: dict, deg: float, flip: bool) -> tuple:
    """A grain sprite (sprite, alpha, x0, y0) turned by ``deg`` about the grain centre (clockwise in the image), and
    mirrored first if ``flip`` (the grain seen from another side)."""
    sp, al, x0, y0 = sprite
    c = (float(g["x"]) - x0, float(g["y"]) - y0)
    if flip:
        m = np.float32([[-1, 0, 2 * c[0]], [0, 1, 0]])  # mirrored about the vertical line through the centre
        sp = cv2.warpAffine(sp, m, sp.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        al = cv2.warpAffine(al, m, al.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0)
    if deg:
        m = cv2.getRotationMatrix2D(c, -float(deg), 1.0)
        sp = cv2.warpAffine(sp, m, sp.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        al = cv2.warpAffine(al, m, al.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0)
    return sp, al, x0, y0


def make(field: str, seed: int, out_dir: str | Path, name: str | None = None, push_frac: float = 0.4,
         vanish_frac: float = 0.25, knock_frac: float = 0.0, flip_frac: float = 0.0, log=print,
         **over) -> tuple[Path, Path]:
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
    # knocks: from their own random stream (a movie without knocks is unchanged)
    rng_k = np.random.default_rng([seed, 7])
    turns, flips = {}, {}
    census = {g["id"]: g for g in json.loads((Path(field) / "grains.json").read_text())["grains"]}
    for gid in movers:
        if knock_frac <= 0 or gid not in scene.sprites or rng_k.random() >= knock_frac:
            continue
        traj, rot = _knocks(rng_k, n, fpb)
        scene.moves[gid][:] = traj  # in place: the tube and the sprite share it
        kinds[gid], turns[gid] = "knock", rot
        if rng_k.random() < flip_frac:
            flips[gid] = int(np.argmax(rot != 0))
        for i, t in enumerate(scene.tubes):
            if t.grain["id"] == gid:  # the tube turns with its grain (0 at the end: the traced end state)
                t.rot = (t.rot if t.rot is not None else 0.0) + rot - rot[-1]
                t.sway = None
                scene.maps[i] = _tube_map(scene, t)
    render_one = scene.render
    cache: dict = {}

    def render(k):
        hidden = {gid: scene.sprites.pop(gid) for gid, f in vanish.items() if k >= f and gid in scene.sprites}
        swapped = {}
        for gid, rot in turns.items():
            if gid not in scene.sprites:
                continue
            flip = gid in flips and k >= flips[gid]
            key = (gid, round(float(rot[k]), 1), flip)
            if key[1] == 0 and not flip:
                continue
            if key not in cache:
                if len(cache) > 256:
                    cache.clear()
                cache[key] = _turned_sprite(scene.sprites[gid], census[gid], key[1], flip)
            swapped[gid] = scene.sprites[gid]
            scene.sprites[gid] = cache[key]
        try:
            return render_one(k)
        finally:
            scene.sprites.update(hidden)
            scene.sprites.update(swapped)

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
        rot = turns.get(gid)
        tracks[gid] = {"xy": np.round(per_bin, 3).tolist(), "kind": kinds.get(gid, "anchored" if gid in scene.anchored
                                                                                else "arriving"),
                       "vanish_bin": vanish[gid] // fpb if gid in vanish else None,
                       "rot": None if rot is None else np.round([rot[b * fpb:(b + 1) * fpb].mean()
                                                                 for b in range(n_bins)], 2).tolist(),
                       "flip_bin": flips[gid] // fpb if gid in flips else None}
    truth["grain_tracks"] = {"about": "per census grain of the source field (census_id): its displacement (dx, dy) "
                                      "per bin, the mean over the bin's frames; vanish_bin: its sprite is not drawn "
                                      "from the frame in that bin on", "tracks": tracks,
                             "movers": {"push_frac": push_frac, "vanish_frac": vanish_frac, "knock_frac": knock_frac,
                                        "flip_frac": flip_frac, **MOVES, **over}}
    truth_path = out_dir / f"{name}_truth.json"
    truth_path.write_text(json.dumps(truth))
    log(f"{name}: {len(scene.tubes)} tubes, {len(movers)} moving grains ({sum(k == 'push' for k in kinds.values())} "
        f"pushed, {len(turns)} knocked ({len(flips)} flipped), {len(vanish)} vanish), {len(scene.anchored)} anchored, {len(scene.arriving)} landing; "
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
    ap.add_argument("--knock-frac", type=float, default=0.0)
    ap.add_argument("--flip-frac", type=float, default=0.0)
    ap.add_argument("--n-frames", type=int, default=4382)
    args = ap.parse_args()
    movie, truth = make(args.field, args.seed, args.out, push_frac=args.push_frac, vanish_frac=args.vanish_frac,
                        knock_frac=args.knock_frac, flip_frac=args.flip_frac, n_frames=args.n_frames)
    prepare(movie, Path(args.out) / f"{movie.stem}_cache")
