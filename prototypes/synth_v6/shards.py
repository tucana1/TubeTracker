"""Training shards from synthetic movies, rendering only the bins a shard uses.

``prototypes.learned_flood.data.build`` samples 60-90 random bins of a finished synthetic movie (plus its
three "before" and three "after" reference bins), after the whole movie was rendered, encoded and binned.
Here the same bins are drawn first and only their frames are rendered and encoded (x264 intra frames with
the real movies' settings, ``synth.X264``), binned (25 frames per bin) and registered like
``stack.prepare`` does - 2.5-3.5x less rendering, and no 0.4-0.9 GB cache. The crops, targets and
normalisation are ``data.build``'s; the tube body comes from ``truth6`` (v6 bulbs and domes).

    python -m prototypes.synth_v6.shards --field runs/sparsetrack/m2 --preset v6 --seed 40 --m2 --out ...
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from prototypes.learned_flood.data import normalise, tip_heatmap
from prototypes.learned_flood.truth import crop
from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.synth import X264, Scene, preset
from sparsetrack.video import FFMPEG

from .truth6 import frame_truth

REPO = Path(__file__).resolve().parents[2]
M2 = dict(n_frames=351 * 25, onset_bins=(10.0, 250.0), rate_px_per_bin=(0.15, 1.5), max_length=280.0)
LOOK = dict(p_bright_core=0.8, width=(1.0, 1.6))  # movie 2's tubes: mostly light-cored, wider


def pick_bins(n_bins: int, n_pick: int, seed: int, rng_seed: int = 0) -> tuple[np.ndarray, np.random.Generator]:
    """``data.build``'s choice of bins (reference start 0)."""
    rng = np.random.default_rng(rng_seed + 1000 * seed)
    lo_b = 3
    chosen = np.sort(rng.choice(np.arange(lo_b, n_bins - 1), size=min(n_pick, n_bins - 1 - lo_b), replace=False))
    return chosen, rng


def render_bins(scene: Scene, bins: list[int], work: Path, name: str, log=print) -> tuple[np.ndarray, list[int]]:
    """Render and x264-encode the frames of ``bins``, decode them and average each bin: (len(bins), H, W)."""
    cfg, fpb = scene.cfg, scene.cfg.frames_per_bin
    frames = [k for b in bins for k in range(b * fpb, min((b + 1) * fpb, cfg.n_frames))]
    movie = work / f"{name}.mp4"
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{scene.w}x{scene.h}",
           "-r", "14/12", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-b:v", "12000k", "-x264-params", X264,
           "-pix_fmt", "yuv420p", str(movie)]
    started = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, k in enumerate(frames):
            proc.stdin.write(scene.render(k).tobytes())
            if i and i % 500 == 0:
                log(f"  {name}: {i}/{len(frames)} frames ({time.time() - started:.0f} s)")
    finally:
        proc.stdin.close()
        proc.wait()
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(movie), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                           stdout=subprocess.PIPE)
    size = scene.w * scene.h
    out = np.lib.format.open_memmap(work / f"{name}_bins.npy", mode="w+", dtype=np.float16,
                                    shape=(len(bins), scene.h, scene.w))
    counts = np.zeros(len(bins), np.int64)
    idx = {b: i for i, b in enumerate(bins)}
    acc, current = np.zeros((scene.h, scene.w), np.float64), None
    for k in frames:  # frames come bin by bin
        buf = dec.stdout.read(size)
        if len(buf) < size:
            raise RuntimeError(f"{movie}: decoded fewer frames than encoded")
        i = idx[k // fpb]
        if i != current:
            if current is not None:
                out[current] = (acc / counts[current]).astype(np.float16)
            acc[:] = 0.0
            current = i
        acc += np.frombuffer(buf, np.uint8).reshape(scene.h, scene.w)
        counts[i] += 1
    out[current] = (acc / counts[current]).astype(np.float16)
    out.flush()
    dec.stdout.close()
    dec.wait()
    movie.unlink()  # the shard keeps what training needs
    return out, counts.tolist()


def build(field: str | Path, preset_name: str, seed: int, out: str | Path, n_bins_pick: int = 60,
          crops_per_bin: int = 16, half: int = 48, pos_frac: float = 0.65, young_frac: float = 0.0,
          work: str | Path | None = None, log=print, **overrides) -> Path:
    out = Path(out)
    work = Path(work or out.parent)
    work.mkdir(parents=True, exist_ok=True)
    for stale in (work / f"{out.stem}.mp4", work / f"{out.stem}_bins.npy", out.with_name(out.stem + ".partial.npz")):
        stale.unlink(missing_ok=True)  # left by a run that was stopped
    started = time.time()
    cfg = preset(preset_name, seed=seed, **overrides)
    scene = Scene(field, cfg)
    fpb = cfg.frames_per_bin
    n_bins = -(-cfg.n_frames // fpb)
    chosen, rng = pick_bins(n_bins, n_bins_pick, seed)
    early, late = [0, 1, 2], list(range(n_bins - 4, n_bins - 1))
    needed = sorted(set(early) | set(late) | set(int(b) for b in chosen))
    log(f"{out.stem}: {len(scene.tubes)} tubes; rendering {len(needed)} of {n_bins} bins")
    arr, counts = render_bins(scene, needed, work, out.stem, log)
    shifts, raw, outlier = stack.estimate_shifts(arr, ref_bins=3)
    meta = {"frames_per_bin": fpb, "n_bins": len(needed), "shifts": np.round(shifts, 3).tolist(), "ref_start": 0}
    R = Renderer(arr, meta)
    pos = {b: i for i, b in enumerate(needed)}
    grains = json.loads((Path(field) / "grains.json").read_text())["grains"]
    h, w = scene.h, scene.w

    def sample(i: int, cx: float, cy: float) -> np.ndarray:
        img = R.crop(i, cx, cy, half)
        e = np.mean([R.crop(pos[b], cx, cy, half) for b in early], axis=0)
        la = np.mean([R.crop(pos[b], cx, cy, half) for b in late], axis=0)
        return normalise(img, e, la)
    xs, bodies, tipmaps, info = [], [], [], []
    n_young = 0
    for b in chosen:
        k = int(b) * fpb + fpb // 2
        tr = frame_truth(scene, k)
        ys, xs_ = np.nonzero(tr["body"])
        young = [(x, y) for x, y, _, L in tr["tips"] if L <= 12.0]
        for _ in range(crops_per_bin):
            u = rng.random()
            if young_frac > 0 and young and rng.random() < young_frac:  # a young tube's bulb in view
                x, y = young[int(rng.integers(len(young)))]
                cx, cy = x + rng.uniform(-half / 2, half / 2), y + rng.uniform(-half / 2, half / 2)
                n_young += 1
            elif u < pos_frac and len(ys):
                j = int(rng.integers(len(ys)))
                cx, cy = xs_[j] + rng.uniform(-half / 2, half / 2), ys[j] + rng.uniform(-half / 2, half / 2)
            elif u < pos_frac + 0.2:
                g = grains[int(rng.integers(len(grains)))]
                cx, cy = g["x"] + rng.uniform(-half / 2, half / 2), g["y"] + rng.uniform(-half / 2, half / 2)
            else:
                cx, cy = rng.uniform(half, w - half), rng.uniform(half, h - half)
            cx, cy = float(np.clip(round(cx), half + 4, w - half - 4)), float(np.clip(round(cy), half + 4, h - half - 4))
            xs.append(sample(pos[int(b)], cx, cy).astype(np.float16))
            bodies.append(crop(tr["body"], cx, cy, half, cv2.INTER_NEAREST).astype(np.uint8))
            tipmaps.append(tip_heatmap(tr["tips"], cx, cy, half).astype(np.float16))
            info.append((int(b), cx, cy))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez_compressed(tmp, x=np.stack(xs), body=np.stack(bodies), tip=np.stack(tipmaps),
                        info=np.array(info, np.float32), preset=preset_name, seed=seed,
                        overrides=json.dumps({k: v for k, v in overrides.items()}))
    tmp.rename(out)
    del R, arr
    (work / f"{out.stem}_bins.npy").unlink()
    log(f"{out.name}: {len(xs)} samples from {len(chosen)} bins ({n_young} young-centred), tube pixels "
        f"{100 * np.mean(np.stack(bodies)):.1f}%, registration outliers {int(outlier.sum())}, "
        f"{(time.time() - started) / 60:.1f} min")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="one training shard from a synthetic movie, rendering only its bins")
    ap.add_argument("--field", required=True)
    ap.add_argument("--preset", default="v6")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--m2", action="store_true", help="sized like movie 2 (351 bins, onsets to bin 250)")
    ap.add_argument("--look", action="store_true", help="movie 2's look: mostly light-cored, wider tubes")
    ap.add_argument("--young-frac", type=float, default=0.0)
    ap.add_argument("--set", nargs="*", default=[], help="extra SynthConfig key=value (tuples as a,b)")
    a = ap.parse_args(argv)
    over = {**(M2 if a.m2 else {}), **(LOOK if a.look else {})}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = tuple(float(x) for x in v.split(",")) if "," in v else (
            v.lower() == "true" if v.lower() in ("true", "false") else float(v))
    build(a.field, a.preset, a.seed, a.out, n_bins_pick=a.bins, young_frac=a.young_frac, **over)


if __name__ == "__main__":
    main()
