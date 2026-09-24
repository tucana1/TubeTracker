"""Training crops from codec-exact synthetic movies with exact truth.

A sample is three registered crops of a prepared SparseTrack cache - the bin being read,
the "before" reference (first bins) and the "after" reference (last full bins) - the
same three images SparseTrack's evidence is computed from, normalised by the before
image's median. Targets are the built tube body at the bin's centre frame and a tip
heatmap, rasterised from the synthetic scene's own geometry (``truth.frame_truth``).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.synth import Scene, preset

from .truth import crop, frame_truth

SCALE = 20.0  # grey levels per input unit
TIP_SIGMA = 1.5


def normalise(img: np.ndarray, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 input from the bin, before and after crops (grey levels)."""
    m = float(np.nanmedian(early))
    x = np.stack([img, early, late]).astype(np.float32)
    return np.nan_to_num((x - m) / SCALE, nan=0.0)


class CacheView:
    """Registered crops of a cache with its before/after references."""

    def __init__(self, cache_dir: str | Path, ref_bins: int = 3, late_bins: int = 3):
        self.bins, self.meta = stack.load(cache_dir)
        self.r = Renderer(self.bins, self.meta)
        self.fpb = int(self.meta["frames_per_bin"])
        self.rs = int(self.meta.get("ref_start", 0))
        self.n_bins = int(self.meta["n_bins"])
        self.early_bins = list(range(self.rs, self.rs + ref_bins))
        self.late_bins = list(range(self.n_bins - 1 - late_bins, self.n_bins - 1))  # the last bin is often partial

    def frame(self, b: int) -> int:
        return b * self.fpb + self.fpb // 2

    def sample(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        img = self.r.crop(b, cx, cy, half)
        early = np.mean([self.r.crop(e, cx, cy, half) for e in self.early_bins], axis=0)
        late = np.mean([self.r.crop(e, cx, cy, half) for e in self.late_bins], axis=0)
        return normalise(img, early, late)


def tip_heatmap(tips, cx: float, cy: float, half: int, sigma: float = TIP_SIGMA) -> np.ndarray:
    size = 2 * half
    hm = np.zeros((size, size), np.float32)
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    for x, y, *_ in tips:
        u, v = x - (cx - half), y - (cy - half)  # crop pixel j is centred on frame column cx - half + j
        if -3 * sigma <= u < size + 3 * sigma and -3 * sigma <= v < size + 3 * sigma:
            hm = np.maximum(hm, np.exp(-((jj - u) ** 2 + (ii - v) ** 2) / (2 * sigma ** 2)))
    return hm


def build(field_cache: str | Path, movie_cache: str | Path, preset_name: str, seed: int, out: str | Path,
          n_bins: int = 60, crops_per_bin: int = 16, half: int = 48, pos_frac: float = 0.65,
          rng_seed: int = 0, log=print) -> Path:
    """Write one shard of training samples for a synthetic movie built by
    ``sparsetrack synth FIELD_CACHE --preset P --seed S`` and binned into ``movie_cache``."""
    scene = Scene(field_cache, preset(preset_name, seed=seed))
    view = CacheView(movie_cache)
    grains = json.loads((Path(field_cache) / "grains.json").read_text())["grains"]
    rng = np.random.default_rng(rng_seed + 1000 * seed)
    lo_b = view.rs + 3
    chosen = np.sort(rng.choice(np.arange(lo_b, view.n_bins - 1), size=min(n_bins, view.n_bins - 1 - lo_b),
                                replace=False))
    h, w = view.r.height, view.r.width
    xs, bodies, tipmaps, info = [], [], [], []
    for b in chosen:
        k = view.frame(int(b))
        tr = frame_truth(scene, k)
        ys, xs_ = np.nonzero(tr["body"])
        for _ in range(crops_per_bin):
            u = rng.random()
            if u < pos_frac and len(ys):
                j = int(rng.integers(len(ys)))
                cx, cy = xs_[j] + rng.uniform(-half / 2, half / 2), ys[j] + rng.uniform(-half / 2, half / 2)
            elif u < pos_frac + 0.2:
                g = grains[int(rng.integers(len(grains)))]
                cx, cy = g["x"] + rng.uniform(-half / 2, half / 2), g["y"] + rng.uniform(-half / 2, half / 2)
            else:
                cx, cy = rng.uniform(half, w - half), rng.uniform(half, h - half)
            cx, cy = float(np.clip(round(cx), half + 4, w - half - 4)), float(np.clip(round(cy), half + 4, h - half - 4))
            xs.append(view.sample(int(b), cx, cy, half).astype(np.float16))
            bodies.append(crop(tr["body"], cx, cy, half, cv2.INTER_NEAREST).astype(np.uint8))
            tipmaps.append(tip_heatmap(tr["tips"], cx, cy, half).astype(np.float16))
            info.append((int(b), cx, cy))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(xs), body=np.stack(bodies), tip=np.stack(tipmaps),
                        info=np.array(info, np.float32), preset=preset_name, seed=seed)
    log(f"{out.name}: {len(xs)} samples from {len(chosen)} bins "
        f"(tube pixels {100 * np.mean(np.stack(bodies)):.1f}%)")
    return out
