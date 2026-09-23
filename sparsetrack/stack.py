"""Binned keyframe averages and their registration to a common reference.

A cache directory holds:

- ``bins.npy``: float16 array (n_bins, height, width); bin ``b`` is the mean of
  the keyframes whose source frame lies in ``[b * frames_per_bin, (b + 1) * frames_per_bin)``.
- ``meta.json``: movie identity, binning, per-bin keyframe counts, and the per-bin
  shift ``(dx, dy)`` such that a point at reference coordinates ``(x, y)`` appears
  at ``(x + dx, y + dy)`` in bin ``b``. The reference is the mean of the first
  ``ref_bins`` bins (pre-germination in a sparse field).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from . import __version__
from .video import MovieInfo, iter_keyframes, probe

SCHEMA = "sparsetrack.cache.v1"


def build_bins(keyframes: Iterable[tuple[int, np.ndarray]], n_frames: int, frames_per_bin: int,
               shape: tuple[int, int], out_path: str | Path) -> np.ndarray:
    """Average keyframes into bins, writing a float16 array to ``out_path``.

    Returns the number of keyframes averaged into each bin.
    """
    n_bins = -(-n_frames // frames_per_bin)
    out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float16, shape=(n_bins, *shape))
    counts = np.zeros(n_bins, dtype=np.int64)
    acc = np.zeros(shape, dtype=np.float64)
    current = -1
    for frame, image in keyframes:
        b = frame // frames_per_bin
        if b >= n_bins:
            raise ValueError(f"keyframe {frame} lies beyond {n_frames} frames")
        if b != current:
            if b < current:
                raise ValueError("keyframes are not in order")
            if current >= 0:
                out[current] = (acc / counts[current]).astype(np.float16)
            acc[:] = 0.0
            current = b
        acc += image
        counts[b] += 1
    if current >= 0:
        out[current] = (acc / counts[current]).astype(np.float16)
    out.flush()
    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"bins without keyframes: {empty[:10]}")
    return counts


def _highpass(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    image = image.astype(np.float64)
    return image - cv2.GaussianBlur(image, (0, 0), sigma)


def estimate_shifts(bins: np.ndarray, ref_bins: int = 3, max_dev: float = 2.5, window: int = 3,
                    upsample: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (dx, dy) of each bin relative to the reference, robust to outliers.

    Upsampled-DFT cross-correlation of high-passed frames (scikit-image), which on
    real bins shifted by known sub-pixel amounts is accurate to <0.06 px, against
    up to 0.46 px for OpenCV's phase-correlation peak centroid. Any shift deviating
    more than ``max_dev`` px from the running median over +/- ``window`` bins is
    replaced by that median (a real stage jump persists and is kept).
    Returns (shifts, raw_shifts, outlier_mask).
    """
    from skimage.registration import phase_cross_correlation

    ref = _highpass(np.asarray(bins[:ref_bins], dtype=np.float64).mean(axis=0))
    raw = np.zeros((len(bins), 2))
    for b in range(len(bins)):
        shift, _, _ = phase_cross_correlation(ref, _highpass(np.asarray(bins[b], dtype=np.float64)),
                                              upsample_factor=upsample, normalization=None)
        raw[b] = (-shift[1], -shift[0])  # content displacement (dx, dy) of bin b relative to the reference
    median = np.array([np.median(raw[max(0, b - window):b + window + 1], axis=0) for b in range(len(raw))])
    outlier = np.hypot(*(raw - median).T) > max_dev
    shifts = np.where(outlier[:, None], median, raw)
    return shifts, raw, outlier


def movie_identity(path: str | Path) -> dict:
    stat = os.stat(path)
    return {"path": str(path), "name": Path(path).name, "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime)}


def prepare(movie: str | Path, out_dir: str | Path, frames_per_bin: int = 300, ref_bins: int = 3,
            log=print) -> dict:
    """Build the cache for ``movie`` in ``out_dir`` and return its metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    info: MovieInfo = probe(movie)
    log(f"{info.path}: {info.n_frames} frames, {len(info.keyframes)} keyframes "
        f"(interval {info.keyframe_interval}), {info.width}x{info.height} @ {info.fps:g} fps")
    counts = build_bins(iter_keyframes(info), info.n_frames, frames_per_bin,
                        (info.height, info.width), out_dir / "bins.npy")
    log(f"binned into {len(counts)} bins of {frames_per_bin} frames in {time.time() - started:.0f} s")
    bins = np.load(out_dir / "bins.npy", mmap_mode="r")
    shifts, raw, outlier = estimate_shifts(bins, ref_bins=ref_bins)
    jumps = [int(b) for b in np.flatnonzero(np.hypot(*np.diff(shifts, axis=0).T) > 3.0) + 1]
    log(f"registration: {int(outlier.sum())} outlier shift(s) replaced; jumps > 3 px at bins {jumps}")
    meta = {
        "schema": SCHEMA,
        "sparsetrack_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "movie": {**movie_identity(movie), "width": info.width, "height": info.height,
                  "n_frames": info.n_frames, "fps_container": info.fps,
                  "n_keyframes": len(info.keyframes), "keyframe_interval": info.keyframe_interval},
        "frames_per_bin": frames_per_bin,
        "n_bins": int(len(counts)),
        "keyframes_per_bin": counts.tolist(),
        "ref_bins": ref_bins,
        "shifts": np.round(shifts, 3).tolist(),
        "raw_shifts": np.round(raw, 3).tolist(),
        "shift_outlier_bins": [int(b) for b in np.flatnonzero(outlier)],
        "jump_bins": jumps,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(cache_dir: str | Path) -> tuple[np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{cache_dir} is not a {SCHEMA} cache")
    return np.load(cache_dir / "bins.npy", mmap_mode="r"), meta
