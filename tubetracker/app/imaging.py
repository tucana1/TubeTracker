"""The movie as 8-bit images: registered keyframe bins read from the cache (memory-mapped).

- ``field(b, mode)``: the whole field at bin ``b`` in reference coordinates (every bin registered to the reference,
  so the grains and their marks stay put while the movie plays). Modes: ``n`` normal (one grey window for the whole
  movie), ``h`` high contrast (a narrow window about the background) and ``g`` growth (this bin minus the bin 6
  bins earlier: a growing tip stands out, still material cancels; ``sparsetrack.render``).
- ``crop(b, x, y, half, zoom, mode, key)``: a close-up about a reference point (a grain where it is at that bin),
  contrast fixed per ``key`` (a grain), outside the frame hatched.

Recently used images are kept (a few dozen fields); reading and registering a bin takes a few milliseconds.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import GROWTH_LAG, GROWTH_WINDOW, Renderer

MODES = ("n", "h", "g")


class LRU:
    def __init__(self, size: int):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return value


class FrameSource:
    def __init__(self, cache_dir, fields: int = 48, crops: int = 256):
        self.bins, self.meta = stack.load(cache_dir)
        self.renderer = Renderer(self.bins, self.meta)
        self.n_bins = int(self.meta["n_bins"])
        self.height, self.width = self.bins.shape[1:]
        self.shifts = np.asarray(self.meta["shifts"], np.float64)
        self.ref_start = int(self.meta.get("ref_start", 0))
        self._fields, self._regs, self._crops = LRU(fields), LRU(fields), LRU(crops)
        self._windows: dict[str, tuple[float, float]] = {}
        self._wlock = threading.Lock()

    def registered(self, b: int) -> np.ndarray:
        """Bin ``b`` moved into reference coordinates (float32)."""
        b = int(b)
        img = self._regs.get(b)
        if img is not None:
            return img
        dx, dy = self.shifts[b]
        img = np.asarray(self.bins[b], np.float32)
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            img = cv2.warpAffine(img, np.float32([[1, 0, -dx], [0, 1, -dy]]), (self.width, self.height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self._regs.put(b, img)

    def window(self, mode: str) -> tuple[float, float]:
        """The grey window of a mode, fixed for the whole movie (no flicker as it plays)."""
        if mode == "g":
            return (-GROWTH_WINDOW, GROWTH_WINDOW)
        with self._wlock:
            if mode not in self._windows:
                rs = self.ref_start
                early = np.mean([self.registered(b)[::4, ::4] for b in range(rs, min(rs + 3, self.n_bins))], axis=0)
                late_bins = range(max(self.n_bins - 4, 0), max(self.n_bins - 1, 1))
                late = np.mean([self.registered(b)[::4, ::4] for b in late_bins], axis=0)
                if mode == "h":
                    bg = float(np.median(early))
                    self._windows[mode] = (bg - 20.0, bg + 12.0)
                else:
                    lo, hi = np.percentile(np.concatenate([early.ravel(), late.ravel()]), [0.3, 99.8])
                    self._windows[mode] = (float(lo), float(hi))
            return self._windows[mode]

    def field(self, b: int, mode: str = "n") -> np.ndarray:
        """The whole registered field at bin ``b`` as 8-bit grey (height, width)."""
        if mode not in MODES:
            raise ValueError(f"unknown view {mode!r}")
        b = int(np.clip(b, 0, self.n_bins - 1))
        key = (b, mode)
        img = self._fields.get(key)
        if img is not None:
            return img
        f = self.registered(b)
        if mode == "g":
            lag = min(GROWTH_LAG, b)
            f = f - self.registered(b - lag) if lag else np.zeros_like(f)
        lo, hi = self.window(mode)
        out = np.clip((f - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
        return self._fields.put(key, out)

    def crop(self, b: int, x: float, y: float, half: int = 48, zoom: float = 4.0, mode: str = "n",
             key: str = "") -> np.ndarray:
        """8-bit close-up (``2 * half * zoom`` px square) about reference (x, y) at bin ``b``."""
        if mode not in MODES:
            raise ValueError(f"unknown view {mode!r}")
        b, half = int(np.clip(b, 0, self.n_bins - 1)), int(np.clip(half, 8, 400))
        zoom = float(np.clip(zoom, 0.25, 12.0))
        x, y = round(float(x), 1), round(float(y), 1)
        k = (b, x, y, half, round(zoom, 2), mode, key)
        img = self._crops.get(k)
        if img is not None:
            return img
        R = self.renderer
        raw = (R.growth_crop if mode == "g" else R.mean_crop)(b, b, x, y, half, mark_outside=True)
        window = R.contrast(key, x, y, half, mode) if key else _local_window(raw, mode)
        return self._crops.put(k, R.to_display(raw, window, zoom))


def _local_window(img: np.ndarray, mode: str) -> tuple[float, float]:
    if mode == "g":
        return (-GROWTH_WINDOW, GROWTH_WINDOW)
    fin = img[np.isfinite(img)]
    if not fin.size:
        return (0.0, 255.0)
    lo, hi = np.percentile(fin, [0.5, 99.5])
    return float(lo), float(hi)
