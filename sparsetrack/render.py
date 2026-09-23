"""Registered crops of binned frames, rendered for human review.

Coordinates are *reference* coordinates (see ``stack``). A rendered crop of
size ``2*half`` centred on reference ``(cx, cy)`` and upscaled by ``zoom`` maps a
canvas position ``u`` (CSS px, continuous) to reference ``x = cx - half + u / zoom``;
likewise for ``y``.
"""

from __future__ import annotations

import cv2
import numpy as np


class Renderer:
    def __init__(self, bins: np.ndarray, meta: dict):
        self.bins = bins
        self.meta = meta
        self.shifts = np.asarray(meta["shifts"], dtype=np.float64)
        self.n_bins = int(meta["n_bins"])
        self.height, self.width = bins.shape[1:]
        self._contrast: dict[tuple, tuple[float, float]] = {}

    def crop(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        """Registered float32 crop (2*half square) of bin ``b`` centred on reference (cx, cy)."""
        dx, dy = self.shifts[b]
        sx, sy = cx + dx, cy + dy
        pad = half + 3
        x0, y0 = int(np.floor(sx)) - pad, int(np.floor(sy)) - pad
        x1, y1 = x0 + 2 * pad + 2, y0 + 2 * pad + 2
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.width, x1), min(self.height, y1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return np.full((2 * half, 2 * half), np.nan, np.float32)
        sub = np.ascontiguousarray(self.bins[b, cy0:cy1, cx0:cx1], dtype=np.float32)
        # getRectSubPix centres the patch at (centre - (size-1)/2); shift by 0.5 so that
        # patch pixel j covers reference x in [cx - half + j, cx - half + j + 1).
        return cv2.getRectSubPix(sub, (2 * half, 2 * half), (sx - cx0 - 0.5, sy - cy0 - 0.5))

    def mean_crop(self, b0: int, b1: int, cx: float, cy: float, half: int) -> np.ndarray:
        b0, b1 = max(0, b0), min(self.n_bins - 1, b1)
        return np.mean([self.crop(b, cx, cy, half) for b in range(b0, b1 + 1)], axis=0)

    def contrast(self, key, cx: float, cy: float, half: int, mode: str = "n") -> tuple[float, float]:
        """Display window for one grain, fixed across all its tiles."""
        cache_key = (key, half, mode)
        if cache_key not in self._contrast:
            early = self.mean_crop(0, 2, cx, cy, half)
            late = self.mean_crop(self.n_bins - 4, self.n_bins - 2, cx, cy, half)
            if mode == "h":  # high contrast: narrow window around the background level
                bg = float(np.median(early))
                window = (bg - 20.0, bg + 12.0)
            else:
                values = np.concatenate([early.ravel(), late.ravel()])
                window = tuple(float(v) for v in np.percentile(values, [0.5, 99.5]))
            self._contrast[cache_key] = window
        return self._contrast[cache_key]

    @staticmethod
    def to_display(image: np.ndarray, window: tuple[float, float], zoom: float) -> np.ndarray:
        lo, hi = window
        u8 = np.clip((np.nan_to_num(image, nan=hi) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        if zoom != 1:
            size = (int(round(u8.shape[1] * zoom)), int(round(u8.shape[0] * zoom)))
            u8 = cv2.resize(u8, size, interpolation=cv2.INTER_CUBIC if zoom > 1 else cv2.INTER_AREA)
        return u8

    def strip(self, key, cx: float, cy: float, ranges: list[tuple[int, int]], labels: list[str],
              half: int, zoom: float, cols: int, mode: str = "n", header: int = 16, gap: int = 2
              ) -> np.ndarray:
        """Grid of tiles, one per bin range, each with a text header above the image."""
        window = self.contrast(key, cx, cy, half, mode)
        side = int(round(2 * half * zoom))
        rows = -(-len(ranges) // cols)
        sheet = np.full((rows * (side + header + gap), cols * (side + gap)), 255, np.uint8)
        for i, ((b0, b1), label) in enumerate(zip(ranges, labels)):
            r, c = divmod(i, cols)
            top, left = r * (side + header + gap), c * (side + gap)
            tile = self.to_display(self.mean_crop(b0, b1, cx, cy, half), window, zoom)
            sheet[top + header:top + header + side, left:left + side] = tile[:side, :side]
            cv2.putText(sheet, label, (left + 3, top + header - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1,
                        cv2.LINE_AA)
        return sheet

    def field(self, which: str = "early", scale: float = 0.75) -> np.ndarray:
        """Whole registered field (mean of three bins) at ``scale``."""
        bins = [0, 1, 2] if which == "early" else [self.n_bins - 4, self.n_bins - 3, self.n_bins - 2]
        acc = np.zeros((self.height, self.width), np.float64)
        for b in bins:
            dx, dy = self.shifts[b]
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            acc += cv2.warpAffine(np.asarray(self.bins[b], np.float32), m, (self.width, self.height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mean = acc / len(bins)
        lo, hi = np.percentile(mean, [0.5, 99.8])
        u8 = np.clip((mean - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        size = (int(round(self.width * scale)), int(round(self.height * scale)))
        return cv2.resize(u8, size, interpolation=cv2.INTER_AREA)


def png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
