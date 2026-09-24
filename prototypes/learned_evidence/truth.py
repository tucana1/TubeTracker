"""Exact per-frame truth rasters from a SparseTrack synthetic scene.

``sparsetrack.synth.Scene`` draws every tube from its centreline geometry (rotation,
drift, substrate anchoring and sideways sway included) but only writes sparse
benchmark labels. This module re-uses the same geometry to rasterise, for any frame,
the physical tube body (every point already built, visible or not yet matured) and
each tube's apex, in the un-drifted frame grid. SparseTrack's registered crops are
aligned to that grid, so a crop of these rasters taken with zero shift lines up with a
``Renderer.crop`` of the binned movie.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from sparsetrack.synth import Scene, Tube

# visible half-width of the two measured cross-sections (sparsetrack.synth.tube_profile):
# the dark line falls to 20% of its peak at 2.3 px, the bright core's walls end near 4.9 px
HALF_WIDTH = {False: 2.0, True: 4.0}


def _tube_maps(scene: Scene, i: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(arclength, distance) maps of tube ``i`` at frame ``k``, as ``Scene.render`` computes them,
    plus the map window origin ``lo``."""
    t = scene.tubes[i]
    lo, hi, xx, yy, s, d = scene.maps[i]
    rot = t.rot is not None and abs(t.rot[k]) > 1e-3
    mv = t.move is not None and bool(np.any(np.abs(t.move[k]) > 1e-3))
    if t.anchor is not None and bool(np.any(np.abs(t.anchor[k]) > 0.25)):
        return (*t.anchored_maps(k, xx, yy), lo, hi)
    if t.sway is not None and abs(t.sway[k]) > 0.125 and not (rot or mv):
        return (*t.swayed_maps(k, xx, yy), lo, hi)
    if rot or mv:
        a = math.radians(-t.rot[k]) if rot else 0.0
        mx, my = t.move[k] if mv else (0.0, 0.0)
        gx, gy = t.grain["x"], t.grain["y"]
        px, py = xx - mx - gx, yy - my - gy
        rx = (gx + math.cos(a) * px - math.sin(a) * py - lo[0]).astype(np.float32)
        ry = (gy + math.sin(a) * px + math.cos(a) * py - lo[1]).astype(np.float32)
        return (cv2.remap(s, rx, ry, cv2.INTER_NEAREST, borderValue=1e6),
                cv2.remap(d, rx, ry, cv2.INTER_NEAREST, borderValue=1e6), lo, hi)
    return s, d, lo, hi


def tube_point(t: Tube, k: int, arclen: float) -> tuple[float, float]:
    """Frame-grid position of the point ``arclen`` px along tube ``t`` at frame ``k``."""
    idx = int(np.clip(round(arclen / 0.25), 0, len(t.path) - 1))
    x, y = (float(v) for v in t.path[idx])
    if t.anchor is not None and bool(np.any(np.abs(t.anchor[k]) > 0.25)):
        w = max(0.0, 1.0 - arclen / max(t.bend, 1e-6))
        dx, dy = (round(float(v) * 2) / 2 for v in t.anchor[k])
        return x + w * dx, y + w * dy
    rot = t.rot is not None and abs(t.rot[k]) > 1e-3
    mv = t.move is not None and bool(np.any(np.abs(t.move[k]) > 1e-3))
    if t.sway is not None and abs(t.sway[k]) > 0.125 and not (rot or mv):
        tang = np.gradient(t.path, axis=0)[idx]
        tang = tang / (np.linalg.norm(tang) + 1e-9)
        w = min(1.0, arclen / max(t.sway_base, 1e-6))
        off = round(t.sway[k] * 4) / 4.0
        return x - w * tang[1] * off, y + w * tang[0] * off
    if rot or mv:
        a = math.radians(t.rot[k]) if rot else 0.0
        gx, gy = t.grain["x"], t.grain["y"]
        px, py = x - gx, y - gy
        x, y = gx + math.cos(a) * px - math.sin(a) * py, gy + math.sin(a) * px + math.cos(a) * py
        if mv:
            x, y = x + float(t.move[k][0]), y + float(t.move[k][1])
    return x, y


def frame_truth(scene: Scene, k: int, scored_only: bool = False) -> dict:
    """Truth rasters at frame ``k``: ``body`` (uint8 0/1, the built tube within its visible
    half-width), ``instance`` (int16, 1 + tube index, 0 = none; later tubes win overlaps) and
    ``tips``: list of (x, y, tube index, length_px) for tubes that exist at frame ``k``."""
    body = np.zeros((scene.h, scene.w), np.uint8)
    inst = np.zeros((scene.h, scene.w), np.int16)
    tips = []
    for i, t in enumerate(scene.tubes):
        if scored_only and not t.scored:
            continue
        L = float(t.length(np.array([float(k)]))[0])
        if L <= 0:
            continue
        s, d, lo, hi = _tube_maps(scene, i, k)
        on = (s <= L) & (d <= HALF_WIDTH[t.bright] * t.width)
        win = (slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
        body[win][on] = 1
        inst[win][on] = i + 1
        tips.append((*tube_point(t, k, L), i, L))
    return {"body": body, "instance": inst, "tips": tips}


def crop(arr: np.ndarray, cx: float, cy: float, half: int, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Crop of a frame-grid raster in ``Renderer.crop``'s convention with zero shift."""
    sub = np.ascontiguousarray(arr, dtype=np.float32)
    if interp == cv2.INTER_NEAREST:
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        out = np.zeros((2 * half, 2 * half), np.float32)
        ys, xs = slice(max(0, y0), min(arr.shape[0], y0 + 2 * half)), slice(max(0, x0), min(arr.shape[1], x0 + 2 * half))
        out[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = sub[ys, xs]
        return out
    return cv2.getRectSubPix(sub, (2 * half, 2 * half), (cx - 0.5, cy - 0.5))
