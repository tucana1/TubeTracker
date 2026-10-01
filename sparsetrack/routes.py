"""A tube's route as it lay at each bin: the stored route bent sideways by that bin's offsets.

A reading stores one route (its tube at the end, centred on the tube: ``learned.centre_route``). Tubes bend and are
pushed as they grow, so the route also carries, per bin, sideways offsets every ``step_px`` px along it
(``bend``: ``{"step_px": 5.0, "px10": [[offset in 0.1 px, ...] per bin]}``, positive along the route's left
normal, an empty row where there is nothing to bend). The analysis, the review, the gallery and the app all bend it
the same way, through ``bent``.
"""

from __future__ import annotations

import numpy as np


def resample(path, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The polyline every ``step`` px of arc (its end kept), and those arc lengths."""
    p = np.asarray(path, float).reshape(-1, 2)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if len(p) < 2 or s[-1] <= 0:
        return p, s[:len(p)]
    q = np.arange(0.0, s[-1], step)
    q = np.append(q, s[-1]) if s[-1] - q[-1] > 1e-6 else q
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], axis=1), q


def arc(pts) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) if len(p) else np.zeros(0)


def normals(pts, k: int = 3) -> np.ndarray:
    """Unit left normals (-ty, tx) of a polyline, its tangent smoothed over ``2k + 1`` points."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros_like(p)
    tan = np.gradient(p, axis=0)
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    tan = np.stack([np.convolve(np.pad(tan[:, j], k, mode="edge"), ker, "valid") for j in range(2)], axis=1)
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    return np.stack([-tan[:, 1], tan[:, 0]], axis=1)


def offsets_along(s: np.ndarray, row, step: float) -> np.ndarray:
    """A bend row's offsets (px) at arc lengths ``s``: linear between its knots, the last knot's beyond them."""
    knots = np.arange(len(row)) * step
    return np.interp(s, knots, np.asarray(row, float) / 10.0)


def bent(path, bend: dict | None, i: int) -> np.ndarray:
    """``path`` bent by row ``i`` of ``bend`` (the path itself where there is no such row)."""
    p = np.asarray(path, float).reshape(-1, 2)
    rows = (bend or {}).get("px10") or []
    if len(p) < 2 or not 0 <= i < len(rows) or not rows[i]:
        return p
    return p + offsets_along(arc(p), rows[i], float(bend["step_px"]))[:, None] * normals(p)


def cut(path, length: float, max_extend: float = 5.0) -> np.ndarray:
    """The route cut to ``length`` px of arc, or carried on along its last direction by at most ``max_extend`` px
    (``tubetracker.app.overlay.to_length``)."""
    p = np.asarray(path, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    s = arc(p)
    if length <= s[-1]:
        i = max(1, int(np.searchsorted(s, length)))
        a = (length - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
        return np.vstack([p[:i], p[i - 1] + a * (p[i] - p[i - 1])])
    back = next((q for q in p[-2::-1] if np.hypot(*(p[-1] - q)) >= 3.0), p[0])
    d = p[-1] - back
    n = float(np.hypot(*d)) or 1.0
    return np.vstack([p, p[-1] + d / n * min(length - s[-1], max_extend)])
