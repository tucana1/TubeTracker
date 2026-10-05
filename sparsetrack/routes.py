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


EXT_BACK_PX = 10.0   # a route is carried on past its end along its direction over about this many px (was 3: a
                     # last curl carried on drew a hook)
LOOP_END_PX = 20.0   # a loop a model route closes within its last this many px is not drawn (tidy)


def cut(path, length: float, max_extend: float = 5.0) -> np.ndarray:
    """The route cut to ``length`` px of arc, or carried on along its direction over its last ``EXT_BACK_PX`` px by
    at most ``max_extend`` px (``tubetracker.app.overlay.to_length``)."""
    p = np.asarray(path, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    s = arc(p)
    if length <= s[-1]:
        i = max(1, int(np.searchsorted(s, length)))
        a = (length - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
        return np.vstack([p[:i], p[i - 1] + a * (p[i] - p[i - 1])])
    back = next((q for q in p[-2::-1] if np.hypot(*(p[-1] - q)) >= EXT_BACK_PX), p[0])
    d = p[-1] - back
    n = float(np.hypot(*d)) or 1.0
    return np.vstack([p, p[-1] + d / n * min(length - s[-1], max_extend)])


def tidy(path) -> np.ndarray:
    """A model route without a curl at its end (the drawing only): where a point of its last ``LOOP_END_PX`` px
    comes back within 1 px of the route 4 to ``LOOP_END_PX`` px of arc before it, the loop between is cut out. Such
    loops are a route's end curling back over its own tube (a tip continuation that turned the wrong way); the tube
    is drawn straight on instead (``cut``). A tube that turns back runs a tube's width from itself, or meets its
    route farther back (left as it is)."""
    p = np.asarray(path, float).reshape(-1, 2)
    if len(p) < 4:
        return p
    q, s = resample(p, 1.0)
    for k in range(len(q) - 1, int(np.searchsorted(s, s[-1] - LOOP_END_PX)) - 1, -1):
        j = np.flatnonzero((np.hypot(*(q[:k] - q[k]).T) <= 1.0) & (s[k] - s[:k] > 4.0) & (s[k] - s[:k] <= LOOP_END_PX))
        if len(j):
            return np.vstack([q[:j[0] + 1], q[k + 1:]])
    return p


def dist_to(pts, line) -> np.ndarray:
    """Each point's distance to the polyline ``line``."""
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.asarray(line, float).reshape(-1, 2)
    if len(q) < 2:
        return np.hypot(*(p - q[0]).T) if len(q) else np.full(len(p), np.inf)
    a, d = q[:-1], np.diff(q, axis=0)
    t = np.clip(((p[:, None] - a[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-9)[None], 0.0, 1.0)
    return np.min(np.hypot(*(p[:, None] - a[None] - t[..., None] * d[None]).transpose(2, 0, 1)), axis=1)


def extends(old, new, mean_px: float = 1.5, max_px: float = 3.0) -> bool:
    """Whether route ``new`` carries route ``old`` on: every point of ``old`` within ``max_px`` of it, ``mean_px`` on
    average (``learned._extends``)."""
    d = dist_to(old, new)
    return bool(len(d)) and float(d.mean()) <= mean_px and float(d.max()) <= max_px


def keep_once(per_bin: list, base: dict | None = None, step: float = 2.0) -> dict | None:
    """Routes drawn per bin as ``path_by_bin`` (``{"routes": [...], "index": [per bin]}``): ``per_bin[i]`` a route,
    or None where bin ``i`` keeps ``base``'s route (or the stored one); a run of routes that extend one another is
    kept once, as its longest (each bin cuts it to its length); routes every ``step`` px. None if no bin has one."""
    kept = [list(r) for r in (base or {}).get("routes") or []]
    old = list((base or {}).get("index") or [])
    index, group = [], None
    for i, r in enumerate(per_bin):
        if r is None:
            index.append(old[i] if i < len(old) else -1)
            group = None
            continue
        q = np.round(resample(r, step)[0], 1)
        if group is not None and extends(np.asarray(kept[group], float), q):
            kept[group] = q.tolist()  # the longer route carries the group's on: it stands for them all
        elif group is None or not extends(q, np.asarray(kept[group], float)):
            kept.append(q.tolist())
            group = len(kept) - 1
        index.append(group)
    used = sorted({k for k in index if k >= 0})
    if not used:
        return None
    new = {k: j for j, k in enumerate(used)}
    return {"routes": [kept[k] for k in used], "index": [new.get(k, -1) for k in index]}


def route_at(res: dict, i: int) -> np.ndarray:
    """The route a reading draws at bin index ``i``: its own route for that bin where a reader read the tube along
    another one then (``path_by_bin``), else the stored route bent as the tube lay then; ``tidy``."""
    by_bin = res.get("path_by_bin") or {}
    index = by_bin.get("index") or []
    if 0 <= i < len(index) and index[i] >= 0:
        return tidy(by_bin["routes"][index[i]])
    return tidy(bent(res.get("path") or [], res.get("bend"), i))
