"""v30 -> v29 adapter: route+front to tip/path/length export.

Refuses tip-only to length: a length needs compatible route geometry.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def _polyline_length(poly: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _truncate_at_s(poly: list, s: float) -> tuple[list, list[float], float]:
    """Walk arclength s along poly; return (truncated, endpoint, length)."""
    import math

    if s <= 0:
        p0 = [float(poly[0][0]), float(poly[0][1])]
        return [p0, p0], p0, 0.0
    out = [[float(poly[0][0]), float(poly[0][1])]]
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= s:
            f = (s - acc) / seg if seg > 0 else 0.0
            end = [x0 + f * (x1 - x0), y0 + f * (y1 - y0)]
            out.append([float(end[0]), float(end[1])])
            return out, [float(end[0]), float(end[1])], float(s)
        out.append([float(x1), float(y1)])
        acc += seg
    end = [float(poly[-1][0]), float(poly[-1][1])]
    return out, end, float(acc)


def _finite_number(v: Any) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"refuse: non-finite geometry value {v!r}")
    return f


def export_tip_path_length(route_polyline: Any, front_s: float | None,
                           tip_xy: Sequence[float] | None = None,
                           status: str = "direct",
                           tip_tolerance_px: float = 5.0) -> dict:
    """Export consistent tip/path/length from ONE selected route curve.

    The tip is the curve TRUNCATED at front_s (not the route endpoint);
    length is that truncated arclength. Raises ValueError for tip-only
    length requests and for supplied tips incompatible with the
    truncated end (beyond tip_tolerance_px).

    rev6: non-finite inputs (NaN/inf front, NaN vertices) are refused
    outright — the old code returned the full route as a direct
    measurement for NaN fronts. Out-of-range fronts (s < 0 or beyond
    route support) are censored to the nearest supported end and
    reported via "withheld": "front-censored", never silently.
    """
    if route_polyline is None:
        if front_s is not None:
            raise ValueError("refuse: cannot compute length from front_s without route geometry")
        if tip_xy is not None:
            # tip-only correction: return tip, withhold length
            # (rev6: a corrected point alone never manufactures a path).
            tx, ty = _finite_number(tip_xy[0]), _finite_number(tip_xy[1])
            return {"tip": [tx, ty], "path": None, "length_px": None,
                    "status": status, "withheld": "length-needs-route"}
        raise ValueError("refuse: tip-only input carries no length information")
    poly = [[_finite_number(q[0]), _finite_number(q[1])]
            for q in route_polyline]
    if len(poly) < 2:
        raise ValueError("refuse: route needs >= 2 vertices")
    support = _polyline_length(poly)
    censored = None
    if front_s is None:
        trunc, end, length = poly, poly[-1], support
    else:
        s = _finite_number(front_s)
        if s < 0 or s > support:
            censored = "front-censored"
            s = min(max(s, 0.0), support)
        trunc, end, length = _truncate_at_s(poly, s)
    if tip_xy is not None:
        tx, ty = _finite_number(tip_xy[0]), _finite_number(tip_xy[1])
        d = math.hypot(tx - end[0], ty - end[1])
        if d > tip_tolerance_px:
            raise ValueError(
                f"refuse: supplied tip {[tx, ty]} disagrees with "
                f"front-truncated end {end} by {d:.1f}px")
    return {"tip": end, "path": trunc, "length_px": float(length),
            "status": status, "withheld": censored}
