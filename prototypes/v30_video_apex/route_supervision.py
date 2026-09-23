"""Conservative, revision-linked judgments of proposed current routes.

A FULL human path licenses owner-route supervision, not generic non-cap
labels. Support beyond the current front is excluded from this judgment.
"""
from __future__ import annotations

import numpy as np

from .targets import polyline_point_projection, resample_polyline


def wrong_route_certificate(sample, polyline, *, current_s=None,
                            tolerance_px=5.0) -> dict:
    result = {"certified_wrong_route": False, "scope": "current_prefix",
              "obs_uuid": sample.obs_uuid, "obs_revision": sample.obs_revision,
              "entry_id": sample.entry_id, "reason": "unreviewed_full_path"}
    if (not sample.path_complete or sample.kind != "path_tip"
            or not sample.obs_uuid or sample.quarantine_reason
            or len(sample.path_xy) < 2):
        return result
    route = np.asarray(polyline, dtype=float)
    gold = np.asarray(sample.path_xy, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 2 or len(route) < 2
            or not np.isfinite(route).all()):
        return dict(result, reason="invalid_route")
    try:
        points, grid = resample_polyline(route)
    except ValueError:
        return dict(result, reason="degenerate_route")
    points, grid = np.asarray(points), np.asarray(grid)
    if current_s is None:
        # A mined route must reach the reviewed front before its prefix
        # can be evaluated; a short genuine partial is not a negative.
        distance, at, _ = polyline_point_projection(route, sample.tip_xy)
        if distance > 15.0:
            return dict(result, reason="current_front_not_located")
        current_s = at
    prefix = points[grid <= float(current_s)]
    if len(prefix) < 12:
        return dict(result, reason="insufficient_current_prefix")
    seg = gold[1:] - gold[:-1]
    denom = np.maximum((seg * seg).sum(1), 1e-9)
    delta = prefix[:, None, :] - gold[:-1][None, :, :]
    along = np.clip((delta * seg).sum(2) / denom, 0, 1)
    projected = gold[:-1][None, :, :] + along[..., None] * seg
    distances = np.linalg.norm(prefix[:, None, :] - projected, axis=2).min(1)
    # Ignore the immediate root neighborhood: an approximate grain centre
    # or the blind support's short pre-root extension is not tube truth.
    reviewed = np.linalg.norm(prefix - gold[0], axis=1) > 10.0
    outside = distances[reviewed] > tolerance_px
    enough = int(reviewed.sum()) >= 12
    wrong = enough and float(outside.mean()) > 0.30
    return dict(result, certified_wrong_route=bool(wrong),
                reason="deviates_from_full_owned_path" if wrong else
                       "no_certified_deviation", current_s=float(current_s),
                reviewed_points=int(reviewed.sum()),
                outside_fraction=float(outside.mean()) if len(outside) else 0.0,
                tolerance_px=float(tolerance_px))
