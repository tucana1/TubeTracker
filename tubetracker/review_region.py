"""Reviewed-extent validation shared by the annotation app and the
training/eval code (rev9 WP-A.1).

A "reviewed extent" is the region an annotator actually looked at. A
`complete` mask may claim background only inside it; material outside
stays unknown. Historical extents recorded by an early napari build
read the camera centre's z component as y, leaving every extent y
symmetric about 0 (H306/H318) — those rasterise into crops and would
turn unreviewed pixels into supervised background, so they must be
quarantined before they reach a target.

The functions here are pure geometry (numpy only) so both the live app
and the offline pipeline can use the same rule.
"""
from __future__ import annotations

import numpy as np

QUARANTINE_REASONS = ("missing", "shape", "nonfinite", "degenerate",
                      "misses-paint")

REVIEW_GEOMETRY_SCHEMA = 'tubetracker.review_geometry.v1'


def licensed_review_region(review_region, provenance=None) -> tuple[list, str]:
    """Resolve a reviewed field without guessing legacy canvas axis order.

    Old two-corner canvas rectangles did not record width/height order.
    Keep only their intersection with the swapped-size interpretation;
    explicit painted foreground is independent of this background licence.
    Versioned captures must reproduce the supplied geometry. Polygons are
    declared fields rather than canvas-size reconstructions.
    """
    try:
        pts = np.asarray(review_region, dtype=float)
    except (TypeError, ValueError):
        return review_region, 'invalid_geometry'
    if pts.size == 0:
        return [], 'missing'
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        return review_region, 'invalid_geometry'
    if provenance:
        try:
            if provenance.get('schema') != REVIEW_GEOMETRY_SCHEMA:
                return [], 'invalid_provenance'
            kind = provenance['kind']
            if kind == 'declared_field':
                declared = np.asarray(provenance['region_xy'], float)
                valid = declared.shape == pts.shape and np.allclose(declared, pts, rtol=0, atol=1e-6)
            elif kind == 'native_canvas':
                size = np.asarray(provenance['canvas_size_wh'], float)
                centre = np.asarray(provenance['camera_center_xy'], float)
                zoom = float(provenance['camera_zoom'])
                if (size.shape != (2,) or centre.shape != (2,) or not np.isfinite(size).all()
                        or not np.isfinite(centre).all() or not np.isfinite(zoom)
                        or (size <= 0).any() or zoom <= 0):
                    return [], 'invalid_provenance'
                expected = np.stack((centre-size/(2*zoom), centre+size/(2*zoom)))
                valid = pts.shape == (2,2) and np.allclose(expected, pts, rtol=0, atol=1e-6)
            else:
                return [], 'invalid_provenance'
            return (pts.tolist(), kind) if valid else ([], 'invalid_provenance')
        except (AttributeError, KeyError, TypeError, ValueError):
            return [], 'invalid_provenance'
    if len(pts) != 2:
        return pts.tolist(), 'declared_polygon'
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre, half = (lo+hi)/2, float((hi-lo).min())/2
    if half <= 0:
        return pts.tolist(), 'invalid_geometry'
    return [(centre-half).tolist(), (centre+half).tolist()], 'legacy_rectangle_intersection'


def extent_hits_paint_xy(review_region, paint_bbox) -> bool:
    """Does a native reviewed extent overlap the paint's bbox?"""
    try:
        pts = np.asarray(review_region, dtype=float)
    except (TypeError, ValueError):
        return False
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return False
    if not np.isfinite(pts).all():
        return False
    ex = (float(pts[:, 0].min()), float(pts[:, 0].max()))
    ey = (float(pts[:, 1].min()), float(pts[:, 1].max()))
    bx0, by0, bx1, by1 = (float(paint_bbox[0]), float(paint_bbox[1]),
                          float(paint_bbox[2]), float(paint_bbox[3]))
    return (ex[0] <= bx1 and bx0 <= ex[1] and ey[0] <= by1 and by0 <= ey[1])


def review_region_usable(review_region, paint_bbox) -> tuple[bool, str]:
    """May this extent licence reviewed-background supervision for a
    paint with this bbox? Fail closed.

    Returns (ok, reason); reason is 'ok' when ok, else one of
    'missing' | 'shape' | 'nonfinite' | 'degenerate' | 'misses-paint'.
    """
    if not review_region:
        return False, "missing"
    try:
        pts = np.asarray(review_region, dtype=float)
    except (TypeError, ValueError):
        return False, "shape"
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return False, "shape"
    if not np.isfinite(pts).all():
        return False, "nonfinite"
    if pts.shape[0] == 2 and (pts[0, 0] == pts[1, 0]
                              or pts[0, 1] == pts[1, 1]):
        return False, "degenerate"
    if not extent_hits_paint_xy(pts, paint_bbox):
        return False, "misses-paint"
    return True, "ok"


def paint_bbox_xy(points_xy) -> tuple[float, float, float, float] | None:
    """Bbox (x0, y0, x1, y1) of native paint points, or None."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2) \
        if points_xy is not None else np.zeros((0, 2))
    if pts.shape[0] == 0:
        return None
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def disc_bbox_xy(center_xy, radius: float
                 ) -> tuple[float, float, float, float] | None:
    """Bbox of a disc, for ball-scoped reviews ('no tube' verdicts)."""
    if not center_xy or len(center_xy) < 2:
        return None
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = max(float(radius), 0.0)
    return (cx - r, cy - r, cx + r, cy + r)
