"""v30 label targets: snapshot observations -> supervised tensors (rev5 #2).

Unknown-by-default: every head has an explicit validity mask. Verified
tips become positive discs, verified-negative regions become negatives,
and everything unreviewed stays 0 (no gradient). A foreign tube's real
cap is never a negative for the generic apex head — only boxes drawn
with an apex-negative class scope suppress it, and only inside the box.

Headless: numpy here; torch tensors are built by the training script.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# rev9 WP-A.1: the reviewed-extent predicate lives in the app package so
# the live annotation UI and the offline pipeline share one rule. The
# names stay importable from here for existing callers/tests.
from tubetracker.review_region import (  # noqa: F401
    extent_hits_paint_xy, paint_bbox_xy, review_region_usable, licensed_review_region)

# 7 visibility classes; order matches the model's vis_head output.
VISIBILITY_INDEX = {
    "direct_visible": 0,
    "visible_imprecise": 1,
    "not_directly_visible": 2,
    "no_tube_visible": 3,
    "out_of_field": 4,
    "lost_identity": 5,
    "owner_uncertain": 5,
    "unresolved_overlap": 6,
}

# Region class scopes that count as apex negatives for the apex head.
APEX_NEGATIVE_SCOPES = frozenset({"apex", "tip", "cap"})


def load_clip_pixels(reader, frame_ids, crop_xywh,
                     missing_mask=()) -> np.ndarray:
    """Read grayscale float32 [0,1] clip (T,Hc,Wc), cropped to native box.

    Missing frames (missing_mask=1) become zeros — never repeated pixels.
    Raises on inexact reads: a label must never attach to a wrong frame.
    """
    x, y, w, h = (int(v) for v in crop_xywh)
    out = []
    for i, fid in enumerate(frame_ids):
        if missing_mask and missing_mask[i]:
            out.append(np.zeros((h, w), np.float32))
            continue
        res = reader.read(int(fid))
        if not res.exact:
            raise ValueError(
                f"inexact decode at source frame {fid} "
                f"(verified {res.verified_id}); refusing label attach")
        g = res.frame
        if g.ndim == 3:
            g = g[:, :, 0].astype(np.float32)  # BGR planes identical (gray)
        else:
            g = g.astype(np.float32)
        g = g / 255.0
        H, W = g.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        crop = np.zeros((h, w), np.float32)
        if x1 > x0 and y1 > y0:
            # Native origin is part of the geometry contract. Clamping
            # the origin must not translate all pixels in an edge crop.
            dx, dy = x0 - x, y0 - y
            crop[dy:dy + y1 - y0, dx:dx + x1 - x0] = g[y0:y1, x0:x1]
        out.append(crop)
    return np.stack(out).astype(np.float32)


def resample_polyline(path_xy, step: float = 1.0):
    """Resample a polyline at ~step px; returns (route_xy (S,2), s (S,))."""
    pts = np.asarray(path_xy, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("route needs >= 2 vertices")
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    total = float(seg.sum())
    if total <= 0:
        raise ValueError("degenerate zero-length route")
    n = max(int(total / step) + 1, 8)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.linspace(0.0, total, n)
    xs = np.interp(ss, s, pts[:, 0])
    ys = np.interp(ss, s, pts[:, 1])
    return np.stack([xs, ys], axis=1), ss


def polyline_point_projection(route_xy, point_xy):
    """rev12 P0.3: continuous point-to-polyline projection.

    ONE helper for the three quantities every consumer needs:
    (distance_px, projected_arclength_px, projected_native_xy).

    The projection is onto the continuous LINE SEGMENTS (not the stored
    vertices) — nearest-vertex distance systematically over-estimates
    proximity for sparse polylines. Repeated and zero-length vertices
    are handled by clamping the projection weight to the segment; a
    zero-length segment degenerates to its endpoint (w = 0).
    """
    r = np.asarray(route_xy, dtype=float)
    q = np.asarray(point_xy, dtype=float)
    if r.ndim != 2 or r.shape[0] == 0:
        return (float("inf"), float("nan"), (float("nan"),
                                             float("nan")))
    if r.shape[0] == 1:
        d = float(np.hypot(*(r[0] - q)))
        return (d, 0.0, (float(r[0, 0]), float(r[0, 1])))
    seg = r[1:] - r[:-1]
    sl = np.hypot(seg[:, 0], seg[:, 1])
    w = ((q[None, :] - r[:-1]) * seg).sum(axis=1) \
        / np.where(sl ** 2 > 0, sl ** 2, 1.0)
    w = np.clip(w, 0.0, 1.0)
    proj = r[:-1] + w[:, None] * seg
    d = np.hypot(*(proj - q).T)
    j = int(np.argmin(d))
    s = float(sl[:j].sum() + w[j] * sl[j])
    return (float(d[j]), s, (float(proj[j, 0]), float(proj[j, 1])))


def project_to_arclength(route_xy, tip_xy) -> float:
    """Arclength of the tip's nearest projection onto the route."""
    return polyline_point_projection(route_xy, tip_xy)[1]


def blind_oracle_route(path_xy, tip_xy, pre_px: float = 8.0,
                       post_px: float = 24.0) -> dict:
    """Endpoint-blinded oracle route: support extends beyond the cap.

    The resampled route starts pre_px before the root and continues
    post_px past the path end along the end tangent, so the last vertex
    is NOT the answer. Returns route_xy, s_grid, s_star (tip arclength)
    and support length. Raises if the tip is not strictly interior.
    """
    pts = np.asarray(path_xy, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("route needs >= 2 vertices")
    d0 = pts[1] - pts[0]
    d0 = d0 / max(float(np.hypot(*d0)), 1e-9)
    d1 = pts[-1] - pts[-2]
    d1 = d1 / max(float(np.hypot(*d1)), 1e-9)
    ext = np.vstack([pts[0] - d0 * pre_px, pts, pts[-1] + d1 * post_px])
    route_xy, s_grid = resample_polyline(ext.tolist())
    s_star = project_to_arclength(route_xy, np.asarray(tip_xy, float))
    total = float(s_grid[-1])
    if not (s_star > 2.0 and s_star < total - 8.0):
        raise ValueError(
            f"tip not interior to blinded support "
            f"(s*={s_star:.1f}, total={total:.1f})")
    return {"route_xy": route_xy, "s_grid": s_grid, "s_star": s_star,
            "support": total}


def apex_heat(h: int, w: int, tip_crop, sigma_px: float = 4.0) -> np.ndarray:
    """Gaussian apex target in crop coords."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xx - float(tip_crop[0])) ** 2 + (yy - float(tip_crop[1])) ** 2
    return np.exp(-d2 / (2.0 * sigma_px ** 2)).astype(np.float32)


def apex_valid_mask(h: int, w: int, origin, tip_crop_or_none,
                    regions_for_frame, tip_radius: float = 8.0) -> np.ndarray:
    """1.0 where apex supervision applies, else 0.0 (unknown).

    Positive disc around a precise tip + verified-negative boxes only.
    Prefer apex_pos_neg_masks (separate normalization); this union is
    kept for readouts that only need coverage.
    """
    pos, neg = apex_pos_neg_masks(h, w, origin, tip_crop_or_none,
                                  regions_for_frame, tip_radius)
    return np.maximum(pos, neg)


def apex_pos_neg_masks(h: int, w: int, origin, tip_crop_or_none,
                       regions_for_frame,
                       tip_radius: float = 8.0,
                       other_tips_crop: Sequence = ()) -> tuple[np.ndarray,
                                                        np.ndarray]:
    """Separate positive / verified-negative masks (P2: normalize each
    on its own, so abundant background zeros cannot outvote the few
    tip pixels through a shared denominator).

    other_tips_crop: additional known tip locations (crop coords) at
    this frame, e.g. banked caps from other observations. A tip pixel
    is never a negative (rev6: v30d-001's accepted box contains the
    p00 cap; disputed pixels stay unknown, never negative)."""
    pos = np.zeros((h, w), np.float32)
    if tip_crop_or_none is not None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot(xx - float(tip_crop_or_none[0]),
                     yy - float(tip_crop_or_none[1]))
        pos[d <= tip_radius] = 1.0
    neg = np.zeros((h, w), np.float32)
    ox, oy = float(origin[0]), float(origin[1])
    for r in regions_for_frame:
        if r.get("kind") != "verified_negative":
            continue
        # rev9 WP-A.6: an owned "no tube here" verdict must never
        # suppress the GENERIC terminal head — a real cap of a
        # neighbouring grain inside that disc is positive for this
        # head. Old snapshots recorded those as cap-scoped
        # verified_negative regions; they are refused here (fail
        # closed) rather than silently mis-supervising.
        if str(r.get("source", "")) == "mask-no-tube":
            continue
        if str(r.get("class_scope", "")) not in APEX_NEGATIVE_SCOPES:
            continue
        poly = np.asarray(r.get("polygon_xy", []), dtype=float)
        if poly.shape[0] < 3:
            continue
        # Intersection BEFORE rasterization (rev6): a box wholly
        # outside the crop must paint zero pixels, not a clipped
        # border sliver. Test in native coords first.
        nxs, nys = poly[:, 0], poly[:, 1]
        if (nxs.max() < ox or nxs.min() > ox + w - 1
                or nys.max() < oy or nys.min() > oy + h - 1):
            continue
        xs = (nxs - ox).clip(0, w - 1).astype(int)
        ys = (nys - oy).astype(int)
        # Boxes are axis-aligned rectangles (neg_click); fill bbox.
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(np.clip(ys.min(), 0, h - 1)), int(
            np.clip(ys.max(), 0, h - 1))
        neg[y0: y1 + 1, x0: x1 + 1] = 1.0
    neg[pos > 0] = 0.0  # a tip pixel is never a negative
    if other_tips_crop:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for ot in other_tips_crop:
            d = np.hypot(xx - float(ot[0]), yy - float(ot[1]))
            neg[d <= tip_radius] = 0.0
    return pos, neg


TUBE_HALF_WIDTH_PX = 4.0  # declared initial body half-width; refined
# by reviewed body masks when they exist (rev6: no masks yet, so the
# ribbon is centerline-derived, never presented as observed width).

HINT_MAX_ARCLENGTH_PX = 24.0  # rev8: attachment hints are short and
# arclength-bounded; a hint must never hand the terminal to the model.


def arclength_bounded_hint(path, max_px: float = HINT_MAX_ARCLENGTH_PX
                           ) -> list:
    """First path points within max_px of arclength from the attachment.

    rev8: the proximal hint used to be `path[:3]`, so a two-vertex
    path handed the model the entire tube AND its terminal. The hint
    now stops short of the terminal by construction.
    """
    import math

    pts = [[float(q[0]), float(q[1])] for q in (path or [])]
    if len(pts) < 2:
        return pts[:1]
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += math.hypot(pts[i][0] - pts[i - 1][0],
                          pts[i][1] - pts[i - 1][1])
        if acc > float(max_px):
            break
        out.append(pts[i])
    if len(out) >= len(pts):
        out = out[:-1]  # never include the terminal vertex
    return out


CENSUS_CAP_CLASSES = frozenset(('tip', 'tips', 'cap', 'caps', 'apex'))


def census_pos_neg_masks(h: int, w: int, origin,
                         tips_native, grains_native,
                         tile_box, complete: bool,
                         tip_radius: float = 8.0,
                         other_tips_crop: Sequence = (),
                         class_scopes: Sequence = ()
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Cap supervision from explicitly reviewed tips and cap-complete fields.

    Grain centres never identify caps. A grain-only census licenses no
    cap background, even if its grain count is exhaustive. Other banked
    caps carve out unknown areas; explicit tip positives retain priority.
    The grains argument remains accepted for existing census consumers.
    """
    pos = np.zeros((h, w), np.float32)
    neg = np.zeros((h, w), np.float32)
    if not tile_box or len(tile_box) != 4:
        return pos, neg
    ox, oy = float(origin[0]), float(origin[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, y0, x1, y1 = (float(tile_box[0]), float(tile_box[1]),
                      float(tile_box[2]), float(tile_box[3]))
    # intersection test before rasterization (a tile outside this crop
    # must paint nothing)
    if x1 < ox or x0 > ox + w - 1 or y1 < oy or y0 > oy + h - 1:
        return pos, neg
    inside = ((xx + ox >= x0) & (xx + ox <= x1) &
              (yy + oy >= y0) & (yy + oy <= y1))
    if complete and CENSUS_CAP_CLASSES.intersection(class_scopes):
        neg[inside] = 1.0
    for q in list(tips_native or []):
        cx, cy = float(q[0]) - ox, float(q[1]) - oy
        disc = (np.hypot(xx - cx, yy - cy) <= tip_radius) & inside
        pos[disc] = 1.0
        neg[disc] = 0.0
    for q in list(other_tips_crop or []):
        neg[np.hypot(xx - float(q[0]), yy - float(q[1])) <= tip_radius] = 0.0
    return pos, neg


def body_ribbon_mask(h: int, w: int, origin, path_crop,
                     visible, half_width: float = TUBE_HALF_WIDTH_PX
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Visible-tube-body target + validity from a centerline path.

    Target 1 inside the ribbon of VISIBLE segments, 0 in a near
    background band around them (both supervised); hidden segments and
    everywhere else are UNKNOWN (0 valid). Connectivity never decides
    identity — this is per-instance support evidence only.
    """
    target = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), np.float32)
    pts = np.asarray(path_crop, dtype=float)
    if pts.shape[0] < 2:
        return target, valid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts[:-1], pts[1:])):
        if i < len(visible) and not visible[i]:
            continue  # hidden material stays unknown, never negative
        seglen = math.hypot(x1 - x0, y1 - y0)
        if seglen < 1e-9:
            d = np.hypot(xx - x0, yy - y0)
        else:
            t = ((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) / seglen ** 2
            t = np.clip(t, 0, 1)
            d = np.hypot(xx - (x0 + t * (x1 - x0)),
                         yy - (y0 + t * (y1 - y0)))
        target[d <= half_width] = 1.0
        valid[(d <= half_width) | (d <= 3 * half_width)] = 1.0
    return target, valid


def _exclude_mask_unknown(ch, h, w, origin, raster):
    if not raster:
        return ch
    unknown = decode_mask_raster(h, w, origin, raster)
    for key in ('target', 'valid', 'fg', 'band', 'bg_reviewed'):
        ch[key][unknown] = 0
    ch['unknown'] |= unknown
    ch['explicit_unknown'] = unknown
    return ch


def body_mask_channels_from_points(h: int, w: int, origin, points_xy,
                                   brush_px: float = 9.0,
                                   complete: bool = False,
                                   review_region=None, unknown_raster=None,
                                   review_region_provenance=None) -> dict:
    """Channels for a brush-stamp paint (mirror of the raster version).

    rev9 WP-A.1: same named channels (`target` = fg, `band`,
    `bg_reviewed`, `unknown`) and the same fail-closed quarantine, so
    the trainer and the evaluator can share one code path regardless of
    how the mask was captured.
    """
    target = np.zeros((h, w), np.float32)
    band = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2) \
        if points_xy is not None else np.zeros((0, 2))
    fg = np.zeros((h, w), dtype=bool)
    info = {"extent_used": False, "quarantine_reason": "no-paint"}
    if pts.shape[0] == 0:
        return {"target": target, "valid": target.copy(), "fg": fg,
                "band": band, "bg_reviewed": bg,
                "unknown": np.ones((h, w), dtype=bool), **info}
    ox, oy = float(origin[0]), float(origin[1])
    r = max(float(brush_px) / 2.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dmin = np.full((h, w), np.inf, np.float32)
    for qx, qy in pts:
        cx, cy = float(qx) - ox, float(qy) - oy
        if cx < -r or cx > w - 1 + r or cy < -r or cy > h - 1 + r:
            continue  # outside this crop: no claim here
        dmin = np.minimum(dmin, np.hypot(xx - cx, yy - cy))
    fg = dmin <= r
    target[fg] = 1.0
    used_extent = False
    reason = "no-review-region"
    region, scope_policy = licensed_review_region(review_region, review_region_provenance)
    if complete:
        bg, used_extent, reason = _review_extent_background(
            h, w, origin, fg, region)
        if scope_policy == 'invalid_provenance':
            reason = 'invalid-provenance'
    if not used_extent:
        band = (dmin <= 3 * r) & ~fg
    valid = np.zeros((h, w), np.float32)
    valid[fg | band | bg] = 1.0
    return _exclude_mask_unknown({"target": target, "valid": valid, "fg": fg, "band": band,
            "bg_reviewed": bg, "unknown": ~(fg | band | bg),
            "extent_used": bool(used_extent),
            "extent_scope_policy": scope_policy,
            "quarantine_reason": ("ok" if used_extent else reason)}, h, w, origin, unknown_raster)


def body_mask_from_points(h: int, w: int, origin, points_xy,
                          brush_px: float = 9.0,
                          complete: bool = False,
                          review_region=None, review_region_provenance=None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Visible-tube-body target + validity from HUMAN-painted mask points.

    points_xy: native brush-stamp centers; the true mask is their union
    of discs (r = brush_px/2). Target 1 inside. Validity: the mask plus
    a near band are supervised (band target 0). If complete AND a USABLE
    reviewed extent is recorded, background is supervised 0 inside that
    extent only; unreviewed material stays unknown (rev8), and an
    unusable extent is quarantined (rev9 WP-A.1).
    A human mask REPLACES the centerline ribbon where present — never
    merged with it (the hand, not the old geometry, is the evidence).
    """
    ch = body_mask_channels_from_points(h, w, origin, points_xy,
                                        brush_px=brush_px,
                                        complete=complete,
                                        review_region=review_region,
                                        review_region_provenance=review_region_provenance)
    return ch["target"], ch["valid"]


def encode_mask_raster(mask_hw: np.ndarray) -> dict:
    """Lossless exact-raster encoding of a boolean paint array.

    rev7: stamp centers + disc redraw EXPAND paint (measured 364px ->
    766px, erased holes refill). The exact painted set is stored as a
    bbox plus zlib-compressed packed bits — pixel-exact roundtrip,
    stdlib only. Native full-frame coords; the loader crops.
    """
    import base64
    import zlib
    H, W = np.asarray(mask_hw).shape[:2]
    ys, xs = np.nonzero(np.asarray(mask_hw).reshape(H, W))
    if len(ys) == 0:
        return {"x0": 0, "y0": 0, "w": 0, "h": 0, "bits_b64": ""}
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bits = np.packbits(
        np.asarray(mask_hw)[y0:y1, x0:x1].reshape(-1).astype(np.uint8))
    return {"x0": x0, "y0": y0, "w": x1 - x0, "h": y1 - y0,
            "bits_b64": base64.b64encode(zlib.compress(
                bits.tobytes())).decode("ascii")}


def decode_mask_raster(h: int, w: int, origin, raster: dict
                       ) -> np.ndarray:
    """Exact painted set in crop coords (bool). Holes stay holes."""
    import base64
    import zlib
    out = np.zeros((h, w), dtype=bool)
    try:
        x0 = int(raster["x0"]) - int(origin[0])
        y0 = int(raster["y0"]) - int(origin[1])
        rw, rh = int(raster["w"]), int(raster["h"])
        if rw == rh == 0 and raster["bits_b64"] == "":
            return out
        if rw <= 0 or rh <= 0:
            raise ValueError("nonempty rasters require positive dimensions")
        raw = zlib.decompress(base64.b64decode(raster["bits_b64"], validate=True))
        if len(raw) != (rw * rh + 7) // 8:
            raise ValueError("raster dimensions disagree with packed data")
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        m = bits[:rw * rh].reshape(rh, rw).astype(bool)
    except (KeyError, TypeError, ValueError, zlib.error) as error:
        # A corrupt full-field mask must never become all-background supervision.
        raise ValueError(f"Invalid mask raster: {error}") from error
    xa, xb = max(0, x0), min(w, x0 + rw)
    ya, yb = max(0, y0), min(h, y0 + rh)
    if xa < xb and ya < yb:
        out[ya:yb, xa:xb] = m[ya - y0:yb - y0, xa - x0:xb - x0]
    return out


def refine_ring_anchor(anchor, path_xy, image, grain_r=13.0, r=14.0):
    """Turn a label-derived anchor into a marker a human reads as centred.

    Two measured corrections, in order:
    1. A path begins at the grain's EDGE, so step one grain radius
       upstream along the tube's first segment to reach the grain
       centre (p03: moves the anchor from 399.8 to 386.8, ~2 px from
       the FRST-detected grain).
    2. Snap onto the dark blob itself (`snap_to_local_blob`), because
       label geometry alone is only good to a few px and the eye judges
       the marker against the blob.

    Returns (xy, src, moved_px). Falls back to the label anchor when
    the frame has no dark structure (empty background) -- a marker is
    never invented where nothing is visible.
    """
    pts = [q for q in (path_xy or [])
           if isinstance(q, (list, tuple)) and len(q) == 2]
    base = [float(anchor[0]), float(anchor[1])]
    src = "label"
    if len(pts) >= 2:
        p0 = np.asarray(pts[0], dtype=np.float64)
        p1 = np.asarray(pts[1], dtype=np.float64)
        d = p0 - p1
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            base = [float(p0[0] + d[0] / n * grain_r),
                    float(p0[1] + d[1] / n * grain_r)]
            src = "grain-exit+1R"
    snapped, moved, ok = snap_to_local_blob(image, base, r=r)
    if ok:
        return snapped, src + "+blob-snap", moved
    return base, src + "+no-blob", 0.0


def measure_blob(image, xy, search_r=45.0, floor=0.35,
                 erode_px=6.0):
    """Measure the dark blob a ring is meant to enclose.

    `snap_to_local_blob` moves a point and says nothing about SIZE,
    which is the other half of the complaint: a ring that the grain
    fills edge to edge reads as sloppy even when it is centred
    (reported against the live app at r=20 px). So threshold the
    window, take the connected component under the anchor, and report
    its centroid, its area-equivalent radius and its bounding box --
    position AND size from the same measurement.

    Returns (centroid, r_equiv, area, bbox) or (None, 0.0, 0, None)
    when there is no dark structure there (empty background must not
    grow a blob out of noise). `erode_px` is the opening radius that
    severs the tube from the grain (tube half-width is ~4 px here).
    """
    from scipy import ndimage
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    if img.max() > 1.5:
        img = img / 255.0
    half = int(max(8, round(search_r)))
    x0 = int(max(0, np.floor(xy[0] - half)))
    x1 = int(min(img.shape[1], np.ceil(xy[0] + half) + 1))
    y0 = int(max(0, np.floor(xy[1] - half)))
    y1 = int(min(img.shape[0], np.ceil(xy[1] + half) + 1))
    sub = img[y0:y1, x0:x1]
    if sub.size < 16:
        return None, 0.0, 0, None
    med = float(np.median(sub))
    lo = float(np.percentile(sub, 2))
    if med - lo < 1e-6:
        return None, 0.0, 0, None
    mask = sub < (med - float(floor) * (med - lo))
    # The tube is CONTIGUOUS with the grain, so a plain component
    # measure merges them (measured: a 25x55 px component for a 27 px
    # grain, centroid dragged down the tube). Opening with a disc wider
    # than the tube's half-width severs it and leaves the grain.
    if erode_px > 0:
        er = float(erode_px)
        yy, xx = np.mgrid[-int(np.ceil(er)):int(np.ceil(er)) + 1,
                          -int(np.ceil(er)):int(np.ceil(er)) + 1]
        disc = (xx ** 2 + yy ** 2) <= er ** 2
        mask = ndimage.binary_opening(mask, structure=disc)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, 0.0, 0, None
    # the component under (or nearest to) the anchor
    ky = int(np.clip(round(xy[1]) - y0, 0, sub.shape[0] - 1))
    kx = int(np.clip(round(xy[0]) - x0, 0, sub.shape[1] - 1))
    target = lab[ky, kx]
    if target == 0:
        # nudge onto the nearest labelled pixel inside the window
        ys, xs = np.nonzero(lab)
        if len(ys) == 0:
            return None, 0.0, 0, None
        k = int(np.argmin((ys - ky) ** 2 + (xs - kx) ** 2))
        target = lab[ys[k], xs[k]]
    ys, xs = np.nonzero(lab == target)
    if len(ys) < 12:
        return None, 0.0, 0, None
    cx = x0 + float(xs.mean())
    cy = y0 + float(ys.mean())
    area = int(len(ys))
    r_equiv = float(np.sqrt(area / np.pi))
    bbox = (x0 + int(xs.min()), y0 + int(ys.min()),
            x0 + int(xs.max()), y0 + int(ys.max()))
    return [cx, cy], r_equiv, area, bbox


def snap_to_local_blob(image, xy, r=14.0, iters=3, floor=0.55):
    """Move a point onto the centre of the dark blob it sits on.

    A ring is a pointer, but it has to LOOK like it points at the
    grain: 7.8 px of error on a 14 px ring radius is more than half a
    radius and reads as off-centre on screen (reported against the
    live app). The offset is measured, not guessed -- a greedy
    intensity centroid (mean-shift, `iters` passes) over the darker
    pixels inside a window of about one ring radius, which converges on
    the blob the marker was meant for.

    Grains are DARK on the light background in this material. Returns
    (xy, moved_px, ok); `ok` is False -- with the input unchanged --
    when the window holds no dark structure, so a task on empty
    background is left exactly as the label put it.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    if img.max() > 1.5:
        img = img / 255.0
    cx, cy = float(xy[0]), float(xy[1])
    ox, oy = cx, cy
    for _ in range(max(1, int(iters))):
        half = max(4, int(round(r)))
        x0 = int(max(0, np.floor(cx - half)))
        x1 = int(min(img.shape[1], np.ceil(cx + half) + 1))
        y0 = int(max(0, np.floor(cy - half)))
        y1 = int(min(img.shape[0], np.ceil(cy + half) + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return [ox, oy], 0.0, False
        sub = img[y0:y1, x0:x1]
        lo, hi = float(sub.min()), float(np.median(sub))
        if hi - lo < 1e-6:
            return [ox, oy], 0.0, False
        w = np.clip((hi - sub) / (hi - lo) - (1.0 - float(floor)), 0, None)
        if w.sum() <= 0:
            return [ox, oy], 0.0, False
        ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
        cx = x0 + float((w * xs).sum() / w.sum())
        cy = y0 + float((w * ys).sum() / w.sum())
    return ([cx, cy], float(np.hypot(cx - ox, cy - oy)), True)


def ring_anchor_for_observation(obs, target_r=14.0):
    """Where a mask task's magenta ring belongs: ON the queried grain.

    rev8 bug this fixes (measured): `focus_xy` does NOT mean the same
    thing across observation families. On obs-v30t-004 it IS the ball
    (detected grain 2.2 px away), so a ring there is right. On
    obs-r4-p03 it is a point on the *path* -- the ring landed 46.2 px
    down the tube, enclosing empty background, and the user reported
    it as off-centre and meaningless. Meanwhile p03's PATH START sits
    12.9 px from the grain centre, i.e. on the grain's edge, because a
    path begins at the tube's exit from the grain.

    Rule: prefer an explicit ball field; else trust `focus_xy` only
    when it is within `target_r` of the path start (then it is the
    ball); else fall back to the path start, which is labelled data
    and lies within a grain radius of the grain centre. Either way the
    ring covers the grain.
    """
    pts = [q for q in (obs.get("path_xy") or [])
           if isinstance(q, (list, tuple)) and len(q) == 2]
    start = ([float(pts[0][0]), float(pts[0][1])] if pts else None)
    for key in ("ball_xy", "grain_xy", "ball", "grain"):
        v = obs.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return [float(v[0]), float(v[1])], "explicit-ball"
    fx = obs.get("focus_xy")
    if isinstance(fx, (list, tuple)) and len(fx) == 2:
        f = [float(fx[0]), float(fx[1])]
        if start is None:
            return f, "focus-no-path"
        d = ((f[0] - start[0]) ** 2 + (f[1] - start[1]) ** 2) ** 0.5
        if d <= float(target_r):
            return f, f"focus-on-ball(d={d:.1f}px)"
    if start is not None:
        return start, "path-start(grain exit)"
    return None, "no-anchor"


def paint_overlap_union(h: int, w: int, origin, *, movie, frame,
                        masks, eligible=None) -> np.ndarray:
    """Pixels painted by TWO OR MORE distinct physical owners (rev10).

    Ambiguous projected overlap must not become a foreign negative: at a
    crossing a pixel can legitimately belong to more than one tube. This
    returns the multi-owner set so the foreign-exclusive selector can
    exclude it while the owners that paint it keep it possible.
    """
    owners_seen: dict[tuple, set] = {}
    for bm in masks or []:
        if str(bm.get("movie", "") or "") != str(movie or ""):
            continue
        if int(bm.get("source_frame", -1)) != int(frame):
            continue
        uuid = str(bm.get("mask_uuid", "") or "")
        if eligible is not None and uuid not in eligible:
            continue
        owner = str(bm.get("owner_key", "") or uuid)
        raster = bm.get("mask_raster")
        if not raster:
            continue
        try:
            m = decode_mask_raster(h, w, origin, raster) > 0
        except Exception:
            continue
        ys, xs = np.nonzero(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            owners_seen.setdefault((y, x), set()).add(owner)
    out = np.zeros((h, w), dtype=bool)
    for (y, x), owners in owners_seen.items():
        if len(owners) >= 2:
            out[y, x] = True
    return out


def confusable_union(h: int, w: int, origin, *, movie, frame, self_uuid,
                     self_owner_key, masks, eligible=None) -> np.ndarray:
    """Union of OTHER owners' painted tubes in this crop (rev9 WP-A.6).

    The rev8 version joined by frame alone and excluded only the same
    mask UUID. The reviewer's rule: the join must match the canonical
    movie identity AND exclude the same physical owner, including other
    annotations or revisions of this very tube — otherwise a
    same-numbered frame in another movie, or a re-annotation of the
    queried tube, becomes a foreign negative.

    `eligible` (optional) restricts the pool to masks with a resolvable
    identity; a quarantined/unlinked mask contributes ZERO, because
    nothing proves it is not this very tube.
    """
    union = np.zeros((h, w), dtype=bool)
    for bm in masks or []:
        if str(bm.get("movie", "") or "") != str(movie or ""):
            continue                      # canonical movie identity
        if int(bm.get("source_frame", -1)) != int(frame):
            continue                      # same frame only
        uuid = str(bm.get("mask_uuid", "") or "")
        if uuid and uuid == str(self_uuid or ""):
            continue                      # this very mask
        owner = str(bm.get("owner_key", "") or "")
        if self_owner_key and owner and owner == str(self_owner_key):
            continue                      # same physical owner/revision
        if eligible is not None and uuid not in eligible:
            continue                      # quarantined: zero contribution
        raster = bm.get("mask_raster")
        if not raster:
            continue
        try:
            union |= decode_mask_raster(h, w, origin, raster)
        except Exception:
            continue
    return union


def add_confusable_validity(body_valid, confusable):
    """Make a foreign labelled tube supervised *negative* material.

    rev8 (H303): another human-painted tube inside the crop is not
    unknown material. The human labelled it as a different owner's
    tube, so for the query at hand it is known **not-my-tube**:
    validity 1 with target 0, which is what lets a negative weight
    apply to it at all. Before this, such pixels sat outside the
    queried mask's reviewed extent, carried validity 0, and every
    weighting multiplied zero -- an A/B on the weight came out
    byte-identical. Unknown stays unknown; *labelled-other* does not.
    """
    if body_valid is None or confusable is None:
        return body_valid
    bv = np.asarray(body_valid)
    cf = np.asarray(confusable)
    if cf.shape != bv.shape:
        raise ValueError("confusable mask shape differs from body_valid")
    return np.maximum(bv, (cf > 0).astype(bv.dtype))


def mask_iou_split(pred_bool, tgt, valid, anchor_xy=None,
                   split_px=40.0) -> dict:
    """IoU over valid pixels, split proximal / distal from the query.

    "Distal" = farther than split_px from the queried grain: the far
    end of the tube, which a root-only prompt cannot reach through a
    local receptive field. `painted_distal_frac` exposes a degenerate
    split (a stub tube scores nothing distally) instead of letting it
    flatter the number.

    split_px="auto" sets the threshold to the MEDIAN distance of the
    painted pixels from the grain (floor 12px), so a short tube still
    gets a real distal half instead of a vacuous one; `split_px_used`
    reports what was applied either way.

    The union is masked on BOTH sides — an unmasked prediction outside
    the reviewed extent must not silently deflate the score.
    """
    m = np.asarray(valid) > 0
    t = np.asarray(tgt) == 1
    p = np.asarray(pred_bool).astype(bool)
    out = {"mask_iou": None, "proximal_iou": None, "distal_iou": None,
           "painted_distal_frac": 0.0, "n_valid": int(m.sum()),
           "split_px_used": None}
    if not m.any():
        return out
    if anchor_xy is not None and len(anchor_xy) == 2:
        ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    else:
        ys, xs = np.nonzero(t)
        if len(xs) == 0:
            return out
        ax, ay = float(xs.mean()), float(ys.mean())
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    dist = np.hypot(xx - ax, yy - ay)
    if isinstance(split_px, str) and split_px == "auto":
        painted = t & m
        if not painted.any():
            return out
        thr = max(12.0, float(np.median(dist[painted])))
    else:
        thr = float(split_px)
    out["split_px_used"] = thr
    far = dist > thr
    for name, sel in (("mask", m), ("proximal", m & ~far),
                      ("distal", m & far)):
        if not sel.any():
            continue
        inter = float((p & t & sel).sum())
        union = float(((p | t) & sel).sum())
        out[f"{name}_iou"] = inter / union if union > 0 else 0.0
    painted = t & m
    out["painted_distal_frac"] = (
        float((painted & far).sum()) / max(1.0, float(painted.sum())))
    return out


def _region_in_crop(h: int, w: int, origin, review_region) -> np.ndarray:
    """Rasterize a native-coord reviewed region into crop coords.

    rev8: 'complete' used to mean the whole translated crop becomes
    background. The reviewed extent is what the painter actually saw;
    outside it material is unknown, never background. Accepts a
    polygon (>=3 native points) or an axis-aligned rect
    (x0, y0, x1, y1). Returns a bool mask (False everywhere if the
    region cannot be interpreted — conservative, never fabricates).
    """
    out = np.zeros((h, w), dtype=bool)
    if not review_region:
        return out
    try:
        pts = np.asarray(review_region, dtype=float)
    except (TypeError, ValueError):
        return out
    if pts.ndim != 2 or pts.shape[0] < 2:
        return out
    ox, oy = float(origin[0]), float(origin[1])
    if pts.shape[0] == 2:  # (x0,y0),(x1,y1) corners
        x0, x1 = sorted((pts[0, 0], pts[1, 0]))
        y0, y1 = sorted((pts[0, 1], pts[1, 1]))
        a, b = int(np.floor(x0 - ox)), int(np.ceil(x1 - ox))
        c, d = int(np.floor(y0 - oy)), int(np.ceil(y1 - oy))
        a, b = max(0, a), min(w, b + 1)
        c, d = max(0, c), min(h, d + 1)
        if a < b and c < d:
            out[c:d, a:b] = True
        return out
    # polygon: even-odd fill via cv2-free scanline on the crop grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xx + ox
    ys = yy + oy
    inside = np.zeros((h, w), dtype=bool)
    n = pts.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cond = ((yi > ys) != (yj > ys))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (ys - yi) / (yj - yi + 0.0) + xi
        inside ^= cond & (xs < xint)
        j = i
    return inside


# extent_hits_paint_xy / review_region_usable live in
# tubetracker/review_region.py (imported above) so the app and the
# pipeline cannot drift apart.


def _review_extent_background(h: int, w: int, origin, paint_bool,
                              review_region
                              ) -> tuple[np.ndarray, bool, str]:
    """Reviewed-background mask licensed by a native review extent.

    Single implementation of the rev9 WP-A.1 quarantine for every
    paint path (exact raster and brush stamps). Returns
    (bg_bool, used, reason): `used` only when the extent is usable AND
    touches the paint at pixel level; otherwise an all-False mask and
    the quarantine reason ('missing' | 'shape' | 'nonfinite' |
    'degenerate' | 'misses-paint' | 'misses-paint-pixels' |
    'no-review-region').
    """
    bg = np.zeros((h, w), dtype=bool)
    if not paint_bool.any():
        return bg, False, "no-paint"
    if not review_region:
        return bg, False, "no-review-region"
    ys, xs = np.nonzero(paint_bool)
    paint_bbox = (float(xs.min()) + float(origin[0]),
                  float(ys.min()) + float(origin[1]),
                  float(xs.max()) + float(origin[0]),
                  float(ys.max()) + float(origin[1]))
    ok, reason = review_region_usable(review_region, paint_bbox)
    if not ok:
        return bg, False, reason
    ext = _region_in_crop(h, w, origin, review_region)
    if not ext.any():
        return bg, False, "empty-in-crop"
    if not (ext & paint_bool).any():
        # an extent for this paint that touches none of it cannot be
        # the human's reviewed region (H306 shape); fail closed
        return bg, False, "misses-paint-pixels"
    return (ext & ~paint_bool), True, "ok"


def body_mask_channels_from_raster(h: int, w: int, origin, raster: dict,
                                   complete: bool = False,
                                   review_region=None, unknown_raster=None,
                                   review_region_provenance=None) -> dict:
    """Explicit per-head channels for a painted body mask.

    rev9 WP-A.1: foreground, reviewed background and unknown are named
    channels rather than one conflated `valid`, and the paint-adjacent
    `band` (the 13.5 px zone the paint itself licenses) is separate from
    `bg_reviewed` (background a HUMAN reviewed extent licenses).
    `valid` = fg | band | bg_reviewed, i.e. everything a consumer may
    supervise. Pixels outside both are `unknown` and contribute zero.

    Quarantine: a review_region that fails `review_region_usable`
    licenses nothing (bg_reviewed stays empty, the paint band remains)
    and the reason is reported, never silently absorbed.
    """
    target = np.zeros((h, w), np.float32)
    fg = np.zeros((h, w), dtype=bool)
    band = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    m = decode_mask_raster(h, w, origin, raster)
    info = {"extent_used": False, "quarantine_reason": "no-paint"}
    if not m.any():
        return {"target": target, "valid": target.copy(), "fg": fg,
                "band": band, "bg_reviewed": bg,
                "unknown": np.ones((h, w), dtype=bool), **info}
    target[m] = 1.0
    fg = m.copy()
    used_extent = False
    reason = "no-review-region"
    region, scope_policy = licensed_review_region(review_region, review_region_provenance)
    if complete:
        bg, used_extent, reason = _review_extent_background(
            h, w, origin, m, region)
        if scope_policy == 'invalid_provenance':
            reason = 'invalid-provenance'
        # quarantined (not used): bg stays all-False, band-only below
    # Near band: the 13.5 px zone around the paint (3x the brush radius).
    # This is a NAMED CHANNEL, not a fallback: it is computed whether or
    # not the reviewed extent is usable. It used to live inside the
    # `not used_extent` branch, which meant that once every extent became
    # usable (snap24: 8/8) the channel was empty everywhere and any
    # consumer of it silently got nothing (measured: band 0 px on all 8
    # masks while bg_reviewed held ~40k).
    inner = np.zeros_like(m)
    inner[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] &
                         m[1:-1, :-2] & m[1:-1, 2:])
    bnd = m & ~inner
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dmin = np.full((h, w), np.inf, np.float32)
    py, px = np.nonzero(bnd)
    for qx, qy in zip(px.tolist(), py.tolist()):
        dmin = np.minimum(dmin, np.hypot(xx - qx, yy - qy))
    band = (dmin <= 13.5) & ~m
    valid = np.zeros((h, w), np.float32)
    valid[fg | band | bg] = 1.0
    return _exclude_mask_unknown({"target": target, "valid": valid, "fg": fg, "band": band,
            "bg_reviewed": bg, "unknown": ~(fg | band | bg),
            "extent_used": bool(used_extent),
            "extent_scope_policy": scope_policy,
            "quarantine_reason": ("ok" if used_extent else reason)}, h, w, origin, unknown_raster)


def body_mask_from_raster(h: int, w: int, origin, raster: dict,
                          complete: bool = False,
                          review_region=None, review_region_provenance=None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Visible-tube-body target + validity from an EXACT paint raster.

    Target 1 exactly where painted (holes stay holes, edges stay
    edges). Validity: paint + near band supervised. Background beyond
    the band is supervised ONLY inside a USABLE reviewed extent
    (review_region); an unusable one (rev9 WP-A.1 quarantine) stays
    UNKNOWN even when the task was marked complete — the human-
    reviewed extent is what licenses a background claim.
    """
    ch = body_mask_channels_from_raster(h, w, origin, raster,
                                        complete=complete,
                                        review_region=review_region,
                                        review_region_provenance=review_region_provenance)
    return ch["target"], ch["valid"]


def front_interval_mask(s_grid, s_star: float,
                        halfwidth_px: float = 2.0) -> np.ndarray:
    """Admissible front set: 1.0 within halfwidth of s*, else 0.0."""
    s = np.asarray(s_grid, dtype=float)
    return (np.abs(s - float(s_star)) <= float(halfwidth_px)).astype(
        np.float32)


def jitter_route(route_xy, tip_xy, degrees: float,
                 rng) -> tuple[np.ndarray, float | None]:
    """Rotate a route about its start by U(-degrees, +degrees).

    Returns (rotated route, s_star) where s_star is the tip's
    projection arclength, or None when the tip is no longer interior
    (caller falls back to the unrotated route). Reproduces rev5's
    required perturbed-route training without leaking the cap: the
    rotation is about the root and never uses the tip except to
    recompute the target projection.
    """
    r = np.asarray(route_xy, dtype=float)
    t = np.asarray(tip_xy, dtype=float)
    ang = float(rng.uniform(-degrees, degrees)) * np.pi / 180.0
    c, sn = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -sn], [sn, c]])
    rj = (r - r[0]) @ rot.T + r[0]
    total = float(np.hypot(np.diff(rj[:, 0]), np.diff(rj[:, 1])).sum())
    s_star = project_to_arclength(rj, t)
    if not (s_star > 2.0 and s_star < total - 8.0):
        return r, None
    return rj, s_star


@dataclass
class SampleDef:
    """One trainable query: joined observation + snapshot context.

    rev8: every sample carries an explicit identity triple
    (movie, owner_key, time) plus its origin uuid, so loss consumption
    can be attributed per unique label and per head.
    """

    entry_id: str
    movie: str
    movie_path: str
    source_frame: int
    tube_ref: str = ""       # tube_uuid or obs_uuid (never fabricated owner)
    kind: str = "tip_only"   # path_tip | tip_only | no_tube | neg_region
    #                          | body_mask | comparison | census_tile
    tip_xy: tuple = ()       # native coords, if a precise tip exists
    region_box: tuple = ()   # native x0,y0,x1,y1 (neg_region / census tile)
    attachment_xy: tuple = ()  # rev6: tube-grain exit (path start)
    proximal_xy: list = field(default_factory=list)  # first widths
    path_visible: list = field(default_factory=list)  # per-vertex
    path_xy: list = field(default_factory=list)
    # The reviewed CURRENT grain-exit-to-cap path, independent of body
    # paint completeness. Never infer this from a generated support tail.
    path_complete: bool = False
    focus_xy: tuple = ()     # task focus (crop anchor when no tip exists)
    direct_state: str = ""
    obs_uuid: str = ""
    obs_revision: int = 1
    task_uuid: str = ""      # rev8: the producing task (mask/duel join key)
    provenance: str = ""     # source project dir (weak-stratum weighting)
    # ---- rev8 identity + extent contract ----
    owner_uuid: str = ""
    owner_key: str = ""      # "<movie>|<owner>" — stable within a movie
    owner_link_source: str = ""   # owner_uuid | task-derived | frame-derived
    sample_key: str = ""     # "<movie>|<owner_key>|<frame>|<kind>"
    # body_mask samples
    mask_uuid: str = ""
    mask_raster: dict | None = None
    mask_unknown_raster: dict | None = None
    mask_points: list = field(default_factory=list)
    brush_px: float = 9.0
    review_region: list = field(default_factory=list)
    review_region_provenance: dict | None = None
    complete: bool = False
    # rev8: the queried grain (native coords) + its radius — the exact
    # object a mask query is about, so a distal readout and a prompt
    # rebuild never have to guess it from the paint.
    target_xy: tuple = ()
    target_r: float = 14.0
    link_source: str = ""    # explicit | task | none (masks, comparisons)
    quarantine_reason: str = ""   # non-empty => NOT consumable as truth
    # rev12 P0.2: truth scope, independent of the sample kind.
    # `certified` = the record is an explicit, grain-scoped human
    # certification (owned-absence regions; masks). Legacy observation
    # records keep certified=False even when kind == "no_tube": their
    # direct_state is preserved as the truth class, never overwritten.
    # `query_kind` = what the query actually is: "grain" | "tube" |
    # "focus" | "box". Owner-specific losses require a resolved owner
    # prompt AND a grain/tube-scoped query ("unassigned" is not an
    # identity).
    certified: bool = False
    query_kind: str = "focus"
    # rev12 P0.2: the durable, movie-qualified physical grain ID (from
    # the append-only grain registry). "" = no physical identity yet.
    grain_id: str = ""
    # comparison samples (route duels)
    lanes_a: list = field(default_factory=list)
    lanes_b: list = field(default_factory=list)
    preference: str = ""     # A | B | neither  (pairwise, NOT validity)
    lanes_validity: str = ""  # "" = unclaimed; else validity judgment
    # census_tile samples
    census_tips: list = field(default_factory=list)
    census_grains: list = field(default_factory=list)
    census_complete: bool = False
    census_class_scopes: list = field(default_factory=list)

    def validate(self) -> None:
        if not self.entry_id or not self.movie or not self.movie_path:
            raise ValueError("SampleDef needs entry/movie/path ids")
        if self.source_frame < 0:
            raise ValueError("SampleDef.source_frame must be >= 0")
        if self.kind not in ("path_tip", "tip_only", "no_tube",
                             "neg_region", "body_mask", "comparison",
                             "census_tile"):
            raise ValueError(f"bad kind {self.kind!r}")
        if self.kind in ("path_tip", "tip_only") and len(self.tip_xy) != 2:
            raise ValueError(f"{self.kind} needs a precise tip")
        if self.kind in ("neg_region", "census_tile") and \
                len(self.region_box) != 4:
            raise ValueError(f"{self.kind} needs a native region_box")
        if self.kind == "path_tip" and len(self.path_xy) < 2:
            raise ValueError("path_tip needs >= 2 path vertices")
        if self.kind == "body_mask" and not self.mask_raster and \
                not self.mask_points:
            raise ValueError("body_mask needs a raster or painted points")
        if self.kind == "comparison":
            if len(self.lanes_a) < 2 or len(self.lanes_b) < 2:
                raise ValueError("comparison needs two lanes")
            if self.preference not in ("A", "B", "neither"):
                raise ValueError(f"bad preference {self.preference!r}")
        if not self.sample_key:
            raise ValueError("SampleDef.sample_key is required (rev8)")


def samples_from_snapshot(snapshot_dir: str) -> list[SampleDef]:
    """Join snapshot observations into trainable query samples.

    path_tip: FULL visible path + precise tip (route + front + apex).
    tip_only: precise tip, no route (apex + visibility only).
    no_tube: reviewed absence (visibility only — never a tip target).
    Dedupes span-confirmation doubles (same tip+frame = one sample).
    """
    import json
    from pathlib import Path
    from tubetracker.review_semantics import precise_tip, is_workflow_record

    snap = Path(snapshot_dir)
    obs = json.loads((snap / "observations.json").read_text())
    obs = [o for o in obs if not is_workflow_record(o)]
    out: list[SampleDef] = []
    seen: set[tuple] = set()
    skipped = {"census_abstention": 0, "tip_conflict": 0,
               "recheck_duplicate": 0}
    # Banked precise tips index (conflict check for absence claims).
    banked_tips: dict[tuple, list] = {}
    for o in obs:
        tip = precise_tip(o)
        if tip and str(o.get("direct_state", "")) in (
                "direct_visible", "visible_imprecise"):
            banked_tips.setdefault(
                (str(o.get("movie", "")), int(o.get("source_frame", -1))),
                []).append((float(tip[0]), float(tip[1])))
    # Anchors first: FULL paths outrank re-click confirmations, so
    # marker-anchored rechecks can never displace real geometry.
    def _anchor_rank(o: dict) -> int:
        return 0 if (o.get("path_complete") and
                     len(o.get("path_xy", []) or []) >= 2) else 1
    accepted_tips: dict[tuple, list] = {}
    for o in sorted(obs, key=_anchor_rank):
        state = str(o.get("direct_state", ""))
        frame = int(o.get("source_frame", -1))
        movie = str(o.get("movie", ""))
        tip = precise_tip(o)
        path = o.get("path_xy", []) or []
        complete = bool(o.get("path_complete", False))
        key = (movie, frame,
               round(float(tip[0]), 1) if tip else None,
               round(float(tip[1]), 1) if tip else None)
        if state in ("direct_visible", "visible_imprecise") and tip:
            if complete and len(path) >= 2:
                kind = "path_tip"
            else:
                kind = "tip_only"
            # Recheck-duplicate exclusion: a tip within 5px of an
            # already-accepted tip at the same (movie, frame) is only
            # excluded when its task SHOWED the known tip (white dot):
            # those clicks land <1px away and carry no independent
            # evidence. Independent agreements (no marker shown) stay.
            # Path anchors sort first, so rechecks can never displace
            # real geometry.
            if tip and o.get("task_showed_tip"):
                dup = (kind == "tip_only" and any(
                    abs(bx - float(tip[0])) < 5.0 and
                    abs(by - float(tip[1])) < 5.0
                    for bx, by in accepted_tips.get((movie, frame), [])))
                if dup:
                    skipped["recheck_duplicate"] += 1
                    continue
            if tip:
                accepted_tips.setdefault((movie, frame), []).append(
                    (float(tip[0]), float(tip[1])))
        elif state in ("no_tube_visible", "not_directly_visible",
                       "out_of_field", "lost_identity", "owner_uncertain",
                       "unresolved_overlap"):
            if str(o.get("task_type", "")) == "census":
                # A census tile's absence click is an abstention ("won't
                # do this tile"), not reviewed absence — these tiles
                # contain known tubes.
                skipped["census_abstention"] += 1
                continue
            # Absence at a banked tip location is a conflict (needs
            # adjudication), not supervision.
            focus = o.get("focus_xy") or [None, None]
            conflict = False
            if focus[0] is not None:
                for bx, by in banked_tips.get((movie, frame), []):
                    if abs(bx - float(focus[0])) < 15.0 and \
                            abs(by - float(focus[1])) < 15.0:
                        conflict = True
                        break
            if conflict:
                skipped["tip_conflict"] += 1
                continue
            kind = "no_tube"
            tip = ()
        else:
            continue  # unrecognized state: never invent supervision
        if kind == "no_tube":
            # Absence judgments dedupe by task: two tasks reviewing the
            # same frame are distinct judgments, not doubles.
            key = (movie, frame, str(o.get("task_uuid", "")))
        if kind != "no_tube" and key in seen:
            continue  # span-confirmation double (pixel-exact re-click)
        seen.add(key)
        tube_ref = str(o.get("tube_uuid") or o.get("obs_uuid", ""))
        entry_id = f"{movie}|{tube_ref}|{frame}"
        # rev6: attachment = path start (tube-grain exit), proximal =
        # first three vertices. Grain CENTER is a different label.
        attach, prox = (), []
        if kind == "path_tip" and len(path) >= 2:
            attach = (float(path[0][0]), float(path[0][1]))
            # rev8: arclength-bounded hint, terminal excluded — a
            # 2-vertex path must not hand over the whole tube.
            prox = arclength_bounded_hint(path)
        owner_uuid = str(o.get("owner_uuid", "") or "")
        if owner_uuid and owner_uuid != "unassigned":
            owner_key = f"{movie}|{owner_uuid}"
            ols = "owner_uuid"
        else:
            owner_key = f"{movie}|task:{o.get('task_uuid', '')}"
            ols = "task-derived"
        s = SampleDef(
            entry_id=entry_id, movie=movie,
            movie_path=str(o.get("movie_path", "")),
            source_frame=frame, tube_ref=tube_ref, kind=kind,
            tip_xy=tuple(float(v) for v in tip) if tip else (),
            attachment_xy=attach, proximal_xy=prox,
            path_xy=[[float(q[0]), float(q[1])] for q in path],
            path_complete=bool(complete and kind == "path_tip"),
            path_visible=[bool(v) for v in (
                o.get("path_visible", [] ) or [])],
            focus_xy=tuple(float(v) for v in (o.get("focus_xy") or ()))
            if o.get("focus_xy") else (),
            direct_state=state, obs_uuid=str(o.get("obs_uuid", "")),
            obs_revision=int(o.get("obs_revision", 1)),
            owner_uuid=owner_uuid, owner_key=owner_key,
            owner_link_source=ols,
            # rev12 P0.2: legacy observation records are NOT certified
            # grain-scoped truth; the direct_state above is the truth
            # class and is never overwritten. The per-record uuid is
            # part of the key (two observations at one owner/frame/kind
            # are distinct records, not collisions).
            certified=False,
            query_kind=("tube" if kind in ("path_tip", "tip_only")
                        else "focus"),
            sample_key=f"{movie}|{owner_key}|{frame}|{kind}|"
                       f"{o.get('obs_uuid', '')}",
            task_uuid=str(o.get("task_uuid", "")),
            provenance=str(o.get("project", "")))
        s.validate()
        out.append(s)
    print(f"samples_from_snapshot: {len(out)} samples, skipped={skipped}")
    # rev6: verified-negative regions are FIRST-CLASS samples with
    # crops centered on their own geometry — never hostage to sharing
    # a frame with a tip-centered sample. Conflict-flagged boxes stay
    # (the loader carves known caps out to unknown); degenerate boxes
    # were already dropped at snapshot build.
    import json as _json
    from pathlib import Path as _Path
    # movie -> path mapping for every later sample type (rev8: shared,
    # defined before all loaders so mask/duel/census joins cannot fail
    # on a missing regions file)
    movies = {s.movie: s.movie_path for s in out if s.movie_path}
    # ...and seeded from the snapshot manifest itself: a mask, duel or
    # census tile can exist with no observation sample in the same
    # snapshot, and its movie identity must still resolve.
    _man = _Path(snapshot_dir) / "snapshot_manifest.json"
    movie_aliases: dict[str, str] = {}
    if _man.exists():
        try:
            _mm = _json.loads(_man.read_text()).get("movies") or {}
            for _k, _v in _mm.items():
                _p = _v.get("path") if isinstance(_v, dict) else None
                if _k and _p and str(_k) not in movies:
                    movies[str(_k)] = str(_p)
            # rev9 WP-A.6: two keys over identical bytes (the live
            # `m1`/`m2` alias) are ONE acquisition. Collapse them so a
            # split can never treat the same frames as independent.
            _by_sha: dict[str, str] = {}
            for _k in sorted(_mm):
                _sha = str((_mm.get(_k) or {}).get("sha256", "") or "")
                if not _sha:
                    continue
                if _sha in _by_sha:
                    movie_aliases[str(_k)] = _by_sha[_sha]
                else:
                    _by_sha[_sha] = str(_k)
            if movie_aliases:
                print("samples_from_snapshot: movie aliases collapsed: "
                      f"{movie_aliases}")
        except (ValueError, AttributeError, TypeError):
            pass
    reg_path = _Path(snapshot_dir) / "regions.json"
    if reg_path.exists():
        reg_rows = _json.loads(reg_path.read_text())
        absence_stats = {"owned_absence_samples": 0}
        for r in reg_rows:
            if r.get("kind") == "owned_absence":
                # rev9 WP-A.6: owned absence trains OWNED
                # presence/visibility for this owner at this grain —
                # never a generic cap negative, never a body target.
                movie = str(r.get("movie_uuid", "") or "")
                frame = int(r.get("source_frame", -1))
                if not movie or frame < 0 or movie not in movies:
                    continue
                grain = r.get("grain_xy") or []
                if len(grain) != 2:
                    continue
                s = SampleDef(
                    entry_id=f"{movie}|{r.get('_region_uuid', 'absence')}"
                    f"|{frame}",
                    movie=movie, movie_path=movies[movie],
                    source_frame=frame,
                    tube_ref=str(r.get("task_uuid", "")),
                    kind="no_tube",
                    direct_state="no_tube_visible",
                    focus_xy=(float(grain[0]), float(grain[1])),
                    target_xy=(float(grain[0]), float(grain[1])),
                    target_r=float(r.get("grain_r", 14.0) or 14.0),
                    obs_uuid=str(r.get("_region_uuid", "")),
                    obs_revision=int(r.get("_region_revision", 1)),
                    task_uuid=str(r.get("task_uuid", "")),
                    owner_uuid=str(r.get("owner_uuid", "") or ""),
                    owner_key=str(r.get("owner_key", "")
                                  or f"{movie}|task:{r.get('task_uuid', '')}"),
                    owner_link_source="snapshot" if r.get("owner_key")
                    else "task-derived",
                    # rev12 P0.2: an owned-absence region is CERTIFIED,
                    # grain-scoped truth (the human marked this exact
                    # grain). The durable physical grain ID rides along;
                    # the per-grain region uuid keeps keys unique.
                    certified=True,
                    query_kind="grain",
                    grain_id=str(r.get("grain_id", "") or ""),
                    sample_key=f"{movie}|{r.get('owner_key') or r.get('task_uuid')}"
                    f"|{frame}|owned_absence|{r.get('_region_uuid', '')}",
                    provenance=str(r.get("_project", "")))
                s.validate()
                out.append(s)
                absence_stats["owned_absence_samples"] += 1
                continue
            if r.get("kind") != "verified_negative":
                continue
            poly = r.get("polygon_xy", []) or []
            try:
                xs = [float(q[0]) for q in poly]
                ys = [float(q[1]) for q in poly]
            except (TypeError, IndexError):
                continue
            box = (min(xs), min(ys), max(xs), max(ys))
            movie = str(r.get("movie_uuid", "") or "")
            frame = int(r.get("source_frame", -1))
            if not movie or frame < 0 or movie not in movies:
                continue
            s = SampleDef(
                entry_id=f"{movie}|{r.get('_region_uuid', 'region')}"
                f"|{frame}",
                movie=movie, movie_path=movies[movie],
                source_frame=frame,
                tube_ref=str(r.get("task_uuid", "")),
                kind="neg_region", region_box=box,
                direct_state="not_directly_visible",
                obs_uuid=str(r.get("_region_uuid", "")),
                obs_revision=int(r.get("_region_revision", 1)),
                task_uuid=str(r.get("task_uuid", "")),
                owner_uuid=str(r.get("owner_uuid", "") or ""),
                owner_key=str(r.get("owner_key", "")
                              or f"{movie}|task:{r.get('task_uuid', '')}"),
                owner_link_source="snapshot" if r.get("owner_key")
                else "task-derived",
                # rev12 P0.2: a verified-negative region is certified,
                # box-scoped truth (no apex inside the reviewed box).
                certified=True,
                query_kind="box",
                sample_key=f"{movie}|{r.get('owner_key') or r.get('task_uuid')}"
                f"|{frame}|neg_region|{r.get('_region_uuid', '')}",
                provenance=str(r.get("_project", "")))
            s.validate()
            out.append(s)
        print(f"samples_from_snapshot: +region samples "
              f"({len(out)} total, owned-absence samples "
              f"{absence_stats['owned_absence_samples']})")

    # ---- rev8: body masks are first-class samples ------------------
    # A mask supervises the BODY head on its own crop (mine, not the
    # path sample's), carrying its exact raster and the extent its
    # painter reviewed. Consumption requires an explicit identity
    # link: source_obs_uuid, or the same task, or the same (movie,
    # owner). Anything else is quarantined and reported — never
    # silently attached to another owner's tube.
    mask_path = _Path(snapshot_dir) / "body_masks.json"
    mask_link_stats = {"explicit": 0, "task": 0, "owner": 0,
                       "owner-self": 0, "quarantined": 0}
    if mask_path.exists():
        mask_rows = _json.loads(mask_path.read_text())
        by_uuid = {s.obs_uuid: s for s in out if s.obs_uuid}
        for m in mask_rows:
            movie = str(m.get("movie", "") or "")
            frame = int(m.get("source_frame", -1))
            if not movie or frame < 0 or movie not in movies:
                mask_link_stats["quarantined"] += 1
                continue
            raster = m.get("mask_raster") or None
            pts = [[float(q[0]), float(q[1])]
                   for q in (m.get("painted_xy") or [])]
            if not raster and not pts:
                mask_link_stats["quarantined"] += 1
                continue
            # link resolution: explicit uuid -> same task -> same owner
            link = str(m.get("source_obs_uuid", "") or "")
            src = by_uuid.get(link) if link else None
            link_source = "explicit" if src is not None else ""
            m_task = str(m.get("task_uuid", "") or "")
            if src is None and m_task:
                cands = [s for s in out if s.task_uuid == m_task
                         and s.obs_uuid]
                if len(cands) == 1:
                    src, link_source = cands[0], "task"
            mkey = str(m.get("owner_key", "") or "")
            if src is None:
                cands = [s for s in out
                         if s.movie == movie and s.source_frame == frame
                         and s.obs_uuid and mkey and s.owner_key == mkey]
                if len(cands) == 1:
                    src, link_source = cands[0], "owner"
            if src is None and str(m.get("owner_uuid", "") or "") \
                    and mkey and not mkey.endswith(f"task:{m_task}"):
                # rev8: the mask carries a NAMED owner (a grain named
                # by its own task, e.g. a paired-crop grain) — an
                # explicit identity, not a proximity guess. The legacy
                # quarantine existed for masks whose owner could only
                # be the "|task:<uuid>" fallback, and those stay
                # quarantined below.
                link_source = "owner-self"
            # rev8: the queried GRAIN is the anchor when the mask
            # carries one. Using the paint centroid put the auto-grain
            # prompt disc 25-30px off the grain on the paired masks
            # (the disc landed on tube body) and framed the crop on the
            # tube's middle, while deployment frames around the grain.
            tgt_xy = [float(q) for q in (m.get("target_xy") or [])]
            geom = np.asarray(pts, float).reshape(-1, 2) if pts else None
            if len(tgt_xy) == 2:
                cx, cy = tgt_xy[0], tgt_xy[1]
            elif geom is not None and geom.shape[0]:
                cx = float(geom[:, 0].mean())
                cy = float(geom[:, 1].mean())
            elif raster:  # raster-only: center on the raster bbox
                x0 = float(raster.get("x0", 0))
                y0 = float(raster.get("y0", 0))
                cx = x0 + float(raster.get("w", 0)) / 2.0
                cy = y0 + float(raster.get("h", 0)) / 2.0
            else:  # neither geometry nor raster: nothing to anchor
                mask_link_stats["quarantined"] += 1
                continue
            owner_key = str(m.get("owner_key", "")
                            or (src.owner_key if src else f"{movie}|task:{m_task}"))
            s = SampleDef(
                entry_id=f"{movie}|{m.get('mask_uuid', 'mask')}|{frame}",
                movie=movie, movie_path=movies[movie], source_frame=frame,
                tube_ref=str(m.get("tube_uuid", "") or
                             (src.tube_ref if src else m_task)),
                kind="body_mask",
                focus_xy=(cx, cy),
                direct_state="direct_visible",
                obs_uuid=str(m.get("mask_uuid", "")),
                obs_revision=int(m.get("mask_revision", 1)),
                task_uuid=m_task,
                owner_uuid=str(m.get("owner_uuid", "")
                               or (src.owner_uuid if src else "")),
                owner_key=owner_key,
                owner_link_source="task-derived"
                if link_source == "owner-self" else (
                    "snapshot" if m.get("owner_key")
                    else (src.owner_link_source if src else "task-derived")),
                sample_key=f"{movie}|{owner_key}|{frame}|body_mask",
                mask_uuid=str(m.get("mask_uuid", "")),
                mask_raster=raster, mask_points=pts,
                mask_unknown_raster=m.get("mask_unknown_raster"),
                brush_px=float(m.get("brush_px", 9.0) or 9.0),
                review_region=m.get("review_region") or [],
                review_region_provenance=m.get("review_region_provenance"),
                complete=bool(m.get("complete", False)),
                link_source=link_source,
                # rev8: the queried grain travels with the mask; the
                # distal readout needs the anchor, and an evaluation
                # must rebuild the deployment prompt from the sample.
                target_xy=tuple(float(q) for q in (
                    m.get("target_xy") or [])[:2]),
                target_r=float(m.get("target_r", 14.0) or 14.0),
                quarantine_reason="" if link_source else "unlinked-mask",
                # rev12 P0.2: a painted mask is certified, tube-scoped
                # human truth.
                certified=True, query_kind="tube",
                provenance=str(m.get("project", "")))
            s.validate()
            mask_link_stats[link_source or "quarantined"] += 1
            out.append(s)
        print(f"samples_from_snapshot: +body_mask samples "
              f"(links={mask_link_stats}, {len(out)} total)")

    # ---- rev8: comparisons (route duels) as independent samples ----
    duel_path = _Path(snapshot_dir) / "duels.json"
    duel_stats = {"consumable": 0, "quarantined": 0, "neither": 0}
    if duel_path.exists():
        for d in _json.loads(duel_path.read_text()):
            movie = str(d.get("movie", "") or "")
            frame = int(d.get("source_frame", -1))
            la = [[float(q[0]), float(q[1])] for q in (d.get("lane_a") or [])]
            lb = [[float(q[0]), float(q[1])] for q in (d.get("lane_b") or [])]
            if not movie or frame < 0 or movie not in movies:
                duel_stats["quarantined"] += 1
                continue
            if len(la) < 2 or len(lb) < 2:
                duel_stats["quarantined"] += 1
                continue
            win = str(d.get("winner", "")).lower()
            pref = "A" if win == "a" else "B" if win == "b" else "neither"
            status = str(d.get("status", "unique"))
            # conflicts are quarantined whole (never two sides of a
            # contradiction reach a loss); an identical repeat carries
            # no new information — the first presentation trains
            if status.startswith("CONFLICT"):
                quar = "duel-conflict-quarantined"
            elif status == "duplicate-agreement":
                quar = "duplicate-agreement"
            else:
                quar = ""
            owner_key = str(d.get("owner_key", "")
                            or f"{movie}|task:{d.get('task_uuid', '')}")
            s = SampleDef(
                entry_id=f"{movie}|{d.get('duel_uuid', 'duel')}|{frame}",
                movie=movie, movie_path=movies[movie], source_frame=frame,
                tube_ref=str(d.get("task_uuid", "")),
                kind="comparison",
                focus_xy=((la[0][0] + lb[0][0]) / 2.0,
                          (la[0][1] + lb[0][1]) / 2.0),
                direct_state="direct_visible",
                obs_uuid=str(d.get("duel_uuid", "")),
                obs_revision=int(d.get("duel_revision", 1)),
                task_uuid=str(d.get("task_uuid", "")),
                owner_uuid=str(d.get("owner_uuid", "") or ""),
                owner_key=owner_key,
                owner_link_source=str(d.get("owner_link_source",
                                           "task-derived")),
                certified=True, query_kind="tube",
                sample_key=f"{movie}|{owner_key}|{frame}|comparison",
                lanes_a=la, lanes_b=lb, preference=pref,
                lanes_validity="",  # preference is NOT a validity claim
                quarantine_reason=quar,
                provenance=str(d.get("project", "")))
            s.validate()
            if quar:
                duel_stats["quarantined"] += 1
            else:
                duel_stats["consumable"] += 1
                duel_stats["neither"] += int(pref == "neither")
            out.append(s)
        print(f"samples_from_snapshot: +comparison samples "
              f"({duel_stats}, {len(out)} total)")

    # ---- rev8: census tiles as first-class samples -----------------
    census_path = _Path(snapshot_dir) / "census.json"
    if census_path.exists():
        from tubetracker.review_semantics import census_grain_points, is_withdrawn, is_workflow_record
        n_cen = 0
        for c in _json.loads(census_path.read_text()):
            if is_withdrawn(c) or is_workflow_record(c):
                continue
            tips = c.get('tips') or []
            scopes = list(c.get('class_scopes') or [])
            cap_complete = bool(c.get('complete') and not c.get('tile_derived', False)
                                and CENSUS_CAP_CLASSES.intersection(scopes))
            # This loader feeds cap/front/body heads. Grain-only records
            # remain in the census for inventory, without any cap loss.
            if not tips and not cap_complete:
                continue
            movie = str(c.get("movie", "") or "")
            tile = c.get("tile_xywh") or []
            if not movie or movie not in movies or len(tile) != 4:
                continue
            frame = int(c.get("source_frame", -1))
            if frame < 0:
                continue
            x0, y0, w, h = (float(tile[0]), float(tile[1]),
                            float(tile[2]), float(tile[3]))
            owner_key = str(c.get("owner_key", "")
                            or f"{movie}|task:{c.get('task_uuid', '')}")
            s = SampleDef(
                entry_id=f"{movie}|{c.get('task_uuid', 'tile')}|{frame}",
                movie=movie, movie_path=movies[movie], source_frame=frame,
                tube_ref=str(c.get("task_uuid", "")),
                kind="census_tile",
                region_box=(x0, y0, x0 + w - 1, y0 + h - 1),
                focus_xy=(x0 + w / 2.0, y0 + h / 2.0),
                direct_state="direct_visible",
                obs_uuid=str(c.get("task_uuid", "")),
                task_uuid=str(c.get("task_uuid", "")),
                owner_uuid=str(c.get("owner_uuid", "") or ""),
                owner_key=owner_key,
                certified=cap_complete,
                query_kind="box",
                sample_key=f"{movie}|{owner_key}|{frame}|census_tile",
                census_tips=[[float(q[0]), float(q[1])]
                             for q in tips],
                census_grains=census_grain_points(c),
                census_complete=cap_complete,
                census_class_scopes=scopes,
                provenance=str(c.get("project", "")),
                complete=cap_complete)
            s.validate()
            out.append(s)
            n_cen += 1
        print(f"samples_from_snapshot: +census_tile samples "
              f"({n_cen}, {len(out)} total)")
    # rev9 WP-A.6: apply the movie-alias collapse to EVERY sample,
    # whatever path built it (identical bytes are one acquisition).
    if movie_aliases:
        for _s in out:
            _alias = movie_aliases.get(_s.movie)
            if _alias and _alias in movies:
                _old_movie = _s.movie
                _s.movie = _alias
                _s.movie_path = movies[_alias]
                # rev10: the entry_id embeds the movie prefix and the
                # qualification guard compares it against movie_id, so
                # an un-rewritten prefix fails the split check with a
                # message about a movie nobody referenced any more.
                if str(_s.entry_id).startswith(_old_movie + "|"):
                    _s.entry_id = _alias + str(_s.entry_id)[len(_old_movie):]

    return out
