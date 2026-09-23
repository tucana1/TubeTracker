"""Grain-rooted curved routes from current body evidence to a shared cap pool.

Graph paths can bend around a grain and preserve alternate continuations.
They are proposals, not certificates of complete observed tube geometry.
"""
from __future__ import annotations

import hashlib
import copy

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from .route_quality import RoutePolicy, inspect_route, route_certificate


#: Versioned contract for model-estimated grain exits (length work, rev16).
#: The exit is the grain-circle crossing with maximal just-outside tube
#: support; it extends (never shortens) the traced rim-rooted path.
EXIT_ESTIMATE_CONTRACT = 'grain_circle_exit.v1'


def estimate_grain_exit(body_probability, origin, grain_native, radius, root_xy,
                        half_window_deg=60.0):
    """Estimate where the tube leaves the grain, model-only.

    Scans the grain circle for the crossing whose just-outside body support
    stays bright at every sampled radius (a tube is a long bright run; bare
    rim halo fades within a few px). Returns (exit_xy, support) or
    (None, 0.0) when the tile carries no usable evidence. Pure geometry on
    the current frame's owner-conditioned map; consumes no human geometry.
    """
    import math
    prob = np.asarray(body_probability, dtype=float)
    if prob.ndim != 2 or not np.isfinite(prob).all():
        return None, 0.0
    h, w = prob.shape
    origin = np.asarray(origin, dtype=float)
    centre = np.asarray(grain_native, dtype=float)
    root = np.asarray(root_xy, dtype=float)
    radius = float(radius)
    root_ang = math.degrees(math.atan2(root[1] - centre[1], root[0] - centre[0]))
    best, best_s = None, -1.0
    deg = 0.0
    while deg < 360.0:
        d = abs((deg - root_ang + 180.0) % 360.0 - 180.0)
        if d <= half_window_deg:
            a = math.radians(deg)
            ux, uy = math.cos(a), math.sin(a)
            vals = []
            for dr in (1.0, 3.0):
                x = centre[0] + ux * (radius + dr) - origin[0]
                y = centre[1] + uy * (radius + dr) - origin[1]
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    vals.append(prob[yi, xi])
            if vals:
                support = float(min(vals))
                if support > best_s + 1e-9:
                    best_s = support
                    best = [centre[0] + ux * radius, centre[1] + uy * radius]
        deg += 1.0
    if best is None:
        return None, 0.0
    return best, best_s


def _trace(cost, start, end):
    graph = MCP_Geometric(cost, fully_connected=True)
    total, _ = graph.find_costs([start], [end])
    if not np.isfinite(total[end]):
        return None
    return np.asarray(graph.traceback(end), dtype=float)[:, ::-1]


def generate_owned_routes(body_probability, origin, owner, caps, *,
                            image_size=None, max_caps=12, alternate_routes=2,
                            other_owners=(), policy=None, validation=None, movie='', source_frame=None,
                            body_evidence=None):
    policy = policy or RoutePolicy()
    frame = int(source_frame if source_frame is not None else
                (caps[0].get('source_frame', 0) if caps else 0))
    declared = {k: copy.deepcopy(owner.get(k)) for k in
                ('attachment_native', 'attachment_verified', 'attachment_source',
                 'identified_at_frame', 'source_task')}
    root_source = owner.get('attachment_source') or {}
    frame_review = (owner.get('attachment_verified')
                    and root_source.get('scope_frame') == frame)
    if policy.root_selection == 'current_body_rim' and not frame_review:
        # A model root proposal never edits the declared human attachment.
        # Exact frame-review roots remain authoritative in assisted modes.
        owner = dict(owner, attachment_verified=False)
        owner.pop('attachment_native', None)
        owner.pop('attachment_source', None)
    prob = np.asarray(body_probability, dtype=float)
    if prob.ndim != 2 or not np.isfinite(prob).all():
        raise ValueError("body evidence must be a finite native probability map")
    h, w = prob.shape
    origin = np.asarray(origin, dtype=float)
    centre = np.asarray(owner["grain_native"], dtype=float)
    radius = float(owner.get("grain_radius_px", 13.0))
    root = owner.get("attachment_native")
    root_verified = bool(root is not None and owner.get("attachment_verified"))
    y, x = np.mgrid[:h, :w]
    distance_to_grain = np.hypot(x + origin[0] - centre[0], y + origin[1] - centre[1])
    cost = 0.05 - np.log(np.clip(prob, .01, .995))
    # A wrap must go around the physical grain, not shortcut its interior.
    cost[distance_to_grain < .72 * radius] = np.inf
    for grain in other_owners:
        if grain['id'] != owner['id']:
            gx, gy = grain['grain_native']
            cost[np.hypot(x+origin[0]-gx, y+origin[1]-gy) < .72*float(grain.get('grain_radius_px', 13))] = np.inf
    if image_size:
        iw, ih = image_size
        cost[(x + origin[0] < 0) | (x + origin[0] >= iw) |
             (y + origin[1] < 0) | (y + origin[1] >= ih)] = np.inf
    rim_root_provisional = root is None
    if root is not None:
        root = np.asarray(root, dtype=float)
        start = np.rint(root - origin).astype(int)
    else:
        rim = (distance_to_grain >= .85 * radius) & (distance_to_grain <= 1.15 * radius)
        if not rim.any():
            return []
        ys, xs = np.where(rim)
        k = int(np.argmax(prob[ys, xs]))
        start = np.array([xs[k], ys[k]])
        root = start + origin
    if not (0 <= start[0] < w and 0 <= start[1] < h):
        return []
    syx = int(start[1]), int(start[0])
    cost[syx] = max(.05, float(-np.log(np.clip(prob[syx], .01, .995))))
    possible = []
    for cap in caps:
        target = np.asarray(cap["tip_xy"])
        p = np.rint(target - origin).astype(int)
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        # No tip-to-root distance cutoff: short nascent tubes live within a
        # few px of the rim root, and that is the emergence regime. Degenerate
        # zero-length traces are rejected after the solve (len(native) < 2).
        endpoint_body = float(prob[max(0, p[1] - 3):min(h, p[1] + 4),
                                   max(0, p[0] - 3):min(w, p[0] + 4)].max())
        possible.append((float(cap["probability"]) * endpoint_body, cap, p))
    possible.sort(key=lambda item: (-item[0], item[1]["cap_id"]))
    possible = possible[:max_caps]
    if not possible:
        return []
    # One shared shortest-path solve supplies the primary route to all caps.
    graph = MCP_Geometric(cost, fully_connected=True)
    ends = [(int(p[1]), int(p[0])) for _, _, p in possible]
    total, _ = graph.find_costs([syx], ends)
    rows = []
    for rank, (_, cap, p) in enumerate(possible):
        end = (int(p[1]), int(p[0]))
        if not np.isfinite(total[end]):
            continue
        paths = [np.asarray(graph.traceback(end), dtype=float)[:, ::-1]]
        # One alternate for the best few destinations: penalize the middle
        # of the first route, keeping its endpoints intact.
        if rank < 3 and alternate_routes > 1 and len(paths[0]) > 30:
            middle = paths[0][10:-10].astype(int)
            penalized = cost.copy()
            corridor = np.zeros_like(prob, dtype=np.uint8)
            corridor[middle[:, 1], middle[:, 0]] = 1
            corridor = cv2.dilate(corridor, np.ones((5, 5), np.uint8))
            penalized[corridor > 0] += 2.0
            alt = _trace(penalized, syx, end)
            if alt is not None:
                paths.append(alt)
        for k, points in enumerate(paths):
            ip = points.astype(int)
            support = prob[ip[:, 1], ip[:, 0]]
            outside = distance_to_grain[ip[:, 1], ip[:, 0]] > radius
            scored = support[outside] if outside.any() else support
            mean = float(scored.mean())
            coverage = float((scored >= .5).mean())
            score = float(np.clip(mean * (0.5 + 0.5 * coverage), .001, .999))
            simplified = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), .8, False).reshape(-1, 2)
            native = simplified + origin
            native[0], native[-1] = root, np.asarray(cap["tip_xy"])
            if len(native) < 2:
                continue
            # Length work (rev16): model-estimated grain exit extends a
            # provisional rim-rooted trace back to the grain boundary
            # before any diagnostic sees the path, so scores, lengths and
            # overlays agree. Stored/declared attachments (verified or
            # preserved-conflicting) are authoritative and never extended.
            # Prepends only; the accepted tip is untouched.
            exit_xy, exit_support = (None, 0.0)
            if rim_root_provisional:
                exit_xy, exit_support = estimate_grain_exit(
                    prob, origin, centre, radius, native[0])
                if exit_xy is not None:
                    native = np.vstack([np.asarray(exit_xy), native])
            diagnostic = inspect_route(prob, origin, native, owner, cap, image_size=image_size,
                                       other_owners=other_owners, policy=policy)
            # A bright cap is not an owned tip when the route assigning it
            # to this grain crosses an unsupported gap or another grain.
            # Root certainty/full-length certification are separate: a
            # connected tube may still have a useful tip without either.
            tip_failures = [f for f in diagnostic['failures'] if f in (
                'weak_current_body_evidence', 'insufficient_supported_extent',
                'unsupported_gap', 'enters_grain_interior', 'grain_pose_unresolved')]
            root_supported = diagnostic['root_basis'] in ('stored_attachment', 'evidence_rim_start')
            certificate, reasons = route_certificate(diagnostic, owner, cap, validation,
                                                     movie=movie, frame=frame)
            digest = hashlib.sha256(native.tobytes()).hexdigest()[:10]
            # A contiguous body-supported root prefix remains available if
            # the apex is later marked hidden. No tail is invented.
            low = np.flatnonzero((support < .5) & (np.arange(len(support)) > 5))
            stop = int(low[0]) if len(low) else len(points)
            partial = (points[:stop] + origin).tolist() if stop >= 2 else []
            # The exit estimate extends the partial trace too: it is the
            # path's observed start, not an invented tail.
            if exit_xy is not None:
                partial = [exit_xy] + partial if partial else [exit_xy]
            rows.append({"candidate_id": f"{owner['id']}:{cap['cap_id']}:{digest}",
                         "cap_id": cap["cap_id"], "tip_xy": cap["tip_xy"],
                         "cap_probability": float(cap["probability"]),
                         "route_probability": score, "current_path_xy": native.tolist(),
                         "observed_partial_path_xy": partial,
                         "exit_xy": exit_xy,
                         "exit_support": exit_support,
                         "exit_contract": EXIT_ESTIMATE_CONTRACT if exit_xy is not None else None,
                         "path_complete": certificate is not None, "root_verified": root_verified,
                         "root_supported": root_supported,
                         "root_provisional": root is None,
                         "length_basis": ('verified_exit' if root_verified
                                          else 'rim_start_provisional'),
                         "length_note": ("route length measured from an arbitrary "
                                         "rim start, not a reviewed exit; tip accuracy "
                                         "does not establish tube length" if root is None
                                         else None),
                         "tip_ownership_supported": not tip_failures,
                         "path_certificate": certificate,
                         "route_evidence": {"body_mean": mean, "body_fraction_ge_05": coverage,
                                            "source": "current-frame owner-conditioned body map",
                                            "alternative": k, "probability_calibrated": False,
                                             "tip_ownership": {"supported": not tip_failures,
                                                 "failures": tip_failures, "basis": "connected current-frame body evidence"},
                                             "root_evidence": {
                                                 "selection": policy.root_selection,
                                                 "basis": diagnostic['root_basis'],
                                                 "declared": declared,
                                                 "current_root_xy": native[0].tolist(),
                                                 "declared_discrepancy_px": (
                                                     float(np.linalg.norm(native[0]-declared['attachment_native']))
                                                     if declared['attachment_native'] is not None else None),
                                                 "frame_review_root": bool(frame_review)},
                                             **({'body_provider': copy.deepcopy(body_evidence)}
                                                if body_evidence else {}),
                                             **({'grain_pose': copy.deepcopy(owner['grain_pose'])}
                                                if owner.get('grain_pose') else {}),
                                             'geometry': diagnostic, 'withheld_reasons': reasons},
                         "uncertainty_px": cap.get("uncertainty_px")})
    return rows
