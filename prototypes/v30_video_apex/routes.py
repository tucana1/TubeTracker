"""v30 automatic route proposals (rev5 #4): competing roots/routes from pixels.

Proposals are generated WITHOUT the gold tip: FRST grains supply roots,
frozen v1 tip heat supplies candidate apices, and each root->peak pair
expands to straight + curved variants plus blind fan fallbacks. Nothing
here sees the human path; oracle routes enter only at scoring time via
evaluate.substitute_oracle_routes.

Headless: numpy + cv2 + torch (CPU) only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_v1(weights: str | Path):
    """Frozen v1 tip model (never trained here, never modified)."""
    import torch

    from tubetracker.cnn_prototype import PointHeatmapNet

    device = torch.device("cpu")
    payload = torch.load(str(weights), map_location=device,
                         weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) \
        and "state_dict" in payload else payload
    model = PointHeatmapNet(input_channels=2, output_channels=2,
                            base_channels=16)
    model.load_state_dict(state)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    return model, device


def v1_tip_heat(model, device, gray: np.ndarray) -> np.ndarray:
    """Sigmoid tip probabilities (native HxW) for a full grayscale frame."""
    import torch

    from tubetracker.cnn_prototype import predict_heatmaps_tiled

    with torch.no_grad():
        out = predict_heatmaps_tiled(model, gray, device)
    return np.asarray(out[1]).astype(np.float32)


def heat_peaks(heat: np.ndarray, thresh: float = 0.35,
               footprint: int = 9) -> list[tuple[float, float, float]]:
    """Isolated local maxima (x, y, heat); spiky-map-safe (H239)."""
    import cv2

    if float(heat.max()) < thresh:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (footprint, footprint))
    dil = cv2.dilate(heat, k)
    mask = (heat >= dil) & (heat >= thresh)
    n, _, _, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return [(float(centroids[i][0]), float(centroids[i][1]),
             float(heat[int(centroids[i][1]), int(centroids[i][0])]))
            for i in range(1, n)]


def auto_root(gray: np.ndarray, anchor_xy,
              max_dist_px: float = 40.0):
    """Nearest FRST grain to the anchor as the automatic root.

    Returns (root_xy, source) where source is 'frst' or 'assisted'.
    The assisted fallback is the anchor itself, honestly labeled —
    a wrong root must not masquerade as an automatic one.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from prototypes.timesfm_tip_forecast.grain_detect import detect_grains

    grains = detect_grains(gray)
    best = None
    for x, y, s in grains:
        d = math.hypot(x - anchor_xy[0], y - anchor_xy[1])
        if best is None or d < best[0]:
            best = (d, (float(x), float(y)))
    if best is not None and best[0] <= max_dist_px:
        return best[1], "frst"
    return (float(anchor_xy[0]), float(anchor_xy[1])), "assisted"


def _curve(root, tip, bend_px: float) -> list:
    root = np.asarray(root, float)
    tip = np.asarray(tip, float)
    mid = (root + tip) / 2
    d = tip - root
    n = math.hypot(*d)
    if n < 1e-9:
        return [root.tolist(), tip.tolist()]
    perp = np.array([-d[1], d[0]]) / n
    mid = mid + perp * bend_px
    return [root.tolist(), mid.tolist(), tip.tolist()]


def _wall_support(gray: np.ndarray, p: np.ndarray, tangent: np.ndarray,
                  half: float = 14.0, n_off: int = 29) -> float:
    """Max transverse gradient across the tube at p (H146 core).

    Centers between the two strongest walls (H149: argmax alone rides
    edges off-tube); returns the support value. Pure numpy remap.
    """
    import cv2

    g = np.asarray(gray, dtype=float)
    H, W = g.shape
    t = np.asarray(tangent, float)
    t = t / (float(np.hypot(*t)) + 1e-9)
    n = np.array([-t[1], t[0]])
    offs = np.linspace(-half, half, n_off)
    xs = np.clip(p[0] + n[0] * offs, 0, W - 1).astype(np.float32)[None, :]
    ys = np.clip(p[1] + n[1] * offs, 0, H - 1).astype(np.float32)[None, :]
    pr = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)[0]
    return float(np.abs(np.diff(pr)).max())


def grow_wall_routes(gray: np.ndarray, root_xy, beam: int = 4,
                     step_px: float = 6.0, max_steps: int = 25,
                     cone_deg: float = 60.0, n_dirs: int = 13,
                     min_support_frac: float = 0.4,
                     stop_patience: int = 3) -> list[dict]:
    """Beam-search routes that FOLLOW tube walls from the root (rev6).

    Each beam steps 6px along headings in a +/-cone; the step score is
    wall support + forward persistence. Beams die when support stays
    below 0.4x their own early reference for 3 steps (tube ended).
    Returns polylines with natural lengths + support traces. No peaks,
    no fans, no gold — image evidence only.
    """
    g = np.asarray(gray, dtype=float)
    H, W = g.shape
    root = np.asarray(root_xy, float)
    # Initial headings: full compass (no prior direction at the root).
    inits = [np.array([math.cos(a), math.sin(a)])
             for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    states = [ {"p": root.copy(), "h": d, "poly": [root.copy()],
                 "score": 0.0, "trace": [], "weak": 0, "ref": None}
               for d in inits ]
    finished = []
    cone = math.radians(cone_deg)
    for _ in range(max_steps):
        nxt = []
        for st in states:
            base = math.atan2(st["h"][1], st["h"][0])
            cands = []
            for k in range(n_dirs):
                ang = base + (2 * k / max(n_dirs - 1, 1) - 1) * cone
                d = np.array([math.cos(ang), math.sin(ang)])
                q = st["p"] + d * step_px
                if not (0 <= q[0] < W and 0 <= q[1] < H):
                    continue
                sup = _wall_support(g, q, d)
                persist = float(np.dot(d, st["h"]))
                cands.append((sup + 0.3 * persist, q, d, sup))
            if not cands:
                finished.append(st)
                continue
            cands.sort(key=lambda z: -z[0])
            moved = False
            for score, q, d, sup in cands[:2]:
                ref = st["ref"]
                trace = st["trace"] + [sup]
                if ref is None and len(trace) >= 5:
                    ref = float(np.median(trace[:5]))
                weak = st["weak"]
                if ref is not None and sup < min_support_frac * ref:
                    weak += 1
                else:
                    weak = 0
                poly = st["poly"] + [q.copy()]
                if weak >= stop_patience:
                    finished.append({**st, "poly": poly, "trace": trace,
                                     "score": st["score"] + score})
                else:
                    nxt.append({"p": q, "h": d, "poly": poly,
                                "score": st["score"] + score,
                                "trace": trace, "weak": weak, "ref": ref})
                    moved = True
            if not moved and not cands[:2]:
                finished.append(st)
        if not nxt:
            finished.extend(states)
            break
        nxt.sort(key=lambda z: -z["score"])
        # Diversity: suppress near-duplicate endpoints.
        kept = []
        for st in nxt:
            if all(math.hypot(*(st["p"] - k["p"])) > 12.0 for k in kept):
                kept.append(st)
            if len(kept) >= beam:
                break
        states = kept or nxt[:beam]
    finished.extend(states)
    out = []
    for i, st in enumerate(sorted(finished, key=lambda z: -z["score"])):
        poly = np.array([p.tolist() for p in st["poly"]])
        if len(poly) < 3:
            continue
        tr = st["trace"] or [0.0]
        out.append({"route_id": f"wall{i}",
                    "polyline": poly.tolist(),
                    "seed": "wall-follow", "seed_heat": 0.0,
                    "mean_support": float(np.mean(tr)),
                    "support_len": len(tr)})
    # Dedupe near-identical endpoints, keep best scores.
    ded = []
    for r in out:
        end = np.asarray(r["polyline"][-1])
        if all(math.hypot(*(end - np.asarray(d["polyline"][-1]))) > 10.0
               for d in ded):
            ded.append(r)
    return ded


def auto_grain(gray: np.ndarray, anchor_xy,
               max_dist_px: float = 40.0):
    """Nearest FRST grain center + size to the anchor.

    Returns (center_xy, radius_px, source). Radius is the FRST size
    estimate (P58 ~13.7px). Falls back to the anchor with radius 13
    when nothing is near, honestly labeled.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from prototypes.timesfm_tip_forecast.grain_detect import detect_grains

    grains = detect_grains(gray)
    best = None
    for x, y, s in grains:
        d = math.hypot(x - anchor_xy[0], y - anchor_xy[1])
        if best is None or d < best[0]:
            best = (d, (float(x), float(y)), float(s))
    if best is not None and best[0] <= max_dist_px:
        return best[1], max(best[2], 6.0), "frst"
    return (float(anchor_xy[0]), float(anchor_xy[1])), 13.0, "assisted"


def propose_from_attachments(center_xy, radius_px, peaks=None,
                             n_attach: int = 8,
                             dirs_per_attach: int = 3,
                             lengths=(60.0, 120.0),
                             radii: tuple = ()) -> list[dict]:
    """Fans from candidate grain-boundary attachments, not the center.

    A 12px root error collapses fan coverage 1.00->0.00 (measured);
    proposing from 8 boundary points recovers 0.50-1.00. Each
    attachment fans outward (radial) +/-30 deg. radii hedges the
    FRST size (measured floors at 6px vs true ~13): circles at
    several radii since the exit ring is uncertain. Records which
    attachment seeded each route for the audit trail.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    rings = radii or (float(radius_px),)
    proposals: list[dict] = []
    for R in rings:
        for k in range(n_attach):
            ang = k * 2 * math.pi / n_attach
            ax, ay = cx + R * math.cos(ang), cy + R * math.sin(ang)
            for j, off in enumerate(
                    [0.0, math.pi / 6, -math.pi / 6][:dirs_per_attach]):
                a2 = ang + off
                for ln in lengths:
                    tip = [ax + ln * math.cos(a2), ay + ln * math.sin(a2)]
                    proposals.append({
                        "route_id": f"a{k}d{j}l{int(ln)}r{int(R)}",
                        "polyline": [[ax, ay], tip],
                        "seed": f"attach-{k}-r{int(R)}",
                        "seed_heat": 0.0,
                        "attachment": [ax, ay]})
    return proposals


def propose_curved_walkers(gray: np.ndarray, center_xy, radius_px: float,
                           n_attach: int = 8,
                           dirs_per_attach: int = 3,
                           lengths=(60.0, 120.0),
                           radii: tuple = (),
                           step_px: float = 4.0,
                           max_steps: int = 30,
                           support_floor: float = 2.5) -> list[dict]:
    """Orientation-following curved routes from grain attachments.

    Straight fans cut across tube bends (measured: best 0.50 coverage
    on curves). Each walker starts at a boundary attachment point and
    steps along the heading with the strongest transverse wall energy
    (the validated geometric sensor from tip correction), with heading
    momentum (+/-25 deg probe fan). Stops when support falls below the
    floor or the walk leaves the frame. Emits prefixes at `lengths`.
    Starts OUTSIDE the grain (boundary points) — the failed wall-beam
    died inside the grain mass.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from prototypes.timesfm_tip_forecast.switch_cut import (
        transverse_wall_energy)
    g = np.asarray(gray, dtype=float)
    H, W = g.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    rings = radii or (float(radius_px),)
    proposals: list[dict] = []
    for R in rings:
        for k in range(n_attach):
            ang = k * 2 * math.pi / n_attach
            ax, ay = cx + R * math.cos(ang), cy + R * math.sin(ang)
            for j, off in enumerate(
                    [0.0, math.pi / 6, -math.pi / 6][:dirs_per_attach]):
                heading = ang + off
                pts = [[ax, ay]]
                x, y = ax, ay
                # Seed step so the tail has a tangent.
                x += step_px * math.cos(heading)
                y += step_px * math.sin(heading)
                pts.append([x, y])
                stopped = False
                for _ in range(max_steps - 1):
                    best_e, best_h = -1.0, heading
                    for turn in (-0.44, -0.21, 0.0, 0.21, 0.44):
                        h = heading + turn
                        nx = x + step_px * math.cos(h)
                        ny = y + step_px * math.sin(h)
                        if not (0 <= nx < W and 0 <= ny < H):
                            continue
                        tail = np.asarray(pts[-2:] + [[nx, ny]], float)
                        try:
                            e = float(transverse_wall_energy(g, tail)[-1])
                        except Exception:
                            continue
                        if e > best_e:
                            best_e, best_h = e, h
                    if best_e < support_floor:
                        stopped = True
                        break
                    heading = best_h
                    x += step_px * math.cos(heading)
                    y += step_px * math.sin(heading)
                    if not (0 <= x < W and 0 <= y < H):
                        stopped = True
                        break
                    pts.append([x, y])
                import numpy as _np
                arclen = float(_np.abs(_np.diff(_np.asarray(pts, float),
                                               axis=0)).sum())
                for ln in lengths:
                    if arclen + 1e-9 < float(ln):
                        continue
                    # Prefix closest to ln by arclength.
                    run, cut = 0.0, len(pts)
                    for i in range(1, len(pts)):
                        run += math.hypot(pts[i][0] - pts[i - 1][0],
                                          pts[i][1] - pts[i - 1][1])
                        if run >= float(ln):
                            cut = i + 1
                            break
                    proposals.append({
                        "route_id": f"w{k}d{j}l{int(ln)}r{int(R)}",
                        "polyline": [list(q) for q in pts[:cut]],
                        "seed": f"walk-{k}-r{int(R)}",
                        "seed_heat": 0.0,
                        "attachment": [ax, ay],
                        "stopped_early": stopped})
    return proposals


def propose_routes(root_xy, peaks, max_routes: int = 25,
                   search_radius: float = 260.0) -> list[dict]:
    """Competing root->tip route proposals (no gold anywhere).

    Each of the top heat peaks near the root seeds straight/curved
    variants; blind fan fallbacks cover a v1 proposal miss. Every
    proposal records its seed for the audit trail.
    """
    near = [(x, y, h) for x, y, h in peaks
            if math.hypot(x - root_xy[0], y - root_xy[1]) <= search_radius]
    near.sort(key=lambda p: -p[2])
    proposals: list[dict] = []
    for i, (x, y, h) in enumerate(near[:3]):
        proposals.append({"route_id": f"p{i}s",
                          "polyline": [list(root_xy), [x, y]],
                          "seed": f"v1-peak-{i}", "seed_heat": h})
        proposals.append({"route_id": f"p{i}c+",
                          "polyline": _curve(root_xy, (x, y), 12.0),
                          "seed": f"v1-peak-{i}", "seed_heat": h})
        proposals.append({"route_id": f"p{i}c-",
                          "polyline": _curve(root_xy, (x, y), -12.0),
                          "seed": f"v1-peak-{i}", "seed_heat": h})
    # Blind fan fallbacks (proposal-miss coverage, scored, never hidden):
    # 8 directions x 2 lengths so a missed v1 peak still leaves a
    # near-true support among the candidates. Ranking (not count) decides.
    import math as _m

    _angs = [j * _m.pi / 4 for j in range(8)]
    for j, ang in enumerate(_angs):
        for k, ln in enumerate((60.0, 120.0)):
            tip = [root_xy[0] + ln * _m.cos(ang),
                   root_xy[1] + ln * _m.sin(ang)]
            proposals.append({"route_id": f"fan{j}l{k}",
                              "polyline": [list(root_xy), tip],
                              "seed": "blind-fan", "seed_heat": 0.0})
    return proposals[:max_routes]


def propose_body_walks(prob, root_xy, min_prob: float = 0.5,
                       min_len_px: float = 30.0, max_routes: int = 8,
                       sep_px: float = 20.0, smooth: int = 3) -> list[dict]:
    """Root-connected routes from owned-body probabilities (rev11 item 6).

    The review: 'Generate root-connected routes from owned-body
    probabilities, retaining crossing alternatives.' The proposals
    paths FOLLOW the owner's own body evidence rather than geometric fans:
    threshold the probability map, keep the connected component
    containing the grain root, skeletonize it, and trace root->endpoint
    paths along the skeleton. Crossing alternatives are retained:
    branches whose paths deviate by fewer than `sep_px` pixels anywhere
    are clustered, and each surviving branch becomes its own route — a
    junction yields alternatives rather than one arbitrary choice.
    (Root-angle separation was tried first and FAILED trunk-then-branch
    geometry: both branches share the trunk, so their first-20px
    directions are identical. Max path deviation is the robust test.)
    Routes carry their natural length (no prefix
    variants); the caller labels and scores them like any candidate.
    """
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage.measure import label as _label

    p = np.asarray(prob, float)
    H, W = p.shape[:2]
    mask = p >= float(min_prob)
    if not mask.any():
        return []
    lab = _label(mask, connectivity=2)
    ry = min(max(int(round(float(root_xy[1]))), 0), H - 1)
    rx = min(max(int(round(float(root_xy[0]))), 0), W - 1)
    root_lab = lab[ry, rx]
    if root_lab == 0:
        ys, xs = np.nonzero(mask)
        d = (ys - ry) ** 2 + (xs - rx) ** 2
        j = int(np.argmin(d))
        ry, rx = int(ys[j]), int(xs[j])
        root_lab = lab[ry, rx]
    comp = lab == root_lab
    if int(comp.sum()) < 4:
        return []
    skel = skeletonize(comp)
    sy, sx = np.nonzero(skel)
    pix = {(int(y), int(x)) for y, x in zip(sy, sx)}
    if not pix:
        return []

    def nbrs(v):
        y, x = v
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pix:
                    out.append(q)
        return out

    root = min(pix, key=lambda v: (v[0] - ry) ** 2 + (v[1] - rx) ** 2)
    from collections import deque
    par: dict = {root: None}
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in nbrs(u):
            if v not in par:
                par[v] = u
                dq.append(v)

    def path_to(v):
        q: list = []
        while v is not None:
            q.append(v)
            v = par[v]
        return q[::-1]

    deg = {u: len(nbrs(u)) for u in pix}
    ends = [u for u in pix if deg[u] == 1 and u != root]
    cand = []
    for v in ends:
        path = path_to(v)
        if len(path) < 6:
            continue
        pts = np.asarray([[x, y] for (y, x) in path], float)
        arclen = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        cand.append((arclen, pts))
    cand.sort(key=lambda t: -t[0])
    kept: list = []
    for arclen, pts in cand:
        if arclen < float(min_len_px):
            continue
        ok = True
        for _al, base in kept:
            # symmetric max deviation between the two paths
            d_ab = max(float(np.min(np.hypot(
                (pts[:, None, 0] - base[None, :, 0]),
                (pts[:, None, 1] - base[None, :, 1])), axis=1).max()), 0.0)
            d_ba = float(np.min(np.hypot(
                (base[:, None, 0] - pts[None, :, 0]),
                (base[:, None, 1] - pts[None, :, 1])), axis=1).max())
            if max(d_ab, d_ba) < float(sep_px):
                ok = False
                break
        if ok:
            kept.append((arclen, pts))
        if len(kept) >= int(max_routes):
            break
    out = []
    for i, (arclen, pts) in enumerate(kept):
        if int(smooth) > 1 and len(pts) > int(smooth):
            ker = np.ones(int(smooth)) / float(smooth)
            pts = np.stack([np.convolve(pts[:, 0], ker, "valid"),
                            np.convolve(pts[:, 1], ker, "valid")], 1)
        out.append({
            "route_id": f"bw{i}",
            "polyline": [[float(x), float(y)] for x, y in pts],
            "seed": f"body-walk-{i}",
            "seed_heat": 0.0,
            "attachment": [float(root_xy[0]), float(root_xy[1])],
            "stopped_early": False,
            "arclen_px": round(float(arclen), 1)})
    return out
