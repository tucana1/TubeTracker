"""Model-only tube route + grain-identity tracking through a reviewed interval.

Fills the prototype gap between onset detection and the human growth curves:
per frame, the tube's CURRENT PATH (rim exit -> tip) reconstructed from the
grain's OWN dark component, plus the identity evidence that keeps the tube
attached to its own grain. Pure image + posed-centre math (emergence_onset
style) -- no model, no GPU.

Path: the validated farthest-component reach finds the tip; from the tip a
descent walk over the component's own pixels follows the tube back to the
grain rim. Unlike a radial ray the walk tolerates a bending tube (S9's tip
angle drifts 155->136 deg while lengthening). Every path pixel lies on the
grain's own dark component = geometrically supported; support is reported,
never assumed.

Cases preserved, never forced:
  state 'no_protrusion'   -- nothing beyond the pore baseline (typed absence)
  flag  'partial_cap'     -- tip at the reach cap: tube leaves the measured disc
  flag  'partial_gap'     -- mask gap broke the walk; path truncated at the gap
  flag  'two_protrusions' -- a second bump >= 45% of peak, >30 deg apart:
                             crossing or second tube -> uncertain, never merged
  flag  'merged_component'-- component far over body+tube area: clump merged
                             into one dark blob -> identity uncertain
uncertain = two_protrusions or merged_component; partial = partial_cap or
partial_gap. Rows survive either way -- uncertain/partial cases are preserved,
not dropped or guessed.

Identity evidence per interval (`identity_checks`): exit-angle stability at
the rim (the one-tube-from-one-pore criterion the investigator uses),
outward-monotone length (biology: s(t) monotone non-decreasing), tip-step
sanity (the LEDGER acceptance gate: per-frame tip motion is noise, growth is
not). All values here are MODEL-ONLY; human-corrected numbers only exist in
the evaluation fit against the reviewed traces.

Persistence: `save_tracks`/`load_tracks` round-trip the full rows; `export_csv`
dumps the tabular view. Recompute is deterministic (same movie + centres ->
same rows), which is what makes save/restart/recompute/export trustworthy.
"""
from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from emergence_onset import boundary_radius_profile, grain_component_mask

PORE_BASELINE = 1.0     # px past rim a stable germ-pore bump may occupy
MAX_REACH = 56          # measured disc cap (matches boundary_radius_profile)
CROP = 62               # half-size tile; > MAX_REACH so the cap, not the crop


def _angle_deg(pt, centre):
    return math.degrees(math.atan2(-(pt[1] - centre[1]), pt[0] - centre[0])) % 360


def _arc(points):
    a = np.asarray(points, float)
    return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))


def _ridge_walk(mask, dist, tip, grain_xy, rim, restrict=None, max_steps=800):
    """Follow the tube's thin dark ridge from the tip back to its visible base.

    Directional walk with inertia -- NOT a radial descent. Tubes curve (S11's
    bends ~80 deg around the body, its dark line even crossing the bright
    centre in projection); a min-distance descent cuts such bends and dives
    into the body ring, truncating the base (the systematic under-read rev19
    scored against the 16 traces).

    Termination mirrors the human convention ("where the tube crosses/cuts
    the grain rim"): cone gap where the visible dark line ends (the faint
    germ-pore case -- the visible base sits INSIDE the median rim circle),
    thickness blowup where the ridge merges into the body ring, or a radial
    backstop deep inside the ring as safety. A mid-tube break is flagged,
    never silently dropped.

    Returns (points tip->base, end) with end in
    'pore_gap' | 'body' | 'deep' | 'mid_break' | 'max_steps'.
    """
    seen = {tip}
    pts = [tip]
    h, w = mask.shape
    # Walkable band: outside rim*0.8. Below it lies the body-ring bulk, not
    # the tube; the faint-pore base ends just inside. Together with the radial
    # descent rule this makes ring-following (walks of 48-79px circling an
    # 80px circumference grain) geometrically impossible.
    ok = mask & (dist >= rim * 0.8)
    if restrict is not None:
        ok = ok & restrict

    def _unit(a, b):
        vx, vy = float(b[0] - a[0]), float(b[1] - a[1])
        n = math.hypot(vx, vy) or 1.0
        return (vx / n, vy / n)

    def _blend(d1, d2, w1=0.5):
        vx, vy = w1 * d1[0] + (1 - w1) * d2[0], w1 * d1[1] + (1 - w1) * d2[1]
        n = math.hypot(vx, vy) or 1.0
        return (vx / n, vy / n)

    def _thickness(p, d):
        nx, ny = -d[1], d[0]
        run = 1
        for sgn in (1.0, -1.0):
            for t in range(1, 7):
                q = (int(round(p[0] + sgn * nx * t)),
                     int(round(p[1] + sgn * ny * t)))
                if not (0 <= q[0] < w and 0 <= q[1] < h) or not mask[q[1], q[0]]:
                    break
                run += 1
        return run

    # seed direction along the tube: toward the local ridge patch centroid
    # (the farthest band pixel grabbed body-ring pixels for short tubes and
    # sent the walk around the rim)
    yy, xx = np.mgrid[:h, :w]
    near = ok & ((xx - tip[0]) ** 2 + (yy - tip[1]) ** 2 <= 20.25) \
        & ((xx - tip[0]) ** 2 + (yy - tip[1]) ** 2 >= 1.0)
    ys, xs = np.nonzero(near)
    if len(xs):
        dirv = _unit(tip, (float(xs.mean()), float(ys.mean())))
    else:
        dirv = _unit(tip, (float(grain_xy[0]), float(grain_xy[1])))

    widths = []
    # radial-stall sliding window: the tube descends toward the body; a walk
    # covering 4px of arc with under 0.45x radial descent while riding THICK
    # pixels (>=4.5px = the grain's dark rim ring) has lost the tube and is
    # sliding around the rim. Measured separation: rim slides ride 5-7.5px
    # pixels at <=0.45 descent rate; real wrapping-tube bends ride 2-3px
    # pixels and must never be cut (the thickness gate spares them).
    hist = deque([(0.0, float(dist[tip[1], tip[0]]), 0)])
    cum = 0.0
    cone = 45.0 * math.pi / 180.0
    for _ in range(max_steps):
        x, y = pts[-1]
        if dist[y, x] <= rim * 0.55:
            return pts, 'deep'
        y0, y1 = max(0, y - 3), min(h, y + 4)
        x0, x1 = max(0, x - 3), min(w, x + 4)
        best, best_ang = None, cone
        for yy, xx in zip(*np.nonzero(ok[y0:y1, x0:x1]
                                      & (dist[y0:y1, x0:x1] <= dist[y, x] + 0.2))):
            p = (int(xx) + x0, int(yy) + y0)
            if p in seen:
                continue
            step = _unit((x, y), p)
            ang = math.acos(max(-1.0, min(1.0,
                            step[0] * dirv[0] + step[1] * dirv[1])))
            if ang < best_ang:
                best, best_ang = p, ang
        if best is None:
            d_now = float(dist[y, x])
            return pts, ('pore_gap' if d_now <= rim * 1.35 else 'mid_break')
        seen.add(best)
        step_dir = _unit(pts[-1], best)
        pts.append(best)
        widths.append(_thickness(best, dirv))
        cum += math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
        d_last = float(dist[pts[-1][1], pts[-1][0]])
        # baseline = the thinnest sustained ridge (low percentile, not the
        # first steps: the TIP FLARE is 4-5px and inflates every threshold
        # past the body-ring width; a low percentile self-corrects as thin
        # tube samples accumulate and never drifts up on ring pixels)
        tube_w = (float(np.percentile(widths, 25)) if len(widths) >= 3
                  else 2.5)
        hist.append((cum, d_last, len(pts) - 1))
        while len(hist) > 2 and cum - hist[1][0] >= 4.5:
            hist.popleft()
        w_arc = cum - hist[0][0]
        if (w_arc >= 4.0 and hist[0][1] - d_last < 0.45 * w_arc
                and widths[-1] >= 4.5):
            return pts[:hist[0][2] + 1], 'body_stall'
        if len(widths) >= 4:
            if widths[-1] >= max(1.8 * tube_w, 4.5):
                # thickness spike: the ridge merges into the body ring OR the
                # tube merely CROSSES the ring band in projection (wrapping
                # tubes do). Probe on: a crossing thins back within a few
                # steps; the true base stays thick -> rewind and end there.
                probe = list(pts)
                thick_run = 0
                for _p in range(5):
                    px, py = probe[-1]
                    if dist[py, px] <= rim * 0.55:
                        break
                    py0, py1 = max(0, py - 3), min(h, py + 4)
                    px0, px1 = max(0, px - 3), min(w, px + 4)
                    cand_p, cand_ang = None, cone
                    for yy2, xx2 in zip(*np.nonzero(
                            ok[py0:py1, px0:px1]
                            & (dist[py0:py1, px0:px1] <= dist[py, px] + 0.2))):
                        q = (int(xx2) + px0, int(yy2) + py0)
                        if q in seen:
                            continue
                        st = _unit((px, py), q)
                        a2 = math.acos(max(-1.0, min(1.0,
                                      st[0] * dirv[0] + st[1] * dirv[1])))
                        if a2 < cand_ang:
                            cand_p, cand_ang = q, a2
                    if cand_p is None:
                        break
                    seen.add(cand_p)
                    probe.append(cand_p)
                    wq = _thickness(cand_p, dirv)
                    if wq >= max(1.8 * tube_w, 4.5):
                        thick_run += 1
                        if thick_run >= 3:
                            # sustained thickness = real base: rewind to spike
                            return pts, 'body'
                    else:
                        # thinned back = a crossing, not the base: resume
                        pts = probe
                        widths.append(wq)
                        break
                else:
                    return pts, 'body'
        dirv = _blend(dirv, step_dir, 0.5)
    return pts, 'max_steps'


def _resample(points, step=1.2):
    """Thin a pixel walk to ~step-spaced polyline points (light smoothing).

    Endpoints are preserved exactly: smoothing them toward the first interior
    point collapses a 2-point walk to zero length and shrinks every base.
    """
    pts = [np.asarray(p, float) for p in points]
    if len(pts) < 2:
        return [tuple(p) for p in pts]
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = float(np.hypot(*(b - a)))
        acc += seg
        if acc >= step:
            out.append(b)
            acc = 0.0
    if np.hypot(*(out[-1] - pts[-1])) > 1e-6:
        out.append(pts[-1])
    if len(out) == 2:
        return [(float(p[0]), float(p[1])) for p in out]
    sm = [tuple(out[0])]
    for i in range(1, len(out) - 1):
        sm.append(tuple(np.mean(out[i - 1:i + 2], axis=0)))
    sm.append(tuple(out[-1]))
    return sm


def tube_path(tile, grain_xy, radius, max_reach=MAX_REACH,
              prior_deg=None, prior_half=35.0):
    """MODEL-ONLY current path of this grain's tube at one frame.

    Returns a dict: state, path_xy (rim exit -> tip), exit_xy, tip_xy,
    length_arc, reach (tip px past rim), exit/tip angle at the grain centre,
    flags, and the uncertain/partial booleans. 'no_protrusion' rows carry no
    path -- typed absence, never an empty trace standing in for one.

    prior_deg (optional, y-up display degrees like exit_angle) is the tracked
    tube angle from the interval: the route then follows the protrusion AT
    THIS GRAIN'S PORE within +/-prior_half, not the loudest bump on the rim
    (a pore-noise bump elsewhere flips the route to the wrong side without
    the lock -- observed as 110-135 deg exit-angle spreads).
    """
    tile = np.asarray(tile, np.float32)
    mask, dist, _labels = grain_component_mask(tile, grain_xy, radius)
    profile, ok = boundary_radius_profile(tile, grain_xy, radius, max_reach)
    p = np.where(ok, profile, np.nan)
    rim = float(np.nanmedian(p))
    reach_v = p - rim
    peak = float(np.nanmax(reach_v)) if np.isfinite(reach_v).any() else 0.0
    out = {'rim_r': round(rim, 2), 'reach': None, 'state': 'no_protrusion',
           'path_xy': None, 'exit_xy': None, 'tip_xy': None,
           'length_arc': None, 'exit_angle': None, 'tip_angle': None,
           'walk_end': None,
           'flags': [], 'uncertain': False, 'partial': False}
    # identity lock: restrict the route to this grain's tracked pore angle
    n_bins = len(reach_v)
    if prior_deg is not None:
        d_per_bin = 360.0 / n_bins
        bin_up = np.array([((-np.pi + (b + 0.5) * 2 * np.pi / n_bins)
                            * -180.0 / np.pi) % 360.0 for b in range(n_bins)])
        dd = np.abs((bin_up - (prior_deg % 360.0) + 180.0) % 360.0 - 180.0)
        reach_v = np.where(dd <= max(prior_half, 2 * d_per_bin), reach_v, np.nan)
    peak = float(np.nanmax(reach_v)) if np.isfinite(reach_v).any() else 0.0
    if not np.isfinite(peak) or peak <= PORE_BASELINE:
        return out
    flags = []
    if peak <= 3.0:
        # inside the pore-noise band: nascent tube vs germ-pore bump cannot be
        # separated at this size (rev18 resolution floor) -- preserved honestly
        flags.append('pore_band')
    # second protrusion = crossing / second tube: uncertain, never merged away.
    # A protrusion is a RUN of >=2 bins over max(2.5px, 45% of peak) -- single
    # noisy rim bins and the wrapping tube's own fading tail do not count (the
    # per-bin test flagged ~every row and gutted the identity evidence).
    peak_bin = int(np.nanargmax(reach_v))
    reach_g = p - rim                      # global reach, outside the lock
    thr2 = max(2.5, 0.45 * peak)
    hi = np.isfinite(reach_g) & (reach_g >= thr2)
    n_bins = len(reach_g)
    visited = np.zeros(n_bins, bool)
    visited[peak_bin] = True
    for b0 in range(n_bins):
        if hi[b0] and not visited[b0]:
            run = [b0]
            visited[b0] = True
            for step in (1, -1):
                k = 1
                while hi[(b0 + step * k) % n_bins] \
                        and not visited[(b0 + step * k) % n_bins]:
                    run.append((b0 + step * k) % n_bins)
                    visited[(b0 + step * k) % n_bins] = True
                    k += 1
            if len(run) >= 2:
                flags.append('two_protrusions')
                break
    # sector = contiguous bin run around the peak above max(pore, 30% peak)
    thr = max(PORE_BASELINE, 0.30 * peak)
    n_bins = len(reach_v)
    sel = {peak_bin % n_bins}
    for step in (1, -1):
        k = 1
        while (peak_bin + step * k) % n_bins not in sel \
                and np.isfinite(reach_v[(peak_bin + step * k) % n_bins]) \
                and reach_v[(peak_bin + step * k) % n_bins] >= thr:
            sel.add((peak_bin + step * k) % n_bins)
            k += 1
    width = len(sel)
    if width * (360.0 / n_bins) > 45.0:
        flags.append('wide_sector')
    # sector membership per pixel (same angular convention as the profile)
    h, w = tile.shape
    yy, xx = np.mgrid[:h, :w]
    ang = np.arctan2(yy - grain_xy[1], xx - grain_xy[0])
    bin_idx = np.floor((ang + np.pi) / (2 * np.pi / n_bins)).astype(int) % n_bins
    sel_m1 = {(b + 1) % n_bins for b in sel} | {(b - 1) % n_bins for b in sel}
    in_sector = np.isin(bin_idx, list(sel))
    in_sector_m = np.isin(bin_idx, list(sel | sel_m1))   # +1 bin walk margin
    # tip = farthest own-component pixel inside the sector wedge
    cand = mask & in_sector & (dist >= rim * 0.55) & (dist < max_reach)
    if not cand.any():
        cand = mask & (dist >= rim * 0.55) & (dist < max_reach)
    ys, xs = np.nonzero(cand)
    k = int(np.argmax(dist[ys, xs]))
    tip = (int(xs[k]), int(ys[k]))
    if dist[tip[1], tip[0]] >= max_reach - 1.5:
        flags.append('partial_cap')
    walk, end = _ridge_walk(mask, dist, tip, grain_xy, rim)
    if end == 'body_stall':
        # walk trimmed at the rim-slide start (the visible base); preserved
        flags.append('stall_trim')
    # plausibility cap from the biology: a tube's arc cannot grossly exceed
    # its reach (observed curved tubes run <=1.6x). Longer = the walk wandered
    # off the tube -- truncate the base end and flag, never report it as path.
    reach_px = float(dist[tip[1], tip[0]]) - rim
    max_arc = 2.2 * max(reach_px, 0.0) + 4.0
    arc_acc, cut = 0.0, len(walk)
    for i in range(1, len(walk)):
        arc_acc += math.hypot(walk[i][0] - walk[i - 1][0],
                              walk[i][1] - walk[i - 1][1])
        if arc_acc > max_arc:
            cut = i
            break
    if cut < len(walk):
        walk = walk[:cut]
        flags.append('walk_truncated')
    if end in ('mid_break', 'max_steps'):
        flags.append('partial_gap')
    if len(walk) < 2:
        flags.append('partial_gap')
        out.update(state='path', flags=flags, uncertain='pore_band' in flags,
                   partial=True,
                   tip_xy=[round(tip[0], 2), round(tip[1], 2)],
                   reach=round(float(dist[tip[1], tip[0]]) - rim, 2),
                   tip_angle=round(_angle_deg(tip, grain_xy), 1))
        return out
    walk = walk[::-1]                       # base -> tip (visible base first)
    path = _resample(walk)
    if len(path) < 2:
        flags.append('partial_gap')
        out.update(state='path', flags=flags, partial=True,
                   uncertain='pore_band' in flags,
                   tip_xy=[round(tip[0], 2), round(tip[1], 2)],
                   reach=round(float(dist[tip[1], tip[0]]) - rim, 2))
        return out
    # clump guard: component far over body+tube area = merged dark blob
    comp_area = int(mask[dist >= rim * 0.55].sum())
    expected = math.pi * (1.4 * radius) ** 2
    if comp_area > expected:
        flags.append('merged_component')
    exit_pt, tip_pt = path[0], path[-1]
    uncertain = ('two_protrusions' in flags) or ('merged_component' in flags) \
        or ('pore_band' in flags)
    partial = ('partial_cap' in flags) or ('partial_gap' in flags) \
        or ('walk_truncated' in flags)
    out.update(state='path',
               path_xy=[[round(float(a), 2), round(float(b), 2)] for a, b in path],
               exit_xy=[round(float(exit_pt[0]), 2), round(float(exit_pt[1]), 2)],
               tip_xy=[round(float(tip_pt[0]), 2), round(float(tip_pt[1]), 2)],
               length_arc=round(_arc(path), 2),
               reach=round(float(dist[tip[1], tip[0]]) - rim, 2),
               exit_angle=round(_angle_deg(exit_pt, grain_xy), 1),
               tip_angle=round(_angle_deg(tip_pt, grain_xy), 1),
               walk_end=end,
               flags=list(dict.fromkeys(flags)), uncertain=uncertain,
               partial=partial)
    return out


def track_interval(reader, sid, anchor_xy, radius, anchor_f, frames, crop=CROP):
    """MODEL-ONLY route rows through the reviewed interval at tracked centres.

    Centre drift uses the validated incremental circle fit (rev18 machinery),
    so every row is the tube AT ITS OWN GRAIN at that frame. Rows are tagged
    provenance=model_only -- human-corrected numbers live only in the fit.
    """
    import sys

    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from build_rev18_length_round import track_centres

    frames = sorted({int(f) for f in frames} | {int(anchor_f)})
    centres = track_centres(reader, anchor_xy, anchor_f, frames)
    rows = []
    prior = None
    # pass 1 (sparse, unlocked): find the interval's dominant tube angle =
    # the identity lock (the pore this tube grows from). The max-reach frame
    # is the mature tube; its tip direction seeds the tracked angle.
    seed_frames = frames[::4] + [int(anchor_f)]
    best_reach, best_tip_ang = -1.0, None
    for f in sorted(set(seed_frames)):
        cx, cy = centres.get(f, anchor_xy)
        x0, y0 = max(0, int(cx) - crop), max(0, int(cy) - crop)
        g = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        tile = g[y0:min(h, int(cy) + crop), x0:min(w, int(cx) + crop)]
        r1 = tube_path(tile, (cx - x0, cy - y0), radius)
        if r1.get('tip_angle') is not None and (r1.get('reach') or 0) > best_reach:
            best_reach = float(r1['reach'])
            best_tip_ang = float(r1['tip_angle'])
    prior = best_tip_ang
    for f in frames:
        cx, cy = centres.get(f, anchor_xy)
        x0, y0 = max(0, int(cx) - crop), max(0, int(cy) - crop)
        g = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        tile = g[y0:min(h, int(cy) + crop), x0:min(w, int(cx) + crop)]
        r = tube_path(tile, (cx - x0, cy - y0), radius, prior_deg=prior)
        # back to frame coordinates (tube_path works in tile pixels)
        if r['path_xy']:
            r['path_xy'] = [[round(q[0] + x0, 1), round(q[1] + y0, 1)]
                            for q in r['path_xy']]
        if r['exit_xy']:
            r['exit_xy'] = [round(r['exit_xy'][0] + x0, 1),
                            round(r['exit_xy'][1] + y0, 1)]
        if r['tip_xy']:
            r['tip_xy'] = [round(r['tip_xy'][0] + x0, 1),
                           round(r['tip_xy'][1] + y0, 1)]
        row = {'sid': sid, 'frame': f, 'centre': [round(cx, 2), round(cy, 2)],
               'provenance': 'model_only', **r}
        rows.append(row)
    return rows


def identity_checks(rows):
    """Interval-level identity evidence: is this one tube from one pore?

    exit spread  -- max deviation of exit angles from their circular mean
    violations   -- length drops beyond pixel-resolution slack (s(t) must not
                    shrink; slack is 0.5px floor / 10% relative on longer
                    tubes, so arc-measurement jitter on a 3px bump is not a
                    violation but a real retraction is)
    tip steps    -- median/max per-step tip movement (growth is slow; thrash
                    is noise, the LEDGER per-frame acceptance gate)
    verdict      -- identity_stable | identity_review (preserved either way)

    Only SUPPORTED rows carry identity evidence: full path, not pore-band
    (below the resolution floor), not partial, not a merged clump. Crossing
    flags vote (the lock pins the route to one pore, so the spread then
    measures real stability) but are preserved on the rows either way.
    """
    rows_all = rows
    rows = [r for r in rows if r.get('state') == 'path'
            and not r.get('partial')
            and 'pore_band' not in (r.get('flags') or [])
            and 'merged_component' not in (r.get('flags') or [])
            and r.get('exit_angle') is not None and r.get('length_arc') is not None
            and r.get('tip_xy') is not None]
    if not rows:
        n_path = sum(1 for r in rows_all if r.get('state') == 'path')
        return {'verdict': 'no_supported_rows' if n_path else 'no_tube_in_interval',
                'n_rows': 0, 'n_path_rows': n_path}
    angs = np.array([r['exit_angle'] for r in rows], float)
    vu = np.array([np.cos(np.radians(angs)), np.sin(np.radians(angs))])
    mean_a = math.degrees(math.atan2(vu[1].mean(), vu[0].mean())) % 360
    dev = np.abs((angs - mean_a + 180.0) % 360.0 - 180.0)
    lens = np.array([r['length_arc'] for r in rows], float)
    slack = np.maximum(0.5, 0.1 * lens[1:])
    violations = int(np.sum(np.diff(lens) < -slack)) if len(lens) >= 2 else 0
    tips = np.array([r['tip_xy'] for r in rows], float)
    steps = np.hypot(np.diff(tips[:, 0]), np.diff(tips[:, 1])) \
        if len(tips) >= 2 else np.array([0.0])
    # verdict on ROBUST stats: one bad base-find is an outlier to report, not
    # an identity switch (a switch is a cluster at another pore). Genuinely
    # drifting bases (S6: human exit drifts 30 deg) and wrapping tubes (S11:
    # human exit spread 22 deg) land in identity_review on merit.
    dev_p50 = float(np.percentile(dev, 50))
    dev_p90 = float(np.percentile(dev, 90))
    outliers = int(np.sum(dev > 45.0))
    outlier_frac = outliers / max(1, len(rows))
    v_frac = violations / max(1, len(rows))
    stable = (dev_p90 <= 20.0 and outlier_frac <= 0.05 and v_frac <= 0.15
              and float(np.median(steps)) <= 3.0)
    return {'verdict': 'identity_stable' if stable else 'identity_review',
            'n_rows': len(rows), 'exit_spread_deg': round(float(np.max(dev)), 1),
            'exit_dev_p50_deg': round(dev_p50, 1),
            'exit_dev_p90_deg': round(dev_p90, 1),
            'exit_outliers': outliers,
            'exit_outlier_frac': round(outlier_frac, 3),
            'exit_angle_mean': round(mean_a, 1), 'monotone_violations': violations,
            'monotone_violation_frac': round(v_frac, 3),
            'tip_step_median_px': round(float(np.median(steps)), 2),
            'tip_step_max_px': round(float(np.max(steps)), 2)}


def save_tracks(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))


def load_tracks(path):
    return json.loads(Path(path).read_text())


def export_csv(path, rows):
    cols = ['sid', 'frame', 'state', 'length_arc', 'reach', 'exit_angle',
            'tip_angle', 'exit_xy', 'tip_xy', 'flags', 'uncertain', 'partial',
            'provenance']
    with Path(path).open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            rr = dict(r)
            rr['flags'] = '|'.join(rr.get('flags') or [])
            wr.writerow(rr)
