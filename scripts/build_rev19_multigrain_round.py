"""rev19 MULTI-GRAIN field-window round: census + grain-identity adjudication.

PROVENANCE NOTE (documented loss): the original round ask is quoted in the
project history only as "MULTI-GRAIN field-window round (359 field windows)
where multiple grains compete for tube ownership (identity/route
adjudication)". No artifact defines the 359 windows and the verbatim ask is
not recoverable from session history (7 targeted searches). The window count
below is therefore RE-DERIVED under the explicit definition in `census()`,
and that definition travels with the output so the number is auditable.

CENSUS (the movie-wide "note the multi-pollen views" ask):
  sample frames every DEF_STRIDE frames; detect grains
  (detect_grains + split_mergers, radius via peak_radii); group each frame's
  grains by single-linkage at LINK_PX -- both grains must fit inside one
  zoom-5 view (find_neighbor_balls' 40-170 px pair bound). A MULTI-GRAIN
  field window = a (frame, cluster) with >= 2 grains. Every window is
  catalogued in rev19_multigrain_census.json.

ROUND (grain_identity tasks = the multi-tracker's human identity anchors;
  grain_identity_reviews -> load_grain_poses / grain_motion already consume
  them downstream): for each rev17 S-grain, one anchor per ANCHOR_STRIDE
  bucket inside its reviewed interval, taken at a MULTI-GRAIN window (the
  frame whose cluster holds a competing grain). Task = "click the centre of
  THIS grain at this frame"; 'ambiguous' and 'out_of_field' are real answers.

Writes tasks via the rev18 insertion pattern (queue fields + AnnotationStore)
into the live annotation project; the verification sheet mirrors rev18_views.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "prototypes/timesfm_tip_forecast"))
sys.path.insert(0, str(REPO / "prototypes/v30_video_apex"))

from grain_detect import (  # noqa: E402
    detect_grains, grain_symmetry_maps, peak_radii, split_mergers)
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from build_rev18_length_round import (  # noqa: E402
    GRAINS, MOVIE, MOVIE_HASH, PROJECT, QUEUE, ROLES, SCRATCH, track_centres)

DEF_STRIDE = 500        # census frame grid
LINK_PX = 170.0         # one zoom-5 view fits a pair <= 170px apart
MIN_PAIR_PX = 40.0      # closer than this = touching clump (still one window)
ANCHOR_STRIDE = 3000    # identity-anchor cadence (grain_motion interpolates)
BATCH = "rev19-multigrain"
STRATUM = "multigrain_identity_adjudication"

WHY = """MULTI-GRAIN WINDOW — {n} pollen grains share this view (multi-pollen).
The tracker must keep each tube with its own grain, so it needs one human
identity anchor here.

This task is about ONE grain: {sid} (green ring + magenta centre cross).
It is the same physical grain as in the reference view.

At frame {f}: click the CENTRE of that same physical grain (one click).
  • Several grains compete in this view — pick {sid}, not its neighbors
    (other detected grains are circled yellow).
  • 'ambiguous': press if you cannot tell the grains apart at this frame.
  • 'out_of_field': press if {sid} has left the view.
Unknown is a real answer; never guess between neighbors."""


def _radii(gray, dets):
    if not dets:
        return []
    maps = grain_symmetry_maps(gray, (11, 14, 17, 20))
    return list(peak_radii(maps, [(x, y) for x, y, _s in dets]))


def _clusters(pts, link=LINK_PX):
    """Single-linkage clusters of (x, y, r, score) at `link` px."""
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= link:
                parent[find(i)] = find(j)
    out = defaultdict(list)
    for i in range(n):
        out[find(i)].append(pts[i])
    return sorted(out.values(), key=lambda c: (c[0][1], c[0][0]))


def census(reader: FrameReader, frame_count: int):
    """Multi-grain field windows on the DEF_STRIDE frame grid (see docstring).

    Returns (windows, per_frame) where each window is
    {'frame', 'cluster', 'grains': [{'xy', 'r', 'score'}...]} and only
    clusters with >= 2 grains are windows.
    """
    windows, per_frame = [], {}
    for f in range(0, frame_count, DEF_STRIDE):
        g = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY)
        det = detect_grains(g)
        det = split_mergers(g, det)
        rs = _radii(g, det)
        pts = [(float(x), float(y), float(r), float(s))
               for (x, y, s), r in zip(det, rs)]
        per_frame[f] = pts
        for k, cl in enumerate(_clusters(pts)):
            if len(cl) >= 2:
                windows.append({'frame': int(f), 'cluster': int(k),
                                'grains': [{'xy': [round(x, 1), round(y, 1)],
                                            'r': round(r, 2), 'score': round(s, 3)}
                                           for x, y, r, s in cl]})
    return windows, per_frame


def main() -> int:
    reader = FrameReader(MOVIE)
    try:
        _cap = cv2.VideoCapture(MOVIE)
        frame_count = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30000
        _cap.release()
    except Exception:
        frame_count = 30000
    frame_count = max(frame_count, 28000)   # intervals reach f27500
    windows, per_frame = census(reader, frame_count)

    # tracked centres at census frames inside each grain's interval
    tracked = {}
    for sid, anchor_xy, rad, anchor_f, ladder in GRAINS:
        lo, hi = min(ladder), max(ladder)
        frames = sorted(f for f in per_frame if lo <= f <= hi)
        frames = sorted(set(frames) | {int(anchor_f)})
        centres = track_centres(reader, anchor_xy, anchor_f, frames)
        tracked[sid] = {'rad': float(rad), 'centres': centres,
                        'interval': (lo, hi)}

    # competitive windows: window holds one tracked grain + a competitor
    comp = []   # (sid, frame, n_grains, competitor_dist, window_ref)
    for w in windows:
        f = w['frame']
        for sid, t in tracked.items():
            if not (t['interval'][0] <= f <= t['interval'][1]):
                continue
            c = t['centres'].get(f)
            if c is None:
                continue
            d = [np.hypot(g['xy'][0] - c[0], g['xy'][1] - c[1])
                 for g in w['grains']]
            mine = [i for i, dd in enumerate(d) if dd <= 6.0]
            if not mine:
                continue
            others = [dd for i, dd in enumerate(d) if i not in mine]
            comp.append({'sid': sid, 'frame': int(f),
                         'n_grains': len(w['grains']),
                         'competitor_px': round(min(others), 1) if others else None,
                         'window': w})

    # one anchor per ANCHOR_STRIDE bucket per grain: the most contested window
    picked = []
    for sid, t in tracked.items():
        lo, hi = t['interval']
        for b0 in range(lo, hi + 1, ANCHOR_STRIDE):
            cands = [c for c in comp
                     if c['sid'] == sid and b0 <= c['frame'] < b0 + ANCHOR_STRIDE]
            if not cands:
                continue
            best = min(cands,
                       key=lambda c: (c['competitor_px'] if c['competitor_px']
                                      is not None else 1e9, c['frame']))
            picked.append(best)
    picked.sort(key=lambda c: (c['sid'], c['frame']))

    # verification sheet
    tiles = []
    for c in picked:
        sid, f = c['sid'], c['frame']
        cx, cy = tracked[sid]['centres'][f]
        rad = tracked[sid]['rad']
        g = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY)
        half = 64
        h, w_ = g.shape
        x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
        crop = g[y0:min(h, int(cy) + half), x0:min(w_, int(cx) + half)]
        crop = cv2.copyMakeBorder(
            crop, 0, max(0, 2 * half - crop.shape[0]),
            0, max(0, 2 * half - crop.shape[1]), cv2.BORDER_REFLECT)
        S = 4
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (rgb.shape[1] * S, rgb.shape[0] * S),
                         interpolation=cv2.INTER_LANCZOS4)
        ccx, ccy = int((cx - x0) * S), int((cy - y0) * S)
        cv2.circle(rgb, (ccx, ccy), int(rad * S), (0, 255, 0), 2)
        cv2.drawMarker(rgb, (ccx, ccy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for gnt in c['window']['grains']:
            gx, gy = gnt['xy']
            if np.hypot(gx - cx, gy - cy) <= 6.0:
                continue
            cv2.circle(rgb, (int((gx - x0) * S), int((gy - y0) * S)),
                       int(max(8.0, gnt['r']) * S), (255, 220, 0), 2)
        cv2.putText(rgb, f"{sid} f{f} n={c['n_grains']}", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(rgb)
    if tiles:
        h, w_ = tiles[0].shape[:2]
        cols = 5
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * h + 8 * (rows + 1), cols * w_ + 8 * (cols + 1),
                         3), 15, np.uint8)
        for k, im in enumerate(tiles):
            yy = 8 + (k // cols) * (h + 8)
            xx = 8 + (k % cols) * (w_ + 8)
            sheet[yy:yy + h, xx:xx + w_] = im
        cv2.imwrite(str(SCRATCH / "rev19_multigrain_views.png"), sheet)
    reader.close()

    census_doc = {
        'definition': {
            'frame_grid': f'every {DEF_STRIDE} frames, 0..{frame_count}',
            'window': (f'single-linkage clusters of detected grains at '
                       f'{LINK_PX}px (one zoom-5 view), >=2 grains = '
                       f'MULTI-GRAIN field window'),
            'detector': 'grain_detect.detect_grains + split_mergers',
            'provenance': ('re-derived: the original "359 field windows" '
                           'definition is not recoverable from any artifact'),
        },
        'n_frames_sampled': len(per_frame),
        'n_windows': len(windows),
        'windows': windows,
    }
    (SCRATCH / 'rev19_multigrain_census.json').write_text(
        json.dumps(census_doc, indent=1))

    store = AnnotationStore(PROJECT / "annotations.db")
    order = 0
    try:
        before = sum(1 for t in store.unfinished_tasks(limit=500)
                     if t.get("review_queue") == QUEUE)
        for c in picked:
            sid, f = c['sid'], c['frame']
            cx, cy = tracked[sid]['centres'][f]
            task = {
                'uuid': f'rev19mg-{sid}-f{f:05d}',
                'task_type': 'grain_identity',
                'movie': 'ld',
                'movie_uuid': 'ld',
                'movie_content_hash': MOVIE_HASH,
                'owner_uuid': f'rev17-{sid}',
                'grain_id': f'rev17-{sid}',
                'query_frames': [int(f)],
                'focus_xy': [round(float(cx), 2), round(float(cy), 2)],
                'target_xy': [round(float(cx), 2), round(float(cy), 2)],
                'target_r': tracked[sid]['rad'],
                'view_zoom': 5.5,
                'review_queue': QUEUE,
                'queue_order': 300 + order,
                'queue_batch': BATCH,
                'queue_position': order + 1,
                'stratum': STRATUM,
                'role': ROLES.get(sid, 'calibration'),
                'geometry_type': 'point',
                'class_scope': 'grain',
                'instance_scope': f'rev17-{sid}',
                'completeness': 'unreviewed',
                'completed': False,
                'why': WHY.format(sid=sid, f=f, n=c['n_grains']),
                'multigrain_window': {
                    'n_grains': c['n_grains'],
                    'competitor_px': c['competitor_px'],
                    'grains': c['window']['grains'],
                },
            }
            store.save('task', task['uuid'], task, actor='rev19mg')
            order += 1
        after = sum(1 for t in store.unfinished_tasks(limit=500)
                    if t.get("review_queue") == QUEUE)
        print(f'QUEUE pending: {before} -> {after} (wrote {order} tasks)')
    finally:
        store.close()

    per_grain = defaultdict(int)
    for c in picked:
        per_grain[c['sid']] += 1
    print(f'census: {len(per_frame)} frames sampled, '
          f'{len(windows)} MULTI-GRAIN field windows '
          f'(definition: every {DEF_STRIDE} frames, link {LINK_PX}px, >=2 grains)')
    print(f'competitive (tracked grain + competitor): {len(comp)}')
    print(f'round: {order} grain_identity anchors '
          f'per grain: {dict(sorted(per_grain.items()))}')
    print(f'wrote rev19_multigrain_census.json + rev19_multigrain_views.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
