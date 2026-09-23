"""rev11 item 6: the assisted interval over the banked crossing.

The review's product requirements exercised here:
- "Build a contiguous assisted interval containing the banked crossing"
  -> frames 51030-51450 (the three human-resolved crossing anchors),
     sampled every 30 frames.
- "Preserve uncertainty": a frame is `measured` only at the three
  human anchor frames (lane polylines drawn by the user); between
  anchors the state is `unmeasured` and the nearest anchor's lane is
  shown as CONTEXT only (dashed) — no geometry is interpolated or
  extrapolated (growth is never extended).
- "Distinguish one tube per physical grain from multiple neighboring
  grains": each lane is attributed to a persistent OWNER (ray-based
  links from the registry, which resolved the A/B label swap), and
  overlap is permitted (the lanes cross) — owner identity is never
  merged.
- "corrections survive restart": store-level test on a COPY of the
  project DB (save -> close -> reopen -> verify + revision history).
- export consistency: exported path endpoint == exported tip;
  arclength == exported length; tip-only withholds length; NaN is
  refused (adapter_v29.export_tip_path_length).

Artifacts: runs/prototypes/v30/rev11_interval_demo/
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

X_DB = Path('/Users/joshjiang/Documents/TubeTracker-annotator-projects/'
            'rev10round/annotations.db')
REG = REPO / 'runs/prototypes/v30/rev11own_verify'
OUT = REPO / 'runs/prototypes/v30/rev11_interval_demo'
MOVIE = '/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4'
F0, F1, STEP = 51030, 51450, 30
CROP = (880, 250, 260, 190)          # x, y, w, h around the crossing


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from prototypes.v30_video_apex.adapter_v29 import (
        export_tip_path_length)
    from tubetracker.annotation_frames import FrameReader

    con = sqlite3.connect(X_DB)
    crossings = {u: json.loads(d) for u, d in con.execute(
        "select uuid, data from entities where kind='crossing'")}
    con.close()
    links = json.loads((REG / 'crossing_owner_links.json').read_text())
    owners = {o['id']: o for o in json.loads(
        (REG / 'owners.json').read_text())['owners']}

    # lane -> owner per crossing frame (from the registry ray test)
    lane_owner: dict[tuple[int, str], str] = {}
    for lk in links:
        lane_owner[(int(lk['frame']), lk['lane'])] = lk['owner']
    anchors: dict[int, dict] = {}
    for c in crossings.values():
        f = int(c['source_frame'])
        anchors[f] = {lab: [list(map(float, q)) for q in pts]
                      for lab, pts in c['lanes'].items()}
    print('anchor frames:', sorted(anchors))

    frames = list(range(F0, F1 + 1, STEP))
    rows = []
    for f in frames:
        for (af, lab), owner in sorted(lane_owner.items()):
            if af != f:
                continue
            pts = anchors[f][lab]
            rows.append({'frame': f, 'owner': owner, 'lane': lab,
                         'state': 'measured', 'path': pts,
                         'tip': pts[-1]})
    # unmeasured frames: nearest anchor lane as context, flagged
    ctx = []
    for f in frames:
        if f in anchors:
            continue
        af = min(anchors, key=lambda a: abs(a - f))
        for lab in sorted(anchors[af]):
            owner = lane_owner.get((af, lab))
            if owner is None:
                continue
            ctx.append({'frame': f, 'owner': owner, 'lane': lab,
                        'state': 'unmeasured',
                        'context_from': af,
                        'delta_frames': abs(af - f),
                        'path': anchors[af][lab], 'tip': None})

    # ---- export consistency -----------------------------------------
    checks = []
    for r in rows:
        e = export_tip_path_length(r['path'], front_s=None)
        arr = np.asarray(r['path'], float)
        seg = np.diff(arr, axis=0)
        arc = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        ok_end = (abs(e['tip'][0] - r['path'][-1][0]) < 1e-6
                  and abs(e['tip'][1] - r['path'][-1][1]) < 1e-6)
        ok_len = abs(e['length_px'] - arc) < 1e-6
        checks.append({'case': f"lane@{r['frame']}/{r['lane']}",
                       'endpoint_eq_tip': bool(ok_end),
                       'arclength_eq_length': bool(ok_len),
                       'length_px': round(arc, 3)})
    tip_only = export_tip_path_length(None, None, tip_xy=[500.0, 500.0])
    checks.append({'case': 'tip-only', 'withheld': tip_only['withheld'],
                   'length_px': tip_only['length_px']})
    try:
        export_tip_path_length(rows[0]['path'], front_s=float('nan'))
        checks.append({'case': 'nan-front', 'refused': False})
    except ValueError:
        checks.append({'case': 'nan-front', 'refused': True})

    # ---- restart / correction persistence (on a COPY) ---------------
    store_db = OUT / 'store_copy.db'
    if store_db.exists():
        store_db.unlink()
    shutil.copy2(X_DB, store_db)
    from tubetracker.annotation_store import AnnotationStore
    cu = next(iter(crossings))
    s1 = AnnotationStore(store_db)
    rev_before = s1.load(cu)['revision']
    data = s1.load(cu)['data']
    corr = json.loads(json.dumps(data))
    key = next(iter(corr['lanes']))
    corr['lanes'][key][0] = [corr['lanes'][key][0][0] + 1.5,
                             corr['lanes'][key][0][1] - 0.75]
    rev_after = s1.save('crossing', cu, corr, actor='demo-correction')
    s1.close()
    s2 = AnnotationStore(store_db)          # <- the "restart"
    got = s2.load(cu)
    persisted = (got['data']['lanes'][key][0]
                 == corr['lanes'][key][0])
    hist = s2.history(cu)
    s2.close()
    restart = {'uuid': cu, 'rev_before': rev_before,
               'rev_after': rev_after,
               'correction_persisted': bool(persisted),
               'n_revisions': len(hist),
               'actors': [h.get('actor') for h in hist]}
    print('restart test:', restart)

    # ---- render: PNGs + montage + mp4 -------------------------------
    import cv2
    r = FrameReader(MOVIE)
    x, y, w, h = CROP
    montage = []
    mp4 = OUT / 'interval.mp4'
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'),
                         5.0, (w, h))
    for f in frames:
        img = r.read(f).frame[y:y + h, x:x + w].copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = img.copy()
        for rw in rows + ctx:
            if rw['frame'] != f:
                continue
            pts = np.asarray(rw['path'], float) - np.array([x, y])
            measured = rw['state'] == 'measured'
            col = ((0, 220, 0) if rw['lane'] == 'A'
                   else (0, 160, 255))
            p = pts.astype(np.int32).reshape(-1, 1, 2)
            if measured:
                cv2.polylines(vis, [p], False, col, 2)
                cv2.circle(vis, tuple(p[-1, 0]), 4, col, -1)
            else:
                cv2.polylines(vis, [p], False, col, 1,
                              lineType=cv2.LINE_AA)
                for a, b in zip(p[:-1, 0], p[1:, 0]):
                    cv2.line(vis, tuple(a), tuple(b), col, 1)
        for oid in ('own-ld-0001', 'own-ld-0002'):
            o = owners[oid]
            gx, gy = np.asarray(o['grain_native']) - np.array([x, y])
            axx, ayy = np.asarray(o['attachment_native']) - np.array([x, y])
            if 0 <= gx < w and 0 <= gy < h:
                cv2.drawMarker(vis, (int(gx), int(gy)), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 12, 2)
                cv2.putText(vis, oid.replace('own-ld-', 'o'),
                            (int(gx) + 6, int(gy) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)
        txt = [f'frame {f}']
        for rw in rows + ctx:
            if rw['frame'] != f:
                continue
            who = (rw['owner'].replace('own-ld-', 'o')
                   if rw['owner'] else 'UNRESOLVED')
            st = (f"L{rw['lane']}->{who}:"
                  + (' MEASURED' if rw['state'] == 'measured'
                     else f" unmeasured (ctx {rw['context_from']}"
                          f" +-{rw['delta_frames']}f)"))
            txt.append(st)
        for i, line in enumerate(txt):
            cv2.putText(vis, line, (6, 14 + i * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f'frame_{f}.png'), vis)
        montage.append(vis)
        vw.write(vis)
    vw.release()
    r.close()
    # montage: first 8 frames in a 2x4 grid
    grid = np.zeros((h * 2, w * 4, 3), np.uint8)
    for i, im in enumerate(montage[:8]):
        grid[(i // 4) * h:(i // 4 + 1) * h,
             (i % 4) * w:(i % 4 + 1) * w] = im
    cv2.imwrite(str(OUT / 'montage.png'), grid)

    res = {'interval': [F0, F1], 'step': STEP, 'frames': frames,
           'acquisition_cadence_s': 3.0,
           'time_note': ('ld movie cadence is 3.0 s/frame (declared); '
                         'frame f is at f*3.0 s from movie start'),
           'calibration': 'native pixels, no resampling',
           'measured_rows': rows, 'context_rows': ctx,
           'export_checks': checks,
           'restart_test': restart,
           'note': 'measured only at human anchors; context lanes are '
                   'never extrapolated geometry'}
    (OUT / 'interval_demo.json').write_text(json.dumps(res, indent=1))
    n_ok = sum(1 for c in checks if c.get('endpoint_eq_tip')
               and c.get('arclength_eq_length'))
    print(f"export checks: {n_ok}/{len(rows)} measured lanes fully "
          f"consistent; tip-only withheld={tip_only['withheld']}; "
          f"nan refused={checks[-1].get('refused')}")
    print(f"-> {OUT}/interval_demo.json, montage.png, interval.mp4 "
          f"({len(frames)} frames)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
