"""rev11 item 6: persistent physical-grain IDs linked ACROSS frames.

The review: "Create persistent physical-grain IDs and link the same
grain across frames. ... Use future context to resolve historical
identity without extending historical growth."

Method (declared, no machine guesses of identity):
- The 7 persistent owners (owners.json) were human-identified at their
  frames (grain + attachment).
- The v29 causal-growth traces (centerlines.csv) give per-frame tube
  geometry per pollen_id. A trace is matched to an owner ONLY if its
  start point at the owner's identification frame lies within
  MATCH_TOL_PX of the owner's recorded attachment — the same
  attachment the human marked.
- The matched pid's own per-frame rows then ARE the owner's track
  (same trace = same physical tube; its start stays at the
  attachment, its tip moves with measured growth). Nothing is
  extrapolated: a frame the trace does not cover stays UNCOVERED, and
  pre-growth frames report the trace's own measured (tiny) extent —
  historical identity from future context, historical growth never
  extended.
- Continuity is checked (per-frame start jump); any jump beyond
  JUMP_TOL_PX is flagged as an identity risk, not smoothed away.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OWNERS = REPO / 'runs/prototypes/v30/rev11own_verify/owners.json'
TRACES = (REPO / 'runs/prototypes/v29/causal_growth_front'
          / 'lowdens_full_v29_18_2/centerlines.csv')
OUT = REPO / 'runs/prototypes/v30/rev11own_verify/owner_tracks.json'

MATCH_TOL_PX = 12.0     # start-to-attachment at the id frame
JUMP_TOL_PX = 5.0       # per-frame start jump = identity risk
STRIDE = 30             # sampled frames (3 s cadence -> 90 s)


def main() -> int:
    data = json.loads(OWNERS.read_text())
    raw = data['owners'] if 'owners' in data else data
    owners = ({str(o['id']): o for o in raw} if isinstance(raw, list)
              else raw)
    rows = list(csv.DictReader(TRACES.open()))
    by_pid: dict[int, dict[int, list[tuple[float, float, float, str]]]] = {}
    for x in rows:
        pid = int(x['pollen_id'])
        f = int(x['source_frame'])
        by_pid.setdefault(pid, {}).setdefault(f, []).append(
            (float(x['source_x_px']), float(x['source_y_px']),
             float(x['arc_length_px']), str(x['causal_status'])))
    for pid in by_pid:
        for f in by_pid[pid]:
            by_pid[pid][f].sort(key=lambda t: t[2])   # arc order
    frames_of = {pid: sorted(fs) for pid, fs in by_pid.items()}

    out: dict[str, dict] = {}
    for oid, o in owners.items():
        f0 = int(o['identified_at_frame'])
        att = np.asarray(o['attachment_native'], float)
        grain = np.asarray(o['grain_native'], float)
        gdir = att - grain
        gdir = gdir / max(float(np.hypot(*gdir)), 1e-9)
        cands = []
        for pid, fs in frames_of.items():
            f = min(fs, key=lambda x: abs(x - f0))
            if abs(f - f0) > 300:
                continue
            pts = by_pid[pid][f]
            p0 = np.asarray(pts[0][:2])
            d = float(np.hypot(*(p0 - att)))
            # trace initial direction: first segment of >= 12 px arc
            tdir = None
            for q in pts[1:]:
                v = np.asarray(q[:2]) - p0
                if float(np.hypot(*v)) >= 12.0:
                    tdir = v / float(np.hypot(*v))
                    break
            cos = (float(np.dot(tdir, gdir))
                   if tdir is not None else float('nan'))
            cands.append((d, pid, f, cos))
        cands.sort()
        best = cands[0] if cands else None
        rec: dict = {
            'owner': oid, 'grain_native': o['grain_native'],
            'attachment_native': o['attachment_native'],
            'identified_at_frame': f0, 'no_tube': bool(o.get('no_tube')),
        }
        # declared identity tests: start within MATCH_TOL_PX of the
        # human attachment; within 6 px the distance alone is the
        # evidence (tube bases curve, so a short direction window can
        # mislead — see tracks_clump_42000.png, vision-corroborated);
        # the 6-12 px band additionally requires the trace's initial
        # direction to agree (cos >= 0.7). 'confirmed' also needs a
        # separated runner-up.
        if best is None or best[0] > MATCH_TOL_PX:
            rec['matched'] = False
            rec['reason'] = ('no trace start within '
                             f'{MATCH_TOL_PX} px at the id frame'
                             if best else 'no covered frame within 300')
            if best:
                rec['best_pid'] = best[1]
                rec['best_dist_px'] = round(best[0], 2)
                rec['best_cos'] = (round(best[3], 3)
                                   if np.isfinite(best[3]) else None)
            out[oid] = rec
            continue
        d, pid, _f, cos = best
        runner = cands[1][0] if len(cands) > 1 else float('inf')
        dir_ok = np.isfinite(cos) and cos >= 0.7
        near = d <= 6.0
        if not (near or dir_ok):
            rec['matched'] = False
            rec['reason'] = (f'nearest trace pid {pid} at {d:.1f} px but '
                             f'beyond 6 px and initial direction '
                             f'disagrees (cos={cos:.2f})')
            rec['best_pid'] = pid
            rec['best_dist_px'] = round(d, 2)
            rec['best_cos'] = round(cos, 3) if np.isfinite(cos) else None
            out[oid] = rec
            continue
        separated = runner > 2.0 * d
        conf = ('confirmed' if (separated and (near or dir_ok))
                else 'candidate')
        rec['matched'] = True
        rec['pid'] = pid
        rec['match_dist_px'] = round(d, 2)
        rec['match_cos'] = round(cos, 3) if np.isfinite(cos) else None
        rec['runner_up_px'] = round(runner, 2)
        rec['confidence'] = conf
        fs = frames_of[pid]
        track = []
        prev_start = None
        n_jump = 0
        for i, f in enumerate(fs):
            pts = by_pid[pid][f]
            start = np.asarray(pts[0][:2])
            tip = np.asarray(pts[-1][:2])
            length = float(pts[-1][2])
            jump = (float(np.hypot(*(start - prev_start)))
                    if prev_start is not None else 0.0)
            if jump > JUMP_TOL_PX:
                n_jump += 1
            prev_start = start
            if i % STRIDE == 0 or f == fs[0] or f == fs[-1] or \
                    abs(f - f0) < 200:
                track.append({
                    'frame': f,
                    'start_xy': [round(float(start[0]), 2),
                                 round(float(start[1]), 2)],
                    'tip_xy': [round(float(tip[0]), 2),
                               round(float(tip[1]), 2)],
                    'length_px': round(length, 2),
                    'state': pts[-1][3],
                    'start_jump_px': round(jump, 2),
                })
        starts = np.array([by_pid[pid][f][0][:2] for f in fs])
        rec.update({
            'n_frames_covered': len(fs),
            'frame_span': [fs[0], fs[-1]],
            'start_span_px': [round(float(np.ptp(starts[:, 0])), 2),
                              round(float(np.ptp(starts[:, 1])), 2)],
            'start_jumps_over_tol': n_jump,
            'track': track,
        })
        out[oid] = rec

    OUT.write_text(json.dumps({
        'acquisition_cadence_s': 3.0,
        'time_note': 'ld movie cadence is 3.0 s/frame (declared)',
        'calibration': 'native pixels, no resampling',
        'method_note': ('trace matching at the owner identification '
                        'frame: start <= 6 px of the human attachment, '
                        'or 6-12 px with initial-direction agreement '
                        '(cos >= 0.7); confirmed also needs a '
                        'separated runner-up (> 2x)'),
        'owners': out}, indent=1) + '\n')
    n_ok = sum(1 for r in out.values() if r.get('matched'))
    n_cf = sum(1 for r in out.values()
               if r.get('confidence') == 'confirmed')
    print(f'owners: {len(out)} | matched to traces: {n_ok} '
          f'({n_cf} confirmed) | unmatched: {len(out) - n_ok}')
    for oid, r in out.items():
        if r.get('matched'):
            print(f"  {oid}: pid {r['pid']} (d={r['match_dist_px']} px, "
                  f"cos={r['match_cos']}, runner-up "
                  f"{r['runner_up_px']} px) [{r['confidence']}] | "
                  f"frames {r['frame_span'][0]}-{r['frame_span'][1]} "
                  f"({r['n_frames_covered']}) | start span "
                  f"{r['start_span_px']} px | jumps>{JUMP_TOL_PX}: "
                  f"{r['start_jumps_over_tol']}")
        else:
            print(f"  {oid}: UNMATCHED — {r.get('reason')}")
    print(f'-> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
