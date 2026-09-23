"""Build persistent owner records from the owner-selection round (item 6).

The review: "Create persistent physical-grain IDs and link the same
grain across frames; local task IDs and A/B lane names are not those
IDs." This tool turns the user-identified balls (rev11own, verified in
H401) into persistent owner records and links the recorded structures
to them:

- owners.json: one record per user-identified ball — persistent id,
  grain center, attachment (tube start), source task/label, and the
  identification provenance (user-identified, owner-selection round).
- crossing_owner_links.json: each rev10x lane end -> owner, with the
  measured distance/angle evidence and an honest confidence
  (clean / likely / ambiguous); ambiguous ends stay UNRESOLVED rather
  than guessed. Lane labels are explicitly NOT identities.
- fit_owner_links.json: the fit-gate masks g0/g1/g2 -> nearest owner
  (or unresolved, with the reason).

Persistent ids are `own-<movie>-<nnnn>` — stable across tasks and
frames, never task-local letters.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OWN_DB = Path('/Users/joshjiang/Documents/TubeTracker-annotator-projects/'
              'rev11own/annotations.db')
X_DB = Path('/Users/joshjiang/Documents/TubeTracker-annotator-projects/'
            'rev10round/annotations.db')
OUT = REPO / 'runs/prototypes/v30/rev11own_verify'


def load(db: Path, kind: str) -> dict:
    con = sqlite3.connect(db)
    try:
        return {u: json.loads(d) for u, d in con.execute(
            "select uuid, data from entities where kind=?", (kind,))}
    finally:
        con.close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load(OWN_DB, 'task')
    crossings = load(X_DB, 'crossing')

    # ---- owners ------------------------------------------------------
    owners: list[dict] = []
    by_task: dict[str, dict[str, str]] = {}
    seq = 0
    for tu in ('rev11o-000', 'rev11o-001'):
        t = tasks[tu]
        by_task[tu] = {}
        for lab, g in sorted((t.get('grains') or {}).items()):
            seq += 1
            oid = f'own-ld-{seq:04d}'
            by_task[tu][lab] = oid
            owners.append({
                'id': oid,
                'movie': str(t.get('movie', 'ld')),
                'grain_native': [round(float(g['xy'][0]), 2),
                                 round(float(g['xy'][1]), 2)],
                'attachment_native': [round(float(g['root_xy'][0]), 2),
                                      round(float(g['root_xy'][1]), 2)]
                if g.get('root_xy') else None,
                'no_tube': bool(g.get('no_tube')),
                'identified_at_frame': int(t['query_frames'][0]),
                'source_task': tu,
                'source_label': lab,
                'provenance': ('user-identified in the rev11 owner-selection '
                               'round (rev11own rev 17/18); not a detector '
                               'or machine guess'),
                'observations': [],
            })
    print(f'=== owners: {len(owners)}')
    for o in owners:
        root = o['attachment_native']
        root_label = f'({root[0]:.0f},{root[1]:.0f})' if root else 'unverified'
        print(f"  {o['id']}  ball=({o['grain_native'][0]:.0f},"
              f"{o['grain_native'][1]:.0f})  root={root_label}  <- {o['source_task']}"
              f":{o['source_label']}")

    # ---- crossing lane links ----------------------------------------
    # NOTE: lane upstream ends pile up near one spot because the traces
    # START near the crossing, not at each tube's attachment (measured:
    # all six ends within 1-10 px of one root). Endpoint distance is
    # therefore NOT identity evidence. The physical test: extend the
    # lane's own direction (ray from its end) and ask which root lies
    # closest to that ray (perpendicular distance), forward along it.
    links: list[dict] = []
    clump = tasks['rev11o-000']['grains']
    roots = {lab: np.asarray(g['root_xy'], float)
             for lab, g in clump.items() if g.get('root_xy')}
    print('=== crossing lane links (ray test):')
    for cu, c in sorted(crossings.items()):
        frame = int(c['source_frame'])
        for lab, pts in sorted(c['lanes'].items()):
            p = np.asarray(pts, float)
            end, inner = p[0], p[1]          # entry = ball-cluster side
            v = end - inner
            v = v / (np.hypot(*v) + 1e-9)
            scored = []
            for bl, r in roots.items():
                w = r - end
                fwd = float(np.dot(w, v))
                perp = float(np.hypot(*(w - fwd * v)))
                scored.append((bl, perp, fwd))
            scored.sort(key=lambda q: q[1])
            if not scored:
                links.append({'crossing': cu, 'frame': frame, 'lane': lab,
                    'owner': None, 'best_candidate': None, 'confidence': 'ambiguous',
                    'note': 'No reviewed roots remain; lane ownership is unresolved.'})
                continue
            bl, perp, fwd = scored[0]
            perp2 = scored[1][1] if len(scored) > 1 else 1e9
            if len(roots) < len(clump):
                conf = 'ambiguous'
            elif fwd < -2.0 or fwd > 60.0:
                conf = 'ambiguous'
            elif perp <= 3.0 and perp2 >= max(2 * perp + 2.0, 6.0):
                conf = 'clean'
            elif perp <= 8.0 and fwd <= 40.0:
                conf = 'likely'
            else:
                conf = 'ambiguous'
            links.append({
                'crossing': cu, 'frame': frame, 'lane': lab,
                'owner': by_task['rev11o-000'][bl] if conf != 'ambiguous'
                else None,
                'best_candidate': by_task['rev11o-000'][bl],
                'perp_px': round(perp, 1),
                'perp_second_px': round(perp2, 1),
                'forward_px': round(fwd, 1),
                'confidence': conf,
                'note': ('lane labels are local names, not identities '
                         '(review item 6); this ray test is the identity '
                         'evidence — the root nearest the lane\'s own '
                         'direction'),
            })
            print(f"  {cu} lane {lab}: -> {links[-1]['owner'] or 'UNRESOLVED'}"
                  f" (best {by_task['rev11o-000'][bl]} perp {perp:.1f}px, "
                  f"fwd {fwd:.0f}px, second {perp2:.1f}px, {conf})")

    # ---- fit-gate mask links ----------------------------------------
    z = np.load(REPO / 'runs/prototypes/v30/rev11_fitcheck_gate600'
                       '.step569.npz', allow_pickle=True)
    x0, y0 = 475, 518
    fit = tasks['rev11o-001']['grains']
    fit_links: list[dict] = []
    print('=== fit-gate mask links:')
    for i, mname in enumerate(('rev8p-42000-g0', 'rev8p-42000-g1',
                               'rev8p-42000-g2')):
        m = z[f'mask_{i}']
        yy, xx = np.nonzero(m)
        cen = np.array([x0 + xx.mean(), y0 + yy.mean()]) if xx.size \
            else np.array([np.nan, np.nan])
        # pixel-min distances (centroids mislead on elongated masks):
        # the closest mask pixel to each ball and to each attachment
        px = np.stack([x0 + xx, y0 + yy], axis=1).astype(float) \
            if xx.size else np.zeros((0, 2))
        best = None
        for lab, g in fit.items():
            d_ball = float(np.min(np.hypot(
                px[:, 0] - g['xy'][0], px[:, 1] - g['xy'][1]))) \
                if px.size else float('nan')
            d_att = float(np.min(np.hypot(
                px[:, 0] - g['root_xy'][0],
                px[:, 1] - g['root_xy'][1]))) if px.size else float('nan')
            score = min(d_ball, d_att)
            if best is None or score < best[2]:
                best = (lab, d_ball, d_att, score)
        bl, d_ball, d_att, _s = best
        # a mask whose pixels TOUCH a ball or its attachment is that
        # owner's structure; otherwise ownership stays unresolved
        resolved = min(d_ball, d_att) <= 6.0
        fit_links.append({
            'mask': mname,
            'owner': by_task['rev11o-001'][bl] if resolved else None,
            'nearest_ball': by_task['rev11o-001'][bl],
            'min_dist_to_ball_px': round(d_ball, 1),
            'min_dist_to_attachment_px': round(d_att, 1),
            'resolved': resolved,
            'note': ('mask pixels reach this owner' if resolved else
                     'mask pixels do not touch any ball or attachment '
                     '(tube/neck region) — ownership stays UNRESOLVED'),
        })
        print(f"  {mname}: centroid=({cen[0]:.0f},{cen[1]:.0f}) -> "
              f"{fit_links[-1]['owner'] or 'UNRESOLVED'} "
              f"(min-to-ball {d_ball:.1f}px / min-to-attach {d_att:.1f}px)")

    touched = {fl['nearest_ball'] for fl in fit_links if fl['resolved']}
    unmasked = [by_task['rev11o-001'][lab] for lab in sorted(fit)
                if by_task['rev11o-001'][lab] not in touched]
    print(f'  owners with NO fit-gate mask: '
          f"{unmasked or 'none'}")

    (OUT / 'owners.json').write_text(json.dumps(
        {'movie': 'ld', 'n_owners': len(owners), 'owners': owners,
         'note': 'persistent ids; source letters are provenance only'},
        indent=1))
    (OUT / 'crossing_owner_links.json').write_text(json.dumps(
        links, indent=1))
    (OUT / 'fit_owner_links.json').write_text(json.dumps(
        {'links': fit_links, 'owners_without_a_mask': unmasked,
         'note': 'mask ownership by pixel contact (min distance to ball '
                 'or attachment); contact <= 6 px counts as touching. '
                 'Owners without a mask are the unmodeled ones.'},
        indent=1))
    print(f'=== wrote owners.json, crossing_owner_links.json, '
          f'fit_owner_links.json -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
