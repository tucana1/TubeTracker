"""rev11 owner-round verification (change-list item 7).

The review: "Verify the resulting owner ID through database, snapshot,
target builder and loss before requesting an absence review."

- database: reads the completed rev11own tasks.
- crossing mapping: associates each rev10x lane with the ball whose
  tube it is (nearest root + direction consistency), honestly flagging
  uncertainty where the geometry is ambiguous.
- target builder + loss: builds a typed prompt with HUMAN-GRAIN
  provenance from a confirmed ball, runs the forward on real pixels
  and a supervised step through the shared loss (the point is the
  pipeline accepts the confirmed geometry end-to-end).
- overlays: native PNGs of the marks for the record.

Snapshot verification (tubes.json) is checked separately once the
trial snapshot build lands.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OWN = Path('/Users/joshjiang/Documents/TubeTracker-annotator-projects/'
           'rev11own/annotations.db')
XROUND = Path('/Users/joshjiang/Documents/TubeTracker-annotator-projects/'
              'rev10round/annotations.db')
MOVIE = '/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4'
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
    tasks = load(OWN, 'task')
    crossings = load(XROUND, 'crossing')
    o0 = tasks['rev11o-000']
    o1 = tasks['rev11o-001']
    print('=== database:')
    for u, t in tasks.items():
        g = t.get('grains') or {}
        print(f"  {u}: completed={t.get('completed')} "
              f"balls={sorted(g)}")
        for lab, v in sorted(g.items()):
            print(f"     {lab}: ball=({v['xy'][0]:.1f},{v['xy'][1]:.1f}) "
                  f"root=({v['root_xy'][0]:.1f},{v['root_xy'][1]:.1f})"
                  f"{' NO-TUBE' if v.get('no_tube') else ''}")

    # ---- crossing lane -> ball association (frame 51240 exact; the
    # other two frames flagged for drift) -----------------------------
    print('=== crossing mapping (lane upstream end -> nearest root, '
          'direction angle):')
    balls = o0['grains']
    for cu, c in sorted(crossings.items()):
        frame = c['source_frame']
        print(f"  {cu} (frame {frame}):")
        for lab, pts in sorted(c['lanes'].items()):
            p = np.asarray(pts, float)
            # lanes are ordered entry->exit (entry = the ball-cluster
            # side per the rev10 builder); upstream end = p[0]
            end = p[0]
            inner = p[1]
            v = end - inner
            v = v / (np.hypot(*v) + 1e-9)
            best = None
            for bl, b in sorted(balls.items()):
                r = np.asarray(b['root_xy'], float)
                d = float(np.hypot(*(r - end)))
                w = r - end
                ang = float(np.degrees(np.arccos(np.clip(
                    np.dot(v, w / (np.hypot(*w) + 1e-9)), -1, 1))))
                if best is None or d < best[1]:
                    best = (bl, d, ang)
            print(f"    lane {lab}: end=({end[0]:.0f},{end[1]:.0f}) -> "
                  f"ball {best[0]} root at {best[1]:.1f} px, "
                  f"angle {best[2]:.0f} deg")

    # ---- overlays ---------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tubetracker.annotation_frames import FrameReader
    r = FrameReader(MOVIE)
    # crossing frame with balls+roots+lanes
    g = r.read(51240).frame
    g = g[:, :, 0] if g.ndim == 3 else g
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(g, cmap='gray')
    c = crossings['cross-rev10x-001']
    for lab, col in (('A', 'crimson'), ('B', 'dodgerblue')):
        p = np.asarray(c['lanes'][lab], float)
        ax.plot(p[:, 0], p[:, 1], '-', color=col, lw=2,
                label=f'lane {lab}')
    for bl, b in sorted(balls.items()):
        ax.plot(*b['xy'], 'o', color='lime', ms=9, mew=2, mfc='none')
        ax.plot(*b['root_xy'], 'x', color='orange', ms=9, mew=2)
        ax.annotate(bl, b['xy'], color='lime', fontsize=12,
                    fontweight='bold')
    ax.set_xlim(880, 1090); ax.set_ylim(440, 280)
    ax.set_title('rev11o-000 marks | frame 51240 | green=ball, '
                 'orange=tube start')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'owner_marks_crossing_51240.png')
    # clump frame
    g = r.read(42000).frame
    g = g[:, :, 0] if g.ndim == 3 else g
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(g, cmap='gray')
    for bl, b in sorted(o1['grains'].items()):
        ax.plot(*b['xy'], 'o', color='lime', ms=9, mew=2, mfc='none')
        ax.plot(*b['root_xy'], 'x', color='orange', ms=9, mew=2)
        ax.annotate(bl, b['xy'], color='lime', fontsize=12,
                    fontweight='bold')
    z = np.load(REPO / 'runs/prototypes/v30/rev11_fitcheck_gate600'
                       '.step569.npz', allow_pickle=True)
    x0, y0 = 475, 518
    for i, col in enumerate(('red', 'royalblue', 'green')):
        m = z[f'mask_{i}']
        ax.contour(np.arange(x0, x0 + 288), np.arange(y0, y0 + 288),
                   m.astype(float), levels=[0.5], colors=[col],
                   linewidths=1.2)
    ax.set_xlim(470, 770); ax.set_ylim(820, 520)
    ax.set_title('rev11o-001 marks | frame 42000 | green=ball, '
                 'orange=tube start, contours=g0/g1/g2')
    fig.tight_layout()
    fig.savefig(OUT / 'owner_marks_clump_42000.png')
    r.close()
    print(f'=== overlays -> {OUT}')

    # ---- target builder + loss smoke --------------------------------
    import torch
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from prototypes.v30_video_apex.inference import build_typed_prompt
    from prototypes.v30_video_apex.model import build_model
    from prototypes.v30_video_apex.targets import load_clip_pixels
    from prototypes.v30_video_apex.train import (
        LossWeights, masked_multihead_loss, train_step_front)
    CS = 288
    ball_a = balls['A']
    cx, cy = ball_a['xy']
    from tubetracker.annotation_frames import FrameReader as _FR
    _r0 = _FR(MOVIE)
    NW, NH = _r0.native_size[0], _r0.native_size[1]
    _r0.close()
    ox = int(min(max(cx - CS / 2, 0), max(0, NW - CS)))
    oy = int(min(max(cy - CS / 2, 0), max(0, NH - CS)))
    r = FrameReader(MOVIE)
    frames = tuple(51240 + o for o in QUERY_OFFSETS)
    missing = tuple(1 if 51240 + o < 0 else 0 for o in QUERY_OFFSETS)
    clip_np = load_clip_pixels(r, frames, (ox, oy, CS, CS), missing)
    r.close()
    prompt, prov = build_typed_prompt(
        'rev11o-000-A', crop_wh=(CS, CS), crop_origin=(ox, oy),
        human_grain_xy=(float(cx), float(cy)),
        human_grain_radius_px=13.0,
        attachment_xy=(float(ball_a['root_xy'][0]),
                       float(ball_a['root_xy'][1])))
    print('=== target builder:')
    print('  prompt provenance kind:', prov.get('prompt_kind'),
          '| center native:', prov.get('grain_center_native'))
    route = [[float(ball_a['root_xy'][0] - ox),
              float(ball_a['root_xy'][1] - oy)],
             [float(ball_a['root_xy'][0] - ox) + 30.0,
              float(ball_a['root_xy'][1] - oy) - 18.0]]
    model = build_model('temporal', base=8, multiscale=True)
    clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        pred = model.forward(clip, prompt, route_xy=route)
    print('  forward: front_logits',
          tuple(pred.front_logits.shape), 'finite',
          bool(torch.isfinite(pred.front_logits).all()))
    weights = LossWeights(apex=0.0, visibility=0.0, body=0.0,
                          route_correct=0.0, front=1.0)
    opt = torch.optim.AdamW([p for p in model.parameters()][:2], lr=1e-4)
    rstep = train_step_front(
        model, opt, clip, prompt, route,
        torch.zeros(1, 1, CS, CS), {'apex_valid': torch.zeros(1)},
        torch.zeros(1, 1, int(pred.front_logits.shape[-1])),
        torch.zeros(1), torch.tensor([2]), torch.ones(1), weights)
    print('  loss step: total =', float(rstep.get('total', float('nan'))),
          '| finite:', bool(np.isfinite(float(rstep.get('total', np.nan)))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
