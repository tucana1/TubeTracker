"""Render frame 42000 (fit clump) with v29 trace starts and the
persistent-owner marks, to eyeball which trace belongs to which owner.

Output: runs/prototypes/v30/rev11own_verify/tracks_clump_42000.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FRAME = 42000
CROP = (470, 560, 320, 220)   # x, y, w, h
TRACES = (REPO / 'runs/prototypes/v29/causal_growth_front'
          / 'lowdens_full_v29_18_2/centerlines.csv')
OUT = REPO / 'runs/prototypes/v30/rev11own_verify/tracks_clump_42000.png'


def main() -> int:
    from tubetracker.annotation_frames import FrameReader
    movie = '/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4'
    r = FrameReader(movie)
    img = r.read(FRAME).frame
    r.close()
    x, y, w, h = CROP
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=130)
    ax.imshow(img[y:y + h, x:x + w], cmap='gray',
              extent=(x, x + w, y + h, y))
    rows = list(csv.DictReader(TRACES.open()))
    by_pid: dict[int, list] = {}
    for row in rows:
        if int(row['source_frame']) == FRAME:
            by_pid.setdefault(int(row['pollen_id']), []).append(
                (float(row['arc_length_px']),
                 float(row['source_x_px']), float(row['source_y_px'])))
    cmap = plt.get_cmap('tab10')
    for i, (pid, pts) in enumerate(sorted(by_pid.items())):
        pts.sort()
        arr = np.array([(p[1], p[2]) for p in pts])
        if not (x - 40 <= arr[:, 0].min() and arr[:, 0].max() <= x + w + 40
                and y - 40 <= arr[:, 1].min() and arr[:, 1].max() <= y + h + 40):
            continue
        ax.plot(arr[:, 0], arr[:, 1], '-', color=cmap(i % 10), lw=2.0,
                label=f'pid {pid}')
        ax.plot(arr[0, 0], arr[0, 1], 'o', color=cmap(i % 10), ms=7)
        ax.annotate(f'pid {pid}', (arr[0, 0], arr[0, 1]),
                    textcoords='offset points', xytext=(6, -10),
                    color=cmap(i % 10), fontsize=9, weight='bold')
    owners = json.loads((REPO / 'runs/prototypes/v30/rev11own_verify'
                         '/owners.json').read_text())['owners']
    for o in owners:
        if o['identified_at_frame'] != 42000:
            continue
        gx, gy = o['grain_native']
        axx, ayy = o['attachment_native']
        ax.plot(gx, gy, 'x', color='red', ms=13, mew=2.5)
        ax.plot(axx, ayy, '+', color='magenta', ms=13, mew=2.5)
        ax.annotate(o['id'].replace('own-ld-', ''), (gx, gy),
                    textcoords='offset points', xytext=(8, 6),
                    color='red', fontsize=10, weight='bold')
    ax.plot([], [], 'x', color='red', label='owner grain')
    ax.plot([], [], '+', color='magenta', label='owner attachment')
    ax.set_xlim(x, x + w)
    ax.set_ylim(y + h, y)
    ax.set_title(f'frame {FRAME}: v29 trace starts (o) + owner marks')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT)
    print('->', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
