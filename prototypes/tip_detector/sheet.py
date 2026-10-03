"""A contact sheet of traces from an evaluate.py dump: appearance, short-interval difference and the detector's map,
with the human apex (green circle), the detector's top 3 peaks (red, orange, yellow) and the |D| top peak (cyan).

    python -m prototypes.tip_detector.sheet runs/research/tip_detector/eval_m1_ldm2.json OUT.png [--fail] [--n 8]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .common import Movie, heat, load_net

COLS = [(0, 0, 255), (0, 140, 255), (0, 230, 230)]


def grey(v, lo=None, hi=None):
    lo, hi = (np.percentile(v, [1, 99]) if lo is None else (lo, hi))
    return cv2.cvtColor(np.clip((v - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--fail", action="store_true", help="only traces the detector misses (top-1 > 4 px)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--view", type=int, default=72, help="half-size of the shown window round the apex")
    ap.add_argument("--cls", default=None, help="only this length class (young, mid, long)")
    ap.add_argument("--pick", nargs="*", default=None, help="grain:bin entries to show")
    a = ap.parse_args()
    res = json.loads(open(a.dump).read())
    mv = Movie(res["movie"])
    net = load_net(res["net"])
    chans = ["ADC".index(c) for c in net.channels]
    tr = [t for t in res["traces"] if (not a.fail or t["methods"]["det"][0][3] > 4) and (a.cls in (None, t["cls"]))]
    if a.pick:
        tr = [t for t in tr if f"{t['grain']}:{t['bin']}" in a.pick]
    rng = np.random.default_rng(0)
    pick = [tr[i] for i in sorted(rng.choice(len(tr), min(a.n, len(tr)), replace=False))]
    rows = []
    for t in pick:
        gid, b, half = t["grain"], t["bin"], t["half"]
        tt = mv.labels[gid]["traces"][str(b)]
        gx, gy, r = mv.grain_at(gid, b, tt)
        raw = mv.raw(b, gx, gy, half)
        hm = heat(net, (raw / mv.scale[:, None, None])[chans].astype(np.float32))[0]
        ax, ay = t["apex"]
        v = min(a.view, half)
        cx, cy = int(np.clip(round(ax), v, 2 * half - v)), int(np.clip(round(ay), v, 2 * half - v))
        sl = (slice(cy - v, cy + v), slice(cx - v, cx + v))
        tiles = [grey(raw[0][sl]), grey(raw[1][sl], -12, 12), grey(hm[sl], 0, 1)]
        out = []
        for tile in tiles:
            tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            P = lambda x, y: (int(round(2 * (x - cx + v) + 1)), int(round(2 * (y - cy + v) + 1)))
            cv2.circle(tile, P(ax, ay), 8, (0, 200, 0), 1)
            for k, (x, y, val, d) in enumerate(t["methods"]["det"][:3]):
                cv2.drawMarker(tile, P(x, y), COLS[k], cv2.MARKER_CROSS, 9, 1)
            x, y, *_ = t["methods"]["absD_0"][0]
            cv2.drawMarker(tile, P(x, y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 9, 1)
            out.append(cv2.resize(tile, (4 * a.view, 4 * a.view), interpolation=cv2.INTER_NEAREST))
        row = np.concatenate(out, 1)
        d = t["methods"]["det"][0][3]
        cv2.putText(row, f"{gid} b{b} L{t['L']:.0f} det {d:.0f}px |D| {t['methods']['absD_0'][0][3]:.0f}px",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        rows.append(row)
    cv2.imwrite(a.out, np.concatenate(rows, 0))
    print(a.out, len(rows))


if __name__ == "__main__":
    main()
