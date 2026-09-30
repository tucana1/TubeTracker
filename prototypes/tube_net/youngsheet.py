"""Maps round a young trace over time: rows = networks, columns = bins; the human trace of that bin in green, the
grain's census circle (r) and the flood's blocked halo (r + 3 px) and start band edge (r + 7 px) as thin rings.

    python -m prototypes.tube_net.youngsheet OUT.png MOVIE GRAIN BIN --models A.pt[:bg] B.pt[:bg] [--span 6 --step 2]
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import learned, stack

from .pixels import MOVIES, REPO


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("movie", choices=tuple(MOVIES))
    ap.add_argument("grain")
    ap.add_argument("bin", type=int)
    ap.add_argument("--models", nargs="+", required=True, help="MODEL.pt[:BG_PX]")
    ap.add_argument("--span", type=int, default=6)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--half", type=int, default=28)
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args(argv)
    cache, lab = (REPO / p for p in MOVIES[a.movie])
    src, meta = stack.load(cache)
    L = json.loads(lab.read_text())
    g = L["grains"][a.grain]
    tr = L["labels"][a.grain]["traces"].get(str(a.bin))
    off = (tr or {}).get("view_offset") or [0.0, 0.0]
    gx, gy = g["x"] + off[0], g["y"] + off[1]
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([learned._registered(src, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([learned._registered(src, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    bins = list(range(a.bin - a.span, a.bin + a.span + 1, a.step))
    h, s = a.half, a.scale
    x0, y0 = int(round(gx)) - h, int(round(gy)) - h
    sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))

    def ring(img, rad, col):
        cv2.circle(img, (int(round((gx - x0) * s)), int(round((gy - y0) * s))), int(round(rad * s)), col, 1)

    def panel(v, lo, hi, b, mark=True):
        u8 = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        p = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        if mark:
            for rad, col in ((g["r"], (255, 120, 0)), (g["r"] + 3, (0, 120, 255)), (g["r"] + 7, (0, 200, 255))):
                ring(p, rad, col)
            if b == a.bin and tr and tr.get("path_xy_ref"):
                q = (np.asarray(tr["path_xy_ref"]) - [x0, y0]) * s
                cv2.polylines(p, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 0), 1)
        cv2.putText(p, str(b), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        return p

    rows = []
    imgs = {b: learned._registered(src, shifts, b) for b in bins}
    m = float(np.median(early[sl]))
    rows.append(np.hstack([panel(imgs[b][sl], m - 50, m + 30, b, False) for b in bins]))
    rows.append(np.hstack([panel((imgs[b] - early)[sl], -30, 30, b) for b in bins]))
    for spec in a.models:
        path, _, bg = spec.partition(":")
        net = learned.load_model(path)
        net.bg_px = int(bg) if bg else net.bg_px
        rows.append(np.hstack([panel(learned.tube_probability(net, imgs[b], early, late)[sl], 0, 1, b) for b in bins]))
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"wrote {a.out}; rows: bin, change, " + ", ".join(Path(m.split(':')[0]).stem for m in a.models))


if __name__ == "__main__":
    main()
