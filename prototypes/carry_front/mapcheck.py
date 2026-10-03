"""Does the tube map mark the human's tube at the traced bin, and stop at its apex? Per scored FULL trace: the map
(0.8.8's network, tubes_bn_real_ld_m2) maxed over +/- 2 px across the human polyline at that very bin: the share of the
polyline beyond its first 3 px marked (P >= theta), the longest unmarked run, and the share marked 3-12 px past the
apex along the polyline's last direction (the tube's tip blur, or a neighbour). Independent of any route carrying.

    python -m prototypes.carry_front.mapcheck [theta]
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from prototypes.carry_front.carry import OUT, Movie, dense, extend
from sparsetrack.routes import normals


def kline(mv: Movie, b: int, pts: np.ndarray, w: float = 2.0) -> np.ndarray:
    x0, y0 = int(np.floor(pts[:, 0].min())) - 8, int(np.floor(pts[:, 1].min())) - 8
    x1, y1 = int(np.ceil(pts[:, 0].max())) + 9, int(np.ceil(pts[:, 1].max())) + 9
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(mv.W, x1), min(mv.H, y1)
    img = np.zeros((y1 - y0, x1 - x0), np.float32)
    bb = min(max(b, mv.rs), mv.nb - 1)
    img[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(mv.P[bb, sy0:sy1, sx0:sx1], np.float32) / 250.0
    n = normals(pts)
    offs = np.arange(-w, w + 1e-9, 0.5)
    # P pixel j covers reference [j, j + 1): its centre is j + 0.5
    qx = (pts[:, 0][None] + offs[:, None] * n[:, 0][None] - x0 - 0.5).astype(np.float32)
    qy = (pts[:, 1][None] + offs[:, None] * n[:, 1][None] - y0 - 0.5).astype(np.float32)
    return cv2.remap(img, qx, qy, cv2.INTER_LINEAR, borderValue=0).max(axis=0)


def main():
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    out = {}
    for movie in ("ld", "m2", "m1"):
        mv = Movie(movie)
        rows = []
        for gid in mv.scored_grains():
            for b, t in mv.full_traces(gid):
                if t.get("contact"):
                    continue
                h = float(t["length_px"])
                pts = dense(t["path_xy_ref"])
                ext = extend(pts, 12.0)[len(pts):]
                k = kline(mv, b, pts)
                body = k[3:] if len(k) > 4 else k
                on = body >= theta
                runs, cur = [0], 0
                for v in on:
                    cur = 0 if v else cur + 1
                    runs.append(cur)
                kb = kline(mv, b, np.vstack([pts[-2:], ext]))[2:] if len(ext) else np.zeros(0)
                beyond = float(np.mean(kb[3:] >= theta)) if len(kb) > 3 else float("nan")
                rows.append({"grain": gid, "bin": b, "h": h, "marked": float(on.mean()) if len(on) else 0.0,
                             "gap": int(max(runs)), "beyond": beyond})
        out[movie] = rows
        m = np.array([r["marked"] for r in rows])
        gp = np.array([r["gap"] for r in rows])
        bd = np.array([r["beyond"] for r in rows])
        print(f"{movie}: {len(rows)} traces; marked share median {np.median(m):.2f}, >= 0.9 {np.mean(m >= 0.9):.0%}; "
              f"longest gap > 5 px {np.mean(gp > 5):.0%}; past apex marked >= 0.5 {np.nanmean(bd >= 0.5):.0%}")
    (OUT / f"mapcheck_{theta:g}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
