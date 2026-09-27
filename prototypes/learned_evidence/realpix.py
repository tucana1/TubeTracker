"""Pixel-level check of a tube-probability cache against the human traces.

    python -m prototypes.learned_evidence.realpix PROB_CACHE LABELS

Along every FULL trace (at its bin): the share of centreline points the network calls tube
(max P over +/-2 px across the path >= 0.5), and the share of points 8-14 px to either side
it calls tube (mostly background next to a tube; other tubes crossing make this an upper bound).
Traces flagged as touching another tube or grain are skipped.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer

from .evaluate import SCALE_P


def trace_points(path_xy: list, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(path_xy, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1] + 1e-9, step)
    pts = np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], 1)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    return pts, np.stack([-tang[:, 1], tang[:, 0]], 1)


def check(prob_cache: str, labels_path: str, skip_px: float = 4.0) -> dict:
    bins, meta = stack.load(prob_cache)
    R = Renderer(bins, meta)
    L = json.load(open(labels_path))
    on_hit = on_n = off_hit = off_n = 0
    per_grain = {}
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        if g.get("excluded"):
            continue
        for b, t in lab.get("traces", {}).items():
            if t["state"] != "full" or t.get("contact") or len(t.get("path_xy_ref") or []) < 2:
                continue
            pts, nrm = trace_points(t["path_xy_ref"])
            keep = np.hypot(pts[:, 0] - g["x"], pts[:, 1] - g["y"]) >= g["r"] + skip_px
            pts, nrm = pts[keep], nrm[keep]
            if len(pts) < 3:
                continue
            half = int(np.ceil(np.max(np.abs(pts - [g["x"], g["y"]])))) + 20
            img = R.crop(int(b), g["x"], g["y"], half) / SCALE_P
            q = pts - [g["x"] - half, g["y"] - half] - 0.5  # crop pixel coordinates

            def samp(o):
                qq = (q + o * nrm).astype(np.float32)
                return cv2.remap(np.nan_to_num(img).astype(np.float32), qq[:, 0][None], qq[:, 1][None],
                                 cv2.INTER_LINEAR)[0]
            on = np.max([samp(o) for o in (-2, -1, 0, 1, 2)], axis=0) >= 0.5
            off = np.concatenate([np.max([samp(s * o) for o in (8, 11, 14)], axis=0) >= 0.5 for s in (-1, 1)])
            on_hit += int(on.sum()); on_n += len(on); off_hit += int(off.sum()); off_n += len(off)
            pg = per_grain.setdefault(gid, [0, 0])
            pg[0] += int(on.sum()); pg[1] += len(on)
    return {"on": on_hit / max(on_n, 1), "off": off_hit / max(off_n, 1), "points": on_n,
            "worst": sorted(((round(h / n, 2), gid) for gid, (h, n) in per_grain.items() if n), key=lambda x: x[0])[:6]}


if __name__ == "__main__":
    r = check(sys.argv[1], sys.argv[2])
    print(f"on the traced tube {100 * r['on']:.1f}% of {r['points']} points | beside it {100 * r['off']:.1f}% | "
          f"worst grains {r['worst']}")
