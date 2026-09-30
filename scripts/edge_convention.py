"""A labels copy with every trace measured from where the traced tube leaves its grain's edge.

    python scripts/edge_convention.py benchmark/labels/m1_v1.json OUT.json [--inside 0.6]

Some tubes emerge from a pore facing the camera: the annotator's trace then starts in the middle of the grain and runs
over it before crossing its edge. SparseTrack (and the other movies' traces) measure length from the grain's visible
edge. Traces whose first point lies within ``--inside`` of the grain's radius from its centre are cut where they first
leave the radius (the grain's place when traced: census + the view's offset) and their length is recomputed; all
other traces and answers are unchanged. Used for movie 1's secondary score (rule fixed before any score existed).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def cut_at_edge(path: list, cx: float, cy: float, r: float) -> list:
    """The polyline from where it first reaches radius ``r`` of (cx, cy) outwards."""
    p = np.asarray(path, float)
    rad = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    out = np.nonzero(rad >= r)[0]
    if not len(out):
        return [list(map(float, p[-1]))]
    k = int(out[0])
    if k == 0:
        return p.tolist()
    a, b = p[k - 1], p[k]
    ra, rb = rad[k - 1], rad[k]
    f = (r - ra) / max(rb - ra, 1e-9)
    edge = a + f * (b - a)
    return [list(map(float, edge))] + p[k:].tolist()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--inside", type=float, default=0.6)
    a = ap.parse_args(argv)
    d = json.loads(Path(a.labels).read_text())
    changed = []
    for gid, lab in d["labels"].items():
        g = d["grains"][gid]
        for b, t in (lab.get("traces") or {}).items():
            if t["state"] not in ("full", "partial") or len(t.get("path_xy_ref") or []) < 2:
                continue
            ox, oy = t.get("view_offset") or [0.0, 0.0]
            cx, cy, r = g["x"] + ox, g["y"] + oy, g["r"]
            x0, y0 = t["path_xy_ref"][0]
            if math.hypot(x0 - cx, y0 - cy) >= a.inside * r:
                continue
            ref = cut_at_edge(t["path_xy_ref"], cx, cy, r)
            shift = np.asarray(ref[0]) - np.asarray(t["path_xy_ref"][0])  # keep the other coordinate copies aligned
            old = t["length_px"]
            t["path_xy_ref"] = [[round(x, 2), round(y, 2)] for x, y in ref]
            for key in ("path_xy", "path_xy_view"):
                if t.get(key):
                    off = np.asarray(t[key][0]) - np.asarray(t["path_xy_ref"][0]) + shift  # same offset as before
                    t[key] = [[round(x + off[0], 2), round(y + off[1], 2)] for x, y in ref]
            t["length_px"] = round(float(np.sum(np.hypot(*np.diff(np.asarray(ref), axis=0).T))) if len(ref) > 1 else 0.0, 2)
            t["edge_convention"] = {"length_as_traced": old}
            changed.append(f"{gid}@{b}: {old} -> {t['length_px']} px")
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(f"{len(changed)} traces measured from the grain edge:", *changed, sep="\n  ")


if __name__ == "__main__":
    main()
