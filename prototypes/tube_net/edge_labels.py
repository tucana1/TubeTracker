"""Movie 1's secondary scoring (fixed 30 Sep 2026 before any movie-1 score; docs/status-2026-09-29.md): traces that
start inside the grain - a tube from a pore facing the camera, lying over its grain before it crosses the edge - have
their lengths measured from where the traced tube crosses the grain's edge, the convention SparseTrack and the other
movies use.

    python -m prototypes.tube_net.edge_labels [--frac 0.6] [--out runs/tube_net/m1_v1_edge.json]

A trace (FULL or PARTIAL) "starts inside" when its first point lies within ``frac`` of the grain's radius from the
grain's centre at that bin (census place + the trace's view offset): 6 of movie 1's 86. Its path is cut where it
first leaves the census circle (radius r) and its length recomputed; a trace that never leaves the circle becomes
0 px long (the tube has not crossed the edge yet). Everything else is copied unchanged. The derived file is for
scoring only (``bench.py --labels m1=...``); benchmark/labels/m1_v1.json stays the annotator's.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def cut_at_edge(path: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """The polyline from where it first crosses the circle outwards (interpolated), or just its last point."""
    d = np.hypot(path[:, 0] - cx, path[:, 1] - cy)
    out = np.flatnonzero(d >= r)
    if not len(out):
        return path[-1:]
    i = int(out[0])
    if i == 0:
        return path
    a, b = path[i - 1], path[i]
    # solve |a + t (b - a) - c| = r for t in [0, 1]
    u, c = b - a, np.array([cx, cy])
    A, B, C = float(u @ u), float(2 * u @ (a - c)), float((a - c) @ (a - c) - r * r)
    t = (-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A) if A > 0 else 1.0
    return np.vstack([a + np.clip(t, 0.0, 1.0) * u, path[i:]])


def derive(labels: dict, frac: float = 0.6) -> tuple[dict, list]:
    out = copy.deepcopy(labels)
    changed = []
    for gid, lab in out["labels"].items():
        g = out["grains"][gid]
        for b, t in (lab.get("traces") or {}).items():
            p = np.asarray(t.get("path_xy_ref") or [], float)
            if t["state"] not in ("full", "partial") or len(p) < 2:
                continue
            off = t.get("view_offset") or [0.0, 0.0]
            cx, cy = g["x"] + off[0], g["y"] + off[1]
            if np.hypot(p[0, 0] - cx, p[0, 1] - cy) >= frac * g["r"]:
                continue
            q = cut_at_edge(p, cx, cy, g["r"])
            L = float(np.sum(np.hypot(*np.diff(q, axis=0).T))) if len(q) > 1 else 0.0
            changed.append((gid, int(b), t["state"], round(float(t["length_px"]), 1), round(L, 1)))
            t["path_xy_ref"] = [[round(float(x), 2), round(float(y), 2)] for x, y in q]
            dx, dy = (np.asarray(t["path_xy"][0]) - np.asarray(p[0])) if t.get("path_xy") else (0.0, 0.0)
            t["path_xy"] = [[round(float(x + dx), 2), round(float(y + dy), 2)] for x, y in q]
            if t.get("path_xy_view"):
                t["path_xy_view"] = [[round(float(x - off[0]), 2), round(float(y - off[1]), 2)] for x, y in q]
            t["length_px"] = round(L, 2)
            t["edge_rule"] = f"cut at the census circle (started within {frac} r of the centre)"
    return out, changed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "benchmark/labels/m1_v1.json"))
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--out", default=str(REPO / "runs/tube_net/m1_v1_edge.json"))
    a = ap.parse_args(argv)
    d, changed = derive(json.loads(Path(a.labels).read_text()), a.frac)
    Path(a.out).write_text(json.dumps(d))
    for c in changed:
        print("  %s@%d %s: %.1f -> %.1f px" % c)
    print(f"{len(changed)} traces changed -> {a.out}")


if __name__ == "__main__":
    main()
