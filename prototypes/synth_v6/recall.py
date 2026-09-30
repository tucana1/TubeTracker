"""Pixel check of a tube-probability cache against the human traces, with young tubes counted.

    python -m prototypes.synth_v6.recall PROB_CACHE LABELS [--json out.json]

``prototypes.learned_flood.realpix`` skips trace points within r + 4 px of the grain centre, which leaves
almost nothing of a young (2-8 px) trace. Here every FULL trace (not touching another tube or grain) is
measured from the grain's visible edge (r - 1 px from its centre at that bin; drifting grains are followed
with the trace's view offset):

- point recall: share of trace points (0.5 px apart) the network calls tube (max P over +/-2 px across >= 0.5);
- stub seen: the trace has at least 1 px of such points beyond the edge (the flood could start on it);
- tip seen: max P within 2.5 px of the traced apex >= 0.5;
- beside: share of points 8-14 px to either side called tube (traces >= 10 px; realpix's false-mark check);
- realpix: the original measure (points beyond r + 4 px), for comparison with earlier numbers.

Split by: all traces, each grain's first trace, and young traces (<= 8 px).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from prototypes.learned_flood.realpix import trace_points
from sparsetrack import stack
from sparsetrack.render import Renderer

P_SCALE = 250.0


def _samp(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    return cv2.remap(img, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR, borderValue=0)[0]


def measure(prob_cache: str | Path, labels_path: str | Path) -> list[dict]:
    bins, meta = prob_cache if isinstance(prob_cache, tuple) else stack.load(prob_cache)  # or an in-memory (bins, meta)
    R = Renderer(bins, meta)
    L = json.loads(Path(labels_path).read_text())
    rows = []
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        if g.get("excluded"):
            continue
        traces = sorted(((int(b), t) for b, t in lab.get("traces", {}).items() if t["state"] == "full"),
                        key=lambda x: x[0])
        first_bin = traces[0][0] if traces else None
        for b, t in traces:
            path = t.get("path_xy_ref") or []
            if t.get("contact") or len(path) < 2:
                continue
            off = t.get("view_offset") or [0.0, 0.0]
            cx, cy = g["x"] + off[0], g["y"] + off[1]  # the grain's centre at this bin
            pts, nrm = trace_points(path, 0.5)
            dc = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            half = int(np.ceil(np.max(np.abs(pts - [g["x"], g["y"]])))) + 24
            img = np.nan_to_num(R.crop(b, g["x"], g["y"], half) / P_SCALE).astype(np.float32)
            q = pts - [g["x"] - half, g["y"] - half] - 0.5
            on = np.max([_samp(img, q + o * nrm) for o in (-2, -1, 0, 1, 2)], axis=0) >= 0.5
            edge = dc >= g["r"] - 1.0
            rp = dc >= g["r"] + 4.0
            yy, xx = np.mgrid[-3:4, -3:4]
            disc = np.hypot(xx, yy) <= 2.5
            tq = q[-1] + np.stack([xx[disc], yy[disc]], 1)
            tip = bool(np.max(_samp(img, tq)) >= 0.5)
            row = {"grain": gid, "bin": b, "length": float(t["length_px"]), "first": b == first_bin,
                   "n_edge": int(edge.sum()), "hit_edge": int((on & edge).sum()),
                   "n_rp": int(rp.sum()), "hit_rp": int((on & rp).sum()),
                   "stub_seen": bool((on & edge).sum() >= 2), "tip_seen": tip}
            if t["length_px"] >= 10:
                off_pts = np.concatenate([np.max([_samp(img, q + s * o * nrm) for o in (8, 11, 14)], axis=0) >= 0.5
                                          for s in (-1, 1)])
                row["beside_n"], row["beside_hit"] = int(len(off_pts)), int(off_pts.sum())
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    out = {}
    for name, sel in (("all", lambda r: True), ("first", lambda r: r["first"]), ("young<=8", lambda r: r["length"] <= 8)):
        s = [r for r in rows if sel(r)]
        ne, he = sum(r["n_edge"] for r in s), sum(r["hit_edge"] for r in s)
        nr, hr = sum(r["n_rp"] for r in s), sum(r["hit_rp"] for r in s)
        bn, bh = sum(r.get("beside_n", 0) for r in s), sum(r.get("beside_hit", 0) for r in s)
        out[name] = {"traces": len(s), "point_recall": he / max(ne, 1), "stub_seen": sum(r["stub_seen"] for r in s),
                     "tip_seen": sum(r["tip_seen"] for r in s), "realpix": hr / max(nr, 1),
                     "beside": bh / max(bn, 1)}
    return out


def fmt(name: str, summ: dict) -> str:
    return " | ".join(f"{k}: n={v['traces']} pts {100 * v['point_recall']:.0f}% stub {v['stub_seen']}/{v['traces']} "
                      f"tip {v['tip_seen']}/{v['traces']} (realpix {100 * v['realpix']:.0f}%, beside {100 * v['beside']:.1f}%)"
                      for k, v in summ.items())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prob_cache")
    ap.add_argument("labels")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = measure(a.prob_cache, a.labels)
    s = summarise(rows)
    print(fmt(a.prob_cache, s))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "rows": rows}))


def rim_false_marks(prob_cache: str | Path, labels_path: str | Path, step: int = 3, min_px: int = 6) -> dict:
    """Before a grain's tube exists (bins up to 2 before the human's first visible bin; every bin of grains that
    never germinate): how often the map marks the rim zone - r to r + 7 px from the grain centre, where the flood
    may start a tube - with at least ``min_px`` pixels of P >= 0.5. Each such grain-bin is a chance for a false start."""
    bins, meta = prob_cache if isinstance(prob_cache, tuple) else stack.load(prob_cache)  # or an in-memory (bins, meta)
    R = Renderer(bins, meta)
    L = json.loads(Path(labels_path).read_text())
    rs = int(meta.get("ref_start", 0))
    nb = int(meta["n_bins"])
    marked = total = 0
    grains_marked = set()
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        if g.get("excluded"):
            continue
        on = lab.get("onset", {})
        if on.get("verdict") == "emerged_within":
            end = int(on["first_visible_bin"]) - 2
        elif on.get("verdict") == "no_emergence_by_end":
            end = nb - 1
        else:
            continue
        tr = sorted(((int(b), t) for b, t in lab.get("traces", {}).items() if t["state"] == "full"), key=lambda x: x[0])
        off = (tr[0][1].get("view_offset") or [0.0, 0.0]) if tr else [0.0, 0.0]
        cx, cy = g["x"] + off[0], g["y"] + off[1]
        half = int(g["r"] + 12)
        yy, xx = np.mgrid[0:2 * half, 0:2 * half]
        d = np.hypot(xx + 0.5 - half, yy + 0.5 - half)
        zone = (d >= g["r"]) & (d <= g["r"] + 7)
        for b in range(rs + 3, end, step):
            p = np.nan_to_num(R.crop(b, cx, cy, half)) / P_SCALE
            total += 1
            if int(((p >= 0.5) & zone).sum()) >= min_px:
                marked += 1
                grains_marked.add(gid)
    return {"grain_bins": total, "marked": marked, "share": marked / max(total, 1), "grains": len(grains_marked)}
