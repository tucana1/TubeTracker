"""Training crops from the human traces, versions 2 and 3 (partial labels, as prototypes/learned_flood/realdata.py).

    python -m prototypes.tube_net.realdata MOVIE OUT.npz [--flat-cap]     # MOVIE = ld | m2; --flat-cap = version 3

Target body = the traced centreline +/- BODY px; scored background = GAP..BAND px off it; the grain's inside is
never tube; everything else unscored. Changes against the first version:

- the grain's inside is placed where the grain is at that bin (census + the trace's ``view_offset``; movie 2's
  grains move), not at its census place;
- PARTIAL traces (the tube goes on beyond the traced end): nothing within BAND + 4 px of the traced end is scored;
- traces flagged as touching another tube or grain: no background band (the band could hold the other tube);
- crops centred where the tube leaves the grain (``exit_crops`` per trace): young tubes and rims are rare otherwise;
- the same trace at neighbouring bins (``neighbours``): more noise realisations of the same tube; nothing within
  3 + 1.5 |k| px of the traced end is scored there (the tip moved), and no tip target;
- negatives at several bins before the human's onset bracket (``neg_back``: bins before ``last_absent_bin``), half
  of them centred on the exit, where a false start of the flood would begin;
- version 3 (``flat_cap``): at a FULL trace's own bin the body ends at the clicked apex - beyond the apex's plane the
  cap within GAP px is background - as the synthetic truth ends at the tube's length while the image blurs on past it
  (version 2's round cap taught the network to mark ~2 px past the apex, which the flood reads as length).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from prototypes.learned_flood.data import CacheView, tip_heatmap

REPO = Path(__file__).resolve().parents[2]
REAL = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
        "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json")}
BODY, GAP, BAND = 3.0, 6.0, 14.0


def _targets(path: np.ndarray, cx: float, cy: float, half: int, gxy: tuple, gr: float, neg: bool, band: bool,
             blind: list[tuple[float, float, float]], flat_cap: bool = False):
    size = 2 * half
    q = path - [cx - half, cy - half] - 0.5  # crop pixel centres
    line = np.zeros((size, size), np.uint8)
    cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
    d = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
    body = (d <= BODY) & (not neg)
    w = (d <= BODY) | (band & (d >= GAP) & (d <= BAND))
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    if flat_cap and not neg:
        # the body ends at the clicked apex, as the synthetic truth ends at the tube's length (the image blurs on
        # past it): beyond the apex's plane, the cap within GAP px of the apex is background
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        back = np.array([np.interp(s[-1] - 4.0, s, q[:, 0]), np.interp(s[-1] - 4.0, s, q[:, 1])])
        u = q[-1] - back
        u = u / (np.linalg.norm(u) + 1e-9)
        beyond = ((jj - q[-1][0]) * u[0] + (ii - q[-1][1]) * u[1] > 0) & \
                 (np.hypot(jj - q[-1][0], ii - q[-1][1]) <= GAP)
        body &= ~beyond
        w |= beyond
    for bx, by, rad in blind:  # unscored discs (reference coordinates)
        w &= np.hypot(jj - (bx - cx + half - 0.5), ii - (by - cy + half - 0.5)) > rad
    inside = np.hypot(jj - (gxy[0] - cx + half - 0.5), ii - (gxy[1] - cy + half - 0.5)) <= gr - 1
    w |= inside  # the grain itself is never tube
    body &= ~inside
    return body.astype(np.uint8), w.astype(np.uint8)


def build(movie: str, out: str | Path, half: int = 48, crops_per_trace: int = 6, exit_crops: int = 2,
          neighbours: tuple = (-2, -1, 1, 2), neg_back: tuple = (0, 4, 12), seed: int = 0, flat_cap: bool = False,
          log=print) -> Path:
    cache, labels = (REPO / p for p in REAL[movie])
    L = json.loads(labels.read_text())
    view = CacheView(cache)
    rng = np.random.default_rng(seed)
    lo, hi = view.rs + 3, view.n_bins - 2
    xs, bodies, tips, ws, info = [], [], [], [], []
    kinds = {"trace": 0, "exit": 0, "neighbour": 0, "negative": 0}

    def add(b, cx, cy, path, gxy, gr, neg, band, blind, tip_xy, kind):
        cx = float(np.clip(round(cx), half + 4, view.r.width - half - 4))
        cy = float(np.clip(round(cy), half + 4, view.r.height - half - 4))
        x = view.sample(b, cx, cy, half)
        if not np.isfinite(x).all():
            return
        # a flat cap only where the traced end is the tube's apex at this very bin (FULL trace, same bin)
        body, w = _targets(path, cx, cy, half, gxy, gr, neg, band, blind, flat_cap and tip_xy is not None)
        tip = tip_heatmap([tip_xy], cx, cy, half) if tip_xy is not None else np.zeros(body.shape, np.float32)
        xs.append(x.astype(np.float16)); bodies.append(body); tips.append(tip.astype(np.float16)); ws.append(w)
        info.append((b, cx, cy, int(neg)))
        kinds[kind] += 1

    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        if g.get("excluded"):
            continue
        on = lab.get("onset") or {}
        traces = sorted(((int(b), t) for b, t in lab.get("traces", {}).items()
                         if t["state"] in ("full", "partial") and len(t.get("path_xy_ref") or []) >= 2),
                        key=lambda x: x[0])
        for b, t in traces:
            if b < lo or b > hi:
                continue
            path = np.asarray(t["path_xy_ref"], float)
            off = t.get("view_offset") or [0.0, 0.0]
            gxy = (g["x"] + off[0], g["y"] + off[1])
            full, band = t["state"] == "full", not t.get("contact")
            end_blind = [] if full else [(path[-1][0], path[-1][1], BAND + 4.0)]
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
            jit = lambda: rng.uniform(-half / 3, half / 3)
            for _ in range(crops_per_trace):
                u = rng.uniform(0, s[-1])
                add(b, np.interp(u, s, path[:, 0]) + jit(), np.interp(u, s, path[:, 1]) + jit(), path, gxy, g["r"],
                    False, band, end_blind, tuple(path[-1]) if full else None, "trace")
            for _ in range(exit_crops):
                add(b, path[0][0] + jit(), path[0][1] + jit(), path, gxy, g["r"], False, band, end_blind,
                    tuple(path[-1]) if full else None, "exit")
            for k in neighbours:
                if not lo <= b + k <= hi:
                    continue
                u = rng.uniform(0, s[-1])
                blind = end_blind + [(path[-1][0], path[-1][1], 3.0 + 1.5 * abs(k))]
                add(b + k, np.interp(u, s, path[:, 0]) + jit(), np.interp(u, s, path[:, 1]) + jit(), path, gxy,
                    g["r"], False, band, blind, None, "neighbour")
        if on.get("verdict") == "emerged_within" and traces:
            b0, t0 = traces[0]
            path = np.asarray(t0["path_xy_ref"], float)
            off = t0.get("view_offset") or [0.0, 0.0]
            gxy = (g["x"] + off[0], g["y"] + off[1])
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
            for j in neg_back:
                nb_ = int(on["last_absent_bin"]) - j
                if not lo <= nb_ <= hi:
                    continue
                for k in range(2):  # one at the exit, one along the (future) trace
                    u = 0.0 if k == 0 else rng.uniform(0, s[-1])
                    add(nb_, np.interp(u, s, path[:, 0]) + rng.uniform(-half / 3, half / 3),
                        np.interp(u, s, path[:, 1]) + rng.uniform(-half / 3, half / 3), path, gxy, g["r"], True,
                        not t0.get("contact"), [], None, "negative")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    B, W = np.stack(bodies), np.stack(ws)
    np.savez_compressed(out, x=np.stack(xs), body=B, tip=np.stack(tips), w=W, info=np.array(info, np.float32),
                        movie=movie)
    log(f"{out.name}: {len(xs)} samples {kinds}, scored pixels {100 * W.mean():.1f}%, tube among them "
        f"{100 * B.sum() / max(W.sum(), 1):.1f}%")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("movie", choices=tuple(REAL))
    ap.add_argument("out")
    ap.add_argument("--flat-cap", action="store_true", help="the body ends at the clicked apex (no round cap)")
    a = ap.parse_args()
    build(a.movie, a.out, flat_cap=a.flat_cap)
