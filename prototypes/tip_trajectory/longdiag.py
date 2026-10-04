"""Why long tubes get no candidate at their apex (diagnosis; human labels used only to look): per long FULL trace
(>= 60 px), at its bin, on the crop the candidates are built from - is the apex in the crop, reachable from the rim
through the passable region, on the map, how is the traced tube marked along its length, where does the detector's
nearest peak rank among the crop's peaks, and is the apex a far end of a map piece (and its rank by path cost)?

    python -m prototypes.tip_trajectory.longdiag m1 [--min 60]
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from .cands import DIL, EPS, HALF, P_LO, RIM_ZONE, crop_u8, resample
from .common import OUT, PROB, baseline, cache_dir, drift_per_bin, labels, scored_grains


def main(movie: str, min_len: float) -> None:
    from sparsetrack import stack
    _, meta = stack.load(cache_dir(movie))
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    prob = np.load(cache_dir(movie) / PROB / "bins.npy", mmap_mode="r")
    lab, base = labels(movie), baseline(movie)
    by = {g["id"]: g for g in base["grains"]}
    H = HALF[movie]
    S = 2 * H
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    for gid in scored_grains(lab):
        g = lab["grains"][gid]
        drift = drift_per_bin(by[gid], rs, nb) if gid in by else np.zeros((nb, 2))
        for b, t in sorted(((int(b), t) for b, t in (lab["labels"][gid].get("traces") or {}).items()), key=lambda x: x[0]):
            if t["state"] != "full" or t.get("contact") or t["length_px"] < min_len:
                continue
            r = float(g["r"])
            cx, cy = g["x"] + drift[b][0], g["y"] + drift[b][1]
            ix, iy = int(round(cx)), int(round(cy))
            gx, gy = cx - (ix - H) - 0.5, cy - (iy - H) - 0.5
            Pc = crop_u8(np.asarray(prob[b]), ix, iy, H).astype(np.float32) / 250.0
            f = OUT / "det" / movie / f"b{b:03d}.npz"
            Dt = crop_u8(np.load(f)["tip"], ix, iy, H).astype(np.float32) / 250.0 if f.exists() else None
            rg = np.hypot(jj - gx, ii - gy)
            poly = np.asarray(t["path_xy_ref"], float)
            pc = poly - np.array([ix - H + 0.5, iy - H + 0.5])  # crop index coordinates (col, row)
            ax, ay = pc[-1]
            inside = 0 <= ax < S and 0 <= ay < S
            line = resample(pc, 1.0)[0]
            li = np.clip(np.round(line).astype(int), 0, S - 1)
            pv = Pc[li[:, 1], li[:, 0]]
            low = np.concatenate([[0], (pv < 0.3).astype(int), [0]])
            dl = np.diff(low)
            runs = np.flatnonzero(dl == -1) - np.flatnonzero(dl == 1)
            passable = cv2.dilate((Pc >= P_LO).astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1))) > 0
            passable |= rg <= r + RIM_ZONE
            passable &= rg >= r - 1.0
            ring = np.argwhere((np.abs(rg - r) <= 0.5) & passable)
            mcp = MCP_Geometric(np.where(passable, 1.0 / (Pc + EPS), np.inf), fully_connected=True)
            cum, _ = mcp.find_costs([tuple(q) for q in ring])
            row = f"{gid} b{b} L={t['length_px']:.0f} drift={math.hypot(*drift[b]):.0f}"
            if not inside:
                print(row, "apex OUTSIDE the crop")
                continue
            ai, aj = int(round(ay)), int(round(ax))
            win = (slice(max(ai - 3, 0), ai + 4), slice(max(aj - 3, 0), aj + 4))
            reach = np.isfinite(cum[win]).any()
            p_apex = float(Pc[win].max())
            row += (f" | traced tube on map >=0.5: {np.mean(pv >= 0.5):.0%}, >=0.2: {np.mean(pv >= 0.2):.0%}, "
                    f"longest gap {int(runs.max()) if len(runs) else 0} px | apex P {p_apex:.2f} reachable {reach}")
            if Dt is not None:
                m = np.where((rg >= r - 2) & (rg <= H - 4), Dt, -1.0).astype(np.float32)
                pk = (m == cv2.dilate(m, np.ones((9, 9), np.uint8))) & (m >= 0.05)
                ys, xs = np.nonzero(pk)
                vals = m[ys, xs]
                d = np.hypot(xs - ax, ys - ay)
                if len(d):
                    k = int(np.argmin(d))
                    rank = int((vals > vals[k]).sum()) + 1
                    row += f" | det: nearest peak {d[k]:.1f} px, value {vals[k]:.2f}, rank {rank}/{len(vals)}"
            valid = np.isfinite(cum) & (Pc >= 0.5) & (rg >= r + 4)
            if valid.any():
                cm = np.where(valid, cum, -1.0).astype(np.float32)
                pk = (cm == cv2.dilate(cm, np.ones((11, 11), np.uint8))) & valid
                ys, xs = np.nonzero(pk)
                d = np.hypot(xs - ax, ys - ay)
                if len(d):
                    k = int(np.argmin(d))
                    rank = int((cm[ys, xs] > cm[ys[k], xs[k]]).sum()) + 1
                    row += f" | map end: nearest {d[k]:.1f} px, rank {rank}/{len(d)}"
            print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--min", type=float, default=60.0)
    a = ap.parse_args()
    main(a.movie, a.min)
