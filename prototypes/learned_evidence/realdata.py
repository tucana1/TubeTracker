"""Training crops from the human benchmark traces (real movies, partial labels).

    python -m prototypes.learned_evidence.realdata MOVIE OUT.npz      # MOVIE = ld | m2

A trace gives one tube centreline at one bin. Target body = the centreline +/- BODY px; the loss
sees only a band round it: the body, and background from GAP to BAND px off the centreline
(the walls in between are left out, and so is everything farther away, where other tubes
were never traced). The same crops at a bin before the grain's onset bracket are pure
negatives inside the band: the tube is not there yet. Tips: the apex of FULL traces.
Samples carry a weight map ``w`` (1 = scored); synthetic shards have none (all scored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .data import CacheView, tip_heatmap

REPO = Path(__file__).resolve().parents[2]
REAL = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
        "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json")}
BODY, GAP, BAND = 3.0, 6.0, 14.0


def build(movie: str, out: str | Path, half: int = 48, crops_per_trace: int = 6, neg_per_trace: int = 3,
          seed: int = 0, log=print) -> Path:
    cache, labels = (REPO / p for p in REAL[movie])
    L = json.loads(labels.read_text())
    view = CacheView(cache)
    rng = np.random.default_rng(seed)
    size = 2 * half
    xs, bodies, tips, ws, info = [], [], [], [], []
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        on = lab.get("onset") or {}
        if g.get("excluded"):
            continue
        neg_bin = on.get("last_absent_bin") if on.get("verdict") == "emerged_within" else None
        for b, t in lab.get("traces", {}).items():
            b = int(b)
            if t["state"] not in ("full", "partial") or len(t.get("path_xy_ref") or []) < 2 or b < view.rs + 3:
                continue
            path = np.asarray(t["path_xy_ref"], float)
            seg = np.hypot(*np.diff(path, axis=0).T)
            s = np.concatenate([[0], np.cumsum(seg)])
            for k in range(crops_per_trace + (neg_per_trace if neg_bin is not None and neg_bin >= view.rs + 3 else 0)):
                neg = k >= crops_per_trace
                u = rng.uniform(0, s[-1])
                cx = float(np.interp(u, s, path[:, 0])) + rng.uniform(-half / 3, half / 3)
                cy = float(np.interp(u, s, path[:, 1])) + rng.uniform(-half / 3, half / 3)
                cx = float(np.clip(round(cx), half + 4, view.r.width - half - 4))
                cy = float(np.clip(round(cy), half + 4, view.r.height - half - 4))
                x = view.sample(neg_bin if neg else b, cx, cy, half)
                if not np.isfinite(x).all():
                    continue
                q = (path - [cx - half, cy - half] - 0.5)  # crop pixel centres
                dist = np.full((size, size), 255, np.uint8)
                line = np.zeros((size, size), np.uint8)
                cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1,
                              cv2.LINE_8, 2)  # 1/4-px precision
                d = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
                body = (d <= BODY) & (not neg)
                w = ((d <= BODY) | ((d >= GAP) & (d <= BAND))).astype(np.float32)
                gr = np.hypot(np.arange(size)[None, :] - (g["x"] - cx + half - 0.5),
                              np.arange(size)[:, None] - (g["y"] - cy + half - 0.5))
                w[gr <= g["r"] - 1] = 1.0  # the grain itself is never tube
                body &= gr > g["r"] - 1
                tip = (tip_heatmap([tuple(path[-1])], cx, cy, half) if t["state"] == "full" and not neg
                       else np.zeros((size, size), np.float32))
                xs.append(x.astype(np.float16)); bodies.append(body.astype(np.uint8))
                tips.append(tip.astype(np.float16)); ws.append(w.astype(np.uint8))
                info.append((neg_bin if neg else b, cx, cy))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(xs), body=np.stack(bodies), tip=np.stack(tips), w=np.stack(ws),
                        info=np.array(info, np.float32), movie=movie)
    log(f"{out.name}: {len(xs)} samples ({int(np.mean([i[0] for i in info]) if info else 0)} mean bin), "
        f"scored pixels {100 * np.mean(np.stack(ws)):.1f}%, tube among them "
        f"{100 * np.stack(bodies).sum() / max(np.stack(ws).sum(), 1):.1f}%")
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
