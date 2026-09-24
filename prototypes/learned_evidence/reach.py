"""Decoder v2 sketch: read the tube where it is in every bin, then make its length monotone.

SparseTrack traces one centreline on the end-state change map and reads every bin along that
path (rotated rigidly). With a tube-probability map for every bin that is no longer necessary:

1. per grain, follow its drift (SparseTrack's ``local_shifts`` on the image cache);
2. per bin, keep P > ``thr`` outside grain bodies, take the connected region attached to the
   grain's rim, and split it from regions reaching other grains' rims by geodesic ownership
   (SparseTrack's ``geodesic_owner``);
3. the bin's raw length is the geodesic reach from the rim through that region;
4. over bins, the length is the monotone curve (growth <= ``vmax`` px per bin) closest to the raw
   lengths in L1, so single-bin misses and flickers do not move it; onset is where it passes
   ``onset_px``.

Writes predictions in SparseTrack's schema, so ``sparsetrack.evaluate.score`` scores them.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

import sparsetrack.analyze as A
from sparsetrack import stack
from sparsetrack.evaluate import PRED_SCHEMA
from sparsetrack.render import Renderer


def monotone_l1(raw: np.ndarray, vmax: float, step: float = 0.5) -> np.ndarray:
    """Non-decreasing sequence with increments <= vmax minimising sum |x - raw| (grid DP)."""
    levels = np.arange(0.0, float(raw.max()) + 2 * step, step)
    k = max(1, int(round(vmax / step)))
    n, m = len(raw), len(levels)
    cost = np.abs(levels[None, :] - raw[:, None])
    best = cost[0].copy()
    back = np.zeros((n, m), np.int32)
    idx = np.arange(m)
    for t in range(1, n):
        pad = np.concatenate([np.full(k, np.inf), best])
        win = np.lib.stride_tricks.sliding_window_view(pad, k + 1)  # win[j] covers levels j-k..j
        j = np.argmin(win, axis=1)
        best = cost[t] + win[idx, j]
        back[t] = idx - (k - j)
    out = np.zeros(n, np.int32)
    out[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return levels[out]


def reach_grain(RP: Renderer, R_img: Renderer, meta: dict, grain: dict, others: list[dict], thr: float = 0.5,
                scale: float = 16.0, half: int = 150, vmax: float = 4.0, onset_px: float = 2.0,
                min_tube_px: float = 8.0, rim_band: float = 5.0) -> dict:
    fpb, rs, nb = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0)), int(meta["n_bins"])
    gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
    centre = half - 0.5
    img = np.stack([R_img.crop(b, gx, gy, half) for b in range(rs, nb)])
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    ls = A.local_shifts(img, centre, gr, 12.0, 3)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - centre, yy - centre)
    blocked = rg < gr - 1.0
    rings = []
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 10 < ox < 2 * half + o["r"] + 10 and -o["r"] - 10 < oy < 2 * half + o["r"] + 10:
            d = np.hypot(xx - ox, yy - oy)
            blocked |= d < o["r"] + 1.0
            rings.append((d >= o["r"] + 1.0) & (d <= o["r"] + rim_band))
    own_ring = (rg >= gr - 1.0) & (rg <= gr + rim_band)
    rim = (rg >= gr - 1.0) & (rg <= gr + 1.5)
    raw = np.zeros(nb - rs)
    for i, b in enumerate(range(rs, nb)):
        p = RP.crop(b, gx, gy, half) / scale
        dx, dy = ls[i]
        p = cv2.warpAffine(np.nan_to_num(p).astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                           (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = (p > thr) & ~blocked
        if not (m & own_ring).any():
            continue
        _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[m & own_ring])
        comp = np.isin(lab, ids[ids > 0])
        rivals = [r for r in rings if (comp & r).any()]
        if rivals:
            comp = A.geodesic_owner(comp, [own_ring] + rivals) == 0
        seeds = comp & rim
        offset = 0.0
        if not seeds.any():  # the region starts beyond the rim (a faint base): seed at its nearest pixels
            rmin = float(rg[comp & own_ring].min()) if (comp & own_ring).any() else float(rg[comp].min())
            seeds = comp & (rg <= rmin + 1.0)
            offset = max(0.0, rmin - gr)
        if not seeds.any():
            continue
        cum, _ = MCP_Geometric(np.where(comp, 1.0, np.inf)).find_costs(list(zip(*np.nonzero(seeds))))
        reach = cum[comp & np.isfinite(cum)]
        raw[i] = (float(reach.max()) + offset) if reach.size else 0.0
    fit = monotone_l1(raw, vmax) if raw.max() > 0 else raw
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    on = np.nonzero(fit >= onset_px)[0]
    flags = []
    if fit[-1] < min_tube_px:
        status, onset, fit = "no_emergence_by_end", None, np.zeros_like(fit)
    elif on[0] == 0:
        status, onset = "emerged_at_start", frames[0]
    else:
        status, onset = "emerged_within", frames[int(on[0])]
    interval = None if onset is None or on[0] == 0 else [frames[int(on[0]) - 1], onset]
    return {"id": grain["id"], "x": gx, "y": gy, "r": gr, "status": status, "onset_frame": onset,
            "onset_interval": interval, "final_length_px": round(float(fit[-1]), 2),
            "length": {"frames": frames, "px": [round(float(v), 2) for v in fit]},
            "raw_reach_px": [round(float(v), 2) for v in raw], "flags": flags}


def analyze(pcache: str | Path, image_cache: str | Path, grains_path: str | Path | None = None, log=print,
            **kw) -> dict:
    import json
    bins_p, meta = stack.load(pcache)
    RP = Renderer(bins_p, meta)
    R_img = Renderer(*stack.load(image_cache))
    src = Path(grains_path) if grains_path else Path(pcache) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in census if not g.get("excluded")]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    started, out = time.time(), []
    for g in grains:
        others = [o for o in physical if o["id"] != g["id"]]
        out.append(reach_grain(RP, R_img, meta, g, others, **kw))
    log(f"reach decoder: {len(out)} grains in {time.time() - started:.0f} s")
    return {"schema": PRED_SCHEMA, "method": "reach decoder (learned-evidence prototype)",
            "frames_per_bin": meta["frames_per_bin"], "grains": out}
