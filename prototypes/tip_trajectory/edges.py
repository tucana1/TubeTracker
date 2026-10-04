"""The grain's visible edge all round it, from the before image: per 5 degrees the steepest radial change (as
``analyze.exit_edge``, here over a wider window, r - ``IN`` .. r + ``OUT_PX``) and then a circular running median over
+/- ``SMOOTH`` degrees, so that one direction's outlier (debris or a neighbour beside the grain, a faint stretch of
rim) takes its neighbours' edge. Returned as offsets from the census radius (72 values, angle k x 5 degrees, image
axes). ``check`` compares raw and smoothed edges with the annotator's exit clicks (diagnosis only).

    python -m prototypes.tip_trajectory.edges ld m2 m1
"""
from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.analyze import exit_edge
from sparsetrack.render import Renderer

from .common import baseline, cache_dir, drift_per_bin, labels, scored_grains

IN, OUT_PX, SMOOTH = 10.0, 8.0, 15.0


def edge_profile(img: np.ndarray, centre: float, r: float, theta: float, lo: float, hi: float, wedge_deg: float = 10.0,
                 n_ang: int = 9, step: float = 0.25) -> float:
    """``analyze.exit_edge`` with its window r - lo .. r + hi."""
    rads = np.arange(0.0, r + hi + 11.0, step)
    angs = theta + np.deg2rad(np.linspace(-wedge_deg, wedge_deg, n_ang))
    xs = (centre + rads[None] * np.cos(angs)[:, None]).astype(np.float32)
    ys = (centre + rads[None] * np.sin(angs)[:, None]).astype(np.float32)
    prof = np.median(cv2.remap(np.nan_to_num(img).astype(np.float32), xs, ys, cv2.INTER_LINEAR), axis=0)
    g = np.gradient(np.convolve(prof, np.ones(5) / 5, mode="same"), rads)
    win = (rads > r - lo) & (rads < r + hi)
    return float(rads[win][np.argmax(np.abs(g[win]))] - r)


def smoothed_edges(early: np.ndarray, centre: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    """(raw, smoothed) edge offsets for the 72 directions k x 5 degrees."""
    raw = np.array([edge_profile(early, centre, r, math.radians(5.0 * k), min(IN, r - 1.0), OUT_PX) for k in range(72)])
    h = int(round(SMOOTH / 5.0))
    ext = np.concatenate([raw[-h:], raw, raw[:h]])
    sm = np.array([np.median(ext[k:k + 2 * h + 1]) for k in range(72)])
    return raw, sm


def check(movie: str) -> None:
    bins, meta = stack.load(cache_dir(movie))
    R = Renderer(bins, meta)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    lab, base = labels(movie), baseline(movie)
    by = {g["id"]: g for g in base["grains"]}
    err = {"census": [], "exit_edge": [], "exit_edge_clip3.3": [], "raw_wide": [], "smoothed": [], "smoothed_clip5": []}
    for gid in scored_grains(lab):
        g = lab["grains"][gid]
        d = drift_per_bin(by[gid], rs, nb) if gid in by else np.zeros((nb, 2))
        cx, cy = g["x"] + d[rs][0], g["y"] + d[rs][1]
        early = np.mean([np.nan_to_num(R.crop(k, cx, cy, 64)) for k in range(rs, rs + 3)], axis=0)
        r = float(g["r"])
        raw, sm = smoothed_edges(early, 63.5, r)
        old = np.array([exit_edge(early, 63.5, r, math.radians(5.0 * k)) for k in range(72)])
        for b, t in (lab["labels"][gid].get("traces") or {}).items():
            b = int(b)
            if t["state"] not in ("full", "partial") or len(t.get("path_xy_ref") or []) < 2:
                continue
            c = np.array([g["x"], g["y"]]) + d[min(b, nb - 1)]
            ex = np.asarray(t["path_xy_ref"][0], float) - c
            k = int(round(math.degrees(math.atan2(ex[1], ex[0])) % 360 / 5.0)) % 72
            hum = float(np.hypot(*ex) - r)
            err["census"].append(abs(hum))
            err["exit_edge"].append(abs(hum - old[k]))
            err["exit_edge_clip3.3"].append(abs(hum - np.clip(old[k], -3.32, 3.32)))
            err["raw_wide"].append(abs(hum - raw[k]))
            err["smoothed"].append(abs(hum - sm[k]))
            err["smoothed_clip5"].append(abs(hum - np.clip(sm[k], -5, 5)))
    print(movie, " | ".join(f"{k}: median {np.median(v):.2f} mean {np.mean(v):.2f} within 2 px {np.mean(np.array(v) <= 2):.0%}"
                            for k, v in err.items()), flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        check(m)
