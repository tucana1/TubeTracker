"""Is the grain's own change as strong, relative to a young tube, in synthetic v6 as in the real movies?

Per grain, over bins before its onset (and all bins of grains that never germinate): the RMS of the change
(bin minus the "before" image, as the network sees it) inside r + 3 px of the grain centre - crescents, rings,
interior. Per young tube: its own contrast, the largest |change| within 2 px of its path beyond the grain's
edge, at its first trace (real: the human's first FULL trace <= 8 px; synthetic: the first bin with 3-8 px of
tube). The ratio of the two is what the network has to see through.

    python -m prototypes.synth_v6.crescent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from prototypes.learned_flood.realpix import trace_points

from .sheet import View

REPO = Path(__file__).resolve().parents[2]


def grain_rms(view: View, x: float, y: float, r: float, bins: list[int], half: int = 32) -> list[float]:
    early = np.mean([view.R.crop(b, x, y, half) for b in range(view.rs, view.rs + 3)], axis=0)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    disc = np.hypot(xx - half + 0.5, yy - half + 0.5) < r + 3
    ring = (np.hypot(xx - half + 0.5, yy - half + 0.5) > r + 12) & (np.hypot(xx - half + 0.5, yy - half + 0.5) < r + 22)
    out = []
    for b in bins:
        d = view.R.crop(b, x, y, half) - early
        if np.all(np.isfinite(d)):
            d = d - np.median(d[ring])  # the field's uniform brightness change is not the grain's
            out.append(float(np.sqrt(np.mean(d[disc] ** 2))))
    return out


def tube_contrast(view: View, gx: float, gy: float, r: float, b: int, path: np.ndarray, half: int = 40) -> float:
    early = np.mean([view.R.crop(e, gx, gy, half) for e in range(view.rs, view.rs + 3)], axis=0)
    d = (view.R.crop(b, gx, gy, half) - early).astype(np.float32)
    pts, _ = trace_points(path, 0.5)
    pts = pts[np.hypot(pts[:, 0] - gx, pts[:, 1] - gy) >= r - 1.0]
    if not len(pts):
        return float("nan")
    best = 0.0
    for dx in np.arange(-2, 2.01, 1.0):
        for dy in np.arange(-2, 2.01, 1.0):
            q = (pts + [dx, dy] - [gx - half, gy - half] - 0.5).astype(np.float32)
            v = cv2.remap(np.nan_to_num(d), q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR)[0]
            best = max(best, float(np.max(np.abs(v))))
    return best


def real(movie: str, step: int = 6) -> dict:
    view = View(REPO / f"runs/sparsetrack/{movie}")
    L = json.loads((REPO / f"benchmark/labels/{movie}_v1.json").read_text())
    rms, con = [], []
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        if g.get("excluded"):
            continue
        on = lab.get("onset", {})
        traces = sorted(((int(b), t) for b, t in lab.get("traces", {}).items() if t["state"] == "full"),
                        key=lambda x: x[0])
        off = (traces[0][1].get("view_offset") or [0.0, 0.0]) if traces else [0.0, 0.0]
        if on.get("verdict") == "emerged_within":
            end = int(on["first_visible_bin"]) - 2
        elif on.get("verdict") == "no_emergence_by_end":
            end = view.nb - 1
        else:
            continue
        # the grain may drift: measure where it was traced (first trace's offset) - near its onset
        rms += grain_rms(view, g["x"] + off[0], g["y"] + off[1], g["r"], list(range(view.rs + 3, end, step)))
        if traces and traces[0][1]["length_px"] <= 8 and not traces[0][1].get("contact"):
            b, t = traces[0]
            con.append(tube_contrast(view, g["x"] + off[0], g["y"] + off[1], g["r"], b, np.asarray(t["path_xy_ref"])))
    return {"rms": rms, "contrast": [c for c in con if np.isfinite(c)]}


def synthetic(cache: str, step: int = 3) -> dict:
    view = View(REPO / cache)
    T = json.loads((REPO / cache / "truth.json").read_text())
    fpb = 25
    rms, con = [], []
    for gid, lab in T["labels"].items():
        g = T["grains"][gid]
        t = lab.get("truth", {})
        if not g.get("isolated"):
            continue
        if "onset_frame" in t:
            end = int(t["onset_frame"] // fpb) - 2
        else:
            end = view.nb - 1
        rms += grain_rms(view, g["x"], g["y"], g["r"], list(range(view.rs + 3, max(end, view.rs + 3), step)))
        if "onset_frame" in t and "path_rel" in t and not t.get("max_drift_px"):
            # the first bin whose centre frame has 3-8 px of tube (length from the benchmark traces' schedule is
            # not stored per bin; use rate and lag: L(k) for the slow start)
            L = [(int(b), tr["length_px"]) for b, tr in lab["traces"].items()]
            onset_b = t["onset_frame"] / fpb
            rate, lag = t["rate_px_per_bin"], t["lag_bins"]
            for b in range(int(onset_b) + 1, view.nb - 1):
                dt = b + 0.5 - onset_b
                length = min(dt, lag) * rate * 0.3 + max(dt - lag, 0.0) * rate  # slow start at 0.3 x rate
                if 3.0 <= length <= 8.0:
                    path = np.asarray(t["path_rel"], float) + [g["x"], g["y"]]
                    seg = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
                    keep = path[seg <= length + 1e-6]
                    if len(keep) >= 2:
                        con.append(tube_contrast(view, g["x"], g["y"], g["r"], b, keep))
                    break
                if length > 8.0:
                    break
    return {"rms": rms, "contrast": [c for c in con if np.isfinite(c)]}


def q(v) -> str:
    v = np.asarray(v, float)
    return f"n={len(v)} p25/50/75/90 " + " ".join(f"{x:.1f}" for x in np.percentile(v, [25, 50, 75, 90])) if len(v) else "n=0"


if __name__ == "__main__":
    res = {}
    for name in sys.argv[1:] or ["real:ld", "real:m2"]:
        kind, what = name.split(":", 1)
        r = real(what) if kind == "real" else synthetic(what)
        res[name] = r
        ratio = np.median(r["rms"]) / np.median(r["contrast"]) if r["rms"] and r["contrast"] else float("nan")
        print(f"{name}: grain change RMS inside r+3 {q(r['rms'])} | young tube contrast {q(r['contrast'])} | "
              f"median ratio {ratio:.2f}", flush=True)
    Path(REPO / "runs/synth_v6/crescent.json").write_text(json.dumps(res))


def from_shard(shard: str, field: str) -> list[float]:
    """The same grain-change RMS measured in a training shard's crops (final generator, real field geometry):
    every census grain lying inside a crop with no tube body within r + 6 px of it."""
    z = np.load(REPO / shard)
    grains = json.loads((REPO / field / "grains.json").read_text())["grains"]
    x, body, info = z["x"], z["body"], z["info"]
    half = x.shape[-1] // 2
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    out = []
    for i in range(len(x)):
        _, cx, cy = info[i]
        chg = (x[i, 0].astype(np.float32) - x[i, 1].astype(np.float32)) * 20.0  # back to grey levels
        for g in grains:
            u, v = g["x"] - (cx - half), g["y"] - (cy - half)  # grain centre in crop pixels (+0.5 convention)
            if not (g["r"] + 3 <= u - 0.5 <= 2 * half - g["r"] - 3 and g["r"] + 3 <= v - 0.5 <= 2 * half - g["r"] - 3):
                continue
            dc = np.hypot(xx + 0.5 - u, yy + 0.5 - v)
            if body[i][dc < g["r"] + 6].any():
                continue
            ring = (dc > g["r"] + 12) & (dc < g["r"] + 22)
            d = chg - (np.median(chg[ring]) if ring.any() else 0.0)
            out.append(float(np.sqrt(np.mean(d[dc < g["r"] + 3] ** 2))))
    return out
