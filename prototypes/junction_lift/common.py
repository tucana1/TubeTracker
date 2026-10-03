"""Shared helpers for the junction-takeover study (3 Oct 2026; read-only on data). Inputs: the 0.8.8 baseline
predictions (BASE, a session scratchpad: regenerate with scripts/synth_bench.py --real ... --dump-real), the
labels and the tube-probability caches of the main checkout. Outputs (cases.json, ...) go to SP + "jt/"."""
import json
import math
import os
import sys

import numpy as np

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, WT)
from sparsetrack import stack  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402
from sparsetrack.report import turned_path  # noqa: E402
from sparsetrack import routes  # noqa: E402

MAIN = "/Users/joshjiang/Documents/TubeTracker/"
SP = "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/"
BASE = SP + "bt/base088/{}_real_0/predictions.json"
FPB = 300

_cache = {}


def movie(m):
    if m not in _cache:
        pred = json.load(open(BASE.format(m)))
        lab = json.load(open(MAIN + f"benchmark/labels/{m}_v1.json"))
        pbins, pmeta = stack.load(MAIN + f"runs/sparsetrack/{m}/prob_tubes_bn_real_ld_m2")
        _cache[m] = {"pred": pred, "G": {g["id"]: g for g in pred["grains"]}, "lab": lab,
                     "pbins": pbins, "pmeta": pmeta}
    return _cache[m]


def idx(g, b):
    return int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * FPB + FPB // 2))))


def length_at(g, b):
    return float(g["length"]["px"][idx(g, b)])


def drift_at(g, b):
    d = (g.get("drift") or {}).get("xy") or []
    i = idx(g, b)
    return np.asarray(d[i], float) if i < len(d) else np.zeros(2)


def tip_at(g, b):
    t = (g.get("tip") or {}).get("xy") or []
    i = idx(g, b)
    if i >= len(t) or t[i] is None or t[i][0] is None:
        return None
    return np.asarray(t[i], float) + drift_at(g, b)


def drawn(m, g, b, cut=True):
    """The model's tube as drawn at bin b in field coords (cut to its length then, or the whole route)."""
    pred = movie(m)["pred"]
    i = idx(g, b)
    Lb = g["length"]["px"][i]
    if not g.get("path"):
        return None
    r = turned_path(g, i, pred)
    if cut:
        if Lb <= 0.5:
            return None
        r = routes.cut(r, Lb)
    return np.asarray(r, float) + drift_at(g, b)


def trace(m, gid, b):
    t = (movie(m)["lab"]["labels"].get(gid, {}).get("traces") or {}).get(str(b))
    if not t or len(t.get("path_xy_ref") or []) < 2:
        return None, t
    return np.asarray(t["path_xy_ref"], float), t


def seg_dists(pts, line):
    """Distance from each point to polyline ``line`` (vectorised)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    line = np.asarray(line, float).reshape(-1, 2)
    if len(line) == 1:
        return np.hypot(*(pts - line[0]).T)
    a, b = line[:-1], line[1:]
    d = b - a
    dd = np.maximum((d ** 2).sum(1), 1e-12)
    v = pts[:, None, :] - a[None]
    s = np.clip((v * d[None]).sum(2) / dd[None], 0, 1)
    proj = a[None] + s[..., None] * d[None]
    return np.hypot(*(proj - pts[:, None, :]).transpose(2, 0, 1)).min(1)


def proj_arc(pts, line):
    """Arc position along ``line`` of each point's nearest point."""
    line = np.asarray(line, float).reshape(-1, 2)
    s = routes.arc(line)
    pts = np.asarray(pts, float).reshape(-1, 2)
    a, b = line[:-1], line[1:]
    d = b - a
    dd = np.maximum((d ** 2).sum(1), 1e-12)
    v = pts[:, None, :] - a[None]
    t = np.clip((v * d[None]).sum(2) / dd[None], 0, 1)
    proj = a[None] + t[..., None] * d[None]
    dist = np.hypot(*(proj - pts[:, None, :]).transpose(2, 0, 1))
    k = dist.argmin(1)
    return s[k] + t[np.arange(len(pts)), k] * np.sqrt(dd[k])


def pmap(m, b, x0, y0, w, h):
    """P map (0..1) of bin b over field window [x0, x0+w) x [y0, y0+h) (pixel indices); outside the frame 0."""
    mv = movie(m)
    H, W = mv["pbins"].shape[1:]
    out = np.zeros((h, w), np.float32)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = np.asarray(mv["pbins"][b, ya:yb, xa:xb], np.float32) / 250.0
    return out


def p_along(m, b, pts):
    """P at field points (continuous ref coords, pixel centres at +0.5)."""
    mv = movie(m)
    H, W = mv["pbins"].shape[1:]
    pts = np.asarray(pts, float).reshape(-1, 2)
    xi = np.clip(np.floor(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.floor(pts[:, 1]).astype(int), 0, H - 1)
    return np.asarray(mv["pbins"][b], np.float32)[yi, xi] / 250.0
