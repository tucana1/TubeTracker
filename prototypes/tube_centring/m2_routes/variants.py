"""Coverage of each bad trace by the drawn tube under variants (no bend, no drift, best translation), and the
network's P(tube) along the trace vs along the drawn tube at that bin."""
import numpy as np

from common import BAD, G, PRED, PR, drift_at, idx, length_at, pdist, resample, to_length, trace
from sparsetrack.routes import bent
import math


def route(g, b, use_bend=True, use_drift=True, cut=True):
    i = idx(g, b)
    p = np.asarray(g["path"], float)
    if use_bend:
        p = bent(p, g.get("bend"), i)
    rot = g.get("rotation_deg") or []
    th = math.radians(rot[i]) if i < len(rot) else 0.0
    c = np.asarray(g["exit_xy"] if (PRED["params"].get("rot_pivot") == "exit" and g.get("exit_xy")) else [g["x"], g["y"]])
    M = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    p = (p - c) @ M.T + c
    if cut:
        p = np.asarray(to_length(p.tolist(), length_at(g, b)), float)
    if use_drift and g.get("drift"):
        p = p + drift_at(g, b)
    return p


def pmap_along(b, pts):
    vals = []
    for x, y in pts:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if 0 <= yi < PR.height and 0 <= xi < PR.width:
            vals.append(float(PR.bins[b, yi, xi]) / 250.0)
    return np.asarray(vals)


def cov(trp, poly):
    return float(np.mean(pdist(trp, poly) <= 2.0)) if poly is not None and len(poly) > 1 else 0.0


for gid, b, f in BAD:
    g = G[gid]
    tr, t = trace(gid, b)
    trp, _ = resample(tr, 1.0)
    rows = {}
    rows["drawn"] = cov(trp, route(g, b))
    rows["no_bend"] = cov(trp, route(g, b, use_bend=False))
    rows["no_drift"] = cov(trp, route(g, b, use_drift=False))
    rows["neither"] = cov(trp, route(g, b, use_bend=False, use_drift=False))
    rows["uncut"] = cov(trp, route(g, b, cut=False))
    d = route(g, b)
    best = (0, 0, rows["drawn"])
    for dx in np.arange(-15, 15.1, 1.0):
        for dy in np.arange(-15, 15.1, 1.0):
            c = cov(trp, route(g, b, cut=False) + [dx, dy])
            if c > best[2] + 1e-9:
                best = (dx, dy, c)
    dp, _ = resample(d, 1.0)
    pt, pd = pmap_along(b, trp), pmap_along(b, dp)
    print(f"{gid}@{b}: " + ", ".join(f"{k} {v:.2f}" for k, v in rows.items())
          + f" | best shift of whole route ({best[0]:+.0f},{best[1]:+.0f}) -> {best[2]:.2f}"
          + f" | P(tube) along trace med {np.median(pt):.2f} (>0.5: {np.mean(pt > 0.5):.2f}); along drawn med"
          f" {np.median(pd):.2f} (>0.5: {np.mean(pd > 0.5):.2f})")
