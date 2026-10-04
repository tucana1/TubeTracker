"""Per labelled grain and bin: tip candidates, each with its tube body, and how consecutive bins' bodies agree.

At bin b the grain is at its census place + 0.8.8's drift (label-free). On a crop of the tube map P round it:
- passable: within ``DIL`` px of P >= ``P_LO``, or within ``RIM_ZONE`` px beyond the rim (young tubes the map misses),
  never inside the grain (r - 1);
- cost 1 / (P + ``EPS``); one multi-source minimal-path search from the rim ring (radius r) gives every candidate's
  body (the cheapest route from the rim to it).
Candidates: the detector's top ``K_DET`` peaks (>= ``DET_MIN``) beyond r - 2; the far ends of the map's pieces reached
from the rim (local maxima of the path cost on P >= 0.5, ``K_END`` farthest); the fresh candidates of the last
``W_CARRY`` bins carried with the grain ("hold"); merged within ``NMS`` px.
Per candidate: smoothed body arc length from the rim, exit angle, the grain's visible edge there (exit_edge), map
support along the body, the longest gap in it, detector value at the tip, detector body head along the body, map
ahead of the tip, radial gain over the first 8 px, the tip's radius.
Per pair of candidates at consecutive bins whose arc lengths differ by <= ``PAIR_DL`` px: distance from the shorter
body's tip to the longer body, and mean distance of the shorter body (from 3 px on) to the longer (grain frame).

    python -m prototypes.tip_trajectory.cands ld m2 m1 [--only g005 g016]
Output: OUT/cands_<movie>.pkl
"""
from __future__ import annotations

import argparse
import math
import pickle
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from sparsetrack import stack
from sparsetrack.analyze import exit_edge
from sparsetrack.render import Renderer

from .common import OUT, PROB, baseline, cache_dir, drift_per_bin, labels, scored_grains

EPS = 0.1
P_LO = 0.2
DIL = 4
RIM_ZONE = 25.0
K_DET = 6
K_NEAR = 3     # + the strongest detector peaks within NEAR_PX of the rim
NEAR_PX = 30.0
DET_MIN = 0.05
K_END = 6
W_CARRY = 8
NMS = 3.0
MAX_C = 32
PAIR_SHRINK = 8.0   # pairs kept: length change from -8 px ...
PAIR_GROW = 16.0    # ... to +16 px per bin apart
HALF = {"ld": 200, "m2": 300, "m1": 300}
FEATS = ("L_arc", "theta", "edge", "sup", "sup_frac", "gap", "det", "det_body", "ahead", "radial", "rtip", "cum",
         "src", "ext")  # src: 1 detector peak, 2 map end, 5 / 6 the same carried from an earlier bin


def crop_u8(img: np.ndarray, ix: int, iy: int, H: int) -> np.ndarray:
    """img[iy-H:iy+H, ix-H:ix+H] with zeros outside the frame."""
    h, w = img.shape
    out = np.zeros((2 * H, 2 * H), np.uint8)
    y0, y1, x0, x1 = iy - H, iy + H, ix - H, ix + H
    a0, a1, b0, b1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if a1 > a0 and b1 > b0:
        out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = img[a0:a1, b0:b1]
    return out


def seg_dist(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance of each point (n, 2) to the polyline (m, 2)."""
    if len(line) == 1:
        return np.hypot(*(pts - line[0]).T)
    a, b = line[:-1], line[1:]
    d = b - a
    dd = np.maximum((d ** 2).sum(1), 1e-9)
    t = np.clip(((pts[:, None, :] - a[None]) * d[None]).sum(-1) / dd[None], 0.0, 1.0)
    q = a[None] + t[..., None] * d[None]
    return np.sqrt(((pts[:, None, :] - q) ** 2).sum(-1)).min(1)


def resample(line: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    if s[-1] <= 0:
        return line[:1].copy(), np.zeros(1)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - q[-1] > 1e-6:
        q = np.append(q, s[-1])
    return np.stack([np.interp(q, s, line[:, 0]), np.interp(q, s, line[:, 1])], 1), q


class GrainState:
    def __init__(self, gid: str, g: dict, drift: np.ndarray, edges: np.ndarray):
        self.gid, self.x, self.y, self.r = gid, float(g["x"]), float(g["y"]), float(g["r"])
        self.drift, self.edges = drift, edges
        self.recent: list[tuple[int, np.ndarray, list]] = []  # (bin, fresh candidates (n, 2), grain frame; sources)
        self.prev = None  # previous bin: (L_arc (n,), bodies [grain frame polylines])
        self.bins: dict[int, dict] = {}


def process(gs: GrainState, b: int, P: np.ndarray, det: dict | None, H: int, grid) -> None:
    jj, ii = grid
    r = gs.r
    cx, cy = gs.x + gs.drift[b][0], gs.y + gs.drift[b][1]
    ix, iy = int(round(cx)), int(round(cy))
    gx, gy = cx - (ix - H) - 0.5, cy - (iy - H) - 0.5  # grain centre in crop index coordinates (col, row)
    to_grain = np.array([ix - H + 0.5 - gs.drift[b][0], iy - H + 0.5 - gs.drift[b][1]])  # crop (col,row) -> grain frame
    Pc = crop_u8(P, ix, iy, H).astype(np.float32) / 250.0
    Dt = crop_u8(det["tip"], ix, iy, H).astype(np.float32) / 250.0 if det is not None else None
    Db = crop_u8(det["body"], ix, iy, H).astype(np.float32) / 250.0 if det is not None else None
    rg = np.hypot(jj - gx, ii - gy)
    # --- detector peaks
    cand, src = [], []
    rim_det = float(Dt[(rg >= r - 2) & (rg <= r + 25)].max()) if Dt is not None else 0.0
    if Dt is not None:
        m = np.where((rg >= r - 2) & (rg <= H - 4), Dt, -1.0).astype(np.float32)
        pk = (m == cv2.dilate(m, np.ones((9, 9), np.uint8))) & (m >= DET_MIN)
        ys, xs = np.nonzero(pk)
        order = np.argsort(-m[ys, xs], kind="stable")
        near = np.hypot(xs - gx, ys - gy) <= r + NEAR_PX
        # the strongest peaks near the rim first (a young tip is weak beside a crowded field's grown tips), then
        # the strongest anywhere
        for sel, extra in ((order[near[order]], K_NEAR), (order, K_DET)):
            kmax = len(cand) + extra
            for o in sel:
                if len(cand) >= kmax:
                    break
                q = (float(xs[o]), float(ys[o]))
                if all(math.hypot(q[0] - a, q[1] - c) >= 4 for a, c in cand):
                    cand.append(q)
                    src.append(1)
    # --- carried (hold): fresh candidates of the last W_CARRY bins, moved with the grain
    carried = [(float(q[0]), float(q[1])) for _, pts, _ in reversed(gs.recent) for q in pts - to_grain]
    carried_src = [4 + sf for _, _, srcs in reversed(gs.recent) for sf in srcs]
    # --- minimal paths from the rim
    passable = cv2.dilate((Pc >= P_LO).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1))) > 0
    passable |= rg <= r + RIM_ZONE
    for q in cand + carried:  # a detector peak a little off the map is still reachable
        if -3 <= q[0] < 2 * H + 3 and -3 <= q[1] < 2 * H + 3:
            y0, y1, x0, x1 = (max(int(q[1]) - 4, 0), min(int(q[1]) + 5, 2 * H), max(int(q[0]) - 4, 0),
                              min(int(q[0]) + 5, 2 * H))
            passable[y0:y1, x0:x1] |= np.hypot(jj[y0:y1, x0:x1] - q[0], ii[y0:y1, x0:x1] - q[1]) <= 3.0
    passable &= rg >= r - 1.0
    cost = np.where(passable, 1.0 / (Pc + EPS), np.inf)
    ring = np.argwhere((np.abs(rg - r) <= 0.5) & passable)
    out = {"n": 0}
    if not len(ring):
        gs.bins[b] = out
        return
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(p) for p in ring])
    # --- far ends of map pieces reached from the rim
    valid = np.isfinite(cum) & (Pc >= 0.5) & (rg >= r + 4)
    if valid.any():
        cm = np.where(valid, cum, -1.0).astype(np.float32)
        pk = (cm == cv2.dilate(cm, np.ones((11, 11), np.uint8))) & valid
        ys, xs = np.nonzero(pk)
        n_end = 0
        for o in np.argsort(-cm[ys, xs], kind="stable"):
            q = (float(xs[o]), float(ys[o]))
            if all(math.hypot(q[0] - a, q[1] - c) >= NMS for a, c in cand):
                cand.append(q)
                src.append(2)
                n_end += 1
            if n_end >= K_END:
                break
    fresh = np.array(cand, float).reshape(-1, 2)
    fresh_src = list(src)
    for q, sf in zip(carried, carried_src):
        if len(cand) >= MAX_C:
            break
        if all(math.hypot(q[0] - a, q[1] - c) >= NMS for a, c in cand):
            cand.append(q)
            src.append(sf)
    gs.recent.append((b, fresh + to_grain, fresh_src))
    gs.recent = [(bb, p, sr) for bb, p, sr in gs.recent if bb > b - W_CARRY]
    # --- bodies and features
    feats, bodies = [], []
    S = 2 * H
    for q, sflag in zip(cand, src):
        j, i = int(round(q[0])), int(round(q[1]))
        if not (0 <= i < S and 0 <= j < S):
            continue
        if not np.isfinite(cum[i, j]):
            y0, y1, x0, x1 = max(i - 2, 0), min(i + 3, S), max(j - 2, 0), min(j + 3, S)
            sub = cum[y0:y1, x0:x1]
            if not np.isfinite(sub).any():
                continue
            k = int(np.argmin(np.where(np.isfinite(sub), sub, np.inf)))
            i, j = y0 + k // sub.shape[1], x0 + k % sub.shape[1]
        path = np.asarray(mcp.traceback((i, j)), float)[:, ::-1]  # (col, row), rim first
        if len(path) >= 3:
            sm = np.stack([gaussian_filter1d(path[:, 0], 2.0, mode="nearest"),
                           gaussian_filter1d(path[:, 1], 2.0, mode="nearest")], 1)
            sm[0], sm[-1] = path[0], path[-1]
        else:
            sm = path
        line, s = resample(sm, 1.0)
        L = float(s[-1])
        k3 = min(len(line) - 1, 2)
        theta = math.atan2(line[k3][1] - gy, line[k3][0] - gx)
        edge = float(gs.edges[int(round(math.degrees(theta) % 360 / 5.0)) % 72])
        li = np.clip(np.round(line).astype(int), 0, S - 1)
        pv = Pc[li[:, 1], li[:, 0]]
        rad = np.hypot(line[:, 0] - gx, line[:, 1] - gy)
        far = rad > r + 3
        if far.any():
            sup = float(pv[far].mean())
            sup_frac = float((pv[far] >= 0.5).mean())
            low = np.concatenate([[0], (pv[far] < 0.3).astype(int), [0]])
            dl = np.diff(low)
            runs = np.flatnonzero(dl == -1) - np.flatnonzero(dl == 1)
            gap = float(runs.max()) if len(runs) else 0.0
            det_body = float(Db[li[far, 1], li[far, 0]].mean()) if Db is not None else 0.0
        else:
            sup = sup_frac = float(Pc[i, j])
            gap = 0.0
            det_body = float(Db[i, j]) if Db is not None else 0.0
        if Dt is not None:
            dsub = Dt[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
            dval = float(dsub.max()) if dsub.size else 0.0
        else:
            dval = 0.0
        if len(line) >= 3:
            back = line[max(0, len(line) - 6)]
            u = line[-1] - back
            nu = math.hypot(*u)
            u = u / nu if nu > 1e-6 else np.array([line[-1][0] - gx, line[-1][1] - gy]) / max(rad[-1], 1e-6)
        else:
            u = np.array([line[-1][0] - gx, line[-1][1] - gy]) / max(rad[-1], 1e-6)
        ahead = 0.0
        for d in (3.0, 5.0, 7.0):
            a = np.round(line[-1] + d * u).astype(int)
            if 0 <= a[0] < S and 0 <= a[1] < S:
                ahead = max(ahead, float(Pc[a[1], a[0]]))
        k8 = min(len(line) - 1, 8)
        radial = float((rad[k8] - rad[0]) / max(s[k8], 1e-6)) if k8 > 0 else 1.0
        ext = 0.0  # how far the map's band goes on past the tip along the tube's end direction (P >= 0.5)
        for d in np.arange(0.5, 8.01, 0.5):
            a = line[-1] + d * u
            x0, y0 = int(math.floor(a[0])), int(math.floor(a[1]))
            if not (0 <= x0 < S - 1 and 0 <= y0 < S - 1):
                break
            fx, fy = a[0] - x0, a[1] - y0
            v = (Pc[y0, x0] * (1 - fx) * (1 - fy) + Pc[y0, x0 + 1] * fx * (1 - fy) + Pc[y0 + 1, x0] * (1 - fx) * fy
                 + Pc[y0 + 1, x0 + 1] * fx * fy)
            if v < 0.5:
                break
            ext = float(d)
        feats.append([L, theta, edge, sup, sup_frac, gap, dval, det_body, ahead, radial, float(rad[-1] - r),
                      float(cum[i, j]), float(sflag), ext])
        bodies.append(line + to_grain)  # 1 px, grain frame (reference coordinates less the drift)
    if not feats:
        gs.bins[b] = out
        return
    F = np.asarray(feats, np.float32)
    tips = np.array([bd[-1] for bd in bodies])
    samples = []  # up to 8 points along each body from 3 px on (the rim end wanders)
    for bd in bodies:
        q = bd[3:] if len(bd) > 4 else bd[-1:]
        samples.append(q[np.linspace(0, len(q) - 1, min(8, len(q))).round().astype(int)])
    trees = [cKDTree(bd) for bd in bodies]
    # --- body agreement with the last bin that had candidates: for each pair whose length change is in bounds,
    # distance from the shorter body's tip to the longer body, and mean distance of its samples to it
    pairs = []
    if gs.prev is not None:
        pb, Lp, tips_p, samples_p, trees_p = gs.prev
        Lc = F[:, 0]
        dL = Lc[None, :] - Lp[:, None]  # (n_prev, n_cur)
        okp = (dL >= -PAIR_SHRINK) & (dL <= PAIR_GROW * (b - pb))
        if okp.any():
            sp_all = np.concatenate(samples_p)
            sp_idx = np.repeat(np.arange(len(samples_p)), [len(q) for q in samples_p])
            sc_all = np.concatenate(samples)
            sc_idx = np.repeat(np.arange(len(samples)), [len(q) for q in samples])
            d_pc_tip = np.zeros(okp.shape)
            d_pc_sh = np.zeros(okp.shape)
            for k in np.flatnonzero(okp.any(axis=0)):  # previous bodies against current body k
                d_pc_tip[:, k] = trees[k].query(tips_p)[0]
                ds = trees[k].query(sp_all)[0]
                d_pc_sh[:, k] = np.bincount(sp_idx, ds, len(samples_p)) / np.bincount(sp_idx, None, len(samples_p))
            d_cp_tip = np.zeros(okp.shape)
            d_cp_sh = np.zeros(okp.shape)
            for j in np.flatnonzero(okp.any(axis=1)):  # current bodies against previous body j
                d_cp_tip[j, :] = trees_p[j].query(tips)[0]
                ds = trees_p[j].query(sc_all)[0]
                d_cp_sh[j, :] = np.bincount(sc_idx, ds, len(samples)) / np.bincount(sc_idx, None, len(samples))
            for j, k in zip(*np.nonzero(okp)):
                if dL[j, k] >= 0:
                    pairs.append((j, k, d_pc_tip[j, k], d_pc_sh[j, k]))
                else:
                    pairs.append((j, k, d_cp_tip[j, k], d_cp_sh[j, k]))
    out = {"n": len(F), "F": F, "tips": tips.astype(np.float32),
           "bodies": [np.round(resample(bd, 2.0)[0] * 10).astype(np.int32) for bd in bodies],
           "pairs": np.asarray(pairs, np.float32).reshape(-1, 4),
           "pairs_with": None if gs.prev is None else gs.prev[0], "rim_det": rim_det}
    gs.bins[b] = out
    gs.prev = (b, F[:, 0].copy(), tips, samples, trees)


def run(movie: str, only=None, log=print) -> None:
    bins, meta = stack.load(cache_dir(movie))
    R = Renderer(bins, meta)
    prob = np.load(cache_dir(movie) / PROB / "bins.npy", mmap_mode="r")
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    H = HALF[movie]
    lab, base = labels(movie), baseline(movie)
    by = {g["id"]: g for g in base["grains"]}
    states = []
    for gid in scored_grains(lab):
        if only and gid not in only:
            continue
        g = lab["grains"][gid]
        drift = drift_per_bin(by[gid], rs, nb) if gid in by else np.zeros((nb, 2))
        cx, cy = g["x"] + drift[rs][0], g["y"] + drift[rs][1]
        early = np.mean([np.nan_to_num(R.crop(k, cx, cy, 64)) for k in range(rs, rs + 3)], axis=0)
        edges = np.array([exit_edge(early, 63.5, float(g["r"]), math.radians(5.0 * k)) for k in range(72)])
        states.append(GrainState(gid, g, drift, edges))
    S = 2 * H
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    det_dir = OUT / "det" / movie
    t0 = time.time()
    for b in range(rs, nb):
        P = np.array(prob[b])
        f = det_dir / f"b{b:03d}.npz"
        det = dict(np.load(f)) if f.exists() else None
        for gs in states:
            process(gs, b, P, det, H, (jj, ii))
        if b % 25 == 0:
            nc = np.mean([gs.bins[b]["n"] for gs in states])
            log(f"{movie} b{b}: {time.time() - t0:.0f} s, {nc:.1f} candidates per grain", flush=True)
    doc = {"movie": movie, "rs": rs, "nb": nb, "fpb": int(meta["frames_per_bin"]), "feats": FEATS,
           "params": {"EPS": EPS, "P_LO": P_LO, "DIL": DIL, "RIM_ZONE": RIM_ZONE, "K_DET": K_DET, "K_NEAR": K_NEAR, "NEAR_PX": NEAR_PX, "DET_MIN": DET_MIN,
                      "K_END": K_END, "W_CARRY": W_CARRY, "NMS": NMS, "MAX_C": MAX_C, "PAIR_SHRINK": PAIR_SHRINK, "PAIR_GROW": PAIR_GROW, "HALF": H},
           "grains": {gs.gid: {"x": gs.x, "y": gs.y, "r": gs.r, "drift": gs.drift, "edges": gs.edges, "bins": gs.bins}
                      for gs in states}}
    name = f"cands_{movie}.pkl" if not only else f"cands_{movie}_{'_'.join(only)}.pkl"
    with open(OUT / name, "wb") as fh:
        pickle.dump(doc, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"{movie}: {len(states)} grains in {time.time() - t0:.0f} s -> {OUT / name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movies", nargs="+")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for m in a.movies:
        run(m, a.only)
