"""Minimal paths on a tube-probability map: plain (x, y) geodesics and bending-penalised geodesics lifted to
(x, y, direction), as in orientation-score tracking (Bekkers et al. 2015) and curvature-penalised minimal paths.

Cost per px along direction theta at (x, y): 1 / (eps + R_theta(x, y)), R_theta = the map's mean along a short
line through (x, y) at theta (oriented evidence; or the map itself for "isotropic" evidence); plus w per radian
turned (total turning, an L1 curvature penalty). Directions come from a grid stencil (16 or 32 directions) so every
move lands on a pixel; a node may turn by one stencil step (two for 32) before each move.
"""
import math

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

STENCIL = {
    16: [(1, 0), (2, 1), (1, 1), (1, 2)],
    32: [(1, 0), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3)],
}


def stencil(K):
    q = STENCIL[K]
    out = []
    for r in range(4):  # rotate the first quadrant by 90 degrees
        for dx, dy in q:
            for _ in range(r):
                dx, dy = -dy, dx
            out.append((dx, dy))
    th = np.array([math.atan2(dy, dx) for dx, dy in out])
    return np.array(out, int), th


def oriented(P, th, length=13.0, sigma=1.0):
    """Mean of P along a line of ``length`` px through each pixel at each direction (a Gaussian of ``sigma`` across)."""
    r = int(math.ceil(length / 2 + 3 * sigma))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    out = np.empty((len(th),) + P.shape, np.float32)
    for k, t in enumerate(th):
        along = xx * math.cos(t) + yy * math.sin(t)
        across = -xx * math.sin(t) + yy * math.cos(t)
        ker = np.exp(-0.5 * (across / sigma) ** 2) * np.clip(length / 2 + 0.5 - np.abs(along), 0, 1)
        ker /= ker.sum()
        out[k] = cv2.filter2D(P.astype(np.float32), -1, ker, borderType=cv2.BORDER_CONSTANT)
    return out


def plain_graph(P, eps=0.05, blocked=None):
    """8-connected graph, edge cost = step length x mean of 1/(eps + P) at its ends."""
    h, w = P.shape
    c = 1.0 / (eps + P.astype(np.float64))
    if blocked is not None:
        c = np.where(blocked, np.inf, c)
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = idx[y0:y1, x0:x1].ravel()
        b = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        v = math.hypot(dx, dy) * 0.5 * (c[y0:y1, x0:x1].ravel() + c[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel())
        ok = np.isfinite(v)
        rows.append(a[ok]); cols.append(b[ok]); vals.append(v[ok])
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))


def lifted_graph(P, K=16, w_bend=10.0, eps=0.05, evidence="oriented", length=13.0, max_turn=None, blocked=None):
    """Directed graph on (k, y, x): from (k, y, x) turn to k' (|k' - k| <= max_turn stencil steps) and move one
    stencil step d_k'. Cost = |d| x mean(c_k'(start), c_k'(mid), c_k'(end)) + w_bend x |turn| (radians)."""
    h, w = P.shape
    d, th = stencil(K)
    if max_turn is None:
        max_turn = 1 if K == 16 else 2
    R = oriented(P, th, length) if evidence == "oriented" else np.repeat(P[None].astype(np.float32), K, 0)
    c = 1.0 / (eps + R.astype(np.float64))
    if blocked is not None:
        c = np.where(blocked[None], np.inf, c)
    N = h * w
    idx = np.arange(N, dtype=np.int32).reshape(h, w)
    rows, cols, vals = [], [], []
    for k2 in range(K):
        dx, dy = d[k2]
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = idx[y0:y1, x0:x1].ravel()
        b = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        cs = c[k2, y0:y1, x0:x1].ravel()
        ce = c[k2, y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        # midpoint: mean of the two pixels nearest the half step
        my0, mx0 = int(math.floor(dy / 2)), int(math.floor(dx / 2))
        my1, mx1 = int(math.ceil(dy / 2)), int(math.ceil(dx / 2))
        cm = 0.5 * (c[k2, y0 + my0:y1 + my0, x0 + mx0:x1 + mx0].ravel() + c[k2, y0 + my1:y1 + my1, x0 + mx1:x1 + mx1].ravel())
        move = math.hypot(dx, dy) * (cs + cm + ce) / 3.0
        ok = np.isfinite(move)
        for t in range(-max_turn, max_turn + 1):
            k1 = (k2 - t) % K
            turn = abs(math.remainder(th[k2] - th[k1], 2 * math.pi))
            rows.append((k1 * N + a[ok]).astype(np.int32))
            cols.append((k2 * N + b[ok]).astype(np.int32))
            vals.append((move[ok] + w_bend * turn).astype(np.float32))
    r_, c_, v_ = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    del rows, cols, vals
    return csr_matrix((v_, (r_, c_)), shape=(K * N, K * N)), th


def run(G, sources, return_pred=True):
    if return_pred:
        dist, pred, src = dijkstra(G, directed=True, indices=np.asarray(sources), min_only=True,
                                   return_predecessors=True)
        return dist, pred
    return dijkstra(G, directed=True, indices=np.asarray(sources), min_only=True), None


def backtrack(pred, node):
    out = [node]
    while pred[out[-1]] >= 0:
        out.append(pred[out[-1]])
    return out[::-1]


def path_xy(nodes, shape):
    h, w = shape
    N = h * w
    pix = np.asarray(nodes) % N
    return np.stack([pix % w, pix // w], 1).astype(float)  # (x, y)


def path_k(nodes, shape):
    return np.asarray(nodes) // (shape[0] * shape[1])


def best_k(dist, pix, K, N):
    """The direction layer with the least distance at pixel index ``pix``."""
    v = dist[np.arange(K) * N + pix]
    k = int(np.argmin(v))
    return k, float(v[k])
