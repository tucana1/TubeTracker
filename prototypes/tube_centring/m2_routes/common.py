"""Shared helpers for the m2 route diagnosis (read-only on the repo)."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402
from sparsetrack.report import turned_path  # noqa: E402
from tubetracker.app.overlay import to_length  # noqa: E402

REPO = "/Users/joshjiang/Documents/TubeTracker/"
SP = "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/"
OUT = SP + "m2_routes/"
PRED = json.load(open(SP + "tipfix/bench/m2_real_0/predictions.json"))
G = {g["id"]: g for g in PRED["grains"]}
L = json.load(open(REPO + "benchmark/labels/m2_v1.json"))
CENSUS = L["grains"]
BAD = [(g, int(b), f) for g, b, f in json.load(open(SP + "m2_bad_traces.json"))]
bins, meta = stack.load(REPO + "runs/sparsetrack/m2")
R = Renderer(bins, meta)
pbins, pmeta = stack.load(REPO + "runs/sparsetrack/m2/prob_tubes_bn_real_ld_m2")
PR = Renderer(pbins, pmeta)
FPB = 300


def idx(g, b):
    return int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * FPB + FPB // 2))))


def length_at(g, b):
    return float(g["length"]["px"][idx(g, b)])


def drift_at(g, b):
    d = (g.get("drift") or {}).get("xy") or []
    i = idx(g, b)
    return np.asarray(d[i], float) if i < len(d) else np.zeros(2)


def drawn(g, b, cut=True, drift=True):
    """The model's tube as the app draws it at bin b (cut=False: the whole route, bent/turned/drifted the same)."""
    i = idx(g, b)
    Lb = g["length"]["px"][i]
    if not g.get("path"):
        return None
    r = turned_path(g, i, PRED)
    if cut:
        if Lb <= 0.5:
            return None
        r = np.asarray(to_length(r.tolist(), Lb), float)
    if drift and g.get("drift"):
        r = r + drift_at(g, b)
    return np.asarray(r, float)


def trace(gid, b):
    t = (L["labels"].get(gid, {}).get("traces") or {}).get(str(b))
    if not t or len(t.get("path_xy_ref") or []) < 2:
        return None, None
    return np.asarray(t["path_xy_ref"], float), t


def resample(path, step=1.0):
    p = np.asarray(path, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return p[:1], np.zeros(1)
    q = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1), q


def arclen(path):
    p = np.asarray(path, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def pdist(pts, poly):
    """Distance from each point to the polyline."""
    pts = np.asarray(pts, float)
    poly = np.asarray(poly, float)
    if len(poly) == 1:
        return np.hypot(*(pts - poly[0]).T)
    a, c = poly[:-1], poly[1:]
    d = c - a
    dd = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((pts[:, None, :] - a[None]) * d[None]).sum(2) / dd[None], 0, 1)
    proj = a[None] + t[..., None] * d[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(2)).min(1)


def to_u8(img, lo=None, hi=None):
    fin = img[np.isfinite(img)]
    if lo is None:
        lo, hi = np.percentile(fin, [1, 99.5])
    return np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


class Tile:
    """A crop of the registered field (or the network's map) at bin b around (cx, cy), scaled to `size` px."""

    def __init__(self, b, cx, cy, half, size=360, prob=False, lohi=None, b0=None, b1=None):
        self.cx, self.cy, self.half, self.size = cx, cy, half, size
        b0 = b if b0 is None else b0
        b1 = b if b1 is None else b1
        if prob:
            img = PR.mean_crop(b0, b1, cx, cy, half) / 250.0
            u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img = R.mean_crop(b0, b1, cx, cy, half)
            u8 = to_u8(img, *(lohi or (None, None)))
        self.raw = img
        big = cv2.resize(u8, (size, size), interpolation=cv2.INTER_CUBIC)
        self.im = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        self.Z = size / (2 * half)

    def to_c(self, q):
        q = np.asarray(q, float).reshape(-1, 2)
        return np.round((q - [self.cx - self.half, self.cy - self.half]) * self.Z * 4).astype(np.int32)

    def line(self, q, col, th=2, closed=False):
        if q is None or len(q) < 2:
            return
        cv2.polylines(self.im, [self.to_c(q).reshape(-1, 1, 2)], closed, col, th, cv2.LINE_AA, shift=2)

    def dashed(self, q, col, th=1, on=4.0, off=3.0):
        if q is None or len(q) < 2:
            return
        pts, s = resample(q, 0.5)
        per = on + off
        seg = []
        for p, ss in zip(pts, s):
            if (ss % per) < on:
                seg.append(p)
            elif seg:
                self.line(np.asarray(seg), col, th)
                seg = []
        if len(seg) > 1:
            self.line(np.asarray(seg), col, th)

    def dot(self, p, col, r=3, filled=True):
        c = self.to_c(p)[0]
        cv2.circle(self.im, tuple(int(v) for v in c), r * 4, col, -1 if filled else 1, cv2.LINE_AA, shift=2)

    def circle(self, p, rad, col, th=1):
        c = self.to_c(p)[0]
        cv2.circle(self.im, tuple(int(v) for v in c), int(round(rad * self.Z * 4)), col, th, cv2.LINE_AA, shift=2)

    def text(self, s, y=16, x=5, scale=0.45, col=(255, 255, 255)):
        cv2.putText(self.im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(self.im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)

    def text_at(self, p, s, col=(255, 255, 255), scale=0.38):
        c = self.to_c(p)[0] / 4
        x, y = int(c[0]) + 4, int(c[1]) - 4
        cv2.putText(self.im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(self.im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


GREEN = (60, 200, 60)
MAGENTA = (220, 60, 220)
PINK_DIM = (160, 90, 160)
CYAN = (230, 200, 40)
YELLOW = (40, 210, 230)
ORANGE = (40, 140, 255)
WHITE = (255, 255, 255)
RED = (60, 60, 230)


def pad(im, w=2, col=255):
    return np.pad(im, ((w, w), (w, w), (0, 0)), constant_values=col)


def grid(tiles, ncol):
    tiles = [pad(t) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    while len(tiles) % ncol:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[k:k + ncol]) for k in range(0, len(tiles), ncol)]
    return np.vstack(rows)
