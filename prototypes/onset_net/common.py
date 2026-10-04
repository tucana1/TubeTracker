"""Shared pieces of the grain-aware onset network study: movies, labels per bin, grain-centred inputs, the network,
the change-point decoder and the onset scoring.

Inputs for grain g at bin b (crops centred on where the grain is AT EACH BIN, so the grain itself is still; all bins
registered; M(k) = mean of the grain-centred crops of bins k-1..k+1):
  A = M(b) - local median                      appearance
  D = M(b) - M(b-6), less its local median     short-interval difference
  C = M(b) - E, less its local median          change from the first (reference) bins (E = mean of bins rs..rs+2)
  G = the grain's own disc (census radius) at the crop centre: WHICH grain is judged
Scales as the tip detector (20, 8, 20 grey levels per unit); background = the crop's median.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/research/onset_net"
OUT.mkdir(parents=True, exist_ok=True)
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
SCRATCH = Path("/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/"
               "scratchpad")
BASE088 = SCRATCH / "bt/base088"                       # 0.8.8 readings (read only)
TIPDET_LATER = SCRATCH / "td/bench_later_young"        # 0.8.8 + tip detector `later` + young, end to end (read only)
HALF = 40          # stored crop half-size (80 px)
NET_HALF = 36      # network input half-size (72 px)
K = 6
SCALE = np.array([20.0, 8.0, 20.0], np.float32)
TOL = 600.0        # onset tolerance (frames), as sparsetrack.evaluate.score
FOLD_TIPDET = {"ld": "tip2_m2m1.pt", "m2": "tip2_ldm1.pt", "m1": "tip2_ldm2.pt"}


def labels(movie: str) -> dict:
    return json.loads((REPO / MOVIES[movie][1]).read_text())


def baseline(movie: str, root: Path = BASE088) -> dict:
    return json.loads((root / f"{movie}_real_0" / "predictions.json").read_text())


def positions(res: dict, rs: int, nb: int) -> np.ndarray:
    """(nb, 2) reference position of a reading's grain per bin: its place + its drift (held before the start)."""
    pos = np.tile(np.array([res["x"], res["y"]], float), (nb, 1))
    if res.get("drift"):
        d = np.nan_to_num(np.asarray(res["drift"]["xy"], float))
        pos[rs:rs + len(d)] += d
        pos[:rs] += d[0]
    return pos


def extract(renderer, pos: np.ndarray, half: int = HALF) -> np.ndarray:
    """(nb, 2h, 2h) float16 registered crops of every bin, each centred on the grain's position at that bin."""
    nb = renderer.n_bins
    out = np.zeros((nb, 2 * half, 2 * half), np.float16)
    for b in range(nb):
        c = renderer.crop(b, float(pos[b][0]), float(pos[b][1]), half)
        if not np.isfinite(c).all():
            c = np.nan_to_num(c, nan=float(np.nanmedian(c)) if np.isfinite(c).any() else 0.0)
        out[b] = c.astype(np.float16)
    return out


# ------------------------------------------------------------------------------------------------ labels per bin
def bin_labels(verdict: str | None, la: int | None, fv: int | None, nb: int) -> np.ndarray:
    """Per bin: 1 tube visible, 0 not, -1 unlabelled (inside the bracket, or no usable verdict)."""
    y = np.full(nb, -1, np.int8)
    if verdict == "emerged_within" and fv is not None:
        la = fv - 1 if la is None else la
        y[:la + 1] = 0
        y[fv:] = 1
    elif verdict == "emerged_at_start":
        y[:] = 1
    elif verdict in ("no_emergence_by_end", "debris"):
        y[:] = 0
    return y


# ------------------------------------------------------------------------------------------------ inputs
def disc(r: float, half: int = NET_HALF, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    j = np.arange(2 * half, dtype=np.float32) - half + 0.5
    d = np.hypot(j[None, :] - cx, j[:, None] - cy)
    return np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)


def window(nb: int, b: int, rs: int) -> tuple[list[int], list[int]]:
    """The bins whose crops make M(b) and M(b-6) (clamped to rs .. nb-1)."""
    m = [k for k in range(max(b - 1, rs), min(b + 1, nb - 1) + 1)]
    p0 = max(b - K - 1, rs)
    p = [k for k in range(p0, min(p0 + 2, nb - 1) + 1)]
    return m, p


def medbg(img: np.ndarray) -> np.ndarray:
    """The crop's background level: its median (on an 80 px crop the tip detector's ~60 px local median is nearly
    the crop's own median; this is 20x faster and the same in training and use)."""
    return np.float32(np.median(img))


def adc(M: np.ndarray, P: np.ndarray, E: np.ndarray) -> np.ndarray:
    A = M - medbg(M)
    D = M - P
    D = D - medbg(D)
    C = M - E
    C = C - medbg(C)
    return np.stack([A, D, C]) / SCALE[:, None, None]


def inputs_at(crops: np.ndarray, b: int, rs: int, E: np.ndarray | None = None) -> np.ndarray:
    """(3, 2H, 2H) A, D, C of one grain at bin b from its grain-centred crops (no augmentation)."""
    if E is None:
        E = crops[rs:rs + 3].astype(np.float32).mean(0)
    m, p = window(len(crops), b, rs)
    M = crops[m].astype(np.float32).mean(0)
    P = crops[p].astype(np.float32).mean(0)
    return adc(M, P, E)


def centre_cut(x: np.ndarray, half: int = NET_HALF, dx: int = 0, dy: int = 0) -> np.ndarray:
    H = x.shape[-1] // 2
    return x[..., H - half + dy:H + half + dy, H - half + dx:H + half + dx]


# ------------------------------------------------------------------------------------------------ network
def make_net(in_ch: int = 4, widths=(16, 32, 64, 64), drop: float = 0.3):
    import torch
    from torch import nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class OnsetNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c = [], in_ch
            for i, w in enumerate(widths):
                layers.append(block(c, w))
                if i < len(widths) - 1:
                    layers.append(nn.MaxPool2d(2))
                c = w
            self.body = nn.Sequential(*layers)
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(2 * c, 1)

        def forward(self, x):
            f = self.body(x)
            f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
            return self.fc(self.drop(f))[:, 0]

    return OnsetNet()


def load_net(path, device=None):
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net = make_net(ck["in_ch"], tuple(ck["widths"]), ck.get("drop", 0.3))
    net.load_state_dict(ck["state"])
    net.cfg = ck
    return net.eval().to(device)


def dihedral(x: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 rotations/flips of (..., H, W)."""
    y = np.rot90(x, k % 4, axes=(-2, -1))
    return np.flip(y, -1) if k >= 4 else y


# ------------------------------------------------------------------------------------------------ decoding
def changepoint(p: np.ndarray, valid_from: int, eps: float = 0.02) -> tuple[int | None, np.ndarray]:
    """The single switch 0 -> 1 that best explains per-bin probabilities p (bins >= valid_from). Returns the first
    visible bin (valid_from = visible from the start; None = never) and the log-likelihood per candidate
    (index t - valid_from; the last entry = never)."""
    q = np.clip(np.asarray(p, float)[valid_from:], eps, 1 - eps)
    l1, l0 = np.log(q), np.log(1 - q)
    n = len(q)
    # LL(t) = sum_{b<t} l0 + sum_{b>=t} l1, t = 0..n (t = n: never)
    c0 = np.concatenate([[0.0], np.cumsum(l0)])
    c1 = np.concatenate([[0.0], np.cumsum(l1)])
    ll = c0 + (c1[-1] - c1)
    t = int(np.argmax(ll))
    return (None if t == n else valid_from + t), ll


def onset_prediction(res: dict, first_visible: int | None, valid_from: int, fpb: int) -> dict:
    """A copy of a 0.8.8 grain reading with the onset replaced (status, onset_frame, onset_interval)."""
    out = dict(res)
    if first_visible is None:
        out.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None)
    elif first_visible <= valid_from:
        out.update(status="emerged_at_start", onset_frame=None, onset_interval=None)
    else:
        f = first_visible * fpb + fpb // 2
        out.update(status="emerged_within", onset_frame=int(f), onset_interval=[int(f - fpb), int(f)])
    return out


# ------------------------------------------------------------------------------------------------ scoring
def onset_error(lab: dict, frame: int | None) -> float | None:
    from sparsetrack.evaluate import interval_distance
    if frame is None:
        return None
    on = lab["onset"]
    return interval_distance(frame, on.get("last_absent_frame"), on.get("first_visible_frame"))


def scored_grains(L: dict) -> dict:
    """The scorer's grains (isolated, not excluded) by human verdict."""
    out: dict = {"emerged_within": [], "emerged_at_start": [], "no_emergence_by_end": [], "unobservable": []}
    for gid, g in sorted(L["grains"].items()):
        if g.get("excluded") or not g.get("isolated", True):
            continue
        v = ((L["labels"].get(gid) or {}).get("onset") or {}).get("verdict")
        if v in out:
            out[v].append(gid)
    return out


def hit_vector(L: dict, preds: dict, gids: list[str]) -> np.ndarray:
    """1 where the grain's predicted onset is within TOL of the human bracket (a grain called never or at start
    counts as a miss), over human 'emerged within' grains."""
    out = []
    for g in gids:
        p = preds.get(g)
        f = p.get("onset_frame") if p and p.get("status") == "emerged_within" else None
        e = onset_error(L["labels"][g], f)
        out.append(int(e is not None and abs(e) <= TOL))
    return np.asarray(out)


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the summed paired difference over grains."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def auroc(neg, pos) -> float:
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def angle_of(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))
