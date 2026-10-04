"""The global tip-trajectory reader (research option, off by default; prototypes/tip_trajectory/README.md).

Per grain, the tube's tip at every bin is chosen in one Viterbi over the whole movie instead of grown bin by bin:

- candidates per bin (where the grain is then: census + the reading's drift): the learned tip detector's strongest
  peaks (prototypes/tip_detector), the far ends of the tube map's pieces reached from the rim, and the fresh
  candidates of the last few bins carried with the grain; each with its body, the cheapest route on that bin's tube
  map (cost 1 / (P + 0.1)) from the grain's rim ring to it;
- states: not germinated, a candidate, or "held" (the tube keeps its last reading); costs: evidence at the tip and
  along the body, a germination that happens once, a length that grows at most ``vmax`` a bin and shrinks little,
  and a body that agrees with the previous bin's (the shorter body's tip and its points lie on the longer body), so
  that jumping onto another tube is ruled out;
- length = body arc length from the rim less the grain's visible edge there (``analyze.exit_edge``) less a tip offset,
  plus part of the map's band beyond the tip, made non-decreasing from the onset.

The detector runs on the whole registered frame of every bin once per movie (``det_cache``, compressed, ~120 KB a
bin). The reader takes over a reading made by the other readers (``read``): it keeps that reading's drift and replaces
its germination call, onset, lengths, tips and route.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .render import Renderer

# candidates and bodies
EPS = 0.1          # cost of a pixel = 1 / (P + EPS)
P_LO = 0.2         # passable: within DIL px of P >= P_LO...
DIL = 4
RIM_ZONE = 25.0    # ...or within this beyond the rim (young tubes the map misses); never inside the grain
K_DET = 6          # detector peaks per bin (>= DET_MIN), beyond r - 2...
K_NEAR = 3         # ...after the strongest K_NEAR within NEAR_PX of the rim
NEAR_PX = 30.0
DET_MIN = 0.05
K_END = 6          # far ends of map pieces per bin
W_CARRY = 8        # fresh candidates of this many bins are carried on
NMS = 3.0
MAX_C = 32
PAIR_SHRINK, PAIR_GROW = 8.0, 16.0  # pairs of consecutive candidates kept: length change -8 .. +16 px a bin
DETECTOR_K = 6     # the detector's short interval (bins)
DET_SCALE = np.array([20.0, 8.0, 20.0], np.float32)
P_SCALE_MAP = 250.0   # tube maps are uint8 P x this (learned.P_SCALE)

# the Viterbi's settings: tuned on the sparse movie (ld) alone and applied unchanged to movies 2 and 1
# (prototypes/tip_trajectory, tune_ld_c; override with Params.tiptraj_weights)
WEIGHTS = {"c": -1.0001, "theta": 0.443, "w_det": 0.1571, "w_sup": 2.0, "w_gap": 0.5081, "w_ahead": 0.2608,
           "w_tan": 1.0459, "w_carry": 0.2216, "w_on": 2.0348, "l0": 5.0, "w_l0": 0.0, "l0_max": 60.0, "vmax": 4.4387,
           "shrink": 4.1388, "w_shrink": 0.6467, "w_cons": 0.8198, "w_share": 0.4721, "cap_tip": 5.1802,
           "cap_share": 6.2131, "iso": True, "w_ext": 0.5007, "c_end": -1.4142, "w_hold": 0.0051, "w_reacq": 2.7783,
           "cap_reacq": 8.3469, "det_norm": 2, "edge_clip": 3.3209}


# ----------------------------------------------------------------------------- the detector's maps
def _medbg(img: np.ndarray, f: int = 4, k: int = 15) -> np.ndarray:
    from scipy.ndimage import median_filter
    h, w = img.shape
    small = cv2.resize(img, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(median_filter(small, size=k, mode="reflect"), (w, h), interpolation=cv2.INTER_LINEAR)


def load_detector(path, device: str | None = None):
    import torch
    from .learned import _unet
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ck.get("channels", "ADC") != "ADC" or ck.get("out_ch", 2) != 2:
        raise ValueError(f"{path}: expected a tip detector with inputs ADC and two outputs")
    net = _unet(tuple(ck.get("widths", (16, 32, 64, 128))), "batch")
    net.load_state_dict(ck["state"])
    return net.eval().to(device)


def _tiles(net, x: np.ndarray, n: int = 2, ctx: int = 64) -> np.ndarray:
    import torch
    _, H, W = x.shape
    out = np.zeros((H, W), np.float32)
    dev = next(net.parameters()).device
    ys, xs = np.linspace(0, H, n + 1).astype(int), np.linspace(0, W, n + 1).astype(int)
    for i in range(n):
        for j in range(n):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            a0, a1, b0, b1 = max(0, y0 - ctx), min(H, y1 + ctx), max(0, x0 - ctx), min(W, x1 + ctx)
            t = x[:, a0:a1, b0:b1]
            h, w = t.shape[1:]
            t = np.pad(t, ((0, 0), (0, (-h) % 8), (0, (-w) % 8)), mode="reflect")
            with torch.no_grad():
                y = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(t))[None].to(dev)))[0, 0].cpu().numpy()
            out[y0:y1, x0:x1] = y[y0 - a0:y1 - a0, x0 - b0:x1 - b0]
    return out


def det_cache(cache_dir: str | Path, model: str | Path, log=print) -> Path:
    """The tip detector's map of every bin on the whole registered frame (inputs as prototypes/tip_detector: the
    3-bin mean, its change over ``DETECTOR_K`` bins and from the first bins, each less its ~60 px local median), stored
    as uint8 P x 250 (below 3 set to 0), compressed, one file per bin: ``<cache>/det_<model stem>/b<bin>.npz``. Bins
    without inputs (the first 7 after the reference start, the last) are not written."""
    from . import stack
    cache_dir = Path(cache_dir)
    out = cache_dir / f"det_{Path(model).stem}"
    if (out / "meta.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    bins, meta = stack.load(cache_dir)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    H, W = bins.shape[1:]
    net = load_detector(model)
    reg, mean = {}, {}

    def R(b):
        if b not in reg:
            dx, dy = shifts[b]
            reg[b] = cv2.warpAffine(np.asarray(bins[b], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return reg[b]

    def M(b):
        if b not in mean:
            mean[b] = np.mean([R(k) for k in range(max(b - 1, rs), min(b + 1, nb - 1) + 1)], axis=0)
        return mean[b]

    E = np.mean([R(k) for k in range(rs, rs + 3)], axis=0)
    started = time.time()
    for b in range(rs + DETECTOR_K + 1, nb - 1):
        for k in [k for k in reg if k < b - DETECTOR_K - 2]:
            del reg[k]
        for k in [k for k in mean if k < b - DETECTOR_K - 1]:
            del mean[k]
        A = M(b) - _medbg(M(b))
        D = M(b) - M(b - DETECTOR_K)
        C = M(b) - E
        x = (np.stack([A, D - _medbg(D), C - _medbg(C)]) / DET_SCALE[:, None, None]).astype(np.float32)
        q = np.clip(np.round(_tiles(net, x) * 250.0), 0, 250).astype(np.uint8)
        q[q < 3] = 0
        np.savez_compressed(out / f"b{b:03d}.npz", tip=q)
    (out / "meta.json").write_text(json.dumps({"model": str(model), "n_bins": nb}))
    log(f"tip detector maps ({Path(model).name}): {nb} bins in {time.time() - started:.0f} s -> {out}")
    return out


class DetMaps:
    """A movie's tip-detector maps (``det_cache``): ``tip(b)`` the full-frame uint8 map of bin b, or None."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._b, self._m = None, None

    def tip(self, b: int) -> np.ndarray | None:
        if b != self._b:
            f = self.path / f"b{b:03d}.npz"
            self._b, self._m = b, (np.load(f)["tip"] if f.exists() else None)
        return self._m

    def scale(self, every: int = 10) -> float:
        """The movie's detector scale without grains or labels: the 99th percentile of the map's local maxima
        (9 x 9, >= 0.05) over the whole frame, every ``every``-th bin (the detector answers more weakly on some
        movies than on others)."""
        f = self.path / "scale.json"
        if f.exists():
            return float(json.loads(f.read_text())["q99"])
        vals = []
        for g in sorted(self.path.glob("b*.npz"))[::every]:
            t = np.load(g)["tip"].astype(np.float32) / 250.0
            pk = (t == cv2.dilate(t, np.ones((9, 9), np.uint8))) & (t >= 0.05)
            vals.append(t[pk])
        v = np.concatenate(vals) if vals else np.zeros(0)
        q = float(np.percentile(v, 99)) if len(v) else 1.0
        try:
            f.write_text(json.dumps({"q99": q, "every": every}))
        except OSError:
            pass
        return q


# ----------------------------------------------------------------------------- candidates
def _crop(img, ix: int, iy: int, H: int) -> np.ndarray:
    """img[iy-H:iy+H, ix-H:ix+H] (a memmap is read only there) with zeros outside the frame."""
    h, w = img.shape[-2:]
    out = np.zeros((2 * H, 2 * H), np.uint8)
    y0, x0 = iy - H, ix - H
    a0, a1, b0, b1 = max(0, y0), min(h, iy + H), max(0, x0), min(w, ix + H)
    if a1 > a0 and b1 > b0:
        out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = img[a0:a1, b0:b1]
    return out


def _resample(line: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    if s[-1] <= 0:
        return line[:1].copy(), np.zeros(1)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - q[-1] > 1e-6:
        q = np.append(q, s[-1])
    return np.stack([np.interp(q, s, line[:, 0]), np.interp(q, s, line[:, 1])], 1), q


# guided second pass (Params.tiptraj_guided): the first reading's tube is followed along the map
EXT_R = 80            # extension candidates: along the map from the guide's tip, within this window (px)...
EXT_STEPS = (6.0, 12.0, 20.0, 30.0, 45.0, 60.0, 80.0)  # ...at these arc lengths on, and its far end
GUIDE_GAP = 60        # a guide reading older than this many bins is not used


def _extension(Pc: np.ndarray, q: np.ndarray, u: np.ndarray) -> list[tuple[float, float]]:
    """Points along the tube map beyond a tip ``q`` (crop index coordinates) in its direction ``u``: the cheapest
    route on 1 / (P + EPS) (within 2 px of P >= 0.35) from q to the farthest point of P >= 0.5 ahead (within 70 deg of
    u, EXT_R px window), sampled at EXT_STEPS px of arc length, and that far point."""
    from skimage.graph import MCP_Geometric
    S = Pc.shape[0]
    x0, x1 = max(int(q[0]) - EXT_R, 0), min(int(q[0]) + EXT_R + 1, S)
    y0, y1 = max(int(q[1]) - EXT_R, 0), min(int(q[1]) + EXT_R + 1, S)
    qi, qj = int(round(q[1])) - y0, int(round(q[0])) - x0
    sub = Pc[y0:y1, x0:x1]
    h, w = sub.shape
    if not (0 <= qi < h and 0 <= qj < w):
        return []
    pas = cv2.dilate((sub >= 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    pas[max(qi - 3, 0):qi + 4, max(qj - 3, 0):qj + 4] = True
    mcp = MCP_Geometric(np.where(pas, 1.0 / (sub + EPS), np.inf), fully_connected=True)
    cum, _ = mcp.find_costs([(qi, qj)])
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - qj, yy - qi
    dd = np.hypot(dx, dy)
    ok = np.isfinite(cum) & (sub >= 0.5) & (dd >= 3) & (dx * u[0] + dy * u[1] >= math.cos(math.radians(70)) * dd)
    if not ok.any():
        return []
    k = int(np.argmax(np.where(ok, cum, -1.0)))
    path = np.asarray(mcp.traceback((k // w, k % w)), float)[:, ::-1] + np.array([x0, y0])
    if len(path) < 2:
        return []
    sarc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out = [(float(np.interp(t, sarc, path[:, 0])), float(np.interp(t, sarc, path[:, 1]))) for t in EXT_STEPS
           if t < sarc[-1]]
    out.append((float(path[-1][0]), float(path[-1][1])))
    return out


# feature columns
L_ARC, THETA, EDGE, SUP, GAP, DET, AHEAD, RADIAL, SRC, EXT = range(10)


class _Track:
    """One grain's candidates, bin by bin."""

    def __init__(self, x: float, y: float, r: float, drift: np.ndarray, edges: np.ndarray, half: int,
                 guide: dict | None = None):
        self.x, self.y, self.r, self.drift, self.edges, self.H = x, y, r, drift, edges, half
        # guide (a first reading of this grain, second pass): {bin: body (n, 2), grain frame} of the bins it chose a
        # candidate at; adds its tip, points along the map beyond it, and a corridor along its body
        self.guide = guide
        self.guide_bins = np.array(sorted(guide)) if guide else np.zeros(0, int)
        S = 2 * half
        self.jj, self.ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
        self.recent: list = []
        self.prev = None
        self.bins: dict[int, dict] = {}

    def step(self, b: int, P, Dfull) -> None:
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial import cKDTree
        from skimage.graph import MCP_Geometric
        H, r, jj, ii = self.H, self.r, self.jj, self.ii
        S = 2 * H
        cx, cy = self.x + self.drift[b][0], self.y + self.drift[b][1]
        ix, iy = int(round(cx)), int(round(cy))
        gx, gy = cx - (ix - H) - 0.5, cy - (iy - H) - 0.5
        to_grain = np.array([ix - H + 0.5 - self.drift[b][0], iy - H + 0.5 - self.drift[b][1]])
        Pc = _crop(P, ix, iy, H).astype(np.float32) / 250.0
        Dt = _crop(Dfull, ix, iy, H).astype(np.float32) / 250.0 if Dfull is not None else None
        rg = np.hypot(jj - gx, ii - gy)
        cand, src = [], []
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
        carried = [(float(q[0]), float(q[1])) for _, pts, _ in reversed(self.recent) for q in pts - to_grain]
        carried_src = [4 + sf for _, _, srcs in reversed(self.recent) for sf in srcs]
        guided, guided_src, lane_line = [], [], None
        if self.guide:
            kg = int(np.searchsorted(self.guide_bins, b, side="right")) - 1
            if kg >= 0 and b - self.guide_bins[kg] <= GUIDE_GAP:
                body = np.asarray(self.guide[int(self.guide_bins[kg])], float) - to_grain
                q = body[-1]
                lane_line = body
                guided.append((float(q[0]), float(q[1])))
                guided_src.append(8)
                u = q - body[max(0, len(body) - 4)]
                nu = math.hypot(*u)
                if nu < 1e-6:
                    u = np.array([q[0] - gx, q[1] - gy])
                    nu = max(math.hypot(*u), 1e-6)
                for e in _extension(Pc, q, u / nu):
                    guided.append(e)
                    guided_src.append(7)
        passable = cv2.dilate((Pc >= P_LO).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1))) > 0
        passable |= rg <= r + RIM_ZONE
        if lane_line is not None and len(lane_line) >= 2:  # the guide's route, through gaps in this bin's map
            lane = np.zeros(passable.shape, np.uint8)
            cv2.polylines(lane, [np.round(lane_line).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness=5)
            passable |= lane > 0
        for q in cand + carried + guided:
            if -3 <= q[0] < S + 3 and -3 <= q[1] < S + 3:
                y0, y1, x0, x1 = max(int(q[1]) - 4, 0), min(int(q[1]) + 5, S), max(int(q[0]) - 4, 0), min(int(q[0]) + 5, S)
                passable[y0:y1, x0:x1] |= np.hypot(jj[y0:y1, x0:x1] - q[0], ii[y0:y1, x0:x1] - q[1]) <= 3.0
        passable &= rg >= r - 1.0
        ring = np.argwhere((np.abs(rg - r) <= 0.5) & passable)
        if not len(ring):
            self.bins[b] = {"n": 0}
            return
        mcp = MCP_Geometric(np.where(passable, 1.0 / (Pc + EPS), np.inf), fully_connected=True)
        cum, _ = mcp.find_costs([tuple(q) for q in ring])
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
        fresh, fresh_src = np.array(cand, float).reshape(-1, 2), list(src)
        for q, sf in zip(guided, guided_src):  # not carried on: the guide is there at every bin
            if all(math.hypot(q[0] - a, q[1] - c) >= NMS for a, c in cand):
                cand.append(q)
                src.append(sf)
        for q, sf in zip(carried, carried_src):
            if len(cand) >= MAX_C:
                break
            if all(math.hypot(q[0] - a, q[1] - c) >= NMS for a, c in cand):
                cand.append(q)
                src.append(sf)
        self.recent.append((b, fresh + to_grain, fresh_src))
        self.recent = [(bb, q, sr) for bb, q, sr in self.recent if bb > b - W_CARRY]
        feats, bodies = [], []
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
            path = np.asarray(mcp.traceback((i, j)), float)[:, ::-1]
            if len(path) >= 3:
                sm = np.stack([gaussian_filter1d(path[:, 0], 2.0, mode="nearest"),
                               gaussian_filter1d(path[:, 1], 2.0, mode="nearest")], 1)
                sm[0], sm[-1] = path[0], path[-1]
            else:
                sm = path
            line, s = _resample(sm, 1.0)
            k3 = min(len(line) - 1, 2)
            theta = math.atan2(line[k3][1] - gy, line[k3][0] - gx)
            edge = float(self.edges[int(round(math.degrees(theta) % 360 / 5.0)) % 72])
            li = np.clip(np.round(line).astype(int), 0, S - 1)
            pv = Pc[li[:, 1], li[:, 0]]
            rad = np.hypot(line[:, 0] - gx, line[:, 1] - gy)
            far = rad > r + 3
            if far.any():
                sup = float(pv[far].mean())
                low = np.concatenate([[0], (pv[far] < 0.3).astype(int), [0]])
                dl = np.diff(low)
                runs = np.flatnonzero(dl == -1) - np.flatnonzero(dl == 1)
                gap = float(runs.max()) if len(runs) else 0.0
            else:
                sup, gap = float(Pc[i, j]), 0.0
            dval = float(Dt[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3].max()) if Dt is not None else 0.0
            if len(line) >= 3:
                u = line[-1] - line[max(0, len(line) - 6)]
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
            feats.append([float(s[-1]), theta, edge, sup, gap, dval, ahead, radial, float(sflag), ext])
            bodies.append(line + to_grain)
        if not feats:
            self.bins[b] = {"n": 0}
            return
        F = np.asarray(feats, np.float32)
        tips = np.array([bd[-1] for bd in bodies])
        samples = []
        for bd in bodies:
            q = bd[3:] if len(bd) > 4 else bd[-1:]
            samples.append(q[np.linspace(0, len(q) - 1, min(8, len(q))).round().astype(int)])
        trees = [cKDTree(bd) for bd in bodies]
        pairs = []
        if self.prev is not None:
            pb, Lp, tips_p, samples_p, trees_p = self.prev
            dL = F[:, L_ARC][None, :] - Lp[:, None]
            okp = (dL >= -PAIR_SHRINK) & (dL <= PAIR_GROW * (b - pb))
            if okp.any():
                sp_all, sp_idx = np.concatenate(samples_p), np.repeat(np.arange(len(samples_p)), [len(q) for q in samples_p])
                sc_all, sc_idx = np.concatenate(samples), np.repeat(np.arange(len(samples)), [len(q) for q in samples])
                d_pc_tip, d_pc_sh, d_cp_tip, d_cp_sh = (np.zeros(okp.shape) for _ in range(4))
                for k in np.flatnonzero(okp.any(axis=0)):
                    d_pc_tip[:, k] = trees[k].query(tips_p)[0]
                    d_pc_sh[:, k] = (np.bincount(sp_idx, trees[k].query(sp_all)[0], len(samples_p))
                                     / np.bincount(sp_idx, None, len(samples_p)))
                for j in np.flatnonzero(okp.any(axis=1)):
                    d_cp_tip[j, :] = trees_p[j].query(tips)[0]
                    d_cp_sh[j, :] = (np.bincount(sc_idx, trees_p[j].query(sc_all)[0], len(samples))
                                     / np.bincount(sc_idx, None, len(samples)))
                for j, k in zip(*np.nonzero(okp)):
                    pairs.append((j, k, d_pc_tip[j, k], d_pc_sh[j, k]) if dL[j, k] >= 0 else
                                 (j, k, d_cp_tip[j, k], d_cp_sh[j, k]))
        self.bins[b] = {"n": len(F), "F": F, "tips": tips, "bodies": [_resample(bd, 2.0)[0] for bd in bodies],
                        "pairs": np.asarray(pairs, np.float64).reshape(-1, 4),
                        "pairs_with": None if self.prev is None else self.prev[0]}
        self.prev = (b, F[:, L_ARC].copy(), tips, samples, trees)


# ----------------------------------------------------------------------------- the Viterbi
def lengths(F: np.ndarray, w: dict) -> np.ndarray:
    src = F[:, SRC]
    ec = w.get("edge_clip", 99.0)  # the visible edge offset is trusted only this far either way
    L = (F[:, L_ARC] - np.clip(F[:, EDGE], -ec, ec) - w["c"] + w.get("w_ext", 0.0) * F[:, EXT]
         - w.get("c_end", 0.0) * ((src == 2) | (src == 6)))
    return np.maximum(L, 0.0)


def unary(F: np.ndarray, w: dict, det_scale: float = 1.0) -> np.ndarray:
    S = (w["w_det"] * np.minimum(F[:, DET] / det_scale, 1.5) + w["w_sup"] * F[:, SUP] - w["w_gap"] * F[:, GAP] / 10.0
         - w["w_ahead"] * F[:, AHEAD] - w["w_tan"] * np.maximum(0.0, 0.5 - F[:, RADIAL]) - w["w_carry"] * (F[:, SRC] >= 4))
    return w["theta"] - S


def viterbi(bins: dict[int, dict], rs: int, nb: int, w: dict, det_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per bin (rs .. nb-1): the chosen candidate (-1: not germinated, -2: held at the last reading) and the length.
    States: not germinated (N), each candidate, and held (H: no candidate explains the bin; it keeps the length, tip
    and body of the reading it came from, at ``w_hold`` a bin, and may hand over to a candidate later at ``w_reacq``
    if the length change and the body agree; it keeps its best predecessor's attributes, a greedy step inside an
    otherwise exact Viterbi). See prototypes/tip_trajectory/dp.py, of which this is a copy."""
    w_hold, w_reacq = w.get("w_hold", 0.3), w.get("w_reacq", 0.5)
    off, Fs, n0 = {}, [], 0
    for b in sorted(bins):
        if bins[b]["n"]:
            Fs.append(bins[b]["F"])
            off[b] = (n0, n0 + bins[b]["n"])
            n0 += bins[b]["n"]
    choice, out_L = np.full(nb - rs, -1), np.zeros(nb - rs)
    if not Fs:
        return choice, out_L
    F_all = np.concatenate(Fs)
    L_all, U_all = lengths(F_all, w), unary(F_all, w, det_scale)
    S_all = np.where(L_all <= w.get("l0_max", 1e9), w["w_on"] + w.get("w_l0", 1e3) * np.maximum(0.0, L_all - w["l0"]) / 10.0,
                     np.inf)
    Lof = lambda bb, kk: float(L_all[off[bb][0] + kk])  # noqa: E731
    cN, cH, hold, prev, back = 0.0, np.inf, None, None, {}
    for b in range(rs, nb):
        B = bins.get(b, {"n": 0})
        if not B["n"]:
            if prev is not None and np.isfinite(prev[1]).any():
                j = int(np.argmin(prev[1]))
                if prev[1][j] + w_hold < cH + w_hold:
                    cH, hold = prev[1][j] + w_hold, (prev[0], j, float(prev[2][j]))
                else:
                    cH += w_hold
                prev = (prev[0], prev[1] + w_hold, prev[2])
            else:
                cH += w_hold
            back[b] = ("empty", hold)
            continue
        s0, s1 = off[b]
        L, U, n = L_all[s0:s1], U_all[s0:s1], s1 - s0
        best = cN + S_all[s0:s1]
        code = np.full(n, -1)
        if prev is not None and B["pairs_with"] == prev[0] and len(B["pairs"]):
            pb, pc, pL = prev
            pr = B["pairs"]
            j, k = pr[:, 0].astype(int), pr[:, 1].astype(int)
            dL = L[k] - pL[j]
            ok = ((dL >= -w["shrink"]) & (dL <= w["vmax"] * (b - pb)) & np.isfinite(pc[j])
                  & (pr[:, 2] <= w["cap_tip"]) & (pr[:, 3] <= w["cap_share"]))
            if ok.any():
                j, k, dL, pr = j[ok], k[ok], dL[ok], pr[ok]
                T = (pc[j] + w["w_shrink"] * np.maximum(0.0, -dL) + w["w_cons"] * np.maximum(0.0, pr[:, 2] - 1.5)
                     + w["w_share"] * np.maximum(0.0, pr[:, 3] - 1.5))
                o = np.lexsort((T, k))
                first = np.ones(len(o), bool)
                first[1:] = k[o][1:] != k[o][:-1]
                kk, tt, jj = k[o][first], T[o][first], j[o][first]
                better = tt < best[kk]
                best[kk[better]], code[kk[better]] = tt[better], jj[better]
        if np.isfinite(cH) and hold is not None:
            hb, hj, hL = hold
            dL = L - hL
            ok = (dL >= -w["shrink"]) & (dL <= w["vmax"] * (b - hb))
            if ok.any():
                tip_h, body_h = bins[hb]["tips"][hj], bins[hb]["bodies"][hj]
                cat = np.concatenate(B["bodies"])
                starts = np.cumsum([0] + [len(q) for q in B["bodies"]])[:-1]
                d_fwd = np.minimum.reduceat(np.hypot(*(cat - tip_h).T), starts)
                d_back = np.sqrt(((B["tips"][:, None, :] - body_h[None]) ** 2).sum(-1)).min(1)
                d = np.where(dL >= 0, d_fwd, d_back)
                T = np.where(ok & (d <= w.get("cap_reacq", w["cap_tip"])), cH + w_reacq
                             + w["w_shrink"] * np.maximum(0.0, -dL) + w["w_cons"] * np.maximum(0.0, d - 1.5), np.inf)
                better = T < best
                best[better], code[better] = T[better], -3
        hold_here, cH_new = hold, cH + w_hold
        if prev is not None and np.isfinite(prev[1]).any():
            j = int(np.argmin(prev[1]))
            if prev[1][j] + w_hold < cH_new:
                cH_new, hold_here = prev[1][j] + w_hold, (prev[0], j, float(prev[2][j]))
        back[b] = (code, hold)
        prev = (b, best + U, L)
        cH, hold = cH_new, hold_here
    endC = prev[1].min() if prev is not None else np.inf
    if min(endC, cH) >= cN:
        return choice, out_L
    state = ("C", prev[0], int(np.argmin(prev[1]))) if endC <= cH else ("H", nb - 1, hold)
    b = nb - 1
    while b >= rs:
        if state[0] == "C":
            sb, k = state[1], state[2]
            for bb in range(sb + 1, b + 1):
                choice[bb - rs], out_L[bb - rs] = -2, Lof(sb, k)
            choice[sb - rs], out_L[sb - rs] = k, Lof(sb, k)
            code = back[sb][0][k]
            if code == -1:
                break
            if code == -3:
                state = ("H", sb - 1, back[sb][1])
            else:
                pb = bins[sb]["pairs_with"]
                for bb in range(pb + 1, sb):
                    choice[bb - rs], out_L[bb - rs] = -2, Lof(pb, int(code))
                state = ("C", pb, int(code))
            b = state[1]
        else:
            _, hb_end, origin = state
            if origin is None:
                break
            ob, oj, oL = origin
            for bb in range(ob + 1, hb_end + 1):
                choice[bb - rs], out_L[bb - rs] = -2, oL
            state, b = ("C", ob, oj), ob
    return choice, out_L


MID_SHIFT, MID_REACH, MID_SMOOTH = 4.0, 8.0, 3


def mid_correction(body: np.ndarray, P: np.ndarray) -> float:
    """How much longer a body (reference coordinates) is along the middle of its tube than as found: the cheapest route
    hugs the inside of a curving tube, the annotator traces its middle (~half the tube's width x the turn: ~11 px for a
    U-turn). The body moved onto the middle of the map's band across it (``learned._band_centres``, median-filtered
    along it, at most ``MID_SHIFT`` px); ``P``: that bin's full-frame map, 0..1, reference coordinates."""
    from .learned import _band_centres, _running_median
    q, s = _resample(np.asarray(body, float), 1.0)
    if len(q) < 4:
        return 0.0
    t = np.gradient(q, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-MID_REACH, MID_REACH + 1e-6, 0.5)
    pts = (q[:, None, :] + offs[None, :, None] * n[:, None, :] - 0.5).astype(np.float32)
    prof = cv2.remap(P, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c = _band_centres(prof, offs, 0.35, 14.0)
    ok = np.isfinite(c)
    if ok.sum() < 3:
        return 0.0
    idx = np.arange(len(c))
    c = np.clip(_running_median(np.interp(idx, idx[ok], c[ok]), MID_SMOOTH), -MID_SHIFT, MID_SHIFT)
    qc = q + c[:, None] * n
    return float(np.sum(np.hypot(*np.diff(qc, axis=0).T)) - s[-1])


def _isotonic(y: np.ndarray) -> np.ndarray:
    vals, wts, cnt = [], [], []
    for v in np.asarray(y, float):
        vals.append(float(v)); wts.append(1.0); cnt.append(1)  # noqa: E702
        while len(vals) > 1 and vals[-2] > vals[-1]:
            ww = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / ww]
            wts[-2:] = [ww]
            cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(np.asarray(vals), cnt) if vals else np.zeros(0)


# ----------------------------------------------------------------------------- the reader
def drift_per_bin(res: dict, rs: int, nb: int) -> np.ndarray:
    """(nb, 2) drift of a reading's grain at every bin (zero where none; held before the first and over gaps)."""
    d = np.zeros((nb, 2))
    if res.get("drift"):
        xy = np.asarray(res["drift"]["xy"], float)
        fin = np.isfinite(xy).all(axis=1)
        if fin.any():
            idx = np.arange(len(xy))
            xy = np.stack([np.interp(idx, idx[fin], xy[fin, k]) for k in (0, 1)], axis=1)
            d[rs:rs + len(xy)] = xy[: nb - rs]
            d[:rs] = xy[0]
    return d


def weights(p) -> dict:
    w = dict(WEIGHTS)
    if getattr(p, "tiptraj_weights", None):
        src = p.tiptraj_weights
        w.update(json.loads(Path(src).read_text())["best"] if str(src).endswith(".json") and Path(src).exists()
                 else json.loads(src))
    return w


def candidates(renderer: Renderer, prob: Renderer, det: DetMaps, meta: dict, grain: dict, drift: np.ndarray,
               half: int, guide: dict | None = None) -> dict[int, dict]:
    """Every bin's candidates for one grain (``_Track``) where the grain is then (census + ``drift``)."""
    from .analyze import exit_edge
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
    cx, cy = gx + drift[rs][0], gy + drift[rs][1]
    early = np.mean([np.nan_to_num(renderer.crop(k, cx, cy, 64)) for k in range(rs, rs + 3)], axis=0)
    edges = np.array([exit_edge(early, 63.5, gr, math.radians(5.0 * k)) for k in range(72)])
    tr = _Track(gx, gy, gr, drift, edges, half, guide)
    for b in range(rs, nb):
        tr.step(b, prob.bins[b], det.tip(b))
    return tr.bins


def read(res: dict, renderer: Renderer, prob: Renderer, meta: dict, grain: dict, p, det: DetMaps,
         det_scale: float = 1.0) -> dict:
    """A reading ``res`` (any reader's, for its drift) read again by the tip-trajectory reader: its germination
    call, onset, lengths, tips and route replaced; flag ``reader:tiptraj``. Tips and route are in the frame the grain
    was read in (census + drift), as the other readers'."""
    rs, nb, fpb = int(meta.get("ref_start", 0)), int(meta["n_bins"]), int(meta["frames_per_bin"])
    w = weights(p)
    drift = drift_per_bin(res, rs, nb)
    half = int(p.tiptraj_half)
    scale = det_scale if w.get("det_norm") else 1.0
    bins = candidates(renderer, prob, det, meta, grain, drift, half)
    choice, L = viterbi(bins, rs, nb, w, scale)
    if getattr(p, "tiptraj_guided", False) and (choice >= 0).any():
        # second pass: the first reading's tube followed along the map (its tip, points beyond it, its corridor)
        guide = {rs + i: bins[rs + i]["bodies"][k] for i, k in enumerate(choice) if k >= 0}
        bins = candidates(renderer, prob, det, meta, grain, drift, half, guide)
        choice, L = viterbi(bins, rs, nb, w, scale)
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    out = {k: v for k, v in res.items() if k not in ("path_by_bin", "bend", "tip", "exit_xy", "path_length_px",
                                                      "onset_interval", "route_centred_px", "drawn_on_tube")}
    out["flags"] = [f for f in res.get("flags", []) if not f.startswith(("onset_lookback", "onset:"))] + ["reader:tiptraj"]
    germ = np.flatnonzero(choice != -1)
    if not len(germ):
        out.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                   length={"frames": frames, "px": [0.0] * len(frames)}, path=[], final_length_px=0.0)
        return out
    t0 = int(germ[0])
    if getattr(p, "tiptraj_mid", False):  # lengths along the middle of the tube (mid_correction), chosen readings
        last = 0.0
        for i, k in enumerate(choice):
            if k >= 0:
                b = rs + i
                P = np.asarray(prob.bins[b], np.float32) / P_SCALE_MAP
                last = mid_correction(bins[b]["bodies"][k] + drift[b], P)
            if k != -1:
                L[i] += last
    if w.get("iso", True):
        L[t0:] = _isotonic(L[t0:])
    tips, last = [], None
    for i, k in enumerate(choice):
        if k >= 0:
            last = bins[rs + i]["tips"][k]
        tips.append(None if last is None else [round(float(last[0]), 2), round(float(last[1]), 2)])
    first = next(t for t in tips if t is not None)
    tips = [first if t is None else t for t in tips]
    kb = max(((rs + i, int(k)) for i, k in enumerate(choice) if k >= 0), key=lambda q: (L[q[0] - rs], q[0]))
    route = bins[kb[0]]["bodies"][kb[1]]  # the body of the longest reading (2 px), grain frame
    out.update(status="emerged_at_start" if t0 == 0 else "emerged_within", onset_frame=frames[t0],
               onset_interval=None if t0 == 0 else [frames[t0 - 1], frames[t0]],
               length={"frames": frames, "px": [round(float(v), 2) for v in L]},
               tip={"frames": frames, "xy": tips}, path=np.round(route, 2).tolist(),
               exit_xy=[round(float(route[0][0]), 2), round(float(route[0][1]), 2)],
               final_length_px=round(float(L[-1]), 2),
               path_length_px=round(float(np.sum(np.hypot(*np.diff(route, axis=0).T))) if len(route) > 1 else 0.0, 2))
    return out
