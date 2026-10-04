"""The learned tip detector: where a growing tube's tip is, bin by bin (research prototype
``prototypes/tip_detector``, 3 Oct 2026; its use here: ``prototypes/tip_track``).

A small U-Net (16-32-64-128, BatchNorm, so a crop's map equals the whole frame's away from the crop's edge)
trained on the human traces of two movies to put a peak (a Gaussian of sigma 2 px) on every growing tube's apex.
Inputs at bin ``b``, registered, in reference coordinates (``M(k)`` = mean of bins ``k-1 .. k+1``):

- ``A = M(b)``, less its local median (current appearance), / 20 grey levels;
- ``D = M(b) - M(b-6)``, less its local median (the short-interval difference: a tube grows at its tip), / 8;
- ``C = M(b) - E`` (``E`` = mean of the reference bins), less its local median (change from the start), / 20.

Local median = median over ~60 px (on a 1/4-scale grid), computed on the crop plus a 32 px margin.

Judged leave one movie out (top-1 peak within 4 px of the human apex): the sparse movie 77%, movie 2 48%, movie 1
61%, against 34 / 20 / 30% for the raw short-interval difference; young tubes 32/34, 16/24, 25/27.

``GrainTipMaps`` computes the map round one grain for every bin, where the grain is at that bin (census + its
drift), in crops: what the tracker uses for germination onsets and young tubes' tips (``analyze.Params.tipdet_*``).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, median_filter

from .render import Renderer

K = 6                 # the short interval (bins)
MARGIN = 32           # extra px round a crop for the local median
F_BG, K_BG = 4, 15    # local median: 1/4 grid, 15 cells (~60 px)
SCALE = np.array([20.0, 8.0, 20.0], np.float32)  # grey levels per input unit: A, D, C
RF = 48               # px of context the network needs on each side of the pixels kept


def medbg(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    small = cv2.resize(img, (max(w // F_BG, 1), max(h // F_BG, 1)), interpolation=cv2.INTER_AREA)
    med = median_filter(small, size=K_BG, mode="reflect")
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def load_detector(path: str | Path, device: str | None = None):
    """The detector network from a ``prototypes/tip_detector/train.py`` checkpoint."""
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


def heat(net, x: np.ndarray, batch: int = 16) -> np.ndarray:
    """Tip maps (n, H, W) for inputs (n, 3, H, W), H and W multiples of 8."""
    import torch
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch], np.float32)).to(dev)
            out.append(torch.sigmoid(net(xb)[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + x.shape[2:], np.float32)


class GrainTipMaps:
    """The detector's tip map round one grain at every bin, the crop centred where the grain is at that bin.

    ``pos``: (n_bins_total, 2) reference (x, y) of the grain per bin (census + drift); ``half``: half-size of the
    map kept (px). Bins whose inputs do not exist (``b - 7 < ref_start`` or ``b + 1 >= n_bins``) are zero.
    ``maps[b]`` pixel (i, j) lies at reference ``(cx[b] - half + j + 0.5, cy[b] - half + i + 0.5)``.
    """

    def __init__(self, renderer: Renderer, net, pos: np.ndarray, half: int = 64, batch: int = 16,
                 sticky_px: float = 8.0):
        meta_rs = renderer.ref_start
        nb = renderer.n_bins
        self.half = int(half)
        self.lo, self.hi = meta_rs + K + 1, nb - 2
        pos = np.asarray(pos, float)
        # the crop stays put until the grain has moved more than sticky_px from its centre (crops are shared)
        self.centre = np.zeros((nb, 2), int)
        c = np.round(pos[0]).astype(int)
        for b in range(nb):
            if np.hypot(*(pos[b] - c)) > sticky_px:
                c = np.round(pos[b]).astype(int)
            self.centre[b] = c
        S = 2 * self.half
        self.maps = np.zeros((nb, S, S), np.float16)
        hn = self.half + RF           # the network's crop
        hr = hn + MARGIN              # the raw crops (with the local median's margin)
        cache: dict = {}

        def reg(k, cx, cy):
            key = (k, cx, cy)
            if key not in cache:
                c = renderer.crop(k, float(cx), float(cy), hr)
                if not np.isfinite(c).all():
                    c = np.nan_to_num(c, nan=float(np.nanmedian(c)) if np.isfinite(c).any() else 0.0)
                cache[key] = c
            return cache[key]

        E = {}
        xs, idx = [], []
        for b in range(self.lo, self.hi + 1):
            cx, cy = int(self.centre[b][0]), int(self.centre[b][1])
            for key in [q for q in cache if q[0] < b - K - 1 or (q[1], q[2]) != (cx, cy)]:
                del cache[key]
            M = np.mean([reg(k, cx, cy) for k in (b - 1, b, b + 1)], axis=0)
            P = np.mean([reg(k, cx, cy) for k in (b - K - 1, b - K, b - K + 1)], axis=0)
            if (cx, cy) not in E:
                E[(cx, cy)] = np.mean([np.nan_to_num(renderer.crop(k, float(cx), float(cy), hr))
                                       for k in range(meta_rs, meta_rs + 3)], axis=0)
            A = M - medbg(M)
            D = M - P
            D = D - medbg(D)
            C = M - E[(cx, cy)]
            C = C - medbg(C)
            x = np.stack([A, D, C])[:, MARGIN:-MARGIN, MARGIN:-MARGIN] / SCALE[:, None, None]
            xs.append(x.astype(np.float32))
            idx.append(b)
            if len(xs) == batch or b == self.hi:
                h = heat(net, np.stack(xs), batch)
                self.maps[idx] = h[:, RF:-RF, RF:-RF].astype(np.float16)
                xs, idx = [], []
            if len(E) > 8:
                E.clear()

    # ------------------------------------------------------------------ geometry
    def to_pix(self, b: int, x: float, y: float) -> tuple[float, float]:
        """Map pixel coordinates (column, row) of reference (x, y) at bin b."""
        cx, cy = self.centre[b]
        return x - (cx - self.half + 0.5), y - (cy - self.half + 0.5)

    def to_ref(self, b: int, j: float, i: float) -> tuple[float, float]:
        cx, cy = self.centre[b]
        return float(cx - self.half + 0.5 + j), float(cy - self.half + 0.5 + i)

    def value(self, b: int, x: float, y: float) -> float:
        j, i = self.to_pix(b, x, y)
        j, i = int(round(j)), int(round(i))
        S = 2 * self.half
        return float(self.maps[b][i, j]) if 0 <= i < S and 0 <= j < S else 0.0

    def peaks(self, b: int, region: np.ndarray, min_value: float = 0.0, size: int = 9) -> list[tuple[float, float, float]]:
        """Local maxima (``size`` x ``size``) of bin b's map inside ``region`` (a mask of the map's shape), best
        first, at least 4 px apart: [(x_ref, y_ref, value)]."""
        m = np.where(region, self.maps[b].astype(np.float32), -np.inf)
        pk = (m == maximum_filter(m, size=size, mode="constant", cval=-np.inf)) & region & (m >= min_value)
        ys, xs = np.nonzero(pk)
        out: list[tuple[float, float, float]] = []
        for o in np.argsort(-m[ys, xs], kind="stable"):
            x, y = int(xs[o]), int(ys[o])
            if all(math.hypot(x - a, y - c) >= 4 for a, c, _ in out):
                out.append((x, y, float(m[y, x])))
        return [(*self.to_ref(b, x, y), v) for x, y, v in out]

    def grid(self, b: int, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Distance of every map pixel from reference (x, y) at bin b, and the angle (radians, atan2(dy, dx))."""
        S = 2 * self.half
        j0, i0 = self.to_pix(b, x, y)
        jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
        return np.hypot(jj - j0, ii - i0), np.arctan2(ii - i0, jj - j0)

    def all_peaks(self, b: int, min_value: float = 0.03) -> list[tuple[float, float, float]]:
        """Every local maximum (9 x 9) of bin b's whole map, best first, at least 4 px apart: [(x_ref, y_ref, value)]."""
        return self.peaks(b, np.ones(self.maps[b].shape, bool), min_value)


# ---------------------------------------------------------------------------- the tracker's use
def sustained(vals: np.ndarray, thr: float, hold: int) -> int | None:
    """First index from which ``vals`` stays >= ``thr`` for ``hold`` consecutive entries (None if never)."""
    above = np.nan_to_num(np.asarray(vals, float), nan=0.0) >= thr
    for t in np.flatnonzero(above):
        if t + hold > len(above):
            break
        if above[t:t + hold].all():
            return int(t)
    return None


def isotonic(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for v in np.asarray(y, float):
        vals.append(float(v))
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
            wts[-2:] = [w]
            cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(np.asarray(vals), cnt) if vals else np.zeros(0)


def grain_positions(res: dict, rs: int, nb: int) -> np.ndarray:
    """(nb, 2) reference place of a reading's grain at every bin: its place + its drift (held where unknown)."""
    pos = np.tile(np.array([res["x"], res["y"]], float), (nb, 1))
    if res.get("drift"):
        d = np.asarray(res["drift"]["xy"], float)
        fin = np.isfinite(d).all(axis=1)
        if fin.any():
            idx = np.arange(len(d))
            d = np.stack([np.interp(idx, idx[fin], d[fin, k]) for k in (0, 1)], axis=1)
            pos[rs:rs + len(d)] += d
            pos[:rs] += d[0]
    return pos


def apply(res: dict, gm, pos: np.ndarray, edge, meta: dict, p) -> dict:
    """The detector's onset and young tubes' lengths in a reading ``res`` (in place; returns a summary).

    ``gm``: the grain's ``GrainTipMaps`` (or anything with ``lo``, ``hi`` and ``all_peaks(b)``); ``pos``: (n_bins, 2)
    where the grain is per bin; ``edge(theta)``: the grain's visible edge in direction ``theta`` as an offset from its
    radius (``analyze.exit_edge``), or None.

    Onset (``p.tipdet_onset``): the rim response at bin b is the strongest detector peak within r - 2 .. r + 25 px of
    the grain; the detector's onset is the first bin from which it stays >= ``p.tipdet_thr`` for ``p.tipdet_hold``
    bins. It replaces the reading's onset where that is later by more than ``p.tipdet_late_bins`` bins ("later"), and
    also where the reading saw no tube ("later_or_missing").
    Young tubes (``p.tipdet_young``): from the onset, at every bin where the reading is shorter than
    ``p.tipdet_young_px`` (or reads nothing yet after a moved onset), the strongest peak (>= ``p.tipdet_young_min``)
    within r - 2 .. r + young + 5 px is the tip; length = its distance from the grain's visible edge in its direction
    + ``p.tipdet_young_k`` (the peak lags the apex; annotators start where the tube leaves the edge). Bins between a
    moved onset and the first length get a ramp; a reading with no tube holds the last young length; the series is
    made non-decreasing again from the onset (pool adjacent violators)."""
    rs = int(meta.get("ref_start", 0))
    frames = list(res["length"]["frames"])
    n = len(frames)
    r = float(res["r"])
    until = res.get("observed_until_frame")
    last_i = n - 1 if until is None else max(0, int(np.searchsorted(frames, until, side="right")) - 1)
    lo, hi = max(gm.lo, rs), min(gm.hi, rs + last_i)
    vals = np.zeros(n)
    peaks_by_i: dict[int, list] = {}
    reach = max(25.0, p.tipdet_young_px + 5.0)
    for b in range(lo, hi + 1):
        i = b - rs
        pk = []
        for x, y, v in gm.all_peaks(b):
            d = math.hypot(x - pos[b][0], y - pos[b][1]) - r
            if -2.0 <= d <= reach:
                pk.append((x, y, v, d))
        peaks_by_i[i] = pk
        vals[i] = next((v for x, y, v, d in pk if d <= 25.0), 0.0)
    t_det = sustained(vals[lo - rs:hi - rs + 1], p.tipdet_thr, p.tipdet_hold) if hi >= lo else None
    i_det = None if t_det is None else lo - rs + t_det
    summary = {"detector_onset_frame": None if i_det is None else frames[i_det], "moved_onset_bins": 0,
               "young_bins": 0}
    old_on = res.get("onset_frame") if res.get("status") == "emerged_within" else None
    no_tube = not str(res.get("status", "")).startswith("emerged")
    i_old = None if old_on is None else int(np.argmin(np.abs(np.asarray(frames) - old_on)))
    if res.get("status") == "emerged_at_start":
        i_old = 0
    start = None
    if p.tipdet_onset in ("later", "later_or_missing") and i_det is not None and i_det > 0:
        if i_old is not None and i_old - i_det > p.tipdet_late_bins:
            start = i_det
        elif no_tube and p.tipdet_onset == "later_or_missing" and res.get("status") != "unobservable":
            start = i_det
    elif p.tipdet_onset not in ("off", "later", "later_or_missing"):
        raise ValueError(f"unknown tipdet_onset {p.tipdet_onset!r}: off, later or later_or_missing")
    px = np.asarray(res["length"]["px"], float)
    tips = list((res.get("tip") or {}).get("xy") or [None] * n)
    drift = pos[rs:rs + n] - np.array([res["x"], res["y"]], float)
    found = np.zeros(n, bool)
    first_on = start if start is not None else i_old
    if p.tipdet_young and first_on is not None:
        for i in range(first_on, last_i + 1):
            if i not in peaks_by_i:
                continue
            if not (0 < px[i] < p.tipdet_young_px or (px[i] <= 0 and start is not None)):
                continue
            best = None
            for x, y, v, d in peaks_by_i[i]:
                if v >= p.tipdet_young_min and d <= p.tipdet_young_px + 5.0 and (best is None or v > best[2]):
                    best = (x, y, v, d)
            if best is None:
                continue
            x, y = best[0], best[1]
            c = pos[rs + i]
            theta = math.atan2(y - c[1], x - c[0])
            e = float(edge(theta)) if edge is not None else 0.0
            px[i] = max(0.0, math.hypot(x - c[0], y - c[1]) - (r + e) + p.tipdet_young_k)
            tips[i] = [round(float(x - drift[i][0]), 2), round(float(y - drift[i][1]), 2)]
            found[i] = True
    flags = res.setdefault("flags", [])
    if start is not None:
        first = next((k for k in range(start, n) if px[k] > 0), None)
        if first is None:
            start = None  # nothing to measure from it: the detector's onset alone does not make a tube
        else:
            for k in range(start, first):
                px[k] = px[first] * (k - start + 1) / (first - start + 1)
                tips[k] = tips[first]
            moved = (i_old - start) if i_old is not None else None
            summary["moved_onset_bins"] = moved
            res["status"] = "emerged_within"
            res["onset_frame"] = frames[start]
            res["onset_interval"] = [frames[start - 1], frames[start]]
            flags.append(f"onset_tip_detector:{moved if moved is not None else 'new'}")
            if no_tube:
                last = None
                for k in range(start, n):
                    if found[k] or k <= first:
                        last = k
                    elif last is not None:
                        px[k], tips[k] = px[last], tips[last]
                # the drawn tube: from the grain's edge towards the detector's last young tip
                k = max((j for j in range(n) if found[j]), default=first)
                tx, ty = tips[k]
                gx, gy = float(res["x"]), float(res["y"])
                a = math.atan2(ty - gy, tx - gx)
                ex = [round(gx + r * math.cos(a), 2), round(gy + r * math.sin(a), 2)]
                res["path"], res["exit_xy"] = [ex, [tx, ty]], ex
                res["path_length_px"] = round(math.hypot(tx - ex[0], ty - ex[1]), 2)
                tips = [ex if t is None else t for t in tips]  # before the tube: at its exit, as the readers do
    if start is not None or found.any():
        on = int(np.argmax(px > 0)) if (px > 0).any() else n
        px[:on] = 0.0
        px[on:] = isotonic(px[on:])
        res["length"] = {"frames": frames, "px": [round(float(v), 2) for v in px]}
        res["final_length_px"] = round(float(px[-1]), 2) if len(px) else 0.0
        res["tip"] = {"frames": frames, "xy": tips}
        summary["young_bins"] = int(found.sum())
        if found.any():
            flags.append(f"young_tip_detector:{int(found.sum())}")
    res["tip_detector"] = summary
    return summary


_NETS: dict = {}


def read(res: dict, renderer: Renderer, meta: dict, p) -> dict | None:
    """``apply`` for one reading of the movie behind ``renderer``: the detector network (``p.tipdet_model``, loaded
    once per process), its maps round the grain where the reading put it at every bin, and the grain's visible edge
    from the before image (``analyze.exit_edge``, every 5 degrees)."""
    from .analyze import exit_edge
    if res.get("status") == "unobservable" or not p.tipdet_model:
        return None
    if not str(res.get("status", "")).startswith("emerged") and p.tipdet_onset != "later_or_missing":
        return None  # nothing the detector may change: no tube, and no onset of its own asked for
    key = str(Path(p.tipdet_model).resolve())
    if key not in _NETS:
        _NETS.clear()
        _NETS[key] = load_detector(key)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    pos = grain_positions(res, rs, nb)
    gm = GrainTipMaps(renderer, _NETS[key], pos, half=p.tipdet_half)
    half = 64
    early = np.mean([np.nan_to_num(renderer.crop(k, float(pos[rs][0]), float(pos[rs][1]), half))
                     for k in range(rs, rs + 3)], axis=0)
    edges: dict = {}

    def edge(theta: float) -> float:
        k = int(round(math.degrees(theta) % 360 / 5.0)) % 72
        if k not in edges:
            edges[k] = exit_edge(early, half - 0.5, float(res["r"]), math.radians(5.0 * k))
        return edges[k]

    return apply(res, gm, pos, edge, meta, p)
