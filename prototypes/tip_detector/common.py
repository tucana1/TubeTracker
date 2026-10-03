"""Shared pieces of the learned tip detector study: movies, labels, network inputs, the network.

Inputs at bin b (all registered, reference coordinates; M(k) = mean of bins k-1..k+1):
  A = M(b) - local median of M(b)                    current appearance
  D = M(b) - M(b-6), less its local median            the short-interval difference (bins b-1..b+1 minus b-7..b-5)
  C = M(b) - E, less its local median                 change from the first bins (E = mean of the reference bins)
Each divided by a fixed scale in grey levels (SCALE: 20, 8, 20; the tube network uses 20). A movie-level MAD scale was
tried first and rejected before any training: the backgrounds are so flat (median |A| 0.09 grey levels on ld, 0.3-2 on
m2/m1) that it would put tubes and grains at hundreds of units. Contrast differences between movies are left to the
gain augmentation. Local median = median over ~60 px (on a 1/4-scale grid), computed on the crop plus a 32 px margin,
so a crop's inputs equal the full frame's away from the frame edge.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from sparsetrack import stack
from sparsetrack.render import Renderer

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/research/tip_detector"
OUT.mkdir(parents=True, exist_ok=True)
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
K = 6          # short interval (bins)
MARGIN = 32    # extra px round a crop for the local median
F_BG, K_BG = 4, 15  # local median: 1/4 grid, 15 cells (~60 px)
SCALE = np.array([20.0, 8.0, 20.0], np.float32)  # grey levels per input unit: A, D, C


def medbg(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    small = cv2.resize(img, (max(w // F_BG, 1), max(h // F_BG, 1)), interpolation=cv2.INTER_AREA)
    med = median_filter(small, size=K_BG, mode="reflect")
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def length_class(L: float) -> str:
    return "young" if L < 15 else ("long" if L >= 60 else "mid")


class Movie:
    def __init__(self, name: str, scales: bool = True):
        self.name = name
        cache, labels = (REPO / p for p in MOVIES[name])
        self.bins, self.meta = stack.load(cache)
        self.R = Renderer(self.bins, self.meta)
        self.rs, self.nb = int(self.meta.get("ref_start", 0)), int(self.meta["n_bins"])
        self.L = json.loads(labels.read_text())
        self.grains = self.L["grains"]
        self.labels = {g: lab for g, lab in self.L["labels"].items() if not self.grains[g].get("excluded")}
        f = OUT / f"offsets_{name}.npz"
        self.offsets = dict(np.load(f)) if f.exists() else {}
        self.lo, self.hi = self.rs + K + 1, self.nb - 2  # bins whose inputs exist (b-7 >= rs, b+1 <= nb-1)
        self.scale = SCALE.copy() if scales else np.ones(3, np.float32)

    # ---------------------------------------------------------------- geometry
    def grain_at(self, gid: str, b: int, trace: dict | None = None) -> tuple[float, float, float]:
        """(x, y, r) of the grain at bin b: census + the trace's view offset, else + the followed offset."""
        g = self.grains[gid]
        if trace is not None and trace.get("view_offset") is not None:
            off = trace["view_offset"]
        elif gid in self.offsets:
            off = self.offsets[gid][b]
        else:
            off = (0.0, 0.0)
        return float(g["x"] + off[0]), float(g["y"] + off[1]), float(g["r"])

    def traces(self, gid: str, states=("full",)) -> list[tuple[int, dict]]:
        tr = self.labels[gid].get("traces") or {}
        return sorted(((int(b), t) for b, t in tr.items() if t["state"] in states), key=lambda x: x[0])

    # ---------------------------------------------------------------- inputs
    def _mean(self, b0: int, b1: int, cx: float, cy: float, half: int) -> np.ndarray:
        b0, b1 = max(b0, self.rs), min(b1, self.nb - 1)
        return np.mean([self.R.crop(k, cx, cy, half) for k in range(b0, b1 + 1)], axis=0).astype(np.float32)

    def raw(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        """(3, 2h, 2h) unscaled A, D, C with their local medians removed (crop + margin, margin cut off)."""
        H = half + MARGIN
        M = self._mean(b - 1, b + 1, cx, cy, H)
        P = self._mean(b - K - 1, b - K + 1, cx, cy, H)
        E = self._mean(self.rs, self.rs + 2, cx, cy, H)
        M, P, E = (np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0) for v in (M, P, E))
        A = M - medbg(M)
        D = M - P
        D = D - medbg(D)
        C = M - E
        C = C - medbg(C)
        x = np.stack([A, D, C])
        return x[:, MARGIN:-MARGIN, MARGIN:-MARGIN]

    def inputs(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        return (self.raw(b, cx, cy, half) / self.scale[:, None, None]).astype(np.float32)

    def outside(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        out = np.zeros((2 * half, 2 * half), bool)
        for k in range(max(b - K - 1, self.rs), min(b + 1, self.nb - 1) + 1, 2):
            out |= self.R.outside(k, cx, cy, half)
        return out


# -------------------------------------------------------------------- network
def unet(in_ch: int = 3, out_ch: int = 2, widths=(16, 32, 64, 128)):
    import torch
    from torch import nn
    import torch.nn.functional as F

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.widths = tuple(widths)
            self.enc = nn.ModuleList()
            c = in_ch
            for w_ in widths:
                self.enc.append(block(c, w_))
                c = w_
            self.dec = nn.ModuleList(block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1)))
            self.head = nn.Conv2d(widths[0], out_ch, 1)

        def forward(self, x):
            skips = []
            for i, blk in enumerate(self.enc):
                x = blk(x if i == 0 else F.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for blk in self.dec:
                s = skips.pop()
                x = blk(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
            return self.head(x)

    return UNet()


def load_net(path, device=None):
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net = unet(len(ck["channels"]), ck.get("out_ch", 2), ck.get("widths", (16, 32, 64, 128)))
    net.load_state_dict(ck["state"])
    net.channels = ck["channels"]
    return net.eval().to(device)


def heat(net, x: np.ndarray) -> np.ndarray:
    """(n_out, H, W) sigmoid maps for one (C, H, W) input (padded to a multiple of 8)."""
    import torch
    c, h, w = x.shape
    k = 2 ** (len(net.widths) - 1)
    xp = np.pad(x, ((0, 0), (0, (-h) % k), (0, (-w) % k)), mode="reflect")
    dev = next(net.parameters()).device
    with torch.no_grad():
        y = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(xp))[None].to(dev)))[0, :, :h, :w]
    return y.cpu().numpy()
