"""Full-frame tip-detector maps for every bin of a movie, with that movie's held-out fold model.

Inputs as prototypes/tip_detector/common.py (A, D, C at bin b, each less its ~60 px local median, / 20, 8, 20 grey
levels), here on the whole registered frame (the network uses BatchNorm, so a crop's map equals the frame's away from
the crop edge), run in four overlapping tiles. Stored per bin as uint8 (P x 250, values below 3 set to 0) in
OUT/det/<movie>/b<bin>.npz: ``tip`` (apex head) and ``body`` (traced-body head). Bins without inputs
(b - 7 < ref_start or b + 1 >= n_bins) are not written (read as zero).

    python -m prototypes.tip_trajectory.detmaps ld m2 m1 [--bins 50 60]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from scipy.ndimage import median_filter

from sparsetrack import stack

from .common import DET_DIR, FOLD, OUT, cache_dir

K = 6
F_BG, K_BG = 4, 15
SCALE = np.array([20.0, 8.0, 20.0], np.float32)
CTX = 64  # px of context round each tile's kept part (the network's reach is ~48)


def medbg(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    small = cv2.resize(img, (max(w // F_BG, 1), max(h // F_BG, 1)), interpolation=cv2.INTER_AREA)
    med = median_filter(small, size=K_BG, mode="reflect")
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def load_net(path, device=None):
    import torch
    from sparsetrack.learned import _unet
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck.get("channels", "ADC") == "ADC" and ck.get("out_ch", 2) == 2
    net = _unet(tuple(ck.get("widths", (16, 32, 64, 128))), "batch")
    net.load_state_dict(ck["state"])
    return net.eval().to(device)


def run_tiles(net, x: np.ndarray, ny: int = 2, nx: int = 2) -> np.ndarray:
    """(2, H, W) sigmoid maps of a (3, H, W) input, in ny x nx tiles with CTX px of context each side."""
    import torch
    _, H, W = x.shape
    out = np.zeros((2, H, W), np.float32)
    dev = next(net.parameters()).device
    ys = np.linspace(0, H, ny + 1).astype(int)
    xs = np.linspace(0, W, nx + 1).astype(int)
    for i in range(ny):
        for j in range(nx):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            a0, a1, b0, b1 = max(0, y0 - CTX), min(H, y1 + CTX), max(0, x0 - CTX), min(W, x1 + CTX)
            t = x[:, a0:a1, b0:b1]
            h, w = t.shape[1:]
            t = np.pad(t, ((0, 0), (0, (-h) % 8), (0, (-w) % 8)), mode="reflect")
            with torch.no_grad():
                y = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(t))[None].to(dev)))[0].cpu().numpy()
            out[:, y0:y1, x0:x1] = y[:, y0 - a0:y1 - a0, x0 - b0:x1 - b0]
    return out


def main(movie: str, bins_only=None, log=print) -> None:
    bins, meta = stack.load(cache_dir(movie))
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    H, W = bins.shape[1:]
    net = load_net(DET_DIR / FOLD[movie])
    out = OUT / "det" / movie
    out.mkdir(parents=True, exist_ok=True)
    reg_cache: dict = {}

    def reg(b):
        if b not in reg_cache:
            dx, dy = shifts[b]
            reg_cache[b] = cv2.warpAffine(np.asarray(bins[b], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                                          (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return reg_cache[b]

    m_cache: dict = {}

    def M(b):  # mean of bins b-1..b+1, clamped to the movie
        if b not in m_cache:
            ks = range(max(b - 1, rs), min(b + 1, nb - 1) + 1)
            m_cache[b] = np.mean([reg(k) for k in ks], axis=0)
        return m_cache[b]

    E = np.mean([reg(k) for k in range(rs, rs + 3)], axis=0)
    C_bg = None
    lo, hi = rs + K + 1, nb - 2
    todo = [b for b in range(lo, hi + 1) if (bins_only is None or b in bins_only)
            and not (out / f"b{b:03d}.npz").exists()]
    t0 = time.time()
    for n, b in enumerate(todo):
        for k in [k for k in reg_cache if k < b - K - 2]:
            del reg_cache[k]
        for k in [k for k in m_cache if k < b - K - 1]:
            del m_cache[k]
        Mb = M(b)
        A = Mb - medbg(Mb)
        D = Mb - M(b - K)
        D = D - medbg(D)
        C = Mb - E
        C = C - medbg(C)
        x = (np.stack([A, D, C]) / SCALE[:, None, None]).astype(np.float32)
        y = run_tiles(net, x)
        q = np.clip(np.round(y * 250.0), 0, 250).astype(np.uint8)
        q[q < 3] = 0
        np.savez_compressed(out / f"b{b:03d}.npz", tip=q[0], body=q[1])
        if n % 20 == 0:
            log(f"{movie} b{b}: {time.time() - t0:.0f} s for {n + 1}/{len(todo)}; tip max {y[0].max():.2f}", flush=True)
    log(f"{movie}: {len(todo)} bins in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movies", nargs="+")
    ap.add_argument("--bins", nargs="*", type=int)
    a = ap.parse_args()
    for m in a.movies:
        main(m, set(a.bins) if a.bins else None)
