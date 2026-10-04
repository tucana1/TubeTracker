"""Full-frame maps of a version 3 tip detector for every bin of a movie, in the reader's format.

    python -m prototypes.tip_detector.maps3 m1 --net runs/research/tip_detector/tip3_ldm2.pt [--out DIR]

sparsetrack/tiptraj.py's ``det_cache`` builds maps only for version 2 checkpoints (inputs A, D, C); the reader takes
precomputed maps instead with ``Params.tiptraj_det = DIR``. This writes DIR/b<bin>.npz (``tip``, and ``body`` with --body; uint8 =
P x 250, below 3 set to 0; the format of ``tiptraj.det_cache`` and prototypes/tip_trajectory/detmaps.py) for bins
ref_start + 7 .. n_bins - 2, and DIR/meta.json. Inputs as common3.py on the whole registered frame (the network uses
BatchNorm, so a crop's map equals the frame's away from the crop edge), run in 2 x 2 tiles with 96 px of context.
Default DIR: runs/research/tip_detector/det3/<net stem>/<movie>. Only ``tip`` is written unless ``--body`` (the
integrated reader reads only ``tip``; ~120 KB a bin, the body head adds ~330 KB).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sparsetrack import stack

from .common import MOVIES, OUT, REPO, load_net
from .common3 import full_frame_inputs

CTX = 96


def run_tiles(net, x: np.ndarray, n: int = 2) -> np.ndarray:
    import torch
    _, H, W = x.shape
    out = np.zeros((2, H, W), np.float32)
    dev = next(net.parameters()).device
    k = 2 ** (len(net.widths) - 1)
    ys, xs = np.linspace(0, H, n + 1).astype(int), np.linspace(0, W, n + 1).astype(int)
    for i in range(n):
        for j in range(n):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            a0, a1, b0, b1 = max(0, y0 - CTX), min(H, y1 + CTX), max(0, x0 - CTX), min(W, x1 + CTX)
            t = x[:, a0:a1, b0:b1]
            h, w = t.shape[1:]
            t = np.pad(t, ((0, 0), (0, (-h) % k), (0, (-w) % k)), mode="reflect")
            with torch.no_grad():
                y = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(t))[None].to(dev)))[0].cpu().numpy()
            out[:, y0:y1, x0:x1] = y[:, y0 - a0:y1 - a0, x0 - b0:x1 - b0]
    return out


def main(movie: str, net_path: str, out: str | None = None, body: bool = False, log=print) -> Path:
    bins, meta = stack.load(REPO / MOVIES[movie][0])
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    net = load_net(net_path)
    assert not isinstance(net.channels, str), "a version 2 checkpoint: use sparsetrack.tiptraj.det_cache"
    d = Path(out) if out else OUT / "det3" / Path(net_path).stem / movie
    d.mkdir(parents=True, exist_ok=True)
    reg: dict = {}
    t0 = time.time()
    todo = [b for b in range(rs + 7, nb - 1) if not (d / f"b{b:03d}.npz").exists()]
    for n, b in enumerate(todo):
        for k in [k for k in reg if k < b - 30 and k > rs + 2]:
            del reg[k]
        x = full_frame_inputs(bins, meta, b, net.channels, reg)
        y = run_tiles(net, x)
        q = np.clip(np.round(y * 250.0), 0, 250).astype(np.uint8)
        q[q < 3] = 0
        np.savez_compressed(d / f"b{b:03d}.npz", **({"tip": q[0], "body": q[1]} if body else {"tip": q[0]}))
        if n % 50 == 0:
            log(f"{movie} b{b}: {n + 1}/{len(todo)} in {time.time() - t0:.0f} s; tip max {y[0].max():.2f}", flush=True)
    (d / "meta.json").write_text(json.dumps({"model": str(net_path), "n_bins": nb, "channels": list(net.channels),
                                             "version": 3}))
    log(f"{movie}: {len(todo)} bins in {time.time() - t0:.0f} s -> {d}", flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--net", required=True)
    ap.add_argument("--out")
    ap.add_argument("--body", action="store_true", help="also store the traced-body head")
    a = ap.parse_args()
    main(a.movie, a.net, a.out, a.body)
