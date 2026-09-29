"""Side-by-side panels of a real movie: the bin, SparseTrack's change evidence |bin - before|, and the
learned tube probability.

    python -m prototypes.learned_evidence.show --model unet.pt --cache runs/sparsetrack/ld \
        --bins 40 90 170 --out panels.png [--x 640 --y 512 --half 160]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from sparsetrack import stack

from .data import normalise
from .evaluate import registered_frame
from .model import load, predict


def _u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def panels(net, cache: str, bins_: list[int], cx: int, cy: int, half: int, zoom: int = 2) -> np.ndarray:
    bins, meta = stack.load(cache)
    sh = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, sh, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, sh, b) for b in range(nb - 4, nb - 1)], axis=0)
    win = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    rows = []
    for b in bins_:
        img = registered_frame(bins, sh, b)
        p = predict(net, normalise(img[win], early[win], late[win]))[0]
        change = cv2.GaussianBlur(np.abs(img - early), (0, 0), 1.0)[win]
        lo, hi = np.percentile(img[win], [0.5, 99.5])
        a = cv2.cvtColor(_u8(img[win], lo, hi), cv2.COLOR_GRAY2BGR)
        c = cv2.applyColorMap(_u8(change, 0, 25), cv2.COLORMAP_MAGMA)
        q = a.copy()
        q[..., 1] = np.maximum(q[..., 1], (p * 255).astype(np.uint8))
        q[..., 0] = (q[..., 0] * (1 - 0.6 * p)).astype(np.uint8)
        q[..., 2] = (q[..., 2] * (1 - 0.6 * p)).astype(np.uint8)
        row = np.hstack([a, c, q])
        row = cv2.resize(row, (row.shape[1] * zoom, row.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
        cv2.putText(row, f"bin {b}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rows.append(row)
    out = np.vstack(rows)
    head = np.full((28, out.shape[1], 3), 255, np.uint8)
    w3 = out.shape[1] // 3
    for i, t in enumerate(("registered bin", "SparseTrack evidence |bin - before|", "learned P(tube) (green)")):
        cv2.putText(head, t, (i * w3 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([head, out])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bins", type=int, nargs="+", required=True)
    ap.add_argument("--x", type=int, default=640)
    ap.add_argument("--y", type=int, default=512)
    ap.add_argument("--half", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cv2.imwrite(args.out, panels(load(args.model), args.cache, args.bins, args.x, args.y, args.half))


if __name__ == "__main__":
    main()
