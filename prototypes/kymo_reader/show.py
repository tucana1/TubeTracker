"""Diagnostic sheets: kymograph (time down, arc right), P(covered), decoded front and labels.

    python -m prototypes.kymo_reader.show --model M.pt --data runs/kymo_reader/data/real_ld.npz \
        --grains g030 g034 --variant st --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from . import decode, store
from .evaluate import logits


def sheet(sm: dict, lg: np.ndarray | None, vmax: float = 4.0, scale: int = 3) -> np.ndarray:
    d = store.dequant(sm["dyn"])
    W = d.shape[2]
    k = d[:, :, W // 2 - 1:W // 2 + 2].mean(axis=2)
    u = cv2.cvtColor(((np.clip(k / 15.0, -1, 1) + 1) * 127.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    panels = [u]
    s = sm["s"].astype(float)
    i0 = int(np.argmin(np.abs(s)))
    if lg is not None:
        p = 1.0 / (1.0 + np.exp(-lg))
        pu = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        F = decode.fronts(lg, s, vmax)
        for t, f in enumerate(F):
            j = int(np.clip(round(f - s[0]), 0, len(s) - 1))
            u[t, j] = (0, 0, 255)
            pu[t, j] = (0, 0, 255)
        panels.append(pu)
    if "lab_m" in sm:
        m, y = sm["lab_m"].astype(bool), sm["lab_y"].astype(bool)
        lab = u.copy()
        lab[m & y] = (0.5 * lab[m & y] + [0, 110, 0]).astype(np.uint8)
        lab[m & ~y] = (0.5 * lab[m & ~y] + [0, 0, 120]).astype(np.uint8)
        panels.append(lab)
    elif "target" in sm:
        L = sm["target"]
        lab = u.copy()
        for t, l in enumerate(L):
            j = int(np.clip(round(l - s[0]), 0, len(s) - 1))
            lab[t, j] = (0, 255, 0)
        panels.append(lab)
    if "st_len" in sm:  # SparseTrack's reported length from the path's first point
        for t, l in enumerate(sm["st_len"]):
            if l > 0:
                j = int(np.clip(round(l - s[0]), 0, len(s) - 1))
                panels[0][t, j] = (255, 128, 0)
    for pnl in panels:
        pnl[:, i0] = (pnl[:, i0] * 0.5 + np.array([128, 0, 128])).astype(np.uint8)
    out = [cv2.resize(pnl, (pnl.shape[1] * scale, pnl.shape[0] * scale), interpolation=cv2.INTER_NEAREST) for pnl in panels]
    sep = np.full((out[0].shape[0], 6, 3), 255, np.uint8)
    return np.hstack(sum([[o, sep] for o in out], [])[:-1])


def main(argv=None):
    from .model import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--grains", nargs="*", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load(a.model, dev) if a.model else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for sm in store.load(a.data):
        inf = sm["info"]
        if a.grains and inf["grain"] not in a.grains:
            continue
        if a.variant and inf["variant"] != a.variant:
            continue
        lg = logits(net, sm, dev) if net is not None else None
        name = f"{inf.get('movie', 'x')}_{inf['grain']}_{inf.get('tube', '')}_{inf['variant']}.png"
        cv2.imwrite(str(out / name), sheet(sm, lg))
        n += 1
        if n >= a.limit:
            break
    print(f"{n} sheets -> {out}")


if __name__ == "__main__":
    main()
