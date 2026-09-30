"""Contact sheet: human traces over the registered bin, its change from the "before" image and each network's map.

    python -m prototypes.tube_net.sheet OUT.png MOVIE GRAIN@BIN [GRAIN@BIN ...] --models A.pt B.pt [--half 90]

Maps are computed on the full frame (as SparseTrack does) and, for comparison, on a crop of the grain alone
(``--crop-too``): with GroupNorm the two differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import learned, stack

from .pixels import MOVIES, REPO


def _panel(img: np.ndarray, lo: float, hi: float, scale: int) -> np.ndarray:
    u8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("movie", choices=tuple(MOVIES))
    ap.add_argument("items", nargs="+", help="GRAIN@BIN")
    ap.add_argument("--models", nargs="+", default=[str(learned.MODEL)])
    ap.add_argument("--half", type=int, default=90)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--crop-too", action="store_true")
    a = ap.parse_args(argv)
    cache, lab = (REPO / p for p in MOVIES[a.movie])
    src, meta = stack.load(cache)
    L = json.loads(lab.read_text())
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([learned._registered(src, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([learned._registered(src, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    nets = [(Path(m).stem, learned.load_model(m)) for m in a.models]
    rows = []
    for it in a.items:
        gid, b = it.split("@")
        b = int(b)
        g = L["grains"][gid]
        img = learned._registered(src, shifts, b)
        h = a.half
        x0, y0 = int(round(g["x"])) - h, int(round(g["y"])) - h
        x0, y0 = min(max(x0, 0), img.shape[1] - 2 * h), min(max(y0, 0), img.shape[0] - 2 * h)
        sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        m = float(np.median(early))
        panels = [_panel(img[sl], m - 60, m + 40, a.scale), _panel((img - early)[sl], -30, 30, a.scale)]
        names = [f"{gid}@{b}", "change"]
        for name, net in nets:
            p = learned.tube_probability(net, img, early, late)
            panels.append(_panel(p[sl], 0, 1, a.scale))
            names.append(f"{name} (frame)")
            if a.crop_too:
                pc = learned.tube_probability(net, img[sl], early[sl], late[sl])
                panels.append(_panel(pc, 0, 1, a.scale))
                names.append(f"{name} (crop)")
        tr = (L["labels"].get(gid, {}).get("traces", {}) or {}).get(str(b))
        for k, pnl in enumerate(panels):
            if tr and tr.get("path_xy_ref"):
                q = (np.asarray(tr["path_xy_ref"]) - [x0, y0]) * a.scale
                if k != 0:  # keep the raw bin unmarked
                    cv2.polylines(pnl, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, (0, 200, 0), 1)
            cv2.putText(pnl, names[k], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        rows.append(np.hstack(panels))
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out} {sheet.shape}")


if __name__ == "__main__":
    main()
