"""Side-by-side probability maps of two models on real traces: bin, change from "before", P(model A), P(model B),
with the human trace drawn (green) on each tile.

    python -m prototypes.synth_v6.mapsheet m2 tubes_synth_v1 tubes_v6r1_ft g085:194 g092:52 ... --out sheet.png
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer

from .sheet import View

REPO = Path(__file__).resolve().parents[2]


def main(movie: str, model_a: str, model_b: str, items: list[str], out: str, half: int = 32, zoom: int = 3) -> None:
    view = View(REPO / f"runs/sparsetrack/{movie}")
    P = {m: Renderer(*stack.load(REPO / f"runs/sparsetrack/{movie}/prob_{m}")) for m in (model_a, model_b)}
    L = json.loads((REPO / f"benchmark/labels/{movie}_v1.json").read_text())
    rows = []
    for it in items:
        gid, b = it.split(":")
        b = int(b)
        g = L["grains"][gid]
        t = L["labels"][gid]["traces"].get(str(b), {})
        off = t.get("view_offset") or [0.0, 0.0]
        cx, cy = g["x"] + off[0], g["y"] + off[1]
        early, imgs, chg = view.crops(cx, cy, [b], half)
        lo, hi = np.percentile(np.concatenate([early.ravel(), imgs[0].ravel()]), [0.5, 99.5])
        tiles = [np.clip((imgs[0] - lo) / (hi - lo) * 255, 0, 255), np.clip((chg[0] / 30 + 1) * 127.5, 0, 255)]
        for m in (model_a, model_b):
            tiles.append(np.clip(np.nan_to_num(P[m].crop(b, cx, cy, half)) / 250 * 255, 0, 255))
        cols = []
        for j, tile in enumerate(tiles):
            u = cv2.cvtColor(cv2.resize(tile.astype(np.uint8), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2BGR)
            path = np.asarray(t.get("path_xy_ref") or [], float)
            if len(path) >= 2:
                q = ((path - [cx - half, cy - half]) * zoom).astype(np.int32)
                cv2.polylines(u, [q], False, (0, 200, 0), 1, cv2.LINE_AA)
            label = [f"{gid} b{b} {t.get('length_px', 0):.1f}px", "change +/-30", model_a, model_b][j]
            cv2.putText(u, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
            cols.append(u)
        rows.append(np.concatenate(cols, axis=1))
    cv2.imwrite(str(REPO / out), np.concatenate(rows, axis=0))
    print(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("items", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.movie, a.model_a, a.model_b, a.items, a.out)
