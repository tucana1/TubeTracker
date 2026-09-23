"""Field-wide apex-visibility survey (H201).

For every accepted owner: window-max tip heatmap heat near the accepted
tip (25px window = apex evidence nearby) in early/mid/late thirds, plus
mean tube length and mean heat, to find correlates of late-life fade
(P8/P3 fade, P13/P22/P58 sustain).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from tubetracker.cnn_prototype import (  # noqa: E402
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)


def winmax(tip: np.ndarray, x: float, y: float, r: int = 25) -> float:
    H, W = tip.shape
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return round(float(tip[y0:y1, x0:x1].max()), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--per-sample", action="store_true",
                    help="H202: write per-sample (sample_index, visibility) "
                    "rows for TimesFM covariate use instead of thirds.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cpu")
    model, _ = load_cnn_checkpoint(Path(a.checkpoint), device=dev)
    model.eval()
    m = pd.read_csv(a.measurements)
    cap = cv2.VideoCapture(a.movie)
    rows = []
    for pid, d in m[m.accepted == 1].groupby("pollen_id"):
        d = d.sort_values("sample_index")
        hs: list[float] = []
        sidx: list[int] = []
        for _, r in list(d.iterrows())[:: a.stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.source_frame))
            ok, fr = cap.read()
            if not ok:
                continue
            with torch.no_grad():
                hm = predict_heatmaps_tiled(model, fr, dev)
            hs.append(
                winmax(np.asarray(hm[1]), float(r.tip_x_px), float(r.tip_y_px))
            )
            sidx.append(int(r.sample_index))
        if a.per_sample:
            for s, h in zip(sidx, hs):
                rows.append(
                    {"pollen_id": int(pid), "sample_index": s,
                     "visibility": h})
            print(f"owner {pid}: n={len(hs)} per-sample", flush=True)
            continue
        if len(hs) < 6:
            continue
        h = np.array(hs)
        n = len(h)
        rows.append(
            {
                "pollen_id": int(pid),
                "n": n,
                "early": round(float(h[: n // 3].mean()), 4),
                "mid": round(float(h[n // 3 : 2 * n // 3].mean()), 4),
                "late": round(float(h[2 * n // 3 :].mean()), 4),
                "fade": round(
                    float(h[: n // 3].mean() - h[2 * n // 3 :].mean()), 4
                ),
                "mean_len": round(float(d.tube_length_px.mean()), 1),
            }
        )
        print(f"owner {pid}: n={n} fade={rows[-1]['fade']}", flush=True)
    cap.release()
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} owners")


if __name__ == "__main__":
    main()
