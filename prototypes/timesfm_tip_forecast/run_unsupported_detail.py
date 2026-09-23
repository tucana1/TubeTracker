"""Per-sample UNSUPPORTED detail for the lowdens pilot (H167 follow-up).

Same logic as run_unsupported_scan.py but records every sampled
(sample_index, source_frame, confirmed, dist_nearest, heat_at_truth)
so unsupported hits can be localized (tail vs mid-life) and eye-checked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototypes.timesfm_tip_forecast.guided_redetect import CONFIRM_PX, MIN_CONF
from tubetracker.cnn_prototype import (
    extract_heatmap_points,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--owners", required=True)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.30)
    a = ap.parse_args()

    device = torch.device("cpu")
    model, _ = load_cnn_checkpoint(a.checkpoint, device=device)
    model.eval()

    m = pd.read_csv(a.measurements)
    cap = cv2.VideoCapture(a.movie)
    peaks_by_src: dict[int, list] = {}
    rows = []
    for pid in [int(x) for x in a.owners.split(",")]:
        d = m[(m.pollen_id == pid) & (m.accepted == 1)].sort_values("sample_index")
        for _, row in d.iloc[:: a.stride].iterrows():
            src = int(row.source_frame)
            if src not in peaks_by_src:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                ok, fr = cap.read()
                assert ok, src
                with torch.no_grad():
                    hm = predict_heatmaps_tiled(model, fr, device)
                tip = np.asarray(hm[1])
                peaks_by_src[src] = (extract_heatmap_points(tip, threshold=a.threshold), tip)
            peaks, tip = peaks_by_src[src]
            strong = [p for p in peaks if p[2] >= MIN_CONF]
            dn = (min(float(np.hypot(x - row.tip_x_px, y - row.tip_y_px)) for x, y, _ in strong)
                  if strong else float("nan"))
            hx = min(max(int(round(row.tip_x_px)), 0), tip.shape[1] - 1)
            hy = min(max(int(round(row.tip_y_px)), 0), tip.shape[0] - 1)
            rows.append({
                "pollen_id": pid, "sample_index": int(row.sample_index),
                "source_frame": src,
                "confirmed": bool(dn <= CONFIRM_PX),
                "dist_nearest": round(dn, 1),
                "heat_at_truth": round(float(tip[hy, hx]), 3),
                "n_strong": len(strong),
            })
        print("done", pid, flush=True)
    cap.release()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "unsupported_detail.csv", index=False)
    print("wrote", out / "unsupported_detail.csv")


if __name__ == "__main__":
    main()
