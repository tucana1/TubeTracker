"""Evaluate the weak-label tip CNN on eye-verified probe frames (H155).

For each probe (movie frame + accepted tip + tag): run the tip-channel
heatmap, extract peaks, report distance from the strongest peak to the
accepted tip. On clean frames the peak should sit on the accepted tip;
on fault frames the interesting question is whether the peak fires at
the TRUE apex instead of the accepted (phantom) tip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tubetracker.cnn_prototype import (
    extract_heatmap_points,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probes", required=True,
                    help="CSV: movie,source_frame,tip_x,tip_y,tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.35)
    a = ap.parse_args()

    import torch

    device = torch.device(a.device)
    model, payload = load_cnn_checkpoint(a.checkpoint, device=device)
    model.eval()
    print("checkpoint provenance:",
          payload.get("training_metadata", {}).get("provenance", "?"))

    probes = pd.read_csv(a.probes)
    rows = []
    for _, p in probes.iterrows():
        cap = cv2.VideoCapture(p.movie)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(p.source_frame))
        ok, fr = cap.read()
        cap.release()
        if not ok:
            print("MISS", p.tag)
            continue
        with torch.no_grad():
            hm = predict_heatmaps_tiled(model, fr, device)
        tip = np.asarray(hm[1])
        peaks = extract_heatmap_points(tip, threshold=a.threshold)
        d0 = (float(np.hypot(peaks[0][0] - p.tip_x, peaks[0][1] - p.tip_y))
              if peaks else float("nan"))
        # H155 fix: dense frames hold dozens of real tips — the metric is
        # nearest-peak recall + heat at the truth, not argmax distance.
        dn = (min(float(np.hypot(x - p.tip_x, y - p.tip_y)) for x, y, _ in peaks)
              if peaks else float("nan"))
        hx = min(max(int(round(p.tip_x)), 0), tip.shape[1] - 1)
        hy = min(max(int(round(p.tip_y)), 0), tip.shape[0] - 1)
        rows.append({
            "tag": p.tag, "n_peaks": len(peaks),
            "peak0_conf": round(peaks[0][2], 3) if peaks else 0.0,
            "dist_argmax": round(d0, 1),
            "dist_nearest": round(dn, 1),
            "heat_at_truth": round(float(tip[hy, hx]), 3),
            "heatmap_max": round(float(tip.max()), 3),
        })
        print(rows[-1], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)
    print("wrote", out / "eval.csv")


if __name__ == "__main__":
    main()
