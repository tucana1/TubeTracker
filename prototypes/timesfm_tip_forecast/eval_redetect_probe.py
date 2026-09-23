"""Probe the forecast-gated re-detection on eye-verified samples (H156)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototypes.timesfm_tip_forecast.guided_redetect import redetect_sample
from tubetracker.cnn_prototype import (
    extract_heatmap_points,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)

PROBES = [
    # (measurements, replay, owner, sample, tag)
    ("dense", 58, 150, "P58-clean"),
    ("dense", 3, 150, "P3-clean"),
    ("dense", 131, 206, "P131-foreign"),
    ("dense", 55, 240, "P55-background"),
    ("lowdens", 22, 200, "P22-clean"),
    ("lowdens", 22, 244, "P22-ghost"),
    ("lowdens", 11, 251, "P11-clean"),
    ("lowdens", 13, 213, "P13-clean"),
]

BASE = Path("/Users/joshjiang/Documents/TubeTracker")
MOV = {
    "dense": "/Users/joshjiang/Downloads/Pollen tube movie 1 7-14-26.mp4",
    "lowdens": "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4",
}
MEAS = {
    "dense": BASE / "runs/prototypes/v29/causal_growth_front/dense_full_v29_33_1/measurements.csv",
    "lowdens": BASE / "runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3/measurements.csv",
}
REP = {
    "dense": BASE / "runs/prototypes/timesfm/guided_replay_full_v2",
    "lowdens": BASE / "runs/prototypes/timesfm/guided_replay_lowdens",
}


def main() -> None:
    device = torch.device("cpu")
    model, _ = load_cnn_checkpoint(
        BASE / "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt",
        device=device,
    )
    model.eval()
    for field, pid, s, tag in PROBES:
        m = pd.read_csv(MEAS[field])
        row = m[(m.pollen_id == pid) & (m.sample_index == s)].iloc[0]
        cap = cv2.VideoCapture(MOV[field])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
        ok, fr = cap.read()
        cap.release()
        assert ok, tag
        with torch.no_grad():
            hm = predict_heatmaps_tiled(model, fr, device)
        peaks = extract_heatmap_points(np.asarray(hm[1]), threshold=0.30)
        rep = pd.read_csv(REP[field] / f"replay_P{pid}.csv")
        vrow = rep[rep.sample_index == s]
        verdict = vrow.verdict.iloc[0] if len(vrow) else "keep"
        d = m[(m.pollen_id == pid) & (m.accepted == 1)].sort_values("sample_index")
        prior = d[d.sample_index < s].tail(10)
        keep_hist = [(float(r.tip_x_px), float(r.tip_y_px)) for _, r in prior.iterrows()]
        g, held, t = redetect_sample(
            (float(row.tip_x_px), float(row.tip_y_px)), verdict, peaks,
            keep_hist, None,
        )
        nstrong = sum(1 for p in peaks if p[2] >= 0.30)
        print(f"{tag}: verdict={verdict} peaks={len(peaks)}/{nstrong} "
              f"-> guided={'HOLD' if held else 'pass'} tag={t}", flush=True)


if __name__ == "__main__":
    main()
