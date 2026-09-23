"""Relabel weak tip labels with validated tip corrections (H181).

Tip labels in the training set ARE the accepted tracker tips, which ride
10-23 px off-apex on biased samples (P58 s210/s220) — 75% of training
patches anchor on them, so the model learns biased supervision.  This
script joins each tip label back to (owner, sample) via measurements,
runs the validated tip_correct snap with centerline tails, and rewrites
labels that move >= --min-move px (provenance 'tip-correct-v1').

Usage:
  relabel_tip_labels.py --dataset DIR --checkpoint PT --out labels.csv
      [--limit N] [--min-move 3.0]
Writes the corrected table to --out (never overwrites in place) plus a
move report on stdout.  Frames come from the dataset itself, so no
movie paths are needed; owner/sample/tails resolve through the
measurements + centerlines referenced below (kept next to the runs,
not the dataset, by design).
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

from prototypes.timesfm_tip_forecast.tip_correct import correct_tip
from tubetracker.cnn_prototype import (
    extract_heatmap_points,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)

BASE = Path("/Users/joshjiang/Documents/TubeTracker")
DENSE_MEAS = BASE / "runs/prototypes/v29/causal_growth_front/dense_full_v29_33_1/measurements.csv"
DENSE_CL = BASE / "runs/prototypes/v29/causal_growth_front/dense_full_v29_33_1/centerlines.csv"
LOW_MEAS = BASE / "runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3/measurements.csv"
LOW_CL = BASE / "runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3/centerlines.csv"


def _tails(field: str, pid: int, s: int, meas: pd.DataFrame,
           cl: pd.DataFrame | None) -> np.ndarray | None:
    if cl is not None:
        sub = cl[(cl.pollen_id == pid) & (cl.sample_index == s)]
        sub = sub.sort_values("point_index")
        if len(sub) >= 8:
            return sub[["source_x_px", "source_y_px"]].to_numpy(float)[-8:]
    dd = meas[(meas.pollen_id == pid) & (meas.accepted == 1)]
    dd = dd.sort_values("sample_index")
    pr = dd[dd.sample_index <= s].tail(8)
    if len(pr) >= 6:
        return pr[["tip_x_px", "tip_y_px"]].to_numpy(float)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-move", type=float, default=3.0)
    a = ap.parse_args()

    ds = Path(a.dataset)
    labels = pd.read_csv(ds / "labels.csv")
    tips = labels[labels.label_type == "tip"].reset_index(drop=True)
    if a.limit:
        tips = tips.iloc[: a.limit]

    meas = {
        "dense": pd.read_csv(DENSE_MEAS),
        "lowdens": pd.read_csv(LOW_MEAS),
    }
    cls = {
        "dense": pd.read_csv(DENSE_CL),
        "lowdens": pd.read_csv(LOW_CL) if LOW_CL.exists() else None,
    }
    device = torch.device("cpu")
    model, _ = load_cnn_checkpoint(Path(a.checkpoint), device=device)
    model.eval()

    moves = []
    corrected: dict[int, tuple[float, float]] = {}
    # One CNN inference per frame; labels on the same image share peaks.
    for image_name, group in tips.groupby("image_name", sort=False):
        field = "dense" if image_name.startswith("dense_") else "lowdens"
        fr = cv2.imread(str(ds / "frames" / image_name))
        assert fr is not None, image_name
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            hm = predict_heatmaps_tiled(model, fr, device)
        peaks = extract_heatmap_points(np.asarray(hm[1]), threshold=0.30)
        for idx, lab in group.iterrows():
            m = meas[field]
            cand = m[(m.accepted == 1)
                     & (m.source_frame == lab.source_frame)].copy()
            if not len(cand):
                continue
            cand["d"] = np.hypot(cand.tip_x_px - lab.x,
                                 cand.tip_y_px - lab.y)
            hit = cand.sort_values("d").iloc[0]
            if hit.d > 5.0:
                continue
            tail = _tails(field, int(hit.pollen_id),
                          int(hit.sample_index), m, cls[field])
            if tail is None:
                continue
            fixed = correct_tip(g, tail, peaks)
            if fixed is not None:
                mv = float(np.hypot(fixed[0] - lab.x, fixed[1] - lab.y))
                if mv >= a.min_move:
                    corrected[idx] = fixed
                    moves.append(mv)
        print(f"frame {image_name}: {len(group)} tips, "
              f"{sum(i in corrected for i in group.index)} moved", flush=True)
    out = tips.copy()
    for idx, (fx, fy) in corrected.items():
        out.loc[idx, "x"] = fx
        out.loc[idx, "y"] = fy
        out.loc[idx, "provenance"] = "tip-correct-v1"
    rest = labels[labels.label_type != "tip"]
    pd.concat([out, rest], ignore_index=True).to_csv(a.out, index=False)
    mv = np.array(moves)
    print(f"labels={len(tips)} joined+corrected={len(mv)} "
          f"move_mean={mv.mean() if len(mv) else 0:.1f} "
          f"move_max={mv.max() if len(mv) else 0:.1f} wrote={a.out}",
          flush=True)


if __name__ == "__main__":
    main()
