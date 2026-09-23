"""Full-series tip correction runner (H190): accepted vs corrected tips.

For one owner: per accepted sample, run the tip CNN, extract peaks,
run tip_correct with the centerline tail (dense) or accepted-tip
history (lowdens), and write corrected_P<id>.csv
(sample_index, source_frame, accepted_x, accepted_y,
corrected_x, corrected_y, moved_px).  Refusals keep accepted coords
with moved_px = 0.  Feeds the corrected-tip overlay video.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--centerlines", default="")
    ap.add_argument("--movie", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--owner", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--replay", default="",
                    help="H198: guided-replay CSV for this owner; samples "
                    "with a fault-suspect verdict refuse entirely (TimesFM "
                    "fault gate inside the corrector).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.measurements)
    d = m[(m.pollen_id == a.owner) & (m.accepted == 1)].sort_values(
        "sample_index").iloc[:: a.stride]
    cl = (pd.read_csv(a.centerlines)
          if a.centerlines else None)
    device = torch.device("cpu")
    model, _ = load_cnn_checkpoint(Path(a.checkpoint), device=device)
    model.eval()
    faults: set[int] = set()
    if a.replay:
        rep = pd.read_csv(a.replay)
        faults = set(
            rep[rep.verdict == "fault-suspect"].sample_index.astype(int))
        print(f"fault-gated samples: {len(faults)}", flush=True)
    cap = cv2.VideoCapture(a.movie)
    rows = []
    for _, row in d.iterrows():
        s = int(row.sample_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            hm = predict_heatmaps_tiled(model, fr, device)
        peaks = extract_heatmap_points(np.asarray(hm[1]), threshold=0.30)
        tail = None
        if cl is not None:
            sub = cl[(cl.pollen_id == a.owner)
                     & (cl.sample_index == s)].sort_values("point_index")
            if len(sub) >= 8:
                tail = sub[["source_x_px", "source_y_px"]].to_numpy(float)
        if tail is None:
            pr = d[d.sample_index <= s].tail(10)
            if len(pr) >= 6:
                tail = pr[["tip_x_px", "tip_y_px"]].to_numpy(float)
        ax, ay = float(row.tip_x_px), float(row.tip_y_px)
        cx, cy, mv, tier = ax, ay, 0.0, -1
        if tail is not None:
            fixed = correct_tip(g, tail, peaks, at_fault=s in faults,
                                return_candidate=True)
            if fixed is not None:
                assert len(fixed) == 3  # return_candidate=True -> (x, y, tier)
                cx, cy, tier = float(fixed[0]), float(fixed[1]), int(fixed[2])
                mv = float(np.hypot(cx - ax, cy - ay))
        rows.append({"sample_index": s, "source_frame": int(row.source_frame),
                     "accepted_x": ax, "accepted_y": ay,
                     "corrected_x": cx, "corrected_y": cy,
                     "moved_px": round(mv, 1), "tier": tier})
        print(f"s{s}: moved {mv:.1f}", flush=True)
    cap.release()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    moved = [r["moved_px"] for r in rows]
    print(f"wrote {out}: n={len(rows)} fixed={sum(v > 0 for v in moved)} "
          f"max_move={max(moved) if moved else 0:.1f}")


if __name__ == "__main__":
    main()
