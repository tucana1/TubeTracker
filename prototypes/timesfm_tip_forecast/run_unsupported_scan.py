"""Full-field UNSUPPORTED scan (H158): CNN-confirm every accepted tip.

For each accepted tip sample: run the tip-channel heatmap on its source
frame (cached per frame), confirm iff a strong peak sits within
CONFIRM_PX. Emit per-owner unsupported counts + rates. Design use: run
once per checkpoint; corroborate veto/root/rollout flags with an
independent appearance instrument. Expensive (full-frame tiled inference
per sampled frame) — sample sparsely (default every 8th sample) and
restrict owners via --owners (default: root-flagged + veto-flagged).
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--owners", default="")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.30)
    a = ap.parse_args()

    device = torch.device(a.device)
    model, _ = load_cnn_checkpoint(a.checkpoint, device=device)
    model.eval()

    m = pd.read_csv(a.measurements)
    owners = ([int(x) for x in a.owners.split(",")] if a.owners
              else sorted(m[m.accepted == 1].pollen_id.unique()))
    cap = cv2.VideoCapture(a.movie)
    peaks_by_src: dict[int, list] = {}
    rows = []
    for pid in owners:
        d = m[(m.pollen_id == pid) & (m.accepted == 1)].sort_values("sample_index")
        sub = d.iloc[:: a.stride]
        un = 0
        for _, row in sub.iterrows():
            src = int(row.source_frame)
            if src not in peaks_by_src:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                ok, fr = cap.read()
                assert ok, src
                with torch.no_grad():
                    hm = predict_heatmaps_tiled(model, fr, device)
                peaks_by_src[src] = extract_heatmap_points(
                    np.asarray(hm[1]), threshold=a.threshold)
            peaks = [p for p in peaks_by_src[src] if p[2] >= MIN_CONF]
            if not any(np.hypot(x - row.tip_x_px, y - row.tip_y_px) <= CONFIRM_PX
                       for (x, y, _) in peaks):
                un += 1
        rows.append({"pollen_id": pid, "n_scanned": len(sub),
                     "n_unsupported": un,
                     "unsupported_rate": round(un / max(len(sub), 1), 3)})
        print(rows[-1], flush=True)
    cap.release()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "unsupported.csv", index=False)
    print("wrote", out / "unsupported.csv")


if __name__ == "__main__":
    main()
