"""Mine weak grain labels from verified FRST detections (H191).

Appends label_type='grain' rows (provenance 'frst-weak-v1') for every
FRST-tiled detection on every dataset frame, and flips grain_reviewed
so the grain channel trains.  FRST field precision ~0.9 adjudicated;
residual junk becomes tolerable label noise.  Never touches existing
rows.  Usage: mine_grain_labels.py --dataset DIR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototypes.timesfm_tip_forecast.grain_detect import detect_grains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--threshold", type=float, default=0.20)
    a = ap.parse_args()

    ds = Path(a.dataset)
    man_frames = pd.DataFrame(
        __import__("json").load(open(ds / "manifest.json"))["frames"]
    )
    labels = pd.read_csv(ds / "labels.csv")
    reviews = pd.read_csv(ds / "frame_reviews.csv")
    rows = []
    for _, rec in man_frames.iterrows():
        fr = cv2.imread(str(ds / "frames" / rec.image_name))
        assert fr is not None, rec.image_name
        dets = detect_grains(
            cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), threshold_frac=a.threshold
        )
        for (x, y, s) in dets:
            rows.append({"image_name": rec.image_name,
                         "source_frame": int(rec.source_frame),
                         "label_type": "grain", "x": round(x, 1),
                         "y": round(y, 1), "provenance": "frst-weak-v1",
                         "confidence": 0.9})
        print(f"{rec.image_name}: {len(dets)} grains", flush=True)
    grain = pd.DataFrame(rows)
    pd.concat([labels, grain], ignore_index=True).to_csv(
        ds / "labels.csv", index=False)
    reviews["grain_reviewed"] = 1
    reviews.to_csv(ds / "frame_reviews.csv", index=False)
    print(f"appended {len(grain)} grain labels; "
          f"grain_reviewed=1 on {len(reviews)} frames")


if __name__ == "__main__":
    main()
