"""Build weak-tip ignore lists for validity-mask training (P2 data wiring).

For every dataset frame, join accepted measurement tips by source frame
(dense and lowdens movies separately); tips farther than MATCH_PX from
any reviewed label become weak (ignore-disc) points. Output JSON:
{image_name: {source_frame, labeled_n, weak_xy: [[x, y], ...]}}.

No images read, no training launched — pure CSV geometry. The training
loop consumes this manifest to build per-patch validity masks.
"""

import argparse
import json

import numpy as np
import pandas as pd

MATCH_PX = 8.0


def build(manifest_path: str, out_path: str) -> dict:
    lab = pd.read_csv("runs/prototypes/timesfm/tip_dataset_v2/labels.csv")
    dense = pd.read_csv(
        "runs/prototypes/v29/causal_growth_front/"
        "dense_full_v29_33_1/measurements.csv")
    lowdens = pd.read_csv(
        "runs/prototypes/v29/causal_growth_front/"
        "lowdens_full_v29_19_3/measurements.csv")
    movies = {"dense": dense, "lowdens": lowdens}
    out: dict = {}
    for img, g in lab.groupby("image_name"):
        tag = str(img).split("_")[0]
        m = movies.get(tag)
        if m is None:
            continue
        src = int(g.source_frame.iloc[0])
        acc = m[(m.accepted == 1) & (m.source_frame == src)]
        lp = g[["x", "y"]].to_numpy(float)
        weak = []
        for _, r in acc.iterrows():
            ax, ay = float(r.tip_x_px), float(r.tip_y_px)
            if len(lp) == 0 or np.hypot(lp[:, 0] - ax, lp[:, 1] - ay).min() > MATCH_PX:
                weak.append([round(ax, 1), round(ay, 1)])
        out[str(img)] = {
            "source_frame": src,
            "movie": tag,
            "labeled_n": int(len(g)),
            "weak_n": len(weak),
            "weak_xy": weak,
        }
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = build("", a.out)
    lab_n = sum(v["labeled_n"] for v in out.values())
    weak_n = sum(v["weak_n"] for v in out.values())
    print(f"frames={len(out)} labeled={lab_n} weak={weak_n} -> {a.out}")


if __name__ == "__main__":
    main()
