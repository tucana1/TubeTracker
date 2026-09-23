#!/usr/bin/env python3
"""Render CLEAN per-owner crop timelines from a v29 challenge run.

CLEAN = no grain circle, no title text; frame numbers ON; red tip dot ok.
Each owner gets one timeline video: native crops centered on the tube path
at evenly spaced accepted samples, tip marked red, frame number labeled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", default=None)
    parser.add_argument("--stages", type=int, default=9)
    parser.add_argument("--crop-radius-px", type=int, default=90)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    measurements = pd.read_csv(run_dir / "measurements.csv")
    centerlines = pd.read_csv(run_dir / "centerlines.csv")
    wanted = (
        sorted(int(v) for v in args.track_ids.split(","))
        if args.track_ids
        else sorted(int(v) for v in measurements.pollen_id.unique())
    )
    capture = cv.VideoCapture(str(args.movie.expanduser()))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {args.movie}")
    args.output.mkdir(parents=True, exist_ok=True)
    made = []
    for pid in wanted:
        owner_meas = measurements[measurements.pollen_id == pid]
        accepted = owner_meas[owner_meas.accepted == 1]
        if not len(accepted):
            continue
        picks = np.unique(
            np.rint(
                np.quantile(
                    accepted.sample_index.to_numpy(),
                    np.linspace(0.0, 1.0, args.stages),
                )
            ).astype(int)
        )
        frames = []
        for sample in picks:
            rows = centerlines[
                (centerlines.pollen_id == pid)
                & (centerlines.sample_index == int(sample))
            ].sort_values("point_index")
            if not len(rows):
                continue
            pts = np.column_stack(
                (rows.source_x_px.to_numpy(), rows.source_y_px.to_numpy())
            )
            source_frame = int(
                owner_meas[owner_meas.sample_index == int(sample)].source_frame.iloc[0]
            )
            capture.set(cv.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode source {source_frame}")
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            x0 = max(0, int(cx - args.crop_radius_px))
            y0 = max(0, int(cy - args.crop_radius_px))
            x1 = min(frame.shape[1], int(cx + args.crop_radius_px))
            y1 = min(frame.shape[0], int(cy + args.crop_radius_px))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            crop = frame[y0:y1, x0:x1].copy()
            local = pts - np.asarray((x0, y0))
            for k in range(1, len(local)):
                cv.line(
                    crop,
                    tuple(np.rint(local[k - 1]).astype(int)),
                    tuple(np.rint(local[k]).astype(int)),
                    (0, 255, 255),
                    1,
                    cv.LINE_AA,
                )
            tip = tuple(np.rint(local[-1]).astype(int))
            cv.circle(crop, tip, 4, (0, 0, 255), -1, cv.LINE_AA)
            label = f"frame {source_frame}"
            cv.rectangle(crop, (4, 4), (150, 26), (0, 0, 0), -1)
            cv.putText(
                crop, label, (9, 21), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv.LINE_AA,
            )
            frames.append(crop)
        if not frames:
            continue
        out = args.output / f"P{pid:02d}_timeline.mp4"
        widths = {frame.shape[1] for frame in frames}
        heights = {frame.shape[0] for frame in frames}
        target_w, target_h = max(widths), max(heights)
        target_w += target_w % 2
        target_h += target_h % 2
        frames = [cv.resize(frame, (target_w, target_h)) for frame in frames]
        size = (target_w, target_h)
        writer = cv.VideoWriter(
            str(out), cv.VideoWriter_fourcc(*"mp4v"), args.fps, size,
        )
        if not writer.isOpened():
            raise RuntimeError("could not open timeline video writer")
        for frame in frames:
            writer.write(frame)
        writer.release()
        made.append(str(out))
        print(f"P{pid}: {len(frames)} stages -> {out}")
    capture.release()
    print(f"wrote {len(made)} timelines")


if __name__ == "__main__":
    main()
