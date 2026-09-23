"""CLEAN forecast-overlay videos: observed tip trail + TimesFM-3 forecast cone.

For each chosen origin two holds are rendered from the RAW movie:
  A (context end):   past tip trail (white) + observed tip (red).
  B (horizon end):   past trail + forecast mean tip (cyan dot) +
                     q10-q90 uncertainty rect (cyan) + actual tip (red).
CLEAN = frame numbers ON, no grain circles, no titles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from .forecast import Q10, Q90, TipForecaster
from .tip_tracks import load_tip_tracks, tip_speed


def pick_origins(csv_path: Path, horizon: int) -> list[int]:
    df = pd.read_csv(csv_path)
    lead = df[df.lead == horizon].sort_values("origin").reset_index(drop=True)
    # Priority: burst case first (most informative), then typical, then best.
    # Enforce min separation = horizon so segments never run backward.
    ordered: list[int] = []
    typical = lead.iloc[int(np.argsort(lead.tip_err.to_numpy())[len(lead) // 2])]
    bursts = lead[(lead.actual_gain > 5.0)].sort_values("burst_p", ascending=False)
    cands: list[int] = []
    if len(bursts):
        cands.append(int(bursts.iloc[0].origin))
    cands.append(int(typical.origin))
    best = lead.iloc[int(np.argmin(lead.tip_err.to_numpy()))]
    cands.append(int(best.origin))
    for c in cands:
        if all(abs(c - k) >= horizon for k in ordered):
            ordered.append(c)
    return sorted(ordered)[:3]


def draw_label(img: np.ndarray, text: str) -> None:
    cv.rectangle(img, (4, 4), (190, 26), (0, 0, 0), -1)
    cv.putText(img, text, (9, 21), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--backtest-out", required=True)
    ap.add_argument("--owner", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--origins", default=None)
    ap.add_argument("--crop-radius-px", type=int, default=110)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tracks = load_tip_tracks(args.measurements, [args.owner])
    track = tracks[args.owner]
    csv_path = Path(args.backtest_out) / f"backtest_P{args.owner}_H{args.horizon}.csv"
    origins = (
        [int(s) for s in args.origins.split(",")]
        if args.origins
        else pick_origins(csv_path, args.horizon)
    )
    forecaster = TipForecaster()
    cap = cv.VideoCapture(args.movie)
    if not cap.isOpened():
        raise RuntimeError("could not open movie")

    arr = track[["tip_x", "tip_y", "length"]].to_numpy(float).T
    speed = tip_speed(track)
    acc = track.accepted.to_numpy(float)
    times = track.time_minutes.to_numpy(float)
    frames_out: list[np.ndarray] = []

    def grab(source_frame: int) -> np.ndarray:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not decode source {source_frame}")
        return frame

    def crop_around(frame: np.ndarray, pts: np.ndarray):
        """Crop centered on the bounding box of pts (trail + markers).

        The old fixed-center crop dropped markers whenever forecast and
        actual disagreed by more than the radius; centering on the joint
        bounding box keeps every element on screen.
        """
        r = args.crop_radius_px
        x0 = float(np.min(pts[:, 0]))
        x1 = float(np.max(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        y1 = float(np.max(pts[:, 1]))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(r, (x1 - x0) / 2 + 20.0, (y1 - y0) / 2 + 20.0)
        ox0 = max(0, int(cx - half))
        oy0 = max(0, int(cy - half))
        ox1 = min(frame.shape[1], int(cx + half))
        oy1 = min(frame.shape[0], int(cy + half))
        return frame[oy0:oy1, ox0:ox1].copy(), ox0, oy0

    def draw_trail(crop: np.ndarray, xy: np.ndarray, gap: np.ndarray) -> None:
        """White trail; gap-crossing segments draw dimmed, not dropped.

        Consecutive accepted rows can be far apart in time/space when
        diagnostic rows intervene. Dropping those segments left a visible
        gap before the tip dot; dimming keeps the trail connected while
        marking the interpolation as unobserved.
        """
        for k in range(1, len(xy)):
            color = (255, 255, 255) if gap[k] <= 1 else (170, 170, 170)
            cv.line(crop, tuple(np.rint(xy[k - 1]).astype(int)),
                    tuple(np.rint(xy[k]).astype(int)), color, 2, cv.LINE_AA)

    for t in origins:
        T = t + 1
        ctx = arr[:, :T].astype(np.float32)
        pc = np.stack([acc[:T], speed[:T]]).astype(np.float32)
        dt = float(np.median(np.diff(times[:T]))) if T > 2 else 0.25
        fut = np.concatenate([times[:T], times[T - 1] + dt * np.arange(1, args.horizon + 1)])
        fc, qu = forecaster.predict_one(ctx, pc, fut, args.horizon)

        # Hold A: context end.
        src_a = int(track.source_frame.iloc[t])
        tip_a = arr[:2, t]
        frame = grab(src_a)
        lo = max(0, t - 24)
        seg_idx = track.sample_index.to_numpy()[lo:T]
        seg_gap = np.diff(seg_idx, prepend=seg_idx[0] - 1)
        crop, x0, y0 = crop_around(frame, arr[:2, lo:T].T)
        draw_trail(
            crop,
            arr[:2, lo:T].T - np.array([x0, y0]),
            seg_gap,
        )
        cv.circle(crop, tuple(np.rint(tip_a - [x0, y0]).astype(int)), 5, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(crop, tuple(np.rint(tip_a - [x0, y0]).astype(int)), 4, (0, 0, 255), -1, cv.LINE_AA)
        draw_label(crop, f"frame {src_a} observed")
        frames_out += [crop] * 6

        # Hold B: horizon end. Joint bbox of trail + box + both tips.
        fx, fy = float(fc[0, -1]), float(fc[1, -1])
        ax, ay = float(arr[0, t + args.horizon]), float(arr[1, t + args.horizon])
        x10, x90 = float(qu[0, -1, Q10]), float(qu[0, -1, Q90])
        y10, y90 = float(qu[1, -1, Q10]), float(qu[1, -1, Q90])
        src_b = int(track.source_frame.iloc[t + args.horizon])
        frame = grab(src_b)
        lo_b = max(0, t - 24)
        seg_idx_b = track.sample_index.to_numpy()[lo_b : t + args.horizon + 1]
        seg_gap_b = np.diff(seg_idx_b, prepend=seg_idx_b[0] - 1)
        all_pts = np.vstack(
            [arr[:2, lo_b : t + args.horizon + 1].T,
             np.array([[fx, fy], [ax, ay], [x10, y10], [x90, y90]])]
        )
        crop, x0, y0 = crop_around(frame, all_pts)
        draw_trail(
            crop,
            arr[:2, lo_b : t + args.horizon + 1].T - np.array([x0, y0]),
            seg_gap_b,
        )
        cv.rectangle(crop, (int(x10 - x0), int(y10 - y0)), (int(x90 - x0), int(y90 - y0)),
                     (255, 255, 0), 1, cv.LINE_AA)
        # Red first, cyan on top: a hit reads as a cyan-centered bullseye
        # instead of the forecast dot vanishing under the actual.
        cv.circle(crop, (int(ax - x0), int(ay - y0)), 5, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(crop, (int(ax - x0), int(ay - y0)), 4, (0, 0, 255), -1, cv.LINE_AA)
        cv.circle(crop, (int(fx - x0), int(fy - y0)), 3, (0, 0, 0), -1, cv.LINE_AA)
        cv.circle(crop, (int(fx - x0), int(fy - y0)), 2, (255, 255, 0), -1, cv.LINE_AA)
        draw_label(crop, f"frame {src_b} forecast vs actual")
        frames_out += [crop] * 10
        print(f"origin {t}: fc=({fx:.1f},{fy:.1f}) actual=({ax:.1f},{ay:.1f}) "
              f"err={np.hypot(fx-ax, fy-ay):.1f}px", flush=True)

    h, w = frames_out[0].shape[:2]
    tw, th = (w // 2) * 2, (h // 2) * 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv.VideoWriter(str(out), cv.VideoWriter_fourcc(*"mp4v"), args.fps, (tw, th))
    for fr in frames_out:
        vw.write(cv.resize(fr, (tw, th)) if fr.shape[:2] != (h, w) else fr)
    vw.release()
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
