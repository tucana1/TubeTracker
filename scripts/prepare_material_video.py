#!/usr/bin/env python3
"""Prepare a raw microscopy video for pollen-anchored material tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from tubetracker.curve_prototype import CurveTraceConfig, detect_and_track_grains


def parse_args():
    """Parse video sampling, image sizing, and grain-detection controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=480)
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Resize to this width before analysis; zero preserves source pixels.",
    )
    parser.add_argument("--min-grain-radius", type=int, default=6)
    parser.add_argument("--max-grain-radius", type=int, default=13)
    parser.add_argument("--grain-threshold", type=int, default=14)
    parser.add_argument("--min-circle-score", type=float, default=0.24)
    return parser.parse_args()


def source_frame_schedule(frame_count, sample_count):
    """Return unique source frames spanning the complete video period."""
    if frame_count <= 0:
        raise ValueError("video contains no readable frames")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(int(frame_count), int(sample_count))
    return np.unique(np.rint(np.linspace(0, frame_count - 1, count)).astype(int))


def _resize_gray(frame, image_width):
    """Convert a decoded frame to grayscale with an optional fixed-width resize."""
    if image_width > 0 and frame.shape[1] != image_width:
        scale = image_width / frame.shape[1]
        frame = cv.resize(
            frame,
            (image_width, max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC,
        )
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def sample_video(video_path, sample_count, image_width=0):
    """Decode an evenly spaced set of source frames without reading the full movie."""
    video_path = Path(video_path).expanduser().resolve()
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    source_frames = source_frame_schedule(frame_count, sample_count)
    grays = []
    for output_index, source_frame in enumerate(source_frames, start=1):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        grays.append(_resize_gray(frame, image_width))
        if output_index == 1 or output_index % 50 == 0:
            print(f"Decoded {output_index}/{len(source_frames)} sampled frames...")
    playback_fps = float(capture.get(cv.CAP_PROP_FPS))
    capture.release()
    return np.stack(grays), source_frames, {
        "video": str(video_path),
        "source_frame_count": frame_count,
        "source_playback_fps": playback_fps,
        "cached_width": int(grays[0].shape[1]),
        "cached_height": int(grays[0].shape[0]),
    }


def grain_cache_tables(tracks, frame_count, source_frames):
    """Convert tracked grain objects into stable arrays and a selectable catalog."""
    positions = np.empty((len(tracks), frame_count, 3), dtype=np.float32)
    rows = []
    for track_index, track in enumerate(tracks):
        pollen_id = f"P{track_index + 1:04d}"
        for frame_index in range(frame_count):
            roi = track.roi_closest_to(frame_index)
            positions[track_index, frame_index] = (
                float(roi.gv3.x),
                float(roi.gv3.y),
                0.5 * max(float(roi.w), float(roi.h)),
            )
        seed_frame = int(track.first_frame())
        seed_roi = track.roi_closest_to(seed_frame)
        rows.append(
            {
                "pollen_id": pollen_id,
                "seed_frame": seed_frame,
                "seed_source_frame": int(source_frames[seed_frame]),
                "x": float(positions[track_index, 0, 0]),
                "y": float(positions[track_index, 0, 1]),
                "radius": float(positions[track_index, 0, 2]),
                "circle_score": float(getattr(seed_roi, "circle_score", np.nan)),
                "tracking_eligible": True,
            }
        )
    ids = np.asarray([row["pollen_id"] for row in rows])
    return ids, positions, pd.DataFrame.from_records(rows)


def write_catalog_preview(path, first_gray, catalog):
    """Draw selectable pollen IDs on the first cached frame."""
    preview = cv.cvtColor(first_gray, cv.COLOR_GRAY2BGR)
    for row in catalog.itertuples(index=False):
        center = (int(round(row.x)), int(round(row.y)))
        radius = max(2, int(round(row.radius)))
        cv.circle(preview, center, radius, (255, 160, 0), 2, cv.LINE_AA)
        cv.putText(
            preview,
            row.pollen_id,
            (center[0] + radius + 2, center[1] - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 220),
            1,
            cv.LINE_AA,
        )
    if not cv.imwrite(str(path), preview):
        raise RuntimeError(f"could not write {path}")


def main():
    """Sample the video, track pollen grains, and write material-pipeline inputs."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grays, source_frames, metadata = sample_video(
        args.video, args.sample_count, args.image_width
    )
    config = CurveTraceConfig(
        min_grain_radius=args.min_grain_radius,
        max_grain_radius=args.max_grain_radius,
        grain_threshold=args.grain_threshold,
        min_grain_circle_score=args.min_circle_score,
    )
    print("Detecting and tracking pollen grains across sampled frames...")
    tracks, _, _ = detect_and_track_grains(list(grays), config)
    if not tracks:
        raise RuntimeError(
            "no pollen grains were detected; review the radius and threshold settings"
        )
    ids, positions, catalog = grain_cache_tables(
        tracks, len(grays), source_frames
    )
    np.save(output_dir / "gray_samples.npy", grays)
    np.save(output_dir / "source_frames.npy", source_frames)
    np.savez_compressed(
        output_dir / "grain_tracks.npz", ids=ids, positions=positions
    )
    catalog.to_csv(output_dir / "pollen_candidates.csv", index=False)
    write_catalog_preview(output_dir / "pollen_catalog.jpg", grays[0], catalog)
    manifest = {
        "workflow": "material_video_preparation_v1",
        "metadata": metadata,
        "sample_count": int(len(grays)),
        "grain_count": int(len(catalog)),
        "configuration": asdict(config),
        "outputs": {
            "gray_cache": "gray_samples.npy",
            "source_frames_cache": "source_frames.npy",
            "grain_cache": "grain_tracks.npz",
            "candidate_catalog": "pollen_candidates.csv",
            "catalog_preview": "pollen_catalog.jpg",
        },
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(grays)} frames and {len(catalog)} pollen candidates.")
    print(f"Choose a pollen ID in {output_dir / 'pollen_catalog.jpg'}")


if __name__ == "__main__":
    main()
