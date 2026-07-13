#!/usr/bin/env python3
"""Inspect a TubeTracker input video without inferring biological timing."""

import argparse
import json
from pathlib import Path

import cv2 as cv


def parse_args():
    """Parse the input video and report-output command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record video metadata and extract representative frames."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def video_fourcc(cap):
    """Decode an OpenCV capture's numeric codec identifier into text."""
    value = int(cap.get(cv.CAP_PROP_FOURCC))
    return "".join(chr((value >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")


def main():
    """Inspect a video and write metadata plus representative frame previews."""
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")

    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open: {video}")

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    playback_fps = float(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "video": str(video),
        "file_size_bytes": video.stat().st_size,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "playback_fps": playback_fps,
        "playback_duration_seconds": (
            frame_count / playback_fps if playback_fps > 0 else None
        ),
        "codec_fourcc": video_fourcc(cap),
        "biological_time_per_frame": None,
        "pixel_size": None,
        "note": "Playback FPS is not assumed to equal the biological frame interval.",
    }

    indices = sorted(
        set(
            max(0, min(frame_count - 1, int((frame_count - 1) * fraction)))
            for fraction in (0, 0.25, 0.5, 0.75, 1)
        )
    )
    previews = []
    for index in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        preview_path = args.output_dir / f"frame_{index:06d}.png"
        cv.imwrite(str(preview_path), frame)
        scale = min(1.0, 260 / frame.shape[0])
        preview = cv.resize(frame, None, fx=scale, fy=scale)
        cv.putText(
            preview,
            f"frame {index}",
            (12, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        previews.append(preview)

    if previews:
        target_height = min(image.shape[0] for image in previews)
        normalized = [
            cv.resize(image, (int(image.shape[1] * target_height / image.shape[0]), target_height))
            for image in previews
        ]
        cv.imwrite(str(args.output_dir / "contact_sheet.jpg"), cv.hconcat(normalized))

    metadata_path = args.output_dir / "video_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote intake files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
