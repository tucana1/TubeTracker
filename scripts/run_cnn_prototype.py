#!/usr/bin/env python3
"""Run CNN point detection, temporal linking, and calibrated CSV export."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pilot import safe_name, software_state  # noqa: E402
from tubetracker.cnn_prototype import (  # noqa: E402
    CNN_LABELS,
    LinkedPoint,
    extract_heatmap_points,
    link_point_detections,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
    preferred_torch_device,
    resolve_grain_source,
)
from tubetracker.curve_prototype import (  # noqa: E402
    CurveTraceConfig,
    detect_and_track_grains,
)


def parse_args():
    """Parse model, video sampling, calibration, and inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=1000)
    parser.add_argument("--time-per-frame", type=float, required=True)
    parser.add_argument("--time-unit", default="sec")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--distance-unit", default="pxl")
    parser.add_argument("--grain-threshold", type=float, default=0.5)
    parser.add_argument("--tip-threshold", type=float, default=0.5)
    parser.add_argument("--grain-min-distance", type=int, default=10)
    parser.add_argument("--tip-min-distance", type=int, default=7)
    parser.add_argument("--grain-max-step", type=float, default=35.0)
    parser.add_argument("--tip-max-step", type=float, default=45.0)
    parser.add_argument("--max-track-gap", type=int, default=2)
    parser.add_argument("--max-grain-tip-distance", type=float, default=100.0)
    parser.add_argument("--grain-radius-px", type=float, default=10.0)
    parser.add_argument(
        "--grain-source",
        choices=("auto", "cnn", "opencv"),
        default="auto",
        help="Use CNN pollen points or the established circle/flow tracker",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_even_frames(video, sample_count, start_frame, end_frame, image_width):
    """Load evenly spaced, aspect-preserving frames and source metadata."""
    capture = cv.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    original_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv.CAP_PROP_FPS))
    final_frame = frame_count - 1 if end_frame <= 0 else min(end_frame, frame_count - 1)
    if start_frame < 0 or final_frame < start_frame:
        capture.release()
        raise ValueError("Invalid video frame range")
    sample_count = min(max(1, sample_count), final_frame - start_frame + 1)
    source_frames = np.linspace(
        start_frame, final_frame, sample_count, dtype=int
    ).tolist()
    image_height = int(round(original_height * image_width / original_width))
    frames = []
    loaded_source_frames = []
    for source_frame in source_frames:
        capture.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        success, frame = capture.read()
        if not success:
            continue
        frames.append(
            cv.resize(
                frame,
                (image_width, image_height),
                interpolation=cv.INTER_AREA,
            )
        )
        loaded_source_frames.append(source_frame)
    capture.release()
    metadata = {
        "frame_count": frame_count,
        "source_fps": source_fps,
        "original_width": original_width,
        "original_height": original_height,
        "image_width": image_width,
        "image_height": image_height,
        "scale_x": image_width / original_width,
        "scale_y": image_height / original_height,
        "source_frame_indices": loaded_source_frames,
    }
    return frames, metadata


def group_tracks(points):
    """Group linked points by track identifier in time order."""
    grouped = {}
    for point in points:
        grouped.setdefault(point.track_id, []).append(point)
    for values in grouped.values():
        values.sort(key=lambda point: point.analysis_frame)
    return grouped


def opencv_grain_points(frames, source_frames):
    """Convert the established circle/flow grain tracks into point records."""
    tracks, _, _ = detect_and_track_grains(
        frames,
        CurveTraceConfig(preprocessing="background"),
    )
    points = []
    for track_id, track in enumerate(tracks, start=1):
        for roi in track.gv1:
            points.append(
                LinkedPoint(
                    track_id=track_id,
                    analysis_frame=roi.gv6,
                    source_frame=int(source_frames[roi.gv6]),
                    x=float(roi.gv3.x),
                    y=float(roi.gv3.y),
                    confidence=float(getattr(roi, "tracking_confidence", 0.5)),
                )
            )
    return points


def pair_tip_tracks(tip_tracks, grain_points, maximum_distance):
    """Pair each new tip trajectory with its nearest simultaneous pollen track."""
    grains_by_frame = {}
    for point in grain_points:
        grains_by_frame.setdefault(point.analysis_frame, []).append(point)
    pairs = {}
    for tip_track_id, points in tip_tracks.items():
        first = points[0]
        candidates = grains_by_frame.get(first.analysis_frame, [])
        if not candidates:
            pairs[tip_track_id] = None
            continue
        nearest = min(
            candidates,
            key=lambda grain: math.hypot(grain.x - first.x, grain.y - first.y),
        )
        distance = math.hypot(nearest.x - first.x, nearest.y - first.y)
        pairs[tip_track_id] = nearest.track_id if distance <= maximum_distance else None
    return pairs


def closest_track_point(track, analysis_frame, maximum_gap=2):
    """Return the nearest pollen observation around a tip's sampled frame."""
    if not track:
        return None
    point = min(track, key=lambda value: abs(value.analysis_frame - analysis_frame))
    return point if abs(point.analysis_frame - analysis_frame) <= maximum_gap else None


def export_measurements(
    output_dir,
    sample_id,
    grain_points,
    tip_points,
    source_frames,
    metadata,
    args,
):
    """Write all detections plus detailed and summary tip-trajectory tables."""
    scale_x = metadata["scale_x"]
    scale_y = metadata["scale_y"]
    with (output_dir / "point_detections.csv").open("w", newline="") as handle:
        fieldnames = (
            "sample_id",
            "label_type",
            "track_id",
            "analysis_frame",
            "source_frame",
            f"time_{args.time_unit}",
            "analysis_x",
            "analysis_y",
            "source_x",
            "source_y",
            "confidence",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label_type, points in (("grain", grain_points), ("tip", tip_points)):
            for point in sorted(points, key=lambda value: (value.analysis_frame, value.track_id)):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "label_type": label_type,
                        "track_id": point.track_id,
                        "analysis_frame": point.analysis_frame,
                        "source_frame": point.source_frame,
                        f"time_{args.time_unit}": point.source_frame * args.time_per_frame,
                        "analysis_x": point.x,
                        "analysis_y": point.y,
                        "source_x": point.x / scale_x,
                        "source_y": point.y / scale_y,
                        "confidence": point.confidence,
                    }
                )

    grain_tracks = group_tracks(grain_points)
    tip_tracks = group_tracks(tip_points)
    pairs = pair_tip_tracks(tip_tracks, grain_points, args.max_grain_tip_distance)
    detail_rows = []
    summary_rows = []
    original_grain_radius = args.grain_radius_px / ((scale_x + scale_y) / 2.0)
    for tip_track_id, points in tip_tracks.items():
        grain_track_id = pairs[tip_track_id]
        grain_track = grain_tracks.get(grain_track_id, [])
        cumulative_length = 0.0
        initial_boundary_distance = None
        previous_relative = None
        first_time = points[0].source_frame * args.time_per_frame
        last_length = None
        paired_observations = 0
        for point in points:
            grain = closest_track_point(
                grain_track, point.analysis_frame, args.max_track_gap
            )
            row = {
                "sample_id": sample_id,
                "tip_track_id": tip_track_id,
                "grain_track_id": "" if grain_track_id is None else grain_track_id,
                "analysis_frame": point.analysis_frame,
                "source_frame": point.source_frame,
                f"time_{args.time_unit}": point.source_frame * args.time_per_frame,
                "tip_source_x": point.x / scale_x,
                "tip_source_y": point.y / scale_y,
                "grain_source_x": "",
                "grain_source_y": "",
                "tip_confidence": point.confidence,
                "estimated_tube_length_px": "",
                f"estimated_tube_length_{args.distance_unit}": "",
            }
            if grain is not None:
                paired_observations += 1
                tip_source = np.array(
                    [point.x / scale_x, point.y / scale_y], dtype=np.float64
                )
                grain_source = np.array(
                    [grain.x / scale_x, grain.y / scale_y], dtype=np.float64
                )
                relative = tip_source - grain_source
                if initial_boundary_distance is None:
                    initial_boundary_distance = max(
                        0.0, float(np.linalg.norm(relative)) - original_grain_radius
                    )
                elif previous_relative is not None:
                    cumulative_length += float(np.linalg.norm(relative - previous_relative))
                previous_relative = relative
                last_length = initial_boundary_distance + cumulative_length
                row["grain_source_x"] = grain_source[0]
                row["grain_source_y"] = grain_source[1]
                row["estimated_tube_length_px"] = last_length
                row[f"estimated_tube_length_{args.distance_unit}"] = (
                    last_length * args.pixel_size
                )
            detail_rows.append(row)
        final_time = points[-1].source_frame * args.time_per_frame
        duration = final_time - first_time
        summary_rows.append(
            {
                "sample_id": sample_id,
                "tip_track_id": tip_track_id,
                "grain_track_id": "" if grain_track_id is None else grain_track_id,
                "start_source_frame": points[0].source_frame,
                "end_source_frame": points[-1].source_frame,
                "observation_count": len(points),
                "paired_observation_count": paired_observations,
                f"duration_{args.time_unit}": duration,
                "final_estimated_tube_length_px": "" if last_length is None else last_length,
                f"final_estimated_tube_length_{args.distance_unit}": (
                    "" if last_length is None else last_length * args.pixel_size
                ),
                f"average_growth_rate_{args.distance_unit}_per_{args.time_unit}": (
                    ""
                    if last_length is None or duration <= 0
                    else (last_length * args.pixel_size) / duration
                ),
            }
        )
    detail_path = output_dir / "tip_trajectory_details.csv"
    summary_path = output_dir / "tip_trajectory_summary.csv"
    detail_fields = list(detail_rows[0]) if detail_rows else ["sample_id"]
    summary_fields = list(summary_rows[0]) if summary_rows else ["sample_id"]
    for path, rows, fieldnames in (
        (detail_path, detail_rows, detail_fields),
        (summary_path, summary_rows, summary_fields),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return grain_tracks, tip_tracks, pairs, detail_rows, summary_rows


def render_preview(frames, grain_tracks, tip_tracks, pairs, output_path):
    """Render model points and accumulated tip trajectories on source imagery."""
    grains_by_frame = {}
    tips_by_frame = {}
    for track_id, points in grain_tracks.items():
        for point in points:
            grains_by_frame.setdefault(point.analysis_frame, []).append(point)
    for track_id, points in tip_tracks.items():
        for point in points:
            tips_by_frame.setdefault(point.analysis_frame, []).append(point)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview: {output_path}")
    for frame_index, frame in enumerate(frames):
        image = frame.copy()
        for point in grains_by_frame.get(frame_index, []):
            cv.circle(image, (round(point.x), round(point.y)), 7, (30, 210, 80), 2)
        for track_id, points in tip_tracks.items():
            visible = [point for point in points if point.analysis_frame <= frame_index]
            if not visible:
                continue
            coordinates = np.array(
                [[round(point.x), round(point.y)] for point in visible], dtype=np.int32
            )
            if len(coordinates) > 1:
                cv.polylines(image, [coordinates], False, (255, 210, 30), 2, cv.LINE_AA)
            current = visible[-1]
            if current.analysis_frame == frame_index:
                center = (round(current.x), round(current.y))
                cv.drawMarker(image, center, (220, 40, 190), cv.MARKER_CROSS, 12, 2)
                cv.putText(
                    image,
                    f"T{track_id} G{pairs.get(track_id) or '-'}",
                    (center[0] + 8, center[1] - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )
        cv.putText(
            image,
            f"CNN prototype | analysis frame {frame_index + 1}/{len(frames)}",
            (12, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        writer.write(image)
    writer.release()


def main():
    """Run trained point detection and export reviewable temporal measurements."""
    args = parse_args()
    if args.time_per_frame <= 0 or args.pixel_size <= 0:
        raise SystemExit("Time and pixel calibration must be greater than zero")
    video = args.video.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = safe_name(args.sample_id or video.stem)
    device = preferred_torch_device(args.device)
    model, checkpoint_payload = load_cnn_checkpoint(checkpoint, device=device)
    model = model.to(device)
    grain_source = resolve_grain_source(
        args.grain_source, checkpoint_payload["training_metadata"]
    )
    frames, metadata = load_even_frames(
        video,
        args.sample_count,
        args.start_frame,
        args.end_frame,
        args.image_width,
    )
    if not frames:
        raise SystemExit("No frames were loaded")
    detections = {label: [] for label in CNN_LABELS}
    for frame_index, frame in enumerate(frames):
        heatmaps = predict_heatmaps_tiled(
            model,
            frame,
            device,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
        )
        if grain_source == "cnn":
            detections["grain"].append(
                extract_heatmap_points(
                    heatmaps[0], args.grain_threshold, args.grain_min_distance
                )
            )
        else:
            detections["grain"].append([])
        detections["tip"].append(
            extract_heatmap_points(
                heatmaps[1], args.tip_threshold, args.tip_min_distance
            )
        )
        print(f"Predicted {frame_index + 1}/{len(frames)} frames", flush=True)
    source_frames = metadata["source_frame_indices"]
    grain_points = (
        link_point_detections(
            detections["grain"],
            source_frames,
            max_distance=args.grain_max_step,
            max_gap=args.max_track_gap,
            direction_weight=0.1,
        )
        if grain_source == "cnn"
        else opencv_grain_points(frames, source_frames)
    )
    tip_points = link_point_detections(
        detections["tip"],
        source_frames,
        max_distance=args.tip_max_step,
        max_gap=args.max_track_gap,
        direction_weight=0.45,
    )
    grain_tracks, tip_tracks, pairs, detail_rows, summary_rows = export_measurements(
        output_dir,
        sample_id,
        grain_points,
        tip_points,
        source_frames,
        metadata,
        args,
    )
    render_preview(
        frames,
        grain_tracks,
        tip_tracks,
        pairs,
        output_dir / "cnn-point-review.mp4",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prototype": "click_supervised_point_heatmap_v1",
        "input_video": str(video),
        "checkpoint": str(checkpoint),
        "sample_id": sample_id,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "video_metadata": metadata,
        "checkpoint_metadata": checkpoint_payload["training_metadata"],
        "grain_source_used": grain_source,
        "software": {**software_state(), "torch": torch.__version__},
        "counts": {
            "grain_detections": len(grain_points),
            "grain_tracks": len(grain_tracks),
            "tip_detections": len(tip_points),
            "tip_tracks": len(tip_tracks),
            "paired_tip_tracks": sum(value is not None for value in pairs.values()),
            "detail_rows": len(detail_rows),
            "summary_rows": len(summary_rows),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"CNN prototype outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
