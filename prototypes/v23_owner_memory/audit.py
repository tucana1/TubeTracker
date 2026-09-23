"""Audit learned pollen instances independently of legacy Hough anchors.

This is an isolated v23 research entry point.  It reads requested source frames,
runs a configurable Cellpose foundation model, links all resulting instances in
time, and renders raw/overlay pairs with stable track IDs.  It does not alter or
call the production TubeTracker workflow.
"""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path

import cv2 as cv
import numpy as np

from tubetracker.owner_memory import (
    PollenObservation,
    fuse_pollen_observations,
    link_pollen_observations,
    pollen_observations_from_labels,
)


def parse_args() -> argparse.Namespace:
    """Parse the learned-pollen counterexample audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument(
        "--frames",
        help=(
            "Comma-separated source-frame indices. When omitted, sample a short "
            "census window near the beginning of the movie."
        ),
    )
    parser.add_argument("--census-start-s", type=float, default=0.0)
    parser.add_argument("--census-duration-s", type=float, default=8.0)
    parser.add_argument("--census-samples", type=int, default=5)
    parser.add_argument(
        "--census-phase",
        choices=("unconstrained", "pre-growth"),
        default="unconstrained",
        help=(
            "Declare whether the sampled window precedes tube emergence. This "
            "provenance is consumed by later owner-identity checks."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--diameter", type=float, default=30.0)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--legacy-owner", help="Legacy x,y location at --legacy-width")
    parser.add_argument("--legacy-width", type=float, default=480.0)
    parser.add_argument("--include-hough", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def resolve_census_frames(
    movie: Path,
    frame_spec: str | None,
    start_seconds: float,
    duration_seconds: float,
    sample_count: int,
) -> tuple[list[int], float, str]:
    """Resolve an explicit schedule or sample a compact movie-opening census."""

    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv.CAP_PROP_FPS))
    capture.release()
    if frame_count < 1 or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("movie metadata does not provide a valid frame count and fps")
    if frame_spec:
        frames = [int(value.strip()) for value in frame_spec.split(",")]
        strategy = "explicit-source-frames"
    else:
        if (
            not np.isfinite(start_seconds)
            or not np.isfinite(duration_seconds)
            or start_seconds < 0.0
            or duration_seconds <= 0.0
            or sample_count < 2
        ):
            raise ValueError("automatic census timing is invalid")
        first = int(round(start_seconds * fps))
        last = int(round((start_seconds + duration_seconds) * fps))
        first = int(np.clip(first, 0, frame_count - 1))
        last = int(np.clip(last, first, frame_count - 1))
        frames = np.rint(np.linspace(first, last, sample_count)).astype(int).tolist()
        strategy = "uniform-opening-window"
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("census frames must be unique")
    if frames != sorted(frames) or frames[0] < 0 or frames[-1] >= frame_count:
        raise ValueError("census frames must be ordered movie frame indices")
    return frames, fps, strategy


def load_source_frames(movie: Path, indices: list[int]) -> list[np.ndarray]:
    """Load exact source frames without decoding a derived sample video."""
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    frames = []
    for index in indices:
        capture.set(cv.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not read source frame {index}")
        frames.append(frame)
    capture.release()
    return frames


def segment_pollen(
    frames: list[np.ndarray],
    model_name: str,
    diameter: float,
    cellprob_threshold: float,
    gpu: bool,
) -> list[np.ndarray]:
    """Run a named Cellpose model whose weights live in the user model cache."""
    try:
        from cellpose import models
    except ImportError as exc:
        raise RuntimeError("install the optional Cellpose research environment") from exc
    try:
        model = models.CellposeModel(
            gpu=gpu,
            pretrained_model=model_name,
            use_bfloat16=False,
        )
    except NameError as exc:
        if model_name.startswith("cpdino"):
            raise RuntimeError(
                "CPDINO requires the optional DINOv3 package and a supported Python 3.11+ research environment"
            ) from exc
        raise
    rgb = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]
    masks, _, _ = model.eval(
        rgb,
        diameter=diameter,
        min_size=80,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=0.4,
        batch_size=min(4, len(frames)),
        bsize=256,
    )
    return [np.asarray(mask, dtype=np.int32) for mask in masks]


def detect_geometric_pollen(
    frames: list[np.ndarray],
    source_frames: list[int],
    analysis_width: int = 480,
) -> list[tuple[PollenObservation, ...]]:
    """Generate recall-oriented Hough proposals without granting semantic status."""
    from tubetracker.curve_prototype import CurveTraceConfig, detect_grain_candidates

    scale = frames[0].shape[1] / float(analysis_width)
    height = round(frames[0].shape[0] / scale)
    resized = [cv.resize(frame, (analysis_width, height), interpolation=cv.INTER_AREA) for frame in frames]
    config = CurveTraceConfig(
        preprocessing="background",
        blur_radius=6,
        min_grain_radius=3,
        max_grain_radius=9,
        grain_threshold=8,
        min_grain_circle_score=0.2,
    )
    detections = detect_grain_candidates(resized, config)
    observations = []
    for source_frame, frame_detections in zip(source_frames, detections):
        frame_observations = []
        for detection_id, roi in enumerate(frame_detections, start=10_001):
            radius = 0.5 * max(roi.w, roi.h) * scale
            frame_observations.append(
                PollenObservation(
                    frame_index=source_frame,
                    detection_id=detection_id,
                    center_yx=(float(roi.gv3.y * scale), float(roi.gv3.x * scale)),
                    area_px=float(np.pi * radius**2),
                    radius_px=float(radius),
                    circularity=float(getattr(roi, "circle_score", 0.0)),
                    solidity=1.0,
                    pollen_score=float(getattr(roi, "circle_score", 0.0)),
                    semantic_support=0.0,
                    evidence_sources=("hough-circle",),
                )
            )
        observations.append(tuple(frame_observations))
    return observations


def render_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    track_by_detection: dict[int, int],
    fused_observations: tuple[PollenObservation, ...],
    source_frame: int,
    legacy_owner_yx: tuple[float, float] | None,
) -> np.ndarray:
    """Render learned instance boundaries and persistent owner IDs."""
    overlay = frame.copy()
    for detection_id in np.unique(mask):
        if detection_id == 0:
            continue
        region = (mask == detection_id).astype(np.uint8)
        contours, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, contours, -1, (255, 120, 0), 2, cv.LINE_AA)
        ys, xs = np.nonzero(region)
        track_id = track_by_detection.get(int(detection_id), 0)
        cv.putText(
            overlay,
            f"P{track_id}",
            (int(xs.mean()) + 3, int(ys.mean()) - 3),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 220),
            1,
            cv.LINE_AA,
        )
    for observation in fused_observations:
        if observation.semantic_support >= 0.5:
            continue
        y, x = observation.center_yx
        track_id = track_by_detection.get(observation.detection_id, 0)
        cv.circle(overlay, (round(x), round(y)), round(observation.radius_px), (0, 165, 255), 1, cv.LINE_AA)
        cv.putText(overlay, f"G{track_id}", (round(x) + 3, round(y) - 3), cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 120, 220), 1, cv.LINE_AA)
    if legacy_owner_yx is not None:
        y, x = legacy_owner_yx
        cv.drawMarker(overlay, (round(x), round(y)), (0, 0, 255), cv.MARKER_CROSS, 14, 2)
        cv.putText(overlay, "legacy Hough owner", (round(x) + 8, round(y) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(overlay, f"source {source_frame}", (12, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 220), 2, cv.LINE_AA)
    return np.concatenate((frame, overlay), axis=1)


def legacy_owner_metrics(
    mask: np.ndarray,
    owner_yx: tuple[float, float] | None,
    observations: tuple[PollenObservation, ...],
    track_by_detection: dict[int, int],
    semantic_counts: dict[int, int],
) -> dict[str, float | int | bool] | None:
    """Measure whether a learned instance supports the inherited Hough owner."""
    if owner_yx is None:
        return None
    y = int(np.clip(round(owner_yx[0]), 0, mask.shape[0] - 1))
    x = int(np.clip(round(owner_yx[1]), 0, mask.shape[1] - 1))
    foreground = (mask > 0).astype(np.uint8)
    distance = cv.distanceTransform(1 - foreground, cv.DIST_L2, 5)
    nearest = min(
        observations,
        key=lambda item: hypot(item.center_yx[0] - owner_yx[0], item.center_yx[1] - owner_yx[1]),
        default=None,
    )
    result = {
        "source_y_px": float(owner_yx[0]),
        "source_x_px": float(owner_yx[1]),
        "learned_instance_id": int(mask[y, x]),
        "covered_by_learned_instance": bool(mask[y, x] > 0),
        "distance_to_learned_instance_px": float(distance[y, x]),
    }
    if nearest is not None:
        track_id = track_by_detection.get(nearest.detection_id, 0)
        result.update(
            {
                "nearest_fused_center_distance_px": float(hypot(nearest.center_yx[0] - owner_yx[0], nearest.center_yx[1] - owner_yx[1])),
                "nearest_fused_evidence_sources": list(nearest.evidence_sources),
                "nearest_fused_semantic_support": float(nearest.semantic_support),
                "nearest_fused_track_id": int(track_id),
                "nearest_fused_track_semantic_observations": int(semantic_counts.get(track_id, 0)),
            }
        )
    return result


def main() -> None:
    """Run the independent pollen-identity gate and write review artifacts."""
    args = parse_args()
    frame_indices, fps, census_strategy = resolve_census_frames(
        args.movie,
        args.frames,
        args.census_start_s,
        args.census_duration_s,
        args.census_samples,
    )
    frames = load_source_frames(args.movie, frame_indices)
    masks = segment_pollen(
        frames,
        args.model,
        args.diameter,
        args.cellprob_threshold,
        not args.cpu,
    )
    learned_by_frame = [
        pollen_observations_from_labels(
            mask,
            frame_index,
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
        )
        for frame, mask, frame_index in zip(frames, masks, frame_indices)
    ]
    geometric_by_frame = (
        detect_geometric_pollen(frames, frame_indices)
        if args.include_hough
        else [() for _ in frames]
    )
    observations_by_frame = [
        fuse_pollen_observations(learned, geometric)
        for learned, geometric in zip(learned_by_frame, geometric_by_frame)
    ]
    tracks = link_pollen_observations(observations_by_frame)
    owner_yx = None
    if args.legacy_owner:
        x, y = (float(value) for value in args.legacy_owner.split(","))
        scale = frames[0].shape[1] / args.legacy_width
        owner_yx = (y * scale, x * scale)

    lookup = {
        (observation.frame_index, observation.detection_id): track.track_id
        for track in tracks
        for observation in track.observations
    }
    semantic_counts = {track.track_id: track.semantic_observation_count for track in tracks}
    args.output.mkdir(parents=True, exist_ok=True)
    overlays = []
    for frame, mask, frame_index in zip(frames, masks, frame_indices):
        frame_position = frame_indices.index(frame_index)
        track_by_detection = {
            observation.detection_id: lookup[(frame_index, observation.detection_id)]
            for observation in observations_by_frame[frame_position]
        }
        overlays.append(
            render_overlay(
                frame,
                mask,
                track_by_detection,
                observations_by_frame[frame_position],
                frame_index,
                owner_yx,
            )
        )
    review_width = 1280
    review_frames = [cv.resize(image, (review_width, round(image.shape[0] * review_width / image.shape[1]))) for image in overlays]
    review_sheet = np.concatenate(review_frames, axis=0)
    cv.imwrite(str(args.output / "pollen_identity_review.jpg"), review_sheet, [cv.IMWRITE_JPEG_QUALITY, 92])

    report = {
        "prototype": "v23_owner_conditioned_video_graph",
        "stage": "independent-learned-pollen-gate",
        "movie": str(args.movie),
        "model": args.model,
        "diameter_px": args.diameter,
        "source_fps": fps,
        "source_frames": frame_indices,
        "census": {
            "phase": args.census_phase,
            "strategy": census_strategy,
            "first_source_frame": int(frame_indices[0]),
            "last_source_frame": int(frame_indices[-1]),
            "first_time_seconds": float(frame_indices[0] / fps),
            "last_time_seconds": float(frame_indices[-1] / fps),
            "sample_count": len(frame_indices),
        },
        "learned_detection_counts": [len(frame) for frame in learned_by_frame],
        "geometric_proposal_counts": [len(frame) for frame in geometric_by_frame],
        "fused_observation_counts": [len(frame) for frame in observations_by_frame],
        "track_count": len(tracks),
        "multi_frame_track_count": sum(len(track.observations) > 1 for track in tracks),
        "semantically_supported_track_count": sum(track.semantic_observation_count >= 2 for track in tracks),
        "tracks": [
            {
                "track_id": track.track_id,
                "observation_count": len(track.observations),
                "semantic_observation_count": track.semantic_observation_count,
                "evidence_sources": list(track.evidence_sources),
                "source_frames": [item.frame_index for item in track.observations],
                "centers_yx": [list(item.center_yx) for item in track.observations],
                "areas_px": [item.area_px for item in track.observations],
                "radii_px": [item.radius_px for item in track.observations],
                "circularities": [item.circularity for item in track.observations],
                "solidities": [item.solidity for item in track.observations],
                "pollen_scores": [item.pollen_score for item in track.observations],
                "semantic_support": [
                    item.semantic_support for item in track.observations
                ],
                "observation_evidence_sources": [
                    list(item.evidence_sources) for item in track.observations
                ],
            }
            for track in tracks
        ],
        "legacy_owner": legacy_owner_metrics(
            masks[len(masks) // 2],
            owner_yx,
            observations_by_frame[len(observations_by_frame) // 2],
            {
                observation.detection_id: lookup[(frame_indices[len(frame_indices) // 2], observation.detection_id)]
                for observation in observations_by_frame[len(observations_by_frame) // 2]
            },
            semantic_counts,
        ),
        "artifacts": {"review_sheet": "pollen_identity_review.jpg"},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
