#!/usr/bin/env python3
"""Constrain dense owner motion with jointly assigned pollen-circle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tubetracker.pollen_motion import (
    DetectionSnapConfig,
    canonicalize_owner_trajectory_seeds,
    constrain_trajectories_to_unique_detections,
    detect_field_pollen_circles,
    semantic_anchor_guard_weights,
)


def parse_args() -> argparse.Namespace:
    """Parse motion, field-context, identity, and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--field-cache", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--hough-threshold", type=int, default=8)
    parser.add_argument("--minimum-circle-score", type=float, default=0.2)
    parser.add_argument("--minimum-semantic-anchors-to-lock-seed", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Build a motion cache whose owner centers obey dense one-to-one evidence."""

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    motion_path = args.motion_cache.expanduser().resolve()
    field_dir = args.field_cache.expanduser().resolve()
    identity_path = args.identity_report.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    identity = json.loads(identity_path.read_text())
    field_frames = np.load(field_dir / "aligned_frames.npy", mmap_mode="r")
    field_shifts_xy = np.load(field_dir / "shifts_xy.npy")
    field_source_frames = np.load(field_dir / "source_frames.npy")

    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    source_frames = np.asarray(arrays["source_frames"], dtype=np.int64)
    if not np.array_equal(source_frames, field_source_frames):
        raise ValueError("motion and field caches use different frame schedules")
    analysis_width = int(arrays["analysis_width"])
    field_width = int(field_frames.shape[2])
    scale = field_width / float(analysis_width)
    owner_radius_field = 0.5 * float(identity["diameter_px"]) * scale
    prior_source = np.asarray(
        arrays.get("semantic_span_source_yx", arrays["multipoint_source_yx"]),
        dtype=np.float64,
    )
    prior_field_aligned = (
        prior_source * scale - np.asarray(field_shifts_xy)[:, ::-1][None]
    )
    detections = detect_field_pollen_circles(
        field_frames,
        owner_radius_field,
        batch_size=args.batch_size,
        hough_threshold=args.hough_threshold,
        minimum_circle_score=args.minimum_circle_score,
    )
    identity_by_id = {
        int(track["track_id"]): track for track in identity["tracks"]
    }
    seed_eligible = np.asarray(
        [
            int(identity_by_id[int(track_id)].get("semantic_observation_count", 0))
            < args.minimum_semantic_anchors_to_lock_seed
            for track_id in arrays["track_ids"]
        ],
        dtype=bool,
    )
    canonical = canonicalize_owner_trajectory_seeds(
        prior_field_aligned,
        detections,
        owner_radius_field,
        eligible_owner_mask=seed_eligible,
    )
    prior_field_aligned = canonical.trajectories_yx
    _, assignments, costs, field_correction = (
        constrain_trajectories_to_unique_detections(
            prior_field_aligned,
            detections,
            owner_radius_field,
            DetectionSnapConfig(),
        )
    )
    semantic_weights = semantic_anchor_guard_weights(
        source_frames,
        np.asarray(arrays["track_ids"], dtype=np.int64),
        identity["tracks"],
    )
    field_correction *= semantic_weights[..., None]
    constrained_field = prior_field_aligned + field_correction
    constrained_source = (
        constrained_field + np.asarray(field_shifts_xy)[:, ::-1][None]
    ) / scale
    source_shifts_yx = np.asarray(arrays["shifts_xy"], dtype=np.float64)[:, ::-1]
    arrays.update(
        seed_canonicalization_correction_field_yx=canonical.corrections_yx,
        seed_canonicalization_track_ids=canonical.selected_track_ids,
        seed_canonicalization_scores=canonical.selected_scores,
        seed_canonicalization_support=canonical.selected_support_fractions,
        detection_constrained_source_yx=constrained_source,
        detection_constrained_aligned_yx=constrained_source
        - source_shifts_yx[None],
        detection_assignment_mask=assignments >= 0,
        detection_assignment_cost=costs,
        detection_correction_field_yx=field_correction,
        detection_semantic_guard_weight=semantic_weights,
        detection_counts=np.asarray([len(items) for items in detections]),
        detection_anchor_counts=np.asarray(
            [
                len(identity_by_id[int(track_id)]["source_frames"])
                for track_id in arrays["track_ids"]
            ],
            dtype=np.int64,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cache = output_dir / "pollen_motion_tracks.npz"
    temporary = output_cache.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_cache)

    assigned_fraction = np.mean(assignments >= 0, axis=1)
    report = {
        "revision": "v28.3.2-semantic-guarded-joint-detection-motion",
        "source_motion_cache": str(motion_path),
        "field_cache": str(field_dir),
        "identity_report": str(identity_path),
        "track_count": int(len(arrays["track_ids"])),
        "sample_count": int(len(source_frames)),
        "median_detection_count": float(np.median(arrays["detection_counts"])),
        "median_owner_assignment_fraction": float(np.median(assigned_fraction)),
        "minimum_owner_assignment_fraction": float(np.min(assigned_fraction)),
        "canonicalized_seed_track_ids": [
            int(track_id)
            for track_id, selected in zip(
                arrays["track_ids"], canonical.selected_track_ids
            )
            if selected >= 0
        ],
        "assigned_fraction_by_track_id": {
            str(int(track_id)): float(fraction)
            for track_id, fraction in zip(arrays["track_ids"], assigned_fraction)
        },
        "output_cache": str(output_cache),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
