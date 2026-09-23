#!/usr/bin/env python3
"""Anchor a dense pollen-motion cache to all sparse semantic observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tubetracker.pollen_motion import (
    anchor_trajectories_to_identity_observations,
    fuse_trajectories_with_identity_span,
)


def parse_args() -> argparse.Namespace:
    """Parse source cache, identity report, and output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_cache", type=Path)
    parser.add_argument("identity_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def interpolated_anchor_errors(
    source_frames: np.ndarray,
    track_ids: np.ndarray,
    tracks_yx: np.ndarray,
    identity_tracks: list[dict],
) -> np.ndarray:
    """Measure trajectory error at every sparse identity observation."""

    identity_by_id = {int(track["track_id"]): track for track in identity_tracks}
    errors = []
    for owner, track_id in enumerate(track_ids):
        identity = identity_by_id[int(track_id)]
        anchor_frames = np.asarray(identity["source_frames"], dtype=np.float64)
        anchor_centers = np.asarray(identity["centers_yx"], dtype=np.float64)
        predicted = np.column_stack(
            [
                np.interp(anchor_frames, source_frames, tracks_yx[owner, :, axis])
                for axis in range(2)
            ]
        )
        errors.extend(np.linalg.norm(predicted - anchor_centers, axis=1))
    return np.asarray(errors, dtype=np.float64)


def main() -> None:
    """Write an immutable anchored cache and quantitative anchor-error report."""

    args = parse_args()
    source_cache = args.motion_cache.expanduser().resolve()
    identity_path = args.identity_report.expanduser().resolve()
    identity = json.loads(identity_path.read_text())
    with np.load(source_cache, allow_pickle=False) as cache:
        arrays = {key: np.asarray(cache[key]) for key in cache.files}
    required = {
        "source_frames",
        "track_ids",
        "analysis_width",
        "shifts_xy",
        "multipoint_source_yx",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"motion cache is missing: {sorted(missing)}")

    source_frames = np.asarray(arrays["source_frames"], dtype=np.float64)
    track_ids = np.asarray(arrays["track_ids"], dtype=np.int64)
    original = np.asarray(arrays["multipoint_source_yx"], dtype=np.float64)
    anchored, corrections, anchor_counts = anchor_trajectories_to_identity_observations(
        source_frames,
        track_ids,
        original,
        identity["tracks"],
    )
    anchored_template, template_corrections, _ = (
        anchor_trajectories_to_identity_observations(
            source_frames,
            track_ids,
            np.asarray(arrays["template_source_yx"], dtype=np.float64),
            identity["tracks"],
        )
    )
    semantic_span, template_used = fuse_trajectories_with_identity_span(
        source_frames,
        track_ids,
        anchored,
        anchored_template,
        identity["tracks"],
    )
    shifts_yx = np.asarray(arrays["shifts_xy"], dtype=np.float64)[:, ::-1]
    arrays.update(
        semantic_anchored_source_yx=anchored,
        semantic_anchored_aligned_yx=anchored - shifts_yx[None, :, :],
        semantic_anchor_correction_yx=corrections,
        semantic_anchored_template_source_yx=anchored_template,
        semantic_anchored_template_aligned_yx=(
            anchored_template - shifts_yx[None, :, :]
        ),
        semantic_template_correction_yx=template_corrections,
        semantic_span_source_yx=semantic_span,
        semantic_span_aligned_yx=semantic_span - shifts_yx[None, :, :],
        semantic_span_template_used=template_used,
        semantic_anchor_counts=anchor_counts,
    )
    before = interpolated_anchor_errors(
        source_frames, track_ids, original, identity["tracks"]
    )
    after = interpolated_anchor_errors(
        source_frames, track_ids, anchored, identity["tracks"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    output_cache = args.output / "pollen_motion_tracks.npz"
    temporary_cache = output_cache.with_suffix(".npz.part")
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_cache.replace(output_cache)
    report = {
        "revision": "v28.2-semantic-span-fusion",
        "source_motion_cache": str(source_cache),
        "identity_report": str(identity_path),
        "track_count": len(track_ids),
        "anchor_count": int(np.sum(anchor_counts)),
        "median_anchor_error_before_px": float(np.median(before)),
        "p90_anchor_error_before_px": float(np.percentile(before, 90)),
        "median_anchor_error_after_px": float(np.median(after)),
        "p90_anchor_error_after_px": float(np.percentile(after, 90)),
        "template_sample_fraction": float(np.mean(template_used)),
        "output_cache": str(output_cache),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
