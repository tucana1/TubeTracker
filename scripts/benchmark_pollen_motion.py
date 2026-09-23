#!/usr/bin/env python3
"""Compare rigid multi-point pollen tracking with the updating-template baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from prototypes.v24_causal_birth_forest.track import (
    load_grayscale_samples,
    movie_schedule,
    stabilize_translations,
)
from prototypes.v25_causal_portal.track import load_portal_owner_seeds
from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds
from tubetracker.pollen_motion import (
    PollenMotionConfig,
    track_pollen_body,
    track_pollen_bodies_in_crop_mosaics,
    track_pollen_bodies_from_seeds,
)


COLORS = {
    "multipoint": (255, 220, 0),
    "template": (0, 128, 255),
    "identity": (0, 220, 80),
}


def parse_args() -> argparse.Namespace:
    """Parse movie, identity, owner, model, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", default="1,5,8,17,18,21,34,39")
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument(
        "--tracking-space",
        choices=("full-field", "fixed-crop", "fixed-crop-mosaic"),
        default="full-field",
    )
    parser.add_argument("--body-crop-size", type=int, default=256)
    parser.add_argument("--mosaic-batch-size", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--review-fps", type=float, default=12.0)
    return parser.parse_args()


def _selected_seeds(identity_path, source_frames, scale, shifts_xy, track_ids):
    """Load requested semantic owner seeds in stable requested order."""

    seeds, identity = load_portal_owner_seeds(
        identity_path,
        source_frames,
        scale,
        shifts_xy,
        include_persistent_late=True,
        minimum_late_semantic_observations=3,
    )
    by_id = {seed.track_id: seed for seed in seeds}
    missing = set(track_ids) - set(by_id)
    if missing:
        raise ValueError(f"requested pollen IDs are unavailable: {sorted(missing)}")
    return [by_id[track_id] for track_id in track_ids], identity


def _source_centers(aligned_yx, shifts_xy, scale):
    """Map aligned analysis centers back to native source coordinates."""

    return (np.asarray(aligned_yx) + shifts_xy[:, ::-1]) / float(scale)


def _interpolate_centers(source_frames, centers_yx, query_frames):
    """Interpolate a sampled trajectory at exact source-frame positions."""

    return np.column_stack(
        [
            np.interp(query_frames, source_frames, centers_yx[:, axis])
            for axis in range(2)
        ]
    )


def _identity_errors(identity, track_id, source_frames, centers_yx):
    """Measure trajectory disagreement at independent identity observations."""

    track = next(
        item for item in identity["tracks"] if int(item["track_id"]) == track_id
    )
    observation_frames = np.asarray(track["source_frames"], dtype=np.float64)
    observed = np.asarray(track["centers_yx"], dtype=np.float64)
    predicted = _interpolate_centers(
        source_frames, centers_yx, observation_frames
    )
    return np.linalg.norm(predicted - observed, axis=1)


def _trajectory_report(
    identity,
    track_ids,
    source_frames,
    multipoint_source_yx,
    template_source_yx,
    trajectories,
):
    """Summarize identity agreement and internal rigid-consensus evidence."""

    owners = {}
    all_multipoint = []
    all_template = []
    for owner, track_id in enumerate(track_ids):
        multipoint_errors = _identity_errors(
            identity, track_id, source_frames, multipoint_source_yx[owner]
        )
        template_errors = _identity_errors(
            identity, track_id, source_frames, template_source_yx[owner]
        )
        all_multipoint.extend(multipoint_errors)
        all_template.extend(template_errors)
        trajectory = trajectories[owner]
        finite_residuals = trajectory.median_residual_px[
            np.isfinite(trajectory.median_residual_px)
        ]
        owners[str(track_id)] = {
            "identity_observations": int(len(multipoint_errors)),
            "multipoint_median_identity_error_px": float(
                np.median(multipoint_errors)
            ),
            "template_median_identity_error_px": float(np.median(template_errors)),
            "multipoint_max_identity_error_px": float(np.max(multipoint_errors)),
            "template_max_identity_error_px": float(np.max(template_errors)),
            "multipoint_observed_fraction": float(np.mean(trajectory.observed)),
            "multipoint_median_inliers": float(
                np.median(trajectory.inlier_point_count)
            ),
            "multipoint_median_rigid_residual_px": (
                float(np.median(finite_residuals))
                if len(finite_residuals)
                else None
            ),
        }
    return {
        "aggregate": {
            "multipoint_median_identity_error_px": float(
                np.median(all_multipoint)
            ),
            "template_median_identity_error_px": float(np.median(all_template)),
            "multipoint_p90_identity_error_px": float(
                np.percentile(all_multipoint, 90)
            ),
            "template_p90_identity_error_px": float(
                np.percentile(all_template, 90)
            ),
        },
        "owners": owners,
    }


def _draw_marker(frame, center_yx, color, label, radius=8):
    """Draw one labeled trajectory marker if its center is finite."""

    center = np.asarray(center_yx, dtype=np.float64)
    if not np.isfinite(center).all():
        return
    point = tuple(np.rint(center[::-1]).astype(int))
    cv.circle(frame, point, radius, color, 2, cv.LINE_AA)
    cv.putText(
        frame,
        label,
        (point[0] + radius + 2, point[1] - radius - 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv.LINE_AA,
    )


def _write_review_video(
    output,
    frames,
    source_frames,
    track_ids,
    multipoint_yx,
    template_yx,
    fps,
):
    """Write a field-wide time-lapse comparing both owner trajectories."""

    height, width = frames.shape[1:]
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width * 2, height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {output}")
    for sample, gray in enumerate(frames):
        canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for owner, track_id in enumerate(track_ids):
            _draw_marker(
                canvas,
                template_yx[owner, sample],
                COLORS["template"],
                f"P{track_id} T",
            )
            _draw_marker(
                canvas,
                multipoint_yx[owner, sample],
                COLORS["multipoint"],
                f"P{track_id} M",
                radius=11,
            )
        cv.putText(
            canvas,
            f"source frame {int(source_frames[sample])}",
            (8, 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        writer.write(cv.resize(canvas, (width * 2, height * 2)))
    writer.release()


def _crop_with_padding(image, center_yx, size):
    """Extract a reflected square crop around one analysis-space center."""

    padding = size
    padded = cv.copyMakeBorder(
        image, padding, padding, padding, padding, cv.BORDER_REFLECT101
    )
    center_y, center_x = np.asarray(center_yx, dtype=np.float64)
    return cv.getRectSubPix(
        padded,
        (size, size),
        (float(center_x + padding), float(center_y + padding)),
    )


def _write_contact_sheet(
    output,
    frames,
    source_frames,
    shifts_xy,
    scale,
    identity,
    track_ids,
    multipoint_yx,
    template_yx,
):
    """Write exact identity-observation crops for rapid visual auditing."""

    crop_size = 112
    columns = 5
    rows = []
    for owner, track_id in enumerate(track_ids):
        track = next(
            item
            for item in identity["tracks"]
            if int(item["track_id"]) == track_id
        )
        observation_frames = np.asarray(track["source_frames"], dtype=int)
        observation_centers = np.asarray(track["centers_yx"], dtype=float)
        chosen = np.linspace(0, len(observation_frames) - 1, columns).round().astype(int)
        cells = []
        for observation_index in chosen:
            source_frame = int(observation_frames[observation_index])
            sample = int(np.argmin(np.abs(source_frames - source_frame)))
            identity_center = observation_centers[observation_index]
            analysis_identity = (
                identity_center * float(scale) - shifts_xy[sample][::-1]
            )
            midpoint = 0.5 * (
                multipoint_yx[owner, sample] + template_yx[owner, sample]
            )
            crop_center = 0.65 * analysis_identity + 0.35 * midpoint
            crop = _crop_with_padding(frames[sample], crop_center, crop_size)
            cell = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
            origin = crop_center - crop_size / 2.0
            _draw_marker(
                cell,
                analysis_identity - origin,
                COLORS["identity"],
                "I",
                radius=6,
            )
            _draw_marker(
                cell,
                template_yx[owner, sample] - origin,
                COLORS["template"],
                "T",
                radius=7,
            )
            _draw_marker(
                cell,
                multipoint_yx[owner, sample] - origin,
                COLORS["multipoint"],
                "M",
                radius=9,
            )
            cv.putText(
                cell,
                f"P{track_id} f{source_frame}",
                (3, crop_size - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    cv.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> int:
    """Run both trackers, compare observations, and save visual evidence."""

    args = parse_args()
    movie = args.movie.expanduser().resolve()
    identity_path = args.identity_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    track_ids = list(
        dict.fromkeys(int(value.strip()) for value in args.track_ids.split(","))
    )
    source_frames, _, _, native_width, native_height = movie_schedule(
        movie, args.sample_interval_s
    )
    scale = args.width / native_width
    frames = load_grayscale_samples(
        movie, source_frames, args.width, native_width, native_height
    )
    aligned, shifts_xy, registration_response = stabilize_translations(frames)
    seeds, identity = _selected_seeds(
        identity_path, source_frames, scale, shifts_xy, track_ids
    )
    seed_centers_yx = np.asarray([seed.center_yx for seed in seeds])
    seed_samples = np.asarray([seed.seed_sample for seed in seeds])
    template_yx, template_scores = track_pollen_centers_from_seeds(
        aligned,
        seed_centers_yx,
        seed_samples,
        template_radius_px=max(5, round(5 * args.width / 480)),
        search_radius_px=max(5, round(5 * args.width / 480)),
        minimum_score=0.25,
    )
    config = PollenMotionConfig.from_environment()
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.tracking_space == "fixed-crop":
        if np.any(seed_samples != 0):
            raise ValueError("fixed-crop benchmark currently requires origin seeds")
        config = replace(config, crop_size=args.body_crop_size)
        trajectories = [
            track_pollen_body(
                aligned,
                seed.center_yx[::-1],
                pollen_radius=15.0 * scale,
                config=config,
            )
            for seed in seeds
        ]
    elif args.tracking_space == "fixed-crop-mosaic":
        config = replace(config, crop_size=args.body_crop_size)
        trajectories = track_pollen_bodies_in_crop_mosaics(
            aligned,
            seed_centers_yx[:, ::-1],
            pollen_radii=np.full(len(seeds), 15.0 * scale),
            seed_samples=seed_samples,
            batch_size=args.mosaic_batch_size,
            config=config,
        )
    else:
        trajectories = track_pollen_bodies_from_seeds(
            aligned,
            seed_centers_yx[:, ::-1],
            pollen_radii=np.full(len(seeds), 15.0 * scale),
            seed_samples=seed_samples,
            config=config,
        )
    multipoint_yx = np.stack(
        [trajectory.centers_xy[:, ::-1] for trajectory in trajectories]
    )
    multipoint_source_yx = np.stack(
        [
            _source_centers(centers, shifts_xy, scale)
            for centers in multipoint_yx
        ]
    )
    template_source_yx = np.stack(
        [
            _source_centers(centers, shifts_xy, scale)
            for centers in template_yx
        ]
    )
    report = {
        "method": f"{args.tracking_space}-cotracker3-rigid-body-consensus",
        "movie": str(movie),
        "identity_report": str(identity_path),
        "track_ids": track_ids,
        "analysis_width": args.width,
        "sample_interval_s": args.sample_interval_s,
        "sample_count": len(source_frames),
        "median_registration_response": float(np.median(registration_response)),
        "median_template_score": float(np.median(template_scores)),
        **_trajectory_report(
            identity,
            track_ids,
            source_frames,
            multipoint_source_yx,
            template_source_yx,
            trajectories,
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "pollen_motion_tracks.npz",
        source_frames=source_frames,
        analysis_width=np.asarray(args.width),
        shifts_xy=shifts_xy,
        track_ids=np.asarray(track_ids),
        multipoint_aligned_yx=multipoint_yx,
        multipoint_source_yx=multipoint_source_yx,
        template_aligned_yx=template_yx,
        template_source_yx=template_source_yx,
        template_scores=template_scores,
        observed=np.stack([trajectory.observed for trajectory in trajectories]),
        inlier_counts=np.stack(
            [trajectory.inlier_point_count for trajectory in trajectories]
        ),
        rigid_residuals=np.stack(
            [trajectory.median_residual_px for trajectory in trajectories]
        ),
    )
    _write_review_video(
        output / "pollen_motion_comparison.mp4",
        aligned,
        source_frames,
        track_ids,
        multipoint_yx,
        template_yx,
        args.review_fps,
    )
    _write_contact_sheet(
        output / "identity_contact_sheet.png",
        aligned,
        source_frames,
        shifts_xy,
        scale,
        identity,
        track_ids,
        multipoint_yx,
        template_yx,
    )
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
