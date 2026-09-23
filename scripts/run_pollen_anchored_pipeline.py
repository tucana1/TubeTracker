#!/usr/bin/env python3
"""Track one moving pollen and its complete attached tube through a video."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from tubetracker.pollen_anchored_chain import (
    PollenAnchoredChainConfig,
    build_pollen_exclusions,
    centerline_to_source_xy,
    prepare_pollen_anchored_evidence,
    stabilize_pollen_frames,
    trace_pollen_anchored_chain,
)
from tubetracker.material_curve_tracking import (
    MaterialCurveConfig,
    build_material_queries,
    trace_material_curve,
)
from tubetracker.material_ribbon import MaterialRibbonConfig
from tubetracker.pollen_motion import (
    PollenMotionConfig,
    install_default_cotracker_checkpoint,
    pollen_trajectory_from_tracks,
    track_query_points_bidirectional,
    track_pollen_body,
)
from tubetracker.topology_aware_tracing import (
    TopologyAwareTraceConfig,
    trace_topology_aware_chain,
)
from tubetracker.sampling import plan_tracking_and_output_indices


def parse_args():
    """Parse cached-video, pollen identity, calibration, and model controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gray-cache",
        type=Path,
        default=Path("runs/cache/P0034/gray_samples.npy"),
    )
    parser.add_argument(
        "--source-frames-cache",
        type=Path,
        default=Path("runs/cache/P0034/source_frames.npy"),
    )
    parser.add_argument(
        "--grain-cache",
        type=Path,
        default=Path("runs/cache/P0034/grain_tracks.npz"),
    )
    parser.add_argument(
        "--candidate-catalog",
        type=Path,
        default=Path("runs/cache/P0034/pollen_candidates.csv"),
    )
    parser.add_argument("--pollen-id", default="P0034")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--output-sample-count", type=int, default=40)
    parser.add_argument("--time-per-source-frame", type=float, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--time-unit", default="sec")
    parser.add_argument("--distance-unit", default="um")
    parser.add_argument("--chain-crop-size", type=int, default=320)
    parser.add_argument("--body-crop-size", type=int, default=256)
    parser.add_argument("--baseline-observations", type=int, default=30)
    parser.add_argument("--body-track-cache", type=Path)
    parser.add_argument("--material-track-cache", type=Path)
    parser.add_argument("--point-checkpoint", type=Path)
    parser.add_argument("--point-device", default=None)
    parser.add_argument("--install-model", action="store_true")
    parser.add_argument("--review-fps", type=float, default=12.0)
    parser.add_argument(
        "--trace-mode",
        choices=("greedy", "topology", "material"),
        default="greedy",
        help="Choose greedy, topology-only, or ordered material-curve tracing.",
    )
    return parser.parse_args()


def _require_positive(value, name):
    """Reject omitted or physically invalid calibration values."""
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _load_inputs(args):
    """Load one stable pollen identity and nested dense/reporting schedules."""
    grays = np.load(args.gray_cache.expanduser().resolve(), mmap_mode="r")
    source_frames = np.load(args.source_frames_cache.expanduser().resolve())
    if grays.ndim != 3 or len(grays) != len(source_frames):
        raise ValueError("gray and source-frame caches must have equal frame counts")
    chosen, output_positions = plan_tracking_and_output_indices(
        len(grays), args.sample_count, args.output_sample_count
    )
    catalog = pd.read_csv(args.candidate_catalog.expanduser().resolve())
    selected = catalog[catalog["pollen_id"].astype(str) == str(args.pollen_id)]
    if len(selected) != 1:
        raise ValueError(f"expected one catalog row for {args.pollen_id}")
    candidate = selected.iloc[0]
    grain_cache = np.load(args.grain_cache.expanduser().resolve())
    positions = np.asarray(grain_cache["positions"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] != len(grays):
        raise ValueError("grain cache must cover every cached frame")
    initial_xy = np.asarray([candidate.x, candidate.y], dtype=np.float64)
    distances = np.linalg.norm(positions[:, 0, :2] - initial_xy, axis=1)
    target_index = int(np.nanargmin(distances))
    if distances[target_index] > max(3.0, float(candidate.radius) * 0.5):
        raise ValueError("pollen catalog does not match the grain trajectory cache")
    return {
        "grays": np.asarray(grays[chosen]),
        "source_frames": np.asarray(source_frames[chosen], dtype=int),
        "chosen": chosen,
        "output_positions": output_positions,
        "candidate": candidate,
        "pollen_positions": positions[:, chosen],
        "target_index": target_index,
    }


def _motion_config(args):
    """Build a reproducible point-tracking configuration from CLI overrides."""
    config = PollenMotionConfig.from_environment()
    changes = {"crop_size": int(args.body_crop_size)}
    if args.point_checkpoint is not None:
        changes["checkpoint"] = args.point_checkpoint.expanduser().resolve()
    if args.point_device is not None:
        changes["device"] = args.point_device
    return replace(config, **changes)


def _load_or_track_motion(args, inputs, config):
    """Reuse exact body landmarks or run CoTracker and preserve every track."""
    cache_path = (
        args.body_track_cache.expanduser().resolve()
        if args.body_track_cache is not None
        else args.output_dir / "pollen_body_tracks.npz"
    )
    if cache_path.is_file():
        cache = np.load(cache_path)
        if "chosen_cache_indices" in cache and not np.array_equal(
            cache["chosen_cache_indices"], inputs["chosen"]
        ):
            raise ValueError("body-track cache uses a different frame schedule")
        trajectory = pollen_trajectory_from_tracks(
            cache["tracks"],
            cache["visibility"],
            cache["initial_points"],
            [inputs["candidate"].x, inputs["candidate"].y],
            config=config,
        )
        return trajectory, cache_path, True
    print("Tracking redundant landmarks across the pollen body...")
    trajectory = track_pollen_body(
        inputs["grays"],
        [inputs["candidate"].x, inputs["candidate"].y],
        float(inputs["candidate"].radius),
        config=config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        tracks=trajectory.tracks_xy,
        visibility=trajectory.visibility,
        initial_points=trajectory.initial_points_xy,
        centers=trajectory.centers_xy,
        angles_radians=trajectory.angles_radians,
        observed=trajectory.observed,
        chosen_cache_indices=inputs["chosen"],
    )
    return trajectory, cache_path, False


def _load_or_track_material(args, stabilized, queries, config):
    """Reuse or compute arbitrary-time tracks for material-curve keyframes."""
    cache_path = (
        args.material_track_cache.expanduser().resolve()
        if args.material_track_cache is not None
        else args.output_dir / "material_point_tracks.npz"
    )
    if cache_path.is_file():
        cache = np.load(cache_path)
        if not np.array_equal(cache["query_points_txy"], queries.points_txy):
            raise ValueError("material-track cache uses different point queries")
        if "bidirectional" not in cache or not bool(cache["bidirectional"]):
            raise ValueError("material-track cache lacks backward identity tracks")
        return cache["tracks"], cache["visibility"], cache_path, True
    if not len(queries.points_txy):
        empty_tracks = np.empty((len(stabilized), 0, 2), dtype=np.float32)
        empty_visibility = np.empty((len(stabilized), 0), dtype=bool)
        return empty_tracks, empty_visibility, cache_path, False
    print("Tracking repeated material landmarks along the complete tube...")
    tracks, visibility = track_query_points_bidirectional(
        stabilized, queries.points_txy, config=config
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        tracks=tracks,
        visibility=visibility,
        query_points_txy=queries.points_txy,
        arc_positions_px=queries.arc_positions_px,
        group_indices=queries.group_indices,
        keyframes=queries.keyframes,
        bidirectional=np.asarray(True),
    )
    return tracks, visibility, cache_path, False


def _measurement_tables(
    args,
    inputs,
    trajectory,
    measurements,
    transforms,
    material_diagnostics=None,
):
    """Create dense measurements and complete accepted/rejected point tables."""
    detail_rows = []
    point_rows = []
    candidate_rows = []
    time_column = f"time_{args.time_unit}"
    length_column = f"tube_length_{args.distance_unit}"
    for frame, measurement in enumerate(measurements):
        source_frame = int(inputs["source_frames"][frame])
        chain = centerline_to_source_xy(
            measurement.centerline_yx, transforms[frame]
        )
        candidate = centerline_to_source_xy(
            measurement.candidate_centerline_yx, transforms[frame]
        )
        tip = chain[-1] if len(chain) else np.asarray([np.nan, np.nan])
        center = trajectory.centers_xy[frame]
        material_diagnostic = (
            material_diagnostics[frame]
            if material_diagnostics is not None
            else None
        )
        has_ribbon = bool(
            material_diagnostic is not None
            and len(material_diagnostic.widths_px) == len(chain)
            and len(chain)
        )
        if has_ribbon:
            left_boundary = centerline_to_source_xy(
                material_diagnostic.left_boundary_yx, transforms[frame]
            )
            right_boundary = centerline_to_source_xy(
                material_diagnostic.right_boundary_yx, transforms[frame]
            )
            median_width_px = float(np.median(material_diagnostic.widths_px))
            ribbon_supported_fraction = float(
                np.mean(material_diagnostic.ribbon_confidence > 0.0)
            )
        else:
            left_boundary = np.empty((0, 2), dtype=np.float64)
            right_boundary = np.empty((0, 2), dtype=np.float64)
            median_width_px = np.nan
            ribbon_supported_fraction = np.nan
        detail_rows.append(
            {
                "pollen_id": args.pollen_id,
                "analysis_frame": frame,
                "cache_frame": int(inputs["chosen"][frame]),
                "source_frame": source_frame,
                time_column: source_frame * float(args.time_per_source_frame),
                "status": measurement.status,
                "usable": bool(measurement.usable),
                "pollen_x_px": center[0],
                "pollen_y_px": center[1],
                "tip_x_px": tip[0],
                "tip_y_px": tip[1],
                "tube_length_px": measurement.length_px,
                length_column: measurement.length_px * float(args.pixel_size),
                "root_tip_distance_px": (
                    material_diagnostic.root_tip_distance_px
                    if has_ribbon
                    else np.nan
                ),
                f"root_tip_distance_{args.distance_unit}": (
                    material_diagnostic.root_tip_distance_px
                    * float(args.pixel_size)
                    if has_ribbon
                    else np.nan
                ),
                "tube_tortuosity": (
                    material_diagnostic.tortuosity if has_ribbon else np.nan
                ),
                "median_tube_width_px": median_width_px,
                f"median_tube_width_{args.distance_unit}": (
                    median_width_px * float(args.pixel_size)
                ),
                "body_pose_observed": bool(trajectory.observed[frame]),
                "body_visible_points": int(trajectory.visible_point_count[frame]),
                "body_inlier_points": int(trajectory.inlier_point_count[frame]),
                "body_median_residual_px": trajectory.median_residual_px[frame],
                "chain_structure_support": measurement.structure_support,
                "chain_supported_fraction": measurement.chain_supported_fraction,
                "root_attachment_error_px": measurement.root_attachment_error_px,
                "maximum_centerline_step_px": measurement.maximum_centerline_step_px,
                "neighbor_contact": measurement.neighbor_contact,
                "seed_confirmations": measurement.seed_confirmations,
                "material_visible_queries": (
                    material_diagnostic.visible_query_count
                    if material_diagnostic is not None
                    else np.nan
                ),
                "material_accepted_keyframes": (
                    material_diagnostic.accepted_group_count
                    if material_diagnostic is not None
                    else np.nan
                ),
                "material_rejected_keyframes": (
                    material_diagnostic.rejected_group_count
                    if material_diagnostic is not None
                    else np.nan
                ),
                "material_supported_fraction": (
                    float(np.mean(material_diagnostic.observed))
                    if material_diagnostic is not None
                    and len(material_diagnostic.observed)
                    else np.nan
                ),
                "material_boundary_supported_fraction": ribbon_supported_fraction,
                "material_proposed_growth_px": (
                    material_diagnostic.proposed_growth_px
                    if has_ribbon
                    else np.nan
                ),
                "material_accepted_growth_px": (
                    material_diagnostic.accepted_growth_px
                    if has_ribbon
                    else np.nan
                ),
                "material_length_change_px": (
                    material_diagnostic.length_change_px
                    if has_ribbon
                    else np.nan
                ),
                "material_length_consistent": (
                    material_diagnostic.length_consistent
                    if has_ribbon
                    else np.nan
                ),
            }
        )
        for point_index, point in enumerate(chain):
            diagnostic = material_diagnostic
            confidence = (
                diagnostic.confidence[point_index]
                if diagnostic is not None
                and point_index < len(diagnostic.confidence)
                else np.nan
            )
            left = (
                left_boundary[point_index]
                if point_index < len(left_boundary)
                else np.asarray([np.nan, np.nan])
            )
            right = (
                right_boundary[point_index]
                if point_index < len(right_boundary)
                else np.asarray([np.nan, np.nan])
            )
            width_px = (
                material_diagnostic.widths_px[point_index]
                if has_ribbon
                else np.nan
            )
            ribbon_confidence = (
                material_diagnostic.ribbon_confidence[point_index]
                if has_ribbon
                else np.nan
            )
            point_rows.append(
                {
                    "pollen_id": args.pollen_id,
                    "analysis_frame": frame,
                    "source_frame": source_frame,
                    "point_index": point_index,
                    "material_node_id": point_index,
                    "x_px": point[0],
                    "y_px": point[1],
                    "material_track_confidence": confidence,
                    "material_observed": bool(confidence > 0.0),
                    "left_boundary_x_px": left[0],
                    "left_boundary_y_px": left[1],
                    "right_boundary_x_px": right[0],
                    "right_boundary_y_px": right[1],
                    "tube_width_px": width_px,
                    f"tube_width_{args.distance_unit}": (
                        width_px * float(args.pixel_size)
                    ),
                    "ribbon_confidence": ribbon_confidence,
                }
            )
        for point_index, point in enumerate(candidate):
            candidate_rows.append(
                {
                    "pollen_id": args.pollen_id,
                    "analysis_frame": frame,
                    "source_frame": source_frame,
                    "point_index": point_index,
                    "x_px": point[0],
                    "y_px": point[1],
                    "status": measurement.status,
                }
            )
    return (
        pd.DataFrame.from_records(detail_rows),
        pd.DataFrame.from_records(point_rows),
        pd.DataFrame.from_records(candidate_rows),
    )


def _summary(args, details):
    """Summarize only chain-attached usable observations without hiding reviews."""
    usable = details[details["usable"]]
    status_set = set(details["status"])
    if "neighbor_contact_censored" in status_set and len(usable) >= 3:
        overall = "measured_until_neighbor_contact"
    elif len(usable) >= 3:
        overall = "measured"
    elif status_set <= {
        "pre_germination",
        "birth_candidate",
        "initial_chain_review",
    }:
        overall = (
            "not_germinated_with_rejected_candidates"
            if "initial_chain_review" in status_set
            else "not_germinated"
        )
    elif len(usable):
        overall = "partial_review"
    else:
        overall = "review"
    time_column = f"time_{args.time_unit}"
    length_column = f"tube_length_{args.distance_unit}"
    growth_rate = np.nan
    if len(usable) >= 3 and usable[time_column].nunique() >= 2:
        growth_rate = float(
            np.polyfit(usable[time_column], usable[length_column], 1)[0]
        )
    first = usable.iloc[0] if len(usable) else None
    last = usable.iloc[-1] if len(usable) else None
    expected_nonmeasurement = details["status"].isin(
        {"pre_germination", "birth_candidate"}
    )
    quality_review = ~details["usable"] & ~expected_nonmeasurement
    return pd.DataFrame.from_records(
        [
            {
                "pollen_id": args.pollen_id,
                "overall_status": overall,
                "sampled_frames": len(details),
                "usable_frames": len(usable),
                "quality_review_frames": int(np.count_nonzero(quality_review)),
                "germination_source_frame": (
                    first["source_frame"] if first is not None else np.nan
                ),
                f"germination_time_{args.time_unit}": (
                    first[time_column] if first is not None else np.nan
                ),
                f"last_usable_length_{args.distance_unit}": (
                    last[length_column] if last is not None else np.nan
                ),
                f"maximum_usable_length_{args.distance_unit}": (
                    usable[length_column].max() if len(usable) else np.nan
                ),
                f"growth_rate_{args.distance_unit}_per_{args.time_unit}": growth_rate,
            }
        ]
    )


def _polyline(image, points_xy, color, thickness=2):
    """Draw one Cartesian centerline when it has enough finite points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        return
    cv.polylines(
        image,
        [np.rint(points).astype(np.int32)],
        False,
        color,
        thickness,
        cv.LINE_AA,
    )


def _material_node_color(point_index):
    """Assign one permanent review color to each ordered material node."""
    hsv = np.uint8([[[int((point_index * 23) % 180), 210, 245]]])
    return tuple(int(value) for value in cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0, 0])


def _draw_material_nodes(image, points_xy, diagnostic):
    """Draw stable node identities with smaller marks for inferred positions."""
    if diagnostic is None:
        return
    for point_index, point in enumerate(np.asarray(points_xy, dtype=np.float64)):
        if not np.isfinite(point).all():
            continue
        observed = (
            point_index < len(diagnostic.observed)
            and diagnostic.observed[point_index]
        )
        radius = 3 if observed else 2
        color = _material_node_color(point_index) if observed else (150, 150, 150)
        cv.circle(
            image,
            tuple(np.rint(point).astype(int)),
            radius,
            color,
            -1,
            cv.LINE_AA,
        )


def _label(image, lines):
    """Render compact high-contrast scientific review labels."""
    for index, line in enumerate(lines):
        position = (8, 20 + 19 * index)
        cv.putText(image, line, position, cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image, line, position, cv.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv.LINE_AA)


def _review_tile(
    gray,
    evidence,
    measurement,
    transform,
    trajectory,
    frame,
    radius,
    material_diagnostic=None,
):
    """Show source geometry beside the pollen-stabilized evidence used to measure it."""
    chain = centerline_to_source_xy(measurement.centerline_yx, transform)
    candidate = centerline_to_source_xy(measurement.candidate_centerline_yx, transform)
    has_ribbon = bool(
        material_diagnostic is not None
        and len(material_diagnostic.widths_px) == len(chain)
        and len(chain)
    )
    if has_ribbon:
        left_boundary = centerline_to_source_xy(
            material_diagnostic.left_boundary_yx, transform
        )
        right_boundary = centerline_to_source_xy(
            material_diagnostic.right_boundary_yx, transform
        )
    else:
        left_boundary = np.empty((0, 2), dtype=np.float64)
        right_boundary = np.empty((0, 2), dtype=np.float64)
    source = cv.cvtColor(np.asarray(gray), cv.COLOR_GRAY2BGR)
    center = trajectory.centers_xy[frame]
    color = (70, 220, 70) if measurement.usable else (0, 165, 255)
    cv.circle(source, tuple(np.rint(center).astype(int)), int(round(radius)), (255, 160, 30), 2, cv.LINE_AA)
    _polyline(source, chain, color, 3)
    _polyline(source, candidate, (180, 70, 255), 2)
    _polyline(source, left_boundary, (255, 120, 40), 2)
    _polyline(source, right_boundary, (40, 120, 255), 2)
    _draw_material_nodes(source, chain, material_diagnostic)
    if len(chain):
        cv.circle(source, tuple(np.rint(chain[-1]).astype(int)), 4, color, -1, cv.LINE_AA)
    local_tracks = trajectory.tracks_xy[frame] + trajectory.crop_origin_xy
    for point in local_tracks[trajectory.visibility[frame]]:
        cv.circle(source, tuple(np.rint(point).astype(int)), 1, (255, 255, 0), -1)
    half = 160
    padded = cv.copyMakeBorder(source, half, half, half, half, cv.BORDER_REFLECT101)
    source_crop = cv.getRectSubPix(
        padded,
        (2 * half, 2 * half),
        (float(center[0] + half), float(center[1] + half)),
    )
    stabilized = cv.cvtColor(evidence.gray, cv.COLOR_GRAY2BGR)
    stabilized_chain = measurement.centerline_yx[:, ::-1]
    stabilized_candidate = measurement.candidate_centerline_yx[:, ::-1]
    stabilized_left = (
        material_diagnostic.left_boundary_yx[:, ::-1]
        if has_ribbon
        else np.empty((0, 2), dtype=np.float64)
    )
    stabilized_right = (
        material_diagnostic.right_boundary_yx[:, ::-1]
        if has_ribbon
        else np.empty((0, 2), dtype=np.float64)
    )
    _polyline(stabilized, stabilized_chain, color, 3)
    _polyline(stabilized, stabilized_candidate, (180, 70, 255), 2)
    _polyline(stabilized, stabilized_left, (255, 120, 40), 2)
    _polyline(stabilized, stabilized_right, (40, 120, 255), 2)
    _draw_material_nodes(stabilized, stabilized_chain, material_diagnostic)
    crop_center = stabilized.shape[0] // 2
    cv.circle(stabilized, (crop_center, crop_center), int(round(radius)), (255, 160, 30), 2, cv.LINE_AA)
    source_labels = [
        f"frame {frame}  {measurement.status}",
        f"arc length {measurement.length_px:.1f}px",
    ]
    if has_ribbon:
        source_labels.append(
            f"chord {material_diagnostic.root_tip_distance_px:.1f}px  "
            f"bend {material_diagnostic.tortuosity:.2f}x"
        )
        if material_diagnostic.accepted_growth_px > 1e-6:
            source_labels.append(
                f"accepted tip growth +{material_diagnostic.accepted_growth_px:.1f}px"
            )
    _label(source_crop, source_labels)
    stabilized_labels = ["pollen-stabilized", "green only when chain passes"]
    if material_diagnostic is not None:
        stabilized_labels.append(
            f"{material_diagnostic.visible_query_count} tracks  "
            f"{material_diagnostic.accepted_group_count} keyframes"
        )
    if has_ribbon:
        stabilized_labels.append(
            f"paired walls {np.mean(material_diagnostic.ribbon_confidence > 0.0):.0%}  "
            f"width {np.median(material_diagnostic.widths_px):.1f}px"
        )
    _label(stabilized, stabilized_labels)
    return np.concatenate((source_crop, stabilized), axis=1)


def _write_review(
    path,
    args,
    inputs,
    trajectory,
    evidence,
    measurements,
    transforms,
    material_diagnostics=None,
):
    """Write a dense audit video and a representative contact sheet."""
    size = int(PollenAnchoredChainConfig(crop_size=args.chain_crop_size).crop_size)
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        float(args.review_fps),
        (size * 2, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {path}")
    tiles = []
    atlas_indices = set(np.rint(np.linspace(0, len(measurements) - 1, 20)).astype(int))
    for frame, measurement in enumerate(measurements):
        tile = _review_tile(
            inputs["grays"][frame],
            evidence[frame],
            measurement,
            transforms[frame],
            trajectory,
            frame,
            float(inputs["candidate"].radius),
            (
                material_diagnostics[frame]
                if material_diagnostics is not None
                else None
            ),
        )
        writer.write(tile)
        if frame in atlas_indices:
            tiles.append(cv.resize(tile, (size, size // 2), interpolation=cv.INTER_AREA))
    writer.release()
    rows = [np.concatenate(tiles[index : index + 4], axis=1) for index in range(0, len(tiles), 4)]
    width = max(row.shape[1] for row in rows)
    padded_rows = [cv.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv.BORDER_CONSTANT) for row in rows]
    cv.imwrite(str(path.with_name("review-atlas.jpg")), np.concatenate(padded_rows, axis=0))


def main():
    """Run moving-pollen tracking, whole-chain tracing, exports, and review media."""
    args = parse_args()
    _require_positive(args.time_per_source_frame, "time_per_source_frame")
    _require_positive(args.pixel_size, "pixel_size")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_config = _motion_config(args)
    if args.install_model:
        install_default_cotracker_checkpoint(motion_config.checkpoint)
    inputs = _load_inputs(args)
    trajectory, body_cache, reused = _load_or_track_motion(
        args, inputs, motion_config
    )
    chain_config = replace(
        PollenAnchoredChainConfig(),
        crop_size=int(args.chain_crop_size),
        baseline_observations=int(args.baseline_observations),
    )
    print("Stabilizing every frame around the tracked pollen body...")
    stabilized, transforms = stabilize_pollen_frames(
        inputs["grays"],
        trajectory.centers_xy,
        trajectory.angles_radians,
        chain_config.crop_size,
    )
    print("Building pre-germination novelty and tube-structure evidence...")
    _, evidence = prepare_pollen_anchored_evidence(stabilized, chain_config)
    exclusions = build_pollen_exclusions(
        transforms,
        inputs["pollen_positions"],
        inputs["target_index"],
        float(inputs["candidate"].radius),
        chain_config,
    )
    material_config = None
    ribbon_config = None
    material_track_cache = None
    material_cache_reused = None
    material_diagnostics = None
    if args.trace_mode in {"topology", "material"}:
        topology_config = TopologyAwareTraceConfig()
        print("Tracing future nodes and their connecting edges through time...")
        coarse_measurements = trace_topology_aware_chain(
            evidence,
            exclusions,
            float(inputs["candidate"].radius),
            chain_config,
            topology_config,
        )
        if args.trace_mode == "material":
            material_config = MaterialCurveConfig()
            ribbon_config = MaterialRibbonConfig(
                root_lock_points=material_config.root_lock_points
            )
            queries = build_material_queries(
                coarse_measurements, material_config
            )
            material_tracks, material_visibility, material_track_cache, material_cache_reused = (
                _load_or_track_material(
                    args,
                    stabilized,
                    queries,
                    motion_config,
                )
            )
            print("Fitting one ordered root-to-tip material curve through time...")
            material_trace = trace_material_curve(
                evidence,
                exclusions,
                float(inputs["candidate"].radius),
                coarse_measurements,
                queries,
                material_tracks,
                material_visibility,
                chain_config,
                material_config,
                ribbon_config,
            )
            measurements = material_trace.measurements
            material_diagnostics = material_trace.diagnostics
        else:
            measurements = coarse_measurements
    else:
        topology_config = None
        print("Tracing one complete pollen-to-tip chain through time...")
        measurements = trace_pollen_anchored_chain(
            evidence,
            exclusions,
            float(inputs["candidate"].radius),
            chain_config,
        )
    details, points, candidates = _measurement_tables(
        args,
        inputs,
        trajectory,
        measurements,
        transforms,
        material_diagnostics,
    )
    selected_frames = set(map(int, inputs["output_positions"]))
    selected_details = details[details["analysis_frame"].isin(selected_frames)]
    selected_points = points[points["analysis_frame"].isin(selected_frames)] if not points.empty else points.copy()
    details.to_csv(args.output_dir / "measurements.csv", index=False)
    selected_details.to_csv(args.output_dir / "selected_measurements.csv", index=False)
    points.to_csv(args.output_dir / "centerline_points.csv", index=False)
    selected_points.to_csv(args.output_dir / "selected_centerline_points.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_centerline_points.csv", index=False)
    summary = _summary(args, details)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    _write_review(
        args.output_dir / "review.mp4",
        args,
        inputs,
        trajectory,
        evidence,
        measurements,
        transforms,
        material_diagnostics,
    )
    manifest = {
        "prototype": (
            "pollen_stabilized_material_curve_v1"
            if material_config is not None
            else (
                "pollen_stabilized_topology_aware_chain_v1"
                if topology_config is not None
                else "pollen_stabilized_complete_chain_v1"
            )
        ),
        "pollen_id": args.pollen_id,
        "architecture": [
            "CoTracker3 redundant pollen-body landmarks",
            "rigid pollen-attached stabilization",
            "pre-germination baseline novelty",
            "short attached germination confirmation",
            "full-chain local refit and distal-only extension",
            *(
                [
                    "future-node and connecting-edge hypotheses",
                    "historical edge ownership and deferred branch selection",
                    "incremental tangent-consistent crossover reacquisition",
                ]
                if topology_config is not None
                else []
            ),
            *(
                [
                    "repeated full-curve material point keyframes",
                    "robust motion-coherent ordered-node fitting",
                    "occlusion inference without branch rewiring",
                    "paired-wall ribbon evidence for lateral curve placement",
                    "inextensible arc length with explicit curvature diagnostics",
                    "distal-only material-node insertion",
                ]
                if material_config is not None
                else []
            ),
        ],
        "inputs": {
            "gray_cache": str(args.gray_cache.expanduser().resolve()),
            "source_frames_cache": str(args.source_frames_cache.expanduser().resolve()),
            "grain_cache": str(args.grain_cache.expanduser().resolve()),
            "candidate_catalog": str(args.candidate_catalog.expanduser().resolve()),
            "chosen_cache_indices": inputs["chosen"].tolist(),
            "output_analysis_indices": inputs["output_positions"].tolist(),
        },
        "calibration": {
            "time_per_source_frame": args.time_per_source_frame,
            "pixel_size": args.pixel_size,
            "time_unit": args.time_unit,
            "distance_unit": args.distance_unit,
        },
        "motion": {
            "config": motion_config.cache_identity(),
            "body_track_cache": str(body_cache),
            "cache_reused": reused,
        },
        "chain_config": asdict(chain_config),
        "topology_config": (
            asdict(topology_config) if topology_config is not None else None
        ),
        "material_config": (
            asdict(material_config) if material_config is not None else None
        ),
        "ribbon_config": (
            asdict(ribbon_config) if ribbon_config is not None else None
        ),
        "material_tracking": (
            {
                "cache": str(material_track_cache),
                "cache_reused": material_cache_reused,
                "query_count": len(material_trace.queries.points_txy),
                "keyframe_count": len(material_trace.queries.keyframes),
            }
            if material_config is not None
            else None
        ),
        "result": summary.iloc[0].to_dict(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(summary.to_string(index=False))
    print(f"Review: {args.output_dir / 'review.mp4'}")


if __name__ == "__main__":
    main()
