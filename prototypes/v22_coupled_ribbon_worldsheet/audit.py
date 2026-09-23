#!/usr/bin/env python3
"""Refine one coarse tube as a high-resolution, two-wall video ribbon.

The existing worldsheet supplies pollen ownership, material order, and a coarse
centerline.  This audit returns to higher-resolution source frames, preserves
the two tube walls and their spacing, and fits every sampled time jointly.  It
tests whether wall evidence can recover faint material and resist crossovers
that are ambiguous after downsampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototypes.v17_birth_topology.track import load_movie_samples  # noqa: E402
from prototypes.v20_orientation_worldsheet.track import (  # noqa: E402
    candidate_bounds,
    foreign_grain_occupancy,
    load_candidate,
    path_orientation_profiles,
    reconstruct_phase,
)
from tubetracker.deformable_worldsheet import (  # noqa: E402
    atlas_prototype_score,
    DeformableWorldsheetConfig,
    fit_deformable_worldsheet,
    open_curve_certificate,
    orientation_blobness,
)
from tubetracker.growth_front import visible_path_front  # noqa: E402
from tubetracker.orientation_worldsheet import (  # noqa: E402
    aggregate_paired_wall_history,
    CoupledRibbonTraceConfig,
    OrientationScoreConfig,
    PairedWallOrientationResult,
    paired_wall_orientation_features,
    persistent_orientation_birth,
    propose_pollen_roots,
    ribbon_path_certificate,
    trace_coupled_ribbon_lifted,
)


RIBBON_REVISION = "v22.3-split-validated-coupled-ribbon-worldsheet"
ROOT_OCCLUSION_COARSE_PX = 6.0


def parse_args() -> argparse.Namespace:
    """Read one retained field case and high-resolution audit settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-run", type=Path, required=True)
    parser.add_argument("--v21-case", type=Path, required=True)
    parser.add_argument("--consensus-id", type=int, required=True)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--normal-radius-px", type=float, default=12.0)
    parser.add_argument("--corridor-margin-px", type=int, default=30)
    parser.add_argument(
        "--validate-temporal-splits",
        action="store_true",
        help="independently reconstruct the ribbon from alternating timepoints",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one retained CSV artifact into dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _centerline_rows(
    path: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """Load sampled frames, deformed centerlines, and their active fronts."""

    measurement_rows = _read_rows(path.parent / "deformable_measurements.csv")
    source_frames = np.asarray(
        [int(row["source_frame"]) for row in measurement_rows],
        dtype=int,
    )
    grouped: dict[int, list[tuple[int, float, float]]] = {}
    for row in _read_rows(path):
        source_frame = int(row["source_frame"])
        grouped.setdefault(source_frame, []).append(
            (
                int(row["path_index"]),
                float(row["y_px"]),
                float(row["x_px"]),
            )
        )
    curves = {
        frame: np.asarray(
            [(y, x) for _, y, x in sorted(points)],
            dtype=np.float64,
        )
        for frame, points in grouped.items()
    }
    fronts = {frame: len(curve) - 1 for frame, curve in curves.items()}
    return source_frames, curves, fronts


def _high_resolution_crops(
    movie: Path,
    output_source_frames: np.ndarray,
    phase_source_frames: np.ndarray,
    aligned_phase: np.ndarray,
    grain,
    low_bounds: tuple[int, int, int, int],
    low_width: int,
    high_width: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Map selected source frames into the original pollen-fixed coordinates."""

    scale = high_width / float(low_width)
    high_frames = load_movie_samples(movie, output_source_frames, high_width)
    phase_lookup = {
        int(frame): index for index, frame in enumerate(phase_source_frames)
    }
    phase_indices = np.asarray(
        [phase_lookup[int(frame)] for frame in output_source_frames],
        dtype=int,
    )
    y0, y1, x0, x1 = low_bounds
    high_bounds = tuple(int(round(value * scale)) for value in (y0, y1, x0, x1))
    high_y0, high_y1, high_x0, high_x1 = high_bounds
    crops = np.empty(
        (
            len(high_frames),
            high_y1 - high_y0,
            high_x1 - high_x0,
        ),
        dtype=np.uint8,
    )
    reference_xy = np.asarray(grain.center_xy, dtype=np.float64)
    height, width = high_frames.shape[1:]
    low_height, low_width = aligned_phase.shape[1:]
    hann = cv.createHanningWindow((low_width, low_height), cv.CV_32F)
    registration_responses = np.zeros(len(high_frames), dtype=np.float32)
    for output_index, phase_index in enumerate(phase_indices):
        low_frame = cv.resize(
            high_frames[output_index],
            (low_width, low_height),
            interpolation=cv.INTER_AREA,
        )
        source_registration = cv.Laplacian(
            cv.GaussianBlur(low_frame, (0, 0), 2.0),
            cv.CV_32F,
        )
        target_registration = cv.Laplacian(
            cv.GaussianBlur(aligned_phase[phase_index], (0, 0), 2.0),
            cv.CV_32F,
        )
        alignment_shift, response = cv.phaseCorrelate(
            source_registration,
            target_registration,
            hann,
        )
        alignment_shift = np.asarray(alignment_shift, dtype=np.float64)
        registration_responses[output_index] = float(response)
        if response < 0.08 or not np.isfinite(alignment_shift).all():
            alignment_shift[:] = 0.0
        grain_displacement = grain.center_at(int(phase_index)) - reference_xy
        translation = scale * (alignment_shift - grain_displacement)
        matrix = np.asarray(
            [
                [1.0, 0.0, translation[0]],
                [0.0, 1.0, translation[1]],
            ],
            dtype=np.float32,
        )
        registered = cv.warpAffine(
            high_frames[output_index],
            matrix,
            (width, height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        crops[output_index] = registered[high_y0:high_y1, high_x0:high_x1]
    return crops, scale, registration_responses


def _path_directions(path_yx: np.ndarray) -> np.ndarray:
    """Estimate a stable tangent angle at every centerline coordinate."""

    tangent = np.gradient(np.asarray(path_yx, dtype=np.float64), axis=0)
    return np.arctan2(tangent[:, 0], tangent[:, 1])


def _curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative analysis-resolution arclength along one curve."""

    path = np.asarray(path_yx, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )


def _sample_ribbon(
    paired_score: np.ndarray,
    half_width_px: np.ndarray,
    path_yx: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample strict wall-pair evidence and its chosen half-width."""

    count = paired_score.shape[0]
    bins = (
        np.rint(np.mod(directions, math.pi) * count / math.pi).astype(int) % count
    )
    paired = np.zeros(len(path_yx), dtype=np.float32)
    widths = np.zeros(len(path_yx), dtype=np.float32)
    for orientation in np.unique(bins):
        indices = np.flatnonzero(bins == orientation)
        map_x = path_yx[indices, 1][None, :].astype(np.float32)
        map_y = path_yx[indices, 0][None, :].astype(np.float32)
        paired[indices] = cv.remap(
            paired_score[int(orientation)],
            map_x,
            map_y,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )[0]
        widths[indices] = cv.remap(
            half_width_px[int(orientation)],
            map_x,
            map_y,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )[0]
    return paired, widths


def _sample_map(image: np.ndarray, path_yx: np.ndarray) -> np.ndarray:
    """Sample one scalar image along an ordered row-column curve."""

    return cv.remap(
        np.asarray(image, dtype=np.float32),
        path_yx[:, 1][None, :].astype(np.float32),
        path_yx[:, 0][None, :].astype(np.float32),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )[0]


def _ribbon_atlas_metrics(
    path_yx: np.ndarray,
    directions: np.ndarray,
    aggregate: PairedWallOrientationResult,
    preexisting_map: np.ndarray,
    foreign_occupancy: np.ndarray,
    root_occlusion_px: float,
) -> dict[str, float | bool]:
    """Score one complete ribbon using topology, walls, age, and ownership."""

    certificate = ribbon_path_certificate(
        aggregate,
        path_yx,
        directions,
        support_threshold=0.08,
        proximal_occlusion_px=root_occlusion_px,
    )
    openness = open_curve_certificate(path_yx)
    arc = _curve_arclength(path_yx)
    evaluated = arc >= root_occlusion_px
    old_material = _sample_map(preexisting_map, path_yx)
    preexisting_support = (
        float(np.mean(old_material[evaluated])) if np.any(evaluated) else 1.0
    )
    terminal_blobness = orientation_blobness(
        aggregate.score,
        path_yx[-1],
        max(4.0, 2.0 * certificate.median_half_width_px),
        support_threshold=0.08,
    )
    terminal_foreign_occupancy = float(
        _sample_map(foreign_occupancy, path_yx[-1:])[0]
    )
    terminal_foreign_body_risk = terminal_blobness * math.sqrt(
        max(preexisting_support, 0.0)
    )
    score = atlas_prototype_score(
        openness.length_px,
        certificate.paired_mean_support,
        certificate.paired_supported_fraction,
        0.0,
        preexisting_support,
        terminal_blobness,
    ) - 2.0 * terminal_foreign_occupancy
    eligible = (
        openness.endpoint_separation_fraction >= 0.25
        and certificate.paired_supported_fraction >= 0.30
        and terminal_foreign_occupancy < 0.20
        and terminal_foreign_body_risk <= 0.18
    )
    return {
        "score": score,
        "eligible": eligible,
        "length_px": openness.length_px,
        "endpoint_separation_fraction": openness.endpoint_separation_fraction,
        "paired_mean_support": certificate.paired_mean_support,
        "paired_supported_fraction": certificate.paired_supported_fraction,
        "median_half_width_px": certificate.median_half_width_px,
        "width_mad_px": certificate.width_mad_px,
        "preexisting_support": preexisting_support,
        "terminal_blobness": terminal_blobness,
        "terminal_foreign_body_risk": terminal_foreign_body_risk,
        "terminal_foreign_occupancy": terminal_foreign_occupancy,
    }


def _discover_coupled_ribbon_atlas(
    aggregate: PairedWallOrientationResult,
    birth_time: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    preexisting_map: np.ndarray,
    foreign_occupancy: np.ndarray,
    maximum_length_px: float,
    root_occlusion_px: float,
) -> tuple[object | None, list[dict], str]:
    """Search the complete pollen rim for a width-aware high-resolution atlas."""

    evidence = np.maximum(aggregate.paired_score, 0.15 * aggregate.score).copy()
    occupied = foreign_occupancy >= 0.20
    evidence[:, occupied] = 0.0
    birth = np.asarray(birth_time, dtype=np.float32).copy()
    birth[:, occupied] = 0.0
    trace_config = CoupledRibbonTraceConfig(
        step_px=1.5,
        maximum_turn_bins=1,
        curvature_penalty=0.16,
        maximum_gap_steps=7,
        maximum_initial_gap_steps=6,
        root_occlusion_px=root_occlusion_px,
        beam_width=180,
        minimum_length_px=max(18.0, 1.5 * pollen_radius_px),
        maximum_length_px=maximum_length_px,
        maximum_reacquisition_width_change_px=max(1.5, 0.35 * pollen_radius_px),
    )
    families = (
        ("causal-new-material", birth, 2.0),
        ("left-censored-preexisting", None, None),
    )
    candidates = []
    best_trace = None
    best_scope = "none"
    for temporal_scope, family_birth, minimum_birth in families:
        proposals = propose_pollen_roots(
            evidence,
            pollen_center_yx,
            pollen_radius_px,
            birth_time=family_birth,
            minimum_material_birth=minimum_birth,
            attachment_count=72,
            direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
            probe_distances_px=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            probe_top_k=4,
            maximum_proposals=20,
            minimum_angle_separation_degrees=9.0,
        )
        for proposal in proposals:
            trace = trace_coupled_ribbon_lifted(
                aggregate,
                proposal.root_yx,
                proposal.direction_yx,
                birth_time=family_birth,
                minimum_material_birth=minimum_birth,
                config=trace_config,
            )
            if trace.length_px < trace_config.minimum_length_px - 1.0:
                continue
            metrics = _ribbon_atlas_metrics(
                trace.path_yx,
                trace.direction_radians,
                aggregate,
                preexisting_map,
                foreign_occupancy,
                root_occlusion_px,
            )
            record = {
                "temporal_scope": temporal_scope,
                "root_yx": proposal.root_yx.tolist(),
                "direction_yx": proposal.direction_yx.tolist(),
                **metrics,
            }
            candidates.append((trace, record))
        eligible = [item for item in candidates if item[1]["eligible"]]
        if eligible:
            best_trace, winner = max(
                eligible,
                key=lambda item: float(item[1]["score"]),
            )
            best_scope = str(winner["temporal_scope"])
            if temporal_scope == "causal-new-material":
                break
    ordered = sorted(
        (record for _, record in candidates),
        key=lambda record: (bool(record["eligible"]), float(record["score"])),
        reverse=True,
    )
    return best_trace, ordered, best_scope


def _normalized_curve_agreement(
    first_yx: np.ndarray,
    second_yx: np.ndarray,
    sample_count: int = 101,
) -> dict[str, float]:
    """Compare two open curves at corresponding normalized arc positions."""

    first = np.asarray(first_yx, dtype=np.float64)
    second = np.asarray(second_yx, dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("curve agreement requires two non-degenerate curves")
    first_arc = _curve_arclength(first)
    second_arc = _curve_arclength(second)
    samples = np.linspace(0.0, 1.0, sample_count)
    first_sampled = np.column_stack(
        [
            np.interp(samples * first_arc[-1], first_arc, first[:, axis])
            for axis in range(2)
        ]
    )
    second_sampled = np.column_stack(
        [
            np.interp(samples * second_arc[-1], second_arc, second[:, axis])
            for axis in range(2)
        ]
    )
    errors = np.linalg.norm(first_sampled - second_sampled, axis=1)
    mean_length = max(0.5 * (first_arc[-1] + second_arc[-1]), 1e-6)
    pairwise = np.linalg.norm(
        first_sampled[:, None, :] - second_sampled[None, :, :],
        axis=2,
    )
    return {
        "first_length_px": float(first_arc[-1]),
        "second_length_px": float(second_arc[-1]),
        "relative_length_difference": float(
            abs(first_arc[-1] - second_arc[-1]) / mean_length
        ),
        "root_error_px": float(errors[0]),
        "endpoint_error_px": float(errors[-1]),
        "median_corresponding_error_px": float(np.median(errors)),
        "p90_corresponding_error_px": float(np.percentile(errors, 90.0)),
        "symmetric_chamfer_error_px": float(
            0.5
            * (
                np.mean(np.min(pairwise, axis=0))
                + np.mean(np.min(pairwise, axis=1))
            )
        ),
    }


def _alternating_timepoint_validation(
    normalized_score: np.ndarray,
    normalized_pair: np.ndarray,
    normalized_width: np.ndarray,
    normalized_balance: np.ndarray,
    source_frames: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    foreign_occupancy: np.ndarray,
    maximum_length_px: float,
    root_occlusion_px: float,
) -> dict:
    """Reconstruct from disjoint timepoints and test cross-split agreement."""

    split_results = []
    for parity, label in enumerate(("even-timepoints", "odd-timepoints")):
        indices = np.arange(parity, len(normalized_score), 2, dtype=int)
        score = normalized_score[indices]
        pair = normalized_pair[indices]
        width = normalized_width[indices]
        balance = normalized_balance[indices]
        aggregate_start = max(0, int(math.floor(0.60 * len(indices))))
        aggregate = aggregate_paired_wall_history(
            score,
            pair,
            width,
            balance,
            start_index=aggregate_start,
            percentile=75.0,
        )
        warmup = max(2, min(len(indices) // 8, len(indices) - 1))
        birth = persistent_orientation_birth(
            score,
            warmup_samples=warmup,
            persistence_samples=3,
            minimum_change=0.06,
        )
        preexisting = np.percentile(
            np.max(score[:warmup], axis=1),
            75.0,
            axis=0,
        ).astype(np.float32)
        trace, _, scope = _discover_coupled_ribbon_atlas(
            aggregate,
            birth,
            pollen_center_yx,
            pollen_radius_px,
            preexisting,
            foreign_occupancy,
            maximum_length_px,
            root_occlusion_px,
        )
        own_metrics = None
        if trace is not None:
            own_metrics = _ribbon_atlas_metrics(
                trace.path_yx,
                trace.direction_radians,
                aggregate,
                preexisting,
                foreign_occupancy,
                root_occlusion_px,
            )
        split_results.append(
            {
                "label": label,
                "indices": indices,
                "source_frames": source_frames[indices],
                "aggregate": aggregate,
                "preexisting": preexisting,
                "trace": trace,
                "temporal_scope": scope,
                "own_metrics": own_metrics,
            }
        )

    for index, split in enumerate(split_results):
        trace = split["trace"]
        held_out = split_results[1 - index]
        split["held_out_metrics"] = (
            _ribbon_atlas_metrics(
                trace.path_yx,
                trace.direction_radians,
                held_out["aggregate"],
                held_out["preexisting"],
                foreign_occupancy,
                root_occlusion_px,
            )
            if trace is not None
            else None
        )

    first_trace = split_results[0]["trace"]
    second_trace = split_results[1]["trace"]
    agreement = None
    supported = False
    thresholds = None
    if first_trace is not None and second_trace is not None:
        agreement = _normalized_curve_agreement(
            first_trace.path_yx,
            second_trace.path_yx,
        )
        half_widths = [
            float(split["own_metrics"]["median_half_width_px"])
            for split in split_results
            if split["own_metrics"] is not None
        ]
        representative_half_width = float(np.median(half_widths))
        thresholds = {
            "maximum_relative_length_difference": 0.20,
            "maximum_median_error_px": max(4.0, 1.5 * representative_half_width),
            "maximum_p90_error_px": max(8.0, 3.0 * representative_half_width),
            "maximum_endpoint_error_px": max(10.0, 4.0 * representative_half_width),
        }
        supported = bool(
            all(bool(split["own_metrics"]["eligible"]) for split in split_results)
            and all(
                float(split["held_out_metrics"]["paired_supported_fraction"])
                >= 0.30
                for split in split_results
            )
            and agreement["relative_length_difference"]
            <= thresholds["maximum_relative_length_difference"]
            and agreement["median_corresponding_error_px"]
            <= thresholds["maximum_median_error_px"]
            and agreement["p90_corresponding_error_px"]
            <= thresholds["maximum_p90_error_px"]
            and agreement["endpoint_error_px"]
            <= thresholds["maximum_endpoint_error_px"]
        )

    serializable_splits = []
    for split in split_results:
        serializable_splits.append(
            {
                "label": split["label"],
                "source_frames": split["source_frames"].tolist(),
                "trace_found": split["trace"] is not None,
                "temporal_scope": split["temporal_scope"],
                "path_yx": split["trace"].path_yx.tolist()
                if split["trace"] is not None
                else None,
                "own_metrics": split["own_metrics"],
                "held_out_metrics": split["held_out_metrics"],
            }
        )
    return {
        "method": "independent alternating-timepoint reconstruction",
        "supported": supported,
        "splits": serializable_splits,
        "agreement": agreement,
        "thresholds": thresholds,
    }


def _draw_polyline(image: np.ndarray, path_yx: np.ndarray, color, width: int) -> None:
    """Draw one open row-column curve with antialiasing."""

    if len(path_yx) < 2:
        return
    points = np.rint(path_yx[:, ::-1]).astype(np.int32)
    cv.polylines(image, [points], False, color, width, cv.LINE_AA)


def _write_split_review(
    output: Path,
    crop: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    validation: dict,
) -> None:
    """Draw the two independently reconstructed centerlines together."""

    overlay = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
    cv.circle(
        overlay,
        tuple(np.rint(pollen_center_yx[::-1]).astype(int)),
        max(2, int(round(pollen_radius_px))),
        (230, 100, 20),
        2,
        cv.LINE_AA,
    )
    colors = ((70, 220, 70), (220, 70, 220))
    for split, color in zip(validation["splits"], colors):
        if split["path_yx"] is not None:
            _draw_polyline(
                overlay,
                np.asarray(split["path_yx"], dtype=np.float64),
                color,
                2,
            )
    status = "PASS" if validation["supported"] else "REVIEW"
    cv.putText(
        overlay,
        f"SPLIT {status} | green even | magenta odd",
        (8, 22),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv.LINE_AA,
    )
    if not cv.imwrite(str(output), overlay):
        raise RuntimeError(f"Could not write split review image to {output}")


def _write_review(
    output: Path,
    crops: np.ndarray,
    source_frames: np.ndarray,
    fixed_atlas: np.ndarray,
    front_indices: np.ndarray,
    fitted,
    paired_stack: np.ndarray,
    width_stack: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    root_occlusion_px: float,
    geometry_supported: bool,
    reproducibility_supported: bool | None,
) -> None:
    """Render centerline and explicit wall tracks over high-resolution data."""

    height, width = crops.shape[1:]
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (2 * width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    for time_index, source_frame in enumerate(source_frames):
        raw = cv.cvtColor(crops[time_index], cv.COLOR_GRAY2BGR)
        overlay = raw.copy()
        cv.circle(
            overlay,
            tuple(np.rint(pollen_center_yx[::-1]).astype(int)),
            max(2, int(round(pollen_radius_px))),
            (230, 100, 20),
            2,
            cv.LINE_AA,
        )
        tip_index = int(front_indices[time_index])
        if geometry_supported and tip_index >= 1:
            fixed = fixed_atlas[: tip_index + 1]
            curve = fitted.curves_yx[time_index, : tip_index + 1]
            directions = _path_directions(curve)
            paired, widths = _sample_ribbon(
                paired_stack[time_index].astype(np.float32) / 255.0,
                width_stack[time_index].astype(np.float32) / 16.0,
                curve,
                directions,
            )
            normals = np.column_stack((-np.cos(directions), np.sin(directions)))
            left = curve - widths[:, None] * normals
            right = curve + widths[:, None] * normals
            evaluated = _curve_arclength(curve) >= root_occlusion_px
            supported = (paired >= 0.08) & evaluated
            left[~supported] = np.nan
            right[~supported] = np.nan
            _draw_polyline(overlay, fixed, (60, 210, 60), 1)
            _draw_polyline(overlay, curve, (210, 60, 220), 2)
            for wall in (left, right):
                finite = np.isfinite(wall[:, 0])
                starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
                ends = np.flatnonzero(finite & np.r_[~finite[1:], True])
                for start, end in zip(starts, ends):
                    if end > start:
                        _draw_polyline(overlay, wall[start : end + 1], (0, 220, 255), 1)
            paired_fraction = (
                float(np.mean(supported[evaluated]))
                if np.any(evaluated)
                else 0.0
            )
            if reproducibility_supported is False:
                status = f"REVIEW split disagreement | walls {paired_fraction:.0%}"
            else:
                status = f"distal paired walls {paired_fraction:.0%}"
        elif geometry_supported:
            status = "pre-growth"
        else:
            status = "WITHHELD: no eligible tube geometry"
        cv.putText(
            raw,
            f"high-res source {int(source_frame)}",
            (8, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            "RIBBON WORLDSHEET",
            (8, 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            f"magenta center | yellow walls | {status}",
            (8, 36),
            cv.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        writer.write(np.hstack((raw, overlay)))
    writer.release()


def main() -> None:
    """Fit and export one high-resolution ribbon-worldsheet audit."""

    args = parse_args()
    if args.analysis_width <= 0:
        raise ValueError("analysis width must be positive")
    if args.corridor_margin_px <= 0:
        raise ValueError("corridor margin must be positive")
    if args.normal_radius_px <= 0.0:
        raise ValueError("normal radius must be positive")
    v18_report = json.loads((args.v18_run / "report.json").read_text())
    v21_report = json.loads((args.v21_case / "report.json").read_text())
    summary, _ = load_candidate(args.v18_run, args.consensus_id)
    source_frames, low_curves, low_fronts = _centerline_rows(
        args.v21_case / "deformable_centerlines.csv"
    )
    final_source_frame = max(low_curves)
    final_curve = low_curves[final_source_frame]
    movie, phase_frames, aligned, _, grain, grains = reconstruct_phase(
        v18_report,
        summary,
    )
    low_width = int(v18_report["configuration"]["width"])
    if args.analysis_width <= low_width:
        raise ValueError("analysis width must exceed the coarse analysis width")
    low_bounds = candidate_bounds(
        final_curve,
        aligned.shape[1:],
        args.corridor_margin_px,
    )
    crops, scale, high_resolution_registration_response = _high_resolution_crops(
        movie,
        source_frames,
        phase_frames,
        aligned,
        grain,
        low_bounds,
        low_width,
        args.analysis_width,
    )
    del aligned
    phase_lookup = {int(frame): index for index, frame in enumerate(phase_frames)}
    phase_indices = np.asarray(
        [phase_lookup[int(frame)] for frame in source_frames],
        dtype=int,
    )
    low_occupancy = foreign_grain_occupancy(
        grains,
        grain,
        phase_indices,
        low_bounds,
    )
    foreign_occupancy = cv.resize(
        low_occupancy,
        (crops.shape[2], crops.shape[1]),
        interpolation=cv.INTER_LINEAR,
    ).astype(np.float32)
    y0, _, x0, _ = low_bounds
    coarse_atlas = (
        final_curve - np.asarray((y0, x0), dtype=np.float64)[None, :]
    ) * scale
    coarse_front_indices = np.asarray(
        [low_fronts.get(int(frame), -1) for frame in source_frames],
        dtype=int,
    )
    coarse_front_indices = np.clip(
        coarse_front_indices,
        -1,
        len(coarse_atlas) - 1,
    )

    score_config = OrientationScoreConfig(
        half_widths_px=tuple(
            value * scale for value in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
        ),
        tangent_samples_px=tuple(value * scale for value in (-2, -1, 0, 1, 2)),
        wall_sigma_px=0.65 * scale,
        background_sigma_px=4.0 * scale,
        structure_sigma_px=1.2 * scale,
    )
    scores = []
    paired_frames = []
    width_frames = []
    balance_frames = []
    for index, crop in enumerate(crops):
        features = paired_wall_orientation_features(crop, score_config)
        scores.append(
            np.rint(
                255.0
                * np.maximum(features.paired_score, 0.15 * features.score)
            ).astype(np.uint8)
        )
        paired_frames.append(
            np.rint(255.0 * features.paired_score).astype(np.uint8)
        )
        width_frames.append(
            np.rint(16.0 * features.half_width_px).astype(np.uint8)
        )
        balance_frames.append(
            np.rint(255.0 * features.wall_balance).astype(np.uint8)
        )
        print(f"[v22] ribbon evidence {index + 1}/{len(crops)}", flush=True)
    score_stack = np.asarray(scores, dtype=np.uint8)
    paired_stack = np.asarray(paired_frames, dtype=np.uint8)
    width_stack = np.asarray(width_frames, dtype=np.uint8)
    balance_stack = np.asarray(balance_frames, dtype=np.uint8)
    occupied = foreign_occupancy >= 0.20
    score_stack[:, :, occupied] = 0
    paired_stack[:, :, occupied] = 0
    width_stack[:, :, occupied] = 0
    balance_stack[:, :, occupied] = 0
    normalized_score = score_stack.astype(np.float32) / 255.0
    normalized_pair = paired_stack.astype(np.float32) / 255.0
    normalized_width = width_stack.astype(np.float32) / 16.0
    normalized_balance = balance_stack.astype(np.float32) / 255.0
    aggregate_start = max(0, int(math.floor(0.60 * len(score_stack))))
    aggregate = aggregate_paired_wall_history(
        normalized_score,
        normalized_pair,
        normalized_width,
        normalized_balance,
        start_index=aggregate_start,
        percentile=75.0,
    )
    birth_warmup = max(2, min(len(score_stack) // 8, len(score_stack) - 1))
    ribbon_birth = persistent_orientation_birth(
        normalized_score,
        warmup_samples=birth_warmup,
        persistence_samples=3,
        minimum_change=0.06,
    )
    preexisting_map = np.percentile(
        np.max(normalized_score[:birth_warmup], axis=1),
        75.0,
        axis=0,
    ).astype(np.float32)
    pollen_center = (
        np.asarray(grain.center_xy[::-1], dtype=np.float64)
        - np.asarray((y0, x0), dtype=np.float64)
    ) * scale
    root_occlusion_px = ROOT_OCCLUSION_COARSE_PX * scale
    maximum_atlas_length_px = min(
        1.30 * float(_curve_arclength(coarse_atlas)[-1]),
        0.90 * math.hypot(crops.shape[1], crops.shape[2]),
    )
    independent_atlas, ribbon_hypotheses, atlas_temporal_scope = (
        _discover_coupled_ribbon_atlas(
            aggregate,
            ribbon_birth,
            pollen_center,
            float(grain.radius_px) * scale,
            preexisting_map,
            foreign_occupancy,
            maximum_length_px=maximum_atlas_length_px,
            root_occlusion_px=root_occlusion_px,
        )
    )
    coarse_directions = _path_directions(coarse_atlas)
    coarse_metrics = _ribbon_atlas_metrics(
        coarse_atlas,
        coarse_directions,
        aggregate,
        preexisting_map,
        foreign_occupancy,
        root_occlusion_px,
    )
    coarse_record = {"temporal_scope": "coarse-causal-atlas", **coarse_metrics}
    selected_atlas_source = "coarse-causal-atlas"
    fixed_atlas = coarse_atlas
    selected_directions = coarse_directions
    if independent_atlas is not None:
        independent_metrics = _ribbon_atlas_metrics(
            independent_atlas.path_yx,
            independent_atlas.direction_radians,
            aggregate,
            preexisting_map,
            foreign_occupancy,
            root_occlusion_px,
        )
        if (
            bool(independent_metrics["eligible"])
            and (
                not bool(coarse_metrics["eligible"])
                or float(independent_metrics["score"])
                > float(coarse_metrics["score"])
            )
        ):
            fixed_atlas = independent_atlas.path_yx
            selected_directions = independent_atlas.direction_radians
            selected_atlas_source = atlas_temporal_scope

    selected_metrics = _ribbon_atlas_metrics(
        fixed_atlas,
        _path_directions(fixed_atlas),
        aggregate,
        preexisting_map,
        foreign_occupancy,
        root_occlusion_px,
    )
    geometry_supported = bool(selected_metrics["eligible"])
    temporal_split_validation = None
    if args.validate_temporal_splits:
        print("[v22] validating alternating timepoint reconstructions", flush=True)
        temporal_split_validation = _alternating_timepoint_validation(
            normalized_score,
            normalized_pair,
            normalized_width,
            normalized_balance,
            source_frames,
            pollen_center,
            float(grain.radius_px) * scale,
            foreign_occupancy,
            maximum_atlas_length_px,
            root_occlusion_px,
        )
    reproducibility_supported = (
        bool(temporal_split_validation["supported"])
        if temporal_split_validation is not None
        else None
    )

    if not geometry_supported:
        front_indices = np.full(len(source_frames), -1, dtype=int)
        front_source = "withheld-ineligible-atlas"
    elif selected_atlas_source == "left-censored-preexisting":
        front_indices = np.full(len(source_frames), len(fixed_atlas) - 1, dtype=int)
        front_source = "left-censored-full-ribbon"
    elif selected_atlas_source == "causal-new-material":
        profiles = path_orientation_profiles(
            score_stack,
            fixed_atlas,
            selected_directions,
            birth_warmup,
        )
        front = visible_path_front(
            profiles,
            _curve_arclength(fixed_atlas),
            warmup_samples=birth_warmup,
            max_step_px=6.0 * scale,
            coverage_cost=0.035,
            absence_weight=0.20,
            motion_penalty=0.12,
            support_window_samples=3,
            require_final_endpoint=True,
        )
        if front.feasible:
            front_indices = np.asarray(front.front_indices, dtype=int)
            front_source = "high-resolution-visible-prefix"
        else:
            front_indices = coarse_front_indices.copy()
            front_source = f"coarse-fallback:{front.reason}"
    else:
        front_indices = coarse_front_indices.copy()
        front_source = "coarse-causal-front"

    if len(fixed_atlas) != len(coarse_atlas) and front_source.startswith("coarse-"):
        coarse_arc = _curve_arclength(coarse_atlas)
        selected_arc = _curve_arclength(fixed_atlas)
        mapped = []
        for index in coarse_front_indices:
            if index < 0:
                mapped.append(-1)
                continue
            fraction = coarse_arc[index] / max(coarse_arc[-1], 1e-6)
            mapped.append(
                int(np.searchsorted(selected_arc, fraction * selected_arc[-1]))
            )
        front_indices = np.clip(mapped, -1, len(fixed_atlas) - 1)

    spacing = float(np.median(np.diff(_curve_arclength(fixed_atlas))))
    fit_config = DeformableWorldsheetConfig(
        normal_radius_px=args.normal_radius_px,
        normal_step_px=1.0,
        pairwise_smoothness=0.34,
        pairwise_truncation_px=4.0 * scale,
        unsupported_cost=1.0,
        root_lock_nodes=max(2, int(math.ceil(root_occlusion_px / spacing))),
        maximum_cycles=10,
    )
    fitted = fit_deformable_worldsheet(
        normalized_score,
        fixed_atlas,
        front_indices=front_indices,
        config=fit_config,
    )
    del scores, paired_frames, width_frames, balance_frames

    rows = []
    centerline_rows = []
    material_arc = _curve_arclength(fixed_atlas)
    for time_index, source_frame in enumerate(source_frames):
        tip_index = int(front_indices[time_index])
        if not geometry_supported:
            rows.append(
                {
                    "source_frame": int(source_frame),
                    "status": "withheld",
                    "paired_supported_fraction": "",
                    "paired_mean_support": "",
                    "median_full_width_px": "",
                    "width_mad_px": "",
                    "material_length_analysis_px": "",
                    "material_length_coarse_px": "",
                    "tip_x_crop_px": "",
                    "tip_y_crop_px": "",
                }
            )
            continue
        if tip_index < 1 or material_arc[tip_index] <= root_occlusion_px:
            rows.append(
                {
                    "source_frame": int(source_frame),
                    "status": "pre-growth",
                    "paired_supported_fraction": "",
                    "paired_mean_support": "",
                    "median_full_width_px": "",
                    "width_mad_px": "",
                    "material_length_analysis_px": "",
                    "material_length_coarse_px": "",
                    "tip_x_crop_px": "",
                    "tip_y_crop_px": "",
                }
            )
            continue
        curve = fitted.curves_yx[time_index, : tip_index + 1]
        paired_score = paired_stack[time_index].astype(np.float32) / 255.0
        half_width = width_stack[time_index].astype(np.float32) / 16.0
        wall_balance = balance_stack[time_index].astype(np.float32) / 255.0
        features = PairedWallOrientationResult(
            score=paired_score,
            paired_score=paired_score,
            half_width_px=half_width,
            wall_balance=wall_balance,
        )
        directions = _path_directions(curve)
        certificate = ribbon_path_certificate(
            features,
            curve,
            directions,
            support_threshold=0.08,
            proximal_occlusion_px=ROOT_OCCLUSION_COARSE_PX * scale,
        )
        sampled_pair, sampled_width = _sample_ribbon(
            paired_score,
            half_width,
            curve,
            directions,
        )
        normals = np.column_stack((-np.cos(directions), np.sin(directions)))
        left = curve - sampled_width[:, None] * normals
        right = curve + sampled_width[:, None] * normals
        for path_index, point in enumerate(curve):
            walls_supported = sampled_pair[path_index] >= 0.08
            centerline_rows.append(
                {
                    "source_frame": int(source_frame),
                    "path_index": path_index,
                    "material_position_analysis_px": round(
                        float(material_arc[path_index]),
                        4,
                    ),
                    "material_position_coarse_px": round(
                        float(material_arc[path_index] / scale),
                        4,
                    ),
                    "center_x_crop_px": round(float(point[1]), 4),
                    "center_y_crop_px": round(float(point[0]), 4),
                    "paired_wall_support": round(
                        float(sampled_pair[path_index]),
                        6,
                    ),
                    "full_width_analysis_px": round(
                        float(2.0 * sampled_width[path_index]),
                        4,
                    )
                    if walls_supported
                    else "",
                    "left_x_crop_px": round(float(left[path_index, 1]), 4)
                    if walls_supported
                    else "",
                    "left_y_crop_px": round(float(left[path_index, 0]), 4)
                    if walls_supported
                    else "",
                    "right_x_crop_px": round(float(right[path_index, 1]), 4)
                    if walls_supported
                    else "",
                    "right_y_crop_px": round(float(right[path_index, 0]), 4)
                    if walls_supported
                    else "",
                }
            )
        rows.append(
            {
                "source_frame": int(source_frame),
                "status": "accepted"
                if (
                    certificate.paired_supported_fraction >= 0.50
                    and reproducibility_supported is not False
                )
                else "review",
                "paired_supported_fraction": round(
                    certificate.paired_supported_fraction,
                    6,
                ),
                "paired_mean_support": round(certificate.paired_mean_support, 6),
                "median_full_width_px": round(
                    2.0 * certificate.median_half_width_px,
                    4,
                ),
                "width_mad_px": round(certificate.width_mad_px, 4),
                "material_length_analysis_px": round(
                    float(material_arc[tip_index]),
                    4,
                ),
                "material_length_coarse_px": round(
                    float(material_arc[tip_index] / scale),
                    4,
                ),
                "tip_x_crop_px": round(float(curve[-1, 1]), 4),
                "tip_y_crop_px": round(float(curve[-1, 0]), 4),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurement_path = args.output_dir / "ribbon_measurements.csv"
    with measurement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_frame",
                "status",
                "paired_supported_fraction",
                "paired_mean_support",
                "median_full_width_px",
                "width_mad_px",
                "material_length_analysis_px",
                "material_length_coarse_px",
                "tip_x_crop_px",
                "tip_y_crop_px",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    centerline_path = args.output_dir / "ribbon_centerlines.csv"
    with centerline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_frame",
                "path_index",
                "material_position_analysis_px",
                "material_position_coarse_px",
                "center_x_crop_px",
                "center_y_crop_px",
                "paired_wall_support",
                "full_width_analysis_px",
                "left_x_crop_px",
                "left_y_crop_px",
                "right_x_crop_px",
                "right_y_crop_px",
            ),
        )
        writer.writeheader()
        writer.writerows(centerline_rows)
    video_path = args.output_dir / "high_resolution_ribbon_review.mp4"
    _write_review(
        video_path,
        crops,
        source_frames,
        fixed_atlas,
        front_indices,
        fitted,
        paired_stack,
        width_stack,
        pollen_center,
        float(grain.radius_px) * scale,
        ROOT_OCCLUSION_COARSE_PX * scale,
        geometry_supported,
        reproducibility_supported,
    )
    active_rows = [
        row for row in rows if row["status"] in {"accepted", "review"}
    ]
    accepted_suffix = 0
    for row in reversed(active_rows):
        if row["status"] != "accepted":
            break
        accepted_suffix += 1
    split_review_path = None
    if temporal_split_validation is not None:
        split_review_path = args.output_dir / "temporal_split_review.png"
        _write_split_review(
            split_review_path,
            crops[-1],
            pollen_center,
            float(grain.radius_px) * scale,
            temporal_split_validation,
        )
    germination_time_supported = bool(
        geometry_supported
        and v21_report.get("germination_time_supported", False)
        and selected_atlas_source != "left-censored-preexisting"
    )
    if not geometry_supported:
        validation_disposition = "rejected-geometry"
    elif reproducibility_supported is False:
        validation_disposition = "review-reproducibility"
    elif reproducibility_supported is True:
        validation_disposition = "accepted-geometry"
    else:
        validation_disposition = "unvalidated-geometry"
    length_measurement_supported = bool(
        geometry_supported
        and reproducibility_supported is not False
        and any(row["status"] == "accepted" for row in active_rows)
    )
    result = {
        "prototype": "v22_high_resolution_ribbon_worldsheet",
        "algorithm_revision": RIBBON_REVISION,
        "consensus_id": args.consensus_id,
        "input_movie": str(movie),
        "coarse_case": str(args.v21_case),
        "coarse_algorithm_revision": v21_report.get("algorithm_revision"),
        "analysis_width": args.analysis_width,
        "scale_from_coarse": scale,
        "corridor_margin_coarse_px": args.corridor_margin_px,
        "root_occlusion_coarse_px": ROOT_OCCLUSION_COARSE_PX,
        "high_resolution_registration_response_median": float(
            np.median(high_resolution_registration_response)
        ),
        "high_resolution_registration_response_minimum": float(
            np.min(high_resolution_registration_response)
        ),
        "selected_atlas_source": selected_atlas_source,
        "front_source": front_source,
        "geometry_supported": geometry_supported,
        "reproducibility_supported": reproducibility_supported,
        "length_measurement_supported": length_measurement_supported,
        "germination_time_supported": germination_time_supported,
        "validation_disposition": validation_disposition,
        "coarse_atlas": coarse_record,
        "selected_atlas": selected_metrics,
        "temporal_split_validation": temporal_split_validation,
        "independent_ribbon_hypotheses": ribbon_hypotheses,
        "active_timepoints": len(active_rows),
        "accepted_ribbon_timepoints": sum(
            row["status"] == "accepted" for row in active_rows
        ),
        "accepted_ribbon_fraction": (
            sum(row["status"] == "accepted" for row in active_rows)
            / len(active_rows)
            if active_rows
            else 0.0
        ),
        "consecutive_accepted_final_timepoints": accepted_suffix,
        "median_paired_supported_fraction": float(
            np.median(
                [float(row["paired_supported_fraction"]) for row in active_rows]
            )
        )
        if active_rows
        else 0.0,
        "artifacts": {
            "review_video": str(video_path),
            "measurements_csv": str(measurement_path),
            "centerlines_csv": str(centerline_path),
            "temporal_split_review": str(split_review_path)
            if split_review_path is not None
            else None,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
