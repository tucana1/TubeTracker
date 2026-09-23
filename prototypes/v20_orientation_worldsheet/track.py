#!/usr/bin/env python3
"""Audit a causal orientation worldsheet on one real v18 tube candidate.

The prototype reconstructs the exact v18 stabilization phase, converts a
pollen-fixed crop into direction-specific paired-wall evidence, estimates when
material first appears in every direction, and traces the tube without creating
a planar junction at crossings.  It remains an isolated research audit and does
not alter v18 output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototypes.v17_birth_topology.track import (  # noqa: E402
    AtlasConfig,
    detect_grain_anchors,
    load_movie_samples,
    movie_metadata,
    source_frame_indices,
    stabilize_translations,
)
from tubetracker.orientation_worldsheet import (  # noqa: E402
    OrientationTraceConfig,
    PollenRootProposal,
    paired_wall_orientation_score,
    persistent_orientation_birth,
    persistent_orientation_score,
    propose_pollen_roots,
    trace_orientation_lifted,
)
from tubetracker.growth_front import visible_path_front  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read the retained v18 run, target candidate, and audit settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-run", type=Path, required=True)
    parser.add_argument("--consensus-id", type=int, required=True)
    parser.add_argument("--evidence-samples", type=int, default=180)
    parser.add_argument("--output-samples", type=int, default=60)
    parser.add_argument("--crop-margin-px", type=int, default=55)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries while preserving its stored values."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_candidate(run_dir: Path, consensus_id: int):
    """Load one selected v18 candidate and its ordered source-phase path."""

    summaries = _read_rows(run_dir / "phase_consensus_summary.csv")
    selected = [
        row
        for row in summaries
        if row["consensus_id"]
        and int(row["consensus_id"]) == consensus_id
        and row["selected"] == "True"
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected consensus candidate {consensus_id}")
    summary = selected[0]
    paths = [
        row
        for row in _read_rows(run_dir / "phase_consensus_paths.csv")
        if row["consensus_id"]
        and int(row["consensus_id"]) == consensus_id
        and row["selected"] == "True"
    ]
    paths.sort(key=lambda row: int(row["path_index"]))
    if len(paths) < 2:
        raise ValueError("Selected candidate does not contain an ordered path")
    path_yx = np.asarray(
        [
            (float(row["source_phase_y_px"]), float(row["source_phase_x_px"]))
            for row in paths
        ],
        dtype=np.float64,
    )
    return summary, path_yx


def reconstruct_phase(report: dict, summary: dict[str, str]):
    """Recreate the exact independently stabilized phase used by v18."""

    movie = Path(report["input_movie"])
    fps, frame_count, _, _ = movie_metadata(movie)
    source_start, source_end = report["source_window"]
    if summary["source_phase"] == "B":
        source_start += int(report["phase_offset_source_frames"])
    source_frames = source_frame_indices(
        frame_count,
        fps,
        int(source_start),
        int(source_end),
        float(report["analysis_interval_s"]),
        report.get("source_frame_interval_seconds"),
    )
    width = int(report["configuration"]["width"])
    frames = load_movie_samples(movie, source_frames, width)
    aligned, _, registration_response = stabilize_translations(frames)
    del frames
    config = AtlasConfig(**report["configuration"])
    grains = detect_grain_anchors(aligned, config.warmup_samples, config)
    grain_id = int(summary["source_grain_id"])
    matches = [grain for grain in grains if grain.grain_id == grain_id]
    if len(matches) != 1:
        raise ValueError(f"Could not reconstruct source-phase pollen {grain_id}")
    return movie, source_frames, aligned, registration_response, matches[0], grains


def fixed_pollen_crops(
    aligned: np.ndarray,
    sample_indices: np.ndarray,
    grain,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Translate each sampled frame into one pollen-fixed crop."""

    y0, y1, x0, x1 = bounds
    reference_xy = np.asarray(grain.center_xy, dtype=np.float64)
    crops = np.empty((len(sample_indices), y1 - y0, x1 - x0), dtype=np.uint8)
    height, width = aligned.shape[1:]
    for output_index, sample_index in enumerate(sample_indices):
        center_xy = grain.center_at(int(sample_index))
        displacement_xy = center_xy - reference_xy
        matrix = np.asarray(
            [
                [1.0, 0.0, -displacement_xy[0]],
                [0.0, 1.0, -displacement_xy[1]],
            ],
            dtype=np.float32,
        )
        registered = cv.warpAffine(
            aligned[int(sample_index)],
            matrix,
            (width, height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        crops[output_index] = registered[y0:y1, x0:x1]
    return crops


def foreign_grain_occupancy(
    grains,
    selected_grain,
    sample_indices: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Map image regions biologically owned by pollen other than the source."""

    y0, y1, x0, x1 = bounds
    shape = (y1 - y0, x1 - x0)
    occupancy = np.zeros(shape, dtype=np.float32)
    reference_xy = np.asarray(selected_grain.center_xy, dtype=np.float64)
    for sample_index in sample_indices:
        selected_displacement = (
            selected_grain.center_at(int(sample_index)) - reference_xy
        )
        frame_mask = np.zeros(shape, dtype=np.uint8)
        for grain in grains:
            if grain.grain_id == selected_grain.grain_id:
                continue
            center_xy = grain.center_at(int(sample_index)) - selected_displacement
            local_xy = center_xy - np.asarray((x0, y0), dtype=np.float64)
            radius = max(3, int(round(0.80 * grain.radius_px)))
            cv.circle(
                frame_mask,
                tuple(np.rint(local_xy).astype(int)),
                radius,
                1,
                -1,
            )
        occupancy += frame_mask
    return occupancy / max(1, len(sample_indices))


def candidate_bounds(
    path_yx: np.ndarray,
    image_shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    """Choose a fixed audit crop around the candidate with room for alternatives."""

    height, width = image_shape
    y0 = max(0, int(math.floor(np.min(path_yx[:, 0]))) - margin)
    y1 = min(height, int(math.ceil(np.max(path_yx[:, 0]))) + margin + 1)
    x0 = max(0, int(math.floor(np.min(path_yx[:, 1]))) - margin)
    x1 = min(width, int(math.ceil(np.max(path_yx[:, 1]))) + margin + 1)
    return y0, y1, x0, x1


def build_orientation_history(crops: np.ndarray) -> np.ndarray:
    """Compute compact direction-specific paired-wall evidence for each crop."""

    scores = []
    for index, crop in enumerate(crops):
        score = paired_wall_orientation_score(crop)
        scores.append(np.rint(255.0 * score).astype(np.uint8))
        if (index + 1) % 20 == 0 or index + 1 == len(crops):
            print(
                f"[v20] orientation evidence {index + 1}/{len(crops)}",
                flush=True,
            )
    return np.asarray(scores, dtype=np.uint8)


def trace_timeline(
    orientation_history: np.ndarray,
    evidence_source_frames: np.ndarray,
    output_indices: np.ndarray,
    path_yx: np.ndarray,
    bounds: tuple[int, int, int, int],
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    onset_source_frame: int,
    foreign_occupancy: np.ndarray | None = None,
    trust_legacy_path: bool = False,
):
    """Trace sampled time points with causal age and persistent material prefixes."""

    y0, _, x0, _ = bounds
    local_path = path_yx - np.asarray((y0, x0), dtype=np.float64)
    local_center = pollen_center_yx - np.asarray((y0, x0), dtype=np.float64)
    legacy_onset_index = int(
        np.searchsorted(evidence_source_frames, onset_source_frame)
    )
    warmup = max(
        4,
        min(
            max(4, legacy_onset_index // 2),
            max(4, len(orientation_history) // 10),
            len(orientation_history) - 1,
        ),
    )
    normalized_history = orientation_history.astype(np.float32) / 255.0
    births = persistent_orientation_birth(
        normalized_history,
        warmup_samples=warmup,
        persistence_samples=3,
        minimum_change=0.08,
    )
    persistent_score = persistent_orientation_score(
        normalized_history,
        warmup_samples=warmup,
    )
    if foreign_occupancy is not None:
        occupied = np.asarray(foreign_occupancy, dtype=np.float32) >= 0.20
        persistent_score[:, occupied] = 0.0
        births[:, occupied] = 0.0
    earliest_new_material = float(max(2, warmup - 1))
    trace_config = OrientationTraceConfig(
        maximum_length_px=max(80.0, 1.8 * _path_length(local_path)),
        maximum_extension_px=18.0,
        maximum_gap_steps=9,
        curvature_penalty=0.25,
    )
    proposals = list(propose_pollen_roots(
        persistent_score,
        local_center,
        pollen_radius_px,
        birth_time=births,
        minimum_material_birth=earliest_new_material,
    ))
    if not proposals:
        raise RuntimeError("No causally valid tube emergence direction was found")
    proposal_priors: list[np.ndarray | None] = [None] * len(proposals)
    if trust_legacy_path:
        direction = local_path[min(5, len(local_path) - 1)] - local_path[0]
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        direction_angle = math.atan2(float(direction[0]), float(direction[1]))
        attachment = local_path[0] - local_center
        attachment_angle = math.atan2(float(attachment[0]), float(attachment[1]))
        orientation_bin = int(
            round((direction_angle % math.pi) * persistent_score.shape[0] / math.pi)
        ) % persistent_score.shape[0]
        point = np.rint(local_path[0]).astype(int)
        y = int(np.clip(point[0], 0, persistent_score.shape[1] - 1))
        x = int(np.clip(point[1], 0, persistent_score.shape[2] - 1))
        proposals.append(
            PollenRootProposal(
                root_yx=local_path[0].copy(),
                direction_yx=direction,
                attachment_angle_radians=attachment_angle,
                direction_angle_radians=direction_angle,
                support=float(persistent_score[orientation_bin, y, x]),
                birth_time=float(births[orientation_bin, y, x]),
            )
        )
        proposal_priors.append(local_path)
    proposal_floors = [
        max(
            earliest_new_material,
            proposal.birth_time - 2.0
            if (
                np.isfinite(proposal.birth_time)
                and proposal.birth_time < len(orientation_history) - 1
            )
            else earliest_new_material,
        )
        for proposal in proposals
    ]
    proposal_traces = [
        trace_orientation_lifted(
            persistent_score,
            proposal.root_yx,
            proposal.direction_yx,
            prior_curve_yx=proposal_prior,
            birth_time=births,
            minimum_material_birth=proposal_floor,
            config=(
                replace(
                    trace_config,
                    minimum_length_px=max(4.0, 0.90 * _path_length(local_path)),
                    prior_position_penalty=0.035,
                    prior_direction_penalty=0.03,
                )
                if proposal_prior is not None
                else trace_config
            ),
        )
        for proposal, proposal_floor, proposal_prior in zip(
            proposals,
            proposal_floors,
            proposal_priors,
        )
    ]
    winner_index = int(
        np.argmax(
            [
                trace.score + 5.0 * proposal.support
                for proposal, trace in zip(proposals, proposal_traces)
            ]
        )
    )
    if trust_legacy_path:
        legacy_index = len(proposals) - 1
        legacy_trace = proposal_traces[legacy_index]
        legacy_valid = (
            legacy_trace.length_px >= 0.80 * _path_length(local_path)
            and legacy_trace.supported_fraction >= 0.65
            and legacy_trace.mean_support >= 0.12
        )
        if legacy_valid:
            winner_index = legacy_index
    winner = proposals[winner_index]
    material_floor = float(proposal_floors[winner_index])
    onset_index = int(
        np.clip(
            round(winner.birth_time)
            if np.isfinite(winner.birth_time)
            else legacy_onset_index,
            warmup,
            len(orientation_history) - 1,
        )
    )
    root = winner.root_yx
    direction = winner.direction_yx
    final_trace = proposal_traces[winner_index]
    profiles = path_orientation_profiles(
        orientation_history,
        final_trace.path_yx,
        final_trace.direction_radians,
        warmup,
    )
    front = visible_path_front(
        profiles,
        _curve_arclength(final_trace.path_yx),
        warmup_samples=warmup,
        max_step_px=6.0,
        coverage_cost=0.035,
        absence_weight=0.20,
        motion_penalty=0.12,
        support_window_samples=3,
        require_final_endpoint=True,
    )
    if not front.feasible:
        raise RuntimeError(f"Global orientation front failed: {front.reason}")
    traces = []
    required_lengths = []
    for output_index in output_indices:
        front_index = int(front.front_indices[output_index])
        if front_index < 2:
            traces.append(None)
            required_lengths.append(None)
            continue
        prior = final_trace.path_yx[: front_index + 1]
        target_length = _path_length(prior)
        required_lengths.append(target_length)
        frame_config = replace(
            trace_config,
            minimum_length_px=max(4.0, target_length),
            maximum_length_px=target_length + 1.5,
            maximum_extension_px=2.0,
        )
        result = trace_orientation_lifted(
            orientation_history[output_index].astype(np.float32) / 255.0,
            root,
            direction,
            prior_curve_yx=prior,
            birth_time=births,
            minimum_material_birth=material_floor,
            config=frame_config,
        )
        traces.append(result)
    return (
        traces,
        births,
        local_path,
        onset_index,
        material_floor,
        winner,
        proposals,
        proposal_traces,
        local_center,
        front,
        required_lengths,
    )


def _curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative Euclidean arclength for one ordered path."""

    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path_yx, axis=0), axis=1)))
    )


def path_orientation_profiles(
    orientation_history: np.ndarray,
    path_yx: np.ndarray,
    direction_radians: np.ndarray,
    warmup_samples: int,
) -> np.ndarray:
    """Measure appearance change along one immutable lifted material curve."""

    scores = np.asarray(orientation_history, dtype=np.float32) / 255.0
    points = np.asarray(path_yx, dtype=np.float32)
    direction_bins = np.rint(
        (np.asarray(direction_radians) % math.pi) * scores.shape[1] / math.pi
    ).astype(int) % scores.shape[1]
    values = np.empty((len(scores), len(points)), dtype=np.float32)
    for sample in range(len(scores)):
        for orientation in np.unique(direction_bins):
            indices = np.flatnonzero(direction_bins == orientation)
            sampled = cv.remap(
                scores[sample, orientation],
                points[indices, 1][None, :],
                points[indices, 0][None, :],
                interpolation=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0.0,
            )
            values[sample, indices] = sampled[0]
    baseline = np.median(values[:warmup_samples], axis=0)
    noise = 1.4826 * np.median(
        np.abs(values[:warmup_samples] - baseline[None, :]),
        axis=0,
    )
    threshold = baseline + np.maximum(0.035, 3.0 * noise)
    scale = np.maximum(0.08, np.percentile(values, 90, axis=0) - threshold)
    profiles = np.tanh((values - threshold[None, :]) / scale[None, :])
    profiles[:warmup_samples] = np.minimum(profiles[:warmup_samples], -0.1)
    return profiles.astype(np.float32)


def _path_length(path_yx: np.ndarray) -> float:
    """Return Euclidean arclength for an ordered path."""

    return float(np.linalg.norm(np.diff(path_yx, axis=0), axis=1).sum())


def _draw_path(image: np.ndarray, path_yx: np.ndarray, color, width=2) -> None:
    """Draw one row-column path and its endpoint on a review image."""

    points = np.rint(path_yx[:, ::-1]).astype(np.int32)
    cv.polylines(image, [points], False, color, width, cv.LINE_AA)
    cv.circle(image, tuple(points[-1]), 4, color, -1, cv.LINE_AA)


def write_review_video(
    output: Path,
    crops: np.ndarray,
    output_indices: np.ndarray,
    source_frames: np.ndarray,
    traces,
    legacy_path_yx: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    onset_source_frame: int,
) -> None:
    """Render raw, legacy-path, and causal orientation views side by side."""

    height, width = crops.shape[1:]
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        6.0,
        (2 * width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create review video: {output}")
    for crop, evidence_index, trace in zip(crops[output_indices], output_indices, traces):
        raw = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
        review = raw.copy()
        _draw_path(review, legacy_path_yx, (255, 190, 0), 1)
        cv.circle(
            review,
            tuple(np.rint(pollen_center_yx[::-1]).astype(int)),
            int(round(pollen_radius_px)),
            (230, 80, 30),
            1,
            cv.LINE_AA,
        )
        if trace is not None:
            _draw_path(review, trace.path_yx, (40, 225, 80), 2)
        source_frame = int(source_frames[evidence_index])
        state = "PRE-GERMINATION" if source_frame < onset_source_frame else "TRACE"
        cv.putText(
            raw,
            f"raw  source {source_frame}",
            (8, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.46,
            (30, 30, 30),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            review,
            f"cyan=v18  green=v20  {state}",
            (8, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        writer.write(np.hstack((raw, review)))
    writer.release()


def write_birth_image(
    output: Path,
    final_crop: np.ndarray,
    births: np.ndarray,
    trace,
) -> None:
    """Visualize directional material age along the final selected trace."""

    image = cv.cvtColor(final_crop, cv.COLOR_GRAY2BGR)
    if trace is not None:
        maximum = max(1.0, float(np.max(births[births < np.max(births)])))
        for point, angle in zip(trace.path_yx, trace.direction_radians):
            direction_bin = int(
                round((angle % math.pi) * births.shape[0] / math.pi)
            ) % births.shape[0]
            y, x = np.rint(point).astype(int)
            if 0 <= y < births.shape[1] and 0 <= x < births.shape[2]:
                age = float(np.clip(births[direction_bin, y, x] / maximum, 0.0, 1.0))
                color = (int(255 * (1.0 - age)), int(220 * age), 40)
                cv.circle(image, (x, y), 2, color, -1, cv.LINE_AA)
    cv.imwrite(str(output), image)


def write_measurements(
    output: Path,
    output_indices: np.ndarray,
    source_frames: np.ndarray,
    traces,
    required_lengths,
    bounds: tuple[int, int, int, int],
) -> dict[str, float | int | None]:
    """Export each sampled centerline measurement and summarize its dynamics."""

    y0, _, x0, _ = bounds
    rows = []
    previous_tip = None
    previous_length = None
    tip_steps = []
    length_steps = []
    accepted_count = 0
    for evidence_index, trace, required_length in zip(
        output_indices,
        traces,
        required_lengths,
    ):
        source_frame = int(source_frames[evidence_index])
        if trace is None:
            rows.append(
                {
                    "source_frame": source_frame,
                    "status": "pre-germination",
                    "worldsheet_length_px": "",
                    "length_px": "",
                    "tip_x_px": "",
                    "tip_y_px": "",
                    "mean_directional_support": "",
                    "supported_fraction": "",
                    "prior_prefix_error_px": "",
                    "tip_step_px": "",
                    "length_step_px": "",
                }
            )
            continue
        reaches_worldsheet = (
            required_length is not None
            and trace.length_px >= required_length - 0.5
        )
        accepted = (
            reaches_worldsheet
            and trace.supported_fraction >= 0.65
            and trace.mean_support >= 0.12
        )
        tip_yx = trace.path_yx[-1] + np.asarray((y0, x0), dtype=np.float64)
        tip_step = (
            float(np.linalg.norm(tip_yx - previous_tip))
            if previous_tip is not None
            else 0.0
        )
        length_step = (
            float(trace.length_px - previous_length)
            if previous_length is not None
            else 0.0
        )
        if accepted:
            accepted_count += 1
            if previous_tip is not None:
                tip_steps.append(tip_step)
                length_steps.append(length_step)
            previous_tip = tip_yx
            previous_length = trace.length_px
        rows.append(
            {
                "source_frame": source_frame,
                "status": "accepted" if accepted else "review",
                "worldsheet_length_px": round(required_length, 4),
                "length_px": round(trace.length_px, 4),
                "tip_x_px": round(float(tip_yx[1]), 4),
                "tip_y_px": round(float(tip_yx[0]), 4),
                "mean_directional_support": round(trace.mean_support, 6),
                "supported_fraction": round(trace.supported_fraction, 6),
                "prior_prefix_error_px": round(trace.prior_prefix_error_px, 4),
                "tip_step_px": round(tip_step, 4),
                "length_step_px": round(length_step, 4),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "accepted_measurement_count": accepted_count,
        "tip_step_median_px": float(np.median(tip_steps)) if tip_steps else None,
        "tip_step_max_px": float(np.max(tip_steps)) if tip_steps else None,
        "length_step_median_px": float(np.median(length_steps))
        if length_steps
        else None,
        "length_step_max_px": float(np.max(np.abs(length_steps)))
        if length_steps
        else None,
        "length_decrease_count": int(np.sum(np.asarray(length_steps) < -1.0))
        if length_steps
        else 0,
    }


def main() -> None:
    """Run one real candidate audit and write reviewable media and metrics."""

    args = parse_args()
    report = json.loads((args.v18_run / "report.json").read_text(encoding="utf-8"))
    summary, path_yx = load_candidate(args.v18_run, args.consensus_id)
    movie, phase_frames, aligned, responses, grain, grains = reconstruct_phase(
        report,
        summary,
    )
    evidence_count = min(max(12, args.evidence_samples), len(phase_frames))
    evidence_indices = np.unique(
        np.rint(np.linspace(0, len(phase_frames) - 1, evidence_count)).astype(int)
    )
    bounds = candidate_bounds(path_yx, aligned.shape[1:], args.crop_margin_px)
    crops = fixed_pollen_crops(aligned, evidence_indices, grain, bounds)
    foreign_occupancy = foreign_grain_occupancy(
        grains,
        grain,
        evidence_indices,
        bounds,
    )
    del aligned
    orientation_history = build_orientation_history(crops)
    output_count = min(max(2, args.output_samples), len(evidence_indices))
    output_indices = np.unique(
        np.rint(np.linspace(0, len(evidence_indices) - 1, output_count)).astype(int)
    )
    onset_candidates = [
        summary.get("source_root_onset_source_frame", ""),
        summary.get("source_rim_onset_source_frame", ""),
    ]
    onset_values = [int(value) for value in onset_candidates if value]
    if not onset_values:
        raise ValueError("Candidate has no source-phase onset bound for causal tracing")
    onset_source_frame = min(onset_values)
    evidence_source_frames = phase_frames[evidence_indices]
    (
        traces,
        births,
        local_path,
        onset_index,
        material_floor,
        root_proposal,
        root_proposals,
        proposal_traces,
        local_center,
        global_front,
        required_lengths,
    ) = trace_timeline(
        orientation_history,
        evidence_source_frames,
        output_indices,
        path_yx,
        bounds,
        grain.center_xy[::-1],
        float(grain.radius_px),
        onset_source_frame,
        foreign_occupancy,
        summary["consensus_measurement_supported"] == "True",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "orientation_worldsheet_review.mp4"
    write_review_video(
        video_path,
        crops,
        output_indices,
        evidence_source_frames,
        traces,
        local_path,
        local_center,
        float(grain.radius_px),
        onset_source_frame,
    )
    final_trace = next((trace for trace in reversed(traces) if trace is not None), None)
    birth_image = args.output_dir / "orientation_birth_trace.jpg"
    write_birth_image(birth_image, crops[-1], births, final_trace)
    measurements_path = args.output_dir / "orientation_measurements.csv"
    dynamics = write_measurements(
        measurements_path,
        output_indices,
        evidence_source_frames,
        traces,
        required_lengths,
        bounds,
    )
    result = {
        "prototype": "v20_causal_orientation_worldsheet",
        "input_movie": str(movie),
        "v18_run": str(args.v18_run),
        "consensus_id": args.consensus_id,
        "source_phase": summary["source_phase"],
        "source_grain_id": int(summary["source_grain_id"]),
        "source_event_id": int(summary["source_event_id"]),
        "v18_quality_flags": summary["source_quality_flags"],
        "v18_measurement_supported": summary["consensus_measurement_supported"],
        "v18_path_length_px": _path_length(local_path),
        "v20_final_length_px": None if final_trace is None else final_trace.length_px,
        "v20_final_mean_support": None
        if final_trace is None
        else final_trace.mean_support,
        "v20_final_supported_fraction": None
        if final_trace is None
        else final_trace.supported_fraction,
        "v20_final_prior_prefix_error_px": None
        if final_trace is None
        else final_trace.prior_prefix_error_px,
        "onset_source_frame": onset_source_frame,
        "onset_evidence_index": onset_index,
        "minimum_material_birth_index": material_floor,
        "selected_root_yx": root_proposal.root_yx.tolist(),
        "selected_root_direction_yx": root_proposal.direction_yx.tolist(),
        "selected_root_support": root_proposal.support,
        "root_proposals": [
            {
                "root_yx": proposal.root_yx.tolist(),
                "direction_yx": proposal.direction_yx.tolist(),
                "support": proposal.support,
                "birth_time": proposal.birth_time,
                "final_length_px": trace.length_px,
                "final_score": trace.score,
                "final_mean_support": trace.mean_support,
                "final_supported_fraction": trace.supported_fraction,
            }
            for proposal, trace in zip(root_proposals, proposal_traces)
        ],
        "global_front_reason": global_front.reason,
        "global_front_direct_support_fraction": global_front.direct_support_fraction,
        "global_front_eventual_support_fraction": (
            global_front.eventual_support_fraction
        ),
        "evidence_sample_count": len(evidence_indices),
        "output_sample_count": len(output_indices),
        "trajectory_dynamics": dynamics,
        "registration_response_median": float(np.median(responses)),
        "artifacts": {
            "review_video": str(video_path),
            "birth_trace_image": str(birth_image),
            "measurements_csv": str(measurements_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
