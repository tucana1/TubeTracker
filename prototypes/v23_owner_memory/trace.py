"""Trace one v23 pollen owner with delayed, owner-conditioned path selection.

The identity report supplies only independently linked pollen observations.  No
legacy tube centerline, crop, direction, or final path is consumed.  Multiple
paired-wall paths are generated on owner-aligned raw frames and retained until
the complete sampled sequence can resolve their material ancestry.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from tubetracker.orientation_worldsheet import (
    CoupledRibbonTraceConfig,
    OrientationScoreConfig,
    aggregate_paired_wall_history,
    paired_wall_orientation_features,
    persistent_orientation_birth,
    propose_pollen_roots,
    trace_coupled_ribbon_lifted,
)
from tubetracker.owner_memory import (
    OwnerMemoryConfig,
    TubePathObservation,
    resolve_owner_hypotheses,
)


def parse_args() -> argparse.Namespace:
    """Parse one independent owner-tracing experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-position", type=int, default=3)
    parser.add_argument("--owner-radius", type=float, default=15.0)
    parser.add_argument(
        "--crop-radius",
        type=int,
        default=0,
        help="Symmetric owner crop radius; zero preserves the original C10 audit crop.",
    )
    parser.add_argument("--minimum-net-growth", type=float, default=8.0)
    return parser.parse_args()


def load_frames(movie: Path, source_frames: list[int]) -> list[np.ndarray]:
    """Load exact source frames requested by the identity audit."""
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    frames = []
    for source_frame in source_frames:
        capture.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not read source frame {source_frame}")
        frames.append(frame)
    capture.release()
    return frames


def owner_aligned_crops(
    frames: list[np.ndarray],
    centers_yx: np.ndarray,
    crop_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Translate raw frames into one owner-relative coordinate system and crop."""
    reference = np.median(centers_yx, axis=0)
    height, width = frames[0].shape[:2]
    if crop_radius > 0:
        y0 = max(0, int(math.floor(reference[0] - crop_radius)))
        y1 = min(height, int(math.ceil(reference[0] + crop_radius + 1)))
        x0 = max(0, int(math.floor(reference[1] - crop_radius)))
        x1 = min(width, int(math.ceil(reference[1] + crop_radius + 1)))
    else:
        y0 = max(0, int(math.floor(reference[0] - 70)))
        y1 = min(height, int(math.ceil(reference[0] + 250)))
        x0 = max(0, int(math.floor(reference[1] - 180)))
        x1 = min(width, int(math.ceil(reference[1] + 210)))
    crops = []
    transforms = []
    for frame, center in zip(frames, centers_yx):
        shift_yx = reference - center
        transform = np.asarray(
            [[1.0, 0.0, shift_yx[1]], [0.0, 1.0, shift_yx[0]]],
            dtype=np.float32,
        )
        aligned = cv.warpAffine(
            frame,
            transform,
            (width, height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        crops.append(cv.cvtColor(aligned[y0:y1, x0:x1], cv.COLOR_BGR2GRAY))
        transforms.append(transform)
    owner_local = reference - np.asarray((y0, x0), dtype=np.float64)
    return np.asarray(crops), owner_local, (y0, y1, x0, x1), np.asarray(transforms)


def foreign_pollen_history(
    identity: dict,
    owner_track_id: int,
    source_frames: list[int],
    bounds: tuple[int, int, int, int],
    transforms: np.ndarray,
    shape: tuple[int, int],
    radius_px: float,
) -> np.ndarray:
    """Rasterize learned foreign pollen identities in owner-aligned coordinates."""
    y0, _, x0, _ = bounds
    occupancy = np.zeros((len(source_frames), *shape), dtype=np.uint8)
    for track in identity["tracks"]:
        if track["track_id"] == owner_track_id or track["semantic_observation_count"] < 1:
            continue
        track_frames = np.asarray(track["source_frames"], dtype=np.float64)
        track_centers = np.asarray(track["centers_yx"], dtype=np.float64)
        starts_at_movie_origin = int(track_frames[0]) == int(source_frames[0])
        for position, source_frame in enumerate(source_frames):
            if source_frame < track_frames[0]:
                continue
            if source_frame > track_frames[-1] and not starts_at_movie_origin:
                continue
            center = np.asarray(
                [
                    np.interp(source_frame, track_frames, track_centers[:, axis])
                    for axis in range(2)
                ]
            )
            aligned_y = center[0] + transforms[position, 1, 2] - y0
            aligned_x = center[1] + transforms[position, 0, 2] - x0
            cv.circle(
                occupancy[position],
                (round(aligned_x), round(aligned_y)),
                round(radius_px),
                1,
                -1,
            )
    return occupancy.astype(bool)


def build_orientation_history(
    crops: np.ndarray,
    foreign_occupancy: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Compute compact paired-wall feature histories for owner-aligned frames."""
    config = OrientationScoreConfig(
        orientation_count=24,
        half_widths_px=(2.0, 3.0, 4.0, 5.0, 6.0),
        tangent_samples_px=(-3.0, -1.5, 0.0, 1.5, 3.0),
        wall_sigma_px=1.0,
        background_sigma_px=6.0,
        structure_sigma_px=1.8,
    )
    histories = [[], [], [], []]
    for position, crop in enumerate(crops):
        features = paired_wall_orientation_features(crop, config)
        values = [
            features.score.copy(),
            features.paired_score.copy(),
            features.half_width_px.copy(),
            features.wall_balance.copy(),
        ]
        for value in values:
            value[:, foreign_occupancy[position]] = 0.0
        for history, value in zip(histories, values):
            history.append(value.astype(np.float16))
        print(f"[v23] paired-wall frame {position + 1}/{len(crops)}", flush=True)
    return tuple(np.asarray(history, dtype=np.float32) for history in histories)


def generate_path_candidates(
    histories: tuple[np.ndarray, ...],
    owner_yx: np.ndarray,
    owner_radius: float,
    source_frames: list[int],
    start_position: int,
    owner_track_id: int,
) -> list[list[TubePathObservation]]:
    """Generate independent full-path proposals at each sampled growth time."""
    score, paired, width, balance = histories
    birth = persistent_orientation_birth(
        score,
        warmup_samples=max(2, start_position - 1),
        persistence_samples=2,
        minimum_change=0.05,
    )
    preexisting = np.percentile(np.max(score[: max(2, start_position - 1)], axis=1), 75.0, axis=0)
    trace_config = CoupledRibbonTraceConfig(
        step_px=2.0,
        maximum_turn_bins=1,
        curvature_penalty=0.15,
        maximum_gap_steps=4,
        maximum_initial_gap_steps=4,
        root_occlusion_px=1.15 * owner_radius,
        beam_width=260,
        minimum_length_px=18.0,
        maximum_length_px=210.0,
        minimum_endpoint_separation_fraction=0.45,
        maximum_reacquisition_width_change_px=3.0,
    )
    short_trace_config = CoupledRibbonTraceConfig(
        step_px=1.5,
        maximum_turn_bins=1,
        curvature_penalty=0.16,
        maximum_gap_steps=3,
        maximum_initial_gap_steps=2,
        root_occlusion_px=2.0,
        beam_width=260,
        minimum_length_px=6.0,
        maximum_length_px=54.0,
        minimum_endpoint_separation_fraction=0.45,
        maximum_reacquisition_width_change_px=2.5,
    )
    frame_candidates = []
    candidate_id = 1
    for position in range(start_position, len(source_frames)):
        aggregate = aggregate_paired_wall_history(
            score[: position + 1],
            paired[: position + 1],
            width[: position + 1],
            balance[: position + 1],
            start_index=max(start_position - 1, position - 2),
            percentile=70.0,
        )
        evidence = np.maximum(aggregate.paired_score, 0.15 * aggregate.score)
        proposals = propose_pollen_roots(
            evidence,
            owner_yx,
            owner_radius,
            birth_time=birth,
            minimum_material_birth=1.0,
            attachment_count=72,
            direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
            probe_distances_px=(3.0, 6.0, 9.0, 12.0, 15.0),
            probe_top_k=5,
            maximum_proposals=24,
            minimum_angle_separation_degrees=7.0,
        )
        candidates = []
        for proposal in proposals:
            long_trace = trace_coupled_ribbon_lifted(
                aggregate,
                proposal.root_yx,
                proposal.direction_yx,
                birth_time=birth,
                minimum_material_birth=1.0,
                config=trace_config,
            )
            traces = [(long_trace, trace_config)]
            if long_trace.paired_supported_fraction < 0.20:
                traces.append(
                    (
                        trace_coupled_ribbon_lifted(
                            aggregate,
                            proposal.root_yx,
                            proposal.direction_yx,
                            birth_time=birth,
                            minimum_material_birth=1.0,
                            config=short_trace_config,
                        ),
                        short_trace_config,
                    )
                )
            for trace, active_config in traces:
                observation = _trace_observation(
                    trace,
                    active_config,
                    preexisting,
                    owner_yx,
                    owner_radius,
                    position,
                    candidate_id,
                    owner_track_id,
                )
                candidate_id += 1
                if observation is not None:
                    candidates.append(observation)
        candidates.sort(key=lambda item: _candidate_rank(item), reverse=True)
        frame_candidates.append(candidates[:12])
        print(f"[v23] source {source_frames[position]}: {len(candidates)} path hypotheses", flush=True)
    return frame_candidates


def _trace_observation(
    trace,
    config: CoupledRibbonTraceConfig,
    preexisting: np.ndarray,
    owner_yx: np.ndarray,
    owner_radius: float,
    frame_index: int,
    candidate_id: int,
    owner_track_id: int,
) -> TubePathObservation | None:
    """Convert one long- or short-range ribbon trace into owner evidence."""
    if trace.length_px < config.minimum_length_px - 1.0:
        return None
    endpoint_radius = float(np.linalg.norm(trace.path_yx[-1] - owner_yx))
    if (
        trace.endpoint_separation_fraction
        < config.minimum_endpoint_separation_fraction
        or endpoint_radius < owner_radius + 4.0
    ):
        return None
    arc = _curve_arclength(trace.path_yx)
    evaluated = arc >= config.root_occlusion_px
    old_fraction = (
        float(np.mean(_sample_map(preexisting, trace.path_yx)[evaluated]))
        if np.any(evaluated)
        else 1.0
    )
    birth_values = trace.birth_time[evaluated & np.isfinite(trace.birth_time)]
    if len(birth_values) >= 2:
        forward_fraction = float(np.mean(np.diff(birth_values) >= -1.0))
        available_fraction = float(
            np.mean(birth_values <= frame_index + 0.5)
        )
        causal_score = 0.5 * (forward_fraction + available_fraction)
    else:
        causal_score = 0.0
    root_distance = float(np.linalg.norm(trace.path_yx[0] - owner_yx))
    attachment = float(
        np.clip(
            1.0
            - abs(root_distance - owner_radius) / max(owner_radius, 1.0),
            0.0,
            1.0,
        )
    )
    return TubePathObservation(
        frame_index=frame_index,
        candidate_id=candidate_id,
        owner_track_id=owner_track_id,
        owner_center_yx=tuple(owner_yx),
        points_yx=trace.path_yx,
        appearance_score=trace.paired_supported_fraction,
        causal_growth_score=causal_score,
        owner_attachment_score=attachment,
        preexisting_fraction=old_fraction,
    )


def _candidate_rank(candidate: TubePathObservation) -> float:
    """Rank duplicate same-frame paths before whole-window resolution."""
    return (
        candidate.appearance_score
        + 1.5 * candidate.causal_growth_score
        + candidate.owner_attachment_score
        - 2.0 * candidate.preexisting_fraction
    )


def _curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength for one path."""
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path_yx, axis=0), axis=1))))


def _sample_map(image: np.ndarray, path_yx: np.ndarray) -> np.ndarray:
    """Sample a scalar map along a row-column path."""
    return cv.remap(
        np.asarray(image, dtype=np.float32),
        path_yx[:, 1][None].astype(np.float32),
        path_yx[:, 0][None].astype(np.float32),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )[0]


def render_review(
    crops: np.ndarray,
    source_frames: list[int],
    start_position: int,
    owner_yx: np.ndarray,
    candidates: list[list[TubePathObservation]],
    resolution,
    output: Path,
) -> None:
    """Render the selected lineage and closest competing path on raw crops."""
    selected = (
        {item.frame_index: item for item in resolution.selected.observations}
        if resolution.selected
        else {}
    )
    alternatives = (
        {item.frame_index: item for item in resolution.alternatives[0].observations}
        if resolution.alternatives
        else {}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = crops.shape[1:]
    writer = cv.VideoWriter(str(output), cv.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
    for offset, position in enumerate(range(start_position, len(source_frames))):
        canvas = cv.cvtColor(crops[position], cv.COLOR_GRAY2BGR)
        cv.circle(canvas, (round(owner_yx[1]), round(owner_yx[0])), 15, (255, 120, 0), 2, cv.LINE_AA)
        if offset < len(candidates) and candidates[offset]:
            points = np.rint(candidates[offset][0].points_yx[:, ::-1]).astype(np.int32)
            cv.polylines(canvas, [points], False, (130, 130, 130), 1, cv.LINE_AA)
        if position in alternatives:
            points = np.rint(alternatives[position].points_yx[:, ::-1]).astype(np.int32)
            cv.polylines(canvas, [points], False, (0, 165, 255), 1, cv.LINE_AA)
        if position in selected:
            points = np.rint(selected[position].points_yx[:, ::-1]).astype(np.int32)
            cv.polylines(canvas, [points], False, (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(canvas, tuple(points[-1]), 4, (0, 255, 0), -1, cv.LINE_AA)
            length = selected[position].length_px
        else:
            length = float("nan")
        cv.putText(canvas, f"source {source_frames[position]}  length {length:.1f}px", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 220), 2, cv.LINE_AA)
        cv.putText(canvas, f"owner-memory: {resolution.reason}", (10, 47), cv.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 220), 1, cv.LINE_AA)
        writer.write(canvas)
    writer.release()


def write_selected_centerlines(resolution, source_frames: list[int], output: Path) -> None:
    """Export every selected path point for downstream full-span reviews."""
    fieldnames = (
        "source_frame",
        "candidate_id",
        "point_index",
        "arclength_px",
        "y_aligned_px",
        "x_aligned_px",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if resolution.selected is None:
            return
        for observation in resolution.selected.observations:
            arclength = _curve_arclength(observation.points_yx)
            for point_index, (point, distance) in enumerate(
                zip(observation.points_yx, arclength)
            ):
                writer.writerow(
                    {
                        "source_frame": source_frames[observation.frame_index],
                        "candidate_id": observation.candidate_id,
                        "point_index": point_index,
                        "arclength_px": float(distance),
                        "y_aligned_px": float(point[0]),
                        "x_aligned_px": float(point[1]),
                    }
                )


def main() -> None:
    """Run independent path generation and delayed owner-memory resolution."""
    args = parse_args()
    identity = json.loads(args.identity_report.read_text())
    track = next((item for item in identity["tracks"] if item["track_id"] == args.track_id), None)
    if track is None:
        raise ValueError(f"track {args.track_id} not found")
    source_frames = [int(value) for value in identity["source_frames"]]
    track_frames = np.asarray(track["source_frames"], dtype=np.float64)
    track_centers = np.asarray(track["centers_yx"], dtype=np.float64)
    centers = np.column_stack(
        [
            np.interp(source_frames, track_frames, track_centers[:, axis])
            for axis in range(2)
        ]
    )
    frames = load_frames(Path(identity["movie"]), source_frames)
    crops, owner_yx, bounds, transforms = owner_aligned_crops(
        frames, centers, args.crop_radius
    )
    foreign_occupancy = foreign_pollen_history(
        identity,
        args.track_id,
        source_frames,
        bounds,
        transforms,
        crops.shape[1:],
        1.35 * args.owner_radius,
    )
    histories = build_orientation_history(crops, foreign_occupancy)
    candidates = generate_path_candidates(
        histories,
        owner_yx,
        args.owner_radius,
        source_frames,
        args.start_position,
        args.track_id,
    )
    resolution = resolve_owner_hypotheses(
        candidates,
        OwnerMemoryConfig(
            beam_width=48,
            maximum_prefix_p90_error_px=8.0,
            maximum_retraction_px=6.0,
            minimum_observations=3,
            minimum_score_margin=0.35,
        ),
    )
    selected_lengths = (
        [item.length_px for item in resolution.selected.observations]
        if resolution.selected
        else []
    )
    net_growth_px = (
        selected_lengths[-1] - selected_lengths[0]
        if len(selected_lengths) >= 2
        else 0.0
    )
    if track["semantic_observation_count"] < 2:
        resolution = replace(
            resolution,
            accepted=False,
            reason="insufficient-owner-identity",
        )
    elif (
        resolution.accepted
        and net_growth_px + 1e-6 < args.minimum_net_growth
    ):
        resolution = replace(
            resolution,
            accepted=False,
            reason="insufficient-observed-growth",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    render_review(
        crops,
        source_frames,
        args.start_position,
        owner_yx,
        candidates,
        resolution,
        args.output / "owner_trace_review.mp4",
    )
    write_selected_centerlines(
        resolution,
        source_frames,
        args.output / "owner_centerlines.csv",
    )
    report = {
        "prototype": "v23_owner_conditioned_video_graph",
        "stage": "independent-owner-tube-memory",
        "identity_report": str(args.identity_report),
        "owner_track_id": args.track_id,
        "owner_semantic_observation_count": track["semantic_observation_count"],
        "owner_observation_count": len(track["source_frames"]),
        "owner_center_interpolation": "linear-with-constant-endpoints",
        "source_frames": source_frames[args.start_position :],
        "crop_radius_px": args.crop_radius,
        "crop_bounds_yxyx": list(bounds),
        "candidate_counts": [len(frame) for frame in candidates],
        "accepted": resolution.accepted,
        "reason": resolution.reason,
        "score_margin": resolution.score_margin,
        "selected_candidate_ids": [item.candidate_id for item in resolution.selected.observations] if resolution.selected else [],
        "selected_lengths_px": [item.length_px for item in resolution.selected.observations] if resolution.selected else [],
        "net_growth_px": net_growth_px,
        "minimum_net_growth_px": args.minimum_net_growth,
        "selected_observations": [
            {
                "source_frame": source_frames[item.frame_index],
                "candidate_id": item.candidate_id,
                "length_px": item.length_px,
                "appearance_score": item.appearance_score,
                "causal_growth_score": item.causal_growth_score,
                "owner_attachment_score": item.owner_attachment_score,
                "preexisting_fraction": item.preexisting_fraction,
                "foreign_owner_score": item.foreign_owner_score,
            }
            for item in resolution.selected.observations
        ]
        if resolution.selected
        else [],
        "maximum_prefix_error_p90_px": resolution.selected.prefix_error_p90_px if resolution.selected else None,
        "artifacts": {
            "review_video": "owner_trace_review.mp4",
            "centerlines": "owner_centerlines.csv",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
