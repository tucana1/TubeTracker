"""Trace all pollen tubes through one movie-wide orientation-and-birth atlas.

The prototype integrates weak paired-wall evidence across the complete movie
before choosing any owner path.  Crossings remain separated by orientation and
appearance time, and every pollen detected at the first frame is retained even
when sparse identity linking later fragments it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess

import cv2 as cv
import numpy as np

from tubetracker.causal_birth_forest import (
    CausalBirthAtlas,
    CausalBirthAtlasConfig,
    birth_order_score,
    build_causal_birth_atlas,
    monotone_path_births,
    track_pollen_centers,
)
from tubetracker.orientation_worldsheet import (
    CoupledRibbonTraceConfig,
    PairedWallOrientationResult,
    OrientationScoreConfig,
    paired_wall_orientation_features,
    propose_pollen_roots,
    trace_coupled_ribbon_lifted,
)


@dataclass(frozen=True)
class OwnerSeed:
    """Describe one frame-zero pollen retained as a possible tube owner."""

    track_id: int
    center_yx: np.ndarray
    semantic_observations: int


@dataclass(frozen=True)
class OwnerPath:
    """Store one selected causal path and its validation measurements."""

    owner_index: int
    owner_track_id: int
    path_yx: np.ndarray
    path_birth_samples: np.ndarray
    score: float
    length_px: float
    paired_fraction: float
    birth_order_fraction: float
    persistent_fraction: float
    preexisting_fraction: float
    foreign_pollen_fraction: float
    owner_halo_fraction: float
    radial_extension_px: float
    root_outward_cosine: float
    accepted: bool
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse movie, identity, sampling, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument("--warmup-samples", type=int, default=8)
    parser.add_argument("--tail-samples", type=int, default=24)
    parser.add_argument("--maximum-proposals", type=int, default=8)
    parser.add_argument(
        "--track-ids",
        help="Optional comma-separated owner IDs for a focused trace audit.",
    )
    return parser.parse_args()


def movie_schedule(
    movie: Path,
    interval_s: float,
) -> tuple[np.ndarray, float, int, int, int]:
    """Return evenly spaced source frames and native movie metadata."""

    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    fps = float(capture.get(cv.CAP_PROP_FPS))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0.0 or frame_count < 2 or interval_s <= 0.0:
        raise ValueError("movie metadata or sample interval is invalid")
    step = max(1, round(fps * interval_s))
    frames = np.arange(0, frame_count, step, dtype=np.int32)
    if frames[-1] != frame_count - 1:
        frames = np.append(frames, frame_count - 1)
    return frames, fps, frame_count, width, height


def load_grayscale_samples(
    movie: Path,
    source_frames: np.ndarray,
    width: int,
    native_width: int,
    native_height: int,
) -> np.ndarray:
    """Decode exact source frames once at the atlas analysis scale."""

    height = round(native_height * width / native_width)
    frames = np.empty((len(source_frames), height, width), dtype=np.uint8)
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[sample] = cv.resize(
            gray,
            (width, height),
            interpolation=cv.INTER_AREA,
        )
        if sample == 0 or (sample + 1) % 50 == 0:
            print(f"[v24 decode] {sample + 1}/{len(source_frames)}", flush=True)
    capture.release()
    return frames


def stabilize_translations(
    frames: np.ndarray,
    maximum_step_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove trustworthy common field drift while preserving local motion."""

    aligned = np.empty_like(frames)
    aligned[0] = frames[0]
    shifts_xy = np.zeros((len(frames), 2), dtype=np.float32)
    responses = np.ones(len(frames), dtype=np.float32)
    height, width = frames.shape[1:]
    hann = cv.createHanningWindow((width, height), cv.CV_32F)

    def registration_image(frame: np.ndarray) -> np.ndarray:
        smooth = cv.GaussianBlur(frame, (0, 0), 2.0)
        return cv.Laplacian(smooth, cv.CV_32F)

    previous = registration_image(frames[0])
    cumulative = np.zeros(2, dtype=np.float64)
    for sample in range(1, len(frames)):
        current = registration_image(frames[sample])
        delta, response = cv.phaseCorrelate(previous, current, hann)
        delta = np.asarray(delta, dtype=np.float64)
        if response >= 0.08 and np.linalg.norm(delta) <= maximum_step_px:
            cumulative += delta
        shifts_xy[sample] = cumulative
        responses[sample] = float(response)
        transform = np.asarray(
            [[1.0, 0.0, -cumulative[0]], [0.0, 1.0, -cumulative[1]]],
            dtype=np.float32,
        )
        aligned[sample] = cv.warpAffine(
            frames[sample],
            transform,
            (width, height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        previous = current
    return aligned, shifts_xy, responses


def load_owner_seeds(
    identity_report: Path,
    analysis_scale: float,
) -> tuple[list[OwnerSeed], dict]:
    """Retain every learned frame-zero pollen, including fragmented tracks."""

    identity = json.loads(identity_report.read_text())
    origin = int(identity["source_frames"][0])
    seeds = []
    for track in identity["tracks"]:
        if (
            int(track["source_frames"][0]) != origin
            or int(track["semantic_observation_count"]) < 1
        ):
            continue
        seeds.append(
            OwnerSeed(
                track_id=int(track["track_id"]),
                center_yx=np.asarray(track["centers_yx"][0], dtype=np.float64)
                * analysis_scale,
                semantic_observations=int(track["semantic_observation_count"]),
            )
        )
    seeds.sort(key=lambda item: item.track_id)
    if not seeds:
        raise RuntimeError("identity report contains no frame-zero pollen")
    return seeds, identity


def orientation_feature_stream(
    frames: np.ndarray,
    config: OrientationScoreConfig,
):
    """Yield paired-wall features without storing a time-by-image tensor."""

    for sample, frame in enumerate(frames):
        yield paired_wall_orientation_features(frame, config)
        if sample == 0 or (sample + 1) % 25 == 0:
            print(f"[v24 atlas] {sample + 1}/{len(frames)}", flush=True)


def pollen_swept_mask(
    center_tracks_yx: np.ndarray,
    excluded_owner: int,
    shape: tuple[int, int],
    radius_px: float,
    tail_samples: int,
) -> np.ndarray:
    """Mask foreign pollen positions represented by the late fused atlas."""

    mask = np.zeros(shape, dtype=np.uint8)
    for owner in range(len(center_tracks_yx)):
        if owner == excluded_owner:
            continue
        start = max(0, center_tracks_yx.shape[1] - tail_samples)
        for sample in range(start, center_tracks_yx.shape[1], 2):
            y, x = center_tracks_yx[owner, sample]
            cv.circle(mask, (round(float(x)), round(float(y))), round(radius_px), 1, -1)
    return mask.astype(bool)


def masked_atlas(
    atlas: CausalBirthAtlas,
    foreign_mask: np.ndarray,
) -> tuple[PairedWallOrientationResult, np.ndarray]:
    """Exclude pollen bodies while retaining overlapping tube layers."""

    values = []
    for source in (
        atlas.aggregate.score,
        atlas.aggregate.paired_score,
        atlas.aggregate.half_width_px,
        atlas.aggregate.wall_balance,
    ):
        value = np.asarray(source, dtype=np.float32).copy()
        value[:, foreign_mask] = 0.0
        values.append(value)
    birth = atlas.birth_sample.copy()
    birth[:, foreign_mask] = 0
    return PairedWallOrientationResult(*values), birth


def trace_owner(
    owner_index: int,
    seed: OwnerSeed,
    owner_radius_px: float,
    atlas: CausalBirthAtlas,
    center_tracks_yx: np.ndarray,
    maximum_proposals: int,
    tail_samples: int,
) -> OwnerPath | None:
    """Select the strongest open causal path for one pollen owner."""

    foreign = pollen_swept_mask(
        center_tracks_yx,
        owner_index,
        atlas.birth_sample.shape[1:],
        1.45 * owner_radius_px,
        tail_samples,
    )
    aggregate, birth = masked_atlas(atlas, foreign)
    persistence = np.max(atlas.persistent_fraction, axis=0)
    evidence = np.maximum(
        aggregate.paired_score,
        0.20 * aggregate.score,
    )
    evidence *= np.clip(0.25 + 7.0 * persistence, 0.25, 1.0)[None]
    owner_center = np.median(
        center_tracks_yx[owner_index, -tail_samples:], axis=0
    )
    proposals = propose_pollen_roots(
        evidence,
        owner_center,
        owner_radius_px,
        birth_time=birth,
        minimum_material_birth=1.0,
        attachment_count=72,
        direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
        probe_distances_px=(2.0, 3.5, 5.0, 7.0, 9.0),
        probe_top_k=5,
        maximum_proposals=maximum_proposals,
        minimum_angle_separation_degrees=7.0,
    )
    trace_config = CoupledRibbonTraceConfig(
        step_px=1.0,
        maximum_turn_bins=1,
        curvature_penalty=0.11,
        minimum_pair_support=0.055,
        minimum_wall_balance=0.12,
        paired_support_weight=1.25,
        merged_support_weight=0.20,
        wall_balance_weight=0.05,
        evidence_floor=0.065,
        length_reward=0.010,
        width_change_penalty=0.10,
        maximum_reacquisition_width_change_px=1.5,
        maximum_gap_steps=5,
        maximum_initial_gap_steps=4,
        root_occlusion_px=1.15 * owner_radius_px,
        beam_width=450,
        minimum_length_px=5.0,
        maximum_length_px=75.0,
        minimum_endpoint_separation_fraction=0.38,
        endpoint_openness_reward=0.16,
        birth_backtrack_tolerance=1.0,
        birth_backtrack_penalty=0.48,
        birth_progress_reward=0.035,
    )
    candidates = []
    for proposal in proposals:
        trace = trace_coupled_ribbon_lifted(
            aggregate,
            proposal.root_yx,
            proposal.direction_yx,
            birth_time=birth,
            minimum_material_birth=1.0,
            config=trace_config,
        )
        if trace.length_px < trace_config.minimum_length_px - 0.5:
            continue
        path = trace.path_yx
        directions = np.mod(
            np.rint(
                trace.direction_radians
                / math.pi
                * atlas.birth_sample.shape[0]
            ).astype(int),
            atlas.birth_sample.shape[0],
        )
        points = np.rint(path).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, foreign.shape[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, foreign.shape[1] - 1)
        raw_birth = atlas.birth_sample[
            directions,
            points[:, 0],
            points[:, 1],
        ].copy()
        raw_birth[
            (raw_birth <= 0) | (raw_birth >= atlas.sample_count)
        ] = atlas.sample_count
        if not np.any(raw_birth < atlas.sample_count):
            continue
        path_birth = monotone_path_births(
            raw_birth,
            unavailable_value=atlas.sample_count,
        )
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        evaluated = arc >= 1.25 * owner_radius_px
        if not np.any(evaluated):
            continue
        order = birth_order_score(
            raw_birth[evaluated],
            unavailable_value=atlas.sample_count,
        )
        persistent = float(
            np.mean(
                atlas.persistent_fraction[
                    directions[evaluated],
                    points[evaluated, 0],
                    points[evaluated, 1],
                ]
            )
        )
        preexisting = float(
            np.mean(
                atlas.preexisting[
                    directions[evaluated],
                    points[evaluated, 0],
                    points[evaluated, 1],
                ]
            )
        )
        foreign_fraction = float(
            np.mean(foreign[points[evaluated, 0], points[evaluated, 1]])
        )
        radial_distance = np.linalg.norm(path - owner_center[None], axis=1)
        owner_halo_fraction = float(
            np.mean(radial_distance[evaluated] <= 1.65 * owner_radius_px)
        )
        radial_extension = float(
            max(0.0, np.max(radial_distance) - owner_radius_px)
        )
        direction_index = min(len(path) - 1, 5)
        root_vector = path[0] - owner_center
        initial_vector = path[direction_index] - path[0]
        root_outward_cosine = float(
            np.dot(root_vector, initial_vector)
            / max(
                np.linalg.norm(root_vector) * np.linalg.norm(initial_vector),
                1e-6,
            )
        )
        score = (
            2.2 * trace.paired_supported_fraction
            + 1.2 * order
            + 1.4 * persistent
            + 0.012 * trace.length_px
            - 2.5 * preexisting
            - 4.0 * foreign_fraction
            - 1.8 * owner_halo_fraction
            + 0.25 * max(root_outward_cosine, 0.0)
        )
        reasons = []
        if trace.paired_supported_fraction < 0.36:
            reasons.append("weak-paired-walls")
        if order < 0.72:
            reasons.append("reversed-birth-order")
        if persistent < 0.025:
            reasons.append("weak-persistence")
        if preexisting > 0.22:
            reasons.append("preexisting-route")
        if foreign_fraction > 0.02:
            reasons.append("foreign-pollen-route")
        if owner_halo_fraction > 0.34:
            reasons.append("owner-rim-following")
        if radial_extension < 1.15 * owner_radius_px:
            reasons.append("insufficient-radial-extension")
        if root_outward_cosine < 0.20:
            reasons.append("non-outward-root")
        if trace.length_px >= trace_config.maximum_length_px - trace_config.step_px:
            reasons.append("maximum-length-censored")
        if (
            np.min(path[:, 0]) <= 1.0
            or np.min(path[:, 1]) <= 1.0
            or np.max(path[:, 0]) >= foreign.shape[0] - 2.0
            or np.max(path[:, 1]) >= foreign.shape[1] - 2.0
        ):
            reasons.append("field-boundary-censored")
        candidates.append(
            OwnerPath(
                owner_index=owner_index,
                owner_track_id=seed.track_id,
                path_yx=path,
                path_birth_samples=path_birth,
                score=float(score),
                length_px=trace.length_px,
                paired_fraction=trace.paired_supported_fraction,
                birth_order_fraction=order,
                persistent_fraction=persistent,
                preexisting_fraction=preexisting,
                foreign_pollen_fraction=foreign_fraction,
                owner_halo_fraction=owner_halo_fraction,
                radial_extension_px=radial_extension,
                root_outward_cosine=root_outward_cosine,
                accepted=not reasons,
                reason="accepted" if not reasons else ";".join(reasons),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.accepted, item.score), reverse=True)
    return candidates[0]


def resolve_duplicate_paths(paths: list[OwnerPath]) -> tuple[list[OwnerPath], list[dict]]:
    """Keep one owner when two roots claim the same extended branch."""

    accepted = [path for path in paths if path.accepted]
    losers = set()
    resolutions = []
    for left_index, left in enumerate(accepted):
        if left.owner_track_id in losers:
            continue
        left_pixels = path_pixel_set(left.path_yx[6:])
        for right in accepted[left_index + 1 :]:
            if right.owner_track_id in losers:
                continue
            right_pixels = path_pixel_set(right.path_yx[6:])
            overlap = len(left_pixels & right_pixels) / max(
                min(len(left_pixels), len(right_pixels)), 1
            )
            if overlap < 0.58:
                continue
            winner, loser = max((left, right), key=lambda item: item.score), min(
                (left, right), key=lambda item: item.score
            )
            losers.add(loser.owner_track_id)
            resolutions.append(
                {
                    "winner_owner_track_id": winner.owner_track_id,
                    "loser_owner_track_id": loser.owner_track_id,
                    "overlap_fraction": round(overlap, 4),
                }
            )
    resolved = []
    for path in paths:
        if path.owner_track_id in losers:
            resolved.append(
                OwnerPath(
                    **{
                        **path.__dict__,
                        "accepted": False,
                        "reason": "duplicate-branch-claim",
                    }
                )
            )
        else:
            resolved.append(path)
    return resolved, resolutions


def path_pixel_set(path_yx: np.ndarray) -> set[tuple[int, int]]:
    """Rasterize a centerline with one-pixel tolerance for overlap tests."""

    points = np.rint(path_yx).astype(int)
    result = set()
    for y, x in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                result.add((int(y + dy), int(x + dx)))
    return result


def write_birth_map(
    output: Path,
    final_frame: np.ndarray,
    atlas: CausalBirthAtlas,
) -> None:
    """Render the earliest orientation-specific birth at each image location."""

    birth = np.min(
        np.where(
            atlas.birth_sample > 0,
            atlas.birth_sample,
            atlas.sample_count,
        ),
        axis=0,
    )
    valid = birth < atlas.sample_count
    scaled = np.zeros_like(final_frame)
    if np.any(valid):
        lo = float(np.min(birth[valid]))
        hi = float(np.max(birth[valid]))
        scaled[valid] = np.clip(
            1.0 + 254.0 * (birth[valid] - lo) / max(hi - lo, 1.0),
            1,
            255,
        ).astype(np.uint8)
    colors = cv.applyColorMap(scaled, cv.COLORMAP_TURBO)
    base = cv.cvtColor(final_frame, cv.COLOR_GRAY2BGR)
    base[valid] = (0.25 * base[valid] + 0.75 * colors[valid]).astype(np.uint8)
    cv.imwrite(str(output), base)


def write_csvs(
    output: Path,
    paths: list[OwnerPath],
    source_frames: np.ndarray,
    analysis_to_native: float,
) -> None:
    """Write path geometry and one length measurement per sampled frame."""

    with (output / "owner_paths.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "owner_track_id",
                "path_index",
                "y_analysis_px",
                "x_analysis_px",
                "arclength_native_px",
                "birth_sample",
                "accepted",
                "reason",
            ]
        )
        for path in paths:
            arc = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(path.path_yx, axis=0), axis=1)))
            )
            for index, (point, distance, birth) in enumerate(
                zip(path.path_yx, arc, path.path_birth_samples)
            ):
                writer.writerow(
                    [
                        path.owner_track_id,
                        index,
                        round(float(point[0]), 4),
                        round(float(point[1]), 4),
                        round(float(distance * analysis_to_native), 4),
                        int(birth),
                        path.accepted,
                        path.reason,
                    ]
                )
    with (output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "owner_track_id",
                "sample_index",
                "source_frame",
                "length_native_px",
                "measurement_status",
            ]
        )
        for path in paths:
            arc = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(path.path_yx, axis=0), axis=1)))
            )
            for sample, source_frame in enumerate(source_frames):
                visible = np.flatnonzero(path.path_birth_samples <= sample)
                length = float(arc[visible[-1]]) if len(visible) and path.accepted else 0.0
                writer.writerow(
                    [
                        path.owner_track_id,
                        sample,
                        int(source_frame),
                        round(length * analysis_to_native, 4),
                        "accepted" if path.accepted else "review-only",
                    ]
                )


def render_video(
    movie: Path,
    output: Path,
    source_frames: np.ndarray,
    source_fps: float,
    native_width: int,
    native_height: int,
    analysis_width: int,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    seeds: list[OwnerSeed],
    paths: list[OwnerPath],
    tail_samples: int,
) -> Path:
    """Render accepted causal growth with every retained pollen owner visible."""

    render_width = 960
    render_height = round(native_height * render_width / native_width)
    header = 50
    temporary = output / "causal_birth_forest_raw.mp4"
    final = output / "causal_birth_forest.mov"
    writer = cv.VideoWriter(
        str(temporary),
        cv.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (render_width, render_height + header),
    )
    capture = cv.VideoCapture(str(movie))
    accepted = [path for path in paths if path.accepted]
    render_scale = render_width / analysis_width
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            writer.release()
            raise RuntimeError(f"could not render source frame {source_frame}")
        frame = cv.resize(frame, (render_width, render_height), interpolation=cv.INTER_AREA)
        canvas = np.zeros((render_height + header, render_width, 3), dtype=np.uint8)
        canvas[header:] = frame
        drift_yx = shifts_xy[sample][::-1]
        for owner, seed in enumerate(seeds):
            center = center_tracks_yx[owner, sample] + drift_yx
            y, x = center * render_scale
            cv.circle(
                canvas,
                (round(float(x)), round(float(y + header))),
                round(6.0 * render_scale),
                (255, 150, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                canvas,
                f"P{seed.track_id:02d}",
                (round(float(x + 7)), round(float(y + header - 5))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.34,
                (40, 80, 20),
                1,
                cv.LINE_AA,
            )
        active_count = 0
        for path in accepted:
            visible = np.flatnonzero(path.path_birth_samples <= sample)
            if not len(visible):
                continue
            active_count += 1
            local_shift = (
                center_tracks_yx[path.owner_index, sample]
                - np.median(
                    center_tracks_yx[path.owner_index, -tail_samples:], axis=0
                )
            )
            points = (
                path.path_yx[: visible[-1] + 1]
                + local_shift[None]
                + drift_yx[None]
            )
            points_xy = np.rint(points[:, ::-1] * render_scale).astype(np.int32)
            points_xy[:, 1] += header
            if len(points_xy) >= 2:
                cv.polylines(canvas, [points_xy], False, (255, 0, 220), 3, cv.LINE_AA)
            tip = tuple(int(value) for value in points_xy[-1])
            cv.circle(canvas, tip, 4, (0, 255, 0), -1, cv.LINE_AA)
        elapsed_minutes = source_frame / source_fps / 60.0
        cv.putText(
            canvas,
            f"V24 CAUSAL BIRTH FOREST | {len(seeds)} POLLEN | source {source_frame}/{int(source_frames[-1])} | {elapsed_minutes:.1f} min",
            (14, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.60,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            f"accepted {len(accepted)} | active {active_count}",
            (render_width - 230, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 150, 235),
            1,
            cv.LINE_AA,
        )
        writer.write(canvas)
    capture.release()
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final),
        ],
        check=True,
    )
    temporary.unlink()
    return final


def main() -> None:
    """Build the complete low-density causal birth forest and review movie."""

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_frames, fps, frame_count, native_width, native_height = movie_schedule(
        args.movie,
        args.sample_interval_s,
    )
    analysis_scale = args.width / native_width
    frames = load_grayscale_samples(
        args.movie,
        source_frames,
        args.width,
        native_width,
        native_height,
    )
    aligned, shifts_xy, registration_response = stabilize_translations(frames)
    seeds, identity = load_owner_seeds(args.identity_report, analysis_scale)
    initial_centers = np.asarray([seed.center_yx for seed in seeds])
    center_tracks, center_scores = track_pollen_centers(
        aligned,
        initial_centers,
        template_radius_px=5,
        search_radius_px=5,
        minimum_score=0.25,
    )
    feature_config = OrientationScoreConfig(
        orientation_count=16,
        half_widths_px=(1.0, 1.5, 2.0, 2.5, 3.0),
        tangent_samples_px=(-2.0, -1.0, 0.0, 1.0, 2.0),
        wall_sigma_px=0.6,
        background_sigma_px=3.5,
        structure_sigma_px=1.1,
        orientation_concentration=4.0,
        center_darkness_penalty=0.15,
        asymmetry_penalty=0.25,
    )
    atlas_config = CausalBirthAtlasConfig(
        warmup_samples=args.warmup_samples,
        tail_samples=args.tail_samples,
    )
    atlas_cache = args.output / "causal_birth_atlas.npz"
    if atlas_cache.exists():
        cached = np.load(atlas_cache)
        atlas = CausalBirthAtlas(
            aggregate=PairedWallOrientationResult(
                score=cached["score"],
                paired_score=cached["paired_score"],
                half_width_px=cached["half_width_px"],
                wall_balance=cached["wall_balance"],
            ),
            birth_sample=cached["birth_sample"],
            persistent_fraction=cached["persistent_fraction"],
            preexisting=cached["preexisting"],
            sample_count=len(source_frames),
        )
        print(f"[v24 atlas] reused {atlas_cache}", flush=True)
    else:
        atlas = build_causal_birth_atlas(
            orientation_feature_stream(aligned, feature_config),
            len(source_frames),
            atlas_config,
        )
        np.savez_compressed(
            atlas_cache,
            score=atlas.aggregate.score,
            paired_score=atlas.aggregate.paired_score,
            half_width_px=atlas.aggregate.half_width_px,
            wall_balance=atlas.aggregate.wall_balance,
            birth_sample=atlas.birth_sample,
            persistent_fraction=atlas.persistent_fraction,
            preexisting=atlas.preexisting,
        )
    paths = []
    selected_ids = (
        None
        if not args.track_ids
        else {int(value) for value in args.track_ids.split(",")}
    )
    for owner_index, seed in enumerate(seeds):
        if selected_ids is not None and seed.track_id not in selected_ids:
            continue
        path = trace_owner(
            owner_index,
            seed,
            15.0 * analysis_scale,
            atlas,
            center_tracks,
            args.maximum_proposals,
            args.tail_samples,
        )
        if path is not None:
            paths.append(path)
        print(
            f"[v24 trace] {owner_index + 1}/{len(seeds)} P{seed.track_id:02d}: "
            + ("no path" if path is None else f"{path.reason}, {path.length_px / analysis_scale:.1f}px"),
            flush=True,
        )
    paths, duplicate_resolutions = resolve_duplicate_paths(paths)
    write_birth_map(args.output / "orientation_birth_map.jpg", aligned[-1], atlas)
    write_csvs(args.output, paths, source_frames, 1.0 / analysis_scale)
    movie = render_video(
        args.movie,
        args.output,
        source_frames,
        fps,
        native_width,
        native_height,
        args.width,
        shifts_xy,
        center_tracks,
        seeds,
        paths,
        args.tail_samples,
    )
    report = {
        "prototype": "v24_causal_birth_forest",
        "method": "streamed-orientation-birth-atlas-and-multi-owner-path-cover",
        "input_movie": str(args.movie),
        "identity_report": str(args.identity_report),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "source_frames": source_frames.tolist(),
        "analysis_width": args.width,
        "analysis_scale_to_native": 1.0 / analysis_scale,
        "owner_count": len(seeds),
        "previously_excluded_fragmented_owner_ids": [
            seed.track_id for seed in seeds if seed.semantic_observations == 1
        ],
        "path_candidate_owner_count": len(paths),
        "accepted_owner_count": sum(path.accepted for path in paths),
        "accepted_owner_ids": [path.owner_track_id for path in paths if path.accepted],
        "duplicate_resolutions": duplicate_resolutions,
        "median_pollen_match_score": float(np.median(center_scores)),
        "minimum_pollen_match_score": float(np.min(center_scores)),
        "median_registration_response": float(np.median(registration_response)),
        "atlas_config": atlas_config.__dict__,
        "paths": [
            {
                "owner_track_id": path.owner_track_id,
                "accepted": path.accepted,
                "reason": path.reason,
                "length_native_px": path.length_px / analysis_scale,
                "score": path.score,
                "paired_fraction": path.paired_fraction,
                "birth_order_fraction": path.birth_order_fraction,
                "persistent_fraction": path.persistent_fraction,
                "preexisting_fraction": path.preexisting_fraction,
                "foreign_pollen_fraction": path.foreign_pollen_fraction,
                "owner_halo_fraction": path.owner_halo_fraction,
                "radial_extension_native_px": path.radial_extension_px
                / analysis_scale,
                "root_outward_cosine": path.root_outward_cosine,
            }
            for path in paths
        ],
        "artifacts": {
            "review_video": str(movie),
            "birth_map": str(args.output / "orientation_birth_map.jpg"),
            "paths_csv": str(args.output / "owner_paths.csv"),
            "measurements_csv": str(args.output / "measurements.csv"),
        },
        "scope": "Research prototype; manual centerline truth is still required before scientific promotion.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("owner_count", "accepted_owner_count", "accepted_owner_ids")}, indent=2))


if __name__ == "__main__":
    main()
