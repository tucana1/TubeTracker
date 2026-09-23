"""Shared configuration, grain tracking, and rendering for the tube prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from .analysis import Detections
from .models import ROI, Track


ANALYSIS_PREPROCESSING_MODES = (
    "raw",
    "clahe",
    "background",
    "background_clahe",
)


@dataclass(frozen=True)
class CurveTraceConfig:
    """Hold the tunable parameters for one experimental tracing run."""

    background_cutoff: int = 55
    blur_radius: int = 10
    preprocessing: str = "background"
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 16
    background_sigma: float = 35.0
    min_grain_radius: int = 6
    max_grain_radius: int = 13
    grain_threshold: int = 14
    min_grain_circle_score: float = 0.24
    grain_snap_distance: float = 28.0
    grain_reacquire_distance: float = 85.0
    grain_max_flow_error: float = 3.0
    late_grain_confirmation_frames: int = 3
    late_grain_min_circle_score: float = 0.45
    max_grains: int = 0
    birth_search_radius: int = 110
    extension_search_radius: int = 48
    attachment_width: int = 4
    min_attachment_probability: float = 0.16
    min_path_probability: float = 0.12
    min_extension_temporal_score: float = 0.07
    max_geodesic_endpoints: int = 24
    burst_search_radius: int = 24
    burst_min_component_area: int = 45
    burst_min_changed_fraction: float = 0.08
    burst_score_threshold: float = 0.55
    burst_min_confirmed_frames: int = 6
    burst_min_direct_extensions: int = 2
    burst_min_curve_length: float = 50.0
    min_curve_length: float = 24.0
    max_initial_curve_length: float = 40.0
    min_curve_straightness: float = 0.68
    min_birth_temporal_score: float = 0.08
    birth_confirmation_frames: int = 3
    min_confirmation_growth: float = 6.0
    chain_refit_radius: int = 4
    chain_root_lock_points: int = 4
    chain_refit_min_support: float = 0.10
    chain_refit_max_length_fraction: float = 0.05
    chain_refit_offset_change_penalty: float = 0.08
    chain_refit_offset_magnitude_penalty: float = 0.005
    max_tip_step: float = 35.0
    max_tip_lateral_step: float = 18.0


@dataclass
class CurveMeasurement:
    """Store one grain's traced curve and quality evidence at one time point."""

    grain_id: str
    analysis_frame: int
    grain_x: float
    grain_y: float
    grain_radius: float
    points: np.ndarray | None
    global_shift_x: float
    global_shift_y: float
    registration_response: float
    grain_shift_x: float
    grain_shift_y: float
    structure_score: float
    temporal_score: float
    prior_deviation: float | None
    confidence: float
    status: str
    used_prior: bool
    grain_tracking_status: str = "unknown"
    candidate_points: np.ndarray | None = None
    burst_candidate: bool = False
    burst_candidate_frame: int = -1
    burst_score: float = 0.0
    burst_reason: str = ""

    @property
    def length_px(self):
        """Return polyline arc length in resized analysis pixels."""
        return curve_length(self.points)

    @property
    def straight_px(self):
        """Return straight distance from the traced root to its distal endpoint."""
        if self.points is None or len(self.points) < 2:
            return math.nan
        return float(np.linalg.norm(self.points[-1] - self.points[0]))


def robust_normalize(image):
    """Scale an image to zero-to-one while limiting isolated extreme pixels."""
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values, (2.0, 99.5))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def prepare_analysis_gray(frame, config):
    """Prepare a grayscale analysis copy without modifying review imagery."""
    mode = config.preprocessing
    if mode not in ANALYSIS_PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    if frame.ndim == 2:
        gray = frame.astype(np.uint8, copy=True)
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if mode in {"background", "background_clahe"}:
        background = cv.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=config.background_sigma,
            sigmaY=config.background_sigma,
            borderType=cv.BORDER_REFLECT,
        )
        midpoint = float(np.median(background))
        corrected = gray.astype(np.float32) - background.astype(np.float32) + midpoint
        gray = np.clip(corrected, 0, 255).astype(np.uint8)
    if mode in {"clahe", "background_clahe"}:
        tile_size = max(2, int(config.clahe_tile_size))
        gray = cv.createCLAHE(
            clipLimit=max(0.1, float(config.clahe_clip_limit)),
            tileGridSize=(tile_size, tile_size),
        ).apply(gray)
    return gray


def curve_length(points, x_scale=1.0, y_scale=1.0):
    """Measure a row-column polyline with optional axis scale correction."""
    if points is None or len(points) < 2:
        return math.nan
    deltas = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    return float(np.hypot(deltas[:, 1] / x_scale, deltas[:, 0] / y_scale).sum())


def resample_curve(points, spacing=2.0, max_points=96):
    """Redistribute a curve at near-uniform arc-length intervals."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0:
        return points[:1].copy()
    count = min(max_points, max(5, int(math.ceil(cumulative[-1] / spacing)) + 1))
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def ordered_prefix_retention(reference_xy, candidate_xy, tolerance_px):
    """Measure root-to-tip agreement at matching material arc positions."""
    reference = np.asarray(reference_xy, dtype=np.float64)
    candidate = np.asarray(candidate_xy, dtype=np.float64)
    if len(reference) < 2 or len(candidate) < 2:
        return 0.0

    def arc_lengths(points):
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )

    reference_arc = arc_lengths(reference)
    candidate_arc = arc_lengths(candidate)
    reference_length = float(reference_arc[-1])
    candidate_length = float(candidate_arc[-1])
    if reference_length <= 0.0 or candidate_length <= 0.0:
        return 0.0
    sample_count = max(8, int(np.ceil(reference_length / 2.0)) + 1)
    sample_arc = np.linspace(0.0, reference_length, sample_count)
    reference_samples = np.column_stack(
        [
            np.interp(sample_arc, reference_arc, reference[:, axis])
            for axis in range(2)
        ]
    )
    available = sample_arc <= candidate_length + 1e-6
    candidate_samples = np.full_like(reference_samples, np.nan)
    for axis in range(2):
        candidate_samples[available, axis] = np.interp(
            sample_arc[available], candidate_arc, candidate[:, axis]
        )
    distances = np.full(sample_count, np.inf, dtype=np.float64)
    distances[available] = np.linalg.norm(
        reference_samples[available] - candidate_samples[available], axis=1
    )
    return float(np.mean(distances <= float(tolerance_px)))


def register_translation(previous_gray, current_gray, max_shift=80.0):
    """Align the previous frame to the current frame with global translation."""
    previous = np.asarray(previous_gray, dtype=np.float32)
    current = np.asarray(current_gray, dtype=np.float32)
    window = cv.createHanningWindow((previous.shape[1], previous.shape[0]), cv.CV_32F)
    (shift_x, shift_y), response = cv.phaseCorrelate(previous, current, window)
    if not np.isfinite((shift_x, shift_y, response)).all() or math.hypot(
        shift_x, shift_y
    ) > max_shift:
        shift_x, shift_y, response = 0.0, 0.0, 0.0
    transform = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    aligned = cv.warpAffine(
        previous_gray,
        transform,
        (previous_gray.shape[1], previous_gray.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT,
    )
    return aligned, float(shift_x), float(shift_y), float(response)


def score_grain_circle(gray, gradient, x, y, radius):
    """Score circular edge completeness, symmetry, and radial contrast."""
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)

    def sample(image, sample_radius):
        map_x = (x + sample_radius * cosines).astype(np.float32)[None, :]
        map_y = (y + sample_radius * sines).astype(np.float32)[None, :]
        return cv.remap(
            image,
            map_x,
            map_y,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )[0]

    edge_samples = np.vstack(
        [sample(gradient, max(1.0, radius + offset)) for offset in (-2, -1, 0, 1, 2)]
    ).max(axis=0)
    coverage = float(np.mean(edge_samples >= 0.16))
    edge_strength = float(np.mean(np.clip(edge_samples / 0.45, 0.0, 1.0)))
    opposite = np.roll(edge_samples, len(edge_samples) // 2)
    symmetry = float(np.mean(np.minimum(edge_samples, opposite) >= 0.12))
    ring_intensity = sample(gray, radius)
    inner = sample(gray, max(1.0, radius - 3.0))
    outer = sample(gray, radius + 3.0)
    contrast = float(
        np.clip(np.mean(0.5 * (inner + outer) - ring_intensity) / 32.0, 0.0, 1.0)
    )
    return float(
        np.clip(
            0.42 * coverage
            + 0.30 * edge_strength
            + 0.18 * symmetry
            + 0.10 * contrast,
            0.0,
            1.0,
        )
    )


def detect_grain_candidates(frames, config):
    """Return every Hough-circle grain candidate in every analysis frame."""
    analysis_frames = [
        cv.cvtColor(prepare_analysis_gray(frame, config), cv.COLOR_GRAY2BGR)
        for frame in frames
    ]
    detections = Detections(
        img_list=analysis_frames,
        bg_threshold=config.background_cutoff,
        blur_radius=config.blur_radius,
    )
    detections_by_frame = []
    minimum_distance = max(1, int(0.75 * (config.min_grain_radius + config.max_grain_radius)))
    for frame_index, edge_image in enumerate(detections.img_list_gray_noiseless):
        gray = prepare_analysis_gray(frames[frame_index], config)
        gradient_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        gradient_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        gradient = robust_normalize(cv.magnitude(gradient_x, gradient_y))
        circles = cv.HoughCircles(
            edge_image,
            cv.HOUGH_GRADIENT,
            1,
            minimum_distance,
            param1=210,
            param2=config.grain_threshold,
            minRadius=config.min_grain_radius,
            maxRadius=config.max_grain_radius,
        )
        frame_detections = []
        if circles is not None:
            for x, y, radius in np.around(circles[0]).astype(np.int32):
                circle_score = score_grain_circle(
                    gray, gradient, int(x), int(y), int(radius)
                )
                if circle_score < config.min_grain_circle_score:
                    continue
                roi = ROI(
                    x_l=int(x - radius),
                    y_t=int(y - radius),
                    x_r=int(x + radius),
                    y_b=int(y + radius),
                    frame=frame_index,
                    detection_method="prototype_hough",
                )
                roi.circle_score = circle_score
                if roi.overlaps_any(detections.rois[frame_index], overlap=0.2):
                    frame_detections.append(roi)
        detections_by_frame.append(frame_detections)
    return detections_by_frame


def _candidate_matches(predictions, radii, candidates, cutoffs):
    """Greedily assign unique nearby circle candidates to predicted grain centers."""
    possible = []
    for track_index, (prediction, radius, cutoff) in enumerate(
        zip(predictions, radii, cutoffs)
    ):
        for candidate_index, candidate in enumerate(candidates):
            candidate_center = np.array(
                [candidate.gv3.x, candidate.gv3.y], dtype=np.float64
            )
            distance = float(np.linalg.norm(candidate_center - prediction))
            if distance > cutoff:
                continue
            radius_difference = abs(max(candidate.w, candidate.h) / 2.0 - radius)
            circle_penalty = 3.0 * (1.0 - getattr(candidate, "circle_score", 0.0))
            possible.append(
                (
                    distance + 0.35 * radius_difference + circle_penalty,
                    track_index,
                    candidate_index,
                )
            )
    matches = {}
    used_candidates = set()
    for _, track_index, candidate_index in sorted(possible):
        if track_index in matches or candidate_index in used_candidates:
            continue
        matches[track_index] = candidate_index
        used_candidates.add(candidate_index)
    return matches, used_candidates


def track_grain_candidates(frames, candidates_by_frame, config):
    """Follow initial grains and promote persistent later pollen candidates."""
    if not frames or not candidates_by_frame or not candidates_by_frame[0]:
        return [], [set() for _ in frames]

    seeds = candidates_by_frame[0]
    if config.max_grains > 0:
        seeds = seeds[: config.max_grains]
    positions = np.asarray(
        [[candidate.gv3.x, candidate.gv3.y] for candidate in seeds], dtype=np.float64
    )
    radii = np.asarray(
        [max(candidate.w, candidate.h) / 2.0 for candidate in seeds],
        dtype=np.float64,
    )
    velocities = np.zeros_like(positions)
    tracks = [[] for _ in seeds]
    pending_tracks = []
    assigned_by_frame = [set() for _ in frames]
    for track_index, seed in enumerate(seeds):
        seed.detection_method = "hough_seed"
        seed.tracking_confidence = 1.0
        tracks[track_index].append(seed)
        assigned_by_frame[0].add(track_index)

    previous_gray = prepare_analysis_gray(frames[0], config)
    lk_parameters = {
        "winSize": (51, 51),
        "maxLevel": 4,
        "criteria": (
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
        "minEigThreshold": 1e-4,
    }
    for frame_index in range(1, len(frames)):
        current_gray = prepare_analysis_gray(frames[frame_index], config)
        source_points = positions.astype(np.float32).reshape(-1, 1, 2)
        flowed, forward_status, _ = cv.calcOpticalFlowPyrLK(
            previous_gray, current_gray, source_points, None, **lk_parameters
        )
        returned, backward_status, _ = cv.calcOpticalFlowPyrLK(
            current_gray, previous_gray, flowed, None, **lk_parameters
        )
        flow_positions = flowed[:, 0].astype(np.float64)
        flow_error = np.linalg.norm(returned[:, 0] - source_points[:, 0], axis=1)
        height, width = current_gray.shape
        flow_good = (
            (forward_status[:, 0] > 0)
            & (backward_status[:, 0] > 0)
            & (flow_error <= config.grain_max_flow_error)
            & (flow_positions[:, 0] >= 0)
            & (flow_positions[:, 0] < width)
            & (flow_positions[:, 1] >= 0)
            & (flow_positions[:, 1] < height)
        )

        _, global_x, global_y, _ = register_translation(
            previous_gray, current_gray
        )
        predictions = positions + velocities
        predictions[~flow_good] = (
            positions[~flow_good] + np.array([global_x, global_y])
        )
        predictions[flow_good] = flow_positions[flow_good]
        cutoffs = np.where(
            flow_good,
            config.grain_snap_distance,
            config.grain_reacquire_distance,
        )
        matches, used_candidates = _candidate_matches(
            predictions, radii, candidates_by_frame[frame_index], cutoffs
        )
        assigned_by_frame[frame_index] = set(used_candidates)

        next_positions = predictions.copy()
        for track_index in range(len(tracks)):
            candidate_index = matches.get(track_index)
            if candidate_index is not None:
                candidate = candidates_by_frame[frame_index][candidate_index]
                candidate_center = np.array(
                    [candidate.gv3.x, candidate.gv3.y], dtype=np.float64
                )
                if flow_good[track_index]:
                    next_positions[track_index] = (
                        0.65 * candidate_center + 0.35 * flow_positions[track_index]
                    )
                    method = "hough+optical_flow"
                    confidence = max(0.55, 1.0 - flow_error[track_index] / 6.0)
                else:
                    next_positions[track_index] = candidate_center
                    method = "hough_reacquired"
                    confidence = 0.65
                radii[track_index] = (
                    0.8 * radii[track_index]
                    + 0.2 * max(candidate.w, candidate.h) / 2.0
                )
            elif flow_good[track_index]:
                method = "optical_flow"
                confidence = max(0.45, 1.0 - flow_error[track_index] / 6.0)
            else:
                method = "motion_predicted"
                confidence = 0.15

            next_positions[track_index, 0] = np.clip(
                next_positions[track_index, 0], 0, width - 1
            )
            next_positions[track_index, 1] = np.clip(
                next_positions[track_index, 1], 0, height - 1
            )
            displacement = next_positions[track_index] - positions[track_index]
            velocities[track_index] = 0.65 * velocities[track_index] + 0.35 * displacement
            x, y = next_positions[track_index]
            radius = radii[track_index]
            roi = ROI(
                x_l=int(round(x - radius)),
                y_t=int(round(y - radius)),
                x_r=int(round(x + radius)),
                y_b=int(round(y + radius)),
                frame=frame_index,
                detection_method=method,
            )
            roi.tracking_confidence = float(confidence)
            tracks[track_index].append(roi)

        available = [
            index
            for index, candidate in enumerate(candidates_by_frame[frame_index])
            if index not in used_candidates
            and getattr(candidate, "circle_score", 0.0)
            >= config.late_grain_min_circle_score
        ]
        possible_pending = []
        for pending_index, pending_track in enumerate(pending_tracks):
            last = pending_track["observations"][-1][1]
            for candidate_index in available:
                candidate = candidates_by_frame[frame_index][candidate_index]
                distance = math.hypot(
                    candidate.gv3.x - last.gv3.x,
                    candidate.gv3.y - last.gv3.y,
                )
                if distance <= config.grain_snap_distance:
                    possible_pending.append(
                        (distance, pending_index, candidate_index)
                    )
        matched_pending = set()
        matched_candidates = set()
        for _, pending_index, candidate_index in sorted(possible_pending):
            if (
                pending_index in matched_pending
                or candidate_index in matched_candidates
            ):
                continue
            candidate = candidates_by_frame[frame_index][candidate_index]
            pending_tracks[pending_index]["observations"].append(
                (candidate_index, candidate)
            )
            pending_tracks[pending_index]["missed"] = 0
            matched_pending.add(pending_index)
            matched_candidates.add(candidate_index)
        for pending_index, pending_track in enumerate(pending_tracks):
            if pending_index not in matched_pending:
                pending_track["missed"] += 1
        pending_tracks = [
            pending_track
            for pending_track in pending_tracks
            if pending_track["missed"] <= 1
        ]

        for candidate_index in available:
            if candidate_index in matched_candidates:
                continue
            candidate = candidates_by_frame[frame_index][candidate_index]
            candidate_center = np.array(
                [candidate.gv3.x, candidate.gv3.y], dtype=np.float64
            )
            if len(next_positions):
                nearest_active = float(
                    np.min(np.linalg.norm(next_positions - candidate_center, axis=1))
                )
                if nearest_active < 1.5 * max(candidate.w, candidate.h):
                    continue
            pending_tracks.append(
                {
                    "observations": [(candidate_index, candidate)],
                    "missed": 0,
                }
            )

        promoted = []
        still_pending = []
        for pending_track in pending_tracks:
            observations = pending_track["observations"]
            if len(observations) < config.late_grain_confirmation_frames:
                still_pending.append(pending_track)
                continue
            promoted.append(pending_track)
        pending_tracks = still_pending
        for pending_track in promoted:
            if config.max_grains > 0 and len(tracks) >= config.max_grains:
                continue
            observations = pending_track["observations"]
            rois = []
            for candidate_index, candidate in observations:
                candidate.detection_method = "late_hough_seed"
                candidate.tracking_confidence = getattr(
                    candidate, "circle_score", 0.65
                )
                rois.append(candidate)
                assigned_by_frame[candidate.gv6].add(candidate_index)
            latest = rois[-1]
            previous_observation = rois[-2]
            frame_delta = max(1, latest.gv6 - previous_observation.gv6)
            velocity = np.array(
                [
                    (latest.gv3.x - previous_observation.gv3.x) / frame_delta,
                    (latest.gv3.y - previous_observation.gv3.y) / frame_delta,
                ],
                dtype=np.float64,
            )
            tracks.append(rois)
            next_positions = np.vstack(
                [next_positions, [latest.gv3.x, latest.gv3.y]]
            )
            radii = np.append(radii, max(latest.w, latest.h) / 2.0)
            velocities = np.vstack([velocities, velocity])
        positions = next_positions
        previous_gray = current_gray

    output = []
    for index, observations in enumerate(tracks, start=1):
        track = Track(observations, ID=f"g{index:03d}")
        track.fill_missing_frames(up_to_frame=len(frames))
        output.append(track)
    return output, assigned_by_frame


def detect_and_track_grains(frames, config):
    """Detect all circle candidates and preserve every initial grain track."""
    candidates_by_frame = detect_grain_candidates(frames, config)
    tracks, assigned_by_frame = track_grain_candidates(
        frames, candidates_by_frame, config
    )
    return tracks, candidates_by_frame, assigned_by_frame


def draw_measurements(
    frames,
    measurements,
    pixel_size=1.0,
    x_scale=1.0,
    y_scale=1.0,
    candidates_by_frame=None,
    assigned_by_frame=None,
    source_frames=None,
    time_per_source_frame=None,
    time_unit="sec",
):
    """Draw every grain, raw candidate, traced arc, and review status."""
    by_frame = {}
    for measurement in measurements:
        by_frame.setdefault(measurement.analysis_frame, []).append(measurement)
    rendered = []
    for frame_index, frame in enumerate(frames):
        image = frame.copy()
        assigned = (
            assigned_by_frame[frame_index]
            if assigned_by_frame is not None
            else set()
        )
        candidates = (
            candidates_by_frame[frame_index]
            if candidates_by_frame is not None
            else []
        )
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in assigned:
                continue
            center = (candidate.gv3.x, candidate.gv3.y)
            radius = max(candidate.w, candidate.h) // 2
            cv.circle(image, center, radius, (255, 205, 135), 1, cv.LINE_AA)
            cv.drawMarker(
                image,
                center,
                (255, 205, 135),
                cv.MARKER_CROSS,
                5,
                1,
                cv.LINE_AA,
            )

        frame_measurements = by_frame.get(frame_index, [])
        curve_count = 0
        for measurement in frame_measurements:
            if measurement.burst_candidate:
                grain_color = (0, 0, 255)
            elif measurement.grain_tracking_status == "motion_predicted":
                grain_color = (0, 165, 255)
            elif measurement.grain_tracking_status == "optical_flow":
                grain_color = (255, 190, 0)
            else:
                grain_color = (255, 120, 0)
            center = (int(round(measurement.grain_x)), int(round(measurement.grain_y)))
            cv.circle(
                image,
                center,
                int(round(measurement.grain_radius)),
                grain_color,
                1,
                cv.LINE_AA,
            )
            label = measurement.grain_id.upper()
            if measurement.candidate_points is not None:
                candidate_xy = np.rint(
                    measurement.candidate_points[:, ::-1]
                ).astype(np.int32)
                cv.polylines(
                    image,
                    [candidate_xy],
                    False,
                    (0, 140, 255),
                    1,
                    cv.LINE_AA,
                )
            if measurement.points is not None:
                curve_count += 1
                if measurement.burst_candidate:
                    curve_color = (0, 0, 255)
                elif measurement.status in {
                    "ok",
                    "attached_growth_confirmed",
                    "tip_extended",
                }:
                    curve_color = (0, 215, 0)
                elif measurement.status.startswith("held_"):
                    curve_color = (255, 180, 0)
                else:
                    curve_color = (0, 180, 255)
                xy = np.rint(measurement.points[:, ::-1]).astype(np.int32)
                cv.polylines(image, [xy], False, curve_color, 2, cv.LINE_AA)
                cv.circle(image, tuple(xy[-1]), 3, (255, 0, 255), -1, cv.LINE_AA)
                calibrated_length = curve_length(
                    measurement.points, x_scale=x_scale, y_scale=y_scale
                ) * pixel_size
                label += f" {calibrated_length:.0f}"
                if measurement.burst_candidate:
                    cv.drawMarker(
                        image,
                        tuple(xy[-1]),
                        (0, 0, 255),
                        cv.MARKER_TILTED_CROSS,
                        13,
                        2,
                        cv.LINE_AA,
                    )
                    label += " BURST?"
            cv.putText(
                image,
                label,
                (center[0] + 5, center[1] - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                label,
                (center[0] + 5, center[1] - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.30,
                grain_color,
                1,
                cv.LINE_AA,
            )

        overlay = image.copy()
        cv.rectangle(overlay, (0, 0), (image.shape[1], 50), (0, 0, 0), -1)
        image = cv.addWeighted(overlay, 0.72, image, 0.28, 0)
        source_frame = (
            source_frames[frame_index]
            if source_frames is not None
            else frame_index
        )
        time_label = ""
        if time_per_source_frame is not None:
            time_label = f"  time {source_frame * time_per_source_frame:.1f} {time_unit}"
        burst_count = sum(item.burst_candidate for item in frame_measurements)
        header = (
            f"source frame {source_frame}{time_label}  |  "
            f"tracked grains {len(frame_measurements)}  |  "
            f"pollen candidates {len(candidates)}  |  tubes {curve_count}  |  rupture flags {burst_count}"
        )
        cv.putText(
            image,
            header,
            (8, 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "blue=tracked pollen  light blue=unassigned pollen candidate  orange circle=predicted  orange path=tube candidate  green=extended  cyan=held  red=BURST?",
            (8, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.39,
            (230, 230, 230),
            1,
            cv.LINE_AA,
        )
        rendered.append(image)
    return rendered


def write_preview_video(path, frames, frames_per_second=5.0):
    """Write annotated prototype frames as an MP4 or MJPG comparison video."""
    path = Path(path)
    height, width = frames[0].shape[:2]
    codec = "mp4v" if path.suffix.lower() == ".mp4" else "MJPG"
    writer = cv.VideoWriter(
        str(path), cv.VideoWriter_fourcc(*codec), frames_per_second, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
