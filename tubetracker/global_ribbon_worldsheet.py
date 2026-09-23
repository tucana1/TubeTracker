"""Select a pollen-owned tube as one global sequence of complete curves.

Framewise tracing is deliberately allowed to remain uncertain.  Each time
point contributes several ordered pollen-to-tip curves, including weaker lanes
through a crossing.  This module compares their material coordinates and uses
the complete movie to select one temporally coherent worldsheet, with explicit
pre-germination and short optical-gap states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RibbonHypothesis:
    """Describe one complete root-to-tip interpretation of a sampled frame."""

    path_yx: np.ndarray
    optical_score: float
    paired_fraction: float
    mean_paired_support: float
    endpoint_openness: float
    source_index: int = 0

    @property
    def length_px(self) -> float:
        """Return the centerline arclength in pixels."""

        path = np.asarray(self.path_yx, dtype=np.float64)
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


@dataclass(frozen=True)
class GlobalRibbonConfig:
    """Configure lifecycle handling without image-specific pixel thresholds."""

    maximum_gap_samples: int = 2
    require_active_final_sample: bool = True
    continuation_log_odds: float = math.log(2.0)


@dataclass(frozen=True)
class GlobalRibbonScales:
    """Record robust transition scales learned from the candidate population."""

    prefix_error_px: float
    root_motion_px: float
    length_change_px: float
    tip_angle_radians: float


@dataclass(frozen=True)
class GlobalRibbonResult:
    """Store the globally selected lifecycle and curve hypothesis sequence."""

    selected: tuple[RibbonHypothesis | None, ...]
    states: tuple[str, ...]
    candidate_indices: tuple[int | None, ...]
    first_active_sample: int | None
    total_score: float
    score_margin: float
    scales: GlobalRibbonScales


def _curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength for one ordered curve."""

    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("candidate paths must have shape (points, 2)")
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )


def _points_at_arclength(
    path_yx: np.ndarray,
    arclength: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Interpolate ordered material coordinates along one curve."""

    return np.column_stack(
        [np.interp(samples, arclength, path_yx[:, axis]) for axis in range(2)]
    )


def ordered_prefix_error(
    previous: RibbonHypothesis,
    current: RibbonHypothesis,
) -> float:
    """Compare shared root-relative material coordinates of two curves."""

    previous_path = np.asarray(previous.path_yx, dtype=np.float64)
    current_path = np.asarray(current.path_yx, dtype=np.float64)
    previous_arc = _curve_arclength(previous_path)
    current_arc = _curve_arclength(current_path)
    shared = min(float(previous_arc[-1]), float(current_arc[-1]))
    samples = np.linspace(0.0, shared, max(2, int(math.ceil(shared)) + 1))
    previous_relative = previous_path - previous_path[0]
    current_relative = current_path - current_path[0]
    previous_points = _points_at_arclength(
        previous_relative,
        previous_arc,
        samples,
    )
    current_points = _points_at_arclength(
        current_relative,
        current_arc,
        samples,
    )
    return float(np.mean(np.linalg.norm(previous_points - current_points, axis=1)))


def _tip_angle(hypothesis: RibbonHypothesis) -> float:
    """Estimate the distal tangent angle from the final material segment."""

    path = np.asarray(hypothesis.path_yx, dtype=np.float64)
    arc = _curve_arclength(path)
    start_arc = max(0.0, float(arc[-1]) - max(2.0, 0.1 * float(arc[-1])))
    start = _points_at_arclength(path, arc, np.asarray((start_arc,)))[0]
    vector = path[-1] - start
    return math.atan2(float(vector[0]), float(vector[1]))


def _angle_difference(first: float, second: float) -> float:
    """Return the smallest absolute difference between two directed angles."""

    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _positive_scale(values: list[float], fallback: float) -> float:
    """Derive a stable positive scale from a nonnegative observation sample."""

    finite = np.asarray(
        [abs(value) for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    if not len(finite):
        return fallback
    scale = float(np.median(finite))
    return max(scale, fallback, np.finfo(np.float64).eps)


def _quality_scores(
    frame_candidates: list[tuple[RibbonHypothesis, ...]],
) -> list[np.ndarray]:
    """Score each curve from its own bounded physical evidence."""

    output = []
    for candidates in frame_candidates:
        values = []
        for candidate in candidates:
            length = max(candidate.length_px, np.finfo(np.float64).eps)
            score_density = max(0.0, candidate.optical_score / length)
            bounded_density = score_density / (1.0 + score_density)
            evidence = np.clip(
                (
                    bounded_density,
                    candidate.paired_fraction,
                    candidate.mean_paired_support,
                    candidate.endpoint_openness,
                ),
                0.0,
                1.0,
            )
            values.append(float(np.mean(evidence)))
        output.append(np.asarray(values, dtype=np.float64))
    return output


def _transition_observations(
    frame_candidates: list[tuple[RibbonHypothesis, ...]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Collect nearest-neighbor changes for automatic transition calibration."""

    prefix_errors: list[float] = []
    root_motions: list[float] = []
    length_changes: list[float] = []
    tip_angles: list[float] = []
    for previous_candidates, current_candidates in zip(
        frame_candidates,
        frame_candidates[1:],
    ):
        independent_previous = tuple(
            candidate
            for candidate in previous_candidates
            if candidate.source_index >= 0
        )
        independent_current = tuple(
            candidate
            for candidate in current_candidates
            if candidate.source_index >= 0
        )
        previous_candidates = independent_previous or previous_candidates
        current_candidates = independent_current or current_candidates
        if not previous_candidates or not current_candidates:
            continue
        for current in current_candidates:
            comparisons = []
            for previous in previous_candidates:
                prefix = ordered_prefix_error(previous, current)
                root = float(
                    np.linalg.norm(
                        np.asarray(current.path_yx[0]) - np.asarray(previous.path_yx[0])
                    )
                )
                length = abs(current.length_px - previous.length_px)
                angle = _angle_difference(_tip_angle(previous), _tip_angle(current))
                comparisons.append((prefix, root, length, angle))
            nearest = min(comparisons, key=lambda values: values[0] + values[1])
            prefix_errors.append(nearest[0])
            root_motions.append(nearest[1])
            length_changes.append(nearest[2])
            tip_angles.append(nearest[3])
    return prefix_errors, root_motions, length_changes, tip_angles


def _learn_scales(
    frame_candidates: list[tuple[RibbonHypothesis, ...]],
) -> GlobalRibbonScales:
    """Learn image-unit transition scales from adjacent candidate sets."""

    prefix, root, length, angle = _transition_observations(frame_candidates)
    segment_lengths = []
    candidate_lengths = []
    for candidates in frame_candidates:
        for candidate in candidates:
            path = np.asarray(candidate.path_yx, dtype=np.float64)
            segment_lengths.extend(np.linalg.norm(np.diff(path, axis=0), axis=1))
            candidate_lengths.append(candidate.length_px)
    spatial_resolution = _positive_scale(segment_lengths, np.finfo(np.float64).eps)
    typical_length = _positive_scale(candidate_lengths, spatial_resolution)
    angular_resolution = spatial_resolution / typical_length
    return GlobalRibbonScales(
        prefix_error_px=_positive_scale(prefix, spatial_resolution),
        root_motion_px=_positive_scale(root, spatial_resolution),
        length_change_px=_positive_scale(length, spatial_resolution),
        tip_angle_radians=_positive_scale(angle, angular_resolution),
    )


def _transition_score(
    previous: RibbonHypothesis,
    current: RibbonHypothesis,
    scales: GlobalRibbonScales,
    elapsed_samples: int = 1,
) -> float:
    """Score material continuity using robust, automatically scaled losses."""

    elapsed = max(1, elapsed_samples)
    prefix = ordered_prefix_error(previous, current) / (
        scales.prefix_error_px * elapsed
    )
    root_motion = float(
        np.linalg.norm(
            np.asarray(current.path_yx[0]) - np.asarray(previous.path_yx[0])
        )
    ) / (scales.root_motion_px * elapsed)
    growth = current.length_px - previous.length_px
    if growth < -scales.prefix_error_px * elapsed:
        return -float("inf")
    length_change = abs(growth) / (scales.length_change_px * elapsed)
    regression = max(0.0, -growth) / (scales.length_change_px * elapsed)
    tip_angle = _angle_difference(_tip_angle(previous), _tip_angle(current)) / (
        scales.tip_angle_radians * elapsed
    )
    excess = np.maximum(
        np.asarray((prefix, root_motion, length_change, tip_angle)) - 1.0,
        0.0,
    )
    losses = np.log1p(np.square(np.append(excess, regression)))
    return -float(np.sum(losses))


def select_global_ribbon_worldsheet(
    frame_candidates: list[tuple[RibbonHypothesis, ...]],
    config: GlobalRibbonConfig | None = None,
) -> GlobalRibbonResult:
    """Select the complete curve sequence jointly across all sampled frames."""

    config = config or GlobalRibbonConfig()
    if not frame_candidates:
        raise ValueError("frame_candidates cannot be empty")
    if config.maximum_gap_samples < 0:
        raise ValueError("maximum_gap_samples cannot be negative")
    if not np.isfinite(config.continuation_log_odds):
        raise ValueError("continuation_log_odds must be finite")
    for candidates in frame_candidates:
        for candidate in candidates:
            _curve_arclength(candidate.path_yx)
            if not all(
                np.isfinite(value)
                for value in (
                    candidate.optical_score,
                    candidate.paired_fraction,
                    candidate.mean_paired_support,
                    candidate.endpoint_openness,
                )
            ):
                raise ValueError("candidate measurements must be finite")

    quality = _quality_scores(frame_candidates)
    scales = _learn_scales(frame_candidates)
    frame_count = len(frame_candidates)
    onset_cost = math.log1p(frame_count)
    gap_cost = onset_cost / (config.maximum_gap_samples + 1)
    scores: list[dict[tuple, float]] = []
    predecessors: list[dict[tuple, tuple | None]] = []

    for frame_index, candidates in enumerate(frame_candidates):
        current_scores: dict[tuple, float] = {("prebirth",): 0.0}
        current_predecessors: dict[tuple, tuple | None] = {
            ("prebirth",): None if frame_index == 0 else ("prebirth",)
        }
        previous_scores = scores[-1] if scores else {}

        for candidate_index, candidate in enumerate(candidates):
            key = ("path", candidate_index)
            emission = float(quality[frame_index][candidate_index])
            best_score = emission - onset_cost
            best_predecessor: tuple | None = (
                None if frame_index == 0 else ("prebirth",)
            )
            for previous_key, previous_score in previous_scores.items():
                if previous_key[0] == "path":
                    previous = frame_candidates[frame_index - 1][previous_key[1]]
                    proposed = (
                        previous_score
                        + emission
                        + _transition_score(previous, candidate, scales)
                        + config.continuation_log_odds
                    )
                elif previous_key[0] == "gap":
                    origin_frame, origin_index = previous_key[1], previous_key[2]
                    previous = frame_candidates[origin_frame][origin_index]
                    proposed = (
                        previous_score
                        + emission
                        + _transition_score(
                            previous,
                            candidate,
                            scales,
                            frame_index - origin_frame,
                        )
                        + config.continuation_log_odds
                    )
                else:
                    continue
                if proposed > best_score:
                    best_score = proposed
                    best_predecessor = previous_key
            current_scores[key] = best_score
            current_predecessors[key] = best_predecessor

        for previous_key, previous_score in previous_scores.items():
            if previous_key[0] == "path" and config.maximum_gap_samples:
                key = ("gap", frame_index - 1, previous_key[1], 1)
            elif (
                previous_key[0] == "gap"
                and previous_key[3] < config.maximum_gap_samples
            ):
                key = (
                    "gap",
                    previous_key[1],
                    previous_key[2],
                    previous_key[3] + 1,
                )
            else:
                continue
            proposed = previous_score - gap_cost
            if proposed > current_scores.get(key, -np.inf):
                current_scores[key] = proposed
                current_predecessors[key] = previous_key

        scores.append(current_scores)
        predecessors.append(current_predecessors)

    final_scores = scores[-1]
    eligible = [
        (key, value)
        for key, value in final_scores.items()
        if not config.require_active_final_sample or key[0] == "path"
    ]
    if not eligible:
        return GlobalRibbonResult(
            selected=tuple(None for _ in frame_candidates),
            states=tuple("prebirth" for _ in frame_candidates),
            candidate_indices=tuple(None for _ in frame_candidates),
            first_active_sample=None,
            total_score=0.0,
            score_margin=float("inf"),
            scales=scales,
        )
    eligible.sort(key=lambda item: item[1], reverse=True)
    final_key, final_score = eligible[0]
    margin = (
        float("inf")
        if len(eligible) == 1
        else float(final_score - eligible[1][1])
    )

    keys: list[tuple] = [final_key]
    for frame_index in range(frame_count - 1, 0, -1):
        previous = predecessors[frame_index][keys[-1]]
        if previous is None:
            previous = ("prebirth",)
        keys.append(previous)
    keys.reverse()

    selected = []
    states = []
    indices = []
    first_active = None
    for frame_index, key in enumerate(keys):
        state = key[0]
        states.append(state)
        if state == "path":
            candidate_index = int(key[1])
            selected.append(frame_candidates[frame_index][candidate_index])
            indices.append(candidate_index)
            if first_active is None:
                first_active = frame_index
        else:
            selected.append(None)
            indices.append(None)
    return GlobalRibbonResult(
        selected=tuple(selected),
        states=tuple(states),
        candidate_indices=tuple(indices),
        first_active_sample=first_active,
        total_score=float(final_score),
        score_margin=margin,
        scales=scales,
    )
