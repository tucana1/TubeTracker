"""Trace growing brightfield tubes in position-orientation space.

A planar binary mask turns every projected crossing into a graph junction.  An
orientation score instead represents a point as ``(row, column, direction)``;
two crossing tubes can occupy the same image pixel while remaining separate in
the lifted space.  This module combines that representation with paired-wall
brightfield evidence and a material-coordinate prior that permits growth only
beyond the previous distal end.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class OrientationScoreConfig:
    """Configure paired-wall evidence over undirected tube orientations."""

    orientation_count: int = 24
    half_widths_px: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5)
    tangent_samples_px: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    wall_sigma_px: float = 0.65
    background_sigma_px: float = 4.0
    structure_sigma_px: float = 1.2
    orientation_concentration: float = 4.0
    center_darkness_penalty: float = 0.18
    asymmetry_penalty: float = 0.30
    oriented_ridge_weight: float = 0.70
    bright_material_weight: float = 0.85
    normalization_percentile: float = 99.0


@dataclass(frozen=True)
class OrientationTraceConfig:
    """Configure a forward, curvature-limited search in lifted space."""

    step_px: float = 1.5
    maximum_turn_bins: int = 1
    curvature_penalty: float = 0.055
    evidence_floor: float = 0.18
    length_reward: float = 0.015
    minimum_support: float = 0.12
    maximum_gap_steps: int = 5
    maximum_initial_gap_steps: int = 5
    root_occlusion_px: float = 0.0
    beam_width: int = 1200
    minimum_length_px: float = 4.0
    maximum_length_px: float = 160.0
    maximum_extension_px: float = 14.0
    prior_position_penalty: float = 0.12
    prior_direction_penalty: float = 0.08
    prior_release_px: float = 3.0
    extension_direction_penalty: float = 0.45
    extension_position_penalty: float = 0.08
    extension_memory_px: float = 20.0
    birth_backtrack_tolerance: float = 1.0
    birth_backtrack_penalty: float = 0.30
    birth_progress_reward: float = 0.025
    birth_delta_clip: float = 8.0
    birth_floor_start_px: float = 5.0


@dataclass(frozen=True)
class OrientationTraceResult:
    """Store one directed centerline and its pointwise optical support."""

    path_yx: np.ndarray
    direction_radians: np.ndarray
    support: np.ndarray
    score: float
    mean_support: float
    supported_fraction: float
    prior_prefix_error_px: float

    @property
    def length_px(self) -> float:
        """Return centerline arclength in pixels."""

        if len(self.path_yx) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.path_yx, axis=0), axis=1).sum())


@dataclass(frozen=True)
class PollenRootProposal:
    """Describe one outward tube direction proposed on a pollen rim."""

    root_yx: np.ndarray
    direction_yx: np.ndarray
    attachment_angle_radians: float
    direction_angle_radians: float
    support: float
    birth_time: float


@dataclass(frozen=True)
class OrientationChangePointResult:
    """Store persistent appearance evidence and its inferred first sample."""

    evidence: np.ndarray
    birth_sample: np.ndarray
    appearance_change: np.ndarray


@dataclass(frozen=True)
class PairedWallOrientationResult:
    """Store merged and explicitly two-sided tube evidence by orientation."""

    score: np.ndarray
    paired_score: np.ndarray
    half_width_px: np.ndarray
    wall_balance: np.ndarray


@dataclass(frozen=True)
class RibbonPathCertificate:
    """Summarize two-sided wall support and width consistency along one path."""

    paired_mean_support: float
    paired_supported_fraction: float
    mean_wall_balance: float
    median_half_width_px: float
    width_mad_px: float
    maximum_width_step_px: float


@dataclass(frozen=True)
class CoupledRibbonTraceConfig:
    """Configure a width-aware lifted search for one pollen-owned ribbon."""

    step_px: float = 1.5
    maximum_turn_bins: int = 1
    maximum_total_turn_bins: int | None = None
    curvature_penalty: float = 0.08
    minimum_pair_support: float = 0.08
    minimum_wall_balance: float = 0.20
    paired_support_weight: float = 1.0
    merged_support_weight: float = 0.15
    wall_balance_weight: float = 0.05
    evidence_floor: float = 0.10
    length_reward: float = 0.012
    width_change_penalty: float = 0.08
    maximum_reacquisition_width_change_px: float = 2.0
    width_state_step_px: float = 0.5
    width_update_blend: float = 0.20
    maximum_gap_steps: int = 6
    maximum_initial_gap_steps: int = 5
    root_occlusion_px: float = 0.0
    beam_width: int = 1600
    minimum_length_px: float = 12.0
    maximum_length_px: float = 180.0
    minimum_endpoint_separation_fraction: float = 0.25
    endpoint_openness_reward: float = 0.12
    maximum_extension_px: float = 14.0
    prior_position_penalty: float = 0.14
    prior_direction_penalty: float = 0.08
    prior_release_px: float = 2.0
    extension_direction_penalty: float = 0.45
    extension_position_penalty: float = 0.08
    extension_memory_px: float = 20.0
    birth_backtrack_tolerance: float = 1.0
    birth_backtrack_penalty: float = 0.30
    birth_progress_reward: float = 0.025
    birth_delta_clip: float = 8.0
    birth_floor_start_px: float = 5.0
    maximum_alternative_paths: int = 0
    alternative_tip_separation_px: float = 4.0
    alternative_prefix_separation_px: float = 1.5


@dataclass(frozen=True)
class CoupledRibbonTraceResult:
    """Store one centerline together with its persistent two-wall state."""

    path_yx: np.ndarray
    direction_radians: np.ndarray
    half_width_px: np.ndarray
    paired_support: np.ndarray
    merged_support: np.ndarray
    wall_balance: np.ndarray
    birth_time: np.ndarray
    score: float
    paired_supported_fraction: float
    endpoint_separation_fraction: float
    prior_prefix_error_px: float
    alternatives: tuple[CoupledRibbonTraceResult, ...] = ()

    @property
    def length_px(self) -> float:
        """Return centerline arclength in pixels."""

        if len(self.path_yx) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.path_yx, axis=0), axis=1).sum())


@dataclass(frozen=True)
class _SearchNode:
    """Retain one beam-search state and its predecessor."""

    y: float
    x: float
    direction_bin: int
    score: float
    gap_steps: int
    has_material_support: bool
    parent: int
    support: float
    birth_time: float
    depth: int


@dataclass(frozen=True)
class _RibbonSearchNode:
    """Retain one width-aware lifted state and its predecessor."""

    y: float
    x: float
    direction_bin: int
    half_width_px: float
    score: float
    gap_steps: int
    has_paired_support: bool
    parent: int
    paired_support: float
    merged_support: float
    wall_balance: float
    birth_time: float
    depth: int


def _bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample one image at arbitrary coordinates with zero outside its bounds."""

    return cv.remap(
        np.asarray(image, dtype=np.float32),
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _normalized_darkness(gray: np.ndarray, config: OrientationScoreConfig) -> np.ndarray:
    """Measure narrow material without assuming one phase-contrast polarity."""

    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray must be a two-dimensional image")
    fine = cv.GaussianBlur(image, (0, 0), config.wall_sigma_px)
    background = cv.GaussianBlur(image, (0, 0), config.background_sigma_px)
    dark_material = np.maximum(background - fine, 0.0)
    bright_material = np.maximum(fine - background, 0.0)
    darkness = np.maximum(
        dark_material,
        config.bright_material_weight * bright_material,
    )
    positive = darkness[darkness > 0.0]
    if not len(positive):
        return np.zeros_like(darkness)
    scale = float(np.percentile(positive, config.normalization_percentile))
    return np.clip(darkness / max(scale, 1e-6), 0.0, 1.0)


def paired_wall_orientation_features(
    gray: np.ndarray,
    config: OrientationScoreConfig | None = None,
) -> PairedWallOrientationResult:
    """Lift one frame while preserving wall pairing and inferred tube width."""

    config = config or OrientationScoreConfig()
    if config.orientation_count < 4:
        raise ValueError("orientation_count must be at least four")
    if not config.half_widths_px or not config.tangent_samples_px:
        raise ValueError("wall widths and tangent samples cannot be empty")
    darkness = _normalized_darkness(gray, config)
    height, width = darkness.shape
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    scores = np.zeros((config.orientation_count, height, width), dtype=np.float32)
    paired_scores = np.zeros_like(scores)
    half_widths = np.zeros_like(scores)
    wall_balances = np.zeros_like(scores)
    tangent_offsets = np.asarray(config.tangent_samples_px, dtype=np.float32)
    gradient_x = cv.Sobel(darkness, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(darkness, cv.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv.GaussianBlur(
        gradient_x * gradient_x,
        (0, 0),
        config.structure_sigma_px,
    )
    tensor_xy = cv.GaussianBlur(
        gradient_x * gradient_y,
        (0, 0),
        config.structure_sigma_px,
    )
    tensor_yy = cv.GaussianBlur(
        gradient_y * gradient_y,
        (0, 0),
        config.structure_sigma_px,
    )
    principal_gradient = 0.5 * np.arctan2(
        2.0 * tensor_xy,
        tensor_xx - tensor_yy,
    )
    tangent_orientation = principal_gradient + 0.5 * math.pi
    anisotropy = np.sqrt(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2
    ) / np.maximum(tensor_xx + tensor_yy, 1e-6)

    for orientation_index in range(config.orientation_count):
        theta = math.pi * orientation_index / config.orientation_count
        tangent_y, tangent_x = math.sin(theta), math.cos(theta)
        normal_y, normal_x = -tangent_x, tangent_y
        alignment = np.exp(
            config.orientation_concentration
            * (np.cos(2.0 * (theta - tangent_orientation)) - 1.0)
        )
        oriented_darkness = darkness * alignment * (0.25 + 0.75 * anisotropy)
        raw_center_samples = []
        oriented_center_samples = []
        for along in tangent_offsets:
            sample_y = grid_y + tangent_y * along
            sample_x = grid_x + tangent_x * along
            raw_center_samples.append(_bilinear_sample(darkness, sample_y, sample_x))
            oriented_center_samples.append(
                _bilinear_sample(oriented_darkness, sample_y, sample_x)
            )
        center = np.mean(raw_center_samples, axis=0)
        oriented_center = np.mean(oriented_center_samples, axis=0)
        best = config.oriented_ridge_weight * oriented_center
        best_paired = np.zeros_like(center)
        best_half_width = np.zeros_like(center)
        best_wall_balance = np.zeros_like(center)
        for half_width in config.half_widths_px:
            left_samples = []
            right_samples = []
            for along in tangent_offsets:
                base_y = grid_y + tangent_y * along
                base_x = grid_x + tangent_x * along
                left_samples.append(
                    _bilinear_sample(
                        oriented_darkness,
                        base_y - normal_y * half_width,
                        base_x - normal_x * half_width,
                    )
                )
                right_samples.append(
                    _bilinear_sample(
                        oriented_darkness,
                        base_y + normal_y * half_width,
                        base_x + normal_x * half_width,
                    )
                )
            left = np.mean(left_samples, axis=0)
            right = np.mean(right_samples, axis=0)
            paired = np.sqrt(np.maximum(left * right, 0.0))
            asymmetry = np.abs(left - right)
            response = (
                paired
                - config.center_darkness_penalty * center
                - config.asymmetry_penalty * asymmetry
            )
            improved = response > best_paired
            best_paired[improved] = response[improved]
            best_half_width[improved] = half_width
            balance = 1.0 - asymmetry / np.maximum(left + right, 1e-6)
            best_wall_balance[improved] = np.clip(balance[improved], 0.0, 1.0)
            best = np.maximum(best, response)
        scores[orientation_index] = np.maximum(best, 0.0)
        paired_scores[orientation_index] = np.maximum(best_paired, 0.0)
        half_widths[orientation_index] = best_half_width
        wall_balances[orientation_index] = best_wall_balance

    positive = scores[scores > 0.0]
    if len(positive):
        scale = float(np.percentile(positive, config.normalization_percentile))
        scores = np.clip(scores / max(scale, 1e-6), 0.0, 1.0)
        paired_scores = np.clip(
            paired_scores / max(scale, 1e-6),
            0.0,
            1.0,
        )
    return PairedWallOrientationResult(
        score=scores,
        paired_score=paired_scores,
        half_width_px=half_widths,
        wall_balance=wall_balances,
    )


def paired_wall_orientation_score(
    gray: np.ndarray,
    config: OrientationScoreConfig | None = None,
) -> np.ndarray:
    """Return merged paired-wall and ridge evidence for compatibility."""

    return paired_wall_orientation_features(gray, config).score


def ribbon_path_certificate(
    features: PairedWallOrientationResult,
    path_yx: np.ndarray,
    direction_radians: np.ndarray,
    *,
    support_threshold: float = 0.12,
    proximal_occlusion_px: float = 0.0,
) -> RibbonPathCertificate:
    """Summarize paired walls along an oriented path without flattening width."""

    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support threshold must lie between zero and one")
    if proximal_occlusion_px < 0.0:
        raise ValueError("proximal occlusion cannot be negative")
    path = np.asarray(path_yx, dtype=np.float64)
    directions = np.asarray(direction_radians, dtype=np.float64)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2:
        raise ValueError("path must contain at least two row-column points")
    if directions.shape != (len(path),):
        raise ValueError("directions must contain one angle per path point")
    arrays = (
        features.score,
        features.paired_score,
        features.half_width_px,
        features.wall_balance,
    )
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("ribbon features must have orientation-row-column shape")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("ribbon feature arrays must share shape")
    if not np.isfinite(path).all() or not np.isfinite(directions).all():
        raise ValueError("path and directions must be finite")

    orientation_count = features.score.shape[0]
    bins = (
        np.rint(np.mod(directions, math.pi) * orientation_count / math.pi)
        .astype(int)
        % orientation_count
    )

    def sample(volume: np.ndarray) -> np.ndarray:
        values = np.zeros(len(path), dtype=np.float32)
        for orientation in np.unique(bins):
            indices = np.flatnonzero(bins == orientation)
            values[indices] = _bilinear_sample(
                volume[int(orientation)],
                path[indices, 0],
                path[indices, 1],
            )
        return values

    paired = sample(features.paired_score)
    widths = sample(features.half_width_px)
    balances = sample(features.wall_balance)
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    evaluated = arc >= proximal_occlusion_px
    if not np.any(evaluated):
        raise ValueError("proximal occlusion excludes the complete path")
    supported = (paired >= support_threshold) & evaluated
    supported_widths = widths[supported & (widths > 0.0)]
    if len(supported_widths):
        median_width = float(np.median(supported_widths))
        width_mad = float(np.median(np.abs(supported_widths - median_width)))
        maximum_width_step = (
            float(np.max(np.abs(np.diff(supported_widths))))
            if len(supported_widths) > 1
            else 0.0
        )
    else:
        median_width = 0.0
        width_mad = 0.0
        maximum_width_step = 0.0
    return RibbonPathCertificate(
        paired_mean_support=float(np.mean(paired[evaluated])),
        paired_supported_fraction=float(np.mean(supported[evaluated])),
        mean_wall_balance=float(np.mean(balances[supported]))
        if np.any(supported)
        else 0.0,
        median_half_width_px=median_width,
        width_mad_px=width_mad,
        maximum_width_step_px=maximum_width_step,
    )


def aggregate_paired_wall_history(
    score_stack: np.ndarray,
    paired_stack: np.ndarray,
    half_width_stack: np.ndarray,
    wall_balance_stack: np.ndarray,
    *,
    start_index: int = 0,
    percentile: float = 70.0,
) -> PairedWallOrientationResult:
    """Fuse late video evidence while preserving supported ribbon width."""

    arrays = tuple(
        np.asarray(array, dtype=np.float32)
        for array in (
            score_stack,
            paired_stack,
            half_width_stack,
            wall_balance_stack,
        )
    )
    if any(array.ndim != 4 for array in arrays):
        raise ValueError("ribbon histories must have time-orientation-image shape")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("ribbon histories must share shape")
    if not 0 <= start_index < len(arrays[0]):
        raise ValueError("start_index must lie inside the ribbon history")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie between zero and one hundred")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("ribbon histories must be finite")
    scores, paired, widths, balances = (
        array[start_index:] for array in arrays
    )
    aggregate_score = np.percentile(scores, percentile, axis=0).astype(np.float32)
    aggregate_pair = np.percentile(paired, percentile, axis=0).astype(np.float32)
    weight = np.maximum(paired, 0.0)
    denominator = np.sum(weight, axis=0)
    aggregate_width = np.divide(
        np.sum(weight * widths, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-6,
    ).astype(np.float32)
    aggregate_balance = np.divide(
        np.sum(weight * balances, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-6,
    ).astype(np.float32)
    return PairedWallOrientationResult(
        score=aggregate_score,
        paired_score=aggregate_pair,
        half_width_px=aggregate_width,
        wall_balance=aggregate_balance,
    )


def persistent_orientation_birth(
    score_stack: np.ndarray,
    warmup_samples: int,
    persistence_samples: int = 3,
    minimum_change: float = 0.10,
    noise_multiplier: float = 4.0,
) -> np.ndarray:
    """Estimate first persistent appearance independently in each orientation."""

    scores = np.asarray(score_stack, dtype=np.float32)
    if scores.ndim != 4:
        raise ValueError(
            "score_stack must have shape (time, orientations, height, width)"
        )
    if not 1 <= warmup_samples < len(scores):
        raise ValueError("warmup_samples must be inside the score timeline")
    if persistence_samples < 1:
        raise ValueError("persistence_samples must be positive")
    baseline = np.median(scores[:warmup_samples], axis=0)
    deviations = np.abs(scores[:warmup_samples] - baseline[None, ...])
    noise = 1.4826 * np.median(deviations, axis=0)
    threshold = baseline + np.maximum(minimum_change, noise_multiplier * noise)
    active = scores >= threshold[None, ...]
    birth = np.full(scores.shape[1:], len(scores), dtype=np.float32)
    preexisting = np.mean(
        scores[:warmup_samples] >= np.maximum(0.12, baseline)[None, ...],
        axis=0,
    ) >= 0.75
    birth[preexisting] = 0.0
    if persistence_samples == 1:
        persistent = active
    else:
        cumulative = np.cumsum(active.astype(np.int16), axis=0)
        window = cumulative[persistence_samples - 1 :].copy()
        if persistence_samples < len(scores):
            window[1:] -= cumulative[:-persistence_samples]
        persistent = window >= persistence_samples
    first = np.argmax(persistent, axis=0)
    has_birth = np.any(persistent, axis=0) & ~preexisting
    birth[has_birth] = first[has_birth] + persistence_samples - 1
    return birth


def causal_orientation_changepoint(
    score_stack: np.ndarray,
    minimum_before_samples: int = 2,
    minimum_after_samples: int = 3,
    minimum_change: float = 0.04,
) -> OrientationChangePointResult:
    """Infer persistent oriented-material appearance without a warm-up window.

    Every admissible split of the full timeline competes as the appearance
    time.  A lasting increase receives a large before/after contrast, whereas
    static structures and short flashes do not.  This makes onset observability
    depend on actual retained samples rather than an arbitrary movie fraction.
    """

    scores = np.asarray(score_stack, dtype=np.float32)
    if scores.ndim != 4:
        raise ValueError(
            "score_stack must have shape (time, orientations, height, width)"
        )
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("score_stack must be finite and nonnegative")
    if minimum_before_samples < 1 or minimum_after_samples < 1:
        raise ValueError("change-point segments must contain at least one sample")
    if minimum_before_samples + minimum_after_samples > len(scores):
        raise ValueError("timeline is too short for requested change-point segments")
    if minimum_change < 0.0:
        raise ValueError("minimum_change cannot be negative")

    total = np.sum(scores, axis=0, dtype=np.float32)
    prefix = np.zeros_like(total)
    best_change = np.zeros_like(total)
    best_birth = np.full(scores.shape[1:], len(scores), dtype=np.float32)
    for split in range(1, len(scores)):
        prefix += scores[split - 1]
        if split < minimum_before_samples:
            continue
        after_count = len(scores) - split
        if after_count < minimum_after_samples:
            break
        before_mean = prefix / float(split)
        after_mean = (total - prefix) / float(after_count)
        change = after_mean - before_mean
        improved = change > best_change
        best_change[improved] = change[improved]
        best_birth[improved] = float(split)

    significant = best_change >= minimum_change
    best_birth[~significant] = float(len(scores))
    evidence = np.where(significant, best_change, 0.0).astype(np.float32)
    positive = evidence[evidence > 0.0]
    if len(positive):
        scale = float(np.percentile(positive, 99.0))
        evidence = np.clip(evidence / max(scale, 1e-6), 0.0, 1.0)
    return OrientationChangePointResult(
        evidence=evidence,
        birth_sample=best_birth,
        appearance_change=best_change,
    )


def persistent_orientation_score(
    score_stack: np.ndarray,
    warmup_samples: int,
    persistence_quantile: float = 70.0,
    tail_fraction: float = 0.15,
) -> np.ndarray:
    """Fuse a movie into evidence that favors material which appears and remains.

    The temporal quantile suppresses brief particles and compression artifacts,
    while the late median preserves faint material that remains after growth.
    Positive change from the early baseline prevents static neighboring tubes from
    dominating merely because they are darker.
    """

    scores = np.asarray(score_stack, dtype=np.float32)
    if scores.ndim != 4:
        raise ValueError(
            "score_stack must have shape (time, orientations, height, width)"
        )
    if not 1 <= warmup_samples < len(scores):
        raise ValueError("warmup_samples must be inside the score timeline")
    if not 0.0 <= persistence_quantile <= 100.0:
        raise ValueError("persistence_quantile must be between zero and 100")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")

    baseline = np.median(scores[:warmup_samples], axis=0)
    post_warmup = scores[warmup_samples:]
    change = np.maximum(post_warmup - baseline[None, ...], 0.0)
    persistent_change = np.percentile(
        change,
        persistence_quantile,
        axis=0,
    )
    tail_count = min(
        len(scores),
        max(7, int(math.ceil(len(scores) * tail_fraction))),
    )
    tail_change = np.median(
        np.maximum(scores[-tail_count:] - baseline[None, ...], 0.0),
        axis=0,
    )
    persistent_absolute = np.percentile(
        post_warmup,
        persistence_quantile,
        axis=0,
    )
    fused = (
        0.55 * persistent_change
        + 0.30 * tail_change
        + 0.15 * persistent_absolute
    ).astype(np.float32)
    positive = fused[fused > 0.0]
    if len(positive):
        scale = float(np.percentile(positive, 99.0))
        fused = np.clip(fused / max(scale, 1e-6), 0.0, 1.0)
    return fused


def _curve_arclength(curve_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength for an ordered row-column curve."""

    curve = np.asarray(curve_yx, dtype=np.float64)
    if len(curve) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
    )


def _curve_point_and_direction(
    curve_yx: np.ndarray,
    arclength: np.ndarray,
    query: float,
) -> tuple[np.ndarray, float]:
    """Interpolate a material point and local direction on an ordered curve."""

    value = float(np.clip(query, 0.0, arclength[-1]))
    point = np.asarray(
        [np.interp(value, arclength, curve_yx[:, axis]) for axis in range(2)]
    )
    before = max(0.0, value - 1.5)
    after = min(float(arclength[-1]), value + 1.5)
    start = np.asarray(
        [np.interp(before, arclength, curve_yx[:, axis]) for axis in range(2)]
    )
    end = np.asarray(
        [np.interp(after, arclength, curve_yx[:, axis]) for axis in range(2)]
    )
    vector = end - start
    return point, math.atan2(float(vector[0]), float(vector[1]))


def _wrapped_angle_difference(first: float, second: float) -> float:
    """Return the smallest absolute directed-angle difference in radians."""

    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _support_at(
    orientation_score: np.ndarray,
    y: float,
    x: float,
    direction_bin: int,
) -> float:
    """Sample undirected evidence for one directed lifted state."""

    value = _state_value_at(orientation_score, y, x, direction_bin)
    return value if np.isfinite(value) else 0.0


def _state_value_at(
    volume: np.ndarray,
    y: float,
    x: float,
    direction_bin: int,
) -> float:
    """Sample one undirected orientation volume for a directed state."""

    orientation_count, height, width = volume.shape
    if y < 0.0 or x < 0.0 or y > height - 1 or x > width - 1:
        return float("nan")
    orientation_index = direction_bin % orientation_count
    image = volume[orientation_index]
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, height - 1)
    x1 = min(x0 + 1, width - 1)
    dy = y - y0
    dx = x - x0
    return float(
        (1.0 - dy) * (1.0 - dx) * image[y0, x0]
        + (1.0 - dy) * dx * image[y0, x1]
        + dy * (1.0 - dx) * image[y1, x0]
        + dy * dx * image[y1, x1]
    )


def propose_pollen_roots(
    orientation_score: np.ndarray,
    pollen_center_yx: tuple[float, float] | np.ndarray,
    pollen_radius_px: float,
    birth_time: np.ndarray | None = None,
    minimum_material_birth: float | None = None,
    attachment_count: int = 48,
    direction_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
    probe_distances_px: tuple[float, ...] = (1.0, 2.5, 4.0, 5.5, 7.0),
    probe_top_k: int | None = None,
    maximum_proposals: int = 8,
    minimum_angle_separation_degrees: float = 20.0,
) -> tuple[PollenRootProposal, ...]:
    """Find new directionally coherent material around the complete pollen rim."""

    evidence = np.asarray(orientation_score, dtype=np.float32)
    if evidence.ndim != 3:
        raise ValueError("orientation_score must be three-dimensional")
    center = np.asarray(pollen_center_yx, dtype=np.float64)
    if center.shape != (2,) or pollen_radius_px <= 0:
        raise ValueError("pollen center and radius must describe one circle")
    births = None if birth_time is None else np.asarray(birth_time, dtype=np.float32)
    if births is not None and births.shape != evidence.shape:
        raise ValueError("birth_time must match orientation_score")
    if minimum_material_birth is not None and births is None:
        raise ValueError("minimum_material_birth requires birth_time")
    if probe_top_k is not None and not 1 <= probe_top_k <= len(probe_distances_px):
        raise ValueError("probe_top_k must fit inside probe_distances_px")

    directed_count = 2 * evidence.shape[0]
    angle_step = 2.0 * math.pi / directed_count
    candidates = []
    for attachment_index in range(attachment_count):
        attachment_angle = 2.0 * math.pi * attachment_index / attachment_count
        radial = np.asarray(
            (math.sin(attachment_angle), math.cos(attachment_angle)),
            dtype=np.float64,
        )
        root = center + (pollen_radius_px + 0.75) * radial
        radial_bin = int(round(attachment_angle / angle_step)) % directed_count
        for offset in direction_offsets:
            direction_bin = (radial_bin + offset) % directed_count
            direction_angle = direction_bin * angle_step
            direction = np.asarray(
                (math.sin(direction_angle), math.cos(direction_angle)),
                dtype=np.float64,
            )
            supports = []
            material_births = []
            for distance in probe_distances_px:
                point = root + distance * direction
                supports.append(
                    _support_at(evidence, point[0], point[1], direction_bin)
                )
                if births is not None:
                    material_births.append(
                        _state_value_at(
                            births,
                            point[0],
                            point[1],
                            direction_bin,
                        )
                    )
            finite_births = [value for value in material_births if np.isfinite(value)]
            candidate_birth = (
                float(np.median(finite_births)) if finite_births else float("nan")
            )
            if (
                minimum_material_birth is not None
                and np.isfinite(candidate_birth)
                and candidate_birth < minimum_material_birth - 1.0
            ):
                continue
            ranked_supports = sorted(supports, reverse=True)
            retained_supports = (
                ranked_supports[:probe_top_k]
                if probe_top_k is not None
                else ranked_supports
            )
            candidates.append(
                PollenRootProposal(
                    root_yx=root,
                    direction_yx=direction,
                    attachment_angle_radians=attachment_angle,
                    direction_angle_radians=direction_angle,
                    support=float(np.mean(retained_supports)),
                    birth_time=candidate_birth,
                )
            )

    ranked = sorted(candidates, key=lambda candidate: candidate.support, reverse=True)
    selected = []
    minimum_separation = math.radians(minimum_angle_separation_degrees)
    for candidate in ranked:
        if any(
            _wrapped_angle_difference(
                candidate.attachment_angle_radians,
                previous.attachment_angle_radians,
            )
            < minimum_separation
            for previous in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum_proposals:
            break
    return tuple(selected)


def trace_orientation_lifted(
    orientation_score: np.ndarray,
    root_yx: tuple[float, float] | np.ndarray,
    initial_direction_yx: tuple[float, float] | np.ndarray,
    prior_curve_yx: np.ndarray | None = None,
    birth_time: np.ndarray | None = None,
    minimum_material_birth: float | None = None,
    config: OrientationTraceConfig | None = None,
) -> OrientationTraceResult:
    """Trace one root-owned tube without collapsing projected crossings."""

    config = config or OrientationTraceConfig()
    evidence = np.asarray(orientation_score, dtype=np.float32)
    if evidence.ndim != 3 or evidence.shape[0] < 4:
        raise ValueError("orientation_score must have shape (orientations, height, width)")
    root = np.asarray(root_yx, dtype=np.float64)
    initial = np.asarray(initial_direction_yx, dtype=np.float64)
    if root.shape != (2,) or initial.shape != (2,):
        raise ValueError("root and initial direction must each contain row and column")
    if float(np.linalg.norm(initial)) <= 1e-9:
        raise ValueError("initial direction cannot be zero")
    if config.step_px <= 0 or config.beam_width < 1:
        raise ValueError("step_px and beam_width must be positive")
    if config.maximum_gap_steps < 0 or config.maximum_initial_gap_steps < 0:
        raise ValueError("gap limits cannot be negative")
    if config.root_occlusion_px < 0.0:
        raise ValueError("root occlusion cannot be negative")
    births = None
    if birth_time is not None:
        births = np.asarray(birth_time, dtype=np.float32)
        if births.shape != evidence.shape:
            raise ValueError("birth_time must match orientation_score")
    if minimum_material_birth is not None and births is None:
        raise ValueError("minimum_material_birth requires birth_time")

    undirected_count = evidence.shape[0]
    directed_count = 2 * undirected_count
    initial_angle = math.atan2(float(initial[0]), float(initial[1])) % (2.0 * math.pi)
    initial_bin = int(round(initial_angle * directed_count / (2.0 * math.pi))) % directed_count

    prior = None
    prior_arc = None
    prior_length = 0.0
    prior_tip = None
    prior_tip_direction = None
    if prior_curve_yx is not None:
        prior = np.asarray(prior_curve_yx, dtype=np.float64)
        if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 2:
            raise ValueError("prior_curve_yx must have shape (points, 2)")
        prior = prior + (root - prior[0])
        prior_arc = _curve_arclength(prior)
        prior_length = float(prior_arc[-1])
        prior_tip, prior_tip_direction = _curve_point_and_direction(
            prior,
            prior_arc,
            prior_length,
        )

    maximum_length = float(config.maximum_length_px)
    if prior is not None:
        maximum_length = min(maximum_length, prior_length + config.maximum_extension_px)
    maximum_depth = max(1, int(math.ceil(maximum_length / config.step_px)))
    minimum_depth = max(1, int(math.ceil(config.minimum_length_px / config.step_px)))
    root_support = _support_at(evidence, root[0], root[1], initial_bin)
    root_birth = (
        _state_value_at(births, root[0], root[1], initial_bin)
        if births is not None
        else float("nan")
    )
    nodes = [
        _SearchNode(
            y=float(root[0]),
            x=float(root[1]),
            direction_bin=initial_bin,
            score=0.0,
            gap_steps=0,
            has_material_support=False,
            parent=-1,
            support=root_support,
            birth_time=root_birth,
            depth=0,
        )
    ]
    beam = [0]
    endpoint_indices: list[int] = []
    angle_step = 2.0 * math.pi / directed_count

    for depth in range(1, maximum_depth + 1):
        candidates: dict[
            tuple[int, int, int, int, bool],
            tuple[float, float, int, float, int, bool, int, float, float],
        ] = {}
        material_position = depth * config.step_px
        for parent_index in beam:
            parent = nodes[parent_index]
            for turn in range(-config.maximum_turn_bins, config.maximum_turn_bins + 1):
                direction_bin = (parent.direction_bin + turn) % directed_count
                theta = angle_step * direction_bin
                y = parent.y + config.step_px * math.sin(theta)
                x = parent.x + config.step_px * math.cos(theta)
                support = _support_at(evidence, y, x, direction_bin)
                state_birth = (
                    _state_value_at(births, y, x, direction_bin)
                    if births is not None
                    else float("nan")
                )
                if (
                    minimum_material_birth is not None
                    and material_position >= config.birth_floor_start_px
                    and np.isfinite(state_birth)
                    and state_birth
                    < minimum_material_birth - config.birth_backtrack_tolerance
                ):
                    continue
                inside_root_occlusion = material_position <= config.root_occlusion_px
                supported = support >= config.minimum_support
                has_material_support = parent.has_material_support or (
                    supported and not inside_root_occlusion
                )
                if inside_root_occlusion:
                    gap_steps = 0
                    increment = config.length_reward
                else:
                    gap_steps = parent.gap_steps + 1 if not supported else 0
                    allowed_gap = (
                        config.maximum_gap_steps
                        if parent.has_material_support
                        else config.maximum_initial_gap_steps
                    )
                    if gap_steps > allowed_gap:
                        continue
                    increment = support - config.evidence_floor + config.length_reward
                increment -= config.curvature_penalty * abs(turn)
                if (
                    births is not None
                    and np.isfinite(parent.birth_time)
                    and np.isfinite(state_birth)
                ):
                    delta_birth = state_birth - parent.birth_time
                    if delta_birth < -config.birth_backtrack_tolerance:
                        increment -= config.birth_backtrack_penalty * min(
                            -delta_birth - config.birth_backtrack_tolerance,
                            config.birth_delta_clip,
                        )
                    elif delta_birth > 0.0:
                        increment += config.birth_progress_reward * min(
                            delta_birth,
                            config.birth_delta_clip,
                        )
                if prior is not None and material_position <= prior_length - config.prior_release_px:
                    prior_point, prior_direction = _curve_point_and_direction(
                        prior,
                        prior_arc,
                        material_position,
                    )
                    increment -= config.prior_position_penalty * float(
                        np.linalg.norm(np.asarray((y, x)) - prior_point)
                    )
                    increment -= config.prior_direction_penalty * _wrapped_angle_difference(
                        theta,
                        prior_direction,
                    )
                elif prior is not None:
                    extension = max(0.0, material_position - prior_length)
                    memory = math.exp(
                        -extension / max(config.extension_memory_px, 1e-6)
                    )
                    tip_vector = np.asarray(
                        (
                            math.sin(prior_tip_direction),
                            math.cos(prior_tip_direction),
                        )
                    )
                    predicted = prior_tip + extension * tip_vector
                    increment -= (
                        config.extension_position_penalty
                        * memory
                        * float(np.linalg.norm(np.asarray((y, x)) - predicted))
                    )
                    increment -= (
                        config.extension_direction_penalty
                        * memory
                        * _wrapped_angle_difference(theta, prior_tip_direction)
                    )
                score = parent.score + increment
                key = (
                    int(round(y)),
                    int(round(x)),
                    direction_bin,
                    gap_steps,
                    has_material_support,
                )
                previous = candidates.get(key)
                if previous is None or score > previous[3]:
                    candidates[key] = (
                        y,
                        x,
                        direction_bin,
                        score,
                        gap_steps,
                        has_material_support,
                        parent_index,
                        support,
                        state_birth,
                    )
        if not candidates:
            break
        ranked = sorted(candidates.values(), key=lambda item: item[3], reverse=True)
        beam = []
        for (
            y,
            x,
            direction_bin,
            score,
            gap_steps,
            has_material_support,
            parent_index,
            support,
            state_birth,
        ) in ranked[: config.beam_width]:
            beam.append(len(nodes))
            nodes.append(
                _SearchNode(
                    y=y,
                    x=x,
                    direction_bin=direction_bin,
                    score=score,
                    gap_steps=gap_steps,
                    has_material_support=has_material_support,
                    parent=parent_index,
                    support=support,
                    birth_time=state_birth,
                    depth=depth,
                )
            )
        if depth >= minimum_depth:
            endpoint_indices.extend(beam[: min(32, len(beam))])

    if not endpoint_indices:
        endpoint_indices = beam
    if not endpoint_indices:
        return OrientationTraceResult(
            path_yx=root[None, :],
            direction_radians=np.asarray([initial_angle]),
            support=np.asarray([root_support]),
            score=0.0,
            mean_support=root_support,
            supported_fraction=float(root_support >= config.minimum_support),
            prior_prefix_error_px=0.0,
        )

    endpoint = max(endpoint_indices, key=lambda index: nodes[index].score)
    chain = []
    while endpoint >= 0:
        chain.append(endpoint)
        endpoint = nodes[endpoint].parent
    chain.reverse()
    path = np.asarray([(nodes[index].y, nodes[index].x) for index in chain])
    directions = np.asarray(
        [angle_step * nodes[index].direction_bin for index in chain],
        dtype=np.float64,
    )
    support = np.asarray([nodes[index].support for index in chain], dtype=np.float64)
    prior_error = 0.0
    if prior is not None:
        path_arc = _curve_arclength(path)
        shared = min(float(path_arc[-1]), prior_length)
        samples = np.linspace(0.0, shared, max(2, int(math.ceil(shared)) + 1))
        path_points = np.column_stack(
            [np.interp(samples, path_arc, path[:, axis]) for axis in range(2)]
        )
        prior_points = np.column_stack(
            [np.interp(samples, prior_arc, prior[:, axis]) for axis in range(2)]
        )
        prior_error = float(np.mean(np.linalg.norm(path_points - prior_points, axis=1)))
    final = nodes[chain[-1]]
    return OrientationTraceResult(
        path_yx=path,
        direction_radians=directions,
        support=support,
        score=float(final.score),
        mean_support=float(np.mean(support[1:])) if len(support) > 1 else float(support[0]),
        supported_fraction=float(np.mean(support[1:] >= config.minimum_support))
        if len(support) > 1
        else float(support[0] >= config.minimum_support),
        prior_prefix_error_px=prior_error,
    )


def trace_coupled_ribbon_lifted(
    features: PairedWallOrientationResult,
    root_yx: tuple[float, float] | np.ndarray,
    initial_direction_yx: tuple[float, float] | np.ndarray,
    *,
    prior_curve_yx: np.ndarray | None = None,
    birth_time: np.ndarray | None = None,
    minimum_material_birth: float | None = None,
    terminal_mask: np.ndarray | None = None,
    config: CoupledRibbonTraceConfig | None = None,
) -> CoupledRibbonTraceResult:
    """Trace one ribbon, optionally requiring it to end on a target mask."""

    config = config or CoupledRibbonTraceConfig()
    arrays = (
        np.asarray(features.score, dtype=np.float32),
        np.asarray(features.paired_score, dtype=np.float32),
        np.asarray(features.half_width_px, dtype=np.float32),
        np.asarray(features.wall_balance, dtype=np.float32),
    )
    if any(array.ndim != 3 or array.shape[0] < 4 for array in arrays):
        raise ValueError("ribbon features must have orientation-row-column shape")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("ribbon feature arrays must share shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("ribbon features must be finite")
    root = np.asarray(root_yx, dtype=np.float64)
    initial = np.asarray(initial_direction_yx, dtype=np.float64)
    if root.shape != (2,) or initial.shape != (2,):
        raise ValueError("root and initial direction must contain row and column")
    if float(np.linalg.norm(initial)) <= 1e-9:
        raise ValueError("initial direction cannot be zero")
    if config.step_px <= 0.0 or config.beam_width < 1:
        raise ValueError("step_px and beam_width must be positive")
    if config.width_state_step_px <= 0.0:
        raise ValueError("width_state_step_px must be positive")
    if not 0.0 <= config.width_update_blend <= 1.0:
        raise ValueError("width_update_blend must lie between zero and one")
    if config.maximum_gap_steps < 0 or config.maximum_initial_gap_steps < 0:
        raise ValueError("gap limits cannot be negative")
    if (
        config.maximum_total_turn_bins is not None
        and config.maximum_total_turn_bins < 0
    ):
        raise ValueError("maximum_total_turn_bins cannot be negative")
    if config.maximum_reacquisition_width_change_px < 0.0:
        raise ValueError("reacquisition width tolerance cannot be negative")
    if not 0.0 <= config.minimum_endpoint_separation_fraction <= 1.0:
        raise ValueError("endpoint separation fraction must lie between zero and one")
    if config.maximum_extension_px < 0.0 or config.prior_release_px < 0.0:
        raise ValueError("prior extension and release distances cannot be negative")
    if config.maximum_alternative_paths < 0:
        raise ValueError("maximum_alternative_paths cannot be negative")
    if any(
        value < 0.0
        for value in (
            config.alternative_tip_separation_px,
            config.alternative_prefix_separation_px,
        )
    ):
        raise ValueError("alternative path separations cannot be negative")
    births = None if birth_time is None else np.asarray(birth_time, dtype=np.float32)
    if births is not None and births.shape != arrays[0].shape:
        raise ValueError("birth_time must match the ribbon feature shape")
    if minimum_material_birth is not None and births is None:
        raise ValueError("minimum_material_birth requires birth_time")
    terminals = None
    if terminal_mask is not None:
        terminals = np.asarray(terminal_mask, dtype=bool)
        if terminals.shape != arrays[0].shape[1:]:
            raise ValueError("terminal_mask must match the ribbon image shape")

    merged_volume, paired_volume, width_volume, balance_volume = arrays
    undirected_count = merged_volume.shape[0]
    directed_count = 2 * undirected_count
    angle_step = 2.0 * math.pi / directed_count
    initial_angle = math.atan2(float(initial[0]), float(initial[1])) % (
        2.0 * math.pi
    )
    initial_bin = int(round(initial_angle / angle_step)) % directed_count

    prior = None
    prior_arc = None
    prior_length = 0.0
    prior_tip = None
    prior_tip_direction = None
    if prior_curve_yx is not None:
        prior = np.asarray(prior_curve_yx, dtype=np.float64)
        if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 2:
            raise ValueError("prior_curve_yx must have shape (points, 2)")
        prior = prior + (root - prior[0])
        prior_arc = _curve_arclength(prior)
        prior_length = float(prior_arc[-1])
        prior_tip, prior_tip_direction = _curve_point_and_direction(
            prior,
            prior_arc,
            prior_length,
        )

    def state_values(y: float, x: float, direction_bin: int):
        """Sample all physical ribbon attributes at one lifted state."""

        values = tuple(
            _state_value_at(volume, y, x, direction_bin) for volume in arrays
        )
        return tuple(value if np.isfinite(value) else 0.0 for value in values)

    def is_terminal(y: float, x: float) -> bool:
        """Return whether one search state has reached the requested target."""

        if terminals is None:
            return False
        row = int(round(y))
        column = int(round(x))
        return bool(
            0 <= row < terminals.shape[0]
            and 0 <= column < terminals.shape[1]
            and terminals[row, column]
        )

    root_merged, root_pair, root_width, root_balance = state_values(
        float(root[0]),
        float(root[1]),
        initial_bin,
    )
    root_birth = (
        _state_value_at(births, root[0], root[1], initial_bin)
        if births is not None
        else float("nan")
    )
    nodes = [
        _RibbonSearchNode(
            y=float(root[0]),
            x=float(root[1]),
            direction_bin=initial_bin,
            half_width_px=float(root_width),
            score=0.0,
            gap_steps=0,
            has_paired_support=False,
            parent=-1,
            paired_support=float(root_pair),
            merged_support=float(root_merged),
            wall_balance=float(root_balance),
            birth_time=float(root_birth),
            depth=0,
        )
    ]
    beam = [0]
    endpoints: list[int] = []
    maximum_length = float(config.maximum_length_px)
    if prior is not None:
        maximum_length = min(
            maximum_length,
            prior_length + config.maximum_extension_px,
        )
    maximum_depth = max(
        1,
        int(math.ceil(maximum_length / config.step_px)),
    )
    minimum_depth = max(
        1,
        int(math.ceil(config.minimum_length_px / config.step_px)),
    )

    for depth in range(1, maximum_depth + 1):
        material_position = depth * config.step_px
        candidates: dict[
            tuple[int, int, int, int, int, bool],
            tuple[float, float, int, float, float, int, bool, int, float, float, float, float],
        ] = {}
        for parent_index in beam:
            parent = nodes[parent_index]
            for turn in range(-config.maximum_turn_bins, config.maximum_turn_bins + 1):
                direction_bin = (parent.direction_bin + turn) % directed_count
                if config.maximum_total_turn_bins is not None:
                    total_turn = abs(direction_bin - initial_bin)
                    total_turn = min(total_turn, directed_count - total_turn)
                    if total_turn > config.maximum_total_turn_bins:
                        continue
                theta = angle_step * direction_bin
                y = parent.y + config.step_px * math.sin(theta)
                x = parent.x + config.step_px * math.cos(theta)
                merged, paired, observed_width, balance = state_values(
                    y,
                    x,
                    direction_bin,
                )
                state_birth = (
                    _state_value_at(births, y, x, direction_bin)
                    if births is not None
                    else float("nan")
                )
                if (
                    minimum_material_birth is not None
                    and material_position >= config.birth_floor_start_px
                    and np.isfinite(state_birth)
                    and state_birth
                    < minimum_material_birth - config.birth_backtrack_tolerance
                ):
                    continue
                inside_root_occlusion = material_position <= config.root_occlusion_px
                strictly_paired = (
                    paired >= config.minimum_pair_support
                    and balance >= config.minimum_wall_balance
                    and observed_width > 0.0
                )
                width = float(parent.half_width_px)
                width_penalty = 0.0
                if strictly_paired:
                    if parent.has_paired_support and parent.half_width_px > 0.0:
                        width_delta = abs(observed_width - parent.half_width_px)
                        if (
                            width_delta
                            > config.maximum_reacquisition_width_change_px
                        ):
                            continue
                        width_penalty = config.width_change_penalty * width_delta
                        width = float(
                            (1.0 - config.width_update_blend)
                            * parent.half_width_px
                            + config.width_update_blend * observed_width
                        )
                    else:
                        width = float(observed_width)
                if inside_root_occlusion:
                    gap_steps = 0
                    has_paired_support = parent.has_paired_support
                    increment = config.length_reward
                else:
                    gap_steps = parent.gap_steps + 1 if not strictly_paired else 0
                    allowed_gap = (
                        config.maximum_gap_steps
                        if parent.has_paired_support
                        else config.maximum_initial_gap_steps
                    )
                    if gap_steps > allowed_gap:
                        continue
                    has_paired_support = parent.has_paired_support or strictly_paired
                    increment = (
                        config.paired_support_weight * paired
                        + config.merged_support_weight * merged
                        + config.wall_balance_weight * balance
                        - config.evidence_floor
                        + config.length_reward
                        - width_penalty
                    )
                increment -= config.curvature_penalty * abs(turn)
                if (
                    births is not None
                    and np.isfinite(parent.birth_time)
                    and np.isfinite(state_birth)
                ):
                    delta_birth = state_birth - parent.birth_time
                    if delta_birth < -config.birth_backtrack_tolerance:
                        increment -= config.birth_backtrack_penalty * min(
                            -delta_birth - config.birth_backtrack_tolerance,
                            config.birth_delta_clip,
                        )
                    elif delta_birth > 0.0:
                        increment += config.birth_progress_reward * min(
                            delta_birth,
                            config.birth_delta_clip,
                        )
                if (
                    prior is not None
                    and material_position <= prior_length - config.prior_release_px
                ):
                    prior_point, prior_direction = _curve_point_and_direction(
                        prior,
                        prior_arc,
                        material_position,
                    )
                    increment -= config.prior_position_penalty * float(
                        np.linalg.norm(np.asarray((y, x)) - prior_point)
                    )
                    increment -= (
                        config.prior_direction_penalty
                        * _wrapped_angle_difference(theta, prior_direction)
                    )
                elif prior is not None:
                    extension = max(0.0, material_position - prior_length)
                    memory = math.exp(
                        -extension / max(config.extension_memory_px, 1e-6)
                    )
                    tip_vector = np.asarray(
                        (
                            math.sin(prior_tip_direction),
                            math.cos(prior_tip_direction),
                        )
                    )
                    predicted = prior_tip + extension * tip_vector
                    increment -= (
                        config.extension_position_penalty
                        * memory
                        * float(np.linalg.norm(np.asarray((y, x)) - predicted))
                    )
                    increment -= (
                        config.extension_direction_penalty
                        * memory
                        * _wrapped_angle_difference(theta, prior_tip_direction)
                    )
                score = parent.score + increment
                width_state = int(round(width / config.width_state_step_px))
                key = (
                    int(round(y)),
                    int(round(x)),
                    direction_bin,
                    width_state,
                    gap_steps,
                    has_paired_support,
                )
                previous = candidates.get(key)
                if previous is None or score > previous[3]:
                    candidates[key] = (
                        y,
                        x,
                        direction_bin,
                        score,
                        width,
                        gap_steps,
                        has_paired_support,
                        parent_index,
                        paired,
                        merged,
                        balance,
                        state_birth,
                    )
        if not candidates:
            break
        ranked = sorted(candidates.values(), key=lambda item: item[3], reverse=True)
        if config.maximum_alternative_paths and len(ranked) > config.beam_width:
            directional_buckets: dict[int, list[tuple]] = {}
            for candidate in ranked:
                displacement_angle = math.atan2(
                    candidate[0] - root[0],
                    candidate[1] - root[1],
                ) % (2.0 * math.pi)
                bucket = int(round(displacement_angle / angle_step)) % directed_count
                directional_buckets.setdefault(bucket, []).append(candidate)
            diverse = [bucket[0] for bucket in directional_buckets.values()]
            retained_ids = {id(candidate) for candidate in diverse}
            diverse.extend(
                candidate
                for candidate in ranked
                if id(candidate) not in retained_ids
            )
            ranked = diverse
        beam = []
        terminal_endpoints = []
        for candidate in ranked[: config.beam_width]:
            (
                y,
                x,
                direction_bin,
                score,
                width,
                gap_steps,
                has_paired_support,
                parent_index,
                paired,
                merged,
                balance,
                state_birth,
            ) = candidate
            node_index = len(nodes)
            nodes.append(
                _RibbonSearchNode(
                    y=y,
                    x=x,
                    direction_bin=direction_bin,
                    half_width_px=width,
                    score=score,
                    gap_steps=gap_steps,
                    has_paired_support=has_paired_support,
                    parent=parent_index,
                    paired_support=paired,
                    merged_support=merged,
                    wall_balance=balance,
                    birth_time=state_birth,
                    depth=depth,
                )
            )
            if is_terminal(y, x):
                terminal_endpoints.append(node_index)
            else:
                beam.append(node_index)
        endpoints.extend(terminal_endpoints)
        if depth >= minimum_depth:
            endpoint_limit = (
                len(beam)
                if config.maximum_alternative_paths
                else min(48, len(beam))
            )
            endpoints.extend(beam[:endpoint_limit])

    if not endpoints:
        endpoints = beam

    def endpoint_value(index: int) -> tuple[int, int, float]:
        """Prefer requested terminals, then open paths, then optical score."""

        node = nodes[index]
        path_length = max(node.depth * config.step_px, config.step_px)
        openness = math.hypot(node.y - root[0], node.x - root[1]) / path_length
        eligible = openness >= config.minimum_endpoint_separation_fraction
        value = node.score + config.endpoint_openness_reward * openness
        return int(is_terminal(node.y, node.x)), int(eligible), value

    def reconstruct(endpoint_index: int) -> CoupledRibbonTraceResult:
        """Reconstruct one complete root-to-endpoint state chain."""

        chain = []
        while endpoint_index >= 0:
            chain.append(endpoint_index)
            endpoint_index = nodes[endpoint_index].parent
        chain.reverse()
        path = np.asarray([(nodes[index].y, nodes[index].x) for index in chain])
        directions = np.asarray(
            [angle_step * nodes[index].direction_bin for index in chain],
            dtype=np.float64,
        )
        widths = np.asarray(
            [nodes[index].half_width_px for index in chain],
            dtype=np.float64,
        )
        paired = np.asarray(
            [nodes[index].paired_support for index in chain],
            dtype=np.float64,
        )
        merged = np.asarray(
            [nodes[index].merged_support for index in chain],
            dtype=np.float64,
        )
        balances = np.asarray(
            [nodes[index].wall_balance for index in chain],
            dtype=np.float64,
        )
        births_out = np.asarray(
            [nodes[index].birth_time for index in chain],
            dtype=np.float64,
        )
        arc = _curve_arclength(path)
        evaluated = arc >= config.root_occlusion_px
        strict = (
            (paired >= config.minimum_pair_support)
            & (balances >= config.minimum_wall_balance)
            & (widths > 0.0)
        )
        supported_fraction = (
            float(np.mean(strict[evaluated])) if np.any(evaluated) else 0.0
        )
        endpoint_separation = (
            float(np.linalg.norm(path[-1] - path[0]) / max(arc[-1], 1e-6))
            if len(path) > 1
            else 0.0
        )
        prior_error = 0.0
        if prior is not None:
            shared = min(float(arc[-1]), prior_length)
            samples = np.linspace(
                0.0,
                shared,
                max(2, int(math.ceil(shared)) + 1),
            )
            path_points = np.column_stack(
                [np.interp(samples, arc, path[:, axis]) for axis in range(2)]
            )
            prior_points = np.column_stack(
                [np.interp(samples, prior_arc, prior[:, axis]) for axis in range(2)]
            )
            prior_error = float(
                np.mean(np.linalg.norm(path_points - prior_points, axis=1))
            )
        return CoupledRibbonTraceResult(
            path_yx=path,
            direction_radians=directions,
            half_width_px=widths,
            paired_support=paired,
            merged_support=merged,
            wall_balance=balances,
            birth_time=births_out,
            score=float(nodes[chain[-1]].score),
            paired_supported_fraction=supported_fraction,
            endpoint_separation_fraction=endpoint_separation,
            prior_prefix_error_px=prior_error,
        )

    ranked_endpoints = sorted(endpoints, key=endpoint_value, reverse=True)
    selected = reconstruct(ranked_endpoints[0])
    if config.maximum_alternative_paths == 0:
        return selected

    retained = [selected]
    selected_eligibility = endpoint_value(ranked_endpoints[0])[0]
    for endpoint_index in ranked_endpoints[1:]:
        if endpoint_value(endpoint_index)[0] < selected_eligibility:
            break
        candidate = reconstruct(endpoint_index)
        distinct = True
        for previous in retained:
            shared = min(candidate.length_px, previous.length_px)
            samples = np.linspace(0.0, shared, max(2, int(math.ceil(shared)) + 1))
            candidate_arc = _curve_arclength(candidate.path_yx)
            previous_arc = _curve_arclength(previous.path_yx)
            candidate_points = np.column_stack(
                [
                    np.interp(samples, candidate_arc, candidate.path_yx[:, axis])
                    for axis in range(2)
                ]
            )
            previous_points = np.column_stack(
                [
                    np.interp(samples, previous_arc, previous.path_yx[:, axis])
                    for axis in range(2)
                ]
            )
            prefix_separation = float(
                np.mean(np.linalg.norm(candidate_points - previous_points, axis=1))
            )
            tip_separation = float(
                np.linalg.norm(candidate.path_yx[-1] - previous.path_yx[-1])
            )
            if (
                tip_separation < config.alternative_tip_separation_px
                or prefix_separation < config.alternative_prefix_separation_px
            ):
                distinct = False
                break
        if not distinct:
            continue
        retained.append(candidate)
        if len(retained) > config.maximum_alternative_paths:
            break
    return CoupledRibbonTraceResult(
        path_yx=selected.path_yx,
        direction_radians=selected.direction_radians,
        half_width_px=selected.half_width_px,
        paired_support=selected.paired_support,
        merged_support=selected.merged_support,
        wall_balance=selected.wall_balance,
        birth_time=selected.birth_time,
        score=selected.score,
        paired_supported_fraction=selected.paired_supported_fraction,
        endpoint_separation_fraction=selected.endpoint_separation_fraction,
        prior_prefix_error_px=selected.prior_prefix_error_px,
        alternatives=tuple(retained[1:]),
    )
