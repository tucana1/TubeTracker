"""Validate pollen-tube paths from causal emergence at the pollen boundary.

A plausible late-frame curve is not enough evidence for a pollen tube: pollen
rims and unrelated crossing tubes can have the same local appearance.  This
module treats a candidate as a time-ordered material path.  A valid path must
appear at the pollen boundary, remain connected there, and reveal an
increasing prefix as growth proceeds away from the grain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2 as cv
import numpy as np

from .orientation_worldsheet import PairedWallOrientationResult


@dataclass(frozen=True)
class PhaseContrastRibbonConfig:
    """Configure one polarity-specific phase-contrast tube model."""

    interior_polarity: str = "bright"
    orientation_count: int = 16
    half_widths_px: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    tangent_samples_px: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    fine_sigma_px: float = 0.6
    background_sigma_px: float = 3.5
    structure_sigma_px: float = 1.1
    orientation_concentration: float = 4.0
    wall_pair_weight: float = 1.0
    matching_center_weight: float = 0.65
    opposite_center_penalty: float = 0.75
    wall_asymmetry_penalty: float = 0.45
    normalization_percentile: float = 99.0
    wall_anisotropy_floor: float = 0.0


def phase_contrast_ribbon_features(
    gray: np.ndarray,
    config: PhaseContrastRibbonConfig = PhaseContrastRibbonConfig(),
) -> PairedWallOrientationResult:
    """Lift phase-contrast tubes while preserving wall polarity and width."""

    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray must be a two-dimensional image")
    if config.orientation_count < 4 or not config.half_widths_px:
        raise ValueError("phase-contrast ribbon configuration is invalid")
    if config.interior_polarity not in {"bright", "dark"}:
        raise ValueError("interior polarity must be 'bright' or 'dark'")
    fine = cv.GaussianBlur(image, (0, 0), config.fine_sigma_px)
    background = cv.GaussianBlur(image, (0, 0), config.background_sigma_px)
    signed = fine - background
    scale = float(
        np.percentile(np.abs(signed), config.normalization_percentile)
    )
    signed /= max(scale, 1e-6)
    bright = np.clip(signed, 0.0, 1.0)
    dark = np.clip(-signed, 0.0, 1.0)
    center_material, center_opposite, wall_material = (
        (bright, dark, dark)
        if config.interior_polarity == "bright"
        else (dark, bright, bright)
    )

    gradient_x = cv.Sobel(signed, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(signed, cv.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv.GaussianBlur(
        gradient_x * gradient_x, (0, 0), config.structure_sigma_px
    )
    tensor_xy = cv.GaussianBlur(
        gradient_x * gradient_y, (0, 0), config.structure_sigma_px
    )
    tensor_yy = cv.GaussianBlur(
        gradient_y * gradient_y, (0, 0), config.structure_sigma_px
    )
    principal_gradient = 0.5 * np.arctan2(
        2.0 * tensor_xy, tensor_xx - tensor_yy
    )
    tangent_orientation = principal_gradient + 0.5 * math.pi
    anisotropy = np.sqrt(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2
    ) / np.maximum(tensor_xx + tensor_yy, 1e-6)

    height, width = image.shape
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    shape = (config.orientation_count, height, width)
    scores = np.zeros(shape, dtype=np.float32)
    paired_scores = np.zeros(shape, dtype=np.float32)
    half_widths = np.zeros(shape, dtype=np.float32)
    wall_balances = np.zeros(shape, dtype=np.float32)
    tangent_offsets = np.asarray(config.tangent_samples_px, dtype=np.float32)

    for orientation in range(config.orientation_count):
        theta = math.pi * orientation / config.orientation_count
        tangent_y, tangent_x = math.sin(theta), math.cos(theta)
        normal_y, normal_x = -tangent_x, tangent_y
        alignment = np.exp(
            config.orientation_concentration
            * (np.cos(2.0 * (theta - tangent_orientation)) - 1.0)
        )
        oriented_walls = (
            wall_material
            * alignment
            * (
                config.wall_anisotropy_floor
                + (1.0 - config.wall_anisotropy_floor) * anisotropy
            )
        )
        center_material_samples = []
        center_opposite_samples = []
        for along in tangent_offsets:
            y = grid_y + tangent_y * along
            x = grid_x + tangent_x * along
            center_material_samples.append(
                _bilinear_sample(center_material, y, x)
            )
            center_opposite_samples.append(
                _bilinear_sample(center_opposite, y, x)
            )
        matching_center = np.mean(center_material_samples, axis=0)
        opposite_center = np.mean(center_opposite_samples, axis=0)
        best = np.zeros_like(image)
        best_paired = np.zeros_like(image)
        best_width = np.zeros_like(image)
        best_balance = np.zeros_like(image)
        for half_width in config.half_widths_px:
            left_samples = []
            right_samples = []
            for along in tangent_offsets:
                base_y = grid_y + tangent_y * along
                base_x = grid_x + tangent_x * along
                left_samples.append(
                    _bilinear_sample(
                        oriented_walls,
                        base_y - normal_y * half_width,
                        base_x - normal_x * half_width,
                    )
                )
                right_samples.append(
                    _bilinear_sample(
                        oriented_walls,
                        base_y + normal_y * half_width,
                        base_x + normal_x * half_width,
                    )
                )
            left = np.mean(left_samples, axis=0)
            right = np.mean(right_samples, axis=0)
            paired_walls = np.sqrt(np.maximum(left * right, 0.0))
            asymmetry = np.abs(left - right)
            response = (
                config.wall_pair_weight * paired_walls
                + config.matching_center_weight * matching_center
                - config.opposite_center_penalty * opposite_center
                - config.wall_asymmetry_penalty * asymmetry
            )
            response = np.maximum(response, 0.0)
            improved = response > best
            best[improved] = response[improved]
            best_paired[improved] = paired_walls[improved]
            best_width[improved] = half_width
            balance = 1.0 - asymmetry / np.maximum(left + right, 1e-6)
            best_balance[improved] = np.clip(balance[improved], 0.0, 1.0)
        scores[orientation] = best
        paired_scores[orientation] = best_paired
        half_widths[orientation] = best_width
        wall_balances[orientation] = best_balance

    positive = scores[scores > 0.0]
    if len(positive):
        response_scale = float(
            np.percentile(positive, config.normalization_percentile)
        )
        scores = np.clip(scores / max(response_scale, 1e-6), 0.0, 1.0)
        paired_scores = np.clip(
            paired_scores / max(response_scale, 1e-6), 0.0, 1.0
        )
    return PairedWallOrientationResult(
        score=scores,
        paired_score=paired_scores,
        half_width_px=half_widths,
        wall_balance=wall_balances,
    )


@dataclass(frozen=True)
class CausalPortalConfig:
    """Configure temporal denoising and connected-prefix validation."""

    warmup_samples: int = 8
    tail_samples: int = 20
    smoothing_radius: int = 1
    confirmation_window: int = 5
    confirmation_required: int = 3
    absolute_support_floor: float = 0.045
    minimum_support_change: float = 0.018
    noise_multiplier: float = 2.5
    maximum_gap_points: int = 3
    initial_search_points: int = 7
    minimum_late_coverage: float = 0.42
    maximum_early_coverage: float = 0.28
    maximum_pre_onset_orphan_fraction: float = 0.65
    minimum_birth_order_fraction: float = 0.68
    minimum_emergence_length_px: float = 1.0
    onset_growth_window_samples: int = 20
    minimum_onset_growth_px: float = 2.0
    minimum_growth_px: float = 5.0
    minimum_final_length_px: float = 7.0


@dataclass(frozen=True)
class CausalPortalResult:
    """Store a candidate's causal support, growth timeline, and decision."""

    smoothed_evidence: np.ndarray
    active: np.ndarray
    point_birth_samples: np.ndarray
    prefix_end_indices: np.ndarray
    prefix_lengths_px: np.ndarray
    first_connected_sample: int | None
    onset_sample: int | None
    early_coverage: float
    preexisting_coverage: float
    late_coverage: float
    pre_onset_orphan_fraction: float
    birth_order_fraction: float
    growth_px: float
    score: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class PollenOutlineConfig:
    """Configure native-resolution pollen-outline emergence measurement."""

    pollen_radius_px: float = 15.0
    fine_sigma_px: float = 1.0
    background_sigma_px: float = 6.0
    darkness_thresholds: tuple[float, ...] = (6.0, 8.0, 10.0)
    seed_inner_radius_factor: float = 0.60
    seed_outer_radius_factor: float = 1.27
    closing_radius_px: int = 1
    extent_percentile: float = 98.0
    warmup_samples: int = 8
    minimum_extension_px: float = 1.5
    preexisting_extent_factor: float = 1.35
    confirmation_window: int = 5
    confirmation_required: int = 3


@dataclass(frozen=True)
class PollenOutlineEmergence:
    """Store native outline extents and the first persistent protrusion."""

    extents_px: np.ndarray
    baseline_extents_px: np.ndarray
    ensemble_extension_px: np.ndarray
    onset_sample: int | None


@dataclass(frozen=True)
class PollenOutlineObservation:
    """Store connected outline radii and their distal coordinates."""

    extents_px: np.ndarray
    tips_yx: np.ndarray


@dataclass(frozen=True)
class NativeCenterlineRefinement:
    """Store one topology-preserving native centerline correction."""

    offsets_px: np.ndarray
    selected_support: np.ndarray
    baseline_support: np.ndarray
    normalized_median_gain: float
    positive_point_fraction: float
    positive_time_fraction: float
    search_edge_fraction: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class NativePathQualityCertificate:
    """Store repeated native paired-wall support for one retained path."""

    accepted: bool
    reason: str
    median_coverage: float
    median_support_margin: float
    observation_count: int
    completion_fraction: float
    proximal_support_ratio: float = float("nan")


@dataclass(frozen=True)
class PollenRingConfig:
    """Configure native validation of a closed phase-dark pollen ring."""

    pollen_radius_px: float = 15.0
    center_radius_factor: float = 0.47
    ring_radius_factors: tuple[float, float] = (0.67, 1.17)
    outer_radius_factors: tuple[float, float] = (1.40, 1.87)
    ring_step_px: float = 0.5
    ring_half_width_px: float = 1.5
    angular_samples: int = 48
    minimum_ring_contrast: float = 10.0
    minimum_angular_coverage: float = 0.65


@dataclass(frozen=True)
class PollenRingAssessment:
    """Store native pollen-ring evidence and its owner-gate decision."""

    ring_contrast: float
    angular_coverage: float
    ring_radius_px: float
    accepted: bool


def assess_pollen_ring(
    gray: np.ndarray,
    config: PollenRingConfig = PollenRingConfig(),
) -> PollenRingAssessment:
    """Require a dark closed ring around a brighter pollen center."""

    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 3.8 * config.pollen_radius_px:
        raise ValueError("gray crop is too small for pollen-ring assessment")
    image = cv.GaussianBlur(image, (0, 0), 1.0)
    height, width = image.shape
    center = np.asarray([(height - 1) / 2.0, (width - 1) / 2.0])
    grid_y, grid_x = np.mgrid[:height, :width]
    radius = np.hypot(grid_y - center[0], grid_x - center[1])
    center_level = float(
        np.mean(image[radius < config.center_radius_factor * config.pollen_radius_px])
    )
    outer_inner, outer_outer = (
        factor * config.pollen_radius_px for factor in config.outer_radius_factors
    )
    outer_level = float(
        np.median(image[(radius > outer_inner) & (radius < outer_outer)])
    )
    ring_inner, ring_outer = (
        factor * config.pollen_radius_px for factor in config.ring_radius_factors
    )
    ring_radii = np.arange(
        ring_inner, ring_outer + 0.5 * config.ring_step_px, config.ring_step_px
    )
    ring_levels = [
        float(
            np.mean(
                image[
                    (radius >= ring_radius - config.ring_half_width_px)
                    & (radius <= ring_radius + config.ring_half_width_px)
                ]
            )
        )
        for ring_radius in ring_radii
    ]
    best_index = int(np.argmin(ring_levels))
    ring_radius = float(ring_radii[best_index])
    ring_level = ring_levels[best_index]
    darkness_threshold = 0.5 * (center_level + outer_level)
    angles = np.linspace(0.0, 2.0 * math.pi, config.angular_samples, endpoint=False)
    sample_y = np.clip(
        np.rint(center[0] + ring_radius * np.sin(angles)).astype(int),
        0,
        height - 1,
    )
    sample_x = np.clip(
        np.rint(center[1] + ring_radius * np.cos(angles)).astype(int),
        0,
        width - 1,
    )
    angular_coverage = float(np.mean(image[sample_y, sample_x] < darkness_threshold))
    ring_contrast = float(min(center_level - ring_level, outer_level - ring_level))
    accepted = bool(
        ring_contrast >= config.minimum_ring_contrast
        and angular_coverage >= config.minimum_angular_coverage
    )
    return PollenRingAssessment(
        ring_contrast=ring_contrast,
        angular_coverage=angular_coverage,
        ring_radius_px=ring_radius,
        accepted=accepted,
    )


def measure_pollen_outline_extents(
    gray: np.ndarray,
    config: PollenOutlineConfig = PollenOutlineConfig(),
) -> np.ndarray:
    """Measure the connected dark-outline radius at several contrast levels."""

    return measure_pollen_outline(gray, config).extents_px


def measure_pollen_outline(
    gray: np.ndarray,
    config: PollenOutlineConfig = PollenOutlineConfig(),
) -> PollenOutlineObservation:
    """Measure connected pollen-outline radii and distal coordinates."""

    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 2 * config.pollen_radius_px:
        raise ValueError("gray crop is too small for the configured pollen radius")
    if not config.darkness_thresholds or any(
        threshold <= 0.0 for threshold in config.darkness_thresholds
    ):
        raise ValueError("darkness thresholds must be positive")
    fine = cv.GaussianBlur(image, (0, 0), config.fine_sigma_px)
    background = cv.GaussianBlur(image, (0, 0), config.background_sigma_px)
    darkness = background - fine
    height, width = image.shape
    center = np.asarray([(height - 1) / 2.0, (width - 1) / 2.0])
    grid_y, grid_x = np.mgrid[:height, :width]
    radius = np.hypot(grid_y - center[0], grid_x - center[1])
    seed_annulus = (
        radius >= config.seed_inner_radius_factor * config.pollen_radius_px
    ) & (radius <= config.seed_outer_radius_factor * config.pollen_radius_px)
    kernel_size = 2 * config.closing_radius_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    extents = np.zeros(len(config.darkness_thresholds), dtype=np.float32)
    tips = np.zeros((len(config.darkness_thresholds), 2), dtype=np.float32)
    for index, threshold in enumerate(config.darkness_thresholds):
        mask = (darkness >= threshold).astype(np.uint8)
        if config.closing_radius_px:
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        _, labels = cv.connectedComponents(mask, connectivity=8)
        seed_labels, counts = np.unique(
            labels[seed_annulus & (labels > 0)], return_counts=True
        )
        if not len(seed_labels):
            continue
        selected_label = int(seed_labels[np.argmax(counts)])
        component_radius = radius[labels == selected_label]
        extent = float(np.percentile(component_radius, config.extent_percentile))
        extents[index] = extent
        component_points = np.argwhere(labels == selected_label)
        distal_points = component_points[
            component_radius >= max(0.0, extent - 1.0)
        ]
        if len(distal_points):
            tips[index] = np.mean(distal_points, axis=0) - center
    return PollenOutlineObservation(extents_px=extents, tips_yx=tips)


def detect_pollen_outline_emergence(
    extents_px: np.ndarray,
    *,
    observable_start_sample: int = 0,
    config: PollenOutlineConfig = PollenOutlineConfig(),
) -> PollenOutlineEmergence:
    """Find a persistent outward extension beyond an owner's dormant outline."""

    extents = np.asarray(extents_px, dtype=np.float32)
    if extents.ndim != 2 or extents.shape[1] != len(config.darkness_thresholds):
        raise ValueError("extents must have shape (time, darkness thresholds)")
    baseline_end = observable_start_sample + config.warmup_samples
    if observable_start_sample < 0 or baseline_end >= len(extents):
        raise ValueError("observable warmup must fit within the extent timeline")
    baseline = np.max(extents[observable_start_sample:baseline_end], axis=0)
    extension = np.median(extents - baseline[None], axis=1)
    active = extension >= config.minimum_extension_px
    onset = None
    last_start = len(active) - config.confirmation_window + 1
    for sample in range(baseline_end, max(baseline_end, last_start)):
        window = active[sample : sample + config.confirmation_window]
        if active[sample] and np.count_nonzero(window) >= config.confirmation_required:
            onset = sample
            break
    return PollenOutlineEmergence(
        extents_px=extents,
        baseline_extents_px=baseline,
        ensemble_extension_px=extension,
        onset_sample=onset,
    )


def sample_native_paired_support(
    gray: np.ndarray,
    path_yx: np.ndarray,
    *,
    normal_search_offsets_px: tuple[float, ...] = (0.0,),
) -> tuple[np.ndarray, np.ndarray]:
    """Measure paired tube walls along a path and nearby control lanes."""

    image = cv.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), 0.7)
    path = np.asarray(path_yx, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_yx must have shape (points, 2)")
    if not normal_search_offsets_px:
        raise ValueError("at least one normal search offset is required")
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    def sample(points: np.ndarray) -> np.ndarray:
        return cv.remap(
            image,
            points[:, 1][None].astype(np.float32),
            points[:, 0][None].astype(np.float32),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT101,
        ).ravel()

    def paired_score(centerline: np.ndarray) -> np.ndarray:
        center = sample(centerline)
        best = np.zeros(len(centerline), dtype=np.float32)
        for half_width in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
            left_delta = sample(centerline + half_width * normal) - center
            right_delta = sample(centerline - half_width * normal) - center
            symmetric = left_delta * right_delta > 0.0
            response = np.where(
                symmetric,
                np.minimum(np.abs(left_delta), np.abs(right_delta))
                - 0.25 * np.abs(np.abs(left_delta) - np.abs(right_delta)),
                0.0,
            )
            best = np.maximum(best, response.astype(np.float32))
        return best

    searched = [
        paired_score(path + offset * normal)
        for offset in normal_search_offsets_px
    ]
    controls = np.concatenate(
        [
            paired_score(path + offset * normal)
            for offset in (-14.0, -10.0, 10.0, 14.0)
        ]
    )
    return np.max(searched, axis=0), controls


def native_ribbon_score_surface(
    gray: np.ndarray,
    path_yx: np.ndarray,
    normal_offsets_px: np.ndarray,
    *,
    half_widths_px: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    interior_polarity: str = "bright",
) -> np.ndarray:
    """Score full paired-wall cross-sections around one fixed path topology."""

    image = cv.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), 0.7)
    path = np.asarray(path_yx, dtype=np.float32)
    offsets = np.asarray(normal_offsets_px, dtype=np.float32)
    widths = np.asarray(half_widths_px, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray must be a two-dimensional image")
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        raise ValueError("path_yx must have shape (at least three points, 2)")
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("normal offsets must be a finite one-dimensional array")
    if np.any(np.diff(offsets) <= 0.0):
        raise ValueError("normal offsets must be strictly increasing")
    if widths.ndim != 1 or not len(widths) or np.any(widths <= 0.0):
        raise ValueError("half widths must be positive")
    if interior_polarity not in {"bright", "dark"}:
        raise ValueError("interior polarity must be 'bright' or 'dark'")
    polarity = 1.0 if interior_polarity == "bright" else -1.0

    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    def sample(points: np.ndarray) -> np.ndarray:
        return cv.remap(
            image,
            points[:, 1][None].astype(np.float32),
            points[:, 0][None].astype(np.float32),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT101,
        ).ravel()

    surface = np.zeros((len(path), len(offsets)), dtype=np.float32)
    tangent_offsets = (-2.0, 0.0, 2.0)
    for state, offset in enumerate(offsets):
        centerline = path + offset * normal
        best = np.zeros(len(path), dtype=np.float32)
        for half_width in widths:
            center_samples = []
            inner_left_samples = []
            inner_right_samples = []
            wall_left_samples = []
            wall_right_samples = []
            outer_left_samples = []
            outer_right_samples = []
            for along in tangent_offsets:
                cross_section = centerline + along * tangent
                center_samples.append(sample(cross_section))
                inner_left_samples.append(
                    sample(cross_section + 0.35 * half_width * normal)
                )
                inner_right_samples.append(
                    sample(cross_section - 0.35 * half_width * normal)
                )
                wall_left_samples.append(
                    sample(cross_section + half_width * normal)
                )
                wall_right_samples.append(
                    sample(cross_section - half_width * normal)
                )
                outer_left_samples.append(
                    sample(cross_section + (half_width + 2.0) * normal)
                )
                outer_right_samples.append(
                    sample(cross_section - (half_width + 2.0) * normal)
                )
            center = np.mean(center_samples, axis=0)
            inner_left = np.mean(inner_left_samples, axis=0)
            inner_right = np.mean(inner_right_samples, axis=0)
            wall_left = np.mean(wall_left_samples, axis=0)
            wall_right = np.mean(wall_right_samples, axis=0)
            outer_left = np.mean(outer_left_samples, axis=0)
            outer_right = np.mean(outer_right_samples, axis=0)

            left_contrast = polarity * (center - wall_left)
            right_contrast = polarity * (center - wall_right)
            paired = np.where(
                (left_contrast > 0.0) & (right_contrast > 0.0),
                np.minimum(left_contrast, right_contrast),
                0.0,
            )
            outer_recovery = np.minimum(
                np.maximum(polarity * (outer_left - wall_left), 0.0),
                np.maximum(polarity * (outer_right - wall_right), 0.0),
            )
            wall_asymmetry = np.abs(left_contrast - right_contrast)
            interior_variation = 0.5 * (
                np.abs(inner_left - center) + np.abs(inner_right - center)
            )
            score = (
                paired
                + 0.40 * outer_recovery
                - 0.30 * wall_asymmetry
                - 0.20 * interior_variation
            )
            best = np.maximum(best, np.maximum(score, 0.0).astype(np.float32))
        surface[:, state] = best
    return surface


def sample_native_ribbon_support(
    gray: np.ndarray,
    path_yx: np.ndarray,
    *,
    normal_search_offsets_px: tuple[float, ...] = (-3.0, 0.0, 3.0),
    control_offsets_px: tuple[float, ...] = (-14.0, -10.0, 10.0, 14.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Measure a bright lumen with paired dark walls against parallel controls."""

    searched = np.asarray(normal_search_offsets_px, dtype=np.float32)
    controls = np.asarray(control_offsets_px, dtype=np.float32)
    if searched.ndim != 1 or not len(searched) or not np.isfinite(searched).all():
        raise ValueError("normal search offsets must be finite")
    if controls.ndim != 1 or not len(controls) or not np.isfinite(controls).all():
        raise ValueError("control offsets must be finite")
    if any(np.any(np.isclose(control, searched)) for control in controls):
        raise ValueError("control offsets must be nonzero")
    offsets = np.unique(np.concatenate((searched, controls))).astype(np.float32)
    surface = native_ribbon_score_surface(gray, path_yx, offsets)
    search_states = np.asarray(
        [np.any(np.isclose(offset, searched)) for offset in offsets]
    )
    control_states = ~search_states
    return np.max(surface[:, search_states], axis=1), surface[:, control_states].ravel()


def fit_native_centerline_offsets(
    score_surfaces: np.ndarray,
    normal_offsets_px: np.ndarray,
    *,
    root_lock_points: int = 2,
    maximum_step_px: float = 2.0,
    smoothness_penalty: float = 0.10,
    curvature_penalty: float = 0.025,
    magnitude_penalty: float = 0.025,
    minimum_normalized_gain: float = 0.025,
    minimum_positive_point_fraction: float = 0.60,
    minimum_positive_time_fraction: float = 0.60,
    maximum_search_edge_fraction: float = 0.15,
) -> NativeCenterlineRefinement:
    """Fit and certify one smooth normal correction over repeated observations."""

    surfaces = np.asarray(score_surfaces, dtype=np.float64)
    offsets = np.asarray(normal_offsets_px, dtype=np.float64)
    if surfaces.ndim != 3 or surfaces.shape[2] != len(offsets):
        raise ValueError("score surfaces must have shape (time, points, offsets)")
    if surfaces.shape[0] < 2 or surfaces.shape[1] < 3:
        raise ValueError("at least two times and three path points are required")
    if not np.isfinite(surfaces).all() or not np.isfinite(offsets).all():
        raise ValueError("scores and offsets must be finite")
    if np.any(np.diff(offsets) <= 0.0) or not np.any(np.isclose(offsets, 0.0)):
        raise ValueError("offsets must increase strictly and include zero")
    if not 1 <= root_lock_points < surfaces.shape[1]:
        raise ValueError("root lock must leave at least one refinable point")
    if maximum_step_px <= 0.0:
        raise ValueError("maximum step must be positive")

    aggregate = np.median(surfaces, axis=0)
    scale = max(float(np.percentile(aggregate, 90.0)), 1e-6)
    normalized = aggregate / scale
    point_count, state_count = aggregate.shape
    zero_state = int(np.flatnonzero(np.isclose(offsets, 0.0))[0])
    energy = np.full((point_count, state_count), -np.inf, dtype=np.float64)
    previous = np.full((point_count, state_count), -1, dtype=np.int32)
    energy[0, zero_state] = normalized[0, zero_state]

    for point in range(1, point_count):
        allowed_states = (
            np.array([zero_state], dtype=np.int32)
            if point < root_lock_points
            else np.arange(state_count, dtype=np.int32)
        )
        for state in allowed_states:
            delta = np.abs(offsets[state] - offsets)
            valid = delta <= maximum_step_px + 1e-9
            transition = (
                energy[point - 1]
                - smoothness_penalty * delta
                - curvature_penalty * delta**2
            )
            transition[~valid] = -np.inf
            source = int(np.argmax(transition))
            if np.isfinite(transition[source]):
                energy[point, state] = (
                    transition[source]
                    + normalized[point, state]
                    - magnitude_penalty * abs(offsets[state])
                )
                previous[point, state] = source

    states = np.empty(point_count, dtype=np.int32)
    states[-1] = int(np.argmax(energy[-1]))
    for point in range(point_count - 1, 0, -1):
        states[point - 1] = previous[point, states[point]]
    if np.any(states < 0):
        raise RuntimeError("no feasible native centerline offset path")

    selected = aggregate[np.arange(point_count), states]
    baseline = aggregate[:, zero_state]
    gain = (selected - baseline) / scale
    refinable = np.arange(point_count) >= root_lock_points
    selected_offsets = offsets[states]
    median_gain = float(np.median(gain[refinable]))
    positive_points = float(np.mean(gain[refinable] > 0.01))
    frame_gain = np.median(
        (
            surfaces[:, np.arange(point_count), states]
            - surfaces[:, :, zero_state]
        )[:, refinable]
        / scale,
        axis=1,
    )
    positive_times = float(np.mean(frame_gain > 0.0))
    edge = np.isclose(selected_offsets, offsets[0]) | np.isclose(
        selected_offsets, offsets[-1]
    )
    edge_fraction = float(np.mean(edge[refinable]))

    if np.allclose(selected_offsets, 0.0):
        reason = "no-native-centerline-correction"
    elif median_gain < minimum_normalized_gain:
        reason = "insufficient-native-support-gain"
    elif positive_points < minimum_positive_point_fraction:
        reason = "native-gain-not-path-consistent"
    elif positive_times < minimum_positive_time_fraction:
        reason = "native-gain-not-time-consistent"
    elif edge_fraction > maximum_search_edge_fraction:
        reason = "native-optimum-reaches-search-boundary"
    else:
        reason = "native-centerline-refinement-verified"
    return NativeCenterlineRefinement(
        offsets_px=selected_offsets.astype(np.float32),
        selected_support=selected.astype(np.float32),
        baseline_support=baseline.astype(np.float32),
        normalized_median_gain=median_gain,
        positive_point_fraction=positive_points,
        positive_time_fraction=positive_times,
        search_edge_fraction=edge_fraction,
        accepted=reason == "native-centerline-refinement-verified",
        reason=reason,
    )


def certify_native_path_quality(
    support: np.ndarray,
    thresholds: np.ndarray,
    *,
    completion_fraction: float,
    minimum_observations: int = 3,
    minimum_completion_fraction: float = 0.85,
    minimum_median_coverage: float = 0.40,
    minimum_distal_support_fraction: float = 0.0,
    distal_fraction: float = 0.5,
    warmup_support: np.ndarray | None = None,
    minimum_emergence_gain: float = 0.0,
    minimum_proximal_support_ratio: float = 0.0,
    proximal_latch_fraction: float = 1.0 / 3.0,
    minimum_latch_path_points: int = 9,
) -> NativePathQualityCertificate:
    """Require repeated paired-wall support before exporting path geometry.

    Median coverage alone can pass a path whose support concentrates at the
    root (dense P92: proximal grain-halo edges score while the distal half
    is background).  When ``minimum_distal_support_fraction`` is positive,
    the distal ``distal_fraction`` of the path must independently reach it.
    The mirror-image failure is a latched distal: dense P76 starts at its
    grain into empty background (proximal-third mean support 2.1) then rides
    a neighbor's tube (distal-third mean 23.7, ratio 0.09) while median
    coverage (0.74) and the distal floor both pass.  When
    ``minimum_proximal_support_ratio`` is positive, the proximal
    ``proximal_latch_fraction`` of the path must reach that fraction of the
    distal fraction's late-median support — a tube must emerge at its own
    root, not just arrive somewhere far away.  Short paths below
    ``minimum_latch_path_points`` skip the test so stubs are never judged on
    one or two noisy points.  When ``warmup_support`` is given with a
    positive ``minimum_emergence_gain``,
    the path must also emerge relative to its own warmup baseline: paths whose
    support was already present before growth began (pre-existing rows, halo
    edges) are withheld as ``insufficient-emergence-gain``.
    """

    values = np.asarray(support, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("support must have shape (time, path points)")
    if limits.shape != (len(values),):
        raise ValueError("one threshold is required per observation")
    if not np.isfinite(values).all() or not np.isfinite(limits).all():
        raise ValueError("support and thresholds must be finite")
    if not 0.0 <= completion_fraction <= 1.0:
        raise ValueError("completion fraction must be a probability")
    if minimum_observations < 1:
        raise ValueError("minimum observations must be positive")
    if not 0.0 <= minimum_completion_fraction <= 1.0:
        raise ValueError("minimum completion fraction must be a probability")
    if not 0.0 <= minimum_median_coverage <= 1.0:
        raise ValueError("minimum median coverage must be a probability")
    if not 0.0 <= minimum_distal_support_fraction <= 1.0:
        raise ValueError("minimum distal support must be a probability")
    if not 0.0 < distal_fraction <= 1.0:
        raise ValueError("distal fraction must be a positive probability")
    if not minimum_proximal_support_ratio >= 0.0:
        raise ValueError("proximal support ratio must be nonnegative")
    if not 0.0 < proximal_latch_fraction <= 1.0:
        raise ValueError("proximal latch fraction must be a positive probability")
    if minimum_latch_path_points < 1:
        raise ValueError("minimum latch path points must be positive")

    if len(values):
        coverage = np.mean(values >= limits[:, None], axis=1)
        median_coverage = float(np.median(coverage))
        median_margin = float(np.median(values - limits[:, None]))
        distal_count = max(1, int(round(values.shape[1] * distal_fraction)))
        distal_coverage = float(
            np.median(np.mean(values[:, -distal_count:] >= limits[:, None], axis=1))
        )
        late_per_point = np.median(values, axis=0)
        latch_count = max(1, int(round(values.shape[1] * proximal_latch_fraction)))
        proximal_mean = float(np.mean(late_per_point[:latch_count]))
        distal_mean = float(np.mean(late_per_point[-latch_count:]))
        if (
            values.shape[1] >= minimum_latch_path_points
            and distal_mean > 0.0
        ):
            proximal_ratio = proximal_mean / distal_mean
        else:
            proximal_ratio = float("nan")
        if warmup_support is not None:
            warmup_values = np.asarray(warmup_support, dtype=np.float64)
            if warmup_values.shape != values.shape[1:]:
                raise ValueError("warmup support must match one path observation")
            warmup_baseline = float(np.median(warmup_values)) if warmup_values.size else 0.0
            newly_supported = float(np.mean(late_per_point > warmup_baseline + 2.0))
            emergence_gain = newly_supported * 10.0
        else:
            emergence_gain = 0.0
    else:
        median_coverage = 0.0
        median_margin = 0.0
        distal_coverage = 0.0
        emergence_gain = 0.0
        proximal_ratio = float("nan")
    if len(values) < minimum_observations:
        reason = "insufficient-native-quality-observations"
    elif completion_fraction < minimum_completion_fraction:
        reason = "insufficient-native-quality-completion"
    elif median_coverage < minimum_median_coverage:
        reason = "insufficient-native-path-coverage"
    elif distal_coverage < minimum_distal_support_fraction:
        reason = "insufficient-distal-path-coverage"
    elif (
        minimum_proximal_support_ratio > 0.0
        and np.isfinite(proximal_ratio)
        and proximal_ratio < minimum_proximal_support_ratio
    ):
        reason = "insufficient-proximal-path-support"
    elif emergence_gain < minimum_emergence_gain:
        reason = "insufficient-emergence-gain"
    else:
        reason = "native-path-quality-verified"
    return NativePathQualityCertificate(
        accepted=reason == "native-path-quality-verified",
        reason=reason,
        median_coverage=median_coverage,
        median_support_margin=median_margin,
        observation_count=len(values),
        completion_fraction=float(completion_fraction),
        proximal_support_ratio=float(proximal_ratio),
    )


def native_connected_prefix_growth(
    support: np.ndarray,
    thresholds: np.ndarray,
    arc_lengths_px: np.ndarray,
    *,
    warmup_samples: int = 8,
    maximum_gap_points: int = 4,
    minimum_strong_growth_px: float = 12.0,
    minimum_weak_growth_px: float = 8.0,
    maximum_growth_px_per_sample: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
    """Convert native wall support into one causal nondecreasing tube front."""

    values = np.asarray(support, dtype=np.float32)
    limits = np.asarray(thresholds, dtype=np.float32)
    arc = np.asarray(arc_lengths_px, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(arc):
        raise ValueError("support and arc lengths describe different paths")
    if limits.shape != (len(values),):
        raise ValueError("one support threshold is required per sample")
    if not 1 <= warmup_samples < len(values):
        raise ValueError("warmup_samples must fit inside the timeline")
    if (
        minimum_weak_growth_px <= 0.0
        or minimum_strong_growth_px < minimum_weak_growth_px
        or (
            maximum_growth_px_per_sample is not None
            and maximum_growth_px_per_sample <= 0.0
        )
    ):
        raise ValueError("growth thresholds must be positive and ordered")

    raw_lengths = np.zeros(len(values), dtype=np.float32)
    for sample, active in enumerate(values >= limits[:, None]):
        last_supported = -1
        gap = 0
        for point, is_supported in enumerate(active):
            gap = 0 if is_supported else gap + 1
            if gap > maximum_gap_points:
                break
            if is_supported:
                last_supported = point
        if last_supported >= 0:
            raw_lengths[sample] = arc[last_supported]

    smoothed = np.asarray(
        [
            np.median(raw_lengths[max(0, sample - 2) : sample + 1])
            for sample in range(len(raw_lengths))
        ],
        dtype=np.float32,
    )
    baseline = float(np.max(smoothed[:warmup_samples]))
    growth = np.maximum.accumulate(np.maximum(smoothed - baseline, 0.0))
    if maximum_growth_px_per_sample is not None:
        for sample in range(1, len(growth)):
            growth[sample] = min(
                growth[sample],
                growth[sample - 1] + maximum_growth_px_per_sample,
            )
    final_growth = float(growth[-1])
    strong_threshold = max(minimum_strong_growth_px, 0.08 * final_growth)
    weak_threshold = max(minimum_weak_growth_px, 0.05 * final_growth)

    def persistent_onset(threshold: float) -> int | None:
        active = growth >= threshold
        for sample in range(warmup_samples, len(active) - 8):
            if active[sample] and np.count_nonzero(active[sample : sample + 5]) >= 4:
                return sample
        return None

    onset = persistent_onset(strong_threshold)
    weak_onset = persistent_onset(weak_threshold) if onset is not None else None
    last_dormant = max(0, weak_onset - 1) if weak_onset is not None else None
    end_indices = np.searchsorted(arc, growth, side="right") - 1
    end_indices = np.clip(end_indices, -1, len(arc) - 1).astype(np.int32)
    return growth, end_indices, onset, last_dormant


def validate_causal_portal(
    evidence: np.ndarray,
    arc_lengths_px: np.ndarray,
    *,
    config: CausalPortalConfig = CausalPortalConfig(),
) -> CausalPortalResult:
    """Require a persistent, outward-growing evidence prefix along one path."""

    values = np.asarray(evidence, dtype=np.float32)
    arc = np.asarray(arc_lengths_px, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(arc):
        raise ValueError("evidence must have shape (time, path points)")
    if len(values) <= config.warmup_samples or len(arc) < 2:
        raise ValueError("insufficient time samples or path points")
    if np.any(np.diff(arc) < 0.0):
        raise ValueError("arc lengths must be nondecreasing")

    smoothed = _temporal_median(values, config.smoothing_radius)
    baseline = np.median(smoothed[: config.warmup_samples], axis=0)
    noise = 1.4826 * np.median(
        np.abs(smoothed[: config.warmup_samples] - baseline[None]),
        axis=0,
    )
    threshold = np.maximum(
        config.absolute_support_floor,
        baseline
        + np.maximum(
            config.minimum_support_change,
            config.noise_multiplier * noise,
        ),
    )
    active = _confirm_activity(
        smoothed >= threshold[None],
        config.confirmation_window,
        config.confirmation_required,
    )

    prefix_ends = np.full(len(values), -1, dtype=np.int32)
    for sample, row in enumerate(active):
        prefix_ends[sample] = _connected_prefix_end(
            row,
            maximum_gap=config.maximum_gap_points,
            initial_search_points=config.initial_search_points,
        )
    prefix_lengths = np.zeros(len(values), dtype=np.float32)
    visible = prefix_ends >= 0
    prefix_lengths[visible] = arc[prefix_ends[visible]]
    prefix_lengths = np.maximum.accumulate(prefix_lengths)

    births = np.full(len(arc), len(values), dtype=np.int32)
    for point in range(len(arc)):
        observations = np.flatnonzero(active[:, point])
        if len(observations):
            births[point] = int(observations[0])
    finite = births < len(values)
    birth_order = _birth_order_fraction(births[finite])
    early_coverage = float(np.mean(active[: config.warmup_samples]))
    preexisting_coverage = float(
        np.mean(baseline >= config.absolute_support_floor)
    )
    late_coverage = float(
        np.mean(active[max(0, len(values) - config.tail_samples) :])
    )
    growth = float(prefix_lengths[-1] - prefix_lengths[config.warmup_samples - 1])
    first_connected_candidates = np.flatnonzero(
        prefix_lengths >= config.minimum_emergence_length_px
    )
    first_connected = (
        int(first_connected_candidates[0]) if len(first_connected_candidates) else None
    )
    onset = _growth_supported_onset(
        prefix_lengths,
        minimum_length_px=config.minimum_emergence_length_px,
        window_samples=config.onset_growth_window_samples,
        minimum_growth_px=config.minimum_onset_growth_px,
    )
    pre_onset_orphan = _pre_onset_orphan_fraction(
        active,
        prefix_ends,
        onset if onset is not None else len(values),
        initial_search_points=config.initial_search_points,
        maximum_gap=config.maximum_gap_points,
    )

    reasons = []
    if first_connected is None:
        reasons.append("no-connected-emergence")
    if max(early_coverage, preexisting_coverage) > config.maximum_early_coverage:
        reasons.append("preexisting-or-rim-like")
    if late_coverage < config.minimum_late_coverage:
        reasons.append("weak-late-tube-support")
    if pre_onset_orphan > config.maximum_pre_onset_orphan_fraction:
        reasons.append("distal-material-before-portal")
    if birth_order < config.minimum_birth_order_fraction:
        reasons.append("noncausal-point-order")
    if growth < config.minimum_growth_px:
        reasons.append("insufficient-connected-growth")
    if prefix_lengths[-1] < config.minimum_final_length_px:
        reasons.append("short-final-prefix")

    final_scale = max(float(arc[-1]), 1.0)
    score = (
        2.2 * late_coverage
        + 1.4 * birth_order
        + min(growth / final_scale, 1.0)
        + min(float(prefix_lengths[-1]) / final_scale, 1.0)
        - 2.5 * early_coverage
        - 0.8 * pre_onset_orphan
    )
    return CausalPortalResult(
        smoothed_evidence=smoothed,
        active=active,
        point_birth_samples=births,
        prefix_end_indices=prefix_ends,
        prefix_lengths_px=prefix_lengths,
        first_connected_sample=first_connected,
        onset_sample=onset,
        early_coverage=early_coverage,
        preexisting_coverage=preexisting_coverage,
        late_coverage=late_coverage,
        pre_onset_orphan_fraction=pre_onset_orphan,
        birth_order_fraction=birth_order,
        growth_px=growth,
        score=float(score),
        accepted=not reasons,
        reason="accepted" if not reasons else ";".join(reasons),
    )


def path_exits_owner_once(
    path_yx: np.ndarray,
    owner_center_yx: np.ndarray,
    owner_radius_px: float,
    *,
    exit_radius_factor: float = 1.55,
    return_radius_factor: float = 1.35,
) -> bool:
    """Reject paths that leave a pollen grain and then curl onto its rim."""

    path = np.asarray(path_yx, dtype=np.float32)
    center = np.asarray(owner_center_yx, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape (points, 2)")
    distance = np.linalg.norm(path - center[None], axis=1)
    exits = np.flatnonzero(distance >= exit_radius_factor * owner_radius_px)
    if not len(exits):
        return False
    return not bool(
        np.any(distance[exits[0] + 1 :] < return_radius_factor * owner_radius_px)
    )


def truncate_self_reentry(
    path_yx: np.ndarray,
    *,
    minimum_index_separation: int = 8,
    reentry_distance_px: float = 1.75,
) -> np.ndarray:
    """Stop an open centerline before it doubles back onto its own history."""

    path = np.asarray(path_yx, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (points, 2)")
    if minimum_index_separation < 2 or reentry_distance_px <= 0.0:
        raise ValueError("self-reentry controls must be positive")
    for distal in range(minimum_index_separation, len(path)):
        prior = path[: distal - minimum_index_separation + 1]
        if np.any(np.linalg.norm(prior - path[distal], axis=1) < reentry_distance_px):
            return path[:distal].copy()
    return path.copy()


def _temporal_median(values: np.ndarray, radius: int) -> np.ndarray:
    """Apply a short edge-padded temporal median without changing shape."""

    if radius < 0:
        raise ValueError("smoothing radius cannot be negative")
    if radius == 0:
        return values.copy()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    windows = [padded[offset : offset + len(values)] for offset in range(2 * radius + 1)]
    return np.median(np.stack(windows), axis=0).astype(np.float32)


def _growth_supported_onset(
    prefix_lengths_px: np.ndarray,
    *,
    minimum_length_px: float,
    window_samples: int,
    minimum_growth_px: float,
) -> int | None:
    """Find the first connected extension that begins sustained outward growth."""

    lengths = np.asarray(prefix_lengths_px, dtype=np.float32)
    if lengths.ndim != 1 or not len(lengths):
        raise ValueError("prefix lengths must be a nonempty one-dimensional array")
    if window_samples < 1 or minimum_length_px < 0.0 or minimum_growth_px <= 0.0:
        raise ValueError("growth-onset controls must be positive")
    previous = np.concatenate(([0.0], lengths[:-1]))
    extension = lengths > previous + 1e-6
    for sample in np.flatnonzero(extension & (lengths >= minimum_length_px)):
        end = min(len(lengths), int(sample) + window_samples + 1)
        future_length = float(np.max(lengths[sample:end]))
        if future_length - float(lengths[sample]) >= minimum_growth_px:
            return int(sample)
    return None


def _bilinear_sample(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Sample one image at floating-point coordinates with reflected borders."""

    return cv.remap(
        np.asarray(image, dtype=np.float32),
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT,
    )


def _confirm_activity(
    active: np.ndarray,
    window: int,
    required: int,
) -> np.ndarray:
    """Keep support that persists within a centered temporal window."""

    if window < 1 or required < 1 or required > window:
        raise ValueError("invalid confirmation window")
    left = window // 2
    right = window - left - 1
    padded = np.pad(active.astype(np.uint8), ((left, right), (0, 0)), mode="edge")
    counts = np.zeros(active.shape, dtype=np.uint16)
    for offset in range(window):
        counts += padded[offset : offset + len(active)]
    return counts >= required


def _connected_prefix_end(
    active: np.ndarray,
    *,
    maximum_gap: int,
    initial_search_points: int,
) -> int:
    """Return the final active point reachable through a short-gapped prefix."""

    row = np.asarray(active, dtype=bool)
    first_candidates = np.flatnonzero(row[:initial_search_points])
    if not len(first_candidates):
        return -1
    first = int(first_candidates[0])
    last_active = first
    gap = 0
    for point in range(first + 1, len(row)):
        if row[point]:
            last_active = point
            gap = 0
        else:
            gap += 1
            if gap > maximum_gap:
                break
    return last_active


def _birth_order_fraction(births: np.ndarray) -> float:
    """Measure whether distal path points appear no earlier than proximal ones."""

    values = np.asarray(births, dtype=np.int32)
    if len(values) < 2:
        return 0.0
    proximal, distal = np.triu_indices(len(values), k=1)
    return float(np.mean(values[distal] >= values[proximal] - 1))


def _pre_onset_orphan_fraction(
    active: np.ndarray,
    prefix_ends: np.ndarray,
    onset: int,
    *,
    initial_search_points: int,
    maximum_gap: int,
) -> float:
    """Measure active distal material that was not connected to the owner."""

    orphan_count = 0
    active_count = 0
    for sample in range(max(0, onset)):
        row = active[sample]
        active_count += int(np.count_nonzero(row))
        end = int(prefix_ends[sample])
        orphan_start = (
            initial_search_points
            if end < 0
            else min(len(row), end + maximum_gap + 1)
        )
        orphan_count += int(np.count_nonzero(row[orphan_start:]))
    if active_count == 0:
        return 0.0
    return float(orphan_count / active_count)
