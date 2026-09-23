"""Fit one pollen-owned material curve jointly across an entire video.

The causal orientation atlas determines tube identity and permanent arclength.
This module estimates only pose: every active material coordinate chooses a
normal displacement in every frame.  A metric-labeling graph cut couples those
choices across both time and arclength, so an isolated high-contrast crossing
cannot seize one frame and bending cannot be mistaken for apical growth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2 as cv
import numpy as np


DEFORMABLE_WORLDSHEET_REVISION = "v21.6-open-curve-causal-ensemble"
WORLDSHEET_PROMOTION_REVISION = "promotion-v4-open-pollen-owned-curve"


@dataclass(frozen=True)
class DeformableWorldsheetConfig:
    """Configure global normal-displacement inference for a material curve."""

    normal_radius_px: float = 6.0
    normal_step_px: float = 1.0
    orientation_neighborhood_bins: int = 1
    pairwise_smoothness: float = 0.28
    pairwise_truncation_px: float = 4.0
    unsupported_cost: float = 1.0
    birth_identity_weight: float = 0.12
    birth_identity_tolerance_samples: float = 2.0
    birth_identity_truncation_samples: float = 8.0
    root_lock_nodes: int = 2
    maximum_cycles: int = 12


@dataclass(frozen=True)
class AtlasCompetitionConfig:
    """Configure video-wide competition between independent root hypotheses."""

    support_length_weight: float = 1.0
    supported_fraction_weight: float = 0.30
    displacement_penalty: float = 0.025
    preexisting_material_penalty: float = 0.85
    terminal_blob_penalty: float = 1.60


@dataclass(frozen=True)
class WorldsheetPromotionConfig:
    """Configure separate geometry and scientific-measurement promotion gates."""

    minimum_prototype_score: float = 0.50
    minimum_prototype_supported_fraction: float = 0.50
    maximum_terminal_foreign_body_risk: float = 0.18
    minimum_accepted_frame_fraction: float = 0.80
    minimum_proximal_eventual_support_fraction: float = 0.30
    maximum_unsupported_material_gap_px: float = 6.0
    minimum_endpoint_separation_fraction: float = 0.25


@dataclass(frozen=True)
class DeformableWorldsheetResult:
    """Store the jointly fitted material surface and its optical evidence."""

    curves_yx: np.ndarray
    displacements_px: np.ndarray
    support: np.ndarray
    birth_identity_error_samples: np.ndarray | None
    active: np.ndarray
    labels: np.ndarray
    offsets_px: np.ndarray
    initial_energy: float
    final_energy: float


@dataclass(frozen=True)
class MaterialContinuityCertificate:
    """Describe whether image-supported material forms a chain from pollen."""

    proximal_eventual_support_fraction: float
    eventual_supported_fraction: float
    maximum_unsupported_gap_px: float
    eventual_support_by_point: np.ndarray


@dataclass(frozen=True)
class OpenCurveCertificate:
    """Describe whether a traced path escapes instead of closing into a loop."""

    length_px: float
    endpoint_separation_px: float
    endpoint_separation_fraction: float


def open_curve_certificate(path_yx: np.ndarray) -> OpenCurveCertificate:
    """Measure global path openness independently of local image confidence."""

    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2:
        raise ValueError("path must contain at least two row-column points")
    if not np.isfinite(path).all():
        raise ValueError("path must be finite")
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if length <= 0.0:
        raise ValueError("path must have positive arclength")
    endpoint_separation = float(np.linalg.norm(path[-1] - path[0]))
    return OpenCurveCertificate(
        length_px=length,
        endpoint_separation_px=endpoint_separation,
        endpoint_separation_fraction=endpoint_separation / length,
    )


def atlas_prototype_score(
    length_px: float,
    mean_support: float,
    supported_fraction: float,
    displacement_p90_px: float,
    preexisting_support: float,
    terminal_blobness: float = 0.0,
    config: AtlasCompetitionConfig | None = None,
) -> float:
    """Score one global curve prototype while penalizing old foreign material."""

    config = config or AtlasCompetitionConfig()
    values = np.asarray(
        [
            length_px,
            mean_support,
            supported_fraction,
            displacement_p90_px,
            preexisting_support,
            terminal_blobness,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("atlas prototype metrics must be finite")
    if length_px <= 0.0 or displacement_p90_px < 0.0:
        raise ValueError("length must be positive and displacement nonnegative")
    if np.any(values[[1, 2, 4, 5]] < 0.0) or np.any(
        values[[1, 2, 4, 5]] > 1.0
    ):
        raise ValueError("support metrics must lie between zero and one")
    length_scale = math.sqrt(length_px)
    return float(
        config.support_length_weight * mean_support * length_scale
        + config.supported_fraction_weight * supported_fraction
        - config.displacement_penalty * displacement_p90_px
        - config.preexisting_material_penalty
        * preexisting_support
        * length_scale
        - config.terminal_blob_penalty * terminal_blobness
    )


def orientation_blobness(
    orientation_score: np.ndarray,
    center_yx: np.ndarray | tuple[float, float],
    radius_px: float,
    support_threshold: float = 0.12,
) -> float:
    """Measure whether a neighborhood is round and multi-directional, not a tip."""

    score = np.asarray(orientation_score, dtype=np.float32)
    if score.ndim != 3 or score.shape[0] < 2:
        raise ValueError("orientation_score must contain multiple directions")
    center = np.asarray(center_yx, dtype=np.float64)
    if center.shape != (2,) or radius_px <= 0.0:
        raise ValueError("center and radius must describe one positive disk")
    mask = np.zeros(score.shape[1:], dtype=np.uint8)
    cv.circle(
        mask,
        tuple(np.rint(center[::-1]).astype(int)),
        max(1, int(round(radius_px))),
        1,
        -1,
    )
    selected = mask.astype(bool)
    if not np.any(selected):
        return 0.0
    directional_energy = np.sum(score[:, selected], axis=1, dtype=np.float64)
    total = float(np.sum(directional_energy))
    if total <= 1e-9:
        return 0.0
    probability = directional_energy / total
    positive = probability[probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(score.shape[0])
    area_fraction = float(
        np.mean(np.max(score[:, selected], axis=0) >= support_threshold)
    )
    return float(np.clip(entropy * area_fraction, 0.0, 1.0))


def worldsheet_promotion_decision(
    prototype_score: float,
    prototype_supported_fraction: float,
    terminal_blobness: float,
    accepted_frame_fraction: float,
    config: WorldsheetPromotionConfig | None = None,
    *,
    terminal_preexisting_support: float = 1.0,
    proximal_eventual_support_fraction: float = 1.0,
    maximum_unsupported_gap_px: float = 0.0,
    endpoint_separation_fraction: float = 1.0,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Decide geometry and measurement support without deleting hypotheses."""

    config = config or WorldsheetPromotionConfig()
    values = np.asarray(
        [
            prototype_score,
            prototype_supported_fraction,
            terminal_blobness,
            accepted_frame_fraction,
            terminal_preexisting_support,
            proximal_eventual_support_fraction,
            maximum_unsupported_gap_px,
            endpoint_separation_fraction,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("promotion metrics must be finite")
    if np.any(values[[1, 2, 3, 4, 5, 7]] < 0.0) or np.any(
        values[[1, 2, 3, 4, 5, 7]] > 1.0
    ):
        raise ValueError("promotion fractions must lie between zero and one")
    if maximum_unsupported_gap_px < 0.0:
        raise ValueError("unsupported material gap cannot be negative")
    reasons = []
    if prototype_score < config.minimum_prototype_score:
        reasons.append("weak-global-prototype-score")
    if prototype_supported_fraction < config.minimum_prototype_supported_fraction:
        reasons.append("insufficient-global-material-support")
    terminal_foreign_body_risk = terminal_blobness * math.sqrt(
        terminal_preexisting_support
    )
    if terminal_foreign_body_risk > config.maximum_terminal_foreign_body_risk:
        reasons.append("terminal-preexisting-pollen-body")
    if (
        proximal_eventual_support_fraction
        < config.minimum_proximal_eventual_support_fraction
    ):
        reasons.append("unsupported-pollen-attachment")
    if maximum_unsupported_gap_px > config.maximum_unsupported_material_gap_px:
        reasons.append("disconnected-material-gap")
    if endpoint_separation_fraction < config.minimum_endpoint_separation_fraction:
        reasons.append("closed-or-looping-curve")
    geometry_supported = not reasons
    if accepted_frame_fraction < config.minimum_accepted_frame_fraction:
        reasons.append("insufficient-accepted-timepoints")
    measurement_supported = geometry_supported and len(reasons) == 0
    return geometry_supported, measurement_supported, tuple(reasons)


def atlas_hypothesis_selection_key(
    prototype_score: float,
    terminal_blobness: float,
    terminal_preexisting_support: float,
    proximal_eventual_support_fraction: float,
    maximum_unsupported_gap_px: float,
    config: WorldsheetPromotionConfig | None = None,
    *,
    endpoint_separation_fraction: float = 1.0,
) -> tuple[int, float]:
    """Rank pollen-attached hypotheses ahead of stronger disconnected curves."""

    config = config or WorldsheetPromotionConfig()
    values = np.asarray(
        [
            prototype_score,
            terminal_blobness,
            terminal_preexisting_support,
            proximal_eventual_support_fraction,
            maximum_unsupported_gap_px,
            endpoint_separation_fraction,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("atlas-selection metrics must be finite")
    if np.any(values[[1, 2, 3, 5]] < 0.0) or np.any(values[[1, 2, 3, 5]] > 1.0):
        raise ValueError("atlas-selection fractions must lie between zero and one")
    if maximum_unsupported_gap_px < 0.0:
        raise ValueError("unsupported material gap cannot be negative")
    terminal_risk = terminal_blobness * math.sqrt(
        terminal_preexisting_support
    )
    attached = (
        proximal_eventual_support_fraction
        >= config.minimum_proximal_eventual_support_fraction
        and maximum_unsupported_gap_px
        <= config.maximum_unsupported_material_gap_px + 1e-6
        and terminal_risk <= config.maximum_terminal_foreign_body_risk
        and endpoint_separation_fraction
        >= config.minimum_endpoint_separation_fraction
    )
    return int(attached), float(prototype_score)


def material_continuity_certificate(
    support: np.ndarray,
    active: np.ndarray,
    atlas_path_yx: np.ndarray,
    *,
    support_threshold: float = 0.12,
    eventual_percentile: float = 75.0,
    proximal_length_px: float = 8.0,
    proximal_occlusion_px: float = 0.0,
) -> MaterialContinuityCertificate:
    """Certify a continuous image-supported material chain from the pollen rim."""

    values = np.asarray(support, dtype=np.float32)
    active_mask = np.asarray(active, dtype=bool)
    path = np.asarray(atlas_path_yx, dtype=np.float64)
    if values.ndim != 2 or active_mask.shape != values.shape:
        raise ValueError("support and active must share shape (time, material)")
    if path.shape != (values.shape[1], 2):
        raise ValueError("atlas path must contain one point per material column")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("support must be finite and nonnegative")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support threshold must lie between zero and one")
    if not 0.0 <= eventual_percentile <= 100.0:
        raise ValueError("eventual percentile must lie between zero and one hundred")
    if proximal_length_px <= 0.0 or proximal_occlusion_px < 0.0:
        raise ValueError("proximal length must be positive and occlusion nonnegative")
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    eventual = np.zeros(values.shape[1], dtype=np.float32)
    for point_index in range(values.shape[1]):
        observed = values[active_mask[:, point_index], point_index]
        if len(observed):
            eventual[point_index] = np.percentile(observed, eventual_percentile)
    supported = eventual >= support_threshold
    evaluated = arc >= proximal_occlusion_px
    proximal = evaluated & (arc <= proximal_occlusion_px + proximal_length_px)
    if not np.any(proximal):
        raise ValueError("proximal interval does not intersect the atlas")
    proximal_fraction = float(np.mean(supported[proximal]))
    point_spacing = float(np.median(np.diff(arc))) if len(arc) > 1 else 0.0
    maximum_gap = 0.0
    gap_start = None
    for point_index, is_supported in enumerate(supported):
        if not evaluated[point_index]:
            continue
        if not is_supported and gap_start is None:
            gap_start = point_index
        if is_supported and gap_start is not None:
            maximum_gap = max(
                maximum_gap,
                float(arc[point_index - 1] - arc[gap_start] + point_spacing),
            )
            gap_start = None
    if gap_start is not None:
        maximum_gap = max(
            maximum_gap,
            float(arc[-1] - arc[gap_start] + point_spacing),
        )
    return MaterialContinuityCertificate(
        proximal_eventual_support_fraction=proximal_fraction,
        eventual_supported_fraction=float(np.mean(supported[evaluated])),
        maximum_unsupported_gap_px=maximum_gap,
        eventual_support_by_point=eventual,
    )


def _curve_tangents_and_normals(path_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate stable unit tangents and normals along an ordered curve."""

    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_yx must have shape (points, 2) with two points")
    tangent = np.empty_like(path)
    tangent[0] = path[1] - path[0]
    tangent[-1] = path[-1] - path[-2]
    if len(path) > 2:
        tangent[1:-1] = path[2:] - path[:-2]
    magnitude = np.linalg.norm(tangent, axis=1)
    if np.any(magnitude <= 1e-9):
        raise ValueError("path_yx cannot contain a locally stationary segment")
    tangent /= magnitude[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return tangent, normal


def _offset_labels(config: DeformableWorldsheetConfig) -> np.ndarray:
    """Create symmetric physical displacement labels including exact zero."""

    if config.normal_radius_px < 0.0 or config.normal_step_px <= 0.0:
        raise ValueError("normal radius must be nonnegative and step must be positive")
    count = int(math.floor(config.normal_radius_px / config.normal_step_px))
    return config.normal_step_px * np.arange(-count, count + 1, dtype=np.float64)


def _active_material_mask(
    time_count: int,
    point_count: int,
    front_indices: np.ndarray | None,
) -> np.ndarray:
    """Convert a monotone tip index into active material coordinates."""

    if front_indices is None:
        return np.ones((time_count, point_count), dtype=bool)
    front = np.asarray(front_indices, dtype=int)
    if front.shape != (time_count,):
        raise ValueError("front_indices must contain one value per frame")
    if np.any(front < -1) or np.any(front >= point_count):
        raise ValueError("front index lies outside the material curve")
    return np.arange(point_count)[None, :] <= front[:, None]


def _orientation_bins(tangent_yx: np.ndarray, orientation_count: int) -> np.ndarray:
    """Map atlas tangent directions to undirected orientation channels."""

    angles = np.arctan2(tangent_yx[:, 0], tangent_yx[:, 1]) % math.pi
    return np.rint(angles * orientation_count / math.pi).astype(int) % orientation_count


def _sample_displacement_support(
    orientation_scores: np.ndarray,
    atlas_yx: np.ndarray,
    normals_yx: np.ndarray,
    orientation_bins: np.ndarray,
    offsets_px: np.ndarray,
    orientation_neighborhood_bins: int,
) -> np.ndarray:
    """Sample direction-aware evidence for every time, material point, and label."""

    scores = np.asarray(orientation_scores, dtype=np.float32)
    time_count, orientation_count, _, _ = scores.shape
    point_count = len(atlas_yx)
    support = np.zeros(
        (time_count, point_count, len(offsets_px)),
        dtype=np.float32,
    )
    for label, offset in enumerate(offsets_px):
        points = atlas_yx + offset * normals_yx
        map_x = points[:, 1][None, :].astype(np.float32)
        map_y = points[:, 0][None, :].astype(np.float32)
        for time_index in range(time_count):
            for orientation in np.unique(orientation_bins):
                indices = np.flatnonzero(orientation_bins == orientation)
                best = np.zeros(len(indices), dtype=np.float32)
                for delta in range(
                    -orientation_neighborhood_bins,
                    orientation_neighborhood_bins + 1,
                ):
                    channel = (int(orientation) + delta) % orientation_count
                    sampled = cv.remap(
                        scores[time_index, channel],
                        map_x[:, indices],
                        map_y[:, indices],
                        interpolation=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=0.0,
                    )[0]
                    best = np.maximum(best, sampled)
                support[time_index, indices, label] = best
    return support


def _metric_pairwise_cost(
    offsets_px: np.ndarray,
    config: DeformableWorldsheetConfig,
) -> np.ndarray:
    """Build a truncated absolute-distance metric for alpha expansion."""

    if config.pairwise_smoothness < 0.0:
        raise ValueError("pairwise_smoothness cannot be negative")
    if config.pairwise_truncation_px <= 0.0:
        raise ValueError("pairwise_truncation_px must be positive")
    distance = np.abs(offsets_px[:, None] - offsets_px[None, :])
    return config.pairwise_smoothness * np.minimum(
        distance,
        config.pairwise_truncation_px,
    )


def _sample_displacement_births(
    orientation_birth: np.ndarray,
    atlas_yx: np.ndarray,
    normals_yx: np.ndarray,
    orientation_bins: np.ndarray,
    offsets_px: np.ndarray,
) -> np.ndarray:
    """Sample the causal appearance time of every displacement hypothesis."""

    births = np.asarray(orientation_birth, dtype=np.float32)
    sampled = np.empty((len(atlas_yx), len(offsets_px)), dtype=np.float32)
    for label, offset in enumerate(offsets_px):
        points = atlas_yx + offset * normals_yx
        map_x = points[:, 1][None, :].astype(np.float32)
        map_y = points[:, 0][None, :].astype(np.float32)
        for orientation in np.unique(orientation_bins):
            indices = np.flatnonzero(orientation_bins == orientation)
            sampled[indices, label] = cv.remap(
                births[int(orientation)],
                map_x[:, indices],
                map_y[:, indices],
                interpolation=cv.INTER_NEAREST,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=float(np.max(births)),
            )[0]
    return sampled


def _birth_identity_cost(
    sampled_births: np.ndarray,
    center_label: int,
    config: DeformableWorldsheetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare every pose label with the atlas material's appearance time."""

    if config.birth_identity_weight < 0.0:
        raise ValueError("birth identity weight cannot be negative")
    if config.birth_identity_tolerance_samples < 0.0:
        raise ValueError("birth identity tolerance cannot be negative")
    if config.birth_identity_truncation_samples <= 0.0:
        raise ValueError("birth identity truncation must be positive")
    unseen = float(np.max(sampled_births))
    expected = sampled_births[:, center_label]
    expected_valid = expected < unseen
    observed_valid = sampled_births < unseen
    error = np.abs(sampled_births - expected[:, None])
    error[~expected_valid, :] = 0.0
    error[expected_valid[:, None] & ~observed_valid] = (
        config.birth_identity_truncation_samples
    )
    excess = np.maximum(
        error - config.birth_identity_tolerance_samples,
        0.0,
    )
    cost = config.birth_identity_weight * np.minimum(
        excess,
        config.birth_identity_truncation_samples,
    )
    return cost, error


def fit_deformable_worldsheet(
    orientation_scores: np.ndarray,
    atlas_path_yx: np.ndarray,
    front_indices: np.ndarray | None = None,
    config: DeformableWorldsheetConfig | None = None,
    orientation_birth: np.ndarray | None = None,
) -> DeformableWorldsheetResult:
    """Jointly fit curve pose over time with one metric-labeling graph cut."""

    config = config or DeformableWorldsheetConfig()
    scores = np.asarray(orientation_scores, dtype=np.float32)
    if scores.ndim != 4 or scores.shape[1] < 4:
        raise ValueError(
            "orientation_scores must have shape (time, orientations, height, width)"
        )
    if not np.isfinite(scores).all():
        raise ValueError("orientation_scores must be finite")
    scores = np.clip(scores, 0.0, 1.0)
    atlas = np.asarray(atlas_path_yx, dtype=np.float64)
    tangent, normal = _curve_tangents_and_normals(atlas)
    offsets = _offset_labels(config)
    center_label = int(np.argmin(np.abs(offsets)))
    active = _active_material_mask(len(scores), len(atlas), front_indices)
    direction_bins = _orientation_bins(tangent, scores.shape[1])
    sampled_support = _sample_displacement_support(
        scores,
        atlas,
        normal,
        direction_bins,
        offsets,
        config.orientation_neighborhood_bins,
    )
    unary = config.unsupported_cost * (1.0 - sampled_support.astype(np.float64))
    sampled_birth_error = None
    if orientation_birth is not None:
        births = np.asarray(orientation_birth, dtype=np.float32)
        if births.shape != scores.shape[1:]:
            raise ValueError("orientation_birth must match one orientation frame")
        if not np.isfinite(births).all() or np.any(births < 0.0):
            raise ValueError("orientation_birth must be finite and nonnegative")
        sampled_births = _sample_displacement_births(
            births,
            atlas,
            normal,
            direction_bins,
            offsets,
        )
        identity_cost, sampled_birth_error = _birth_identity_cost(
            sampled_births,
            center_label,
            config,
        )
        unary += identity_cost[None, :, :]
    inactive = ~active
    unary[inactive] = config.unsupported_cost * 4.0
    unary[inactive, center_label] = 0.0
    root_count = min(max(0, int(config.root_lock_nodes)), len(atlas))
    if root_count:
        unary[:, :root_count, :] = config.unsupported_cost * 8.0
        unary[:, :root_count, center_label] = 0.0
    pairwise = _metric_pairwise_cost(offsets, config).astype(np.float64)
    initial_labels = np.full(active.shape, center_label, dtype=np.int8)

    try:
        from maxflow.fastmin import aexpansion_grid, energy_of_grid_labeling
    except ImportError as error:
        raise RuntimeError(
            "Deformable worldsheet fitting requires the research dependencies"
        ) from error

    initial_energy = float(
        energy_of_grid_labeling(unary, pairwise, initial_labels)
    )
    labels = aexpansion_grid(
        unary,
        pairwise,
        max_cycles=int(config.maximum_cycles),
        labels=initial_labels.copy(),
    )
    final_energy = float(energy_of_grid_labeling(unary, pairwise, labels))
    displacement = offsets[labels]
    curves = atlas[None, :, :] + displacement[:, :, None] * normal[None, :, :]
    chosen_support = np.take_along_axis(
        sampled_support,
        labels[..., None],
        axis=2,
    )[..., 0]
    chosen_support[~active] = 0.0
    chosen_birth_error = (
        np.take_along_axis(
            sampled_birth_error[None, :, :],
            labels[..., None],
            axis=2,
        )[..., 0]
        if sampled_birth_error is not None
        else None
    )
    if chosen_birth_error is not None:
        chosen_birth_error[~active] = 0.0
    return DeformableWorldsheetResult(
        curves_yx=curves,
        displacements_px=displacement,
        support=chosen_support,
        birth_identity_error_samples=chosen_birth_error,
        active=active,
        labels=labels,
        offsets_px=offsets,
        initial_energy=initial_energy,
        final_energy=final_energy,
    )
