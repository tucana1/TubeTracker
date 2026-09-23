"""Generate native-resolution centerlines by explicitly pairing tube walls.

Reduced-resolution traces can settle onto one phase-contrast wall instead of
the lumen midpoint.  This module retraces mature tubes at source resolution,
masks pollen interiors, and preserves orientation so projected crossings do
not become graph junctions.  It only proposes geometry; movie-wide causal and
ownership checks remain responsible for accepting a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2 as cv
import numpy as np

from .causal_portal import PhaseContrastRibbonConfig, phase_contrast_ribbon_features
from .orientation_worldsheet import (
    CoupledRibbonTraceConfig,
    CoupledRibbonTraceResult,
    PairedWallOrientationResult,
    propose_pollen_roots,
    trace_coupled_ribbon_lifted,
)


@dataclass(frozen=True)
class NativeBoundaryTracingConfig:
    """Configure source-resolution wall pairing and rooted path proposals."""

    pollen_radius_px: float = 15.0
    owner_mask_radius_factor: float = 0.87
    foreign_mask_radius_factor: float = 1.27
    attachment_count: int = 96
    direction_offsets: tuple[int, ...] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    probe_distances_px: tuple[float, ...] = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
    proposal_count: int = 16
    output_count: int = 6
    minimum_root_angle_separation_degrees: float = 8.0
    minimum_tip_separation_px: float = 4.0
    ribbon: PhaseContrastRibbonConfig = field(
        default_factory=lambda: PhaseContrastRibbonConfig(
            orientation_count=32,
            half_widths_px=(1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            tangent_samples_px=(-4.0, -2.0, 0.0, 2.0, 4.0),
            fine_sigma_px=0.8,
            background_sigma_px=7.0,
            structure_sigma_px=1.5,
        )
    )
    trace: CoupledRibbonTraceConfig = field(
        default_factory=lambda: CoupledRibbonTraceConfig(
            step_px=1.5,
            maximum_turn_bins=1,
            curvature_penalty=0.10,
            minimum_pair_support=0.04,
            minimum_wall_balance=0.08,
            paired_support_weight=1.5,
            merged_support_weight=0.15,
            wall_balance_weight=0.05,
            evidence_floor=0.05,
            length_reward=0.006,
            width_change_penalty=0.08,
            maximum_reacquisition_width_change_px=2.5,
            maximum_gap_steps=7,
            maximum_initial_gap_steps=7,
            root_occlusion_px=18.0,
            beam_width=1000,
            minimum_length_px=10.0,
            maximum_length_px=180.0,
            minimum_endpoint_separation_fraction=0.35,
            endpoint_openness_reward=0.15,
        )
    )

    def for_boundary_censoring(self) -> NativeBoundaryTracingConfig:
        """Permit faint continuation only for an image-edge-censored path."""

        return replace(
            self,
            trace=replace(
                self.trace,
                maximum_gap_steps=max(self.trace.maximum_gap_steps, 30),
                maximum_initial_gap_steps=max(
                    self.trace.maximum_initial_gap_steps,
                    10,
                ),
                maximum_total_turn_bins=4,
                minimum_pair_support=min(self.trace.minimum_pair_support, 0.015),
                evidence_floor=min(self.trace.evidence_floor, 0.02),
                length_reward=max(self.trace.length_reward, 0.014),
            ),
        )

    def for_faint_prefix_extension(self) -> NativeBoundaryTracingConfig:
        """Bridge a short weak segment while preserving the rooted heading."""

        return replace(
            self,
            trace=replace(
                self.trace,
                maximum_gap_steps=max(self.trace.maximum_gap_steps, 12),
                maximum_initial_gap_steps=max(
                    self.trace.maximum_initial_gap_steps,
                    12,
                ),
                maximum_total_turn_bins=4,
                minimum_pair_support=min(self.trace.minimum_pair_support, 0.025),
                evidence_floor=min(self.trace.evidence_floor, 0.03),
                length_reward=max(self.trace.length_reward, 0.008),
            ),
        )


@dataclass(frozen=True)
class NativeBoundaryCandidate:
    """Store one ranked source-resolution centerline proposal."""

    trace: CoupledRibbonTraceResult
    proposal_support: float
    radial_extension_px: float
    ranking_score: float


def path_extends_rooted_prefix(
    reference_path_yx: np.ndarray,
    candidate_path_yx: np.ndarray,
    *,
    minimum_length_gain_px: float = 6.0,
    comparison_length_px: float | None = None,
    maximum_root_distance_px: float = 4.0,
    maximum_prefix_p90_error_px: float = 3.0,
) -> bool:
    """Return whether a longer curve preserves the complete trusted prefix."""

    reference = np.asarray(reference_path_yx, dtype=np.float64)
    candidate = np.asarray(candidate_path_yx, dtype=np.float64)
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[1:] != (2,)
        or candidate.shape[1:] != (2,)
        or len(reference) < 2
        or len(candidate) < 2
    ):
        raise ValueError("paths must have shape (at least two points, 2)")
    if (
        minimum_length_gain_px <= 0.0
        or (comparison_length_px is not None and comparison_length_px <= 0.0)
        or maximum_root_distance_px < 0.0
        or maximum_prefix_p90_error_px < 0.0
    ):
        raise ValueError("prefix-extension distances must be positive")

    def arclength(path: np.ndarray) -> np.ndarray:
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )

    reference_arc = arclength(reference)
    candidate_arc = arclength(candidate)
    if candidate_arc[-1] < reference_arc[-1] + minimum_length_gain_px:
        return False
    if np.linalg.norm(candidate[0] - reference[0]) > maximum_root_distance_px:
        return False
    shared_length = min(
        reference_arc[-1] if comparison_length_px is None else comparison_length_px,
        reference_arc[-1],
        candidate_arc[-1],
    )
    samples = np.linspace(0.0, shared_length, max(8, int(shared_length) + 1))
    reference_points = np.column_stack(
        [np.interp(samples, reference_arc, reference[:, axis]) for axis in range(2)]
    )
    candidate_points = np.column_stack(
        [np.interp(samples, candidate_arc, candidate[:, axis]) for axis in range(2)]
    )
    errors = np.linalg.norm(reference_points - candidate_points, axis=1)
    return bool(np.percentile(errors, 90) <= maximum_prefix_p90_error_px)


def clip_path_to_image_bounds(
    path_yx: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """Stop a polyline at its first exact intersection with the image edge."""

    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_yx must have shape (points, 2), with at least two points")
    height, width = (int(value) for value in image_shape)
    if height < 2 or width < 2 or not np.isfinite(path).all():
        raise ValueError("image bounds and path coordinates must be finite and valid")
    maximum = np.asarray((height - 1.0, width - 1.0), dtype=np.float64)

    def inside(point: np.ndarray) -> bool:
        return bool(np.all(point >= 0.0) and np.all(point <= maximum))

    if not inside(path[0]):
        raise ValueError("path must begin inside the image")
    retained = [path[0]]
    for point in path[1:]:
        if inside(point):
            retained.append(point)
            continue
        previous = retained[-1]
        delta = point - previous
        intersections = []
        for axis in range(2):
            if delta[axis] < 0.0:
                intersections.append((0.0 - previous[axis]) / delta[axis])
            elif delta[axis] > 0.0:
                intersections.append((maximum[axis] - previous[axis]) / delta[axis])
        for fraction in sorted(
            value for value in intersections if 0.0 <= value <= 1.0
        ):
            intersection = previous + fraction * delta
            if inside(np.clip(intersection, 0.0, maximum)):
                retained.append(np.clip(intersection, 0.0, maximum))
                return np.asarray(retained), True
        raise RuntimeError("path left the image without a valid boundary intersection")
    endpoint = retained[-1]
    touches = bool(np.any(np.isclose(endpoint, 0.0)) or np.any(np.isclose(endpoint, maximum)))
    return np.asarray(retained), touches


def extend_path_to_image_boundary(
    path_yx: np.ndarray,
    image_shape: tuple[int, int],
    maximum_extension_px: float = 2.0,
) -> tuple[np.ndarray, bool]:
    """Extend an outward endpoint a tiny distance to its exact image edge."""

    path, touches = clip_path_to_image_bounds(path_yx, image_shape)
    if touches:
        return path, True
    if maximum_extension_px < 0.0:
        raise ValueError("maximum_extension_px cannot be negative")
    tail_start = max(0, len(path) - 5)
    direction = path[-1] - path[tail_start]
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-9:
        return path, False
    direction /= magnitude
    maximum = np.asarray((image_shape[0] - 1.0, image_shape[1] - 1.0))
    intersections = []
    for axis in range(2):
        if direction[axis] < -1e-9:
            intersections.append((0.0 - path[-1, axis]) / direction[axis])
        elif direction[axis] > 1e-9:
            intersections.append((maximum[axis] - path[-1, axis]) / direction[axis])
    for distance in sorted(value for value in intersections if value >= 0.0):
        intersection = path[-1] + distance * direction
        if distance <= maximum_extension_px and np.all(intersection >= -1e-6) and np.all(
            intersection <= maximum + 1e-6
        ):
            return np.vstack((path, np.clip(intersection, 0.0, maximum))), True
    return path, False


def source_boundary_terminal_mask(
    crop_shape: tuple[int, int],
    source_shape: tuple[int, int],
    source_offset_yx: tuple[float, float] | np.ndarray,
    band_px: float = 0.75,
) -> np.ndarray:
    """Mark true source-image edges in a possibly padded local crop."""

    crop_height, crop_width = (int(value) for value in crop_shape)
    source_height, source_width = (int(value) for value in source_shape)
    offset = np.asarray(source_offset_yx, dtype=np.float64)
    if crop_height < 1 or crop_width < 1 or source_height < 2 or source_width < 2:
        raise ValueError("crop and source shapes must be positive")
    if offset.shape != (2,) or not np.isfinite(offset).all() or band_px < 0.0:
        raise ValueError("source boundary mapping is invalid")
    local_y, local_x = np.mgrid[:crop_height, :crop_width]
    source_y = local_y + offset[0]
    source_x = local_x + offset[1]
    inside_y = (source_y >= -band_px) & (source_y <= source_height - 1 + band_px)
    inside_x = (source_x >= -band_px) & (source_x <= source_width - 1 + band_px)
    horizontal = inside_x & (
        (np.abs(source_y) <= band_px)
        | (np.abs(source_y - (source_height - 1)) <= band_px)
    )
    vertical = inside_y & (
        (np.abs(source_x) <= band_px)
        | (np.abs(source_x - (source_width - 1)) <= band_px)
    )
    return horizontal | vertical


def owner_aligned_temporal_consensus(
    grays: list[np.ndarray] | tuple[np.ndarray, ...],
    owner_centers_yx: np.ndarray,
    crop_radius_px: int,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Combine mature owner-aligned views so persistent faint walls reinforce."""

    images = [np.asarray(gray) for gray in grays]
    centers = np.asarray(owner_centers_yx, dtype=np.float64)
    if len(images) < 2 or any(image.ndim != 2 or not image.size for image in images):
        raise ValueError("at least two nonempty grayscale views are required")
    if centers.shape != (len(images), 2) or not np.isfinite(centers).all():
        raise ValueError("one finite owner center is required for every view")
    if crop_radius_px < 2:
        raise ValueError("consensus crop radius must be at least two pixels")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("consensus normalization percentiles are invalid")

    size = 2 * crop_radius_px
    normalized = []
    for image, center_yx in zip(images, centers):
        padding = crop_radius_px + 2
        padded = cv.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            cv.BORDER_REFLECT101,
        )
        crop = cv.getRectSubPix(
            padded,
            (size, size),
            (
                float(center_yx[1] + padding),
                float(center_yx[0] + padding),
            ),
        ).astype(np.float32)
        low, high = np.percentile(crop, (lower_percentile, upper_percentile))
        normalized.append(
            np.clip((crop - low) * 255.0 / max(float(high - low), 1.0), 0.0, 255.0)
        )
    return np.rint(np.median(np.stack(normalized), axis=0)).astype(np.uint8)


def replace_ribbon_polarity(
    config: PhaseContrastRibbonConfig, polarity: str
) -> PhaseContrastRibbonConfig:
    """Return a copy of a ribbon config with a different interior polarity."""

    return replace(config, interior_polarity=polarity)


@dataclass(frozen=True)
class DualPolarityRibbonFeatures:
    """Bright- and dark-lumen wall evidence with a physical OR consensus.

    A pollen-halo saddle scores bright-lumen pairing strongly while dark-lumen
    pairing stays near zero; a real bright tube scores bright strongly.  The
    old per-state minimum (AND) demanded both polarities confirm the same
    state, but the two models are mutually exclusive by construction — a
    bright lumen forces the dark response toward zero and vice versa — so the
    AND vetoed every real tube (dense P3 consensus collapsed to zero along
    its whole visible length).  Consensus is now the per-state maximum (OR):
    structure either polarity confirms is kept.  Halo saddles are still
    rejected downstream because they fail the dark-wall cross-section in the
    polarity-aware support audit, not because tracing never proposes them.
    """

    bright: PairedWallOrientationResult
    dark: PairedWallOrientationResult

    def consensus_paired(self) -> np.ndarray:
        """Return per-state paired support confirmed by either polarity."""

        return np.maximum(
            np.asarray(self.bright.paired_score, dtype=np.float32),
            np.asarray(self.dark.paired_score, dtype=np.float32),
        )

    def consensus_score(self) -> np.ndarray:
        """Return per-state merged support confirmed by either polarity."""

        return np.maximum(
            np.asarray(self.bright.score, dtype=np.float32),
            np.asarray(self.dark.score, dtype=np.float32),
        )

    def consensus_width(self) -> np.ndarray:
        """Return the bright-model width where both polarities agree."""

        return np.asarray(self.bright.half_width_px, dtype=np.float32)

    def consensus_balance(self) -> np.ndarray:
        """Return the weaker wall-balance reading of the two polarities."""

        return np.minimum(
            np.asarray(self.bright.wall_balance, dtype=np.float32),
            np.asarray(self.dark.wall_balance, dtype=np.float32),
        )


def _masked_features(
    features: PairedWallOrientationResult,
    owner_center_yx: np.ndarray,
    foreign_centers_yx: np.ndarray,
    config: NativeBoundaryTracingConfig,
) -> PairedWallOrientationResult:
    """Remove pollen interiors while retaining tube material at their rims."""

    height, width = features.score.shape[1:]
    grid_y, grid_x = np.mgrid[:height, :width]
    mask = np.hypot(
        grid_y - owner_center_yx[0],
        grid_x - owner_center_yx[1],
    ) < config.owner_mask_radius_factor * config.pollen_radius_px
    for center in foreign_centers_yx:
        mask |= np.hypot(
            grid_y - center[0],
            grid_x - center[1],
        ) < config.foreign_mask_radius_factor * config.pollen_radius_px
    arrays = []
    for source in (
        features.score,
        features.paired_score,
        features.half_width_px,
        features.wall_balance,
    ):
        value = np.asarray(source, dtype=np.float32).copy()
        value[:, mask] = 0.0
        arrays.append(value)
    return PairedWallOrientationResult(*arrays)


def _candidate_is_distinct(
    candidate: CoupledRibbonTraceResult,
    retained: list[NativeBoundaryCandidate],
    minimum_tip_separation_px: float,
) -> bool:
    """Keep alternatives whose endpoint or rooted material route differs."""

    for previous in retained:
        tip_distance = float(
            np.linalg.norm(candidate.path_yx[-1] - previous.trace.path_yx[-1])
        )
        shared = min(len(candidate.path_yx), len(previous.trace.path_yx), 16)
        prefix_distance = float(
            np.mean(
                np.linalg.norm(
                    candidate.path_yx[:shared] - previous.trace.path_yx[:shared],
                    axis=1,
                )
            )
        )
        if tip_distance < minimum_tip_separation_px and prefix_distance < 2.0:
            return False
    return True


def trace_native_boundary_candidates(
    gray: np.ndarray,
    owner_center_yx: np.ndarray,
    foreign_centers_yx: np.ndarray | None = None,
    config: NativeBoundaryTracingConfig | None = None,
    *,
    terminal_mask: np.ndarray | None = None,
) -> tuple[NativeBoundaryCandidate, ...]:
    """Trace ranked pollen-rooted centerlines between native paired walls."""

    image = np.asarray(gray)
    center = np.asarray(owner_center_yx, dtype=np.float64)
    foreign = (
        np.empty((0, 2), dtype=np.float64)
        if foreign_centers_yx is None
        else np.asarray(foreign_centers_yx, dtype=np.float64)
    )
    config = config or NativeBoundaryTracingConfig()
    if image.ndim != 2 or image.size == 0:
        raise ValueError("gray must be a nonempty two-dimensional image")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("owner_center_yx must contain one finite row-column point")
    if foreign.ndim != 2 or foreign.shape[1:] != (2,) or not np.isfinite(foreign).all():
        raise ValueError("foreign_centers_yx must have shape (owners, 2)")
    if config.pollen_radius_px <= 0.0 or config.output_count < 1:
        raise ValueError("native boundary configuration is invalid")
    if terminal_mask is not None and np.asarray(terminal_mask).shape != image.shape:
        raise ValueError("terminal_mask must match gray")

    features = DualPolarityRibbonFeatures(
        bright=_masked_features(
            phase_contrast_ribbon_features(
                image, replace(config.ribbon, wall_anisotropy_floor=0.30)
            ),
            center,
            foreign,
            config,
        ),
        dark=_masked_features(
            phase_contrast_ribbon_features(
                image,
                replace(
                    replace_ribbon_polarity(config.ribbon, "dark"),
                    wall_anisotropy_floor=0.30,
                ),
            ),
            center,
            foreign,
            config,
        ),
    )
    root_evidence = np.maximum(
        features.consensus_paired(), 0.20 * features.consensus_score()
    )
    proposals = propose_pollen_roots(
        root_evidence,
        center,
        config.pollen_radius_px,
        attachment_count=config.attachment_count,
        direction_offsets=config.direction_offsets,
        probe_distances_px=config.probe_distances_px,
        probe_top_k=min(4, len(config.probe_distances_px)),
        maximum_proposals=config.proposal_count,
        minimum_angle_separation_degrees=(
            config.minimum_root_angle_separation_degrees
        ),
    )
    ranked = []
    consensus = PairedWallOrientationResult(
        features.consensus_score(),
        features.consensus_paired(),
        features.consensus_width(),
        features.consensus_balance(),
    )
    for proposal in proposals:
        trace = trace_coupled_ribbon_lifted(
            consensus,
            proposal.root_yx,
            proposal.direction_yx,
            terminal_mask=terminal_mask,
            config=config.trace,
        )
        radial_extension = float(
            np.max(np.linalg.norm(trace.path_yx - center[None], axis=1))
            - config.pollen_radius_px
        )
        ranking_score = float(
            2.0 * trace.paired_supported_fraction
            + trace.endpoint_separation_fraction
            + 0.004 * trace.length_px
            + 0.002 * max(radial_extension, 0.0)
            + 0.05 * proposal.support
        )
        ranked.append(
            NativeBoundaryCandidate(
                trace=trace,
                proposal_support=proposal.support,
                radial_extension_px=radial_extension,
                ranking_score=ranking_score,
            )
        )
    retained: list[NativeBoundaryCandidate] = []
    for candidate in sorted(ranked, key=lambda item: item.ranking_score, reverse=True):
        if not _candidate_is_distinct(
            candidate.trace,
            retained,
            config.minimum_tip_separation_px,
        ):
            continue
        retained.append(candidate)
        if len(retained) >= config.output_count:
            break
    return tuple(retained)
