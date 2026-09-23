"""Trace a complete tube in a rigid coordinate system attached to its pollen."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from .anchored_tracing import (
    enhance_tube_probability,
    find_attached_birth_path,
    find_tip_extension,
    refine_chain_from_prior,
)
from .curve_prototype import (
    CurveTraceConfig,
    curve_length,
    ordered_prefix_retention,
    resample_curve,
)


def _default_trace_config():
    """Return named defaults for short-onset, distal-only tube growth."""
    return CurveTraceConfig(
        preprocessing="background_clahe",
        birth_search_radius=60,
        min_path_probability=0.07,
        min_attachment_probability=0.07,
        min_extension_temporal_score=0.04,
        min_curve_length=5.0,
        max_initial_curve_length=28.0,
        min_curve_straightness=0.20,
        max_geodesic_endpoints=64,
        max_tip_step=22.0,
        max_tip_lateral_step=14.0,
        chain_refit_radius=5,
        chain_root_lock_points=3,
        chain_refit_min_support=0.07,
        chain_refit_max_length_fraction=0.15,
    )


@dataclass(frozen=True)
class PollenAnchoredChainConfig:
    """Configure reference subtraction, seed confirmation, and chain updates."""

    crop_size: int = 320
    baseline_observations: int = 30
    novelty_window: int = 3
    novelty_dark_offset: float = 3.0
    novelty_intensity_scale: float = 24.0
    pollen_exclusion_padding_px: float = 4.0
    neighbor_contact_margin_px: int = 4
    target_interior_padding_px: float = 1.0
    seed_confirmation_frames: int = 4
    seed_prefix_tolerance_px: float = 4.5
    minimum_seed_prefix_retention: float = 0.50
    minimum_seed_growth_px: float = 8.0
    minimum_seed_radial_gain_radii: float = 1.0
    maximum_seed_regression_px: float = 3.0
    maximum_seed_gap_frames: int = 1
    maximum_direct_extension_px: float = 25.0
    maximum_centerline_points: int = 256
    chain_support_threshold: float = 0.06
    minimum_chain_supported_fraction: float = 0.65
    maximum_unsupported_gap_px: float = 5.0
    maximum_centerline_step_px: float = 4.5
    root_attachment_tolerance_px: float = 5.0
    trace: CurveTraceConfig = field(default_factory=_default_trace_config)


@dataclass(frozen=True)
class PollenAnchoredEvidence:
    """Hold structural and pre-germination novelty evidence for one frame."""

    gray: np.ndarray
    vessel_probability: np.ndarray
    novelty_probability: np.ndarray
    fused_probability: np.ndarray


@dataclass(frozen=True)
class PollenAnchoredMeasurement:
    """Describe one pollen-rooted material arc and its review state."""

    status: str
    usable: bool
    length_px: float
    centerline_yx: np.ndarray
    candidate_centerline_yx: np.ndarray
    structure_support: float
    seed_confirmations: int
    chain_supported_fraction: float
    root_attachment_error_px: float
    maximum_centerline_step_px: float
    neighbor_contact: bool


def _rigid_transform(initial_center_xy, current_center_xy, relative_angle):
    """Build an affine transform from the initial pollen pose to one frame."""
    cosine = float(np.cos(relative_angle))
    sine = float(np.sin(relative_angle))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    translation = np.asarray(current_center_xy) - rotation @ np.asarray(
        initial_center_xy
    )
    return np.column_stack((rotation, translation))


def stabilize_pollen_frames(grays, centers_xy, angles_radians, crop_size):
    """Undo pollen translation and rotation and return source-to-crop transforms."""
    grays = np.asarray(grays)
    centers = np.asarray(centers_xy, dtype=np.float64)
    angles = np.asarray(angles_radians, dtype=np.float64)
    if grays.ndim != 3:
        raise ValueError("grays must have shape (frames, height, width)")
    if centers.shape != (len(grays), 2):
        raise ValueError("centers_xy must have shape (frames, 2)")
    if angles.shape != (len(grays),):
        raise ValueError("angles_radians must have one value per frame")
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    output_center = np.asarray([crop_size / 2.0, crop_size / 2.0])
    initial_center = centers[0]
    initial_angle = angles[0]
    frames = []
    source_to_crop = []
    for gray, center, angle in zip(grays, centers, angles):
        initial_to_current = _rigid_transform(
            initial_center, center, float(angle - initial_angle)
        )
        matrix = cv.invertAffineTransform(initial_to_current)
        matrix[:, 2] += output_center - initial_center
        frames.append(
            cv.warpAffine(
                np.asarray(gray, dtype=np.uint8),
                matrix,
                (crop_size, crop_size),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT101,
            )
        )
        source_to_crop.append(matrix)
    return np.stack(frames), np.stack(source_to_crop)


def persistent_baseline_novelty(
    stabilized_grays,
    baseline_observations,
    window,
    dark_offset,
    intensity_scale,
):
    """Measure dark material that appeared and persisted after the opening baseline."""
    frames = np.asarray(stabilized_grays, dtype=np.uint8)
    if frames.ndim != 3 or not len(frames):
        raise ValueError("stabilized_grays must contain two-dimensional frames")
    baseline_count = min(max(1, int(baseline_observations)), len(frames))
    window = max(1, int(window))
    scale = float(intensity_scale)
    if scale <= 0:
        raise ValueError("intensity_scale must be positive")
    baseline = np.median(frames[:baseline_count].astype(np.float32), axis=0)
    instantaneous = np.clip(
        (
            baseline[None]
            - frames.astype(np.float32)
            - float(dark_offset)
        )
        / scale,
        0.0,
        1.0,
    )
    persistent = np.empty_like(instantaneous)
    for frame in range(len(frames)):
        start = max(0, frame - window + 1)
        persistent[frame] = np.median(instantaneous[start : frame + 1], axis=0)
    return baseline, persistent


def prepare_pollen_anchored_evidence(
    stabilized_grays,
    config=None,
    progress_callback=None,
):
    """Fuse vessel enhancement with persistent change from pre-germination frames."""
    config = config or PollenAnchoredChainConfig()
    baseline, novelty = persistent_baseline_novelty(
        stabilized_grays,
        config.baseline_observations,
        config.novelty_window,
        config.novelty_dark_offset,
        config.novelty_intensity_scale,
    )
    evidence = []
    for frame, (gray, novelty_frame) in enumerate(
        zip(stabilized_grays, novelty)
    ):
        prepared, vessel, _ = enhance_tube_probability(
            gray, aligned_previous=None, config=config.trace
        )
        fused = np.sqrt(
            np.clip(vessel, 0.0, 1.0)
            * np.clip(novelty_frame, 0.0, 1.0)
        )
        evidence.append(
            PollenAnchoredEvidence(
                gray=prepared,
                vessel_probability=vessel,
                novelty_probability=novelty_frame,
                fused_probability=fused,
            )
        )
        if progress_callback is not None:
            progress_callback(frame + 1, len(stabilized_grays))
    return baseline, evidence


def transform_points_xy(points_xy, matrix):
    """Apply one two-dimensional affine transform to Cartesian points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (points, 2)")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3)")
    return points @ matrix[:, :2].T + matrix[:, 2]


def build_pollen_exclusions(
    source_to_crop,
    pollen_positions,
    target_index,
    target_radius,
    config=None,
):
    """Mask tracked neighboring pollen while leaving the target attachment annulus."""
    config = config or PollenAnchoredChainConfig()
    transforms = np.asarray(source_to_crop, dtype=np.float64)
    positions = np.asarray(pollen_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("pollen_positions must have shape (objects, frames, 3)")
    if positions.shape[1] != len(transforms):
        raise ValueError("pollen positions and transforms must cover equal frames")
    if not 0 <= int(target_index) < len(positions):
        raise ValueError("target_index lies outside pollen_positions")
    size = int(config.crop_size)
    center = (int(round(size / 2.0)), int(round(size / 2.0)))
    exclusions = []
    for frame, matrix in enumerate(transforms):
        image = np.zeros((size, size), dtype=np.uint8)
        transformed = transform_points_xy(positions[:, frame, :2], matrix)
        for pollen, (point, radius) in enumerate(
            zip(transformed, positions[:, frame, 2])
        ):
            if (
                pollen == int(target_index)
                or not np.isfinite(point).all()
                or not np.isfinite(radius)
                or radius <= 0
            ):
                continue
            cv.circle(
                image,
                tuple(np.rint(point).astype(int)),
                max(
                    1,
                    int(
                        round(
                            radius + config.pollen_exclusion_padding_px
                        )
                    ),
                ),
                1,
                -1,
            )
        cv.circle(
            image,
            center,
            max(
                1,
                int(
                    round(
                        float(target_radius)
                        + config.target_interior_padding_px
                    )
                ),
            ),
            1,
            -1,
        )
        exclusions.append(image.astype(bool))
    return np.stack(exclusions)


def _empty_points():
    """Return a consistently shaped empty centerline."""
    return np.empty((0, 2), dtype=np.float64)


def _measurement(
    status,
    usable,
    chain=None,
    candidate=None,
    structure_support=0.0,
    seed_confirmations=0,
    chain_supported_fraction=0.0,
    root_attachment_error_px=np.nan,
    maximum_centerline_step_px=np.nan,
    neighbor_contact=False,
):
    """Build one normalized measurement without dropping review geometry."""
    points = _empty_points() if chain is None else np.asarray(chain, dtype=np.float64)
    candidate_points = (
        _empty_points()
        if candidate is None
        else np.asarray(candidate, dtype=np.float64)
    )
    return PollenAnchoredMeasurement(
        status=status,
        usable=bool(usable),
        length_px=0.0 if len(points) < 2 else float(curve_length(points)),
        centerline_yx=points,
        candidate_centerline_yx=candidate_points,
        structure_support=float(structure_support),
        seed_confirmations=int(seed_confirmations),
        chain_supported_fraction=float(chain_supported_fraction),
        root_attachment_error_px=float(root_attachment_error_px),
        maximum_centerline_step_px=float(maximum_centerline_step_px),
        neighbor_contact=bool(neighbor_contact),
    )


def chain_integrity(
    probability,
    chain,
    center_yx,
    pollen_radius,
    config=None,
    exclusion=None,
):
    """Measure full-path support, continuity, and target-root attachment."""
    config = config or PollenAnchoredChainConfig()
    probability = np.asarray(probability, dtype=np.float32)
    points = np.asarray(chain, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return False, 0.0, np.inf, np.inf, False
    rounded = np.rint(points).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < probability.shape[0])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < probability.shape[1])
    )
    local_support = np.zeros(len(points), dtype=np.float32)
    for index, (row, column) in enumerate(rounded):
        if not inside[index]:
            continue
        top = max(0, row - 1)
        bottom = min(probability.shape[0], row + 2)
        left = max(0, column - 1)
        right = min(probability.shape[1], column + 2)
        local_support[index] = float(
            np.max(probability[top:bottom, left:right])
        )
    supported_fraction = float(
        np.mean(local_support >= config.chain_support_threshold)
    )
    supported = local_support >= config.chain_support_threshold
    longest_run = current_run = 0
    for point_supported in supported:
        if point_supported:
            current_run = 0
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    typical_step = float(np.median(segment_lengths)) if len(segment_lengths) else 0.0
    unsupported_gap = longest_run * typical_step
    root_radius = float(np.linalg.norm(points[0] - np.asarray(center_yx)))
    expected_root_radius = float(pollen_radius) + 1.0
    root_error = abs(root_radius - expected_root_radius)
    maximum_step = float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    neighbor_contact = False
    if exclusion is not None:
        exclusion = np.asarray(exclusion, dtype=bool)
        if exclusion.shape != probability.shape:
            raise ValueError("exclusion must match probability")
        margin = max(0, int(config.neighbor_contact_margin_px))
        if margin:
            size = 2 * margin + 1
            contact_mask = cv.dilate(
                exclusion.astype(np.uint8),
                cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)),
            ).astype(bool)
        else:
            contact_mask = exclusion
        proximal = min(len(rounded), config.trace.chain_root_lock_points + 1)
        valid_indices = rounded[proximal:][inside[proximal:]]
        neighbor_contact = bool(
            len(valid_indices)
            and np.any(
                contact_mask[valid_indices[:, 0], valid_indices[:, 1]]
            )
        )
    valid = (
        np.all(inside)
        and supported_fraction >= config.minimum_chain_supported_fraction
        and unsupported_gap <= config.maximum_unsupported_gap_px
        and root_error <= config.root_attachment_tolerance_px
        and maximum_step <= config.maximum_centerline_step_px
        and not neighbor_contact
    )
    return valid, supported_fraction, root_error, maximum_step, neighbor_contact


def trace_pollen_anchored_chain(
    evidence,
    exclusions,
    pollen_radius,
    config=None,
):
    """Confirm a short attached birth and then update only that complete material arc."""
    config = config or PollenAnchoredChainConfig()
    exclusions = np.asarray(exclusions, dtype=bool)
    if exclusions.shape != (
        len(evidence),
        config.crop_size,
        config.crop_size,
    ):
        raise ValueError("exclusions must match the configured evidence crops")
    center_yx = np.asarray(
        [config.crop_size / 2.0, config.crop_size / 2.0]
    )
    pending = None
    chain = None
    contact_censored = False
    output = []
    for frame, (frame_evidence, exclusion) in enumerate(
        zip(evidence, exclusions)
    ):
        if contact_censored:
            output.append(
                _measurement(
                    "neighbor_contact_censored",
                    False,
                    chain=chain,
                    neighbor_contact=True,
                )
            )
            continue
        if chain is None:
            found = find_attached_birth_path(
                frame_evidence.fused_probability,
                frame_evidence.novelty_probability,
                center_yx,
                pollen_radius,
                exclusion,
                config.trace,
            )
            if found is None:
                if pending is not None:
                    pending["gap"] += 1
                    if pending["gap"] > config.maximum_seed_gap_frames:
                        pending = None
                output.append(_measurement("pre_germination", False))
                continue
            _, candidate, metrics = found
            candidate_length = float(curve_length(candidate))
            if pending is None:
                pending = {
                    "path": candidate,
                    "length": candidate_length,
                    "initial": candidate_length,
                    "maximum": candidate_length,
                    "confirmations": 1,
                    "gap": 0,
                }
            else:
                retention = ordered_prefix_retention(
                    pending["path"],
                    candidate,
                    config.seed_prefix_tolerance_px,
                )
                continues = (
                    retention >= config.minimum_seed_prefix_retention
                    and candidate_length
                    >= pending["length"] - config.maximum_seed_regression_px
                )
                if continues:
                    pending.update(
                        path=candidate,
                        length=candidate_length,
                        maximum=max(pending["maximum"], candidate_length),
                        confirmations=pending["confirmations"] + 1,
                        gap=0,
                    )
                else:
                    pending = {
                        "path": candidate,
                        "length": candidate_length,
                        "initial": candidate_length,
                        "maximum": candidate_length,
                        "confirmations": 1,
                        "gap": 0,
                    }
            growth = pending["maximum"] - pending["initial"]
            confirmed = (
                pending["confirmations"] >= config.seed_confirmation_frames
                and growth >= config.minimum_seed_growth_px
                and metrics["radial_gain"]
                >= config.minimum_seed_radial_gain_radii * float(pollen_radius)
            )
            if not confirmed:
                output.append(
                    _measurement(
                        "birth_candidate",
                        False,
                        candidate=candidate,
                        structure_support=metrics["support"],
                        seed_confirmations=pending["confirmations"],
                    )
                )
                continue
            chain = resample_curve(
                candidate,
                spacing=2.0,
                max_points=config.maximum_centerline_points,
            )
            confirmations = pending["confirmations"]
            pending = None
            integrity = chain_integrity(
                frame_evidence.vessel_probability,
                chain,
                center_yx,
                pollen_radius,
                config,
                exclusion,
            )
            if not integrity[0]:
                output.append(
                    _measurement(
                        "initial_chain_review",
                        False,
                        chain=chain,
                        structure_support=metrics["support"],
                        seed_confirmations=confirmations,
                        chain_supported_fraction=integrity[1],
                        root_attachment_error_px=integrity[2],
                        maximum_centerline_step_px=integrity[3],
                        neighbor_contact=integrity[4],
                    )
                )
                chain = None
                continue
            output.append(
                _measurement(
                    "chain_initialized",
                    True,
                    chain=chain,
                    structure_support=metrics["support"],
                    seed_confirmations=confirmations,
                    chain_supported_fraction=integrity[1],
                    root_attachment_error_px=integrity[2],
                    maximum_centerline_step_px=integrity[3],
                    neighbor_contact=integrity[4],
                )
            )
            continue

        refined, support, _ = refine_chain_from_prior(
            frame_evidence.vessel_probability,
            chain,
            exclusion=exclusion,
            search_radius=config.trace.chain_refit_radius,
            root_lock_points=config.trace.chain_root_lock_points,
            offset_change_penalty=(
                config.trace.chain_refit_offset_change_penalty
            ),
            offset_magnitude_penalty=(
                config.trace.chain_refit_offset_magnitude_penalty
            ),
        )
        previous_length = float(curve_length(chain))
        refined_length = float(curve_length(refined))
        length_change = abs(refined_length - previous_length) / max(
            previous_length, 1.0
        )
        refit_valid = (
            support >= config.trace.chain_refit_min_support
            and length_change <= config.trace.chain_refit_max_length_fraction
        )
        integrity = chain_integrity(
            frame_evidence.vessel_probability,
            refined,
            center_yx,
            pollen_radius,
            config,
            exclusion,
        )
        refit_valid = refit_valid and integrity[0]
        if refit_valid:
            chain = refined
            status = "chain_updated"
            usable = True
        else:
            if integrity[4]:
                status = "neighbor_contact_censored"
                contact_censored = True
            else:
                status = "chain_held_for_review"
            usable = False

        extension = None
        if refit_valid:
            extension = find_tip_extension(
                frame_evidence.vessel_probability,
                frame_evidence.novelty_probability,
                chain,
                exclusion,
                config.trace,
            )
        candidate = None
        if extension is not None:
            _, extension_yx, _, _ = extension
            joined = np.vstack((chain, extension_yx[1:]))
            proposed = resample_curve(
                joined,
                spacing=2.0,
                max_points=config.maximum_centerline_points,
            )
            added = float(curve_length(proposed) - curve_length(chain))
            if 0.0 < added <= config.maximum_direct_extension_px:
                proposed_integrity = chain_integrity(
                    frame_evidence.vessel_probability,
                    proposed,
                    center_yx,
                    pollen_radius,
                    config,
                    exclusion,
                )
                if proposed_integrity[0]:
                    chain = proposed
                    integrity = proposed_integrity
                    status = "chain_extended"
                    usable = True
                else:
                    candidate = proposed
                    status = "extension_integrity_review"
                    usable = False
            elif added > config.maximum_direct_extension_px:
                candidate = proposed
                status = "long_extension_review"
                usable = False
        output.append(
            _measurement(
                status,
                usable,
                chain=chain,
                candidate=candidate,
                structure_support=support,
                chain_supported_fraction=integrity[1],
                root_attachment_error_px=integrity[2],
                maximum_centerline_step_px=integrity[3],
                neighbor_contact=integrity[4],
            )
        )
    return output


def centerline_to_source_xy(centerline_yx, source_to_crop):
    """Map one stabilized row-column centerline back to source Cartesian pixels."""
    points = np.asarray(centerline_yx, dtype=np.float64)
    if len(points) == 0:
        return _empty_points()
    crop_to_source = cv.invertAffineTransform(
        np.asarray(source_to_crop, dtype=np.float64)
    )
    return transform_points_xy(points[:, ::-1], crop_to_source)
