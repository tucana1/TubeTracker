"""Track a pollen body with redundant CoTracker landmarks and rigid consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import ssl
import urllib.request

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from .grain_pose import GrainPoseConfig, estimate_grain_poses
from .owner_memory import PollenLinkConfig, PollenObservation, link_pollen_observations


COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_CHECKPOINT_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"
)
COTRACKER_CHECKPOINT_SHA256 = (
    "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
)


def detect_field_pollen_circles(
    frames: np.ndarray,
    owner_radius_px: float,
    *,
    batch_size: int = 24,
    hough_threshold: int = 8,
    minimum_circle_score: float = 0.2,
) -> list[np.ndarray]:
    """Detect recall-oriented pollen circles in bounded-memory batches."""

    from .curve_prototype import CurveTraceConfig, detect_grain_candidates

    images = np.asarray(frames)
    if images.ndim != 3:
        raise ValueError("frames must have shape (time, height, width)")
    if owner_radius_px <= 0.0 or batch_size < 1:
        raise ValueError("owner radius and batch size must be positive")
    minimum_radius = max(2, int(np.floor(0.55 * owner_radius_px)))
    maximum_radius = max(
        minimum_radius + 1,
        int(np.ceil(1.6 * owner_radius_px)),
    )
    config = CurveTraceConfig(
        preprocessing="background",
        blur_radius=max(2, int(round(owner_radius_px))),
        background_sigma=max(10.0, 6.0 * owner_radius_px),
        min_grain_radius=minimum_radius,
        max_grain_radius=maximum_radius,
        grain_threshold=hough_threshold,
        min_grain_circle_score=minimum_circle_score,
    )
    output: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        for detections in detect_grain_candidates(batch, config):
            output.append(
                np.asarray(
                    [
                        (
                            float(item.gv3.y),
                            float(item.gv3.x),
                            0.5 * float(max(item.w, item.h)),
                            float(getattr(item, "circle_score", 0.0)),
                        )
                        for item in detections
                    ],
                    dtype=np.float64,
                ).reshape(-1, 4)
            )
    return output


@dataclass(frozen=True)
class PollenMotionConfig:
    """Configure reproducible pollen-body point tracking without bundled weights."""

    checkpoint: Path
    device: str = "auto"
    crop_size: int = 256
    ring_point_count: int = 12
    ring_radius_fraction: float = 0.62
    add_support_grid: bool = True
    model_revision: str = COTRACKER_REVISION
    pose: GrainPoseConfig = field(default_factory=GrainPoseConfig)

    @classmethod
    def from_environment(cls):
        """Resolve the external checkpoint and compute device from the environment."""
        default = Path.home() / ".cache/tubetracker/checkpoints/cotracker3_online.pth"
        checkpoint = Path(
            os.environ.get("TUBETRACKER_POINT_CHECKPOINT", default)
        ).expanduser()
        device = os.environ.get("TUBETRACKER_POINT_DEVICE", "auto")
        return cls(checkpoint=checkpoint, device=device)

    def cache_identity(self):
        """Describe the model and settings that determine a trajectory cache."""
        identity = {
            "model": "CoTracker3 online",
            "model_revision": self.model_revision,
            "checkpoint": str(self.checkpoint),
            "crop_size": self.crop_size,
            "ring_point_count": self.ring_point_count,
            "ring_radius_fraction": self.ring_radius_fraction,
            "requested_device": self.device,
        }
        try:
            identity["resolved_device"] = _resolve_device(self.device)
        except (ImportError, RuntimeError):
            identity["resolved_device"] = "unavailable"
        if self.checkpoint.exists():
            stat = self.checkpoint.stat()
            identity.update(
                checkpoint_size=stat.st_size,
                checkpoint_mtime_ns=stat.st_mtime_ns,
                checkpoint_sha256=_sha256_file(self.checkpoint),
            )
        return identity


@dataclass(frozen=True)
class PollenBodyTrajectory:
    """Store global pollen pose, body landmarks, and per-frame quality evidence."""

    centers_xy: np.ndarray
    angles_radians: np.ndarray
    transforms: np.ndarray
    observed: np.ndarray
    visible_point_count: np.ndarray
    inlier_point_count: np.ndarray
    median_residual_px: np.ndarray
    tracks_xy: np.ndarray
    visibility: np.ndarray
    initial_points_xy: np.ndarray
    crop_origin_xy: np.ndarray


@dataclass(frozen=True)
class DetectionSnapConfig:
    """Configure joint, radius-scaled assignment to visible pollen circles."""

    maximum_distance_radii: float = 1.6
    unmatched_cost: float = 1.15
    radius_cost_weight: float = 0.3
    quality_cost_weight: float = 0.2
    smoothing_window: int = 7
    minimum_support_samples: int = 3
    maximum_temporal_residual_radii: float = 0.8
    maximum_unsupported_windows: float = 2.0


@dataclass(frozen=True)
class PollenBodyRadialEvidence:
    """Summarize closed, opposite-side boundary support around an owner center."""

    median_radial_contrast: float
    angular_boundary_fraction: float
    opposite_boundary_fraction: float
    valid_angular_fraction: float


def pollen_body_radial_evidence(
    gray: np.ndarray,
    center_yx: tuple[float, float] | np.ndarray,
    pollen_radius_px: float,
    *,
    angular_samples: int = 72,
    radial_samples: int = 28,
    boundary_gradient_threshold: float = 3.0,
) -> PollenBodyRadialEvidence:
    """Measure whether a proposed center is enclosed by a pollen-like boundary."""

    image = np.asarray(gray)
    center = np.asarray(center_yx, dtype=np.float64)
    if image.ndim != 2 or not image.size:
        raise ValueError("gray must be a nonempty two-dimensional image")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_yx must contain one finite row-column point")
    if pollen_radius_px <= 0.0 or not np.isfinite(pollen_radius_px):
        raise ValueError("pollen radius must be finite and positive")
    if angular_samples < 8 or angular_samples % 2:
        raise ValueError("angular samples must be an even value of at least eight")
    if (
        radial_samples < 8
        or not np.isfinite(boundary_gradient_threshold)
        or boundary_gradient_threshold < 0.0
    ):
        raise ValueError("radial sampling and gradient threshold are invalid")

    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    radii = np.linspace(0.0, 1.7 * pollen_radius_px, radial_samples)
    sample_y = center[0] + np.sin(angles)[:, None] * radii[None]
    sample_x = center[1] + np.cos(angles)[:, None] * radii[None]
    profiles = cv.remap(
        image.astype(np.float32),
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT101,
    )
    profiles = cv.GaussianBlur(profiles, (1, 3), 0.0)
    gradients = np.abs(np.diff(profiles, axis=1))
    boundary_band = (
        (radii[1:] >= 0.55 * pollen_radius_px)
        & (radii[1:] <= 1.45 * pollen_radius_px)
    )
    band_y = sample_y[:, 1:][:, boundary_band]
    band_x = sample_x[:, 1:][:, boundary_band]
    valid_angles = (
        (band_y >= 0.0)
        & (band_y <= image.shape[0] - 1.0)
        & (band_x >= 0.0)
        & (band_x <= image.shape[1] - 1.0)
    ).all(axis=1)
    radial_contrast = np.max(gradients[:, boundary_band], axis=1)
    boundary_present = valid_angles & (
        radial_contrast >= boundary_gradient_threshold
    )
    opposite_valid = valid_angles & np.roll(valid_angles, angular_samples // 2)
    opposite_present = boundary_present & np.roll(
        boundary_present, angular_samples // 2
    )
    valid_count = int(np.count_nonzero(valid_angles))
    opposite_valid_count = int(np.count_nonzero(opposite_valid))
    return PollenBodyRadialEvidence(
        median_radial_contrast=(
            float(np.median(radial_contrast[valid_angles]))
            if valid_count
            else 0.0
        ),
        angular_boundary_fraction=(
            float(np.count_nonzero(boundary_present) / valid_count)
            if valid_count
            else 0.0
        ),
        opposite_boundary_fraction=(
            float(np.count_nonzero(opposite_present & opposite_valid) / opposite_valid_count)
            if opposite_valid_count
            else 0.0
        ),
        valid_angular_fraction=float(np.mean(valid_angles)),
    )


@dataclass(frozen=True)
class OwnerMotionFusionConfig:
    """Configure identity-first fusion of redundant pollen motion estimates."""

    minimum_inlier_fraction: float = 0.40
    minimum_inlier_count: int = 4
    minimum_template_score: float = 0.85


@dataclass(frozen=True)
class OwnerMotionFusionResult:
    """Store fused centers and the source selected at every owner-time sample."""

    centers_yx: np.ndarray
    source_codes: np.ndarray
    required_inlier_counts: np.ndarray


@dataclass(frozen=True)
class SeedCanonicalizationConfig:
    """Configure persistent pollen-body selection around uncertain seeds."""

    initialization_samples: int = 5
    minimum_support_fraction: float = 0.6
    maximum_link_distance_radii: float = 0.8
    maximum_seed_correction_radii: float = 2.0
    maximum_radius_error_fraction: float = 0.65
    support_weight: float = 3.0
    quality_weight: float = 2.0
    distance_penalty: float = 0.35
    radius_penalty: float = 0.75
    minimum_score: float = 2.0
    minimum_score_margin: float = 0.2


@dataclass(frozen=True)
class SeedCanonicalizationResult:
    """Store corrected owner priors and the persistent circles that justified them."""

    trajectories_yx: np.ndarray
    corrections_yx: np.ndarray
    selected_track_ids: np.ndarray
    selected_scores: np.ndarray
    selected_support_fractions: np.ndarray


def canonicalize_owner_trajectory_seeds(
    prior_tracks_yx: np.ndarray,
    detections_by_sample: list[np.ndarray],
    owner_radius_px: float,
    eligible_owner_mask: np.ndarray | None = None,
    config: SeedCanonicalizationConfig | None = None,
) -> SeedCanonicalizationResult:
    """Move uncertain trajectory seeds to unique persistent pollen bodies.

    A single nearest-circle snap can prefer a rounded tube tip over a slightly
    farther pollen body. This initialization pass first links circle evidence
    across several views, then ranks complete tracks by persistence, circle
    quality, expected pollen size, and distance from each prior trajectory.
    """

    config = config or SeedCanonicalizationConfig()
    priors = np.asarray(prior_tracks_yx, dtype=np.float64)
    if priors.ndim != 3 or priors.shape[2] != 2:
        raise ValueError("prior trajectories must have shape (owners, samples, 2)")
    owner_count, sample_count, _ = priors.shape
    if len(detections_by_sample) != sample_count:
        raise ValueError("one detection array is required for every sample")
    if owner_radius_px <= 0.0:
        raise ValueError("owner radius must be positive")
    if not (
        config.initialization_samples >= 2
        and 0.0 < config.minimum_support_fraction <= 1.0
        and config.maximum_link_distance_radii > 0.0
        and config.maximum_seed_correction_radii > 0.0
        and config.maximum_radius_error_fraction >= 0.0
        and config.minimum_score_margin >= 0.0
    ):
        raise ValueError("seed canonicalization thresholds are invalid")
    eligible = (
        np.ones(owner_count, dtype=bool)
        if eligible_owner_mask is None
        else np.asarray(eligible_owner_mask, dtype=bool)
    )
    if eligible.shape != (owner_count,):
        raise ValueError("eligible owner mask must match the owner count")
    if not np.isfinite(priors).all():
        raise ValueError("prior trajectories must be finite")

    initialization_count = min(config.initialization_samples, sample_count)
    observations_by_sample = []
    for sample in range(initialization_count):
        detections = np.asarray(detections_by_sample[sample], dtype=np.float64)
        if detections.size == 0:
            detections = np.empty((0, 4), dtype=np.float64)
        if detections.ndim != 2 or detections.shape[1] != 4:
            raise ValueError("detections must have y, x, radius, and quality columns")
        if not np.isfinite(detections).all():
            raise ValueError("pollen detections must be finite")
        observations_by_sample.append(
            tuple(
                PollenObservation(
                    frame_index=sample,
                    detection_id=index,
                    center_yx=(float(row[0]), float(row[1])),
                    area_px=float(np.pi * row[2] ** 2),
                    radius_px=float(row[2]),
                    circularity=float(np.clip(row[3], 0.0, 1.0)),
                    solidity=1.0,
                    pollen_score=float(np.clip(row[3], 0.0, 1.0)),
                    semantic_support=0.0,
                    evidence_sources=("circle-detection",),
                )
                for index, row in enumerate(detections)
                if row[2] > 0.0
            )
        )
    linked = link_pollen_observations(
        observations_by_sample,
        PollenLinkConfig(
            maximum_step_distance_px=(
                config.maximum_link_distance_radii * owner_radius_px
            ),
            maximum_radius_change_fraction=(
                config.maximum_radius_error_fraction
            ),
        ),
    )
    minimum_support = max(
        2,
        int(np.ceil(config.minimum_support_fraction * initialization_count)),
    )
    persistent = [
        track
        for track in linked
        if len({item.frame_index for item in track.observations}) >= minimum_support
    ]

    corrections = np.zeros((owner_count, 2), dtype=np.float64)
    selected_ids = np.full(owner_count, -1, dtype=np.int64)
    selected_scores = np.full(owner_count, np.nan, dtype=np.float64)
    selected_support = np.zeros(owner_count, dtype=np.float64)
    if not persistent or not np.any(eligible):
        return SeedCanonicalizationResult(
            trajectories_yx=priors.copy(),
            corrections_yx=corrections,
            selected_track_ids=selected_ids,
            selected_scores=selected_scores,
            selected_support_fractions=selected_support,
        )

    score = np.full((owner_count, len(persistent)), -np.inf, dtype=np.float64)
    residual_by_pair: dict[tuple[int, int], np.ndarray] = {}
    support_by_track = np.zeros(len(persistent), dtype=np.float64)
    for track_index, track in enumerate(persistent):
        samples = np.asarray(
            [item.frame_index for item in track.observations], dtype=np.int64
        )
        centers = np.asarray(
            [item.center_yx for item in track.observations], dtype=np.float64
        )
        radii = np.asarray(
            [item.radius_px for item in track.observations], dtype=np.float64
        )
        qualities = np.asarray(
            [item.pollen_score for item in track.observations], dtype=np.float64
        )
        support_fraction = len(np.unique(samples)) / initialization_count
        support_by_track[track_index] = support_fraction
        radius_error = abs(float(np.median(radii)) - owner_radius_px) / owner_radius_px
        if radius_error > config.maximum_radius_error_fraction:
            continue
        for owner in np.flatnonzero(eligible):
            residual = np.median(centers - priors[owner, samples], axis=0)
            distance_radii = float(np.linalg.norm(residual) / owner_radius_px)
            if distance_radii > config.maximum_seed_correction_radii:
                continue
            residual_by_pair[(int(owner), track_index)] = residual
            score[owner, track_index] = (
                config.support_weight * support_fraction
                + config.quality_weight * float(np.median(qualities))
                - config.distance_penalty * distance_radii
                - config.radius_penalty * radius_error
            )

    candidate = score.copy()
    for owner in np.flatnonzero(eligible):
        finite = np.sort(candidate[owner, np.isfinite(candidate[owner])])[::-1]
        if not len(finite) or finite[0] < config.minimum_score:
            candidate[owner] = -np.inf
            continue
        if len(finite) > 1 and finite[0] - finite[1] < config.minimum_score_margin:
            candidate[owner] = -np.inf
    finite_cost = np.where(np.isfinite(candidate), -candidate, 1e6)
    dummy_cost = np.zeros((owner_count, owner_count), dtype=np.float64)
    rows, columns = linear_sum_assignment(
        np.concatenate((finite_cost, dummy_cost), axis=1)
    )
    for owner, column in zip(rows, columns):
        if column >= len(persistent) or not np.isfinite(candidate[owner, column]):
            continue
        residual = residual_by_pair[(int(owner), int(column))]
        corrections[owner] = residual
        selected_ids[owner] = persistent[column].track_id
        selected_scores[owner] = score[owner, column]
        selected_support[owner] = support_by_track[column]

    return SeedCanonicalizationResult(
        trajectories_yx=priors + corrections[:, None, :],
        corrections_yx=corrections,
        selected_track_ids=selected_ids,
        selected_scores=selected_scores,
        selected_support_fractions=selected_support,
    )


def fuse_redundant_owner_motion(
    multipoint_yx: np.ndarray,
    template_yx: np.ndarray,
    detection_yx: np.ndarray,
    observed: np.ndarray,
    inlier_counts: np.ndarray,
    template_scores: np.ndarray,
    config: OwnerMotionFusionConfig | None = None,
) -> OwnerMotionFusionResult:
    """Prefer rigid temporal identity, then template, then circle detection.

    Circle detections are useful observations but can lock onto round tube tips.
    A rigid landmark consensus is therefore authoritative whenever enough body
    landmarks remain visible. Template matching covers low-landmark intervals;
    geometric detections are used only when both temporal estimates are weak.
    """

    config = config or OwnerMotionFusionConfig()
    multipoint = np.asarray(multipoint_yx, dtype=np.float64)
    template = np.asarray(template_yx, dtype=np.float64)
    detection = np.asarray(detection_yx, dtype=np.float64)
    visible = np.asarray(observed, dtype=bool)
    inliers = np.asarray(inlier_counts, dtype=np.int64)
    scores = np.asarray(template_scores, dtype=np.float64)
    if multipoint.ndim != 3 or multipoint.shape[2] != 2:
        raise ValueError("owner trajectories must have shape (owners, samples, 2)")
    if template.shape != multipoint.shape or detection.shape != multipoint.shape:
        raise ValueError("all owner trajectory sources must have the same shape")
    if visible.shape != multipoint.shape[:2]:
        raise ValueError("observed mask must match owner trajectory samples")
    if inliers.shape != visible.shape or scores.shape != visible.shape:
        raise ValueError("owner confidence arrays must match trajectory samples")
    if not (
        0.0 < config.minimum_inlier_fraction <= 1.0
        and config.minimum_inlier_count >= 1
        and 0.0 <= config.minimum_template_score <= 1.0
    ):
        raise ValueError("owner motion fusion thresholds are invalid")
    if not (
        np.isfinite(multipoint).all()
        and np.isfinite(template).all()
        and np.isfinite(detection).all()
        and np.isfinite(scores).all()
    ):
        raise ValueError("owner motion inputs must be finite")

    peak_inliers = np.maximum(np.max(inliers, axis=1), 1)
    required = np.maximum(
        config.minimum_inlier_count,
        np.ceil(config.minimum_inlier_fraction * peak_inliers).astype(np.int64),
    )
    multipoint_trusted = visible & (inliers >= required[:, None])
    template_trusted = scores >= config.minimum_template_score
    source_codes = np.full(visible.shape, 2, dtype=np.uint8)
    source_codes[template_trusted] = 1
    source_codes[multipoint_trusted] = 0
    centers = detection.copy()
    centers[source_codes == 1] = template[source_codes == 1]
    centers[source_codes == 0] = multipoint[source_codes == 0]
    return OwnerMotionFusionResult(
        centers_yx=centers,
        source_codes=source_codes,
        required_inlier_counts=required,
    )


def assign_unique_pollen_detections(
    prior_centers_yx: np.ndarray,
    detections_yxrs: np.ndarray,
    owner_radius_px: float,
    config: DetectionSnapConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each visible circle to at most one owner using a global optimum."""

    config = config or DetectionSnapConfig()
    priors = np.asarray(prior_centers_yx, dtype=np.float64)
    detections = np.asarray(detections_yxrs, dtype=np.float64)
    if priors.ndim != 2 or priors.shape[1] != 2:
        raise ValueError("prior centers must have shape (owners, 2)")
    if detections.size == 0:
        detections = np.empty((0, 4), dtype=np.float64)
    if detections.ndim != 2 or detections.shape[1] != 4:
        raise ValueError("detections must have y, x, radius, and quality columns")
    if owner_radius_px <= 0.0 or config.maximum_distance_radii <= 0.0:
        raise ValueError("owner radius and assignment distance must be positive")
    if config.unmatched_cost <= 0.0:
        raise ValueError("unmatched cost must be positive")
    owner_count, detection_count = len(priors), len(detections)
    assignments = np.full(owner_count, -1, dtype=np.int64)
    assignment_costs = np.full(owner_count, config.unmatched_cost, dtype=np.float64)
    if owner_count == 0 or detection_count == 0:
        return assignments, assignment_costs

    gate = config.maximum_distance_radii * owner_radius_px
    distances = np.linalg.norm(
        priors[:, None, :] - detections[None, :, :2], axis=2
    )
    radius_error = np.abs(detections[:, 2] - owner_radius_px) / owner_radius_px
    quality = np.clip(detections[:, 3], 0.0, 1.0)
    real_cost = (
        distances / gate
        + config.radius_cost_weight * radius_error[None, :]
        + config.quality_cost_weight * (1.0 - quality[None, :])
    )
    prohibited = config.unmatched_cost + 1.0
    real_cost[distances > gate] = prohibited
    dummy_cost = np.full((owner_count, owner_count), config.unmatched_cost)
    cost = np.concatenate((real_cost, dummy_cost), axis=1)
    owner_indices, column_indices = linear_sum_assignment(cost)
    for owner, column in zip(owner_indices, column_indices):
        if column < detection_count and cost[owner, column] < config.unmatched_cost:
            assignments[owner] = int(column)
            assignment_costs[owner] = float(cost[owner, column])
    return assignments, assignment_costs


def constrain_trajectories_to_unique_detections(
    prior_tracks_yx: np.ndarray,
    detections_by_sample: list[np.ndarray],
    owner_radius_px: float,
    config: DetectionSnapConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth globally unique circle observations into dense owner trajectories."""

    config = config or DetectionSnapConfig()
    priors = np.asarray(prior_tracks_yx, dtype=np.float64)
    if priors.ndim != 3 or priors.shape[2] != 2:
        raise ValueError("prior trajectories must have shape (owners, samples, 2)")
    owner_count, sample_count, _ = priors.shape
    if len(detections_by_sample) != sample_count:
        raise ValueError("one detection array is required for every sample")
    if config.minimum_support_samples < 1:
        raise ValueError("minimum support must be positive")
    if config.smoothing_window < 1 or config.maximum_unsupported_windows <= 0.0:
        raise ValueError("smoothing and unsupported-span controls must be positive")

    assignments = np.full((owner_count, sample_count), -1, dtype=np.int64)
    costs = np.full(
        (owner_count, sample_count), config.unmatched_cost, dtype=np.float64
    )
    observed = np.full_like(priors, np.nan)
    for sample, detections in enumerate(detections_by_sample):
        sample_assignments, sample_costs = assign_unique_pollen_detections(
            priors[:, sample], detections, owner_radius_px, config
        )
        assignments[:, sample] = sample_assignments
        costs[:, sample] = sample_costs
        detected = sample_assignments >= 0
        if np.any(detected):
            observed[detected, sample] = np.asarray(detections)[
                sample_assignments[detected], :2
            ]

    corrections = np.zeros_like(priors)
    timeline = np.arange(sample_count, dtype=np.float64)
    for owner in range(owner_count):
        supported = np.flatnonzero(assignments[owner] >= 0)
        if len(supported) < config.minimum_support_samples:
            assignments[owner] = -1
            costs[owner] = config.unmatched_cost
            continue
        raw = observed[owner, supported] - priors[owner, supported]
        preliminary = np.column_stack(
            [
                np.interp(timeline, supported, raw[:, axis])
                for axis in range(2)
            ]
        )
        window = min(config.smoothing_window, sample_count)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            preliminary = savgol_filter(
                preliminary,
                window_length=window,
                polyorder=min(2, window - 1),
                axis=0,
                mode="interp",
            )
        deviation = np.linalg.norm(raw - preliminary[supported], axis=1)
        robust = deviation <= (
            config.maximum_temporal_residual_radii * owner_radius_px
        )
        supported = supported[robust]
        if len(supported) < config.minimum_support_samples:
            assignments[owner] = -1
            costs[owner] = config.unmatched_cost
            continue
        raw = observed[owner, supported] - priors[owner, supported]
        correction = np.column_stack(
            [
                np.interp(timeline, supported, raw[:, axis])
                for axis in range(2)
            ]
        )
        if window >= 3:
            correction = savgol_filter(
                correction,
                window_length=window,
                polyorder=min(2, window - 1),
                axis=0,
                mode="interp",
            )
        insertion = np.searchsorted(supported, np.arange(sample_count))
        left = supported[np.clip(insertion - 1, 0, len(supported) - 1)]
        right = supported[np.clip(insertion, 0, len(supported) - 1)]
        nearest_distance = np.minimum(
            np.abs(np.arange(sample_count) - left),
            np.abs(right - np.arange(sample_count)),
        )
        support_radius = max(
            1.0,
            config.maximum_unsupported_windows * config.smoothing_window,
        )
        support_weight = np.clip(
            1.0 - nearest_distance / support_radius,
            0.0,
            1.0,
        )
        correction *= support_weight[:, None]
        corrections[owner] = correction
        rejected = np.ones(sample_count, dtype=bool)
        rejected[supported] = False
        assignments[owner, rejected] = -1
        costs[owner, rejected] = config.unmatched_cost
    return priors + corrections, assignments, costs, corrections


def anchor_trajectories_to_identity_observations(
    source_frames: np.ndarray,
    track_ids: np.ndarray,
    source_tracks_yx: np.ndarray,
    identity_tracks: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothly correct dense trajectories through all semantic identity anchors."""

    source_frames = np.asarray(source_frames, dtype=np.float64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    source_tracks = np.asarray(source_tracks_yx, dtype=np.float64)
    if source_frames.ndim != 1 or len(source_frames) < 2:
        raise ValueError("source frame schedule must contain at least two frames")
    if np.any(np.diff(source_frames) <= 0):
        raise ValueError("source frame schedule must be strictly increasing")
    if source_tracks.shape != (len(track_ids), len(source_frames), 2):
        raise ValueError("source owner trajectories have an invalid shape")
    if len(np.unique(track_ids)) != len(track_ids):
        raise ValueError("track IDs must be unique")

    identity_by_id = {
        int(track["track_id"]): track for track in identity_tracks
    }
    missing = set(int(track_id) for track_id in track_ids) - set(identity_by_id)
    if missing:
        raise ValueError(f"identity observations are missing owners: {sorted(missing)}")

    corrections = np.zeros_like(source_tracks)
    anchor_counts = np.zeros(len(track_ids), dtype=np.int64)
    for owner, track_id in enumerate(track_ids):
        identity = identity_by_id[int(track_id)]
        anchor_frames = np.asarray(identity["source_frames"], dtype=np.float64)
        anchor_centers = np.asarray(identity["centers_yx"], dtype=np.float64)
        if (
            anchor_frames.ndim != 1
            or anchor_centers.shape != (len(anchor_frames), 2)
            or len(anchor_frames) == 0
            or np.any(np.diff(anchor_frames) <= 0)
        ):
            raise ValueError(f"owner {int(track_id)} has invalid identity anchors")
        predicted_at_anchors = np.column_stack(
            [
                np.interp(
                    anchor_frames,
                    source_frames,
                    source_tracks[owner, :, axis],
                )
                for axis in range(2)
            ]
        )
        anchor_residuals = anchor_centers - predicted_at_anchors
        corrections[owner] = np.column_stack(
            [
                np.interp(
                    source_frames,
                    anchor_frames,
                    anchor_residuals[:, axis],
                )
                for axis in range(2)
            ]
        )
        anchor_counts[owner] = len(anchor_frames)
    return source_tracks + corrections, corrections, anchor_counts


def fuse_trajectories_with_identity_span(
    source_frames: np.ndarray,
    track_ids: np.ndarray,
    anchored_multipoint_yx: np.ndarray,
    anchored_template_yx: np.ndarray,
    identity_tracks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Use point consensus only where each owner has semantic identity support."""

    source_frames = np.asarray(source_frames, dtype=np.float64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    multipoint = np.asarray(anchored_multipoint_yx, dtype=np.float64)
    template = np.asarray(anchored_template_yx, dtype=np.float64)
    expected_shape = (len(track_ids), len(source_frames), 2)
    if multipoint.shape != expected_shape or template.shape != expected_shape:
        raise ValueError("anchored owner trajectories have an invalid shape")
    identity_by_id = {
        int(track["track_id"]): track for track in identity_tracks
    }
    if set(int(track_id) for track_id in track_ids) - set(identity_by_id):
        raise ValueError("identity spans are missing one or more owners")

    fused = template.copy()
    template_used = np.ones(expected_shape[:2], dtype=bool)
    for owner, track_id in enumerate(track_ids):
        identity = identity_by_id[int(track_id)]
        anchor_frames = np.asarray(identity["source_frames"], dtype=np.float64)
        if len(anchor_frames) == 0:
            raise ValueError(f"owner {int(track_id)} has no identity span")
        supported = (source_frames >= anchor_frames[0]) & (
            source_frames <= anchor_frames[-1]
        )
        fused[owner, supported] = multipoint[owner, supported]
        template_used[owner, supported] = False
    return fused, template_used


def semantic_anchor_guard_weights(
    source_frames: np.ndarray,
    track_ids: np.ndarray,
    identity_tracks: list[dict],
    *,
    guard_samples: float = 1.0,
    recovery_samples: float = 4.0,
) -> np.ndarray:
    """Protect trusted semantic checkpoints from geometric trajectory updates."""

    frames = np.asarray(source_frames, dtype=np.float64)
    ids = np.asarray(track_ids, dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError("source frame schedule must be strictly increasing")
    if guard_samples < 0.0 or recovery_samples <= guard_samples:
        raise ValueError("anchor recovery must exceed the nonnegative guard")
    identity_by_id = {int(track["track_id"]): track for track in identity_tracks}
    missing = set(int(track_id) for track_id in ids) - set(identity_by_id)
    if missing:
        raise ValueError(f"identity anchors are missing owners: {sorted(missing)}")
    sample_spacing = float(np.median(np.diff(frames)))
    weights = np.ones((len(ids), len(frames)), dtype=np.float64)
    for owner, track_id in enumerate(ids):
        anchors = np.asarray(
            identity_by_id[int(track_id)]["source_frames"], dtype=np.float64
        )
        if anchors.ndim != 1 or not len(anchors):
            raise ValueError(f"owner {int(track_id)} has no identity anchors")
        distance = np.min(np.abs(frames[:, None] - anchors[None]), axis=1)
        distance /= sample_spacing
        weights[owner] = np.clip(
            (distance - guard_samples) / (recovery_samples - guard_samples),
            0.0,
            1.0,
        )
    return weights


def _sha256_file(path):
    """Hash a checkpoint without reading it into one large allocation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_default_cotracker_checkpoint(destination=None):
    """Download and verify the official CoTracker3 online checkpoint."""
    try:
        import certifi
    except ImportError as error:
        raise RuntimeError(
            "Model installation requires the pollen-motion dependencies."
        ) from error
    config = PollenMotionConfig.from_environment()
    destination = Path(destination or config.checkpoint).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(
            COTRACKER_CHECKPOINT_URL,
            headers={"User-Agent": "TubeTracker model installer"},
        )
        with urllib.request.urlopen(request, context=context) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    digest = _sha256_file(destination)
    if digest != COTRACKER_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"CoTracker checkpoint checksum mismatch at {destination}; remove "
            "it and install the model again"
        )
    return destination


def pollen_body_query_points(center_xy, pollen_radius, config=None):
    """Place one center landmark and an even ring inside the pollen body."""
    config = config or PollenMotionConfig.from_environment()
    if config.ring_point_count < 3:
        raise ValueError("ring_point_count must be at least three")
    center = np.asarray(center_xy, dtype=np.float64)
    angles = np.linspace(
        0.0, 2.0 * np.pi, config.ring_point_count, endpoint=False
    )
    radius = float(pollen_radius) * float(config.ring_radius_fraction)
    ring = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((center, ring))


def _fixed_crop_rgb_frames(grays, center_xy, crop_size):
    """Create a fixed reflected crop for point tracking around one initial grain."""
    center = np.asarray(center_xy, dtype=np.float64)
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    padding = crop_size
    frames = []
    for gray in grays:
        source = np.asarray(gray, dtype=np.uint8)
        padded = cv.copyMakeBorder(
            source,
            padding,
            padding,
            padding,
            padding,
            cv.BORDER_REFLECT101,
        )
        crop = cv.getRectSubPix(
            padded,
            (crop_size, crop_size),
            (float(center[0] + padding), float(center[1] + padding)),
        )
        frames.append(np.repeat(crop[..., None], 3, axis=2))
    return np.stack(frames)


def _resolve_device(requested):
    """Select an available accelerator while allowing an explicit override."""
    import torch

    requested = str(requested).lower()
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU acceleration is unavailable")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("NVIDIA GPU acceleration is unavailable")
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_online_model(config, device):
    """Load the pinned optional CoTracker package with actionable errors."""
    try:
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor
    except ImportError as error:
        raise RuntimeError(
            "Pollen motion support is not installed. Install the pollen-motion "
            "extra described in README.md."
        ) from error
    if not config.checkpoint.exists():
        raise FileNotFoundError(
            f"CoTracker checkpoint not found at {config.checkpoint}. Install "
            "the external model once before tracking pollen motion."
        )
    return CoTrackerOnlinePredictor(checkpoint=str(config.checkpoint)).to(
        device
    )


def _online_query_tracks(
    model,
    frames_rgb,
    query_points_txy,
    device,
    add_support_grid=True,
):
    """Track arbitrary space-time queries with CoTracker's online model."""
    import torch

    frames = np.asarray(frames_rgb, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape (frames, height, width, 3)")
    if len(frames) < 2:
        raise ValueError("at least two frames are required for point tracking")
    step = int(model.step)
    if step < 1:
        raise RuntimeError("CoTracker returned an invalid online window step")
    queries = np.asarray(query_points_txy, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries):
        raise ValueError("query_points_txy must have shape (points, 3)")
    if not np.isfinite(queries).all():
        raise ValueError("query points must be finite")
    if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= len(frames)):
        raise ValueError("query frame lies outside the video")
    if (
        np.any(queries[:, 1] < 0)
        or np.any(queries[:, 1] >= frames.shape[2])
        or np.any(queries[:, 2] < 0)
        or np.any(queries[:, 2] >= frames.shape[1])
    ):
        raise ValueError("query coordinate lies outside the video")
    queries = queries[None]

    def tensor(window):
        return (
            torch.from_numpy(np.ascontiguousarray(window))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(device)
        )

    first = frames[: min(len(frames), 2 * step)]
    model(
        video_chunk=tensor(first),
        is_first_step=True,
        queries=torch.from_numpy(queries).to(device),
        add_support_grid=bool(add_support_grid),
    )
    tracks = visibility = None
    for start in range(0, max(1, len(frames) - step), step):
        window = frames[start : min(len(frames), start + 2 * step)]
        tracks, visibility = model(
            video_chunk=tensor(window),
            add_support_grid=bool(add_support_grid),
        )
    if tracks is None or visibility is None:
        raise RuntimeError("CoTracker did not return a trajectory")
    tracks = tracks[0, : len(frames)].detach().cpu().numpy()
    visibility = visibility[0, : len(frames)].detach().cpu().numpy().astype(bool)
    if len(tracks) != len(frames):
        raise RuntimeError("CoTracker trajectory does not cover every input frame")
    return tracks, visibility


def _online_point_tracks(
    model,
    frames_rgb,
    query_points_xy,
    device,
    add_support_grid=True,
):
    """Track points introduced in the first frame of an online sequence."""
    points = np.asarray(query_points_xy, dtype=np.float32)
    queries = np.column_stack(
        (np.zeros(len(points), dtype=np.float32), points)
    )
    return _online_query_tracks(
        model,
        frames_rgb,
        queries,
        device,
        add_support_grid=add_support_grid,
    )


def track_query_points(grays, query_points_txy, config=None):
    """Track points introduced at different frames in one grayscale video."""
    config = config or PollenMotionConfig.from_environment()
    grays = np.asarray(grays, dtype=np.uint8)
    if grays.ndim != 3:
        raise ValueError("grays must have shape (frames, height, width)")
    frames = np.repeat(grays[..., None], 3, axis=3)
    device = _resolve_device(config.device)
    model = _load_online_model(config, device)
    return _online_query_tracks(
        model,
        frames,
        query_points_txy,
        device,
        add_support_grid=config.add_support_grid,
    )


def track_query_points_bidirectional(grays, query_points_txy, config=None):
    """Track each space-time query forward and backward from its seed frame."""
    config = config or PollenMotionConfig.from_environment()
    grays = np.asarray(grays, dtype=np.uint8)
    queries = np.asarray(query_points_txy, dtype=np.float32)
    forward_tracks, forward_visibility = track_query_points(
        grays, queries, config=config
    )
    reversed_queries = queries.copy()
    reversed_queries[:, 0] = len(grays) - 1 - reversed_queries[:, 0]
    reversed_tracks, reversed_visibility = track_query_points(
        grays[::-1], reversed_queries, config=config
    )
    backward_tracks = reversed_tracks[::-1]
    backward_visibility = reversed_visibility[::-1]
    tracks = forward_tracks.copy()
    visibility = forward_visibility.copy()
    for query, frame in enumerate(np.rint(queries[:, 0]).astype(int)):
        tracks[:frame, query] = backward_tracks[:frame, query]
        visibility[:frame, query] = backward_visibility[:frame, query]
    return tracks, visibility


def track_pollen_bodies_from_seeds(
    grays,
    seed_centers_xy,
    pollen_radii,
    seed_samples=None,
    config=None,
):
    """Track many pollen bodies in one shared full-field CoTracker pass."""

    config = config or PollenMotionConfig.from_environment()
    grays = np.asarray(grays, dtype=np.uint8)
    centers = np.asarray(seed_centers_xy, dtype=np.float64)
    radii = np.broadcast_to(
        np.asarray(pollen_radii, dtype=np.float64), (len(centers),)
    )
    if grays.ndim != 3:
        raise ValueError("grays must have shape (frames, height, width)")
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
        raise ValueError("seed_centers_xy must have shape (owners, 2)")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError("pollen seeds and radii must be finite")
    if np.any(radii <= 0):
        raise ValueError("pollen radii must be positive")
    if seed_samples is None:
        samples = np.zeros(len(centers), dtype=np.int32)
    else:
        samples = np.asarray(seed_samples, dtype=np.int32)
    if samples.shape != (len(centers),):
        raise ValueError("one seed sample is required for every pollen body")
    if np.any(samples < 0) or np.any(samples >= len(grays)):
        raise ValueError("seed samples must lie within the frame sequence")

    point_groups = [
        pollen_body_query_points(center, radius, config)
        for center, radius in zip(centers, radii)
    ]
    points_per_owner = len(point_groups[0])
    queries = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(points_per_owner, sample, dtype=np.float64),
                    points,
                )
            )
            for points, sample in zip(point_groups, samples)
        ]
    )
    if np.all(samples == 0):
        tracks, visibility = track_query_points(grays, queries, config=config)
    else:
        tracks, visibility = track_query_points_bidirectional(
            grays, queries, config=config
        )

    trajectories = []
    for owner, (points, center) in enumerate(zip(point_groups, centers)):
        start = owner * points_per_owner
        stop = start + points_per_owner
        trajectories.append(
            pollen_trajectory_from_tracks(
                tracks[:, start:stop],
                visibility[:, start:stop],
                points,
                center,
                config=config,
            )
        )
    return trajectories


def track_pollen_bodies_in_crop_mosaics(
    grays,
    seed_centers_xy,
    pollen_radii,
    seed_samples=None,
    *,
    batch_size=6,
    gap=24,
    config=None,
):
    """Track native-detail fixed crops in small isolated CoTracker mosaics."""

    config = config or PollenMotionConfig.from_environment()
    grays = np.asarray(grays, dtype=np.uint8)
    centers = np.asarray(seed_centers_xy, dtype=np.float64)
    radii = np.broadcast_to(
        np.asarray(pollen_radii, dtype=np.float64), (len(centers),)
    )
    samples = (
        np.zeros(len(centers), dtype=np.int32)
        if seed_samples is None
        else np.asarray(seed_samples, dtype=np.int32)
    )
    if grays.ndim != 3:
        raise ValueError("grays must have shape (frames, height, width)")
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
        raise ValueError("seed_centers_xy must have shape (owners, 2)")
    if samples.shape != (len(centers),):
        raise ValueError("one seed sample is required for every pollen body")
    if np.any(samples < 0) or np.any(samples >= len(grays)):
        raise ValueError("seed samples must lie within the frame sequence")
    if np.any(radii <= 0) or not np.isfinite(radii).all():
        raise ValueError("pollen radii must be positive and finite")
    batch_size = int(batch_size)
    gap = int(gap)
    crop_size = int(config.crop_size)
    if batch_size < 1 or gap < 0 or crop_size < 1:
        raise ValueError("mosaic geometry must be nonnegative and nonempty")

    trajectories = []
    for batch_start in range(0, len(centers), batch_size):
        batch_stop = min(len(centers), batch_start + batch_size)
        batch_centers = centers[batch_start:batch_stop]
        batch_radii = radii[batch_start:batch_stop]
        batch_samples = samples[batch_start:batch_stop]
        columns = min(3, len(batch_centers))
        rows = int(np.ceil(len(batch_centers) / columns))
        mosaic_width = columns * crop_size + (columns + 1) * gap
        mosaic_height = rows * crop_size + (rows + 1) * gap
        background = int(np.median(grays[:, ::8, ::8]))
        mosaics = np.full(
            (len(grays), mosaic_height, mosaic_width),
            background,
            dtype=np.uint8,
        )
        origins = []
        local_points = []
        queries = []
        for owner, (center, radius, sample) in enumerate(
            zip(batch_centers, batch_radii, batch_samples)
        ):
            row, column = divmod(owner, columns)
            origin = np.asarray(
                [
                    gap + column * (crop_size + gap),
                    gap + row * (crop_size + gap),
                ],
                dtype=np.float64,
            )
            origins.append(origin)
            crop_rgb = _fixed_crop_rgb_frames(grays, center, crop_size)
            x0, y0 = np.rint(origin).astype(int)
            mosaics[:, y0 : y0 + crop_size, x0 : x0 + crop_size] = crop_rgb[..., 0]
            local_center = np.full(2, crop_size / 2.0)
            points = pollen_body_query_points(local_center, radius, config)
            local_points.append(points)
            mosaic_points = points + origin
            queries.append(
                np.column_stack(
                    (
                        np.full(len(points), sample, dtype=np.float64),
                        mosaic_points,
                    )
                )
            )
        queries = np.concatenate(queries)
        if np.all(batch_samples == 0):
            tracks, visibility = track_query_points(mosaics, queries, config=config)
        else:
            tracks, visibility = track_query_points_bidirectional(
                mosaics, queries, config=config
            )
        points_per_owner = len(local_points[0])
        for owner, (center, points, origin) in enumerate(
            zip(batch_centers, local_points, origins)
        ):
            start = owner * points_per_owner
            stop = start + points_per_owner
            trajectories.append(
                pollen_trajectory_from_tracks(
                    tracks[:, start:stop] - origin,
                    visibility[:, start:stop],
                    points,
                    center,
                    config=config,
                )
            )
    return trajectories


def _global_rigid_transforms(initial_center_xy, centers_xy, angles_radians):
    """Build initial-source to current-source transforms for every pollen pose."""
    initial = np.asarray(initial_center_xy, dtype=np.float64)
    matrices = []
    initial_angle = float(angles_radians[0])
    for center, angle in zip(centers_xy, angles_radians):
        relative = float(angle) - initial_angle
        cosine = float(np.cos(relative))
        sine = float(np.sin(relative))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        translation = np.asarray(center) - rotation @ initial
        matrices.append(np.column_stack((rotation, translation)))
    return np.stack(matrices)


def track_pollen_body(grays, initial_center_xy, pollen_radius, config=None):
    """Track redundant body landmarks and recover a robust global pollen pose."""
    config = config or PollenMotionConfig.from_environment()
    grays = np.asarray(grays)
    frames = _fixed_crop_rgb_frames(
        grays, initial_center_xy, config.crop_size
    )
    local_center = np.asarray(
        [config.crop_size / 2.0, config.crop_size / 2.0]
    )
    initial_points = pollen_body_query_points(
        local_center, pollen_radius, config
    )
    device = _resolve_device(config.device)
    model = _load_online_model(config, device)
    tracks, visibility = _online_point_tracks(
        model,
        frames,
        initial_points,
        device,
        add_support_grid=config.add_support_grid,
    )
    return pollen_trajectory_from_tracks(
        tracks,
        visibility,
        initial_points,
        initial_center_xy,
        config=config,
    )


def pollen_trajectory_from_tracks(
    tracks_xy,
    visibility,
    initial_points_xy,
    initial_center_xy,
    config=None,
):
    """Recover a global pollen pose from cached local body-point tracks."""
    config = config or PollenMotionConfig.from_environment()
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    initial_points = np.asarray(initial_points_xy, dtype=np.float64)
    pose = estimate_grain_poses(
        initial_points, tracks, visibility, config=config.pose
    )
    crop_origin = (
        np.asarray(initial_center_xy, dtype=np.float64) - initial_points[0]
    )
    centers = pose.centers + crop_origin
    transforms = _global_rigid_transforms(
        initial_center_xy, centers, pose.angles_radians
    )
    return PollenBodyTrajectory(
        centers_xy=centers,
        angles_radians=pose.angles_radians,
        transforms=transforms,
        observed=pose.observed,
        visible_point_count=pose.visible_point_count,
        inlier_point_count=pose.inlier_point_count,
        median_residual_px=pose.median_residual_px,
        tracks_xy=tracks,
        visibility=visibility,
        initial_points_xy=initial_points,
        crop_origin_xy=crop_origin,
    )
