#!/usr/bin/env python3
"""Trace germination events as birth-time topology, not framewise detections.

This prototype fixes three structural problems found during the v16 audit:

* persistence is local in time instead of an all-video cumulative count;
* source-frame timing is converted to seconds exactly once;
* each reported tip is the end of one simple path connected back to material
  that was already present before germination.
* threshold-timing jumps can be refined only after path identity and
  germination acceptance are frozen.

The output contains an event summary, a detailed length/tip CSV, a birth-time
map, and a review video on stabilized raw imagery. It intentionally keeps
pre-existing tubes out of the event map rather than recoloring the whole field.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from skimage.morphology import skeletonize

from tubetracker.growth_front import (
    PathLockedFrontResult,
    isotonic_increasing,
    path_novelty_profiles,
    path_locked_temporal_front,
)
from tubetracker.material_ribbon import (
    MaterialRibbonConfig,
    observe_material_ribbon,
)


@dataclass(frozen=True)
class AtlasConfig:
    """Configuration for birth detection and pollen-anchored path recovery."""

    width: int = 480
    warmup_samples: int = 12
    persistence_window: int = 5
    persistence_required: int = 3
    preexisting_min_observations: int = 2
    preexisting_dilation_px: int = 1
    preexisting_grain_protection_px: float = 5.0
    spatial_link_px: int = 2
    temporal_link_samples: int = 64
    min_event_pixels: int = 24
    min_path_length_px: float = 12.0
    min_germination_length_px: float = 3.0
    min_radial_extension_px: float = 3.0
    min_radial_extension_ratio: float = 0.25
    max_root_distance_px: float = 12.0
    max_grain_attachment_px: float = 8.0
    foreign_grain_clearance_margin_px: float = 1.0
    max_path_tortuosity: float = 3.0
    grain_exit_margin_px: float = 2.0
    grain_exit_run_points: int = 6
    trajectory_resolution_samples: int = 20
    min_birth_order_correlation: float = 0.2
    min_birth_order_forward_fraction: float = 0.6
    min_growth_step_count: int = 5
    max_tip_step_median_px: float = 1.5
    max_tip_step_px: float = 3.75
    max_event_width_px: float = 12.0
    grain_anchor_view_count: int = 24
    grain_anchor_warmup_view_count: int = 3
    grain_anchor_min_warmup_observations: int = 2
    grain_anchor_match_distance_px: float = 8.0
    grain_anchor_motion_px_per_sample: float = 0.08
    grain_anchor_max_match_distance_px: float = 20.0
    grain_anchor_max_displacement_px: float = 10.0
    grain_anchor_dedup_radius_ratio: float = 0.6
    grain_detection_blur_radius_px: int = 6
    grain_detection_min_radius_px: int = 3
    grain_detection_max_radius_px: int = 9
    grain_detection_threshold: int = 8
    grain_detection_min_circle_score: float = 0.20
    branch_direction_floor_weight: float = 0.2
    atlas_branch_weight: float = 0.65
    junction_direction_window_px: float = 5.0
    junction_turn_soft_limit_degrees: float = 45.0
    junction_turn_penalty_weight: float = 2.0
    contact_bridge_max_turn_degrees: float = 45.0
    minimum_root_score_factor: float = 0.1
    full_duration_score_samples: int = 10
    nominal_tube_width_px: float = 6.0
    competing_path_distance_px: float = 2.0
    competing_path_overlap_fraction: float = 0.6
    rim_history_samples: int = 12
    rim_persistence_window: int = 5
    rim_persistence_required: int = 3
    rim_path_length_px: float = 5.0
    rim_signal_percentile: float = 80.0
    rim_min_delta: float = 4.0
    rim_noise_multiplier: float = 2.5
    rim_contrast_halfwidth_px: float = 1.0
    rim_contrast_flank_offset_px: float = 3.0
    rim_contrast_min_delta: float = 2.0
    occlusion_blur_radius_px: float = 12.0
    occlusion_min_area_px: int = 400
    occlusion_min_dark_delta: float = 8.0
    occlusion_noise_multiplier: float = 2.5
    max_path_occluded_fraction: float = 0.5
    warmup_review_min_ribbon_fraction: float = 0.35
    warmup_contact_review_min_ribbon_fraction: float = 0.30
    proximal_front_length_px: float = 8.0
    rim_direction_min_rotation_degrees: int = 60
    rim_direction_rotation_step_degrees: int = 30
    rim_direction_normal_halfwidth_px: float = 1.0
    rim_direction_min_connected_fraction: float = 0.8
    rim_direction_timing_tolerance_samples: int = 12
    front_normal_halfwidth_px: float = 2.0
    front_normal_sample_count: int = 5
    front_temporal_median_samples: int = 3
    front_absence_weight: float = 0.05
    front_motion_penalty: float = 0.25
    front_min_eventual_support_fraction: float = 0.8
    front_min_direct_support_fraction: float = 0.8

    @classmethod
    def for_width(cls, width: int, **overrides: object) -> "AtlasConfig":
        """Build a configuration whose pixel gates scale with image width."""

        base = cls()
        scale = width / base.width
        values: dict[str, object] = {
            "width": width,
            "spatial_link_px": max(1, int(round(base.spatial_link_px * scale))),
            "preexisting_dilation_px": max(
                1, int(round(base.preexisting_dilation_px * scale))
            ),
            "preexisting_grain_protection_px": (
                base.preexisting_grain_protection_px * scale
            ),
            "min_event_pixels": max(4, int(round(base.min_event_pixels * scale**2))),
            "min_path_length_px": base.min_path_length_px * scale,
            "min_germination_length_px": base.min_germination_length_px * scale,
            "min_radial_extension_px": base.min_radial_extension_px * scale,
            "max_root_distance_px": base.max_root_distance_px * scale,
            "max_grain_attachment_px": base.max_grain_attachment_px * scale,
            "foreign_grain_clearance_margin_px": (
                base.foreign_grain_clearance_margin_px * scale
            ),
            "grain_exit_margin_px": base.grain_exit_margin_px * scale,
            "max_tip_step_median_px": base.max_tip_step_median_px * scale,
            "max_tip_step_px": base.max_tip_step_px * scale,
            "max_event_width_px": base.max_event_width_px * scale,
            "nominal_tube_width_px": base.nominal_tube_width_px * scale,
            "junction_direction_window_px": (
                base.junction_direction_window_px * scale
            ),
            "competing_path_distance_px": (
                base.competing_path_distance_px * scale
            ),
            "rim_path_length_px": base.rim_path_length_px * scale,
            "rim_contrast_halfwidth_px": (
                base.rim_contrast_halfwidth_px * scale
            ),
            "rim_contrast_flank_offset_px": (
                base.rim_contrast_flank_offset_px * scale
            ),
            "rim_min_delta": base.rim_min_delta,
            "occlusion_blur_radius_px": (
                base.occlusion_blur_radius_px * scale
            ),
            "occlusion_min_area_px": max(
                16, int(round(base.occlusion_min_area_px * scale**2))
            ),
            "front_normal_halfwidth_px": (
                base.front_normal_halfwidth_px * scale
            ),
            "proximal_front_length_px": (
                base.proximal_front_length_px * scale
            ),
            "rim_direction_normal_halfwidth_px": (
                base.rim_direction_normal_halfwidth_px * scale
            ),
            "grain_anchor_match_distance_px": (
                base.grain_anchor_match_distance_px * scale
            ),
            "grain_anchor_motion_px_per_sample": (
                base.grain_anchor_motion_px_per_sample * scale
            ),
            "grain_anchor_max_match_distance_px": (
                base.grain_anchor_max_match_distance_px * scale
            ),
            "grain_anchor_max_displacement_px": (
                base.grain_anchor_max_displacement_px * scale
            ),
            "grain_detection_blur_radius_px": max(
                1, int(round(base.grain_detection_blur_radius_px * scale))
            ),
            "grain_detection_min_radius_px": max(
                2, int(round(base.grain_detection_min_radius_px * scale))
            ),
            "grain_detection_max_radius_px": max(
                3, int(round(base.grain_detection_max_radius_px * scale))
            ),
        }
        values.update(overrides)
        return cls(**values)

    def for_sample_interval(
        self,
        sample_seconds: float,
        reference_seconds: float = 3.0,
        **overrides: object,
    ) -> "AtlasConfig":
        """Preserve time-domain gates when the analysis cadence changes."""

        if sample_seconds <= 0 or reference_seconds <= 0:
            raise ValueError("sample intervals must be positive")
        rate = reference_seconds / sample_seconds

        def count(value: int) -> int:
            return max(1, int(round(value * rate)))

        def inclusive_span(value: int) -> int:
            return max(1, int(round((value - 1) * rate)) + 1)

        values: dict[str, object] = {
            "warmup_samples": count(self.warmup_samples),
            "persistence_window": inclusive_span(self.persistence_window),
            "persistence_required": inclusive_span(self.persistence_required),
            "preexisting_min_observations": inclusive_span(
                self.preexisting_min_observations
            ),
            "temporal_link_samples": count(self.temporal_link_samples),
            "trajectory_resolution_samples": count(
                self.trajectory_resolution_samples
            ),
            "grain_anchor_warmup_view_count": count(
                self.grain_anchor_warmup_view_count
            ),
            "grain_anchor_motion_px_per_sample": (
                self.grain_anchor_motion_px_per_sample / rate
            ),
            "full_duration_score_samples": count(
                self.full_duration_score_samples
            ),
            "rim_history_samples": count(self.rim_history_samples),
            "rim_persistence_window": inclusive_span(
                self.rim_persistence_window
            ),
            "rim_persistence_required": inclusive_span(
                self.rim_persistence_required
            ),
            "rim_direction_timing_tolerance_samples": count(
                self.rim_direction_timing_tolerance_samples
            ),
            "front_temporal_median_samples": inclusive_span(
                self.front_temporal_median_samples
            ),
        }
        values.update(overrides)
        return replace(self, **values)


@dataclass(frozen=True)
class GrainAnchor:
    """A pollen identity with sparse observations across analysis time."""

    grain_id: int
    center_xy: np.ndarray
    radius_px: float
    observations: int
    circle_score: float
    warmup_observations: int | None = None
    sample_indices: np.ndarray | None = None
    centers_xy: np.ndarray | None = None

    def center_at(self, sample: int) -> np.ndarray:
        """Interpolate this grain's position at one analysis sample."""

        if (
            self.sample_indices is None
            or self.centers_xy is None
            or len(self.sample_indices) == 0
        ):
            return self.center_xy.copy()
        samples = np.asarray(self.sample_indices, dtype=np.float64)
        centers = np.asarray(self.centers_xy, dtype=np.float64)
        return np.asarray(
            [
                np.interp(sample, samples, centers[:, axis])
                for axis in range(2)
            ],
            dtype=np.float64,
        )


@dataclass
class BirthEvent:
    """One connected construction event and its simple root-to-tip path."""

    event_id: int
    component_id: int
    component_root_count: int
    area_px: int
    path_xy: np.ndarray
    path_birth_raw: np.ndarray
    path_birth: np.ndarray
    arclength_px: np.ndarray
    root_distance_px: float
    tortuosity: float
    width_px: float
    max_junction_turn_degrees: float
    birth_order_correlation: float
    birth_order_forward_fraction: float
    birth_progress_samples: float
    growth_step_count: int
    tip_step_median_px: float
    tip_step_max_px: float
    length_step_max_px: float
    birth_start: int
    germination_sample: int
    birth_end: int
    score: float
    rim_emergence_delta: float
    rim_emergence_noise: float
    rim_emergence_persistent_samples: int
    rim_emergence_confirmed: bool | None
    rim_emergence_sample: int | None
    rim_emergence_timing_error_samples: int | None
    rim_contrast_delta: float
    rim_contrast_noise: float
    rim_contrast_persistent_samples: int
    rim_contrast_confirmed: bool
    rim_contrast_emergence_sample: int | None
    grain_id: int | None = None
    grain_center_xy: np.ndarray | None = None
    grain_radius_px: float | None = None
    grain_attachment_px: float | None = None
    radial_extension_px: float | None = None
    foreign_grain_clearance_px: float | None = None
    quality_status: str = "review-required"
    quality_flags: tuple[str, ...] = ()
    auto_accepted: bool = False
    trajectory_accepted: bool = False
    primary_for_grain: bool = False
    broad_occlusion_fraction: float | None = None
    warmup_prefix_length_px: float = 0.0
    warmup_ribbon_fraction: float = 0.0
    germination_left_censored: bool = False
    warmup_attachment_ambiguous: bool = False
    proximal_front_verified: bool = False
    proximal_emergence_ambiguous: bool = False
    proximal_front_reason: str = "not-evaluated"
    proximal_eventual_support_fraction: float = 0.0
    proximal_direct_support_fraction: float = 0.0
    proximal_inferred_fraction: float = 0.0
    proximal_median_confirmation_lag_samples: float = 0.0
    proximal_max_confirmation_lag_samples: int = 0
    proximal_reference_motion_max_px: float = 0.0
    proximal_static_front_reason: str = "not-evaluated"
    proximal_static_eventual_support_fraction: float = 0.0
    proximal_static_direct_support_fraction: float = 0.0
    proximal_motion_binding_class: str = "not-evaluated"
    proximal_motion_support_margin: float = 0.0
    proximal_rim_bridge_evaluated: bool = False
    proximal_rim_bridge_accepted: bool = False
    proximal_rim_bridge_reason: str = "not-evaluated"
    proximal_rim_bridge_direct_support_fraction: float = 0.0
    proximal_rim_bridge_eventual_support_fraction: float = 0.0
    proximal_rim_bridge_start_sample: int | None = None
    proximal_rim_bridge_end_sample: int | None = None
    rim_direction_specificity_evaluated: bool = False
    rim_direction_specific: bool | None = None
    rim_direction_competitor_count: int = 0
    rim_direction_competitor_rotation_degrees: int | None = None
    rim_direction_competitor_sample: int | None = None
    rim_direction_selected_connected_sample: int | None = None
    rim_direction_reason: str = "not-evaluated"
    path_front_birth: np.ndarray | None = None
    path_front_direct_support: np.ndarray | None = None
    path_front_eventual_support: np.ndarray | None = None
    front_refinement_applied: bool = False
    front_refinement_reason: str = "not-attempted"
    front_eventual_support_fraction: float = 0.0
    front_direct_support_fraction: float = 0.0
    front_inferred_fraction: float = 0.0
    front_median_confirmation_lag_samples: float = 0.0
    front_max_confirmation_lag_samples: int = 0
    full_path_temporal_evaluated: bool = False
    full_path_temporal_verified: bool = False
    full_path_temporal_reason: str = "not-evaluated"
    observed_growth_step_count: int = 0
    observed_tip_step_median_px: float = 0.0
    observed_tip_step_max_px: float = 0.0
    observed_length_step_max_px: float = 0.0
    foreign_grain_contact_path_index: int | None = None
    foreign_grain_contact_grain_id: int | None = None
    foreign_grain_contact_exit_path_index: int | None = None
    contact_bridge_next_foreign_contact_path_index: int | None = None
    contact_bridge_hidden_length_px: float | None = None
    contact_bridge_ingress_chord_angle_degrees: float | None = None
    contact_bridge_chord_egress_angle_degrees: float | None = None
    contact_bridge_total_turn_degrees: float | None = None
    contact_bridge_identity_supported: bool = False
    contact_bridge_viable_competitor_count: int = 0
    contact_bridge_trajectory_accepted: bool = False
    contact_bridge_last_path_index: int | None = None
    contact_bridge_censor_sample: int | None = None
    contact_bridge_front_birth: np.ndarray | None = None
    contact_bridge_front_direct_support: np.ndarray | None = None
    contact_bridge_front_eventual_support: np.ndarray | None = None
    contact_bridge_reason: str = "not-evaluated"
    contact_bridge_direct_support_fraction: float = 0.0
    contact_bridge_eventual_support_fraction: float = 0.0
    contact_bridge_post_direct_support_fraction: float = 0.0
    contact_bridge_post_eventual_support_fraction: float = 0.0
    contact_bridge_growth_step_count: int = 0
    contact_bridge_post_growth_step_count: int = 0
    contact_bridge_tip_step_median_px: float = 0.0
    contact_bridge_tip_step_max_px: float = 0.0
    contact_bridge_length_step_max_px: float = 0.0
    precontact_trajectory_accepted: bool = False
    precontact_censor_sample: int | None = None
    precontact_front_birth: np.ndarray | None = None
    precontact_front_direct_support: np.ndarray | None = None
    precontact_front_eventual_support: np.ndarray | None = None
    precontact_front_reason: str = "not-evaluated"
    precontact_direct_support_fraction: float = 0.0
    precontact_eventual_support_fraction: float = 0.0
    precontact_growth_step_count: int = 0
    precontact_tip_step_median_px: float = 0.0
    precontact_tip_step_max_px: float = 0.0
    precontact_length_step_max_px: float = 0.0


@dataclass(frozen=True)
class OnsetAssessment:
    """Describe the supported onset estimate without changing event validity."""

    sample: int | None
    quality: str
    accepted: bool
    supporting_cues: tuple[str, ...]
    support_start_sample: int | None
    support_end_sample: int | None


@dataclass(frozen=True)
class TipTimingAssessment:
    """Describe the evidence supporting one reported tip time."""

    state: str
    measurement_accepted: bool
    direct_support_at_arrival: bool | None
    eventual_support_before_confirmation: bool | None
    threshold_confirmation_sample: int | None
    remaining_confirmation_lag_samples: int | None


def movie_metadata(movie: Path) -> tuple[float, int, int, int]:
    """Return FPS, frame count, width, and height from an input movie."""

    cap = cv.VideoCapture(str(movie))
    if not cap.isOpened():
        raise ValueError(f"Could not open movie: {movie}")
    fps = float(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Movie metadata is incomplete: {movie}")
    return fps, frame_count, width, height


def source_frame_indices(
    frame_count: int,
    fps: float,
    start: int,
    end: int | None,
    sample_seconds: float,
    source_frame_interval_seconds: float | None = None,
) -> np.ndarray:
    """Choose source frames at a requested real-time interval."""

    stop = frame_count if end is None else min(end, frame_count)
    if not 0 <= start < stop:
        raise ValueError(f"Invalid source window {start}:{stop}")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if source_frame_interval_seconds is not None:
        if source_frame_interval_seconds <= 0:
            raise ValueError("source_frame_interval_seconds must be positive")
        step = max(1, int(round(sample_seconds / source_frame_interval_seconds)))
    else:
        step = max(1, int(round(sample_seconds * fps)))
    return np.arange(start, stop, step, dtype=np.int64)


def sample_interval_seconds(
    source_frames: np.ndarray,
    fps: float,
    source_frame_interval_seconds: float | None = None,
) -> float:
    """Convert source-frame spacing to elapsed seconds exactly once."""

    if len(source_frames) < 2:
        raise ValueError("At least two source frames are required")
    if source_frame_interval_seconds is not None:
        if source_frame_interval_seconds <= 0:
            raise ValueError("source_frame_interval_seconds must be positive")
        seconds_per_frame = source_frame_interval_seconds
    else:
        if fps <= 0:
            raise ValueError("A positive FPS is required without time calibration")
        seconds_per_frame = 1.0 / fps
    return float(np.median(np.diff(source_frames))) * seconds_per_frame


def load_movie_samples(
    movie: Path,
    source_frames: np.ndarray,
    width: int,
) -> np.ndarray:
    """Read selected source frames as consistently sized grayscale images."""

    cap = cv.VideoCapture(str(movie))
    native_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    height = int(round(native_h * width / native_w))
    frames = np.empty((len(source_frames), height, width), dtype=np.uint8)
    for i, source_frame in enumerate(source_frames):
        cap.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[i] = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
    cap.release()
    return frames


def stabilize_translations(
    frames: np.ndarray,
    max_step_px: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove trustworthy common translation while rejecting bad estimates."""

    aligned = np.empty_like(frames)
    aligned[0] = frames[0]
    shifts = np.zeros((len(frames), 2), dtype=np.float32)
    responses = np.ones(len(frames), dtype=np.float32)
    h, w = frames.shape[1:]
    hann = cv.createHanningWindow((w, h), cv.CV_32F)

    def registration_image(frame: np.ndarray) -> np.ndarray:
        smooth = cv.GaussianBlur(frame, (0, 0), 2.0)
        return cv.Laplacian(smooth, cv.CV_32F)

    previous = registration_image(frames[0])
    cumulative = np.zeros(2, dtype=np.float64)
    for i in range(1, len(frames)):
        current = registration_image(frames[i])
        delta, response = cv.phaseCorrelate(previous, current, hann)
        delta = np.asarray(delta, dtype=np.float64)
        trustworthy = response >= 0.08 and np.linalg.norm(delta) <= max_step_px
        if trustworthy:
            cumulative += delta
        else:
            delta[:] = 0.0
        shifts[i] = cumulative
        responses[i] = float(response)
        matrix = np.array(
            [[1.0, 0.0, -cumulative[0]], [0.0, 1.0, -cumulative[1]]],
            dtype=np.float32,
        )
        aligned[i] = cv.warpAffine(
            frames[i],
            matrix,
            (w, h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        previous = current
    return aligned, shifts, responses


def ridge_evidence(frames: np.ndarray) -> np.ndarray:
    """Measure dark, narrow material while retaining faint tube walls."""

    evidence = np.empty_like(frames)
    kernels = [
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        for size in (9, 15)
    ]
    for i, frame in enumerate(frames):
        responses = [cv.morphologyEx(frame, cv.MORPH_BLACKHAT, k) for k in kernels]
        evidence[i] = np.maximum(responses[0], responses[1])
    return evidence


def broad_occlusion_mask(frame: np.ndarray, config: AtlasConfig) -> np.ndarray:
    """Find large low-frequency dark regions that can hide tube evidence."""

    if frame.ndim != 2:
        raise ValueError("occlusion input must be a grayscale frame")
    smooth = cv.GaussianBlur(
        frame.astype(np.float32),
        (0, 0),
        config.occlusion_blur_radius_px,
    )
    baseline = float(np.median(smooth))
    noise = 1.4826 * float(np.median(np.abs(smooth - baseline)))
    threshold = baseline - max(
        config.occlusion_min_dark_delta,
        config.occlusion_noise_multiplier * noise,
    )
    candidate = (smooth < threshold).astype(np.uint8)
    component_count, labels, statistics, _ = cv.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    occlusion = np.zeros(frame.shape, dtype=bool)
    for component in range(1, component_count):
        if statistics[component, cv.CC_STAT_AREA] >= config.occlusion_min_area_px:
            occlusion |= labels == component
    return occlusion


def _apply_broad_occlusion_review(
    events: list[BirthEvent],
    frames: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Demote paths hidden by a broad object while retaining their measurements."""

    if frames is None or not len(frames):
        return
    masks: dict[int, np.ndarray] = {}
    h, w = frames.shape[1:]
    for event in events:
        sample = int(np.clip(event.birth_end, 0, len(frames) - 1))
        if sample not in masks:
            masks[sample] = broad_occlusion_mask(frames[sample], config)
        mask = masks[sample]
        points = np.rint(event.path_xy).astype(np.int32)
        x = np.clip(points[:, 0], 0, w - 1)
        y = np.clip(points[:, 1], 0, h - 1)
        fraction = float(np.mean(mask[y, x])) if len(points) else 0.0
        event.broad_occlusion_fraction = fraction
        if fraction <= config.max_path_occluded_fraction:
            continue
        event.auto_accepted = False
        event.trajectory_accepted = False
        event.quality_status = "review-required"
        event.quality_flags = tuple(
            dict.fromkeys((*event.quality_flags, "broad-object-occlusion"))
        )


def _global_evidence_floor(evidence: np.ndarray, warmup: int) -> float:
    """Return the shared absolute ridge floor used during warmup."""

    baseline = np.median(evidence[:warmup].astype(np.float32), axis=0)
    return max(6.0, float(np.percentile(baseline, 85)) + 2.0)


def _warmup_path_prefix_length(
    path_xy: np.ndarray,
    arclength: np.ndarray,
    support: np.ndarray,
    maximum_gap_px: float,
) -> float:
    """Measure persistent warmup support connected outward from a pollen."""

    points = np.rint(path_xy).astype(np.int32)
    h, w = support.shape
    x = np.clip(points[:, 0], 0, w - 1)
    y = np.clip(points[:, 1], 0, h - 1)
    supported = support[y, x]
    last_supported = -1
    gap_start = None
    for index, present in enumerate(supported):
        if present:
            last_supported = index
            gap_start = None
        elif gap_start is None:
            gap_start = index
        elif arclength[index] - arclength[gap_start] > maximum_gap_px:
            break
    if last_supported < 0:
        return 0.0
    return float(arclength[last_supported])


def _apply_warmup_onset_review(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    frames: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Withhold onset claims for tubes already attached during warmup."""

    if (
        evidence is None
        or frames is None
        or len(evidence) < config.warmup_samples
        or len(frames) < config.warmup_samples
    ):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    diameter = 2 * config.spatial_link_px + 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    local_warmup = np.stack(
        [
            cv.dilate(sample, kernel)
            for sample in evidence[: config.warmup_samples]
        ]
    )
    persistent_support = (
        np.count_nonzero(local_warmup >= evidence_floor, axis=0)
        >= config.preexisting_min_observations
    )
    review_samples = np.unique(
        np.rint(
            np.linspace(0, config.warmup_samples - 1, num=3)
        ).astype(np.int32)
    )
    ribbon_config = MaterialRibbonConfig(
        center_search_radius_px=max(1, config.spatial_link_px),
    )
    minimum_prefix = (
        config.grain_exit_margin_px + config.min_germination_length_px
    )
    for event in events:
        prefix_length = _warmup_path_prefix_length(
            event.path_xy,
            event.arclength_px,
            persistent_support,
            maximum_gap_px=config.spatial_link_px + 1.0,
        )
        ribbon_fractions = [
            observe_material_ribbon(
                frames[int(sample)],
                event.path_xy[:, ::-1],
                config=ribbon_config,
            ).supported_fraction
            for sample in review_samples
        ]
        ribbon_fraction = float(np.median(ribbon_fractions))
        event.warmup_prefix_length_px = prefix_length
        event.warmup_ribbon_fraction = ribbon_fraction
        if not event.auto_accepted:
            continue
        prefix_present = prefix_length >= minimum_prefix
        ribbon_threshold = config.warmup_review_min_ribbon_fraction
        if "foreign-grain-contact" in event.quality_flags:
            ribbon_threshold = min(
                ribbon_threshold,
                config.warmup_contact_review_min_ribbon_fraction,
            )
        ribbon_present = ribbon_fraction >= ribbon_threshold
        if not prefix_present and not ribbon_present:
            continue
        event.germination_left_censored = prefix_present and ribbon_present
        event.warmup_attachment_ambiguous = not event.germination_left_censored
        event.auto_accepted = False
        if event.germination_left_censored:
            event.quality_status = (
                "left-censored-trajectory"
                if event.trajectory_accepted
                else "left-censored-growth"
            )
            review_flag = "left-censored-germination"
        else:
            event.quality_status = (
                "warmup-ambiguous-trajectory"
                if event.trajectory_accepted
                else "warmup-ambiguous-growth"
            )
            review_flag = "warmup-attachment-ambiguous"
        event.quality_flags = tuple(
            dict.fromkeys((*event.quality_flags, review_flag))
        )


def _grain_anchor_sample_indices(
    total_samples: int,
    warmup: int,
    survey_view_count: int,
    warmup_view_count: int = 3,
) -> list[int]:
    """Choose repeated warmup and field-wide views for the pollen census."""

    if total_samples <= 0:
        return []
    warmup_stop = min(total_samples, max(1, warmup))
    warmup_count = min(warmup_stop, max(1, warmup_view_count))
    warmup_indices = set(
        np.rint(
            np.linspace(0, warmup_stop - 1, warmup_count)
        ).astype(int)
    )
    survey_count = min(total_samples, max(1, survey_view_count))
    survey_indices = np.rint(
        np.linspace(0, total_samples - 1, survey_count)
    ).astype(int)
    later_survey_indices = {
        int(index) for index in survey_indices if index >= warmup_stop
    }
    return sorted(warmup_indices | later_survey_indices)


def detect_grain_anchors(
    frames: np.ndarray,
    warmup: int,
    config: AtlasConfig | None = None,
) -> list[GrainAnchor]:
    """Seed pollen in warmup, then update only those identities over time."""

    from tubetracker.curve_prototype import CurveTraceConfig, detect_grain_candidates

    config = config or AtlasConfig.for_width(frames.shape[2])
    sample_indices = _grain_anchor_sample_indices(
        len(frames),
        warmup,
        config.grain_anchor_view_count,
        config.grain_anchor_warmup_view_count,
    )
    selected = [frames[index] for index in sample_indices]
    trace_config = CurveTraceConfig(
        preprocessing="background",
        blur_radius=config.grain_detection_blur_radius_px,
        min_grain_radius=config.grain_detection_min_radius_px,
        max_grain_radius=config.grain_detection_max_radius_px,
        grain_threshold=config.grain_detection_threshold,
        min_grain_circle_score=config.grain_detection_min_circle_score,
    )
    detections = detect_grain_candidates(selected, trace_config)
    clusters: list[dict[str, object]] = []
    for view, candidates in enumerate(detections):
        candidates = sorted(
            candidates,
            key=lambda roi: float(getattr(roi, "circle_score", 0.0)),
            reverse=True,
        )
        for candidate in candidates:
            center = np.asarray([candidate.gv3.x, candidate.gv3.y], dtype=np.float64)
            radius = float(max(candidate.w, candidate.h) / 2.0)
            score = float(getattr(candidate, "circle_score", 0.0))
            sample = sample_indices[view]
            possible = []
            for index, cluster in enumerate(clusters):
                if view in cluster["views"]:
                    continue
                centers = cluster["centers"]
                samples = cluster["sample_indices"]
                initial_centers = [
                    value
                    for value, observed_sample in zip(centers, samples)
                    if observed_sample < warmup
                ]
                reference_center = np.median(initial_centers, axis=0)
                if (
                    np.linalg.norm(center - reference_center)
                    > config.grain_anchor_max_displacement_px
                ):
                    continue
                prediction = centers[-1]
                if len(centers) >= 2 and samples[-1] > samples[-2]:
                    velocity = (centers[-1] - centers[-2]) / (
                        samples[-1] - samples[-2]
                    )
                    prediction = prediction + velocity * (sample - samples[-1])
                elapsed = max(0, sample - samples[-1])
                cutoff = min(
                    config.grain_anchor_max_match_distance_px,
                    config.grain_anchor_match_distance_px
                    + config.grain_anchor_motion_px_per_sample * elapsed,
                )
                distance = float(np.linalg.norm(center - prediction))
                if distance <= cutoff:
                    possible.append((distance, index))
            if possible:
                _, index = min(possible)
                clusters[index]["centers"].append(center)
                clusters[index]["radii"].append(radius)
                clusters[index]["scores"].append(score)
                clusters[index]["views"].add(view)
                clusters[index]["sample_indices"].append(sample)
            elif sample < warmup:
                clusters.append(
                    {
                        "centers": [center],
                        "radii": [radius],
                        "scores": [score],
                        "views": {view},
                        "sample_indices": [sample],
                    }
                )

    confirmed = []
    for cluster in clusters:
        observations = len(cluster["views"])
        order = np.argsort(cluster["sample_indices"])
        centers = np.asarray(cluster["centers"], dtype=np.float64)[order]
        radii = np.asarray(cluster["radii"], dtype=np.float64)[order]
        scores = np.asarray(cluster["scores"], dtype=np.float64)[order]
        samples = np.asarray(cluster["sample_indices"], dtype=np.int32)[order]
        warmup_mask = samples < warmup
        warmup_centers = centers[warmup_mask]
        warmup_observations = len(warmup_centers)
        if (
            warmup_observations
            < config.grain_anchor_min_warmup_observations
        ):
            continue
        confirmed.append(
            (
                np.median(warmup_centers, axis=0),
                float(np.median(radii[warmup_mask])),
                observations,
                warmup_observations,
                float(np.median(scores[warmup_mask])),
                samples,
                centers,
            )
        )
    confirmed.sort(key=lambda item: (float(item[0][1]), float(item[0][0])))

    anchors: list[GrainAnchor] = []
    for (
        center,
        radius,
        observations,
        warmup_observations,
        score,
        samples,
        centers,
    ) in confirmed:
        duplicate = next(
            (
                anchor
                for anchor in anchors
                if np.linalg.norm(anchor.center_xy - center)
                <= config.grain_anchor_dedup_radius_ratio
                * max(anchor.radius_px, radius)
            ),
            None,
        )
        if duplicate is not None:
            continue
        anchors.append(
            GrainAnchor(
                grain_id=len(anchors) + 1,
                center_xy=center,
                radius_px=radius,
                observations=observations,
                circle_score=score,
                warmup_observations=warmup_observations,
                sample_indices=samples,
                centers_xy=centers,
            )
        )
    return anchors


def local_persistent_births(
    evidence: np.ndarray,
    warmup: int,
    window: int,
    required: int,
    preexisting_min_observations: int = 2,
    preexisting_dilation_px: int = 1,
    grain_anchors: list[GrainAnchor] | None = None,
    grain_protection_px: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Find first locally persistent evidence relative to a warmup baseline."""

    if not 1 <= required <= window:
        raise ValueError("persistence_required must be between 1 and window")
    if not window <= len(evidence) or not warmup >= window:
        raise ValueError("video is too short for the warmup/persistence settings")
    if not 1 <= preexisting_min_observations <= warmup:
        raise ValueError("preexisting observations must fit inside warmup")
    if preexisting_dilation_px < 0:
        raise ValueError("preexisting dilation cannot be negative")
    if grain_protection_px < 0:
        raise ValueError("grain protection cannot be negative")

    baseline_stack = evidence[:warmup].astype(np.float32)
    baseline = np.median(baseline_stack, axis=0)
    noise = 1.4826 * np.median(np.abs(baseline_stack - baseline), axis=0)
    global_floor = _global_evidence_floor(evidence, warmup)
    change_floor = np.maximum(4.0, 2.5 * noise)
    threshold = np.maximum(global_floor, baseline + change_floor)
    active = evidence >= threshold[None, :, :]

    preexisting = np.count_nonzero(
        baseline_stack >= global_floor, axis=0
    ) >= preexisting_min_observations
    preexisting = cv.morphologyEx(
        preexisting.astype(np.uint8),
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    undilated_preexisting = preexisting.copy()
    if preexisting_dilation_px:
        diameter = 2 * preexisting_dilation_px + 1
        preexisting = cv.dilate(
            preexisting.astype(np.uint8),
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter)),
        ).astype(bool)
    if grain_anchors and preexisting_dilation_px:
        protected_rims = np.zeros(preexisting.shape, dtype=np.uint8)
        for grain in grain_anchors:
            center = tuple(
                np.rint(grain.center_at(max(0, warmup - 1))).astype(int)
            )
            radius = int(round(grain.radius_px + grain_protection_px))
            cv.circle(protected_rims, center, max(1, radius), 1, -1)
        protected_rims = protected_rims.astype(bool)
        preexisting[protected_rims] = undilated_preexisting[protected_rims]

    total = len(evidence)
    birth = np.full(evidence.shape[1:], total, dtype=np.int32)
    rolling = np.zeros(evidence.shape[1:], dtype=np.uint8)
    for sample in range(total):
        rolling += active[sample]
        if sample >= window:
            rolling -= active[sample - window]
        if sample < warmup:
            continue
        newly_confirmed = (
            (rolling >= required) & (birth == total) & ~preexisting
        )
        birth[newly_confirmed] = sample
    birth[preexisting] = 0
    calibration = {
        "global_evidence_floor": round(global_floor, 3),
        "median_change_floor": round(float(np.median(change_floor)), 3),
        "preexisting_min_observations": preexisting_min_observations,
        "preexisting_dilation_px": preexisting_dilation_px,
        "preexisting_grain_protection_px": grain_protection_px,
        "preexisting_fraction": round(float(preexisting.mean()), 5),
        "newborn_fraction": round(float(((birth > 0) & (birth < total)).mean()), 5),
    }
    return birth, preexisting, calibration


def birth_topology_labels(
    birth: np.ndarray,
    total_samples: int,
    warmup: int,
    spatial_link_px: int,
    temporal_link_samples: int,
) -> tuple[np.ndarray, int]:
    """Connect nearby pixels only when their appearance times are compatible."""

    valid = (birth > warmup) & (birth < total_samples)
    node_map = np.full(birth.shape, -1, dtype=np.int32)
    node_map[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    node_count = int(valid.sum())
    labels_image = np.full(birth.shape, -1, dtype=np.int32)
    if node_count == 0:
        return labels_image, 0

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    h, w = birth.shape
    offsets = []
    for dy in range(0, spatial_link_px + 1):
        for dx in range(-spatial_link_px, spatial_link_px + 1):
            if dy == 0 and dx <= 0:
                continue
            if dx * dx + dy * dy > spatial_link_px * spatial_link_px:
                continue
            offsets.append((dy, dx))

    for dy, dx in offsets:
        y0a, y1a = 0, h - dy
        y0b, y1b = dy, h
        if dx >= 0:
            x0a, x1a, x0b, x1b = 0, w - dx, dx, w
        else:
            x0a, x1a, x0b, x1b = -dx, w, 0, w + dx
        left = node_map[y0a:y1a, x0a:x1a]
        right = node_map[y0b:y1b, x0b:x1b]
        compatible = (left >= 0) & (right >= 0)
        ba = birth[y0a:y1a, x0a:x1a]
        bb = birth[y0b:y1b, x0b:x1b]
        compatible &= np.abs(ba - bb) <= temporal_link_samples
        if compatible.any():
            rows.append(left[compatible])
            cols.append(right[compatible])

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        graph = coo_matrix(
            (np.ones(len(row) * 2, dtype=np.uint8),
             (np.concatenate([row, col]), np.concatenate([col, row]))),
            shape=(node_count, node_count),
        )
        component_count, node_labels = connected_components(graph, directed=False)
    else:
        component_count = node_count
        node_labels = np.arange(node_count, dtype=np.int32)
    labels_image[valid] = node_labels
    return labels_image, int(component_count)


def _nearest_births(mask: np.ndarray, birth: np.ndarray) -> np.ndarray:
    """Fill small closed-mask gaps with the nearest observed birth time."""

    _, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    return birth[nearest[0], nearest[1]]


def _skeleton_neighbor_indices(
    y: int,
    x: int,
    index: dict[tuple[int, int], int],
) -> list[int]:
    """Return graph neighbors without diagonal shortcuts around a corner."""

    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = index.get((y + dy, x + dx))
            if neighbor is None:
                continue
            if dy != 0 and dx != 0 and (
                (y, x + dx) in index or (y + dy, x) in index
            ):
                continue
            neighbors.append(neighbor)
    return neighbors


def _maximum_junction_turn_degrees(
    path_yx: np.ndarray,
    arclength: np.ndarray,
    skeleton: np.ndarray,
    direction_window_px: float,
) -> float:
    """Measure the sharpest direction change at a true skeleton junction."""

    if len(path_yx) < 3:
        return 0.0
    points = [tuple(int(value) for value in point) for point in np.argwhere(skeleton)]
    index = {point: node for node, point in enumerate(points)}
    junction_indices = np.asarray(
        [
            path_index
            for path_index, (y, x) in enumerate(path_yx)
            if len(_skeleton_neighbor_indices(int(y), int(x), index)) >= 3
        ],
        dtype=np.int32,
    )
    if not len(junction_indices):
        return 0.0

    group_starts = np.r_[0, np.flatnonzero(np.diff(junction_indices) > 1) + 1]
    group_ends = np.r_[group_starts[1:] - 1, len(junction_indices) - 1]
    maximum_turn = 0.0
    path = path_yx.astype(np.float64)
    distances = np.asarray(arclength, dtype=np.float64)
    for group_start, group_end in zip(group_starts, group_ends):
        start = int(junction_indices[group_start])
        end = int(junction_indices[group_end])
        if start == 0 or end >= len(path) - 1:
            continue
        before = int(
            np.searchsorted(
                distances,
                distances[start] - direction_window_px,
                side="right",
            )
            - 1
        )
        after = int(
            np.searchsorted(
                distances,
                distances[end] + direction_window_px,
                side="left",
            )
        )
        before = min(start - 1, max(0, before))
        after = max(end + 1, min(len(path) - 1, after))
        incoming = path[start] - path[before]
        outgoing = path[after] - path[end]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= 1e-9:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
        maximum_turn = max(maximum_turn, float(np.degrees(np.arccos(cosine))))
    return maximum_turn


def _skeleton_path_candidates(
    skeleton: np.ndarray,
    root_yx: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shortest root paths to every open endpoint of a skeleton."""

    ys, xs = np.nonzero(skeleton)
    points = list(zip(ys.tolist(), xs.tolist()))
    index = {point: i for i, point in enumerate(points)}
    root = index[root_yx]
    distance = np.full(len(points), np.inf)
    predecessor = np.full(len(points), -1, dtype=np.int32)
    distance[root] = 0.0
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        y, x = points[node]
        for neighbor in _skeleton_neighbor_indices(y, x, index):
            neighbor_y, neighbor_x = points[neighbor]
            proposed = current_distance + float(
                np.hypot(neighbor_x - x, neighbor_y - y)
            )
            if proposed < distance[neighbor]:
                distance[neighbor] = proposed
                predecessor[neighbor] = node
                heapq.heappush(queue, (proposed, neighbor))

    endpoint_nodes = []
    for node, (y, x) in enumerate(points):
        if node == root or not np.isfinite(distance[node]):
            continue
        degree = len(_skeleton_neighbor_indices(y, x, index))
        if degree <= 1:
            endpoint_nodes.append(node)
    if not endpoint_nodes:
        endpoint_nodes = [
            int(np.nanargmax(np.where(np.isfinite(distance), distance, -1.0)))
        ]

    candidates = []
    for tip in endpoint_nodes:
        chain = []
        node = tip
        while node >= 0:
            chain.append(node)
            if node == root:
                break
            node = int(predecessor[node])
        if not chain or chain[-1] != root:
            continue
        chain.reverse()
        indices = np.asarray(chain, dtype=np.int32)
        path_yx = np.asarray([points[node] for node in chain], dtype=np.int32)
        candidates.append((path_yx, distance[indices]))
    return candidates


def _birth_order_metrics(
    raw_birth: np.ndarray,
    arclength: np.ndarray,
    tolerance_samples: float = 1.0,
) -> tuple[float, float, float]:
    """Measure whether construction progresses outward along a candidate path."""

    raw_birth = np.asarray(raw_birth, dtype=np.float64)
    arclength = np.asarray(arclength, dtype=np.float64)
    if len(raw_birth) < 3:
        return 0.0, 0.0, 0.0
    window = min(9, len(raw_birth) if len(raw_birth) % 2 else len(raw_birth) - 1)
    smoothed = ndimage.median_filter(raw_birth, size=max(1, window), mode="nearest")
    if np.ptp(smoothed) > 0 and np.ptp(arclength) > 0:
        correlation = float(np.corrcoef(rankdata(arclength), rankdata(smoothed))[0, 1])
    else:
        correlation = 0.0
    lag = int(np.searchsorted(arclength, arclength[0] + 8.0, side="left"))
    lag = min(max(1, lag), len(smoothed) - 1)
    forward = float(
        np.mean(smoothed[lag:] >= smoothed[:-lag] - tolerance_samples)
    )
    edge_count = max(1, len(smoothed) // 5)
    progress = float(
        np.median(smoothed[-edge_count:]) - np.median(smoothed[:edge_count])
    )
    return correlation, forward, progress


def _select_birth_ordered_path(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    birth_image: np.ndarray,
    config: AtlasConfig,
    skeleton: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Choose a branch using construction order and junction continuation."""

    ranked = []
    for path_yx, arclength in candidates:
        raw_birth = birth_image[path_yx[:, 0], path_yx[:, 1]].astype(np.int32)
        correlation, forward, progress = _birth_order_metrics(raw_birth, arclength)
        duration = int(raw_birth.max() - raw_birth.min())
        if duration >= config.trajectory_resolution_samples:
            direction_quality = max(0.0, correlation) * forward
            floor = config.branch_direction_floor_weight
            selection_factor = floor + (1.0 - floor) * direction_quality
        else:
            # An atlas-only event can establish a final path, but not its
            # framewise direction. Keep geometry influential without letting
            # a long simultaneous branch automatically dominate.
            selection_factor = config.atlas_branch_weight
        junction_turn = (
            0.0
            if skeleton is None
            else _maximum_junction_turn_degrees(
                path_yx,
                arclength,
                skeleton,
                config.junction_direction_window_px,
            )
        )
        excess_turn = max(
            0.0,
            junction_turn - config.junction_turn_soft_limit_degrees,
        )
        turn_range = max(
            1.0,
            180.0 - config.junction_turn_soft_limit_degrees,
        )
        continuation_factor = 1.0 / (
            1.0
            + config.junction_turn_penalty_weight * excess_turn / turn_range
        )
        selection_score = (
            float(arclength[-1]) * selection_factor * continuation_factor
        )
        ranked.append(
            (
                selection_score,
                float(arclength[-1]),
                path_yx,
                arclength,
                raw_birth,
                correlation,
                forward,
                progress,
            )
        )
    if not ranked:
        raise ValueError("skeleton contains no root-to-endpoint path")
    _, _, path_yx, arclength, raw_birth, correlation, forward, progress = max(
        ranked, key=lambda item: (item[0], item[1])
    )
    return path_yx, arclength, raw_birth, correlation, forward, progress


def _trim_grain_rim_prefix(
    path_yx: np.ndarray,
    grain_center_xy: np.ndarray,
    grain_radius_px: float,
    margin_px: float = 2.0,
    outside_run: int = 6,
) -> np.ndarray:
    """Start tube length at the first stable exit from the pollen rim."""

    if len(path_yx) <= outside_run:
        return path_yx
    path_xy = path_yx[:, ::-1].astype(np.float64)
    distance = np.linalg.norm(path_xy - grain_center_xy, axis=1)
    outside = distance > grain_radius_px + margin_px
    for index in range(1, len(path_yx) - outside_run + 1):
        if np.all(outside[index : index + outside_run]):
            return path_yx[max(0, index - 1) :]
    return path_yx


def _prepend_grain_rim_connector(
    path_yx: np.ndarray,
    grain_center_xy: np.ndarray,
    grain_radius_px: float,
    margin_px: float,
) -> tuple[np.ndarray, int]:
    """Bridge a short excluded-mask gap from the pollen rim to new material."""

    if not len(path_yx):
        return path_yx, 0
    root_xy = path_yx[0, ::-1].astype(np.float64)
    offset = root_xy - grain_center_xy
    distance = float(np.linalg.norm(offset))
    target_distance = grain_radius_px + margin_px
    gap = distance - target_distance
    if distance <= 0 or gap <= 1.0:
        return path_yx, 0
    target_xy = grain_center_xy + offset / distance * target_distance
    connector_xy = np.rint(
        np.linspace(target_xy, root_xy, max(2, int(np.ceil(gap)) + 1))
    ).astype(np.int32)
    keep = np.r_[True, np.any(np.diff(connector_xy, axis=0) != 0, axis=1)]
    connector_yx = connector_xy[keep, ::-1]
    if len(connector_yx) <= 1:
        return path_yx, 0
    prefix = connector_yx[:-1]
    return np.vstack([prefix, path_yx]), len(prefix)


def _persistent_change_metrics(
    signal: np.ndarray,
    config: AtlasConfig,
    minimum_delta: float,
) -> tuple[float, float, int, bool, int | None]:
    """Find the first locally persistent increase over a warmup baseline."""

    values = np.asarray(signal, dtype=np.float64)
    baseline_stop = min(config.rim_history_samples, len(values))
    if baseline_stop < config.rim_persistence_required:
        return 0.0, 0.0, 0, False, None
    before = values[:baseline_stop]
    baseline = float(np.median(before))
    noise = float(1.4826 * np.median(np.abs(before - baseline)))
    threshold = max(minimum_delta, config.rim_noise_multiplier * noise)
    active = values >= baseline + threshold
    confirmation_sample = None
    persistent = 0
    for timepoint in range(baseline_stop, len(values)):
        window_start = max(
            baseline_stop,
            timepoint - config.rim_persistence_window + 1,
        )
        persistent = int(np.count_nonzero(active[window_start : timepoint + 1]))
        if persistent >= config.rim_persistence_required:
            confirmation_sample = timepoint
            break
    if confirmation_sample is None:
        remaining = values[baseline_stop:]
        delta = float(np.max(remaining) - baseline) if len(remaining) else 0.0
        return delta, noise, persistent, False, None
    window_start = max(
        baseline_stop,
        confirmation_sample - config.rim_persistence_window + 1,
    )
    delta = float(
        np.median(values[window_start : confirmation_sample + 1]) - baseline
    )
    return delta, noise, persistent, True, confirmation_sample


def _rim_emergence_metrics(
    evidence: np.ndarray,
    grain: GrainAnchor,
    path_xy: np.ndarray,
    sample: int,
    config: AtlasConfig,
) -> tuple[float, float, int, bool, int | None]:
    """Confirm a persistent new filament in the claimed pollen-rim sector."""

    if len(path_xy) < 2:
        return 0.0, 0.0, 0, False, None
    arclength = np.concatenate(
        [
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)),
        ]
    )
    segment = path_xy[arclength <= config.rim_path_length_px]
    if len(segment) < 2:
        segment = path_xy[: min(2, len(path_xy))]
    reference_center = grain.center_at(sample)
    offsets = segment - reference_center
    signal = []
    for timepoint in range(len(evidence)):
        moving_center = grain.center_at(timepoint)
        points = moving_center + offsets
        x = np.rint(points[:, 0]).astype(int)
        y = np.rint(points[:, 1]).astype(int)
        x = np.clip(x[:, None] + np.asarray([-1, 0, 1]), 0, evidence.shape[2] - 1)
        y = np.clip(y[:, None] + np.asarray([-1, 0, 1]), 0, evidence.shape[1] - 1)
        x = np.broadcast_to(x[:, None, :], (len(points), 3, 3))
        y = np.broadcast_to(y[:, :, None], (len(points), 3, 3))
        signal.append(
            float(np.percentile(evidence[timepoint, y, x], config.rim_signal_percentile))
        )

    return _persistent_change_metrics(
        np.asarray(signal, dtype=np.float64),
        config,
        config.rim_min_delta,
    )


def _oriented_rim_contrast_metrics(
    evidence: np.ndarray,
    grain: GrainAnchor,
    path_xy: np.ndarray,
    sample: int,
    config: AtlasConfig,
) -> tuple[float, float, int, bool, int | None]:
    """Detect a new proximal filament relative to its two local flanks."""

    if len(path_xy) < 2:
        return 0.0, 0.0, 0, False, None
    arclength = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
    )
    segment = path_xy[arclength <= config.rim_path_length_px]
    if len(segment) < 2:
        segment = path_xy[: min(2, len(path_xy))]
    tangents = np.gradient(segment.astype(np.float64), axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    reference_center = grain.center_at(sample)
    local_path = segment - reference_center
    centers = np.asarray(
        [grain.center_at(timepoint) for timepoint in range(len(evidence))],
        dtype=np.float64,
    )

    halfwidth = config.rim_contrast_halfwidth_px
    flank = config.rim_contrast_flank_offset_px
    on_offsets = np.asarray([-halfwidth, 0.0, halfwidth])
    flank_offsets = np.asarray(
        [-flank - halfwidth, -flank, flank, flank + halfwidth]
    )

    def sample_offsets(offsets: np.ndarray) -> np.ndarray:
        points = (
            centers[:, None, None, :]
            + local_path[None, :, None, :]
            + normals[None, :, None, :] * offsets[None, None, :, None]
        )
        x = np.clip(
            np.rint(points[..., 0]).astype(np.int32),
            0,
            evidence.shape[2] - 1,
        )
        y = np.clip(
            np.rint(points[..., 1]).astype(np.int32),
            0,
            evidence.shape[1] - 1,
        )
        time = np.arange(len(evidence), dtype=np.int32)[:, None, None]
        return evidence[time, y, x]

    on_path = sample_offsets(on_offsets)
    flanks = sample_offsets(flank_offsets)
    signal = np.percentile(on_path, config.rim_signal_percentile, axis=(1, 2))
    signal -= np.median(flanks, axis=(1, 2))
    return _persistent_change_metrics(
        signal,
        config,
        config.rim_contrast_min_delta,
    )


def _quality_classification(
    path_xy: np.ndarray,
    length_px: float,
    duration_samples: int,
    root_distance_px: float,
    width_px: float,
    tortuosity: float,
    birth_correlation: float,
    birth_forward: float,
    birth_progress: float,
    growth_step_count: int,
    tip_step_median_px: float,
    tip_step_max_px: float,
    radial_extension: float | None,
    foreign_clearance: float | None,
    image_shape: tuple[int, int],
    config: AtlasConfig,
    grain_attachment: float | None = None,
    rim_emergence_confirmed: bool | None = None,
    rim_emergence_timing_consistent: bool | None = None,
    grain_center_xy: np.ndarray | None = None,
    grain_radius_px: float | None = None,
) -> tuple[str, tuple[str, ...], bool, bool]:
    """Classify an event without deleting uncertain biological observations."""

    flags = []
    trajectory_resolved = duration_samples >= config.trajectory_resolution_samples
    if length_px < config.min_path_length_px:
        flags.append("short-after-rim-exit")
    if duration_samples < 2:
        flags.append("simultaneous-atlas-event")
    if root_distance_px > config.max_root_distance_px:
        flags.append("distant-from-preexisting-material")
    if (
        grain_attachment is not None
        and grain_attachment > config.max_grain_attachment_px
    ):
        flags.append("detached-from-grain")
    if rim_emergence_confirmed is False:
        flags.append("no-rim-emergence")
    if rim_emergence_timing_consistent is False:
        flags.append("rim-emergence-timing-mismatch")
    if tortuosity > config.max_path_tortuosity:
        flags.append("high-tortuosity")
    birth_order_testable = growth_step_count >= 2
    if birth_order_testable and (
        birth_correlation < config.min_birth_order_correlation
        or birth_forward < config.min_birth_order_forward_fraction
        or birth_progress <= 0
    ):
        flags.append("inconsistent-birth-order")
    if not birth_order_testable:
        flags.append("insufficient-birth-order-evidence")
    if trajectory_resolved and growth_step_count < config.min_growth_step_count:
        flags.append("sparse-growth-observations")
    if trajectory_resolved and tip_step_median_px > config.max_tip_step_median_px:
        flags.append("incoherent-tip-steps")
    if trajectory_resolved and tip_step_max_px > config.max_tip_step_px:
        flags.append("abrupt-tip-jump")
    if radial_extension is not None:
        minimum_radial_extension = max(
            config.min_radial_extension_px,
            config.min_radial_extension_ratio * length_px,
        )
        if radial_extension < minimum_radial_extension:
            flags.append("weak-radial-extension")
    if width_px > config.max_event_width_px:
        flags.append("broad-change")
    h, w = image_shape
    if np.any(
        (path_xy[:, 0] <= 1)
        | (path_xy[:, 0] >= w - 2)
        | (path_xy[:, 1] <= 1)
        | (path_xy[:, 1] >= h - 2)
    ):
        flags.append("partial-field")
    if grain_center_xy is not None and grain_radius_px is not None:
        center = np.asarray(grain_center_xy, dtype=np.float64)
        margin = float(grain_radius_px) + 1.0
        if (
            center[0] < margin
            or center[0] > w - 1 - margin
            or center[1] < margin
            or center[1] > h - 1 - margin
        ):
            flags.append("partial-grain")
    if (
        foreign_clearance is not None
        and foreign_clearance <= config.foreign_grain_clearance_margin_px
    ):
        flags.append("foreign-grain-contact")
        if not trajectory_resolved:
            flags.append("ambiguous-grain-ownership")

    germination_blocking = {
        "inconsistent-birth-order",
        "insufficient-birth-order-evidence",
        "simultaneous-atlas-event",
        "weak-radial-extension",
        "broad-change",
        "distant-from-preexisting-material",
        "detached-from-grain",
        "no-rim-emergence",
        "rim-emergence-timing-mismatch",
        "high-tortuosity",
        "ambiguous-grain-ownership",
        "partial-field",
        "partial-grain",
    }
    trajectory_blocking = germination_blocking | {
        "partial-field",
        "short-after-rim-exit",
        "sparse-growth-observations",
        "incoherent-tip-steps",
        "abrupt-tip-jump",
        "foreign-grain-contact",
    }
    germination_accepted = not any(
        flag in germination_blocking for flag in flags
    )
    trajectory_accepted = trajectory_resolved and not any(
        flag in trajectory_blocking for flag in flags
    )
    if not germination_accepted:
        status = "review-required"
    elif "partial-field" in flags:
        status = "partial-field-event"
    elif "short-after-rim-exit" in flags:
        status = "emergence-only"
    elif not trajectory_resolved:
        status = "atlas-event"
    elif trajectory_accepted:
        status = "trajectory-growth"
    else:
        status = "germination-only"
    return status, tuple(flags), germination_accepted, trajectory_accepted


def _component_width_px(area_px: int, skeleton: np.ndarray) -> float:
    """Estimate component width without charging every branch to one path."""

    return float(area_px) / max(1, int(np.count_nonzero(skeleton)))


def _promote_resolved_short_trajectories(events: list[BirthEvent]) -> None:
    """Promote primary short paths only after ownership and onset are fixed."""

    for event in events:
        if (
            not event.primary_for_grain
            or not event.auto_accepted
            or event.trajectory_accepted
            or set(event.quality_flags) != {"short-after-rim-exit"}
        ):
            continue
        event.trajectory_accepted = True
        event.quality_status = "trajectory-growth"


def _tip_dynamics_metrics(
    path_xy: np.ndarray,
    path_birth: np.ndarray,
    arclength: np.ndarray,
) -> tuple[int, float, float, float]:
    """Summarize observed tip updates along a monotone construction path."""

    if len(path_xy) < 2:
        return 0, 0.0, 0.0, 0.0
    group_ends = np.flatnonzero(
        np.r_[path_birth[1:] != path_birth[:-1], True]
    )
    if len(group_ends) < 2:
        return 0, 0.0, 0.0, 0.0
    tip_steps = np.linalg.norm(np.diff(path_xy[group_ends], axis=0), axis=1)
    length_steps = np.diff(arclength[group_ends])
    nonzero = tip_steps[tip_steps > 1e-6]
    return (
        int(len(nonzero)),
        float(np.median(nonzero)) if len(nonzero) else 0.0,
        float(np.max(nonzero)) if len(nonzero) else 0.0,
        float(np.max(length_steps)) if len(length_steps) else 0.0,
    )


def _fit_event_temporal_front(
    event: BirthEvent,
    evidence: np.ndarray,
    config: AtlasConfig,
    evidence_floor: float,
    maximum_arclength_px: float | None = None,
    path_offsets_xy: np.ndarray | None = None,
    anchor_measurement_start: bool = True,
) -> PathLockedFrontResult:
    """Fit one immutable event path, optionally only near the pollen rim."""

    stop = len(event.path_xy)
    if maximum_arclength_px is not None:
        stop = max(
            2,
            int(
                np.searchsorted(
                    event.arclength_px,
                    maximum_arclength_px,
                    side="right",
                )
            ),
        )
        stop = min(stop, len(event.path_xy))
    path_xy = event.path_xy[:stop]
    arclength = event.arclength_px[:stop]
    observed_birth = event.path_birth[:stop]
    germination_index = None
    onset_sample = None
    if anchor_measurement_start:
        germination_index = min(
            int(
                np.searchsorted(
                    arclength,
                    config.min_germination_length_px,
                    side="left",
                )
            ),
            len(observed_birth) - 1,
        )
        onset_sample = event.germination_sample
    return path_locked_temporal_front(
        evidence,
        path_xy,
        arclength,
        observed_birth,
        warmup_samples=config.warmup_samples,
        max_step_px=config.max_tip_step_px,
        max_confirmation_lag_samples=config.temporal_link_samples,
        absolute_evidence_floor=evidence_floor,
        normal_halfwidth_px=config.front_normal_halfwidth_px,
        normal_sample_count=config.front_normal_sample_count,
        minimum_change=config.rim_min_delta,
        noise_multiplier=config.rim_noise_multiplier,
        temporal_median_samples=config.front_temporal_median_samples,
        absence_weight=config.front_absence_weight,
        motion_penalty=config.front_motion_penalty,
        onset_point_index=germination_index,
        onset_sample=onset_sample,
        path_offsets_xy=path_offsets_xy,
    )


def _store_path_locked_front(
    event: BirthEvent,
    front: PathLockedFrontResult,
) -> None:
    """Retain one fitted front and its pointwise evidence on an event."""

    event.front_eventual_support_fraction = front.eventual_support_fraction
    event.front_direct_support_fraction = front.direct_support_fraction
    event.front_inferred_fraction = front.inferred_fraction
    event.front_median_confirmation_lag_samples = (
        front.median_confirmation_lag_samples
    )
    event.front_max_confirmation_lag_samples = front.max_confirmation_lag_samples
    if not front.feasible:
        return
    event.path_front_birth = front.birth_samples.copy()
    event.path_front_direct_support = front.direct_support_mask.copy()
    event.path_front_eventual_support = front.eventual_support_mask.copy()


def _select_path_locked_front(
    event: BirthEvent,
    front: PathLockedFrontResult,
    reason: str,
    dynamics: tuple[int, float, float, float] | None = None,
) -> None:
    """Use a supported temporal front as the reported tip timeline."""

    if dynamics is None:
        dynamics = _tip_dynamics_metrics(
            event.path_xy,
            front.birth_samples,
            event.arclength_px,
        )
    (
        event.growth_step_count,
        event.tip_step_median_px,
        event.tip_step_max_px,
        event.length_step_max_px,
    ) = dynamics
    event.front_refinement_applied = True
    event.front_refinement_reason = reason


def _front_reported_tips_are_supported(
    event: BirthEvent,
    front: PathLockedFrontResult,
) -> bool:
    """Reject a replacement timeline that would report an inferred early tip."""

    timing = np.asarray(front.birth_samples, dtype=np.int32)
    direct = np.asarray(front.direct_support_mask, dtype=bool)
    eventual = np.asarray(front.eventual_support_mask, dtype=bool)
    confirmation = np.asarray(event.path_birth, dtype=np.int32)
    if (
        timing.shape != direct.shape
        or timing.shape != eventual.shape
        or timing.ndim != 1
        or len(timing) == 0
        or len(timing) > len(confirmation)
    ):
        return False
    confirmation = confirmation[: len(timing)]
    start = max(event.germination_sample, int(np.min(timing)))
    stop = int(np.max(confirmation))
    for sample in range(start, stop + 1):
        indices = np.flatnonzero(timing <= sample)
        if not len(indices):
            continue
        path_index = int(indices[-1])
        if sample >= confirmation[path_index]:
            continue
        if not (direct[path_index] or eventual[path_index]):
            return False
    return True


def _apply_proximal_emergence_review(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
) -> None:
    """Withhold germinations whose first tube segment lacks a causal front."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    grains_by_id = {
        grain.grain_id: grain for grain in (grain_anchors or [])
    }

    def support_score(candidate: PathLockedFrontResult) -> float:
        if not candidate.feasible:
            return 0.0
        return min(
            candidate.direct_support_fraction,
            candidate.eventual_support_fraction,
        )

    for event in events:
        if not event.primary_for_grain:
            event.proximal_front_reason = "not-primary-event"
            continue
        if not event.auto_accepted:
            event.proximal_front_reason = "germination-not-accepted"
            continue
        path_offsets = None
        grain = grains_by_id.get(getattr(event, "grain_id", None))
        if grain is not None:
            reference_center = grain.center_at(event.germination_sample)
            path_offsets = np.asarray(
                [
                    grain.center_at(sample) - reference_center
                    for sample in range(len(evidence))
                ],
                dtype=np.float64,
            )
            event.proximal_reference_motion_max_px = float(
                np.max(np.linalg.norm(path_offsets, axis=1))
            )
        static_front = _fit_event_temporal_front(
            event,
            evidence,
            config,
            evidence_floor,
            maximum_arclength_px=config.proximal_front_length_px,
        )
        front = (
            static_front
            if path_offsets is None
            else _fit_event_temporal_front(
                event,
                evidence,
                config,
                evidence_floor,
                maximum_arclength_px=config.proximal_front_length_px,
                path_offsets_xy=path_offsets,
            )
        )
        event.proximal_static_front_reason = static_front.reason
        event.proximal_static_eventual_support_fraction = (
            static_front.eventual_support_fraction
        )
        event.proximal_static_direct_support_fraction = (
            static_front.direct_support_fraction
        )

        static_supported = (
            static_front.feasible
            and static_front.eventual_support_fraction
            >= config.front_min_eventual_support_fraction
            and static_front.direct_support_fraction
            >= config.front_min_direct_support_fraction
        )
        moving_supported = (
            front.feasible
            and front.eventual_support_fraction
            >= config.front_min_eventual_support_fraction
            and front.direct_support_fraction
            >= config.front_min_direct_support_fraction
        )
        event.proximal_motion_support_margin = (
            support_score(front) - support_score(static_front)
        )
        if path_offsets is None:
            event.proximal_motion_binding_class = "static-reference-only"
        elif static_supported and moving_supported:
            event.proximal_motion_binding_class = "both-supported"
        elif moving_supported:
            event.proximal_motion_binding_class = "pollen-following-only"
        elif static_supported:
            event.proximal_motion_binding_class = "static-only"
        else:
            event.proximal_motion_binding_class = "neither-supported"
        event.proximal_front_reason = front.reason
        event.proximal_eventual_support_fraction = (
            front.eventual_support_fraction
        )
        event.proximal_direct_support_fraction = front.direct_support_fraction
        event.proximal_inferred_fraction = front.inferred_fraction
        event.proximal_median_confirmation_lag_samples = (
            front.median_confirmation_lag_samples
        )
        event.proximal_max_confirmation_lag_samples = (
            front.max_confirmation_lag_samples
        )
        if event.proximal_motion_binding_class == "static-only":
            event.proximal_front_reason = "static-only-proximal-structure"
        elif front.feasible and (
            front.eventual_support_fraction
            < config.front_min_eventual_support_fraction
        ):
            event.proximal_front_reason = (
                "insufficient-proximal-eventual-support"
            )
        elif front.feasible and (
            front.direct_support_fraction
            < config.front_min_direct_support_fraction
        ):
            event.proximal_front_reason = "insufficient-proximal-direct-support"
        elif front.feasible:
            event.proximal_front_verified = True
            event.proximal_front_reason = "verified"
            continue

        event.proximal_emergence_ambiguous = True
        event.auto_accepted = False
        event.quality_status = (
            "proximal-ambiguous-trajectory"
            if event.trajectory_accepted
            else "review-required"
        )
        event.quality_flags = tuple(
            dict.fromkeys(
                (*event.quality_flags, "unresolved-proximal-emergence")
            )
        )


def _connected_proximal_emergence_sample(
    evidence: np.ndarray,
    path_xy: np.ndarray,
    path_offsets_xy: np.ndarray,
    config: AtlasConfig,
    evidence_floor: float,
) -> int | None:
    """Find a persistent new evidence chain spanning one proximal path."""

    path = np.asarray(path_xy, dtype=np.float64)
    if len(path) < 2:
        return None
    arclength = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )
    stop = max(
        2,
        int(
            np.searchsorted(
                arclength,
                config.proximal_front_length_px,
                side="right",
            )
        ),
    )
    path = path[: min(stop, len(path))]
    profiles = path_novelty_profiles(
        evidence,
        path,
        config.warmup_samples,
        evidence_floor,
        config.rim_direction_normal_halfwidth_px,
        3,
        config.rim_min_delta,
        config.rim_noise_multiplier,
        config.front_temporal_median_samples,
        path_offsets_xy,
    )
    active = profiles >= 0.0
    bridged = active.copy()
    if active.shape[1] >= 3:
        bridged[:, 1:-1] |= active[:, :-2] & active[:, 2:]
    no_wide_gap = np.ones(len(active), dtype=bool)
    if active.shape[1] >= 2:
        no_wide_gap = ~np.any(
            (~active[:, :-1]) & (~active[:, 1:]),
            axis=1,
        )
    connected = (
        active[:, 0]
        & active[:, -1]
        & no_wide_gap
        & (
            np.mean(bridged, axis=1)
            >= config.rim_direction_min_connected_fraction
        )
    )
    for sample in range(config.warmup_samples, len(connected)):
        start = max(
            config.warmup_samples,
            sample - config.rim_persistence_window + 1,
        )
        if (
            np.count_nonzero(connected[start : sample + 1])
            >= config.rim_persistence_required
        ):
            return sample
    return None


def _apply_rim_direction_specificity_review(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
) -> None:
    """Withhold short rim changes repeated in independent directions."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    grains_by_id = {
        grain.grain_id: grain for grain in (grain_anchors or [])
    }
    minimum_rotation = config.rim_direction_min_rotation_degrees
    rotation_step = config.rim_direction_rotation_step_degrees
    if rotation_step <= 0 or not 0 < minimum_rotation < 180:
        raise ValueError("invalid rim-direction rotation configuration")

    for event in events:
        if not event.primary_for_grain:
            event.rim_direction_reason = "not-primary-event"
            continue
        if not event.auto_accepted:
            event.rim_direction_reason = "germination-not-accepted"
            continue
        grain = grains_by_id.get(event.grain_id)
        if grain is None or len(event.path_xy) < 2:
            event.rim_direction_reason = "grain-or-path-unavailable"
            continue

        reference_center = grain.center_at(event.germination_sample)
        path_offsets = np.asarray(
            [
                grain.center_at(sample) - reference_center
                for sample in range(len(evidence))
            ],
            dtype=np.float64,
        )
        event.rim_direction_selected_connected_sample = (
            _connected_proximal_emergence_sample(
                evidence,
                event.path_xy,
                path_offsets,
                config,
                evidence_floor,
            )
        )

        competitors: list[tuple[int, int]] = []
        margin = int(np.ceil(config.rim_direction_normal_halfwidth_px)) + 1
        height, width = evidence.shape[1:]
        prefix_stop = min(
            len(event.path_xy),
            max(
                2,
                int(
                    np.searchsorted(
                        event.arclength_px,
                        config.proximal_front_length_px,
                        side="right",
                    )
                ),
            ),
        )
        for rotation_degrees in range(
            minimum_rotation,
            360 - minimum_rotation + 1,
            rotation_step,
        ):
            theta = np.deg2rad(rotation_degrees)
            rotation = np.asarray(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ],
                dtype=np.float64,
            )
            rotated = (
                reference_center
                + (event.path_xy - reference_center) @ rotation.T
            )
            rotated_prefix = rotated[:prefix_stop]
            if (
                np.linalg.norm(
                    rotated_prefix[-1] - event.path_xy[prefix_stop - 1]
                )
                < config.nominal_tube_width_px
            ):
                continue
            if np.any(
                (rotated_prefix[:, 0] < margin)
                | (rotated_prefix[:, 0] >= width - margin)
                | (rotated_prefix[:, 1] < margin)
                | (rotated_prefix[:, 1] >= height - margin)
            ):
                continue
            sample = _connected_proximal_emergence_sample(
                evidence,
                rotated,
                path_offsets,
                config,
                evidence_floor,
            )
            if sample is None or (
                abs(sample - event.birth_start)
                > config.rim_direction_timing_tolerance_samples
            ):
                continue
            competitors.append((rotation_degrees, sample))

        event.rim_direction_specificity_evaluated = True
        event.rim_direction_competitor_count = len(competitors)
        event.rim_direction_specific = not competitors
        if not competitors:
            event.rim_direction_reason = "direction-specific"
            continue

        best_rotation, best_sample = min(
            competitors,
            key=lambda item: (abs(item[1] - event.birth_start), item[0]),
        )
        event.rim_direction_competitor_rotation_degrees = best_rotation
        event.rim_direction_competitor_sample = best_sample
        short_path = event.arclength_px[-1] < config.min_path_length_px
        if short_path and not event.trajectory_accepted:
            event.auto_accepted = False
            event.quality_status = "review-required"
            event.quality_flags = tuple(
                dict.fromkeys(
                    (*event.quality_flags, "non-specific-rim-change")
                )
            )
            event.rim_direction_reason = "ambiguous-short-rim-change"
        else:
            event.rim_direction_reason = "competitor-recorded-path-retained"


def _apply_path_locked_front_refinement(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Rescue only frozen primary paths that failed on threshold-tip jumps."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    removable_dynamic_flags = {
        "sparse-growth-observations",
        "incoherent-tip-steps",
        "abrupt-tip-jump",
    }
    for event in events:
        if not event.primary_for_grain:
            event.front_refinement_reason = "not-primary-event"
            continue
        if not event.auto_accepted:
            event.front_refinement_reason = "germination-not-accepted"
            continue
        if event.quality_status != "germination-only":
            event.front_refinement_reason = "no-dynamic-rescue-needed"
            continue
        if any(
            flag not in removable_dynamic_flags
            for flag in event.quality_flags
        ):
            event.front_refinement_reason = "non-dynamic-trajectory-blocker"
            continue

        front = _fit_event_temporal_front(
            event,
            evidence,
            config,
            evidence_floor,
        )
        event.front_refinement_reason = front.reason
        _store_path_locked_front(event, front)
        if not front.feasible:
            continue
        if (
            front.eventual_support_fraction
            < config.front_min_eventual_support_fraction
        ):
            event.front_refinement_reason = "insufficient-eventual-path-support"
            continue
        if (
            front.direct_support_fraction
            < config.front_min_direct_support_fraction
        ):
            event.front_refinement_reason = "insufficient-direct-tip-support"
            continue
        if not _front_reported_tips_are_supported(event, front):
            event.front_refinement_reason = (
                "refined-front-has-unsupported-tip-state"
            )
            continue

        growth_steps, median_step, maximum_step, length_step = (
            _tip_dynamics_metrics(
                event.path_xy,
                front.birth_samples,
                event.arclength_px,
            )
        )
        if (
            growth_steps < config.min_growth_step_count
            or median_step > config.max_tip_step_median_px
            or maximum_step > config.max_tip_step_px
        ):
            event.front_refinement_reason = "refined-front-failed-dynamics"
            continue

        _select_path_locked_front(
            event,
            front,
            front.reason,
            (growth_steps, median_step, maximum_step, length_step),
        )
        event.quality_flags = tuple(
            flag
            for flag in event.quality_flags
            if flag not in removable_dynamic_flags
        )
        event.trajectory_accepted = True
        event.quality_status = "trajectory-growth"


def _apply_full_path_temporal_validation(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Validate each selected path and adopt its front only for supported tips."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    for event in events:
        if not event.primary_for_grain or not event.trajectory_accepted:
            continue
        event.full_path_temporal_evaluated = True
        if event.front_refinement_applied:
            event.full_path_temporal_verified = True
            event.full_path_temporal_reason = "verified-by-applied-refinement"
            continue
        front = _fit_event_temporal_front(
            event,
            evidence,
            config,
            evidence_floor,
            anchor_measurement_start=False,
        )
        _store_path_locked_front(event, front)
        if not front.feasible:
            reason = "unresolved-full-path-temporal-front"
            event.full_path_temporal_reason = front.reason
        elif (
            front.eventual_support_fraction
            < config.front_min_eventual_support_fraction
        ):
            reason = "insufficient-full-path-eventual-support"
            event.full_path_temporal_reason = reason
        elif (
            front.direct_support_fraction
            < config.front_min_direct_support_fraction
        ):
            reason = "insufficient-full-path-direct-support"
            event.full_path_temporal_reason = reason
        else:
            event.full_path_temporal_verified = True
            event.full_path_temporal_reason = "verified"
            if _front_reported_tips_are_supported(event, front):
                _select_path_locked_front(
                    event,
                    front,
                    "verified-full-path-front",
                )
            else:
                event.front_refinement_reason = (
                    "verified-front-has-unsupported-tip-state"
                )
            continue

        event.trajectory_accepted = False
        event.quality_status = (
            "germination-only" if event.auto_accepted else "review-required"
        )
        event.quality_flags = tuple(
            dict.fromkeys((*event.quality_flags, reason))
        )


def _apply_rim_bridged_proximal_recovery(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
) -> None:
    """Recover tube existence when rim cues bridge unsupported root pixels."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    grains_by_id = {
        grain.grain_id: grain for grain in (grain_anchors or [])
    }
    required_flags = {"unresolved-proximal-emergence"}
    for event in events:
        if (
            not event.primary_for_grain
            or event.auto_accepted
            or set(event.quality_flags) != required_flags
        ):
            continue
        event.proximal_rim_bridge_evaluated = True
        if not (
            event.trajectory_accepted
            and event.full_path_temporal_verified
        ):
            event.proximal_rim_bridge_reason = "full-trajectory-not-verified"
            continue
        rim_sample = event.rim_emergence_sample
        contrast_sample = event.rim_contrast_emergence_sample
        if not (
            event.rim_emergence_confirmed
            and event.rim_contrast_confirmed
            and rim_sample is not None
            and contrast_sample is not None
        ):
            event.proximal_rim_bridge_reason = "independent-rim-cues-unavailable"
            continue
        if abs(rim_sample - contrast_sample) > config.temporal_link_samples:
            event.proximal_rim_bridge_reason = "independent-rim-cues-disagree"
            continue
        grain = grains_by_id.get(event.grain_id)
        if grain is None:
            event.proximal_rim_bridge_reason = "pollen-motion-unavailable"
            continue
        reference_center = grain.center_at(event.germination_sample)
        path_offsets = np.asarray(
            [
                grain.center_at(sample) - reference_center
                for sample in range(len(evidence))
            ],
            dtype=np.float64,
        )
        front = _fit_event_temporal_front(
            event,
            evidence,
            config,
            evidence_floor,
            maximum_arclength_px=config.proximal_front_length_px,
            path_offsets_xy=path_offsets,
            anchor_measurement_start=False,
        )
        if not front.feasible:
            event.proximal_rim_bridge_reason = front.reason
            continue
        point_count = len(front.birth_samples)
        arclength = event.arclength_px[:point_count]
        outside_rim_transition = arclength >= config.grain_exit_margin_px
        eligible_indices = np.flatnonzero(outside_rim_transition)
        if (
            len(eligible_indices) < 3
            or arclength[eligible_indices[-1]]
            - arclength[eligible_indices[0]]
            < config.min_germination_length_px
        ):
            event.proximal_rim_bridge_reason = "insufficient-distal-proximal-span"
            continue
        direct_fraction = float(
            np.mean(front.direct_support_mask[eligible_indices])
        )
        eventual_fraction = float(
            np.mean(front.eventual_support_mask[eligible_indices])
        )
        event.proximal_rim_bridge_direct_support_fraction = direct_fraction
        event.proximal_rim_bridge_eventual_support_fraction = eventual_fraction
        front_start = int(np.min(front.birth_samples[eligible_indices]))
        front_end = int(np.max(front.birth_samples[eligible_indices]))
        event.proximal_rim_bridge_start_sample = front_start
        event.proximal_rim_bridge_end_sample = front_end
        if (
            direct_fraction < config.front_min_direct_support_fraction
            or eventual_fraction < config.front_min_eventual_support_fraction
        ):
            event.proximal_rim_bridge_reason = "insufficient-distal-proximal-support"
            continue

        def cue_distance(sample: int) -> int:
            if sample < front_start:
                return front_start - sample
            if sample > front_end:
                return sample - front_end
            return 0

        if cue_distance(rim_sample) > config.rim_history_samples:
            event.proximal_rim_bridge_reason = "rim-cue-outside-front-window"
            continue
        if (
            contrast_sample < front_start - config.rim_history_samples
            or contrast_sample > front_end + config.temporal_link_samples
        ):
            event.proximal_rim_bridge_reason = "contrast-cue-outside-front-window"
            continue

        event.proximal_rim_bridge_accepted = True
        event.proximal_rim_bridge_reason = "verified"
        event.proximal_front_verified = True
        event.proximal_emergence_ambiguous = False
        event.proximal_front_reason = "verified-rim-bridged"
        event.auto_accepted = True
        event.quality_status = "trajectory-growth"
        event.quality_flags = tuple(
            flag
            for flag in event.quality_flags
            if flag != "unresolved-proximal-emergence"
        )


def _apply_precontact_trajectory_recovery(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Accept only the supported prefix before foreign-pollen contact."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    allowed_flags = {
        "foreign-grain-contact",
        "sparse-growth-observations",
        "incoherent-tip-steps",
        "abrupt-tip-jump",
    }
    for event in events:
        if (
            not event.primary_for_grain
            or not event.auto_accepted
            or event.trajectory_accepted
            or "foreign-grain-contact" not in event.quality_flags
        ):
            continue
        if any(flag not in allowed_flags for flag in event.quality_flags):
            event.precontact_front_reason = "non-contact-trajectory-blocker"
            continue
        stop = event.foreign_grain_contact_path_index
        if stop is None or stop <= 1 or stop >= len(event.path_xy):
            event.precontact_front_reason = "invalid-contact-censor-point"
            continue
        safe_length = float(event.arclength_px[stop - 1])
        if safe_length < config.min_germination_length_px:
            event.precontact_front_reason = "insufficient-precontact-length"
            continue

        front = _fit_event_temporal_front(
            event,
            evidence,
            config,
            evidence_floor,
            maximum_arclength_px=safe_length,
            anchor_measurement_start=False,
        )
        event.precontact_front_reason = front.reason
        event.precontact_direct_support_fraction = (
            front.direct_support_fraction
        )
        event.precontact_eventual_support_fraction = (
            front.eventual_support_fraction
        )
        if not front.feasible:
            continue
        if len(front.birth_samples) != stop:
            event.precontact_front_reason = "precontact-front-shape-mismatch"
            continue
        event.precontact_front_birth = front.birth_samples.copy()
        event.precontact_front_direct_support = (
            front.direct_support_mask.copy()
        )
        event.precontact_front_eventual_support = (
            front.eventual_support_mask.copy()
        )
        if (
            front.eventual_support_fraction
            < config.front_min_eventual_support_fraction
        ):
            event.precontact_front_reason = (
                "insufficient-precontact-eventual-support"
            )
            continue
        if (
            front.direct_support_fraction
            < config.front_min_direct_support_fraction
        ):
            event.precontact_front_reason = "insufficient-precontact-direct-support"
            continue
        if not _front_reported_tips_are_supported(event, front):
            event.precontact_front_reason = "unsupported-precontact-tip-state"
            continue

        first_unsafe_threshold = int(event.path_birth[stop])
        first_sample_after_safe_tip = int(front.birth_samples[-1]) + 1
        event.precontact_censor_sample = max(
            event.germination_sample + 1,
            min(first_unsafe_threshold, first_sample_after_safe_tip),
        )
        if (
            event.precontact_censor_sample - event.germination_sample
            < config.trajectory_resolution_samples
        ):
            event.precontact_front_reason = "insufficient-precontact-duration"
            continue

        dynamics = _tip_dynamics_metrics(
            event.path_xy[:stop],
            front.birth_samples,
            event.arclength_px[:stop],
        )
        (
            event.precontact_growth_step_count,
            event.precontact_tip_step_median_px,
            event.precontact_tip_step_max_px,
            event.precontact_length_step_max_px,
        ) = dynamics
        if (
            event.precontact_growth_step_count < config.min_growth_step_count
            or event.precontact_tip_step_median_px
            > config.max_tip_step_median_px
            or event.precontact_tip_step_max_px > config.max_tip_step_px
        ):
            event.precontact_front_reason = "precontact-front-failed-dynamics"
            continue

        event.precontact_trajectory_accepted = True
        event.precontact_front_reason = "verified-until-foreign-grain-contact"
        event.quality_status = "contact-censored-growth"


def _contact_bridge_competitor_is_viable(event: BirthEvent) -> bool:
    """Return whether another pollen root remains biologically plausible."""

    root_blocking_flags = {
        "simultaneous-atlas-event",
        "distant-from-preexisting-material",
        "detached-from-grain",
        "no-rim-emergence",
        "rim-emergence-timing-mismatch",
        "inconsistent-birth-order",
        "insufficient-birth-order-evidence",
        "weak-radial-extension",
        "high-tortuosity",
        "partial-field",
        "partial-grain",
    }
    return not root_blocking_flags.intersection(event.quality_flags)


def _apply_contact_bridge_recovery(
    events: list[BirthEvent],
    evidence: np.ndarray | None,
    config: AtlasConfig,
) -> None:
    """Reacquire a supported path prefix after one unambiguous grain contact."""

    if evidence is None or not len(evidence):
        return
    evidence_floor = _global_evidence_floor(evidence, config.warmup_samples)
    for event in events:
        if (
            not event.primary_for_grain
            or not event.auto_accepted
            or event.trajectory_accepted
            or "foreign-grain-contact" not in event.quality_flags
        ):
            continue

        entry = event.foreign_grain_contact_path_index
        exit_index = event.foreign_grain_contact_exit_path_index
        if (
            entry is None
            or exit_index is None
            or entry <= 1
            or exit_index <= entry
            or exit_index >= len(event.path_xy) - 1
        ):
            event.contact_bridge_reason = "contact-does-not-have-two-visible-sides"
            continue
        angles = (
            event.contact_bridge_ingress_chord_angle_degrees,
            event.contact_bridge_chord_egress_angle_degrees,
            event.contact_bridge_total_turn_degrees,
        )
        if any(angle is None for angle in angles):
            event.contact_bridge_reason = "insufficient-contact-tangent-span"
            continue
        if any(
            float(angle) > config.contact_bridge_max_turn_degrees
            for angle in angles
        ):
            event.contact_bridge_reason = "excessive-contact-turn"
            continue

        postcontact_path = event.path_xy[exit_index:]
        competitors = []
        for peer in events:
            if (
                peer is event
                or peer.component_id != event.component_id
                or peer.grain_id is None
                or peer.grain_id == event.grain_id
                or not _contact_bridge_competitor_is_viable(peer)
            ):
                continue
            overlap = _path_overlap_fraction(
                postcontact_path,
                peer.path_xy,
                config.competing_path_distance_px,
            )
            if overlap >= config.competing_path_overlap_fraction:
                competitors.append(peer)
        event.contact_bridge_viable_competitor_count = len(competitors)
        if competitors:
            event.contact_bridge_reason = "viable-competing-pollen-root"
            continue
        event.contact_bridge_identity_supported = True

        if not event.precontact_trajectory_accepted:
            event.contact_bridge_reason = "unverified-precontact-trajectory"
            continue
        minimum_length = (
            float(event.arclength_px[exit_index])
            + config.min_germination_length_px
        )
        minimum_endpoint = int(
            np.searchsorted(event.arclength_px, minimum_length, side="left")
        )
        endpoint_limit = len(event.path_xy)
        next_contact = event.contact_bridge_next_foreign_contact_path_index
        if next_contact is not None:
            endpoint_limit = min(endpoint_limit, next_contact)
        if minimum_endpoint >= endpoint_limit:
            event.contact_bridge_reason = "insufficient-postcontact-length"
            continue

        best: tuple[
            int,
            PathLockedFrontResult,
            tuple[int, float, float, float],
            int,
            float,
            float,
        ] | None = None
        for endpoint in range(minimum_endpoint, endpoint_limit):
            front = _fit_event_temporal_front(
                event,
                evidence,
                config,
                evidence_floor,
                maximum_arclength_px=float(event.arclength_px[endpoint]),
                anchor_measurement_start=False,
            )
            if (
                not front.feasible
                or len(front.birth_samples) != endpoint + 1
                or len(front.direct_support_mask) != endpoint + 1
                or len(front.eventual_support_mask) != endpoint + 1
            ):
                if best is not None:
                    break
                continue
            post = slice(exit_index, endpoint + 1)
            post_direct = float(np.mean(front.direct_support_mask[post]))
            post_eventual = float(np.mean(front.eventual_support_mask[post]))
            dynamics = _tip_dynamics_metrics(
                event.path_xy[: endpoint + 1],
                front.birth_samples,
                event.arclength_px[: endpoint + 1],
            )
            post_dynamics = _tip_dynamics_metrics(
                event.path_xy[post],
                front.birth_samples[post],
                event.arclength_px[post]
                - float(event.arclength_px[exit_index]),
            )
            post_duration = int(
                front.birth_samples[endpoint]
                - front.birth_samples[exit_index]
            )
            eligible = (
                front.direct_support_fraction
                >= config.front_min_direct_support_fraction
                and front.eventual_support_fraction
                >= config.front_min_eventual_support_fraction
                and post_direct >= config.front_min_direct_support_fraction
                and post_eventual >= config.front_min_eventual_support_fraction
                and bool(np.all(front.eventual_support_mask[post]))
                and _front_reported_tips_are_supported(event, front)
                and dynamics[0] >= config.min_growth_step_count
                and dynamics[1] <= config.max_tip_step_median_px
                and dynamics[2] <= config.max_tip_step_px
                and post_dynamics[0] >= config.min_growth_step_count
                and post_dynamics[1] <= config.max_tip_step_median_px
                and post_dynamics[2] <= config.max_tip_step_px
                and post_duration >= config.trajectory_resolution_samples
            )
            if not eligible:
                if best is not None:
                    break
                continue
            best = (
                endpoint,
                front,
                dynamics,
                post_dynamics[0],
                post_direct,
                post_eventual,
            )

        if best is None:
            event.contact_bridge_reason = "insufficient-postcontact-support"
            continue
        (
            endpoint,
            front,
            dynamics,
            post_growth_steps,
            post_direct,
            post_eventual,
        ) = best
        event.contact_bridge_trajectory_accepted = True
        event.contact_bridge_last_path_index = endpoint
        event.contact_bridge_front_birth = front.birth_samples.copy()
        event.contact_bridge_front_direct_support = (
            front.direct_support_mask.copy()
        )
        event.contact_bridge_front_eventual_support = (
            front.eventual_support_mask.copy()
        )
        event.contact_bridge_direct_support_fraction = (
            front.direct_support_fraction
        )
        event.contact_bridge_eventual_support_fraction = (
            front.eventual_support_fraction
        )
        event.contact_bridge_post_direct_support_fraction = post_direct
        event.contact_bridge_post_eventual_support_fraction = post_eventual
        (
            event.contact_bridge_growth_step_count,
            event.contact_bridge_tip_step_median_px,
            event.contact_bridge_tip_step_max_px,
            event.contact_bridge_length_step_max_px,
        ) = dynamics
        event.contact_bridge_post_growth_step_count = post_growth_steps
        stop = endpoint + 1
        if stop < len(event.path_xy):
            event.contact_bridge_censor_sample = max(
                event.germination_sample + 1,
                min(
                    int(event.path_birth[stop]),
                    int(front.birth_samples[-1]) + 1,
                ),
            )
            event.contact_bridge_reason = (
                "verified-through-contact-until-distal-support-loss"
            )
        else:
            event.contact_bridge_reason = "verified-through-contact"
        event.quality_status = "contact-bridged-growth"


def _component_root_choices(
    sx: np.ndarray,
    sy: np.ndarray,
    skeleton_birth: np.ndarray,
    candidate_indices: np.ndarray,
    grain_anchors: list[GrainAnchor] | None,
    distance_to_old: np.ndarray,
    config: AtlasConfig,
) -> list[tuple[int, GrainAnchor | None, float | None]]:
    """Return one plausible early skeleton root for every attached pollen."""

    if not len(candidate_indices):
        return []
    if not grain_anchors:
        root_choice = int(
            candidate_indices[
                np.argmin(distance_to_old[sy[candidate_indices], sx[candidate_indices]])
            ]
        )
        return [(root_choice, None, None)]

    choices: list[tuple[int, GrainAnchor | None, float | None]] = []
    for grain in grain_anchors:
        assignments = []
        for candidate_index in candidate_indices:
            point_xy = np.asarray([sx[candidate_index], sy[candidate_index]])
            candidate_sample = int(skeleton_birth[candidate_index])
            center_distance = float(
                np.linalg.norm(point_xy - grain.center_at(candidate_sample))
            )
            attachment = abs(center_distance - grain.radius_px)
            assignments.append((attachment, int(candidate_index)))
        attachment, root_choice = min(assignments)
        if attachment <= config.max_grain_attachment_px:
            choices.append((root_choice, grain, attachment))
    return sorted(
        choices,
        key=lambda choice: (
            float("inf") if choice[2] is None else choice[2],
            -1 if choice[1] is None else choice[1].grain_id,
        ),
    )


def _angle_between_vectors_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    """Return the unsigned angle between two nonzero vectors."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return None
    cosine = float(
        np.clip(
            np.dot(first, second) / (first_norm * second_norm),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def _contact_bridge_geometry(
    path_xy: np.ndarray,
    arclength_px: np.ndarray,
    clearance_px: np.ndarray,
    contact_path_index: int,
    config: AtlasConfig,
) -> tuple[int | None, float | None, float | None, float | None, float | None]:
    """Measure a single through-grain contact and its two-sided tangents."""

    path = np.asarray(path_xy, dtype=np.float64)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    clearance = np.asarray(clearance_px, dtype=np.float64)
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or arclength.shape != (len(path),)
        or clearance.shape != (len(path),)
        or not 0 <= contact_path_index < len(path)
    ):
        return None, None, None, None, None

    contact = clearance <= config.foreign_grain_clearance_margin_px
    if not contact[contact_path_index]:
        return None, None, None, None, None
    exit_index = contact_path_index
    while exit_index < len(path) and contact[exit_index]:
        exit_index += 1
    if exit_index >= len(path) or np.any(contact[exit_index:]):
        return None, None, None, None, None

    window = config.junction_direction_window_px
    before_index = max(
        0,
        int(
            np.searchsorted(
                arclength,
                arclength[contact_path_index] - window,
                side="left",
            )
        ),
    )
    after_index = min(
        len(path) - 1,
        int(
            np.searchsorted(
                arclength,
                arclength[exit_index] + window,
                side="left",
            )
        ),
    )
    if before_index >= contact_path_index or after_index <= exit_index:
        return exit_index, None, None, None, None

    ingress = path[contact_path_index] - path[before_index]
    bridge = path[exit_index] - path[contact_path_index]
    egress = path[after_index] - path[exit_index]
    return (
        exit_index,
        float(arclength[exit_index] - arclength[contact_path_index]),
        _angle_between_vectors_degrees(ingress, bridge),
        _angle_between_vectors_degrees(bridge, egress),
        _angle_between_vectors_degrees(ingress, egress),
    )


def _birth_event_from_root(
    component_id: int,
    component_root_count: int,
    area: int,
    skeleton: np.ndarray,
    nearest_birth: np.ndarray,
    root_yx: tuple[int, int],
    birth: np.ndarray,
    distance_to_old: np.ndarray,
    config: AtlasConfig,
    assigned_grain: GrainAnchor | None,
    initial_grain_attachment: float | None,
    grain_anchors: list[GrainAnchor] | None,
    evidence: np.ndarray | None,
) -> BirthEvent | None:
    """Build and score one pollen-rooted path hypothesis from a component."""

    candidates_paths = _skeleton_path_candidates(skeleton, root_yx)
    if not candidates_paths:
        return None
    (
        path_yx,
        arclength,
        path_birth_raw,
        birth_correlation,
        birth_forward,
        birth_progress,
    ) = _select_birth_ordered_path(
        candidates_paths,
        nearest_birth,
        config,
        skeleton=skeleton,
    )
    max_junction_turn = _maximum_junction_turn_degrees(
        path_yx,
        arclength,
        skeleton,
        config.junction_direction_window_px,
    )
    if float(arclength[-1]) < config.min_germination_length_px:
        return None

    grain_attachment = initial_grain_attachment
    root_distance = float(distance_to_old[root_yx])
    if assigned_grain is not None:
        root_sample = int(nearest_birth[root_yx])
        event_grain_center = assigned_grain.center_at(root_sample)
        path_yx = _trim_grain_rim_prefix(
            path_yx,
            event_grain_center,
            assigned_grain.radius_px,
            margin_px=config.grain_exit_margin_px,
            outside_run=config.grain_exit_run_points,
        )
        if len(path_yx) < 2:
            return None
        path_yx, connector_points = _prepend_grain_rim_connector(
            path_yx,
            event_grain_center,
            assigned_grain.radius_px,
            config.grain_exit_margin_px,
        )
        steps = np.linalg.norm(
            np.diff(path_yx.astype(np.float64), axis=0), axis=1
        )
        arclength = np.concatenate([[0.0], np.cumsum(steps)])
        if float(arclength[-1]) < config.min_germination_length_px:
            return None
        path_birth_raw = nearest_birth[
            path_yx[:, 0], path_yx[:, 1]
        ].astype(np.int32)
        if connector_points:
            path_birth_raw[:connector_points] = path_birth_raw[connector_points]
        birth_correlation, birth_forward, birth_progress = _birth_order_metrics(
            path_birth_raw, arclength
        )
        root_distance = float(distance_to_old[tuple(path_yx[0])])
    if len(path_yx) < 2:
        return None

    path_xy = path_yx[:, ::-1].astype(np.float32)
    fitted_birth = isotonic_increasing(
        path_birth_raw.astype(np.float64),
        np.ones(len(path_birth_raw), dtype=np.float64),
    )
    path_birth = np.maximum.accumulate(np.rint(fitted_birth).astype(np.int32))
    length = float(arclength[-1])
    chord = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    tortuosity = length / max(chord, 1.0)
    width = _component_width_px(area, skeleton)
    birth_start = int(path_birth.min())
    birth_end = int(path_birth.max())
    germination_index = min(
        int(
            np.searchsorted(
                arclength,
                config.min_germination_length_px,
                side="left",
            )
        ),
        len(path_birth) - 1,
    )
    germination_sample = int(path_birth[germination_index])
    duration = birth_end - germination_sample
    (
        observed_growth_steps,
        observed_tip_step_median,
        observed_tip_step_max,
        observed_length_step_max,
    ) = _tip_dynamics_metrics(path_xy, path_birth, arclength)
    growth_steps = observed_growth_steps
    tip_step_median = observed_tip_step_median
    tip_step_max = observed_tip_step_max
    length_step_max = observed_length_step_max
    root_factor = max(
        config.minimum_root_score_factor,
        1.0 - root_distance / config.max_root_distance_px,
    )
    duration_factor = min(
        1.0, duration / max(1, config.full_duration_score_samples)
    )
    score = (
        length
        * root_factor
        * duration_factor
        / max(1.0, width / config.nominal_tube_width_px)
    )

    radial_extension = None
    foreign_clearance = None
    foreign_contact_path_index = None
    foreign_contact_grain_id = None
    foreign_contact_exit_path_index = None
    next_foreign_contact_path_index = None
    bridge_hidden_length = None
    bridge_ingress_chord_angle = None
    bridge_chord_egress_angle = None
    bridge_total_turn = None
    event_grain_center = None
    if assigned_grain is not None:
        event_grain_center = assigned_grain.center_at(germination_sample)
        grain_attachment = abs(
            float(np.linalg.norm(path_xy[0] - event_grain_center))
            - assigned_grain.radius_px
        )
        radial_extension = float(
            np.linalg.norm(path_xy[-1] - event_grain_center)
            - assigned_grain.radius_px
        )
        other_grains = [
            grain
            for grain in grain_anchors or []
            if grain.grain_id != assigned_grain.grain_id
        ]
        if other_grains:
            clearance_by_grain = [
                (
                    grain,
                    np.linalg.norm(
                        path_xy - grain.center_at(germination_sample),
                        axis=1,
                    )
                    - grain.radius_px,
                )
                for grain in other_grains
            ]
            foreign_clearance = min(
                float(np.min(clearance))
                for _, clearance in clearance_by_grain
            )
            contacts = []
            for grain, clearance in clearance_by_grain:
                indices = np.flatnonzero(
                    clearance <= config.foreign_grain_clearance_margin_px
                )
                if len(indices):
                    path_index = int(indices[0])
                    contacts.append(
                        (
                            path_index,
                            float(clearance[path_index]),
                            grain.grain_id,
                        )
                    )
            if contacts:
                (
                    foreign_contact_path_index,
                    _,
                    foreign_contact_grain_id,
                ) = min(contacts)
                selected_clearance = next(
                    clearance
                    for grain, clearance in clearance_by_grain
                    if grain.grain_id == foreign_contact_grain_id
                )
                (
                    foreign_contact_exit_path_index,
                    bridge_hidden_length,
                    bridge_ingress_chord_angle,
                    bridge_chord_egress_angle,
                    bridge_total_turn,
                ) = _contact_bridge_geometry(
                    path_xy,
                    arclength,
                    selected_clearance,
                    foreign_contact_path_index,
                    config,
                )
                if foreign_contact_exit_path_index is not None:
                    later_contacts = [
                        int(index)
                        for grain, clearance in clearance_by_grain
                        if grain.grain_id != foreign_contact_grain_id
                        for index in np.flatnonzero(
                            clearance
                            <= config.foreign_grain_clearance_margin_px
                        )
                        if index >= foreign_contact_exit_path_index
                    ]
                    if later_contacts:
                        next_foreign_contact_path_index = min(later_contacts)

    rim_delta = 0.0
    rim_noise = 0.0
    rim_persistent = 0
    rim_confirmed = None
    rim_sample = None
    rim_timing_error = None
    rim_timing_consistent = None
    contrast_delta = 0.0
    contrast_noise = 0.0
    contrast_persistent = 0
    contrast_confirmed = False
    contrast_sample = None
    if evidence is not None and assigned_grain is not None:
        (
            rim_delta,
            rim_noise,
            rim_persistent,
            rim_confirmed,
            rim_sample,
        ) = _rim_emergence_metrics(
            evidence,
            assigned_grain,
            path_xy,
            germination_sample,
            config,
        )
        if rim_sample is not None:
            rim_timing_error = abs(rim_sample - birth_start)
            rim_timing_consistent = (
                rim_timing_error <= config.temporal_link_samples
            )
        (
            contrast_delta,
            contrast_noise,
            contrast_persistent,
            contrast_confirmed,
            contrast_sample,
        ) = _oriented_rim_contrast_metrics(
            evidence,
            assigned_grain,
            path_xy,
            germination_sample,
            config,
        )

    (
        quality_status,
        quality_flags,
        auto_accepted,
        trajectory_accepted,
    ) = _quality_classification(
        path_xy,
        length,
        duration,
        root_distance,
        width,
        tortuosity,
        birth_correlation,
        birth_forward,
        birth_progress,
        growth_steps,
        tip_step_median,
        tip_step_max,
        radial_extension,
        foreign_clearance,
        birth.shape,
        config,
        grain_attachment,
        rim_confirmed,
        rim_timing_consistent,
        event_grain_center,
        None if assigned_grain is None else assigned_grain.radius_px,
    )
    if duration >= config.trajectory_resolution_samples:
        floor = config.branch_direction_floor_weight
        score *= (
            floor
            + (1.0 - floor) * max(0.0, birth_correlation) * birth_forward
        )
    return BirthEvent(
        event_id=0,
        component_id=component_id,
        component_root_count=component_root_count,
        area_px=area,
        path_xy=path_xy,
        path_birth_raw=path_birth_raw,
        path_birth=path_birth,
        arclength_px=arclength.astype(np.float32),
        root_distance_px=root_distance,
        tortuosity=tortuosity,
        width_px=width,
        max_junction_turn_degrees=max_junction_turn,
        birth_order_correlation=birth_correlation,
        birth_order_forward_fraction=birth_forward,
        birth_progress_samples=birth_progress,
        growth_step_count=growth_steps,
        tip_step_median_px=tip_step_median,
        tip_step_max_px=tip_step_max,
        length_step_max_px=length_step_max,
        birth_start=birth_start,
        germination_sample=germination_sample,
        birth_end=birth_end,
        score=score,
        rim_emergence_delta=rim_delta,
        rim_emergence_noise=rim_noise,
        rim_emergence_persistent_samples=rim_persistent,
        rim_emergence_confirmed=rim_confirmed,
        rim_emergence_sample=rim_sample,
        rim_emergence_timing_error_samples=rim_timing_error,
        rim_contrast_delta=contrast_delta,
        rim_contrast_noise=contrast_noise,
        rim_contrast_persistent_samples=contrast_persistent,
        rim_contrast_confirmed=contrast_confirmed,
        rim_contrast_emergence_sample=contrast_sample,
        grain_id=None if assigned_grain is None else assigned_grain.grain_id,
        grain_center_xy=(
            None if event_grain_center is None else event_grain_center.copy()
        ),
        grain_radius_px=(
            None if assigned_grain is None else assigned_grain.radius_px
        ),
        grain_attachment_px=grain_attachment,
        radial_extension_px=radial_extension,
        foreign_grain_clearance_px=foreign_clearance,
        quality_status=quality_status,
        quality_flags=quality_flags,
        auto_accepted=auto_accepted,
        trajectory_accepted=trajectory_accepted,
        observed_growth_step_count=observed_growth_steps,
        observed_tip_step_median_px=observed_tip_step_median,
        observed_tip_step_max_px=observed_tip_step_max,
        observed_length_step_max_px=observed_length_step_max,
        foreign_grain_contact_path_index=foreign_contact_path_index,
        foreign_grain_contact_grain_id=foreign_contact_grain_id,
        foreign_grain_contact_exit_path_index=(
            foreign_contact_exit_path_index
        ),
        contact_bridge_next_foreign_contact_path_index=(
            next_foreign_contact_path_index
        ),
        contact_bridge_hidden_length_px=bridge_hidden_length,
        contact_bridge_ingress_chord_angle_degrees=(
            bridge_ingress_chord_angle
        ),
        contact_bridge_chord_egress_angle_degrees=(
            bridge_chord_egress_angle
        ),
        contact_bridge_total_turn_degrees=bridge_total_turn,
    )


def _path_overlap_fraction(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    maximum_distance_px: float,
) -> float:
    """Measure how much of the shorter path is explained by the other path."""

    first = np.asarray(first_xy, dtype=np.float64)
    second = np.asarray(second_xy, dtype=np.float64)
    if not len(first) or not len(second):
        return 0.0
    shorter, longer = (first, second) if len(first) <= len(second) else (second, first)
    nearest = np.min(
        np.linalg.norm(shorter[:, None, :] - longer[None, :, :], axis=2),
        axis=1,
    )
    return float(np.mean(nearest <= maximum_distance_px))


def _resolve_component_ownership(
    events: list[BirthEvent],
    config: AtlasConfig,
) -> None:
    """Keep overlapping multi-pollen explanations visible but out of auto counts."""

    invalid_root_flags = {
        "detached-from-grain",
        "distant-from-preexisting-material",
        "no-rim-emergence",
        "rim-emergence-timing-mismatch",
    }
    component_ids = sorted({event.component_id for event in events})
    for component_id in component_ids:
        indices = [
            index
            for index, event in enumerate(events)
            if event.component_id == component_id
            and event.grain_id is not None
            and not invalid_root_flags.intersection(event.quality_flags)
        ]
        if len(indices) < 2:
            continue

        parent = {index: index for index in indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if events[left].grain_id == events[right].grain_id:
                    continue
                overlap = _path_overlap_fraction(
                    events[left].path_xy,
                    events[right].path_xy,
                    config.competing_path_distance_px,
                )
                if overlap >= config.competing_path_overlap_fraction:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in indices:
            groups.setdefault(find(index), []).append(index)
        for group in groups.values():
            if len({events[index].grain_id for index in group}) < 2:
                continue
            accepted = [index for index in group if events[index].auto_accepted]
            trajectory = [
                index for index in accepted if events[index].trajectory_accepted
            ]
            keep = trajectory[0] if len(trajectory) == 1 else None
            for index in accepted:
                if index == keep:
                    continue
                event = events[index]
                event.auto_accepted = False
                event.trajectory_accepted = False
                event.quality_status = "review-required"
                event.quality_flags = tuple(
                    dict.fromkeys(
                        (*event.quality_flags, "ambiguous-component-ownership")
                    )
                )


def _resolve_grain_event_ambiguity(
    events: list[BirthEvent],
    config: AtlasConfig,
) -> None:
    """Prevent distinct plausible branches from one pollen being counted blindly."""

    grain_ids = sorted(
        {
            event.grain_id
            for event in events
            if event.grain_id is not None and event.auto_accepted
        }
    )
    for grain_id in grain_ids:
        accepted = [
            event
            for event in events
            if event.grain_id == grain_id and event.auto_accepted
        ]
        if len(accepted) < 2:
            continue
        overlaps = [
            _path_overlap_fraction(
                left.path_xy,
                right.path_xy,
                config.competing_path_distance_px,
            )
            for position, left in enumerate(accepted)
            for right in accepted[position + 1 :]
        ]
        same_path = bool(overlaps) and all(
            overlap >= config.competing_path_overlap_fraction
            for overlap in overlaps
        )
        trajectories = [event for event in accepted if event.trajectory_accepted]
        if same_path:
            keep = max(
                accepted,
                key=lambda event: (
                    event.trajectory_accepted,
                    event.quality_status == "atlas-event",
                    event.score,
                ),
            )
        elif len(trajectories) == 1:
            keep = trajectories[0]
        else:
            keep = None

        for event in accepted:
            if event is keep:
                continue
            event.auto_accepted = False
            event.trajectory_accepted = False
            event.quality_status = "review-required"
            flag = (
                "secondary-event-for-grain"
                if keep is not None
                else "ambiguous-multiple-events-for-grain"
            )
            event.quality_flags = tuple(
                dict.fromkeys((*event.quality_flags, flag))
            )


def _rank_and_mark_primary_events(events: list[BirthEvent]) -> list[BirthEvent]:
    """Rank hypotheses and mark one primary interpretation for each pollen."""

    events.sort(
        key=lambda event: (
            event.auto_accepted,
            event.trajectory_accepted,
            event.quality_status == "atlas-event",
            event.score,
        ),
        reverse=True,
    )
    claimed_grains: set[int] = set()
    for event_id, event in enumerate(events, start=1):
        event.event_id = event_id
        event.primary_for_grain = False
        if event.grain_id is not None and event.grain_id not in claimed_grains:
            event.primary_for_grain = True
            claimed_grains.add(event.grain_id)
    return events


def extract_birth_events(
    birth: np.ndarray,
    preexisting: np.ndarray,
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
    evidence: np.ndarray | None = None,
    frames: np.ndarray | None = None,
) -> list[BirthEvent]:
    """Recover every credible pollen-rooted path from birth-time topology."""

    total = int(birth.max()) if birth.size else 0
    total = max(total, config.warmup_samples + 1)
    labels, component_count = birth_topology_labels(
        birth,
        total,
        config.warmup_samples,
        config.spatial_link_px,
        config.temporal_link_samples,
    )
    distance_to_old = cv.distanceTransform(
        (~preexisting).astype(np.uint8), cv.DIST_L2, 3
    )
    events: list[BirthEvent] = []
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    for component in range(component_count):
        raw_mask = labels == component
        area = int(raw_mask.sum())
        if area < config.min_event_pixels:
            continue
        mask = cv.morphologyEx(
            raw_mask.astype(np.uint8), cv.MORPH_CLOSE, close_kernel
        ).astype(bool)
        skeleton = skeletonize(mask)
        if skeleton.sum() < 3:
            continue
        neighbor_count = (
            ndimage.convolve(
                skeleton.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                mode="constant",
            )
            - skeleton.astype(np.uint8)
        )
        if not np.any(skeleton & (neighbor_count <= 1)):
            continue

        nearest_birth = _nearest_births(raw_mask, birth)
        sy, sx = np.nonzero(skeleton)
        skeleton_birth = nearest_birth[sy, sx]
        early_limit = float(np.percentile(skeleton_birth, 20))
        candidate_indices = np.flatnonzero(
            skeleton_birth <= early_limit + config.temporal_link_samples
        )
        root_choices = _component_root_choices(
            sx,
            sy,
            skeleton_birth,
            candidate_indices,
            grain_anchors,
            distance_to_old,
            config,
        )
        for root_choice, assigned_grain, grain_attachment in root_choices:
            event = _birth_event_from_root(
                component_id=component,
                component_root_count=len(root_choices),
                area=area,
                skeleton=skeleton,
                nearest_birth=nearest_birth,
                root_yx=(int(sy[root_choice]), int(sx[root_choice])),
                birth=birth,
                distance_to_old=distance_to_old,
                config=config,
                assigned_grain=assigned_grain,
                initial_grain_attachment=grain_attachment,
                grain_anchors=grain_anchors,
                evidence=evidence,
            )
            if event is not None:
                events.append(event)

    _resolve_component_ownership(events, config)
    _apply_broad_occlusion_review(events, frames, config)
    _resolve_grain_event_ambiguity(events, config)
    _apply_warmup_onset_review(events, evidence, frames, config)
    ranked = _rank_and_mark_primary_events(events)
    # Tip timing is independent evidence: a later germination-only review must
    # not erase an otherwise valid tube trajectory.
    _apply_path_locked_front_refinement(ranked, evidence, config)
    _apply_proximal_emergence_review(
        ranked,
        evidence,
        config,
        grain_anchors,
    )
    _promote_resolved_short_trajectories(ranked)
    _apply_full_path_temporal_validation(ranked, evidence, config)
    _apply_rim_bridged_proximal_recovery(
        ranked,
        evidence,
        config,
        grain_anchors,
    )
    _apply_rim_direction_specificity_review(
        ranked,
        evidence,
        config,
        grain_anchors,
    )
    _apply_precontact_trajectory_recovery(ranked, evidence, config)
    _apply_contact_bridge_recovery(ranked, evidence, config)
    return ranked


def _selected_tip_timing(event: BirthEvent) -> np.ndarray:
    """Return the selected full or safely censored path timing."""

    if (
        getattr(event, "front_refinement_applied", False)
        and getattr(event, "path_front_birth", None) is not None
    ):
        return event.path_front_birth
    if (
        getattr(event, "contact_bridge_trajectory_accepted", False)
        and getattr(event, "contact_bridge_front_birth", None) is not None
        and getattr(event, "contact_bridge_last_path_index", None) is not None
    ):
        stop = event.contact_bridge_last_path_index + 1
        if len(event.contact_bridge_front_birth) == stop:
            timing = event.path_birth.copy()
            timing[:stop] = event.contact_bridge_front_birth
            return timing
    if (
        getattr(event, "precontact_trajectory_accepted", False)
        and getattr(event, "precontact_front_birth", None) is not None
        and getattr(event, "foreign_grain_contact_path_index", None) is not None
    ):
        stop = event.foreign_grain_contact_path_index
        if len(event.precontact_front_birth) == stop:
            timing = event.path_birth.copy()
            timing[:stop] = event.precontact_front_birth
            return timing
    return event.path_birth


def _tip_timing_method(event: BirthEvent) -> str:
    """Name the timing source selected for one event's detailed rows."""

    if getattr(event, "front_refinement_applied", False):
        return "path-locked-temporal-front"
    if getattr(event, "contact_bridge_trajectory_accepted", False):
        return "contact-bridged-temporal-front"
    if getattr(event, "precontact_trajectory_accepted", False):
        return "contact-censored-temporal-front"
    return "isotonic-threshold-birth"


def _trajectory_measurement_scope(event: BirthEvent) -> str:
    """Describe the complete, bridged, censored, or unavailable tip scope."""

    if event.trajectory_accepted:
        return "full"
    if getattr(event, "contact_bridge_trajectory_accepted", False):
        return "contact-bridged-prefix"
    if getattr(event, "precontact_trajectory_accepted", False):
        return "pre-contact"
    return "none"


def _contact_bridge_phase(event: BirthEvent, path_index: int) -> str:
    """Locate a reported tip before, within, or after foreign-grain contact."""

    entry = getattr(event, "foreign_grain_contact_path_index", None)
    exit_index = getattr(event, "foreign_grain_contact_exit_path_index", None)
    if path_index < 0 or entry is None:
        return ""
    if path_index < entry:
        return "pre-contact"
    if exit_index is None or path_index < exit_index:
        return "contact-span"
    return "post-contact"


def _tip_measurement_is_accepted(
    event: BirthEvent,
    sample: int,
    path_index: int,
) -> bool:
    """Accept a tip only while it remains inside the validated path scope."""

    if path_index < 0 or not event.primary_for_grain:
        return False
    if event.trajectory_accepted:
        return True
    bridge_stop = getattr(event, "contact_bridge_last_path_index", None)
    bridge_censor = getattr(event, "contact_bridge_censor_sample", None)
    if getattr(event, "contact_bridge_trajectory_accepted", False):
        return bool(
            bridge_stop is not None
            and path_index <= bridge_stop
            and (bridge_censor is None or sample < bridge_censor)
        )
    stop = getattr(event, "foreign_grain_contact_path_index", None)
    censor = getattr(event, "precontact_censor_sample", None)
    return bool(
        getattr(event, "precontact_trajectory_accepted", False)
        and stop is not None
        and censor is not None
        and path_index < stop
        and sample < censor
    )


def event_tip_at(
    event: BirthEvent,
    sample: int,
    *,
    use_refined_front: bool = True,
) -> tuple[float, np.ndarray | None, int]:
    """Return connected arclength and tip for one event at one sample."""

    if sample < event.germination_sample:
        return 0.0, None, -1
    timing = _selected_tip_timing(event) if use_refined_front else event.path_birth
    indices = np.flatnonzero(timing <= sample)
    if not len(indices):
        return 0.0, None, -1
    path_index = int(indices[-1])
    return (
        float(event.arclength_px[path_index]),
        event.path_xy[path_index],
        path_index,
    )


def _tip_timing_assessment(
    event: BirthEvent,
    sample: int,
    path_index: int | None = None,
) -> TipTimingAssessment:
    """Classify whether one tip is observed, retrospectively supported, or inferred."""

    if path_index is None:
        _, _, path_index = event_tip_at(event, sample)
    measurement_accepted = _tip_measurement_is_accepted(
        event,
        sample,
        path_index,
    )
    if path_index < 0:
        return TipTimingAssessment(
            state="not-measured",
            measurement_accepted=False,
            direct_support_at_arrival=None,
            eventual_support_before_confirmation=None,
            threshold_confirmation_sample=None,
            remaining_confirmation_lag_samples=None,
        )

    confirmation = int(event.path_birth[path_index])
    remaining_lag = max(0, confirmation - int(sample))
    full_front = getattr(event, "front_refinement_applied", False)
    contact_bridge_front = bool(
        getattr(event, "contact_bridge_trajectory_accepted", False)
        and getattr(event, "contact_bridge_last_path_index", None) is not None
        and path_index <= event.contact_bridge_last_path_index
    )
    precontact_front = bool(
        getattr(event, "precontact_trajectory_accepted", False)
        and getattr(event, "foreign_grain_contact_path_index", None) is not None
        and path_index < event.foreign_grain_contact_path_index
    )
    if not full_front and not contact_bridge_front and not precontact_front:
        return TipTimingAssessment(
            state="threshold-confirmed",
            measurement_accepted=measurement_accepted,
            direct_support_at_arrival=None,
            eventual_support_before_confirmation=None,
            threshold_confirmation_sample=confirmation,
            remaining_confirmation_lag_samples=remaining_lag,
        )

    if full_front:
        direct_mask = event.path_front_direct_support
        eventual_mask = event.path_front_eventual_support
    elif contact_bridge_front:
        direct_mask = getattr(event, "contact_bridge_front_direct_support", None)
        eventual_mask = getattr(
            event,
            "contact_bridge_front_eventual_support",
            None,
        )
    else:
        direct_mask = getattr(event, "precontact_front_direct_support", None)
        eventual_mask = getattr(event, "precontact_front_eventual_support", None)
    direct = (
        None
        if direct_mask is None or path_index >= len(direct_mask)
        else bool(direct_mask[path_index])
    )
    eventual = (
        None
        if eventual_mask is None or path_index >= len(eventual_mask)
        else bool(eventual_mask[path_index])
    )
    if remaining_lag == 0:
        state = "threshold-confirmed"
    elif direct:
        state = "front-direct-supported"
    elif eventual:
        state = "front-later-supported"
    elif direct is None or eventual is None:
        state = "front-evidence-unavailable"
    else:
        state = "front-inferred"
    return TipTimingAssessment(
        state=state,
        measurement_accepted=measurement_accepted,
        direct_support_at_arrival=direct,
        eventual_support_before_confirmation=eventual,
        threshold_confirmation_sample=confirmation,
        remaining_confirmation_lag_samples=remaining_lag,
    )


def _germination_phase(
    event: BirthEvent,
    sample: int,
    onset: OnsetAssessment | None = None,
) -> str:
    """Describe accepted emergence separately from measurable tube growth."""

    if not event.primary_for_grain or not event.auto_accepted:
        return "review-required"
    if onset is not None and not onset.accepted:
        return (
            "onset-timing-review"
            if sample < event.germination_sample
            else "measurable-growth"
        )
    onset_sample = (
        event.rim_emergence_sample
        if onset is None
        else onset.sample
    )
    if onset_sample is None:
        return "onset-unavailable"
    measurement_sample = event.germination_sample
    if sample < min(onset_sample, measurement_sample):
        return "pre-germination"
    if onset_sample <= sample < measurement_sample:
        return "emerged-below-measurement"
    if measurement_sample <= sample < onset_sample:
        return "measurable-before-onset-cue"
    return "measurable-growth"


def _germination_onset_assessment(
    event: BirthEvent,
    config: AtlasConfig,
) -> OnsetAssessment:
    """Combine root, rim, and oriented-contrast cues conservatively."""

    rim = event.rim_emergence_sample
    root = event.birth_start
    contrast = getattr(event, "rim_contrast_emergence_sample", None)
    if not event.primary_for_grain or not event.auto_accepted:
        cues = () if rim is None else ("root", "rim")
        return OnsetAssessment(
            sample=None,
            quality="review-required",
            accepted=False,
            supporting_cues=cues,
            support_start_sample=(None if rim is None else min(root, rim)),
            support_end_sample=(None if rim is None else max(root, rim)),
        )
    if rim is None:
        return OnsetAssessment(
            sample=None,
            quality="unavailable",
            accepted=False,
            supporting_cues=(),
            support_start_sample=None,
            support_end_sample=None,
        )

    root_rim_gap = abs(root - rim)
    if root_rim_gap <= config.rim_history_samples:
        cue_samples = {"root": root, "rim": rim}
        if contrast is not None and max(
            abs(contrast - root), abs(contrast - rim)
        ) <= config.rim_history_samples:
            cue_samples["contrast"] = contrast
        quality = (
            "high-agreement"
            if root_rim_gap <= config.rim_persistence_window
            else "moderate-agreement"
        )
        accepted = True
    elif (
        contrast is not None
        and abs(root - contrast) <= config.rim_history_samples
    ):
        cue_samples = {"root": root, "contrast": contrast}
        quality = "contrast-supported"
        accepted = True
    elif (
        contrast is not None
        and abs(rim - contrast) <= config.rim_history_samples
    ):
        cue_samples = {"rim": rim, "contrast": contrast}
        quality = "rim-contrast-supported"
        accepted = True
    else:
        cue_samples = {"root": root, "rim": rim}
        if contrast is not None:
            cue_samples["contrast"] = contrast
        quality = "low-agreement"
        accepted = False

    values = tuple(cue_samples.values())
    onset = int(np.rint(np.median(values)))
    if onset > event.germination_sample:
        quality = "reversed-cue-order"
        accepted = False
    return OnsetAssessment(
        sample=onset if accepted else None,
        quality=quality,
        accepted=accepted,
        supporting_cues=tuple(cue_samples),
        support_start_sample=min(values),
        support_end_sample=max(values),
    )


def _onset_timing_scope(
    event: BirthEvent,
    onset: OnsetAssessment,
) -> str:
    """Separate exact optical timing from review-only cue windows."""

    if not event.primary_for_grain or not event.auto_accepted:
        return "not-accepted"
    if onset.accepted:
        return "point-estimate"
    if (
        onset.support_start_sample is not None
        and onset.support_end_sample is not None
    ):
        return "optical-review-window"
    return "unavailable"


def write_birth_map(
    output: Path,
    frame: np.ndarray,
    birth: np.ndarray,
    warmup: int,
) -> None:
    """Write a raw-image overlay colored only where new material appeared."""

    total = int(birth.max())
    valid = (birth > warmup) & (birth < total)
    scaled = np.zeros_like(frame)
    if valid.any():
        values = birth[valid].astype(np.float32)
        lo, hi = float(values.min()), float(values.max())
        scaled[valid] = np.clip((values - lo) / max(hi - lo, 1.0) * 255, 1, 255)
    colors = cv.applyColorMap(scaled, cv.COLORMAP_TURBO)
    base = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[valid] = colors[valid]
    cv.addWeighted(overlay, 0.72, base, 0.28, 0, base)
    cv.imwrite(str(output), base)


def write_grain_outputs(
    output_dir: Path,
    frame: np.ndarray,
    grain_anchors: list[GrainAnchor],
) -> tuple[Path, Path]:
    """Write an auditable grain catalog and circle-only baseline overlay."""

    csv_path = output_dir / "grain_anchors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grain_id", "x_px", "y_px", "radius_px",
                "median_x_px", "median_y_px", "observations",
                "warmup_observations", "first_sample", "last_sample",
                "circle_score", "confidence",
            ]
        )
        for grain in grain_anchors:
            first_center = grain.center_at(0)
            samples = (
                np.asarray(grain.sample_indices, dtype=np.int32)
                if grain.sample_indices is not None and len(grain.sample_indices)
                else np.asarray([0], dtype=np.int32)
            )
            writer.writerow(
                [
                    grain.grain_id,
                    round(float(first_center[0]), 3),
                    round(float(first_center[1]), 3),
                    round(grain.radius_px, 3),
                    round(float(grain.center_xy[0]), 3),
                    round(float(grain.center_xy[1]), 3),
                    grain.observations,
                    grain.warmup_observations,
                    int(samples[0]),
                    int(samples[-1]),
                    round(grain.circle_score, 4),
                    (
                        "high"
                        if (grain.warmup_observations or 0) >= 3
                        else "confirmed"
                    ),
                ]
            )

    image_path = output_dir / "grain_anchors.jpg"
    image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for grain in grain_anchors:
        center = tuple(np.rint(grain.center_at(0)).astype(int))
        color = (
            (40, 210, 70)
            if (grain.warmup_observations or 0) >= 3
            else (20, 190, 240)
        )
        cv.circle(
            image,
            center,
            max(2, int(round(grain.radius_px))),
            color,
            1,
            cv.LINE_AA,
        )
    cv.imwrite(str(image_path), image)
    return csv_path, image_path


def write_event_audit_image(
    path: Path,
    frame: np.ndarray,
    events: list[BirthEvent],
    accepted: bool,
) -> None:
    """Draw labeled primary events for rapid accepted/review adjudication."""

    image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for event in events:
        if not event.primary_for_grain or event.auto_accepted != accepted:
            continue
        points = np.rint(event.path_xy).astype(np.int32)
        if event.quality_status == "atlas-event":
            color = (20, 170, 240)
        elif event.quality_status in {
            "germination-only",
            "emergence-only",
            "partial-field-event",
        }:
            color = (20, 210, 220)
        elif accepted:
            color = (45, 200, 55)
        else:
            color = (20, 215, 240)
        cv.polylines(image, [points], False, color, 2, cv.LINE_AA)
        root = tuple(points[0])
        tip = tuple(points[-1])
        cv.circle(image, root, 3, (255, 120, 20), -1, cv.LINE_AA)
        cv.circle(image, tip, 3, (0, 0, 230), -1, cv.LINE_AA)
        label = f"E{event.event_id}/G{event.grain_id}"
        cv.putText(
            image,
            label,
            (root[0] + 4, max(12, root[1] - 4)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            label,
            (root[0] + 4, max(12, root[1] - 4)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    if not cv.imwrite(str(path), image):
        raise RuntimeError(f"could not write event audit image: {path}")


def _event_audit_samples(
    event: BirthEvent,
    total_samples: int,
    config: AtlasConfig,
) -> tuple[tuple[str, int], ...]:
    """Choose comparable temporal landmarks for one event review strip."""

    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    last_sample = total_samples - 1
    assessment = _germination_onset_assessment(event, config)
    onset = (
        assessment.sample
        if assessment.sample is not None
        else min(
            sample
            for sample in (
                event.birth_start,
                event.rim_emergence_sample,
                event.rim_contrast_emergence_sample,
            )
            if sample is not None
        )
    )
    rim = (
        event.rim_emergence_sample
        if event.rim_emergence_sample is not None
        else onset
    )
    contrast = (
        event.rim_contrast_emergence_sample
        if event.rim_contrast_emergence_sample is not None
        else onset
    )
    history = max(
        config.persistence_window + 1,
        config.rim_history_samples // 2,
    )

    def bounded(sample: int) -> int:
        return min(last_sample, max(0, int(sample)))

    landmarks = [
        ("start", 0),
        ("warmup", bounded(config.warmup_samples - 1)),
        ("onset-before", bounded(onset - history)),
        ("root", bounded(event.birth_start)),
        ("rim", bounded(rim)),
        ("contrast", bounded(contrast)),
        ("measure", bounded(event.germination_sample)),
        ("final", bounded(event.birth_end)),
    ]
    return tuple(sorted(landmarks, key=lambda item: item[1]))


def write_event_timeline_audit(
    path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    config: AtlasConfig,
    tile_size: int = 180,
) -> int:
    """Write temporal landmarks for accepted and warmup-withheld events."""

    selected = [
        event
        for event in events
        if event.primary_for_grain
        and (
            event.auto_accepted
            or event.trajectory_accepted
            or event.germination_left_censored
            or event.warmup_attachment_ambiguous
        )
    ]
    selected.sort(key=lambda event: event.event_id)
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    h, w = frames.shape[1:]
    strips = []
    for event in selected:
        points = np.asarray(event.path_xy, dtype=np.float64)
        review_points = points
        if event.grain_center_xy is not None:
            review_points = np.vstack([review_points, event.grain_center_xy])
        bounds_min = review_points.min(axis=0)
        bounds_max = review_points.max(axis=0)
        center = (bounds_min + bounds_max) / 2.0
        side = min(
            h,
            w,
            max(72, int(np.ceil(np.max(bounds_max - bounds_min))) + 28),
        )
        x0 = int(np.clip(round(center[0] - side / 2), 0, w - side))
        y0 = int(np.clip(round(center[1] - side / 2), 0, h - side))
        path_points = np.rint(points).astype(np.int32)
        marker_arclength = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        ]
        marker_indices = np.unique(
            np.searchsorted(
                marker_arclength,
                np.arange(0.0, marker_arclength[-1] + 1.0, 8.0),
            ).clip(max=len(path_points) - 1)
        )
        tiles = []
        for stage, sample in _event_audit_samples(event, len(frames), config):
            image = cv.cvtColor(frames[sample], cv.COLOR_GRAY2BGR)
            for marker_index in marker_indices[1:]:
                cv.circle(
                    image,
                    tuple(path_points[marker_index]),
                    2,
                    (235, 145, 20),
                    1,
                    cv.LINE_AA,
                )
            _, tip, path_index = event_tip_at(event, sample)
            if path_index >= 1:
                cv.polylines(
                    image,
                    [path_points[: path_index + 1]],
                    False,
                    (45, 205, 60),
                    2,
                    cv.LINE_AA,
                )
            cv.circle(
                image,
                tuple(path_points[0]),
                3,
                (255, 100, 20),
                1,
                cv.LINE_AA,
            )
            if tip is not None:
                cv.circle(
                    image,
                    tuple(np.rint(tip).astype(int)),
                    3,
                    (0, 0, 235),
                    -1,
                    cv.LINE_AA,
                )
            crop = image[y0 : y0 + side, x0 : x0 + side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 23), (0, 0, 0), -1)
            label = (
                f"E{event.event_id}/G{event.grain_id} "
                f"{stage} F{int(source_frames[sample])}"
            )
            cv.putText(
                tile,
                label,
                (4, 16),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))

    strip_width = tile_size * 8
    blank = np.zeros((tile_size, strip_width, 3), dtype=np.uint8)
    rows = []
    for index in range(0, len(strips), 2):
        right = strips[index + 1] if index + 1 < len(strips) else blank
        rows.append(cv.hconcat([strips[index], right]))
    sheet = cv.vconcat(rows) if rows else cv.hconcat([blank, blank])
    if not cv.imwrite(str(path), sheet, [cv.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"could not write event timeline audit: {path}")
    return len(selected)


def write_proximal_emergence_audit(
    path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
    tile_size: int = 160,
) -> int:
    """Show raw full-cadence crops for every evaluated proximal front."""

    selected = [
        event
        for event in events
        if event.primary_for_grain
        and (
            event.proximal_front_verified
            or event.proximal_emergence_ambiguous
        )
    ]
    selected.sort(key=lambda event: event.event_id)
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    offsets = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
    h, w = frames.shape[1:]
    grains_by_id = {
        grain.grain_id: grain for grain in (grain_anchors or [])
    }
    strips = []
    for event in selected:
        stop = max(
            2,
            int(
                np.searchsorted(
                    event.arclength_px,
                    config.proximal_front_length_px,
                    side="right",
                )
            ),
        )
        stop = min(stop, len(event.path_xy))
        points = np.asarray(event.path_xy[:stop], dtype=np.float64)
        x0, y0, side = _square_review_crop(
            points,
            event.grain_center_xy,
            (h, w),
        )
        grain = grains_by_id.get(event.grain_id)
        reference_center = (
            None
            if grain is None
            else grain.center_at(event.germination_sample)
        )
        tiles = []
        review_center = getattr(event, "rim_emergence_sample", None)
        if review_center is None:
            review_center = event.germination_sample
        for offset in offsets:
            sample = int(
                np.clip(
                    review_center + offset,
                    0,
                    len(frames) - 1,
                )
            )
            image = cv.cvtColor(frames[sample], cv.COLOR_GRAY2BGR)
            sample_points = points
            if grain is not None and reference_center is not None:
                sample_points = points + (
                    grain.center_at(sample) - reference_center
                )
            cv.circle(
                image,
                tuple(np.rint(points[0]).astype(int)),
                4,
                (220, 220, 220),
                1,
                cv.LINE_AA,
            )
            cv.circle(
                image,
                tuple(np.rint(points[-1]).astype(int)),
                4,
                (220, 220, 220),
                1,
                cv.LINE_AA,
            )
            cv.circle(
                image,
                tuple(np.rint(sample_points[0]).astype(int)),
                2,
                (255, 100, 20),
                1,
                cv.LINE_AA,
            )
            cv.circle(
                image,
                tuple(np.rint(sample_points[-1]).astype(int)),
                2,
                (20, 220, 255),
                1,
                cv.LINE_AA,
            )
            crop = image[y0 : y0 + side, x0 : x0 + side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 23), (0, 0, 0), -1)
            binding_class = getattr(
                event,
                "proximal_motion_binding_class",
                "not-evaluated",
            )
            if event.proximal_front_verified:
                decision = "PASS"
            elif binding_class == "static-only":
                decision = "STATIC-ONLY"
            else:
                decision = "REVIEW"
            label_color = (
                (120, 255, 120)
                if event.proximal_front_verified
                else (80, 190, 255)
            )
            label = (
                f"E{event.event_id}/G{event.grain_id} {decision} {offset:+d} "
                f"F{int(source_frames[sample])}"
            )
            label_scale = min(
                0.34,
                0.34
                * (tile_size - 8)
                / max(
                    1,
                    cv.getTextSize(
                        label,
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.34,
                        1,
                    )[0][0],
                ),
            )
            cv.putText(
                tile,
                label,
                (4, 16),
                cv.FONT_HERSHEY_SIMPLEX,
                label_scale,
                label_color,
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))

    sheet_width = tile_size * len(offsets)
    sheet = (
        cv.vconcat(strips)
        if strips
        else np.zeros((tile_size, sheet_width, 3), dtype=np.uint8)
    )
    if not cv.imwrite(str(path), sheet, [cv.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"could not write proximal emergence audit: {path}")
    return len(selected)


def write_rim_direction_specificity_audit(
    path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor] | None = None,
    tile_size: int = 160,
) -> int:
    """Show raw timelines for every independent rim-direction competitor."""

    selected = [
        event
        for event in events
        if event.primary_for_grain
        and event.rim_direction_specificity_evaluated
        and event.rim_direction_specific is False
        and event.rim_direction_competitor_rotation_degrees is not None
    ]
    selected.sort(key=lambda event: event.event_id)
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    grains_by_id = {
        grain.grain_id: grain for grain in (grain_anchors or [])
    }
    height, width = frames.shape[1:]
    strips = []
    stage_count = 6
    for event in selected:
        grain = grains_by_id.get(event.grain_id)
        reference_center = (
            event.grain_center_xy
            if grain is None
            else grain.center_at(event.germination_sample)
        )
        if reference_center is None:
            continue
        stop = min(
            len(event.path_xy),
            max(
                2,
                int(
                    np.searchsorted(
                        event.arclength_px,
                        config.proximal_front_length_px,
                        side="right",
                    )
                ),
            ),
        )
        points = np.asarray(event.path_xy[:stop], dtype=np.float64)
        theta = np.deg2rad(
            event.rim_direction_competitor_rotation_degrees
        )
        rotation = np.asarray(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )
        competitor = reference_center + (points - reference_center) @ rotation.T
        x0, y0, side = _square_review_crop(
            np.vstack([points, competitor]),
            reference_center,
            (height, width),
            minimum_side_px=72,
        )
        stages = [
            (
                "before",
                event.birth_start
                - config.rim_direction_timing_tolerance_samples,
            ),
            ("root", event.birth_start),
            ("competitor", event.rim_direction_competitor_sample),
            ("measure", event.germination_sample),
            ("selected-chain", event.rim_direction_selected_connected_sample),
            ("final", event.birth_end),
        ]
        stages = [
            (
                stage,
                int(np.clip(sample, 0, len(frames) - 1)),
            )
            for stage, sample in stages
            if sample is not None
        ]
        stages.sort(key=lambda item: item[1])
        while len(stages) < stage_count:
            stages.append(
                (
                    "final",
                    int(np.clip(event.birth_end, 0, len(frames) - 1)),
                )
            )
        stages = stages[:stage_count]
        tiles = []
        for stage, sample in stages:
            motion = (
                np.zeros(2, dtype=np.float64)
                if grain is None
                else grain.center_at(sample) - reference_center
            )
            sample_points = points + motion
            sample_competitor = competitor + motion
            image = cv.cvtColor(frames[sample], cv.COLOR_GRAY2BGR)
            cv.polylines(
                image,
                [np.rint(sample_points).astype(np.int32)],
                False,
                (40, 220, 70),
                1,
                cv.LINE_AA,
            )
            cv.polylines(
                image,
                [np.rint(sample_competitor).astype(np.int32)],
                False,
                (220, 70, 210),
                1,
                cv.LINE_AA,
            )
            cv.circle(
                image,
                tuple(np.rint(sample_points[-1]).astype(int)),
                2,
                (40, 220, 70),
                1,
                cv.LINE_AA,
            )
            cv.circle(
                image,
                tuple(np.rint(sample_competitor[-1]).astype(int)),
                2,
                (220, 70, 210),
                1,
                cv.LINE_AA,
            )
            crop = image[y0 : y0 + side, x0 : x0 + side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 35), (0, 0, 0), -1)
            label = (
                f"E{event.event_id}/G{event.grain_id} {stage} "
                f"F{int(source_frames[sample])}"
            )
            label_scale = min(
                0.32,
                0.32
                * (tile_size - 8)
                / max(
                    1,
                    cv.getTextSize(
                        label,
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        1,
                    )[0][0],
                ),
            )
            cv.putText(
                tile,
                label,
                (4, 14),
                cv.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                tile,
                (
                    "green selected / magenta competitor "
                    f"{event.rim_direction_competitor_rotation_degrees}deg"
                ),
                (4, 29),
                cv.FONT_HERSHEY_SIMPLEX,
                0.25,
                (230, 230, 230),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))

    sheet_width = tile_size * stage_count
    sheet = (
        cv.vconcat(strips)
        if strips
        else np.zeros((tile_size, sheet_width, 3), dtype=np.uint8)
    )
    if not cv.imwrite(str(path), sheet, [cv.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"could not write rim-direction audit: {path}")
    return len(strips)


def _square_review_crop(
    points_xy: np.ndarray,
    grain_center_xy: np.ndarray | None,
    image_shape: tuple[int, int],
    padding_px: int = 28,
    minimum_side_px: int = 64,
) -> tuple[int, int, int]:
    """Return a bounded square crop around one pollen-attached path."""

    points = np.asarray(points_xy, dtype=np.float64)
    if grain_center_xy is not None:
        points = np.vstack([points, np.asarray(grain_center_xy, dtype=np.float64)])
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    height, width = image_shape
    side = min(
        height,
        width,
        max(
            int(minimum_side_px),
            int(np.ceil(np.max(bounds_max - bounds_min))) + int(padding_px),
        ),
    )
    x0 = int(np.clip(round(center[0] - side / 2), 0, width - side))
    y0 = int(np.clip(round(center[1] - side / 2), 0, height - side))
    return x0, y0, side


def write_trajectory_timeline_audit(
    path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    tile_size: int = 160,
    stage_count: int = 9,
) -> int:
    """Show full, bridged, and contact-censored accepted trajectories."""

    selected = [
        event
        for event in events
        if event.primary_for_grain
        and (
            event.trajectory_accepted
            or getattr(event, "contact_bridge_trajectory_accepted", False)
            or getattr(event, "precontact_trajectory_accepted", False)
        )
    ]
    selected.sort(key=lambda event: event.event_id)
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    if stage_count < 2:
        raise ValueError("stage_count must be at least two")
    height, width = frames.shape[1:]
    strips = []
    for event in selected:
        timing = _selected_tip_timing(event)
        start = int(np.clip(event.germination_sample, 0, len(frames) - 1))
        if (
            getattr(event, "contact_bridge_trajectory_accepted", False)
            and getattr(event, "contact_bridge_last_path_index", None)
            is not None
        ):
            stop = (
                event.contact_bridge_censor_sample - 1
                if event.contact_bridge_censor_sample is not None
                else int(np.max(timing))
            )
            review_path = event.path_xy[
                : event.contact_bridge_last_path_index + 1
            ]
        elif (
            getattr(event, "precontact_trajectory_accepted", False)
            and getattr(event, "precontact_censor_sample", None) is not None
        ):
            stop = event.precontact_censor_sample - 1
            contact_index = getattr(
                event,
                "foreign_grain_contact_path_index",
                None,
            )
            review_path = event.path_xy[
                : None if contact_index is None else contact_index + 1
            ]
        else:
            stop = int(np.max(timing))
            review_path = event.path_xy
        stop = int(np.clip(stop, start, len(frames) - 1))
        samples = np.rint(
            np.linspace(start, stop, num=stage_count)
        ).astype(np.int32)
        x0, y0, side = _square_review_crop(
            review_path,
            event.grain_center_xy,
            (height, width),
        )
        tiles = []
        for sample in samples:
            length, tip, path_index = event_tip_at(event, int(sample))
            timing_assessment = _tip_timing_assessment(
                event,
                int(sample),
                path_index,
            )
            image = cv.cvtColor(frames[sample], cv.COLOR_GRAY2BGR)
            cv.circle(
                image,
                tuple(np.rint(event.path_xy[0]).astype(int)),
                2,
                (255, 100, 20),
                1,
                cv.LINE_AA,
            )
            if tip is not None:
                tip_color = {
                    "threshold-confirmed": (20, 20, 235),
                    "front-direct-supported": (35, 190, 90),
                    "front-later-supported": (30, 190, 220),
                    "front-inferred": (200, 60, 180),
                    "front-evidence-unavailable": (150, 150, 150),
                }.get(timing_assessment.state, (20, 20, 235))
                cv.circle(
                    image,
                    tuple(np.rint(tip).astype(int)),
                    3,
                    tip_color,
                    1,
                    cv.LINE_AA,
                )
            crop = image[y0 : y0 + side, x0 : x0 + side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 37), (0, 0, 0), -1)
            if getattr(event, "contact_bridge_trajectory_accepted", False):
                decision = "BRIDGED"
            elif getattr(event, "precontact_trajectory_accepted", False):
                decision = "PRE-CONTACT"
            else:
                decision = "PASS" if event.auto_accepted else "G-REVIEW"
            label = (
                f"E{event.event_id}/G{event.grain_id} {decision} "
                f"F{int(source_frames[sample])} L{length:.1f}"
            )
            label_scale = min(
                0.32,
                0.32
                * (tile_size - 8)
                / max(
                    1,
                    cv.getTextSize(
                        label,
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        1,
                    )[0][0],
                ),
            )
            cv.putText(
                tile,
                label,
                (4, 14),
                cv.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            evidence_label = {
                "threshold-confirmed": "CONFIRMED",
                "front-direct-supported": "DIRECT",
                "front-later-supported": "LATER-SUPPORTED",
                "front-inferred": "INFERRED",
                "front-evidence-unavailable": "EVIDENCE N/A",
                "not-measured": "NO TIP",
            }[timing_assessment.state]
            lag = timing_assessment.remaining_confirmation_lag_samples
            if lag:
                evidence_label = f"{evidence_label} +{lag} samples"
            evidence_scale = min(
                0.28,
                0.28
                * (tile_size - 8)
                / max(
                    1,
                    cv.getTextSize(
                        evidence_label,
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        1,
                    )[0][0],
                ),
            )
            cv.putText(
                tile,
                evidence_label,
                (4, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                evidence_scale,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))

    sheet_width = tile_size * stage_count
    sheet = (
        cv.vconcat(strips)
        if strips
        else np.zeros((tile_size, sheet_width, 3), dtype=np.uint8)
    )
    if not cv.imwrite(str(path), sheet, [cv.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"could not write trajectory timeline audit: {path}")
    return len(selected)


def _validation_cases(
    events: list[BirthEvent],
    grain_anchors: list[GrainAnchor],
    maximum_per_stratum: int,
    boundary_grain_ids: set[int] | None = None,
) -> list[tuple[str, BirthEvent | None, GrainAnchor]]:
    """Choose a deterministic precision-and-recall review sample."""

    if maximum_per_stratum <= 0:
        return []
    grains_by_id = {grain.grain_id: grain for grain in grain_anchors}
    boundary_ids = set(boundary_grain_ids or ()) & set(grains_by_id)
    core_grain_ids = set(grains_by_id) - boundary_ids
    primaries = [
        event
        for event in events
        if event.primary_for_grain and event.grain_id in core_grain_ids
    ]
    onset_review = [
        event
        for event in primaries
        if event.germination_left_censored
        or event.warmup_attachment_ambiguous
    ]
    onset_review_ids = {id(event) for event in onset_review}
    accepted = [event for event in primaries if event.auto_accepted]
    rejected = [
        event
        for event in primaries
        if not event.auto_accepted and id(event) not in onset_review_ids
    ]
    event_grain_ids = {
        event.grain_id for event in events if event.grain_id in core_grain_ids
    }
    primary_grain_ids = {event.grain_id for event in primaries}
    nonprimary_only = []
    for grain_id in sorted(event_grain_ids - primary_grain_ids):
        candidates = [event for event in events if event.grain_id == grain_id]
        nonprimary_only.append(max(candidates, key=lambda event: event.score))
    no_hypothesis = [
        grain
        for grain in grain_anchors
        if grain.grain_id in core_grain_ids
        and grain.grain_id not in event_grain_ids
    ]

    def spaced(items: list[object]) -> list[object]:
        if len(items) <= maximum_per_stratum:
            return items
        indices = np.rint(
            np.linspace(0, len(items) - 1, maximum_per_stratum)
        ).astype(int)
        return [items[int(index)] for index in np.unique(indices)]

    cases: list[tuple[str, BirthEvent | None, GrainAnchor]] = []
    for stratum, candidates in (
        ("accepted", sorted(accepted, key=lambda event: event.event_id)),
        (
            "onset-review",
            sorted(onset_review, key=lambda event: event.event_id),
        ),
        ("rejected", sorted(rejected, key=lambda event: event.event_id)),
    ):
        for event in spaced(candidates):
            cases.append((stratum, event, grains_by_id[event.grain_id]))
    for event in spaced(nonprimary_only):
        cases.append(
            ("nonprimary-only", event, grains_by_id[event.grain_id])
        )
    for grain in spaced(sorted(no_hypothesis, key=lambda item: item.grain_id)):
        cases.append(("no-hypothesis", None, grain))
    boundary_cases = []
    for grain_id in sorted(boundary_ids):
        candidates = [event for event in events if event.grain_id == grain_id]
        event = (
            max(candidates, key=lambda item: (item.primary_for_grain, item.score))
            if candidates
            else None
        )
        boundary_cases.append((event, grains_by_id[grain_id]))
    for event, grain in spaced(boundary_cases):
        cases.append(("boundary", event, grain))

    def blind_key(case: tuple[str, BirthEvent | None, GrainAnchor]) -> int:
        _, event, grain = case
        event_id = 0 if event is None else event.event_id
        return (
            grain.grain_id * 2_654_435_761
            ^ event_id * 2_246_822_519
        ) & 0xFFFFFFFF

    return sorted(cases, key=blind_key)


def write_blinded_validation_audit(
    image_path: Path,
    manifest_path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    grain_anchors: list[GrainAnchor],
    config: AtlasConfig,
    tile_size: int = 160,
    maximum_per_stratum: int = 8,
) -> dict[str, int]:
    """Write neutral pollen-centered time strips plus a review manifest."""

    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    if len(frames) != len(source_frames) or not len(frames):
        raise ValueError("frames and source_frames must have equal nonzero length")
    height, width = frames.shape[1:]
    crop_side = min(
        height,
        width,
        max(48, int(round(96 * config.width / 480))),
    )
    boundary_grain_ids = set()
    boundary_samples = (0, min(config.warmup_samples - 1, len(frames) - 1), len(frames) - 1)
    for grain in grain_anchors:
        clearance = (
            grain.radius_px
            + config.proximal_front_length_px
            + config.grain_exit_margin_px
        )
        for sample in boundary_samples:
            center = grain.center_at(sample)
            if (
                center[0] < clearance
                or center[0] > width - 1 - clearance
                or center[1] < clearance
                or center[1] > height - 1 - clearance
            ):
                boundary_grain_ids.add(grain.grain_id)
                break
    cases = _validation_cases(
        events,
        grain_anchors,
        maximum_per_stratum,
        boundary_grain_ids,
    )
    history = max(
        config.persistence_window + 1,
        config.rim_history_samples // 2,
    )
    counts: dict[str, int] = {}
    strips = []
    manifest_rows = []
    for case_index, (stratum, event, grain) in enumerate(cases, start=1):
        case_id = f"V{case_index:03d}"
        counts[stratum] = counts.get(stratum, 0) + 1
        if event is None:
            samples = np.asarray(
                [
                    0,
                    config.warmup_samples - 1,
                    round((len(frames) - 1) / 3),
                    round(2 * (len(frames) - 1) / 3),
                    len(frames) - 1,
                ],
                dtype=int,
            )
        else:
            onset = (
                event.rim_emergence_sample
                if event.rim_emergence_sample is not None
                else event.germination_sample
            )
            samples = np.asarray(
                [
                    0,
                    config.warmup_samples - 1,
                    onset - history,
                    onset,
                    len(frames) - 1,
                ],
                dtype=int,
            )
        samples = np.clip(samples, 0, len(frames) - 1)
        tiles = []
        for stage_index, sample in enumerate(samples, start=1):
            image = cv.cvtColor(frames[int(sample)], cv.COLOR_GRAY2BGR)
            center = grain.center_at(int(sample))
            center_int = np.rint(center).astype(int)
            marker_radius = max(4, int(round(grain.radius_px + 3)))
            tick = 3
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                inner = (
                    int(center_int[0] + dx * marker_radius),
                    int(center_int[1] + dy * marker_radius),
                )
                outer = (
                    int(center_int[0] + dx * (marker_radius + tick)),
                    int(center_int[1] + dy * (marker_radius + tick)),
                )
                cv.line(image, inner, outer, (255, 100, 20), 1, cv.LINE_AA)
            x0 = int(np.clip(round(center[0] - crop_side / 2), 0, width - crop_side))
            y0 = int(np.clip(round(center[1] - crop_side / 2), 0, height - crop_side))
            crop = image[y0 : y0 + crop_side, x0 : x0 + crop_side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 23), (0, 0, 0), -1)
            cv.putText(
                tile,
                f"{case_id} T{stage_index} F{int(source_frames[sample])}",
                (4, 16),
                cv.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))
        manifest_rows.append(
            {
                "validation_case_id": case_id,
                "tracker_stratum": stratum,
                "grain_id": grain.grain_id,
                "event_id": "" if event is None else event.event_id,
                "quality_status": "" if event is None else event.quality_status,
                "auto_accepted": "" if event is None else event.auto_accepted,
                "trajectory_accepted": (
                    "" if event is None else event.trajectory_accepted
                ),
                "score": "" if event is None else round(event.score, 4),
                "quality_flags": (
                    "" if event is None else ";".join(event.quality_flags)
                ),
                "review_source_frames": ";".join(
                    str(int(source_frames[sample])) for sample in samples
                ),
                "human_germination_present": "",
                "human_onset_visible": "",
                "human_tip_trackable": "",
                "notes": "",
            }
        )

    strip_width = tile_size * 5
    blank = np.zeros((tile_size, strip_width, 3), dtype=np.uint8)
    rows = []
    for index in range(0, len(strips), 2):
        right = strips[index + 1] if index + 1 < len(strips) else blank
        rows.append(cv.hconcat([strips[index], right]))
    sheet = cv.vconcat(rows) if rows else cv.hconcat([blank, blank])
    if not cv.imwrite(str(image_path), sheet, [cv.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"could not write blinded validation audit: {image_path}")
    fieldnames = list(manifest_rows[0]) if manifest_rows else [
        "validation_case_id",
        "tracker_stratum",
        "grain_id",
        "event_id",
        "quality_status",
        "auto_accepted",
        "trajectory_accepted",
        "score",
        "quality_flags",
        "review_source_frames",
        "human_germination_present",
        "human_onset_visible",
        "human_tip_trackable",
        "notes",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    counts["total"] = len(cases)
    return counts


def write_front_refinement_audit(
    path: Path,
    frames: np.ndarray,
    source_frames: np.ndarray,
    events: list[BirthEvent],
    tile_size: int = 180,
) -> int:
    """Write raw-image strips for every fitted temporal-front candidate."""

    selected = [
        event
        for event in events
        if event.primary_for_grain
        and (
            event.path_front_birth is not None
            or getattr(event, "contact_bridge_front_birth", None) is not None
            or getattr(event, "precontact_front_birth", None) is not None
            or getattr(event, "full_path_temporal_evaluated", False)
        )
    ]
    selected.sort(key=lambda event: event.event_id)
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 pixels")
    h, w = frames.shape[1:]
    strips = []
    for event in selected:
        contact_bridge_birth = getattr(
            event,
            "contact_bridge_front_birth",
            None,
        )
        precontact_birth = getattr(event, "precontact_front_birth", None)
        if event.path_front_birth is not None:
            points = np.asarray(event.path_xy, dtype=np.float64)
            candidate_birth = np.asarray(
                event.path_front_birth,
                dtype=np.int32,
            )
            audit_end_sample = event.birth_end
        elif contact_bridge_birth is not None:
            candidate_birth = np.asarray(
                contact_bridge_birth,
                dtype=np.int32,
            )
            points = np.asarray(
                event.path_xy[: len(candidate_birth)],
                dtype=np.float64,
            )
            audit_end_sample = (
                event.contact_bridge_censor_sample
                if event.contact_bridge_censor_sample is not None
                else int(candidate_birth[-1])
            )
        elif precontact_birth is not None:
            candidate_birth = np.asarray(precontact_birth, dtype=np.int32)
            points = np.asarray(
                event.path_xy[: len(candidate_birth)],
                dtype=np.float64,
            )
            audit_end_sample = (
                event.precontact_censor_sample
                if event.precontact_censor_sample is not None
                else int(candidate_birth[-1])
            )
        else:
            points = np.asarray(event.path_xy, dtype=np.float64)
            candidate_birth = np.asarray(event.path_birth, dtype=np.int32)
            audit_end_sample = event.birth_end
        review_points = points
        contact_index = getattr(event, "foreign_grain_contact_path_index", None)
        if (
            precontact_birth is not None
            and contact_index is not None
            and contact_index < len(event.path_xy)
        ):
            review_points = np.vstack(
                [review_points, event.path_xy[contact_index]]
            )
        if event.grain_center_xy is not None:
            review_points = np.vstack([review_points, event.grain_center_xy])
        bounds_min = review_points.min(axis=0)
        bounds_max = review_points.max(axis=0)
        center = (bounds_min + bounds_max) / 2.0
        side = min(
            h,
            w,
            max(72, int(np.ceil(np.max(bounds_max - bounds_min))) + 28),
        )
        x0 = int(np.clip(round(center[0] - side / 2), 0, w - side))
        y0 = int(np.clip(round(center[1] - side / 2), 0, h - side))
        candidate_times = np.unique(candidate_birth)
        quantile_indices = np.rint(
            np.linspace(0, len(candidate_times) - 1, 4)
        ).astype(int)
        samples = [
            max(0, event.germination_sample - 1),
            event.germination_sample,
            *(int(candidate_times[index]) for index in quantile_indices[1:3]),
            int(candidate_birth[-1]),
            audit_end_sample,
        ]
        path_points = np.rint(points).astype(np.int32)
        tiles = []
        for sample in samples:
            sample = int(np.clip(sample, 0, len(frames) - 1))
            image = cv.cvtColor(frames[sample], cv.COLOR_GRAY2BGR)
            candidate_indices = np.flatnonzero(candidate_birth <= sample)
            if len(candidate_indices):
                candidate_index = int(candidate_indices[-1])
                color = (
                    (45, 205, 60)
                    if event.front_refinement_applied
                    or getattr(
                        event,
                        "contact_bridge_trajectory_accepted",
                        False,
                    )
                    or getattr(event, "precontact_trajectory_accepted", False)
                    else (
                        (230, 130, 30)
                        if getattr(event, "full_path_temporal_verified", False)
                        else (20, 175, 240)
                    )
                )
                if candidate_index >= 1:
                    cv.polylines(
                        image,
                        [path_points[: candidate_index + 1]],
                        False,
                        color,
                        1,
                        cv.LINE_AA,
                    )
                cv.circle(
                    image,
                    tuple(path_points[candidate_index]),
                    3,
                    (0, 0, 235),
                    -1,
                    cv.LINE_AA,
                )
            _, threshold_tip, _ = event_tip_at(
                event,
                sample,
                use_refined_front=False,
            )
            if threshold_tip is not None:
                cv.circle(
                    image,
                    tuple(np.rint(threshold_tip).astype(int)),
                    4,
                    (0, 230, 230),
                    1,
                    cv.LINE_AA,
                )
            cv.circle(
                image,
                tuple(path_points[0]),
                2,
                (255, 100, 20),
                -1,
                cv.LINE_AA,
            )
            crop = image[y0 : y0 + side, x0 : x0 + side]
            tile = cv.resize(
                crop,
                (tile_size, tile_size),
                interpolation=cv.INTER_NEAREST,
            )
            cv.rectangle(tile, (0, 0), (tile_size - 1, 23), (0, 0, 0), -1)
            if getattr(event, "contact_bridge_trajectory_accepted", False):
                disposition = "CONTACT-BRIDGED"
            elif getattr(event, "precontact_trajectory_accepted", False):
                disposition = "PRE-CONTACT"
            elif getattr(event, "precontact_front_birth", None) is not None:
                disposition = "CONTACT-WITHHELD"
            elif event.front_refinement_applied:
                disposition = "APPLIED"
            elif getattr(event, "full_path_temporal_verified", False):
                disposition = "VALIDATED"
            elif getattr(event, "full_path_temporal_evaluated", False):
                disposition = "REJECTED"
            else:
                disposition = "WITHHELD"
            label = (
                f"E{event.event_id}/G{event.grain_id} {disposition} "
                f"F{int(source_frames[sample])}"
            )
            cv.putText(
                tile,
                label,
                (4, 16),
                cv.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)
        strips.append(cv.hconcat(tiles))

    sheet_width = tile_size * 6
    sheet = (
        cv.vconcat(strips)
        if strips
        else np.zeros((tile_size, sheet_width, 3), dtype=np.uint8)
    )
    if not cv.imwrite(str(path), sheet, [cv.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"could not write front refinement audit: {path}")
    return len(selected)


def write_outputs(
    output_dir: Path,
    movie: Path,
    aligned: np.ndarray,
    source_frames: np.ndarray,
    fps: float,
    sample_seconds: float,
    source_frame_interval_seconds: float | None,
    pixel_size: float | None,
    distance_unit: str,
    birth: np.ndarray,
    events: list[BirthEvent],
    calibration: dict[str, float],
    shifts: np.ndarray,
    responses: np.ndarray,
    config: AtlasConfig,
    grain_anchors: list[GrainAnchor],
    max_events: int,
    render_event_ids: list[int] | None,
    write_review_video: bool,
) -> None:
    """Write experiment-grade JSON/CSV/map/video artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if render_event_ids:
        wanted = set(render_event_ids)
        selected = [event for event in events if event.event_id in wanted]
    else:
        primary = [
            event
            for event in events
            if event.primary_for_grain
            and (
                event.trajectory_accepted
                or event.contact_bridge_trajectory_accepted
                or event.precontact_trajectory_accepted
                or event.quality_status == "atlas-event"
            )
        ]
        selected = primary[:max_events] if max_events > 0 else []
    actual_sample_seconds = sample_interval_seconds(
        source_frames, fps, source_frame_interval_seconds
    )
    seconds_per_source_frame = (
        source_frame_interval_seconds
        if source_frame_interval_seconds is not None
        else 1.0 / fps
    )
    time_basis = (
        "experiment-calibrated"
        if source_frame_interval_seconds is not None
        else "encoded-video-playback"
    )
    native_width = movie_metadata(movie)[2]
    scale_to_native = native_width / config.width
    onset_assessment_by_event = {
        event.event_id: _germination_onset_assessment(event, config)
        for event in events
    }
    detailed_path = output_dir / "event_measurements.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_id", "sample", "source_frame", "elapsed_time_sec",
                "time_basis", "primary_for_grain", "germination_accepted",
                "germination_onset_time_accepted",
                "germination_onset_quality",
                "germination_onset_timing_scope",
                "trajectory_accepted",
                "contact_bridge_trajectory_accepted",
                "trajectory_measurement_scope",
                "contact_bridge_phase",
                "germination_phase", "length_px", "length_native_px",
                "length_calibrated", "distance_unit", "tip_x_px", "tip_y_px",
                "tip_x_source_px", "tip_y_source_px",
                "tip_timing_method", "tip_timing_state",
                "tip_measurement_accepted",
                "tip_direct_support_at_arrival",
                "tip_eventual_support_before_confirmation",
                "tip_threshold_confirmation_sample",
                "tip_threshold_confirmation_source_frame",
                "tip_threshold_confirmation_lag_samples",
                "tip_threshold_confirmation_lag_sec", "threshold_length_px",
                "threshold_length_native_px", "threshold_tip_x_px",
                "threshold_tip_y_px", "threshold_tip_x_source_px",
                "threshold_tip_y_source_px",
            ]
        )
        for event in events:
            for sample, source_frame in enumerate(source_frames):
                length, tip, path_index = event_tip_at(event, sample)
                tip_timing = _tip_timing_assessment(
                    event,
                    sample,
                    path_index,
                )
                threshold_length, threshold_tip, _ = event_tip_at(
                    event,
                    sample,
                    use_refined_front=False,
                )
                native_length = length * scale_to_native
                writer.writerow(
                    [
                        event.event_id,
                        sample,
                        int(source_frame),
                        round(
                            float(source_frame - source_frames[0])
                            * seconds_per_source_frame,
                            4,
                        ),
                        time_basis,
                        event.primary_for_grain,
                        event.primary_for_grain and event.auto_accepted,
                        onset_assessment_by_event[event.event_id].accepted,
                        onset_assessment_by_event[event.event_id].quality,
                        _onset_timing_scope(
                            event,
                            onset_assessment_by_event[event.event_id],
                        ),
                        event.primary_for_grain and event.trajectory_accepted,
                        (
                            event.primary_for_grain
                            and event.contact_bridge_trajectory_accepted
                        ),
                        _trajectory_measurement_scope(event),
                        _contact_bridge_phase(event, path_index),
                        _germination_phase(
                            event,
                            sample,
                            onset_assessment_by_event[event.event_id],
                        ),
                        round(length, 4),
                        round(native_length, 4),
                        (
                            ""
                            if pixel_size is None
                            else round(native_length * pixel_size, 6)
                        ),
                        "px" if pixel_size is None else distance_unit,
                        "" if tip is None else round(float(tip[0]), 3),
                        "" if tip is None else round(float(tip[1]), 3),
                        (
                            ""
                            if tip is None
                            else round(float(tip[0] + shifts[sample, 0]), 3)
                        ),
                        (
                            ""
                            if tip is None
                            else round(float(tip[1] + shifts[sample, 1]), 3)
                        ),
                        _tip_timing_method(event),
                        tip_timing.state,
                        tip_timing.measurement_accepted,
                        (
                            ""
                            if tip_timing.direct_support_at_arrival is None
                            else tip_timing.direct_support_at_arrival
                        ),
                        (
                            ""
                            if tip_timing.eventual_support_before_confirmation
                            is None
                            else tip_timing.eventual_support_before_confirmation
                        ),
                        (
                            ""
                            if tip_timing.threshold_confirmation_sample is None
                            else tip_timing.threshold_confirmation_sample
                        ),
                        (
                            ""
                            if tip_timing.threshold_confirmation_sample is None
                            else int(
                                source_frames[
                                    tip_timing.threshold_confirmation_sample
                                ]
                            )
                        ),
                        (
                            ""
                            if tip_timing.remaining_confirmation_lag_samples
                            is None
                            else tip_timing.remaining_confirmation_lag_samples
                        ),
                        (
                            ""
                            if tip_timing.threshold_confirmation_sample is None
                            else round(
                                max(
                                    0.0,
                                    float(
                                        source_frames[
                                            tip_timing.threshold_confirmation_sample
                                        ]
                                        - source_frame
                                    )
                                    * seconds_per_source_frame,
                                ),
                                4,
                            )
                        ),
                        round(threshold_length, 4),
                        round(threshold_length * scale_to_native, 4),
                        (
                            ""
                            if threshold_tip is None
                            else round(float(threshold_tip[0]), 3)
                        ),
                        (
                            ""
                            if threshold_tip is None
                            else round(float(threshold_tip[1]), 3)
                        ),
                        (
                            ""
                            if threshold_tip is None
                            else round(
                                float(threshold_tip[0] + shifts[sample, 0]),
                                3,
                            )
                        ),
                        (
                            ""
                            if threshold_tip is None
                            else round(
                                float(threshold_tip[1] + shifts[sample, 1]),
                                3,
                            )
                        ),
                    ]
                )

    summaries = []
    for event in events:
        duration_samples = event.birth_end - event.germination_sample
        duration_seconds = duration_samples * actual_sample_seconds
        onset = onset_assessment_by_event[event.event_id]
        onset_scope = _onset_timing_scope(event, onset)
        onset_sample = onset.sample
        onset_agreement_samples = (
            None
            if onset.support_start_sample is None
            else onset.support_end_sample - onset.support_start_sample
        )
        onset_lower = onset.support_start_sample
        onset_upper = onset.support_end_sample
        measurement_start_length, _, _ = event_tip_at(
            event,
            event.germination_sample,
        )
        tip_timings = []
        for sample in range(len(source_frames)):
            _, _, path_index = event_tip_at(event, sample)
            if path_index >= 0:
                tip_timings.append(
                    _tip_timing_assessment(event, sample, path_index)
                )
        preconfirmation_timings = [
            timing
            for timing in tip_timings
            if timing.state.startswith("front-")
        ]
        preconfirmation_lags = [
            timing.remaining_confirmation_lag_samples
            for timing in preconfirmation_timings
            if timing.remaining_confirmation_lag_samples is not None
        ]
        contact_index = event.foreign_grain_contact_path_index
        precontact_safe_length = (
            None
            if contact_index is None or contact_index <= 0
            else float(event.arclength_px[contact_index - 1])
        )
        contact_bridge_max_length = (
            None
            if event.contact_bridge_last_path_index is None
            else float(
                event.arclength_px[event.contact_bridge_last_path_index]
            )
        )
        maximum_accepted_length = (
            float(event.arclength_px[-1])
            if event.trajectory_accepted
            else (
                contact_bridge_max_length
                if event.contact_bridge_trajectory_accepted
                else (
                    precontact_safe_length
                    if event.precontact_trajectory_accepted
                    else None
                )
            )
        )
        summaries.append(
            {
                "event_id": event.event_id,
                "component_id": event.component_id,
                "component_root_count": event.component_root_count,
                "grain_id": event.grain_id,
                "primary_for_grain": event.primary_for_grain,
                "auto_accepted": event.auto_accepted,
                "germination_accepted": (
                    event.primary_for_grain and event.auto_accepted
                ),
                "trajectory_accepted": event.trajectory_accepted,
                "contact_bridge_trajectory_accepted": (
                    event.contact_bridge_trajectory_accepted
                ),
                "precontact_trajectory_accepted": (
                    event.precontact_trajectory_accepted
                ),
                "trajectory_measurement_scope": (
                    _trajectory_measurement_scope(event)
                ),
                "quality_status": event.quality_status,
                "quality_flags": ";".join(event.quality_flags),
                "score": round(event.score, 3),
                "area_px": event.area_px,
                "root_change_sample": event.birth_start,
                "root_change_source_frame": int(source_frames[event.birth_start]),
                "root_change_time_sec": round(
                    float(source_frames[event.birth_start] - source_frames[0])
                    * seconds_per_source_frame,
                    3,
                ),
                "germination_onset_sample": onset_sample,
                "germination_onset_source_frame": (
                    None
                    if onset_sample is None
                    else int(source_frames[onset_sample])
                ),
                "germination_onset_time_sec": (
                    None
                    if onset_sample is None
                    else round(
                        float(source_frames[onset_sample] - source_frames[0])
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "germination_onset_quality": onset.quality,
                "germination_onset_time_accepted": onset.accepted,
                "germination_onset_timing_scope": onset_scope,
                "germination_onset_review_window_available": (
                    onset_scope == "optical-review-window"
                ),
                "germination_onset_supporting_cues": "+".join(
                    onset.supporting_cues
                ),
                "germination_onset_agreement_samples": onset_agreement_samples,
                "germination_onset_cue_span_samples": onset_agreement_samples,
                "germination_onset_cue_span_sec": (
                    None
                    if onset_agreement_samples is None
                    else round(
                        onset_agreement_samples * actual_sample_seconds,
                        3,
                    )
                ),
                "germination_onset_lower_sample": onset_lower,
                "germination_onset_upper_sample": onset_upper,
                "germination_onset_lower_source_frame": (
                    None if onset_lower is None else int(source_frames[onset_lower])
                ),
                "germination_onset_upper_source_frame": (
                    None if onset_upper is None else int(source_frames[onset_upper])
                ),
                "germination_onset_lower_time_sec": (
                    None
                    if onset_lower is None
                    else round(
                        float(source_frames[onset_lower] - source_frames[0])
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "germination_onset_upper_time_sec": (
                    None
                    if onset_upper is None
                    else round(
                        float(source_frames[onset_upper] - source_frames[0])
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "measurement_start_sample": event.germination_sample,
                "measurement_start_source_frame": int(
                    source_frames[event.germination_sample]
                ),
                "measurement_start_time_sec": round(
                    float(
                        source_frames[event.germination_sample] - source_frames[0]
                    )
                    * seconds_per_source_frame,
                    3,
                ),
                "onset_to_measurement_delay_samples": (
                    None
                    if onset_sample is None
                    else event.germination_sample - onset_sample
                ),
                "onset_to_measurement_delay_sec": (
                    None
                    if onset_sample is None
                    else round(
                        (event.germination_sample - onset_sample)
                        * actual_sample_seconds,
                        3,
                    )
                ),
                "measurement_minimum_length_px": (
                    config.min_germination_length_px
                ),
                "measurement_start_length_px": round(
                    measurement_start_length,
                    4,
                ),
                "germination_sample": event.germination_sample,
                "germination_source_frame": int(
                    source_frames[event.germination_sample]
                ),
                "germination_time_sec": round(
                    float(
                        source_frames[event.germination_sample] - source_frames[0]
                    )
                    * seconds_per_source_frame,
                    3,
                ),
                "time_basis": time_basis,
                "construction_duration_s": round(duration_seconds, 3),
                "final_length_px": round(float(event.arclength_px[-1]), 3),
                "maximum_accepted_length_px": (
                    None
                    if maximum_accepted_length is None
                    else round(maximum_accepted_length, 3)
                ),
                "maximum_accepted_length_native_px": (
                    None
                    if maximum_accepted_length is None
                    else round(maximum_accepted_length * scale_to_native, 3)
                ),
                "maximum_accepted_length_calibrated": (
                    None
                    if maximum_accepted_length is None or pixel_size is None
                    else round(
                        maximum_accepted_length * scale_to_native * pixel_size,
                        6,
                    )
                ),
                "final_length_native_px": round(
                    float(event.arclength_px[-1]) * scale_to_native, 3
                ),
                "final_length_calibrated": (
                    None
                    if pixel_size is None
                    else round(
                        float(event.arclength_px[-1])
                        * scale_to_native
                        * pixel_size,
                        6,
                    )
                ),
                "distance_unit": "px" if pixel_size is None else distance_unit,
                "root_xy": [round(float(v), 2) for v in event.path_xy[0]],
                "final_tip_xy": [round(float(v), 2) for v in event.path_xy[-1]],
                "root_source_xy": [
                    round(float(event.path_xy[0, axis] + shifts[event.germination_sample, axis]), 2)
                    for axis in range(2)
                ],
                "final_tip_source_xy": [
                    round(float(event.path_xy[-1, axis] + shifts[event.birth_end, axis]), 2)
                    for axis in range(2)
                ],
                "root_distance_px": round(event.root_distance_px, 3),
                "grain_attachment_px": (
                    None
                    if event.grain_attachment_px is None
                    else round(event.grain_attachment_px, 3)
                ),
                "grain_center_xy": (
                    None
                    if event.grain_center_xy is None
                    else [round(float(v), 2) for v in event.grain_center_xy]
                ),
                "grain_center_source_xy": (
                    None
                    if event.grain_center_xy is None
                    else [
                        round(
                            float(
                                event.grain_center_xy[axis]
                                + shifts[event.germination_sample, axis]
                            ),
                            2,
                        )
                        for axis in range(2)
                    ]
                ),
                "grain_radius_px": (
                    None
                    if event.grain_radius_px is None
                    else round(event.grain_radius_px, 3)
                ),
                "rim_emergence_delta": round(event.rim_emergence_delta, 3),
                "rim_emergence_noise": round(event.rim_emergence_noise, 3),
                "rim_emergence_persistent_samples": (
                    event.rim_emergence_persistent_samples
                ),
                "rim_emergence_confirmed": event.rim_emergence_confirmed,
                "rim_emergence_sample": event.rim_emergence_sample,
                "rim_emergence_source_frame": (
                    None
                    if event.rim_emergence_sample is None
                    else int(source_frames[event.rim_emergence_sample])
                ),
                "rim_emergence_timing_error_samples": (
                    event.rim_emergence_timing_error_samples
                ),
                "rim_emergence_offset_samples": (
                    None
                    if event.rim_emergence_sample is None
                    else event.rim_emergence_sample - event.birth_start
                ),
                "rim_emergence_timing_error_s": (
                    None
                    if event.rim_emergence_timing_error_samples is None
                    else round(
                        event.rim_emergence_timing_error_samples
                        * actual_sample_seconds,
                        3,
                    )
                ),
                "rim_contrast_delta": round(event.rim_contrast_delta, 3),
                "rim_contrast_noise": round(event.rim_contrast_noise, 3),
                "rim_contrast_persistent_samples": (
                    event.rim_contrast_persistent_samples
                ),
                "rim_contrast_confirmed": event.rim_contrast_confirmed,
                "rim_contrast_emergence_sample": (
                    event.rim_contrast_emergence_sample
                ),
                "rim_contrast_emergence_source_frame": (
                    None
                    if event.rim_contrast_emergence_sample is None
                    else int(
                        source_frames[event.rim_contrast_emergence_sample]
                    )
                ),
                "rim_contrast_emergence_time_sec": (
                    None
                    if event.rim_contrast_emergence_sample is None
                    else round(
                        float(
                            source_frames[event.rim_contrast_emergence_sample]
                            - source_frames[0]
                        )
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "tortuosity": round(event.tortuosity, 3),
                "estimated_width_px": round(event.width_px, 3),
                "max_junction_turn_degrees": round(
                    event.max_junction_turn_degrees, 3
                ),
                "birth_order_correlation": round(event.birth_order_correlation, 4),
                "birth_order_forward_fraction": round(
                    event.birth_order_forward_fraction, 4
                ),
                "birth_progress_samples": round(event.birth_progress_samples, 3),
                "proximal_front_verified": event.proximal_front_verified,
                "proximal_emergence_ambiguous": (
                    event.proximal_emergence_ambiguous
                ),
                "proximal_front_reason": event.proximal_front_reason,
                "proximal_eventual_support_fraction": round(
                    event.proximal_eventual_support_fraction, 4
                ),
                "proximal_direct_support_fraction": round(
                    event.proximal_direct_support_fraction, 4
                ),
                "proximal_inferred_fraction": round(
                    event.proximal_inferred_fraction, 4
                ),
                "proximal_median_confirmation_lag_samples": round(
                    event.proximal_median_confirmation_lag_samples, 3
                ),
                "proximal_max_confirmation_lag_samples": (
                    event.proximal_max_confirmation_lag_samples
                ),
                "proximal_reference_motion_max_px": round(
                    event.proximal_reference_motion_max_px, 3
                ),
                "proximal_static_front_reason": (
                    event.proximal_static_front_reason
                ),
                "proximal_static_eventual_support_fraction": round(
                    event.proximal_static_eventual_support_fraction,
                    4,
                ),
                "proximal_static_direct_support_fraction": round(
                    event.proximal_static_direct_support_fraction,
                    4,
                ),
                "proximal_motion_binding_class": (
                    event.proximal_motion_binding_class
                ),
                "proximal_motion_support_margin": round(
                    event.proximal_motion_support_margin,
                    4,
                ),
                "proximal_rim_bridge_evaluated": (
                    event.proximal_rim_bridge_evaluated
                ),
                "proximal_rim_bridge_accepted": (
                    event.proximal_rim_bridge_accepted
                ),
                "proximal_rim_bridge_reason": (
                    event.proximal_rim_bridge_reason
                ),
                "proximal_rim_bridge_direct_support_fraction": round(
                    event.proximal_rim_bridge_direct_support_fraction,
                    4,
                ),
                "proximal_rim_bridge_eventual_support_fraction": round(
                    event.proximal_rim_bridge_eventual_support_fraction,
                    4,
                ),
                "proximal_rim_bridge_start_sample": (
                    event.proximal_rim_bridge_start_sample
                ),
                "proximal_rim_bridge_end_sample": (
                    event.proximal_rim_bridge_end_sample
                ),
                "rim_direction_specificity_evaluated": (
                    event.rim_direction_specificity_evaluated
                ),
                "rim_direction_specific": event.rim_direction_specific,
                "rim_direction_competitor_count": (
                    event.rim_direction_competitor_count
                ),
                "rim_direction_competitor_rotation_degrees": (
                    event.rim_direction_competitor_rotation_degrees
                ),
                "rim_direction_competitor_sample": (
                    event.rim_direction_competitor_sample
                ),
                "rim_direction_competitor_source_frame": (
                    None
                    if event.rim_direction_competitor_sample is None
                    else int(
                        source_frames[event.rim_direction_competitor_sample]
                    )
                ),
                "rim_direction_competitor_offset_samples": (
                    None
                    if event.rim_direction_competitor_sample is None
                    else (
                        event.rim_direction_competitor_sample
                        - event.birth_start
                    )
                ),
                "rim_direction_selected_connected_sample": (
                    event.rim_direction_selected_connected_sample
                ),
                "rim_direction_selected_connected_source_frame": (
                    None
                    if event.rim_direction_selected_connected_sample is None
                    else int(
                        source_frames[
                            event.rim_direction_selected_connected_sample
                        ]
                    )
                ),
                "rim_direction_reason": event.rim_direction_reason,
                "tip_timing_method": _tip_timing_method(event),
                "tip_bearing_sample_count": len(tip_timings),
                "accepted_tip_measurement_sample_count": sum(
                    timing.measurement_accepted for timing in tip_timings
                ),
                "front_preconfirmation_tip_sample_count": len(
                    preconfirmation_timings
                ),
                "front_preconfirmation_tip_fraction": round(
                    len(preconfirmation_timings) / max(1, len(tip_timings)),
                    4,
                ),
                "front_direct_supported_tip_sample_count": sum(
                    timing.state == "front-direct-supported"
                    for timing in preconfirmation_timings
                ),
                "front_later_supported_tip_sample_count": sum(
                    timing.state == "front-later-supported"
                    for timing in preconfirmation_timings
                ),
                "front_inferred_tip_sample_count": sum(
                    timing.state == "front-inferred"
                    for timing in preconfirmation_timings
                ),
                "front_evidence_unavailable_tip_sample_count": sum(
                    timing.state == "front-evidence-unavailable"
                    for timing in preconfirmation_timings
                ),
                "front_preconfirmation_lag_median_samples": (
                    None
                    if not preconfirmation_lags
                    else round(float(np.median(preconfirmation_lags)), 3)
                ),
                "front_preconfirmation_lag_max_samples": (
                    None
                    if not preconfirmation_lags
                    else int(max(preconfirmation_lags))
                ),
                "front_preconfirmation_lag_median_sec": (
                    None
                    if not preconfirmation_lags
                    else round(
                        float(np.median(preconfirmation_lags))
                        * actual_sample_seconds,
                        3,
                    )
                ),
                "front_preconfirmation_lag_max_sec": (
                    None
                    if not preconfirmation_lags
                    else round(
                        max(preconfirmation_lags) * actual_sample_seconds,
                        3,
                    )
                ),
                "full_path_temporal_evaluated": (
                    event.full_path_temporal_evaluated
                ),
                "full_path_temporal_verified": (
                    event.full_path_temporal_verified
                ),
                "full_path_temporal_reason": event.full_path_temporal_reason,
                "front_refinement_applied": event.front_refinement_applied,
                "front_refinement_reason": event.front_refinement_reason,
                "front_eventual_support_fraction": round(
                    event.front_eventual_support_fraction, 4
                ),
                "front_direct_support_fraction": round(
                    event.front_direct_support_fraction, 4
                ),
                "front_inferred_fraction": round(
                    event.front_inferred_fraction, 4
                ),
                "front_median_confirmation_lag_samples": round(
                    event.front_median_confirmation_lag_samples, 3
                ),
                "front_max_confirmation_lag_samples": (
                    event.front_max_confirmation_lag_samples
                ),
                "growth_step_count": event.growth_step_count,
                "tip_step_median_px": round(event.tip_step_median_px, 3),
                "tip_step_max_px": round(event.tip_step_max_px, 3),
                "length_step_max_px": round(event.length_step_max_px, 3),
                "threshold_growth_step_count": (
                    event.observed_growth_step_count
                ),
                "threshold_tip_step_median_px": round(
                    event.observed_tip_step_median_px, 3
                ),
                "threshold_tip_step_max_px": round(
                    event.observed_tip_step_max_px, 3
                ),
                "threshold_length_step_max_px": round(
                    event.observed_length_step_max_px, 3
                ),
                "radial_extension_px": (
                    None
                    if event.radial_extension_px is None
                    else round(event.radial_extension_px, 3)
                ),
                "foreign_grain_clearance_px": (
                    None
                    if event.foreign_grain_clearance_px is None
                    else round(event.foreign_grain_clearance_px, 3)
                ),
                "foreign_grain_contact_path_index": contact_index,
                "foreign_grain_contact_grain_id": (
                    event.foreign_grain_contact_grain_id
                ),
                "foreign_grain_contact_exit_path_index": (
                    event.foreign_grain_contact_exit_path_index
                ),
                "contact_bridge_next_foreign_contact_path_index": (
                    event.contact_bridge_next_foreign_contact_path_index
                ),
                "contact_bridge_hidden_length_px": (
                    None
                    if event.contact_bridge_hidden_length_px is None
                    else round(event.contact_bridge_hidden_length_px, 3)
                ),
                "contact_bridge_ingress_chord_angle_degrees": (
                    None
                    if event.contact_bridge_ingress_chord_angle_degrees is None
                    else round(
                        event.contact_bridge_ingress_chord_angle_degrees,
                        3,
                    )
                ),
                "contact_bridge_chord_egress_angle_degrees": (
                    None
                    if event.contact_bridge_chord_egress_angle_degrees is None
                    else round(
                        event.contact_bridge_chord_egress_angle_degrees,
                        3,
                    )
                ),
                "contact_bridge_total_turn_degrees": (
                    None
                    if event.contact_bridge_total_turn_degrees is None
                    else round(event.contact_bridge_total_turn_degrees, 3)
                ),
                "contact_bridge_identity_supported": (
                    event.contact_bridge_identity_supported
                ),
                "contact_bridge_viable_competitor_count": (
                    event.contact_bridge_viable_competitor_count
                ),
                "contact_bridge_last_path_index": (
                    event.contact_bridge_last_path_index
                ),
                "contact_bridge_max_length_px": (
                    None
                    if contact_bridge_max_length is None
                    else round(contact_bridge_max_length, 3)
                ),
                "contact_bridge_censor_sample": (
                    event.contact_bridge_censor_sample
                ),
                "contact_bridge_censor_source_frame": (
                    None
                    if event.contact_bridge_censor_sample is None
                    else int(
                        source_frames[event.contact_bridge_censor_sample]
                    )
                ),
                "contact_bridge_censor_time_sec": (
                    None
                    if event.contact_bridge_censor_sample is None
                    else round(
                        float(
                            source_frames[event.contact_bridge_censor_sample]
                            - source_frames[0]
                        )
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "contact_bridge_reason": event.contact_bridge_reason,
                "contact_bridge_direct_support_fraction": round(
                    event.contact_bridge_direct_support_fraction,
                    4,
                ),
                "contact_bridge_eventual_support_fraction": round(
                    event.contact_bridge_eventual_support_fraction,
                    4,
                ),
                "contact_bridge_post_direct_support_fraction": round(
                    event.contact_bridge_post_direct_support_fraction,
                    4,
                ),
                "contact_bridge_post_eventual_support_fraction": round(
                    event.contact_bridge_post_eventual_support_fraction,
                    4,
                ),
                "contact_bridge_growth_step_count": (
                    event.contact_bridge_growth_step_count
                ),
                "contact_bridge_post_growth_step_count": (
                    event.contact_bridge_post_growth_step_count
                ),
                "contact_bridge_tip_step_median_px": round(
                    event.contact_bridge_tip_step_median_px,
                    3,
                ),
                "contact_bridge_tip_step_max_px": round(
                    event.contact_bridge_tip_step_max_px,
                    3,
                ),
                "contact_bridge_length_step_max_px": round(
                    event.contact_bridge_length_step_max_px,
                    3,
                ),
                "precontact_safe_length_px": (
                    None
                    if precontact_safe_length is None
                    else round(precontact_safe_length, 3)
                ),
                "precontact_censor_sample": event.precontact_censor_sample,
                "precontact_censor_source_frame": (
                    None
                    if event.precontact_censor_sample is None
                    else int(source_frames[event.precontact_censor_sample])
                ),
                "precontact_censor_time_sec": (
                    None
                    if event.precontact_censor_sample is None
                    else round(
                        float(
                            source_frames[event.precontact_censor_sample]
                            - source_frames[0]
                        )
                        * seconds_per_source_frame,
                        3,
                    )
                ),
                "precontact_front_reason": event.precontact_front_reason,
                "precontact_direct_support_fraction": round(
                    event.precontact_direct_support_fraction,
                    4,
                ),
                "precontact_eventual_support_fraction": round(
                    event.precontact_eventual_support_fraction,
                    4,
                ),
                "precontact_growth_step_count": (
                    event.precontact_growth_step_count
                ),
                "precontact_tip_step_median_px": round(
                    event.precontact_tip_step_median_px,
                    3,
                ),
                "precontact_tip_step_max_px": round(
                    event.precontact_tip_step_max_px,
                    3,
                ),
                "precontact_length_step_max_px": round(
                    event.precontact_length_step_max_px,
                    3,
                ),
                "broad_occlusion_fraction": (
                    None
                    if event.broad_occlusion_fraction is None
                    else round(event.broad_occlusion_fraction, 4)
                ),
                "warmup_prefix_length_px": round(
                    event.warmup_prefix_length_px, 3
                ),
                "warmup_ribbon_fraction": round(
                    event.warmup_ribbon_fraction, 4
                ),
                "germination_left_censored": (
                    event.germination_left_censored
                ),
                "warmup_attachment_ambiguous": (
                    event.warmup_attachment_ambiguous
                ),
                "max_interval_for_trajectory_s": round(
                    duration_seconds / config.trajectory_resolution_samples, 3
                ),
                "cadence_verdict": (
                    "trajectory-valid"
                    if actual_sample_seconds
                    <= duration_seconds / config.trajectory_resolution_samples
                    else "atlas-only"
                ),
            }
        )

    summary_path = output_dir / "event_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summaries:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    paths_path = output_dir / "event_paths.csv"
    with paths_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_id", "path_index", "x_px", "y_px",
                "source_x_px", "source_y_px", "tip_source_x_px",
                "tip_source_y_px", "front_candidate_source_x_px",
                "front_candidate_source_y_px", "birth_sample_raw",
                "birth_sample", "tip_birth_sample",
                "front_candidate_birth_sample",
                "front_confirmation_lag_samples",
                "front_direct_support_at_arrival",
                "front_eventual_support_before_confirmation",
                "contact_bridge_front_birth_sample",
                "contact_bridge_measurement_eligible",
                "contact_bridge_front_direct_support_at_arrival",
                "contact_bridge_front_eventual_support_before_confirmation",
                "contact_bridge_phase",
                "precontact_front_birth_sample",
                "precontact_measurement_eligible",
                "precontact_front_direct_support_at_arrival",
                "precontact_front_eventual_support_before_confirmation",
                "arclength_px",
            ]
        )
        for event in events:
            candidate_birth = event.path_front_birth
            candidate_direct_support = event.path_front_direct_support
            candidate_eventual_support = event.path_front_eventual_support
            bridge_birth = event.contact_bridge_front_birth
            bridge_direct = event.contact_bridge_front_direct_support
            bridge_eventual = event.contact_bridge_front_eventual_support
            precontact_birth = event.precontact_front_birth
            precontact_direct = event.precontact_front_direct_support
            precontact_eventual = event.precontact_front_eventual_support
            contact_index = event.foreign_grain_contact_path_index
            tip_birth = _selected_tip_timing(event)
            for path_index, (point, raw_birth, birth_sample, arclength) in enumerate(
                zip(
                    event.path_xy,
                    event.path_birth_raw,
                    event.path_birth,
                    event.arclength_px,
                )
            ):
                writer.writerow(
                    [
                        event.event_id,
                        path_index,
                        round(float(point[0]), 3),
                        round(float(point[1]), 3),
                        round(float(point[0] + shifts[birth_sample, 0]), 3),
                        round(float(point[1] + shifts[birth_sample, 1]), 3),
                        round(
                            float(
                                point[0]
                                + shifts[int(tip_birth[path_index]), 0]
                            ),
                            3,
                        ),
                        round(
                            float(
                                point[1]
                                + shifts[int(tip_birth[path_index]), 1]
                            ),
                            3,
                        ),
                        (
                            ""
                            if candidate_birth is None
                            else round(
                                float(
                                    point[0]
                                    + shifts[
                                        int(candidate_birth[path_index]), 0
                                    ]
                                ),
                                3,
                            )
                        ),
                        (
                            ""
                            if candidate_birth is None
                            else round(
                                float(
                                    point[1]
                                    + shifts[
                                        int(candidate_birth[path_index]), 1
                                    ]
                                ),
                                3,
                            )
                        ),
                        int(raw_birth),
                        int(birth_sample),
                        int(tip_birth[path_index]),
                        (
                            ""
                            if candidate_birth is None
                            else int(candidate_birth[path_index])
                        ),
                        (
                            ""
                            if candidate_birth is None
                            else int(
                                birth_sample - candidate_birth[path_index]
                            )
                        ),
                        (
                            ""
                            if candidate_direct_support is None
                            else bool(candidate_direct_support[path_index])
                        ),
                        (
                            ""
                            if candidate_eventual_support is None
                            else bool(candidate_eventual_support[path_index])
                        ),
                        (
                            ""
                            if bridge_birth is None
                            or path_index >= len(bridge_birth)
                            else int(bridge_birth[path_index])
                        ),
                        bool(
                            event.contact_bridge_trajectory_accepted
                            and event.contact_bridge_last_path_index is not None
                            and path_index <= event.contact_bridge_last_path_index
                        ),
                        (
                            ""
                            if bridge_direct is None
                            or path_index >= len(bridge_direct)
                            else bool(bridge_direct[path_index])
                        ),
                        (
                            ""
                            if bridge_eventual is None
                            or path_index >= len(bridge_eventual)
                            else bool(bridge_eventual[path_index])
                        ),
                        _contact_bridge_phase(event, path_index),
                        (
                            ""
                            if precontact_birth is None
                            or path_index >= len(precontact_birth)
                            else int(precontact_birth[path_index])
                        ),
                        bool(
                            event.precontact_trajectory_accepted
                            and contact_index is not None
                            and path_index < contact_index
                        ),
                        (
                            ""
                            if precontact_direct is None
                            or path_index >= len(precontact_direct)
                            else bool(precontact_direct[path_index])
                        ),
                        (
                            ""
                            if precontact_eventual is None
                            or path_index >= len(precontact_eventual)
                            else bool(precontact_eventual[path_index])
                        ),
                        round(float(arclength), 4),
                    ]
                )

    shifts_path = output_dir / "registration_shifts.csv"
    with shifts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample", "source_frame", "shift_x_px", "shift_y_px", "response"]
        )
        for sample, source_frame in enumerate(source_frames):
            writer.writerow(
                [
                    sample,
                    int(source_frame),
                    round(float(shifts[sample, 0]), 4),
                    round(float(shifts[sample, 1]), 4),
                    round(float(responses[sample]), 5),
                ]
            )

    review_video_path = output_dir / "review.mp4"
    report = {
        "prototype": "v17_birth_topology",
        "input_movie": str(movie.resolve()),
        "fps": fps,
        "source_frame_interval_seconds": source_frame_interval_seconds,
        "time_basis": time_basis,
        "source_window": [int(source_frames[0]), int(source_frames[-1])],
        "requested_sample_interval_s": sample_seconds,
        "sample_interval_s": actual_sample_seconds,
        "sample_count": len(source_frames),
        "analysis_scale_to_native": scale_to_native,
        "measurement_calibration": {
            "pixel_size_per_native_pixel": pixel_size,
            "distance_unit": "px" if pixel_size is None else distance_unit,
        },
        "field_semantics": {
            "germination_onset": (
                "consensus optical estimate from the connected-root and "
                "pollen-rim cues; an oriented on-path-minus-flank cue may "
                "resolve timing only when it agrees with the root or rim cue"
            ),
            "germination_onset_bounds": (
                "for accepted point estimates, the span of agreeing cues; for "
                "optical-review-window events, the span of every observed root, "
                "rim, and oriented-contrast cue"
            ),
            "germination_onset_timing_scope": (
                "point-estimate is eligible for exact-time analysis; optical-"
                "review-window localizes disagreement for manual review but is "
                "not an accepted exact time or a biological confidence interval, "
                "because optical maturation can lag construction"
            ),
            "measurement_start": (
                "first sample where the connected path reaches the configured "
                "minimum germination length"
            ),
            "legacy_germination_fields": (
                "aliases for measurement_start retained for compatibility"
            ),
            "tip_coordinates": (
                "blank before measurement_start; rim change alone does not "
                "create a synthetic tip"
            ),
            "tip_timing_state": (
                "threshold-confirmed uses the persistent birth map; front-direct-"
                "supported has local image evidence at its earlier fitted arrival; "
                "front-later-supported is supported only later before threshold "
                "confirmation; front-inferred has neither local evidence source"
            ),
            "full_path_temporal_validation": (
                "every trajectory_accepted event must admit an outward-only, "
                "step-bounded temporal front with the configured direct and "
                "eventual image-support fractions; that front replaces the "
                "reported threshold timeline only when every newly exposed "
                "early tip has direct or later pre-confirmation support"
            ),
            "precontact_trajectory": (
                "a separately accepted tip timeline ending before the first "
                "foreign-pollen safety envelope; the complete candidate path "
                "remains review-only and maximum_accepted_length reports only "
                "the validated prefix"
            ),
            "contact_bridge_trajectory": (
                "a separately accepted prefix that crosses one foreign pollen "
                "only when the approach and exit are directionally continuous, "
                "no viable competing pollen root explains the far side, and "
                "the reacquired segment independently passes temporal support "
                "and tip-motion gates; later unsupported material stays censored"
            ),
            "proximal_motion_binding": (
                "compares the pollen-following proximal tube with the same "
                "candidate held static; static-only structures do not establish "
                "attachment to the moving pollen"
            ),
            "proximal_rim_bridge": (
                "after complete-path validation, a candidate blocked only by "
                "unsupported root-transition pixels may establish germination "
                "existence when independent rim and on-path-minus-flank cues "
                "agree with a pollen-following proximal front beyond the rim; "
                "this does not accept an exact onset time"
            ),
            "rim_direction_specificity": (
                "rotates the same short pollen-adjacent path into spatially "
                "independent rim sectors and requires a continuous new-evidence "
                "chain; a synchronous competing sector withholds only a short "
                "germination claim, while a validated longer path is retained "
                "with the competitor recorded for review"
            ),
            "detailed_measurement_acceptance": (
                "use germination_accepted to count germination events, "
                "germination_onset_time_accepted for onset-time analyses, "
                "trajectory_measurement_scope for full, contact-bridged, or "
                "pre-contact coverage, and tip_measurement_accepted as the definitive "
                "framewise length/tip filter; other rows remain exported for review"
            ),
        },
        "configuration": config.__dict__,
        "effective_quality_thresholds": {
            "grain_exit_margin_px": config.grain_exit_margin_px,
            "max_tip_step_median_px": config.max_tip_step_median_px,
            "max_tip_step_px": config.max_tip_step_px,
            "max_event_width_px": config.max_event_width_px,
            "junction_direction_window_px": config.junction_direction_window_px,
            "junction_turn_soft_limit_degrees": (
                config.junction_turn_soft_limit_degrees
            ),
            "junction_turn_penalty_weight": config.junction_turn_penalty_weight,
            "contact_bridge_max_turn_degrees": (
                config.contact_bridge_max_turn_degrees
            ),
            "min_radial_extension_px": config.min_radial_extension_px,
            "min_radial_extension_ratio": config.min_radial_extension_ratio,
            "foreign_grain_clearance_margin_px": (
                config.foreign_grain_clearance_margin_px
            ),
            "max_path_occluded_fraction": config.max_path_occluded_fraction,
            "warmup_review_min_prefix_px": (
                config.grain_exit_margin_px
                + config.min_germination_length_px
            ),
            "warmup_review_min_ribbon_fraction": (
                config.warmup_review_min_ribbon_fraction
            ),
            "warmup_contact_review_min_ribbon_fraction": (
                config.warmup_contact_review_min_ribbon_fraction
            ),
            "proximal_front_length_px": config.proximal_front_length_px,
            "proximal_front_max_step_px": config.max_tip_step_px,
            "rim_direction_min_rotation_degrees": (
                config.rim_direction_min_rotation_degrees
            ),
            "rim_direction_rotation_step_degrees": (
                config.rim_direction_rotation_step_degrees
            ),
            "rim_direction_normal_halfwidth_px": (
                config.rim_direction_normal_halfwidth_px
            ),
            "rim_direction_min_connected_fraction": (
                config.rim_direction_min_connected_fraction
            ),
            "rim_direction_timing_tolerance_samples": (
                config.rim_direction_timing_tolerance_samples
            ),
            "rim_direction_timing_tolerance_s": round(
                config.rim_direction_timing_tolerance_samples
                * actual_sample_seconds,
                3,
            ),
            "rim_contrast_halfwidth_px": config.rim_contrast_halfwidth_px,
            "rim_contrast_flank_offset_px": (
                config.rim_contrast_flank_offset_px
            ),
            "rim_contrast_min_delta": config.rim_contrast_min_delta,
            "max_rim_timing_error_samples": config.temporal_link_samples,
            "max_rim_timing_error_s": round(
                config.temporal_link_samples * actual_sample_seconds,
                3,
            ),
            "front_max_confirmation_lag_samples": (
                config.temporal_link_samples
            ),
            "front_max_confirmation_lag_s": round(
                config.temporal_link_samples * actual_sample_seconds,
                3,
            ),
            "front_min_eventual_support_fraction": (
                config.front_min_eventual_support_fraction
            ),
            "front_min_direct_support_fraction": (
                config.front_min_direct_support_fraction
            ),
        },
        "calibration": calibration,
        "registration": {
            "median_step_px": round(float(np.median(np.linalg.norm(np.diff(shifts, axis=0), axis=1))), 4),
            "max_shift_px": round(float(np.max(np.linalg.norm(shifts, axis=1))), 4),
            "low_confidence_fraction": round(float((responses < 0.08).mean()), 4),
        },
        "grain_count": len(grain_anchors),
        "grains": [
            {
                "grain_id": grain.grain_id,
                "center_xy": [round(float(v), 3) for v in grain.center_at(0)],
                "median_center_xy": [
                    round(float(v), 3) for v in grain.center_xy
                ],
                "radius_px": round(grain.radius_px, 3),
                "observations": grain.observations,
                "warmup_observations": grain.warmup_observations,
                "observed_sample_range": (
                    [
                        int(grain.sample_indices[0]),
                        int(grain.sample_indices[-1]),
                    ]
                    if grain.sample_indices is not None
                    and len(grain.sample_indices)
                    else [0, 0]
                ),
                "confidence": (
                    "high"
                    if (grain.warmup_observations or 0) >= 3
                    else "confirmed"
                ),
                "circle_score": round(grain.circle_score, 4),
            }
            for grain in grain_anchors
        ],
        "credible_event_count": len(events),
        "auto_accepted_event_count": sum(event.auto_accepted for event in events),
        "germination_accepted_event_count": sum(
            event.primary_for_grain and event.auto_accepted for event in events
        ),
        "germination_onset_time_accepted_event_count": sum(
            assessment.accepted
            for assessment in onset_assessment_by_event.values()
        ),
        "germination_onset_review_window_event_count": sum(
            _onset_timing_scope(event, onset_assessment_by_event[event.event_id])
            == "optical-review-window"
            for event in events
        ),
        "trajectory_accepted_event_count": sum(
            event.trajectory_accepted for event in events
        ),
        "contact_bridge_trajectory_accepted_event_count": sum(
            event.contact_bridge_trajectory_accepted for event in events
        ),
        "precontact_trajectory_accepted_event_count": sum(
            event.precontact_trajectory_accepted for event in events
        ),
        "front_candidate_event_count": sum(
            event.path_front_birth is not None for event in events
        ),
        "front_refinement_event_count": sum(
            event.front_refinement_applied for event in events
        ),
        "full_path_temporal_evaluated_event_count": sum(
            event.full_path_temporal_evaluated for event in events
        ),
        "full_path_temporal_verified_event_count": sum(
            event.full_path_temporal_verified for event in events
        ),
        "left_censored_event_count": sum(
            event.germination_left_censored for event in events
        ),
        "warmup_ambiguous_event_count": sum(
            event.warmup_attachment_ambiguous for event in events
        ),
        "proximal_ambiguous_event_count": sum(
            event.proximal_emergence_ambiguous for event in events
        ),
        "proximal_rim_bridge_accepted_event_count": sum(
            event.proximal_rim_bridge_accepted for event in events
        ),
        "rim_direction_ambiguous_event_count": sum(
            event.rim_direction_specificity_evaluated
            and event.rim_direction_specific is False
            for event in events
        ),
        "rim_direction_rejected_event_count": sum(
            "non-specific-rim-change" in event.quality_flags
            for event in events
        ),
        "proximal_motion_binding_counts": {
            binding_class: sum(
                event.proximal_motion_binding_class == binding_class
                for event in events
            )
            for binding_class in sorted(
                {
                    event.proximal_motion_binding_class
                    for event in events
                    if event.proximal_motion_binding_class != "not-evaluated"
                }
            )
        },
        "primary_event_count": sum(event.primary_for_grain for event in events),
        "accepted_primary_event_count": sum(
            event.primary_for_grain and event.auto_accepted for event in events
        ),
        "trajectory_primary_event_count": sum(
            event.primary_for_grain and event.trajectory_accepted for event in events
        ),
        "measurable_trajectory_primary_event_count": sum(
            event.primary_for_grain
            and (
                event.trajectory_accepted
                or event.contact_bridge_trajectory_accepted
                or event.precontact_trajectory_accepted
            )
            for event in events
        ),
        "multi_root_component_count": len(
            {
                event.component_id
                for event in events
                if event.component_root_count > 1
            }
        ),
        "multi_root_event_count": sum(
            event.component_root_count > 1 for event in events
        ),
        "ambiguous_component_ownership_count": sum(
            "ambiguous-component-ownership" in event.quality_flags
            for event in events
        ),
        "rendered_event_count": len(selected),
        "events": summaries,
        "artifacts": {
            "measurements_csv": str(detailed_path),
            "summary_csv": str(summary_path),
            "paths_csv": str(paths_path),
            "registration_shifts_csv": str(shifts_path),
            "birth_map": str(output_dir / "birth_time_map.jpg"),
            "review_video": str(review_video_path) if write_review_video else None,
        },
    }
    grain_csv, grain_image = write_grain_outputs(
        output_dir, aligned[0], grain_anchors
    )
    accepted_image = output_dir / "accepted_events.jpg"
    review_image = output_dir / "review_required_events.jpg"
    timeline_image = output_dir / "germination_review_timelines.jpg"
    proximal_timeline_image = output_dir / "proximal_emergence_review.jpg"
    rim_direction_timeline_image = (
        output_dir / "rim_direction_specificity_review.jpg"
    )
    trajectory_timeline_image = output_dir / "trajectory_review_timelines.jpg"
    validation_image = output_dir / "blinded_validation_timelines.jpg"
    validation_manifest = output_dir / "blinded_validation_manifest.csv"
    front_timeline_image = output_dir / "front_refinement_timelines.jpg"
    write_event_audit_image(accepted_image, aligned[-1], events, accepted=True)
    write_event_audit_image(review_image, aligned[-1], events, accepted=False)
    timeline_count = write_event_timeline_audit(
        timeline_image,
        aligned,
        source_frames,
        events,
        config,
    )
    proximal_timeline_count = write_proximal_emergence_audit(
        proximal_timeline_image,
        aligned,
        source_frames,
        events,
        config,
        grain_anchors=grain_anchors,
    )
    rim_direction_timeline_count = write_rim_direction_specificity_audit(
        rim_direction_timeline_image,
        aligned,
        source_frames,
        events,
        config,
        grain_anchors=grain_anchors,
    )
    trajectory_timeline_count = write_trajectory_timeline_audit(
        trajectory_timeline_image,
        aligned,
        source_frames,
        events,
    )
    validation_counts = write_blinded_validation_audit(
        validation_image,
        validation_manifest,
        aligned,
        source_frames,
        events,
        grain_anchors,
        config,
    )
    front_timeline_count = write_front_refinement_audit(
        front_timeline_image,
        aligned,
        source_frames,
        events,
    )
    report["germination_review_timeline_count"] = timeline_count
    report["proximal_emergence_review_count"] = proximal_timeline_count
    report["rim_direction_specificity_review_count"] = (
        rim_direction_timeline_count
    )
    report["trajectory_review_timeline_count"] = trajectory_timeline_count
    report["blinded_validation_case_counts"] = validation_counts
    report["front_refinement_timeline_count"] = front_timeline_count
    report["artifacts"]["grain_catalog_csv"] = str(grain_csv)
    report["artifacts"]["grain_audit_image"] = str(grain_image)
    report["artifacts"]["accepted_event_audit_image"] = str(accepted_image)
    report["artifacts"]["review_event_audit_image"] = str(review_image)
    report["artifacts"]["germination_review_timeline"] = str(timeline_image)
    report["artifacts"]["proximal_emergence_review"] = str(
        proximal_timeline_image
    )
    report["artifacts"]["rim_direction_specificity_review"] = str(
        rim_direction_timeline_image
    )
    report["artifacts"]["trajectory_review_timeline"] = str(
        trajectory_timeline_image
    )
    report["artifacts"]["blinded_validation_timeline"] = str(
        validation_image
    )
    report["artifacts"]["blinded_validation_manifest"] = str(
        validation_manifest
    )
    report["artifacts"]["front_refinement_timeline"] = str(
        front_timeline_image
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    write_birth_map(
        output_dir / "birth_time_map.jpg",
        aligned[-1],
        birth,
        config.warmup_samples,
    )

    if not write_review_video:
        review_video_path.unlink(missing_ok=True)
        return

    h, w = aligned.shape[1:]
    writer = cv.VideoWriter(
        str(review_video_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (w, h),
    )
    palette = [
        (42, 42, 230), (230, 130, 30), (35, 190, 90), (200, 60, 180),
        (30, 190, 220), (220, 210, 50),
    ]
    for sample, gray in enumerate(aligned):
        frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for index, event in enumerate(selected):
            if (
                event.contact_bridge_trajectory_accepted
                and event.contact_bridge_censor_sample is not None
                and sample >= event.contact_bridge_censor_sample
                and event.contact_bridge_last_path_index is not None
            ):
                first_withheld = min(
                    event.contact_bridge_last_path_index + 1,
                    len(event.path_xy) - 1,
                )
                contact = tuple(
                    np.rint(event.path_xy[first_withheld]).astype(int)
                )
                cv.drawMarker(
                    frame,
                    contact,
                    (0, 150, 255),
                    cv.MARKER_TILTED_CROSS,
                    8,
                    1,
                    cv.LINE_AA,
                )
                continue
            if (
                not event.contact_bridge_trajectory_accepted
                and event.precontact_trajectory_accepted
                and event.precontact_censor_sample is not None
                and sample >= event.precontact_censor_sample
                and event.foreign_grain_contact_path_index is not None
            ):
                contact = tuple(
                    np.rint(
                        event.path_xy[event.foreign_grain_contact_path_index]
                    ).astype(int)
                )
                cv.drawMarker(
                    frame,
                    contact,
                    (0, 150, 255),
                    cv.MARKER_TILTED_CROSS,
                    8,
                    1,
                    cv.LINE_AA,
                )
                continue
            _, tip, path_index = event_tip_at(event, sample)
            if tip is None:
                continue
            points = np.rint(event.path_xy[: path_index + 1]).astype(np.int32)
            color = palette[index % len(palette)]
            if len(points) >= 2:
                cv.polylines(frame, [points], False, color, 2, cv.LINE_AA)
            cv.circle(frame, tuple(np.rint(tip).astype(int)), 3, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(
            frame,
            f"frame {int(source_frames[sample])}",
            (10, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible prototype run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-end", type=int)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument(
        "--source-frame-interval-seconds",
        type=float,
        help=(
            "Experimental time represented by one encoded source frame. "
            "Without this, time uses MP4 playback metadata."
        ),
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Physical distance represented by one native video pixel.",
    )
    parser.add_argument("--distance-unit", default="um")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument(
        "--warmup-samples",
        type=int,
        help=(
            "Override the time-scaled warmup length in analysis samples."
        ),
    )
    parser.add_argument(
        "--temporal-link-samples",
        type=int,
        help=(
            "Override the time-scaled topology horizon in analysis samples."
        ),
    )
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument(
        "--render-events",
        default="",
        help="comma-separated event IDs to render instead of the top-ranked events",
    )
    parser.add_argument(
        "--no-review-video",
        action="store_true",
        help="write CSV, JSON, and audit images without rendering an MP4",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run birth detection, topology tracing, and artifact generation."""

    args = parse_args()
    if args.width <= 0:
        raise ValueError("width must be positive")
    if args.pixel_size is not None and args.pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    if not args.distance_unit.strip():
        raise ValueError("distance_unit cannot be empty")
    fps, frame_count, _, _ = movie_metadata(args.movie)
    source_frames = source_frame_indices(
        frame_count,
        fps,
        args.source_start,
        args.source_end,
        args.sample_seconds,
        args.source_frame_interval_seconds,
    )
    config = AtlasConfig.for_width(args.width).for_sample_interval(
        args.sample_seconds
    )
    explicit_temporal_settings = {}
    if args.warmup_samples is not None:
        explicit_temporal_settings["warmup_samples"] = args.warmup_samples
    if args.temporal_link_samples is not None:
        explicit_temporal_settings["temporal_link_samples"] = (
            args.temporal_link_samples
        )
    if explicit_temporal_settings:
        config = replace(config, **explicit_temporal_settings)
    print(f"[v17] reading {len(source_frames)} samples", flush=True)
    frames = load_movie_samples(args.movie, source_frames, config.width)
    print("[v17] stabilizing common field motion", flush=True)
    aligned, shifts, responses = stabilize_translations(frames)
    del frames
    print("[v17] measuring locally persistent new material", flush=True)
    grain_anchors = detect_grain_anchors(
        aligned, config.warmup_samples, config
    )
    print(f"[v17] {len(grain_anchors)} persistent pollen anchors", flush=True)
    evidence = ridge_evidence(aligned)
    birth, preexisting, calibration = local_persistent_births(
        evidence,
        config.warmup_samples,
        config.persistence_window,
        config.persistence_required,
        config.preexisting_min_observations,
        config.preexisting_dilation_px,
        grain_anchors,
        config.preexisting_grain_protection_px,
    )
    print("[v17] recovering pollen-anchored simple paths", flush=True)
    events = extract_birth_events(
        birth,
        preexisting,
        config,
        grain_anchors,
        evidence,
        aligned,
    )
    del evidence
    print(f"[v17] {len(events)} credible events", flush=True)
    render_event_ids = (
        [int(value) for value in args.render_events.split(",") if value.strip()]
        if args.render_events
        else None
    )
    write_outputs(
        args.output_dir,
        args.movie,
        aligned,
        source_frames,
        fps,
        args.sample_seconds,
        args.source_frame_interval_seconds,
        args.pixel_size,
        args.distance_unit,
        birth,
        events,
        calibration,
        shifts,
        responses,
        config,
        grain_anchors,
        args.max_events,
        render_event_ids,
        not args.no_review_video,
    )
    print(f"[v17] outputs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
