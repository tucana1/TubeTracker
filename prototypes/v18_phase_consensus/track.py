#!/usr/bin/env python3
"""Cross-validate pollen germination and tip paths across sampling phases.

The v17 birth map is deliberately conservative, but H71 showed that its branch
ranking can change when a sparse sampling grid is shifted. This prototype runs
two denser, interleaved phases. A path proposed in one phase is kept fixed and
tested against independent birth order, path evidence, and pollen-rim cues in
the other phase. Competing paths are retained, and incompatible paths from the
same pollen are never silently resolved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototypes.v17_birth_topology.track import (
    AtlasConfig,
    BirthEvent,
    GrainAnchor,
    OnsetAssessment,
    _birth_order_metrics,
    _germination_onset_assessment,
    _global_evidence_floor,
    _oriented_rim_contrast_metrics,
    _rim_emergence_metrics,
    _selected_tip_timing,
    _tip_dynamics_metrics,
    _tip_measurement_is_accepted,
    _trajectory_measurement_scope,
    detect_grain_anchors,
    extract_birth_events,
    load_movie_samples,
    local_persistent_births,
    movie_metadata,
    ridge_evidence,
    sample_interval_seconds,
    source_frame_indices,
    stabilize_translations,
)
from tubetracker.curve_prototype import ordered_prefix_retention
from tubetracker.growth_front import (
    isotonic_increasing,
    path_novelty_profiles,
    visible_path_front,
)


@dataclass
class PhaseAnalysis:
    """Hold one complete v17 analysis phase in memory."""

    name: str
    source_frames: np.ndarray
    aligned: np.ndarray
    shifts: np.ndarray
    responses: np.ndarray
    grains: list[GrainAnchor]
    evidence: np.ndarray
    birth: np.ndarray
    events: list[BirthEvent]
    calibration: dict[str, float]


@dataclass(frozen=True)
class PhaseConsensusConfig:
    """Define auditable agreement gates between interleaved analyses."""

    tip_timing_tolerance_intervals: float
    min_tip_timing_agreement_fraction: float = 0.8
    onset_timing_tolerance_intervals: float = 1.0
    tip_length_tolerance_px: float = 1.5
    max_tip_length_disagreement_px: float = 3.75
    min_tip_length_agreement_fraction: float = 0.8
    path_registration_mode: str = "branch-locked"
    grain_motion_falloff_px: float = 12.0
    normal_search_radius_px: float = 3.0
    normal_search_step_px: float = 1.0
    registration_spatial_window_points: int = 7
    registration_temporal_window_samples: int = 7
    registration_iterations: int = 3
    registration_prior_weight: float = 0.35
    registration_birth_lead_samples: int = 8
    registration_root_lock_points: int = 3
    registration_max_normal_step_px: float = 1.0
    visible_front_coverage_cost: float = 0.05


@dataclass(frozen=True)
class CrossPhaseAssessment:
    """Describe independent support for one fixed path in another phase."""

    target_grain_id: int | None
    grain_match_distance_px: float | None
    path_birth_coverage: float
    eventual_path_support_fraction: float
    birth_order_correlation: float
    birth_order_forward_fraction: float
    birth_progress_samples: float
    growth_step_count: int
    tip_step_median_px: float
    tip_step_max_px: float
    path_supported: bool
    trajectory_supported: bool
    onset_supported: bool
    onset_quality: str
    onset_sample: int | None
    onset_lower_sample: int | None
    onset_upper_sample: int | None
    reason: str
    root_birth_sample: int | None = None
    rim_emergence_sample: int | None = None
    rim_contrast_emergence_sample: int | None = None
    germination_sample: int | None = None
    path_reference_sample: int | None = None
    tip_birth_samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        repr=False,
        compare=False,
    )


@dataclass
class ConsensusCandidate:
    """Pair one source-phase event with its independent phase assessment."""

    source_phase: str
    event: BirthEvent
    source_grain: GrainAnchor
    target_phase: PhaseAnalysis
    target_grain: GrainAnchor | None
    cross: CrossPhaseAssessment
    canonical_grain_id: int
    path_reference_xy: np.ndarray
    source_onset: OnsetAssessment
    source_analysis: PhaseAnalysis | None = None
    path_reference_phase: PhaseAnalysis | None = None
    path_reference_offsets_xy: np.ndarray | None = None
    source_tip_birth_source_frames: np.ndarray | None = None
    target_tip_birth_source_frames: np.ndarray | None = None
    consensus_tip_birth_source_frames: np.ndarray | None = None
    tip_timing_agreement_fraction: float = 0.0
    tip_timing_disagreement_median_s: float | None = None
    tip_timing_disagreement_max_s: float | None = None
    tip_timing_supported: bool = False
    tip_length_agreement_fraction: float = 0.0
    tip_length_disagreement_median_px: float | None = None
    tip_length_disagreement_p90_px: float | None = None
    tip_length_disagreement_max_px: float | None = None
    tip_length_supported: bool = False
    source_front_supported: bool = False
    target_front_supported: bool = False
    source_front_reason: str = "not-evaluated"
    target_front_reason: str = "not-evaluated"
    source_front_direct_support_fraction: float = 0.0
    target_front_direct_support_fraction: float = 0.0
    source_front_eventual_support_fraction: float = 0.0
    target_front_eventual_support_fraction: float = 0.0
    geometry_conflict: bool = False
    selected: bool = False
    consensus_id: int | None = None
    germination_accepted: bool = False
    onset_time_accepted: bool = False
    measurement_supported: bool = False
    trajectory_accepted: bool = False
    onset_source_frame: int | None = None
    onset_lower_source_frame: int | None = None
    onset_upper_source_frame: int | None = None
    measurable_growth_onset_supported: bool = False
    measurable_growth_onset_source_frame: int | None = None
    measurable_growth_onset_lower_source_frame: int | None = None
    measurable_growth_onset_upper_source_frame: int | None = None
    measurable_growth_onset_uncertainty_s: float | None = None


def analyze_phase(
    name: str,
    movie: Path,
    source_frames: np.ndarray,
    config: AtlasConfig,
) -> PhaseAnalysis:
    """Run one v17 phase without writing duplicate intermediate artifacts."""

    print(f"[v18] phase {name}: reading {len(source_frames)} samples", flush=True)
    frames = load_movie_samples(movie, source_frames, config.width)
    aligned, shifts, responses = stabilize_translations(frames)
    del frames
    grains = detect_grain_anchors(aligned, config.warmup_samples, config)
    print(f"[v18] phase {name}: {len(grains)} pollen anchors", flush=True)
    evidence = ridge_evidence(aligned)
    birth, preexisting, calibration = local_persistent_births(
        evidence,
        config.warmup_samples,
        config.persistence_window,
        config.persistence_required,
        config.preexisting_min_observations,
        config.preexisting_dilation_px,
        grains,
        config.preexisting_grain_protection_px,
    )
    events = extract_birth_events(
        birth,
        preexisting,
        config,
        grains,
        evidence,
        aligned,
    )
    print(f"[v18] phase {name}: {len(events)} path hypotheses", flush=True)
    return PhaseAnalysis(
        name=name,
        source_frames=source_frames,
        aligned=aligned,
        shifts=shifts,
        responses=responses,
        grains=grains,
        evidence=evidence,
        birth=birth,
        events=events,
        calibration=calibration,
    )


def nearest_grain(
    source: GrainAnchor,
    targets: list[GrainAnchor],
    maximum_distance_px: float,
) -> tuple[GrainAnchor | None, float | None]:
    """Find the closest alternate-phase pollen without forcing one-to-one IDs."""

    if not targets:
        return None, None
    distances = np.asarray(
        [np.linalg.norm(target.center_xy - source.center_xy) for target in targets],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > maximum_distance_px:
        return None, distance
    return targets[index], distance


def local_path_birth_observations(
    birth: np.ndarray,
    path_xy: np.ndarray,
    warmup_samples: int,
    neighborhood_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample robust local birth times around every fixed path point."""

    total = int(np.max(birth))
    observed = np.full(len(path_xy), np.nan, dtype=np.float64)
    present = np.zeros(len(path_xy), dtype=bool)
    height, width = birth.shape
    for index, (x, y) in enumerate(np.rint(path_xy).astype(np.int32)):
        x0 = max(0, int(x) - neighborhood_px)
        x1 = min(width, int(x) + neighborhood_px + 1)
        y0 = max(0, int(y) - neighborhood_px)
        y1 = min(height, int(y) + neighborhood_px + 1)
        values = birth[y0:y1, x0:x1]
        values = values[(values > warmup_samples) & (values < total)]
        if len(values):
            observed[index] = float(np.median(values))
            present[index] = True
    return observed, present


def grain_relative_path_offsets(
    grain: GrainAnchor,
    sample_count: int,
    reference_sample: int,
) -> np.ndarray:
    """Translate a rigid tube path with its pollen's measured field motion."""

    reference = grain.center_at(reference_sample)
    return np.asarray(
        [grain.center_at(sample) - reference for sample in range(sample_count)],
        dtype=np.float64,
    )


def phase_evidence_floor(phase: PhaseAnalysis, config: AtlasConfig) -> float:
    """Reuse one phase's calibrated ridge floor without recomputing its median."""

    calibrated = phase.calibration.get("global_evidence_floor")
    if calibrated is not None:
        return float(calibrated)
    return _global_evidence_floor(phase.evidence, config.warmup_samples)


def branch_locked_path_offsets(
    phase: PhaseAnalysis,
    path_xy: np.ndarray,
    grain: GrainAnchor,
    reference_sample: int,
    preliminary_birth_samples: np.ndarray,
    config: AtlasConfig,
    consensus_config: PhaseConsensusConfig,
) -> np.ndarray:
    """Register one fixed branch with smooth, pollen-anchored deformation.

    The pollen's displacement is applied at the root and fades along the
    proximal shank. Interior nodes may move only along the original path's
    normal inside a narrow corridor. Median regularization across time and
    arclength suppresses isolated foreign crossings without changing path
    order, length coordinates, or branch identity.
    """

    path = np.asarray(path_xy, dtype=np.float64)
    births = np.asarray(preliminary_birth_samples, dtype=np.int32)
    sample_count = len(phase.evidence)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_xy must contain at least two x/y points")
    if births.shape != (len(path),):
        raise ValueError("preliminary births must match the path")

    rigid_offsets = grain_relative_path_offsets(
        grain,
        sample_count,
        reference_sample,
    )
    mode = consensus_config.path_registration_mode
    if mode == "fixed":
        return np.zeros((sample_count, len(path), 2), dtype=np.float64)
    if mode == "rigid":
        return np.broadcast_to(
            rigid_offsets[:, None, :],
            (sample_count, len(path), 2),
        ).copy()
    if mode != "branch-locked":
        raise ValueError(f"unsupported path registration mode: {mode}")

    arclength = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    falloff = max(float(consensus_config.grain_motion_falloff_px), 1e-6)
    root_influence = np.clip(1.0 - arclength / falloff, 0.0, 1.0)
    root_influence = root_influence * root_influence * (3.0 - 2.0 * root_influence)
    base_offsets = rigid_offsets[:, None, :] * root_influence[None, :, None]

    tangents = np.gradient(path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    radius = max(float(consensus_config.normal_search_radius_px), 0.0)
    step = max(float(consensus_config.normal_search_step_px), 1e-6)
    normal_states = np.arange(-radius, radius + 0.5 * step, step)
    if not np.any(np.isclose(normal_states, 0.0)):
        normal_states = np.sort(np.append(normal_states, 0.0))

    coordinates = (
        path[None, :, None, :]
        + base_offsets[:, :, None, :]
        + normals[None, :, None, :] * normal_states[None, None, :, None]
    )
    x = np.clip(
        np.rint(coordinates[..., 0]).astype(np.int32),
        0,
        phase.evidence.shape[2] - 1,
    )
    y = np.clip(
        np.rint(coordinates[..., 1]).astype(np.int32),
        0,
        phase.evidence.shape[1] - 1,
    )
    time = np.arange(sample_count, dtype=np.int32)[:, None, None]
    sampled = phase.evidence[time, y, x].astype(np.float32)
    low = np.min(sampled, axis=2)
    spread = np.max(sampled, axis=2) - low
    normalized = (sampled - low[:, :, None]) / np.maximum(
        spread[:, :, None],
        1.0,
    )
    evidence_scale = max(
        phase_evidence_floor(phase, config),
        config.rim_min_delta,
        1.0,
    )
    authority = np.clip(spread / evidence_scale, 0.0, 1.0)

    births = np.maximum.accumulate(
        np.clip(births, config.warmup_samples, sample_count - 1)
    )
    active_after = births - max(
        0,
        int(consensus_config.registration_birth_lead_samples),
    )
    active = (
        np.arange(sample_count, dtype=np.int32)[:, None]
        >= active_after[None, :]
    )
    active[: config.warmup_samples] = False
    prior_weight = max(float(consensus_config.registration_prior_weight), 0.0)
    state_scores = authority[:, :, None] * normalized
    state_scores -= prior_weight * (
        np.abs(normal_states)[None, None, :] / max(radius, step)
    )
    selected = np.argmax(state_scores, axis=2)
    normal_offsets = normal_states[selected]
    normal_offsets[~active] = 0.0

    root_lock = min(
        max(1, int(consensus_config.registration_root_lock_points)),
        len(path),
    )
    spatial_window = max(
        1,
        int(consensus_config.registration_spatial_window_points),
    )
    temporal_window = max(
        1,
        int(consensus_config.registration_temporal_window_samples),
    )
    if spatial_window % 2 == 0:
        spatial_window += 1
    if temporal_window % 2 == 0:
        temporal_window += 1
    for _ in range(max(1, int(consensus_config.registration_iterations))):
        prior = median_filter(
            normal_offsets,
            size=(temporal_window, spatial_window),
            mode="nearest",
        )
        scores = authority[:, :, None] * normalized
        scores -= prior_weight * (
            np.abs(normal_states[None, None, :] - prior[:, :, None])
            / max(radius, step)
        )
        normal_offsets = normal_states[np.argmax(scores, axis=2)]
        for sample in range(sample_count):
            active_count = int(np.count_nonzero(active[sample]))
            if active_count == 0:
                normal_offsets[sample] = 0.0
            elif active_count < len(path):
                normal_offsets[sample, active_count:] = normal_offsets[
                    sample,
                    active_count - 1,
                ]
        normal_offsets[:, :root_lock] = 0.0

    maximum_step = max(
        float(consensus_config.registration_max_normal_step_px),
        0.0,
    )
    if maximum_step > 0.0:
        for sample in range(1, sample_count):
            normal_offsets[sample] = np.clip(
                normal_offsets[sample],
                normal_offsets[sample - 1] - maximum_step,
                normal_offsets[sample - 1] + maximum_step,
            )
        for sample in range(sample_count - 2, -1, -1):
            normal_offsets[sample] = np.clip(
                normal_offsets[sample],
                normal_offsets[sample + 1] - maximum_step,
                normal_offsets[sample + 1] + maximum_step,
            )
    normal_offsets[:, :root_lock] = 0.0
    return base_offsets + normal_offsets[:, :, None] * normals[None, :, :]


def persistent_profile_births(
    profiles: np.ndarray,
    warmup_samples: int,
    window: int,
    required: int,
    preexisting_min_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find confirmation times along a moving path's normalized evidence profile."""

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (time, path points)")
    if not 1 <= required <= window <= len(profiles):
        raise ValueError("persistence window and requirement must fit the timeline")
    if not window <= warmup_samples <= len(profiles):
        raise ValueError("warmup must contain at least one persistence window")
    active = profiles >= 0.0
    preexisting = (
        np.count_nonzero(active[:warmup_samples], axis=0)
        >= preexisting_min_observations
    )
    total = len(profiles)
    births = np.full(profiles.shape[1], total, dtype=np.int32)
    rolling = np.zeros(profiles.shape[1], dtype=np.int32)
    for sample in range(total):
        rolling += active[sample]
        if sample >= window:
            rolling -= active[sample - window]
        if sample < warmup_samples:
            continue
        newly_confirmed = (
            (rolling >= required) & (births == total) & ~preexisting
        )
        births[newly_confirmed] = sample
    births[preexisting] = 0
    present = (births > warmup_samples) & (births < total)
    return births, present


def dynamic_path_birth_observations(
    phase: PhaseAnalysis,
    path_xy: np.ndarray,
    grain: GrainAnchor,
    reference_sample: int,
    preliminary_birth_samples: np.ndarray,
    config: AtlasConfig,
    consensus_config: PhaseConsensusConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure one fixed branch after registering its material points over time."""

    offsets = branch_locked_path_offsets(
        phase,
        path_xy,
        grain,
        reference_sample,
        preliminary_birth_samples,
        config,
        consensus_config,
    )
    profiles = path_novelty_profiles(
        phase.evidence,
        path_xy,
        config.warmup_samples,
        phase_evidence_floor(phase, config),
        config.front_normal_halfwidth_px,
        config.front_normal_sample_count,
        config.rim_min_delta,
        config.rim_noise_multiplier,
        config.front_temporal_median_samples,
        offsets,
    )
    births, present = persistent_profile_births(
        profiles,
        config.warmup_samples,
        config.persistence_window,
        config.persistence_required,
        config.preexisting_min_observations,
    )
    return births, present, profiles, offsets


def interpolate_monotone_births(
    observed: np.ndarray,
    present: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    """Fill missing path confirmations and enforce outward birth order."""

    if np.any(present):
        point_indices = np.arange(len(observed), dtype=np.float64)
        filled = np.interp(
            point_indices,
            point_indices[present],
            observed[present],
        )
    else:
        filled = np.full(len(observed), total_samples, dtype=np.float64)
    return np.maximum.accumulate(
        np.rint(
            isotonic_increasing(filled, np.ones(len(filled), dtype=np.float64))
        ).astype(np.int32)
    )


def cross_validate_path(
    event: BirthEvent,
    source: PhaseAnalysis,
    source_grain: GrainAnchor,
    target: PhaseAnalysis,
    target_grain: GrainAnchor | None,
    grain_match_distance_px: float | None,
    config: AtlasConfig,
    consensus_config: PhaseConsensusConfig,
) -> tuple[CrossPhaseAssessment, np.ndarray, np.ndarray, np.ndarray]:
    """Test immutable source-phase geometry against alternate-phase evidence."""

    path = np.asarray(event.path_xy, dtype=np.float64)
    if target_grain is None:
        return (
            CrossPhaseAssessment(
                target_grain_id=None,
                grain_match_distance_px=grain_match_distance_px,
                path_birth_coverage=0.0,
                eventual_path_support_fraction=0.0,
                birth_order_correlation=0.0,
                birth_order_forward_fraction=0.0,
                birth_progress_samples=0.0,
                growth_step_count=0,
                tip_step_median_px=0.0,
                tip_step_max_px=0.0,
                path_supported=False,
                trajectory_supported=False,
                onset_supported=False,
                onset_quality="unavailable",
                onset_sample=None,
                onset_lower_sample=None,
                onset_upper_sample=None,
                reason="no-alternate-phase-grain",
            ),
            path.copy(),
            np.empty((0, len(path), 2), dtype=np.float64),
            np.empty((0, len(path)), dtype=np.float32),
        )

    source_reference_sample = int(
        np.clip(event.germination_sample, 0, len(source.source_frames) - 1)
    )
    source_reference_frame = int(source.source_frames[source_reference_sample])
    target_reference_sample = _nearest_phase_sample(
        target,
        source_reference_frame,
    )
    translation = (
        target_grain.center_at(target_reference_sample)
        - source_grain.center_at(source_reference_sample)
    )
    target_path = path + translation
    preliminary_observed, preliminary_present = local_path_birth_observations(
        target.birth,
        target_path,
        config.warmup_samples,
        config.spatial_link_px,
    )
    preliminary_births = interpolate_monotone_births(
        preliminary_observed,
        preliminary_present,
        len(target.source_frames),
    )
    observed, present, profiles, path_offsets = dynamic_path_birth_observations(
        target,
        target_path,
        target_grain,
        target_reference_sample,
        preliminary_births,
        config,
        consensus_config,
    )
    coverage = float(np.mean(present)) if len(present) else 0.0
    fitted = interpolate_monotone_births(
        observed,
        present,
        len(target.source_frames),
    )
    correlation, forward, progress = _birth_order_metrics(
        fitted,
        event.arclength_px,
        tolerance_samples=max(1.0, config.persistence_required / 3.0),
    )
    growth_steps, median_step, maximum_step, _ = _tip_dynamics_metrics(
        target_path,
        fitted,
        event.arclength_px,
    )

    eventual = np.any(profiles[config.warmup_samples :] >= 0.0, axis=0)
    eventual_fraction = float(np.mean(eventual)) if len(eventual) else 0.0
    path_supported = bool(
        coverage >= config.front_min_eventual_support_fraction
        and eventual_fraction >= config.front_min_eventual_support_fraction
        and correlation >= config.min_birth_order_correlation
        and forward >= config.min_birth_order_forward_fraction
        and progress > 0.0
    )
    trajectory_supported = bool(
        path_supported
        and growth_steps >= config.min_growth_step_count
        and median_step <= config.max_tip_step_median_px
        and maximum_step <= config.max_tip_step_px
    )

    germination_index = min(
        int(
            np.searchsorted(
                event.arclength_px,
                config.min_germination_length_px,
                side="left",
            )
        ),
        len(fitted) - 1,
    )
    measurement_sample = int(
        np.clip(fitted[germination_index], 0, len(target.evidence) - 1)
    )
    root_sample = int(np.clip(fitted[0], 0, len(target.evidence) - 1))
    _, _, _, rim_confirmed, rim_sample = _rim_emergence_metrics(
        target.evidence,
        target_grain,
        target_path,
        measurement_sample,
        config,
    )
    _, _, _, contrast_confirmed, contrast_sample = (
        _oriented_rim_contrast_metrics(
            target.evidence,
            target_grain,
            target_path,
            measurement_sample,
            config,
        )
    )
    proxy = SimpleNamespace(
        primary_for_grain=True,
        auto_accepted=True,
        birth_start=root_sample,
        germination_sample=measurement_sample,
        rim_emergence_sample=rim_sample if rim_confirmed else None,
        rim_contrast_emergence_sample=(
            contrast_sample if contrast_confirmed else None
        ),
    )
    onset = _germination_onset_assessment(proxy, config)
    reasons = []
    if coverage < config.front_min_eventual_support_fraction:
        reasons.append("insufficient-alternate-birth-coverage")
    if eventual_fraction < config.front_min_eventual_support_fraction:
        reasons.append("insufficient-alternate-path-evidence")
    if correlation < config.min_birth_order_correlation:
        reasons.append("inconsistent-alternate-birth-order")
    if forward < config.min_birth_order_forward_fraction:
        reasons.append("insufficient-alternate-forward-growth")
    if progress <= 0.0:
        reasons.append("no-alternate-outward-progress")
    if path_supported and not trajectory_supported:
        reasons.append("alternate-tip-dynamics-unresolved")
    if not onset.accepted:
        reasons.append("alternate-onset-timing-review")
    return (
        CrossPhaseAssessment(
            target_grain_id=target_grain.grain_id,
            grain_match_distance_px=grain_match_distance_px,
            path_birth_coverage=coverage,
            eventual_path_support_fraction=eventual_fraction,
            birth_order_correlation=correlation,
            birth_order_forward_fraction=forward,
            birth_progress_samples=progress,
            growth_step_count=growth_steps,
            tip_step_median_px=median_step,
            tip_step_max_px=maximum_step,
            path_supported=path_supported,
            trajectory_supported=trajectory_supported,
            onset_supported=onset.accepted,
            onset_quality=onset.quality,
            onset_sample=onset.sample,
            onset_lower_sample=onset.support_start_sample,
            onset_upper_sample=onset.support_end_sample,
            reason="ok" if not reasons else ";".join(reasons),
            root_birth_sample=root_sample,
            rim_emergence_sample=(rim_sample if rim_confirmed else None),
            rim_contrast_emergence_sample=(
                contrast_sample if contrast_confirmed else None
            ),
            germination_sample=measurement_sample,
            path_reference_sample=target_reference_sample,
            tip_birth_samples=fitted,
        ),
        target_path,
        path_offsets,
        profiles,
    )


def paths_share_rooted_prefix(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    tolerance_px: float,
    minimum_fraction: float,
) -> bool:
    """Return whether two paths follow the same root-to-tip material prefix."""

    first = np.asarray(first_xy, dtype=np.float64)
    second = np.asarray(second_xy, dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        return False
    first_length = float(np.linalg.norm(np.diff(first, axis=0), axis=1).sum())
    second_length = float(np.linalg.norm(np.diff(second, axis=0), axis=1).sum())
    shorter, longer = (
        (first, second) if first_length <= second_length else (second, first)
    )
    return bool(
        ordered_prefix_retention(shorter, longer, tolerance_px)
        >= minimum_fraction
    )


def _source_seconds_per_frame(
    fps: float,
    source_frame_interval_seconds: float | None,
) -> float:
    """Return experimental or playback seconds represented by one source frame."""

    return (
        float(source_frame_interval_seconds)
        if source_frame_interval_seconds is not None
        else 1.0 / fps
    )


def _sample_source_frame(
    phase: PhaseAnalysis,
    sample: int | None,
) -> int | None:
    """Convert an optional phase sample to its encoded source frame."""

    if sample is None:
        return None
    return int(phase.source_frames[int(np.clip(sample, 0, len(phase.source_frames) - 1))])


def fit_phase_tip_front(
    arclength_px: np.ndarray,
    profiles: np.ndarray,
    config: AtlasConfig,
    consensus_config: PhaseConsensusConfig,
) -> tuple[np.ndarray, bool, str, float, float]:
    """Fit the same directly visible front estimator in either sampling phase."""

    front = visible_path_front(
        profiles,
        arclength_px,
        warmup_samples=config.warmup_samples,
        max_step_px=config.max_tip_step_px,
        coverage_cost=consensus_config.visible_front_coverage_cost,
        absence_weight=config.front_absence_weight,
        motion_penalty=config.front_motion_penalty,
        support_window_samples=config.persistence_window,
    )
    supported = bool(
        front.feasible
        and front.direct_support_fraction >= config.front_min_direct_support_fraction
        and front.eventual_support_fraction
        >= config.front_min_eventual_support_fraction
    )
    timing = (
        np.asarray(front.birth_samples, dtype=np.int32)
        if front.feasible
        else np.empty(0, dtype=np.int32)
    )
    return (
        timing,
        supported,
        front.reason,
        float(front.direct_support_fraction),
        float(front.eventual_support_fraction),
    )


def fuse_tip_birth_times(
    source_phase: PhaseAnalysis,
    source_samples: np.ndarray,
    target_phase: PhaseAnalysis,
    target_samples: np.ndarray,
    seconds_per_source_frame: float,
    tolerance_intervals: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Fuse per-point birth times and quantify phase localization agreement."""

    source_samples = np.asarray(source_samples, dtype=np.int32)
    target_samples = np.asarray(target_samples, dtype=np.int32)
    if len(source_samples) == 0 or len(source_samples) != len(target_samples):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, 0.0, float("inf"), float("inf")
    source_frames = source_phase.source_frames[
        np.clip(source_samples, 0, len(source_phase.source_frames) - 1)
    ].astype(np.int64)
    target_frames = target_phase.source_frames[
        np.clip(target_samples, 0, len(target_phase.source_frames) - 1)
    ].astype(np.int64)
    consensus_frames = np.maximum.accumulate(
        np.rint((source_frames + target_frames) / 2.0).astype(np.int64)
    )
    disagreements_s = (
        np.abs(source_frames - target_frames).astype(np.float64)
        * seconds_per_source_frame
    )
    source_interval_s = sample_interval_seconds(
        source_phase.source_frames,
        fps=1.0 / seconds_per_source_frame,
        source_frame_interval_seconds=seconds_per_source_frame,
    )
    target_interval_s = sample_interval_seconds(
        target_phase.source_frames,
        fps=1.0 / seconds_per_source_frame,
        source_frame_interval_seconds=seconds_per_source_frame,
    )
    tolerance_s = tolerance_intervals * max(source_interval_s, target_interval_s)
    agreement = float(np.mean(disagreements_s <= tolerance_s))
    return (
        source_frames,
        target_frames,
        consensus_frames,
        agreement,
        float(np.median(disagreements_s)),
        float(np.max(disagreements_s)),
    )


def compare_tip_lengths(
    arclength_px: np.ndarray,
    source_phase: PhaseAnalysis,
    source_birth_frames: np.ndarray,
    target_phase: PhaseAnalysis,
    target_birth_frames: np.ndarray,
    tolerance_px: float,
) -> tuple[float, float, float, float]:
    """Compare two phase trajectories as length error at matched times.

    A time error is misleading for a slowly growing tube because a one-pixel
    localization difference can correspond to minutes. This comparison uses
    the actual scientific observable: path arclength at the same source frame.
    """

    arclength = np.asarray(arclength_px, dtype=np.float64)
    source_births = np.asarray(source_birth_frames, dtype=np.int64)
    target_births = np.asarray(target_birth_frames, dtype=np.int64)
    if (
        len(arclength) == 0
        or source_births.shape != arclength.shape
        or target_births.shape != arclength.shape
    ):
        return 0.0, float("inf"), float("inf"), float("inf")
    start = min(int(source_births[0]), int(target_births[0]))
    stop = max(int(source_births[-1]), int(target_births[-1]))
    timeline = np.unique(
        np.concatenate((source_phase.source_frames, target_phase.source_frames))
    )
    timeline = timeline[(timeline >= start) & (timeline <= stop)]
    if not len(timeline):
        return 0.0, float("inf"), float("inf"), float("inf")

    def lengths_at(births: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(births, timeline, side="right") - 1
        return np.where(
            indices >= 0,
            arclength[np.clip(indices, 0, len(arclength) - 1)],
            0.0,
        )

    disagreement = np.abs(lengths_at(source_births) - lengths_at(target_births))
    return (
        float(np.mean(disagreement <= tolerance_px)),
        float(np.median(disagreement)),
        float(np.percentile(disagreement, 90.0)),
        float(np.max(disagreement)),
    )


def measurable_growth_onset_interval(
    arclength_px: np.ndarray,
    minimum_length_px: float,
    source_birth_frames: np.ndarray,
    target_birth_frames: np.ndarray,
    seconds_per_source_frame: float,
) -> tuple[int | None, int | None, int | None, float | None]:
    """Bound when both phases first reach the measurable-length threshold."""

    arclength = np.asarray(arclength_px, dtype=np.float64)
    source_births = np.asarray(source_birth_frames, dtype=np.int64)
    target_births = np.asarray(target_birth_frames, dtype=np.int64)
    if (
        len(arclength) == 0
        or source_births.shape != arclength.shape
        or target_births.shape != arclength.shape
    ):
        return None, None, None, None
    index = min(
        int(np.searchsorted(arclength, minimum_length_px, side="left")),
        len(arclength) - 1,
    )
    source_frame = int(source_births[index])
    target_frame = int(target_births[index])
    lower = min(source_frame, target_frame)
    upper = max(source_frame, target_frame)
    midpoint = int(round((source_frame + target_frame) / 2))
    uncertainty_s = (upper - lower) * seconds_per_source_frame
    return midpoint, lower, upper, float(uncertainty_s)


def consensus_tip_at(
    candidate: ConsensusCandidate,
    source_frame: int,
) -> tuple[float, np.ndarray | None, int]:
    """Read one tip from the fused material-construction timeline."""

    timing = candidate.consensus_tip_birth_source_frames
    if timing is None or len(timing) == 0:
        return 0.0, None, -1
    indices = np.flatnonzero(timing <= source_frame)
    if not len(indices):
        return 0.0, None, -1
    path_index = int(indices[-1])
    path = candidate_path_at(candidate, source_frame)
    return (
        float(candidate.event.arclength_px[path_index]),
        path[path_index],
        path_index,
    )


def candidate_path_at(
    candidate: ConsensusCandidate,
    source_frame: int,
) -> np.ndarray:
    """Return one candidate's branch-locked material curve at a source frame."""

    path = np.asarray(candidate.path_reference_xy, dtype=np.float64)
    offsets = getattr(candidate, "path_reference_offsets_xy", None)
    phase = getattr(candidate, "path_reference_phase", None)
    if offsets is None or phase is None or len(offsets) == 0:
        return path
    sample = _nearest_phase_sample(phase, source_frame)
    if np.asarray(offsets).shape[1:] != path.shape:
        return path
    return path + np.asarray(offsets[sample], dtype=np.float64)


def length_at_source_frame(
    arclength_px: np.ndarray,
    birth_source_frames: np.ndarray | None,
    source_frame: int,
) -> float:
    """Read a path length from one monotone source-frame birth timeline."""

    if birth_source_frames is None or len(birth_source_frames) == 0:
        return 0.0
    births = np.asarray(birth_source_frames, dtype=np.int64)
    index = int(np.searchsorted(births, source_frame, side="right") - 1)
    if index < 0:
        return 0.0
    arclength = np.asarray(arclength_px, dtype=np.float64)
    return float(arclength[min(index, len(arclength) - 1)])


def build_candidates(
    phase_a: PhaseAnalysis,
    phase_b: PhaseAnalysis,
    config: AtlasConfig,
    consensus_config: PhaseConsensusConfig,
    seconds_per_source_frame: float,
) -> list[ConsensusCandidate]:
    """Cross-validate primary events and select one non-conflicting path per pollen."""

    candidates: list[ConsensusCandidate] = []
    grains_a = {grain.grain_id: grain for grain in phase_a.grains}
    grains_b = {grain.grain_id: grain for grain in phase_b.grains}
    for source, target in ((phase_a, phase_b), (phase_b, phase_a)):
        source_grains = grains_a if source is phase_a else grains_b
        for event in source.events:
            scope = _trajectory_measurement_scope(event)
            if not event.primary_for_grain or not (event.auto_accepted or scope != "none"):
                continue
            source_grain = source_grains.get(event.grain_id)
            if source_grain is None:
                continue
            target_grain, distance = nearest_grain(
                source_grain,
                target.grains,
                config.grain_anchor_match_distance_px,
            )
            (
                cross,
                target_path,
                target_path_offsets,
                target_profiles,
            ) = cross_validate_path(
                event,
                source,
                source_grain,
                target,
                target_grain,
                distance,
                config,
                consensus_config,
            )
            if source is phase_a:
                canonical_id = source_grain.grain_id
                path_reference = np.asarray(event.path_xy, dtype=np.float64)
            elif target_grain is not None:
                canonical_id = target_grain.grain_id
                path_reference = target_path
            else:
                canonical_id = -(source_grain.grain_id + 1)
                path_reference = np.asarray(event.path_xy, dtype=np.float64)
            (
                _,
                _,
                source_profiles,
                source_path_offsets,
            ) = dynamic_path_birth_observations(
                source,
                event.path_xy,
                source_grain,
                event.germination_sample,
                event.path_birth,
                config,
                consensus_config,
            )
            if source is phase_a or target_grain is None:
                path_reference_phase = source
                path_reference_offsets = source_path_offsets
            else:
                path_reference_phase = target
                path_reference_offsets = target_path_offsets
            (
                source_front_timing,
                source_front_supported,
                source_front_reason,
                source_front_direct,
                source_front_eventual,
            ) = fit_phase_tip_front(
                event.arclength_px,
                source_profiles,
                config,
                consensus_config,
            )
            if not len(target_profiles):
                target_front_timing = np.empty(0, dtype=np.int32)
                target_front_supported = False
                target_front_reason = "unavailable-alternate-front"
                target_front_direct = 0.0
                target_front_eventual = 0.0
            else:
                (
                    target_front_timing,
                    target_front_supported,
                    target_front_reason,
                    target_front_direct,
                    target_front_eventual,
                ) = fit_phase_tip_front(
                    event.arclength_px,
                    target_profiles,
                    config,
                    consensus_config,
                )
            (
                source_tip_frames,
                target_tip_frames,
                consensus_tip_frames,
                timing_agreement,
                timing_median_s,
                timing_max_s,
            ) = fuse_tip_birth_times(
                source,
                source_front_timing,
                target,
                target_front_timing,
                seconds_per_source_frame,
                consensus_config.tip_timing_tolerance_intervals,
            )
            (
                length_agreement,
                length_median_px,
                length_p90_px,
                length_max_px,
            ) = compare_tip_lengths(
                event.arclength_px,
                source,
                source_tip_frames,
                target,
                target_tip_frames,
                consensus_config.tip_length_tolerance_px,
            )
            (
                measurable_consensus_frame,
                measurable_lower_frame,
                measurable_upper_frame,
                measurable_uncertainty_s,
            ) = measurable_growth_onset_interval(
                event.arclength_px,
                config.min_germination_length_px,
                source_tip_frames,
                target_tip_frames,
                seconds_per_source_frame,
            )
            candidates.append(
                ConsensusCandidate(
                    source_phase=source.name,
                    event=event,
                    source_grain=source_grain,
                    target_phase=target,
                    target_grain=target_grain,
                    cross=cross,
                    canonical_grain_id=canonical_id,
                    path_reference_xy=path_reference,
                    source_onset=_germination_onset_assessment(event, config),
                    source_analysis=source,
                    path_reference_phase=path_reference_phase,
                    path_reference_offsets_xy=path_reference_offsets,
                    source_tip_birth_source_frames=source_tip_frames,
                    target_tip_birth_source_frames=target_tip_frames,
                    consensus_tip_birth_source_frames=consensus_tip_frames,
                    tip_timing_agreement_fraction=timing_agreement,
                    tip_timing_disagreement_median_s=timing_median_s,
                    tip_timing_disagreement_max_s=timing_max_s,
                    tip_timing_supported=(
                        source_front_supported
                        and target_front_supported
                        and timing_agreement
                        >= consensus_config.min_tip_timing_agreement_fraction
                    ),
                    tip_length_agreement_fraction=length_agreement,
                    tip_length_disagreement_median_px=length_median_px,
                    tip_length_disagreement_p90_px=length_p90_px,
                    tip_length_disagreement_max_px=length_max_px,
                    tip_length_supported=(
                        source_front_supported
                        and target_front_supported
                        and length_agreement
                        >= consensus_config.min_tip_length_agreement_fraction
                        and length_max_px
                        <= consensus_config.max_tip_length_disagreement_px
                    ),
                    source_front_supported=source_front_supported,
                    target_front_supported=target_front_supported,
                    source_front_reason=source_front_reason,
                    target_front_reason=target_front_reason,
                    source_front_direct_support_fraction=source_front_direct,
                    target_front_direct_support_fraction=target_front_direct,
                    source_front_eventual_support_fraction=source_front_eventual,
                    target_front_eventual_support_fraction=target_front_eventual,
                    measurable_growth_onset_source_frame=(
                        measurable_consensus_frame
                    ),
                    measurable_growth_onset_lower_source_frame=(
                        measurable_lower_frame
                    ),
                    measurable_growth_onset_upper_source_frame=(
                        measurable_upper_frame
                    ),
                    measurable_growth_onset_uncertainty_s=(
                        measurable_uncertainty_s
                    ),
                )
            )

    by_grain: dict[int, list[ConsensusCandidate]] = {}
    for candidate in candidates:
        by_grain.setdefault(candidate.canonical_grain_id, []).append(candidate)

    consensus_index = 0
    onset_tolerance_seconds = (
        consensus_config.onset_timing_tolerance_intervals
        * sample_interval_seconds(
            phase_a.source_frames,
            fps=1.0 / seconds_per_source_frame,
            source_frame_interval_seconds=seconds_per_source_frame,
        )
    )
    for group in by_grain.values():
        eligible = [
            candidate
            for candidate in group
            if candidate.cross.path_supported
            and (
                candidate.event.auto_accepted
                or _trajectory_measurement_scope(candidate.event) != "none"
            )
        ]
        conflict = any(
            not paths_share_rooted_prefix(
                first.path_reference_xy,
                second.path_reference_xy,
                config.nominal_tube_width_px,
                config.front_min_eventual_support_fraction,
            )
            for index, first in enumerate(eligible)
            for second in eligible[index + 1 :]
        )
        for candidate in group:
            candidate.geometry_conflict = conflict
        if not eligible:
            continue
        winner = max(
            eligible,
            key=lambda candidate: (
                candidate.event.auto_accepted
                and candidate.cross.onset_supported,
                candidate.source_onset.accepted
                and candidate.cross.onset_supported,
                candidate.cross.onset_supported,
                candidate.cross.trajectory_supported,
                _trajectory_measurement_scope(candidate.event) != "none",
                candidate.source_front_supported,
                candidate.target_front_supported,
                candidate.tip_length_supported,
                candidate.tip_length_agreement_fraction,
                candidate.tip_timing_agreement_fraction,
                candidate.event.auto_accepted,
                candidate.event.score,
            ),
        )
        consensus_index += 1
        winner.selected = True
        winner.consensus_id = consensus_index
        winner.germination_accepted = bool(
            winner.event.auto_accepted
            and winner.cross.path_supported
            and winner.cross.onset_supported
            and not conflict
        )
        winner.measurement_supported = bool(
            _trajectory_measurement_scope(winner.event) != "none"
            and winner.cross.trajectory_supported
            and winner.source_front_supported
            and winner.target_front_supported
            and not conflict
        )
        winner.trajectory_accepted = bool(
            winner.measurement_supported
            and winner.tip_length_supported
        )
        winner.measurable_growth_onset_supported = bool(
            winner.trajectory_accepted
            and winner.measurable_growth_onset_source_frame is not None
        )

        source_phase = phase_a if winner.source_phase == phase_a.name else phase_b
        source_frame = _sample_source_frame(source_phase, winner.source_onset.sample)
        target_frame = _sample_source_frame(
            winner.target_phase,
            winner.cross.onset_sample,
        )
        if source_frame is not None and target_frame is not None:
            disagreement_seconds = (
                abs(source_frame - target_frame) * seconds_per_source_frame
            )
            winner.onset_time_accepted = bool(
                winner.germination_accepted
                and winner.source_onset.accepted
                and winner.cross.onset_supported
                and disagreement_seconds <= onset_tolerance_seconds
            )
            if winner.onset_time_accepted:
                winner.onset_source_frame = int(round((source_frame + target_frame) / 2))

        source_bounds = [
            _sample_source_frame(source_phase, winner.source_onset.support_start_sample),
            _sample_source_frame(source_phase, winner.source_onset.support_end_sample),
        ]
        target_bounds = [
            _sample_source_frame(winner.target_phase, winner.cross.onset_lower_sample),
            _sample_source_frame(winner.target_phase, winner.cross.onset_upper_sample),
        ]
        bounds = [value for value in (*source_bounds, *target_bounds) if value is not None]
        if bounds:
            winner.onset_lower_source_frame = min(bounds)
            winner.onset_upper_source_frame = max(bounds)
    return candidates


def write_summary(
    path: Path,
    candidates: list[ConsensusCandidate],
    analysis_scale_to_native: float = 1.0,
    pixel_size: float | None = None,
    distance_unit: str = "um",
) -> None:
    """Write every source candidate and its independent phase provenance."""

    fields = [
        "consensus_id", "selected", "source_phase", "source_event_id",
        "source_grain_id", "canonical_grain_id", "source_germination_accepted",
        "source_trajectory_scope", "source_quality_status", "source_quality_flags",
        "source_final_length_px", "source_final_length_native_px",
        "source_final_length_calibrated", "distance_unit",
        "source_grain_x_px", "source_grain_y_px",
        "source_grain_radius_px", "source_grain_circle_score",
        "source_grain_observations", "source_grain_warmup_observations",
        "target_grain_id", "target_grain_x_px", "target_grain_y_px",
        "target_grain_radius_px", "target_grain_circle_score",
        "target_grain_observations", "target_grain_warmup_observations",
        "grain_match_distance_px",
        "cross_path_birth_coverage", "cross_eventual_path_support_fraction",
        "cross_birth_order_correlation", "cross_birth_order_forward_fraction",
        "cross_birth_progress_samples", "cross_growth_step_count",
        "cross_tip_step_median_px", "cross_tip_step_max_px",
        "cross_path_supported", "cross_trajectory_supported",
        "cross_onset_supported", "cross_onset_quality", "cross_reason",
        "source_root_onset_sample", "source_root_onset_source_frame",
        "source_rim_onset_sample", "source_rim_onset_source_frame",
        "source_contrast_onset_sample", "source_contrast_onset_source_frame",
        "source_consensus_onset_sample", "source_consensus_onset_source_frame",
        "target_root_onset_sample", "target_root_onset_source_frame",
        "target_rim_onset_sample", "target_rim_onset_source_frame",
        "target_contrast_onset_sample", "target_contrast_onset_source_frame",
        "target_consensus_onset_sample", "target_consensus_onset_source_frame",
        "source_front_supported", "target_front_supported",
        "source_front_reason", "target_front_reason",
        "source_front_direct_support_fraction",
        "target_front_direct_support_fraction",
        "source_front_eventual_support_fraction",
        "target_front_eventual_support_fraction",
        "tip_timing_agreement_fraction", "tip_timing_disagreement_median_s",
        "tip_timing_disagreement_max_s", "tip_timing_supported",
        "tip_length_agreement_fraction", "tip_length_disagreement_median_px",
        "tip_length_disagreement_p90_px", "tip_length_disagreement_max_px",
        "tip_length_supported",
        "geometry_conflict", "consensus_germination_accepted",
        "consensus_onset_time_accepted", "consensus_trajectory_accepted",
        "consensus_measurement_supported",
        "consensus_onset_source_frame", "consensus_onset_lower_source_frame",
        "consensus_onset_upper_source_frame",
        "measurable_growth_onset_supported",
        "measurable_growth_onset_source_frame",
        "measurable_growth_onset_lower_source_frame",
        "measurable_growth_onset_upper_source_frame",
        "measurable_growth_onset_uncertainty_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            event = candidate.event
            cross = candidate.cross
            source_phase = candidate.source_analysis
            target_phase = candidate.target_phase
            source_root = getattr(event, "birth_start", None)
            source_rim = getattr(event, "rim_emergence_sample", None)
            source_contrast = getattr(
                event,
                "rim_contrast_emergence_sample",
                None,
            )
            source_consensus = getattr(candidate.source_onset, "sample", None)

            native_length = (
                float(event.arclength_px[-1]) * analysis_scale_to_native
            )
            writer.writerow(
                {
                    "consensus_id": candidate.consensus_id or "",
                    "selected": candidate.selected,
                    "source_phase": candidate.source_phase,
                    "source_event_id": event.event_id,
                    "source_grain_id": event.grain_id,
                    "canonical_grain_id": candidate.canonical_grain_id,
                    "source_germination_accepted": event.auto_accepted,
                    "source_trajectory_scope": _trajectory_measurement_scope(event),
                    "source_quality_status": event.quality_status,
                    "source_quality_flags": ";".join(event.quality_flags),
                    "source_final_length_px": round(
                        float(event.arclength_px[-1]), 4
                    ),
                    "source_final_length_native_px": round(native_length, 4),
                    "source_final_length_calibrated": (
                        "" if pixel_size is None else
                        round(native_length * pixel_size, 6)
                    ),
                    "distance_unit": "px" if pixel_size is None else distance_unit,
                    "source_grain_x_px": round(
                        float(candidate.source_grain.center_xy[0]), 4
                    ),
                    "source_grain_y_px": round(
                        float(candidate.source_grain.center_xy[1]), 4
                    ),
                    "source_grain_radius_px": round(
                        candidate.source_grain.radius_px, 4
                    ),
                    "source_grain_circle_score": round(
                        candidate.source_grain.circle_score, 4
                    ),
                    "source_grain_observations": candidate.source_grain.observations,
                    "source_grain_warmup_observations": (
                        candidate.source_grain.warmup_observations or ""
                    ),
                    "target_grain_id": cross.target_grain_id or "",
                    "target_grain_x_px": (
                        "" if candidate.target_grain is None else
                        round(float(candidate.target_grain.center_xy[0]), 4)
                    ),
                    "target_grain_y_px": (
                        "" if candidate.target_grain is None else
                        round(float(candidate.target_grain.center_xy[1]), 4)
                    ),
                    "target_grain_radius_px": (
                        "" if candidate.target_grain is None else
                        round(candidate.target_grain.radius_px, 4)
                    ),
                    "target_grain_circle_score": (
                        "" if candidate.target_grain is None else
                        round(candidate.target_grain.circle_score, 4)
                    ),
                    "target_grain_observations": (
                        "" if candidate.target_grain is None else
                        candidate.target_grain.observations
                    ),
                    "target_grain_warmup_observations": (
                        "" if candidate.target_grain is None else
                        candidate.target_grain.warmup_observations or ""
                    ),
                    "grain_match_distance_px": (
                        "" if cross.grain_match_distance_px is None
                        else round(cross.grain_match_distance_px, 4)
                    ),
                    "cross_path_birth_coverage": round(cross.path_birth_coverage, 4),
                    "cross_eventual_path_support_fraction": round(
                        cross.eventual_path_support_fraction, 4
                    ),
                    "cross_birth_order_correlation": round(
                        cross.birth_order_correlation, 4
                    ),
                    "cross_birth_order_forward_fraction": round(
                        cross.birth_order_forward_fraction, 4
                    ),
                    "cross_birth_progress_samples": round(
                        cross.birth_progress_samples, 4
                    ),
                    "cross_growth_step_count": cross.growth_step_count,
                    "cross_tip_step_median_px": round(cross.tip_step_median_px, 4),
                    "cross_tip_step_max_px": round(cross.tip_step_max_px, 4),
                    "cross_path_supported": cross.path_supported,
                    "cross_trajectory_supported": cross.trajectory_supported,
                    "cross_onset_supported": cross.onset_supported,
                    "cross_onset_quality": cross.onset_quality,
                    "cross_reason": cross.reason,
                    "source_root_onset_sample": (
                        source_root if source_root is not None else ""
                    ),
                    "source_root_onset_source_frame": _csv_source_frame(
                        source_phase,
                        source_root,
                    ),
                    "source_rim_onset_sample": (
                        source_rim if source_rim is not None else ""
                    ),
                    "source_rim_onset_source_frame": _csv_source_frame(
                        source_phase,
                        source_rim,
                    ),
                    "source_contrast_onset_sample": (
                        source_contrast if source_contrast is not None else ""
                    ),
                    "source_contrast_onset_source_frame": _csv_source_frame(
                        source_phase,
                        source_contrast,
                    ),
                    "source_consensus_onset_sample": (
                        source_consensus if source_consensus is not None else ""
                    ),
                    "source_consensus_onset_source_frame": _csv_source_frame(
                        source_phase,
                        source_consensus,
                    ),
                    "target_root_onset_sample": (
                        cross.root_birth_sample
                        if cross.root_birth_sample is not None else ""
                    ),
                    "target_root_onset_source_frame": _csv_source_frame(
                        target_phase,
                        cross.root_birth_sample,
                    ),
                    "target_rim_onset_sample": (
                        cross.rim_emergence_sample
                        if cross.rim_emergence_sample is not None else ""
                    ),
                    "target_rim_onset_source_frame": _csv_source_frame(
                        target_phase,
                        cross.rim_emergence_sample,
                    ),
                    "target_contrast_onset_sample": (
                        cross.rim_contrast_emergence_sample
                        if cross.rim_contrast_emergence_sample is not None else ""
                    ),
                    "target_contrast_onset_source_frame": _csv_source_frame(
                        target_phase,
                        cross.rim_contrast_emergence_sample,
                    ),
                    "target_consensus_onset_sample": (
                        cross.onset_sample if cross.onset_sample is not None else ""
                    ),
                    "target_consensus_onset_source_frame": _csv_source_frame(
                        target_phase,
                        cross.onset_sample,
                    ),
                    "source_front_supported": candidate.source_front_supported,
                    "target_front_supported": candidate.target_front_supported,
                    "source_front_reason": candidate.source_front_reason,
                    "target_front_reason": candidate.target_front_reason,
                    "source_front_direct_support_fraction": round(
                        candidate.source_front_direct_support_fraction, 4
                    ),
                    "target_front_direct_support_fraction": round(
                        candidate.target_front_direct_support_fraction, 4
                    ),
                    "source_front_eventual_support_fraction": round(
                        candidate.source_front_eventual_support_fraction, 4
                    ),
                    "target_front_eventual_support_fraction": round(
                        candidate.target_front_eventual_support_fraction, 4
                    ),
                    "tip_timing_agreement_fraction": round(
                        candidate.tip_timing_agreement_fraction, 4
                    ),
                    "tip_timing_disagreement_median_s": (
                        "" if candidate.tip_timing_disagreement_median_s is None else
                        round(candidate.tip_timing_disagreement_median_s, 4)
                    ),
                    "tip_timing_disagreement_max_s": (
                        "" if candidate.tip_timing_disagreement_max_s is None else
                        round(candidate.tip_timing_disagreement_max_s, 4)
                    ),
                    "tip_timing_supported": candidate.tip_timing_supported,
                    "tip_length_agreement_fraction": round(
                        candidate.tip_length_agreement_fraction,
                        4,
                    ),
                    "tip_length_disagreement_median_px": (
                        "" if candidate.tip_length_disagreement_median_px is None else
                        round(candidate.tip_length_disagreement_median_px, 4)
                    ),
                    "tip_length_disagreement_p90_px": (
                        "" if candidate.tip_length_disagreement_p90_px is None else
                        round(candidate.tip_length_disagreement_p90_px, 4)
                    ),
                    "tip_length_disagreement_max_px": (
                        "" if candidate.tip_length_disagreement_max_px is None else
                        round(candidate.tip_length_disagreement_max_px, 4)
                    ),
                    "tip_length_supported": candidate.tip_length_supported,
                    "geometry_conflict": candidate.geometry_conflict,
                    "consensus_germination_accepted": candidate.germination_accepted,
                    "consensus_onset_time_accepted": candidate.onset_time_accepted,
                    "consensus_trajectory_accepted": candidate.trajectory_accepted,
                    "consensus_measurement_supported": (
                        candidate.measurement_supported
                    ),
                    "consensus_onset_source_frame": candidate.onset_source_frame or "",
                    "consensus_onset_lower_source_frame": (
                        candidate.onset_lower_source_frame or ""
                    ),
                    "consensus_onset_upper_source_frame": (
                        candidate.onset_upper_source_frame or ""
                    ),
                    "measurable_growth_onset_supported": (
                        candidate.measurable_growth_onset_supported
                    ),
                    "measurable_growth_onset_source_frame": (
                        candidate.measurable_growth_onset_source_frame or ""
                    ),
                    "measurable_growth_onset_lower_source_frame": (
                        candidate.measurable_growth_onset_lower_source_frame or ""
                    ),
                    "measurable_growth_onset_upper_source_frame": (
                        candidate.measurable_growth_onset_upper_source_frame or ""
                    ),
                    "measurable_growth_onset_uncertainty_s": (
                        ""
                        if candidate.measurable_growth_onset_uncertainty_s is None
                        else round(
                            candidate.measurable_growth_onset_uncertainty_s,
                            4,
                        )
                    ),
                }
            )


def onset_cue_agreement_report(
    candidates: list[ConsensusCandidate],
    seconds_per_source_frame: float,
    tolerance_seconds: float,
) -> dict[str, object]:
    """Summarize phase stability of raw onset cues on trusted trajectories."""

    trusted = [
        candidate
        for candidate in candidates
        if candidate.selected and candidate.trajectory_accepted
    ]
    cue_samples = {
        "connected_root": lambda candidate: (
            getattr(candidate.event, "birth_start", None),
            candidate.cross.root_birth_sample,
        ),
        "pollen_rim": lambda candidate: (
            getattr(candidate.event, "rim_emergence_sample", None),
            candidate.cross.rim_emergence_sample,
        ),
        "path_contrast": lambda candidate: (
            getattr(candidate.event, "rim_contrast_emergence_sample", None),
            candidate.cross.rim_contrast_emergence_sample,
        ),
        "phase_consensus": lambda candidate: (
            getattr(candidate.source_onset, "sample", None),
            candidate.cross.onset_sample,
        ),
    }
    cues: dict[str, object] = {}
    for name, samples_for in cue_samples.items():
        disagreements = []
        for candidate in trusted:
            source_sample, target_sample = samples_for(candidate)
            source_frame = _sample_source_frame(
                candidate.source_analysis,
                source_sample,
            )
            target_frame = _sample_source_frame(
                candidate.target_phase,
                target_sample,
            )
            if source_frame is None or target_frame is None:
                continue
            disagreements.append(
                abs(source_frame - target_frame) * seconds_per_source_frame
            )

        values = np.asarray(disagreements, dtype=np.float64)
        paired_count = int(len(values))
        within_count = int(np.sum(values <= tolerance_seconds))
        cues[name] = {
            "paired_count": paired_count,
            "missing_count": len(trusted) - paired_count,
            "within_tolerance_count": within_count,
            "within_tolerance_fraction": (
                round(within_count / paired_count, 4) if paired_count else None
            ),
            "median_disagreement_s": (
                round(float(np.median(values)), 4) if paired_count else None
            ),
            "p90_disagreement_s": (
                round(float(np.percentile(values, 90)), 4) if paired_count else None
            ),
            "maximum_disagreement_s": (
                round(float(np.max(values)), 4) if paired_count else None
            ),
        }
    return {
        "population": "selected automatic trajectories",
        "trajectory_count": len(trusted),
        "exact_time_tolerance_s": tolerance_seconds,
        "cues": cues,
    }


def _nearest_phase_sample(phase: PhaseAnalysis, source_frame: int) -> int:
    """Return the phase sample nearest one requested source frame."""

    position = int(np.searchsorted(phase.source_frames, source_frame))
    choices = np.clip([position - 1, position], 0, len(phase.source_frames) - 1)
    return int(
        min(choices, key=lambda index: abs(int(phase.source_frames[index]) - source_frame))
    )


def _csv_source_frame(
    phase: PhaseAnalysis | None,
    sample: int | None,
) -> int | str:
    """Format one optional phase sample as a CSV-safe source frame."""

    if phase is None:
        return ""
    value = _sample_source_frame(phase, sample)
    return "" if value is None else value


def write_measurements(
    path: Path,
    candidates: list[ConsensusCandidate],
    phase_a: PhaseAnalysis,
    phase_b: PhaseAnalysis,
    output_seconds: float,
    seconds_per_source_frame: float,
    analysis_scale_to_native: float,
    pixel_size: float | None,
    distance_unit: str,
    tip_length_tolerance_px: float,
) -> dict[str, int]:
    """Write phase-consensus tip and arclength measurements at output cadence."""

    actual_analysis_seconds = sample_interval_seconds(
        phase_a.source_frames,
        fps=1.0 / seconds_per_source_frame,
        source_frame_interval_seconds=seconds_per_source_frame,
    )
    stride = max(1, int(round(output_seconds / actual_analysis_seconds)))
    selected = [candidate for candidate in candidates if candidate.selected]
    fields = [
        "consensus_id", "source_frame", "elapsed_time_sec", "source_phase",
        "source_event_id", "canonical_grain_id", "germination_accepted",
        "onset_time_accepted", "measurement_supported", "trajectory_accepted",
        "tip_measurement_supported", "tip_measurement_accepted",
        "measurable_growth_onset_supported",
        "elapsed_since_measurable_growth_onset_sec",
        "measurable_growth_onset_uncertainty_s",
        "measurement_status", "length_analysis_px", "length_native_px",
        "length_calibrated", "distance_unit", "tip_x_px", "tip_y_px",
        "tip_phase_length_disagreement_px", "tip_phase_length_agreement",
        "tip_phase_disagreement_sec",
    ]
    counts = {"accepted": 0, "review": 0, "unavailable": 0}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for output_sample in range(0, len(phase_a.source_frames), stride):
            source_frame = int(phase_a.source_frames[output_sample])
            elapsed = (
                source_frame - int(phase_a.source_frames[0])
            ) * seconds_per_source_frame
            for candidate in selected:
                source_phase = (
                    phase_a if candidate.source_phase == phase_a.name else phase_b
                )
                source_sample = _nearest_phase_sample(source_phase, source_frame)
                length, registered_tip, path_index = consensus_tip_at(
                    candidate,
                    source_frame,
                )
                source_tip_accepted = _tip_measurement_is_accepted(
                    candidate.event,
                    source_sample,
                    path_index,
                )
                supported = bool(
                    candidate.measurement_supported and source_tip_accepted
                )
                tip = (
                    registered_tip
                    if supported and path_index >= 0
                    else None
                )
                native_length = length * analysis_scale_to_native
                calibrated_length = (
                    None if pixel_size is None else native_length * pixel_size
                )
                elapsed_since_measurable = ""
                if (
                    candidate.measurable_growth_onset_supported
                    and candidate.measurable_growth_onset_source_frame is not None
                ):
                    elapsed_since_measurable = round(
                        (
                            source_frame
                            - candidate.measurable_growth_onset_source_frame
                        )
                        * seconds_per_source_frame,
                        4,
                    )
                tip_disagreement_s: float | None = None
                tip_length_disagreement_px: float | None = None
                if supported and path_index >= 0:
                    source_tip_frames = candidate.source_tip_birth_source_frames
                    target_tip_frames = candidate.target_tip_birth_source_frames
                    if (
                        source_tip_frames is not None
                        and target_tip_frames is not None
                        and path_index < len(source_tip_frames)
                        and path_index < len(target_tip_frames)
                    ):
                        tip_disagreement_s = round(
                            abs(
                                int(source_tip_frames[path_index])
                                - int(target_tip_frames[path_index])
                            )
                            * seconds_per_source_frame,
                            4,
                        )
                        source_phase_length = length_at_source_frame(
                            candidate.event.arclength_px,
                            source_tip_frames,
                            source_frame,
                        )
                        target_phase_length = length_at_source_frame(
                            candidate.event.arclength_px,
                            target_tip_frames,
                            source_frame,
                        )
                        tip_length_disagreement_px = round(
                            abs(source_phase_length - target_phase_length),
                            4,
                        )
                tip_phase_length_agreement = bool(
                    tip_length_disagreement_px is not None
                    and tip_length_disagreement_px <= tip_length_tolerance_px
                )
                accepted = bool(supported and tip_phase_length_agreement)
                if accepted:
                    measurement_status = "accepted"
                elif supported:
                    measurement_status = "review"
                else:
                    measurement_status = "unavailable"
                counts[measurement_status] += 1
                writer.writerow(
                    {
                        "consensus_id": candidate.consensus_id,
                        "source_frame": source_frame,
                        "elapsed_time_sec": round(elapsed, 4),
                        "source_phase": candidate.source_phase,
                        "source_event_id": candidate.event.event_id,
                        "canonical_grain_id": candidate.canonical_grain_id,
                        "germination_accepted": candidate.germination_accepted,
                        "onset_time_accepted": candidate.onset_time_accepted,
                        "measurement_supported": candidate.measurement_supported,
                        "trajectory_accepted": candidate.trajectory_accepted,
                        "tip_measurement_supported": supported,
                        "tip_measurement_accepted": accepted,
                        "measurable_growth_onset_supported": (
                            candidate.measurable_growth_onset_supported
                        ),
                        "elapsed_since_measurable_growth_onset_sec": (
                            elapsed_since_measurable
                        ),
                        "measurable_growth_onset_uncertainty_s": (
                            ""
                            if candidate.measurable_growth_onset_uncertainty_s is None
                            else round(
                                candidate.measurable_growth_onset_uncertainty_s,
                                4,
                            )
                        ),
                        "measurement_status": measurement_status,
                        "length_analysis_px": round(length, 4) if supported else "",
                        "length_native_px": (
                            round(native_length, 4) if supported else ""
                        ),
                        "length_calibrated": (
                            "" if not supported or calibrated_length is None else
                            round(calibrated_length, 6)
                        ),
                        "distance_unit": (
                            "px" if pixel_size is None else distance_unit
                        ),
                        "tip_x_px": "" if tip is None else round(float(tip[0]), 3),
                        "tip_y_px": "" if tip is None else round(float(tip[1]), 3),
                        "tip_phase_length_disagreement_px": (
                            ""
                            if tip_length_disagreement_px is None
                            else tip_length_disagreement_px
                        ),
                        "tip_phase_length_agreement": tip_phase_length_agreement,
                        "tip_phase_disagreement_sec": (
                            "" if tip_disagreement_s is None else tip_disagreement_s
                        ),
                    }
                )
    return counts


def write_paths(
    path: Path,
    candidates: list[ConsensusCandidate],
    phase_a: PhaseAnalysis,
    phase_b: PhaseAnalysis,
    seconds_per_source_frame: float,
) -> None:
    """Write every candidate path and its selected tip timing without loss."""

    fields = [
        "source_phase", "source_event_id", "source_grain_id",
        "canonical_grain_id", "consensus_id", "selected", "geometry_conflict",
        "consensus_germination_accepted", "consensus_measurement_supported",
        "consensus_trajectory_accepted",
        "path_index", "reference_x_px", "reference_y_px", "source_phase_x_px",
        "source_phase_y_px", "arclength_px", "original_tip_birth_sample",
        "original_tip_birth_source_frame", "source_front_tip_birth_source_frame",
        "target_tip_birth_source_frame",
        "consensus_tip_birth_source_frame", "tip_phase_disagreement_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            source_phase = (
                phase_a if candidate.source_phase == phase_a.name else phase_b
            )
            timing = _selected_tip_timing(candidate.event)
            for index, (
                reference_point,
                source_point,
                arclength,
                birth_sample,
            ) in enumerate(
                zip(
                    candidate.path_reference_xy,
                    candidate.event.path_xy,
                    candidate.event.arclength_px,
                    timing,
                )
            ):
                writer.writerow(
                    {
                        "source_phase": candidate.source_phase,
                        "source_event_id": candidate.event.event_id,
                        "source_grain_id": candidate.event.grain_id,
                        "canonical_grain_id": candidate.canonical_grain_id,
                        "consensus_id": candidate.consensus_id or "",
                        "selected": candidate.selected,
                        "geometry_conflict": candidate.geometry_conflict,
                        "consensus_germination_accepted": (
                            candidate.germination_accepted
                        ),
                        "consensus_measurement_supported": (
                            candidate.measurement_supported
                        ),
                        "consensus_trajectory_accepted": (
                            candidate.trajectory_accepted
                        ),
                        "path_index": index,
                        "reference_x_px": round(float(reference_point[0]), 4),
                        "reference_y_px": round(float(reference_point[1]), 4),
                        "source_phase_x_px": round(float(source_point[0]), 4),
                        "source_phase_y_px": round(float(source_point[1]), 4),
                        "arclength_px": round(float(arclength), 4),
                        "original_tip_birth_sample": int(birth_sample),
                        "original_tip_birth_source_frame": _sample_source_frame(
                            source_phase,
                            int(birth_sample),
                        ),
                        "source_front_tip_birth_source_frame": (
                            "" if candidate.source_tip_birth_source_frames is None
                            or index >= len(candidate.source_tip_birth_source_frames)
                            else int(candidate.source_tip_birth_source_frames[index])
                        ),
                        "target_tip_birth_source_frame": (
                            "" if candidate.target_tip_birth_source_frames is None
                            or index >= len(candidate.target_tip_birth_source_frames)
                            else int(candidate.target_tip_birth_source_frames[index])
                        ),
                        "consensus_tip_birth_source_frame": (
                            "" if candidate.consensus_tip_birth_source_frames is None
                            or index >= len(candidate.consensus_tip_birth_source_frames)
                            else int(candidate.consensus_tip_birth_source_frames[index])
                        ),
                        "tip_phase_disagreement_sec": (
                            "" if candidate.source_tip_birth_source_frames is None
                            or candidate.target_tip_birth_source_frames is None
                            or index >= len(candidate.source_tip_birth_source_frames)
                            or index >= len(candidate.target_tip_birth_source_frames)
                            else round(
                                abs(
                                    int(candidate.source_tip_birth_source_frames[index])
                                    - int(candidate.target_tip_birth_source_frames[index])
                                )
                                * seconds_per_source_frame,
                                4,
                            )
                        ),
                    }
                )


def write_overview(
    path: Path,
    candidates: list[ConsensusCandidate],
    final_frame: np.ndarray,
) -> None:
    """Draw selected consensus paths on the final stabilized raw image."""

    frame = cv.cvtColor(final_frame, cv.COLOR_GRAY2BGR)
    selected = [candidate for candidate in candidates if candidate.selected]
    for candidate in selected:
        points = np.rint(candidate.path_reference_xy).astype(np.int32)
        if candidate.geometry_conflict:
            color = (0, 120, 255)
        elif candidate.trajectory_accepted:
            color = (40, 190, 70)
        elif candidate.measurement_supported:
            color = (20, 150, 230)
        elif candidate.germination_accepted:
            color = (30, 200, 220)
        else:
            color = (150, 150, 150)
        cv.polylines(frame, [points], False, color, 2, cv.LINE_AA)
        cv.circle(frame, tuple(points[0]), 3, (230, 100, 20), -1, cv.LINE_AA)
        cv.circle(frame, tuple(points[-1]), 3, (20, 20, 230), -1, cv.LINE_AA)
        cv.putText(
            frame,
            f"C{candidate.consensus_id}",
            tuple(points[-1] + np.asarray([3, -3])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            f"C{candidate.consensus_id}",
            tuple(points[-1] + np.asarray([3, -3])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
    cv.imwrite(str(path), frame)


def candidate_audit_source_frames(
    candidate: ConsensusCandidate,
    source_start: int,
    source_end: int,
    before_source_frames: int,
) -> list[int]:
    """Choose pre-growth and path-milestone frames for one visual audit row."""

    timing = candidate.consensus_tip_birth_source_frames
    if timing is None or len(timing) == 0:
        return []
    values = np.asarray(timing, dtype=np.int64)
    milestone_indices = np.rint(
        np.linspace(0, len(values) - 1, 5)
    ).astype(np.int32)
    frames = [
        max(source_start, int(values[0]) - before_source_frames),
        *(int(values[index]) for index in milestone_indices),
    ]
    return list(
        dict.fromkeys(int(np.clip(frame, source_start, source_end)) for frame in frames)
    )


def write_candidate_review_sheet(
    path: Path,
    candidates: list[ConsensusCandidate],
    display_phase: PhaseAnalysis,
    output_seconds: float,
    seconds_per_source_frame: float,
) -> None:
    """Pair raw and annotated crops for every retained manual-review path."""

    reviews = [
        candidate
        for candidate in candidates
        if candidate.selected
        and candidate.measurement_supported
        and not candidate.trajectory_accepted
    ]
    tile_size = 112
    caption_height = 18
    label_width = 118
    column_count = 6
    row_height = caption_height + 2 * tile_size + 8
    header_height = 28
    canvas = np.full(
        (
            header_height + max(1, len(reviews)) * row_height,
            label_width + column_count * tile_size,
            3,
        ),
        238,
        dtype=np.uint8,
    )
    cv.putText(
        canvas,
        "raw image (top) / proposed rooted path (bottom)",
        (label_width, 19),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1,
        cv.LINE_AA,
    )
    if not reviews:
        cv.putText(
            canvas,
            "No retained review paths",
            (12, header_height + 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv.LINE_AA,
        )
        cv.imwrite(str(path), canvas)
        return

    source_start = int(display_phase.source_frames[0])
    source_end = int(display_phase.source_frames[-1])
    before_source_frames = max(
        1,
        int(round(output_seconds / seconds_per_source_frame)),
    )
    for row_index, candidate in enumerate(reviews):
        row_top = header_height + row_index * row_height
        status = (
            "onset review"
            if candidate.tip_length_supported
            else "length review"
        )
        cv.putText(
            canvas,
            f"C{candidate.consensus_id}",
            (8, row_top + 42),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            (25, 25, 25),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            status,
            (8, row_top + 62),
            cv.FONT_HERSHEY_SIMPLEX,
            0.37,
            (70, 70, 70),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            f"{float(candidate.event.arclength_px[-1]):.1f}px",
            (8, row_top + 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.37,
            (70, 70, 70),
            1,
            cv.LINE_AA,
        )

        audit_frames = candidate_audit_source_frames(
            candidate,
            source_start,
            source_end,
            before_source_frames,
        )
        paths = [candidate_path_at(candidate, frame) for frame in audit_frames]
        all_points = np.concatenate(paths, axis=0)
        center = np.mean(
            np.vstack((np.min(all_points, axis=0), np.max(all_points, axis=0))),
            axis=0,
        )
        span = np.ptp(all_points, axis=0)
        crop_size = int(max(64, np.ceil(float(np.max(span))) + 32))
        crop_size = min(
            crop_size,
            display_phase.aligned.shape[1],
            display_phase.aligned.shape[2],
        )
        x0 = int(
            np.clip(
                round(center[0] - crop_size / 2),
                0,
                display_phase.aligned.shape[2] - crop_size,
            )
        )
        y0 = int(
            np.clip(
                round(center[1] - crop_size / 2),
                0,
                display_phase.aligned.shape[1] - crop_size,
            )
        )
        x1, y1 = x0 + crop_size, y0 + crop_size

        for column, source_frame in enumerate(audit_frames[:column_count]):
            sample = _nearest_phase_sample(display_phase, source_frame)
            actual_source_frame = int(display_phase.source_frames[sample])
            raw = display_phase.aligned[sample]
            annotated = cv.cvtColor(raw, cv.COLOR_GRAY2BGR)
            length, tip, path_index = consensus_tip_at(candidate, actual_source_frame)
            current_path = candidate_path_at(candidate, actual_source_frame)
            if tip is not None and path_index >= 0:
                points = np.rint(current_path[: path_index + 1]).astype(np.int32)
                if len(points) >= 2:
                    cv.polylines(
                        annotated,
                        [points],
                        False,
                        (20, 150, 230),
                        2,
                        cv.LINE_AA,
                    )
                cv.circle(
                    annotated,
                    tuple(np.rint(tip).astype(np.int32)),
                    3,
                    (20, 20, 230),
                    -1,
                    cv.LINE_AA,
                )
            raw_crop = cv.resize(
                raw[y0:y1, x0:x1],
                (tile_size, tile_size),
                interpolation=cv.INTER_AREA,
            )
            raw_crop = cv.cvtColor(raw_crop, cv.COLOR_GRAY2BGR)
            annotated_crop = cv.resize(
                annotated[y0:y1, x0:x1],
                (tile_size, tile_size),
                interpolation=cv.INTER_AREA,
            )
            column_left = label_width + column * tile_size
            cv.putText(
                canvas,
                f"f{actual_source_frame} {length:.1f}px",
                (column_left + 3, row_top + 13),
                cv.FONT_HERSHEY_SIMPLEX,
                0.3,
                (50, 50, 50),
                1,
                cv.LINE_AA,
            )
            canvas[
                row_top + caption_height : row_top + caption_height + tile_size,
                column_left : column_left + tile_size,
            ] = raw_crop
            canvas[
                row_top + caption_height + tile_size :
                row_top + caption_height + 2 * tile_size,
                column_left : column_left + tile_size,
            ] = annotated_crop
    cv.imwrite(str(path), canvas)


def write_review_video(
    path: Path,
    candidates: list[ConsensusCandidate],
    phase_a: PhaseAnalysis,
    phase_b: PhaseAnalysis,
    output_seconds: float,
    seconds_per_source_frame: float,
) -> None:
    """Render only independently validated tip paths at requested cadence."""

    actual_analysis_seconds = sample_interval_seconds(
        phase_a.source_frames,
        fps=1.0 / seconds_per_source_frame,
        source_frame_interval_seconds=seconds_per_source_frame,
    )
    stride = max(1, int(round(output_seconds / actual_analysis_seconds)))
    height, width = phase_a.aligned.shape[1:]
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    selected = [
        candidate
        for candidate in candidates
        if candidate.selected and candidate.measurement_supported
    ]
    palette = [(40, 190, 70), (220, 120, 30), (190, 50, 190), (30, 180, 220)]
    for output_sample in range(0, len(phase_a.source_frames), stride):
        source_frame = int(phase_a.source_frames[output_sample])
        frame = cv.cvtColor(phase_a.aligned[output_sample], cv.COLOR_GRAY2BGR)
        for index, candidate in enumerate(selected):
            source_phase = (
                phase_a if candidate.source_phase == phase_a.name else phase_b
            )
            source_sample = _nearest_phase_sample(source_phase, source_frame)
            _, _, path_index = consensus_tip_at(candidate, source_frame)
            if not _tip_measurement_is_accepted(
                candidate.event,
                source_sample,
                path_index,
            ):
                continue
            points = np.rint(
                candidate_path_at(candidate, source_frame)[: path_index + 1]
            ).astype(np.int32)
            color = (
                palette[index % len(palette)]
                if candidate.trajectory_accepted
                else (20, 150, 230)
            )
            if len(points) >= 2:
                cv.polylines(
                    frame,
                    [points],
                    False,
                    color,
                    2,
                    cv.LINE_AA,
                )
            cv.circle(frame, tuple(points[-1]), 3, (20, 20, 230), -1, cv.LINE_AA)
        cv.putText(
            frame,
            f"frame {source_frame}",
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
    """Parse reproducible phase-consensus analysis settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-end", type=int)
    parser.add_argument("--output-seconds", type=float, default=3.0)
    parser.add_argument(
        "--analysis-seconds",
        type=float,
        help="Internal cadence; defaults to half the requested output interval.",
    )
    parser.add_argument(
        "--source-frame-interval-seconds",
        type=float,
        help="Experimental time represented by one encoded source frame.",
    )
    parser.add_argument(
        "--tip-timing-tolerance-intervals",
        type=float,
        help=(
            "Allowed phase disagreement in internal intervals; defaults to the "
            "temporal median's full support width."
        ),
    )
    parser.add_argument(
        "--min-tip-timing-agreement-fraction",
        type=float,
        default=0.8,
        help="Minimum path fraction whose two tip timelines must localize together.",
    )
    parser.add_argument(
        "--tip-length-tolerance-px",
        type=float,
        help="Ordinary phase-to-phase tip-length tolerance in analysis pixels.",
    )
    parser.add_argument(
        "--max-tip-length-disagreement-px",
        type=float,
        help="Largest allowed phase-to-phase tip-length error.",
    )
    parser.add_argument(
        "--min-tip-length-agreement-fraction",
        type=float,
        default=0.8,
        help="Required fraction of matched times within the length tolerance.",
    )
    parser.add_argument(
        "--onset-timing-tolerance-intervals",
        type=float,
        default=1.0,
        help="Allowed phase disagreement for an exact germination time.",
    )
    parser.add_argument(
        "--path-registration",
        choices=("fixed", "rigid", "branch-locked"),
        default="branch-locked",
        help="How one fixed tube branch is aligned across analysis time.",
    )
    parser.add_argument("--grain-motion-falloff-px", type=float)
    parser.add_argument("--normal-search-radius-px", type=float)
    parser.add_argument("--normal-search-step-px", type=float, default=1.0)
    parser.add_argument(
        "--registration-spatial-window-points",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--registration-temporal-window-samples",
        type=int,
        default=7,
    )
    parser.add_argument("--registration-iterations", type=int, default=3)
    parser.add_argument("--registration-prior-weight", type=float, default=0.35)
    parser.add_argument("--registration-birth-lead-samples", type=int, default=8)
    parser.add_argument("--registration-root-lock-points", type=int, default=3)
    parser.add_argument(
        "--registration-max-normal-step-px",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--visible-front-coverage-cost",
        type=float,
        default=0.05,
        help="Penalty for extending the visible tip across unsupported path points.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Physical distance represented by one native source pixel.",
    )
    parser.add_argument("--distance-unit", default="um")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--no-review-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run two phases, cross-validate paths, and write compact consensus outputs."""

    args = parse_args()
    analysis_seconds = args.analysis_seconds or args.output_seconds / 2.0
    if args.output_seconds <= 0 or analysis_seconds <= 0:
        raise ValueError("analysis and output intervals must be positive")
    if analysis_seconds > args.output_seconds:
        raise ValueError("analysis interval cannot exceed output interval")
    if (
        args.tip_timing_tolerance_intervals is not None
        and args.tip_timing_tolerance_intervals <= 0
    ):
        raise ValueError("tip timing tolerance must be positive")
    if not 0 < args.min_tip_timing_agreement_fraction <= 1:
        raise ValueError("tip timing agreement fraction must be in (0, 1]")
    if args.tip_length_tolerance_px is not None and args.tip_length_tolerance_px <= 0:
        raise ValueError("tip length tolerance must be positive")
    if (
        args.max_tip_length_disagreement_px is not None
        and args.max_tip_length_disagreement_px <= 0
    ):
        raise ValueError("maximum tip length disagreement must be positive")
    if not 0 < args.min_tip_length_agreement_fraction <= 1:
        raise ValueError("tip length agreement fraction must be in (0, 1]")
    if args.onset_timing_tolerance_intervals <= 0:
        raise ValueError("onset timing tolerance must be positive")
    if args.grain_motion_falloff_px is not None and args.grain_motion_falloff_px <= 0:
        raise ValueError("grain motion falloff must be positive")
    if args.normal_search_radius_px is not None and args.normal_search_radius_px < 0:
        raise ValueError("normal search radius cannot be negative")
    if args.normal_search_step_px <= 0:
        raise ValueError("normal search step must be positive")
    if (
        args.registration_spatial_window_points <= 0
        or args.registration_temporal_window_samples <= 0
        or args.registration_iterations <= 0
        or args.registration_root_lock_points <= 0
    ):
        raise ValueError("registration windows, iterations, and root lock must be positive")
    if args.registration_prior_weight < 0:
        raise ValueError("registration prior weight cannot be negative")
    if args.registration_birth_lead_samples < 0:
        raise ValueError("registration birth lead cannot be negative")
    if args.registration_max_normal_step_px < 0:
        raise ValueError("registration normal step limit cannot be negative")
    if args.visible_front_coverage_cost < 0:
        raise ValueError("visible front coverage cost cannot be negative")
    if args.pixel_size is not None and args.pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    fps, frame_count, native_width, _ = movie_metadata(args.movie)
    seconds_per_source_frame = _source_seconds_per_frame(
        fps,
        args.source_frame_interval_seconds,
    )
    phase_offset_frames = max(
        1,
        int(round(0.5 * analysis_seconds / seconds_per_source_frame)),
    )
    source_end = frame_count if args.source_end is None else args.source_end
    phase_a_frames = source_frame_indices(
        frame_count,
        fps,
        args.source_start,
        source_end,
        analysis_seconds,
        args.source_frame_interval_seconds,
    )
    phase_b_start = args.source_start + phase_offset_frames
    phase_b_frames = source_frame_indices(
        frame_count,
        fps,
        phase_b_start,
        source_end,
        analysis_seconds,
        args.source_frame_interval_seconds,
    )
    config = AtlasConfig.for_width(args.width).for_sample_interval(
        analysis_seconds
    )
    tip_timing_tolerance_intervals = (
        args.tip_timing_tolerance_intervals
        if args.tip_timing_tolerance_intervals is not None
        else max(1.0, float(config.front_temporal_median_samples - 1))
    )
    consensus_config = PhaseConsensusConfig(
        tip_timing_tolerance_intervals=tip_timing_tolerance_intervals,
        min_tip_timing_agreement_fraction=(
            args.min_tip_timing_agreement_fraction
        ),
        onset_timing_tolerance_intervals=args.onset_timing_tolerance_intervals,
        tip_length_tolerance_px=(
            args.tip_length_tolerance_px
            if args.tip_length_tolerance_px is not None
            else config.max_tip_step_median_px
        ),
        max_tip_length_disagreement_px=(
            args.max_tip_length_disagreement_px
            if args.max_tip_length_disagreement_px is not None
            else config.max_tip_step_px
        ),
        min_tip_length_agreement_fraction=(
            args.min_tip_length_agreement_fraction
        ),
        path_registration_mode=args.path_registration,
        grain_motion_falloff_px=(
            args.grain_motion_falloff_px
            if args.grain_motion_falloff_px is not None
            else 12.0 * args.width / 480.0
        ),
        normal_search_radius_px=(
            args.normal_search_radius_px
            if args.normal_search_radius_px is not None
            else 3.0 * args.width / 480.0
        ),
        normal_search_step_px=args.normal_search_step_px,
        registration_spatial_window_points=(
            args.registration_spatial_window_points
        ),
        registration_temporal_window_samples=(
            args.registration_temporal_window_samples
        ),
        registration_iterations=args.registration_iterations,
        registration_prior_weight=args.registration_prior_weight,
        registration_birth_lead_samples=args.registration_birth_lead_samples,
        registration_root_lock_points=args.registration_root_lock_points,
        registration_max_normal_step_px=args.registration_max_normal_step_px,
        visible_front_coverage_cost=args.visible_front_coverage_cost,
    )
    phase_a = analyze_phase("A", args.movie, phase_a_frames, config)
    phase_b = analyze_phase("B", args.movie, phase_b_frames, config)
    candidates = build_candidates(
        phase_a,
        phase_b,
        config,
        consensus_config,
        seconds_per_source_frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "phase_consensus_summary.csv"
    measurements_path = args.output_dir / "phase_consensus_measurements.csv"
    paths_path = args.output_dir / "phase_consensus_paths.csv"
    overview_path = args.output_dir / "phase_consensus_overview.jpg"
    review_sheet_path = args.output_dir / "phase_consensus_review_sheet.jpg"
    video_path = args.output_dir / "phase_consensus_review.mp4"
    analysis_scale_to_native = native_width / args.width
    write_summary(
        summary_path,
        candidates,
        analysis_scale_to_native,
        args.pixel_size,
        args.distance_unit,
    )
    measurement_row_counts = write_measurements(
        measurements_path,
        candidates,
        phase_a,
        phase_b,
        args.output_seconds,
        seconds_per_source_frame,
        analysis_scale_to_native,
        args.pixel_size,
        args.distance_unit,
        consensus_config.tip_length_tolerance_px,
    )
    write_paths(
        paths_path,
        candidates,
        phase_a,
        phase_b,
        seconds_per_source_frame,
    )
    write_overview(overview_path, candidates, phase_a.aligned[-1])
    write_candidate_review_sheet(
        review_sheet_path,
        candidates,
        phase_a,
        args.output_seconds,
        seconds_per_source_frame,
    )
    if not args.no_review_video:
        write_review_video(
            video_path,
            candidates,
            phase_a,
            phase_b,
            args.output_seconds,
            seconds_per_source_frame,
        )

    selected = [candidate for candidate in candidates if candidate.selected]
    report = {
        "prototype": "v18_phase_consensus",
        "input_movie": str(args.movie.resolve()),
        "source_window": [args.source_start, source_end],
        "output_interval_s": args.output_seconds,
        "analysis_interval_s": analysis_seconds,
        "phase_offset_source_frames": phase_offset_frames,
        "phase_offset_s": phase_offset_frames * seconds_per_source_frame,
        "phase_sample_counts": {
            phase_a.name: len(phase_a.source_frames),
            phase_b.name: len(phase_b.source_frames),
        },
        "phase_grain_counts": {
            phase_a.name: len(phase_a.grains),
            phase_b.name: len(phase_b.grains),
        },
        "phase_hypothesis_counts": {
            phase_a.name: len(phase_a.events),
            phase_b.name: len(phase_b.events),
        },
        "source_candidate_count": len(candidates),
        "selected_consensus_count": len(selected),
        "consensus_germination_count": sum(
            candidate.germination_accepted for candidate in selected
        ),
        "phase_supported_germination_review_count": sum(
            candidate.event.auto_accepted
            and candidate.cross.path_supported
            and not candidate.germination_accepted
            for candidate in selected
        ),
        "consensus_exact_onset_count": sum(
            candidate.onset_time_accepted for candidate in selected
        ),
        "measurable_growth_onset_count": sum(
            candidate.measurable_growth_onset_supported
            for candidate in selected
        ),
        "consensus_measurement_supported_count": sum(
            candidate.measurement_supported for candidate in selected
        ),
        "consensus_trajectory_count": sum(
            candidate.trajectory_accepted for candidate in selected
        ),
        "trajectory_without_accepted_germination_count": sum(
            candidate.trajectory_accepted and not candidate.germination_accepted
            for candidate in selected
        ),
        "measurement_row_counts": measurement_row_counts,
        "phase_supported_trajectory_review_count": sum(
            candidate.measurement_supported and not candidate.trajectory_accepted
            for candidate in selected
        ),
        "tip_timing_supported_count": sum(
            candidate.tip_timing_supported for candidate in selected
        ),
        "tip_length_supported_count": sum(
            candidate.tip_length_supported for candidate in selected
        ),
        "geometry_conflict_count": sum(
            candidate.geometry_conflict for candidate in selected
        ),
        "onset_cue_phase_agreement": onset_cue_agreement_report(
            selected,
            seconds_per_source_frame,
            consensus_config.onset_timing_tolerance_intervals
            * analysis_seconds,
        ),
        "configuration": asdict(config),
        "phase_consensus_configuration": asdict(consensus_config),
        "measurement_calibration": {
            "analysis_scale_to_native": analysis_scale_to_native,
            "pixel_size_per_native_pixel": args.pixel_size,
            "distance_unit": "px" if args.pixel_size is None else args.distance_unit,
        },
        "phase_calibration": {
            phase_a.name: phase_a.calibration,
            phase_b.name: phase_b.calibration,
        },
        "artifacts": {
            "summary_csv": str(summary_path),
            "measurements_csv": str(measurements_path),
            "paths_csv": str(paths_path),
            "overview_image": str(overview_path),
            "review_sheet": str(review_sheet_path),
            "review_video": None if args.no_review_video else str(video_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        "[v18] "
        f"{report['consensus_germination_count']} germinations, "
        f"{report['consensus_trajectory_count']} automatic trajectories, "
        f"{report['phase_supported_trajectory_review_count']} trajectory reviews "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
