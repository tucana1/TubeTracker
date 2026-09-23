"""Resolve pollen-tube continuations with explicit future-node and edge evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2 as cv
import numpy as np

from .anchored_tracing import (
    find_tip_extension_candidates,
    refine_chain_from_prior,
)
from .curve_prototype import curve_length, ordered_prefix_retention, resample_curve
from .pollen_anchored_chain import (
    PollenAnchoredChainConfig,
    PollenAnchoredMeasurement,
    chain_integrity,
    trace_pollen_anchored_chain,
)


@dataclass(frozen=True)
class TopologyAwareTraceConfig:
    """Configure temporal edge attribution and deferred branch selection."""

    historical_probability_threshold: float = 0.10
    historical_dilation_px: int = 1
    maximum_historical_overlap: float = 0.50
    maximum_historical_run_px: float = 4.0
    historical_reacquisition_maximum_growth_px: float = 8.0
    historical_reacquisition_minimum_structure_support: float = 0.45
    historical_reacquisition_minimum_temporal_support: float = 0.65
    historical_reacquisition_minimum_tangent_agreement: float = 0.75
    minimum_edge_support: float = 0.04
    minimum_tangent_agreement: float = 0.35
    maximum_incremental_growth_px: float = 8.0
    maximum_candidates: int = 6
    hypothesis_confirmation_frames: int = 2
    hypothesis_prefix_tolerance_px: float = 4.5
    minimum_hypothesis_prefix_retention: float = 0.70
    ambiguity_quality_margin: float = 0.08
    minimum_distinct_endpoint_distance_px: float = 6.0
    minimum_distinct_branch_angle_degrees: float = 25.0
    maximum_unconfirmed_growth_px: float = 6.0


@dataclass(frozen=True)
class FutureEdgeHypothesis:
    """Describe one future node and the complete edge connecting it to the tip."""

    chain_yx: np.ndarray
    extension_yx: np.ndarray
    quality: float
    structure_support: float
    temporal_support: float
    edge_support: float
    historical_overlap: float
    maximum_historical_run_px: float
    tangent_agreement: float
    added_length_px: float
    historical_reacquisition: bool = False
    confirmations: int = 1


def _empty_points():
    """Return an empty row-column centerline with a stable shape."""
    return np.empty((0, 2), dtype=np.float64)


def _measurement(
    status,
    usable,
    chain=None,
    candidate=None,
    structure_support=0.0,
    chain_supported_fraction=0.0,
    root_attachment_error_px=np.nan,
    maximum_centerline_step_px=np.nan,
    neighbor_contact=False,
):
    """Build one topology-aware result using the shared export contract."""
    points = _empty_points() if chain is None else np.asarray(chain, dtype=np.float64)
    candidate_points = (
        _empty_points()
        if candidate is None
        else np.asarray(candidate, dtype=np.float64)
    )
    return PollenAnchoredMeasurement(
        status=str(status),
        usable=bool(usable),
        length_px=0.0 if len(points) < 2 else float(curve_length(points)),
        centerline_yx=points,
        candidate_centerline_yx=candidate_points,
        structure_support=float(structure_support),
        seed_confirmations=0,
        chain_supported_fraction=float(chain_supported_fraction),
        root_attachment_error_px=float(root_attachment_error_px),
        maximum_centerline_step_px=float(maximum_centerline_step_px),
        neighbor_contact=bool(neighbor_contact),
    )


def historical_edge_occupancy(
    historical_probability,
    config=None,
):
    """Mark structure that was already present before a proposed future edge."""
    config = config or TopologyAwareTraceConfig()
    probability = np.asarray(historical_probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("historical_probability must be two-dimensional")
    occupied = (
        probability >= float(config.historical_probability_threshold)
    ).astype(np.uint8)
    radius = max(0, int(config.historical_dilation_px))
    if radius:
        size = 2 * radius + 1
        occupied = cv.dilate(
            occupied,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)),
        )
    return occupied.astype(bool)


def predict_future_edge_cue(
    vessel_probability,
    novelty_probability,
    historical_probability,
    config=None,
):
    """Estimate edges that are supported now but were absent in earlier frames."""
    config = config or TopologyAwareTraceConfig()
    vessel = np.clip(np.asarray(vessel_probability, dtype=np.float32), 0.0, 1.0)
    novelty = np.clip(np.asarray(novelty_probability, dtype=np.float32), 0.0, 1.0)
    if vessel.shape != novelty.shape:
        raise ValueError("vessel and novelty probabilities must match")
    occupied = historical_edge_occupancy(historical_probability, config)
    if occupied.shape != vessel.shape:
        raise ValueError("historical probability must match current evidence")
    cue = np.sqrt(vessel * novelty)
    cue[occupied] = 0.0
    return cue.astype(np.float32), occupied


def _sample_path(image, path):
    """Sample a two-dimensional map along a floating-point row-column path."""
    points = np.rint(np.asarray(path, dtype=np.float64)).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[0] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image.shape[1] - 1)
    return np.asarray(image)[points[:, 0], points[:, 1]]


def _tangent_agreement(prior, extension):
    """Compare the previous distal tangent with the proposed connecting edge."""
    prior = np.asarray(prior, dtype=np.float64)
    extension = np.asarray(extension, dtype=np.float64)
    prior_vector = prior[-1] - prior[max(0, len(prior) - 6)]
    edge_vector = extension[min(len(extension) - 1, 5)] - extension[0]
    prior_norm = float(np.linalg.norm(prior_vector))
    edge_norm = float(np.linalg.norm(edge_vector))
    if prior_norm <= 1e-6 or edge_norm <= 1e-6:
        return -1.0
    return float(np.dot(prior_vector / prior_norm, edge_vector / edge_norm))


def _maximum_true_run_length(mask, path):
    """Measure the longest continuously occupied portion of one candidate edge."""
    occupied = np.asarray(mask, dtype=bool)
    points = np.asarray(path, dtype=np.float64)
    values = _sample_path(occupied, points).astype(bool)
    steps = np.concatenate(
        ([0.0], np.linalg.norm(np.diff(points, axis=0), axis=1))
    )
    longest = current = 0.0
    for value, step in zip(values, steps):
        if value:
            current += max(float(step), 1.0)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def future_edge_hypotheses(
    frame_evidence,
    historical_probability,
    prior,
    exclusion,
    chain_config,
    config=None,
):
    """Generate future nodes and retain only edges with valid temporal ownership."""
    config = config or TopologyAwareTraceConfig()
    edge_cue, occupied = predict_future_edge_cue(
        frame_evidence.vessel_probability,
        frame_evidence.novelty_probability,
        historical_probability,
        config,
    )
    raw = find_tip_extension_candidates(
        frame_evidence.vessel_probability,
        frame_evidence.novelty_probability,
        prior,
        exclusion,
        chain_config.trace,
        edge_cue=edge_cue,
        maximum_candidates=config.maximum_candidates,
    )
    hypotheses = []
    for _, extension, structure, temporal in raw or ():
        distal = extension[max(1, len(extension) // 4) :]
        if not len(distal):
            continue
        edge_support = float(np.mean(_sample_path(edge_cue, distal)))
        historical_overlap = float(np.mean(_sample_path(occupied, distal)))
        historical_run = _maximum_true_run_length(occupied, distal)
        tangent = _tangent_agreement(prior, extension)
        joined = np.vstack((prior, extension[1:]))
        chain = resample_curve(
            joined,
            spacing=2.0,
            max_points=chain_config.maximum_centerline_points,
        )
        added = float(curve_length(chain) - curve_length(prior))
        historical_conflict = bool(
            historical_overlap > config.maximum_historical_overlap
            or historical_run > config.maximum_historical_run_px
            or edge_support < config.minimum_edge_support
        )
        historical_reacquisition = bool(
            historical_conflict
            and added
            <= config.historical_reacquisition_maximum_growth_px
            and structure
            >= config.historical_reacquisition_minimum_structure_support
            and temporal
            >= config.historical_reacquisition_minimum_temporal_support
            and tangent
            >= config.historical_reacquisition_minimum_tangent_agreement
        )
        if (
            added <= 0.0
            or added > chain_config.maximum_direct_extension_px
            or added > config.maximum_incremental_growth_px
            or (historical_conflict and not historical_reacquisition)
            or tangent < config.minimum_tangent_agreement
        ):
            continue
        ownership_support = edge_support
        if historical_reacquisition:
            ownership_support = float(
                np.sqrt(
                    temporal
                    * np.clip((tangent + 1.0) / 2.0, 0.0, 1.0)
                )
            )
        cues = np.clip(
            [
                structure,
                temporal,
                ownership_support,
                (tangent + 1.0) / 2.0,
            ],
            1e-6,
            1.0,
        )
        quality = float(np.exp(np.mean(np.log(cues))))
        hypotheses.append(
            FutureEdgeHypothesis(
                chain_yx=chain,
                extension_yx=np.asarray(extension, dtype=np.float64),
                quality=quality,
                structure_support=float(structure),
                temporal_support=float(temporal),
                edge_support=edge_support,
                historical_overlap=historical_overlap,
                maximum_historical_run_px=historical_run,
                tangent_agreement=tangent,
                added_length_px=added,
                historical_reacquisition=historical_reacquisition,
            )
        )
    return tuple(sorted(hypotheses, key=lambda item: item.quality, reverse=True))


def _carry_hypothesis_confirmations(previous, current, config):
    """Match competing branches across frames without collapsing the hypothesis set."""
    updated = []
    for candidate in current:
        confirmations = 1
        for earlier in previous:
            retention = ordered_prefix_retention(
                earlier.chain_yx,
                candidate.chain_yx,
                config.hypothesis_prefix_tolerance_px,
            )
            if retention >= config.minimum_hypothesis_prefix_retention:
                confirmations = max(confirmations, earlier.confirmations + 1)
        updated.append(replace(candidate, confirmations=confirmations))
    return tuple(sorted(updated, key=lambda item: item.quality, reverse=True))


def _is_ambiguous(hypotheses, config):
    """Report whether two differently located future nodes remain similarly likely."""
    if len(hypotheses) < 2:
        return False
    first, second = hypotheses[:2]
    endpoint_distance = float(
        np.linalg.norm(first.extension_yx[-1] - second.extension_yx[-1])
    )
    first_direction = first.extension_yx[-1] - first.extension_yx[0]
    second_direction = second.extension_yx[-1] - second.extension_yx[0]
    first_direction /= max(float(np.linalg.norm(first_direction)), 1e-6)
    second_direction /= max(float(np.linalg.norm(second_direction)), 1e-6)
    direction_agreement = float(
        np.clip(np.dot(first_direction, second_direction), -1.0, 1.0)
    )
    minimum_agreement = float(
        np.cos(np.deg2rad(config.minimum_distinct_branch_angle_degrees))
    )
    return (
        endpoint_distance >= config.minimum_distinct_endpoint_distance_px
        and direction_agreement < minimum_agreement
        and first.quality - second.quality <= config.ambiguity_quality_margin
    )


def trace_topology_aware_chain(
    evidence,
    exclusions,
    pollen_radius,
    chain_config=None,
    config=None,
):
    """Trace a full chain while deferring uncertain node-edge continuation choices."""
    chain_config = chain_config or PollenAnchoredChainConfig()
    config = config or TopologyAwareTraceConfig()
    baseline = trace_pollen_anchored_chain(
        evidence,
        exclusions,
        pollen_radius,
        chain_config,
    )
    initialized = next(
        (
            index
            for index, measurement in enumerate(baseline)
            if measurement.usable and len(measurement.centerline_yx) >= 2
        ),
        None,
    )
    if initialized is None:
        return baseline

    output = list(baseline[: initialized + 1])
    chain = baseline[initialized].centerline_yx.copy()
    pending = ()
    contact_censored = False
    center_yx = np.asarray(
        [chain_config.crop_size / 2.0, chain_config.crop_size / 2.0]
    )
    baseline_count = min(
        max(1, int(chain_config.baseline_observations)),
        len(evidence),
    )
    historical = np.max(
        np.stack(
            [
                frame_evidence.vessel_probability
                for frame_evidence in evidence[:baseline_count]
            ]
        ),
        axis=0,
    )

    for frame in range(initialized + 1, len(evidence)):
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

        frame_evidence = evidence[frame]
        exclusion = np.asarray(exclusions[frame], dtype=bool)
        refined, support, _ = refine_chain_from_prior(
            frame_evidence.vessel_probability,
            chain,
            exclusion=exclusion,
            search_radius=chain_config.trace.chain_refit_radius,
            root_lock_points=chain_config.trace.chain_root_lock_points,
            offset_change_penalty=(
                chain_config.trace.chain_refit_offset_change_penalty
            ),
            offset_magnitude_penalty=(
                chain_config.trace.chain_refit_offset_magnitude_penalty
            ),
        )
        previous_length = float(curve_length(chain))
        refined_length = float(curve_length(refined))
        refit_valid = (
            support >= chain_config.trace.chain_refit_min_support
            and abs(refined_length - previous_length)
            / max(previous_length, 1.0)
            <= chain_config.trace.chain_refit_max_length_fraction
        )
        integrity = chain_integrity(
            frame_evidence.vessel_probability,
            refined,
            center_yx,
            pollen_radius,
            chain_config,
            exclusion,
        )
        refit_valid = refit_valid and integrity[0]
        if not refit_valid:
            if integrity[4]:
                contact_censored = True
                status = "neighbor_contact_censored"
            else:
                status = "topology_chain_held_for_review"
            output.append(
                _measurement(
                    status,
                    False,
                    chain=chain,
                    structure_support=support,
                    chain_supported_fraction=integrity[1],
                    root_attachment_error_px=integrity[2],
                    maximum_centerline_step_px=integrity[3],
                    neighbor_contact=integrity[4],
                )
            )
            pending = ()
            continue

        chain = refined
        integrity = chain_integrity(
            frame_evidence.vessel_probability,
            chain,
            center_yx,
            pollen_radius,
            chain_config,
            exclusion,
        )
        hypotheses = future_edge_hypotheses(
            frame_evidence,
            historical,
            chain,
            exclusion,
            chain_config,
            config,
        )
        pending = _carry_hypothesis_confirmations(pending, hypotheses, config)
        accepted = None
        if pending and not _is_ambiguous(pending, config):
            strongest = pending[0]
            if (
                strongest.added_length_px
                <= config.maximum_unconfirmed_growth_px
                or strongest.confirmations
                >= config.hypothesis_confirmation_frames
            ):
                proposed_integrity = chain_integrity(
                    frame_evidence.vessel_probability,
                    strongest.chain_yx,
                    center_yx,
                    pollen_radius,
                    chain_config,
                    exclusion,
                )
                if proposed_integrity[0]:
                    accepted = strongest
                    chain = strongest.chain_yx
                    integrity = proposed_integrity

        if accepted is not None:
            status = (
                "topology_historical_reacquired"
                if accepted.historical_reacquisition
                else "topology_chain_extended"
            )
            usable = True
            pending = ()
            candidate = None
            support = accepted.structure_support
        elif pending:
            status = (
                "topology_crossing_unresolved"
                if _is_ambiguous(pending, config)
                else "topology_edge_pending"
            )
            usable = False
            candidate = pending[0].chain_yx
        else:
            status = "topology_chain_updated"
            usable = True
            candidate = None

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
