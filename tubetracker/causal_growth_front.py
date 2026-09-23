"""Recover tube growth as an ordered sequence of image changepoints.

The mature centerline fixes branch identity.  This module only decides how far
along that immutable path the tube was visible at each sampled time.  Unlike an
occupancy objective, each path point is rewarded once, when it first appears.
That prevents a persistent neighbouring tube from winning simply because it is
visible for many frames after a crossover.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


@dataclass(frozen=True)
class CausalGrowthFrontResult:
    """One outward-only front fitted to path-relative appearance evidence."""

    birth_samples: np.ndarray
    front_indices: np.ndarray
    direct_support_mask: np.ndarray
    eventual_support_mask: np.ndarray
    feasible: bool
    reason: str
    objective_score: float
    final_point_index: int
    final_length_px: float
    direct_support_fraction: float
    eventual_support_fraction: float


@dataclass(frozen=True)
class ScalarPathDeformationResult:
    """A narrow-band material curve fitted jointly over time and arclength."""

    curves_xy: np.ndarray
    offsets_px: np.ndarray
    support: np.ndarray
    initial_energy: float
    final_energy: float


@dataclass(frozen=True)
class CausalPathHypothesis:
    """Summarize one mature geometry after rigid causal reconstruction."""

    label: str
    result: CausalGrowthFrontResult
    total_length_px: float
    root_distance_px: float
    radial_excursion_px: float
    terminates_at_foreign_owner: bool = False
    topology_accepted: bool = True
    topology_reason: str = "owner-path-topology-certified"


@dataclass(frozen=True)
class CausalPathSelection:
    """Record the selected mature geometry and why it displaced the baseline."""

    index: int
    reason: str


@dataclass(frozen=True)
class RootedGrowthCertificate:
    """State whether a causal path visibly leaves its owner as a tube."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class OwnerPathTopologyCertificate:
    """Describe whether a retained centerline exits one pollen body cleanly."""

    accepted: bool
    reason: str
    root_distance_px: float
    minimum_distance_px: float
    maximum_distance_px: float
    radial_excursion_px: float
    first_halo_exit_index: int | None
    first_halo_exit_arclength_px: float | None


@dataclass(frozen=True)
class GrowthDirectionCertificate:
    """State whether causal evidence supports growth away from the owner."""

    accepted: bool
    reason: str
    outward_normalized_score: float
    inward_normalized_score: float
    inward_score_margin: float


@dataclass(frozen=True)
class NativePathGrowthCertificate:
    """State whether native paired walls independently verify a growth path."""

    accepted: bool
    reason: str
    completion_fraction: float
    onset_delay_samples: int | None


@dataclass(frozen=True)
class NativeFrontRefinement:
    """Store a prior-constrained native material-front reconstruction."""

    lengths_px: np.ndarray
    point_indices: np.ndarray
    mean_data_gain: float
    mean_absolute_correction_px: float
    corridor_edge_fraction: float
    accepted: bool
    reason: str


def fit_prior_constrained_native_front(
    support: np.ndarray,
    thresholds: np.ndarray,
    arc_lengths_px: np.ndarray,
    prior_lengths_px: np.ndarray,
    *,
    corridor_radius_px: float = 4.0,
    prior_sigma_px: float = 2.0,
    maximum_growth_px: float = 8.0,
    minimum_refinable_length_px: float = 5.0,
    transition_penalty: float = 0.02,
    absolute_evidence_weight: float = 0.35,
    causal_evidence_weight: float = 0.65,
    causal_training_margin_px: float = 4.0,
    minimum_training_samples: int = 6,
    minimum_causal_separation: float = 1.0,
    minimum_mean_data_gain: float = 0.01,
    maximum_corridor_edge_fraction: float = 0.20,
) -> NativeFrontRefinement:
    """Fit one monotone tube tip from native support near an existing front.

    Absolute paired-wall evidence localizes the physical endpoint.  A second,
    self-calibrated term compares each material point before and after the
    coarse front passed it.  That term suppresses static clutter because a
    structure that was already present cannot imitate tube arrival.
    """

    values = np.asarray(support, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    arc = np.asarray(arc_lengths_px, dtype=np.float64)
    prior = np.asarray(prior_lengths_px, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(arc):
        raise ValueError("support and arc lengths describe different paths")
    if limits.shape != (len(values),) or prior.shape != (len(values),):
        raise ValueError("thresholds and prior lengths must match time")
    if len(arc) < 2 or np.any(np.diff(arc) <= 0.0) or arc[0] != 0.0:
        raise ValueError("arc lengths must start at zero and increase strictly")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(limits).all()
        or not np.isfinite(prior).all()
    ):
        raise ValueError("native front inputs must be finite")
    if np.any(np.diff(prior) < -1e-6) or np.any(prior < 0.0):
        raise ValueError("prior lengths must be nonnegative and nondecreasing")
    if np.any(prior > arc[-1] + corridor_radius_px + 1e-6):
        raise ValueError("prior lengths extend beyond the native path")
    if (
        corridor_radius_px <= 0.0
        or prior_sigma_px <= 0.0
        or maximum_growth_px <= 0.0
        or transition_penalty < 0.0
        or absolute_evidence_weight < 0.0
        or causal_evidence_weight < 0.0
        or absolute_evidence_weight + causal_evidence_weight <= 0.0
        or causal_training_margin_px < corridor_radius_px
        or minimum_training_samples < 2
        or minimum_causal_separation <= 0.0
    ):
        raise ValueError("native front constraints must be positive")

    smoothed_values = median_filter(values, size=(3, 1), mode="nearest")
    evidence_scale = np.maximum(2.0, 0.5 * np.maximum(limits, 0.0))
    absolute_logits = np.clip(
        (smoothed_values - limits[:, None]) / evidence_scale[:, None],
        -8.0,
        8.0,
    )
    causal_logits = np.zeros_like(absolute_logits)
    arrival_informative = np.zeros(len(arc), dtype=bool)
    preexisting_static = np.zeros(len(arc), dtype=bool)
    for point, position in enumerate(arc):
        before = prior <= position - causal_training_margin_px
        after = prior >= position + causal_training_margin_px
        if np.count_nonzero(before) < minimum_training_samples:
            continue
        before_absolute = float(np.median(absolute_logits[before, point]))
        if np.count_nonzero(after) < minimum_training_samples:
            if before_absolute > 0.5:
                causal_logits[:, point] = -min(before_absolute, 4.0)
                preexisting_static[point] = True
            continue
        before_level = float(np.median(smoothed_values[before, point]))
        after_level = float(np.median(smoothed_values[after, point]))
        residuals = np.concatenate(
            (
                smoothed_values[before, point] - before_level,
                smoothed_values[after, point] - after_level,
            )
        )
        noise = max(
            1.0,
            1.4826 * float(np.median(np.abs(residuals))),
        )
        separation = (after_level - before_level) / noise
        if separation < minimum_causal_separation:
            if before_absolute > 0.5:
                causal_logits[:, point] = -min(before_absolute, 4.0)
                preexisting_static[point] = True
            continue
        midpoint = 0.5 * (before_level + after_level)
        reliability = float(
            np.clip(
                (separation - minimum_causal_separation) / 2.0,
                0.0,
                1.0,
            )
        )
        causal_logits[:, point] = reliability * np.clip(
            (smoothed_values[:, point] - midpoint) / noise,
            -8.0,
            8.0,
        )
        arrival_informative[point] = True
    logits = absolute_logits.copy()
    logits[:, arrival_informative] = (
        absolute_evidence_weight * absolute_logits[:, arrival_informative]
        + causal_evidence_weight * causal_logits[:, arrival_informative]
    ) / (absolute_evidence_weight + causal_evidence_weight)
    logits[:, preexisting_static] = (
        np.minimum(absolute_logits[:, preexisting_static], 0.0)
        + causal_logits[:, preexisting_static]
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    present_cost = -np.log(np.clip(probability, 1e-6, 1.0))
    absent_cost = -np.log(np.clip(1.0 - probability, 1e-6, 1.0))
    present_prefix = np.cumsum(present_cost, axis=1)
    absent_suffix = np.cumsum(absent_cost[:, ::-1], axis=1)[:, ::-1]
    data_cost = present_prefix.copy()
    data_cost[:, :-1] += absent_suffix[:, 1:]

    time_count, point_count = values.shape
    allowed = np.abs(arc[None] - prior[:, None]) <= corridor_radius_px + 1e-9
    frozen = prior < minimum_refinable_length_px
    nearest_prior = np.argmin(np.abs(arc[None] - prior[:, None]), axis=1)
    allowed[frozen] = False
    allowed[np.flatnonzero(frozen), nearest_prior[frozen]] = True
    if np.any(~np.any(allowed, axis=1)):
        raise ValueError("native corridor contains no feasible state")

    prior_cost = 0.5 * (
        (arc[None] - prior[:, None]) / prior_sigma_px
    ) ** 2
    observation_cost = data_cost + prior_cost
    observation_cost[~allowed] = np.inf
    energy = np.full((time_count, point_count), np.inf, dtype=np.float64)
    previous = np.full((time_count, point_count), -1, dtype=np.int32)
    energy[0] = observation_cost[0]
    prior_steps = np.diff(prior, prepend=prior[0])
    for sample in range(1, time_count):
        for state in np.flatnonzero(allowed[sample]):
            growth = arc[state] - arc
            valid = (
                (growth >= -1e-9)
                & (growth <= maximum_growth_px + 1e-9)
                & np.isfinite(energy[sample - 1])
            )
            if not np.any(valid):
                continue
            transition = energy[sample - 1] + transition_penalty * (
                growth - prior_steps[sample]
            ) ** 2
            transition[~valid] = np.inf
            source = int(np.argmin(transition))
            energy[sample, state] = observation_cost[sample, state] + transition[source]
            previous[sample, state] = source

    final_state = int(np.argmin(energy[-1]))
    if not np.isfinite(energy[-1, final_state]):
        raise ValueError("native front constraints have no complete path")
    states = np.empty(time_count, dtype=np.int32)
    states[-1] = final_state
    for sample in range(time_count - 1, 0, -1):
        states[sample - 1] = previous[sample, states[sample]]
    if np.any(states < 0):
        raise RuntimeError("native front backtracking failed")

    lengths = arc[states]
    prior_states = nearest_prior
    refinable = ~frozen
    if np.any(refinable):
        selected_data = data_cost[np.arange(time_count), states]
        prior_data = data_cost[np.arange(time_count), prior_states]
        mean_gain = float(
            np.mean((prior_data[refinable] - selected_data[refinable]) / point_count)
        )
        mean_correction = float(np.mean(np.abs(lengths[refinable] - prior[refinable])))
        edge_hits = np.zeros(time_count, dtype=bool)
        for sample in np.flatnonzero(refinable):
            states_here = np.flatnonzero(allowed[sample])
            edge_hits[sample] = states[sample] in {states_here[0], states_here[-1]}
        edge_fraction = float(np.mean(edge_hits[refinable]))
    else:
        mean_gain = 0.0
        mean_correction = 0.0
        edge_fraction = 0.0

    if not np.any(refinable):
        reason = "no-native-front-to-refine"
    elif mean_gain < minimum_mean_data_gain:
        reason = "insufficient-native-front-gain"
    elif edge_fraction > maximum_corridor_edge_fraction:
        reason = "native-front-reaches-corridor-boundary"
    else:
        reason = "native-front-refinement-verified"
    return NativeFrontRefinement(
        lengths_px=lengths.astype(np.float32),
        point_indices=states,
        mean_data_gain=mean_gain,
        mean_absolute_correction_px=mean_correction,
        corridor_edge_fraction=edge_fraction,
        accepted=reason == "native-front-refinement-verified",
        reason=reason,
    )


def certify_native_path_growth(
    prefix_lengths_px: np.ndarray,
    *,
    native_onset_sample: int | None,
    causal_onset_sample: int | None,
    total_length_px: float,
    minimum_completion_fraction: float = 0.75,
    maximum_onset_delay_samples: int = 12,
) -> NativePathGrowthCertificate:
    """Accept complete native-wall growth and identify timing corrections."""

    lengths = np.asarray(prefix_lengths_px, dtype=np.float64)
    if lengths.ndim != 1 or not len(lengths) or not np.isfinite(lengths).all():
        raise ValueError("prefix_lengths_px must be a finite one-dimensional timeline")
    if np.any(np.diff(lengths) < -1e-6):
        raise ValueError("native prefix lengths must be nondecreasing")
    if total_length_px <= 0.0 or not np.isfinite(total_length_px):
        raise ValueError("total_length_px must be finite and positive")
    if not 0.0 <= minimum_completion_fraction <= 1.0:
        raise ValueError("minimum completion fraction must be a probability")
    if maximum_onset_delay_samples < 0:
        raise ValueError("maximum onset delay cannot be negative")

    completion = float(lengths[-1] / total_length_px)
    delay = (
        None
        if native_onset_sample is None or causal_onset_sample is None
        else int(native_onset_sample - causal_onset_sample)
    )
    if native_onset_sample is None:
        reason = "no-native-connected-growth"
    elif completion < minimum_completion_fraction:
        reason = "insufficient-native-path-completion"
    elif causal_onset_sample is None:
        reason = "native-path-growth-verified-with-native-onset"
    elif delay > maximum_onset_delay_samples:
        reason = "native-path-growth-verified-with-later-onset"
    else:
        reason = "native-path-growth-verified"
    return NativePathGrowthCertificate(
        accepted=reason.startswith("native-path-growth-verified"),
        reason=reason,
        completion_fraction=completion,
        onset_delay_samples=delay,
    )


def certify_growth_direction(
    outward: CausalGrowthFrontResult,
    inward: CausalGrowthFrontResult,
    *,
    total_length_px: float,
    minimum_inward_score_margin: float = 0.50,
    minimum_inward_coverage: float = 0.65,
    minimum_inward_direct_support: float = 0.85,
    minimum_inward_eventual_support: float = 0.95,
) -> GrowthDirectionCertificate:
    """Reject a path whose appearance propagates toward its proposed owner.

    The same material profile is fitted in both directions.  Reverse evidence
    must be substantially stronger, cover most of the path, and have strong
    direct and eventual support before it can overturn the owner orientation.
    """

    if total_length_px <= 0.0 or not np.isfinite(total_length_px):
        raise ValueError("total_length_px must be finite and positive")
    if (
        minimum_inward_score_margin < 0.0
        or not 0.0 <= minimum_inward_coverage <= 1.0
        or not 0.0 <= minimum_inward_direct_support <= 1.0
        or not 0.0 <= minimum_inward_eventual_support <= 1.0
    ):
        raise ValueError("direction thresholds must be nonnegative probabilities")

    def normalized_score(result: CausalGrowthFrontResult) -> float:
        reached = max(1, result.final_point_index + 1)
        return float(result.objective_score / np.sqrt(reached))

    outward_score = normalized_score(outward)
    inward_score = normalized_score(inward)
    margin = inward_score - outward_score
    inward_dominant = bool(
        inward.feasible
        and inward.final_length_px / total_length_px >= minimum_inward_coverage
        and inward.direct_support_fraction >= minimum_inward_direct_support
        and inward.eventual_support_fraction >= minimum_inward_eventual_support
        and margin >= minimum_inward_score_margin
    )
    return GrowthDirectionCertificate(
        accepted=not inward_dominant,
        reason=(
            "distal-inward-growth-dominant"
            if inward_dominant
            else "owner-outward-direction-not-contradicted"
        ),
        outward_normalized_score=outward_score,
        inward_normalized_score=inward_score,
        inward_score_margin=margin,
    )


def certify_rooted_growth(
    hypothesis: CausalPathHypothesis,
    *,
    owner_radius_px: float,
    boundary_censored: bool = False,
    minimum_radial_excursion_radii: float = 0.65,
    minimum_direct_support: float = 0.75,
    boundary_minimum_direct_support: float = 0.60,
    minimum_eventual_support: float = 0.90,
) -> RootedGrowthCertificate:
    """Reject pollen-rim motion and unsupported paths before measurement.

    A short path around the grain can produce a strong image changepoint without
    being a tube.  Certification therefore requires both material appearance
    and an outward excursion beyond the owner body.  Boundary-censored tubes get
    a modest direct-support allowance because part of their ribbon is unobserved.
    """

    if owner_radius_px <= 0.0 or not np.isfinite(owner_radius_px):
        raise ValueError("owner_radius_px must be finite and positive")
    result = hypothesis.result
    if not result.feasible:
        return RootedGrowthCertificate(False, result.reason)
    if not hypothesis.topology_accepted:
        return RootedGrowthCertificate(False, hypothesis.topology_reason)
    if (
        hypothesis.radial_excursion_px
        < minimum_radial_excursion_radii * owner_radius_px
    ):
        return RootedGrowthCertificate(False, "insufficient-owner-departure")
    required_direct = (
        boundary_minimum_direct_support
        if boundary_censored
        else minimum_direct_support
    )
    if result.direct_support_fraction < required_direct:
        return RootedGrowthCertificate(False, "insufficient-direct-birth-support")
    if result.eventual_support_fraction < minimum_eventual_support:
        return RootedGrowthCertificate(False, "insufficient-eventual-birth-support")
    return RootedGrowthCertificate(True, "owner-rooted-growth-certified")


def certify_owner_path_topology(
    path_xy: np.ndarray,
    owner_center_xy: np.ndarray,
    *,
    owner_radius_px: float,
    minimum_root_radius_factor: float = 0.65,
    maximum_root_radius_factor: float = 1.65,
    minimum_body_clearance_factor: float = 0.65,
    halo_exit_radius_factor: float = 1.55,
    halo_return_radius_factor: float = 1.35,
    maximum_halo_exit_arclength_radii: float = 1.75,
    minimum_radial_excursion_radii: float = 0.65,
) -> OwnerPathTopologyCertificate:
    """Require a retained path to make one prompt departure from its owner.

    Tube walls and the pollen rim have similar local contrast.  Their topology
    differs: a tube begins near the body boundary, leaves the body halo once,
    and remains outside it.  This certificate is evaluated only on the causal
    prefix that the tracker claims is tube, never on unsupported distal points.
    """

    path = np.asarray(path_xy, dtype=np.float64)
    center = np.asarray(owner_center_xy, dtype=np.float64)
    factors = np.asarray(
        [
            minimum_root_radius_factor,
            maximum_root_radius_factor,
            minimum_body_clearance_factor,
            halo_exit_radius_factor,
            halo_return_radius_factor,
            maximum_halo_exit_arclength_radii,
            minimum_radial_excursion_radii,
        ],
        dtype=np.float64,
    )
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 1:
        raise ValueError("path_xy must have shape (points, 2) with at least one point")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("owner_center_xy must contain one finite x/y point")
    if not np.isfinite(path).all():
        raise ValueError("path_xy must contain only finite coordinates")
    if owner_radius_px <= 0.0 or not np.isfinite(owner_radius_px):
        raise ValueError("owner_radius_px must be finite and positive")
    if np.any(factors <= 0.0):
        raise ValueError("owner topology factors must be positive")
    if minimum_root_radius_factor >= maximum_root_radius_factor:
        raise ValueError("root radius factors must be ordered")
    if halo_return_radius_factor >= halo_exit_radius_factor:
        raise ValueError("halo return radius must be smaller than the exit radius")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distance = np.linalg.norm(path - center[None], axis=1)
    root_distance = float(distance[0])
    minimum_distance = float(np.min(distance))
    maximum_distance = float(np.max(distance))
    radial_excursion = max(0.0, maximum_distance - root_distance)
    exit_points = np.flatnonzero(
        distance >= halo_exit_radius_factor * owner_radius_px
    )
    exit_index = int(exit_points[0]) if len(exit_points) else None
    exit_arc = float(arc[exit_index]) if exit_index is not None else None

    reasons = []
    if root_distance < minimum_root_radius_factor * owner_radius_px:
        reasons.append("root-inside-owner-body")
    if root_distance > maximum_root_radius_factor * owner_radius_px:
        reasons.append("root-detached-from-owner")
    if minimum_distance < minimum_body_clearance_factor * owner_radius_px:
        reasons.append("path-enters-owner-body")
    if exit_index is None:
        reasons.append("path-never-exits-owner-halo")
    else:
        if exit_arc > maximum_halo_exit_arclength_radii * owner_radius_px:
            reasons.append("tangential-owner-rim-walk")
        if np.any(
            distance[exit_index + 1 :]
            < halo_return_radius_factor * owner_radius_px
        ):
            reasons.append("path-reenters-owner-halo")
    if radial_excursion < minimum_radial_excursion_radii * owner_radius_px:
        reasons.append("insufficient-retained-owner-departure")

    return OwnerPathTopologyCertificate(
        accepted=not reasons,
        reason=(
            "owner-path-topology-certified" if not reasons else ";".join(reasons)
        ),
        root_distance_px=root_distance,
        minimum_distance_px=minimum_distance,
        maximum_distance_px=maximum_distance,
        radial_excursion_px=radial_excursion,
        first_halo_exit_index=exit_index,
        first_halo_exit_arclength_px=exit_arc,
    )


def select_decisive_causal_path(
    hypotheses: list[CausalPathHypothesis],
    *,
    owner_radius_px: float,
    maximum_root_distance_radii: float = 2.0,
    minimum_radial_excursion_radii: float = 2.0,
    contact_minimum_radial_excursion_radii: float = 1.0,
    minimum_direct_support: float = 0.75,
    minimum_eventual_support: float = 0.90,
    maximum_weak_baseline_coverage: float = 0.25,
    minimum_length_gain_radii: float = 4.0,
    contact_minimum_length_gain_radii: float = 1.25,
    minimum_normalized_score_gain: float = 0.75,
    contact_minimum_normalized_score_gain: float = 0.20,
) -> CausalPathSelection:
    """Rescue a failed baseline only with overwhelming owner-rooted evidence.

    The first hypothesis is the current field geometry.  Later hypotheses may
    replace it only when that baseline covers almost none of its mature path or
    has poor direct birth support.  This asymmetry prevents an attractive long
    neighbouring tube from displacing a healthy owner path.
    """

    if not hypotheses:
        raise ValueError("at least one causal path hypothesis is required")
    if owner_radius_px <= 0.0 or not np.isfinite(owner_radius_px):
        raise ValueError("owner_radius_px must be finite and positive")

    def coverage(hypothesis: CausalPathHypothesis) -> float:
        if hypothesis.total_length_px <= 0.0:
            return 0.0
        return hypothesis.result.final_length_px / hypothesis.total_length_px

    def normalized_score(hypothesis: CausalPathHypothesis) -> float:
        reached = max(1, hypothesis.result.final_point_index + 1)
        return hypothesis.result.objective_score / np.sqrt(reached)

    baseline = hypotheses[0]
    baseline_is_weak = bool(
        not baseline.topology_accepted
        or not baseline.result.feasible
        or coverage(baseline) <= maximum_weak_baseline_coverage
        or baseline.result.direct_support_fraction < 0.50
    )
    if not baseline_is_weak:
        return CausalPathSelection(0, "baseline-causally-supported")

    maximum_root_distance = maximum_root_distance_radii * owner_radius_px
    baseline_score = normalized_score(baseline)
    eligible: list[tuple[float, int]] = []
    for index, hypothesis in enumerate(hypotheses[1:], start=1):
        result = hypothesis.result
        score = normalized_score(hypothesis)
        required_excursion = owner_radius_px * (
            contact_minimum_radial_excursion_radii
            if hypothesis.terminates_at_foreign_owner
            else minimum_radial_excursion_radii
        )
        required_length_gain = owner_radius_px * (
            contact_minimum_length_gain_radii
            if hypothesis.terminates_at_foreign_owner
            else minimum_length_gain_radii
        )
        required_score_gain = (
            contact_minimum_normalized_score_gain
            if hypothesis.terminates_at_foreign_owner
            else minimum_normalized_score_gain
        )
        if (
            hypothesis.topology_accepted
            and result.feasible
            and hypothesis.root_distance_px <= maximum_root_distance
            and hypothesis.radial_excursion_px >= required_excursion
            and result.direct_support_fraction >= minimum_direct_support
            and result.eventual_support_fraction >= minimum_eventual_support
            and result.final_length_px
            >= baseline.result.final_length_px + required_length_gain
            and score >= baseline_score + required_score_gain
        ):
            eligible.append((score, index))
    if not eligible:
        return CausalPathSelection(0, "no-decisive-causal-rescue")
    _, selected = max(eligible)
    return CausalPathSelection(selected, "decisive-owner-rooted-causal-rescue")


def project_inextensible_paths(
    paths_xy: np.ndarray,
    material_arclength_px: np.ndarray,
) -> np.ndarray:
    """Project deformable curves onto fixed neighboring material distances."""

    paths = np.asarray(paths_xy, dtype=np.float64)
    material_arc = np.asarray(material_arclength_px, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError("paths_xy must have shape (time, path points, 2)")
    if material_arc.shape != (paths.shape[1],) or np.any(np.diff(material_arc) <= 0):
        raise ValueError("material arclength must strictly increase along the path")
    if not np.isfinite(paths).all() or not np.isfinite(material_arc).all():
        raise ValueError("paths and material arclength must be finite")

    normalized_material = material_arc / material_arc[-1]
    projected = np.empty_like(paths)
    for sample, path in enumerate(paths):
        geometric_arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        if geometric_arc[-1] <= 1e-9:
            raise ValueError("deformed path cannot have zero length")
        query = normalized_material * geometric_arc[-1]
        guide = np.column_stack(
            [np.interp(query, geometric_arc, path[:, axis]) for axis in range(2)]
        )
        projected[sample, 0] = guide[0]
        fallback = np.array([1.0, 0.0])
        for point in range(1, len(guide)):
            direction = guide[point] - guide[point - 1]
            magnitude = float(np.linalg.norm(direction))
            if magnitude > 1e-9:
                fallback = direction / magnitude
            step = material_arc[point] - material_arc[point - 1]
            projected[sample, point] = projected[sample, point - 1] + fallback * step
    return projected


def dynamic_path_novelty_profiles(
    evidence: np.ndarray,
    paths_xy: np.ndarray,
    *,
    warmup_samples: int,
    absolute_evidence_floor: float,
    normal_halfwidth_px: float,
    normal_sample_count: int,
    minimum_change: float,
    noise_multiplier: float,
    temporal_median_samples: int,
) -> np.ndarray:
    """Sample warmup-relative evidence along a moving material path."""

    values = np.asarray(evidence)
    paths = np.asarray(paths_xy, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("evidence must have shape (time, height, width)")
    if paths.ndim != 3 or paths.shape[0] != len(values) or paths.shape[2] != 2:
        raise ValueError("paths_xy must have shape (time, path points, 2)")
    if paths.shape[1] < 2:
        raise ValueError("paths_xy must contain at least two points")
    if not 1 <= warmup_samples <= len(values):
        raise ValueError("warmup_samples must fit inside the evidence sequence")
    if normal_sample_count < 1:
        raise ValueError("normal_sample_count must be positive")

    tangents = np.gradient(paths, axis=1)
    tangent_norms = np.linalg.norm(tangents, axis=2, keepdims=True)
    tangents /= np.maximum(tangent_norms, 1e-9)
    normals = np.stack((-tangents[..., 1], tangents[..., 0]), axis=2)
    normal_offsets = np.linspace(
        -normal_halfwidth_px,
        normal_halfwidth_px,
        normal_sample_count,
    )
    coordinates = (
        paths[:, :, None, :]
        + normals[:, :, None, :] * normal_offsets[None, None, :, None]
    )
    x_float = coordinates[..., 0]
    y_float = coordinates[..., 1]
    valid = (
        (x_float >= 0.0)
        & (x_float <= values.shape[2] - 1)
        & (y_float >= 0.0)
        & (y_float <= values.shape[1] - 1)
    )
    x = np.clip(np.rint(x_float).astype(np.int32), 0, values.shape[2] - 1)
    y = np.clip(np.rint(y_float).astype(np.int32), 0, values.shape[1] - 1)
    time = np.arange(len(values), dtype=np.int32)[:, None, None]
    sampled = values[time, y, x].astype(np.float32)

    warmup = sampled[:warmup_samples]
    warmup_valid = valid[:warmup_samples]
    valid_count = np.sum(warmup_valid, axis=0)
    safe_warmup = np.where(warmup_valid, warmup, 0.0)
    baseline = np.sum(safe_warmup, axis=0) / np.maximum(valid_count, 1)
    absolute_deviation = np.where(
        warmup_valid,
        np.abs(warmup - baseline[None, :, :]),
        0.0,
    )
    # A mean absolute deviation is stable with the deliberately short warmup
    # needed to retain early germinations; 1.253 converts it to a Gaussian
    # standard-deviation estimate.
    noise = 1.253 * np.sum(absolute_deviation, axis=0) / np.maximum(valid_count, 1)
    change_floor = np.maximum(minimum_change, noise_multiplier * noise)
    threshold = np.maximum(absolute_evidence_floor, baseline + change_floor)
    scale = np.maximum(2.0, change_floor)
    normalized = (sampled - threshold[None, :, :]) / scale[None, :, :]
    normalized[~valid] = -np.inf
    profiles = np.max(normalized, axis=2)
    profiles[~np.any(valid, axis=2)] = -1.0

    temporal_window = max(1, int(temporal_median_samples))
    if temporal_window % 2 == 0:
        temporal_window += 1
    if temporal_window > 1:
        profiles = median_filter(
            profiles,
            size=(temporal_window, 1),
            mode="nearest",
        )
    return np.tanh(profiles).astype(np.float32)


def fit_scalar_path_deformation(
    guide: np.ndarray,
    base_paths_xy: np.ndarray,
    *,
    normal_radius_px: float = 4.0,
    normal_step_px: float = 1.0,
    pairwise_smoothness: float = 0.24,
    pairwise_truncation_px: float = 3.0,
    offset_magnitude_penalty: float = 0.08,
    unsupported_cost: float = 1.0,
    root_lock_points: int = 2,
    maximum_cycles: int = 8,
    temporal_smoothing_sigma: float = 1.0,
    material_smoothing_sigma: float = 2.0,
) -> ScalarPathDeformationResult:
    """Fit one topology-preserving normal displacement surface.

    The label grid is coupled across both time and material coordinate.  A
    magnitude prior keeps not-yet-visible material on its pollen-translated
    prediction, while supported tube material can bend inside the narrow band.
    """

    evidence = np.asarray(guide, dtype=np.float32)
    paths = np.asarray(base_paths_xy, dtype=np.float64)
    if evidence.ndim != 3:
        raise ValueError("guide must have shape (time, height, width)")
    if paths.ndim != 3 or paths.shape[0] != len(evidence) or paths.shape[2] != 2:
        raise ValueError("base_paths_xy must have shape (time, path points, 2)")
    if paths.shape[1] < 2 or not np.isfinite(paths).all():
        raise ValueError("base paths must contain finite ordered curves")
    if not np.isfinite(evidence).all():
        raise ValueError("guide must be finite")
    if (
        normal_radius_px < 0.0
        or normal_step_px <= 0.0
        or pairwise_smoothness < 0.0
        or pairwise_truncation_px <= 0.0
        or offset_magnitude_penalty < 0.0
        or unsupported_cost <= 0.0
        or maximum_cycles < 1
        or temporal_smoothing_sigma < 0.0
        or material_smoothing_sigma < 0.0
    ):
        raise ValueError("deformation constraints must be positive")

    evidence = np.clip(evidence, 0.0, 1.0)
    tangent = np.gradient(paths, axis=1)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=2, keepdims=True), 1e-9)
    normals = np.stack((-tangent[..., 1], tangent[..., 0]), axis=2)
    label_count_each_side = int(np.floor(normal_radius_px / normal_step_px))
    labels_px = normal_step_px * np.arange(
        -label_count_each_side,
        label_count_each_side + 1,
        dtype=np.float64,
    )
    center_label = int(np.argmin(np.abs(labels_px)))
    support = np.zeros(paths.shape[:2] + (len(labels_px),), dtype=np.float32)
    height, width = evidence.shape[1:]
    for label, offset in enumerate(labels_px):
        points = paths + normals * offset
        for sample in range(len(evidence)):
            support[sample, :, label] = cv.remap(
                evidence[sample],
                points[sample, :, 0][None, :].astype(np.float32),
                points[sample, :, 1][None, :].astype(np.float32),
                interpolation=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0.0,
            )[0]
    unary = unsupported_cost * (1.0 - support.astype(np.float64))
    unary += offset_magnitude_penalty * np.abs(labels_px)[None, None, :]
    locked = min(max(0, int(root_lock_points)), paths.shape[1])
    if locked:
        unary[:, :locked, :] += unsupported_cost * 8.0
        unary[:, :locked, center_label] = unsupported_cost * (
            1.0 - support[:, :locked, center_label]
        )
    pairwise = pairwise_smoothness * np.minimum(
        np.abs(labels_px[:, None] - labels_px[None, :]),
        pairwise_truncation_px,
    )
    initial_labels = np.full(paths.shape[:2], center_label, dtype=np.int8)
    try:
        from maxflow.fastmin import aexpansion_grid, energy_of_grid_labeling
    except ImportError as error:
        raise RuntimeError(
            "Scalar path deformation requires the research dependencies"
        ) from error
    initial_energy = float(
        energy_of_grid_labeling(unary, pairwise, initial_labels)
    )
    fitted_labels = aexpansion_grid(
        unary,
        pairwise,
        labels=initial_labels.copy(),
        max_cycles=int(maximum_cycles),
    )
    final_energy = float(
        energy_of_grid_labeling(unary, pairwise, fitted_labels)
    )
    offsets = labels_px[fitted_labels]
    offsets = gaussian_filter(
        offsets,
        sigma=(temporal_smoothing_sigma, material_smoothing_sigma),
        mode="nearest",
    )
    if locked:
        offsets[:, :locked] = 0.0
    curves = paths + normals * offsets[:, :, None]
    chosen_support = np.take_along_axis(
        support,
        fitted_labels[..., None],
        axis=2,
    )[..., 0]
    return ScalarPathDeformationResult(
        curves_xy=curves,
        offsets_px=offsets,
        support=chosen_support,
        initial_energy=initial_energy,
        final_energy=final_energy,
    )


def _changepoint_emissions(
    profiles: np.ndarray,
    *,
    warmup_samples: int,
    change_window_samples: int,
    persistence_window_samples: int,
    minimum_post_samples: int,
    coverage_cost: float,
    change_weight: float,
    support_weight: float,
    persistence_weight: float,
) -> np.ndarray:
    """Score each possible first-appearance time for every path point."""

    sample_count, point_count = profiles.shape
    emissions = np.full((sample_count, point_count), -np.inf, dtype=np.float64)
    for sample in range(warmup_samples, sample_count - minimum_post_samples + 1):
        before = profiles[max(0, sample - change_window_samples) : sample]
        after = profiles[sample : min(sample_count, sample + change_window_samples)]
        persistent = profiles[
            sample : min(sample_count, sample + persistence_window_samples)
        ]
        if len(after) < minimum_post_samples or not len(before):
            continue
        confidence = min(1.0, len(after) / change_window_samples)
        before_mean = np.mean(before, axis=0)
        after_mean = np.mean(after, axis=0)
        persistent_mean = np.mean(persistent, axis=0)
        emissions[sample] = confidence * (
            change_weight * (after_mean - before_mean)
            + support_weight * after_mean
            + persistence_weight * persistent_mean
            - coverage_cost
        )
    return emissions


def causal_changepoint_front(
    profiles: np.ndarray,
    arclength_px: np.ndarray,
    *,
    warmup_samples: int,
    max_step_px: float,
    change_window_samples: int = 9,
    persistence_window_samples: int = 27,
    minimum_post_samples: int = 3,
    coverage_cost: float = 0.35,
    change_weight: float = 0.65,
    support_weight: float = 0.25,
    persistence_weight: float = 0.10,
    support_window_samples: int = 12,
) -> CausalGrowthFrontResult:
    """Fit an optional, root-connected growth prefix from appearance events.

    Dynamic programming advances through time and material arclength together.
    A path point contributes only at its fitted birth sample, and a transition
    cannot cover more than ``max_step_px``.  The best final state may stop before
    the mature endpoint when the remaining path lacks a causally ordered birth.
    """

    values = np.asarray(profiles, dtype=np.float32)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    sample_count = len(values) if values.ndim >= 1 else 0
    point_count = values.shape[1] if values.ndim == 2 else 0
    empty_births = np.full(point_count, sample_count, dtype=np.int32)
    empty_front = np.full(sample_count, -1, dtype=np.int32)

    def failed(reason: str) -> CausalGrowthFrontResult:
        return CausalGrowthFrontResult(
            birth_samples=empty_births.copy(),
            front_indices=empty_front.copy(),
            direct_support_mask=np.zeros(point_count, dtype=bool),
            eventual_support_mask=np.zeros(point_count, dtype=bool),
            feasible=False,
            reason=reason,
            objective_score=0.0,
            final_point_index=-1,
            final_length_px=0.0,
            direct_support_fraction=0.0,
            eventual_support_fraction=0.0,
        )

    if values.ndim != 2 or point_count < 2 or arclength.shape != (point_count,):
        return failed("incompatible-front-arrays")
    if not 1 <= warmup_samples < sample_count:
        return failed("invalid-warmup")
    if (
        max_step_px <= 0.0
        or change_window_samples < 1
        or persistence_window_samples < 1
        or minimum_post_samples < 1
        or support_window_samples < 1
        or coverage_cost < 0.0
    ):
        return failed("invalid-front-constraints")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(arclength)):
        return failed("nonfinite-front-arrays")
    if np.any(np.diff(arclength) < 0.0):
        return failed("nonmonotone-arclength")

    emissions = _changepoint_emissions(
        values,
        warmup_samples=warmup_samples,
        change_window_samples=change_window_samples,
        persistence_window_samples=persistence_window_samples,
        minimum_post_samples=minimum_post_samples,
        coverage_cost=coverage_cost,
        change_weight=change_weight,
        support_weight=support_weight,
        persistence_weight=persistence_weight,
    )
    state_lengths = np.concatenate(([0.0], arclength))
    state_count = point_count + 1
    row_count = sample_count - warmup_samples
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    parents = np.full((row_count, state_count), -1, dtype=np.int32)

    for row, sample in enumerate(range(warmup_samples, sample_count)):
        current = previous.copy()
        parents[row] = np.arange(state_count, dtype=np.int32)
        finite_emissions = np.isfinite(emissions[sample])
        for state in range(1, state_count):
            minimum_length = state_lengths[state] - max_step_px
            low_parent = int(
                np.searchsorted(state_lengths, minimum_length, side="left")
            )
            for parent in range(low_parent, state):
                newly_born = emissions[sample, parent:state]
                if not np.all(finite_emissions[parent:state]):
                    continue
                score = previous[parent] + float(np.sum(newly_born))
                if score > current[state]:
                    current[state] = score
                    parents[row, state] = parent
        previous = current

    final_state = int(np.argmax(previous))
    if final_state == 0 or not np.isfinite(previous[final_state]):
        return failed("no-positive-causal-prefix")

    states = np.zeros(row_count, dtype=np.int32)
    states[-1] = final_state
    for row in range(row_count - 1, 0, -1):
        parent = int(parents[row, states[row]])
        if parent < 0:
            return failed("causal-front-backtrack-failed")
        states[row - 1] = parent

    front_indices = empty_front.copy()
    front_indices[warmup_samples:] = states - 1
    births = empty_births.copy()
    for point in range(final_state):
        arrivals = np.flatnonzero(front_indices >= point)
        if not len(arrivals):
            return failed("causal-front-missing-birth")
        births[point] = int(arrivals[0])

    reached = births < sample_count
    direct_support = np.zeros(point_count, dtype=bool)
    eventual_support = np.zeros(point_count, dtype=bool)
    for point in np.flatnonzero(reached):
        arrival = int(births[point])
        direct_support[point] = values[arrival, point] >= 0.0
        stop = min(sample_count, arrival + support_window_samples)
        eventual_support[point] = np.any(values[arrival:stop, point] >= 0.0)

    lengths = np.where(
        front_indices >= 0,
        arclength[np.clip(front_indices, 0, point_count - 1)],
        0.0,
    )
    if np.any(np.diff(lengths) > max_step_px + 1e-6):
        return failed("causal-front-step-limit-violated")

    return CausalGrowthFrontResult(
        birth_samples=births,
        front_indices=front_indices,
        direct_support_mask=direct_support,
        eventual_support_mask=eventual_support,
        feasible=True,
        reason="ok" if final_state == point_count else "partial-causal-prefix",
        objective_score=float(previous[final_state]),
        final_point_index=final_state - 1,
        final_length_px=float(arclength[final_state - 1]),
        direct_support_fraction=float(np.mean(direct_support[reached])),
        eventual_support_fraction=float(np.mean(eventual_support[reached])),
    )
