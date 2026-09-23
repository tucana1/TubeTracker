"""Tube-intrinsic space-time growth-front measurement.

Builds an (arclength x time) kymograph for one traced pollen tube by sampling
ridge evidence along each frame's own accepted centerline plus a ridge-traced
extension corridor beyond the final tip, then extracts tube length over time
as a globally optimal monotone non-decreasing front via dynamic programming.

Why this exists (research prototype, validated on run P0034 only):
- Growth is monotone: material is only added at the distal tip. In (s, t)
  coordinates the tip is therefore a monotone staircase, and offline we can
  fit it globally instead of making irreversible per-frame decisions.
- Blur is transient: the front integrates evidence over the whole video, so
  a multi-frame blur burst does not freeze or truncate the measurement.
- Crossing tubes are foreign material: they arrive in the corridor as
  disconnected blocks (or with arrival times inconsistent with a front
  advancing from the root), so evidence above the front quantifies
  contamination instead of silently annexing a neighbour.

Requires an existing material-pipeline run (per-frame centerlines +
measurement table) and the cached grays / pollen body tracks.

Usage:
    python scripts/prototype_growth_front.py \
        --cache-dir runs/cache/P0034 --run-dir runs/current/P0034 \
        --output-dir runs/research/P0034
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, median_filter


@dataclass(frozen=True)
class PathLockedFrontResult:
    """A monotone temporal front constrained to one immutable tube path."""

    birth_samples: np.ndarray
    front_indices: np.ndarray
    direct_support_mask: np.ndarray
    eventual_support_mask: np.ndarray
    feasible: bool
    reason: str
    eventual_support_fraction: float
    direct_support_fraction: float
    inferred_fraction: float
    median_confirmation_lag_samples: float
    max_confirmation_lag_samples: int


def path_novelty_profiles(
    evidence: np.ndarray,
    path_xy: np.ndarray,
    warmup_samples: int,
    absolute_evidence_floor: float,
    normal_halfwidth_px: float,
    normal_sample_count: int,
    minimum_change: float,
    noise_multiplier: float,
    temporal_median_samples: int,
    path_offsets_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Sample warmup-relative evidence along a fixed, optionally deformed path.

    ``path_offsets_xy`` may contain one translation per timepoint or one offset
    per timepoint and path point. The latter preserves point order while a
    separately registered material curve bends within its original corridor.
    """

    path = np.asarray(path_xy, dtype=np.float64)
    if evidence.ndim != 3:
        raise ValueError("evidence must have shape (time, height, width)")
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_xy must contain at least two x/y points")
    if not 1 <= warmup_samples <= len(evidence):
        raise ValueError("warmup_samples must fit inside the evidence sequence")
    if normal_sample_count < 1:
        raise ValueError("normal_sample_count must be positive")
    if path_offsets_xy is None:
        path_offsets = np.zeros((len(evidence), len(path), 2), dtype=np.float64)
    else:
        path_offsets = np.asarray(path_offsets_xy, dtype=np.float64)
        if path_offsets.shape == (len(evidence), 2):
            path_offsets = np.broadcast_to(
                path_offsets[:, None, :],
                (len(evidence), len(path), 2),
            )
        elif path_offsets.shape != (len(evidence), len(path), 2):
            raise ValueError(
                "path_offsets_xy must have shape (time, 2) or "
                "(time, path points, 2)"
            )

    tangents = np.gradient(path, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-9)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    offsets = np.linspace(
        -normal_halfwidth_px,
        normal_halfwidth_px,
        normal_sample_count,
    )
    coordinates = (
        path[None, :, None, :]
        + normals[None, :, None, :] * offsets[None, None, :, None]
        + path_offsets[:, :, None, :]
    )
    x = np.clip(
        np.rint(coordinates[..., 0]).astype(np.int32),
        0,
        evidence.shape[2] - 1,
    )
    y = np.clip(
        np.rint(coordinates[..., 1]).astype(np.int32),
        0,
        evidence.shape[1] - 1,
    )
    time = np.arange(len(evidence), dtype=np.int32)[:, None, None]
    sampled = evidence[time, y, x].astype(np.float32)
    warmup = sampled[:warmup_samples]
    baseline = np.median(warmup, axis=0)
    noise = 1.4826 * np.median(np.abs(warmup - baseline), axis=0)
    change_floor = np.maximum(minimum_change, noise_multiplier * noise)
    threshold = np.maximum(absolute_evidence_floor, baseline + change_floor)
    scale = np.maximum(2.0, change_floor)
    normalized_margin = (sampled - threshold[None, :, :]) / scale[None, :, :]

    # The centerline can lie on either wall of a thin ribbon. A maximum over a
    # narrow normal cross-section preserves that tube while each offset keeps
    # its own warmup baseline, so stationary crossing material contributes no
    # novelty merely because it is dark.
    profiles = np.max(normalized_margin, axis=2)
    temporal_window = max(1, int(temporal_median_samples))
    if temporal_window % 2 == 0:
        temporal_window += 1
    if temporal_window > 1:
        profiles = median_filter(profiles, size=(temporal_window, 1), mode="nearest")
    return np.tanh(profiles).astype(np.float32)


def visible_path_front(
    profiles: np.ndarray,
    arclength_px: np.ndarray,
    *,
    warmup_samples: int,
    max_step_px: float,
    coverage_cost: float,
    absence_weight: float,
    motion_penalty: float,
    support_window_samples: int,
    require_final_endpoint: bool = True,
) -> PathLockedFrontResult:
    """Fit the connected visible endpoint along one immutable tube branch.

    The state is a root-connected prefix of the supplied path. It can only
    stay still or move outward by ``max_step_px`` per sample, so a crossing
    beyond an unsupported gap cannot become the tip on its own. Unlike the
    construction-time model, this estimator does not move an observation
    backward from a later confirmation; it reports the front supported by the
    image sequence at that time.
    """

    profile_values = np.asarray(profiles, dtype=np.float32)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    sample_count = len(profile_values)
    point_count = profile_values.shape[1] if profile_values.ndim == 2 else 0
    empty_front = np.full(sample_count, -1, dtype=np.int32)
    empty_births = np.full(point_count, sample_count, dtype=np.int32)

    def failed(reason: str) -> PathLockedFrontResult:
        return PathLockedFrontResult(
            birth_samples=empty_births.copy(),
            front_indices=empty_front.copy(),
            direct_support_mask=np.zeros(point_count, dtype=bool),
            eventual_support_mask=np.zeros(point_count, dtype=bool),
            feasible=False,
            reason=reason,
            eventual_support_fraction=0.0,
            direct_support_fraction=0.0,
            inferred_fraction=0.0,
            median_confirmation_lag_samples=0.0,
            max_confirmation_lag_samples=0,
        )

    if (
        profile_values.ndim != 2
        or point_count < 2
        or arclength.shape != (point_count,)
    ):
        return failed("incompatible-profile-arrays")
    if not 1 <= warmup_samples < sample_count:
        return failed("invalid-visible-front-warmup")
    if max_step_px <= 0 or coverage_cost < 0 or absence_weight < 0:
        return failed("invalid-visible-front-constraints")
    if motion_penalty < 0 or support_window_samples < 1:
        return failed("invalid-visible-front-regularization")
    if np.any(np.diff(arclength) < 0):
        return failed("nonmonotone-arclength")

    utility = np.where(
        profile_values >= 0.0,
        profile_values,
        profile_values * absence_weight,
    ) - coverage_cost
    emission = np.zeros((sample_count, point_count + 1), dtype=np.float32)
    emission[:, 1:] = np.cumsum(utility, axis=1)
    state_lengths = np.concatenate(([0.0], arclength))
    state_count = len(state_lengths)
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    row_count = sample_count - warmup_samples
    parents = np.full((row_count, state_count), -1, dtype=np.int32)

    for row, sample in enumerate(range(warmup_samples, sample_count)):
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state in range(state_count):
            minimum_length = state_lengths[state] - max_step_px
            low_state = int(
                np.searchsorted(state_lengths, minimum_length, side="left")
            )
            candidates = previous[low_state : state + 1]
            if not np.any(np.isfinite(candidates)):
                continue
            prior_states = np.arange(low_state, state + 1)
            movement = state_lengths[state] - state_lengths[prior_states]
            transition = candidates - motion_penalty * (
                movement / max_step_px
            ) ** 2
            local_best = int(np.argmax(transition))
            parent = low_state + local_best
            current[state] = transition[local_best] + emission[sample, state]
            parents[row, state] = parent
        previous = current

    final_state = point_count if require_final_endpoint else int(np.argmax(previous))
    if not np.isfinite(previous[final_state]):
        return failed("visible-front-constraints-infeasible")
    states = np.zeros(row_count, dtype=np.int32)
    states[-1] = final_state
    for row in range(row_count - 1, 0, -1):
        parent = parents[row, states[row]]
        if parent < 0:
            return failed("visible-front-backtrack-failed")
        states[row - 1] = parent

    front_indices = np.full(sample_count, -1, dtype=np.int32)
    front_indices[warmup_samples:] = states - 1
    births = np.full(point_count, sample_count, dtype=np.int32)
    for point in range(point_count):
        arrivals = np.flatnonzero(front_indices >= point)
        if len(arrivals):
            births[point] = int(arrivals[0])
    reached = births < sample_count
    if require_final_endpoint and not np.all(reached):
        return failed("visible-front-does-not-reach-endpoint")

    direct_support = np.zeros(point_count, dtype=bool)
    eventual_support = np.zeros(point_count, dtype=bool)
    for point in np.flatnonzero(reached):
        arrival = int(births[point])
        stop = min(sample_count, arrival + support_window_samples)
        direct_support[point] = profile_values[arrival, point] >= 0.0
        eventual_support[point] = np.any(
            profile_values[arrival:stop, point] >= 0.0
        )
    reached_count = int(np.count_nonzero(reached))
    direct_fraction = (
        float(np.mean(direct_support[reached])) if reached_count else 0.0
    )
    eventual_fraction = (
        float(np.mean(eventual_support[reached])) if reached_count else 0.0
    )
    inferred_fraction = (
        float(np.mean(~direct_support[reached] & eventual_support[reached]))
        if reached_count
        else 0.0
    )
    return PathLockedFrontResult(
        birth_samples=births,
        front_indices=front_indices,
        direct_support_mask=direct_support,
        eventual_support_mask=eventual_support,
        feasible=True,
        reason="ok" if reached_count == point_count else "partial-prefix",
        eventual_support_fraction=eventual_fraction,
        direct_support_fraction=direct_fraction,
        inferred_fraction=inferred_fraction,
        median_confirmation_lag_samples=0.0,
        max_confirmation_lag_samples=0,
    )


def confirmation_bounded_path_front(
    profiles: np.ndarray,
    arclength_px: np.ndarray,
    confirmation_samples: np.ndarray,
    earliest_samples: np.ndarray,
    *,
    warmup_samples: int,
    max_step_px: float,
    coverage_cost: float = 0.08,
    absence_weight: float = 0.20,
    motion_penalty: float = 0.18,
    support_window_samples: int = 0,
    support_deadline_samples: np.ndarray | None = None,
) -> PathLockedFrontResult:
    """Recover dense growth between sparse whole-path confirmations.

    Each path point has a latest arrival supplied by the first independently
    accepted centerline that contains it. Its earliest arrival is the previous
    accepted centerline, or the end of warmup for the first observed segment.
    The fitted state is always one root-connected prefix and can only advance,
    which prevents a detached distal structure from becoming a tube tip.
    """

    values = np.asarray(profiles, dtype=np.float32)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    confirmations = np.asarray(confirmation_samples, dtype=np.int32)
    earliest = np.asarray(earliest_samples, dtype=np.int32)
    deadlines = (
        np.asarray(support_deadline_samples, dtype=np.int32)
        if support_deadline_samples is not None
        else None
    )
    sample_count = len(values)
    point_count = values.shape[1] if values.ndim == 2 else 0
    empty_front = np.full(sample_count, -1, dtype=np.int32)

    def failed(reason: str) -> PathLockedFrontResult:
        return PathLockedFrontResult(
            birth_samples=confirmations.copy(),
            front_indices=empty_front.copy(),
            direct_support_mask=np.zeros(point_count, dtype=bool),
            eventual_support_mask=np.zeros(point_count, dtype=bool),
            feasible=False,
            reason=reason,
            eventual_support_fraction=0.0,
            direct_support_fraction=0.0,
            inferred_fraction=0.0,
            median_confirmation_lag_samples=0.0,
            max_confirmation_lag_samples=0,
        )

    if (
        values.ndim != 2
        or point_count < 2
        or arclength.shape != (point_count,)
        or confirmations.shape != (point_count,)
        or earliest.shape != (point_count,)
        or (deadlines is not None and deadlines.shape != (point_count,))
    ):
        return failed("incompatible-confirmation-front-arrays")
    if not 1 <= warmup_samples < sample_count:
        return failed("invalid-confirmation-front-warmup")
    if max_step_px <= 0 or coverage_cost < 0 or absence_weight < 0:
        return failed("invalid-confirmation-front-constraints")
    if motion_penalty < 0 or support_window_samples < 0:
        return failed("invalid-confirmation-front-regularization")
    if np.any(np.diff(arclength) < 0):
        return failed("nonmonotone-arclength")
    if np.any(np.diff(confirmations) < 0) or np.any(np.diff(earliest) < 0):
        return failed("nonmonotone-confirmation-window")
    if np.any(earliest < warmup_samples) or np.any(earliest > confirmations):
        return failed("invalid-confirmation-window")
    if np.any(confirmations >= sample_count):
        return failed("confirmation-outside-timeline")
    if deadlines is not None and (
        np.any(deadlines < confirmations) or np.any(deadlines >= sample_count)
    ):
        return failed("invalid-support-deadline")

    utility = np.where(values >= 0.0, values, values * absence_weight) - coverage_cost
    emission = np.zeros((sample_count, point_count + 1), dtype=np.float32)
    emission[:, 1:] = np.cumsum(utility, axis=1)
    state_lengths = np.concatenate(([0.0], arclength))
    previous = np.full(point_count + 1, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    parents = np.full(
        (sample_count - warmup_samples, point_count + 1),
        -1,
        dtype=np.int32,
    )

    for row, sample in enumerate(range(warmup_samples, sample_count)):
        required_state = int(np.searchsorted(confirmations, sample, side="right"))
        allowed_state = int(np.searchsorted(earliest, sample, side="right"))
        current = np.full(point_count + 1, -np.inf, dtype=np.float64)
        for state in range(required_state, allowed_state + 1):
            minimum_length = state_lengths[state] - max_step_px
            low_state = int(
                np.searchsorted(state_lengths, minimum_length, side="left")
            )
            candidates = previous[low_state : state + 1]
            if not np.any(np.isfinite(candidates)):
                continue
            prior_states = np.arange(low_state, state + 1)
            movement = state_lengths[state] - state_lengths[prior_states]
            transition = candidates - motion_penalty * (movement / max_step_px) ** 2
            local_best = int(np.argmax(transition))
            parent = low_state + local_best
            current[state] = transition[local_best] + emission[sample, state]
            parents[row, state] = parent
        previous = current

    if not np.isfinite(previous[point_count]):
        return failed("confirmation-front-constraints-infeasible")
    states = np.zeros(sample_count - warmup_samples, dtype=np.int32)
    states[-1] = point_count
    for row in range(len(states) - 1, 0, -1):
        parent = parents[row, states[row]]
        if parent < 0:
            return failed("confirmation-front-backtrack-failed")
        states[row - 1] = parent

    front_indices = empty_front.copy()
    front_indices[warmup_samples:] = states - 1
    births = np.full(point_count, sample_count, dtype=np.int32)
    for point in range(point_count):
        arrivals = np.flatnonzero(front_indices >= point)
        if not len(arrivals):
            return failed("confirmation-front-does-not-reach-endpoint")
        births[point] = int(arrivals[0])
    if np.any(births < earliest) or np.any(births > confirmations):
        return failed("confirmation-window-violated")

    lengths = np.where(
        front_indices >= 0,
        arclength[np.clip(front_indices, 0, point_count - 1)],
        0.0,
    )
    if np.any(np.diff(lengths) > max_step_px + 1e-6):
        return failed("tip-step-constraint-violated")

    direct_support = values[births, np.arange(point_count)] >= 0.0
    eventual_support = np.asarray(
        [
            np.any(
                values[
                    arrival : min(
                        sample_count,
                        (
                            deadlines[point]
                            if deadlines is not None
                            else confirmation + support_window_samples
                        )
                        + 1,
                    ),
                    point,
                ]
                >= 0.0
            )
            for point, (arrival, confirmation) in enumerate(
                zip(births, confirmations)
            )
        ],
        dtype=bool,
    )
    lag = confirmations - births
    return PathLockedFrontResult(
        birth_samples=births,
        front_indices=front_indices,
        direct_support_mask=direct_support,
        eventual_support_mask=eventual_support,
        feasible=True,
        reason="ok",
        eventual_support_fraction=float(np.mean(eventual_support)),
        direct_support_fraction=float(np.mean(direct_support)),
        inferred_fraction=float(np.mean(lag > 0)),
        median_confirmation_lag_samples=float(np.median(lag)),
        max_confirmation_lag_samples=int(np.max(lag)),
    )


def path_locked_temporal_front(
    evidence: np.ndarray,
    path_xy: np.ndarray,
    arclength_px: np.ndarray,
    observed_birth_samples: np.ndarray,
    *,
    warmup_samples: int,
    max_step_px: float,
    max_confirmation_lag_samples: int,
    absolute_evidence_floor: float,
    normal_halfwidth_px: float = 2.0,
    normal_sample_count: int = 5,
    minimum_change: float = 4.0,
    noise_multiplier: float = 2.5,
    temporal_median_samples: int = 3,
    absence_weight: float = 0.05,
    motion_penalty: float = 0.25,
    onset_point_index: int | None = None,
    onset_sample: int | None = None,
    path_offsets_xy: np.ndarray | None = None,
) -> PathLockedFrontResult:
    """Fit an evidence-guided tip timeline without changing tube geometry.

    The threshold-derived construction time is treated as a confirmation time:
    a point may be inferred earlier, because tube contrast matures late, but by
    no more than ``max_confirmation_lag_samples``. The front can only move
    outward along ``path_xy`` and cannot advance farther than ``max_step_px`` in
    one sample. Optional per-frame or per-point offsets register the same ordered
    material path without changing its topology. These hard constraints make
    branch switching inexpressible.
    """

    path = np.asarray(path_xy, dtype=np.float64)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    observed = np.asarray(observed_birth_samples, dtype=np.int32)
    empty_front = np.full(len(evidence), -1, dtype=np.int32)
    point_count = len(observed) if observed.ndim == 1 else 0

    def failed(reason: str) -> PathLockedFrontResult:
        return PathLockedFrontResult(
            birth_samples=observed.copy(),
            front_indices=empty_front.copy(),
            direct_support_mask=np.zeros(point_count, dtype=bool),
            eventual_support_mask=np.zeros(point_count, dtype=bool),
            feasible=False,
            reason=reason,
            eventual_support_fraction=0.0,
            direct_support_fraction=0.0,
            inferred_fraction=0.0,
            median_confirmation_lag_samples=0.0,
            max_confirmation_lag_samples=0,
        )

    if (
        path.ndim != 2
        or path.shape[1] != 2
        or len(path) < 2
        or arclength.shape != (len(path),)
        or observed.shape != (len(path),)
    ):
        return failed("incompatible-path-arrays")
    if len(evidence) <= warmup_samples:
        return failed("insufficient-timepoints")
    if max_step_px <= 0 or max_confirmation_lag_samples < 0:
        return failed("invalid-front-constraints")
    if (onset_point_index is None) != (onset_sample is None):
        return failed("incomplete-onset-anchor")
    if onset_point_index is not None and not (
        0 <= onset_point_index < len(path)
        and warmup_samples <= int(onset_sample) < len(evidence)
    ):
        return failed("invalid-onset-anchor")
    if np.any(np.diff(arclength) < 0):
        return failed("nonmonotone-arclength")
    if np.any((observed < warmup_samples) | (observed >= len(evidence))):
        return failed("unobserved-path-point")
    if path_offsets_xy is not None:
        offset_shape = np.asarray(path_offsets_xy).shape
        if offset_shape not in {
            (len(evidence), 2),
            (len(evidence), len(path), 2),
        }:
            return failed("incompatible-path-offsets")

    observed = np.maximum.accumulate(observed)
    profiles = path_novelty_profiles(
        evidence,
        path,
        warmup_samples,
        absolute_evidence_floor,
        normal_halfwidth_px,
        normal_sample_count,
        minimum_change,
        noise_multiplier,
        temporal_median_samples,
        path_offsets_xy,
    )
    utility = np.where(profiles >= 0.0, profiles, profiles * absence_weight)
    state_emission = np.zeros((len(evidence), len(path) + 1), dtype=np.float32)
    state_emission[:, 1:] = np.cumsum(utility, axis=1)

    first_time = max(warmup_samples, int(observed[0]) - max_confirmation_lag_samples)
    last_time = int(observed[-1])
    state_lengths = np.concatenate([[0.0], arclength])
    state_count = len(state_lengths)
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    parents = np.full(
        (last_time - first_time + 1, state_count),
        -1,
        dtype=np.int32,
    )

    for row, sample in enumerate(range(first_time, last_time + 1)):
        required_state = int(np.searchsorted(observed, sample, side="right"))
        allowed_state = int(
            np.searchsorted(
                observed,
                sample + max_confirmation_lag_samples,
                side="right",
            )
        )
        if onset_point_index is not None:
            if sample < int(onset_sample):
                allowed_state = min(allowed_state, onset_point_index)
            else:
                required_state = max(required_state, onset_point_index + 1)
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state in range(required_state, allowed_state + 1):
            minimum_length = state_lengths[state] - max_step_px
            low_state = int(
                np.searchsorted(state_lengths, minimum_length, side="left")
            )
            candidates = previous[low_state : state + 1]
            if not np.any(np.isfinite(candidates)):
                continue
            prior_states = np.arange(low_state, state + 1)
            movement = state_lengths[state] - state_lengths[prior_states]
            transition = candidates - motion_penalty * (movement / max_step_px) ** 2
            local_best = int(np.argmax(transition))
            parent = low_state + local_best
            current[state] = transition[local_best] + state_emission[sample, state]
            parents[row, state] = parent
        previous = current

    final_state = len(path)
    if not np.isfinite(previous[final_state]):
        return failed("front-constraints-infeasible")

    states = np.zeros(last_time - first_time + 1, dtype=np.int32)
    states[-1] = final_state
    for row in range(len(states) - 1, 0, -1):
        parent = parents[row, states[row]]
        if parent < 0:
            return failed("front-backtrack-failed")
        states[row - 1] = parent

    front_indices = np.full(len(evidence), -1, dtype=np.int32)
    front_indices[first_time : last_time + 1] = states - 1
    front_indices[last_time + 1 :] = len(path) - 1
    refined = np.empty(len(path), dtype=np.int32)
    for point in range(len(path)):
        arrivals = np.flatnonzero(front_indices >= point)
        if not len(arrivals):
            return failed("front-does-not-reach-endpoint")
        refined[point] = int(arrivals[0])

    lag = observed - refined
    if np.any(lag < 0) or np.any(lag > max_confirmation_lag_samples):
        return failed("confirmation-lag-constraint-violated")
    front_lengths = np.where(
        front_indices >= 0,
        arclength[np.clip(front_indices, 0, len(arclength) - 1)],
        0.0,
    )
    if np.any(np.diff(front_lengths) > max_step_px + 1e-6):
        return failed("tip-step-constraint-violated")

    direct_support = np.zeros(len(path), dtype=bool)
    eventual_support = np.zeros(len(path), dtype=bool)
    for point, (arrival, confirmation) in enumerate(zip(refined, observed)):
        direct_support[point] = profiles[arrival, point] >= 0.0
        eventual_support[point] = np.any(
            profiles[arrival : confirmation + 1, point] >= 0.0
        )
    return PathLockedFrontResult(
        birth_samples=refined,
        front_indices=front_indices,
        direct_support_mask=direct_support,
        eventual_support_mask=eventual_support,
        feasible=True,
        reason="ok",
        eventual_support_fraction=float(np.mean(eventual_support)),
        direct_support_fraction=float(np.mean(direct_support)),
        inferred_fraction=float(np.mean(lag > 0)),
        median_confirmation_lag_samples=float(np.median(lag)),
        max_confirmation_lag_samples=int(np.max(lag)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arc-step-px", type=float, default=1.5)
    parser.add_argument("--normal-halfwidth-px", type=float, default=4.0)
    parser.add_argument("--extension-steps", type=int, default=40)
    parser.add_argument("--dog-sigma-tube", type=float, default=1.2)
    parser.add_argument("--dog-sigma-background", type=float, default=6.0)
    parser.add_argument("--grain-radius-px", type=float, default=12.0,
                        help="grain radius used when seeding the corridor "
                             "directly from the grain (no accepted centerlines)")
    parser.add_argument("--evidence-mode", choices=("dark", "abs"), default="dark",
                        help="'dark' detects dark-ridge tubes; 'abs' is "
                             "polarity-insensitive (bright-cored tubes with "
                             "dark walls)")
    parser.add_argument("--retrace-per-window", action="store_true",
                        help="retrace the corridor from the root in every time "
                             "window instead of using one rigid corridor; "
                             "needed when the tube sways strongly relative to "
                             "the grain")
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--window-halfwidth", type=int, default=8)
    parser.add_argument("--evidence-midpoint", type=float, default=2.5,
                        help="DoG value treated as neutral evidence")
    parser.add_argument("--coverage-cost", type=float, default=0.15,
                        help="penalty per covered arc step without evidence")
    parser.add_argument("--max-growth-px-per-frame", type=float, default=7.5)
    return parser.parse_args()


def dog(gray: np.ndarray, sigma_tube: float, sigma_background: float) -> np.ndarray:
    g = gray.astype(np.float32)
    return cv.GaussianBlur(g, (0, 0), sigma_background) - cv.GaussianBlur(g, (0, 0), sigma_tube)


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arcs = np.concatenate([[0.0], np.cumsum(deltas)])
    grid = np.arange(0.0, arcs[-1], step)
    return np.stack([np.interp(grid, arcs, points[:, 0]),
                     np.interp(grid, arcs, points[:, 1])], axis=1)


def similarity_to_reference(tracks: np.ndarray, visibility: np.ndarray,
                            frame: int, reference: int) -> np.ndarray:
    joint = visibility[frame] & visibility[reference]
    matrix, _ = cv.estimateAffinePartial2D(tracks[frame][joint], tracks[reference][joint],
                                           method=cv.LMEDS)
    if matrix is None:
        raise RuntimeError(f"pose estimation failed for frame {frame}")
    return matrix


def load_body_tracks(cache_dir: Path, run_dir: Path):
    """Load CoTracker body landmarks converted to SOURCE pixel coordinates.

    The pipeline stores tracks in its fixed body-crop frame; a similarity
    estimated there is wrong to apply to source images whenever the grain
    rotates (translation error ~ rotation x crop offset). The stored source
    'centers' give the crop offset: the center landmark starts at the crop
    center."""
    body_path = cache_dir / "pollen_body_tracks.npz"
    if not body_path.exists():
        body_path = run_dir / "pollen_body_tracks.npz"
    body = np.load(body_path)
    tracks = body["tracks"].astype(np.float64).copy()
    measurements = pd.read_csv(run_dir / "measurements.csv")
    measured = measurements[["pollen_x_px", "pollen_y_px"]].to_numpy()
    finite = np.where(np.isfinite(measured).all(axis=1))[0]
    anchor = measured[finite[0]] if len(finite) else None
    stored = body["centers"][0] if "centers" in body else None
    if stored is not None and anchor is not None \
            and np.linalg.norm(stored - anchor) < 30.0:
        offset = stored - tracks[0, 0]      # centers already in source coords
    elif anchor is not None:
        offset = anchor - tracks[0, 0]      # old cache format: crop-frame centers
    elif stored is not None:
        offset = stored - tracks[0, 0]
    else:
        offset = np.zeros(2)
    tracks += np.asarray(offset)[None, None, :]
    return tracks, body["visibility"], body["chosen_cache_indices"]


def validated_transform(tracks: np.ndarray, visibility: np.ndarray,
                        centers: np.ndarray, frame: int, reference: int,
                        tolerance_px: float = 3.0) -> np.ndarray:
    """Body-track similarity, cross-checked against the measured grain center.

    CoTracker body landmarks can silently drift (observed: 47-76 px on a
    near-static grain while visibility stayed 13/13). The measured center is
    an independent witness: if the similarity maps it more than tolerance_px
    away from its reference position, fall back to pure translation by the
    measured centers."""
    matrix = similarity_to_reference(tracks, visibility, frame, reference)
    center, ref_center = centers[frame], centers[reference]
    if np.all(np.isfinite(center)) and np.all(np.isfinite(ref_center)):
        mapped = cv.transform(center.reshape(1, 1, 2).astype(np.float32), matrix)[0, 0]
        if np.linalg.norm(mapped - ref_center) > tolerance_px:
            shift = ref_center - center
            matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]],
                              np.float64)
    return matrix


def smoothed_pose_transforms(tracks: np.ndarray, visibility: np.ndarray,
                             centers: np.ndarray, frame_count: int,
                             reference: int | None = None,
                             grays: np.ndarray | None = None,
                             cache_indices: np.ndarray | None = None) -> list[np.ndarray]:
    """Temporally consistent frame->reference similarities.

    Per-frame independent estimates flip between good similarities and
    translation fallbacks, which swings any structure far from the grain by
    rotation x lever-arm. Instead: take rotation/scale from the landmark
    similarity, median-filter them over time (kills drift spikes), clamp the
    scale, and set the translation so the measured grain center maps exactly
    onto its reference position."""
    reference = frame_count - 1 if reference is None else reference
    angles = np.zeros(frame_count)
    scales = np.ones(frame_count)
    valid = np.zeros(frame_count, bool)
    for frame in range(frame_count):
        try:
            matrix = similarity_to_reference(tracks, visibility, frame, reference)
        except (RuntimeError, cv.error):
            continue
        # trust rotation only when the similarity maps the independently
        # measured grain center correctly (rejects drifted landmarks)
        center, ref_center = centers[frame], centers[reference]
        if np.all(np.isfinite(center)) and np.all(np.isfinite(ref_center)):
            mapped = cv.transform(center.reshape(1, 1, 2).astype(np.float32),
                                  matrix)[0, 0]
            if np.linalg.norm(mapped - ref_center) > 3.0:
                continue
        angles[frame] = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        scales[frame] = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        valid[frame] = True
    # rotations are only safe when a strong majority of frames validate:
    # a minority of "valid" frames can carry drift-corrupted rotations that
    # interpolation would then spread across the whole video
    if valid.mean() < 0.6:
        angles[:] = 0.0
        scales[:] = 1.0
    else:
        good = np.where(valid)[0]
        bad = np.where(~valid)[0]
        if len(bad):
            angles[bad] = np.interp(bad, good, angles[good])
            scales[bad] = np.interp(bad, good, scales[good])
        angles = median_filter(angles, size=9, mode="nearest")
        scales = np.clip(median_filter(scales, size=9, mode="nearest"), 0.95, 1.05)

    filled = centers.copy().astype(float)
    for axis in range(2):
        column = filled[:, axis]
        good = np.isfinite(column)
        if good.any() and not good.all():
            column[~good] = np.interp(np.where(~good)[0], np.where(good)[0],
                                      column[good])
        elif not good.any():
            column[:] = 0.0
    reference_center = filled[reference]

    def build(all_angles, all_scales):
        result = []
        for frame in range(frame_count):
            cos_a = np.cos(all_angles[frame]) * all_scales[frame]
            sin_a = np.sin(all_angles[frame]) * all_scales[frame]
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            translation = reference_center - rotation @ filled[frame]
            result.append(np.array([[rotation[0, 0], rotation[0, 1], translation[0]],
                                    [rotation[1, 0], rotation[1, 1], translation[1]]]))
        return result

    # image-content arbitration: landmarks sliding around the grain produce
    # spurious rotations about the center that no landmark/center check can
    # catch. Warp sample frames both ways and let patch correlation around
    # the grain decide whether the rotation series is real.
    if grays is not None and cache_indices is not None and np.abs(angles).max() > 0.02:
        # off-center ring patches: rotation about the grain center is invisible
        # in a grain-centered patch, so measure where the lever arm is large.
        # Rotation must WIN CLEARLY; translation-only is the safe default.
        h, w = grays.shape[1], grays.shape[2]
        half = 24
        ring = []
        for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            px = int(round(reference_center[0] + 55 * np.cos(angle)))
            py = int(round(reference_center[1] + 55 * np.sin(angle)))
            if half <= px < w - half and half <= py < h - half:
                ring.append((px, py))
        reference_image = grays[cache_indices[reference]]
        with_rotation = build(angles, scales)
        translation_only = build(np.zeros(frame_count), np.ones(frame_count))
        score_rotation = score_translation = 0.0
        for frame in np.linspace(0, frame_count - 1, 12).astype(int):
            if frame == reference:
                continue
            image = grays[cache_indices[frame]]
            for matrices_candidate, is_rotation in ((with_rotation, True),
                                                    (translation_only, False)):
                warped = cv.warpAffine(image, matrices_candidate[frame], (w, h),
                                       borderValue=255)
                for px, py in ring:
                    patch = warped[py - half:py + half, px - half:px + half].astype(np.float32)
                    target = reference_image[py - half:py + half,
                                             px - half:px + half].astype(np.float32)
                    a = patch - patch.mean()
                    b = target - target.mean()
                    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
                    score = float((a * b).sum() / denom)
                    if is_rotation:
                        score_rotation += score
                    else:
                        score_translation += score
        if score_rotation < score_translation + 0.5:
            angles[:] = 0.0
            scales[:] = 1.0
    return build(angles, scales)


def trace_extension_corridor(ridge: np.ndarray, tip_xy: np.ndarray, tangent_xy: np.ndarray,
                             steps: int, step_px: float = 2.0,
                             bend_penalty: float = 8.0,
                             stop_below: float = 0.5,
                             stop_patience: int = 4) -> np.ndarray:
    """Greedy bending-penalized ridge following beyond the final tip.

    Stops once the ridge response stays below `stop_below` for `stop_patience`
    consecutive steps, so the corridor does not wander onto unrelated
    structures after the tube ends."""
    height, width = ridge.shape
    path = [tip_xy.astype(np.float64)]
    direction = tangent_xy / (np.linalg.norm(tangent_xy) + 1e-9)
    weak_run = 0
    for _ in range(steps):
        best_value, best_raw, best_point, best_direction = -np.inf, -np.inf, None, None
        for angle in np.linspace(-0.5, 0.5, 13):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            candidate_direction = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a,
            ])
            candidate = path[-1] + candidate_direction * step_px
            if not (0 <= candidate[0] < width and 0 <= candidate[1] < height):
                continue
            raw = map_coordinates(ridge, [[candidate[1]], [candidate[0]]], order=1)[0]
            value = raw - bend_penalty * abs(angle)
            if value > best_value:
                best_value, best_raw = value, raw
                best_point, best_direction = candidate, candidate_direction
        if best_point is None:
            break
        weak_run = weak_run + 1 if best_raw < stop_below else 0
        if weak_run >= stop_patience:
            break
        path.append(best_point)
        direction = 0.7 * direction + 0.3 * best_direction
        direction /= np.linalg.norm(direction)
    return np.asarray(path)


def lateral_refit(path: np.ndarray, guide: np.ndarray,
                  search_radius_px: float = 5.0, step_px: float = 1.0,
                  change_penalty: float = 0.6) -> np.ndarray:
    """Re-center an ordered path onto the guide ridge by a Viterbi over
    per-node lateral offsets. The root node (index 0) stays pinned."""
    if len(path) < 3:
        return path.copy()
    tangents = np.gradient(path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    offsets = np.arange(-search_radius_px, search_radius_px + 1e-6, step_px)
    n_nodes, n_states = len(path), len(offsets)
    coords = path[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    values = map_coordinates(
        guide, [coords[..., 1].ravel(), coords[..., 0].ravel()],
        order=1).reshape(n_nodes, n_states)

    zero_state = int(np.argmin(np.abs(offsets)))
    score = np.full((n_nodes, n_states), -np.inf)
    parent = np.zeros((n_nodes, n_states), np.int32)
    score[0, zero_state] = 0.0  # root pinned to the grain attachment
    for node in range(1, n_nodes):
        for state in range(n_states):
            lo, hi = max(0, state - 2), min(n_states, state + 3)
            prev = score[node - 1, lo:hi] - change_penalty * np.abs(
                np.arange(lo, hi) - state) * step_px
            best = int(np.argmax(prev))
            score[node, state] = prev[best] + values[node, state]
            parent[node, state] = lo + best
    states = np.zeros(n_nodes, np.int32)
    states[-1] = int(np.argmax(score[-1]))
    for node in range(n_nodes - 1, 0, -1):
        states[node - 1] = parent[node, states[node]]
    return path + normals * offsets[states][:, None]


def seed_from_grain(guide: np.ndarray, center_xy: np.ndarray,
                    radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Pick the emergence point/direction as the ring angle with the strongest
    guide response just outside the grain boundary."""
    best_angle, best_score = 0.0, -np.inf
    for angle in np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        radii = np.arange(radius_px + 2.0, radius_px + 9.0, 1.0)
        points = center_xy[None, :] + direction[None, :] * radii[:, None]
        values = map_coordinates(guide, [points[:, 1], points[:, 0]], order=1)
        score = float(values.max())
        if score > best_score:
            best_score, best_angle = score, angle
    direction = np.array([np.cos(best_angle), np.sin(best_angle)])
    return center_xy + direction * (radius_px + 2.0), direction


def build_kymograph(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    grays = np.load(args.cache_dir / "gray_samples.npy", mmap_mode="r")
    tracks, visibility, cache_indices = load_body_tracks(args.cache_dir, args.run_dir)
    frame_count = tracks.shape[0]
    reference = frame_count - 1

    centerlines = pd.read_csv(args.run_dir / "centerline_points.csv")
    measurements = pd.read_csv(args.run_dir / "measurements.csv")
    grain_centers = measurements[["pollen_x_px", "pollen_y_px"]].to_numpy()
    pose = smoothed_pose_transforms(tracks, visibility, grain_centers, frame_count,
                                    grays=grays, cache_indices=cache_indices)
    frames_with_centerline = set(centerlines.analysis_frame.unique())
    # a handful of review-only chains is not a usable per-frame trace: fall
    # back to candidate/grain seeding in that case
    grain_seeded = len(frames_with_centerline) < 5
    first_material_frame = None if grain_seeded else min(frames_with_centerline)

    def centerline_xy(analysis_frame: int) -> np.ndarray:
        rows = centerlines[centerlines.analysis_frame == analysis_frame]
        return rows.sort_values("point_index")[["x_px", "y_px"]].to_numpy()

    # extension corridor traced on a temporally denoised late window (reference coords)
    window = []
    for frame in range(max(reference - 16, 0), frame_count):
        matrix = pose[frame]
        window.append(cv.warpAffine(grays[cache_indices[frame]], matrix,
                                    (grays.shape[2], grays.shape[1]), borderValue=255))
    late_ridge = dog(np.median(np.stack(window), axis=0),
                     args.dog_sigma_tube, args.dog_sigma_background)
    if args.evidence_mode == "abs":
        late_ridge = np.abs(late_ridge)
    if grain_seeded:
        # no accepted centerlines at all: anchor the corridor on the grain and
        # follow newly-grown material (early-vs-late novelty) outward, which
        # suppresses structures that already existed before germination
        early = []
        for frame in range(min(30, frame_count)):
            matrix = pose[frame]
            early.append(cv.warpAffine(grays[cache_indices[frame]], matrix,
                                       (grays.shape[2], grays.shape[1]), borderValue=255))
        late_median = np.median(np.stack(window), axis=0)
        novelty = np.median(np.stack(early), axis=0).astype(np.float32) - late_median
        novelty = cv.GaussianBlur(np.abs(novelty), (0, 0), 1.5)
        candidate_path = args.run_dir / "candidate_centerline_points.csv"
        candidates = (pd.read_csv(candidate_path)
                      if candidate_path.exists() else pd.DataFrame())
        if len(candidates):
            # the tracer proposed (and refused) germination chains: use the
            # longest one as the corridor base instead of re-deriving it
            counts = candidates.groupby("analysis_frame").size()
            base_frame = int(counts[counts == counts.max()].index.max())
            rows = candidates[candidates.analysis_frame == base_frame]
            base = rows.sort_values("point_index")[["x_px", "y_px"]].to_numpy()
            to_ref = pose[base_frame]
            base_ref = cv.transform(base[None].astype(np.float32), to_ref)[0]
            tangent = base_ref[-1] - base_ref[-3]
            extension = trace_extension_corridor(novelty, base_ref[-1], tangent,
                                                 args.extension_steps, stop_below=1.5)
            extension_ref = np.vstack([base_ref, extension[1:]])
        else:
            centers = measurements[["pollen_x_px", "pollen_y_px"]].to_numpy()
            center = centers[min(reference, len(centers) - 1)]
            seed, direction = seed_from_grain(novelty, center, args.grain_radius_px)
            extension_ref = trace_extension_corridor(novelty, seed, direction,
                                                     args.extension_steps,
                                                     stop_below=1.5)
    else:
        final_frame = max(frames_with_centerline)
        final_curve = centerline_xy(final_frame)
        tangent = final_curve[-1] - final_curve[-5]
        extension_ref = trace_extension_corridor(late_ridge, final_curve[-1], tangent,
                                                 args.extension_steps,
                                                 stop_below=0.25, stop_patience=8)

    offsets = np.linspace(-args.normal_halfwidth_px, args.normal_halfwidth_px, 7)
    rows = []
    corridors = []
    for frame in range(frame_count):
        to_frame = cv.invertAffineTransform(
            pose[frame])
        extension = cv.transform(extension_ref[None].astype(np.float32), to_frame)[0]
        if grain_seeded:
            corridor = resample_polyline(extension, args.arc_step_px)
        else:
            analysis_frame = frame if frame in frames_with_centerline else first_material_frame
            body_curve = centerline_xy(analysis_frame)
            if analysis_frame != frame:
                to_ref = pose[analysis_frame]
                body_curve = cv.transform(
                    cv.transform(body_curve[None].astype(np.float32), to_ref), to_frame)[0]
            corridor = np.vstack([resample_polyline(body_curve, args.arc_step_px),
                                  resample_polyline(extension, args.arc_step_px)[1:]])
        tangents = np.gradient(corridor, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        ridge = dog(grays[cache_indices[frame]], args.dog_sigma_tube, args.dog_sigma_background)
        if args.evidence_mode == "abs":
            ridge = np.abs(ridge)
        samples = [map_coordinates(ridge,
                                   [(corridor + normals * off)[:, 1],
                                    (corridor + normals * off)[:, 0]], order=1)
                   for off in offsets]
        rows.append(np.max(samples, axis=0))
        corridors.append(corridor)

    columns = max(len(row) for row in rows)
    kymograph = np.full((len(rows), columns), -5.0, np.float32)
    corridor_xy = np.zeros((len(rows), columns, 2), np.float32)
    for i, (row, corridor) in enumerate(zip(rows, corridors)):
        kymograph[i, :len(row)] = row
        corridor_xy[i, :len(corridor)] = corridor
        corridor_xy[i, len(corridor):] = corridor[-1]
    return kymograph, corridor_xy, measurements


def monotone_front(kymograph: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    smoothed = median_filter(kymograph, size=(7, 1))
    evidence = np.tanh((smoothed - args.evidence_midpoint) / 2.0)
    gains = np.cumsum(evidence - args.coverage_cost, axis=1)
    frame_count, columns = gains.shape
    max_step = max(1, int(round(args.max_growth_px_per_frame / args.arc_step_px)))

    value = np.full((frame_count, columns), -np.inf)
    parent = np.zeros((frame_count, columns), np.int32)
    value[0] = gains[0]
    for frame in range(1, frame_count):
        for column in range(columns):
            low = max(0, column - max_step)
            window = value[frame - 1, low:column + 1]
            best = int(np.argmax(window))
            value[frame, column] = window[best] + gains[frame, column]
            parent[frame, column] = low + best
    front = np.zeros(frame_count, np.int32)
    front[-1] = int(np.argmax(value[-1]))
    for frame in range(frame_count - 1, 0, -1):
        front[frame - 1] = parent[frame, front[frame]]
    return front


def retrace_per_window(args: argparse.Namespace):
    """Per-window root-anchored retracing for tubes that sway too much for a
    single rigid corridor. Returns (kymograph, corridor_xy, front, measurements)."""
    grays = np.load(args.cache_dir / "gray_samples.npy", mmap_mode="r")
    tracks, visibility, cache_indices = load_body_tracks(args.cache_dir, args.run_dir)
    frame_count = tracks.shape[0]
    reference = frame_count - 1
    measurements = pd.read_csv(args.run_dir / "measurements.csv")
    grain_centers = measurements[["pollen_x_px", "pollen_y_px"]].to_numpy()
    pose = smoothed_pose_transforms(tracks, visibility, grain_centers, frame_count,
                                    grays=grays, cache_indices=cache_indices)

    stabilized = np.empty((frame_count,) + grays.shape[1:], np.uint8)
    for frame in range(frame_count):
        matrix = pose[frame]
        stabilized[frame] = cv.warpAffine(grays[cache_indices[frame]], matrix,
                                          (grays.shape[2], grays.shape[1]),
                                          borderValue=255)
    early_median = np.median(stabilized[:30], axis=0).astype(np.float32)
    early_band = cv.GaussianBlur(
        np.abs(dog(early_median, args.dog_sigma_tube, args.dog_sigma_background)),
        (0, 0), 2.5)
    # suppress structures that already existed before germination
    fresh_mask = np.clip(1.0 - early_band / 8.0, 0.0, 1.0)

    candidates = pd.read_csv(args.run_dir / "candidate_centerline_points.csv")
    counts = candidates.groupby("analysis_frame").size()
    base_frame = int(counts[counts == counts.max()].index.max())
    rows = candidates[candidates.analysis_frame == base_frame]
    base = rows.sort_values("point_index")[["x_px", "y_px"]].to_numpy()
    to_ref = pose[base_frame]
    base_ref = cv.transform(base[None].astype(np.float32), to_ref)[0]
    root = base_ref[0]
    direction = base_ref[min(3, len(base_ref) - 1)] - base_ref[0]

    window_centers = list(range(0, frame_count, args.window_stride))
    guides = {}
    for center in window_centers:
        lo = max(0, center - args.window_halfwidth)
        hi = min(frame_count, center + args.window_halfwidth + 1)
        window_median = np.median(stabilized[lo:hi], axis=0).astype(np.float32)
        band = cv.GaussianBlur(
            np.abs(dog(window_median, args.dog_sigma_tube, args.dog_sigma_background)),
            (0, 0), 2.0)
        guides[center] = band * fresh_mask

    paths, lengths = {}, {}
    extension_budget = max(2, int(args.window_stride * args.max_growth_px_per_frame / 2.0))
    current = None
    for index, center in enumerate(window_centers):
        guide = guides[center]
        if current is None or len(current) < 3:
            # not yet germinated (or nothing traceable so far): try a fresh trace
            path = trace_extension_corridor(guide, root, direction,
                                            steps=args.extension_steps,
                                            stop_below=0.7)
            current = resample_polyline(path, args.arc_step_px) if len(path) > 2 \
                else path.reshape(-1, 2)
        else:
            support = float(np.median(map_coordinates(
                guide, [current[:, 1], current[:, 0]], order=1)))
            if support >= 0.4:  # tube visible this window: follow the sway
                refit = lateral_refit(current, guide)
                refit[0] = root
                refit = resample_polyline(refit, args.arc_step_px)
                if len(refit) >= 3:
                    tangent = refit[-1] - refit[-3]
                    extension = trace_extension_corridor(
                        guide, refit[-1], tangent,
                        steps=extension_budget, stop_below=0.7)
                    if len(extension) > 2:
                        extension = resample_polyline(extension, args.arc_step_px)
                        # annexation guard: growth must be NEW material - if it
                        # was already band-supported a few windows ago, it
                        # belongs to another tube
                        old_center = window_centers[max(0, index - 3)]
                        old_support = map_coordinates(
                            guides[old_center],
                            [extension[:, 1], extension[:, 0]], order=1)
                        foreign = np.where(old_support >= 0.8)[0]
                        keep = int(foreign[0]) if len(foreign) else len(extension)
                        if keep > 1:
                            refit = np.vstack([refit, extension[1:keep]])
                if len(refit) >= len(current):  # material never shrinks
                    current = refit
            # low support (blur burst): hold the corridor unchanged
        paths[center] = current.copy()
        lengths[center] = args.arc_step_px * max(len(current) - 1, 0)

    # backward pass: the mature corridor is reliable; walk it back in time,
    # refitting laterally and trimming the tip where support ends. This
    # recovers early growth that forward fresh-tracing was too weak to find.
    forward_lengths = np.array([lengths[c] for c in window_centers])
    anchor = int(np.argmax(forward_lengths >= forward_lengths.max() - 1e-6))
    previous = paths[window_centers[anchor]]
    for index in range(anchor, -1, -1):
        center = window_centers[index]
        guide = guides[center]
        if len(previous) >= 5:
            refit = lateral_refit(previous, guide)
            refit[0] = root
            refit = resample_polyline(refit, args.arc_step_px)
            values = map_coordinates(guide, [refit[:, 1], refit[:, 0]], order=1)
            smooth = np.convolve(values, np.ones(5) / 5.0, mode="same")
            supported = np.where(smooth >= 0.35)[0]
            tip_node = int(supported[-1]) + 1 if len(supported) else 2
            trimmed = refit[: max(tip_node + 1, 3)]
            if len(trimmed) <= len(previous) + 1:
                previous = trimmed
        paths[center] = previous.copy()
        lengths[center] = args.arc_step_px * max(len(previous) - 1, 0)

    raw = np.interp(np.arange(frame_count), window_centers,
                    [lengths[c] for c in window_centers])
    smoothed = median_filter(raw, size=9)
    monotone = np.zeros(frame_count)
    for frame in range(1, frame_count):
        ceiling = monotone[frame - 1] + args.max_growth_px_per_frame
        monotone[frame] = min(ceiling, max(monotone[frame - 1], smoothed[frame]))

    columns = max(len(paths[c]) for c in window_centers)
    corridor_xy = np.zeros((frame_count, columns, 2), np.float32)
    kymograph = np.full((frame_count, columns), -5.0, np.float32)
    for frame in range(frame_count):
        center = window_centers[min(range(len(window_centers)),
                                    key=lambda i: abs(window_centers[i] - frame))]
        path = paths[center]
        padded = np.vstack([path, np.repeat(path[-1:], columns - len(path), axis=0)]) \
            if len(path) < columns else path[:columns]
        to_frame = cv.invertAffineTransform(
            pose[frame])
        corridor = cv.transform(padded[None], to_frame)[0]
        corridor_xy[frame] = corridor
        tangents = np.gradient(corridor, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        ridge = dog(grays[cache_indices[frame]], args.dog_sigma_tube,
                    args.dog_sigma_background)
        if args.evidence_mode == "abs":
            ridge = np.abs(ridge)
        offsets = np.linspace(-args.normal_halfwidth_px, args.normal_halfwidth_px, 7)
        samples = [map_coordinates(ridge,
                                   [(corridor + normals * off)[:, 1],
                                    (corridor + normals * off)[:, 0]], order=1)
                   for off in offsets]
        kymograph[frame, :len(path)] = np.max(samples, axis=0)[:len(path)]

    front = np.clip(np.round(monotone / args.arc_step_px).astype(int), 0, columns - 1)
    return kymograph, corridor_xy, front, measurements


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: best non-decreasing fit to values."""
    level = list(values.astype(float))
    weight = list(weights.astype(float))
    count = [1] * len(level)
    blocks = []
    for v, w, c in zip(level, weight, count):
        blocks.append([v, max(w, 1e-9), c])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            merged = (v1 * w1 + v2 * w2) / (w1 + w2)
            blocks.append([merged, w1 + w2, c1 + c2])
    out = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return np.asarray(out)


def subpixel_tip_arclength(kymograph: np.ndarray, front: np.ndarray,
                           arc_step: float, search_cols: int = 10
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame subpixel tip arclength from the kymograph rows.

    CHUKNORRIS-style: on each temporally-denoised evidence profile along the
    tube, the tip is the descending edge from tube plateau to background.
    Locate the steepest descent near the DP front and refine to subpixel with
    a quadratic fit on the derivative. Returns (arclength_px, edge_contrast)."""
    smoothed = median_filter(kymograph, size=(7, 1))
    frame_count, columns = smoothed.shape
    tips = np.zeros(frame_count)
    contrast = np.zeros(frame_count)
    for frame in range(frame_count):
        row = np.convolve(smoothed[frame], np.ones(3) / 3.0, mode="same")
        center = int(front[frame])
        lo = max(1, center - search_cols)
        hi = min(columns - 2, center + search_cols)
        if hi <= lo + 2 or center < 2:
            tips[frame] = front[frame] * arc_step
            continue
        derivative = np.gradient(row)
        window = derivative[lo:hi]
        k = lo + int(np.argmin(window))
        # quadratic subpixel refinement on the derivative minimum
        left, mid, right = derivative[k - 1], derivative[k], derivative[k + 1]
        denominator = left - 2 * mid + right
        delta = 0.5 * (left - right) / denominator if abs(denominator) > 1e-9 else 0.0
        tips[frame] = (k + float(np.clip(delta, -0.5, 0.5))) * arc_step
        before = row[max(0, k - 6):k].mean() if k > 0 else 0.0
        after = row[k + 1:k + 7].mean() if k + 7 <= columns else 0.0
        contrast[frame] = max(before - after, 0.0)
    return tips, contrast


def point_at_arclength(polyline: np.ndarray, arclength: float,
                       step: float) -> np.ndarray:
    """Interpolated point at a fractional arclength along a resampled polyline."""
    position = arclength / step
    index = int(np.clip(np.floor(position), 0, len(polyline) - 2))
    fraction = float(np.clip(position - index, 0.0, 1.0))
    return polyline[index] * (1.0 - fraction) + polyline[index + 1] * fraction


def project_to_polyline(polyline: np.ndarray, point: np.ndarray,
                        step: float) -> tuple[float, float]:
    """(arclength, signed normal offset) of a point w.r.t. a resampled polyline."""
    deltas = polyline[1:] - polyline[:-1]
    lengths = np.linalg.norm(deltas, axis=1) + 1e-12
    to_point = point[None, :] - polyline[:-1]
    along = np.clip((to_point * deltas).sum(axis=1) / lengths ** 2, 0.0, 1.0)
    feet = polyline[:-1] + deltas * along[:, None]
    distances = np.linalg.norm(point[None, :] - feet, axis=1)
    best = int(np.argmin(distances))
    tangent = deltas[best] / lengths[best]
    offset = point - feet[best]
    signed = float(offset[0] * -tangent[1] + offset[1] * tangent[0])
    return (best + float(along[best])) * step, signed



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.retrace_per_window:
        kymograph, corridor_xy, front, measurements = retrace_per_window(args)
    else:
        kymograph, corridor_xy, measurements = build_kymograph(args)
        front = monotone_front(kymograph, args)
    length_px = front * args.arc_step_px

    smoothed = median_filter(kymograph, size=(7, 1))
    evidence = np.tanh((smoothed - args.evidence_midpoint) / 2.0)
    contamination = np.array([float(evidence[t, front[t] + 2:].clip(0).sum())
                              for t in range(len(front))])

    # tip in path coordinates: subpixel edge regression on each kymograph row,
    # monotone (tip-only growth) + light Savitzky-Golay so pulsatile growth
    # survives while endpoint jitter does not
    from scipy.signal import savgol_filter

    frame_count = len(front)
    tip_arc_raw, tip_contrast = subpixel_tip_arclength(kymograph, front,
                                                       args.arc_step_px)
    tip_arc = isotonic_increasing(tip_arc_raw, np.clip(tip_contrast, 0.05, None))
    if frame_count >= 9:
        tip_arc = np.maximum.accumulate(savgol_filter(tip_arc, 9, 2))
    tip_confidence = np.tanh(tip_contrast / 2.0)
    weights = np.clip(tip_confidence, 0.1, None)
    grain_centers = measurements[["pollen_x_px", "pollen_y_px"]].to_numpy()
    anchor_centers = np.where(np.isfinite(grain_centers), grain_centers, 0.0)

    # assemble the final tip in the path coordinates of ONE fixed reference
    # midline (final corridor, grain-relative): s(t) is monotone growth,
    # n(t) is the (smooth, real) lateral sway. Smooth those 1D series and
    # reconstruct - the trajectory is then smooth by construction.
    tracks, visibility, cache_indices = load_body_tracks(args.cache_dir, args.run_dir)
    grays = np.load(args.cache_dir / "gray_samples.npy", mmap_mode="r")
    pose = smoothed_pose_transforms(tracks, visibility, grain_centers, frame_count,
                                    grays=grays, cache_indices=cache_indices)
    rotations = [m[:2, :2] for m in pose]
    reference = frame_count - 1
    midline = (corridor_xy[reference] - anchor_centers[reference]).astype(np.float64)
    # drop padded (repeated) tail points: they corrupt tangents/projection
    keep = np.ones(len(midline), bool)
    keep[1:] = np.linalg.norm(np.diff(midline, axis=0), axis=1) > 1e-6
    midline = midline[keep]
    if len(midline) < 3:
        midline = (corridor_xy[reference] - anchor_centers[reference])[:3].astype(np.float64)

    raw_s = np.zeros(frame_count)
    raw_n = np.zeros(frame_count)
    for t in range(frame_count):
        raw_tip = point_at_arclength(corridor_xy[t], tip_arc[t], args.arc_step_px)
        relative = rotations[t] @ (raw_tip - anchor_centers[t])
        raw_s[t], raw_n[t] = project_to_polyline(midline, relative, args.arc_step_px)

    s_series = isotonic_increasing(raw_s, weights)
    if frame_count >= 9:
        s_series = np.maximum.accumulate(savgol_filter(s_series, 9, 2))
    n_series = median_filter(raw_n, size=7)
    if frame_count >= 11:
        n_series = savgol_filter(n_series, 11, 2)

    tangents = np.gradient(midline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    tips_src = np.empty((frame_count, 2))
    for t in range(frame_count):
        base = point_at_arclength(midline, s_series[t], args.arc_step_px)
        index = int(np.clip(round(s_series[t] / args.arc_step_px), 0, len(midline) - 1))
        normal = np.array([-tangents[index, 1], tangents[index, 0]])
        relative = base + normal * n_series[t]
        tips_src[t] = rotations[t].T @ relative + anchor_centers[t]

    table = pd.DataFrame({
        "analysis_frame": np.arange(len(front)),
        "front_length_px": length_px,
        "pipeline_length_px": measurements.tube_length_px.fillna(0).to_numpy()[:len(front)],
        "evidence_above_front": contamination,
        "tip_x_px": tips_src[:, 0],
        "tip_y_px": tips_src[:, 1],
        "tip_confidence": tip_confidence,
    })
    table.to_csv(args.output_dir / "growth_front.csv", index=False)
    np.save(args.output_dir / "kymograph.npy", kymograph)
    np.savez_compressed(args.output_dir / "corridor_points.npz",
                        corridor_xy=corridor_xy, front_columns=front,
                        arc_step_px=args.arc_step_px)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 6))
    image = axis.imshow(np.clip(smoothed.T, -2, 12), aspect="auto", origin="lower",
                        cmap="inferno",
                        extent=[0, len(front), 0, kymograph.shape[1] * args.arc_step_px])
    axis.plot(table.analysis_frame, table.front_length_px, "c-", lw=2,
              label="growth front (monotone DP)")
    axis.plot(table.analysis_frame, table.pipeline_length_px, "w--", lw=1.5,
              label="pipeline length")
    axis.set_xlabel("analysis frame")
    axis.set_ylabel("arclength from root (px)")
    axis.set_title("tube-intrinsic kymograph with monotone growth front")
    axis.legend(loc="upper left", fontsize=9)
    figure.colorbar(image, ax=axis, label="max DoG within normal band")
    figure.tight_layout()
    figure.savefig(args.output_dir / "growth_front.png", dpi=110)

    # identity plausibility: while a tube is genuinely growing there should be
    # almost no evidence beyond its tip. Contamination concurrent with the
    # growth phase means the corridor annexed foreign material. Contamination
    # after growth ended is just crossing traffic and only worth a note.
    reasons, notes = [], []
    final_length = float(length_px[-1])
    growing = length_px < 0.9 * final_length if final_length > 0 \
        else np.ones(len(length_px), bool)
    during_growth = float(contamination[growing].mean()) if growing.any() else 0.0
    if during_growth > 1.0:
        reasons.append("foreign material present while growth was measured "
                       f"(mean contamination {during_growth:.1f})")
    if len(length_px) > 5:
        burst = float(np.max(length_px[5:] - length_px[:-5]) / 5.0)
        if burst >= 0.9 * args.max_growth_px_per_frame:
            notes.append(f"growth ramp reached the rate cap ({burst:.1f} px/frame): "
                         "onset may be compressed / measured late")
    if float(contamination[~growing].mean() if (~growing).any() else 0.0) > 1.0:
        notes.append("crossing traffic above the tip after growth ended")
    summary = {
        "final_front_length_px": float(length_px[-1]),
        "final_pipeline_length_px": float(table.pipeline_length_px.iloc[-1]),
        "mean_contamination_first_half": float(contamination[: len(front) // 2].mean()),
        "mean_contamination_second_half": float(contamination[len(front) // 2:].mean()),
        "identity_flag": bool(reasons),
        "identity_reasons": reasons,
        "notes": notes,
    }
    active = front[:-1] > 5
    if active.any():
        relative = tips_src - np.where(np.isfinite(grain_centers),
                                       grain_centers, 0.0)
        steps = np.linalg.norm(np.diff(relative, axis=0), axis=1)[active]
        summary["tip_step_median_px"] = float(np.median(steps))
        summary["tip_step_p90_px"] = float(np.percentile(steps, 90))
        summary["tip_step_max_px"] = float(steps.max())
        summary["tip_tracked_fraction"] = float((tip_confidence > 0.35).mean())
    (args.output_dir / "growth_front_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
