"""Track a pollen tube as one ordered deformable material curve through time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchored_tracing import refine_chain_from_prior
from .curve_prototype import curve_length, resample_curve
from .material_ribbon import (
    MaterialRibbonConfig,
    observe_material_ribbon,
)
from .pollen_anchored_chain import (
    PollenAnchoredChainConfig,
    PollenAnchoredMeasurement,
    chain_integrity,
)
from .topology_aware_tracing import (
    TopologyAwareTraceConfig,
    future_edge_hypotheses,
)


@dataclass(frozen=True)
class MaterialCurveConfig:
    """Configure material-point seeding, coherent motion, and distal growth."""

    node_spacing_px: float = 2.0
    query_spacing_px: float = 4.0
    keyframe_interval: int = 12
    keyframe_minimum_growth_px: float = 3.0
    maximum_query_points: int = 384
    keyframe_median_alignment_px: float = 6.0
    keyframe_p90_alignment_px: float = 10.0
    motion_bandwidth_px: float = 12.0
    motion_recency_frames: float = 80.0
    maximum_group_disagreement_px: float = 5.0
    maximum_point_residual_px: float = 8.0
    minimum_point_image_support: float = 0.04
    point_image_support_radius_px: int = 2
    maximum_frame_displacement_px: float = 4.0
    tracker_blend: float = 0.75
    velocity_blend: float = 0.10
    velocity_decay: float = 0.45
    image_search_radius_px: int = 3
    image_blend: float = 0.55
    ribbon_blend: float = 0.35
    displacement_bending_weight: float = 1.5
    root_lock_points: int = 3
    minimum_track_supported_fraction: float = 0.20
    maximum_extension_px: float = 12.0
    backward_identity_lookback_frames: int = 12
    backward_identity_samples: int = 3
    backward_identity_minimum_matches: int = 2
    backward_identity_median_px: float = 5.0
    backward_identity_p90_px: float = 8.0
    backward_identity_maximum_coherent_offset_px: float = 10.0
    backward_identity_coherent_median_px: float = 3.0
    backward_identity_coherent_p90_px: float = 8.0
    reacquisition_median_alignment_px: float = 10.0
    reacquisition_p90_alignment_px: float = 15.0


@dataclass(frozen=True)
class MaterialQuerySet:
    """Store point queries and their permanent arc-length identities."""

    points_txy: np.ndarray
    arc_positions_px: np.ndarray
    group_indices: np.ndarray
    keyframes: np.ndarray


@dataclass(frozen=True)
class MaterialCurveDiagnostic:
    """Expose material-node support so review media can reveal inference."""

    confidence: np.ndarray
    observed: np.ndarray
    accepted_group_count: int
    rejected_group_count: int
    visible_query_count: int
    left_boundary_yx: np.ndarray
    right_boundary_yx: np.ndarray
    widths_px: np.ndarray
    ribbon_confidence: np.ndarray
    prior_length_px: float
    proposed_growth_px: float
    accepted_growth_px: float
    length_change_px: float
    root_tip_distance_px: float
    tortuosity: float
    length_consistent: bool


@dataclass(frozen=True)
class MaterialGrowthProposal:
    """Describe distal geometry without confusing bending with new length."""

    extension_yx: np.ndarray
    target_length_px: float


@dataclass(frozen=True)
class MaterialCurveTrace:
    """Bundle measurements with the point identities that produced them."""

    measurements: tuple[PollenAnchoredMeasurement, ...]
    diagnostics: tuple[MaterialCurveDiagnostic, ...]
    queries: MaterialQuerySet


def _empty_points():
    """Return an empty row-column curve with a stable shape."""
    return np.empty((0, 2), dtype=np.float64)


def _curve_arc_positions(curve):
    """Return cumulative arc length for an ordered row-column curve."""
    points = np.asarray(curve, dtype=np.float64)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def interpolate_curve(curve, arc_positions_px):
    """Interpolate ordered curve coordinates at absolute material positions."""
    points = np.asarray(curve, dtype=np.float64)
    requested = np.asarray(arc_positions_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("curve must have shape (points, 2) with two points")
    arc = _curve_arc_positions(points)
    clipped = np.clip(requested, 0.0, arc[-1])
    return np.column_stack(
        (
            np.interp(clipped, arc, points[:, 0]),
            np.interp(clipped, arc, points[:, 1]),
        )
    )


def build_material_queries(coarse_measurements, config=None):
    """Retain repeated candidate keyframes for later material-model validation."""
    config = config or MaterialCurveConfig()
    selected = []
    previous_frame = None
    previous_length = 0.0
    for frame, measurement in enumerate(coarse_measurements):
        curve = np.asarray(measurement.centerline_yx, dtype=np.float64)
        if measurement.neighbor_contact or len(curve) < 2:
            continue
        length = float(curve_length(curve))
        due = (
            previous_frame is None
            or frame - previous_frame >= int(config.keyframe_interval)
            or length - previous_length >= config.keyframe_minimum_growth_px
        )
        if not due:
            continue
        selected.append((frame, curve.copy(), length))
        previous_frame = frame
        previous_length = length
    if not selected:
        return MaterialQuerySet(
            points_txy=np.empty((0, 3), dtype=np.float32),
            arc_positions_px=np.empty(0, dtype=np.float64),
            group_indices=np.empty(0, dtype=int),
            keyframes=np.empty(0, dtype=int),
        )

    groups = []
    for group, (frame, curve, length) in enumerate(selected):
        arc = np.arange(0.0, length, float(config.query_spacing_px))
        arc = np.unique(np.concatenate((arc, [length])))
        points_yx = interpolate_curve(curve, arc)
        groups.append(
            (
                np.column_stack(
                    (
                        np.full(len(arc), frame, dtype=np.float32),
                        points_yx[:, ::-1].astype(np.float32),
                    )
                ),
                arc,
                np.full(len(arc), group, dtype=int),
            )
        )
    total = sum(len(group[0]) for group in groups)
    if total > int(config.maximum_query_points):
        stride = int(np.ceil(total / config.maximum_query_points))
        reduced = []
        for points, arc, group in groups:
            keep = np.unique(
                np.concatenate((np.arange(0, len(points), stride), [len(points) - 1]))
            )
            reduced.append((points[keep], arc[keep], group[keep]))
        groups = reduced
    return MaterialQuerySet(
        points_txy=np.concatenate([group[0] for group in groups]),
        arc_positions_px=np.concatenate([group[1] for group in groups]),
        group_indices=np.concatenate([group[2] for group in groups]),
        keyframes=np.asarray([item[0] for item in selected], dtype=int),
    )


def _weighted_median(values, weights):
    """Return a deterministic weighted median for one-dimensional values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    threshold = 0.5 * float(cumulative[-1])
    return float(values[order[np.searchsorted(cumulative, threshold)]])


def _group_motion_fields(
    prior,
    frame,
    queries,
    tracks_xy,
    visibility,
    accepted_groups,
    probability,
    config,
):
    """Convert each point-track keyframe into one coherent displacement field."""
    node_arc = _curve_arc_positions(prior)
    query_frames = np.rint(queries.points_txy[:, 0]).astype(int)
    fields = []
    discarded = 0
    for group in sorted(accepted_groups):
        indices = np.flatnonzero(queries.group_indices == group)
        valid = (
            visibility[frame, indices]
            & (query_frames[indices] <= frame)
            & np.isfinite(tracks_xy[frame, indices]).all(axis=1)
        )
        indices = indices[valid]
        if len(indices) < 2:
            discarded += 1
            continue
        arc = queries.arc_positions_px[indices]
        within = arc <= node_arc[-1] + config.node_spacing_px
        indices = indices[within]
        arc = arc[within]
        if len(indices) < 2:
            discarded += 1
            continue
        order = np.argsort(arc)
        arc = arc[order]
        observed_yx = tracks_xy[frame, indices[order], ::-1]
        prior_yx = interpolate_curve(prior, arc)
        displacement = observed_yx - prior_yx
        plausible = (
            np.linalg.norm(displacement, axis=1)
            <= config.maximum_point_residual_px
        )
        if probability is not None:
            probability = np.asarray(probability, dtype=np.float32)
            radius = max(0, int(config.point_image_support_radius_px))
            supported_by_image = []
            for point in observed_yx:
                row, column = np.rint(point).astype(int)
                top = max(0, row - radius)
                bottom = min(probability.shape[0], row + radius + 1)
                left = max(0, column - radius)
                right = min(probability.shape[1], column + radius + 1)
                supported_by_image.append(
                    top < bottom
                    and left < right
                    and float(np.max(probability[top:bottom, left:right]))
                    >= config.minimum_point_image_support
                )
            plausible &= np.asarray(supported_by_image, dtype=bool)
        arc = arc[plausible]
        displacement = displacement[plausible]
        if len(arc) < 2:
            discarded += 1
            continue
        supported = node_arc <= arc[-1] + config.motion_bandwidth_px
        field = np.column_stack(
            (
                np.interp(node_arc, arc, displacement[:, 0]),
                np.interp(node_arc, arc, displacement[:, 1]),
            )
        )
        age = max(0, frame - int(query_frames[indices[0]]))
        weight = max(
            0.35,
            float(np.exp(-age / max(config.motion_recency_frames, 1.0))),
        )
        fields.append((group, field, supported, weight, len(indices)))
    return fields, discarded


def coherent_material_prediction(
    prior,
    velocity,
    frame,
    queries,
    tracks_xy,
    visibility,
    accepted_groups,
    config=None,
    probability=None,
):
    """Predict ordered nodes from robust consensus across material keyframes."""
    config = config or MaterialCurveConfig()
    prior = np.asarray(prior, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    fields, discarded = _group_motion_fields(
        prior,
        frame,
        queries,
        tracks_xy,
        visibility,
        accepted_groups,
        probability,
        config,
    )
    if not fields:
        predicted = prior + config.velocity_decay * velocity
        predicted[: config.root_lock_points] = prior[: config.root_lock_points]
        return predicted, np.zeros(len(prior)), 0, discarded, 0

    stacked = np.stack([item[1] for item in fields])
    supports = np.stack([item[2] for item in fields])
    consensus = np.median(stacked, axis=0)
    retained = []
    rejected = discarded
    for item in fields:
        _, field, supported, _, _ = item
        comparable = supported & np.any(supports, axis=0)
        disagreement = (
            float(np.median(np.linalg.norm(field[comparable] - consensus[comparable], axis=1)))
            if np.any(comparable)
            else np.inf
        )
        if len(fields) == 1 or disagreement <= config.maximum_group_disagreement_px:
            retained.append(item)
        else:
            rejected += 1
    if not retained:
        retained = [max(fields, key=lambda item: item[3])]
        rejected = discarded + len(fields) - 1

    displacement = np.zeros_like(prior)
    confidence = np.zeros(len(prior), dtype=np.float64)
    for node in range(len(prior)):
        candidates = [item for item in retained if item[2][node]]
        if not candidates:
            displacement[node] = config.velocity_decay * velocity[node]
            continue
        weights = np.asarray([item[3] for item in candidates])
        displacement[node, 0] = _weighted_median(
            [item[1][node, 0] for item in candidates], weights
        )
        displacement[node, 1] = _weighted_median(
            [item[1][node, 1] for item in candidates], weights
        )
        confidence[node] = min(1.0, float(np.sum(weights)) / 1.5)
    norms = np.linalg.norm(displacement, axis=1)
    over = norms > config.maximum_frame_displacement_px
    displacement[over] *= (
        config.maximum_frame_displacement_px / norms[over]
    )[:, None]
    predicted = (
        prior
        + config.tracker_blend * displacement
        + config.velocity_blend * config.velocity_decay * velocity
    )
    locked = min(config.root_lock_points, len(prior))
    predicted[:locked] = prior[:locked]
    visible_queries = sum(item[4] for item in retained)
    return predicted, confidence, len(retained), rejected, visible_queries


def regularize_material_curve(target, prior, confidence, config=None):
    """Fit a smooth displacement field without changing material-node order."""
    config = config or MaterialCurveConfig()
    target = np.asarray(target, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if target.shape != prior.shape or confidence.shape != (len(prior),):
        raise ValueError("target, prior, and confidence shapes do not agree")
    count = len(prior)
    data_weights = 0.5 + confidence
    locked = min(config.root_lock_points, count)
    data_weights[:locked] = 1_000.0
    if count >= 3:
        second = np.zeros((count - 2, count), dtype=np.float64)
        for row in range(count - 2):
            second[row, row : row + 3] = (1.0, -2.0, 1.0)
        smoothness = (
            config.displacement_bending_weight * second.T @ second
        )
    else:
        smoothness = np.zeros((count, count), dtype=np.float64)
    matrix = np.diag(data_weights) + smoothness
    desired = target - prior
    desired[:locked] = 0.0
    displacement = np.column_stack(
        (
            np.linalg.solve(matrix, data_weights * desired[:, 0]),
            np.linalg.solve(matrix, data_weights * desired[:, 1]),
        )
    )
    fitted = prior + displacement
    fitted[:locked] = prior[:locked]
    return fitted


def project_inextensible_curve(target, prior, root_lock_points):
    """Preserve every existing material edge length while allowing bending."""
    target = np.asarray(target, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if target.shape != prior.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target and prior must be equally shaped curves")
    output = target.copy()
    locked = min(max(1, int(root_lock_points)), len(prior))
    output[:locked] = prior[:locked]
    for index in range(locked, len(output)):
        edge_length = float(np.linalg.norm(prior[index] - prior[index - 1]))
        direction = output[index] - output[index - 1]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            direction = prior[index] - prior[index - 1]
            direction_norm = max(float(np.linalg.norm(direction)), 1e-6)
        output[index] = (
            output[index - 1] + edge_length * direction / direction_norm
        )
    return output


def append_material_extension(
    chain,
    extension,
    spacing,
    maximum_points,
    maximum_added_length_px=None,
):
    """Append distal nodes while preserving every existing material-node index."""
    chain = np.asarray(chain, dtype=np.float64)
    extension = np.asarray(extension, dtype=np.float64)
    if len(extension) < 2 or len(chain) >= int(maximum_points):
        return chain.copy()
    if np.linalg.norm(extension[0] - chain[-1]) > max(3.0, 2.0 * spacing):
        extension = np.vstack((chain[-1], extension))
    else:
        extension = extension.copy()
        extension[0] = chain[-1]
    arc = _curve_arc_positions(extension)
    available_length = float(arc[-1])
    if maximum_added_length_px is not None:
        available_length = min(
            available_length,
            max(0.0, float(maximum_added_length_px)),
        )
    if available_length <= 1e-6:
        return chain.copy()
    samples = np.arange(float(spacing), available_length, float(spacing))
    samples = np.unique(np.concatenate((samples, [available_length])))
    added = interpolate_curve(extension, samples)
    available = int(maximum_points) - len(chain)
    return np.vstack((chain, added[:available]))


def _coherent_keyframe_alignment(observed, expected, arc, config):
    """Accept a root-anchored systematic offset but reject shape disagreement."""
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    arc = np.asarray(arc, dtype=np.float64)
    if observed.shape != expected.shape or observed.shape != (len(arc), 2):
        raise ValueError("observed, expected, and arc must describe one curve")
    if len(arc) < 3:
        return False
    delta = observed - expected
    root = int(np.argmin(arc))
    if (
        np.linalg.norm(delta[root])
        > config.backward_identity_p90_px
    ):
        return False
    distal = arc >= (
        config.root_lock_points * config.node_spacing_px
    )
    if np.count_nonzero(distal) < 3:
        return False
    distal_delta = delta[distal]
    offset = np.median(distal_delta, axis=0)
    residual = np.linalg.norm(distal_delta - offset, axis=1)
    return bool(
        np.linalg.norm(offset)
        <= config.backward_identity_maximum_coherent_offset_px
        and np.median(residual)
        <= config.backward_identity_coherent_median_px
        and np.percentile(residual, 90)
        <= config.backward_identity_coherent_p90_px
    )


def _measurement(
    status,
    usable,
    chain,
    candidate,
    support,
    integrity,
):
    """Build the shared measurement contract for a material-curve frame."""
    points = np.asarray(chain, dtype=np.float64)
    candidate_points = (
        _empty_points()
        if candidate is None
        else np.asarray(candidate, dtype=np.float64)
    )
    return PollenAnchoredMeasurement(
        status=str(status),
        usable=bool(usable),
        length_px=float(curve_length(points)),
        centerline_yx=points,
        candidate_centerline_yx=candidate_points,
        structure_support=float(support),
        seed_confirmations=0,
        chain_supported_fraction=float(integrity[1]),
        root_attachment_error_px=float(integrity[2]),
        maximum_centerline_step_px=float(integrity[3]),
        neighbor_contact=bool(integrity[4]),
    )


def _activate_keyframe_groups(
    frame,
    chain,
    queries,
    accepted_groups,
    rejected_groups,
    curve_history,
    tracks_xy,
    visibility,
    config,
):
    """Admit keyframes only when their entire existing prefix still agrees."""
    growth_proposal = None
    for group, keyframe in enumerate(queries.keyframes):
        if int(keyframe) != frame or group in accepted_groups or group in rejected_groups:
            continue
        indices = np.flatnonzero(queries.group_indices == group)
        arc = queries.arc_positions_px[indices]
        query_yx = queries.points_txy[indices, 1:][:, ::-1]
        current_length = float(curve_length(chain))
        overlap = arc <= current_length + config.node_spacing_px
        if np.count_nonzero(overlap) < 3:
            rejected_groups.add(group)
            continue
        distances = np.linalg.norm(
            query_yx[overlap]
            - interpolate_curve(chain, arc[overlap]),
            axis=1,
        )
        strict_alignment = (
            np.median(distances) <= config.keyframe_median_alignment_px
            and np.percentile(distances, 90) <= config.keyframe_p90_alignment_px
        )
        history_start = max(
            0, frame - int(config.backward_identity_lookback_frames)
        )
        historical_frames = [
            past
            for past in range(history_start, frame)
            if curve_history[past] is not None
        ]
        if len(historical_frames) > config.backward_identity_samples:
            positions = np.rint(
                np.linspace(
                    0,
                    len(historical_frames) - 1,
                    config.backward_identity_samples,
                )
            ).astype(int)
            historical_frames = [historical_frames[item] for item in positions]
        historical_matches = 0
        for past in historical_frames:
            past_curve = curve_history[past]
            past_length = float(curve_length(past_curve))
            past_valid = visibility[past, indices] & (arc <= past_length)
            if np.count_nonzero(past_valid) < 3:
                continue
            past_distances = np.linalg.norm(
                tracks_xy[past, indices[past_valid], ::-1]
                - interpolate_curve(past_curve, arc[past_valid]),
                axis=1,
            )
            directly_aligned = (
                np.median(past_distances)
                <= config.backward_identity_median_px
                and np.percentile(past_distances, 90)
                <= config.backward_identity_p90_px
            )
            coherently_aligned = _coherent_keyframe_alignment(
                tracks_xy[past, indices[past_valid], ::-1],
                interpolate_curve(past_curve, arc[past_valid]),
                arc[past_valid],
                config,
            )
            if directly_aligned or coherently_aligned:
                historical_matches += 1
        historically_connected = (
            historical_matches >= config.backward_identity_minimum_matches
        )
        relaxed_alignment = (
            np.median(distances) <= config.reacquisition_median_alignment_px
            and np.percentile(distances, 90)
            <= config.reacquisition_p90_alignment_px
        )
        if strict_alignment or (historically_connected and relaxed_alignment):
            accepted_groups.add(group)
            distal = arc > current_length + 0.5 * config.node_spacing_px
            if np.any(distal):
                proposal = MaterialGrowthProposal(
                    extension_yx=np.vstack((chain[-1], query_yx[distal])),
                    target_length_px=float(np.max(arc)),
                )
                if (
                    growth_proposal is None
                    or proposal.target_length_px
                    > growth_proposal.target_length_px
                ):
                    growth_proposal = proposal
        else:
            rejected_groups.add(group)
    return growth_proposal


def trace_material_curve(
    evidence,
    exclusions,
    pollen_radius,
    coarse_measurements,
    queries,
    tracks_xy,
    visibility,
    chain_config=None,
    config=None,
    ribbon_config=None,
):
    """Trace one root-to-tip material curve using tracks, images, and history."""
    chain_config = chain_config or PollenAnchoredChainConfig()
    config = config or MaterialCurveConfig()
    ribbon_config = ribbon_config or MaterialRibbonConfig(
        root_lock_points=config.root_lock_points
    )
    tracks_xy = np.asarray(tracks_xy, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if tracks_xy.shape != (len(evidence), len(queries.points_txy), 2):
        raise ValueError("point tracks do not match evidence and queries")
    if visibility.shape != tracks_xy.shape[:2]:
        raise ValueError("point visibility must match point tracks")
    initialized = next(
        (
            frame
            for frame, measurement in enumerate(coarse_measurements)
            if measurement.usable and len(measurement.centerline_yx) >= 2
        ),
        None,
    )
    empty_diagnostic = MaterialCurveDiagnostic(
        confidence=np.empty(0, dtype=np.float64),
        observed=np.empty(0, dtype=bool),
        accepted_group_count=0,
        rejected_group_count=0,
        visible_query_count=0,
        left_boundary_yx=_empty_points(),
        right_boundary_yx=_empty_points(),
        widths_px=np.empty(0, dtype=np.float64),
        ribbon_confidence=np.empty(0, dtype=np.float64),
        prior_length_px=np.nan,
        proposed_growth_px=0.0,
        accepted_growth_px=0.0,
        length_change_px=0.0,
        root_tip_distance_px=np.nan,
        tortuosity=np.nan,
        length_consistent=True,
    )
    if initialized is None or not len(queries.points_txy):
        return MaterialCurveTrace(
            measurements=tuple(coarse_measurements),
            diagnostics=tuple(empty_diagnostic for _ in coarse_measurements),
            queries=queries,
        )

    output = list(coarse_measurements[:initialized])
    diagnostics = [empty_diagnostic for _ in range(initialized)]
    curve_history = [None for _ in evidence]
    chain = resample_curve(
        coarse_measurements[initialized].centerline_yx,
        spacing=config.node_spacing_px,
        max_points=chain_config.maximum_centerline_points,
    )
    velocity = np.zeros_like(chain)
    accepted_groups = set()
    rejected_groups = set()
    center_yx = np.asarray(
        [chain_config.crop_size / 2.0, chain_config.crop_size / 2.0]
    )
    baseline_count = min(
        max(1, int(chain_config.baseline_observations)), len(evidence)
    )
    historical = np.max(
        np.stack(
            [item.vessel_probability for item in evidence[:baseline_count]]
        ),
        axis=0,
    )
    topology_config = TopologyAwareTraceConfig()
    frames_since_material_support = 0
    ribbon_widths = None

    for frame in range(initialized, len(evidence)):
        prior_length = float(curve_length(chain))
        keyframe_growth = _activate_keyframe_groups(
            frame,
            chain,
            queries,
            accepted_groups,
            rejected_groups,
            curve_history,
            tracks_xy,
            visibility,
            config,
        )
        keyframe_extended = False
        proposed_growth = 0.0
        accepted_growth = 0.0
        if keyframe_growth is not None:
            proposed_growth = max(
                0.0,
                keyframe_growth.target_length_px - prior_length,
            )
            previous_count = len(chain)
            chain = append_material_extension(
                chain,
                keyframe_growth.extension_yx,
                config.node_spacing_px,
                chain_config.maximum_centerline_points,
                maximum_added_length_px=proposed_growth,
            )
            if len(chain) > previous_count:
                keyframe_extended = True
                accepted_growth = float(curve_length(chain)) - prior_length
                added = len(chain) - previous_count
                tail_velocity = velocity[-1] if len(velocity) else np.zeros(2)
                velocity = np.vstack(
                    (velocity, np.repeat(tail_velocity[None], added, axis=0))
                )
                if ribbon_widths is not None:
                    tail_width = float(np.median(ribbon_widths[-min(5, len(ribbon_widths)) :]))
                    ribbon_widths = np.concatenate(
                        (ribbon_widths, np.full(added, tail_width))
                    )
        predicted, confidence, retained, rejected, visible_count = (
            coherent_material_prediction(
                chain,
                velocity,
                frame,
                queries,
                tracks_xy,
                visibility,
                accepted_groups,
                config,
                probability=evidence[frame].vessel_probability,
            )
        )
        track_fraction = float(np.mean(confidence > 0.0)) if len(confidence) else 0.0
        if (
            track_fraction == 0.0
            and frames_since_material_support >= 2
        ):
            predicted = chain.copy()
            velocity = np.zeros_like(velocity)
        refined, image_support, _ = refine_chain_from_prior(
            evidence[frame].vessel_probability,
            predicted,
            exclusion=exclusions[frame],
            search_radius=config.image_search_radius_px,
            root_lock_points=config.root_lock_points,
            offset_change_penalty=(
                chain_config.trace.chain_refit_offset_change_penalty
            ),
            offset_magnitude_penalty=(
                chain_config.trace.chain_refit_offset_magnitude_penalty
            ),
        )
        if track_fraction >= config.minimum_track_supported_fraction:
            frames_since_material_support = 0
        else:
            frames_since_material_support += 1
        image_authority = min(
            1.0,
            track_fraction / max(config.minimum_track_supported_fraction, 1e-6),
        )
        ribbon = observe_material_ribbon(
            evidence[frame].gray,
            predicted,
            prior_widths_px=ribbon_widths,
            exclusion=exclusions[frame],
            config=ribbon_config,
        )
        target = (
            predicted
            + config.image_blend
            * image_authority
            * (refined - predicted)
            + config.ribbon_blend
            * image_authority
            * ribbon.confidence[:, None]
            * (ribbon.centerline_yx - predicted)
        )
        fitted = regularize_material_curve(target, chain, confidence, config)
        fitted = project_inextensible_curve(
            fitted, chain, config.root_lock_points
        )
        previous_chain = chain
        chain = fitted
        velocity = config.velocity_decay * velocity + (chain - previous_chain)
        boundary_shift = chain - ribbon.centerline_yx
        left_boundary = ribbon.left_boundary_yx + boundary_shift
        right_boundary = ribbon.right_boundary_yx + boundary_shift
        if ribbon_widths is None:
            ribbon_widths = ribbon.widths_px.copy()
        else:
            observed_width = ribbon.confidence > 0.0
            blend = float(ribbon_config.width_update_blend)
            ribbon_widths[observed_width] = (
                (1.0 - blend) * ribbon_widths[observed_width]
                + blend * ribbon.widths_px[observed_width]
            )
        integrity = chain_integrity(
            evidence[frame].vessel_probability,
            chain,
            center_yx,
            pollen_radius,
            chain_config,
            exclusions[frame],
        )
        usable = bool(
            integrity[2] <= chain_config.root_attachment_tolerance_px
            and integrity[3] <= chain_config.maximum_centerline_step_px
            and (
                track_fraction >= config.minimum_track_supported_fraction
                or (
                    frames_since_material_support <= 2
                    and integrity[1]
                    >= chain_config.minimum_chain_supported_fraction
                )
            )
        )
        if track_fraction >= config.minimum_track_supported_fraction:
            status = "material_curve_tracked"
        elif usable:
            status = "material_curve_image_inferred"
        else:
            status = "material_curve_temporally_inferred"
        if keyframe_extended:
            status = "material_curve_keyframe_extended"

        candidate = None
        hypotheses = future_edge_hypotheses(
            evidence[frame],
            historical,
            chain,
            exclusions[frame],
            chain_config,
            topology_config,
        )
        strongest = (
            next(
                (
                    item
                    for item in hypotheses
                    if item.added_length_px <= config.maximum_extension_px
                ),
                None,
            )
            if usable
            else None
        )
        if strongest is not None:
            candidate = append_material_extension(
                chain,
                strongest.extension_yx,
                config.node_spacing_px,
                chain_config.maximum_centerline_points,
            )
            if not keyframe_extended:
                status = "material_curve_growth_candidate"

        integrity = chain_integrity(
            evidence[frame].vessel_probability,
            chain,
            center_yx,
            pollen_radius,
            chain_config,
            exclusions[frame],
        )
        current_length = float(curve_length(chain))
        length_change = current_length - prior_length
        root_tip_distance = float(np.linalg.norm(chain[-1] - chain[0]))
        tortuosity = current_length / max(root_tip_distance, 1e-6)
        length_consistent = bool(
            abs(length_change - accepted_growth) <= 1e-5
            and length_change >= -1e-5
        )
        output.append(
            _measurement(
                status,
                usable,
                chain,
                candidate,
                image_support,
                integrity,
            )
        )
        diagnostics.append(
            MaterialCurveDiagnostic(
                confidence=confidence.copy(),
                observed=confidence > 0.0,
                accepted_group_count=len(accepted_groups),
                rejected_group_count=len(rejected_groups) + rejected,
                visible_query_count=visible_count,
                left_boundary_yx=left_boundary.copy(),
                right_boundary_yx=right_boundary.copy(),
                widths_px=ribbon_widths.copy(),
                ribbon_confidence=ribbon.confidence.copy(),
                prior_length_px=prior_length,
                proposed_growth_px=proposed_growth,
                accepted_growth_px=accepted_growth,
                length_change_px=length_change,
                root_tip_distance_px=root_tip_distance,
                tortuosity=tortuosity,
                length_consistent=length_consistent,
            )
        )
        curve_history[frame] = chain.copy()
    return MaterialCurveTrace(
        measurements=tuple(output),
        diagnostics=tuple(diagnostics),
        queries=queries,
    )
