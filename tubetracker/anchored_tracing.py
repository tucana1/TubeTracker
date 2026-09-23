"""Pollen-anchored geodesic tracing for growing pollen tubes."""

from __future__ import annotations

import math

import cv2 as cv
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import frangi, sato
from skimage.graph import MCP_Geometric

from .curve_prototype import (
    CurveMeasurement,
    curve_length,
    register_translation,
    resample_curve,
    robust_normalize,
    prepare_analysis_gray,
)


def enhance_tube_probability(frame, aligned_previous=None, config=None):
    """Estimate dark tubular structure and newly darkened growth probability."""
    gray = (
        cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if config is None
        else prepare_analysis_gray(frame, config)
    )
    gray_float = gray.astype(np.float32) / 255.0
    sigmas = (0.8, 1.2, 1.8, 2.6, 3.6)
    frangi_response = robust_normalize(
        frangi(gray_float, sigmas=sigmas, black_ridges=True, mode="reflect")
    )
    sato_response = robust_normalize(
        sato(gray_float, sigmas=sigmas, black_ridges=True, mode="reflect")
    )
    blackhat = cv.morphologyEx(
        gray,
        cv.MORPH_BLACKHAT,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)),
    )
    probability = robust_normalize(
        0.52 * frangi_response
        + 0.28 * sato_response
        + 0.20 * robust_normalize(blackhat)
    )
    if aligned_previous is None:
        temporal = np.zeros_like(probability)
    else:
        newly_dark = cv.subtract(aligned_previous, gray)
        temporal = robust_normalize(newly_dark.astype(np.float32))
        temporal = cv.GaussianBlur(temporal, (0, 0), 1.0)
    return gray, probability, temporal


def _crop_bounds(center_yx, radius, shape):
    """Return clamped crop bounds around a row-column center."""
    center_y, center_x = center_yx
    left = max(0, int(math.floor(center_x - radius)))
    top = max(0, int(math.floor(center_y - radius)))
    right = min(shape[1], int(math.ceil(center_x + radius + 1)))
    bottom = min(shape[0], int(math.ceil(center_y + radius + 1)))
    return left, top, right, bottom


def build_grain_exclusion(shape, tracks, frame_index, target_track):
    """Create hard barriers for pollen bodies and report local crowding."""
    exclusion = np.zeros(shape, dtype=np.uint8)
    target = target_track.roi_closest_to(frame_index)
    target_center = np.array([target.gv3.y, target.gv3.x], dtype=np.float64)
    target_radius = max(target.w, target.h) / 2.0
    crowding = 0
    for track in tracks:
        if frame_index < track.first_frame() or frame_index > track.last_frame():
            continue
        roi = track.roi_closest_to(frame_index)
        center = (int(round(roi.gv3.x)), int(round(roi.gv3.y)))
        radius = max(roi.w, roi.h) / 2.0
        padding = 1 if track is target_track else 4
        cv.circle(exclusion, center, max(1, int(round(radius + padding))), 1, -1)
        if track is not target_track:
            distance = math.hypot(
                roi.gv3.x - target_center[1], roi.gv3.y - target_center[0]
            )
            if distance < 1.25 * (radius + target_radius):
                crowding += 1
    return exclusion.astype(bool), target, crowding


def _geodesic_cost(probability, exclusion):
    """Convert tube probability into a high-contrast path traversal cost."""
    clipped = np.clip(probability, 0.0, 1.0)
    cost = 1.0 + 35.0 * np.power(1.0 - clipped, 3.0)
    cost = cv.GaussianBlur(cost.astype(np.float32), (0, 0), 0.7)
    cost[exclusion] = 1e6
    return cost


def _peak_coordinates(score, mask, threshold, count):
    """Return a bounded set of well-separated candidate coordinates."""
    if not np.any(mask):
        return np.empty((0, 2), dtype=int)
    return peak_local_max(
        score,
        labels=mask.astype(np.uint8),
        min_distance=3,
        threshold_abs=threshold,
        num_peaks=count,
        exclude_border=False,
    )


def _path_metrics(path, probability, temporal, center_yx):
    """Measure support, growth evidence, geometry, and radial departure."""
    indices = np.rint(path).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, probability.shape[0] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, probability.shape[1] - 1)
    values = probability[indices[:, 0], indices[:, 1]]
    distal_start = max(1, int(len(indices) * 0.7))
    distal = indices[distal_start:]
    temporal_score = (
        float(np.mean(temporal[distal[:, 0], distal[:, 1]]))
        if len(distal)
        else 0.0
    )
    length = curve_length(path)
    straight = float(np.linalg.norm(path[-1] - path[0]))
    straightness = straight / length if length > 0 else 0.0
    radial = np.linalg.norm(path - np.asarray(center_yx), axis=1)
    radial_progress = (
        float(np.mean(np.diff(radial) >= -0.75)) if len(radial) > 1 else 0.0
    )
    return {
        "length": length,
        "support": float(np.mean(values)),
        "strong_fraction": float(np.mean(values >= 0.10)),
        "temporal": temporal_score,
        "straightness": straightness,
        "radial_progress": radial_progress,
        "radial_gain": float(radial[-1] - radial[0]),
    }


def find_attached_birth_path(
    probability,
    temporal,
    center_yx,
    grain_radius,
    exclusion,
    config,
):
    """Find a geodesic that starts immediately outside one pollen boundary."""
    crop_radius = config.birth_search_radius
    left, top, right, bottom = _crop_bounds(center_yx, crop_radius, probability.shape)
    local_probability = probability[top:bottom, left:right]
    local_temporal = temporal[top:bottom, left:right]
    local_exclusion = exclusion[top:bottom, left:right].copy()
    local_center = np.asarray(center_yx) - np.array([top, left], dtype=np.float64)
    y_grid, x_grid = np.indices(local_probability.shape)
    radial = np.hypot(y_grid - local_center[0], x_grid - local_center[1])
    attachment = (
        (radial >= grain_radius + 1)
        & (radial <= grain_radius + config.attachment_width)
        & ~local_exclusion
    )
    root_threshold = max(
        config.min_attachment_probability,
        float(np.quantile(local_probability[attachment], 0.55))
        if np.any(attachment)
        else 1.0,
    )
    roots = _peak_coordinates(
        local_probability,
        attachment,
        root_threshold,
        config.max_geodesic_endpoints,
    )
    endpoint_mask = (
        (radial >= config.min_curve_length)
        & (radial <= crop_radius - 2)
        & ~local_exclusion
    )
    endpoint_score = 0.72 * local_probability + 0.28 * local_temporal
    endpoint_threshold = max(
        config.min_path_probability,
        float(np.quantile(endpoint_score[endpoint_mask], 0.82))
        if np.any(endpoint_mask)
        else 1.0,
    )
    endpoints = _peak_coordinates(
        endpoint_score,
        endpoint_mask,
        endpoint_threshold,
        config.max_geodesic_endpoints,
    )
    if len(roots) == 0 or len(endpoints) == 0:
        return None

    cost = _geodesic_cost(local_probability, local_exclusion)
    solver = MCP_Geometric(cost, fully_connected=True)
    cumulative, _ = solver.find_costs([tuple(point) for point in roots])
    best = None
    offset = np.array([top, left], dtype=np.float64)
    for endpoint in endpoints:
        endpoint_tuple = tuple(endpoint)
        if not np.isfinite(cumulative[endpoint_tuple]):
            continue
        try:
            local_path = np.asarray(solver.traceback(endpoint_tuple), dtype=np.float64)
        except ValueError:
            continue
        if len(local_path) < 3:
            continue
        path = local_path + offset
        metrics = _path_metrics(path, probability, temporal, center_yx)
        proximal_count = min(len(local_path), max(4, config.attachment_width + 2))
        proximal = local_path[:proximal_count].astype(int)
        attachment_support = float(
            np.mean(local_probability[proximal[:, 0], proximal[:, 1]])
        )
        if (
            metrics["length"] < config.min_curve_length
            or metrics["length"] > config.max_initial_curve_length
            or metrics["support"] < config.min_path_probability
            or metrics["strong_fraction"] < 0.55
            or attachment_support < config.min_attachment_probability
            or metrics["straightness"] < config.min_curve_straightness
            or metrics["radial_progress"] < 0.70
            or metrics["radial_gain"] < 0.55 * config.min_curve_length
        ):
            continue
        normalized_cost = float(cumulative[endpoint_tuple]) / metrics["length"]
        score = (
            2.2 * metrics["support"]
            + 0.9 * metrics["temporal"]
            + 0.7 * metrics["straightness"]
            + 0.5 * metrics["radial_progress"]
            + 0.004 * min(metrics["length"], 120.0)
            - 0.035 * normalized_cost
        )
        candidate = (score, resample_curve(path, max_points=160), metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def find_tip_extension_candidates(
    probability,
    temporal,
    prior,
    exclusion,
    config,
    edge_cue=None,
    maximum_candidates=8,
):
    """Return diverse future nodes together with their connecting geodesic edges."""
    tip = np.asarray(prior[-1], dtype=np.float64)
    tangent = tip - prior[max(0, len(prior) - 6)]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 0:
        return None
    tangent /= tangent_norm
    crop_radius = config.extension_search_radius
    left, top, right, bottom = _crop_bounds(tip, crop_radius, probability.shape)
    local_probability = probability[top:bottom, left:right]
    local_temporal = temporal[top:bottom, left:right]
    if edge_cue is None:
        local_edge_cue = None
        connection_probability = local_probability
    else:
        edge_cue = np.asarray(edge_cue, dtype=np.float32)
        if edge_cue.shape != probability.shape:
            raise ValueError("edge_cue must match probability")
        local_edge_cue = np.clip(edge_cue[top:bottom, left:right], 0.0, 1.0)
        connection_probability = 0.5 * (local_probability + local_edge_cue)
    local_exclusion = exclusion[top:bottom, left:right].copy()
    local_tip = tip - np.array([top, left], dtype=np.float64)
    y_grid, x_grid = np.indices(local_probability.shape)
    delta_y = y_grid - local_tip[0]
    delta_x = x_grid - local_tip[1]
    distance = np.hypot(delta_y, delta_x)
    forward = delta_y * tangent[0] + delta_x * tangent[1]
    lateral = np.abs(delta_y * tangent[1] - delta_x * tangent[0])
    endpoint_mask = (
        (distance >= 3.0)
        & (distance <= config.max_tip_step)
        & (forward >= 1.5)
        & (lateral <= config.max_tip_lateral_step)
        & ~local_exclusion
    )
    if local_edge_cue is None:
        endpoint_score = 0.58 * local_probability + 0.42 * local_temporal
    else:
        endpoint_score = (
            local_probability + local_temporal + local_edge_cue
        ) / 3.0
    endpoints = _peak_coordinates(
        endpoint_score,
        endpoint_mask,
        max(config.min_path_probability, config.min_extension_temporal_score),
        config.max_geodesic_endpoints,
    )
    if len(endpoints) == 0:
        return None

    tip_pixel = tuple(np.rint(local_tip).astype(int))
    if not (
        0 <= tip_pixel[0] < local_probability.shape[0]
        and 0 <= tip_pixel[1] < local_probability.shape[1]
    ):
        return None
    local_exclusion[max(0, tip_pixel[0] - 2) : tip_pixel[0] + 3,
                    max(0, tip_pixel[1] - 2) : tip_pixel[1] + 3] = False
    cost = _geodesic_cost(connection_probability, local_exclusion)
    solver = MCP_Geometric(cost, fully_connected=True)
    cumulative, _ = solver.find_costs([tip_pixel])
    candidates = []
    offset = np.array([top, left], dtype=np.float64)
    for endpoint in endpoints:
        endpoint_tuple = tuple(endpoint)
        if not np.isfinite(cumulative[endpoint_tuple]):
            continue
        try:
            local_path = np.asarray(solver.traceback(endpoint_tuple), dtype=np.float64)
        except ValueError:
            continue
        if len(local_path) < 3:
            continue
        extension = local_path + offset
        extension_length = curve_length(extension)
        tip_delta = extension[-1] - tip
        endpoint_forward = float(np.dot(tip_delta, tangent))
        endpoint_lateral = float(
            np.linalg.norm(tip_delta - endpoint_forward * tangent)
        )
        indices = np.rint(extension).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, probability.shape[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, probability.shape[1] - 1)
        support = float(np.mean(probability[indices[:, 0], indices[:, 1]]))
        distal = indices[max(1, len(indices) // 2) :]
        temporal_score = float(
            np.mean(temporal[distal[:, 0], distal[:, 1]])
        )
        edge_support = (
            float(np.mean(edge_cue[distal[:, 0], distal[:, 1]]))
            if edge_cue is not None
            else 0.0
        )
        relative = extension - tip
        forward_positions = relative @ tangent
        lateral_positions = np.abs(
            relative[:, 0] * tangent[1] - relative[:, 1] * tangent[0]
        )
        forward_fraction = endpoint_forward / max(extension_length, 1e-6)
        forward_progress = (
            float(np.mean(np.diff(forward_positions) >= -0.25))
            if len(forward_positions) > 1
            else 0.0
        )
        proximal_direction = extension[min(3, len(extension) - 1)] - tip
        proximal_norm = float(np.linalg.norm(proximal_direction))
        proximal_alignment = (
            float(np.dot(proximal_direction / proximal_norm, tangent))
            if proximal_norm > 0
            else -1.0
        )
        overlaps_history = False
        if len(prior) > 8 and len(extension) > 2:
            historical = prior[:-6]
            distal_extension = extension[2:]
            distances = np.linalg.norm(
                distal_extension[:, None, :] - historical[None, :, :], axis=2
            )
            overlaps_history = bool(np.min(distances) < 3.0)
        if (
            extension_length < 2.5
            or extension_length > 1.35 * config.max_tip_step
            or endpoint_forward < 1.5
            or endpoint_lateral > config.max_tip_lateral_step
            or forward_fraction < 0.58
            or forward_progress < 0.78
            or proximal_alignment < 0.45
            or np.max(lateral_positions) > config.max_tip_lateral_step
            or overlaps_history
            or support < config.min_path_probability
            or temporal_score < config.min_extension_temporal_score
        ):
            continue
        normalized_cost = float(cumulative[endpoint_tuple]) / extension_length
        score = (
            2.0 * support
            + 1.6 * temporal_score
            + 0.025 * endpoint_forward
            - 0.025 * endpoint_lateral
            - 0.035 * normalized_cost
        )
        if edge_cue is not None:
            score += edge_support
        candidates.append((score, extension, support, temporal_score))

    candidates.sort(key=lambda item: item[0], reverse=True)
    diverse = []
    for candidate in candidates:
        endpoint = candidate[1][-1]
        if any(
            np.linalg.norm(endpoint - retained[1][-1]) < 4.0
            for retained in diverse
        ):
            continue
        diverse.append(candidate)
        if len(diverse) >= max(1, int(maximum_candidates)):
            break
    return tuple(diverse)


def find_tip_extension(probability, temporal, prior, exclusion, config):
    """Find the strongest supported future edge for legacy greedy tracing."""
    candidates = find_tip_extension_candidates(
        probability,
        temporal,
        prior,
        exclusion,
        config,
        maximum_candidates=1,
    )
    return candidates[0] if candidates else None


def refine_chain_from_prior(
    probability,
    prior,
    exclusion=None,
    search_radius=4,
    root_lock_points=4,
    offset_change_penalty=0.08,
    offset_magnitude_penalty=0.005,
):
    """Refit an ordered tube chain locally without changing its topology."""
    probability = np.asarray(probability, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError("probability must be a two-dimensional image")
    if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 2:
        raise ValueError("prior must have shape (points, 2) with two points")
    if exclusion is None:
        exclusion = np.zeros(probability.shape, dtype=bool)
    else:
        exclusion = np.asarray(exclusion, dtype=bool)
        if exclusion.shape != probability.shape:
            raise ValueError("exclusion must match probability")
    search_radius = max(0, int(search_radius))
    offsets = np.arange(-search_radius, search_radius + 1, dtype=np.float64)
    tangents = np.gradient(prior, axis=0)
    tangents /= np.maximum(
        np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6
    )
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    candidates = prior[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    map_x = candidates[:, :, 1].astype(np.float32)
    map_y = candidates[:, :, 0].astype(np.float32)
    evidence = cv.remap(
        probability,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    excluded = cv.remap(
        exclusion.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=1,
    ).astype(bool)
    evidence[excluded] = -np.inf
    center_offset = search_radius
    locked = min(max(1, int(root_lock_points)), len(prior))
    evidence[:locked] = -np.inf
    evidence[:locked, center_offset] = cv.remap(
        probability,
        prior[:locked, 1].astype(np.float32)[:, None],
        prior[:locked, 0].astype(np.float32)[:, None],
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )[:, 0]

    state_count = len(offsets)
    scores = np.full((len(prior), state_count), -np.inf, dtype=np.float64)
    back = np.full((len(prior), state_count), -1, dtype=int)
    scores[0] = evidence[0]
    for point_index in range(1, len(prior)):
        for state in range(state_count):
            if not np.isfinite(evidence[point_index, state]):
                continue
            start = max(0, state - 1)
            stop = min(state_count, state + 2)
            previous = scores[point_index - 1, start:stop]
            if not np.any(np.isfinite(previous)):
                continue
            previous_state = start + int(np.argmax(previous))
            scores[point_index, state] = (
                scores[point_index - 1, previous_state]
                + evidence[point_index, state]
                - float(offset_change_penalty)
                * abs(offsets[state] - offsets[previous_state])
                - float(offset_magnitude_penalty) * abs(offsets[state])
            )
            back[point_index, state] = previous_state
    state = int(np.argmax(scores[-1]))
    if not np.isfinite(scores[-1, state]):
        return prior.copy(), 0.0, 0.0
    states = np.empty(len(prior), dtype=int)
    for point_index in range(len(prior) - 1, -1, -1):
        states[point_index] = state
        state = back[point_index, state]
        if point_index and state < 0:
            return prior.copy(), 0.0, 0.0
    selected_offsets = offsets[states]
    if len(selected_offsets) >= 5:
        selected_offsets = np.convolve(
            np.pad(selected_offsets, (2, 2), mode="edge"),
            np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0,
            mode="valid",
        )
        selected_offsets[:locked] = 0.0
    refined = prior + selected_offsets[:, None] * normals
    support = evidence[np.arange(len(prior)), states]
    finite = np.isfinite(support)
    mean_support = float(np.mean(support[finite])) if np.any(finite) else 0.0
    median_shift = float(np.median(np.abs(selected_offsets)))
    return refined, mean_support, median_shift


def _local_change_maps(previous_gray, current_gray, grain_delta, exclusion):
    """Measure newly dark and absolute changes after local grain alignment."""
    transform = np.float32(
        [[1.0, 0.0, grain_delta[1]], [0.0, 1.0, grain_delta[0]]]
    )
    aligned = cv.warpAffine(
        previous_gray,
        transform,
        (current_gray.shape[1], current_gray.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT,
    )
    newly_dark = cv.subtract(aligned, current_gray)
    temporal = robust_normalize(newly_dark.astype(np.float32))
    temporal = cv.GaussianBlur(temporal, (0, 0), 1.0)
    temporal[exclusion] = 0.0
    absolute = cv.absdiff(aligned, current_gray).astype(np.float32) / 255.0
    absolute = cv.GaussianBlur(absolute, (0, 0), 0.8)
    absolute[exclusion] = 0.0
    return temporal, absolute


def detect_tip_burst(change, tip_yx, config):
    """Flag a broad connected change around a confirmed tube tip."""
    radius = config.burst_search_radius
    left, top, right, bottom = _crop_bounds(tip_yx, radius, change.shape)
    local = change[top:bottom, left:right]
    if local.size == 0:
        return False, 0.0, ""
    local_tip = np.asarray(tip_yx) - np.array([top, left], dtype=np.float64)
    disk = np.zeros(local.shape, dtype=np.uint8)
    cv.circle(
        disk,
        tuple(np.rint(local_tip[::-1]).astype(int)),
        radius,
        1,
        -1,
    )
    values = local[disk.astype(bool)]
    if values.size == 0:
        return False, 0.0, ""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.07, median + 3.5 * max(mad, 0.005))
    changed = ((local >= threshold) & disk.astype(bool)).astype(np.uint8)
    changed = cv.morphologyEx(
        changed,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, _, stats, _ = cv.connectedComponentsWithStats(changed, 8)
    maximum_area = (
        int(np.max(stats[1:, cv.CC_STAT_AREA])) if component_count > 1 else 0
    )
    changed_fraction = float(np.mean(changed[disk.astype(bool)] > 0))
    changed_values = local[changed.astype(bool)]
    mean_strength = float(np.mean(changed_values)) if changed_values.size else 0.0
    area_score = min(1.0, maximum_area / (2.0 * config.burst_min_component_area))
    fraction_score = min(
        1.0, changed_fraction / (2.0 * config.burst_min_changed_fraction)
    )
    strength_score = min(1.0, mean_strength / 0.18)
    score = float(0.40 * area_score + 0.35 * fraction_score + 0.25 * strength_score)
    candidate = (
        maximum_area >= config.burst_min_component_area
        and changed_fraction >= config.burst_min_changed_fraction
        and score >= config.burst_score_threshold
    )
    reason = (
        f"broad tip change: area={maximum_area}px, "
        f"fraction={changed_fraction:.3f}, strength={mean_strength:.3f}"
        if candidate
        else ""
    )
    return candidate, score, reason


def trace_anchored_curves(
    frames,
    grain_tracks,
    config,
    masking_tracks=None,
    x_scale=1.0,
    y_scale=1.0,
    progress_callback=None,
):
    """Trace tubes from pollen boundaries and grow them only by tip extension."""
    if not frames:
        return []
    masking_tracks = masking_tracks or grain_tracks
    measurements = []
    previous_gray = None
    previous_curves = {}
    previous_centers = {}
    pending = {}
    burst_events = {}
    tube_history = {}
    for frame_index, frame in enumerate(frames):
        current_gray = prepare_analysis_gray(frame, config)
        if previous_gray is None:
            aligned_previous = None
            shift_x = shift_y = response = 0.0
        else:
            aligned_previous, shift_x, shift_y, response = register_translation(
                previous_gray, current_gray
            )
        _, probability, global_temporal = enhance_tube_probability(
            frame, aligned_previous, config
        )
        for track in grain_tracks:
            if frame_index < track.first_frame():
                continue
            exclusion, grain, crowding = build_grain_exclusion(
                probability.shape, masking_tracks, frame_index, track
            )
            center = np.array([grain.gv3.y, grain.gv3.x], dtype=np.float64)
            radius = max(grain.w, grain.h) / 2.0
            previous_center = previous_centers.get(track.id)
            grain_delta = (
                np.array([shift_y, shift_x], dtype=np.float64)
                if previous_center is None
                else center - previous_center
            )
            previous = previous_curves.get(track.id)
            translated_prior = None if previous is None else previous + grain_delta
            if previous_gray is None:
                local_temporal = global_temporal
                local_change = np.zeros_like(probability)
            else:
                local_temporal, local_change = _local_change_maps(
                    previous_gray, current_gray, grain_delta, exclusion
                )
            candidate_points = None
            structure_score = temporal_score = confidence = 0.0
            deviation = None
            used_prior = False
            burst_event = burst_events.get(track.id)
            history = tube_history.get(track.id)
            if (
                translated_prior is not None
                and burst_event is None
                and history is not None
                and frame_index - history["confirmed_frame"]
                >= config.burst_min_confirmed_frames
                and history["direct_extensions"]
                >= config.burst_min_direct_extensions
                and curve_length(translated_prior, x_scale, y_scale)
                >= config.burst_min_curve_length
                and grain.detection_method != "motion_predicted"
            ):
                is_candidate, burst_score, burst_reason = detect_tip_burst(
                    local_change, translated_prior[-1], config
                )
                if is_candidate:
                    burst_event = {
                        "frame": frame_index,
                        "score": burst_score,
                        "reason": burst_reason,
                    }
                    burst_events[track.id] = burst_event

            if burst_event is not None:
                points = translated_prior
                status = (
                    "burst_candidate"
                    if burst_event["frame"] == frame_index
                    else "post_burst_hold"
                )
                structure_score = 0.0
                temporal_score = burst_event["score"]
                confidence = burst_event["score"]
                used_prior = True
            elif translated_prior is None:
                prior_candidate = pending.get(track.id)
                if prior_candidate is not None:
                    translated_candidate = (
                        prior_candidate["points"]
                        + center
                        - prior_candidate["center"]
                    )
                    extension = find_tip_extension(
                        probability,
                        local_temporal,
                        translated_candidate,
                        exclusion,
                        config,
                    )
                    if extension is None:
                        pending.pop(track.id, None)
                        candidate_points = None
                        candidate_state = None
                    else:
                        _, extension_points, structure_score, temporal_score = extension
                        candidate_points = resample_curve(
                            np.vstack(
                                [translated_candidate, extension_points[1:]]
                            ),
                            max_points=192,
                        )
                        candidate_length = curve_length(
                            candidate_points, x_scale, y_scale
                        )
                        candidate_state = {
                            "points": candidate_points,
                            "center": center,
                            "confirmations": prior_candidate["confirmations"] + 1,
                            "initial_length": prior_candidate["initial_length"],
                            "maximum_length": max(
                                prior_candidate["maximum_length"], candidate_length
                            ),
                            "radial_progress": prior_candidate["radial_progress"],
                        }
                else:
                    birth = find_attached_birth_path(
                        probability,
                        local_temporal,
                        center,
                        radius,
                        exclusion,
                        config,
                    )
                    if birth is None:
                        candidate_state = None
                    else:
                        _, candidate_points, metrics = birth
                        structure_score = metrics["support"]
                        temporal_score = metrics["temporal"]
                        initial_length = curve_length(
                            candidate_points, x_scale, y_scale
                        )
                        candidate_state = {
                            "points": candidate_points,
                            "center": center,
                            "confirmations": 1,
                            "initial_length": initial_length,
                            "maximum_length": initial_length,
                            "radial_progress": metrics["radial_progress"],
                        }

                if candidate_state is None:
                    points = None
                    status = "no_attached_tube"
                else:
                    acceptable = (
                        temporal_score >= config.min_birth_temporal_score
                        and crowding == 0
                        and grain.detection_method != "motion_predicted"
                    )
                    if acceptable:
                        pending[track.id] = candidate_state
                    else:
                        pending.pop(track.id, None)
                        candidate_state["confirmations"] = 0
                    growth = (
                        candidate_state["maximum_length"]
                        - candidate_state["initial_length"]
                    )
                    if (
                        candidate_state["confirmations"]
                        >= config.birth_confirmation_frames
                        and growth >= config.min_confirmation_growth
                    ):
                        points = resample_curve(candidate_points, max_points=192)
                        previous_curves[track.id] = points
                        tube_history[track.id] = {
                            "confirmed_frame": frame_index,
                            "direct_extensions": 0,
                        }
                        pending.pop(track.id, None)
                        candidate_points = None
                        status = "attached_growth_confirmed"
                        confidence = float(
                            np.clip(
                                0.55 * structure_score
                                + 0.25 * temporal_score
                                + 0.20 * candidate_state["radial_progress"],
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        points = None
                        status = (
                            "candidate_crowded"
                            if crowding
                            else "candidate_unconfirmed"
                        )
            else:
                refined_prior, refit_support, refit_shift = (
                    refine_chain_from_prior(
                        probability,
                        translated_prior,
                        exclusion=exclusion,
                        search_radius=config.chain_refit_radius,
                        root_lock_points=config.chain_root_lock_points,
                        offset_change_penalty=(
                            config.chain_refit_offset_change_penalty
                        ),
                        offset_magnitude_penalty=(
                            config.chain_refit_offset_magnitude_penalty
                        ),
                    )
                )
                prior_length = curve_length(translated_prior)
                refit_length = curve_length(refined_prior)
                length_change = abs(refit_length - prior_length) / max(
                    prior_length, 1.0
                )
                if (
                    refit_support >= config.chain_refit_min_support
                    and length_change
                    <= config.chain_refit_max_length_fraction
                ):
                    translated_prior = refined_prior
                    deviation = refit_shift
                extension = find_tip_extension(
                    probability,
                    local_temporal,
                    translated_prior,
                    exclusion,
                    config,
                )
                if extension is None:
                    points = translated_prior
                    status = "held_no_extension"
                    structure_score = float(
                        np.mean(
                            probability[
                                np.clip(
                                    np.rint(points[:, 0]).astype(int),
                                    0,
                                    probability.shape[0] - 1,
                                ),
                                np.clip(
                                    np.rint(points[:, 1]).astype(int),
                                    0,
                                    probability.shape[1] - 1,
                                ),
                            ]
                        )
                    )
                    confidence = 0.18 * structure_score
                    used_prior = True
                else:
                    _, extension_points, structure_score, temporal_score = extension
                    joined = np.vstack([translated_prior, extension_points[1:]])
                    points = resample_curve(joined, spacing=2.0, max_points=256)
                    status = "tip_extended"
                    confidence = float(
                        np.clip(
                            0.55 * structure_score + 0.45 * temporal_score,
                            0.0,
                            1.0,
                        )
                    )
                    used_prior = True
                    previous_curves[track.id] = points
                    tube_history[track.id]["direct_extensions"] += 1

            if points is not None:
                previous_curves[track.id] = points
            previous_centers[track.id] = center
            measurements.append(
                CurveMeasurement(
                    grain_id=track.id,
                    analysis_frame=frame_index,
                    grain_x=float(center[1]),
                    grain_y=float(center[0]),
                    grain_radius=float(radius),
                    points=points,
                    global_shift_x=shift_x,
                    global_shift_y=shift_y,
                    registration_response=response,
                    grain_shift_x=float(grain_delta[1]),
                    grain_shift_y=float(grain_delta[0]),
                    structure_score=structure_score,
                    temporal_score=temporal_score,
                    prior_deviation=deviation,
                    confidence=confidence,
                    status=status,
                    used_prior=used_prior,
                    grain_tracking_status=grain.detection_method,
                    candidate_points=candidate_points,
                    burst_candidate=burst_event is not None,
                    burst_candidate_frame=(
                        burst_event["frame"] if burst_event is not None else -1
                    ),
                    burst_score=(
                        burst_event["score"] if burst_event is not None else 0.0
                    ),
                    burst_reason=(
                        burst_event["reason"] if burst_event is not None else ""
                    ),
                )
            )
        previous_gray = current_gray
        if progress_callback is not None:
            progress_callback(frame_index + 1, len(frames))
    return measurements
