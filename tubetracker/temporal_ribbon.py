"""Track a pollen-owned ribbon backward from a clearly mature centerline.

Offline microscopy analysis can use the complete movie: a mature tube provides
an unambiguous identity, while each earlier frame determines how much of that
same root-connected curve was already present.  This module deforms the whole
curve between samples and limits backward propagation to supported shortening.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math

import cv2 as cv
import numpy as np
from skimage.graph import MCP_Geometric

from tubetracker.orientation_worldsheet import (
    CoupledRibbonTraceConfig,
    CoupledRibbonTraceResult,
    PairedWallOrientationResult,
    propose_pollen_roots,
    trace_coupled_ribbon_lifted,
)
from tubetracker.global_ribbon_worldsheet import (
    GlobalRibbonConfig,
    GlobalRibbonResult,
    RibbonHypothesis,
    select_global_ribbon_worldsheet,
)


@dataclass(frozen=True)
class TemporalRibbonConfig:
    """Configure backward deformation and optical acceptance of one tube."""

    crop_size: int = 180
    portal_angle_offsets_degrees: tuple[float, ...] = (0.0,)
    minimum_length_px: float = 2.5
    minimum_paired_fraction: float = 0.16
    minimum_mean_paired_support: float = 0.025
    minimum_radial_extension_px: float = 2.5
    minimum_endpoint_radial_efficiency: float = 0.55
    minimum_contact_radial_excursion_radii: float = 1.0
    minimum_radial_excursion_radii: float = 1.5
    minimum_curved_radial_excursion_radii: float = 2.0
    minimum_left_censored_baseline_excursion_radii: float = 1.5
    minimum_sustained_growth_score: float = 0.1
    portal_exit_radius_factor: float = 1.8
    portal_radius_margin_px: float = 0.75
    portal_connector_cost_range: float = 2.0
    portal_minimum_distal_points: int = 4
    return_proximity_px: float = 10.0
    return_minimum_index_gap: int = 8
    prior_error_weight: float = 0.80
    absence_persistence: int = 3
    worldsheet_alternative_paths: int = 12
    worldsheet_maximum_gap_samples: int = 2
    mature_root_proposals: int = 12
    mature_alternatives_per_root: int = 2
    mature_root_separation_degrees: float = 15.0
    trace: CoupledRibbonTraceConfig = CoupledRibbonTraceConfig(
        step_px=1.0,
        maximum_turn_bins=1,
        curvature_penalty=0.11,
        minimum_pair_support=0.035,
        minimum_wall_balance=0.08,
        paired_support_weight=1.35,
        merged_support_weight=0.18,
        wall_balance_weight=0.04,
        evidence_floor=0.047,
        length_reward=0.008,
        width_change_penalty=0.08,
        maximum_reacquisition_width_change_px=1.7,
        maximum_gap_steps=5,
        maximum_initial_gap_steps=3,
        root_occlusion_px=2.0,
        beam_width=360,
        minimum_length_px=2.0,
        maximum_length_px=100.0,
        minimum_endpoint_separation_fraction=0.68,
        endpoint_openness_reward=0.12,
        maximum_extension_px=3.0,
        prior_position_penalty=0.18,
        prior_direction_penalty=0.10,
        prior_release_px=1.5,
        extension_position_penalty=0.12,
        extension_direction_penalty=0.55,
    )


@dataclass(frozen=True)
class TemporalRibbonFrame:
    """Store one dynamic centerline and the evidence supporting it."""

    path_relative_yx: np.ndarray
    length_px: float
    paired_fraction: float
    mean_paired_support: float
    prior_error_px: float
    accepted: bool
    interpolated: bool = False
    observed_length_px: float | None = None
    temporally_confirmed: bool = True


@dataclass(frozen=True)
class TemporalRibbonHistory:
    """Store the complete backward reconstruction for one pollen tube."""

    frames: tuple[TemporalRibbonFrame, ...]
    first_persistent_sample: int | None
    portal_rebased: bool = False
    portal_removed_prefix_px: float = 0.0
    portal_connector_length_px: float = 0.0

    @property
    def lengths_px(self) -> np.ndarray:
        """Return one centerline length for every sampled frame."""

        return np.asarray([frame.length_px for frame in self.frames])


def endpoint_radial_efficiency(frame: TemporalRibbonFrame) -> float:
    """Measure net outward progress from the pollen relative to curve length."""

    path = np.asarray(frame.path_relative_yx, dtype=np.float64)
    if frame.length_px <= np.finfo(np.float64).eps or len(path) < 2:
        return 0.0
    radial_gain = float(np.linalg.norm(path[-1]) - np.linalg.norm(path[0]))
    return radial_gain / frame.length_px


def radial_excursion_efficiency(frame: TemporalRibbonFrame) -> float:
    """Measure maximum pollen departure without penalizing a later curve."""

    path = np.asarray(frame.path_relative_yx, dtype=np.float64)
    if frame.length_px <= np.finfo(np.float64).eps or len(path) < 2:
        return 0.0
    radial_distance = np.linalg.norm(path, axis=1)
    radial_gain = float(np.max(radial_distance) - radial_distance[0])
    return radial_gain / frame.length_px


def radial_excursion_px(frame: TemporalRibbonFrame) -> float:
    """Return the greatest radial departure beyond the attachment point."""

    path = np.asarray(frame.path_relative_yx, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    radial_distance = np.linalg.norm(path, axis=1)
    return float(max(0.0, np.max(radial_distance) - radial_distance[0]))


@dataclass(frozen=True)
class PollenPortalRebase:
    """Describe replacement of a grain-boundary walk by its true exit."""

    path_yx: np.ndarray
    succeeded: bool
    removed_prefix_px: float
    connector_length_px: float


def rebase_pollen_portal(
    path_yx: np.ndarray,
    features: PairedWallOrientationResult,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    config: TemporalRibbonConfig | None = None,
) -> PollenPortalRebase:
    """Replace a pollen-rim prefix with the shortest supported body exit.

    A genuine tube has one final, irreversible departure from the pollen body.
    Earlier excursions beyond the body halo can be portions of the grain rim.
    Starting at that final departure, a local geodesic reconnects the distal
    tube to the pollen boundary without inheriting the longer rim walk.
    """

    config = config or TemporalRibbonConfig()
    path = np.asarray(path_yx, dtype=np.float64)
    center = np.asarray(pollen_center_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape (points, 2)")
    if center.shape != (2,) or pollen_radius_px <= 0.0:
        raise ValueError("pollen center and radius must be valid")
    if config.portal_exit_radius_factor <= 1.0:
        raise ValueError("portal exit radius factor must exceed one")
    if config.portal_radius_margin_px < 0.0:
        raise ValueError("portal radius margin cannot be negative")
    if config.portal_connector_cost_range < 0.0:
        raise ValueError("portal connector cost range cannot be negative")
    if config.portal_minimum_distal_points < 2:
        raise ValueError("portal minimum distal points must be at least two")

    arrays = (features.score, features.paired_score)
    if any(array.ndim != 3 for array in arrays) or arrays[0].shape != arrays[1].shape:
        raise ValueError("ribbon feature volumes must share orientation and image axes")
    if arrays[0].shape[1:] != arrays[1].shape[1:]:
        raise ValueError("ribbon feature images must share shape")

    distance = np.linalg.norm(path - center, axis=1)
    exit_radius = config.portal_exit_radius_factor * pollen_radius_px
    inside = np.flatnonzero(distance < exit_radius)
    if not len(inside):
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)
    exit_index = int(inside[-1] + 1)
    if exit_index + config.portal_minimum_distal_points > len(path):
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)

    portal_radius = pollen_radius_px + config.portal_radius_margin_px
    height, width = arrays[0].shape[1:]
    grid_y, grid_x = np.mgrid[:height, :width]
    radial = np.hypot(grid_y - center[0], grid_x - center[1])
    target_ring = np.abs(radial - portal_radius) <= 0.75
    search_region = (
        (radial >= max(0.0, portal_radius - 1.0))
        & (radial <= exit_radius + 1.5 * config.trace.step_px)
    )
    target_ring &= search_region
    if not np.any(target_ring):
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)

    evidence = np.maximum(
        np.max(np.asarray(features.score, dtype=np.float64), axis=0),
        np.max(np.asarray(features.paired_score, dtype=np.float64), axis=0),
    )
    evidence = np.clip(evidence, 0.0, 1.0)
    costs = 1.0 + config.portal_connector_cost_range * (1.0 - evidence)
    costs[~search_region] = np.inf
    start = np.rint(path[exit_index]).astype(int)
    start = np.clip(start, (0, 0), (height - 1, width - 1))
    if not np.isfinite(costs[tuple(start)]):
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)

    solver = MCP_Geometric(costs, fully_connected=True)
    cumulative, _ = solver.find_costs([tuple(start)])
    reachable_targets = target_ring & np.isfinite(cumulative)
    if not np.any(reachable_targets):
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)
    target_flat = int(np.argmin(np.where(reachable_targets, cumulative, np.inf)))
    target = np.unravel_index(target_flat, cumulative.shape)
    route = np.asarray(solver.traceback(target), dtype=np.float64)
    if len(route) < 2:
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)
    if np.linalg.norm(route[0] - start) > np.linalg.norm(route[-1] - start):
        route = route[::-1]

    connector = route[::-1]
    portal_vector = connector[0] - center
    portal_distance = float(np.linalg.norm(portal_vector))
    if portal_distance <= np.finfo(np.float64).eps:
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)
    connector[0] = center + portal_radius * portal_vector / portal_distance
    connector[-1] = path[exit_index]
    rebased = np.vstack((connector, path[exit_index + 1 :]))
    removed = _curve_length(path[: exit_index + 1])
    connector_length = _curve_length(connector)
    if connector_length >= removed - config.trace.step_px:
        return PollenPortalRebase(path.copy(), False, 0.0, 0.0)
    return PollenPortalRebase(
        path_yx=rebased,
        succeeded=True,
        removed_prefix_px=removed,
        connector_length_px=connector_length,
    )


def _resample_path(path_yx: np.ndarray, point_count: int) -> np.ndarray:
    """Resample an ordered curve at evenly spaced material coordinates."""

    path = np.asarray(path_yx, dtype=np.float64)
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    if arc[-1] <= np.finfo(np.float64).eps:
        return np.repeat(path[:1], point_count, axis=0)
    samples = np.linspace(0.0, float(arc[-1]), point_count)
    return np.column_stack(
        [np.interp(samples, arc, path[:, axis]) for axis in range(2)]
    )


def interpolate_sampled_history(
    history: TemporalRibbonHistory,
    sample_indices: np.ndarray,
    frame_count: int,
) -> TemporalRibbonHistory:
    """Expand validated sparse curve anchors across the complete timeline."""

    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.shape != (len(history.frames),):
        raise ValueError("sample indices must match the history length")
    if frame_count < 1 or not len(indices):
        raise ValueError("frame count and sampled history must be nonempty")
    if (
        indices[0] < 0
        or indices[-1] >= frame_count
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("sample indices must be ordered within the target timeline")

    root = np.asarray(history.frames[-1].path_relative_yx[0], dtype=np.float64)
    expanded = [_absent_frame(root) for _ in range(frame_count)]
    for index, frame in zip(indices, history.frames):
        expanded[int(index)] = frame

    for position, (left_index, right_index) in enumerate(
        zip(indices, indices[1:])
    ):
        left = history.frames[position]
        right = history.frames[position + 1]
        if not left.accepted or not right.accepted:
            continue
        point_count = max(
            len(left.path_relative_yx),
            len(right.path_relative_yx),
            2,
        )
        left_path = _resample_path(left.path_relative_yx, point_count)
        right_path = _resample_path(right.path_relative_yx, point_count)
        span = int(right_index - left_index)
        for offset in range(1, span):
            fraction = offset / span
            path = (1.0 - fraction) * left_path + fraction * right_path
            target_length = (
                (1.0 - fraction) * left.length_px
                + fraction * right.length_px
            )
            path = _match_curve_length(path, target_length)
            expanded[int(left_index) + offset] = TemporalRibbonFrame(
                path_relative_yx=path,
                length_px=target_length,
                paired_fraction=(
                    (1.0 - fraction) * left.paired_fraction
                    + fraction * right.paired_fraction
                ),
                mean_paired_support=(
                    (1.0 - fraction) * left.mean_paired_support
                    + fraction * right.mean_paired_support
                ),
                prior_error_px=(
                    (1.0 - fraction) * left.prior_error_px
                    + fraction * right.prior_error_px
                ),
                accepted=True,
                interpolated=True,
                observed_length_px=None,
                temporally_confirmed=(
                    left.temporally_confirmed and right.temporally_confirmed
                ),
            )

    first_persistent = history.first_persistent_sample
    return TemporalRibbonHistory(
        frames=tuple(expanded),
        first_persistent_sample=(
            None if first_persistent is None else int(indices[first_persistent])
        ),
        portal_rebased=history.portal_rebased,
        portal_removed_prefix_px=history.portal_removed_prefix_px,
        portal_connector_length_px=history.portal_connector_length_px,
    )


@dataclass(frozen=True)
class _TemporalRibbonCandidate:
    """Keep one local curve together with its global-sequence measurements."""

    frame: TemporalRibbonFrame
    hypothesis: RibbonHypothesis
    objective: float


def _curve_length(path_yx: np.ndarray) -> float:
    """Return centerline arclength for one ordered path."""

    path = np.asarray(path_yx, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def truncate_antiparallel_return(
    path_yx: np.ndarray,
    *,
    proximity_px: float = 10.0,
    minimum_index_gap: int = 8,
) -> np.ndarray:
    """Remove a false trip back along the opposite boundary of one tube."""

    path = np.asarray(path_yx, dtype=np.float64)
    if len(path) < minimum_index_gap + 3:
        return path.copy()
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    root_distance = np.linalg.norm(path - path[0], axis=1)
    for later in range(minimum_index_gap, len(path)):
        earlier_limit = later - minimum_index_gap
        separation = np.linalg.norm(path[:earlier_limit] - path[later], axis=1)
        opposition = tangent[:earlier_limit] @ tangent[later]
        returning = np.flatnonzero(
            (separation <= proximity_px) & (opposition <= -0.35)
        )
        if not len(returning):
            continue
        earlier = int(returning[np.argmin(separation[returning])])
        turn = earlier + int(np.argmax(root_distance[earlier : later + 1]))
        if turn >= 2:
            return path[: turn + 1].copy()
    return path.copy()


def terminates_at_foreign_owner(
    path_yx: np.ndarray,
    foreign_centers_yx: np.ndarray,
    owner_radius_px: float,
    *,
    clearance_radii: float = 1.8,
    distal_fraction: float = 0.25,
) -> bool:
    """Return whether a proposed distal tip enters another pollen grain."""

    path = np.asarray(path_yx, dtype=np.float64)
    centers = np.asarray(foreign_centers_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (points, 2)")
    if centers.size == 0:
        return False
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("foreign centers must have shape (owners, 2)")
    if owner_radius_px <= 0.0 or not 0.0 < distal_fraction <= 1.0:
        raise ValueError("owner radius and distal fraction must be positive")
    start = max(1, int(math.floor((1.0 - distal_fraction) * len(path))))
    distance = np.linalg.norm(
        path[start:, None] - centers[None],
        axis=2,
    )
    return bool(np.min(distance) <= clearance_radii * owner_radius_px)


def intersects_foreign_owner(
    path_yx: np.ndarray,
    foreign_centers_yx: np.ndarray,
    owner_radius_px: float,
    *,
    clearance_radii: float = 1.4,
    root_clearance_radii: float = 1.5,
) -> bool:
    """Return whether a tube passes through another pollen after leaving its root."""

    path = np.asarray(path_yx, dtype=np.float64)
    centers = np.asarray(foreign_centers_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (points, 2)")
    if centers.size == 0:
        return False
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("foreign centers must have shape (owners, 2)")
    if owner_radius_px <= 0.0:
        raise ValueError("owner_radius_px must be positive")
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    distal = path[arc >= root_clearance_radii * owner_radius_px]
    if not len(distal):
        return False
    distance = np.linalg.norm(distal[:, None] - centers[None], axis=2)
    return bool(np.min(distance) <= clearance_radii * owner_radius_px)


def truncate_at_foreign_owner(
    path_yx: np.ndarray,
    foreign_centers_yx: np.ndarray,
    owner_radius_px: float,
    *,
    clearance_radii: float = 1.5,
    root_clearance_radii: float = 1.5,
) -> np.ndarray:
    """Keep only the owner-connected prefix before the first foreign body."""

    path = np.asarray(path_yx, dtype=np.float64)
    centers = np.asarray(foreign_centers_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (points, 2)")
    if centers.size == 0:
        return path.copy()
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("foreign centers must have shape (owners, 2)")
    if owner_radius_px <= 0.0:
        raise ValueError("owner_radius_px must be positive")
    if clearance_radii <= 0.0 or root_clearance_radii < 0.0:
        raise ValueError("clearance factors must be nonnegative")

    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    eligible = arc >= root_clearance_radii * owner_radius_px
    distances = np.linalg.norm(path[:, None] - centers[None], axis=2)
    collisions = np.flatnonzero(
        eligible & (np.min(distances, axis=1) <= clearance_radii * owner_radius_px)
    )
    if not len(collisions):
        return path.copy()
    stop = max(1, int(collisions[0]))
    return path[:stop].copy()


def _truncate_curve(path_yx: np.ndarray, maximum_length_px: float) -> np.ndarray:
    """Cut a curve at an interpolated arclength without changing its prefix."""

    path = np.asarray(path_yx, dtype=np.float64)
    if len(path) < 2 or maximum_length_px <= 0.0:
        return path[:1].copy()
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segments)))
    if arc[-1] <= maximum_length_px:
        return path.copy()
    end = int(np.searchsorted(arc, maximum_length_px, side="right"))
    before = end - 1
    fraction = (maximum_length_px - arc[before]) / max(
        arc[end] - arc[before], 1e-9
    )
    endpoint = path[before] + fraction * (path[end] - path[before])
    return np.vstack((path[:end], endpoint))


def _match_curve_length(path_yx: np.ndarray, target_length_px: float) -> np.ndarray:
    """Preserve curve shape while matching an interpolated material length."""

    path = np.asarray(path_yx, dtype=np.float64)
    if target_length_px <= 0.0 or len(path) < 2:
        return path[:1].copy()
    actual_length = _curve_length(path)
    if actual_length <= np.finfo(np.float64).eps:
        return path[:1].copy()
    if actual_length > target_length_px:
        return _truncate_curve(path, target_length_px)
    if actual_length < target_length_px:
        return path[0] + (path - path[0]) * (target_length_px / actual_length)
    return path.copy()


def regularize_monotone_history(
    history: TemporalRibbonHistory,
) -> TemporalRibbonHistory:
    """Fit accepted material lengths to a nondecreasing evidence-weighted path."""

    accepted = [
        index for index, frame in enumerate(history.frames) if frame.accepted
    ]
    if len(accepted) < 2:
        return history
    values = np.asarray(
        [history.frames[index].length_px for index in accepted],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            max(
                history.frames[index].paired_fraction
                * history.frames[index].mean_paired_support,
                1e-3,
            )
            for index in accepted
        ],
        dtype=np.float64,
    )
    blocks: list[list[float | int]] = []
    for position, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(value), float(weight), position, position + 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left[1]) + float(right[1])
            merged_value = (
                float(left[0]) * float(left[1])
                + float(right[0]) * float(right[1])
            ) / merged_weight
            blocks.append([merged_value, merged_weight, left[2], right[3]])

    fitted = np.empty_like(values)
    for value, _, start, end in blocks:
        fitted[int(start) : int(end)] = float(value)

    frames = list(history.frames)
    for position, target_length in zip(accepted, fitted):
        frame = frames[position]
        if abs(float(target_length) - frame.length_px) <= 1e-9:
            continue
        path = _match_curve_length(frame.path_relative_yx, float(target_length))
        frames[position] = replace(
            frame,
            path_relative_yx=path,
            length_px=_curve_length(path),
            observed_length_px=(
                frame.length_px
                if frame.observed_length_px is None
                else frame.observed_length_px
            ),
        )
    return replace(history, frames=tuple(frames))


def _resample_curve(path_yx: np.ndarray, point_count: int) -> np.ndarray:
    """Resample an ordered curve at uniform normalized arclength."""

    path = np.asarray(path_yx, dtype=np.float64)
    if len(path) < 2 or point_count < 2:
        return np.repeat(path[:1], max(point_count, 1), axis=0)
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    if arc[-1] <= 1e-9:
        return np.repeat(path[:1], point_count, axis=0)
    samples = np.linspace(0.0, arc[-1], point_count)
    return np.column_stack(
        [np.interp(samples, arc, path[:, axis]) for axis in range(2)]
    )


def _centered_crop(image: np.ndarray, center_yx: np.ndarray, size: int) -> np.ndarray:
    """Extract a fixed-size crop, padding only when an owner nears an edge."""

    if size < 8 or size % 2:
        raise ValueError("crop size must be an even integer of at least eight")
    center = np.rint(center_yx).astype(int)
    half = size // 2
    top, left = int(center[0] - half), int(center[1] - half)
    bottom, right = top + size, left + size
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - image.shape[0])
    pad_right = max(0, right - image.shape[1])
    if pad_top or pad_left or pad_bottom or pad_right:
        image = cv.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv.BORDER_REFLECT_101,
        )
        top += pad_top
        left += pad_left
    return image[top : top + size, left : left + size]


def _rotate(vector_yx: np.ndarray, angle_radians: float) -> np.ndarray:
    """Rotate a row-column vector in the image plane."""

    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    y, x = np.asarray(vector_yx, dtype=np.float64)
    return np.asarray((cosine * y + sine * x, -sine * y + cosine * x))


def _absent_frame(root_relative_yx: np.ndarray) -> TemporalRibbonFrame:
    """Represent a sampled frame with no supported root-connected tube."""

    return TemporalRibbonFrame(
        path_relative_yx=np.asarray(root_relative_yx, dtype=np.float64)[None],
        length_px=0.0,
        paired_fraction=0.0,
        mean_paired_support=0.0,
        prior_error_px=0.0,
        accepted=False,
    )


def bridge_short_tracking_gaps(
    frames: list[TemporalRibbonFrame],
    maximum_gap: int,
) -> None:
    """Interpolate complete curves across brief bounded optical dropouts."""

    if maximum_gap < 1:
        return
    sample = 1
    while sample < len(frames) - 1:
        if frames[sample].accepted:
            sample += 1
            continue
        start = sample
        while sample < len(frames) and not frames[sample].accepted:
            sample += 1
        gap = sample - start
        if sample >= len(frames) or gap > maximum_gap:
            continue
        left = frames[start - 1]
        right = frames[sample]
        if not left.accepted or not right.accepted:
            continue
        point_count = max(len(left.path_relative_yx), len(right.path_relative_yx))
        left_path = _resample_curve(left.path_relative_yx, point_count)
        right_path = _resample_curve(right.path_relative_yx, point_count)
        for offset in range(gap):
            fraction = (offset + 1) / (gap + 1)
            path = (1.0 - fraction) * left_path + fraction * right_path
            target_length = (
                (1.0 - fraction) * left.length_px
                + fraction * right.length_px
            )
            path = _match_curve_length(path, target_length)
            frames[start + offset] = TemporalRibbonFrame(
                path_relative_yx=path,
                length_px=_curve_length(path),
                paired_fraction=0.0,
                mean_paired_support=0.0,
                prior_error_px=0.0,
                accepted=True,
                interpolated=True,
                observed_length_px=_curve_length(path),
            )


def confirm_persistent_growth(
    frames: list[TemporalRibbonFrame],
    persistence_samples: int,
) -> None:
    """Report new distal growth only after it persists in a later sample."""

    if persistence_samples < 2 or not frames:
        return
    observed = np.asarray([frame.length_px for frame in frames], dtype=np.float64)
    confirmed_length = 0.0
    for sample, frame in enumerate(frames):
        window_end = sample + persistence_samples
        if window_end <= len(frames):
            candidate = float(np.min(observed[sample:window_end]))
            confirmed_length = max(confirmed_length, candidate)
            extension_confirmed = frame.length_px <= confirmed_length + 1e-6
        else:
            extension_confirmed = frame.length_px <= confirmed_length + 1e-6
        retained = min(frame.length_px, confirmed_length)
        path = _truncate_curve(frame.path_relative_yx, retained)
        frames[sample] = replace(
            frame,
            path_relative_yx=path,
            length_px=_curve_length(path),
            observed_length_px=frame.length_px,
            temporally_confirmed=extension_confirmed,
        )


def _candidate_from_trace(
    trace: CoupledRibbonTraceResult,
    crop_center: np.ndarray,
    config: TemporalRibbonConfig,
    source_index: int,
    maximum_length_px: float | None,
) -> _TemporalRibbonCandidate:
    """Convert one lifted trace into comparable local and global records."""

    open_path = truncate_antiparallel_return(
        trace.path_yx,
        proximity_px=config.return_proximity_px,
        minimum_index_gap=config.return_minimum_index_gap,
    )
    path = (
        open_path
        if maximum_length_px is None
        else _truncate_curve(open_path, maximum_length_px)
    )
    length = _curve_length(path)
    paired = trace.paired_support[1 : len(path)]
    balances = trace.wall_balance[1 : len(path)]
    widths = trace.half_width_px[1 : len(path)]
    paired_fraction = (
        float(
            np.mean(
                (paired >= config.trace.minimum_pair_support)
                & (balances >= config.trace.minimum_wall_balance)
                & (widths > 0.0)
            )
        )
        if len(paired)
        else 0.0
    )
    mean_pair = float(np.mean(paired)) if len(paired) else 0.0
    removed_return = trace.length_px - _curve_length(open_path)
    objective = (
        trace.score
        + 0.5 * paired_fraction
        - config.prior_error_weight * trace.prior_prefix_error_px
        - removed_return
    )
    radial_extension = (
        float(np.max(np.linalg.norm(path - crop_center, axis=1)))
        - float(np.linalg.norm(path[0] - crop_center))
    )
    accepted = bool(
        length >= config.minimum_length_px
        and radial_extension >= config.minimum_radial_extension_px
        and paired_fraction >= config.minimum_paired_fraction
        and mean_pair >= config.minimum_mean_paired_support
    )
    relative_path = path - crop_center
    openness = (
        float(np.linalg.norm(path[-1] - path[0]) / max(length, 1e-9))
        if len(path) > 1
        else 0.0
    )
    score_fraction = length / max(trace.length_px, config.trace.step_px)
    frame = TemporalRibbonFrame(
        path_relative_yx=relative_path,
        length_px=length,
        paired_fraction=paired_fraction,
        mean_paired_support=mean_pair,
        prior_error_px=trace.prior_prefix_error_px,
        accepted=accepted,
        observed_length_px=length,
    )
    return _TemporalRibbonCandidate(
        frame=frame,
        hypothesis=RibbonHypothesis(
            path_yx=relative_path,
            optical_score=float(trace.score * score_fraction),
            paired_fraction=paired_fraction,
            mean_paired_support=mean_pair,
            endpoint_openness=openness,
            source_index=source_index,
        ),
        objective=float(objective),
    )


def _trace_frame_candidates(
    crop: np.ndarray,
    prior: np.ndarray,
    feature_builder: Callable[[np.ndarray], PairedWallOrientationResult],
    config: TemporalRibbonConfig,
    alternative_count: int,
) -> list[_TemporalRibbonCandidate]:
    """Generate distinct complete curves for one owner-centered frame."""

    features = feature_builder(crop)
    crop_center = np.full(2, config.crop_size / 2.0, dtype=np.float64)
    portal = prior[0] - crop_center
    tangent = np.mean(np.diff(prior[: min(6, len(prior))], axis=0), axis=0)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
    candidates = []
    for angle_degrees in config.portal_angle_offsets_degrees:
        angle = math.radians(angle_degrees)
        root = crop_center + _rotate(portal, angle)
        direction = _rotate(tangent, angle)
        trace = trace_coupled_ribbon_lifted(
            features,
            root,
            direction,
            prior_curve_yx=prior,
            config=replace(
                config.trace,
                maximum_length_px=max(
                    config.minimum_length_px,
                    min(
                        config.trace.maximum_length_px,
                        _curve_length(prior) + config.trace.maximum_extension_px,
                    ),
                ),
                maximum_alternative_paths=alternative_count,
            ),
        )
        for source_index, option in enumerate((trace, *trace.alternatives)):
            candidates.append(
                _candidate_from_trace(
                    option,
                    crop_center,
                    config,
                    source_index,
                    maximum_length_px=_curve_length(prior),
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.objective, reverse=True)


def discover_mature_ribbon_candidates(
    crop: np.ndarray,
    pollen_radius_px: float,
    feature_builder: Callable[[np.ndarray], PairedWallOrientationResult],
    config: TemporalRibbonConfig | None = None,
) -> tuple[TemporalRibbonFrame, ...]:
    """Trace plausible mature tubes outward from the complete pollen rim."""

    config = config or TemporalRibbonConfig()
    if pollen_radius_px <= 0.0:
        raise ValueError("pollen_radius_px must be positive")
    features = feature_builder(crop)
    crop_center = np.full(2, config.crop_size / 2.0, dtype=np.float64)
    roots = propose_pollen_roots(
        features.score,
        crop_center,
        pollen_radius_px,
        probe_top_k=3,
        maximum_proposals=config.mature_root_proposals,
        minimum_angle_separation_degrees=config.mature_root_separation_degrees,
    )
    candidates = []
    source_index = 0
    for root in roots:
        trace = trace_coupled_ribbon_lifted(
            features,
            root.root_yx,
            root.direction_yx,
            config=replace(
                config.trace,
                maximum_alternative_paths=config.mature_alternatives_per_root,
            ),
        )
        for option in (trace, *trace.alternatives):
            candidates.append(
                _candidate_from_trace(
                    option,
                    crop_center,
                    config,
                    source_index,
                    maximum_length_px=None,
                )
            )
            source_index += 1
    ranked = sorted(candidates, key=lambda candidate: candidate.objective, reverse=True)
    accepted = [candidate for candidate in ranked if candidate.frame.accepted]
    ordered = []
    while accepted:
        if not ordered:
            selected = 0
        else:
            represented = [
                math.atan2(
                    candidate.frame.path_relative_yx[0, 0],
                    candidate.frame.path_relative_yx[0, 1],
                )
                for candidate in ordered
            ]

            def audit_priority(candidate: _TemporalRibbonCandidate) -> tuple[float, float]:
                angle = math.atan2(
                    candidate.frame.path_relative_yx[0, 0],
                    candidate.frame.path_relative_yx[0, 1],
                )
                separation = min(
                    abs(math.atan2(math.sin(angle - prior), math.cos(angle - prior)))
                    for prior in represented
                )
                return separation, candidate.objective

            selected = max(range(len(accepted)), key=lambda index: audit_priority(accepted[index]))
        ordered.append(accepted.pop(selected))
    ordered.extend(candidate for candidate in ranked if not candidate.frame.accepted)
    return tuple(candidate.frame for candidate in ordered)


def _prepare_mature_path(
    frames: np.ndarray,
    centers_yx: np.ndarray,
    mature_relative_yx: np.ndarray,
    pollen_radius_px: float | None,
    feature_builder: Callable[[np.ndarray], PairedWallOrientationResult],
    config: TemporalRibbonConfig,
) -> tuple[np.ndarray, PollenPortalRebase]:
    """Canonicalize a mature curve at its last supported pollen exit."""

    crop_center = np.full(2, config.crop_size / 2.0, dtype=np.float64)
    mature = truncate_antiparallel_return(
        mature_relative_yx,
        proximity_px=config.return_proximity_px,
        minimum_index_gap=config.return_minimum_index_gap,
    )
    unchanged = PollenPortalRebase(mature.copy(), False, 0.0, 0.0)
    if pollen_radius_px is None:
        return mature, unchanged
    if pollen_radius_px <= 0.0:
        raise ValueError("pollen_radius_px must be positive")
    final_crop = _centered_crop(frames[-1], centers_yx[-1], config.crop_size)
    rebased = rebase_pollen_portal(
        mature + crop_center,
        feature_builder(final_crop),
        crop_center,
        pollen_radius_px,
        config,
    )
    return rebased.path_yx - crop_center, rebased


def trace_mature_ribbon_backward(
    aligned_frames: np.ndarray,
    owner_centers_yx: np.ndarray,
    mature_path_relative_yx: np.ndarray,
    feature_builder: Callable[[np.ndarray], PairedWallOrientationResult],
    config: TemporalRibbonConfig | None = None,
    *,
    pollen_radius_px: float | None = None,
) -> TemporalRibbonHistory:
    """Deform one mature centerline backward to estimate birth and growth."""

    config = config or TemporalRibbonConfig()
    frames = np.asarray(aligned_frames)
    centers = np.asarray(owner_centers_yx, dtype=np.float64)
    mature = np.asarray(mature_path_relative_yx, dtype=np.float64)
    if frames.ndim != 3 or centers.shape != (len(frames), 2):
        raise ValueError("frames and owner centers must share one sample axis")
    if mature.ndim != 2 or mature.shape[1] != 2 or len(mature) < 2:
        raise ValueError("mature path must have shape (points, 2)")
    if config.absence_persistence < 1:
        raise ValueError("absence persistence must be positive")

    crop_center = np.full(2, config.crop_size / 2.0, dtype=np.float64)
    mature, portal_rebase = _prepare_mature_path(
        frames,
        centers,
        mature,
        pollen_radius_px,
        feature_builder,
        config,
    )
    prior = mature + crop_center
    root_relative = mature[0].copy()
    history: list[TemporalRibbonFrame | None] = [None] * len(frames)
    absent_run = 0

    for sample in range(len(frames) - 1, -1, -1):
        crop = _centered_crop(frames[sample], centers[sample], config.crop_size)
        candidates = _trace_frame_candidates(
            crop,
            prior,
            feature_builder,
            config,
            alternative_count=0,
        )
        winner = candidates[0].frame
        if not winner.accepted:
            absent_run += 1
            history[sample] = _absent_frame(root_relative)
            if absent_run >= config.absence_persistence:
                for earlier in range(sample):
                    history[earlier] = _absent_frame(root_relative)
                break
            continue

        absent_run = 0
        prior = winner.path_relative_yx + crop_center
        root_relative = winner.path_relative_yx[0]
        history[sample] = winner

    completed = [
        frame if frame is not None else _absent_frame(root_relative)
        for frame in history
    ]
    active = np.asarray([frame.accepted for frame in completed], dtype=bool)
    first_persistent = None
    for sample in range(0, max(0, len(active) - config.absence_persistence + 1)):
        if np.all(active[sample : sample + config.absence_persistence]):
            first_persistent = sample
            break
    if first_persistent is not None:
        for sample in range(first_persistent):
            completed[sample] = _absent_frame(
                completed[sample].path_relative_yx[0]
            )
        bridge_short_tracking_gaps(
            completed,
            maximum_gap=config.absence_persistence - 1,
        )
        confirm_persistent_growth(completed, persistence_samples=2)
    else:
        completed = [
            _absent_frame(frame.path_relative_yx[0]) for frame in completed
        ]
    return TemporalRibbonHistory(
        frames=tuple(completed),
        first_persistent_sample=first_persistent,
        portal_rebased=portal_rebase.succeeded,
        portal_removed_prefix_px=portal_rebase.removed_prefix_px,
        portal_connector_length_px=portal_rebase.connector_length_px,
    )


def trace_mature_ribbon_worldsheet(
    aligned_frames: np.ndarray,
    owner_centers_yx: np.ndarray,
    mature_path_relative_yx: np.ndarray,
    feature_builder: Callable[[np.ndarray], PairedWallOrientationResult],
    config: TemporalRibbonConfig | None = None,
    *,
    pollen_radius_px: float | None = None,
) -> tuple[TemporalRibbonHistory, GlobalRibbonResult]:
    """Resolve backward curve hypotheses jointly across the complete movie."""

    config = config or TemporalRibbonConfig()
    frames = np.asarray(aligned_frames)
    centers = np.asarray(owner_centers_yx, dtype=np.float64)
    mature = np.asarray(mature_path_relative_yx, dtype=np.float64)
    if frames.ndim != 3 or centers.shape != (len(frames), 2):
        raise ValueError("frames and owner centers must share one sample axis")
    if mature.ndim != 2 or mature.shape[1] != 2 or len(mature) < 2:
        raise ValueError("mature path must have shape (points, 2)")
    if config.worldsheet_alternative_paths < 1:
        raise ValueError("worldsheet_alternative_paths must be positive")
    if not 0.0 <= config.minimum_endpoint_radial_efficiency <= 1.0:
        raise ValueError("minimum endpoint radial efficiency must be within [0, 1]")
    if config.minimum_radial_excursion_radii < 0.0:
        raise ValueError("minimum radial excursion radii cannot be negative")
    if config.minimum_contact_radial_excursion_radii < 0.0:
        raise ValueError("minimum contact radial excursion cannot be negative")
    if config.minimum_left_censored_baseline_excursion_radii < 0.0:
        raise ValueError("left-censored baseline excursion cannot be negative")
    if (
        config.minimum_curved_radial_excursion_radii
        < config.minimum_radial_excursion_radii
    ):
        raise ValueError("curved radial excursion must include the minimum excursion")

    crop_center = np.full(2, config.crop_size / 2.0, dtype=np.float64)
    mature, portal_rebase = _prepare_mature_path(
        frames,
        centers,
        mature,
        pollen_radius_px,
        feature_builder,
        config,
    )
    prior = mature + crop_center
    root_relative = mature[0].copy()
    candidate_sets: list[tuple[RibbonHypothesis, ...]] = [()] * len(frames)
    candidate_records: list[tuple[_TemporalRibbonCandidate, ...]] = [()] * len(frames)

    for sample in range(len(frames) - 1, -1, -1):
        crop = _centered_crop(frames[sample], centers[sample], config.crop_size)
        candidates = _trace_frame_candidates(
            crop,
            prior,
            feature_builder,
            config,
            alternative_count=config.worldsheet_alternative_paths,
        )
        supported = tuple(
            candidate for candidate in candidates if candidate.frame.accepted
        )
        candidate_records[sample] = supported
        candidate_sets[sample] = tuple(
            candidate.hypothesis for candidate in supported
        )
        accepted = next(
            (
                candidate.frame
                for candidate in candidates
                if candidate.frame.accepted and candidate.hypothesis.source_index >= 0
            ),
            None,
        )
        if accepted is not None:
            prior = accepted.path_relative_yx + crop_center
            root_relative = accepted.path_relative_yx[0]

    if candidate_records[-1]:
        mature_length = max(
            candidate.frame.length_px for candidate in candidate_records[-1]
        )
        complete_final = tuple(
            candidate
            for candidate in candidate_records[-1]
            if candidate.frame.length_px
            >= mature_length - config.trace.step_px
        )
        candidate_records[-1] = complete_final
        candidate_sets[-1] = tuple(
            candidate.hypothesis for candidate in complete_final
        )

    worldsheet = select_global_ribbon_worldsheet(
        candidate_sets,
        GlobalRibbonConfig(
            maximum_gap_samples=config.worldsheet_maximum_gap_samples,
        ),
    )
    completed = []
    for sample, candidate_index in enumerate(worldsheet.candidate_indices):
        if candidate_index is None:
            completed.append(_absent_frame(root_relative))
            continue
        completed.append(candidate_records[sample][candidate_index].frame)

    if worldsheet.first_active_sample is not None:
        bridge_short_tracking_gaps(
            completed,
            maximum_gap=config.worldsheet_maximum_gap_samples,
        )
        confirm_persistent_growth(completed, persistence_samples=2)
    return (
        TemporalRibbonHistory(
            frames=tuple(completed),
            first_persistent_sample=worldsheet.first_active_sample,
            portal_rebased=portal_rebase.succeeded,
            portal_removed_prefix_px=portal_rebase.removed_prefix_px,
            portal_connector_length_px=portal_rebase.connector_length_px,
        ),
        worldsheet,
    )
