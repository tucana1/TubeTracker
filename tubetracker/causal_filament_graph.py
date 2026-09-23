"""Resolve projected tube crossings in a directed, causal path space.

The image skeleton is only an observation graph: two unrelated tubes can share
the same pixels where they cross.  This module therefore scores *directed*
root-to-endpoint paths.  A path carries its incoming orientation, construction
order, and agreement with an earlier rooted curve, so turning onto a strong
foreign tube is distinguishable from continuing through the same image point.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest

import numpy as np


@dataclass(frozen=True)
class CausalFilamentGraphConfig:
    """Configure lane continuity, causal order, and candidate exploration."""

    direction_window_px: float = 5.0
    maximum_junction_turn_degrees: float = 60.0
    turn_penalty: float = 1.25
    prior_prefix_distance_penalty: float = 0.35
    prior_extension_turn_penalty: float = 2.0
    birth_backtrack_tolerance: float = 2.0
    birth_backtrack_penalty: float = 1.5
    birth_progress_reward: float = 0.35
    evidence_reward: float = 0.35
    length_reward_per_px: float = 0.004
    maximum_candidates: int = 256
    maximum_search_states: int = 100_000
    maximum_path_points: int = 2_000


@dataclass(frozen=True)
class CausalPathCandidate:
    """Describe one pollen-rooted lane through the observed filament graph."""

    path_yx: np.ndarray
    arclength_px: np.ndarray
    score: float
    maximum_junction_turn_degrees: float
    prior_prefix_error_px: float
    distal_turn_degrees: float
    birth_backtrack_fraction: float
    birth_progress: float
    mean_evidence: float


@dataclass(frozen=True)
class CausalPathResult:
    """Return the selected lane and the evidence margin to its nearest rival."""

    selected: CausalPathCandidate
    alternatives: tuple[CausalPathCandidate, ...]
    score_margin: float
    search_truncated: bool


@dataclass(frozen=True)
class CausalWorldsheetConfig:
    """Configure global association of complete rooted curves across time."""

    maximum_growth_px_per_step: float = 8.0
    maximum_shrinkage_px: float = 2.0
    maximum_prefix_error_px: float = 4.0
    prefix_error_penalty: float = 1.5


@dataclass(frozen=True)
class CausalWorldsheetResult:
    """Store the globally selected curve at every analyzed time point."""

    selected: tuple[CausalPathCandidate, ...]
    total_score: float
    score_margin: float
    frame_candidate_indices: tuple[int, ...]


def _validate_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Return a boolean two-dimensional skeleton or raise a clear error."""

    value = np.asarray(skeleton, dtype=bool)
    if value.ndim != 2:
        raise ValueError("skeleton must be two-dimensional")
    if not np.any(value):
        raise ValueError("skeleton must contain at least one point")
    return value


def _pixel_neighbors(point: tuple[int, int], skeleton: np.ndarray):
    """Yield skeleton neighbors without diagonal corner-cutting triangles."""

    y, x = point
    height, width = skeleton.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]):
                continue
            if dy and dx:
                if skeleton[y, nx] or skeleton[ny, x]:
                    continue
            yield ny, nx


def _nearest_skeleton_point(
    skeleton: np.ndarray,
    root_yx: tuple[float, float] | np.ndarray,
) -> tuple[int, int]:
    """Snap a pollen-root location to its nearest observed filament point."""

    points = np.argwhere(skeleton)
    root = np.asarray(root_yx, dtype=np.float64)
    if root.shape != (2,):
        raise ValueError("root_yx must contain row and column")
    return tuple(points[np.argmin(np.sum((points - root) ** 2, axis=1))])


def _enumerate_rooted_paths(
    skeleton: np.ndarray,
    root_yx: tuple[int, int],
    config: CausalFilamentGraphConfig,
) -> tuple[list[np.ndarray], bool]:
    """Enumerate simple root-to-endpoint paths without collapsing junctions."""

    neighbor_cache = {
        tuple(point): tuple(_pixel_neighbors(tuple(point), skeleton))
        for point in np.argwhere(skeleton)
    }
    stack = [(root_yx, (root_yx,), frozenset((root_yx,)))]
    paths: list[np.ndarray] = []
    states = 0
    truncated = False
    while stack:
        current, path, visited = stack.pop()
        states += 1
        if states > config.maximum_search_states:
            truncated = True
            break
        available = [
            neighbor
            for neighbor in neighbor_cache[current]
            if neighbor not in visited
        ]
        if not available or len(path) >= config.maximum_path_points:
            if len(path) > 1:
                paths.append(np.asarray(path, dtype=np.float64))
            continue
        for neighbor in available:
            stack.append(
                (neighbor, path + (neighbor,), visited | frozenset((neighbor,)))
            )

    if len(paths) > config.maximum_candidates:
        paths = nlargest(
            config.maximum_candidates,
            paths,
            key=lambda path: _curve_arclength(path)[-1],
        )
        truncated = True
    return paths, truncated


def _curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative Euclidean arclength for one ordered curve."""

    path = np.asarray(path_yx, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )


def _point_at_arclength(
    path_yx: np.ndarray,
    arclength: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Interpolate row-column points at requested path arclengths."""

    query = np.clip(np.asarray(query, dtype=np.float64), 0.0, arclength[-1])
    return np.column_stack(
        [np.interp(query, arclength, path_yx[:, axis]) for axis in range(2)]
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector while preserving a stable zero fallback."""

    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector, dtype=np.float64) if norm <= 1e-9 else vector / norm


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angle between two vectors in degrees."""

    first_unit = _unit(first)
    second_unit = _unit(second)
    if not np.any(first_unit) or not np.any(second_unit):
        return 180.0
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _junction_turns(
    path_yx: np.ndarray,
    arclength: np.ndarray,
    skeleton: np.ndarray,
    window_px: float,
) -> list[float]:
    """Measure incoming-to-outgoing direction only at projection junctions."""

    turns = []
    for index in range(1, len(path_yx) - 1):
        point = tuple(np.rint(path_yx[index]).astype(int))
        if len(tuple(_pixel_neighbors(point, skeleton))) <= 2:
            continue
        before = int(
            np.searchsorted(arclength, arclength[index] - window_px, side="left")
        )
        after = int(
            np.searchsorted(arclength, arclength[index] + window_px, side="right")
            - 1
        )
        before = min(index - 1, max(0, before))
        after = max(index + 1, min(len(path_yx) - 1, after))
        turns.append(
            _angle_degrees(
                path_yx[index] - path_yx[before],
                path_yx[after] - path_yx[index],
            )
        )
    return turns


def _prior_metrics(
    path_yx: np.ndarray,
    arclength: np.ndarray,
    prior_yx: np.ndarray | None,
    window_px: float,
) -> tuple[float, float]:
    """Compare a candidate prefix and distal direction with an earlier curve."""

    if prior_yx is None:
        return 0.0, 0.0
    prior = np.asarray(prior_yx, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 2:
        raise ValueError("prior_yx must have shape (points, 2)")
    prior_arc = _curve_arclength(prior)
    shared = min(float(arclength[-1]), float(prior_arc[-1]))
    samples = np.linspace(0.0, shared, max(2, int(np.ceil(shared)) + 1))
    candidate_points = _point_at_arclength(path_yx, arclength, samples)
    prior_points = _point_at_arclength(prior, prior_arc, samples)
    prefix_error = float(np.mean(np.linalg.norm(candidate_points - prior_points, axis=1)))

    lookback = max(0.0, float(prior_arc[-1]) - window_px)
    prior_start = _point_at_arclength(prior, prior_arc, np.asarray([lookback]))[0]
    prior_direction = prior[-1] - prior_start
    candidate_start = _point_at_arclength(
        path_yx,
        arclength,
        np.asarray([min(float(arclength[-1]), shared)]),
    )[0]
    candidate_end = _point_at_arclength(
        path_yx,
        arclength,
        np.asarray([min(float(arclength[-1]), shared + window_px)]),
    )[0]
    distal_turn = _angle_degrees(prior_direction, candidate_end - candidate_start)
    return prefix_error, distal_turn


def _birth_metrics(
    path_yx: np.ndarray,
    birth_time: np.ndarray | None,
    tolerance: float,
) -> tuple[float, float]:
    """Measure violations and net progress of outward construction time."""

    if birth_time is None:
        return 0.0, 0.0
    birth = np.asarray(birth_time, dtype=np.float64)
    points = np.rint(path_yx).astype(int)
    values = birth[points[:, 0], points[:, 1]]
    finite = np.isfinite(values)
    values = values[finite]
    if len(values) < 2:
        return 0.0, 0.0
    running = np.maximum.accumulate(values)
    backtrack = values < running - tolerance
    edge = max(1, len(values) // 5)
    progress = float(np.median(values[-edge:]) - np.median(values[:edge]))
    return float(np.mean(backtrack)), progress


def score_causal_path(
    path_yx: np.ndarray,
    skeleton: np.ndarray,
    evidence: np.ndarray | None = None,
    birth_time: np.ndarray | None = None,
    prior_yx: np.ndarray | None = None,
    config: CausalFilamentGraphConfig | None = None,
) -> CausalPathCandidate:
    """Score one directed lane using geometry, ancestry, and temporal causality."""

    config = config or CausalFilamentGraphConfig()
    skeleton = _validate_skeleton(skeleton)
    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_yx must have shape (points, 2)")
    arclength = _curve_arclength(path)
    turns = _junction_turns(
        path,
        arclength,
        skeleton,
        config.direction_window_px,
    )
    maximum_turn = max(turns, default=0.0)
    prefix_error, distal_turn = _prior_metrics(
        path,
        arclength,
        prior_yx,
        config.direction_window_px,
    )
    backtrack, progress = _birth_metrics(
        path,
        birth_time,
        config.birth_backtrack_tolerance,
    )
    if evidence is None:
        mean_evidence = 1.0
    else:
        observed = np.asarray(evidence, dtype=np.float64)
        if observed.shape != skeleton.shape:
            raise ValueError("evidence must match skeleton")
        points = np.rint(path).astype(int)
        mean_evidence = float(np.mean(observed[points[:, 0], points[:, 1]]))

    normalized_turn = sum((turn / 90.0) ** 2 for turn in turns)
    score = (
        config.evidence_reward * mean_evidence
        + config.length_reward_per_px * float(arclength[-1])
        + config.birth_progress_reward * max(0.0, progress)
        - config.turn_penalty * normalized_turn
        - config.prior_prefix_distance_penalty * prefix_error
        - config.prior_extension_turn_penalty * (distal_turn / 90.0) ** 2
        - config.birth_backtrack_penalty * backtrack
    )
    if maximum_turn > config.maximum_junction_turn_degrees:
        score -= 1_000.0 + maximum_turn
    return CausalPathCandidate(
        path_yx=path,
        arclength_px=arclength,
        score=float(score),
        maximum_junction_turn_degrees=float(maximum_turn),
        prior_prefix_error_px=prefix_error,
        distal_turn_degrees=distal_turn,
        birth_backtrack_fraction=backtrack,
        birth_progress=progress,
        mean_evidence=mean_evidence,
    )


def trace_causal_filament(
    skeleton: np.ndarray,
    root_yx: tuple[float, float] | np.ndarray,
    evidence: np.ndarray | None = None,
    birth_time: np.ndarray | None = None,
    prior_yx: np.ndarray | None = None,
    config: CausalFilamentGraphConfig | None = None,
) -> CausalPathResult:
    """Select a pollen-rooted directed lane while preserving crossing identity."""

    config = config or CausalFilamentGraphConfig()
    skeleton = _validate_skeleton(skeleton)
    root = _nearest_skeleton_point(skeleton, root_yx)
    paths, truncated = _enumerate_rooted_paths(skeleton, root, config)
    if not paths:
        raise ValueError("no path leaves the snapped root")
    ranked = sorted(
        (
            score_causal_path(
                path,
                skeleton,
                evidence=evidence,
                birth_time=birth_time,
                prior_yx=prior_yx,
                config=config,
            )
            for path in paths
        ),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    margin = float("inf") if len(ranked) == 1 else ranked[0].score - ranked[1].score
    return CausalPathResult(
        selected=ranked[0],
        alternatives=tuple(ranked[1:]),
        score_margin=float(margin),
        search_truncated=truncated,
    )


def _root_relative_prefix_error(
    previous: CausalPathCandidate,
    current: CausalPathCandidate,
) -> float:
    """Compare shared material coordinates after removing pollen translation."""

    previous_path = previous.path_yx - previous.path_yx[0]
    current_path = current.path_yx - current.path_yx[0]
    shared = min(
        float(previous.arclength_px[-1]),
        float(current.arclength_px[-1]),
    )
    samples = np.linspace(0.0, shared, max(2, int(np.ceil(shared)) + 1))
    previous_points = _point_at_arclength(
        previous_path,
        previous.arclength_px,
        samples,
    )
    current_points = _point_at_arclength(
        current_path,
        current.arclength_px,
        samples,
    )
    return float(np.mean(np.linalg.norm(previous_points - current_points, axis=1)))


def select_causal_worldsheet(
    frame_candidates: list[tuple[CausalPathCandidate, ...]],
    config: CausalWorldsheetConfig | None = None,
) -> CausalWorldsheetResult:
    """Find the globally optimal pollen-rooted curve sequence with Viterbi DP.

    Candidate curves may share image pixels at crossings.  The transition model
    compares their ordered, root-relative material coordinates and permits only
    plausible growth, making a foreign-branch handoff expensive over the whole
    clip rather than merely suspicious in one frame.
    """

    config = config or CausalWorldsheetConfig()
    if not frame_candidates or any(not candidates for candidates in frame_candidates):
        raise ValueError("every frame must provide at least one path candidate")
    scores = [
        np.full(len(candidates), -np.inf, dtype=np.float64)
        for candidates in frame_candidates
    ]
    predecessors = [
        np.full(len(candidates), -1, dtype=np.int32)
        for candidates in frame_candidates
    ]
    scores[0] = np.asarray(
        [candidate.score for candidate in frame_candidates[0]],
        dtype=np.float64,
    )
    for frame_index in range(1, len(frame_candidates)):
        for current_index, current in enumerate(frame_candidates[frame_index]):
            current_length = float(current.arclength_px[-1])
            for previous_index, previous in enumerate(frame_candidates[frame_index - 1]):
                if not np.isfinite(scores[frame_index - 1][previous_index]):
                    continue
                previous_length = float(previous.arclength_px[-1])
                growth = current_length - previous_length
                if growth < -config.maximum_shrinkage_px:
                    continue
                if growth > config.maximum_growth_px_per_step:
                    continue
                prefix_error = _root_relative_prefix_error(previous, current)
                if prefix_error > config.maximum_prefix_error_px:
                    continue
                proposed = (
                    scores[frame_index - 1][previous_index]
                    + current.score
                    - config.prefix_error_penalty * prefix_error
                )
                if proposed > scores[frame_index][current_index]:
                    scores[frame_index][current_index] = proposed
                    predecessors[frame_index][current_index] = previous_index
        if not np.any(np.isfinite(scores[frame_index])):
            raise ValueError(
                f"no causally valid curve transition reaches frame {frame_index}"
            )
    final_order = np.argsort(scores[-1])[::-1]
    final = int(final_order[0])
    best_score = float(scores[-1][final])
    margin = (
        float("inf")
        if len(final_order) == 1 or not np.isfinite(scores[-1][final_order[1]])
        else best_score - float(scores[-1][final_order[1]])
    )
    indices = [final]
    for frame_index in range(len(frame_candidates) - 1, 0, -1):
        previous = int(predecessors[frame_index][indices[-1]])
        if previous < 0:
            raise RuntimeError("worldsheet backtracking reached an invalid state")
        indices.append(previous)
    indices.reverse()
    selected = tuple(
        frame_candidates[frame_index][candidate_index]
        for frame_index, candidate_index in enumerate(indices)
    )
    return CausalWorldsheetResult(
        selected=selected,
        total_score=best_score,
        score_margin=float(margin),
        frame_candidate_indices=tuple(indices),
    )
