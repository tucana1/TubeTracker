"""Owner-conditioned pollen and tube tracking primitives.

The module separates two identities that earlier prototypes conflated: a pollen
grain is a persistent video object, while its tube is a growing object owned by
that grain.  Tube hypotheses may overlap in image space without becoming the
same object, and ambiguous crossing decisions are resolved over the complete
time window instead of greedily in one frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi
from typing import Iterable, Sequence

import cv2 as cv
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class PollenObservation:
    """One learned pollen-mask observation in one sampled video frame."""

    frame_index: int
    detection_id: int
    center_yx: tuple[float, float]
    area_px: float
    radius_px: float
    circularity: float
    solidity: float
    pollen_score: float
    appearance: tuple[float, ...] = ()
    semantic_support: float = 1.0
    evidence_sources: tuple[str, ...] = ("learned-mask",)


@dataclass(frozen=True)
class PollenTrack:
    """A globally linked pollen identity, including late-entering grains."""

    track_id: int
    observations: tuple[PollenObservation, ...]

    @property
    def semantic_observation_count(self) -> int:
        """Count frames in which a learned mask supports this identity."""
        return sum(item.semantic_support >= 0.5 for item in self.observations)

    @property
    def evidence_sources(self) -> tuple[str, ...]:
        """Return every detector family contributing to this track."""
        return tuple(
            sorted(
                {
                    source
                    for observation in self.observations
                    for source in observation.evidence_sources
                }
            )
        )


@dataclass(frozen=True)
class PollenLinkConfig:
    """Configuration for global pollen-instance path cover."""

    maximum_step_distance_px: float = 35.0
    maximum_radius_change_fraction: float = 0.55
    maximum_gap_steps: int = 2
    gap_penalty: float = 0.18
    radius_penalty: float = 0.45
    appearance_penalty: float = 0.25
    continuation_reward: float = 1.25


@dataclass(frozen=True)
class TubePathObservation:
    """One owner-specific whole-tube proposal in one sampled frame."""

    frame_index: int
    candidate_id: int
    owner_track_id: int
    owner_center_yx: tuple[float, float]
    points_yx: np.ndarray
    appearance_score: float
    causal_growth_score: float
    owner_attachment_score: float
    preexisting_fraction: float = 0.0
    foreign_owner_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate and freeze the candidate geometry."""
        points = np.asarray(self.points_yx, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("points_yx must have shape (n, 2), with n >= 2")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points_yx", points)

    @property
    def length_px(self) -> float:
        """Return polyline arclength in pixels."""
        return float(np.linalg.norm(np.diff(self.points_yx, axis=0), axis=1).sum())


@dataclass(frozen=True)
class OwnerMemoryConfig:
    """Scoring and acceptance policy for owner-conditioned temporal decoding."""

    beam_width: int = 32
    maximum_prefix_p90_error_px: float = 5.0
    maximum_retraction_px: float = 3.0
    maximum_growth_px_per_frame: float = 14.0
    prefix_error_penalty: float = 0.35
    retraction_penalty: float = 0.8
    appearance_weight: float = 1.0
    causal_growth_weight: float = 1.5
    attachment_weight: float = 1.4
    preexisting_penalty: float = 2.0
    foreign_owner_penalty: float = 3.0
    lineage_birth_penalty: float = 0.5
    minimum_attachment_score: float = 0.3
    minimum_observations: int = 3
    minimum_score_margin: float = 0.5


@dataclass(frozen=True)
class OwnerLineage:
    """One complete time-consistent tube interpretation for an owner."""

    observations: tuple[TubePathObservation, ...]
    score: float
    prefix_error_p90_px: float


@dataclass(frozen=True)
class OwnerResolution:
    """Best owner lineage, retained alternatives, and promotion decision."""

    selected: OwnerLineage | None
    alternatives: tuple[OwnerLineage, ...]
    score_margin: float
    accepted: bool
    reason: str


def pollen_observations_from_labels(
    labels: np.ndarray,
    frame_index: int,
    gray: np.ndarray | None = None,
) -> tuple[PollenObservation, ...]:
    """Describe every nonzero instance label without discarding weak candidates."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional instance map")
    if gray is not None and np.asarray(gray).shape != labels.shape:
        raise ValueError("gray must match labels")

    observations = []
    for detection_id in np.unique(labels):
        if detection_id == 0:
            continue
        region = (labels == detection_id).astype(np.uint8)
        area = float(region.sum())
        if area <= 0:
            continue
        contours, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)
        perimeter = float(cv.arcLength(contour, True))
        hull = cv.convexHull(contour)
        hull_area = max(float(cv.contourArea(hull)), 1.0)
        ys, xs = np.nonzero(region)
        circularity = 4.0 * pi * area / max(perimeter * perimeter, 1.0)
        solidity = min(1.0, area / hull_area)
        shape_score = float(np.clip(0.55 * circularity + 0.45 * solidity, 0.0, 1.0))
        appearance = _appearance_histogram(gray, region) if gray is not None else ()
        observations.append(
            PollenObservation(
                frame_index=int(frame_index),
                detection_id=int(detection_id),
                center_yx=(float(ys.mean()), float(xs.mean())),
                area_px=area,
                radius_px=float(np.sqrt(area / pi)),
                circularity=float(circularity),
                solidity=float(solidity),
                pollen_score=shape_score,
                appearance=appearance,
            )
        )
    return tuple(observations)


def link_pollen_observations(
    observations_by_frame: Sequence[Sequence[PollenObservation]],
    config: PollenLinkConfig = PollenLinkConfig(),
) -> tuple[PollenTrack, ...]:
    """Link pollen masks with one global minimum-cost path-cover optimization."""
    observations = [item for frame in observations_by_frame for item in frame]
    if not observations:
        return ()
    observations.sort(key=lambda item: (item.frame_index, item.detection_id))
    frame_values = sorted({item.frame_index for item in observations})
    frame_position = {value: position for position, value in enumerate(frame_values)}

    edges: list[tuple[int, int, float]] = []
    for source_index, source in enumerate(observations):
        for target_index in range(source_index + 1, len(observations)):
            target = observations[target_index]
            gap = frame_position[target.frame_index] - frame_position[source.frame_index]
            if gap <= 0 or gap > config.maximum_gap_steps:
                continue
            cost = _pollen_link_cost(source, target, gap, config)
            if cost is not None and cost < config.continuation_reward:
                edges.append((source_index, target_index, cost - config.continuation_reward))

    selected_edges: list[tuple[int, int]] = []
    if edges:
        constraint = np.zeros((2 * len(observations), len(edges)), dtype=np.float64)
        for edge_index, (source_index, target_index, _) in enumerate(edges):
            constraint[source_index, edge_index] = 1.0
            constraint[len(observations) + target_index, edge_index] = 1.0
        result = milp(
            c=np.asarray([edge[2] for edge in edges]),
            integrality=np.ones(len(edges)),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(constraint, 0.0, 1.0),
            options={"disp": False},
        )
        if result.success and result.x is not None:
            selected_edges = [
                (edges[index][0], edges[index][1])
                for index, value in enumerate(result.x)
                if value > 0.5
            ]

    successor = {source: target for source, target in selected_edges}
    predecessor = {target: source for source, target in selected_edges}
    starts = [index for index in range(len(observations)) if index not in predecessor]
    tracks = []
    for track_id, start in enumerate(starts, start=1):
        indices = [start]
        while indices[-1] in successor:
            indices.append(successor[indices[-1]])
        tracks.append(PollenTrack(track_id, tuple(observations[index] for index in indices)))
    return tuple(tracks)


def fuse_pollen_observations(
    learned: Sequence[PollenObservation],
    geometric: Sequence[PollenObservation],
    maximum_center_distance_px: float = 18.0,
) -> tuple[PollenObservation, ...]:
    """Fuse learned masks with recall-oriented geometry while retaining both."""
    learned = tuple(learned)
    geometric = tuple(geometric)
    if learned and geometric:
        frame_indices = {item.frame_index for item in learned + geometric}
        if len(frame_indices) != 1:
            raise ValueError("all observations must come from one frame")

    possible = []
    for learned_index, learned_item in enumerate(learned):
        for geometric_index, geometric_item in enumerate(geometric):
            distance = hypot(
                learned_item.center_yx[0] - geometric_item.center_yx[0],
                learned_item.center_yx[1] - geometric_item.center_yx[1],
            )
            adaptive_limit = max(
                maximum_center_distance_px,
                0.75 * (learned_item.radius_px + geometric_item.radius_px),
            )
            if distance <= adaptive_limit:
                possible.append((distance, learned_index, geometric_index))

    matches = {}
    used_geometric = set()
    for _, learned_index, geometric_index in sorted(possible):
        if learned_index in matches or geometric_index in used_geometric:
            continue
        matches[learned_index] = geometric_index
        used_geometric.add(geometric_index)

    fused = []
    for learned_index, learned_item in enumerate(learned):
        geometric_index = matches.get(learned_index)
        if geometric_index is None:
            fused.append(learned_item)
            continue
        geometric_item = geometric[geometric_index]
        fused.append(
            PollenObservation(
                frame_index=learned_item.frame_index,
                detection_id=learned_item.detection_id,
                center_yx=(
                    0.75 * learned_item.center_yx[0] + 0.25 * geometric_item.center_yx[0],
                    0.75 * learned_item.center_yx[1] + 0.25 * geometric_item.center_yx[1],
                ),
                area_px=learned_item.area_px,
                radius_px=0.75 * learned_item.radius_px + 0.25 * geometric_item.radius_px,
                circularity=learned_item.circularity,
                solidity=learned_item.solidity,
                pollen_score=max(learned_item.pollen_score, geometric_item.pollen_score),
                appearance=learned_item.appearance,
                semantic_support=max(learned_item.semantic_support, geometric_item.semantic_support),
                evidence_sources=tuple(sorted(set(learned_item.evidence_sources + geometric_item.evidence_sources))),
            )
        )
    fused.extend(
        item for index, item in enumerate(geometric) if index not in used_geometric
    )
    fused.sort(key=lambda item: (item.center_yx[0], item.center_yx[1], item.detection_id))
    return tuple(fused)


def resolve_owner_hypotheses(
    candidates_by_frame: Sequence[Sequence[TubePathObservation]],
    config: OwnerMemoryConfig = OwnerMemoryConfig(),
) -> OwnerResolution:
    """Resolve crossing ambiguity with a whole-window, multi-hypothesis beam."""
    frames = [tuple(frame) for frame in candidates_by_frame if frame]
    if not frames:
        return OwnerResolution(None, (), 0.0, False, "no-candidates")
    owners = {candidate.owner_track_id for frame in frames for candidate in frame}
    if len(owners) != 1:
        raise ValueError("resolve each pollen owner independently")

    beam: list[OwnerLineage] = []
    for frame in frames:
        expanded = [
            OwnerLineage(
                (candidate,),
                _unary_score(candidate, config) - config.lineage_birth_penalty,
                0.0,
            )
            for candidate in frame
        ]
        for lineage in beam:
            previous = lineage.observations[-1]
            for candidate in frame:
                transition = _transition_score(previous, candidate, config)
                if transition is None:
                    continue
                transition_score, prefix_p90 = transition
                expanded.append(
                    OwnerLineage(
                        lineage.observations + (candidate,),
                        lineage.score + _unary_score(candidate, config) + transition_score,
                        max(lineage.prefix_error_p90_px, prefix_p90),
                    )
                )
        expanded.sort(key=lambda item: item.score, reverse=True)
        beam = _distinct_lineages(expanded, config.beam_width)

    beam.sort(key=lambda item: item.score, reverse=True)
    distinct = _geometrically_distinct_lineages(beam)
    selected = distinct[0]
    alternatives = tuple(distinct[1:])
    margin = selected.score - alternatives[0].score if alternatives else float("inf")
    if len(selected.observations) < config.minimum_observations:
        return OwnerResolution(selected, alternatives, margin, False, "too-few-observations")
    if margin < config.minimum_score_margin:
        return OwnerResolution(selected, alternatives, margin, False, "ambiguous-owner-lineage")
    return OwnerResolution(selected, alternatives, margin, True, "accepted")


def _appearance_histogram(gray: np.ndarray, region: np.ndarray) -> tuple[float, ...]:
    """Return a normalized intensity histogram for pollen association."""
    values = np.asarray(gray, dtype=np.uint8)[region.astype(bool)]
    histogram, _ = np.histogram(values, bins=8, range=(0, 256))
    histogram = histogram.astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    return tuple(float(value) for value in histogram)


def _pollen_link_cost(
    source: PollenObservation,
    target: PollenObservation,
    gap: int,
    config: PollenLinkConfig,
) -> float | None:
    """Score one possible temporal pollen-instance link."""
    distance = hypot(
        source.center_yx[0] - target.center_yx[0],
        source.center_yx[1] - target.center_yx[1],
    )
    distance_limit = config.maximum_step_distance_px * gap
    radius_change = abs(source.radius_px - target.radius_px) / max(source.radius_px, target.radius_px, 1.0)
    if distance > distance_limit or radius_change > config.maximum_radius_change_fraction:
        return None
    appearance = 0.0
    if source.appearance and target.appearance:
        appearance = float(np.abs(np.asarray(source.appearance) - np.asarray(target.appearance)).sum() / 2.0)
    return (
        distance / distance_limit
        + config.radius_penalty * radius_change
        + config.appearance_penalty * appearance
        + config.gap_penalty * (gap - 1)
    )


def _unary_score(candidate: TubePathObservation, config: OwnerMemoryConfig) -> float:
    """Score one owner-specific tube proposal without temporal context."""
    if candidate.owner_attachment_score < config.minimum_attachment_score:
        return -1_000_000.0
    return (
        config.appearance_weight * candidate.appearance_score
        + config.causal_growth_weight * candidate.causal_growth_score
        + config.attachment_weight * candidate.owner_attachment_score
        - config.preexisting_penalty * candidate.preexisting_fraction
        - config.foreign_owner_penalty * candidate.foreign_owner_score
    )


def _transition_score(
    previous: TubePathObservation,
    candidate: TubePathObservation,
    config: OwnerMemoryConfig,
) -> tuple[float, float] | None:
    """Score persistence of owned material after compensating pollen motion."""
    frame_gap = max(candidate.frame_index - previous.frame_index, 1)
    growth = candidate.length_px - previous.length_px
    if growth < -config.maximum_retraction_px:
        return None
    if growth > config.maximum_growth_px_per_frame * frame_gap:
        return None
    median_error, p90_error = _owner_relative_prefix_error(previous, candidate)
    if p90_error > config.maximum_prefix_p90_error_px:
        return None
    score = -config.prefix_error_penalty * median_error
    if growth < 0.0:
        score -= config.retraction_penalty * abs(growth)
    return score, p90_error


def _owner_relative_prefix_error(
    previous: TubePathObservation,
    candidate: TubePathObservation,
) -> tuple[float, float]:
    """Compare the already-grown prefix in each pollen owner's coordinate frame."""
    shared_length = min(previous.length_px, candidate.length_px)
    sample_count = max(8, int(np.ceil(shared_length)) + 1)
    distances = np.linspace(0.0, shared_length, sample_count)
    previous_points = _sample_polyline(previous.points_yx, distances)
    candidate_points = _sample_polyline(candidate.points_yx, distances)
    previous_points -= np.asarray(previous.owner_center_yx)
    candidate_points -= np.asarray(candidate.owner_center_yx)
    errors = np.linalg.norm(previous_points - candidate_points, axis=1)
    return float(np.median(errors)), float(np.percentile(errors, 90))


def _sample_polyline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Interpolate a polyline at requested arclength positions."""
    arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return np.column_stack(
        (
            np.interp(distances, arclength, points[:, 0]),
            np.interp(distances, arclength, points[:, 1]),
        )
    )


def _distinct_lineages(lineages: Iterable[OwnerLineage], limit: int) -> list[OwnerLineage]:
    """Retain competing recent ancestries instead of duplicate beam states."""
    selected = []
    signatures = set()
    for lineage in lineages:
        signature = tuple(item.candidate_id for item in lineage.observations[-3:])
        if signature in signatures:
            continue
        selected.append(lineage)
        signatures.add(signature)
        if len(selected) >= limit:
            break
    return selected


def _geometrically_distinct_lineages(
    lineages: Sequence[OwnerLineage],
) -> list[OwnerLineage]:
    """Collapse seed-level duplicates while preserving different branch choices."""
    distinct = []
    for lineage in lineages:
        if any(_lineages_share_terminal_topology(lineage, other) for other in distinct):
            continue
        distinct.append(lineage)
    return distinct


def _lineages_share_terminal_topology(
    first: OwnerLineage,
    second: OwnerLineage,
) -> bool:
    """Return whether two terminal paths are the same geometric interpretation."""
    first_path = first.observations[-1]
    second_path = second.observations[-1]
    mean_length = max(0.5 * (first_path.length_px + second_path.length_px), 1.0)
    if abs(first_path.length_px - second_path.length_px) / mean_length > 0.12:
        return False
    samples = np.linspace(0.0, min(first_path.length_px, second_path.length_px), 64)
    first_points = _sample_polyline(first_path.points_yx, samples) - np.asarray(first_path.owner_center_yx)
    second_points = _sample_polyline(second_path.points_yx, samples) - np.asarray(second_path.owner_center_yx)
    errors = np.linalg.norm(first_points - second_points, axis=1)
    return float(np.median(errors)) <= 3.0 and float(errors[-1]) <= 6.0
