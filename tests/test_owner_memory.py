"""Tests for learned pollen identities and delayed owner-path resolution."""

import cv2 as cv
import numpy as np

from tubetracker.owner_memory import (
    OwnerMemoryConfig,
    PollenLinkConfig,
    PollenObservation,
    TubePathObservation,
    fuse_pollen_observations,
    link_pollen_observations,
    pollen_observations_from_labels,
    resolve_owner_hypotheses,
)


def _pollen(frame, detection, y, x, radius=8.0):
    """Create a compact synthetic pollen observation."""
    return PollenObservation(frame, detection, (y, x), np.pi * radius**2, radius, 0.9, 0.95, 0.92)


def _path(frame, candidate, points, *, causal=0.8, appearance=0.8, foreign=0.0, preexisting=0.0, owner=1):
    """Create one synthetic owner-specific tube proposal."""
    return TubePathObservation(
        frame,
        candidate,
        owner,
        (50.0, 10.0),
        np.asarray(points, dtype=float),
        appearance,
        causal,
        0.95,
        preexisting,
        foreign,
    )


def test_mask_descriptors_retain_every_learned_instance():
    labels = np.zeros((80, 100), dtype=np.int32)
    cv.circle(labels, (20, 20), 8, 1, -1)
    cv.ellipse(labels, (70, 50), (12, 5), 20, 0, 360, 2, -1)

    observations = pollen_observations_from_labels(labels, 12)

    assert [item.detection_id for item in observations] == [1, 2]
    assert observations[0].pollen_score > observations[1].pollen_score


def test_global_linker_starts_tracks_after_the_first_frame():
    frames = [
        [_pollen(0, 1, 20, 20)],
        [_pollen(1, 1, 21, 22), _pollen(1, 2, 70, 70)],
        [_pollen(2, 1, 22, 24), _pollen(2, 2, 71, 72)],
    ]

    tracks = link_pollen_observations(frames, PollenLinkConfig(maximum_step_distance_px=10.0))

    assert sorted(len(track.observations) for track in tracks) == [2, 3]
    late_track = next(track for track in tracks if len(track.observations) == 2)
    assert late_track.observations[0].frame_index == 1


def test_detector_fusion_keeps_recall_but_records_semantic_identity_support():
    learned = [_pollen(0, 1, 20, 20)]
    geometric = [
        PollenObservation(0, 101, (21, 19), 190, 7.8, 0.9, 0.9, 0.8, (), 0.0, ("hough-circle",)),
        PollenObservation(0, 102, (60, 70), 180, 7.5, 0.9, 0.9, 0.7, (), 0.0, ("hough-circle",)),
    ]

    fused = fuse_pollen_observations(learned, geometric)

    assert len(fused) == 2
    supported = next(item for item in fused if item.semantic_support >= 0.5)
    unsupported = next(item for item in fused if item.semantic_support < 0.5)
    assert supported.evidence_sources == ("hough-circle", "learned-mask")
    assert unsupported.evidence_sources == ("hough-circle",)


def test_early_semantic_support_can_carry_an_owner_through_late_mask_loss():
    frames = [
        [_pollen(0, 1, 20, 20)],
        [_pollen(1, 1, 21, 20)],
        [PollenObservation(2, 101, (22, 20), 190, 7.8, 0.9, 0.9, 0.8, (), 0.0, ("hough-circle",))],
        [PollenObservation(3, 101, (23, 20), 190, 7.8, 0.9, 0.9, 0.8, (), 0.0, ("hough-circle",))],
    ]

    tracks = link_pollen_observations(frames, PollenLinkConfig(maximum_step_distance_px=8.0))

    assert len(tracks) == 1
    assert tracks[0].semantic_observation_count == 2
    assert tracks[0].evidence_sources == ("hough-circle", "learned-mask")


def test_owner_memory_uses_future_frames_to_reject_a_crossing_handoff():
    straight_1 = np.column_stack((np.full(31, 50.0), np.arange(10.0, 41.0)))
    straight_2 = np.column_stack((np.full(41, 50.0), np.arange(10.0, 51.0)))
    straight_3 = np.column_stack((np.full(51, 50.0), np.arange(10.0, 61.0)))
    wrong_2 = np.vstack((straight_1, np.column_stack((np.arange(49.0, 19.0, -1.0), np.full(30, 40.0)))))
    wrong_3 = np.vstack((straight_1, np.column_stack((np.arange(49.0, 9.0, -1.0), np.full(40, 40.0)))))
    frames = [
        [_path(0, 1, straight_1)],
        [
            _path(1, 2, straight_2, appearance=0.65, causal=0.9),
            _path(1, 3, wrong_2, appearance=1.0, causal=0.2, foreign=0.7, preexisting=0.8),
        ],
        [
            _path(2, 4, straight_3, appearance=0.7, causal=1.0),
            _path(2, 5, wrong_3, appearance=1.0, causal=0.1, foreign=0.8, preexisting=0.9),
        ],
    ]

    result = resolve_owner_hypotheses(frames, OwnerMemoryConfig(minimum_score_margin=0.1))

    assert result.accepted
    assert [item.candidate_id for item in result.selected.observations] == [1, 2, 4]


def test_crossing_paths_remain_unresolved_when_ownership_evidence_is_equal():
    base = np.column_stack((np.full(31, 50.0), np.arange(10.0, 41.0)))
    branch_up = np.vstack((base, np.column_stack((np.arange(49.0, 39.0, -1.0), np.full(10, 40.0)))))
    branch_down = np.vstack((base, np.column_stack((np.arange(51.0, 61.0), np.full(10, 40.0)))))
    frames = [
        [_path(0, 1, base)],
        [_path(1, 2, branch_up), _path(1, 3, branch_down)],
        [_path(2, 4, branch_up), _path(2, 5, branch_down)],
    ]

    result = resolve_owner_hypotheses(
        frames,
        OwnerMemoryConfig(
            minimum_score_margin=0.5,
            maximum_growth_px_per_frame=25.0,
        ),
    )

    assert not result.accepted
    assert result.reason == "ambiguous-owner-lineage"


def test_different_owners_can_use_overlapping_image_pixels():
    shared = np.column_stack((np.full(31, 50.0), np.arange(10.0, 41.0)))
    owner_one = [[_path(frame, frame + 1, shared, owner=1)] for frame in range(3)]
    owner_two = [[_path(frame, frame + 11, shared, owner=2)] for frame in range(3)]

    first = resolve_owner_hypotheses(owner_one)
    second = resolve_owner_hypotheses(owner_two)

    assert first.accepted and second.accepted
    assert np.array_equal(first.selected.observations[-1].points_yx, second.selected.observations[-1].points_yx)


def test_tube_lineage_can_begin_after_pollen_identity_exists():
    unstable = np.column_stack((np.arange(30.0, 61.0), np.full(31, 10.0)))
    stable_1 = np.column_stack((np.full(31, 50.0), np.arange(10.0, 41.0)))
    stable_2 = np.column_stack((np.full(41, 50.0), np.arange(10.0, 51.0)))
    stable_3 = np.column_stack((np.full(51, 50.0), np.arange(10.0, 61.0)))
    frames = [
        [_path(0, 1, unstable, causal=0.1)],
        [_path(1, 2, stable_1)],
        [_path(2, 3, stable_2)],
        [_path(3, 4, stable_3)],
    ]

    result = resolve_owner_hypotheses(frames, OwnerMemoryConfig(minimum_score_margin=0.1))

    assert result.accepted
    assert [item.candidate_id for item in result.selected.observations] == [2, 3, 4]


def test_duplicate_seed_paths_do_not_create_false_topology_ambiguity():
    base = np.column_stack((np.full(31, 50.0), np.arange(10.0, 41.0)))
    grown = np.column_stack((np.full(51, 50.0), np.arange(10.0, 61.0)))
    duplicate = grown + np.asarray((0.5, 0.0))
    frames = [
        [_path(0, 1, base)],
        [_path(1, 2, grown), _path(1, 3, duplicate)],
        [_path(2, 4, grown), _path(2, 5, duplicate)],
    ]

    result = resolve_owner_hypotheses(
        frames,
        OwnerMemoryConfig(
            minimum_score_margin=0.5,
            maximum_growth_px_per_frame=25.0,
        ),
    )

    assert result.accepted
    assert result.score_margin == float("inf")
