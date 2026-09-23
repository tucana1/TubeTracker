"""Checks for pollen-body motion, pose recovery, and stabilization."""

import cv2 as cv
import numpy as np

import tubetracker.pollen_motion as pollen_motion
from tubetracker.grain_pose import (
    GrainPoseConfig,
    estimate_grain_poses,
    stabilization_matrix,
)
from tubetracker.pollen_motion import (
    DetectionSnapConfig,
    OwnerMotionFusionConfig,
    PollenMotionConfig,
    SeedCanonicalizationConfig,
    assign_unique_pollen_detections,
    anchor_trajectories_to_identity_observations,
    canonicalize_owner_trajectory_seeds,
    constrain_trajectories_to_unique_detections,
    fuse_redundant_owner_motion,
    fuse_trajectories_with_identity_span,
    pollen_body_query_points,
    pollen_body_radial_evidence,
    pollen_trajectory_from_tracks,
    track_pollen_bodies_in_crop_mosaics,
    track_pollen_bodies_from_seeds,
    track_query_points_bidirectional,
    semantic_anchor_guard_weights,
)


def test_radial_body_evidence_separates_closed_grain_from_thin_tube():
    """Opposite circumference support should reject a line through the center."""

    center = (40, 40)
    closed = np.full((81, 81), 180, dtype=np.uint8)
    line = closed.copy()
    cv.circle(closed, center[::-1], 12, 105, -1, cv.LINE_AA)
    cv.line(line, (0, center[0]), (80, center[0]), 105, 3, cv.LINE_AA)

    grain = pollen_body_radial_evidence(closed, center, 12)
    tube = pollen_body_radial_evidence(line, center, 12)

    assert grain.valid_angular_fraction == 1.0
    assert grain.median_radial_contrast > 10.0
    assert grain.opposite_boundary_fraction > 0.9
    assert tube.median_radial_contrast < 3.5
    assert tube.opposite_boundary_fraction < 0.35
    assert grain.opposite_boundary_fraction > tube.opposite_boundary_fraction + 0.6


def test_radial_body_evidence_does_not_use_reflected_padding_as_a_boundary():
    """Image-edge padding must not create false closed-body evidence."""

    image = np.full((61, 61), 180, dtype=np.uint8)
    cv.circle(image, (2, 30), 10, 100, -1, cv.LINE_AA)

    evidence = pollen_body_radial_evidence(image, (30, 2), 10)

    assert evidence.valid_angular_fraction < 0.75
    assert 0.0 <= evidence.angular_boundary_fraction <= 1.0
    assert 0.0 <= evidence.opposite_boundary_fraction <= 1.0


def test_radial_body_evidence_rejects_invalid_inputs():
    """Malformed geometry and thresholds should fail before image sampling."""

    image = np.zeros((20, 20), dtype=np.uint8)
    invalid_calls = (
        lambda: pollen_body_radial_evidence(image[0], (5, 5), 4),
        lambda: pollen_body_radial_evidence(image, (5,), 4),
        lambda: pollen_body_radial_evidence(image, (5, 5), 0),
        lambda: pollen_body_radial_evidence(image, (5, 5), 4, angular_samples=9),
        lambda: pollen_body_radial_evidence(image, (5, 5), 4, radial_samples=7),
        lambda: pollen_body_radial_evidence(
            image, (5, 5), 4, boundary_gradient_threshold=float("nan")
        ),
    )
    for call in invalid_calls:
        with np.testing.assert_raises(ValueError):
            call()


def test_persistent_body_canonicalization_beats_nearer_rounded_tip():
    """A complete high-quality pollen track should beat a nearby tube endpoint."""

    priors = np.repeat(np.asarray([[[90.0, 100.0]]]), 5, axis=1)
    detections = []
    for sample in range(5):
        rows = [[70.0, 90.0, 19.0, 0.90]]
        if sample > 0:
            rows.append([105.0, 105.0, 11.0, 0.68])
        detections.append(np.asarray(rows))

    result = canonicalize_owner_trajectory_seeds(
        priors,
        detections,
        owner_radius_px=15.0,
    )

    assert result.selected_track_ids[0] >= 0
    assert result.selected_support_fractions[0] == 1.0
    assert np.allclose(result.corrections_yx[0], [-20.0, -10.0])
    assert np.allclose(result.trajectories_yx[0, 0], [70.0, 90.0])


def test_ambiguous_seed_canonicalization_preserves_the_prior():
    """Two equally plausible bodies should not trigger an arbitrary reseed."""

    priors = np.repeat(np.asarray([[[50.0, 50.0]]]), 4, axis=1)
    detections = [
        np.asarray(
            [
                [50.0, 45.0, 10.0, 0.9],
                [50.0, 55.0, 10.0, 0.9],
            ]
        )
        for _ in range(4)
    ]

    result = canonicalize_owner_trajectory_seeds(
        priors,
        detections,
        owner_radius_px=10.0,
        config=SeedCanonicalizationConfig(initialization_samples=4),
    )

    assert result.selected_track_ids.tolist() == [-1]
    assert np.allclose(result.trajectories_yx, priors)


def test_joint_detection_assignment_prevents_two_owners_using_one_circle():
    priors = np.asarray([[10.0, 10.0], [10.0, 10.5]])
    detections = np.asarray(
        [[10.0, 9.0, 4.0, 0.9], [10.0, 14.0, 4.0, 0.8]]
    )

    assignments, costs = assign_unique_pollen_detections(
        priors,
        detections,
        owner_radius_px=4.0,
    )

    assert sorted(assignments.tolist()) == [0, 1]
    assert np.all(costs < DetectionSnapConfig().unmatched_cost)


def test_owner_motion_fusion_keeps_trusted_rigid_identity_over_far_detection():
    multipoint = np.asarray([[[10.0, 10.0], [10.0, 11.0]]])
    template = multipoint + 0.25
    detection = np.asarray([[[10.0, 10.0], [30.0, 30.0]]])

    result = fuse_redundant_owner_motion(
        multipoint,
        template,
        detection,
        observed=np.ones((1, 2), dtype=bool),
        inlier_counts=np.full((1, 2), 10),
        template_scores=np.full((1, 2), 0.98),
    )

    assert np.allclose(result.centers_yx, multipoint)
    assert result.source_codes.tolist() == [[0, 0]]


def test_owner_motion_fusion_falls_back_from_weak_landmarks_in_priority_order():
    multipoint = np.zeros((1, 2, 2), dtype=float)
    template = np.full((1, 2, 2), 5.0)
    detection = np.full((1, 2, 2), 9.0)

    result = fuse_redundant_owner_motion(
        multipoint,
        template,
        detection,
        observed=np.asarray([[False, False]]),
        inlier_counts=np.zeros((1, 2), dtype=int),
        template_scores=np.asarray([[0.95, 0.20]]),
        config=OwnerMotionFusionConfig(minimum_inlier_count=2),
    )

    assert result.centers_yx[0, 0].tolist() == [5.0, 5.0]
    assert result.centers_yx[0, 1].tolist() == [9.0, 9.0]
    assert result.source_codes.tolist() == [[1, 2]]


def test_joint_detection_assignment_preserves_prior_when_circle_is_too_far():
    assignments, costs = assign_unique_pollen_detections(
        np.asarray([[10.0, 10.0]]),
        np.asarray([[40.0, 40.0, 4.0, 1.0]]),
        owner_radius_px=4.0,
    )

    assert assignments.tolist() == [-1]
    assert costs.tolist() == [DetectionSnapConfig().unmatched_cost]


def test_dense_unique_detections_separate_collapsed_owner_priors():
    priors = np.zeros((2, 7, 2), dtype=float)
    priors[0, :, 1] = 10.0
    priors[1, :, 1] = 10.5
    detections = [
        np.asarray([[0.0, 8.0, 4.0, 1.0], [0.0, 14.0, 4.0, 1.0]])
        for _ in range(7)
    ]

    constrained, assignments, _, _ = constrain_trajectories_to_unique_detections(
        priors,
        detections,
        owner_radius_px=4.0,
        config=DetectionSnapConfig(smoothing_window=3),
    )

    assert np.all(assignments >= 0)
    assert np.all(assignments[0] != assignments[1])
    assert np.allclose(
        np.sort(constrained[:, :, 1], axis=0), [[8.0] * 7, [14.0] * 7]
    )


def test_sparse_detection_correction_does_not_spread_across_unsupported_movie():
    priors = np.zeros((1, 21, 2), dtype=float)
    detections = [np.empty((0, 4)) for _ in range(21)]
    for sample in range(3):
        detections[sample] = np.asarray([[0.0, 5.0, 4.0, 1.0]])

    constrained, assignments, _, _ = constrain_trajectories_to_unique_detections(
        priors,
        detections,
        owner_radius_px=4.0,
        config=DetectionSnapConfig(
            smoothing_window=3,
            maximum_unsupported_windows=1.0,
        ),
    )

    assert np.all(assignments[0, :3] == 0)
    assert constrained[0, 1, 1] > 4.0
    assert constrained[0, -1, 1] == 0.0


def test_semantic_anchor_guards_are_local_hard_constraints():
    frames = np.arange(11, dtype=float) * 10.0
    weights = semantic_anchor_guard_weights(
        frames,
        np.asarray([7]),
        [{"track_id": 7, "source_frames": [50]}],
        guard_samples=1.0,
        recovery_samples=3.0,
    )

    assert np.all(weights[0, 4:7] == 0.0)
    assert weights[0, 3] == 0.5
    assert weights[0, 0] == 1.0


def test_dense_owner_motion_is_smoothly_anchored_to_all_identity_observations():
    frames = np.asarray([0, 10, 20, 30], dtype=np.int64)
    tracks = np.asarray(
        [[[0.0, 0.0], [11.0, 0.0], [22.0, 0.0], [33.0, 0.0]]]
    )
    identity = [
        {
            "track_id": 7,
            "source_frames": [0, 20, 30],
            "centers_yx": [[0.0, 0.0], [20.0, 2.0], [30.0, 3.0]],
        }
    ]

    anchored, corrections, counts = anchor_trajectories_to_identity_observations(
        frames,
        np.asarray([7]),
        tracks,
        identity,
    )

    assert counts.tolist() == [3]
    assert np.allclose(anchored[0, [0, 2, 3]], identity[0]["centers_yx"])
    assert np.allclose(corrections[0, 1], (-1.0, 1.0))
    assert np.allclose(anchored[0, 1], (10.0, 1.0))


def test_identity_span_fusion_uses_template_only_outside_semantic_support():
    frames = np.asarray([0, 10, 20, 30, 40])
    multipoint = np.zeros((1, 5, 2), dtype=float)
    template = np.full((1, 5, 2), 9.0)
    identity = [
        {
            "track_id": 4,
            "source_frames": [10, 30],
            "centers_yx": [[1.0, 1.0], [3.0, 3.0]],
        }
    ]

    fused, template_used = fuse_trajectories_with_identity_span(
        frames,
        np.asarray([4]),
        multipoint,
        template,
        identity,
    )

    assert template_used.tolist() == [[True, False, False, False, True]]
    assert np.all(fused[0, 1:4] == 0.0)
    assert np.all(fused[0, [0, 4]] == 9.0)


def _config():
    """Return a model-free configuration for geometry tests."""
    return PollenMotionConfig(
        checkpoint=None,
        ring_point_count=8,
        pose=GrainPoseConfig(
            center_smoothing_window=1,
            angle_smoothing_window=1,
        ),
    )


def test_pollen_body_queries_cover_center_and_interior_ring():
    """Body consensus should use a center plus evenly spaced interior points."""
    config = _config()
    center = np.asarray([40.0, 30.0])
    points = pollen_body_query_points(center, pollen_radius=10.0, config=config)
    assert points.shape == (9, 2)
    assert np.allclose(points[0], center)
    assert np.allclose(
        np.linalg.norm(points[1:] - center, axis=1),
        10.0 * config.ring_radius_fraction,
    )


def test_cached_body_points_recover_global_pollen_motion():
    """Local point tracks should become a robust global pollen trajectory."""
    config = _config()
    local_center = np.asarray([128.0, 128.0])
    initial = pollen_body_query_points(
        local_center, pollen_radius=10.0, config=config
    )
    angles = np.linspace(0.0, 0.25, 6)
    shifts = np.column_stack((np.linspace(0, 15, 6), np.linspace(0, -8, 6)))
    tracks = []
    for angle, shift in zip(angles, shifts):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        tracks.append((initial - local_center) @ rotation.T + local_center + shift)
    trajectory = pollen_trajectory_from_tracks(
        np.stack(tracks),
        np.ones((6, len(initial)), dtype=bool),
        initial,
        initial_center_xy=np.asarray([438.5, 164.5]),
        config=config,
    )
    expected = np.asarray([438.5, 164.5]) + shifts
    assert np.allclose(trajectory.centers_xy, expected, atol=0.25)
    assert np.allclose(trajectory.angles_radians, angles, atol=0.03)
    mapped = np.asarray(
        [
            transform[:, :2] @ np.asarray([438.5, 164.5]) + transform[:, 2]
            for transform in trajectory.transforms
        ]
    )
    assert np.allclose(mapped, trajectory.centers_xy)


def test_bidirectional_queries_use_backward_tracks_before_each_seed(monkeypatch):
    """Late keyframes should carry identity evidence into preceding frames."""
    calls = []

    def fake_track(grays, queries, config=None):
        calls.append(np.asarray(queries).copy())
        value = float(len(calls))
        tracks = np.full((len(grays), len(queries), 2), value)
        return tracks, np.ones(tracks.shape[:2], dtype=bool)

    monkeypatch.setattr(pollen_motion, "track_query_points", fake_track)
    grays = np.zeros((6, 8, 8), dtype=np.uint8)
    queries = np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32)
    tracks, visibility = track_query_points_bidirectional(
        grays, queries, config=_config()
    )
    assert len(calls) == 2
    assert calls[1][0, 0] == 2.0
    assert np.all(tracks[:3] == 2.0)
    assert np.all(tracks[3:] == 1.0)
    assert np.all(visibility)


def test_multiple_pollen_bodies_share_one_model_pass(monkeypatch):
    """Owner landmarks should be grouped after one full-field point-tracking call."""

    calls = []

    def fake_track(grays, queries, config=None):
        calls.append(np.asarray(queries).copy())
        tracks = np.repeat(queries[None, :, 1:], len(grays), axis=0)
        tracks[:, :9] += np.asarray([2.0, -1.0])
        tracks[:, 9:] += np.asarray([-3.0, 4.0])
        return tracks, np.ones(tracks.shape[:2], dtype=bool)

    monkeypatch.setattr(pollen_motion, "track_query_points", fake_track)
    grays = np.zeros((5, 80, 100), dtype=np.uint8)
    centers = np.asarray([[30.0, 25.0], [70.0, 55.0]])
    trajectories = track_pollen_bodies_from_seeds(
        grays,
        centers,
        pollen_radii=[8.0, 10.0],
        config=_config(),
    )
    assert len(calls) == 1
    assert calls[0].shape == (18, 3)
    assert len(trajectories) == 2
    assert np.allclose(trajectories[0].centers_xy, centers[0] + [2.0, -1.0])
    assert np.allclose(trajectories[1].centers_xy, centers[1] + [-3.0, 4.0])


def test_crop_mosaic_recovers_global_owner_motion(monkeypatch):
    """Mosaic-local landmarks should map back to each owner's source coordinates."""

    calls = []

    def fake_track(grays, queries, config=None):
        calls.append((np.asarray(grays).shape, np.asarray(queries).copy()))
        tracks = np.repeat(queries[None, :, 1:], len(grays), axis=0)
        tracks[:, :9] += np.asarray([3.0, -2.0])
        tracks[:, 9:] += np.asarray([-4.0, 5.0])
        return tracks, np.ones(tracks.shape[:2], dtype=bool)

    monkeypatch.setattr(pollen_motion, "track_query_points", fake_track)
    config = PollenMotionConfig(
        checkpoint=None,
        crop_size=32,
        ring_point_count=8,
        pose=GrainPoseConfig(
            center_smoothing_window=1,
            angle_smoothing_window=1,
        ),
    )
    grays = np.zeros((5, 64, 96), dtype=np.uint8)
    centers = np.asarray([[20.0, 20.0], [70.0, 42.0]])
    trajectories = track_pollen_bodies_in_crop_mosaics(
        grays,
        centers,
        pollen_radii=[6.0, 7.0],
        batch_size=2,
        gap=8,
        config=config,
    )

    assert len(calls) == 1
    assert calls[0][0] == (5, 48, 88)
    assert np.allclose(trajectories[0].centers_xy, centers[0] + [3.0, -2.0])
    assert np.allclose(trajectories[1].centers_xy, centers[1] + [-4.0, 5.0])


def _transform(points, angle, translation):
    """Apply one rigid transform to Cartesian points."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T + translation


def test_grain_pose_recovers_translation_and_rotation_through_occlusion():
    """Visible body points should preserve pose when several points disappear."""
    initial = np.asarray(
        [[20, 20], [14, 20], [26, 20], [20, 14], [20, 26], [16, 16]],
        dtype=float,
    )
    angles = np.linspace(0.0, 0.4, 9)
    translations = np.column_stack(
        (np.linspace(0.0, 12.0, 9), np.linspace(0.0, -5.0, 9))
    )
    tracks = np.stack(
        [
            _transform(initial - initial[0], angle, initial[0] + shift)
            for angle, shift in zip(angles, translations)
        ]
    )
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    visibility[4, 4:] = False
    result = estimate_grain_poses(
        initial,
        tracks,
        visibility,
        config=GrainPoseConfig(
            center_smoothing_window=1, angle_smoothing_window=1
        ),
    )
    assert np.allclose(result.centers, initial[0] + translations, atol=0.25)
    assert np.allclose(result.angles_radians, angles, atol=0.03)
    assert result.observed[4]


def test_stabilization_matrix_returns_pose_to_fixed_crop_center():
    """The inverse pose should map a moving grain center to the crop center."""
    initial_center = np.asarray([30.0, 40.0])
    current_center = np.asarray([55.0, 25.0])
    angle = 0.3
    moving = _transform(
        initial_center[None] - initial_center,
        angle,
        current_center,
    )[0]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    transform = np.column_stack(
        (rotation, current_center - rotation @ initial_center)
    )
    crop_center = np.asarray([100.0, 100.0])
    stabilized = cv.transform(
        moving.reshape(1, 1, 2),
        stabilization_matrix(transform, initial_center, crop_center),
    )[0, 0]
    assert np.allclose(stabilized, crop_center)


def test_snap_corrected_anchoring_measures_unsmoothed_assignment_gap():
    """Smoothing-corrected centers must report their raw assignment gap."""

    from prototypes.v29_causal_growth_front.bootstrap import (
        _snap_corrected_mature_anchoring,
    )

    # One owner, three samples: raw Hough hit sits 4 analysis px away while
    # the smoothed exported center sits at the origin.
    template = np.zeros((1, 3, 2), dtype=float)
    assignments = np.asarray([[0, 0, -1]], dtype=np.int64)
    detections = [
        np.asarray([[0.0, 4.0, 4.0, 1.0]]),
        np.asarray([[0.0, 4.0, 4.0, 1.0]]),
        np.empty((0, 4)),
    ]
    snap = _snap_corrected_mature_anchoring(
        template, assignments, detections, analysis_scale=0.375
    )
    assert snap.shape == (1, 3)
    assert abs(float(snap[0, 0]) - 4.0 / 0.375) < 1e-6
    assert abs(float(snap[0, 1]) - 4.0 / 0.375) < 1e-6
    assert bool(np.isnan(snap[0, 2]))
