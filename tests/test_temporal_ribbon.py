"""Test temporal whole-curve constraints used by the v26 prototype."""

from dataclasses import replace

import cv2 as cv
import numpy as np

from tubetracker.orientation_worldsheet import paired_wall_orientation_features
from tubetracker.temporal_ribbon import (
    TemporalRibbonConfig,
    TemporalRibbonFrame,
    TemporalRibbonHistory,
    bridge_short_tracking_gaps,
    confirm_persistent_growth,
    discover_mature_ribbon_candidates,
    endpoint_radial_efficiency,
    interpolate_sampled_history,
    intersects_foreign_owner,
    radial_excursion_efficiency,
    radial_excursion_px,
    rebase_pollen_portal,
    terminates_at_foreign_owner,
    truncate_antiparallel_return,
    truncate_at_foreign_owner,
)


def test_portal_rebase_replaces_a_grain_rim_walk_with_supported_exit():
    image = np.full((80, 80), 190, dtype=np.uint8)
    cv.line(image, (45, 38), (70, 38), 50, 2, cv.LINE_AA)
    cv.line(image, (45, 42), (70, 42), 50, 2, cv.LINE_AA)
    image = cv.GaussianBlur(image, (0, 0), 0.7)
    features = paired_wall_orientation_features(image)
    center = np.asarray((40.0, 40.0))
    angles = np.linspace(-0.5 * np.pi, 0.0, 10)
    rim = center + 6.0 * np.column_stack((np.sin(angles), np.cos(angles)))
    distal = np.column_stack((np.full(20, 40.0), np.arange(47.0, 67.0)))
    false_path = np.vstack((rim, distal))

    result = rebase_pollen_portal(
        false_path,
        features,
        center,
        pollen_radius_px=5.0,
    )

    assert result.succeeded
    assert result.removed_prefix_px > result.connector_length_px + 3.0
    assert np.isclose(np.linalg.norm(result.path_yx[0] - center), 5.75)
    assert np.allclose(result.path_yx[-1], false_path[-1])
    assert result.path_yx[0, 1] > center[1]


def test_portal_rebase_retains_a_short_curve_that_has_not_cleared_body_halo():
    image = np.full((40, 40), 190, dtype=np.uint8)
    features = paired_wall_orientation_features(image)
    center = np.asarray((20.0, 20.0))
    path = np.column_stack((np.full(5, 20.0), np.arange(26.0, 31.0)))

    result = rebase_pollen_portal(path, features, center, pollen_radius_px=5.0)

    assert not result.succeeded
    assert np.array_equal(result.path_yx, path)


def test_antiparallel_boundary_return_is_cut_at_distal_turn():
    outward = np.column_stack((np.zeros(21), np.arange(21, dtype=float)))
    turn = np.asarray(((1.0, 21.0), (2.0, 20.0)))
    returning = np.column_stack(
        (np.full(15, 3.0), np.arange(19.0, 4.0, -1.0))
    )
    boundary_walk = np.vstack((outward, turn, returning))

    result = truncate_antiparallel_return(
        boundary_walk,
        proximity_px=4.0,
        minimum_index_gap=8,
    )

    assert len(result) < len(boundary_walk)
    assert np.max(result[:, 1]) >= 20.0
    assert result[-1, 1] >= 19.0


def test_distal_path_into_another_pollen_is_rejected():
    path = np.column_stack((np.zeros(41), np.arange(41, dtype=float)))
    foreign = np.asarray(((0.0, 43.0), (50.0, 50.0)))

    assert terminates_at_foreign_owner(path, foreign, 5.0)
    assert not terminates_at_foreign_owner(path, foreign + (30.0, 0.0), 5.0)


def test_path_through_another_pollen_is_rejected_even_if_it_exits_again():
    path = np.column_stack((np.zeros(81), np.arange(81, dtype=float)))
    foreign = np.asarray(((0.0, 40.0),))

    assert not terminates_at_foreign_owner(path, foreign, 5.0)
    assert intersects_foreign_owner(path, foreign, 5.0)


def test_path_is_censored_at_first_foreign_pollen_without_losing_prefix():
    path = np.column_stack((np.zeros(81), np.arange(81, dtype=float)))
    foreign = np.asarray(((0.0, 40.0),))

    censored = truncate_at_foreign_owner(path, foreign, 5.0)

    assert 25.0 < censored[-1, 1] < 40.0
    assert np.array_equal(censored, path[: len(censored)])


def test_short_tracking_gap_interpolates_the_complete_curve():
    left_path = np.column_stack((np.zeros(6), np.arange(6, dtype=float)))
    right_path = np.column_stack((np.ones(8), np.arange(8, dtype=float)))
    accepted = lambda path: TemporalRibbonFrame(
        path_relative_yx=path,
        length_px=float(len(path) - 1),
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )
    missing = TemporalRibbonFrame(
        path_relative_yx=np.zeros((1, 2)),
        length_px=0.0,
        paired_fraction=0.0,
        mean_paired_support=0.0,
        prior_error_px=0.0,
        accepted=False,
    )
    frames = [accepted(left_path), missing, accepted(right_path)]

    bridge_short_tracking_gaps(frames, maximum_gap=1)

    assert frames[1].accepted
    assert frames[1].interpolated
    assert len(frames[1].path_relative_yx) == 8
    assert 5.0 < frames[1].length_px < 7.1


def test_sampled_history_expands_complete_curves_between_observed_anchors():
    def frame(length, *, accepted=True):
        path = np.column_stack((np.zeros(length + 1), np.arange(length + 1)))
        return TemporalRibbonFrame(
            path_relative_yx=path,
            length_px=float(length),
            paired_fraction=1.0 if accepted else 0.0,
            mean_paired_support=1.0 if accepted else 0.0,
            prior_error_px=0.0,
            accepted=accepted,
        )

    sparse = TemporalRibbonHistory(
        frames=(frame(0, accepted=False), frame(4), frame(10)),
        first_persistent_sample=1,
    )

    expanded = interpolate_sampled_history(
        sparse,
        np.asarray((0, 3, 6)),
        frame_count=7,
    )

    assert expanded.first_persistent_sample == 3
    assert not expanded.frames[2].accepted
    assert expanded.frames[3].length_px == 4.0
    assert expanded.frames[4].interpolated
    assert expanded.frames[5].interpolated
    assert expanded.frames[4].length_px == 6.0
    assert expanded.frames[5].length_px == 8.0
    for expected, result in ((6.0, expanded.frames[4]), (8.0, expanded.frames[5])):
        actual = np.linalg.norm(
            np.diff(result.path_relative_yx, axis=0), axis=1
        ).sum()
        assert np.isclose(actual, expected)
    assert expanded.frames[4].observed_length_px is None


def test_sampled_history_preserves_material_length_when_curves_bend_oppositely():
    left_path = np.asarray(((0.0, 0.0), (5.0, 5.0), (0.0, 10.0)))
    right_path = np.asarray(((0.0, 0.0), (-5.0, 5.0), (0.0, 10.0)))
    material_length = float(
        np.linalg.norm(np.diff(left_path, axis=0), axis=1).sum()
    )

    def frame(path):
        return TemporalRibbonFrame(
            path_relative_yx=path,
            length_px=material_length,
            paired_fraction=1.0,
            mean_paired_support=1.0,
            prior_error_px=0.0,
            accepted=True,
        )

    expanded = interpolate_sampled_history(
        TemporalRibbonHistory(
            frames=(frame(left_path), frame(right_path)),
            first_persistent_sample=0,
        ),
        np.asarray((0, 2)),
        frame_count=3,
    )

    middle = expanded.frames[1]
    actual = np.linalg.norm(
        np.diff(middle.path_relative_yx, axis=0), axis=1
    ).sum()
    assert middle.interpolated
    assert np.isclose(middle.length_px, material_length)
    assert np.isclose(actual, material_length)


def test_endpoint_radial_efficiency_rejects_a_pollen_rim_arc():
    outward_path = np.column_stack((np.zeros(11), np.arange(5.0, 16.0)))
    angles = np.linspace(0.0, np.pi, 17)
    rim_path = np.column_stack((5.0 * np.sin(angles), 5.0 * np.cos(angles)))

    def frame(path):
        return TemporalRibbonFrame(
            path_relative_yx=path,
            length_px=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            paired_fraction=1.0,
            mean_paired_support=1.0,
            prior_error_px=0.0,
            accepted=True,
        )

    assert endpoint_radial_efficiency(frame(outward_path)) > 0.9
    assert endpoint_radial_efficiency(frame(rim_path)) < 0.1


def test_radial_excursion_accepts_a_real_curve_that_returns_toward_its_owner():
    curved_path = np.asarray(
        ((0.0, 5.0), (0.0, 15.0), (10.0, 25.0), (20.0, 15.0), (10.0, 7.0))
    )
    frame = TemporalRibbonFrame(
        path_relative_yx=curved_path,
        length_px=float(
            np.linalg.norm(np.diff(curved_path, axis=0), axis=1).sum()
        ),
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )

    assert endpoint_radial_efficiency(frame) < 0.2
    assert radial_excursion_efficiency(frame) > 0.4
    assert radial_excursion_px(frame) > 20.0


def test_unconfirmed_final_extension_is_retained_but_not_reported():
    def frame(length):
        path = np.column_stack((np.zeros(length + 1), np.arange(length + 1)))
        return TemporalRibbonFrame(
            path_relative_yx=path,
            length_px=float(length),
            paired_fraction=1.0,
            mean_paired_support=1.0,
            prior_error_px=0.0,
            accepted=True,
        )

    frames = [frame(5), frame(7), frame(7), frame(12)]

    confirm_persistent_growth(frames, persistence_samples=2)

    assert [item.length_px for item in frames] == [5.0, 7.0, 7.0, 7.0]
    assert frames[-1].observed_length_px == 12.0
    assert not frames[-1].temporally_confirmed


def test_mature_discovery_searches_the_complete_pollen_rim():
    image = np.full((180, 180), 190, dtype=np.uint8)
    cv.line(image, (96, 88), (165, 88), 55, 2, cv.LINE_AA)
    cv.line(image, (96, 92), (165, 92), 55, 2, cv.LINE_AA)
    image = cv.GaussianBlur(image, (0, 0), 0.7)
    config = TemporalRibbonConfig(
        crop_size=180,
        mature_root_proposals=8,
        mature_alternatives_per_root=1,
        trace=replace(
            TemporalRibbonConfig().trace,
            beam_width=180,
            maximum_length_px=80.0,
        ),
    )

    candidates = discover_mature_ribbon_candidates(
        image,
        pollen_radius_px=5.0,
        feature_builder=paired_wall_orientation_features,
        config=config,
    )

    assert any(
        candidate.accepted and candidate.path_relative_yx[-1, 1] > 60.0
        for candidate in candidates
    )
