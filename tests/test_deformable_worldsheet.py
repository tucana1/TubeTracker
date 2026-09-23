"""Tests for globally coupled material-coordinate video surfaces."""

import math

import numpy as np

from tubetracker.deformable_worldsheet import (
    atlas_hypothesis_selection_key,
    atlas_prototype_score,
    DeformableWorldsheetConfig,
    fit_deformable_worldsheet,
    material_continuity_certificate,
    open_curve_certificate,
    orientation_blobness,
    worldsheet_promotion_decision,
)


def _straight_atlas(point_count=51):
    """Return a horizontal test atlas with one-pixel material spacing."""

    return np.column_stack(
        (np.full(point_count, 45.0), np.linspace(20.0, 120.0, point_count))
    )


def _scores_for_offsets(atlas, offsets, time_count=15, orientation_count=24):
    """Create exact horizontal orientation evidence at requested offsets."""

    scores = np.zeros((time_count, orientation_count, 100, 145), np.float32)
    for time_index, offset in enumerate(offsets):
        y = int(round(atlas[0, 0] - offset))
        x = np.rint(atlas[:, 1]).astype(int)
        scores[time_index, 0, y, x] = 1.0
    return scores


class DeformableWorldsheetTests:
    def test_open_curve_certificate_rejects_loop_but_keeps_curved_tube(self):
        angles = np.linspace(0.0, 1.8 * math.pi, 80)
        loop = np.column_stack(
            (40.0 + 8.0 * np.sin(angles), 40.0 + 8.0 * np.cos(angles))
        )
        curved_tube = np.column_stack(
            (
                40.0 + 8.0 * np.sin(np.linspace(0.0, 1.2, 80)),
                np.linspace(20.0, 100.0, 80),
            )
        )

        loop_certificate = open_curve_certificate(loop)
        tube_certificate = open_curve_certificate(curved_tube)

        assert loop_certificate.endpoint_separation_fraction < 0.25
        assert tube_certificate.endpoint_separation_fraction > 0.90

    def test_attached_trace_outranks_stronger_disconnected_filament(self):
        disconnected = atlas_hypothesis_selection_key(
            prototype_score=4.0,
            terminal_blobness=0.05,
            terminal_preexisting_support=0.05,
            proximal_eventual_support_fraction=0.0,
            maximum_unsupported_gap_px=18.0,
        )
        attached = atlas_hypothesis_selection_key(
            prototype_score=1.0,
            terminal_blobness=0.05,
            terminal_preexisting_support=0.05,
            proximal_eventual_support_fraction=1.0,
            maximum_unsupported_gap_px=1.0,
        )

        assert attached > disconnected

    def test_new_faint_atlas_beats_brighter_preexisting_branch(self):
        old_branch = atlas_prototype_score(
            length_px=46.0,
            mean_support=0.43,
            supported_fraction=0.83,
            displacement_p90_px=4.0,
            preexisting_support=0.42,
            terminal_blobness=0.75,
        )
        new_branch = atlas_prototype_score(
            length_px=52.0,
            mean_support=0.26,
            supported_fraction=0.55,
            displacement_p90_px=4.0,
            preexisting_support=0.05,
            terminal_blobness=0.05,
        )

        assert new_branch > old_branch

    def test_orientation_entropy_separates_round_body_from_linear_tip(self):
        round_body = np.zeros((12, 41, 41), dtype=np.float32)
        line_tip = np.zeros_like(round_body)
        yy, xx = np.mgrid[:41, :41]
        disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 6**2
        round_body[:, disk] = 0.7
        line_tip[0, 18:23, 14:27] = 0.7

        body_score = orientation_blobness(round_body, (20, 20), 7)
        tip_score = orientation_blobness(line_tip, (20, 20), 7)

        assert body_score > 0.7
        assert tip_score < 0.1

    def test_geometry_and_measurement_promotion_are_separate(self):
        geometry, measurement, reasons = worldsheet_promotion_decision(
            prototype_score=1.5,
            prototype_supported_fraction=0.66,
            terminal_blobness=0.02,
            accepted_frame_fraction=0.45,
        )

        assert geometry
        assert not measurement
        assert reasons == ("insufficient-accepted-timepoints",)

    def test_new_rounded_tip_is_not_confused_with_preexisting_pollen(self):
        new_tip = worldsheet_promotion_decision(
            prototype_score=1.5,
            prototype_supported_fraction=0.90,
            terminal_blobness=0.65,
            accepted_frame_fraction=0.90,
            terminal_preexisting_support=0.04,
        )
        old_body = worldsheet_promotion_decision(
            prototype_score=1.5,
            prototype_supported_fraction=0.90,
            terminal_blobness=0.65,
            accepted_frame_fraction=0.90,
            terminal_preexisting_support=0.25,
        )

        assert new_tip[:2] == (True, True)
        assert old_body[:2] == (False, False)
        assert "terminal-preexisting-pollen-body" in old_body[2]

    def test_unsupported_foreign_hypothesis_is_not_promoted(self):
        geometry, measurement, reasons = worldsheet_promotion_decision(
            prototype_score=0.25,
            prototype_supported_fraction=0.29,
            terminal_blobness=0.07,
            accepted_frame_fraction=0.0,
        )

        assert not geometry
        assert not measurement
        assert "weak-global-prototype-score" in reasons
        assert "insufficient-global-material-support" in reasons

    def test_disconnected_distal_tube_cannot_pass_as_pollen_attachment(self):
        geometry, measurement, reasons = worldsheet_promotion_decision(
            prototype_score=1.4,
            prototype_supported_fraction=0.80,
            terminal_blobness=0.05,
            accepted_frame_fraction=0.90,
            proximal_eventual_support_fraction=0.0,
            maximum_unsupported_gap_px=18.0,
        )

        assert not geometry
        assert not measurement
        assert "unsupported-pollen-attachment" in reasons
        assert "disconnected-material-gap" in reasons

    def test_closed_grain_boundary_cannot_pass_as_tube_geometry(self):
        geometry, measurement, reasons = worldsheet_promotion_decision(
            prototype_score=2.0,
            prototype_supported_fraction=0.90,
            terminal_blobness=0.02,
            accepted_frame_fraction=0.95,
            proximal_eventual_support_fraction=1.0,
            maximum_unsupported_gap_px=1.0,
            endpoint_separation_fraction=0.14,
        )

        assert not geometry
        assert not measurement
        assert "closed-or-looping-curve" in reasons

    def test_material_certificate_finds_pollen_gap_without_rejecting_small_hole(self):
        atlas = _straight_atlas(point_count=51)
        active = np.ones((12, len(atlas)), dtype=bool)
        support = np.full(active.shape, 0.7, dtype=np.float32)
        support[:, :10] = 0.01

        disconnected = material_continuity_certificate(support, active, atlas)
        support[:, :10] = 0.7
        support[:, 24] = 0.01
        small_hole = material_continuity_certificate(support, active, atlas)
        short_hidden_neck = material_continuity_certificate(
            np.where(np.arange(len(atlas))[None, :] < 3, 0.01, 0.7).repeat(
                len(active), axis=0
            ).astype(np.float32),
            active,
            atlas,
            proximal_occlusion_px=6.0,
        )

        assert disconnected.proximal_eventual_support_fraction == 0.0
        assert disconnected.maximum_unsupported_gap_px >= 18.0
        assert small_hole.proximal_eventual_support_fraction == 1.0
        assert small_hole.maximum_unsupported_gap_px < 6.0
        assert short_hidden_neck.proximal_eventual_support_fraction == 1.0
        assert short_hidden_neck.maximum_unsupported_gap_px == 0.0

    def test_recovers_smooth_curve_motion_without_changing_material_order(self):
        atlas = _straight_atlas()
        true_offsets = 4.0 * np.sin(np.linspace(0.0, 2.0 * math.pi, 15))
        scores = _scores_for_offsets(atlas, true_offsets)

        result = fit_deformable_worldsheet(scores, atlas)

        fitted = np.median(result.displacements_px[:, 3:], axis=1)
        assert np.median(np.abs(fitted - true_offsets)) < 0.75
        assert np.allclose(result.curves_yx[:, :, 1], atlas[None, :, 1])
        assert result.final_energy < result.initial_energy

    def test_global_surface_rejects_one_frame_parallel_distractor(self):
        atlas = _straight_atlas()
        scores = _scores_for_offsets(atlas, np.zeros(15))
        middle = len(scores) // 2
        x = np.rint(atlas[12:42, 1]).astype(int)
        scores[middle, 0, 45, x] = 0.0
        scores[middle, 0, 40, x] = 1.0
        framewise_offsets = np.argmax(
            scores[middle, 0][:, x],
            axis=0,
        )

        result = fit_deformable_worldsheet(
            scores,
            atlas,
            config=DeformableWorldsheetConfig(pairwise_smoothness=0.35),
        )

        assert np.median(framewise_offsets) == 40
        assert np.max(np.abs(result.displacements_px[middle, 12:42])) <= 1.0

    def test_causal_material_signature_rejects_persistent_brighter_crossover(self):
        atlas = _straight_atlas(point_count=61)
        scores = np.zeros((18, 24, 110, 145), dtype=np.float32)
        x = np.rint(atlas[:, 1]).astype(int)
        scores[:, 0, 45, x] = 0.42
        births = np.full((24, 110, 145), 18.0, dtype=np.float32)
        births[0, 45, x] = np.linspace(4.0, 14.0, len(atlas))
        foreign_y = np.rint(45 + 0.20 * (atlas[:, 1] - 70)).astype(int)
        for time_index in range(7, len(scores)):
            scores[time_index, 0, foreign_y, x] = 1.0
        births[0, foreign_y, x] = 0.0
        births[0, 45, x] = np.linspace(4.0, 14.0, len(atlas))
        config = DeformableWorldsheetConfig(
            normal_radius_px=6.0,
            pairwise_smoothness=0.35,
        )

        appearance_only = fit_deformable_worldsheet(scores, atlas, config=config)
        causal = fit_deformable_worldsheet(
            scores,
            atlas,
            config=config,
            orientation_birth=births,
        )

        assert np.max(np.abs(appearance_only.displacements_px)) == 6.0
        assert np.max(np.abs(causal.displacements_px)) == 0.0
        assert np.max(causal.birth_identity_error_samples) == 0.0

    def test_inactive_future_material_and_root_are_locked(self):
        atlas = _straight_atlas()
        scores = _scores_for_offsets(atlas, np.full(15, 3.0))
        front = np.arange(15) + 8

        result = fit_deformable_worldsheet(scores, atlas, front_indices=front)

        assert np.all(result.displacements_px[:, :2] == 0.0)
        for time_index, tip in enumerate(front):
            assert np.all(result.displacements_px[time_index, tip + 1 :] == 0.0)
            assert not np.any(result.active[time_index, tip + 1 :])
