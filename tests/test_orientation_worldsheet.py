"""Tests for crossing-preserving position-orientation tube tracing."""

import math

import cv2 as cv
import numpy as np

from prototypes.v22_coupled_ribbon_worldsheet.audit import (
    _normalized_curve_agreement,
)
from tubetracker.orientation_worldsheet import (
    aggregate_paired_wall_history,
    causal_orientation_changepoint,
    CoupledRibbonTraceConfig,
    OrientationScoreConfig,
    OrientationTraceConfig,
    paired_wall_orientation_features,
    paired_wall_orientation_score,
    persistent_orientation_birth,
    persistent_orientation_score,
    propose_pollen_roots,
    ribbon_path_certificate,
    trace_coupled_ribbon_lifted,
    trace_orientation_lifted,
)


def _draw_tube_walls(image, start_yx, end_yx, half_width, value):
    """Draw a synthetic brightfield tube as two dark parallel walls."""

    start = np.asarray(start_yx, dtype=np.float64)
    end = np.asarray(end_yx, dtype=np.float64)
    tangent = end - start
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray((-tangent[1], tangent[0]))
    for side in (-half_width, half_width):
        first = tuple(np.rint((start + side * normal)[::-1]).astype(int))
        last = tuple(np.rint((end + side * normal)[::-1]).astype(int))
        cv.line(image, first, last, value, 1, cv.LINE_AA)


def _crossing_frame(angle_degrees=20):
    """Create a faint owned tube crossed by a darker pre-existing tube."""

    image = np.full((130, 175), 190, dtype=np.uint8)
    _draw_tube_walls(image, (65, 8), (65, 155), 3, 85)
    angle = math.radians(angle_degrees)
    direction = np.asarray((math.sin(angle), math.cos(angle)))
    crossing = np.asarray((65.0, 82.0))
    foreign_start = crossing - 95.0 * direction
    foreign_end = crossing + 95.0 * direction
    _draw_tube_walls(image, foreign_start, foreign_end, 3, 20)
    return cv.GaussianBlur(image, (0, 0), 0.7), foreign_start, foreign_end


class OrientationWorldsheetTests:
    def test_normalized_curve_agreement_exposes_offset_and_length_change(self):
        first = np.column_stack((np.zeros(11), np.linspace(0.0, 10.0, 11)))
        shifted = first + np.asarray((1.0, 0.0))
        shorter = np.column_stack((np.zeros(9), np.linspace(0.0, 8.0, 9)))

        offset = _normalized_curve_agreement(first, shifted)
        length_change = _normalized_curve_agreement(first, shorter)

        assert np.isclose(offset["median_corresponding_error_px"], 1.0)
        assert np.isclose(offset["relative_length_difference"], 0.0)
        assert length_change["relative_length_difference"] > 0.20

    def test_changepoint_recovers_early_growth_without_fixed_warmup(self):
        scores = np.zeros((20, 4, 7, 9), dtype=np.float32)
        scores[:, 0, 2, 2] = 0.7
        scores[3:, 1, 3, 4] = 0.55
        scores[8:11, 2, 4, 6] = 0.9

        result = causal_orientation_changepoint(scores)

        assert result.birth_sample[0, 2, 2] == len(scores)
        assert result.birth_sample[1, 3, 4] == 3
        assert result.evidence[1, 3, 4] > 0.5
        assert result.evidence[1, 3, 4] > result.evidence[2, 4, 6]

    def test_pollen_rim_search_finds_new_tube_without_legacy_path_seed(self):
        image = np.full((90, 130), 190, dtype=np.uint8)
        cv.circle(image, (35, 45), 8, 45, 2, cv.LINE_AA)
        cv.line(image, (43, 42), (118, 42), 65, 1, cv.LINE_AA)
        cv.line(image, (43, 48), (118, 48), 65, 1, cv.LINE_AA)
        score = paired_wall_orientation_score(image)
        births = np.full(score.shape, 100.0, dtype=np.float32)
        old_pollen = np.zeros(image.shape, dtype=np.uint8)
        cv.circle(old_pollen, (35, 45), 10, 1, 3)
        births[:, old_pollen.astype(bool)] = 0.0
        births[0, 40:51, 42:122] = 10.0

        proposals = propose_pollen_roots(
            score,
            (45, 35),
            8,
            birth_time=births,
            minimum_material_birth=2.0,
        )

        assert proposals
        assert proposals[0].direction_yx[1] > 0.95
        assert abs(proposals[0].direction_yx[0]) < 0.1
        assert proposals[0].support > 0.5

    def test_paired_wall_score_retains_tube_direction(self):
        image = np.full((80, 120), 190, dtype=np.uint8)
        _draw_tube_walls(image, (40, 8), (40, 110), 3, 55)

        score = paired_wall_orientation_score(image)

        horizontal = float(np.mean(score[0, 40, 20:100]))
        vertical = float(np.mean(score[score.shape[0] // 2, 40, 20:100]))
        assert horizontal > 0.6
        assert horizontal > 3.0 * vertical

    def test_paired_wall_features_preserve_width_and_two_sided_evidence(self):
        tube = np.full((80, 120), 190, dtype=np.uint8)
        single_line = tube.copy()
        _draw_tube_walls(tube, (40, 8), (40, 110), 3, 55)
        cv.line(single_line, (8, 40), (110, 40), 55, 1, cv.LINE_AA)

        tube_features = paired_wall_orientation_features(tube)
        line_features = paired_wall_orientation_features(single_line)
        region = np.s_[20:100]
        tube_pair = float(np.mean(tube_features.paired_score[0, 40, region]))
        line_pair = float(np.mean(line_features.paired_score[0, 40, region]))
        inferred_width = float(
            np.median(tube_features.half_width_px[0, 40, region])
        )

        assert tube_pair > 0.5
        assert tube_pair > 2.0 * line_pair
        assert abs(inferred_width - 3.0) <= 0.5

        path = np.column_stack(
            (np.full(81, 40.0), np.linspace(20.0, 100.0, 81))
        )
        directions = np.zeros(len(path), dtype=np.float64)
        certificate = ribbon_path_certificate(
            tube_features,
            path,
            directions,
        )

        assert certificate.paired_supported_fraction > 0.8
        assert abs(certificate.median_half_width_px - 3.0) <= 0.5

    def test_temporal_ribbon_aggregate_keeps_supported_material_width(self):
        first = np.full((80, 120), 190, dtype=np.uint8)
        second = first.copy()
        _draw_tube_walls(first, (40, 8), (40, 110), 3, 70)
        _draw_tube_walls(second, (40, 8), (40, 110), 3, 60)
        features = [
            paired_wall_orientation_features(image) for image in (first, second)
        ]

        aggregate = aggregate_paired_wall_history(
            np.asarray([item.score for item in features]),
            np.asarray([item.paired_score for item in features]),
            np.asarray([item.half_width_px for item in features]),
            np.asarray([item.wall_balance for item in features]),
        )

        assert float(np.mean(aggregate.paired_score[0, 40, 20:100])) > 0.5
        assert abs(float(np.median(aggregate.half_width_px[0, 40, 20:100])) - 3.0) <= 0.5

    def test_orientation_score_recognizes_bright_phase_contrast_tube(self):
        image = np.full((80, 120), 90, dtype=np.uint8)
        _draw_tube_walls(image, (40, 8), (40, 110), 3, 185)

        score = paired_wall_orientation_score(image)

        horizontal = float(np.mean(score[0, 40, 20:100]))
        vertical = float(np.mean(score[score.shape[0] // 2, 40, 20:100]))
        assert horizontal > 0.5
        assert horizontal > 3.0 * vertical

    def test_persistent_birth_is_specific_to_orientation_and_time(self):
        scores = np.zeros((12, 4, 7, 9), dtype=np.float32)
        scores[:, 0, 3, 2] = 0.8
        scores[6:, 1, 3, 2] = 0.9

        birth = persistent_orientation_birth(
            scores,
            warmup_samples=4,
            persistence_samples=3,
            minimum_change=0.1,
        )

        assert birth[0, 3, 2] == 0
        assert birth[1, 3, 2] == 8
        assert birth[2, 3, 2] == len(scores)

    def test_persistent_score_prefers_faint_material_over_late_flash(self):
        scores = np.zeros((20, 4, 7, 9), dtype=np.float32)
        scores[7:, 0, 3, 2] = 0.18
        scores[-3:, 0, 3, 6] = 0.90

        fused = persistent_orientation_score(scores, warmup_samples=5)

        assert fused[0, 3, 2] > 0.5
        assert fused[0, 3, 2] > 3.0 * fused[0, 3, 6]

    def test_causal_orientation_birth_prevents_shallow_crossing_switch(self):
        image, foreign_start, foreign_end = _crossing_frame(20)
        score = paired_wall_orientation_score(image)
        births = np.full(score.shape, 200.0, dtype=np.float32)
        foreign = np.zeros(image.shape, dtype=np.uint8)
        cv.line(
            foreign,
            tuple(np.rint(foreign_start[::-1]).astype(int)),
            tuple(np.rint(foreign_end[::-1]).astype(int)),
            1,
            9,
        )
        births[:, foreign.astype(bool)] = 0.0
        for x in range(5, 161):
            births[0, 62:69, x] = max(2.0, (x - 5) / 2.0)
        config = OrientationTraceConfig(
            maximum_length_px=150.0,
            minimum_length_px=20.0,
            maximum_gap_steps=10,
            curvature_penalty=0.25,
        )

        planar_age_trace = trace_orientation_lifted(
            score,
            (65, 8),
            (0, 1),
            config=config,
        )
        causal_trace = trace_orientation_lifted(
            score,
            (65, 8),
            (0, 1),
            birth_time=births,
            minimum_material_birth=2.0,
            config=config,
        )

        assert np.max(np.abs(planar_age_trace.path_yx[:, 0] - 65.0)) > 15.0
        assert np.max(np.abs(causal_trace.path_yx[:, 0] - 65.0)) < 2.0
        assert causal_trace.path_yx[-1, 1] > 150.0

    def test_coupled_ribbon_keeps_width_and_birth_identity_at_crossing(self):
        image = np.full((130, 175), 190, dtype=np.uint8)
        _draw_tube_walls(image, (65, 8), (65, 160), 2, 100)
        angle = math.radians(20)
        direction = np.asarray((math.sin(angle), math.cos(angle)))
        crossing = np.asarray((65.0, 82.0))
        foreign_start = crossing - 95.0 * direction
        foreign_end = crossing + 95.0 * direction
        _draw_tube_walls(image, foreign_start, foreign_end, 5, 20)
        image = cv.GaussianBlur(image, (0, 0), 0.7)
        features = paired_wall_orientation_features(image)
        births = np.full(features.score.shape, 200.0, dtype=np.float32)
        foreign = np.zeros(image.shape, dtype=np.uint8)
        cv.line(
            foreign,
            tuple(np.rint(foreign_start[::-1]).astype(int)),
            tuple(np.rint(foreign_end[::-1]).astype(int)),
            1,
            12,
        )
        births[:, foreign.astype(bool)] = 0.0
        for x in range(5, 165):
            births[0, 60:71, x] = max(2.0, (x - 5) / 2.0)

        result = trace_coupled_ribbon_lifted(
            features,
            (65, 8),
            (0, 1),
            birth_time=births,
            minimum_material_birth=2.0,
            config=CoupledRibbonTraceConfig(
                maximum_length_px=150.0,
                minimum_length_px=20.0,
                maximum_gap_steps=10,
                curvature_penalty=0.20,
                beam_width=150,
            ),
        )

        assert result.path_yx[-1, 1] > 150.0
        assert np.max(np.abs(result.path_yx[:, 0] - 65.0)) < 3.0
        assert result.paired_supported_fraction > 0.85
        assert result.endpoint_separation_fraction > 0.95

    def test_coupled_ribbon_reaches_explicit_censored_terminal(self):
        """A known image edge outranks an attractive premature endpoint."""

        image = np.full((100, 165), 190, dtype=np.uint8)
        _draw_tube_walls(image, (50, 8), (50, 120), 3, 70)
        features = paired_wall_orientation_features(
            cv.GaussianBlur(image, (0, 0), 0.7)
        )
        terminal = np.zeros(image.shape, dtype=bool)
        terminal[:, 144:147] = True

        result = trace_coupled_ribbon_lifted(
            features,
            (50, 8),
            (0, 1),
            terminal_mask=terminal,
            config=CoupledRibbonTraceConfig(
                maximum_length_px=150.0,
                minimum_length_px=20.0,
                maximum_gap_steps=24,
                maximum_total_turn_bins=3,
                curvature_penalty=0.20,
                beam_width=180,
            ),
        )

        assert result.path_yx[-1, 1] >= 144.0
        assert abs(result.path_yx[-1, 0] - 50.0) < 5.0

    def test_ribbon_search_retains_a_weaker_complete_lane_as_an_alternative(self):
        image, _, _ = _crossing_frame(angle_degrees=30)
        features = paired_wall_orientation_features(image)

        result = trace_coupled_ribbon_lifted(
            features,
            (65, 8),
            (0, 1),
            config=CoupledRibbonTraceConfig(
                maximum_length_px=145.0,
                minimum_length_px=20.0,
                maximum_gap_steps=10,
                beam_width=180,
                maximum_alternative_paths=16,
                alternative_tip_separation_px=8.0,
                alternative_prefix_separation_px=3.0,
            ),
        )

        assert abs(result.path_yx[-1, 0] - 65.0) > 20.0
        assert any(
            abs(candidate.path_yx[-1, 0] - 65.0) < 6.0
            and candidate.path_yx[-1, 1] > 145.0
            for candidate in result.alternatives
        )

    def test_material_prior_moves_with_root_and_preserves_prefix(self):
        image = np.full((100, 150), 190, dtype=np.uint8)
        _draw_tube_walls(image, (54, 14), (54, 130), 3, 55)
        score = paired_wall_orientation_score(image)
        prior = np.column_stack(
            (np.full(50, 50.0), np.linspace(10.0, 90.0, 50))
        )

        result = trace_orientation_lifted(
            score,
            (54, 14),
            (0, 1),
            prior_curve_yx=prior,
            config=OrientationTraceConfig(
                maximum_extension_px=45.0,
                maximum_gap_steps=8,
            ),
        )

        assert result.prior_prefix_error_px < 1.0
        assert result.path_yx[-1, 1] > 120.0
        assert np.max(np.abs(result.path_yx[:, 0] - 54.0)) < 2.0

    def test_ribbon_prior_preserves_complete_root_connected_path(self):
        image = np.full((120, 175), 190, dtype=np.uint8)
        _draw_tube_walls(image, (64, 8), (64, 160), 3, 75)
        crossing_angle = math.radians(25)
        crossing_direction = np.asarray(
            (math.sin(crossing_angle), math.cos(crossing_angle))
        )
        crossing = np.asarray((64.0, 85.0))
        _draw_tube_walls(
            image,
            crossing - 95.0 * crossing_direction,
            crossing + 95.0 * crossing_direction,
            3,
            25,
        )
        features = paired_wall_orientation_features(
            cv.GaussianBlur(image, (0, 0), 0.7)
        )
        prior = np.column_stack(
            (np.full(80, 60.0), np.linspace(4.0, 145.0, 80))
        )

        result = trace_coupled_ribbon_lifted(
            features,
            (64, 8),
            (0, 1),
            prior_curve_yx=prior,
            config=CoupledRibbonTraceConfig(
                maximum_length_px=155.0,
                maximum_extension_px=12.0,
                minimum_length_px=20.0,
                maximum_gap_steps=8,
                curvature_penalty=0.12,
                beam_width=180,
            ),
        )

        assert result.prior_prefix_error_px < 1.5
        assert result.path_yx[-1, 1] > 145.0
        assert np.max(np.abs(result.path_yx[:, 0] - 64.0)) < 3.0

    def test_root_portal_bridges_one_hidden_neck_but_not_a_later_gap(self):
        score = np.zeros((24, 70, 120), dtype=np.float32)
        score[0, 35, 29:61] = 0.8
        score[0, 35, 70:111] = 0.8
        result = trace_orientation_lifted(
            score,
            (35, 20),
            (0, 1),
            config=OrientationTraceConfig(
                step_px=1.0,
                root_occlusion_px=6.0,
                maximum_initial_gap_steps=3,
                maximum_gap_steps=2,
                minimum_length_px=5.0,
                maximum_length_px=100.0,
                beam_width=200,
            ),
        )

        assert result.path_yx[-1, 1] >= 58.0
        assert result.path_yx[-1, 1] < 70.0
