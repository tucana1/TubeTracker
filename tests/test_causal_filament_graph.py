"""Tests for directed crossing resolution in the causal filament graph."""

import cv2 as cv
import numpy as np

from prototypes.v17_birth_topology.track import (
    AtlasConfig,
    _select_birth_ordered_path,
    _skeleton_path_candidates,
)
from tubetracker.causal_filament_graph import (
    CausalFilamentGraphConfig,
    CausalWorldsheetConfig,
    score_causal_path,
    select_causal_worldsheet,
    trace_causal_filament,
)


def _long_crossing():
    skeleton = np.zeros((201, 111), dtype=np.uint8)
    cv.line(skeleton, (10, 100), (100, 100), 1, 1)
    cv.line(skeleton, (60, 0), (60, 200), 1, 1)
    prior = np.column_stack(
        (np.full(48, 100.0), np.arange(10.0, 58.0))
    )
    return skeleton.astype(bool), prior


def test_directed_lane_beats_longer_planar_crossover():
    skeleton, prior = _long_crossing()
    root = (100, 10)
    legacy_path = _select_birth_ordered_path(
        _skeleton_path_candidates(skeleton, root),
        np.zeros(skeleton.shape, dtype=np.int32),
        AtlasConfig(),
        skeleton,
    )[0]

    result = trace_causal_filament(skeleton, root, prior_yx=prior)

    assert tuple(legacy_path[-1]) == (0, 60)
    assert tuple(result.selected.path_yx[-1]) == (100.0, 100.0)
    assert result.selected.maximum_junction_turn_degrees == 0.0
    assert result.score_margin > 0.0


def test_construction_order_rejects_a_preexisting_branch_without_a_prior():
    skeleton = np.zeros((121, 151), dtype=np.uint8)
    cv.line(skeleton, (10, 60), (140, 60), 1, 1)
    cv.line(skeleton, (70, 60), (135, 105), 1, 1)
    birth = np.zeros(skeleton.shape, dtype=np.float64)
    birth[60, 10:141] = np.linspace(10.0, 80.0, 131)
    config = CausalFilamentGraphConfig(
        maximum_junction_turn_degrees=80.0,
        turn_penalty=0.0,
        prior_prefix_distance_penalty=0.0,
        prior_extension_turn_penalty=0.0,
        length_reward_per_px=0.01,
    )

    result = trace_causal_filament(
        skeleton,
        (60, 10),
        birth_time=birth,
        config=config,
    )

    assert tuple(result.selected.path_yx[-1]) == (60.0, 140.0)
    assert result.selected.birth_progress > 0.0
    assert all(
        candidate.birth_backtrack_fraction > 0.0
        for candidate in result.alternatives
        if candidate.path_yx[-1, 0] > 90
    )


def test_path_score_reports_junction_identity_violation():
    skeleton, prior = _long_crossing()
    wrong = np.vstack(
        (
            np.column_stack((np.full(51, 100.0), np.arange(10.0, 61.0))),
            np.column_stack((np.arange(99.0, -1.0, -1.0), np.full(100, 60.0))),
        )
    )

    candidate = score_causal_path(wrong, skeleton, prior_yx=prior)

    assert candidate.maximum_junction_turn_degrees >= 89.0
    assert candidate.score < -1_000.0


def test_root_is_snapped_to_the_nearest_filament_point():
    skeleton = np.zeros((20, 30), dtype=bool)
    skeleton[10, 5:25] = True

    result = trace_causal_filament(skeleton, (12.0, 4.0))

    assert tuple(result.selected.path_yx[0]) == (10.0, 5.0)
    assert tuple(result.selected.path_yx[-1]) == (10.0, 24.0)


def test_worldsheet_prevents_a_late_foreign_branch_handoff():
    config = CausalFilamentGraphConfig(
        maximum_junction_turn_degrees=80.0,
        turn_penalty=0.0,
        prior_prefix_distance_penalty=0.0,
        prior_extension_turn_penalty=0.0,
        length_reward_per_px=0.02,
    )
    frame_candidates = []
    target_tips = [50, 58, 66, 74, 82, 90]
    for target_tip in target_tips:
        skeleton = np.zeros((161, 161), dtype=np.uint8)
        cv.line(skeleton, (10, 80), (target_tip, 80), 1, 1)
        cv.line(skeleton, (1, 40), (139, 120), 1, 1)
        result = trace_causal_filament(
            skeleton,
            (80, 10),
            config=config,
        )
        frame_candidates.append((result.selected, *result.alternatives))

    framewise_final = frame_candidates[-1][0]
    worldsheet = select_causal_worldsheet(
        frame_candidates,
        CausalWorldsheetConfig(maximum_growth_px_per_step=10.0),
    )

    assert tuple(framewise_final.path_yx[-1]) != (80.0, 90.0)
    assert tuple(worldsheet.selected[-1].path_yx[-1]) == (80.0, 90.0)
    assert np.all(
        np.diff([path.arclength_px[-1] for path in worldsheet.selected]) >= 0.0
    )
