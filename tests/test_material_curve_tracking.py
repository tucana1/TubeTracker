"""Checks for ordered material curves, paired walls, and crossover resistance."""

import cv2 as cv
import numpy as np

from tubetracker.material_curve_tracking import (
    MaterialCurveConfig,
    MaterialQuerySet,
    _coherent_keyframe_alignment,
    append_material_extension,
    build_material_queries,
    coherent_material_prediction,
    project_inextensible_curve,
    trace_material_curve,
)
from tubetracker.pollen_anchored_chain import (
    PollenAnchoredChainConfig,
    PollenAnchoredEvidence,
    PollenAnchoredMeasurement,
)
from tubetracker.material_ribbon import observe_material_ribbon


def _measurement(curve, usable=True, status="chain_updated"):
    """Build one compact coarse measurement for material-tracking tests."""
    curve = np.asarray(curve, dtype=np.float64)
    length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
    return PollenAnchoredMeasurement(
        status=status,
        usable=usable,
        length_px=length,
        centerline_yx=curve,
        candidate_centerline_yx=np.empty((0, 2)),
        structure_support=1.0,
        seed_confirmations=4,
        chain_supported_fraction=1.0,
        root_attachment_error_px=0.0,
        maximum_centerline_step_px=2.0,
        neighbor_contact=False,
    )


def test_material_queries_retain_uncertain_curves_for_later_validation():
    """Candidate generation should preserve data rather than discard review curves."""
    first = np.column_stack((np.full(8, 20.0), np.arange(10.0, 26.0, 2.0)))
    grown = np.column_stack((np.full(10, 20.0), np.arange(10.0, 30.0, 2.0)))
    switched = grown.copy()
    switched[5:, 0] += np.arange(1.0, 6.0) * 3.0
    config = MaterialCurveConfig(
        keyframe_interval=1,
        keyframe_minimum_growth_px=1.0,
    )
    queries = build_material_queries(
        [_measurement(first), _measurement(grown), _measurement(switched)],
        config,
    )
    assert np.array_equal(queries.keyframes, [0, 1, 2])
    assert set(queries.group_indices) == {0, 1, 2}


def test_material_consensus_rejects_one_keyframe_that_switches_branches():
    """Two coherent curve histories should overrule one crossing-branch jump."""
    prior = np.column_stack((np.full(11, 20.0), np.arange(10.0, 32.0, 2.0)))
    arc = np.arange(0.0, 22.0, 2.0)
    query_yx = prior.copy()
    points = []
    arcs = []
    groups = []
    for group in range(3):
        points.append(
            np.column_stack((np.zeros(len(arc)), query_yx[:, ::-1]))
        )
        arcs.append(arc)
        groups.append(np.full(len(arc), group))
    queries = MaterialQuerySet(
        points_txy=np.concatenate(points).astype(np.float32),
        arc_positions_px=np.concatenate(arcs),
        group_indices=np.concatenate(groups).astype(int),
        keyframes=np.zeros(3, dtype=int),
    )
    tracks = np.repeat(queries.points_txy[None, :, 1:], 2, axis=0)
    tracks[1, queries.group_indices == 0] += np.asarray([1.0, 1.0])
    tracks[1, queries.group_indices == 1] += np.asarray([1.2, 1.1])
    tracks[1, queries.group_indices == 2] += np.asarray([0.0, 12.0])
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    predicted, confidence, retained, rejected, _ = coherent_material_prediction(
        prior,
        np.zeros_like(prior),
        1,
        queries,
        tracks,
        visibility,
        {0, 1, 2},
        MaterialCurveConfig(root_lock_points=1),
    )
    assert retained == 2
    assert rejected == 1
    assert np.all(confidence[1:] > 0.0)
    assert np.median(predicted[2:, 0] - prior[2:, 0]) < 2.0
    assert np.median(predicted[2:, 1] - prior[2:, 1]) > 0.5


def test_distal_growth_keeps_every_existing_material_node_unchanged():
    """Growth may append node identities but cannot resample the existing body."""
    chain = np.asarray([[20.0, 10.0], [20.0, 12.0], [20.0, 14.0]])
    extension = np.asarray([[20.0, 14.0], [20.0, 18.0], [22.0, 22.0]])
    grown = append_material_extension(chain, extension, 2.0, 20)
    assert len(grown) > len(chain)
    assert np.array_equal(grown[: len(chain)], chain)


def test_material_projection_cannot_stretch_existing_edges():
    """Tracker disagreement may bend a tube but cannot create false length."""
    prior = np.column_stack((np.full(8, 20.0), np.arange(10.0, 26.0, 2.0)))
    target = prior.copy()
    target[3:, 1] += np.arange(1.0, 6.0) * 12.0
    projected = project_inextensible_curve(target, prior, root_lock_points=2)
    assert np.allclose(
        np.linalg.norm(np.diff(projected, axis=0), axis=1),
        np.linalg.norm(np.diff(prior, axis=0), axis=1),
    )
    assert np.array_equal(projected[:2], prior[:2])


def test_material_projection_allows_bending_without_length_change():
    """Arc length should remain fixed even when the root-to-tip chord shortens."""
    prior = np.column_stack((np.zeros(8), np.arange(0.0, 16.0, 2.0)))
    angles = np.linspace(0.0, np.pi / 2.0, len(prior) - 1)
    target = np.vstack(
        (
            prior[0],
            prior[0]
            + np.cumsum(
                2.0 * np.column_stack((np.sin(angles), np.cos(angles))),
                axis=0,
            ),
        )
    )
    projected = project_inextensible_curve(target, prior, root_lock_points=1)
    prior_length = np.sum(np.linalg.norm(np.diff(prior, axis=0), axis=1))
    projected_length = np.sum(
        np.linalg.norm(np.diff(projected, axis=0), axis=1)
    )
    assert np.isclose(projected_length, prior_length)
    assert np.linalg.norm(projected[-1] - projected[0]) < prior_length


def test_keyframe_bending_cannot_be_counted_as_distal_growth():
    """A bent keyframe may add only its new arc length, not its tip displacement."""
    size = 80
    first = np.column_stack(
        (np.full(7, 40.0), np.arange(38.0, 52.0, 2.0))
    )
    directions = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    second = np.vstack((first[0], first[0] + np.cumsum(2.0 * directions, axis=0)))
    probability = np.ones((size, size), dtype=np.float32)
    gray = np.full((size, size), 180, dtype=np.uint8)
    evidence = [
        PollenAnchoredEvidence(gray, probability, probability, probability)
        for _ in range(2)
    ]
    coarse = [_measurement(first), _measurement(second)]
    config = MaterialCurveConfig(
        keyframe_interval=1,
        keyframe_minimum_growth_px=0.5,
        root_lock_points=1,
    )
    queries = build_material_queries(coarse, config)
    tracks = np.repeat(queries.points_txy[None, :, 1:], 2, axis=0)
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    result = trace_material_curve(
        evidence,
        np.zeros((2, size, size), dtype=bool),
        pollen_radius=5.0,
        coarse_measurements=coarse,
        queries=queries,
        tracks_xy=tracks,
        visibility=visibility,
        chain_config=PollenAnchoredChainConfig(
            crop_size=size,
            baseline_observations=1,
            minimum_chain_supported_fraction=0.0,
        ),
        config=config,
    )
    first_length = result.measurements[0].length_px
    second_length = result.measurements[1].length_px
    claimed_growth = coarse[1].length_px - coarse[0].length_px
    assert np.isclose(second_length - first_length, claimed_growth, atol=1e-5)
    assert result.diagnostics[1].length_consistent


def test_coherent_keyframe_offset_preserves_rooted_curve_identity():
    """Boundary recentering may shift a curve body without changing its shape."""
    config = MaterialCurveConfig(root_lock_points=2, node_spacing_px=2.0)
    arc = np.arange(0.0, 28.0, 4.0)
    expected = np.column_stack((np.full(len(arc), 30.0), 20.0 + arc))
    observed = expected.copy()
    observed[1:] += np.asarray([6.0, 3.0])
    assert _coherent_keyframe_alignment(observed, expected, arc, config)


def test_coherent_keyframe_offset_rejects_a_branch_shaped_deviation():
    """A progressively turning branch is not a coherent lateral recentering."""
    config = MaterialCurveConfig(root_lock_points=2, node_spacing_px=2.0)
    arc = np.arange(0.0, 28.0, 4.0)
    expected = np.column_stack((np.full(len(arc), 30.0), 20.0 + arc))
    observed = expected.copy()
    observed[1:, 0] += np.linspace(1.0, 14.0, len(arc) - 1)
    assert not _coherent_keyframe_alignment(observed, expected, arc, config)


def test_material_curve_stays_ordered_at_an_image_crossing():
    """A vertical crossing must not rewire a horizontally tracked material tube."""
    size = 64
    probability = np.zeros((size, size), dtype=np.float32)
    probability[31:34, 38:55] = 1.0
    probability[20:48, 47:50] = 1.0
    gray = np.full((size, size), 180, dtype=np.uint8)
    evidence = [
        PollenAnchoredEvidence(gray, probability, probability, probability)
        for _ in range(4)
    ]
    curve = np.column_stack((np.full(8, 32.0), np.arange(38.0, 54.0, 2.0)))
    coarse = [_measurement(curve) for _ in evidence]
    material_config = MaterialCurveConfig(
        keyframe_interval=20,
        keyframe_minimum_growth_px=20.0,
        root_lock_points=2,
        image_search_radius_px=2,
    )
    queries = build_material_queries(coarse, material_config)
    tracks = np.repeat(
        queries.points_txy[None, :, 1:], len(evidence), axis=0
    )
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    chain_config = PollenAnchoredChainConfig(
        crop_size=size,
        baseline_observations=1,
        minimum_chain_supported_fraction=0.5,
    )
    result = trace_material_curve(
        evidence,
        np.zeros((len(evidence), size, size), dtype=bool),
        pollen_radius=5.0,
        coarse_measurements=coarse,
        queries=queries,
        tracks_xy=tracks,
        visibility=visibility,
        chain_config=chain_config,
        config=material_config,
    )
    final = result.measurements[-1].centerline_yx
    assert np.max(np.abs(final[:, 0] - 32.0)) < 1.0
    assert np.all(np.diff(final[:, 1]) > 0.0)
    assert result.diagnostics[-1].accepted_group_count == 1


def test_unsupported_curve_does_not_drift_onto_unrelated_image_structure():
    """Without identity tracks, image ridges cannot gradually rewrite the curve."""
    size = 64
    probability = np.zeros((size, size), dtype=np.float32)
    probability[38:41, 38:55] = 1.0
    gray = np.full((size, size), 180, dtype=np.uint8)
    evidence = [
        PollenAnchoredEvidence(gray, probability, probability, probability)
        for _ in range(5)
    ]
    curve = np.column_stack((np.full(8, 32.0), np.arange(38.0, 54.0, 2.0)))
    coarse = [_measurement(curve) for _ in evidence]
    config = MaterialCurveConfig(keyframe_interval=20)
    queries = build_material_queries(coarse, config)
    tracks = np.repeat(
        queries.points_txy[None, :, 1:], len(evidence), axis=0
    )
    result = trace_material_curve(
        evidence,
        np.zeros((len(evidence), size, size), dtype=bool),
        pollen_radius=5.0,
        coarse_measurements=coarse,
        queries=queries,
        tracks_xy=tracks,
        visibility=np.zeros(tracks.shape[:2], dtype=bool),
        chain_config=PollenAnchoredChainConfig(
            crop_size=size,
            baseline_observations=1,
        ),
        config=config,
    )
    final = result.measurements[-1].centerline_yx
    assert np.max(np.abs(final[:, 0] - 32.0)) < 0.1
    assert not result.measurements[-1].usable


def _horizontal_ribbon():
    """Build a dark paired-wall ribbon with a deliberately offset prior."""
    gray = np.full((64, 64), 180, dtype=np.uint8)
    cv.line(gray, (8, 28), (56, 28), 50, 1)
    cv.line(gray, (8, 36), (56, 36), 50, 1)
    prior = np.column_stack(
        (np.full(20, 30.0), np.linspace(10.0, 54.0, 20))
    )
    return gray, prior


def test_paired_walls_recenter_an_offset_material_curve():
    """Two tube walls should locate their shared center instead of one edge."""
    gray, prior = _horizontal_ribbon()
    observation = observe_material_ribbon(gray, prior)
    assert np.median(observation.centerline_yx[4:, 0]) == 32.0
    assert np.median(observation.widths_px[4:]) == 8.0
    assert np.median(observation.confidence[4:]) > 0.5


def test_width_history_keeps_a_crossing_from_redefining_the_ribbon():
    """A perpendicular dark structure should not widen or turn the target tube."""
    gray, prior = _horizontal_ribbon()
    cv.line(gray, (32, 10), (32, 54), 50, 1)
    observation = observe_material_ribbon(
        gray,
        prior,
        prior_widths_px=np.full(len(prior), 8.0),
    )
    assert np.max(np.abs(observation.centerline_yx[4:, 0] - 32.0)) <= 1.0
    assert np.median(observation.widths_px[4:]) == 8.0
