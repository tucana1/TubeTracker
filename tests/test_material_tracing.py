"""Checks for pollen-attached tracing, chain integrity, and edge ownership."""

from dataclasses import replace
import unittest

import cv2 as cv
import numpy as np

import tubetracker.pollen_anchored_chain as anchored
from tubetracker.anchored_tracing import (
    detect_tip_burst,
    find_attached_birth_path,
    find_tip_extension,
    refine_chain_from_prior,
)
from tubetracker.curve_prototype import CurveTraceConfig, curve_length
from tubetracker.pollen_anchored_chain import (
    PollenAnchoredChainConfig,
    PollenAnchoredEvidence,
    centerline_to_source_xy,
    chain_integrity,
    persistent_baseline_novelty,
    stabilize_pollen_frames,
    trace_pollen_anchored_chain,
)
from tubetracker.topology_aware_tracing import (
    TopologyAwareTraceConfig,
    future_edge_hypotheses,
    predict_future_edge_cue,
)


def _evidence(probability):
    """Build synthetic evidence with identical structural and novelty maps."""
    probability = np.asarray(probability, dtype=np.float32)
    return PollenAnchoredEvidence(
        gray=np.zeros(probability.shape, dtype=np.uint8),
        vessel_probability=probability,
        novelty_probability=probability,
        fused_probability=probability,
    )


def test_stabilization_moves_a_translating_pollen_back_to_crop_center():
    """Pollen and attached material should share one fixed coordinate system."""
    frames = np.full((3, 80, 90), 220, dtype=np.uint8)
    centers = np.asarray([[30, 35], [37, 31], [44, 27]], dtype=float)
    for frame, center in zip(frames, centers):
        cv.circle(frame, tuple(center.astype(int)), 5, 40, -1)
        cv.line(
            frame,
            tuple((center + [5, 0]).astype(int)),
            tuple((center + [18, 0]).astype(int)),
            60,
            2,
        )
    stabilized, transforms = stabilize_pollen_frames(
        frames, centers, np.zeros(3), crop_size=64
    )
    darkest = [np.unravel_index(np.argmin(frame), frame.shape) for frame in stabilized]
    assert all(np.linalg.norm(np.asarray(point[::-1]) - [32, 32]) <= 5 for point in darkest)
    mapped = np.asarray(
        [
            transforms[index][:, :2] @ centers[index] + transforms[index][:, 2]
            for index in range(3)
        ]
    )
    assert np.allclose(mapped, [32, 32])


def test_baseline_novelty_suppresses_old_structure_and_keeps_new_tube():
    """Only material appearing after the baseline should receive novelty evidence."""
    frames = np.full((8, 60, 70), 180, dtype=np.uint8)
    frames[:, 10:13, 5:55] = 80
    frames[4:, 35:38, 20:60] = 70
    _, novelty = persistent_baseline_novelty(
        frames,
        baseline_observations=3,
        window=2,
        dark_offset=3,
        intensity_scale=24,
    )
    assert float(np.max(novelty[-1, 10:13, 5:55])) == 0.0
    assert float(np.mean(novelty[-1, 35:38, 20:60])) > 0.9


def test_chain_integrity_requires_support_continuity_and_root_attachment():
    """A plausible tip is invalid when its complete pollen-rooted path is broken."""
    config = replace(PollenAnchoredChainConfig(), crop_size=100)
    probability = np.zeros((100, 100), dtype=np.float32)
    cv.line(probability, (59, 50), (82, 50), 1.0, 3)
    chain = np.asarray([[50.0, x] for x in range(59, 83, 2)])
    valid, supported, root_error, step, contact = chain_integrity(
        probability, chain, [50, 50], pollen_radius=8, config=config
    )
    assert valid
    assert supported == 1.0
    assert root_error == 0.0
    assert step <= 2.0
    assert not contact

    broken = probability.copy()
    broken[:, 68:78] = 0.0
    valid, supported, _, _, _ = chain_integrity(
        broken, chain, [50, 50], pollen_radius=8, config=config
    )
    assert not valid
    assert supported < 1.0

    exclusion = np.zeros_like(probability, dtype=bool)
    exclusion[48:53, 78:84] = True
    valid, _, _, _, contact = chain_integrity(
        probability,
        chain,
        [50, 50],
        pollen_radius=8,
        config=config,
        exclusion=exclusion,
    )
    assert not valid
    assert contact


def test_sequence_confirms_one_birth_then_updates_the_same_full_chain(monkeypatch):
    """Growth must initialize from a repeated short path and retain that path."""
    config = replace(
        PollenAnchoredChainConfig(),
        crop_size=100,
        seed_confirmation_frames=2,
        minimum_seed_growth_px=4.0,
    )
    probability = np.ones((100, 100), dtype=np.float32)
    paths = iter(
        [
            np.asarray([[50.0, x] for x in range(59, 65)]),
            np.asarray([[50.0, x] for x in range(59, 72)]),
        ]
    )

    def fake_birth(*_args, **_kwargs):
        path = next(paths)
        return 1.0, path, {"support": 1.0, "radial_gain": 12.0}

    monkeypatch.setattr(anchored, "find_attached_birth_path", fake_birth)
    monkeypatch.setattr(
        anchored,
        "refine_chain_from_prior",
        lambda _probability, prior, **_kwargs: (prior.copy(), 1.0, 0.0),
    )
    monkeypatch.setattr(anchored, "find_tip_extension", lambda *_args: None)
    measurements = trace_pollen_anchored_chain(
        [_evidence(probability) for _ in range(3)],
        np.zeros((3, 100, 100), dtype=bool),
        pollen_radius=8,
        config=config,
    )
    assert [item.status for item in measurements] == [
        "birth_candidate",
        "chain_initialized",
        "chain_updated",
    ]
    assert not measurements[0].usable
    assert measurements[1].usable and measurements[2].usable
    assert np.allclose(
        measurements[1].centerline_yx,
        measurements[2].centerline_yx,
    )


def test_unsupported_full_chain_cannot_accept_a_new_tip(monkeypatch):
    """A lost centerline must fail closed before distal extension is considered."""
    config = replace(
        PollenAnchoredChainConfig(),
        crop_size=100,
        seed_confirmation_frames=2,
        minimum_seed_growth_px=4.0,
    )
    probability = np.ones((100, 100), dtype=np.float32)
    paths = iter(
        [
            np.asarray([[50.0, x] for x in range(59, 65)]),
            np.asarray([[50.0, x] for x in range(59, 72)]),
        ]
    )
    monkeypatch.setattr(
        anchored,
        "find_attached_birth_path",
        lambda *_args: (
            1.0,
            next(paths),
            {"support": 1.0, "radial_gain": 12.0},
        ),
    )
    monkeypatch.setattr(
        anchored,
        "refine_chain_from_prior",
        lambda _probability, prior, **_kwargs: (prior.copy(), 0.0, 0.0),
    )

    def forbidden_extension(*_args, **_kwargs):
        raise AssertionError("a broken parent chain cannot grow a new tip")

    monkeypatch.setattr(anchored, "find_tip_extension", forbidden_extension)
    measurements = trace_pollen_anchored_chain(
        [_evidence(probability) for _ in range(3)],
        np.zeros((3, 100, 100), dtype=bool),
        pollen_radius=8,
        config=config,
    )
    assert measurements[-1].status == "chain_held_for_review"
    assert not measurements[-1].usable


def test_neighbor_contact_permanently_censors_later_measurements(monkeypatch):
    """A later separation cannot restore identity after an ambiguous contact."""
    config = replace(
        PollenAnchoredChainConfig(),
        crop_size=100,
        neighbor_contact_margin_px=0,
        seed_confirmation_frames=2,
        minimum_seed_growth_px=4.0,
    )
    probability = np.ones((100, 100), dtype=np.float32)
    paths = iter(
        [
            np.asarray([[50.0, x] for x in range(59, 65)]),
            np.asarray([[50.0, x] for x in range(59, 72)]),
        ]
    )
    monkeypatch.setattr(
        anchored,
        "find_attached_birth_path",
        lambda *_args: (
            1.0,
            next(paths),
            {"support": 1.0, "radial_gain": 12.0},
        ),
    )
    monkeypatch.setattr(
        anchored,
        "refine_chain_from_prior",
        lambda _probability, prior, **_kwargs: (prior.copy(), 1.0, 0.0),
    )
    monkeypatch.setattr(anchored, "find_tip_extension", lambda *_args: None)
    exclusions = np.zeros((4, 100, 100), dtype=bool)
    exclusions[2, 48:53, 68:74] = True
    measurements = trace_pollen_anchored_chain(
        [_evidence(probability) for _ in range(4)],
        exclusions,
        pollen_radius=8,
        config=config,
    )
    assert measurements[2].status == "neighbor_contact_censored"
    assert measurements[2].neighbor_contact
    assert measurements[3].status == "neighbor_contact_censored"
    assert not measurements[3].usable


def test_centerline_round_trip_returns_source_coordinates():
    """Exported points should be mapped out of the stabilized analysis crop."""
    source_to_crop = np.asarray([[1.0, 0.0, -20.0], [0.0, 1.0, 15.0]])
    centerline_yx = np.asarray([[35.0, 30.0], [38.0, 34.0]])
    source = centerline_to_source_xy(centerline_yx, source_to_crop)
    assert np.allclose(source, [[50.0, 20.0], [54.0, 23.0]])


class AnchoredTracingTests(unittest.TestCase):
    """Protect attachment, rupture evidence, and forward-only growth constraints."""

    def test_birth_path_starts_at_grain_and_ignores_remote_debris(self):
        """A remote high-probability line cannot become this pollen's tube."""
        probability = np.zeros((100, 120), dtype=np.float32)
        temporal = np.zeros_like(probability)
        cv.line(probability, (40, 50), (72, 50), 1.0, 2)
        cv.line(temporal, (58, 50), (72, 50), 1.0, 2)
        cv.line(probability, (10, 85), (110, 85), 1.0, 3)
        exclusion = np.zeros(probability.shape, dtype=np.uint8)
        cv.circle(exclusion, (30, 50), 9, 1, -1)
        result = find_attached_birth_path(
            probability,
            temporal,
            center_yx=np.array([50.0, 30.0]),
            grain_radius=8.0,
            exclusion=exclusion.astype(bool),
            config=CurveTraceConfig(),
        )
        self.assertIsNotNone(result)
        _, path, _ = result
        self.assertLess(np.linalg.norm(path[0] - [50.0, 40.0]), 5.0)
        self.assertLess(np.max(np.abs(path[:, 0] - 50.0)), 4.0)
        self.assertLessEqual(curve_length(path), 40.0)

    def test_extension_moves_forward_without_retracing_history(self):
        """An accepted extension should add a short distal segment only."""
        probability = np.zeros((100, 120), dtype=np.float32)
        temporal = np.zeros_like(probability)
        cv.line(probability, (40, 50), (78, 50), 1.0, 2)
        cv.line(temporal, (64, 50), (78, 50), 1.0, 2)
        prior = np.array([[50.0, x] for x in range(40, 65, 2)])
        result = find_tip_extension(
            probability,
            temporal,
            prior,
            exclusion=np.zeros(probability.shape, dtype=bool),
            config=CurveTraceConfig(max_tip_step=18.0, max_tip_lateral_step=6.0),
        )
        self.assertIsNotNone(result)
        _, extension, _, temporal_score = result
        self.assertGreater(extension[-1, 1], prior[-1, 1])
        self.assertLess(np.max(np.abs(extension[:, 0] - 50.0)), 3.0)
        self.assertGreaterEqual(temporal_score, 0.07)

    def test_burst_requires_broad_change_not_narrow_tip_growth(self):
        """A broad local demarcation should score unlike a thin growing segment."""
        config = CurveTraceConfig()
        broad = np.zeros((100, 100), dtype=np.float32)
        cv.circle(broad, (50, 50), 10, 0.35, -1)
        candidate, score, reason = detect_tip_burst(
            broad, np.array([50.0, 50.0]), config
        )
        self.assertTrue(candidate)
        self.assertGreaterEqual(score, config.burst_score_threshold)
        self.assertIn("broad tip change", reason)

        narrow = np.zeros_like(broad)
        cv.line(narrow, (50, 50), (65, 50), 0.35, 2)
        candidate, _, _ = detect_tip_burst(
            narrow, np.array([50.0, 50.0]), config
        )
        self.assertFalse(candidate)

    def test_chain_refit_follows_local_ridge_without_switching_at_crossing(self):
        """A prior chain may move locally but must retain its ordered topology."""
        probability = np.zeros((100, 120), dtype=np.float32)
        cv.line(probability, (20, 52), (100, 52), 0.8, 2)
        cv.line(probability, (62, 20), (62, 85), 1.0, 2)
        prior = np.asarray([[50.0, x] for x in range(20, 101, 2)])
        refined, support, shift = refine_chain_from_prior(
            probability,
            prior,
            search_radius=4,
            root_lock_points=3,
        )
        self.assertTrue(np.allclose(refined[:3], prior[:3]))
        self.assertLessEqual(np.median(np.abs(refined[8:, 0] - 52.0)), 1.0)
        self.assertGreater(support, 0.6)
        self.assertGreater(shift, 0.0)
        self.assertTrue(np.all(np.diff(refined[:, 1]) > 0.0))
        self.assertLess(
            abs(curve_length(refined) - curve_length(prior)),
            0.05 * curve_length(prior),
        )


def _crossing_maps():
    """Create a new horizontal tube crossing an older vertical distractor."""
    historical = np.zeros((100, 120), dtype=np.float32)
    cv.line(historical, (54, 18), (54, 82), 1.0, 3)
    vessel = historical.copy()
    cv.line(vessel, (20, 50), (82, 50), 1.0, 3)
    novelty = vessel.copy()
    return historical, vessel, novelty


def test_future_edge_cue_suppresses_an_old_branch_but_retains_new_growth():
    """A crossing pixel may be shared without making the old branch a future edge."""
    historical, vessel, novelty = _crossing_maps()
    cue, occupied = predict_future_edge_cue(
        vessel,
        novelty,
        historical,
        TopologyAwareTraceConfig(historical_dilation_px=0),
    )
    assert occupied[35, 54]
    assert cue[35, 54] == 0.0
    assert cue[50, 72] > 0.9
    assert cue[50, 54] == 0.0


def test_future_edge_hypotheses_follow_the_new_edge_through_a_crossing():
    """The selected node must be joined by new horizontal edge pixels."""
    historical, vessel, novelty = _crossing_maps()
    evidence = PollenAnchoredEvidence(
        gray=np.zeros(vessel.shape, dtype=np.uint8),
        vessel_probability=vessel,
        novelty_probability=novelty,
        fused_probability=np.sqrt(vessel * novelty),
    )
    prior = np.asarray([[50.0, x] for x in range(29, 53, 2)])
    trace = replace(
        CurveTraceConfig(),
        min_path_probability=0.04,
        min_extension_temporal_score=0.04,
        max_tip_step=25.0,
        max_tip_lateral_step=24.0,
    )
    chain_config = replace(
        PollenAnchoredChainConfig(),
        crop_size=100,
        maximum_direct_extension_px=25.0,
        trace=trace,
    )
    hypotheses = future_edge_hypotheses(
        evidence,
        historical,
        prior,
        np.zeros(vessel.shape, dtype=bool),
        chain_config,
        TopologyAwareTraceConfig(
            historical_dilation_px=0,
            maximum_incremental_growth_px=25.0,
        ),
    )
    assert hypotheses
    endpoint = hypotheses[0].extension_yx[-1]
    assert endpoint[1] > 65.0
    assert abs(endpoint[0] - 50.0) < 4.0
    assert hypotheses[0].historical_overlap <= 0.30
    assert all(item.maximum_historical_run_px <= 4.0 for item in hypotheses)
    assert all(abs(item.extension_yx[-1, 0] - 50.0) < 4.0 for item in hypotheses)


def test_short_changed_continuation_can_reacquire_across_historical_pixels():
    """A target may advance over an older aligned tube when change persists."""
    shape = (80, 100)
    historical = np.zeros(shape, dtype=np.float32)
    cv.line(historical, (20, 40), (80, 40), 1.0, 3)
    vessel = historical.copy()
    novelty = historical.copy()
    evidence = PollenAnchoredEvidence(
        gray=np.zeros(shape, dtype=np.uint8),
        vessel_probability=vessel,
        novelty_probability=novelty,
        fused_probability=vessel,
    )
    prior = np.asarray([[40.0, x] for x in range(20, 53, 2)])
    trace = replace(
        CurveTraceConfig(),
        min_path_probability=0.04,
        min_extension_temporal_score=0.04,
        max_tip_step=10.0,
        max_tip_lateral_step=8.0,
    )
    hypotheses = future_edge_hypotheses(
        evidence,
        historical,
        prior,
        np.zeros(shape, dtype=bool),
        replace(PollenAnchoredChainConfig(), crop_size=80, trace=trace),
        TopologyAwareTraceConfig(historical_dilation_px=0),
    )
    assert hypotheses
    assert all(item.historical_reacquisition for item in hypotheses)
    assert all(item.added_length_px <= 8.0 for item in hypotheses)
    assert hypotheses[0].extension_yx[-1, 1] > prior[-1, 1]


def test_historical_reacquisition_rejects_stale_or_turning_branches():
    """Historical pixels need both new change and target-tangent agreement."""
    shape = (80, 100)
    historical = np.zeros(shape, dtype=np.float32)
    cv.line(historical, (20, 40), (80, 40), 1.0, 3)
    cv.line(historical, (52, 20), (52, 65), 1.0, 3)
    prior = np.asarray([[40.0, x] for x in range(20, 53, 2)])
    trace = replace(
        CurveTraceConfig(),
        min_path_probability=0.04,
        min_extension_temporal_score=0.04,
        max_tip_step=10.0,
        max_tip_lateral_step=10.0,
    )
    chain_config = replace(
        PollenAnchoredChainConfig(), crop_size=80, trace=trace
    )
    exclusion = np.zeros(shape, dtype=bool)
    stale = PollenAnchoredEvidence(
        gray=np.zeros(shape, dtype=np.uint8),
        vessel_probability=historical,
        novelty_probability=np.full(shape, 0.05, dtype=np.float32),
        fused_probability=historical,
    )
    assert not future_edge_hypotheses(
        stale,
        historical,
        prior,
        exclusion,
        chain_config,
        TopologyAwareTraceConfig(historical_dilation_px=0),
    )

    turning_only = np.zeros(shape, dtype=np.float32)
    cv.line(turning_only, (52, 40), (52, 65), 1.0, 3)
    turning = PollenAnchoredEvidence(
        gray=np.zeros(shape, dtype=np.uint8),
        vessel_probability=historical,
        novelty_probability=turning_only,
        fused_probability=historical,
    )
    assert not future_edge_hypotheses(
        turning,
        historical,
        prior,
        exclusion,
        chain_config,
        TopologyAwareTraceConfig(historical_dilation_px=0),
    )
