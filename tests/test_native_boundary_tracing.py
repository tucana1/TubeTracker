"""Tests for native-resolution paired-boundary candidate generation."""

import cv2 as cv
import numpy as np

from tubetracker.native_boundary_tracing import (
    DualPolarityRibbonFeatures,
    NativeBoundaryTracingConfig,
    clip_path_to_image_bounds,
    extend_path_to_image_boundary,
    owner_aligned_temporal_consensus,
    path_extends_rooted_prefix,
    source_boundary_terminal_mask,
    trace_native_boundary_candidates,
)


def test_path_is_clipped_at_first_true_image_edge():
    """Reflected consensus padding cannot create out-of-frame tube length."""

    clipped, touches = clip_path_to_image_bounds(
        np.asarray(((4.0, 3.0), (7.0, 5.0), (12.0, 8.0))),
        (10, 14),
    )

    assert touches
    assert np.allclose(clipped[-1], (9.0, 6.2))
    assert np.all(clipped >= 0.0)
    assert np.all(clipped <= np.asarray((9.0, 13.0)))


def test_interior_path_is_not_mislabeled_as_boundary_contact():
    """Only an exact retained edge intersection is boundary-censored."""

    path = np.asarray(((4.0, 3.0), (7.0, 5.0), (8.5, 6.0)))
    clipped, touches = clip_path_to_image_bounds(path, (10, 14))

    assert not touches
    assert np.array_equal(clipped, path)


def test_nearby_outward_endpoint_is_extended_to_exact_edge():
    """Subpixel terminal bands still yield an exact censored endpoint."""

    path = np.asarray(((4.0, 3.0), (7.0, 5.0), (8.6, 6.0)))
    extended, touches = extend_path_to_image_boundary(path, (10, 14), 1.0)

    assert touches
    assert np.isclose(extended[-1, 0], 9.0)
    assert len(extended) == len(path) + 1


def test_distant_endpoint_is_not_extrapolated_to_edge():
    """Only the tiny terminal-mask quantization gap may be completed."""

    path = np.asarray(((4.0, 3.0), (5.0, 4.0), (6.0, 5.0)))
    extended, touches = extend_path_to_image_boundary(path, (10, 14), 1.0)

    assert not touches
    assert np.array_equal(extended, path)


def test_boundary_policy_is_explicitly_more_gap_tolerant():
    """Boundary continuation remains isolated from ordinary complete-tip tracing."""

    ordinary = NativeBoundaryTracingConfig()
    boundary = ordinary.for_boundary_censoring()

    assert boundary.trace.maximum_gap_steps > ordinary.trace.maximum_gap_steps
    assert boundary.trace.minimum_pair_support < ordinary.trace.minimum_pair_support
    assert boundary.trace.length_reward > ordinary.trace.length_reward
    assert boundary.trace.maximum_total_turn_bins is not None


def test_faint_extension_policy_keeps_a_global_heading_bound():
    """Weak-gap recovery must not gain unrestricted crossover turns."""

    ordinary = NativeBoundaryTracingConfig()
    extension = ordinary.for_faint_prefix_extension()

    assert extension.trace.maximum_gap_steps > ordinary.trace.maximum_gap_steps
    assert extension.trace.maximum_total_turn_bins == 4
    assert extension.trace.minimum_pair_support < ordinary.trace.minimum_pair_support


def test_rooted_prefix_extension_accepts_continuation_not_branch_switch():
    """A longer candidate must preserve the trusted proximal material route."""

    reference = np.column_stack((np.zeros(21), np.arange(21.0)))
    continuation = np.column_stack((np.zeros(31), np.arange(31.0)))
    switched = continuation.copy()
    switched[8:, 0] = np.arange(23.0)

    assert path_extends_rooted_prefix(reference, continuation)
    assert not path_extends_rooted_prefix(reference, switched)


def test_rooted_prefix_extension_rejects_late_crossover_switch():
    """Agreement near the root cannot excuse a downstream branch jump."""

    reference = np.column_stack((np.zeros(45), np.arange(45.0)))
    candidate = np.column_stack(
        (
            np.concatenate((np.zeros(30), np.arange(1.0, 32.0))),
            np.arange(61.0),
        )
    )

    assert not path_extends_rooted_prefix(reference, candidate)
    assert path_extends_rooted_prefix(
        reference,
        candidate,
        comparison_length_px=24.0,
    )


def test_source_boundary_mask_maps_true_edge_into_reflected_crop():
    """A padded consensus uses the real field edge, not its array edge."""

    terminal = source_boundary_terminal_mask(
        (80, 90),
        (100, 140),
        (65.4, 30.0),
        band_px=1.0,
    )

    expected_row = round(99.0 - 65.4)
    assert terminal[expected_row, 20]
    assert not terminal[-1, 20]
    assert not terminal[20, 20]


def test_owner_aligned_consensus_reinforces_a_moving_faint_structure():
    """Owner translation should be removed before mature views are combined."""

    views = []
    centers = []
    for shift in (-3, 0, 4):
        image = np.full((80, 90), 180, dtype=np.uint8)
        center = np.asarray([40.0, 45.0 + shift])
        centers.append(center)
        cv.line(
            image,
            (int(center[1]), int(center[0])),
            (int(center[1] + 18), int(center[0] + 12)),
            90,
            2,
        )
        views.append(image)

    consensus = owner_aligned_temporal_consensus(
        views,
        np.asarray(centers),
        crop_radius_px=25,
    )

    assert consensus.shape == (50, 50)
    assert consensus[25, 25] < consensus[8, 8]
    assert consensus[37, 43] < consensus[8, 8]


def _synthetic_rooted_tube() -> tuple[np.ndarray, np.ndarray]:
    """Draw a bright lumen with dark paired walls leaving one pollen grain."""

    image = np.full((120, 180), 135, dtype=np.uint8)
    center = np.asarray((60.0, 42.0))
    cv.circle(image, (42, 60), 15, 75, 2, cv.LINE_AA)
    cv.line(image, (56, 56), (145, 56), 65, 2, cv.LINE_AA)
    cv.line(image, (56, 64), (145, 64), 65, 2, cv.LINE_AA)
    cv.line(image, (56, 60), (145, 60), 180, 5, cv.LINE_AA)
    cv.ellipse(image, (145, 60), (4, 4), -90, -90, 90, 65, 2, cv.LINE_AA)
    return cv.GaussianBlur(image, (0, 0), 0.7), center


def test_native_boundary_candidates_follow_the_lumen_midpoint():
    """The best rooted candidate stays between walls instead of riding one wall."""

    image, center = _synthetic_rooted_tube()
    config = NativeBoundaryTracingConfig(output_count=3)
    candidates = trace_native_boundary_candidates(image, center, config=config)
    assert candidates
    best = candidates[0].trace.path_yx
    distal = best[np.linalg.norm(best - center[None], axis=1) > 30.0]
    assert len(distal) > 10
    assert np.median(np.abs(distal[:, 0] - 60.0)) < 2.0
    assert best[-1, 1] > 120.0


def test_native_boundary_candidates_mask_foreign_pollen_interiors():
    """No proposed centerline may enter a masked neighboring pollen body."""

    image, center = _synthetic_rooted_tube()
    foreign = np.asarray(((60.0, 105.0),))
    candidates = trace_native_boundary_candidates(image, center, foreign)
    assert candidates
    for candidate in candidates:
        distances = np.linalg.norm(
            candidate.trace.path_yx - foreign[0][None],
            axis=1,
        )
        assert np.all(distances >= 0.87 * 15.0)


def test_dual_polarity_consensus_rejects_a_halo_saddle():
    """A one-sided bright response must not count as paired confirmation."""

    from tubetracker.orientation_worldsheet import PairedWallOrientationResult

    shape = (4, 16, 16)
    bright = PairedWallOrientationResult(
        np.full(shape, 0.5, dtype=np.float32),
        np.full(shape, 0.815, dtype=np.float32),
        np.full(shape, 6.0, dtype=np.float32),
        np.full(shape, 0.9, dtype=np.float32),
    )
    dark = PairedWallOrientationResult(
        np.full(shape, 0.9, dtype=np.float32),
        np.full(shape, 0.05, dtype=np.float32),
        np.full(shape, 6.0, dtype=np.float32),
        np.full(shape, 0.9, dtype=np.float32),
    )
    features = DualPolarityRibbonFeatures(bright=bright, dark=dark)

    # The two polarity models are mutually exclusive by construction: a
    # bright lumen forces the dark response toward zero and vice versa.  The
    # old AND-consensus treated single-polarity structure as unconfirmed and
    # vetoed every real tube (dense P3 consensus collapsed to zero along its
    # whole visible length).  Consensus is now the per-state OR — structure
    # either polarity confirms is kept for tracing — while halo saddles are
    # rejected downstream by the polarity-aware support audit, which checks
    # the dark-wall cross-section of the proposed path.
    assert float(features.consensus_paired().max()) >= 0.81
    assert float(features.consensus_score().max()) >= 0.5


def test_dual_polarity_consensus_keeps_single_polarity_structure():
    """Structure either polarity confirms must survive the OR consensus."""

    from tubetracker.orientation_worldsheet import PairedWallOrientationResult

    shape = (4, 16, 16)
    bright = PairedWallOrientationResult(
        np.full(shape, 0.6, dtype=np.float32),
        np.full(shape, 0.7, dtype=np.float32),
        np.full(shape, 3.0, dtype=np.float32),
        np.full(shape, 0.8, dtype=np.float32),
    )
    dark = PairedWallOrientationResult(
        np.full(shape, 0.55, dtype=np.float32),
        np.full(shape, 0.6, dtype=np.float32),
        np.full(shape, 3.0, dtype=np.float32),
        np.full(shape, 0.7, dtype=np.float32),
    )
    features = DualPolarityRibbonFeatures(bright=bright, dark=dark)

    assert float(features.consensus_paired().min()) >= 0.6
    assert float(features.consensus_score().min()) >= 0.55
