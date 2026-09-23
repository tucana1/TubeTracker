"""Tests for causal pollen-boundary emergence validation."""

import numpy as np
import cv2 as cv

from prototypes.v25_causal_portal.track import (
    PortalCandidate,
    _detect_directional_portal_emergence,
    _detect_rupture_like_transition,
    _owner_track_visibility,
    _path_returns_to_owner,
    _rescale_candidate_geometry,
    _scale_feature_config,
    truncate_discontinuous_growth,
)

from tubetracker.causal_portal import (
    CausalPortalConfig,
    PhaseContrastRibbonConfig,
    PollenOutlineConfig,
    PollenRingConfig,
    assess_pollen_ring,
    certify_native_path_quality,
    detect_pollen_outline_emergence,
    fit_native_centerline_offsets,
    measure_pollen_outline,
    measure_pollen_outline_extents,
    native_connected_prefix_growth,
    native_ribbon_score_surface,
    path_exits_owner_once,
    phase_contrast_ribbon_features,
    sample_native_paired_support,
    sample_native_ribbon_support,
    truncate_self_reentry,
    validate_causal_portal,
)


def _test_config() -> CausalPortalConfig:
    """Return compact thresholds suitable for deterministic synthetic tests."""

    return CausalPortalConfig(
        warmup_samples=3,
        tail_samples=3,
        smoothing_radius=0,
        confirmation_window=1,
        confirmation_required=1,
        absolute_support_floor=0.2,
        minimum_support_change=0.1,
        noise_multiplier=1.0,
        maximum_gap_points=1,
        initial_search_points=2,
        minimum_late_coverage=0.6,
        maximum_early_coverage=0.2,
        maximum_pre_onset_orphan_fraction=0.5,
        minimum_birth_order_fraction=0.8,
        minimum_emergence_length_px=1.0,
        onset_growth_window_samples=3,
        minimum_onset_growth_px=2.0,
        minimum_growth_px=3.0,
        minimum_final_length_px=3.0,
    )


def test_causal_portal_accepts_an_ordered_growing_prefix():
    """A connected path that appears root-first should be accepted."""

    evidence = np.full((10, 6), 0.05, dtype=np.float32)
    for sample in range(3, 10):
        evidence[sample, : min(6, sample - 1)] = 0.8
    result = validate_causal_portal(
        evidence,
        np.arange(6, dtype=np.float32),
        config=_test_config(),
    )
    assert result.accepted
    assert result.onset_sample == 3
    assert result.prefix_lengths_px[-1] == 5.0


def test_causal_portal_rejects_a_static_pollen_rim():
    """Strong evidence already present during warmup is not emergence."""

    evidence = np.full((10, 6), 0.8, dtype=np.float32)
    result = validate_causal_portal(
        evidence,
        np.arange(6, dtype=np.float32),
        config=_test_config(),
    )
    assert not result.accepted
    assert "preexisting-or-rim-like" in result.reason


def test_causal_portal_ignores_a_short_rim_flash_before_real_growth():
    """Germination starts at sustained extension, not the first short protrusion."""

    evidence = np.full((12, 6), 0.05, dtype=np.float32)
    evidence[3:, :2] = 0.8
    evidence[8:, :4] = 0.8
    evidence[9:, :6] = 0.8
    result = validate_causal_portal(
        evidence,
        np.arange(6, dtype=np.float32),
        config=_test_config(),
    )
    assert result.accepted
    assert result.onset_sample == 8


def test_causal_portal_rejects_distal_material_that_appears_first():
    """A crossing tube appearing distal-first must not become this owner's tube."""

    evidence = np.full((10, 6), 0.05, dtype=np.float32)
    evidence[3:, 3:] = 0.8
    evidence[7:, :3] = 0.8
    result = validate_causal_portal(
        evidence,
        np.arange(6, dtype=np.float32),
        config=_test_config(),
    )
    assert not result.accepted
    assert result.pre_onset_orphan_fraction > 0.3
    assert "noncausal-point-order" in result.reason


def test_owner_exit_rejects_a_path_that_returns_to_the_grain():
    """A trace may curve after departure but cannot loop back onto the owner rim."""

    center = np.array([0.0, 0.0])
    valid = np.array([[0, 1], [0, 2], [1, 3], [2, 4]], dtype=np.float32)
    returning = np.array([[0, 1], [0, 2], [1, 3], [0, 1.1]], dtype=np.float32)
    assert path_exits_owner_once(valid, center, 1.0)
    assert not path_exits_owner_once(returning, center, 1.0)


def test_open_curve_stops_before_doubling_back_onto_itself():
    """A distal loop is removed without shortening an ordinary curved path."""

    open_path = np.column_stack((np.zeros(12), np.arange(12))).astype(np.float32)
    looped = np.vstack((open_path, [[0, 10], [0, 9], [0, 8], [0, 7]])).astype(
        np.float32
    )
    np.testing.assert_array_equal(truncate_self_reentry(open_path), open_path)
    assert len(truncate_self_reentry(looped)) == 15


def test_phase_contrast_ribbon_peaks_between_dark_walls():
    """A bright tube interior wins over either dark boundary as centerline."""

    image = np.full((64, 96), 128, dtype=np.uint8)
    image[27:29, 12:84] = 55
    image[29:35, 12:84] = 205
    image[35:37, 12:84] = 55
    feature = phase_contrast_ribbon_features(
        image,
        PhaseContrastRibbonConfig(
            orientation_count=8,
            half_widths_px=(2.0, 3.0, 4.0, 5.0),
        ),
    )
    horizontal = feature.score[0, :, 48]
    peak = int(np.argmax(horizontal[24:40]) + 24)
    assert 30 <= peak <= 34
    assert horizontal[peak] > horizontal[28]
    assert horizontal[peak] > horizontal[36]


def test_phase_contrast_ribbon_supports_inverted_tube_polarity():
    """A dark tube between bright walls is centered in its own state layer."""

    image = np.full((64, 96), 128, dtype=np.uint8)
    image[27:29, 12:84] = 210
    image[29:35, 12:84] = 50
    image[35:37, 12:84] = 210
    feature = phase_contrast_ribbon_features(
        image,
        PhaseContrastRibbonConfig(
            interior_polarity="dark",
            orientation_count=8,
            half_widths_px=(2.0, 3.0, 4.0, 5.0),
        ),
    )
    horizontal = feature.score[0, :, 48]
    peak = int(np.argmax(horizontal[24:40]) + 24)
    assert 30 <= peak <= 34
    assert horizontal[peak] > horizontal[28]
    assert horizontal[peak] > horizontal[36]


def test_native_outline_detects_a_persistent_connected_protrusion():
    """A growing dark wall extending from a pollen ring marks germination."""

    config = PollenOutlineConfig(
        pollen_radius_px=15.0,
        darkness_thresholds=(6.0, 8.0, 10.0),
        warmup_samples=3,
        minimum_extension_px=1.5,
        confirmation_window=3,
        confirmation_required=2,
    )
    extents = []
    for sample in range(9):
        image = np.full((96, 96), 128, dtype=np.uint8)
        cv.circle(image, (48, 48), 14, 55, 2)
        if sample >= 4:
            cv.line(image, (48, 62), (48, 70 + sample - 4), 55, 2)
        extents.append(measure_pollen_outline_extents(image, config))
    result = detect_pollen_outline_emergence(np.asarray(extents), config=config)
    assert result.onset_sample == 4
    assert result.ensemble_extension_px[-1] > 4.0
    observation = measure_pollen_outline(image, config)
    assert observation.tips_yx[:, 0].mean() > 15.0


def test_native_outline_rejects_a_single_frame_boundary_artifact():
    """One transient edge spike cannot become a germination event."""

    config = PollenOutlineConfig(
        pollen_radius_px=15.0,
        warmup_samples=3,
        confirmation_window=3,
        confirmation_required=2,
    )
    extents = np.full((9, 3), 15.0, dtype=np.float32)
    extents[4] = 20.0
    result = detect_pollen_outline_emergence(extents, config=config)
    assert result.onset_sample is None


def test_native_pollen_ring_rejects_a_rounded_tube_tip():
    """A dark tip blob cannot pass as a closed pollen grain."""

    pollen = np.full((64, 64), 170, dtype=np.uint8)
    cv.circle(pollen, (32, 32), 11, 115, 3)
    tip = np.full((64, 64), 170, dtype=np.uint8)
    cv.ellipse(tip, (32, 38), (7, 14), 0, 0, 360, 105, -1)
    config = PollenRingConfig(pollen_radius_px=15.0)
    assert assess_pollen_ring(pollen, config).accepted
    assert not assess_pollen_ring(tip, config).accepted


def test_native_support_prefers_a_paired_wall_centerline_to_blank_controls():
    """Native evidence rewards a tube centerline rather than nearby background."""

    image = np.full((128, 128), 128, dtype=np.uint8)
    image[59:61, 20:108] = 55
    image[61:67, 20:108] = 205
    image[67:69, 20:108] = 55
    path = np.column_stack((np.full(80, 64.0), np.arange(24.0, 104.0)))
    support, controls = sample_native_paired_support(image, path)
    assert np.median(support) > 20.0
    assert np.median(support) > np.median(controls) + 15.0


def test_native_ribbon_surface_centers_a_complete_cross_section():
    """Two tube walls and recovered exterior identify their shared center."""

    image = np.full((96, 128), 128, dtype=np.uint8)
    image[41:43, 16:112] = 55
    image[43:51, 16:112] = 205
    image[51:53, 16:112] = 55
    path = np.column_stack((np.full(80, 47.0), np.arange(24.0, 104.0)))
    offsets = np.arange(-6.0, 7.0)
    surface = native_ribbon_score_surface(image, path, offsets)
    best = offsets[np.argmax(np.median(surface, axis=0))]
    assert abs(best) <= 1.0


def test_strict_native_ribbon_support_requires_a_bright_paired_lumen():
    """Blank shading and a single wall cannot mimic a full tube cross-section."""

    image = np.full((96, 128), 128, dtype=np.uint8)
    image[41:43, 16:112] = 55
    image[43:51, 16:112] = 205
    image[51:53, 16:112] = 55
    path = np.column_stack((np.full(80, 47.0), np.arange(24.0, 104.0)))
    support, controls = sample_native_ribbon_support(image, path)
    assert np.median(support) > np.median(controls) + 20.0

    single_wall = np.full_like(image, 128)
    single_wall[41:43, 16:112] = 55
    false_support, _ = sample_native_ribbon_support(single_wall, path)
    assert np.median(false_support) < 0.25 * np.median(support)


def test_native_offset_fit_recovers_smooth_shift_and_locks_root():
    """Repeated support can move a material path without moving its attachment."""

    offsets = np.arange(-5.0, 6.0)
    truth = np.concatenate((np.zeros(2), np.linspace(0.0, 3.0, 28)))
    surfaces = []
    for time in range(8):
        noise = np.random.default_rng(time).normal(0.0, 0.03, (30, len(offsets)))
        surfaces.append(1.0 - 0.18 * (offsets[None] - truth[:, None]) ** 2 + noise)
    result = fit_native_centerline_offsets(np.asarray(surfaces), offsets)
    assert result.accepted
    np.testing.assert_array_equal(result.offsets_px[:2], 0.0)
    assert np.mean(np.abs(result.offsets_px[5:] - truth[5:])) < 0.8
    assert np.max(np.abs(np.diff(result.offsets_px))) <= 2.0


def test_native_offset_fit_withholds_transient_or_boundary_lures():
    """A brief crossing and an optimum outside the corridor are not refinements."""

    offsets = np.arange(-4.0, 5.0)
    stable = np.ones((8, 24, len(offsets)), dtype=np.float32)
    stable -= 0.15 * offsets[None, None] ** 2
    transient = stable.copy()
    transient[4, 8:18] += 2.0 * np.exp(
        -0.5 * ((offsets[None] - 3.0) / 0.5) ** 2
    )
    assert not fit_native_centerline_offsets(transient, offsets).accepted

    boundary = np.ones_like(stable)
    boundary += 0.20 * offsets[None, None]
    result = fit_native_centerline_offsets(boundary, offsets)
    assert not result.accepted
    assert result.reason == "native-optimum-reaches-search-boundary"


def test_native_path_quality_requires_repeated_path_coverage():
    """A supported lumen passes while a mostly unsupported curve is withheld."""

    thresholds = np.full(5, 2.0)
    supported = np.full((5, 20), 4.0)
    supported[:, 16:] = 0.5
    accepted = certify_native_path_quality(
        supported,
        thresholds,
        completion_fraction=0.90,
    )
    weak = np.full((5, 20), 0.5)
    weak[:, :6] = 4.0
    rejected = certify_native_path_quality(
        weak,
        thresholds,
        completion_fraction=0.90,
    )
    assert accepted.accepted
    assert accepted.median_coverage == 0.8
    assert not rejected.accepted
    assert rejected.reason == "insufficient-native-path-coverage"


def test_native_path_quality_requires_distal_support():
    """Root-concentrated halo support cannot certify a full path."""

    thresholds = np.full(8, 2.0)
    proximal_only = np.full((8, 20), 0.5)
    proximal_only[:, :12] = 4.0
    rejected = certify_native_path_quality(
        proximal_only,
        thresholds,
        completion_fraction=0.90,
        minimum_median_coverage=0.50,
        minimum_distal_support_fraction=0.40,
    )
    assert not rejected.accepted
    assert rejected.reason == "insufficient-distal-path-coverage"
    supported = np.full((8, 20), 4.0)
    supported[:, 14:] = 0.5
    accepted = certify_native_path_quality(
        supported,
        thresholds,
        completion_fraction=0.90,
        minimum_median_coverage=0.50,
        minimum_distal_support_fraction=0.40,
    )
    assert accepted.accepted


def test_native_path_quality_requires_emergence_gain():
    """A pre-existing structure scores at warmup and stays diagnostic."""

    thresholds = np.full(8, 2.0)
    preexisting = np.full((8, 20), 6.0)
    rejected = certify_native_path_quality(
        preexisting,
        thresholds,
        completion_fraction=1.0,
        minimum_median_coverage=0.50,
        minimum_distal_support_fraction=0.40,
        warmup_support=np.full(20, 6.0),
        minimum_emergence_gain=0.30,
    )
    assert not rejected.accepted
    assert rejected.reason == "insufficient-emergence-gain"
    emerging = np.full((8, 20), 6.0)
    accepted = certify_native_path_quality(
        emerging,
        thresholds,
        completion_fraction=1.0,
        minimum_median_coverage=0.50,
        minimum_distal_support_fraction=0.40,
        warmup_support=np.full(20, 1.0),
        minimum_emergence_gain=0.30,
    )
    assert accepted.accepted


def test_native_path_quality_requires_repetition_and_completion():
    """One late frame or a short verified fragment cannot certify a full path."""

    support = np.full((2, 20), 4.0)
    threshold = np.full(2, 2.0)
    sparse = certify_native_path_quality(
        support,
        threshold,
        completion_fraction=0.95,
    )
    partial = certify_native_path_quality(
        np.vstack((support, support[:1])),
        np.full(3, 2.0),
        completion_fraction=0.60,
    )
    assert sparse.reason == "insufficient-native-quality-observations"
    assert partial.reason == "insufficient-native-quality-completion"


def test_directional_portal_requires_persistent_angle_continuity():
    """A lasting portal is detected while an isolated angular flash is ignored."""

    profiles = np.zeros((30, 72), dtype=np.float32)
    profiles[10:, 20] = 5.0
    visible = np.ones(30, dtype=bool)
    onset, dormant, _ = _detect_directional_portal_emergence(
        profiles,
        visible,
        observable_start_sample=0,
        root_angle_radians=20 * 2.0 * np.pi / 72,
    )
    assert onset == 10
    assert dormant == 9

    profiles[:] = 0.0
    profiles[10, 20] = 5.0
    onset, dormant, _ = _detect_directional_portal_emergence(
        profiles,
        visible,
        observable_start_sample=0,
        root_angle_radians=20 * 2.0 * np.pi / 72,
    )
    assert onset is None
    assert dormant is None

    profiles[10:, 20] = 5.0
    visible[9:] = False
    onset, dormant, _ = _detect_directional_portal_emergence(
        profiles,
        visible,
        observable_start_sample=0,
        root_angle_radians=20 * 2.0 * np.pi / 72,
    )
    assert onset is None
    assert dormant is None


def test_owner_closure_gate_allows_a_u_shaped_open_tube():
    """Curvature alone is not a loop when the distal endpoint stays away."""

    angles = np.linspace(0.0, np.pi, 30)
    open_u = np.column_stack((8.0 * np.sin(angles), 6.0 + 18.0 * angles))
    closed_angles = np.linspace(0.0, 2.0 * np.pi, 60)
    closed = np.column_stack(
        (7.5 * np.sin(closed_angles), 7.5 * np.cos(closed_angles))
    )
    center = np.zeros(2, dtype=np.float32)
    assert not _path_returns_to_owner(open_u, center, 5.0)
    assert _path_returns_to_owner(closed, center, 5.0)


def test_native_prefix_recovery_requires_sustained_connected_growth():
    """Native recovery ignores flashes and retains a monotone connected front."""

    support = np.zeros((30, 20), dtype=np.float32)
    support[5, :18] = 5.0
    support[10:, :15] = 5.0
    growth, ends, onset, dormant = native_connected_prefix_growth(
        support,
        np.ones(30, dtype=np.float32),
        np.arange(20, dtype=np.float32),
    )
    assert onset == 11
    assert dormant == 10
    assert growth[-1] == 14.0
    assert ends[-1] == 14
    assert np.all(np.diff(growth) >= 0.0)


def test_native_prefix_recovery_supports_a_root_local_emergence_threshold():
    """A small persistent root extension can be audited before mature growth."""

    support = np.zeros((24, 12), dtype=np.float32)
    support[10:, :6] = 5.0
    _, _, onset, dormant = native_connected_prefix_growth(
        support,
        np.ones(24, dtype=np.float32),
        np.arange(12, dtype=np.float32),
        minimum_strong_growth_px=3.0,
        minimum_weak_growth_px=2.0,
    )
    assert onset == 11
    assert dormant == 10


def test_native_prefix_recovery_caps_a_late_visibility_jump():
    """Newly visible distal material still advances at a physical rate."""

    support = np.zeros((18, 30), dtype=np.float32)
    support[8:, :8] = 5.0
    support[13:, :] = 5.0
    growth, _, _, _ = native_connected_prefix_growth(
        support,
        np.ones(18, dtype=np.float32),
        np.arange(30, dtype=np.float32),
        minimum_strong_growth_px=3.0,
        minimum_weak_growth_px=2.0,
        maximum_growth_px_per_sample=4.0,
    )

    assert np.max(np.diff(growth)) <= 4.0
    assert growth[-1] > growth[13]


def test_multiscale_feature_config_preserves_physical_widths():
    """The recovery pass doubles pixel scales without changing score weights."""

    config = PhaseContrastRibbonConfig(
        interior_polarity="dark",
        half_widths_px=(1.0, 2.5),
        tangent_samples_px=(-2.0, 0.0, 2.0),
        fine_sigma_px=0.6,
        background_sigma_px=3.5,
        structure_sigma_px=1.1,
        matching_center_weight=0.7,
    )
    scaled = _scale_feature_config(config, 2.0)
    assert scaled.half_widths_px == (2.0, 5.0)
    assert scaled.tangent_samples_px == (-4.0, 0.0, 4.0)
    assert scaled.fine_sigma_px == 1.2
    assert scaled.background_sigma_px == 7.0
    assert scaled.matching_center_weight == config.matching_center_weight


def test_multiscale_candidate_returns_to_analysis_coordinates():
    """High-resolution recovery retains the same physical path after rescaling."""

    candidate = PortalCandidate(
        owner_index=0,
        owner_track_id=1,
        observable_start_sample=0,
        appearance_mode="dark",
        local_center_yx=np.array([160.0, 160.0], dtype=np.float32),
        local_path_yx=np.array(
            [[160.0, 172.0], [164.0, 180.0], [170.0, 190.0]],
            dtype=np.float32,
        ),
        direction_bins=np.zeros(3, dtype=np.int32),
        normal_yx=np.zeros((3, 2), dtype=np.float32),
        geometric_score=1.0,
        paired_fraction=0.8,
        radial_extension_px=30.0,
        evidence_by_mode={"dark": np.zeros((2, 3), dtype=np.float32)},
        normal_offsets_by_mode={"dark": np.zeros((2, 3), dtype=np.int8)},
    )
    _rescale_candidate_geometry(candidate, scale=2.0, destination_crop_size=160)
    np.testing.assert_allclose(candidate.local_center_yx, [80.0, 80.0])
    np.testing.assert_allclose(
        candidate.local_path_yx,
        [[80.0, 86.0], [82.0, 90.0], [85.0, 95.0]],
    )
    assert candidate.radial_extension_px == 15.0


def test_owner_visibility_uses_relative_template_continuity():
    """A low-contrast owner stays visible until its own template score collapses."""

    scores = np.array([0.0, 0.72, 0.70, 0.74, 0.69, 0.42, 0.34, 0.20])
    visible = _owner_track_visibility(scores, observable_start_sample=1)
    np.testing.assert_array_equal(
        visible,
        [False, True, True, True, True, True, False, False],
    )


def test_discontinuous_growth_freezes_the_last_supported_prefix():
    """A large late branch jump preserves earlier length without claiming the branch."""

    candidate = PortalCandidate(
        owner_index=0,
        owner_track_id=1,
        observable_start_sample=0,
        appearance_mode="dark",
        local_center_yx=np.array([80.0, 80.0], dtype=np.float32),
        local_path_yx=np.column_stack(
            (np.full(6, 80.0), np.arange(86.0, 92.0))
        ).astype(np.float32),
        direction_bins=np.zeros(6, dtype=np.int32),
        normal_yx=np.zeros((6, 2), dtype=np.float32),
        geometric_score=1.0,
        paired_fraction=0.8,
        radial_extension_px=20.0,
        evidence_by_mode={"dark": np.zeros((6, 6), dtype=np.float32)},
        normal_offsets_by_mode={"dark": np.zeros((6, 6), dtype=np.int8)},
        accepted=True,
    )
    candidate.validation = validate_causal_portal(
        np.ones((6, 6), dtype=np.float32),
        np.arange(6, dtype=np.float32),
        config=_test_config(),
    )
    candidate.native_prefix_lengths_px = np.array(
        [0.0, 5.0, 12.0, 80.0, 86.0, 90.0], dtype=np.float32
    )
    candidate.native_prefix_end_indices = np.array(
        [-1, 0, 1, 4, 4, 5], dtype=np.int32
    )
    truncate_discontinuous_growth([candidate], analysis_scale=0.5)
    np.testing.assert_array_equal(
        candidate.native_prefix_lengths_px,
        [0.0, 5.0, 12.0, 12.0, 12.0, 12.0],
    )
    assert candidate.growth_discontinuity_sample == 3
    assert candidate.measurement_scope == "pre-discontinuity"

    candidate.native_prefix_lengths_px = None
    candidate.native_prefix_end_indices = None
    candidate.validation.prefix_lengths_px[:] = [0.0, 2.5, 6.0, 40.0, 43.0, 45.0]
    candidate.validation.prefix_end_indices[:] = [-1, 0, 1, 4, 4, 5]
    candidate.growth_discontinuity_sample = None
    candidate.measurement_scope = "full"
    truncate_discontinuous_growth([candidate], analysis_scale=0.5)
    np.testing.assert_array_equal(
        candidate.validation.prefix_lengths_px,
        [0.0, 2.5, 6.0, 6.0, 6.0, 6.0],
    )


def test_rupture_transition_requires_owner_collapse_and_rapid_short_event():
    """A short sudden mass after identity collapse is separate from smooth emergence."""

    scores = np.array([0.95, 0.96, 0.94, 0.95, 0.94, 0.40, 0.42, 0.70, 0.80])
    lengths = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 28.0])
    assert (
        _detect_rupture_like_transition(
            scores,
            lengths,
            observable_start_sample=0,
            onset_sample=7,
        )
        == 5
    )
    assert (
        _detect_rupture_like_transition(
            np.full(9, 0.95),
            lengths,
            observable_start_sample=0,
            onset_sample=7,
        )
        is None
    )


def test_native_path_quality_rejects_latched_distal_without_proximal():
    """A path that only arrives far away is not an emergence (dense P76)."""

    thresholds = np.full(8, 2.0)
    latched = np.full((8, 21), 0.5)
    latched[:, 14:] = 24.0  # strong stolen distal, empty background at root
    rejected = certify_native_path_quality(
        latched,
        thresholds,
        completion_fraction=1.0,
        minimum_median_coverage=0.30,
        minimum_proximal_support_ratio=0.25,
    )
    assert not rejected.accepted
    assert rejected.reason == "insufficient-proximal-path-support"
    assert rejected.proximal_support_ratio < 0.25


def test_native_path_quality_keeps_uniform_faint_emergence():
    """A faint-but-uniform tube passes the proximal ratio (dense P3)."""

    thresholds = np.full(8, 2.0)
    faint = np.full((8, 21), 4.6)
    faint[:, 14:] = 7.8  # proximal weaker than distal, same order
    accepted = certify_native_path_quality(
        faint,
        thresholds,
        completion_fraction=1.0,
        minimum_median_coverage=0.30,
        minimum_proximal_support_ratio=0.25,
    )
    assert accepted.accepted
    assert accepted.proximal_support_ratio >= 0.25


def test_native_path_quality_proximal_gate_defaults_off_and_skips_stubs():
    """Default behavior is unchanged; short stubs skip the latch test."""

    thresholds = np.full(8, 2.0)
    latched = np.full((8, 21), 0.5)
    latched[:, 14:] = 24.0
    assert certify_native_path_quality(
        latched, thresholds, completion_fraction=1.0,
        minimum_median_coverage=0.30,
    ).accepted
    stub = np.full((8, 6), 0.5)
    stub[:, 4:] = 24.0
    skipped = certify_native_path_quality(
        stub, thresholds, completion_fraction=1.0,
        minimum_median_coverage=0.30,
        minimum_proximal_support_ratio=0.25,
    )
    assert skipped.accepted
    assert not np.isfinite(skipped.proximal_support_ratio)
