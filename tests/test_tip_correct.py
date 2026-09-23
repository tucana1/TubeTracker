"""Unit tests for forward-curve tip correction (H180)."""

import numpy as np


def _bar_frame():
    import cv2

    g = np.full((100, 100), 160, dtype=np.uint8)
    g[20:81, 47:49] = 60
    g[20:81, 52:54] = 60
    return g


def test_correct_tip_snaps_ahead_to_apex_peak():
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = _bar_frame()
    tail = np.array([[50.0, 80.0], [50.0, 70.0], [50.0, 60.0], [50.0, 52.0]])
    out = correct_tip(g, tail, [(50.0, 40.0, 0.9), (10.0, 10.0, 0.9)])
    assert out is not None
    assert abs(out[0] - 50.0) < 3.0 and abs(out[1] - 40.0) < 5.0


def test_correct_tip_refuses_background():
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = np.full((100, 100), 160, dtype=np.uint8)
    tail = np.array([[20.0, 80.0], [20.0, 70.0], [20.0, 60.0], [20.0, 52.0]])
    assert correct_tip(g, tail, [(80.0, 20.0, 0.9)]) is None
    assert correct_tip(g, tail, []) is None
    assert correct_tip(g, tail[:2], [(20.0, 40.0, 0.9)]) is None


def test_correct_tip_confirms_against_tip_on_short_tail():
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = _bar_frame()
    # Near-stationary tail (jitter < MIN_FIT_ARCLEN): confirm mode.
    tail = np.array([[50.0, 52.4], [50.2, 52.0], [49.8, 52.2], [50.0, 52.0]])
    out = correct_tip(g, tail, [(56.0, 48.0, 0.7)])
    assert out is not None
    assert correct_tip(g, tail, [(90.0, 90.0, 0.9)]) is None


def test_confirm_grains_keeps_persistent_drops_flicker():
    from prototypes.timesfm_tip_forecast.grain_detect import confirm_grains

    stable = [(100.0, 100.0, 0.9), (200.0, 200.0, 0.8)]
    shifted = [(102.0, 99.0, 0.7), (205.0, 203.0, 0.6)]
    flicker = [(101.0, 101.0, 0.85), (500.0, 500.0, 0.9)]
    out = confirm_grains([stable, shifted, flicker])
    assert len(out) == 2
    assert out[0][2] == 0.9  # max score kept
    assert confirm_grains([]) == []
    assert confirm_grains([[(1.0, 1.0, 0.5)]]) == []


def test_angular_coverage_disk_full_arc_partial():
    import cv2
    from prototypes.timesfm_tip_forecast.grain_detect import angular_coverage

    g = np.full((100, 100), 160, dtype=np.uint8)
    cv2.circle(g, (30, 50), 15, 60, 2)  # full dark ring
    cv2.ellipse(g, (75, 50), (15, 15), 0, 0, 120, 60, 2)  # arc only
    cov = angular_coverage(g, [(30.0, 50.0), (75.0, 50.0)])
    assert cov[0] >= 0.8, cov
    assert cov[1] < 0.6, cov


def test_peak_radii_recovers_disk_radius():
    import cv2
    from prototypes.timesfm_tip_forecast.grain_detect import (
        grain_symmetry_maps,
        peak_radii,
    )

    g = np.full((100, 100), 160, dtype=np.uint8)
    cv2.circle(g, (50, 50), 15, 60, 2)
    maps = grain_symmetry_maps(g, radii=(11, 14, 17, 20))
    (r,) = peak_radii(maps, [(50.0, 50.0)])
    assert 12.0 <= r <= 18.0, r


def test_split_mergers_splits_pair_keeps_single():
    import cv2
    from prototypes.timesfm_tip_forecast.grain_detect import split_mergers

    g = np.full((120, 120), 160, dtype=np.uint8)
    cv2.circle(g, (40, 60), 13, 60, 2)  # pair 16px apart
    cv2.circle(g, (56, 60), 13, 60, 2)
    cv2.circle(g, (95, 60), 13, 60, 2)  # lone grain
    out = split_mergers(g, [(48.0, 60.0, 0.9), (95.0, 60.0, 0.9)])
    near_pair = [d for d in out if abs(d[1] - 60) < 12 and d[0] < 75]
    near_lone = [d for d in out if abs(d[1] - 60) < 12 and d[0] >= 75]
    assert len(near_pair) >= 2, out
    assert len(near_lone) == 1, out


def test_consensus_smooth_vetoes_flicker_keeps_trend():
    from prototypes.timesfm_tip_forecast.tip_correct import consensus_smooth

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 50.0], [3.0, 0.0], [4.0, 0.0]])
    keep = consensus_smooth(pts, np.ones(5, dtype=bool))
    assert keep.tolist() == [True, True, False, True, True]
    assert consensus_smooth(pts, np.zeros(5, dtype=bool)).tolist() == [False] * 5


def test_extrapolation_rounds_bend_to_apex():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = np.full((120, 120), 160, dtype=np.uint8)
    g[20:79, 47:49] = 60  # vertical walls to the bend
    g[20:79, 52:54] = 60
    g[74:76, 47:95] = 60  # horizontal walls after the bend
    g[78:80, 47:95] = 60
    tail = np.array([
        [50.0, 44.0], [50.5, 50.0], [51.0, 56.0], [52.0, 62.0],
        [54.0, 67.0], [57.0, 71.0], [61.0, 74.0], [65.0, 76.5],
    ])
    out = correct_tip(g, tail, [(61.0, 75.0, 0.9), (88.0, 79.0, 0.9)])
    assert out is not None
    assert abs(out[0] - 88.0) < 6.0, out  # apex, not the corner


def test_silent_fallback_returns_live_curve_end():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = np.full((100, 100), 160, dtype=np.uint8)
    g[20:81, 47:49] = 60
    g[20:81, 52:54] = 60
    tail = np.array([[50.0, 70.0], [50.0, 60.0], [50.0, 50.0], [50.0, 40.0],
                     [50.0, 30.0]])
    out = correct_tip(g, tail, [])  # CNN silent: no peaks at all
    assert out is not None
    assert abs(out[0] - 50.0) < 4.0 and 14.0 <= out[1] <= 26.0, out


def test_silent_fallback_refuses_flicker_scale():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = np.full((200, 200), 160, dtype=np.uint8)
    g[20:181, 47:49] = 60
    g[20:181, 52:54] = 60
    tail = np.array([[50.0, 120.0], [50.0, 100.0], [50.0, 80.0], [50.0, 60.0]])
    assert correct_tip(g, tail, []) is None  # end 40px out: capped


def test_refine_subpix_locks_corner_stays_bounded():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import refine_subpix

    g = np.full((60, 60), 160, dtype=np.uint8)
    g[10:, 28:30] = 40
    g[28:30, 10:] = 40  # L corner at (29, 29)
    q = refine_subpix(g, (30.5, 29.0))
    assert abs(q[0] - 29) < 1.5 and abs(q[1] - 29) < 1.5, q
    flat = np.full((60, 60), 160, dtype=np.uint8)
    assert refine_subpix(flat, (30.0, 30.0)) == (30.0, 30.0)


def test_fault_samples_refuse_entirely():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    g = np.full((100, 100), 160, dtype=np.uint8)
    g[20:81, 47:49] = 60
    g[20:81, 52:54] = 60
    tail = np.array([[50.0, 70.0], [50.0, 60.0], [50.0, 50.0], [50.0, 40.0],
                     [50.0, 30.0]])
    # Same geometry snaps cleanly when not at fault...
    assert correct_tip(g, tail, [(50.0, 22.0, 0.9)]) is not None
    # ...but a TimesFM fault-suspect verdict refuses (H198: never
    # validate a phantom, e.g. snapping onto a firing foreign apex).
    assert correct_tip(g, tail, [(50.0, 22.0, 0.9)], at_fault=True) is None


def test_anchor_cap_refuses_far_curve_peak():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    # H211: a peak 90px out along the tube must refuse even though the
    # extrapolated curve reaches toward it (P92/P101/P64 hijacks); peaks
    # within the 45px anchor cap still snap (s220's 39.6px verified fix).
    g = np.full((60, 200), 160, dtype=np.uint8)
    g[28:33, :150] = 60
    tail = np.array([[20.0, 30.5], [40.0, 30.5], [60.0, 30.5], [80.0, 30.5],
                     [100.0, 30.5]])
    assert correct_tip(g, tail, [(110.0, 30.5, 0.9)]) is not None
    assert correct_tip(g, tail, [(140.0, 30.5, 0.9)]) is not None
    assert correct_tip(g, tail, [(190.0, 30.5, 0.9)]) is None


def test_candidate_flag_returns_tier_without_changing_decisions():
    import cv2
    from prototypes.timesfm_tip_forecast.tip_correct import correct_tip

    # H212: return_candidate=True annotates the same decisions with tiers
    # (near snap -> tier 0); far-peak tier 1 needs quad blowup geometry
    # and is proven by the live P58 regen, not synthetics.
    g = np.full((60, 200), 160, dtype=np.uint8)
    g[28:33, :150] = 60
    tail = np.array([[20.0, 30.5], [40.0, 30.5], [60.0, 30.5], [80.0, 30.5],
                     [100.0, 30.5]])
    out = correct_tip(g, tail, [(110.0, 30.5, 0.9)], return_candidate=True)
    assert out is not None and len(out) == 3 and out[2] == 0
    assert correct_tip(g, tail, [(190.0, 30.5, 0.9)],
                       return_candidate=True) is None
