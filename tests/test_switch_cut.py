"""Unit tests for the switch-cut rule (pure logic, synthetic support)."""

import numpy as np

from prototypes.timesfm_tip_forecast.switch_cut import (
    cut_at_support_dip,
    normalized_root_support,
    transverse_wall_energy,
)


def _path(base=20.0, dip=3.0, n=30, dip_at=14, dip_w=12, post: object = "dip"):
    w = np.full(n, base)
    w[dip_at : dip_at + dip_w] = dip
    if post == "dip":
        w[dip_at + dip_w :] = dip  # dip-then-background (switch prototype)
    elif post is not None:
        w[dip_at + dip_w :] = post
    return w, np.arange(n, dtype=float)


def test_deep_midpath_dip_cuts_at_dip():
    w, arc = _path()
    assert cut_at_support_dip(w, arc) == 14 + 4 - 1


def test_narrow_valley_does_not_cut():
    # H150: a deep but narrow valley on a real tube must not cut.
    w, arc = _path(dip_w=4, post=20.0)
    assert cut_at_support_dip(w, arc) is None


def test_dip_with_strong_recovery_does_not_cut():
    # H150: wide dip but strong recovery past the weak run = valley.
    w, arc = _path(dip_at=8, dip_w=12, post=20.0)
    assert cut_at_support_dip(w, arc) is None


def test_dip_with_weak_foreign_recovery_cuts():
    # H150: dip-then-weaker-foreign-walls still cuts (s245 class).
    w, arc = _path(dip_at=8, dip_w=12, post=10.0)
    assert cut_at_support_dip(w, arc) is not None


def test_supported_path_passes():
    rng = np.random.default_rng(0)
    w = 20.0 + rng.normal(0, 1.5, 26)
    assert cut_at_support_dip(w, np.arange(26, dtype=float)) is None


def test_faint_flat_path_passes():
    # Uniformly low support (faint-real) has no RELATIVE dip: no cut.
    w = np.full(25, 4.0)
    assert cut_at_support_dip(w, np.arange(25, dtype=float)) is None


def test_apex_weakness_ignored():
    # Weak tail only (apex naturally weak): excluded window, no cut.
    w = np.full(26, 20.0)
    w[-2:] = 2.0
    assert cut_at_support_dip(w, np.arange(26, dtype=float)) is None


def test_detached_root_returns_none():
    w = np.zeros(20)
    assert cut_at_support_dip(w, np.arange(20, dtype=float)) is None


def test_short_path_returns_none():
    assert cut_at_support_dip(np.ones(5), np.arange(5, dtype=float)) is None


def test_degenerate_path_support_is_zero():
    # Single/double-point paths have no tangent (H147 crash): zero support.
    g = np.full((40, 40), 128.0)
    pts = np.array([[20.0, 20.0], [21.0, 20.5]])
    out = transverse_wall_energy(g, pts)
    assert out.shape == (2,)
    assert bool(np.all(out == 0.0))


def test_normalized_root_support_separates_bands():
    # H148 calibration bands: faults <=0.036, clean >=0.199 (nrootsup).
    arc = np.array([0.0, 3.0, 6.0, 9.0, 20.0, 40.0])
    fault = np.array([2.0, 2.4, 1.8, 2.2, 20.0, 22.0])
    clean = np.array([18.0, 20.0, 17.0, 19.0, 21.0, 20.0])
    assert normalized_root_support(fault, arc, 85.0) < 0.10
    assert normalized_root_support(clean, arc, 85.0) >= 0.10
    assert np.isnan(normalized_root_support(clean, arc, 0.0))


def _synthetic_tube(width=100, height=100, bar_rows=(45, 55), bar_x=(10, 60)):
    # Bright bar (tube) on dark background, blunt end at bar_x[1].
    img = np.full((height, width), 40.0)
    img[bar_rows[0]:bar_rows[1], bar_x[0]:bar_x[1]] = 180.0
    return img


def test_retreat_recovers_supported_ground():
    # Start inside the post-apex void: retreat -x must recover the bar.
    from prototypes.timesfm_tip_forecast.switch_cut import retreat_to_support

    g = _synthetic_tube()
    pos, ok = retreat_to_support(
        g, np.array([75.0, 50.0]), np.array([-1.0, 0.0]), ref_support=100.0
    )
    assert ok and pos[0] < 65.0


def test_extend_stops_near_bar_end():
    # From mid-bar along +x: endpoint near the blunt end, not runaway.
    from prototypes.timesfm_tip_forecast.switch_cut import extend_tip_along_ridge

    g = _synthetic_tube()
    end, tr = extend_tip_along_ridge(
        g, np.array([30.0, 50.0]), np.array([1.0, 0.0]), ref_support=100.0
    )
    assert 55.0 <= end[0] <= 75.0
    assert abs(end[1] - 50.0) <= 4.0
