"""Sticky bias tracker: adopt / ramp / hold / flicker-reject (H205)."""

import numpy as np

from prototypes.timesfm_tip_forecast.sticky_bias import sticky_bias


def _series(n=45, eras=(), flicks=()):
    acc = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    cor = acc.copy()
    m = np.zeros(n, bool)
    for a, b, dx in eras:
        cor[a:b] += [dx, 0.0]
        m[a:b] = True
    for idx, dx in flicks:
        cor[idx] += [dx, 0.0]
        m[idx] = True
    return acc, cor, m


def test_adopts_sustained_era_with_bounded_ramp():
    acc, cor, m = _series(eras=[(20, 32, 12.0)])
    off = sticky_bias(acc, cor, m) - acc
    assert abs(off[31, 0] - 12.0) < 1e-9
    assert float(np.abs(np.diff(off[:, 0])).max()) <= 12.0 / 6 + 1e-9


def test_flicker_never_adopts():
    acc, cor, m = _series(flicks=[(10, 15.0), (12, 16.0), (14, 14.0)])
    off = sticky_bias(acc, cor, m) - acc
    assert float(np.abs(off).max()) < 1e-9


def test_unfixed_holds_bias():
    acc, cor, m = _series(eras=[(10, 20, 9.0)])
    m[20:30] = False  # gap holds
    off = sticky_bias(acc, cor, m) - acc
    assert abs(off[29, 0] - 9.0) < 1e-9


def test_reset_forces_zero_and_drops_bias():
    acc, cor, m = _series(eras=[(10, 30, 12.0)])
    reset = np.zeros(45, bool)
    reset[20] = True
    off = sticky_bias(acc, cor, m, reset_mask=reset) - acc
    assert float(np.hypot(*off[20])) < 1e-9
    # post-reset era re-adopts on fresh agreement
    assert abs(off[29, 0] - 12.0) < 1e-9


def test_reset_splits_agreement_window():
    acc, cor, m = _series(flicks=[(10, 12.0), (11, 12.0), (13, 12.0), (14, 12.0)])
    reset = np.zeros(45, bool)
    reset[12] = True  # 2+2 never agree across the split
    off = sticky_bias(acc, cor, m, reset_mask=reset) - acc
    assert float(np.abs(off).max()) < 1e-9


def test_far_candidates_adopt_on_era_ignore_isolated():
    acc, cor, m = _series()
    n = 45
    cand_xy = acc.copy()
    cand = np.zeros(n, bool)
    # agreeing far era
    for i in range(20, 25):
        cand_xy[i] += [40.0, 5.0]
        cand[i] = True
    # isolated far snap
    cand_xy[35] += [90.0, -20.0]
    cand[35] = True
    off = sticky_bias(acc, cor, m, cand_xy=cand_xy, cand_mask=cand) - acc
    assert abs(off[24, 0] - 40.0) < 1e-9
    # isolated far snap ignored: s35 holds the adopted era bias, not its
    # own 90px candidate
    assert abs(off[35, 0] - 40.0) < 1e-9 and abs(off[35, 1] - 5.0) < 1e-9
