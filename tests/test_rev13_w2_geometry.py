"""rev13 W2 acceptance: the current-path geometry contract.

- the current path is the support truncated at the accepted front
  arclength with its final segment interpolated ONCE;
- current_path[-1] == tip and arclength(current_path) == length_px
  within floating-point tolerance;
- full length is withheld unless the root link is verified AND the
  path is complete (cap interior to the support);
- support length stays a SEPARATE named quantity.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_truncation_and_single_interpolation():
    from prototypes.v30_video_apex.ownership import (
        current_path_from_support)
    support = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]
    path, length = current_path_from_support(support, 40.0)
    assert abs(path[-1][0] - 40.0) < 1e-9 and abs(path[-1][1]) < 1e-9
    assert abs(length - 40.0) < 1e-9
    assert path[0] == [0.0, 0.0]
    # final segment interpolated ONCE: no duplicate intermediate points
    assert len(path) == 2


def test_cap_beyond_support_clamps():
    from prototypes.v30_video_apex.ownership import (
        current_path_from_support)
    support = [[0.0, 0.0], [30.0, 40.0]]
    path, length = current_path_from_support(support, 999.0)
    assert abs(path[-1][0] - 30.0) < 1e-9
    assert abs(path[-1][1] - 40.0) < 1e-9
    assert abs(length - 50.0) < 1e-9


def test_geometry_consistency_flags_mismatch():
    from prototypes.v30_video_apex.ownership import geometry_consistent
    ok = geometry_consistent([[0.0, 0.0], [30.0, 40.0]],
                             (30.0, 40.0), 50.0)
    assert ok["endpoint_ok"] and ok["length_ok"]
    bad = geometry_consistent([[0.0, 0.0], [30.0, 40.0]],
                              (30.0, 41.0), 60.0)
    assert not bad["endpoint_ok"] and not bad["length_ok"]


def test_route_hypothesis_cap_is_never_the_support_end():
    from prototypes.v30_video_apex.ownership import RouteHypothesis
    h = RouteHypothesis(
        owner_id="g1", frame=1, route_id="r", attachment=(0.0, 0.0),
        support_xy=[[0.0, 0.0], [50.0, 0.0], [120.0, 0.0]],
        current_prefix_len_px=50.0, local_cap_score=1.0,
        whole_route_score=1.0,
        current_path_xy=[[0.0, 0.0], [50.0, 0.0]], tip_xy=(50.0, 0.0))
    assert h.cap_xy() == (50.0, 0.0)          # not (120, 0)!
    h2 = RouteHypothesis(
        owner_id="g1", frame=1, route_id="r", attachment=(0.0, 0.0),
        support_xy=[[0.0, 0.0], [120.0, 0.0]], current_prefix_len_px=50.0,
        local_cap_score=1.0, whole_route_score=1.0,
        current_path_xy=[[0.0, 0.0], [50.0, 0.0]])
    assert h2.cap_xy() == (50.0, 0.0)         # from the current path
    h3 = RouteHypothesis(
        owner_id="g1", frame=1, route_id="r", attachment=(0.0, 0.0),
        support_xy=[[0.0, 0.0], [120.0, 0.0]], current_prefix_len_px=50.0,
        local_cap_score=1.0, whole_route_score=1.0)
    assert h3.cap_xy() is None                # no invented cap


def test_cache_keys_split_pixel_vs_measurement():
    """rev13 W4: a cadence-only change must change the measurement key
    and NOT the pixel key; everything else moves both."""
    from scripts.rev12_build_inferred_interval import cache_keys
    base = dict(frames=[1, 2], owners=["a"], ckpt_sha="x" * 16,
                corrections_digest="c", ui_corrections_digest="u",
                acquisition_cadence_s=None, cadence_source="")
    p0, m0 = cache_keys(**base)
    # cadence-only change: measurement moves, pixel does not
    p1, m1 = cache_keys(**{**base, "acquisition_cadence_s": 3.0,
                           "cadence_source": "acquisition log"})
    assert p1 == p0 and m1 != m0
    # calibration-only change: same property
    p2, m2 = cache_keys(**{**base, "calibration": "0.5 um/px"})
    assert p2 == p0 and m2 != m0
    # a real inference dependency moves BOTH
    p3, m3 = cache_keys(**{**base, "ui_corrections_digest": "u2"})
    assert p3 != p0 and m3 != m0
    # identical inputs are identical keys (the reuse case)
    p4, m4 = cache_keys(**base)
    assert (p4, m4) == (p0, m0)
