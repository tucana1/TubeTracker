"""Length work (rev16): grain-circle exit estimator regression tests."""
import math

import numpy as np

from prototypes.v30_video_apex.route_evidence import (
    EXIT_ESTIMATE_CONTRACT,
    estimate_grain_exit,
)


def _tube_map(shape=(120, 120), centre=(60.0, 60.0), radius=13.0,
              exit_angle_deg=100.0, tube_len=30.0):
    """Synthetic body map: bright grain disc + bright bar leaving at the exit."""
    prob = np.full(shape, 0.02, dtype=float)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    prob[np.hypot(xx - centre[0], yy - centre[1]) <= radius] = 1.0
    a = math.radians(exit_angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    for t in np.arange(0.0, tube_len, 0.25):
        x, y = centre[0] + ux * (radius + t), centre[1] + uy * (radius + t)
        xi, yi = int(round(x)), int(round(y))
        prob[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] = 0.98
    exit_xy = [centre[0] + ux * radius, centre[1] + uy * radius]
    return prob, exit_xy


def test_exit_estimator_hits_synthetic_crossing():
    prob, truth = _tube_map()
    root = [truth[0] + 2.0, truth[1] + 3.0]  # rim root a few px down-tube
    est, support = estimate_grain_exit(prob, (0, 0), (60.0, 60.0), 13.0, root)
    assert est is not None
    assert math.dist(est, truth) <= 1.0
    assert support > 0.9


def test_exit_estimator_contract_versioned():
    assert EXIT_ESTIMATE_CONTRACT == 'grain_circle_exit.v1'


def test_exit_estimator_returns_none_on_empty_map():
    prob = np.full((40, 40), np.nan)
    est, support = estimate_grain_exit(prob, (0, 0), (20.0, 20.0), 13.0, (33.0, 20.0))
    assert est is None and support == 0.0


def test_short_nascent_tip_survives_routing():
    """A true tip 5.3 px from the rim root (cf70@9000 regime) must not be
    filtered by any tip-to-root distance cutoff."""
    import math
    from prototypes.v30_video_apex.route_evidence import generate_owned_routes
    prob = np.full((80, 80), 0.02, dtype=float)
    cx, cy, radius = 40.0, 40.0, 13.0
    yy, xx = np.mgrid[:80, :80]
    prob[np.hypot(xx - cx, yy - cy) <= radius] = 1.0
    # Bright 6 px tube leaving west; rim root will sit ~1 px outside it.
    for t in np.arange(0.0, 7.0, 0.25):
        x, y = cx - (radius + t), cy
        prob[max(0, int(y) - 1):int(y) + 2, max(0, int(x) - 1):int(x) + 2] = 0.98
    tip = [cx - radius - 5.3, cy]
    owner = {'id': 'o', 'grain_native': [cx, cy], 'grain_radius_px': radius}
    caps = [{'cap_id': 'c1', 'tip_xy': tip, 'probability': 0.99,
             'source_frame': 0}]
    rows = generate_owned_routes(prob, (0, 0), owner, caps)
    assert rows, 'nascent tip was filtered before routing'
    assert math.dist(rows[0]['tip_xy'], tip) < 1e-6


def test_exit_estimator_outside_window_ignored():
    # Tube leaves 150 deg away from the root angle: outside the window,
    # so the estimator must not claim the true crossing.
    prob, truth = _tube_map(exit_angle_deg=250.0)
    root_angle_exit = [60.0 + 13.0 * math.cos(math.radians(70.0)),
                       60.0 + 13.0 * math.sin(math.radians(70.0))]
    est, _ = estimate_grain_exit(prob, (0, 0), (60.0, 60.0), 13.0,
                                 root_angle_exit, half_window_deg=30.0)
    if est is not None:
        assert math.dist(est, truth) > 5.0
