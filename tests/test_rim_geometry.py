"""Tubes near their own grain's rim: synthetic geometry for the change reader's tip continuation."""

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from sparsetrack.analyze import Params, analyze_grain, arrival_map, dp_front
from sparsetrack.render import Renderer

SIZE, GX, GY, R = 320, 160.0, 160.0, 13.0


def _polyline(points, step=0.25):
    """Dense samples and their arc length along a polyline given as (x, y) vertices."""
    pts = [np.asarray(points[0], float)]
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    pts = np.array(pts)
    return pts, np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def _hairpin():
    """A tube leaving the grain's left side, 24 px out, turning up and back (a half circle of radius 9) and running
    back to the right 5 px above the grain's top - every point of that last leg nearer the rim than the turn."""
    first = [(GX - R, GY), (GX - R - 24, GY)]
    cx, cy = GX - R - 24, GY - 9
    turn = [(cx - 9 * np.sin(a), cy + 9 * np.cos(a)) for a in np.linspace(0, np.pi, 30)]
    back = [(GX - R - 24, GY - 18), (GX + R, GY - 18)]
    return _polyline(first + turn[1:] + back[1:])


def _render(line, arc, rate=2.0, onset=6, n_bins=60, seed=0, rim_change_bin=4):
    """The tube grows along ``line`` at ``rate`` px/bin from ``onset``; from ``rim_change_bin`` the grain's rim
    darkens all round (the body's own change, as at germination), so the whole rim band is one change region."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    ring = np.exp(-((np.hypot(xx - GX, yy - GY) - R) ** 2) / 3.0)
    outside = np.hypot(xx - GX, yy - GY) >= R
    pix = np.stack([xx.ravel(), yy.ravel()], axis=1)
    bins, true = [], []
    for t in range(n_bins):
        L = min(arc[-1], max(0.0, (t - onset + 1) * rate))
        true.append(L)
        img = 175 - (90 + (14 if t >= rim_change_bin else 0)) * ring
        if L > 0:
            d = cKDTree(line[arc <= L]).query(pix)[0].reshape(SIZE, SIZE)
            img += outside * (18 * np.exp(-d ** 2 / 0.8) - 22 * np.exp(-(d - 1.6) ** 2 / 0.5))
        bins.append(img + rng.normal(0, 0.4, img.shape))
    return np.stack(bins).astype(np.float16), np.array(true)


def test_arrival_map_needs_the_change_to_stay():
    diffs = np.zeros((20, 8, 8), np.float32)
    diffs[5:, 2, 2] = 10.0        # arrives at bin 5 and stays
    diffs[3:6, 5, 5] = 10.0       # a transient
    arr = arrival_map(diffs, thr=4.0, sigma=0.3)
    assert arr[2, 2] == 5 and arr[5, 5] == -1 and arr[0, 0] == -1


def test_a_tube_turning_back_along_its_grain_is_read_the_long_way():
    line, arc = _hairpin()
    bins, true = _render(line, arc)
    meta = {"shifts": [[0.0, 0.0]] * len(bins), "n_bins": len(bins), "frames_per_bin": 300}
    grain = {"id": "g001", "x": GX, "y": GY, "r": R}
    p = Params(half=100, exit_edge=False, rotate=False)
    off = analyze_grain(Renderer(bins, meta), meta, grain, [], p)
    on = analyze_grain(Renderer(bins, meta), meta, grain, [], replace(p, tip_continue=True))
    L_off, L_on = np.array(off["length"]["px"]), np.array(on["length"]["px"])
    assert any(f.startswith("tip_continued") for f in on["flags"])
    # without it the reading stops near the turn; with it, the whole tube
    assert L_off[-1] < 0.6 * true[-1]
    assert abs(L_on[-1] - true[-1]) < 0.12 * true[-1]
    # while the tube is still on its first leg nothing changes
    first_leg = true <= 20
    assert np.allclose(L_on[first_leg], L_off[first_leg])
    assert np.all(np.diff(L_on) >= -1e-9)


def test_a_straight_tube_is_not_continued():
    line, arc = _polyline([(GX - R, GY), (GX - R - 70, GY)])
    bins, true = _render(line, arc, rate=1.5, n_bins=40)
    meta = {"shifts": [[0.0, 0.0]] * len(bins), "n_bins": len(bins), "frames_per_bin": 300}
    grain = {"id": "g001", "x": GX, "y": GY, "r": R}
    p = Params(half=100, exit_edge=False, tip_continue=True)
    res = analyze_grain(Renderer(bins, meta), meta, grain, [], p)
    assert not any(f.startswith("tip_continued") for f in res["flags"])


def test_second_front_starts_at_the_tip():
    """dp_front may start anywhere; the continuation anchors its second front with a first row only 0 explains."""
    ev = -np.ones((12, 30))
    ev[:, 5:25] = 1.0             # material beyond a short gap, there from the start
    free = dp_front(ev, vmax=4)
    anchored = dp_front(np.vstack([np.full((1, 30), -1e3), ev]), vmax=4)[1:]
    assert free[0] == 25
    assert anchored[0] <= 4 and np.all(np.diff(anchored) <= 4) and anchored[-1] == 25
