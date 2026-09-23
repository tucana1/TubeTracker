"""Measuring the grain a ring must enclose (rev8).

The complaint that started this: the magenta marker on a mask task read
as off-centre. Two measured causes -- the anchor came from a path point
rather than the grain, and the marker was the same size as the grain so
any small error clipped it. These tests pin the measurement used to fix
both: centroid AND size from the same connected component, with the
tube severed first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    measure_blob, refine_ring_anchor)


def _synthetic_grain(xy=(50.0, 50.0), r=12.0, tube=True, size=140):
    img = np.full((size, size), 0.9, dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    img[np.hypot(xx - xy[0], yy - xy[1]) <= r] = 0.2
    if tube:                      # a stalk leaves the grain downward
        img[(yy > xy[1]) & (np.abs(xx - xy[0]) <= 3.5) & (yy < size)] = 0.3
    return img


def test_blob_measure_survives_the_tube():
    """The tube is contiguous with the grain: without severing it the
    component is a 25x55 px sliver and the centroid slides down the
    tube (measured on the real frames)."""
    img = _synthetic_grain((50.0, 50.0), r=12.0)
    c, rad, area, bb = measure_blob(img, (54.0, 46.0), search_r=45.0)
    assert c is not None
    # centroid lands on the grain, not on the tube below it
    assert abs(c[0] - 50.0) <= 1.5 and abs(c[1] - 50.0) <= 1.5, c
    # a 12 px grain measures ~12 px, not 20 (the tube would inflate it)
    assert 10.0 <= rad <= 14.0, rad
    # the bbox is round once the tube is severed
    assert (bb[2] - bb[0]) <= 2 * 14 and (bb[3] - bb[1]) <= 2 * 14, bb
    # and the size is what the ring is sized from, with visible margin
    ring_r = max(20.0, round(1.5 * rad * 2.0) / 2.0)
    assert ring_r - rad >= 5.0, (ring_r, rad)


def test_blob_measure_refuses_empty_background():
    flat = np.full((120, 120), 0.9)
    assert measure_blob(flat, (60.0, 60.0))[0] is None
    # noise alone must not become a grain
    rng = np.random.default_rng(0)
    noisy = 0.9 + 0.01 * rng.standard_normal((120, 120))
    c, _rad, area, _bb = measure_blob(noisy, (60.0, 60.0))
    if c is not None:               # if anything is found it must be tiny
        assert area < 40, (area, _rad)


def test_refine_step_walks_from_the_grain_exit_to_the_grain():
    """A path starts at the grain's EDGE: one grain radius upstream."""
    img = _synthetic_grain((60.0, 40.0), r=12.0)
    # path starts at the grain's lower edge and runs away downward
    path = [[60.0, 54.0], [60.0, 70.0], [60.0, 90.0]]
    xy, src, _moved = refine_ring_anchor([60.0, 54.0], path, img, grain_r=12.0)
    assert abs(xy[0] - 60.0) <= 2.0 and abs(xy[1] - 40.0) <= 2.5, xy
    assert '1R' in src


def test_ring_size_comes_from_the_measurement_not_a_constant():
    """Bigger grain, bigger ring -- a fixed radius cannot serve both."""
    big = _synthetic_grain((70.0, 70.0), r=22.0, size=200)
    small = _synthetic_grain((70.0, 70.0), r=9.0, size=200)
    _, rb, _, _ = measure_blob(big, (74.0, 66.0), search_r=60.0)
    _, rs, _, _ = measure_blob(small, (74.0, 66.0), search_r=60.0)
    assert rb > rs + 8.0, (rb, rs)
    assert max(20.0, round(1.5 * rb * 2.0) / 2.0) > max(20.0, round(1.5 * rs * 2.0) / 2.0)
