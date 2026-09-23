"""rev8: a mask task's ring must mark the GRAIN, not a point on the tube.

The bug this pins, measured on real observations: `focus_xy` does not
mean the same thing across observation families. On obs-v30t-004 it is
the ball (detected grain 2.2 px away). On obs-r4-p03 it is a path
point, and a ring placed there landed 46.2 px down the tube, enclosing
empty background -- the user reported the app as off-centre and the
ring as meaningless. p03's path start sits 12.9 px from the grain
centre (a path begins at the tube's exit from the grain), so the
fallback anchor is labelled data, not a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    ring_anchor_for_observation)

P03 = {
    'obs_uuid': 'obs-r4-p03', 'source_frame': 48300,
    'focus_xy': [356.5, 432.7],
    'path_xy': [[351.5, 399.8], [360.5, 435.8], [391.1, 452.4]],
}
V04 = {
    'obs_uuid': 'obs-v30t-004', 'source_frame': 52500,
    'focus_xy': [1078.0, 872.0],
    'path_xy': [[1065.1, 876.7], [1017.9, 876.7], [984.4, 876.1]],
}


def test_off_ball_focus_falls_back_to_the_path_start():
    """p03: focus is 33 px down the tube -> ring goes to the grain exit."""
    anchor, src = ring_anchor_for_observation(P03)
    assert anchor == [351.5, 399.8]
    assert 'path-start' in src
    # the whole point: the chosen anchor is at the grain, the rejected
    # one was 46 px away inside empty background
    assert abs(anchor[0] - 356.5) > 4.0 or abs(anchor[1] - 432.7) > 30.0


def test_on_ball_focus_is_kept():
    """v30t-004: focus IS the ball (13.7 px from the path start)."""
    anchor, src = ring_anchor_for_observation(V04)
    assert anchor == [1078.0, 872.0]
    assert 'focus-on-ball' in src


def test_explicit_ball_field_wins_over_focus():
    obs = dict(P03, ball_xy=[10.0, 20.0])
    assert ring_anchor_for_observation(obs) == ([10.0, 20.0], 'explicit-ball')
    obs = dict(P03, grain_xy=[11.0, 21.0])
    assert ring_anchor_for_observation(obs)[0] == [11.0, 21.0]


def test_threshold_is_the_ring_radius_not_a_constant():
    """A focus 15 px along the tube is *not* the ball for a 14 px ring."""
    obs = dict(P03, path_xy=[[100.0, 100.0], [200.0, 100.0]],
               focus_xy=[115.0, 100.0])
    assert ring_anchor_for_observation(obs, target_r=14.0)[0] == [100.0, 100.0]
    assert ring_anchor_for_observation(obs, target_r=16.0)[0] == [115.0, 100.0]


def test_degrades_honestly_without_a_path_or_anchor():
    assert ring_anchor_for_observation({'focus_xy': [3.0, 4.0]}) == \
        ([3.0, 4.0], 'focus-no-path')
    assert ring_anchor_for_observation({}) == (None, 'no-anchor')
    # malformed path points never crash the builder
    assert ring_anchor_for_observation(
        {'focus_xy': [1.0, 2.0], 'path_xy': [[1.0], 'x', None]})[0] == [1.0, 2.0]
