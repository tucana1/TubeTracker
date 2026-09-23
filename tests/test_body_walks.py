"""rev11 item 6: root-connected routes from owned-body probabilities.

The review requires proposals that FOLLOW the owner's own body
evidence with crossing alternatives retained. These tests pin the
behavior on synthetic maps: straight tube -> one route; Y-junction ->
both branches (alternatives); unrelated components -> ignored; empty
map -> nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.routes import propose_body_walks  # noqa: E402


def _tube_map(sz=120):
    p = np.zeros((sz, sz), np.float32)
    # horizontal tube, 5 px half width, from x=10 to x=110 at y=60
    p[57:64, 10:110] = 1.0
    return p


def test_straight_tube_single_route():
    p = _tube_map()
    routes = propose_body_walks(p, (12.0, 60.0), min_len_px=30.0)
    assert len(routes) >= 1
    r = routes[0]
    pts = np.asarray(r["polyline"], float)
    assert r["seed"].startswith("body-walk")
    # starts at the root end and runs along the tube (mostly +x)
    assert abs(pts[0, 1] - 60.0) < 4.0 and pts[0, 0] < 25.0
    dx = pts[-1, 0] - pts[0, 0]
    assert dx > 60.0, f"did not traverse the tube: {pts[0]} -> {pts[-1]}"
    assert abs(pts[-1, 1] - 60.0) < 6.0


def test_y_junction_keeps_crossing_alternatives():
    sz = 140
    p = np.zeros((sz, sz), np.float32)
    p[87:94, 10:70] = 1.0            # trunk
    # two branches at ±35-40 deg from (70, 90)
    for t in np.linspace(0, 55, 60):
        x = 70 + t * np.cos(np.radians(35))
        y = 90 - t * np.sin(np.radians(35))
        p[int(round(y)) - 3:int(round(y)) + 4,
          max(int(round(x)) - 1, 0):int(round(x)) + 2] = 1.0
        x = 70 + t * np.cos(np.radians(-35))
        y = 90 - t * np.sin(np.radians(-35))
        p[int(round(y)) - 3:int(round(y)) + 4,
          max(int(round(x)) - 1, 0):int(round(x)) + 2] = 1.0
    routes = propose_body_walks(p, (12.0, 90.0), min_len_px=30.0,
                                sep_px=25.0)
    assert len(routes) >= 2, f"Y-junction lost alternatives: {len(routes)}"
    # the two routes should diverge: sign of the mean y-slope differs
    slopes = []
    for r in routes:
        pts = np.asarray(r["polyline"], float)
        slopes.append(float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0]))
    assert max(slopes) * min(slopes) < 0, f"no sign divergence: {slopes}"


def test_unrelated_component_ignored():
    p = _tube_map()
    p[15:25, 15:25] = 1.0            # a blob far from the root component
    routes = propose_body_walks(p, (12.0, 60.0), min_len_px=30.0)
    hit_blob = False
    for r in routes:
        pts = np.asarray(r["polyline"], float)
        if pts[:, 1].min() < 40:
            hit_blob = True
    assert not hit_blob


def test_empty_map_returns_nothing():
    p = np.zeros((80, 80), np.float32)
    assert propose_body_walks(p, (10.0, 10.0)) == []
