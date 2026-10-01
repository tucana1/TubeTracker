"""The flood's start and stop rules as options (sparsetrack.learned.flood / read_grain; Params.flood_*): the defaults
are the rules as they were, each option changes only what it names."""

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sparsetrack.analyze import Params
from sparsetrack.learned import P_SCALE, flood, read_grain
from sparsetrack.render import Renderer

SIZE, C, GR = 131, 65.0, 10.0


def _polar():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return np.hypot(xx - C, yy - C), np.arctan2(yy - C, xx - C)


def _stub(arr, rg, ang, deg, r0, r1, t0, per_px=1.0, half_deg=6.0):
    """A radial stub at ``deg`` from ``r0`` to ``r1`` px beyond the rim, arriving outwards from bin ``t0``."""
    sel = (np.abs(np.angle(np.exp(1j * (ang - np.deg2rad(deg))))) <= np.deg2rad(half_deg)) & (rg >= GR + r0) & (rg <= GR + r1)
    arr[sel] = np.minimum(arr[sel], t0 + np.floor((rg[sel] - GR - r0) * per_px).astype(int))
    return arr


def test_the_default_options_are_the_rules_as_they_were():
    rg, ang = _polar()
    arr = np.full((SIZE, SIZE), 90)
    _stub(arr, rg, ang, 0.0, 3.0, 40.0, 10)
    _stub(arr, rg, ang, 120.0, 3.0, 5.0, 4)                                   # a false start that never grows
    arr[(rg > GR + 3) & (rg < GR + 5) & (np.abs(ang + 1.5) < 0.9)] = 6         # an arc round the rim
    a = flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR)
    b = flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR, 12, 4, 4.0, 8.0, 40, 3.0, 60.0, 10.0, 3)
    assert a["emerge"] == b["emerge"] and np.array_equal(a["length"], b["length"])
    assert np.array_equal(a["tube"], b["tube"])


def test_a_stub_inside_the_halo_is_read_with_a_smaller_halo():
    rg, ang = _polar()
    arr = np.full((SIZE, SIZE), 60)
    _stub(arr, rg, ang, 30.0, 1.0, 40.0, 10)                                  # 1 px per bin out from r + 1 px
    default = flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR)
    near = flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR, halo=1.0)
    assert default["emerge"] == 12 and near["emerge"] == 10                  # seen 2 bins sooner
    assert default["length"][11] == 0.0 and 1.5 < near["length"][11] < 3.5
    assert abs(near["length"][-1] - default["length"][-1]) < 1.0


def test_the_arc_and_old_far_tests_are_options():
    rg, ang = _polar()
    arr = np.full((SIZE, SIZE), 60)
    arr[(rg > GR + 3) & (rg < GR + 5) & (np.abs(ang) < np.deg2rad(40))] = 10  # 80 degrees round the rim at once...
    _stub(arr, rg, ang, 0.0, 5.0, 40.0, 11)                                   # ...and a tube out of it
    assert flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR)["emerge"] == 11
    assert flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR, arc_deg=90.0)["emerge"] == 10
    arr = np.full((SIZE, SIZE), 60)
    _stub(arr, rg, ang, 0.0, 9.0, 30.0, 5, per_px=0.0)                         # a structure there from bin 5...
    _stub(arr, rg, ang, 0.0, 3.0, 9.5, 20, per_px=0.0)                         # ...reaching the rim at bin 20
    assert flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR)["emerge"] is None
    assert flood(arr, rg, ang, np.zeros((SIZE, SIZE), bool), GR, old_far_px=40.0)["emerge"] == 20


def test_a_tube_hugging_the_rim_survives_a_lower_min_len_or_a_longer_give_up():
    rg, ang = _polar()
    arr = np.full((SIZE, SIZE), 150)
    _stub(arr, rg, ang, 45.0, 3.0, 7.5, 5)                                     # 7.5 px out by bin 9, then 50 bins still
    _stub(arr, rg, ang, 45.0, 7.5, 30.0, 60)                                   # then on out, beyond the start band
    blocked = np.zeros((SIZE, SIZE), bool)
    assert flood(arr, rg, ang, blocked, GR)["length"][-1] == 0.0               # forgotten at bin 49: never restarts
    for kw in ({"min_len": 5.0}, {"give_up": 60}):
        fl = flood(arr, rg, ang, blocked, GR, **kw)
        assert fl["emerge"] == 5 and fl["length"][-1] > 25


def _movie(n_bins, r_image, prob_pixels):
    """A static grain (a bright disc of radius ``r_image`` px) on a textured field, and a probability movie in which each pixel
    of ``prob_pixels`` [(y, x, first bin)] is tube from that bin on."""
    size = 2 * 60
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = np.hypot(xx - 60.0, yy - 60.0)
    img = gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 20 + 175 + gaussian_filter(60.0 * (d < r_image), 0.8)
    bins = np.stack([img + rng.normal(0, 0.6, img.shape) for _ in range(n_bins)]).astype(np.float16)
    prob = np.zeros((n_bins, size, size), np.uint8)
    for y, x, t in prob_pixels:
        prob[t:, y, x] = int(P_SCALE)
    meta = {"shifts": [[0.0, 0.0]] * n_bins, "n_bins": n_bins, "frames_per_bin": 300}
    return Renderer(bins, meta), Renderer(prob, meta), meta


def _radial(r_from, r_to, deg, t0, per_px=1.0, width=1):
    """Pixels (y, x, bin) of a radial bar (reference coordinates, grain at 60, 60) growing ``per_px`` bins per px."""
    u = (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    out = []
    for k, s in enumerate(np.arange(r_from, r_to, 0.5)):
        for w in range(-width, width + 1):
            x, y = 60.0 + s * u[0] - w * u[1], 60.0 + s * u[1] + w * u[0]
            out.append((int(math.floor(y)), int(math.floor(x)), t0 + int((s - r_from) * per_px)))
    return out


def test_lengths_from_the_visible_edge():
    """The census circle (r 10) lies 3 px inside the grain's visible edge (13): the flood counts from the census rim,
    an annotator from the visible edge."""
    renderer, prob, meta = _movie(40, 13.0, _radial(13.2, 33.0, 0.0, 5, per_px=1.0))
    grain = {"id": "g001", "x": 60.0, "y": 60.0, "r": 10.0}
    p = dict(grain_track="phase", flood_half=50, flood_lookback=0.0)
    census = read_grain(renderer, prob, meta, grain, [], Params(**p))
    edge = read_grain(renderer, prob, meta, grain, [], Params(**p, flood_exit_edge=True))
    assert abs(census["length"]["px"][-1] - 23.0) < 1.5                     # 20 px of tube + 3 px of grain
    assert abs(edge["exit_edge_px"] - 3.0) < 1.0 and abs(edge["length"]["px"][-1] - 20.0) < 1.5
    assert census["onset_frame"] == edge["onset_frame"]


def test_the_radial_tip_reads_a_young_blob_widening_along_the_rim():
    """The flood starts on one piece at the halo, pieces joining along the rim get ever greater rim distances, and
    the stub's own pixels out to 7 px arrive later: the pixel of greatest rim distance is on the rim."""
    along_rim = []
    for k, deg in enumerate((0.0, 12.0, 24.0, 36.0)):                         # 1 px pieces round the rim, 2.8 px apart
        along_rim += [(y, x, 5 + k) for y, x, _ in _radial(13.5, 14.0, deg, 0, width=0)]
    stub = _radial(13.0, 17.4, 0.0, 10, per_px=0.0)                           # the stub itself, to 7 px, at bin 10
    renderer, prob, meta = _movie(40, 10.0, along_rim + stub)
    grain = {"id": "g001", "x": 60.0, "y": 60.0, "r": 10.0}
    p = dict(grain_track="phase", flood_half=50, flood_lookback=0.0, min_tube_px=0.0)
    dist_tip = read_grain(renderer, prob, meta, grain, [], Params(**p))
    radial = read_grain(renderer, prob, meta, grain, [], Params(**p, flood_tip="radial"))
    assert dist_tip["length"]["px"][-1] < 5.0                                  # the rim piece's own distance out
    assert abs(radial["length"]["px"][-1] - 7.0) < 1.0
    assert radial["tip"]["xy"][-1][0] > 66.0                                  # the tip is the stub's end, not the rim


def test_a_big_piece_turning_away_from_the_tube_is_a_passing_tube_not_growth():
    from sparsetrack.learned import _turns_away
    h, w = 60, 80
    tube = np.zeros((h, w), bool)
    tube[30, 10:40] = True                       # a tube growing along +x, its far end at x = 39
    dist = np.full((h, w), np.inf)
    dist[30, 10:40] = np.arange(30, dtype=float)
    straight = np.zeros((h, w), bool)
    straight[30, 41:52] = True                   # 12 px more, straight on
    sideways = np.zeros((h, w), bool)
    sideways[31:44, 41] = True                   # 12 px, at right angles
    small = np.zeros((h, w), bool)
    small[31:34, 41] = True                      # 3 px at right angles: a tube may curve
    sources = tube.copy()
    assert not _turns_away(dist, tube, straight, sources, 4, 50.0, 6.0)
    assert _turns_away(dist, tube, sideways, sources, 4, 50.0, 6.0)
    assert not _turns_away(dist, tube, small, sources, 4, 50.0, 6.0)
