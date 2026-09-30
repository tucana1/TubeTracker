"""The flood's tip tracker (Params.flood_tip_track, sparsetrack/tiptrack.py) on synthetic movies: it follows a
growing tip through the short-interval difference, and a stalled flood is carried on only along material laid down
tip first."""

import math

import numpy as np
from scipy.special import erfc

from sparsetrack import tiptrack
from sparsetrack.analyze import Params


class _Stack:
    """A movie behind tiptrack's frame interface."""

    def __init__(self, img):
        self.img, self.n, self.half = img, len(img), img.shape[1] // 2

    def __len__(self):
        return self.n

    def __call__(self, t):
        return self.img[t]

    def mean3(self, t):
        return self.img[max(t - 1, 0):min(t + 2, self.n)].mean(axis=0)


def _growing_line(n=50, size=120, x0=20.0, y=60.0, onset=5, rate=1.5, stop=30, seed=0):
    """A dark line growing along +x from (x0, y) at ``rate`` px/bin from ``onset``, stopping at bin ``stop``, its
    end blurred like a real tip, on noise; and its true tip x per bin."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img, tip = [], []
    for t in range(n):
        L = max(0.0, (min(t, stop) - onset + 1) * rate)
        tip.append(x0 + L)
        cap = 0.5 * erfc((xx - x0 - L) / (np.sqrt(2) * 1.2)) * (xx >= x0) if L > 0 else 0.0
        img.append(175 - 20 * cap * np.exp(-(yy - y) ** 2 / 1.5) + rng.normal(0, 0.4, (size, size)))
    return np.stack(img).astype(np.float32), np.array(tip)


def test_track_tip_follows_a_growing_tip():
    img, tip = _growing_line()
    blocked = np.zeros(img.shape[1:], bool)
    tr = tiptrack.track_tip(_Stack(img), ~blocked, blocked, 8, tip[8], 60.0, 0.0, vmax=3.0)
    for t in (20, 29):
        assert abs(tr[t, 0] - tip[t]) < 6.0 and abs(tr[t, 1] - 60.0) < 2.0  # (it runs a few px behind: 3-bin means)


def test_track_tip_never_steps_onto_a_grain():
    img, tip = _growing_line(stop=60)
    blocked = np.zeros(img.shape[1:], bool)
    blocked[50:70, 50:70] = True      # a grain across the line's way
    tr = tiptrack.track_tip(_Stack(img), ~blocked, blocked, 8, tip[8], 60.0, 0.0, vmax=3.0)
    xi, yi = np.round(tr[9:, 0]).astype(int), np.round(tr[9:, 1]).astype(int)
    assert not blocked[yi, xi].any()


def test_retrace_measures_a_curved_tube_along_its_probability():
    size = 200
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r0, cx, cy = 60.0, 100.0, 100.0   # a quarter circle of radius 60: 94.2 px
    ring = np.exp(-(np.hypot(xx - cx, yy - cy) - r0) ** 2 / 4.0) * ((xx >= cx) & (yy >= cy))
    blocked = np.zeros((size, size), bool)
    L = tiptrack.retrace(ring, blocked, (cx + r0, cy), (cx, cy + r0))
    assert abs(L - 0.5 * math.pi * r0) < 0.05 * 0.5 * math.pi * r0


def test_arrival_order_tells_a_tip_laid_path_from_material_that_came_at_once():
    img, tip = _growing_line(stop=40, rate=1.0)
    stack = _Stack(img)
    blocked = np.zeros(img.shape[1:], bool)
    along = np.stack([np.arange(22.0, 50.0), np.full(28, 60.0)], axis=1)
    s, a = tiptrack.arrivals_along(stack, along, blocked)
    assert tiptrack.arrival_order(s, a, len(img)) > 0.9 and tiptrack.longest_block(s, a, len(img)) < 10.0
    late = img.copy()
    late[30:, 30:34, 20:100] -= 20.0  # a band that appears all at once at bin 30
    across = np.stack([np.arange(22.0, 90.0), np.full(68, 32.0)], axis=1)
    s, a = tiptrack.arrivals_along(_Stack(late), across, blocked)
    assert not tiptrack.arrival_order(s, a, len(img)) >= 0.6
    assert tiptrack.longest_block(s, a, len(img)) > 50.0


def test_stall_bin_is_the_last_bin_the_length_grew():
    L = np.concatenate([np.linspace(0, 30, 40), np.full(30, 30.0)])  # grows to bin 39
    assert 39 <= tiptrack.stall_bin(L) < 49  # (by more than 1 px over the last 10 bins)
    assert tiptrack.stall_bin(np.zeros(50)) is None


def _stalled_flood(img, onset=5, rate=1.0, x0=20.0, y=60.0, map_end=50.0):
    """What the flood reports for a tube the probability map shows only up to x = ``map_end``: its length stops
    there, its tip stands there, its claimed pixels and their bins; and that map (uint8 P x 250)."""
    n, h, w = img.shape
    L = np.array([min(max(0.0, (t - onset + 1) * rate), map_end - x0) for t in range(n)])
    tips = np.stack([x0 + L, np.full(n, y)], axis=1)
    tube = np.zeros((h, w), bool)
    t_in = np.full((h, w), -1, np.int32)
    xs = np.arange(int(x0), int(map_end) + 1)
    tube[int(y) - 1:int(y) + 2, xs] = True
    t_in[int(y) - 1:int(y) + 2, xs] = np.clip(onset + (xs - x0) / rate, 0, n - 1).astype(np.int32)
    pstack = np.zeros((n, h, w), np.uint8)
    for t in range(n):
        pstack[t, int(y) - 1:int(y) + 2, int(x0):int(tips[t, 0]) + 1] = 250
    line = [(y, x) for x in np.arange(x0, map_end + 1.0)]
    return {"tube": tube, "t_in": t_in, "emerge": onset}, L, tips, line, pstack


def test_continue_flood_follows_the_tip_the_map_lost():
    img, tip = _growing_line(n=80, rate=1.0, stop=80)  # the tube grows to x = 95; the map sees it to x = 50
    fl, L, tips, line, pstack = _stalled_flood(img)
    blocked = np.zeros(img.shape[1:], bool)
    p = Params(flood_tip_track=True)
    cont = tiptrack.continue_flood(_Stack(img), pstack, blocked, fl, L, tips, line, fl["emerge"], 3.0, p)
    assert cont is not None and cont["gain"] > 30
    assert abs(cont["length"][-1] - (tip[-1] - 20.0)) < 0.1 * (tip[-1] - 20.0)
    assert np.all(cont["length"][:cont["start"]] == L[:cont["start"]])  # the flood's reading before it stopped


def test_continue_flood_rejects_material_that_came_all_at_once():
    img, tip = _growing_line(n=80, rate=1.0, stop=30)  # the tube stops at x = 46...
    img[60:, 59:62, 47:90] -= 20.0                     # ...and a straight piece lands beyond it at bin 60
    fl, L, tips, line, pstack = _stalled_flood(img, map_end=46.0)
    blocked = np.zeros(img.shape[1:], bool)
    p = Params(flood_tip_track=True)
    assert tiptrack.continue_flood(_Stack(img), pstack, blocked, fl, L, tips, line, fl["emerge"], 3.0, p) is None
