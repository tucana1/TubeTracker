"""A reading's route moved onto the middle of its tube, and how it lay at each bin (sparsetrack.learned.centre_route,
sparsetrack.routes), drawn the same way by the app (tubetracker.app.overlay)."""

import math

import numpy as np

from sparsetrack import learned, routes
from sparsetrack.analyze import Params
from sparsetrack.render import Renderer

FPB, NB, H, W = 300, 10, 120, 160


def _prob(band, width=8.0):
    """A probability movie (P x P_SCALE, reference coordinates): at bin b one straight band through ``band(b)[0]``
    at ``band(b)[1]`` degrees, ``width`` px wide at half maximum (no band where ``band(b)`` is None)."""
    bins = np.zeros((NB, H, W), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5  # continuous pixel centres
    for b in range(NB):
        if band(b) is None:
            continue
        (cx, cy), deg = band(b)
        a = math.radians(deg)
        d = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)  # distance across the band
        bins[b] = np.clip(learned.P_SCALE * np.exp(-0.5 * (d / (width / 2.355)) ** 2), 0, 255).astype(np.uint8)
    return Renderer(bins, {"n_bins": NB, "frames_per_bin": FPB, "shifts": [[0.0, 0.0]] * NB}), \
        {"n_bins": NB, "frames_per_bin": FPB}


def _res(path, tips=None, drift=None, rot=None, L=None):
    frames = [b * FPB + FPB // 2 for b in range(NB)]
    res = {"x": 20.0, "y": 60.0, "path": [list(map(float, q)) for q in path],
           "length": {"frames": frames, "px": list(L) if L is not None else [60.0] * NB},
           "exit_xy": list(map(float, path[0]))}
    if tips is not None:
        res["tip"] = {"frames": frames, "xy": tips}
    if drift is not None:
        res["drift"] = {"frames": frames, "xy": drift}
    if rot is not None:
        res["rotation_deg"] = rot
    return res


def _across(pts, centre, deg):
    a = math.radians(deg)
    pts = np.asarray(pts, float)
    return -(pts[:, 0] - centre[0]) * math.sin(a) + (pts[:, 1] - centre[1]) * math.cos(a)


def test_a_route_along_a_wall_moves_to_the_middle_of_the_band():
    prob, meta = _prob(lambda b: ((80.0, 60.0), 0.0))
    route = [(40.0 + i, 63.5) for i in range(80)]  # 3.5 px off the band's middle (y = 60)
    res = _res(route, tips=[[100.0, 63.5]] * NB)
    learned.centre_route(res, prob, meta, Params())
    assert np.all(np.abs(np.asarray(res["path"])[:, 1] - 60.0) < 0.5)
    assert abs(res["tip"]["xy"][-1][1] - 60.0) < 0.5 and abs(res["tip"]["xy"][-1][0] - 100.0) < 0.5
    assert res["exit_xy"] == [40.0, 63.5] and res["route_centred_px"] > 3.0  # the turning pivot stays


def test_the_band_is_found_where_the_grain_was_then():
    prob, meta = _prob(lambda b: ((84.0, 57.0) if b >= NB - 4 else (80.0, 60.0), 30.0))
    a = math.radians(30.0)
    route = [(80.0 + s * math.cos(a) - 3.0 * math.sin(a), 60.0 + s * math.sin(a) + 3.0 * math.cos(a))
             for s in np.arange(-30, 30, 1.0)]  # 3 px beside the band, in the grain's own frame
    drift = [[0.0, 0.0]] * (NB - 4) + [[4.0, -3.0]] * 4  # the late bins: the grain (and band) moved
    res = _res(route, drift=drift)
    learned.centre_route(res, prob, meta, Params())
    assert np.all(np.abs(_across(res["path"], (80.0, 60.0), 30.0)) < 0.6)


def test_a_route_that_turns_with_its_grain_is_centred_as_it_lay():
    pivot = (40.0, 63.5)
    deg = 12.0
    a = math.radians(deg)
    # the tube turned by 12 degrees about its exit in the late bins; the stored route (unturned) is 3 px off it
    late_mid = (pivot[0] + 40.0 * math.cos(a) + 3.0 * math.sin(a), pivot[1] + 40.0 * math.sin(a) - 3.0 * math.cos(a))
    prob, meta = _prob(lambda b: (late_mid, deg) if b >= NB - 4 else ((80.0, 60.5), 0.0))
    route = [(40.0 + i, 63.5) for i in range(80)]
    res = _res(route, rot=[0.0] * (NB - 4) + [deg] * 4)
    learned.centre_route(res, prob, meta, Params())
    from sparsetrack.report import turned_path
    late = turned_path(res, NB - 2, {"params": {"rot_pivot": "exit"}})
    assert np.all(np.abs(_across(late[5:-5], late_mid, deg)) < 0.6)


def test_how_the_tube_lay_at_each_bin_is_kept_and_drawn_the_same_by_the_app():
    from tubetracker.app.overlay import bent_at
    # the tube lies 2 px below its final place until bin 4, then moves up to it (pushed sideways as it grew)
    prob, meta = _prob(lambda b: ((80.0, 62.0), 0.0) if b < 5 else ((80.0, 60.0), 0.0))
    route = [(40.0 + i, 60.0) for i in range(80)]
    res = _res(route, tips=[[90.0, 60.0]] * NB, L=[0.0] + [70.0] * (NB - 1))
    learned.centre_route(res, prob, meta, Params())
    assert res["bend"]["px10"][0] == []  # no tube yet
    early, late = routes.bent(res["path"], res["bend"], 2), routes.bent(res["path"], res["bend"], NB - 1)
    assert np.all(np.abs(early[:65, 1] - 62.0) < 0.5) and np.all(np.abs(late[:, 1] - 60.0) < 0.5)
    assert abs(res["tip"]["xy"][2][1] - 62.0) < 0.5 and abs(res["tip"]["xy"][NB - 1][1] - 60.0) < 0.5
    rec = {"path": res["path"], "bend": {"step": res["bend"]["step_px"], "rows": res["bend"]["px10"],
                                         "s": routes.arc(res["path"]), "n": routes.normals(res["path"])}}
    assert np.allclose(bent_at(rec, 2), early) and np.allclose(bent_at(rec, NB - 1), late)


def test_no_band_or_a_band_wider_than_a_tube_leaves_the_route_as_it_is():
    route = [(40.0 + i, 63.5) for i in range(40)]
    for band, width in ((lambda b: None, 8.0), (lambda b: ((80.0, 60.0), 0.0), 20.0)):
        prob, meta = _prob(band, width)
        res = _res(route)
        learned.centre_route(res, prob, meta, Params())
        assert res["path"] == [list(map(float, q)) for q in route] and "bend" not in res


def _band_centres_loop(prof, offs, min_p, max_width):
    """The band search written out point by point (the reference the vectorised one must match)."""
    out = np.full(len(prof), np.nan)
    j0, n = int(np.argmin(np.abs(offs))), prof.shape[1]
    for i, row in enumerate(prof):
        j = j0
        while True:
            left = row[j - 1] if j > 0 else -np.inf
            right = row[j + 1] if j < n - 1 else -np.inf
            if max(left, right) <= row[j]:
                break
            j = j - 1 if left > right else j + 1
        peak = float(row[j])
        if peak < min_p:
            continue
        half = 0.5 * peak
        a = b = j
        while a > 0 and row[a - 1] >= half:
            a -= 1
        while b < n - 1 and row[b + 1] >= half:
            b += 1
        if a == 0 or b == n - 1:
            continue
        lo = offs[a - 1] + (half - row[a - 1]) / max(row[a] - row[a - 1], 1e-9) * (offs[a] - offs[a - 1])
        hi = offs[b] + (row[b] - half) / max(row[b] - row[b + 1], 1e-9) * (offs[b + 1] - offs[b])
        if hi - lo <= max_width:
            out[i] = 0.5 * (lo + hi)
    return out


def test_the_band_search_matches_its_point_by_point_definition():
    rng = np.random.default_rng(0)
    offs = np.arange(-10, 10 + 1e-6, 0.25)
    x = offs[None, :]
    for _ in range(50):
        n = 60
        prof = sum(rng.uniform(0, 1, (n, 1)) * np.exp(-0.5 * ((x - rng.uniform(-10, 10, (n, 1))) / rng.uniform(1.5, 5, (n, 1))) ** 2)
                   for _ in range(2)) + rng.normal(0, 0.03, (n, len(offs)))
        prof = (np.round(np.clip(prof, 0, 1) * 250) / 250).astype(np.float32)  # quantised like the stored maps
        want, got = _band_centres_loop(prof, offs, 0.35, 14.0), learned._band_centres(prof, offs, 0.35, 14.0)
        assert np.array_equal(np.isnan(want), np.isnan(got))
        assert np.allclose(want[np.isfinite(want)], got[np.isfinite(got)], atol=1e-6)


def test_a_route_is_cut_to_a_length_as_the_app_cuts_it():
    from tubetracker.app.overlay import to_length
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0)]
    for length in (0.0, 4.0, 10.0, 15.5, 30.0, 33.0, 60.0):
        assert np.allclose(routes.cut(route, length), to_length(route, length))


def test_a_tube_drawn_off_the_network_s_tube_is_flagged():
    prob, meta = _prob(lambda b: ((80.0, 60.0), 0.0))
    p = Params(centre_route=False)
    on = _res([(40.0 + i, 60.0) for i in range(80)], L=[60.0] * NB)
    on.update(status="emerged_within", flags=[], r=10.0, x=28.0, y=60.0)
    learned.drawn_check(on, prob, meta, p)
    assert not any(f.startswith("drawn_off_tube") for f in on["flags"]) and min(on["drawn_on_tube"]["share"]) > 0.9
    off = _res([(40.0 + i, 75.0) for i in range(80)], L=[60.0] * NB)  # 15 px beside the band
    off.update(status="emerged_within", flags=[], r=10.0, x=28.0, y=75.0)
    learned.drawn_check(off, prob, meta, p)
    assert any(f.startswith("drawn_off_tube:") for f in off["flags"])


def test_a_bin_drawn_along_its_own_route_where_the_flood_read_another():
    from tubetracker.app.overlay import bent_at
    final = [[float(x), 60.0] for x in range(40, 120)]
    other = [[40.0, 60.0 + y] for y in range(0, 40)]  # the tube went down then (another branch)
    res = {"path": final, "path_by_bin": {"routes": [other], "index": [-1, 0, 0, -1]}}
    assert np.allclose(routes.route_at(res, 1), other) and np.allclose(routes.route_at(res, 3), final)
    rec = {"path": final, "by_bin": {"routes": [other], "index": [-1, 0, 0, -1]}}
    assert bent_at(rec, 2) == other and bent_at(rec, 0) == final


def test_only_routes_that_leave_the_final_one_are_kept_per_bin():
    to_ref = lambda y, x: [float(x), float(y)]
    final = [[float(x), 60.0] for x in range(40, 120)]
    along = [(60.0, float(x)) for x in range(40, 80)]   # the final route's first 40 px (crop (y, x))
    down = [(60.0 + y, 40.0) for y in range(0, 30)]     # another branch
    lines = [None, along, down, None, along]
    exit_len = np.array([0.0, 39.0, 29.0, 0.0, 41.0])
    out = learned._routes_by_bin(lines, exit_len, final, to_ref, 2.5)
    assert out is None  # bin 2's route is shorter than bin 1's: the length (and route) of bin 1 hold
    exit_len = np.array([0.0, 20.0, 29.0, 0.0, 41.0])
    out = learned._routes_by_bin(lines, exit_len, final, to_ref, 2.5)
    assert out["index"] == [-1, -1, 0, 0, -1] and len(out["routes"]) == 1
