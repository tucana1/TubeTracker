"""Following grains through a movie (sparsetrack/track.py) and reading grains that move or are lost."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

from sparsetrack.analyze import Params, analyze_grain, followed_drift, hold_after
from sparsetrack.evaluate import tip_error
from sparsetrack.render import Renderer
from sparsetrack.report import onset_intervals
from sparsetrack.track import FollowConfig, follow


def _grain(xx, yy, x, y, r):
    """A dark-rimmed grain with a slightly bright core (the census looks for the rim)."""
    d = np.hypot(xx - x, yy - y)
    return -90 * np.exp(-((d - r) ** 2) / 3.0) + 15 * np.exp(-(d ** 2) / (2 * (0.5 * r) ** 2))


def _movie(n_bins, positions, size=240, r=12.0, seed=0, blobs=(), tubes=None):
    """Bins of a textured field with grains at ``positions[k](t)`` (None: the grain is gone) and passing blobs
    (t0, t1, x, y, radius). ``tubes[k]`` = (angle, onset bin, px per bin): a tube attached to grain k."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    texture = gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 40 + 175
    bins = []
    for t in range(n_bins):
        img = texture.copy()
        for k, pos in enumerate(positions):
            p = pos(t)
            if p is None:
                continue
            img += _grain(xx, yy, p[0], p[1], r)
            if tubes and tubes.get(k):
                a, onset, rate = tubes[k]
                L = max(0.0, (t - onset + 1) * rate)
                if L > 0:
                    u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                    along = (xx - p[0]) * u[0] + (yy - p[1]) * u[1] - r
                    across = -(xx - p[0]) * u[1] + (yy - p[1]) * u[0]
                    cap = np.where(along > 0, 0.5 * erfc((along - L) / (np.sqrt(2) * 1.2)), 0.0)
                    img += cap * (18 * np.exp(-across ** 2 / 0.8) - 22 * np.exp(-(np.abs(across) - 1.6) ** 2 / 0.5))
        for t0, t1, bx, by, br in blobs:
            if t0 <= t < t1:
                img -= 80 * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br ** 2))
        bins.append(img + rng.normal(0, 0.6, img.shape))
    bins = np.stack(bins).astype(np.float16)
    meta = {"shifts": [[0.0, 0.0]] * n_bins, "n_bins": n_bins, "frames_per_bin": 300}
    return bins, meta


def test_follow_keeps_a_static_grain_still():
    bins, meta = _movie(40, [lambda t: (120.0, 120.0)])
    tr = follow(Renderer(bins, meta), 120.0, 120.0, 12.0, 0, 40)
    assert tr["lost_from"] is None
    assert np.max(np.hypot(*tr["xy"].T)) < 0.5


def test_follow_tracks_a_drifting_grain_to_subpixel():
    path = lambda t: (80.0 + max(0, t - 10) * 1.3, 90.0 + max(0, t - 10) * 0.7)  # 39 px over 30 bins
    bins, meta = _movie(40, [path])
    tr = follow(Renderer(bins, meta), 80.0, 90.0, 12.0, 0, 40)
    truth = np.array([[path(t)[0] - 80.0, path(t)[1] - 90.0] for t in range(40)])
    assert tr["lost_from"] is None
    assert np.max(np.hypot(*(tr["xy"] - truth).T)) < 0.8


def test_follow_bridges_a_passing_blob_and_a_push():
    # a dark blob hides the grain for 4 bins, during which it is pushed 14 px
    path = lambda t: (110.0, 110.0) if t < 20 else (124.0, 110.0)
    bins, meta = _movie(40, [path], blobs=[(18, 22, 115.0, 110.0, 25.0)])
    tr = follow(Renderer(bins, meta), 110.0, 110.0, 12.0, 0, 40)
    assert tr["lost_from"] is None
    assert np.hypot(*(tr["xy"][30] - [14.0, 0.0])) < 1.0


def test_follow_loses_a_grain_that_vanishes_and_does_not_jump_to_its_neighbour():
    # grain 0 bursts at bin 25; an identical neighbour sits 30 px away all along
    bins, meta = _movie(50, [lambda t: (100.0, 120.0) if t < 25 else None, lambda t: (130.0, 120.0)])
    others = [(130.0, 120.0, 12.0)]
    tr = follow(Renderer(bins, meta), 100.0, 120.0, 12.0, 0, 50, others)
    assert tr["lost_from"] == 25 and tr["lost_reason"] == "gap"
    assert np.all(np.isnan(tr["xy"][25:]))
    assert np.max(np.hypot(*tr["xy"][:25].T)) < 0.5


def test_follow_loses_a_grain_whose_centre_leaves_the_frame():
    path = lambda t: (30.0 - 2.0 * max(0, t - 5), 120.0)  # leaves on the left at bin 20
    bins, meta = _movie(40, [path])
    tr = follow(Renderer(bins, meta), 30.0, 120.0, 12.0, 0, 40, cfg=FollowConfig())
    assert tr["lost_from"] is not None and 16 <= tr["lost_from"] <= 21


def test_analyze_grain_follows_a_grain_moving_with_its_tube():
    """A grain pushed 20 px with its tube: read in the grain's own frame (grain_track "follow"), the lengths come
    out as for a still grain; the phase-correlation reading (0.6.0) is also given a chance."""
    n_bins, onset, rate = 60, 12, 1.2
    path = lambda t: (150.0 + min(max(t - 30, 0), 10) * 2.0, 160.0)
    bins, meta = _movie(n_bins, [path], size=320, r=13.0, tubes={0: (200.0, onset, rate)})
    grain = {"id": "g001", "x": 150.0, "y": 160.0, "r": 13.0}
    true = np.array([max(0.0, (t - onset + 1) * rate) for t in range(n_bins)])
    res = analyze_grain(Renderer(bins, meta), meta, grain, [], Params(half=100, grain_track="follow", exit_edge=False))
    assert res["status"] == "emerged_within"
    est = np.array(res["length"]["px"])
    assert np.median(np.abs(est[25:] - true[25:])) < 2.5
    drift = np.array(res["drift"]["xy"])
    assert abs(drift[-1][0] - 20.0) < 1.0 and abs(drift[-1][1]) < 1.0
    assert not any(f.startswith("grain_lost_after") for f in res["flags"])


def test_analyze_grain_holds_the_readings_of_a_lost_grain():
    n_bins, onset, rate = 60, 10, 1.5
    bins, meta = _movie(n_bins, [lambda t: (150.0, 160.0) if t < 40 else None], size=320, r=13.0,
                        tubes={0: (200.0, onset, rate)})
    grain = {"id": "g001", "x": 150.0, "y": 160.0, "r": 13.0}
    res = analyze_grain(Renderer(bins, meta), meta, grain, [], Params(half=100, grain_track="follow"))
    assert "grain_lost_after:11850" in res["flags"]  # bin 39 is the last followed: frame 39 * 300 + 150
    assert res["observed_until_frame"] == 11850
    px = np.array(res["length"]["px"])
    assert len(px) == n_bins and np.all(px[40:] == px[39]) and px[39] > 10
    # a grain lost almost at once cannot be read (lost_min_bins 15; present in most of the settling bins)
    bins, meta = _movie(40, [lambda t: (150.0, 160.0) if t < 13 else None], size=320, r=13.0)
    res = analyze_grain(Renderer(bins, meta), meta, grain, [], Params(half=100, grain_track="follow"))
    assert res["status"] == "unobservable" and res["flags"] == ["grain_lost_after:3750"]


def test_followed_drift_refines_to_the_phase_correlation():
    path = lambda t: (120.0 + 0.37 * t, 110.0 - 0.21 * t)
    bins, meta = _movie(30, [path])
    fd = followed_drift(Renderer(bins, meta), meta, {"id": "g", "x": 120.0, "y": 110.0, "r": 12.0}, [], Params())
    truth = np.array([[0.37 * t, -0.21 * t] for t in range(30)])
    truth -= truth[:3].mean(axis=0)  # drift is measured from the grain's place in the reference bins (their mean)
    assert fd["lost_from"] is None
    assert np.median(np.hypot(*(fd["drift"] - truth).T)) < 0.25


def test_followed_drift_is_the_phase_track_where_it_stays_with_the_grain():
    """Where the phase-correlation track (0.6.0) agrees with the grain's own track the drift is exactly the phase
    track (so the readings do not change there); where it locks onto a passing blob, the grain's own track."""
    from sparsetrack.analyze import local_shifts
    path = lambda t: (130.0 + 0.3 * t, 120.0)
    bins, meta = _movie(40, [path], size=320)
    r, p, g = Renderer(bins, meta), Params(grain_track="follow"), {"id": "g", "x": 130.0, "y": 120.0, "r": 12.0}
    crops = np.stack([r.crop(b, 130.0, 120.0, p.half) for b in range(40)])
    phase = local_shifts(crops, p.half - 0.5, 12.0, p.reg_pad, p.ref_bins)
    fd = followed_drift(r, meta, g, [], p)
    assert np.array_equal(fd["drift"], phase)
    # a big dark blob passes over the grain for 5 bins while it is pushed 12 px
    path = lambda t: (130.0, 120.0) if t < 20 else (142.0, 120.0)
    bins, meta = _movie(40, [path], size=320, blobs=[(18, 23, 134.0, 118.0, 30.0)])
    fd = followed_drift(Renderer(bins, meta), meta, g, [], p)
    assert fd["lost_from"] is None
    assert np.max(np.hypot(*(fd["drift"][25:] - [12.0, 0.0]).T)) < 1.0


def test_auto_reads_a_grain_in_its_own_frame_only_once_it_moves_off_its_place():
    from sparsetrack.analyze import reads_in_grain_frame
    p = Params(grain_track="auto")  # track_far_r 2: off its place = further than its diameter
    near = np.array([[0.0, 0.0], [10.0, 5.0], [np.nan, np.nan]])
    far = np.array([[0.0, 0.0], [20.0, 20.0], [np.nan, np.nan]])
    assert not reads_in_grain_frame(near, 12.0, p) and reads_in_grain_frame(far, 12.0, p)
    assert reads_in_grain_frame(near, 12.0, Params(grain_track="follow"))
    assert not reads_in_grain_frame(far, 12.0, Params())
    # end to end: a grain pushed 40 px with its tube is read in its own frame (its drift in the result)
    n_bins, onset, rate = 60, 12, 1.2
    path = lambda t: (140.0 + min(max(t - 30, 0), 10) * 4.0, 160.0)
    bins, meta = _movie(n_bins, [path], size=320, r=13.0, tubes={0: (200.0, onset, rate)})
    res = analyze_grain(Renderer(bins, meta), meta, {"id": "g001", "x": 140.0, "y": 160.0, "r": 13.0}, [],
                        Params(half=100, grain_track="auto", exit_edge=False))
    assert abs(res["drift"]["xy"][-1][0] - 40.0) < 1.0
    true = np.array([max(0.0, (t - onset + 1) * rate) for t in range(n_bins)])
    assert np.median(np.abs(np.array(res["length"]["px"])[25:] - true[25:])) < 2.5


def test_hold_after_pads_series_and_flags():
    frames = [150, 450, 750, 1050]
    res = {"flags": [], "length": {"frames": frames[:2], "px": [0.0, 5.0]},
           "tip": {"frames": frames[:2], "xy": [[1, 1], [2, 2]]}, "rotation_deg": [0.0, 1.0],
           "drift": {"frames": frames[:2], "xy": [[0, 0], [3, 4]]}}
    out = hold_after(res, frames, 2, "grain_lost_after:450")
    assert out["length"]["px"] == [0.0, 5.0, 5.0, 5.0] and out["length"]["frames"] == frames
    assert out["tip"]["xy"][-1] == [2, 2] and out["drift"]["xy"][-1] == [3, 4] and out["rotation_deg"][-1] == 1.0
    assert out["observed_until_frame"] == 450 and out["flags"] == ["grain_lost_after:450"]


def test_a_grain_lost_before_it_germinated_is_censored_where_it_was_lost():
    pred = {"grains": [{"id": "a", "status": "no_emergence_by_end", "length": {"frames": [150, 450, 750]}},
                       {"id": "b", "status": "no_emergence_by_end", "length": {"frames": [150, 450, 750]},
                        "observed_until_frame": 450}]}
    assert onset_intervals(pred) == [(750.0, math.inf), (450.0, math.inf)]


def test_tip_error_is_judged_in_the_field_when_the_prediction_gives_its_drift():
    trace = {"path_xy_ref": [[0, 0], [30.0, 40.0]], "path_xy_view": [[0, 0], [30.0, 40.0]]}
    pred = {"tip": {"frames": [100], "xy": [[20.0, 40.0]]}}
    assert tip_error(pred, 100, trace) == 10.0  # no drift: the grain-following view, as before
    pred["drift"] = {"frames": [100], "xy": [[10.0, 0.0]]}
    assert tip_error(pred, 100, trace) == 0.0   # the grain had moved 10 px: its tip is where the human clicked
