"""Knocked grains found again (track.follow ``refind``), and the labelling tool following them (Bench.follow): the
tracker wherever it has the grain, and from wherever the annotator says the grain is."""

import json
import math
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sparsetrack import stack
from sparsetrack.analyze import Params, followed_drift
from sparsetrack.bench.server import Bench, make_handler
from sparsetrack.grains import annotate_layout
from sparsetrack.render import Renderer
from sparsetrack.track import FollowConfig, follow, turned


def _grain(xx, yy, x, y, r, turn, tube_len=0.0):
    """A dark-rimmed grain with a dark and a bright patch inside and a dark crescent on its rim (so that a turn
    shows, as the interiors of real grains do) and, optionally, a tube leaving it; turned by ``turn`` degrees
    (clockwise in the image)."""
    d = np.hypot(xx - x, yy - y)
    phi = np.arctan2(yy - y, xx - x)
    a = math.radians(30.0 + turn)
    crescent = 0.5 + 0.5 * np.cos(phi - a - math.pi / 2)
    img = -(35 + 70 * crescent ** 3) * np.exp(-((d - r) ** 2) / 3.0)
    for k, (amp, rad) in enumerate(((-80.0, 0.5), (60.0, 0.45))):
        aa = a + k * 2.3
        sx, sy = x + rad * r * math.cos(aa), y + rad * r * math.sin(aa)
        img += amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 2.5 ** 2))
    if tube_len > 0:
        u = np.array([math.cos(math.radians(160.0 + turn)), math.sin(math.radians(160.0 + turn))])
        along = (xx - x) * u[0] + (yy - y) * u[1] - r
        across = -(xx - x) * u[1] + (yy - y) * u[0]
        img -= 70 * ((along > 0) & (along < tube_len)) * np.exp(-across ** 2 / 2.0)
    return img


def _knock_movie(tmp_path=None, n_bins=40, size=260, knock_bin=20, jump=(-26.0, 6.0), turn=70.0, r=12.0,
                 start=(140.0, 120.0), seed=0, gone_from=None, back_at=None):
    """A grain with its tube, knocked at ``knock_bin``: it jumps by ``jump`` and turns by ``turn``. With
    ``gone_from``/``back_at`` it is out of view in between (the tracker loses it) and back at the jumped place."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    texture = gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 40 + 175
    frames = []
    for b in range(n_bins):
        img = texture.copy()
        knocked = b >= knock_bin
        x, y = (start[0] + jump[0], start[1] + jump[1]) if knocked else start
        if gone_from is None or not (gone_from <= b < back_at):
            img += _grain(xx, yy, x, y, r, turn if knocked else 0.0, tube_len=min(30.0, 1.5 * max(0, b - 5)))
        frames.append(img + rng.normal(0, 0.6, img.shape))
    bins = np.stack(frames).astype(np.float16)
    meta = {"schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
            "movie": {"name": "knock.mp4", "size_bytes": 1, "n_frames": n_bins * 300, "width": size, "height": size}}
    if tmp_path is not None:
        np.save(tmp_path / "bins.npy", bins)
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "grains.json").write_text(json.dumps({"grains": annotate_layout(
            [{"x": start[0], "y": start[1], "r": r}], (size, size))}))
    return bins, meta


def test_turned_rotates_clockwise_in_the_image():
    t = np.zeros((21, 21), np.float32)
    t[10, 15] = 1.0  # right of the centre
    assert np.unravel_index(int(np.argmax(turned(t, 90.0))), t.shape) == (15, 10)  # below it (y down)


def test_a_knocked_grain_is_found_again_and_carries_its_turn():
    bins, meta = _knock_movie()
    r = Renderer(bins, meta)
    old = follow(r, 140.0, 120.0, 12.0, 0, 40)  # 0.7.0: its look no longer matches, it is lost at the knock
    assert old["lost_from"] == 20
    new = follow(r, 140.0, 120.0, 12.0, 0, 40, cfg=FollowConfig(refind=True))
    assert new["lost_from"] is None and len(new["refound"]) == 1
    assert new["refound"][0]["bin"] in (20, 21) and abs(new["refound"][0]["moved_px"] - math.hypot(26, 6)) < 2.0
    assert np.max(np.hypot(*(new["xy"][24:] - [-26.0, 6.0]).T)) < 1.0
    assert abs(new["angle"][-1] - 70.0) <= 10.0  # its turn, to the search step
    assert np.max(np.hypot(*new["xy"][:20].T)) < 0.5  # nothing changes before the knock


def test_refind_leaves_an_unknocked_track_as_it_was():
    bins, meta = _knock_movie(turn=0.0, jump=(0.0, 0.0))
    r = Renderer(bins, meta)
    a = follow(r, 140.0, 120.0, 12.0, 0, 40)
    b = follow(r, 140.0, 120.0, 12.0, 0, 40, cfg=FollowConfig(refind=True))
    assert np.array_equal(a["xy"], b["xy"]) and b["refound"] == []


def test_refind_does_not_take_a_neighbour_for_a_grain_that_vanished():
    """A grain bursts at bin 25 beside an identical neighbour that stays: searched for by its last look, turned, it is
    not found in the neighbour (still at its place, so not searched) nor anywhere else: lost, as before."""
    size, n_bins = 240, 50
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    texture = gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 40 + 175
    frames = []
    for t in range(n_bins):
        img = texture + _grain(xx, yy, 130.0, 120.0, 12.0, 0.0)
        if t < 25:
            img = img + _grain(xx, yy, 100.0, 120.0, 12.0, 0.0)
        frames.append(img + rng.normal(0, 0.6, img.shape))
    meta = {"shifts": [[0.0, 0.0]] * n_bins, "n_bins": n_bins, "frames_per_bin": 300}
    tr = follow(Renderer(np.stack(frames).astype(np.float16), meta), 100.0, 120.0, 12.0, 0, n_bins,
                [(130.0, 120.0, 12.0)], FollowConfig(refind=True))
    assert tr["lost_from"] == 25 and tr["refound"] == []


def test_followed_drift_passes_the_refind_option_and_its_findings():
    bins, meta = _knock_movie()
    fd = followed_drift(Renderer(bins, meta), meta, {"id": "g", "x": 140.0, "y": 120.0, "r": 12.0}, [],
                        Params(grain_track="follow", track_refind=True))
    assert fd["lost_from"] is None and fd["refound"] and np.isfinite(fd["angle"][-1])
    assert np.max(np.hypot(*(fd["drift"][25:] - [-26.0, 6.0]).T)) < 1.5


# ---------------------------------------------------------------------------- the labelling tool
def test_labelling_views_follow_the_tracker_and_a_review_keeps_the_analysis_frame(tmp_path):
    _knock_movie(tmp_path)
    label = Bench(tmp_path, tmp_path / "labels.json")
    assert label.follow_mode == "label" and label.state()["follow_mode"] == "label"
    off = label.follow("g001")
    assert np.max(np.hypot(*(off[25:] - [-26.0, 6.0]).T)) < 1.5  # knocked, followed: moved less than its diameter
    info = label.follow_info("g001")
    assert info["lost"] == [] and info["lost_from"] is None and info["refound"]
    review = Bench(tmp_path, tmp_path / "labels.json", follow_mode="review")
    # the analysis (Params defaults) loses the knocked grain and holds it where it was last seen
    assert np.max(np.hypot(*review.follow("g001")[25:].T)) < 1.0
    assert review.follow_info("g001")["lost_from"] == 20
    with pytest.raises(ValueError):
        review.set_refind("g001", {"bin": 30, "x": 100.0, "y": 100.0})
    with pytest.raises(ValueError):
        Bench(tmp_path, tmp_path / "labels.json", follow_mode="other")


def test_a_saved_trace_keeps_its_reference_coordinates_whatever_the_view(tmp_path):
    """A trace is stored as clicked plus the view's offset then: the page draws it from those reference coordinates less
    the view's offset now, so an answer given in an older view is still drawn on its tube."""
    _knock_movie(tmp_path)
    bench = Bench(tmp_path, tmp_path / "labels.json")
    off = bench.follow("g001")[30]
    rec = bench.set_trace("g001", {"bin": 30, "state": "full", "points": [[128.0, 120.0], [100.0, 130.0]]})
    assert rec["view_offset"] == [round(float(v), 2) for v in off]
    assert np.allclose(np.asarray(rec["path_xy_ref"]) - rec["view_offset"], rec["path_xy_view"], atol=0.02)


def test_the_grain_is_here_is_kept_and_followed_from_that_bin(tmp_path):
    """The grain is out of view for bins 18-29 and back 26 px away: the tracker loses it (the tool says so); the
    annotator places it at bin 30; the tool keeps that in the labels file and follows the grain from there."""
    _knock_movie(tmp_path, knock_bin=18, gone_from=18, back_at=30, turn=40.0)
    labels = tmp_path / "labels.json"
    bench = Bench(tmp_path, labels)
    info = bench.follow_info("g001")
    assert info["lost"] and info["lost"][0][0] == 18 and info["lost_from"] == 18
    held = np.asarray(info["offsets"][30])
    assert np.hypot(*held) < 1.0  # held where it was last seen
    # the annotator clicks the grain's centre in bin 30's view (view = reference - offset)
    info = bench.set_refind("g001", {"bin": 30, "x": 114.0 - held[0], "y": 126.0 - held[1]})
    assert [r["bin"] for r in info["refinds"]] == [30] and info["lost"] == [[18, 30, "gap"]]
    assert info["lost_from"] is None
    assert np.max(np.hypot(*(np.asarray(info["offsets"][30:]) - [-26.0, 6.0]).T)) < 1.0
    doc = json.loads(labels.read_text())
    rec = doc["labels"]["g001"]["refinds"]["30"]
    assert rec["xy_ref"] == [114.0, 126.0] and rec["offset"] == [-26.0, 6.0] and rec["source_frame"] == 9150
    assert '"refind"' in labels.with_suffix(".journal.jsonl").read_text()
    again = Bench(tmp_path, labels)  # kept: a reopened tool follows it from there too
    assert np.max(np.hypot(*(again.follow("g001")[30:] - [-26.0, 6.0]).T)) < 1.0
    info = again.set_refind("g001", {"bin": 30, "clear": True})  # taken back
    assert info["refinds"] == [] and info["lost_from"] == 18
    assert "refinds" not in json.loads(labels.read_text())["labels"]["g001"]


def test_follow_and_refind_over_http(tmp_path):
    _knock_movie(tmp_path, knock_bin=18, gone_from=18, back_at=30, turn=40.0)
    bench = Bench(tmp_path, tmp_path / "labels.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(bench))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        info = json.load(urllib.request.urlopen(base + "/api/follow/g001"))
        assert len(info["offsets"]) == 40 and info["lost_from"] == 18 and info["mode"] == "label"
        req = urllib.request.Request(base + "/api/refind/g001", method="POST",
                                     headers={"Content-Type": "application/json"},
                                     data=json.dumps({"bin": 30, "x": 114.0, "y": 126.0}).encode())
        info = json.load(urllib.request.urlopen(req))
        assert info["refinds"][0]["bin"] == 30 and info["lost_from"] is None
        bad = urllib.request.Request(base + "/api/refind/g001", method="POST",
                                     headers={"Content-Type": "application/json"},
                                     data=json.dumps({"bin": 400, "x": 1.0, "y": 1.0}).encode())
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        server.shutdown()
        server.server_close()
