"""Learned-evidence helpers around SparseTrack: adaptive crop, chunked kymograph, monotone fit, burst cut."""

import numpy as np
import pytest

pytest.importorskip("torch")  # evaluate.py imports the model

from sparsetrack import analyze as A  # noqa: E402


def test_adaptive_crop_reads_edge_paths_again(monkeypatch):
    calls = []

    def fake(renderer, meta, grain, others, p, _settled=False):
        calls.append((grain["id"], p.half))
        side = 2 * p.half
        # "long" runs to the edge of any crop; "short" ends 40 px from the grain
        end = (side - 2.0, p.half) if grain["id"] == "long" else (p.half + 40.0, p.half)
        pts = np.array([[p.half, p.half], end], float)
        return {"id": grain["id"], "flags": [],
                "_diag": (np.zeros((side, side), np.float32), None, None, pts, None, None, p.half - 0.5)}

    monkeypatch.setattr(A, "analyze_grain", fake)
    from prototypes.learned_evidence.evaluate import adaptive_crop

    with adaptive_crop(big=300):
        long_ = A.analyze_grain(None, {}, {"id": "long"}, [], A.Params())
        short = A.analyze_grain(None, {}, {"id": "short"}, [], A.Params())
    assert A.analyze_grain is fake
    assert A.matched_kymograph.__module__ == "sparsetrack.analyze"
    assert long_["flags"] == ["crop_grown:300"]
    assert short["flags"] == []
    # read again once, at the bigger crop, even though that path still ends at the edge
    assert calls == [("long", 150), ("long", 300), ("short", 150)]


def test_chunked_matched_kymograph_reads_long_paths():
    import cv2

    from prototypes.learned_evidence.evaluate import _chunked

    rng = np.random.default_rng(0)
    signed = rng.normal(0, 1, (2, 700, 700)).astype(np.float32)
    pts = np.stack([np.linspace(50, 650, 1200), np.full(1200, 350.0)], axis=1)  # a 600 px path, 0.5 px steps
    normal = np.tile([0.0, 1.0], (len(pts), 1))
    across = np.arange(-3.5, 3.5 + 1e-9, 0.5)
    template = rng.normal(0, 1, (len(pts), len(across))).astype(np.float32)
    angles = np.arange(-45, 45 + 1e-9, 1.5)  # SparseTrack's 61 rotation angles
    with pytest.raises(cv2.error):
        A.matched_kymograph(signed, template, pts, normal, 350.0, across, angles)
    out = _chunked(A.matched_kymograph)(signed, template, pts, normal, 350.0, across, angles)
    assert out.shape == (2, len(angles), len(pts))
    ref = A.matched_kymograph(signed, template, pts, normal, 350.0, across, angles[20:26], lateral_offset=0.0)
    np.testing.assert_array_equal(out[:, 20:26], ref)


def test_monotone_l1_ignores_a_one_bin_flicker():
    from prototypes.learned_evidence.reach import monotone_l1

    raw = np.array([0, 0, 1, 2, 30, 4, 5, 6, 0, 8], float)  # a merge with another tube, then a missed bin
    fit = monotone_l1(raw, vmax=4.0)
    assert np.all(np.diff(fit) >= 0) and np.all(np.diff(fit) <= 4.0)
    assert fit[4] <= 6.0 and fit[-1] == pytest.approx(8.0, abs=0.5)


def test_burst_cut_finds_a_collapse_that_lasts():
    from prototypes.learned_evidence.reach import burst_cut

    grow = np.linspace(0, 40, 30)
    assert burst_cut(np.concatenate([grow, np.zeros(20)])) == 30  # the tube vanished for good
    assert burst_cut(np.concatenate([grow, [0, 0], np.full(18, 41.0)])) is None  # a two-bin gap, then the tube again
    assert burst_cut(np.concatenate([grow, np.full(20, 40.0)])) is None  # it stopped growing, still visible
    assert burst_cut(np.concatenate([np.linspace(0, 5, 30), np.zeros(20)])) is None  # never a tube (< 8 px)


def test_per_grain_csv_with_the_per_bin_run_alone(tmp_path):
    import csv

    from prototypes.learned_evidence.pipeline import write_per_grain

    perbin = {"grains": [{"id": "g1", "x": 10.0, "y": 20.0, "status": "emerged_within", "onset_interval": [150, 450],
                          "final_length_px": 42.0, "burst_frame": None}]}
    write_per_grain(tmp_path, {"perbin": perbin}, 1.0, um_per_px=0.5, s_per_frame=6.0)
    (row,) = list(csv.DictReader(open(tmp_path / "per_grain.csv")))
    assert row["grain"] == "g1" and row["perbin_final_length_um"] == "21.0" and row["perbin_onset_by_min"] == "45.0"


def test_grain_gone_sees_a_grain_leave_but_not_the_light_change():
    from prototypes.learned_evidence.reach import grain_gone

    half, gr, n = 40, 9.0, 30
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    centre = half - 0.5
    rg = np.hypot(xx - centre, yy - centre)
    grain = np.where(rg < gr, 100.0, 180.0)
    stack_ = np.stack([grain.copy() for _ in range(n)])
    ls = np.zeros((n, 2))
    brighter = stack_ * np.linspace(1.0, 1.3, n)[:, None, None]  # the whole movie brightens: not a departure
    assert grain_gone(brighter, ls, centre, gr, rg) is None
    stack_[12:] = 180.0  # from bin 12 the grain has gone
    assert grain_gone(stack_, ls, centre, gr, rg) == 12
    blink = np.stack([grain.copy() for _ in range(n)])
    blink[5:8] = 180.0  # three bins covered, then back: shorter than a run
    assert grain_gone(blink, ls, centre, gr, rg) is None


def test_smooth_length_reads_a_pixel_staircase_as_the_line_it_steps_along():
    from skimage.draw import line

    from prototypes.learned_evidence.reach import smooth_length

    for x1, y1 in ((100, 0), (92, 38), (87, 50), (71, 71)):  # 0, 22.5, 30 and 45 degrees
        rr, cc = line(0, 0, y1, x1)
        pts = np.c_[cc, rr].astype(float)
        stair = np.hypot(*np.diff(pts, axis=0).T).sum()
        assert smooth_length(pts) == pytest.approx(np.hypot(x1, y1), abs=0.01)
        assert stair >= np.hypot(x1, y1) - 1e-9  # through pixel centres: never shorter, up to 8% longer
    bend = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert smooth_length(bend) == pytest.approx(20.0)  # a real bend is kept
    assert smooth_length(bend[:1]) == 0.0


def _grain_movie(tmp_path, n_bins=12):
    """A grain (r 9) at (100.5, 50.5) whose tube (P = 1) runs straight down from 12 px below its centre,
    growing 6 px a bin from bin 2, and a speck touching its rim on the left, nearer its centre."""
    import json

    from sparsetrack import stack

    size = 200
    img = np.full((n_bins, size, size), 180.0)
    yy, xx = np.mgrid[0:size, 0:size]
    img[:, np.hypot(xx - 100.0, yy - 50.0) < 9] = 120.0
    prob = np.zeros((n_bins, size, size))
    for b in range(2, n_bins):
        prob[b, 62:62 + 6 * (b - 1), 99:102] = 16.0
    prob[:, 49:52, 90:92] = 16.0  # the speck: 8.5-10 px from the centre
    paths = []
    for name, bins in (("cache", img), ("prob", prob)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "bins.npy", bins.astype(np.float16))
        (d / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * n_bins, "width": size, "height": size}}))
        (d / "grains.json").write_text(json.dumps({"grains": [{"id": "g1", "x": 100.5, "y": 50.5, "r": 9.0}]}))
        paths.append(d)
    return paths


def test_a_speck_on_the_rim_does_not_hide_the_tube_with_pieces(tmp_path):
    from prototypes.learned_evidence import reach

    cache, prob = _grain_movie(tmp_path)
    kw = dict(log=lambda *a: None, half=100)
    plain = reach.analyze(prob, cache, **kw)["grains"][0]["raw_reach_px"]
    pieces = reach.analyze(prob, cache, pieces=True, **kw)["grains"][0]["raw_reach_px"]
    assert max(plain) < 10  # measured from the speck, the nearest pixels to the grain
    assert pieces[-1] == pytest.approx(3 + 6 * 10, abs=4)  # the tube: from the rim (12 - 9) plus 60 px


def test_memory_holds_a_reading_the_evidence_still_shows(tmp_path):
    from prototypes.learned_evidence import reach

    cache, prob = _grain_movie(tmp_path)
    bins = np.load(prob / "bins.npy").astype(float)
    bins[:, 49:52, 90:92] = 0.0  # no speck here
    bins[8, 62:66, 99:102] = 0.0  # bin 8: the tube's base drops out, so the region leaves the rim
    bins[11, :, :] = 0.0  # bin 11: the whole tube is gone (a burst): nothing is held
    np.save(prob / "bins.npy", bins.astype(np.float16))
    kw = dict(log=lambda *a: None, half=100, pieces=True)
    plain = reach.analyze(prob, cache, **kw)["grains"][0]["raw_reach_px"]
    held = reach.analyze(prob, cache, memory=True, **kw)["grains"][0]["raw_reach_px"]
    assert plain[8] == 0.0 and held[8] == pytest.approx(held[7])  # the rest of the tube is still shown
    assert held[9] > held[8] and held[11] == 0.0  # growth goes on; a tube that is gone is not held
