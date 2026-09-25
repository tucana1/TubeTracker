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


def test_anchor_reads_a_curl_lying_back_on_its_grain_from_its_own_exit(tmp_path):
    """A tube leaves its grain (r 9) straight down, then turns and comes back up to touch the grain's rim
    at the same distance as its base: measured from both contacts, the reading ends half-way along."""
    import json

    from sparsetrack import stack

    from prototypes.learned_evidence import reach

    n, size = 10, 200
    img = np.full((n, size, size), 180.0)
    yy, xx = np.mgrid[0:size, 0:size]
    img[:, np.hypot(xx - 100.0, yy - 50.0) < 9] = 120.0
    prob = np.zeros((n, size, size))
    prob[2:4, 61:70, 99:102] = 16.0  # a young tube, straight down from the rim
    for b in range(4, n):
        prob[b, 61:86, 99:102] = 16.0  # down
        prob[b, 83:86, 91:102] = 16.0  # left
        prob[b, 58:86, 91:94] = 16.0  # and back up to the rim, as far from the grain's centre as its base
    for name, bins in (("cache", img), ("prob", prob)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "bins.npy", bins.astype(np.float16))
        (d / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n, "shifts": [[0.0, 0.0]] * n,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * n, "width": size, "height": size}}))
        (d / "grains.json").write_text(json.dumps({"grains": [{"id": "g1", "x": 100.5, "y": 50.5, "r": 9.0}]}))
    kw = dict(log=lambda *a: None, half=80)
    both = reach.analyze(tmp_path / "prob", tmp_path / "cache", **kw)["grains"][0]["raw_reach_px"]
    own = reach.analyze(tmp_path / "prob", tmp_path / "cache", anchor=True, **kw)["grains"][0]["raw_reach_px"]
    assert both[-1] < 40 and own[-1] > 55  # about 60 px along the tube: down 24, across 8, up 27
    assert own[3] == pytest.approx(both[3])  # the young tube: the same reading either way
