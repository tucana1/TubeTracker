"""Pre-filled review labels: the decoder's reading as the labelling tool's onset and trace answers."""

import numpy as np
import pytest

pytest.importorskip("torch")

from prototypes.learned_evidence import prefill as P  # noqa: E402


def test_simplify_keeps_bends_and_caps_segments():
    straight = np.stack([np.linspace(0, 100, 201), np.zeros(201)], axis=1)
    s = P.simplify(straight)
    assert s[0].tolist() == [0.0, 0.0] and s[-1].tolist() == [100.0, 0.0]
    assert np.hypot(*np.diff(s, axis=0).T).max() <= 25.0 + 1e-9
    corner = np.vstack([straight[:101], np.stack([np.full(100, 50.0), np.linspace(0.5, 50, 100)], axis=1)])
    assert any(np.allclose(p, [50.0, 0.0]) for p in P.simplify(corner))  # the bend is a click


def test_to_length_cuts_or_carries_on():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cut = P.to_length(pts, 15.0)
    assert np.allclose(cut[-1], [10.0, 5.0]) and len(cut) == 3
    longer = P.to_length(pts, 25.0)
    assert np.allclose(longer[-1], [10.0, 15.0])
    assert np.isclose(np.hypot(*np.diff(P.to_length(pts, 7.5), axis=0).T).sum(), 7.5)


def test_onset_answers():
    fpb = 300
    res = {"status": "emerged_within", "onset_frame": 30 * fpb + fpb // 2}
    assert P.onset_body(res, fpb) == {"verdict": "emerged_within", "first_visible_bin": 30, "last_absent_bin": 29}
    assert P.onset_body({"status": "emerged_at_start"}, fpb) == {"verdict": "emerged_at_start"}
    assert P.onset_body({"status": "no_emergence_by_end"}, fpb) == {"verdict": "no_emergence_by_end"}


def test_trace_answer_is_the_reported_length_in_the_tools_view():
    rs, n = 3, 20
    lengths = [0.0] * 8 + [12.0] * (n - 8)
    path = [[100.0, 50.0], [100.0, 60.0], [100.0, 80.0]]  # rim, then along the tube
    res = {"length": {"px": lengths}, "_paths": {10: path}}
    follow = np.zeros((rs + n, 2))
    follow[rs + 12] = [2.0, -1.0]  # the grain has drifted: the tool's view is the reference minus that
    body = P.trace_body(res, rs + 12, rs, follow)  # no path of its own at 12: borrows bin 10's
    pts = np.asarray(body["points"])
    assert body["state"] == "full" and np.allclose(pts[0], [98.0, 51.0])
    assert np.isclose(np.hypot(*np.diff(pts, axis=0).T).sum(), 12.0)
    assert P.trace_body(res, rs + 5, rs, follow)["state"] == "no_tube"  # not grown yet


def test_a_staircase_path_is_measured_as_the_tool_measures_the_trace():
    # a pixel path at 22.5 deg: through pixel centres it is 8% longer than the line it steps along
    from skimage.draw import line

    rr, cc = line(0, 0, 38, 92)
    path = np.c_[cc + 100.0, rr + 50.0]
    stair = np.hypot(*np.diff(path, axis=0).T).sum()
    straight = float(np.hypot(92, 38))
    assert stair > 1.07 * straight
    res = {"length": {"px": [0.0, 90.0]}, "_paths": {1: path.tolist()}}
    pts = np.asarray(P.trace_body(res, 1, 0, np.zeros((2, 2)))["points"])
    assert np.isclose(np.hypot(*np.diff(pts, axis=0).T).sum(), 90.0, atol=0.05)  # the length the model reported
    assert np.hypot(*(pts[-1] - pts[0])) == pytest.approx(90.0, abs=0.5)  # along the line, not its staircase


def test_refuses_to_overwrite_or_write_into_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the repository
    out = tmp_path / "review_labels.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="corrections"):
        P.main(["--field", str(tmp_path), "--work", str(tmp_path), "--out", str(out)])
    with pytest.raises(SystemExit, match="benchmark"):
        P.main(["--field", str(tmp_path), "--work", str(tmp_path), "--out", "benchmark/labels/x.json"])
    # a folder called benchmark above the repository is no reason to refuse
    elsewhere = tmp_path / "benchmark" / "repo"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    with pytest.raises(SystemExit, match="run pipeline.py"):
        P.main(["--field", str(tmp_path), "--work", str(elsewhere / "runs" / "w")])


def test_decoder_paths_are_in_the_tools_reference_coordinates(tmp_path):
    """A straight tube at frame column 100, rows 60-140, below a grain at (100.5, 50.5): the path starts on
    the rim where the tube leaves the grain and runs along the tube (pixel centres at i + 0.5)."""
    import json

    from sparsetrack import stack
    from sparsetrack.render import Renderer

    from prototypes.learned_evidence.reach import reach_grain

    size, n = 240, 8
    yy, xx = np.mgrid[0:size, 0:size]
    grain_img = np.where(np.hypot(xx - 100.0, yy - 50.0) < 9, 120.0, 180.0)
    img = np.stack([grain_img] * n)
    prob = np.zeros((n, size, size))
    for b in range(3, n):
        img[b, 60:141, 99:102] = 150.0
        prob[b, 57:141, 99:102] = 16.0  # P = 1 (x16), touching the grain's ring
    for name, arr in (("img", img), ("prob", prob)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "bins.npy", arr.astype(np.float16))
        (d / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n, "shifts": [[0.0, 0.0]] * n,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * n, "width": size, "height": size}}))
    RP, R_img = Renderer(*stack.load(tmp_path / "prob")), Renderer(*stack.load(tmp_path / "img"))
    meta = stack.load(tmp_path / "prob")[1]
    grain = {"id": "g1", "x": 100.5, "y": 50.5, "r": 9.0}
    res = reach_grain(RP, R_img, meta, grain, [], half=100, paths=(6,))
    path = np.asarray(res["_paths"][6])
    assert np.allclose(path[0], [100.5, 59.5], atol=1.0)  # on the rim, where the tube leaves
    assert np.abs(path[:, 0] - 100.5).max() <= 1.0  # along the tube's centre column (pixel 100 -> 100.5)
    assert abs(path[-1, 1] - 140.5) <= 2.0  # to its far end


def test_a_held_length_takes_the_path_seen_at_that_length_and_is_not_invented():
    rs, n = 0, 12
    long_path = [[100.0, 50.0], [100.0, 90.0]]    # bin 5: the tube seen 40 px long
    faded = [[100.0, 50.0], [104.0, 58.0]]         # bin 9: the evidence fades, 9 px read, pointing elsewhere
    res = {"length": {"px": [0.0] * 3 + [20.0, 30.0, 40.0] + [40.0] * 6}, "_paths": {5: long_path, 9: faded}}
    pts = np.asarray(P.trace_body(res, 9, rs, np.zeros((n, 2)))["points"])
    assert np.allclose(pts[-1], [100.0, 90.0])  # the path from bin 5, not the faded one stretched 31 px
    only_faded = {"length": res["length"], "_paths": {9: faded}}
    short = np.asarray(P.trace_body(only_faded, 9, rs, np.zeros((n, 2)))["points"])
    assert P.arc(short) <= np.hypot(4, 8) + 5.0 + 1e-6  # nothing better: left short, extended by 5 px at most


def test_after_the_grain_left_the_proposal_is_unsure():
    res = {"length": {"px": [0.0, 10.0, 20.0, 30.0]}, "_paths": {3: [[100.0, 50.0], [100.0, 80.0]]}}
    body = P.trace_body(res, 3, 0, np.zeros((4, 2)), unsure=True)
    assert body["state"] == "unsure" and len(body["points"]) >= 2  # the points stay for the reviewer
    assert P.left_at(["burst_after:900", "no_grain_after:1350"]) == 1350 and P.left_at([]) is None


def test_a_spell_away_makes_only_its_own_proposals_unsure():
    def centre(b):
        return b * 300 + 150
    back = {"flags": ["gone:900-1200"], "gone_bins": [3, 4]}  # away in bins 3-4, then back
    assert [P.away(back, b, 0, centre) for b in (2, 3, 4, 5, 9)] == [False, True, True, False, False]
    left = {"flags": ["no_grain_after:1350"]}  # gone for good from bin 4
    assert [P.away(left, b, 0, centre) for b in (3, 4, 9)] == [False, True, True]
