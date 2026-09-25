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
