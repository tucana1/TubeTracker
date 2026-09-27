"""Fused evidence: the base network's map, with the thick-tube network's where that one marks a wide structure."""

import json

import numpy as np
import pytest

from sparsetrack import stack

from prototypes.learned_evidence import fuse
from prototypes.learned_evidence.evaluate import SCALE_P


def _maps():
    base = np.zeros((96, 96), np.float16)
    thick = np.zeros((96, 96), np.float16)
    base[10:13, 5:90] = 0.9 * SCALE_P          # a thin tube the base network sees (3 px)
    thick[10:13, 5:90] = 0.8 * SCALE_P         # ...and the thick network too
    thick[30:33, 5:90] = 0.9 * SCALE_P         # a thin line only the thick network marks: not added
    thick[50:60, 5:90] = 0.95 * SCALE_P        # a 10 px band only the thick network marks: added
    base[52:54, 40:50] = 0.99 * SCALE_P        # inside the band the larger probability is kept
    thick[70:82, 5:90] = 0.9 * SCALE_P         # a hollow band: two 4 px walls round a 4 px hole
    thick[74:78, 8:87] = 0.0
    return base, thick


def test_wide_structures_are_added_thin_ones_stay_the_base_networks():
    base, thick = _maps()
    out = fuse.fuse(base, thick)
    assert out.dtype == base.dtype
    assert np.array_equal(out[10:13], base[10:13])                   # thin: the base network's
    assert not (out[30:33] > 0.5 * SCALE_P).any()                     # thin line of the thick network: left out
    assert (out[50:60, 10:85] > 0.5 * SCALE_P).all()                  # band: the thick network's
    assert np.all(out[52:54, 40:50] == base[52:54, 40:50])            # max of the two inside the gate
    walls = np.r_[70:74, 78:82]
    assert (out[walls, 10:85] > 0.5 * SCALE_P).all()                  # hollow band: both walls (hole filled for the gate)
    assert not (out[75:77, 10:85] > 0.5 * SCALE_P).any()              # ...and the hole itself stays a hole
    assert not fuse.fuse(base, thick, fill=0)[walls, 10:85].any()      # without the fill the 4 px walls are thin


def test_the_radius_sets_the_width_that_counts_as_wide():
    thick = np.zeros((64, 64), np.float16)
    thick[20:26, 5:60] = SCALE_P  # 6 px wide
    assert not fuse.wide_gate(thick, radius=3.0).any()
    assert fuse.wide_gate(thick, radius=2.5)[22, 30]


def _cache(path, bins, sha="x"):
    path.mkdir()
    np.save(path / "bins.npy", bins.astype(np.float16))
    n = bins.shape[0]
    (path / "meta.json").write_text(json.dumps({"schema": stack.SCHEMA, "frames_per_bin": 1, "n_bins": n,
                                                "shifts": [[0.0, 0.0]] * n, "model_sha1": sha,
                                                "evidence": "learned tube probability x 16.0"}))
    (path / "grains.json").write_text(json.dumps({"grains": [{"id": "g001", "x": 10.0, "y": 10.0, "r": 5.0}]}))
    return path


def test_fused_cache_is_a_cache_of_its_own_used_again_only_for_the_same_inputs(tmp_path):
    base, thick = _maps()
    b = _cache(tmp_path / "base", np.stack([base, base]), "aaaa")
    t = _cache(tmp_path / "thick", np.stack([thick, np.zeros_like(thick)]), "bbbb")
    notes = []
    out = fuse.fused_cache(b, t, tmp_path / "fused", log=notes.append)
    bins, meta = stack.load(out)
    assert bins.shape == (2, 96, 96) and meta["n_bins"] == 2
    assert np.array_equal(bins[0], fuse.fuse(base, thick)) and np.array_equal(bins[1], base)
    assert "model_sha1" not in meta  # not one network's evidence (trace_once.find_cache must not take it for one)
    assert meta["fusion"]["base_sha1"] == "aaaa" and meta["fusion"]["thick_sha1"] == "bbbb"
    assert json.loads((out / "grains.json").read_text())["grains"][0]["id"] == "g001"
    notes.clear()
    fuse.fused_cache(b, t, tmp_path / "fused", log=notes.append)
    assert not notes  # used again
    fuse.fused_cache(b, t, tmp_path / "fused", log=notes.append, radius=2.5)
    assert any("other evidence" in n for n in notes)
    assert stack.load(out)[1]["fusion"]["radius"] == 2.5


def test_caches_of_different_movies_are_refused(tmp_path):
    b = _cache(tmp_path / "base", np.zeros((2, 32, 32)))
    t = _cache(tmp_path / "thick", np.zeros((3, 32, 32)))
    with pytest.raises(ValueError):
        fuse.fused_cache(b, t, tmp_path / "fused")


def test_the_per_bin_decoder_reads_a_wide_hollow_tube_from_the_fused_cache(tmp_path):
    """An 11 px hollow tube growing 3 px per bin: the base network marks only its first 8 px, the thick network all
    of it. On the fused cache the decoder reads it as on the thick network's; a thin tube of the base network's is
    read as before."""
    from prototypes.learned_evidence import reach

    size, n, r = 240, 30, 9.0
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.where((np.hypot(xx - 60.0, yy - 50.0) < r) | (np.hypot(xx - 180.0, yy - 50.0) < r), 120.0, 180.0)
    img = np.repeat(img[None], n, axis=0)
    base = np.zeros((n, size, size))
    thick = np.zeros((n, size, size))
    for b in range(5, n):
        end = 60 + 3 * (b - 4)
        thick[b, 57:end, 55:66] = SCALE_P  # the wide tube (11 px) below the first grain...
        thick[b, 61:end - 3, 59:62] = 0.0  # ...hollow, closed at both ends
        base[b, 57:65, 55:66] = SCALE_P  # the base network sees its first 8 px only
        base[b, 57:end, 179:182] = SCALE_P  # a thin tube below the second grain, the base network's alone
    grains = [{"id": "g001", "x": 60.5, "y": 50.5, "r": r, "isolated": True},
              {"id": "g002", "x": 180.5, "y": 50.5, "r": r, "isolated": True}]

    def cache(path, bins):
        path.mkdir()
        np.save(path / "bins.npy", bins.astype(np.float16))
        (path / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n, "shifts": [[0.0, 0.0]] * n,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * n, "width": size, "height": size}}))
        (path / "grains.json").write_text(json.dumps({"grains": grains}))
        return path

    field = cache(tmp_path / "cache", img)
    fused = fuse.fused_cache(cache(tmp_path / "base", base), cache(tmp_path / "thick", thick), tmp_path / "fused",
                             log=lambda *a: None)

    def final(pc):
        pred = reach.analyze(pc, field, log=lambda *a: None, ghosts=False, burst=True, vmax=4.0)
        return {g["id"]: g["final_length_px"] for g in pred["grains"]}

    on_base, on_thick, on_fused = final(tmp_path / "base"), final(tmp_path / "thick"), final(fused)
    assert on_base["g001"] == 0.0 and on_thick["g001"] > 60 and abs(on_fused["g001"] - on_thick["g001"]) <= 2.0
    assert on_fused["g002"] == on_base["g002"] > 60 and on_thick["g002"] == 0.0


def test_evidence_takes_the_dev_tests_thick_cache_when_that_network_built_it(tmp_path, monkeypatch):
    from prototypes.learned_evidence import evaluate, model
    base, thick = _maps()
    b = _cache(tmp_path / "base", np.stack([base, base]), sha="thin")
    assert fuse.evidence(b, tmp_path / "field", tmp_path / "work", None) == b  # no thick network: the model's own
    dev = _cache(tmp_path / "dev_thick", np.stack([thick, thick]), sha="thick")
    built = []

    def prob_cache(field, net, out, log=print):
        built.append(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        return _cache(out, np.stack([thick, thick]), sha="thick")

    monkeypatch.setattr(model, "load", lambda path: "thick net")
    monkeypatch.setattr(evaluate, "prob_cache", prob_cache)
    monkeypatch.setattr(evaluate, "fingerprint", lambda net: "thick")
    quiet = dict(log=lambda *a: None)
    out = fuse.evidence(b, tmp_path / "field", tmp_path / "work", "thick.pt", tag="_x", reuse=[dev], **quiet)
    assert out == tmp_path / "work" / "prob_fused_x" and not built  # the dev test's cache, not built again
    assert json.loads((out / "meta.json").read_text())["fusion"]["thick"] == str(dev)
    assert np.array_equal(np.load(out / "bins.npy")[0], fuse.fuse(base, thick))
    monkeypatch.setattr(evaluate, "fingerprint", lambda net: "another")  # the dev test's was another network's
    fuse.evidence(b, tmp_path / "field", tmp_path / "work", "thick.pt", tag="_x", reuse=[dev], **quiet)
    assert built == [tmp_path / "work" / "prob_thick"]
    built.clear()  # the same network, but the dev test's cache is of another preparation of the movie
    monkeypatch.setattr(evaluate, "fingerprint", lambda net: "thick")
    (tmp_path / "field").mkdir()
    (tmp_path / "field" / "meta.json").write_text(json.dumps({"n_bins": 2, "created": "2026-09-28T10:00:00"}))
    fuse.evidence(b, tmp_path / "field", tmp_path / "work2", "thick.pt", tag="_x", reuse=[dev], **quiet)
    assert built == [tmp_path / "work2" / "prob_thick"]


def test_readings_are_recorded_and_records_from_before_mean_the_models_own_evidence_without_continuity():
    assert fuse.THICK.exists() and fuse.DEFAULT == {"thick_model": str(fuse.THICK), "continuity": "path"}
    assert fuse.same_reading({}, fuse.reading()) and not fuse.same_reading({}, fuse.DEFAULT)
    assert fuse.same_reading({"end_px": 1.0, "thick_model": "/elsewhere/unet_thick_b3.pt", "continuity": "path"},
                             fuse.DEFAULT)  # where the network lies does not matter
    assert not fuse.same_reading(fuse.reading(fuse.THICK, False), fuse.DEFAULT)
    assert fuse.describe({}) == "the model's own evidence without continuity"
    assert fuse.describe(fuse.DEFAULT) == "fused evidence with continuity"
