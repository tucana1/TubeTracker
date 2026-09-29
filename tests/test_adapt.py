"""Adapting the tube network to a movie: sizing, finding the adapted network, and fresh probability movies."""

import json

import numpy as np
import pytest

from sparsetrack import stack
from sparsetrack.adapt import MODEL_NAME, adapted_model, sized_like


def test_synthetic_movies_are_sized_like_the_movie():
    ld, m2 = sized_like({"n_bins": 176}), sized_like({"n_bins": 351})
    assert ld["n_frames"] == 176 * 25 and m2["n_frames"] == 351 * 25
    assert ld["onset_bins"][1] == pytest.approx(123.2) and m2["onset_bins"][1] == pytest.approx(245.7)
    assert 120.0 <= ld["max_length"] <= 160.0 and m2["max_length"] == pytest.approx(280.8)


def test_the_adapted_network_is_found_in_the_run_folder(tmp_path):
    assert adapted_model(tmp_path) is None
    (tmp_path / "adapt").mkdir()
    (tmp_path / "adapt" / MODEL_NAME).write_bytes(b"x")
    assert adapted_model(tmp_path) == tmp_path / "adapt" / MODEL_NAME


def test_a_network_adapted_again_gets_a_fresh_probability_movie(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    np.save(cache / "bins.npy", (150 + 20 * rng.standard_normal((8, 32, 32))).astype(np.float16))
    (cache / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": 8, "shifts": [[0.0, 0.0]] * 8,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 2400, "width": 32, "height": 32}}))
    (cache / "grains.json").write_text(json.dumps({"grains": []}))
    model = tmp_path / MODEL_NAME

    def save(seed):
        torch.manual_seed(seed)
        net = learned._unet((8, 16))
        torch.save({"state": net.state_dict(), "widths": (8, 16)}, model)

    save(0)
    out = learned.prob_cache(cache, model, log=lambda *a: None)
    first = np.load(out / "bins.npy").copy()
    assert learned.prob_cache(cache, model, log=lambda *a: None) == out  # the same network: used again
    save(1)  # adapted again, same file name
    notes = []
    learned.prob_cache(cache, model, log=notes.append)
    assert any("building it again" in n for n in notes) and not np.array_equal(np.load(out / "bins.npy"), first)
