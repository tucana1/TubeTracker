"""The probability movie next to a cache is rebuilt when the network file changes."""

import json

import numpy as np
import pytest

from sparsetrack import stack


def test_a_changed_network_gets_a_fresh_probability_movie(tmp_path):
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
    model = tmp_path / "tubes.pt"

    def save(seed):
        torch.manual_seed(seed)
        net = learned._unet((8, 16))
        torch.save({"state": net.state_dict(), "widths": (8, 16)}, model)

    save(0)
    out = learned.prob_cache(cache, model, log=lambda *a: None)
    first = np.load(out / "bins.npy").copy()
    assert learned.prob_cache(cache, model, log=lambda *a: None) == out  # the same network: used again
    save(1)  # retrained, same file name
    notes = []
    learned.prob_cache(cache, model, log=notes.append)
    assert any("building it again" in n for n in notes) and not np.array_equal(np.load(out / "bins.npy"), first)
