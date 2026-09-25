"""The probability cache: used again for the model that built it, built again for any other."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sparsetrack import stack  # noqa: E402

from prototypes.learned_evidence import evaluate  # noqa: E402
from prototypes.learned_evidence.model import UNet  # noqa: E402


@pytest.fixture
def image_cache(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "cache"
    path.mkdir()
    np.save(path / "bins.npy", (150 + 20 * rng.standard_normal((6, 64, 64))).astype(np.float16))
    (path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": 6, "shifts": [[0.0, 0.0]] * 6,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 1800, "width": 64, "height": 64}}))
    (path / "grains.json").write_text(json.dumps({"grains": []}))
    return path


def _net(seed):
    torch.manual_seed(seed)
    return UNet(widths=(4, 8)).eval()


def test_a_cache_is_used_again_only_by_the_model_that_built_it(image_cache, tmp_path):
    out, notes = tmp_path / "prob", []
    a, b = _net(0), _net(1)
    evaluate.prob_cache(image_cache, a, out, log=notes.append)
    built = np.load(out / "bins.npy").copy()
    assert json.loads((out / "meta.json").read_text())["model_sha1"] == evaluate.fingerprint(a)
    notes.clear()
    evaluate.prob_cache(image_cache, _net(0), out, log=notes.append)  # the same weights, loaded again
    assert not notes
    evaluate.prob_cache(image_cache, b, out, log=notes.append)  # another model: its own evidence
    assert any("another model" in n for n in notes)
    assert json.loads((out / "meta.json").read_text())["model_sha1"] == evaluate.fingerprint(b)
    assert not np.array_equal(np.load(out / "bins.npy"), built)


def test_a_cache_from_before_fingerprints_is_built_again(image_cache, tmp_path):
    out, notes = tmp_path / "prob", []
    evaluate.prob_cache(image_cache, _net(0), out, log=notes.append)
    meta = json.loads((out / "meta.json").read_text())
    del meta["model_sha1"]
    (out / "meta.json").write_text(json.dumps(meta))
    evaluate.prob_cache(image_cache, _net(0), out, log=notes.append)
    assert any("another model" in n for n in notes) and "model_sha1" in json.loads((out / "meta.json").read_text())
