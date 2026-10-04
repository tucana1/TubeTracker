"""The tube network's maps (sparsetrack.learned): options a checkpoint may name (BatchNorm, the local background)
and the probability movie kept next to a cache."""

import json

import numpy as np
import pytest

from sparsetrack import stack


def _images(shape=(96, 112), seed=0):
    rng = np.random.default_rng(seed)
    return [(150 + 12 * rng.standard_normal(shape)).astype(np.float32) for _ in range(3)]


def _save(tmp_path, torch, learned, norm, **fields):
    torch.manual_seed(0)
    net = learned._unet((8, 16), norm=norm)
    if norm == "batch":  # running statistics other than the initial 0 / 1
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.3, 0.3)
                m.running_var.uniform_(0.5, 2.0)
    path = tmp_path / f"{norm}.pt"
    torch.save({"state": net.state_dict(), "widths": (8, 16), **({"norm": norm} if norm != "group" else {}),
                **fields}, path)
    return path


def test_a_checkpoint_without_options_is_read_as_before(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    net = learned.load_model(_save(tmp_path, torch, learned, "group"), device="cpu")
    assert net.bg_px == 0 and isinstance(net.enc[0][1], torch.nn.GroupNorm)
    img, early, late = _images()
    x = (np.stack([img, early, late]) - float(np.median(early))) / learned.IN_SCALE
    with torch.no_grad():
        want = torch.sigmoid(net(torch.from_numpy(x.astype(np.float32))[None]))[0, 0].numpy()
    assert np.array_equal(learned.tube_probability(net, img, early, late), want)


def test_the_local_background_follows_uneven_illumination(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    net = learned.load_model(_save(tmp_path, torch, learned, "batch", bg_px=32), device="cpu")
    assert net.bg_px == 32 and isinstance(net.enc[0][1], torch.nn.BatchNorm2d)
    img, early, late = _images()
    ramp = np.linspace(-40, 40, img.shape[1], dtype=np.float32)[None, :]  # illumination falling across the frame
    lit = [a + ramp for a in (img, early, late)]
    change_local = np.abs(learned.tube_probability(net, *lit) - learned.tube_probability(net, img, early, late))
    net.bg_px = 0  # one median for the whole frame
    change_frame = np.abs(learned.tube_probability(net, *lit) - learned.tube_probability(net, img, early, late))
    inner = (slice(16, -16), slice(16, -16))  # away from the borders the local median sees one side of
    assert change_local[inner].mean() < 0.25 * change_frame[inner].mean()


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
