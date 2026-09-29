"""The learned-evidence experiment's truth rasters line up with what the synthetic renderer draws."""

import json

import numpy as np
import pytest

from sparsetrack import stack
from sparsetrack.grains import annotate_layout


def _grain(size, cx, cy, r):
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - cx, yy - cy)
    return np.where(d < r, 140.0, 180.0) - 40.0 * np.exp(-((d - r) ** 2) / 2.0)


@pytest.fixture
def field(tmp_path):
    size, n_bins = 360, 4
    img = np.full((size, size), 180.0)
    centres = [(80 + 100 * (i % 3), 80 + 100 * (i // 3)) for i in range(9)]
    for cx, cy in centres:
        img = np.minimum(img, _grain(size, cx, cy, 11.0))
    np.save(tmp_path / "bins.npy", np.stack([img] * n_bins).astype(np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 1200, "width": size, "height": size}}))
    census = annotate_layout([{"x": float(cx), "y": float(cy), "r": 11.0, "ring_contrast": 40.0} for cx, cy in centres],
                             (size, size))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": census}))
    return tmp_path


@pytest.mark.parametrize("name,extra", [("v1", {}), ("v4", {"p_rotate": 0.5, "p_sway": 1.0})])
def test_truth_body_covers_what_the_renderer_draws(field, name, extra):
    from prototypes.learned_evidence.truth import frame_truth, tube_point
    from sparsetrack.synth import Scene, preset
    cfg = preset(name, n_frames=60, frames_per_bin=5, onset_bins=(0.5, 2.0), rate_px_per_bin=(4.0, 6.0),
                 noise_sigma=0.0, gain_amp=0.0, drift_sigma=0.0, jump_px=(0.0, 0.0), n_debris=0,
                 maturation_bins=(0.0, 0.0), amplitude=(2.0, 2.0), p_germinate=1.0, seed=2,
                 **({"p_move": 0.0, "p_dock": 0.0, "p_anchor": 0.0, "p_arrive": 0.0, "p_stub": 0.0} if name != "v1" else {}),
                 **extra)
    scene = Scene(field, cfg)
    assert scene.tubes, "the fixture should grow tubes"
    k = cfg.n_frames - 1
    background = scene.background.astype(np.float32)
    drawn = np.abs(scene.render(k).astype(np.float32) - background) > 6.0
    tr = frame_truth(scene, k)
    body = tr["body"].astype(bool)
    near = np.zeros_like(body)
    ys, xs = np.nonzero(body)
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            near[np.clip(ys + dy, 0, body.shape[0] - 1), np.clip(xs + dx, 0, body.shape[1] - 1)] = True
    # every clearly drawn tube pixel lies on (or right beside) the truth body...
    assert drawn[~near].sum() <= 0.02 * max(drawn.sum(), 1)
    # ...and most of the truth body is drawn (young, faint sections may not pass the threshold)
    assert drawn[body].mean() > 0.6
    # each tip sits on its own tube
    for x, y, i, L in tr["tips"]:
        assert (tr["instance"][max(0, int(round(y)) - 2):int(round(y)) + 3,
                               max(0, int(round(x)) - 2):int(round(x)) + 3] == i + 1).any()
        assert tube_point(scene.tubes[i], k, L) == pytest.approx((x, y))


def test_truth_clamps_frames_past_the_movie_end(field):
    from prototypes.learned_evidence.truth import frame_truth
    from sparsetrack.synth import Scene, preset
    cfg = preset("v4", n_frames=60, frames_per_bin=5, onset_bins=(0.5, 1.0), rate_px_per_bin=(4.0, 6.0),
                 n_debris=0, p_sway=1.0, rate_jitter=0.0, pauses=0.0, p_stop=0.0, lag_bins=(0.5, 1.0), seed=3)
    scene = Scene(field, cfg)
    last = frame_truth(scene, cfg.n_frames - 1)
    past = frame_truth(scene, cfg.n_frames + 7)  # the centre of a partial last bin
    assert np.array_equal(last["body"], past["body"])


def test_extra_presets_extend_the_generator():
    from prototypes.learned_evidence.data import synth_config
    from sparsetrack.synth import preset

    assert synth_config("v5w", seed=3).width == (1.3, 2.5)  # wide tubes
    assert synth_config("v5w", seed=3).p_sway == preset("v5", seed=3).p_sway  # otherwise v5
    assert synth_config("v5", seed=3) == preset("v5", seed=3)
