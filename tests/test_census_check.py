"""The census check (sparsetrack.census_check): features of census "grains" that tell grains from debris."""

import json

import numpy as np
import pytest

from sparsetrack import census_check, stack
from sparsetrack.grains import _ring_scores, annotate_layout


def _grain_image(size=200, moved=0.0):
    """A ringed grain (dark rim, lighter inside), a dark filled blob, and a faint smudge that has moved off by
    ``moved`` px (a passing thing), on a flat background with a little noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.full((size, size), 170.0)
    d1 = np.hypot(xx - 50, yy - 50)
    img -= 80 * np.exp(-((d1 - 12) ** 2) / 3.0) + 5 * (d1 < 9)  # grain: dark rim, a little darker inside
    d2 = np.hypot(xx - 150, yy - 50)
    img -= 95 / (1 + np.exp((d2 - 11) / 0.8))  # debris: dark all through
    d3 = np.hypot(xx - 100 - moved, yy - 150)
    img -= 30 * np.exp(-((d3 - 11) ** 2) / 12.0)  # a faint smudge
    return img + rng.normal(0, 0.5, img.shape)


def _cache(tmp_path, n_bins=24):
    bins = np.stack([_grain_image(moved=0.0 if b < 6 else 25.0) for b in range(n_bins)]).astype(np.float16)
    np.save(tmp_path / "bins.npy", bins)
    meta = {"schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
            "movie": {"name": "synthetic.mp4", "size_bytes": 1, "n_frames": n_bins * 300, "width": 200,
                      "height": 200}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    ref = bins[:3].astype(np.float64).mean(axis=0)
    found = []
    for x, y in ((50.0, 50.0), (150.0, 50.0), (100.0, 150.0)):
        ring, body = _ring_scores(ref, x, y, 12.0)
        found.append({"x": x, "y": y, "r": 12.0, "ring_contrast": round(ring, 2), "body_contrast": round(body, 2)})
    grains = annotate_layout(found, (200, 200))
    (tmp_path / "grains.json").write_text(json.dumps({"reference_bins": [0, 1, 2], "flatfield": False,
                                                      "grains": grains}))
    return tmp_path, {(g["x"], g["y"]): g["id"] for g in grains}


def test_features_tell_a_ringed_grain_from_dark_debris_and_a_passing_smudge(tmp_path):
    cache, ids = _cache(tmp_path)
    f = census_check.features(cache)
    grain, blob, smudge = f[ids[(50.0, 50.0)]], f[ids[(150.0, 50.0)]], f[ids[(100.0, 150.0)]]
    assert blob["disc_dark"] > grain["disc_dark"] + 0.5  # dark all through, not just at its rim
    assert blob["fill"] > grain["fill"]
    assert grain["stay"] > 0.9 and blob["stay"] > 0.9  # both still there after the reference bins
    assert smudge["stay"] < 0.5  # gone from its place
    assert set(census_check.FEATURES) <= set(grain)


def test_flagged_grains_are_those_at_or_above_the_models_threshold(tmp_path):
    cache, ids = _cache(tmp_path)
    toy = {"features": ["disc_dark"], "coef": [4.0], "intercept": 0.0, "mean": [1.0], "scale": [0.5],
           "threshold": 0.5}
    fl = census_check.flagged(cache, model=toy)
    assert ids[(150.0, 50.0)] in fl and ids[(50.0, 50.0)] not in fl
    assert all(v >= 0.5 for v in fl.values())


def test_the_fitted_model_scores_dark_debris_above_a_ringed_grain(tmp_path):
    cache, ids = _cache(tmp_path)
    if not census_check.MODEL:
        pytest.skip("no fitted model")
    p = census_check.check(cache)
    assert p[ids[(150.0, 50.0)]] > p[ids[(50.0, 50.0)]]


def test_the_census_check_is_off_by_default():
    from sparsetrack.analyze import Params
    assert Params().census_check is False


def test_the_analysis_flags_likely_not_grains_and_leaves_them_out_of_the_population(tmp_path, monkeypatch):
    cache, ids = _cache(tmp_path)
    toy = {"features": ["disc_dark"], "coef": [4.0], "intercept": 0.0, "mean": [1.0], "scale": [0.5],
           "threshold": 0.5}
    monkeypatch.setattr(census_check, "MODEL", toy)
    from sparsetrack.analyze import Params, analyze
    p = dict(reader="change", half=60, vmax_auto=False)
    off = analyze(cache, tmp_path / "off", params=Params(**p), log=lambda *a: None)
    assert not any(f.startswith(census_check.FLAG) for g in off["grains"] for f in g["flags"])
    on = analyze(cache, tmp_path / "on", params=Params(census_check=True, **p), log=lambda *a: None)
    flags = {g["id"]: g["flags"] for g in on["grains"]}
    blob, grain = ids[(150.0, 50.0)], ids[(50.0, 50.0)]
    assert any(f.startswith(census_check.FLAG + ":") for f in flags[blob])
    assert not any(f.startswith(census_check.FLAG) for f in flags[grain])
    # still read and shown, but with the clumped / edge grains, not in the isolated population
    html = (tmp_path / "on" / "index.html").read_text()
    assert html.index(f"id='{blob}'") > html.index("Clumped / edge grains") > html.index(f"id='{grain}'")
