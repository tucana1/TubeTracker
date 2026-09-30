"""A tube that grows over its grain from a pore facing the camera before it leaves the rim (analyze.over_grain,
Params.over_grain; movie 1: g015, g030)."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

from sparsetrack.analyze import Params, analyze_grain
from sparsetrack.render import Renderer

GX, GY, R, ANGLE = 150.0, 160.0, 13.0, 200.0
PORE = 0.3        # the pore, as a share of the radius from the centre
ONSET, RATE = 12, 1.0  # the tube appears at the pore at bin 12 and grows 1 px per bin: it reaches the rim at ~bin 21


def _over_grain_movie(n_bins=60, size=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    texture = gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 40 + 175
    d = np.hypot(xx - GX, yy - GY)
    grain = -90 * np.exp(-((d - R) ** 2) / 3.0) + 15 * np.exp(-(d ** 2) / (2 * (0.5 * R) ** 2))
    u = np.array([math.cos(math.radians(ANGLE)), math.sin(math.radians(ANGLE))])
    along = (xx - GX) * u[0] + (yy - GY) * u[1] - PORE * R  # from the pore
    across = -(xx - GX) * u[1] + (yy - GY) * u[0]
    bins = []
    for t in range(n_bins):
        L = max(0.0, (t - ONSET + 1) * RATE)
        img = texture + grain
        if L > 0:
            cap = np.where(along > 0, 0.5 * erfc((along - L) / (np.sqrt(2) * 1.2)), 0.0)
            img = img - 40 * cap * np.exp(-across ** 2 / 2.0)  # a dark line, over the grain and beyond it
        bins.append(img + rng.normal(0, 0.6, img.shape))
    meta = {"shifts": [[0.0, 0.0]] * n_bins, "n_bins": n_bins, "frames_per_bin": 300}
    return np.stack(bins).astype(np.float16), meta


def _read(mode):
    bins, meta = _over_grain_movie()
    return analyze_grain(Renderer(bins, meta), meta, {"id": "g001", "x": GX, "y": GY, "r": R}, [],
                         Params(half=100, exit_edge=False, grain_track="phase", over_grain=mode))


def test_the_tube_over_the_grain_is_found_and_read_as_asked():
    rim = _read("off")
    assert rim["status"] == "emerged_within" and "over_grain" not in rim
    b_rim = rim["onset_frame"] // 300
    assert b_rim >= ONSET + 6  # seen only once it reaches the rim
    flag = _read("flag")
    og = flag["over_grain"]
    assert any(f.startswith("tube_over_grain") for f in flag["flags"])
    assert abs(og["onset_frame"] // 300 - ONSET) <= 2 and flag["onset_frame"] == rim["onset_frame"]
    assert 5.0 <= og["inner_px"] <= 12.0  # the pore to the rim: 0.7 of the radius, less the rim band
    assert np.hypot(og["pore_xy"][0] - GX, og["pore_xy"][1] - GY) < 0.6 * R
    onset = _read("onset")
    assert abs(onset["onset_frame"] // 300 - ONSET) <= 2
    assert onset["length"]["px"] == rim["length"]["px"] or max(onset["length"]["px"][:b_rim]) == 0.0
    length = _read("length")
    L = np.array(length["length"]["px"])
    true = np.array([max(0.0, (t - ONSET + 1) * RATE) for t in range(len(L))])
    assert "length_from_pore" in length["flags"] and L[ONSET - 2] == 0.0
    assert np.median(np.abs(L[30:] - true[30:])) < 3.0  # lengths from the pore, as the annotator traced such tubes
    assert np.median(np.abs(np.array(rim["length"]["px"])[30:] - (true[30:] - (1 - PORE) * R))) < 3.0  # from the rim


def test_a_grain_whose_inside_changes_all_at_once_has_no_tube_over_it():
    """The whole disc darkens at once (the grain turns over, its cytoplasm moves) before an ordinary tube leaves the
    rim: nothing grew from a pore to the exit, so the rim onset stands."""
    bins, meta = _over_grain_movie()
    yy, xx = np.mgrid[0:bins.shape[1], 0:bins.shape[2]].astype(np.float64)
    inside = np.hypot(xx - GX, yy - GY) < R - 1.5
    b = bins.astype(np.float64)
    b[15:, inside] -= 30.0  # the body darkens at bin 15
    res = analyze_grain(Renderer(b.astype(np.float16), meta), meta, {"id": "g001", "x": GX, "y": GY, "r": R}, [],
                        Params(half=100, exit_edge=False, grain_track="phase", over_grain="onset"))
    og = res.get("over_grain")
    assert og is None or og["onset_frame"] // 300 >= 15 - 2
