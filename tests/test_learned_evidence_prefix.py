"""Prefix decoder: the joint DP, length finishing, anchors from labels, and whole-grain decoding (free and anchored)
on a small synthetic movie with a known growth curve."""

import numpy as np
import pytest

from prototypes.learned_evidence import prefix as PF
from sparsetrack.render import Renderer

T, H, W = 50, 160, 160
GX, GY, GR = 80.0, 80.0, 10.0
EXIT_X = GX + GR


def true_length(t):
    return float(np.clip((t - 10) * 1.5, 0.0, 36.0))


def movie():
    """A dark grain with a thin dark tube growing to its right (1.5 px per bin from bin 10, stopping at 36 px), and the
    tube's probability map (scaled by 16, as the caches are)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    rg = np.hypot(xx - GX, yy - GY)
    img = np.full((T, H, W), 130.0, np.float32)
    prob = np.zeros((T, H, W), np.float32)
    for t in range(T):
        L = true_length(t)
        img[t][rg < GR] = 90.0
        img[t][(rg >= GR - 2) & (rg < GR)] = 60.0
        if L > 0:
            along = (xx >= EXIT_X) & (xx <= EXIT_X + L)
            d = np.abs(yy - GY)
            img[t] -= np.where(along, 35.0 * np.exp(-d ** 2 / 2.0), 0.0).astype(np.float32)
            prob[t] = np.where(along & (d <= 1.5), 16.0, 0.0)
        img[t] += rng.normal(0, 2.0, (H, W)).astype(np.float32)
    meta = {"shifts": np.zeros((T, 2)).tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    return Renderer(prob, meta), Renderer(img, meta), meta


@pytest.fixture(scope="module")
def renderers():
    return movie()


GRAIN = {"id": "g001", "x": GX, "y": GY, "r": GR}
PARAMS = PF.Params(half=60, big=60)


def test_dp_joint_recovers_a_growth_front():
    n, vmax = 60, 4
    front = np.clip((np.arange(30) - 8) * 2, 0, 50)
    E = np.where(np.arange(n)[None, :] < front[:, None], 1.0, -1.0).astype(np.float32)[:, None, :]
    lev, st, _ = PF.dp_joint(E, vmax, 0, 0.0, None)
    assert np.array_equal(lev, front)
    assert np.all(np.diff(lev) >= 0) and np.all(np.diff(lev) <= vmax)


def test_dp_joint_pins_the_final_level_and_keeps_steps_capped():
    rng = np.random.default_rng(1)
    E = rng.normal(0, 1, (25, 3, 40)).astype(np.float32)
    lev, st, _ = PF.dp_joint(E, 2, 1, 0.1, pin=1, pin_level=33)
    assert lev[-1] == 33 and st[-1] == 1
    assert np.all(np.diff(lev) >= 0) and np.all(np.diff(lev) <= 2)
    assert np.all(np.abs(np.diff(st)) <= 1)


def test_finalise_young_correction_and_stopped_cap():
    p = PF.Params(young=1.5, young_px=3.0, cap_frac=1.0, stop_bins=3, cap_max_px=3.0)
    front = np.array([0.0, 1.0, 3.0, 10.0, 20.0, 20.0, 20.0, 20.0])
    L = PF.finalise(front, None, p)
    assert L[0] == 0.0 and L[1] == pytest.approx(2.0) and L[2] == pytest.approx(3.0) and L[3] == 10.0
    capped = PF.finalise(front, 2.0, p)  # stopped for 3 bins: the round end's half-width comes off
    assert capped[-1] == pytest.approx(18.0) and capped[3] == 10.0


def test_anchors_from_labels_takes_the_latest_full_trace_with_a_path():
    labels = {"labels": {
        "g001": {"traces": {"40": {"state": "full", "length_px": 12.0, "path_xy_ref": [[0, 0], [12, 0]]},
                            "90": {"state": "full", "length_px": 30.0, "path_xy_ref": [[0, 0], [30, 0]]},
                            "120": {"state": "partial", "length_px": 35.0, "path_xy_ref": [[0, 0], [35, 0]]},
                            "150": {"state": "full", "length_px": 40.0}}},
        "g002": {"traces": {"50": {"state": "no_tube", "length_px": 0.0}}},
        "g003": {"traces": {"60": {"state": "full", "length_px": 9.0, "path_xy_ref": [[1, 1], [10, 1]]}}}}}
    anchors = PF.anchors_from_labels(labels, drop={"g003"})
    assert set(anchors) == {"g001"}
    assert anchors["g001"]["bin"] == 90 and anchors["g001"]["length_px"] == 30.0


def test_free_decoding_follows_the_growth_curve(renderers):
    RP, R_img, meta = renderers
    res = PF.decode_grain(RP, R_img, meta, GRAIN, [], PARAMS)
    assert res["status"] == "emerged_within"
    L = res["length"]["px"]
    for t in (15, 25, 35, 45):
        assert abs(L[t] - true_length(t)) <= 2.0, (t, L[t])
    assert abs(res["onset_frame"] - 12) <= 2


def test_anchored_decoding_pins_the_trace_and_holds_after_it(renderers):
    RP, R_img, meta = renderers
    trace = [[EXIT_X + 30.0, GY + 0.4], [EXIT_X + 12.0, GY - 0.3], [EXIT_X, GY]]  # drawn apex first, a little off
    res = PF.decode_grain(RP, R_img, meta, GRAIN, [], PARAMS, anchor={"bin": 30, "path_xy_ref": trace})
    L = res["length"]["px"]
    assert res["anchor"]["bin"] == 30 and abs(res["anchor"]["path_px"] - 30.0) <= 1.5
    assert abs(L[30] - 30.0) <= 1.5
    for t in (15, 20, 25):
        assert abs(L[t] - true_length(t)) <= 2.0, (t, L[t])
    assert L[31:] == [L[30]] * (T - 31)  # not decoded past the trace's bin
    assert any(f.startswith("held_after:") for f in res["flags"])
    assert abs(res["onset_frame"] - 12) <= 2


def test_trace_in_grain_frame_orders_from_the_grain(renderers):
    RP, R_img, meta = renderers
    G = PF.GrainStack(RP, R_img, meta, GRAIN, [], PARAMS, PARAMS.half, anchor_bin=30)
    xy = PF.trace_in_grain_frame(G, [[EXIT_X + 30.0, GY], [EXIT_X, GY]], G.T - 1)
    assert np.hypot(*(xy[0] - G.centre)) < np.hypot(*(xy[-1] - G.centre))
    assert xy[0] == pytest.approx([G.centre + GR - G.ls[-1][0], G.centre - G.ls[-1][1]])
