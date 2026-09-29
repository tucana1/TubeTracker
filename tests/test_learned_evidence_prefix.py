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


def test_anchors_from_labels_takes_the_latest_full_trace_the_tool_still_asks_for():
    seen = {"verdict": "emerged_within", "first_visible_bin": 30, "last_absent_bin": 29}  # traces at 36, 70, 122, 174

    def full(L, path=True):
        return {"state": "full", "length_px": L, **({"path_xy_ref": [[0, 0], [L, 0]]} if path else {})}

    labels = {"n_bins": 176, "grains": {"g001": {}, "g002": {}, "g003": {}, "g004": {}, "g005": {"excluded": True}},
              "labels": {
                  "g001": {"onset": seen, "traces": {"36": full(12.0), "70": full(30.0),
                                                     "122": {**full(35.0), "state": "partial"},
                                                     "174": full(40.0, path=False)}},
                  "g002": {"onset": seen, "traces": {"50": full(20.0)}},  # not a bin the tool asks for
                  "g003": {"onset": {"verdict": "no_emergence_by_end"}, "traces": {"70": full(9.0)}},  # stale
                  "g004": {"onset": seen, "traces": {"70": full(9.0)}},
                  "g005": {"onset": seen, "traces": {"70": full(9.0)}}}}
    anchors = PF.anchors_from_labels(labels, drop={"g004"})
    assert set(anchors) == {"g001"}
    assert anchors["g001"]["bin"] == 70 and anchors["g001"]["length_px"] == 30.0


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
    # clicked exit first, 2 px inside the rim, then along the tube a little off its centreline
    trace = [[EXIT_X - 2.0, GY], [EXIT_X + 12.0, GY - 0.6], [EXIT_X + 30.0, GY + 0.5]]
    traced = float(np.hypot(*np.diff(np.array(trace), axis=0).T).sum())
    res = PF.decode_grain(RP, R_img, meta, GRAIN, [], PARAMS,
                          anchor={"bin": 30, "path_xy_ref": trace, "length_px": traced})
    L = res["length"]["px"]
    assert res["anchor"]["bin"] == 30 and abs(res["anchor"]["inside_rim_px"] - 1.0) <= 0.6
    assert abs(L[30] - traced) <= 0.05  # the trace's bin reads the traced length
    for t in (15, 20, 25):  # the trace's convention (from its first click) at the other bins
        assert abs(L[t] - (true_length(t) + 2.0)) <= 2.0, (t, L[t])
    assert L[31:] == [L[30]] * (T - 31)  # not decoded past the trace's bin
    assert any(f.startswith("held_after:") for f in res["flags"])
    assert abs(res["onset_frame"] - 12) <= 2


def test_a_trace_keeps_the_order_it_was_clicked_in(renderers):
    RP, R_img, meta = renderers
    G = PF.GrainStack(RP, R_img, meta, GRAIN, [], PARAMS, PARAMS.half, anchor_bin=30)
    # a tube that leaves the rim and curls back: its apex ends nearer the grain than its exit
    curl = [[EXIT_X + 3.0, GY], [EXIT_X + 20.0, GY + 10.0], [EXIT_X + 5.0, GY + 14.0], [GX + 2.0, GY + GR + 1.0]]
    xy = PF.trace_in_grain_frame(G, curl, G.T - 1)
    assert xy[0] == pytest.approx([G.centre + GR + 3.0 - G.ls[-1][0], G.centre - G.ls[-1][1]])
    pts, ss, inside, after = PF.anchored_path(G, xy, PARAMS)
    assert inside == 0.0 and np.hypot(*(pts[0] - xy[0])) < 1.5  # starts at the exit, not at the apex


def test_support_fusion_lets_the_image_add_evidence_but_not_veto(renderers, monkeypatch):
    RP, R_img, meta = renderers

    def hostile(G, best, X, Y, NX, NY, p):  # an image term that says "no tube" everywhere near the exit
        return -np.ones((G.T, X.shape[1], int(round(p.img_s_max / p.step)) + 1), np.float32), {}

    monkeypatch.setattr(PF, "image_evidence", hostile)
    off = PF.decode_grain(RP, R_img, meta, GRAIN, [], PF.Params(half=60, big=60, image_term=False))
    support = PF.decode_grain(RP, R_img, meta, GRAIN, [], PF.Params(half=60, big=60, img_fuse="support"))
    vetoed = PF.decode_grain(RP, R_img, meta, GRAIN, [], PF.Params(half=60, big=60, img_fuse="sum"))
    assert support["length"]["px"] == off["length"]["px"]
    assert vetoed["onset_frame"] is None or vetoed["onset_frame"] > off["onset_frame"]


def test_the_decoded_path_is_reported_as_the_labelling_tool_records_a_trace(renderers):
    RP, R_img, meta = renderers
    res = PF.decode_grain(RP, R_img, meta, GRAIN, [], PARAMS)
    xy = np.array(res["path_xy_ref"])
    assert res["path_bin"] == T - 2
    assert abs(np.hypot(*(xy[0] - [GX, GY])) - GR) <= 1.5  # starts at the exit
    assert abs(float(np.hypot(*np.diff(xy, axis=0).T).sum()) - res["front_px"][T - 2]) <= 1.5  # ends at the apex
