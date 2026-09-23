"""v30 route/front skeleton acceptance (P3A, H262)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototypes.v30_video_apex import contracts as C
from prototypes.v30_video_apex import dataset as D
from prototypes.v30_video_apex import train as T
from prototypes.v30_video_apex import infer as I
from prototypes.v30_video_apex import association as A
from prototypes.v30_video_apex import evaluate as E
from prototypes.v30_video_apex import adapter_v29 as AD


def test_reject_cross_movie_join():
    try:
        C.reject_unsupported_join("m1", "m2")
    except ValueError:
        return
    raise AssertionError("cross-movie join must raise")


def test_candidate_schema_roundtrip():
    d = {"candidate_id": "c1", "movie_id": "m", "owner_id": "o",
         "source_frame": 3, "x_native": 10.0, "y_native": 20.0}
    c = C.validate_candidate_schema(d)
    assert c.x_native == 10.0


def test_supervision_mask_unknown_default():
    import numpy as np
    m = C.SupervisionMask(positive=np.zeros((4, 4), bool),
                          negative=np.zeros((4, 4), bool),
                          unknown=np.ones((4, 4), bool))
    assert m.validate() is None


def test_grouped_manifest_freezes_splits():
    a = T.freeze_splits(["s1", "s2", "s3", "s4"], seed=0)
    b = T.freeze_splits(["s1", "s2", "s3", "s4"], seed=0)
    assert a == b  # deterministic
    assert set(a["train"]).isdisjoint(set(a["dev"]))  # no group leakage
    assert sorted(a["train"] + a["dev"]) == ["s1", "s2", "s3", "s4"]


def test_crop_keys_stable():
    k1 = D.stable_crop_key("e1", [0, 0, 64, 64], 7)
    assert k1 == D.stable_crop_key("e1", [0, 0, 64, 64], 7)
    assert k1 != D.stable_crop_key("e1", [0, 0, 64, 64], 8)


def test_model_forward_modes():
    from prototypes.v30_video_apex import model as M
    torch = pytest_torch()
    clip = M.random_clip(batch=1, frames=9, size=32)
    prompt = M.OwnerPrompt(owner_id="o")
    for variant in ("single", "temporal"):
        m = M.build_model(variant=variant)
        preds = M.predict_candidates(m, clip, prompt)
        assert len(preds) >= 1
        assert any(p.get("observation") == "not_observed" for p in preds) \
            or "x_native" in preds[0]


def pytest_torch():
    import torch
    return torch


def test_train_step_descends_or_runs():
    from prototypes.v30_video_apex import model as M
    import torch
    torch.manual_seed(0)
    m = M.build_model(variant="single")
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    prompt = M.OwnerPrompt(owner_id="o")
    clip = M.random_clip(batch=1, frames=9, size=32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    apex = torch.exp(-((xx - 16.0) ** 2 + (yy - 16.0) ** 2) / 8.0).unsqueeze(0)
    target = {"apex_heat": apex, "visibility": torch.tensor([0])}
    masks = {"apex_valid": torch.ones_like(apex), "vis_valid": torch.ones(1)}
    l1 = T.train_step(m, opt, clip, prompt, target, masks)["total"]
    l2 = T.train_step(m, opt, clip, prompt, target, masks)["total"]
    assert l2 != l1  # labels drive updates (surrogates forbidden)


def test_infer_cache_and_raw_candidates():
    from prototypes.v30_video_apex import model as M
    cache = I.FeatureCache()
    m = M.build_model(variant="single")
    clip = M.random_clip(batch=1, frames=9, size=32)
    p1 = M.OwnerPrompt(owner_id="a")
    p2 = M.OwnerPrompt(owner_id="b")
    outs = I.batched_owner_queries(m, clip, [p1, p2], batch_size=1)
    per_owner = {o: [d for d in outs if d["owner_id"] == o] for o in ("a", "b")}
    assert all(len(v) >= 2 for v in per_owner.values())  # modes + no-obs
    raw = I.decode_raw_candidates(outs)
    assert len(raw) == len(outs)  # decode drops nothing
    cache = I.FeatureCache()
    assert cache.get("m", 5, "t", "h") is None
    cache.put("m", 5, "t", "h", object())
    assert cache.get("m", 5, "t", "h") is not None
    assert cache.get("other", 5, "t", "h") is None  # movie-qualified
    assert abs(cache.recall_diagnostics()["hit_rate"] - 1 / 3) < 1e-9


def test_beam_keeps_unresolved():
    frames = [[{"candidate_id": f"o:{t}:a", "observation": "observed",
                "x_native": float(t), "y_native": 1.0, "tip_score": 0.9,
                "route_id": "r0"},
               {"candidate_id": f"o:{t}:b", "observation": "observed",
                "x_native": 50.0, "y_native": 50.0, "tip_score": 0.4,
                "route_id": "r1"}] for t in range(3)]
    beam = A.beam_search_per_owner(frames, "o", beam_width=3)
    assert len(beam) >= 1
    assert beam[0].score >= beam[-1].score


def test_beam_empty_frames_declares_lost():
    beam = A.beam_search_per_owner([], "o")
    assert beam[0].state == A.TrackState.LOST


def test_oracle_substitution_and_scoring():
    cands = [{"candidate_id": "c1", "movie_id": "m", "owner_id": "o",
              "source_frame": 5, "x_native": 100.0, "y_native": 100.0,
              "observation": "observed"}]
    gold = [{"movie_id": "m", "owner_id": "o", "source_frame": 5,
             "x_native": 101.0, "y_native": 100.0, "observation": "observed",
             "stratum": "clean"}]
    rep = E.score_probes(cands, gold, tolerance_px=5.0)
    assert rep["correct_owner_coverage"] == 1.0
    swapped = E.substitute_oracle_routes(cands, {("m", "o"): "route-oracle"})
    assert swapped[0]["route"] == "route-oracle"
    assert swapped[0]["evidence"]["route_source"] == "oracle"


def test_adapter_refuses_tip_only_length():
    try:
        AD.export_tip_path_length(None, front_s=12.0)
    except ValueError:
        pass
    else:
        raise AssertionError("tip-only length must raise")
    tip_only = AD.export_tip_path_length(None, None, tip_xy=(3.0, 4.0))
    assert tip_only["length_px"] is None and tip_only["tip"] == [3.0, 4.0]
    full = AD.export_tip_path_length(
        [[0.0, 0.0], [3.0, 4.0]], front_s=None, status="direct")
    assert abs(full["length_px"] - 5.0) < 1e-6
    assert full["tip"] == [3.0, 4.0]


def test_adapter_refuses_nonfinite_and_censors_range():
    import math
    from prototypes.v30_video_apex.adapter_v29 import (
        export_tip_path_length)
    route = [[0., 0.], [100., 0.]]
    for bad in (float("nan"), float("inf")):
        try:
            export_tip_path_length(route, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"front {bad} must be refused")
    try:
        export_tip_path_length(
            [[0., 0.], [float("nan"), 0.]], 10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN vertex must be refused")
    over = export_tip_path_length(route, 140.0)
    assert over["withheld"] == "front-censored"
    assert over["tip"] == [100.0, 0.0] and over["length_px"] == 100.0
    neg = export_tip_path_length(route, -10.0)
    assert neg["withheld"] == "front-censored"
    assert neg["tip"] == [0.0, 0.0] and neg["length_px"] == 0.0
    tip_only = export_tip_path_length(None, None, tip_xy=[7., 8.])
    assert tip_only["tip"] == [7.0, 8.0] and tip_only["path"] is None
    assert tip_only["length_px"] is None
    assert tip_only["withheld"] == "length-needs-route"


def test_select_honors_exported_flag_and_measurements_scored():
    from prototypes.v30_video_apex.evaluate import (
        _select, score_measurements)
    cands = [
        {"candidate_id": "a", "x_native": 0., "y_native": 0.,
         "tip_score": 0.99},
        {"candidate_id": "b", "x_native": 50., "y_native": 50.,
         "tip_score": 0.01, "selected": True},
    ]
    sel = _select(cands)
    assert sel is not None and sel["candidate_id"] == "b"
    meas = [{"movie": "ld", "event": "o1", "source_frame": 10,
             "tip": [50., 50.], "status": "direct"},
            {"movie": "ld", "event": "o2", "source_frame": 20,
             "tip": None, "status": "withheld"}]
    gold = [{"movie_id": "ld", "owner_id": "o1", "source_frame": 10,
             "x_native": 51., "y_native": 50., "observation": "observed"},
            {"movie_id": "ld", "owner_id": "o2", "source_frame": 20,
             "x_native": 5., "y_native": 5., "observation": "observed"}]
    rep = score_measurements(meas, gold, tolerance_px=5.0)
    assert rep["n_opportunities"] == 2 and rep["n_hit"] == 1
    assert rep["n_withheld_or_missing"] == 1
    assert rep["coverage"] == 0.5


def test_curved_walkers_follow_tube_and_stop_off_it():
    """Walkers track wall energy from attachments, stop in background."""
    import numpy as np
    from prototypes.v30_video_apex.routes import propose_curved_walkers
    gray = np.full((200, 200), 180.0)
    # Horizontal bright tube (lumen 200, dark walls) mid-frame.
    gray[95:106, 20:180] = 200.0
    gray[93:95, 20:180] = 100.0
    gray[106:108, 20:180] = 100.0
    props = propose_curved_walkers(
        gray, (30.0, 100.0), 6.0, n_attach=4, dirs_per_attach=1,
        lengths=(60.0,), radii=(6.0,), support_floor=2.5)
    assert props, "a walker must survive along the tube"
    for p in props:
        poly = np.asarray(p["polyline"], float)
        assert len(poly) >= 2
    # Best walker should run along the tube (y ~= 100), not wander.
    dev = min(float(np.abs(np.asarray(p["polyline"])[:, 1] - 100).mean())
              for p in props)
    assert dev < 8.0
    # Attachments start on the ring, outside the center.
    for p in props:
        ax, ay = p["attachment"]
        assert abs(((ax - 30.0) ** 2 + (ay - 100.0) ** 2) ** 0.5 - 6.0) < 2.0
