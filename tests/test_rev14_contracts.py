"""Acceptance probes at the real snapshot/model/loss boundaries."""
import json

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.model import build_model, build_owner_prompt
from prototypes.v30_video_apex.targets import samples_from_snapshot


def sample(tmp_path, complete=True):
    row = {"obs_uuid": "review-1", "obs_revision": 3, "task_uuid": "task-1",
           "movie": "ld", "movie_path": "/example.mp4", "source_frame": 10,
           "owner_uuid": "grain-1", "direct_state": "direct_visible",
           "direct_xy": [52, 32], "path_xy": [[12, 32], [52, 32]],
           "path_complete": complete, "tip_source": "explicit_point"}
    (tmp_path / "observations.json").write_text(json.dumps([row]))
    return samples_from_snapshot(str(tmp_path))[0]


@pytest.mark.parametrize("mode", ["rotated", "hard"])
@pytest.mark.parametrize("valid_query,complete", [(True, True), (False, True), (True, False)])
def test_loaded_negative_reaches_route_loss(tmp_path, mode, valid_query, complete):
    from prototypes.v30_video_apex.batch_builder import negative_route_masks
    from prototypes.v30_video_apex.route_supervision import wrong_route_certificate
    from prototypes.v30_video_apex.train import LossWeights, train_step_front
    from scripts.train_v30_front import owner_query_valid
    s = sample(tmp_path, complete)
    assert s.path_complete is complete
    assert s.complete is False  # independently scoped body-mask flag
    route = ([[12, 32], [12, 56]] if mode == "rotated" else
             [[12, 32], [12, 52], [52, 52], [52, 32]])
    cert = wrong_route_certificate(s, route, current_s=24 if mode == "rotated" else None)
    assert cert["certified_wrong_route"] is complete
    assert cert["obs_revision"] == 3
    query = owner_query_valid(s, "tube" if valid_query else "none")
    masks = negative_route_masks({"apex_valid": torch.ones(1),
                                  "vis_valid": torch.ones(1),
                                  "front_valid": torch.ones(1)},
                                 query_valid=query, certified=cert["certified_wrong_route"])
    torch.manual_seed(3)
    model = build_model("temporal", base=4, multiscale=True)
    clip = torch.rand(1, 9, 1, 64, 64)
    prompt = build_owner_prompt("grain-1")
    with torch.no_grad():
        n = model.forward(clip, prompt, route_xy=route).front_logits.shape[-1]
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    result = train_step_front(model, opt, clip, prompt, route,
        torch.zeros(1, 1, 64, 64), masks, torch.zeros(n), torch.ones(1),
        torch.tensor([0]), torch.ones(1),
        LossWeights(route_correct=1, apex=1, front=1, visibility=1, body=0),
        route_t=torch.zeros(1), route_valid=torch.ones(1))
    expect = valid_query and complete
    assert result["consumed"]["route_valid"]["consumed"] is expect
    assert (result["route_correct"] > 0) is expect
    grad = model.route_head.bias.grad
    assert (float(grad.sum()) > 0) is expect  # BCE gradient lowers wrong-route logit
    assert result["visibility"] == result["front"] == result["apex"] == 0


def test_support_tail_does_not_license_wrong_route(tmp_path):
    from prototypes.v30_video_apex.route_supervision import wrong_route_certificate
    s = sample(tmp_path)
    r = [[12, 32], [52, 32], [52, 62], [12, 62]]
    assert not wrong_route_certificate(s, r)["certified_wrong_route"]


@pytest.mark.parametrize("encoder", [False, True])
def test_actual_query_and_context_dependencies(encoder):
    torch.manual_seed(0)
    model = build_model("temporal", base=4, multiscale=True,
                        cap_window_head=True, cap_window_encoder=encoder)
    x = torch.rand(1, 9, 1, 64, 64, requires_grad=True)
    xy, tg = torch.tensor([[[32., 32.]]]), torch.tensor([[[1., 0.]]])
    for one in (False, True):
        out = model.score_cap_window(x, build_owner_prompt("o"), xy, tg, frame_only=one)
        g = torch.autograd.grad(out["logits"].sum() + out["dxy_px"].sum(), x)[0]
        consumed = set(torch.nonzero(g.abs().sum((0, 2, 3, 4)) > 0).flatten().tolist())
        assert consumed == ({4} if one else set(range(9)))
        assert set(out["consumed_indices"]) == consumed
    with pytest.raises(ValueError, match="contain the query"):
        model.score_cap_window(x, build_owner_prompt("o"), xy, tg, window_indices=[0, 1])


def test_offset_basis_and_checkpoint_function_roundtrip(tmp_path):
    from prototypes.v30_video_apex.train import save_checkpoint
    from prototypes.v30_video_apex.model_factory import build_model_from_checkpoint
    torch.manual_seed(4)
    model = build_model("temporal", base=4, multiscale=True, cap_window_head=True).eval()
    with torch.no_grad():
        model.cw_dxy.weight.zero_()
        model.cw_dxy.bias.copy_(torch.tensor([0.2, -0.3]))
    x = torch.rand(1, 9, 1, 64, 64)
    xy = torch.tensor([[[32., 32.]]]); p = build_owner_prompt("o")
    a = model.score_cap_window(x, p, xy, torch.tensor([[[1., 0.]]]))
    b = model.score_cap_window(x, p, xy, torch.tensor([[[0., 1.]]]))
    assert torch.allclose(b["dxy_px"], torch.stack([-a["dxy_px"][..., 1], a["dxy_px"][..., 0]], -1))
    ck = tmp_path / "new.pt"
    save_checkpoint(ck, model, config={"variant": "temporal", "base": 4, "multiscale": True})
    loaded, info = build_model_from_checkpoint(ck)
    loaded.eval()
    c = loaded.score_cap_window(x, p, xy, torch.tensor([[[1., 0.]]]))
    assert info["config"]["cap_window_semantics"] == 2
    for key in ("logits", "dxy_px", "logvar"):
        assert torch.equal(c[key], a[key]), key


def test_manifest_inputs_samples_are_never_results(tmp_path):
    from scripts.compare_run_manifests import compare
    a = {"manifest_schema": "tubetracker.run.v1", "inputs": {"samples": ["a"], "seed": 1}, "results": {"samples": [1]}}
    b = json.loads(json.dumps(a))
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    b["results"]["samples"] = [999]
    pb.write_text(json.dumps(b))
    assert compare(pa, pb, "inputs.seed")["verdict"] == "identical"
    b["inputs"]["samples"] = ["other"]
    pb.write_text(json.dumps(b))
    assert compare(pa, pb, "inputs.seed")["verdict"] == "NOT-single-variable"
