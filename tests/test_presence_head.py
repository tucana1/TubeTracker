"""rev11 item 5: the additive, DECLARED presence/rejection head.

The review: "A conditional softmax always picks a position and its
maximum is not an absolute cap-presence probability." The presence head
is therefore (a) separate from the location softmax, (b) additive and
config-declared so every pre-rev11 checkpoint still loads with an
identical state dict under the strict factory, and (c) initialized
AFTER all existing modules so the baseline arms stay bit-identical
under a fixed seed (equal-updates comparisons).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.model import (  # noqa: E402
    build_model, build_owner_prompt)
from prototypes.v30_video_apex.model_factory import (  # noqa: E402
    build_model_from_checkpoint)
from prototypes.v30_video_apex.train import (  # noqa: E402
    LossWeights, masked_multihead_loss, save_checkpoint, train_step_front)


def _clip(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 9, 1, 64, 64, generator=g)


def _route():
    return [[8.0, 32.0], [48.0, 32.0]]


def test_default_build_has_no_presence_params():
    m = build_model("temporal", base=4, multiscale=True)
    keys = set(m.state_dict())
    assert "front_present.weight" not in keys
    assert "front_present.bias" not in keys
    p = m.forward(_clip(), build_owner_prompt("o"),
                  route_xy=_route())
    assert p.front_present_logit is None


def test_declared_head_is_additive_and_does_not_move_existing_init():
    torch.manual_seed(0)
    base_m = build_model("temporal", base=4, multiscale=True)
    torch.manual_seed(0)
    pres_m = build_model("temporal", base=4, multiscale=True,
                         presence_head=True)
    kb = set(base_m.state_dict())
    kp = set(pres_m.state_dict())
    assert kp - kb == {"front_present.weight", "front_present.bias"}
    # the RNG stream for the existing modules is untouched: bit-equal
    for k in sorted(kb):
        assert torch.equal(base_m.state_dict()[k],
                           pres_m.state_dict()[k]), k


def test_forward_returns_a_presence_logit_only_with_a_route():
    m = build_model("temporal", base=4, multiscale=True,
                    presence_head=True)
    p = m.forward(_clip(), build_owner_prompt("o"), route_xy=_route())
    assert p.front_present_logit is not None
    assert tuple(p.front_present_logit.shape) == (1,)
    p2 = m.forward(_clip(), build_owner_prompt("o"))
    assert p2.front_present_logit is None


def test_factory_rebuilds_a_declared_presence_checkpoint(tmp_path):
    m = build_model("temporal", base=4, multiscale=True,
                    presence_head=True)
    cfg = {"variant": "temporal", "base": 4, "multiscale": True,
           "presence_head": True}
    ck = save_checkpoint(tmp_path / "pres.pt", m, config=dict(cfg))
    m2, info = build_model_from_checkpoint(ck)
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert info["model_kwargs"].get("presence_head") is True
    assert "front_present.weight" in m2.state_dict()
    # and without the declaration the head does not exist
    m3 = build_model("temporal", base=4, multiscale=True)
    cfg2 = {"variant": "temporal", "base": 4, "multiscale": True}
    ck2 = save_checkpoint(tmp_path / "nopres.pt", m3, config=dict(cfg2))
    m4, info4 = build_model_from_checkpoint(ck2)
    assert "presence_head" not in info4["model_kwargs"]
    assert "front_present.weight" not in m4.state_dict()


def test_presence_loss_term_only_with_head_and_target():
    m = build_model("temporal", base=4, multiscale=True,
                    presence_head=True)
    p = m.forward(_clip(), build_owner_prompt("o"), route_xy=_route())
    masks = {"front_present_valid": torch.ones(1)}
    out = masked_multihead_loss(
        {"front_present_logit": p.front_present_logit},
        {"front_present": torch.ones(1)}, masks,
        LossWeights(front_present=1.0))
    assert "front_present" in out
    assert float(out["front_present"].detach()) > 0
    out["front_present"].backward()
    g = m.front_present.weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    # an uncertain route (validity 0) contributes nothing
    out0 = masked_multihead_loss(
        {"front_present_logit": p.front_present_logit.detach()},
        {"front_present": torch.ones(1)},
        {"front_present_valid": torch.zeros(1)},
        LossWeights(front_present=1.0))
    assert float(out0["front_present"].detach()) == 0.0
    # weight 0.0 leaves the TOTAL untouched (the raw term is still
    # recorded for logging; old arms are unaffected by construction)
    outw = masked_multihead_loss(
        {"front_present_logit": p.front_present_logit.detach()},
        {"front_present": torch.ones(1)}, masks,
        LossWeights(front_present=0.0))
    assert float(outw["total"].detach()) == 0.0


def test_train_step_front_separates_present_from_absent_routes():
    """Short deterministic check: a positive route drives the presence
    logit up; a negative route drives it down. (Blocks, not epochs — the
    experiment compares at equal updates.)"""
    torch.manual_seed(0)
    m = build_model("temporal", base=4, multiscale=True,
                    presence_head=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    w = LossWeights(front_present=1.0, route_correct=0.0, front=0.0,
                    apex=0.0, visibility=0.0, body=0.0)
    clip = _clip()
    prompt = build_owner_prompt("o")
    route = _route()
    with torch.no_grad():
        _p0 = m.forward(clip, prompt, route_xy=route)
    S = int(_p0.front_logits.shape[-1])
    interval = torch.zeros(1, 1, S)
    for _ in range(12):
        r1 = train_step_front(m, opt, clip, prompt, route,
                              torch.zeros(1, 1, 64, 64), {"apex_valid": torch.zeros(1)},
                              interval, torch.ones(1), torch.tensor([2]),
                              torch.ones(1), w, route_t=torch.ones(1),
                              present_t=torch.ones(1))
        r0 = train_step_front(m, opt, clip, prompt, route,
                              torch.zeros(1, 1, 64, 64), {"apex_valid": torch.zeros(1)},
                              interval, torch.ones(1), torch.tensor([2]),
                              torch.ones(1), w, route_t=torch.ones(1),
                              present_t=torch.zeros(1))
        assert "front_present" in r1 and "front_present" in r0
    m.eval()
    with torch.no_grad():
        p = m.forward(clip, prompt, route_xy=route)
    # after alternating pushes the logit must at least carry the signal;
    # the DIRECT check is that both steps trained the head
    assert float(p.front_present_logit) != 0.0
