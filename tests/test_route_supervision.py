"""rev11 item 5: route-supervision label + masking contract.

The route head is trained on the deployment proposer's ACTUAL
proposals: present (<= 8 px of the cap) -> positive; decoy (> 24 px)
-> negative; the 8-24 px band stays UNKNOWN and is MASKED (never a
silent negative). These tests pin the mapping and the masking
semantics of train_step_front.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")


def test_route_label_mapping():
    from scripts.rev11_proposal_fit import route_label
    l1, l0 = route_label(1), route_label(0)
    assert l1 is not None and float(l1) == 1.0
    assert l0 is not None and float(l0) == 0.0
    assert route_label(None) is None
    # any non-0/1 value is treated as uncertain (masked), never a
    # silent negative
    assert route_label(-1) is None


def _setup():
    from prototypes.v30_video_apex.model import (
        build_model, build_owner_prompt)
    torch.manual_seed(0)
    m = build_model("temporal", base=4, multiscale=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    clip = torch.zeros(1, 9, 1, 64, 64)
    prompt = build_owner_prompt("o")
    route = [[8.0, 40.0], [56.0, 40.0]]
    with torch.no_grad():
        p = m.forward(clip, prompt, route_xy=route)
    S = int(p.front_logits.shape[-1])
    return m, opt, clip, prompt, route, S


def _step(m, opt, clip, prompt, route, S, route_t, route_valid):
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)
    w = LossWeights(route_correct=1.0, front=0.0, apex=0.0,
                    visibility=0.0, body=0.0, front_present=0.0)
    return train_step_front(
        m, opt, clip, prompt, route, torch.zeros(1, 1, 64, 64),
        {"apex_valid": torch.zeros(1)}, torch.zeros(1, 1, S),
        torch.zeros(1), torch.tensor([2]), torch.ones(1), w,
        route_t=route_t, route_valid=route_valid)


def test_route_loss_present_when_labeled():
    m, opt, clip, prompt, route, S = _setup()
    r = _step(m, opt, clip, prompt, route, S, torch.ones(1),
              torch.ones(1))
    assert "route_correct" in r
    assert float(r["route_correct"]) >= 0.0


def test_route_loss_absent_when_unlabeled():
    m, opt, clip, prompt, route, S = _setup()
    r = _step(m, opt, clip, prompt, route, S, None, None)
    assert float(r.get("route_correct", 0.0)) == 0.0


def test_route_loss_masked_when_invalid():
    """route_valid=0 (the 8-24 px band) must zero the route term —
    the same forward with a target must not contribute gradient."""
    m, opt, clip, prompt, route, S = _setup()
    r = _step(m, opt, clip, prompt, route, S, torch.zeros(1),
              torch.zeros(1))
    assert float(r.get("route_correct", 0.0)) == 0.0


def test_uncertain_proposal_gets_no_route_gradient():
    """End-to-end label semantics: present->1, absent->0, uncertain->
    no route loss at all (None target)."""
    from scripts.rev11_proposal_fit import route_label
    m, opt, clip, prompt, route, S = _setup()
    r_unc = _step(m, opt, clip, prompt, route, S,
                  route_label(None), None)
    assert float(r_unc.get("route_correct", 0.0)) == 0.0
    r_pos = _step(m, opt, clip, prompt, route, S,
                  route_label(1), torch.ones(1))
    assert float(r_pos["route_correct"]) >= 0.0
