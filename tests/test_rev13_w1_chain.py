"""rev13 W1 acceptance: the real path-training call chain honors the
finalized per-head validity masks, and what the loss consumed is
recorded AT the loss boundary.

The rev13 audit's counterexample: `train_step_front` unconditionally
overwrote the caller's `vis_valid`/`route_valid`/`front_valid` masks
with its positional arguments, so a batch finalized with an explicit
zero (no/invalid query) still trained the owner-specific heads. These
tests exercise the real chain (real model, real step) with valid,
none/unresolved and invalid queries across the positive and wrong-route
branches — no helper-only substitute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")


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


def _vis_step(m, opt, clip, prompt, route, S, masks_extra, pos_vis_valid):
    """One step with ONLY the visibility term weighted; the positional
    vis_valid argument deliberately disagrees with the masks."""
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)
    w = LossWeights(visibility=1.0, route_correct=0.0, front=0.0,
                    apex=0.0, body=0.0, front_present=0.0)
    masks = {"apex_valid": torch.zeros(1)}
    masks.update(masks_extra)
    return train_step_front(
        m, opt, clip, prompt, route, torch.zeros(1, 1, 64, 64),
        masks, torch.zeros(1, 1, S), torch.zeros(1),
        torch.tensor([2]), pos_vis_valid, w)


def _vis_head_grad_abs(m) -> float:
    g = m.vis_head.weight.grad
    return 0.0 if g is None else float(g.abs().sum())


def test_audit_counterexample_masks_are_authoritative():
    """masks say zero, positional arg says ones -> loss consumes zero."""
    m, opt, clip, prompt, route, S = _setup()
    r = _vis_step(m, opt, clip, prompt, route, S,
                  {"vis_valid": torch.zeros(1)}, torch.ones(1))
    cons = r["consumed"]["vis_valid"]
    assert cons["consumed"] is False and cons["mass"] == 0.0
    assert float(r.get("visibility", 0.0)) == 0.0
    assert _vis_head_grad_abs(m) == 0.0


def test_valid_query_still_learns():
    m, opt, clip, prompt, route, S = _setup()
    r = _vis_step(m, opt, clip, prompt, route, S,
                  {"vis_valid": torch.ones(1)}, torch.zeros(1))
    cons = r["consumed"]["vis_valid"]
    assert cons["consumed"] is True and cons["mass"] == 1.0
    assert float(r.get("visibility", 0.0)) > 0.0
    assert _vis_head_grad_abs(m) > 0.0


def test_zeroed_route_and_front_reach_the_loss():
    """The wrong-route branch's all-zero mask set must zero every
    owner-specific term even though the positional args say ones."""
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)
    m, opt, clip, prompt, route, S = _setup()
    w = LossWeights(route_correct=1.0, front=1.0, visibility=1.0,
                    apex=0.0, body=0.0, front_present=0.0)
    zero_masks = {"apex_valid": torch.zeros(1),
                  "vis_valid": torch.zeros(1),
                  "route_valid": torch.zeros(1),
                  "front_valid": torch.zeros(1)}
    r = train_step_front(
        m, opt, clip, prompt, route, torch.zeros(1, 1, 64, 64),
        zero_masks, torch.zeros(1, 1, S), torch.ones(1),
        torch.tensor([2]), torch.ones(1), w,
        route_t=torch.ones(1), route_valid=torch.ones(1))
    for mk in ("vis_valid", "route_valid", "front_valid"):
        assert r["consumed"][mk]["consumed"] is False, mk
    assert float(r.get("route_correct", 0.0)) == 0.0
    assert float(r.get("front", 0.0)) == 0.0
    assert _vis_head_grad_abs(m) == 0.0


def test_consumed_reports_weight_gate_too():
    """A nonzero mask with a zero effective weight is not 'consumed'."""
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)
    m, opt, clip, prompt, route, S = _setup()
    w = LossWeights(visibility=0.0, route_correct=0.0, front=0.0,
                    apex=0.0, body=0.0, front_present=0.0)
    r = train_step_front(
        m, opt, clip, prompt, route, torch.zeros(1, 1, 64, 64),
        {"apex_valid": torch.zeros(1), "vis_valid": torch.ones(1)},
        torch.zeros(1, 1, S), torch.zeros(1),
        torch.tensor([2]), torch.ones(1), w)
    cons = r["consumed"]["vis_valid"]
    assert cons["mass"] == 1.0 and cons["weight"] == 0.0
    assert cons["consumed"] is False


def test_query_matrix_zero_when_invalid_learns_when_valid():
    """The four query states as the trainer resolves them: the
    owner-specific heads are exactly zero for none/unresolved/invalid
    and nonzero for valid — in the same call chain."""
    from scripts.train_v30_front import owner_query_valid

    class S_:
        def __init__(self, owner_uuid="", query_kind="focus"):
            self.owner_uuid = owner_uuid
            self.query_kind = query_kind

    states = {
        "valid": (S_(owner_uuid="obs-1", query_kind="tube"), "tube", True),
        "none": (S_(owner_uuid="", query_kind="none"), "none", False),
        "unresolved": (S_(owner_uuid="", query_kind="focus"), "focus",
                       False),
        "invalid": (S_(owner_uuid="unassigned", query_kind="focus"),
                    "unassigned", False),
    }
    for name, (s, resolved, expect) in states.items():
        assert owner_query_valid(s, resolved) is expect, name
        m, opt, clip, prompt, route, S = _setup()
        vv = torch.ones(1) if expect else torch.zeros(1)
        r = _vis_step(m, opt, clip, prompt, route, S,
                      {"vis_valid": vv}, torch.ones(1))
        assert r["consumed"]["vis_valid"]["consumed"] is expect, name
        assert (_vis_head_grad_abs(m) > 0.0) is expect, name
