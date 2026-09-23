"""rev12 P0.2 acceptance 3: no-query owner-specific gradients are ZERO.

Mechanical checks:
- the visibility term under a zero vis_valid mask has EXACTLY zero
  gradient wrt the visibility logits (the repaired gating);
- the same for the route head under a zero route_valid mask;
- the gating helper's declared semantics: 'none' resolved prompt and
  focus-scoped unassigned records are NOT valid owner queries (their
  owner-specific losses are zero); grain/tube-scoped records and real
  owners are valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")


def _zero_grad_visibility(vis_valid_value: float) -> float:
    from prototypes.v30_video_apex.train import masked_multihead_loss
    logits = torch.zeros(1, 7, requires_grad=True)
    pred = {"visibility_logits": logits}
    target = {"visibility": torch.tensor([3])}
    masks = {"vis_valid": torch.tensor([vis_valid_value])}
    out = masked_multihead_loss(
        pred, target, masks, None)
    out["total"].backward()
    return float(logits.grad.abs().sum())


def test_noquery_vis_gradient_exactly_zero():
    # repaired gating: vis_valid = 0 -> EXACTLY zero gradient
    assert _zero_grad_visibility(0.0) == 0.0
    # sanity: an active mask does produce gradient
    assert _zero_grad_visibility(1.0) > 0.0


def test_route_negative_gradient_zero_when_invalid():
    from prototypes.v30_video_apex.train import masked_multihead_loss
    rl = torch.zeros(1, requires_grad=True)
    pred = {"route_logit": rl}
    target = {"route": torch.zeros(1)}
    masks = {"route_valid": torch.zeros(1)}
    out = masked_multihead_loss(pred, target, masks, None)
    out["total"].backward()
    assert float(rl.grad.abs().sum()) == 0.0


def test_owner_query_valid_semantics():
    from scripts.train_v30_front import owner_query_valid

    class S:
        def __init__(self, owner_uuid="", query_kind="focus"):
            self.owner_uuid = owner_uuid
            self.query_kind = query_kind

    # resolved no-query prompt -> never valid, even with a real owner
    assert not owner_query_valid(S("obs-r4-p01", "grain"), "none")
    # 'unassigned' is not an identity
    assert not owner_query_valid(S("unassigned", "focus"), "owner")
    assert not owner_query_valid(S("", "focus"), "owner")
    # focus-scoped legacy records are not owner queries
    assert not owner_query_valid(S("unassigned", "focus"), "auto-grain")
    # grain-scoped (certified owned-absence) records ARE valid queries
    assert owner_query_valid(S("unassigned", "grain"), "human-grain")
    # real owners and tube-scoped records are valid
    assert owner_query_valid(S("obs-r4-p01", "tube"), "auto-grain")
    assert owner_query_valid(S("own-ld-0001", "focus"), "auto-grain")
    # box-scoped (neg regions) are not
    assert not owner_query_valid(S("", "box"), "auto-grain")
