"""rev9 WP-A.6: a preference ORDERS two lanes; it never certifies one.

The old path branch trained the preferred lane with `route_t = 1.0`
(absolute correctness) and treated "neither" as two zeros. These tests
pin the replacement: margin ordering for "A"/"B", rejection of both for
"neither", and no silent acceptance of an unknown verdict.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.route_pref import (
    pairwise_route_loss, pairwise_route_step)

LANE_A = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]      # rightward
LANE_B = [[0.0, 50.0], [10.0, 50.0], [20.0, 50.0]]    # downward-ish


class _StubModel:
    """Route logit = linear readout of the lane's mean position."""

    def __init__(self):
        self.w = torch.zeros(2, requires_grad=True)

    def forward(self, clip, prompt, query_index=4, route_xy=None):
        m = torch.as_tensor(np.asarray(route_xy, dtype=np.float32).mean(0))
        class _P:
            pass
        p = _P()
        p.route_logit = (m * self.w).sum().reshape(1)
        return p

    def parameters(self):
        return [self.w]


class _FixedModel:
    """Returns preset logits, keyed by a lane's first x coordinate."""

    def __init__(self, table):
        self.table = dict(table)

    def forward(self, clip, prompt, query_index=4, route_xy=None):
        key = float(np.asarray(route_xy, dtype=float)[0][0])
        class _P:
            pass
        p = _P()
        p.route_logit = torch.tensor([self.table[key]],
                                     requires_grad=True).reshape(1)
        return p


def test_preference_depends_only_on_the_order_not_the_level():
    """The loss is invariant to shifting BOTH lanes together, and a
    larger gap is cheaper: that is what 'relative preference' means.
    """
    a = [[0.0, 0.0], [10.0, 0.0]]
    b = [[100.0, 0.0], [110.0, 0.0]]
    far = [[200.0, 0.0], [210.0, 0.0]]
    m1 = _FixedModel({0.0: 0.0, 100.0: 0.0})            # equal
    m2 = _FixedModel({0.0: 40.0, 100.0: 40.0})          # both shifted +40
    l1, _ = pairwise_route_loss(m1, None, None, a, b, "A")
    l2, _ = pairwise_route_loss(m2, None, None, a, b, "A")
    assert float(l1.detach()) == pytest.approx(float(l2.detach())), (
        "a common-mode shift must not change the ordering loss")
    # a comfortable gap is cheaper than a tie
    m3 = _FixedModel({0.0: 4.0, 200.0: 0.0})            # A wins by 4
    l3, _ = pairwise_route_loss(m3, None, None, a, far, "A")
    assert float(l3.detach()) < float(l1.detach())
    # and the wrong order is penalised more than a tie
    m4 = _FixedModel({0.0: -4.0, 100.0: 0.0})           # A loses by 4
    l4, _ = pairwise_route_loss(m4, None, None, a, b, "A")
    assert float(l4.detach()) > float(l1.detach())


def test_optimising_a_preference_orders_the_lanes():
    model = _StubModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.5)
    for _ in range(60):
        info = pairwise_route_step(model, opt, None, None, LANE_A, LANE_B, "A")
    assert info["logit_a"] > info["logit_b"], info
    # and the reverse verdict orders them the other way
    model2 = _StubModel()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.5)
    for _ in range(60):
        info2 = pairwise_route_step(model2, opt2, None, None, LANE_A, LANE_B,
                                    "B")
    assert info2["logit_b"] > info2["logit_a"], info2


def test_neither_rejects_both():
    model = _StubModel()
    with torch.no_grad():
        model.w += torch.tensor([2.0, 2.0])       # both lanes score high
    opt = torch.optim.AdamW(model.parameters(), lr=0.5)
    for _ in range(80):
        info = pairwise_route_step(model, opt, None, None, LANE_A, LANE_B,
                                   "neither")
    assert info["formulation"] == "neither(reject-both)"
    assert info["logit_a"] < 0.5 and info["logit_b"] < 0.5, info
    assert max(info["logit_a"], info["logit_b"]) < 0.5


def test_unknown_preference_is_refused():
    model = _StubModel()
    with pytest.raises(ValueError):
        pairwise_route_loss(model, None, None, LANE_A, LANE_B, "")
    with pytest.raises(ValueError):
        pairwise_route_loss(model, None, None, LANE_A, LANE_B, "maybe")
