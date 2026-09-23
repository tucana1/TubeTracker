"""Measurement hygiene for the whole-instance criterion (rev8).

Two contracts, both learned from a real failure this session:

1. The swap summary must not flatter a dead model. An all-zero matrix
   scored "self-peaked=3/3" because `0 >= 0` holds for every column,
   so the weight-10 run -- which had collapsed to predicting nothing
   (held-out IoU 0.000) -- read as a passing swap test.
2. The balanced confusable mode must never let the wrong-instance
   penalty carry more total weight than the paint, which is what made
   a fixed 10x collapse the body head on the full dataset (83k px of
   other tubes against ~1-5k px of paint).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from prototypes.v30_video_apex.train import (  # noqa: E402
    confusable_neg_weights)


def _swap(iou):
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_whole_instance import summarize_swap
    return summarize_swap(iou, ["g0", "g1", "g2"])


def test_all_zero_swap_is_degenerate_not_a_pass():
    zero = {a: {b: 0.0 for b in ("g0", "g1", "g2")} for a in ("g0", "g1", "g2")}
    s = _swap(zero)
    assert s["degenerate"] is True
    assert s["rows_peaking_on_own_tube"] == 0, s
    assert s["max_row_spread_iou"] == 0.0


def test_strict_diagonal_still_counts_as_peaking():
    good = {"g0": {"g0": 0.4, "g1": 0.0, "g2": 0.0},
            "g1": {"g0": 0.0, "g1": 0.3, "g2": 0.0},
            "g2": {"g0": 0.0, "g1": 0.0, "g2": 0.35}}
    s = _swap(good)
    assert s["degenerate"] is False
    assert s["rows_peaking_on_own_tube"] == 3, s


def test_ties_are_not_evidence():
    """A query-invariant row (diag == off-diag) is the failure itself."""
    flat = {"g0": {"g0": 0.18, "g1": 0.18, "g2": 0.18},
            "g1": {"g0": 0.18, "g1": 0.18, "g2": 0.18},
            "g2": {"g0": 0.18, "g1": 0.18, "g2": 0.18}}
    s = _swap(flat)
    assert s["rows_peaking_on_own_tube"] == 0, s
    assert s["max_row_spread_iou"] == 0.0


def test_balance_mode_caps_the_confusable_mass_at_the_paint():
    neg = torch.ones(1, 1, 40, 40)
    conf = torch.zeros(1, 1, 40, 40)
    conf[0, 0, :4, :] = 1.0                 # 160 confusable px
    n_pos = torch.tensor(16.0)              # 16 px of paint
    w = confusable_neg_weights(neg, conf, 10.0, "balance", n_pos)
    extra = float((w - neg)[conf > 0].sum())
    assert extra <= float(n_pos) * 1.001, extra      # capped (float32)
    assert extra > 0.0                              # but present
    # fixed mode keeps the per-pixel multiplier, which is the mode that
    # collapsed the full-data run
    wf = confusable_neg_weights(neg, conf, 10.0, "fixed", n_pos)
    assert abs(float((wf - neg)[conf > 0].sum()) - 10.0 * 160.0) < 1e-3


def test_no_confusable_mask_is_a_no_op():
    neg = torch.ones(1, 1, 8, 8)
    assert torch.equal(confusable_neg_weights(neg, None, 10.0, "balance",
                                              torch.tensor(4.0)), neg)
    assert torch.equal(confusable_neg_weights(neg, torch.ones(1, 1, 8, 8),
                                              0.0, "balance",
                                              torch.tensor(4.0)), neg)


def test_dice_punishes_a_constant_field_that_bce_rewards():
    """H310: why the body loss needed an overlap term.

    With ~700 px of paint against tens of thousands of background
    pixels, plain BCE is nearly minimised by a constant field at
    p ~= paint/valid, so the head never learns to localise (measured:
    body logit std ~0.04 while the trunk's features carry 0.05-0.29).
    Dice cannot be reduced that way.
    """
    import numpy as np
    import torch
    from prototypes.v30_video_apex.train import soft_dice_loss

    t = torch.zeros(1, 1, 64, 64)
    t[0, 0, 30:34, 10:50] = 1.0            # 160 px of "tube"
    v = torch.ones_like(t)
    constant = torch.full_like(t, 0.5)      # the equilibrium BCE likes
    localised = torch.zeros_like(t)
    localised[0, 0, 29:35, 8:52] = 0.9      # roughly the tube

    d_const = float(soft_dice_loss(constant, t, v))
    d_loc = float(soft_dice_loss(localised, t, v))
    assert d_loc < d_const, (d_loc, d_const)
    # a constant well above the paint fraction must not be a good score
    assert d_const > 0.5, d_const
    # all-zero background is not a perfect score either (eps floor)
    assert float(soft_dice_loss(torch.zeros_like(t), t, v)) > 0.9
