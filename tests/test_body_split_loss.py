"""rev8 step 3: the body loss must not PREFER a low constant.

Measured failure (single-sample plumbing test): trained 40 updates on
one painted tube, the head reached proximal IoU 0.92 and distal 0.00 —
it memorised the near end and left the far end at p~0.44 even though
that region is 100% tube in the example. Arithmetic showed why: a
painted tube is ~0.4% of its reviewed extent, so an extent-averaged
masked BCE is minimised by a low constant (p=0.068 with a 20x
pos_weight cap), and the far end can never win that comparison.

Fix: split normalization (the apex head's P2 convention) — each class
normalizes on its own pixel count, so a constant is minimised at p=0.5
and tube pixels and background pixels have equal say. The
withheld-region guarantee must survive unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.train import masked_multihead_loss  # noqa: E402

H = W = 96
LOG2 = math.log(2.0)


def _loss(tgt, valid, logit_value=0.0):
    logits = torch.full((1, 1, H, W), float(logit_value))
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": tgt, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": valid, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    return float(out["body"])


def _case(paint_px=300):
    t = torch.zeros(1, 1, H, W)
    t[0, 0].view(-1)[:paint_px] = 1.0        # a thin tube's worth of paint
    v = torch.ones(1, 1, H, W)               # a big reviewed extent
    return t, v


def test_a_constant_prediction_is_minimised_at_one_half():
    """The core property the old loss violated: with equal class say, the
    best constant sits at p=0.5, not near zero."""
    t, v = _case()
    low = _loss(t, v, logit_value=-2.0)     # p = 0.119
    mid = _loss(t, v, logit_value=0.0)      # p = 0.5
    high = _loss(t, v, logit_value=2.0)     # p = 0.881
    assert mid < low and mid < high, (low, mid, high)
    assert abs(mid - LOG2) < 1e-6, mid      # p=0.5 costs log 2 per class


def test_tube_pixels_and_background_have_equal_weight():
    """0.5*(pos_term + neg_term): the count ratio must not decide the
    score, or a 0.4%-positive mask would be ~250x diluted again."""
    t_small, v_small = _case(paint_px=100)
    t_big, v_big = _case(paint_px=1000)
    a, b = _loss(t_small, v_small), _loss(t_big, v_big)
    assert abs(a - b) < 1e-6, (a, b)        # same loss for 1% and 10%


def test_gradient_reaches_the_distal_pixels_proportionally():
    """Equal say = equal TOTAL gradient mass per class, which is the
    identity that makes a thin tube learnable: each side's per-pixel
    weight is inversely proportional to its pixel count.
    """
    t, v = _case()
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": t, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": v, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    out["total"].backward()
    g = logits.grad[0, 0]
    pos = t[0, 0] > 0.5
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    mass_pos = float(g[pos].sum().abs())
    mass_neg = float(g[neg].sum().abs())
    assert 0.9 < mass_pos / mass_neg < 1.1, (mass_pos, mass_neg)
    per_px = float(g[pos].mean().abs() / g[neg].mean().abs())
    assert abs(per_px - n_neg / n_pos) < 1e-3 * (n_neg / n_pos), per_px


def test_withheld_pixels_still_get_exactly_zero():
    t, v = _case()
    v[:, :, :, 60:] = 0.0
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": t, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": v, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    out["total"].backward()
    assert float(logits.grad[0, 0, :, 60:].abs().max()) == 0.0


def test_single_class_masks_do_not_divide_by_zero():
    """A mask with no negatives (or no positives) still yields a finite
    loss instead of a NaN that would poison every weight."""
    t, v = _case(paint_px=300)
    v_all_pos = torch.ones(1, 1, H, W)
    t_all = torch.ones(1, 1, H, W)          # everything painted
    a = _loss(t_all, v_all_pos)
    assert a == a and a > 0.0                # not NaN
    t_none = torch.zeros(1, 1, H, W)
    b = _loss(t_none, v)
    assert b == b and b > 0.0
