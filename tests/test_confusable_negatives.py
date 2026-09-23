"""rev8: a wrong-instance activation must COST something.

Measured shortcut this pins: in a crop holding three labelled tubes,
predicting all three cost the body loss ~0.003 while predicting only
the queried one cost ~0.000 — because the negative term is averaged
over the whole crop, where the other tubes are ~1% of the pixels.
Training therefore collapsed to a query-invariant "every tube-like
pixel" answer, which is exactly what the swap matrix caught (identical
rows; a background query returning a grain query's row).

Fix: other LABELLED tubes in the same crop are confusable negatives
and carry CONFUSABLE_W extra weight, so covering them is expensive.
The withheld-region guarantee must be unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.train import (  # noqa: E402
    CONFUSABLE_W, masked_multihead_loss)

H = W = 96


def _scene():
    """Own tube (rows 30-33) plus a neighbouring tube (rows 60-63)."""
    own = torch.zeros(1, 1, H, W)
    own[0, 0, 30:34, 8:40] = 1.0
    other = torch.zeros(1, 1, H, W)
    other[0, 0, 60:64, 8:40] = 1.0
    valid = torch.ones(1, 1, H, W)
    return own, other, valid


def _varying_logits():
    """Fires on BOTH tubes, quiet elsewhere — the shortcut the swap test
    caught (a query-invariant 'every tube-like pixel' answer)."""
    logits = torch.full((1, 1, H, W), -2.0)
    logits[0, 0, 30:34, 8:40] = 2.0     # own tube (target 1)
    logits[0, 0, 60:64, 8:40] = 2.0     # the neighbouring tube (target 0)
    return logits


def _body_loss(logits, confusable=None):
    own, other, valid = _scene()
    masks = {"body_valid": valid,
             "apex_pos": torch.zeros(1, 1, H, W),
             "apex_neg": torch.zeros(1, 1, H, W)}
    if confusable is not None:
        masks["body_confusable"] = confusable
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": own, "apex_heat": torch.zeros(1, 1, H, W)}, masks)
    return float(out["body"])


def test_covering_another_tube_costs_materially_more_with_confusables():
    _, other, _ = _scene()
    without = _body_loss(_varying_logits(), confusable=None)
    with_conf = _body_loss(_varying_logits(), confusable=other)
    assert with_conf > without * 1.5, (without, with_conf)


def test_confusable_weight_is_bounded_and_declared():
    _, other, _ = _scene()
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    own, _o, valid = _scene()
    masks = {"body_valid": valid,
             "body_confusable": other,
             "apex_pos": torch.zeros(1, 1, H, W),
             "apex_neg": torch.zeros(1, 1, H, W)}
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": own, "apex_heat": torch.zeros(1, 1, H, W)}, masks)
    out["total"].backward()
    g = logits.grad[0, 0]
    on_other = g[60:64, 8:40].abs().mean()
    far_bg = g[80:90, 8:40].abs().mean()
    ratio = float(on_other / far_bg)
    assert abs(ratio - (1.0 + CONFUSABLE_W)) < 0.5, ratio


def test_withheld_pixels_still_get_zero_even_with_confusables():
    own, other, valid = _scene()
    valid = valid.clone()
    valid[0, 0, :, 70:] = 0.0
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": own, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": valid, "body_confusable": other,
         "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    out["total"].backward()
    assert float(logits.grad[0, 0, :, 70:].abs().max()) == 0.0


def test_foreign_labelled_tube_is_known_negative_not_unknown():
    """H303 contract: a labelled foreign tube gets validity 1, target 0.

    The bug this pins: the foreign tube lay outside the queried
    mask's reviewed extent (validity 0, i.e. "unknown"), so the
    negative weight multiplied zero and an A/B on the weight came
    out byte-identical.
    """
    from prototypes.v30_video_apex.targets import add_confusable_validity
    valid = np.zeros((6, 6), np.float32)
    valid[0, 0] = 1.0
    foreign = np.zeros((6, 6), np.float32)
    foreign[3:5, 2:5] = 1.0
    out = add_confusable_validity(valid, foreign)
    # foreign region now supervised, as a negative
    assert out[3, 2] == 1.0 and out[4, 4] == 1.0
    # the queried tube's own pixels are untouched, and nothing else
    # became valid
    assert out[0, 0] == 1.0 and out[1, 1] == 0.0
    # unknown material stays unknown
    assert out[2, 0] == 0.0
    # idempotent, and a None confusable is a no-op
    assert np.array_equal(add_confusable_validity(out, foreign), out)
    assert add_confusable_validity(valid, None) is valid
    # shape mismatch is a programming error, not a silent broadcast
    try:
        add_confusable_validity(valid, np.zeros((3, 3), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")
