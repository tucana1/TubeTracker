"""rev8 step 3: regions that were never reviewed must generate ZERO
loss — no gradient, no effect on any head.

The consultant's exit criteria: "verify withheld regions produce zero
loss". Two failure modes this guards:

* unreviewed material silently becoming background supervision (a
  mask marked complete used to claim the WHOLE translated crop), and
* a ball-scoped "can't tell" verdict leaking out as a full-frame
  negative (the P0/H242 rule).

These run the real target builders and the real loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    apex_pos_neg_masks, body_mask_from_raster, encode_mask_raster)
from prototypes.v30_video_apex.train import masked_multihead_loss  # noqa: E402

H = W = 64


def _raster():
    """A painted strip, encoded with the real (pixel-exact) encoder."""
    paint = np.zeros((H, W), dtype=bool)
    paint[0:11, 0:11] = True
    return encode_mask_raster(paint)


def test_distant_background_is_withheld_not_negative():
    """A complete mask licenses background INSIDE its reviewed extent
    only; a crop that extends past the extent keeps that material
    unknown."""
    region = [(0, 0), (32, 0), (32, 32), (0, 32)]     # top-left quadrant
    tgt, valid = body_mask_from_raster(
        H, W, (0, 0), _raster(), complete=True, review_region=region)
    assert tgt[5, 5] == 1.0 and valid[5, 5] == 1.0        # painted
    assert tgt[20, 20] == 0.0 and valid[20, 20] == 1.0    # reviewed band
    assert valid[50, 50] == 0.0, "unreviewed material is unknown"
    assert tgt[50, 50] == 0.0

    # Without an extent, even a 'complete' mask claims nothing beyond
    # the paint band: unreviewed background stays unknown.
    _, valid_none = body_mask_from_raster(
        H, W, (0, 0), _raster(), complete=True, review_region=None)
    assert valid_none[50, 50] == 0.0
    assert valid_none[20, 20] == 0.0, "no extent -> no background claim"


def test_withheld_pixels_get_no_gradient_and_no_loss():
    region = [(0, 0), (24, 0), (24, 24), (0, 24)]
    tgt, valid = body_mask_from_raster(
        H, W, (0, 0), _raster(), complete=True, review_region=region)
    t = torch.from_numpy(tgt)[None, None]
    v = torch.from_numpy(valid)[None, None]
    logits = torch.zeros(1, 1, H, W, requires_grad=True)

    out = masked_multihead_loss(
        {"body": logits, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": t, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": v, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    out["total"].backward()
    g = logits.grad[0, 0]
    outside = v[0, 0] == 0
    assert outside.any()
    assert float(g[outside].abs().max()) == 0.0, \
        "withheld pixels must receive exactly zero gradient"

    # ...and the loss value cannot depend on what sits there: corrupt
    # the target outside the valid mask and re-run.
    t2 = t.clone()
    t2[0, 0][outside] = 1.0 - t2[0, 0][outside]
    out2 = masked_multihead_loss(
        {"body": logits.detach(), "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": t2, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": v, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    out1 = masked_multihead_loss(
        {"body": logits.detach(), "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_mask": t, "apex_heat": torch.zeros(1, 1, H, W)},
        {"body_valid": v, "apex_pos": torch.zeros(1, 1, H, W),
         "apex_neg": torch.zeros(1, 1, H, W)})
    assert float(out1["body"]) == float(out2["body"]), \
        "loss must be invariant to targets outside the valid mask"


def test_ball_scoped_verdict_never_becomes_a_frame_negative():
    """A cap-scoped "can't tell" is a bounded box, not a frame claim.

    APEX_NEGATIVE_SCOPES = {apex, tip, cap}: the verdict must name the
    object it is about, and a scope outside that set is ignored.
    """
    box_xy = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
    regs = [{"kind": "verified_negative", "class_scope": "cap",
             "polygon_xy": box_xy},
            # a scope outside the declared set must be ignored
            {"kind": "verified_negative", "class_scope": "movie",
             "polygon_xy": [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0]]}]
    pos, neg = apex_pos_neg_masks(H, W, (0, 0), None, regs)
    assert pos.sum() == 0.0
    assert neg[20, 20] == 1.0, "the scoped disc is the negative"
    assert neg[50, 50] == 0.0, "outside the verdict is NOT negative"
    assert float(neg.sum()) <= 21 * 21 + 1, \
        "negative must stay inside the stated box, never the frame"

    # a real tip inside a 'can't tell' box stays unknown, not negative
    pos2, neg2 = apex_pos_neg_masks(H, W, (0, 0), (20.0, 20.0), regs)
    assert pos2[20, 20] == 1.0 and neg2[20, 20] == 0.0
