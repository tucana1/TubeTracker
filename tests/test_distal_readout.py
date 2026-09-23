"""rev8: the body readout must expose a root-only model.

A prediction that covers only the near end of a tube is exactly the
failure mode the review measured (a local receptive field cannot carry
the distal tube from a root prompt). The distal split names that case
instead of averaging it away, and the union is masked on both sides so
a prediction sprawling outside the reviewed extent cannot deflate the
score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import mask_iou_split  # noqa: E402

H = W = 128
ANCHOR = (10.0, 64.0)          # query grain at the left edge


def _tube():
    t = np.zeros((H, W), np.float32)
    t[62:67, 10:110] = 1.0     # a 100px horizontal tube from the grain
    v = np.zeros((H, W), np.float32)
    v[57:72, 5:115] = 1.0      # reviewed band around it
    return t, v


def test_perfect_prediction_scores_one():
    t, v = _tube()
    r = mask_iou_split(t > 0, t, v, anchor_xy=ANCHOR)
    assert r["mask_iou"] == 1.0
    assert r["proximal_iou"] == 1.0
    assert r["distal_iou"] == 1.0
    assert 0.0 < r["painted_distal_frac"] < 1.0


def test_root_only_prediction_is_named_as_distal_failure():
    t, v = _tube()
    pred = np.zeros((H, W), bool)
    pred[62:67, 10:50] = True   # only the first 40px from the grain
    r = mask_iou_split(pred, t, v, anchor_xy=ANCHOR)
    assert r["proximal_iou"] > 0.6, r
    assert r["distal_iou"] < 0.35, r
    assert r["distal_iou"] < r["proximal_iou"]


def test_prediction_outside_the_valid_mask_cannot_deflate():
    """A sprawling prediction must be judged inside the mask only."""
    t, v = _tube()
    pred = np.zeros((H, W), bool)
    pred[62:67, 10:110] = True
    pred[:] |= False
    pred[0:20, :] = True        # a big blob well outside the band
    r = mask_iou_split(pred, t, v, anchor_xy=ANCHOR)
    inter = float((pred & (t > 0) & (v > 0)).sum())
    union = float(((pred | (t > 0)) & (v > 0)).sum())
    assert abs(r["mask_iou"] - inter / union) < 1e-12, r
    assert r["mask_iou"] > 0.9, r


def test_degenerate_split_is_visible():
    """A stub tube with no distal paint reports frac 0, not a free pass."""
    t = np.zeros((H, W), np.float32)
    t[62:67, 10:20] = 1.0       # entirely within the 40px proximal zone
    v = np.zeros((H, W), np.float32)
    v[57:72, 5:30] = 1.0
    r = mask_iou_split(t > 0, t, v, anchor_xy=ANCHOR)
    assert r["painted_distal_frac"] == 0.0
    assert r["distal_iou"] is None, r
    assert r["proximal_iou"] == 1.0


def test_auto_split_gives_a_short_tube_a_real_distal_half():
    """An absolute 40px split makes every short tube vacuous; the auto
    split uses the median painted distance so the far end still counts.
    """
    t = np.zeros((H, W), np.float32)
    t[62:67, 10:40] = 1.0          # a 30px tube: all within 40px
    v = np.zeros((H, W), np.float32)
    v[57:72, 5:50] = 1.0
    abs_split = mask_iou_split(t > 0, t, v, anchor_xy=ANCHOR)
    assert abs_split["distal_iou"] is None, abs_split   # vacuous
    assert abs_split["painted_distal_frac"] == 0.0
    auto = mask_iou_split(t > 0, t, v, anchor_xy=ANCHOR, split_px="auto")
    assert auto["distal_iou"] == 1.0, auto
    assert auto["split_px_used"] < 40.0, auto
    assert 0.3 < auto["painted_distal_frac"] < 0.7, auto
    # a root-only prediction is still caught by the auto split
    pred = np.zeros((H, W), bool)
    pred[62:67, 10:22] = True
    root_only = mask_iou_split(pred, t, v, anchor_xy=ANCHOR,
                               split_px="auto")
    assert root_only["distal_iou"] < root_only["proximal_iou"]
