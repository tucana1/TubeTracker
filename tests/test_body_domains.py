"""rev12 P0.1: licensed body-domain selectors + gradient contract.

Every supported body objective/domain combination must:
- supervise ONLY the licensed selectors (self | reviewed bg | foreign);
- give exactly zero gradient to unknown pixels (the audit's naive
  extent switch gave 1,373 unknown gradients for g1; the repaired
  path must not);
- keep self-positive pixels positive even inside ambiguous overlap;
- abstain (unknown, zero gradient) on foreign pixels inside
  ambiguous multi-owner overlap;
- never let the geometric band grant review status (band pixels
  outside the reviewed extent stay unknown);
- sample reviewed negatives inside and outside the band with equal
  mass under the balanced 'strata' region.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

H = W = 40


@pytest.fixture(autouse=True)
def _restore_body_state():
    """The body-loss knobs are GLOBAL module state; restore the lot
    around every test so no other file sees a polluted configuration."""
    from prototypes.v30_video_apex import train as T
    saved = (T.body_objective(), T.body_dice_balanced(),
             T.body_dice_region(), T.body_bg_region(),
             T.body_dice_weight(), T.body_self_bce_weight(),
             T.body_bg_bce_weight(), T.body_foreign_bce_weight())
    yield
    T.set_body_objective(saved[0])
    T.set_body_dice_balanced(saved[1])
    T.set_body_dice_region(saved[2])
    T.set_body_bg_region(saved[3])
    T.set_body_dice_weight(saved[4])
    T.set_body_self_bce_weight(saved[5])
    T.set_body_bg_bce_weight(saved[6])
    T.set_body_foreign_bce_weight(saved[7])


def _toy_targets():
    from prototypes.v30_video_apex.batch_builder import (
        BodyTargets, finalize_body_supervision)
    target = np.zeros((H, W), np.float32)
    target[5:9, 5:9] = 1.0
    fg = target > 0
    # reviewed extent minus paint: INCLUDES the band ring (the real
    # channel relation: band <= bg for g0/g2; g1 has unreviewed band
    # pixels outside bg — mirrored by the separate band-only test).
    bg = np.zeros((H, W), bool)
    bg[2:20, 2:20] = True
    bg &= ~fg
    band = np.zeros((H, W), bool)
    band[4:10, 4:10] = True
    band &= ~fg
    unknown = ~(fg | bg)
    valid = np.zeros((H, W), np.float32)
    valid[fg | band | bg] = 1.0
    ch = {"target": target, "valid": valid, "fg": fg, "band": band,
          "bg_reviewed": bg, "unknown": unknown, "extent_used": True,
          "quarantine_reason": "ok"}
    bt = BodyTargets.from_channels(ch, source="toy")
    conf = np.zeros((H, W), bool)
    conf[12:16, 12:16] = True
    ov = np.zeros((H, W), bool)
    ov[14:18, 12:16] = True          # lower half of the foreign blob
    bt = finalize_body_supervision(bt, confusable=conf, overlap=ov)
    return bt, conf, ov


def _loss_grad(bt, conf, ov, objective, bg_region, dice_region,
               prob=0.5):
    from prototypes.v30_video_apex import train as T
    T.set_body_objective(objective)
    T.set_body_dice_balanced(True)
    T.set_body_dice_region(dice_region)
    T.set_body_bg_region(bg_region)
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)
    from prototypes.v30_video_apex.batch_builder import (
        body_mask_tensors)
    ch = body_mask_tensors(bt, confusable=conf, overlap=ov)
    masks = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))[
        None, None] for k, v in ch.items()}
    p = np.full((H, W), prob, np.float32)
    logits = torch.tensor(np.log(p / (1 - p))[None, None],
                          requires_grad=True)
    out = T.masked_multihead_loss(
        {"body": logits},
        {"body_mask": torch.from_numpy(bt.target)[None, None]},
        masks, T.LossWeights(body=1.0))
    out["total"].backward()
    return logits.grad.numpy()[0, 0]


def test_selectors_disjoint_and_complete():
    bt, _c, _o = _toy_targets()
    s = [bt.sel_self, bt.sel_bg, bt.sel_foreign, bt.sel_unknown]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (s[i] & s[j]).any(), f"selectors {i}/{j} overlap"
    total = s[0].astype(int) + s[1] + s[2] + s[3]
    assert (total == 1).all()


def test_self_positive_never_stripped_by_overlap():
    bt, _c, _o = _toy_targets()
    # paint a self pixel inside the overlap zone: it must stay self
    from prototypes.v30_video_apex.batch_builder import (
        finalize_body_supervision)
    t2 = bt.target.copy()
    t2[14, 13] = 1.0
    from dataclasses import replace
    bt2 = replace(bt, target=t2)
    bt2 = finalize_body_supervision(bt2, confusable=_c, overlap=_o)
    assert bt2.sel_self[14, 13]
    assert not bt2.sel_unknown[14, 13]


def test_foreign_overlap_abstains():
    bt, _c, _o = _toy_targets()
    # (15, 13) is confusable AND overlap, not self -> unknown
    assert bt.sel_unknown[15, 13]
    assert not bt.sel_foreign[15, 13]
    # (13, 13) is confusable, not overlap -> foreign
    assert bt.sel_foreign[13, 13]


@pytest.mark.parametrize("objective", ["split", "dice_pixel",
                                       "dice_selectors",
                                       "dice_selectors2"])
@pytest.mark.parametrize("bg_region", ["reviewed", "band", "strata"])
@pytest.mark.parametrize("dice_region", ["extent", "band"])
def test_unknown_zero_gradient_every_combination(objective, bg_region,
                                                 dice_region):
    bt, conf, ov = _toy_targets()
    grad = _loss_grad(bt, conf, ov, objective, bg_region, dice_region)
    nz = np.abs(grad) > 0
    assert not (bt.sel_unknown & nz).any(), (
        f"unknown pixels carry gradient in {objective}/{bg_region}/"
        f"{dice_region}")
    # licensed domains keep their gradients
    assert (bt.sel_self & nz).any()
    assert (bt.sel_foreign & nz).any()
    # sampled reviewed background carries gradient (all modes sample
    # at least the band part, which is nonempty in the toy scene)
    assert (bt.sel_bg & nz).any()


def test_strata_balances_inside_and_outside_band():
    """Equal per-stratum normalization: with a 4-px band part and a
    much larger far part, the gradient mass on the two strata is
    comparable (0.5/0.5), not proportional to pixel counts."""
    bt, conf, ov = _toy_targets()
    grad = _loss_grad(bt, conf, ov, "dice_selectors", "strata",
                      "extent")
    band_part = bt.sel_bg & (bt.band > 0)
    far_part = bt.sel_bg & ~(bt.band > 0)
    g_band = float(np.abs(grad[band_part]).sum())
    g_far = float(np.abs(grad[far_part]).sum())
    assert g_band > 0 and g_far > 0
    ratio = g_band / g_far
    assert 0.3 < ratio < 3.0, (f"strata not balanced: {ratio:.2f} "
                               f"(band mass {g_band:.4f}, far "
                               f"{g_far:.4f})")


def test_band_never_grants_review_status():
    """A band pixel OUTSIDE the reviewed extent must stay unknown with
    zero gradient in every mode — the band samples, never licenses."""
    from prototypes.v30_video_apex.batch_builder import (
        BodyTargets, finalize_body_supervision)
    target = np.zeros((H, W), np.float32)
    target[5:9, 5:9] = 1.0
    fg = target > 0
    band = np.zeros((H, W), bool)
    band[4:10, 4:10] = True
    band &= ~fg
    # bg_reviewed licenses only a TINY region far from the band
    bg = np.zeros((H, W), bool)
    bg[30:36, 30:36] = True
    unknown = ~(fg | band | bg)
    valid = np.zeros((H, W), np.float32)
    valid[fg | band | bg] = 1.0
    bt = BodyTargets.from_channels({
        "target": target, "valid": valid, "fg": fg, "band": band,
        "bg_reviewed": bg, "unknown": unknown, "extent_used": True,
        "quarantine_reason": "ok"}, source="toy-band-only")
    bt = finalize_body_supervision(bt)
    # every band pixel is unknown (never granted by geometry alone):
    # implication form — band <= sel_unknown
    assert not (band & ~np.asarray(bt.sel_unknown)).any()
    grad = _loss_grad(bt, None, None, "dice_selectors", "strata",
                      "extent")
    assert not (band & (np.abs(grad) > 0)).any()


def test_selector_channels_ride_with_tensors():
    from prototypes.v30_video_apex.batch_builder import (
        body_mask_tensors)
    bt, conf, ov = _toy_targets()
    ch = body_mask_tensors(bt, confusable=conf, overlap=ov)
    for k in ("body_sel_self", "body_sel_bg", "body_sel_foreign",
              "body_sel_unknown"):
        assert k in ch
    tot = sum(ch[k] for k in ("body_sel_self", "body_sel_bg",
                              "body_sel_foreign", "body_sel_unknown"))
    assert (tot == 1).all()
