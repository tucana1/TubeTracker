"""rev10 WP-A exit evidence: the configured loss, not a helper.

Every test here corresponds to an exit requirement of the rev10 review:

  * own-only beats own+union on clump-like views (they were IDENTICAL to
    full precision before this work: 0.0012975569115951657 both, with
    all 13,896 foreign-tube exposures receiving zero loss gradient);
  * verified foreign pixels receive a downward gradient no matter how
    far they sit from the owner's band;
  * ambiguous multi-owner overlap is not a foreign negative;
  * an owned-absence sample (zero paint) keeps a real rejection term;
  * the balanced Dice actually applies its class weights (its loss must
    not move with background size).

The band in these scenes is deliberately NARROW and the foreign block
sits far from it — exactly the situation that produced the defect.
Global loss state is saved and restored around every test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex import train as T

H = W = 64


@pytest.fixture(autouse=True)
def _restore_loss_state():
    """Loss knobs are global module state; leaving any changed polluted
    unrelated tests before (6 failures in a full run, all passing in
    isolation). Save and restore the lot."""
    saved = (T.body_objective(), T.body_self_bce_weight(),
             T.body_bg_bce_weight(), T.body_foreign_bce_weight(),
             T.body_dice_balanced(), T.body_bg_region(),
             T.body_dice_region(), T.body_dice_weight(),
             T.body_pixel_bce_weight())
    yield
    T.set_body_objective(saved[0])
    T.set_body_self_bce_weight(saved[1])
    T.set_body_bg_bce_weight(saved[2])
    T.set_body_foreign_bce_weight(saved[3])
    T.set_body_dice_balanced(saved[4])
    T.set_body_bg_region(saved[5])
    T.set_body_dice_region(saved[6])
    T.set_body_dice_weight(saved[7])
    T.set_body_pixel_bce_weight(saved[8])


def _scene(zero_paint: bool = False):
    """own paint top-left, foreign block bottom-right, narrow band."""
    tgt = np.zeros((1, 1, H, W), np.float32)
    if not zero_paint:
        tgt[0, 0, 8:24, 8:16] = 1.0            # this owner's tube
    foreign = np.zeros((1, 1, H, W), np.float32)
    foreign[0, 0, 40:60, 40:56] = 1.0          # a neighbour's tube
    valid = np.ones((1, 1, H, W), np.float32)
    reviewed = np.ones((1, 1, H, W), np.float32)   # reviewed background
    band = np.zeros((1, 1, H, W), np.float32)
    band[0, 0, 6:26, 6:18] = 1.0               # narrow, near the paint
    band[0, 0, tgt[0, 0] > 0] = 0.0            # band excludes the paint
    return tgt, foreign, valid, reviewed, band


def _masks(tgt, foreign, valid, reviewed, band, overlap=None):
    m = {"body_valid": torch.from_numpy(valid),
         "body_bg_reviewed": torch.from_numpy(reviewed),
         "body_band": torch.from_numpy(band),
         "body_confusable": torch.from_numpy(foreign)}
    if overlap is not None:
        m["body_overlap"] = torch.from_numpy(overlap)
    return m


def _configure(objective="dice_selectors", bg="band", dice="extent",
               balanced=False):
    T.set_body_objective(objective)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)
    T.set_body_dice_balanced(balanced)
    T.set_body_bg_region(bg)
    T.set_body_dice_region(dice)


def test_own_only_beats_own_plus_union_on_the_configured_loss():
    """THE counterexample: these two were identical before this work."""
    tgt, foreign, valid, reviewed, band = _scene()
    logits_own = np.full((1, 1, H, W), -6.0, np.float32)
    logits_own[0, 0, 8:24, 8:16] = 6.0                 # own tube only
    logits_union = logits_own.copy()
    logits_union[0, 0, 40:60, 40:56] = 6.0             # ... plus neighbour
    _configure(bg="band")
    l_own = float(T.masked_multihead_loss(
        {"body": torch.from_numpy(logits_own)},
        {"body_mask": torch.from_numpy(tgt)},
        _masks(tgt, foreign, valid, reviewed, band),
        T.LossWeights(body=1.0))["body"])
    l_union = float(T.masked_multihead_loss(
        {"body": torch.from_numpy(logits_union)},
        {"body_mask": torch.from_numpy(tgt)},
        _masks(tgt, foreign, valid, reviewed, band),
        T.LossWeights(body=1.0))["body"])
    print(f"own-only {l_own:.6f} vs own+union {l_union:.6f}")
    assert l_union > l_own + 0.05, (
        "painting the neighbour's tube must cost strictly more than not "
        f"painting it (own {l_own}, union {l_union})")


def test_foreign_pixels_get_downward_gradient_far_from_the_band():
    """Every verified foreign pixel is pushed down, at any distance."""
    tgt, foreign, valid, reviewed, band = _scene()
    _configure(bg="band")
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    out = T.masked_multihead_loss(
        {"body": logits}, {"body_mask": torch.from_numpy(tgt)},
        _masks(tgt, foreign, valid, reviewed, band),
        T.LossWeights(body=1.0))
    out["body"].backward()
    assert logits.grad is not None, "the foreign term must be in the graph"
    gg = logits.grad.detach().numpy()[0, 0]
    fo = foreign[0, 0] > 0
    assert fo.any()
    assert float(gg[fo].min()) > 0.0, (
        "foreign pixels must carry a positive (downward) gradient even "
        f"when outside the band; got min {float(gg[fo].min())}")


def test_ambiguous_overlap_is_not_a_foreign_negative():
    """Pixels painted for two owners stay ambiguous, not negative."""
    tgt, foreign, valid, reviewed, band = _scene()
    overlap = np.zeros((1, 1, H, W), np.float32)
    overlap[0, 0, 40:50, 40:48] = 1.0        # half the foreign block
    _configure(bg="reviewed")
    # isolate the foreign term: the Dice term legitimately pulls on every
    # valid pixel, so the claim under test concerns the foreign selector.
    T.set_body_self_bce_weight(0.0)
    T.set_body_bg_bce_weight(0.0)
    T.set_body_dice_weight(0.0)
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    out = T.masked_multihead_loss(
        {"body": logits}, {"body_mask": torch.from_numpy(tgt)},
        _masks(tgt, foreign, valid, reviewed, band, overlap=overlap),
        T.LossWeights(body=1.0))
    assert "body_foreign_bce" in out, sorted(out)
    out["body"].backward()
    gg = logits.grad.detach().numpy()[0, 0]
    ov = overlap[0, 0] > 0
    pure_fo = (foreign[0, 0] > 0) & ~ov
    assert ov.any() and pure_fo.any()
    assert np.all(gg[ov] == 0.0), "overlap pixels must contribute zero"
    assert np.all(gg[pure_fo] > 0), "pure foreign pixels must still pull"


def test_owned_absence_keeps_a_rejection_term():
    """Zero paint must not delete the negatives absence needs."""
    tgt, foreign, valid, reviewed, band = _scene(zero_paint=True)
    _configure(bg="reviewed")
    logits = torch.full((1, 1, H, W), 6.0, requires_grad=True)
    out = T.masked_multihead_loss(
        {"body": logits}, {"body_mask": torch.from_numpy(tgt)},
        _masks(tgt, foreign, valid, reviewed, band),
        T.LossWeights(body=1.0))
    assert "body_foreign_bce" in out, (
        "with zero paint the foreign term must still exist; the cap floor "
        f"is {T.ABSENCE_PAINT_FLOOR} px")
    out["body"].backward()
    gg = logits.grad.detach().numpy()[0, 0]
    assert np.all(gg[foreign[0, 0] > 0] > 0), \
        "an owned-absence sample must still push foreign pixels down"


def test_balanced_dice_weights_are_actually_used():
    """The documented formula is background-invariant (~0.5)."""
    vals = []
    for n_bg in (10, 1000, 10000):
        tgt = np.zeros((1, 1, 1, 1 + n_bg), np.float32)
        tgt[0, 0, 0, 0] = 1.0
        valid = np.ones_like(tgt)
        _configure(balanced=True, bg="reviewed")
        out = T.masked_multihead_loss(
            {"body": torch.zeros(1, 1, 1, 1 + n_bg)},
            {"body_mask": torch.from_numpy(tgt)},
            {"body_valid": torch.from_numpy(valid)},
            T.LossWeights(body=1.0))
        assert "body_dice" in out, sorted(out)
        vals.append(float(out["body_dice"]))
    print("balanced dice across backgrounds:", [round(v, 6) for v in vals])
    assert abs(vals[0] - vals[-1]) < 1e-4, (
        "the balanced Dice must not move with background size: "
        f"{vals}")


def test_temporal_activation_is_leaky_not_dead():
    """H347: a ReLU there zeroed every head; it must stay leaky."""
    import prototypes.v30_video_apex.model as M
    src = open(M.__file__).read()
    assert "leaky_relu" in src, "temporal activation must be leaky"
    # a BARE relu on the temporal conv is the defect; "leaky_relu" is fine
    for bare in ("_t.relu(self.temporal", "torch.relu(self.temporal"):
        assert bare not in src, (
            "the dead ReLU is back: pre-activations were measured entirely "
            "negative (min -0.690, max -0.042), so relu returned exactly 0")
