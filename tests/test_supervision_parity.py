"""rev11 item 3: ONE supervised target builder for trainer and helper.

The rev11 audit: the trainer extended validity over verified foreign
paint while the fit helper did not, so the fit test fed DIFFERENT
supervision to the same objective — g1 on snap24 kept only 60 of its 683
known foreign pixels (462 of 683 on snap25). These tests pin the shared
step (`batch_builder.finalize_body_supervision` +
`body_mask_tensors`), its semantics, and the no-drift rule that both
consumers call it.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.batch_builder import (  # noqa: E402
    BodyTargets, body_mask_tensors, finalize_body_supervision)

SNAP24 = REPO / "runs/prototypes/v30/snapshots/snap24"
SNAP25 = REPO / "runs/prototypes/v30/snapshots/snap25"


def _mk_targets(h: int = 32, w: int = 32) -> BodyTargets:
    fg = np.zeros((h, w), bool)
    fg[4:12, 4:12] = True
    ch = {"target": fg.astype(np.float32), "valid": fg.astype(np.float32),
          "fg": fg, "band": np.zeros((h, w), bool),
          "bg_reviewed": np.zeros((h, w), bool),
          "unknown": ~fg, "extent_used": True, "quarantine_reason": ""}
    return BodyTargets.from_channels(ch, source="test")


def test_foreign_exclusive_becomes_supervised_negative():
    bt = _mk_targets()
    conf = np.zeros((32, 32), bool)
    conf[10:20, 10:20] = True
    ov = np.zeros((32, 32), bool)
    ov[14:20, 14:20] = True            # ambiguous overlap, not self paint
    out = finalize_body_supervision(bt, confusable=conf, overlap=ov)
    v = out.valid > 0
    excl = conf & ~ov & ~bt.fg
    assert excl.sum() > 0
    assert v[excl].all(), "verified foreign-exclusive pixels must be valid"
    strict_ov = ov & ~bt.fg
    assert strict_ov.sum() > 0
    assert not v[strict_ov].any(), "ambiguous overlap must stay unknown"
    assert v[bt.fg].all(), "self-positive paint stays positive"
    assert np.array_equal(out.target, bt.target), "target never changes"
    assert not v[30, 30], "untouched unknown stays unknown"


def test_overlap_is_stripped_from_a_preexisting_valid_region():
    """Overlap must never be supervised even when the base builder put
    it inside `valid`."""
    bt = replace(_mk_targets(), valid=np.ones((32, 32), np.float32))
    ov = np.zeros((32, 32), bool)
    ov[20:24, 20:24] = True
    out = finalize_body_supervision(bt, confusable=None, overlap=ov)
    v = out.valid > 0
    assert not v[ov].any()
    assert v[bt.fg].all()
    assert v[0, 0], "non-overlap valid stays valid"


def test_body_mask_tensors_matches_the_validated_channels():
    bt = _mk_targets()
    conf = np.zeros((32, 32), bool)
    conf[10:20, 10:20] = True
    ov = np.zeros((32, 32), bool)
    ov[14:20, 14:20] = True
    bt = finalize_body_supervision(bt, confusable=conf, overlap=ov)
    ch = body_mask_tensors(bt, confusable=conf, overlap=ov)
    assert np.array_equal(ch["body_valid"], (bt.valid > 0).astype(np.float32))
    assert np.array_equal(ch["body_band"], bt.band.astype(np.float32))
    assert np.array_equal(ch["body_bg_reviewed"],
                          bt.bg_reviewed.astype(np.float32))
    assert np.array_equal(ch["body_confusable"], conf.astype(np.float32))
    assert np.array_equal(ch["body_overlap"], ov.astype(np.float32))


def test_consumers_share_the_one_builder():
    """Drift guard: the trainer and the fit helper must both come
    through the shared step; neither may call the raw helper."""
    for name in ("scripts/train_v30_front.py",
                 "scripts/rev10_fit_check.py"):
        src = (REPO / name).read_text()
        assert "finalize_body_supervision(" in src, name
        assert "body_mask_tensors(" in src, name
        assert "add_confusable_validity(" not in src, \
            f"{name} bypasses the shared builder"


def _fit_scene(snapshot: Path):
    spec = importlib.util.spec_from_file_location(
        "rev10_fit_check_mod", REPO / "scripts" / "rev10_fit_check.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._scene(snapshot)


@pytest.mark.skipif(not (SNAP24.exists() and SNAP25.exists()),
                    reason="snapshots absent")
@pytest.mark.parametrize("snap,expect", [("snap24", 683), ("snap25", 683)])
def test_fit_scene_foreign_pixels_are_fully_valid(snap, expect):
    """The audited numbers: g1 has 683 known foreign pixels; the helper
    admitted 60 (snap24) / 462 (snap25). After the shared step every
    verified foreign-exclusive pixel is inside the valid mask."""
    owners = _fit_scene(REPO / f"runs/prototypes/v30/snapshots/{snap}")
    g1 = next((o for o in owners
               if str(o["owner"]).split("-")[-1] == "g1"), None)
    assert g1 is not None, "scene must contain owner g1"
    fo = g1["foreign"]
    v = g1["valid"] > 0
    assert int(fo.sum()) == expect
    assert int((v & fo).sum()) == int(fo.sum())
    # the loss-facing channels carry exactly that valid mask
    assert np.array_equal(g1["channels"]["body_valid"],
                          v.astype(np.float32))
    # overlap never becomes supervised
    ov = g1["overlap"] > 0
    assert not (v & ov & ~g1["mask"]).any()


# ------------------------------------------------- acceptance (rev11 item 3)
def _balanced_objective():
    from prototypes.v30_video_apex import train as T
    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("band")
    T.set_body_bg_region("band")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)
    return T


def test_no_query_owned_body_loss_is_zero():
    """A no-query update zeroes the body target AND its masks; under the
    full configured objective every body term must then be exactly zero
    and no gradient may flow."""
    import torch as _t
    from prototypes.v30_video_apex.train import LossWeights, \
        masked_multihead_loss
    T = _balanced_objective()
    leaf = _t.zeros(1, 1, 32, 32, requires_grad=True)
    masks = {
        "body_valid": _t.zeros(1, 1, 32, 32),
        "body_band": _t.zeros(1, 1, 32, 32),
        "body_bg_reviewed": _t.zeros(1, 1, 32, 32),
        "body_confusable": _t.zeros(1, 1, 32, 32),
    }
    out = masked_multihead_loss(
        {"body": leaf},
        {"body_mask": _t.zeros(1, 1, 32, 32)}, masks,
        LossWeights(body=1.0))
    assert float(out["total"].detach()) == 0.0
    assert float(out["body_dice"].detach()) == 0.0
    out["total"].backward()
    assert leaf.grad is None or float(leaf.grad.abs().max()) == 0.0


def test_zero_paint_owned_absence_retains_rejection():
    """A zero-paint (owned-absence) label must keep a foreign rejection
    term: the balance cap uses a paint-mass floor so capping against a
    zero paint mass cannot delete exactly the negatives absence needs."""
    import torch as _t
    from prototypes.v30_video_apex.train import (ABSENCE_PAINT_FLOOR,
                                                 LossWeights,
                                                 masked_multihead_loss)
    assert ABSENCE_PAINT_FLOOR > 0
    T = _balanced_objective()
    leaf = _t.zeros(1, 1, 32, 32, requires_grad=True)
    valid = np.zeros((1, 1, 32, 32), np.float32)
    conf = np.zeros((1, 1, 32, 32), np.float32)
    valid[0, 0, 0:16, :] = 1.0          # reviewed region, no paint
    conf[0, 0, 4:12, 4:12] = 1.0        # the neighbor's tube
    masks = {"body_valid": _t.from_numpy(valid),
             "body_band": _t.zeros(1, 1, 32, 32),
             "body_bg_reviewed": _t.from_numpy(valid),
             "body_confusable": _t.from_numpy(conf)}
    out = masked_multihead_loss(
        {"body": leaf},
        {"body_mask": _t.zeros(1, 1, 32, 32)}, masks,
        LossWeights(body=1.0))
    assert float(out["body_mass_foreign"]) > 0
    assert abs(float(out["body_foreign_bce"].detach())) > 0
    out["total"].backward()
    assert leaf.grad is not None
    g = leaf.grad.detach().numpy()[0, 0]
    assert np.abs(g[4:12, 4:12]).sum() > 0, \
        "foreign pixels must carry a rejection gradient"
