"""Shared per-sample target construction (rev9 WP-A.3).

Why this exists: the rev9 audit found the trainer's own dev probe
rebuilding body targets that the training path had already built — two
implementations free to drift — and an external evaluator feeding a
different temporal window (one frame where the trainer feeds nine).
This module is the single entry point for body-mask targets:

* `own_mask_target`    — a `body_mask` sample's own paint (training and
                         the dev probe call the SAME code)
* `linked_mask_target` — a mask reached through an observation's link
                         (the trainer's `used_mask` branch)
* `QUERY_OFFSETS` / `QUERY_INDEX` re-exported from `dataset` so no
  consumer re-derives the temporal window.

Every result carries the named channels (`fg`, `band`, `bg_reviewed`,
`unknown`), the label-source id used in `label_usage.json`, and the
extent quarantine reason — so training, dev rows and any evaluator
describe the same supervision with the same words.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
from prototypes.v30_video_apex.targets import (
    body_mask_channels_from_points, body_mask_channels_from_raster)

# The temporal window is the model's, not the consumer's: a single
# query frame sits at this index of QUERY_OFFSETS.
QUERY_INDEX = QUERY_OFFSETS.index(0)


def negative_route_masks(masks, *, query_valid: bool, certified: bool):
    """Finalize a route-only negative at the same boundary as other targets.

    An explicit zero remains authoritative in train_step_front; therefore
    the licensed negative validity belongs in this dictionary, not in a
    competing positional argument. Unreviewed routes carry no claim.
    """
    result = {k: torch.zeros_like(v) for k, v in masks.items()}
    reference = next(iter(masks.values()), torch.zeros(1))
    result["route_valid"] = reference.new_tensor([float(query_valid and certified)])
    return result

CHANNEL_KEYS = ("fg", "band", "bg_reviewed", "unknown")


@dataclass(frozen=True)
class BodyTargets:
    """Named channels + provenance for one body-mask target."""

    target: np.ndarray
    valid: np.ndarray
    fg: np.ndarray
    band: np.ndarray
    bg_reviewed: np.ndarray
    unknown: np.ndarray
    source: str
    extent_used: bool
    quarantine_reason: str
    explicit_unknown: np.ndarray | None = None
    extent_scope_policy: str = ''
    # rev12 P0.1: the explicit, disjoint supervision selectors, filled
    # by `finalize_body_supervision`. They are the AUTHORITATIVE loss
    # domains: self-positive, reviewed ordinary background,
    # foreign-exclusive negative, and unknown (zero-gradient). The
    # geometric band only ever SELECTS reviewed background for
    # balanced sampling; it never grants review status.
    sel_self: np.ndarray | None = None
    sel_bg: np.ndarray | None = None
    sel_foreign: np.ndarray | None = None
    sel_unknown: np.ndarray | None = None

    @classmethod
    def from_channels(cls, ch: dict, source: str) -> "BodyTargets":
        return cls(target=ch["target"], valid=ch["valid"], fg=ch["fg"],
                   band=ch["band"], bg_reviewed=ch["bg_reviewed"],
                   unknown=ch["unknown"], source=source,
                   explicit_unknown=ch.get("explicit_unknown"),
                   extent_scope_policy=ch.get('extent_scope_policy', ''),
                   extent_used=bool(ch["extent_used"]),
                   quarantine_reason=str(ch["quarantine_reason"]))

    def licensed(self) -> np.ndarray:
        """The licensed supervision domain (self | reviewed bg | foreign).

        Raises when the selectors were never finalized — callers must
        not silently supervise a domain that was never constructed.
        """
        if (self.sel_self is None or self.sel_bg is None
                or self.sel_foreign is None):
            raise ValueError(
                "body selectors not finalized: run "
                "finalize_body_supervision before consuming domains")
        return np.asarray(self.sel_self) | np.asarray(self.sel_bg) \
            | np.asarray(self.sel_foreign)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(target, valid) as [1, 1, H, W] float tensors."""
        t = torch.from_numpy(self.target).unsqueeze(0).unsqueeze(0)
        v = torch.from_numpy(self.valid).unsqueeze(0).unsqueeze(0)
        return t, v

    def readout(self) -> dict:
        """Small honest summary for dev rows / label_usage."""
        return {"body_source": self.source,
                "extent_used": self.extent_used,
                "extent_scope_policy": self.extent_scope_policy,
                "quarantine_reason": self.quarantine_reason,
                "reviewed_bg_px": int(self.bg_reviewed.sum()),
                "paint_px": int(self.fg.sum())}


def _channels(h: int, w: int, origin, *, raster=None, points=None,
              brush_px: float = 9.0, complete: bool = False,
              review_region=None, unknown_raster=None, review_region_provenance=None) -> dict:
    if raster:
        return body_mask_channels_from_raster(
            h, w, origin, raster, complete=complete,
            review_region=review_region, unknown_raster=unknown_raster,
            review_region_provenance=review_region_provenance)
    return body_mask_channels_from_points(
        h, w, origin, points or [], brush_px=brush_px, complete=complete,
        review_region=review_region, unknown_raster=unknown_raster,
        review_region_provenance=review_region_provenance)


def own_mask_target(sample, h: int, w: int, origin) -> BodyTargets:
    """The target for a `body_mask` sample's own paint."""
    ch = _channels(h, w, origin,
                   raster=getattr(sample, "mask_raster", None),
                   points=getattr(sample, "mask_points", None),
                   brush_px=float(getattr(sample, "brush_px", 9.0) or 9.0),
                   complete=bool(getattr(sample, "complete", False)),
                   review_region=getattr(sample, "review_region", None),
                   review_region_provenance=getattr(sample, 'review_region_provenance', None),
                   unknown_raster=getattr(sample, "mask_unknown_raster", None))
    uuid = str(getattr(sample, "mask_uuid", "")
               or getattr(sample, "obs_uuid", ""))
    kind = ("own-mask-raster:" if getattr(sample, "mask_raster", None)
            else "own-mask-stamps:")
    return BodyTargets.from_channels(ch, source=kind + uuid)


def linked_mask_target(h: int, w: int, origin, mask_record: dict,
                       link: str = "") -> BodyTargets:
    """The target for a mask reached through an observation's link.

    `mask_record` is the snapshot's mask dict (native `painted_xy`,
    optional `mask_raster`, `complete`, `review_region`).
    """
    raster = mask_record.get("mask_raster")
    ch = _channels(h, w, origin, raster=raster,
                   points=mask_record.get("painted_xy"),
                   brush_px=float(mask_record.get("brush_px", 9.0) or 9.0),
                   complete=bool(mask_record.get("complete", False)),
                   review_region=mask_record.get("review_region"),
                   review_region_provenance=mask_record.get('review_region_provenance'),
                   unknown_raster=mask_record.get("mask_unknown_raster"))
    uuid = str(mask_record.get("mask_uuid", ""))
    kind = "raster:" if raster else "stamps-legacy:"
    return BodyTargets.from_channels(ch, source=f"{kind}{uuid}|{link}")


def finalize_body_supervision(bt: BodyTargets,
                              confusable=None,
                              overlap=None) -> BodyTargets:
    """The ONE shared foreign-validity step (rev11).

    The rev11 audit: the trainer extended validity over verified foreign
    paint while the fit helper did not, so the fit test discarded 623 of
    g1's 683 known foreign pixels on snap24 — the two consumers fed
    DIFFERENT supervision to the same objective. This function is that
    step, once:

    * verified foreign-EXCLUSIVE paint (`confusable & ~overlap`) becomes
      supervised negative material: validity 1, target untouched;
    * ambiguous multi-owner overlap stays UNSUPERVISED (validity 0)
      unless it is this owner's own (self-positive) paint;
    * unknown stays unknown; self-positive pixels stay positive.

    Trainer and fit helper must both come through here; a direct call to
    `add_confusable_validity` in a consumer is a drift bug.
    """
    valid = np.asarray(bt.valid).astype(np.float32)
    if confusable is not None:
        conf = np.asarray(confusable)
        conf_excl = conf > 0
        if overlap is not None:
            conf_excl = conf_excl & ~(np.asarray(overlap) > 0)
        if conf_excl.shape != valid.shape:
            raise ValueError(
                "confusable mask shape differs from body_valid")
        valid = np.maximum(valid, conf_excl.astype(np.float32))
    if overlap is not None:
        ov = np.asarray(overlap) > 0
        if ov.shape != valid.shape:
            raise ValueError("overlap mask shape differs from body_valid")
        # ambiguous overlap must stay unsupervised — but never strip the
        # owner's own paint (self-positive) or positive target pixels
        strip = ov & ~(np.asarray(bt.fg) > 0) & ~(np.asarray(bt.target) > 0)
        valid = np.where(strip, 0.0, valid)
    # rev12 P0.1: the explicit, disjoint selectors. Mirrors the audit's
    # licensed construction exactly (audit_loss_scopes.py):
    #   self    = target > 0                      (never stripped)
    #   foreign = confusable & ~overlap & ~self   (verified foreign)
    #   bg      = bg_reviewed & ~self & ~foreign & valid
    #   unknown = everything else                 (exactly zero gradient)
    # A geometric band may later SELECT from sel_bg; it never adds to it.
    sel_self = np.asarray(bt.target) > 0
    if confusable is not None:
        _conf = np.asarray(confusable) > 0
        if overlap is not None:
            _conf = _conf & ~(np.asarray(overlap) > 0)
        sel_foreign = _conf & ~sel_self
    else:
        sel_foreign = np.zeros_like(sel_self)
    if bt.explicit_unknown is not None:
        excluded = np.asarray(bt.explicit_unknown, bool)
        valid[excluded] = 0
        sel_self &= ~excluded
        sel_foreign &= ~excluded
    sel_bg = ((np.asarray(bt.bg_reviewed) > 0) & ~sel_self
              & ~sel_foreign & (valid > 0))
    sel_unknown = ~(sel_self | sel_bg | sel_foreign)
    return replace(bt, valid=valid, sel_self=sel_self, sel_bg=sel_bg,
                   sel_foreign=sel_foreign, sel_unknown=sel_unknown)


def body_mask_tensors(bt: BodyTargets, confusable=None,
                      overlap=None) -> dict:
    """The exact per-sample BODY mask set both consumers feed the loss.

    Returns float32 arrays keyed like the trainer's `masks` dict:
    `body_valid`, `body_band`, `body_bg_reviewed`, plus
    `body_confusable`/`body_overlap` when supplied. Consumers build
    their tensors from THIS function so the same sample/crop loses the
    same pixels in the trainer and in the fit helper (rev11 item 3).
    """
    out = {"body_valid": (np.asarray(bt.valid) > 0).astype(np.float32),
           "body_band": np.asarray(bt.band).astype(np.float32),
           "body_bg_reviewed": np.asarray(bt.bg_reviewed).astype(
               np.float32)}
    if bt.sel_self is not None:
        # rev12 P0.1: the explicit disjoint selectors ride along, so
        # every consumer (trainer and fit helper) feeds the loss the
        # same AUTHORITATIVE domains: self-positive, reviewed ordinary
        # background, foreign-exclusive, unknown (zero gradient).
        out["body_sel_self"] = np.asarray(bt.sel_self).astype(np.float32)
        out["body_sel_bg"] = np.asarray(bt.sel_bg).astype(np.float32)
        out["body_sel_foreign"] = np.asarray(
            bt.sel_foreign).astype(np.float32)
        out["body_sel_unknown"] = np.asarray(
            bt.sel_unknown).astype(np.float32)
    if confusable is not None:
        out["body_confusable"] = np.asarray(confusable).astype(np.float32)
    if overlap is not None:
        out["body_overlap"] = np.asarray(overlap).astype(np.float32)
    return out
