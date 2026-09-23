"""Pairwise route preference (rev9 WP-A.6).

A comparison sample is a judgment about TWO lanes: "A", "B" or
"neither". The reviewer's warning is precise: a preferred lane is not a
certified lane. The old path branch joined duels by (movie, frame) and
trained the preferred lane with an absolute `route_t = 1.0`, which
claims *correctness* the annotator never gave. It also kept quarantined
contradictions one `status` string away from a loss.

This module implements the semantics directly:

* preference "A"/"B" -> a MARGIN loss on the model's route logits:
  the preferred lane only has to score `margin` above the other. It is
  never pushed toward "correct" in absolute terms.
* preference "neither" -> both lanes are rejected (both logits pushed
  toward 0), which is what "neither" means and what a relative label
  can never express.

The loss is computed from two forward passes (one per lane) because the
route head scores a single proposal per call.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_route_loss(model, clip, prompt, lane_a, lane_b, preference: str,
                        margin: float = 1.0, query_index: int = 4,
                        temperature: float = 1.0):
    """Margin / rejection loss for one pairwise lane judgment.

    Returns (loss, info). `info` carries both logits (detached) and the
    formulation used, so the run can report what a label actually did.
    Raises ValueError on an unknown preference — a comparison with no
    verdict is not a comparison.
    """
    pref = str(preference).strip().upper()
    if pref not in ("A", "B", "NEITHER"):
        raise ValueError(f"unknown preference {preference!r}")
    pred_a = model.forward(clip, prompt, query_index=query_index,
                           route_xy=lane_a)
    pred_b = model.forward(clip, prompt, query_index=query_index,
                           route_xy=lane_b)
    if pred_a.route_logit is None or pred_b.route_logit is None:
        raise RuntimeError("route head missing: cannot train a preference")
    la = pred_a.route_logit.reshape(1)
    lb = pred_b.route_logit.reshape(1)
    if not (bool(la.requires_grad) or bool(lb.requires_grad)):
        raise RuntimeError("route logits carry no gradient")
    t = float(temperature) if temperature else 1.0
    if pref in ("A", "B"):
        win, lose = (la, lb) if pref == "A" else (lb, la)
        # order only: win - lose >= margin (no absolute target)
        loss = F.softplus(-(win - lose) / t + float(margin) / t)
        formulation = f"margin(pref={pref}, m={margin:g})"
    else:
        # "neither": both proposals are wrong for this owner
        loss = (F.softplus(la / t) + F.softplus(lb / t)).mean()
        formulation = "neither(reject-both)"
    info = {"formulation": formulation,
            "logit_a": float(la.detach()), "logit_b": float(lb.detach()),
            "loss": float(loss.detach())}
    return loss, info


def pairwise_route_step(model, optimizer, clip, prompt, lane_a, lane_b,
                        preference: str, **kw):
    """One optimizer step on a pairwise judgment. Returns `info`."""
    optimizer.zero_grad()
    loss, info = pairwise_route_loss(model, clip, prompt, lane_a, lane_b,
                                     preference, **kw)
    if not bool(loss.requires_grad):
        raise RuntimeError("pairwise_route_step: loss carries no gradient")
    loss.backward()
    optimizer.step()
    info["backward"] = True
    return info
