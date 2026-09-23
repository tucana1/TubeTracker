"""v30 train: masked multi-head loss, frozen splits, immutable checkpoints.

Headless: torch import guarded for metadata helpers; loss needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# rev8: how much extra weight a false positive on ANOTHER labelled
# tube carries in the body loss (a wrong-instance activation is the
# error the swap test measures, and plain background averaging hides it)
CONFUSABLE_W = 10.0


def confusable_weight() -> float:
    """Effective confusable weight (0.0 disables the term entirely).

    Declared as a function so the A/B can switch it without editing the
    loss: `set_confusable_weight(0.0)` reproduces the pre-rev8 body loss
    exactly.
    """
    return _CONFUSABLE_W_STATE[0]


def set_confusable_weight(w: float) -> None:
    _CONFUSABLE_W_STATE[0] = float(w)


_CONFUSABLE_W_STATE = [CONFUSABLE_W]

# rev8: HOW the confusable weight is applied. "fixed" multiplies every
# confusable pixel by CONFUSABLE_W; "balance" scales the square so the
# confusable pixels carry at most as much TOTAL weight as the paint.
# Why this exists (measured, full dataset): at a fixed 10x the body
# head collapsed to predicting nothing -- 24 epochs on snap18 with
# weight 10 gave held-out IoU 0.000 at ep11, ep15 and the final
# checkpoint, while weight 0 gave 0.217 at ep11. The crops hold ~83k px
# of other tubes against ~1-5k px of paint, so a per-pixel 10x put
# 10-20x the positive mass on "not-my-tube" and silence was the cheap
# answer. The matched 2-grain test had picked 0.183 -> 0.062 off-diag
# at that weight, which is exactly the small-N trap the ledger warns
# about: the mechanism transferred, the magnitude did not.
_CONFUSABLE_MODE_STATE = ["fixed"]


def confusable_mode() -> str:
    return _CONFUSABLE_MODE_STATE[0]


def set_confusable_mode(mode: str) -> None:
    m = str(mode).strip().lower()
    if m not in ("fixed", "balance"):
        raise ValueError("confusable mode must be 'fixed' or 'balance'")
    _CONFUSABLE_MODE_STATE[0] = m


@dataclass
class LossWeights:
    apex: float = 1.0
    apex_neg: float = 1.0  # verified-negative term, own denominator
    body: float = 1.0  # rev6: visible-body BCE, own valid mask
    visibility: float = 0.5
    route: float = 0.5
    front: float = 0.5
    aux_mask: float = 0.25
    route_correct: float = 0.5  # route-head BCE (true vs wrong routes)
    # rev11: explicit owned-cap presence/rejection BCE (0.0 = head not
    # trained; old runs are unaffected by construction).
    front_present: float = 0.0
    grain: float = 0.0  # S1 diagnostic default: grain objective off


# rev8 H310: weight of the soft-Dice term added to the body loss
# (0.0 = plain BCE, i.e. the behaviour every run in this session had).
# Why it exists: with ~700 px of paint against 16k-83k px of
# background, plain BCE is nearly minimised by a SPATIALLY CONSTANT
# field at p ~= paint/valid, so the body head has almost no incentive
# to localise (measured: logit std ~0.04 while the trunk's features
# carry std 0.05-0.29). Dice cannot be reduced by a constant, so it
# supplies exactly the missing gradient.
BODY_DICE_W = 0.0


def body_dice_weight() -> float:
    return _BODY_DICE_STATE[0]


def set_body_dice_weight(w: float) -> None:
    _BODY_DICE_STATE[0] = float(w)


_BODY_DICE_STATE = [BODY_DICE_W]


# rev8 H312: how the body term is assembled.
#   "split"      -- per-class mass normalisation (H298). Measured to
#                   have a degenerate FLAT VALLEY: the uniform
#                   direction has exactly zero net gradient, so the
#                   model oscillates between all-foreground and
#                   all-background at loss 0.6931 forever (the entropy
#                   of a coin flip) and never learns structure.
#   "dice_pixel" -- soft Dice plus a SMALL pixel-normalised BCE. Dice
#                   cannot be reduced by a constant and its gradient
#                   points at the tube; measured to escape the valley
#                   (logit std 0.001 -> 4.4 in 1500 steps) though the
#                   rate still needs ~10-20k updates to fit ONE tube.
BODY_OBJECTIVE = "split"
_BODY_OBJ_STATE = [BODY_OBJECTIVE]


_BODY_BALANCED = [False]


def set_body_dice_balanced(on: bool) -> None:
    """Class-balanced Dice term.

    The plain masked Dice divides by the whole reviewed region, so with
    a ~30k-px reviewed background its per-pixel pull on a ~600-px paint
    is ~1/30k — measured: both rev9 WP-B arms sat all-background at
    equal updates while the same objective escaped under the old
    ~2k-px regions (H341). Balanced Dice weights each class by its own
    mass (w = 1/(2 n_pos) on paint, 1/(2 n_neg) elsewhere, unknown 0),
    so the paint's per-pixel pull is ~50x stronger, the truth is the
    optimum (loss 0), all-one costs 0.33 and all-zero costs 1.0.
    """
    _BODY_BALANCED[0] = bool(on)


_BODY_DICE_REGION = ["extent"]

# rev10 WP-A: owned-absence samples carry no paint, so a foreign-term cap
# proportional to paint mass would silence their only useful gradient.
# The floor keeps a real (bounded) rejection term for them.
ABSENCE_PAINT_FLOOR = 500.0

_BODY_BG_REGION = ["reviewed"]


def set_body_bg_region(mode: str) -> None:
    """Which pixels the reviewed-background BCE terms use.

    rev12 P0.1 — all three modes draw ONLY from the licensed
    reviewed-background selector (sel_bg); the band never grants
    review status, it only changes SAMPLING:

    "strata" (repaired default) = balanced sampling of licensed
    reviewed negatives INSIDE and OUTSIDE the paint-adjacent band,
    each stratum normalized on its own mass (0.5/0.5), so the far
    reviewed background — including currently firing false-positive
    regions — carries half the background pull instead of being
    diluted into a ~40k-px mean.

    "reviewed" = every licensed reviewed-background pixel, one mean.

    "band" = legacy band-only sampling (kept for reproducing old
    configurations; still licensed-only under the selectors).
    """
    if mode not in ("reviewed", "band", "strata"):
        raise ValueError(
            "body bg region must be 'reviewed', 'band' or 'strata'")
    _BODY_BG_REGION[0] = mode


def body_bg_region() -> str:
    return _BODY_BG_REGION[0]


def set_body_dice_region(mode: str) -> None:
    """Where the Dice term is evaluated.

    "extent" (default) = the whole reviewed region.
    "band" = only the paint plus its 13.5 px surround, i.e. learn3's
    supervision scale. learn3 trained under QUARANTINED extents, so its
    `valid` was paint+band (~2-3k px) and its Dice was never diluted by
    tens of thousands of reviewed-background pixels. Every rev9 arm
    trains under the corrected extents (40k px) and stays constant while
    learn3 moved; the reviewed extent still defines what is VERIFIED and
    evaluation still measures over it in full, so focusing the loss is
    not the same as shrinking the claim.

    Diagnostics that motivated the knob (H346): trunk lr 0.01 leaves the
    body field flat (std 1.2e-07) and 0.03 diverges (std 2.5e+07), so
    the optimizer is not the blocker; the supervision scale is.
    """
    if mode not in ("extent", "band"):
        raise ValueError("body dice region must be 'extent' or 'band'")
    _BODY_DICE_REGION[0] = mode


def body_dice_region() -> str:
    return _BODY_DICE_REGION[0]


def body_dice_balanced() -> bool:
    return bool(_BODY_BALANCED[0])


def body_objective() -> str:
    return _BODY_OBJ_STATE[0]


def set_body_objective(name: str) -> None:
    n = str(name).strip().lower()
    if n not in ("split", "dice_pixel", "dice_selectors",
                 "dice_selectors2"):
        raise ValueError("body objective must be 'split', 'dice_pixel', "
                         "'dice_selectors' or 'dice_selectors2'")
    _BODY_OBJ_STATE[0] = n


# rev9 WP-B: the review's objective — masked Dice plus SEPARATELY
# normalized BCE contributions from three explicit, disjoint
# selectors: self foreground, reviewed ordinary background, and
# foreign-exclusive background (`foreign & reviewed & ~self_positive`).
# Unknown pixels contribute zero by construction. Coefficients are
# engineering starting values (the review's own words), not truths.
BODY_SELF_BCE_W = 0.5
BODY_BG_BCE_W = 0.15
BODY_FOREIGN_BCE_W = 1.0


def body_self_bce_weight() -> float:
    return _BODY_SELF_BCE_STATE[0]


def set_body_self_bce_weight(w: float) -> None:
    _BODY_SELF_BCE_STATE[0] = float(w)


def body_bg_bce_weight() -> float:
    return _BODY_BG_BCE_STATE[0]


def set_body_bg_bce_weight(w: float) -> None:
    _BODY_BG_BCE_STATE[0] = float(w)


def body_foreign_bce_weight() -> float:
    return _BODY_FOREIGN_BCE_STATE[0]


def set_body_foreign_bce_weight(w: float) -> None:
    _BODY_FOREIGN_BCE_STATE[0] = float(w)


_BODY_SELF_BCE_STATE = [BODY_SELF_BCE_W]
_BODY_BG_BCE_STATE = [BODY_BG_BCE_W]
_BODY_FOREIGN_BCE_STATE = [BODY_FOREIGN_BCE_W]


BODY_PIXEL_BCE_W = 0.1


def body_pixel_bce_weight() -> float:
    return _BODY_PIXEL_BCE_STATE[0]


def set_body_pixel_bce_weight(w: float) -> None:
    _BODY_PIXEL_BCE_STATE[0] = float(w)


_BODY_PIXEL_BCE_STATE = [BODY_PIXEL_BCE_W]


def body_loss_config() -> dict:
    """The complete effective body-loss configuration (rev12 P0.1).

    Serialized into run manifests so any comparison can be checked
    mechanically (the P0.4 manifest comparator) instead of described
    in prose. `selectors` names the domain construction: with the
    explicit licensed selectors the band only samples reviewed
    background, and unknown pixels receive zero gradient in every
    objective/domain combination.
    """
    return {
        "objective": body_objective(),
        "dice_region": body_dice_region(),
        "dice_balanced": bool(body_dice_balanced()),
        "bg_region": body_bg_region(),
        "selectors": "licensed-explicit",
        "weights": {
            "dice": float(body_dice_weight()),
            "self_bce": float(body_self_bce_weight()),
            "bg_bce": float(body_bg_bce_weight()),
            "foreign_bce": float(body_foreign_bce_weight()),
            "pixel_bce": float(body_pixel_bce_weight()),
            "confusable": float(confusable_weight()),
            "confusable_mode": str(confusable_mode()),
        },
    }


def soft_dice_loss(prob, target, valid, eps=1.0):
    """1 - Dice on the positive class, restricted to `valid`.

    `eps` is a smoothing floor; without it a prediction of all-zero
    background would be a perfect Dice score.
    """
    p = (prob * valid).reshape(-1)
    t = (target * valid).reshape(-1)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def confusable_neg_weights(neg_sel, conf, cw, mode, n_pos):
    """Weight the negative selector, extra-weighting confusable pixels.

    Contract (tested): in "balance" mode the extra weight carried by the
    confusable pixels never exceeds the positive mass `n_pos`, so a
    wrong-instance penalty can never swamp the tube being learned; in
    "fixed" mode each confusable pixel simply carries `cw` times the
    background weight.
    """
    if conf is None or float(cw) <= 0:
        return neg_sel
    conf = conf.to(dtype=neg_sel.dtype)
    if str(mode) == "balance":
        cmass = float((conf * neg_sel).sum())
        pmass = float(n_pos)
        scale = float(cw) if cmass <= 0 else min(float(cw), pmass / cmass)
        return neg_sel * (1.0 + scale * conf)
    return neg_sel * (1.0 + float(cw) * conf)


def masked_multihead_loss(pred: dict, target: dict, masks: dict,
                           weights: LossWeights | None = None) -> dict:
    """Each head normalized over its own valid mask so background can't dominate."""
    if torch is None:
        raise ImportError("masked_multihead_loss requires torch")
    import torch.nn.functional as F

    weights = weights or LossWeights()
    out: dict[str, Any] = {}
    total = 0.0

    def add(name: str, loss, w: float):  # type: ignore[no-untyped-def]
        nonlocal total
        out[name] = loss
        total = total + w * loss

    if "apex_heat" in pred and "apex_heat" in target:
        diff2 = (pred["apex_heat"] - target["apex_heat"]) ** 2
        if "apex_pos" in masks or "apex_neg" in masks:
            # Split normalization (P2): positives and verified
            # negatives each carry their own denominator, so abundant
            # background zeros cannot outvote the few tip pixels.
            zero = torch.zeros((), dtype=diff2.dtype,
                               device=diff2.device)
            pos_total, neg_total = zero, zero
            if "apex_pos" in masks:
                mp = masks["apex_pos"].to(dtype=diff2.dtype)
                pos_total = (diff2 * mp).sum() / mp.sum().clamp_min(1.0)
                add("apex", pos_total, weights.apex)
            if "apex_neg" in masks:
                mn = masks["apex_neg"].to(dtype=diff2.dtype)
                neg_total = (diff2 * mn).sum() / mn.sum().clamp_min(1.0)
                add("apex_neg", neg_total, weights.apex_neg)
        else:
            m = masks.get("apex_valid",
                          torch.ones_like(target["apex_heat"]))
            denom = m.sum().clamp_min(1.0)
            add("apex", (diff2 * m).sum() / denom, weights.apex)
    if "body" in pred and "body_mask" in target:
        m = masks.get("body_valid",
                      torch.zeros_like(target["body_mask"])).to(
                          dtype=pred["body"].dtype)
        tgt_b = target["body_mask"].to(dtype=pred["body"].dtype)
        # rev8 step 3: SPLIT normalization, the same convention the apex
        # head already uses (P2). Averaging one masked BCE over the
        # whole reviewed extent gives the positive class ~0.4% of the
        # weight, so the loss MINIMUM is a low constant (measured:
        # p=0.068 with a 20x pos_weight cap) and the distal tube never
        # earns a gradient it can win — verified by a single-sample
        # plumbing test: proximal IoU 0.92, distal 0.00 on the very
        # tube being memorised. Each side now normalizes on its own
        # pixels, so tube and background have equal say, and a constant
        # prediction is minimized at p=0.5 instead of near zero. Still
        # exactly zero outside the valid mask.
        bce = F.binary_cross_entropy_with_logits(
            pred["body"], tgt_b, reduction="none")
        pos_sel = tgt_b * m
        neg_sel = (1.0 - tgt_b) * m
        n_pos = pos_sel.sum()
        n_neg = neg_sel.sum()
        # rev8: CONFUSABLE negatives. Measured shortcut: in a crop with
        # three labelled tubes, predicting ALL of them costs ~0.003
        # while predicting only the queried one costs ~0.000, because
        # the negative term averages over the whole crop and the other
        # tubes are ~1% of it. Training then collapses to a
        # query-invariant "all tube-like pixels" answer (swap rows
        # identical, background query returning a grain query's row).
        # Other LABELLED tubes in the same crop are exactly the
        # confusion that must be penalised, so they carry CONFUSABLE_W.
        conf = masks.get("body_confusable")
        w_neg = confusable_neg_weights(neg_sel, conf, confusable_weight(),
                                      confusable_mode(), n_pos)
        n_wn = w_neg.sum()
        _dice_w = body_dice_weight()

        # ---- rev12 P0.1: licensed selector domains -------------------
        # When the shared builder supplies the explicit disjoint
        # selectors, they are the AUTHORITATIVE loss domains. The band
        # is then only a SAMPLING selector over licensed reviewed
        # background (it never grants review status), and unknown
        # pixels are in no term, so every objective/domain combination
        # keeps exactly zero unknown gradient.
        def _as_t(v):
            if torch is None:  # unreachable (checked at entry)
                raise ImportError("masked_multihead_loss requires torch")
            return (torch.as_tensor(v, dtype=m.dtype)
                    if not torch.is_tensor(v) else v.to(dtype=m.dtype))

        _sel_self = _as_t(masks["body_sel_self"]) \
            if "body_sel_self" in masks else None
        _sel_bg_all = _as_t(masks["body_sel_bg"]) \
            if "body_sel_bg" in masks else None
        _sel_fo_l = _as_t(masks["body_sel_foreign"]) \
            if "body_sel_foreign" in masks else None
        _bg_terms = None
        _md_domain = None
        if _sel_self is not None and _sel_bg_all is not None \
                and _sel_fo_l is not None:
            _band_m = None
            _band_ch = masks.get("body_band")
            if _band_ch is not None:
                _band_m = (_as_t(_band_ch) > 0).to(dtype=m.dtype)
            # balanced strata over LICENSED reviewed background
            if body_bg_region() == "strata" and _band_m is not None:
                _bg_terms = [_sel_bg_all * _band_m,
                             _sel_bg_all * (1.0 - _band_m)]
            elif body_bg_region() == "band" and _band_m is not None:
                _bg_terms = [_sel_bg_all * _band_m]
            else:
                _bg_terms = [_sel_bg_all]
            _bg_dice = (_sel_bg_all * _band_m
                        if (body_dice_region() == "band"
                            and _band_m is not None) else _sel_bg_all)
            _md_domain = torch.clamp(
                _sel_self + _bg_dice + _sel_fo_l, max=1.0)
            pos_sel = _sel_self
            neg_sel = torch.clamp(_sel_bg_all + _sel_fo_l, max=1.0)
            n_pos = pos_sel.sum()
            n_neg = neg_sel.sum()
            w_neg = confusable_neg_weights(neg_sel, conf,
                                           confusable_weight(),
                                           confusable_mode(), n_pos)
            n_wn = w_neg.sum()

        def _licensed_band():
            """rev11: the band may select REVIEWED background; it does not
            independently certify negative truth. Returns the licensed
            band selector, or None when there is no band mask (legacy
            full-valid behavior) or no reviewed-background license
            (fail closed — record it and drop band negatives)."""
            _band = masks.get("body_band")
            if _band is None:
                return None
            _band = (torch.as_tensor(_band, dtype=m.dtype)
                     if not torch.is_tensor(_band) else _band)
            _lic = masks.get("body_bg_reviewed")
            if _lic is None:
                out["body_band_unlicensed"] = True
                return None
            _lic = (torch.as_tensor(_lic, dtype=m.dtype)
                    if not torch.is_tensor(_lic) else _lic)
            return ((_band > 0) & (_lic > 0)).to(dtype=m.dtype)

        if body_objective() in ("dice_selectors", "dice_selectors2"):
            # rev9 WP-B, the reviewer's objective: masked Dice plus
            # SEPARATELY normalized BCE from three explicit, disjoint
            # selectors —
            #   self    = tgt * m
            #   bg_rev  = (1-tgt) * m * ~foreign     (reviewed ordinary)
            #   foreign = foreign * m * (1-tgt)      (their expression:
            #             `foreign & reviewed & ~self_positive`)
            # so every selector is disjoint by construction and unknown
            # pixels (outside `m`) contribute exactly zero. The foreign
            # term is balance-capped so its total weight can never
            # exceed the paint's (H303/H308: an uncapped foreign term
            # collapsed the model at full scale).
            _prob = torch.sigmoid(pred["body"])
            if body_dice_balanced():
                # rev10 WP-A: the class weights MUST be applied. They were
                # computed and then dropped when the dice-domain option was
                # added, which made this branch background-sensitive
                # (reproduced: 0.846 -> 0.9998 as background grows; the
                # documented formula is invariant at 0.4999995).
                _eps = 1e-6
                if _md_domain is not None:
                    # rev12 P0.1: the Dice domain is the LICENSED
                    # selector union (band only samples its bg part).
                    _md = _md_domain
                else:
                    _md = m
                    if body_dice_region() == "band":
                        # rev11: Dice negatives come from the SAME licensed
                        # band as BCE — the unlicensed paint-adjacent band
                        # received nonzero Dice gradients on 908 pixels
                        # across four current masks (audit).
                        _lb = _licensed_band()
                        if _lb is not None:
                            _md = m * ((_lb > 0) | (tgt_b > 0)).to(
                                dtype=m.dtype)
                _pos = tgt_b * _md
                _neg = (1.0 - tgt_b) * _md
                _npos = float(_pos.sum())
                _nneg = float(_neg.sum())
                _w = torch.zeros_like(m)
                if _npos > 0:
                    _w = _w + _pos / (2.0 * _npos)
                if _nneg > 0:
                    _w = _w + _neg / (2.0 * _nneg)
                _num = ((_w * _prob * tgt_b).sum())
                _den = ((_w * _prob).sum() + (_w * tgt_b).sum())
                _dice = 1.0 - (2.0 * _num + _eps) / (_den + _eps)
            elif body_objective() == "dice_selectors2":
                # Two-class Dice over the SAME valid region. The
                # one-class form has its OPTIMUM at p=1 everywhere
                # whenever the reviewed mask contains background as
                # positives (gradient check: d/dp of the one-class dice
                # is positive for every flat field). Probe evidence:
                # one-class -> pred_frac 1.0000, crop-IoU 0.0036.
                _eps = 1e-6
                _m2 = _md_domain if _md_domain is not None else m
                _pm, _tm = _prob * _m2, tgt_b * _m2
                _dfg = ((2 * (_pm * _tm).sum() + _eps)
                        / (_pm.sum() + _tm.sum() + _eps))
                _dbg = ((2 * ((_m2 - _pm) * (_m2 - _tm)).sum() + _eps)
                        / ((_m2 - _pm).sum() + (_m2 - _tm).sum() + _eps))
                _dice = 1.0 - 0.5 * (_dfg + _dbg)
            else:
                _dice = soft_dice_loss(
                    _prob, tgt_b, _md_domain if _md_domain is not None
                    else m)
            _parts = {"body_dice": _dice}
            _sel_self = pos_sel
            _n_self = _sel_self.sum()
            # rev10 WP-A: publish the selector masses the loss SAW (they
            # go to `out`, never to `_parts`, which is summed into the
            # loss value).
            out["body_mass_self"] = float(_n_self)
            if float(_n_self) > 0 and body_self_bce_weight() > 0:
                _parts["body_self_bce"] = (
                    body_self_bce_weight() * (bce * _sel_self).sum()
                    / _n_self)
            # rev10 WP-A: the band may restrict the ORDINARY-BACKGROUND
            # term only. It must NEVER gate the foreign-exclusive term —
            # every verified foreign pixel in the saved views lies
            # outside the band, so gating it made own-only and
            # own+union losses identical to full precision.
            if _bg_terms is not None:
                # rev12 P0.1: licensed reviewed background, sampled by
                # balanced strata (inside/outside the band). The band is
                # a sampling selector only; unknown pixels are absent by
                # construction (the selectors are disjoint and complete).
                _masses = [float(t.sum()) for t in _bg_terms]
                _n_bg = float(sum(_masses))
                out["body_mass_bg"] = _n_bg
                # strata masses as separate FLOAT keys (the parts dict is
                # summed numerically by consumers; a list breaks that)
                out["body_mass_bg_in_band"] = (
                    _masses[0] if _masses else 0.0)
                out["body_mass_bg_out_band"] = (
                    _masses[1] if len(_masses) > 1 else 0.0)
                if _n_bg > 0 and body_bg_bce_weight() > 0:
                    _terms = [(bce * t).sum() / t.sum().clamp_min(1.0)
                              for t, mm in zip(_bg_terms, _masses)
                              if mm > 0]
                    _parts["body_bg_bce"] = (
                        body_bg_bce_weight() * sum(_terms)
                        / max(len(_terms), 1))
                _sel_fo = _sel_fo_l
            else:
                _region = m
                if body_bg_region() == "band":
                    # rev11: band negatives are licensed only where the
                    # reviewer actually reviewed (shared with the Dice term
                    # via _licensed_band). A band with no license certifies
                    # nothing — fail closed.
                    _lb = _licensed_band()
                    if _lb is not None:
                        _region = _lb
                    elif masks.get("body_band") is not None:
                        _region = (tgt_b > 0).to(dtype=m.dtype)
                if conf is not None:
                    _cf = (conf > 0).to(dtype=m.dtype)
                    # F_i: verified foreign pixels, excluding self-positive,
                    # and excluding ambiguous multi-owner overlap.
                    _sel_fo = (1.0 - tgt_b) * m * _cf
                    _ov = masks.get("body_overlap")
                    if _ov is not None:
                        _ov = (torch.as_tensor(_ov, dtype=m.dtype)
                               if not torch.is_tensor(_ov) else _ov)
                        _sel_fo = _sel_fo * (1.0 - (_ov > 0).to(
                            dtype=m.dtype))
                    _sel_bg = (1.0 - tgt_b) * _region * (1.0 - _cf)
                else:
                    _sel_bg, _sel_fo = (1.0 - tgt_b) * _region, None
                _n_bg = _sel_bg.sum()
                out["body_mass_bg"] = float(_n_bg)
                if float(_n_bg) > 0 and body_bg_bce_weight() > 0:
                    _parts["body_bg_bce"] = (
                        body_bg_bce_weight() * (bce * _sel_bg).sum()
                        / _n_bg)
            if _sel_fo is not None:
                _n_fo = float(_sel_fo.sum())
                # rev10 WP-A: publish the selector masses the loss SAW.
                # These go to `out`, never to `_parts` (which is summed
                # into the loss value).
                out["body_mass_foreign"] = _n_fo
                if _n_fo > 0 and body_foreign_bce_weight() > 0:
                    # balance cap: foreign total <= paint mass
                    # rev10 WP-A: an owned-absence sample has zero paint
                    # and must KEEP a rejection term; capping against a
                    # paint mass of 0 would delete exactly the negatives
                    # absence needs.
                    _pmass = max(float(_n_self), ABSENCE_PAINT_FLOOR)
                    _scale = min(float(body_foreign_bce_weight()),
                                 (_pmass / _n_fo) if _pmass > 0 else 0.0)
                    if _scale > 0:
                        _parts["body_foreign_bce"] = (
                            _scale * (bce * _sel_fo).sum() / _n_fo)
            # rev10 WP-A: honour the declared Dice weight; it was ignored
            # in this branch (a `--body-dice 0.0` probe still carried a
            # 0.9407 dice term and dice gradient on foreign pixels).
            _dice = _dice * body_dice_weight()
            _parts["body_dice"] = _dice
            _bt = _dice + sum(v for k, v in _parts.items()
                              if k != "body_dice")
            for _pk, _pv in _parts.items():
                out[_pk] = _pv
            add("body", _bt, weights.body)
        elif body_objective() == "dice_pixel":
            _prob = torch.sigmoid(pred["body"])
            _pmass = m.sum().clamp_min(1.0)
            _bt = soft_dice_loss(_prob, tgt_b, m)
            if body_pixel_bce_weight() > 0:
                _bt = _bt + body_pixel_bce_weight() * (bce * m).sum() / _pmass
            add("body", _bt, weights.body)
        else:
            _dice_term = None
            if _dice_w > 0 and float(n_pos) > 0:
                _prob = torch.sigmoid(pred["body"])
                _dice_term = soft_dice_loss(_prob, tgt_b, m)
            if float(n_pos) > 0 and float(n_wn) > 0:
                _bt = 0.5 * ((bce * pos_sel).sum() / n_pos
                             + (bce * w_neg).sum() / n_wn)
                if _dice_term is not None:
                    _bt = (1.0 - _dice_w) * _bt + _dice_w * _dice_term
                add("body", _bt, weights.body)
            elif float(n_pos) > 0:
                add("body", (bce * pos_sel).sum() / n_pos, weights.body)
            else:
                add("body", (bce * neg_sel).sum() / n_neg.clamp_min(1.0),
                    weights.body)
    if "visibility_logits" in pred and "visibility" in target:
        m = masks.get("vis_valid", torch.ones(target["visibility"].shape[0]))
        ce = F.cross_entropy(pred["visibility_logits"], target["visibility"], reduction="none")
        add("visibility", (ce * m).sum() / m.sum().clamp_min(1.0), weights.visibility)
    if "route_logits" in pred and "route" in target:
        m = masks.get("route_valid", torch.ones(target["route"].shape[0]))
        ce = F.cross_entropy(pred["route_logits"], target["route"], reduction="none")
        add("route", (ce * m).sum() / m.sum().clamp_min(1.0), weights.route)
    if "front_logits" in pred and "front" in target:
        # interval supervision: -log sum q over admissible set; point = one-hot set
        logq = torch.log_softmax(pred["front_logits"], dim=-1)
        interval = target["front"].clamp_min(0)
        mass = (logq.exp() * interval).sum(dim=-1).clamp_min(1e-8)
        m = masks.get("front_valid", torch.ones_like(mass))
        add("front", (-mass.log() * m).sum() / m.sum().clamp_min(1.0), weights.front)
    if "route_logit" in pred and "route" in target:
        # Route correctness: true ribbon (1) vs wrong ribbon (0).
        # Free supervision — wrong routes are generated, never drawn.
        bce = F.binary_cross_entropy_with_logits(
            pred["route_logit"].reshape(-1),
            target["route"].reshape(-1).to(
                dtype=pred["route_logit"].dtype), reduction="none")
        m = masks.get("route_valid", torch.ones_like(bce))
        add("route_correct", (bce * m).sum() / m.sum().clamp_min(1.0),
            weights.route_correct)
    if "front_present_logit" in pred and "front_present" in target:
        # rev11: the owned-cap presence/rejection score. Separate from
        # the location softmax by construction (a conditional softmax
        # always picks a position; its maximum is not a presence
        # probability). Uncertain routes carry front_present_valid=0
        # and contribute nothing.
        bce = F.binary_cross_entropy_with_logits(
            pred["front_present_logit"].reshape(-1),
            target["front_present"].reshape(-1).to(
                dtype=pred["front_present_logit"].dtype), reduction="none")
        m = masks.get("front_present_valid", torch.ones_like(bce))
        add("front_present", (bce * m).sum() / m.sum().clamp_min(1.0),
            weights.front_present)
    if "aux_mask" in pred and "aux_mask" in target:
        m = masks.get("aux_valid", torch.ones_like(target["aux_mask"]))
        denom = m.sum().clamp_min(1.0)
        add("aux_mask", (((pred["aux_mask"] - target["aux_mask"]) ** 2) * m).sum() / denom,
            weights.aux_mask)
    out["total"] = total
    return out


def freeze_splits(groups: Sequence[str], seed: int = 0,
                  dev_frac: float = 0.2) -> dict[str, list[str]]:
    """Deterministic grouped split; frozen by seed. Groups never leak across splits."""
    import hashlib as _h

    keyed = sorted(groups, key=lambda g: _h.sha256(f"{seed}:{g}".encode()).hexdigest())
    n_dev = max(1, int(len(keyed) * dev_frac)) if len(keyed) > 1 else 0
    return {"train": keyed[: len(keyed) - n_dev], "dev": keyed[len(keyed) - n_dev:],
            "test": []}


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def save_checkpoint_immutable(path: str | Path, state: dict, config: dict,
                               manifest: dict | None = None) -> Path:
    """Write checkpoint once; refuse to overwrite (immutability)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refuse to overwrite immutable checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": {k: (v.tolist() if torch is not None and isinstance(v, torch.Tensor) else v)
                  for k, v in state.items()},
        "config": config,
        "config_hash": config_hash(config),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    resolved = {
        "checkpoint": str(path),
        "config": config,
        "config_hash": payload["config_hash"],
        "manifest": manifest or {},
    }
    path.with_suffix(".resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True,
                                                             default=str))
    return path


def train_step(model, optimizer, clip, prompt, target, masks,  # type: ignore[no-untyped-def]
               weights=None, query_index=4, route_xy=None) -> dict:
    """One REAL supervised optimizer step (rev5: no surrogate allowed).

    target/masks follow masked_multihead_loss conventions; the model
    forward keeps every gradient. Returns per-head detached losses.
    """
    if torch is None:
        raise ImportError("train_step requires torch")
    optimizer.zero_grad()
    kw = {"route_xy": route_xy} if route_xy is not None and getattr(
        model, "variant", "") == "temporal" else {}
    pred = model.forward(clip, prompt, query_index=query_index, **kw)
    pred_dict = {"apex_heat": pred.heat,
                 "visibility_logits": pred.visibility_logits,
                 "not_observed": pred.not_observed_score}
    if pred.body is not None:
        pred_dict["body"] = pred.body
    if pred.front_logits is not None:
        pred_dict["front_logits"] = pred.front_logits
    # mode tip scores supervise as an auxiliary dense signal is overkill;
    # the heads above carry the supervised loss. Modes stay inference-side.
    losses = masked_multihead_loss(pred_dict, target, masks,
                                   weights or LossWeights())
    total = losses["total"]
    if not bool(total.requires_grad):
        raise RuntimeError("train_step: loss carries no gradient "
                           "(detached outputs forbidden)")
    total.backward()
    optimizer.step()
    return {k: (float(v.detach()) if hasattr(v, "detach") else v)
            for k, v in losses.items()}


def train_step_front(model, optimizer, clip, prompt, route_xy,  # type: ignore[no-untyped-def]
                     apex_heat_t, apex_masks, front_interval, front_valid,
                     visibility_t, vis_valid, weights=None,
                     query_index=4, route_t=None, route_valid=None,
                     body_t=None, present_t=None, present_valid=None) -> dict:
    """One REAL supervised ribbon step: apex heat + front interval + vis.

    front_interval is the admissible set (1.0 near s*, else 0.0) on the
    model's own s grid (pred.front_s, detached metadata). route_t is the
    optional route-correctness label (1 true / 0 wrong ribbon). Returns
    per-head detached losses plus the selected front error in px
    (argmax s vs interval center) for monitoring — never for selection.
    """
    if torch is None:
        raise ImportError("train_step_front requires torch")
    optimizer.zero_grad()
    pred = model.forward(clip, prompt, query_index=query_index,
                         route_xy=route_xy)
    if pred.front_logits is None or pred.front_s is None:
        raise RuntimeError("train_step_front needs a route-conditioned "
                           "model (front head is None)")
    if not bool(pred.front_logits.requires_grad):
        raise RuntimeError("train_step_front: front carries no gradient "
                           "(detached outputs forbidden)")
    interval = front_interval.to(dtype=pred.front_logits.dtype)
    if interval.shape[-1] != pred.front_logits.shape[-1]:
        raise ValueError(
            f"front interval S={interval.shape[-1]} != model "
            f"S={pred.front_logits.shape[-1]} (stale grid?)")
    pred_dict = {"apex_heat": pred.heat,
                 "visibility_logits": pred.visibility_logits,
                 "front_logits": pred.front_logits}
    if pred.body is not None:
        pred_dict["body"] = pred.body
    target = {"apex_heat": apex_heat_t, "visibility": visibility_t,
              "front": interval.unsqueeze(0)
              if interval.ndim == 1 else interval}
    if body_t is not None:
        target["body_mask"] = body_t
    try:
        masks = dict(apex_masks)  # split pos/neg dict (preferred)
    except (TypeError, ValueError):
        masks = {"apex_valid": apex_masks}  # legacy single-mask tensor
    # rev13 W1: the caller's FINALIZED per-head validity masks are
    # AUTHORITATIVE. The positional arguments fill only gaps; they must
    # never overwrite an explicit zero (the audit's counterexample:
    # vis_valid=0 in the batch masks was replaced by a positional
    # ones() and the no-query gating was bypassed at the loss boundary).
    if "vis_valid" not in masks:
        masks["vis_valid"] = vis_valid
    if "front_valid" not in masks:
        masks["front_valid"] = front_valid
    if route_t is not None and pred.route_logit is not None:
        pred_dict["route_logit"] = pred.route_logit
        target["route"] = route_t.reshape(1) \
            if route_t.ndim == 0 else route_t
        if "route_valid" not in masks:
            masks["route_valid"] = torch.ones(1) if route_valid is None \
                else route_valid
    if present_t is not None and getattr(
            pred, "front_present_logit", None) is not None:
        # rev11: route-level owned-cap presence/rejection label.
        pred_dict["front_present_logit"] = pred.front_present_logit
        target["front_present"] = present_t.reshape(1) \
            if present_t.ndim == 0 else present_t
        if "front_present_valid" not in masks:
            masks["front_present_valid"] = (torch.ones(1)
                                            if present_valid is None
                                            else present_valid)
    losses = masked_multihead_loss(pred_dict, target, masks,
                                   weights or LossWeights())
    total = losses["total"]
    if not bool(total.requires_grad):
        raise RuntimeError("train_step_front: loss carries no gradient")
    total.backward()
    optimizer.step()
    out = {k: (float(v.detach()) if hasattr(v, "detach") else v)
           for k, v in losses.items()}
    # rev13 W1: the ACTUAL post-mask, post-weight contributions are
    # recorded at the loss boundary — from the FINAL masks the loss
    # consumed, not from counters computed before any overwrite. Each
    # head reports (valid mass, effective weight, consumed = mass>0 and
    # weight>0) so invalidity is auditable by reason at the consumer.
    _w = weights or LossWeights()
    _head_map = (("vis_valid", "visibility"), ("route_valid",
                                               "route_correct"),
                 ("front_valid", "front"), ("front_present_valid",
                                            "front_present"),
                 ("body_valid", "body"))
    consumed = {}
    for _mk, _wk in _head_map:
        _m = masks.get(_mk)
        if _m is None:
            continue
        try:
            _mass = float(_m.detach().sum())
        except (TypeError, ValueError):
            continue
        _wt = float(getattr(_w, _wk, 0.0))
        consumed[_mk] = {"mass": _mass, "weight": _wt,
                         "consumed": bool(_mass > 0 and _wt > 0)}
    out["consumed"] = consumed
    with torch.no_grad():
        import torch as _t

        q = _t.softmax(pred.front_logits.detach(), dim=-1)[0]
        s = pred.front_s.detach()
        s_hat = float(s[int(q.argmax())])
        idx = (interval > 0).nonzero()
        s_star = float(s[int(idx.float().mean())]) if len(idx) else float("nan")
        out["front_err_px"] = abs(s_hat - s_star)
        out["s_hat"] = s_hat
        out["s_star"] = s_star
    return out


def save_checkpoint(path: str | Path, model, optimizer=None,  # type: ignore[no-untyped-def]
                    scheduler=None, config=None, manifest=None,
                    rng_state=None) -> Path:
    """Faithful torch checkpoint: weights + optimizer/scheduler + RNG.

    Immutable (refuses overwrite). Reload via load_checkpoint reproduces
    inference bit-exactly on the same build. The old JSON-only writer
    (save_checkpoint_immutable) stays for config manifests only.
    """
    if torch is None:
        raise ImportError("save_checkpoint requires torch")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refuse to overwrite immutable checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch as _t

    # rev10 WP-A: declare the semantics that make these weights mean
    # what they mean. ReLU and LeakyReLU keep identical state-dict
    # shapes but are different functions, so a checkpoint without a
    # declared activation cannot be replayed faithfully.
    from prototypes.v30_video_apex.model_factory import (
        CURRENT_ACTIVATION, CURRENT_MODEL_SCHEMA, CURRENT_PREPROCESSING)
    config = dict(config or {})
    if getattr(model, "cap_window_head", False):
        # Semantics alter the function without altering tensor shapes.
        # Every writer, including diagnostic callers, must preserve them.
        actual = int(model.cap_window_semantics)
        if "cap_window_semantics" in config and int(config["cap_window_semantics"]) != actual:
            raise ValueError("checkpoint cap-window semantics disagree with model")
        config.update(cap_window_head=True, cap_window_semantics=actual,
                      cap_window_encoder=bool(model.cap_window_encoder))
    payload = {
        "model_state": model.state_dict(),
        "config": config or {},
        "config_hash": config_hash(config or {}),
        "manifest": manifest or {},
        "torch_version": str(_t.__version__),
        "model_schema": int(CURRENT_MODEL_SCHEMA),
        "activation": str(CURRENT_ACTIVATION),
        "preprocessing": str(CURRENT_PREPROCESSING),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    payload["rng_state"] = {
        "torch": _t.get_rng_state(),
        # Full NumPy Generator/legacy state (rev6: the old code saved
        # only 8 values and never restored them).
        "numpy_legacy": [__import__("numpy").random.get_state()[0],
                         __import__("numpy").random.get_state()[1].tolist(),
                         __import__("numpy").random.get_state()[2],
                         __import__("numpy").random.get_state()[3],
                         __import__("numpy").random.get_state()[4]],
        **(rng_state or {}),
    }
    _t.save(payload, str(path))
    path.with_suffix(".resolved.json").write_text(
        json.dumps({"checkpoint": str(path),
                    "config": config or {},
                    "config_hash": payload["config_hash"],
                    "manifest": manifest or {}}, indent=2, sort_keys=True,
                   default=str))
    return path


def load_checkpoint(path: str | Path, model, optimizer=None,  # type: ignore[no-untyped-def]
                    scheduler=None, strict: bool = True) -> dict:
    """Restore a save_checkpoint file; returns its manifest.

    Restores weights (+optimizer/scheduler/RNG when held). Raises on
    config-hash mismatch when the caller supplies an expected hash via
    model.expected_config_hash (never silently resume a foreign run).
    """
    if torch is None:
        raise ImportError("load_checkpoint requires torch")
    import torch as _t

    payload = _t.load(str(path), map_location="cpu", weights_only=False)
    try:
        compat = model.load_state_dict(payload["model_state"], strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            "refuse: checkpoint weights incompatible with current "
            f"model ({e}); check variant/base") from e
    if optimizer is not None and "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except ValueError as e:
            # rev6: never silently continue on a fresh optimizer while
            # reporting a resume. The caller decides (fresh start).
            raise RuntimeError(
                "refuse: optimizer state incompatible with current "
                f"model ({e}); restart without --init-checkpoint for a "
                "fresh optimizer") from e
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if "rng_state" in payload and "torch" in payload["rng_state"]:
        _t.set_rng_state(payload["rng_state"]["torch"])
    if "rng_state" in payload and "numpy_legacy" in payload["rng_state"]:
        import numpy as _np
        st = payload["rng_state"]["numpy_legacy"]
        _np.random.set_state((st[0], _np.asarray(st[1], dtype="uint32"),
                              st[2], st[3], st[4]))
    expected = getattr(model, "expected_config_hash", None)
    if expected and payload.get("config_hash", "") != expected:
        raise RuntimeError(
            "refuse: checkpoint config hash "
            f"{payload.get('config_hash', '')!r} != expected "
            f"{expected!r} (foreign run?)")
    return {"config": payload.get("config", {}),
            "config_hash": payload.get("config_hash", ""),
            "manifest": payload.get("manifest", {}),
            "torch_version": payload.get("torch_version", ""),
            # rev7: expose saved RNG so callers can restore it (a
            # resume that re-seeds from scratch is not a resume).
            "rng_state": payload.get("rng_state", {}),
            "missing_keys": list(getattr(compat, "missing_keys", [])),
            "unexpected_keys": list(getattr(compat, "unexpected_keys", []))}
