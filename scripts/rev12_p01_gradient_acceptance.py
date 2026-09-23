"""rev12 P0.1 acceptance: licensed body-domain gradients on saved logits.

Reproduces the audit's counterfactual (audit_loss_scopes.py) on the
saved step-569 predictions of rev11_fitcheck_gate600, and then proves
the REPAIRED implementation on the same fixed logits:

- legacy path (no explicit selectors), published band objective:
  ring pixels get ZERO gradient (the audit's 954/40/1,098);
- legacy path, naive reviewed extent over the broad valid mask:
  unreviewed pixels get gradient (the audit's 1,373 for g1) — the
  trap the repair must avoid;
- legacy path, reviewed extent with a licensed valid mask:
  ring gradients restored, zero unknown (the audit's in-memory fix);
- REPAIRED path (explicit licensed selectors, bg region 'strata'):
  ring gradients on every ring pixel, nonzero own/foreign gradients,
  and exactly zero unknown gradients — without any manual masking.

No optimizer is used anywhere: logits are the saved probabilities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STEP = "569"
OUT = REPO / "runs/prototypes/v30/rev12_p01_gradient_acceptance.json"
NPZ = REPO / f"runs/prototypes/v30/rev11_fitcheck_gate600.step{STEP}.npz"


def published_objective(T) -> None:
    """Exactly the configuration the audited runs used."""
    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("band")
    T.set_body_bg_region("band")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)


def grad_counts(T, o: dict, prob: np.ndarray, masks: dict) -> dict:
    p = np.clip(prob, 1e-5, 1 - 1e-5)
    logits = torch.tensor(np.log(p / (1 - p))[None, None],
                          requires_grad=True)
    out = T.masked_multihead_loss(
        {"body": logits},
        {"body_mask": torch.from_numpy(o["target"])[None, None]},
        masks, T.LossWeights(body=1.0))
    out["total"].backward()
    grad = logits.grad.numpy()[0, 0]
    return grad, float(out["total"].detach())


def main() -> int:
    from scripts.rev10_fit_check import _scene
    from prototypes.v30_video_apex import train as T

    torch.set_num_threads(2)
    scene = _scene(REPO / "runs/prototypes/v30/snapshots/snap24")
    saved = np.load(NPZ, allow_pickle=True)
    report = {"step": STEP, "npz": str(NPZ), "rows": []}
    for i, o in enumerate(scene):
        ch = o["channels"]
        sel_self = ch["body_sel_self"] > 0
        sel_bg = ch["body_sel_bg"] > 0
        sel_fo = ch["body_sel_foreign"] > 0
        band = o["band"] > 0
        licensed = sel_self | sel_bg | sel_fo
        unknown = ~licensed
        ring = (saved[f"prob_{i}"] > 0.5) & sel_bg & ~band
        modes = [
            # (name, selectors_kept, bg_region, dice_region, valid_mask)
            ("legacy_published_band", False, "band", "band", None),
            ("legacy_naive_extent", False, "reviewed", "extent", None),
            ("legacy_licensed_extent", False, "reviewed", "extent",
             licensed),
            ("repaired_strata", True, "strata", "extent", None),
            ("repaired_reviewed", True, "reviewed", "extent", None),
            ("repaired_band_only", True, "band", "band", None),
        ]
        for name, keep_sel, bg, dice, vmask in modes:
            published_objective(T)
            T.set_body_bg_region(bg)
            T.set_body_dice_region(dice)
            m2 = {k: np.asarray(v, dtype=np.float32).copy()
                  for k, v in ch.items()}
            if not keep_sel:
                for k in ("body_sel_self", "body_sel_bg",
                          "body_sel_foreign", "body_sel_unknown"):
                    m2.pop(k, None)
            if vmask is not None:
                m2["body_valid"] = m2["body_valid"] * vmask.astype(
                    np.float32)
            masks = {k: torch.from_numpy(v)[None, None]
                     for k, v in m2.items()}
            grad, loss = grad_counts(T, o, saved[f"prob_{i}"], masks)
            nz = np.abs(grad) > 0
            report["rows"].append({
                "owner": o["owner"], "mode": name,
                "ring_positive_pixels": int(ring.sum()),
                "ring_nonzero_gradient_pixels": int((ring & nz).sum()),
                "ring_suppressive_gradient_pixels": int(
                    (ring & (grad > 0)).sum()),
                "unknown_nonzero_gradient_pixels": int(
                    (unknown & nz).sum()),
                "own_nonzero_gradient_pixels": int((sel_self & nz).sum()),
                "foreign_nonzero_gradient_pixels": int(
                    (sel_fo & nz).sum()),
                "loss": round(loss, 8),
            })
            r = report["rows"][-1]
            print(f"{o['owner']:>16} {name:>24}: ring "
                  f"{r['ring_nonzero_gradient_pixels']}"
                  f"/{r['ring_positive_pixels']} | unknown-grad "
                  f"{r['unknown_nonzero_gradient_pixels']} | own "
                  f"{r['own_nonzero_gradient_pixels']} | foreign "
                  f"{r['foreign_nonzero_gradient_pixels']}")
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
