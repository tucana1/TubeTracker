"""rev12 P0.1: body-fire diagnostic by reviewed extent and class scope.

The gate600 trajectory: recall on reviewed body is high for all three
owners, but many positive pixels fall outside the reviewed structures.
The old diagnostic defined "unreviewed" as anything outside own/other
tube paint — which ignored the human-reviewed background extent and
therefore mislabelled 954/40/1,098 actually-reviewed pixels.

This version classifies every fire (prob > 0.5 at the saved step) by
the EXPLICIT licensed selectors and reports, separately:
- `unknown_fire`        — fires outside every licensed selector;
- `ignored_reviewed_bg_fire` — fires on licensed reviewed background
  that receive ZERO loss gradient (the P0.1 defect; zero under the
  repaired 'strata' configuration);
- `foreign_confusion`   — fires on verified foreign-exclusive pixels.
Each class carries its gradient status from the repaired objective.
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
NPZ = REPO / f"runs/prototypes/v30/rev11_fitcheck_gate600.step{STEP}.npz"
OUT = REPO / "runs/prototypes/v30/rev12_p01_fire_diagnostic.json"


def main() -> int:
    from scripts.rev10_fit_check import _scene
    from prototypes.v30_video_apex import train as T

    torch.set_num_threads(2)
    scene = _scene(REPO / "runs/prototypes/v30/snapshots/snap24")
    saved = np.load(NPZ, allow_pickle=True)
    # the repaired configuration: licensed selectors, balanced strata
    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("extent")
    T.set_body_bg_region("strata")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)

    rows = []
    for i, o in enumerate(scene):
        ch = o["channels"]
        sel_self = ch["body_sel_self"] > 0
        sel_bg = ch["body_sel_bg"] > 0
        sel_fo = ch["body_sel_foreign"] > 0
        sel_unk = ch["body_sel_unknown"] > 0
        band = o["band"] > 0
        prob = saved[f"prob_{i}"]
        fire = prob > 0.5

        p = np.clip(prob, 1e-5, 1 - 1e-5)
        logits = torch.tensor(np.log(p / (1 - p))[None, None],
                              requires_grad=True)
        masks = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))[
            None, None] for k, v in ch.items()}
        out = T.masked_multihead_loss(
            {"body": logits},
            {"body_mask": torch.from_numpy(o["target"])[None, None]},
            masks, T.LossWeights(body=1.0))
        out["total"].backward()
        grad = logits.grad.numpy()[0, 0]
        nz = np.abs(grad) > 0

        def _c(sel):
            f = fire & sel
            return {"fires": int(f.sum()),
                    "with_gradient": int((f & nz).sum()),
                    "zero_gradient": int((f & ~nz).sum())}

        row = {
            "owner": o["owner"],
            "fires_total": int(fire.sum()),
            "self": _c(sel_self),
            "reviewed_bg_band": _c(sel_bg & band),
            "reviewed_bg_far": _c(sel_bg & ~band),
            "foreign_confusion": _c(sel_fo),
            "unknown_fire": _c(sel_unk),
        }
        row["ignored_reviewed_bg_fire"] = (
            row["reviewed_bg_band"]["zero_gradient"]
            + row["reviewed_bg_far"]["zero_gradient"])
        rows.append(row)
        print(f"{o['owner']}: fires {row['fires_total']} | self "
              f"{row['self']['fires']} | reviewed-bg band "
              f"{row['reviewed_bg_band']['fires']} (zero-grad "
              f"{row['reviewed_bg_band']['zero_gradient']}) | far "
              f"{row['reviewed_bg_far']['fires']} (zero-grad "
              f"{row['reviewed_bg_far']['zero_gradient']}) | foreign "
              f"{row['foreign_confusion']['fires']} (zero-grad "
              f"{row['foreign_confusion']['zero_gradient']}) | unknown "
              f"{row['unknown_fire']['fires']} (zero-grad "
              f"{row['unknown_fire']['zero_gradient']})")

    res = {
        "step": STEP,
        "objective": T.body_loss_config(),
        "rows": rows,
        "totals": {
            "ignored_reviewed_bg_fire": sum(
                r["ignored_reviewed_bg_fire"] for r in rows),
            "unknown_fire": sum(r["unknown_fire"]["fires"] for r in rows),
            "foreign_confusion": sum(
                r["foreign_confusion"]["fires"] for r in rows),
        },
        "note": ("fires classified by the explicit licensed selectors; "
                 "unknown fire is reported separately from ignored "
                 "reviewed-background fire and foreign confusion"),
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
