"""rev11 acceptance probe: unreviewed-band pixels carry ZERO gradient.

The rev11 audit measured the gradient of the FULL configured body
objective (dice_selectors + balanced + band regions) against synthetic
logits and found, across four current masks, 908 band pixels OUTSIDE the
reviewed scope with nonzero Dice gradient:

    mask-rev8p-001   403 px     mask-rev8m2-001  122 px
    mask-rev10a-000   34 px     mask-rev10a-002  349 px

The change-list rule: the geometric band may select REVIEWED background;
it does not independently certify negative truth — under Dice as well as
BCE. This probe rebuilds the same masks with the same crops the training
run froze (rev10_front_snap25/samples.json), applies the shared builder
(finalize_body_supervision + body_mask_tensors), and measures the
gradient of the full objective at every pixel.

Pass criteria, per mask:
  * unreviewed-band gradient-nonzero count == 0   (new)
  * licensed-band gradient-nonzero count  > 0     (band supervision
    still flows where the reviewer actually reviewed)
  * self-positive gradient-nonzero count  > 0     (paint still trains)

Usage:
  .venv/bin/python scripts/rev11_gradient_probe.py \
      [--snapshot runs/prototypes/v30/snapshots/snap25] \
      [--out runs/prototypes/v30/rev11_gradient_probe.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex import train as T  # noqa: E402
from prototypes.v30_video_apex.batch_builder import (  # noqa: E402
    body_mask_tensors, finalize_body_supervision,
    own_mask_target)
from prototypes.v30_video_apex.targets import (  # noqa: E402
    confusable_union, paint_overlap_union, samples_from_snapshot)

CS = 288

# The four masks the audit measured, with the crops frozen in
# rev10_front_snap25/samples.json (entry_id -> crop_xywh origin).
TARGETS = (
    ("mask-rev8p-001", 42000, (490, 498), 403),
    ("mask-rev8m2-001", 52500, (933, 728), 122),
    ("mask-rev10a-000", 28770, (931, 0), 34),
    ("mask-rev10a-002", 26460, (931, 0), 349),
)


def _configure_objective() -> None:
    """Exactly the configuration the audited runs used."""
    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("band")
    T.set_body_bg_region("band")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap25"))
    ap.add_argument("--out", default=str(
        REPO / "runs/prototypes/v30/rev11_gradient_probe.json"))
    a = ap.parse_args()

    _configure_objective()
    weights = T.LossWeights(apex=0.0, front=0.0, visibility=0.0,
                            body=1.0)
    rows = samples_from_snapshot(str(a.snapshot))
    report = {"snapshot": str(a.snapshot), "masks": [], "pass": True}
    for uuid, frame, (ox, oy), audit_px in TARGETS:
        s = next((x for x in rows
                  if x.kind == "body_mask" and x.mask_raster
                  and str(x.mask_uuid) == uuid
                  and int(x.source_frame or -1) == frame), None)
        if s is None:
            report["masks"].append({"mask": uuid, "skipped": "not found"})
            continue
        pool = [{"mask_uuid": x.mask_uuid, "movie": x.movie,
                 "source_frame": x.source_frame, "owner_key": x.owner_key,
                 "mask_raster": x.mask_raster} for x in rows
                if x.kind == "body_mask" and x.movie == s.movie]
        eligible = {str(x.mask_uuid) for x in rows
                    if x.kind == "body_mask" and x.mask_uuid
                    and not x.quarantine_reason}
        bt = own_mask_target(s, CS, CS, (ox, oy))
        fo = confusable_union(CS, CS, (ox, oy), movie=s.movie,
                              frame=int(s.source_frame),
                              self_uuid=s.mask_uuid,
                              self_owner_key=s.owner_key, masks=pool,
                              eligible=eligible)
        ov = paint_overlap_union(CS, CS, (ox, oy), movie=s.movie,
                                 frame=int(s.source_frame),
                                 masks=pool, eligible=eligible)
        bt = finalize_body_supervision(bt, confusable=fo, overlap=ov)
        ch = body_mask_tensors(bt, confusable=fo, overlap=ov)

        band = ch["body_band"] > 0
        lic = ch["body_bg_reviewed"] > 0
        fg = np.asarray(bt.target) > 0
        unreviewed_band = band & ~lic & ~fg
        licensed_band = band & lic & ~fg

        leaf = torch.zeros(1, 1, CS, CS, requires_grad=True)
        masks_t = {k: torch.from_numpy(v)[None, None]
                   for k, v in ch.items()}
        out = T.masked_multihead_loss(
            {"body": leaf},
            {"body_mask": torch.from_numpy(
                np.asarray(bt.target, dtype=np.float32))[None, None]},
            masks_t, weights)
        out["total"].backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().numpy()[0, 0]
        g_nz = np.abs(grad) > 0

        n_unrev = int(unreviewed_band.sum())
        n_unrev_grad = int((g_nz & unreviewed_band).sum())
        n_lic = int(licensed_band.sum())
        n_lic_grad = int((g_nz & licensed_band).sum())
        n_fg_grad = int((g_nz & fg).sum())
        row = {"mask": uuid, "frame": frame, "crop": [ox, oy],
               "unreviewed_band_px": n_unrev,
               "unreviewed_band_nonzero_gradient_px": n_unrev_grad,
               "audit_unreviewed_band_px": audit_px,
               "licensed_band_px": n_lic,
               "licensed_band_nonzero_gradient_px": n_lic_grad,
               "self_positive_nonzero_gradient_px": n_fg_grad,
               "ok": (n_unrev_grad == 0 and
                      (n_lic == 0 or n_lic_grad > 0) and n_fg_grad > 0)}
        report["masks"].append(row)
        report["pass"] = report["pass"] and row["ok"]
        print(f"{uuid:>16} f{frame}: unreviewed band {n_unrev} px "
              f"(audit {audit_px}) -> grad {n_unrev_grad} "
              f"| licensed band {n_lic} -> grad {n_lic_grad} "
              f"| self grad {n_fg_grad} | OK {row['ok']}")

    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"pass: {report['pass']} -> {a.out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
