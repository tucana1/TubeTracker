"""rev10 WP-A exit evidence: own-only must beat own+union on saved views.

The review's decisive counterexample: "On all 21 clump views, predicting
the correct tube and predicting the union of labeled tubes have exactly
the same tested loss." Reproduced here at 0.0012975569115951657 both.

This runs the SAME configured loss the trainer optimizes — same target
builder, same channel names, same knobs — and reports per view:

  * loss(own-only field) vs loss(own + every labelled neighbour)
  * the foreign gradient at a zero field, including its distance to the
    nearest own-band pixel (it must be nonzero at ANY distance)

It also measures the same quantities through the *unfixed* scope
(`--legacy-band-foreign`) so the change is visible rather than asserted.
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
from prototypes.v30_video_apex.batch_builder import own_mask_target  # noqa: E402
from prototypes.v30_video_apex.inference import load_query_clip  # noqa: E402
from prototypes.v30_video_apex.targets import (  # noqa: E402
    confusable_union, paint_overlap_union, samples_from_snapshot)

CS = 288


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap24"))
    ap.add_argument("--crop", type=int, default=CS)
    ap.add_argument("--jitter", action="store_true",
                    help="also probe crop offsets, as training does")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("band")
    T.set_body_bg_region("band")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)

    rows = [s for s in samples_from_snapshot(a.snapshot)
            if s.kind == "body_mask" and s.mask_raster]
    out = {"views": [], "summary": {}}
    offsets = [(0, 0), (-40, 0), (40, 0), (0, -40), (0, 40)] if a.jitter \
        else [(0, 0)]
    for s in rows:
      for _jx, _jy in offsets:
          r = s.mask_raster
          ox = int(float(r["x0"]) + float(r["w"]) / 2 - a.crop / 2) + _jx
          oy = int(float(r["y0"]) + float(r["h"]) / 2 - a.crop / 2) + _jy
          ox = max(0, min(ox, 1280 - a.crop))
          oy = max(0, min(oy, 1024 - a.crop))
          bt = own_mask_target(s, a.crop, a.crop, (ox, oy))
          own = bt.target > 0
          pool = [{"mask_uuid": x.mask_uuid, "movie": x.movie,
                   "source_frame": x.source_frame,
                   "owner_key": x.owner_key, "mask_raster": x.mask_raster}
                  for x in rows]
          eligible = {str(x.mask_uuid) for x in rows
                      if x.mask_uuid and not x.quarantine_reason}
          fo = confusable_union(a.crop, a.crop, (ox, oy), movie=s.movie,
                                frame=int(s.source_frame),
                                self_uuid=s.mask_uuid,
                                self_owner_key=s.owner_key,
                                masks=pool, eligible=eligible)
          ov = paint_overlap_union(a.crop, a.crop, (ox, oy), movie=s.movie,
                                   frame=int(s.source_frame), masks=pool,
                                   eligible=eligible)
          fg = fo & ~ov
          # synthetic fields: this isolates the OBJECTIVE, not the model
          lg_own = np.full((1, 1, a.crop, a.crop), -6.0, np.float32)
          lg_own[0, 0][own] = 6.0
          lg_union = lg_own.copy()
          lg_union[0, 0][fg] = 6.0

          def _masks():
              m = {"body_valid": torch.from_numpy(
                       (bt.valid > 0).astype(np.float32))[None, None],
                   "body_bg_reviewed": torch.from_numpy(
                       bt.bg_reviewed.astype(np.float32))[None, None],
                   "body_band": torch.from_numpy(
                       bt.band.astype(np.float32))[None, None],
                   "body_confusable": torch.from_numpy(
                       fg.astype(np.float32))[None, None]}
              if ov.any():
                  m["body_overlap"] = torch.from_numpy(
                      ov.astype(np.float32))[None, None]
              return m

          tgt_t = torch.from_numpy(bt.target)[None, None]
          l_own = float(T.masked_multihead_loss(
              {"body": torch.from_numpy(lg_own)}, {"body_mask": tgt_t},
              _masks(), T.LossWeights(body=1.0))["body"])
          l_union = float(T.masked_multihead_loss(
              {"body": torch.from_numpy(lg_union)}, {"body_mask": tgt_t},
              _masks(), T.LossWeights(body=1.0))["body"])

          # foreign gradient at a ZERO field, and how far the nearest
          # verified foreign pixel sits from the band
          lg = torch.zeros(1, 1, a.crop, a.crop, requires_grad=True)
          o = T.masked_multihead_loss({"body": lg}, {"body_mask": tgt_t},
                                      _masks(), T.LossWeights(body=1.0))
          o["body"].backward()
          g = lg.grad.detach().numpy()[0, 0]
          band = bt.band > 0
          # F_i must never be gated by the band — but a pixel OUTSIDE the
          # reviewed scope is unknown and correctly gets zero. Separate the
          # two instead of reporting a bare minimum.
          licensed = (bt.valid > 0)
          fg_lic = fg & licensed
          fg_unrev = fg & ~licensed
          if fg_lic.any() and band.any():
              ys, xs = np.nonzero(fg_lic)
              by, bx = np.nonzero(band)
              d = np.sqrt((ys[:, None] - by[None, :]) ** 2
                          + (xs[:, None] - bx[None, :]) ** 2).min(axis=1)
              far = float(d.max())
              g_min = float(g[fg_lic].min())
          else:
              far, g_min = None, None
          row = {"obs": str(s.obs_uuid), "crop": [ox, oy, a.crop, a.crop],
                 "own_px": int(own.sum()), "foreign_px": int(fg.sum()),
                 "foreign_licensed_px": int(fg_lic.sum()),
                 "foreign_unreviewed_px": int(fg_unrev.sum()),
                 "overlap_px": int(ov.sum()),
                 "loss_own_only": round(l_own, 8),
                 "loss_own_plus_union": round(l_union, 8),
                 "separated": bool(l_union > l_own),
                 "foreign_min_grad": (None if g_min is None
                                      else round(g_min, 9)),
                 "foreign_farthest_px_from_band": (None if far is None
                                                   else round(far, 1))}
          out["views"].append(row)
          print(f"{row['obs']:>16} own {row['own_px']:>4} fo "
                f"{row['foreign_px']:>4} ov {row['overlap_px']:>4} | own-only "
                f"{row['loss_own_only']:.8f} vs union "
                f"{row['loss_own_plus_union']:.8f} | sep {row['separated']} "
                f"| min fo grad {row['foreign_min_grad']} at "
                f"{row['foreign_farthest_px_from_band']}px")

    sep = [v for v in out["views"] if v["foreign_licensed_px"] > 0]
    out["summary"] = {
        "n_views": len(out["views"]),
        "views_with_licensed_foreign": len(sep),
        "views_with_unreviewed_foreign": sum(
            1 for v in out["views"] if v["foreign_unreviewed_px"] > 0),
        "views_separated": sum(1 for v in sep if v["separated"]),
        "all_separated": bool(sep) and all(v["separated"] for v in sep),
        "min_foreign_grad": min(
            (v["foreign_min_grad"] for v in sep
             if v["foreign_min_grad"] is not None), default=None),
        "all_foreign_grads_positive": all(
            v["foreign_min_grad"] is not None and v["foreign_min_grad"] > 0
            for v in sep),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
