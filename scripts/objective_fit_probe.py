"""Two-minute objective comparison on ONE real mask (frozen trunk).

The WP-B run's loss was flat to four decimals by epoch 17, so before
spending hours: freeze the trunk, optimise only the body head, and ask
which objective can actually move it on a real crop — with the real
selectors (self / reviewed / foreign) and the same step budget.

Same discipline as `head_fit_probe.py`: answer rate/loss questions here,
not in an hour-long run.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.dataset import QUERY_OFFSETS  # noqa: E402
from prototypes.v30_video_apex.model import build_model  # noqa: E402
from prototypes.v30_video_apex.targets import (  # noqa: E402
    body_mask_channels_from_raster, confusable_union, load_clip_pixels,
    samples_from_snapshot)
from prototypes.v30_video_apex.train import (  # noqa: E402
    LossWeights, masked_multihead_loss, set_body_dice_balanced,
    set_body_objective)
from tubetracker.annotation_frames import FrameReader  # noqa: E402

CS = 288


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap23"))
    ap.add_argument("--ref", default="mask-rev8p-000")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--head-lr", type=float, default=0.5)
    ap.add_argument("--dice-balanced", action="store_true")
    a = ap.parse_args()

    samples = samples_from_snapshot(a.snapshot)
    masks = [s for s in samples if s.kind == "body_mask" and s.mask_raster]
    mine = next((s for s in masks if a.ref in (s.obs_uuid or "")
                 or a.ref in (s.entry_id or "")), None)
    if mine is None:
        print(f"no mask sample matching {a.ref}")
        return 1
    man = __import__("json").loads(
        (Path(a.snapshot) / "snapshot_manifest.json").read_text())
    reader = FrameReader({k: v.get("path")
                          for k, v in man["movies"].items()}[mine.movie])
    r = mine.mask_raster
    ox = int(float(r["x0"]) + float(r["w"]) / 2.0 - CS / 2)
    oy = int(float(r["y0"]) + float(r["h"]) / 2.0 - CS / 2)
    crop = (ox, oy, CS, CS)
    frames = tuple(int(mine.source_frame) + o for o in QUERY_OFFSETS)
    missing = tuple(1 if f < 0 else 0 for f in frames)
    frames = tuple(max(0, f) for f in frames)
    clip_np = load_clip_pixels(reader, frames, crop, missing)
    reader.close()
    clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)

    ch = body_mask_channels_from_raster(CS, CS, (ox, oy), r,
                                        complete=bool(mine.complete),
                                        review_region=mine.review_region)
    tgt = torch.from_numpy(ch["target"]).unsqueeze(0).unsqueeze(0)
    valid = torch.from_numpy(ch["valid"]).unsqueeze(0).unsqueeze(0)
    eligible = {str(s.mask_uuid) for s in masks
                if s.mask_uuid and not s.quarantine_reason}
    un = confusable_union(CS, CS, (ox, oy), movie=mine.movie,
                          frame=int(mine.source_frame),
                          self_uuid=mine.mask_uuid,
                          self_owner_key=mine.owner_key,
                          masks=[{"mask_uuid": s.mask_uuid,
                                  "movie": s.movie,
                                  "source_frame": s.source_frame,
                                  "owner_key": s.owner_key,
                                  "mask_raster": s.mask_raster}
                                 for s in masks],
                          eligible=eligible)
    conf = torch.from_numpy(un.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    print(f"mask {mine.obs_uuid} frame {mine.source_frame} crop {crop}")
    print(f"target {int(ch['target'].sum())} px | valid {int(ch['valid'].sum())} px"
          f" | confusable {int(un.sum())} px | extent "
          f"{ch['quarantine_reason']}")

    from prototypes.v30_video_apex.model import build_owner_prompt
    # neutral prompt, identical in both arms: this probe isolates the
    # OBJECTIVE's gradient, not the prompt's informativeness
    prompt = build_owner_prompt("probe", provenance="none")

    torch.manual_seed(0)
    base = build_model("temporal", base=8, multiscale=True)
    base_state = copy.deepcopy(base.state_dict())

    for objective in ("dice_pixel", "dice_selectors",
                      "dice_selectors2"):
        model = build_model("temporal", base=8, multiscale=True)
        model.load_state_dict(base_state)
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("body."))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.head_lr)
        set_body_objective(objective)
        set_body_dice_balanced(bool(a.dice_balanced))
        model.train()
        first = None
        out = None
        for i in range(a.steps):
            opt.zero_grad()
            pred = model.forward(clip, prompt)
            out = masked_multihead_loss(
                {"body": pred.body}, {"body_mask": tgt},
                {"body_valid": valid, "body_confusable": conf},
                LossWeights(body=1.0))
            out["body"].backward()
            opt.step()
            if i == 0:
                first = float(out["body"].detach())
        with torch.no_grad():
            logits = model.forward(clip, prompt).body[0, 0]
            prob = torch.sigmoid(logits).numpy()
        pred_b = prob > 0.5
        t = ch["target"] > 0
        inter = float((pred_b & t).sum())
        union = float((pred_b | t).sum())
        crop_iou = inter / union if union else 0.0
        parts = {k: round(float(v.detach()), 4) for k, v in out.items()
                 if k.startswith("body")}
        print(f"{objective:15s} loss {first:.4f} -> "
              f"{float(out['body'].detach()):.4f} | logit std "
              f"{float(logits.std()):.4f} | crop-IoU {crop_iou:.4f} | "
              f"pred_frac {float(pred_b.mean()):.4f}")
        print(f"                parts: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
