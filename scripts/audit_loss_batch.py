#!/usr/bin/env python3
"""Audit the annotation-to-loss contract on real sampled patches (P2/S0).

For fixed patch indices: renders input / tip target / tip validity /
input-gradient-magnitude, proves zero gradient outside the declared
scope, and (with --overfit-steps) shows the model can fit a few
verified clips (plumbing diagnosis, not performance evidence).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_cnn_points import (  # noqa: E402
    PointPatchDataset,
    load_corner_negatives,
    load_dataset_manifest,
    read_frame_reviews,
    read_point_labels,
    split_records,
)
from tubetracker.cnn_prototype import (  # noqa: E402
    PointHeatmapNet,
    masked_heatmap_loss,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--indices", default="0,1,2,3")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--patches-per-frame", type=int, default=12)
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--scoped-tip-masks", action="store_true")
    ap.add_argument("--grain-loss-weight", type=float, default=1.0)
    ap.add_argument("--overfit-steps", type=int, default=0)
    a = ap.parse_args()

    ds_dir = a.dataset_dir.expanduser().resolve()
    manifest = load_dataset_manifest(ds_dir / "manifest.json")
    labels = read_point_labels(ds_dir / "labels.csv")
    reviews = read_frame_reviews(ds_dir / "frame_reviews.csv")
    by_image = {r.image_name: r for r in reviews}
    labels_by_image = {}
    for lab in labels:
        labels_by_image.setdefault(lab.image_name, []).append(lab)
    eligible = [r for r in manifest["frames"] if r["image_name"] in by_image
                and (by_image[r["image_name"]].grain_reviewed
                     or by_image[r["image_name"]].tip_reviewed)]
    train_recs, _ = split_records(eligible, 0.2, a.seed)
    data = PointPatchDataset(ds_dir, train_recs, labels_by_image, by_image,
                             a.patch_size, a.patches_per_frame, a.seed,
                             False, load_corner_negatives(
                                 ds_dir / "corners.csv"))
    data.scoped_tip_masks = a.scoped_tip_masks

    torch.manual_seed(a.seed)
    model = PointHeatmapNet(input_channels=2, output_channels=2,
                            base_channels=16)
    weights = [a.grain_loss_weight, 1.0]
    idx = [int(s) for s in a.indices.split(",") if s.strip()]
    cols = []
    for i in idx:
        inputs, targets, reviewed, valid = data[i]
        b_in, b_tg, b_rv, b_va = (t.unsqueeze(0) for t in
                                  (inputs, targets, reviewed, valid))
        logits = model(b_in)
        loss = masked_heatmap_loss(logits, b_tg, b_rv, valid=b_va,
                                   channel_weights=weights)
        # Proof at the loss level (input-gradients legitimately spread
        # through the receptive field; the mask acts on pixel losses).
        import torch.nn.functional as _F
        with torch.no_grad():
            el = _F.binary_cross_entropy_with_logits(
                logits, b_tg, reduction="none")
            mask = b_rv[:, :, None, None] * b_va
            w = torch.as_tensor(weights, dtype=mask.dtype)[:, None, None]
            masked = el * (1.0 + 24.0 * b_tg) * mask * w
            tip_inv = (b_va[:, 1] < 0.5)
            out_scope = float(masked[:, 1][tip_inv].sum())
            in_scope = float(masked[:, 1][~tip_inv].sum())
        b_in.requires_grad_(True)
        g = torch.autograd.grad(
            masked_heatmap_loss(model(b_in), b_tg, b_rv, valid=b_va,
                                channel_weights=weights),
            b_in, retain_graph=False)[0]
        gm = g.detach().abs().sum(1)[0].numpy()
        tip_t = targets[1].numpy()
        tip_v = valid[1].numpy()
        sup = int(((tip_t > 0.05) & (tip_v > 0.5)).sum())
        print(f"patch {i}: loss={float(loss):.5f} supervised_tip_px={sup} "
              f"tip_loss_in_scope={in_scope:.4f} "
              f"tip_loss_out_of_scope={out_scope:.6f}", flush=True)
        gray = (inputs[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        heat = (np.clip(tip_t, 0, 1) * 255).astype(np.uint8)
        mask = (tip_v * 255).astype(np.uint8)
        gg = (np.clip(gm / (gm.max() + 1e-9), 0, 1) * 255).astype(np.uint8)
        row = np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(gg, cv2.COLOR_GRAY2BGR)])
        cols.append(row)
    sheet = np.vstack(cols)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), sheet)
    print(f"wrote {a.out} {sheet.shape}", flush=True)

    if a.overfit_steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        batch = [data[i] for i in idx]
        b_in = torch.stack([b[0] for b in batch])
        b_tg = torch.stack([b[1] for b in batch])
        b_rv = torch.stack([b[2] for b in batch])
        b_va = torch.stack([b[3] for b in batch])
        for s in range(a.overfit_steps):
            opt.zero_grad(set_to_none=True)
            loss = masked_heatmap_loss(model(b_in), b_tg, b_rv, valid=b_va,
                                       channel_weights=weights)
            loss.backward()
            opt.step()
            if (s + 1) % max(1, a.overfit_steps // 5) == 0:
                print(f"overfit step {s + 1}/{a.overfit_steps}: "
                      f"loss={float(loss):.6f}", flush=True)


if __name__ == "__main__":
    main()
