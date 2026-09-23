#!/usr/bin/env python3
"""Train the click-supervised pollen-center and tube-tip heatmap model."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tubetracker.cnn_prototype import (  # noqa: E402
    CNN_LABELS,
    PointHeatmapNet,
    frame_to_model_input,
    load_dataset_manifest,
    masked_heatmap_loss,
    point_heatmaps,
    preferred_torch_device,
    read_frame_reviews,
    read_point_labels,
    save_cnn_checkpoint,
)


def parse_args():
    """Parse annotation data, optimization, and checkpoint settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-frame", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--corners-csv", type=Path, default=None,
                        help="H181: hard-negative corner CSV "
                        "(image_name,x,y). Defaults to corners.csv inside "
                        "the dataset dir when present.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="H161: warm-start weights from a previous best "
                        "checkpoint so SIGTERM-killed runs chain without losing "
                        "progress (optimizer restarts; same seed keeps data order).")
    parser.add_argument("--faint-contrast-csv", type=Path, default=None,
                        help="H207: label-contrast CSV (image_name,x,y,"
                        "contrast); tip labels with contrast below "
                        "--faint-contrast-below train with wide "
                        "--faint-sigma Gaussians (diffuse faint response).")
    parser.add_argument("--faint-contrast-below", type=float, default=8.0)
    parser.add_argument("--faint-sigma", type=float, default=4.5)
    parser.add_argument("--validity-manifest", type=Path, default=None,
                        help="H227: weak-tip manifest JSON "
                        "{image_name: {weak_xy: [[x, y]]}}; weak tips train "
                        "as ignore-discs (zero gradient) instead of "
                        "background. Missing file -> feature off.")
    parser.add_argument("--no-faint-override", action="store_true",
                        help="S1: never apply wide-sigma faint overrides, "
                        "even when label_contrast.csv is present. A null "
                        "--faint-contrast-csv is NOT evidence the feature "
                        "was off (it auto-loads); this flag is.")
    parser.add_argument("--no-corners", action="store_true",
                        help="S1: never sample inherited corner negatives, "
                        "even when corners.csv is present. Same null-means-"
                        "auto-load trap as faint overrides.")
    parser.add_argument("--no-corner-sampling", action="store_true",
                        help="S2: load corner positions (verified-negative "
                        "discs under --scoped-tip-masks) but never center "
                        "patches on them. Separates verified negatives "
                        "from the inherited 15% patch sampler.")
    parser.add_argument("--grain-loss-weight", type=float, default=1.0,
                        help="S1: multiply the grain-head loss (0.0 isolates "
                        "tip-only diagnosis; 1.0 preserves legacy weighting).")
    parser.add_argument("--scoped-tip-masks", action="store_true",
                        help="S1: unknown-by-default tip validity — only "
                        "discs around reviewed tip positives and verified "
                        "corner negatives carry gradient; all other tip "
                        "pixels are ignored instead of background.")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="S0: JSON {train:[...], validation:[...]} frame "
                        "lists. If the file exists it is loaded and checked "
                        "against eligible records (frozen splits); if not, "
                        "the computed split is saved there.")
    return parser.parse_args()


class PointPatchDataset(Dataset):
    """Sample positive-biased image patches with independently masked channels."""

    def __init__(
        self,
        dataset_dir,
        records,
        labels_by_image,
        reviews_by_image,
        patch_size,
        patches_per_frame,
        seed,
        augment,
        corners_by_image=None,
    ):
        self.dataset_dir = dataset_dir
        self.records = records
        self.labels_by_image = labels_by_image
        self.reviews_by_image = reviews_by_image
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.seed = seed
        self.augment = augment
        self.corners_by_image = corners_by_image or {}
        self.corner_sampling: bool = True
        self.faint_keys: set = set()
        self.faint_sigma: float = 4.5
        self.weak_by_image: dict = {}
        self.neg_by_image: dict = {}
        self.scoped_tip_masks: bool = False
        self.image_cache = {}

    def __len__(self):
        """Return the number of independently sampled training patches."""
        return len(self.records) * self.patches_per_frame

    def load_image(self, image_name):
        """Load and cache one exported annotation frame."""
        if image_name not in self.image_cache:
            frame = cv.imread(str(self.dataset_dir / "frames" / image_name))
            if frame is None:
                raise RuntimeError(f"Could not load annotation image: {image_name}")
            self.image_cache[image_name] = frame
        return self.image_cache[image_name]

    def __getitem__(self, index):
        """Return one model patch, heatmap target, and channel-review mask."""
        record = self.records[index // self.patches_per_frame]
        rng = np.random.default_rng(self.seed + index)
        frame = self.load_image(record["image_name"])
        height, width = frame.shape[:2]
        patch_height = min(self.patch_size, height)
        patch_width = min(self.patch_size, width)
        labels = self.labels_by_image.get(record["image_name"], [])
        review = self.reviews_by_image[record["image_name"]]
        reviewed_types = {
            "grain": review.grain_reviewed,
            "tip": review.tip_reviewed,
        }
        positive_labels = [
            label for label in labels if reviewed_types[label.label_type]
        ]
        if positive_labels and rng.random() < 0.75:
            corners = self.corners_by_image.get(record["image_name"], [])
            if corners and self.corner_sampling and rng.random() < 0.15:
                cx, cy = corners[int(rng.integers(0, len(corners)))]
                left = int(round(cx - patch_width / 2 + rng.uniform(-32, 32)))
                top = int(round(cy - patch_height / 2 + rng.uniform(-32, 32)))
            else:
                anchor = positive_labels[int(rng.integers(0, len(positive_labels)))]
                left = int(round(anchor.x - patch_width / 2 + rng.uniform(-32, 32)))
                top = int(round(anchor.y - patch_height / 2 + rng.uniform(-32, 32)))
        else:
            left = int(rng.integers(0, max(1, width - patch_width + 1)))
            top = int(rng.integers(0, max(1, height - patch_height + 1)))
        left = int(np.clip(left, 0, max(0, width - patch_width)))
        top = int(np.clip(top, 0, max(0, height - patch_height)))
        patch = frame[top : top + patch_height, left : left + patch_width]
        local_labels = [
            type(label)(
                image_name=label.image_name,
                source_frame=label.source_frame,
                label_type=label.label_type,
                x=label.x - left,
                y=label.y - top,
                provenance=label.provenance,
                confidence=label.confidence,
            )
            for label in labels
            if left <= label.x < left + patch_width
            and top <= label.y < top + patch_height
        ]
        inputs = frame_to_model_input(patch)
        overrides = [
            (self.faint_sigma
             if (label.image_name, round(label.x), round(label.y))
             in self.faint_keys else None)
            for label in labels
            if left <= label.x < left + patch_width
            and top <= label.y < top + patch_height
        ]
        targets = point_heatmaps(patch_height, patch_width, local_labels,
                                 sigma_overrides=overrides or None)
        valid = None
        if self.weak_by_image:
            weak = self.weak_by_image.get(record["image_name"], [])
            in_patch = [(x - left, y - top) for x, y in weak
                        if left <= x < left + patch_width
                        and top <= y < top + patch_height]
            if in_patch:
                from tubetracker.validity_masks import build_validity
                _, ign = build_validity(
                    patch_height, patch_width, [], in_patch)
                tip_valid = (~ign).astype(np.float32)
                valid = np.stack(
                    [np.ones_like(tip_valid), tip_valid], axis=0)
        neg_boxes = self.neg_by_image.get(record["image_name"], [])
        if neg_boxes:
            from tubetracker.validity_masks import neg_region_mask
            box = neg_region_mask(
                patch_height, patch_width,
                [(x0 - left, y0 - top, x1 - left, y1 - top)
                 for x0, y0, x1, y1 in neg_boxes]).astype(np.float32)
            if valid is None:
                valid = np.stack(
                    [np.ones((patch_height, patch_width), dtype=np.float32),
                     np.ones((patch_height, patch_width), dtype=np.float32)],
                    axis=0)
            # Verified negatives are SUPERVISED background (target stays 0
            # where no label sits): union into the tip channel, never grain.
            valid[1] = np.maximum(valid[1], box)
        if self.scoped_tip_masks and valid is None:
            # S1 unknown-by-default: tip gradient lives ONLY on discs
            # around reviewed tip positives (r=8) and verified corner
            # negatives (r=8). Every other tip pixel is ignored, never
            # background. Grain channel keeps legacy ones (S1 zeroes its
            # loss weight instead).
            yy, xx = np.mgrid[0:patch_height, 0:patch_width]
            tip_valid = np.zeros((patch_height, patch_width),
                                 dtype=np.float32)
            for label in local_labels:
                if label.label_type != "tip":
                    continue
                if not reviewed_types.get("tip", False):
                    continue
                tip_valid[np.hypot(xx - label.x, yy - label.y) <= 8.0] = 1.0
            for cx, cy in self.corners_by_image.get(
                    record["image_name"], []):
                lx, ly = cx - left, cy - top
                if 0 <= lx < patch_width and 0 <= ly < patch_height:
                    tip_valid[np.hypot(xx - lx, yy - ly) <= 8.0] = 1.0
            valid = np.stack(
                [np.ones((patch_height, patch_width), dtype=np.float32),
                 tip_valid], axis=0)
        if self.augment and rng.random() < 0.5:
            inputs = inputs[:, :, ::-1].copy()
            targets = targets[:, :, ::-1].copy()
            if valid is not None:
                valid = valid[:, :, ::-1].copy()
        if self.augment and rng.random() < 0.5:
            inputs = inputs[:, ::-1, :].copy()
            targets = targets[:, ::-1, :].copy()
            if valid is not None:
                valid = valid[:, ::-1, :].copy()
        if self.augment:
            gain = float(rng.uniform(0.88, 1.12))
            offset = float(rng.uniform(-0.06, 0.06))
            inputs = np.clip(inputs * gain + offset, 0.0, 1.0)
        reviewed = np.array(
            [review.grain_reviewed, review.tip_reviewed], dtype=np.float32
        )
        valid_out = (torch.from_numpy(valid.astype(np.float32))
                     if valid is not None
                     else torch.ones_like(torch.from_numpy(targets)))
        return (
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(targets),
            torch.from_numpy(reviewed),
            valid_out,
        )


def load_faint_keys(path, below: float) -> set:
    """Load faint tip-label keys: {(image_name, round(x), round(y))}.

    H207: labels whose local image contrast falls below threshold train
    with wide-sigma Gaussians. Missing file -> set() (feature off);
    malformed rows skipped, never fatal.
    """
    import csv

    keys: set = set()
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    if float(row["contrast"]) < below:
                        keys.add((row["image_name"], round(float(row["x"])),
                                  round(float(row["y"]))))
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        return set()
    return keys


def load_corner_negatives(path) -> dict:
    """Load hard-negative corner points: {image_name: [(x, y), ...]}.

    H181: tube bends the apex detector fires on (P58's elbow). Missing
    file -> {} (negatives off); malformed rows are skipped, never fatal.
    """
    import csv

    corners: dict = {}
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    name = row["image_name"]
                    pt = (float(row["x"]), float(row["y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                corners.setdefault(name, []).append(pt)
    except OSError:
        pass
    return corners


def split_records(records, validation_fraction, seed):
    """Split complete frames so patches from one frame cannot leak across sets."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or validation_fraction <= 0:
        return shuffled, []
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def evaluate(model, loader, device, channel_weights=None):
    """Return mean masked heatmap loss for one data loader."""
    if loader is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets, reviewed, valid in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reviewed = reviewed.to(device)
            valid = valid.to(device)
            losses.append(
                float(masked_heatmap_loss(model(inputs), targets, reviewed,
                                          valid=valid,
                                          channel_weights=channel_weights).cpu())
            )
    return float(np.mean(losses)) if losses else None


def main():
    """Train a reproducible point-heatmap model and save the best checkpoint."""
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patch_size < 32:
        raise SystemExit("Epochs, batch size, and patch size must be positive")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_dataset_manifest(dataset_dir / "manifest.json")
    labels = read_point_labels(dataset_dir / "labels.csv")
    reviews = read_frame_reviews(dataset_dir / "frame_reviews.csv")
    reviews_by_image = {review.image_name: review for review in reviews}
    labels_by_image = {}
    for label in labels:
        labels_by_image.setdefault(label.image_name, []).append(label)
    eligible_records = [
        record
        for record in manifest["frames"]
        if record["image_name"] in reviews_by_image
        and (
            reviews_by_image[record["image_name"]].grain_reviewed
            or reviews_by_image[record["image_name"]].tip_reviewed
        )
    ]
    if not eligible_records:
        raise SystemExit("No frames have a completed annotation channel")
    if not labels:
        raise SystemExit("No point labels are available for training")
    reviewed_frame_counts = {
        "grain": sum(review.grain_reviewed for review in reviews),
        "tip": sum(review.tip_reviewed for review in reviews),
    }
    point_label_counts = {
        label_type: sum(label.label_type == label_type for label in labels)
        for label_type in CNN_LABELS
    }
    train_records, validation_records = split_records(
        eligible_records, args.validation_fraction, args.seed
    )
    if args.split_manifest is not None:
        import hashlib as _hl

        def _names(recs):
            # Training order, NOT sorted: split_records shuffles, and the
            # loader preserves file order — sorted would silently change
            # sample exposure on reload (rev5 ordering defect).
            return [r["image_name"] for r in recs]

        if args.split_manifest.exists():
            # S0 frozen split: load and verify against eligible records.
            frozen = json.loads(args.split_manifest.read_text())
            by_name = {r["image_name"]: r for r in eligible_records}
            missing = [n for n in frozen["train"] + frozen["validation"]
                       if n not in by_name]
            extra = [n for n in by_name
                     if n not in set(frozen["train"]) | set(frozen["validation"])]
            if missing or extra:
                raise SystemExit(
                    f"split manifest mismatch: {len(missing)} missing, "
                    f"{len(extra)} unassigned "
                    f"(e.g. {((missing + extra)[:3])})")
            train_records = [by_name[n] for n in frozen["train"]]
            validation_records = [by_name[n] for n in frozen["validation"]]
            print(f"loaded frozen split: {len(train_records)} train / "
                  f"{len(validation_records)} validation frames",
                  flush=True)
        else:
            args.split_manifest.write_text(json.dumps(
                {"train": _names(train_records),
                 "validation": _names(validation_records)}, indent=2) + "\n")
            print(f"wrote canonical split to {args.split_manifest}",
                  flush=True)
    else:
        import hashlib as _hl
    # S1 resolved implicit settings: a null CLI path auto-loaded the
    # dataset file, so null never meant off. Record what actually ran.
    def _sha(p):
        try:
            h = _hl.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            return h.hexdigest()[:16]
        except OSError:
            return "missing"
    corners_path = None if args.no_corners else (
        args.corners_csv or (dataset_dir / "corners.csv"))
    corners_by_image = ({} if corners_path is None
                        else load_corner_negatives(corners_path))
    if corners_by_image:
        print(f"corner negatives: {sum(map(len, corners_by_image.values()))} "
              f"on {len(corners_by_image)} frames from {corners_path}",
              flush=True)
    elif args.no_corners:
        print("corner negatives: explicitly OFF (--no-corners)", flush=True)
    faint_path = None if args.no_faint_override else (
        args.faint_contrast_csv or (dataset_dir / "label_contrast.csv"))
    faint_keys = (load_faint_keys(faint_path, args.faint_contrast_below)
                  if faint_path is not None and Path(faint_path).exists()
                  else set())
    if faint_keys:
        print(f"faint wide-sigma labels: {len(faint_keys)} "
              f"(contrast<{args.faint_contrast_below}, "
              f"sigma={args.faint_sigma}) from {faint_path}", flush=True)
    elif args.no_faint_override:
        print("faint wide-sigma override: explicitly OFF", flush=True)
    training_data = PointPatchDataset(
        dataset_dir,
        train_records,
        labels_by_image,
        reviews_by_image,
        args.patch_size,
        args.patches_per_frame,
        args.seed,
        True,
        corners_by_image,
    )
    validation_data = (
        PointPatchDataset(
            dataset_dir,
            validation_records,
            labels_by_image,
            reviews_by_image,
            args.patch_size,
            max(2, args.patches_per_frame // 3),
            args.seed + 100000,
            False,
            corners_by_image,
        )
        if validation_records
        else None
    )
    for ds in (training_data, validation_data):
        if ds is not None:
            ds.faint_keys = faint_keys
            ds.faint_sigma = args.faint_sigma
            ds.scoped_tip_masks = bool(args.scoped_tip_masks)
            ds.corner_sampling = not bool(args.no_corner_sampling)
    if args.scoped_tip_masks:
        print("tip validity: SCOPED (unknown-by-default discs)", flush=True)
    if args.validity_manifest is not None:
        import json as _json

        try:
            _vman = _json.loads(args.validity_manifest.expanduser().read_text())
            weak_by_image = {
                name: [tuple(p) for p in entry.get("weak_xy", [])]
                for name, entry in _vman.items() if isinstance(entry, dict)}
            neg_by_image = {
                name: [tuple(b) for b in entry.get("neg_boxes", [])]
                for name, entry in _vman.items()
                if isinstance(entry, dict) and entry.get("neg_boxes")}
        except (OSError, ValueError):
            weak_by_image = {}
            neg_by_image = {}
        if weak_by_image:
            print(f"validity ignore-discs: "
                  f"{sum(map(len, weak_by_image.values()))} weak tips on "
                  f"{len(weak_by_image)} frames from {args.validity_manifest}",
                  flush=True)
            for ds in (training_data, validation_data):
                if ds is not None:
                    ds.weak_by_image = weak_by_image
        if neg_by_image:
            print(f"validity verified-negatives: "
                  f"{sum(map(len, neg_by_image.values()))} boxes on "
                  f"{len(neg_by_image)} frames from {args.validity_manifest}",
                  flush=True)
            for ds in (training_data, validation_data):
                if ds is not None:
                    ds.neg_by_image = neg_by_image
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = (
        DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if validation_data is not None
        else None
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = preferred_torch_device(args.device)
    model_config = {
        "input_channels": 2,
        "output_channels": len(CNN_LABELS),
        "base_channels": args.base_channels,
    }
    model = PointHeatmapNet(**model_config).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(Path(args.init_checkpoint), map_location=device,
                             weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print(f"Warm-started from {args.init_checkpoint} "
              f"(best loss {payload.get('training_metadata', {}).get('best_comparison_loss', '?')})",
              flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_loss = float("inf")
    history = []
    checkpoint_path = output_dir / "best-point-heatmap-model.pt"
    created_utc = datetime.now(timezone.utc).isoformat()
    channel_weights = [float(args.grain_loss_weight), 1.0]
    if args.grain_loss_weight != 1.0:
        print(f"grain loss weight: {args.grain_loss_weight}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for inputs, targets, reviewed, valid in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reviewed = reviewed.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_heatmap_loss(model(inputs), targets, reviewed,
                                       valid=valid,
                                       channel_weights=channel_weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses))
        validation_loss = evaluate(model, validation_loader, device,
                                   channel_weights=channel_weights)
        comparison_loss = train_loss if validation_loss is None else validation_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": "" if validation_loss is None else validation_loss,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: train={train_loss:.5f} "
            + (
                "validation=not-available"
                if validation_loss is None
                else f"validation={validation_loss:.5f}"
            ),
            flush=True,
        )
        if comparison_loss < best_loss:
            best_loss = comparison_loss
            save_cnn_checkpoint(
                checkpoint_path,
                model,
                model_config,
                {
                    "created_utc": created_utc,
                    "dataset_manifest": str(dataset_dir / "manifest.json"),
                    "reviewed_frame_count": len(eligible_records),
                    "point_label_count": len(labels),
                    "reviewed_frame_counts": reviewed_frame_counts,
                    "point_label_counts": point_label_counts,
                    "training_arguments": vars(args),
                    "best_comparison_loss": best_loss,
                    "device": str(device),
                    "torch_version": torch.__version__,
                },
            )
    with (output_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    run_manifest = {
        "created_utc": created_utc,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "train_frames": [record["image_name"] for record in train_records],
        "validation_frames": [record["image_name"] for record in validation_records],
        "best_loss": best_loss,
        # P0: the exact human data reaching optimization, by split and
        # channel — a run claiming human supervision with zeros here
        # is void (the v3human failure). Audit with audit_export.py.
        # S0 fix: a label counts only when its frame reviews that
        # channel (the old counter ignored flags and could credit
        # masked-out points, e.g. ld_006572).
        "human_labels_eligible": {
            split: {
                label_type: sum(
                    1 for label in labels
                    if label.image_name in names
                    and label.label_type == label_type
                    and "human" in str(label.provenance)
                    and bool(reviews_by_image.get(label.image_name)
                             and getattr(reviews_by_image[label.image_name],
                                          f"{label_type}_reviewed", False))
                )
                for label_type in CNN_LABELS
            }
            for split, names in (
                ("train", {r["image_name"] for r in train_records}),
                ("validation", {r["image_name"] for r in validation_records}),
            )
        },
        # S0 resolved effective configuration: null CLI paths auto-load,
        # so only resolved files/hashes/counts prove what trained.
        "resolved": {
            "corners_csv": (str(corners_path) if corners_path is not None
                            else "off"),
            "corners_sha16": (_sha(str(corners_path))
                              if corners_path is not None else "off"),
            "corner_points_active": sum(map(len, corners_by_image.values())),
            "faint_contrast_csv": (str(faint_path)
                                   if faint_path is not None else "off"),
            "faint_contrast_sha16": (_sha(str(faint_path))
                                     if faint_path is not None else "off"),
            "faint_keys_active": len(faint_keys),
            "faint_keys_matched_training_tips": sum(
                1 for label in labels
                if label.label_type == "tip"
                and (label.image_name, round(label.x), round(label.y))
                in faint_keys),
            "grain_loss_weight": float(args.grain_loss_weight),
            "scoped_tip_masks": bool(args.scoped_tip_masks),
            "split_manifest": (str(args.split_manifest)
                               if args.split_manifest is not None else "none"),
            "split_sha16": _sha(str(args.split_manifest))
                           if args.split_manifest is not None else "none",
            "updates_per_epoch": -(-len(train_records)
                                                 * args.patches_per_frame
                                                 // args.batch_size),
            "total_updates": (-(-len(train_records) * args.patches_per_frame
                                // args.batch_size) * args.epochs),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    print(f"Best checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
