"""Click-label data, CNN heatmaps, and temporal linking for the CNN prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.feature import peak_local_max

from .curve_prototype import CurveTraceConfig, prepare_analysis_gray

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
except ImportError:  # pragma: no cover - exercised only without the cnn extra
    torch = None
    nn = None
    torch_functional = None


CNN_LABELS = ("grain", "tip")
CNN_DATASET_VERSION = 1
CNN_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class PointLabel:
    """Store one pollen-center or tube-tip click in an exported frame."""

    image_name: str
    source_frame: int
    label_type: str
    x: float
    y: float
    provenance: str = "manual"
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameReview:
    """Record which label channels are complete for one exported frame."""

    image_name: str
    source_frame: int
    grain_reviewed: bool = False
    tip_reviewed: bool = False


@dataclass
class LinkedPoint:
    """Store one confidence-scored point assigned to a temporal track."""

    track_id: int
    analysis_frame: int
    source_frame: int
    x: float
    y: float
    confidence: float


def write_point_labels(path, labels):
    """Write point annotations in a stable, human-readable CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_name",
                "source_frame",
                "label_type",
                "x",
                "y",
                "provenance",
                "confidence",
            ),
        )
        writer.writeheader()
        for label in sorted(
            labels,
            key=lambda item: (item.source_frame, item.label_type, item.y, item.x),
        ):
            if label.label_type not in CNN_LABELS:
                raise ValueError(f"Unknown label type: {label.label_type}")
            writer.writerow(asdict(label))


def read_point_labels(path):
    """Read point annotations, returning an empty list for a new dataset."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = []
    for row in rows:
        label_type = row["label_type"]
        if label_type not in CNN_LABELS:
            raise ValueError(f"Unknown label type: {label_type}")
        labels.append(
            PointLabel(
                image_name=row["image_name"],
                source_frame=int(row["source_frame"]),
                label_type=label_type,
                x=float(row["x"]),
                y=float(row["y"]),
                provenance=row.get("provenance") or "manual",
                confidence=float(row.get("confidence") or 1.0),
            )
        )
    return labels


def write_frame_reviews(path, reviews):
    """Write per-channel annotation-completeness flags for exported frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_name",
                "source_frame",
                "grain_reviewed",
                "tip_reviewed",
            ),
        )
        writer.writeheader()
        for review in sorted(reviews, key=lambda item: item.source_frame):
            row = asdict(review)
            row["grain_reviewed"] = int(review.grain_reviewed)
            row["tip_reviewed"] = int(review.tip_reviewed)
            writer.writerow(row)


def read_frame_reviews(path):
    """Read per-channel annotation-completeness flags."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        FrameReview(
            image_name=row["image_name"],
            source_frame=int(row["source_frame"]),
            grain_reviewed=bool(int(row["grain_reviewed"])),
            tip_reviewed=bool(int(row["tip_reviewed"])),
        )
        for row in rows
    ]


def write_dataset_manifest(path, payload):
    """Write reproducible annotation-dataset metadata."""
    document = {"dataset_version": CNN_DATASET_VERSION, **payload}
    Path(path).write_text(json.dumps(document, indent=2) + "\n")


def load_dataset_manifest(path):
    """Load and validate annotation-dataset metadata."""
    document = json.loads(Path(path).read_text())
    if document.get("dataset_version") != CNN_DATASET_VERSION:
        raise ValueError("Unsupported CNN annotation dataset version")
    return document


def frame_to_model_input(frame, preprocessing="background"):
    """Return raw and illumination-corrected grayscale model channels."""
    raw = (
        frame.astype(np.uint8, copy=False)
        if frame.ndim == 2
        else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    )
    corrected = prepare_analysis_gray(
        frame,
        CurveTraceConfig(preprocessing=preprocessing),
    )
    return np.stack((raw, corrected)).astype(np.float32) / 255.0


def point_heatmaps(height, width, labels, sigma_by_label=None,
                   sigma_overrides=None):
    """Render click coordinates as two Gaussian supervision heatmaps.

    ``sigma_overrides`` (H207): optional per-label sigma aligned with
    ``labels`` (e.g. wide Gaussians for faint tip labels whose heat
    response is diffuse); falls back to ``sigma_by_label`` per type.
    """
    sigma_by_label = sigma_by_label or {"grain": 3.0, "tip": 2.0}
    heatmaps = np.zeros((len(CNN_LABELS), height, width), dtype=np.float32)
    for k, label in enumerate(labels):
        if label.label_type not in CNN_LABELS:
            raise ValueError(f"Unknown label type: {label.label_type}")
        if sigma_overrides is not None and sigma_overrides[k] is not None:
            sigma = float(sigma_overrides[k])
        else:
            sigma = float(sigma_by_label[label.label_type])
        radius = max(1, int(math.ceil(3.0 * sigma)))
        center_x = int(round(label.x))
        center_y = int(round(label.y))
        left = max(0, center_x - radius)
        right = min(width, center_x + radius + 1)
        top = max(0, center_y - radius)
        bottom = min(height, center_y + radius + 1)
        if left >= right or top >= bottom:
            continue
        y_grid, x_grid = np.ogrid[top:bottom, left:right]
        gaussian = np.exp(
            -((x_grid - label.x) ** 2 + (y_grid - label.y) ** 2)
            / (2.0 * sigma * sigma)
        ).astype(np.float32)
        channel = CNN_LABELS.index(label.label_type)
        heatmaps[channel, top:bottom, left:right] = np.maximum(
            heatmaps[channel, top:bottom, left:right], gaussian
        )
    return heatmaps


if nn is not None:

    class _ConvBlock(nn.Module):
        """Apply two normalized convolutions with smooth nonlinearities."""

        def __init__(self, input_channels, output_channels):
            super().__init__()
            groups = max(1, min(8, output_channels // 4))
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, inputs):
            """Return the transformed feature map."""
            return self.layers(inputs)


    class PointHeatmapNet(nn.Module):
        """Predict pollen-center and tube-tip heatmaps with a compact U-Net."""

        def __init__(self, input_channels=2, output_channels=2, base_channels=16):
            super().__init__()
            self.encoder1 = _ConvBlock(input_channels, base_channels)
            self.encoder2 = _ConvBlock(base_channels, base_channels * 2)
            self.encoder3 = _ConvBlock(base_channels * 2, base_channels * 4)
            self.bottleneck = _ConvBlock(base_channels * 4, base_channels * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, 2, stride=2
            )
            self.decoder3 = _ConvBlock(base_channels * 8, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 2, stride=2
            )
            self.decoder2 = _ConvBlock(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(
                base_channels * 2, base_channels, 2, stride=2
            )
            self.decoder1 = _ConvBlock(base_channels * 2, base_channels)
            self.head = nn.Conv2d(base_channels, output_channels, 1)

        def forward(self, inputs):
            """Return one unnormalized heatmap per point class."""
            encoder1 = self.encoder1(inputs)
            encoder2 = self.encoder2(self.pool(encoder1))
            encoder3 = self.encoder3(self.pool(encoder2))
            bottleneck = self.bottleneck(self.pool(encoder3))
            decoder3 = self.decoder3(
                torch.cat((self.up3(bottleneck), encoder3), dim=1)
            )
            decoder2 = self.decoder2(
                torch.cat((self.up2(decoder3), encoder2), dim=1)
            )
            decoder1 = self.decoder1(
                torch.cat((self.up1(decoder2), encoder1), dim=1)
            )
            return self.head(decoder1)

else:

    class PointHeatmapNet:  # pragma: no cover - only used without torch installed
        """Explain how to enable the optional CNN dependency."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install TubeTracker with the 'cnn' optional dependency")


def masked_heatmap_loss(logits, targets, reviewed_channels, valid=None,
                        channel_weights=None):
    """Compute class-balanced heatmap loss only for fully reviewed channels.

    ``valid`` (H226, optional per-pixel 0/1 broadcastable to logits):
    pixels marked 0 contribute zero gradient (unreviewed weak-tip
    regions).  Default None preserves legacy channel-only behavior.

    ``channel_weights`` (S1, optional 2-vector): multiplies the two
    head losses (grain, tip). Grain weight 0 isolates tip-only
    diagnosis; default None preserves legacy equal weighting.
    """
    if torch_functional is None:
        raise RuntimeError("PyTorch is required for CNN training")
    channel_mask = reviewed_channels[:, :, None, None]
    if valid is not None:
        channel_mask = channel_mask * valid
    if channel_weights is not None:
        w = torch.as_tensor(list(channel_weights),
                            dtype=channel_mask.dtype,
                            device=channel_mask.device)[:, None, None]
        channel_mask = channel_mask * w
    weights = 1.0 + 24.0 * targets
    element_loss = torch_functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weighted = element_loss * weights * channel_mask
    if valid is None:
        denominator = torch.clamp(channel_mask.sum() * targets.shape[-1] * targets.shape[-2], min=1.0)
    else:
        denominator = torch.clamp(channel_mask.sum(), min=1.0)
    return weighted.sum() / denominator


def preferred_torch_device(requested="auto"):
    """Select Apple Metal when available, with deterministic CPU fallback."""
    if torch is None:
        raise RuntimeError("PyTorch is required for the CNN prototype")
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_grain_source(requested, training_metadata):
    """Choose CNN grains unless a tip-only checkpoint requires OpenCV grains."""
    if requested not in {"auto", "cnn", "opencv"}:
        raise ValueError(f"Unknown grain source: {requested}")
    if requested != "auto":
        return requested
    reviewed_counts = training_metadata.get("reviewed_frame_counts")
    if reviewed_counts is not None and reviewed_counts.get("grain", 0) == 0:
        return "opencv"
    return "cnn"


def save_cnn_checkpoint(path, model, model_config, training_metadata):
    """Save learned parameters with architecture and training provenance."""
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN checkpoints")
    payload = {
        "checkpoint_version": CNN_CHECKPOINT_VERSION,
        "labels": CNN_LABELS,
        "model_config": dict(model_config),
        "training_metadata": dict(training_metadata),
        "state_dict": model.state_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_cnn_checkpoint(path, device="cpu"):
    """Reconstruct a point-heatmap model from a versioned checkpoint."""
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN checkpoints")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != CNN_CHECKPOINT_VERSION:
        raise ValueError("Unsupported CNN checkpoint version")
    if tuple(payload.get("labels", ())) != CNN_LABELS:
        raise ValueError("Checkpoint label channels do not match this prototype")
    model = PointHeatmapNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model, payload


def extract_heatmap_points(heatmap, threshold=0.35, min_distance=6):
    """Convert one probability heatmap into confidence-scored point detections."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    coordinates = peak_local_max(
        heatmap,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(threshold),
        exclude_border=False,
    )
    return [
        (float(column), float(row), float(heatmap[row, column]))
        for row, column in coordinates
    ]


def predict_heatmaps_tiled(
    model,
    frame,
    device,
    tile_size=256,
    overlap=48,
    preprocessing="background",
):
    """Predict full-frame heatmaps with overlap-weighted fixed-size tiles."""
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN inference")
    if tile_size < 32 or tile_size % 8 != 0:
        raise ValueError("Tile size must be at least 32 and divisible by eight")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than the tile")
    inputs = frame_to_model_input(frame, preprocessing=preprocessing)
    _, height, width = inputs.shape
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.pad(
        inputs,
        (
            (0, 0),
            (0, padded_height - height),
            (0, padded_width - width),
        ),
        mode="reflect",
    )
    stride = tile_size - overlap

    def starts(length):
        values = list(range(0, max(1, length - tile_size + 1), stride))
        final = length - tile_size
        if not values or values[-1] != final:
            values.append(final)
        return values

    y_starts = starts(padded_height)
    x_starts = starts(padded_width)
    window_1d = np.hanning(tile_size).astype(np.float32)
    window = np.maximum(np.outer(window_1d, window_1d), 0.05)
    accumulated = np.zeros(
        (len(CNN_LABELS), padded_height, padded_width), dtype=np.float32
    )
    weights = np.zeros((padded_height, padded_width), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for top in y_starts:
            for left in x_starts:
                tile = torch.from_numpy(
                    padded[:, top : top + tile_size, left : left + tile_size]
                )[None].to(device)
                prediction = torch.sigmoid(model(tile))[0].cpu().numpy()
                accumulated[
                    :, top : top + tile_size, left : left + tile_size
                ] += prediction * window[None]
                weights[top : top + tile_size, left : left + tile_size] += window
    accumulated /= np.maximum(weights[None], 1e-6)
    return accumulated[:, :height, :width]


def _track_prediction(history):
    """Predict one track's next point with constant velocity."""
    if len(history) < 2:
        return np.array([history[-1].x, history[-1].y], dtype=np.float64)
    previous = np.array([history[-2].x, history[-2].y], dtype=np.float64)
    current = np.array([history[-1].x, history[-1].y], dtype=np.float64)
    return current + (current - previous)


def link_point_detections(
    detections_by_frame,
    source_frames=None,
    max_distance=35.0,
    max_gap=2,
    direction_weight=0.35,
):
    """Link points using predicted position and prior trajectory direction."""
    source_frames = source_frames or list(range(len(detections_by_frame)))
    tracks = {}
    last_seen = {}
    next_track_id = 1
    output = []
    for frame_index, detections in enumerate(detections_by_frame):
        available_tracks = [
            track_id
            for track_id, seen_frame in last_seen.items()
            if frame_index - seen_frame <= max_gap + 1
        ]
        possible = []
        for track_id in available_tracks:
            history = tracks[track_id]
            prediction = _track_prediction(history)
            current = np.array([history[-1].x, history[-1].y], dtype=np.float64)
            velocity = prediction - current
            velocity_norm = float(np.linalg.norm(velocity))
            for detection_index, (x, y, confidence) in enumerate(detections):
                point = np.array([x, y], dtype=np.float64)
                distance = float(np.linalg.norm(point - prediction))
                if distance > max_distance:
                    continue
                direction_penalty = 0.0
                if velocity_norm > 1.0:
                    displacement = point - current
                    displacement_norm = float(np.linalg.norm(displacement))
                    if displacement_norm > 0:
                        alignment = float(
                            np.dot(velocity, displacement)
                            / (velocity_norm * displacement_norm)
                        )
                        direction_penalty = direction_weight * max_distance * (1.0 - alignment)
                possible.append(
                    (
                        distance + direction_penalty - 2.0 * confidence,
                        track_id,
                        detection_index,
                    )
                )
        assignments = {}
        assigned_detections = set()
        for _, track_id, detection_index in sorted(possible):
            if track_id in assignments or detection_index in assigned_detections:
                continue
            assignments[track_id] = detection_index
            assigned_detections.add(detection_index)

        for track_id, detection_index in assignments.items():
            x, y, confidence = detections[detection_index]
            point = LinkedPoint(
                track_id=track_id,
                analysis_frame=frame_index,
                source_frame=int(source_frames[frame_index]),
                x=float(x),
                y=float(y),
                confidence=float(confidence),
            )
            tracks[track_id].append(point)
            last_seen[track_id] = frame_index
            output.append(point)
        for detection_index, (x, y, confidence) in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            point = LinkedPoint(
                track_id=next_track_id,
                analysis_frame=frame_index,
                source_frame=int(source_frames[frame_index]),
                x=float(x),
                y=float(y),
                confidence=float(confidence),
            )
            tracks[next_track_id] = [point]
            last_seen[next_track_id] = frame_index
            output.append(point)
            next_track_id += 1
    return output
