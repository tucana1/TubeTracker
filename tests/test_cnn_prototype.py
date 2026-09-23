"""Focused checks for point labels, CNN heatmaps, and temporal linking."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tubetracker.cnn_prototype import (
    FrameReview,
    PointHeatmapNet,
    PointLabel,
    extract_heatmap_points,
    link_point_detections,
    load_cnn_checkpoint,
    masked_heatmap_loss,
    point_heatmaps,
    read_frame_reviews,
    read_point_labels,
    resolve_grain_source,
    save_cnn_checkpoint,
    write_frame_reviews,
    write_point_labels,
)


class CnnPrototypeTests(unittest.TestCase):
    """Protect click-only supervision and prior-informed point trajectories."""

    def test_annotation_tables_round_trip_independent_review_channels(self):
        """Tip-only reviewed frames must remain distinct from grain labels."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = [
                PointLabel(
                    "frame.png",
                    12,
                    "tip",
                    31.5,
                    42.0,
                    provenance="opencv_hough_suggestion",
                    confidence=0.77,
                )
            ]
            reviews = [FrameReview("frame.png", 12, False, True)]
            write_point_labels(root / "labels.csv", labels)
            write_frame_reviews(root / "reviews.csv", reviews)
            self.assertEqual(read_point_labels(root / "labels.csv"), labels)
            self.assertEqual(read_frame_reviews(root / "reviews.csv"), reviews)

    def test_point_heatmap_recovers_grain_and_tip_clicks(self):
        """Each click should create one peak in its own output channel."""
        labels = [
            PointLabel("frame.png", 0, "grain", 20.0, 25.0),
            PointLabel("frame.png", 0, "tip", 61.0, 50.0),
        ]
        heatmaps = point_heatmaps(80, 90, labels)
        grain = extract_heatmap_points(heatmaps[0], threshold=0.9)
        tip = extract_heatmap_points(heatmaps[1], threshold=0.9)
        self.assertEqual(grain, [(20.0, 25.0, 1.0)])
        self.assertEqual(tip, [(61.0, 50.0, 1.0)])

    def test_model_preserves_patch_shape_and_checkpoint_round_trips(self):
        """The CNN should emit two aligned maps and reload learned parameters."""
        model = PointHeatmapNet(base_channels=4)
        inputs = torch.zeros((2, 2, 64, 72), dtype=torch.float32)
        self.assertEqual(tuple(model(inputs).shape), (2, 2, 64, 72))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            config = {"input_channels": 2, "output_channels": 2, "base_channels": 4}
            save_cnn_checkpoint(path, model, config, {"test": True})
            loaded, payload = load_cnn_checkpoint(path)
            self.assertTrue(payload["training_metadata"]["test"])
            self.assertEqual(tuple(loaded(inputs).shape), (2, 2, 64, 72))

    def test_loss_ignores_an_unreviewed_channel(self):
        """Unreviewed pollen must not be interpreted as negative supervision."""
        logits = torch.zeros((1, 2, 16, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 0, 5, 5] = 1.0
        reviewed = torch.tensor([[0.0, 1.0]])
        baseline = masked_heatmap_loss(logits, targets, reviewed)
        altered = logits.detach().clone()
        altered[0, 0] = 20.0
        changed = masked_heatmap_loss(altered, targets, reviewed)
        self.assertAlmostEqual(
            float(baseline.detach()), float(changed.detach()), places=6
        )

    def test_linking_uses_prior_motion_when_detections_cross(self):
        """A growing trajectory should retain its forward identity at a crossing."""
        detections = [
            [(10.0, 20.0, 0.9), (50.0, 20.0, 0.9)],
            [(20.0, 20.0, 0.9), (40.0, 20.0, 0.9)],
            [(31.0, 20.0, 0.9), (29.0, 20.0, 0.9)],
            [(40.0, 20.0, 0.9), (20.0, 20.0, 0.9)],
        ]
        linked = link_point_detections(
            detections, max_distance=20.0, direction_weight=0.8
        )
        first_track = [point for point in linked if point.track_id == 1]
        self.assertEqual([round(point.x) for point in first_track], [10, 20, 31, 40])

    def test_tip_only_checkpoint_uses_established_grain_tracker(self):
        """Tip-only annotation should not run an untrained CNN grain channel."""
        metadata = {"reviewed_frame_counts": {"grain": 0, "tip": 12}}
        self.assertEqual(resolve_grain_source("auto", metadata), "opencv")
        self.assertEqual(resolve_grain_source("cnn", metadata), "cnn")


if __name__ == "__main__":
    unittest.main()
