"""Tests for bounded, provenance-preserving pilot video loading."""

import unittest
from pathlib import Path

from scripts.run_pilot import load_video, source_frame_for
from scripts.prepare_material_video import source_frame_schedule
from tubetracker.sampling import plan_tracking_and_output_indices


REPO_ROOT = Path(__file__).resolve().parents[1]


class PilotVideoTests(unittest.TestCase):
    """Verify large videos can be sampled without losing source-frame identity."""

    def test_load_video_honors_frame_range_step_and_limit(self):
        """Only requested source frames should be retained in memory."""
        frames, metadata = load_video(
            REPO_ROOT / "sample_movie.avi",
            screen_size=(100, 80),
            max_frames=4,
            rotate=False,
            start_frame=2,
            end_frame=20,
            frame_step=3,
        )

        self.assertEqual(len(frames), 4)
        self.assertEqual(metadata["source_frame_indices"], [2, 5, 8, 11])
        self.assertEqual(metadata["source_frame_step"], 3)
        self.assertEqual(frames[0].shape[:2], (80, 100))

    def test_source_frame_mapping_rejects_missing_analysis_frames(self):
        """Unavailable event frames should remain explicit missing values."""
        source_frames = [10, 20, 30]
        self.assertEqual(source_frame_for(source_frames, 1), 20)
        self.assertEqual(source_frame_for(source_frames, -1), -1)
        self.assertEqual(source_frame_for(source_frames, 3), -1)

    def test_sparse_outputs_are_nested_inside_dense_tracking_frames(self):
        """Scientific output sampling must not create identity-tracking gaps."""
        tracking, output_positions = plan_tracking_and_output_indices(480, 240, 40)
        self.assertEqual(len(tracking), 240)
        self.assertEqual(len(output_positions), 40)
        self.assertEqual(tracking[output_positions[0]], 0)
        self.assertEqual(tracking[output_positions[-1]], 479)
        self.assertLessEqual(max(tracking[1:] - tracking[:-1]), 3)

    def test_frame_schedule_rejects_nonpositive_counts(self):
        """Invalid frame schedules should fail before a long analysis starts."""
        for values in ((0, 5, 3), (10, 0, 3), (10, 5, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan_tracking_and_output_indices(*values)

    def test_material_preparation_spans_the_complete_source_video(self):
        """Prepared caches should include both video boundaries without duplicates."""
        self.assertEqual(
            source_frame_schedule(100, 5).tolist(), [0, 25, 50, 74, 99]
        )
        self.assertEqual(source_frame_schedule(3, 10).tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
