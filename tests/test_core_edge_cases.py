"""Regression tests for core tracking edge cases and exported measurements."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import TubeTracker as tt


class CoreEdgeCaseTests(unittest.TestCase):
    """Verify graceful behavior for sparse inputs and reviewed event state."""

    def test_empty_detection_file_yields_no_frames(self):
        """An empty detection CSV should produce an empty generator."""
        tracker = tt.Tracker()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "detections.csv"
            path.touch()
            self.assertEqual(list(tracker.detections_generator(str(path))), [])

    def test_track_elongation_accepts_frames_without_tips(self):
        """Tracking should return no tracks when every frame lacks tips."""
        tracker = tt.Tracker()
        tracker.valid_tips = [[], []]
        tracker.track_elongation()
        self.assertEqual(tracker.valid_tracks, [])

    def test_grain_detection_accepts_frame_without_circles(self):
        """Grain detection should tolerate images with no Hough circles."""
        tracker = tt.Tracker(screen_size=(100, 80))
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        tracker.all_detections = tt.Detections(
            img_list=[frame], bg_threshold=55, blur_radius=1
        )
        tracker.grain_det_start = 0
        tracker.grain_det_stop = 0
        tracker.find_grains()
        self.assertEqual(tracker.valid_grains, [])

    def test_reviewed_burst_candidate_updates_grain_state(self):
        """Confirming rupture should clear incompatible germination state."""
        grain = tt.Track(
            boxes=[
                tt.ROI(0, 0, 10, 10, frame=0),
                tt.ROI(1, 0, 11, 10, frame=1),
            ],
            ID="7",
        )
        grain.is_germinated = True
        grain.ger_frame = 1
        grain.set_burst_candidate(
            frame=0,
            confidence=0.7,
            reason="test_evidence",
            track_id="3",
        )

        grain.accept_burst_candidate()

        self.assertTrue(grain.is_bursted)
        self.assertEqual(grain.burst_frame, 0)
        self.assertEqual(grain.burst_method, "reviewed_burst_candidate")
        self.assertFalse(grain.is_burst_candidate)
        self.assertFalse(grain.is_germinated)

    def test_empty_burst_review_is_recorded_as_evaluated(self):
        """An empty candidate pass should still be marked as completed."""
        tracker = tt.Tracker()
        self.assertFalse(tracker.burst_candidates_evaluated)

        self.assertEqual(tracker.find_burst_candidates(), [])

        self.assertTrue(tracker.burst_candidates_evaluated)

    def test_coordinate_export_contains_time_position_and_growth(self):
        """Coordinate rows should include calibrated positions and growth."""
        tracker = tt.Tracker()
        tracker.file_names = ["frame-0", "frame-1"]
        tracker.time_p_frame = 10
        tracker.time_unit = "sec"
        tracker.pxl_dis = 0.5
        tracker.dis_unit = "um"
        tracker.valid_tracks = [
            tt.Track(
                boxes=[
                    tt.ROI(0, 0, 10, 10, frame=0),
                    tt.ROI(3, 4, 13, 14, frame=1),
                ],
                ID="1",
            )
        ]

        rows = tracker.coordinate_rows()

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][3], "time_sec")
        self.assertEqual(rows[1][4:8], [5, 5, 0, 0])
        self.assertEqual(rows[2][3:9], [10, 8, 9, 5.0, 2.5, 0.25])


if __name__ == "__main__":
    unittest.main()
