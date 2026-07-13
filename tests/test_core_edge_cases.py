"""Regression tests for core tracking edge cases and exported measurements."""

import unittest

import numpy as np

import tubetracker as tt


class CoreEdgeCaseTests(unittest.TestCase):
    """Verify graceful behavior for sparse inputs and reviewed event state."""

    def test_empty_detection_sequence_yields_no_tracks(self):
        """An empty detection sequence should produce no linked tracks."""
        tracker = tt.Tracker()
        self.assertEqual(tracker.link_detections([[], []], max_distance=5), [])

    def test_laptrack_closes_short_detection_gap(self):
        """A nearby detection after a short gap should retain its identity."""
        tracker = tt.Tracker()
        detections = [
            [tt.ROI(0, 0, 10, 10, frame=0, is_tip=True)],
            [tt.ROI(2, 0, 12, 10, frame=1, is_tip=True)],
            [],
            [tt.ROI(4, 0, 14, 10, frame=3, is_tip=True)],
        ]

        tracks = tracker.link_detections(
            detections,
            max_distance=5,
            gap_frames=2,
            gap_distance=5,
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual([roi.gv6 for roi in tracks[0]], [0, 1, 3])

    def test_laptrack_keeps_distant_tips_separate(self):
        """Spatially distinct detections should not be merged into one track."""
        tracker = tt.Tracker()
        detections = [
            [
                tt.ROI(0, 0, 10, 10, frame=0, is_tip=True),
                tt.ROI(90, 0, 100, 10, frame=0, is_tip=True),
            ],
            [
                tt.ROI(2, 0, 12, 10, frame=1, is_tip=True),
                tt.ROI(88, 0, 98, 10, frame=1, is_tip=True),
            ],
        ]

        tracks = tracker.link_detections(detections, max_distance=5)

        self.assertEqual(sorted(len(track) for track in tracks), [2, 2])

    def test_laptrack_respects_minimum_box_overlap(self):
        """The legacy overlap control should remain an actual IoU threshold."""
        tracker = tt.Tracker()
        detections = [
            [tt.ROI(0, 0, 10, 10, frame=0, is_tip=True)],
            [tt.ROI(8, 0, 18, 10, frame=1, is_tip=True)],
        ]

        permissive = tracker.link_detections(detections, min_overlap=0.10)
        strict = tracker.link_detections(detections, min_overlap=0.20)

        self.assertEqual([len(track) for track in permissive], [2])
        self.assertEqual(sorted(len(track) for track in strict), [1, 1])

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
