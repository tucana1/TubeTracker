"""Focused checks for shared prototype geometry and grain tracking."""

import unittest

import cv2 as cv
import numpy as np

from tubetracker.curve_prototype import (
    CurveTraceConfig,
    curve_length,
    ordered_prefix_retention,
    prepare_analysis_gray,
    register_translation,
    robust_normalize,
    score_grain_circle,
    track_grain_candidates,
)
from tubetracker.models import ROI


class CurvePrototypeTests(unittest.TestCase):
    """Protect calibration, registration, circularity, and track continuity."""

    def test_curve_length_respects_nonuniform_resize_scales(self):
        """Reported original-pixel lengths should undo both resize axes."""
        curve = np.array([[0.0, 0.0], [3.0, 4.0]])
        self.assertAlmostEqual(curve_length(curve), 5.0)
        self.assertAlmostEqual(
            curve_length(curve, x_scale=2.0, y_scale=3.0), np.sqrt(5.0)
        )

    def test_ordered_prefix_retention_rejects_a_crossing_route_switch(self):
        """A nearby crossing must not replace the established rooted material path."""
        reference = np.array([[0, 0], [0, 5], [0, 10], [0, 15]], dtype=float)
        extended = np.array(
            [[0, 0], [0, 5], [0, 10], [0, 15], [0, 20]], dtype=float
        )
        switched = np.array(
            [[0, 0], [0, 3], [5, 3], [10, 3], [15, 3]], dtype=float
        )
        self.assertEqual(ordered_prefix_retention(reference, extended, 1.0), 1.0)
        self.assertLess(ordered_prefix_retention(reference, switched, 1.0), 0.5)

    def test_registration_aligns_a_known_translation(self):
        """Global registration should align a frame translated in x and y."""
        previous = np.zeros((96, 96), dtype=np.uint8)
        cv.circle(previous, (35, 50), 9, 220, -1)
        cv.line(previous, (12, 15), (70, 30), 120, 2)
        transform = np.float32([[1, 0, 6], [0, 1, -4]])
        current = cv.warpAffine(previous, transform, (96, 96))
        aligned, shift_x, shift_y, response = register_translation(previous, current)
        self.assertAlmostEqual(shift_x, 6.0, delta=0.6)
        self.assertAlmostEqual(shift_y, -4.0, delta=0.6)
        self.assertGreater(response, 0.5)
        self.assertLess(np.mean(cv.absdiff(aligned, current)), 2.0)

    def test_complete_ring_scores_above_line_debris(self):
        """Pollen-like circular edges should outrank isolated line structure."""
        gray = np.full((100, 120), 220, dtype=np.uint8)
        cv.circle(gray, (35, 50), 9, 40, 2)
        cv.line(gray, (75, 35), (105, 65), 40, 2)
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        gradient = robust_normalize(cv.magnitude(gx, gy))
        circle = score_grain_circle(gray, gradient, 35, 50, 9)
        line = score_grain_circle(gray, gradient, 90, 50, 9)
        self.assertGreater(circle, 0.5)
        self.assertGreater(circle, line + 0.2)

    def test_background_correction_reduces_broad_illumination_difference(self):
        """Background correction should flatten lighting without erasing a ring."""
        x_gradient = np.linspace(70, 220, 160, dtype=np.float32)
        gray = np.repeat(x_gradient[None, :], 100, axis=0).astype(np.uint8)
        cv.circle(gray, (80, 50), 9, 25, 2)
        frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        corrected = prepare_analysis_gray(
            frame,
            CurveTraceConfig(preprocessing="background", background_sigma=25.0),
        )
        raw_side_difference = abs(float(gray[:, :20].mean()) - float(gray[:, -20:].mean()))
        corrected_side_difference = abs(
            float(corrected[:, :20].mean()) - float(corrected[:, -20:].mean())
        )
        self.assertLess(corrected_side_difference, 0.45 * raw_side_difference)
        self.assertGreater(
            float(corrected[50, 80]) - float(corrected[50, 89]),
            20.0,
        )

    def test_unknown_preprocessing_mode_is_rejected(self):
        """A misspelled analysis profile must not silently change results."""
        with self.assertRaises(ValueError):
            prepare_analysis_gray(
                np.zeros((20, 20, 3), dtype=np.uint8),
                CurveTraceConfig(preprocessing="mystery"),
            )

    def test_every_initial_grain_receives_a_position_in_every_frame(self):
        """Tracking should expose weak positions instead of dropping identities."""
        frames = []
        candidates = []
        for frame_index in range(4):
            image = np.full((100, 120, 3), 220, dtype=np.uint8)
            frame_candidates = []
            for x, y in ((30 + 3 * frame_index, 35), (75, 65 + frame_index)):
                cv.circle(image, (x, y), 8, (40, 40, 40), 2)
                roi = ROI(
                    x_l=x - 8,
                    y_t=y - 8,
                    x_r=x + 8,
                    y_b=y + 8,
                    frame=frame_index,
                )
                roi.circle_score = 1.0
                frame_candidates.append(roi)
            frames.append(image)
            candidates.append(frame_candidates)
        tracks, assigned = track_grain_candidates(
            frames, candidates, CurveTraceConfig()
        )
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track.gv1) == len(frames) for track in tracks))
        self.assertTrue(all(len(frame_assignments) == 2 for frame_assignments in assigned))

    def test_persistent_late_pollen_is_promoted_to_a_normal_track(self):
        """A pollen missed initially should receive an ID after later confirmation."""
        frames = []
        candidates = []
        for frame_index in range(5):
            image = np.full((100, 140, 3), 220, dtype=np.uint8)
            frame_candidates = []
            positions = [(30 + frame_index, 40)]
            if frame_index >= 1:
                positions.append((100, 65 + frame_index))
            for x, y in positions:
                cv.circle(image, (x, y), 8, (40, 40, 40), 2)
                roi = ROI(
                    x_l=x - 8,
                    y_t=y - 8,
                    x_r=x + 8,
                    y_b=y + 8,
                    frame=frame_index,
                )
                roi.circle_score = 0.9
                frame_candidates.append(roi)
            frames.append(image)
            candidates.append(frame_candidates)
        tracks, assigned = track_grain_candidates(
            frames, candidates, CurveTraceConfig()
        )
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[1].first_frame(), 1)
        self.assertEqual(tracks[1].last_frame(), 4)
        self.assertIn(1, assigned[1])

        limited_tracks, _ = track_grain_candidates(
            frames, candidates, CurveTraceConfig(max_grains=1)
        )
        self.assertEqual(len(limited_tracks), 1)


if __name__ == "__main__":
    unittest.main()
