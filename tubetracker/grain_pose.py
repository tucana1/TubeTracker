"""Estimate pollen-attached rigid coordinates from tracked body points."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class GrainPoseConfig:
    """Configure robust rigid-pose recovery and temporal smoothing."""

    minimum_visible_points: int = 4
    minimum_inlier_points: int = 3
    ransac_threshold_px: float = 3.0
    minimum_scale_for_rotation: float = 0.65
    maximum_scale_for_rotation: float = 1.35
    center_smoothing_window: int = 5
    angle_smoothing_window: int = 21


@dataclass(frozen=True)
class GrainPoseSequence:
    """Store initial-to-current grain transforms and their evidence quality."""

    transforms: np.ndarray
    centers: np.ndarray
    angles_radians: np.ndarray
    observed: np.ndarray
    visible_point_count: np.ndarray
    inlier_point_count: np.ndarray
    median_residual_px: np.ndarray


def _valid_smoothing_window(requested, length):
    """Return an odd Savitzky-Golay window that fits the sequence."""
    window = min(max(1, int(requested)), int(length))
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def _interpolate_finite(values):
    """Interpolate finite samples and hold the nearest value at each edge."""
    values = np.asarray(values, dtype=np.float64)
    output = values.copy()
    frames = np.arange(len(values), dtype=np.float64)
    if values.ndim == 1:
        finite = np.isfinite(values)
        if not np.any(finite):
            raise ValueError("pose sequence contains no finite observations")
        return np.interp(frames, frames[finite], values[finite])
    for column in range(values.shape[1]):
        output[:, column] = _interpolate_finite(values[:, column])
    return output


def _smooth_series(values, requested_window):
    """Smooth one finite series without changing its sequence length."""
    window = _valid_smoothing_window(requested_window, len(values))
    if window < 3:
        return np.asarray(values, dtype=np.float64)
    return savgol_filter(
        np.asarray(values, dtype=np.float64),
        window_length=window,
        polyorder=min(2, window - 1),
        mode="interp",
    )


def _pose_matrix(initial_center, current_center, angle):
    """Return the rigid affine transform from initial to current coordinates."""
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    translation = np.asarray(current_center) - rotation @ np.asarray(initial_center)
    return np.column_stack((rotation, translation))


def estimate_grain_poses(
    initial_points,
    tracks,
    visibility,
    center_index=0,
    config=None,
):
    """Recover a smoothed rigid grain pose from redundant point trajectories."""
    config = config or GrainPoseConfig()
    initial_points = np.asarray(initial_points, dtype=np.float64)
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if initial_points.ndim != 2 or initial_points.shape[1] != 2:
        raise ValueError("initial_points must have shape (points, 2)")
    if tracks.ndim != 3 or tracks.shape[1:] != initial_points.shape:
        raise ValueError("tracks must have shape (frames, points, 2)")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape (frames, points)")
    if not 0 <= int(center_index) < len(initial_points):
        raise ValueError("center_index lies outside initial_points")

    frame_count = len(tracks)
    initial_center = initial_points[int(center_index)]
    centers = np.full((frame_count, 2), np.nan, dtype=np.float64)
    angles = np.full(frame_count, np.nan, dtype=np.float64)
    visible_counts = np.zeros(frame_count, dtype=int)
    inlier_counts = np.zeros(frame_count, dtype=int)
    residuals = np.full(frame_count, np.nan, dtype=np.float64)
    observed = np.zeros(frame_count, dtype=bool)
    cv.setRNGSeed(7)

    for frame in range(frame_count):
        finite = np.isfinite(tracks[frame]).all(axis=1)
        good = visibility[frame] & finite
        visible_counts[frame] = int(np.count_nonzero(good))
        if visible_counts[frame] < config.minimum_visible_points:
            continue
        source = initial_points[good]
        target = tracks[frame, good]
        centers[frame] = initial_center + np.median(target - source, axis=0)
        matrix, inliers = cv.estimateAffinePartial2D(
            source,
            target,
            method=cv.RANSAC,
            ransacReprojThreshold=float(config.ransac_threshold_px),
            maxIters=2000,
            confidence=0.99,
            refineIters=20,
        )
        if matrix is None:
            continue
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        inlier_counts[frame] = inlier_count
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if (
            inlier_count < config.minimum_inlier_points
            or not config.minimum_scale_for_rotation
            <= scale
            <= config.maximum_scale_for_rotation
        ):
            continue
        angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        angles[frame] = angle
        rigid = _pose_matrix(initial_center, centers[frame], angle)
        predicted = source @ rigid[:, :2].T + rigid[:, 2]
        residuals[frame] = float(
            np.median(np.linalg.norm(predicted - target, axis=1))
        )
        observed[frame] = True

    centers = _interpolate_finite(centers)
    centers[:, 0] = _smooth_series(
        centers[:, 0], config.center_smoothing_window
    )
    centers[:, 1] = _smooth_series(
        centers[:, 1], config.center_smoothing_window
    )
    finite_angles = np.isfinite(angles)
    if np.any(finite_angles):
        unwrapped = np.full_like(angles, np.nan)
        unwrapped[finite_angles] = np.unwrap(angles[finite_angles])
        angles = _interpolate_finite(unwrapped)
        angles = _smooth_series(angles, config.angle_smoothing_window)
    else:
        angles = np.zeros(frame_count, dtype=np.float64)

    transforms = np.stack(
        [
            _pose_matrix(initial_center, centers[frame], angles[frame])
            for frame in range(frame_count)
        ]
    )
    return GrainPoseSequence(
        transforms=transforms,
        centers=centers,
        angles_radians=angles,
        observed=observed,
        visible_point_count=visible_counts,
        inlier_point_count=inlier_counts,
        median_residual_px=residuals,
    )


def stabilization_matrix(transform, initial_center, output_center):
    """Map one current source frame into a pollen-attached output crop."""
    inverse = cv.invertAffineTransform(np.asarray(transform, dtype=np.float64))
    offset = np.asarray(output_center, dtype=np.float64) - np.asarray(
        initial_center, dtype=np.float64
    )
    inverse[:, 2] += offset
    return inverse
