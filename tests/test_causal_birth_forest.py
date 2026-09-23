"""Focused tests for the movie-wide causal orientation atlas."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from tubetracker.causal_birth_forest import (
    CausalBirthAtlasConfig,
    birth_order_score,
    build_causal_birth_atlas,
    monotone_path_births,
    track_pollen_centers,
    track_pollen_centers_from_seeds,
)
from tubetracker.orientation_worldsheet import PairedWallOrientationResult


def feature(score: np.ndarray) -> PairedWallOrientationResult:
    """Construct a complete synthetic ribbon feature observation."""

    score = np.asarray(score, dtype=np.float32)
    return PairedWallOrientationResult(
        score=score,
        paired_score=0.9 * score,
        half_width_px=np.full_like(score, 2.0),
        wall_balance=np.ones_like(score),
    )


def test_atlas_recovers_persistent_oriented_birth_and_ignores_flash() -> None:
    """Only material that persists across the configured window is born."""

    observations = []
    for sample in range(12):
        score = np.zeros((2, 5, 6), dtype=np.float32)
        score[0, 2, 3] = 0.2
        if sample >= 5:
            score[1, 1, 4] = 0.8
        if sample == 6:
            score[0, 4, 1] = 1.0
        observations.append(feature(score))
    atlas = build_causal_birth_atlas(
        observations,
        len(observations),
        CausalBirthAtlasConfig(
            warmup_samples=3,
            persistence_window=3,
            persistence_required=2,
            minimum_change=0.1,
            noise_multiplier=0.0,
            tail_samples=4,
        ),
    )
    assert atlas.birth_sample[0, 2, 3] == 0
    assert atlas.birth_sample[1, 1, 4] == 5
    assert atlas.birth_sample[0, 4, 1] == len(observations)
    assert atlas.aggregate.paired_score[1, 1, 4] > 0.6


def test_orientation_layers_keep_crossing_births_separate() -> None:
    """Two directions at one pixel retain independent appearance times."""

    observations = []
    for sample in range(11):
        score = np.zeros((2, 3, 3), dtype=np.float32)
        if sample >= 4:
            score[0, 1, 1] = 0.7
        if sample >= 7:
            score[1, 1, 1] = 0.8
        observations.append(feature(score))
    atlas = build_causal_birth_atlas(
        observations,
        len(observations),
        CausalBirthAtlasConfig(
            warmup_samples=3,
            persistence_window=2,
            persistence_required=2,
            minimum_change=0.1,
            noise_multiplier=0.0,
            tail_samples=3,
        ),
    )
    assert atlas.birth_sample[:, 1, 1].tolist() == [4, 7]


def test_pollen_tracking_follows_translation() -> None:
    """Dense local registration bridges motion missed by sparse linking."""

    base = np.full((51, 61), 180, dtype=np.uint8)
    cv.circle(base, (24, 22), 5, 70, -1)
    frames = []
    for dx, dy in [(0, 0), (2, 1), (4, 2), (6, 3)]:
        matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        frames.append(
            cv.warpAffine(base, matrix, (61, 51), borderValue=180)
        )
    tracks, scores = track_pollen_centers(
        np.asarray(frames),
        np.asarray([[22.0, 24.0]]),
        template_radius_px=6,
        search_radius_px=3,
    )
    np.testing.assert_allclose(tracks[0, -1], [25.0, 30.0], atol=0.6)
    assert np.min(scores) > 0.8


def test_pollen_tracking_can_start_from_a_later_reliable_sighting() -> None:
    """A later seed reconstructs the same grain path in both time directions."""

    base = np.full((51, 61), 180, dtype=np.uint8)
    cv.circle(base, (24, 22), 5, 70, -1)
    frames = []
    for dx, dy in [(0, 0), (2, 1), (4, 2), (6, 3)]:
        matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        frames.append(cv.warpAffine(base, matrix, (61, 51), borderValue=180))
    tracks, scores = track_pollen_centers_from_seeds(
        np.asarray(frames),
        np.asarray([[24.0, 28.0]]),
        np.asarray([2]),
        template_radius_px=6,
        search_radius_px=3,
    )
    np.testing.assert_allclose(tracks[0, 0], [22.0, 24.0], atol=0.6)
    np.testing.assert_allclose(tracks[0, -1], [25.0, 30.0], atol=0.6)
    assert np.min(scores) > 0.8


def test_monotone_births_prevent_distal_material_from_appearing_first() -> None:
    """A traced prefix obeys apical growth even with noisy local birth times."""

    fitted = monotone_path_births(
        np.asarray([4, 6, 5, 9, 99]),
        unavailable_value=99,
    )
    assert fitted.tolist() == [4, 6, 6, 9, 99]
    assert birth_order_score(
        np.asarray([4, 6, 5, 9]),
        unavailable_value=99,
    ) == 1.0
