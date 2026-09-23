"""Build a causal, orientation-lifted atlas of growing tubular material.

The atlas integrates evidence before tracing.  Each image location retains a
separate appearance time for every tube orientation, so two projected tubes
can cross without becoming the same graph state.  This is intentionally
different from frame-wise segmentation followed by association: weak material
may accumulate support over the complete movie before any owner path is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from .orientation_worldsheet import PairedWallOrientationResult


@dataclass(frozen=True)
class CausalBirthAtlasConfig:
    """Configure persistent oriented-material discovery and late fusion."""

    warmup_samples: int = 8
    persistence_window: int = 4
    persistence_required: int = 3
    minimum_change: float = 0.055
    noise_multiplier: float = 3.0
    preexisting_score: float = 0.12
    preexisting_fraction: float = 0.75
    tail_samples: int = 24


@dataclass(frozen=True)
class CausalBirthAtlas:
    """Store fused ribbon evidence and one causal birth time per orientation."""

    aggregate: PairedWallOrientationResult
    birth_sample: np.ndarray
    persistent_fraction: np.ndarray
    preexisting: np.ndarray
    sample_count: int


def build_causal_birth_atlas(
    features: Iterable[PairedWallOrientationResult],
    sample_count: int,
    config: CausalBirthAtlasConfig = CausalBirthAtlasConfig(),
) -> CausalBirthAtlas:
    """Integrate a feature stream into a persistent orientation-and-birth atlas."""

    _validate_config(config, sample_count)
    iterator = iter(features)
    warmup = []
    for _ in range(config.warmup_samples):
        try:
            feature = next(iterator)
        except StopIteration as error:
            raise ValueError("feature stream ended during warmup") from error
        _validate_feature(feature)
        warmup.append(np.asarray(feature.score, dtype=np.float32))

    warmup_stack = np.stack(warmup)
    baseline = np.median(warmup_stack, axis=0)
    noise = 1.4826 * np.median(
        np.abs(warmup_stack - baseline[None, ...]), axis=0
    )
    threshold = baseline + np.maximum(
        config.minimum_change,
        config.noise_multiplier * noise,
    )
    preexisting = np.mean(
        warmup_stack >= config.preexisting_score,
        axis=0,
    ) >= config.preexisting_fraction

    shape = baseline.shape
    birth = np.full(shape, sample_count, dtype=np.int32)
    birth[preexisting] = 0
    active_observations = np.zeros(shape, dtype=np.uint16)
    rolling = np.zeros(shape, dtype=np.uint8)
    activity_ring = np.zeros(
        (config.persistence_window, *shape), dtype=np.uint8
    )

    tail_start = max(config.warmup_samples, sample_count - config.tail_samples)
    tail_count = 0
    score_sum = np.zeros(shape, dtype=np.float32)
    paired_sum = np.zeros(shape, dtype=np.float32)
    width_weighted_sum = np.zeros(shape, dtype=np.float32)
    balance_weighted_sum = np.zeros(shape, dtype=np.float32)
    support_weight_sum = np.zeros(shape, dtype=np.float32)

    processed = config.warmup_samples
    for sample, feature in enumerate(iterator, start=config.warmup_samples):
        if sample >= sample_count:
            raise ValueError("feature stream contains more samples than declared")
        _validate_feature(feature, shape)
        score = np.asarray(feature.score, dtype=np.float32)
        active = score >= threshold
        active_observations += active

        ring_index = sample % config.persistence_window
        rolling -= activity_ring[ring_index]
        activity_ring[ring_index] = active
        rolling += activity_ring[ring_index]
        newly_born = (
            (rolling >= config.persistence_required)
            & (birth == sample_count)
            & ~preexisting
        )
        birth[newly_born] = max(
            config.warmup_samples,
            sample - config.persistence_required + 1,
        )

        if sample >= tail_start:
            paired = np.asarray(feature.paired_score, dtype=np.float32)
            weight = np.maximum(paired, 0.05 * score)
            score_sum += score
            paired_sum += paired
            width_weighted_sum += np.asarray(
                feature.half_width_px, dtype=np.float32
            ) * weight
            balance_weighted_sum += np.asarray(
                feature.wall_balance, dtype=np.float32
            ) * weight
            support_weight_sum += weight
            tail_count += 1
        processed = sample + 1

    if processed != sample_count:
        raise ValueError(
            f"feature stream supplied {processed} samples; expected {sample_count}"
        )
    if tail_count < 1:
        raise ValueError("no samples were available for late evidence fusion")

    denominator = np.maximum(support_weight_sum, 1e-6)
    aggregate = PairedWallOrientationResult(
        score=(score_sum / tail_count).astype(np.float32),
        paired_score=(paired_sum / tail_count).astype(np.float32),
        half_width_px=(width_weighted_sum / denominator).astype(np.float32),
        wall_balance=(balance_weighted_sum / denominator).astype(np.float32),
    )
    persistent_fraction = active_observations.astype(np.float32) / max(
        sample_count - config.warmup_samples,
        1,
    )
    return CausalBirthAtlas(
        aggregate=aggregate,
        birth_sample=birth,
        persistent_fraction=persistent_fraction,
        preexisting=preexisting,
        sample_count=sample_count,
    )


def track_pollen_centers(
    frames: np.ndarray,
    initial_centers_yx: np.ndarray,
    *,
    template_radius_px: int = 5,
    search_radius_px: int = 5,
    minimum_score: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow pollen bodies in stabilized frames using slowly updated templates."""

    images = np.asarray(frames, dtype=np.uint8)
    centers = np.asarray(initial_centers_yx, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError("frames must have shape (time, height, width)")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("initial centers must have shape (owners, 2)")
    if template_radius_px < 2 or search_radius_px < 1:
        raise ValueError("template and search radii are too small")

    trajectories = np.empty((len(centers), len(images), 2), dtype=np.float32)
    scores = np.ones((len(centers), len(images)), dtype=np.float32)
    templates = [
        _subpixel_patch(images[0], center, template_radius_px).astype(np.float32)
        for center in centers
    ]
    trajectories[:, 0] = centers
    current_centers = centers.copy()

    for sample in range(1, len(images)):
        for owner, (center, template) in enumerate(
            zip(current_centers, templates)
        ):
            search = _subpixel_patch(
                images[sample],
                center,
                template_radius_px + search_radius_px,
            ).astype(np.float32)
            response = cv.matchTemplate(search, template, cv.TM_CCOEFF_NORMED)
            _, score, _, location = cv.minMaxLoc(response)
            offset_xy = np.asarray(location, dtype=np.float64) - search_radius_px
            proposed = center + offset_xy[::-1]
            if np.isfinite(score) and score >= minimum_score:
                current_centers[owner] = proposed
            scores[owner, sample] = float(score)
            trajectories[owner, sample] = current_centers[owner]
            patch = _subpixel_patch(
                images[sample], current_centers[owner], template_radius_px
            ).astype(np.float32)
            templates[owner] = (0.92 * template + 0.08 * patch).astype(
                np.float32
            )
    return trajectories, scores


def track_pollen_centers_from_seeds(
    frames: np.ndarray,
    seed_centers_yx: np.ndarray,
    seed_samples: np.ndarray,
    *,
    template_radius_px: int = 5,
    search_radius_px: int = 5,
    minimum_score: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Track pollen both forward and backward from individual seed frames."""

    images = np.asarray(frames, dtype=np.uint8)
    centers = np.asarray(seed_centers_yx, dtype=np.float64)
    samples = np.asarray(seed_samples, dtype=np.int32)
    if images.ndim != 3:
        raise ValueError("frames must have shape (time, height, width)")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("seed centers must have shape (owners, 2)")
    if samples.shape != (len(centers),):
        raise ValueError("one seed sample is required for every owner")
    if np.any(samples < 0) or np.any(samples >= len(images)):
        raise ValueError("seed samples must lie within the frame sequence")

    trajectories = np.empty((len(centers), len(images), 2), dtype=np.float32)
    scores = np.empty((len(centers), len(images)), dtype=np.float32)
    for seed_sample in np.unique(samples):
        owners = np.flatnonzero(samples == seed_sample)
        forward, forward_scores = track_pollen_centers(
            images[seed_sample:],
            centers[owners],
            template_radius_px=template_radius_px,
            search_radius_px=search_radius_px,
            minimum_score=minimum_score,
        )
        backward, backward_scores = track_pollen_centers(
            images[: seed_sample + 1][::-1],
            centers[owners],
            template_radius_px=template_radius_px,
            search_radius_px=search_radius_px,
            minimum_score=minimum_score,
        )
        trajectories[owners, seed_sample:] = forward
        scores[owners, seed_sample:] = forward_scores
        trajectories[owners, : seed_sample + 1] = backward[:, ::-1]
        scores[owners, : seed_sample + 1] = backward_scores[:, ::-1]
    return trajectories, scores


def monotone_path_births(
    birth_samples: np.ndarray,
    *,
    unavailable_value: int,
) -> np.ndarray:
    """Enforce that material farther from a pollen cannot predate its prefix."""

    values = np.asarray(birth_samples, dtype=np.int32).copy()
    if values.ndim != 1:
        raise ValueError("birth samples must be one-dimensional")
    if not len(values):
        return values
    finite_indices = np.flatnonzero(values < unavailable_value)
    if not len(finite_indices):
        return values
    first = int(finite_indices[0])
    last_finite = int(finite_indices[-1])
    values[:first] = values[first]
    for left, right in zip(finite_indices[:-1], finite_indices[1:]):
        if right > left + 1:
            values[left + 1 : right] = values[right]
    values[: last_finite + 1] = np.maximum.accumulate(
        values[: last_finite + 1]
    )
    return values


def birth_order_score(
    birth_samples: np.ndarray,
    *,
    unavailable_value: int,
    tolerance_samples: int = 1,
) -> float:
    """Measure how consistently a candidate grows away from its pollen root."""

    values = np.asarray(birth_samples, dtype=np.float64)
    values = values[values < unavailable_value]
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.diff(values) >= -float(tolerance_samples)))


def _subpixel_patch(
    image: np.ndarray,
    center_yx: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Extract one reflected, subpixel-centered square patch."""

    padded = cv.copyMakeBorder(
        image,
        radius_px,
        radius_px,
        radius_px,
        radius_px,
        cv.BORDER_REFLECT101,
    )
    size = 2 * radius_px + 1
    return cv.getRectSubPix(
        padded,
        (size, size),
        (
            float(center_yx[1] + radius_px),
            float(center_yx[0] + radius_px),
        ),
    )


def _validate_config(
    config: CausalBirthAtlasConfig,
    sample_count: int,
) -> None:
    """Reject impossible temporal settings before allocating atlas arrays."""

    if not 1 <= config.warmup_samples < sample_count:
        raise ValueError("warmup_samples must be inside the timeline")
    if not 1 <= config.persistence_required <= config.persistence_window:
        raise ValueError("invalid persistence requirement")
    if config.tail_samples < 1:
        raise ValueError("tail_samples must be positive")
    if config.minimum_change < 0.0 or config.noise_multiplier < 0.0:
        raise ValueError("change thresholds cannot be negative")
    if not 0.0 <= config.preexisting_fraction <= 1.0:
        raise ValueError("preexisting_fraction must be between zero and one")


def _validate_feature(
    feature: PairedWallOrientationResult,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    """Ensure all ribbon feature arrays describe the same finite volume."""

    arrays = (
        np.asarray(feature.score),
        np.asarray(feature.paired_score),
        np.asarray(feature.half_width_px),
        np.asarray(feature.wall_balance),
    )
    shape = arrays[0].shape
    if len(shape) != 3 or any(array.shape != shape for array in arrays):
        raise ValueError("ribbon features must share (orientation, y, x) shape")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("ribbon feature shape changed within the stream")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("ribbon features must be finite")
