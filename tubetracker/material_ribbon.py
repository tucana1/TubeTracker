"""Observe paired pollen-tube walls around an ordered material centerline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class MaterialRibbonConfig:
    """Configure local paired-wall detection around a known tube identity."""

    center_search_radius_px: int = 3
    minimum_half_width_px: float = 1.5
    maximum_half_width_px: float = 5.5
    half_width_step_px: float = 0.5
    wall_sigma_px: float = 0.7
    background_sigma_px: float = 3.0
    center_continuity_penalty: float = 0.10
    width_continuity_penalty: float = 0.08
    width_prior_penalty: float = 0.06
    center_response_penalty: float = 0.20
    minimum_pair_response: float = 0.08
    width_update_blend: float = 0.25
    root_lock_points: int = 3


@dataclass(frozen=True)
class MaterialRibbonObservation:
    """Store a centerline observation with ordered left and right tube walls."""

    centerline_yx: np.ndarray
    left_boundary_yx: np.ndarray
    right_boundary_yx: np.ndarray
    widths_px: np.ndarray
    confidence: np.ndarray

    @property
    def supported_fraction(self):
        """Return the fraction of nodes with a supported paired-wall response."""
        if not len(self.confidence):
            return 0.0
        return float(np.mean(self.confidence > 0.0))


def _curve_normals(curve):
    """Return stable unit normals for an ordered row-column curve."""
    tangents = np.gradient(np.asarray(curve, dtype=np.float64), axis=0)
    tangents /= np.maximum(
        np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6
    )
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def _dark_wall_response(gray, config):
    """Enhance locally dark tube walls while suppressing illumination drift."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray must be a two-dimensional image")
    fine = cv.GaussianBlur(image, (0, 0), config.wall_sigma_px)
    background = cv.GaussianBlur(image, (0, 0), config.background_sigma_px)
    response = np.maximum(background - fine, 0.0)
    positive = response[response > 0.0]
    if not len(positive):
        return np.zeros_like(response)
    scale = float(np.percentile(positive, 98.0))
    if scale <= 1e-6:
        return np.zeros_like(response)
    return np.clip(response / scale, 0.0, 1.0)


def _sample_image(image, points_yx, border_value=0.0):
    """Bilinearly sample an image at an arbitrary array of row-column points."""
    points = np.asarray(points_yx, dtype=np.float32)
    return cv.remap(
        np.asarray(image, dtype=np.float32),
        points[..., 1],
        points[..., 0],
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def _ribbon_states(config):
    """Enumerate center-offset and half-width states for ribbon fitting."""
    offsets = np.arange(
        -int(config.center_search_radius_px),
        int(config.center_search_radius_px) + 1,
        dtype=np.float64,
    )
    half_widths = np.arange(
        float(config.minimum_half_width_px),
        float(config.maximum_half_width_px) + 0.5 * config.half_width_step_px,
        float(config.half_width_step_px),
        dtype=np.float64,
    )
    return np.asarray(
        [(offset, half_width) for offset in offsets for half_width in half_widths],
        dtype=np.float64,
    )


def observe_material_ribbon(
    gray,
    centerline_yx,
    prior_widths_px=None,
    exclusion=None,
    config=None,
):
    """Fit an ordered pair of dark tube walls around a known material curve."""
    config = config or MaterialRibbonConfig()
    curve = np.asarray(centerline_yx, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError("centerline_yx must have shape (points, 2)")
    response = _dark_wall_response(gray, config)
    if exclusion is None:
        exclusion = np.zeros(response.shape, dtype=bool)
    else:
        exclusion = np.asarray(exclusion, dtype=bool)
        if exclusion.shape != response.shape:
            raise ValueError("exclusion must match gray")
    if prior_widths_px is not None:
        prior_widths = np.asarray(prior_widths_px, dtype=np.float64)
        if prior_widths.shape != (len(curve),):
            raise ValueError("prior_widths_px must match the centerline")
    else:
        prior_widths = None

    normals = _curve_normals(curve)
    states = _ribbon_states(config)
    center_offsets = states[:, 0]
    half_widths = states[:, 1]
    centers = curve[:, None, :] + center_offsets[None, :, None] * normals[:, None, :]
    left = centers - half_widths[None, :, None] * normals[:, None, :]
    right = centers + half_widths[None, :, None] * normals[:, None, :]
    pair_response = 0.5 * (
        _sample_image(response, left) + _sample_image(response, right)
    )
    center_response = _sample_image(response, centers)
    evidence = pair_response - config.center_response_penalty * center_response
    excluded = (
        _sample_image(exclusion.astype(np.float32), left, border_value=1.0) > 0.5
    ) | (
        _sample_image(exclusion.astype(np.float32), right, border_value=1.0) > 0.5
    )
    evidence[excluded] = -np.inf
    locked = min(max(1, int(config.root_lock_points)), len(curve))
    evidence[:locked, center_offsets != 0.0] = -np.inf
    if prior_widths is not None:
        evidence -= config.width_prior_penalty * np.abs(
            2.0 * half_widths[None, :] - prior_widths[:, None]
        )

    state_count = len(states)
    scores = np.full((len(curve), state_count), -np.inf, dtype=np.float64)
    back = np.full((len(curve), state_count), -1, dtype=int)
    scores[0] = evidence[0]
    center_delta = np.abs(center_offsets[:, None] - center_offsets[None, :])
    width_delta = np.abs(half_widths[:, None] - half_widths[None, :])
    transition = (
        config.center_continuity_penalty * center_delta
        + config.width_continuity_penalty * width_delta
    )
    for point_index in range(1, len(curve)):
        candidate_scores = scores[point_index - 1][:, None] - transition
        previous = np.argmax(candidate_scores, axis=0)
        best = candidate_scores[previous, np.arange(state_count)]
        scores[point_index] = evidence[point_index] + best
        back[point_index] = previous
    state = int(np.argmax(scores[-1]))
    if not np.isfinite(scores[-1, state]):
        zeros = np.zeros(len(curve), dtype=np.float64)
        return MaterialRibbonObservation(
            centerline_yx=curve.copy(),
            left_boundary_yx=curve.copy(),
            right_boundary_yx=curve.copy(),
            widths_px=zeros,
            confidence=zeros,
        )
    selected = np.empty(len(curve), dtype=int)
    for point_index in range(len(curve) - 1, -1, -1):
        selected[point_index] = state
        state = back[point_index, state]
    selected_offsets = center_offsets[selected]
    selected_half_widths = half_widths[selected]
    selected_centers = curve + selected_offsets[:, None] * normals
    selected_left = selected_centers - selected_half_widths[:, None] * normals
    selected_right = selected_centers + selected_half_widths[:, None] * normals
    selected_response = pair_response[np.arange(len(curve)), selected]
    confidence = np.clip(
        (selected_response - config.minimum_pair_response)
        / max(1.0 - config.minimum_pair_response, 1e-6),
        0.0,
        1.0,
    )
    confidence[:locked] = 0.0
    return MaterialRibbonObservation(
        centerline_yx=selected_centers,
        left_boundary_yx=selected_left,
        right_boundary_yx=selected_right,
        widths_px=2.0 * selected_half_widths,
        confidence=confidence,
    )
