"""Recover germination-to-final-growth timelines for accepted v23 tube paths.

Sparse owner-memory centerlines define tube identity and provide latest-arrival
confirmations. Dense raw-video samples determine when each connected path
prefix actually becomes visible. The topology cannot switch at a crossing:
every reported tip is the endpoint of one pollen-attached prefix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from tubetracker.growth_front import confirmation_bounded_path_front


@dataclass
class OwnerTimeline:
    """Hold one owner path, dense geometry, evidence, and fitted front."""

    track_id: int
    track: dict
    trace: dict
    arclength_px: np.ndarray
    aligned_paths_yx: np.ndarray
    world_paths_yx: np.ndarray
    confirmation_samples: np.ndarray
    earliest_samples: np.ndarray
    support_deadline_samples: np.ndarray
    confirmation_frames: np.ndarray
    structural_history: np.ndarray
    cross_sections: np.ndarray
    batch_accepted: bool
    batch_reason: str
    field_conflict: bool
    dense_centers_yx: np.ndarray | None = None
    dense_center_scores: np.ndarray | None = None
    rim_material_history: np.ndarray | None = None
    rim_extension_px: np.ndarray | None = None
    rim_direction_radians: np.ndarray | None = None
    rim_novelty: np.ndarray | None = None
    rim_emergence_sample: int | None = None
    rim_confirmation_sample: int | None = None
    durable_rim_emergence_sample: int | None = None
    profile: np.ndarray | None = None
    front: object | None = None
    emergence_profile: np.ndarray | None = None
    direct_front_indices: np.ndarray | None = None
    fitted_front_indices: np.ndarray | None = None
    emergence_sample: int | None = None
    emergence_confirmation_sample: int | None = None
    state_emergence_sample: int | None = None
    path_emergence_sample: int | None = None
    onset_evidence: str = "not-confirmed"
    accepted: bool = False


def parse_args() -> argparse.Namespace:
    """Parse field reports, temporal resolution, and evidence thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument(
        "--override-batch-report",
        type=Path,
        action="append",
        default=[],
        help="Replace matching owners with retraced paths from another batch.",
    )
    parser.add_argument("--consistency-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", help="Optional comma-separated owner IDs")
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Restrict analysis to paths accepted by the earlier sparse pass.",
    )
    parser.add_argument("--sample-interval-frames", type=int, default=210)
    parser.add_argument("--warmup-samples", type=int, default=8)
    parser.add_argument("--path-step-px", type=float, default=1.5)
    parser.add_argument("--maximum-growth-px-per-sample", type=float, default=8.0)
    parser.add_argument("--germination-length-px", type=float, default=1.5)
    parser.add_argument("--minimum-direct-support", type=float, default=0.45)
    parser.add_argument("--minimum-eventual-support", type=float, default=0.75)
    parser.add_argument(
        "--minimum-prior-supported-direct-support", type=float, default=0.30
    )
    parser.add_argument(
        "--minimum-prior-supported-eventual-support", type=float, default=0.75
    )
    parser.add_argument("--support-window-samples", type=int, default=4)
    parser.add_argument("--owner-radius-px", type=float, default=15.0)
    parser.add_argument(
        "--foreign-pollen-clearance-factor", type=float, default=1.30
    )
    parser.add_argument("--minimum-external-extension-px", type=float, default=15.0)
    parser.add_argument(
        "--minimum-unconfirmed-external-extension-px",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--minimum-onset-after-warmup-samples", type=int, default=2
    )
    parser.add_argument(
        "--minimum-confirmation-appearance", type=float, default=0.30
    )
    parser.add_argument(
        "--minimum-confirmation-causal-growth", type=float, default=0.45
    )
    parser.add_argument(
        "--emergence-direct-threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--emergence-persistence-window", type=int, default=5
    )
    parser.add_argument(
        "--emergence-persistence-required", type=int, default=3
    )
    parser.add_argument(
        "--emergence-followup-window", type=int, default=12
    )
    parser.add_argument(
        "--emergence-minimum-followup-growth-px", type=float, default=3.0
    )
    parser.add_argument("--dense-center-search-radius-px", type=int, default=7)
    parser.add_argument("--dense-center-template-radius-px", type=int, default=11)
    parser.add_argument("--dense-center-minimum-score", type=float, default=0.35)
    parser.add_argument("--rim-angle-count", type=int, default=72)
    parser.add_argument("--rim-search-length-px", type=float, default=30.0)
    parser.add_argument("--rim-direction-tolerance-degrees", type=float, default=45.0)
    parser.add_argument("--rim-minimum-emergence-px", type=float, default=2.0)
    parser.add_argument("--rim-minimum-directional-prominence-px", type=float, default=1.0)
    parser.add_argument("--rim-durability-window-samples", type=int, default=20)
    parser.add_argument("--rim-durability-required-fraction", type=float, default=0.75)
    parser.add_argument("--rim-durability-minimum-growth-px", type=float, default=6.0)
    parser.add_argument("--path-onset-agreement-samples", type=int, default=22)
    return parser.parse_args()


def curve_arclength(curve_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength for one row-column centerline."""
    curve = np.asarray(curve_yx, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
    )


def load_centerlines(path: Path) -> dict[int, np.ndarray]:
    """Load sparse selected centerlines grouped by source frame."""
    grouped: dict[int, list[tuple[int, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["source_frame"]), []).append(
                (
                    int(row["point_index"]),
                    float(row["y_aligned_px"]),
                    float(row["x_aligned_px"]),
                )
            )
    return {
        frame: np.asarray(
            [(y, x) for _, y, x in sorted(points)], dtype=np.float64
        )
        for frame, points in grouped.items()
    }


def interpolate_owner_centers(track: dict, source_frames: np.ndarray) -> np.ndarray:
    """Interpolate one persistent pollen center at dense source frames."""
    track_frames = np.asarray(track["source_frames"], dtype=np.float64)
    centers = np.asarray(track["centers_yx"], dtype=np.float64)
    return np.column_stack(
        [
            np.interp(source_frames, track_frames, centers[:, axis])
            for axis in range(2)
        ]
    )


def material_path_history(
    curves: dict[int, np.ndarray],
    source_frames: np.ndarray,
    step_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the same ordered material coordinate through sparse paths."""
    if not curves:
        raise ValueError("at least one selected centerline is required")
    if step_px <= 0:
        raise ValueError("path step must be positive")
    anchors = np.asarray(sorted(curves), dtype=np.int64)
    anchor_curves = [curves[int(frame)] for frame in anchors]
    anchor_arcs = [curve_arclength(curve) for curve in anchor_curves]
    anchor_lengths = np.asarray([arc[-1] for arc in anchor_arcs])
    final_length = float(np.max(anchor_lengths))
    arclength = np.arange(0.0, final_length + 1e-6, step_px)
    if final_length - arclength[-1] > 0.25 * step_px:
        arclength = np.append(arclength, final_length)
    aligned = np.zeros((len(source_frames), len(arclength), 2), dtype=np.float32)
    tolerance = 0.5 * step_px + 1e-6
    for point, distance in enumerate(arclength):
        available = np.flatnonzero(anchor_lengths + tolerance >= distance)
        if not len(available):
            available = np.asarray([int(np.argmax(anchor_lengths))])
        positions = []
        for index in available:
            curve = anchor_curves[int(index)]
            arc = anchor_arcs[int(index)]
            positions.append(
                [
                    np.interp(distance, arc, curve[:, axis])
                    for axis in range(2)
                ]
            )
        positions_array = np.asarray(positions)
        available_frames = anchors[available]
        for axis in range(2):
            aligned[:, point, axis] = np.interp(
                source_frames,
                available_frames,
                positions_array[:, axis],
            )
    return arclength, aligned, anchors


def confirmation_windows(
    source_frames: np.ndarray,
    confirmation_frames: np.ndarray,
    confirmation_lengths_px: np.ndarray,
    arclength_px: np.ndarray,
    warmup_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sparse path anchors into per-point earliest/latest samples."""
    anchor_samples = np.searchsorted(
        source_frames, confirmation_frames, side="left"
    )
    anchor_samples = np.clip(anchor_samples, 0, len(source_frames) - 1)
    confirmation_anchor = np.asarray(
        [
            np.flatnonzero(confirmation_lengths_px + 0.75 >= distance)[0]
            for distance in arclength_px
        ],
        dtype=np.int32,
    )
    confirmation = anchor_samples[confirmation_anchor].astype(np.int32)
    earliest = np.full(len(confirmation), warmup_samples, dtype=np.int32)
    later = confirmation_anchor > 0
    earliest[later] = anchor_samples[confirmation_anchor[later] - 1]
    deadline_anchor = np.minimum(
        confirmation_anchor + 1, len(anchor_samples) - 1
    )
    deadlines = anchor_samples[deadline_anchor].astype(np.int32)
    deadlines[confirmation_anchor == len(anchor_samples) - 1] = (
        len(source_frames) - 1
    )
    return confirmation, earliest, deadlines


def source_path_history(
    aligned_paths_yx: np.ndarray,
    track: dict,
    identity_source_frames: list[int],
    source_frames: np.ndarray,
    crop_bounds_yxyx: list[int],
    owner_centers_yx: np.ndarray | None = None,
) -> np.ndarray:
    """Transform owner-aligned material paths back into source coordinates."""
    identity_frames = np.asarray(identity_source_frames, dtype=np.float64)
    reference = np.median(
        interpolate_owner_centers(track, identity_frames), axis=0
    )
    centers = (
        interpolate_owner_centers(track, source_frames)
        if owner_centers_yx is None
        else np.asarray(owner_centers_yx, dtype=np.float64)
    )
    if centers.shape != (len(source_frames), 2):
        raise ValueError("owner centers must match the dense source timeline")
    y0, _, x0, _ = (int(value) for value in crop_bounds_yxyx)
    return (
        aligned_paths_yx
        + np.asarray((y0, x0), dtype=np.float32)
        - (reference[None, None, :] - centers[:, None, :])
    ).astype(np.float32)


def image_patch(
    gray: np.ndarray,
    center_yx: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Extract one subpixel-centered square patch with reflected borders."""
    if radius_px < 1:
        raise ValueError("patch radius must be positive")
    padded = cv.copyMakeBorder(
        gray,
        radius_px,
        radius_px,
        radius_px,
        radius_px,
        cv.BORDER_REFLECT101,
    )
    center_xy = (
        float(center_yx[1] + radius_px),
        float(center_yx[0] + radius_px),
    )
    size = 2 * radius_px + 1
    return cv.getRectSubPix(padded, (size, size), center_xy)


def match_owner_center(
    gray: np.ndarray,
    predicted_center_yx: np.ndarray,
    template: np.ndarray,
    *,
    search_radius_px: int,
    minimum_score: float,
) -> tuple[np.ndarray, float]:
    """Refine one interpolated pollen center against its recent appearance."""
    if template.ndim != 2 or template.shape[0] != template.shape[1]:
        raise ValueError("template must be one square grayscale patch")
    if search_radius_px < 0 or not -1.0 <= minimum_score <= 1.0:
        raise ValueError("invalid center-matching controls")
    template_radius = (template.shape[0] - 1) // 2
    search = image_patch(
        gray,
        predicted_center_yx,
        template_radius + search_radius_px,
    ).astype(np.float32)
    result = cv.matchTemplate(
        search,
        template.astype(np.float32),
        cv.TM_CCOEFF_NORMED,
    )
    _, score, _, location = cv.minMaxLoc(result)
    offset_xy = np.asarray(location, dtype=np.float64) - search_radius_px
    if not np.isfinite(score) or score < minimum_score:
        return np.asarray(predicted_center_yx, dtype=np.float64), float(score)
    return (
        np.asarray(predicted_center_yx, dtype=np.float64)
        + offset_xy[::-1],
        float(score),
    )


def refine_dense_owner_centers(
    capture: cv.VideoCapture,
    owners: list[OwnerTimeline],
    source_frames: np.ndarray,
    identity_source_frames: list[int],
    *,
    template_radius_px: int,
    search_radius_px: int,
    minimum_score: float,
) -> None:
    """Track every pollen body densely so edge motion cannot imitate a tube."""
    predictions = {
        owner.track_id: interpolate_owner_centers(owner.track, source_frames)
        for owner in owners
    }
    templates: dict[int, np.ndarray] = {}
    for owner in owners:
        owner.dense_centers_yx = predictions[owner.track_id].copy()
        owner.dense_center_scores = np.full(
            len(source_frames), np.nan, dtype=np.float32
        )

    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        for owner in owners:
            predicted = predictions[owner.track_id][sample]
            if sample == 0:
                center = predicted
                score = 1.0
            else:
                center, score = match_owner_center(
                    gray,
                    predicted,
                    templates[owner.track_id],
                    search_radius_px=search_radius_px,
                    minimum_score=minimum_score,
                )
            owner.dense_centers_yx[sample] = center
            owner.dense_center_scores[sample] = score
            current = image_patch(gray, center, template_radius_px).astype(
                np.float32
            )
            previous = templates.get(owner.track_id)
            templates[owner.track_id] = (
                current
                if previous is None
                else (0.85 * previous + 0.15 * current).astype(np.float32)
            )
        if sample == 0 or (sample + 1) % 50 == 0:
            print(
                f"[v23 pollen registration] {sample + 1}/{len(source_frames)} frames",
                flush=True,
            )

    for owner in owners:
        owner.world_paths_yx = source_path_history(
            owner.aligned_paths_yx,
            owner.track,
            identity_source_frames,
            source_frames,
            owner.trace["crop_bounds_yxyx"],
            owner.dense_centers_yx,
        )


def path_cross_sections(
    gray: np.ndarray,
    path_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure paired structure and grayscale cross-sections along one path."""
    path = np.asarray(path_yx, dtype=np.float32)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)
    offsets = np.arange(-8, 9, dtype=np.float32)

    intensity = []
    for offset in offsets:
        points = path + normal * offset
        map_x = np.ascontiguousarray(points[:, 1][None], dtype=np.float32)
        map_y = np.ascontiguousarray(points[:, 0][None], dtype=np.float32)
        intensity.append(
            cv.remap(
                gray,
                map_x,
                map_y,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT101,
            )[0]
        )
    intensity_values = np.stack(intensity, axis=1).astype(np.float32)

    lower = np.maximum(np.floor(path.min(axis=0) - 18).astype(int), 0)
    upper = np.minimum(
        np.ceil(path.max(axis=0) + 19).astype(int),
        np.asarray(gray.shape),
    )
    roi = gray[lower[0] : upper[0], lower[1] : upper[1]].astype(np.float32)
    fine = cv.GaussianBlur(roi, (0, 0), 0.8)
    background = cv.GaussianBlur(roi, (0, 0), 6.0)
    material = np.abs(fine - background)
    local_path = path - lower
    material_samples = []
    for offset in offsets:
        points = local_path + normal * offset
        map_x = np.ascontiguousarray(points[:, 1][None], dtype=np.float32)
        map_y = np.ascontiguousarray(points[:, 0][None], dtype=np.float32)
        material_samples.append(
            cv.remap(
                material,
                map_x,
                map_y,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT101,
            )[0]
        )
    material_values = np.stack(material_samples, axis=0)
    paired = []
    center_index = 8
    for center_offset in range(-2, 3):
        for half_width in range(2, 7):
            left = material_values[center_index + center_offset - half_width]
            right = material_values[center_index + center_offset + half_width]
            paired.append(np.sqrt(np.maximum(left * right, 0.0)))
    paired_response = np.max(np.stack(paired), axis=0)
    ridge_response = np.percentile(material_values, 90.0, axis=0)
    return np.maximum(paired_response, ridge_response), intensity_values


def polar_rim_material(
    material: np.ndarray,
    center_yx: np.ndarray,
    radii_px: np.ndarray,
    angles_radians: np.ndarray,
    *,
    normal_halfwidth_px: float = 4.0,
) -> np.ndarray:
    """Sample filament material along radial strips around one pollen body."""
    image = np.asarray(material, dtype=np.float32)
    center = np.asarray(center_yx, dtype=np.float32)
    radii = np.asarray(radii_px, dtype=np.float32)
    angles = np.asarray(angles_radians, dtype=np.float32)
    if image.ndim != 2 or center.shape != (2,):
        raise ValueError("material and center must describe one grayscale image")
    if radii.ndim != 1 or angles.ndim != 1 or not len(radii) or not len(angles):
        raise ValueError("polar sampling axes cannot be empty")
    normal_offsets = np.linspace(
        -normal_halfwidth_px,
        normal_halfwidth_px,
        7,
        dtype=np.float32,
    )
    radial = np.stack((np.sin(angles), np.cos(angles)), axis=1)
    normal = np.stack((np.cos(angles), -np.sin(angles)), axis=1)
    points = (
        center[None, None, None, :]
        + radial[:, None, None, :] * radii[None, :, None, None]
        + normal[:, None, None, :]
        * normal_offsets[None, None, :, None]
    )
    sampled = cv.remap(
        image,
        np.ascontiguousarray(
            points[..., 1].reshape(len(angles), -1), dtype=np.float32
        ),
        np.ascontiguousarray(
            points[..., 0].reshape(len(angles), -1), dtype=np.float32
        ),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT101,
    ).reshape(len(angles), len(radii), len(normal_offsets))
    return np.percentile(sampled, 75.0, axis=2).astype(np.float32)


def rim_extension_history(
    material_history: np.ndarray,
    radii_px: np.ndarray,
    expected_angles_radians: np.ndarray,
    *,
    owner_radius_px: float,
    warmup_samples: int,
    direction_tolerance_degrees: float,
    minimum_directional_prominence_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure the strongest new, direction-specific extension from the rim."""
    history = np.asarray(material_history, dtype=np.float32)
    radii = np.asarray(radii_px, dtype=np.float64)
    expected = np.asarray(expected_angles_radians, dtype=np.float64)
    if history.ndim != 3 or history.shape[2] != len(radii):
        raise ValueError("material_history must have time, angle, and radius axes")
    if expected.shape != (len(history),):
        raise ValueError("one expected rim direction is required per sample")
    flattened = history.reshape(len(history), -1)
    novelty = novelty_profile(flattened, warmup_samples).reshape(history.shape)
    novelty = median_filter(novelty, size=(3, 1, 1), mode="nearest")
    active = novelty >= 0.0
    angle_axis = np.linspace(0.0, 2.0 * np.pi, history.shape[1], endpoint=False)
    tolerance = np.deg2rad(direction_tolerance_degrees)
    extensions = np.zeros(len(history), dtype=np.float32)
    directions = np.full(len(history), np.nan, dtype=np.float32)
    prominence = np.zeros(len(history), dtype=np.float32)
    radial_extensions = radii - owner_radius_px

    for sample, frame_active in enumerate(active):
        angle_delta = np.abs(
            np.angle(np.exp(1j * (angle_axis - expected[sample])))
        )
        eligible = angle_delta <= tolerance
        per_angle = np.zeros(len(angle_axis), dtype=np.float32)
        for angle_index, radial_active in enumerate(frame_active):
            starts = np.flatnonzero(radial_active[: min(3, len(radii))])
            if not len(starts):
                continue
            point = int(starts[0])
            best = point
            while point < len(radial_active):
                if radial_active[point]:
                    best = point
                    point += 1
                    continue
                if point + 1 < len(radial_active) and radial_active[point + 1]:
                    best = point + 1
                    point += 2
                    continue
                break
            per_angle[angle_index] = max(0.0, radial_extensions[best])
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            continue
        winner = int(candidates[np.argmax(per_angle[candidates])])
        best_extension = float(per_angle[winner])
        background_extension = float(np.percentile(per_angle, 75.0))
        prominence[sample] = best_extension - background_extension
        if prominence[sample] < minimum_directional_prominence_px:
            continue
        extensions[sample] = best_extension
        directions[sample] = angle_axis[winner]
    return extensions, directions, novelty


def persistent_directional_emergence(
    extensions_px: np.ndarray,
    directions_radians: np.ndarray,
    *,
    warmup_samples: int,
    minimum_emergence_px: float,
    persistence_window: int,
    persistence_required: int,
    followup_window: int,
    minimum_followup_growth_px: float,
    maximum_direction_drift_degrees: float = 35.0,
) -> tuple[int | None, int | None]:
    """Confirm a persistent rim extension whose direction and length agree."""
    extensions = np.asarray(extensions_px, dtype=np.float64)
    directions = np.asarray(directions_radians, dtype=np.float64)
    active = (extensions >= minimum_emergence_px) & np.isfinite(directions)
    maximum_drift = np.deg2rad(maximum_direction_drift_degrees)
    for confirmation in range(warmup_samples, len(active)):
        if not active[confirmation]:
            continue
        lower = max(warmup_samples, confirmation - persistence_window + 1)
        for onset in range(lower, confirmation + 1):
            segment = active[onset : confirmation + 1]
            if not segment[0] or np.count_nonzero(segment) < persistence_required:
                continue
            if len(segment) >= 2 and np.any((~segment[:-1]) & (~segment[1:])):
                continue
            selected = directions[onset : confirmation + 1][segment]
            relative = np.angle(np.exp(1j * (selected - selected[0])))
            if np.max(np.abs(relative)) > maximum_drift:
                continue
            stop = min(len(extensions), onset + followup_window + 1)
            if (
                float(np.max(extensions[onset:stop]))
                < extensions[onset] + minimum_followup_growth_px
            ):
                continue
            return onset, confirmation
    return None, None


def durable_rim_emergence_sample(
    extensions_px: np.ndarray,
    *,
    warmup_samples: int,
    minimum_emergence_px: float,
    window_samples: int,
    required_fraction: float,
    minimum_growth_px: float,
) -> int | None:
    """Find the first rim extension that remains present and grows afterward."""
    extensions = np.asarray(extensions_px, dtype=np.float64)
    if window_samples < 2 or not 0.0 < required_fraction <= 1.0:
        raise ValueError("invalid rim durability controls")
    active = extensions >= minimum_emergence_px
    for onset in range(warmup_samples, len(extensions) - window_samples + 1):
        window = extensions[onset : onset + window_samples]
        if not active[onset] or np.mean(active[onset : onset + window_samples]) < required_fraction:
            continue
        first = float(np.median(window[: max(2, window_samples // 4)]))
        last = float(np.median(window[-max(2, window_samples // 4) :]))
        if max(float(np.max(window) - extensions[onset]), last - first) < minimum_growth_px:
            continue
        return onset
    return None


def select_emergence_consensus(
    path_sample: int | None,
    state_sample: int | None,
    durable_rim_sample: int | None,
    *,
    path_agreement_samples: int,
    timeline_samples: int | None = None,
) -> tuple[int | None, str]:
    """Choose onset from two agreeing cues, with guarded single-cue fallback."""
    if path_agreement_samples < 0:
        raise ValueError("path agreement tolerance cannot be negative")
    cues = {
        "path": path_sample,
        "state": state_sample,
        "rim": durable_rim_sample,
    }
    cue_items = list(cues.items())
    pairs = []
    for first_index, (first_name, first) in enumerate(cue_items):
        for second_name, second in cue_items[first_index + 1 :]:
            if first is None or second is None:
                continue
            difference = abs(int(first) - int(second))
            if difference <= path_agreement_samples:
                pairs.append((difference, first_name, second_name))
    if pairs:
        _, first_name, second_name = min(pairs)
        selected = [int(cues[first_name]), int(cues[second_name])]
        midpoint = float(np.mean(selected))
        selected.extend(
            int(value)
            for name, value in cues.items()
            if name not in {first_name, second_name}
            and value is not None
            and abs(int(value) - midpoint) <= path_agreement_samples
        )
        names = "-".join(sorted({first_name, second_name}))
        return int(round(float(np.median(selected)))), f"{names}-consensus"
    if state_sample is not None:
        if (
            durable_rim_sample is not None
            and timeline_samples is not None
            and state_sample >= 0.9 * timeline_samples
        ):
            return durable_rim_sample, "durable-rim-late-template-fallback"
        return state_sample, "persistent-owner-local-tube-state"
    if durable_rim_sample is not None:
        return durable_rim_sample, "durable-directional-rim-growth"
    return None, "not-confirmed"


def novelty_profile(values: np.ndarray, warmup_samples: int) -> np.ndarray:
    """Normalize one path signal against a robust per-position warmup."""
    baseline = np.median(values[:warmup_samples], axis=0)
    noise = 1.4826 * np.median(
        np.abs(values[:warmup_samples] - baseline), axis=0
    )
    change = np.maximum(1.0, 2.5 * noise)
    return np.tanh(
        (values - (baseline + change)) / np.maximum(change, 1.5)
    ).astype(np.float32)


def combine_path_evidence(owner: OwnerTimeline, warmup_samples: int) -> np.ndarray:
    """Fuse persistent paired walls with baseline-subtracted shape change."""
    structural = novelty_profile(owner.structural_history, warmup_samples)
    gradients = np.diff(owner.cross_sections, axis=2)
    baseline = np.median(gradients[:warmup_samples], axis=0)
    gradient_change = np.percentile(
        np.abs(gradients - baseline[None]), 75.0, axis=2
    )
    changed = novelty_profile(gradient_change, warmup_samples)
    profile = np.maximum(structural, changed - 0.15)
    profile[:, owner.arclength_px <= 12.0] = structural[
        :, owner.arclength_px <= 12.0
    ]
    return median_filter(profile, size=(3, 1), mode="nearest")


def cross_section_novelty(
    cross_sections: np.ndarray,
    warmup_samples: int,
) -> np.ndarray:
    """Measure new tube-shaped contrast independently at each path position."""
    values = np.asarray(cross_sections, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 5:
        raise ValueError("cross_sections must have time, path, and offset axes")
    if not 1 <= warmup_samples < len(values):
        raise ValueError("warmup_samples must fit inside cross_sections")
    centered = values - np.median(values, axis=2, keepdims=True)
    baseline_shape = np.median(centered[:warmup_samples], axis=0)
    shape_change = np.percentile(
        np.abs(centered - baseline_shape[None]), 75.0, axis=2
    )
    gradients = np.diff(values, axis=2)
    baseline_gradient = np.median(gradients[:warmup_samples], axis=0)
    gradient_change = np.percentile(
        np.abs(gradients - baseline_gradient[None]), 75.0, axis=2
    )
    change = np.maximum(shape_change, gradient_change)
    return median_filter(
        novelty_profile(change, warmup_samples),
        size=(3, 1),
        mode="nearest",
    )


def tube_state_profile(
    cross_sections: np.ndarray,
    warmup_samples: int,
    positive_samples: np.ndarray,
    *,
    positive_window_samples: int = 5,
    minimum_template_separation: float = 1.0,
) -> np.ndarray:
    """Classify each path position against its own early and tube templates."""
    values = np.asarray(cross_sections, dtype=np.float32)
    confirmations = np.asarray(positive_samples, dtype=np.int32)
    if values.ndim != 3 or confirmations.shape != (values.shape[1],):
        raise ValueError("one positive sample is required per path position")
    if not 1 <= warmup_samples < len(values):
        raise ValueError("warmup_samples must fit inside cross_sections")
    if positive_window_samples < 1 or minimum_template_separation < 0:
        raise ValueError("invalid tube-template controls")

    centered = values - np.median(values, axis=2, keepdims=True)
    gradients = np.diff(values, axis=2)
    early_shape = np.median(centered[:warmup_samples], axis=0)
    early_gradient = np.median(gradients[:warmup_samples], axis=0)
    tube_shape = np.zeros_like(early_shape)
    tube_gradient = np.zeros_like(early_gradient)
    for point, confirmation in enumerate(confirmations):
        start = int(np.clip(confirmation, warmup_samples, len(values) - 1))
        stop = min(len(values), start + positive_window_samples)
        tube_shape[point] = np.median(centered[start:stop, point], axis=0)
        tube_gradient[point] = np.median(gradients[start:stop, point], axis=0)

    early_distance = np.mean(
        np.abs(centered - early_shape[None]), axis=2
    )
    tube_distance = np.mean(
        np.abs(centered - tube_shape[None]), axis=2
    )
    early_gradient_distance = np.mean(
        np.abs(gradients - early_gradient[None]), axis=2
    )
    tube_gradient_distance = np.mean(
        np.abs(gradients - tube_gradient[None]), axis=2
    )
    early_distance += early_gradient_distance
    tube_distance += tube_gradient_distance
    separation = np.mean(
        np.abs(tube_shape - early_shape), axis=1
    ) + np.mean(np.abs(tube_gradient - early_gradient), axis=1)
    score = (early_distance - tube_distance) / np.maximum(separation[None], 1.0)
    score[:, separation < minimum_template_separation] = -1.0
    return np.clip(
        median_filter(score, size=(3, 1), mode="nearest"),
        -1.0,
        1.0,
    ).astype(np.float32)


def connected_direct_front_indices(
    profile: np.ndarray,
    *,
    threshold: float = 0.0,
    root_occlusion_points: int = 1,
) -> np.ndarray:
    """Return the directly visible prefix joined continuously to the pollen rim."""
    evidence = np.asarray(profile, dtype=np.float32)
    if evidence.ndim != 2:
        raise ValueError("profile must have time and path axes")
    if not 0 <= root_occlusion_points < evidence.shape[1]:
        raise ValueError("root_occlusion_points must fit inside the path")
    active = evidence >= threshold
    fronts = np.full(len(active), -1, dtype=np.int32)
    start = root_occlusion_points
    for sample, row in enumerate(active):
        point = start
        while point < len(row):
            if row[point]:
                fronts[sample] = point
                point += 1
                continue
            if point + 1 < len(row) and row[point + 1]:
                fronts[sample] = point + 1
                point += 2
                continue
            break
    return fronts


def persistent_emergence_sample(
    direct_front_indices: np.ndarray,
    arclength_px: np.ndarray,
    *,
    warmup_samples: int,
    minimum_length_px: float,
    persistence_window: int,
    persistence_required: int,
    followup_window: int,
    minimum_followup_growth_px: float,
) -> tuple[int | None, int | None]:
    """Find the first persistent rim-connected protrusion followed by growth."""
    fronts = np.asarray(direct_front_indices, dtype=np.int32)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    if fronts.ndim != 1 or arclength.ndim != 1:
        raise ValueError("front indices and arclength must be one-dimensional")
    if persistence_window < 1 or not 1 <= persistence_required <= persistence_window:
        raise ValueError("persistence requirement must fit inside its window")
    if followup_window < 1 or minimum_followup_growth_px < 0:
        raise ValueError("follow-up controls must be nonnegative")
    lengths = np.zeros(len(fronts), dtype=np.float64)
    visible = fronts >= 0
    lengths[visible] = arclength[fronts[visible]]
    active = lengths >= minimum_length_px
    for confirmation in range(warmup_samples, len(active)):
        if not active[confirmation]:
            continue
        lower = max(warmup_samples, confirmation - persistence_window + 1)
        onset = None
        for candidate in range(lower, confirmation + 1):
            segment = active[candidate : confirmation + 1]
            if not segment[0] or np.count_nonzero(segment) < persistence_required:
                continue
            if len(segment) >= 2 and np.any((~segment[:-1]) & (~segment[1:])):
                continue
            onset = candidate
            break
        if onset is None:
            continue
        stop = min(len(lengths), onset + followup_window + 1)
        if (
            float(np.max(lengths[onset:stop]))
            < lengths[onset] + minimum_followup_growth_px
        ):
            continue
        return onset, confirmation
    return None, None


def apply_emergence_gate(
    front_indices: np.ndarray,
    direct_front_indices: np.ndarray,
    arclength_px: np.ndarray,
    emergence_sample: int | None,
    maximum_growth_px_per_sample: float,
    emergence_seed_index: int | None = None,
) -> np.ndarray:
    """Suppress inferred pre-emergence length and seed growth from direct evidence."""
    fitted = np.full_like(np.asarray(front_indices, dtype=np.int32), -1)
    if emergence_sample is None:
        return fitted
    direct = np.asarray(direct_front_indices, dtype=np.int32)
    arclength = np.asarray(arclength_px, dtype=np.float64)
    previous = -1
    for sample in range(emergence_sample, len(fitted)):
        proposed = int(front_indices[sample])
        observed = int(direct[sample])
        if sample == emergence_sample:
            candidate = max(
                observed,
                -1 if emergence_seed_index is None else emergence_seed_index,
            )
        else:
            candidate = max(proposed, observed, previous)
        if candidate < 0:
            continue
        if previous < 0:
            candidate = min(
                candidate,
                int(
                    np.searchsorted(
                        arclength,
                        maximum_growth_px_per_sample,
                        side="right",
                    )
                    - 1
                ),
            )
        else:
            maximum_length = (
                arclength[previous] + maximum_growth_px_per_sample
            )
            candidate = min(
                candidate,
                int(np.searchsorted(arclength, maximum_length, side="right") - 1),
            )
        fitted[sample] = candidate
        previous = candidate
    return fitted


def eligible_owner_entries(
    batch: dict,
    requested_ids: set[int] | None,
    accepted_only: bool,
) -> list[dict]:
    """Return all available owner paths selected for dense analysis."""
    return [
        entry
        for entry in batch.get("owners", [])
        if (entry.get("accepted") or not accepted_only)
        and (
            requested_ids is None
            or int(entry["owner_track_id"]) in requested_ids
        )
    ]


def fitted_indices(owner: OwnerTimeline) -> np.ndarray | None:
    """Return emergence-gated indices, with legacy front fallback for audits."""
    gated = getattr(owner, "fitted_front_indices", None)
    if gated is not None:
        return np.asarray(gated, dtype=np.int32)
    front = getattr(owner, "front", None)
    if front is None or not front.feasible:
        return None
    return np.asarray(front.front_indices, dtype=np.int32)


def temporal_duplicate_claims(
    owners: list[OwnerTimeline],
    *,
    root_exclusion_px: float = 12.0,
    distance_px: float = 6.0,
    minimum_overlap: float = 0.65,
) -> tuple[set[int], list[dict]]:
    """Resolve owners that claim substantially the same fitted tube path."""
    curves: dict[int, np.ndarray] = {}
    owner_by_id = {owner.track_id: owner for owner in owners}
    for owner in owners:
        indices = fitted_indices(owner)
        if indices is None:
            continue
        final_index = int(indices[-1])
        if final_index < 1:
            continue
        path = owner.world_paths_yx[-1, : final_index + 1]
        arc = curve_arclength(path)
        distal = path[arc >= root_exclusion_px]
        if len(distal) >= 2:
            curves[owner.track_id] = distal

    parent = {track_id: track_id for track_id in curves}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    claims = []
    track_ids = sorted(curves)
    for index, first_id in enumerate(track_ids):
        first = curves[first_id]
        first_tree = cKDTree(first)
        for second_id in track_ids[index + 1 :]:
            second = curves[second_id]
            second_tree = cKDTree(second)
            first_overlap = float(
                np.mean(second_tree.query(first, k=1)[0] <= distance_px)
            )
            second_overlap = float(
                np.mean(first_tree.query(second, k=1)[0] <= distance_px)
            )
            if min(first_overlap, second_overlap) < minimum_overlap:
                continue
            union(first_id, second_id)
            claims.append(
                {
                    "first_owner": first_id,
                    "second_owner": second_id,
                    "first_overlap_fraction": first_overlap,
                    "second_overlap_fraction": second_overlap,
                }
            )

    components: dict[int, list[int]] = {}
    for track_id in track_ids:
        components.setdefault(find(track_id), []).append(track_id)
    losers: set[int] = set()
    for component in components.values():
        if len(component) < 2:
            continue
        winner = max(
            component,
            key=lambda track_id: (
                owner_by_id[track_id].batch_accepted,
                owner_by_id[track_id].front.direct_support_fraction
                + owner_by_id[track_id].front.eventual_support_fraction,
                owner_by_id[track_id].front.direct_support_fraction,
                -track_id,
            ),
        )
        losers.update(set(component) - {winner})
        for claim in claims:
            if claim["first_owner"] in component:
                claim["component_winner"] = winner
    return losers, claims


def consistency_duplicate_losers(
    owners: list[OwnerTimeline],
    duplicate_claims: list[dict],
) -> tuple[set[int], list[dict]]:
    """Resolve duplicate components recorded by the independent field audit."""
    owner_by_id = {owner.track_id: owner for owner in owners}
    parent = {track_id: track_id for track_id in owner_by_id}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    for claim in duplicate_claims:
        first = int(claim["first_owner"])
        second = int(claim["second_owner"])
        if first not in parent or second not in parent:
            continue
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    components: dict[int, list[int]] = {}
    for track_id in owner_by_id:
        components.setdefault(find(track_id), []).append(track_id)
    losers: set[int] = set()
    resolutions = []
    for component in components.values():
        if len(component) < 2:
            continue
        winner = max(
            component,
            key=lambda track_id: (
                owner_by_id[track_id].batch_accepted,
                owner_by_id[track_id].front.direct_support_fraction
                + owner_by_id[track_id].front.eventual_support_fraction,
                owner_by_id[track_id].front.direct_support_fraction,
                -track_id,
            ),
        )
        component_losers = sorted(set(component) - {winner})
        losers.update(component_losers)
        resolutions.append(
            {
                "owner_ids": sorted(component),
                "winner_owner_id": winner,
                "loser_owner_ids": component_losers,
            }
        )
    return losers, resolutions


def temporal_foreign_pollen_contacts(
    owners: list[OwnerTimeline],
    source_frame: int,
    *,
    owner_radius_px: float,
    clearance_factor: float = 1.30,
    root_exclusion_px: float = 12.0,
) -> tuple[set[int], list[dict]]:
    """Find fitted tubes whose distal path enters another pollen body."""
    centers = {
        owner.track_id: (
            np.asarray(owner.dense_centers_yx[-1], dtype=np.float64)
            if getattr(owner, "dense_centers_yx", None) is not None
            else interpolate_owner_centers(
                owner.track, np.asarray([source_frame])
            )[0]
        )
        for owner in owners
    }
    contacted: set[int] = set()
    contacts = []
    for owner in owners:
        indices = fitted_indices(owner)
        if indices is None:
            continue
        final_index = int(indices[-1])
        if final_index < 1:
            continue
        path = owner.world_paths_yx[-1, : final_index + 1]
        arc = curve_arclength(path)
        distal = path[arc >= root_exclusion_px]
        if not len(distal):
            continue
        for other_id, center in centers.items():
            if other_id == owner.track_id:
                continue
            distance = float(np.min(np.linalg.norm(distal - center, axis=1)))
            if distance >= clearance_factor * owner_radius_px:
                continue
            contacted.add(owner.track_id)
            contacts.append(
                {
                    "owner_track_id": owner.track_id,
                    "foreign_owner_track_id": other_id,
                    "minimum_distance_px": distance,
                }
            )
    return contacted, contacts


def main() -> None:
    """Fit and export dense, pollen-attached growth timelines."""
    args = parse_args()
    if args.sample_interval_frames <= 0:
        raise ValueError("sample interval must be positive")
    if args.owner_radius_px <= 0:
        raise ValueError("owner radius must be positive")
    if args.foreign_pollen_clearance_factor <= 1.0:
        raise ValueError("foreign-pollen clearance factor must exceed one")
    if (
        args.dense_center_search_radius_px < 0
        or args.dense_center_template_radius_px < 2
        or not -1.0 <= args.dense_center_minimum_score <= 1.0
    ):
        raise ValueError("invalid dense pollen-center matching controls")
    if (
        args.rim_angle_count < 24
        or args.rim_search_length_px <= 3.0
        or not 0.0 < args.rim_direction_tolerance_degrees <= 180.0
        or args.rim_minimum_emergence_px <= 0.0
    ):
        raise ValueError("invalid pollen-rim emergence controls")
    if (
        args.rim_durability_window_samples < 2
        or not 0.0 < args.rim_durability_required_fraction <= 1.0
        or args.rim_durability_minimum_growth_px < 0.0
        or args.path_onset_agreement_samples < 0
    ):
        raise ValueError("invalid emergence-consensus controls")
    if (
        args.emergence_persistence_window < 1
        or not 1
        <= args.emergence_persistence_required
        <= args.emergence_persistence_window
    ):
        raise ValueError("emergence persistence requirement must fit its window")
    if (
        args.minimum_unconfirmed_external_extension_px
        < args.minimum_external_extension_px
    ):
        raise ValueError(
            "unconfirmed extension threshold cannot be below the base threshold"
        )
    identity = json.loads(args.identity_report.read_text())
    batch = json.loads(args.batch_report.read_text())
    entry_roots = {
        int(entry["owner_track_id"]): args.batch_report.parent
        for entry in batch.get("owners", [])
    }
    entries_by_id = {
        int(entry["owner_track_id"]): entry
        for entry in batch.get("owners", [])
    }
    for override_path in args.override_batch_report:
        override = json.loads(override_path.read_text())
        for entry in override.get("owners", []):
            track_id = int(entry["owner_track_id"])
            entries_by_id[track_id] = entry
            entry_roots[track_id] = override_path.parent
    batch["owners"] = list(entries_by_id.values())
    consistency = (
        json.loads(args.consistency_report.read_text())
        if args.consistency_report
        else {}
    )
    requested = (
        {int(value) for value in args.track_ids.split(",")}
        if args.track_ids
        else None
    )
    entries = eligible_owner_entries(batch, requested, args.accepted_only)
    if not entries:
        raise RuntimeError("no owner paths were selected")
    conflict_ids = set(consistency.get("duplicate_claim_owner_ids", []))
    conflict_ids.update(consistency.get("foreign_endpoint_owner_ids", []))

    capture = cv.VideoCapture(identity["movie"])
    if not capture.isOpened():
        raise RuntimeError(f"could not open {identity['movie']}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv.CAP_PROP_FPS))
    source_frames = np.unique(
        np.append(
            np.arange(0, frame_count, args.sample_interval_frames),
            frame_count - 1,
        )
    ).astype(np.int64)
    if not 1 <= args.warmup_samples < len(source_frames):
        raise ValueError("warmup samples must fit inside the dense timeline")
    rim_radii_px = np.arange(
        args.owner_radius_px + 1.0,
        args.owner_radius_px + args.rim_search_length_px + 1e-6,
        args.path_step_px,
        dtype=np.float32,
    )
    rim_angles_radians = np.linspace(
        0.0,
        2.0 * np.pi,
        args.rim_angle_count,
        endpoint=False,
        dtype=np.float32,
    )

    tracks = {int(track["track_id"]): track for track in identity["tracks"]}
    owners: list[OwnerTimeline] = []
    unresolved_reports: list[dict] = []
    for entry in entries:
        track_id = int(entry["owner_track_id"])
        trace_path = entry_roots[track_id] / entry["directory"] / "report.json"
        if not trace_path.exists():
            unresolved_reports.append(
                {
                    "owner_track_id": track_id,
                    "trajectory_accepted": False,
                    "front_feasible": False,
                    "reason": "missing-sparse-trace-report",
                    "germination_source_frame": None,
                    "batch_accepted": bool(entry.get("accepted")),
                    "batch_reason": entry.get("reason"),
                    "field_conflict": track_id in conflict_ids,
                }
            )
            continue
        trace = json.loads(trace_path.read_text())
        curves = load_centerlines(
            trace_path.parent / trace["artifacts"]["centerlines"]
        )
        valid_observations = [
            observation
            for observation in trace["selected_observations"]
            if observation["appearance_score"]
            >= args.minimum_confirmation_appearance
            and observation["causal_growth_score"]
            >= args.minimum_confirmation_causal_growth
        ]
        if not valid_observations:
            unresolved_reports.append(
                {
                    "owner_track_id": track_id,
                    "trajectory_accepted": False,
                    "front_feasible": False,
                    "reason": "no-evidence-backed-path-confirmation",
                    "germination_source_frame": None,
                    "batch_accepted": bool(entry.get("accepted")),
                    "batch_reason": entry.get("reason"),
                    "field_conflict": track_id in conflict_ids,
                }
            )
            continue
        arclength, aligned, _ = material_path_history(
            curves, source_frames, args.path_step_px
        )
        confirmation_frames = np.asarray(
            [item["source_frame"] for item in valid_observations],
            dtype=np.int64,
        )
        confirmation_lengths = np.asarray(
            [item["length_px"] for item in valid_observations],
            dtype=np.float64,
        )
        if np.max(confirmation_lengths) + 0.75 < arclength[-1]:
            raise RuntimeError(
                f"owner {track_id} final path lacks an evidence-backed confirmation"
            )
        confirmation, earliest, support_deadlines = confirmation_windows(
            source_frames,
            confirmation_frames,
            confirmation_lengths,
            arclength,
            args.warmup_samples,
        )
        world = source_path_history(
            aligned,
            tracks[track_id],
            identity["source_frames"],
            source_frames,
            trace["crop_bounds_yxyx"],
        )
        owners.append(
            OwnerTimeline(
                track_id=track_id,
                track=tracks[track_id],
                trace=trace,
                arclength_px=arclength,
                aligned_paths_yx=aligned,
                world_paths_yx=world,
                confirmation_samples=confirmation,
                earliest_samples=earliest,
                support_deadline_samples=support_deadlines,
                confirmation_frames=confirmation_frames,
                structural_history=np.zeros(
                    (len(source_frames), len(arclength)), dtype=np.float32
                ),
                cross_sections=np.zeros(
                    (len(source_frames), len(arclength), 17), dtype=np.float32
                ),
                batch_accepted=bool(entry.get("accepted")),
                batch_reason=str(entry.get("reason", "unknown")),
                field_conflict=track_id in conflict_ids,
                rim_material_history=np.zeros(
                    (
                        len(source_frames),
                        len(rim_angles_radians),
                        len(rim_radii_px),
                    ),
                    dtype=np.float32,
                ),
            )
        )

    refine_dense_owner_centers(
        capture,
        owners,
        source_frames,
        identity["source_frames"],
        template_radius_px=args.dense_center_template_radius_px,
        search_radius_px=args.dense_center_search_radius_px,
        minimum_score=args.dense_center_minimum_score,
    )

    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fine = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
        background = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 5.0)
        material = np.abs(fine - background)
        for owner in owners:
            structure, cross = path_cross_sections(
                gray, owner.world_paths_yx[sample]
            )
            owner.structural_history[sample] = structure
            owner.cross_sections[sample] = cross
            owner.rim_material_history[sample] = polar_rim_material(
                material,
                owner.dense_centers_yx[sample],
                rim_radii_px,
                rim_angles_radians,
            )
        if sample == 0 or (sample + 1) % 25 == 0:
            print(
                f"[v23 temporal] {sample + 1}/{len(source_frames)} frames",
                flush=True,
            )
    capture.release()

    args.output.mkdir(parents=True, exist_ok=True)
    measurement_rows = []
    centerline_rows = []
    owner_reports = []
    profile_arrays: dict[str, np.ndarray] = {"source_frames": source_frames}
    for owner in owners:
        direction_point = min(
            len(owner.arclength_px) - 1,
            int(np.searchsorted(owner.arclength_px, 8.0, side="left")),
        )
        direction_vector = (
            owner.world_paths_yx[:, direction_point]
            - owner.dense_centers_yx
        )
        expected_rim_angles = np.arctan2(
            direction_vector[:, 0], direction_vector[:, 1]
        )
        (
            owner.rim_extension_px,
            owner.rim_direction_radians,
            owner.rim_novelty,
        ) = rim_extension_history(
            owner.rim_material_history,
            rim_radii_px,
            expected_rim_angles,
            owner_radius_px=args.owner_radius_px,
            warmup_samples=args.warmup_samples,
            direction_tolerance_degrees=args.rim_direction_tolerance_degrees,
            minimum_directional_prominence_px=(
                args.rim_minimum_directional_prominence_px
            ),
        )
        (
            owner.rim_emergence_sample,
            owner.rim_confirmation_sample,
        ) = persistent_directional_emergence(
            owner.rim_extension_px,
            owner.rim_direction_radians,
            warmup_samples=args.warmup_samples,
            minimum_emergence_px=args.rim_minimum_emergence_px,
            persistence_window=args.emergence_persistence_window,
            persistence_required=args.emergence_persistence_required,
            followup_window=args.emergence_followup_window,
            minimum_followup_growth_px=(
                args.emergence_minimum_followup_growth_px
            ),
        )
        owner.profile = combine_path_evidence(owner, args.warmup_samples)
        owner.front = confirmation_bounded_path_front(
            owner.profile,
            owner.arclength_px,
            owner.confirmation_samples,
            owner.earliest_samples,
            warmup_samples=args.warmup_samples,
            max_step_px=args.maximum_growth_px_per_sample,
            support_window_samples=args.support_window_samples,
            support_deadline_samples=owner.support_deadline_samples,
        )
        owner.emergence_profile = tube_state_profile(
            owner.cross_sections,
            args.warmup_samples,
            owner.confirmation_samples,
        )
        owner.direct_front_indices = connected_direct_front_indices(
            owner.emergence_profile,
            threshold=args.emergence_direct_threshold,
        )
        (
            owner.state_emergence_sample,
            state_confirmation_sample,
        ) = persistent_emergence_sample(
            owner.direct_front_indices,
            owner.arclength_px,
            warmup_samples=args.warmup_samples,
            minimum_length_px=args.germination_length_px,
            persistence_window=args.emergence_persistence_window,
            persistence_required=args.emergence_persistence_required,
            followup_window=args.emergence_followup_window,
            minimum_followup_growth_px=(
                args.emergence_minimum_followup_growth_px
            ),
        )
        owner.durable_rim_emergence_sample = durable_rim_emergence_sample(
            owner.rim_extension_px,
            warmup_samples=args.warmup_samples,
            minimum_emergence_px=args.rim_minimum_emergence_px,
            window_samples=args.rim_durability_window_samples,
            required_fraction=args.rim_durability_required_fraction,
            minimum_growth_px=args.rim_durability_minimum_growth_px,
        )
        old_front_active = owner.front.front_indices >= 0
        old_front_lengths = np.zeros(len(source_frames), dtype=np.float64)
        old_front_lengths[old_front_active] = owner.arclength_px[
            owner.front.front_indices[old_front_active]
        ]
        old_onsets = np.flatnonzero(
            old_front_lengths >= args.germination_length_px
        )
        old_onset_sample = int(old_onsets[0]) if len(old_onsets) else None
        owner.path_emergence_sample = old_onset_sample
        owner.emergence_sample, owner.onset_evidence = select_emergence_consensus(
            old_onset_sample,
            owner.state_emergence_sample,
            owner.durable_rim_emergence_sample,
            path_agreement_samples=args.path_onset_agreement_samples,
            timeline_samples=len(source_frames),
        )
        if "state" not in owner.onset_evidence:
            owner.emergence_confirmation_sample = min(
                len(source_frames) - 1,
                owner.emergence_sample + args.rim_durability_window_samples - 1,
            )
        else:
            owner.emergence_confirmation_sample = state_confirmation_sample
        seed_index = None
        if owner.emergence_sample is not None:
            rim_length = float(owner.rim_extension_px[owner.emergence_sample])
            if rim_length >= args.rim_minimum_emergence_px:
                seed_index = int(
                    np.searchsorted(
                        owner.arclength_px,
                        min(rim_length, args.maximum_growth_px_per_sample),
                        side="right",
                    )
                    - 1
                )
        owner.fitted_front_indices = apply_emergence_gate(
            owner.front.front_indices,
            owner.direct_front_indices,
            owner.arclength_px,
            owner.emergence_sample,
            args.maximum_growth_px_per_sample,
            seed_index,
        )
    duplicate_losers, duplicate_claims = temporal_duplicate_claims(owners)
    field_duplicate_losers, field_duplicate_resolutions = (
        consistency_duplicate_losers(
            owners,
            consistency.get("duplicate_path_claims", []),
        )
    )
    duplicate_losers.update(field_duplicate_losers)
    foreign_contact_ids, foreign_contacts = temporal_foreign_pollen_contacts(
        owners,
        int(source_frames[-1]),
        owner_radius_px=args.owner_radius_px,
        clearance_factor=args.foreign_pollen_clearance_factor,
    )

    for owner in owners:
        lengths = np.zeros(len(source_frames), dtype=np.float64)
        if owner.front.feasible and owner.fitted_front_indices is not None:
            active = owner.fitted_front_indices >= 0
            lengths[active] = owner.arclength_px[
                owner.fitted_front_indices[active]
            ]
        germination_samples = np.flatnonzero(
            lengths >= args.germination_length_px
        )
        germination_sample = (
            int(germination_samples[0]) if len(germination_samples) else None
        )
        final_center = owner.dense_centers_yx[-1]
        final_extension = float(
            np.linalg.norm(owner.world_paths_yx[-1, -1] - final_center)
            - args.owner_radius_px
        )
        if not owner.front.feasible:
            trajectory_reason = owner.front.reason
        elif owner.emergence_sample is None:
            trajectory_reason = "no-confirmed-emergence"
        elif germination_sample is None:
            trajectory_reason = "no-measurable-germination"
        elif (
            germination_sample
            < args.warmup_samples + args.minimum_onset_after_warmup_samples
        ):
            trajectory_reason = "left-censored-at-warmup"
        elif final_extension < args.minimum_external_extension_px:
            trajectory_reason = "insufficient-external-extension"
        elif (
            final_extension < args.minimum_unconfirmed_external_extension_px
            and not owner.batch_accepted
        ):
            trajectory_reason = "short-unconfirmed-growth"
        elif (
            owner.front.direct_support_fraction
            < (
                args.minimum_prior_supported_direct_support
                if owner.batch_accepted
                else args.minimum_direct_support
            )
            or owner.front.eventual_support_fraction
            < (
                args.minimum_prior_supported_eventual_support
                if owner.batch_accepted
                else args.minimum_eventual_support
            )
        ):
            trajectory_reason = "insufficient-temporal-path-support"
        elif owner.track_id in foreign_contact_ids:
            trajectory_reason = "foreign-pollen-intersection"
        elif owner.track_id in duplicate_losers:
            trajectory_reason = "competing-owner-claim"
        else:
            trajectory_reason = "accepted"
        owner.accepted = trajectory_reason == "accepted"
        first_confirmation = int(owner.confirmation_frames[0])
        germination_frame = (
            int(source_frames[germination_sample])
            if germination_sample is not None
            else None
        )
        owner_reports.append(
            {
                "owner_track_id": owner.track_id,
                "trajectory_accepted": owner.accepted,
                "front_feasible": bool(owner.front.feasible),
                "reason": trajectory_reason,
                "batch_accepted": owner.batch_accepted,
                "batch_reason": owner.batch_reason,
                "field_conflict": owner.field_conflict,
                "temporal_duplicate_loser": (
                    owner.track_id in duplicate_losers
                ),
                "foreign_pollen_contact": (
                    owner.track_id in foreign_contact_ids
                ),
                "germination_source_frame": germination_frame,
                "emergence_confirmation_source_frame": (
                    int(source_frames[owner.emergence_confirmation_sample])
                    if owner.emergence_confirmation_sample is not None
                    else None
                ),
                "rim_emergence_source_frame": (
                    int(source_frames[owner.rim_emergence_sample])
                    if owner.rim_emergence_sample is not None
                    else None
                ),
                "rim_confirmation_source_frame": (
                    int(source_frames[owner.rim_confirmation_sample])
                    if owner.rim_confirmation_sample is not None
                    else None
                ),
                "emergence_evidence": (
                    owner.onset_evidence
                ),
                "tube_state_emergence_source_frame": (
                    int(source_frames[owner.state_emergence_sample])
                    if owner.state_emergence_sample is not None
                    else None
                ),
                "path_front_emergence_source_frame": (
                    int(source_frames[owner.path_emergence_sample])
                    if owner.path_emergence_sample is not None
                    else None
                ),
                "durable_rim_emergence_source_frame": (
                    int(source_frames[owner.durable_rim_emergence_sample])
                    if owner.durable_rim_emergence_sample is not None
                    else None
                ),
                "germination_playback_time_s": (
                    germination_frame / fps
                    if germination_frame is not None and fps > 0
                    else None
                ),
                "first_sparse_confirmation_source_frame": first_confirmation,
                "recovered_before_first_confirmation_s": (
                    (first_confirmation - germination_frame) / fps
                    if germination_frame is not None and fps > 0
                    else None
                ),
                "final_length_px": float(lengths[-1]),
                "final_external_extension_px": final_extension,
                "direct_support_fraction": owner.front.direct_support_fraction,
                "eventual_support_fraction": owner.front.eventual_support_fraction,
                "maximum_confirmation_lag_samples": (
                    owner.front.max_confirmation_lag_samples
                ),
                "valid_sparse_confirmation_frames": (
                    owner.confirmation_frames.tolist()
                ),
                "dense_center_median_match_score": float(
                    np.nanmedian(owner.dense_center_scores)
                ),
                "dense_center_fallback_fraction": float(
                    np.mean(
                        owner.dense_center_scores
                        < args.dense_center_minimum_score
                    )
                ),
            }
        )
        profile_arrays[f"P{owner.track_id:03d}_arclength_px"] = owner.arclength_px
        profile_arrays[f"P{owner.track_id:03d}_profile"] = owner.profile
        profile_arrays[f"P{owner.track_id:03d}_emergence_profile"] = (
            owner.emergence_profile
        )
        profile_arrays[f"P{owner.track_id:03d}_direct_front_index"] = (
            owner.direct_front_indices
        )
        profile_arrays[f"P{owner.track_id:03d}_rim_extension_px"] = (
            owner.rim_extension_px
        )
        profile_arrays[f"P{owner.track_id:03d}_rim_direction_radians"] = (
            owner.rim_direction_radians
        )
        profile_arrays[f"P{owner.track_id:03d}_length_px"] = lengths

        for sample, source_frame in enumerate(source_frames):
            length = float(lengths[sample])
            point = (
                int(owner.fitted_front_indices[sample])
                if owner.front.feasible
                and owner.fitted_front_indices is not None
                else -1
            )
            germinated = length >= args.germination_length_px and point >= 0
            tip_y = tip_x = ""
            tip_evidence = "not-applicable"
            if germinated:
                tip = owner.world_paths_yx[sample, point]
                tip_y, tip_x = float(tip[0]), float(tip[1])
                if owner.profile[sample, point] >= 0.0:
                    tip_evidence = "direct"
                elif owner.front.eventual_support_mask[point]:
                    tip_evidence = "later-supported"
                else:
                    tip_evidence = "anchor-constrained"
                for point_index in range(point + 1):
                    coordinate = owner.world_paths_yx[sample, point_index]
                    centerline_rows.append(
                        {
                            "owner_track_id": owner.track_id,
                            "source_frame": int(source_frame),
                            "point_index": point_index,
                            "arclength_px": float(
                                owner.arclength_px[point_index]
                            ),
                            "y_source_px": float(coordinate[0]),
                            "x_source_px": float(coordinate[1]),
                        }
                    )
            measurement_rows.append(
                {
                    "owner_track_id": owner.track_id,
                    "sample_index": sample,
                    "source_frame": int(source_frame),
                    "playback_time_s": (
                        float(source_frame) / fps if fps > 0 else ""
                    ),
                    "length_px": length if germinated else 0.0,
                    "tip_y_source_px": tip_y,
                    "tip_x_source_px": tip_x,
                    "measurement_status": (
                        "pre-germination"
                        if not germinated
                        else "accepted"
                        if owner.accepted
                        else "review-only"
                    ),
                    "tip_evidence": tip_evidence,
                }
            )

    measurement_fields = tuple(measurement_rows[0])
    with (args.output / "temporal_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=measurement_fields)
        writer.writeheader()
        writer.writerows(measurement_rows)
    centerline_fields = (
        "owner_track_id",
        "source_frame",
        "point_index",
        "arclength_px",
        "y_source_px",
        "x_source_px",
    )
    with (args.output / "temporal_centerlines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=centerline_fields)
        writer.writeheader()
        writer.writerows(centerline_rows)
    np.savez_compressed(args.output / "temporal_profiles.npz", **profile_arrays)
    owner_reports.extend(unresolved_reports)
    owner_reports.sort(key=lambda owner: int(owner["owner_track_id"]))
    report = {
        "prototype": "v23_owner_conditioned_video_graph",
        "stage": "multi-cue-emergence-and-temporal-front",
        "identity_report": str(args.identity_report),
        "batch_report": str(args.batch_report),
        "override_batch_reports": [
            str(path) for path in args.override_batch_report
        ],
        "source_frame_count": frame_count,
        "source_fps": fps,
        "sample_interval_frames": args.sample_interval_frames,
        "sample_interval_playback_s": args.sample_interval_frames / fps,
        "germination_timing_resolution_s": args.sample_interval_frames / fps,
        "sample_count": len(source_frames),
        "warmup_samples": args.warmup_samples,
        "support_window_samples": args.support_window_samples,
        "owner_radius_px": args.owner_radius_px,
        "foreign_pollen_clearance_factor": (
            args.foreign_pollen_clearance_factor
        ),
        "minimum_external_extension_px": args.minimum_external_extension_px,
        "minimum_unconfirmed_external_extension_px": (
            args.minimum_unconfirmed_external_extension_px
        ),
        "minimum_confirmation_appearance": (
            args.minimum_confirmation_appearance
        ),
        "minimum_confirmation_causal_growth": (
            args.minimum_confirmation_causal_growth
        ),
        "emergence_direct_threshold": args.emergence_direct_threshold,
        "emergence_persistence_window": args.emergence_persistence_window,
        "emergence_persistence_required": args.emergence_persistence_required,
        "emergence_followup_window": args.emergence_followup_window,
        "emergence_minimum_followup_growth_px": (
            args.emergence_minimum_followup_growth_px
        ),
        "dense_center_search_radius_px": args.dense_center_search_radius_px,
        "dense_center_template_radius_px": args.dense_center_template_radius_px,
        "dense_center_minimum_score": args.dense_center_minimum_score,
        "rim_angle_count": args.rim_angle_count,
        "rim_search_length_px": args.rim_search_length_px,
        "rim_direction_tolerance_degrees": (
            args.rim_direction_tolerance_degrees
        ),
        "rim_minimum_emergence_px": args.rim_minimum_emergence_px,
        "rim_minimum_directional_prominence_px": (
            args.rim_minimum_directional_prominence_px
        ),
        "rim_durability_window_samples": args.rim_durability_window_samples,
        "rim_durability_required_fraction": (
            args.rim_durability_required_fraction
        ),
        "rim_durability_minimum_growth_px": (
            args.rim_durability_minimum_growth_px
        ),
        "path_onset_agreement_samples": args.path_onset_agreement_samples,
        "minimum_direct_support": args.minimum_direct_support,
        "minimum_eventual_support": args.minimum_eventual_support,
        "minimum_prior_supported_direct_support": (
            args.minimum_prior_supported_direct_support
        ),
        "minimum_prior_supported_eventual_support": (
            args.minimum_prior_supported_eventual_support
        ),
        "germination_length_px": args.germination_length_px,
        "owner_count": len(entries),
        "candidate_path_count": len(owners),
        "unresolved_path_count": len(unresolved_reports),
        "accepted_trajectory_count": sum(owner.accepted for owner in owners),
        "temporal_duplicate_claims": duplicate_claims,
        "temporal_duplicate_loser_ids": sorted(duplicate_losers),
        "field_duplicate_resolutions": field_duplicate_resolutions,
        "temporal_foreign_pollen_contacts": foreign_contacts,
        "temporal_foreign_pollen_contact_ids": sorted(foreign_contact_ids),
        "time_scope": (
            "Encoded-video playback time only; experimental frame interval "
            "must be supplied before biological time analysis."
        ),
        "owners": owner_reports,
        "artifacts": {
            "measurements": "temporal_measurements.csv",
            "centerlines": "temporal_centerlines.csv",
            "profiles": "temporal_profiles.npz",
        },
    }
    (args.output / "temporal_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
