#!/usr/bin/env python3
"""Fit a globally deformable material-coordinate surface to one v20 audit.

V20 discovers a pollen-owned causal atlas and a monotone material front.  This
prototype keeps both fixed, then jointly estimates the normal displacement of
every active material coordinate at every sampled time.  Length therefore stays
in permanent material coordinates while the visible tube may bend or sway.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototypes.v20_orientation_worldsheet.track import (  # noqa: E402
    _curve_arclength,
    build_orientation_history,
    candidate_bounds,
    fixed_pollen_crops,
    foreign_grain_occupancy,
    load_candidate,
    path_orientation_profiles,
    reconstruct_phase,
    trace_timeline,
)
from tubetracker.deformable_worldsheet import (  # noqa: E402
    atlas_hypothesis_selection_key,
    atlas_prototype_score,
    DEFORMABLE_WORLDSHEET_REVISION,
    DeformableWorldsheetConfig,
    fit_deformable_worldsheet,
    material_continuity_certificate,
    open_curve_certificate,
    orientation_blobness,
    WORLDSHEET_PROMOTION_REVISION,
    worldsheet_promotion_decision,
)
from tubetracker.growth_front import visible_path_front  # noqa: E402
from tubetracker.orientation_worldsheet import (  # noqa: E402
    causal_orientation_changepoint,
    OrientationTraceConfig,
    paired_wall_orientation_features,
    persistent_orientation_birth,
    persistent_orientation_score,
    propose_pollen_roots,
    ribbon_path_certificate,
    trace_orientation_lifted,
)


def parse_args() -> argparse.Namespace:
    """Read the retained v18 candidate and v21 review settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-run", type=Path, required=True)
    parser.add_argument("--consensus-id", type=int, required=True)
    parser.add_argument("--evidence-samples", type=int, default=120)
    parser.add_argument("--output-samples", type=int, default=30)
    parser.add_argument("--crop-margin-px", type=int, default=70)
    parser.add_argument("--normal-radius-px", type=float, default=6.0)
    parser.add_argument("--independent-atlas", action="store_true")
    parser.add_argument("--independent-search-radius-px", type=float, default=110.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _selected_atlas(root, proposals, traces):
    """Recover the exact proposal trace selected internally by v20."""

    distances = [
        float(np.linalg.norm(proposal.root_yx - root.root_yx))
        + float(np.linalg.norm(proposal.direction_yx - root.direction_yx))
        for proposal in proposals
    ]
    return traces[int(np.argmin(distances))]


def _pollen_search_bounds(
    center_xy: np.ndarray,
    image_shape: tuple[int, int],
    radius_px: float,
) -> tuple[int, int, int, int]:
    """Choose a direction-neutral square search region around one pollen."""

    if radius_px <= 0.0:
        raise ValueError("independent search radius must be positive")
    height, width = image_shape
    center_x, center_y = np.asarray(center_xy, dtype=np.float64)
    return (
        max(0, int(math.floor(center_y - radius_px))),
        min(height, int(math.ceil(center_y + radius_px)) + 1),
        max(0, int(math.floor(center_x - radius_px))),
        min(width, int(math.ceil(center_x + radius_px)) + 1),
    )


ONSET_BASELINE_SAMPLES = 2
ROOT_HALO_OCCLUSION_PX = 6.0


def _material_scores(
    orientation_history: np.ndarray,
    baseline_samples: int = ONSET_BASELINE_SAMPLES,
) -> np.ndarray:
    """Emphasize material that changes relative to the early movie baseline."""

    scores = np.asarray(orientation_history, dtype=np.float32) / 255.0
    warmup = min(max(1, baseline_samples), len(scores) - 1)
    baseline = np.median(scores[:warmup], axis=0)
    change = np.maximum(scores - baseline[None, ...], 0.0)
    for time_index in range(len(change)):
        positive = change[time_index][change[time_index] > 0.0]
        if len(positive):
            scale = float(np.percentile(positive, 99.0))
            change[time_index] = np.clip(
                change[time_index] / max(scale, 1e-6),
                0.0,
                1.0,
            )
    return np.maximum(0.85 * change, 0.15 * scores)


def discover_video_wide_atlas(
    orientation_history: np.ndarray,
    material_scores: np.ndarray,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
    foreign_occupancy: np.ndarray,
    maximum_length_px: float,
    ribbon_reference,
):
    """Select a new or left-censored pollen-rim atlas through the whole video."""

    normalized = np.asarray(orientation_history, dtype=np.float32) / 255.0
    warmup = min(ONSET_BASELINE_SAMPLES, len(normalized) - 1)
    changepoint = causal_orientation_changepoint(
        normalized,
        minimum_before_samples=ONSET_BASELINE_SAMPLES,
        minimum_after_samples=3,
        minimum_change=0.04,
    )
    births = changepoint.birth_sample
    fused = changepoint.evidence
    occupied = np.asarray(foreign_occupancy, dtype=np.float32) >= 0.20
    fused[:, occupied] = 0.0
    births[:, occupied] = 0.0
    earliest = float(max(2, warmup - 1))
    trace_config = OrientationTraceConfig(
        maximum_length_px=maximum_length_px,
        minimum_length_px=max(18.0, 2.0 * pollen_radius_px),
        maximum_gap_steps=5,
        maximum_initial_gap_steps=5,
        root_occlusion_px=ROOT_HALO_OCCLUSION_PX,
        minimum_support=0.04,
        evidence_floor=0.08,
        curvature_penalty=0.18,
        beam_width=1600,
    )
    candidates = []
    late_start = max(warmup, int(round(0.60 * len(material_scores))))
    late_scores = material_scores[late_start:].copy()
    late_scores[:, :, occupied] = 0.0
    preexisting_map = np.percentile(
        np.max(normalized[:warmup], axis=1),
        75.0,
        axis=0,
    ).astype(np.float32)
    terminal_score = np.percentile(
        normalized[late_start:],
        75.0,
        axis=0,
    ).astype(np.float32)
    surface_config = DeformableWorldsheetConfig(
        normal_radius_px=4.0,
        pairwise_smoothness=0.30,
        maximum_cycles=8,
    )

    def record_candidate(
        proposal,
        trace,
        score_volume: np.ndarray,
        identity_birth: np.ndarray | None,
        temporal_scope: str,
        evidence_model: str,
        model_warmup_samples: int,
    ) -> None:
        """Fit and retain one root hypothesis under its temporal scope."""

        if trace.length_px < trace_config.minimum_length_px - 1.0:
            return
        surface = fit_deformable_worldsheet(
            score_volume,
            trace.path_yx,
            config=surface_config,
            orientation_birth=identity_birth,
        )
        mean_support = float(np.mean(surface.support))
        supported_fraction = float(np.mean(surface.support >= 0.12))
        displacement_p90 = float(
            np.percentile(np.abs(surface.displacements_px), 90)
        )
        sampled_preexisting = cv.remap(
            preexisting_map,
            trace.path_yx[:, 1][None, :].astype(np.float32),
            trace.path_yx[:, 0][None, :].astype(np.float32),
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )[0]
        preexisting_support = float(np.mean(sampled_preexisting))
        terminal_blobness = orientation_blobness(
            terminal_score,
            trace.path_yx[-1],
            max(4.0, 0.85 * pollen_radius_px),
        )
        score = atlas_prototype_score(
            trace.length_px,
            mean_support,
            supported_fraction,
            displacement_p90,
            (
                preexisting_support
                if temporal_scope == "causal-new-material"
                else 0.0
            ),
            terminal_blobness,
        )
        continuity = material_continuity_certificate(
            surface.support,
            surface.active,
            trace.path_yx,
            proximal_occlusion_px=ROOT_HALO_OCCLUSION_PX,
        )
        openness = open_curve_certificate(trace.path_yx)
        ribbon = ribbon_path_certificate(
            ribbon_reference,
            trace.path_yx,
            trace.direction_radians,
        )
        selection_key = atlas_hypothesis_selection_key(
            score,
            terminal_blobness,
            preexisting_support,
            continuity.proximal_eventual_support_fraction,
            continuity.maximum_unsupported_gap_px,
            endpoint_separation_fraction=openness.endpoint_separation_fraction,
        )
        candidates.append(
            {
                "proposal": proposal,
                "trace": trace,
                "score": score,
                "mean_support": mean_support,
                "supported_fraction": supported_fraction,
                "displacement_p90_px": displacement_p90,
                "preexisting_support": preexisting_support,
                "terminal_blobness": terminal_blobness,
                "proximal_eventual_support_fraction": (
                    continuity.proximal_eventual_support_fraction
                ),
                "eventual_supported_fraction": (
                    continuity.eventual_supported_fraction
                ),
                "maximum_unsupported_gap_px": (
                    continuity.maximum_unsupported_gap_px
                ),
                "endpoint_separation_px": openness.endpoint_separation_px,
                "endpoint_separation_fraction": (
                    openness.endpoint_separation_fraction
                ),
                "paired_wall_mean_support": ribbon.paired_mean_support,
                "paired_wall_supported_fraction": (
                    ribbon.paired_supported_fraction
                ),
                "mean_wall_balance": ribbon.mean_wall_balance,
                "median_half_width_px": ribbon.median_half_width_px,
                "width_mad_px": ribbon.width_mad_px,
                "maximum_width_step_px": ribbon.maximum_width_step_px,
                "attachment_eligible": bool(selection_key[0]),
                "temporal_scope": temporal_scope,
                "evidence_model": evidence_model,
                "identity_birth": identity_birth,
                "model_warmup_samples": model_warmup_samples,
            }
        )

    def trace_causal_family(
        family_fused: np.ndarray,
        family_births: np.ndarray,
        family_scores: np.ndarray,
        family_earliest: float,
        evidence_model: str,
        model_warmup_samples: int,
    ) -> None:
        """Generate one complete pollen-rim family from a causal evidence model."""

        proposals = propose_pollen_roots(
            family_fused,
            pollen_center_yx,
            pollen_radius_px,
            birth_time=family_births,
            minimum_material_birth=family_earliest,
            attachment_count=72,
            direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
            probe_distances_px=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
            probe_top_k=4,
            maximum_proposals=24,
            minimum_angle_separation_degrees=10.0,
        )
        for proposal in proposals:
            floor = family_earliest
            if (
                np.isfinite(proposal.birth_time)
                and proposal.birth_time < len(normalized) - 1
            ):
                floor = max(family_earliest, float(proposal.birth_time) - 2.0)
            trace = trace_orientation_lifted(
                family_fused,
                proposal.root_yx,
                proposal.direction_yx,
                birth_time=family_births,
                minimum_material_birth=floor,
                config=trace_config,
            )
            record_candidate(
                proposal,
                trace,
                family_scores,
                family_births,
                "causal-new-material",
                evidence_model,
                model_warmup_samples,
            )

    trace_causal_family(
        fused,
        births,
        late_scores,
        earliest,
        "full-timeline-changepoint",
        warmup,
    )
    persistence_warmup = max(4, min(len(normalized) // 10, len(normalized) - 1))
    persistence_births = persistent_orientation_birth(
        normalized,
        warmup_samples=persistence_warmup,
        persistence_samples=3,
        minimum_change=0.08,
    )
    persistence_fused = persistent_orientation_score(
        normalized,
        warmup_samples=persistence_warmup,
    )
    persistence_fused[:, occupied] = 0.0
    persistence_births[:, occupied] = 0.0
    persistence_scores = _material_scores(
        orientation_history,
        baseline_samples=persistence_warmup,
    )
    persistence_scores[:, :, occupied] = 0.0
    persistence_late_scores = persistence_scores[late_start:]
    trace_causal_family(
        persistence_fused,
        persistence_births,
        persistence_late_scores,
        float(max(2, persistence_warmup - 1)),
        "warmup-persistence",
        persistence_warmup,
    )
    if not candidates:
        raise RuntimeError("No independent pollen-rim atlas survived global fitting")
    causal_winner = max(
        candidates,
        key=lambda candidate: atlas_hypothesis_selection_key(
            candidate["score"],
            candidate["terminal_blobness"],
            candidate["preexisting_support"],
            candidate["proximal_eventual_support_fraction"],
            candidate["maximum_unsupported_gap_px"],
            endpoint_separation_fraction=candidate[
                "endpoint_separation_fraction"
            ],
        ),
    )
    if not causal_winner["attachment_eligible"] or causal_winner["score"] < 0.50:
        absolute_score = np.percentile(normalized, 70.0, axis=0).astype(np.float32)
        absolute_score[:, occupied] = 0.0
        absolute_proposals = propose_pollen_roots(
            absolute_score,
            pollen_center_yx,
            pollen_radius_px,
            attachment_count=72,
            direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
            probe_distances_px=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
            probe_top_k=4,
            maximum_proposals=48,
            minimum_angle_separation_degrees=7.5,
        )
        for proposal in absolute_proposals:
            trace = trace_orientation_lifted(
                absolute_score,
                proposal.root_yx,
                proposal.direction_yx,
                config=trace_config,
            )
            record_candidate(
                proposal,
                trace,
                normalized,
                None,
                "left-censored-preexisting",
                "absolute-preexisting",
                warmup,
            )
    def selection_key(candidate):
        return atlas_hypothesis_selection_key(
            candidate["score"],
            candidate["terminal_blobness"],
            candidate["preexisting_support"],
            candidate["proximal_eventual_support_fraction"],
            candidate["maximum_unsupported_gap_px"],
            endpoint_separation_fraction=candidate[
                "endpoint_separation_fraction"
            ],
        )

    winner = max(candidates, key=selection_key)
    report = [
        {
            "root_yx": candidate["proposal"].root_yx.tolist(),
            "direction_yx": candidate["proposal"].direction_yx.tolist(),
            "length_px": candidate["trace"].length_px,
            "mean_support": candidate["mean_support"],
            "supported_fraction": candidate["supported_fraction"],
            "displacement_p90_px": candidate["displacement_p90_px"],
            "preexisting_support": candidate["preexisting_support"],
            "terminal_blobness": candidate["terminal_blobness"],
            "proximal_eventual_support_fraction": candidate[
                "proximal_eventual_support_fraction"
            ],
            "eventual_supported_fraction": candidate[
                "eventual_supported_fraction"
            ],
            "maximum_unsupported_gap_px": candidate[
                "maximum_unsupported_gap_px"
            ],
            "endpoint_separation_px": candidate["endpoint_separation_px"],
            "endpoint_separation_fraction": candidate[
                "endpoint_separation_fraction"
            ],
            "paired_wall_mean_support": candidate[
                "paired_wall_mean_support"
            ],
            "paired_wall_supported_fraction": candidate[
                "paired_wall_supported_fraction"
            ],
            "mean_wall_balance": candidate["mean_wall_balance"],
            "median_half_width_px": candidate["median_half_width_px"],
            "width_mad_px": candidate["width_mad_px"],
            "maximum_width_step_px": candidate["maximum_width_step_px"],
            "attachment_eligible": candidate["attachment_eligible"],
            "temporal_scope": candidate["temporal_scope"],
            "evidence_model": candidate["evidence_model"],
            "model_warmup_samples": candidate["model_warmup_samples"],
            "global_prototype_score": candidate["score"],
        }
        for candidate in sorted(
            candidates,
            key=selection_key,
            reverse=True,
        )
    ]
    return (
        winner["trace"],
        report,
        winner["model_warmup_samples"],
        winner["identity_birth"],
        winner["temporal_scope"],
        winner["evidence_model"],
    )


def _draw_curve(image: np.ndarray, curve_yx: np.ndarray, color, thickness: int) -> None:
    """Draw one ordered row-column curve without obscuring its source image."""

    if len(curve_yx) < 2:
        return
    points = np.rint(curve_yx[:, ::-1]).astype(np.int32)
    cv.polylines(image, [points], False, color, thickness, cv.LINE_AA)
    cv.circle(image, tuple(points[-1]), thickness + 2, color, -1, cv.LINE_AA)


def write_review_video(
    output: Path,
    crops: np.ndarray,
    evidence_source_frames: np.ndarray,
    output_indices: np.ndarray,
    local_legacy_path: np.ndarray,
    atlas_path: np.ndarray,
    front_indices: np.ndarray,
    fitted,
    geometry_supported: bool,
    measurement_supported: bool,
    temporal_scope: str,
    pollen_center_yx: np.ndarray,
    pollen_radius_px: float,
) -> None:
    """Render candidates without presenting rejected geometry as a result."""

    height, width = crops.shape[1:]
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (2 * width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    for evidence_index in output_indices:
        raw = cv.cvtColor(crops[evidence_index], cv.COLOR_GRAY2BGR)
        overlay = raw.copy()
        cv.circle(
            overlay,
            tuple(np.rint(np.asarray(pollen_center_yx)[::-1]).astype(int)),
            max(2, int(round(pollen_radius_px))),
            (230, 100, 20),
            1,
            cv.LINE_AA,
        )
        _draw_curve(overlay, local_legacy_path, (255, 210, 0), 1)
        tip_index = int(front_indices[evidence_index])
        if tip_index >= 1:
            fixed = atlas_path[: tip_index + 1]
            deformed = fitted.curves_yx[evidence_index, : tip_index + 1]
            fixed_color = (70, 220, 70) if geometry_supported else (0, 165, 255)
            fit_color = (210, 60, 220) if geometry_supported else (0, 45, 220)
            _draw_curve(overlay, fixed, fixed_color, 1)
            _draw_curve(overlay, deformed, fit_color, 2)
            active_support = fitted.support[evidence_index, : tip_index + 1]
            state = f"support {float(np.mean(active_support)):.2f}"
        else:
            state = "pre-growth"
        if not geometry_supported:
            decision = "REJECTED HYPOTHESIS"
            legend = "blue ring pollen | cyan legacy | orange/red rejected"
        elif temporal_scope == "left-censored-preexisting":
            decision = (
                "PREEXISTING TUBE | ONSET UNKNOWN"
                if measurement_supported
                else "PREEXISTING GEOMETRY | REVIEW"
            )
            legend = "blue ring pollen | cyan legacy | green atlas | magenta fit"
        elif measurement_supported:
            decision = "MEASUREMENT ACCEPTED"
            legend = "blue ring pollen | cyan legacy | green atlas | magenta fit"
        else:
            decision = "GEOMETRY FOUND | REVIEW DATA"
            legend = "blue ring pollen | cyan legacy | green atlas | magenta fit"
        source_frame = int(evidence_source_frames[evidence_index])
        cv.putText(
            raw,
            f"raw source {source_frame}",
            (8, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            decision,
            (5, 14),
            cv.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            legend,
            (5, 27),
            cv.FONT_HERSHEY_SIMPLEX,
            0.27,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            state,
            (5, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.30,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        writer.write(np.hstack((raw, overlay)))
    writer.release()


def write_measurements(
    output: Path,
    output_indices: np.ndarray,
    evidence_source_frames: np.ndarray,
    atlas_path: np.ndarray,
    front_indices: np.ndarray,
    fixed_support: np.ndarray,
    fitted,
    bounds: tuple[int, int, int, int],
) -> dict[str, float | int | None]:
    """Export material length, deformed tip position, and support improvement."""

    y0, _, x0, _ = bounds
    arc = _curve_arclength(atlas_path)
    rows = []
    accepted_lengths = []
    accepted_tips = []
    support_changes = []
    active_count = 0
    for evidence_index in output_indices:
        source_frame = int(evidence_source_frames[evidence_index])
        tip_index = int(front_indices[evidence_index])
        if tip_index < 1:
            rows.append(
                {
                    "source_frame": source_frame,
                    "status": "pre-growth",
                    "material_length_px": "",
                    "tip_x_px": "",
                    "tip_y_px": "",
                    "fixed_mean_support": "",
                    "deformed_mean_support": "",
                    "support_change": "",
                    "median_normal_displacement_px": "",
                    "maximum_normal_displacement_px": "",
                }
            )
            continue
        active_slice = slice(0, tip_index + 1)
        active_count += 1
        deformed_support = fitted.support[evidence_index, active_slice]
        baseline_support = fixed_support[evidence_index, active_slice]
        displacement = fitted.displacements_px[evidence_index, active_slice]
        mean_support = float(np.mean(deformed_support))
        fixed_mean = float(np.mean(baseline_support))
        support_change = mean_support - fixed_mean
        accepted = mean_support >= 0.12 and np.mean(deformed_support >= 0.12) >= 0.65
        tip = fitted.curves_yx[evidence_index, tip_index]
        global_tip = tip + np.asarray((y0, x0), dtype=np.float64)
        length = float(arc[tip_index])
        if accepted:
            accepted_lengths.append(length)
            accepted_tips.append(global_tip)
            support_changes.append(support_change)
        rows.append(
            {
                "source_frame": source_frame,
                "status": "accepted" if accepted else "review",
                "material_length_px": round(length, 4),
                "tip_x_px": round(float(global_tip[1]), 4),
                "tip_y_px": round(float(global_tip[0]), 4),
                "fixed_mean_support": round(fixed_mean, 6),
                "deformed_mean_support": round(mean_support, 6),
                "support_change": round(support_change, 6),
                "median_normal_displacement_px": round(
                    float(np.median(np.abs(displacement))),
                    4,
                ),
                "maximum_normal_displacement_px": round(
                    float(np.max(np.abs(displacement))),
                    4,
                ),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    length_steps = np.diff(accepted_lengths)
    tip_steps = (
        np.linalg.norm(np.diff(np.asarray(accepted_tips), axis=0), axis=1)
        if len(accepted_tips) > 1
        else np.empty(0)
    )
    return {
        "accepted_measurement_count": len(accepted_lengths),
        "active_measurement_count": active_count,
        "accepted_frame_fraction": (
            len(accepted_lengths) / active_count if active_count else 0.0
        ),
        "median_support_change": float(np.median(support_changes))
        if support_changes
        else None,
        "length_decrease_count": int(np.sum(length_steps < -1e-6)),
        "maximum_tip_step_px": float(np.max(tip_steps)) if len(tip_steps) else None,
    }


def write_centerlines(
    output: Path,
    output_indices: np.ndarray,
    evidence_source_frames: np.ndarray,
    atlas_path: np.ndarray,
    front_indices: np.ndarray,
    fitted,
    bounds: tuple[int, int, int, int],
) -> None:
    """Export every point of each sampled deformable material centerline."""

    y0, _, x0, _ = bounds
    arc = _curve_arclength(atlas_path)
    rows = []
    for evidence_index in output_indices:
        source_frame = int(evidence_source_frames[evidence_index])
        tip_index = int(front_indices[evidence_index])
        if tip_index < 0:
            continue
        for path_index in range(tip_index + 1):
            point = fitted.curves_yx[evidence_index, path_index]
            rows.append(
                {
                    "source_frame": source_frame,
                    "path_index": path_index,
                    "material_position_px": round(float(arc[path_index]), 4),
                    "x_px": round(float(point[1] + x0), 4),
                    "y_px": round(float(point[0] + y0), 4),
                    "directional_support": round(
                        float(fitted.support[evidence_index, path_index]),
                        6,
                    ),
                    "normal_displacement_px": round(
                        float(fitted.displacements_px[evidence_index, path_index]),
                        4,
                    ),
                }
            )
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_alignment(candidate_yx: np.ndarray, reference_yx: np.ndarray) -> dict:
    """Measure root-relative geometric agreement without assuming equal sampling."""

    candidate = np.asarray(candidate_yx, dtype=np.float64)
    reference = np.asarray(reference_yx, dtype=np.float64)
    distances = np.linalg.norm(
        candidate[:, None, :] - reference[None, :, :],
        axis=2,
    )
    nearest = np.min(distances, axis=1)
    reverse = np.min(distances, axis=0)
    return {
        "candidate_to_reference_median_px": float(np.median(nearest)),
        "candidate_to_reference_p90_px": float(np.percentile(nearest, 90)),
        "symmetric_chamfer_px": float(0.5 * (np.mean(nearest) + np.mean(reverse))),
        "tip_distance_px": float(np.linalg.norm(candidate[-1] - reference[-1])),
    }


def main() -> None:
    """Build one real v21 audit from source video through review artifacts."""

    args = parse_args()
    report = json.loads((args.v18_run / "report.json").read_text(encoding="utf-8"))
    summary, path_yx = load_candidate(args.v18_run, args.consensus_id)
    movie, phase_frames, aligned, responses, grain, grains = reconstruct_phase(
        report,
        summary,
    )
    evidence_count = min(max(12, args.evidence_samples), len(phase_frames))
    evidence_indices = np.unique(
        np.rint(np.linspace(0, len(phase_frames) - 1, evidence_count)).astype(int)
    )
    bounds = (
        _pollen_search_bounds(
            grain.center_xy,
            aligned.shape[1:],
            args.independent_search_radius_px,
        )
        if args.independent_atlas
        else candidate_bounds(path_yx, aligned.shape[1:], args.crop_margin_px)
    )
    crops = fixed_pollen_crops(aligned, evidence_indices, grain, bounds)
    occupancy = foreign_grain_occupancy(grains, grain, evidence_indices, bounds)
    del aligned
    orientation_history = build_orientation_history(crops)
    ribbon_start = max(0, int(math.floor(0.80 * len(crops))))
    ribbon_image = np.median(crops[ribbon_start:], axis=0).astype(np.uint8)
    ribbon_reference = paired_wall_orientation_features(ribbon_image)
    output_count = min(max(2, args.output_samples), len(evidence_indices))
    output_indices = np.unique(
        np.rint(np.linspace(0, len(evidence_indices) - 1, output_count)).astype(int)
    )
    onset_values = [
        int(value)
        for value in (
            summary.get("source_root_onset_source_frame", ""),
            summary.get("source_rim_onset_source_frame", ""),
        )
        if value
    ]
    if not onset_values:
        raise ValueError("Candidate has no source-phase onset bound")
    evidence_source_frames = phase_frames[evidence_indices]
    (
        _,
        causal_births,
        local_legacy_path,
        _,
        _,
        selected_root,
        proposals,
        proposal_traces,
        local_center,
        global_front,
        _,
    ) = trace_timeline(
        orientation_history,
        evidence_source_frames,
        output_indices,
        path_yx,
        bounds,
        grain.center_xy[::-1],
        float(grain.radius_px),
        min(onset_values),
        occupancy,
        summary["consensus_measurement_supported"] == "True",
    )
    causal_atlas = _selected_atlas(selected_root, proposals, proposal_traces)
    material_scores = _material_scores(orientation_history)
    occupied = occupancy >= 0.20
    material_scores[:, :, occupied] = 0.0
    independent_candidates = []
    if args.independent_atlas:
        (
            atlas,
            independent_candidates,
            warmup,
            material_births,
            atlas_temporal_scope,
            atlas_evidence_model,
        ) = (
            discover_video_wide_atlas(
                orientation_history,
                material_scores,
                local_center,
                float(grain.radius_px),
                occupancy,
                maximum_length_px=max(
                    20.0,
                    args.independent_search_radius_px - 8.0,
                ),
                ribbon_reference=ribbon_reference,
            )
        )
        if atlas_temporal_scope == "left-censored-preexisting":
            front_indices = np.full(
                len(orientation_history),
                len(atlas.path_yx) - 1,
                dtype=int,
            )
            fit_scores = np.asarray(orientation_history, dtype=np.float32) / 255.0
            fit_scores[:, :, occupied] = 0.0
        else:
            if atlas_evidence_model == "warmup-persistence":
                fit_scores = _material_scores(
                    orientation_history,
                    baseline_samples=warmup,
                )
                fit_scores[:, :, occupied] = 0.0
            else:
                fit_scores = material_scores
            front_baseline_samples = min(
                ONSET_BASELINE_SAMPLES,
                len(orientation_history) - 1,
            )
            profiles = path_orientation_profiles(
                orientation_history,
                atlas.path_yx,
                atlas.direction_radians,
                front_baseline_samples,
            )
            global_front = visible_path_front(
                profiles,
                _curve_arclength(atlas.path_yx),
                warmup_samples=front_baseline_samples,
                max_step_px=6.0,
                coverage_cost=0.035,
                absence_weight=0.20,
                motion_penalty=0.12,
                support_window_samples=3,
                require_final_endpoint=True,
            )
            if not global_front.feasible:
                raise RuntimeError(
                    f"Independent atlas has no feasible material front: "
                    f"{global_front.reason}"
                )
            front_indices = np.clip(
                np.asarray(global_front.front_indices, dtype=int),
                -1,
                len(atlas.path_yx) - 1,
            )
    else:
        atlas = causal_atlas
        material_births = causal_births
        atlas_temporal_scope = "causal-new-material"
        atlas_evidence_model = "v20-causal-atlas"
        fit_scores = material_scores
        front_indices = np.clip(
            np.asarray(global_front.front_indices, dtype=int),
            -1,
            len(atlas.path_yx) - 1,
        )
    config = DeformableWorldsheetConfig(normal_radius_px=args.normal_radius_px)
    fitted = fit_deformable_worldsheet(
        fit_scores,
        atlas.path_yx,
        front_indices=front_indices,
        config=config,
        orientation_birth=material_births,
    )
    fixed = fit_deformable_worldsheet(
        fit_scores,
        atlas.path_yx,
        front_indices=front_indices,
        config=replace(config, normal_radius_px=0.0),
        orientation_birth=material_births,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "deformable_worldsheet_review.mp4"
    measurement_path = args.output_dir / "deformable_measurements.csv"
    dynamics = write_measurements(
        measurement_path,
        output_indices,
        evidence_source_frames,
        atlas.path_yx,
        front_indices,
        fixed.support,
        fitted,
        bounds,
    )
    centerline_path = args.output_dir / "deformable_centerlines.csv"
    write_centerlines(
        centerline_path,
        output_indices,
        evidence_source_frames,
        atlas.path_yx,
        front_indices,
        fitted,
        bounds,
    )
    active = fitted.active
    final_supported_fraction = float(np.mean(fitted.support[active] >= 0.12))
    continuity = material_continuity_certificate(
        fitted.support,
        fitted.active,
        atlas.path_yx,
        proximal_occlusion_px=ROOT_HALO_OCCLUSION_PX,
    )
    openness = open_curve_certificate(atlas.path_yx)
    if independent_candidates:
        selected_candidate = independent_candidates[0]
        terminal_preexisting_support = selected_candidate["preexisting_support"]
        terminal_foreign_body_risk = selected_candidate[
            "terminal_blobness"
        ] * math.sqrt(terminal_preexisting_support)
        geometry_supported, measurement_supported, decision_reasons = (
            worldsheet_promotion_decision(
                selected_candidate["global_prototype_score"],
                final_supported_fraction,
                selected_candidate["terminal_blobness"],
                dynamics["accepted_frame_fraction"],
                terminal_preexisting_support=terminal_preexisting_support,
                proximal_eventual_support_fraction=(
                    continuity.proximal_eventual_support_fraction
                ),
                maximum_unsupported_gap_px=(
                    continuity.maximum_unsupported_gap_px
                ),
                endpoint_separation_fraction=(
                    openness.endpoint_separation_fraction
                ),
            )
        )
    else:
        terminal_foreign_body_risk = 0.0
        geometry_supported, measurement_supported, decision_reasons = (
            worldsheet_promotion_decision(
                prototype_score=1e6,
                prototype_supported_fraction=final_supported_fraction,
                terminal_blobness=0.0,
                accepted_frame_fraction=dynamics["accepted_frame_fraction"],
                proximal_eventual_support_fraction=(
                    continuity.proximal_eventual_support_fraction
                ),
                maximum_unsupported_gap_px=(
                    continuity.maximum_unsupported_gap_px
                ),
                endpoint_separation_fraction=(
                    openness.endpoint_separation_fraction
                ),
            )
        )
    write_review_video(
        video_path,
        crops,
        evidence_source_frames,
        output_indices,
        local_legacy_path,
        atlas.path_yx,
        front_indices,
        fitted,
        geometry_supported,
        measurement_supported,
        atlas_temporal_scope,
        local_center,
        float(grain.radius_px),
    )
    result = {
        "prototype": "v21_deformable_orientation_worldsheet",
        "algorithm_revision": DEFORMABLE_WORLDSHEET_REVISION,
        "promotion_policy_revision": WORLDSHEET_PROMOTION_REVISION,
        "input_movie": str(movie),
        "v18_run": str(args.v18_run),
        "consensus_id": args.consensus_id,
        "v18_measurement_supported": summary["consensus_measurement_supported"],
        "v18_path_length_px": float(_curve_arclength(local_legacy_path)[-1]),
        "v20_atlas_length_px": float(_curve_arclength(causal_atlas.path_yx)[-1]),
        "selected_atlas_source": (
            "independent-video-wide-root-competition"
            if args.independent_atlas
            else "v20-causal-atlas"
        ),
        "atlas_temporal_scope": atlas_temporal_scope,
        "atlas_evidence_model": atlas_evidence_model,
        "front_baseline_samples": (
            front_baseline_samples
            if args.independent_atlas
            and atlas_temporal_scope == "causal-new-material"
            else None
        ),
        "germination_time_supported": bool(
            geometry_supported
            and atlas_temporal_scope == "causal-new-material"
        ),
        "selected_atlas_length_px": float(_curve_arclength(atlas.path_yx)[-1]),
        "pollen_center_yx": np.asarray(local_center, dtype=float).tolist(),
        "pollen_radius_px": float(grain.radius_px),
        "independent_atlas_candidates": independent_candidates,
        "atlas_geometry_supported": geometry_supported,
        "scientific_measurement_supported": measurement_supported,
        "decision_reasons": list(decision_reasons),
        "terminal_foreign_body_risk": terminal_foreign_body_risk,
        "legacy_alignment": _curve_alignment(atlas.path_yx, local_legacy_path),
        "v21_mean_active_support": float(np.mean(fitted.support[active])),
        "fixed_mean_active_support": float(np.mean(fixed.support[active])),
        "v21_supported_fraction": final_supported_fraction,
        "fixed_supported_fraction": float(np.mean(fixed.support[active] >= 0.12)),
        "material_continuity": {
            "proximal_eventual_support_fraction": (
                continuity.proximal_eventual_support_fraction
            ),
            "eventual_supported_fraction": continuity.eventual_supported_fraction,
            "maximum_unsupported_gap_px": (
                continuity.maximum_unsupported_gap_px
            ),
        },
        "open_curve": {
            "endpoint_separation_px": openness.endpoint_separation_px,
            "endpoint_separation_fraction": (
                openness.endpoint_separation_fraction
            ),
        },
        "median_absolute_displacement_px": float(
            np.median(np.abs(fitted.displacements_px[active]))
        ),
        "p90_absolute_displacement_px": float(
            np.percentile(np.abs(fitted.displacements_px[active]), 90)
        ),
        "maximum_absolute_displacement_px": float(
            np.max(np.abs(fitted.displacements_px[active]))
        ),
        "median_birth_identity_error_samples": (
            float(np.median(fitted.birth_identity_error_samples[active]))
            if fitted.birth_identity_error_samples is not None
            else None
        ),
        "p90_birth_identity_error_samples": (
            float(
                np.percentile(
                    fitted.birth_identity_error_samples[active],
                    90,
                )
            )
            if fitted.birth_identity_error_samples is not None
            else None
        ),
        "initial_graph_energy": fitted.initial_energy,
        "final_graph_energy": fitted.final_energy,
        "trajectory_dynamics": dynamics,
        "registration_response_median": float(np.median(responses)),
        "artifacts": {
            "review_video": str(video_path),
            "measurements_csv": str(measurement_path),
            "centerlines_csv": str(centerline_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
