"""Track low-density pollen tubes from verified boundary emergence.

This prototype motion-corrects every pollen, builds a late paired-wall atlas in
that pollen's coordinate system, and traces several possible open curves.  It
then replays the complete movie and accepts only a curve whose optical support
appears as a connected, root-to-tip prefix.  Static pollen rims and tubes that
arrive from elsewhere therefore cannot become measurements merely because
they resemble a tube in the final frame.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import pickle
import subprocess

import cv2 as cv
import numpy as np

from prototypes.v24_causal_birth_forest.track import (
    load_grayscale_samples,
    movie_schedule,
    stabilize_translations,
)
from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds
from tubetracker.causal_portal import (
    CausalPortalConfig,
    CausalPortalResult,
    PhaseContrastRibbonConfig,
    PollenOutlineConfig,
    PollenRingConfig,
    assess_pollen_ring,
    measure_pollen_outline,
    native_connected_prefix_growth,
    path_exits_owner_once,
    phase_contrast_ribbon_features,
    sample_native_paired_support,
    truncate_self_reentry,
    validate_causal_portal,
)
from tubetracker.orientation_worldsheet import (
    CoupledRibbonTraceConfig,
    OrientationScoreConfig,
    PairedWallOrientationResult,
    paired_wall_orientation_features,
    propose_pollen_roots,
    trace_coupled_ribbon_lifted,
)


CAUSAL_EVIDENCE_REVISION = "v25.5-native-authoritative-germination"
MULTISCALE_EVIDENCE_REVISION = "v25.13-multiscale-candidate-recovery"
CAUSAL_PORTAL_REVISION = "v25.15-discontinuity-aware-growth"


@dataclass
class PortalCandidate:
    """Store one late geometry hypothesis and its full temporal evidence."""

    owner_index: int
    owner_track_id: int
    observable_start_sample: int
    appearance_mode: str
    local_center_yx: np.ndarray
    local_path_yx: np.ndarray
    direction_bins: np.ndarray
    normal_yx: np.ndarray
    geometric_score: float
    paired_fraction: float
    radial_extension_px: float
    evidence_by_mode: dict[str, np.ndarray]
    normal_offsets_by_mode: dict[str, np.ndarray]
    temporal_mode: str | None = None
    normal_offsets_px: np.ndarray | None = None
    validation: CausalPortalResult | None = None
    accepted: bool = False
    reason: str = "not-validated"
    combined_score: float = -math.inf
    outline_baseline_extents_px: np.ndarray | None = None
    outline_extension_px: np.ndarray | None = None
    outline_tip_offsets_yx: np.ndarray | None = None
    outline_preexisting: bool = False
    radial_openness_fraction: float = 0.0
    geometry_valid: bool = True
    native_support_fraction: float = 0.0
    native_supported_length_px: float = 0.0
    native_mean_support: float = 0.0
    native_arc_length_px: float = 0.0
    native_temporal_completion_fraction: float = 0.0
    portal_emergence_sample: int | None = None
    portal_last_dormant_sample: int | None = None
    portal_angle_radians: np.ndarray | None = None
    native_owner_visible: np.ndarray | None = None
    native_owner_visible_fraction: float = 0.0
    longest_native_owner_visibility_gap: int = 0
    minimum_foreign_owner_distance_px: float = math.inf
    native_prefix_onset_sample: int | None = None
    native_prefix_last_dormant_sample: int | None = None
    native_prefix_lengths_px: np.ndarray | None = None
    native_prefix_end_indices: np.ndarray | None = None
    growth_discontinuity_sample: int | None = None
    growth_discontinuity_length_px: float | None = None
    rupture_candidate_sample: int | None = None
    measurement_scope: str = "full"


@dataclass(frozen=True)
class PathRegistrationConfig:
    """Configure coherent normal deformation of a centerline in one frame."""

    normal_offsets_px: tuple[int, ...] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    spatial_step_penalty: float = 0.055
    temporal_step_penalty: float = 0.035


@dataclass(frozen=True)
class PortalOwnerSeed:
    """Describe one persistent pollen first recognized at a known movie sample."""

    track_id: int
    center_yx: np.ndarray
    seed_sample: int
    observable_start_sample: int
    semantic_observations: int


def parse_args() -> argparse.Namespace:
    """Parse movie, pollen identity, sampling, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument("--late-samples", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--maximum-proposals", type=int, default=10)
    parser.add_argument("--maximum-candidates", type=int, default=5)
    parser.add_argument(
        "--recovery-width",
        type=int,
        default=960,
        help="Second-pass width for unresolved pollen; 0 disables recovery.",
    )
    parser.add_argument(
        "--owner-policy",
        choices=("origin", "persistent"),
        default="persistent",
        help="Retain only frame-zero grains or also strong later sightings.",
    )
    parser.add_argument("--minimum-late-semantic-observations", type=int, default=3)
    parser.add_argument(
        "--keep-evidence",
        action="store_true",
        help="Retain the replay checkpoint for rapid prototype revalidation.",
    )
    parser.add_argument(
        "--interior-polarity",
        choices=("bright", "dark", "both", "generic", "all"),
        default="all",
    )
    parser.add_argument(
        "--track-ids",
        help="Optional comma-separated pollen IDs for a focused audit.",
    )
    return parser.parse_args()


def load_portal_owner_seeds(
    identity_report: Path,
    source_frames: np.ndarray,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    *,
    include_persistent_late: bool,
    minimum_late_semantic_observations: int,
) -> tuple[list[PortalOwnerSeed], dict]:
    """Load origin grains and reliable later sightings in aligned coordinates."""

    identity = json.loads(identity_report.read_text())
    identity_origin = int(identity["source_frames"][0])
    seeds = []
    for track in identity["tracks"]:
        semantic_count = int(track["semantic_observation_count"])
        observation_count = int(track["observation_count"])
        first_source_frame = int(track["source_frames"][0])
        is_origin = first_source_frame == identity_origin
        if is_origin:
            if semantic_count < 1:
                continue
        elif (
            not include_persistent_late
            or semantic_count < minimum_late_semantic_observations
            or observation_count < minimum_late_semantic_observations
        ):
            continue
        seed_sample = int(np.argmin(np.abs(source_frames - first_source_frame)))
        source_center = np.asarray(track["centers_yx"][0], dtype=np.float64)
        aligned_center = (
            source_center * analysis_scale - shifts_xy[seed_sample][::-1]
        )
        seeds.append(
            PortalOwnerSeed(
                track_id=int(track["track_id"]),
                center_yx=aligned_center,
                seed_sample=seed_sample,
                observable_start_sample=0 if is_origin else seed_sample,
                semantic_observations=semantic_count,
            )
        )
    seeds.sort(key=lambda item: (item.seed_sample, item.track_id))
    if not seeds:
        raise RuntimeError("identity report contains no reliable pollen sightings")
    return seeds, identity


def retain_distinct_owner_tracks(
    seeds: list[PortalOwnerSeed],
    center_tracks_yx: np.ndarray,
    center_scores: np.ndarray,
    *,
    owner_radius_px: float,
) -> tuple[list[PortalOwnerSeed], np.ndarray, np.ndarray, list[dict]]:
    """Drop later identity fragments that coincide with a retained pollen track."""

    retained_indices = []
    suppressed = []
    duplicate_distance = 0.90 * owner_radius_px
    for index, seed in enumerate(seeds):
        duplicate_of = None
        duplicate_separation = math.inf
        if seed.seed_sample > 0:
            sample = seed.seed_sample
            for retained_index in retained_indices:
                separation = float(
                    np.linalg.norm(
                        center_tracks_yx[index, sample]
                        - center_tracks_yx[retained_index, sample]
                    )
                )
                if separation < duplicate_distance and separation < duplicate_separation:
                    duplicate_of = seeds[retained_index].track_id
                    duplicate_separation = separation
        if duplicate_of is not None:
            suppressed.append(
                {
                    "track_id": seed.track_id,
                    "duplicate_of": duplicate_of,
                    "separation_px": duplicate_separation,
                }
            )
        else:
            retained_indices.append(index)
    return (
        [seeds[index] for index in retained_indices],
        center_tracks_yx[retained_indices],
        center_scores[retained_indices],
        suppressed,
    )


def retain_native_pollen_owners(
    movie: Path,
    source_frames: np.ndarray,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    seeds: list[PortalOwnerSeed],
    center_tracks_yx: np.ndarray,
    center_scores: np.ndarray,
    *,
    crop_size: int = 64,
) -> tuple[list[PortalOwnerSeed], np.ndarray, np.ndarray, list[dict]]:
    """Reject later tube-tip detections that lack a complete pollen ring."""

    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie} for pollen-ring validation")
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    config = PollenRingConfig(pollen_radius_px=15.0)
    retained_indices = []
    rejected = []
    for index, seed in enumerate(seeds):
        if seed.seed_sample == 0:
            retained_indices.append(index)
            continue
        center_yx = (
            center_tracks_yx[index, seed.seed_sample]
            + shifts_xy[seed.seed_sample][::-1]
        ) / analysis_scale
        edge_margin = config.outer_radius_factors[1] * config.pollen_radius_px
        if (
            center_yx[0] < edge_margin
            or center_yx[0] >= height - edge_margin
            or center_yx[1] < edge_margin
            or center_yx[1] >= width - edge_margin
        ):
            rejected.append({"track_id": seed.track_id, "reason": "partial-frame"})
            continue
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frames[seed.seed_sample]))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("could not decode a pollen-ring validation frame")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        crop = cv.getRectSubPix(
            gray,
            (crop_size, crop_size),
            (float(center_yx[1]), float(center_yx[0])),
        )
        assessment = assess_pollen_ring(crop, config)
        if assessment.accepted:
            retained_indices.append(index)
        else:
            rejected.append(
                {
                    "track_id": seed.track_id,
                    "reason": "not-a-closed-pollen-ring",
                    "ring_contrast": round(assessment.ring_contrast, 4),
                    "angular_coverage": round(assessment.angular_coverage, 4),
                }
            )
    capture.release()
    return (
        [seeds[index] for index in retained_indices],
        center_tracks_yx[retained_indices],
        center_scores[retained_indices],
        rejected,
    )


def _crop_stack(
    stack: np.ndarray,
    center_yx: np.ndarray,
    size: int,
) -> np.ndarray:
    """Extract an edge-reflected orientation stack around a subpixel center."""

    half = size // 2
    center = np.rint(center_yx).astype(int)
    padded = np.pad(stack, ((0, 0), (half, half), (half, half)), mode="reflect")
    y = int(center[0] + half)
    x = int(center[1] + half)
    return padded[:, y - half : y - half + size, x - half : x - half + size]


def _extract_features(
    frame: np.ndarray,
    config: PhaseContrastRibbonConfig | OrientationScoreConfig,
) -> PairedWallOrientationResult:
    """Evaluate one frame without mixing phase appearance state layers."""

    if isinstance(config, PhaseContrastRibbonConfig):
        return phase_contrast_ribbon_features(frame, config)
    return paired_wall_orientation_features(frame, config)


def build_owner_late_atlases(
    frames: np.ndarray,
    center_tracks_yx: np.ndarray,
    feature_config: PhaseContrastRibbonConfig | OrientationScoreConfig,
    *,
    late_samples: int,
    crop_size: int,
) -> list[PairedWallOrientationResult]:
    """Fuse late tube evidence after translating each frame with its pollen."""

    owner_count = len(center_tracks_yx)
    orientation_count = feature_config.orientation_count
    shape = (owner_count, orientation_count, crop_size, crop_size)
    score_sum = np.zeros(shape, dtype=np.float32)
    paired_sum = np.zeros(shape, dtype=np.float32)
    width_sum = np.zeros(shape, dtype=np.float32)
    balance_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    start = max(0, len(frames) - late_samples)
    for offset, sample in enumerate(range(start, len(frames)), start=1):
        feature = _extract_features(frames[sample], feature_config)
        for owner in range(owner_count):
            center = center_tracks_yx[owner, sample]
            score = _crop_stack(feature.score, center, crop_size)
            paired = _crop_stack(feature.paired_score, center, crop_size)
            width = _crop_stack(feature.half_width_px, center, crop_size)
            balance = _crop_stack(feature.wall_balance, center, crop_size)
            weight = np.maximum(paired, 0.05 * score)
            score_sum[owner] += score
            paired_sum[owner] += paired
            width_sum[owner] += width * weight
            balance_sum[owner] += balance * weight
            weight_sum[owner] += weight
        print(f"[v25 late atlas] {offset}/{len(frames) - start}", flush=True)
    denominator = np.maximum(weight_sum, 1e-6)
    count = max(1, len(frames) - start)
    return [
        PairedWallOrientationResult(
            score=score_sum[owner] / count,
            paired_score=paired_sum[owner] / count,
            half_width_px=width_sum[owner] / denominator[owner],
            wall_balance=balance_sum[owner] / denominator[owner],
        )
        for owner in range(owner_count)
    ]


def _scale_feature_config(
    config: PhaseContrastRibbonConfig | OrientationScoreConfig,
    scale: float,
) -> PhaseContrastRibbonConfig | OrientationScoreConfig:
    """Scale pixel-valued feature settings without changing score semantics."""

    common = {
        "half_widths_px": tuple(value * scale for value in config.half_widths_px),
        "tangent_samples_px": tuple(
            value * scale for value in config.tangent_samples_px
        ),
        "background_sigma_px": config.background_sigma_px * scale,
        "structure_sigma_px": config.structure_sigma_px * scale,
    }
    if isinstance(config, PhaseContrastRibbonConfig):
        return replace(config, fine_sigma_px=config.fine_sigma_px * scale, **common)
    return replace(config, wall_sigma_px=config.wall_sigma_px * scale, **common)


def _build_owner_relative_atlas(
    frames: np.ndarray,
    centers_yx: np.ndarray,
    feature_config: PhaseContrastRibbonConfig | OrientationScoreConfig,
    *,
    crop_size: int,
) -> PairedWallOrientationResult:
    """Fuse high-resolution crops after keeping one moving owner centered."""

    orientation_count = feature_config.orientation_count
    shape = (orientation_count, crop_size, crop_size)
    score_sum = np.zeros(shape, dtype=np.float32)
    paired_sum = np.zeros(shape, dtype=np.float32)
    width_sum = np.zeros(shape, dtype=np.float32)
    balance_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    for frame, center_yx in zip(frames, centers_yx):
        crop = cv.getRectSubPix(
            frame,
            (crop_size, crop_size),
            (float(center_yx[1]), float(center_yx[0])),
        )
        feature = _extract_features(crop, feature_config)
        weight = np.maximum(feature.paired_score, 0.05 * feature.score)
        score_sum += feature.score
        paired_sum += feature.paired_score
        width_sum += feature.half_width_px * weight
        balance_sum += feature.wall_balance * weight
        weight_sum += weight
    count = max(1, len(frames))
    denominator = np.maximum(weight_sum, 1e-6)
    return PairedWallOrientationResult(
        score=score_sum / count,
        paired_score=paired_sum / count,
        half_width_px=width_sum / denominator,
        wall_balance=balance_sum / denominator,
    )


def _rescale_candidate_geometry(
    candidate: PortalCandidate,
    *,
    scale: float,
    destination_crop_size: int,
) -> None:
    """Return high-resolution geometry to the main analysis coordinate system."""

    destination_center = np.full(
        2, destination_crop_size / 2.0, dtype=np.float32
    )
    candidate.local_path_yx = (
        destination_center
        + (candidate.local_path_yx - candidate.local_center_yx) / scale
    ).astype(np.float32)
    candidate.local_center_yx = destination_center
    candidate.radial_extension_px /= scale


def generate_multiscale_recovery_candidates(
    movie: Path,
    source_frames: np.ndarray,
    native_width: int,
    native_height: int,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    seeds: list[PortalOwnerSeed],
    selected: list[PortalCandidate],
    feature_configs: dict[
        str, PhaseContrastRibbonConfig | OrientationScoreConfig
    ],
    *,
    analysis_crop_size: int,
    owner_radius_px: float,
    late_samples: int,
    recovery_width: int,
    maximum_proposals: int,
    maximum_candidates: int,
) -> list[PortalCandidate]:
    """Regenerate unresolved path geometry at higher spatial resolution."""

    accepted_owner_ids = {
        candidate.owner_track_id for candidate in selected if candidate.accepted
    }
    unresolved = [
        owner
        for owner, seed in enumerate(seeds)
        if seed.track_id not in accepted_owner_ids
    ]
    analysis_width = native_width * analysis_scale
    scale = recovery_width / analysis_width
    if not unresolved or recovery_width <= analysis_width or scale <= 1.0:
        return []
    recovery_crop_size = int(round(analysis_crop_size * scale))
    recovery_crop_size += recovery_crop_size % 2
    tail_count = min(late_samples, len(source_frames))
    tail_frames = load_grayscale_samples(
        movie,
        source_frames[-tail_count:],
        recovery_width,
        native_width,
        native_height,
    )
    raw_centers_yx = (
        center_tracks_yx[:, -tail_count:]
        + shifts_xy[-tail_count:, ::-1][None]
    ) * scale
    aligned_centers_yx = center_tracks_yx * scale
    scaled_configs = {
        mode: _scale_feature_config(config, scale)
        for mode, config in feature_configs.items()
    }
    recovered: list[PortalCandidate] = []
    for count, owner in enumerate(unresolved, start=1):
        seed = seeds[owner]
        owner_candidates: list[PortalCandidate] = []
        for mode, config in scaled_configs.items():
            atlas = _build_owner_relative_atlas(
                tail_frames,
                raw_centers_yx[owner],
                config,
                crop_size=recovery_crop_size,
            )
            traced = trace_owner_candidates(
                owner,
                seed.track_id,
                seed.observable_start_sample,
                mode,
                atlas,
                aligned_centers_yx,
                owner_radius_px=owner_radius_px * scale,
                crop_size=recovery_crop_size,
                late_samples=tail_count,
                sample_count=len(source_frames),
                support_modes=tuple(feature_configs),
                maximum_proposals=maximum_proposals,
                maximum_candidates=maximum_candidates,
                spatial_scale=scale,
            )
            for candidate in traced:
                _rescale_candidate_geometry(
                    candidate,
                    scale=scale,
                    destination_crop_size=analysis_crop_size,
                )
            owner_candidates.extend(traced)
        recovered.extend(owner_candidates)
        print(
            f"[v25 multiscale] {count}/{len(unresolved)} P{seed.track_id:02d}: "
            f"{len(owner_candidates)} candidates at width {recovery_width}",
            flush=True,
        )
    return recovered


def _mask_local_atlas(
    atlas: PairedWallOrientationResult,
    owner_index: int,
    center_tracks_yx: np.ndarray,
    *,
    crop_size: int,
    owner_radius_px: float,
    late_samples: int,
) -> PairedWallOrientationResult:
    """Remove pollen bodies while retaining surrounding tubular material."""

    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    local_center = np.full(2, crop_size / 2.0, dtype=np.float32)
    cv.circle(
        mask,
        tuple(np.rint(local_center[::-1]).astype(int)),
        round(1.08 * owner_radius_px),
        1,
        -1,
    )
    start = max(0, center_tracks_yx.shape[1] - late_samples)
    for other in range(len(center_tracks_yx)):
        if other == owner_index:
            continue
        for sample in range(start, center_tracks_yx.shape[1], 2):
            relative = (
                local_center
                + center_tracks_yx[other, sample]
                - center_tracks_yx[owner_index, sample]
            )
            if np.all(relative >= -owner_radius_px) and np.all(
                relative < crop_size + owner_radius_px
            ):
                cv.circle(
                    mask,
                    tuple(np.rint(relative[::-1]).astype(int)),
                    round(1.32 * owner_radius_px),
                    1,
                    -1,
                )
    arrays = []
    for source in (
        atlas.score,
        atlas.paired_score,
        atlas.half_width_px,
        atlas.wall_balance,
    ):
        value = np.asarray(source, dtype=np.float32).copy()
        value[:, mask.astype(bool)] = 0.0
        arrays.append(value)
    return PairedWallOrientationResult(*arrays)


def _path_directions(
    path_yx: np.ndarray,
    orientation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected orientation bins and normal vectors along a path."""

    tangent = np.gradient(np.asarray(path_yx, dtype=np.float32), axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-6)
    angles = np.mod(np.arctan2(tangent[:, 0], tangent[:, 1]), math.pi)
    bins = np.mod(
        np.rint(angles / math.pi * orientation_count).astype(np.int32),
        orientation_count,
    )
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)
    return bins, normal


def _path_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Measure symmetric raster overlap between two candidate centerlines."""

    def pixels(path: np.ndarray) -> set[tuple[int, int]]:
        result = set()
        for y, x in np.rint(path).astype(int):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    result.add((int(y + dy), int(x + dx)))
        return result

    left_pixels = pixels(left)
    right_pixels = pixels(right)
    return len(left_pixels & right_pixels) / max(
        min(len(left_pixels), len(right_pixels)), 1
    )


def trace_owner_candidates(
    owner_index: int,
    owner_track_id: int,
    observable_start_sample: int,
    appearance_mode: str,
    atlas: PairedWallOrientationResult,
    center_tracks_yx: np.ndarray,
    *,
    owner_radius_px: float,
    crop_size: int,
    late_samples: int,
    sample_count: int,
    support_modes: tuple[str, ...],
    maximum_proposals: int,
    maximum_candidates: int,
    spatial_scale: float = 1.0,
) -> list[PortalCandidate]:
    """Trace distinct late paths from every plausible owner-boundary portal."""

    aggregate = _mask_local_atlas(
        atlas,
        owner_index,
        center_tracks_yx,
        crop_size=crop_size,
        owner_radius_px=owner_radius_px,
        late_samples=late_samples,
    )
    evidence = np.maximum(aggregate.paired_score, 0.20 * aggregate.score)
    local_center = np.full(2, crop_size / 2.0, dtype=np.float32)
    proposals = propose_pollen_roots(
        evidence,
        local_center,
        owner_radius_px,
        attachment_count=96,
        direction_offsets=(-3, -2, -1, 0, 1, 2, 3),
        probe_distances_px=tuple(
            spatial_scale * value for value in (2.0, 3.5, 5.0, 7.0, 9.0, 12.0)
        ),
        probe_top_k=4,
        maximum_proposals=maximum_proposals,
        minimum_angle_separation_degrees=10.0,
    )
    trace_config = CoupledRibbonTraceConfig(
        step_px=1.0,
        maximum_turn_bins=1,
        curvature_penalty=0.10,
        minimum_pair_support=0.045,
        minimum_wall_balance=0.10,
        paired_support_weight=1.30,
        merged_support_weight=0.18,
        wall_balance_weight=0.05,
        evidence_floor=0.052,
        length_reward=0.012 / spatial_scale,
        width_change_penalty=0.10 / spatial_scale,
        maximum_reacquisition_width_change_px=1.7 * spatial_scale,
        maximum_gap_steps=round(6 * spatial_scale),
        maximum_initial_gap_steps=round(5 * spatial_scale),
        root_occlusion_px=1.10 * owner_radius_px,
        beam_width=420,
        minimum_length_px=6.0 * spatial_scale,
        maximum_length_px=min(
            90.0 * spatial_scale, crop_size / 2.0 - 3.0 * spatial_scale
        ),
        minimum_endpoint_separation_fraction=0.34,
        endpoint_openness_reward=0.16,
    )
    candidates = []
    for proposal in proposals:
        trace = trace_coupled_ribbon_lifted(
            aggregate,
            proposal.root_yx,
            proposal.direction_yx,
            config=trace_config,
        )
        if trace.length_px < trace_config.minimum_length_px:
            continue
        path = trace.path_yx
        path = truncate_self_reentry(path)
        distance = np.linalg.norm(path - local_center[None], axis=1)
        clear = np.flatnonzero(distance >= 1.06 * owner_radius_px)
        if not len(clear):
            continue
        path = path[int(clear[0]) :]
        if len(path) < 4:
            continue
        radial_extension = float(np.max(distance) - owner_radius_px)
        if radial_extension < 1.05 * owner_radius_px:
            continue
        if not path_exits_owner_once(path, local_center, owner_radius_px):
            continue
        bins, normal = _path_directions(path, evidence.shape[0])
        geometric_score = (
            1.8 * trace.paired_supported_fraction
            + 0.012 * trace.length_px / spatial_scale
            + 0.4 * trace.endpoint_separation_fraction
        )
        candidate = PortalCandidate(
            owner_index=owner_index,
            owner_track_id=owner_track_id,
            observable_start_sample=observable_start_sample,
            appearance_mode=appearance_mode,
            local_center_yx=local_center.copy(),
            local_path_yx=path,
            direction_bins=bins,
            normal_yx=normal,
            geometric_score=float(geometric_score),
            paired_fraction=float(trace.paired_supported_fraction),
            radial_extension_px=radial_extension,
            evidence_by_mode={
                mode: np.zeros((sample_count, len(path)), dtype=np.float32)
                for mode in support_modes
            },
            normal_offsets_by_mode={
                mode: np.zeros((sample_count, len(path)), dtype=np.int8)
                for mode in support_modes
            },
        )
        if any(
            _path_overlap(candidate.local_path_yx, retained.local_path_yx) >= 0.62
            for retained in candidates
        ):
            continue
        candidates.append(candidate)
        if len(candidates) >= maximum_candidates:
            break
    return candidates


def _sample_candidate(
    feature: PairedWallOrientationResult,
    candidate: PortalCandidate,
    owner_center_yx: np.ndarray,
    previous_offsets_px: np.ndarray | None = None,
    registration: PathRegistrationConfig = PathRegistrationConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one path using a spatially and temporally coherent deformation."""

    offsets_from_owner = candidate.local_path_yx - candidate.local_center_yx
    base = owner_center_yx[None] + offsets_from_owner
    height, width = feature.score.shape[1:]
    normal_offsets = np.asarray(registration.normal_offsets_px, dtype=np.int8)
    support_by_offset = np.zeros((len(normal_offsets), len(base)), dtype=np.float32)
    for state, normal_offset in enumerate(normal_offsets):
        points = base + float(normal_offset) * candidate.normal_yx
        points = np.rint(points).astype(np.int32)
        y = np.clip(points[:, 0], 0, height - 1)
        x = np.clip(points[:, 1], 0, width - 1)
        for direction_offset in (-1, 0, 1):
            bins = np.mod(
                candidate.direction_bins + direction_offset,
                feature.score.shape[0],
            )
            value = np.maximum(
                feature.paired_score[bins, y, x],
                0.20 * feature.score[bins, y, x],
            )
            support_by_offset[state] = np.maximum(
                support_by_offset[state], value
            )

    state_count, point_count = support_by_offset.shape
    scores = support_by_offset[:, 0].astype(np.float64)
    if previous_offsets_px is not None:
        scores -= registration.temporal_step_penalty * np.abs(
            normal_offsets - previous_offsets_px[0]
        )
    predecessors = np.zeros((point_count, state_count), dtype=np.int16)
    transition = registration.spatial_step_penalty * np.abs(
        normal_offsets[:, None] - normal_offsets[None, :]
    )
    for point in range(1, point_count):
        proposed = scores[:, None] - transition
        predecessor = np.argmax(proposed, axis=0)
        scores = proposed[predecessor, np.arange(state_count)]
        scores += support_by_offset[:, point]
        if previous_offsets_px is not None:
            scores -= registration.temporal_step_penalty * np.abs(
                normal_offsets - previous_offsets_px[point]
            )
        predecessors[point] = predecessor

    states = np.empty(point_count, dtype=np.int16)
    states[-1] = int(np.argmax(scores))
    for point in range(point_count - 1, 0, -1):
        states[point - 1] = predecessors[point, states[point]]
    selected_offsets = normal_offsets[states]
    selected_support = support_by_offset[states, np.arange(point_count)]
    return selected_support, selected_offsets


def collect_temporal_evidence(
    frames: np.ndarray,
    center_tracks_yx: np.ndarray,
    candidates: list[PortalCandidate],
    feature_configs: dict[
        str, PhaseContrastRibbonConfig | OrientationScoreConfig
    ],
    *,
    start_sample: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_signature: dict | None = None,
) -> None:
    """Replay the movie once and sample every late geometry hypothesis."""

    for sample in range(start_sample, len(frames)):
        frame = frames[sample]
        features = {
            mode: _extract_features(frame, config)
            for mode, config in feature_configs.items()
        }
        for candidate in candidates:
            if sample < candidate.observable_start_sample:
                continue
            for mode, feature in features.items():
                support, offset = _sample_candidate(
                    feature,
                    candidate,
                    center_tracks_yx[candidate.owner_index, sample],
                    (
                        candidate.normal_offsets_by_mode[mode][sample - 1]
                        if sample > 0
                        else None
                    ),
                )
                candidate.evidence_by_mode[mode][sample] = support
                candidate.normal_offsets_by_mode[mode][sample] = offset
        if sample == 0 or (sample + 1) % 20 == 0:
            print(f"[v25 causal replay] {sample + 1}/{len(frames)}", flush=True)
        if checkpoint_path is not None and (
            (sample + 1) % 20 == 0 or sample == len(frames) - 1
        ):
            _write_replay_checkpoint(
                checkpoint_path,
                next_sample=sample + 1,
                candidates=candidates,
                signature=checkpoint_signature or {},
            )


def _write_replay_checkpoint(
    path: Path,
    *,
    next_sample: int,
    candidates: list[PortalCandidate],
    signature: dict,
) -> None:
    """Atomically persist temporal evidence so a long replay can resume."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "next_sample": next_sample,
                "candidates": candidates,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(path)


def _load_replay_checkpoint(
    path: Path,
    expected_signature: dict,
) -> tuple[int, list[PortalCandidate]]:
    """Load a checkpoint created by this prototype in its output directory."""

    with path.open("rb") as handle:
        state = pickle.load(handle)
    if state.get("signature") != expected_signature:
        raise RuntimeError(
            "existing replay checkpoint does not match the requested analysis"
        )
    return int(state["next_sample"]), list(state["candidates"])


def validate_candidates(
    candidates: list[PortalCandidate],
    owner_radius_px: float,
) -> list[PortalCandidate]:
    """Choose at most one causally valid path per pollen owner."""

    config = CausalPortalConfig(
        warmup_samples=8,
        tail_samples=20,
        smoothing_radius=1,
        confirmation_window=5,
        confirmation_required=3,
        absolute_support_floor=0.045,
        minimum_support_change=0.018,
        noise_multiplier=2.5,
        maximum_gap_points=4,
        initial_search_points=8,
        minimum_late_coverage=0.38,
        maximum_early_coverage=0.34,
        maximum_pre_onset_orphan_fraction=0.95,
        minimum_birth_order_fraction=0.20,
        minimum_emergence_length_px=0.15 * owner_radius_px,
        onset_growth_window_samples=20,
        minimum_onset_growth_px=0.35 * owner_radius_px,
        minimum_growth_px=0.90 * owner_radius_px,
        minimum_final_length_px=1.10 * owner_radius_px,
    )
    by_owner: dict[int, list[PortalCandidate]] = {}
    for candidate in candidates:
        arc = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.linalg.norm(np.diff(candidate.local_path_yx, axis=0), axis=1)
                ),
            )
        )
        candidate.radial_openness_fraction = candidate.radial_extension_px / max(
            float(arc[-1]), 1.0
        )
        candidate.geometry_valid = candidate.radial_openness_fraction >= 0.45
        validations = {
            mode: validate_causal_portal(evidence, arc, config=config)
            for mode, evidence in candidate.evidence_by_mode.items()
        }
        accepted_modes = [
            (mode, result)
            for mode, result in validations.items()
            if result.accepted
        ]
        if accepted_modes:
            temporal_mode, result = min(
                accepted_modes,
                key=lambda item: (
                    item[1].onset_sample is None,
                    item[1].onset_sample
                    if item[1].onset_sample is not None
                    else math.inf,
                    -item[1].score,
                ),
            )
        else:
            temporal_mode, result = max(
                validations.items(), key=lambda item: item[1].score
            )
        candidate.temporal_mode = temporal_mode
        candidate.normal_offsets_px = candidate.normal_offsets_by_mode[temporal_mode]
        candidate.validation = result
        candidate.reason = (
            result.reason
            if candidate.geometry_valid
            else f"{result.reason};owner-halo-loop"
        )
        candidate.combined_score = result.score + 0.35 * candidate.geometric_score
        by_owner.setdefault(candidate.owner_track_id, []).append(candidate)
    selected = []
    for owner_candidates in by_owner.values():
        accepted = [
            item
            for item in owner_candidates
            if item.validation and item.validation.accepted and item.geometry_valid
        ]
        phase_consistent = [
            item for item in accepted if item.appearance_mode != "generic"
        ]
        eligible = phase_consistent or accepted
        winner = (
            max(eligible, key=lambda item: item.combined_score)
            if eligible
            else max(owner_candidates, key=lambda item: item.combined_score)
        )
        winner.accepted = bool(
            winner.validation and winner.validation.accepted and winner.geometry_valid
        )
        selected.append(winner)
    return selected


def score_native_candidate_geometry(
    movie: Path,
    source_frames: np.ndarray,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    candidates: list[PortalCandidate],
    *,
    late_samples: int,
    audit_frames: int = 6,
) -> None:
    """Score every mature candidate against native paired-wall evidence."""

    tail_start = max(0, len(source_frames) - late_samples)
    sample_indices = np.unique(
        np.rint(
            np.linspace(tail_start, len(source_frames) - 1, audit_frames)
        ).astype(int)
    )
    support_by_candidate: list[list[np.ndarray]] = [[] for _ in candidates]
    threshold_by_candidate: list[list[float]] = [[] for _ in candidates]
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie} for native path scoring")
    for sample in sample_indices:
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frames[sample]))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(
                f"could not decode native source frame {source_frames[sample]}"
            )
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        drift_yx = shifts_xy[sample][::-1]
        for index, candidate in enumerate(candidates):
            offsets = (
                candidate.normal_offsets_px[sample].astype(np.float32)
                if candidate.normal_offsets_px is not None
                else np.zeros(len(candidate.local_path_yx), dtype=np.float32)
            )
            path = candidate.local_path_yx + offsets[:, None] * candidate.normal_yx
            path += (
                center_tracks_yx[candidate.owner_index, sample]
                - candidate.local_center_yx
                + drift_yx
            )[None]
            native_path = path / analysis_scale
            support, control = sample_native_paired_support(gray, native_path)
            control_median = float(np.median(control))
            control_noise = 1.4826 * float(
                np.median(np.abs(control - control_median))
            )
            threshold = max(2.0, control_median + 2.0 * control_noise)
            support_by_candidate[index].append(support)
            threshold_by_candidate[index].append(threshold)
    capture.release()

    for candidate, support_frames, thresholds in zip(
        candidates, support_by_candidate, threshold_by_candidate
    ):
        support = np.median(np.stack(support_frames), axis=0)
        threshold = float(np.median(thresholds))
        arc_length = float(
            np.sum(
                np.linalg.norm(np.diff(candidate.local_path_yx, axis=0), axis=1)
            )
            / analysis_scale
        )
        candidate.native_support_fraction = float(np.mean(support >= threshold))
        candidate.native_arc_length_px = arc_length
        candidate.native_supported_length_px = (
            arc_length * candidate.native_support_fraction
        )
        candidate.native_mean_support = float(np.mean(support))


def select_native_supported_candidates(
    candidates: list[PortalCandidate],
    owner_radius_px: float,
    analysis_scale: float,
    center_tracks_yx: np.ndarray,
    *,
    late_samples: int,
) -> list[PortalCandidate]:
    """Select one owner path by persistent native supported length."""

    by_owner: dict[int, list[PortalCandidate]] = {}
    native_radius = owner_radius_px / analysis_scale
    late_centers = np.median(center_tracks_yx[:, -late_samples:], axis=1)
    for candidate in candidates:
        candidate.accepted = False
        arc_length = float(
            np.sum(
                np.linalg.norm(np.diff(candidate.local_path_yx, axis=0), axis=1)
            )
        )
        candidate.radial_openness_fraction = candidate.radial_extension_px / max(
            arc_length, 1.0
        )
        returns_to_owner = _path_returns_to_owner(
            candidate.local_path_yx,
            candidate.local_center_yx,
            owner_radius_px,
        )
        candidate.geometry_valid = not returns_to_owner
        global_path = (
            candidate.local_path_yx
            - candidate.local_center_yx
            + late_centers[candidate.owner_index]
        )
        foreign_centers = np.delete(
            late_centers, candidate.owner_index, axis=0
        )
        candidate.minimum_foreign_owner_distance_px = (
            float(
                np.min(
                    np.linalg.norm(
                        global_path[3:, None] - foreign_centers[None], axis=2
                    )
                )
            )
            if len(foreign_centers) and len(global_path) > 3
            else math.inf
        )
        by_owner.setdefault(candidate.owner_track_id, []).append(candidate)

    selected = []
    for owner_candidates in by_owner.values():
        eligible = []
        for candidate in owner_candidates:
            validation = candidate.validation
            recoverable_order_failure = bool(
                validation is not None
                and validation.reason == "noncausal-point-order"
            )
            temporal_valid = bool(
                validation is not None
                and (validation.accepted or recoverable_order_failure)
            )
            temporal_final_length = (
                float(validation.prefix_lengths_px[-1]) / analysis_scale
                if validation is not None
                else 0.0
            )
            candidate.native_temporal_completion_fraction = (
                temporal_final_length / max(candidate.native_arc_length_px, 1.0)
            )
            native_valid = bool(
                candidate.native_support_fraction >= 0.45
                and candidate.native_supported_length_px
                >= max(15.0, native_radius)
                and candidate.native_temporal_completion_fraction >= 0.35
                and candidate.minimum_foreign_owner_distance_px
                >= 1.30 * owner_radius_px
            )
            if temporal_valid and native_valid and candidate.geometry_valid:
                eligible.append(candidate)
        if eligible:
            winner = max(
                eligible,
                key=lambda item: (
                    min(
                        item.native_supported_length_px,
                        float(item.validation.prefix_lengths_px[-1])
                        / analysis_scale,
                    )
                    + 1.5 * item.native_mean_support,
                    item.combined_score,
                ),
            )
            winner.accepted = True
            winner.combined_score = (
                min(
                    winner.native_supported_length_px,
                    float(winner.validation.prefix_lengths_px[-1])
                    / analysis_scale,
                )
                + 1.5 * winner.native_mean_support
            )
            winner.reason = (
                "accepted-native-supported"
                if winner.validation and winner.validation.accepted
                else "accepted-native-supported-order-recovery"
            )
        else:
            winner = max(
                owner_candidates,
                key=lambda item: (
                    item.native_supported_length_px
                    + 1.5 * item.native_mean_support,
                    item.combined_score,
                ),
            )
            winner.reason = f"{winner.reason};insufficient-native-path-support"
        selected.append(winner)
    return selected


def _path_returns_to_owner(
    path_yx: np.ndarray,
    center_yx: np.ndarray,
    owner_radius_px: float,
) -> bool:
    """Distinguish a closed owner-rim loop from an open curved tube."""

    path = np.asarray(path_yx, dtype=np.float32)
    arc_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    endpoint_radius = float(np.linalg.norm(path[-1] - center_yx))
    return bool(
        endpoint_radius <= 1.60 * owner_radius_px
        and arc_length >= 2.0 * owner_radius_px
    )


def recover_native_temporal_prefixes(
    movie: Path,
    source_frames: np.ndarray,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    selected: list[PortalCandidate],
    *,
    owner_radius_px: float,
) -> None:
    """Recover coarse-front failures only when native evidence is decisive."""

    recovery = [
        candidate
        for candidate in selected
        if not candidate.accepted
        and candidate.validation is not None
        and candidate.validation.accepted
        and candidate.geometry_valid
        and candidate.native_support_fraction >= 0.65
        and candidate.native_supported_length_px >= 30.0
        and 0.15 <= candidate.native_temporal_completion_fraction < 0.35
        and candidate.minimum_foreign_owner_distance_px
        >= 1.30 * owner_radius_px
    ]
    if not recovery:
        return
    support = {
        candidate.owner_track_id: np.zeros(
            (len(source_frames), len(candidate.local_path_yx)), dtype=np.float32
        )
        for candidate in recovery
    }
    thresholds = {
        candidate.owner_track_id: np.zeros(len(source_frames), dtype=np.float32)
        for candidate in recovery
    }
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie} for native prefix recovery")
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode native source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        drift_yx = shifts_xy[sample][::-1]
        for candidate in recovery:
            offsets = candidate.normal_offsets_px[sample].astype(np.float32)
            path = candidate.local_path_yx + offsets[:, None] * candidate.normal_yx
            path += (
                center_tracks_yx[candidate.owner_index, sample]
                - candidate.local_center_yx
                + drift_yx
            )[None]
            native_support, control = sample_native_paired_support(
                gray, path / analysis_scale
            )
            control_median = float(np.median(control))
            control_noise = 1.4826 * float(
                np.median(np.abs(control - control_median))
            )
            support[candidate.owner_track_id][sample] = native_support
            thresholds[candidate.owner_track_id][sample] = max(
                2.0, control_median + 2.0 * control_noise
            )
    capture.release()

    for candidate in recovery:
        arc = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.linalg.norm(
                        np.diff(candidate.local_path_yx / analysis_scale, axis=0),
                        axis=1,
                    )
                ),
            )
        ).astype(np.float32)
        growth, end_indices, onset, last_dormant = native_connected_prefix_growth(
            support[candidate.owner_track_id],
            thresholds[candidate.owner_track_id],
            arc,
        )
        completion = float(growth[-1] / max(candidate.native_arc_length_px, 1.0))
        if onset is None or completion < 0.50:
            continue
        candidate.accepted = True
        candidate.reason = "accepted-native-prefix-recovery"
        candidate.native_prefix_lengths_px = growth
        candidate.native_prefix_end_indices = end_indices
        candidate.native_prefix_onset_sample = onset
        candidate.native_prefix_last_dormant_sample = last_dormant
        candidate.combined_score = max(
            candidate.combined_score,
            float(growth[-1] + 1.5 * candidate.native_mean_support),
        )


def truncate_discontinuous_growth(
    selected: list[PortalCandidate],
    analysis_scale: float,
    *,
    minimum_previous_length_px: float = 10.0,
    minimum_jump_px: float = 30.0,
    minimum_final_fraction: float = 0.45,
) -> None:
    """Freeze a valid prefix when one sample suddenly acquires a foreign branch."""

    for candidate in selected:
        if not candidate.accepted or candidate.validation is None:
            continue
        if candidate.native_prefix_lengths_px is not None:
            lengths = candidate.native_prefix_lengths_px
            end_indices = candidate.native_prefix_end_indices
        else:
            lengths = candidate.validation.prefix_lengths_px / analysis_scale
            end_indices = candidate.validation.prefix_end_indices
        final_length = float(lengths[-1])
        if final_length <= 0.0:
            continue
        jumps = np.diff(lengths, prepend=lengths[0])
        discontinuities = np.flatnonzero(
            (np.r_[0.0, lengths[:-1]] >= minimum_previous_length_px)
            & (jumps >= minimum_jump_px)
            & (jumps >= minimum_final_fraction * final_length)
        )
        if not len(discontinuities):
            continue
        sample = int(discontinuities[0])
        retained_length = float(lengths[sample - 1])
        if candidate.native_prefix_lengths_px is not None:
            candidate.native_prefix_lengths_px[sample:] = retained_length
        else:
            candidate.validation.prefix_lengths_px[sample:] = (
                retained_length * analysis_scale
            )
        end_indices[sample:] = end_indices[sample - 1]
        candidate.growth_discontinuity_sample = sample
        candidate.growth_discontinuity_length_px = retained_length
        candidate.measurement_scope = "pre-discontinuity"
        candidate.reason = f"{candidate.reason};truncated-at-growth-discontinuity"


def resolve_cross_owner_conflicts(
    selected: list[PortalCandidate],
    center_tracks_yx: np.ndarray,
    late_samples: int,
) -> list[dict]:
    """Prevent two pollen owners from claiming the same global tube branch."""

    late_centers = np.median(center_tracks_yx[:, -late_samples:], axis=1)
    accepted = sorted(
        (candidate for candidate in selected if candidate.accepted),
        key=lambda item: item.combined_score,
        reverse=True,
    )
    resolutions = []
    retained: list[PortalCandidate] = []
    for candidate in accepted:
        candidate_global = (
            candidate.local_path_yx
            - candidate.local_center_yx
            + late_centers[candidate.owner_index]
        )
        conflict = None
        for winner in retained:
            winner_global = (
                winner.local_path_yx
                - winner.local_center_yx
                + late_centers[winner.owner_index]
            )
            overlap = _path_overlap(candidate_global[5:], winner_global[5:])
            if overlap < 0.55:
                continue
            conflict = (winner, overlap)
            break
        if conflict is None:
            retained.append(candidate)
        else:
            winner, overlap = conflict
            candidate.accepted = False
            candidate.reason = f"duplicate-of-P{winner.owner_track_id:02d}"
            resolutions.append(
                {
                    "winner_owner_track_id": winner.owner_track_id,
                    "loser_owner_track_id": candidate.owner_track_id,
                    "overlap_fraction": round(overlap, 4),
                }
            )
    return resolutions


def measurement_disposition(
    candidate: PortalCandidate,
) -> tuple[str, int | None]:
    """Separate measured tube length from an observable germination event."""

    validation = candidate.validation
    if candidate.rupture_candidate_sample is not None:
        return "rupture-candidate", None
    if not candidate.accepted or validation is None:
        return "not-measured", None
    if candidate.observable_start_sample > 0 or candidate.outline_preexisting:
        return "left-censored", None
    if candidate.portal_emergence_sample is None:
        return "measured-no-germination-event", None
    return "measured", candidate.portal_emergence_sample


def _refine_native_pollen_center(
    gray: np.ndarray,
    *,
    maximum_offset_px: float = 8.0,
) -> tuple[np.ndarray, bool]:
    """Refine a predicted pollen center with a nearby native circular ring."""

    image = np.asarray(gray, dtype=np.uint8)
    height, width = image.shape
    center_xy = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0])
    circles = cv.HoughCircles(
        cv.GaussianBlur(image, (5, 5), 1.1),
        cv.HOUGH_GRADIENT,
        dp=1.0,
        minDist=20.0,
        param1=70.0,
        param2=17.0,
        minRadius=10,
        maxRadius=20,
    )
    if circles is None:
        return np.zeros(2, dtype=np.float32), False
    values = circles[0]
    distances = np.linalg.norm(values[:, :2] - center_xy[None], axis=1)
    index = int(np.argmin(distances))
    if distances[index] > maximum_offset_px:
        return np.zeros(2, dtype=np.float32), False
    offset_xy = values[index, :2] - center_xy
    return offset_xy[::-1].astype(np.float32), True


def _owner_track_visibility(
    scores: np.ndarray,
    *,
    observable_start_sample: int,
    baseline_samples: int = 8,
    minimum_absolute_score: float = 0.30,
    minimum_relative_score: float = 0.55,
) -> np.ndarray:
    """Keep an owner visible while its tracked template remains recognizable."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("owner tracking scores must be one-dimensional")
    baseline_end = min(
        len(values), observable_start_sample + max(1, baseline_samples)
    )
    if observable_start_sample < 0 or observable_start_sample >= baseline_end:
        raise ValueError("observable owner baseline must fit within the timeline")
    baseline = float(np.median(values[observable_start_sample:baseline_end]))
    threshold = max(minimum_absolute_score, minimum_relative_score * baseline)
    visible = values >= threshold
    visible[:observable_start_sample] = False
    return visible


def _detect_rupture_like_transition(
    tracking_scores: np.ndarray,
    length_timeline_px: np.ndarray,
    *,
    observable_start_sample: int,
    onset_sample: int | None,
    maximum_short_event_length_px: float = 45.0,
) -> int | None:
    """Flag a short mass that appears immediately after owner appearance collapses."""

    if onset_sample is None:
        return None
    visible = _owner_track_visibility(
        tracking_scores,
        observable_start_sample=observable_start_sample,
    )
    lengths = np.asarray(length_timeline_px, dtype=np.float32)
    if lengths.shape != visible.shape or lengths[-1] > maximum_short_event_length_px:
        return None
    missing = ~visible
    sample = observable_start_sample
    while sample < len(missing):
        if not missing[sample]:
            sample += 1
            continue
        start = sample
        while sample < len(missing) and missing[sample]:
            sample += 1
        if sample - start < 2:
            continue
        if sample <= onset_sample <= sample + 3:
            rapid_end = min(len(lengths), onset_sample + 3)
            if np.max(lengths[:rapid_end]) >= 0.80 * max(float(lengths[-1]), 1.0):
                return start
    return None


def _native_directional_portal_profile(
    gray: np.ndarray,
    *,
    angle_count: int = 72,
    maximum_radius_px: int = 48,
) -> np.ndarray:
    """Measure tube-like contrast leaving every angle of a pollen boundary."""

    image = np.asarray(gray, dtype=np.float32)
    fine = cv.GaussianBlur(image, (0, 0), 0.8)
    background = cv.GaussianBlur(image, (0, 0), 4.0)
    high_pass = np.abs(fine - background)
    height, width = image.shape
    polar = cv.warpPolar(
        high_pass,
        (maximum_radius_px, angle_count),
        ((width - 1) / 2.0, (height - 1) / 2.0),
        maximum_radius_px,
        cv.WARP_POLAR_LINEAR + cv.INTER_LINEAR,
    )
    wrapped = np.vstack((polar[-2:], polar, polar[:2]))
    smoothed = cv.GaussianBlur(wrapped, (1, 5), 0.0)[2:-2]
    return (
        np.mean(smoothed[:, 15:26], axis=1)
        + 0.45 * np.mean(smoothed[:, 26:41], axis=1)
    ).astype(np.float32)


def _detect_directional_portal_emergence(
    profiles: np.ndarray,
    visible: np.ndarray,
    *,
    observable_start_sample: int,
    root_angle_radians: float,
) -> tuple[int | None, int | None, np.ndarray]:
    """Follow one mature portal backward until persistent evidence disappears."""

    values = np.asarray(profiles, dtype=np.float32)
    owner_visible = np.asarray(visible, dtype=bool)
    sample_count, angle_count = values.shape
    baseline_end = observable_start_sample + 4
    noise_end = observable_start_sample + 8
    if noise_end >= sample_count:
        return None, None, np.zeros(sample_count, dtype=np.float32)
    baseline = np.median(values[observable_start_sample:baseline_end], axis=0)
    noise_values = values[observable_start_sample:noise_end]
    noise = 1.4826 * np.median(
        np.abs(noise_values - np.median(noise_values, axis=0)[None]), axis=0
    )
    noise_floor = max(0.35, float(np.median(noise)))
    standardized = (values - baseline[None]) / (noise[None] + noise_floor)

    tail_start = max(observable_start_sample, sample_count - 20)
    tail_indices = np.flatnonzero(owner_visible[tail_start:]) + tail_start
    if len(tail_indices) < 3:
        return None, None, np.zeros(sample_count, dtype=np.float32)
    late = np.median(standardized[tail_indices], axis=0)
    root_bin = int(
        np.rint((root_angle_radians % (2.0 * math.pi)) / (2.0 * math.pi) * angle_count)
    ) % angle_count
    indices = np.arange(angle_count)
    root_distance = np.minimum(
        np.abs(indices - root_bin), angle_count - np.abs(indices - root_bin)
    )
    state = int(np.argmax(late - 0.05 * root_distance))
    states = np.empty(sample_count, dtype=np.int16)
    path_score = np.zeros(sample_count, dtype=np.float32)
    for sample in range(sample_count - 1, observable_start_sample - 1, -1):
        if owner_visible[sample]:
            options = np.mod(np.arange(state - 2, state + 3), angle_count)
            distance = np.minimum(
                np.abs(options - state), angle_count - np.abs(options - state)
            )
            option_scores = standardized[sample, options] - 0.20 * distance
            state = int(options[np.argmax(option_scores)])
            path_score[sample] = standardized[sample, state]
        states[sample] = state
    states[:observable_start_sample] = states[observable_start_sample]
    late_level = float(np.median(path_score[tail_indices]))
    if late_level < 2.5:
        angles = states.astype(np.float32) * (2.0 * math.pi / angle_count)
        return None, None, angles
    threshold = max(2.5, 0.15 * late_level)
    active = path_score >= threshold
    onset = None
    for sample in range(observable_start_sample + 2, sample_count - 8):
        confirmation = active[sample : sample + 5]
        future = active[sample : min(sample + 20, sample_count)]
        if (
            active[sample]
            and np.count_nonzero(confirmation) >= 4
            and np.mean(future) >= 0.70
            and np.count_nonzero(owner_visible[sample : sample + 5]) >= 3
        ):
            onset = sample
            break
    last_dormant = None
    if onset is not None:
        weak_threshold = max(1.25, 0.05 * late_level)
        weak_active = path_score >= weak_threshold
        weak_onset = onset
        for sample in range(observable_start_sample + 2, onset + 1):
            confirmation = weak_active[sample : sample + 5]
            future = weak_active[sample : min(sample + 20, sample_count)]
            if (
                weak_active[sample]
                and np.count_nonzero(confirmation) >= 3
                and np.mean(future) >= 0.60
                and np.count_nonzero(owner_visible[sample : sample + 5]) >= 2
            ):
                weak_onset = sample
                break
        last_dormant = max(observable_start_sample, weak_onset - 1)
    angles = states.astype(np.float32) * (2.0 * math.pi / angle_count)
    return onset, last_dormant, angles


def refine_native_portal_emergence(
    movie: Path,
    source_frames: np.ndarray,
    analysis_scale: float,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    center_scores: np.ndarray,
    selected: list[PortalCandidate],
    *,
    crop_size: int = 96,
) -> None:
    """Measure owner-connected emergence in native pollen-centered coordinates."""

    accepted = [candidate for candidate in selected if candidate.accepted]
    if not accepted:
        return
    if crop_size % 2 or crop_size < 48:
        raise ValueError("native pollen crop size must be even and at least 48 pixels")
    config = PollenOutlineConfig(pollen_radius_px=15.0)
    extents = {
        candidate.owner_track_id: np.zeros(
            (len(source_frames), len(config.darkness_thresholds)), dtype=np.float32
        )
        for candidate in accepted
    }
    profiles = {
        candidate.owner_track_id: np.zeros((len(source_frames), 72), dtype=np.float32)
        for candidate in accepted
    }
    visible = {
        candidate.owner_track_id: _owner_track_visibility(
            center_scores[candidate.owner_index],
            observable_start_sample=candidate.observable_start_sample,
        )
        for candidate in accepted
    }
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie} for native portal analysis")
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode native source frame {source_frame}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        for candidate in accepted:
            if sample < candidate.observable_start_sample:
                continue
            center_yx = (
                center_tracks_yx[candidate.owner_index, sample]
                + shifts_xy[sample][::-1]
            ) / analysis_scale
            crop = cv.getRectSubPix(
                gray,
                (crop_size, crop_size),
                (float(center_yx[1]), float(center_yx[0])),
            )
            center_offset_yx, ring_visible = _refine_native_pollen_center(crop)
            if ring_visible:
                center_yx = center_yx + center_offset_yx
                crop = cv.getRectSubPix(
                    gray,
                    (crop_size, crop_size),
                    (float(center_yx[1]), float(center_yx[0])),
                )
            profiles[candidate.owner_track_id][sample] = (
                _native_directional_portal_profile(crop)
            )
            observation = measure_pollen_outline(crop, config)
            extents[candidate.owner_track_id][sample] = observation.extents_px
        if sample == 0 or (sample + 1) % 50 == 0:
            print(f"[v25 native portal] {sample + 1}/{len(source_frames)}", flush=True)
    capture.release()
    for candidate in accepted:
        root_vector = np.mean(
            (candidate.local_path_yx - candidate.local_center_yx)[:4], axis=0
        )
        root_angle = math.atan2(float(root_vector[0]), float(root_vector[1]))
        onset, last_dormant, angles = _detect_directional_portal_emergence(
            profiles[candidate.owner_track_id],
            visible[candidate.owner_track_id],
            observable_start_sample=candidate.observable_start_sample,
            root_angle_radians=root_angle,
        )
        if onset is None and candidate.native_prefix_onset_sample is not None:
            onset = candidate.native_prefix_onset_sample
            last_dormant = candidate.native_prefix_last_dormant_sample
        candidate.portal_emergence_sample = onset
        candidate.portal_last_dormant_sample = last_dormant
        candidate.portal_angle_radians = angles
        candidate.native_owner_visible = visible[candidate.owner_track_id]
        baseline_end = candidate.observable_start_sample + config.warmup_samples
        baseline = np.max(
            extents[candidate.owner_track_id][
                candidate.observable_start_sample:baseline_end
            ],
            axis=0,
        )
        candidate.outline_baseline_extents_px = baseline
        candidate.outline_preexisting = bool(
            np.median(baseline)
            >= config.preexisting_extent_factor * config.pollen_radius_px
        )
        extension = np.maximum(
            np.median(extents[candidate.owner_track_id] - baseline[None], axis=1),
            0.0,
        )
        if onset is not None:
            extension[:onset] = 0.0
            extension[onset:] = np.maximum.accumulate(extension[onset:])
        candidate.outline_extension_px = extension
        directional_tips = np.zeros((len(source_frames), 2), dtype=np.float32)
        if onset is not None:
            radii = config.pollen_radius_px + np.maximum(extension, 2.0)
            directional_tips[:, 0] = np.sin(angles) * radii
            directional_tips[:, 1] = np.cos(angles) * radii
        candidate.outline_tip_offsets_yx = directional_tips
        owner_visibility = visible[candidate.owner_track_id]
        observable_visibility = owner_visibility[candidate.observable_start_sample :]
        candidate.native_owner_visible_fraction = float(
            np.mean(observable_visibility)
        )
        longest_gap = current_gap = 0
        for missing in ~observable_visibility:
            current_gap = current_gap + 1 if missing else 0
            longest_gap = max(longest_gap, current_gap)
        candidate.longest_native_owner_visibility_gap = longest_gap
        validation = candidate.validation
        if candidate.native_prefix_lengths_px is not None:
            length_timeline_px = candidate.native_prefix_lengths_px
        elif validation is not None:
            length_timeline_px = validation.prefix_lengths_px / analysis_scale
        else:
            length_timeline_px = np.zeros(len(source_frames), dtype=np.float32)
        rupture_sample = _detect_rupture_like_transition(
            center_scores[candidate.owner_index],
            length_timeline_px,
            observable_start_sample=candidate.observable_start_sample,
            onset_sample=onset,
        )
        if rupture_sample is not None:
            candidate.rupture_candidate_sample = rupture_sample
            candidate.measurement_scope = "rupture-candidate"
            candidate.accepted = False
            candidate.reason = "rupture-candidate-owner-appearance-discontinuity"
            continue
        if onset is None and longest_gap >= max(40, round(0.35 * len(source_frames))):
            candidate.accepted = False
            candidate.reason = f"{candidate.reason};prolonged-owner-identity-loss"


def mature_centerline_start(candidate: PortalCandidate) -> int | None:
    """Find when the low-resolution path grows beyond its outline-onset plateau."""

    validation = candidate.validation
    if candidate.native_prefix_onset_sample is not None:
        return candidate.native_prefix_onset_sample
    onset = (
        candidate.portal_last_dormant_sample + 1
        if candidate.portal_last_dormant_sample is not None
        else candidate.portal_emergence_sample
    )
    if validation is None or onset is None:
        return candidate.observable_start_sample if candidate.accepted else None
    baseline = float(validation.prefix_lengths_px[onset])
    later = np.flatnonzero(
        validation.prefix_lengths_px[onset:] >= baseline + 0.5
    )
    return int(onset + later[0]) if len(later) else None


def tube_length_at_sample(
    candidate: PortalCandidate,
    sample: int,
    native_scale: float,
) -> float:
    """Combine native early protrusion length with the mature centerline length."""

    validation = candidate.validation
    status, germination_sample = measurement_disposition(candidate)
    if validation is None or not candidate.accepted:
        return 0.0
    centerline_length = float(
        np.max(validation.prefix_lengths_px[: sample + 1])
    ) * native_scale
    outline_length = (
        max(
            0.0,
            float(np.max(candidate.outline_extension_px[: sample + 1])),
        )
        if candidate.outline_extension_px is not None
        else 0.0
    )
    native_prefix_length = (
        float(candidate.native_prefix_lengths_px[sample])
        if candidate.native_prefix_lengths_px is not None
        else 0.0
    )
    if status == "measured" and germination_sample is not None:
        if sample < germination_sample:
            return 0.0
        mature_start = mature_centerline_start(candidate)
        if mature_start is None or sample < mature_start:
            return max(outline_length, native_prefix_length)
    return max(centerline_length, outline_length, native_prefix_length)


def write_outputs(
    output: Path,
    selected: list[PortalCandidate],
    source_frames: np.ndarray,
    source_fps: float,
    native_scale: float,
    center_tracks_yx: np.ndarray,
    shifts_xy: np.ndarray,
    late_samples: int,
) -> None:
    """Write one owner summary and one complete time-series measurement table."""

    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollen_id",
                "status",
                "germination_source_frame",
                "germination_time_minutes",
                "last_dormant_source_frame",
                "last_dormant_time_minutes",
                "final_length_px",
                "measurement_scope",
                "growth_discontinuity_source_frame",
                "growth_discontinuity_time_minutes",
                "rupture_candidate_source_frame",
                "rupture_candidate_time_minutes",
                "geometry_mode",
                "temporal_mode",
                "reason",
            ]
        )
        for candidate in selected:
            validation = candidate.validation
            status, onset = measurement_disposition(candidate)
            final_length = tube_length_at_sample(
                candidate, len(source_frames) - 1, native_scale
            )
            writer.writerow(
                [
                    candidate.owner_track_id,
                    status,
                    "" if onset is None else int(source_frames[onset]),
                    ""
                    if onset is None
                    else round(float(source_frames[onset] / source_fps / 60.0), 4),
                    ""
                    if candidate.portal_last_dormant_sample is None
                    else int(source_frames[candidate.portal_last_dormant_sample]),
                    ""
                    if candidate.portal_last_dormant_sample is None
                    else round(
                        float(
                            source_frames[candidate.portal_last_dormant_sample]
                            / source_fps
                            / 60.0
                        ),
                        4,
                    ),
                    round(final_length, 4),
                    candidate.measurement_scope,
                    ""
                    if candidate.growth_discontinuity_sample is None
                    else int(source_frames[candidate.growth_discontinuity_sample]),
                    ""
                    if candidate.growth_discontinuity_sample is None
                    else round(
                        float(
                            source_frames[candidate.growth_discontinuity_sample]
                            / source_fps
                            / 60.0
                        ),
                        4,
                    ),
                    ""
                    if candidate.rupture_candidate_sample is None
                    else int(source_frames[candidate.rupture_candidate_sample]),
                    ""
                    if candidate.rupture_candidate_sample is None
                    else round(
                        float(
                            source_frames[candidate.rupture_candidate_sample]
                            / source_fps
                            / 60.0
                        ),
                        4,
                    ),
                    candidate.appearance_mode,
                    candidate.temporal_mode or "",
                    candidate.reason,
                ]
            )
    with (output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollen_id",
                "sample_index",
                "source_frame",
                "time_minutes",
                "tube_length_px",
                "status",
                "measurement_scope",
            ]
        )
        for candidate in selected:
            status, _ = measurement_disposition(candidate)
            for sample, source_frame in enumerate(source_frames):
                length = tube_length_at_sample(candidate, sample, native_scale)
                writer.writerow(
                    [
                        candidate.owner_track_id,
                        sample,
                        int(source_frame),
                        round(float(source_frame / source_fps / 60.0), 4),
                        round(length, 4),
                        status,
                        candidate.measurement_scope,
                    ]
                )
    late_centers = np.median(center_tracks_yx[:, -late_samples:], axis=1)
    final_drift_yx = shifts_xy[-1][::-1]
    with (output / "centerlines.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollen_id",
                "status",
                "measurement_scope",
                "point_index",
                "final_source_y_px",
                "final_source_x_px",
                "arc_length_px",
                "geometry_mode",
                "temporal_mode",
            ]
        )
        for candidate in selected:
            status, _ = measurement_disposition(candidate)
            global_path = (
                candidate.local_path_yx
                - candidate.local_center_yx
                + late_centers[candidate.owner_index]
                + final_drift_yx
            )
            arc = np.concatenate(
                (
                    [0.0],
                    np.cumsum(np.linalg.norm(np.diff(global_path, axis=0), axis=1)),
                )
            )
            for point_index, (point, distance) in enumerate(zip(global_path, arc)):
                writer.writerow(
                    [
                        candidate.owner_track_id,
                        status,
                        candidate.measurement_scope,
                        point_index,
                        round(float(point[0] * native_scale), 4),
                        round(float(point[1] * native_scale), 4),
                        round(float(distance * native_scale), 4),
                        candidate.appearance_mode,
                        candidate.temporal_mode or "",
                    ]
                )


def render_review(
    movie: Path,
    output: Path,
    source_frames: np.ndarray,
    source_fps: float,
    native_width: int,
    native_height: int,
    analysis_width: int,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    owner_ids: list[int],
    owner_start_samples: list[int],
    selected: list[PortalCandidate],
) -> Path:
    """Render only causally accepted growth, never discarded hypotheses."""

    render_width = 960
    render_height = round(native_height * render_width / native_width)
    header = 50
    render_scale = render_width / analysis_width
    analysis_scale = analysis_width / native_width
    temporary = output / "causal_portal_raw.mp4"
    final = output / "causal_portal.mov"
    writer = cv.VideoWriter(
        str(temporary),
        cv.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (render_width, render_height + header),
    )
    capture = cv.VideoCapture(str(movie))
    accepted = [candidate for candidate in selected if candidate.accepted]
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not render source frame {source_frame}")
        frame = cv.resize(frame, (render_width, render_height), interpolation=cv.INTER_AREA)
        canvas = np.zeros((render_height + header, render_width, 3), dtype=np.uint8)
        canvas[header:] = frame
        drift_yx = shifts_xy[sample][::-1]
        for owner, track_id in enumerate(owner_ids):
            if sample < owner_start_samples[owner]:
                continue
            center = (center_tracks_yx[owner, sample] + drift_yx) * render_scale
            cv.circle(
                canvas,
                (round(float(center[1])), round(float(center[0] + header))),
                round(6.0 * render_scale),
                (255, 150, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                canvas,
                f"P{track_id:02d}",
                (round(float(center[1] + 7)), round(float(center[0] + header - 5))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.34,
                (40, 80, 20),
                1,
                cv.LINE_AA,
            )
        active_count = 0
        for candidate in accepted:
            validation = candidate.validation
            if validation is None:
                continue
            status, germination_sample = measurement_disposition(candidate)
            if (
                status == "measured"
                and germination_sample is not None
                and sample < germination_sample
            ):
                continue
            mature_start = mature_centerline_start(candidate)
            use_outline_tip = (
                status == "measured"
                and germination_sample is not None
                and sample >= germination_sample
                and (mature_start is None or sample < mature_start)
                and candidate.outline_tip_offsets_yx is not None
            )
            if use_outline_tip:
                tip_offset_native = candidate.outline_tip_offsets_yx[sample]
                tip_radius = float(np.linalg.norm(tip_offset_native))
                if tip_radius > 15.0:
                    center = (
                        center_tracks_yx[candidate.owner_index, sample] + drift_yx
                    )
                    direction = tip_offset_native / tip_radius
                    outline_path = np.vstack(
                        (
                            center + direction * (15.0 * analysis_scale),
                            center + tip_offset_native * analysis_scale,
                        )
                    )
                    points = np.rint(
                        outline_path[:, ::-1] * render_scale
                    ).astype(np.int32)
                    points[:, 1] += header
                    cv.polylines(
                        canvas, [points], False, (255, 0, 220), 3, cv.LINE_AA
                    )
                    cv.circle(
                        canvas, tuple(points[-1]), 4, (0, 255, 0), -1, cv.LINE_AA
                    )
                    active_count += 1
                    continue
            end = (
                int(candidate.native_prefix_end_indices[sample])
                if candidate.native_prefix_end_indices is not None
                else int(np.maximum.accumulate(validation.prefix_end_indices)[sample])
            )
            if end < 1:
                continue
            active_count += 1
            path = candidate.local_path_yx[: end + 1].copy()
            normal = candidate.normal_yx[: end + 1]
            if candidate.normal_offsets_px is None:
                continue
            offsets = candidate.normal_offsets_px[sample, : end + 1].astype(np.float32)
            if len(offsets) >= 5:
                offsets = cv.GaussianBlur(offsets[:, None], (1, 5), 0).ravel()
            path += offsets[:, None] * normal
            path += (
                center_tracks_yx[candidate.owner_index, sample]
                - candidate.local_center_yx
                + drift_yx
            )[None]
            points = np.rint(path[:, ::-1] * render_scale).astype(np.int32)
            points[:, 1] += header
            cv.polylines(canvas, [points], False, (255, 0, 220), 3, cv.LINE_AA)
            cv.circle(canvas, tuple(points[-1]), 4, (0, 255, 0), -1, cv.LINE_AA)
        minutes = source_frame / source_fps / 60.0
        cv.putText(
            canvas,
            f"CAUSAL PORTAL | {len(owner_ids)} POLLEN | source {source_frame}/{int(source_frames[-1])} | {minutes:.1f} min",
            (14, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.60,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            f"measured {len(accepted)} | growing {active_count}",
            (render_width - 220, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 150, 235),
            1,
            cv.LINE_AA,
        )
        writer.write(canvas)
    capture.release()
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final),
        ],
        check=True,
    )
    temporary.unlink()
    return final


def main() -> None:
    """Run owner-relative geometry discovery and causal portal validation."""

    args = parse_args()
    if args.crop_size % 2:
        raise ValueError("crop size must be even")
    selected_ids = (
        None
        if not args.track_ids
        else {int(value) for value in args.track_ids.split(",")}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_frames, fps, frame_count, native_width, native_height = movie_schedule(
        args.movie, args.sample_interval_s
    )
    analysis_scale = args.width / native_width
    frames = load_grayscale_samples(
        args.movie,
        source_frames,
        args.width,
        native_width,
        native_height,
    )
    aligned, shifts_xy, registration_response = stabilize_translations(frames)
    owner_radius = 15.0 * analysis_scale
    seeds, _ = load_portal_owner_seeds(
        args.identity_report,
        source_frames,
        analysis_scale,
        shifts_xy,
        include_persistent_late=args.owner_policy == "persistent",
        minimum_late_semantic_observations=(
            args.minimum_late_semantic_observations
        ),
    )
    seed_centers = np.asarray([seed.center_yx for seed in seeds])
    seed_samples = np.asarray([seed.seed_sample for seed in seeds])
    center_tracks, center_scores = track_pollen_centers_from_seeds(
        aligned,
        seed_centers,
        seed_samples,
        template_radius_px=5,
        search_radius_px=5,
        minimum_score=0.25,
    )
    seeds, center_tracks, center_scores, suppressed_identity_fragments = (
        retain_distinct_owner_tracks(
            seeds,
            center_tracks,
            center_scores,
            owner_radius_px=owner_radius,
        )
    )
    seeds, center_tracks, center_scores, rejected_late_owner_candidates = (
        retain_native_pollen_owners(
            args.movie,
            source_frames,
            analysis_scale,
            shifts_xy,
            seeds,
            center_tracks,
            center_scores,
        )
    )
    if selected_ids is not None:
        retained_indices = [
            index for index, seed in enumerate(seeds) if seed.track_id in selected_ids
        ]
        seeds = [seeds[index] for index in retained_indices]
        center_tracks = center_tracks[retained_indices]
        center_scores = center_scores[retained_indices]
        missing_ids = selected_ids - {seed.track_id for seed in seeds}
        if missing_ids:
            raise ValueError(f"requested pollen IDs were not retained: {sorted(missing_ids)}")
    modes = {
        "both": ("bright", "dark"),
        "all": ("bright", "dark", "generic"),
    }.get(args.interior_polarity, (args.interior_polarity,))
    feature_configs = {
        mode: PhaseContrastRibbonConfig(
            interior_polarity=mode,
            orientation_count=16,
            half_widths_px=(1.0, 1.5, 2.0, 2.5, 3.0),
            tangent_samples_px=(-2.0, -1.0, 0.0, 1.0, 2.0),
            fine_sigma_px=0.6,
            background_sigma_px=3.5,
            structure_sigma_px=1.1,
            orientation_concentration=4.0,
            wall_pair_weight=1.0,
            matching_center_weight=0.65,
            opposite_center_penalty=0.75,
            wall_asymmetry_penalty=0.45,
        )
        for mode in modes
        if mode != "generic"
    }
    if "generic" in modes:
        feature_configs["generic"] = OrientationScoreConfig(
            orientation_count=16,
            half_widths_px=(1.0, 1.5, 2.0, 2.5, 3.0),
            tangent_samples_px=(-2.0, -1.0, 0.0, 1.0, 2.0),
            wall_sigma_px=0.6,
            background_sigma_px=3.5,
            structure_sigma_px=1.1,
            orientation_concentration=4.0,
            center_darkness_penalty=0.15,
            asymmetry_penalty=0.25,
        )
    checkpoint_signature = {
        "movie": str(args.movie.resolve()),
        "identity_report": str(args.identity_report.resolve()),
        "source_frames": source_frames.tolist(),
        "analysis_width": args.width,
        "late_samples": args.late_samples,
        "crop_size": args.crop_size,
        "modes": list(modes),
        "track_ids": sorted(selected_ids) if selected_ids is not None else None,
        "maximum_proposals": args.maximum_proposals,
        "maximum_candidates": args.maximum_candidates,
        "owner_policy": args.owner_policy,
        "owner_track_ids": [seed.track_id for seed in seeds],
        "revision": CAUSAL_EVIDENCE_REVISION,
    }
    replay_checkpoint = args.output / "replay_checkpoint.pkl"
    if replay_checkpoint.exists():
        replay_start, candidates = _load_replay_checkpoint(
            replay_checkpoint, checkpoint_signature
        )
        print(
            f"[v25 resume] temporal replay resumes at {replay_start}/{len(aligned)}",
            flush=True,
        )
    else:
        atlases = {
            mode: build_owner_late_atlases(
                aligned,
                center_tracks,
                config,
                late_samples=args.late_samples,
                crop_size=args.crop_size,
            )
            for mode, config in feature_configs.items()
        }
        candidates = []
        for owner, seed in enumerate(seeds):
            traced = []
            for mode in modes:
                traced.extend(
                    trace_owner_candidates(
                        owner,
                        seed.track_id,
                        seed.observable_start_sample,
                        mode,
                        atlases[mode][owner],
                        center_tracks,
                        owner_radius_px=owner_radius,
                        crop_size=args.crop_size,
                        late_samples=args.late_samples,
                        sample_count=len(source_frames),
                        support_modes=tuple(modes),
                        maximum_proposals=args.maximum_proposals,
                        maximum_candidates=args.maximum_candidates,
                    )
                )
            candidates.extend(traced)
            print(
                f"[v25 geometry] {owner + 1}/{len(seeds)} P{seed.track_id:02d}: {len(traced)} distinct candidates",
                flush=True,
            )
        replay_start = 0
        _write_replay_checkpoint(
            replay_checkpoint,
            next_sample=0,
            candidates=candidates,
            signature=checkpoint_signature,
        )
    collect_temporal_evidence(
        aligned,
        center_tracks,
        candidates,
        feature_configs,
        start_sample=replay_start,
        checkpoint_path=replay_checkpoint,
        checkpoint_signature=checkpoint_signature,
    )
    validate_candidates(candidates, owner_radius)
    score_native_candidate_geometry(
        args.movie,
        source_frames,
        analysis_scale,
        shifts_xy,
        center_tracks,
        candidates,
        late_samples=args.late_samples,
    )
    selected = select_native_supported_candidates(
        candidates,
        owner_radius,
        analysis_scale,
        center_tracks,
        late_samples=args.late_samples,
    )
    multiscale_checkpoint = args.output / "multiscale_replay_checkpoint.pkl"
    multiscale_signature = {
        **checkpoint_signature,
        "recovery_width": args.recovery_width,
        "revision": MULTISCALE_EVIDENCE_REVISION,
    }
    if multiscale_checkpoint.exists():
        recovery_start, recovery_candidates = _load_replay_checkpoint(
            multiscale_checkpoint, multiscale_signature
        )
        if recovery_start != len(aligned):
            raise ValueError("multiscale replay checkpoint is incomplete")
        print(
            f"[v25 multiscale resume] {len(recovery_candidates)} candidates",
            flush=True,
        )
    else:
        recovery_candidates = generate_multiscale_recovery_candidates(
            args.movie,
            source_frames,
            native_width,
            native_height,
            analysis_scale,
            shifts_xy,
            center_tracks,
            seeds,
            selected,
            feature_configs,
            analysis_crop_size=args.crop_size,
            owner_radius_px=owner_radius,
            late_samples=args.late_samples,
            recovery_width=args.recovery_width,
            maximum_proposals=args.maximum_proposals,
            maximum_candidates=args.maximum_candidates,
        )
    if recovery_candidates and not multiscale_checkpoint.exists():
        collect_temporal_evidence(
            aligned,
            center_tracks,
            recovery_candidates,
            feature_configs,
        )
        _write_replay_checkpoint(
            multiscale_checkpoint,
            next_sample=len(aligned),
            candidates=recovery_candidates,
            signature=multiscale_signature,
        )
    if recovery_candidates:
        validate_candidates(recovery_candidates, owner_radius)
        score_native_candidate_geometry(
            args.movie,
            source_frames,
            analysis_scale,
            shifts_xy,
            center_tracks,
            recovery_candidates,
            late_samples=args.late_samples,
        )
        candidates.extend(recovery_candidates)
        selected = select_native_supported_candidates(
            candidates,
            owner_radius,
            analysis_scale,
            center_tracks,
            late_samples=args.late_samples,
        )
    recover_native_temporal_prefixes(
        args.movie,
        source_frames,
        analysis_scale,
        shifts_xy,
        center_tracks,
        selected,
        owner_radius_px=owner_radius,
    )
    truncate_discontinuous_growth(selected, analysis_scale)
    selected.sort(key=lambda item: item.owner_track_id)
    refine_native_portal_emergence(
        args.movie,
        source_frames,
        analysis_scale,
        shifts_xy,
        center_tracks,
        center_scores,
        selected,
    )
    duplicate_resolutions = resolve_cross_owner_conflicts(
        selected, center_tracks, args.late_samples
    )
    for candidate in selected:
        if not candidate.accepted and candidate.rupture_candidate_sample is None:
            candidate.measurement_scope = "none"
    write_outputs(
        args.output,
        selected,
        source_frames,
        fps,
        1.0 / analysis_scale,
        center_tracks,
        shifts_xy,
        args.late_samples,
    )
    movie = render_review(
        args.movie,
        args.output,
        source_frames,
        fps,
        native_width,
        native_height,
        args.width,
        shifts_xy,
        center_tracks,
        [seed.track_id for seed in seeds],
        [seed.observable_start_sample for seed in seeds],
        selected,
    )
    report = {
        "prototype": "v25_causal_portal",
        "revision": CAUSAL_PORTAL_REVISION,
        "method": "native-supported-owner-path-plus-causal-connected-prefix",
        "temporal_support": "angle-continuous-native-portal-plus-deformable-path-prefix",
        "resumable_replay": True,
        "input_movie": str(args.movie),
        "identity_report": str(args.identity_report),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "source_frames": source_frames.tolist(),
        "analysis_width": args.width,
        "owner_count": len(seeds),
        "owner_policy": args.owner_policy,
        "suppressed_identity_fragments": suppressed_identity_fragments,
        "rejected_late_owner_candidates": rejected_late_owner_candidates,
        "candidate_count": len(candidates),
        "multiscale_recovery_candidate_count": len(recovery_candidates),
        "multiscale_recovery_width": args.recovery_width,
        "measured_owner_ids": [
            candidate.owner_track_id for candidate in selected if candidate.accepted
        ],
        "duplicate_resolutions": duplicate_resolutions,
        "median_pollen_match_score": float(np.median(center_scores)),
        "median_registration_response": float(np.median(registration_response)),
        "paths": [
            {
                "owner_track_id": candidate.owner_track_id,
                "accepted": candidate.accepted,
                "appearance_mode": candidate.appearance_mode,
                "temporal_mode": candidate.temporal_mode,
                "reason": candidate.reason,
                "germination_status": measurement_disposition(candidate)[0],
                "measurement_scope": candidate.measurement_scope,
                "growth_discontinuity_sample": (
                    candidate.growth_discontinuity_sample
                ),
                "growth_discontinuity_length_px": (
                    candidate.growth_discontinuity_length_px
                ),
                "rupture_candidate_sample": candidate.rupture_candidate_sample,
                "combined_score": candidate.combined_score,
                "paired_fraction": candidate.paired_fraction,
                "radial_extension_native_px": candidate.radial_extension_px
                / analysis_scale,
                "radial_openness_fraction": candidate.radial_openness_fraction,
                "native_support_fraction": candidate.native_support_fraction,
                "native_supported_length_px": candidate.native_supported_length_px,
                "native_mean_support": candidate.native_mean_support,
                "native_arc_length_px": candidate.native_arc_length_px,
                "native_temporal_completion_fraction": (
                    candidate.native_temporal_completion_fraction
                ),
                "native_owner_visible_fraction": (
                    candidate.native_owner_visible_fraction
                ),
                "longest_native_owner_visibility_gap": (
                    candidate.longest_native_owner_visibility_gap
                ),
                "minimum_foreign_owner_distance_native_px": (
                    candidate.minimum_foreign_owner_distance_px / analysis_scale
                ),
                "native_prefix_recovery_onset_sample": (
                    candidate.native_prefix_onset_sample
                ),
                "native_prefix_recovery_final_length_px": (
                    float(candidate.native_prefix_lengths_px[-1])
                    if candidate.native_prefix_lengths_px is not None
                    else None
                ),
                "first_visible_sample": candidate.validation.first_connected_sample
                if candidate.validation and candidate.accepted
                else None,
                "germination_onset_sample": measurement_disposition(candidate)[1],
                "last_dormant_sample": candidate.portal_last_dormant_sample,
                "first_confirmed_emergence_sample": candidate.portal_emergence_sample,
                "native_outline_preexisting": candidate.outline_preexisting,
                "native_outline_baseline_extents_px": (
                    candidate.outline_baseline_extents_px.tolist()
                    if candidate.outline_baseline_extents_px is not None
                    else None
                ),
                "maximum_native_outline_extension_px": (
                    float(np.max(candidate.outline_extension_px))
                    if candidate.outline_extension_px is not None
                    else None
                ),
                "early_coverage": candidate.validation.early_coverage
                if candidate.validation
                else None,
                "preexisting_coverage": candidate.validation.preexisting_coverage
                if candidate.validation
                else None,
                "late_coverage": candidate.validation.late_coverage
                if candidate.validation
                else None,
                "pre_onset_orphan_fraction": candidate.validation.pre_onset_orphan_fraction
                if candidate.validation
                else None,
                "birth_order_fraction": candidate.validation.birth_order_fraction
                if candidate.validation
                else None,
                "growth_native_px": candidate.validation.growth_px / analysis_scale
                if candidate.validation
                else None,
            }
            for candidate in selected
        ],
        "artifacts": {
            "review_video": str(movie),
            "summary_csv": str(args.output / "summary.csv"),
            "measurements_csv": str(args.output / "measurements.csv"),
            "centerlines_csv": str(args.output / "centerlines.csv"),
        },
        "scope": "Research prototype; quantitative promotion still requires manual centerline validation.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.keep_evidence:
        replay_checkpoint.unlink(missing_ok=True)
        multiscale_checkpoint.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "measured_owner_ids": report["measured_owner_ids"],
                "review_video": str(movie),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
