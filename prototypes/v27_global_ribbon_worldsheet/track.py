"""Resolve complete pollen-tube curves jointly across the sampled movie.

The prototype retains multiple root-connected curves at every time point and
selects their full temporal sequence globally.  It is intentionally separate
from the application and v26 so real-video comparisons remain honest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from prototypes.v24_causal_birth_forest.track import (
    load_grayscale_samples,
    movie_schedule,
    stabilize_translations,
)
from prototypes.v25_causal_portal.track import (
    load_portal_owner_seeds,
    retain_distinct_owner_tracks,
    retain_native_pollen_owners,
)
from prototypes.v26_bidirectional_ribbon.track import (
    feature_builder_for_mode,
    load_mature_centerlines,
    render_review,
    write_outputs,
)
from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds
from tubetracker.temporal_ribbon import (
    TemporalRibbonConfig,
    discover_mature_ribbon_candidates,
    endpoint_radial_efficiency,
    interpolate_sampled_history,
    intersects_foreign_owner,
    radial_excursion_efficiency,
    radial_excursion_px,
    regularize_monotone_history,
    trace_mature_ribbon_worldsheet,
    truncate_at_foreign_owner,
)


REVISION = "v27.6-event-and-boundary-status"
MULTIPOINT_OWNER_REVISION = "v28.5-monotone-material-history"
GLOBAL_ALLOCATION_REVISION = "v28.6.1-minimum-reassignment-branch-allocation"
FIELD_CONTEXT_REVISION = 1


@dataclass(frozen=True)
class FieldContext:
    """Store one validated decoded and registered movie sampling context."""

    source_frames: np.ndarray
    fps: float
    native_width: int
    native_height: int
    aligned_frames: np.ndarray
    shifts_xy: np.ndarray
    registration_response: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse movie, identity, mature-path, sampling, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument(
        "--base-run",
        type=Path,
        help="Optional earlier run whose mature centerlines become extra hypotheses.",
    )
    parser.add_argument(
        "--causal-atlas-run",
        type=Path,
        help="Optional v24 run whose movie-wide birth paths become extra hypotheses.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--field-cache",
        type=Path,
        help="Optional shared decoded and registered frame context.",
    )
    parser.add_argument("--track-ids", help="Optional comma-separated pollen IDs.")
    parser.add_argument(
        "--owner-motion-cache",
        type=Path,
        help="Optional shared CoTracker rigid-body trajectory cache.",
    )
    parser.add_argument("--crop-size", type=int, default=180)
    parser.add_argument("--alternatives", type=int, default=12)
    parser.add_argument(
        "--rediscover-mature",
        action="store_true",
        help="Search the complete pollen rim even when inherited centerlines exist.",
    )
    parser.add_argument("--mature-hypotheses", type=int, default=6)
    parser.add_argument("--selection-stride", type=int, default=4)
    parser.add_argument(
        "--export-alternatives",
        action="store_true",
        help="Save every audited mature-path history for field-level allocation.",
    )
    parser.add_argument(
        "--force-mature-seed",
        help=(
            "Use one exact, audited mature-seed label selected by field-level "
            "allocation; valid only when tracking one pollen ID."
        ),
    )
    parser.add_argument(
        "--dense-timeline",
        action="store_true",
        help="Trace every sampled frame instead of interpolating validated anchors.",
    )
    parser.add_argument(
        "--list-track-ids",
        action="store_true",
        help="Print retained pollen IDs after field preparation, then exit.",
    )
    return parser.parse_args()


def _field_context_manifest(movie: Path, analysis_width: int, interval_s: float) -> dict:
    """Describe the exact source and settings that make a field cache valid."""

    movie = movie.expanduser().resolve()
    stat = movie.stat()
    return {
        "revision": FIELD_CONTEXT_REVISION,
        "movie": str(movie),
        "movie_size": int(stat.st_size),
        "movie_mtime_ns": int(stat.st_mtime_ns),
        "analysis_width": int(analysis_width),
        "sample_interval_s": float(interval_s),
    }


def load_field_context(
    cache_dir: Path,
    movie: Path,
    analysis_width: int,
    interval_s: float,
) -> FieldContext:
    """Load a complete field cache only when source identity and shapes agree."""

    cache_dir = cache_dir.expanduser().resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    expected = _field_context_manifest(movie, analysis_width, interval_s)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"field cache mismatch for {key}")
    source_frames = np.load(cache_dir / "source_frames.npy", mmap_mode="r")
    aligned = np.load(cache_dir / "aligned_frames.npy", mmap_mode="r")
    shifts_xy = np.load(cache_dir / "shifts_xy.npy", mmap_mode="r")
    response = np.load(cache_dir / "registration_response.npy", mmap_mode="r")
    sample_count = len(source_frames)
    expected_height = round(
        int(manifest["native_height"]) * analysis_width / int(manifest["native_width"])
    )
    if source_frames.ndim != 1 or source_frames.dtype.kind not in "iu":
        raise ValueError("field cache source frames are invalid")
    if aligned.shape != (sample_count, expected_height, analysis_width):
        raise ValueError("field cache aligned frames have an invalid shape")
    if aligned.dtype != np.uint8:
        raise ValueError("field cache aligned frames must be uint8")
    if shifts_xy.shape != (sample_count, 2) or response.shape != (sample_count,):
        raise ValueError("field cache registration arrays have an invalid shape")
    return FieldContext(
        source_frames=source_frames,
        fps=float(manifest["fps"]),
        native_width=int(manifest["native_width"]),
        native_height=int(manifest["native_height"]),
        aligned_frames=aligned,
        shifts_xy=shifts_xy,
        registration_response=response,
    )


def _save_array_atomically(path: Path, values: np.ndarray) -> None:
    """Publish one NumPy array only after its complete payload is on disk."""

    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def prepare_field_context(
    cache_dir: Path,
    movie: Path,
    analysis_width: int,
    interval_s: float,
) -> FieldContext:
    """Create or reuse one authoritative decoded and registered field context."""

    cache_dir = cache_dir.expanduser().resolve()
    try:
        return load_field_context(cache_dir, movie, analysis_width, interval_s)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    source_frames, fps, _, native_width, native_height = movie_schedule(
        movie, interval_s
    )
    frames = load_grayscale_samples(
        movie,
        source_frames,
        analysis_width,
        native_width,
        native_height,
    )
    aligned, shifts_xy, response = stabilize_translations(frames)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_array_atomically(cache_dir / "source_frames.npy", source_frames)
    _save_array_atomically(cache_dir / "aligned_frames.npy", aligned)
    _save_array_atomically(cache_dir / "shifts_xy.npy", shifts_xy)
    _save_array_atomically(cache_dir / "registration_response.npy", response)
    manifest = {
        **_field_context_manifest(movie, analysis_width, interval_s),
        "fps": float(fps),
        "native_width": int(native_width),
        "native_height": int(native_height),
        "sample_count": int(len(source_frames)),
    }
    manifest_temporary = cache_dir / "manifest.json.part"
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_temporary.replace(cache_dir / "manifest.json")
    return load_field_context(cache_dir, movie, analysis_width, interval_s)


def load_owner_motion_cache(
    cache_path: Path,
    source_frames: np.ndarray,
    analysis_width: int,
    *,
    shifts_xy: np.ndarray | None = None,
    analysis_scale: float | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Load exact aligned owner tracks from a compatible multi-point run."""

    with np.load(cache_path) as cache:
        required = {
            "source_frames",
            "track_ids",
            "analysis_width",
            "multipoint_aligned_yx",
            "template_aligned_yx",
            "observed",
            "inlier_counts",
        }
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"owner-motion cache is missing: {sorted(missing)}")
        cached_frames = np.asarray(cache["source_frames"], dtype=np.int64)
        cached_ids = np.asarray(cache["track_ids"], dtype=np.int64)
        cached_width = int(np.asarray(cache["analysis_width"]).item())
        detection_constrained = "detection_constrained_aligned_yx" in cache.files
        semantic_span = "semantic_span_aligned_yx" in cache.files
        trajectory_key = (
            "detection_constrained_aligned_yx"
            if detection_constrained
            else (
                "semantic_span_aligned_yx"
                if semantic_span
                else "multipoint_aligned_yx"
            )
        )
        trajectories = np.asarray(
            cache[trajectory_key],
            dtype=np.float64,
        )
        template_trajectories = np.asarray(
            cache["template_aligned_yx"], dtype=np.float64
        )
        source_key = (
            "detection_constrained_source_yx"
            if detection_constrained
            else (
                "semantic_span_source_yx"
                if semantic_span
                else "multipoint_source_yx"
            )
        )
        source_trajectories = (
            np.asarray(cache[source_key], dtype=np.float64)
            if source_key in cache.files
            else None
        )
        source_template_trajectories = (
            np.asarray(cache["template_source_yx"], dtype=np.float64)
            if "template_source_yx" in cache.files
            else None
        )
        observed = np.asarray(cache["observed"], dtype=bool)
        inliers = np.asarray(cache["inlier_counts"], dtype=np.float64)
    if not np.array_equal(cached_frames, np.asarray(source_frames, dtype=np.int64)):
        raise ValueError("owner-motion cache uses a different frame schedule")
    if len(np.unique(cached_ids)) != len(cached_ids):
        raise ValueError("owner-motion cache contains duplicate pollen IDs")
    expected_shape = (len(cached_ids), len(source_frames), 2)
    if trajectories.shape != expected_shape or template_trajectories.shape != expected_shape:
        raise ValueError("owner-motion trajectories have an invalid shape")
    if observed.shape != trajectories.shape[:2] or inliers.shape != observed.shape:
        raise ValueError("owner-motion quality arrays have an invalid shape")
    if cached_width != int(analysis_width):
        if (
            source_trajectories is None
            or source_template_trajectories is None
            or shifts_xy is None
            or analysis_scale is None
        ):
            raise ValueError(
                "cross-resolution owner motion requires source trajectories and registration"
            )
        if np.asarray(shifts_xy).shape != (len(source_frames), 2):
            raise ValueError("registration shifts have an invalid shape")
        trajectories = (
            source_trajectories * float(analysis_scale)
            - np.asarray(shifts_xy)[:, ::-1][None, :, :]
        )
        template_trajectories = (
            source_template_trajectories * float(analysis_scale)
            - np.asarray(shifts_xy)[:, ::-1][None, :, :]
        )
    maximum_inliers = np.maximum(np.max(inliers, axis=1, keepdims=True), 1.0)
    quality = observed * (inliers / maximum_inliers)
    return (
        [int(track_id) for track_id in cached_ids],
        trajectories,
        template_trajectories,
        quality,
    )


def load_causal_atlas_paths(
    run_dir: Path,
    selected_ids: set[int] | None,
    analysis_width: int,
) -> tuple[dict[int, np.ndarray], dict]:
    """Load reusable movie-wide causal paths at the current analysis scale."""

    report = json.loads((run_dir / "report.json").read_text())
    source_width = int(report["analysis_width"])
    if source_width < 1 or analysis_width < 1:
        raise ValueError("causal atlas analysis widths must be positive")
    paths: dict[int, list[tuple[int, float, float]]] = {}
    with (run_dir / "owner_paths.csv").open(newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            track_id = int(row["owner_track_id"])
            accepted = row["accepted"].strip().lower() in {"1", "true", "yes"}
            reusable_review = row.get("reason", "") in {
                "duplicate-branch-claim",
                "maximum-length-censored",
            }
            if (not accepted and not reusable_review) or (
                selected_ids is not None and track_id not in selected_ids
            ):
                continue
            paths.setdefault(track_id, []).append(
                (
                    int(row["path_index"]),
                    float(row["y_analysis_px"]),
                    float(row["x_analysis_px"]),
                )
            )
    scale = analysis_width / source_width
    return (
        {
            track_id: scale
            * np.asarray(
                [(y, x) for _, y, x in sorted(points)],
                dtype=np.float64,
            )
            for track_id, points in paths.items()
            if len(points) >= 2
        },
        report,
    )


def _causal_audit_key(history, worldsheet) -> tuple[float, ...]:
    """Rank a mature seed by growth, persistence, and normalized evidence."""

    onset = history.first_persistent_sample
    if onset is None:
        return (0.0, -float("inf"), -float("inf"), -float("inf"))
    active = [frame for frame in history.frames[onset:] if frame.accepted]
    if not active:
        return (0.0, -float("inf"), -float("inf"), -float("inf"))
    first_length = active[0].length_px
    final_length = history.frames[-1].length_px
    growth_fraction = max(
        0.0,
        (final_length - first_length) / max(final_length, 1e-9),
    )
    support_fraction = len(active) / len(history.frames)
    sustained_growth = growth_fraction * np.sqrt(support_fraction)
    evidence = worldsheet.total_score / max(len(active), 1)
    return (1.0, sustained_growth, support_fraction, evidence)


def _radial_geometry_audit(
    frame,
    owner_radius: float,
    config: TemporalRibbonConfig,
    *,
    contact_censored: bool = False,
) -> tuple[bool, float, float]:
    """Certify a tube that leaves its pollen even when it later curves back."""

    endpoint_efficiency = endpoint_radial_efficiency(frame)
    excursion_radii = radial_excursion_px(frame) / owner_radius
    minimum_excursion = (
        config.minimum_contact_radial_excursion_radii
        if contact_censored
        else config.minimum_radial_excursion_radii
    )
    eligible = (
        frame.length_px > 0.0
        and excursion_radii >= minimum_excursion
        and (
            endpoint_efficiency >= config.minimum_endpoint_radial_efficiency
            or excursion_radii >= config.minimum_curved_radial_excursion_radii
        )
    )
    return eligible, endpoint_efficiency, excursion_radii


def _baseline_departure_audit(
    history,
    owner_radius: float,
    config: TemporalRibbonConfig,
    *,
    contact_censored: bool = False,
) -> tuple[bool, float | None]:
    """Require a claimed preexisting tube to clear the pollen at baseline."""

    if history.first_persistent_sample != 0 or contact_censored:
        return True, None
    excursion_radii = radial_excursion_px(history.frames[0]) / owner_radius
    return (
        excursion_radii
        >= config.minimum_left_censored_baseline_excursion_radii,
        excursion_radii,
    )


def _has_event_certificate(
    active_sample_count: int,
    sustained_growth_score: float,
    geometry_eligible: bool,
    config: TemporalRibbonConfig,
) -> bool:
    """Accept a visible growth event independently of arbitrary score zero."""

    return (
        active_sample_count >= config.absence_persistence
        and np.isfinite(sustained_growth_score)
        and sustained_growth_score >= config.minimum_sustained_growth_score
        and geometry_eligible
    )


def _mature_selection_key(audit: dict, maximum_length_px: float) -> tuple[float, ...]:
    """Balance sustained growth with completeness of the mature geometry."""

    causal_key = tuple(float(value) for value in audit["causal_key"])
    final_length = max(0.0, float(audit["final_length_px"]))
    if not np.isfinite(causal_key[1]):
        return (causal_key[0], -float("inf"), causal_key[2], causal_key[3], final_length)
    length_fraction = final_length / max(maximum_length_px, 1e-9)
    complete_growth = causal_key[1] * np.sqrt(np.clip(length_fraction, 0.0, 1.0))
    return (
        causal_key[0],
        complete_growth,
        causal_key[2],
        causal_key[3],
        final_length,
    )


def _path_exits_field(
    frame,
    center_yx: np.ndarray,
    shift_xy: np.ndarray,
    field_shape: tuple[int, int],
) -> bool:
    """Return whether the distal tip lies outside the source image field."""

    if not frame.accepted or len(frame.path_relative_yx) == 0:
        return False
    source_tip_yx = (
        frame.path_relative_yx[-1] + center_yx + np.asarray(shift_xy)[::-1]
    )
    height, width = field_shape
    return not (
        0.0 <= float(source_tip_yx[0]) < float(height)
        and 0.0 <= float(source_tip_yx[1]) < float(width)
    )


def _classify_measurement_status(
    first_active_sample: int | None,
    *,
    contact_censored: bool,
    event_certified: bool,
    review_reasons: list[str],
    path_exits_field: bool,
) -> str:
    """Classify usable, censored, absent-growth, and review outcomes."""

    if first_active_sample is None:
        return "unavailable"
    if not event_certified:
        return "no_growth_detected"
    if contact_censored:
        status = (
            "review_left_censored_contact"
            if first_active_sample == 0
            else "contact_censored"
        )
    elif first_active_sample == 0:
        status = "left_censored"
    else:
        status = "measured"
    if review_reasons:
        return "review_quality"
    if path_exits_field and status in {"measured", "left_censored"}:
        return "boundary_censored"
    return status


def _identity_body_hypotheses(
    identity: dict,
    source_frame: int,
    owner_track_id: int,
    owner_center_yx: np.ndarray,
    retained_track_ids: set[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return temporally nearby, semantically supported foreign pollen centers."""

    schedule = np.asarray(identity["source_frames"], dtype=np.float64)
    maximum_age = float(np.max(np.diff(schedule))) if len(schedule) > 1 else 0.0
    radius = 0.5 * float(identity["diameter_px"])
    centers = []
    track_ids = []
    for track in identity["tracks"]:
        track_id = int(track["track_id"])
        if track_id == owner_track_id or int(track["semantic_observation_count"]) < 1:
            continue
        frames = np.asarray(track["source_frames"], dtype=np.float64)
        retained_owner = (
            retained_track_ids is None or track_id in retained_track_ids
        )
        if not retained_owner and int(frames[0]) != int(schedule[0]):
            # A late round tube tip can satisfy both the learned mask and Hough
            # detector. Only independently retained late tracks may censor an
            # older owner's growing material.
            continue
        if source_frame < frames[0] - maximum_age or source_frame > frames[-1] + maximum_age:
            continue
        observations = np.asarray(track["centers_yx"], dtype=np.float64)
        center = np.asarray(
            [
                np.interp(source_frame, frames, observations[:, axis])
                for axis in range(2)
            ]
        )
        if (
            not retained_owner
            and np.linalg.norm(center - owner_center_yx) <= 1.5 * radius
        ):
            continue
        if any(np.linalg.norm(center - existing) <= 0.5 * radius for existing in centers):
            continue
        centers.append(center)
        track_ids.append(track_id)
    if not centers:
        return np.empty((0, 2), dtype=np.float64), ()
    return np.asarray(centers), tuple(track_ids)


def write_candidate_history_archive(
    output_path: Path,
    labels: list[str],
    histories: list,
    audits: list[dict],
    audit_samples: np.ndarray,
    source_frames: np.ndarray,
    owner_centers_yx: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    selected_index: int,
) -> Path:
    """Save sparse alternative curves in aligned and source coordinates."""

    candidate_count = len(histories)
    sample_count = len(audit_samples)
    maximum_points = max(
        len(frame.path_relative_yx)
        for history in histories
        for frame in history.frames
    )
    aligned_yx = np.full(
        (candidate_count, sample_count, maximum_points, 2),
        np.nan,
        dtype=np.float32,
    )
    source_yx = np.full_like(aligned_yx, np.nan)
    point_counts = np.zeros((candidate_count, sample_count), dtype=np.int32)
    lengths_px = np.zeros((candidate_count, sample_count), dtype=np.float32)
    accepted = np.zeros((candidate_count, sample_count), dtype=bool)
    shifts_yx = np.asarray(shifts_xy, dtype=np.float64)[:, ::-1]
    sparse_centers_aligned = np.asarray(owner_centers_yx[audit_samples])
    sparse_centers_source = (
        sparse_centers_aligned + shifts_yx[audit_samples]
    ) / analysis_scale
    for candidate, history in enumerate(histories):
        for sparse_sample, frame in enumerate(history.frames):
            if not frame.accepted:
                continue
            full_sample = int(audit_samples[sparse_sample])
            path = (
                np.asarray(frame.path_relative_yx, dtype=np.float64)
                + owner_centers_yx[full_sample]
            )
            count = len(path)
            aligned_yx[candidate, sparse_sample, :count] = path
            source_yx[candidate, sparse_sample, :count] = (
                path + shifts_yx[full_sample]
            ) / analysis_scale
            point_counts[candidate, sparse_sample] = count
            lengths_px[candidate, sparse_sample] = frame.length_px / analysis_scale
            accepted[candidate, sparse_sample] = True

    onset_samples = np.asarray(
        [
            -1
            if history.first_persistent_sample is None
            else int(audit_samples[history.first_persistent_sample])
            for history in histories
        ],
        dtype=np.int32,
    )
    temporary = output_path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=np.asarray(labels),
            audit_sample_indices=np.asarray(audit_samples, dtype=np.int32),
            source_frames=np.asarray(source_frames[audit_samples], dtype=np.int64),
            owner_centers_aligned_yx=sparse_centers_aligned,
            owner_centers_source_yx=sparse_centers_source,
            aligned_yx=aligned_yx,
            source_yx=source_yx,
            point_counts=point_counts,
            lengths_px=lengths_px,
            accepted=accepted,
            onset_sample_indices=onset_samples,
            causal_keys=np.asarray(
                [audit["causal_key"] for audit in audits], dtype=np.float64
            ),
            geometry_eligible=np.asarray(
                [audit["geometry_eligible"] for audit in audits], dtype=bool
            ),
            contact_censored=np.asarray(
                ["contact-censored" in label for label in labels], dtype=bool
            ),
            near_foreign_owner=np.asarray(
                [audit["near_foreign_owner"] for audit in audits], dtype=bool
            ),
            selected_index=np.asarray(selected_index, dtype=np.int32),
        )
    temporary.replace(output_path)
    return output_path


def main() -> None:
    """Run the global worldsheet experiment and write comparable artifacts."""

    args = parse_args()
    if args.mature_hypotheses < 1 or args.selection_stride < 1:
        raise ValueError("mature hypotheses and selection stride must be positive")
    selected_ids = (
        None
        if not args.track_ids
        else {int(value) for value in args.track_ids.split(",")}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.field_cache is None:
        source_frames, fps, _, native_width, native_height = movie_schedule(
            args.movie,
            args.sample_interval_s,
        )
        frames = load_grayscale_samples(
            args.movie,
            source_frames,
            args.width,
            native_width,
            native_height,
        )
        aligned, shifts_xy, registration_response = stabilize_translations(frames)
    else:
        context = load_field_context(
            args.field_cache,
            args.movie,
            args.width,
            args.sample_interval_s,
        )
        source_frames = context.source_frames
        fps = context.fps
        native_width = context.native_width
        native_height = context.native_height
        aligned = context.aligned_frames
        shifts_xy = context.shifts_xy
        registration_response = context.registration_response
    analysis_scale = args.width / native_width
    owner_radius = 15.0 * analysis_scale
    seeds, identity = load_portal_owner_seeds(
        args.identity_report,
        source_frames,
        analysis_scale,
        shifts_xy,
        include_persistent_late=True,
        minimum_late_semantic_observations=3,
    )
    center_tracks, center_scores = track_pollen_centers_from_seeds(
        aligned,
        np.asarray([seed.center_yx for seed in seeds]),
        np.asarray([seed.seed_sample for seed in seeds]),
        template_radius_px=5,
        search_radius_px=5,
        minimum_score=0.25,
    )
    seeds, center_tracks, center_scores, _ = retain_distinct_owner_tracks(
        seeds,
        center_tracks,
        center_scores,
        owner_radius_px=owner_radius,
    )
    seeds, center_tracks, center_scores, _ = retain_native_pollen_owners(
        args.movie,
        source_frames,
        analysis_scale,
        shifts_xy,
        seeds,
        center_tracks,
        center_scores,
    )
    owner_ids = [seed.track_id for seed in seeds]
    owner_motion_method = "updating-template"
    cached_owner_ids: list[int] = []
    if args.owner_motion_cache is not None:
        (
            cached_owner_ids,
            cached_tracks,
            cached_template_tracks,
            cached_scores,
        ) = load_owner_motion_cache(
            args.owner_motion_cache.expanduser().resolve(),
            source_frames,
            args.width,
            shifts_xy=shifts_xy,
            analysis_scale=analysis_scale,
        )
        owner_lookup = {track_id: index for index, track_id in enumerate(owner_ids)}
        unknown = set(cached_owner_ids) - set(owner_lookup)
        if unknown:
            raise ValueError(
                f"owner-motion cache contains unavailable pollen IDs: {sorted(unknown)}"
            )
        fallback_owner_ids = []
        for cache_owner, track_id in enumerate(cached_owner_ids):
            owner = owner_lookup[track_id]
            observed_fraction = float(np.mean(cached_scores[cache_owner] > 0.0))
            if observed_fraction >= 0.5:
                center_tracks[owner] = cached_tracks[cache_owner]
                center_scores[owner] = cached_scores[cache_owner]
            else:
                center_tracks[owner] = cached_template_tracks[cache_owner]
                fallback_owner_ids.append(track_id)
        with np.load(args.owner_motion_cache.expanduser().resolve()) as motion_cache:
            detection_constrained = (
                "detection_constrained_aligned_yx" in motion_cache.files
            )
            semantic_span = "semantic_span_aligned_yx" in motion_cache.files
        owner_motion_method = (
            "joint-detection-constrained-semantic-motion"
            if detection_constrained
            else (
                "semantic-span-cotracker-template-fusion"
                if semantic_span
                else "cotracker3-rigid-consensus-with-template-fallback"
            )
        )
    else:
        fallback_owner_ids = []
    owner_index = {track_id: index for index, track_id in enumerate(owner_ids)}
    if args.base_run is None:
        mature_paths, geometry_modes = {}, {}
    else:
        mature_paths, geometry_modes = load_mature_centerlines(
            args.base_run / "centerlines.csv",
            selected_ids,
        )
    if args.causal_atlas_run is None:
        causal_atlas_paths, causal_atlas_report = {}, None
    else:
        causal_atlas_paths, causal_atlas_report = load_causal_atlas_paths(
            args.causal_atlas_run,
            selected_ids,
            args.width,
        )
        atlas_movie = Path(causal_atlas_report["input_movie"]).expanduser().resolve()
        if atlas_movie != args.movie.expanduser().resolve():
            raise ValueError("causal atlas was built from a different movie")
    if selected_ids is not None:
        target_ids = sorted(selected_ids)
    elif args.base_run is not None:
        target_ids = sorted(mature_paths)
    else:
        target_ids = sorted(owner_ids)
    missing = set(target_ids) - set(owner_index)
    if missing:
        raise ValueError(f"requested pollen IDs are unavailable: {sorted(missing)}")
    if args.force_mature_seed is not None and len(target_ids) != 1:
        raise ValueError("a forced mature seed requires exactly one target pollen ID")
    uncached_targets = set(target_ids) - set(cached_owner_ids)
    if args.owner_motion_cache is not None and uncached_targets:
        raise ValueError(
            f"owner-motion cache lacks requested pollen IDs: {sorted(uncached_targets)}"
        )
    if args.list_track_ids:
        print(json.dumps(target_ids))
        return

    histories = {}
    diagnostics = {}
    rejected_mature_paths = {}
    contact_censored_paths = {}
    measurement_statuses = {}
    candidate_history_archives = {}
    final_drift_yx = shifts_xy[-1][::-1]
    for position, track_id in enumerate(target_ids, 1):
        owner = owner_index[track_id]
        source_path = mature_paths.get(track_id)
        geometry_mode = geometry_modes.get(track_id, "generic")
        final_center_source = (
            center_tracks[owner, -1] + final_drift_yx
        ) / analysis_scale
        foreign_source_centers, foreign_track_ids = _identity_body_hypotheses(
            identity,
            int(source_frames[-1]),
            track_id,
            final_center_source,
            set(owner_ids),
        )
        foreign_aligned_centers = (
            foreign_source_centers * analysis_scale - final_drift_yx
        )
        trace_config = TemporalRibbonConfig(
            crop_size=args.crop_size,
            worldsheet_alternative_paths=args.alternatives,
        )
        mature_candidates = []
        mature_candidate_contacts = {}
        if source_path is not None:
            censored_source_path = truncate_at_foreign_owner(
                source_path,
                foreign_source_centers,
                0.5 * float(identity["diameter_px"]),
            )
            inherited_was_censored = len(censored_source_path) < len(source_path)
            mature_relative = (
                censored_source_path - final_center_source
            ) * analysis_scale
        else:
            censored_source_path = np.empty((0, 2), dtype=np.float64)
            inherited_was_censored = False
            mature_relative = np.empty((0, 2), dtype=np.float64)
        if len(mature_relative) >= 2:
            inherited_label = "inherited-contact-censored" if inherited_was_censored else "inherited"
            mature_candidates.append((inherited_label, mature_relative))
            if inherited_was_censored:
                collision_point = source_path[len(censored_source_path)]
                collision_index = int(
                    np.argmin(
                        np.linalg.norm(
                            foreign_source_centers - collision_point,
                            axis=1,
                        )
                    )
                )
                mature_candidate_contacts[inherited_label] = {
                    "original_length_px": float(
                        np.linalg.norm(np.diff(source_path, axis=0), axis=1).sum()
                    ),
                    "retained_length_px": float(
                        np.linalg.norm(
                            np.diff(censored_source_path, axis=0), axis=1
                        ).sum()
                    ),
                    "foreign_track_id": foreign_track_ids[collision_index],
                }
        causal_path = causal_atlas_paths.get(track_id)
        if causal_path is not None:
            censored_causal_path = truncate_at_foreign_owner(
                causal_path,
                foreign_aligned_centers,
                owner_radius,
            )
            if len(censored_causal_path) >= 2:
                causal_was_censored = len(censored_causal_path) < len(causal_path)
                causal_label = (
                    "causal-atlas-contact-censored"
                    if causal_was_censored
                    else "causal-atlas"
                )
                mature_candidates.append(
                    (
                        causal_label,
                        censored_causal_path - center_tracks[owner, -1],
                    )
                )
                if causal_was_censored:
                    collision_point = causal_path[len(censored_causal_path)]
                    collision_index = int(
                        np.argmin(
                            np.linalg.norm(
                                foreign_aligned_centers - collision_point,
                                axis=1,
                            )
                        )
                    )
                    mature_candidate_contacts[causal_label] = {
                        "original_length_px": float(
                            np.linalg.norm(np.diff(causal_path, axis=0), axis=1).sum()
                            / analysis_scale
                        ),
                        "retained_length_px": float(
                            np.linalg.norm(
                                np.diff(censored_causal_path, axis=0), axis=1
                            ).sum()
                            / analysis_scale
                        ),
                        "foreign_track_id": foreign_track_ids[collision_index],
                    }
        if args.rediscover_mature or source_path is None:
            final_crop = cv.getRectSubPix(
                aligned[-1],
                (args.crop_size, args.crop_size),
                tuple(center_tracks[owner, -1][::-1]),
            )
            discovered = discover_mature_ribbon_candidates(
                final_crop,
                pollen_radius_px=owner_radius,
                feature_builder=feature_builder_for_mode(geometry_mode),
                config=trace_config,
            )
            for candidate_index, candidate in enumerate(discovered):
                if not candidate.accepted:
                    continue
                global_path = candidate.path_relative_yx + center_tracks[owner, -1]
                censored_global_path = truncate_at_foreign_owner(
                    global_path,
                    foreign_aligned_centers,
                    owner_radius,
                )
                if len(censored_global_path) < 2:
                    continue
                relative_path = censored_global_path - center_tracks[owner, -1]
                was_censored = len(censored_global_path) < len(global_path)
                label = (
                    f"rim-{candidate_index}-contact-censored"
                    if was_censored
                    else f"rim-{candidate_index}"
                )
                mature_candidates.append((label, relative_path))
                if was_censored:
                    collision_point = global_path[len(censored_global_path)]
                    collision_index = int(
                        np.argmin(
                            np.linalg.norm(
                                foreign_aligned_centers - collision_point,
                                axis=1,
                            )
                        )
                    )
                    mature_candidate_contacts[label] = {
                        "original_length_px": float(candidate.length_px / analysis_scale),
                        "retained_length_px": float(
                            np.linalg.norm(
                                np.diff(censored_global_path, axis=0), axis=1
                            ).sum()
                            / analysis_scale
                        ),
                        "foreign_track_id": foreign_track_ids[collision_index],
                    }
                if len(mature_candidates) >= args.mature_hypotheses:
                    break
        if not mature_candidates:
            rejected_mature_paths[track_id] = "no-causal-pollen-owned-mature-path"
            continue

        mature_audits = []
        mature_audit_histories = []
        mature_audit_worldsheets = []
        audit_samples = np.unique(
            np.append(
                np.arange(0, len(aligned), args.selection_stride),
                len(aligned) - 1,
            )
        )
        selected_mature = mature_candidates[0]
        selected_audit_history = None
        selected_audit_worldsheet = None
        for label, candidate_path in mature_candidates:
            audit_history, audit_worldsheet = trace_mature_ribbon_worldsheet(
                aligned[audit_samples],
                center_tracks[owner, audit_samples],
                candidate_path,
                feature_builder_for_mode(geometry_mode),
                trace_config,
                pollen_radius_px=owner_radius,
            )
            audit_history = regularize_monotone_history(audit_history)
            key = _causal_audit_key(audit_history, audit_worldsheet)
            audit_final = audit_history.frames[-1]
            (
                final_geometry_eligible,
                audit_endpoint_efficiency,
                audit_excursion_radii,
            ) = _radial_geometry_audit(
                audit_final,
                owner_radius,
                trace_config,
                contact_censored="contact-censored" in label,
            )
            baseline_eligible, baseline_excursion_radii = (
                _baseline_departure_audit(
                    audit_history,
                    owner_radius,
                    trace_config,
                    contact_censored="contact-censored" in label,
                )
            )
            geometry_eligible = final_geometry_eligible and baseline_eligible
            audit_near_foreign_owner = intersects_foreign_owner(
                audit_final.path_relative_yx + center_tracks[owner, -1],
                foreign_aligned_centers,
                owner_radius,
                clearance_radii=2.0,
                root_clearance_radii=1.0,
            )
            mature_audit_histories.append(audit_history)
            mature_audit_worldsheets.append(audit_worldsheet)
            mature_audits.append(
                {
                    "label": label,
                    "causal_key": key,
                    "onset_sample": audit_history.first_persistent_sample,
                    "final_length_px": audit_history.lengths_px[-1],
                    "endpoint_radial_efficiency": audit_endpoint_efficiency,
                    "radial_excursion_radii": audit_excursion_radii,
                    "baseline_excursion_radii": baseline_excursion_radii,
                    "baseline_departure_eligible": baseline_eligible,
                    "geometry_eligible": geometry_eligible,
                    "near_foreign_owner": audit_near_foreign_owner,
                }
            )
            print(
                f"[v27 seed audit] P{track_id:02d} {label}: "
                f"onset={audit_history.first_persistent_sample}, "
                f"growth={key[1]:.3f}, evidence={key[3]:.3f}",
                flush=True,
            )
        eligible_positions = [
            index
            for index, audit in enumerate(mature_audits)
            if audit["geometry_eligible"]
        ]
        selection_pool = eligible_positions or list(range(len(mature_audits)))
        maximum_candidate_length = max(
            float(mature_audits[index]["final_length_px"])
            for index in selection_pool
        )
        independently_selected_position = max(
            selection_pool,
            key=lambda index: _mature_selection_key(
                mature_audits[index],
                maximum_candidate_length,
            ),
        )
        selected_position = independently_selected_position
        if args.force_mature_seed is not None:
            forced_positions = [
                index
                for index, audit in enumerate(mature_audits)
                if audit["label"] == args.force_mature_seed
            ]
            if len(forced_positions) != 1:
                available = ", ".join(audit["label"] for audit in mature_audits)
                raise ValueError(
                    f"forced mature seed {args.force_mature_seed!r} is unavailable; "
                    f"audited labels: {available}"
                )
            forced_position = forced_positions[0]
            forced_audit = mature_audits[forced_position]
            forced_key = forced_audit["causal_key"]
            forced_event_certified = (
                forced_audit["onset_sample"] is not None
                and forced_key[0] > 0.5
                and np.isfinite(forced_key[1])
                and forced_key[1] >= 0.1
            )
            forced_contact_censored = "contact-censored" in forced_audit["label"]
            forced_admissible = (
                forced_audit["geometry_eligible"]
                and forced_event_certified
                and (
                    forced_contact_censored
                    or not forced_audit["near_foreign_owner"]
                )
            )
            if not forced_admissible:
                raise ValueError(
                    f"forced mature seed {args.force_mature_seed!r} is not a "
                    "certified owner-specific branch"
                )
            selected_position = forced_position
        selected_mature = mature_candidates[selected_position]
        selected_audit_history = mature_audit_histories[selected_position]
        selected_audit_worldsheet = mature_audit_worldsheets[selected_position]
        selected_sustained_growth = float(
            mature_audits[selected_position]["causal_key"][1]
        )
        if args.export_alternatives:
            candidate_history_archives[track_id] = write_candidate_history_archive(
                args.output / f"candidate_histories_P{track_id:02d}.npz",
                [label for label, _ in mature_candidates],
                mature_audit_histories,
                mature_audits,
                audit_samples,
                source_frames,
                center_tracks[owner],
                shifts_xy,
                analysis_scale,
                selected_position,
            )

        selected_contact = mature_candidate_contacts.get(selected_mature[0])
        if selected_contact is not None:
            contact_censored_paths[track_id] = selected_contact

        if not args.dense_timeline:
            history = interpolate_sampled_history(
                selected_audit_history,
                audit_samples,
                len(aligned),
            )
            worldsheet = selected_audit_worldsheet
            used_sampled_timeline = True
        else:
            history, worldsheet = trace_mature_ribbon_worldsheet(
                aligned,
                center_tracks[owner],
                selected_mature[1],
                feature_builder_for_mode(geometry_mode),
                trace_config,
                pollen_radius_px=owner_radius,
            )
            history = regularize_monotone_history(history)
            collapse_boundary = len(aligned) - trace_config.absence_persistence
            dense_collapsed = (
                history.first_persistent_sample is None
                or history.first_persistent_sample > collapse_boundary
                or history.frames[-1].length_px <= 0.0
                or worldsheet.total_score <= 0.0
            )
            used_sampled_timeline = False
            if (
                dense_collapsed
                and selected_audit_history.first_persistent_sample is not None
                and selected_audit_history.frames[-1].length_px > 0.0
                and selected_audit_worldsheet.total_score > 0.0
            ):
                history = interpolate_sampled_history(
                    selected_audit_history,
                    audit_samples,
                    len(aligned),
                )
                worldsheet = selected_audit_worldsheet
                used_sampled_timeline = True
        histories[track_id] = history
        active_sample_count = sum(frame.accepted for frame in history.frames)
        final_radial_efficiency = endpoint_radial_efficiency(history.frames[-1])
        final_radial_excursion_efficiency = radial_excursion_efficiency(
            history.frames[-1]
        )
        final_radial_excursion_radii = (
            radial_excursion_px(history.frames[-1]) / owner_radius
        )
        final_geometry_eligible, _, _ = _radial_geometry_audit(
            history.frames[-1],
            owner_radius,
            trace_config,
            contact_censored=selected_contact is not None,
        )
        baseline_eligible, baseline_excursion_radii = _baseline_departure_audit(
            history,
            owner_radius,
            trace_config,
            contact_censored=selected_contact is not None,
        )
        final_geometry_eligible = final_geometry_eligible and baseline_eligible
        final_global_path = (
            history.frames[-1].path_relative_yx + center_tracks[owner, -1]
        )
        near_foreign_owner = intersects_foreign_owner(
            final_global_path,
            foreign_aligned_centers,
            owner_radius,
            clearance_radii=2.0,
            root_clearance_radii=1.0,
        )
        event_certified = _has_event_certificate(
            active_sample_count,
            selected_sustained_growth,
            final_geometry_eligible,
            trace_config,
        )
        final_path_exits_field = _path_exits_field(
            history.frames[-1],
            center_tracks[owner, -1],
            shifts_xy[-1],
            aligned[-1].shape[:2],
        )
        review_reasons = []
        if active_sample_count < trace_config.absence_persistence:
            review_reasons.append("insufficient-active-support")
        if worldsheet.total_score <= 0.0 and not event_certified:
            review_reasons.append("nonpositive-global-evidence")
        if (
            np.isfinite(selected_sustained_growth)
            and selected_sustained_growth
            < trace_config.minimum_sustained_growth_score
        ):
            review_reasons.append("insufficient-sustained-growth")
        if history.frames[-1].length_px <= 0.0:
            review_reasons.append("nonpositive-final-length")
        if history.frames[-1].length_px > 0.0 and not final_geometry_eligible:
            review_reasons.append("pollen-rim-like-geometry")
        if (
            history.first_persistent_sample == 0
            and selected_contact is None
            and near_foreign_owner
        ):
            review_reasons.append("left-censored-near-foreign-owner")

        measurement_statuses[track_id] = _classify_measurement_status(
            history.first_persistent_sample,
            contact_censored=selected_contact is not None,
            event_certified=event_certified,
            review_reasons=review_reasons,
            path_exits_field=final_path_exits_field,
        )
        diagnostics[track_id] = {
            "first_active_sample": history.first_persistent_sample,
            "total_score": worldsheet.total_score,
            "score_margin": (
                worldsheet.score_margin
                if np.isfinite(worldsheet.score_margin)
                else None
            ),
            "alternative_selected_samples": int(
                sum(
                    index not in {None, 0}
                    for index in worldsheet.candidate_indices
                )
            ),
            "gap_samples": int(sum(state == "gap" for state in worldsheet.states)),
            "mature_seed": selected_mature[0],
            "independently_selected_mature_seed": mature_candidates[
                independently_selected_position
            ][0],
            "mature_seed_selection_mode": (
                "field-level-distinct-branch-allocation"
                if args.force_mature_seed is not None
                else "independent-causal-audit"
            ),
            "sustained_growth_score": selected_sustained_growth,
            "measurement_status": measurement_statuses.get(track_id, "measured"),
            "active_sample_count": active_sample_count,
            "final_endpoint_radial_efficiency": final_radial_efficiency,
            "final_radial_excursion_efficiency": (
                final_radial_excursion_efficiency
            ),
            "final_radial_excursion_radii": final_radial_excursion_radii,
            "baseline_excursion_radii": baseline_excursion_radii,
            "baseline_departure_eligible": baseline_eligible,
            "event_certified": event_certified,
            "final_tip_outside_field": final_path_exits_field,
            "near_foreign_owner": near_foreign_owner,
            "review_reasons": review_reasons,
            "temporal_resolution": (
                f"observed-every-{args.selection_stride}-samples-with-curve-interpolation"
                if used_sampled_timeline
                else "dense"
            ),
            "portal_rebased": history.portal_rebased,
            "portal_removed_prefix_px": (
                history.portal_removed_prefix_px / analysis_scale
            ),
            "portal_connector_length_px": (
                history.portal_connector_length_px / analysis_scale
            ),
            "mature_seed_audits": mature_audits,
            "learned_scales": {
                "prefix_error_px": worldsheet.scales.prefix_error_px,
                "root_motion_px": worldsheet.scales.root_motion_px,
                "length_change_px": worldsheet.scales.length_change_px,
                "tip_angle_radians": worldsheet.scales.tip_angle_radians,
            },
        }
        print(
            f"[v27 global] {position}/{len(target_ids)} P{track_id:02d}: "
            f"onset={history.first_persistent_sample}, "
            f"alternatives={diagnostics[track_id]['alternative_selected_samples']}, "
            f"final={history.lengths_px[-1] / analysis_scale:.1f}px",
            flush=True,
        )

    revision = (
        GLOBAL_ALLOCATION_REVISION
        if args.force_mature_seed is not None
        else MULTIPOINT_OWNER_REVISION
        if args.owner_motion_cache is not None
        else REVISION
    )
    write_outputs(
        args.output,
        histories,
        source_frames,
        fps,
        1.0 / analysis_scale,
        center_tracks,
        owner_index,
        shifts_xy,
        revision=revision,
        measurement_statuses=measurement_statuses,
    )
    review = render_review(
        args.movie,
        args.output,
        source_frames,
        fps,
        native_width,
        native_height,
        args.width,
        shifts_xy,
        center_tracks,
        owner_ids,
        histories,
        header_label=(
            "COTRACKER RIGID-OWNER WORLDSHEET"
            if args.owner_motion_cache is not None
            else "GLOBAL RIBBON WORLDSHEET"
        ),
        output_stem="global_ribbon_worldsheet",
        measurement_statuses=measurement_statuses,
    )
    report = {
        "prototype": "v27_global_ribbon_worldsheet",
        "revision": revision,
        "method": "body-aware-global-multi-hypothesis-curve-selection",
        "owner_motion_method": owner_motion_method,
        "owner_motion_cache": (
            None
            if args.owner_motion_cache is None
            else str(args.owner_motion_cache.expanduser().resolve())
        ),
        "cotracker_owner_ids": sorted(cached_owner_ids),
        "template_fallback_owner_ids": sorted(fallback_owner_ids),
        "input_movie": str(args.movie),
        "base_run": None if args.base_run is None else str(args.base_run),
        "causal_atlas_run": (
            None if args.causal_atlas_run is None else str(args.causal_atlas_run)
        ),
        "analysis_width": args.width,
        "sample_count": len(source_frames),
        "alternatives_per_local_trace": args.alternatives,
        "rediscovered_mature_paths": args.rediscover_mature,
        "forced_mature_seed": args.force_mature_seed,
        "dense_timeline": args.dense_timeline,
        "measured_owner_ids": sorted(
            track_id
            for track_id in histories
            if measurement_statuses.get(track_id, "measured")
            in {
                "measured",
                "contact_censored",
                "left_censored",
                "boundary_censored",
            }
        ),
        "review_owner_ids": sorted(
            track_id
            for track_id in histories
            if measurement_statuses.get(track_id, "measured").startswith("review_")
        ),
        "unavailable_owner_ids": sorted(
            track_id
            for track_id in histories
            if measurement_statuses.get(track_id) == "unavailable"
        ),
        "no_growth_owner_ids": sorted(
            track_id
            for track_id in histories
            if measurement_statuses.get(track_id) == "no_growth_detected"
        ),
        "rejected_mature_paths": rejected_mature_paths,
        "contact_censored_mature_paths": contact_censored_paths,
        "worldsheet_diagnostics": diagnostics,
        "median_registration_response": float(np.median(registration_response)),
        "median_owner_match_score": float(np.median(center_scores)),
        "artifacts": {
            "review_video": str(review),
            "summary_csv": str(args.output / "summary.csv"),
            "measurements_csv": str(args.output / "measurements.csv"),
            "centerlines_csv": str(args.output / "centerlines.csv"),
            **{
                f"candidate_histories_P{track_id:02d}": str(path)
                for track_id, path in candidate_history_archives.items()
            },
        },
        "scope": "Research prototype; compare against manual centerlines before quantitative use.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["artifacts"], indent=2))


if __name__ == "__main__":
    main()
