#!/usr/bin/env python3
"""Reconstruct low-density tube growth from ordered path changepoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2 as cv
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tubetracker.causal_growth_front import (
    CausalPathHypothesis,
    CausalGrowthFrontResult,
    GrowthDirectionCertificate,
    NativePathGrowthCertificate,
    NativeFrontRefinement,
    OwnerPathTopologyCertificate,
    RootedGrowthCertificate,
    ScalarPathDeformationResult,
    causal_changepoint_front,
    certify_growth_direction,
    certify_native_path_growth,
    certify_owner_path_topology,
    certify_rooted_growth,
    dynamic_path_novelty_profiles,
    fit_scalar_path_deformation,
    fit_prior_constrained_native_front,
    project_inextensible_paths,
    select_decisive_causal_path,
)
from tubetracker.causal_portal import (
    NativeCenterlineRefinement,
    NativePathQualityCertificate,
    certify_native_path_quality,
    fit_native_centerline_offsets,
    native_connected_prefix_growth,
    native_ribbon_score_surface,
    sample_native_paired_support,
    sample_native_ribbon_support,
)
from tubetracker.field_arbitration import (
    FieldOwnerEvidence,
    first_sustained_overlap_index,
    sustained_branch_capture_index,
    sustained_duplicate_claims,
)
from tubetracker.native_boundary_tracing import (
    NativeBoundaryTracingConfig,
    clip_path_to_image_bounds,
    extend_path_to_image_boundary,
    owner_aligned_temporal_consensus,
    path_extends_rooted_prefix,
    source_boundary_terminal_mask,
    trace_native_boundary_candidates,
)
from tubetracker.pollen_motion import fuse_redundant_owner_motion


AUTOMATIC_STATUSES = {"measured", "contact_censored", "boundary_censored"}
NATIVE_ONSET_ACCEPTED_REASONS = {
    "native-root-onset-corroborated",
    "native-root-onset-corrected",
}
REVISION = "v29.38-sticky-tip-evidence"
IDENTITY_TIER_RANK = {"provisional": 0, "supported": 1, "high": 2, "unknown": 2}
BODY_STATUS_RANK = {"rejected": 0, "indeterminate": 1, "verified": 2}


class NativeGrayFrameCache:
    """Decode each requested source-resolution grayscale frame at most once."""

    def __init__(self, movie: Path):
        self.movie = movie.expanduser().resolve()
        self._capture = self._open_capture()
        self._frames: dict[int, np.ndarray] = {}
        self.decode_count = 0
        self.hit_count = 0
        self.retry_count = 0

    def _open_capture(self) -> cv.VideoCapture:
        """Open the source movie or fail before an analysis silently degrades."""

        capture = cv.VideoCapture(str(self.movie))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open native source movie {self.movie}")
        return capture

    def read(self, source_frame: int) -> np.ndarray:
        """Return one immutable grayscale frame, retrying one failed seek."""

        frame_index = int(source_frame)
        if frame_index < 0:
            raise ValueError("source frame cannot be negative")
        cached = self._frames.get(frame_index)
        if cached is not None:
            self.hit_count += 1
            return cached
        frame = None
        for attempt in range(2):
            self._capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, decoded = self._capture.read()
            if ok:
                frame = decoded
                break
            if attempt == 0:
                self.retry_count += 1
                self._capture.release()
                self._capture = self._open_capture()
        if frame is None:
            raise RuntimeError(f"could not decode native source frame {frame_index}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray.setflags(write=False)
        self._frames[frame_index] = gray
        self.decode_count += 1
        return gray

    def report(self) -> dict[str, int]:
        """Summarize decoder work and in-memory cache use for reproducibility."""

        return {
            "decoded_frame_count": self.decode_count,
            "cache_hit_count": self.hit_count,
            "decoder_retry_count": self.retry_count,
            "resident_frame_count": len(self._frames),
            "resident_bytes": int(sum(frame.nbytes for frame in self._frames.values())),
        }

    def close(self) -> None:
        """Release the source decoder and cached image memory."""

        self._capture.release()
        self._frames.clear()


def _read_native_gray(
    capture: cv.VideoCapture | None,
    source_frame: int,
    native_frame_cache: NativeGrayFrameCache | None,
) -> np.ndarray:
    """Read through a shared cache or a function-local fallback decoder."""

    if native_frame_cache is not None:
        return native_frame_cache.read(source_frame)
    if capture is None:
        raise RuntimeError("native frame reader is unavailable")
    capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode native source frame {source_frame}")
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def _open_fallback_capture(
    movie: Path,
    native_frame_cache: NativeGrayFrameCache | None,
) -> cv.VideoCapture | None:
    """Open a local decoder only when a shared run cache was not supplied."""

    if native_frame_cache is not None:
        return None
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {movie} for native frame access")
    return capture


def _release_fallback_capture(capture: cv.VideoCapture | None) -> None:
    """Release an optional function-local source decoder."""

    if capture is not None:
        capture.release()


@dataclass
class MaturePathCandidate:
    """Store one mature owner-path hypothesis before temporal reconstruction."""

    source: str
    arclength_px: np.ndarray
    dynamic_paths_xy: np.ndarray
    root_distance_px: float
    radial_excursion_px: float
    foreign_contact_owner_id: int | None
    uncensored_length_px: float
    profile: np.ndarray | None = None
    result: CausalGrowthFrontResult | None = None
    reverse_result: CausalGrowthFrontResult | None = None
    growth_certificate: RootedGrowthCertificate | None = None
    direction_certificate: GrowthDirectionCertificate | None = None
    native_quality_certificate: NativePathQualityCertificate | None = None
    proposal_score: float = 0.0
    deformation: ScalarPathDeformationResult | None = None
    rigid_paths_xy: np.ndarray | None = None
    temporal_reference_source: str | None = None
    native_root_onset_sample: int | None = None
    native_root_onset_delay_samples: int | None = None
    native_root_onset_reason: str = "not-audited"
    topology_certificate: OwnerPathTopologyCertificate | None = None
    proposal_frame_root_xy: np.ndarray | None = None
    proposal_frame_owner_xy: np.ndarray | None = None
    proposal_frame_gap_px: float | None = None
    boundary_contact: bool = False
    native_boundary_final_length_px: float = 0.0
    native_boundary_lengths_px: np.ndarray | None = None
    native_boundary_onset_sample: int | None = None
    native_boundary_timeline_reason: str = "not-audited"


@dataclass
class OwnerGeometry:
    """Store one immutable branch and its frame-specific material coordinates."""

    track_id: int
    field_status: str
    original_measurements: pd.DataFrame
    mature_candidates: list[MaturePathCandidate]
    arclength_px: np.ndarray
    dynamic_paths_xy: np.ndarray
    identity_tier: str = "unknown"
    identity_observation_count: int = 0
    semantic_observation_count: int = 0
    geometric_assignment_fraction: float = float("nan")
    median_owner_template_score: float = float("nan")
    body_status: str = "legacy-unverified"
    identity_certificate_basis: str = "legacy-unverified"
    pregrowth_semantic_consensus: bool = False
    body_median_radial_contrast: float = float("nan")
    body_median_angular_boundary_fraction: float = float("nan")
    body_median_opposite_boundary_fraction: float = float("nan")
    body_median_valid_angular_fraction: float = float("nan")
    body_valid_sample_count: int = 0
    native_boundary_candidates: list[MaturePathCandidate] = field(default_factory=list)
    rigid_paths_xy: np.ndarray | None = None
    rigid_profile: np.ndarray | None = None
    profile: np.ndarray | None = None
    rigid_result: CausalGrowthFrontResult | None = None
    reverse_result: CausalGrowthFrontResult | None = None
    deformation: ScalarPathDeformationResult | None = None
    deformed_result: CausalGrowthFrontResult | None = None
    selected_pose_model: str = "deformable"
    selected_mature_path: str = "field"
    mature_path_selection_reason: str = "baseline-causally-supported"
    selected_root_distance_px: float = 0.0
    selected_radial_excursion_px: float = 0.0
    selected_foreign_contact_owner_id: int | None = None
    shared_branch_owner_id: int | None = None
    foreign_branch_owner_id: int | None = None
    foreign_branch_capture_point: int | None = None
    ownership_conflict_ids: list[int] = field(default_factory=list)
    growth_certificate: RootedGrowthCertificate | None = None
    topology_certificate: OwnerPathTopologyCertificate | None = None
    direction_certificate: GrowthDirectionCertificate | None = None
    native_prefix_lengths_px: np.ndarray | None = None
    native_onset_sample: int | None = None
    native_last_dormant_sample: int | None = None
    native_certificate: NativePathGrowthCertificate | None = None
    native_centerline_refinement: NativeCenterlineRefinement | None = None
    native_refinement_observation_samples: list[int] = field(default_factory=list)
    native_path_quality_certificate: NativePathQualityCertificate | None = None
    native_recovery_refinement: NativeCenterlineRefinement | None = None
    native_recovery_paths_xy: np.ndarray | None = None
    native_recovery_arclength_px: np.ndarray | None = None
    native_recovery_holdout_samples: list[int] = field(default_factory=list)
    native_recovery_reason: str = "not-audited"
    native_recovery_original_coverage: float = 0.0
    native_recovery_coverage: float = 0.0
    native_root_onset_sample: int | None = None
    native_root_last_dormant_sample: int | None = None
    native_root_onset_reason: str = "not-audited"
    native_root_onset_delay_samples: int | None = None
    native_root_final_growth_px: float = 0.0
    timeline_source: str = "coarse-causal"
    native_front_refinement: NativeFrontRefinement | None = None
    native_front_paths_xy: np.ndarray | None = None
    result: CausalGrowthFrontResult | None = None
    forecast_fault_samples: int = 0
    forecast_replay_samples: int = 0
    forecast_stability_veto: bool = False
    forecast_tip_hold_samples: int = 0
    forecast_tip_hold_max_run: int = 0
    forecast_tip_series_samples: int = 0
    sticky_tip_corrected_samples: int = 0
    sticky_tip_max_move_px: float = 0.0
    sticky_tip_mean_move_px: float = 0.0
    sticky_tip_series_samples: int = 0


def parse_args() -> argparse.Namespace:
    """Parse source runs, owner selection, image scale, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-run", type=Path, required=True)
    parser.add_argument("--field-cache", type=Path, required=True)
    parser.add_argument("--owner-motion-cache", type=Path, required=True)
    parser.add_argument(
        "--owner-motion-mode",
        choices=("identity-first", "detection", "multipoint", "template"),
        default="identity-first",
    )
    parser.add_argument("--causal-atlas-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", help="Comma-separated owner IDs; default is all")
    parser.add_argument("--arc-step-px", type=float, default=1.0)
    parser.add_argument("--warmup-samples", type=int, default=4)
    parser.add_argument("--normal-halfwidth-px", type=float, default=4.0)
    parser.add_argument("--normal-samples", type=int, default=9)
    parser.add_argument("--max-growth-px-per-sample", type=float, default=2.0)
    parser.add_argument("--germination-length-source-px", type=float, default=10.0)
    parser.add_argument("--pollen-radius-source-px", type=float, default=15.0)
    parser.add_argument("--foreign-owner-exclusion-radii", type=float, default=4.0 / 3.0)
    parser.add_argument("--minimum-inward-direction-score-margin", type=float, default=0.50)
    parser.add_argument("--minimum-inward-direction-coverage", type=float, default=0.65)
    parser.add_argument("--minimum-inward-direction-direct-support", type=float, default=0.85)
    parser.add_argument("--minimum-inward-direction-eventual-support", type=float, default=0.95)
    parser.add_argument("--native-minimum-completion-fraction", type=float, default=0.75)
    parser.add_argument("--native-maximum-onset-delay-samples", type=int, default=12)
    parser.add_argument("--native-normal-search-px", type=float, default=3.0)
    parser.add_argument("--native-coverage-tie-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--native-minimum-proximal-support-ratio",
        type=float,
        default=0.25,
        help=(
            "Minimum proximal-third over distal-third late-median native "
            "support before a candidate counts as emerging from its own "
            "grain (dense P76 latched a neighbor tube at ratio 0.09 while "
            "verified owners read 0.59+)."
        ),
    )
    parser.add_argument(
        "--native-recenter-search-radius-px",
        type=int,
        default=16,
        help=(
            "Half-width of the lateral corridor when recentering rescued "
            "native-extension paths (dense P97's off-axis claim sits 8-15 "
            "px from the true walls, outside the pre-rescue radius-8 stage)."
        ),
    )
    parser.add_argument(
        "--native-recenter-minimum-offset-px",
        type=float,
        default=3.0,
        help=(
            "Minimum median fitted offset before a rescue recentering moves "
            "a path; smaller corrections leave centered claims untouched."
        ),
    )
    parser.add_argument(
        "--native-deformation-minimum-coverage-gain",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--native-consensus-minimum-length-fraction",
        type=float,
        default=0.90,
        help=(
            "Minimum consensus-path length relative to the longest verified "
            "single-frame path for the same owner"
        ),
    )
    parser.add_argument(
        "--native-boundary-minimum-length-gain-fraction",
        type=float,
        default=0.15,
        help=(
            "Minimum gain over the best non-edge path before a fully verified "
            "edge-censored path may displace it"
        ),
    )
    parser.add_argument("--branch-capture-distance-radii", type=float, default=1.0 / 3.0)
    parser.add_argument("--branch-capture-minimum-shared-radii", type=float, default=0.75)
    parser.add_argument("--branch-capture-minimum-onset-lead-samples", type=int, default=12)
    parser.add_argument("--demo-fps", type=float, default=4.0)
    parser.add_argument(
        "--forecast-veto-dir",
        type=Path,
        default=None,
        help=(
            "Directory of TimesFM guided-replay CSVs (replay_P*.csv). When "
            "given, owners whose closed-loop veto trips fail closed to "
            "forecast-stability-review instead of automatic measurement "
            "(dense P131 flags 9/125 while all other owners flag zero). "
            "Omitted disables the gate."
        ),
    )
    parser.add_argument("--forecast-veto-min-faults", type=int, default=3)
    parser.add_argument("--forecast-veto-min-rate", type=float, default=0.05)
    parser.add_argument(
        "--forecast-tip-dir",
        type=Path,
        default=None,
        help=(
            "Directory of TimesFM guided tip-filter CSVs (guided_P*.csv). "
            "When given, per-owner HOLD counts are recorded as evidence "
            "columns only (no geometry change, no status change). Omitted "
            "disables the evidence."
        ),
    )
    parser.add_argument(
        "--sticky-tip-dir",
        type=Path,
        default=None,
        help=(
            "Directory of sticky coherent tip-correction CSVs "
            "(sticky_P<id>.csv with sample_index, corrected_x/y, "
            "moved_px, keep). When given, per-owner correction counts "
            "are recorded as evidence columns only (no geometry change, "
            "no status change). Omitted disables the evidence."
        ),
    )
    return parser.parse_args()


def _resample_path(path_xy: np.ndarray, step_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at uniform arclength while retaining its endpoint."""

    path = np.asarray(path_xy, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0.0:
        raise ValueError("cannot resample a zero-length path")
    samples = np.arange(0.0, cumulative[-1], step_px)
    if not len(samples) or cumulative[-1] - samples[-1] > 1e-6:
        samples = np.append(samples, cumulative[-1])
    resampled = np.column_stack(
        [np.interp(samples, cumulative, path[:, axis]) for axis in range(2)]
    )
    return resampled, samples


def _load_motion_ambiguity_ids(motion_path: Path) -> set[int]:
    """Return bootstrap-flagged motion-ambiguous owners, if the cache has any."""

    try:
        with np.load(motion_path) as cache:
            if "motion_ambiguous_owner_ids" not in cache.files:
                return set()
            return {int(value) for value in cache["motion_ambiguous_owner_ids"]}
    except (OSError, ValueError):
        return set()


def _load_forecast_veto_counts(veto_dir: Path) -> dict[int, tuple[int, int]]:
    """Return per-owner (fault-suspect, replayed) counts from guided replay.

    Reads ``replay_P<id>.csv`` files written by the TimesFM guided-replay
    prototype.  A missing directory, unreadable file, or malformed row fails
    open to exclusion (the gate stays off for that owner) — this loader never
    blocks measurement by itself; only an explicit trip downstream does.
    """

    counts: dict[int, tuple[int, int]] = {}
    try:
        files = sorted(Path(veto_dir).glob("replay_P*.csv"))
    except OSError:
        return counts
    for path in files:
        try:
            owner = int(path.stem.split("_P")[1])
            frame = pd.read_csv(path)
        except (ValueError, IndexError, OSError, pd.errors.ParserError):
            continue
        if "verdict" not in frame.columns or len(frame) == 0:
            continue
        faults = int((frame["verdict"] == "fault-suspect").sum())
        counts[owner] = (faults, int(len(frame)))
    return counts


def _forecast_stability_veto_trips(
    n_fault: int,
    n: int,
    *,
    min_faults: int,
    min_rate: float,
) -> bool:
    """Return whether closed-loop veto flags fail-close an owner.

    Both an absolute count and a rate are required so a single noisy replay
    row cannot demote an owner, while a sustained flicker zone (dense P131:
    9 fault-suspect of 125 replayed) always trips.
    """

    if n <= 0 or n_fault < min_faults:
        return False
    return (n_fault / n) >= min_rate


def _load_forecast_tip_holds(tip_dir: Path) -> dict[int, tuple[int, int, int]]:
    """Return per-owner (held, max_held_run, total) from guided tip series.

    Reads ``guided_P<id>.csv`` files written by the TimesFM guided tip
    filter (``held`` boolean column).  Evidence only: a missing directory,
    unreadable file, or malformed row fails open to (0, 0, 0) for that
    owner — this loader never changes geometry or status by itself.
    """

    holds: dict[int, tuple[int, int, int]] = {}
    try:
        files = sorted(Path(tip_dir).glob("guided_P*.csv"))
    except OSError:
        return holds
    for path in files:
        try:
            owner = int(path.stem.split("_P")[1])
            frame = pd.read_csv(path)
        except (ValueError, IndexError, OSError, pd.errors.ParserError):
            continue
        if "held" not in frame.columns or len(frame) == 0:
            continue
        is_held = frame["held"].fillna(False).astype(bool).to_numpy()
        n_held = int(is_held.sum())
        run = best = 0
        for flag in is_held:
            run = run + 1 if flag else 0
            best = max(best, run)
        holds[owner] = (n_held, int(best), int(len(frame)))
    return holds


def _load_sticky_tip_evidence(tip_dir: Path) -> dict[int, tuple[int, float, float, int]]:
    """Return per-owner (kept, max_move, mean_move, total) from sticky series.

    Reads ``sticky_P<id>.csv`` files written by ``build_sticky_series``
    (``keep`` boolean column, ``moved_px``).  Evidence only: a missing
    directory, unreadable file, or malformed row fails open to
    (0, 0.0, 0.0, 0) for that owner — this loader never changes
    geometry or status by itself.
    """

    evidence: dict[int, tuple[int, float, float, int]] = {}
    try:
        files = sorted(Path(tip_dir).glob("sticky_P*.csv"))
    except OSError:
        return evidence
    for path in files:
        try:
            owner = int(path.stem.split("_P")[1])
            frame = pd.read_csv(path)
        except (ValueError, IndexError, OSError, pd.errors.ParserError):
            continue
        if "keep" not in frame.columns or "moved_px" not in frame.columns:
            continue
        if len(frame) == 0:
            continue
        moves = frame["moved_px"].fillna(0.0).to_numpy(dtype=float)
        kept = frame["keep"].fillna(False).astype(bool).to_numpy()
        kept_moves = moves[kept]
        evidence[owner] = (
            int(kept.sum()),
            round(float(kept_moves.max()) if len(kept_moves) else 0.0, 1),
            round(float(kept_moves.mean()) if len(kept_moves) else 0.0, 1),
            int(len(frame)),
        )
    return evidence


def _load_owner_centers(
    motion_path: Path,
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    mode: str,
) -> tuple[dict[int, np.ndarray], dict]:
    """Load owner motion with redundant temporal identity as the default."""

    with np.load(motion_path) as cache:
        if not np.array_equal(cache["source_frames"], source_frames):
            raise ValueError("owner motion and field cache use different samples")
        track_ids = np.asarray(cache["track_ids"], dtype=np.int64)
        if mode == "identity-first":
            required = {
                "multipoint_source_yx",
                "template_source_yx",
                "detection_constrained_source_yx",
                "observed",
                "inlier_counts",
                "template_scores",
            }
            missing = required - set(cache.files)
            if missing:
                raise ValueError(
                    "identity-first owner motion requires cache fields: "
                    + ", ".join(sorted(missing))
                )
            fusion = fuse_redundant_owner_motion(
                cache["multipoint_source_yx"],
                cache["template_source_yx"],
                cache["detection_constrained_source_yx"],
                cache["observed"],
                cache["inlier_counts"],
                cache["template_scores"],
            )
            source_yx = fusion.centers_yx
            source_codes = fusion.source_codes
            detection = np.asarray(
                cache["detection_constrained_source_yx"],
                dtype=np.float64,
            )
            disagreement = np.linalg.norm(source_yx - detection, axis=2)
            fusion_report = {
                "mode": mode,
                "source_code_legend": {
                    "0": "multipoint-rigid-consensus",
                    "1": "template-fallback",
                    "2": "circle-detection-fallback",
                },
                "source_sample_counts": {
                    str(code): int(np.sum(source_codes == code))
                    for code in range(3)
                },
                "required_inlier_counts_by_owner": {
                    str(int(track_id)): int(required_count)
                    for track_id, required_count in zip(
                        track_ids,
                        fusion.required_inlier_counts,
                    )
                },
                "detection_disagreement_p90_source_px_by_owner": {
                    str(int(track_id)): float(np.percentile(values, 90.0))
                    for track_id, values in zip(track_ids, disagreement)
                },
            }
        else:
            source_key = {
                "detection": "detection_constrained_source_yx",
                "multipoint": "multipoint_source_yx",
                "template": "template_source_yx",
            }[mode]
            if source_key not in cache.files:
                raise ValueError(f"owner motion cache lacks {source_key}")
            source_yx = np.asarray(cache[source_key], dtype=np.float64)
            fusion_report = {"mode": mode, "source_key": source_key}
    aligned_yx = source_yx * analysis_scale - shifts_xy[None, :, ::-1]
    return (
        {
            int(track_id): aligned_yx[index, :, ::-1]
            for index, track_id in enumerate(track_ids)
        },
        fusion_report,
    )


def _owner_translated_material_paths(
    centerlines: pd.DataFrame,
    centers_xy: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    arc_step_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Register one mature branch and move it only with its pollen owner.

    Earlier threshold-derived centerlines are deliberately not used here.  Their
    delayed expansion is the observation being corrected; using it to deform the
    corridor would manufacture an image changepoint at the previous jump time.
    """

    reference_sample = int(centerlines["sample_index"].max())
    reference_rows = centerlines[
        centerlines["sample_index"] == reference_sample
    ].sort_values("point_index")
    reference_yx = (
        reference_rows[["source_y_px", "source_x_px"]].to_numpy()
        * analysis_scale
        - shifts_xy[reference_sample, ::-1]
    )
    canonical_xy, arclength = _resample_path(reference_yx[:, ::-1], arc_step_px)
    paths = canonical_xy[None, :, :] + (
        centers_xy - centers_xy[reference_sample]
    )[:, None, :]
    return paths, arclength


def _atlas_material_paths(
    owner_rows: pd.DataFrame,
    centers_xy: np.ndarray,
    arc_step_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move an aligned causal-atlas path with its owner from the final sample."""

    canonical_xy, arclength = _resample_path(
        owner_rows.sort_values("path_index")[["x_analysis_px", "y_analysis_px"]]
        .to_numpy(dtype=np.float64),
        arc_step_px,
    )
    paths = canonical_xy[None, :, :] + (
        centers_xy - centers_xy[-1]
    )[:, None, :]
    return paths, arclength


def _candidate_geometry(
    source: str,
    paths_xy: np.ndarray,
    arclength_px: np.ndarray,
    owner_center_xy: np.ndarray,
    foreign_contact_owner_id: int | None = None,
    uncensored_length_px: float | None = None,
    boundary_contact: bool = False,
) -> MaturePathCandidate:
    """Measure owner attachment and outward excursion for one mature path."""

    mature = paths_xy[-1]
    distances = np.linalg.norm(mature - owner_center_xy[-1], axis=1)
    return MaturePathCandidate(
        source=source,
        arclength_px=arclength_px,
        dynamic_paths_xy=paths_xy,
        root_distance_px=float(distances[0]),
        radial_excursion_px=float(np.max(distances) - distances[0]),
        foreign_contact_owner_id=foreign_contact_owner_id,
        uncensored_length_px=(
            float(arclength_px[-1])
            if uncensored_length_px is None
            else float(uncensored_length_px)
        ),
        boundary_contact=boundary_contact,
    )


def _certify_candidate_topology(
    candidate: MaturePathCandidate,
    owner_center_xy: np.ndarray,
    owner_radius_px: float,
    *,
    native_root_yx: np.ndarray | None = None,
) -> OwnerPathTopologyCertificate:
    """Certify only the final causal prefix of one mature path candidate.

    When the proposal was traced in a different frame than the owner's
    (e.g. an owner-aligned consensus crop), ``native_root_yx`` carries the
    proposal-frame root so attachment is measured against the grain the
    tracer actually started from — not a consensus-averaged owner center
    that may sit on background between grains.
    """

    if candidate.result is None:
        raise ValueError("candidate result must be fitted before topology review")
    point_count = max(1, candidate.result.final_point_index + 1)
    point_count = min(point_count, candidate.dynamic_paths_xy.shape[1])
    certificate = certify_owner_path_topology(
        candidate.dynamic_paths_xy[-1, :point_count],
        owner_center_xy[-1],
        owner_radius_px=owner_radius_px,
    )
    candidate.topology_certificate = certificate
    candidate.root_distance_px = certificate.root_distance_px
    candidate.radial_excursion_px = certificate.radial_excursion_px
    if native_root_yx is not None:
        anchor = np.asarray(native_root_yx, dtype=np.float64)
        if anchor.shape == (2,) and np.isfinite(anchor).all():
            candidate.proposal_frame_owner_xy = anchor.copy()
    return certificate


def _candidate_hypothesis(
    candidate: MaturePathCandidate,
) -> CausalPathHypothesis:
    """Convert a fitted candidate and its retained topology into a hypothesis."""

    if candidate.result is None or candidate.topology_certificate is None:
        raise ValueError("candidate result and topology must be available")
    return CausalPathHypothesis(
        label=candidate.source,
        result=candidate.result,
        total_length_px=float(candidate.arclength_px[-1]),
        root_distance_px=candidate.root_distance_px,
        radial_excursion_px=candidate.radial_excursion_px,
        terminates_at_foreign_owner=(
            candidate.foreign_contact_owner_id is not None
        ),
        topology_accepted=candidate.topology_certificate.accepted,
        topology_reason=candidate.topology_certificate.reason,
    )


def _refresh_owner_topology(
    owner: OwnerGeometry,
    owner_center_xy: np.ndarray,
    owner_radius_px: float,
) -> OwnerPathTopologyCertificate:
    """Refresh retained owner geometry after pose, refinement, or arbitration."""

    if owner.result is None:
        raise ValueError("owner result must be fitted before topology review")
    point_count = max(1, owner.result.final_point_index + 1)
    point_count = min(point_count, owner.dynamic_paths_xy.shape[1])
    certificate = certify_owner_path_topology(
        owner.dynamic_paths_xy[-1, :point_count],
        owner_center_xy[-1],
        owner_radius_px=owner_radius_px,
    )
    owner.topology_certificate = certificate
    owner.selected_root_distance_px = certificate.root_distance_px
    owner.selected_radial_excursion_px = certificate.radial_excursion_px
    owner.growth_certificate = certify_rooted_growth(
        CausalPathHypothesis(
            label=owner.selected_mature_path,
            result=owner.result,
            total_length_px=float(owner.arclength_px[-1]),
            root_distance_px=certificate.root_distance_px,
            radial_excursion_px=certificate.radial_excursion_px,
            terminates_at_foreign_owner=(
                owner.selected_foreign_contact_owner_id is not None
            ),
            topology_accepted=certificate.accepted,
            topology_reason=certificate.reason,
        ),
        owner_radius_px=owner_radius_px,
        boundary_censored=owner.field_status == "boundary_censored",
    )
    return certificate


def _audit_retained_owner_topologies(
    owners: list[OwnerGeometry],
    centers_by_id: dict[int, np.ndarray],
    owner_radius_px: float,
) -> dict:
    """Recompute owner-relative topology after all retained-path refinements."""

    results = []
    for owner in owners:
        if owner.result is None:
            continue
        certificate = _refresh_owner_topology(
            owner,
            centers_by_id[owner.track_id],
            owner_radius_px,
        )
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": certificate.accepted,
                "reason": certificate.reason,
                "root_distance_px": certificate.root_distance_px,
                "minimum_distance_px": certificate.minimum_distance_px,
                "maximum_distance_px": certificate.maximum_distance_px,
                "radial_excursion_px": certificate.radial_excursion_px,
                "first_halo_exit_index": certificate.first_halo_exit_index,
                "first_halo_exit_arclength_px": (
                    certificate.first_halo_exit_arclength_px
                ),
            }
        )
    return {
        "candidate_owner_ids": [int(row["owner_id"]) for row in results],
        "verified_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "withheld_owner_ids": [
            int(row["owner_id"]) for row in results if not row["accepted"]
        ],
        "results": results,
    }


def _foreign_owner_safe_prefix(
    paths_xy: np.ndarray,
    arclength_px: np.ndarray,
    *,
    track_id: int,
    centers_by_id: dict[int, np.ndarray],
    owner_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, int | None, float]:
    """Stop a mature hypothesis before its first foreign pollen body."""

    uncensored_length = float(arclength_px[-1])
    mature = paths_xy[-1]
    first_contact = len(mature)
    contact_owner: int | None = None
    for other_id, centers_xy in centers_by_id.items():
        if other_id == track_id:
            continue
        contacts = np.flatnonzero(
            np.linalg.norm(mature - centers_xy[-1], axis=1) <= owner_radius_px
        )
        if len(contacts) and int(contacts[0]) < first_contact:
            first_contact = int(contacts[0])
            contact_owner = other_id
    if contact_owner is None:
        return paths_xy, arclength_px, None, uncensored_length
    stop = max(2, first_contact)
    return (
        paths_xy[:, :stop],
        arclength_px[:stop],
        contact_owner,
        uncensored_length,
    )


def _blocking_centers_for_owner(
    centers_by_id: dict[int, np.ndarray],
    body_status_by_id: dict[int, str],
    identity_tiers_by_id: dict[int, str],
    track_id: int,
) -> dict[int, np.ndarray]:
    """Use verified foreign pollen bodies, with explicit legacy compatibility."""

    owner_rank = IDENTITY_TIER_RANK.get(
        identity_tiers_by_id.get(track_id, "unknown"),
        IDENTITY_TIER_RANK["unknown"],
    )
    return {
        other_id: centers
        for other_id, centers in centers_by_id.items()
        if (
            other_id == track_id
            or body_status_by_id.get(other_id, "legacy-unverified") == "verified"
            or (
                body_status_by_id.get(other_id, "legacy-unverified")
                == "legacy-unverified"
                and IDENTITY_TIER_RANK.get(
                    identity_tiers_by_id.get(other_id, "unknown"),
                    IDENTITY_TIER_RANK["unknown"],
                )
                >= owner_rank
            )
        )
    }


def _feature_stack(frames: np.ndarray, mode: str, pollen_radius_px: float) -> np.ndarray:
    """Build one polarity-sensitive tube evidence channel for all samples."""

    output = np.empty(frames.shape, dtype=np.float32)
    diameter = max(5, int(round(2.0 * pollen_radius_px)) | 1)
    blackhat_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    for sample, gray in enumerate(frames):
        if mode == "dark":
            output[sample] = 255.0 - gray
        elif mode == "dog":
            values = gray.astype(np.float32)
            output[sample] = cv.GaussianBlur(
                values, (0, 0), max(2.5, pollen_radius_px * 0.65)
            ) - cv.GaussianBlur(values, (0, 0), max(0.55, pollen_radius_px * 0.12))
        elif mode == "blackhat":
            output[sample] = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, blackhat_kernel)
        else:
            raise ValueError(f"unknown feature mode: {mode}")
    return output


def _write_kymograph(
    output: Path,
    owner: OwnerGeometry,
    analysis_scale: float,
) -> None:
    """Render the causal front against the prior length history."""

    assert owner.profile is not None and owner.result is not None
    result = owner.result
    causal = np.where(
        result.front_indices >= 0,
        owner.arclength_px[
            np.clip(result.front_indices, 0, len(owner.arclength_px) - 1)
        ],
        0.0,
    ) / analysis_scale
    original = owner.original_measurements["tube_length_px"].to_numpy()
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.imshow(
        owner.profile.T,
        origin="lower",
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        extent=[0, len(causal) - 1, 0, owner.arclength_px[-1] / analysis_scale],
    )
    axis.plot(original, "k--", linewidth=1.2, label="previous length")
    axis.plot(causal, color="#30f070", linewidth=2.0, label="causal front")
    axis.set_xlabel("15-second sample")
    axis.set_ylabel("arclength from pollen (source pixels)")
    axis.set_title(
        f"P{owner.track_id:02d}: {result.reason}; "
        f"{original[-1]:.1f} -> {causal[-1]:.1f} px"
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output / f"P{owner.track_id:02d}_causal_kymograph.png", dpi=120)
    plt.close(figure)


def _source_path(
    aligned_xy: np.ndarray,
    shift_xy: np.ndarray,
    analysis_scale: float,
) -> np.ndarray:
    """Convert aligned x/y coordinates to source x/y coordinates."""

    return (aligned_xy + shift_xy) / analysis_scale


def _first_threshold_sample(lengths_px: np.ndarray, threshold_px: float) -> int | None:
    """Return the first sample whose material length reaches a threshold."""

    matches = np.flatnonzero(lengths_px >= threshold_px)
    return int(matches[0]) if len(matches) else None


def _measurement_eligible(owner: OwnerGeometry) -> bool:
    """Return whether field or independent native evidence permits measurement."""

    prior_evidence_permits = bool(
        owner.field_status in AUTOMATIC_STATUSES
        or (owner.native_certificate is not None and owner.native_certificate.accepted)
    )
    native_quality_permits = bool(
        owner.native_path_quality_certificate is None
        or owner.native_path_quality_certificate.accepted
    )
    topology_permits = bool(
        owner.topology_certificate is not None
        and owner.topology_certificate.accepted
    )
    body_identity_permits = owner.body_status not in {"rejected", "indeterminate"}
    return (
        prior_evidence_permits
        and native_quality_permits
        and topology_permits
        and body_identity_permits
    )


def _promote_native_prefix_timeline(
    owner: OwnerGeometry,
    native_lengths_source_px: np.ndarray,
    analysis_scale: float,
) -> None:
    """Replace a premature coarse front with independently observed native growth."""

    lengths = np.asarray(native_lengths_source_px, dtype=np.float64) * analysis_scale
    _replace_owner_front_from_lengths(owner, lengths, "native-paired-prefix")


def _replace_owner_front_from_lengths(
    owner: OwnerGeometry,
    lengths_analysis_px: np.ndarray,
    timeline_source: str,
) -> None:
    """Replace one owner's front with a monotone physical-length timeline."""

    if owner.result is None:
        raise ValueError("owner result must exist before timeline replacement")
    lengths = np.asarray(lengths_analysis_px, dtype=np.float64)
    if lengths.shape != owner.result.front_indices.shape:
        raise ValueError("replacement length timeline must match the sampled movie")
    if not np.isfinite(lengths).all() or np.any(np.diff(lengths) < -1e-6):
        raise ValueError("replacement lengths must be finite and nondecreasing")
    lengths = np.clip(lengths, 0.0, owner.arclength_px[-1])
    fronts = np.searchsorted(owner.arclength_px, lengths, side="right") - 1
    fronts[lengths <= 0.0] = -1
    fronts = fronts.astype(np.int32)
    birth_samples = np.full(len(owner.arclength_px), len(fronts), dtype=np.int32)
    for point in range(len(owner.arclength_px)):
        observations = np.flatnonzero(fronts >= point)
        if len(observations):
            birth_samples[point] = int(observations[0])
    supported = birth_samples < len(fronts)
    final_point = int(fronts[-1])
    owner.timeline_source = timeline_source
    owner.result = replace(
        owner.result,
        birth_samples=birth_samples,
        front_indices=fronts,
        direct_support_mask=supported.copy(),
        eventual_support_mask=supported.copy(),
        feasible=final_point >= 0,
        final_point_index=final_point,
        final_length_px=(
            0.0 if final_point < 0 else float(owner.arclength_px[final_point])
        ),
        direct_support_fraction=(
            0.0 if final_point < 0 else float(np.mean(supported[: final_point + 1]))
        ),
        eventual_support_fraction=(
            0.0 if final_point < 0 else float(np.mean(supported[: final_point + 1]))
        ),
    )


def _promote_native_root_timeline(
    owner: OwnerGeometry,
    native_root_lengths_source_px: np.ndarray,
    native_onset_sample: int,
    analysis_scale: float,
    maximum_growth_analysis_px_per_sample: float,
) -> None:
    """Anchor coarse full-path growth to the observed native root emergence."""

    if owner.result is None or maximum_growth_analysis_px_per_sample <= 0.0:
        raise ValueError("owner result and a positive growth limit are required")
    coarse = np.where(
        owner.result.front_indices >= 0,
        owner.arclength_px[
            np.clip(owner.result.front_indices, 0, len(owner.arclength_px) - 1)
        ],
        0.0,
    )
    native_root = (
        np.asarray(native_root_lengths_source_px, dtype=np.float64)
        * analysis_scale
    )
    proposed = np.maximum(coarse, native_root)
    proposed[:native_onset_sample] = 0.0
    elapsed = np.arange(len(proposed), dtype=np.float64) - native_onset_sample + 1.0
    growth_cap = np.maximum(elapsed, 0.0) * maximum_growth_analysis_px_per_sample
    proposed = np.minimum(proposed, growth_cap)
    proposed = np.maximum.accumulate(proposed)
    _replace_owner_front_from_lengths(
        owner,
        proposed,
        "native-root-timed-prefix",
    )


def _verify_review_paths_at_native_resolution(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    germination_length_px: float,
    minimum_completion_fraction: float,
    maximum_onset_delay_samples: int,
    normal_search_px: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Rescue withheld paths only when native tube walls reproduce their growth."""

    candidates = [
        owner
        for owner in owners
        if owner.field_status not in AUTOMATIC_STATUSES
        and owner.result is not None
        and owner.result.final_length_px / analysis_scale >= germination_length_px
        and owner.growth_certificate is not None
        and owner.growth_certificate.accepted
        and owner.direction_certificate is not None
        and owner.direction_certificate.accepted
        and not owner.ownership_conflict_ids
    ]
    if not candidates:
        return {"candidate_owner_ids": [], "verified_owner_ids": [], "results": []}

    support = {
        owner.track_id: np.zeros(
            (len(source_frames), owner.result.final_point_index + 1),
            dtype=np.float32,
        )
        for owner in candidates
    }
    thresholds = {
        owner.track_id: np.zeros(len(source_frames), dtype=np.float32)
        for owner in candidates
    }
    capture = _open_fallback_capture(movie, native_frame_cache)
    search_offsets = (-normal_search_px, 0.0, normal_search_px)
    for sample, source_frame in enumerate(source_frames):
        gray = _read_native_gray(capture, source_frame, native_frame_cache)
        for owner in candidates:
            point_count = owner.result.final_point_index + 1
            source_path_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            path_support, controls = sample_native_paired_support(
                gray,
                source_path_yx,
                normal_search_offsets_px=search_offsets,
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support[owner.track_id][sample] = path_support
            thresholds[owner.track_id][sample] = max(
                2.0,
                control_median + 2.0 * control_noise,
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        point_count = owner.result.final_point_index + 1
        source_arc = owner.arclength_px[:point_count] / analysis_scale
        lengths, _, onset, last_dormant = native_connected_prefix_growth(
            support[owner.track_id],
            thresholds[owner.track_id],
            source_arc,
            warmup_samples=8,
            maximum_gap_points=2,
        )
        causal_lengths = np.where(
            owner.result.front_indices >= 0,
            owner.arclength_px[
                np.clip(
                    owner.result.front_indices,
                    0,
                    len(owner.arclength_px) - 1,
                )
            ]
            / analysis_scale,
            0.0,
        )
        causal_onset = _first_threshold_sample(
            causal_lengths,
            germination_length_px,
        )
        owner.native_prefix_lengths_px = lengths
        owner.native_onset_sample = onset
        owner.native_last_dormant_sample = last_dormant
        owner.native_certificate = certify_native_path_growth(
            lengths,
            native_onset_sample=onset,
            causal_onset_sample=causal_onset,
            total_length_px=owner.result.final_length_px / analysis_scale,
            minimum_completion_fraction=minimum_completion_fraction,
            maximum_onset_delay_samples=maximum_onset_delay_samples,
        )
        if owner.native_certificate.accepted:
            _promote_native_prefix_timeline(owner, lengths, analysis_scale)
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": owner.native_certificate.accepted,
                "reason": owner.native_certificate.reason,
                "completion_fraction": owner.native_certificate.completion_fraction,
                "onset_delay_samples": owner.native_certificate.onset_delay_samples,
                "final_native_length_px": float(lengths[-1]),
            }
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "verified_owner_ids": [
            owner.track_id
            for owner in candidates
            if owner.native_certificate is not None and owner.native_certificate.accepted
        ],
        "results": results,
    }


def _refine_centerlines_at_native_resolution(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    germination_length_px: float,
    search_radius_px: int = 8,
    observation_count: int = 8,
    minimum_observations: int = 3,
    maximum_length_change_fraction: float = 0.20,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Center supported material paths between native phase-contrast walls."""

    candidates = []
    samples_by_id: dict[int, np.ndarray] = {}
    for owner in owners:
        if (
            owner.result is None
            or not _measurement_eligible(owner)
            or owner.growth_certificate is None
            or not owner.growth_certificate.accepted
            or owner.direction_certificate is None
            or not owner.direction_certificate.accepted
            or owner.ownership_conflict_ids
            or owner.result.final_point_index < 2
            or _owner_onset_sample(
                owner,
                germination_length_px * analysis_scale,
            )
            is None
        ):
            continue
        complete = np.flatnonzero(
            owner.result.front_indices == owner.result.final_point_index
        )
        selected = complete[-observation_count:]
        samples_by_id[owner.track_id] = selected
        candidates.append(owner)

    scheduled: dict[int, list[OwnerGeometry]] = {}
    for owner in candidates:
        for sample in samples_by_id[owner.track_id]:
            scheduled.setdefault(int(sample), []).append(owner)
    surfaces_by_id: dict[int, list[np.ndarray]] = {
        owner.track_id: [] for owner in candidates
    }
    offsets_px = np.arange(
        -float(search_radius_px),
        float(search_radius_px) + 1.0,
        dtype=np.float32,
    )
    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample in sorted(scheduled):
        gray = _read_native_gray(
            capture,
            int(source_frames[sample]),
            native_frame_cache,
        )
        for owner in scheduled[sample]:
            point_count = owner.result.final_point_index + 1
            source_path_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            surfaces_by_id[owner.track_id].append(
                native_ribbon_score_surface(gray, source_path_yx, offsets_px)
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        selected_samples = samples_by_id[owner.track_id]
        surfaces = surfaces_by_id[owner.track_id]
        if len(surfaces) < minimum_observations:
            results.append(
                {
                    "owner_id": owner.track_id,
                    "accepted": False,
                    "reason": "insufficient-complete-native-observations",
                    "observation_count": len(surfaces),
                }
            )
            continue
        refinement = fit_native_centerline_offsets(
            np.asarray(surfaces),
            offsets_px,
        )
        point_count = owner.result.final_point_index + 1
        guides = None
        refined_arc = None
        length_change_fraction = 0.0
        if refinement.accepted:
            guides = owner.dynamic_paths_xy[:, :point_count].copy()
            tangents = np.gradient(guides, axis=1)
            tangents /= np.maximum(
                np.linalg.norm(tangents, axis=2, keepdims=True),
                1e-6,
            )
            right_normals = np.stack(
                (tangents[..., 1], -tangents[..., 0]),
                axis=2,
            )
            guides += (
                analysis_scale
                * refinement.offsets_px[None, :, None]
                * right_normals
            )
            refined_arc = np.concatenate(
                (
                    [0.0],
                    np.cumsum(
                        np.linalg.norm(np.diff(guides[-1], axis=0), axis=1)
                    ),
                )
            )
            length_change_fraction = abs(
                float(refined_arc[-1]) - owner.result.final_length_px
            ) / max(owner.result.final_length_px, 1e-6)
            if length_change_fraction > maximum_length_change_fraction:
                refinement = replace(
                    refinement,
                    accepted=False,
                    reason="native-refinement-changes-length-too-much",
                )
        owner.native_centerline_refinement = refinement
        owner.native_refinement_observation_samples = [
            int(sample) for sample in selected_samples
        ]
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": refinement.accepted,
                "reason": refinement.reason,
                "observation_count": len(surfaces),
                "normalized_median_gain": refinement.normalized_median_gain,
                "positive_point_fraction": refinement.positive_point_fraction,
                "positive_time_fraction": refinement.positive_time_fraction,
                "search_edge_fraction": refinement.search_edge_fraction,
                "offset_p90_source_px": float(
                    np.percentile(np.abs(refinement.offsets_px), 90.0)
                ),
                "length_change_fraction": length_change_fraction,
            }
        )
        if not refinement.accepted:
            continue
        assert guides is not None and refined_arc is not None
        owner.dynamic_paths_xy[:, :point_count] = project_inextensible_paths(
            guides,
            refined_arc,
        )
        previous_arc = owner.arclength_px.copy()
        owner.arclength_px[:point_count] = refined_arc
        if point_count < len(owner.arclength_px):
            owner.arclength_px[point_count:] = (
                refined_arc[-1]
                + previous_arc[point_count:]
                - previous_arc[point_count - 1]
            )
        owner.result = replace(
            owner.result,
            final_length_px=float(refined_arc[-1]),
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "refined_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "search_radius_source_px": search_radius_px,
        "maximum_observations": observation_count,
        "minimum_observations": minimum_observations,
        "maximum_length_change_fraction": maximum_length_change_fraction,
        "results": results,
    }


def _recenter_length_change_is_lateral(
    length_change_fraction: float,
    maximum_length_change_fraction: float = 0.05,
) -> bool:
    """A lateral recenter must preserve arclength.

    Shifting a ~150px curve laterally by at most the 16px search radius
    changes its length by a few percent at most.  Dense P97 proved the
    failure mode: a 3px median shift lengthened the path 14.6% by sliding
    along the structure into a neighboring grain cluster — a structural
    change, not a lateral fix — while passing the old 0.20 bound.
    """
    return length_change_fraction <= maximum_length_change_fraction


def _gate_rescue_recenter(
    refinement: NativeCenterlineRefinement,
    minimum_median_abs_offset_px: float,
) -> tuple[bool, str]:
    """Decide whether a fitted rescue recentering should move the path.

    The fit's own gain/consistency gates come first; then require a
    material correction so already-centered claims (dense P58: fitted
    offsets wander but gain reads 0.0) are never jittered.
    """

    if not refinement.accepted:
        return False, refinement.reason
    refinable = np.arange(len(refinement.offsets_px)) >= 2
    median_abs = float(np.median(np.abs(refinement.offsets_px[refinable])))
    if median_abs < minimum_median_abs_offset_px:
        return False, "rescue-path-already-centered"
    return True, "rescue-recenter-verified"


def _recenter_rescued_paths_at_native_resolution(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    search_radius_px: int = 16,
    observation_count: int = 8,
    minimum_observations: int = 3,
    maximum_length_change_fraction: float = 0.05,
    minimum_median_abs_offset_px: float = 3.0,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Laterally recenter rescued native-extension paths inside wide corridors.

    Rescue selection maximizes coverage and length, which can pick a
    candidate running parallel to the true tube (dense-field P97: proximal
    third ~10 source px off-axis at 0.97 coverage, selection reason
    verified-faint-extension-of-rooted-prefix).  The pre-rescue refinement
    stage never sees rescued geometry, so re-fit the smooth normal
    correction here on rescued paths, both prefix-extension and consensus
    selections (dense P97's off-axis claim is a consensus path).  The corridor is wider than the
    pre-rescue radius-8 stage because off-axis claims verified in the field
    sit 8-15 px from the true walls; corrections that fail the fit's
    gain/consistency gates (dense P76 rim-latch, P58 centered) or that move
    less than ``minimum_median_abs_offset_px`` leave the path untouched.
    A lateral fix must also preserve arclength (``_recenter_length_change_is_lateral``):
    dense P97's accepted recenter slid 14.6% longer into a neighbor grain
    cluster, proving the ribbon surface's strongest ridge can be the wrong
    structure — no surface-guided shift may lengthen the path.
    """

    candidates = []
    samples_by_id: dict[int, np.ndarray] = {}
    for owner in owners:
        if (
            owner.result is None
            or owner.selected_mature_path is None
            or not owner.selected_mature_path.startswith(
                ("native-prefix-extension-", "native-consensus-")
            )
            or owner.result.final_point_index < 5
        ):
            continue
        complete = np.flatnonzero(
            owner.result.front_indices == owner.result.final_point_index
        )
        if len(complete) < minimum_observations:
            continue
        samples_by_id[owner.track_id] = complete[-observation_count:]
        candidates.append(owner)

    scheduled: dict[int, list[OwnerGeometry]] = {}
    for owner in candidates:
        for sample in samples_by_id[owner.track_id]:
            scheduled.setdefault(int(sample), []).append(owner)
    surfaces_by_id: dict[int, list[np.ndarray]] = {
        owner.track_id: [] for owner in candidates
    }
    offsets_px = np.arange(
        -float(search_radius_px),
        float(search_radius_px) + 1.0,
        dtype=np.float32,
    )
    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample in sorted(scheduled):
        gray = _read_native_gray(
            capture,
            int(source_frames[sample]),
            native_frame_cache,
        )
        for owner in scheduled[sample]:
            point_count = owner.result.final_point_index + 1
            if sample >= owner.dynamic_paths_xy.shape[0]:
                surfaces_by_id[owner.track_id].append(None)
                continue
            source_path_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            surfaces_by_id[owner.track_id].append(
                native_ribbon_score_surface(gray, source_path_yx, offsets_px)
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        selected_samples = samples_by_id[owner.track_id]
        surfaces = [
            surface
            for surface, sample in zip(
                surfaces_by_id[owner.track_id], selected_samples
            )
            if surface is not None
            and np.isfinite(np.asarray(surface)).all()
            and np.asarray(surface).shape
            == (owner.result.final_point_index + 1, len(offsets_px))
        ]
        if len(surfaces) < minimum_observations:
            results.append(
                {
                    "owner_id": owner.track_id,
                    "accepted": False,
                    "reason": "insufficient-finite-recenter-observations",
                    "observation_count": len(surfaces),
                    "normalized_median_gain": 0.0,
                    "positive_point_fraction": 0.0,
                    "positive_time_fraction": 0.0,
                    "search_edge_fraction": 0.0,
                    "median_abs_offset_px": 0.0,
                    "offset_p90_source_px": 0.0,
                    "length_change_fraction": 0.0,
                }
            )
            continue
        print(
            f"recenter {owner.track_id:03d}: "
            f"{owner.selected_mature_path} "
            f"{len(surfaces)} observations",
            flush=True,
        )
        point_count = owner.result.final_point_index + 1
        try:
            refinement = fit_native_centerline_offsets(
                np.asarray(surfaces),
                offsets_px,
            )
        except (ValueError, RuntimeError) as exc:
            results.append(
                {
                    "owner_id": owner.track_id,
                    "accepted": False,
                    "reason": f"recenter-fit-failed:{type(exc).__name__}",
                    "observation_count": len(surfaces),
                    "normalized_median_gain": 0.0,
                    "positive_point_fraction": 0.0,
                    "positive_time_fraction": 0.0,
                    "search_edge_fraction": 0.0,
                    "median_abs_offset_px": 0.0,
                    "offset_p90_source_px": 0.0,
                    "length_change_fraction": 0.0,
                }
            )
            continue
        apply, reason = _gate_rescue_recenter(
            refinement, minimum_median_abs_offset_px
        )
        guides = None
        refined_arc = None
        length_change_fraction = 0.0
        if apply:
            guides = owner.dynamic_paths_xy[:, :point_count].copy()
            tangents = np.gradient(guides, axis=1)
            tangents /= np.maximum(
                np.linalg.norm(tangents, axis=2, keepdims=True),
                1e-6,
            )
            right_normals = np.stack(
                (tangents[..., 1], -tangents[..., 0]),
                axis=2,
            )
            guides += (
                analysis_scale
                * refinement.offsets_px[None, :, None]
                * right_normals
            )
            refined_arc = np.concatenate(
                (
                    [0.0],
                    np.cumsum(
                        np.linalg.norm(np.diff(guides[-1], axis=0), axis=1)
                    ),
                )
            )
            length_change_fraction = abs(
                float(refined_arc[-1]) - owner.result.final_length_px
            ) / max(owner.result.final_length_px, 1e-6)
            if not _recenter_length_change_is_lateral(
                length_change_fraction, maximum_length_change_fraction
            ):
                apply = False
                reason = "rescue-recenter-changes-length-too-much"
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": bool(apply),
                "reason": reason,
                "observation_count": len(surfaces),
                "normalized_median_gain": refinement.normalized_median_gain,
                "positive_point_fraction": refinement.positive_point_fraction,
                "positive_time_fraction": refinement.positive_time_fraction,
                "search_edge_fraction": refinement.search_edge_fraction,
                "median_abs_offset_px": float(
                    np.median(
                        np.abs(refinement.offsets_px[np.arange(len(refinement.offsets_px)) >= 2])
                    )
                ),
                "offset_p90_source_px": float(
                    np.percentile(np.abs(refinement.offsets_px), 90.0)
                ),
                "length_change_fraction": length_change_fraction,
            }
        )
        if not apply:
            continue
        assert guides is not None and refined_arc is not None
        owner.dynamic_paths_xy[:, :point_count] = project_inextensible_paths(
            guides,
            refined_arc,
        )
        previous_arc = owner.arclength_px.copy()
        owner.arclength_px[:point_count] = refined_arc
        if point_count < len(owner.arclength_px):
            owner.arclength_px[point_count:] = (
                refined_arc[-1]
                + previous_arc[point_count:]
                - previous_arc[point_count - 1]
            )
        owner.result = replace(
            owner.result,
            final_length_px=float(refined_arc[-1]),
        )
        owner.native_centerline_refinement = refinement
        owner.native_refinement_observation_samples = [
            int(sample) for sample in selected_samples
        ]
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "recentered_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "search_radius_source_px": search_radius_px,
        "maximum_observations": observation_count,
        "minimum_observations": minimum_observations,
        "maximum_length_change_fraction": maximum_length_change_fraction,
        "minimum_median_abs_offset_px": minimum_median_abs_offset_px,
        "results": results,
    }


def _certify_retained_paths_at_native_resolution(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    observation_count: int = 8,
    minimum_observations: int = 3,
    minimum_completion_fraction: float = 0.85,
    minimum_median_coverage: float = 0.40,
    minimum_distal_support_fraction: float = 0.40,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Withhold retained geometry that lacks repeated native paired walls."""

    candidates = [
        owner
        for owner in owners
        if owner.result is not None
        and _measurement_eligible(owner)
        and owner.growth_certificate is not None
        and owner.growth_certificate.accepted
        and owner.direction_certificate is not None
        and owner.direction_certificate.accepted
        and not owner.ownership_conflict_ids
        and owner.result.final_point_index >= 2
    ]
    samples_by_id: dict[int, np.ndarray] = {}
    point_count_by_id: dict[int, int] = {}
    for owner in candidates:
        final_count = owner.result.final_point_index + 1
        minimum_count = max(3, int(np.ceil(minimum_completion_fraction * final_count)))
        eligible = np.flatnonzero(owner.result.front_indices + 1 >= minimum_count)
        selected = eligible[-observation_count:]
        samples_by_id[owner.track_id] = selected
        point_count_by_id[owner.track_id] = (
            int(np.min(owner.result.front_indices[selected] + 1))
            if len(selected)
            else 0
        )

    scheduled: dict[int, list[OwnerGeometry]] = {}
    for owner in candidates:
        for sample in samples_by_id[owner.track_id]:
            scheduled.setdefault(int(sample), []).append(owner)
    support_by_id: dict[int, list[np.ndarray]] = {
        owner.track_id: [] for owner in candidates
    }
    thresholds_by_id: dict[int, list[float]] = {
        owner.track_id: [] for owner in candidates
    }
    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample in sorted(scheduled):
        gray = _read_native_gray(
            capture,
            int(source_frames[sample]),
            native_frame_cache,
        )
        for owner in scheduled[sample]:
            point_count = point_count_by_id[owner.track_id]
            source_path_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            support, controls = sample_native_paired_support(
                gray,
                source_path_yx,
                normal_search_offsets_px=(-3.0, 0.0, 3.0),
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support_by_id[owner.track_id].append(support)
            thresholds_by_id[owner.track_id].append(
                max(2.0, control_median + 2.0 * control_noise)
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        final_count = owner.result.final_point_index + 1
        point_count = point_count_by_id[owner.track_id]
        support = support_by_id[owner.track_id]
        thresholds = thresholds_by_id[owner.track_id]
        if support:
            values = np.asarray(support)
        else:
            values = np.empty((0, max(point_count, 1)), dtype=np.float32)
        certificate = certify_native_path_quality(
            values,
            np.asarray(thresholds),
            completion_fraction=point_count / final_count,
            minimum_observations=minimum_observations,
            minimum_completion_fraction=minimum_completion_fraction,
            minimum_median_coverage=minimum_median_coverage,
            minimum_distal_support_fraction=minimum_distal_support_fraction,
        )
        owner.native_path_quality_certificate = certificate
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": certificate.accepted,
                "reason": certificate.reason,
                "median_coverage": certificate.median_coverage,
                "median_support_margin": certificate.median_support_margin,
                "observation_count": certificate.observation_count,
                "completion_fraction": certificate.completion_fraction,
            }
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "verified_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "withheld_owner_ids": [
            int(row["owner_id"]) for row in results if not row["accepted"]
        ],
        "maximum_observations": observation_count,
        "minimum_observations": minimum_observations,
        "minimum_completion_fraction": minimum_completion_fraction,
        "minimum_median_coverage": minimum_median_coverage,
        "results": results,
    }


def _audit_unresolved_native_path_recovery(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    search_radius_px: int = 24,
    observation_count: int = 12,
    minimum_holdout_observations: int = 3,
    minimum_holdout_coverage: float = 0.40,
    minimum_holdout_gain: float = 0.15,
    maximum_length_change_fraction: float = 0.20,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Seek broad native corrections for rejected paths on held-out frames."""

    candidates = [
        owner
        for owner in owners
        if owner.result is not None
        and owner.native_path_quality_certificate is not None
        and not owner.native_path_quality_certificate.accepted
        and owner.result.final_point_index >= 2
    ]
    train_by_id: dict[int, np.ndarray] = {}
    holdout_by_id: dict[int, np.ndarray] = {}
    for owner in candidates:
        complete = np.flatnonzero(
            owner.result.front_indices == owner.result.final_point_index
        )[-observation_count:]
        train_by_id[owner.track_id] = complete[::2]
        holdout_by_id[owner.track_id] = complete[1::2]

    offsets_px = np.arange(
        -float(search_radius_px),
        float(search_radius_px) + 1.0,
        dtype=np.float32,
    )
    scheduled: dict[int, list[OwnerGeometry]] = {}
    for owner in candidates:
        for sample in train_by_id[owner.track_id]:
            scheduled.setdefault(int(sample), []).append(owner)
    surfaces_by_id: dict[int, list[np.ndarray]] = {
        owner.track_id: [] for owner in candidates
    }
    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample in sorted(scheduled):
        gray = _read_native_gray(
            capture,
            int(source_frames[sample]),
            native_frame_cache,
        )
        for owner in scheduled[sample]:
            point_count = owner.result.final_point_index + 1
            source_path_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            surfaces_by_id[owner.track_id].append(
                native_ribbon_score_surface(gray, source_path_yx, offsets_px)
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        train_samples = train_by_id[owner.track_id]
        holdout_samples = holdout_by_id[owner.track_id]
        owner.native_recovery_holdout_samples = [
            int(sample) for sample in holdout_samples
        ]
        if (
            len(train_samples) < minimum_holdout_observations
            or len(holdout_samples) < minimum_holdout_observations
        ):
            owner.native_recovery_reason = "insufficient-recovery-observations"
            results.append(
                {
                    "owner_id": owner.track_id,
                    "accepted": False,
                    "reason": owner.native_recovery_reason,
                    "training_observations": len(train_samples),
                    "holdout_observations": len(holdout_samples),
                }
            )
            continue
        refinement = fit_native_centerline_offsets(
            np.asarray(surfaces_by_id[owner.track_id]),
            offsets_px,
            root_lock_points=3,
            maximum_step_px=2.0,
            magnitude_penalty=0.015,
        )
        point_count = owner.result.final_point_index + 1
        guides = owner.dynamic_paths_xy[:, :point_count].copy()
        tangents = np.gradient(guides, axis=1)
        tangents /= np.maximum(
            np.linalg.norm(tangents, axis=2, keepdims=True),
            1e-6,
        )
        right_normals = np.stack((tangents[..., 1], -tangents[..., 0]), axis=2)
        guides += (
            analysis_scale
            * refinement.offsets_px[None, :, None]
            * right_normals
        )
        refined_arc = np.concatenate(
            (
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(guides[-1], axis=0), axis=1)),
            )
        )
        length_change_fraction = abs(
            float(refined_arc[-1]) - owner.result.final_length_px
        ) / max(owner.result.final_length_px, 1e-6)
        recovered_paths = project_inextensible_paths(guides, refined_arc)
        owner.native_recovery_refinement = refinement
        owner.native_recovery_paths_xy = recovered_paths
        owner.native_recovery_arclength_px = refined_arc

        original_coverages = []
        recovered_coverages = []
        validation_capture = _open_fallback_capture(movie, native_frame_cache)
        for sample in holdout_samples:
            gray = _read_native_gray(
                validation_capture,
                int(source_frames[sample]),
                native_frame_cache,
            )
            original_yx = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            recovered_yx = _source_path(
                recovered_paths[sample],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            original_support, original_controls = sample_native_paired_support(
                gray,
                original_yx,
                normal_search_offsets_px=(-2.0, 0.0, 2.0),
            )
            recovered_support, recovered_controls = sample_native_paired_support(
                gray,
                recovered_yx,
                normal_search_offsets_px=(-2.0, 0.0, 2.0),
            )
            controls = np.concatenate((original_controls, recovered_controls))
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            threshold = max(2.0, control_median + 2.0 * control_noise)
            original_coverages.append(float(np.mean(original_support >= threshold)))
            recovered_coverages.append(float(np.mean(recovered_support >= threshold)))
        _release_fallback_capture(validation_capture)
        original_coverage = float(np.median(original_coverages))
        recovered_coverage = float(np.median(recovered_coverages))
        holdout_gain = recovered_coverage - original_coverage
        owner.native_recovery_original_coverage = original_coverage
        owner.native_recovery_coverage = recovered_coverage

        if not refinement.accepted:
            reason = refinement.reason
        elif length_change_fraction > maximum_length_change_fraction:
            reason = "native-recovery-changes-length-too-much"
        elif recovered_coverage < minimum_holdout_coverage:
            reason = "insufficient-held-out-native-coverage"
        elif holdout_gain < minimum_holdout_gain:
            reason = "insufficient-held-out-native-gain"
        else:
            reason = "held-out-native-path-recovery-verified"
        owner.native_recovery_reason = reason
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": reason == "held-out-native-path-recovery-verified",
                "reason": reason,
                "training_observations": len(train_samples),
                "holdout_observations": len(holdout_samples),
                "training_median_gain": refinement.normalized_median_gain,
                "training_positive_point_fraction": (
                    refinement.positive_point_fraction
                ),
                "training_search_edge_fraction": refinement.search_edge_fraction,
                "offset_p90_source_px": float(
                    np.percentile(np.abs(refinement.offsets_px), 90.0)
                ),
                "original_holdout_coverage": original_coverage,
                "recovered_holdout_coverage": recovered_coverage,
                "holdout_coverage_gain": holdout_gain,
                "length_change_fraction": length_change_fraction,
            }
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "verified_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "search_radius_source_px": search_radius_px,
        "maximum_observations": observation_count,
        "minimum_holdout_observations": minimum_holdout_observations,
        "minimum_holdout_coverage": minimum_holdout_coverage,
        "minimum_holdout_gain": minimum_holdout_gain,
        "maximum_length_change_fraction": maximum_length_change_fraction,
        "results": results,
    }


def _certify_native_candidate_geometries(
    movie: Path,
    candidates: list[MaturePathCandidate],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    *,
    observation_count: int = 8,
    minimum_completion_fraction: float = 0.85,
    minimum_median_coverage: float = 0.50,
    minimum_distal_support_fraction: float = 0.40,
    minimum_emergence_gain: float = 0.30,
    native_frame_cache: NativeGrayFrameCache | None = None,
    minimum_proximal_support_ratio: float = 0.0,
) -> dict[int, NativePathQualityCertificate]:
    """Evaluate all proposed paths while decoding each source frame once."""

    if not candidates:
        return {}
    point_count_by_id: dict[int, int] = {}
    support_by_id: dict[int, list[np.ndarray]] = {}
    thresholds_by_id: dict[int, list[float]] = {}
    warmup_by_id: dict[int, np.ndarray] = {}
    scheduled: dict[int, list[MaturePathCandidate]] = {}
    for candidate in candidates:
        assert candidate.result is not None
        key = id(candidate)
        final_count = candidate.result.final_point_index + 1
        minimum_count = max(
            3,
            int(np.ceil(minimum_completion_fraction * final_count)),
        )
        eligible = np.flatnonzero(
            candidate.result.front_indices + 1 >= minimum_count
        )
        samples = eligible[-observation_count:]
        point_count_by_id[key] = (
            int(np.min(candidate.result.front_indices[samples] + 1))
            if len(samples)
            else 0
        )
        support_by_id[key] = []
        thresholds_by_id[key] = []
        for sample in samples:
            scheduled.setdefault(int(sample), []).append(candidate)

    if scheduled:
        capture = _open_fallback_capture(movie, native_frame_cache)
        warmup_gray = _read_native_gray(
            capture,
            int(source_frames[0]),
            native_frame_cache,
        )
        for candidate in candidates:
            key = id(candidate)
            warmup_count = int(point_count_by_id[key])
            if warmup_count >= 2:
                warmup_path_yx = _source_path(
                    candidate.dynamic_paths_xy[
                        0,
                        :warmup_count,
                    ],
                    shifts_xy[0],
                    analysis_scale,
                )[:, ::-1]
                warmup_values, _ = sample_native_paired_support(
                    warmup_gray,
                    warmup_path_yx,
                    normal_search_offsets_px=(0.0,),
                )
                warmup_by_id[key] = warmup_values
        for sample in sorted(scheduled):
            gray = _read_native_gray(
                capture,
                int(source_frames[sample]),
                native_frame_cache,
            )
            for candidate in scheduled[sample]:
                key = id(candidate)
                path_yx = _source_path(
                    candidate.dynamic_paths_xy[
                        sample,
                        : point_count_by_id[key],
                    ],
                    shifts_xy[sample],
                    analysis_scale,
                )[:, ::-1]
                values, controls = sample_native_paired_support(
                    gray,
                    path_yx,
                    normal_search_offsets_px=(-3.0, 0.0, 3.0),
                )
                control_median = float(np.median(controls))
                control_noise = 1.4826 * float(
                    np.median(np.abs(controls - control_median))
                )
                support_by_id[key].append(values)
                thresholds_by_id[key].append(
                    max(2.0, control_median + 2.0 * control_noise)
                )
        _release_fallback_capture(capture)

    certificates = {}
    for candidate in candidates:
        assert candidate.result is not None
        key = id(candidate)
        final_count = candidate.result.final_point_index + 1
        point_count = point_count_by_id[key]
        support = support_by_id[key]
        certificates[key] = certify_native_path_quality(
            np.asarray(support)
            if support
            else np.empty((0, max(point_count, 1)), dtype=np.float32),
            np.asarray(thresholds_by_id[key]),
            completion_fraction=(
                point_count / final_count if final_count > 0 else 0.0
            ),
            minimum_completion_fraction=minimum_completion_fraction,
            minimum_median_coverage=minimum_median_coverage,
            minimum_distal_support_fraction=minimum_distal_support_fraction,
            warmup_support=warmup_by_id.get(key),
            minimum_emergence_gain=minimum_emergence_gain,
            minimum_proximal_support_ratio=minimum_proximal_support_ratio,
        )
    return certificates


def _replace_candidate_front_from_native_growth(
    candidate: MaturePathCandidate,
    native_lengths_source_px: np.ndarray,
    native_active: np.ndarray,
    analysis_scale: float,
) -> None:
    """Promote a boundary candidate using independently observed native growth."""

    if candidate.result is None:
        raise ValueError("candidate result is required before native promotion")
    lengths = np.asarray(native_lengths_source_px, dtype=np.float64) * analysis_scale
    active = np.asarray(native_active, dtype=bool)
    if lengths.shape != candidate.result.front_indices.shape:
        raise ValueError("native candidate timeline has the wrong sample count")
    if active.shape != (len(lengths), len(candidate.arclength_px)):
        raise ValueError("native candidate activity has the wrong shape")
    lengths = np.clip(np.maximum.accumulate(lengths), 0.0, candidate.arclength_px[-1])
    fronts = np.searchsorted(candidate.arclength_px, lengths, side="right") - 1
    fronts[lengths <= 0.0] = -1
    fronts = fronts.astype(np.int32)
    births = np.full(len(candidate.arclength_px), len(fronts), dtype=np.int32)
    direct = np.zeros(len(candidate.arclength_px), dtype=bool)
    eventual = np.zeros(len(candidate.arclength_px), dtype=bool)
    for point in range(len(candidate.arclength_px)):
        arrivals = np.flatnonzero(fronts >= point)
        if not len(arrivals):
            continue
        birth = int(arrivals[0])
        births[point] = birth
        direct[point] = active[birth, point]
        eventual[point] = np.any(active[birth : min(len(active), birth + 12), point])
    reached = births < len(fronts)
    final_point = int(fronts[-1])
    candidate.result = replace(
        candidate.result,
        birth_samples=births,
        front_indices=fronts,
        direct_support_mask=direct,
        eventual_support_mask=eventual,
        feasible=final_point >= 0,
        reason="native-paired-boundary-prefix",
        final_point_index=final_point,
        final_length_px=(
            0.0 if final_point < 0 else float(candidate.arclength_px[final_point])
        ),
        direct_support_fraction=(
            0.0 if not np.any(reached) else float(np.mean(direct[reached]))
        ),
        eventual_support_fraction=(
            0.0 if not np.any(reached) else float(np.mean(eventual[reached]))
        ),
    )


def _audit_native_boundary_growth(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    maximum_growth_analysis_px_per_sample: float,
    *,
    minimum_completion_fraction: float = 0.95,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Recover complete edge-censored growth from native paired-wall support."""

    rigid_candidates = [
        candidate
        for owner in owners
        if owner.field_status == "boundary_censored"
        for candidate in owner.native_boundary_candidates
        if candidate.boundary_contact and candidate.deformation is None
    ]
    if not rigid_candidates:
        return {"candidate_count": 0, "verified_candidates": [], "results": []}
    support_by_candidate = {
        id(candidate): np.zeros(
            (len(source_frames), len(candidate.arclength_px)),
            dtype=np.float32,
        )
        for candidate in rigid_candidates
    }
    thresholds_by_candidate = {
        id(candidate): np.zeros(len(source_frames), dtype=np.float32)
        for candidate in rigid_candidates
    }
    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample, source_frame in enumerate(source_frames):
        gray = _read_native_gray(capture, source_frame, native_frame_cache)
        for candidate in rigid_candidates:
            path_yx = _source_path(
                candidate.dynamic_paths_xy[sample],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            support, controls = sample_native_paired_support(
                gray,
                path_yx,
                normal_search_offsets_px=(-3.0, 0.0, 3.0),
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support_by_candidate[id(candidate)][sample] = support
            thresholds_by_candidate[id(candidate)][sample] = max(
                2.0,
                control_median + 2.0 * control_noise,
            )
    _release_fallback_capture(capture)

    results = []
    verified_sources = []
    maximum_growth_source_px = (
        maximum_growth_analysis_px_per_sample / analysis_scale
    )
    for candidate in rigid_candidates:
        key = id(candidate)
        source_arc = candidate.arclength_px / analysis_scale
        growth, _, onset, _ = native_connected_prefix_growth(
            support_by_candidate[key],
            thresholds_by_candidate[key],
            source_arc,
            warmup_samples=8,
            maximum_gap_points=2,
            minimum_strong_growth_px=3.0,
            minimum_weak_growth_px=2.0,
            maximum_growth_px_per_sample=maximum_growth_source_px,
        )
        growth = np.minimum(growth, source_arc[-1])
        completion = float(growth[-1] / max(source_arc[-1], 1e-6))
        if onset is None:
            reason = "native-boundary-growth-not-detected"
        elif completion < minimum_completion_fraction:
            reason = "incomplete-native-boundary-growth"
        else:
            reason = "native-boundary-growth-verified"
            active = support_by_candidate[key] >= thresholds_by_candidate[key][:, None]
            _replace_candidate_front_from_native_growth(
                candidate,
                growth,
                active,
                analysis_scale,
            )
            verified_sources.append(candidate.source)
        candidate.native_boundary_final_length_px = float(growth[-1])
        candidate.native_boundary_lengths_px = growth.copy()
        candidate.native_boundary_onset_sample = onset
        candidate.native_boundary_timeline_reason = reason
        results.append(
            {
                "candidate": candidate.source,
                "accepted": reason == "native-boundary-growth-verified",
                "reason": reason,
                "onset_sample": onset,
                "final_length_source_px": float(growth[-1]),
                "completion_fraction": completion,
                "maximum_growth_source_px_per_sample": maximum_growth_source_px,
            }
        )

    for owner in owners:
        candidates_by_source = {
            candidate.source: candidate
            for candidate in owner.native_boundary_candidates
        }
        for candidate in owner.native_boundary_candidates:
            if candidate.deformation is None:
                continue
            rigid_source = candidate.source.rsplit("-", 1)[0] + "-rigid"
            rigid = candidates_by_source[rigid_source]
            candidate.native_boundary_final_length_px = (
                rigid.native_boundary_final_length_px
            )
            candidate.native_boundary_lengths_px = rigid.native_boundary_lengths_px
            candidate.native_boundary_onset_sample = rigid.native_boundary_onset_sample
            candidate.native_boundary_timeline_reason = (
                rigid.native_boundary_timeline_reason
            )
            if rigid_source in verified_sources:
                candidate.result = rigid.result
    return {
        "candidate_count": len(rigid_candidates),
        "verified_candidates": verified_sources,
        "minimum_completion_fraction": minimum_completion_fraction,
        "results": results,
    }


def _audit_native_candidate_root_onsets(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    *,
    root_extent_source_px: float = 20.0,
    minimum_growth_source_px: float = 3.0,
    maximum_onset_difference_samples: int = 14,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> None:
    """Corroborate each proposed attachment from source-resolution emergence."""

    candidates = [
        candidate
        for owner in owners
        for candidate in owner.native_boundary_candidates
        if candidate.result is not None
    ]
    support_by_candidate: dict[int, np.ndarray] = {}
    thresholds_by_candidate: dict[int, np.ndarray] = {}
    point_count_by_candidate: dict[int, int] = {}
    for candidate in candidates:
        source_arc = candidate.arclength_px / analysis_scale
        point_count = max(
            2,
            min(
                len(source_arc),
                int(np.searchsorted(source_arc, root_extent_source_px, side="right")),
            ),
        )
        key = id(candidate)
        point_count_by_candidate[key] = point_count
        support_by_candidate[key] = np.zeros(
            (len(source_frames), point_count),
            dtype=np.float32,
        )
        thresholds_by_candidate[key] = np.zeros(len(source_frames), dtype=np.float32)

    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample, source_frame in enumerate(source_frames):
        gray = _read_native_gray(capture, source_frame, native_frame_cache)
        for candidate in candidates:
            key = id(candidate)
            point_count = point_count_by_candidate[key]
            root_paths_xy = (
                candidate.rigid_paths_xy
                if candidate.rigid_paths_xy is not None
                else candidate.dynamic_paths_xy
            )
            path_yx = _source_path(
                root_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            support, controls = sample_native_paired_support(
                gray,
                path_yx,
                normal_search_offsets_px=(-3.0, 0.0, 3.0),
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support_by_candidate[key][sample] = support
            thresholds_by_candidate[key][sample] = max(
                2.0,
                control_median + 2.0 * control_noise,
            )
    _release_fallback_capture(capture)

    for candidate in candidates:
        assert candidate.result is not None
        key = id(candidate)
        point_count = point_count_by_candidate[key]
        source_arc = candidate.arclength_px[:point_count] / analysis_scale
        _, _, native_onset, _ = native_connected_prefix_growth(
            support_by_candidate[key],
            thresholds_by_candidate[key],
            source_arc,
            warmup_samples=8,
            maximum_gap_points=2,
            minimum_strong_growth_px=minimum_growth_source_px,
            minimum_weak_growth_px=max(1.0, minimum_growth_source_px - 1.0),
        )
        causal_point = int(
            np.searchsorted(
                candidate.arclength_px,
                5.0 * analysis_scale,
                side="left",
            )
        )
        causal_samples = np.flatnonzero(candidate.result.front_indices >= causal_point)
        causal_onset = int(causal_samples[0]) if len(causal_samples) else None
        delay = (
            None
            if native_onset is None or causal_onset is None
            else int(native_onset - causal_onset)
        )
        if native_onset is None:
            reason = "native-root-emergence-not-detected"
        elif causal_onset is None:
            reason = "causal-root-emergence-not-detected"
        elif abs(delay) > maximum_onset_difference_samples:
            reason = "native-root-onset-corrected"
        else:
            reason = "native-root-onset-corroborated"
        candidate.native_root_onset_sample = native_onset
        candidate.native_root_onset_delay_samples = delay
        candidate.native_root_onset_reason = reason


def _requires_native_boundary_audit(
    owner: OwnerGeometry,
    owner_radius_px: float,
    minimum_independent_departure_radii: float = 1.0,
) -> bool:
    """Return whether geometry needs an independent source-resolution proposal."""

    if owner_radius_px <= 0.0 or minimum_independent_departure_radii <= 0.0:
        raise ValueError("native boundary audit distances must be positive")
    topology_failed = bool(
        owner.topology_certificate is not None
        and not owner.topology_certificate.accepted
    )
    short_departure = bool(
        owner.topology_certificate is not None
        and owner.topology_certificate.radial_excursion_px
        < minimum_independent_departure_radii * owner_radius_px
    )
    native_quality_failed = bool(
        owner.native_path_quality_certificate is not None
        and not owner.native_path_quality_certificate.accepted
    )
    return topology_failed or short_departure or native_quality_failed


def _audit_native_boundary_candidates(
    movie: Path,
    frames: np.ndarray,
    deformation_guide: np.ndarray,
    owners: list[OwnerGeometry],
    centers_by_id: dict[int, np.ndarray],
    body_status_by_id: dict[int, str],
    identity_tiers_by_id: dict[int, str],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    pollen_radius_px: float,
    warmup_samples: int,
    normal_halfwidth_px: float,
    normal_samples: int,
    max_growth_px_per_sample: float,
    direction_kwargs: dict,
    crop_radius_source_px: int = 150,
    native_frame_cache: NativeGrayFrameCache | None = None,
    minimum_proximal_support_ratio: float = 0.0,
) -> dict:
    """Generate native paired-boundary alternatives for unresolved owners."""

    unresolved = [
        owner
        for owner in owners
        if _requires_native_boundary_audit(
            owner,
            pollen_radius_px,
        )
    ]
    if not unresolved:
        return {"candidate_owner_ids": [], "verified_owner_ids": [], "results": []}
    near_samples = np.arange(max(0, len(source_frames) - 3), len(source_frames))
    tail_samples = np.arange(max(0, len(source_frames) - 11), len(source_frames))
    consensus_sample_groups = [("native-consensus-near", near_samples)]
    if not np.array_equal(tail_samples, near_samples):
        consensus_sample_groups.append(("native-consensus-tail", tail_samples))
    native_samples = np.unique(
        np.concatenate([samples for _, samples in consensus_sample_groups])
    )
    capture = _open_fallback_capture(movie, native_frame_cache)
    native_grays = {}
    for sample in native_samples:
        native_grays[int(sample)] = _read_native_gray(
            capture,
            int(source_frames[sample]),
            native_frame_cache,
        )
    _release_fallback_capture(capture)
    gray = native_grays[len(source_frames) - 1]
    source_centers_xy = {
        track_id: (centers[-1] + shifts_xy[-1]) / analysis_scale
        for track_id, centers in centers_by_id.items()
    }
    trace_metadata: dict[tuple[int, int], dict] = {}
    for owner in unresolved:
        source_center_xy = source_centers_xy[owner.track_id]
        blocking_ids = set(
            _blocking_centers_for_owner(
                centers_by_id,
                body_status_by_id,
                identity_tiers_by_id,
                owner.track_id,
            )
        )
        blocking_centers_by_id = {
            track_id: centers_by_id[track_id] for track_id in blocking_ids
        }
        x0 = max(0, int(np.floor(source_center_xy[0] - crop_radius_source_px)))
        x1 = min(gray.shape[1], int(np.ceil(source_center_xy[0] + crop_radius_source_px)))
        y0 = max(0, int(np.floor(source_center_xy[1] - crop_radius_source_px)))
        y1 = min(gray.shape[0], int(np.ceil(source_center_xy[1] + crop_radius_source_px)))
        crop = gray[y0:y1, x0:x1]
        local_center_yx = source_center_xy[::-1] - np.asarray((y0, x0))
        local_foreign = []
        for track_id, center_xy in source_centers_xy.items():
            if track_id == owner.track_id or track_id not in blocking_ids:
                continue
            center_yx = center_xy[::-1] - np.asarray((y0, x0))
            if (
                -30.0 < center_yx[0] < crop.shape[0] + 30.0
                and -30.0 < center_yx[1] < crop.shape[1] + 30.0
            ):
                local_foreign.append(center_yx)
        trace_config = NativeBoundaryTracingConfig(
            pollen_radius_px=pollen_radius_px / analysis_scale,
        )
        raw_terminal_mask = None
        if owner.field_status == "boundary_censored":
            raw_terminal_mask = source_boundary_terminal_mask(
                crop.shape,
                gray.shape,
                (y0, x0),
            )
        frame_proposals = trace_native_boundary_candidates(
            crop,
            local_center_yx,
            np.asarray(local_foreign, dtype=np.float64).reshape(-1, 2),
            trace_config,
        )
        faint_proposals = trace_native_boundary_candidates(
            crop,
            local_center_yx,
            np.asarray(local_foreign, dtype=np.float64).reshape(-1, 2),
            trace_config.for_faint_prefix_extension(),
        )
        prefix_extensions = tuple(
            proposal
            for proposal in faint_proposals
            if any(
                path_extends_rooted_prefix(
                    reference.trace.path_yx,
                    proposal.trace.path_yx,
                )
                for reference in frame_proposals
            )
        )
        proposal_groups = [
            (
                "native-boundary",
                frame_proposals,
                np.asarray((y0, x0), dtype=np.float64),
                False,
            )
        ]
        if prefix_extensions:
            proposal_groups.append(
                (
                    "native-prefix-extension",
                    prefix_extensions,
                    np.asarray((y0, x0), dtype=np.float64),
                    False,
                )
            )
        if raw_terminal_mask is not None and np.any(raw_terminal_mask):
            proposal_groups.append(
                (
                    "native-edge-boundary",
                    trace_native_boundary_candidates(
                        crop,
                        local_center_yx,
                        np.asarray(local_foreign, dtype=np.float64).reshape(-1, 2),
                        replace(
                            trace_config.for_boundary_censoring(),
                            output_count=1,
                        ),
                        terminal_mask=raw_terminal_mask,
                    ),
                    np.asarray((y0, x0), dtype=np.float64),
                    True,
                )
            )
        for consensus_source, consensus_samples in consensus_sample_groups:
            if len(consensus_samples) < 2:
                continue
            owner_centers_yx = np.asarray(
                [
                    (
                        centers_by_id[owner.track_id][sample]
                        + shifts_xy[sample]
                    )[::-1]
                    / analysis_scale
                    for sample in consensus_samples
                ],
                dtype=np.float64,
            )
            consensus = owner_aligned_temporal_consensus(
                [native_grays[int(sample)] for sample in consensus_samples],
                owner_centers_yx,
                crop_radius_source_px,
            )
            consensus_center_yx = np.full(2, crop_radius_source_px, dtype=np.float64)
            consensus_foreign = np.asarray(
                [
                    center_xy[::-1]
                    - source_center_xy[::-1]
                    + consensus_center_yx
                    for track_id, center_xy in source_centers_xy.items()
                    if track_id != owner.track_id
                    and track_id in blocking_ids
                    and np.all(
                        np.abs(center_xy[::-1] - source_center_xy[::-1])
                        < crop_radius_source_px + 30.0
                    )
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
            consensus_source_offset_yx = (
                source_center_xy[::-1] - consensus_center_yx
            )
            consensus_terminal_mask = None
            if owner.field_status == "boundary_censored":
                consensus_terminal_mask = source_boundary_terminal_mask(
                    consensus.shape,
                    gray.shape,
                    consensus_source_offset_yx,
                )
            proposal_groups.append(
                (
                    consensus_source,
                    trace_native_boundary_candidates(
                        consensus,
                        consensus_center_yx,
                        consensus_foreign,
                        replace(trace_config, output_count=3),
                    ),
                    consensus_source_offset_yx,
                    False,
                )
            )
            if consensus_terminal_mask is not None and np.any(
                consensus_terminal_mask
            ):
                proposal_groups.append(
                    (
                        consensus_source.replace("native-", "native-edge-", 1),
                        trace_native_boundary_candidates(
                            consensus,
                            consensus_center_yx,
                            consensus_foreign,
                            replace(
                                trace_config.for_boundary_censoring(),
                                output_count=1,
                            ),
                            terminal_mask=consensus_terminal_mask,
                        ),
                        consensus_source_offset_yx,
                        True,
                    )
                )
        owner.native_boundary_candidates = []
        flattened_proposals = [
            (proposal_source, proposal, source_offset_yx, boundary_targeted)
            for (
                proposal_source,
                proposals,
                source_offset_yx,
                boundary_targeted,
            ) in proposal_groups
            for proposal in proposals
        ]
        for index, (
            proposal_source,
            proposal,
            source_offset_yx,
            boundary_targeted,
        ) in enumerate(flattened_proposals):
            source_proposal_yx = proposal.trace.path_yx + source_offset_yx
            if (
                np.any(source_proposal_yx[0] < 0.0)
                or np.any(
                    source_proposal_yx[0]
                    > np.asarray(gray.shape, dtype=np.float64) - 1.0
                )
            ):
                # A traced root outside the source frame (edge-owner crop
                # offset) cannot become a candidate: skip it instead of
                # aborting the complete field run.
                continue
            if boundary_targeted:
                source_yx, boundary_contact = extend_path_to_image_boundary(
                    source_proposal_yx,
                    gray.shape,
                )
            else:
                source_yx, boundary_contact = clip_path_to_image_bounds(
                    source_proposal_yx,
                    gray.shape,
                )
                boundary_contact = False
            aligned_yx = (
                source_yx * analysis_scale - shifts_xy[-1, ::-1]
            )
            canonical_xy, arclength = _resample_path(
                aligned_yx[:, ::-1],
                1.0,
            )
            dynamic_paths = canonical_xy[None] + (
                centers_by_id[owner.track_id] - centers_by_id[owner.track_id][-1]
            )[:, None]
            (
                dynamic_paths,
                arclength,
                foreign_contact_owner,
                uncensored_length,
            ) = _foreign_owner_safe_prefix(
                dynamic_paths,
                arclength,
                track_id=owner.track_id,
                centers_by_id=blocking_centers_by_id,
                owner_radius_px=4.0 / 3.0 * pollen_radius_px,
            )
            boundary_contact = boundary_contact and foreign_contact_owner is None
            rigid_paths = project_inextensible_paths(dynamic_paths, arclength)
            deformation = fit_scalar_path_deformation(
                deformation_guide,
                rigid_paths,
                normal_radius_px=max(6.0, 2.0 * normal_halfwidth_px),
                offset_magnitude_penalty=0.05,
                material_smoothing_sigma=4.0,
            )
            deformed_paths = project_inextensible_paths(
                deformation.curves_xy,
                arclength,
            )
            pose_hypotheses = (
                ("rigid", rigid_paths, None),
                ("deformable", deformed_paths, deformation),
            )
            for pose_name, pose_paths, pose_deformation in pose_hypotheses:
                endpoint_source_yx = (
                    (pose_paths[-1, -1] + shifts_xy[-1]) / analysis_scale
                )[::-1]
                endpoint_edge_distance = min(
                    endpoint_source_yx[0],
                    endpoint_source_yx[1],
                    gray.shape[0] - 1.0 - endpoint_source_yx[0],
                    gray.shape[1] - 1.0 - endpoint_source_yx[1],
                )
                candidate = _candidate_geometry(
                    f"{proposal_source}-{index + 1}-{pose_name}",
                    pose_paths,
                    arclength,
                    centers_by_id[owner.track_id],
                    foreign_contact_owner_id=foreign_contact_owner,
                    uncensored_length_px=uncensored_length,
                    boundary_contact=bool(
                        boundary_contact and endpoint_edge_distance <= 2.0
                    ),
                )
                candidate.proposal_score = proposal.ranking_score
                candidate.deformation = pose_deformation
                candidate.rigid_paths_xy = rigid_paths.copy()
                owner.native_boundary_candidates.append(candidate)
                candidate_index = len(owner.native_boundary_candidates) - 1
                trace_metadata[(owner.track_id, candidate_index)] = {
                    "native_trace_length_px": proposal.trace.length_px,
                    "native_trace_paired_fraction": (
                        proposal.trace.paired_supported_fraction
                    ),
                    "native_trace_endpoint_openness": (
                        proposal.trace.endpoint_separation_fraction
                    ),
                    "native_trace_radial_extension_px": (
                        proposal.radial_extension_px
                    ),
                    "pose_model": pose_name,
                    "proposal_source": proposal_source,
                    "native_trace_boundary_contact": boundary_contact,
                }

    profiles: dict[tuple[int, int], list[np.ndarray]] = {
        (owner.track_id, index): []
        for owner in unresolved
        for index in range(len(owner.native_boundary_candidates))
    }
    for mode, minimum_change in (("dark", 3.0), ("dog", 0.8), ("blackhat", 0.8)):
        evidence = _feature_stack(frames, mode, pollen_radius_px)
        for owner in unresolved:
            for index, candidate in enumerate(owner.native_boundary_candidates):
                profiles[(owner.track_id, index)].append(
                    dynamic_path_novelty_profiles(
                        evidence,
                        candidate.dynamic_paths_xy,
                        warmup_samples=warmup_samples,
                        absolute_evidence_floor=0.0,
                        normal_halfwidth_px=normal_halfwidth_px,
                        normal_sample_count=normal_samples,
                        minimum_change=minimum_change,
                        noise_multiplier=2.0,
                        temporal_median_samples=3,
                    )
                )
        del evidence

    for owner in unresolved:
        for index, candidate in enumerate(owner.native_boundary_candidates):
            candidate.profile = np.mean(
                profiles[(owner.track_id, index)], axis=0
            ).astype(np.float32)
            candidate.result = causal_changepoint_front(
                candidate.profile,
                candidate.arclength_px,
                warmup_samples=warmup_samples,
                max_step_px=max_growth_px_per_sample,
                change_window_samples=7,
                persistence_window_samples=21,
                minimum_post_samples=1,
            )
            reverse_arclength = (
                candidate.arclength_px[-1] - candidate.arclength_px[::-1]
            )
            candidate.reverse_result = causal_changepoint_front(
                candidate.profile[:, ::-1],
                reverse_arclength,
                warmup_samples=warmup_samples,
                max_step_px=max_growth_px_per_sample,
                change_window_samples=7,
                persistence_window_samples=21,
                minimum_post_samples=1,
            )
    for owner in unresolved:
        candidates_by_source = {
            candidate.source: candidate
            for candidate in owner.native_boundary_candidates
        }
        for candidate in owner.native_boundary_candidates:
            rigid_source = candidate.source.rsplit("-", 1)[0] + "-rigid"
            rigid_candidate = candidates_by_source[rigid_source]
            candidate.temporal_reference_source = rigid_source
            if candidate.deformation is None:
                continue
            candidate.profile = rigid_candidate.profile
            candidate.result = rigid_candidate.result
            candidate.reverse_result = rigid_candidate.reverse_result

    native_boundary_growth = _audit_native_boundary_growth(
        movie,
        unresolved,
        source_frames,
        shifts_xy,
        analysis_scale,
        max_growth_px_per_sample,
        native_frame_cache=native_frame_cache,
    )
    all_candidates = [
        candidate
        for owner in unresolved
        for candidate in owner.native_boundary_candidates
    ]
    for owner in unresolved:
        for candidate in owner.native_boundary_candidates:
            _certify_candidate_topology(
                candidate,
                centers_by_id[owner.track_id],
                pollen_radius_px,
            )
            candidate.growth_certificate = certify_rooted_growth(
                _candidate_hypothesis(candidate),
                owner_radius_px=pollen_radius_px,
                boundary_censored=owner.field_status == "boundary_censored",
            )
            candidate.direction_certificate = certify_growth_direction(
                candidate.result,
                candidate.reverse_result,
                total_length_px=float(candidate.arclength_px[-1]),
                **direction_kwargs,
            )
    native_quality_by_candidate = _certify_native_candidate_geometries(
        movie,
        all_candidates,
        source_frames,
        shifts_xy,
        analysis_scale,
        minimum_distal_support_fraction=0.40,
        minimum_proximal_support_ratio=minimum_proximal_support_ratio,
        native_frame_cache=native_frame_cache,
    )
    for candidate in all_candidates:
        candidate.native_quality_certificate = native_quality_by_candidate[
            id(candidate)
        ]
    _audit_native_candidate_root_onsets(
        movie,
        unresolved,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )

    results = []
    for owner in unresolved:
        for index, candidate in enumerate(owner.native_boundary_candidates):
            assert candidate.result is not None
            assert candidate.growth_certificate is not None
            assert candidate.direction_certificate is not None
            assert candidate.native_quality_certificate is not None
            metadata = trace_metadata[(owner.track_id, index)]
            geometry_verified = bool(
                candidate.topology_certificate is not None
                and candidate.topology_certificate.accepted
                and candidate.growth_certificate.accepted
                and candidate.direction_certificate.accepted
                and candidate.native_quality_certificate.accepted
            )
            accepted = bool(
                geometry_verified
                and candidate.native_root_onset_reason
                in NATIVE_ONSET_ACCEPTED_REASONS
                and (
                    not candidate.boundary_contact
                    or candidate.native_boundary_timeline_reason
                    == "native-boundary-growth-verified"
                )
            )
            results.append(
                {
                    "owner_id": owner.track_id,
                    "candidate": candidate.source,
                    "accepted": accepted,
                    "geometry_verified": geometry_verified,
                    "proposal_score": candidate.proposal_score,
                    "temporal_reference": candidate.temporal_reference_source,
                    **metadata,
                    "causal_final_length_px": (
                        candidate.result.final_length_px / analysis_scale
                    ),
                    "causal_reason": candidate.result.reason,
                    "growth_reason": candidate.growth_certificate.reason,
                    "topology_reason": candidate.topology_certificate.reason,
                    "retained_radial_excursion_px": (
                        candidate.topology_certificate.radial_excursion_px
                        / analysis_scale
                    ),
                    "direction_reason": candidate.direction_certificate.reason,
                    "native_quality_reason": (
                        candidate.native_quality_certificate.reason
                    ),
                    "native_median_coverage": (
                        candidate.native_quality_certificate.median_coverage
                    ),
                    "native_proximal_support_ratio": (
                        candidate.native_quality_certificate.proximal_support_ratio
                    ),
                    "native_root_onset_reason": candidate.native_root_onset_reason,
                    "native_root_onset_sample": candidate.native_root_onset_sample,
                    "native_root_onset_delay_samples": (
                        candidate.native_root_onset_delay_samples
                    ),
                    "native_boundary_timeline_reason": (
                        candidate.native_boundary_timeline_reason
                    ),
                    "native_boundary_final_length_px": (
                        candidate.native_boundary_final_length_px
                    ),
                    "native_boundary_onset_sample": (
                        candidate.native_boundary_onset_sample
                    ),
                    "foreign_contact_owner_id": candidate.foreign_contact_owner_id,
                    "boundary_contact": candidate.boundary_contact,
                }
            )
    verified = sorted(
        {
            int(row["owner_id"])
            for row in results
            if row["accepted"]
        }
    )
    return {
        "candidate_owner_ids": [owner.track_id for owner in unresolved],
        "verified_owner_ids": verified,
        "crop_radius_source_px": crop_radius_source_px,
        "native_boundary_growth": native_boundary_growth,
        "results": results,
    }


def _exclude_short_consensus_paths(
    candidates: list[MaturePathCandidate],
    minimum_length_fraction: float,
    coverage_tie_tolerance: float,
) -> tuple[list[MaturePathCandidate], list[MaturePathCandidate], float | None]:
    """Keep consensus paths that preserve a verified frame path's extent."""

    if not 0.0 <= minimum_length_fraction <= 1.0:
        raise ValueError("consensus minimum length fraction must be in [0, 1]")
    verified_frame_candidates = [
        candidate
        for candidate in candidates
        if candidate.source.startswith("native-boundary-")
        and candidate.result is not None
    ]
    if not verified_frame_candidates:
        return candidates, [], None
    best_frame_coverage = max(
        candidate.native_quality_certificate.median_coverage
        for candidate in verified_frame_candidates
    )
    frame_coverage_tied = [
        candidate
        for candidate in verified_frame_candidates
        if candidate.native_quality_certificate.median_coverage
        >= best_frame_coverage - coverage_tie_tolerance
    ]
    reference_frame = max(
        frame_coverage_tied,
        key=lambda candidate: (
            candidate.proposal_score,
            candidate.native_quality_certificate.median_coverage,
        ),
    )
    minimum_consensus_length = (
        reference_frame.result.final_length_px * minimum_length_fraction
    )
    rejected = [
        candidate
        for candidate in candidates
        if candidate.source.startswith("native-consensus-")
        and candidate.result is not None
        and candidate.result.final_length_px < minimum_consensus_length
    ]
    rejected_ids = {id(candidate) for candidate in rejected}
    retained = [
        candidate for candidate in candidates if id(candidate) not in rejected_ids
    ]
    return retained, rejected, minimum_consensus_length


def _select_native_boundary_rescues(
    owners: list[OwnerGeometry],
    analysis_scale: float,
    owner_radius_px: float,
    coverage_tie_tolerance: float,
    deformation_minimum_coverage_gain: float,
    consensus_minimum_length_fraction: float,
    boundary_minimum_length_gain_fraction: float = 0.15,
) -> dict:
    """Replace unresolved geometry with the strongest corroborated native path."""

    if boundary_minimum_length_gain_fraction < 0.0:
        raise ValueError("boundary length gain fraction cannot be negative")
    selections = []
    rejected_short_consensus = []
    for owner in owners:
        if not _requires_native_boundary_audit(owner, owner_radius_px):
            continue
        if owner.body_status in {"rejected", "indeterminate"}:
            # A contested identity — duplicate seeds on one grain (dense
            # P65/P69, P80/P81) or split fragments (P89) — cannot take an
            # automatic measurement even when a neighbor's tube traces
            # cleanly from nearby.  The candidates stay exported as
            # diagnostics; selection simply does not happen.
            continue
        eligible = [
            candidate
            for candidate in owner.native_boundary_candidates
            if candidate.result is not None
            and candidate.reverse_result is not None
            and candidate.topology_certificate is not None
            and candidate.topology_certificate.accepted
            and candidate.growth_certificate is not None
            and candidate.growth_certificate.accepted
            and candidate.direction_certificate is not None
            and candidate.direction_certificate.accepted
            and candidate.native_quality_certificate is not None
            and candidate.native_quality_certificate.accepted
            and candidate.native_root_onset_reason
            in NATIVE_ONSET_ACCEPTED_REASONS
        ]
        if not eligible:
            continue
        candidates_by_source = {
            candidate.source: candidate
            for candidate in owner.native_boundary_candidates
        }
        pose_verified = []
        for candidate in eligible:
            if candidate.deformation is None:
                pose_verified.append(candidate)
                continue
            rigid_source = candidate.source.rsplit("-", 1)[0] + "-rigid"
            rigid_candidate = candidates_by_source[rigid_source]
            if (
                rigid_candidate.native_quality_certificate is None
                or not rigid_candidate.native_quality_certificate.accepted
            ):
                continue
            rigid_coverage = (
                rigid_candidate.native_quality_certificate.median_coverage
            )
            coverage_gain = (
                candidate.native_quality_certificate.median_coverage
                - rigid_coverage
            )
            if coverage_gain >= deformation_minimum_coverage_gain:
                pose_verified.append(candidate)
        eligible = pose_verified
        if not eligible:
            continue
        eligible, rejected, minimum_consensus_length = (
            _exclude_short_consensus_paths(
                eligible,
                consensus_minimum_length_fraction,
                coverage_tie_tolerance,
            )
        )
        for candidate in rejected:
            rejected_short_consensus.append(
                {
                    "owner_id": owner.track_id,
                    "candidate": candidate.source,
                    "final_length_source_px": (
                        candidate.result.final_length_px / analysis_scale
                    ),
                    "minimum_length_source_px": (
                        minimum_consensus_length / analysis_scale
                    ),
                    "reason": "shorter-than-verified-frame-path",
                }
            )
        eligible = [
            candidate
            for candidate in eligible
            if not candidate.source.startswith("native-edge-")
            or (
                candidate.source.startswith("native-edge-boundary-")
                and candidate.boundary_contact
                and candidate.native_boundary_timeline_reason
                == "native-boundary-growth-verified"
            )
        ]
        if not eligible:
            continue
        verified_edge = [
            candidate
            for candidate in eligible
            if candidate.source.startswith("native-edge-")
        ]
        ordinary = [
            candidate
            for candidate in eligible
            if not candidate.source.startswith("native-edge-")
            and not candidate.source.startswith("native-prefix-extension-")
        ]
        ordinary_length = max(
            (candidate.result.final_length_px for candidate in ordinary),
            default=0.0,
        )
        if ordinary:
            best_ordinary_coverage = max(
                candidate.native_quality_certificate.median_coverage
                for candidate in ordinary
            )
            reference_ordinary = max(
                (
                    candidate
                    for candidate in ordinary
                    if candidate.native_quality_certificate.median_coverage
                    >= best_ordinary_coverage - coverage_tie_tolerance
                ),
                key=lambda candidate: (
                    candidate.proposal_score,
                    candidate.result.final_length_px,
                ),
            )
            reference_ordinary_length = (
                reference_ordinary.result.final_length_px
            )
        else:
            reference_ordinary_length = 0.0
        decisive_edge = [
            candidate
            for candidate in verified_edge
            if candidate.result.final_length_px
            >= ordinary_length * (1.0 + boundary_minimum_length_gain_fraction)
        ]
        decisive_edge_ids = {id(candidate) for candidate in decisive_edge}
        decisive_extensions = [
            candidate
            for candidate in eligible
            if candidate.source.startswith("native-prefix-extension-")
            and candidate.result.final_length_px
            >= reference_ordinary_length + 6.0 * analysis_scale
            and candidate.native_quality_certificate.median_coverage >= 0.5
        ]
        decisive_extension_ids = {
            id(candidate) for candidate in decisive_extensions
        }
        selection_pool = decisive_edge or decisive_extensions or eligible
        best_coverage = max(
            candidate.native_quality_certificate.median_coverage
            for candidate in selection_pool
        )
        coverage_tied = [
            candidate
            for candidate in selection_pool
            if candidate.native_quality_certificate.median_coverage
            >= best_coverage - coverage_tie_tolerance
        ]
        selected = max(
            coverage_tied,
            key=lambda candidate: (
                candidate.proposal_score,
                candidate.native_quality_certificate.median_coverage,
            ),
        )
        owner.selected_mature_path = selected.source
        owner.mature_path_selection_reason = (
            "verified-edge-path-with-substantial-length-gain"
            if id(selected) in decisive_edge_ids
            else (
                "verified-faint-extension-of-rooted-prefix"
                if id(selected) in decisive_extension_ids
                else "native-boundary-rescue-after-legacy-geometry-rejected"
            )
        )
        owner.selected_root_distance_px = selected.root_distance_px
        owner.selected_radial_excursion_px = selected.radial_excursion_px
        owner.selected_foreign_contact_owner_id = (
            selected.foreign_contact_owner_id
        )
        owner.shared_branch_owner_id = None
        owner.foreign_branch_owner_id = None
        owner.foreign_branch_capture_point = None
        owner.ownership_conflict_ids = []
        owner.arclength_px = selected.arclength_px.copy()
        owner.dynamic_paths_xy = selected.dynamic_paths_xy.copy()
        owner.rigid_paths_xy = (
            selected.rigid_paths_xy.copy()
            if selected.rigid_paths_xy is not None
            else selected.dynamic_paths_xy.copy()
        )
        rigid_source = selected.source.rsplit("-", 1)[0] + "-rigid"
        rigid_candidate = next(
            (
                candidate
                for candidate in owner.native_boundary_candidates
                if candidate.source == rigid_source
            ),
            selected,
        )
        owner.rigid_profile = rigid_candidate.profile.copy()
        owner.profile = selected.profile.copy()
        owner.rigid_result = rigid_candidate.result
        owner.deformed_result = (
            selected.result if selected.deformation is not None else None
        )
        owner.result = selected.result
        owner.reverse_result = selected.reverse_result
        owner.growth_certificate = selected.growth_certificate
        owner.topology_certificate = selected.topology_certificate
        owner.direction_certificate = selected.direction_certificate
        owner.native_path_quality_certificate = selected.native_quality_certificate
        owner.native_centerline_refinement = None
        owner.native_refinement_observation_samples = []
        owner.native_root_onset_sample = selected.native_root_onset_sample
        owner.native_root_onset_delay_samples = (
            selected.native_root_onset_delay_samples
        )
        owner.native_root_onset_reason = selected.native_root_onset_reason
        if selected.boundary_contact:
            owner.native_prefix_lengths_px = (
                None
                if selected.native_boundary_lengths_px is None
                else selected.native_boundary_lengths_px.copy()
            )
            owner.native_onset_sample = selected.native_boundary_onset_sample
            native_completion = selected.native_boundary_final_length_px / max(
                selected.arclength_px[-1] / analysis_scale,
                1e-6,
            )
            native_onset_delay = (
                None
                if selected.native_boundary_onset_sample is None
                or selected.native_root_onset_sample is None
                else selected.native_boundary_onset_sample
                - selected.native_root_onset_sample
            )
        else:
            owner.native_prefix_lengths_px = None
            owner.native_onset_sample = selected.native_root_onset_sample
            native_completion = selected.native_quality_certificate.completion_fraction
            native_onset_delay = selected.native_root_onset_delay_samples
        owner.native_last_dormant_sample = None
        owner.timeline_source = (
            "native-paired-boundary-prefix"
            if selected.boundary_contact
            else "coarse-causal"
        )
        owner.native_certificate = NativePathGrowthCertificate(
            accepted=True,
            reason="native-boundary-path-growth-verified",
            completion_fraction=native_completion,
            onset_delay_samples=native_onset_delay,
        )
        owner.native_front_refinement = None
        owner.native_front_paths_xy = None
        owner.selected_pose_model = (
            "native-boundary-deformable"
            if selected.deformation is not None
            else "native-boundary-rigid"
        )
        owner.deformation = selected.deformation
        selections.append(
            {
                "owner_id": owner.track_id,
                "candidate": selected.source,
                "final_length_source_px": (
                    selected.result.final_length_px / analysis_scale
                ),
                "native_median_coverage": (
                    selected.native_quality_certificate.median_coverage
                ),
                "native_root_onset_sample": selected.native_root_onset_sample,
                "native_root_onset_delay_samples": (
                    selected.native_root_onset_delay_samples
                ),
                "boundary_contact": selected.boundary_contact,
                "deformation_coverage_gain": (
                    None
                    if selected.deformation is None
                    else selected.native_quality_certificate.median_coverage
                    - rigid_candidate.native_quality_certificate.median_coverage
                ),
            }
        )
    return {
        "rescued_owner_ids": [int(row["owner_id"]) for row in selections],
        "selections": selections,
        "rejected_short_consensus": rejected_short_consensus,
    }


def _audit_native_root_emergence(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    maximum_growth_analysis_px_per_sample: float,
    root_extent_source_px: float = 20.0,
    minimum_growth_source_px: float = 3.0,
    maximum_onset_difference_samples: int = 12,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Anchor germination timing to native growth near the owner root."""

    candidates = [
        owner
        for owner in owners
        if owner.result is not None
        and owner.native_path_quality_certificate is not None
        and owner.native_path_quality_certificate.accepted
        and _measurement_eligible(owner)
        and owner.growth_certificate is not None
        and owner.growth_certificate.accepted
        and owner.direction_certificate is not None
        and owner.direction_certificate.accepted
        and not owner.ownership_conflict_ids
    ]
    point_count_by_id = {}
    support_by_id = {}
    thresholds_by_id = {}
    for owner in candidates:
        source_arc = owner.arclength_px / analysis_scale
        point_count = int(
            np.searchsorted(source_arc, root_extent_source_px, side="right")
        )
        point_count = min(
            max(3, point_count),
            owner.result.final_point_index + 1,
        )
        point_count_by_id[owner.track_id] = point_count
        support_by_id[owner.track_id] = np.zeros(
            (len(source_frames), point_count),
            dtype=np.float32,
        )
        thresholds_by_id[owner.track_id] = np.zeros(
            len(source_frames),
            dtype=np.float32,
        )

    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample, source_frame in enumerate(source_frames):
        gray = _read_native_gray(capture, source_frame, native_frame_cache)
        for owner in candidates:
            point_count = point_count_by_id[owner.track_id]
            root_paths_xy = (
                owner.rigid_paths_xy
                if owner.selected_pose_model.startswith("native-boundary")
                and owner.rigid_paths_xy is not None
                else owner.dynamic_paths_xy
            )
            source_path_yx = _source_path(
                root_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            support, controls = sample_native_paired_support(
                gray,
                source_path_yx,
                normal_search_offsets_px=(-3.0, 0.0, 3.0),
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support_by_id[owner.track_id][sample] = support
            thresholds_by_id[owner.track_id][sample] = max(
                2.0,
                control_median + 2.0 * control_noise,
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        point_count = point_count_by_id[owner.track_id]
        source_arc = owner.arclength_px[:point_count] / analysis_scale
        growth, _, onset, last_dormant = native_connected_prefix_growth(
            support_by_id[owner.track_id],
            thresholds_by_id[owner.track_id],
            source_arc,
            warmup_samples=8,
            maximum_gap_points=2,
            minimum_strong_growth_px=minimum_growth_source_px,
            minimum_weak_growth_px=max(1.0, minimum_growth_source_px - 1.0),
        )
        causal_lower_onset = _owner_onset_sample(
            owner,
            5.0 * analysis_scale,
        )
        delay = (
            None
            if onset is None or causal_lower_onset is None
            else int(onset - causal_lower_onset)
        )
        timing_applied = onset is not None
        if onset is None:
            reason = "native-root-emergence-not-detected"
        elif causal_lower_onset is None:
            reason = "native-root-onset-corrected"
        elif abs(delay) > maximum_onset_difference_samples:
            reason = "native-root-onset-corrected"
        else:
            reason = "native-root-onset-corroborated"
        if onset is not None:
            _promote_native_root_timeline(
                owner,
                growth,
                onset,
                analysis_scale,
                maximum_growth_analysis_px_per_sample,
            )
        owner.native_root_onset_sample = onset
        owner.native_root_last_dormant_sample = last_dormant
        owner.native_root_onset_reason = reason
        owner.native_root_onset_delay_samples = delay
        owner.native_root_final_growth_px = float(growth[-1])
        results.append(
            {
                "owner_id": owner.track_id,
                "reason": reason,
                "native_onset_sample": onset,
                "causal_lower_onset_sample": causal_lower_onset,
                "onset_delay_samples": delay,
                "final_root_growth_source_px": float(growth[-1]),
                "root_point_count": point_count,
                "timing_applied": timing_applied,
            }
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "corroborated_owner_ids": [
            int(row["owner_id"])
            for row in results
            if row["reason"] == "native-root-onset-corroborated"
        ],
        "corrected_owner_ids": [
            int(row["owner_id"])
            for row in results
            if row["reason"] == "native-root-onset-corrected"
        ],
        "not_detected_owner_ids": [
            int(row["owner_id"])
            for row in results
            if row["reason"] == "native-root-emergence-not-detected"
        ],
        "root_extent_source_px": root_extent_source_px,
        "minimum_growth_source_px": minimum_growth_source_px,
        "maximum_onset_difference_samples": maximum_onset_difference_samples,
        "results": results,
    }


def _audit_native_tip_fronts(
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    analysis_scale: float,
    source_shape: tuple[int, int],
    dense_step_source_px: float = 1.0,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> dict:
    """Fit native material fronts for uncensored tubes without applying them."""

    candidates = []
    for owner in owners:
        if (
            owner.result is None
            or owner.native_path_quality_certificate is None
            or not owner.native_path_quality_certificate.accepted
            or not _measurement_eligible(owner)
            or owner.growth_certificate is None
            or not owner.growth_certificate.accepted
            or owner.direction_certificate is None
            or not owner.direction_certificate.accepted
            or owner.ownership_conflict_ids
        ):
            continue
        boundary_contact = _retained_path_reaches_source_boundary(
            owner,
            shifts_xy,
            analysis_scale,
            source_shape,
        )
        if (
            _causal_status(
                owner,
                True,
                _partial_tail_diagnostics(owner),
                boundary_contact=boundary_contact,
            )
            == "measured"
        ):
            candidates.append(owner)

    dense_arc_by_id = {}
    prior_by_id = {}
    support_by_id = {}
    thresholds_by_id = {}
    for owner in candidates:
        point_count = owner.result.final_point_index + 1
        source_arc = owner.arclength_px[:point_count] / analysis_scale
        dense_arc = np.arange(
            0.0,
            source_arc[-1],
            dense_step_source_px,
            dtype=np.float64,
        )
        if not len(dense_arc) or source_arc[-1] - dense_arc[-1] > 1e-6:
            dense_arc = np.append(dense_arc, source_arc[-1])
        dense_arc_by_id[owner.track_id] = dense_arc
        query = dense_arc * analysis_scale
        dense_paths = np.empty((len(source_frames), len(dense_arc), 2), dtype=np.float64)
        for sample in range(len(source_frames)):
            dense_paths[sample] = np.column_stack(
                [
                    np.interp(
                        query,
                        owner.arclength_px[:point_count],
                        owner.dynamic_paths_xy[sample, :point_count, axis],
                    )
                    for axis in range(2)
                ]
            )
        owner.native_front_paths_xy = dense_paths
        prior_by_id[owner.track_id] = np.where(
            owner.result.front_indices >= 0,
            owner.arclength_px[
                np.clip(owner.result.front_indices, 0, point_count - 1)
            ]
            / analysis_scale,
            0.0,
        )
        support_by_id[owner.track_id] = np.zeros(
            (len(source_frames), len(dense_arc)),
            dtype=np.float32,
        )
        thresholds_by_id[owner.track_id] = np.zeros(
            len(source_frames),
            dtype=np.float32,
        )

    capture = _open_fallback_capture(movie, native_frame_cache)
    for sample, source_frame in enumerate(source_frames):
        gray = _read_native_gray(capture, source_frame, native_frame_cache)
        for owner in candidates:
            source_path_yx = _source_path(
                owner.native_front_paths_xy[sample],
                shifts_xy[sample],
                analysis_scale,
            )[:, ::-1]
            support, controls = sample_native_paired_support(
                gray,
                source_path_yx,
                normal_search_offsets_px=(-2.0, 0.0, 2.0),
            )
            control_median = float(np.median(controls))
            control_noise = 1.4826 * float(
                np.median(np.abs(controls - control_median))
            )
            support_by_id[owner.track_id][sample] = support
            thresholds_by_id[owner.track_id][sample] = max(
                2.0,
                control_median + 2.0 * control_noise,
            )
    _release_fallback_capture(capture)

    results = []
    for owner in candidates:
        refinement = fit_prior_constrained_native_front(
            support_by_id[owner.track_id],
            thresholds_by_id[owner.track_id],
            dense_arc_by_id[owner.track_id],
            prior_by_id[owner.track_id],
        )
        owner.native_front_refinement = refinement
        results.append(
            {
                "owner_id": owner.track_id,
                "accepted": refinement.accepted,
                "reason": refinement.reason,
                "mean_data_gain": refinement.mean_data_gain,
                "mean_absolute_correction_px": (
                    refinement.mean_absolute_correction_px
                ),
                "corridor_edge_fraction": refinement.corridor_edge_fraction,
                "final_prior_length_px": float(prior_by_id[owner.track_id][-1]),
                "final_native_length_px": float(refinement.lengths_px[-1]),
            }
        )
    return {
        "candidate_owner_ids": [owner.track_id for owner in candidates],
        "verified_owner_ids": [
            int(row["owner_id"]) for row in results if row["accepted"]
        ],
        "dense_step_source_px": dense_step_source_px,
        "results": results,
    }


def _normalized_front_score(result: CausalGrowthFrontResult) -> float:
    """Normalize a causal-front score for comparison across reached lengths."""

    reached = max(1, result.final_point_index + 1)
    return float(result.objective_score / np.sqrt(reached))


def _partial_tail_diagnostics(owner: OwnerGeometry) -> dict[str, float | int]:
    """Distinguish a pre-existing crossover tail from a recording-end tail."""

    assert owner.profile is not None and owner.result is not None
    result = owner.result
    tail = owner.profile[:, result.final_point_index + 1 :]
    if not tail.shape[1]:
        return {
            "split_birth_sample": int(result.birth_samples[result.final_point_index]),
            "tail_preexisting_fraction": 0.0,
            "tail_terminal_fraction": 0.0,
        }
    support = tail >= 0.0
    split_birth = int(result.birth_samples[result.final_point_index])
    precontact_stop = max(0, split_birth - 12)
    precontact_start = max(0, precontact_stop - 60)
    precontact = support[precontact_start:precontact_stop]
    preexisting = (
        np.mean(precontact, axis=0) >= 0.50
        if len(precontact)
        else np.zeros(tail.shape[1], dtype=bool)
    )
    terminal = np.mean(support[-14:], axis=0) >= 0.50
    return {
        "split_birth_sample": split_birth,
        "tail_preexisting_fraction": float(np.mean(preexisting)),
        "tail_terminal_fraction": float(np.mean(terminal)),
    }


def _causal_status(
    owner: OwnerGeometry,
    accepted: bool,
    tail_diagnostics: dict[str, float | int],
    *,
    boundary_contact: bool = False,
) -> str:
    """Assign a censoring reason without overriding prior review decisions."""

    assert owner.result is not None
    if owner.body_status in {"rejected", "indeterminate"}:
        return "invalid_pollen_identity"
    if owner.forecast_stability_veto:
        # TimesFM closed-loop veto tripped (sustained fault-suspect replays
        # with negligible length gain, dense P131 precedent).  This overrides
        # even native verification: the trace flickers between phantom
        # extension and stub while coverage stays high.
        return "forecast_stability_review"
    if owner.direction_certificate is not None and not owner.direction_certificate.accepted:
        return "wrong_owner_growth_direction"
    if owner.foreign_branch_owner_id is not None:
        return "foreign_branch_censored" if accepted else "foreign_branch_capture"
    if (
        owner.native_path_quality_certificate is not None
        and not owner.native_path_quality_certificate.accepted
    ):
        return "native_geometry_unresolved"
    if owner.native_certificate is not None and owner.native_certificate.accepted:
        if owner.field_status == "boundary_censored" and boundary_contact:
            return "boundary_censored"
        if owner.field_status == "contact_censored":
            return owner.field_status
        return "native_verified_measured"
    if owner.field_status not in AUTOMATIC_STATUSES:
        return owner.field_status
    if owner.ownership_conflict_ids:
        return "ownership_conflict"
    if owner.shared_branch_owner_id is not None:
        return "shared_branch_censored" if accepted else "foreign_branch_capture"
    if not accepted:
        return "review_no_causal_growth"
    reaches_foreign_contact = bool(
        owner.selected_foreign_contact_owner_id is not None
        and owner.result.final_length_px >= 0.95 * owner.arclength_px[-1]
    )
    if reaches_foreign_contact:
        return "contact_censored"
    if owner.result.reason == "ok":
        if owner.field_status == "boundary_censored":
            return "boundary_censored" if boundary_contact else "measured"
        return owner.field_status
    if owner.field_status == "boundary_censored" and boundary_contact:
        return "boundary_censored"
    if owner.field_status == "contact_censored":
        return owner.field_status
    split_birth = int(tail_diagnostics["split_birth_sample"])
    if float(tail_diagnostics["tail_preexisting_fraction"]) >= 0.20:
        return "crossover_censored"
    if (
        split_birth >= len(owner.profile) - 28
        and float(tail_diagnostics["tail_terminal_fraction"]) >= 0.20
    ):
        return "recording_end_censored"
    return "causal_support_censored"


def _retained_path_reaches_source_boundary(
    owner: OwnerGeometry,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    source_shape: tuple[int, int],
) -> bool:
    """Report whether the retained final centerline reaches the source field edge."""

    assert owner.result is not None
    front = int(owner.result.front_indices[-1])
    if front < 1:
        return False
    source_path = _source_path(
        owner.dynamic_paths_xy[-1, : front + 1],
        shifts_xy[-1],
        analysis_scale,
    )
    try:
        _, boundary_contact = clip_path_to_image_bounds(
            source_path[:, ::-1],
            source_shape,
        )
    except ValueError:
        return False
    return boundary_contact


def _active_curve_yx(owner: OwnerGeometry, sample: int) -> np.ndarray:
    """Return one fitted owner prefix in row-column coordinates."""

    assert owner.result is not None
    front = int(owner.result.front_indices[sample])
    if front < 1:
        return np.empty((0, 2), dtype=np.float64)
    return owner.dynamic_paths_xy[sample, : front + 1, ::-1]


def _refit_truncated_owner(
    owner: OwnerGeometry,
    *,
    stop_point: int,
    owner_center_xy: np.ndarray,
    owner_radius_px: float,
    warmup_samples: int,
    max_growth_px_per_sample: float,
    direction_kwargs: dict[str, float],
) -> None:
    """Refit one owner after retaining only its independent path prefix."""

    stop = min(len(owner.arclength_px), max(2, int(stop_point)))
    owner.arclength_px = owner.arclength_px[:stop]
    owner.dynamic_paths_xy = owner.dynamic_paths_xy[:, :stop]
    owner.profile = owner.profile[:, :stop]
    if owner.rigid_paths_xy is not None:
        owner.rigid_paths_xy = owner.rigid_paths_xy[:, :stop]
    if owner.rigid_profile is not None:
        owner.rigid_profile = owner.rigid_profile[:, :stop]
    owner.result = causal_changepoint_front(
        owner.profile,
        owner.arclength_px,
        warmup_samples=warmup_samples,
        max_step_px=max_growth_px_per_sample,
        change_window_samples=7,
        persistence_window_samples=21,
        minimum_post_samples=1,
    )
    if owner.selected_pose_model in {
        "pollen-translated",
        "native-boundary-rigid",
    }:
        owner.rigid_result = owner.result
    else:
        owner.deformed_result = owner.result
    _refresh_owner_topology(owner, owner_center_xy, owner_radius_px)
    reverse_arclength = owner.arclength_px[-1] - owner.arclength_px[::-1]
    owner.reverse_result = causal_changepoint_front(
        owner.profile[:, ::-1],
        reverse_arclength,
        warmup_samples=warmup_samples,
        max_step_px=max_growth_px_per_sample,
        change_window_samples=7,
        persistence_window_samples=21,
        minimum_post_samples=1,
    )
    owner.direction_certificate = certify_growth_direction(
        owner.result,
        owner.reverse_result,
        total_length_px=float(owner.arclength_px[-1]),
        **direction_kwargs,
    )


def _owner_onset_sample(
    owner: OwnerGeometry,
    germination_length_px: float,
) -> int | None:
    """Return when one fitted material front first reaches reporting length."""

    assert owner.result is not None
    lengths = np.where(
        owner.result.front_indices >= 0,
        owner.arclength_px[
            np.clip(
                owner.result.front_indices,
                0,
                len(owner.arclength_px) - 1,
            )
        ],
        0.0,
    )
    return _first_threshold_sample(lengths, germination_length_px)


def _arbitrate_shared_branches(
    owners: list[OwnerGeometry],
    centers_by_id: dict[int, np.ndarray],
    *,
    owner_radius_px: float,
    germination_length_px: float,
    warmup_samples: int,
    max_growth_px_per_sample: float,
    direction_kwargs: dict[str, float],
) -> dict:
    """Assign a sustained shared tail to its earlier causal owner."""

    eligible = [
        owner
        for owner in owners
        if _measurement_eligible(owner)
        and owner.growth_certificate is not None
        and owner.growth_certificate.accepted
        and owner.direction_certificate is not None
        and owner.direction_certificate.accepted
        and owner.result is not None
    ]
    evidence: list[FieldOwnerEvidence] = []
    onset_by_id: dict[int, int] = {}
    for owner in eligible:
        onset = _owner_onset_sample(owner, germination_length_px)
        if onset is None:
            continue
        onset_by_id[owner.track_id] = onset
        evidence.append(
            FieldOwnerEvidence(
                track_id=owner.track_id,
                status="measured",
                event_certified=True,
                onset_sample=onset,
                curves_by_sample={
                    sample: curve
                    for sample in range(len(owner.result.front_indices))
                    if len(curve := _active_curve_yx(owner, sample)) >= 2
                },
            )
        )
    distance_px = max(1.0, owner_radius_px / 3.0)
    claims = sustained_duplicate_claims(
        evidence,
        distance_px=distance_px,
        root_exclusion_px=owner_radius_px,
        minimum_directional_overlap=0.30,
        minimum_sustained_fraction=0.60,
        trailing_samples=40,
        minimum_common_samples=12,
    )
    owner_by_id = {owner.track_id: owner for owner in owners}
    resolutions = []
    for claim in sorted(
        claims,
        key=lambda item: -abs(
            onset_by_id[int(item["first_owner"])]
            - onset_by_id[int(item["second_owner"])]
        ),
    ):
        first = int(claim["first_owner"])
        second = int(claim["second_owner"])
        first_onset, second_onset = onset_by_id[first], onset_by_id[second]
        if abs(first_onset - second_onset) < 12:
            for owner_id, other_id in ((first, second), (second, first)):
                owner = owner_by_id[owner_id]
                if other_id not in owner.ownership_conflict_ids:
                    owner.ownership_conflict_ids.append(other_id)
            resolutions.append(
                {
                    "owner_ids": [first, second],
                    "resolution": "ambiguous-simultaneous-shared-branch",
                    "winner_owner_id": None,
                }
            )
            continue
        winner_id, loser_id = (
            (first, second) if first_onset < second_onset else (second, first)
        )
        winner = owner_by_id[winner_id]
        loser = owner_by_id[loser_id]
        winner_curve = _active_curve_yx(winner, len(winner.result.front_indices) - 1)
        loser_curve = _active_curve_yx(loser, len(loser.result.front_indices) - 1)
        overlap = first_sustained_overlap_index(
            loser_curve,
            winner_curve,
            distance_px=distance_px,
        )
        if overlap is None:
            continue
        _refit_truncated_owner(
            loser,
            stop_point=overlap,
            owner_center_xy=centers_by_id[loser_id],
            owner_radius_px=owner_radius_px,
            warmup_samples=warmup_samples,
            max_growth_px_per_sample=max_growth_px_per_sample,
            direction_kwargs=direction_kwargs,
        )
        loser.shared_branch_owner_id = winner_id
        resolutions.append(
            {
                "owner_ids": [first, second],
                "resolution": "earlier-causal-owner-with-safe-loser-prefix",
                "winner_owner_id": winner_id,
                "censored_owner_id": loser_id,
                "censored_before_point": int(overlap),
            }
        )
    return {"claims": claims, "resolutions": resolutions}


def _arbitrate_foreign_branch_captures(
    owners: list[OwnerGeometry],
    centers_by_id: dict[int, np.ndarray],
    *,
    owner_radius_px: float,
    germination_length_px: float,
    distance_radii: float,
    minimum_shared_radii: float,
    minimum_onset_lead_samples: int,
    warmup_samples: int,
    max_growth_px_per_sample: float,
    direction_kwargs: dict[str, float],
) -> dict:
    """Attribute a late aligned branch to an earlier causal pollen owner."""

    onset_by_id = {
        owner.track_id: onset
        for owner in owners
        if owner.result is not None
        and (onset := _owner_onset_sample(owner, germination_length_px)) is not None
    }
    references = {
        owner.track_id: owner
        for owner in owners
        if owner.track_id in onset_by_id
        and _measurement_eligible(owner)
        and owner.growth_certificate is not None
        and owner.growth_certificate.accepted
        and owner.direction_certificate is not None
        and owner.direction_certificate.accepted
        and not owner.ownership_conflict_ids
    }
    distance_px = distance_radii * owner_radius_px
    minimum_shared_length_px = minimum_shared_radii * owner_radius_px
    claims = []
    for candidate in sorted(
        owners,
        key=lambda item: onset_by_id.get(item.track_id, len(owners) * 10_000),
    ):
        candidate_onset = onset_by_id.get(candidate.track_id)
        if (
            candidate_onset is None
            or candidate.result is None
            or candidate.shared_branch_owner_id is not None
            or candidate.ownership_conflict_ids
        ):
            continue
        candidate_curve = _active_curve_yx(
            candidate,
            len(candidate.result.front_indices) - 1,
        )
        if len(candidate_curve) < 2:
            continue
        matches = []
        for reference_id, reference in references.items():
            if reference_id == candidate.track_id or reference.foreign_branch_owner_id:
                continue
            reference_onset = onset_by_id[reference_id]
            if reference_onset > candidate_onset - minimum_onset_lead_samples:
                continue
            reference_curve = _active_curve_yx(
                reference,
                len(reference.result.front_indices) - 1,
            )
            if len(reference_curve) < 2:
                continue
            capture_point = sustained_branch_capture_index(
                candidate_curve,
                reference_curve,
                distance_px=distance_px,
                minimum_shared_length_px=minimum_shared_length_px,
            )
            if capture_point is not None:
                matches.append((capture_point, reference_onset, reference_id))
        if not matches:
            continue
        capture_point, reference_onset, reference_id = min(matches)
        was_measurement_eligible = _measurement_eligible(candidate)
        candidate.foreign_branch_owner_id = reference_id
        candidate.foreign_branch_capture_point = capture_point
        retained_stop = max(2, capture_point)
        retained_length = float(candidate.arclength_px[retained_stop - 1])
        if was_measurement_eligible:
            _refit_truncated_owner(
                candidate,
                stop_point=capture_point,
                owner_center_xy=centers_by_id[candidate.track_id],
                owner_radius_px=owner_radius_px,
                warmup_samples=warmup_samples,
                max_growth_px_per_sample=max_growth_px_per_sample,
                direction_kwargs=direction_kwargs,
            )
        claims.append(
            {
                "evidence": "earlier-aligned-branch",
                "candidate_owner_id": candidate.track_id,
                "foreign_branch_owner_id": reference_id,
                "candidate_onset_sample": candidate_onset,
                "foreign_owner_onset_sample": reference_onset,
                "capture_point": capture_point,
                "retained_prefix_length_analysis_px": retained_length,
                "candidate_was_measurement_eligible": was_measurement_eligible,
            }
        )
    for candidate in owners:
        contact_owner = candidate.selected_foreign_contact_owner_id
        if (
            contact_owner is None
            or candidate.foreign_branch_owner_id is not None
            or candidate.shared_branch_owner_id is not None
            or candidate.result is None
            or candidate.growth_certificate is None
            or candidate.growth_certificate.accepted
            or candidate.result.final_length_px >= germination_length_px
        ):
            continue
        candidate.foreign_branch_owner_id = contact_owner
        candidate.foreign_branch_capture_point = len(candidate.arclength_px) - 1
        claims.append(
            {
                "evidence": "foreign-contact-without-independent-prefix",
                "candidate_owner_id": candidate.track_id,
                "foreign_branch_owner_id": contact_owner,
                "candidate_onset_sample": None,
                "foreign_owner_onset_sample": onset_by_id.get(contact_owner),
                "capture_point": candidate.foreign_branch_capture_point,
                "retained_prefix_length_analysis_px": float(
                    candidate.result.final_length_px
                ),
                "candidate_was_measurement_eligible": _measurement_eligible(candidate),
            }
        )
    return {
        "claims": claims,
        "distance_analysis_px": distance_px,
        "minimum_shared_length_analysis_px": minimum_shared_length_px,
        "minimum_onset_lead_samples": minimum_onset_lead_samples,
    }


def _audit_retained_duplicate_claims(
    owners: list[OwnerGeometry],
    *,
    owner_radius_px: float,
    germination_length_px: float,
) -> dict:
    """Rescan centered retained histories for duplicate material claims."""

    evidence = []
    for owner in owners:
        if (
            owner.result is None
            or not _measurement_eligible(owner)
            or owner.growth_certificate is None
            or not owner.growth_certificate.accepted
            or owner.direction_certificate is None
            or not owner.direction_certificate.accepted
            or owner.ownership_conflict_ids
        ):
            continue
        onset = _owner_onset_sample(owner, germination_length_px)
        if onset is None:
            continue
        evidence.append(
            FieldOwnerEvidence(
                track_id=owner.track_id,
                status="measured",
                event_certified=True,
                onset_sample=onset,
                curves_by_sample={
                    sample: curve
                    for sample in range(len(owner.result.front_indices))
                    if len(curve := _active_curve_yx(owner, sample)) >= 2
                },
            )
        )
    claims = sustained_duplicate_claims(
        evidence,
        distance_px=max(1.0, owner_radius_px / 3.0),
        root_exclusion_px=owner_radius_px,
        minimum_directional_overlap=0.30,
        minimum_sustained_fraction=0.60,
        trailing_samples=40,
        minimum_common_samples=12,
    )
    return {
        "retained_owner_ids": [item.track_id for item in evidence],
        "claims": claims,
    }


def _audit_export_consistency(
    summaries: list[dict],
    measurements: list[dict],
    centerlines: list[dict],
    *,
    length_tolerance_px: float = 0.01,
) -> dict:
    """Reject internally inconsistent science-facing export tables."""

    accepted_owner_ids = {
        int(row["pollen_id"])
        for row in summaries
        if int(row["causal_measurement_accepted"]) == 1
    }
    invalid_topology_ids = sorted(
        int(row["pollen_id"])
        for row in summaries
        if int(row["pollen_id"]) in accepted_owner_ids
        and row["owner_path_topology_certificate"]
        != "owner-path-topology-certified"
    )

    owner_lengths: dict[int, list[float]] = {
        owner_id: [] for owner_id in accepted_owner_ids
    }
    accepted_lengths_by_key: dict[tuple[int, int], float] = {}
    for row in measurements:
        owner_id = int(row["pollen_id"])
        if owner_id not in accepted_owner_ids:
            continue
        length = float(row["tube_length_px"])
        if not np.isfinite(length):
            raise RuntimeError(
                f"accepted owner P{owner_id} has a non-finite tube length"
            )
        owner_lengths[owner_id].append(length)
        if int(row["accepted"]) == 1:
            visible_length = float(row.get("visible_centerline_length_px", length))
            accepted_lengths_by_key[(owner_id, int(row["sample_index"]))] = (
                visible_length
            )

    backward_step_ids = sorted(
        owner_id
        for owner_id, lengths in owner_lengths.items()
        if len(lengths) > 1
        and np.any(np.diff(np.asarray(lengths)) < -length_tolerance_px)
    )

    maximum_arc_by_key: dict[tuple[int, int], float] = {}
    for row in centerlines:
        key = (int(row["pollen_id"]), int(row["sample_index"]))
        maximum_arc_by_key[key] = max(
            maximum_arc_by_key.get(key, 0.0),
            float(row["arc_length_px"]),
        )
    missing_centerline_keys = sorted(
        key for key in accepted_lengths_by_key if key not in maximum_arc_by_key
    )
    orphan_centerline_keys = sorted(
        key for key in maximum_arc_by_key if key not in accepted_lengths_by_key
    )
    mismatched_centerline_keys = sorted(
        key
        for key, length in accepted_lengths_by_key.items()
        if key in maximum_arc_by_key
        and abs(maximum_arc_by_key[key] - length) > length_tolerance_px
    )

    problems = {
        "invalid_topology_owner_ids": invalid_topology_ids,
        "backward_step_owner_ids": backward_step_ids,
        "missing_centerline_keys": missing_centerline_keys,
        "orphan_centerline_keys": orphan_centerline_keys,
        "mismatched_centerline_keys": mismatched_centerline_keys,
    }
    if any(problems.values()):
        preview = {
            key: values[:10]
            for key, values in problems.items()
            if values
        }
        raise RuntimeError(f"science-facing export consistency failed: {preview}")

    return {
        "accepted_owner_count": len(accepted_owner_ids),
        "accepted_measurement_count": len(accepted_lengths_by_key),
        "accepted_centerline_point_count": len(centerlines),
        "length_tolerance_px": length_tolerance_px,
        "invalid_topology_owner_ids": [],
        "backward_step_owner_ids": [],
        "missing_centerline_count": 0,
        "orphan_centerline_count": 0,
        "mismatched_centerline_count": 0,
    }


def _resolve_final_duplicate_claims(
    owners: list[OwnerGeometry],
    duplicate_audit: dict,
) -> list[int]:
    """Prefer a verified pollen body and fail closed on equivalent evidence."""

    owner_by_id = {owner.track_id: owner for owner in owners}
    vetoed = set()
    for claim in duplicate_audit["claims"]:
        first = int(claim["first_owner"])
        second = int(claim["second_owner"])
        first_owner = owner_by_id[first]
        second_owner = owner_by_id[second]
        first_body_rank = BODY_STATUS_RANK.get(first_owner.body_status)
        second_body_rank = BODY_STATUS_RANK.get(second_owner.body_status)
        legacy_pair = first_body_rank is None and second_body_rank is None
        if legacy_pair:
            first_rank = IDENTITY_TIER_RANK.get(
                first_owner.identity_tier,
                IDENTITY_TIER_RANK["unknown"],
            )
            second_rank = IDENTITY_TIER_RANK.get(
                second_owner.identity_tier,
                IDENTITY_TIER_RANK["unknown"],
            )
        else:
            first_rank = -1 if first_body_rank is None else first_body_rank
            second_rank = -1 if second_body_rank is None else second_body_rank
        if first_rank == second_rank:
            losers = ((first, second), (second, first))
            claim["ownership_priority_resolution"] = (
                "legacy-equal-tier-withhold-both"
                if legacy_pair
                else "equal-body-evidence-withhold-both"
            )
        else:
            winner, loser = (
                (first, second) if first_rank > second_rank else (second, first)
            )
            losers = ((loser, winner),)
            claim["ownership_priority_resolution"] = (
                "legacy-stronger-identity-retained"
                if legacy_pair
                else "verified-body-priority"
            )
            claim["ownership_priority_winner_owner_id"] = winner
            claim["ownership_priority_loser_owner_id"] = loser
        for owner_id, other_id in losers:
            owner = owner_by_id[owner_id]
            if other_id not in owner.ownership_conflict_ids:
                owner.ownership_conflict_ids.append(other_id)
            vetoed.add(owner_id)
    return sorted(vetoed)


def _export_tables(
    output: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    fps: float,
    analysis_scale: float,
    germination_length_px: float,
    source_shape: tuple[int, int],
) -> tuple[dict, dict]:
    """Write summary, time-series, and complete centerline tables."""

    summaries: list[dict] = []
    measurements: list[dict] = []
    centerlines: list[dict] = []
    diagnostic_centerlines: list[dict] = []
    for owner in owners:
        assert owner.result is not None
        result = owner.result
        front = result.front_indices
        lengths = np.where(
            front >= 0,
            owner.arclength_px[np.clip(front, 0, len(owner.arclength_px) - 1)],
            0.0,
        )
        source_lengths = lengths / analysis_scale
        original = owner.original_measurements["tube_length_px"].to_numpy()
        onset = _first_threshold_sample(source_lengths, germination_length_px)
        onset_lower = _first_threshold_sample(
            source_lengths,
            0.5 * germination_length_px,
        )
        onset_established = _first_threshold_sample(
            source_lengths,
            1.5 * germination_length_px,
        )
        automatic = _measurement_eligible(owner)
        assert owner.growth_certificate is not None
        accepted = bool(
            automatic
            and owner.growth_certificate.accepted
            and owner.direction_certificate is not None
            and owner.direction_certificate.accepted
            and onset is not None
            and not owner.ownership_conflict_ids
        )
        tail_diagnostics = _partial_tail_diagnostics(owner)
        final_boundary_contact = _retained_path_reaches_source_boundary(
            owner,
            shifts_xy,
            analysis_scale,
            source_shape,
        )
        causal_status = _causal_status(
            owner,
            accepted,
            tail_diagnostics,
            boundary_contact=final_boundary_contact,
        )
        terminal_window = min(12, len(source_lengths) - 1)
        growth_active_at_end = bool(
            terminal_window > 0
            and source_lengths[-1] - source_lengths[-1 - terminal_window]
            >= 0.5 * germination_length_px
        )
        rigid_final = min(
            owner.arclength_px[-1],
            owner.rigid_result.final_length_px if owner.rigid_result else 0.0,
        )
        deformed_final = min(
            owner.arclength_px[-1],
            owner.deformed_result.final_length_px if owner.deformed_result else 0.0,
        )
        reverse = owner.reverse_result
        summaries.append(
            {
                "pollen_id": owner.track_id,
                "owner_identity_tier": owner.identity_tier,
                "identity_observation_count": owner.identity_observation_count,
                "semantic_observation_count": owner.semantic_observation_count,
                "geometric_assignment_fraction": (
                    owner.geometric_assignment_fraction
                ),
                "median_owner_template_score": owner.median_owner_template_score,
                "owner_body_status": owner.body_status,
                "owner_identity_certificate_basis": (
                    owner.identity_certificate_basis
                ),
                "owner_pregrowth_semantic_consensus": int(
                    owner.pregrowth_semantic_consensus
                ),
                "forecast_fault_samples": owner.forecast_fault_samples,
                "forecast_replay_samples": owner.forecast_replay_samples,
                "forecast_stability_veto": bool(owner.forecast_stability_veto),
                "forecast_tip_hold_samples": owner.forecast_tip_hold_samples,
                "forecast_tip_hold_max_run": owner.forecast_tip_hold_max_run,
                "forecast_tip_series_samples": owner.forecast_tip_series_samples,
                "sticky_tip_corrected_samples": owner.sticky_tip_corrected_samples,
                "sticky_tip_max_move_px": owner.sticky_tip_max_move_px,
                "sticky_tip_mean_move_px": owner.sticky_tip_mean_move_px,
                "sticky_tip_series_samples": owner.sticky_tip_series_samples,
                "body_median_radial_contrast": (
                    owner.body_median_radial_contrast
                ),
                "body_median_angular_boundary_fraction": (
                    owner.body_median_angular_boundary_fraction
                ),
                "body_median_opposite_boundary_fraction": (
                    owner.body_median_opposite_boundary_fraction
                ),
                "body_median_valid_angular_fraction": (
                    owner.body_median_valid_angular_fraction
                ),
                "body_valid_sample_count": owner.body_valid_sample_count,
                "previous_field_status": owner.field_status,
                "causal_status": causal_status,
                "causal_measurement_accepted": int(accepted),
                "timeline_source": owner.timeline_source,
                "growth_certificate": owner.growth_certificate.reason,
                "owner_path_topology_certificate": (
                    ""
                    if owner.topology_certificate is None
                    else owner.topology_certificate.reason
                ),
                "retained_minimum_owner_distance_px": (
                    np.nan
                    if owner.topology_certificate is None
                    else round(
                        owner.topology_certificate.minimum_distance_px
                        / analysis_scale,
                        4,
                    )
                ),
                "retained_halo_exit_arclength_px": (
                    np.nan
                    if owner.topology_certificate is None
                    or owner.topology_certificate.first_halo_exit_arclength_px
                    is None
                    else round(
                        owner.topology_certificate.first_halo_exit_arclength_px
                        / analysis_scale,
                        4,
                    )
                ),
                "growth_direction_certificate": (
                    ""
                    if owner.direction_certificate is None
                    else owner.direction_certificate.reason
                ),
                "native_path_growth_certificate": (
                    ""
                    if owner.native_certificate is None
                    else owner.native_certificate.reason
                ),
                "native_path_completion_fraction": (
                    np.nan
                    if owner.native_certificate is None
                    else round(owner.native_certificate.completion_fraction, 4)
                ),
                "native_path_onset_delay_samples": (
                    np.nan
                    if owner.native_certificate is None
                    or owner.native_certificate.onset_delay_samples is None
                    else owner.native_certificate.onset_delay_samples
                ),
                "native_path_final_length_px": (
                    np.nan
                    if owner.native_prefix_lengths_px is None
                    else round(float(owner.native_prefix_lengths_px[-1]), 4)
                ),
                "native_path_onset_source_frame": (
                    np.nan
                    if owner.native_onset_sample is None
                    else int(source_frames[owner.native_onset_sample])
                ),
                "native_centerline_refinement": (
                    ""
                    if owner.native_centerline_refinement is None
                    else owner.native_centerline_refinement.reason
                ),
                "native_centerline_gain": (
                    np.nan
                    if owner.native_centerline_refinement is None
                    else round(
                        owner.native_centerline_refinement.normalized_median_gain,
                        4,
                    )
                ),
                "native_centerline_offset_p90_px": (
                    np.nan
                    if owner.native_centerline_refinement is None
                    else round(
                        float(
                            np.percentile(
                                np.abs(owner.native_centerline_refinement.offsets_px),
                                90.0,
                            )
                        ),
                        4,
                    )
                ),
                "native_centerline_observation_count": len(
                    owner.native_refinement_observation_samples
                ),
                "native_path_quality_certificate": (
                    ""
                    if owner.native_path_quality_certificate is None
                    else owner.native_path_quality_certificate.reason
                ),
                "native_path_quality_coverage": (
                    np.nan
                    if owner.native_path_quality_certificate is None
                    else round(
                        owner.native_path_quality_certificate.median_coverage,
                        4,
                    )
                ),
                "native_path_quality_margin": (
                    np.nan
                    if owner.native_path_quality_certificate is None
                    else round(
                        owner.native_path_quality_certificate.median_support_margin,
                        4,
                    )
                ),
                "native_path_quality_observation_count": (
                    0
                    if owner.native_path_quality_certificate is None
                    else owner.native_path_quality_certificate.observation_count
                ),
                "native_root_onset_certificate": owner.native_root_onset_reason,
                "native_root_onset_source_frame": (
                    np.nan
                    if owner.native_root_onset_sample is None
                    else int(source_frames[owner.native_root_onset_sample])
                ),
                "native_root_onset_time_minutes": (
                    np.nan
                    if owner.native_root_onset_sample is None
                    else round(
                        source_frames[owner.native_root_onset_sample] / fps / 60.0,
                        4,
                    )
                ),
                "native_root_onset_delay_samples": (
                    np.nan
                    if owner.native_root_onset_delay_samples is None
                    else owner.native_root_onset_delay_samples
                ),
                "native_root_final_growth_px": round(
                    owner.native_root_final_growth_px,
                    4,
                ),
                "germination_timing_verified": int(
                    owner.native_root_onset_reason
                    in {
                        "native-root-onset-corroborated",
                        "native-root-onset-corrected",
                    }
                ),
                "verified_germination_source_frame": (
                    np.nan
                    if owner.native_root_onset_reason
                    not in {
                        "native-root-onset-corroborated",
                        "native-root-onset-corrected",
                    }
                    or owner.native_root_onset_sample is None
                    else int(source_frames[owner.native_root_onset_sample])
                ),
                "verified_germination_time_minutes": (
                    np.nan
                    if owner.native_root_onset_reason
                    not in {
                        "native-root-onset-corroborated",
                        "native-root-onset-corrected",
                    }
                    or owner.native_root_onset_sample is None
                    else round(
                        source_frames[owner.native_root_onset_sample] / fps / 60.0,
                        4,
                    )
                ),
                "native_front_refinement": (
                    ""
                    if owner.native_front_refinement is None
                    else owner.native_front_refinement.reason
                ),
                "native_front_mean_data_gain": (
                    np.nan
                    if owner.native_front_refinement is None
                    else round(owner.native_front_refinement.mean_data_gain, 4)
                ),
                "native_front_mean_absolute_correction_px": (
                    np.nan
                    if owner.native_front_refinement is None
                    else round(
                        owner.native_front_refinement.mean_absolute_correction_px,
                        4,
                    )
                ),
                "first_causal_source_frame": (
                    "" if not accepted or onset is None else int(source_frames[onset])
                ),
                "first_causal_time_minutes": (
                    ""
                    if not accepted or onset is None
                    else round(source_frames[onset] / fps / 60.0, 4)
                ),
                "optical_onset_lower_time_minutes": (
                    ""
                    if not accepted or onset_lower is None
                    else round(source_frames[onset_lower] / fps / 60.0, 4)
                ),
                "established_growth_time_minutes": (
                    ""
                    if not accepted or onset_established is None
                    else round(source_frames[onset_established] / fps / 60.0, 4)
                ),
                "previous_final_length_px": round(float(original[-1]), 4),
                "causal_final_length_px": (
                    round(float(source_lengths[-1]), 4) if accepted else np.nan
                ),
                "diagnostic_candidate_final_length_px": round(
                    float(source_lengths[-1]), 4
                ),
                "final_length_is_lower_bound": int(
                    accepted
                    and causal_status == "boundary_censored"
                    and final_boundary_contact
                ),
                "retained_path_fraction": (
                    round(float(source_lengths[-1] / max(original[-1], 1e-9)), 4)
                    if accepted and original[-1] > 0.0
                    else np.nan
                ),
                "direct_birth_support_fraction": round(result.direct_support_fraction, 4),
                "eventual_birth_support_fraction": round(result.eventual_support_fraction, 4),
                "front_reason": result.reason,
                "growth_active_at_recording_end": int(growth_active_at_end),
                **{
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in tail_diagnostics.items()
                },
                "objective_score": round(result.objective_score, 4),
                "owner_outward_normalized_score": round(
                    _normalized_front_score(result), 4
                ),
                "distal_inward_normalized_score": (
                    np.nan
                    if reverse is None
                    else round(_normalized_front_score(reverse), 4)
                ),
                "distal_inward_final_length_px": (
                    np.nan
                    if reverse is None
                    else round(reverse.final_length_px / analysis_scale, 4)
                ),
                "distal_inward_direct_support_fraction": (
                    np.nan
                    if reverse is None
                    else round(reverse.direct_support_fraction, 4)
                ),
                "distal_inward_eventual_support_fraction": (
                    np.nan
                    if reverse is None
                    else round(reverse.eventual_support_fraction, 4)
                ),
                "rigid_causal_final_length_px": round(
                    rigid_final / analysis_scale,
                    4,
                ),
                "deformed_causal_final_length_px": round(
                    deformed_final / analysis_scale,
                    4,
                ),
                "pose_model_lower_length_px": round(
                    min(
                        rigid_final,
                        deformed_final,
                    )
                    / analysis_scale,
                    4,
                ),
                "pose_model_upper_length_px": round(
                    max(
                        rigid_final,
                        deformed_final,
                    )
                    / analysis_scale,
                    4,
                ),
                "selected_pose_model": owner.selected_pose_model,
                "selected_mature_path": owner.selected_mature_path,
                "mature_path_selection_reason": owner.mature_path_selection_reason,
                "mature_path_candidate_count": (
                    len(owner.mature_candidates) + len(owner.native_boundary_candidates)
                ),
                "selected_foreign_contact_owner_id": (
                    ""
                    if owner.selected_foreign_contact_owner_id is None
                    else owner.selected_foreign_contact_owner_id
                ),
                "shared_branch_owner_id": (
                    ""
                    if owner.shared_branch_owner_id is None
                    else owner.shared_branch_owner_id
                ),
                "foreign_branch_owner_id": (
                    ""
                    if owner.foreign_branch_owner_id is None
                    else owner.foreign_branch_owner_id
                ),
                "foreign_branch_capture_point": (
                    ""
                    if owner.foreign_branch_capture_point is None
                    else owner.foreign_branch_capture_point
                ),
                "ownership_conflict_ids": ";".join(
                    str(owner_id) for owner_id in sorted(owner.ownership_conflict_ids)
                ),
                "selected_root_distance_px": round(
                    owner.selected_root_distance_px / analysis_scale,
                    4,
                ),
                "selected_radial_excursion_px": round(
                    owner.selected_radial_excursion_px / analysis_scale,
                    4,
                ),
                "deformation_p90_px": round(
                    float(np.percentile(np.abs(owner.deformation.offsets_px), 90))
                    if owner.deformation is not None
                    else 0.0,
                    4,
                ),
                "revision": REVISION,
            }
        )
        for sample, (front_index, length) in enumerate(zip(front, source_lengths)):
            tip_xy = np.array([np.nan, np.nan])
            path = np.empty((0, 2), dtype=np.float64)
            path_arc = np.empty(0, dtype=np.float64)
            boundary_contact = False
            if front_index >= 0:
                source_path = _source_path(
                    owner.dynamic_paths_xy[sample, : front_index + 1],
                    shifts_xy[sample],
                    analysis_scale,
                )
                if len(source_path) >= 2:
                    try:
                        clipped_yx, boundary_contact = clip_path_to_image_bounds(
                            source_path[:, ::-1],
                            source_shape,
                        )
                    except ValueError:
                        # An edge-owner path rooted outside the source frame
                        # cannot be clipped: retain it unclipped as diagnostic.
                        clipped_yx, boundary_contact = (
                            source_path[:, ::-1],
                            False,
                        )
                    path = clipped_yx[:, ::-1]
                else:
                    maximum_xy = np.asarray(
                        (source_shape[1] - 1.0, source_shape[0] - 1.0)
                    )
                    path = np.clip(source_path, 0.0, maximum_xy)
                path_arc = np.concatenate(
                    ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                )
                if accepted:
                    tip_xy = path[-1]
            visible_length = float(path_arc[-1]) if len(path_arc) else 0.0
            measurements.append(
                {
                    "pollen_id": owner.track_id,
                    "owner_identity_tier": owner.identity_tier,
                    "owner_body_status": owner.body_status,
                    "owner_identity_certificate_basis": (
                        owner.identity_certificate_basis
                    ),
                    "sample_index": sample,
                    "source_frame": int(source_frames[sample]),
                    "time_minutes": round(source_frames[sample] / fps / 60.0, 4),
                    "tube_length_px": (
                        round(float(length), 4) if accepted else np.nan
                    ),
                    "visible_centerline_length_px": (
                        round(visible_length, 4) if accepted else np.nan
                    ),
                    "length_is_lower_bound": int(
                        accepted
                        and causal_status == "boundary_censored"
                        and boundary_contact
                    ),
                    "diagnostic_candidate_length_px": round(float(length), 4),
                    "previous_tube_length_px": round(float(original[sample]), 4),
                    "tip_x_px": round(float(tip_xy[0]), 4),
                    "tip_y_px": round(float(tip_xy[1]), 4),
                    "front_point_index": int(front_index),
                    "accepted": int(accepted and onset is not None and sample >= onset),
                    "causal_status": causal_status,
                }
            )
            if front_index < 0:
                continue
            for point, (xy, distance) in enumerate(zip(path, path_arc)):
                row = {
                    "pollen_id": owner.track_id,
                    "sample_index": sample,
                    "source_frame": int(source_frames[sample]),
                    "point_index": point,
                    "source_x_px": round(float(xy[0]), 4),
                    "source_y_px": round(float(xy[1]), 4),
                    "arc_length_px": round(float(distance), 4),
                    "boundary_contact": int(boundary_contact and point == len(path) - 1),
                    "causal_status": causal_status,
                }
                diagnostic_centerlines.append(row)
                if accepted and onset is not None and sample >= onset:
                    centerlines.append(row.copy())
    export_consistency = _audit_export_consistency(
        summaries,
        measurements,
        centerlines,
    )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(measurements).to_csv(output / "measurements.csv", index=False)
    pd.DataFrame(centerlines).to_csv(output / "centerlines.csv", index=False)
    pd.DataFrame(diagnostic_centerlines).to_csv(
        output / "diagnostic_centerlines.csv",
        index=False,
    )
    counts = {
        "owner_count": len(summaries),
        "accepted_owner_count": int(sum(row["causal_measurement_accepted"] for row in summaries)),
        "identity_tier_counts": {
            tier: sum(row["owner_identity_tier"] == tier for row in summaries)
            for tier in sorted({row["owner_identity_tier"] for row in summaries})
        },
        "accepted_identity_tier_counts": {
            tier: sum(
                row["owner_identity_tier"] == tier
                and row["causal_measurement_accepted"]
                for row in summaries
            )
            for tier in sorted({row["owner_identity_tier"] for row in summaries})
        },
        "body_status_counts": {
            status: sum(row["owner_body_status"] == status for row in summaries)
            for status in sorted({row["owner_body_status"] for row in summaries})
        },
        "accepted_body_status_counts": {
            status: sum(
                row["owner_body_status"] == status
                and row["causal_measurement_accepted"]
                for row in summaries
            )
            for status in sorted({row["owner_body_status"] for row in summaries})
        },
        "status_counts": {
            status: sum(row["causal_status"] == status for row in summaries)
            for status in sorted({row["causal_status"] for row in summaries})
        },
        "censored_owner_ids": {
            status: [
                row["pollen_id"]
                for row in summaries
                if row["causal_status"] == status
            ]
            for status in (
                "boundary_censored",
                "contact_censored",
                "crossover_censored",
                "recording_end_censored",
                "shared_branch_censored",
                "causal_support_censored",
                "native_geometry_unresolved",
                "review_no_causal_growth",
                "ownership_conflict",
                "wrong_owner_growth_direction",
                "foreign_branch_capture",
                "foreign_branch_censored",
                "invalid_pollen_identity",
                "edge_seed_censored",
                "forecast_stability_review",
            )
        },
    }
    return counts, export_consistency


def _render_demo(
    output: Path,
    frames: np.ndarray,
    owners: list[OwnerGeometry],
    centers_by_id: dict[int, np.ndarray],
    source_frames: np.ndarray,
    fps: float,
    pollen_radius_px: float,
    demo_fps: float,
) -> Path:
    """Render field-wide pollen ownership and causal tube tips."""

    path = output / "causal_growth_field_demo.mp4"
    height, width = frames.shape[1:]
    writer = cv.VideoWriter(
        str(path), cv.VideoWriter_fourcc(*"mp4v"), demo_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open the demo video writer")
    radius = max(3, int(round(pollen_radius_px)))
    for sample, gray in enumerate(frames):
        canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for owner in owners:
            center = np.rint(centers_by_id[owner.track_id][sample]).astype(int)
            cv.circle(canvas, tuple(center), radius, (255, 120, 30), 1, cv.LINE_AA)
            cv.putText(
                canvas,
                f"P{owner.track_id}",
                (center[0] + radius + 1, center[1] - radius),
                cv.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 120, 30),
                1,
                cv.LINE_AA,
            )
            if owner.result is None:
                continue
            front = int(owner.result.front_indices[sample])
            if (
                front < 1
                or not _measurement_eligible(owner)
                or owner.growth_certificate is None
                or not owner.growth_certificate.accepted
                or owner.direction_certificate is None
                or not owner.direction_certificate.accepted
                or owner.ownership_conflict_ids
            ):
                continue
            points = np.rint(owner.dynamic_paths_xy[sample, : front + 1]).astype(np.int32)
            color = (60, 230, 90) if owner.result.reason == "ok" else (230, 220, 50)
            cv.polylines(canvas, [points], False, color, 1, cv.LINE_AA)
            cv.circle(canvas, tuple(points[-1]), 3, color, -1, cv.LINE_AA)
        cv.rectangle(canvas, (4, 4), (193, 29), (0, 0, 0), -1)
        cv.putText(
            canvas,
            f"{source_frames[sample] / fps / 60.0:5.2f} min   "
            f"{REVISION.split('-', 1)[0]} causal tips",
            (9, 21),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    return path


def _render_native_centerline_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path:
    """Render every retained final centerline over one native source frame."""

    sample = len(source_frames) - 1
    capture = _open_fallback_capture(movie, native_frame_cache)
    gray = _read_native_gray(
        capture,
        int(source_frames[sample]),
        native_frame_cache,
    )
    _release_fallback_capture(capture)
    retained = []
    for owner in owners:
        if (
            owner.result is None
            or owner.native_path_quality_certificate is None
            or owner.growth_certificate is None
            or not owner.growth_certificate.accepted
            or owner.direction_certificate is None
            or not owner.direction_certificate.accepted
            or owner.ownership_conflict_ids
        ):
            continue
        front = int(owner.result.front_indices[sample])
        if front < 1:
            continue
        path = _source_path(
            owner.dynamic_paths_xy[sample, : front + 1],
            shifts_xy[sample],
            analysis_scale,
        )
        retained.append((owner, path))

    columns = 4
    rows = max(1, int(np.ceil(len(retained) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (owner, path) in zip(axes, retained):
        margin = 22
        x0 = max(0, int(np.floor(path[:, 0].min() - margin)))
        x1 = min(gray.shape[1], int(np.ceil(path[:, 0].max() + margin)))
        y0 = max(0, int(np.floor(path[:, 1].min() - margin)))
        y1 = min(gray.shape[0], int(np.ceil(path[:, 1].max() + margin)))
        axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
        verified = owner.native_path_quality_certificate.accepted
        axis.plot(
            path[:, 0] - x0,
            path[:, 1] - y0,
            color="#ff00dd" if verified else "#ff7a18",
            linewidth=1.4,
        )
        axis.scatter(
            [path[-1, 0] - x0],
            [path[-1, 1] - y0],
            s=18,
            color="#00ffff" if verified else "#ffe45c",
        )
        refined = bool(
            owner.native_centerline_refinement is not None
            and owner.native_centerline_refinement.accepted
        )
        placement = "centered" if refined else "retained"
        decision = "verified" if verified else "withheld"
        coverage = owner.native_path_quality_certificate.median_coverage
        axis.set_title(
            f"P{owner.track_id} {decision} {coverage:.2f} ({placement})",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[len(retained) :]:
        axis.axis("off")
    figure.tight_layout()
    path = output / "native_centerline_audit.jpg"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _render_native_path_recovery_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path | None:
    """Compare unresolved paths with broad native proposals on held-out frames."""

    candidates = [
        owner
        for owner in owners
        if owner.result is not None
        and owner.native_recovery_paths_xy is not None
        and owner.native_recovery_holdout_samples
    ]
    if not candidates:
        return None
    columns = 3
    figure, axes = plt.subplots(
        len(candidates),
        columns,
        figsize=(12, 3.3 * len(candidates)),
        squeeze=False,
    )
    capture = _open_fallback_capture(movie, native_frame_cache)
    for row, owner in enumerate(candidates):
        holdout = np.asarray(owner.native_recovery_holdout_samples)
        selected = np.unique(
            np.rint(np.quantile(holdout, (0.0, 0.5, 1.0))).astype(int)
        )
        if len(selected) < columns:
            selected = np.pad(selected, (0, columns - len(selected)), mode="edge")
        point_count = owner.result.final_point_index + 1
        for column, sample in enumerate(selected[:columns]):
            gray = _read_native_gray(
                capture,
                int(source_frames[sample]),
                native_frame_cache,
            )
            original = _source_path(
                owner.dynamic_paths_xy[sample, :point_count],
                shifts_xy[sample],
                analysis_scale,
            )
            recovered = _source_path(
                owner.native_recovery_paths_xy[sample],
                shifts_xy[sample],
                analysis_scale,
            )
            combined = np.vstack((original, recovered))
            margin = 24
            x0 = max(0, int(np.floor(combined[:, 0].min() - margin)))
            x1 = min(gray.shape[1], int(np.ceil(combined[:, 0].max() + margin)))
            y0 = max(0, int(np.floor(combined[:, 1].min() - margin)))
            y1 = min(gray.shape[0], int(np.ceil(combined[:, 1].max() + margin)))
            axis = axes[row, column]
            axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
            axis.plot(
                original[:, 0] - x0,
                original[:, 1] - y0,
                color="#ff7a18",
                linewidth=1.2,
            )
            axis.plot(
                recovered[:, 0] - x0,
                recovered[:, 1] - y0,
                color="#32ff7e",
                linewidth=1.4,
            )
            axis.scatter(
                [recovered[-1, 0] - x0],
                [recovered[-1, 1] - y0],
                color="#00ffff",
                marker="x",
                s=22,
            )
            decision = (
                "verified"
                if owner.native_recovery_reason
                == "held-out-native-path-recovery-verified"
                else "withheld"
            )
            axis.set_title(
                f"P{owner.track_id} s{sample} {decision} "
                f"{owner.native_recovery_original_coverage:.2f}->"
                f"{owner.native_recovery_coverage:.2f}",
                fontsize=8,
            )
            axis.axis("off")
    _release_fallback_capture(capture)
    figure.tight_layout()
    path = output / "native_path_recovery_audit.jpg"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _render_unresolved_candidate_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path | None:
    """Compare every mature-path source for owners rejected after selection."""

    candidates = [
        owner
        for owner in owners
        if owner.native_path_quality_certificate is not None
        and not owner.native_path_quality_certificate.accepted
        and len(owner.mature_candidates) > 1
    ]
    if not candidates:
        return None
    sample = len(source_frames) - 1
    capture = _open_fallback_capture(movie, native_frame_cache)
    gray = _read_native_gray(
        capture,
        int(source_frames[sample]),
        native_frame_cache,
    )
    _release_fallback_capture(capture)
    columns = max(len(owner.mature_candidates) for owner in candidates)
    figure, axes = plt.subplots(
        len(candidates),
        columns,
        figsize=(5.2 * columns, 3.7 * len(candidates)),
        squeeze=False,
    )
    for row, owner in enumerate(candidates):
        paths = []
        for candidate in owner.mature_candidates:
            if candidate.result is None:
                paths.append(None)
                continue
            front = int(candidate.result.front_indices[sample])
            paths.append(
                None
                if front < 1
                else _source_path(
                    candidate.dynamic_paths_xy[sample, : front + 1],
                    shifts_xy[sample],
                    analysis_scale,
                )
            )
        visible = [path for path in paths if path is not None]
        if visible:
            combined = np.vstack(visible)
            margin = 26
            x0 = max(0, int(np.floor(combined[:, 0].min() - margin)))
            x1 = min(gray.shape[1], int(np.ceil(combined[:, 0].max() + margin)))
            y0 = max(0, int(np.floor(combined[:, 1].min() - margin)))
            y1 = min(gray.shape[0], int(np.ceil(combined[:, 1].max() + margin)))
        else:
            x0, y0, x1, y1 = 0, 0, gray.shape[1], gray.shape[0]
        for column, candidate in enumerate(owner.mature_candidates):
            axis = axes[row, column]
            axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
            path = paths[column]
            if path is not None:
                axis.plot(
                    path[:, 0] - x0,
                    path[:, 1] - y0,
                    color="#00e5ff" if column else "#ff7a18",
                    linewidth=1.5,
                )
                axis.scatter(
                    [path[-1, 0] - x0],
                    [path[-1, 1] - y0],
                    color="#ffe45c",
                    s=20,
                )
            selected = candidate.source == owner.selected_mature_path
            final_length = (
                0.0
                if candidate.result is None
                else candidate.result.final_length_px / analysis_scale
            )
            axis.set_title(
                f"P{owner.track_id} {candidate.source} "
                f"{'SELECTED' if selected else 'alternate'} {final_length:.1f}px",
                fontsize=8,
            )
            axis.axis("off")
        for column in range(len(owner.mature_candidates), columns):
            axes[row, column].axis("off")
    figure.tight_layout()
    path = output / "unresolved_candidate_audit.jpg"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _render_native_boundary_candidate_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path | None:
    """Render every causal native-boundary proposal for unresolved owners."""

    retained = [owner for owner in owners if owner.native_boundary_candidates]
    if not retained:
        return None
    sample = len(source_frames) - 1
    capture = _open_fallback_capture(movie, native_frame_cache)
    gray = _read_native_gray(
        capture,
        int(source_frames[sample]),
        native_frame_cache,
    )
    _release_fallback_capture(capture)
    columns = max(len(owner.native_boundary_candidates) for owner in retained)
    figure, axes = plt.subplots(
        len(retained),
        columns,
        figsize=(3.4 * columns, 3.2 * len(retained)),
        squeeze=False,
    )
    for row, owner in enumerate(retained):
        paths = []
        for candidate in owner.native_boundary_candidates:
            assert candidate.result is not None
            front = int(candidate.result.front_indices[sample])
            paths.append(
                None
                if front < 1
                else _source_path(
                    candidate.dynamic_paths_xy[sample, : front + 1],
                    shifts_xy[sample],
                    analysis_scale,
                )
            )
        visible = [path for path in paths if path is not None]
        combined = np.vstack(visible)
        margin = 24
        x0 = max(0, int(np.floor(combined[:, 0].min() - margin)))
        x1 = min(gray.shape[1], int(np.ceil(combined[:, 0].max() + margin)))
        y0 = max(0, int(np.floor(combined[:, 1].min() - margin)))
        y1 = min(gray.shape[0], int(np.ceil(combined[:, 1].max() + margin)))
        for column, (candidate, path) in enumerate(
            zip(owner.native_boundary_candidates, paths)
        ):
            axis = axes[row, column]
            axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
            accepted = bool(
                candidate.growth_certificate is not None
                and candidate.growth_certificate.accepted
                and candidate.direction_certificate is not None
                and candidate.direction_certificate.accepted
                and candidate.native_quality_certificate is not None
                and candidate.native_quality_certificate.accepted
                and candidate.native_root_onset_reason
                in NATIVE_ONSET_ACCEPTED_REASONS
            )
            if path is not None:
                axis.plot(
                    path[:, 0] - x0,
                    path[:, 1] - y0,
                    color="#32ff7e" if accepted else "#ff5c5c",
                    linewidth=1.4,
                )
                axis.scatter(
                    [path[-1, 0] - x0],
                    [path[-1, 1] - y0],
                    color="#00e5ff",
                    marker="x",
                    s=22,
                )
            coverage = (
                0.0
                if candidate.native_quality_certificate is None
                else candidate.native_quality_certificate.median_coverage
            )
            final_length = (
                0.0
                if candidate.result is None
                else candidate.result.final_length_px / analysis_scale
            )
            axis.set_title(
                f"P{owner.track_id} {candidate.source} "
                f"{'verified' if accepted else 'withheld'} "
                f"L{final_length:.0f} C{coverage:.2f}",
                fontsize=7,
            )
            axis.axis("off")
        for column in range(len(owner.native_boundary_candidates), columns):
            axes[row, column].axis("off")
    figure.tight_layout()
    path = output / "native_boundary_candidate_audit.jpg"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_native_boundary_rescue_time_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    centers_by_id: dict[int, np.ndarray],
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path | None:
    """Show each selected native rescue at emergence, mid-growth, and maturity."""

    rescued = [
        owner
        for owner in owners
        if owner.result is not None
        and owner.selected_mature_path.startswith(
            ("native-boundary-", "native-consensus-", "native-edge-")
        )
    ]
    if not rescued:
        return None
    figure, axes = plt.subplots(
        len(rescued),
        3,
        figsize=(12, 3.2 * len(rescued)),
        squeeze=False,
    )
    capture = _open_fallback_capture(movie, native_frame_cache)
    for row, owner in enumerate(rescued):
        fronts = owner.result.front_indices
        lengths = np.where(
            fronts >= 0,
            owner.arclength_px[
                np.clip(fronts, 0, len(owner.arclength_px) - 1)
            ],
            0.0,
        )
        final_length = float(lengths[-1])
        early = np.flatnonzero(lengths >= min(5.0 * analysis_scale, final_length))
        middle = np.flatnonzero(lengths >= 0.5 * final_length)
        samples = [
            int(early[0]) if len(early) else 0,
            int(middle[0]) if len(middle) else len(source_frames) // 2,
            len(source_frames) - 1,
        ]
        for column, sample in enumerate(samples):
            gray = _read_native_gray(
                capture,
                int(source_frames[sample]),
                native_frame_cache,
            )
            front = max(1, int(fronts[sample]))
            selected_path = _source_path(
                owner.dynamic_paths_xy[sample, : front + 1],
                shifts_xy[sample],
                analysis_scale,
            )
            rigid_path = None
            if (
                owner.selected_pose_model == "native-boundary-deformable"
                and owner.rigid_paths_xy is not None
            ):
                rigid_path = _source_path(
                    owner.rigid_paths_xy[sample, : front + 1],
                    shifts_xy[sample],
                    analysis_scale,
                )
            combined = (
                selected_path
                if rigid_path is None
                else np.vstack((selected_path, rigid_path))
            )
            margin = 28
            x0 = max(0, int(np.floor(combined[:, 0].min() - margin)))
            x1 = min(gray.shape[1], int(np.ceil(combined[:, 0].max() + margin)))
            y0 = max(0, int(np.floor(combined[:, 1].min() - margin)))
            y1 = min(gray.shape[0], int(np.ceil(combined[:, 1].max() + margin)))
            axis = axes[row, column]
            axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
            owner_center_xy = (
                centers_by_id[owner.track_id][sample] + shifts_xy[sample]
            ) / analysis_scale
            axis.add_patch(
                plt.Circle(
                    (owner_center_xy[0] - x0, owner_center_xy[1] - y0),
                    15.0,
                    fill=False,
                    color="#ff3d71",
                    linewidth=1.2,
                )
            )
            if rigid_path is not None:
                axis.plot(
                    rigid_path[:, 0] - x0,
                    rigid_path[:, 1] - y0,
                    color="#ff8a3d",
                    linewidth=1.0,
                    alpha=0.75,
                )
            axis.plot(
                selected_path[:, 0] - x0,
                selected_path[:, 1] - y0,
                color="#00e5ff",
                linewidth=1.6,
            )
            axis.scatter(
                [selected_path[0, 0] - x0],
                [selected_path[0, 1] - y0],
                color="#3d7eff",
                s=20,
            )
            axis.scatter(
                [selected_path[-1, 0] - x0],
                [selected_path[-1, 1] - y0],
                color="#ffe45c",
                s=22,
            )
            axis.set_title(
                f"P{owner.track_id} s{sample} {owner.selected_mature_path} "
                f"{lengths[sample] / analysis_scale:.1f}px",
                fontsize=8,
            )
            axis.axis("off")
    _release_fallback_capture(capture)
    figure.tight_layout()
    path = output / "native_boundary_rescue_time_audit.jpg"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_native_front_audit(
    output: Path,
    movie: Path,
    owners: list[OwnerGeometry],
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
    native_frame_cache: NativeGrayFrameCache | None = None,
) -> Path | None:
    """Compare coarse and proposed native tips at three times per measured tube."""

    candidates = [
        owner
        for owner in owners
        if owner.result is not None
        and owner.native_front_refinement is not None
        and owner.native_front_paths_xy is not None
    ]
    if not candidates:
        return None
    capture = _open_fallback_capture(movie, native_frame_cache)
    columns = 3
    figure, axes = plt.subplots(
        len(candidates),
        columns,
        figsize=(12, 3.1 * len(candidates)),
        squeeze=False,
    )
    for row, owner in enumerate(candidates):
        refinable = np.flatnonzero(
            owner.native_front_refinement.lengths_px >= 5.0
        )
        selected = np.unique(
            np.rint(np.quantile(refinable, (0.1, 0.5, 1.0))).astype(int)
        )
        if len(selected) < columns:
            selected = np.pad(selected, (0, columns - len(selected)), mode="edge")
        for column, sample in enumerate(selected[:columns]):
            gray = _read_native_gray(
                capture,
                int(source_frames[sample]),
                native_frame_cache,
            )
            prior_front = int(owner.result.front_indices[sample])
            prior_path = _source_path(
                owner.dynamic_paths_xy[sample, : prior_front + 1],
                shifts_xy[sample],
                analysis_scale,
            )
            native_index = int(owner.native_front_refinement.point_indices[sample])
            native_path = _source_path(
                owner.native_front_paths_xy[sample, : native_index + 1],
                shifts_xy[sample],
                analysis_scale,
            )
            tips = np.vstack((prior_path[-1], native_path[-1]))
            center = np.mean(tips, axis=0)
            radius = 28
            x0 = max(0, int(center[0] - radius))
            x1 = min(gray.shape[1], int(center[0] + radius))
            y0 = max(0, int(center[1] - radius))
            y1 = min(gray.shape[0], int(center[1] + radius))
            axis = axes[row, column]
            axis.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=40, vmax=210)
            axis.plot(
                prior_path[-5:, 0] - x0,
                prior_path[-5:, 1] - y0,
                color="#00d9ff",
                linewidth=1.2,
            )
            axis.plot(
                native_path[-12:, 0] - x0,
                native_path[-12:, 1] - y0,
                color="#ffe45c",
                linewidth=1.2,
            )
            axis.scatter(
                [prior_path[-1, 0] - x0],
                [prior_path[-1, 1] - y0],
                color="#00d9ff",
                s=20,
            )
            axis.scatter(
                [native_path[-1, 0] - x0],
                [native_path[-1, 1] - y0],
                color="#ffe45c",
                marker="x",
                s=25,
            )
            correction = (
                owner.native_front_refinement.lengths_px[sample]
                - owner.arclength_px[prior_front] / analysis_scale
            )
            axis.set_title(
                f"P{owner.track_id} s{sample} {correction:+.1f}px",
                fontsize=8,
            )
            axis.axis("off")
    _release_fallback_capture(capture)
    figure.tight_layout()
    path = output / "native_front_refinement_audit.jpg"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _render_foreign_branch_audit(
    output: Path,
    frames: np.ndarray,
    owners: list[OwnerGeometry],
) -> Path | None:
    """Draw every foreign-branch attribution against its earlier owner path."""

    captured = [
        owner
        for owner in owners
        if owner.foreign_branch_owner_id is not None
        or (
            owner.shared_branch_owner_id is not None
            and owner.growth_certificate is not None
            and not owner.growth_certificate.accepted
        )
    ]
    if not captured:
        return None
    owner_by_id = {owner.track_id: owner for owner in owners}
    canvas = cv.cvtColor(np.asarray(frames[-1]), cv.COLOR_GRAY2BGR)
    for owner in captured:
        assert owner.result is not None
        reference_id = (
            owner.foreign_branch_owner_id
            if owner.foreign_branch_owner_id is not None
            else owner.shared_branch_owner_id
        )
        assert reference_id is not None
        reference = owner_by_id.get(reference_id)
        candidate_curve = _active_curve_yx(
            owner,
            len(owner.result.front_indices) - 1,
        )[:, ::-1]
        if reference is not None and reference.result is not None:
            reference_curve = _active_curve_yx(
                reference,
                len(reference.result.front_indices) - 1,
            )[:, ::-1]
            cv.polylines(
                canvas,
                [np.rint(reference_curve).astype(np.int32)],
                False,
                (60, 220, 80),
                2,
                cv.LINE_AA,
            )
        cv.polylines(
            canvas,
            [np.rint(candidate_curve).astype(np.int32)],
            False,
            (70, 90, 255),
            2,
            cv.LINE_AA,
        )
        capture_point = min(
            owner.foreign_branch_capture_point or 0,
            len(candidate_curve) - 1,
        )
        marker = tuple(np.rint(candidate_curve[capture_point]).astype(int))
        cv.circle(canvas, marker, 4, (0, 230, 255), -1, cv.LINE_AA)
        cv.putText(
            canvas,
            f"P{owner.track_id}->P{reference_id}",
            (marker[0] + 5, marker[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            f"P{owner.track_id}->P{reference_id}",
            (marker[0] + 5, marker[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    title_top = canvas.shape[0] - 29
    cv.rectangle(
        canvas,
        (4, title_top),
        (282, canvas.shape[0] - 4),
        (0, 0, 0),
        -1,
    )
    cv.putText(
        canvas,
        "foreign branch audit: candidate / earlier owner",
        (9, canvas.shape[0] - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    path = output / "foreign_branch_audit.png"
    cv.imwrite(str(path), canvas)
    return path


def main() -> None:
    """Run causal reconstruction, export measurements, and render diagnostics."""

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.field_cache / "manifest.json").read_text())
    movie = Path(manifest["movie"])
    source_frames = np.load(args.field_cache / "source_frames.npy")
    shifts_xy = np.load(args.field_cache / "shifts_xy.npy")
    frames = np.load(args.field_cache / "aligned_frames.npy", mmap_mode="r")
    fps = float(manifest["fps"])
    native_width = int(manifest["native_width"])
    native_height = int(manifest["native_height"])
    source_shape = (native_height, native_width)
    analysis_width = int(manifest["analysis_width"])
    analysis_scale = analysis_width / native_width
    pollen_radius_px = args.pollen_radius_source_px * analysis_scale
    if (
        not 0.0 <= args.native_minimum_completion_fraction <= 1.0
        or args.native_maximum_onset_delay_samples < 0
        or args.native_normal_search_px < 0.0
        or not 0.0
        <= args.native_consensus_minimum_length_fraction
        <= 1.0
        or args.native_boundary_minimum_length_gain_fraction < 0.0
        or args.branch_capture_distance_radii <= 0.0
        or args.branch_capture_minimum_shared_radii <= 0.0
        or args.branch_capture_minimum_onset_lead_samples < 0
        or args.forecast_veto_min_faults < 0
        or not 0.0 <= args.forecast_veto_min_rate <= 1.0
        ):
        raise ValueError("invalid native or branch-attribution controls")
    direction_kwargs = {
        "minimum_inward_score_margin": args.minimum_inward_direction_score_margin,
        "minimum_inward_coverage": args.minimum_inward_direction_coverage,
        "minimum_inward_direct_support": (
            args.minimum_inward_direction_direct_support
        ),
        "minimum_inward_eventual_support": (
            args.minimum_inward_direction_eventual_support
        ),
    }
    selected = (
        None if not args.track_ids else {int(value) for value in args.track_ids.split(",")}
    )
    centers_by_id, owner_motion_fusion = _load_owner_centers(
        args.owner_motion_cache,
        source_frames,
        shifts_xy,
        analysis_scale,
        args.owner_motion_mode,
    )
    motion_ambiguity_ids = _load_motion_ambiguity_ids(args.owner_motion_cache)
    if motion_ambiguity_ids:
        owner_motion_fusion = dict(
            owner_motion_fusion,
            motion_ambiguous_owner_ids=sorted(motion_ambiguity_ids),
            motion_ambiguity_state="fail-closed-identity-switch-review",
        )
    else:
        owner_motion_fusion = dict(
            owner_motion_fusion,
            motion_ambiguous_owner_ids=[],
            motion_ambiguity_state="no-ambiguous-owner-motion",
        )
    forecast_veto_counts = (
        _load_forecast_veto_counts(args.forecast_veto_dir)
        if args.forecast_veto_dir is not None
        else {}
    )
    forecast_veto_ids = {
        owner
        for owner, (n_fault, n) in forecast_veto_counts.items()
        if _forecast_stability_veto_trips(
            n_fault,
            n,
            min_faults=args.forecast_veto_min_faults,
            min_rate=args.forecast_veto_min_rate,
        )
    }
    for owner in sorted(forecast_veto_ids):
        print(
            f"P{owner:02d}: TimesFM closed-loop veto tripped "
            f"({forecast_veto_counts[owner][0]} fault-suspect of "
            f"{forecast_veto_counts[owner][1]} replayed); "
            "fail-closed to forecast-stability-review",
            flush=True,
        )
    forecast_veto_report = {
        "veto_dir": (
            None
            if args.forecast_veto_dir is None
            else str(args.forecast_veto_dir.resolve())
        ),
        "minimum_faults": args.forecast_veto_min_faults,
        "minimum_rate": args.forecast_veto_min_rate,
        "vetoed_owner_ids": sorted(forecast_veto_ids),
        "fault_counts_by_owner": {
            str(owner): list(forecast_veto_counts[owner])
            for owner in sorted(forecast_veto_counts)
        },
    }
    forecast_tip_holds = (
        _load_forecast_tip_holds(args.forecast_tip_dir)
        if args.forecast_tip_dir is not None
        else {}
    )
    forecast_tip_hold_report = {
        "tip_dir": (
            None
            if args.forecast_tip_dir is None
            else str(args.forecast_tip_dir.resolve())
        ),
        "holds_by_owner": {
            str(owner): list(forecast_tip_holds[owner])
            for owner in sorted(forecast_tip_holds)
        },
    }
    sticky_tip_evidence = (
        _load_sticky_tip_evidence(args.sticky_tip_dir)
        if args.sticky_tip_dir is not None
        else {}
    )
    sticky_tip_report = {
        "tip_dir": (
            None
            if args.sticky_tip_dir is None
            else str(args.sticky_tip_dir.resolve())
        ),
        "corrections_by_owner": {
            str(owner): list(sticky_tip_evidence[owner])
            for owner in sorted(sticky_tip_evidence)
        },
    }
    field_summary = pd.read_csv(args.field_run / "field_summary.csv")
    atlas_paths = (
        pd.read_csv(args.causal_atlas_run / "owner_paths.csv")
        if args.causal_atlas_run is not None
        else pd.DataFrame()
    )
    status_by_id = dict(zip(field_summary.pollen_id, field_summary.field_status))
    identity_tier_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "owner_identity_tier",
                pd.Series("unknown", index=field_summary.index),
            ),
        )
    )
    identity_observations_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "identity_observation_count",
                pd.Series(0, index=field_summary.index),
            ),
        )
    )
    semantic_observations_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "semantic_observation_count",
                pd.Series(0, index=field_summary.index),
            ),
        )
    )
    assignment_fraction_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "geometric_assignment_fraction",
                pd.Series(np.nan, index=field_summary.index),
            ),
        )
    )
    median_template_score_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "median_template_score",
                pd.Series(np.nan, index=field_summary.index),
            ),
        )
    )
    body_status_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "owner_body_status",
                pd.Series("legacy-unverified", index=field_summary.index),
            ),
        )
    )
    identity_certificate_basis_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "owner_identity_certificate_basis",
                pd.Series("legacy-unverified", index=field_summary.index),
            ),
        )
    )
    pregrowth_consensus_by_id = dict(
        zip(
            field_summary.pollen_id,
            field_summary.get(
                "owner_pregrowth_semantic_consensus",
                pd.Series(0, index=field_summary.index),
            ),
        )
    )

    def optional_owner_metric(column: str, default: float = np.nan) -> dict:
        """Return one optional bootstrap metric keyed by pollen identifier."""

        return dict(
            zip(
                field_summary.pollen_id,
                field_summary.get(
                    column,
                    pd.Series(default, index=field_summary.index),
                ),
            )
        )

    body_radial_contrast_by_id = optional_owner_metric(
        "body_median_radial_contrast"
    )
    body_angular_fraction_by_id = optional_owner_metric(
        "body_median_angular_boundary_fraction"
    )
    body_opposite_fraction_by_id = optional_owner_metric(
        "body_median_opposite_boundary_fraction"
    )
    body_valid_fraction_by_id = optional_owner_metric(
        "body_median_valid_angular_fraction"
    )
    body_valid_samples_by_id = optional_owner_metric(
        "body_valid_sample_count", 0
    )
    owners: list[OwnerGeometry] = []
    for track_id in sorted(status_by_id):
        track_id = int(track_id)
        if selected is not None and track_id not in selected:
            continue
        owner_dir = args.field_run / f"P{track_id:02d}"
        centerline_path = owner_dir / "centerlines.csv"
        if track_id not in centers_by_id or not centerline_path.exists():
            continue
        if track_id in motion_ambiguity_ids:
            print(
                f"P{track_id:02d}: withheld before tracing "
                "(motion-ambiguous owner identity; "
                "fail-closed-identity-switch-review)",
                flush=True,
            )
            continue
        centerlines = pd.read_csv(centerline_path)
        if len(centerlines) < 2:
            continue
        measurements = pd.read_csv(owner_dir / "measurements.csv")
        blocking_centers = _blocking_centers_for_owner(
            centers_by_id,
            body_status_by_id,
            identity_tier_by_id,
            track_id,
        )
        dynamic_paths, arclength = _owner_translated_material_paths(
            centerlines,
            centers_by_id[track_id],
            shifts_xy,
            analysis_scale,
            args.arc_step_px,
        )
        (
            dynamic_paths,
            arclength,
            field_contact_owner,
            field_uncensored_length,
        ) = _foreign_owner_safe_prefix(
            dynamic_paths,
            arclength,
            track_id=track_id,
            centers_by_id=blocking_centers,
            owner_radius_px=(
                pollen_radius_px * args.foreign_owner_exclusion_radii
            ),
        )
        field_source = (
            "field"
            if field_contact_owner is None
            else f"field-contact-P{field_contact_owner:02d}"
        )
        mature_candidates = [
            _candidate_geometry(
                field_source,
                dynamic_paths,
                arclength,
                centers_by_id[track_id],
                foreign_contact_owner_id=field_contact_owner,
                uncensored_length_px=field_uncensored_length,
            )
        ]
        if not atlas_paths.empty:
            atlas_owner = atlas_paths[atlas_paths["owner_track_id"] == track_id]
            if len(atlas_owner) >= 2:
                atlas_dynamic, atlas_arclength = _atlas_material_paths(
                    atlas_owner,
                    centers_by_id[track_id],
                    args.arc_step_px,
                )
                (
                    atlas_dynamic,
                    atlas_arclength,
                    contact_owner,
                    uncensored_length,
                ) = _foreign_owner_safe_prefix(
                    atlas_dynamic,
                    atlas_arclength,
                    track_id=track_id,
                    centers_by_id=blocking_centers,
                    owner_radius_px=(
                        pollen_radius_px * args.foreign_owner_exclusion_radii
                    ),
                )
                atlas_source = (
                    "causal-atlas"
                    if contact_owner is None
                    else f"causal-atlas-contact-P{contact_owner:02d}"
                )
                mature_candidates.append(
                    _candidate_geometry(
                        atlas_source,
                        atlas_dynamic,
                        atlas_arclength,
                        centers_by_id[track_id],
                        foreign_contact_owner_id=contact_owner,
                        uncensored_length_px=uncensored_length,
                    )
                )
        owners.append(
            OwnerGeometry(
                track_id=track_id,
                field_status=str(status_by_id[track_id]),
                identity_tier=str(identity_tier_by_id[track_id]),
                identity_observation_count=int(
                    identity_observations_by_id[track_id]
                ),
                semantic_observation_count=int(
                    semantic_observations_by_id[track_id]
                ),
                geometric_assignment_fraction=float(
                    assignment_fraction_by_id[track_id]
                ),
                median_owner_template_score=float(
                    median_template_score_by_id[track_id]
                ),
                body_status=str(body_status_by_id[track_id]),
                identity_certificate_basis=str(
                    identity_certificate_basis_by_id[track_id]
                ),
                pregrowth_semantic_consensus=bool(
                    pregrowth_consensus_by_id[track_id]
                ),
                body_median_radial_contrast=float(
                    body_radial_contrast_by_id[track_id]
                ),
                body_median_angular_boundary_fraction=float(
                    body_angular_fraction_by_id[track_id]
                ),
                body_median_opposite_boundary_fraction=float(
                    body_opposite_fraction_by_id[track_id]
                ),
                body_median_valid_angular_fraction=float(
                    body_valid_fraction_by_id[track_id]
                ),
                body_valid_sample_count=int(body_valid_samples_by_id[track_id]),
                original_measurements=measurements,
                mature_candidates=mature_candidates,
                arclength_px=arclength,
                dynamic_paths_xy=dynamic_paths,
                forecast_fault_samples=int(
                    forecast_veto_counts.get(track_id, (0, 0))[0]
                ),
                forecast_replay_samples=int(
                    forecast_veto_counts.get(track_id, (0, 0))[1]
                ),
                forecast_stability_veto=track_id in forecast_veto_ids,
                forecast_tip_hold_samples=int(
                    forecast_tip_holds.get(track_id, (0, 0, 0))[0]
                ),
                forecast_tip_hold_max_run=int(
                    forecast_tip_holds.get(track_id, (0, 0, 0))[1]
                ),
                forecast_tip_series_samples=int(
                    forecast_tip_holds.get(track_id, (0, 0, 0))[2]
                ),
                sticky_tip_corrected_samples=int(
                    sticky_tip_evidence.get(track_id, (0, 0.0, 0.0, 0))[0]
                ),
                sticky_tip_max_move_px=float(
                    sticky_tip_evidence.get(track_id, (0, 0.0, 0.0, 0))[1]
                ),
                sticky_tip_mean_move_px=float(
                    sticky_tip_evidence.get(track_id, (0, 0.0, 0.0, 0))[2]
                ),
                sticky_tip_series_samples=int(
                    sticky_tip_evidence.get(track_id, (0, 0.0, 0.0, 0))[3]
                ),
            )
        )
    if not owners:
        raise ValueError("no owner paths were available for causal reconstruction")

    rigid_profiles: dict[tuple[int, int], list[np.ndarray]] = {
        (owner.track_id, index): []
        for owner in owners
        for index in range(len(owner.mature_candidates))
    }
    deformation_guide = np.zeros(frames.shape, dtype=np.float32)
    for mode, minimum_change in (("dark", 3.0), ("dog", 0.8), ("blackhat", 0.8)):
        evidence = _feature_stack(frames, mode, pollen_radius_px)
        if mode in {"dog", "blackhat"}:
            positive = np.maximum(evidence, 0.0)
            scales = np.percentile(
                positive.reshape(len(positive), -1),
                99.0,
                axis=1,
            )
            deformation_guide += 0.5 * np.clip(
                positive / np.maximum(scales[:, None, None], 1e-6),
                0.0,
                1.0,
            )
        for owner in owners:
            for index, candidate in enumerate(owner.mature_candidates):
                rigid_profiles[(owner.track_id, index)].append(
                    dynamic_path_novelty_profiles(
                        evidence,
                        candidate.dynamic_paths_xy,
                        warmup_samples=args.warmup_samples,
                        absolute_evidence_floor=0.0,
                        normal_halfwidth_px=args.normal_halfwidth_px,
                        normal_sample_count=args.normal_samples,
                        minimum_change=minimum_change,
                        noise_multiplier=2.0,
                        temporal_median_samples=3,
                    )
                )
        del evidence

    for owner in owners:
        for index, candidate in enumerate(owner.mature_candidates):
            candidate.profile = np.mean(
                rigid_profiles[(owner.track_id, index)], axis=0
            ).astype(np.float32)
            candidate.result = causal_changepoint_front(
                candidate.profile,
                candidate.arclength_px,
                warmup_samples=args.warmup_samples,
                max_step_px=args.max_growth_px_per_sample,
                change_window_samples=7,
                persistence_window_samples=21,
            )
            _certify_candidate_topology(
                candidate,
                centers_by_id[owner.track_id],
                pollen_radius_px,
            )
        selection = select_decisive_causal_path(
            [_candidate_hypothesis(candidate) for candidate in owner.mature_candidates],
            owner_radius_px=pollen_radius_px,
        )
        selected_candidate = owner.mature_candidates[selection.index]
        owner.selected_mature_path = selected_candidate.source
        owner.mature_path_selection_reason = selection.reason
        owner.selected_root_distance_px = selected_candidate.root_distance_px
        owner.selected_radial_excursion_px = selected_candidate.radial_excursion_px
        owner.selected_foreign_contact_owner_id = (
            selected_candidate.foreign_contact_owner_id
        )
        owner.arclength_px = selected_candidate.arclength_px
        owner.dynamic_paths_xy = selected_candidate.dynamic_paths_xy
        owner.rigid_profile = selected_candidate.profile
        owner.rigid_result = selected_candidate.result
        owner.rigid_paths_xy = owner.dynamic_paths_xy.copy()
        owner.deformation = fit_scalar_path_deformation(
            deformation_guide,
            owner.dynamic_paths_xy,
            normal_radius_px=args.normal_halfwidth_px,
            material_smoothing_sigma=4.0,
        )
        owner.dynamic_paths_xy = project_inextensible_paths(
            owner.deformation.curves_xy,
            owner.arclength_px,
        )

    deformed_profiles_by_id: dict[int, list[np.ndarray]] = {
        owner.track_id: [] for owner in owners
    }
    for mode, minimum_change in (("dark", 3.0), ("dog", 0.8), ("blackhat", 0.8)):
        evidence = _feature_stack(frames, mode, pollen_radius_px)
        for owner in owners:
            deformed_profiles_by_id[owner.track_id].append(
                dynamic_path_novelty_profiles(
                    evidence,
                    owner.dynamic_paths_xy,
                    warmup_samples=args.warmup_samples,
                    absolute_evidence_floor=0.0,
                    normal_halfwidth_px=args.normal_halfwidth_px,
                    normal_sample_count=args.normal_samples,
                    minimum_change=minimum_change,
                    noise_multiplier=2.0,
                    temporal_median_samples=3,
                )
            )
        del evidence

    for owner in owners:
        owner.profile = np.mean(
            deformed_profiles_by_id[owner.track_id], axis=0
        ).astype(np.float32)
        owner.deformed_result = causal_changepoint_front(
            owner.profile,
            owner.arclength_px,
            warmup_samples=args.warmup_samples,
            max_step_px=args.max_growth_px_per_sample,
            change_window_samples=7,
            persistence_window_samples=21,
            minimum_post_samples=1,
        )
        rigid_is_supported = bool(
            owner.rigid_result.feasible
            and owner.rigid_result.direct_support_fraction >= 0.80
            and owner.rigid_result.eventual_support_fraction >= 0.90
        )
        if (
            rigid_is_supported
            and owner.rigid_result.final_length_px
            > owner.deformed_result.final_length_px + 0.5 * args.arc_step_px
        ):
            owner.result = owner.rigid_result
            owner.profile = owner.rigid_profile
            owner.dynamic_paths_xy = owner.rigid_paths_xy
            owner.selected_pose_model = "pollen-translated"
        else:
            owner.result = owner.deformed_result
            owner.selected_pose_model = "deformable"
        reverse_arclength = owner.arclength_px[-1] - owner.arclength_px[::-1]
        owner.reverse_result = causal_changepoint_front(
            owner.profile[:, ::-1],
            reverse_arclength,
            warmup_samples=args.warmup_samples,
            max_step_px=args.max_growth_px_per_sample,
            change_window_samples=7,
            persistence_window_samples=21,
            minimum_post_samples=1,
        )
        _refresh_owner_topology(
            owner,
            centers_by_id[owner.track_id],
            pollen_radius_px,
        )
        owner.direction_certificate = certify_growth_direction(
            owner.result,
            owner.reverse_result,
            total_length_px=float(owner.arclength_px[-1]),
            **direction_kwargs,
        )

    native_frame_cache = NativeGrayFrameCache(movie)
    native_review_verification = _verify_review_paths_at_native_resolution(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        germination_length_px=args.germination_length_source_px,
        minimum_completion_fraction=args.native_minimum_completion_fraction,
        maximum_onset_delay_samples=args.native_maximum_onset_delay_samples,
        normal_search_px=args.native_normal_search_px,
        native_frame_cache=native_frame_cache,
    )
    shared_branch_arbitration = _arbitrate_shared_branches(
        owners,
        centers_by_id,
        owner_radius_px=pollen_radius_px,
        germination_length_px=(
            args.germination_length_source_px * analysis_scale
        ),
        warmup_samples=args.warmup_samples,
        max_growth_px_per_sample=args.max_growth_px_per_sample,
        direction_kwargs=direction_kwargs,
    )
    foreign_branch_arbitration = _arbitrate_foreign_branch_captures(
        owners,
        centers_by_id,
        owner_radius_px=pollen_radius_px,
        germination_length_px=(
            args.germination_length_source_px * analysis_scale
        ),
        distance_radii=args.branch_capture_distance_radii,
        minimum_shared_radii=args.branch_capture_minimum_shared_radii,
        minimum_onset_lead_samples=(
            args.branch_capture_minimum_onset_lead_samples
        ),
        warmup_samples=args.warmup_samples,
        max_growth_px_per_sample=args.max_growth_px_per_sample,
        direction_kwargs=direction_kwargs,
    )
    native_centerline_refinement = _refine_centerlines_at_native_resolution(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        germination_length_px=args.germination_length_source_px,
        native_frame_cache=native_frame_cache,
    )
    retained_owner_topology = _audit_retained_owner_topologies(
        owners,
        centers_by_id,
        pollen_radius_px,
    )
    native_path_quality = _certify_retained_paths_at_native_resolution(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        minimum_distal_support_fraction=0.40,
        native_frame_cache=native_frame_cache,
    )
    native_boundary_candidates = _audit_native_boundary_candidates(
        movie,
        frames,
        deformation_guide,
        owners,
        centers_by_id,
        body_status_by_id,
        identity_tier_by_id,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        pollen_radius_px=pollen_radius_px,
        warmup_samples=args.warmup_samples,
        normal_halfwidth_px=args.normal_halfwidth_px,
        normal_samples=args.normal_samples,
        max_growth_px_per_sample=args.max_growth_px_per_sample,
        direction_kwargs=direction_kwargs,
        native_frame_cache=native_frame_cache,
        minimum_proximal_support_ratio=(
            args.native_minimum_proximal_support_ratio
        ),
    )
    native_boundary_rescue = _select_native_boundary_rescues(
        owners,
        analysis_scale,
        pollen_radius_px,
        args.native_coverage_tie_tolerance,
        args.native_deformation_minimum_coverage_gain,
        args.native_consensus_minimum_length_fraction,
        args.native_boundary_minimum_length_gain_fraction,
    )
    native_rescue_recenter = _recenter_rescued_paths_at_native_resolution(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        search_radius_px=args.native_recenter_search_radius_px,
        maximum_length_change_fraction=0.05,
        minimum_median_abs_offset_px=args.native_recenter_minimum_offset_px,
        native_frame_cache=native_frame_cache,
    )
    retained_owner_topology = _audit_retained_owner_topologies(
        owners,
        centers_by_id,
        pollen_radius_px,
    )
    native_path_recovery = _audit_unresolved_native_path_recovery(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    native_root_onset_audit = _audit_native_root_emergence(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        maximum_growth_analysis_px_per_sample=(
            args.max_growth_px_per_sample
        ),
        native_frame_cache=native_frame_cache,
    )
    native_front_refinement = _audit_native_tip_fronts(
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale=analysis_scale,
        source_shape=source_shape,
        native_frame_cache=native_frame_cache,
    )
    post_refinement_duplicate_audit = _audit_retained_duplicate_claims(
        owners,
        owner_radius_px=pollen_radius_px,
        germination_length_px=(
            args.germination_length_source_px * analysis_scale
        ),
    )
    native_boundary_rescue["duplicate_veto_owner_ids"] = (
        _resolve_final_duplicate_claims(
            owners,
            post_refinement_duplicate_audit,
        )
    )
    for owner in owners:
        _write_kymograph(args.output, owner, analysis_scale)
        print(
            f"P{owner.track_id:02d}: {owner.result.reason}, "
            f"{owner.original_measurements.tube_length_px.iloc[-1]:.1f} -> "
            f"{owner.result.final_length_px / analysis_scale:.1f} px "
            f"(rigid {owner.rigid_result.final_length_px / analysis_scale:.1f})",
            flush=True,
        )

    counts, export_consistency = _export_tables(
        args.output,
        owners,
        source_frames,
        shifts_xy,
        fps,
        analysis_scale,
        args.germination_length_source_px,
        source_shape,
    )
    demo = _render_demo(
        args.output,
        frames,
        owners,
        centers_by_id,
        source_frames,
        fps,
        pollen_radius_px,
        args.demo_fps,
    )
    foreign_branch_audit = _render_foreign_branch_audit(
        args.output,
        frames,
        owners,
    )
    native_centerline_audit = _render_native_centerline_audit(
        args.output,
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    native_path_recovery_audit = _render_native_path_recovery_audit(
        args.output,
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    unresolved_candidate_audit = _render_unresolved_candidate_audit(
        args.output,
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    native_boundary_candidate_audit = _render_native_boundary_candidate_audit(
        args.output,
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    native_boundary_rescue_time_audit = (
        _render_native_boundary_rescue_time_audit(
            args.output,
            movie,
            owners,
            source_frames,
            shifts_xy,
            analysis_scale,
            centers_by_id,
            native_frame_cache=native_frame_cache,
        )
    )
    native_front_audit = _render_native_front_audit(
        args.output,
        movie,
        owners,
        source_frames,
        shifts_xy,
        analysis_scale,
        native_frame_cache=native_frame_cache,
    )
    boundary_trace_config = (
        NativeBoundaryTracingConfig().for_boundary_censoring().trace
    )
    faint_extension_trace_config = (
        NativeBoundaryTracingConfig().for_faint_prefix_extension().trace
    )
    report = {
        "revision": REVISION,
        "field_run": str(args.field_run.resolve()),
        "field_cache": str(args.field_cache.resolve()),
        "owner_motion_cache": str(args.owner_motion_cache.resolve()),
        "owner_motion_fusion": owner_motion_fusion,
        "native_frame_cache": native_frame_cache.report(),
        "causal_atlas_run": (
            None
            if args.causal_atlas_run is None
            else str(args.causal_atlas_run.resolve())
        ),
        "sample_interval_seconds": float(np.median(np.diff(source_frames)) / fps),
        "analysis_scale": analysis_scale,
        "configuration": {
            "foreign_owner_exclusion_radii": args.foreign_owner_exclusion_radii,
            "pollen_body_identity": {
                "tier_order": ["provisional", "supported", "high"],
                "foreign_mask_rule": "verified-pollen-bodies-only",
                "duplicate_rule": "verified-body-priority-equal-evidence-withheld",
                "rejected_body_automatic_measurement": False,
                "indeterminate_body_automatic_measurement": False,
                "motion_ambiguity_rule": (
                    "motion-ambiguous owners keep their predicted coordinate "
                    "but fail closed to identity-switch-review before tracing"
                ),
                "pregrowth_consensus_rule": (
                    "explicit repeated pre-tube semantic identity supersedes "
                    "ambiguous low-resolution radial shape"
                ),
                "legacy_bootstrap_fallback": "equal-or-higher-tier-only",
                "retains_all_learned_owner_candidates": True,
            },
            "growth_direction": direction_kwargs,
            "native_review_verification": {
                "minimum_completion_fraction": (
                    args.native_minimum_completion_fraction
                ),
                "maximum_onset_delay_samples": (
                    args.native_maximum_onset_delay_samples
                ),
                "normal_search_px": args.native_normal_search_px,
            },
            "native_centerline_refinement": {
                "search_radius_source_px": 8,
                "maximum_observations": 8,
                "minimum_observations": 3,
                "maximum_length_change_fraction": 0.20,
            },
            "native_rescue_recenter": {
                "search_radius_source_px": args.native_recenter_search_radius_px,
                "maximum_observations": 8,
                "minimum_observations": 3,
                "maximum_length_change_fraction": 0.05,
                "minimum_median_abs_offset_px": (
                    args.native_recenter_minimum_offset_px
                ),
            },
            "retained_owner_topology": {
                "minimum_root_radius_factor": 0.65,
                "maximum_root_radius_factor": 1.65,
                "minimum_body_clearance_factor": 0.65,
                "halo_exit_radius_factor": 1.55,
                "halo_return_radius_factor": 1.35,
                "maximum_halo_exit_arclength_radii": 1.75,
                "minimum_radial_excursion_radii": 0.65,
            },
            "native_path_quality": {
                "maximum_observations": 8,
                "minimum_observations": 3,
                "minimum_completion_fraction": 0.85,
                "minimum_median_coverage": 0.40,
            },
            "native_path_recovery": {
                "search_radius_source_px": 24,
                "maximum_observations": 12,
                "minimum_holdout_observations": 3,
                "minimum_holdout_coverage": 0.40,
                "minimum_holdout_gain": 0.15,
                "maximum_length_change_fraction": 0.20,
            },
            "native_boundary_candidates": {
                "crop_radius_source_px": 150,
                "native_wall_pairing": True,
                "movie_wide_causal_validation": True,
                "repeated_native_quality_validation": True,
                "minimum_median_native_coverage": 0.50,
                "coverage_tie_tolerance": (
                    args.native_coverage_tie_tolerance
                ),
                "deformation_minimum_coverage_gain": (
                    args.native_deformation_minimum_coverage_gain
                ),
                "maximum_root_onset_difference_samples": 14,
                "minimum_independent_departure_radii": 1.0,
                "owner_aligned_consensus_sample_counts": [3, 11],
                "consensus_minimum_length_fraction": (
                    args.native_consensus_minimum_length_fraction
                ),
                "boundary_minimum_length_gain_fraction": (
                    args.native_boundary_minimum_length_gain_fraction
                ),
                "boundary_censored_trace": {
                    "requires_retained_image_edge_contact": True,
                    "maximum_gap_steps": boundary_trace_config.maximum_gap_steps,
                    "maximum_total_turn_bins": (
                        boundary_trace_config.maximum_total_turn_bins
                    ),
                    "minimum_pair_support": (
                        boundary_trace_config.minimum_pair_support
                    ),
                    "evidence_floor": boundary_trace_config.evidence_floor,
                    "length_reward": boundary_trace_config.length_reward,
                },
                "faint_prefix_extension": {
                    "requires_complete_reference_prefix": True,
                    "minimum_length_gain_source_px": 6.0,
                    "maximum_root_distance_source_px": 4.0,
                    "maximum_prefix_p90_error_source_px": 3.0,
                    "maximum_gap_steps": (
                        faint_extension_trace_config.maximum_gap_steps
                    ),
                    "maximum_total_turn_bins": (
                        faint_extension_trace_config.maximum_total_turn_bins
                    ),
                    "minimum_pair_support": (
                        faint_extension_trace_config.minimum_pair_support
                    ),
                    "evidence_floor": faint_extension_trace_config.evidence_floor,
                    "length_reward": faint_extension_trace_config.length_reward,
                },
            },
            "native_root_onset_audit": {
                "root_extent_source_px": 20.0,
                "minimum_growth_source_px": 3.0,
                "maximum_onset_difference_samples": 12,
                "maximum_growth_analysis_px_per_sample": (
                    args.max_growth_px_per_sample
                ),
            },
            "native_front_refinement": {
                "dense_step_source_px": 1.0,
                "corridor_radius_source_px": 4.0,
            },
            "foreign_branch_attribution": {
                "distance_radii": args.branch_capture_distance_radii,
                "minimum_shared_radii": (
                    args.branch_capture_minimum_shared_radii
                ),
                "minimum_onset_lead_samples": (
                    args.branch_capture_minimum_onset_lead_samples
                ),
            },
            "forecast_stability_veto": {
                "veto_dir": (
                    None
                    if args.forecast_veto_dir is None
                    else str(args.forecast_veto_dir)
                ),
                "minimum_faults": args.forecast_veto_min_faults,
                "minimum_rate": args.forecast_veto_min_rate,
                "rule": (
                    "TimesFM closed-loop fault-suspect replays with "
                    "negligible length gain fail an owner closed to "
                    "forecast-stability-review (diagnostic, no automatic "
                    "measurement); omitted veto input disables the gate"
                ),
            },
        },
        "native_review_verification": native_review_verification,
        "native_centerline_refinement": native_centerline_refinement,
        "retained_owner_topology": retained_owner_topology,
        "native_path_quality": native_path_quality,
        "native_path_recovery": native_path_recovery,
        "native_boundary_candidates": native_boundary_candidates,
        "native_boundary_rescue": native_boundary_rescue,
        "native_rescue_recenter": native_rescue_recenter,
        "forecast_stability_veto": forecast_veto_report,
        "forecast_tip_holds": forecast_tip_hold_report,
        "sticky_tip_corrections": sticky_tip_report,
        "native_root_onset_audit": native_root_onset_audit,
        "native_front_refinement": native_front_refinement,
        "post_refinement_duplicate_audit": post_refinement_duplicate_audit,
        "shared_branch_arbitration": shared_branch_arbitration,
        "foreign_branch_arbitration": foreign_branch_arbitration,
        "counts": counts,
        "export_consistency": export_consistency,
        "artifacts": {
            "summary": str((args.output / "summary.csv").resolve()),
            "measurements": str((args.output / "measurements.csv").resolve()),
            "centerlines": str((args.output / "centerlines.csv").resolve()),
            "diagnostic_centerlines": str(
                (args.output / "diagnostic_centerlines.csv").resolve()
            ),
            "demo_video": str(demo.resolve()),
            "foreign_branch_audit": (
                None
                if foreign_branch_audit is None
                else str(foreign_branch_audit.resolve())
            ),
            "native_centerline_audit": str(native_centerline_audit.resolve()),
            "native_path_recovery_audit": (
                None
                if native_path_recovery_audit is None
                else str(native_path_recovery_audit.resolve())
            ),
            "unresolved_candidate_audit": (
                None
                if unresolved_candidate_audit is None
                else str(unresolved_candidate_audit.resolve())
            ),
            "native_boundary_candidate_audit": (
                None
                if native_boundary_candidate_audit is None
                else str(native_boundary_candidate_audit.resolve())
            ),
            "native_boundary_rescue_time_audit": (
                None
                if native_boundary_rescue_time_audit is None
                else str(native_boundary_rescue_time_audit.resolve())
            ),
            "native_front_refinement_audit": (
                None
                if native_front_audit is None
                else str(native_front_audit.resolve())
            ),
        },
        "scope": "Research prototype; validate against blinded manual centerlines before quantitative use.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    native_frame_cache.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
