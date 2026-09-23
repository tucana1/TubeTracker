"""Prepare, lock, and score blinded pollen-tube centerline references."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import random
import tempfile
from typing import Iterable

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree


REFERENCE_OUTCOMES = (
    "unreviewed",
    "no_visible_tube",
    "visible_tube",
    "field_censored",
    "owner_not_visible",
    "incorrect_target",
    "ambiguous",
)
POSITIVE_REFERENCE_OUTCOMES = {"visible_tube", "field_censored"}
USABLE_PREDICTION_STATUSES = {
    "measured",
    "contact_censored",
    "left_censored",
    "boundary_censored",
}
REFERENCE_FRAME_FIELDS = (
    "case_id",
    "pollen_id",
    "record_id",
    "sample_index",
    "source_frame",
    "time_seconds",
    "stratum",
    "reference_outcome",
    "reference_length_px",
    "point_count",
)
REFERENCE_CENTERLINE_FIELDS = (
    "case_id",
    "pollen_id",
    "record_id",
    "source_frame",
    "point_index",
    "source_x_px",
    "source_y_px",
    "arc_length_px",
)


def _read_json(path: Path) -> dict:
    """Read one JSON object from disk."""

    return json.loads(path.read_text())


def _atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON file without exposing a partially written annotation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable reference artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque_case_id(seed: int, movie: Path, track_id: int) -> str:
    """Create a stable case label that does not reveal the pollen ID."""

    token = f"{seed}:{movie.resolve()}:{track_id}".encode()
    return hashlib.sha256(token).hexdigest()[:8].upper()


def _parse_track_ids(value: str | Iterable[int]) -> list[int]:
    """Normalize unique positive track IDs while preserving input order."""

    if isinstance(value, str):
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [int(item) for item in value]
    result = list(dict.fromkeys(values))
    if not result or any(track_id < 1 for track_id in result):
        raise ValueError("track IDs must be positive integers")
    return result


def _sample_indices(
    lengths: np.ndarray,
    onset_index: int | None,
    uniform_count: int,
    challenge_count: int,
) -> list[tuple[int, str]]:
    """Select fixed-coverage and prediction-challenge samples deterministically."""

    sample_count = len(lengths)
    if sample_count == 0:
        return []
    uniform_count = min(max(1, uniform_count), sample_count)
    uniform = {
        int(round(index))
        for index in np.linspace(0, sample_count - 1, uniform_count)
    }
    challenge_candidates: list[int] = []
    if challenge_count > 0:
        if onset_index is not None:
            for offset in (0, -1, 1, -2, 2, -4, 4):
                challenge_candidates.append(onset_index + offset)
        changes = np.abs(np.diff(lengths, prepend=lengths[0]))
        challenge_candidates.extend(
            int(index) for index in np.argsort(changes)[::-1]
        )
    challenge: list[int] = []
    for index in challenge_candidates:
        if not 0 <= index < sample_count or index in uniform or index in challenge:
            continue
        challenge.append(index)
        if len(challenge) >= max(0, challenge_count):
            break
    selected = [(index, "uniform") for index in uniform]
    selected.extend((index, "challenge") for index in challenge)
    return sorted(selected)


def _interpolate_owner_center(track: dict, source_frame: int) -> tuple[float, float]:
    """Interpolate a pollen center from its sparse identity observations."""

    frames = np.asarray(track["source_frames"], dtype=float)
    centers = np.asarray(track["centers_yx"], dtype=float)
    if len(frames) == 0 or centers.shape != (len(frames), 2):
        raise ValueError(f"invalid center history for track {track.get('track_id')}")
    y = float(np.interp(source_frame, frames, centers[:, 0]))
    x = float(np.interp(source_frame, frames, centers[:, 1]))
    return y, x


def _raw_crop(
    frame: np.ndarray,
    center_yx: tuple[float, float],
    size: int,
) -> tuple[np.ndarray, int, int]:
    """Crop around a pollen without hiding portions outside the movie field."""

    if size < 32:
        raise ValueError("crop size must be at least 32 pixels")
    center_y, center_x = center_yx
    height, width = frame.shape[:2]
    centered_x0 = int(round(center_x - size / 2.0))
    centered_y0 = int(round(center_y - size / 2.0))
    x0 = (
        int(np.clip(centered_x0, 0, width - size))
        if width >= size
        else (width - size) // 2
    )
    y0 = (
        int(np.clip(centered_y0, 0, height - size))
        if height >= size
        else (height - size) // 2
    )
    x1, y1 = x0 + size, y0 + size
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(width, x1), min(height, y1)
    fill = np.median(frame.reshape(-1, frame.shape[2]), axis=0).astype(frame.dtype)
    crop = np.empty((size, size, frame.shape[2]), dtype=frame.dtype)
    crop[:] = fill
    if source_x1 > source_x0 and source_y1 > source_y0:
        crop[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = frame[source_y0:source_y1, source_x0:source_x1]
    return crop, x0, y0


def _verify_pollen_ring(
    frame: np.ndarray,
    center_yx: tuple[float, float],
    pollen_radius_px: float,
) -> bool:
    """Report whether a fixed owner center contains a closed pollen ring."""

    from tubetracker.causal_portal import PollenRingConfig, assess_pollen_ring

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    center_y, center_x = center_yx
    config = PollenRingConfig(
        pollen_radius_px=pollen_radius_px,
        minimum_ring_contrast=6.0,
        minimum_angular_coverage=0.50,
    )
    patch_size = max(32, int(round(4.3 * pollen_radius_px)))

    def assessment(y: float, x: float):
        patch = cv.getRectSubPix(gray, (patch_size, patch_size), (float(x), float(y)))
        return assess_pollen_ring(patch, config)

    return bool(assessment(center_y, center_x).accepted)


def _prediction_rows(prediction_run: Path) -> tuple[dict, dict[int, list[dict]]]:
    """Load owner summaries and ordered time-point measurements."""

    with (prediction_run / "summary.csv").open(newline="") as handle:
        summaries = {int(row["pollen_id"]): row for row in csv.DictReader(handle)}
    measurements: dict[int, list[dict]] = defaultdict(list)
    with (prediction_run / "measurements.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            measurements[int(row["pollen_id"])].append(row)
    for rows in measurements.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    return summaries, measurements


def _registered_owner_centers(
    movie: Path,
    identity_report: Path,
    source_frames: np.ndarray,
    native_width: int,
    native_height: int,
    analysis_width: int = 480,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Reconstruct the registered pollen trajectories used by v27."""

    from prototypes.v24_causal_birth_forest.track import (
        load_grayscale_samples,
        stabilize_translations,
    )
    from prototypes.v25_causal_portal.track import (
        load_portal_owner_seeds,
        retain_distinct_owner_tracks,
        retain_native_pollen_owners,
    )
    from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds

    scale = analysis_width / native_width
    frames = load_grayscale_samples(
        movie,
        source_frames,
        analysis_width,
        native_width,
        native_height,
    )
    aligned, shifts_xy, _ = stabilize_translations(frames)
    seeds, _ = load_portal_owner_seeds(
        identity_report,
        source_frames,
        scale,
        shifts_xy,
        include_persistent_late=True,
        minimum_late_semantic_observations=3,
    )
    trajectories, scores = track_pollen_centers_from_seeds(
        aligned,
        np.asarray([seed.center_yx for seed in seeds]),
        np.asarray([seed.seed_sample for seed in seeds]),
        template_radius_px=5,
        search_radius_px=5,
        minimum_score=0.25,
    )
    seeds, trajectories, scores, _ = retain_distinct_owner_tracks(
        seeds,
        trajectories,
        scores,
        owner_radius_px=15.0 * scale,
    )
    seeds, trajectories, scores, _ = retain_native_pollen_owners(
        movie,
        source_frames,
        scale,
        shifts_xy,
        seeds,
        trajectories,
        scores,
    )
    source_centers = (
        trajectories + shifts_xy[:, ::-1][None, :, :]
    ) / scale
    return {
        (seed.track_id, int(source_frame)): (
            float(source_centers[owner, sample, 0]),
            float(source_centers[owner, sample, 1]),
            float(scores[owner, sample]),
        )
        for owner, seed in enumerate(seeds)
        for sample, source_frame in enumerate(source_frames)
    }


def prepare_reference_kit(
    movie: Path,
    identity_report: Path,
    prediction_run: Path,
    output: Path,
    *,
    track_ids: str | Iterable[int] | None = None,
    uniform_samples: int = 8,
    challenge_samples: int = 4,
    crop_size: int = 420,
    seed: int = 2706,
    refine_centers: bool = True,
) -> dict:
    """Create a blinded, resumable reference kit from raw movie pixels."""

    movie = movie.expanduser().resolve()
    identity_report = identity_report.expanduser().resolve()
    prediction_run = prediction_run.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)

    identity = _read_json(identity_report)
    summaries, measurements = _prediction_rows(prediction_run)
    if track_ids is None:
        validation = prediction_run / "validation_manifest.json"
        requested = (
            _read_json(validation)["requested_track_ids"]
            if validation.exists()
            else sorted(summaries)
        )
    else:
        requested = _parse_track_ids(track_ids)
    missing = sorted(set(requested) - set(summaries))
    if missing:
        raise ValueError(f"prediction run is missing pollen IDs: {missing}")
    identity_tracks = {int(track["track_id"]): track for track in identity["tracks"]}
    missing = sorted(set(requested) - set(identity_tracks))
    if missing:
        raise ValueError(f"identity report is missing pollen IDs: {missing}")

    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open movie: {movie}")
    fps = float(capture.get(cv.CAP_PROP_FPS))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    sampled_source_frames = np.asarray(
        sorted(
            {
                int(row["source_frame"])
                for track_id in requested
                for row in measurements[track_id]
            }
        ),
        dtype=np.int64,
    )
    registered_centers = (
        _registered_owner_centers(
            movie,
            identity_report,
            sampled_source_frames,
            width,
            height,
        )
        if refine_centers
        else {}
    )
    cases = []
    records_by_frame: dict[int, list[dict]] = defaultdict(list)
    diameter = float(identity.get("diameter_px", 20.0))
    for track_id in requested:
        rows = measurements[track_id]
        lengths = np.asarray([float(row["tube_length_px"]) for row in rows])
        onset_source = summaries[track_id].get("first_persistent_source_frame", "")
        onset_index = None
        if onset_source != "":
            sources = np.asarray([int(row["source_frame"]) for row in rows])
            onset_index = int(np.argmin(np.abs(sources - int(onset_source))))
        case_id = _opaque_case_id(seed, movie, track_id)
        case_records = []
        for ordinal, (sample_index, stratum) in enumerate(
            _sample_indices(
                lengths,
                onset_index,
                uniform_samples,
                challenge_samples,
            )
        ):
            row = rows[sample_index]
            source_frame = int(row["source_frame"])
            center_score = None
            base_method = "sparse_identity_interpolation"
            if refine_centers:
                center = registered_centers.get((track_id, source_frame))
                if center is None:
                    capture.release()
                    raise ValueError(
                        f"registered owner track is missing P{track_id} at {source_frame}"
                    )
                registered_y, registered_x, center_score = center
                if int(identity_tracks[track_id].get("semantic_observation_count", 0)) >= 2:
                    center_y, center_x = _interpolate_owner_center(
                        identity_tracks[track_id], source_frame
                    )
                    base_method = "multi_observation_identity"
                else:
                    center_y, center_x = registered_y, registered_x
                    base_method = "registered_single_observation_owner"
            else:
                center_y, center_x = _interpolate_owner_center(
                    identity_tracks[track_id], source_frame
                )
            record = {
                "record_id": f"{case_id}-{ordinal:02d}",
                "case_id": case_id,
                "sample_index": sample_index,
                "source_frame": source_frame,
                "time_seconds": source_frame / fps,
                "stratum": stratum,
                "crop_image": f"crops/{case_id}-{ordinal:02d}.png",
                "crop_origin_x": None,
                "crop_origin_y": None,
                "target_source_x": center_x,
                "target_source_y": center_y,
                "target_base_source_x": center_x,
                "target_base_source_y": center_y,
                "target_radius_px": diameter / 2.0,
                "target_in_frame": bool(0 <= center_x < width and 0 <= center_y < height),
                "target_match_score": center_score,
                "target_center_base_method": base_method,
                "target_ring_verified": None,
            }
            case_records.append(record)
            records_by_frame[source_frame].append(record)
        cases.append({"case_id": case_id, "track_id": track_id, "records": case_records})

    random.Random(seed).shuffle(cases)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for source_frame, records in sorted(records_by_frame.items()):
        capture.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        for record in records:
            if refine_centers:
                record["target_ring_verified"] = _verify_pollen_ring(
                    frame,
                    (record["target_source_y"], record["target_source_x"]),
                    diameter / 2.0,
                )
            crop, x0, y0 = _raw_crop(
                frame,
                (record["target_source_y"], record["target_source_x"]),
                crop_size,
            )
            record["crop_origin_x"] = x0
            record["crop_origin_y"] = y0
            if not cv.imwrite(str(output / record["crop_image"]), crop):
                capture.release()
                raise RuntimeError(f"could not write {record['crop_image']}")
    capture.release()

    manifest = {
        "schema_version": 1,
        "blinded": True,
        "source_video": str(movie),
        "source_video_size_bytes": movie.stat().st_size,
        "source_frame_count": frame_count,
        "source_fps": fps,
        "source_width": width,
        "source_height": height,
        "identity_report": str(identity_report),
        "prediction_run": str(prediction_run),
        "crop_size_px": crop_size,
        "uniform_samples_per_case": uniform_samples,
        "challenge_samples_per_case": challenge_samples,
        "random_seed": seed,
        "target_center_method": (
            "multi-observation-identity-or-registered-owner-with-ring-audit"
            if refine_centers
            else "sparse_identity_interpolation"
        ),
        "case_count": len(cases),
        "record_count": sum(len(case["records"]) for case in cases),
        "cases": cases,
    }
    annotations = {
        "schema_version": 1,
        "locked": False,
        "records": {
            record["record_id"]: {
                "outcome": "unreviewed",
                "completed": False,
                "points_source_xy": [],
            }
            for case in cases
            for record in case["records"]
        },
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(output / "annotations.json", annotations)
    return manifest


def load_reference_kit(kit_dir: Path) -> tuple[dict, dict]:
    """Load immutable kit metadata and mutable annotations."""

    kit_dir = kit_dir.expanduser().resolve()
    return _read_json(kit_dir / "manifest.json"), _read_json(
        kit_dir / "annotations.json"
    )


def save_annotations(kit_dir: Path, annotations: dict) -> None:
    """Persist annotations unless the reference set has already been locked."""

    kit_dir = kit_dir.expanduser().resolve()
    if (kit_dir / "reference_lock.json").exists() or annotations.get("locked"):
        raise RuntimeError("reference annotations are locked")
    _atomic_json(kit_dir / "annotations.json", annotations)


def polyline_length(points_xy: Iterable[Iterable[float]]) -> float:
    """Return open-polyline arclength in source-image pixels."""

    points = np.asarray(list(points_xy), dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def finalize_reference_kit(kit_dir: Path, annotator: str = "") -> dict:
    """Validate annotations, export reference CSVs, and checksum-lock the kit."""

    kit_dir = kit_dir.expanduser().resolve()
    manifest, annotations = load_reference_kit(kit_dir)
    frames = []
    centerlines = []
    errors = []
    for case in manifest["cases"]:
        for record in case["records"]:
            annotation = annotations["records"][record["record_id"]]
            outcome = annotation.get("outcome", "unreviewed")
            points = annotation.get("points_source_xy", [])
            if not annotation.get("completed") or outcome == "unreviewed":
                errors.append(f"{record['record_id']} is incomplete")
                continue
            if outcome not in REFERENCE_OUTCOMES:
                errors.append(f"{record['record_id']} has invalid outcome {outcome}")
                continue
            if outcome in POSITIVE_REFERENCE_OUTCOMES and len(points) < 2:
                errors.append(f"{record['record_id']} needs a root-to-tip centerline")
                continue
            if outcome not in POSITIVE_REFERENCE_OUTCOMES and points:
                errors.append(f"{record['record_id']} has points but no visible tube")
                continue
            frames.append(
                {
                    "case_id": case["case_id"],
                    "pollen_id": case["track_id"],
                    "record_id": record["record_id"],
                    "sample_index": record["sample_index"],
                    "source_frame": record["source_frame"],
                    "time_seconds": round(record["time_seconds"], 6),
                    "stratum": record["stratum"],
                    "reference_outcome": outcome,
                    "reference_length_px": round(polyline_length(points), 6)
                    if points
                    else "",
                    "point_count": len(points),
                }
            )
            cumulative = 0.0
            for index, point in enumerate(points):
                if index:
                    cumulative += float(
                        np.linalg.norm(np.asarray(point) - np.asarray(points[index - 1]))
                    )
                centerlines.append(
                    {
                        "case_id": case["case_id"],
                        "pollen_id": case["track_id"],
                        "record_id": record["record_id"],
                        "source_frame": record["source_frame"],
                        "point_index": index,
                        "source_x_px": round(float(point[0]), 6),
                        "source_y_px": round(float(point[1]), 6),
                        "arc_length_px": round(cumulative, 6),
                    }
                )
    if errors:
        preview = "; ".join(errors[:8])
        raise ValueError(f"cannot finalize reference kit: {preview}")

    frame_path = kit_dir / "reference_frames.csv"
    centerline_path = kit_dir / "reference_centerlines.csv"
    _write_dict_rows(frame_path, frames, REFERENCE_FRAME_FIELDS)
    _write_dict_rows(centerline_path, centerlines, REFERENCE_CENTERLINE_FIELDS)
    annotations["locked"] = True
    _atomic_json(kit_dir / "annotations.json", annotations)
    lock = {
        "schema_version": 1,
        "annotator": annotator,
        "record_count": len(frames),
        "positive_record_count": sum(
            row["reference_outcome"] in POSITIVE_REFERENCE_OUTCOMES for row in frames
        ),
        "sha256": {
            path.name: _sha256(path)
            for path in (
                kit_dir / "manifest.json",
                kit_dir / "annotations.json",
                frame_path,
                centerline_path,
            )
        },
    }
    _atomic_json(kit_dir / "reference_lock.json", lock)
    return lock


def _write_dict_rows(
    path: Path,
    rows: list[dict],
    fieldnames: Iterable[str] | None = None,
) -> None:
    """Write homogeneous dictionaries to a CSV file."""

    fields = list(fieldnames or (list(rows[0]) if rows else []))
    if not fields:
        raise ValueError(f"cannot write empty reference file: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dense_polyline(points_xy: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Resample a sparse manual polyline for geometric distance scoring."""

    if len(points_xy) < 2:
        return points_xy.copy()
    segments = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    if cumulative[-1] <= 0.0:
        return points_xy[:1].copy()
    samples = np.linspace(0.0, cumulative[-1], max(2, int(cumulative[-1] / spacing) + 1))
    return np.column_stack(
        (
            np.interp(samples, cumulative, points_xy[:, 0]),
            np.interp(samples, cumulative, points_xy[:, 1]),
        )
    )


def _centerline_distances(reference_xy: np.ndarray, prediction_xy: np.ndarray) -> tuple[float, float]:
    """Return symmetric mean and 95th-percentile point-to-curve distances."""

    reference = _dense_polyline(reference_xy)
    prediction = _dense_polyline(prediction_xy)
    ref_distance = cKDTree(prediction).query(reference)[0]
    pred_distance = cKDTree(reference).query(prediction)[0]
    combined = np.concatenate((ref_distance, pred_distance))
    return float(np.mean(combined)), float(np.percentile(combined, 95))


def _scored_subset(rows: list[dict]) -> dict:
    """Summarize classification and geometry for one sampling stratum."""

    definitive = [
        row for row in rows if not row["classification"].startswith("excluded_")
    ]
    counts = {
        name: sum(row["classification"] == name for row in definitive)
        for name in ("true_positive", "true_negative", "false_positive", "false_negative")
    }
    length_errors = [
        float(row["length_absolute_error_px"])
        for row in rows
        if row["length_absolute_error_px"] != ""
    ]
    tip_errors = [
        float(row["tip_error_px"]) for row in rows if row["tip_error_px"] != ""
    ]
    centerline_errors = [
        float(row["centerline_mean_error_px"])
        for row in rows
        if row["centerline_mean_error_px"] != ""
    ]
    return {
        "definitive_frames": len(definitive),
        "identity_failure_frames": sum(
            row["reference_outcome"] == "incorrect_target" for row in rows
        ),
        "confusion": counts,
        "accuracy": (counts["true_positive"] + counts["true_negative"])
        / max(len(definitive), 1),
        "length_mae_px": float(np.mean(length_errors)) if length_errors else None,
        "tip_median_error_px": float(np.median(tip_errors)) if tip_errors else None,
        "centerline_mean_error_px": float(np.mean(centerline_errors))
        if centerline_errors
        else None,
    }


def _germination_window_rows(references: list[dict], summaries: dict) -> list[dict]:
    """Derive sampled germination intervals with and without challenge frames."""

    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in references:
        by_case[row["case_id"]].append(row)
    results = []
    for case_id, case_rows in sorted(by_case.items()):
        owner = int(case_rows[0]["pollen_id"])
        summary = summaries[owner]
        predicted = None
        if (
            summary["status"] in {"measured", "contact_censored", "boundary_censored"}
            and summary.get("first_persistent_source_frame", "") != ""
        ):
            predicted = int(summary["first_persistent_source_frame"])
        for scope in ("uniform", "uniform_plus_challenge"):
            rows = [
                row
                for row in case_rows
                if row["reference_outcome"]
                not in {"ambiguous", "owner_not_visible", "incorrect_target"}
                and (scope == "uniform_plus_challenge" or row["stratum"] == "uniform")
            ]
            rows.sort(key=lambda row: int(row["source_frame"]))
            positives = [
                row
                for row in rows
                if row["reference_outcome"] in POSITIVE_REFERENCE_OUTCOMES
            ]
            negatives = [
                row for row in rows if row["reference_outcome"] == "no_visible_tube"
            ]
            lower = upper = None
            if positives:
                upper = int(positives[0]["source_frame"])
                earlier = [
                    int(row["source_frame"])
                    for row in negatives
                    if int(row["source_frame"]) < upper
                ]
                if earlier:
                    lower = max(earlier)
                    timing_scope = "sampled_interval"
                    if predicted is None:
                        verdict = "prediction_unavailable"
                    elif lower < predicted <= upper:
                        verdict = "within_interval"
                    elif predicted <= lower:
                        verdict = "too_early"
                    else:
                        verdict = "too_late"
                else:
                    timing_scope = "left_censored"
                    verdict = "not_scored"
            elif negatives:
                timing_scope = "not_observed"
                verdict = "correct_no_growth" if predicted is None else "false_germination"
            else:
                timing_scope = "ambiguous"
                verdict = "not_scored"
            results.append(
                {
                    "case_id": case_id,
                    "pollen_id": owner,
                    "sampling_scope": scope,
                    "reference_timing_scope": timing_scope,
                    "last_no_tube_source_frame": "" if lower is None else lower,
                    "first_tube_source_frame": "" if upper is None else upper,
                    "predicted_germination_source_frame": ""
                    if predicted is None
                    else predicted,
                    "verdict": verdict,
                }
            )
    return results


def _germination_summary(rows: list[dict], scope: str) -> dict:
    """Summarize germination interval and no-growth decisions for one scope."""

    selected = [row for row in rows if row["sampling_scope"] == scope]
    intervals = [
        row for row in selected if row["reference_timing_scope"] == "sampled_interval"
    ]
    no_growth = [
        row for row in selected if row["reference_timing_scope"] == "not_observed"
    ]
    return {
        "sampled_interval_cases": len(intervals),
        "predictions_within_interval": sum(
            row["verdict"] == "within_interval" for row in intervals
        ),
        "interval_coverage": sum(
            row["verdict"] == "within_interval" for row in intervals
        )
        / max(len(intervals), 1),
        "no_growth_cases": len(no_growth),
        "correct_no_growth": sum(
            row["verdict"] == "correct_no_growth" for row in no_growth
        ),
        "left_censored_cases": sum(
            row["reference_timing_scope"] == "left_censored" for row in selected
        ),
        "ambiguous_cases": sum(
            row["reference_timing_scope"] == "ambiguous" for row in selected
        ),
    }


def score_reference_kit(kit_dir: Path, prediction_run: Path) -> dict:
    """Score a prediction run only after the blinded references are locked."""

    kit_dir = kit_dir.expanduser().resolve()
    prediction_run = prediction_run.expanduser().resolve()
    if not (kit_dir / "reference_lock.json").exists():
        raise RuntimeError("finalize and lock references before scoring")
    with (kit_dir / "reference_frames.csv").open(newline="") as handle:
        references = list(csv.DictReader(handle))
    summaries, measurements_by_owner = _prediction_rows(prediction_run)
    measurements = {
        (owner, int(row["source_frame"])): row
        for owner, rows in measurements_by_owner.items()
        for row in rows
    }
    predicted_paths: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with (prediction_run / "centerlines.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            predicted_paths[(int(row["pollen_id"]), int(row["source_frame"]))].append(row)
    reference_paths: dict[str, list[dict]] = defaultdict(list)
    with (kit_dir / "reference_centerlines.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            reference_paths[row["record_id"]].append(row)

    scored_rows = []
    definitive = true_positive = true_negative = false_positive = false_negative = 0
    length_errors = []
    relative_length_errors = []
    tip_errors = []
    centerline_means = []
    centerline_p95s = []
    for reference in references:
        owner = int(reference["pollen_id"])
        source_frame = int(reference["source_frame"])
        outcome = reference["reference_outcome"]
        measurement = measurements.get((owner, source_frame))
        status = summaries[owner]["status"]
        predicted_positive = bool(
            measurement
            and status in USABLE_PREDICTION_STATUSES
            and measurement["accepted"] == "1"
            and float(measurement["tube_length_px"]) > 0.0
        )
        reference_positive = outcome in POSITIVE_REFERENCE_OUTCOMES
        classification = (
            "excluded_owner_not_visible"
            if outcome == "owner_not_visible"
            else (
                "excluded_incorrect_target"
                if outcome == "incorrect_target"
                else "excluded_ambiguous"
            )
        )
        if outcome not in {"ambiguous", "owner_not_visible", "incorrect_target"}:
            definitive += 1
            if reference_positive and predicted_positive:
                true_positive += 1
                classification = "true_positive"
            elif reference_positive:
                false_negative += 1
                classification = "false_negative"
            elif predicted_positive:
                false_positive += 1
                classification = "false_positive"
            else:
                true_negative += 1
                classification = "true_negative"

        result = {
            **reference,
            "prediction_status": status,
            "predicted_visible_tube": int(predicted_positive),
            "classification": classification,
            "predicted_length_px": "",
            "length_absolute_error_px": "",
            "length_relative_error": "",
            "tip_error_px": "",
            "centerline_mean_error_px": "",
            "centerline_p95_error_px": "",
        }
        prediction_rows = predicted_paths.get((owner, source_frame), [])
        if reference_positive and predicted_positive and prediction_rows:
            prediction_rows.sort(key=lambda row: int(row["point_index"]))
            reference_rows = sorted(
                reference_paths[reference["record_id"]],
                key=lambda row: int(row["point_index"]),
            )
            prediction_xy = np.asarray(
                [[float(row["source_x_px"]), float(row["source_y_px"])] for row in prediction_rows]
            )
            reference_xy = np.asarray(
                [[float(row["source_x_px"]), float(row["source_y_px"])] for row in reference_rows]
            )
            mean_error, p95_error = _centerline_distances(reference_xy, prediction_xy)
            centerline_means.append(mean_error)
            centerline_p95s.append(p95_error)
            result["centerline_mean_error_px"] = round(mean_error, 6)
            result["centerline_p95_error_px"] = round(p95_error, 6)
            if outcome == "visible_tube":
                reference_length = float(reference["reference_length_px"])
                predicted_length = float(measurement["tube_length_px"])
                absolute_error = abs(predicted_length - reference_length)
                relative_error = absolute_error / max(reference_length, 1e-6)
                tip_error = float(np.linalg.norm(prediction_xy[-1] - reference_xy[-1]))
                length_errors.append(absolute_error)
                relative_length_errors.append(relative_error)
                tip_errors.append(tip_error)
                result["predicted_length_px"] = predicted_length
                result["length_absolute_error_px"] = round(absolute_error, 6)
                result["length_relative_error"] = round(relative_error, 6)
                result["tip_error_px"] = round(tip_error, 6)
        scored_rows.append(result)

    score_path = kit_dir / "frame_scores.csv"
    _write_dict_rows(score_path, scored_rows)
    germination_rows = _germination_window_rows(references, summaries)
    germination_path = kit_dir / "germination_scores.csv"
    _write_dict_rows(germination_path, germination_rows)
    identity_failures = sum(
        row["reference_outcome"] == "incorrect_target" for row in scored_rows
    )
    identity_evaluable = sum(
        row["reference_outcome"] not in {"ambiguous", "owner_not_visible"}
        for row in scored_rows
    )
    report = {
        "schema_version": 1,
        "prediction_run": str(prediction_run),
        "definitive_reference_frames": definitive,
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "visible_tube_accuracy": (true_positive + true_negative) / max(definitive, 1),
        "visible_tube_precision": true_positive / max(true_positive + false_positive, 1),
        "visible_tube_recall": true_positive / max(true_positive + false_negative, 1),
        "owner_identity": {
            "incorrect_target_frames": identity_failures,
            "evaluable_target_frames": identity_evaluable,
            "accuracy": 1.0 - identity_failures / max(identity_evaluable, 1),
        },
        "length_mae_px": float(np.mean(length_errors)) if length_errors else None,
        "length_median_absolute_error_px": float(np.median(length_errors)) if length_errors else None,
        "length_median_relative_error": float(np.median(relative_length_errors))
        if relative_length_errors
        else None,
        "tip_median_error_px": float(np.median(tip_errors)) if tip_errors else None,
        "centerline_mean_error_px": float(np.mean(centerline_means))
        if centerline_means
        else None,
        "centerline_median_p95_error_px": float(np.median(centerline_p95s))
        if centerline_p95s
        else None,
        "by_sampling_stratum": {
            stratum: _scored_subset(
                [row for row in scored_rows if row["stratum"] == stratum]
            )
            for stratum in ("uniform", "challenge")
        },
        "germination": {
            scope: _germination_summary(germination_rows, scope)
            for scope in ("uniform", "uniform_plus_challenge")
        },
        "scored_frame_csv": str(score_path),
        "germination_score_csv": str(germination_path),
    }
    _atomic_json(kit_dir / "score.json", report)
    return report
