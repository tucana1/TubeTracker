#!/usr/bin/env python3
"""Render owner-following, event-timed audit sheets for a v28 field run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np


STAGE_NAMES = ("baseline", "pre", "onset", "early", "mid", "final")
SPAN_STAGE_NAMES = ("start", "20%", "40%", "60%", "80%", "final")


def parse_args() -> argparse.Namespace:
    """Parse a completed validation directory and visual-audit controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--owner-motion-cache", type=Path, required=True)
    parser.add_argument(
        "--field-cache",
        type=Path,
        help="Override the field cache recorded by the validation run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--header-px", type=int, default=50)
    return parser.parse_args()


def event_timed_samples(
    sample_count: int,
    onset_sample: int | None,
    early_offset: int = 4,
) -> np.ndarray:
    """Choose stages that expose emergence and subsequent identity continuity."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    final = sample_count - 1
    if onset_sample is None or onset_sample <= 0 or onset_sample >= final:
        return np.rint(np.linspace(0, final, len(STAGE_NAMES))).astype(np.int64)
    onset = int(np.clip(onset_sample, 0, final))
    samples = np.asarray(
        (
            0,
            max(0, onset - 1),
            onset,
            min(final, onset + early_offset),
            round((onset + final) / 2),
            final,
        ),
        dtype=np.int64,
    )
    # Very late onsets can collapse adjacent stages. Fill any duplicate slots with
    # unused nearby samples while preserving chronological order.
    if len(np.unique(samples)) != len(samples):
        candidates = np.rint(np.linspace(0, final, len(STAGE_NAMES) * 4)).astype(int)
        chosen = sorted(set(int(value) for value in samples))
        for candidate in candidates:
            if len(chosen) == len(STAGE_NAMES):
                break
            if candidate not in chosen:
                chosen.append(candidate)
                chosen.sort()
        samples = np.asarray(chosen[: len(STAGE_NAMES)], dtype=np.int64)
    return samples


def audit_stage_names(sample_count: int, onset_sample: int | None) -> tuple[str, ...]:
    """Label true event stages without inventing an onset for censored histories."""

    final = sample_count - 1
    if onset_sample is None or onset_sample <= 0 or onset_sample >= final:
        return SPAN_STAGE_NAMES
    return STAGE_NAMES


def centered_crop(
    image: np.ndarray,
    center_xy: np.ndarray,
    size: int,
    fill: int = 18,
) -> np.ndarray:
    """Crop around a moving owner with deterministic padding at field edges."""

    if size < 1:
        raise ValueError("crop size must be positive")
    half = size // 2
    x0 = int(round(float(center_xy[0]))) - half
    y0 = int(round(float(center_xy[1]))) - half
    x1, y1 = x0 + size, y0 + size
    crop = np.full((size, size, 3), fill, dtype=np.uint8)
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return crop
    target_x0, target_y0 = source_x0 - x0, source_y0 - y0
    crop[
        target_y0 : target_y0 + source_y1 - source_y0,
        target_x0 : target_x0 + source_x1 - source_x0,
    ] = image[source_y0:source_y1, source_x0:source_x1]
    return crop


def draw_outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw compact text that remains readable over microscopy imagery."""

    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (10, 10, 10),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv.LINE_AA,
    )


def read_video_frames(path: Path, sample_indices: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Decode selected review frames and return them with playback FPS."""

    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open review video: {path}")
    fps = float(capture.get(cv.CAP_PROP_FPS))
    frames = []
    for sample in sample_indices:
        capture.set(cv.CAP_PROP_POS_FRAMES, int(sample))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode sample {sample} from {path}")
        frames.append(frame)
    capture.release()
    return frames, fps


def status_color(status: str) -> tuple[int, int, int]:
    """Return an audit border color for one aggregate measurement status."""

    if status in {"usable", "measured"}:
        return 60, 205, 80
    if status == "contact_censored":
        return 50, 205, 235
    if status in {"left_censored", "boundary_censored"}:
        return 220, 170, 50
    if status in {"no_growth", "no_growth_detected"}:
        return 220, 170, 40
    if status == "review" or status.startswith("review_"):
        return 50, 130, 245
    return 80, 80, 230


def resolve_field_cache(
    validation_dir: Path,
    validation_manifest: dict,
    override: Path | None,
) -> Path:
    """Resolve shared field context without requiring duplicated cache files."""

    if override is not None:
        return override.expanduser().resolve()
    recorded = validation_manifest.get("field_cache")
    if recorded:
        return Path(recorded).expanduser().resolve()
    return validation_dir / "field_context"


def preferred_source_positions(cache: np.lib.npyio.NpzFile) -> np.ndarray:
    """Load the same highest-fidelity owner path used by the v28 tracer."""

    if "detection_constrained_source_yx" in cache.files:
        key = "detection_constrained_source_yx"
    elif "semantic_span_source_yx" in cache.files:
        key = "semantic_span_source_yx"
    else:
        key = "multipoint_source_yx"
    return np.asarray(cache[key], dtype=np.float64)


def owner_row(
    validation_dir: Path,
    track_id: int,
    status: str,
    positions_yx: np.ndarray,
    source_frames: np.ndarray,
    source_fps: float,
    native_width: int,
    crop_size: int,
    header_px: int,
) -> tuple[np.ndarray, dict]:
    """Render all diagnostic stages for one pollen owner."""

    owner_dir = validation_dir / f"P{track_id:02d}"
    report = json.loads((owner_dir / "report.json").read_text())
    diagnostics = report["worldsheet_diagnostics"][str(track_id)]
    onset = diagnostics.get("first_active_sample")
    if status in {"no_growth", "no_growth_detected", "unavailable"}:
        onset = None
    samples = event_timed_samples(len(source_frames), onset)
    stage_names = audit_stage_names(len(source_frames), onset)
    review_value = report["artifacts"]["review_video"]
    review_path = owner_dir / Path(review_value).name
    frames, _ = read_video_frames(review_path, samples)
    render_scale = frames[0].shape[1] / native_width
    tile_height = crop_size + 54
    row = np.full(
        (tile_height, crop_size * len(samples), 3), 18, dtype=np.uint8
    )
    border = status_color(status)
    for column, (stage, sample, frame) in enumerate(
        zip(stage_names, samples, frames)
    ):
        center_yx = positions_yx[int(sample)] * render_scale
        center_xy = np.asarray(
            (center_yx[1], center_yx[0] + header_px), dtype=np.float64
        )
        crop = centered_crop(frame, center_xy, crop_size)
        x0 = column * crop_size
        row[:crop_size, x0 : x0 + crop_size] = crop
        cv.rectangle(
            row,
            (x0 + 1, 1),
            (x0 + crop_size - 2, crop_size - 2),
            border,
            3,
        )
        source_frame = int(source_frames[int(sample)])
        minutes = source_frame / source_fps / 60.0
        draw_outlined_text(
            row,
            f"{stage} | s{int(sample)} | {minutes:.1f} min",
            (x0 + 8, crop_size + 22),
            0.44,
            (235, 235, 235),
        )
        if column == 0:
            draw_outlined_text(
                row,
                f"P{track_id:02d} | {status}",
                (x0 + 8, crop_size + 46),
                0.55,
                border,
            )
    return row, {
        "track_id": track_id,
        "status": status,
        "onset_sample": onset,
        "sample_indices": [int(value) for value in samples],
        "source_frames": [int(source_frames[int(value)]) for value in samples],
        "review_video": str(review_path),
    }


def main() -> None:
    """Create paged event-timed audit sheets and a machine-readable index."""

    args = parse_args()
    if args.rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")
    validation_dir = args.validation_dir.expanduser().resolve()
    manifest = json.loads((validation_dir / "validation_manifest.json").read_text())
    field_cache = resolve_field_cache(validation_dir, manifest, args.field_cache)
    field_manifest = json.loads(
        (field_cache / "manifest.json").read_text()
    )
    track_ids = [int(value) for value in manifest["completed_track_ids"]]
    field_manifest_path = validation_dir / "field_validation_manifest.json"
    status_source = (
        json.loads(field_manifest_path.read_text())["field_status_by_track_id"]
        if field_manifest_path.exists()
        else manifest["status_by_track_id"]
    )
    status_by_id = {int(key): value for key, value in status_source.items()}
    with np.load(args.owner_motion_cache.expanduser().resolve(), allow_pickle=False) as cache:
        cache_ids = np.asarray(cache["track_ids"], dtype=np.int64)
        source_frames = np.asarray(cache["source_frames"], dtype=np.int64)
        native_width = int(cache["analysis_width"])
        multipoint = preferred_source_positions(cache)
        template = np.asarray(cache["template_source_yx"], dtype=np.float64)
    if len(source_frames) != int(field_manifest["sample_count"]):
        raise ValueError("owner-motion cache and field context sample counts differ")
    source_fps = float(field_manifest["fps"])
    cache_index = {int(track_id): index for index, track_id in enumerate(cache_ids)}

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    rows: list[np.ndarray] = []
    for track_id in track_ids:
        owner_dir = validation_dir / f"P{track_id:02d}"
        report = json.loads((owner_dir / "report.json").read_text())
        positions = (
            template[cache_index[track_id]]
            if track_id in report.get("template_fallback_owner_ids", [])
            else multipoint[cache_index[track_id]]
        )
        row, record = owner_row(
            validation_dir,
            track_id,
            status_by_id[track_id],
            positions,
            source_frames,
            source_fps,
            native_width,
            args.crop_size,
            args.header_px,
        )
        rows.append(row)
        records.append(record)

    page_count = math.ceil(len(rows) / args.rows_per_page)
    page_paths = []
    for page in range(page_count):
        page_rows = rows[page * args.rows_per_page : (page + 1) * args.rows_per_page]
        sheet = np.vstack(page_rows)
        page_path = args.output / f"field_audit_{page + 1:02d}.jpg"
        if not cv.imwrite(str(page_path), sheet, [cv.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"could not write {page_path}")
        page_paths.append(str(page_path))
        for record in records[
            page * args.rows_per_page : (page + 1) * args.rows_per_page
        ]:
            record["page"] = page + 1

    index = {
        "validation_dir": str(validation_dir),
        "owner_motion_cache": str(args.owner_motion_cache.expanduser().resolve()),
        "owner_count": len(records),
        "event_stage_names": list(STAGE_NAMES),
        "span_stage_names": list(SPAN_STAGE_NAMES),
        "pages": page_paths,
        "owners": records,
    }
    index_path = args.output / "field_audit_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"index": str(index_path), "pages": page_paths}, indent=2))


if __name__ == "__main__":
    main()
