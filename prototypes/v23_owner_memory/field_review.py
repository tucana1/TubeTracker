"""Render a full-span, all-owner review from a completed v23 field batch."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import cv2 as cv
import numpy as np

from full_span_review import interpolate_centerline, load_centerlines


@dataclass(frozen=True)
class FieldOwner:
    """One persistent pollen owner and its optional accepted tube lineage."""

    track_id: int
    track: dict
    trace: dict | None
    centerlines: dict[int, np.ndarray]
    reference_yx: np.ndarray
    temporal_centerlines: dict[int, np.ndarray]
    temporal_accepted: bool
    temporal_review: bool
    use_temporal: bool


def parse_args() -> argparse.Namespace:
    """Parse field batch, temporal sampling, and output controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--consistency-report", type=Path)
    parser.add_argument(
        "--temporal-dir",
        type=Path,
        help="Optional dense germination-to-growth output from temporal_front.py.",
    )
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--playback-fps", type=float, default=8.0)
    parser.add_argument(
        "--show-review",
        action="store_true",
        help="Draw review-only tube hypotheses in orange for diagnostics.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def owner_center(track: dict, source_frame: int | np.ndarray) -> np.ndarray:
    """Interpolate a pollen owner at one or more source frames."""
    query = np.asarray(source_frame, dtype=np.float64)
    track_frames = np.asarray(track["source_frames"], dtype=np.float64)
    centers = np.asarray(track["centers_yx"], dtype=np.float64)
    result = np.stack(
        [np.interp(query, track_frames, centers[:, axis]) for axis in range(2)],
        axis=-1,
    )
    return result


def load_temporal_results(
    temporal_dir: Path,
) -> tuple[
    dict[int, dict[int, np.ndarray]],
    set[int],
    set[int],
    list[int],
    list[int],
]:
    """Load dense source-coordinate centerlines and temporal event metadata."""
    report = json.loads((temporal_dir / "temporal_report.json").read_text())
    accepted = {
        int(owner["owner_track_id"])
        for owner in report["owners"]
        if owner["trajectory_accepted"]
    }
    review = {
        int(owner["owner_track_id"])
        for owner in report["owners"]
        if not owner["trajectory_accepted"]
        and owner.get("front_feasible", False)
        and owner.get("germination_source_frame") is not None
    }
    germinations = sorted(
        int(owner["germination_source_frame"])
        for owner in report["owners"]
        if (owner["trajectory_accepted"] or owner["owner_track_id"] in review)
        and owner.get("germination_source_frame") is not None
    )
    frames = []
    with (temporal_dir / "temporal_measurements.csv").open(newline="") as handle:
        frames = sorted({int(row["source_frame"]) for row in csv.DictReader(handle)})
    grouped: dict[int, dict[int, list[tuple[int, float, float]]]] = {}
    with (temporal_dir / "temporal_centerlines.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            owner_id = int(row["owner_track_id"])
            source_frame = int(row["source_frame"])
            grouped.setdefault(owner_id, {}).setdefault(source_frame, []).append(
                (
                    int(row["point_index"]),
                    float(row["y_source_px"]),
                    float(row["x_source_px"]),
                )
            )
    curves = {
        owner_id: {
            frame: np.asarray(
                [(y, x) for _, y, x in sorted(points)], dtype=np.float64
            )
            for frame, points in owner_frames.items()
        }
        for owner_id, owner_frames in grouped.items()
    }
    return curves, accepted, review, frames, germinations


def load_field_owners(
    identity: dict,
    batch_report: Path,
    temporal_curves: dict[int, dict[int, np.ndarray]] | None = None,
    temporal_accepted_ids: set[int] | None = None,
    temporal_review_ids: set[int] | None = None,
) -> list[FieldOwner]:
    """Combine all origin pollen identities with accepted batch traces."""
    batch = json.loads(batch_report.read_text())
    batch_by_owner = {
        int(item["owner_track_id"]): item for item in batch.get("owners", [])
    }
    origin = int(identity["source_frames"][0])
    use_temporal = temporal_curves is not None
    owners = []
    for track in identity["tracks"]:
        if (
            int(track["source_frames"][0]) != origin
            or int(track["semantic_observation_count"]) < 2
        ):
            continue
        track_id = int(track["track_id"])
        entry = batch_by_owner.get(track_id)
        trace = None
        centerlines: dict[int, np.ndarray] = {}
        if entry is not None:
            trace_path = batch_report.parent / entry["directory"] / "report.json"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                if trace["accepted"]:
                    centerline_path = trace_path.parent / trace["artifacts"]["centerlines"]
                    centerlines = load_centerlines(centerline_path)
        identity_centers = owner_center(
            track, np.asarray(identity["source_frames"], dtype=np.float64)
        )
        owners.append(
            FieldOwner(
                track_id=track_id,
                track=track,
                trace=trace,
                centerlines=centerlines,
                reference_yx=np.median(identity_centers, axis=0),
                temporal_centerlines=(temporal_curves or {}).get(track_id, {}),
                temporal_accepted=track_id in (temporal_accepted_ids or set()),
                temporal_review=track_id in (temporal_review_ids or set()),
                use_temporal=use_temporal,
            )
        )
    return sorted(owners, key=lambda item: item.track_id)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw compact, legible field-review text."""
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (15, 15, 15),
        thickness + 3,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv.LINE_AA,
    )


def draw_owner(
    frame: np.ndarray,
    owner: FieldOwner,
    source_frame: int,
    excluded_owner_ids: set[int] | None = None,
    show_review: bool = False,
) -> bool:
    """Draw one pollen identity and its active accepted tube, if present."""
    center_yx = owner_center(owner.track, source_frame)
    center_xy = tuple(np.rint(center_yx[::-1]).astype(np.int32))
    cv.circle(frame, center_xy, 15, (255, 150, 0), 2, cv.LINE_AA)
    draw_text(
        frame,
        f"P{owner.track_id:02d}",
        (center_xy[0] + 11, center_xy[1] - 10),
        0.38,
        (255, 210, 40),
    )
    if owner.use_temporal:
        if not owner.temporal_accepted and not (
            show_review and owner.temporal_review
        ):
            return False
    elif (
        not owner.centerlines
        or owner.track_id in (excluded_owner_ids or set())
    ):
        return False
    world_yx = owner_world_curve(owner, source_frame)
    if world_yx is None:
        return False
    inside = (
        (world_yx[:, 0] >= 0)
        & (world_yx[:, 0] < frame.shape[0])
        & (world_yx[:, 1] >= 0)
        & (world_yx[:, 1] < frame.shape[1])
    )
    if np.count_nonzero(inside) < 2:
        return False
    world_xy = np.rint(world_yx[inside, ::-1]).astype(np.int32)
    path_color = (
        (255, 0, 255)
        if not owner.use_temporal or owner.temporal_accepted
        else (0, 165, 255)
    )
    tip_color = (
        (0, 235, 80)
        if not owner.use_temporal or owner.temporal_accepted
        else (0, 190, 255)
    )
    cv.polylines(frame, [world_xy], False, path_color, 3, cv.LINE_AA)
    cv.circle(frame, tuple(world_xy[-1]), 5, tip_color, -1, cv.LINE_AA)
    return True


def owner_world_curve(
    owner: FieldOwner,
    source_frame: int,
) -> np.ndarray | None:
    """Transform one accepted aligned centerline back into source pixels."""
    if owner.use_temporal:
        return owner.temporal_centerlines.get(source_frame)
    if not owner.centerlines or owner.trace is None:
        return None
    curve, _ = interpolate_centerline(source_frame, owner.centerlines)
    if curve is None:
        return None
    center_yx = owner_center(owner.track, source_frame)
    y0, _, x0, _ = (int(value) for value in owner.trace["crop_bounds_yxyx"])
    shift_yx = owner.reference_yx - center_yx
    return curve + np.asarray((y0, x0), dtype=np.float64) - shift_yx


def render_progress(
    canvas: np.ndarray,
    source_frame: int,
    final_frame: int,
    anchor_frames: list[int],
) -> None:
    """Render recording progress and common measured checkpoints."""
    left, right, y = 22, canvas.shape[1] - 22, canvas.shape[0] - 16
    cv.line(canvas, (left, y), (right, y), (105, 105, 105), 2, cv.LINE_AA)
    for anchor in anchor_frames:
        x = round(left + (right - left) * anchor / max(1, final_frame))
        cv.line(canvas, (x, y - 4), (x, y + 4), (255, 0, 255), 1, cv.LINE_AA)
    current_x = round(left + (right - left) * source_frame / max(1, final_frame))
    cv.circle(canvas, (current_x, y), 5, (0, 225, 255), -1, cv.LINE_AA)


def encode_quicktime(source: Path, output: Path) -> None:
    """Encode the rendered review as QuickTime-compatible H.264."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
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
            "-an",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    """Render all pollen owners over the complete low-density time span."""
    args = parse_args()
    identity = json.loads(args.identity_report.read_text())
    temporal_curves = None
    temporal_accepted_ids: set[int] | None = None
    temporal_review_ids: set[int] | None = None
    temporal_frames: list[int] = []
    temporal_events: list[int] = []
    if args.temporal_dir:
        (
            temporal_curves,
            temporal_accepted_ids,
            temporal_review_ids,
            temporal_frames,
            temporal_events,
        ) = load_temporal_results(args.temporal_dir)
    owners = load_field_owners(
        identity,
        args.batch_report,
        temporal_curves,
        temporal_accepted_ids,
        temporal_review_ids,
    )
    excluded_owner_ids: set[int] = set()
    if args.consistency_report:
        consistency = json.loads(args.consistency_report.read_text())
        excluded_owner_ids.update(consistency.get("duplicate_claim_owner_ids", []))
        excluded_owner_ids.update(consistency.get("foreign_endpoint_owner_ids", []))
    accepted_owners = [
        owner
        for owner in owners
        if (
            owner.temporal_accepted
            if args.temporal_dir
            else owner.centerlines and owner.track_id not in excluded_owner_ids
        )
    ]
    review_owners = [
        owner
        for owner in owners
        if args.temporal_dir and owner.temporal_review
    ]
    anchor_frames = (
        temporal_events
        if args.temporal_dir
        else sorted({frame for owner in accepted_owners for frame in owner.centerlines})
    )

    capture = cv.VideoCapture(identity["movie"])
    if not capture.isOpened():
        raise RuntimeError(f"could not open {identity['movie']}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv.CAP_PROP_FPS))
    if args.temporal_dir:
        source_frames = np.asarray(temporal_frames, dtype=np.int64)
    else:
        regular = np.rint(
            np.linspace(0, frame_count - 1, min(args.sample_count, frame_count))
        ).astype(np.int64)
        source_frames = np.unique(
            np.concatenate((regular, np.asarray(anchor_frames)))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    working = args.output.with_name(f".{args.output.stem}.working.mp4")
    canvas_size = (1280, 1120)
    writer = cv.VideoWriter(
        str(working),
        cv.VideoWriter_fourcc(*"mp4v"),
        args.playback_fps,
        canvas_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {working}")

    rows = []
    for review_index, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            writer.release()
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        active_count = sum(
            draw_owner(
                frame,
                owner,
                int(source_frame),
                excluded_owner_ids,
                args.show_review,
            )
            for owner in owners
        )
        canvas = np.full((canvas_size[1], canvas_size[0], 3), 18, dtype=np.uint8)
        canvas[58:1082] = frame
        minutes = source_frame / source_fps / 60.0 if source_fps > 0 else 0.0
        draw_text(
            canvas,
            f"LOW DENSITY | ALL {len(owners)} POLLEN | source {source_frame}/{frame_count - 1} | {minutes:.1f} min",
            (18, 37),
            0.72,
            (242, 242, 242),
            2,
        )
        draw_text(
            canvas,
            (
                f"growth-tracked {len(accepted_owners)} | "
                + (
                    f"review shown {len(review_owners)} | "
                    if args.show_review
                    else ""
                )
                + f"active traces {active_count}"
                if args.temporal_dir
                else f"accepted tube owners {len(accepted_owners)} | active traces {active_count}"
            ),
            (855, 37),
            0.48,
            (255, 170, 255),
        )
        render_progress(
            canvas,
            int(source_frame),
            frame_count - 1,
            anchor_frames,
        )
        writer.write(canvas)
        rows.append(
            {
                "review_frame": review_index,
                "source_frame": int(source_frame),
                "source_time_minutes": minutes,
                "pollen_owner_count": len(owners),
                "accepted_tube_owner_count": len(accepted_owners),
                "review_tube_owner_count": len(review_owners),
                "field_conflict_owner_count": len(excluded_owner_ids),
                "active_trace_count": active_count,
            }
        )
        if review_index == 0 or (review_index + 1) % 40 == 0:
            print(
                f"[v23 field review] {review_index + 1}/{len(source_frames)}",
                flush=True,
            )
    writer.release()
    capture.release()
    encode_quicktime(working, args.output)
    working.unlink()

    schedule = args.output.with_name(f"{args.output.stem}_schedule.csv")
    with schedule.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    print(
        json.dumps(
            {
                "video": str(args.output),
                "schedule": str(schedule),
                "pollen_owners": len(owners),
                "accepted_tube_owners": len(accepted_owners),
                "review_tube_owners": len(review_owners),
                "field_conflict_owner_ids": sorted(excluded_owner_ids),
                "review_frames": len(source_frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
