"""Render a concise review spanning an entire owner-memory source movie.

The selected v23 centerlines are the only measured tube geometries. Frames
between those anchors receive an explicitly labelled visual interpolation;
frames before the first accepted path remain unmeasured.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse identity, centerline, sampling, and rendering inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--playback-fps", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_centerlines(path: Path) -> dict[int, np.ndarray]:
    """Load selected aligned centerlines grouped by source frame."""
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
        source_frame: np.asarray(
            [(row[1], row[2]) for row in sorted(rows)], dtype=np.float64
        )
        for source_frame, rows in grouped.items()
    }


def curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength for one row-column curve."""
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path_yx, axis=0), axis=1)))
    )


def resample_curve(path_yx: np.ndarray, count: int) -> np.ndarray:
    """Resample a curve at evenly spaced normalized-arclength positions."""
    arclength = curve_arclength(path_yx)
    if arclength[-1] <= 0:
        return np.repeat(path_yx[:1], count, axis=0)
    positions = np.linspace(0.0, arclength[-1], count)
    return np.column_stack(
        [np.interp(positions, arclength, path_yx[:, axis]) for axis in range(2)]
    )


def interpolate_centerline(
    source_frame: int,
    centerlines: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, str]:
    """Return measured or explicitly interpolated geometry at one timepoint."""
    anchors = np.asarray(sorted(centerlines), dtype=np.int64)
    if len(anchors) == 0 or source_frame < anchors[0]:
        return None, "NO VERIFIED TUBE TRACE"
    if source_frame in centerlines:
        return centerlines[source_frame], "MEASURED V23 TRACE"
    if source_frame >= anchors[-1]:
        return centerlines[int(anchors[-1])], "LAST MEASURED TRACE"
    upper_index = int(np.searchsorted(anchors, source_frame, side="right"))
    lower_frame = int(anchors[upper_index - 1])
    upper_frame = int(anchors[upper_index])
    fraction = (source_frame - lower_frame) / (upper_frame - lower_frame)
    lower = resample_curve(centerlines[lower_frame], 96)
    upper = resample_curve(centerlines[upper_frame], 96)
    return (1.0 - fraction) * lower + fraction * upper, "BETWEEN MEASURED TRACES"


def interpolate_owner_centers(
    source_frames: np.ndarray,
    track: dict,
) -> np.ndarray:
    """Interpolate the linked pollen center across review timepoints."""
    track_frames = np.asarray(track["source_frames"], dtype=np.float64)
    centers = np.asarray(track["centers_yx"], dtype=np.float64)
    return np.column_stack(
        [
            np.interp(source_frames, track_frames, centers[:, axis])
            for axis in range(2)
        ]
    )


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw readable outlined review text."""
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
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


def crop_owner_view(
    frame: np.ndarray,
    center_yx: np.ndarray,
    reference_yx: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create the same owner-aligned crop used by the v23 tracer."""
    height, width = frame.shape[:2]
    shift_yx = reference_yx - center_yx
    transform = np.asarray(
        [[1.0, 0.0, shift_yx[1]], [0.0, 1.0, shift_yx[0]]],
        dtype=np.float32,
    )
    aligned = cv.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT,
    )
    y0, y1, x0, x1 = bounds
    return aligned[y0:y1, x0:x1], shift_yx


def render_timeline(
    canvas: np.ndarray,
    source_frame: int,
    final_frame: int,
    anchor_frames: list[int],
) -> None:
    """Draw recording progress and the six measured anchor positions."""
    left, right, y = 28, canvas.shape[1] - 28, canvas.shape[0] - 19
    cv.line(canvas, (left, y), (right, y), (110, 110, 110), 2, cv.LINE_AA)
    for anchor in anchor_frames:
        x = round(left + (right - left) * anchor / max(1, final_frame))
        cv.line(canvas, (x, y - 5), (x, y + 5), (255, 0, 255), 2, cv.LINE_AA)
    x = round(left + (right - left) * source_frame / max(1, final_frame))
    cv.circle(canvas, (x, y), 5, (0, 230, 255), -1, cv.LINE_AA)


def main() -> None:
    """Render a full-time-span field-and-close-up owner review."""
    args = parse_args()
    identity = json.loads(args.identity_report.read_text())
    trace = json.loads(args.trace_report.read_text())
    track = next(
        item for item in identity["tracks"] if item["track_id"] == args.track_id
    )
    centerline_path = args.trace_report.parent / trace["artifacts"]["centerlines"]
    centerlines = load_centerlines(centerline_path)
    anchor_frames = sorted(centerlines)
    if not anchor_frames:
        raise RuntimeError("the trace report contains no selected centerlines")

    movie = Path(identity["movie"])
    capture = cv.VideoCapture(str(movie))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {movie}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv.CAP_PROP_FPS))
    regular = np.rint(
        np.linspace(0, frame_count - 1, min(args.sample_count, frame_count))
    ).astype(np.int64)
    source_frames = np.unique(np.concatenate((regular, np.asarray(anchor_frames))))
    centers = interpolate_owner_centers(source_frames, track)
    reference_yx = np.median(np.asarray(track["centers_yx"], dtype=np.float64), axis=0)
    bounds = tuple(int(value) for value in trace["crop_bounds_yxyx"])
    owner_local_yx = reference_yx - np.asarray((bounds[0], bounds[2]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas_size = (1280, 580)
    writer = cv.VideoWriter(
        str(args.output),
        cv.VideoWriter_fourcc(*"mp4v"),
        args.playback_fps,
        canvas_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {args.output}")

    schedule_rows = []
    for output_index, (source_frame, center_yx) in enumerate(
        zip(source_frames, centers), start=1
    ):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = capture.read()
        if not ok:
            writer.release()
            capture.release()
            raise RuntimeError(f"could not read source frame {source_frame}")

        owner_crop, shift_yx = crop_owner_view(
            frame, center_yx, reference_yx, bounds
        )
        curve, state = interpolate_centerline(int(source_frame), centerlines)
        measured = int(source_frame) in centerlines
        length = float(curve_arclength(curve)[-1]) if curve is not None else float("nan")

        full = frame.copy()
        center_xy = tuple(np.rint(center_yx[::-1]).astype(int))
        cv.circle(full, center_xy, 18, (255, 150, 0), 3, cv.LINE_AA)
        y0, y1, x0, x1 = bounds
        original_box = np.asarray(
            [x0 - shift_yx[1], y0 - shift_yx[0], x1 - shift_yx[1], y1 - shift_yx[0]]
        )
        cv.rectangle(
            full,
            tuple(np.rint(original_box[:2]).astype(int)),
            tuple(np.rint(original_box[2:]).astype(int)),
            (255, 150, 0),
            2,
            cv.LINE_AA,
        )
        if curve is not None:
            world_yx = curve + np.asarray((bounds[0], bounds[2])) - shift_yx
            world_xy = np.rint(world_yx[:, ::-1]).astype(np.int32)
            cv.polylines(full, [world_xy], False, (255, 0, 255), 3, cv.LINE_AA)
            cv.circle(full, tuple(world_xy[-1]), 6, (0, 220, 80), -1, cv.LINE_AA)

            crop_xy = np.rint(curve[:, ::-1]).astype(np.int32)
            cv.polylines(owner_crop, [crop_xy], False, (255, 0, 255), 3, cv.LINE_AA)
            cv.circle(
                owner_crop,
                tuple(crop_xy[-1]),
                6,
                (0, 220, 80),
                -1,
                cv.LINE_AA,
            )
        cv.circle(
            owner_crop,
            tuple(np.rint(owner_local_yx[::-1]).astype(int)),
            18,
            (255, 150, 0),
            3,
            cv.LINE_AA,
        )

        full_panel = cv.resize(full, (640, 512), interpolation=cv.INTER_AREA)
        zoom_panel = cv.resize(owner_crop, (640, 512), interpolation=cv.INTER_CUBIC)
        canvas = np.full((580, 1280, 3), 20, dtype=np.uint8)
        canvas[48:560, :640] = full_panel
        canvas[48:560, 640:] = zoom_panel
        minutes = source_frame / source_fps / 60.0 if source_fps > 0 else 0.0
        draw_label(
            canvas,
            f"LOW DENSITY | FULL SPAN | source {source_frame}/{frame_count - 1} | {minutes:.1f} min",
            (18, 31),
            0.62,
            (235, 235, 235),
        )
        state_color = (255, 0, 255) if measured else (0, 210, 255)
        state_label = state
        if curve is not None:
            state_label += f" | length {length:.1f} px"
        draw_label(canvas, state_label, (657, 78), 0.55, state_color)
        draw_label(canvas, "COMPLETE FIELD", (18, 78), 0.55, (255, 150, 0))
        render_timeline(
            canvas,
            int(source_frame),
            frame_count - 1,
            anchor_frames,
        )
        writer.write(canvas)
        schedule_rows.append(
            {
                "review_frame": output_index - 1,
                "source_frame": int(source_frame),
                "source_time_minutes": minutes,
                "state": state,
                "length_px": "" if curve is None else length,
            }
        )
        if output_index == 1 or output_index % 40 == 0:
            print(f"[v23 full review] {output_index}/{len(source_frames)}", flush=True)

    writer.release()
    capture.release()
    schedule_path = args.output.with_name(f"{args.output.stem}_schedule.csv")
    with schedule_path.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=schedule_rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(schedule_rows)
    print(
        json.dumps(
            {
                "video": str(args.output),
                "schedule": str(schedule_path),
                "review_frames": len(source_frames),
                "source_frames": frame_count,
                "measured_anchors": anchor_frames,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
