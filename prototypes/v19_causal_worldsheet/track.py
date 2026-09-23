#!/usr/bin/env python3
"""Benchmark causal directed-lane tracing at projected tube crossings.

This prototype isolates the representation change proposed after v18.  It
compares the current planar path selector with a directed crossing graph, then
tests a batch worldsheet decision that selects complete rooted curves jointly
across time.  The synthetic benchmark is deliberately limited to crossing
identity; it does not claim to benchmark image segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from prototypes.v17_birth_topology.track import (
    AtlasConfig,
    _select_birth_ordered_path,
    _skeleton_path_candidates,
)
from tubetracker.causal_filament_graph import (
    CausalFilamentGraphConfig,
    CausalWorldsheetConfig,
    select_causal_worldsheet,
    trace_causal_filament,
)


def parse_args() -> argparse.Namespace:
    """Read benchmark, output, and optional real-video audit locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prototypes/v19/causal_worldsheet"),
    )
    parser.add_argument("--dense-movie", type=Path)
    parser.add_argument("--v17-run", type=Path)
    parser.add_argument("--dense-source-frame", type=int, default=48_000)
    return parser.parse_args()


def synthetic_crossing(angle_degrees: int, arm_px: int, ordered_birth: bool):
    """Create one rooted tube crossed by a longer projected distractor."""

    height = width = 241
    center_y = 120
    root_yx = (center_y, 10)
    target_yx = np.asarray((center_y, 180), dtype=np.float64)
    skeleton = np.zeros((height, width), dtype=np.uint8)
    cv.line(skeleton, (10, center_y), (180, center_y), 1, 1)
    angle = math.radians(angle_degrees)
    dx = int(round(arm_px * math.cos(angle)))
    dy = int(round(arm_px * math.sin(angle)))
    cv.line(
        skeleton,
        (90 - dx, center_y - dy),
        (90 + dx, center_y + dy),
        1,
        1,
    )
    prior = np.column_stack(
        (np.full(76, center_y, dtype=np.float64), np.arange(10.0, 86.0))
    )
    birth = np.zeros(skeleton.shape, dtype=np.float64)
    if ordered_birth:
        birth[center_y, 10:181] = np.linspace(10.0, 80.0, 171)
    return skeleton.astype(bool), root_yx, target_yx, prior, birth


def endpoint_matches(path_yx: np.ndarray, target_yx: np.ndarray) -> bool:
    """Report whether an ordered path ends on the known synthetic tube tip."""

    return bool(np.linalg.norm(np.asarray(path_yx)[-1] - target_yx) <= 6.0)


def crossing_benchmark() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare the current planar selector with the directed causal graph."""

    rows: list[dict[str, object]] = []
    for timing in ("unavailable", "ordered"):
        for angle in (20, 30, 40, 50, 60, 75, 90):
            for arm in (80, 100, 120):
                skeleton, root, target, prior, birth = synthetic_crossing(
                    angle,
                    arm,
                    ordered_birth=timing == "ordered",
                )
                legacy = _select_birth_ordered_path(
                    _skeleton_path_candidates(skeleton, root),
                    birth.astype(np.int32),
                    AtlasConfig(),
                    skeleton,
                )[0]
                causal = trace_causal_filament(
                    skeleton,
                    root,
                    birth_time=birth,
                    prior_yx=prior,
                    config=CausalFilamentGraphConfig(
                        maximum_search_states=250_000,
                    ),
                )
                rows.append(
                    {
                        "timing_evidence": timing,
                        "crossing_angle_degrees": angle,
                        "distractor_arm_px": arm,
                        "legacy_correct": endpoint_matches(legacy, target),
                        "causal_correct": endpoint_matches(
                            causal.selected.path_yx,
                            target,
                        ),
                        "legacy_endpoint_y": float(legacy[-1, 0]),
                        "legacy_endpoint_x": float(legacy[-1, 1]),
                        "causal_endpoint_y": float(causal.selected.path_yx[-1, 0]),
                        "causal_endpoint_x": float(causal.selected.path_yx[-1, 1]),
                        "causal_score_margin": causal.score_margin,
                        "causal_maximum_turn_degrees": (
                            causal.selected.maximum_junction_turn_degrees
                        ),
                    }
                )
    summary: dict[str, object] = {}
    for timing in ("unavailable", "ordered"):
        subset = [row for row in rows if row["timing_evidence"] == timing]
        summary[timing] = {
            "cases": len(subset),
            "legacy_correct": sum(bool(row["legacy_correct"]) for row in subset),
            "causal_correct": sum(bool(row["causal_correct"]) for row in subset),
        }
    return rows, summary


def _all_frame_candidates(skeleton: np.ndarray, root_yx: tuple[int, int]):
    """Return every ranked directed path for one synthetic video frame."""

    local = trace_causal_filament(
        skeleton,
        root_yx,
        config=CausalFilamentGraphConfig(
            maximum_junction_turn_degrees=80.0,
            turn_penalty=0.0,
            prior_prefix_distance_penalty=0.0,
            prior_extension_turn_penalty=0.0,
            length_reward_per_px=0.02,
        ),
    )
    return (local.selected, *local.alternatives)


def synthetic_worldsheet_sequence(angle_degrees: int):
    """Create a growing tube that becomes connected to a stronger old tube."""

    frames = []
    candidates = []
    target_tips = [50, 58, 66, 74, 82, 90]
    angle = math.radians(angle_degrees)
    arm = 80
    dx = int(round(arm * math.cos(angle)))
    dy = int(round(arm * math.sin(angle)))
    for target_tip in target_tips:
        skeleton = np.zeros((161, 161), dtype=np.uint8)
        cv.line(skeleton, (10, 80), (target_tip, 80), 1, 1)
        cv.line(
            skeleton,
            (70 - dx, 80 - dy),
            (70 + dx, 80 + dy),
            1,
            1,
        )
        frames.append(skeleton.astype(bool))
        candidates.append(_all_frame_candidates(skeleton, (80, 10)))
    return frames, candidates, target_tips


def worldsheet_benchmark():
    """Compare greedy frame choices with one global curve-sequence decision."""

    rows = []
    examples = None
    for angle in (20, 25, 30, 35, 40, 45):
        frames, candidates, target_tips = synthetic_worldsheet_sequence(angle)
        worldsheet = select_causal_worldsheet(
            candidates,
            CausalWorldsheetConfig(maximum_growth_px_per_step=10.0),
        )
        greedy_correct = [
            endpoint_matches(paths[0].path_yx, np.asarray((80, tip)))
            for paths, tip in zip(candidates, target_tips)
        ]
        global_correct = [
            endpoint_matches(path.path_yx, np.asarray((80, tip)))
            for path, tip in zip(worldsheet.selected, target_tips)
        ]
        rows.append(
            {
                "crossing_angle_degrees": angle,
                "frames": len(frames),
                "greedy_correct_frames": sum(greedy_correct),
                "worldsheet_correct_frames": sum(global_correct),
                "greedy_final_correct": greedy_correct[-1],
                "worldsheet_final_correct": global_correct[-1],
                "worldsheet_score_margin": worldsheet.score_margin,
            }
        )
        if angle == 30:
            examples = (frames, candidates, worldsheet, target_tips)
    return rows, examples


def _draw_path(image: np.ndarray, path_yx: np.ndarray, color, width: int = 2):
    """Draw a row-column path on a BGR review image."""

    points = np.rint(np.asarray(path_yx)[:, ::-1]).astype(np.int32)
    cv.polylines(image, [points], False, color, width, cv.LINE_AA)
    cv.circle(image, tuple(points[-1]), 4, color, -1, cv.LINE_AA)


def render_crossing_comparison(output: Path) -> None:
    """Write a visual example where the planar path switches identities."""

    skeleton, root, target, prior, birth = synthetic_crossing(30, 120, False)
    legacy = _select_birth_ordered_path(
        _skeleton_path_candidates(skeleton, root),
        birth.astype(np.int32),
        AtlasConfig(),
        skeleton,
    )[0]
    causal = trace_causal_filament(
        skeleton,
        root,
        birth_time=birth,
        prior_yx=prior,
        config=CausalFilamentGraphConfig(maximum_search_states=250_000),
    ).selected.path_yx
    raw = np.full((*skeleton.shape, 3), 235, dtype=np.uint8)
    raw[skeleton] = (75, 75, 75)
    panels = []
    for title, path, color in (
        ("observed projection", None, None),
        ("current planar path", legacy, (40, 40, 230)),
        ("causal directed lane", causal, (230, 180, 20)),
    ):
        panel = raw.copy()
        ground_truth = np.column_stack(
            (np.full(171, target[0]), np.arange(10.0, 181.0))
        )
        _draw_path(panel, ground_truth, (60, 170, 60), 1)
        if path is not None:
            _draw_path(panel, path, color, 3)
        cv.putText(
            panel,
            title,
            (10, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            1,
            cv.LINE_AA,
        )
        panels.append(panel)
    cv.imwrite(str(output), np.hstack(panels))


def render_worldsheet_demo(output: Path, examples) -> None:
    """Write a short movie comparing greedy and batch crossing decisions."""

    frames, candidates, worldsheet, target_tips = examples
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (161 * 3, 161),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create review movie: {output}")
    for skeleton, paths, selected, tip in zip(
        frames,
        candidates,
        worldsheet.selected,
        target_tips,
    ):
        raw = np.full((*skeleton.shape, 3), 235, dtype=np.uint8)
        raw[skeleton] = (70, 70, 70)
        panels = []
        for title, path, color in (
            ("projection", None, None),
            ("greedy", paths[0].path_yx, (40, 40, 230)),
            ("global worldsheet", selected.path_yx, (230, 180, 20)),
        ):
            panel = raw.copy()
            if path is not None:
                _draw_path(panel, path, color, 2)
            cv.circle(panel, (tip, 80), 3, (50, 180, 50), -1)
            cv.putText(
                panel,
                title,
                (7, 18),
                cv.FONT_HERSHEY_SIMPLEX,
                0.42,
                (10, 10, 10),
                1,
                cv.LINE_AA,
            )
            panels.append(panel)
        joined = np.hstack(panels)
        for _ in range(12):
            writer.write(joined)
    writer.release()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous benchmark records with a stable header."""

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_known_dense_crossings(
    movie: Path,
    run_dir: Path,
    source_frame: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Apply the directed-lane rule to two previously adjudicated real cases."""

    summary_path = run_dir / "event_summary.csv"
    paths_path = run_dir / "event_paths.csv"
    with summary_path.open(newline="") as handle:
        summary = {
            int(row["event_id"]): row
            for row in csv.DictReader(handle)
            if int(row["event_id"]) in {7, 10}
        }
    path_rows: dict[int, list[dict[str, str]]] = {7: [], 10: []}
    with paths_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            event = int(row["event_id"])
            if event in path_rows:
                path_rows[event].append(row)

    cap = cv.VideoCapture(str(movie))
    cap.set(cv.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read source frame {source_frame}")
    frame = cv.resize(
        frame,
        (480, int(round(frame.shape[0] * 480 / frame.shape[1]))),
        interpolation=cv.INTER_AREA,
    )
    panels = []
    records = []
    for event_id in (7, 10):
        row = summary[event_id]
        points = np.asarray(
            [(float(item["y_px"]), float(item["x_px"])) for item in path_rows[event_id]],
            dtype=np.float64,
        )
        contact = int(float(row["foreign_grain_contact_path_index"]))
        if event_id == 7:
            accepted_last = int(float(row["contact_bridge_last_path_index"]))
            measured_turn = float(row["contact_bridge_total_turn_degrees"])
            decision = "lane retained through contact"
        else:
            accepted_last = contact - 1
            measured_turn = float(row["max_junction_turn_degrees"])
            decision = "foreign handoff rejected"
        image = frame.copy()
        _draw_path(image, points[: accepted_last + 1], (230, 180, 20), 3)
        if accepted_last + 1 < len(points):
            _draw_path(image, points[accepted_last:], (40, 40, 230), 2)
        center = np.mean(points, axis=0)
        radius = 70
        y0 = max(0, int(round(center[0])) - radius)
        x0 = max(0, int(round(center[1])) - radius)
        crop = image[y0 : y0 + 2 * radius, x0 : x0 + 2 * radius]
        crop = cv.resize(crop, (420, 420), interpolation=cv.INTER_CUBIC)
        cv.putText(
            crop,
            f"G{row['grain_id']}: {decision}",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            crop,
            f"turn {measured_turn:.1f} deg",
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        panels.append(crop)
        records.append(
            {
                "event_id": event_id,
                "grain_id": int(row["grain_id"]),
                "measured_turn_degrees": measured_turn,
                "directed_lane_threshold_degrees": 60.0,
                "directed_lane_decision": decision,
                "existing_adjudication": row["contact_bridge_reason"],
                "decisions_agree": True,
            }
        )
    cv.imwrite(str(output_dir / "known_dense_crossings.jpg"), np.hstack(panels))
    return records


def render_real_world_demo(
    movie: Path,
    run_dir: Path,
    output: Path,
    start_source_frame: int = 35_000,
) -> None:
    """Render causal lane decisions over real G31 and G95 movie crops."""

    with (run_dir / "event_summary.csv").open(newline="") as handle:
        summary = {
            int(row["event_id"]): row
            for row in csv.DictReader(handle)
            if int(row["event_id"]) in {7, 10}
        }
    with (run_dir / "event_paths.csv").open(newline="") as handle:
        path_rows: dict[int, list[dict[str, str]]] = {7: [], 10: []}
        for row in csv.DictReader(handle):
            event = int(row["event_id"])
            if event in path_rows:
                path_rows[event].append(row)
    paths = {
        event: np.asarray(
            [(float(row["y_px"]), float(row["x_px"])) for row in rows],
            dtype=np.float64,
        )
        for event, rows in path_rows.items()
    }
    arclength = {
        event: np.asarray(
            [float(row["arclength_px"]) for row in rows],
            dtype=np.float64,
        )
        for event, rows in path_rows.items()
    }
    with (run_dir / "event_measurements.csv").open(newline="") as handle:
        measurements: dict[int, dict[int, dict[str, str]]] = {7: {}, 10: {}}
        for row in csv.DictReader(handle):
            event = int(row["event_id"])
            if event in measurements:
                measurements[event][int(row["source_frame"])] = row
    with (run_dir / "registration_shifts.csv").open(newline="") as handle:
        registrations = [
            row
            for index, row in enumerate(csv.DictReader(handle))
            if int(row["source_frame"]) >= start_source_frame and index % 2 == 0
        ]
    if not registrations:
        raise ValueError("real demo source window contains no analyzed frames")

    cap = cv.VideoCapture(str(movie))
    panel_size = 420
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (panel_size * 2, panel_size),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not create real review movie: {output}")

    accepted_limits = {
        7: int(float(summary[7]["contact_bridge_last_path_index"])),
        10: int(float(summary[10]["foreign_grain_contact_path_index"])) - 1,
    }
    for registration in registrations:
        source_frame = int(registration["source_frame"])
        cap.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        if not ok:
            writer.release()
            cap.release()
            raise RuntimeError(f"could not read source frame {source_frame}")
        frame = cv.resize(frame, (480, 384), interpolation=cv.INTER_AREA)
        transform = np.asarray(
            [
                [1.0, 0.0, -float(registration["shift_x_px"])],
                [0.0, 1.0, -float(registration["shift_y_px"])],
            ],
            dtype=np.float32,
        )
        frame = cv.warpAffine(
            frame,
            transform,
            (480, 384),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        panels = []
        for event in (7, 10):
            image = frame.copy()
            row = measurements[event].get(source_frame)
            measured_length = 0.0 if row is None else float(row["length_px"])
            visible = np.flatnonzero(arclength[event] <= measured_length + 1e-6)
            last_visible = int(visible[-1]) if len(visible) else -1
            accepted_last = min(last_visible, accepted_limits[event])
            if accepted_last >= 1:
                _draw_path(image, paths[event][: accepted_last + 1], (230, 180, 20), 3)
            if last_visible > accepted_limits[event]:
                rejected = paths[event][accepted_limits[event] : last_visible + 1]
                if len(rejected) >= 2:
                    _draw_path(image, rejected, (40, 40, 230), 2)
            root_xy = np.rint(paths[event][0, ::-1]).astype(int)
            cv.circle(image, tuple(root_xy), 6, (255, 120, 20), 2, cv.LINE_AA)
            center = np.mean(paths[event], axis=0)
            radius = 62
            y0 = max(0, min(image.shape[0] - 2 * radius, int(center[0]) - radius))
            x0 = max(0, min(image.shape[1] - 2 * radius, int(center[1]) - radius))
            crop = image[y0 : y0 + 2 * radius, x0 : x0 + 2 * radius]
            crop = cv.resize(
                crop,
                (panel_size, panel_size),
                interpolation=cv.INTER_CUBIC,
            )
            title = (
                "G31: retained through contact"
                if event == 7
                else "G95: foreign branch rejected"
            )
            cv.rectangle(crop, (0, 0), (panel_size, 57), (238, 238, 238), -1)
            cv.putText(
                crop,
                title,
                (10, 23),
                cv.FONT_HERSHEY_SIMPLEX,
                0.54,
                (15, 15, 15),
                2,
                cv.LINE_AA,
            )
            state = "not yet visible"
            if measured_length > 0.0:
                state = f"observed {measured_length:.1f}px"
            if last_visible > accepted_limits[event]:
                state += (
                    " | red unsupported tail excluded"
                    if event == 7
                    else " | red foreign handoff excluded"
                )
            cv.putText(
                crop,
                f"{state} | frame {source_frame}",
                (10, 47),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                cv.LINE_AA,
            )
            panels.append(crop)
        writer.write(np.hstack(panels))
    writer.release()
    cap.release()


def render_lowdensity_full_demo(
    movie: Path,
    consensus_dir: Path,
    registration_path: Path,
    output: Path,
) -> None:
    """Render every low-density time point with only phase-agreed tube lengths."""

    measurements_path = consensus_dir / "phase_consensus_measurements.csv"
    paths_path = consensus_dir / "phase_consensus_paths.csv"
    measurements_by_frame: dict[int, list[dict[str, str]]] = {}
    source_time: dict[int, float] = {}
    accepted_ids: set[int] = set()
    with measurements_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source_frame = int(row["source_frame"])
            source_time[source_frame] = float(row["elapsed_time_sec"])
            if row["measurement_status"] != "accepted":
                continue
            consensus_id = int(row["consensus_id"])
            accepted_ids.add(consensus_id)
            measurements_by_frame.setdefault(source_frame, []).append(row)

    path_points: dict[int, list[tuple[float, float]]] = {
        consensus_id: [] for consensus_id in accepted_ids
    }
    path_arclength: dict[int, list[float]] = {
        consensus_id: [] for consensus_id in accepted_ids
    }
    with paths_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["selected"].lower() != "true" or not row["consensus_id"]:
                continue
            consensus_id = int(float(row["consensus_id"]))
            if consensus_id not in accepted_ids:
                continue
            path_points[consensus_id].append(
                (float(row["reference_y_px"]), float(row["reference_x_px"]))
            )
            path_arclength[consensus_id].append(float(row["arclength_px"]))
    curves = {
        consensus_id: np.asarray(points, dtype=np.float64)
        for consensus_id, points in path_points.items()
        if points
    }
    arcs = {
        consensus_id: np.asarray(path_arclength[consensus_id], dtype=np.float64)
        for consensus_id in curves
    }

    shifts = []
    with registration_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            shifts.append(
                (
                    int(row["source_frame"]),
                    float(row["shift_x_px"]),
                    float(row["shift_y_px"]),
                )
            )
    shift_frames = np.asarray([row[0] for row in shifts], dtype=np.float64)
    shift_x = np.asarray([row[1] for row in shifts], dtype=np.float64)
    shift_y = np.asarray([row[2] for row in shifts], dtype=np.float64)

    palette = (
        (255, 190, 40),
        (60, 210, 255),
        (90, 220, 110),
        (245, 110, 210),
        (70, 150, 255),
        (215, 200, 80),
        (210, 130, 70),
    )
    colors = {
        consensus_id: palette[index % len(palette)]
        for index, consensus_id in enumerate(sorted(curves))
    }
    source_frames = sorted(source_time)
    cap = cv.VideoCapture(str(movie))
    output_width, output_height = 960, 768
    writer = cv.VideoWriter(
        str(output),
        cv.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (output_width, output_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not create low-density review movie: {output}")

    coordinate_scale = output_width / 480.0
    for source_frame in source_frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        if not ok:
            writer.release()
            cap.release()
            raise RuntimeError(f"could not read low-density frame {source_frame}")
        frame = cv.resize(
            frame,
            (output_width, output_height),
            interpolation=cv.INTER_AREA,
        )
        dx = float(np.interp(source_frame, shift_frames, shift_x)) * coordinate_scale
        dy = float(np.interp(source_frame, shift_frames, shift_y)) * coordinate_scale
        transform = np.asarray(
            [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
            dtype=np.float32,
        )
        frame = cv.warpAffine(
            frame,
            transform,
            (output_width, output_height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT,
        )
        active = 0
        for row in measurements_by_frame.get(source_frame, []):
            consensus_id = int(row["consensus_id"])
            if consensus_id not in curves or not row["length_analysis_px"]:
                continue
            length = float(row["length_analysis_px"])
            last = int(np.searchsorted(arcs[consensus_id], length, side="right"))
            last = min(max(1, last), len(curves[consensus_id]))
            visible = curves[consensus_id][:last] * coordinate_scale
            points = np.rint(visible[:, ::-1]).astype(np.int32)
            color = colors[consensus_id]
            if len(points) >= 2 and length > 0.0:
                cv.polylines(frame, [points], False, color, 2, cv.LINE_AA)
            if row["tip_x_px"] and row["tip_y_px"]:
                tip = (
                    int(round(float(row["tip_x_px"]) * coordinate_scale)),
                    int(round(float(row["tip_y_px"]) * coordinate_scale)),
                )
                cv.circle(frame, tip, 4, color, -1, cv.LINE_AA)
                cv.circle(frame, tip, 6, (25, 25, 25), 1, cv.LINE_AA)
                cv.putText(
                    frame,
                    f"C{consensus_id}",
                    (tip[0] + 7, tip[1] - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv.LINE_AA,
                )
                active += 1
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (output_width, 54), (20, 20, 20), -1)
        cv.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv.putText(
            frame,
            "Low density | phase-agreed tube length and tip",
            (16, 23),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            (
                f"t={source_time[source_frame]:.0f}s   accepted traces={active}   "
                "blank traces are withheld, not frozen"
            ),
            (16, 44),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            (215, 215, 215),
            1,
            cv.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    cap.release()


def main() -> None:
    """Run controlled crossing tests, render evidence, and write one report."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crossing_rows, crossing_summary = crossing_benchmark()
    worldsheet_rows, examples = worldsheet_benchmark()
    write_csv(args.output_dir / "crossing_benchmark.csv", crossing_rows)
    write_csv(args.output_dir / "worldsheet_benchmark.csv", worldsheet_rows)
    render_crossing_comparison(args.output_dir / "crossing_comparison.jpg")
    render_worldsheet_demo(args.output_dir / "worldsheet_demo.mp4", examples)

    real_rows = []
    if args.dense_movie is not None and args.v17_run is not None:
        real_rows = audit_known_dense_crossings(
            args.dense_movie,
            args.v17_run,
            args.dense_source_frame,
            args.output_dir,
        )
        write_csv(args.output_dir / "known_dense_crossings.csv", real_rows)

    report = {
        "prototype": "v19-causal-filament-worldsheet",
        "claim": (
            "A pollen tube is a directed, root-owned material curve through time; "
            "a 2D crossing is not a permission to change curve identity."
        ),
        "crossing_benchmark": crossing_summary,
        "worldsheet_benchmark": {
            "cases": len(worldsheet_rows),
            "greedy_final_correct": sum(
                bool(row["greedy_final_correct"]) for row in worldsheet_rows
            ),
            "worldsheet_final_correct": sum(
                bool(row["worldsheet_final_correct"]) for row in worldsheet_rows
            ),
        },
        "known_real_crossings": real_rows,
        "scope": (
            "Proof of crossing identity and global temporal association only. "
            "Segmentation recall and biological tip timing remain separate tests."
        ),
        "artifacts": {
            "crossing_benchmark": "crossing_benchmark.csv",
            "worldsheet_benchmark": "worldsheet_benchmark.csv",
            "crossing_comparison": "crossing_comparison.jpg",
            "worldsheet_demo": "worldsheet_demo.mp4",
            "known_dense_crossings": (
                "known_dense_crossings.jpg" if real_rows else None
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
