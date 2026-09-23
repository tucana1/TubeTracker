"""Reconstruct dynamic pollen-tube centerlines from mature tubes backward.

This prototype uses the v25 mature centerlines as automatic identity seeds. It
then re-traces the complete pollen-connected ribbon in every sampled frame,
starting from the clear final state and moving backward toward germination.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

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
from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds
from tubetracker.causal_portal import (
    PhaseContrastRibbonConfig,
    phase_contrast_ribbon_features,
)
from tubetracker.orientation_worldsheet import (
    OrientationScoreConfig,
    PairedWallOrientationResult,
    paired_wall_orientation_features,
)
from tubetracker.temporal_ribbon import (
    TemporalRibbonConfig,
    TemporalRibbonHistory,
    terminates_at_foreign_owner,
    trace_mature_ribbon_backward,
)


REVISION = "v26.2-bidirectional-root-connected-ribbon"


def parse_args() -> argparse.Namespace:
    """Parse movie, mature-centerline, sampling, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--track-ids",
        help="Optional comma-separated pollen IDs; all mature paths are the default.",
    )
    parser.add_argument("--crop-size", type=int, default=180)
    return parser.parse_args()


def load_mature_centerlines(
    centerlines_csv: Path,
    selected_ids: set[int] | None,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Load mature paths from either summary-style or detailed exports."""

    paths: dict[int, list[tuple[int, float, float]]] = {}
    modes: dict[int, str] = {}
    with centerlines_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    detailed = {"sample_index", "source_y_px", "source_x_px"} <= fields
    if detailed:
        final_sample = {}
        for row in rows:
            track_id = int(row["pollen_id"])
            final_sample[track_id] = max(
                final_sample.get(track_id, -1),
                int(row["sample_index"]),
            )
    else:
        final_sample = {}

    for row in rows:
        track_id = int(row["pollen_id"])
        if selected_ids is not None and track_id not in selected_ids:
            continue
        if row.get("status") in {"not-measured", "rupture-candidate"}:
            continue
        if detailed and int(row["sample_index"]) != final_sample[track_id]:
            continue
        paths.setdefault(track_id, []).append(
            (
                int(row["point_index"]),
                float(
                    row["source_y_px"]
                    if detailed
                    else row["final_source_y_px"]
                ),
                float(
                    row["source_x_px"]
                    if detailed
                    else row["final_source_x_px"]
                ),
            )
        )
        modes[track_id] = row.get("geometry_mode") or "generic"
    ordered = {
        track_id: np.asarray(
            [(y, x) for _, y, x in sorted(points)], dtype=np.float64
        )
        for track_id, points in paths.items()
        if len(points) >= 2
    }
    return ordered, modes


def generic_feature_builder(gray: np.ndarray):
    """Build polarity-agnostic paired-wall evidence for one owner crop."""

    return paired_wall_orientation_features(
        gray,
        OrientationScoreConfig(
            orientation_count=16,
            half_widths_px=(1.0, 1.5, 2.0, 2.5, 3.0),
            tangent_samples_px=(-2.0, -1.0, 0.0, 1.0, 2.0),
            wall_sigma_px=0.6,
            background_sigma_px=3.5,
            structure_sigma_px=1.1,
            orientation_concentration=4.0,
            center_darkness_penalty=0.15,
            asymmetry_penalty=0.25,
        ),
    )


def feature_builder_for_mode(mode: str):
    """Select the mature path's validated phase-contrast polarity."""

    if mode == "generic":
        return generic_feature_builder
    if mode not in {"bright", "dark"}:
        raise ValueError(f"unsupported geometry mode: {mode}")
    config = PhaseContrastRibbonConfig(
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

    def build(gray: np.ndarray):
        """Fuse polarity-specific and generic paired-wall evidence."""

        phase = phase_contrast_ribbon_features(gray, config)
        generic = generic_feature_builder(gray)
        use_phase = phase.paired_score >= generic.paired_score
        return PairedWallOrientationResult(
            score=np.maximum(phase.score, generic.score),
            paired_score=np.maximum(phase.paired_score, generic.paired_score),
            half_width_px=np.where(
                use_phase, phase.half_width_px, generic.half_width_px
            ),
            wall_balance=np.where(
                use_phase, phase.wall_balance, generic.wall_balance
            ),
        )

    return build


def write_outputs(
    output: Path,
    histories: dict[int, TemporalRibbonHistory],
    source_frames: np.ndarray,
    fps: float,
    native_scale: float,
    center_tracks_yx: np.ndarray,
    owner_index: dict[int, int],
    shifts_xy: np.ndarray,
    revision: str = REVISION,
    measurement_statuses: dict[int, str] | None = None,
) -> None:
    """Export dynamic lengths and a centerline for every accepted time point."""

    measurement_statuses = measurement_statuses or {}

    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollen_id",
                "status",
                "first_persistent_source_frame",
                "first_persistent_time_minutes",
                "final_length_px",
                "revision",
            ]
        )
        for track_id, history in sorted(histories.items()):
            onset = history.first_persistent_sample
            writer.writerow(
                [
                    track_id,
                    measurement_statuses.get(
                        track_id,
                        "measured"
                        if history.first_persistent_sample is not None
                        else "unavailable",
                    ),
                    "" if onset is None else int(source_frames[onset]),
                    "" if onset is None else round(source_frames[onset] / fps / 60.0, 4),
                    round(float(history.lengths_px[-1] * native_scale), 4),
                    revision,
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
                "observed_tube_length_px",
                "accepted",
                "paired_fraction",
                "mean_paired_support",
                "prior_error_px",
                "interpolated",
                "temporally_confirmed",
                "measurement_status",
            ]
        )
        for track_id, history in sorted(histories.items()):
            status = measurement_statuses.get(
                track_id,
                "measured"
                if history.first_persistent_sample is not None
                else "unavailable",
            )
            for sample, frame in enumerate(history.frames):
                writer.writerow(
                    [
                        track_id,
                        sample,
                        int(source_frames[sample]),
                        round(source_frames[sample] / fps / 60.0, 4),
                        round(frame.length_px * native_scale, 4),
                        round(
                            (
                                frame.length_px
                                if frame.observed_length_px is None
                                else frame.observed_length_px
                            )
                            * native_scale,
                            4,
                        ),
                        int(frame.accepted),
                        round(frame.paired_fraction, 4),
                        round(frame.mean_paired_support, 4),
                        round(frame.prior_error_px * native_scale, 4),
                        int(frame.interpolated),
                        int(frame.temporally_confirmed),
                        status,
                    ]
                )

    with (output / "centerlines.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pollen_id",
                "sample_index",
                "source_frame",
                "point_index",
                "source_y_px",
                "source_x_px",
                "arc_length_px",
                "measurement_status",
            ]
        )
        for track_id, history in sorted(histories.items()):
            owner = owner_index[track_id]
            status = measurement_statuses.get(
                track_id,
                "measured"
                if history.first_persistent_sample is not None
                else "unavailable",
            )
            for sample, frame in enumerate(history.frames):
                if not frame.accepted:
                    continue
                global_path = (
                    frame.path_relative_yx
                    + center_tracks_yx[owner, sample]
                    + shifts_xy[sample][::-1]
                )
                arc = np.concatenate(
                    (
                        [0.0],
                        np.cumsum(np.linalg.norm(np.diff(global_path, axis=0), axis=1)),
                    )
                )
                for point, (y, x), distance in zip(
                    range(len(global_path)), global_path, arc
                ):
                    writer.writerow(
                        [
                            track_id,
                            sample,
                            int(source_frames[sample]),
                            point,
                            round(y * native_scale, 4),
                            round(x * native_scale, 4),
                            round(distance * native_scale, 4),
                            status,
                        ]
                    )


def render_review(
    movie: Path,
    output: Path,
    source_frames: np.ndarray,
    fps: float,
    native_width: int,
    native_height: int,
    analysis_width: int,
    shifts_xy: np.ndarray,
    center_tracks_yx: np.ndarray,
    owner_ids: list[int],
    histories: dict[int, TemporalRibbonHistory],
    header_label: str = "BIDIRECTIONAL RIBBON",
    output_stem: str = "bidirectional_ribbon",
    measurement_statuses: dict[int, str] | None = None,
) -> Path:
    """Render dynamic centerlines over the complete real microscopy field."""

    measurement_statuses = measurement_statuses or {}
    render_width = 960
    render_height = round(native_height * render_width / native_width)
    header = 50
    render_scale = render_width / analysis_width
    temporary = output / f"{output_stem}_raw.mp4"
    final = output / f"{output_stem}.mov"
    writer = cv.VideoWriter(
        str(temporary),
        cv.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (render_width, render_height + header),
    )
    capture = cv.VideoCapture(str(movie))
    owner_index = {track_id: index for index, track_id in enumerate(owner_ids)}
    for sample, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, source = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode source frame {source_frame}")
        source = cv.resize(source, (render_width, render_height), interpolation=cv.INTER_AREA)
        canvas = np.zeros((render_height + header, render_width, 3), dtype=np.uint8)
        canvas[header:] = source
        drift = shifts_xy[sample][::-1]
        active = 0
        for track_id, history in histories.items():
            owner = owner_index[track_id]
            status = measurement_statuses.get(track_id, "measured")
            center = (center_tracks_yx[owner, sample] + drift) * render_scale
            cv.circle(
                canvas,
                (round(float(center[1])), round(float(center[0] + header))),
                round(6.0 * render_scale),
                (255, 150, 0),
                2,
                cv.LINE_AA,
            )
            if status == "contact_censored":
                status_label = "contact"
            elif status == "boundary_censored":
                status_label = "boundary"
            elif status == "no_growth_detected":
                status_label = "no growth"
            elif status.startswith("review_"):
                status_label = "review"
            else:
                status_label = ""
            cv.putText(
                canvas,
                f"P{track_id:02d} {status_label}".rstrip(),
                (round(float(center[1] + 7)), round(float(center[0] + header - 5))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (20, 75, 20),
                1,
                cv.LINE_AA,
            )
            frame = history.frames[sample]
            if (
                not frame.accepted
                or status.startswith("review_")
                or status in {"no_growth_detected", "unavailable"}
            ):
                continue
            active += 1
            path = (frame.path_relative_yx + center_tracks_yx[owner, sample] + drift)
            points = np.rint(path[:, ::-1] * render_scale).astype(np.int32)
            points[:, 1] += header
            cv.polylines(canvas, [points], False, (255, 0, 220), 3, cv.LINE_AA)
            cv.circle(canvas, tuple(points[-1]), 4, (0, 255, 0), -1, cv.LINE_AA)
        minutes = source_frame / fps / 60.0
        cv.putText(
            canvas,
            f"{header_label} | source {source_frame}/{int(source_frames[-1])} | {minutes:.1f} min",
            (14, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            f"dynamic tubes {active}",
            (render_width - 175, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
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
    """Run automatic backward reconstruction and write review artifacts."""

    args = parse_args()
    selected_ids = (
        None
        if not args.track_ids
        else {int(value) for value in args.track_ids.split(",")}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_frames, fps, _, native_width, native_height = movie_schedule(
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
        include_persistent_late=True,
        minimum_late_semantic_observations=3,
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
    owner_index = {track_id: index for index, track_id in enumerate(owner_ids)}
    mature_paths, geometry_modes = load_mature_centerlines(
        args.base_run / "centerlines.csv", selected_ids
    )
    missing = set(mature_paths) - set(owner_index)
    if missing:
        raise ValueError(f"mature paths have no retained pollen owner: {sorted(missing)}")

    histories = {}
    rejected_mature_paths = {}
    final_drift_yx = shifts_xy[-1][::-1]
    for position, (track_id, source_path) in enumerate(sorted(mature_paths.items()), 1):
        owner = owner_index[track_id]
        source_path_analysis = source_path * analysis_scale
        foreign_centers = np.delete(
            center_tracks[:, -1] + final_drift_yx,
            owner,
            axis=0,
        )
        if terminates_at_foreign_owner(
            source_path_analysis,
            foreign_centers,
            owner_radius,
        ):
            rejected_mature_paths[track_id] = "distal-path-enters-foreign-pollen"
            print(
                f"[v26 ribbon] P{track_id:02d}: rejected mature path ending at another pollen",
                flush=True,
            )
            continue
        final_center_source = (
            center_tracks[owner, -1] + final_drift_yx
        ) / analysis_scale
        mature_relative = (source_path - final_center_source) * analysis_scale
        history = trace_mature_ribbon_backward(
            aligned,
            center_tracks[owner],
            mature_relative,
            feature_builder_for_mode(geometry_modes[track_id]),
            TemporalRibbonConfig(crop_size=args.crop_size),
        )
        histories[track_id] = history
        print(
            f"[v26 ribbon] {position}/{len(mature_paths)} P{track_id:02d}: "
            f"onset={history.first_persistent_sample}, "
            f"final={history.lengths_px[-1] / analysis_scale:.1f}px",
            flush=True,
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
    )
    report = {
        "prototype": "v26_bidirectional_ribbon",
        "revision": REVISION,
        "method": "mature-seeded-backward-whole-curve-deformation",
        "input_movie": str(args.movie),
        "base_run": str(args.base_run),
        "analysis_width": args.width,
        "sample_count": len(source_frames),
        "measured_owner_ids": sorted(
            track_id
            for track_id, history in histories.items()
            if history.first_persistent_sample is not None
        ),
        "unresolved_owner_ids": sorted(
            track_id
            for track_id, history in histories.items()
            if history.first_persistent_sample is None
        ),
        "rejected_mature_paths": rejected_mature_paths,
        "geometry_modes_from_base": geometry_modes,
        "median_registration_response": float(np.median(registration_response)),
        "median_owner_match_score": float(np.median(center_scores)),
        "artifacts": {
            "review_video": str(review),
            "summary_csv": str(args.output / "summary.csv"),
            "measurements_csv": str(args.output / "measurements.csv"),
            "centerlines_csv": str(args.output / "centerlines.csv"),
        },
        "scope": "Research prototype; compare against manual centerlines before quantitative use.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["artifacts"], indent=2))


if __name__ == "__main__":
    main()
