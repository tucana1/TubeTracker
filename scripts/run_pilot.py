#!/usr/bin/env python3
"""Run a reproducible TubeTracker germination and elongation pilot."""

import argparse
import contextlib
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2 as cv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tubetracker as tt  # noqa: E402


def parse_args():
    """Parse analysis, calibration, sample, and output command-line options."""
    parser = argparse.ArgumentParser(
        description="Analyze germination and pollen tube tip movement from a video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--genotype", default="unknown")
    parser.add_argument("--biological-replicate", default="unknown")
    parser.add_argument("--imaging-session", default="unknown")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--time-per-frame", type=float, required=True)
    parser.add_argument(
        "--time-unit",
        choices=("sec", "min", "hour", "day"),
        default="sec",
    )
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--distance-unit", default="pxl")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=0,
        help="Last source frame to include; 0 means the final video frame.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every Nth source frame to bound memory use.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--screen-width", type=int, default=1000)
    parser.add_argument("--screen-height", type=int, default=725)
    parser.add_argument("--background-cutoff", type=int, default=55)
    parser.add_argument("--blur-radius", type=int, default=10)
    parser.add_argument("--grain-start", type=int, default=1)
    parser.add_argument("--grain-stop", type=int, default=10)
    parser.add_argument("--min-grain-radius", type=float, default=10)
    parser.add_argument("--max-grain-radius", type=float, default=20)
    parser.add_argument("--grain-threshold", type=int, default=20)
    parser.add_argument(
        "--tip-method",
        choices=("segment-edge", "template-match"),
        default="segment-edge",
    )
    parser.add_argument("--tip-identity", type=float, default=0.80)
    parser.add_argument("--min-tip-side", type=int, default=18)
    parser.add_argument("--gap-closing", type=int, default=10)
    parser.add_argument("--min-points-per-track", type=int, default=10)
    parser.add_argument("--min-overlap", type=float, default=0.10)
    parser.add_argument(
        "--germination-method",
        choices=("tip-overlap", "area-change"),
        default="tip-overlap",
    )
    parser.add_argument("--confirmation-frames", type=int, default=8)
    parser.add_argument("--acceptance-ratio", type=float, default=0.50)
    parser.add_argument(
        "--burst-candidates",
        action="store_true",
        help="Run the experimental reviewer-only rupture candidate detector.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def safe_name(value):
    """Convert user-provided sample text into a filesystem-safe identifier."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "sample"


def git_state():
    """Report the repository revision, branch, and working-tree state."""
    def run(*args):
        """Run a read-only Git command and return its stripped standard output."""
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def software_state():
    """Report exact runtime and repository versions for reproducibility."""
    state = git_state()
    state.update(
        {
            "python": sys.version.split()[0],
            "opencv": cv.__version__,
            "opencv_distribution": version("opencv-contrib-python"),
            "laptrack": version("laptrack"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "tracking_engine": "laptrack",
        }
    )
    return state


def load_video(
    path,
    screen_size,
    max_frames,
    rotate,
    start_frame=0,
    end_frame=0,
    frame_step=1,
):
    """Read a bounded source-frame sequence and resize it for analysis."""
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    metadata = {
        "frame_count_reported": frame_count,
        "playback_fps": float(cap.get(cv.CAP_PROP_FPS)),
        "original_width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "original_height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
    }
    source_end = (
        frame_count - 1 if end_frame <= 0 else min(end_frame, frame_count - 1)
    )
    source_indices = []
    frames = []
    source_frame = start_frame
    while source_frame <= source_end and (max_frames <= 0 or len(frames) < max_frames):
        cap.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        if not ok:
            break
        if rotate:
            frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(cv.resize(frame, screen_size))
        source_indices.append(source_frame)
        source_frame += frame_step
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    metadata["frames_analyzed"] = len(frames)
    metadata["source_frame_indices"] = source_indices
    metadata["source_start_frame"] = source_indices[0]
    metadata["source_end_frame"] = source_indices[-1]
    metadata["source_frame_step"] = frame_step
    return frames, metadata


def run_stage(label, callback, log_file, verbose):
    """Run one pipeline stage with either live or file-captured output."""
    print(label, flush=True)
    if verbose:
        return callback()
    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        return callback()


def sample_fields(args, sample_id):
    """Return sample metadata columns shared by exported result tables."""
    return [
        sample_id,
        args.genotype,
        args.biological_replicate,
        args.imaging_session,
    ]


def source_frame_for(source_frames, analysis_frame):
    """Map an analysis-frame index back to its original video frame."""
    if analysis_frame < 0 or analysis_frame >= len(source_frames):
        return -1
    return source_frames[int(analysis_frame)]


def write_grain_summary(path, tracker, args, sample_id, source_frames):
    """Write one QC-aware germination and burst row per detected grain."""
    header = [
        "sample_id",
        "genotype",
        "biological_replicate",
        "imaging_session",
        "grain_id",
        "detection_analysis_frame",
        "detection_source_frame",
        f"detection_time_{args.time_unit}",
        "germinated",
        "germination_analysis_frame",
        "germination_source_frame",
        f"germination_time_{args.time_unit}",
        "germination_p_value",
        "detection_method",
        "germination_method",
        "burst_candidate",
        "burst_candidate_analysis_frame",
        "burst_candidate_source_frame",
        "burst_candidate_confidence",
        "associated_track_id",
        "qc_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for grain in tracker.valid_grains:
            detection_source_frame = source_frame_for(
                source_frames, grain.first_frame()
            )
            germination_source_frame = source_frame_for(
                source_frames, grain.ger_frame
            )
            burst_source_frame = source_frame_for(
                source_frames, grain.burst_candidate_frame
            )
            status = "pass"
            if not grain.is_germinated:
                status = "review_not_germinated"
            elif grain.is_burst_candidate:
                status = "review_burst_candidate"
            writer.writerow(
                sample_fields(args, sample_id)
                + [
                    grain.id,
                    grain.first_frame(),
                    detection_source_frame,
                    detection_source_frame * args.time_per_frame,
                    grain.is_germinated,
                    grain.ger_frame,
                    germination_source_frame,
                    germination_source_frame * args.time_per_frame
                    if germination_source_frame >= 0
                    else -1,
                    grain.ger_p_value,
                    grain.detection_method,
                    grain.germination_method,
                    grain.is_burst_candidate,
                    grain.burst_candidate_frame,
                    burst_source_frame,
                    grain.burst_candidate_confidence,
                    grain.burst_candidate_track_id,
                    status,
                ]
            )


def write_track_details(path, tracker, args, sample_id, source_frames):
    """Write every retained tip point and its trajectory-based measurements."""
    rows = tracker.coordinate_rows()
    core_header = rows[0]
    header = [
        "sample_id",
        "genotype",
        "biological_replicate",
        "imaging_session",
    ] + core_header[:2] + [
        "analysis_frame",
        "source_frame",
        f"time_{args.time_unit}",
    ] + core_header[4:]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows[1:]:
            source_frame = source_frame_for(source_frames, row[2])
            current_time = source_frame * args.time_per_frame
            writer.writerow(
                sample_fields(args, sample_id)
                + row[:3]
                + [source_frame, current_time]
                + row[4:]
            )


def write_track_summary(path, tracker, args, sample_id, source_frames):
    """Write one final cumulative-movement summary row per accepted track."""
    rows = tracker.track_summary_rows()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "genotype",
                "biological_replicate",
                "imaging_session",
            ]
            + [
                "grain_id",
                "track_id",
                "start_analysis_frame",
                "start_source_frame",
                "end_analysis_frame",
                "end_source_frame",
                f"start_time_{args.time_unit}",
                f"end_time_{args.time_unit}",
                f"duration_{args.time_unit}",
            ]
            + rows[0][7:]
        )
        for row in rows[1:]:
            start_source_frame = source_frame_for(source_frames, row[2])
            end_source_frame = source_frame_for(source_frames, row[3])
            start_time = start_source_frame * args.time_per_frame
            end_time = end_source_frame * args.time_per_frame
            writer.writerow(
                sample_fields(args, sample_id)
                + [
                    row[0],
                    row[1],
                    row[2],
                    start_source_frame,
                    row[3],
                    end_source_frame,
                    start_time,
                    end_time,
                    end_time - start_time,
                ]
                + row[7:]
            )
    return rows


def validate_args(args):
    """Reject inconsistent or physically invalid analysis parameters."""
    if args.time_per_frame <= 0:
        raise SystemExit("--time-per-frame must be greater than zero")
    if args.pixel_size <= 0:
        raise SystemExit("--pixel-size must be greater than zero")
    if args.start_frame < 0:
        raise SystemExit("--start-frame cannot be negative")
    if args.end_frame > 0 and args.end_frame < args.start_frame:
        raise SystemExit("--end-frame must be at or after --start-frame")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be greater than zero")
    if not 0 < args.acceptance_ratio <= 1:
        raise SystemExit("--acceptance-ratio must be in (0, 1]")
    if not 0 < args.min_overlap <= 1:
        raise SystemExit("--min-overlap must be in (0, 1]")


def main():
    """Execute a reproducible command-line TubeTracker pilot analysis."""
    args = parse_args()
    validate_args(args)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    sample_id = safe_name(args.sample_id or video.stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / sample_id / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "running",
        "created_utc": timestamp,
        "input_video": str(video),
        "sample_id": sample_id,
        "parameters": vars(args).copy(),
        "software": software_state(),
    }
    manifest["parameters"]["video"] = str(video)
    manifest["parameters"]["output_dir"] = str(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    try:
        screen_size = (args.screen_width, args.screen_height)
        frames, video_metadata = load_video(
            video,
            screen_size,
            args.max_frames,
            args.rotate,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
        )
        source_width = video_metadata["original_width"]
        source_height = video_metadata["original_height"]
        if args.rotate:
            source_width, source_height = source_height, source_width

        tracker = tt.Tracker(screen_size=screen_size)
        tracker.file_names = [f"frame_{i:06d}.png" for i in range(len(frames))]
        tracker.img_rp = tt.Point(
            x=screen_size[0] / source_width,
            y=screen_size[1] / source_height,
        )
        tracker.img_ratio = (tracker.img_rp.x + tracker.img_rp.y) / 2
        tracker.pxl_dis = args.pixel_size
        tracker.dis_unit = args.distance_unit
        tracker.time_p_frame = args.time_per_frame * args.frame_step
        tracker.time_unit = args.time_unit
        tracker.bg_threshold = args.background_cutoff
        tracker.filter_radius = args.blur_radius
        tracker.grain_det_start = max(0, args.grain_start - 1)
        tracker.grain_det_stop = max(0, args.grain_stop - 1)
        tracker.min_grain_radius = max(1, int(args.min_grain_radius))
        tracker.max_grain_radius = max(
            tracker.min_grain_radius + 1,
            int(args.max_grain_radius),
        )
        tracker.grain_tresh = args.grain_threshold
        tracker.tip_det_threshold_percent = args.tip_identity
        tracker.min_tip_side = args.min_tip_side
        tracker.tip_gap_closing = args.gap_closing
        tracker.min_tip_per_trk = args.min_points_per_track
        tracker.tip_max_step = args.min_overlap
        tracker.ger_confirm_frames = args.confirmation_frames
        tracker.aceptance_ratio = args.acceptance_ratio
        tracker.enable_burst_candidates = args.burst_candidates

        log_path = output_dir / "analysis.log"
        with log_path.open("w") as log_file:
            run_stage(
                "Segmenting frames",
                lambda: setattr(
                    tracker,
                    "all_detections",
                    tt.Detections(
                        img_list=frames,
                        bg_threshold=tracker.bg_threshold,
                        blur_radius=tracker.filter_radius,
                    ),
                ),
                log_file,
                args.verbose,
            )
            run_stage("Detecting grains", tracker.find_grains, log_file, args.verbose)
            tip_callback = (
                tracker.find_tips_se
                if args.tip_method == "segment-edge"
                else tracker.find_tips_tm
            )
            run_stage("Detecting tips", tip_callback, log_file, args.verbose)
            run_stage(
                "Tracking tube elongation",
                tracker.track_elongation,
                log_file,
                args.verbose,
            )
            if tracker.valid_grains:
                germination_callback = (
                    tracker.track_germination_via_tips
                    if args.germination_method == "tip-overlap"
                    else tracker.track_germination_via_area
                )
                run_stage(
                    "Tracking germination",
                    germination_callback,
                    log_file,
                    args.verbose,
                )
            if args.burst_candidates:
                run_stage(
                    "Finding burst candidates",
                    tracker.find_burst_candidates,
                    log_file,
                    args.verbose,
                )
            prefix = str(output_dir / f"{sample_id}.")
            run_stage(
                "Writing TubeTracker outputs",
                lambda: tracker.save_results(prefix),
                log_file,
                args.verbose,
            )

        write_grain_summary(
            output_dir / "pilot.grains.csv",
            tracker,
            args,
            sample_id,
            video_metadata["source_frame_indices"],
        )
        write_track_details(
            output_dir / "pilot.track_details.csv",
            tracker,
            args,
            sample_id,
            video_metadata["source_frame_indices"],
        )
        track_summary_rows = write_track_summary(
            output_dir / "pilot.track_summary.csv",
            tracker,
            args,
            sample_id,
            video_metadata["source_frame_indices"],
        )
        tip_count = sum(len(tips) for tips in tracker.valid_tips)
        germinated_count = sum(
            1 for grain in tracker.valid_grains if grain.is_germinated
        )
        trajectory_qc_counts = {}
        image_qc_counts = {}
        trajectory_qc_index = track_summary_rows[0].index(
            "trajectory_length_qc_status"
        )
        image_qc_index = track_summary_rows[0].index(
            "final_image_centerline_qc_status"
        )
        for row in track_summary_rows[1:]:
            trajectory_status = row[trajectory_qc_index]
            image_status = row[image_qc_index]
            trajectory_qc_counts[trajectory_status] = (
                trajectory_qc_counts.get(trajectory_status, 0) + 1
            )
            image_qc_counts[image_status] = image_qc_counts.get(image_status, 0) + 1
        summary = {
            "sample_id": sample_id,
            "frames_analyzed": len(frames),
            "grain_count": len(tracker.valid_grains),
            "germinated_count": germinated_count,
            "germinated_fraction": (
                germinated_count / len(tracker.valid_grains)
                if tracker.valid_grains
                else None
            ),
            "tip_detection_count": tip_count,
            "track_count": len(tracker.valid_tracks),
            "tracking_engine": tracker.tracking_engine,
            "burst_candidate_count": len(tracker.burst_candidates),
            "clean_trajectory_length_count": trajectory_qc_counts.get("ok", 0),
            "trajectory_length_qc": trajectory_qc_counts,
            "clean_image_centerline_count": image_qc_counts.get("ok", 0),
            "image_centerline_qc": image_qc_counts,
            "review_required": {
                "not_germinated": sum(
                    1 for grain in tracker.valid_grains if not grain.is_germinated
                ),
                "burst_candidates": len(tracker.burst_candidates),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        manifest.update(
            {"status": "complete", "video_metadata": video_metadata, "summary": summary}
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Pilot outputs: {output_dir}")
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        raise


if __name__ == "__main__":
    main()
