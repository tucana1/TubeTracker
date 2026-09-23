#!/usr/bin/env python3
"""Run resumable, per-owner v27 validation across a complete movie field."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototypes.v27_global_ribbon_worldsheet.track import (
    GLOBAL_ALLOCATION_REVISION,
    prepare_field_context,
)
from tubetracker.field_arbitration import arbitrate_validation_directory
from tubetracker.temporal_ribbon import TemporalRibbonConfig


EXPECTED_REVISION = "v27.6-event-and-boundary-status"
MULTIPOINT_EXPECTED_REVISION = "v28.5-monotone-material-history"
RIM_GEOMETRY_REVISION = "v28.1-rim-geometry-owner-motion"
LEGACY_MULTIPOINT_REVISION = "v28.0-cotracker-rigid-owner-motion"
SEMANTIC_GUARDED_REVISION = "v28.3.2-semantic-guarded-joint-detection-motion"
BASELINE_LIFECYCLE_REVISION = "v28.4-baseline-departure-lifecycle"


def parse_args() -> argparse.Namespace:
    """Parse movie, identity, concurrency, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--causal-atlas-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--track-ids",
        required=True,
        help="Comma-separated retained pollen IDs to validate.",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--owner-motion-cache",
        type=Path,
        help="Optional v28 native-detail pollen-motion cache.",
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--field-cache",
        type=Path,
        help="Shared frame context; defaults inside the output directory.",
    )
    parser.add_argument("--selection-stride", type=int, default=4)
    parser.add_argument("--alternatives", type=int, default=6)
    parser.add_argument("--mature-hypotheses", type=int, default=7)
    parser.add_argument(
        "--export-alternatives",
        action="store_true",
        help="Save all sparse mature-path histories for global branch allocation.",
    )
    parser.add_argument(
        "--mature-seed-overrides",
        type=Path,
        help="JSON object mapping pollen IDs to globally allocated mature-seed labels.",
    )
    return parser.parse_args()


def parse_track_ids(value: str) -> list[int]:
    """Return unique positive pollen IDs in user-specified order."""

    track_ids = list(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    if not track_ids or any(track_id < 1 for track_id in track_ids):
        raise ValueError("track IDs must be positive integers")
    return track_ids


def load_mature_seed_overrides(path: Path | None) -> dict[int, str]:
    """Load and validate optional field-level mature-seed assignments."""

    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if not isinstance(payload, dict):
        raise ValueError("mature-seed overrides must be a JSON object")
    overrides = {}
    for raw_owner, raw_label in payload.items():
        try:
            owner_id = int(raw_owner)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid override owner ID: {raw_owner!r}") from exc
        if owner_id < 1 or not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"invalid mature-seed override for P{owner_id:02d}")
        overrides[owner_id] = raw_label.strip()
    return overrides


def completed_report(
    output_dir: Path,
    track_id: int,
    expected_revision: str = EXPECTED_REVISION,
) -> dict | None:
    """Load a valid completed single-owner report, if one exists."""

    report_path = output_dir / "report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    reported_ids = {
        int(owner_id)
        for key in (
            "measured_owner_ids",
            "review_owner_ids",
            "unavailable_owner_ids",
            "no_growth_owner_ids",
        )
        for owner_id in report.get(key, [])
    }
    if reported_ids != {track_id}:
        return None
    if report.get("revision") != expected_revision:
        if not compatible_multipoint_report(
            report,
            expected_revision,
            output_dir=output_dir,
        ):
            return None
        promote_report_revision(output_dir, report, expected_revision)
    return report


def compatible_multipoint_report(
    report: dict,
    expected_revision: str,
    *,
    output_dir: Path | None = None,
) -> bool:
    """Reuse only reports proven equivalent to the requested algorithm revision."""

    if (
        expected_revision == MULTIPOINT_EXPECTED_REVISION
        and report.get("revision") == BASELINE_LIFECYCLE_REVISION
    ):
        measurements = None if output_dir is None else output_dir / "measurements.csv"
        return measurements is not None and measurement_lengths_are_monotone(
            measurements
        )

    if (
        expected_revision == MULTIPOINT_EXPECTED_REVISION
        and report.get("revision") == SEMANTIC_GUARDED_REVISION
    ):
        diagnostics = report.get("worldsheet_diagnostics", {})
        measurements = None if output_dir is None else output_dir / "measurements.csv"
        if not diagnostics or measurements is None:
            return False
        return (
            all(
                diagnostic.get("first_active_sample") != 0
                or "contact" in str(diagnostic.get("measurement_status", ""))
                for diagnostic in diagnostics.values()
            )
            and measurement_lengths_are_monotone(measurements)
        )

    if (
        expected_revision != RIM_GEOMETRY_REVISION
        or report.get("revision") != LEGACY_MULTIPOINT_REVISION
    ):
        return False
    config = TemporalRibbonConfig()
    diagnostics = report.get("worldsheet_diagnostics", {})
    if not diagnostics:
        return False
    for owner in diagnostics.values():
        for audit in owner.get("mature_seed_audits", []):
            endpoint = audit.get("endpoint_radial_efficiency")
            excursion = audit.get("radial_excursion_radii")
            if endpoint is None or excursion is None:
                return False
            if (
                audit.get("geometry_eligible", False)
                and float(endpoint) < config.minimum_endpoint_radial_efficiency
                and float(excursion) < config.minimum_curved_radial_excursion_radii
            ):
                return False
    return True


def measurement_lengths_are_monotone(path: Path, tolerance: float = 1e-6) -> bool:
    """Return whether one cached owner has no accepted backward length steps."""

    if not path.exists():
        return False
    previous: float | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("accepted", "0").strip().lower() not in {"1", "true", "yes"}:
                continue
            current = float(row["tube_length_px"])
            if previous is not None and current < previous - tolerance:
                return False
            previous = current
    return previous is not None


def promote_report_revision(
    output_dir: Path,
    report: dict,
    expected_revision: str,
) -> None:
    """Atomically mark an evidence-compatible cached report as current."""

    previous_revision = report["revision"]
    report["revision"] = expected_revision
    report["compatible_from_revision"] = previous_revision
    report_path = output_dir / "report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    temporary_report.replace(report_path)

    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "revision" not in fieldnames:
        raise ValueError(f"missing revision column in {summary_path}")
    for row in rows:
        row["revision"] = expected_revision
    temporary_summary = summary_path.with_suffix(".csv.tmp")
    with temporary_summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_summary.replace(summary_path)


def localize_report_artifacts(output_dir: Path, report: dict) -> dict:
    """Point a resumed report at the artifacts beside its current copy."""

    artifacts = report.get("artifacts", {})
    changed = False
    for key, value in artifacts.items():
        local_path = output_dir / Path(value).name
        if Path(value) != local_path:
            artifacts[key] = str(local_path)
            changed = True
    if changed:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_owner(args: argparse.Namespace, track_id: int) -> tuple[int, dict]:
    """Run or resume one owner and return its completed report."""

    owner_dir = args.output / f"P{track_id:02d}"
    owner_dir.mkdir(parents=True, exist_ok=True)
    forced_seed = args.mature_seed_overrides.get(track_id)
    expected_revision = (
        GLOBAL_ALLOCATION_REVISION
        if forced_seed is not None
        else MULTIPOINT_EXPECTED_REVISION
        if args.owner_motion_cache is not None
        else EXPECTED_REVISION
    )
    report = completed_report(owner_dir, track_id, expected_revision)
    if report is not None and report.get("forced_mature_seed") != forced_seed:
        report = None
    if report is not None and args.export_alternatives:
        artifact_key = f"candidate_histories_P{track_id:02d}"
        artifact = report.get("artifacts", {}).get(artifact_key)
        if artifact is None or not (owner_dir / Path(artifact).name).exists():
            report = None
    if report is not None:
        return track_id, localize_report_artifacts(owner_dir, report)

    command = [
        sys.executable,
        "-m",
        "prototypes.v27_global_ribbon_worldsheet.track",
        str(args.movie),
        "--identity-report",
        str(args.identity_report),
        "--causal-atlas-run",
        str(args.causal_atlas_run),
        "--output",
        str(owner_dir),
        "--track-ids",
        str(track_id),
        "--width",
        str(args.width),
        "--sample-interval-s",
        str(args.sample_interval_s),
        "--field-cache",
        str(args.field_cache),
        "--selection-stride",
        str(args.selection_stride),
        "--alternatives",
        str(args.alternatives),
        "--mature-hypotheses",
        str(args.mature_hypotheses),
    ]
    if args.owner_motion_cache is not None:
        command.extend(("--owner-motion-cache", str(args.owner_motion_cache)))
    if args.export_alternatives:
        command.append("--export-alternatives")
    if forced_seed is not None:
        command.extend(("--force-mature-seed", forced_seed))
    log_path = owner_dir / "run.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"P{track_id:02d} failed; see {log_path}")
    report = completed_report(owner_dir, track_id, expected_revision)
    if report is None:
        raise RuntimeError(f"P{track_id:02d} did not produce a valid report")
    return track_id, localize_report_artifacts(owner_dir, report)


def combine_csv_files(
    output_path: Path,
    owner_dirs: list[Path],
    filename: str,
) -> int:
    """Combine matching per-owner CSV files while preserving one header."""

    fieldnames: list[str] | None = None
    combined_rows: list[dict[str, str]] = []
    for owner_dir in owner_dirs:
        source = owner_dir / filename
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"CSV columns differ in {source}")
            combined_rows.extend(reader)
    if fieldnames is None:
        return 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    return len(combined_rows)


def validate_combined_outputs(output_dir: Path, tolerance: float = 1e-3) -> dict:
    """Verify growth, summary, and centerline invariants in aggregate exports."""

    with (output_dir / "summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_lengths = {
        int(row["pollen_id"]): float(row["final_length_px"])
        for row in summary_rows
    }
    if len(summary_lengths) != len(summary_rows):
        raise ValueError("summary.csv contains duplicate pollen IDs")

    histories: dict[int, list[tuple[int, float, bool]]] = {}
    measurement_lengths: dict[tuple[int, int], float] = {}
    with (output_dir / "measurements.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            track_id = int(row["pollen_id"])
            sample = int(row["sample_index"])
            length = float(row["tube_length_px"])
            accepted = row["accepted"].strip().lower() in {"1", "true", "yes"}
            histories.setdefault(track_id, []).append((sample, length, accepted))
            measurement_lengths[(track_id, sample)] = length
    if set(histories) != set(summary_lengths):
        raise ValueError("summary and measurement owner IDs differ")

    maximum_summary_error = 0.0
    for track_id, history in histories.items():
        ordered = sorted(history)
        accepted_lengths = [length for _, length, accepted in ordered if accepted]
        if any(
            later < earlier - tolerance
            for earlier, later in zip(accepted_lengths, accepted_lengths[1:])
        ):
            raise ValueError(f"P{track_id:02d} has a backward accepted length step")
        final_error = abs(ordered[-1][1] - summary_lengths[track_id])
        maximum_summary_error = max(maximum_summary_error, final_error)
    if maximum_summary_error > tolerance:
        raise ValueError("summary and final measurement lengths differ")

    centerline_lengths: dict[tuple[int, int], float] = {}
    with (output_dir / "centerlines.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["pollen_id"]), int(row["sample_index"]))
            centerline_lengths[key] = max(
                centerline_lengths.get(key, 0.0),
                float(row["arc_length_px"]),
            )
    unknown = set(centerline_lengths) - set(measurement_lengths)
    if unknown:
        raise ValueError("centerlines contain unknown owner-sample rows")
    maximum_centerline_error = max(
        (
            abs(length - measurement_lengths[key])
            for key, length in centerline_lengths.items()
        ),
        default=0.0,
    )
    if maximum_centerline_error > tolerance:
        raise ValueError("measurement and centerline lengths differ")
    return {
        "owner_count": len(histories),
        "nondecreasing_owner_count": len(histories),
        "maximum_summary_final_error_px": maximum_summary_error,
        "maximum_centerline_length_error_px": maximum_centerline_error,
    }


def measurement_status(report: dict, track_id: int) -> str:
    """Return the report-level measurement class for one owner."""

    if track_id in report.get("measured_owner_ids", []):
        return "usable"
    if track_id in report.get("review_owner_ids", []):
        return "review"
    if track_id in report.get("unavailable_owner_ids", []):
        return "unavailable"
    if track_id in report.get("no_growth_owner_ids", []):
        return "no_growth"
    return "missing"


def write_manifest(
    args: argparse.Namespace,
    reports: dict[int, dict],
    failures: dict[int, str],
) -> Path:
    """Write aggregate status and combine complete per-owner CSV artifacts."""

    completed_ids = sorted(reports)
    owner_dirs = [args.output / f"P{track_id:02d}" for track_id in completed_ids]
    row_counts = {
        filename: combine_csv_files(args.output / filename, owner_dirs, filename)
        for filename in ("summary.csv", "measurements.csv", "centerlines.csv")
    }
    consistency = validate_combined_outputs(args.output)
    statuses = {
        track_id: measurement_status(reports[track_id], track_id)
        for track_id in completed_ids
    }
    manifest = {
        "revision": (
            GLOBAL_ALLOCATION_REVISION
            if args.mature_seed_overrides
            else MULTIPOINT_EXPECTED_REVISION
            if args.owner_motion_cache is not None
            else EXPECTED_REVISION
        ),
        "movie": str(args.movie),
        "field_cache": str(args.field_cache),
        "owner_motion_cache": (
            str(args.owner_motion_cache)
            if args.owner_motion_cache is not None
            else None
        ),
        "mature_seed_overrides": {
            str(owner): label
            for owner, label in sorted(args.mature_seed_overrides.items())
        },
        "requested_track_ids": parse_track_ids(args.track_ids),
        "completed_track_ids": completed_ids,
        "failed_track_ids": {str(key): value for key, value in sorted(failures.items())},
        "status_by_track_id": {str(key): value for key, value in statuses.items()},
        "status_counts": dict(sorted(Counter(statuses.values()).items())),
        "combined_csv_rows": row_counts,
        "consistency_checks": consistency,
    }
    manifest_path = args.output / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> int:
    """Run pending owners concurrently and create whole-field artifacts."""

    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    args.movie = args.movie.resolve()
    args.identity_report = args.identity_report.resolve()
    args.causal_atlas_run = args.causal_atlas_run.resolve()
    if args.owner_motion_cache is not None:
        args.owner_motion_cache = args.owner_motion_cache.resolve()
    args.mature_seed_overrides = load_mature_seed_overrides(
        args.mature_seed_overrides
    )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.field_cache = (
        args.field_cache.resolve()
        if args.field_cache is not None
        else args.output / "field_context"
    )
    prepare_field_context(
        args.field_cache,
        args.movie,
        args.width,
        args.sample_interval_s,
    )
    track_ids = parse_track_ids(args.track_ids)
    unknown_overrides = set(args.mature_seed_overrides) - set(track_ids)
    if unknown_overrides:
        raise ValueError(
            "mature-seed overrides include unrequested pollen IDs: "
            f"{sorted(unknown_overrides)}"
        )
    reports: dict[int, dict] = {}
    failures: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_owner, args, track_id): track_id for track_id in track_ids}
        for future in as_completed(futures):
            track_id = futures[future]
            try:
                _, report = future.result()
                reports[track_id] = report
                status = measurement_status(report, track_id)
                print(
                    f"[{len(reports) + len(failures)}/{len(track_ids)}] "
                    f"P{track_id:02d}: {status}",
                    flush=True,
                )
            except Exception as exc:  # Keep independent owners running.
                failures[track_id] = str(exc)
                print(
                    f"[{len(reports) + len(failures)}/{len(track_ids)}] "
                    f"P{track_id:02d}: FAILED ({exc})",
                    flush=True,
                )

    manifest = write_manifest(args, reports, failures)
    if not failures:
        arbitrate_validation_directory(args.output)
    print(f"Whole-field manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
