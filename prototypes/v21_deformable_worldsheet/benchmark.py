#!/usr/bin/env python3
"""Run and summarize independent v21 reconstruction across one retained field."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tubetracker.deformable_worldsheet import (  # noqa: E402
    DEFORMABLE_WORLDSHEET_REVISION,
    WORLDSHEET_PROMOTION_REVISION,
    worldsheet_promotion_decision,
)


PROTOTYPE_NAME = "v21_deformable_orientation_worldsheet"
TRACKER = Path(__file__).with_name("track.py")


def parse_args() -> argparse.Namespace:
    """Read field-wide benchmark and resource settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-samples", type=int, default=120)
    parser.add_argument("--output-samples", type=int, default=30)
    parser.add_argument("--search-radius-px", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_bool(value: str | bool | None) -> bool:
    """Read the explicit boolean spelling used by retained CSV outputs."""

    return value is True or str(value).strip().lower() == "true"


def load_selected_references(v18_run: Path) -> list[dict]:
    """Load every selected v18 case without treating it as ground truth."""

    summary_path = v18_run / "phase_consensus_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if not _parse_bool(row.get("selected")) or not row.get("consensus_id"):
            continue
        selected.append(
            {
                "consensus_id": int(row["consensus_id"]),
                "legacy_measurement_supported": _parse_bool(
                    row.get("consensus_measurement_supported")
                ),
                "legacy_trajectory_accepted": _parse_bool(
                    row.get("consensus_trajectory_accepted")
                ),
                "legacy_germination_accepted": _parse_bool(
                    row.get("consensus_germination_accepted")
                ),
                "legacy_length_px": float(row["source_final_length_px"]),
                "legacy_quality_status": row.get("source_quality_status", ""),
                "legacy_quality_flags": row.get("source_quality_flags", ""),
            }
        )
    return sorted(selected, key=lambda row: row["consensus_id"])


def _valid_cached_report(report_path: Path, consensus_id: int) -> bool:
    """Accept only a complete report from the expected prototype and case."""

    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    artifacts = report.get("artifacts", {})
    return (
        report.get("prototype") == PROTOTYPE_NAME
        and report.get("algorithm_revision") == DEFORMABLE_WORLDSHEET_REVISION
        and report.get("consensus_id") == consensus_id
        and all(Path(path).is_file() for path in artifacts.values())
    )


def _run_case(
    reference: dict,
    v18_run: Path,
    output_dir: Path,
    evidence_samples: int,
    output_samples: int,
    search_radius_px: float,
    force: bool,
) -> dict:
    """Run one isolated candidate, preserving failures as benchmark rows."""

    consensus_id = reference["consensus_id"]
    case_dir = output_dir / f"c{consensus_id:04d}"
    report_path = case_dir / "report.json"
    started = time.monotonic()
    cached = not force and _valid_cached_report(report_path, consensus_id)
    error = ""
    if not cached:
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(TRACKER),
            "--v18-run",
            str(v18_run),
            "--consensus-id",
            str(consensus_id),
            "--evidence-samples",
            str(evidence_samples),
            "--output-samples",
            str(output_samples),
            "--independent-atlas",
            "--independent-search-radius-px",
            str(search_radius_px),
            "--output-dir",
            str(case_dir),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode:
            error = completed.stderr.strip()[-2000:]
    elapsed = time.monotonic() - started
    row = dict(reference)
    row.update(
        {
            "status": "failed" if error or not report_path.is_file() else "completed",
            "cached": cached,
            "runtime_s": round(elapsed, 3),
            "error": error,
            "case_dir": str(case_dir),
        }
    )
    if row["status"] == "failed":
        return row
    report = json.loads(report_path.read_text(encoding="utf-8"))
    winner = (report.get("independent_atlas_candidates") or [{}])[0]
    dynamics = report.get("trajectory_dynamics", {})
    alignment = report.get("legacy_alignment", {})
    continuity = report.get("material_continuity", {})
    openness = report.get("open_curve", {})
    geometry_supported, measurement_supported, decision_reasons = (
        worldsheet_promotion_decision(
            prototype_score=float(winner.get("global_prototype_score", float("nan"))),
            prototype_supported_fraction=float(report["v21_supported_fraction"]),
            terminal_blobness=float(winner.get("terminal_blobness", float("nan"))),
            accepted_frame_fraction=float(
                dynamics.get("accepted_frame_fraction", 0.0)
            ),
            terminal_preexisting_support=float(
                winner.get("preexisting_support", 1.0)
            ),
            proximal_eventual_support_fraction=float(
                continuity.get("proximal_eventual_support_fraction", 1.0)
            ),
            maximum_unsupported_gap_px=float(
                continuity.get("maximum_unsupported_gap_px", 0.0)
            ),
            endpoint_separation_fraction=float(
                openness.get("endpoint_separation_fraction", 1.0)
            ),
        )
    )
    terminal_foreign_body_risk = float(
        winner.get("terminal_blobness", float("nan"))
    ) * np.sqrt(float(winner.get("preexisting_support", 1.0)))
    birth_error_p90 = report.get("p90_birth_identity_error_samples")
    row.update(
        {
            "v21_geometry_supported": geometry_supported,
            "algorithm_revision": report.get("algorithm_revision", ""),
            "promotion_policy_revision": WORLDSHEET_PROMOTION_REVISION,
            "atlas_temporal_scope": report.get("atlas_temporal_scope", ""),
            "atlas_evidence_model": report.get("atlas_evidence_model", ""),
            "germination_time_supported": bool(
                report.get("germination_time_supported", False)
            ),
            "v21_measurement_supported": measurement_supported,
            "v21_length_px": float(report["selected_atlas_length_px"]),
            "v21_supported_fraction": float(report["v21_supported_fraction"]),
            "v21_prototype_score": float(
                winner.get("global_prototype_score", float("nan"))
            ),
            "v21_terminal_blobness": float(
                winner.get("terminal_blobness", float("nan"))
            ),
            "v21_terminal_foreign_body_risk": terminal_foreign_body_risk,
            "v21_endpoint_separation_fraction": float(
                openness.get("endpoint_separation_fraction", float("nan"))
            ),
            "v21_accepted_frame_fraction": float(
                dynamics.get("accepted_frame_fraction", 0.0)
            ),
            "v21_birth_error_p90_samples": (
                float(birth_error_p90)
                if birth_error_p90 is not None
                else float("nan")
            ),
            "endpoint_reference_distance_px": float(
                alignment.get("tip_distance_px", float("nan"))
            ),
            "centerline_reference_chamfer_px": float(
                alignment.get("symmetric_chamfer_px", float("nan"))
            ),
            "decision_reasons": ";".join(decision_reasons),
            "proximal_eventual_support_fraction": float(
                continuity.get(
                    "proximal_eventual_support_fraction",
                    float("nan"),
                )
            ),
            "maximum_unsupported_gap_px": float(
                continuity.get(
                    "maximum_unsupported_gap_px",
                    float("nan"),
                )
            ),
        }
    )
    return row


def _finite_values(rows: list[dict], key: str) -> np.ndarray:
    """Collect finite numeric values from completed benchmark rows."""

    values = np.asarray(
        [row.get(key, float("nan")) for row in rows],
        dtype=np.float64,
    )
    return values[np.isfinite(values)]


def summarize_cases(rows: list[dict]) -> dict:
    """Summarize coverage and legacy agreement without claiming ground truth."""

    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    both = sum(
        row["legacy_measurement_supported"] and row["v21_measurement_supported"]
        for row in completed
    )
    v21_only = sum(
        not row["legacy_measurement_supported"]
        and row["v21_measurement_supported"]
        for row in completed
    )
    legacy_only = sum(
        row["legacy_measurement_supported"]
        and not row["v21_measurement_supported"]
        for row in completed
    )
    neither = len(completed) - both - v21_only - legacy_only
    comparable = [
        row
        for row in completed
        if row["legacy_measurement_supported"] and row["v21_geometry_supported"]
    ]
    relative_length_error = np.asarray(
        [
            abs(row["v21_length_px"] - row["legacy_length_px"])
            / max(row["legacy_length_px"], 1e-6)
            for row in comparable
        ],
        dtype=np.float64,
    )
    endpoint = _finite_values(comparable, "endpoint_reference_distance_px")
    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "v21_geometry_supported_count": sum(
            row["v21_geometry_supported"] for row in completed
        ),
        "v21_measurement_supported_count": sum(
            row["v21_measurement_supported"] for row in completed
        ),
        "v21_geometry_review_count": sum(
            row["v21_geometry_supported"]
            and not row["v21_measurement_supported"]
            for row in completed
        ),
        "legacy_measurement_supported_count": sum(
            row["legacy_measurement_supported"] for row in completed
        ),
        "legacy_agreement": {
            "both_measurement_supported": both,
            "v21_only_measurement_supported": v21_only,
            "legacy_only_measurement_supported": legacy_only,
            "neither_measurement_supported": neither,
        },
        "legacy_supported_geometry_comparison_count": len(comparable),
        "relative_length_error_median": (
            float(np.median(relative_length_error))
            if len(relative_length_error)
            else None
        ),
        "relative_length_error_p90": (
            float(np.percentile(relative_length_error, 90))
            if len(relative_length_error)
            else None
        ),
        "endpoint_reference_distance_median_px": (
            float(np.median(endpoint)) if len(endpoint) else None
        ),
        "endpoint_reference_distance_p90_px": (
            float(np.percentile(endpoint, 90)) if len(endpoint) else None
        ),
        "failed_consensus_ids": [row["consensus_id"] for row in failed],
        "interpretation": (
            "Legacy agreement measures consistency with v18, not biological accuracy; "
            "manual blinded centerlines remain the accuracy reference."
        ),
    }


def write_benchmark(rows: list[dict], output_dir: Path) -> dict:
    """Write one case table and one aggregate report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "benchmark_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_cases(rows)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Run selected cases concurrently and preserve incremental progress."""

    args = parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be at least one")
    references = load_selected_references(args.v18_run)
    if args.limit is not None:
        references = references[: max(0, args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _run_case,
                reference,
                args.v18_run,
                args.output_dir,
                args.evidence_samples,
                args.output_samples,
                args.search_radius_px,
                args.force,
            ): reference["consensus_id"]
            for reference in references
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[v21 benchmark] C{row['consensus_id']} {row['status']} "
                f"({len(rows)}/{len(references)})",
                flush=True,
            )
            write_benchmark(sorted(rows, key=lambda item: item["consensus_id"]), args.output_dir)
    summary = write_benchmark(
        sorted(rows, key=lambda item: item["consensus_id"]),
        args.output_dir,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
