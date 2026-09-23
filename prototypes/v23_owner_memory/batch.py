"""Run independent v23 owner-conditioned tracing across a complete pollen field."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    """Parse field identity, owner selection, and trace controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", help="Optional comma-separated owner IDs")
    parser.add_argument("--start-position", type=int, default=3)
    parser.add_argument("--owner-radius", type=float, default=15.0)
    parser.add_argument("--crop-radius", type=int, default=210)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def initial_owner_ids(identity: dict) -> list[int]:
    """Return learned pollen owners present at the recording origin."""
    origin = int(identity["source_frames"][0])
    return sorted(
        int(track["track_id"])
        for track in identity["tracks"]
        if int(track["source_frames"][0]) == origin
        and int(track["semantic_observation_count"]) >= 2
    )


def write_batch_report(
    output: Path,
    identity_report: Path,
    owner_ids: list[int],
    failures: dict[int, str],
) -> dict:
    """Summarize completed, accepted, withheld, and failed owner runs."""
    existing_ids = {
        int(path.parent.name[1:])
        for path in output.glob("P[0-9][0-9][0-9]/report.json")
    }
    identity = json.loads(identity_report.read_text())
    confirmed_ids = set(initial_owner_ids(identity))
    report_owner_ids = sorted((set(owner_ids) | existing_ids) & confirmed_ids)
    excluded_ids = sorted(existing_ids - confirmed_ids)
    owner_runs = []
    for owner_id in report_owner_ids:
        report_path = output / f"P{owner_id:03d}" / "report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        owner_runs.append(
            {
                "owner_track_id": owner_id,
                "accepted": bool(report["accepted"]),
                "reason": report["reason"],
                "score_margin": report["score_margin"],
                "selected_observation_count": len(report["selected_candidate_ids"]),
                "selected_lengths_px": report["selected_lengths_px"],
                "net_growth_px": report.get("net_growth_px"),
                "directory": f"P{owner_id:03d}",
            }
        )
    summary = {
        "prototype": "v23_owner_conditioned_video_graph",
        "stage": "field-owner-batch",
        "identity_report": str(identity_report),
        "requested_owner_count": len(owner_ids),
        "reported_owner_count": len(report_owner_ids),
        "excluded_unconfirmed_owner_ids": excluded_ids,
        "completed_owner_count": len(owner_runs),
        "accepted_owner_count": sum(item["accepted"] for item in owner_runs),
        "withheld_owner_count": sum(not item["accepted"] for item in owner_runs),
        "failed_owner_count": len(failures),
        "failures": {str(key): value for key, value in sorted(failures.items())},
        "owners": owner_runs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "field_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary_fields = (
        "owner_track_id",
        "accepted",
        "reason",
        "score_margin",
        "selected_observation_count",
        "first_length_px",
        "final_length_px",
        "net_growth_px",
        "directory",
    )
    with (output / "field_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for owner in owner_runs:
            lengths = owner["selected_lengths_px"]
            writer.writerow(
                {
                    **{key: owner.get(key) for key in summary_fields},
                    "first_length_px": lengths[0] if lengths else "",
                    "final_length_px": lengths[-1] if lengths else "",
                }
            )

    measurement_fields = (
        "owner_track_id",
        "source_frame",
        "length_px",
        "measurement_status",
        "reason",
    )
    with (output / "field_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=measurement_fields)
        writer.writeheader()
        for owner in owner_runs:
            centerline_path = (
                output
                / owner["directory"]
                / "owner_centerlines.csv"
            )
            if not centerline_path.exists():
                continue
            length_by_frame: dict[int, float] = {}
            with centerline_path.open(newline="") as centerline_handle:
                for row in csv.DictReader(centerline_handle):
                    source_frame = int(row["source_frame"])
                    length_by_frame[source_frame] = max(
                        length_by_frame.get(source_frame, 0.0),
                        float(row["arclength_px"]),
                    )
            for source_frame, length_px in sorted(length_by_frame.items()):
                writer.writerow(
                    {
                        "owner_track_id": owner["owner_track_id"],
                        "source_frame": source_frame,
                        "length_px": length_px,
                        "measurement_status": (
                            "accepted" if owner["accepted"] else "review-only"
                        ),
                        "reason": owner["reason"],
                    }
                )
    return summary


def main() -> None:
    """Trace every requested origin pollen in isolated sequential processes."""
    args = parse_args()
    identity = json.loads(args.identity_report.read_text())
    owner_ids = (
        sorted({int(value) for value in args.track_ids.split(",")})
        if args.track_ids
        else initial_owner_ids(identity)
    )
    if not owner_ids:
        raise RuntimeError("the identity report contains no eligible origin pollen")

    trace_script = Path(__file__).with_name("trace.py")
    failures: dict[int, str] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for position, owner_id in enumerate(owner_ids, start=1):
        owner_output = args.output / f"P{owner_id:03d}"
        report_path = owner_output / "report.json"
        if args.resume and report_path.exists():
            print(
                f"[v23 field] {position}/{len(owner_ids)} P{owner_id:03d}: existing",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(trace_script),
            "--identity-report",
            str(args.identity_report),
            "--track-id",
            str(owner_id),
            "--output",
            str(owner_output),
            "--start-position",
            str(args.start_position),
            "--owner-radius",
            str(args.owner_radius),
            "--crop-radius",
            str(args.crop_radius),
        ]
        print(
            f"[v23 field] {position}/{len(owner_ids)} P{owner_id:03d}: tracing",
            flush=True,
        )
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            failures[owner_id] = (result.stderr or result.stdout)[-2000:]
            print(f"[v23 field] P{owner_id:03d}: failed", flush=True)
        else:
            report = json.loads(report_path.read_text())
            verdict = "accepted" if report["accepted"] else f"withheld ({report['reason']})"
            print(f"[v23 field] P{owner_id:03d}: {verdict}", flush=True)
        write_batch_report(
            args.output,
            args.identity_report,
            owner_ids,
            failures,
        )
    print(
        json.dumps(
            write_batch_report(
                args.output,
                args.identity_report,
                owner_ids,
                failures,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
