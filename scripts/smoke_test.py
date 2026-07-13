#!/usr/bin/env python3
"""Exercise the bundled sample through the pilot analysis path."""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASELINE = {
    "frames_analyzed": 40,
    "grain_count": 31,
    "germinated_count": 20,
    "tip_detection_count": 662,
    "track_count": 41,
    "tracking_engine": "laptrack",
    "burst_candidate_count": 0,
}


def main():
    """Run the sample video and verify required outputs and nonzero results."""
    with tempfile.TemporaryDirectory(prefix="tubetracker-smoke-") as temp_dir:
        output_dir = Path(temp_dir) / "output"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pilot.py"),
            str(REPO_ROOT / "sample_movie.avi"),
            "--sample-id",
            "bundled-sample",
            "--genotype",
            "test",
            "--biological-replicate",
            "test-1",
            "--time-per-frame",
            "60",
            "--max-frames",
            "40",
            "--min-points-per-track",
            "4",
            "--gap-closing",
            "5",
            "--smoothing-gap",
            "2",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        summary = json.loads((output_dir / "summary.json").read_text())
        manifest = json.loads((output_dir / "run_manifest.json").read_text())
        expected = [
            output_dir / "pilot.grains.csv",
            output_dir / "pilot.tracks.csv",
            output_dir / "bundled-sample.survival.raw.data.csv",
            output_dir / "bundled-sample.tracks.raw.data.csv",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise SystemExit(f"Smoke test outputs missing: {missing}")
        for key in ("grain_count", "germinated_count", "tip_detection_count", "track_count"):
            if not summary[key]:
                raise SystemExit(f"Smoke test expected nonzero {key}: {summary}")
        with (output_dir / "pilot.grains.csv").open() as handle:
            grains = list(csv.DictReader(handle))
        with (output_dir / "pilot.tracks.csv").open() as handle:
            tracks = list(csv.DictReader(handle))
        if len(grains) != summary["grain_count"]:
            raise SystemExit("Grain summary row count does not match summary.json")
        if not tracks or not any(row["grain_id"] for row in tracks):
            raise SystemExit("Track summary did not contain grain-to-track associations")
        rate_column = "growth_rate_pxl_per_sec"
        if not any(row[rate_column] for row in tracks):
            raise SystemExit("Track summary did not contain interval growth rates")
        if summary["burst_candidate_count"] != 0:
            raise SystemExit("Burst candidates should be opt-in for the pilot workflow")
        actual_baseline = {
            key: summary[key]
            for key in EXPECTED_BASELINE
        }
        if actual_baseline != EXPECTED_BASELINE:
            raise SystemExit(
                "Bundled sample changed from the reviewed OpenCV 5/LapTrack "
                f"baseline: expected {EXPECTED_BASELINE}, got {actual_baseline}"
            )
        software = manifest["software"]
        if software["opencv_distribution"] != "5.0.0.93":
            raise SystemExit(f"Unexpected OpenCV version: {software}")
        if software["laptrack"] != "0.17.1":
            raise SystemExit(f"Unexpected LapTrack version: {software}")
        print("TubeTracker smoke test passed")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
