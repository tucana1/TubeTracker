"""Tests for resumable field-wide deformable-worldsheet evaluation."""

import csv
import json

from prototypes.v21_deformable_worldsheet.benchmark import (
    _valid_cached_report,
    load_selected_references,
    summarize_cases,
)
from tubetracker.deformable_worldsheet import DEFORMABLE_WORLDSHEET_REVISION


class WorldsheetBenchmarkTests:
    def test_reference_loader_keeps_every_selected_case(self, tmp_path):
        run = tmp_path / "v18"
        run.mkdir()
        rows = [
            {
                "consensus_id": "2",
                "selected": "True",
                "consensus_measurement_supported": "True",
                "consensus_trajectory_accepted": "True",
                "consensus_germination_accepted": "True",
                "source_final_length_px": "42.5",
                "source_quality_status": "trajectory-growth",
                "source_quality_flags": "",
            },
            {
                "consensus_id": "",
                "selected": "False",
                "consensus_measurement_supported": "False",
                "consensus_trajectory_accepted": "False",
                "consensus_germination_accepted": "False",
                "source_final_length_px": "17.0",
                "source_quality_status": "review",
                "source_quality_flags": "jump",
            },
        ]
        with (run / "phase_consensus_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        references = load_selected_references(run)

        assert len(references) == 1
        assert references[0]["consensus_id"] == 2
        assert references[0]["legacy_measurement_supported"]

    def test_cache_requires_expected_case_and_existing_artifacts(self, tmp_path):
        artifact = tmp_path / "review.mp4"
        artifact.write_bytes(b"video")
        report = tmp_path / "report.json"
        report.write_text(
            json.dumps(
                {
                    "prototype": "v21_deformable_orientation_worldsheet",
                    "algorithm_revision": DEFORMABLE_WORLDSHEET_REVISION,
                    "consensus_id": 4,
                    "artifacts": {"review_video": str(artifact)},
                }
            ),
            encoding="utf-8",
        )

        assert _valid_cached_report(report, 4)
        assert not _valid_cached_report(report, 5)
        artifact.unlink()
        assert not _valid_cached_report(report, 4)

    def test_summary_separates_legacy_agreement_from_accuracy(self):
        rows = [
            {
                "consensus_id": 1,
                "status": "completed",
                "legacy_measurement_supported": True,
                "legacy_length_px": 100.0,
                "v21_geometry_supported": True,
                "v21_measurement_supported": True,
                "v21_length_px": 90.0,
                "endpoint_reference_distance_px": 3.0,
            },
            {
                "consensus_id": 2,
                "status": "completed",
                "legacy_measurement_supported": False,
                "legacy_length_px": 50.0,
                "v21_geometry_supported": True,
                "v21_measurement_supported": False,
                "v21_length_px": 48.0,
                "endpoint_reference_distance_px": 2.0,
            },
            {
                "consensus_id": 3,
                "status": "failed",
                "legacy_measurement_supported": True,
            },
        ]

        summary = summarize_cases(rows)

        assert summary["completed_count"] == 2
        assert summary["failed_consensus_ids"] == [3]
        assert summary["v21_geometry_review_count"] == 1
        assert summary["legacy_agreement"]["both_measurement_supported"] == 1
        assert summary["relative_length_error_median"] == 0.1
        assert "not biological accuracy" in summary["interpretation"]
