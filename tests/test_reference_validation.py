"""Tests for blinded manual centerline preparation and scoring."""

import csv
import json

import cv2 as cv
import numpy as np
import pytest

from tubetracker.reference_validation import (
    _sample_indices,
    _verify_pollen_ring,
    finalize_reference_kit,
    load_reference_kit,
    prepare_reference_kit,
    save_annotations,
    score_reference_kit,
)


def _write_csv(path, fields, rows):
    """Write a compact CSV fixture."""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_inputs(tmp_path):
    """Create one moving-pollen movie and a matching prediction run."""

    movie = tmp_path / "movie.avi"
    writer = cv.VideoWriter(
        str(movie),
        cv.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(6):
        frame = np.full((48, 64, 3), 190, dtype=np.uint8)
        cv.circle(frame, (30, 24), 5, (70, 70, 70), 1)
        if index >= 2:
            cv.line(frame, (35, 24), (40, 24), (80, 80, 80), 2)
        writer.write(frame)
    writer.release()

    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "diameter_px": 10.0,
                "tracks": [
                    {
                        "track_id": 7,
                        "source_frames": [0, 5],
                        "centers_yx": [[24.0, 30.0], [24.0, 30.0]],
                    }
                ],
            }
        )
    )
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    _write_csv(
        prediction / "summary.csv",
        ("pollen_id", "status", "first_persistent_source_frame", "final_length_px"),
        [
            {
                "pollen_id": 7,
                "status": "measured",
                "first_persistent_source_frame": 2,
                "final_length_px": 10.0,
            }
        ],
    )
    _write_csv(
        prediction / "measurements.csv",
        ("pollen_id", "sample_index", "source_frame", "tube_length_px", "accepted"),
        [
            {
                "pollen_id": 7,
                "sample_index": index,
                "source_frame": index,
                "tube_length_px": 0.0 if index < 2 else 10.0,
                "accepted": int(index >= 2),
            }
            for index in range(6)
        ],
    )
    _write_csv(
        prediction / "centerlines.csv",
        (
            "pollen_id",
            "sample_index",
            "source_frame",
            "point_index",
            "source_x_px",
            "source_y_px",
            "arc_length_px",
        ),
        [
            {
                "pollen_id": 7,
                "sample_index": source_frame,
                "source_frame": source_frame,
                "point_index": point_index,
                "source_x_px": x,
                "source_y_px": 24.0,
                "arc_length_px": point_index * 10.0,
            }
            for source_frame in range(2, 6)
            for point_index, x in enumerate((30.0, 40.0))
        ],
    )
    return movie, identity, prediction


def test_sample_schedule_has_fixed_coverage_and_optional_challenges():
    lengths = np.asarray([0.0, 0.0, 2.0, 5.0, 5.0, 8.0])

    without_challenge = _sample_indices(lengths, 2, 3, 0)
    with_challenge = _sample_indices(lengths, 2, 3, 2)

    assert without_challenge == [(0, "uniform"), (2, "uniform"), (5, "uniform")]
    assert len(with_challenge) == 5
    assert {kind for _, kind in with_challenge} == {"uniform", "challenge"}


def test_fixed_owner_center_reports_pollen_ring_support_without_searching():
    frame = np.full((100, 100, 3), 190, dtype=np.uint8)
    cv.circle(frame, (58, 50), 10, (40, 40, 40), 2, cv.LINE_AA)
    cv.circle(frame, (58, 50), 7, (220, 220, 220), -1, cv.LINE_AA)

    assert _verify_pollen_ring(frame, (50.0, 58.0), 15.0)
    assert not _verify_pollen_ring(frame, (50.0, 42.0), 15.0)


def test_blinded_kit_runs_from_raw_crops_through_locked_scoring(tmp_path):
    movie, identity, prediction = _synthetic_inputs(tmp_path)
    kit = tmp_path / "kit"

    manifest = prepare_reference_kit(
        movie,
        identity,
        prediction,
        kit,
        track_ids=[7],
        uniform_samples=2,
        challenge_samples=1,
        crop_size=32,
        seed=11,
        refine_centers=False,
    )

    assert manifest["blinded"] is True
    assert manifest["record_count"] == 3
    assert manifest["cases"][0]["case_id"] != "7"
    assert all((kit / record["crop_image"]).exists() for record in manifest["cases"][0]["records"])

    _, annotations = load_reference_kit(kit)
    for record in manifest["cases"][0]["records"]:
        annotation = annotations["records"][record["record_id"]]
        annotation["completed"] = True
        if record["source_frame"] == 0:
            annotation["outcome"] = "no_visible_tube"
        elif record["source_frame"] == 2:
            annotation["outcome"] = "incorrect_target"
        else:
            annotation["outcome"] = "visible_tube"
            annotation["points_source_xy"] = [[30.0, 24.0], [40.0, 24.0]]
    save_annotations(kit, annotations)

    lock = finalize_reference_kit(kit, "test reviewer")
    score = score_reference_kit(kit, prediction)

    assert lock["record_count"] == 3
    assert score["visible_tube_accuracy"] == 1.0
    assert score["length_mae_px"] == 0.0
    assert score["tip_median_error_px"] == 0.0
    assert score["germination"]["uniform"]["predictions_within_interval"] == 1
    assert score["by_sampling_stratum"]["challenge"]["identity_failure_frames"] == 1
    assert score["owner_identity"]["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(RuntimeError, match="locked"):
        save_annotations(kit, annotations)
