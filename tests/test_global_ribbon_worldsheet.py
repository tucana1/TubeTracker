"""Tests for global multi-hypothesis pollen-tube reconstruction."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import prototypes.v27_global_ribbon_worldsheet.track as v27_track

from prototypes.v27_global_ribbon_worldsheet.track import (
    _causal_audit_key,
    _classify_measurement_status,
    _has_event_certificate,
    _identity_body_hypotheses,
    _mature_selection_key,
    _baseline_departure_audit,
    _path_exits_field,
    _radial_geometry_audit,
    load_causal_atlas_paths,
    load_field_context,
    load_owner_motion_cache,
    prepare_field_context,
    write_candidate_history_archive,
)
from scripts.audit_v28_field_validation import (
    SPAN_STAGE_NAMES,
    audit_stage_names,
    centered_crop,
    event_timed_samples,
    preferred_source_positions,
    resolve_field_cache,
)
from scripts.run_v27_field_validation import (
    EXPECTED_REVISION,
    BASELINE_LIFECYCLE_REVISION,
    LEGACY_MULTIPOINT_REVISION,
    MULTIPOINT_EXPECTED_REVISION,
    RIM_GEOMETRY_REVISION,
    SEMANTIC_GUARDED_REVISION,
    compatible_multipoint_report,
    completed_report,
    localize_report_artifacts,
    load_mature_seed_overrides,
    measurement_status,
    measurement_lengths_are_monotone,
    parse_track_ids,
    validate_combined_outputs,
)
from tubetracker.global_ribbon_worldsheet import (
    GlobalRibbonConfig,
    RibbonHypothesis,
    ordered_prefix_error,
    select_global_ribbon_worldsheet,
)
from tubetracker.field_arbitration import (
    CandidateBranchEvidence,
    FieldOwnerEvidence,
    allocate_distinct_candidate_branches,
    candidate_duplicate_fraction,
    first_sustained_overlap_index,
    resolve_ownership_components,
    sustained_branch_capture_index,
    sustained_duplicate_claims,
)
from tubetracker.temporal_ribbon import (
    TemporalRibbonConfig,
    TemporalRibbonFrame,
    TemporalRibbonHistory,
    regularize_monotone_history,
    trace_mature_ribbon_worldsheet,
)


def _horizontal(length, *, quality=1.0, y=0.0, source=0):
    path = np.column_stack(
        (np.full(length + 1, y), np.arange(length + 1, dtype=float))
    )
    return RibbonHypothesis(
        path_yx=path,
        optical_score=quality * max(length, 1),
        paired_fraction=quality,
        mean_paired_support=quality,
        endpoint_openness=1.0,
        source_index=source,
    )


def _branch(length, turn_at, *, quality=1.0, source=0):
    prefix = np.column_stack(
        (np.zeros(turn_at + 1), np.arange(turn_at + 1, dtype=float))
    )
    branch_length = length - turn_at
    branch = np.column_stack(
        (
            np.arange(1, branch_length + 1, dtype=float),
            np.full(branch_length, float(turn_at)),
        )
    )
    path = np.vstack((prefix, branch))
    return RibbonHypothesis(
        path_yx=path,
        optical_score=quality * length,
        paired_fraction=quality,
        mean_paired_support=quality,
        endpoint_openness=0.8,
        source_index=source,
    )


def _history(lengths, onset):
    frames = []
    for length in lengths:
        path = np.column_stack((np.zeros(length + 1), np.arange(length + 1)))
        frames.append(
            TemporalRibbonFrame(
                path_relative_yx=path,
                length_px=float(length),
                paired_fraction=1.0,
                mean_paired_support=1.0,
                prior_error_px=0.0,
                accepted=length > 0,
            )
        )
    return TemporalRibbonHistory(tuple(frames), onset)


def test_event_timed_audit_samples_expose_emergence_and_later_continuity():
    samples = event_timed_samples(100, onset_sample=10)

    assert samples.tolist() == [0, 9, 10, 14, 54, 99]


def test_event_timed_audit_samples_cover_full_span_without_an_event():
    samples = event_timed_samples(11, onset_sample=None)

    assert samples.tolist() == [0, 2, 4, 6, 8, 10]
    assert audit_stage_names(11, None) == SPAN_STAGE_NAMES


def test_centered_audit_crop_pads_field_edges_without_shifting_owner():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)

    crop = centered_crop(image, np.asarray((0.0, 0.0)), size=6, fill=17)

    assert crop.shape == (6, 6, 3)
    assert crop[3, 3].tolist() == [1, 2, 3]
    assert crop[0, 0].tolist() == [17, 17, 17]


def test_audit_resolves_shared_field_cache_with_explicit_override(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    recorded = tmp_path / "recorded"
    override = tmp_path / "override"

    assert resolve_field_cache(
        validation,
        {"field_cache": str(recorded)},
        override,
    ) == override.resolve()
    assert resolve_field_cache(
        validation,
        {"field_cache": str(recorded)},
        None,
    ) == recorded.resolve()
    assert resolve_field_cache(validation, {}, None) == validation / "field_context"


def test_audit_and_trace_prefer_detection_constrained_owner_motion(tmp_path):
    path = tmp_path / "motion.npz"
    frames = np.asarray((0, 10), dtype=np.int64)
    shape = (1, 2, 2)
    np.savez_compressed(
        path,
        source_frames=frames,
        track_ids=np.asarray((7,), dtype=np.int64),
        analysis_width=np.asarray(480),
        multipoint_aligned_yx=np.zeros(shape),
        semantic_span_aligned_yx=np.ones(shape),
        detection_constrained_aligned_yx=np.full(shape, 2.0),
        multipoint_source_yx=np.full(shape, 3.0),
        semantic_span_source_yx=np.full(shape, 4.0),
        detection_constrained_source_yx=np.full(shape, 5.0),
        template_aligned_yx=np.zeros(shape),
        template_source_yx=np.zeros(shape),
        observed=np.ones(shape[:2], dtype=bool),
        inlier_counts=np.ones(shape[:2]),
    )

    track_ids, trajectories, _, _ = load_owner_motion_cache(path, frames, 480)
    with np.load(path) as cache:
        audit_positions = preferred_source_positions(cache)

    assert track_ids == [7]
    assert np.all(trajectories == 2.0)
    assert np.all(audit_positions == 5.0)


def test_mature_seed_ranking_prefers_sustained_growth_over_last_frame_spike():
    sustained = _history([0] * 4 + [4, 8, 12, 16, 20, 24], onset=4)
    late_spike = _history([0] * 8 + [2, 24], onset=8)
    worldsheet = SimpleNamespace(total_score=10.0)

    assert _causal_audit_key(sustained, worldsheet) > _causal_audit_key(
        late_spike,
        worldsheet,
    )


def test_event_certificate_uses_growth_and_geometry_not_absolute_global_score():
    config = TemporalRibbonConfig()

    assert _has_event_certificate(12, 0.3, True, config)
    assert not _has_event_certificate(12, 0.0, True, config)
    assert not _has_event_certificate(12, 0.3, False, config)


def test_non_event_and_image_exit_receive_explicit_statuses():
    frame = TemporalRibbonFrame(
        path_relative_yx=np.asarray(((0.0, 5.0), (0.0, 12.0))),
        length_px=7.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )

    assert _path_exits_field(
        frame,
        center_yx=np.asarray((5.0, 5.0)),
        shift_xy=np.zeros(2),
        field_shape=(20, 15),
    )
    assert (
        _classify_measurement_status(
            4,
            contact_censored=False,
            event_certified=True,
            review_reasons=[],
            path_exits_field=True,
        )
        == "boundary_censored"
    )
    assert (
        _classify_measurement_status(
            0,
            contact_censored=False,
            event_certified=False,
            review_reasons=["pollen-rim-like-geometry"],
            path_exits_field=False,
        )
        == "no_growth_detected"
    )


def test_review_ambiguity_takes_precedence_over_boundary_censoring():
    assert (
        _classify_measurement_status(
            0,
            contact_censored=False,
            event_certified=True,
            review_reasons=["left-censored-near-foreign-owner"],
            path_exits_field=True,
        )
        == "review_quality"
    )


def test_mature_selection_prefers_complete_path_in_a_close_growth_race():
    partial = {
        "causal_key": (1.0, 0.702, 0.69, 1.16),
        "final_length_px": 26.0,
    }
    complete = {
        "causal_key": (1.0, 0.663, 0.73, 1.17),
        "final_length_px": 31.0,
    }

    assert _mature_selection_key(complete, 31.0) > _mature_selection_key(
        partial,
        31.0,
    )


def test_short_contact_prefix_can_pass_before_full_radial_departure():
    path = np.asarray(((0.0, 5.0), (0.0, 12.0)))
    frame = TemporalRibbonFrame(
        path_relative_yx=path,
        length_px=7.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )
    config = TemporalRibbonConfig()

    ordinary, _, _ = _radial_geometry_audit(frame, 5.0, config)
    contact, _, _ = _radial_geometry_audit(
        frame,
        5.0,
        config,
        contact_censored=True,
    )

    assert not ordinary
    assert contact


def test_frame_zero_rim_arc_fails_baseline_departure_lifecycle():
    frame = TemporalRibbonFrame(
        path_relative_yx=np.asarray(((0.0, 5.0), (3.0, 4.0), (5.0, 0.0))),
        length_px=8.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )
    history = TemporalRibbonHistory((frame,), first_persistent_sample=0)

    eligible, excursion = _baseline_departure_audit(
        history,
        owner_radius=5.0,
        config=TemporalRibbonConfig(),
    )

    assert not eligible
    assert excursion is not None and excursion < 1.5


def test_later_emergence_and_contact_censoring_bypass_baseline_gate():
    frame = TemporalRibbonFrame(
        path_relative_yx=np.asarray(((0.0, 5.0), (1.0, 5.0))),
        length_px=1.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )
    config = TemporalRibbonConfig()

    later = TemporalRibbonHistory((frame,), first_persistent_sample=1)
    contact = TemporalRibbonHistory((frame,), first_persistent_sample=0)

    assert _baseline_departure_audit(later, 5.0, config) == (True, None)
    assert _baseline_departure_audit(
        contact, 5.0, config, contact_censored=True
    ) == (True, None)


def test_negative_left_censored_baseline_excursion_is_rejected():
    frames = np.zeros((2, 32, 32), dtype=np.uint8)
    centers = np.full((2, 2), 16.0, dtype=np.float64)
    mature = np.asarray(((0.0, 0.0), (0.0, 6.0)), dtype=np.float64)

    with pytest.raises(ValueError, match="left-censored baseline excursion"):
        trace_mature_ribbon_worldsheet(
            frames,
            centers,
            mature,
            lambda crop: None,
            TemporalRibbonConfig(
                crop_size=32,
                minimum_left_censored_baseline_excursion_radii=-0.1,
            ),
        )


def test_material_history_isotonic_fit_removes_shrinkage_and_keeps_raw_lengths():
    history = _history([5, 3, 7], onset=0)

    regularized = regularize_monotone_history(history)

    assert np.allclose(regularized.lengths_px, (4.0, 4.0, 7.0))
    assert np.allclose(
        [
            np.linalg.norm(np.diff(frame.path_relative_yx, axis=0), axis=1).sum()
            for frame in regularized.frames
        ],
        regularized.lengths_px,
    )
    assert regularized.frames[0].observed_length_px == 5.0
    assert regularized.frames[1].observed_length_px == 3.0
    assert regularized.frames[2].observed_length_px is None


def test_candidate_history_archive_preserves_all_sparse_curve_hypotheses(
    tmp_path,
):
    histories = [_history([2, 4], onset=0), _history([0, 3], onset=1)]
    audits = [
        {
            "causal_key": (1.0, 0.5, 0.8, 1.0),
            "geometry_eligible": True,
            "near_foreign_owner": False,
        },
        {
            "causal_key": (1.0, 0.3, 0.4, 0.7),
            "geometry_eligible": True,
            "near_foreign_owner": True,
        },
    ]
    output = tmp_path / "candidates.npz"

    write_candidate_history_archive(
        output,
        ["rim-0", "rim-1-contact-censored"],
        histories,
        audits,
        np.asarray((0, 2)),
        np.asarray((0, 10, 20)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        analysis_scale=0.5,
        selected_index=0,
    )

    with np.load(output, allow_pickle=False) as archive:
        assert archive["labels"].tolist() == [
            "rim-0",
            "rim-1-contact-censored",
        ]
        assert archive["source_frames"].tolist() == [0, 20]
        assert archive["selected_index"].item() == 0
        assert archive["onset_sample_indices"].tolist() == [0, 2]
        assert archive["lengths_px"].tolist() == [[4.0, 8.0], [0.0, 6.0]]
        assert archive["contact_censored"].tolist() == [False, True]
        assert archive["near_foreign_owner"].tolist() == [False, True]
        assert archive["owner_centers_aligned_yx"].shape == (2, 2)
        assert np.allclose(
            archive["source_yx"][0, 1, :5, 1],
            np.arange(5) * 2.0,
        )


def test_low_excursion_half_circle_is_rejected_as_pollen_rim_geometry():
    frame = TemporalRibbonFrame(
        path_relative_yx=np.asarray(((0.0, 10.0), (0.0, 27.0), (0.0, 25.0))),
        length_px=30.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )

    eligible, endpoint_efficiency, excursion_radii = _radial_geometry_audit(
        frame,
        owner_radius=10.0,
        config=TemporalRibbonConfig(),
    )

    assert endpoint_efficiency == 0.5
    assert excursion_radii == 1.7
    assert not eligible


def test_large_radial_excursion_preserves_curved_tube_geometry():
    frame = TemporalRibbonFrame(
        path_relative_yx=np.asarray(((0.0, 10.0), (0.0, 31.0), (0.0, 25.0))),
        length_px=30.0,
        paired_fraction=1.0,
        mean_paired_support=1.0,
        prior_error_px=0.0,
        accepted=True,
    )

    eligible, endpoint_efficiency, excursion_radii = _radial_geometry_audit(
        frame,
        owner_radius=10.0,
        config=TemporalRibbonConfig(),
    )

    assert endpoint_efficiency == 0.5
    assert excursion_radii == 2.1
    assert eligible


def _field_owner(
    track_id,
    curves,
    *,
    onset,
    certified=True,
    status="usable",
    near_foreign=False,
):
    return FieldOwnerEvidence(
        track_id=track_id,
        status=status,
        event_certified=certified,
        onset_sample=onset,
        curves_by_sample={
            sample: np.asarray(curve, dtype=np.float64)
            for sample, curve in enumerate(curves)
        },
        near_foreign_owner=near_foreign,
    )


def test_field_arbitration_requires_sustained_not_incidental_path_overlap():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    vertical = np.column_stack((np.arange(31, dtype=float), np.zeros(31)))
    first = _field_owner(1, [horizontal] * 20, onset=4)
    mostly_distinct = _field_owner(2, [vertical] * 19 + [horizontal], onset=4)

    assert sustained_duplicate_claims([first, mostly_distinct]) == []


def test_field_arbitration_detects_repeated_duplicate_distal_centerlines():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    offset = horizontal + np.asarray((1.0, 0.0))
    first = _field_owner(1, [horizontal] * 20, onset=4)
    second = _field_owner(2, [offset] * 20, onset=8)

    claims = sustained_duplicate_claims([first, second])

    assert len(claims) == 1
    assert claims[0]["first_owner"] == 1
    assert claims[0]["second_owner"] == 2
    assert claims[0]["sustained_fraction"] == 1.0


def test_field_arbitration_detects_persistent_late_shared_tail():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    vertical = np.column_stack((np.arange(31, dtype=float), np.zeros(31)))
    offset = horizontal + np.asarray((0.5, 0.0))
    first = _field_owner(1, [vertical] * 9 + [horizontal] * 11, onset=2)
    second = _field_owner(2, [vertical + 20.0] * 9 + [offset] * 11, onset=8)

    claims = sustained_duplicate_claims([first, second])

    assert len(claims) == 1
    assert claims[0]["sustained_fraction"] < 0.60
    assert claims[0]["terminal_shared_tail_persistent"] is True


def test_first_sustained_overlap_ignores_crossing_and_finds_shared_tail():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    crossing = np.column_stack((np.arange(-15.0, 16.0), np.full(31, 15.0)))
    shared_tail = np.vstack(
        (
            np.column_stack((np.arange(8.0), np.full(8, -4.0))),
            horizontal[8:] + np.asarray((0.5, 0.0)),
        )
    )

    assert (
        first_sustained_overlap_index(
            crossing,
            horizontal,
            distance_px=1.0,
        )
        is None
    )
    assert first_sustained_overlap_index(
        shared_tail,
        horizontal,
        distance_px=1.0,
    ) == 8


def test_branch_capture_requires_a_long_aligned_foreign_lane():
    """Brief or perpendicular contacts cannot trigger foreign-branch capture."""

    reference = np.column_stack((np.zeros(40), np.arange(40, dtype=float)))
    captured = np.vstack(
        (
            np.column_stack((np.arange(10.0), np.full(10, -4.0))),
            reference[10:] + np.asarray((0.5, 0.0)),
        )
    )
    brief = np.vstack(
        (
            np.column_stack((np.arange(10.0), np.full(10, -4.0))),
            reference[10:15] + np.asarray((0.5, 0.0)),
            np.column_stack((np.arange(15.0, 25.0), np.full(10, 4.0))),
        )
    )
    crossing = np.column_stack((np.arange(-20.0, 20.0), np.full(40, 20.0)))

    assert sustained_branch_capture_index(
        captured,
        reference,
        distance_px=1.0,
        minimum_shared_length_px=12.0,
    ) == 10
    assert sustained_branch_capture_index(
        brief,
        reference,
        distance_px=1.0,
        minimum_shared_length_px=12.0,
    ) is None
    assert sustained_branch_capture_index(
        crossing,
        reference,
        distance_px=1.0,
        minimum_shared_length_px=12.0,
    ) is None


def test_branch_capture_recognizes_a_reversed_duplicate_lane():
    """Tube ownership is independent of centerline point ordering."""

    reference = np.column_stack((np.zeros(40), np.arange(40, dtype=float)))
    reversed_path = reference[::-1] + np.asarray((0.5, 0.0))

    assert sustained_branch_capture_index(
        reversed_path,
        reference,
        distance_px=1.0,
        minimum_shared_length_px=20.0,
    ) == 0


def _branch_candidate(
    owner,
    index,
    curve,
    *,
    selected=False,
    near_foreign=False,
    contact=False,
    growth=0.6,
):
    return CandidateBranchEvidence(
        owner_id=owner,
        candidate_index=index,
        label=f"candidate-{index}",
        causal_key=(1.0, growth, 0.8, 1.0),
        final_length_px=30.0,
        onset_sample=4,
        geometry_eligible=True,
        contact_censored=contact,
        near_foreign_owner=near_foreign,
        selected_independently=selected,
        curves_by_sample={sample: curve for sample in range(12)},
    )


def test_candidate_duplicate_fraction_matches_sustained_sparse_overlap():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    first = _branch_candidate(1, 0, horizontal)
    second = _branch_candidate(2, 0, horizontal + (1.0, 0.0))

    assert candidate_duplicate_fraction(first, second) == 1.0


def test_global_candidate_allocation_recovers_distinct_second_branch():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    vertical = np.column_stack((np.arange(31, dtype=float), np.zeros(31)))
    owners = {
        1: [
            _branch_candidate(1, 0, horizontal, selected=True, growth=0.8),
            _branch_candidate(1, 1, vertical, growth=0.5),
        ],
        2: [_branch_candidate(2, 0, horizontal, selected=True, growth=0.7)],
    }

    allocation = allocate_distinct_candidate_branches(owners)

    assert allocation[1].candidate_index == 1
    assert allocation[2].candidate_index == 0


def test_global_candidate_allocation_preserves_supported_independent_owner():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    vertical = np.column_stack((np.arange(31, dtype=float), np.zeros(31)))
    diagonal = np.column_stack(
        (np.arange(31, dtype=float), np.arange(31, dtype=float))
    )
    owners = {
        19: [
            _branch_candidate(19, 0, horizontal, growth=0.56),
            _branch_candidate(19, 1, vertical, growth=0.24),
        ],
        20: [_branch_candidate(20, 0, horizontal, near_foreign=True, growth=0.78)],
        78: [
            _branch_candidate(78, 0, horizontal, selected=True, growth=0.52),
            _branch_candidate(78, 1, diagonal, growth=0.23),
        ],
    }

    allocation = allocate_distinct_candidate_branches(owners)

    assert allocation[19].candidate_index == 1
    assert allocation[20] is None
    assert allocation[78].candidate_index == 0


def test_global_candidate_allocation_rejects_foreign_crossing_unless_censored():
    horizontal = np.column_stack((np.zeros(31), np.arange(31, dtype=float)))
    ordinary = _branch_candidate(1, 0, horizontal, near_foreign=True)
    contact = _branch_candidate(
        2,
        0,
        horizontal,
        near_foreign=True,
        contact=True,
    )

    allocation = allocate_distinct_candidate_branches({1: [ordinary], 2: [contact]})

    assert allocation[1] is None
    assert allocation[2] is contact


def test_unique_causal_root_wins_duplicate_component():
    owners = [
        _field_owner(40, [], onset=64),
        _field_owner(41, [], onset=0, status="review"),
    ]
    claims = [{"first_owner": 40, "second_owner": 41}]

    resolutions = resolve_ownership_components(owners, claims)

    assert resolutions == [
        {
            "owner_ids": [40, 41],
            "resolution": "unique-causal-root",
            "winner_owner_id": 40,
            "conflict_owner_ids": [41],
        }
    ]


def test_exact_censored_status_can_still_supply_unique_causal_root():
    owners = [
        _field_owner(8, [], onset=72, status="contact_censored"),
        _field_owner(9, [], onset=0, status="review_quality"),
    ]

    resolutions = resolve_ownership_components(
        owners,
        [{"first_owner": 8, "second_owner": 9}],
    )

    assert resolutions[0]["winner_owner_id"] == 8


def test_two_causal_roots_remain_explicit_ownership_conflict():
    owners = [
        _field_owner(19, [], onset=144),
        _field_owner(78, [], onset=148),
    ]
    claims = [{"first_owner": 19, "second_owner": 78}]

    resolutions = resolve_ownership_components(owners, claims)

    assert resolutions[0]["resolution"] == "ambiguous-owner"
    assert resolutions[0]["winner_owner_id"] is None
    assert resolutions[0]["conflict_owner_ids"] == [19, 78]


def test_unique_unobstructed_causal_path_wins_shared_distal_tube():
    owners = [
        _field_owner(19, [], onset=160, near_foreign=True),
        _field_owner(20, [], onset=76, near_foreign=True),
        _field_owner(78, [], onset=160, near_foreign=False),
    ]
    claims = [
        {"first_owner": 19, "second_owner": 78},
        {"first_owner": 20, "second_owner": 78},
    ]

    resolutions = resolve_ownership_components(owners, claims)

    assert resolutions == [
        {
            "owner_ids": [19, 20, 78],
            "resolution": "unique-unobstructed-causal-root",
            "winner_owner_id": 78,
            "conflict_owner_ids": [19, 20],
        }
    ]


def test_close_retained_pollen_remains_a_foreign_body_hypothesis():
    identity = {
        "source_frames": [0, 10],
        "diameter_px": 20.0,
        "tracks": [
            {
                "track_id": 1,
                "semantic_observation_count": 2,
                "source_frames": [0, 10],
                "centers_yx": [[10.0, 10.0], [10.0, 10.0]],
            },
            {
                "track_id": 2,
                "semantic_observation_count": 2,
                "source_frames": [0, 10],
                "centers_yx": [[10.0, 22.0], [10.0, 22.0]],
            },
            {
                "track_id": 3,
                "semantic_observation_count": 2,
                "source_frames": [0, 10],
                "centers_yx": [[10.0, 11.0], [10.0, 11.0]],
            },
        ],
    }

    centers, track_ids = _identity_body_hypotheses(
        identity,
        source_frame=10,
        owner_track_id=1,
        owner_center_yx=np.asarray((10.0, 10.0)),
        retained_track_ids={1, 2},
    )

    assert track_ids == (2,)
    assert np.array_equal(centers, ((10.0, 22.0),))


def test_unretained_late_round_detection_cannot_censor_an_existing_tube():
    identity = {
        "source_frames": [0, 10, 20],
        "diameter_px": 20.0,
        "tracks": [
            {
                "track_id": 1,
                "semantic_observation_count": 3,
                "source_frames": [0, 10, 20],
                "centers_yx": [[20.0, 20.0]] * 3,
            },
            {
                "track_id": 56,
                "semantic_observation_count": 2,
                "source_frames": [10, 20],
                "centers_yx": [[20.0, 45.0], [20.0, 55.0]],
            },
        ],
    }

    centers, track_ids = _identity_body_hypotheses(
        identity,
        source_frame=20,
        owner_track_id=1,
        owner_center_yx=np.asarray((20.0, 20.0)),
        retained_track_ids={1},
    )

    assert track_ids == ()
    assert centers.shape == (0, 2)


def test_causal_atlas_loader_keeps_only_accepted_paths_and_rescales(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps({"analysis_width": 200, "input_movie": "movie.mp4"})
    )
    with (tmp_path / "owner_paths.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "owner_track_id",
                "path_index",
                "y_analysis_px",
                "x_analysis_px",
                "accepted",
                "reason",
            )
        )
        writer.writerow((3, 1, 12.0, 22.0, True, "accepted"))
        writer.writerow((3, 0, 10.0, 20.0, True, "accepted"))
        writer.writerow((4, 0, 30.0, 40.0, False, "weak-paired-walls"))
        writer.writerow((5, 0, 50.0, 60.0, False, "duplicate-branch-claim"))
        writer.writerow((5, 1, 52.0, 62.0, False, "duplicate-branch-claim"))

    paths, report = load_causal_atlas_paths(tmp_path, {3, 4}, 400)

    assert report["input_movie"] == "movie.mp4"
    assert set(paths) == {3}
    assert np.array_equal(paths[3], ((20.0, 40.0), (24.0, 44.0)))

    review_paths, _ = load_causal_atlas_paths(tmp_path, {5}, 200)
    assert np.array_equal(review_paths[5], ((50.0, 60.0), (52.0, 62.0)))


def test_owner_motion_cache_selects_requested_ids_and_quality(tmp_path):
    cache_path = tmp_path / "motion.npz"
    frames = np.asarray([0, 10, 20])
    tracks = np.asarray(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[11, 12], [13, 14], [15, 16]],
        ],
        dtype=float,
    )
    np.savez(
        cache_path,
        source_frames=frames,
        track_ids=np.asarray([8, 3]),
        analysis_width=np.asarray(480),
        multipoint_aligned_yx=tracks,
        template_aligned_yx=tracks + 100,
        observed=np.asarray([[True, False, True], [True, True, True]]),
        inlier_counts=np.asarray([[9, 0, 6], [9, 3, 9]]),
    )

    owner_ids, selected, fallback, quality = load_owner_motion_cache(
        cache_path, frames, analysis_width=480
    )

    assert owner_ids == [8, 3]
    assert np.array_equal(selected, tracks)
    assert np.array_equal(fallback, tracks + 100)
    assert np.allclose(quality[0], (1.0, 0.0, 2.0 / 3.0))
    assert np.allclose(quality[1], (1.0, 1.0 / 3.0, 1.0))


def test_owner_motion_cache_maps_native_source_tracks_to_analysis_space(tmp_path):
    cache_path = tmp_path / "native-motion.npz"
    frames = np.asarray([0, 10])
    source_tracks = np.asarray([[[20.0, 40.0], [24.0, 44.0]]])
    np.savez(
        cache_path,
        source_frames=frames,
        track_ids=np.asarray([5]),
        analysis_width=np.asarray(1280),
        multipoint_aligned_yx=source_tracks,
        multipoint_source_yx=source_tracks,
        template_aligned_yx=source_tracks + 10,
        template_source_yx=source_tracks + 10,
        observed=np.ones((1, 2), dtype=bool),
        inlier_counts=np.full((1, 2), 9),
    )

    owner_ids, tracks, fallback, _ = load_owner_motion_cache(
        cache_path,
        frames,
        analysis_width=480,
        shifts_xy=np.asarray([[0.0, 0.0], [2.0, 4.0]]),
        analysis_scale=0.375,
    )

    assert owner_ids == [5]
    assert np.allclose(tracks[0], ((7.5, 15.0), (5.0, 14.5)))
    assert np.allclose(fallback[0], ((11.25, 18.75), (8.75, 18.25)))


def test_field_context_is_validated_and_reused(tmp_path, monkeypatch):
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie identity")
    cache = tmp_path / "field"
    calls = []

    def fake_schedule(path, interval):
        calls.append("schedule")
        return np.asarray([0, 5, 9]), 2.0, 10, 20, 12

    def fake_load(path, frames, width, native_width, native_height):
        calls.append("decode")
        return np.arange(3 * 6 * 10, dtype=np.uint8).reshape(3, 6, 10)

    def fake_stabilize(frames):
        calls.append("register")
        return frames + 1, np.zeros((3, 2), dtype=np.float32), np.ones(3)

    monkeypatch.setattr(v27_track, "movie_schedule", fake_schedule)
    monkeypatch.setattr(v27_track, "load_grayscale_samples", fake_load)
    monkeypatch.setattr(v27_track, "stabilize_translations", fake_stabilize)

    created = prepare_field_context(cache, movie, 10, 2.5)
    reused = prepare_field_context(cache, movie, 10, 2.5)

    assert calls == ["schedule", "decode", "register"]
    assert np.array_equal(created.source_frames, (0, 5, 9))
    assert np.array_equal(created.aligned_frames, reused.aligned_frames)
    assert created.aligned_frames.flags.writeable is False
    try:
        load_field_context(cache, movie, 12, 2.5)
    except ValueError as exc:
        assert "analysis_width" in str(exc)
    else:
        raise AssertionError("a mismatched field cache must be rejected")


def test_ordered_prefix_error_exposes_a_crossover_handoff():
    owned = _horizontal(50)
    foreign = _branch(50, 25)

    assert ordered_prefix_error(owned, owned) == 0.0
    assert ordered_prefix_error(owned, foreign) > 5.0


def test_complete_timeline_beats_a_stronger_final_foreign_branch():
    frames = []
    for length in (20, 28, 36, 44, 52):
        frames.append((_horizontal(length, quality=0.8),))
    frames.append(
        (
            _branch(60, 36, quality=1.4, source=1),
            _horizontal(60, quality=0.8, source=2),
        )
    )

    result = select_global_ribbon_worldsheet(frames)

    assert result.candidate_indices[-1] == 1
    assert result.selected[-1].source_index == 2
    assert all(state == "path" for state in result.states)


def test_worldsheet_finds_germination_and_bridges_one_missing_observation():
    frames = [
        (_branch(18, 7, quality=0.05),),
        (_branch(22, 6, quality=0.08),),
        (_branch(19, 9, quality=0.06),),
        (_horizontal(12, quality=0.9),),
        (_horizontal(18, quality=0.95),),
        (),
        (_horizontal(30, quality=1.0),),
    ]

    result = select_global_ribbon_worldsheet(
        frames,
        GlobalRibbonConfig(maximum_gap_samples=1),
    )

    assert result.first_active_sample == 3
    assert result.states[:3] == ("prebirth", "prebirth", "prebirth")
    assert result.states[5] == "gap"
    assert result.states[-1] == "path"


def test_field_validation_track_ids_are_positive_unique_and_ordered():
    assert parse_track_ids("3,1,3,8") == [3, 1, 8]

    try:
        parse_track_ids("3,0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonpositive track IDs must be rejected")


def test_field_validation_loads_valid_mature_seed_overrides(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"6": " rim-2-contact-censored ", "19": "rim-5"}))

    assert load_mature_seed_overrides(path) == {
        6: "rim-2-contact-censored",
        19: "rim-5",
    }

    path.write_text(json.dumps(["rim-5"]))
    with pytest.raises(ValueError, match="JSON object"):
        load_mature_seed_overrides(path)


def test_field_validation_resumes_new_outcome_classes(tmp_path):
    owner_dir = tmp_path / "P17"
    owner_dir.mkdir()
    report = {
        "revision": EXPECTED_REVISION,
        "measured_owner_ids": [],
        "review_owner_ids": [],
        "unavailable_owner_ids": [],
        "no_growth_owner_ids": [17],
    }
    (owner_dir / "report.json").write_text(json.dumps(report))

    assert completed_report(owner_dir, 17) == report
    assert measurement_status(report, 17) == "no_growth"
    assert completed_report(owner_dir, 18) is None

    report["revision"] = MULTIPOINT_EXPECTED_REVISION
    (owner_dir / "report.json").write_text(json.dumps(report))
    assert (
        completed_report(owner_dir, 17, MULTIPOINT_EXPECTED_REVISION) == report
    )


def _legacy_multipoint_report(endpoint_efficiency, excursion_radii):
    return {
        "revision": LEGACY_MULTIPOINT_REVISION,
        "measured_owner_ids": [18],
        "review_owner_ids": [],
        "unavailable_owner_ids": [],
        "no_growth_owner_ids": [],
        "worldsheet_diagnostics": {
            "18": {
                "mature_seed_audits": [
                    {
                        "geometry_eligible": True,
                        "endpoint_radial_efficiency": endpoint_efficiency,
                        "radial_excursion_radii": excursion_radii,
                    }
                ]
            }
        },
    }


def test_v28_cache_compatibility_rejects_changed_rim_geometry_band():
    affected = _legacy_multipoint_report(0.51, 1.68)
    unaffected = _legacy_multipoint_report(0.80, 1.68)

    assert not compatible_multipoint_report(
        affected,
        RIM_GEOMETRY_REVISION,
    )
    assert compatible_multipoint_report(
        unaffected,
        RIM_GEOMETRY_REVISION,
    )


def test_v283_cache_promotion_requires_baseline_and_monotone_compatibility(
    tmp_path,
):
    later = {
        "revision": SEMANTIC_GUARDED_REVISION,
        "worldsheet_diagnostics": {
            "4": {"first_active_sample": 8, "measurement_status": "measured"}
        },
    }
    contact = {
        "revision": SEMANTIC_GUARDED_REVISION,
        "worldsheet_diagnostics": {
            "9": {
                "first_active_sample": 0,
                "measurement_status": "contact_censored",
            }
        },
    }
    frame_zero = {
        "revision": SEMANTIC_GUARDED_REVISION,
        "worldsheet_diagnostics": {
            "24": {"first_active_sample": 0, "measurement_status": "measured"}
        },
    }

    (tmp_path / "measurements.csv").write_text(
        "accepted,tube_length_px\n1,2.0\n1,3.0\n"
    )

    assert compatible_multipoint_report(
        later,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )
    assert compatible_multipoint_report(
        contact,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )
    assert not compatible_multipoint_report(
        frame_zero,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )
    (tmp_path / "measurements.csv").write_text(
        "accepted,tube_length_px\n1,3.0\n1,2.0\n"
    )
    assert not compatible_multipoint_report(
        later,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )


def test_v284_cache_promotion_reruns_only_nonmonotone_history(tmp_path):
    report = {"revision": BASELINE_LIFECYCLE_REVISION}
    measurements = tmp_path / "measurements.csv"
    measurements.write_text("accepted,tube_length_px\n1,2.0\n1,3.0\n")

    assert measurement_lengths_are_monotone(measurements)
    assert compatible_multipoint_report(
        report,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )

    measurements.write_text("accepted,tube_length_px\n1,3.0\n1,2.0\n")
    assert not measurement_lengths_are_monotone(measurements)
    assert not compatible_multipoint_report(
        report,
        MULTIPOINT_EXPECTED_REVISION,
        output_dir=tmp_path,
    )


def test_aggregate_validation_checks_growth_and_centerline_invariants(tmp_path):
    (tmp_path / "summary.csv").write_text(
        "pollen_id,final_length_px\n1,3.0\n"
    )
    (tmp_path / "measurements.csv").write_text(
        "pollen_id,sample_index,tube_length_px,accepted\n"
        "1,0,2.0,1\n1,1,3.0,1\n"
    )
    (tmp_path / "centerlines.csv").write_text(
        "pollen_id,sample_index,arc_length_px\n"
        "1,0,0.0\n1,0,2.0\n1,1,0.0\n1,1,3.0\n"
    )

    result = validate_combined_outputs(tmp_path)

    assert result["owner_count"] == 1
    assert result["nondecreasing_owner_count"] == 1
    assert result["maximum_summary_final_error_px"] == 0.0
    assert result["maximum_centerline_length_error_px"] == 0.0


def test_aggregate_validation_rejects_backward_material_growth(tmp_path):
    (tmp_path / "summary.csv").write_text(
        "pollen_id,final_length_px\n1,2.0\n"
    )
    (tmp_path / "measurements.csv").write_text(
        "pollen_id,sample_index,tube_length_px,accepted\n"
        "1,0,3.0,1\n1,1,2.0,1\n"
    )
    (tmp_path / "centerlines.csv").write_text(
        "pollen_id,sample_index,arc_length_px\n"
    )

    with pytest.raises(ValueError, match="backward accepted length"):
        validate_combined_outputs(tmp_path)


def test_completed_report_promotes_only_evidence_compatible_v28_cache(tmp_path):
    owner_dir = tmp_path / "P18"
    owner_dir.mkdir()
    report = _legacy_multipoint_report(0.80, 1.68)
    (owner_dir / "report.json").write_text(json.dumps(report))
    (owner_dir / "summary.csv").write_text(
        "pollen_id,revision\n18,v28.0-cotracker-rigid-owner-motion\n"
    )

    promoted = completed_report(owner_dir, 18, RIM_GEOMETRY_REVISION)

    assert promoted["revision"] == RIM_GEOMETRY_REVISION
    assert promoted["compatible_from_revision"] == LEGACY_MULTIPOINT_REVISION
    assert RIM_GEOMETRY_REVISION in (owner_dir / "summary.csv").read_text()


def test_field_validation_localizes_artifacts_after_copy(tmp_path):
    owner_dir = tmp_path / "P03"
    owner_dir.mkdir()
    report = {"artifacts": {"review_video": "/old/control/P03/review.mov"}}

    localized = localize_report_artifacts(owner_dir, report)

    expected = str(owner_dir / "review.mov")
    assert localized["artifacts"]["review_video"] == expected
    assert json.loads((owner_dir / "report.json").read_text()) == localized
