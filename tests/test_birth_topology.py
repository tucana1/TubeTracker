"""Tests for local birth persistence and simple pollen-anchored paths."""

import csv

from types import SimpleNamespace

import cv2 as cv
import numpy as np

from prototypes.v17_birth_topology.track import (
    AtlasConfig,
    GrainAnchor,
    _apply_broad_occlusion_review,
    _apply_contact_bridge_recovery,
    _apply_full_path_temporal_validation,
    _apply_path_locked_front_refinement,
    _apply_precontact_trajectory_recovery,
    _apply_proximal_emergence_review,
    _apply_rim_bridged_proximal_recovery,
    _apply_rim_direction_specificity_review,
    _birth_order_metrics,
    _connected_proximal_emergence_sample,
    _component_width_px,
    _contact_bridge_geometry,
    _event_audit_samples,
    _germination_onset_assessment,
    _germination_phase,
    _grain_anchor_sample_indices,
    _maximum_junction_turn_degrees,
    _oriented_rim_contrast_metrics,
    _onset_timing_scope,
    _prepend_grain_rim_connector,
    _promote_resolved_short_trajectories,
    _quality_classification,
    _apply_warmup_onset_review,
    _resolve_component_ownership,
    _resolve_grain_event_ambiguity,
    _rim_emergence_metrics,
    _select_birth_ordered_path,
    _skeleton_path_candidates,
    _tip_timing_assessment,
    _trajectory_measurement_scope,
    _trim_grain_rim_prefix,
    _warmup_path_prefix_length,
    broad_occlusion_mask,
    event_tip_at,
    detect_grain_anchors,
    extract_birth_events,
    local_persistent_births,
    ridge_evidence,
    sample_interval_seconds,
    source_frame_indices,
    write_event_timeline_audit,
    write_blinded_validation_audit,
    write_front_refinement_audit,
    write_proximal_emergence_audit,
    write_rim_direction_specificity_audit,
    write_trajectory_timeline_audit,
)
from prototypes.v18_phase_consensus.track import (
    ConsensusCandidate,
    CrossPhaseAssessment,
    PhaseAnalysis,
    PhaseConsensusConfig,
    branch_locked_path_offsets,
    compare_tip_lengths,
    candidate_audit_source_frames,
    consensus_tip_at,
    fuse_tip_birth_times,
    measurable_growth_onset_interval,
    onset_cue_agreement_report,
    persistent_profile_births,
    local_path_birth_observations,
    nearest_grain,
    paths_share_rooted_prefix,
    write_summary,
)


class LocalPersistenceTests:
    def test_phase_consensus_compares_slow_growth_in_length_domain(self):
        arclength = np.arange(5, dtype=np.float64)
        source = SimpleNamespace(source_frames=np.arange(0, 101, 10))
        target = SimpleNamespace(source_frames=np.arange(5, 106, 10))
        source_births = np.asarray([10, 30, 50, 70, 90])
        target_births = np.asarray([20, 40, 60, 80, 100])

        agreement, median_px, p90_px, maximum_px = compare_tip_lengths(
            arclength,
            source,
            source_births,
            target,
            target_births,
            tolerance_px=1.0,
        )

        assert agreement == 1.0
        assert median_px <= 1.0
        assert p90_px <= 1.0
        assert maximum_px == 1.0

    def test_measurable_growth_onset_retains_phase_uncertainty(self):
        midpoint, lower, upper, uncertainty_s = measurable_growth_onset_interval(
            np.asarray([0.0, 2.0, 4.0, 6.0]),
            3.0,
            np.asarray([10, 20, 30, 40]),
            np.asarray([12, 22, 38, 48]),
            seconds_per_source_frame=0.5,
        )

        assert (midpoint, lower, upper) == (34, 30, 38)
        assert uncertainty_s == 4.0

    def test_branch_locked_registration_tapers_pollen_motion(self):
        sample_count = 36
        point_count = 24
        path = np.column_stack(
            (
                np.arange(10.0, 10.0 + point_count),
                np.full(point_count, 22.0),
            )
        )
        births = 8 + np.arange(point_count, dtype=np.int32) // 2
        grain_motion = np.clip(
            (np.arange(sample_count, dtype=np.float64) - 12.0) / 8.0,
            0.0,
            1.0,
        ) * 4.0
        centers = np.column_stack(
            (np.full(sample_count, 5.0), 5.0 + grain_motion)
        )
        grain = GrainAnchor(
            1,
            centers[0],
            5.0,
            sample_count,
            0.9,
            sample_indices=np.arange(sample_count),
            centers_xy=centers,
        )
        falloff = 10.0
        arc = np.arange(point_count, dtype=np.float64)
        influence = np.clip(1.0 - arc / falloff, 0.0, 1.0)
        influence = influence * influence * (3.0 - 2.0 * influence)
        normal_bend = np.zeros(point_count, dtype=np.float64)
        normal_bend[3:] = np.minimum(2.0, (arc[3:] - 2.0) / 4.0)
        evidence = np.zeros((sample_count, 48, 64), dtype=np.uint8)
        for sample in range(sample_count):
            born = np.flatnonzero(births <= sample)
            if not len(born):
                continue
            total_y = grain_motion[sample] * influence[born] + normal_bend[born]
            points = np.rint(
                path[born] + np.column_stack((np.zeros(len(born)), total_y))
            ).astype(int)
            evidence[sample, points[:, 1], points[:, 0]] = 20
        phase = PhaseAnalysis(
            name="synthetic",
            source_frames=np.arange(sample_count),
            aligned=np.zeros_like(evidence),
            shifts=np.zeros((sample_count, 2), dtype=np.float32),
            responses=np.ones(sample_count, dtype=np.float32),
            grains=[grain],
            evidence=evidence,
            birth=np.full(evidence.shape[1:], sample_count, dtype=np.int32),
            events=[],
            calibration={"global_evidence_floor": 6.0},
        )
        atlas_config = AtlasConfig(warmup_samples=8)
        consensus_config = PhaseConsensusConfig(
            tip_timing_tolerance_intervals=2.0,
            grain_motion_falloff_px=falloff,
            registration_birth_lead_samples=1,
        )

        offsets = branch_locked_path_offsets(
            phase,
            path,
            grain,
            0,
            births,
            atlas_config,
            consensus_config,
        )
        registered = path + offsets[-1]

        assert np.allclose(offsets[:, 0, 1], grain_motion)
        assert np.median(np.abs(registered[12:, 1] - 24.0)) <= 1.0
        assert np.median(np.abs(registered[12:, 1] - 26.0)) >= 1.0

    def test_phase_consensus_profile_births_follow_persistence_and_preexistence(self):
        profiles = np.full((12, 2), -1.0, dtype=np.float32)
        profiles[5:, 0] = 1.0
        profiles[:2, 1] = 1.0

        births, present = persistent_profile_births(
            profiles,
            warmup_samples=4,
            window=3,
            required=2,
            preexisting_min_observations=2,
        )

        assert births.tolist() == [6, 0]
        assert present.tolist() == [True, False]

    def test_phase_consensus_fuses_tip_times_in_source_frame_units(self):
        source = SimpleNamespace(source_frames=np.asarray([0, 10, 20, 30]))
        target = SimpleNamespace(source_frames=np.asarray([5, 15, 25, 35]))

        source_frames, target_frames, fused, agreement, median_s, maximum_s = (
            fuse_tip_birth_times(
                source,
                np.asarray([1, 2, 3]),
                target,
                np.asarray([0, 0, 3]),
                seconds_per_source_frame=0.1,
                tolerance_intervals=1.0,
            )
        )

        assert source_frames.tolist() == [10, 20, 30]
        assert target_frames.tolist() == [5, 5, 35]
        assert fused.tolist() == [8, 12, 32]
        assert agreement == 2 / 3
        assert median_s == 0.5
        assert maximum_s == 1.5

    def test_phase_consensus_tip_uses_fused_timeline(self):
        candidate = SimpleNamespace(
            consensus_tip_birth_source_frames=np.asarray([10, 20, 30]),
            event=SimpleNamespace(arclength_px=np.asarray([0.0, 2.0, 5.0])),
            path_reference_xy=np.asarray(
                [[4.0, 5.0], [6.0, 5.0], [9.0, 5.0]]
            ),
        )

        assert consensus_tip_at(candidate, 9) == (0.0, None, -1)
        length, tip, index = consensus_tip_at(candidate, 25)
        assert length == 2.0
        assert tip.tolist() == [6.0, 5.0]
        assert index == 1

    def test_phase_consensus_samples_neighboring_birth_evidence(self):
        birth = np.full((12, 12), 40, dtype=np.int32)
        birth[5, 6] = 17
        observed, present = local_path_birth_observations(
            birth,
            np.asarray([[5.0, 5.0], [9.0, 9.0]]),
            warmup_samples=8,
            neighborhood_px=1,
        )

        assert present.tolist() == [True, False]
        assert observed[0] == 17
        assert np.isnan(observed[1])

    def test_phase_consensus_grain_match_is_not_forced(self):
        source = GrainAnchor(1, np.asarray([10.0, 10.0]), 5.0, 3, 0.8)
        near = GrainAnchor(2, np.asarray([12.0, 10.0]), 5.0, 3, 0.8)
        far = GrainAnchor(3, np.asarray([30.0, 30.0]), 5.0, 3, 0.8)

        match, distance = nearest_grain(source, [far, near], 8.0)
        assert match is near
        assert distance == 2.0
        match, distance = nearest_grain(source, [far], 8.0)
        assert match is None
        assert distance > 8.0

    def test_phase_consensus_accepts_only_a_shared_rooted_prefix(self):
        reference = np.column_stack(
            [np.arange(10, dtype=np.float64), np.zeros(10)]
        )
        same_prefix = np.column_stack(
            [np.arange(7, dtype=np.float64), np.full(7, 0.5)]
        )
        crossing_branch = np.column_stack(
            [np.zeros(10), np.arange(10, dtype=np.float64)]
        )

        assert paths_share_rooted_prefix(reference, same_prefix, 1.0, 0.8)
        assert not paths_share_rooted_prefix(
            reference,
            crossing_branch,
            1.0,
            0.8,
        )

    def test_phase_consensus_summary_uses_path_arclength(self, tmp_path):
        event = SimpleNamespace(
            event_id=4,
            grain_id=2,
            auto_accepted=True,
            trajectory_accepted=True,
            contact_bridge_trajectory_accepted=False,
            precontact_trajectory_accepted=False,
            quality_status="trajectory-growth",
            quality_flags=(),
            arclength_px=np.asarray([0.0, 2.5, 4.0]),
            birth_start=6,
            rim_emergence_sample=7,
            rim_contrast_emergence_sample=8,
        )
        cross = CrossPhaseAssessment(
            target_grain_id=3,
            grain_match_distance_px=1.0,
            path_birth_coverage=1.0,
            eventual_path_support_fraction=1.0,
            birth_order_correlation=1.0,
            birth_order_forward_fraction=1.0,
            birth_progress_samples=8.0,
            growth_step_count=6,
            tip_step_median_px=1.0,
            tip_step_max_px=1.5,
            path_supported=True,
            trajectory_supported=True,
            onset_supported=True,
            onset_quality="high-agreement",
            onset_sample=10,
            onset_lower_sample=9,
            onset_upper_sample=11,
            reason="ok",
            root_birth_sample=9,
            rim_emergence_sample=10,
            rim_contrast_emergence_sample=11,
        )
        grain = GrainAnchor(2, np.asarray([5.0, 5.0]), 4.0, 3, 0.9)
        source_phase = SimpleNamespace(
            source_frames=np.arange(100, 130, 2, dtype=np.int32),
        )
        target_phase = SimpleNamespace(
            source_frames=np.arange(101, 131, 2, dtype=np.int32),
        )
        candidate = ConsensusCandidate(
            source_phase="A",
            event=event,
            source_grain=grain,
            target_phase=target_phase,
            target_grain=grain,
            cross=cross,
            canonical_grain_id=2,
            path_reference_xy=np.asarray([[5.0, 5.0], [9.0, 5.0]]),
            source_onset=SimpleNamespace(sample=9),
            source_analysis=source_phase,
        )

        output = tmp_path / "summary.csv"
        write_summary(output, [candidate])

        with output.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert float(row["source_final_length_px"]) == 4.0
        assert row["source_root_onset_source_frame"] == "112"
        assert row["source_rim_onset_source_frame"] == "114"
        assert row["source_contrast_onset_source_frame"] == "116"
        assert row["source_consensus_onset_source_frame"] == "118"
        assert row["target_root_onset_source_frame"] == "119"
        assert row["target_rim_onset_source_frame"] == "121"
        assert row["target_contrast_onset_source_frame"] == "123"
        assert row["target_consensus_onset_source_frame"] == "121"

    def test_onset_cue_report_compares_only_trusted_trajectory_phases(self):
        phase = SimpleNamespace(source_frames=np.arange(100, 200, dtype=np.int32))
        candidate = SimpleNamespace(
            selected=True,
            trajectory_accepted=True,
            source_analysis=phase,
            target_phase=phase,
            event=SimpleNamespace(
                birth_start=10,
                rim_emergence_sample=20,
                rim_contrast_emergence_sample=30,
            ),
            source_onset=SimpleNamespace(sample=40),
            cross=SimpleNamespace(
                root_birth_sample=12,
                rim_emergence_sample=None,
                rim_contrast_emergence_sample=33,
                onset_sample=46,
            ),
        )
        rejected = SimpleNamespace(selected=True, trajectory_accepted=False)

        report = onset_cue_agreement_report(
            [candidate, rejected],
            seconds_per_source_frame=0.5,
            tolerance_seconds=2.0,
        )

        assert report["trajectory_count"] == 1
        assert report["cues"]["connected_root"]["median_disagreement_s"] == 1.0
        assert report["cues"]["connected_root"]["within_tolerance_fraction"] == 1.0
        assert report["cues"]["pollen_rim"]["paired_count"] == 0
        assert report["cues"]["path_contrast"]["median_disagreement_s"] == 1.5
        assert report["cues"]["phase_consensus"]["within_tolerance_fraction"] == 0.0

    def test_candidate_audit_frames_include_pre_growth_and_path_milestones(self):
        candidate = SimpleNamespace(
            consensus_tip_birth_source_frames=np.asarray(
                [100, 120, 140, 160, 180],
                dtype=np.int32,
            )
        )

        frames = candidate_audit_source_frames(
            candidate,
            source_start=50,
            source_end=170,
            before_source_frames=30,
        )

        assert frames == [70, 100, 120, 140, 160, 170]

    def test_germination_phase_separates_onset_from_measurement(self):
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            rim_emergence_sample=5,
            germination_sample=8,
        )

        assert _germination_phase(event, 4) == "pre-germination"
        assert _germination_phase(event, 5) == "emerged-below-measurement"
        assert _germination_phase(event, 7) == "emerged-below-measurement"
        assert _germination_phase(event, 8) == "measurable-growth"

    def test_germination_phase_exposes_cue_order_disagreement(self):
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            rim_emergence_sample=8,
            germination_sample=5,
        )

        assert _germination_phase(event, 4) == "pre-germination"
        assert _germination_phase(event, 5) == "measurable-before-onset-cue"
        assert _germination_phase(event, 7) == "measurable-before-onset-cue"
        assert _germination_phase(event, 8) == "measurable-growth"

    def test_germination_phase_withholds_unaccepted_or_missing_onsets(self):
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=False,
            rim_emergence_sample=5,
            germination_sample=8,
        )
        assert _germination_phase(event, 8) == "review-required"

        event.auto_accepted = True
        event.rim_emergence_sample = None
        assert _germination_phase(event, 8) == "onset-unavailable"

    def test_germination_onset_assessment_requires_agreeing_cues(self):
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            rim_emergence_sample=20,
            rim_contrast_emergence_sample=21,
            birth_start=22,
            germination_sample=30,
        )
        config = AtlasConfig(
            rim_persistence_window=5,
            rim_history_samples=12,
        )

        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "high-agreement"
        assert assessment.accepted
        assert assessment.sample == 21
        assert assessment.supporting_cues == ("root", "rim", "contrast")
        assert _onset_timing_scope(event, assessment) == "point-estimate"

        event.birth_start = 28
        event.rim_contrast_emergence_sample = 25
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "moderate-agreement"
        assert assessment.accepted

        event.birth_start = 40
        event.rim_contrast_emergence_sample = 38
        event.germination_sample = 50
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "contrast-supported"
        assert assessment.accepted
        assert assessment.sample == 39
        assert assessment.supporting_cues == ("root", "contrast")

        event.birth_start = 40
        event.rim_emergence_sample = 20
        event.rim_contrast_emergence_sample = 23
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "rim-contrast-supported"
        assert assessment.accepted
        assert assessment.sample == 22
        assert assessment.supporting_cues == ("rim", "contrast")

        event.rim_contrast_emergence_sample = 60
        event.germination_sample = 70
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "low-agreement"
        assert not assessment.accepted
        assert assessment.supporting_cues == ("root", "rim", "contrast")
        assert assessment.support_start_sample == 20
        assert assessment.support_end_sample == 60
        assert _onset_timing_scope(event, assessment) == (
            "optical-review-window"
        )

        event.rim_contrast_emergence_sample = None
        event.germination_sample = 50
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "low-agreement"
        assert not assessment.accepted
        assert assessment.sample is None
        assert _germination_phase(event, 25, assessment) == "onset-timing-review"
        assert _germination_phase(event, 50, assessment) == "measurable-growth"

        event.birth_start = 31
        event.rim_emergence_sample = 31
        event.germination_sample = 30
        assessment = _germination_onset_assessment(event, config)
        assert assessment.quality == "reversed-cue-order"
        assert not assessment.accepted

        event.rim_emergence_sample = None
        unavailable = _germination_onset_assessment(event, config)
        assert unavailable.quality == "unavailable"
        assert _onset_timing_scope(event, unavailable) == "unavailable"
        event.auto_accepted = False
        rejected = _germination_onset_assessment(event, config)
        assert rejected.quality == "review-required"
        assert _onset_timing_scope(event, rejected) == "not-accepted"

    def test_oriented_rim_contrast_rejects_uniform_change(self):
        evidence = np.zeros((20, 30, 40), dtype=np.uint8)
        evidence[12:, 14:17, 15:21] = 10
        grain = GrainAnchor(
            grain_id=1,
            center_xy=np.asarray([10.0, 15.0]),
            radius_px=5.0,
            observations=3,
            circle_score=0.9,
        )
        path = np.column_stack([np.arange(15, 21), np.full(6, 15)])
        config = AtlasConfig(
            rim_history_samples=8,
            rim_persistence_window=5,
            rim_persistence_required=3,
            rim_contrast_min_delta=2.0,
        )

        _, _, _, confirmed, sample = _oriented_rim_contrast_metrics(
            evidence,
            grain,
            path,
            sample=16,
            config=config,
        )
        assert confirmed
        assert sample == 14

        uniform = np.zeros_like(evidence)
        uniform[12:] = 10
        _, _, _, confirmed, sample = _oriented_rim_contrast_metrics(
            uniform,
            grain,
            path,
            sample=16,
            config=config,
        )
        assert not confirmed
        assert sample is None

    def test_component_width_uses_all_branches_not_one_selected_path(self):
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[15, 1:21] = True
        skeleton[5:15, 10] = True

        assert np.isclose(_component_width_px(150, skeleton), 5.0)

    def test_warmup_prefix_requires_support_connected_from_the_root(self):
        support = np.zeros((20, 30), dtype=bool)
        support[10, 5:11] = True
        support[10, 18:23] = True
        path_xy = np.column_stack([np.arange(5, 23), np.full(18, 10)])
        arclength = np.arange(18, dtype=float)

        length = _warmup_path_prefix_length(
            path_xy,
            arclength,
            support,
            maximum_gap_px=2.0,
        )

        assert length == 5.0

    def test_left_censored_germination_preserves_valid_tip_trajectory(
        self, monkeypatch
    ):
        evidence = np.zeros((8, 20, 30), dtype=np.uint8)
        evidence[:2, 10, 5:14] = 20
        frames = np.full_like(evidence, 170)
        event = SimpleNamespace(
            path_xy=np.column_stack([np.arange(5, 15), np.full(10, 10)]),
            arclength_px=np.arange(10, dtype=float),
            auto_accepted=True,
            trajectory_accepted=True,
            quality_status="trajectory-growth",
            quality_flags=(),
            warmup_prefix_length_px=0.0,
            warmup_ribbon_fraction=0.0,
            germination_left_censored=False,
            warmup_attachment_ambiguous=False,
        )
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.observe_material_ribbon",
            lambda *args, **kwargs: SimpleNamespace(supported_fraction=0.8),
        )

        _apply_warmup_onset_review(
            [event],
            evidence,
            frames,
            AtlasConfig(
                warmup_samples=8,
                preexisting_min_observations=2,
                spatial_link_px=1,
                grain_exit_margin_px=1.0,
                min_germination_length_px=2.0,
            ),
        )

        assert event.germination_left_censored
        assert not event.auto_accepted
        assert event.trajectory_accepted
        assert event.quality_status == "left-censored-trajectory"
        assert "left-censored-germination" in event.quality_flags

    def test_contact_path_uses_stricter_warmup_ribbon_review(
        self, monkeypatch
    ):
        evidence = np.zeros((8, 20, 30), dtype=np.uint8)
        frames = np.full_like(evidence, 170)
        event = SimpleNamespace(
            path_xy=np.column_stack([np.arange(5, 15), np.full(10, 10)]),
            arclength_px=np.arange(10, dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=("foreign-grain-contact",),
            warmup_prefix_length_px=0.0,
            warmup_ribbon_fraction=0.0,
            germination_left_censored=False,
            warmup_attachment_ambiguous=False,
        )
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.observe_material_ribbon",
            lambda *args, **kwargs: SimpleNamespace(supported_fraction=0.32),
        )

        _apply_warmup_onset_review(
            [event],
            evidence,
            frames,
            AtlasConfig(warmup_samples=8),
        )

        assert not event.auto_accepted
        assert event.warmup_attachment_ambiguous
        assert "warmup-attachment-ambiguous" in event.quality_flags

    def test_single_warmup_cue_withholds_onset_as_ambiguous(self, monkeypatch):
        evidence = np.zeros((8, 20, 30), dtype=np.uint8)
        evidence[:2, 10, 5:14] = 20
        frames = np.full_like(evidence, 170)
        event = SimpleNamespace(
            path_xy=np.column_stack([np.arange(5, 15), np.full(10, 10)]),
            arclength_px=np.arange(10, dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=(),
            warmup_prefix_length_px=0.0,
            warmup_ribbon_fraction=0.0,
            germination_left_censored=False,
            warmup_attachment_ambiguous=False,
        )
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.observe_material_ribbon",
            lambda *args, **kwargs: SimpleNamespace(supported_fraction=0.1),
        )

        _apply_warmup_onset_review(
            [event],
            evidence,
            frames,
            AtlasConfig(
                warmup_samples=8,
                preexisting_min_observations=2,
                spatial_link_px=1,
                grain_exit_margin_px=1.0,
                min_germination_length_px=2.0,
            ),
        )

        assert not event.germination_left_censored
        assert event.warmup_attachment_ambiguous
        assert not event.auto_accepted
        assert event.quality_status == "warmup-ambiguous-growth"
        assert "warmup-attachment-ambiguous" in event.quality_flags

    def test_warmup_ribbon_alone_withholds_a_preexisting_path(self, monkeypatch):
        evidence = np.zeros((8, 20, 30), dtype=np.uint8)
        frames = np.full_like(evidence, 170)
        event = SimpleNamespace(
            path_xy=np.column_stack([np.arange(5, 15), np.full(10, 10)]),
            arclength_px=np.arange(10, dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="atlas-event",
            quality_flags=(),
            warmup_prefix_length_px=0.0,
            warmup_ribbon_fraction=0.0,
            germination_left_censored=False,
            warmup_attachment_ambiguous=False,
        )
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.observe_material_ribbon",
            lambda *args, **kwargs: SimpleNamespace(supported_fraction=0.4),
        )

        _apply_warmup_onset_review(
            [event],
            evidence,
            frames,
            AtlasConfig(
                warmup_samples=8,
                preexisting_min_observations=2,
                spatial_link_px=1,
            ),
        )

        assert event.warmup_prefix_length_px == 0.0
        assert event.warmup_ribbon_fraction == 0.4
        assert event.warmup_attachment_ambiguous
        assert not event.auto_accepted
        assert event.quality_status == "warmup-ambiguous-growth"

    def test_broad_occlusion_mask_ignores_a_narrow_tube(self):
        frame = np.full((384, 480), 180, dtype=np.uint8)
        cv.line(frame, (100, 20), (100, 360), 150, 3)
        cv.circle(frame, (330, 190), 50, 70, -1)

        mask = broad_occlusion_mask(frame, AtlasConfig())

        assert mask[190, 330]
        assert not mask[190, 100]

    def test_path_inside_broad_occlusion_is_retained_but_demoted(self):
        frame = np.full((384, 480), 180, dtype=np.uint8)
        cv.circle(frame, (330, 190), 50, 70, -1)
        event = SimpleNamespace(
            birth_end=0,
            path_xy=np.asarray([[x, 190.0] for x in range(310, 351)]),
            broad_occlusion_fraction=None,
            auto_accepted=True,
            trajectory_accepted=True,
            quality_status="trajectory-growth",
            quality_flags=(),
        )

        _apply_broad_occlusion_review([event], frame[None, ...], AtlasConfig())

        assert event.broad_occlusion_fraction > 0.5
        assert not event.auto_accepted
        assert not event.trajectory_accepted
        assert event.quality_status == "review-required"
        assert "broad-object-occlusion" in event.quality_flags

    def test_grain_anchor_interpolates_position_at_event_time(self):
        grain = GrainAnchor(
            grain_id=3,
            center_xy=np.asarray([20.0, 10.0]),
            radius_px=6.0,
            observations=2,
            circle_score=0.9,
            sample_indices=np.asarray([0, 10]),
            centers_xy=np.asarray([[10.0, 8.0], [30.0, 12.0]]),
        )

        assert np.allclose(grain.center_at(5), [20.0, 10.0])
        assert np.allclose(grain.center_at(-5), [10.0, 8.0])
        assert np.allclose(grain.center_at(20), [30.0, 12.0])

    def test_pollen_census_includes_late_movie_views(self):
        indices = _grain_anchor_sample_indices(
            total_samples=100,
            warmup=12,
            survey_view_count=5,
        )
        assert {0, 6, 11, 99}.issubset(indices)
        assert any(index > 50 for index in indices)
        assert [index for index in indices if index < 12] == [0, 6, 11]

    def test_late_views_update_but_cannot_confirm_a_one_view_seed(
        self, monkeypatch
    ):
        def candidate(x, y, size=12):
            return SimpleNamespace(
                gv3=SimpleNamespace(x=x, y=y),
                w=size,
                h=size,
                circle_score=0.9,
            )

        detections = [
            [candidate(10, 10), candidate(30, 30)],
            [candidate(10, 10)],
            [],
            [candidate(10, 10, size=30), candidate(30, 30)],
            [candidate(30, 30)],
            [],
            [],
        ]
        import tubetracker.curve_prototype as curve_prototype

        monkeypatch.setattr(
            curve_prototype,
            "detect_grain_candidates",
            lambda frames, config: detections,
        )

        anchors = detect_grain_anchors(
            np.zeros((100, 40, 40), dtype=np.uint8),
            warmup=12,
            config=AtlasConfig(grain_anchor_view_count=5),
        )

        assert len(anchors) == 1
        assert np.allclose(anchors[0].center_xy, [10, 10])
        assert anchors[0].warmup_observations == 2
        assert anchors[0].radius_px == 6.0

    def test_pixel_domain_configuration_scales_with_analysis_width(self):
        config = AtlasConfig.for_width(960)
        assert config.min_path_length_px == 24.0
        assert config.max_tip_step_px == 7.5
        assert config.max_event_width_px == 24.0
        assert config.min_event_pixels == 96
        assert config.grain_anchor_max_match_distance_px == 40.0
        assert config.junction_direction_window_px == 10.0
        assert config.foreign_grain_clearance_margin_px == 2.0
        assert config.rim_contrast_halfwidth_px == 2.0
        assert config.rim_contrast_flank_offset_px == 6.0

    def test_time_domain_configuration_scales_with_analysis_interval(self):
        config = AtlasConfig.for_width(480).for_sample_interval(1.5)

        assert config.warmup_samples == 24
        assert config.persistence_window == 9
        assert config.persistence_required == 5
        assert config.preexisting_min_observations == 3
        assert config.temporal_link_samples == 128
        assert config.trajectory_resolution_samples == 40
        assert config.grain_anchor_warmup_view_count == 6
        assert config.max_tip_step_median_px == 1.5
        assert config.max_tip_step_px == 3.75
        assert config.grain_anchor_motion_px_per_sample == 0.04
        assert config.full_duration_score_samples == 20
        assert config.rim_history_samples == 24
        assert config.rim_persistence_window == 9
        assert config.rim_persistence_required == 5
        assert config.rim_direction_timing_tolerance_samples == 24
        assert config.front_temporal_median_samples == 5

    def test_three_second_configuration_remains_the_reference(self):
        assert AtlasConfig.for_width(480).for_sample_interval(3.0) == (
            AtlasConfig.for_width(480)
        )

    def test_source_spacing_is_converted_to_seconds_once(self):
        source_frames = np.arange(26000, 26420, 42)
        assert sample_interval_seconds(source_frames, fps=14.0) == 3.0

    def test_experiment_frame_interval_controls_sampling_and_elapsed_time(self):
        source_frames = source_frame_indices(
            frame_count=30,
            fps=14.0,
            start=2,
            end=20,
            sample_seconds=1.0,
            source_frame_interval_seconds=0.25,
        )
        assert np.array_equal(source_frames, np.asarray([2, 6, 10, 14, 18]))
        assert sample_interval_seconds(
            source_frames,
            fps=14.0,
            source_frame_interval_seconds=0.25,
        ) == 1.0

    def test_separated_flicker_does_not_become_a_birth(self):
        evidence = np.zeros((24, 8, 8), dtype=np.uint8)
        evidence[[12, 17, 22], 4, 4] = 40
        birth, _, _ = local_persistent_births(evidence, 8, 5, 3)
        assert birth[4, 4] == len(evidence)

    def test_local_persistence_records_confirmation_time(self):
        evidence = np.zeros((24, 8, 8), dtype=np.uint8)
        evidence[12:, 4, 4] = 40
        birth, _, _ = local_persistent_births(evidence, 8, 5, 3)
        assert birth[4, 4] == 14

    def test_brief_warmup_visibility_marks_old_material_and_nearby_edges(self):
        evidence = np.zeros((20, 9, 9), dtype=np.uint8)
        evidence[[0, 3], 4, 4] = 40
        evidence[8:, 4, 4:6] = 40

        birth, preexisting, calibration = local_persistent_births(
            evidence,
            warmup=8,
            window=5,
            required=3,
            preexisting_min_observations=2,
            preexisting_dilation_px=1,
        )

        assert preexisting[4, 4]
        assert preexisting[4, 5]
        assert birth[4, 4] == 0
        assert birth[4, 5] == 0
        assert calibration["preexisting_min_observations"] == 2

    def test_preexisting_dilation_does_not_consume_a_new_grain_rim_exit(self):
        evidence = np.zeros((20, 11, 11), dtype=np.uint8)
        evidence[[0, 3], 5, 5] = 40
        evidence[8:, 5, 6] = 40
        grain = GrainAnchor(
            grain_id=1,
            center_xy=np.asarray([5.0, 5.0]),
            radius_px=1.0,
            observations=3,
            circle_score=0.9,
        )

        birth, preexisting, _ = local_persistent_births(
            evidence,
            warmup=8,
            window=5,
            required=3,
            preexisting_min_observations=2,
            preexisting_dilation_px=1,
            grain_anchors=[grain],
            grain_protection_px=2.0,
        )

        assert preexisting[5, 5]
        assert not preexisting[5, 6]
        assert birth[5, 6] == 10


class BirthPathTests:
    def test_tip_timing_assessment_separates_confirmation_from_front_evidence(
        self,
    ):
        event = SimpleNamespace(
            primary_for_grain=True,
            trajectory_accepted=True,
            germination_sample=10,
            path_birth=np.asarray([10, 20, 30]),
            path_front_birth=np.asarray([10, 15, 25]),
            path_front_direct_support=np.asarray([True, True, False]),
            path_front_eventual_support=np.asarray([True, True, True]),
            front_refinement_applied=True,
            arclength_px=np.asarray([0.0, 1.0, 2.0]),
            path_xy=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        )

        before = _tip_timing_assessment(event, 9)
        direct = _tip_timing_assessment(event, 16)
        confirmed = _tip_timing_assessment(event, 20)
        event.path_front_direct_support[1] = False
        later = _tip_timing_assessment(event, 16)
        event.path_front_eventual_support[1] = False
        inferred = _tip_timing_assessment(event, 16)

        assert before.state == "not-measured"
        assert not before.measurement_accepted
        assert direct.state == "front-direct-supported"
        assert direct.measurement_accepted
        assert direct.threshold_confirmation_sample == 20
        assert direct.remaining_confirmation_lag_samples == 4
        assert confirmed.state == "threshold-confirmed"
        assert confirmed.remaining_confirmation_lag_samples == 0
        assert later.state == "front-later-supported"
        assert inferred.state == "front-inferred"

    def test_refined_timing_is_used_only_after_promotion(self):
        event = SimpleNamespace(
            germination_sample=10,
            path_birth=np.asarray([10, 20, 30]),
            path_front_birth=np.asarray([10, 15, 25]),
            front_refinement_applied=False,
            arclength_px=np.asarray([0.0, 1.0, 2.0]),
            path_xy=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        )

        assert event_tip_at(event, 16)[2] == 0
        event.front_refinement_applied = True
        assert event_tip_at(event, 16)[2] == 1
        assert event_tip_at(event, 16, use_refined_front=False)[2] == 0

    def test_front_refinement_cannot_change_primary_geometry_or_event_count(
        self, monkeypatch
    ):
        path = np.column_stack([np.arange(6), np.zeros(6)]).astype(float)

        def candidate(primary):
            return SimpleNamespace(
                primary_for_grain=primary,
                auto_accepted=True,
                quality_status="germination-only",
                quality_flags=("abrupt-tip-jump",),
                trajectory_accepted=False,
                path_xy=path.copy(),
                arclength_px=np.arange(6, dtype=float),
                path_birth=np.asarray([10, 10, 14, 14, 14, 14]),
                germination_sample=10,
                path_front_birth=None,
                path_front_direct_support=None,
                path_front_eventual_support=None,
                front_refinement_applied=False,
                front_refinement_reason="not-attempted",
                front_eventual_support_fraction=0.0,
                front_direct_support_fraction=0.0,
                front_inferred_fraction=0.0,
                front_median_confirmation_lag_samples=0.0,
                front_max_confirmation_lag_samples=0,
                growth_step_count=1,
                tip_step_median_px=4.0,
                tip_step_max_px=4.0,
                length_step_max_px=4.0,
            )

        primary = candidate(True)
        secondary = candidate(False)
        fitted = np.asarray([10, 10, 11, 12, 13, 14])
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            lambda *args, **kwargs: SimpleNamespace(
                birth_samples=fitted,
                direct_support_mask=np.ones(6, dtype=bool),
                eventual_support_mask=np.ones(6, dtype=bool),
                feasible=True,
                reason="ok",
                eventual_support_fraction=1.0,
                direct_support_fraction=1.0,
                inferred_fraction=4 / 6,
                median_confirmation_lag_samples=1.0,
                max_confirmation_lag_samples=3,
            ),
        )

        events = [primary, secondary]
        _apply_path_locked_front_refinement(
            events,
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=3,
            ),
        )

        assert len(events) == 2
        assert np.array_equal(primary.path_xy, path)
        assert primary.front_refinement_applied
        assert primary.path_front_direct_support.all()
        assert primary.path_front_eventual_support.all()
        assert primary.trajectory_accepted
        assert primary.auto_accepted
        assert "abrupt-tip-jump" not in primary.quality_flags
        assert not secondary.front_refinement_applied
        assert secondary.front_refinement_reason == "not-primary-event"

        withheld = candidate(True)
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            lambda *args, **kwargs: SimpleNamespace(
                birth_samples=fitted,
                direct_support_mask=np.asarray(
                    [True, True, True, True, False, False]
                ),
                eventual_support_mask=np.ones(6, dtype=bool),
                feasible=True,
                reason="ok",
                eventual_support_fraction=1.0,
                direct_support_fraction=0.79,
                inferred_fraction=4 / 6,
                median_confirmation_lag_samples=1.0,
                max_confirmation_lag_samples=3,
            ),
        )
        _apply_path_locked_front_refinement(
            [withheld],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=3,
            ),
        )

        assert np.array_equal(withheld.path_front_birth, fitted)
        assert np.array_equal(
            withheld.path_front_direct_support,
            [True, True, True, True, False, False],
        )
        assert not withheld.front_refinement_applied
        assert not withheld.trajectory_accepted
        assert withheld.front_refinement_reason == "insufficient-direct-tip-support"

        unsupported = candidate(True)
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            lambda *args, **kwargs: SimpleNamespace(
                birth_samples=fitted,
                direct_support_mask=np.asarray(
                    [True, True, False, True, True, True]
                ),
                eventual_support_mask=np.asarray(
                    [True, True, False, True, True, True]
                ),
                feasible=True,
                reason="ok",
                eventual_support_fraction=5 / 6,
                direct_support_fraction=5 / 6,
                inferred_fraction=1 / 6,
                median_confirmation_lag_samples=1.0,
                max_confirmation_lag_samples=3,
            ),
        )
        _apply_path_locked_front_refinement(
            [unsupported],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=3,
            ),
        )

        assert np.array_equal(unsupported.path_front_birth, fitted)
        assert not unsupported.front_refinement_applied
        assert not unsupported.trajectory_accepted
        assert unsupported.front_refinement_reason == (
            "refined-front-has-unsupported-tip-state"
        )

        contact = candidate(True)
        contact.quality_flags = (
            "abrupt-tip-jump",
            "foreign-grain-contact",
        )
        _apply_path_locked_front_refinement(
            [contact],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=3,
            ),
        )

        assert contact.path_front_birth is None
        assert not contact.front_refinement_applied
        assert not contact.trajectory_accepted
        assert contact.front_refinement_reason == "non-dynamic-trajectory-blocker"

    def test_full_path_temporal_validation_withholds_only_weak_trajectories(
        self,
        monkeypatch,
    ):
        def candidate(event_id, trajectory=True):
            return SimpleNamespace(
                event_id=event_id,
                primary_for_grain=True,
                auto_accepted=True,
                trajectory_accepted=trajectory,
                quality_status="trajectory-growth",
                quality_flags=(),
                path_xy=np.column_stack([np.arange(6), np.zeros(6)]),
                arclength_px=np.arange(6, dtype=float),
                path_birth=np.arange(10, 16),
                germination_sample=10,
                path_front_birth=None,
                path_front_direct_support=None,
                path_front_eventual_support=None,
                front_refinement_applied=False,
                front_eventual_support_fraction=0.0,
                front_direct_support_fraction=0.0,
                front_inferred_fraction=0.0,
                front_median_confirmation_lag_samples=0.0,
                front_max_confirmation_lag_samples=0,
                full_path_temporal_evaluated=False,
                full_path_temporal_verified=False,
                full_path_temporal_reason="not-evaluated",
            )

        verified = candidate(1)
        weak = candidate(2)
        infeasible = candidate(3)
        ignored = candidate(4, trajectory=False)
        applied = candidate(5)
        unsafe = candidate(6)
        applied.front_refinement_applied = True
        applied.path_front_birth = np.asarray([10, 10, 11, 12, 13, 14])
        applied.path_front_direct_support = np.ones(6, dtype=bool)
        applied.path_front_eventual_support = np.ones(6, dtype=bool)

        validation_calls = []

        def fitted(event, *_args, **kwargs):
            validation_calls.append(kwargs)
            feasible = event.event_id != 3
            direct = 0.75 if event.event_id == 2 else 1.0
            birth_samples = np.arange(10, 16)
            direct_mask = np.full(6, direct == 1.0)
            eventual_mask = np.ones(6, dtype=bool)
            if event.event_id == 6:
                birth_samples = np.arange(8, 14)
                direct_mask[2] = False
                eventual_mask[2] = False
                direct = 5 / 6
            return SimpleNamespace(
                birth_samples=birth_samples,
                direct_support_mask=direct_mask,
                eventual_support_mask=eventual_mask,
                feasible=feasible,
                reason="ok" if feasible else "front-constraints-infeasible",
                eventual_support_fraction=1.0,
                direct_support_fraction=direct,
                inferred_fraction=0.5,
                median_confirmation_lag_samples=2.0,
                max_confirmation_lag_samples=4,
            )

        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            fitted,
        )
        _apply_full_path_temporal_validation(
            [verified, weak, infeasible, ignored, applied, unsafe],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(warmup_samples=8),
        )

        assert verified.trajectory_accepted
        assert verified.full_path_temporal_verified
        assert verified.full_path_temporal_reason == "verified"
        assert np.array_equal(verified.path_front_birth, np.arange(10, 16))
        assert verified.front_refinement_applied
        assert verified.front_refinement_reason == "verified-full-path-front"
        assert not weak.trajectory_accepted
        assert weak.quality_status == "germination-only"
        assert "insufficient-full-path-direct-support" in weak.quality_flags
        assert weak.full_path_temporal_reason == (
            "insufficient-full-path-direct-support"
        )
        assert not infeasible.trajectory_accepted
        assert "unresolved-full-path-temporal-front" in infeasible.quality_flags
        assert infeasible.full_path_temporal_reason == "front-constraints-infeasible"
        assert not ignored.full_path_temporal_evaluated
        assert applied.trajectory_accepted
        assert applied.full_path_temporal_verified
        assert applied.full_path_temporal_reason == (
            "verified-by-applied-refinement"
        )
        assert np.array_equal(
            applied.path_front_birth,
            [10, 10, 11, 12, 13, 14],
        )
        assert unsafe.trajectory_accepted
        assert unsafe.full_path_temporal_verified
        assert not unsafe.front_refinement_applied
        assert unsafe.front_refinement_reason == (
            "verified-front-has-unsupported-tip-state"
        )
        assert len(validation_calls) == 4
        assert all(
            call["anchor_measurement_start"] is False
            for call in validation_calls
        )

    def test_rim_bridge_recovers_existence_but_not_exact_onset(
        self,
        monkeypatch,
    ):
        arclength = np.asarray([0.0, 1.0, 2.0, 3.4, 4.8, 6.2, 7.2])

        def candidate(flags=("unresolved-proximal-emergence",)):
            return SimpleNamespace(
                event_id=3,
                grain_id=7,
                primary_for_grain=True,
                auto_accepted=False,
                trajectory_accepted=True,
                full_path_temporal_verified=True,
                quality_status="proximal-ambiguous-trajectory",
                quality_flags=flags,
                path_xy=np.column_stack([arclength, np.zeros(len(arclength))]),
                arclength_px=arclength,
                path_birth=np.asarray([335] * len(arclength)),
                birth_start=335,
                germination_sample=335,
                rim_emergence_confirmed=True,
                rim_emergence_sample=281,
                rim_contrast_confirmed=True,
                rim_contrast_emergence_sample=313,
                proximal_front_verified=False,
                proximal_emergence_ambiguous=True,
                proximal_front_reason="front-constraints-infeasible",
                proximal_rim_bridge_evaluated=False,
                proximal_rim_bridge_accepted=False,
                proximal_rim_bridge_reason="not-evaluated",
                proximal_rim_bridge_direct_support_fraction=0.0,
                proximal_rim_bridge_eventual_support_fraction=0.0,
                proximal_rim_bridge_start_sample=None,
                proximal_rim_bridge_end_sample=None,
            )

        recovered = candidate()
        blocked = candidate(
            ("unresolved-proximal-emergence", "foreign-grain-contact")
        )
        front_birth = np.asarray([271, 271, 271, 271, 272, 272, 273])
        distal_support = np.asarray(
            [False, False, True, True, True, True, True]
        )
        calls = []

        def fitted(*_args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                birth_samples=front_birth,
                direct_support_mask=distal_support,
                eventual_support_mask=distal_support,
                feasible=True,
                reason="ok",
            )

        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            fitted,
        )
        grain = GrainAnchor(
            grain_id=7,
            center_xy=np.asarray([0.0, 0.0]),
            radius_px=5.0,
            observations=3,
            circle_score=1.0,
        )
        config = AtlasConfig(
            warmup_samples=8,
            temporal_link_samples=64,
            rim_history_samples=12,
            grain_exit_margin_px=2.0,
            min_germination_length_px=3.0,
            proximal_front_length_px=8.0,
        )
        _apply_rim_bridged_proximal_recovery(
            [recovered, blocked],
            np.zeros((400, 12, 12), dtype=np.uint8),
            config,
            [grain],
        )

        assert recovered.proximal_rim_bridge_evaluated
        assert recovered.proximal_rim_bridge_accepted
        assert recovered.proximal_rim_bridge_reason == "verified"
        assert recovered.proximal_rim_bridge_direct_support_fraction == 1.0
        assert recovered.proximal_rim_bridge_eventual_support_fraction == 1.0
        assert recovered.proximal_rim_bridge_start_sample == 271
        assert recovered.proximal_rim_bridge_end_sample == 273
        assert recovered.proximal_front_verified
        assert not recovered.proximal_emergence_ambiguous
        assert recovered.auto_accepted
        assert recovered.quality_status == "trajectory-growth"
        assert recovered.quality_flags == ()
        onset = _germination_onset_assessment(recovered, config)
        assert not onset.accepted
        assert _onset_timing_scope(recovered, onset) == "optical-review-window"
        assert not blocked.proximal_rim_bridge_evaluated
        assert not blocked.auto_accepted
        assert len(calls) == 1
        assert calls[0]["anchor_measurement_start"] is False
        assert calls[0]["maximum_arclength_px"] == 8.0

    def test_rim_bridge_rejects_weak_support_or_mistimed_cues(
        self,
        monkeypatch,
    ):
        def candidate(event_id, contrast_sample=31):
            return SimpleNamespace(
                event_id=event_id,
                grain_id=event_id,
                primary_for_grain=True,
                auto_accepted=False,
                trajectory_accepted=True,
                full_path_temporal_verified=True,
                quality_status="proximal-ambiguous-trajectory",
                quality_flags=("unresolved-proximal-emergence",),
                path_xy=np.column_stack([np.arange(7), np.zeros(7)]),
                arclength_px=np.arange(7, dtype=float),
                path_birth=np.full(7, 40),
                birth_start=40,
                germination_sample=40,
                rim_emergence_confirmed=True,
                rim_emergence_sample=30,
                rim_contrast_confirmed=True,
                rim_contrast_emergence_sample=contrast_sample,
                proximal_front_verified=False,
                proximal_emergence_ambiguous=True,
                proximal_front_reason="front-constraints-infeasible",
                proximal_rim_bridge_evaluated=False,
                proximal_rim_bridge_accepted=False,
                proximal_rim_bridge_reason="not-evaluated",
                proximal_rim_bridge_direct_support_fraction=0.0,
                proximal_rim_bridge_eventual_support_fraction=0.0,
                proximal_rim_bridge_start_sample=None,
                proximal_rim_bridge_end_sample=None,
            )

        weak = candidate(1)
        mistimed = candidate(2, contrast_sample=100)
        support = np.asarray([False, False, True, True, False, False, True])
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            lambda *_args, **_kwargs: SimpleNamespace(
                birth_samples=np.asarray([20, 20, 20, 21, 22, 23, 24]),
                direct_support_mask=support,
                eventual_support_mask=support,
                feasible=True,
                reason="ok",
            ),
        )
        grains = [
            GrainAnchor(
                grain_id=event_id,
                center_xy=np.zeros(2),
                radius_px=5.0,
                observations=3,
                circle_score=1.0,
            )
            for event_id in (1, 2)
        ]
        _apply_rim_bridged_proximal_recovery(
            [weak, mistimed],
            np.zeros((120, 12, 12), dtype=np.uint8),
            AtlasConfig(warmup_samples=8),
            grains,
        )

        assert not weak.proximal_rim_bridge_accepted
        assert weak.proximal_rim_bridge_reason == (
            "insufficient-distal-proximal-support"
        )
        assert not mistimed.proximal_rim_bridge_accepted
        assert mistimed.proximal_rim_bridge_reason == (
            "independent-rim-cues-disagree"
        )

    def test_precontact_recovery_accepts_only_rows_before_contact(
        self,
        monkeypatch,
    ):
        path = np.column_stack([np.arange(6), np.zeros(6)]).astype(float)
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=("abrupt-tip-jump", "foreign-grain-contact"),
            path_xy=path,
            arclength_px=np.arange(6, dtype=float),
            path_birth=np.asarray([10, 10, 11, 12, 13, 14]),
            germination_sample=10,
            foreign_grain_contact_path_index=5,
            foreign_grain_contact_grain_id=8,
            front_refinement_applied=False,
            path_front_birth=None,
            path_front_direct_support=None,
            path_front_eventual_support=None,
            precontact_trajectory_accepted=False,
            precontact_censor_sample=None,
            precontact_front_birth=None,
            precontact_front_direct_support=None,
            precontact_front_eventual_support=None,
            precontact_front_reason="not-evaluated",
            precontact_direct_support_fraction=0.0,
            precontact_eventual_support_fraction=0.0,
            precontact_growth_step_count=0,
            precontact_tip_step_median_px=0.0,
            precontact_tip_step_max_px=0.0,
            precontact_length_step_max_px=0.0,
        )
        fitted = np.asarray([10, 10, 11, 12, 13])
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            lambda *args, **kwargs: SimpleNamespace(
                birth_samples=fitted,
                direct_support_mask=np.ones(5, dtype=bool),
                eventual_support_mask=np.ones(5, dtype=bool),
                feasible=True,
                reason="ok",
                eventual_support_fraction=1.0,
                direct_support_fraction=1.0,
            ),
        )

        _apply_precontact_trajectory_recovery(
            [event],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=3,
                trajectory_resolution_samples=3,
            ),
        )

        assert event.precontact_trajectory_accepted
        assert not event.trajectory_accepted
        assert event.quality_status == "contact-censored-growth"
        assert event.precontact_censor_sample == 14
        assert np.array_equal(event.precontact_front_birth, fitted)
        assert event_tip_at(event, 12)[2] == 3
        assert _tip_timing_assessment(event, 13).measurement_accepted
        assert not _tip_timing_assessment(event, 14).measurement_accepted

        blocked = SimpleNamespace(**vars(event))
        blocked.precontact_trajectory_accepted = False
        blocked.quality_flags = (
            "foreign-grain-contact",
            "ambiguous-component-ownership",
        )
        _apply_precontact_trajectory_recovery(
            [blocked],
            np.zeros((20, 10, 10), dtype=np.uint8),
            AtlasConfig(warmup_samples=8),
        )
        assert not blocked.precontact_trajectory_accepted
        assert blocked.precontact_front_reason == "non-contact-trajectory-blocker"

    def test_contact_bridge_recovers_only_contiguous_supported_far_side(
        self,
        monkeypatch,
    ):
        path = np.column_stack([np.arange(12), np.zeros(12)]).astype(float)
        event = SimpleNamespace(
            event_id=1,
            component_id=3,
            grain_id=4,
            primary_for_grain=True,
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="contact-censored-growth",
            quality_flags=("abrupt-tip-jump", "foreign-grain-contact"),
            path_xy=path,
            arclength_px=np.arange(12, dtype=float),
            path_birth=np.arange(10, 22),
            germination_sample=10,
            foreign_grain_contact_path_index=4,
            foreign_grain_contact_grain_id=8,
            foreign_grain_contact_exit_path_index=6,
            contact_bridge_next_foreign_contact_path_index=None,
            contact_bridge_hidden_length_px=2.0,
            contact_bridge_ingress_chord_angle_degrees=0.0,
            contact_bridge_chord_egress_angle_degrees=0.0,
            contact_bridge_total_turn_degrees=0.0,
            contact_bridge_identity_supported=False,
            contact_bridge_viable_competitor_count=0,
            contact_bridge_trajectory_accepted=False,
            contact_bridge_last_path_index=None,
            contact_bridge_censor_sample=None,
            contact_bridge_front_birth=None,
            contact_bridge_front_direct_support=None,
            contact_bridge_front_eventual_support=None,
            contact_bridge_reason="not-evaluated",
            contact_bridge_direct_support_fraction=0.0,
            contact_bridge_eventual_support_fraction=0.0,
            contact_bridge_post_direct_support_fraction=0.0,
            contact_bridge_post_eventual_support_fraction=0.0,
            contact_bridge_growth_step_count=0,
            contact_bridge_post_growth_step_count=0,
            contact_bridge_tip_step_median_px=0.0,
            contact_bridge_tip_step_max_px=0.0,
            contact_bridge_length_step_max_px=0.0,
            precontact_trajectory_accepted=True,
        )
        invalid_competitor = SimpleNamespace(
            event_id=2,
            component_id=3,
            grain_id=8,
            primary_for_grain=False,
            auto_accepted=False,
            trajectory_accepted=False,
            quality_flags=("no-rim-emergence",),
            path_xy=path[6:].copy(),
        )

        def fitted(candidate, *_args, **kwargs):
            endpoint = int(round(kwargs["maximum_arclength_px"]))
            count = endpoint + 1
            direct = np.ones(count, dtype=bool)
            eventual = np.ones(count, dtype=bool)
            if endpoint >= 10:
                direct[-1] = False
                eventual[-1] = False
            return SimpleNamespace(
                birth_samples=np.arange(10, 10 + count),
                direct_support_mask=direct,
                eventual_support_mask=eventual,
                feasible=True,
                reason="ok",
                direct_support_fraction=float(np.mean(direct)),
                eventual_support_fraction=float(np.mean(eventual)),
            )

        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            fitted,
        )
        _apply_contact_bridge_recovery(
            [event, invalid_competitor],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(
                warmup_samples=8,
                min_germination_length_px=1.0,
                min_growth_step_count=2,
                trajectory_resolution_samples=2,
            ),
        )

        assert event.contact_bridge_identity_supported
        assert event.contact_bridge_viable_competitor_count == 0
        assert event.contact_bridge_trajectory_accepted
        assert event.contact_bridge_last_path_index == 9
        assert event.contact_bridge_censor_sample == 20
        assert event.contact_bridge_post_growth_step_count == 3
        assert event.contact_bridge_reason == (
            "verified-through-contact-until-distal-support-loss"
        )
        assert event.quality_status == "contact-bridged-growth"
        assert _trajectory_measurement_scope(event) == "contact-bridged-prefix"
        assert event_tip_at(event, 19)[2] == 9
        assert _tip_timing_assessment(event, 19).measurement_accepted
        assert not _tip_timing_assessment(event, 20).measurement_accepted

    def test_contact_bridge_rejects_turns_and_viable_competing_roots(
        self,
        monkeypatch,
    ):
        path = np.column_stack([np.arange(10), np.zeros(10)]).astype(float)

        def candidate(total_turn=0.0):
            return SimpleNamespace(
                event_id=1,
                component_id=3,
                grain_id=4,
                primary_for_grain=True,
                auto_accepted=True,
                trajectory_accepted=False,
                quality_status="contact-censored-growth",
                quality_flags=("foreign-grain-contact",),
                path_xy=path,
                arclength_px=np.arange(10, dtype=float),
                path_birth=np.arange(10, 20),
                germination_sample=10,
                foreign_grain_contact_path_index=3,
                foreign_grain_contact_grain_id=8,
                foreign_grain_contact_exit_path_index=5,
                contact_bridge_next_foreign_contact_path_index=None,
                contact_bridge_ingress_chord_angle_degrees=5.0,
                contact_bridge_chord_egress_angle_degrees=5.0,
                contact_bridge_total_turn_degrees=total_turn,
                contact_bridge_identity_supported=False,
                contact_bridge_viable_competitor_count=0,
                contact_bridge_trajectory_accepted=False,
                contact_bridge_reason="not-evaluated",
                precontact_trajectory_accepted=True,
            )

        sharp = candidate(total_turn=80.0)
        _apply_contact_bridge_recovery(
            [sharp],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(warmup_samples=8),
        )
        assert not sharp.contact_bridge_identity_supported
        assert sharp.contact_bridge_reason == "excessive-contact-turn"

        ambiguous = candidate()
        competitor = SimpleNamespace(
            event_id=2,
            component_id=3,
            grain_id=8,
            primary_for_grain=False,
            auto_accepted=False,
            trajectory_accepted=False,
            quality_flags=(),
            path_xy=path[5:].copy(),
        )
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track._fit_event_temporal_front",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ambiguous bridges must not fit a timeline")
            ),
        )
        _apply_contact_bridge_recovery(
            [ambiguous, competitor],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(warmup_samples=8),
        )
        assert not ambiguous.contact_bridge_identity_supported
        assert ambiguous.contact_bridge_viable_competitor_count == 1
        assert ambiguous.contact_bridge_reason == "viable-competing-pollen-root"

    def test_contact_bridge_geometry_distinguishes_straight_and_turning_paths(
        self,
    ):
        straight = np.column_stack([np.arange(12), np.zeros(12)]).astype(float)
        arclength = np.arange(12, dtype=float)
        clearance = np.full(12, 2.0)
        clearance[4:7] = 0.0
        geometry = _contact_bridge_geometry(
            straight,
            arclength,
            clearance,
            4,
            AtlasConfig(junction_direction_window_px=2.0),
        )

        turning = straight.copy()
        turning[7:, 0] = 7.0
        turning[7:, 1] = np.arange(5)
        turned = _contact_bridge_geometry(
            turning,
            np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(turning, axis=0), axis=1))]),
            clearance,
            4,
            AtlasConfig(junction_direction_window_px=2.0),
        )

        assert geometry[0] == 7
        assert geometry[1] == 3.0
        assert geometry[2:] == (0.0, 0.0, 0.0)
        assert turned[-1] is not None
        assert turned[-1] >= 89.0

    def test_proximal_front_withholds_only_unverified_primary_germination(
        self, monkeypatch
    ):
        def candidate(primary=True):
            return SimpleNamespace(
                primary_for_grain=primary,
                auto_accepted=True,
                trajectory_accepted=True,
                quality_status="trajectory-growth",
                quality_flags=(),
                grain_id=7,
                path_xy=np.column_stack([np.arange(10), np.zeros(10)]),
                arclength_px=np.arange(10, dtype=float),
                path_birth=np.arange(10, 20),
                germination_sample=13,
                proximal_front_verified=False,
                proximal_emergence_ambiguous=False,
                proximal_front_reason="not-evaluated",
                proximal_eventual_support_fraction=0.0,
                proximal_direct_support_fraction=0.0,
                proximal_inferred_fraction=0.0,
                proximal_median_confirmation_lag_samples=0.0,
                proximal_max_confirmation_lag_samples=0,
            )

        failed = candidate()
        secondary = candidate(primary=False)
        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            lambda *args, **kwargs: SimpleNamespace(
                feasible=False,
                reason="front-constraints-infeasible",
                eventual_support_fraction=0.0,
                direct_support_fraction=0.0,
                inferred_fraction=0.0,
                median_confirmation_lag_samples=0.0,
                max_confirmation_lag_samples=0,
            ),
        )

        _apply_proximal_emergence_review(
            [failed, secondary],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(warmup_samples=8, min_germination_length_px=1.0),
        )

        assert not failed.auto_accepted
        assert failed.trajectory_accepted
        assert failed.proximal_emergence_ambiguous
        assert failed.quality_status == "proximal-ambiguous-trajectory"
        assert "unresolved-proximal-emergence" in failed.quality_flags
        assert secondary.auto_accepted
        assert secondary.proximal_front_reason == "not-primary-event"

        verified = candidate()
        captured = {}

        def verified_front(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                feasible=True,
                reason="ok",
                eventual_support_fraction=1.0,
                direct_support_fraction=0.9,
                inferred_fraction=0.5,
                median_confirmation_lag_samples=2.0,
                max_confirmation_lag_samples=5,
            )

        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            verified_front,
        )
        grain = GrainAnchor(
            grain_id=7,
            center_xy=np.asarray([10.0, 10.0]),
            radius_px=5.0,
            observations=3,
            circle_score=0.9,
            sample_indices=np.asarray([0, 15, 29]),
            centers_xy=np.asarray([[10.0, 10.0], [13.0, 11.5], [15.0, 14.0]]),
        )
        _apply_proximal_emergence_review(
            [verified],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(warmup_samples=8, min_germination_length_px=1.0),
            [grain],
        )

        assert verified.auto_accepted
        assert verified.proximal_front_verified
        assert verified.proximal_front_reason == "verified"
        assert verified.proximal_motion_binding_class == "both-supported"
        assert verified.proximal_static_direct_support_fraction == 0.9
        assert verified.proximal_static_eventual_support_fraction == 1.0
        offsets = captured["path_offsets_xy"]
        assert offsets.shape == (30, 2)
        assert np.allclose(offsets[verified.germination_sample], 0.0)
        assert np.allclose(
            offsets[0],
            grain.center_at(0) - grain.center_at(verified.germination_sample),
        )

        static_only = candidate()

        def basis_front(*args, **kwargs):
            moving = kwargs["path_offsets_xy"] is not None
            support = 0.5 if moving else 1.0
            return SimpleNamespace(
                feasible=True,
                reason="ok",
                eventual_support_fraction=support,
                direct_support_fraction=support,
                inferred_fraction=0.5,
                median_confirmation_lag_samples=2.0,
                max_confirmation_lag_samples=5,
            )

        monkeypatch.setattr(
            "prototypes.v17_birth_topology.track.path_locked_temporal_front",
            basis_front,
        )
        _apply_proximal_emergence_review(
            [static_only],
            np.zeros((30, 20, 20), dtype=np.uint8),
            AtlasConfig(warmup_samples=8, min_germination_length_px=1.0),
            [grain],
        )

        assert not static_only.auto_accepted
        assert static_only.proximal_emergence_ambiguous
        assert static_only.proximal_motion_binding_class == "static-only"
        assert static_only.proximal_motion_support_margin == -0.5
        assert static_only.proximal_front_reason == (
            "static-only-proximal-structure"
        )
        assert np.isclose(
            verified.proximal_reference_motion_max_px,
            np.max(np.linalg.norm(offsets, axis=1)),
        )

    def test_connected_proximal_emergence_requires_one_outward_chain(self):
        evidence = np.zeros((24, 40, 40), dtype=np.uint8)
        path = np.column_stack([np.arange(12, 20), np.full(8, 20)])
        evidence[10:16, 20, 12:20] = 20
        offsets = np.zeros((len(evidence), 2), dtype=float)
        config = AtlasConfig(
            warmup_samples=8,
            proximal_front_length_px=8.0,
            rim_direction_normal_halfwidth_px=0.5,
        )

        sample = _connected_proximal_emergence_sample(
            evidence,
            path,
            offsets,
            config,
            evidence_floor=6.0,
        )
        disconnected = evidence.copy()
        disconnected[:, 20, 15:17] = 0
        missing_sample = _connected_proximal_emergence_sample(
            disconnected,
            path,
            offsets,
            config,
            evidence_floor=6.0,
        )

        assert sample == 12
        assert missing_sample is None

    def test_independent_rim_change_withholds_only_a_short_claim(self):
        evidence = np.zeros((30, 60, 60), dtype=np.uint8)
        evidence[10:18, 30, 20:25] = 20
        grain = GrainAnchor(
            grain_id=4,
            center_xy=np.asarray([30.0, 30.0]),
            radius_px=4.0,
            observations=3,
            circle_score=0.9,
        )

        def candidate(*, trajectory_accepted=False):
            path = np.column_stack(
                [np.arange(36.0, 41.0), np.full(5, 30.0)]
            )
            return SimpleNamespace(
                primary_for_grain=True,
                auto_accepted=True,
                trajectory_accepted=trajectory_accepted,
                quality_status=(
                    "trajectory-growth"
                    if trajectory_accepted
                    else "emergence-only"
                ),
                quality_flags=("short-after-rim-exit",),
                grain_id=4,
                path_xy=path,
                arclength_px=np.arange(5, dtype=float),
                birth_start=10,
                germination_sample=12,
            )

        short = candidate()
        full = candidate(trajectory_accepted=True)
        config = AtlasConfig(
            warmup_samples=8,
            rim_direction_normal_halfwidth_px=0.5,
            rim_direction_timing_tolerance_samples=4,
        )
        _apply_rim_direction_specificity_review(
            [short, full], evidence, config, [grain]
        )

        assert short.rim_direction_specificity_evaluated
        assert not short.rim_direction_specific
        assert short.rim_direction_competitor_count >= 1
        assert short.rim_direction_competitor_rotation_degrees == 180
        assert short.rim_direction_competitor_sample == 12
        assert not short.auto_accepted
        assert short.quality_status == "review-required"
        assert "non-specific-rim-change" in short.quality_flags
        assert full.auto_accepted
        assert full.rim_direction_reason == (
            "competitor-recorded-path-retained"
        )

    def test_timeline_audit_samples_use_rim_emergence_when_available(self):
        event = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            rim_emergence_sample=20,
            rim_contrast_emergence_sample=21,
            birth_start=22,
            germination_sample=24,
            birth_end=80,
        )

        landmarks = _event_audit_samples(
            event,
            total_samples=60,
            config=AtlasConfig(warmup_samples=12),
        )

        assert landmarks == (
            ("start", 0),
            ("warmup", 11),
            ("onset-before", 15),
            ("rim", 20),
            ("contrast", 21),
            ("root", 22),
            ("measure", 24),
            ("final", 59),
        )

    def test_rim_direction_audit_shows_selected_and_competing_sectors(
        self, tmp_path
    ):
        frames = np.full((20, 60, 80), 170, dtype=np.uint8)
        source_frames = np.arange(20, dtype=np.int64) * 42
        event = SimpleNamespace(
            event_id=3,
            grain_id=7,
            primary_for_grain=True,
            rim_direction_specificity_evaluated=True,
            rim_direction_specific=False,
            rim_direction_competitor_rotation_degrees=180,
            rim_direction_competitor_sample=11,
            rim_direction_selected_connected_sample=15,
            grain_center_xy=np.asarray([20.0, 30.0]),
            path_xy=np.asarray(
                [[26.0, 30.0], [27.0, 30.0], [28.0, 30.0]]
            ),
            arclength_px=np.asarray([0.0, 1.0, 2.0]),
            birth_start=10,
            germination_sample=12,
            birth_end=18,
        )
        grain = GrainAnchor(
            grain_id=7,
            center_xy=np.asarray([20.0, 30.0]),
            radius_px=5.0,
            observations=3,
            circle_score=0.9,
        )
        output = tmp_path / "rim-directions.jpg"

        count = write_rim_direction_specificity_audit(
            output,
            frames,
            source_frames,
            [event],
            AtlasConfig(warmup_samples=4),
            [grain],
            tile_size=64,
        )
        image = cv.imread(str(output))

        assert count == 1
        assert image.shape == (64, 384, 3)
        assert np.any(image)

    def test_timeline_audit_writes_reviewable_primary_events(self, tmp_path):
        frames = np.full((12, 40, 50), 170, dtype=np.uint8)
        source_frames = np.arange(12, dtype=np.int64) * 42
        event = SimpleNamespace(
            event_id=3,
            grain_id=7,
            primary_for_grain=True,
            auto_accepted=True,
            grain_center_xy=np.asarray([12.0, 20.0]),
            path_xy=np.asarray([[15.0, 20.0], [20.0, 20.0], [25.0, 20.0]]),
            path_birth=np.asarray([5, 6, 7]),
            arclength_px=np.asarray([0.0, 5.0, 10.0]),
            rim_emergence_sample=5,
            rim_contrast_emergence_sample=6,
            birth_start=5,
            germination_sample=5,
            birth_end=7,
        )
        ignored = SimpleNamespace(primary_for_grain=False, auto_accepted=True)
        output = tmp_path / "timeline.jpg"

        count = write_event_timeline_audit(
            output,
            frames,
            source_frames,
            [event, ignored],
            AtlasConfig(warmup_samples=3),
            tile_size=64,
        )
        image = cv.imread(str(output))

        assert count == 1
        assert image.shape == (64, 1024, 3)
        assert np.any(image)

    def test_front_audit_includes_applied_and_withheld_candidates(self, tmp_path):
        frames = np.full((20, 40, 50), 170, dtype=np.uint8)
        source_frames = np.arange(20, dtype=np.int64) * 42

        def candidate(event_id, applied):
            return SimpleNamespace(
                event_id=event_id,
                grain_id=event_id + 10,
                primary_for_grain=True,
                front_refinement_applied=applied,
                grain_center_xy=np.asarray([12.0, 20.0]),
                path_xy=np.asarray(
                    [[15.0, 20.0], [20.0, 20.0], [25.0, 20.0]]
                ),
                path_birth=np.asarray([8, 12, 16]),
                path_front_birth=np.asarray([8, 10, 14]),
                arclength_px=np.asarray([0.0, 5.0, 10.0]),
                germination_sample=8,
                birth_end=16,
            )

        output = tmp_path / "fronts.jpg"
        failed = candidate(3, False)
        failed.path_front_birth = None
        failed.full_path_temporal_evaluated = True
        failed.full_path_temporal_verified = False
        partial = candidate(4, False)
        partial.path_front_birth = None
        partial.precontact_front_birth = np.asarray([8, 10])
        partial.precontact_trajectory_accepted = True
        partial.precontact_censor_sample = 11
        partial.foreign_grain_contact_path_index = 2
        bridged = candidate(5, False)
        bridged.path_front_birth = None
        bridged.contact_bridge_front_birth = np.asarray([8, 10, 12])
        bridged.contact_bridge_trajectory_accepted = True
        bridged.contact_bridge_censor_sample = 13
        count = write_front_refinement_audit(
            output,
            frames,
            source_frames,
            [candidate(1, True), candidate(2, False), failed, partial, bridged],
            tile_size=64,
        )
        image = cv.imread(str(output))

        assert count == 5
        assert image.shape == (320, 384, 3)
        assert np.any(image)

    def test_proximal_audit_writes_every_evaluated_primary(self, tmp_path):
        frames = np.full((20, 40, 50), 170, dtype=np.uint8)
        source_frames = np.arange(20, dtype=np.int64) * 42
        event = SimpleNamespace(
            event_id=4,
            grain_id=9,
            primary_for_grain=True,
            proximal_front_verified=False,
            proximal_emergence_ambiguous=True,
            grain_center_xy=np.asarray([12.0, 20.0]),
            path_xy=np.asarray(
                [[15.0, 20.0], [20.0, 20.0], [25.0, 20.0]]
            ),
            arclength_px=np.asarray([0.0, 5.0, 10.0]),
            germination_sample=10,
        )
        verified = SimpleNamespace(
            **{
                **vars(event),
                "event_id": 5,
                "proximal_front_verified": True,
                "proximal_emergence_ambiguous": False,
            }
        )
        output = tmp_path / "proximal.jpg"

        count = write_proximal_emergence_audit(
            output,
            frames,
            source_frames,
            [event, verified],
            AtlasConfig(proximal_front_length_px=8.0),
            tile_size=64,
        )
        image = cv.imread(str(output))

        assert count == 2
        assert image.shape == (128, 576, 3)
        assert np.any(image)

    def test_trajectory_audit_writes_every_primary_trajectory(self, tmp_path):
        frames = np.full((20, 40, 50), 170, dtype=np.uint8)
        source_frames = np.arange(20, dtype=np.int64) * 42
        event = SimpleNamespace(
            event_id=4,
            grain_id=9,
            primary_for_grain=True,
            trajectory_accepted=True,
            auto_accepted=False,
            grain_center_xy=np.asarray([12.0, 20.0]),
            path_xy=np.asarray(
                [[15.0, 20.0], [20.0, 20.0], [25.0, 20.0]]
            ),
            arclength_px=np.asarray([0.0, 5.0, 10.0]),
            path_birth=np.asarray([6, 10, 14]),
            path_front_birth=np.asarray([6, 8, 12]),
            path_front_direct_support=np.asarray([True, True, False]),
            path_front_eventual_support=np.asarray([True, True, True]),
            front_refinement_applied=True,
            germination_sample=6,
        )
        ignored = SimpleNamespace(
            primary_for_grain=False,
            trajectory_accepted=True,
        )
        partial = SimpleNamespace(
            **{
                **vars(event),
                "event_id": 5,
                "trajectory_accepted": False,
                "auto_accepted": True,
                "path_front_birth": None,
                "path_front_direct_support": None,
                "path_front_eventual_support": None,
                "front_refinement_applied": False,
                "precontact_trajectory_accepted": True,
                "precontact_censor_sample": 12,
                "foreign_grain_contact_path_index": 2,
                "precontact_front_birth": np.asarray([6, 8]),
                "precontact_front_direct_support": np.ones(2, dtype=bool),
                "precontact_front_eventual_support": np.ones(2, dtype=bool),
            }
        )
        bridged = SimpleNamespace(
            **{
                **vars(event),
                "event_id": 6,
                "trajectory_accepted": False,
                "auto_accepted": True,
                "path_front_birth": None,
                "path_front_direct_support": None,
                "path_front_eventual_support": None,
                "front_refinement_applied": False,
                "contact_bridge_trajectory_accepted": True,
                "contact_bridge_last_path_index": 2,
                "contact_bridge_censor_sample": 13,
                "contact_bridge_front_birth": np.asarray([6, 8, 12]),
                "contact_bridge_front_direct_support": np.ones(3, dtype=bool),
                "contact_bridge_front_eventual_support": np.ones(3, dtype=bool),
            }
        )
        output = tmp_path / "trajectories.jpg"

        count = write_trajectory_timeline_audit(
            output,
            frames,
            source_frames,
            [event, partial, bridged, ignored],
            tile_size=64,
        )
        image = cv.imread(str(output))

        assert count == 3
        assert image.shape == (192, 576, 3)
        assert np.any(image)

    def test_blinded_validation_mixes_decisions_without_labeling_the_image(
        self, tmp_path
    ):
        frames = np.full((20, 60, 80), 170, dtype=np.uint8)
        source_frames = np.arange(20, dtype=np.int64) * 42
        grains = [
            GrainAnchor(1, np.asarray([20.0, 30.0]), 6.0, 3, 0.9),
            GrainAnchor(2, np.asarray([60.0, 30.0]), 6.0, 3, 0.9),
            GrainAnchor(3, np.asarray([4.0, 30.0]), 3.0, 3, 0.9),
        ]
        accepted = SimpleNamespace(
            event_id=4,
            grain_id=1,
            primary_for_grain=True,
            auto_accepted=True,
            trajectory_accepted=True,
            germination_left_censored=False,
            warmup_attachment_ambiguous=False,
            rim_emergence_sample=10,
            germination_sample=11,
            birth_end=16,
            quality_status="trajectory-growth",
            score=8.0,
            quality_flags=(),
        )
        image_path = tmp_path / "validation.jpg"
        manifest_path = tmp_path / "validation.csv"

        counts = write_blinded_validation_audit(
            image_path,
            manifest_path,
            frames,
            source_frames,
            [accepted],
            grains,
            AtlasConfig(warmup_samples=4),
            tile_size=64,
            maximum_per_stratum=2,
        )
        image = cv.imread(str(image_path))
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert counts == {
            "accepted": 1,
            "no-hypothesis": 1,
            "boundary": 1,
            "total": 3,
        }
        assert image.shape == (128, 640, 3)
        assert {row["tracker_stratum"] for row in rows} == {
            "accepted",
            "no-hypothesis",
            "boundary",
        }
        no_hypothesis = next(
            row for row in rows if row["tracker_stratum"] == "no-hypothesis"
        )
        assert no_hypothesis["review_source_frames"] == "0;126;252;546;798"
        assert all(not row["human_germination_present"] for row in rows)
        assert all(not row["human_tip_trackable"] for row in rows)

    def test_rim_emergence_requires_a_persistent_directional_change(self):
        evidence = np.zeros((30, 50, 50), dtype=np.uint8)
        evidence[12:, 18:23, 26:31] = 40
        grain = GrainAnchor(
            grain_id=1,
            center_xy=np.asarray([20.0, 20.0]),
            radius_px=5.0,
            observations=3,
            circle_score=0.9,
        )
        config = AtlasConfig(
            rim_history_samples=8,
            rim_persistence_window=5,
            rim_persistence_required=3,
        )

        delta, _, persistent, confirmed, emergence_sample = _rim_emergence_metrics(
            evidence,
            grain,
            path_xy=np.asarray(
                [[27.0, 20.0], [28.0, 20.0], [29.0, 20.0], [30.0, 20.0]]
            ),
            sample=12,
            config=config,
        )
        no_delta, _, _, already_present, no_emergence_sample = _rim_emergence_metrics(
            np.where(evidence > 0, evidence, 40),
            grain,
            path_xy=np.asarray(
                [[27.0, 20.0], [28.0, 20.0], [29.0, 20.0], [30.0, 20.0]]
            ),
            sample=12,
            config=config,
        )

        assert delta >= 40.0
        assert persistent == 3
        assert confirmed
        assert emergence_sample == 14
        assert no_delta == 0.0
        assert not already_present
        assert no_emergence_sample is None

    def test_rim_emergence_handles_a_clip_ending_at_the_baseline(self):
        evidence = np.zeros((8, 30, 30), dtype=np.uint8)
        grain = GrainAnchor(
            grain_id=1,
            center_xy=np.asarray([15.0, 15.0]),
            radius_px=4.0,
            observations=3,
            circle_score=0.9,
        )

        delta, noise, persistent, confirmed, sample = _rim_emergence_metrics(
            evidence,
            grain,
            path_xy=np.asarray([[20.0, 15.0], [21.0, 15.0]]),
            sample=7,
            config=AtlasConfig(rim_history_samples=8),
        )

        assert delta == 0.0
        assert noise == 0.0
        assert persistent == 0
        assert not confirmed
        assert sample is None

    def test_birth_order_metrics_distinguish_outward_growth(self):
        arclength = np.arange(20, dtype=float)
        outward = np.repeat(np.arange(5), 4)
        reversed_birth = outward[::-1]

        correlation, forward, progress = _birth_order_metrics(outward, arclength)
        reverse_correlation, reverse_forward, reverse_progress = (
            _birth_order_metrics(reversed_birth, arclength)
        )

        assert correlation > 0.9
        assert forward == 1.0
        assert progress > 0
        assert reverse_correlation < -0.9
        assert reverse_forward < forward
        assert reverse_progress < 0

    def test_branch_selection_prefers_ordered_growth_over_long_reversed_branch(self):
        birth = np.zeros((4, 24), dtype=np.int32)
        ordered_yx = np.column_stack(
            [np.zeros(12, dtype=int), np.arange(12, dtype=int)]
        )
        reversed_yx = np.column_stack(
            [np.ones(22, dtype=int), np.arange(22, dtype=int)]
        )
        birth[0, :12] = np.linspace(10, 50, 12).astype(int)
        birth[1, :22] = np.linspace(60, 10, 22).astype(int)
        candidates = [
            (ordered_yx, np.arange(12, dtype=float)),
            (reversed_yx, np.arange(22, dtype=float)),
        ]

        selected_yx, _, _, correlation, _, progress = (
            _select_birth_ordered_path(candidates, birth, AtlasConfig())
        )

        assert np.array_equal(selected_yx, ordered_yx)
        assert correlation > 0.9
        assert progress > 0

    def test_crossing_selection_prefers_smooth_tube_continuation(self):
        skeleton = np.zeros((50, 50), dtype=bool)
        skeleton[25, 5:36] = True
        skeleton[5:26, 25] = True
        birth = np.full(skeleton.shape, 100, dtype=np.int32)
        birth[25, 5:36] = np.linspace(10, 40, 31).astype(np.int32)
        birth[5:26, 25] = np.linspace(65, 30, 21).astype(np.int32)
        candidates = _skeleton_path_candidates(skeleton, (25, 5))

        selected_yx, selected_arclength, *_ = _select_birth_ordered_path(
            candidates,
            birth,
            AtlasConfig(),
            skeleton=skeleton,
        )

        assert tuple(selected_yx[-1]) == (25, 35)
        assert _maximum_junction_turn_degrees(
            selected_yx,
            selected_arclength,
            skeleton,
            direction_window_px=5.0,
        ) == 0.0
        foreign_yx, foreign_arclength = next(
            candidate for candidate in candidates if tuple(candidate[0][-1]) == (5, 25)
        )
        assert 85.0 <= _maximum_junction_turn_degrees(
            foreign_yx,
            foreign_arclength,
            skeleton,
            direction_window_px=5.0,
        ) <= 95.0

    def test_sharp_nonbranch_bend_is_not_mistaken_for_a_crossing(self):
        skeleton = np.zeros((40, 40), dtype=bool)
        skeleton[25, 5:26] = True
        skeleton[5:26, 25] = True
        path_yx, arclength = _skeleton_path_candidates(skeleton, (25, 5))[0]

        assert tuple(path_yx[-1]) == (5, 25)
        assert _maximum_junction_turn_degrees(
            path_yx,
            arclength,
            skeleton,
            direction_window_px=5.0,
        ) == 0.0

    def test_germination_can_survive_an_unusable_tip_trajectory(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [40.0, 20.0]]),
                length_px=20.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=12.0,
                radial_extension=18.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
            )
        )

        assert germination_accepted
        assert not trajectory_accepted
        assert status == "germination-only"
        assert "abrupt-tip-jump" in flags

    def test_rim_following_arc_is_retained_for_review_not_counted(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [35.0, 25.0]]),
                length_px=20.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.4,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=2.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "weak-radial-extension" in flags

    def test_partial_field_event_is_retained_for_review_not_counted(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [0.0, 30.0]]),
                length_px=24.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.1,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=20.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "partial-field" in flags

    def test_partial_pollen_body_is_retained_for_review_not_counted(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[8.0, 20.0], [30.0, 20.0]]),
                length_px=22.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=20.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
                grain_center_xy=np.asarray([4.0, 20.0]),
                grain_radius_px=5.0,
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "partial-grain" in flags

    def test_under_sampled_path_touching_two_grains_has_ambiguous_ownership(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [40.0, 20.0]]),
                length_px=20.0,
                duration_samples=3,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.0,
                birth_forward=0.8,
                birth_progress=0.0,
                growth_step_count=2,
                tip_step_median_px=8.0,
                tip_step_max_px=16.0,
                radial_extension=18.0,
                foreign_clearance=0.5,
                image_shape=(100, 100),
                config=AtlasConfig(),
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "ambiguous-grain-ownership" in flags

    def test_resolved_foreign_grain_contact_withholds_only_tip_trajectory(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [40.0, 20.0]]),
                length_px=20.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=18.0,
                foreign_clearance=-1.0,
                image_shape=(100, 100),
                config=AtlasConfig(),
            )
        )

        assert germination_accepted
        assert not trajectory_accepted
        assert status == "germination-only"
        assert "foreign-grain-contact" in flags

    def test_event_detached_from_time_aware_grain_is_never_accepted(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [40.0, 20.0]]),
                length_px=20.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=18.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
                grain_attachment=20.0,
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "detached-from-grain" in flags

    def test_event_with_mismatched_rim_timing_is_never_accepted(self):
        status, flags, germination_accepted, trajectory_accepted = (
            _quality_classification(
                path_xy=np.asarray([[20.0, 20.0], [40.0, 20.0]]),
                length_px=20.0,
                duration_samples=40,
                root_distance_px=1.0,
                width_px=4.0,
                tortuosity=1.0,
                birth_correlation=0.9,
                birth_forward=0.9,
                birth_progress=30.0,
                growth_step_count=10,
                tip_step_median_px=1.0,
                tip_step_max_px=2.0,
                radial_extension=18.0,
                foreign_clearance=None,
                image_shape=(100, 100),
                config=AtlasConfig(),
                grain_attachment=1.0,
                rim_emergence_confirmed=True,
                rim_emergence_timing_consistent=False,
            )
        )

        assert not germination_accepted
        assert not trajectory_accepted
        assert status == "review-required"
        assert "rim-emergence-timing-mismatch" in flags

    def test_tube_length_starts_after_stable_grain_rim_exit(self):
        rim_xy = np.asarray(
            [[5, 10], [6, 7], [8, 5], [10, 5], [12, 5], [14, 7], [15, 10]],
            dtype=int,
        )
        tube_xy = np.column_stack([np.arange(16, 27), np.full(11, 10)])
        path_xy = np.vstack([rim_xy, tube_xy])
        path_yx = path_xy[:, ::-1]

        trimmed = _trim_grain_rim_prefix(
            path_yx,
            grain_center_xy=np.asarray([10.0, 10.0]),
            grain_radius_px=5.0,
            margin_px=0.5,
            outside_run=4,
        )

        assert len(trimmed) < len(path_yx)
        assert tuple(trimmed[0][::-1]) == (15, 10)
        assert tuple(trimmed[-1][::-1]) == (26, 10)

    def test_short_mask_gap_is_connected_back_to_the_pollen_rim(self):
        path_yx = np.asarray([[10, 20], [10, 21], [10, 22]])

        connected, prefix_count = _prepend_grain_rim_connector(
            path_yx,
            grain_center_xy=np.asarray([10.0, 10.0]),
            grain_radius_px=5.0,
            margin_px=2.0,
        )

        assert prefix_count == 3
        assert tuple(connected[0][::-1]) == (17, 10)
        assert tuple(connected[-1][::-1]) == (22, 10)

    def test_recovers_one_simple_path_connected_to_existing_grain(self):
        total, height, width = 50, 80, 100
        frames = np.full((total, height, width), 180, dtype=np.uint8)
        for frame in frames:
            cv.circle(frame, (22, 40), 8, 70, -1)
        for sample in range(14, total):
            end_x = min(30 + sample - 14, 74)
            cv.line(frames[sample], (29, 40), (end_x, 40), 85, 3)

        evidence = ridge_evidence(frames)
        birth, preexisting, _ = local_persistent_births(evidence, 10, 5, 3)
        events = extract_birth_events(
            birth,
            preexisting,
            AtlasConfig(
                width=width,
                warmup_samples=10,
                min_event_pixels=12,
                min_path_length_px=8.0,
            ),
            [
                GrainAnchor(
                    grain_id=7,
                    center_xy=np.asarray([22.0, 40.0]),
                    radius_px=8.0,
                    observations=3,
                    circle_score=0.9,
                )
            ],
        )

        assert events
        event = events[0]
        assert event.root_distance_px <= 5.0
        assert event.grain_id == 7
        assert event.primary_for_grain
        assert event.arclength_px[-1] >= 30.0
        assert event.tortuosity < 1.3
        assert event.birth_order_correlation > 0.5
        assert event.birth_progress_samples > 0
        assert event.growth_step_count >= 5
        assert event.tip_step_median_px <= 3.0
        assert event.tip_step_max_px <= 8.0
        assert event.auto_accepted
        assert event.trajectory_accepted
        assert event.quality_status == "trajectory-growth"
        assert event_tip_at(event, event.germination_sample - 1)[1] is None
        germination_length, germination_tip, _ = event_tip_at(
            event, event.germination_sample
        )
        assert germination_tip is not None
        assert germination_length >= 3.0
        assert np.all(np.diff(event.path_birth) >= 0)

    def test_retains_short_germination_with_resolved_tip_trajectory(self):
        total = 60
        birth = np.full((50, 70), total, dtype=np.int32)
        for x in range(28, 40):
            birth[24:27, x] = 10 + 3 * (x - 28)
        preexisting = np.zeros_like(birth, dtype=np.uint8)
        cv.circle(preexisting, (20, 25), 6, 1, -1)

        events = extract_birth_events(
            birth,
            preexisting.astype(bool),
            AtlasConfig(
                warmup_samples=5,
                min_event_pixels=20,
                min_germination_length_px=3.0,
                min_path_length_px=12.0,
            ),
            [
                GrainAnchor(
                    grain_id=1,
                    center_xy=np.asarray([20.0, 25.0]),
                    radius_px=6.0,
                    observations=3,
                    circle_score=0.9,
                )
            ],
        )

        assert len(events) == 1
        event = events[0]
        assert 3.0 <= event.arclength_px[-1] < 12.0
        assert event.auto_accepted
        assert event.trajectory_accepted
        assert event.quality_status == "trajectory-growth"
        assert "short-after-rim-exit" in event.quality_flags

    def test_short_trajectory_promotion_does_not_change_nonprimary_event(self):
        primary = SimpleNamespace(
            primary_for_grain=True,
            auto_accepted=True,
            trajectory_accepted=False,
            quality_flags=("short-after-rim-exit",),
            quality_status="emergence-only",
        )
        nonprimary = SimpleNamespace(
            primary_for_grain=False,
            auto_accepted=True,
            trajectory_accepted=False,
            quality_flags=("short-after-rim-exit",),
            quality_status="emergence-only",
        )

        _promote_resolved_short_trajectories([primary, nonprimary])

        assert primary.trajectory_accepted
        assert primary.quality_status == "trajectory-growth"
        assert not nonprimary.trajectory_accepted

    def test_component_retains_a_hypothesis_for_each_attached_grain(self):
        total = 60
        birth = np.full((70, 120), total, dtype=np.int32)
        for x in range(28, 93):
            distance_from_nearest_root = min(x - 28, 92 - x)
            birth[34:37, x] = 10 + distance_from_nearest_root
        preexisting = np.zeros_like(birth, dtype=np.uint8)
        cv.circle(preexisting, (20, 35), 6, 1, -1)
        cv.circle(preexisting, (100, 35), 6, 1, -1)
        grains = [
            GrainAnchor(
                grain_id=1,
                center_xy=np.asarray([20.0, 35.0]),
                radius_px=6.0,
                observations=3,
                circle_score=0.9,
            ),
            GrainAnchor(
                grain_id=2,
                center_xy=np.asarray([100.0, 35.0]),
                radius_px=6.0,
                observations=3,
                circle_score=0.9,
            ),
        ]

        events = extract_birth_events(
            birth,
            preexisting.astype(bool),
            AtlasConfig(
                warmup_samples=5,
                min_event_pixels=20,
                min_path_length_px=8.0,
            ),
            grains,
        )

        assert {event.grain_id for event in events} == {1, 2}
        assert {event.component_id for event in events} == {0}
        assert {event.component_root_count for event in events} == {2}

    def test_component_ownership_keeps_only_one_coherent_overlapping_path(self):
        def event(
            grain_id,
            path,
            *,
            accepted=True,
            trajectory=False,
            flags=(),
        ):
            return SimpleNamespace(
                component_id=4,
                grain_id=grain_id,
                path_xy=np.asarray(path, dtype=float),
                auto_accepted=accepted,
                trajectory_accepted=trajectory,
                quality_status="trajectory-growth" if trajectory else "atlas-event",
                quality_flags=flags,
            )

        coherent = event(
            1,
            [[x, 10] for x in range(20)],
            trajectory=True,
        )
        competing = event(2, [[x, 11] for x in range(20)])
        distinct = event(3, [[10, y] for y in range(20)])
        detached = event(
            4,
            [[x, 10] for x in range(20)],
            accepted=False,
            flags=("detached-from-grain",),
        )
        events = [coherent, competing, distinct, detached]

        _resolve_component_ownership(events, AtlasConfig())

        assert coherent.auto_accepted
        assert not competing.auto_accepted
        assert "ambiguous-component-ownership" in competing.quality_flags
        assert distinct.auto_accepted
        assert "ambiguous-component-ownership" not in detached.quality_flags

    def test_distinct_coherent_paths_from_one_grain_require_review(self):
        def event(path, *, trajectory, score=1.0):
            return SimpleNamespace(
                grain_id=7,
                path_xy=np.asarray(path, dtype=float),
                auto_accepted=True,
                trajectory_accepted=trajectory,
                quality_status="trajectory-growth" if trajectory else "atlas-event",
                quality_flags=(),
                score=score,
            )

        left = event([[x, 10] for x in range(20)], trajectory=True)
        right = event([[10, y] for y in range(20)], trajectory=True)

        _resolve_grain_event_ambiguity([left, right], AtlasConfig())

        assert not left.auto_accepted
        assert not right.auto_accepted
        assert "ambiguous-multiple-events-for-grain" in left.quality_flags
        assert "ambiguous-multiple-events-for-grain" in right.quality_flags

    def test_unique_trajectory_wins_over_a_distinct_weaker_grain_event(self):
        coherent = SimpleNamespace(
            grain_id=7,
            path_xy=np.asarray([[x, 10] for x in range(20)], dtype=float),
            auto_accepted=True,
            trajectory_accepted=True,
            quality_status="trajectory-growth",
            quality_flags=(),
            score=5.0,
        )
        weaker = SimpleNamespace(
            grain_id=7,
            path_xy=np.asarray([[10, y] for y in range(20)], dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=(),
            score=20.0,
        )

        _resolve_grain_event_ambiguity([coherent, weaker], AtlasConfig())

        assert coherent.auto_accepted
        assert coherent.trajectory_accepted
        assert not weaker.auto_accepted
        assert "secondary-event-for-grain" in weaker.quality_flags

    def test_overlapping_repeats_keep_the_stronger_observation(self):
        weaker = SimpleNamespace(
            grain_id=7,
            path_xy=np.asarray([[x, 10] for x in range(20)], dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=(),
            score=5.0,
        )
        stronger = SimpleNamespace(
            grain_id=7,
            path_xy=np.asarray([[x, 11] for x in range(20)], dtype=float),
            auto_accepted=True,
            trajectory_accepted=False,
            quality_status="germination-only",
            quality_flags=(),
            score=20.0,
        )

        _resolve_grain_event_ambiguity([weaker, stronger], AtlasConfig())

        assert not weaker.auto_accepted
        assert stronger.auto_accepted
        assert "secondary-event-for-grain" in weaker.quality_flags

    def test_path_length_cannot_loop_around_a_compact_object(self):
        birth = np.full((80, 80), 60, dtype=np.int32)
        preexisting = cv.circle(
            np.zeros_like(birth, dtype=np.uint8), (40, 40), 7, 1, -1
        ).astype(bool)
        # A new ring can produce a very long wall-following skeleton, but it
        # is not a valid open pollen tube and must fail the path-shape gate.
        cv.circle(birth, (40, 40), 18, 20, 3)
        events = extract_birth_events(
            birth,
            preexisting,
            AtlasConfig(
                warmup_samples=10,
                min_event_pixels=10,
                min_path_length_px=8.0,
                max_path_tortuosity=2.0,
            ),
        )
        assert events == []
