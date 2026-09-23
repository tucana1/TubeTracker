"""Tests for the space-time growth-front prototype."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pandas as pd

from prototypes.v23_owner_memory.temporal_front import (
    apply_emergence_gate,
    connected_direct_front_indices,
    consistency_duplicate_losers,
    cross_section_novelty,
    durable_rim_emergence_sample,
    image_patch,
    match_owner_center,
    persistent_directional_emergence,
    persistent_emergence_sample,
    rim_extension_history,
    select_emergence_consensus,
    tube_state_profile,
    temporal_duplicate_claims,
    temporal_foreign_pollen_contacts,
)
from prototypes.v29_causal_growth_front.run import (
    MaturePathCandidate,
    NativeGrayFrameCache,
    OwnerGeometry,
    _audit_export_consistency,
    _blocking_centers_for_owner,
    _causal_status,
    _certify_native_candidate_geometries,
    _exclude_short_consensus_paths,
    _forecast_stability_veto_trips,
    _gate_rescue_recenter,
    _load_forecast_tip_holds,
    _load_sticky_tip_evidence,
    _load_forecast_veto_counts,
    _measurement_eligible,
    _recenter_length_change_is_lateral,
    _render_foreign_branch_audit,
    _resolve_final_duplicate_claims,
    _select_native_boundary_rescues,
)
from prototypes.v29_causal_growth_front.bootstrap import (
    OwnerBodyCertificates,
    certify_owner_pollen_bodies,
    identity_confidence_tier,
    neutral_owner_stub,
    reconcile_owner_identity_certificates,
    resolve_owner_motion_mode,
    select_persistent_identity_tracks,
    source_visibility_mask,
)
from tubetracker.pollen_motion import PollenMotionConfig
from tubetracker.growth_front import (
    confirmation_bounded_path_front,
    monotone_front,
    path_locked_temporal_front,
    resample_polyline,
    trace_extension_corridor,
    visible_path_front,
)
from tubetracker.causal_growth_front import (
    CausalPathHypothesis,
    CausalGrowthFrontResult,
    NativePathGrowthCertificate,
    causal_changepoint_front,
    certify_growth_direction,
    certify_native_path_growth,
    certify_owner_path_topology,
    certify_rooted_growth,
    dynamic_path_novelty_profiles,
    fit_scalar_path_deformation,
    fit_prior_constrained_native_front,
    project_inextensible_paths,
    select_decisive_causal_path,
)


def test_export_consistency_accepts_matching_monotone_geometry():
    """Accepted time rows must have an equally long exported centerline."""

    audit = _audit_export_consistency(
        [
            {
                "pollen_id": 3,
                "causal_measurement_accepted": 1,
                "owner_path_topology_certificate": (
                    "owner-path-topology-certified"
                ),
            }
        ],
        [
            {
                "pollen_id": 3,
                "sample_index": 0,
                "tube_length_px": 0.0,
                "accepted": 0,
            },
            {
                "pollen_id": 3,
                "sample_index": 1,
                "tube_length_px": 12.0,
                "accepted": 1,
            },
        ],
        [
            {
                "pollen_id": 3,
                "sample_index": 1,
                "arc_length_px": 0.0,
            },
            {
                "pollen_id": 3,
                "sample_index": 1,
                "arc_length_px": 12.0,
            },
        ],
    )

    assert audit["accepted_owner_count"] == 1
    assert audit["accepted_measurement_count"] == 1
    assert audit["mismatched_centerline_count"] == 0


def test_export_consistency_rejects_mismatched_geometry():
    """A length without a matching coordinate path must stop the export."""

    with np.testing.assert_raises_regex(RuntimeError, "missing_centerline_keys"):
        _audit_export_consistency(
            [
                {
                    "pollen_id": 3,
                    "causal_measurement_accepted": 1,
                    "owner_path_topology_certificate": (
                        "owner-path-topology-certified"
                    ),
                }
            ],
            [
                {
                    "pollen_id": 3,
                    "sample_index": 1,
                    "tube_length_px": 12.0,
                    "accepted": 1,
                }
            ],
            [],
        )


def test_export_consistency_uses_visible_length_for_censored_coordinates():
    """An edge-censored material estimate may exceed its in-frame polyline."""

    audit = _audit_export_consistency(
        [
            {
                "pollen_id": 4,
                "causal_measurement_accepted": 1,
                "owner_path_topology_certificate": (
                    "owner-path-topology-certified"
                ),
            }
        ],
        [
            {
                "pollen_id": 4,
                "sample_index": 2,
                "tube_length_px": 15.0,
                "visible_centerline_length_px": 13.5,
                "accepted": 1,
            }
        ],
        [
            {
                "pollen_id": 4,
                "sample_index": 2,
                "arc_length_px": 13.5,
            }
        ],
    )

    assert audit["mismatched_centerline_count"] == 0


def test_owner_path_topology_accepts_a_curved_outward_tube():
    """A curved tube may turn after making a prompt irreversible body exit."""

    path = np.asarray(
        [[5.0, 0.0], [7.0, 0.5], [10.0, 2.0], [14.0, 5.0], [18.0, 9.0]]
    )

    certificate = certify_owner_path_topology(
        path,
        np.zeros(2),
        owner_radius_px=5.0,
    )

    assert certificate.accepted
    assert certificate.reason == "owner-path-topology-certified"
    assert certificate.first_halo_exit_index == 2


def test_owner_path_topology_accepts_a_short_clean_germination():
    """A ten-pixel protrusion can be valid when it promptly clears the body halo."""

    certificate = certify_owner_path_topology(
        np.asarray([[6.3, 0.0], [8.9, 0.2], [10.3, 0.5]]),
        np.zeros(2),
        owner_radius_px=5.625,
    )

    assert certificate.accepted


def test_owner_path_topology_rejects_a_tangential_rim_walk():
    """A long trip around the body cannot borrow a later outward branch."""

    angles = np.linspace(0.0, 0.9 * np.pi, 18)
    rim = np.column_stack((5.2 * np.cos(angles), 5.2 * np.sin(angles)))
    path = np.vstack((rim, [[-8.5, 4.0], [-13.0, 5.0], [-18.0, 6.0]]))

    certificate = certify_owner_path_topology(
        path,
        np.zeros(2),
        owner_radius_px=5.0,
    )

    assert not certificate.accepted
    assert "tangential-owner-rim-walk" in certificate.reason


def test_owner_path_topology_rejects_a_curve_through_the_grain():
    """Paired contrast on a pollen rim is invalid when the path enters its body."""

    path = np.asarray(
        [[6.0, 0.0], [4.0, 0.0], [2.0, 0.0], [4.0, 1.0], [8.0, 3.0], [14.0, 6.0]]
    )

    certificate = certify_owner_path_topology(
        path,
        np.zeros(2),
        owner_radius_px=5.0,
    )

    assert not certificate.accepted
    assert "path-enters-owner-body" in certificate.reason


def test_owner_path_topology_rejects_halo_reentry():
    """A candidate cannot leave the owner and then return to its rim."""

    path = np.asarray(
        [[5.0, 0.0], [8.0, 0.0], [12.0, 1.0], [16.0, 2.0], [6.0, 2.0]]
    )

    certificate = certify_owner_path_topology(
        path,
        np.zeros(2),
        owner_radius_px=5.0,
    )

    assert not certificate.accepted
    assert "path-reenters-owner-halo" in certificate.reason


def test_native_path_certificate_accepts_complete_timely_growth():
    """Native walls can rescue a path when they reproduce its extent and onset."""

    certificate = certify_native_path_growth(
        np.linspace(0.0, 38.0, 30),
        native_onset_sample=12,
        causal_onset_sample=10,
        total_length_px=40.0,
    )
    assert certificate.accepted
    assert certificate.completion_fraction == 0.95
    assert certificate.onset_delay_samples == 2


def test_native_path_certificate_rejects_partial_and_corrects_late_evidence():
    """Complete native growth corrects timing, while a fragment stays rejected."""

    partial = certify_native_path_growth(
        np.linspace(0.0, 20.0, 30),
        native_onset_sample=12,
        causal_onset_sample=10,
        total_length_px=40.0,
    )
    delayed = certify_native_path_growth(
        np.linspace(0.0, 38.0, 30),
        native_onset_sample=25,
        causal_onset_sample=10,
        total_length_px=40.0,
    )
    assert not partial.accepted
    assert partial.reason == "insufficient-native-path-completion"
    assert delayed.accepted
    assert delayed.reason == "native-path-growth-verified-with-later-onset"


def test_empty_causal_candidate_is_withheld_without_opening_a_movie():
    """A native candidate with no causal prefix is a normal rejected case."""

    result = CausalGrowthFrontResult(
        birth_samples=np.full(2, 1, dtype=np.int32),
        front_indices=np.asarray([-1], dtype=np.int32),
        direct_support_mask=np.zeros(2, dtype=bool),
        eventual_support_mask=np.zeros(2, dtype=bool),
        feasible=False,
        reason="no-causal-path",
        objective_score=0.0,
        final_point_index=-1,
        final_length_px=0.0,
        direct_support_fraction=0.0,
        eventual_support_fraction=0.0,
    )
    candidate = MaturePathCandidate(
        source="empty",
        arclength_px=np.asarray([0.0, 1.0]),
        dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
        root_distance_px=0.0,
        radial_excursion_px=0.0,
        foreign_contact_owner_id=None,
        uncensored_length_px=1.0,
        result=result,
    )

    certificates = _certify_native_candidate_geometries(
        Path("/movie/does/not/exist.mp4"),
        [candidate],
        np.asarray([0]),
        np.zeros((1, 2)),
        1.0,
    )
    certificate = certificates[id(candidate)]

    assert not certificate.accepted
    assert certificate.reason == "insufficient-native-quality-observations"


def test_foreign_branch_audit_allows_unselected_reference_owner(tmp_path):
    """A selected-owner probe can cite a pollen outside the traced subset."""

    result = CausalGrowthFrontResult(
        birth_samples=np.asarray([0, 0], dtype=np.int32),
        front_indices=np.asarray([1], dtype=np.int32),
        direct_support_mask=np.ones(2, dtype=bool),
        eventual_support_mask=np.ones(2, dtype=bool),
        feasible=True,
        reason="ok",
        objective_score=1.0,
        final_point_index=1,
        final_length_px=1.0,
        direct_support_fraction=1.0,
        eventual_support_fraction=1.0,
    )
    owner = OwnerGeometry(
        track_id=20,
        field_status="review_quality",
        original_measurements=pd.DataFrame(),
        mature_candidates=[],
        arclength_px=np.asarray([0.0, 1.0]),
        dynamic_paths_xy=np.asarray([[[5.0, 5.0], [8.0, 8.0]]]),
        result=result,
        foreign_branch_owner_id=62,
        foreign_branch_capture_point=1,
    )

    audit = _render_foreign_branch_audit(
        tmp_path,
        np.full((1, 16, 16), 127, dtype=np.uint8),
        [owner],
    )

    assert audit is not None
    assert audit.exists()


def test_only_verified_foreign_pollen_bodies_mask_owner_geometry():
    """A real provisional pollen blocks tubes while a false strong point cannot."""

    centers = {
        owner_id: np.zeros((2, 2), dtype=np.float64)
        for owner_id in (1, 2, 3)
    }
    tiers = {1: "high", 2: "supported", 3: "provisional"}
    bodies = {1: "verified", 2: "rejected", 3: "verified"}

    assert set(_blocking_centers_for_owner(centers, bodies, tiers, 1)) == {1, 3}
    assert set(_blocking_centers_for_owner(centers, bodies, tiers, 2)) == {1, 2, 3}
    assert set(_blocking_centers_for_owner(centers, bodies, tiers, 3)) == {1, 3}


def test_legacy_owner_masks_retain_explicit_tier_compatibility():
    """Old bootstrap fixtures remain reproducible instead of changing silently."""

    centers = {
        owner_id: np.zeros((2, 2), dtype=np.float64)
        for owner_id in (1, 2, 3)
    }
    tiers = {1: "high", 2: "supported", 3: "provisional"}
    legacy = {owner_id: "legacy-unverified" for owner_id in centers}

    assert set(_blocking_centers_for_owner(centers, legacy, tiers, 1)) == {1}
    assert set(_blocking_centers_for_owner(centers, legacy, tiers, 2)) == {1, 2}
    assert set(_blocking_centers_for_owner(centers, legacy, tiers, 3)) == {1, 2, 3}


def test_rejected_pollen_body_cannot_become_an_automatic_measurement():
    """Tube-like evidence at a false owner remains diagnostic only."""

    owner = OwnerGeometry(
        track_id=11,
        field_status="measured",
        original_measurements=pd.DataFrame(),
        mature_candidates=[],
        arclength_px=np.asarray([0.0, 1.0]),
        dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
        body_status="rejected",
        topology_certificate=SimpleNamespace(accepted=True),
    )

    assert not _measurement_eligible(owner)
    owner.body_status = "verified"
    assert _measurement_eligible(owner)


def test_native_frame_cache_decodes_repeated_source_frame_once(tmp_path):
    """Native analysis stages share one immutable decode per sampled frame."""

    movie = tmp_path / "native-cache.avi"
    writer = cv.VideoWriter(
        str(movie),
        cv.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (24, 16),
    )
    assert writer.isOpened()
    for value in (20, 80, 160):
        writer.write(np.full((16, 24, 3), value, dtype=np.uint8))
    writer.release()

    cache = NativeGrayFrameCache(movie)
    first = cache.read(1)
    second = cache.read(1)
    report = cache.report()
    cache.close()

    assert first is second
    assert not first.flags.writeable
    assert report["decoded_frame_count"] == 1
    assert report["cache_hit_count"] == 1
    assert report["decoder_retry_count"] == 0


def test_final_duplicate_claim_keeps_verified_body_over_false_coordinate():
    """A false point on a tube cannot suppress its verified pollen owner."""

    def owner(track_id, tier, body_status):
        return OwnerGeometry(
            track_id=track_id,
            field_status="review_quality",
            original_measurements=pd.DataFrame(),
            mature_candidates=[],
            arclength_px=np.asarray([0.0, 1.0]),
            dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
            identity_tier=tier,
            body_status=body_status,
        )

    trusted = owner(5, "high", "verified")
    weaker = owner(11, "supported", "rejected")
    audit = {"claims": [{"first_owner": 5, "second_owner": 11}]}

    withheld = _resolve_final_duplicate_claims([trusted, weaker], audit)

    assert withheld == [11]
    assert trusted.ownership_conflict_ids == []
    assert weaker.ownership_conflict_ids == [5]
    assert audit["claims"][0]["ownership_priority_winner_owner_id"] == 5
    assert audit["claims"][0]["ownership_priority_resolution"] == (
        "verified-body-priority"
    )


def test_final_duplicate_claim_withholds_two_verified_pollen_owners():
    """Body evidence cannot assign a shared tube between two genuine pollen."""

    owners = [
        OwnerGeometry(
            track_id=track_id,
            field_status="review_quality",
            original_measurements=pd.DataFrame(),
            mature_candidates=[],
            arclength_px=np.asarray([0.0, 1.0]),
            dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
            identity_tier=tier,
            body_status="verified",
        )
        for track_id, tier in ((117, "high"), (255, "provisional"))
    ]
    audit = {"claims": [{"first_owner": 117, "second_owner": 255}]}

    withheld = _resolve_final_duplicate_claims(owners, audit)

    assert withheld == [117, 255]
    assert audit["claims"][0]["ownership_priority_resolution"] == (
        "equal-body-evidence-withhold-both"
    )


def test_dense_bootstrap_keeps_only_persistent_semantic_owners():
    """Geometric-only and one-frame proposals cannot seed a dense field."""

    tracks = [
        {"track_id": 3, "observation_count": 5, "semantic_observation_count": 4,
         "centers_yx": [[0.0, 0.0]], "radii_px": [10.0]},
        {"track_id": 1, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[100.0, 100.0]], "radii_px": [10.0]},
        {"track_id": 2, "observation_count": 4, "semantic_observation_count": 4,
         "centers_yx": [[200.0, 200.0]], "radii_px": [10.0]},
        {"track_id": 4, "observation_count": 5, "semantic_observation_count": 0,
         "centers_yx": [[300.0, 300.0]], "radii_px": [10.0]},
    ]

    selected = select_persistent_identity_tracks(tracks, 5, 4)

    assert [track["track_id"] for track in selected] == [1, 3]


def test_dense_bootstrap_can_retain_single_learned_owner_without_geometry_only():
    """Recall mode keeps learned pollen while excluding geometric-only circles."""

    tracks = [
        {"track_id": 2, "observation_count": 1, "semantic_observation_count": 1,
         "centers_yx": [[0.0, 0.0]], "radii_px": [10.0]},
        {"track_id": 1, "observation_count": 5, "semantic_observation_count": 0,
         "centers_yx": [[100.0, 100.0]], "radii_px": [10.0]},
    ]

    selected = select_persistent_identity_tracks(tracks, 1, 1)

    assert [track["track_id"] for track in selected] == [2]
    assert identity_confidence_tier(selected[0]) == "provisional"


def test_dense_bootstrap_merges_mask_split_cluster_fragment():
    """A sliver mostly covered by a larger neighbor is not an owner."""

    tracks = [
        {"track_id": 89, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[726.8, 692.2]], "radii_px": [10.0]},
        {"track_id": 92, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[732.0, 694.0]], "radii_px": [14.3]},
    ]

    selected = select_persistent_identity_tracks(
        tracks, 3, 3, overlap_merge_fraction=0.5
    )

    assert 89 not in [track["track_id"] for track in selected]
    survivor = next(
        track for track in selected if track["track_id"] == 92
    )
    assert 89 in survivor.get("merged_fragment_ids", ())


def test_contested_cluster_identity_cannot_be_automatically_verified():
    """A persistent census identity overlapping a neighbor stays diagnostic."""

    tracks = [
        {"track_id": 89, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[726.8, 692.2]], "radii_px": [10.0]},
        {"track_id": 92, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[744.7, 696.9]], "radii_px": [14.3]},
    ]
    body = OwnerBodyCertificates(
        statuses=np.asarray(["verified", "verified"]),
        verified=np.asarray([True, True]),
        rejected=np.asarray([False, False]),
        median_radial_contrast=np.zeros(2),
        median_angular_boundary_fraction=np.zeros(2),
        median_opposite_boundary_fraction=np.zeros(2),
        median_valid_angular_fraction=np.zeros(2),
        valid_sample_counts=np.zeros(2, dtype=np.int32),
        sample_indices=np.zeros((2, 1), dtype=np.int32),
        radial_contrast_samples=np.zeros((2, 1)),
        angular_boundary_samples=np.zeros((2, 1)),
        opposite_boundary_samples=np.zeros((2, 1)),
        valid_angular_samples=np.zeros((2, 1)),
    )

    certificates = reconcile_owner_identity_certificates(
        tracks, body, "pre-growth", 3
    )

    assert list(certificates.statuses) == ["indeterminate", "verified"]
    assert list(certificates.bases) == [
        "contested-cluster-identity",
        "pre-growth-semantic-consensus",
    ]
    assert not list(certificates.verified)[0]


def test_duplicate_seeds_on_one_grain_are_contested():
    """Two near-coincident census seeds on one body cannot both verify."""

    from prototypes.v29_causal_growth_front.bootstrap import (
        _flag_contested_identity_tracks,
    )

    # Dense P65/P69: eye-verified as two seeds on ONE grain (centers 1.54
    # small-radii apart, lens overlap only 0.23 of the small mask, so the
    # 0.08 overlap rule alone is not the mechanism here).
    tracks = [
        {"track_id": 65, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[561.3, 408.1]], "radii_px": [12.7]},
        {"track_id": 69, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[572.9, 419.6]], "radii_px": [10.6]},
    ]
    flagged = _flag_contested_identity_tracks(tracks)
    assert bool(flagged[1]) and not bool(flagged[0])

    # Genuinely separated grains stay untouched.
    far = [
        {"track_id": 47, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[0.0, 0.0]], "radii_px": [14.4]},
        {"track_id": 50, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[56.0, 0.0]], "radii_px": [10.7]},
    ]
    assert not bool(_flag_contested_identity_tracks(far).any())


def test_background_seed_cannot_take_pregrowth_verification():
    """A census seed on empty background stays an indeterminate diagnostic."""

    from prototypes.v29_causal_growth_front.bootstrap import (
        OwnerBodyCertificates,
        reconcile_owner_identity_certificates,
    )

    # Dense P41: stable 5/5-frame seed whose interior disk never goes dark
    # (p10 161) against real grains at p10 75-152.
    tracks = [
        {"track_id": 76, "observation_count": 5, "semantic_observation_count": 5,
         "centers_yx": [[605.7, 396.3]], "radii_px": [14.5]},
    ]
    empty = np.zeros((1, 1), dtype=np.int32)
    body = OwnerBodyCertificates(
        statuses=np.asarray(["verified"]),
        verified=np.asarray([True]),
        rejected=np.asarray([False]),
        median_radial_contrast=np.zeros(1),
        median_angular_boundary_fraction=np.zeros(1),
        median_opposite_boundary_fraction=np.zeros(1),
        median_valid_angular_fraction=np.zeros(1),
        valid_sample_counts=np.zeros(1, dtype=np.int32),
        sample_indices=empty,
        radial_contrast_samples=np.zeros((1, 1)),
        angular_boundary_samples=np.zeros((1, 1)),
        opposite_boundary_samples=np.zeros((1, 1)),
        valid_angular_samples=np.zeros((1, 1)),
    )

    certificates = reconcile_owner_identity_certificates(
        tracks, body, "pre-growth", 3,
        seed_interior_darkness=np.asarray([161.0]),
    )
    assert list(certificates.statuses) == ["indeterminate"]
    assert list(certificates.bases) == ["background-seed-no-grain-interior"]
    assert not bool(certificates.verified[0])

    # A dark interior keeps automatic verification.
    certificates = reconcile_owner_identity_certificates(
        tracks, body, "pre-growth", 3,
        seed_interior_darkness=np.asarray([120.0]),
    )
    assert list(certificates.statuses) == ["verified"]
    assert list(certificates.bases) == ["pre-growth-semantic-consensus"]


def test_dense_bootstrap_stub_stays_inside_the_source_field():
    """An edge owner receives an inward placeholder, never invented tube length."""

    stub = neutral_owner_stub(np.asarray((4.0, 95.0)), (100, 100))

    assert np.all(stub >= 0.0)
    assert np.all(stub <= 99.0)
    assert np.isclose(np.linalg.norm(stub[1] - stub[0]), 5.0)


def test_dense_bootstrap_visibility_uses_the_real_source_boundary():
    """Reflected registration padding cannot make an off-screen owner visible."""

    centers = np.asarray(
        [
            [
                (50.0, 50.0),
                (5.0, 50.0),
                (50.0, 105.0),
            ]
        ]
    )

    visible = source_visibility_mask(centers, (100, 100), margin_px=10.0)

    assert visible.tolist() == [[True, False, False]]


def test_dense_bootstrap_distinguishes_verified_rejected_and_edge_bodies():
    """Body evidence should be positive, negative, or visibility-indeterminate."""

    frame = np.full((81, 91), 180, dtype=np.uint8)
    cv.circle(frame, (20, 20), 10, 105, -1, cv.LINE_AA)
    cv.line(frame, (32, 50), (68, 50), 105, 3, cv.LINE_AA)
    cv.circle(frame, (2, 40), 10, 105, -1, cv.LINE_AA)
    frames = np.repeat(frame[None], 7, axis=0)
    owner_centers = np.asarray([[20, 20], [50, 50], [40, 2]], dtype=float)
    centers = np.repeat(owner_centers[:, None], len(frames), axis=1)

    result = certify_owner_pollen_bodies(
        frames,
        centers,
        np.asarray([3, 3, 3]),
        10.0,
    )

    assert result.statuses.tolist() == ["verified", "rejected", "indeterminate"]
    assert result.verified.tolist() == [True, False, False]
    assert result.rejected.tolist() == [False, True, False]
    assert result.valid_sample_counts.tolist() == [5, 5, 0]
    assert result.sample_indices.tolist() == [[1, 2, 3, 4, 5]] * 3


def test_pregrowth_semantic_consensus_supersedes_ambiguous_radial_shape():
    """A repeated pre-tube owner must not be rejected as a tube fragment."""

    frame = np.full((61, 61), 180, dtype=np.uint8)
    cv.line(frame, (10, 30), (50, 30), 105, 3, cv.LINE_AA)
    frames = np.repeat(frame[None], 5, axis=0)
    body = certify_owner_pollen_bodies(
        frames,
        np.repeat(np.asarray([[[30.0, 30.0]]]), 5, axis=1),
        np.asarray([2]),
        10.0,
    )
    track = {
        "track_id": 1,
        "observation_count": 5,
        "semantic_observation_count": 5,
        "centers_yx": [[30.0, 30.0]],
        "radii_px": [10.0],
    }

    pregrowth = reconcile_owner_identity_certificates(
        [track], body, "pre-growth"
    )
    unconstrained = reconcile_owner_identity_certificates(
        [track], body, "unconstrained"
    )

    assert body.statuses.tolist() == ["rejected"]
    assert pregrowth.statuses.tolist() == ["verified"]
    assert pregrowth.bases.tolist() == ["pre-growth-semantic-consensus"]
    assert pregrowth.pregrowth_semantic_consensus.tolist() == [True]
    assert unconstrained.statuses.tolist() == ["rejected"]
    assert unconstrained.bases.tolist() == ["radial-body-certificate"]


def test_weak_pregrowth_sighting_does_not_override_body_rejection():
    """One early semantic fragment remains diagnostic rather than promoted."""

    frame = np.full((61, 61), 180, dtype=np.uint8)
    cv.line(frame, (10, 30), (50, 30), 105, 3, cv.LINE_AA)
    body = certify_owner_pollen_bodies(
        np.repeat(frame[None], 5, axis=0),
        np.repeat(np.asarray([[[30.0, 30.0]]]), 5, axis=1),
        np.asarray([2]),
        10.0,
    )

    result = reconcile_owner_identity_certificates(
        [{
            "track_id": 1,
            "observation_count": 1,
            "semantic_observation_count": 1,
            "centers_yx": [[30.0, 30.0]],
            "radii_px": [10.0],
        }],
        body,
        "pre-growth",
    )

    assert result.statuses.tolist() == ["rejected"]
    assert result.pregrowth_semantic_consensus.tolist() == [False]


def test_motion_ambiguity_audit_names_unsupported_low_visibility_owners():
    """Owners with no geometric support and weak tracking fail closed."""

    from prototypes.v29_causal_growth_front.bootstrap import (
        _audit_owner_motion_ambiguity,
    )

    tracks = [
        {"track_id": 9, "centers_yx": [[0.0, 0.0]], "radii_px": [10.0]},
        {"track_id": 3, "centers_yx": [[500.0, 500.0]], "radii_px": [10.0]},
    ]
    assigned = np.asarray([0.025, 0.975])
    costs = np.full((2, 4), 1.15)
    costs[1] = 0.4
    scores = np.full((2, 4), 0.97)
    scores[0, 0] = 0.0
    visible = np.ones((2, 4), dtype=bool)
    visible[0] = False

    audit = _audit_owner_motion_ambiguity(
        tracks, assigned, costs, scores, visible
    )

    assert list(audit["ambiguous_owner_ids"]) == [9]  # type: ignore[union-attr]
    assignment_map = audit["assignment_fraction_p10_by_owner"]  # type: ignore[assignment]
    visibility_map = audit["point_tracker_visibility_by_owner"]  # type: ignore[assignment]
    score_map = audit["template_score_p10_by_owner"]  # type: ignore[assignment]
    assert assignment_map[9] == 0.025  # type: ignore[index]
    assert visibility_map[9] == 0.0  # type: ignore[index]
    assert score_map[3] == 0.97  # type: ignore[index]


def test_unsupported_contested_identity_is_demoted_without_deletion():
    """A contested owner with no mature-field support stays diagnostic."""

    from prototypes.v29_causal_growth_front.bootstrap import (
        _demote_unsupported_identity_tracks,
    )

    tracks = [
        {"track_id": 89, "centers_yx": [[726.8, 692.2]], "radii_px": [10.0]},
        {"track_id": 92, "centers_yx": [[744.7, 696.9]], "radii_px": [14.3]},
    ]
    body = OwnerBodyCertificates(
        statuses=np.asarray(["verified", "verified"]),
        verified=np.asarray([True, True]),
        rejected=np.asarray([False, False]),
        median_radial_contrast=np.zeros(2),
        median_angular_boundary_fraction=np.zeros(2),
        median_opposite_boundary_fraction=np.zeros(2),
        median_valid_angular_fraction=np.zeros(2),
        valid_sample_counts=np.zeros(2, dtype=np.int32),
        sample_indices=np.zeros((2, 1), dtype=np.int32),
        radial_contrast_samples=np.zeros((2, 1)),
        angular_boundary_samples=np.zeros((2, 1)),
        opposite_boundary_samples=np.zeros((2, 1)),
        valid_angular_samples=np.zeros((2, 1)),
    )
    certificates = reconcile_owner_identity_certificates(
        tracks, body, "pre-growth", 3
    )
    demoted = _demote_unsupported_identity_tracks(
        tracks, certificates, np.asarray([0.025, 0.48])
    )

    assert demoted.statuses.tolist() == ["indeterminate", "verified"]
    assert demoted.bases.tolist() == [
        "contested-cluster-identity",
        "radial-body-certificate",
    ]


def test_owner_motion_auto_mode_has_an_explicit_optional_fallback(tmp_path):
    """Missing optional weights fall back only when automatic mode is requested."""

    config = PollenMotionConfig(checkpoint=tmp_path / "missing.pth")

    resolved, reason = resolve_owner_motion_mode(
        "auto", "pre-growth", config
    )

    assert resolved == "template"
    assert "checkpoint not installed" in reason
    with np.testing.assert_raises_regex(RuntimeError, "checkpoint not installed"):
        resolve_owner_motion_mode("cotracker", "pre-growth", config)
    assert resolve_owner_motion_mode("template", "pre-growth", config) == (
        "template",
        None,
    )


def test_consensus_cannot_replace_verified_frame_path_with_short_branch():
    """Temporal consensus may stabilize or extend, but not truncate, geometry."""

    def candidate(source: str, length: float) -> MaturePathCandidate:
        result = CausalGrowthFrontResult(
            birth_samples=np.zeros(2, dtype=np.int32),
            front_indices=np.asarray([1], dtype=np.int32),
            direct_support_mask=np.ones(2, dtype=bool),
            eventual_support_mask=np.ones(2, dtype=bool),
            feasible=True,
            reason="causal-path",
            objective_score=1.0,
            final_point_index=1,
            final_length_px=length,
            direct_support_fraction=1.0,
            eventual_support_fraction=1.0,
        )
        return MaturePathCandidate(
            source=source,
            arclength_px=np.asarray([0.0, length]),
            dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
            root_distance_px=0.0,
            radial_excursion_px=length,
            foreign_contact_owner_id=None,
            uncensored_length_px=length,
            result=result,
            native_quality_certificate=SimpleNamespace(
                median_coverage=0.8,
            ),
        )

    frame = candidate("native-boundary-2-rigid", 72.0)
    short_consensus = candidate("native-consensus-8-rigid", 29.0)
    extending_consensus = candidate("native-consensus-9-rigid", 78.0)
    retained, rejected, threshold = _exclude_short_consensus_paths(
        [frame, short_consensus, extending_consensus],
        0.90,
        0.02,
    )

    assert threshold == 64.8
    assert [item.source for item in retained] == [
        "native-boundary-2-rigid",
        "native-consensus-9-rigid",
    ]
    assert [item.source for item in rejected] == [
        "native-consensus-8-rigid"
    ]


def test_native_front_recovers_a_small_systematic_tip_offset():
    """The native prefix boundary can refine a nearby coarse material front."""

    arc = np.arange(60, dtype=np.float32)
    truth = np.minimum(np.maximum(np.arange(80) - 12, 0), 50).astype(np.float32)
    prior = np.maximum.accumulate(np.clip(truth - 2.0, 0.0, arc[-1]))
    support = np.zeros((len(truth), len(arc)), dtype=np.float32)
    for sample, length in enumerate(truth):
        support[sample, arc <= length] = 8.0
    result = fit_prior_constrained_native_front(
        support,
        np.full(len(truth), 2.0),
        arc,
        prior,
    )
    assert result.accepted
    assert np.mean(np.abs(result.lengths_px[20:] - truth[20:])) < 1.0
    assert np.all(np.diff(result.lengths_px) >= 0.0)


def test_native_front_ignores_distal_clutter_and_transient_occlusion():
    """A foreign distal block and brief support loss cannot change topology."""

    arc = np.arange(70, dtype=np.float32)
    truth = np.minimum(np.maximum(np.arange(90) - 15, 0), 48).astype(np.float32)
    prior = np.maximum.accumulate(np.clip(truth + 1.0, 0.0, arc[-1]))
    support = np.zeros((len(truth), len(arc)), dtype=np.float32)
    for sample, length in enumerate(truth):
        support[sample, arc <= length] = 8.0
    support[35:45] = 0.0
    support[25:, 62:68] = 10.0
    result = fit_prior_constrained_native_front(
        support,
        np.full(len(truth), 2.0),
        arc,
        prior,
    )
    assert np.all(np.diff(result.lengths_px) >= 0.0)
    assert result.lengths_px[-1] <= 52.0
    assert np.max(np.abs(result.lengths_px - prior)) <= 4.0


def test_native_front_uses_arrival_evidence_to_ignore_static_clutter():
    """A static tube-like patch cannot impersonate a newly arriving front."""

    arc = np.arange(70, dtype=np.float32)
    truth = np.minimum(np.maximum(np.arange(100) - 20, 0), 45).astype(np.float32)
    prior = np.maximum.accumulate(np.clip(truth + 3.0, 0.0, arc[-1]))
    support = np.zeros((len(truth), len(arc)), dtype=np.float32)
    for sample, length in enumerate(truth):
        support[sample, arc <= length] = 8.0
    support[:, 48:53] = 10.0
    result = fit_prior_constrained_native_front(
        support,
        np.full(len(truth), 2.0),
        arc,
        prior,
    )
    assert result.accepted
    assert abs(result.lengths_px[-1] - truth[-1]) <= 1.0
    assert np.mean(np.abs(result.lengths_px[30:] - truth[30:])) < 1.5


def test_native_front_withholds_an_uninformative_profile():
    """No native contrast leaves the prior untouched as an unverified estimate."""

    arc = np.arange(40, dtype=np.float32)
    prior = np.minimum(np.arange(50), 30).astype(np.float32)
    support = np.full((len(prior), len(arc)), 2.0, dtype=np.float32)
    result = fit_prior_constrained_native_front(
        support,
        np.full(len(prior), 2.0),
        arc,
        prior,
    )
    assert not result.accepted
    assert result.reason == "insufficient-native-front-gain"


def front_args(**overrides):
    values = {
        "arc_step_px": 1.5,
        "evidence_midpoint": 2.5,
        "coverage_cost": 0.15,
        "max_growth_px_per_frame": 7.5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def synthetic_kymograph(frame_count, columns, front, tube_value=8.0, noise=0.0, seed=0):
    """Evidence below the given per-frame front column, background elsewhere."""
    rng = np.random.default_rng(seed)
    kymograph = np.zeros((frame_count, columns), np.float32)
    for frame in range(frame_count):
        kymograph[frame, : front[frame] + 1] = tube_value
    if noise:
        kymograph += rng.normal(0.0, noise, kymograph.shape).astype(np.float32)
    return kymograph


class ResamplePolylineTests:
    def test_uniform_spacing_on_straight_line(self):
        points = np.array([[0.0, 0.0], [10.0, 0.0]])
        resampled = resample_polyline(points, step=1.0)
        steps = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        assert np.allclose(steps, 1.0)
        assert np.allclose(resampled[0], [0.0, 0.0])

    def test_preserves_total_arclength_within_step(self):
        angles = np.linspace(0.0, np.pi, 50)
        points = np.stack([np.cos(angles) * 20.0, np.sin(angles) * 20.0], axis=1)
        resampled = resample_polyline(points, step=1.5)
        original = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        recovered = np.linalg.norm(np.diff(resampled, axis=0), axis=1).sum()
        assert abs(original - recovered) <= 1.5


class MonotoneFrontTests:
    def test_recovers_linear_growth(self):
        frame_count, columns = 120, 80
        truth = np.minimum(np.arange(frame_count) // 2, columns - 1)
        kymograph = synthetic_kymograph(frame_count, columns, truth, noise=0.5)
        front = monotone_front(kymograph, front_args())
        assert np.all(np.diff(front) >= 0)
        # median-filter warmup blurs the first frames; judge the settled region
        assert np.abs(front[10:] - truth[10:]).mean() < 3.0

    def test_front_is_monotone_despite_transient_occlusion(self):
        frame_count, columns = 120, 80
        truth = np.minimum(np.arange(frame_count) // 2, columns - 1)
        kymograph = synthetic_kymograph(frame_count, columns, truth, noise=0.5)
        kymograph[60:70] = 0.0  # blur burst wipes all evidence for ten frames
        front = monotone_front(kymograph, front_args())
        assert np.all(np.diff(front) >= 0)
        assert abs(front[-1] - truth[-1]) <= 3

    def test_foreign_block_above_front_is_not_annexed(self):
        frame_count, columns = 120, 80
        truth = np.full(frame_count, 30, dtype=int)
        kymograph = synthetic_kymograph(frame_count, columns, truth, noise=0.5)
        # a crossing tube appears far beyond the tip, disconnected from the front
        kymograph[80:, 60:70] = 8.0
        front = monotone_front(kymograph, front_args())
        assert front[-1] <= 40

    def test_growth_rate_cap_limits_jumps(self):
        frame_count, columns = 60, 80
        truth = np.zeros(frame_count, dtype=int)
        truth[30:] = 79  # implausible instantaneous jump
        kymograph = synthetic_kymograph(frame_count, columns, truth)
        front = monotone_front(kymograph, front_args())
        max_step = int(round(7.5 / 1.5))
        assert np.max(np.diff(front)) <= max_step


class PathLockedTemporalFrontTests:
    @staticmethod
    def straight_path(point_count=36):
        path = np.column_stack(
            [np.arange(10, 10 + point_count), np.full(point_count, 20)]
        ).astype(np.float32)
        arclength = np.arange(point_count, dtype=np.float32)
        return path, arclength

    @staticmethod
    def evidence_from_births(path, births, frame_count=80):
        evidence = np.zeros((frame_count, 40, 64), dtype=np.uint8)
        for point, (x, y) in enumerate(path.astype(int)):
            evidence[int(births[point]) :, y, x] = 12
        return evidence

    def fit(self, evidence, path, arclength, observed, **overrides):
        values = {
            "warmup_samples": 8,
            "max_step_px": 2.25,
            "max_confirmation_lag_samples": 12,
            "absolute_evidence_floor": 4.0,
            "normal_halfwidth_px": 0.0,
            "normal_sample_count": 1,
            "minimum_change": 2.0,
        }
        values.update(overrides)
        return path_locked_temporal_front(
            evidence,
            path,
            arclength,
            observed,
            **values,
        )

    def test_recovers_gradual_growth_without_changing_path(self):
        path, arclength = self.straight_path()
        truth = 12 + np.arange(len(path)) // 2
        observed = truth + 2
        evidence = self.evidence_from_births(path, observed)

        result = self.fit(evidence, path, arclength, observed)

        assert result.feasible
        assert np.all(np.diff(result.birth_samples) >= 0)
        assert np.all(result.birth_samples <= observed)
        lengths = np.where(
            result.front_indices >= 0,
            arclength[np.clip(result.front_indices, 0, len(arclength) - 1)],
            0.0,
        )
        assert np.max(np.diff(lengths)) <= 2.25
        assert result.eventual_support_fraction == 1.0
        assert result.direct_support_mask.shape == (len(path),)
        assert result.eventual_support_mask.shape == (len(path),)
        assert np.array_equal(
            result.direct_support_mask,
            np.ones(len(path), dtype=bool),
        )
        assert np.array_equal(
            result.eventual_support_mask,
            np.ones(len(path), dtype=bool),
        )

    def test_uses_speed_prior_to_resolve_blocked_optical_confirmations(self):
        path, arclength = self.straight_path(30)
        truth = 12 + np.arange(len(path))
        observed = ((truth + 5 + 5) // 6) * 6
        evidence = self.evidence_from_births(path, observed)

        result = self.fit(
            evidence,
            path,
            arclength,
            observed,
            max_step_px=2.0,
            max_confirmation_lag_samples=12,
        )

        assert result.feasible
        assert np.max(observed - result.birth_samples) <= 12
        assert np.max(np.diff(arclength[np.maximum(result.front_indices, 0)])) <= 2.0
        assert np.mean(np.abs(result.birth_samples - truth)) < np.mean(
            np.abs(observed - truth)
        )

    def test_early_foreign_crossing_cannot_pull_front_past_confirmation_bound(self):
        path, arclength = self.straight_path(40)
        observed = 12 + np.arange(len(path))
        evidence = self.evidence_from_births(path, observed)
        distal = path[32:38].astype(int)
        evidence[8:35, distal[:, 1], distal[:, 0]] = 20

        result = self.fit(
            evidence,
            path,
            arclength,
            observed,
            max_confirmation_lag_samples=6,
        )

        assert result.feasible
        assert result.front_indices[30] < 32
        assert np.all(result.birth_samples >= observed - 6)

    def test_unconfirmed_path_is_not_smoothed_into_a_trajectory(self):
        path, arclength = self.straight_path(20)
        evidence = np.zeros((40, 40, 64), dtype=np.uint8)
        observed = np.full(len(path), len(evidence), dtype=np.int32)

        result = self.fit(evidence, path, arclength, observed)

        assert not result.feasible
        assert result.reason == "unobserved-path-point"

    def test_path_offsets_follow_a_translating_proximal_tube(self):
        path, arclength = self.straight_path(12)
        frame_count = 48
        observed = 12 + np.arange(len(path), dtype=np.int32)
        offsets = np.zeros((frame_count, 2), dtype=np.float32)
        offsets[10:, 1] = 4.0
        evidence = np.zeros((frame_count, 40, 64), dtype=np.uint8)
        for sample in range(frame_count):
            born = np.flatnonzero(observed <= sample)
            if not len(born):
                continue
            points = np.rint(path[born] + offsets[sample]).astype(int)
            evidence[sample, points[:, 1], points[:, 0]] = 12

        fixed = self.fit(evidence, path, arclength, observed)
        moving = self.fit(
            evidence,
            path,
            arclength,
            observed,
            path_offsets_xy=offsets,
        )

        assert moving.feasible
        assert moving.direct_support_fraction == 1.0
        assert moving.eventual_support_fraction == 1.0
        assert moving.direct_support_mask.all()
        assert moving.eventual_support_mask.all()
        assert fixed.direct_support_fraction == 0.0
        assert fixed.eventual_support_fraction == 0.0
        assert not fixed.direct_support_mask.any()
        assert not fixed.eventual_support_mask.any()

    def test_path_offsets_follow_a_nonrigid_material_curve(self):
        path, arclength = self.straight_path(18)
        frame_count = 56
        observed = 12 + np.arange(len(path), dtype=np.int32)
        offsets = np.zeros((frame_count, len(path), 2), dtype=np.float32)
        offsets[10:, :, 1] = np.linspace(0.0, 5.0, len(path))
        evidence = np.zeros((frame_count, 48, 64), dtype=np.uint8)
        for sample in range(frame_count):
            born = np.flatnonzero(observed <= sample)
            if not len(born):
                continue
            points = np.rint(path[born] + offsets[sample, born]).astype(int)
            evidence[sample, points[:, 1], points[:, 0]] = 12

        fixed = self.fit(evidence, path, arclength, observed)
        moving = self.fit(
            evidence,
            path,
            arclength,
            observed,
            path_offsets_xy=offsets,
        )

        assert moving.feasible
        assert moving.direct_support_fraction == 1.0
        assert moving.eventual_support_fraction == 1.0
        assert fixed.direct_support_fraction < moving.direct_support_fraction

    def test_rejects_path_offsets_with_wrong_time_axis(self):
        path, arclength = self.straight_path(12)
        observed = 12 + np.arange(len(path), dtype=np.int32)
        evidence = self.evidence_from_births(path, observed, frame_count=48)

        result = self.fit(
            evidence,
            path,
            arclength,
            observed,
            path_offsets_xy=np.zeros((len(evidence) - 1, 2), dtype=np.float32),
        )

        assert not result.feasible
        assert result.reason == "incompatible-path-offsets"
        assert result.direct_support_mask.shape == (len(path),)
        assert result.eventual_support_mask.shape == (len(path),)
        assert not result.direct_support_mask.any()
        assert not result.eventual_support_mask.any()


class VisiblePathFrontTests:
    def test_recovers_connected_visible_tip_without_backward_inference(self):
        sample_count = 56
        point_count = 32
        truth = np.clip(np.arange(sample_count) - 12, -1, point_count - 1)
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for sample, tip in enumerate(truth):
            if tip >= 0:
                profiles[sample, : tip + 1] = 1.0

        result = visible_path_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=1.0,
            coverage_cost=0.05,
            absence_weight=0.25,
            motion_penalty=0.1,
            support_window_samples=3,
        )

        assert result.feasible
        assert np.max(np.abs(result.front_indices[12:] - truth[12:])) <= 1
        assert result.birth_samples[0] >= 12

    def test_foreign_crossing_beyond_a_gap_cannot_become_the_tip(self):
        sample_count = 64
        point_count = 40
        truth = np.clip((np.arange(sample_count) - 10) // 2, -1, point_count - 1)
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for sample, tip in enumerate(truth):
            if tip >= 0:
                profiles[sample, : tip + 1] = 1.0
        profiles[20:38, 30:36] = 1.0

        result = visible_path_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=1.0,
            coverage_cost=0.08,
            absence_weight=0.4,
            motion_penalty=0.1,
            support_window_samples=3,
        )

        assert result.feasible
        assert np.max(result.front_indices[20:38] - truth[20:38]) <= 2

    def test_can_end_at_supported_prefix_before_contaminated_mature_tail(self):
        sample_count = 64
        point_count = 44
        supported_count = 24
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for sample in range(12, sample_count):
            tip = min((sample - 12) // 2, supported_count - 1)
            profiles[sample, : tip + 1] = 1.0
        profiles[20:, 34:42] = 1.0

        result = visible_path_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=1.0,
            coverage_cost=0.08,
            absence_weight=0.5,
            motion_penalty=0.1,
            support_window_samples=3,
            require_final_endpoint=False,
        )

        assert result.feasible
        assert result.reason == "partial-prefix"
        assert supported_count - 2 <= result.front_indices[-1] < 30
        assert np.all(result.birth_samples[result.front_indices[-1] + 1 :] == sample_count)


class CausalChangepointFrontTests:
    @staticmethod
    def hypothesis(
        label,
        *,
        final_length,
        total_length,
        direct,
        eventual,
        objective,
        reached_points,
        root_distance=6.0,
        radial_excursion=40.0,
        terminates_at_foreign_owner=False,
    ):
        result = CausalGrowthFrontResult(
            birth_samples=np.zeros(reached_points, dtype=np.int32),
            front_indices=np.array([reached_points - 1], dtype=np.int32),
            direct_support_mask=np.ones(reached_points, dtype=bool),
            eventual_support_mask=np.ones(reached_points, dtype=bool),
            feasible=True,
            reason="ok",
            objective_score=objective,
            final_point_index=reached_points - 1,
            final_length_px=final_length,
            direct_support_fraction=direct,
            eventual_support_fraction=eventual,
        )
        return CausalPathHypothesis(
            label=label,
            result=result,
            total_length_px=total_length,
            root_distance_px=root_distance,
            radial_excursion_px=radial_excursion,
            terminates_at_foreign_owner=terminates_at_foreign_owner,
        )

    def test_decisive_path_selection_rescues_failed_short_baseline(self):
        baseline = self.hypothesis(
            "field",
            final_length=1.0,
            total_length=20.0,
            direct=0.0,
            eventual=1.0,
            objective=0.5,
            reached_points=2,
        )
        recovered = self.hypothesis(
            "atlas",
            final_length=62.0,
            total_length=64.0,
            direct=0.84,
            eventual=0.94,
            objective=18.0,
            reached_points=63,
        )

        selected = select_decisive_causal_path(
            [baseline, recovered], owner_radius_px=5.5
        )

        assert selected.index == 1
        assert selected.reason == "decisive-owner-rooted-causal-rescue"

    def test_growth_direction_rejects_stronger_supported_inward_sequence(self):
        outward = self.hypothesis(
            "outward",
            final_length=70.0,
            total_length=80.0,
            direct=0.80,
            eventual=1.0,
            objective=10.0,
            reached_points=70,
        ).result
        inward = self.hypothesis(
            "inward",
            final_length=80.0,
            total_length=80.0,
            direct=0.95,
            eventual=1.0,
            objective=16.0,
            reached_points=80,
        ).result

        certificate = certify_growth_direction(
            outward,
            inward,
            total_length_px=80.0,
        )

        assert certificate.accepted is False
        assert certificate.reason == "distal-inward-growth-dominant"

    def test_growth_direction_keeps_owner_when_reverse_margin_is_small(self):
        outward = self.hypothesis(
            "outward",
            final_length=80.0,
            total_length=80.0,
            direct=0.90,
            eventual=1.0,
            objective=14.0,
            reached_points=80,
        ).result
        inward = self.hypothesis(
            "inward",
            final_length=80.0,
            total_length=80.0,
            direct=0.95,
            eventual=1.0,
            objective=15.0,
            reached_points=80,
        ).result

        certificate = certify_growth_direction(
            outward,
            inward,
            total_length_px=80.0,
        )

        assert certificate.accepted is True
        assert certificate.reason == "owner-outward-direction-not-contradicted"

    def test_growth_direction_ignores_poorly_supported_reverse_fit(self):
        outward = self.hypothesis(
            "outward",
            final_length=70.0,
            total_length=80.0,
            direct=0.90,
            eventual=1.0,
            objective=8.0,
            reached_points=70,
        ).result
        inward = self.hypothesis(
            "inward",
            final_length=80.0,
            total_length=80.0,
            direct=0.60,
            eventual=1.0,
            objective=18.0,
            reached_points=80,
        ).result

        certificate = certify_growth_direction(
            outward,
            inward,
            total_length_px=80.0,
        )

        assert certificate.accepted is True


    def test_decisive_path_selection_preserves_healthy_baseline(self):
        baseline = self.hypothesis(
            "field",
            final_length=35.0,
            total_length=36.0,
            direct=0.9,
            eventual=1.0,
            objective=10.0,
            reached_points=36,
        )
        longer = self.hypothesis(
            "atlas",
            final_length=80.0,
            total_length=80.0,
            direct=0.95,
            eventual=1.0,
            objective=24.0,
            reached_points=80,
        )

        selected = select_decisive_causal_path(
            [baseline, longer], owner_radius_px=5.5
        )

        assert selected.index == 0
        assert selected.reason == "baseline-causally-supported"

    def test_decisive_path_selection_can_rescue_owner_safe_contact_prefix(self):
        baseline = self.hypothesis(
            "field",
            final_length=1.0,
            total_length=9.0,
            direct=0.0,
            eventual=1.0,
            objective=0.6,
            reached_points=2,
            radial_excursion=4.0,
        )
        contact_prefix = self.hypothesis(
            "atlas-contact",
            final_length=10.0,
            total_length=10.0,
            direct=0.90,
            eventual=1.0,
            objective=4.2,
            reached_points=11,
            radial_excursion=7.0,
            terminates_at_foreign_owner=True,
        )

        selected = select_decisive_causal_path(
            [baseline, contact_prefix], owner_radius_px=5.5
        )

        assert selected.index == 1
        assert selected.reason == "decisive-owner-rooted-causal-rescue"

    def test_decisive_path_selection_rejects_rim_and_wrong_owner_paths(self):
        baseline = self.hypothesis(
            "field",
            final_length=1.0,
            total_length=20.0,
            direct=0.0,
            eventual=1.0,
            objective=0.5,
            reached_points=2,
        )
        rim = self.hypothesis(
            "rim",
            final_length=70.0,
            total_length=70.0,
            direct=0.95,
            eventual=1.0,
            objective=20.0,
            reached_points=70,
            radial_excursion=4.0,
        )
        wrong_owner = self.hypothesis(
            "wrong-owner",
            final_length=80.0,
            total_length=80.0,
            direct=0.95,
            eventual=1.0,
            objective=24.0,
            reached_points=80,
            root_distance=18.0,
        )

        selected = select_decisive_causal_path(
            [baseline, rim, wrong_owner], owner_radius_px=5.5
        )

        assert selected.index == 0
        assert selected.reason == "no-decisive-causal-rescue"

    def test_growth_certificate_rejects_pollen_rim_motion(self):
        rim = self.hypothesis(
            "rim",
            final_length=18.0,
            total_length=18.0,
            direct=1.0,
            eventual=1.0,
            objective=8.0,
            reached_points=18,
            radial_excursion=3.0,
        )

        certificate = certify_rooted_growth(rim, owner_radius_px=5.5)

        assert not certificate.accepted
        assert certificate.reason == "insufficient-owner-departure"

    def test_growth_certificate_requires_direct_and_eventual_support(self):
        weak_direct = self.hypothesis(
            "weak-direct",
            final_length=35.0,
            total_length=35.0,
            direct=0.70,
            eventual=1.0,
            objective=8.0,
            reached_points=35,
        )
        weak_eventual = self.hypothesis(
            "weak-eventual",
            final_length=35.0,
            total_length=35.0,
            direct=0.90,
            eventual=0.80,
            objective=8.0,
            reached_points=35,
        )

        direct = certify_rooted_growth(weak_direct, owner_radius_px=5.5)
        eventual = certify_rooted_growth(weak_eventual, owner_radius_px=5.5)

        assert direct.reason == "insufficient-direct-birth-support"
        assert eventual.reason == "insufficient-eventual-birth-support"

    def test_growth_certificate_allows_supported_boundary_path(self):
        boundary = self.hypothesis(
            "boundary",
            final_length=30.0,
            total_length=60.0,
            direct=0.65,
            eventual=1.0,
            objective=7.0,
            reached_points=30,
            radial_excursion=20.0,
        )

        regular = certify_rooted_growth(boundary, owner_radius_px=5.5)
        censored = certify_rooted_growth(
            boundary,
            owner_radius_px=5.5,
            boundary_censored=True,
        )

        assert not regular.accepted
        assert censored.accepted

    def test_inextensible_projection_preserves_material_spacing(self):
        material_arc = np.arange(18, dtype=np.float64)
        paths = np.zeros((3, len(material_arc), 2), dtype=np.float64)
        paths[:, :, 0] = material_arc
        paths[1, :, 1] = 3.0 * np.sin(material_arc / 3.0)
        paths[2, :, 1] = np.linspace(0.0, 5.0, len(material_arc))

        projected = project_inextensible_paths(paths, material_arc)

        segment_lengths = np.linalg.norm(np.diff(projected, axis=1), axis=2)
        assert np.allclose(segment_lengths, 1.0)
        assert np.allclose(projected[:, 0], paths[:, 0])

    def test_scalar_deformation_follows_smooth_sideways_motion(self):
        sample_count = 18
        point_count = 14
        base = np.zeros((sample_count, point_count, 2), dtype=np.float32)
        base[:, :, 0] = np.arange(point_count)[None, :] + 8.0
        base[:, :, 1] = 20.0
        truth = np.rint(2.0 * np.sin(np.linspace(0.0, np.pi, sample_count)))
        guide = np.zeros((sample_count, 40, 40), dtype=np.float32)
        for sample, offset in enumerate(truth):
            y = int(20 + offset)
            guide[sample, y, 8 : 8 + point_count] = 1.0

        result = fit_scalar_path_deformation(
            guide,
            base,
            normal_radius_px=3.0,
            offset_magnitude_penalty=0.02,
            root_lock_points=1,
        )

        assert result.final_energy < result.initial_energy
        assert np.all(result.offsets_px[:, 0] == 0.0)
        assert np.median(np.abs(result.offsets_px[:, 3:] - truth[:, None])) <= 1.0

    def test_dynamic_profiles_follow_a_translating_path(self):
        sample_count = 36
        point_count = 12
        paths = np.zeros((sample_count, point_count, 2), dtype=np.float32)
        paths[:, :, 0] = np.arange(point_count)[None, :] + 10.0
        paths[:, :, 1] = np.arange(sample_count)[:, None] * 0.25 + 14.0
        evidence = np.zeros((sample_count, 40, 48), dtype=np.uint8)
        for sample in range(10, sample_count):
            count = min(point_count, sample - 9)
            points = np.rint(paths[sample, :count]).astype(int)
            evidence[sample, points[:, 1], points[:, 0]] = 20

        profiles = dynamic_path_novelty_profiles(
            evidence,
            paths,
            warmup_samples=6,
            absolute_evidence_floor=0.0,
            normal_halfwidth_px=0.0,
            normal_sample_count=1,
            minimum_change=2.0,
            noise_multiplier=2.0,
            temporal_median_samples=1,
        )

        assert np.all(profiles[20:, 0] > 0.9)
        assert np.all(profiles[:8] < 0.0)

    def test_recovers_full_ordered_growth_without_large_tip_steps(self):
        sample_count = 64
        point_count = 28
        births = 12 + np.arange(point_count) // 2
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for point, birth in enumerate(births):
            profiles[birth:, point] = 1.0

        result = causal_changepoint_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=2.0,
        )

        assert result.feasible
        assert result.reason == "ok"
        assert result.final_point_index == point_count - 1
        assert np.max(np.diff(np.maximum(result.front_indices, -1))) <= 2
        assert np.mean(np.abs(result.birth_samples - births)) <= 1.0

    def test_stops_before_early_foreign_tail_beyond_unsupported_gap(self):
        sample_count = 72
        point_count = 42
        supported_count = 24
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for point in range(supported_count):
            profiles[14 + point :, point] = 1.0
        profiles[18:, 33:40] = 1.0

        result = causal_changepoint_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=2.0,
        )

        assert result.feasible
        assert result.reason == "partial-causal-prefix"
        assert supported_count - 2 <= result.final_point_index < 30
        assert np.all(result.birth_samples[result.final_point_index + 1 :] == sample_count)

    def test_rejects_path_with_no_appearance_event(self):
        profiles = np.full((48, 20), -1.0, dtype=np.float32)

        result = causal_changepoint_front(
            profiles,
            np.arange(20, dtype=np.float64),
            warmup_samples=8,
            max_step_px=2.0,
        )

        assert not result.feasible
        assert result.reason == "no-positive-causal-prefix"

    def test_can_retain_growth_seen_at_the_recording_boundary(self):
        sample_count = 48
        point_count = 16
        births = 32 + np.arange(point_count)
        profiles = np.full((sample_count, point_count), -1.0, dtype=np.float32)
        for point, birth in enumerate(births):
            profiles[birth:, point] = 1.0

        result = causal_changepoint_front(
            profiles,
            np.arange(point_count, dtype=np.float64),
            warmup_samples=8,
            max_step_px=2.0,
            minimum_post_samples=1,
        )

        assert result.feasible
        assert result.reason == "ok"
        assert result.birth_samples[-1] == sample_count - 1


class ConfirmationBoundedFrontTests:
    def test_recovers_growth_before_sparse_whole_path_confirmations(self):
        frame_count, point_count = 72, 36
        arclength = np.arange(point_count, dtype=np.float64)
        truth_births = np.maximum(
            12 + np.arange(point_count),
            np.where(np.arange(point_count) < 18, 8, 40),
        )
        profiles = np.full((frame_count, point_count), -0.8, dtype=np.float32)
        for point, birth in enumerate(truth_births):
            profiles[birth:, point] = 0.9
        confirmations = np.where(
            np.arange(point_count) < 18,
            40,
            64,
        )
        earliest = np.where(
            np.arange(point_count) < 18,
            8,
            40,
        )

        result = confirmation_bounded_path_front(
            profiles,
            arclength,
            confirmations,
            earliest,
            warmup_samples=8,
            max_step_px=2.0,
        )

        assert result.feasible
        assert result.birth_samples[3] < confirmations[3] - 15
        assert np.mean(np.abs(result.birth_samples - truth_births)) < 2.0
        assert result.direct_support_fraction == 1.0

    def test_detached_distal_evidence_cannot_escape_its_confirmation_window(self):
        frame_count, point_count = 70, 40
        arclength = np.arange(point_count, dtype=np.float64)
        profiles = np.full((frame_count, point_count), -0.8, dtype=np.float32)
        for point in range(24):
            profiles[12 + point :, point] = 0.9
        profiles[10:45, 32:38] = 1.0
        confirmations = np.r_[np.full(24, 42), np.full(16, 64)]
        earliest = np.r_[np.full(24, 8), np.full(16, 42)]

        result = confirmation_bounded_path_front(
            profiles,
            arclength,
            confirmations,
            earliest,
            warmup_samples=8,
            max_step_px=2.0,
        )

        assert result.feasible
        assert result.front_indices[35] < 24
        assert np.all(result.birth_samples[24:] >= 42)

    def test_eventual_support_can_mature_just_after_sparse_confirmation(self):
        frame_count, point_count = 48, 12
        arclength = np.arange(point_count, dtype=np.float64)
        profiles = np.full((frame_count, point_count), -0.8, dtype=np.float32)
        confirmations = np.full(point_count, 30, dtype=np.int32)
        earliest = np.full(point_count, 8, dtype=np.int32)
        profiles[32:, :] = 0.9

        immediate = confirmation_bounded_path_front(
            profiles,
            arclength,
            confirmations,
            earliest,
            warmup_samples=8,
            max_step_px=2.0,
        )
        delayed = confirmation_bounded_path_front(
            profiles,
            arclength,
            confirmations,
            earliest,
            warmup_samples=8,
            max_step_px=2.0,
            support_window_samples=3,
        )

        assert immediate.feasible and delayed.feasible
        assert immediate.eventual_support_fraction == 0.0
        assert delayed.eventual_support_fraction == 1.0


class TraceExtensionCorridorTests:
    def test_follows_bright_ridge(self):
        image = np.zeros((200, 200), np.float32)
        ridge_rows = np.arange(60, 160)
        image[ridge_rows, 100] = 10.0
        image[ridge_rows, 99] = 6.0
        image[ridge_rows, 101] = 6.0
        path = trace_extension_corridor(
            image,
            tip_xy=np.array([100.0, 60.0]),
            tangent_xy=np.array([0.0, 1.0]),
            steps=30,
        )
        assert len(path) == 31
        assert np.abs(path[:, 0] - 100.0).max() < 4.0
        assert path[-1, 1] > path[0, 1] + 40.0

    def test_stops_at_image_border(self):
        image = np.zeros((50, 50), np.float32)
        path = trace_extension_corridor(
            image,
            tip_xy=np.array([25.0, 45.0]),
            tangent_xy=np.array([0.0, 1.0]),
            steps=30,
        )
        assert np.all(path[:, 1] < 50)


class LateralRefitTests:
    def test_recovers_shifted_ridge(self):
        from tubetracker.growth_front import lateral_refit

        image = np.zeros((120, 120), np.float32)
        image[:, 60] = 10.0  # vertical ridge at x=60
        image[:, 59] = 6.0
        image[:, 61] = 6.0
        path = np.stack([np.full(40, 63.0), np.linspace(10, 110, 40)], axis=1)
        refit = lateral_refit(path, image)
        assert abs(refit[0, 0] - 63.0) < 1e-6  # root pinned
        assert np.abs(refit[5:, 0] - 60.0).mean() < 1.2

    def test_short_path_returned_unchanged(self):
        from tubetracker.growth_front import lateral_refit

        image = np.zeros((50, 50), np.float32)
        path = np.array([[10.0, 10.0], [12.0, 10.0]])
        assert np.allclose(lateral_refit(path, image), path)


class ValidatedTransformTests:
    def make_tracks(self, drift):
        rng = np.random.default_rng(1)
        base = rng.uniform(40.0, 60.0, (13, 2))
        tracks = np.stack([base, base + drift])
        visibility = np.ones((2, 13), bool)
        return tracks.astype(np.float32), visibility

    def test_good_similarity_is_kept(self):
        from tubetracker.growth_front import validated_transform

        tracks, visibility = self.make_tracks(drift=np.array([5.0, -3.0]))
        centers = np.array([[50.0, 50.0], [55.0, 47.0]])
        matrix = validated_transform(tracks, visibility, centers, 0, 1)
        mapped = matrix @ np.array([50.0, 50.0, 1.0])
        assert np.linalg.norm(mapped - [55.0, 47.0]) < 0.5

    def test_drifting_landmarks_fall_back_to_translation(self):
        from tubetracker.growth_front import validated_transform

        # landmarks slid off the grain together (the observed failure) while
        # the measured grain center stayed put
        tracks, visibility = self.make_tracks(drift=np.array([25.0, 18.0]))
        centers = np.array([[50.0, 50.0], [50.0, 50.0]])
        matrix = validated_transform(tracks, visibility, centers, 0, 1)
        assert np.allclose(matrix[:, :2], np.eye(2))  # pure translation
        mapped = matrix @ np.array([50.0, 50.0, 1.0])
        assert np.linalg.norm(mapped - [50.0, 50.0]) < 1e-6


class PathCoordinateTipTests:
    def test_isotonic_increasing_enforces_monotonicity(self):
        from tubetracker.growth_front import isotonic_increasing

        values = np.array([0.0, 5.0, 3.0, 4.0, 10.0, 9.0, 12.0])
        weights = np.ones(len(values))
        fitted = isotonic_increasing(values, weights)
        assert np.all(np.diff(fitted) >= -1e-9)
        # pooled regions average their members
        assert abs(fitted[1] - 4.0) < 1e-9 and abs(fitted[2] - 4.0) < 1e-9

    def test_isotonic_respects_weights(self):
        from tubetracker.growth_front import isotonic_increasing

        values = np.array([0.0, 10.0, 2.0])
        heavy_last = isotonic_increasing(values, np.array([1.0, 0.01, 100.0]))
        assert heavy_last[1] < 3.0  # low-weight outlier pulled to trusted level

    def test_project_to_polyline_recovers_arclength_and_offset(self):
        from tubetracker.growth_front import project_to_polyline

        line = np.stack([np.arange(0.0, 30.0, 1.5), np.zeros(20)], axis=1)
        s, n = project_to_polyline(line, np.array([9.75, 2.0]), step=1.5)
        assert abs(s - 9.75) < 0.1
        assert abs(n - 2.0) < 0.1
        s2, n2 = project_to_polyline(line, np.array([6.0, -1.0]), step=1.5)
        assert abs(n2 + 1.0) < 0.1

    def test_subpixel_tip_tracks_synthetic_edge(self):
        from tubetracker.growth_front import subpixel_tip_arclength

        frames, columns = 60, 80
        kymograph = np.zeros((frames, columns), np.float32)
        true_edge = 20.0 + 0.5 * np.arange(frames)  # half a column per frame
        for t in range(frames):
            kymograph[t, : int(true_edge[t])] = 8.0
        front = np.clip(true_edge.astype(int), 0, columns - 1)
        tips, contrast = subpixel_tip_arclength(kymograph, front, arc_step=1.5)
        # settled region: recovered edge should follow truth closely
        error = np.abs(tips[10:50] / 1.5 - true_edge[10:50])
        assert error.mean() < 2.0
        assert contrast[10:50].min() > 1.0


class FieldOwnershipCertificateTests:
    @staticmethod
    def owner(track_id, path, *, batch_accepted, support):
        path = np.asarray(path, dtype=np.float64)
        return SimpleNamespace(
            track_id=track_id,
            track={"source_frames": [0], "centers_yx": [[0.0, 0.0]]},
            world_paths_yx=path[None],
            batch_accepted=batch_accepted,
            front=SimpleNamespace(
                feasible=True,
                front_indices=np.asarray([len(path) - 1]),
                direct_support_fraction=support,
                eventual_support_fraction=1.0,
            ),
        )

    def test_duplicate_path_keeps_prior_supported_owner(self):
        path = np.column_stack((np.zeros(30), np.arange(30.0)))
        trusted = self.owner(1, path, batch_accepted=True, support=0.70)
        rival = self.owner(
            2,
            path + np.asarray([1.0, 0.0]),
            batch_accepted=False,
            support=0.95,
        )
        losers, claims = temporal_duplicate_claims([trusted, rival])
        assert losers == {2}
        assert claims[0]["component_winner"] == 1

    def test_distal_path_entering_another_pollen_is_flagged(self):
        crossing = self.owner(
            1,
            [[0.0, 15.0], [0.0, 25.0], [0.0, 35.0]],
            batch_accepted=True,
            support=0.9,
        )
        neighbor = self.owner(
            2,
            [[0.0, 35.0], [0.0, 40.0]],
            batch_accepted=True,
            support=0.9,
        )
        neighbor.track["centers_yx"] = [[0.0, 35.0]]
        contacted, contacts = temporal_foreign_pollen_contacts(
            [crossing, neighbor],
            0,
            owner_radius_px=10.0,
        )
        assert 1 in contacted
        assert any(contact["foreign_owner_track_id"] == 2 for contact in contacts)

    def test_independent_field_duplicate_component_keeps_one_owner(self):
        path = np.column_stack((np.zeros(20), np.arange(20.0)))
        weak = self.owner(1, path, batch_accepted=True, support=0.5)
        strong = self.owner(2, path, batch_accepted=True, support=0.9)
        rival = self.owner(3, path, batch_accepted=False, support=1.0)
        losers, resolutions = consistency_duplicate_losers(
            [weak, strong, rival],
            [
                {"first_owner": 1, "second_owner": 2},
                {"first_owner": 2, "second_owner": 3},
            ],
        )
        assert losers == {1, 3}
        assert resolutions[0]["winner_owner_id"] == 2


class DirectEmergenceGateTests:
    def test_consensus_uses_agreeing_path_cues_then_durable_rim(self):
        sample, source = select_emergence_consensus(
            40, 44, 20, path_agreement_samples=6
        )
        assert (sample, source) == (42, "path-state-consensus")
        sample, source = select_emergence_consensus(
            20,
            60,
            35,
            path_agreement_samples=6,
            timeline_samples=65,
        )
        assert (sample, source) == (35, "durable-rim-late-template-fallback")

    def test_durable_rim_rejects_short_episode(self):
        extension = np.zeros(40)
        extension[8:12] = [2.0, 3.0, 5.0, 8.0]
        extension[20:36] = np.linspace(2.0, 12.0, 16)
        onset = durable_rim_emergence_sample(
            extension,
            warmup_samples=6,
            minimum_emergence_px=2.0,
            window_samples=12,
            required_fraction=0.75,
            minimum_growth_px=5.0,
        )
        assert onset == 20

    def test_directional_rim_emergence_rejects_ring_change_and_keeps_growth(self):
        history = np.zeros((20, 72, 12), dtype=np.float32)
        history[7, :, :4] = 8.0  # non-specific pollen deformation
        history[10:13, 0, :3] = 8.0
        history[12:, 0, :8] = 8.0
        radii = np.arange(16.0, 28.0)
        extension, direction, _ = rim_extension_history(
            history,
            radii,
            np.zeros(20),
            owner_radius_px=15.0,
            warmup_samples=6,
            direction_tolerance_degrees=45.0,
            minimum_directional_prominence_px=1.0,
        )
        onset, confirmation = persistent_directional_emergence(
            extension,
            direction,
            warmup_samples=6,
            minimum_emergence_px=2.0,
            persistence_window=5,
            persistence_required=3,
            followup_window=8,
            minimum_followup_growth_px=3.0,
        )
        assert extension[7] == 0.0
        assert onset == 10
        assert confirmation == 12

    def test_dense_center_match_recovers_local_translation(self):
        y, x = np.mgrid[:80, :80]
        first = np.exp(-((y - 40.0) ** 2 + (x - 40.0) ** 2) / 35.0)
        first += 0.4 * np.exp(
            -((y - 37.0) ** 2 + (x - 43.0) ** 2) / 4.0
        )
        moved = np.exp(-((y - 43.0) ** 2 + (x - 36.0) ** 2) / 35.0)
        moved += 0.4 * np.exp(
            -((y - 40.0) ** 2 + (x - 39.0) ** 2) / 4.0
        )
        template = image_patch(first.astype(np.float32), np.array([40.0, 40.0]), 9)
        center, score = match_owner_center(
            moved.astype(np.float32),
            np.array([40.0, 40.0]),
            template,
            search_radius_px=6,
            minimum_score=0.2,
        )
        assert np.allclose(center, [43.0, 36.0], atol=0.5)
        assert score > 0.9

    def test_cross_section_change_appears_only_after_new_tube(self):
        cross = np.full((18, 6, 17), 120.0, dtype=np.float32)
        cross[8:, 1:5, 5:7] = 85.0
        cross[8:, 1:5, 10:12] = 85.0
        profile = cross_section_novelty(cross, warmup_samples=6)
        assert np.max(profile[:6]) < 0.0
        assert np.median(profile[10:, 1:5]) > 0.0

    def test_tube_state_prefers_confirmed_shape_over_unrelated_change(self):
        cross = np.full((18, 5, 17), 120.0, dtype=np.float32)
        cross[7, :, 8] = 70.0  # transient change unlike the paired tube walls
        cross[10:, :, 5:7] = 80.0
        cross[10:, :, 10:12] = 80.0
        profile = tube_state_profile(
            cross,
            warmup_samples=6,
            positive_samples=np.full(5, 12),
        )
        assert np.max(profile[:6]) < 0.0
        assert np.median(profile[10:]) > 0.0
        assert np.median(profile[7]) < 0.0

    def test_emergence_requires_persistence_and_later_growth(self):
        profile = np.full((20, 8), -1.0, dtype=np.float32)
        profile[7, 1:3] = 1.0  # one-frame artifact
        profile[10:13, 1:3] = 1.0
        profile[12:16, 1:5] = 1.0
        direct = connected_direct_front_indices(profile)
        onset, confirmation = persistent_emergence_sample(
            direct,
            np.arange(8, dtype=np.float64) * 1.5,
            warmup_samples=6,
            minimum_length_px=1.5,
            persistence_window=5,
            persistence_required=3,
            followup_window=8,
            minimum_followup_growth_px=3.0,
        )
        assert onset == 10
        assert confirmation == 12

    def test_gate_removes_inferred_history_and_limits_first_growth(self):
        inferred = np.asarray([1, 2, 3, 4, 5, 6, 7, 7], dtype=np.int32)
        direct = np.asarray([-1, -1, -1, -1, 1, 2, 4, 6], dtype=np.int32)
        fitted = apply_emergence_gate(
            inferred,
            direct,
            np.arange(8, dtype=np.float64) * 1.5,
            emergence_sample=4,
            maximum_growth_px_per_sample=3.0,
        )
        assert np.all(fitted[:4] == -1)
        assert fitted[4] == 1
        assert np.all(np.diff(fitted[4:]) <= 2)

    def test_gate_caps_first_visible_prefix_when_evidence_arrives_late(self):
        inferred = np.asarray([-1, -1, 8, 9, 10], dtype=np.int32)
        direct = np.asarray([-1, -1, 8, 9, 10], dtype=np.int32)
        fitted = apply_emergence_gate(
            inferred,
            direct,
            np.arange(12, dtype=np.float64) * 1.5,
            emergence_sample=0,
            maximum_growth_px_per_sample=6.0,
        )
        assert fitted[2] == 4
        assert np.max(np.diff(fitted[2:]) * 1.5) <= 6.0


def test_edge_seed_census_identity_is_withheld_from_automatic_tracing():
    """A census seed born inside the boundary margin stays diagnostic."""

    from prototypes.v29_causal_growth_front.bootstrap import _write_owner_field_inputs
    import tempfile

    source_frames = np.arange(4)
    fps = 14.0
    # owner 7 starts at the top edge (clipped seed), owner 8 starts interior
    centers = np.asarray(
        [
            [[5.0, 500.0], [200.0, 500.0], [300.0, 500.0], [500.0, 500.0]],
            [[500.0, 500.0], [500.0, 500.0], [500.0, 500.0], [500.0, 500.0]],
        ]
    )
    visible = np.ones((2, 4), dtype=bool)
    from prototypes.v29_causal_growth_front.bootstrap import (
        OwnerBodyCertificates,
        OwnerIdentityCertificates,
    )
    body = OwnerBodyCertificates(
        statuses=np.asarray(["verified", "verified"]),
        verified=np.asarray([True, True]),
        rejected=np.asarray([False, False]),
        median_radial_contrast=np.zeros(2),
        median_angular_boundary_fraction=np.zeros(2),
        median_opposite_boundary_fraction=np.zeros(2),
        median_valid_angular_fraction=np.zeros(2),
        valid_sample_counts=np.zeros(2, dtype=np.int32),
        sample_indices=np.zeros((2, 1), dtype=np.int32),
        radial_contrast_samples=np.zeros((2, 1)),
        angular_boundary_samples=np.zeros((2, 1)),
        opposite_boundary_samples=np.zeros((2, 1)),
        valid_angular_samples=np.zeros((2, 1)),
    )
    identity = OwnerIdentityCertificates(
        statuses=np.asarray(["verified", "verified"]),
        verified=np.asarray([True, True]),
        rejected=np.asarray([False, False]),
        bases=np.asarray(["pre-growth-semantic-consensus"] * 2),
        pregrowth_semantic_consensus=np.asarray([True, True]),
    )
    with tempfile.TemporaryDirectory() as tmp:
        field_run = Path(tmp) / "field"
        counts = _write_owner_field_inputs(
            field_run,
            np.asarray([7, 8]),
            source_frames,
            fps,
            centers,
            visible,
            visible,
            np.asarray([5, 5]),
            np.asarray([5, 5]),
            np.asarray(["high", "high"]),
            np.asarray([0.9, 0.9]),
            np.asarray([0.97, 0.97]),
            body,
            identity,
            (1024, 1280),
            150.0,
        )
        summary = pd.read_csv(field_run / "field_summary.csv")
    statuses = dict(zip(summary.pollen_id, summary.field_status))
    assert statuses[7] == "edge_seed_censored"
    assert statuses[8] == "review_quality"
    assert counts["edge_seed_censored_owner_count"] == 1


def _rescue_candidate(
    source,
    final_length_analysis_px,
    median_coverage,
    proposal_score,
    deformation=None,
):
    """Build one selection-eligible native boundary candidate."""
    from tubetracker.causal_growth_front import (
        GrowthDirectionCertificate,
        OwnerPathTopologyCertificate,
        RootedGrowthCertificate,
    )
    from tubetracker.causal_portal import NativePathQualityCertificate

    n = 8
    arclength = np.linspace(0.0, final_length_analysis_px, n)
    paths = np.zeros((3, n, 2))
    paths[:, :, 0] = arclength[None, :]
    births = np.zeros(n)
    fronts = np.arange(n)
    mask = np.ones(n, dtype=bool)
    result = CausalGrowthFrontResult(
        birth_samples=births,
        front_indices=fronts,
        direct_support_mask=mask,
        eventual_support_mask=mask,
        feasible=True,
        reason="causal-front-verified",
        objective_score=float(proposal_score),
        final_point_index=n - 1,
        final_length_px=float(final_length_analysis_px),
        direct_support_fraction=1.0,
        eventual_support_fraction=1.0,
    )
    return MaturePathCandidate(
        source=source,
        arclength_px=arclength.copy(),
        dynamic_paths_xy=paths.copy(),
        root_distance_px=5.0,
        radial_excursion_px=float(final_length_analysis_px),
        foreign_contact_owner_id=None,
        uncensored_length_px=float(final_length_analysis_px),
        profile=np.ones((3, n), dtype=np.float32),
        result=result,
        reverse_result=result,
        growth_certificate=RootedGrowthCertificate(
            accepted=True, reason="rooted-growth-verified"
        ),
        direction_certificate=GrowthDirectionCertificate(
            accepted=True,
            reason="growth-direction-verified",
            outward_normalized_score=1.0,
            inward_normalized_score=0.0,
            inward_score_margin=1.0,
        ),
        native_quality_certificate=NativePathQualityCertificate(
            accepted=True,
            reason="native-path-quality-verified",
            median_coverage=float(median_coverage),
            median_support_margin=1.0,
            observation_count=8,
            completion_fraction=1.0,
        ),
        proposal_score=float(proposal_score),
        deformation=deformation,
        rigid_paths_xy=paths.copy(),
        native_root_onset_sample=28,
        native_root_onset_delay_samples=5,
        native_root_onset_reason="native-root-onset-corroborated",
        topology_certificate=OwnerPathTopologyCertificate(
            accepted=True,
            reason="owner-path-topology-certified",
            root_distance_px=5.0,
            minimum_distance_px=5.0,
            maximum_distance_px=float(final_length_analysis_px),
            radial_excursion_px=float(final_length_analysis_px),
            first_halo_exit_index=4,
            first_halo_exit_arclength_px=4.0,
        ),
    )


def test_rescue_prefers_best_coverage_extension_over_longer_poor_one():
    """Coverage decides among length-qualified extensions, not raw score.

    Dense-field P76 motivation: several faint extensions can clear the
    length-gain gate while riding different support (0.95 vs 0.74 median
    coverage).  Within the decisive pool the best-covered candidate must
    win even when a longer, higher-scored rival exists.  (On real P76 data
    the longer rivals additionally fail the proximal-emergence gate, so the
    field falls back to the 53 px verified root — this test isolates only
    the coverage tie-break.)
    """

    from tubetracker.causal_growth_front import OwnerPathTopologyCertificate

    scale = 0.375
    ordinary = _rescue_candidate(
        "native-boundary-4-rigid", 90.0 * scale, 0.98, 1.0
    )
    long_poor = _rescue_candidate(
        "native-prefix-extension-7-deformable",
        147.0 * scale,
        0.7368,
        2.58,
        deformation=object(),
    )
    # The deformable twin must show a real coverage gain over its rigid twin.
    long_poor_rigid = _rescue_candidate(
        "native-prefix-extension-7-rigid", 147.0 * scale, 0.614, 2.58
    )
    short_good = _rescue_candidate(
        "native-prefix-extension-9-rigid", 106.5 * scale, 0.95, 2.088
    )
    n = 8
    owner = OwnerGeometry(
        track_id=76,
        field_status="measured",
        original_measurements=pd.DataFrame(
            {"sample_index": [0], "tube_length_px": [0.0]}
        ),
        mature_candidates=[],
        arclength_px=np.linspace(0.0, 10.0, n),
        dynamic_paths_xy=np.zeros((3, n, 2)),
        body_status="verified",
        topology_certificate=OwnerPathTopologyCertificate(
            accepted=False,
            reason="legacy-geometry-rejected",
            root_distance_px=5.0,
            minimum_distance_px=5.0,
            maximum_distance_px=10.0,
            radial_excursion_px=1.0,
            first_halo_exit_index=None,
            first_halo_exit_arclength_px=None,
        ),
        native_boundary_candidates=[
            ordinary,
            long_poor,
            long_poor_rigid,
            short_good,
        ],
    )
    report = _select_native_boundary_rescues(
        [owner],
        scale,
        5.90625,
        0.02,
        0.10,
        0.90,
    )
    assert owner.selected_mature_path == "native-prefix-extension-9-rigid"
    assert report["rescued_owner_ids"] == [76]


def _veto_test_owner(veto: bool) -> OwnerGeometry:
    """Minimal native-verified owner for the forecast-veto status test."""

    result = CausalGrowthFrontResult(
        birth_samples=np.asarray([0, 0], dtype=np.int32),
        front_indices=np.asarray([1], dtype=np.int32),
        direct_support_mask=np.ones(2, dtype=bool),
        eventual_support_mask=np.ones(2, dtype=bool),
        feasible=True,
        reason="ok",
        objective_score=1.0,
        final_point_index=1,
        final_length_px=1.0,
        direct_support_fraction=1.0,
        eventual_support_fraction=1.0,
    )
    return OwnerGeometry(
        track_id=131,
        field_status="measured",
        original_measurements=pd.DataFrame(),
        mature_candidates=[],
        arclength_px=np.asarray([0.0, 1.0]),
        dynamic_paths_xy=np.zeros((1, 2, 2), dtype=np.float64),
        body_status="verified",
        result=result,
        native_certificate=NativePathGrowthCertificate(
            accepted=True,
            reason="native-growth-verified",
            completion_fraction=1.0,
            onset_delay_samples=0,
        ),
        forecast_fault_samples=9 if veto else 0,
        forecast_replay_samples=125,
        forecast_stability_veto=veto,
    )


def test_forecast_veto_loader_reads_replay_counts(tmp_path):
    """Guided-replay CSVs become per-owner (fault, replayed) counts."""

    (tmp_path / "replay_P7.csv").write_text(
        "sample_index,verdict\n0,keep\n1,fault-suspect\n2,burst\n3,fault-suspect\n"
    )
    (tmp_path / "replay_P9.csv").write_text("sample_index,verdict\n0,keep\n")
    (tmp_path / "replay_P11.csv").write_text("sample_index,resid\n0,3.0\n")
    (tmp_path / "notes.txt").write_text("not a replay file\n")

    counts = _load_forecast_veto_counts(tmp_path)
    assert counts == {7: (2, 4), 9: (0, 1)}
    assert _load_forecast_veto_counts(tmp_path / "missing") == {}


def test_forecast_veto_trip_needs_count_and_rate():
    """One noisy row cannot demote; sustained flicker always trips."""

    assert _forecast_stability_veto_trips(9, 125, min_faults=3, min_rate=0.05)
    assert not _forecast_stability_veto_trips(2, 125, min_faults=3, min_rate=0.05)
    # Three flags of 125 replayed is a 0.024 rate: below the floor.
    assert not _forecast_stability_veto_trips(3, 125, min_faults=3, min_rate=0.05)
    assert _forecast_stability_veto_trips(3, 40, min_faults=3, min_rate=0.05)
    assert not _forecast_stability_veto_trips(0, 0, min_faults=3, min_rate=0.05)


def test_forecast_veto_overrides_native_verification():
    """A tripped TimesFM veto fail-closes even a native-verified claim.

    Dense-field P131 motivation: 9 fault-suspect replays of 125 with a
    native-verified 75 px claim whose trace flickers between phantom
    extension and stub.
    """

    assert (
        _causal_status(_veto_test_owner(True), True, {})
        == "forecast_stability_review"
    )
    assert (
        _causal_status(_veto_test_owner(False), True, {})
        == "native_verified_measured"
    )


def test_forecast_tip_loader_reads_hold_runs(tmp_path):
    """Guided tip-filter CSVs become per-owner (held, max_run, total)."""

    (tmp_path / "guided_P7.csv").write_text(
        "sample_index,held\n"
        "0,False\n1,True\n2,True\n3,False\n4,True\n5,True\n6,True\n"
    )
    (tmp_path / "guided_P9.csv").write_text("sample_index,held\n0,False\n")
    (tmp_path / "guided_P11.csv").write_text("sample_index,guided_x\n0,3.0\n")
    (tmp_path / "notes.txt").write_text("not a guided file\n")

    holds = _load_forecast_tip_holds(tmp_path)
    assert holds == {7: (5, 3, 7), 9: (0, 0, 1)}
    assert _load_forecast_tip_holds(tmp_path / "missing") == {}


def test_forecast_tip_loader_is_evidence_only(tmp_path):
    """Malformed rows fail open; the loader changes no status by itself."""

    (tmp_path / "guided_P7.csv").write_text("sample_index,held\n0,True\n")
    (tmp_path / "replay_P7.csv").write_text("sample_index,verdict\n0,keep\n")

    holds = _load_forecast_tip_holds(tmp_path)
    assert holds == {7: (1, 1, 1)}


def _recenter_refinement(offsets, accepted=True, reason="native-centerline-refinement-verified"):
    from tubetracker.causal_portal import NativeCenterlineRefinement

    offsets = np.asarray(offsets, dtype=np.float32)
    return NativeCenterlineRefinement(
        offsets_px=offsets,
        selected_support=np.ones_like(offsets),
        baseline_support=np.zeros_like(offsets),
        normalized_median_gain=0.9,
        positive_point_fraction=0.8,
        positive_time_fraction=0.9,
        search_edge_fraction=0.0,
        accepted=accepted,
        reason=reason,
    )


def test_gate_rescue_recenter_applies_material_off_axis_correction():
    """A P97-class fit (proximal third ~+10 px, gain verified) must apply."""

    offsets = np.zeros(21, dtype=np.float32)
    offsets[2:9] = np.linspace(4.0, 12.0, 7)
    offsets[9:14] = 12.0
    offsets[14:] = np.linspace(10.0, 0.0, 7)
    apply, reason = _gate_rescue_recenter(
        _recenter_refinement(offsets), 3.0
    )
    assert apply
    assert reason == "rescue-recenter-verified"


def test_gate_rescue_recenter_leaves_centered_claims_untouched():
    """A centered claim (median correction below the floor) must not move."""

    offsets = np.zeros(21, dtype=np.float32)
    offsets[2:5] = (1.0, 2.0, 1.0)
    apply, reason = _gate_rescue_recenter(
        _recenter_refinement(offsets), 3.0
    )
    assert not apply
    assert reason == "rescue-path-already-centered"


def test_gate_rescue_recenter_respects_fit_rejection():
    """A fit rejected for inconsistency (P76 rim-latch) must not apply."""

    apply, reason = _gate_rescue_recenter(
        _recenter_refinement(
            np.full(21, 8.0, dtype=np.float32),
            accepted=False,
            reason="native-gain-not-path-consistent",
        ),
        3.0,
    )
    assert not apply
    assert reason == "native-gain-not-path-consistent"


def test_recenter_length_change_rejects_sliding_onto_foreign_structure():
    """Dense P97: a 3px median shift that lengthens 14.6% is structural."""

    assert not _recenter_length_change_is_lateral(0.1457975925345664, 0.05)


def test_recenter_length_change_accepts_true_lateral_fix():
    """A genuine lateral shift preserves arclength within a few percent."""

    assert _recenter_length_change_is_lateral(0.0, 0.05)
    assert _recenter_length_change_is_lateral(0.03, 0.05)
    assert _recenter_length_change_is_lateral(0.05, 0.05)
    assert not _recenter_length_change_is_lateral(0.051, 0.05)


def test_sticky_tip_loader_reads_keep_moves(tmp_path):
    """Sticky series become per-owner (kept, max, mean, total)."""

    (tmp_path / "sticky_P7.csv").write_text(
        "sample_index,corrected_x,corrected_y,moved_px,keep\n"
        "0,1.0,2.0,0.0,0\n1,5.0,6.0,10.0,1\n2,7.0,8.0,20.0,1\n"
    )
    (tmp_path / "sticky_P9.csv").write_text(
        "sample_index,corrected_x,corrected_y,moved_px,keep\n0,1.0,2.0,0.0,0\n"
    )
    (tmp_path / "sticky_P11.csv").write_text("sample_index,corrected_x\n0,3.0\n")
    (tmp_path / "notes.txt").write_text("not a sticky file\n")

    ev = _load_sticky_tip_evidence(tmp_path)
    assert ev == {7: (2, 20.0, 15.0, 3), 9: (0, 0.0, 0.0, 1)}
    assert _load_sticky_tip_evidence(tmp_path / "missing") == {}


def test_sticky_tip_loader_is_evidence_only(tmp_path):
    """Malformed rows fail open; the loader changes no status by itself."""

    (tmp_path / "sticky_P7.csv").write_text("sample_index,keep\n0,True\n")

    ev = _load_sticky_tip_evidence(tmp_path)
    assert ev == {}
