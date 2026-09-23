#!/usr/bin/env python3
"""Bootstrap v29 owner inputs from a persistent pollen-identity census."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tubetracker.causal_birth_forest import track_pollen_centers_from_seeds
from tubetracker.pollen_motion import (
    DetectionSnapConfig,
    PollenMotionConfig,
    constrain_trajectories_to_unique_detections,
    detect_field_pollen_circles,
    pollen_body_radial_evidence,
    semantic_anchor_guard_weights,
    track_query_points,
    track_query_points_bidirectional,
)


def parse_args() -> argparse.Namespace:
    """Parse identity, frame-cache, confidence, and output controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity_report", type=Path)
    parser.add_argument("field_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-observations",
        type=int,
        help="Default: 3 for a pre-growth census, otherwise 1.",
    )
    parser.add_argument(
        "--minimum-semantic-observations",
        type=int,
        help="Default: 3 for a pre-growth census, otherwise 1.",
    )
    parser.add_argument("--template-radius-px", type=int, default=5)
    parser.add_argument("--search-radius-px", type=int, default=5)
    parser.add_argument("--minimum-template-score", type=float, default=0.25)
    parser.add_argument(
        "--motion-mode",
        choices=("auto", "template", "cotracker"),
        default="auto",
        help=(
            "Owner motion prior. Auto uses CoTracker for a pre-growth census "
            "when its external checkpoint is installed."
        ),
    )
    parser.add_argument("--boundary-distance-source-px", type=float, default=150.0)
    parser.add_argument(
        "--geometric-motion-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse template motion with one-to-one pollen-circle observations.",
    )
    parser.add_argument("--geometric-batch-size", type=int, default=24)
    parser.add_argument("--geometric-hough-threshold", type=int, default=8)
    parser.add_argument("--geometric-minimum-circle-score", type=float, default=0.2)
    parser.add_argument("--visibility-margin-source-px", type=float, default=15.0)
    parser.add_argument("--body-evidence-samples", type=int, default=5)
    parser.add_argument("--minimum-body-valid-samples", type=int, default=3)
    parser.add_argument("--body-gradient-threshold", type=float, default=3.0)
    parser.add_argument("--minimum-body-radial-contrast", type=float, default=3.5)
    parser.add_argument("--minimum-body-opposite-coverage", type=float, default=0.35)
    parser.add_argument(
        "--minimum-body-visible-circumference",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--minimum-pregrowth-semantic-observations",
        type=int,
        default=3,
        help=(
            "Independent learned-mask sightings required to certify an owner "
            "from an explicitly declared pre-growth census."
        ),
    )
    return parser.parse_args()


def select_persistent_identity_tracks(
    tracks: list[dict],
    minimum_observations: int,
    minimum_semantic_observations: int,
    *,
    overlap_merge_fraction: float = 0.5,
) -> list[dict]:
    """Retain repeat observations with independent learned-mask support.

    Cellpose can split one cracked-cluster grain into a stable small fragment
    plus the true body in every census frame (dense P89: a 10px overlap sliver
    persisting 5/5 frames beside real owners P90/P92).  Such fragments pass
    persistence gates and later trace phantom tubes, so flag a smaller track
    as a merge fragment when its mask is mostly covered by a larger neighbor
    (coverage of the smaller area above ``overlap_merge_fraction``) before
    promotion.
    """

    if minimum_observations < 1 or minimum_semantic_observations < 1:
        raise ValueError("identity confidence thresholds must be positive")
    if not 0.0 < overlap_merge_fraction < 1.0:
        raise ValueError("overlap merge fraction must be between zero and one")
    merged = _merge_overlapping_identity_tracks(tracks, overlap_merge_fraction)
    return sorted(
        (
            track
            for track in merged
            if int(track["observation_count"]) >= minimum_observations
            and int(track["semantic_observation_count"])
            >= minimum_semantic_observations
        ),
        key=lambda track: int(track["track_id"]),
    )


def _merge_overlapping_identity_tracks(
    tracks: list[dict],
    overlap_merge_fraction: float,
) -> list[dict]:
    """Fold stable mask-split fragments back into their largest neighbor.

    A split fragment sits mostly INSIDE one true body (coverage of the small
    mask near one), while two touching grains share only a thin lens (dense
    P89/P92 share 0.19 of the small mask; P89/P90 share 0.09).  The default
    0.5 fraction therefore keeps both members of a touching pair.
    """

    import numpy as np

    remaining = [dict(track) for track in tracks]
    if len(remaining) < 2:
        return remaining
    centers = np.asarray(
        [[float(value) for value in np.mean(track["centers_yx"], axis=0)]
         for track in remaining],
        dtype=np.float64,
    )
    radii = np.asarray(
        [float(np.mean(track["radii_px"])) for track in remaining],
        dtype=np.float64,
    )
    suppressed = np.zeros(len(remaining), dtype=bool)
    for small in np.argsort(np.pi * radii**2):
        if suppressed[small]:
            continue
        small_area = float(np.pi * radii[small] ** 2)
        best: int | None = None
        best_area = small_area
        for large in range(len(remaining)):
            if large == small or suppressed[large]:
                continue
            distance = float(np.linalg.norm(centers[small] - centers[large]))
            if distance >= float(radii[small] + radii[large]):
                continue
            overlap = _circle_overlap_area(
                float(radii[small]), float(radii[large]), distance
            )
            if overlap < overlap_merge_fraction * small_area:
                continue
            large_area = float(np.pi * radii[large] ** 2)
            if large_area > best_area:
                best_area = large_area
                best = large
        if best is not None:
            suppressed[small] = True
            survivor = remaining[best]
            fragment = remaining[small]
            survivor_notes = list(survivor.get("merged_fragment_ids", ()))
            survivor_notes.append(int(fragment["track_id"]))
            survivor["merged_fragment_ids"] = survivor_notes
    return [track for index, track in enumerate(remaining) if not suppressed[index]]


def _circle_overlap_area(radius_a: float, radius_b: float, distance: float) -> float:
    """Return the lens area shared by two circles."""

    import math

    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return math.pi * min(radius_a, radius_b) ** 2
    if distance <= 0.0:
        return math.pi * min(radius_a, radius_b) ** 2
    first = (
        radius_a**2
        * math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    (distance**2 + radius_a**2 - radius_b**2)
                    / (2.0 * distance * radius_a),
                ),
            )
        )
    )
    second = (
        radius_b**2
        * math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    (distance**2 + radius_b**2 - radius_a**2)
                    / (2.0 * distance * radius_b),
                ),
            )
        )
    )
    lens = 0.5 * math.sqrt(
        max(
            0.0,
            (-distance + radius_a + radius_b)
            * (distance + radius_a - radius_b)
            * (distance - radius_a + radius_b)
            * (distance + radius_a + radius_b),
        )
    )
    return first + second - lens


def identity_confidence_tier(track: dict) -> str:
    """Label owner identity strength without discarding learned detections."""

    observations = int(track["observation_count"])
    semantic = int(track["semantic_observation_count"])
    if observations >= 5 and semantic >= 4:
        return "high"
    if observations >= 3 and semantic >= 2:
        return "supported"
    return "provisional"


def identity_seed(
    track: dict,
    source_frames: np.ndarray,
    shifts_xy: np.ndarray,
    analysis_scale: float,
) -> tuple[int, np.ndarray]:
    """Map one median identity observation into an aligned analysis frame."""

    observation_frames = np.asarray(track["source_frames"], dtype=np.int64)
    centers_yx = np.asarray(track["centers_yx"], dtype=np.float64)
    middle = len(observation_frames) // 2
    sample = int(np.argmin(np.abs(source_frames - observation_frames[middle])))
    center_yx = centers_yx[middle]
    aligned_yx = center_yx * analysis_scale - shifts_xy[sample, ::-1]
    return sample, aligned_yx


def resolve_owner_motion_mode(
    requested_mode: str,
    census_phase: str,
    config: PollenMotionConfig | None = None,
) -> tuple[str, str | None]:
    """Resolve reproducible owner motion with an actionable optional fallback."""

    if requested_mode not in {"auto", "template", "cotracker"}:
        raise ValueError("owner motion mode is invalid")
    if requested_mode == "template":
        return "template", None
    config = config or PollenMotionConfig.from_environment()
    unavailable_reason = None
    if not config.checkpoint.exists():
        unavailable_reason = f"checkpoint not installed at {config.checkpoint}"
    elif importlib.util.find_spec("cotracker") is None:
        unavailable_reason = "CoTracker package is not installed"
    if requested_mode == "cotracker":
        if unavailable_reason:
            raise RuntimeError(unavailable_reason)
        return "cotracker", None
    if census_phase == "pre-growth" and unavailable_reason is None:
        return "cotracker", None
    fallback = unavailable_reason if census_phase == "pre-growth" else None
    return "template", fallback


def track_owner_centers_with_cotracker(
    frames: np.ndarray,
    seed_centers_yx: np.ndarray,
    seed_samples: np.ndarray,
    config: PollenMotionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track all clean pollen centers jointly in full-field temporal context."""

    images = np.asarray(frames, dtype=np.uint8)
    centers = np.asarray(seed_centers_yx, dtype=np.float64)
    samples = np.asarray(seed_samples, dtype=np.int64)
    if images.ndim != 3 or len(images) < 2:
        raise ValueError("owner point tracking requires a grayscale video")
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
        raise ValueError("owner seeds must have shape (owners, 2)")
    if samples.shape != (len(centers),):
        raise ValueError("one seed sample is required per owner")
    if (
        not np.isfinite(centers).all()
        or np.any(samples < 0)
        or np.any(samples >= len(images))
    ):
        raise ValueError("owner point-tracking seeds are invalid")
    queries = np.column_stack((samples, centers[:, ::-1]))
    if np.all(samples == 0):
        tracks_xy, visibility = track_query_points(images, queries, config=config)
    else:
        tracks_xy, visibility = track_query_points_bidirectional(
            images,
            queries,
            config=config,
        )
    centers_yx = np.asarray(tracks_xy, dtype=np.float64).transpose(1, 0, 2)[
        :, :, ::-1
    ]
    visible = np.asarray(visibility, dtype=bool).T
    expected = (len(centers), len(images))
    if centers_yx.shape != (*expected, 2) or visible.shape != expected:
        raise RuntimeError("CoTracker returned an unexpected owner trajectory shape")
    return centers_yx, visible


@dataclass(frozen=True)
class OwnerBodyCertificates:
    """Store robust owner-level decisions and their frame-level radial evidence."""

    statuses: np.ndarray
    verified: np.ndarray
    rejected: np.ndarray
    median_radial_contrast: np.ndarray
    median_angular_boundary_fraction: np.ndarray
    median_opposite_boundary_fraction: np.ndarray
    median_valid_angular_fraction: np.ndarray
    valid_sample_counts: np.ndarray
    sample_indices: np.ndarray
    radial_contrast_samples: np.ndarray
    angular_boundary_samples: np.ndarray
    opposite_boundary_samples: np.ndarray
    valid_angular_samples: np.ndarray


@dataclass(frozen=True)
class OwnerIdentityCertificates:
    """Store the reconciled owner decision and its auditable evidence source."""

    statuses: np.ndarray
    verified: np.ndarray
    rejected: np.ndarray
    bases: np.ndarray
    pregrowth_semantic_consensus: np.ndarray
    seed_interior_darkness: np.ndarray | None = None


def _flag_contested_identity_tracks(
    tracks: list[dict],
    *,
    overlap_fraction: float = 0.08,
    duplicate_center_fraction: float = 1.75,
) -> "np.ndarray":
    """Flag census tracks whose masks materially overlap a larger neighbor.

    Dense P89's sliver shares only 0.09-0.24 of its small mask with P90/P92,
    so a low fraction catches split fragments while keeping cleanly separated
    grains untouched.  A second, independent rule catches duplicate seeds on
    ONE body: when two centers sit closer than ``duplicate_center_fraction``
    of the smaller radius (dense P65/P69 at 1.54R, P80/P81 at 1.64R — both
    eye-verified as one grain each carrying two seeds), the smaller is
    contested however small the lens overlap is.  At 1.75 small-radii only
    these near-coincident seeds trigger: the next-closest census pair sits
    at 1.91R, and genuinely touching grains (P47/P50 at 5.2R) are untouched.
    Flagged owners stay in the report as diagnostics but cannot receive automatic pre-growth
    verification.
    """

    import numpy as np

    flagged = np.zeros(len(tracks), dtype=bool)
    centers = np.asarray(
        [[float(value) for value in np.mean(track["centers_yx"], axis=0)]
         for track in tracks],
        dtype=np.float64,
    )
    radii = np.asarray(
        [float(np.mean(track["radii_px"])) for track in tracks],
        dtype=np.float64,
    )
    areas = np.pi * radii**2
    for small in range(len(tracks)):
        for large in range(len(tracks)):
            if large == small or areas[large] <= areas[small]:
                continue
            distance = float(np.linalg.norm(centers[small] - centers[large]))
            if distance < duplicate_center_fraction * float(radii[small]):
                # Two seeds well inside one body radius: duplicate seeds on
                # a single grain, not two touching bodies.
                flagged[small] = True
                break
            if distance >= float(radii[small] + radii[large]):
                continue
            overlap = _circle_overlap_area(
                float(radii[small]), float(radii[large]), distance
            )
            if overlap >= overlap_fraction * float(areas[small]):
                flagged[small] = True
                break
    return flagged


def _seed_interior_darkness(
    frames: "np.ndarray",
    seed_centers_yx: "np.ndarray",
    seed_samples: "np.ndarray",
    pollen_radius_px: float,
) -> "np.ndarray":
    """Measure seed-anchored grain-interior darkness on the census frame.

    The radial body certificate is computed on tracked mature-frame centers,
    which can drift onto neighboring bodies in crowded clusters.  This probe
    is anchored to the pre-growth census seed itself: a low percentile of
    the gray values inside a small interior disk on the seed's own sample.
    Real grains read dark somewhere inside (dense-field grain interiors
    reach 60-130 against a ~179 bright background); seeds floating on empty
    background stay bright throughout (dense P76: p10 158).  The low
    percentile — not the median — is the right statistic because a seed a
    few pixels off a grain center still overlaps dark interior somewhere in
    the disk, while background never goes dark.  Exported as evidence and
    used to refuse automatic pre-growth verification — never to delete an
    owner.
    """

    import numpy as np

    images = np.asarray(frames)
    seeds = np.asarray(seed_samples, dtype=np.int64)
    centers = np.asarray(seed_centers_yx, dtype=np.float64)
    darkness = np.full(len(seeds), np.nan, dtype=np.float64)
    disk_radius = max(2.0, 0.55 * float(pollen_radius_px))
    for owner in range(len(seeds)):
        sample = int(seeds[owner])
        if sample < 0 or sample >= len(images):
            continue
        image = np.asarray(images[sample], dtype=np.float64)
        if image.ndim != 2 or not image.size:
            continue
        center = centers[owner]
        if center.shape != (2,) or not np.isfinite(center).all():
            continue
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        disk = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= disk_radius**2
        if not bool(disk.any()):
            continue
        darkness[owner] = float(np.percentile(image[disk], 10.0))
    return darkness


def reconcile_owner_identity_certificates(
    tracks: list[dict],
    body_certificates: OwnerBodyCertificates,
    census_phase: str,
    minimum_pregrowth_semantic_observations: int = 3,
    seed_interior_darkness: "np.ndarray | None" = None,
    maximum_seed_interior_gray: float = 160.0,
) -> OwnerIdentityCertificates:
    """Prefer a persistent pre-growth census over ambiguous mature-tube shape.

    A closed radial profile is useful when a pollen first appears after tubes
    exist, but at low analysis resolution it can reject a real, blurred pollen
    or accept two parallel tube walls.  An explicitly declared pre-growth
    census removes that ambiguity: persistent learned instances cannot be tube
    fragments because the tubes have not emerged yet.

    When ``seed_interior_darkness`` is given, census seeds whose interior
    disk never goes dark (eye-verified background seeds P8/P14/P41 at p10
    161-181 vs real grains at p10 75-152) are refused automatic pre-growth
    verification: the seed floats on empty background, so no grain was ever
    there to own a tube.  They stay exported as indeterminate diagnostics —
    never deleted.
    """

    if census_phase not in {"unconstrained", "pre-growth"}:
        raise ValueError("census phase must be unconstrained or pre-growth")
    if minimum_pregrowth_semantic_observations < 1:
        raise ValueError("pregrowth semantic support must be positive")
    owner_count = len(tracks)
    if body_certificates.statuses.shape != (owner_count,):
        raise ValueError("body certificates must match the selected tracks")

    statuses = body_certificates.statuses.astype("<U13", copy=True)
    contested = np.zeros(owner_count, dtype=bool)
    if owner_count >= 2:
        contested = _flag_contested_identity_tracks(tracks)
        statuses[contested & (statuses == "verified")] = "indeterminate"
    bases = np.asarray(
        [
            (
                "radial-body-certificate"
                if status != "indeterminate"
                else "insufficient-visible-body-evidence"
            )
            for status in statuses
        ],
        dtype="<U40",
    )
    if contested.any():
        bases[contested] = np.where(
            statuses[contested] == "indeterminate",
            "contested-cluster-identity",
            bases[contested],
        )
    pregrowth = np.zeros(owner_count, dtype=bool)
    if census_phase == "pre-growth":
        pregrowth = np.asarray(
            [
                int(track.get("semantic_observation_count", 0))
                >= minimum_pregrowth_semantic_observations
                for track in tracks
            ],
            dtype=bool,
        )
        background_seed = np.zeros(owner_count, dtype=bool)
        if seed_interior_darkness is not None:
            darkness = np.asarray(seed_interior_darkness, dtype=np.float64)
            if darkness.shape != (owner_count,):
                raise ValueError("seed darkness must match the selected tracks")
            background_seed = np.isfinite(darkness) & (
                darkness > maximum_seed_interior_gray
            )
            statuses[background_seed & (statuses == "verified")] = "indeterminate"
        uncontested = pregrowth & ~contested & ~background_seed
        statuses[uncontested] = "verified"
        bases[uncontested] = "pre-growth-semantic-consensus"
        bases[pregrowth & contested] = "contested-cluster-identity"
        bases[pregrowth & background_seed & ~contested] = (
            "background-seed-no-grain-interior"
        )
    return OwnerIdentityCertificates(
        statuses=statuses,
        verified=statuses == "verified",
        rejected=statuses == "rejected",
        bases=bases,
        pregrowth_semantic_consensus=pregrowth,
        seed_interior_darkness=(
            None
            if seed_interior_darkness is None
            else np.asarray(seed_interior_darkness, dtype=np.float64)
        ),
    )


def _demote_unsupported_identity_tracks(
    tracks: list[dict],
    certificates: OwnerIdentityCertificates,
    assigned_fraction: "np.ndarray | None",
    *,
    minimum_supported_assignment_fraction: float = 0.25,
) -> OwnerIdentityCertificates:
    """Demote census owners the mature field never geometrically supports.

    Dense P89 persists 5/5 census frames yet receives a circle assignment on
    only 2.5% of mature samples (its larger neighbors win every Hungarian
    round).  Contested owners below the support floor stay exported but lose
    automatic verification, since no independent detector confirms a body
    there.  Uncontested owners are untouched however low their assignment is.
    """

    import numpy as np

    if assigned_fraction is None:
        return certificates
    fractions = np.asarray(assigned_fraction, dtype=np.float64)
    if fractions.shape != (len(tracks),):
        raise ValueError("assignment fractions must match the selected tracks")
    contested = _flag_contested_identity_tracks(tracks)
    demote = (
        contested
        & certificates.verified
        & (fractions < minimum_supported_assignment_fraction)
    )
    if not bool(np.any(demote)):
        return certificates
    statuses = certificates.statuses.astype("<U13", copy=True)
    bases = certificates.bases.astype("<U40", copy=True)
    statuses[demote] = "indeterminate"
    bases[demote] = "contested-cluster-identity"
    return OwnerIdentityCertificates(
        statuses=statuses,
        verified=statuses == "verified",
        rejected=statuses == "rejected",
        bases=bases,
        pregrowth_semantic_consensus=certificates.pregrowth_semantic_consensus,
        seed_interior_darkness=certificates.seed_interior_darkness,
    )


def _snap_corrected_mature_anchoring(
    template_yx: "np.ndarray",
    assignments: "np.ndarray",
    detections_by_sample: list["np.ndarray"],
    analysis_scale: float,
) -> "np.ndarray":
    """Measure mature-frame anchoring against raw Hough assignments, unsmoothed.

    The exported trajectory carries Savitzky-Golay-smoothed detection
    corrections, which can drag a center off its grain when a neighboring
    body wins nearby Hungarian rounds.  This probe replays the raw
    per-sample assignments WITHOUT smoothing: for each owner and sample it
    records the distance from the UNSMOOTHED observed detection to the
    smoothed exported center.  Large sustained snap distances mean the
    smoothing — not the image evidence — placed the center, so downstream
    ownership of a tube traced from that center is suspect.  Evidence only;
    the exported trajectory is untouched.
    """

    import numpy as np

    template = np.asarray(template_yx, dtype=np.float64)
    assigned = np.asarray(assignments, dtype=np.int64)
    owner_count = template.shape[0]
    sample_count = template.shape[1]
    snap = np.full((owner_count, sample_count), np.nan, dtype=np.float64)
    for sample in range(sample_count):
        detections = np.asarray(detections_by_sample[sample], dtype=np.float64)
        if detections.size == 0:
            continue
        detections = np.asarray(detections).reshape(-1, 4)
        if detections.shape[1] != 4:
            continue
        for owner in range(owner_count):
            column = int(assigned[owner, sample])
            if column < 0 or column >= len(detections):
                continue
            observed = detections[column, :2]
            if not np.isfinite(observed).all():
                continue
            snap[owner, sample] = float(
                np.linalg.norm(observed - template[owner, sample])
                / max(float(analysis_scale), 1e-9)
            )
    return snap


def _audit_owner_motion_ambiguity(
    tracks: list[dict],
    assigned_fraction: "np.ndarray",
    assignment_costs: "np.ndarray | None",
    template_scores: "np.ndarray",
    point_tracker_visible: "np.ndarray | None",
    *,
    maximum_supported_assignment_fraction: float = 0.35,
    minimum_template_score_p10: float = 0.5,
    minimum_tracker_visibility: float = 0.9,
) -> dict[str, object]:
    """Score global owner-motion ambiguity with best/second-best evidence.

    An owner is motion-ambiguous when the mature field never geometrically
    supports it (low Hungarian assignment fraction at saturated cost), its
    local template match is weak (low p10 score), or its point tracker
    spends long spans invisible.  Ambiguous owners keep their predicted
    coordinate, but the audit names them so run.py can fail closed on
    identity switches instead of tracing phantom tubes from drifted centers.
    """

    import numpy as np

    fractions = np.asarray(assigned_fraction, dtype=np.float64)
    scores = np.asarray(template_scores, dtype=np.float64)
    if fractions.shape != (len(tracks),) or scores.shape[0] != len(tracks):
        raise ValueError("motion audit arrays must match the selected tracks")
    if assignment_costs is not None:
        costs = np.asarray(assignment_costs, dtype=np.float64)
        if costs.shape != scores.shape:
            raise ValueError("assignment costs must match the score array")
        cost_p90 = np.percentile(costs, 90.0, axis=1)
    else:
        cost_p90 = np.full(len(tracks), np.nan)
    score_p10 = np.percentile(scores, 10.0, axis=1)
    if point_tracker_visible is not None:
        visibility = np.mean(np.asarray(point_tracker_visible, dtype=bool), axis=1)
    else:
        visibility = np.full(len(tracks), np.nan)
    ambiguous_ids: list[int] = []
    second_best_margin: list[float] = []
    for index, track in enumerate(tracks):
        unsupported = (
            float(fractions[index]) < maximum_supported_assignment_fraction
            and (
                assignment_costs is None
                or float(cost_p90[index]) >= 1.15 - 1e-9
            )
        )
        weak_template = float(score_p10[index]) < minimum_template_score_p10
        low_visibility = (
            point_tracker_visible is not None
            and float(visibility[index]) < minimum_tracker_visibility
            and float(fractions[index]) < maximum_supported_assignment_fraction
        )
        if unsupported or weak_template or low_visibility:
            ambiguous_ids.append(int(track["track_id"]))
            second_best_margin.append(
                round(
                    float(
                        max(
                            maximum_supported_assignment_fraction
                            - float(fractions[index]),
                            minimum_template_score_p10 - float(score_p10[index]),
                            0.0,
                        )
                    ),
                    4,
                )
            )
    return {
        "ambiguous_owner_ids": ambiguous_ids,
        "second_best_margin_by_owner": dict(zip(ambiguous_ids, second_best_margin)),
        "assignment_fraction_p10_by_owner": {
            int(track["track_id"]): round(float(value), 4)
            for track, value in zip(tracks, fractions)
        },
        "assignment_cost_p90_by_owner": {
            int(track["track_id"]): (
                None if assignment_costs is None else round(float(value), 4)
            )
            for track, value in zip(tracks, cost_p90)
        },
        "template_score_p10_by_owner": {
            int(track["track_id"]): round(float(value), 4)
            for track, value in zip(tracks, score_p10)
        },
        "point_tracker_visibility_by_owner": {
            int(track["track_id"]): (
                None
                if point_tracker_visible is None
                else round(float(value), 4)
            )
            for track, value in zip(tracks, visibility)
        },
    }


def certify_owner_pollen_bodies(
    frames: np.ndarray,
    centers_yx: np.ndarray,
    seed_samples: np.ndarray,
    pollen_radius_px: float,
    *,
    evidence_samples: int = 5,
    minimum_valid_samples: int = 3,
    boundary_gradient_threshold: float = 3.0,
    minimum_radial_contrast: float = 3.5,
    minimum_opposite_coverage: float = 0.35,
    minimum_visible_circumference: float = 0.75,
) -> OwnerBodyCertificates:
    """Certify closed pollen bodies near semantic seeds without circle snapping."""

    images = np.asarray(frames)
    centers = np.asarray(centers_yx, dtype=np.float64)
    seeds = np.asarray(seed_samples, dtype=np.int64)
    scalar_values = (
        pollen_radius_px,
        boundary_gradient_threshold,
        minimum_radial_contrast,
        minimum_opposite_coverage,
        minimum_visible_circumference,
    )
    if images.ndim != 3 or not images.shape[0]:
        raise ValueError("frames must contain time, row, and column dimensions")
    if centers.shape != (len(seeds), len(images), 2) or not np.isfinite(centers).all():
        raise ValueError("centers must provide one finite yx point per owner and frame")
    if len(seeds) == 0 or np.any(seeds < 0) or np.any(seeds >= len(images)):
        raise ValueError("seed samples must identify existing frames")
    if evidence_samples < 1 or minimum_valid_samples < 1:
        raise ValueError("body evidence sample counts must be positive")
    if minimum_valid_samples > min(evidence_samples, len(images)):
        raise ValueError("minimum valid samples exceed available evidence samples")
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("body evidence thresholds must be finite")
    if pollen_radius_px <= 0.0 or boundary_gradient_threshold < 0.0:
        raise ValueError("body radius and gradient threshold are invalid")
    if minimum_radial_contrast < 0.0:
        raise ValueError("minimum radial contrast cannot be negative")
    if not 0.0 <= minimum_opposite_coverage <= 1.0:
        raise ValueError("minimum opposite coverage must be between zero and one")
    if not 0.0 <= minimum_visible_circumference <= 1.0:
        raise ValueError("minimum visible circumference must be between zero and one")

    sample_count = min(evidence_samples, len(images))
    owner_count = len(seeds)
    sample_indices = np.empty((owner_count, sample_count), dtype=np.int32)
    radial_samples = np.empty((owner_count, sample_count), dtype=np.float64)
    angular_samples = np.empty_like(radial_samples)
    opposite_samples = np.empty_like(radial_samples)
    valid_samples = np.empty_like(radial_samples)
    frame_indices = np.arange(len(images))
    for owner, seed in enumerate(seeds):
        selected = np.argsort(np.abs(frame_indices - seed), kind="stable")[:sample_count]
        selected.sort()
        sample_indices[owner] = selected
        for column, sample in enumerate(selected):
            evidence = pollen_body_radial_evidence(
                images[sample],
                centers[owner, sample],
                pollen_radius_px,
                boundary_gradient_threshold=boundary_gradient_threshold,
            )
            radial_samples[owner, column] = evidence.median_radial_contrast
            angular_samples[owner, column] = evidence.angular_boundary_fraction
            opposite_samples[owner, column] = evidence.opposite_boundary_fraction
            valid_samples[owner, column] = evidence.valid_angular_fraction

    usable = valid_samples >= minimum_visible_circumference
    valid_counts = np.count_nonzero(usable, axis=1).astype(np.int32)
    medians = []
    for values in (radial_samples, angular_samples, opposite_samples, valid_samples):
        medians.append(
            np.asarray(
                [
                    float(np.median(values[owner, usable[owner]]))
                    if valid_counts[owner]
                    else 0.0
                    for owner in range(owner_count)
                ],
                dtype=np.float64,
            )
        )
    radial_median, angular_median, opposite_median, valid_median = medians
    enough_evidence = valid_counts >= minimum_valid_samples
    verified = (
        enough_evidence
        & (radial_median >= minimum_radial_contrast)
        & (opposite_median >= minimum_opposite_coverage)
    )
    rejected = enough_evidence & ~verified
    statuses = np.full(owner_count, "indeterminate", dtype="<U13")
    statuses[rejected] = "rejected"
    statuses[verified] = "verified"
    return OwnerBodyCertificates(
        statuses=statuses,
        verified=verified,
        rejected=rejected,
        median_radial_contrast=radial_median,
        median_angular_boundary_fraction=angular_median,
        median_opposite_boundary_fraction=opposite_median,
        median_valid_angular_fraction=valid_median,
        valid_sample_counts=valid_counts,
        sample_indices=sample_indices,
        radial_contrast_samples=radial_samples,
        angular_boundary_samples=angular_samples,
        opposite_boundary_samples=opposite_samples,
        valid_angular_samples=valid_samples,
    )


def neutral_owner_stub(
    center_yx: np.ndarray,
    source_shape: tuple[int, int],
    pollen_radius_px: float = 15.0,
    stub_length_px: float = 5.0,
) -> np.ndarray:
    """Create an in-frame placeholder that cannot pass as measured growth."""

    center = np.asarray(center_yx, dtype=np.float64)
    image_center = 0.5 * (np.asarray(source_shape, dtype=np.float64) - 1.0)
    direction = image_center - center
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-9:
        direction = np.asarray((1.0, 0.0))
    else:
        direction /= magnitude
    return np.vstack(
        (
            center + pollen_radius_px * direction,
            center + (pollen_radius_px + stub_length_px) * direction,
        )
    )


def source_visibility_mask(
    source_centers_yx: np.ndarray,
    source_shape: tuple[int, int],
    margin_px: float,
) -> np.ndarray:
    """Mark samples whose pollen center remains safely inside the source frame."""

    centers = np.asarray(source_centers_yx, dtype=np.float64)
    if centers.ndim != 3 or centers.shape[2] != 2:
        raise ValueError("source centers must have shape (owners, samples, 2)")
    if margin_px < 0.0 or 2.0 * margin_px >= min(source_shape):
        raise ValueError("visibility margin must fit inside the source frame")
    return (
        (centers[:, :, 0] >= margin_px)
        & (centers[:, :, 0] <= source_shape[0] - 1.0 - margin_px)
        & (centers[:, :, 1] >= margin_px)
        & (centers[:, :, 1] <= source_shape[1] - 1.0 - margin_px)
    )


def _write_owner_field_inputs(
    field_run: Path,
    track_ids: np.ndarray,
    source_frames: np.ndarray,
    fps: float,
    source_centers_yx: np.ndarray,
    visible_mask: np.ndarray,
    reliably_visible_mask: np.ndarray,
    observation_counts: np.ndarray,
    semantic_observation_counts: np.ndarray,
    identity_tiers: np.ndarray,
    assignment_fractions: np.ndarray,
    median_template_scores: np.ndarray,
    body_certificates: OwnerBodyCertificates,
    identity_certificates: OwnerIdentityCertificates,
    source_shape: tuple[int, int],
    boundary_distance_px: float,
) -> dict[str, object]:
    """Write neutral per-owner histories in the established field-run format."""

    summary_rows = []
    boundary_count = 0
    censored_count = 0
    for owner, track_id in enumerate(track_ids):
        visible_samples = np.flatnonzero(visible_mask[owner])
        if not len(visible_samples):
            raise ValueError(f"owner {int(track_id)} is never visible")
        reference_sample = int(visible_samples[-1])
        final_center = source_centers_yx[owner, reference_sample]
        first_center = source_centers_yx[owner, int(visible_samples[0])]
        edge_distance = float(
            np.min(
                (
                    final_center[0],
                    final_center[1],
                    source_shape[0] - 1.0 - final_center[0],
                    source_shape[1] - 1.0 - final_center[1],
                )
            )
        )
        first_edge_distance = float(
            np.min(
                (
                    first_center[0],
                    first_center[1],
                    source_shape[0] - 1.0 - first_center[0],
                    source_shape[1] - 1.0 - first_center[1],
                )
            )
        )
        if edge_distance <= boundary_distance_px:
            field_status = "boundary_censored"
        elif first_edge_distance <= boundary_distance_px:
            # A census seed born inside the boundary margin is usually a
            # clipped edge object, not a full pollen body: its center cannot
            # anchor owner-relative topology.  Withhold from automatic tracing
            # while retaining all diagnostics.
            field_status = "edge_seed_censored"
            censored_count += 1
        else:
            field_status = "review_quality"
        boundary_count += int(field_status == "boundary_censored")
        owner_dir = field_run / f"P{int(track_id):02d}"
        owner_dir.mkdir(parents=True, exist_ok=True)
        stub = neutral_owner_stub(final_center, source_shape)
        pd.DataFrame(
            {
                "pollen_id": int(track_id),
                "sample_index": reference_sample,
                "source_frame": int(source_frames[reference_sample]),
                "point_index": (0, 1),
                "source_y_px": stub[:, 0],
                "source_x_px": stub[:, 1],
                "arc_length_px": (0.0, 5.0),
                "measurement_status": field_status,
            }
        ).to_csv(owner_dir / "centerlines.csv", index=False)
        pd.DataFrame(
            {
                "pollen_id": int(track_id),
                "sample_index": np.arange(len(source_frames)),
                "source_frame": source_frames,
                "time_minutes": source_frames / fps / 60.0,
                "tube_length_px": 0.0,
                "observed_tube_length_px": 0.0,
                "accepted": 0,
                "paired_fraction": 0.0,
                "mean_paired_support": 0.0,
                "prior_error_px": 0.0,
                "interpolated": 0,
                "temporally_confirmed": 1,
                "owner_visible": visible_mask[owner].astype(np.uint8),
                "owner_reliably_visible": reliably_visible_mask[owner].astype(
                    np.uint8
                ),
                "owner_identity_tier": str(identity_tiers[owner]),
                "owner_body_status": str(identity_certificates.statuses[owner]),
                "owner_body_verified": int(identity_certificates.verified[owner]),
                "owner_identity_certificate_basis": str(
                    identity_certificates.bases[owner]
                ),
                "measurement_status": field_status,
            }
        ).to_csv(owner_dir / "measurements.csv", index=False)
        summary_rows.append(
            {
                "pollen_id": int(track_id),
                "status": field_status,
                "first_persistent_source_frame": "",
                "first_persistent_time_minutes": "",
                "final_length_px": 5.0,
                "revision": "v29-identity-bootstrap",
                "field_status": field_status,
                "identity_observation_count": int(observation_counts[owner]),
                "semantic_observation_count": int(
                    semantic_observation_counts[owner]
                ),
                "owner_identity_tier": str(identity_tiers[owner]),
                "geometric_assignment_fraction": round(
                    float(assignment_fractions[owner]),
                    4,
                ),
                "median_template_score": round(
                    float(median_template_scores[owner]),
                    4,
                ),
                "owner_body_status": str(identity_certificates.statuses[owner]),
                "owner_body_verified": int(identity_certificates.verified[owner]),
                "owner_body_rejected": int(identity_certificates.rejected[owner]),
                "owner_identity_certificate_basis": str(
                    identity_certificates.bases[owner]
                ),
                "owner_pregrowth_semantic_consensus": int(
                    identity_certificates.pregrowth_semantic_consensus[owner]
                ),
                "owner_radial_body_status": str(
                    body_certificates.statuses[owner]
                ),
                "body_median_radial_contrast": round(
                    float(body_certificates.median_radial_contrast[owner]), 4
                ),
                "body_median_angular_boundary_fraction": round(
                    float(
                        body_certificates.median_angular_boundary_fraction[owner]
                    ),
                    4,
                ),
                "body_median_opposite_boundary_fraction": round(
                    float(
                        body_certificates.median_opposite_boundary_fraction[owner]
                    ),
                    4,
                ),
                "body_median_valid_angular_fraction": round(
                    float(body_certificates.median_valid_angular_fraction[owner]), 4
                ),
                "body_valid_sample_count": int(
                    body_certificates.valid_sample_counts[owner]
                ),
                "seed_interior_median_gray": (
                    ""
                    if identity_certificates.seed_interior_darkness is None
                    else round(
                        float(
                            identity_certificates.seed_interior_darkness[owner]
                        ),
                        1,
                    )
                ),
                "first_visible_sample": int(visible_samples[0]),
                "last_visible_sample": int(visible_samples[-1]),
                "visible_sample_fraction": float(np.mean(visible_mask[owner])),
            }
        )
    pd.DataFrame(summary_rows).to_csv(field_run / "field_summary.csv", index=False)
    return {
        "owner_count": len(track_ids),
        "boundary_candidate_owner_count": boundary_count,
        "edge_seed_censored_owner_count": censored_count,
        "owner_identity_tier_counts": {
            str(tier): int(np.count_nonzero(identity_tiers == tier))
            for tier in ("high", "supported", "provisional")
        },
        "owner_body_status_counts": {
            status: int(np.count_nonzero(identity_certificates.statuses == status))
            for status in ("verified", "rejected", "indeterminate")
        },
        "owner_radial_body_status_counts": {
            status: int(np.count_nonzero(body_certificates.statuses == status))
            for status in ("verified", "rejected", "indeterminate")
        },
        "pregrowth_semantic_consensus_owner_count": int(
            np.count_nonzero(identity_certificates.pregrowth_semantic_consensus)
        ),
    }


def main() -> None:
    """Track persistent pollen owners and write neutral v29 bootstrap inputs."""

    args = parse_args()
    identity_path = args.identity_report.expanduser().resolve()
    field_cache = args.field_cache.expanduser().resolve()
    identity = json.loads(identity_path.read_text())
    census_phase = str(identity.get("census", {}).get("phase", "unconstrained"))
    minimum_observations = (
        args.minimum_observations
        if args.minimum_observations is not None
        else (3 if census_phase == "pre-growth" else 1)
    )
    minimum_semantic_observations = (
        args.minimum_semantic_observations
        if args.minimum_semantic_observations is not None
        else (3 if census_phase == "pre-growth" else 1)
    )
    manifest = json.loads((field_cache / "manifest.json").read_text())
    source_frames = np.load(field_cache / "source_frames.npy")
    shifts_xy = np.load(field_cache / "shifts_xy.npy")
    frames = np.load(field_cache / "aligned_frames.npy", mmap_mode="r")
    analysis_scale = float(manifest["analysis_width"]) / float(
        manifest["native_width"]
    )
    selected = select_persistent_identity_tracks(
        identity["tracks"],
        minimum_observations,
        minimum_semantic_observations,
    )
    if not selected:
        raise ValueError("identity report contains no owners at this confidence")

    seeds = [
        identity_seed(track, source_frames, shifts_xy, analysis_scale)
        for track in selected
    ]
    seed_samples = np.asarray([sample for sample, _ in seeds], dtype=np.int64)
    seed_centers_yx = np.asarray([center for _, center in seeds])
    raw_template_yx, template_scores = track_pollen_centers_from_seeds(
        frames,
        seed_centers_yx,
        seed_samples,
        template_radius_px=args.template_radius_px,
        search_radius_px=args.search_radius_px,
        minimum_score=args.minimum_template_score,
    )
    track_ids = np.asarray(
        [int(track["track_id"]) for track in selected],
        dtype=np.int64,
    )
    observation_counts = np.asarray(
        [int(track["observation_count"]) for track in selected],
        dtype=np.int32,
    )
    semantic_observation_counts = np.asarray(
        [int(track["semantic_observation_count"]) for track in selected],
        dtype=np.int32,
    )
    identity_tiers = np.asarray(
        [identity_confidence_tier(track) for track in selected],
    )
    owner_radius_analysis_px = (
        0.5 * float(identity["diameter_px"]) * analysis_scale
    )
    motion_config = PollenMotionConfig.from_environment()
    resolved_motion_mode, motion_fallback_reason = resolve_owner_motion_mode(
        args.motion_mode,
        census_phase,
        motion_config,
    )
    motion_prior_yx = raw_template_yx
    point_tracker_visible = np.zeros(raw_template_yx.shape[:2], dtype=bool)
    if resolved_motion_mode == "cotracker":
        motion_prior_yx, point_tracker_visible = track_owner_centers_with_cotracker(
            frames,
            seed_centers_yx,
            seed_samples,
            motion_config,
        )
    body_certificates = certify_owner_pollen_bodies(
        frames,
        motion_prior_yx,
        seed_samples,
        owner_radius_analysis_px,
        evidence_samples=args.body_evidence_samples,
        minimum_valid_samples=args.minimum_body_valid_samples,
        boundary_gradient_threshold=args.body_gradient_threshold,
        minimum_radial_contrast=args.minimum_body_radial_contrast,
        minimum_opposite_coverage=args.minimum_body_opposite_coverage,
        minimum_visible_circumference=(
            args.minimum_body_visible_circumference
        ),
    )
    identity_certificates = reconcile_owner_identity_certificates(
        selected,
        body_certificates,
        census_phase,
        args.minimum_pregrowth_semantic_observations,
        seed_interior_darkness=_seed_interior_darkness(
            frames,
            seed_centers_yx,
            seed_samples,
            owner_radius_analysis_px,
        ),
    )
    assigned_fraction = np.zeros(len(track_ids), dtype=np.float64)
    assignment_costs: "np.ndarray | None" = None
    if args.geometric_motion_fusion:
        detections = detect_field_pollen_circles(
            frames,
            owner_radius_analysis_px,
            batch_size=args.geometric_batch_size,
            hough_threshold=args.geometric_hough_threshold,
            minimum_circle_score=args.geometric_minimum_circle_score,
        )
        _, assignments, assignment_costs, correction = (
            constrain_trajectories_to_unique_detections(
                motion_prior_yx,
                detections,
                owner_radius_analysis_px,
                DetectionSnapConfig(),
            )
        )
        correction *= semantic_anchor_guard_weights(
            source_frames,
            track_ids,
            identity["tracks"],
        )[..., None]
        template_yx = motion_prior_yx + correction
        assigned_fraction = np.asarray(
            np.mean(assignments >= 0, axis=1), dtype=np.float64
        )
        assignment_costs = np.asarray(assignment_costs, dtype=np.float64)
    else:
        template_yx = motion_prior_yx
    identity_certificates = _demote_unsupported_identity_tracks(
        selected,
        identity_certificates,
        assigned_fraction,
    )
    motion_ambiguity = _audit_owner_motion_ambiguity(
        selected,
        assigned_fraction,
        assignment_costs if args.geometric_motion_fusion else None,
        template_scores,
        point_tracker_visible
        if resolved_motion_mode == "cotracker"
        else None,
    )
    motion_ambiguity_ids_for_cache = sorted(
        int(value) for value in motion_ambiguity["ambiguous_owner_ids"]  # type: ignore[union-attr]
    )
    detection_arrays: dict[str, np.ndarray] = {}
    motion_fusion_report: dict[str, object] = {
        "enabled": args.geometric_motion_fusion,
        "requested_motion_mode": args.motion_mode,
        "resolved_motion_mode": resolved_motion_mode,
        "fallback_reason": motion_fallback_reason,
        "point_tracker_visible_fraction": (
            float(np.mean(point_tracker_visible))
            if resolved_motion_mode == "cotracker"
            else None
        ),
        "point_tracker_model": (
            motion_config.cache_identity()
            if resolved_motion_mode == "cotracker"
            else None
        ),
    }
    if args.geometric_motion_fusion:
        detections = detect_field_pollen_circles(
            frames,
            owner_radius_analysis_px,
            batch_size=args.geometric_batch_size,
            hough_threshold=args.geometric_hough_threshold,
            minimum_circle_score=args.geometric_minimum_circle_score,
        )
        _, assignments, assignment_costs, correction = (
            constrain_trajectories_to_unique_detections(
                motion_prior_yx,
                detections,
                owner_radius_analysis_px,
                DetectionSnapConfig(),
            )
        )
        correction *= semantic_anchor_guard_weights(
            source_frames,
            track_ids,
            identity["tracks"],
        )[..., None]
        template_yx = motion_prior_yx + correction
        assigned_fraction = np.mean(assignments >= 0, axis=1)
        correction_source_px = np.linalg.norm(correction, axis=2) / analysis_scale
        snap_distance = _snap_corrected_mature_anchoring(
            template_yx,
            assignments,
            detections,
            analysis_scale,
        )
        detection_arrays = {
            "detection_assignment_mask": assignments >= 0,
            "detection_assignment_cost": assignment_costs,
            "detection_correction_aligned_yx": correction,
            "mature_snap_distance_source_px": snap_distance,
            "detection_counts": np.asarray(
                [len(items) for items in detections],
                dtype=np.int32,
            ),
        }
        motion_fusion_report.update(
            median_detection_count=float(
                np.median(detection_arrays["detection_counts"])
            ),
            median_owner_assignment_fraction=float(np.median(assigned_fraction)),
            p10_owner_assignment_fraction=float(
                np.percentile(assigned_fraction, 10)
            ),
            minimum_owner_assignment_fraction=float(np.min(assigned_fraction)),
            median_applied_correction_source_px=float(
                np.median(correction_source_px)
            ),
            p90_applied_correction_source_px=float(
                np.percentile(correction_source_px, 90)
            ),
            median_mature_snap_distance_source_px=float(
                np.nanmedian(snap_distance)
            ),
            p90_mature_snap_distance_source_px=float(
                np.nanpercentile(snap_distance, 90)
            ),
            assignment_fraction_by_identity_tier={
                str(tier): {
                    "owner_count": int(np.count_nonzero(identity_tiers == tier)),
                    "median": float(
                        np.median(assigned_fraction[identity_tiers == tier])
                    ),
                    "p10": float(
                        np.percentile(
                            assigned_fraction[identity_tiers == tier],
                            10,
                        )
                    ),
                }
                for tier in ("high", "supported", "provisional")
                if np.any(identity_tiers == tier)
            },
        )
    source_centers_yx = (
        template_yx + shifts_xy[None, :, ::-1]
    ) / analysis_scale
    source_shape = (
        int(manifest["native_height"]),
        int(manifest["native_width"]),
    )
    visible_mask = source_visibility_mask(
        source_centers_yx,
        source_shape,
        0.0,
    )
    reliably_visible_mask = source_visibility_mask(
        source_centers_yx,
        source_shape,
        args.visibility_margin_source_px,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    motion_path = args.output / "owner_motion.npz"
    np.savez_compressed(
        motion_path,
        source_frames=source_frames,
        track_ids=track_ids,
        template_source_yx=source_centers_yx,
        template_aligned_yx=template_yx,
        motion_prior_source_yx=(
            motion_prior_yx + shifts_xy[None, :, ::-1]
        )
        / analysis_scale,
        motion_prior_aligned_yx=motion_prior_yx,
        resolved_owner_motion_mode=np.asarray(resolved_motion_mode),
        point_tracker_visible=point_tracker_visible,
        uncorrected_template_source_yx=(
            raw_template_yx + shifts_xy[None, :, ::-1]
        )
        / analysis_scale,
        uncorrected_template_aligned_yx=raw_template_yx,
        template_scores=template_scores,
        owner_visible_mask=visible_mask,
        owner_reliably_visible_mask=reliably_visible_mask,
        owner_identity_observation_counts=observation_counts,
        owner_semantic_observation_counts=semantic_observation_counts,
        owner_identity_tiers=identity_tiers,
        motion_ambiguous_owner_ids=np.asarray(
            list(motion_ambiguity_ids_for_cache),
            dtype=np.int64,
        ),
        owner_body_statuses=identity_certificates.statuses,
        owner_body_verified=identity_certificates.verified,
        owner_body_rejected=identity_certificates.rejected,
        owner_identity_certificate_basis=identity_certificates.bases,
        owner_pregrowth_semantic_consensus=(
            identity_certificates.pregrowth_semantic_consensus
        ),
        owner_radial_body_statuses=body_certificates.statuses,
        owner_radial_body_verified=body_certificates.verified,
        owner_radial_body_rejected=body_certificates.rejected,
        owner_body_median_radial_contrast=(
            body_certificates.median_radial_contrast
        ),
        owner_body_median_angular_boundary_fraction=(
            body_certificates.median_angular_boundary_fraction
        ),
        owner_body_median_opposite_boundary_fraction=(
            body_certificates.median_opposite_boundary_fraction
        ),
        owner_body_median_valid_angular_fraction=(
            body_certificates.median_valid_angular_fraction
        ),
        owner_body_valid_sample_counts=body_certificates.valid_sample_counts,
        owner_body_evidence_sample_indices=body_certificates.sample_indices,
        owner_body_radial_contrast_samples=(
            body_certificates.radial_contrast_samples
        ),
        owner_body_angular_boundary_samples=(
            body_certificates.angular_boundary_samples
        ),
        owner_body_opposite_boundary_samples=(
            body_certificates.opposite_boundary_samples
        ),
        owner_body_valid_angular_samples=(
            body_certificates.valid_angular_samples
        ),
        **detection_arrays,
    )
    field_run = args.output / "field_run"
    field_run.mkdir(exist_ok=True)
    counts = _write_owner_field_inputs(
        field_run,
        track_ids,
        source_frames,
        float(manifest["fps"]),
        source_centers_yx,
        visible_mask,
        reliably_visible_mask,
        observation_counts,
        semantic_observation_counts,
        identity_tiers,
        assigned_fraction,
        np.median(template_scores, axis=1),
        body_certificates,
        identity_certificates,
        source_shape,
        args.boundary_distance_source_px,
    )
    report = {
        "revision": "v29.32-snap-anchored-bootstrap",
        "identity_report": str(identity_path),
        "field_cache": str(field_cache),
        "selection": {
            "minimum_observations": minimum_observations,
            "minimum_semantic_observations": (
                minimum_semantic_observations
            ),
        },
        "motion_fusion": motion_fusion_report,
        "pollen_body_certificate": {
            "coordinate_source": (
                "unsnapped-cotracker-owner-trajectory"
                if resolved_motion_mode == "cotracker"
                else "unsnapped-semantic-seed-template-trajectory"
            ),
            "pollen_radius_analysis_px": owner_radius_analysis_px,
            "evidence_samples": args.body_evidence_samples,
            "minimum_valid_samples": args.minimum_body_valid_samples,
            "boundary_gradient_threshold": args.body_gradient_threshold,
            "minimum_radial_contrast": args.minimum_body_radial_contrast,
            "minimum_opposite_coverage": (
                args.minimum_body_opposite_coverage
            ),
            "minimum_visible_circumference": (
                args.minimum_body_visible_circumference
            ),
        },
        "owner_identity_reconciliation": {
            "census_phase": census_phase,
            "minimum_pregrowth_semantic_observations": (
                args.minimum_pregrowth_semantic_observations
            ),
            "rule": (
                "persistent learned-mask identity from an explicitly declared "
                "pre-growth census supersedes low-resolution radial shape; "
                "contested cluster identities and unsupported contested owners "
                "stay diagnostic; otherwise the radial body certificate "
                "remains binding"
            ),
            "certificate_basis_counts": {
                str(basis): int(np.count_nonzero(identity_certificates.bases == basis))
                for basis in np.unique(identity_certificates.bases)
            },
        },
        "owner_motion_ambiguity": motion_ambiguity,
        "visibility": {
            "reliable_margin_source_px": args.visibility_margin_source_px,
            "owner_count_visible_at_first_sample": int(
                np.count_nonzero(visible_mask[:, 0])
            ),
            "owner_count_visible_at_last_sample": int(
                np.count_nonzero(visible_mask[:, -1])
            ),
            "owner_count_visible_for_all_samples": int(
                np.count_nonzero(np.all(visible_mask, axis=1))
            ),
            "owner_count_reliably_visible_for_all_samples": int(
                np.count_nonzero(np.all(reliably_visible_mask, axis=1))
            ),
        },
        **counts,
        "median_template_score": float(np.median(template_scores)),
        "p10_template_score": float(np.percentile(template_scores, 10)),
        "owner_motion_cache": str(motion_path.resolve()),
        "field_run": str(field_run.resolve()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
