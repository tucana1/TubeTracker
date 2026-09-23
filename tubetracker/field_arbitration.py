"""Resolve duplicate pollen-tube ownership claims across a completed field."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


AUTOMATIC_MEASUREMENT_STATUSES = {
    "usable",
    "measured",
    "contact_censored",
    "left_censored",
    "boundary_censored",
}


@dataclass(frozen=True)
class FieldOwnerEvidence:
    """Store one owner's status, causal evidence, and sampled centerlines."""

    track_id: int
    status: str
    event_certified: bool
    onset_sample: int | None
    curves_by_sample: dict[int, np.ndarray]
    near_foreign_owner: bool = False


@dataclass(frozen=True)
class CandidateBranchEvidence:
    """Store one complete owner-specific branch candidate through sparse time."""

    owner_id: int
    candidate_index: int
    label: str
    causal_key: tuple[float, float, float, float]
    final_length_px: float
    onset_sample: int | None
    geometry_eligible: bool
    contact_censored: bool
    near_foreign_owner: bool
    selected_independently: bool
    curves_by_sample: dict[int, np.ndarray]

    @property
    def event_certified(self) -> bool:
        """Return whether this candidate has persistent geometric growth evidence."""

        return (
            self.geometry_eligible
            and self.onset_sample is not None
            and self.causal_key[0] > 0.5
            and np.isfinite(self.causal_key[1])
            and self.causal_key[1] >= 0.1
        )

    @property
    def admissible(self) -> bool:
        """Reject ordinary candidates whose owner path crosses another pollen."""

        return self.event_certified and (
            self.contact_censored or not self.near_foreign_owner
        )


def curve_arclength(path_yx: np.ndarray) -> np.ndarray:
    """Return cumulative arclength along one row-column centerline."""

    path = np.asarray(path_yx, dtype=np.float64)
    if len(path) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )


def distal_curve(path_yx: np.ndarray, root_exclusion_px: float) -> np.ndarray:
    """Remove the owner-specific pollen-rim prefix before overlap scoring."""

    path = np.asarray(path_yx, dtype=np.float64)
    distal = path[curve_arclength(path) >= root_exclusion_px]
    return distal if len(distal) >= 2 else path


def directional_path_overlap(
    first_yx: np.ndarray,
    second_yx: np.ndarray,
    distance_px: float,
    root_exclusion_px: float,
) -> tuple[float, float]:
    """Measure how much of each distal path lies on the other path."""

    first = distal_curve(first_yx, root_exclusion_px)
    second = distal_curve(second_yx, root_exclusion_px)
    if len(first) < 2 or len(second) < 2:
        return 0.0, 0.0
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    return (
        float(np.mean(second_tree.query(first, k=1)[0] <= distance_px)),
        float(np.mean(first_tree.query(second, k=1)[0] <= distance_px)),
    )


def first_sustained_overlap_index(
    path_yx: np.ndarray,
    reference_yx: np.ndarray,
    *,
    distance_px: float,
    window_points: int = 5,
    required_points: int = 3,
    minimum_tangent_alignment: float = 0.75,
) -> int | None:
    """Return where a path first begins following another sustained branch.

    Nearby points count only when their local centerline directions agree.  This
    separates a shared tail from an incidental perpendicular crossing.
    """

    path = np.asarray(path_yx, dtype=np.float64)
    reference = np.asarray(reference_yx, dtype=np.float64)
    if (
        path.ndim != 2
        or reference.ndim != 2
        or path.shape[1:] != (2,)
        or reference.shape[1:] != (2,)
    ):
        raise ValueError("paths must have shape (points, 2)")
    if (
        distance_px <= 0.0
        or window_points < 1
        or not 1 <= required_points <= window_points
        or not 0.0 <= minimum_tangent_alignment <= 1.0
    ):
        raise ValueError("overlap constraints must be positive and consistent")
    if len(path) < required_points or len(reference) < 2:
        return None
    nearby = _aligned_overlap_mask(
        path,
        reference,
        distance_px=distance_px,
        minimum_tangent_alignment=minimum_tangent_alignment,
    )
    for index in range(len(nearby) - required_points + 1):
        window = nearby[index : index + window_points]
        if np.count_nonzero(window) >= required_points:
            return index + int(np.flatnonzero(window)[0])
    return None


def sustained_branch_capture_index(
    path_yx: np.ndarray,
    reference_yx: np.ndarray,
    *,
    distance_px: float,
    minimum_shared_length_px: float,
    window_points: int = 7,
    required_points: int = 5,
    maximum_gap_points: int = 2,
    minimum_tangent_alignment: float = 0.75,
) -> int | None:
    """Find where a path begins following a substantial foreign branch."""

    path = np.asarray(path_yx, dtype=np.float64)
    reference = np.asarray(reference_yx, dtype=np.float64)
    first = first_sustained_overlap_index(
        path,
        reference,
        distance_px=distance_px,
        window_points=window_points,
        required_points=required_points,
        minimum_tangent_alignment=minimum_tangent_alignment,
    )
    if first is None:
        return None
    if minimum_shared_length_px <= 0.0 or maximum_gap_points < 0:
        raise ValueError("branch-capture length must be positive and gaps nonnegative")
    nearby = _aligned_overlap_mask(
        path,
        reference,
        distance_px=distance_px,
        minimum_tangent_alignment=minimum_tangent_alignment,
    )
    last = first
    gap = 0
    for index in range(first, len(path)):
        if nearby[index]:
            last = index
            gap = 0
        else:
            gap += 1
            if gap > maximum_gap_points:
                break
    shared_length = float(
        np.sum(np.linalg.norm(np.diff(path[first : last + 1], axis=0), axis=1))
    )
    return first if shared_length >= minimum_shared_length_px else None


def _aligned_overlap_mask(
    path_yx: np.ndarray,
    reference_yx: np.ndarray,
    *,
    distance_px: float,
    minimum_tangent_alignment: float,
) -> np.ndarray:
    """Mark path points that lie on the same unoriented lane as a reference."""

    path = np.asarray(path_yx, dtype=np.float64)
    reference = np.asarray(reference_yx, dtype=np.float64)
    distances, reference_indices = cKDTree(reference).query(path, k=1)

    def unit_tangents(points: np.ndarray) -> np.ndarray:
        tangents = np.gradient(points, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / np.maximum(norms, 1e-12)

    path_tangents = unit_tangents(path)
    reference_tangents = unit_tangents(reference)
    alignment = np.abs(
        np.sum(path_tangents * reference_tangents[reference_indices], axis=1)
    )
    return (distances <= distance_px) & (
        alignment >= minimum_tangent_alignment
    )


def candidate_duplicate_fraction(
    first: CandidateBranchEvidence,
    second: CandidateBranchEvidence,
    *,
    distance_px: float = 6.0,
    root_exclusion_px: float = 20.0,
    minimum_directional_overlap: float = 0.50,
    trailing_samples: int = 10,
    minimum_common_samples: int = 3,
) -> float:
    """Measure sustained late overlap between two sparse branch candidates."""

    common = sorted(
        set(first.curves_by_sample) & set(second.curves_by_sample)
    )[-trailing_samples:]
    if len(common) < minimum_common_samples:
        return 0.0
    overlaps = np.asarray(
        [
            directional_path_overlap(
                first.curves_by_sample[sample],
                second.curves_by_sample[sample],
                distance_px,
                root_exclusion_px,
            )
            for sample in common
        ],
        dtype=np.float64,
    )
    return float(np.mean(np.min(overlaps, axis=1) >= minimum_directional_overlap))


def allocate_distinct_candidate_branches(
    candidates_by_owner: dict[int, list[CandidateBranchEvidence]],
    *,
    minimum_duplicate_fraction: float = 0.60,
) -> dict[int, CandidateBranchEvidence | None]:
    """Find the exact highest-evidence set of distinct owner branch candidates."""

    owner_ids = sorted(candidates_by_owner)
    options = [
        [None]
        + [candidate for candidate in candidates_by_owner[owner] if candidate.admissible]
        for owner in owner_ids
    ]
    maximum_length = {
        owner: max(
            (candidate.final_length_px for candidate in options[index][1:]),
            default=1.0,
        )
        for index, owner in enumerate(owner_ids)
    }
    best_assignment = None
    best_objective = None
    for assignment in product(*options):
        selected = [candidate for candidate in assignment if candidate is not None]
        incompatible = any(
            candidate_duplicate_fraction(first, second)
            >= minimum_duplicate_fraction
            for first_index, first in enumerate(selected)
            for second in selected[first_index + 1 :]
        )
        if incompatible:
            continue
        complete_growth = sum(
            candidate.causal_key[1]
            * np.sqrt(
                np.clip(
                    candidate.final_length_px / maximum_length[candidate.owner_id],
                    0.0,
                    1.0,
                )
            )
            for candidate in selected
        )
        retained_independent = sum(
            candidate.selected_independently for candidate in selected
        )
        objective = (
            len(selected),
            retained_independent,
            float(complete_growth),
            float(sum(candidate.causal_key[2] for candidate in selected)),
            float(sum(candidate.causal_key[3] for candidate in selected)),
            -sum(candidate.candidate_index for candidate in selected),
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_assignment = assignment
    if best_assignment is None:
        return {owner: None for owner in owner_ids}
    return dict(zip(owner_ids, best_assignment))


def sustained_duplicate_claims(
    owners: list[FieldOwnerEvidence],
    *,
    distance_px: float = 6.0,
    root_exclusion_px: float = 20.0,
    minimum_directional_overlap: float = 0.50,
    minimum_sustained_fraction: float = 0.60,
    trailing_samples: int = 40,
    minimum_common_samples: int = 12,
    terminal_window_samples: int = 12,
    minimum_terminal_tail_samples: int = 8,
) -> list[dict]:
    """Find owner pairs that repeatedly claim the same distal centerline.

    A claim may be established either by broad overlap through the trailing
    history or by a directionally aligned shared tail that persists at the end.
    The latter catches late branch adoption without promoting brief crossings.
    """

    claims = []
    for first_index, first in enumerate(owners):
        for second in owners[first_index + 1 :]:
            common = sorted(
                set(first.curves_by_sample) & set(second.curves_by_sample)
            )[-trailing_samples:]
            if len(common) < minimum_common_samples:
                continue
            overlaps = np.asarray(
                [
                    directional_path_overlap(
                        first.curves_by_sample[sample],
                        second.curves_by_sample[sample],
                        distance_px,
                        root_exclusion_px,
                    )
                    for sample in common
                ],
                dtype=np.float64,
            )
            shared = np.min(overlaps, axis=1)
            sustained_fraction = float(
                np.mean(shared >= minimum_directional_overlap)
            )
            terminal_common = common[-terminal_window_samples:]
            terminal_shared_tail = np.asarray(
                [
                    first_sustained_overlap_index(
                        first.curves_by_sample[sample],
                        second.curves_by_sample[sample],
                        distance_px=distance_px,
                    )
                    is not None
                    and first_sustained_overlap_index(
                        second.curves_by_sample[sample],
                        first.curves_by_sample[sample],
                        distance_px=distance_px,
                    )
                    is not None
                    for sample in terminal_common
                ],
                dtype=bool,
            )
            terminal_tail_count = int(np.count_nonzero(terminal_shared_tail))
            terminal_tail_persistent = bool(
                len(terminal_common) >= minimum_terminal_tail_samples
                and terminal_tail_count >= minimum_terminal_tail_samples
                and np.all(
                    terminal_shared_tail[
                        -min(4, minimum_terminal_tail_samples) :
                    ]
                )
            )
            if (
                sustained_fraction < minimum_sustained_fraction
                and not terminal_tail_persistent
            ):
                continue
            claims.append(
                {
                    "first_owner": first.track_id,
                    "second_owner": second.track_id,
                    "common_sample_count": len(common),
                    "sustained_fraction": sustained_fraction,
                    "median_first_overlap": float(np.median(overlaps[:, 0])),
                    "median_second_overlap": float(np.median(overlaps[:, 1])),
                    "terminal_shared_tail_count": terminal_tail_count,
                    "terminal_shared_tail_samples": len(terminal_common),
                    "terminal_shared_tail_persistent": terminal_tail_persistent,
                    "distance_px": distance_px,
                    "root_exclusion_px": root_exclusion_px,
                }
            )
    return claims


def resolve_ownership_components(
    owners: list[FieldOwnerEvidence],
    claims: list[dict],
) -> list[dict]:
    """Resolve a duplicate component only when one owner has unique causal birth."""

    owner_by_id = {owner.track_id: owner for owner in owners}
    parent = {track_id: track_id for track_id in owner_by_id}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    claimed_ids = set()
    for claim in claims:
        first, second = int(claim["first_owner"]), int(claim["second_owner"])
        union(first, second)
        claimed_ids.update((first, second))

    components: dict[int, list[int]] = {}
    for track_id in claimed_ids:
        components.setdefault(find(track_id), []).append(track_id)

    resolutions = []
    for component in sorted(components.values(), key=min):
        component = sorted(component)
        causal = [
            track_id
            for track_id in component
            if owner_by_id[track_id].status in AUTOMATIC_MEASUREMENT_STATUSES
            and owner_by_id[track_id].event_certified
            and owner_by_id[track_id].onset_sample is not None
            and owner_by_id[track_id].onset_sample > 0
        ]
        unobstructed = [
            track_id
            for track_id in causal
            if not owner_by_id[track_id].near_foreign_owner
        ]
        uniquely_unobstructed = len(causal) > 1 and len(unobstructed) == 1
        winner = (
            unobstructed[0]
            if uniquely_unobstructed
            else causal[0] if len(causal) == 1 else None
        )
        conflicts = (
            component
            if winner is None
            else [track_id for track_id in component if track_id != winner]
        )
        resolutions.append(
            {
                "owner_ids": component,
                "resolution": (
                    "unique-unobstructed-causal-root"
                    if uniquely_unobstructed
                    else "unique-causal-root"
                    if winner is not None
                    else "ambiguous-owner"
                ),
                "winner_owner_id": winner,
                "conflict_owner_ids": conflicts,
            }
        )
    return resolutions


def load_centerlines(path: Path) -> dict[int, np.ndarray]:
    """Load source-coordinate centerlines grouped by sample index."""

    grouped: dict[int, list[tuple[int, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["sample_index"]), []).append(
                (
                    int(row["point_index"]),
                    float(row["source_y_px"]),
                    float(row["source_x_px"]),
                )
            )
    return {
        sample: np.asarray(
            [(y, x) for _, y, x in sorted(points)], dtype=np.float64
        )
        for sample, points in grouped.items()
    }


def append_field_status(
    source: Path,
    output: Path,
    field_statuses: dict[int, str],
) -> int:
    """Copy one aggregate CSV while adding the post-arbitration field status."""

    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "pollen_id" not in fieldnames:
        raise ValueError(f"missing pollen_id column in {source}")
    if "field_status" not in fieldnames:
        fieldnames.append("field_status")
    for row in rows:
        row["field_status"] = field_statuses[int(row["pollen_id"])]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def arbitrate_validation_directory(validation_dir: Path) -> Path:
    """Create field-safe statuses and CSVs from completed per-owner results."""

    validation_dir = validation_dir.expanduser().resolve()
    raw_manifest = json.loads(
        (validation_dir / "validation_manifest.json").read_text()
    )
    raw_statuses = {
        int(key): value
        for key, value in raw_manifest["status_by_track_id"].items()
    }
    measurement_statuses: dict[int, str] = {}
    owners = []
    for track_id in raw_manifest["completed_track_ids"]:
        track_id = int(track_id)
        owner_dir = validation_dir / f"P{track_id:02d}"
        report = json.loads((owner_dir / "report.json").read_text())
        diagnostics = report["worldsheet_diagnostics"][str(track_id)]
        measurement_statuses[track_id] = diagnostics["measurement_status"]
        owners.append(
            FieldOwnerEvidence(
                track_id=track_id,
                status=measurement_statuses[track_id],
                event_certified=bool(diagnostics.get("event_certified", False)),
                onset_sample=diagnostics.get("first_active_sample"),
                curves_by_sample=load_centerlines(owner_dir / "centerlines.csv"),
                near_foreign_owner=bool(
                    diagnostics.get("near_foreign_owner", False)
                ),
            )
        )

    claims = sustained_duplicate_claims(owners)
    resolutions = resolve_ownership_components(owners, claims)
    conflict_ids = {
        track_id
        for resolution in resolutions
        for track_id in resolution["conflict_owner_ids"]
        if measurement_statuses[track_id] in AUTOMATIC_MEASUREMENT_STATUSES
    }
    field_statuses = dict(measurement_statuses)
    for track_id in conflict_ids:
        field_statuses[track_id] = "ownership_conflict"

    row_counts = {}
    for filename in ("summary.csv", "measurements.csv", "centerlines.csv"):
        row_counts[f"field_{filename}"] = append_field_status(
            validation_dir / filename,
            validation_dir / f"field_{filename}",
            field_statuses,
        )
    report = {
        "revision": "v28.2-causal-path-ownership-arbitration",
        "source_validation_manifest": str(
            validation_dir / "validation_manifest.json"
        ),
        "completed_track_ids": raw_manifest["completed_track_ids"],
        "raw_status_by_track_id": {
            str(key): value for key, value in sorted(raw_statuses.items())
        },
        "measurement_status_by_track_id": {
            str(key): value for key, value in sorted(measurement_statuses.items())
        },
        "field_status_by_track_id": {
            str(key): value for key, value in sorted(field_statuses.items())
        },
        "field_status_counts": dict(
            sorted(Counter(field_statuses.values()).items())
        ),
        "ownership_conflict_ids": sorted(conflict_ids),
        "duplicate_path_claims": claims,
        "ownership_resolutions": resolutions,
        "field_csv_rows": row_counts,
        "scope": (
            "Field-level ownership arbitration retains every raw coordinate while "
            "excluding unresolved duplicate tube claims from automatic measurements."
        ),
    }
    output = validation_dir / "field_validation_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output
