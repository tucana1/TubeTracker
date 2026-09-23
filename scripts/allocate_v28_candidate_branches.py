#!/usr/bin/env python3
"""Allocate distinct pollen-tube branches from retained v28 alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from tubetracker.field_arbitration import (
    CandidateBranchEvidence,
    allocate_distinct_candidate_branches,
    candidate_duplicate_fraction,
)


REVISION = "v28.6.1-minimum-reassignment-branch-allocation-prototype"


def parse_args() -> argparse.Namespace:
    """Parse candidate run, field cache, conflict components, and output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--field-cache", type=Path, required=True)
    parser.add_argument("--components", required=True, help="Semicolon-separated ID groups.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=150)
    return parser.parse_args()


def parse_components(value: str) -> list[list[int]]:
    """Return unique multi-owner conflict components from compact text."""

    components = []
    seen = set()
    for group in value.split(";"):
        owners = list(dict.fromkeys(int(item.strip()) for item in group.split(",")))
        if len(owners) < 2 or any(owner < 1 for owner in owners):
            raise ValueError("each component requires at least two positive owner IDs")
        overlap = seen & set(owners)
        if overlap:
            raise ValueError(f"owners occur in multiple components: {sorted(overlap)}")
        seen.update(owners)
        components.append(owners)
    return components


def load_owner_candidates(
    candidate_run: Path,
    owner_id: int,
) -> tuple[list[CandidateBranchEvidence], dict[str, np.ndarray]]:
    """Load one owner's safe alternative archive and rendering arrays."""

    path = candidate_run / f"P{owner_id:02d}" / f"candidate_histories_P{owner_id:02d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "labels",
        "audit_sample_indices",
        "aligned_yx",
        "point_counts",
        "lengths_px",
        "accepted",
        "onset_sample_indices",
        "causal_keys",
        "geometry_eligible",
        "contact_censored",
        "near_foreign_owner",
        "selected_index",
        "owner_centers_aligned_yx",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"P{owner_id:02d} archive is missing: {sorted(missing)}")
    selected = int(arrays["selected_index"])
    candidates = []
    for candidate_index, label in enumerate(arrays["labels"]):
        curves = {}
        for position, sample in enumerate(arrays["audit_sample_indices"]):
            count = int(arrays["point_counts"][candidate_index, position])
            if arrays["accepted"][candidate_index, position] and count >= 2:
                curves[int(sample)] = np.asarray(
                    arrays["aligned_yx"][candidate_index, position, :count],
                    dtype=np.float64,
                )
        onset = int(arrays["onset_sample_indices"][candidate_index])
        candidates.append(
            CandidateBranchEvidence(
                owner_id=owner_id,
                candidate_index=candidate_index,
                label=str(label),
                causal_key=tuple(
                    float(value) for value in arrays["causal_keys"][candidate_index]
                ),
                final_length_px=float(arrays["lengths_px"][candidate_index, -1]),
                onset_sample=None if onset < 0 else onset,
                geometry_eligible=bool(
                    arrays["geometry_eligible"][candidate_index]
                ),
                contact_censored=bool(arrays["contact_censored"][candidate_index]),
                near_foreign_owner=bool(
                    arrays["near_foreign_owner"][candidate_index]
                ),
                selected_independently=candidate_index == selected,
                curves_by_sample=curves,
            )
        )
    return candidates, arrays


def candidate_record(candidate: CandidateBranchEvidence | None) -> dict | None:
    """Convert an allocated branch into JSON-safe evidence fields."""

    if candidate is None:
        return None
    return {
        "candidate_index": candidate.candidate_index,
        "label": candidate.label,
        "onset_sample": candidate.onset_sample,
        "final_length_px": candidate.final_length_px,
        "causal_key": list(candidate.causal_key),
        "contact_censored": candidate.contact_censored,
        "near_foreign_owner": candidate.near_foreign_owner,
        "selected_independently": candidate.selected_independently,
    }


def audit_samples(schedule: np.ndarray, onset: int | None) -> np.ndarray:
    """Choose six sparse positions around one candidate's inferred emergence."""

    if onset is None:
        return np.rint(np.linspace(0, len(schedule) - 1, 6)).astype(int)
    onset_position = int(np.searchsorted(schedule, onset))
    final = len(schedule) - 1
    return np.asarray(
        (
            0,
            max(0, onset_position - 1),
            onset_position,
            min(final, onset_position + 1),
            round((onset_position + final) / 2),
            final,
        ),
        dtype=int,
    )


def centered_crop(image: np.ndarray, center_yx: np.ndarray, size: int) -> np.ndarray:
    """Crop one fixed owner view with padding at image boundaries."""

    half = size // 2
    center = np.rint(center_yx).astype(int)
    padded = cv.copyMakeBorder(
        image,
        half,
        half,
        half,
        half,
        cv.BORDER_CONSTANT,
        value=(18, 18, 18),
    )
    return padded[center[0] : center[0] + size, center[1] : center[1] + size]


def render_allocation_audit(
    output: Path,
    aligned_frames: np.ndarray,
    allocation: dict[int, CandidateBranchEvidence | None],
    archives: dict[int, dict[str, np.ndarray]],
    crop_size: int,
) -> Path:
    """Render owner-following lifecycle rows for the globally allocated branches."""

    rows = []
    for owner_id in sorted(allocation):
        candidate = allocation[owner_id]
        arrays = archives[owner_id]
        schedule = arrays["audit_sample_indices"].astype(int)
        positions = audit_samples(
            schedule,
            None if candidate is None else candidate.onset_sample,
        )
        tiles = []
        for position in positions:
            sample = int(schedule[position])
            image = cv.cvtColor(
                np.asarray(aligned_frames[sample]),
                cv.COLOR_GRAY2BGR,
            )
            center = arrays["owner_centers_aligned_yx"][position]
            if candidate is not None:
                path = candidate.curves_by_sample.get(sample)
                if path is not None:
                    points = np.rint(path[:, ::-1]).astype(np.int32)
                    cv.polylines(image, [points], False, (220, 30, 220), 2, cv.LINE_AA)
                    cv.circle(image, tuple(points[-1]), 3, (20, 240, 20), -1, cv.LINE_AA)
            cv.circle(
                image,
                tuple(np.rint(center[::-1]).astype(int)),
                10,
                (255, 140, 20),
                2,
                cv.LINE_AA,
            )
            crop = centered_crop(image, center, crop_size)
            cv.putText(
                crop,
                f"s{sample}",
                (5, crop_size - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (245, 245, 245),
                1,
                cv.LINE_AA,
            )
            tiles.append(crop)
        body = np.hstack(tiles)
        header = np.full((34, body.shape[1], 3), 18, dtype=np.uint8)
        label = "WITHHELD" if candidate is None else candidate.label
        disposition = (
            "retained"
            if candidate is not None and candidate.selected_independently
            else "reassigned"
            if candidate is not None
            else "no distinct certified branch"
        )
        cv.putText(
            header,
            f"P{owner_id:02d} | {label} | {disposition}",
            (6, 23),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv.LINE_AA,
        )
        rows.append(np.vstack((header, body)))
    canvas = np.vstack(rows)
    path = output / "branch_allocation_audit.jpg"
    cv.imwrite(str(path), canvas, [cv.IMWRITE_JPEG_QUALITY, 95])
    return path


def main() -> int:
    """Solve exact conflict components and write evidence plus a visual audit."""

    args = parse_args()
    if args.crop_size < 64:
        raise ValueError("crop size must be at least 64 pixels")
    candidate_run = args.candidate_run.expanduser().resolve()
    field_cache = args.field_cache.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    components = parse_components(args.components)
    all_candidates = {}
    archives = {}
    for owner_id in sorted({owner for component in components for owner in component}):
        all_candidates[owner_id], archives[owner_id] = load_owner_candidates(
            candidate_run,
            owner_id,
        )

    allocation = {}
    component_reports = []
    for component in components:
        candidates = {owner: all_candidates[owner] for owner in component}
        selected = allocate_distinct_candidate_branches(candidates)
        allocation.update(selected)
        allocated = [candidate for candidate in selected.values() if candidate is not None]
        component_reports.append(
            {
                "owner_ids": component,
                "assignments": {
                    str(owner): candidate_record(selected[owner]) for owner in component
                },
                "allocated_owner_count": len(allocated),
                "maximum_allocated_duplicate_fraction": max(
                    (
                        candidate_duplicate_fraction(first, second)
                        for index, first in enumerate(allocated)
                        for second in allocated[index + 1 :]
                    ),
                    default=0.0,
                ),
            }
        )

    aligned_frames = np.load(field_cache / "aligned_frames.npy", mmap_mode="r")
    audit = render_allocation_audit(
        output,
        aligned_frames,
        allocation,
        archives,
        args.crop_size,
    )
    mature_seed_overrides = {
        str(owner): candidate.label
        for owner, candidate in sorted(allocation.items())
        if candidate is not None and not candidate.selected_independently
    }
    override_path = output / "mature_seed_overrides.json"
    override_path.write_text(json.dumps(mature_seed_overrides, indent=2) + "\n")
    report = {
        "revision": REVISION,
        "candidate_run": str(candidate_run),
        "field_cache": str(field_cache),
        "components": component_reports,
        "assigned_owner_ids": sorted(
            owner for owner, candidate in allocation.items() if candidate is not None
        ),
        "withheld_owner_ids": sorted(
            owner for owner, candidate in allocation.items() if candidate is None
        ),
        "audit": str(audit),
        "mature_seed_overrides": str(override_path),
        "scope": (
            "Exact allocation among independently generated, certified alternatives; "
            "visual audit remains required before full-field promotion."
        ),
    }
    report_path = output / "branch_allocation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
