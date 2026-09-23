"""Audit accepted v23 owner paths for cross-owner conflicts at the field level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from field_review import load_field_owners, owner_center, owner_world_curve


def parse_args() -> argparse.Namespace:
    """Parse identity, batch, and field-consistency inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-distance", type=float, default=5.0)
    parser.add_argument("--overlap-fraction", type=float, default=0.55)
    parser.add_argument("--foreign-pollen-distance", type=float, default=18.0)
    return parser.parse_args()


def distal_curve(path_yx: np.ndarray, root_exclusion_px: float = 20.0) -> np.ndarray:
    """Remove the owner-rim segment before cross-owner overlap scoring."""
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path_yx, axis=0), axis=1)))
    )
    distal = path_yx[arc >= root_exclusion_px]
    return distal if len(distal) >= 2 else path_yx


def path_overlap(
    first_yx: np.ndarray,
    second_yx: np.ndarray,
    distance_px: float,
) -> tuple[float, float]:
    """Return directional fractions of two paths lying on the same pixels."""
    distances = cdist(distal_curve(first_yx), distal_curve(second_yx))
    return (
        float(np.mean(np.min(distances, axis=1) <= distance_px)),
        float(np.mean(np.min(distances, axis=0) <= distance_px)),
    )


def main() -> None:
    """Write duplicate-claim and foreign-endpoint diagnostics for the field."""
    args = parse_args()
    identity = json.loads(args.identity_report.read_text())
    owners = load_field_owners(identity, args.batch_report)
    accepted = [owner for owner in owners if owner.centerlines]
    source_frame = int(identity["source_frames"][-1])
    curves = {
        owner.track_id: owner_world_curve(owner, source_frame)
        for owner in accepted
    }
    centers = {
        owner.track_id: owner_center(owner.track, source_frame)
        for owner in owners
    }

    overlaps = []
    for first_index, first in enumerate(accepted):
        first_curve = curves[first.track_id]
        if first_curve is None:
            continue
        for second in accepted[first_index + 1 :]:
            second_curve = curves[second.track_id]
            if second_curve is None:
                continue
            first_fraction, second_fraction = path_overlap(
                first_curve,
                second_curve,
                args.overlap_distance,
            )
            if min(first_fraction, second_fraction) >= args.overlap_fraction:
                overlaps.append(
                    {
                        "first_owner": first.track_id,
                        "second_owner": second.track_id,
                        "first_overlap_fraction": first_fraction,
                        "second_overlap_fraction": second_fraction,
                        "root_distance_px": float(
                            np.linalg.norm(
                                centers[first.track_id] - centers[second.track_id]
                            )
                        ),
                    }
                )

    endpoint_contacts = []
    for owner in accepted:
        curve = curves[owner.track_id]
        if curve is None:
            continue
        foreign_id, distance = min(
            (
                (track_id, float(np.linalg.norm(curve[-1] - center)))
                for track_id, center in centers.items()
                if track_id != owner.track_id
            ),
            key=lambda item: item[1],
        )
        if distance <= args.foreign_pollen_distance:
            endpoint_contacts.append(
                {
                    "owner_track_id": owner.track_id,
                    "foreign_owner_track_id": foreign_id,
                    "endpoint_distance_px": distance,
                }
            )

    report = {
        "prototype": "v23_owner_conditioned_video_graph",
        "stage": "field-consistency-audit",
        "source_frame": source_frame,
        "confirmed_owner_count": len(owners),
        "accepted_owner_count": len(accepted),
        "duplicate_path_claims": overlaps,
        "foreign_pollen_endpoint_contacts": endpoint_contacts,
        "duplicate_claim_owner_ids": sorted(
            {
                item[key]
                for item in overlaps
                for key in ("first_owner", "second_owner")
            }
        ),
        "foreign_endpoint_owner_ids": sorted(
            {item["owner_track_id"] for item in endpoint_contacts}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
