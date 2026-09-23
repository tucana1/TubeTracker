"""Build a compact visual audit sheet for every completed v23 field owner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse a completed batch directory and audit-sheet output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--consistency-report", type=Path)
    return parser.parse_args()


def final_video_frame(path: Path) -> np.ndarray:
    """Decode the final rendered frame from one owner review."""
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    capture.set(cv.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode {path}")
    return frame


def owner_tile(
    owner: dict,
    batch_dir: Path,
    size: tuple[int, int],
    conflict_ids: set[int],
) -> np.ndarray:
    """Render one final owner view with its promotion diagnostics."""
    owner_id = int(owner["owner_track_id"])
    owner_dir = batch_dir / owner["directory"]
    report = json.loads((owner_dir / "report.json").read_text())
    frame = final_video_frame(owner_dir / report["artifacts"]["review_video"])
    image_height = size[1] - 50
    tile = np.full((size[1], size[0], 3), 18, dtype=np.uint8)
    tile[:image_height] = cv.resize(
        frame, (size[0], image_height), interpolation=cv.INTER_AREA
    )
    conflict = owner_id in conflict_ids
    accepted = bool(report["accepted"]) and not conflict
    border = (
        (0, 175, 255)
        if conflict
        else (40, 210, 70)
        if accepted
        else (40, 80, 230)
    )
    cv.rectangle(tile, (1, 1), (size[0] - 2, size[1] - 2), border, 4)
    selected = report.get("selected_lengths_px", [])
    growth = report.get("net_growth_px")
    if growth is None and len(selected) >= 2:
        growth = float(selected[-1] - selected[0])
    label = f"P{owner_id:02d} {'CONFLICT' if conflict else 'ACCEPT' if accepted else 'WITHHOLD'}"
    detail = "shared-path ownership" if conflict else report["reason"]
    if growth is not None:
        detail += f" | growth {growth:.0f}px"
    for text, y, scale in (
        (label, image_height + 20, 0.56),
        (detail, image_height + 41, 0.38),
    ):
        cv.putText(
            tile,
            text,
            (8, y),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            tile,
            text,
            (8, y),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            border,
            1,
            cv.LINE_AA,
        )
    return tile


def main() -> None:
    """Write a tiled final-state audit for all completed owner runs."""
    args = parse_args()
    batch = json.loads(args.batch_report.read_text())
    owners = sorted(batch["owners"], key=lambda item: item["owner_track_id"])
    if not owners:
        raise RuntimeError("the batch report contains no completed owner runs")
    conflict_ids: set[int] = set()
    if args.consistency_report:
        consistency = json.loads(args.consistency_report.read_text())
        conflict_ids.update(consistency.get("duplicate_claim_owner_ids", []))
        conflict_ids.update(consistency.get("foreign_endpoint_owner_ids", []))
    tile_size = (336, 250)
    columns = max(1, args.columns)
    rows = math.ceil(len(owners) / columns)
    sheet = np.full(
        (rows * tile_size[1], columns * tile_size[0], 3),
        20,
        dtype=np.uint8,
    )
    for index, owner in enumerate(owners):
        tile = owner_tile(
            owner,
            args.batch_report.parent,
            tile_size,
            conflict_ids,
        )
        row, column = divmod(index, columns)
        y0, x0 = row * tile_size[1], column * tile_size[0]
        sheet[y0 : y0 + tile_size[1], x0 : x0 + tile_size[0]] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(args.output), sheet, [cv.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"could not write {args.output}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "owner_count": len(owners),
                "accepted_count": sum(
                    item["accepted"]
                    and item["owner_track_id"] not in conflict_ids
                    for item in owners
                ),
                "field_conflict_count": len(conflict_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
