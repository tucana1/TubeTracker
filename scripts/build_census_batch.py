"""Build census-tile tasks at explicit frames/positions (rev7 batch).

A census tile declares a complete 320px box: the annotator marks EVERY
visible tip + ball inside, then Tile-complete. Writes a FRESH dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# (movie, frame, x, y): spread across field quadrants + times, away
# from previously censused tiles.
TILES = [
    ("ld", 12000, 320.0, 256.0),
    ("ld", 24000, 960.0, 768.0),
    ("ld", 36000, 320.0, 768.0),
    ("ld", 52500, 960.0, 256.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="v30c")
    ap.add_argument("--actor", default="v30census")
    a = ap.parse_args()

    from tubetracker.annotation_store import AnnotationStore

    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return 1
    proj.mkdir(parents=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        for i, (movie, fr, x, y) in enumerate(TILES):
            uuid = f"{a.tag}-{i:03d}"
            store.save("task", uuid, {
                "uuid": uuid, "owner_uuid": "unassigned",
                "query_frames": [int(fr)], "task_type": "census",
                "movie": movie, "focus_xy": [float(x), float(y)],
                "completed": False, "stratum": f"{a.tag}-census",
                "priority": 10.0,
                "why": "Exhaustive tile: mark EVERY visible tip + ball "
                       "in the 320px box around this point, then "
                       "Tile-complete. Skip nothing inside the box.",
            }, actor=a.actor)
    finally:
        store.close()
    print(f"{a.tag}: +{len(TILES)} census tasks -> {proj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
