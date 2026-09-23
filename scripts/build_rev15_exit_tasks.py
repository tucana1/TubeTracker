"""rev15 task 2: true-exit annotation batch for corrected 0003/3756.

One body_mask task per grain at frame 51270 (both grains have owned
tube paint there under corrected ownership): paint the tube that leaves
the MAGENTA-ringed grain starting AT its rim exit, both walls and the
far end. If no tube visibly leaves the ringed grain here, use the
no-tube verdict instead of painting.

Appended to the LIVE project (completed 12/12 queue is preserved, never
reset); the two new tasks reopen the queue with exactly this named work.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402

PROJECT = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/rev14analysis")

TASKS = [
    {"uuid": "rev15-exit-0003-51270", "owner": "own-ld-0003",
     "grain": [1022.42, 376.58], "label": "RIGHT grain 0003"},
    {"uuid": "rev15-exit-3756-51270", "owner": "ld|review-141e74fa42fa46d98b2d5c55eec33756",
     "grain": [1006.45, 378.90], "label": "MIDDLE grain 3756"},
]

FRAME = 51270


def main() -> int:
    store = AnnotationStore(PROJECT / "annotations.db")
    try:
        pending = store.unfinished_tasks(limit=500)
        print(f"live queue before: {len(pending)} pending")
        for t in TASKS:
            task = {
                "uuid": t["uuid"], "owner_uuid": t["owner"],
                "owner_key": f"ld|{t['owner']}", "tube_uuid": t["owner"],
                "query_frames": [FRAME], "task_type": "body_mask",
                "movie": "ld",
                "focus_xy": [round((1022.42 + 1006.45) / 2, 1), 377.7],
                "target_xy": list(t["grain"]),
                "target_r": 14.0,
                "brush_px": 5.0, "completed": False,
                "stratum": "rev15-exit", "priority": 30.0,
                "why": (
                    f"TRUE-EXIT TRACE — {t['label']} at frame {FRAME}. Paint the tube "
                    "that leaves the MAGENTA-ringed grain STARTING AT its rim exit "
                    "(both walls and the far end). This grain's exit was never "
                    "verified: two inherited paths were downgraded to PARTIAL because "
                    "their starts cannot be assumed to be this grain's exit. If no "
                    "tube visibly leaves the ringed grain here, use the no-tube "
                    "verdict instead of painting — never guess, never paint the "
                    "neighbour grain's tube."
                ),
            }
            store.save("task", t["uuid"], task, actor="rev15exit")
            print(f"wrote {t['uuid']}: {t['label']} grain {t['grain']}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
