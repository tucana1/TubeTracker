"""Build the rev11 owned-absence review task (change-list item 7 tail).

The review: "Use an owner-selection task to let the user identify the
grain, then offer visible tube / verified absent / uncertain. ...
Verify the resulting owner ID through database, snapshot, target
builder and loss before requesting an absence review. Do not present
another machine-guessed location as an absent grain."

H375 (the parking rule) required the absence case to be built from a
human-placed grain, with the new-owner linking verified first. Both
conditions now hold: the owner task type is proven (rev11own revs
17/18) and its records verified through database -> snapshot -> target
builder -> loss (H401), with the persistent registry built (H402).

Site evidence (recorded, not invented):
- obs-r4-p01's human-traced tube at frame 47250 starts at its root
  (984.4, 57.9); the v29 trace of the same tube (pollen_id 4,
  lowdens_full_v29_18_2) begins at frame 13650 with a short static
  stub at (938,57)->(937,68), 9 px below a distinct round ball at
  ~(940,45-50) — the same root-to-trace relationship as at 47250.
- No growing tube is apparent at 13650-17000 (visual check at zoom;
  the final judgment is the user's, which is the point of the task).

The task presents a VIEW, never an absent-grain claim: the user marks
the ball; 'No tube' banks the verified absence; a visible tube gets
marked instead; uncertain is a valid outcome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="rev11own")
    a = ap.parse_args()

    task = {
        "uuid": "rev11o-002",
        "task_type": "owner",
        "movie": "ld",
        "owner_uuid": "obs-r4-p01",
        "query_frames": [13650],
        "focus_xy": [940.0, 55.0],
        "view_zoom": 4.0,
        "stratum": "rev11-owner-absence",
        "location_provenance": (
            "machine-chosen VIEW CENTER from the v29 trace (pollen_id 4) "
            "at frame 13650; the grain identity rests on trace continuity "
            "to the human-confirmed attachment at frame 47250, and the "
            "user confirms or rejects it in this task"),
        "why": (
            "OWNED-ABSENCE REVIEW — event obs-r4-p01's grain, early era "
            "(frame 13650). The ball near the middle of this view should "
            "be that grain; below it there is a short 11 px nub whose "
            "status is exactly what we need you to judge. Mark the ball. "
            "If its tube is ABSENT at this frame, press 'No tube' in its "
            "row — that banks the verified absence. If a tube IS visible, "
            "mark its start instead. If you cannot tell, 'Too tangled' "
            "or leaving it unanswered is a valid outcome — never guess."),
        "completed": False,
        "completeness": "partial",
    }

    proj = Path(a.project_dir)
    store = AnnotationStore(proj / "annotations.db")
    try:
        store.save("task", task["uuid"], task, actor=a.actor)
        print(f"wrote {task['uuid']} to {proj}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
