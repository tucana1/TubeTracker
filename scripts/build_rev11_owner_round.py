"""Build the rev11 owner-selection round (change-list item 7).

The review, item 7: "Use an owner-selection task to let the user
identify the grain, then offer visible tube / verified absent /
uncertain. ... The three new crossings have unresolved mappings and no
tip labels; obtain only the missing persistent-owner/cap information
actually needed. Keep unknown as a valid outcome. No broad redraw or
replacement batch is justified by these fixes."

Two SMALL owner-selection tasks, each grounded in recorded evidence:

1. `rev11o-000` — the crossing cluster (frame 51240, the interval's
   ambiguous middle the annotator already traced as rev10x-001). The
   two traced lanes come from this clump; their persistent-owner
   mappings are unresolved. Marking each ball and where its tube comes
   out resolves which ball owns which lane. No machine-guessed balls:
   the user identifies every ball (the app starts with an empty table).

2. `rev11o-001` — the three-owner fit-gate clump (frame 42000, the
   site of `rev11_fitcheck_gate600`, owners rev8p-42000-g0/g1/g2).
   The gate cannot separate the three owners; human-confirmed ball
   positions and tube starts are the missing persistent-owner records
   (item 6). If a fourth ball belongs to the pile, marking it is the
   honest outcome — the review wants what is actually there.

Neither task carries a target ring or seeded ball: both are
identify-the-grain tasks. View centers are placement, not claims
(recorded as such). Writes a FRESH project dir; never touches live
projects.
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

    tasks = [
        {
            "uuid": "rev11o-000",
            "task_type": "owner",
            "movie": "ld",
            "owner_uuid": "unassigned",
            "query_frames": [51240],
            "focus_xy": [975.0, 350.0],
            "view_zoom": 4.0,
            "stratum": "rev11-owner",
            # The view center is placement, not a claim: every ball is
            # marked by the user (the table starts empty).
            "location_provenance": (
                "machine-chosen VIEW CENTER only; the balls and tube "
                "starts are user-marked, no machine-guessed targets"),
            "why": (
                "CROSSING OWNERS. The two tubes traced at the crossing "
                "interval (rev10x-000/001/002, v29 owners 19/22) come "
                "from this clump, and which physical ball owns which "
                "tube is unresolved. Mark every ball you can see and "
                "where ITS tube comes out (the app starts empty — you "
                "identify each ball). A ball with no tube: press "
                "'No tube' in its row. If the clump cannot be "
                "separated, press 'Too tangled'. Unknown is a valid "
                "outcome."),
            "completed": False,
            "completeness": "partial",
        },
        {
            "uuid": "rev11o-001",
            "task_type": "owner",
            "movie": "ld",
            "owner_uuid": "unassigned",
            "query_frames": [42000],
            "focus_xy": [619.0, 662.0],
            "view_zoom": 4.0,
            "stratum": "rev11-owner",
            "location_provenance": (
                "machine-chosen VIEW CENTER only; the balls and tube "
                "starts are user-marked, no machine-guessed targets"),
            "why": (
                "THREE-OWNER CLUMP (fit-gate site, frame 42000). The "
                "three reviewed bodies g0/g1/g2 live in this pile and "
                "the fit gate cannot separate them cleanly. Mark each "
                "ball and where ITS tube comes out — these become the "
                "persistent-owner records. If the pile has a fourth "
                "ball, mark it too; if it cannot be separated, press "
                "'Too tangled'. Unknown is a valid outcome."),
            "completed": False,
            "completeness": "partial",
        },
    ]

    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
        print(f"wrote {len(tasks)} tasks to {proj}: "
              f"{[t['uuid'] for t in tasks]}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
