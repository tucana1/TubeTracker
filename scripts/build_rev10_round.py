"""Build the rev10 round-3 batch: one owned-absence case + a crossing interval.

Review (rev10, "the next human request should be small and targeted"):
  - add one owned-absence case beside a foreign tube;
  - annotate a short held-out clump/crossing interval including clear
    anchors and an ambiguous middle.

Both cases mined from recorded evidence, never invented:

OWNED ABSENCE (`rev10a-000`): grain `obs-r4-p02` (one of the three clump
grains, owner of mask-rev8m-001). Its human-traced tube at frame 50400
matches v29 centerline owner 13 to 4.4 px mean. The v29 trace has NO tube
for owner 13 at frame 28770, while a FOREIGN tube (v29 owner 8) passes
39 px from the grain centre. Verified visually: the grain sits alone.

CROSSING INTERVAL (`rev10x-000..002`): v29 owners 19/22. At the sampled
crossing frame 51240 their paths close to 4.1 px (verified visually: the
two strands converge and the assignment is ambiguous); at 51030 and 51450
the same pair is 26.9/27.5 px apart on two clearly separable tubes
(verified visually at 51030). That is a clear anchor - ambiguous middle -
clear anchor interval.

Never touches live projects: writes a FRESH project dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="rev10round")
    a = ap.parse_args()

    tasks = []

    # ---- owned absence beside a foreign tube -------------------------
    tasks.append({
        "uuid": "rev10a-000",
        "task_type": "body_mask",
        "movie": "ld",
        "owner_uuid": "obs-r4-p02",
        "owner_key": "ld|obs-r4-p02",
        "tube_uuid": "obs-r4-p02",
        "source_obs_uuid": "obs-r4-p02",
        "query_frames": [28770],
        "focus_xy": [1075.2, 134.4],
        "target_xy": [1075.2, 134.4],
        "target_r": 14.0,
        # rev11 item 4/7: this anchor is MACHINE-GUESSED (detector/trace
        # placement), NOT a human-confirmed grain center. Any supervised
        # label built from this task must carry that provenance.
        "location_provenance": (
            "machine-guessed (v29 trace + detector placement; not a "
            "human-confirmed grain center)"),
        "guide_path": [],          # no own route exists here: that IS the point
        "brush_px": 9.0,
        "priority": 10.0,
        "stratum": "rev10-absence",
        "mask_points": [],
        "completed": False,
        "completeness": "partial",
        "why": ("OWNED-ABSENCE CASE. This is grain obs-r4-p02. At this "
                "frame the recorded v29 trace has NO tube for this owner, "
                "while a FOREIGN tube passes about 39 px away. If you "
                "agree there is no tube of THIS grain visible, press "
                "'No tube' (that banks the owned-absence case the review "
                "asks for). If a tube of this grain IS visible, paint it "
                "as usual."),
    })

    # ---- crossing interval: anchor - ambiguous middle - anchor -------
    x_defs = [
        ("rev10x-000", 51030, [995.2, 366.0], 26.9,
         "anchor BEFORE the crossing: owners 19/22 are 26.9 px apart on "
         "two clearly separate tubes - trace each lane separately"),
        ("rev10x-001", 51240, [954.0, 332.0], 4.1,
         "AMBIGUOUS MIDDLE: owners 19/22 close to 4.1 px here and the "
         "two strands converge - trace each tube as far as you can "
         "follow it and mark what stays uncertain"),
        ("rev10x-002", 51450, [996.0, 367.0], 27.5,
         "anchor AFTER the crossing: owners 19/22 are 27.5 px apart on "
         "two clearly separate tubes - trace each lane separately"),
    ]
    for uuid, fid, focus, gap, why in x_defs:
        tasks.append({
            "uuid": uuid,
            "task_type": "crossing",
            "movie": "ld",
            "owner_uuid": "unassigned",
            "query_frames": [fid],
            "focus_xy": focus,
            "stratum": "rev10-crossing",
            # rev11 item 4: a crossing anchor is a MACHINE-GUESSED
            # location on the frame; it is not a certain grain center,
            # and local A/B lane names do not establish persistent
            # owners (review section 7).
            "location_provenance": (
                "machine-guessed anchor (crossing frame placement; local "
                "A/B lanes carry no persistent owner meaning)"),
            "class_scope": "crossing",
            "geometry_type": "path",
            "completeness": "partial",
            "completed": False,
            "why": f"crossing interval (v29 owners 19/22, {gap} px): {why}",
        })

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
