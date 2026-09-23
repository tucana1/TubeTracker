"""rev12 P1.3: build the emergence-interval review (earliest banked stub).

One grain (the r4-p02 tube's), five frames: the earliest validated stub
(26460, tube visible — mark the grain and its tube start) then four
earlier frames (26040, 25830, 25620, 25410) where the question is
whether this grain's tube is ABSENT. 'No tube' at an earlier frame
certifies the last-verified-absent bound; together with the validated
first-verified-present (26460, 1.63 px off the certified trajectory)
this produces the FIRST confirmed emergence interval.

View centers are machine-chosen VIEWS from the validated stub masks;
the grain identity rests on the user's own click (H375: never a
machine-guessed absent grain).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402

STUB_XY = (1089.0, 148.0)      # stub/base region centre (view only)
OWNER = "obs-r4-p02"           # the tube whose stub was validated
FRAMES = [26460, 26040, 25830, 25620, 25410]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="rev12emergence")
    ap.add_argument("--frames", default=",".join(str(f) for f in FRAMES),
                    help="comma-separated frames, newest first")
    ap.add_argument("--start", type=int, default=1,
                    help="first task index (uuid rev12e-<start:03d>+)")
    ap.add_argument("--first-why", default="",
                    help="override the first task's instructions")
    ap.add_argument("--absent-why", default="",
                    help="override the absent-frame instructions "
                         "(format fields {i}/{f})")
    a = ap.parse_args()

    frames = [int(f) for f in a.frames.split(",") if f.strip()]
    why_first = a.first_why or (
        "EMERGENCE REVIEW (1/5) — the earliest banked stub of this tube. "
        "The short tube paint in this view is the tube's first visible "
        "growth (its tip sits ~1.6 px from the tube's later certified "
        "path; this stub was validated). Mark the GRAIN (ball) this tube "
        "grows out of, then click its tube start (the stub's base). The "
        "next four tasks step BACKWARDS in time: there the question is "
        "whether this grain's tube is ABSENT — press 'No tube' in its "
        "row if so. If the grain is hard to pin down, 'Can't tell' is a "
        "valid outcome — never guess.")
    why_absent = a.absent_why or (
        "EMERGENCE REVIEW ({i}/5) — same grain, earlier frame {f}. Mark "
        "the SAME ball you marked in task 1 (it should look unchanged; "
        "only its tube should be missing). If its tube is ABSENT at this "
        "frame, press 'No tube — save & next' in its row: that certifies "
        "the last-verified-absent bound. If a tube IS visible, mark its "
        "start instead. If you cannot tell, 'Can't tell whose tube' is a "
        "valid outcome — never guess.")

    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        for j, f in enumerate(frames):
            i = a.start + j
            uuid = f"rev12e-{i:03d}"
            task = {
                "uuid": uuid,
                "task_type": "owner",
                "movie": "ld",
                "owner_uuid": OWNER,
                "query_frames": [f],
                "focus_xy": [STUB_XY[0], STUB_XY[1]],
                "view_zoom": 5.0,
                "stratum": "rev12-emergence",
                "location_provenance": (
                    "machine-chosen VIEW CENTRE from the validated stub "
                    "masks (mask-rev10a-002 @26460, min pixel distance "
                    "1.63 px to the certified r4-p02 trajectory); the "
                    "grain identity is the user's own click in this task"),
                "why": (why_first if j == 0
                        else why_absent.format(i=i, f=f)),
                "completed": False,
                "completeness": "partial",
            }
            store.save("task", uuid, task, actor=a.actor)
            print(f"wrote {uuid} (frame {f})", flush=True)
    finally:
        store.close()
    print(f"-> {proj}", flush=True)


if __name__ == "__main__":
    main()
