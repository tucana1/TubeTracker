"""rev17 emergence-bracket round: publish germination_event tasks to the LIVE
project so the investigator can bracket the rim-disruption onset (last-smooth /
first-bump) on a handful of isolated sparse grains.

Grains from the rev17 field scan, re-centred with a tube-robust circle fit and
vision-verified: four clean grains with real tubes (S3/S4/S6/S9) to bracket the
emergence onset, and one clean round grain with no tube (S11) as a negative.
S10 was dropped: the scan's position was a Hough false positive — at the open
frame there is only a fragment / empty space there (confirmed by vision), not a
real grain. Roles preserve independence: S3/S4/S6 calibration (fit the
confidence threshold), S9 + S11 held-out (one onset + one negative) for
independent timing/false-positive evaluation.

Centre = the tube-robust circle-fit centre of the grain body AT THE OPEN FRAME,
so the grain is properly centred the moment it is first shown (the request).
Grains drift 2-12px across their windows; no per-frame tracking is applied —
the ring may sit a few px off the grain later in the scrub, which is accepted.

The investigator cannot mark frame-perfectly and is not asked to: the bracket is
recorded as an approximate window (last_absent -> first_visible) and the onset
detector already reports a resolution-limited range. Appended to the LIVE project
(12/12 completed queue is preserved); launches scoped to just these tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from tubetracker.annotation_tasks import germination_episode  # noqa: E402
from tubetracker.analysis_project import QUEUE  # noqa: E402

PROJECT = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/rev14analysis")
MOVIE_HASH = "238ebd68f4d97ceca32889cc68191d6d11e71be5fd5c2352bbf22438498b13e1"

# id, grain_native (tube-robust circle-fit centre AT open frame), radius,
# window_start, window_end, open_frame, role
GRAINS = [
    ("S3",  [490.66, 252.57],  11.86, 13000, 23000, 18500, "calibration"),
    ("S4",  [1085.39, 870.02], 11.87,  9000, 19000, 14500, "calibration"),
    ("S6",  [202.11, 744.91],  13.03, 17000, 27000, 22500, "calibration"),
    ("S9",  [883.91, 187.22],  12.98,  5000, 15000, 10500, "heldout"),
    ("S11", [631.81, 642.40],  11.89, 15000, 25000, 20000, "heldout"),
]

WHY = (
    "EMERGENCE-BRACKET ({sid}, {role}) — watch THIS grain (green ring) only. "
    "The question is whether its circular rim ever breaks into a tiny bump or "
    "outgrowth (a nascent pollen tube). Scrub the frames in [{ws}, {we}] with "
    "the -300/-30/-1 buttons or type a frame number in 'Go to frame #…'.\n\n"
    "If a bump emerges from the rim during this range:\n"
    "  • scrub to the LAST frame where the rim is still smooth -> 'Mark last "
    "smooth rim HERE'\n"
    "  • scrub to the FIRST frame where a bump/outgrowth breaks the circle -> "
    "'Mark first bump/outgrowth HERE'\n"
    "  • press 'Emerged within window' (your two marks become the bracket).\n"
    "If the bump is already there at the first frame -> 'Already emerged at "
    "start'. If the rim stays circular through the whole range -> 'No emergence "
    "by end'. If the grain is too buried to tell -> 'Unobservable'.\n\n"
    "Do NOT expect to be frame-perfect — I don't need that. Mark your best "
    "estimate within a solid realm of reason; an approximate bracket (within a "
    "few hundred frames) is exactly what I need to calibrate the detector. "
    "The marks can be stamped on any frame you are viewing (buttons never grey "
    "out)."
)


def main() -> int:
    store = AnnotationStore(PROJECT / "annotations.db")
    try:
        before = len(store.unfinished_tasks(limit=500))
        for idx, (sid, grain, rad, ws, we, openf, role) in enumerate(GRAINS):
            uuid = f"rev17-germ-{sid}"
            ep = germination_episode(
                uuid=uuid,
                grain_id=f"rev17-{sid}",
                movie_id="ld",
                movie_content_hash=MOVIE_HASH,
                window_start=ws, window_end=we,
                unresolved_error=(
                    "the emergence-onset threshold (where a rim bump becomes a "
                    "reliably-visible tube) is calibrated on only cf70+G0 and "
                    "cannot be trusted to generalize"),
                why_existing_insufficient=(
                    "cf70/G0 brackets alone cannot separate a clean growth knee "
                    "from resolution-limited early drift; more reviewed onsets "
                    "across grains are needed to fit and independently validate "
                    "the confidence/threshold"),
                cheapest_answer=(
                    "temporal last-smooth / first-bump bracket on this grain "
                    "(approximate is fine) + one verdict"),
                consuming_loss_or_eval=(
                    "emergence_onset.detect_onset threshold + confidence fit "
                    "(calibration) or frozen onset-vs-bracket error (heldout)"),
                role=role,
                before_after_comparison=(
                    "detector onset vs the reviewed bracket on this grain before "
                    "vs after threshold refit; onset error in frames"),
                stratum="sparse_emergence_bracket",
            )
            ep.update({
                "movie": "ld",
                # These fields are what make the task appear in the app's
                # "Annotation queue" tab: ReviewQueuePanel reads the store for
                # tasks with review_queue == QUEUE, sorted by queue_order.
                # Without them the tab is empty and nothing renders.
                "review_queue": QUEUE,
                "queue_order": 100 + idx,
                "queue_batch": "rev17-germ",
                "queue_position": idx + 1,
                "query_frames": [openf],
                "focus_xy": list(grain),
                "target_xy": list(grain),
                "target_r": rad,
                "view_zoom": 5.5,
                "why": WHY.format(sid=sid, role=role, ws=ws, we=we),
                "completed": False,
                "completeness": "unreviewed",
            })
            store.save("task", uuid, ep, actor="rev17germ")
            print(f"wrote {uuid} grain {grain} window [{ws},{we}] role={role}")
        after = len(store.unfinished_tasks(limit=500))
        print(f"live queue pending: {before} -> {after} (added {len(GRAINS)})")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
