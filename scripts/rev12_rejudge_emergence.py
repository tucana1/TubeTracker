"""Re-judgment pass for the emergence review (rev12e-002..010).

Round 1's app flow let 'click a tube start' and 'No tube' blur together:
a root click could be the user completing the form, not a positive
claim. This script rewrites the instructions so the two answers are
unmistakable and resets the four earlier frames for a clean re-judgment
(Can't tell is a first-class outcome). The 26460 task stays as
confirmed (stub visible, per the user).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402

PROJ = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/"
            "rev12emergence")

WHY_REJUDGE = (
    "RE-JUDGE (same grain, frame {f}) — the previous pass mixed up the "
    "two answers, so this one records your true reading. Click the BALL "
    "first. Then exactly one of:\n"
    "- you see ANY tube paint from this grain (even a tiny stub) -> "
    "click where the tube comes out ('Tube start');\n"
    "- you see NO tube paint at all -> press 'No tube — save & next';\n"
    "- you cannot tell -> press 'Can't tell whose tube — save & next'.\n"
    "'I think so' is not one of the options — if it is not clearly one "
    "way or the other, Can't tell is the right answer and costs nothing.")


def main() -> None:
    store = AnnotationStore(PROJ / "annotations.db")
    try:
        for uid, data in list(store._db.execute(
                "select uuid, data from entities where kind='task'")):
            d = json.loads(data)
            n = int(uid.split("-")[1])
            if n in (2, 3, 4, 5):
                f = int(d["query_frames"][0])
                d["why"] = WHY_REJUDGE.format(f=f)
                d["completed"] = False
                d.pop("grains", None)          # clean re-judgment
                store.save("task", uid, d, actor="rev12-rejudge")
                print(f"reset {uid} (frame {f})")
            elif n in (6, 7, 8, 9, 10):
                f = int(d["query_frames"][0])
                d["why"] = WHY_REJUDGE.format(f=f)
                store.save("task", uid, d, actor="rev12-rejudge")
                print(f"updated {uid} (frame {f})")
    finally:
        store.close()
    print("done")


if __name__ == "__main__":
    main()
