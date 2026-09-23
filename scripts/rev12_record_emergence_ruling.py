"""Record the investigator's emergence ruling + close the review round.

Josh's ruling (18 Sep): the tip-like structure at this grain's exit may
be detritus that was always there, but it looks enough like a tip that
the grain should be classified as STARTED GERMINATED AND NEVER GREW —
i.e., germinated at movie start (left-censored, no certified absent
bound), no growth in the reviewed era. This is the investigator's
germination rule element, stored explicitly (never invented by the
pipeline). rev12e-016 (22680) is withdrawn: the no-tube hunt is moot
under the ruling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402

PROJ = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/"
            "rev12emergence")

RULING = {
    "kind": "investigator-ruling",
    "uuid": "ruling-001",
    "grain_owner": "obs-r4-p02",
    "reviewed_frames": [26460, 26040, 25830, 25620, 25410, 25200, 24990,
                        24780, 24570, 24360, 23940, 23520, 23310, 23100,
                        22890],
    "ruling": ("started germinated and never grew"),
    "verbatim": ("There's a chance that this is just detritus that was "
                 "always there, but it looks enough like a tip that I'd "
                 "rather you just say it started germinated and never "
                 "grew"),
    "rule_elements": [
        "a tip-like structure at the grain exit counts as germinated "
        "(never an absence) when it cannot be distinguished from "
        "detritus",
        "a germinated grain with no elongation over the reviewed era is "
        "recorded as germinated-never-grew (left-censored; no emergence "
        "event)",
        "the no-tube hunt for this grain is closed by this ruling",
    ],
    "actor": "josh",
    "date": "2026-09-18",
}


def main() -> None:
    store = AnnotationStore(PROJ / "annotations.db")
    try:
        # withdraw the remaining no-tube hunt task
        t = store.load("rev12e-016")
        assert t is not None
        d = dict(t["data"])
        d["completed"] = True
        d["withdrawn"] = True
        d["withdrawn_reason"] = (
            "moot under ruling-001 (started germinated and never grew): "
            "no certified-absent bound is sought for this grain")
        store.save("task", "rev12e-016", d, actor="rev12-rejudge")
        print("withdrew rev12e-016 (22680)")
        store.save("ruling", RULING["uuid"], RULING, actor="josh")
        print("stored ruling-001")
    finally:
        store.close()
    print("done")


if __name__ == "__main__":
    main()
