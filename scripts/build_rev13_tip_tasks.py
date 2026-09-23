"""rev13: tip-label verification batch (the supervision bottleneck).

The bounded W3 experiment closed with a coin-flip within-row diagnostic:
the trained scorer separates nothing at the labeled tips. Before any
more model work, the LABELS themselves get eyes: one `review_tip` task
per development event, showing the human tip position as a DRAFT (never
truth) — the reviewer confirms it exactly, clicks the true cap, bounds
it as an area, or says the tube doesn't reach there.

Six tasks, one per event (the gate panel's tips). Small and targeted,
per the work order's annotation guidance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402

MANIFEST = REPO / "runs/prototypes/v30/rev12_proposals.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="rev13tips")
    a = ap.parse_args()

    rows = json.loads(MANIFEST.read_text())["rows"]
    tips: dict[tuple, dict] = {}
    for r in rows:
        if r.get("present") != 1:
            continue
        k = (r["movie"], int(r["frame"]))
        if k in tips:
            continue
        tips[k] = {"owner": r["owner"], "tip": [float(v) for v in
                                                r["tip_native"]],
                   "n_rows": 0, "crop": list(r.get("crop_xywh") or [])}
        tips[k]["n_rows"] = sum(1 for q in rows if q.get("present") == 1
                                and (q["movie"], int(q["frame"])) == k)

    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        for j, ((movie, frame), t) in enumerate(sorted(tips.items())):
            uuid = f"rev13tip-{j + 1:03d}"
            task = {
                "uuid": uuid,
                "task_type": "review_tip",
                "movie": movie,
                "owner_uuid": t["owner"],
                "query_frames": [frame],
                "focus_xy": list(t["tip"]),
                "draft_xy": list(t["tip"]),
                "draft_task": f"manifest|{movie}|{frame}",
                "view_zoom": 6.0,
                "stratum": "rev13-tip-label-check",
                "location_provenance": (
                    f"the manifest's human tip for {t['owner']} at "
                    f"frame {frame} ({t['n_rows']} labeled present "
                    f"rows share it); shown as a DRAFT — the label is "
                    f"under review, not truth"),
                "why": (
                    "TIP LABEL CHECK — is the tube's current cap exactly "
                    "where the dashed mark is? The scorer trained on "
                    "these labels shows no separation, so the label is "
                    "the suspect. If the cap is precisely localizable, "
                    "click it (or 'Confirm draft' if exact). If the "
                    "mark is off, click the real cap. If the tube "
                    "doesn't reach here / you can't tell, say so — "
                    "'Can't tell' is a valid outcome, never guess."),
                "completed": False,
                "completeness": "partial",
            }
            store.save("task", uuid, task, actor=a.actor)
            print(f"wrote {uuid}: {movie}|{frame} tip {t['tip']} "
                  f"({t['owner']})", flush=True)
    finally:
        store.close()
    print(f"-> {proj} ({len(tips)} tasks)", flush=True)


if __name__ == "__main__":
    main()
