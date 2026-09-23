"""Build PAIRED mask tasks: several grains in ONE camera view.

rev8 step 3's query-swap test needs two (or more) DIFFERENT grains
whose tubes are visible in the same crop, each masked by hand. A
normal mask task frames one grain and traces its path; here every task
shares one view centre, so all the tubes of the clump are painted in
the same crop, and the magenta target ring says which grain the
current task is about.

No guide path is drawn for these: there is no traced geometry for the
neighbouring grains, and a guide would bias the painter toward a
proposal. The ring is the only marker — the hand, not a proposal, is
the evidence.

Writes a FRESH project dir (never touches live projects).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", required=True,
                    help="output of find_neighbor_balls.py")
    ap.add_argument("--frame", required=True, help="movie:frame")
    ap.add_argument("--index", type=int, default=0,
                    help="which pair entry in that frame (score order)")
    ap.add_argument("--extra-ball", default="",
                    help="optional 'x,y' third grain in the same view")
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="rev8p")
    ap.add_argument("--actor", default="rev8pair")
    ap.add_argument("--target-r", type=float, default=14.0)
    ap.add_argument("--view-center", default="",
                    help="x,y override for the shared camera centre")
    a = ap.parse_args()

    from tubetracker.annotation_store import AnnotationStore

    data = json.loads(Path(a.pairs_json).read_text())
    entry = data.get(a.frame)
    if not entry:
        print(f"no pair entry for {a.frame}; have {sorted(data)}")
        return 1
    pr = entry[int(a.index)]
    movie, _, fr_s = a.frame.partition(":")
    frame = int(fr_s)
    balls = [pr["a"], pr["b"]]
    if a.extra_ball:
        x, y = [float(v) for v in a.extra_ball.split(",")]
        balls.append([x, y, 0.0])

    if a.view_center:
        cx, cy = [float(v) for v in a.view_center.split(",")]
    else:
        cx = sum(b[0] for b in balls) / len(balls)
        cy = sum(b[1] for b in balls) / len(balls)

    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return 1
    proj.mkdir(parents=True)

    tasks = []
    for i, b in enumerate(balls):
        # rev8 identity: the owner is the GRAIN in this frame, named by
        # the task itself (task-derived link), never a proximity guess.
        owner = f"{a.tag}-{frame}-g{i}"
        tasks.append({
            "uuid": f"{a.tag}-{i:03d}", "owner_uuid": owner,
            "owner_key": f"{movie}|{owner}", "tube_uuid": owner,
            "query_frames": [frame], "task_type": "body_mask",
            "movie": movie,
            "focus_xy": [round(cx, 1), round(cy, 1)],
            "target_xy": [round(float(b[0]), 1), round(float(b[1]), 1)],
            "target_r": float(a.target_r),
            "paired_frame": a.frame,
            "brush_px": 9.0, "completed": False,
            "stratum": f"{a.tag}-pair", "priority": 20.0,
            "why": "Paint the tube that leaves the MAGENTA-ringed grain "
                   "(both walls and the far end). Other grains in this "
                   "view belong to other tasks — leave them unpainted. "
                   "If the ringed grain has no visible tube here, use "
                   "the no-tube verdict instead of painting.",
        })
    store = AnnotationStore(proj / "annotations.db")
    try:
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
    finally:
        store.close()
    print(f"{a.tag}: +{len(tasks)} paired mask tasks -> {proj}")
    for t in tasks:
        print(f"   {t['uuid']}: target=({t['target_xy'][0]:.0f},"
              f"{t['target_xy'][1]:.0f}) view=({t['focus_xy'][0]:.0f},"
              f"{t['focus_xy'][1]:.0f}) frame={frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
