"""Prepare the v30 pilot label batch: verified negatives + census tiles (rev5 #2).

Writes tasks into a FRESH project dir (never touches live projects).
The user draws each box / marks each tile in the annotator; nothing
here claims geometry — suggestions are gaze points with a stated
reason, and every box becomes supervision only after human confirmation
(SupervisionRegion, kind=verified_negative).

Contents (pilot allocation):
  - per FULL path: 1 elbow box task (path midpoint = verified tube
    interior, nonterminal for that tube's front) + 1 root-rim task.
  - per precise tip: 1 open-review task (nearest non-tip structure).
  - 4 census tiles: 2 adjacent to events, 2 random (seeded).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--project-dir", required=True,
                    help="fresh dir (must not exist)")
    ap.add_argument("--tag", default="v30n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actor", default="v30neg")
    a = ap.parse_args()

    import numpy as np

    from tubetracker.annotation_store import AnnotationStore

    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return
    proj.mkdir(parents=True)

    snap = Path(a.snapshot)
    obs = json.loads((snap / "observations.json").read_text())
    movies = json.loads(
        (snap / "snapshot_manifest.json").read_text())["movies"]
    tip_at: dict[tuple, list] = {}
    for o in obs:
        if o.get("direct_xy") and str(o.get("direct_state", "")) in (
                "direct_visible", "visible_imprecise"):
            tip_at.setdefault(
                (str(o.get("movie", "")),
                 int(o.get("source_frame", -1))),
                []).append([float(o["direct_xy"][0]),
                            float(o["direct_xy"][1])])

    tasks: list[dict] = []
    n = 0

    def add(task_type: str, movie: str, frame: int, focus,
            why: str, **kw) -> None:
        nonlocal n
        t = {"uuid": f"{a.tag}-{n:03d}", "owner_uuid": "unassigned",
             "query_frames": [int(frame)], "task_type": task_type,
             "movie": movie, "focus_xy": [float(focus[0]), float(focus[1])],
             "completed": False, "stratum": f"{a.tag}-neg",
             "why": why, "priority": 10.0}
        t.update(kw)
        if task_type == "neg_region":
            # Show the known tip so the user boxes look-alikes, never it.
            known = tip_at.get((movie, int(frame)))
            if known:
                t["known_tip_xy"] = list(known[0])
        tasks.append(t)
        n += 1

    rng = np.random.default_rng(a.seed)
    path_frames, tip_frames = [], []
    seen_geom: set[tuple] = set()  # span doubles re-click the same geometry
    for o in sorted(obs, key=lambda d: str(d.get("obs_uuid", ""))):
        movie = str(o.get("movie", ""))
        frame = int(o.get("source_frame", -1))
        tip = o.get("direct_xy")
        gkey = (movie, frame,
                round(float(tip[0]), 1) if tip else None,
                round(float(tip[1]), 1) if tip else None)
        if gkey in seen_geom:
            continue
        seen_geom.add(gkey)
        if o.get("path_complete") and len(o.get("path_xy", [])) >= 2:
            pts = np.asarray(o["path_xy"], float)
            mid = pts[len(pts) // 2]
            add("neg_region", movie, frame, mid,
                "elbow review: box the nonterminal bend/shaft (cap-negative, "
                "shaft stays shaft)", neg_class="apex")
            add("neg_region", movie, frame, pts[0],
                "root-rim review: box the grain rim/exit (cap-negative, "
                "grain stays grain)", neg_class="apex")
            path_frames.append((movie, frame, o))
        elif o.get("direct_xy") and str(o.get("direct_state", "")) in (
                "direct_visible", "visible_imprecise"):
            tip = o["direct_xy"]
            add("neg_region", movie, frame, tip,
                "open review: box the nearest NON-tip structure "
                "(elbow/rim/shaft ok; leave unresolvable caps alone)",
                neg_class="apex")
            tip_frames.append((movie, frame, o))

    # Census: 2 tiles adjacent to banked events + 2 random ld frames.
    anchors = (path_frames + tip_frames)[:2]
    for movie, frame, _ in anchors:
        add("census", movie, frame, (640.0, 512.0),
            "exhaustive tile: mark EVERY visible tip + ball in the "
            "320px box around this point, then Tile-complete")
    ld_frames = sorted({f for m, f, _ in (path_frames + tip_frames)
                        if m == "ld"})
    lo = min(ld_frames + [5000])
    for i in range(2):
        f = int(rng.integers(max(0, lo - 20000), lo + 20000))
        add("census", "ld", f,
            (float(rng.uniform(320, 960)), float(rng.uniform(320, 704))),
            "random tile (independent discovery check): mark EVERY "
            "visible tip + ball in the 320px box, then Tile-complete")

    store = AnnotationStore(proj / "annotations.db")
    try:
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
    finally:
        store.close()
    by = {}
    for t in tasks:
        by[t["task_type"]] = by.get(t["task_type"], 0) + 1
    print(f"{a.tag}: +{len(tasks)} {by} -> {proj}")
    print("movies:", {k: v["path"] for k, v in movies.items()})


if __name__ == "__main__":
    main()
