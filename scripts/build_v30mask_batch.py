"""Prepare the rev6 body-mask pilot: paint the tube on 2 train paths.

Each task shows ONE confirmed FULL path as a yellow guide line. The
user paints the visible tube body (both walls + far end) with the
brush. Two tasks only (p00, p03 — train events, dev stays clean):
this pilots whether human masks move distal IoU before any larger
mask batch is requested.

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
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="v30m")
    ap.add_argument("--actor", default="v30mask")
    ap.add_argument("--refs",
                    default="obs-r4-p00,obs-r4-p02,obs-r4-p05",
                    help="comma-separated obs_uuids to mask (train only)")
    ap.add_argument("--target-r", type=float, default=20.0,
                    help="ring radius in px; generous on purpose -- the "
                         "grain centre is only known to a few px and a "
                         "marker that clips the grain reads as off")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the frame-based anchor refinement")
    ap.add_argument("--dev-refs", default="r4-p01,r4-p04,dt-005,dt-006",
                    help="event refs that must NOT be masked (held out)")
    a = ap.parse_args()

    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore

    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return 1
    proj.mkdir(parents=True)

    snap = Path(a.snapshot)
    obs = json.loads((snap / "observations.json").read_text())
    want = [w.strip() for w in a.refs.split(",") if w.strip()]
    tasks = []
    for i, ref in enumerate(want):
        o = next((x for x in obs if x.get("obs_uuid") == ref), None)
        if o is None or not o.get("path_xy") or len(o["path_xy"]) < 2:
            print(f"skip {ref}: no confirmed path")
            continue
        if any(d in ref for d in
               [w for w in a.dev_refs.split(",") if w]):
            print(f"refuse {ref}: dev event, masks stay on train")
            return 1
        import numpy as np
        from prototypes.v30_video_apex.targets import (
            refine_ring_anchor, ring_anchor_for_observation)
        pts = np.asarray(o["path_xy"], float)
        movie = str(o.get("movie", ""))
        anchor, anchor_src = ring_anchor_for_observation(
            o, target_r=float(a.target_r))
        if anchor is None:
            print(f"skip {ref}: no ring anchor")
            continue
        if not a.no_refine and o.get("movie_path"):
            # a path starts at the grain's EDGE: step one grain radius
            # upstream, then snap onto the blob the marker must enclose
            try:
                _fr = FrameReader(str(o["movie_path"]))
                _im = _fr.read(int(o["source_frame"])).frame
                _gray = (_im if _im.ndim == 2 else _im[..., :3].mean(-1))
                anchor, _rsrc, _moved = refine_ring_anchor(
                    anchor, o.get("path_xy") or [], _gray,
                    r=float(a.target_r))
                anchor_src = f"{anchor_src}+{_rsrc}(moved {_moved:.1f}px)"
            except Exception as exc:
                print(f"   {ref}: refine skipped ({exc})")
        print(f"   {ref}: ring at ({anchor[0]:.1f},{anchor[1]:.1f}) "
              f"[{anchor_src}]")
        # rev8 identity: the mask belongs to a NAMED owner (the event),
        # not to an anonymous task — this is what makes the link
        # explicit instead of a proximity guess.
        owner = ref
        # rev8: the ring must mark the GRAIN, not the tube's middle.
        # focus_xy stays the path centroid (the view is framed on the
        # tube), target_xy is the observation's own focus (the ball).
        # Drawing the ring at the centroid put it 7px off on p03 and
        # 56px off on v30t-004 — the user spotted it immediately.
        tasks.append({
            "uuid": f"{a.tag}-{i:03d}", "owner_uuid": owner,
            "owner_key": f"{movie}|{owner}",
            "tube_uuid": owner,
            "query_frames": [int(o["source_frame"])],
            "task_type": "body_mask", "movie": movie,
            # focus_xy frames the view (path mean); target_xy is the
            # ring, and the ring marks the QUERIED GRAIN. The two are
            # different points and are now computed by different rules;
            # this dict previously carried target_xy/target_r TWICE
            # (silently keeping the last), which is how the ring ended
            # up 46px down the tube on obs-r4-p03.
            "focus_xy": [float(pts[:, 0].mean()),
                         float(pts[:, 1].mean())],
            "target_xy": [float(anchor[0]), float(anchor[1])],
            "target_r": float(a.target_r),
            "ring_anchor": anchor_src,
            "guide_path": [[float(q[0]), float(q[1])]
                           for q in o["path_xy"]],
            "source_obs_uuid": str(o.get("obs_uuid", ref)),
            "brush_px": 9.0, "completed": False,
            "stratum": f"{a.tag}-mask", "priority": 10.0,
            "why": "Paint the visible tube body (both walls and the far "
                   "end) along the yellow guide, then answer whether "
                   "you checked the whole view.",
        })
    store = AnnotationStore(proj / "annotations.db")
    try:
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
    finally:
        store.close()
    print(f"{a.tag}: +{len(tasks)} body_mask tasks -> {proj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
