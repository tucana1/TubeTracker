"""Prepare the v30 round-2 batch: draft-adjudication negatives (rev5 #2).

Every task carries ONE pre-drawn candidate box (yellow draft). One
click inside it confirms the box as a verified negative and advances;
anything else stays with guidance. No drawing skill needed, no junk
boxes possible (declining advices nothing into supervision).

Draft sources (verified geometry first):
  - elbows: interior vertices (1/3, 2/3 arclength) of confirmed FULL
    paths — verified tube shaft, nonterminal for that tube's front.
  - rims: 56px boxes on confirmed roots (grain rim + exit, cap far).
  - peak candidates: v1 peaks >=0.5, >60px from any banked tip on that
    frame (UNVERIFIED — the user adjudicates; a real tip inside must
    NOT be confirmed).
Writes a FRESH project dir (never touches live projects).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--v1-weights", required=True)
    ap.add_argument("--tag", default="v30d")
    ap.add_argument("--actor", default="v30draft")
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
    movies = {k: v["path"] for k, v in json.loads(
        (snap / "snapshot_manifest.json").read_text())["movies"].items()}

    banked_tips: dict[tuple, list] = {}
    for o in obs:
        if o.get("direct_xy") and str(o.get("direct_state", "")) in (
                "direct_visible", "visible_imprecise"):
            banked_tips.setdefault(
                (str(o.get("movie", "")),
                 int(o.get("source_frame", -1))), []).append(
                [float(o["direct_xy"][0]), float(o["direct_xy"][1])])

    tasks: list[dict] = []
    n = 0

    def add(movie: str, frame: int, box, source: str, why: str) -> None:
        nonlocal n
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        tasks.append({
            "uuid": f"{a.tag}-{n:03d}", "owner_uuid": "unassigned",
            "query_frames": [int(frame)], "task_type": "neg_draft",
            "movie": movie, "focus_xy": [float(cx), float(cy)],
            "draft_boxes": [[float(v) for v in box]],
            "neg_class": "apex", "draft_source": source,
            "completed": False, "stratum": f"{a.tag}-neg",
            "why": why, "priority": 10.0})
        n += 1

    # 1-2. Elbows + rims from confirmed FULL paths.
    seen_geom: set[tuple] = set()
    for o in sorted(obs, key=lambda d: str(d.get("obs_uuid", ""))):
        if not (o.get("path_complete") and len(o.get("path_xy", [])) >= 2):
            continue
        tip = o.get("direct_xy")
        gkey = (str(o.get("movie", "")), int(o.get("source_frame", -1)),
                round(float(tip[0]), 1) if tip else None,
                round(float(tip[1]), 1) if tip else None)
        if gkey in seen_geom:
            continue  # span-confirmation double
        seen_geom.add(gkey)
        pts = np.asarray(o["path_xy"], float)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        movie, frame = str(o.get("movie", "")), int(o.get("source_frame"))
        for frac, label in ((1.0 / 3, "elbow"), (2.0 / 3, "elbow")):
            tgt = s[-1] * frac
            j = int(np.searchsorted(s, tgt).clip(1, len(pts) - 1))
            mx, my = float(pts[j][0]), float(pts[j][1])
            add(movie, frame, [mx - 28, my - 28, mx + 28, my + 28],
                "elbow",
                "candidate NON-tip: tube bend. Click INSIDE the yellow "
                "box only if no tip is inside it — that confirms it.")
        rx, ry = float(pts[0][0]), float(pts[0][1])
        add(movie, frame, [rx - 28, ry - 28, rx + 28, ry + 28], "rim",
            "candidate NON-tip: ball rim where the tube leaves. Click "
            "INSIDE the yellow box only if no tip is inside it.")

    # 3. v1 peak candidates on banked frames (needs the frozen model).
    from build_annotation_batch import (load_tip_model,  # noqa: E402
                                        significant_peaks,
                                        tip_heat_for_frame)
    from tubetracker.annotation_frames import FrameReader  # noqa: E402

    model, device = load_tip_model(Path(a.v1_weights))
    frames: dict[tuple, None] = {}
    for o in obs:
        if o.get("direct_xy"):
            frames[(str(o.get("movie", "")),
                    int(o.get("source_frame", -1)))] = None
    # Peak candidates are capped TIGHT (2/frame, 12 total by heat):
    # adjudication time is the budget, geometry drafts come first.
    peak_cands: list[tuple] = []
    for (movie, frame) in sorted(frames):
        if movie not in movies:
            continue
        reader = FrameReader(movies[movie])
        try:
            res = reader.read(frame)
        finally:
            reader.close()
        gray = res.frame
        heat = tip_heat_for_frame(model, device, gray)
        H, W = heat.shape
        rim_centers = [(t["focus_xy"][0], t["focus_xy"][1]) for t in tasks
                       if t["movie"] == movie and
                       t["query_frames"] == [frame]]
        made = 0
        for x, y in significant_peaks(heat, 0.5):
            if made >= 2:
                break
            if any(abs(x - bx) < 60 and abs(y - by) < 60
                   for bx, by in banked_tips.get((movie, frame), [])):
                continue  # near a banked tip: not a negative candidate
            if any(abs(x - rx) < 30 and abs(y - ry) < 30
                   for rx, ry in rim_centers):
                continue  # already covered by a rim draft
            if not (24 <= x <= W - 24 and 24 <= y <= H - 24):
                continue
            peak_cands.append((0.0, movie, frame, float(x), float(y),
                               float(heat[int(y), int(x)])))
            made += 1
    peak_cands.sort(key=lambda r: -r[5])
    for _, movie, frame, x, y, _h in peak_cands[:12]:
        add(movie, frame, [x - 24, y - 24, x + 24, y + 24],
            "peak-candidate",
            "UNVERIFIED bright spot. Click INSIDE only if it is "
            "NOT a tip (shaft, elbow, rim, debris). If it IS a "
            "real tip, press Can't-tell instead.")

    store = AnnotationStore(proj / "annotations.db")
    try:
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
    finally:
        store.close()
    by: dict[str, int] = {}
    for t in tasks:
        by[t["draft_source"]] = by.get(t["draft_source"], 0) + 1
    print(f"{a.tag}: +{len(tasks)} {by} -> {proj}")


if __name__ == "__main__":
    main()
