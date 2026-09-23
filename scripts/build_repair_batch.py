#!/usr/bin/env python3
"""Repair-and-extension annotation batch (P0A rev4, H261).

Appends review tasks to their SOURCE projects so draft uuids resolve
in-project (no cross-DB merge-back):
  - DT project: review_tip for each visible dim tip (precise-vs-region
    judgment; draft shown, never auto-truth).
  - R3 project: review_path for each path stroke (span/root span
    confirmation), review_crossing for each crossing (lane re-trace +
    continuation assignment), plus fresh germinated-clump owner tasks
    (FRST cluster + tip-heat = has tubes; the missing root attachments).
Strokes are preserved as editable drafts; nothing is redrawn from zero.
"""

from __future__ import annotations

import argparse
import math
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from grain_detect import detect_grains  # noqa: E402

M2 = "/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4"


def _db(project: Path):
    con = sqlite3.connect(str(project / "annotations.db"))
    tasks = {u: json.loads(d) for u, d in
             con.execute("SELECT uuid, data FROM entities WHERE kind='task'")}
    obs = {u: json.loads(d) for u, d in con.execute(
        "SELECT uuid, data FROM entities WHERE kind='observation'")}
    by_task: dict[str, dict] = {}
    for _u, _o in obs.items():
        if _o.get("task_uuid"):
            by_task[_o["task_uuid"]] = _o
    return con, tasks, by_task


def _existing_uuids(tasks: dict) -> set:
    return set(tasks)


def dt_review_tips(dt: Path) -> list[dict]:
    """review_tip tasks for DT visible dim tips."""
    _, tasks, by_task = _db(dt)
    have = _existing_uuids(tasks)
    out = []
    for uuid in sorted(tasks):
        t = tasks[uuid]
        if not t.get("completed") or t.get("resolution") in (
                "unresolvable", "not_a_ball"):
            continue
        o = by_task.get(uuid)
        if (o is None or o.get("direct_state") != "direct_visible"
                or not o.get("direct_xy")):
            continue
        rid = f"{uuid}-recheck"
        if rid in have:
            continue
        fid = int(t["query_frames"][0])
        out.append({
            "uuid": rid,
            "owner_uuid": t.get("owner_uuid", "unassigned"),
            "movie": t.get("movie", "ld"),
            "query_frames": [fid],
            "focus_xy": list(o["direct_xy"]),
            "draft_xy": list(o["direct_xy"]),
            "draft_task": uuid,
            "task_type": "review_tip",
            "stratum": "repair-tip",
            "why": ("re-check dim tip: precise point or bounded area; "
                    "draft shown, never truth"),
            "completed": False,
        })
    return out


def r3_reviews(r3: Path) -> list[dict]:
    """review_path + review_crossing tasks from R3 strokes."""
    con, tasks, by_task = _db(r3)
    cross = {u: json.loads(d) for u, d in con.execute(
        "SELECT uuid, data FROM entities WHERE kind='crossing'")}
    con.close()
    have = _existing_uuids(tasks)
    out = []
    for uuid in sorted(tasks):
        t = tasks[uuid]
        if not t.get("completed") or t.get("resolution") in (
                "unresolvable", "not_a_ball"):
            continue
        o = by_task.get(uuid)
        if t.get("task_type") == "centerline" and o is not None and o.get(
                "path_xy"):
            rid = f"{uuid}-span"
            if rid not in have:
                path = [list(q) for q in o["path_xy"]]
                out.append({
                    "uuid": rid,
                    "owner_uuid": t.get("owner_uuid", "unassigned"),
                    "movie": t.get("movie", "ld"),
                    "query_frames": list(t["query_frames"]),
                    "focus_xy": list(path[0]),
                    "draft_path": {"path_xy": path,
                                   "path_visible": list(o.get(
                                       "path_visible", [True] * len(path)))},
                    "draft_task": uuid,
                    "task_type": "review_path",
                    "stratum": "repair-path",
                    "why": ("confirm span: FULL root-to-tip or PARTIAL; "
                            "draft stroke shown, not redrawn"),
                    "completed": False,
                })
    for xid in sorted(cross):
        rid = f"{xid}-relink"
        if rid in have:
            continue
        lanes = {k: [list(q) for q in v]
                 for k, v in cross[xid].get("lanes", {}).items()}
        if len([v for v in lanes.values() if len(v) >= 2]) < 2:
            continue
        xs = [q[0] for v in lanes.values() for q in v]
        ys = [q[1] for v in lanes.values() for q in v]
        # Source frame/movie: find the task that created it, else skip.
        src = [t for t in tasks.values()
               if t.get("completed")]
        fid = movie = None
        for t in tasks.values():
            if str(t.get("uuid", "")) in xid or xid.replace(
                    "cross-", "") == str(t.get("uuid", "")):
                fid = int(t["query_frames"][0])
                movie = t.get("movie", "ld")
        if fid is None:  # fall back: task whose frame matches lanes best
            continue
        out.append({
            "uuid": rid,
            "owner_uuid": "unassigned",
            "movie": movie,
            "query_frames": [fid],
            "focus_xy": [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))],
            "draft_crossing": xid,
            "draft_lanes": lanes,
            "task_type": "review_crossing",
            "stratum": "repair-crossing",
            "why": ("re-trace both lanes from drafts; assign stable-tube "
                    "continuations where resolvable, else keep unresolved"),
            "completed": False,
        })
    return out


def germinated_clumps(movie: str, heat_model, device, n_frames: int = 24,
                      want: int = 4, tag: str = "rc",
                      movie_key: str = "m2") -> list[dict]:
    """Fresh owner tasks on clumps that visibly germinate (have tubes)."""
    from build_annotation_batch import (  # noqa: PLC0415
        SEARCH_R, RIM_EXCL, EDGE_M, tip_heat_for_frame,
    )
    reader = FrameReader(movie)
    try:
        n = len(reader)
        fids = [min(int(i * n / n_frames), n - 1) for i in range(n_frames)]
        cands = []
        for fid in fids:
            gray = reader.read(fid).frame
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            grains = [g for g in detect_grains(gray)
                      if EDGE_M < g[0] < w - EDGE_M
                      and EDGE_M < g[1] < h - EDGE_M][:40]
            if len(grains) < 2:
                continue
            heat = tip_heat_for_frame(heat_model, device, gray)
            germ = []
            for gx, gy, _s in grains:
                x0, x1 = max(0, int(gx - SEARCH_R)), min(w, int(gx + SEARCH_R))
                y0, y1 = max(0, int(gy - SEARCH_R)), min(h, int(gy + SEARCH_R))
                patch = heat[y0:y1, x0:x1]
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dist = np.hypot(xx - gx, yy - gy)
                ring = patch[(dist > RIM_EXCL) & (dist < SEARCH_R)]
                germ.append(float(ring.max()) if ring.size else 0.0)
            # Clump = >=2 grains within 150px, brightest germinating.
            for i, (gx, gy, _s) in enumerate(grains):
                # TOUCHING clump: center distance under ~2.5 grain radii.
                # Loose neighborhoods are not root-attachment tasks.
                near = [j for j, (ox, oy, _o) in enumerate(grains)
                        if j != i and math.hypot(ox - gx, oy - gy) < 80]
                # Mass veto: reject corners of giant masses (whole-clump
                # spec + 8-row table both fail there). H261 eye-check.
                mass = sum(1 for j, (ox, oy, _o) in enumerate(grains)
                           if j != i and math.hypot(ox - gx, oy - gy) < 250)
                # Table UI separates at most 8 balls; bigger clusters
                # stay a future dense-mask job, not a pile task.
                if 1 <= len(near) <= 7 and mass <= 10 and germ[i] >= 0.6:
                    cands.append({"fid": fid, "x": float(gx),
                                  "y": float(gy), "heat": germ[i],
                                  "members": len(near) + 1})
        cands.sort(key=lambda c: -c["heat"])
        picked, seen = [], []
        for c in cands:
            if any(abs(c["fid"] - s["fid"]) < 8000
                   and abs(c["x"] - s["x"]) < 150
                   and abs(c["y"] - s["y"]) < 150 for s in seen):
                continue
            picked.append(c)
            seen.append(c)
            if len(picked) >= want:
                break
        return [{
            "uuid": f"{tag}-{i:03d}",
            "owner_uuid": "unassigned",
            "movie": movie_key,
            "query_frames": [c["fid"]],
            "focus_xy": [c["x"], c["y"]],
            "task_type": "owner",
            "stratum": "repair-clump",
            "why": (f"germinated clump (~{c['members']} balls, tube heat "
                    f"{c['heat']:.2f}): separate every grain, mark each "
                    f"root exit and proximal tube"),
            "completed": False,
        } for i, c in enumerate(picked)]
    finally:
        reader.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3", default="/tmp/annotator_r3")
    ap.add_argument("--dt", default="/tmp/annotator_dt")
    ap.add_argument("--clump-movie", default=M2)
    ap.add_argument("--clump-movie-key", default="m2")
    ap.add_argument("--clump-tag", default="rc")
    ap.add_argument("--clump-frames", type=int, default=24)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--actor", default="repair")
    a = ap.parse_args()

    dt_tasks = dt_review_tips(Path(a.dt))
    store = AnnotationStore(Path(a.dt) / "annotations.db")
    try:
        for t in dt_tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
    finally:
        store.close()
    print(f"dt: +{len(dt_tasks)} review_tip", flush=True)

    tasks = r3_reviews(Path(a.r3))
    if a.weights:
        import torch  # noqa: PLC0415
        from build_annotation_batch import load_tip_model  # noqa: PLC0415
        model, device = load_tip_model(Path(a.weights))
        try:
            tasks += germinated_clumps(
                a.clump_movie, model, device, n_frames=a.clump_frames,
                tag=a.clump_tag, movie_key=a.clump_movie_key)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    store = AnnotationStore(Path(a.r3) / "annotations.db")
    try:
        from collections import Counter
        by: dict[str, int] = {}
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
            by[t["task_type"]] = by.get(t["task_type"], 0) + 1
    finally:
        store.close()
    print(f"r3: +{len(tasks)} {dict(by)}", flush=True)


if __name__ == "__main__":
    main()
