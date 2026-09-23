"""Round-4 event batch: crossings, paths, clumps (P0A/P1).

Mines from recorded evidence, never invented geometry:
  - crossings: v29 centerline owner-pairs whose paths pass within
    12px on the same source frame (both >=20px long)
  - paths: long (>=80px) isolated owners (no neighbor within 30px)
    at a clear mid-life frame -> centerline tasks
  - clumps: FRST grain clusters (>=3 within 90px) on movie 2
    -> owner (grain/root separation) tasks
Tasks carry movie keys (m2/ld) for the multi-movie workbench.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))

from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from grain_detect import detect_grains  # noqa: E402

LOWDENS = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
MOVIE2_PATH = "/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4"


def load_paths(centerlines_csv: Path):
    """(frame -> {owner: [(x,y)...]}) polylines in source coords."""
    frames: dict[int, dict[int, list]] = defaultdict(dict)
    with open(centerlines_csv) as f:
        for row in csv.DictReader(f):
            o = int(row["pollen_id"])
            fr = int(row["source_frame"])
            frames[fr].setdefault(o, []).append(
                (float(row["source_x_px"]), float(row["source_y_px"])))
    return frames


def path_length(pts) -> float:
    return sum(float(np.hypot(b[0] - a[0], b[1] - a[1]))
               for a, b in zip(pts[:-1], pts[1:]))


def min_pair_dist(a, b) -> tuple[float, tuple, tuple]:
    best = (float("inf"), a[0], b[0])
    for pa in a[::2]:
        for pb in b[::2]:
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if d < best[0]:
                best = (d, pa, pb)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--centerlines", default=str(
        REPO / "runs/prototypes/v29/causal_growth_front/"
        "lowdens_full_v29_19_3/centerlines.csv"))
    ap.add_argument("--n-cross", type=int, default=6)
    ap.add_argument("--n-path", type=int, default=6)
    ap.add_argument("--n-clump", type=int, default=6)
    ap.add_argument("--actor", default="builder")
    a = ap.parse_args()

    frames = load_paths(Path(a.centerlines))
    fids = sorted(frames)
    tasks: list[dict] = []

    # Crossings: close-approach pairs, spread across the movie.
    cross_cands = []
    for i, fr in enumerate(fids[::max(1, len(fids) // 30)]):
        owners = [(o, p) for o, p in frames[fr].items() if len(p) >= 4]
        for j, (o1, p1) in enumerate(owners):
            if path_length(p1) < 20:
                continue
            for o2, p2 in owners[j + 1:]:
                if path_length(p2) < 20:
                    continue
                d, pa, pb = min_pair_dist(p1, p2)
                if d < 12:
                    cross_cands.append({
                        "fid": fr, "owners": (o1, o2),
                        "focus": ((pa[0] + pb[0]) / 2,
                                  (pa[1] + pb[1]) / 2),
                        "gap": d,
                    })
    picked = []
    for c in sorted(cross_cands, key=lambda c: c["gap"]):
        if len(picked) >= a.n_cross:
            break
        if all(abs(c["fid"] - p["fid"]) > 3000 for p in picked):
            picked.append(c)
    for i, c in enumerate(picked):
        tasks.append({
            "uuid": f"r4-x{i:02d}", "owner_uuid": "unassigned",
            "movie": "ld", "query_frames": [c["fid"]],
            "focus_xy": [round(c["focus"][0], 1), round(c["focus"][1], 1)],
            "task_type": "crossing",
            "stratum": "r4-crossing",
            "why": (f"owners {c['owners'][0]}/{c['owners'][1]} pass "
                    f"{c['gap']:.1f}px apart (v29 centerlines)"),
            "geometry_type": "path", "class_scope": "crossing",
            "completeness": "partial", "completed": False,
        })

    # Paths: long isolated owners at mid-life frames.
    path_cands = []
    by_owner: dict[int, list[tuple[int, list]]] = defaultdict(list)
    for fr in fids:
        for o, p in frames[fr].items():
            by_owner[o].append((fr, p))
    for o, series in by_owner.items():
        long_enough = [(fr, p) for fr, p in series
                       if path_length(p) >= 80]
        if not long_enough:
            continue
        fr, p = long_enough[len(long_enough) // 2]
        others = [q for oo, q in frames[fr].items() if oo != o]
        dmin = min([min_pair_dist(p, q)[0] for q in others
                    if len(q) >= 2] or [float("inf")])
        if dmin < 30:
            continue
        mid = p[len(p) // 2]
        path_cands.append({"fid": fr, "owner": o, "focus": mid,
                           "length": path_length(p)})
    for i, c in enumerate(sorted(path_cands, key=lambda c: -c["length"]
                                 )[:a.n_path]):
        tasks.append({
            "uuid": f"r4-p{i:02d}", "owner_uuid": "unassigned",
            "movie": "ld", "query_frames": [c["fid"]],
            "focus_xy": [round(c["focus"][0], 1), round(c["focus"][1], 1)],
            "task_type": "centerline",
            "stratum": "r4-path",
            "why": (f"owner {c['owner']} isolated tube "
                    f"{c['length']:.0f}px long"),
            "geometry_type": "path", "class_scope": "tube",
            "completeness": "complete", "completed": False,
        })

    # Clumps on movie 2: >=3 grains within 90px.
    reader = FrameReader(MOVIE2_PATH)
    try:
        n = len(reader)
        n_cl = 0
        used: list[tuple[float, float]] = []  # static field: dedup ACROSS frames
        for i in range(12):
            if n_cl >= a.n_clump:
                break
            fid = min(int((i + 0.5) * n / 12), n - 1)
            res = reader.read(fid)
            gray = (res.frame if res.frame.ndim == 2 else
                    cv2.cvtColor(res.frame, cv2.COLOR_BGR2GRAY))
            gs = [(g[0], g[1]) for g in detect_grains(gray)]
            for gx, gy in gs:
                cluster = [(ox, oy) for ox, oy in gs
                           if abs(ox - gx) < 90 and abs(oy - gy) < 90]
                if len(cluster) >= 3 and not any(
                        abs(gx - ux) < 150 and abs(gy - uy) < 150
                        for ux, uy in used):
                    used.append((gx, gy))
                    # Focus the member grain nearest the centroid (H-eye):
                    # a raw centroid can sit in empty space between balls.
                    cx = sum(p[0] for p in cluster) / len(cluster)
                    cy = sum(p[1] for p in cluster) / len(cluster)
                    bx, by = min(cluster, key=lambda p: abs(p[0] - cx)
                                 + abs(p[1] - cy))
                    tasks.append({
                        "uuid": f"r4-c{n_cl:02d}", "owner_uuid": "unassigned",
                        "movie": "m2", "query_frames": [fid],
                        "focus_xy": [round(bx, 1), round(by, 1)],
                        "task_type": "owner",
                        "stratum": "r4-clump",
                        "why": (f"{len(cluster)} grains within 90px: "
                                "separate balls A/B/C + each tube start"),
                        "geometry_type": "point", "class_scope": "grain",
                        "completeness": "partial", "completed": False,
                    })
                    n_cl += 1
                    if n_cl >= a.n_clump:
                        break
    finally:
        reader.close()

    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        by_stratum: dict[str, int] = {}
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
            by_stratum[t["stratum"]] = by_stratum.get(t["stratum"], 0) + 1
        print(f"wrote {len(tasks)} tasks: {by_stratum}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
