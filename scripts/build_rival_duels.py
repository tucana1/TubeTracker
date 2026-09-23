"""Mine winner-vs-runnerup duels from a movie run (rev7 batch).

For each event: the selected route vs the highest-tip_score unselected
route = two real alternatives, no truth needed (human preference is
the label). Skips pairs already judged in the snapshot (same pair_key
scheme as build_v30_snapshot). Writes a FRESH project dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def pair_key(movie, frame, la, lb) -> str:
    return hashlib.sha256(repr(sorted([
        (movie, int(frame),
         tuple((round(float(q[0]), 1), round(float(q[1]), 1)) for q in la)),
        (movie, int(frame),
         tuple((round(float(q[0]), 1), round(float(q[1]), 1)) for q in lb)),
    ])).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="v30w")
    ap.add_argument("--actor", default="v30duel")
    a = ap.parse_args()

    from tubetracker.annotation_store import AnnotationStore

    judged = set()
    if a.snapshot:
        dp = Path(a.snapshot) / "duels.json"
        if dp.exists():
            judged = {d.get("pair_key", "")
                      for d in json.loads(dp.read_text())}
    cands = json.loads(Path(a.candidates).read_text())
    # FULL paths (where they exist) pick the rival: the best-coverage
    # non-winner teaches more than the tip_score runner-up. Without a
    # path, fall back to runner-up.
    path_by_event: dict[str, list] = {}
    if a.snapshot:
        op = Path(a.snapshot) / "observations.json"
        if op.exists():
            for o in json.loads(op.read_text()):
                if o.get("path_xy") and len(o["path_xy"]) >= 2:
                    path_by_event[str(o.get("obs_uuid", ""))] = [
                        [float(q[0]), float(q[1])]
                        for q in o["path_xy"]]
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from mine_hard_routes import coverage
    except ImportError:
        coverage = None  # type: ignore[assignment]
    by_event: dict[str, list] = {}
    for c in cands:
        by_event.setdefault(str(c["candidate_id"].rsplit(":", 1)[0]),
                            []).append(c)
    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return 1
    proj.mkdir(parents=True)
    store = AnnotationStore(proj / "annotations.db")
    n = 0
    try:
        for key, rows in sorted(by_event.items()):
            win = [r for r in rows if r.get("selected")]
            if not win:
                continue
            win = win[0]
            alts = [r for r in rows if not r.get("selected")]
            if not alts:
                continue
            obs_uuid = str(win.get("owner_id", ""))
            truth = path_by_event.get(obs_uuid)
            if truth is not None and coverage is not None:
                scored = sorted(
                    alts,
                    key=lambda r: -coverage(r["polyline_native"], truth))
                alt = scored[0]
                alt_kind = "best-coverage"
            else:
                alt = sorted(alts,
                             key=lambda r: -float(
                                 r.get("tip_score", 0)))[0]
                alt_kind = "runner-up"
            la = [[float(q[0]), float(q[1])]
                  for q in win["polyline_native"]]
            lb = [[float(q[0]), float(q[1])]
                  for q in alt["polyline_native"]]
            if len(la) < 2 or len(lb) < 2:
                continue
            movie = str(win.get("movie_id", "ld"))
            frame = int(win.get("source_frame", -1))
            if pair_key(movie, frame, la, lb) in judged:
                print(f"skip {key}: already judged")
                continue
            import numpy as np
            mid = (np.asarray(la + lb).reshape(-1, 2)).mean(axis=0)
            uuid = f"{a.tag}-{n:03d}"
            store.save("task", uuid, {
                "uuid": uuid, "owner_uuid": "unassigned",
                "query_frames": [frame], "task_type": "route_duel",
                "movie": movie,
                "focus_xy": [float(mid[0]), float(mid[1])],
                "draft_lanes": {"A": la, "B": lb},
                "lane_truth": {"A": str(win.get("route_id", "")),
                               "B": str(alt.get("route_id", ""))},
                "completed": False, "stratum": f"{a.tag}-duel",
                "priority": 10.0,
                "why": "Which curve follows the tube? Click near the "
                       "better one (or Neither).",
            }, actor=a.actor)
            print(f"{uuid}: {key} {win.get('route_id')} vs "
                  f"{alt.get('route_id')} [{alt_kind}]")
            n += 1
    finally:
        store.close()
    print(f"{a.tag}: +{n} route_duel tasks -> {proj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
