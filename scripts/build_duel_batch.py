"""Build route-duel tasks: the model's pick vs the best-coverage alternative.

For each FULL-path event: duel = (movie winner route, best-coverage
route among the same candidate set). Only pairs where the answers
differ and the good side has coverage >= 0.30 (a real better answer,
not less-bad). Display sides randomized (seeded). Appends to the
project dir (mask batch lives there too).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--candidates", required=True,
                    help="candidates.json of the model run to second-guess")
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="v30q")
    ap.add_argument("--actor", default="v30duel")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from mine_hard_routes import coverage
    from tubetracker.annotation_store import AnnotationStore
    from prototypes.v30_video_apex.targets import samples_from_snapshot

    cands = json.loads(Path(a.candidates).read_text())
    by: dict[str, list] = {}
    for x in cands:
        by.setdefault(str(x["owner_id"]), []).append(x)
    paths = {s.obs_uuid: s for s in samples_from_snapshot(a.snapshot)
             if s.kind == "path_tip"}
    rng = np.random.default_rng(a.seed)
    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    n = 0
    try:
        for own, xs in sorted(by.items()):
            s = paths.get(own)
            if s is None or not s.path_xy:
                continue
            scored = []
            for x in xs:
                poly = x.get("polyline_native") or []
                if len(poly) < 2:
                    continue
                scored.append((coverage(poly, s.path_xy), x))
            if not scored:
                continue
            scored.sort(key=lambda z: -z[0])
            best_cov, best = scored[0]
            win = next((x for x in xs if x.get("selected")), None)
            if win is None or best_cov < 0.30:
                continue
            wpoly = win.get("polyline_native") or []
            wcov = coverage(wpoly, s.path_xy) if len(wpoly) >= 2 else 0.0
            if win["route_id"] == best["route_id"] or wcov >= best_cov:
                continue  # model already right: no duel to judge
            pair = [(best["route_id"], best["polyline_native"]),
                    (win["route_id"], wpoly)]
            if rng.random() < 0.5:
                pair = pair[::-1]
            (id_a, pa), (id_b, pb) = pair
            mid = (np.asarray(pa + pb).reshape(-1, 2)).mean(axis=0)
            store.save("task", f"{a.tag}-{n:03d}", {
                "uuid": f"{a.tag}-{n:03d}", "owner_uuid": "unassigned",
                "query_frames": [int(s.source_frame)],
                "task_type": "route_duel", "movie": s.movie,
                "focus_xy": [float(mid[0]), float(mid[1])],
                "draft_lanes": {
                    "A": [[float(q[0]), float(q[1])] for q in pa],
                    "B": [[float(q[0]), float(q[1])] for q in pb]},
                "lane_truth": {"A": id_a, "B": id_b},
                "gold_route": {"route_id": best["route_id"],
                               "polyline": best["polyline_native"]},
                "completed": False, "stratum": f"{a.tag}-duel",
                "priority": 10.0,
                "why": "Which curve follows the tube? Click near the "
                       "better one (or Neither).",
            }, actor=a.actor)
            print(f"{a.tag}-{n:03d}: {s.obs_uuid[:12]} "
                  f"{id_a}(cov {best_cov:.2f}) vs "
                  f"{id_b}(cov {wcov:.2f}, model pick)")
            n += 1
    finally:
        store.close()
    print(f"{a.tag}: +{n} route_duel tasks -> {proj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
