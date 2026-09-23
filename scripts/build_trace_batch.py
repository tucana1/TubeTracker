"""Build fresh tube-trace tasks: 6 new centerline tracings (rev6 expansion).

Mines NEW owned events: FRST grains paired with a nearby v1 peak
(30-160px, plausible tube), excluding anything within 40px of an
existing confirmed path or banked tip, spread across frames and the
field. Each task is a plain centerline trace (click dots ball->tip).

Writes a FRESH project dir (never touches live projects).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--movie", action="append", default=[])
    ap.add_argument("--v1-weights", default="")
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--tag", default="v30t")
    ap.add_argument("--actor", default="v30trace")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", default="",
                    help="comma-separated source frames to mine")
    a = ap.parse_args()

    from prototypes.v30_video_apex import routes as R
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore

    movies = {}
    for spec in a.movie:
        key, path = spec.split("=", 1)
        movies[key] = path
    readers = {k: FrameReader(p) for k, p in movies.items()}

    import torch
    v1, dev = R.load_v1(a.v1_weights) if a.v1_weights else (None, None)

    # Existing geometry to avoid (paths + banked tips per movie+frame).
    paths, tips = [], []
    for s in samples_from_snapshot(a.snapshot):
        if s.movie != "ld":
            continue
        if s.kind == "path_tip" and len(s.path_xy) >= 2:
            paths.append((int(s.source_frame), np.asarray(s.path_xy)))
        if s.tip_xy and len(s.tip_xy) == 2:
            tips.append((int(s.source_frame), np.asarray(s.tip_xy)))

    frames = [int(f) for f in a.frames.split(",") if f.strip()] or \
        [15000, 30000, 42000, 48300, 50400, 60000]
    cands = []
    for fr in frames:
        try:
            gray = np.asarray(readers["ld"].read(fr).frame)
        except OSError:
            print(f"skip frame {fr}: decode failed")
            continue
        gray = gray[:, :, 0] if gray.ndim == 3 else gray
        sys.path.insert(0, str(REPO / "prototypes"))
        from timesfm_tip_forecast.grain_detect import detect_grains
        grains = [(float(x), float(y)) for x, y, _ in detect_grains(gray)]
        heat = R.v1_tip_heat(v1, dev, gray) if v1 is not None else None
        peaks = [(float(x), float(y), float(h))
                 for x, y, h in (R.heat_peaks(heat)
                                 if heat is not None else [])]
        for gx, gy in grains:
            near = [(x, y, h) for x, y, h in peaks
                    if 30.0 <= np.hypot(x - gx, y - gy) <= 120.0]
            if not near:
                continue
            near.sort(key=lambda z: -z[2])
            x, y, h = near[0]
            if any(np.hypot(x - px, y - py) < 25.0
                   for _, _, _, px, py, _ in cands):
                continue  # peak already claimed: nearest grain wins
            mid = np.array([(gx + x) / 2, (gy + y) / 2])
            if any(f == fr and np.hypot(*(p - mid).T).min() < 40.0
                   for f, p in paths):
                continue
            if any(f == fr and np.hypot(x - t[0], y - t[1]) < 40.0
                   for f, t in tips):
                continue
            cands.append((fr, gx, gy, x, y, h))
    # Spread across frames: round-robin by frame, best heat first.
    by_fr: dict[int, list] = {}
    for cd in sorted(cands, key=lambda z: -z[5]):
        by_fr.setdefault(cd[0], []).append(cd)
    picked = []
    while len(picked) < a.n and any(by_fr.values()):
        for fr in sorted(by_fr):
            if by_fr[fr] and len(picked) < a.n:
                # field spread: skip if near an already-picked grain
                cd = by_fr[fr].pop(0)
                if all(np.hypot(cd[1] - p[1], cd[2] - p[2]) > 120.0
                       for p in picked):
                    picked.append(cd)
    proj = Path(a.project_dir)
    if proj.exists():
        print(f"refusing to reuse {proj}")
        return 1
    proj.mkdir(parents=True)
    store = AnnotationStore(proj / "annotations.db")
    try:
        for i, (fr, gx, gy, x, y, h) in enumerate(picked):
            store.save("task", f"{a.tag}-{i:03d}", {
                "uuid": f"{a.tag}-{i:03d}", "owner_uuid": "unassigned",
                "query_frames": [int(fr)], "task_type": "centerline",
                "movie": "ld", "focus_xy": [float(gx), float(gy)],
                # Rendered as the hollow yellow draft circle (guidance,
                # never truth); kept under its own key for provenance.
                "draft_xy": [float(x), float(y)],
                "hint_tip_xy": [float(x), float(y)],
                "completed": False, "stratum": f"{a.tag}-trace",
                "priority": 10.0,
                "why": "NEW tube: trace dots ball->tip along the "
                       "middle to the far end. The hollow yellow circle "
                       "is a GUESS at the tip direction, not truth — "
                       "follow the real tube, not the circle.",
            }, actor=a.actor)
    finally:
        store.close()
    for p in picked:
        print(f"  frame={p[0]} grain=({p[1]:.0f},{p[2]:.0f}) "
              f"hint=({p[3]:.0f},{p[4]:.0f}) heat={p[5]:.2f}")
    print(f"{a.tag}: +{len(picked)} centerline tasks -> {proj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
