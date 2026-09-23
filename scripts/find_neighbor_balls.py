"""Find frames where two grain detections share one camera view.

rev8 step 3 asks to "prove whole-instance learning on a few
neighbouring tubes": the query-swap test needs two DIFFERENT grains
whose tubes are visible in the same crop, so a single frame/crop can
be painted for both. This script detects grains (FRST, no weights) on
requested frames and reports close pairs, where "close" means both
fit inside one zoom-5 view (min_dist .. max_dist px apart).

Output: JSON {frame_ref: [{a, b, dist, known_a, known_b}, ...]} where
`known_*` names the observation whose focus sits on that detection, so
the batch builder can attach the traced guide path where one exists.
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
    ap.add_argument("--movie", action="append", default=[],
                    help="movie mapping key=path (repeatable)")
    ap.add_argument("--frames", required=True,
                    help="comma-separated movie:frame entries")
    ap.add_argument("--snapshot", default="",
                    help="optional snapshot: labels known foci")
    ap.add_argument("--min-dist", type=float, default=40.0)
    ap.add_argument("--max-dist", type=float, default=170.0)
    ap.add_argument("--per-frame", type=int, default=2,
                    help="keep at most this many pairs per frame")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prototypes.timesfm_tip_forecast.grain_detect import detect_grains
    from tubetracker.annotation_frames import FrameReader

    movies: dict[str, str] = {}
    for spec in a.movie:
        k, _, p = spec.partition("=")
        movies[k] = p
    known: dict[tuple[str, int], list] = {}
    if a.snapshot:
        obs = json.loads(
            (Path(a.snapshot) / "observations.json").read_text())
        for o in obs:
            mk = str(o.get("source_movie") or o.get("movie")
                     or o.get("movie_uuid") or "")
            fr = int(o.get("source_frame", -1))
            f = o.get("focus_xy") or []
            if fr >= 0 and len(f) == 2:
                known.setdefault((mk, fr), []).append(
                    {"obs": str(o.get("obs_uuid")), "xy": [float(f[0]),
                                                           float(f[1])]})

    readers: dict[str, FrameReader] = {}
    out: dict[str, list] = {}
    try:
        for spec in [s for s in a.frames.split(",") if s]:
            mk, _, fr_s = spec.partition(":")
            fr = int(fr_s)
            if mk not in readers:
                readers[mk] = FrameReader(movies[mk])
            frame = readers[mk].read(fr)
            img = frame.frame
            gray = img[:, :, 0] if img.ndim == 3 else img
            grains = detect_grains(gray)
            pairs = []
            for i in range(len(grains)):
                for j in range(i + 1, len(grains)):
                    xa, ya, sa = grains[i]
                    xb, yb, sb = grains[j]
                    d = float(np.hypot(xa - xb, ya - yb))
                    if not (a.min_dist <= d <= a.max_dist):
                        continue
                    ka = kb = ""
                    for krec in known.get((mk, fr), []):
                        if np.hypot(*(np.asarray(krec["xy"]) -
                                      np.array([xa, ya]))) < 12.0:
                            ka = krec["obs"]
                        if np.hypot(*(np.asarray(krec["xy"]) -
                                      np.array([xb, yb]))) < 12.0:
                            kb = krec["obs"]
                    pairs.append({
                        "a": [round(xa, 1), round(ya, 1), round(sa, 3)],
                        "b": [round(xb, 1), round(yb, 1), round(sb, 3)],
                        "dist": round(d, 1),
                        "known_a": ka, "known_b": kb,
                        "score_sum": round(sa + sb, 3)})
            pairs.sort(key=lambda r: -r["score_sum"])
            if a.per_frame > 0:
                pairs = pairs[:a.per_frame]
            out[f"{mk}:{fr}"] = pairs
            print(f"{mk}:{fr}: {len(grains)} grains, "
                  f"{len(pairs)} close pair(s)")
            for pr in pairs:
                print(f"   a={pr['a'][:2]} b={pr['b'][:2]} "
                      f"d={pr['dist']} known=({pr['known_a'] or '-'},"
                      f"{pr['known_b'] or '-'})")
    finally:
        for r in readers.values():
            r.close()
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
