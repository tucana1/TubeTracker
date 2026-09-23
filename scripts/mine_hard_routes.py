"""Mine hard route negatives from REAL proposals (rev6 Pkg3).

For each path_tip sample: generate the 25 proposals (FRST root +
blind fan, same as the movie runner), score each against the FULL
confirmed path (coverage within 5px + tip error), and save the
confusing kind — tip-near but body-poor — as route-head negatives.
Rotations are generic; these are the model's actual failure mode
(e.g. p05's p1s: tip 0.4px, coverage 0.17).

Output JSON: {entry_id: [{route_id, polyline_native, coverage_5px,
tip_err_px, hard: bool}]}. No training here; the train script consumes
it via --hard-routes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def coverage(poly, gold, tol=5.0) -> float:
    p = np.asarray(poly, float)
    g = np.asarray(gold, float)
    if len(p) < 2 or len(g) < 2:
        return 0.0
    seg = g[1:] - g[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1]).clip(min=1e-9)
    hits = 0
    for q in p:
        w = (((q - g[:-1]) * seg).sum(axis=1) / (seglen ** 2))
        proj = g[:-1] + np.clip(w, 0, 1)[:, None] * seg
        if float(np.hypot(*(proj - q).T).min()) <= tol:
            hits += 1
    return hits / len(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--movie", action="append", default=[],
                    help="KEY=path (repeatable)")
    ap.add_argument("--v1-weights", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prototypes.v30_video_apex import routes as R
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    from prototypes.v30_video_apex.route_supervision import wrong_route_certificate
    from tubetracker.annotation_frames import FrameReader

    movies = {}
    for spec in a.movie:
        key, path = spec.split("=", 1)
        movies[key] = path
    readers = {k: FrameReader(p) for k, p in movies.items()}

    try:
        import torch
        dev = torch.device("cpu")
        v1 = None
        if a.v1_weights:
            v1, dev = R.load_v1(a.v1_weights)
    except Exception as e:
        print(f"no v1 heat ({e}); peaks from fan only")
        v1, dev = None, None

    out = {}
    n_hard = 0
    for s in samples_from_snapshot(a.snapshot):
        if s.kind != "path_tip" or not s.path_xy:
            continue
        reader = readers.get(s.movie)
        if reader is None:
            continue
        gray = reader.read(int(s.source_frame)).frame
        gray = gray[:, :, 0] if gray.ndim == 3 else gray
        anchor = (float(s.path_xy[0][0]), float(s.path_xy[0][1]))
        root_xy, _ = R.auto_root(np.asarray(gray), anchor)
        heat = R.v1_tip_heat(v1, dev,
                             np.asarray(gray)) if v1 is not None else None
        peaks = [p for p in (R.heat_peaks(heat) if heat is not None else [])
                 if np.hypot(p[0] - root_xy[0], p[1] - root_xy[1]) > 26.0]
        proposals = R.propose_routes(root_xy, peaks)
        tip = np.asarray(s.tip_xy, float)
        rows = []
        for pr in proposals:
            poly = np.asarray(pr["polyline"], float)
            cov = coverage(poly, s.path_xy)
            terr = float(np.hypot(*(poly - tip).T).min())
            certificate = wrong_route_certificate(s, poly)
            hard = bool(terr < 15.0 and certificate["certified_wrong_route"])
            n_hard += int(hard)
            rows.append({"route_id": pr["route_id"],
                         "polyline_native": poly.tolist(),
                         "coverage_5px": cov, "tip_err_px": terr,
                         "hard": hard,
                         "certified_wrong_route": certificate["certified_wrong_route"],
                         "certificate": certificate})
        rows.sort(key=lambda d: -d["coverage_5px"])
        out[s.entry_id] = rows
        print(f"{s.entry_id}: {len(rows)} proposals, "
              f"{sum(1 for d in rows if d['hard'])} hard, "
              f"best coverage={rows[0]['coverage_5px']:.2f}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"mined -> {a.out}: {len(out)} events, {n_hard} hard negatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
