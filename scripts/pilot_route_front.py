#!/usr/bin/env python3
"""Route/front pilot on confirmed FULL paths (P3A first light, H265).

Oracle route (human-confirmed FULL centerline) + image front: sample the
frozen tip heat along the route arclength to form a 1D front profile
q(s). No training; demonstrates the route/front decomposition and the
oracle-vs-predicted evaluation split on real data.
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.cnn_prototype import (  # noqa: E402
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)

MOVIES = {
    "ld": "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4",
    "m2": "/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4",
}


def arclength_profile(path, heat, step=1.0):
    pts = np.asarray(path, float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    ss = np.arange(0, total, step)
    xs = np.interp(ss, s, pts[:, 0]).clip(0, heat.shape[1] - 1)
    ys = np.interp(ss, s, pts[:, 1]).clip(0, heat.shape[0] - 1)
    vals = heat[ys.astype(int), xs.astype(int)]
    return ss, xs, ys, vals, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/tmp/annotator_r3")
    ap.add_argument("--spans", nargs="+", default=["r4-p00"])
    ap.add_argument("--checkpoint", default=(
        "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"))
    ap.add_argument("--out", default="/tmp/route_front_pilot.png")
    a = ap.parse_args()

    con = sqlite3.connect(str(Path(a.project) / "annotations.db"))
    tasks = {u: json.loads(d) for u, d in con.execute(
        "SELECT uuid, data FROM entities WHERE kind='task'")}
    obs = {u: json.loads(d) for u, d in con.execute(
        "SELECT uuid, data FROM entities WHERE kind='observation'")}
    con.close()
    by_task = {}
    for _u, _o in obs.items():
        if _o.get("task_uuid"):
            by_task[_o["task_uuid"]] = _o

    device = torch.device("cpu")
    model, _ = load_cnn_checkpoint(a.checkpoint, device=device)
    model.eval()

    panels = []
    for span in a.spans:
        t = tasks[span + "-span"]
        o = by_task[span + "-span"]
        fid = int(t["query_frames"][0])
        reader = FrameReader(MOVIES[t.get("movie", "ld")])
        try:
            frame = reader.read(fid).frame
        finally:
            reader.close()
        gray = (frame if frame.ndim == 2 else
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        with torch.no_grad():
            heat = np.asarray(
                predict_heatmaps_tiled(model, frame, device)[1])
        path = o["path_xy"]
        ss, xs, ys, vals, total = arclength_profile(path, heat)
        front = ss[int(np.argmax(vals))]
        end = np.asarray(o["direct_xy"], float)
        # Overlay: route polyline + heat argmax-on-route + confirmed end.
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                     (0, 255, 255), 2)
        fi = int(np.argmax(vals))
        cv2.circle(vis, (int(xs[fi]), int(ys[fi])), 9, (255, 0, 255), 2)
        cv2.drawMarker(vis, (int(end[0]), int(end[1])), (0, 255, 0),
                       cv2.MARKER_CROSS, 24, 2)
        x0b = max(0, int(end[0]) - 200)
        crop = vis[:, x0b:x0b + 400]
        # Profile strip: q(s) with front + true-end markers.
        strip = np.zeros((90, 400, 3), np.uint8)
        if len(ss) > 1:
            qx = (ss / ss[-1] * 399).astype(int)
            qy = 80 - (np.clip(vals, 0, 1) * 72).astype(int)
            for i in range(len(ss) - 1):
                cv2.line(strip, (qx[i], qy[i]), (qx[i + 1], qy[i + 1]),
                         (255, 0, 255), 1)
            cv2.line(strip, (qx[fi], 0), (qx[fi], 89), (255, 0, 255), 1)
            cv2.line(strip, (399, 0), (399, 89), (0, 255, 0), 2)
        cv2.putText(crop, f"{span} L={total:.0f}px front-s={front:.0f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        panels.append(np.vstack([crop, strip]))
    sheet = np.hstack(panels)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out} {sheet.shape}", flush=True)


if __name__ == "__main__":
    main()
