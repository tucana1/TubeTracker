"""Score tip-CNN predictions against human labels (H240).

For each completed task:
  - visible labels: distance from the human click to the nearest
    significant tip-heat peak (model-human agreement)
  - can't-tell labels: whether the model also stays silent (agreement)
    or fires anyway (model hallucinating)

Writes runs/prototypes/timesfm/label_audit_r2/{scores.csv,summary.json}
plus a disagreement review sheet (eye-check candidates).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))

from scripts.build_annotation_batch import (  # noqa: E402
    load_tip_model, tip_heat_for_frame, significant_peaks,
    RIM_EXCL, SEARCH_R, NEAR_R,
)

AGREE_PX = 15.0  # human click within this of a peak = agreement


def load_project(project_dir: Path):
    con = sqlite3.connect(str(project_dir / "annotations.db"))
    try:
        tasks = {u: json.loads(d) for u, d in con.execute(
            "SELECT uuid, data FROM entities WHERE kind='task'").fetchall()}
        obs = {u: json.loads(d) for u, d in con.execute(
            "SELECT uuid, data FROM entities WHERE kind='observation'"
        ).fetchall()}
    finally:
        con.close()
    return tasks, obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--weights", default=str(
        REPO / "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"))
    ap.add_argument("--out-dir", default=str(
        REPO / "runs/prototypes/timesfm/label_audit_r2"))
    a = ap.parse_args()

    from tubetracker.annotation_frames import FrameReader
    tasks, obs = load_project(Path(a.project_dir))
    # P0 repair: join observations to tasks via explicit task_uuid
    # (backfilled + migrated); task-derived IDs only as fallback.
    by_task: dict[str, dict] = {}
    for u, o in obs.items():
        if o.get("task_uuid"):
            by_task[o["task_uuid"]] = o
    model, device = load_tip_model(Path(a.weights))
    reader = FrameReader(a.movie)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows = []
        heat_cache: dict[int, np.ndarray] = {}
        for uuid in sorted(tasks):
            t = tasks[uuid]
            if not t.get("completed"):
                continue
            if t.get("resolution") == "unresolvable":
                continue  # escape hatch: never train on forced tangles
            fid = int(t["query_frames"][0])
            fx, fy = t["focus_xy"]
            if fid not in heat_cache:
                res = reader.read(fid)
                gray = (res.frame if res.frame.ndim == 2 else
                        cv2.cvtColor(res.frame, cv2.COLOR_BGR2GRAY))
                heat_cache[fid] = (gray, tip_heat_for_frame(
                    model, device, gray))
            gray, heat = heat_cache[fid]
            h, w = gray.shape
            x0, x1 = max(0, int(fx - SEARCH_R)), min(w, int(fx + SEARCH_R))
            y0, y1 = max(0, int(fy - SEARCH_R)), min(h, int(fy + SEARCH_R))
            patch = heat[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = np.hypot(xx - fx, yy - fy)
            masked = np.where(((dist > RIM_EXCL) & (dist < SEARCH_R)),
                              patch, 0.0)
            peaks = [(x0 + px, y0 + py)
                     for px, py in significant_peaks(masked, 0.6)]
            near = [(px, py) for px, py in peaks
                    if abs(px - fx) < NEAR_R and abs(py - fy) < NEAR_R]
            key = f"obs-{uuid}-0"
            alt = f"obs-{uuid}"
            o = by_task.get(uuid, obs.get(key, obs.get(alt)))
            if o is None or o.get("direct_state") != "direct_visible":
                state = "cant_tell"
                agree = len(near) == 0  # both silent = agreement
                dist_px = ""
            else:
                state = "visible"
                hx, hy = o["direct_xy"]
                dmin = min([float(np.hypot(px - hx, py - hy))
                            for px, py in near] + [float("inf")])
                dist_px = round(dmin, 1)
                agree = bool(dmin <= AGREE_PX)
            rows.append({
                "task": uuid, "stratum": t.get("stratum", "?"),
                "frame": fid, "state": state,
                "n_near_peaks": len(near),
                "dist_px": dist_px, "agree": int(agree),
                "why": t.get("why", ""),
            })
        import csv
        with open(out / "scores.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        vis = [r for r in rows if r["state"] == "visible"]
        ct = [r for r in rows if r["state"] == "cant_tell"]
        summary = {
            "n_tasks": len(rows),
            "n_visible": len(vis),
            "n_cant_tell": len(ct),
            "visible_agree_rate": round(
                sum(r["agree"] for r in vis) / max(1, len(vis)), 3),
            "visible_median_dist_px": round(float(np.median(
                [r["dist_px"] for r in vis
                 if r["dist_px"] != ""])), 1) if vis else None,
            "cant_tell_model_silent_rate": round(
                sum(r["agree"] for r in ct) / max(1, len(ct)), 3),
            "disagreements": [r["task"] for r in rows if not r["agree"]],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
