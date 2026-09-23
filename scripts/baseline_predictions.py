"""Turn earlier methods' saved outputs on the sparse movie into sparsetrack predictions.

- ``v29.19.3``: the early-September field-wide pipeline (all 42 owners), from
  runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3 (grain centres from path roots).
- ``tube_track+emergence_onset``: the 22 Sep state — rev19 ridge-walk lengths (model-only
  rows) and the rev18 operating onset config, for S3/S4/S6/S9/S11 only.

    .venv/bin/python scripts/baseline_predictions.py runs/sparsetrack/ld runs/sparsetrack/baselines
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sparsetrack import stack  # noqa: E402
from sparsetrack.evaluate import PRED_SCHEMA  # noqa: E402

V29 = REPO / "runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3"
REV19 = REPO / "runs/prototypes/v30/rev19_tracks.json"
REV18_ONSET = Path.home() / ".hermes/cache/scratch/rev18_onset_calib.json"
LEGACY = REPO / "benchmark/labels/legacy_v0.json"


def v29(meta: dict) -> dict:
    rows = {int(r["pollen_id"]): r for r in csv.DictReader(open(V29 / "summary.csv"))}
    series = defaultdict(list)
    for r in csv.DictReader(open(V29 / "measurements.csv")):
        series[int(r["pollen_id"])].append((int(r["source_frame"]), float(r["tube_length_px"] or 0.0)))
    grains = []
    # grain centre ~ one grain radius back from the path root along the tube's first
    # segment, at the earliest traced sample, converted to reference coordinates
    shifts, fpb = np.asarray(meta["shifts"]), meta["frames_per_bin"]
    first: dict[int, list] = {}
    for r in csv.DictReader(open(V29 / "centerlines.csv")):
        pid, frame, k = int(r["pollen_id"]), int(r["source_frame"]), int(r["point_index"])
        if pid in first and first[pid][0] < frame:
            continue
        if pid not in first or first[pid][0] > frame:
            first[pid] = [frame, {}]
        first[pid][1][k] = (float(r["source_x_px"]), float(r["source_y_px"]))
    pos = {}
    for pid, (frame, pts) in first.items():
        if 0 not in pts or 1 not in pts:
            continue
        (x0, y0), (x1, y1) = pts[0], pts[max(k for k in pts if k <= 4)]
        u = np.array([x1 - x0, y1 - y0]) / (np.hypot(x1 - x0, y1 - y0) + 1e-9)
        dx, dy = shifts[min(frame // fpb, len(shifts) - 1)]
        pos[pid] = (x0 - 13.0 * u[0] - dx, y0 - 13.0 * u[1] - dy)
    for pid, r in rows.items():
        if pid not in pos:
            continue
        onset = r.get("verified_germination_source_frame") or r.get("native_root_onset_source_frame")
        status = "emerged_within" if onset else ("unobservable" if "censored" in r["causal_status"] else "no_emergence_by_end")
        s = sorted(series[pid])
        grains.append({"x": pos[pid][0], "y": pos[pid][1], "status": status,
                       "onset_frame": int(float(onset)) if onset else None,
                       "length": {"frames": [f for f, _ in s], "px": [L for _, L in s]},
                       "source_id": f"P{pid:02d}", "source_status": r["causal_status"]})
    return {"schema": PRED_SCHEMA, "method": "v29.19.3 field pipeline (Sep)", "grains": grains,
            "match_radius_px": 20.0,
            "note": "grain centres estimated from path roots (the v28 grain tracks were deleted in cleanup)"}


def tube_track(meta: dict) -> dict:
    legacy = json.loads(LEGACY.read_text())
    by_name = {g["legacy_name"]: (gid, g) for gid, g in legacy["grains"].items()}
    tracks = json.loads(REV19.read_text())["grains"]
    onsets = json.loads(REV18_ONSET.read_text())["onset"] if REV18_ONSET.exists() else {}
    grains = []
    for name, data in tracks.items():
        if name not in by_name:
            continue
        gid, g = by_name[name]
        rows = sorted(data["rows"], key=lambda r: r["frame"])
        lengths = [(r["frame"], float(r["length_arc"] or 0.0)) for r in rows]
        onset = (onsets.get(name) or {}).get("onset")
        grains.append({"id": gid, "x": g["x"], "y": g["y"], "status": "emerged_within" if onset else "unobservable",
                       "onset_frame": onset, "length": {"frames": [f for f, _ in lengths], "px": [L for _, L in lengths]},
                       "source_id": name})
    return {"schema": PRED_SCHEMA, "method": "tube_track + emergence_onset (22 Sep)", "grains": grains,
            "note": "covers S3/S4/S6/S9/S11 only; length rows are model-only ridge-walk arc lengths"}


def main(cache_dir: str, out_dir: str) -> None:
    _, meta = stack.load(cache_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("v29_19_3", v29), ("tube_track_rev19", tube_track)):
        pred = fn(meta)
        (out / f"{name}.json").write_text(json.dumps(pred))
        print(f"{name}: {len(pred['grains'])} grains -> {out / (name + '.json')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
