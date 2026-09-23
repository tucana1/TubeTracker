"""Convert the pre-reset human answers on the sparse movie into a benchmark file.

Sources (all human, read-only): the rev14analysis annotation project — cf70's
absence/tip/FULL-path answers, the rev18 exit-to-apex traces and "no tube" answers
on S3/S4/S6/S9/S11, the rev17 window verdicts, and G0's refined bracket. These were
all judged on single compressed frames, so this file is a development reference,
not the benchmark: the grains must also be labelled blind in the new tool.

Positions are converted to the cache's reference coordinates (raw - shift[bin]) and
each grain is mapped to the nearest census grain within 12 px.

    .venv/bin/python scripts/build_legacy_benchmark.py runs/sparsetrack/ld benchmark/labels/legacy_v0.json
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sparsetrack import stack  # noqa: E402

PROJECT = Path.home() / "Documents/TubeTracker-annotator-projects/rev14analysis/annotations.db"
CF70 = "ld|review-53b55d8191864f77b5ed91beef56cf70"
# grain: (raw x, raw y, frame of that position, owner id in the DB)
GRAINS = {
    "cf70": (914.95, 492.22, 51120, CF70),
    "S3": (490.66, 252.57, 18500, "rev17-S3"),
    "S4": (1085.39, 870.02, 14500, "rev17-S4"),
    "S6": (202.11, 744.91, 22500, "rev17-S6"),
    "S9": (883.91, 187.22, 10500, "rev17-S9"),
    "S11": (631.81, 642.4, 20000, "rev17-S11"),
    "G0": (535.1, 640.1, 26000, "survey-G0-isolated"),
}
# human brackets (last absent, first visible], LEDGER H486 / DB germination_event records
BRACKETS = {"cf70": (300, 6000), "S3": (10000, 13000), "S4": (8500, 9000), "S6": (7000, 12000),
            "S9": (4000, 5000), "S11": (12000, 15000), "G0": (8000, 8050)}


def main(cache_dir: str, out_path: str) -> None:
    bins, meta = stack.load(cache_dir)
    fpb, shifts = meta["frames_per_bin"], np.array(meta["shifts"])
    census = json.loads((Path(cache_dir) / "grains.json").read_text())["grains"]

    def to_ref(x, y, frame):
        dx, dy = shifts[min(int(frame) // fpb, len(shifts) - 1)]
        return x - dx, y - dy

    con = sqlite3.connect(f"file:{PROJECT}?immutable=1", uri=True)
    obs = [json.loads(d) for (d,) in con.execute("select data from entities where kind='observation'")]
    doc = {"schema": "sparsetrack.bench.v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "origin": "legacy human answers on single compressed frames (rev16-rev18); development reference only",
           "movie": meta["movie"], "frames_per_bin": fpb, "n_bins": meta["n_bins"],
           "grains": {}, "labels": {}, "retest": {"grains": [], "labels": {}}}
    for name, (x, y, frame, owner) in GRAINS.items():
        rx, ry = to_ref(x, y, frame)
        best = min(census, key=lambda g: math.hypot(g["x"] - rx, g["y"] - ry))
        dist = math.hypot(best["x"] - rx, best["y"] - ry)
        if dist > 12:
            print(f"{name}: no census grain within 12 px (nearest {best['id']} at {dist:.1f}); skipped")
            continue
        gid = best["id"]
        doc["grains"][gid] = {**best, "legacy_name": name, "legacy_match_px": round(dist, 1)}
        la, fv = BRACKETS[name]
        entry = {"onset": {"verdict": "emerged_within", "last_absent_frame": la, "first_visible_frame": fv,
                           "last_absent_bin": la // fpb, "first_visible_bin": fv // fpb,
                           "review_origin": "human", "view": "single frames"}, "traces": {}}
        for o in obs:
            if o.get("owner_uuid") != owner or o.get("review_status", "active") != "active":
                continue
            f = o.get("source_frame")
            if f is None:
                continue
            key = str(int(f) // fpb)
            state = o.get("direct_state")
            path = o.get("path_xy") or []
            if state == "no_tube_visible":
                entry["traces"][key] = {"state": "no_tube", "source_frame": int(f), "length_px": 0.0,
                                        "path_xy_ref": [], "review_origin": "human"}
            elif state == "direct_visible" and len(path) >= 2:
                ref = [list(to_ref(px, py, f)) for px, py in path]
                length = float(sum(math.dist(ref[i], ref[i + 1]) for i in range(len(ref) - 1)))
                full = bool(o.get("path_complete", True))
                entry["traces"][key] = {"state": "full" if full else "partial", "source_frame": int(f),
                                        "length_px": round(length, 3), "path_xy_ref": ref,
                                        "n_clicks": len(path), "review_origin": "human"}
        doc["labels"][gid] = entry
        n_full = sum(t["state"] == "full" for t in entry["traces"].values())
        n_abs = sum(t["state"] == "no_tube" for t in entry["traces"].values())
        print(f"{name} -> {gid} ({dist:.1f} px): bracket ({la}, {fv}], {n_full} traces, {n_abs} absences")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
