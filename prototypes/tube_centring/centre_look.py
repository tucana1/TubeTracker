"""Close-ups of traced tubes: the registered image at a traced bin, the annotator's trace (green), the model's path
as the app draws it (magenta: the route cut to the length, plus the stored drift if any) and, if given, the same
path moved by a second predictions file's drift (cyan).

    python centre_look.py MOVIE PRED.json OUT.png [PRED_WITH_DRIFT.json] [--grains g001,g002] [--last]
"""
import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402
from tubetracker.app.overlay import to_length, turned  # noqa: E402

CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json", "m1": "benchmark/labels/m1_v1.json"}
REPO = "/Users/joshjiang/Documents/TubeTracker/"

ap = argparse.ArgumentParser()
ap.add_argument("movie")
ap.add_argument("pred")
ap.add_argument("out")
ap.add_argument("pred2", nargs="?")
ap.add_argument("--grains", default="")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--half", type=int, default=40)
ap.add_argument("--zoom", type=int, default=5)
a = ap.parse_args()

bins, meta = stack.load(REPO + CACHE[a.movie])
R = Renderer(bins, meta)
fpb = int(meta["frames_per_bin"])
L = json.load(open(REPO + LABELS[a.movie]))
P = {g["id"]: g for g in json.load(open(a.pred))["grains"]}
P2 = {g["id"]: g for g in json.load(open(a.pred2))["grains"]} if a.pred2 else {}


def model_path(g, b):
    i = int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * fpb + fpb // 2))))
    Lb = g["length"]["px"][i]
    if Lb <= 0.5 or not g.get("path"):
        return None, i
    rot = g.get("rotation_deg") or []
    pivot = g.get("exit_xy") if False else [g["x"], g["y"]]
    route = turned(g["path"], rot[i] if i < len(rot) else 0.0, pivot)
    return np.asarray(to_length(route, Lb), float), i


tiles = []
ids = a.grains.split(",") if a.grains else sorted(L["labels"])
for gid in ids:
    lab = L["labels"].get(gid) or {}
    full = sorted((int(b), t) for b, t in (lab.get("traces") or {}).items() if t["state"] == "full"
                  and len(t.get("path_xy_ref") or []) >= 2)
    if not full or gid not in P:
        continue
    b, t = full[-1] if True else full[len(full) // 2]
    tr = np.asarray(t["path_xy_ref"], float)
    cx, cy = (tr[:, 0].mean() + L["grains"][gid]["x"]) / 2, (tr[:, 1].mean() + L["grains"][gid]["y"]) / 2
    half, Z = a.half, a.zoom
    img = R.mean_crop(b, b, cx, cy, half)
    fin = img[np.isfinite(img)]
    lo, hi = np.percentile(fin, [1, 99.5])
    u8 = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    big = cv2.cvtColor(cv2.resize(u8, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    to_c = lambda q: np.round((np.asarray(q, float) - [cx - half, cy - half]) * Z - 0.5).astype(np.int32)
    gr = L["grains"][gid]
    cv2.circle(big, tuple(to_c([gr["x"] + (t.get("view_offset") or [0, 0])[0], gr["y"] + (t.get("view_offset") or [0, 0])[1]])), int(gr["r"] * Z), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(big, [to_c(tr).reshape(-1, 1, 2)], False, (60, 200, 60), 2, cv2.LINE_AA)
    g = P[gid]
    mp, i = model_path(g, b)
    if mp is not None:
        d = np.asarray((g.get("drift") or {}).get("xy", [[0, 0]] * (i + 1))[i], float) if g.get("drift") else np.zeros(2)
        cv2.polylines(big, [to_c(mp + d).reshape(-1, 1, 2)], False, (220, 60, 220), 2, cv2.LINE_AA)
        if gid in P2 and P2[gid].get("drift"):
            d2 = np.asarray(P2[gid]["drift"]["xy"][i], float)
            mp2, _ = model_path(P2[gid], b)
            if mp2 is not None:
                cv2.polylines(big, [to_c(mp2 + d2).reshape(-1, 1, 2)], False, (230, 200, 40), 2, cv2.LINE_AA)
    cv2.putText(big, f"{gid} b{b}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(big, f"{gid} b{b}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(big)
    if len(tiles) >= a.n:
        break
cols = 4
rows = -(-len(tiles) // cols)
H, W = tiles[0].shape[:2]
sheet = np.full((rows * (H + 6), cols * (W + 6), 3), 255, np.uint8)
for k, tile in enumerate(tiles):
    r, c = divmod(k, cols)
    sheet[r * (H + 6):r * (H + 6) + H, c * (W + 6):c * (W + 6) + W] = tile
cv2.imwrite(a.out, sheet)
print("wrote", a.out, len(tiles), "tiles")
