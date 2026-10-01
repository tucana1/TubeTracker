"""One grain at each of its traced bins: annotator's trace (green), the route as drawn then (magenta)."""
import json, sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack
from sparsetrack.render import Renderer
from tubetracker.app.overlay import to_length, turned
REPO = "/Users/joshjiang/Documents/TubeTracker/"
CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2"}
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json"}
movie, pred, out = sys.argv[1:4]
gids = sys.argv[4].split(",")
bins, meta = stack.load(REPO + CACHE[movie])
R = Renderer(bins, meta)
fpb = int(meta["frames_per_bin"])
L = json.load(open(REPO + LABELS[movie]))
P = {g["id"]: g for g in json.load(open(pred))["grains"]}
rows = []
for gid in gids:
    g, lab = P[gid], L["labels"][gid]
    full = sorted((int(b), t) for b, t in lab["traces"].items() if t["state"] == "full" and len(t["path_xy_ref"]) >= 2)
    last = np.asarray(full[-1][1]["path_xy_ref"], float)
    cx, cy = (last[:, 0].mean() + g["x"]) / 2, (last[:, 1].mean() + g["y"]) / 2
    half, Z = 44, 4
    tiles = []
    for b, t in full[-5:]:
        img = R.mean_crop(b, b, cx, cy, half)
        fin = img[np.isfinite(img)]
        lo, hi = np.percentile(fin, [1, 99.5])
        u8 = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        big = cv2.cvtColor(cv2.resize(u8, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        to_c = lambda q: np.round((np.asarray(q, float) - [cx - half, cy - half]) * Z - 0.5).astype(np.int32)
        i = int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * fpb + fpb // 2))))
        Lb = g["length"]["px"][i]
        if Lb > 0.5 and g.get("path"):
            rot = g.get("rotation_deg") or []
            route = np.asarray(to_length(turned(g["path"], rot[i] if i < len(rot) else 0.0, [g["x"], g["y"]]), Lb), float)
            if g.get("drift"):
                route = route + np.asarray(g["drift"]["xy"][i], float)
            cv2.polylines(big, [to_c(route).reshape(-1, 1, 2)], False, (220, 60, 220), 2, cv2.LINE_AA)
        cv2.polylines(big, [to_c(t["path_xy_ref"]).reshape(-1, 1, 2)], False, (60, 200, 60), 2, cv2.LINE_AA)
        d = g["drift"]["xy"][i] if g.get("drift") else [0, 0]
        cv2.putText(big, f"{gid} b{b} drift {d[0]:.1f},{d[1]:.1f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(big, f"{gid} b{b} drift {d[0]:.1f},{d[1]:.1f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(big)
    while len(tiles) < 5:
        tiles.append(np.full_like(tiles[0], 255))
    rows.append(np.hstack([np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255) for t in tiles]))
cv2.imwrite(out, np.vstack(rows))
print("wrote", out)
