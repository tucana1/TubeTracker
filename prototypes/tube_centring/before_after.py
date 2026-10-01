"""Before/after close-ups: at several traced bins of a grain, the annotator's trace (green), the route as the app
drew it before (red: no drift for unfollowed grains, the reader's route) and now (magenta: drift, centred, bent)."""
import json, sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.report import turned_path
from tubetracker.app.overlay import to_length
REPO = "/Users/joshjiang/Documents/TubeTracker/"
CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json", "m1": "benchmark/labels/m1_v1.json"}
movie, before, after, out = sys.argv[1:5]
gids = sys.argv[5].split(",")
nshow = int(sys.argv[6]) if len(sys.argv) > 6 else 4
bins, meta = stack.load(REPO + CACHE[movie])
R = Renderer(bins, meta)
fpb = int(meta["frames_per_bin"])
L = json.load(open(REPO + LABELS[movie]))
B, A = json.load(open(before)), json.load(open(after))
PB, PA = {g["id"]: g for g in B["grains"]}, {g["id"]: g for g in A["grains"]}


def drawn(g, pred, b, use_drift=True):
    i = int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * fpb + fpb // 2))))
    Lb = g["length"]["px"][i]
    if Lb <= 0.5 or not g.get("path"):
        return None
    r = np.asarray(to_length(turned_path(g, i, pred).tolist(), Lb), float)
    return r + (np.asarray(g["drift"]["xy"][i], float) if use_drift and g.get("drift") else 0.0)


rows = []
for gid in gids:
    lab = L["labels"][gid]
    full = sorted((int(b), t) for b, t in lab["traces"].items() if t["state"] == "full" and len(t["path_xy_ref"]) >= 2)
    pick = [full[k] for k in np.linspace(0, len(full) - 1, min(nshow, len(full))).round().astype(int)]
    last = np.asarray(full[-1][1]["path_xy_ref"], float)
    tiles = []
    for b, t in pick:
        tr = np.asarray(t["path_xy_ref"], float)
        cx, cy = tr[:, 0].mean(), tr[:, 1].mean()
        half, Z = max(28, int(np.ptp(tr, axis=0).max() / 2 + 14)), 0
        Z = max(2, min(6, int(300 / (2 * half))))
        img = R.mean_crop(b, b, cx, cy, half)
        fin = img[np.isfinite(img)]
        lo, hi = np.percentile(fin, [1, 99.5])
        u8 = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        big = cv2.cvtColor(cv2.resize(u8, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        big = cv2.resize(big, (300, 300))
        Zs = 300 / (2 * half)
        to_c = lambda q: np.round((np.asarray(q, float) - [cx - half, cy - half]) * Zs - 0.5).astype(np.int32)
        cv2.polylines(big, [to_c(tr).reshape(-1, 1, 2)], False, (60, 200, 60), 2, cv2.LINE_AA)
        old = drawn(PB[gid], B, b, use_drift=bool(PB[gid].get("drift")))
        new = drawn(PA[gid], A, b)
        if old is not None:
            cv2.polylines(big, [to_c(old).reshape(-1, 1, 2)], False, (60, 60, 230), 2, cv2.LINE_AA)
        if new is not None:
            cv2.polylines(big, [to_c(new).reshape(-1, 1, 2)], False, (220, 60, 220), 2, cv2.LINE_AA)
        cv2.putText(big, f"{gid} bin {b}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(big, f"{gid} bin {b}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.pad(big, ((3, 3), (3, 3), (0, 0)), constant_values=255))
    while len(tiles) < nshow:
        tiles.append(np.full_like(tiles[0], 255))
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows))
print("wrote", out)
