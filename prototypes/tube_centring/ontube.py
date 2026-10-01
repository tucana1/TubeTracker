"""Share of the drawn tube (as the app draws it at bin index i) lying on the tube network's map (P >= 0.5) then,
beyond the grain's rim and short of the tip's last 2 px."""
import json, sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import routes, stack
from sparsetrack.report import turned_path
from tubetracker.app.overlay import to_length


def on_tube(g, pred, i, pb, fpb, step=1.0):
    L = g["length"]["px"][i]
    if L < 10 or not g.get("path"):
        return None
    route = np.asarray(to_length(turned_path(g, i, pred).tolist(), L), float)
    pts, s = routes.resample(route, step)
    if g.get("drift"):
        pts = pts + np.asarray(g["drift"]["xy"][i], float)
    keep = (np.hypot(pts[:, 0] - g["x"] - (g["drift"]["xy"][i][0] if g.get("drift") else 0),
                     pts[:, 1] - g["y"] - (g["drift"]["xy"][i][1] if g.get("drift") else 0)) > g["r"] + 3) & (s < s[-1] - 2)
    if keep.sum() < 4:
        return None
    b = int(g["length"]["frames"][i]) // fpb
    q = (pts[keep] - 0.5).astype(np.float32)
    v = cv2.remap(np.asarray(pb[b]), q[:, 0], q[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return float(np.mean(v.ravel() >= 0.5 * 250))


if __name__ == "__main__":
    S = "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad"
    for m in ("ld", "m2"):
        pb, pm = stack.load(f"runs/sparsetrack/{m}/prob_tubes_bn_real_ld_m2")
        pred = json.load(open(f"{S}/tipfix/bench/{m}_real_0/predictions.json"))
        P = {g["id"]: g for g in pred["grains"]}
        d = np.load(f"{S}/prof_{m}_084.npz")
        gid, b, mod = d["gid"], d["bin"], d["model"]
        good, bad = [], []
        for g, bb in sorted(set(zip(gid, b))):
            sel = (gid == g) & (b == bb)
            cov = float(np.mean(np.abs(np.nan_to_num(mod[sel], nan=99)) <= 2))
            if g not in P:
                continue
            i = int(np.argmin(np.abs(np.asarray(P[g]["length"]["frames"]) - (int(bb) * 300 + 150))))
            f = on_tube(P[g], pred, i, pb, 300)
            if f is None:
                continue
            (good if cov >= 0.8 else bad if cov < 0.5 else []).append((str(g), int(bb), round(f, 2)))
        for thr in (0.4, 0.5, 0.6):
            print(f"{m} threshold {thr}: flags {sum(x[2] < thr for x in bad)}/{len(bad)} bad traces, "
                  f"{sum(x[2] < thr for x in good)}/{len(good)} good")
        print("   bad:", bad)
