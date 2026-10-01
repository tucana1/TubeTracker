"""Cross-sections of traced tubes. For every full trace (registered image of its bin), points every 1 px along the
annotator's trace (beyond the grain), the image and the tube network's map sampled along the trace's normal (-12..12
px), and where the model's path (as the app draws it: route cut to the length, plus its stored drift) crosses that
normal. Saves an npz per movie and prints a summary.

    python centre_profiles.py MOVIE PRED.json OUT.npz
"""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from tubetracker.app.imaging import FrameSource  # noqa: E402
from tubetracker.app.overlay import to_length, turned  # noqa: E402

REPO = "/Users/joshjiang/Documents/TubeTracker/"
CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json", "m1": "benchmark/labels/m1_v1.json"}
OFFS = np.arange(-12.0, 12.01, 0.5)

movie, pred_path, out = sys.argv[1:4]
F = FrameSource(REPO + CACHE[movie], fields=4)
PF = FrameSource(REPO + CACHE[movie] + "/prob_tubes_bn_real_ld_m2", fields=4)
fpb = int(F.meta["frames_per_bin"])
L = json.load(open(REPO + LABELS[movie]))
P = {g["id"]: g for g in json.load(open(pred_path))["grains"]}
census = L["grains"]


def resample(path, step=1.0):
    p = np.asarray(path, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 2:
        return None, None
    q = np.arange(0, s[-1], step)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1), q


def sample(img, xy):
    m = (xy - 0.5).astype(np.float32)  # continuous coordinates (pixel i covers [i, i+1)) -> OpenCV pixel centres
    return cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


PRED = json.load(open(pred_path))


def model_route(g, b):
    """The route as the app draws it at bin b: bent and turned as the tube lay then, cut to the length, moved by the
    drift."""
    from sparsetrack.report import turned_path
    i = int(np.argmin(np.abs(np.asarray(g["length"]["frames"]) - (b * fpb + fpb // 2))))
    Lb = g["length"]["px"][i]
    if Lb <= 0.5 or not g.get("path"):
        return None
    route = np.asarray(to_length(turned_path(g, i, PRED).tolist(), Lb), float)
    if g.get("drift"):
        route = route + np.asarray(g["drift"]["xy"][i], float)
    return route


def crossing(route, p0, n, reach=12.0):
    """Signed offset along normal n at which the polyline crosses the line p0 + t n (|t| <= reach), nearest 0."""
    best = None
    for a, c in zip(route[:-1], route[1:]):
        d = c - a
        M = np.array([[n[0], -d[0]], [n[1], -d[1]]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, u = np.linalg.solve(M, a - p0)
        if 0 <= u <= 1 and abs(t) <= reach and (best is None or abs(t) < abs(best)):
            best = t
    return best


rows = {k: [] for k in ("gid", "bin", "s", "ang", "prof", "pmap", "model", "length_px")}
for gid, lab in sorted(L["labels"].items()):
    g0 = census.get(gid)
    if not g0:
        continue
    for bkey, t in sorted((lab.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
        if t["state"] != "full" or len(t.get("path_xy_ref") or []) < 2 or t.get("length_px", 0) < 8:
            continue
        b = int(bkey)
        pts, s = resample(t["path_xy_ref"])
        if pts is None:
            continue
        vo = np.asarray(t.get("view_offset") or [0.0, 0.0], float)
        gc = np.array([g0["x"], g0["y"]]) + vo
        keep = np.hypot(*(pts - gc).T) > g0["r"] + 3.0
        others = [(o["x"], o["y"], o["r"]) for oid, o in census.items() if oid != gid]
        for ox, oy, orr in others:
            keep &= np.hypot(pts[:, 0] - ox - vo[0], pts[:, 1] - oy - vo[1]) > orr + 4.0
        if keep.sum() < 3:
            continue
        tan = np.gradient(pts, axis=0)
        k = np.ones(5) / 5
        tan = np.stack([np.convolve(np.pad(tan[:, j], 2, mode="edge"), k, "valid") for j in range(2)], 1)
        tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        grid = pts[:, None, :] + OFFS[None, :, None] * nrm[:, None, :]
        img = F.registered(b)
        prof = sample(img, grid)
        pm = sample(PF.registered(b).astype(np.float32), grid)
        route = model_route(P[gid], b) if gid in P else None
        for i in np.nonzero(keep)[0]:
            rows["gid"].append(gid)
            rows["bin"].append(b)
            rows["s"].append(s[i])
            rows["ang"].append(math.atan2(nrm[i, 1], nrm[i, 0]))
            rows["prof"].append(prof[i])
            rows["pmap"].append(pm[i])
            c = crossing(route, pts[i], nrm[i]) if route is not None and len(route) >= 2 else None
            rows["model"].append(np.nan if c is None else c)
            rows["length_px"].append(t["length_px"])
arr = {k: np.asarray(v) for k, v in rows.items()}
np.savez_compressed(out, offs=OFFS, **arr)

prof, pm, model = arr["prof"], arr["pmap"], arr["model"]
prof = prof - np.nanmedian(np.concatenate([prof[:, :6], prof[:, -6:]], 1), axis=1, keepdims=True)
win = np.abs(OFFS) <= 6
dark = OFFS[win][np.nanargmin(np.where(np.isnan(prof[:, win]), np.inf, prof[:, win]), axis=1)]
bright = OFFS[win][np.nanargmax(np.where(np.isnan(prof[:, win]), -np.inf, prof[:, win]), axis=1)]
ppk = OFFS[win][np.nanargmax(np.where(np.isnan(pm[:, win]), -1, pm[:, win]), axis=1)]
ok = np.isfinite(model)
print(f"{movie}: {len(prof)} points on {len(set(zip(arr['gid'], arr['bin'])))} traces; model path crosses at "
      f"{ok.sum()} of them")
print(f"  darkest point vs the annotator's trace: median |offset| {np.median(np.abs(dark)):.2f} px; "
      f"brightest {np.median(np.abs(bright)):.2f} px; network map peak {np.median(np.abs(ppk)):.2f} px")
print(f"  model path vs the annotator's trace: median |offset| {np.median(np.abs(model[ok])):.2f} px, "
      f"within 1 px {np.mean(np.abs(model[ok]) <= 1):.0%}, 2 px {np.mean(np.abs(model[ok]) <= 2):.0%}")
print(f"  model path vs the darkest point: median |offset| {np.median(np.abs(model[ok] - dark[ok])):.2f} px; "
      f"vs the map peak {np.median(np.abs(model[ok] - ppk[ok])):.2f} px")
print(f"  same side as the darkest point (sign agrees, both >0.5 px): "
      f"{np.mean(np.sign(model[ok]) == np.sign(dark[ok])):.0%} of {ok.sum()}")
