"""How far the drawn tube is from the annotator's trace at the traced bin: the stored route (as the app draws it) vs
the route re-centred on that bin's own network map (an upper bound for per-bin centring), and a rigid per-bin shift."""
import json, sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack, learned
from sparsetrack.analyze import Params, _resample
from tubetracker.app.overlay import to_length, turned
REPO = "/Users/joshjiang/Documents/TubeTracker/"
CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json", "m1": "benchmark/labels/m1_v1.json"}
movie, pred = sys.argv[1:3]
win = int(sys.argv[3]) if len(sys.argv) > 3 else 1
p = Params()
pb, pm = stack.load(REPO + CACHE[movie] + "/prob_tubes_bn_real_ld_m2")
meta = stack.load(REPO + CACHE[movie])[1]
fpb, nb = int(meta["frames_per_bin"]), int(meta["n_bins"])
L = json.load(open(REPO + LABELS[movie]))
P = {g["id"]: g for g in json.load(open(pred))["grains"]}


def normals(pts):
    tan = np.gradient(pts, axis=0)
    k = np.ones(7) / 7.0
    tan = np.stack([np.convolve(np.pad(tan[:, j], 3, mode="edge"), k, "valid") for j in range(2)], axis=1)
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    return np.stack([-tan[:, 1], tan[:, 0]], axis=1)


def pmap_profile(pts, nrm, bins_, shift):
    offs = np.arange(-p.centre_reach, p.centre_reach + 1e-6, 0.25)
    across = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    prof = np.zeros(across.shape[:2], np.float32)
    for b in bins_:
        q = (across + shift - 0.5).astype(np.float32)
        prof += cv2.remap(np.asarray(pb[b], np.float32), q[..., 0], q[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return prof / (len(bins_) * learned.P_SCALE), offs


def recentre(route, bins_, shift):
    pts, _ = _resample(np.asarray(route, float), 1.0)
    if len(pts) < 3:
        return pts
    nrm = normals(pts)
    prof, offs = pmap_profile(pts, nrm, bins_, shift)
    c = learned._band_centres(prof, offs, p.centre_min_p, p.centre_max_width)
    ok = np.isfinite(c)
    if ok.sum() < 2:
        return pts
    n = np.arange(len(c))
    c = np.interp(n, n[ok], c[ok])
    h = p.centre_smooth // 2
    c = np.clip([np.median(c[max(0, i - h):i + h + 1]) for i in n], -p.centre_max_shift, p.centre_max_shift)
    return pts + np.asarray(c)[:, None] * nrm


def rigid(route, bins_, shift, reach=5):
    """The translation (within +-reach px) maximising the mean P along the route."""
    pts, _ = _resample(np.asarray(route, float), 1.0)
    img = np.mean([np.asarray(pb[b], np.float32) for b in bins_], axis=0)
    best, bv = np.zeros(2), -1
    for dy in np.arange(-reach, reach + 0.01, 0.5):
        for dx in np.arange(-reach, reach + 0.01, 0.5):
            q = (pts + shift + [dx, dy] - 0.5).astype(np.float32)
            v = cv2.remap(img, q[:, 0], q[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).mean()
            if v > bv:
                bv, best = v, np.array([dx, dy])
    return pts + best


def offsets(trace, route, gr0, vo):
    tp, _ = _resample(np.asarray(trace, float), 1.0)
    if len(tp) < 3 or route is None or len(route) < 2:
        return []
    nrm = normals(tp)
    keep = np.hypot(*(tp - (np.array([gr0["x"], gr0["y"]]) + vo)).T) > gr0["r"] + 3
    out = []
    R = np.asarray(route, float)
    for i in np.nonzero(keep)[0]:
        best = None
        for a, c in zip(R[:-1], R[1:]):
            d = c - a
            M = np.array([[nrm[i][0], -d[0]], [nrm[i][1], -d[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            t, u = np.linalg.solve(M, a - tp[i])
            if 0 <= u <= 1 and abs(t) <= 12 and (best is None or abs(t) < abs(best)):
                best = t
        if best is not None:
            out.append(abs(best))
    return out


res = {"stored": [], "per-bin": [], "rigid": []}
late_flag = []
for gid, lab in sorted(L["labels"].items()):
    g = P.get(gid)
    if not g or not g.get("path") or len(g["path"]) < 2:
        continue
    gr0 = L["grains"][gid]
    frames = np.asarray(g["length"]["frames"])
    for bkey, t in (lab.get("traces") or {}).items():
        if t["state"] != "full" or len(t.get("path_xy_ref") or []) < 2 or t.get("length_px", 0) < 8:
            continue
        b = int(bkey)
        i = int(np.argmin(np.abs(frames - (b * fpb + fpb // 2))))
        Lb = g["length"]["px"][i]
        if Lb <= 0.5:
            continue
        rot = g.get("rotation_deg") or []
        route = np.asarray(turned(g["path"], rot[i] if i < len(rot) else 0.0, [g["x"], g["y"]]), float)
        drift = np.asarray(g["drift"]["xy"][i], float) if g.get("drift") else np.zeros(2)
        vo = np.asarray(t.get("view_offset") or [0, 0], float)
        stored = np.asarray(to_length(route, Lb), float) + drift
        bins_ = [bb for bb in range(b - win // 2, b + win // 2 + 1) if 0 <= bb < nb]
        ext = np.asarray(to_length(route, Lb + 8.0, 0.0), float)
        pb_route = recentre(ext, bins_, drift)
        pb_route = np.asarray(to_length(pb_route.tolist(), Lb), float) + drift
        rg = rigid(np.asarray(to_length(route, Lb), float), bins_, drift) + drift
        for k, r in (("stored", stored), ("per-bin", pb_route), ("rigid", rg)):
            o = offsets(t["path_xy_ref"], r, gr0, vo)
            res[k] += o
            if k == "stored":
                late_flag += [b >= nb - 16] * len(o)
for k, v in res.items():
    v = np.asarray(v)
    print(f"{movie} {k:8s}: n={len(v):5d} median {np.median(v):.2f} px  <=1 {np.mean(v <= 1):.0%}  <=2 {np.mean(v <= 2):.0%}")
