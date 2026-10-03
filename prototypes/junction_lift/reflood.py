"""Re-run the 0.8.8 flood for one grain, capturing its arrival map and flood state (crop frame)."""
import pickle
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAIN, SP, movie  # noqa: E402

from sparsetrack import learned, stack  # noqa: E402
from sparsetrack.analyze import Params, reads_in_grain_frame  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402

CACHE = SP + "jt/flood_cache/"
_r = {}


def renderers(m):
    if m not in _r:
        bins, meta = stack.load(MAIN + f"runs/sparsetrack/{m}")
        pb, pm = stack.load(MAIN + f"runs/sparsetrack/{m}/prob_tubes_bn_real_ld_m2")
        _r[m] = (Renderer(bins, meta), Renderer(pb, pm), meta)
    return _r[m]


def base_params(m, **kw):
    vmax = movie(m)["pred"]["params"]["vmax_px"]
    return Params(vmax_px=vmax, vmax_auto=False, **kw)


def grain_ctx(m, gid):
    mv = movie(m)
    census = mv["lab"]["grains"]
    census = list(census.values()) if isinstance(census, dict) else census
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    g = next(c for c in census if c["id"] == gid)
    others = [o for o in physical if o["id"] != gid]
    return g, others


def run(m, gid, p=None, use_cache=True, check=False):
    """dict: res (read_grain result), arr, blocked, fl (tube, t_in, dist, emerge, length), rg, ang, centre, half,
    gx, gy, gr, drift (n, 2: grain frame -> field), rs, followed."""
    key = Path(CACHE) / f"{m}_{gid}.pkl"
    if p is None and use_cache and key.exists():
        return pickle.load(open(key, "rb"))
    renderer, prob, meta = renderers(m)
    pp = p or base_params(m)
    g, others = grain_ctx(m, gid)
    cap = {}
    orig_flood, orig_arr = learned.flood, learned.arrivals

    def flood_spy(arr, rg, ang, blocked, gr, *a, **k):
        out = orig_flood(arr, rg, ang, blocked, gr, *a, **k)
        cap.update(arr=arr, rg=rg, ang=ang, blocked=blocked, fl=out)
        return out

    def arr_spy(present, blocked, *a, **k):
        if check:
            cap["present"] = present
        return orig_arr(present, blocked, *a, **k)

    learned.flood, learned.arrivals = flood_spy, arr_spy
    try:
        res = learned.read_grain(renderer, prob, meta, g, others, pp)
    finally:
        learned.flood, learned.arrivals = orig_flood, orig_arr
    res.pop("_diag", None)
    half = pp.flood_half
    n = len(res["length"]["frames"])
    drift = np.asarray(res["drift"]["xy"], float) if res.get("drift") else np.zeros((n, 2))
    fl = cap["fl"]
    out = {"res": res, "arr": cap["arr"].astype(np.int16), "blocked": cap["blocked"],
           "fl": {"tube": fl["tube"], "t_in": fl["t_in"].astype(np.int16), "dist": fl["dist"].astype(np.float32),
                  "emerge": fl["emerge"], "length": fl["length"]},
           "rg": cap["rg"], "ang": cap["ang"], "centre": half - 0.5, "half": half, "gx": g["x"], "gy": g["y"],
           "gr": g["r"], "drift": drift, "rs": int(meta.get("ref_start", 0)),
           "followed": reads_in_grain_frame(drift if res.get("drift") else None, g["r"], pp), "m": m, "id": gid}
    if check:
        i = int(fl["emerge"] or 0) + 20
        pc = pcrop(out, out["rs"] + i, pp)
        print("present check at", i, "agree", float(np.mean((pc >= pp.flood_p * 250) == cap["present"][i])))
    if p is None and use_cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(key, "wb"))
    return out


def pcrop(st, b, p=None):
    """The P crop (uint8 x250) the flood saw at absolute bin b (grain frame)."""
    renderer, prob, meta = renderers(st["m"])
    p = p or base_params(st["m"])
    half, rs, drift = st["half"], st["rs"], st["drift"]
    n = len(drift)
    if st["followed"]:
        off = np.round(drift) if np.abs(drift).max() > p.track_recentre_px else np.zeros_like(drift)
        off_abs = np.zeros((int(meta["n_bins"]), 2))
        off_abs[rs:rs + n] = off
        resid = drift - off
    else:
        off_abs, resid = None, drift
    s = resid[b - rs]
    c = prob.crop(b, st["gx"], st["gy"], half, off_abs)
    w = cv2.warpAffine(np.nan_to_num(c).astype(np.float32), np.float32([[1, 0, -s[0]], [0, 1, -s[1]]]),
                       (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.clip(w, 0, 255).astype(np.uint8)


def to_crop(st, xy, i):
    """Field coords at series index i -> crop (x, y) continuous pixel coords (pixel centres at integers)."""
    xy = np.asarray(xy, float).reshape(-1, 2) - st["drift"][i]
    return np.stack([xy[:, 0] - st["gx"] + st["centre"], xy[:, 1] - st["gy"] + st["centre"]], 1)


def to_field(st, xy_crop, i):
    xy = np.asarray(xy_crop, float).reshape(-1, 2)
    return np.stack([xy[:, 0] + st["gx"] - st["centre"], xy[:, 1] + st["gy"] - st["centre"]], 1) + st["drift"][i]


if __name__ == "__main__":
    m = sys.argv[1]
    for gid in sys.argv[2:]:
        st = run(m, gid, check=True, use_cache=True)
        r = st["res"]
        base = movie(m)["G"][gid]
        print(m, gid, r["status"], r.get("final_length_px"), "base", base.get("final_length_px"), "same lengths:",
              np.allclose(r["length"]["px"], base["length"]["px"]), "followed", st["followed"], flush=True)
