"""O3: the same carry + front with SparseTrack 0.8.8's own routes instead of the human's (no labels used to build a
route). Per scored grain that 0.8.8 drew a route for (grains it read no tube for keep 0.8.8's prediction):

  O3b  0.8.8's final route (its drawn route at the last bin: sparsetrack.report.turned_path + drift there, i.e.
       res["path"], centred), carried on straight 40 px, carried backward like O1's b (carry.carry_inext);
  O3c  re-anchored on 0.8.8's own per-bin routes (sparsetrack.routes.route_at via turned_path, + drift) at anchor bins
       every 25 bins and wherever path_by_bin switches routes: from each anchor to the next the anchor's route is
       carried forward and replaces the route up to its length, O3b's route joined on at its end (as O1's c).

Kymographs go to runs/research/carry_front/o3/<movie>/ in build.py's format; read them with o3eval.py.

    python -m prototypes.carry_front.o3 ld m2 m1
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from prototypes.carry_front.build import mean_dist, save
from prototypes.carry_front.carry import (EXTEND_PX, OUT, Crop, Movie, arclen, carry_inext, dense, extend,
                                          kymograph)
from prototypes.carry_front.o1 import baseline
from sparsetrack.report import turned_path

ROOT = OUT / "o3"
ANCHOR_EVERY = 25


def route_at_index(res: dict, pred: dict, i: int) -> np.ndarray | None:
    """0.8.8's drawn route at series index i, in reference coordinates where the tube lay then."""
    if not res.get("path") or len(res["path"]) < 2:
        return None
    try:
        path = turned_path(res, i, pred)
    except Exception:
        return None
    drift = (res.get("drift") or {}).get("xy")
    if drift:
        path = path + np.asarray(drift[min(i, len(drift) - 1)], float)
    return path if len(path) >= 2 and arclen(path)[-1] >= 3.0 else None


def anchors(res: dict, n: int) -> list[int]:
    a = set(range(0, n, ANCHOR_EVERY)) | {n - 1}
    idx = ((res.get("path_by_bin") or {}).get("index")) or []
    a |= {i for i in range(1, min(n, len(idx))) if idx[i] != idx[i - 1]}
    return sorted(a)


def scored(mv: Movie) -> list[str]:
    return [gid for gid, g in sorted(mv.labels["grains"].items()) if not g.get("excluded") and g.get("isolated", True)]


def build_auto(mv: Movie, gid: str, res: dict, pred: dict) -> dict | None:
    n = len(res["length"]["frames"])
    F_ref = route_at_index(res, pred, n - 1)
    if F_ref is None:
        return None
    g = mv.labels["grains"][gid]
    traces = mv.full_traces(gid)
    bT = mv.nb - 1
    F = extend(dense(F_ref), EXTEND_PX)
    S = arclen(F)
    anc = [(i, route_at_index(res, pred, i)) for i in anchors(res, n)]
    anc = [(i, r) for i, r in anc if r is not None]
    crop = Crop(mv, [F] + [dense(t["path_xy_ref"]) for _, t in traces] + [dense(r) for _, r in anc], g)
    bins = range(mv.rs, mv.nb)
    Fc = crop.to_crop(F)
    routes = {"a": np.repeat(Fc[None], len(bins), axis=0)}
    car = carry_inext(crop, Fc, bT, mv.rs)
    routes["b"] = np.stack([car[b] for b in bins])
    Xc = routes["b"].copy()
    for (ik, Rk), (ik1, _) in zip(anc, anc[1:]):
        Y = crop.to_crop(dense(Rk))
        sY = arclen(Y)
        inside = S <= sY[-1] + 1e-6
        j = int(np.nonzero(inside)[0][-1]) if inside.any() else 0
        cy = carry_inext(crop, Y, mv.rs + ik, mv.rs + ik1 - 1) if ik1 - 1 > ik else {mv.rs + ik: Y}
        for i in range(ik, ik1):
            Yb = cy[mv.rs + i]
            row = Xc[i].copy()
            row[inside] = np.stack([np.interp(S[inside], sY, Yb[:, 0]), np.interp(S[inside], sY, Yb[:, 1])], 1)
            if (~inside).any():
                row[~inside] = routes["b"][i][~inside] + (Yb[-1] - routes["b"][i][j])
            Xc[i] = row
    routes["c"] = Xc
    crop.clear()
    K = kymograph(crop, routes, bins)
    cover = {}
    for b, t in traces:
        hum = crop.to_crop(dense(t["path_xy_ref"], 0.5))
        sel = S <= float(t["length_px"]) + 1e-6
        i = min(max(b, mv.rs), mv.nb - 1) - mv.rs
        cover[str(b)] = {v: round(mean_dist(r[i][sel], hum), 2) for v, r in routes.items()}
    return {"gid": gid, "bT": bT, "S": S, "rs": mv.rs, "nb": mv.nb, "anchors": [i for i, _ in anc],
            "origin": [crop.cx - crop.half + 0.5, crop.cy - crop.half + 0.5],
            "routes": {v: routes[v].astype(np.float32) for v in ("b", "c")},
            "K": {v: {w: k.astype(np.float16) for w, k in kw.items()} for v, kw in K.items()}, "cover": cover}


def main():
    for movie in sys.argv[1:]:
        mv = Movie(movie)
        pred = baseline(movie)
        byid = {g["id"]: g for g in pred["grains"]}
        od = ROOT / movie
        od.mkdir(parents=True, exist_ok=True)
        for gid in scored(mv):
            res = byid.get(gid)
            t0 = time.time()
            out = build_auto(mv, gid, res, pred) if res else None
            if out is None:
                print(f"{movie} {gid}: 0.8.8 drew no route; its prediction is kept", flush=True)
                continue
            save(od / f"{gid}.npz", out)
            meta = json.loads((od / f"{gid}.json").read_text())
            meta["anchors"] = out["anchors"]
            (od / f"{gid}.json").write_text(json.dumps(meta))
            cov = " ".join(f"b{b}:" + "/".join(f"{v}{d:.1f}" for v, d in c.items()) for b, c in out["cover"].items())
            print(f"{movie} {gid} N={len(out['S'])} anchors {len(out['anchors'])} {time.time() - t0:.1f}s {cov}",
                  flush=True)


if __name__ == "__main__":
    main()
