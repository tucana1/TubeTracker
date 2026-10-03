"""Build the oracle-route kymographs (experiment O1) for every scored grain with a FULL human trace.

Route = the grain's LATEST full human trace (its exit point first), carried on straight by EXTEND_PX past the apex.
Variants, each a route per bin (reference coordinates):
  a   the route fixed in reference coordinates;
  braw  the route carried with plain composed DIS flow (every point by the flow at it; diagnostic);
  a2  the route moved rigidly with SparseTrack 0.8.8's grain drift (the change reader's "fixed route in the grain's
      frame"; needs the baseline predictions);
  b   the route carried backward (and forward) from its bin with composed DIS flow, as a tube: the flow's median
      over the base plus its sideways part along the tube, material spacing kept (carry.carry_inext);
  c   as b, but between two human traces the earlier trace's polyline, carried forward from its bin, replaces the
      route up to its length (the rest of b's route joined on at its end): at a scored trace k+1 the route comes from
      trace k carried on, never from trace k+1 itself.
Per variant and band half-width w: K (n_bins, N) = max of the tube map over the route normal within +/- w px.

    python -m prototypes.carry_front.build ld [--baseline .../ld_real_0/predictions.json] [--only g001,g002]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from prototypes.carry_front.carry import (EXTEND_PX, OUT, Crop, Movie, arclen, carry, carry_inext, dense, extend,
                                          kymograph)


def mean_dist(route: np.ndarray, human: np.ndarray) -> float:
    """Mean distance of route points to the (dense) human polyline."""
    if len(route) == 0 or len(human) == 0:
        return float("nan")
    d = np.hypot(route[:, None, 0] - human[None, :, 0], route[:, None, 1] - human[None, :, 1]).min(axis=1)
    return float(d.mean())


CARRY = {}  # carry_inext overrides (--step, --max-step): a sensitivity check, not tuned


def build_grain(mv: Movie, gid: str, drift: dict | None) -> dict:
    g = mv.labels["grains"][gid]
    traces = mv.full_traces(gid)
    bT, tT = traces[-1]
    F = extend(dense(tT["path_xy_ref"]), EXTEND_PX)
    S = arclen(F)
    crop = Crop(mv, [F] + [dense(t["path_xy_ref"]) for _, t in traces], g)
    bins = range(mv.rs, mv.nb)
    nb = len(bins)
    Fc = crop.to_crop(F)
    routes = {"a": np.repeat(Fc[None], nb, axis=0)}
    if drift is not None:
        # rigid: the route where it was at bT, moved by the grain's drift since
        dxy = np.asarray(drift["xy"], float)
        fr = np.asarray(drift["frames"])
        bin_of = (fr - mv.fpb // 2) // mv.fpb
        d = np.zeros((nb, 2))
        for i, b in enumerate(bins):
            d[i] = dxy[int(np.argmin(np.abs(bin_of - b)))]
        dT = d[bT - mv.rs]
        routes["a2"] = Fc[None] + (d - dT)[:, None, :]
    car = {}
    car.update(carry(crop, Fc, bT, mv.rs))
    car.update(carry(crop, Fc, bT, mv.nb - 1))
    routes["braw"] = np.stack([car[b] for b in bins])
    car = {}
    car.update(carry_inext(crop, Fc, bT, mv.rs, **CARRY))
    car.update(carry_inext(crop, Fc, bT, mv.nb - 1, **CARRY))
    routes["b"] = np.stack([car[b] for b in bins])
    Xc = routes["b"].copy()
    for (bk, tk), (bk1, _) in zip(traces, traces[1:]):
        bk, bk1 = max(bk, mv.rs), max(bk1, mv.rs)  # a trace before the analysed bins (m2 g082 at bin 6) counts at rs
        if bk1 <= bk:
            continue
        Y = crop.to_crop(dense(tk["path_xy_ref"]))
        sY = arclen(Y)
        Lk = sY[-1]
        cy = carry_inext(crop, Y, bk, bk1, **CARRY)
        inside = S <= Lk + 1e-6
        j = int(np.nonzero(inside)[0][-1]) if inside.any() else 0
        for b in range(bk + 1, bk1 + 1):
            i = b - mv.rs
            Yb = cy[b]
            xs = np.interp(S[inside], sY, Yb[:, 0])
            ys = np.interp(S[inside], sY, Yb[:, 1])
            row = Xc[i].copy()
            row[inside] = np.stack([xs, ys], 1)
            if (~inside).any():
                row[~inside] = routes["b"][i][~inside] + (Yb[-1] - routes["b"][i][j])
            Xc[i] = row
    routes["c"] = Xc
    crop.clear()
    K = kymograph(crop, routes, bins)
    # how far each variant's route lies from the human trace at the traced bins (over the human's length)
    cover = {}
    for b, t in traces:
        h = float(t["length_px"])
        hum = crop.to_crop(dense(t["path_xy_ref"], 0.5))
        sel = S <= h + 1e-6
        i = min(max(b, mv.rs), mv.nb - 1) - mv.rs
        cover[str(b)] = {v: round(mean_dist(r[i][sel], hum), 2) for v, r in routes.items()}
    out = {"gid": gid, "bT": bT, "S": S, "rs": mv.rs, "nb": mv.nb,
           "origin": [crop.cx - crop.half + 0.5, crop.cy - crop.half + 0.5],
           "routes": {v: r.astype(np.float32) for v, r in routes.items() if v in ("b", "c", "braw")},
           "K": {v: {w: k.astype(np.float16) for w, k in kw.items()} for v, kw in K.items()},
           "cover": cover}
    return out


def save(path, res):
    arrs = {"S": res["S"]}
    for v, r in res["routes"].items():
        arrs[f"X_{v}"] = r
    for v, kw in res["K"].items():
        for w, k in kw.items():
            arrs[f"K_{v}_{w:g}"] = k
    np.savez_compressed(path, **arrs)
    meta = {k: res[k] for k in ("gid", "bT", "rs", "nb", "origin", "cover")}
    path.with_suffix(".json").write_text(json.dumps(meta))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--baseline", help="0.8.8 predictions.json (for variant a2: the route moved with the grain)")
    ap.add_argument("--only")
    ap.add_argument("--step", type=int, help="flow step (bins) for the tube carry")
    ap.add_argument("--max-step", type=float, help="largest move per step (px)")
    ap.add_argument("--root", help="output folder (default runs/research/carry_front)")
    a = ap.parse_args(argv)
    mv = Movie(a.movie)
    drifts = {}
    if a.baseline:
        for g in json.loads(open(a.baseline).read())["grains"]:
            if g.get("drift"):
                drifts[g["id"]] = g["drift"]
    if a.step:
        CARRY["step"] = a.step
    if a.max_step:
        CARRY["max_step_px"] = a.max_step
    od = (Path(a.root) if a.root else OUT) / a.movie
    od.mkdir(parents=True, exist_ok=True)
    gids = mv.scored_grains()
    if a.only:
        gids = [g for g in gids if g in a.only.split(",")]
    for gid in gids:
        t0 = time.time()
        res = build_grain(mv, gid, drifts.get(gid) if a.baseline else None)
        save(od / f"{gid}.npz", res)
        cov = " ".join(f"b{b}:" + "/".join(f"{v}{d:.1f}" for v, d in c.items()) for b, c in res["cover"].items())
        print(f"{a.movie} {gid} bT={res['bT']} N={len(res['S'])} {time.time() - t0:.1f}s  {cov}", flush=True)


if __name__ == "__main__":
    main()
