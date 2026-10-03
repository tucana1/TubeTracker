"""Step 1: find junction takeovers among the missed traces of SparseTrack 0.8.8 (base088), and every trace whose
human route passes within ~6 px of another tube (P map at the trace bin)."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SP, drawn, idx, length_at, movie, p_along, pmap, proj_arc, routes, seg_dists, tip_at,  # noqa
                    trace)

OUT = SP + "jt/"


def near_other_tube(m, gid, b, H, census, band=6.0, near=11.0, reach=14.0):
    """Tube material (P >= 0.5 at bin b) not of the traced tube, within ``near`` px of the human centreline (i.e.
    ~6 px of the tube's edge), that extends at least ``reach`` px from it; grains masked out. Returns the min distance
    of such material to the route and the arc position along the route where it is closest (or None)."""
    x0, y0 = int(np.floor(H[:, 0].min())) - 40, int(np.floor(H[:, 1].min())) - 40
    x1, y1 = int(np.ceil(H[:, 0].max())) + 40, int(np.ceil(H[:, 1].max())) + 40
    P = pmap(m, b, x0, y0, x1 - x0, y1 - y0)
    yy, xx = np.mgrid[0:P.shape[0], 0:P.shape[1]]
    pts = np.stack([xx.ravel() + x0 + 0.5, yy.ravel() + y0 + 0.5], 1)
    Hs, _ = routes.resample(H, 1.0)
    d = seg_dists(pts, Hs).reshape(P.shape)
    mask = (P >= 0.5) & (d > band)
    for c in census.values():
        if c.get("exclude_reason") == "not_a_grain":
            continue
        mask &= np.hypot(xx + x0 + 0.5 - c["x"], yy + y0 + 0.5 - c["y"]) > c["r"] + 4.0
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    best = None
    for l in range(1, n):
        c = lab == l
        if c.sum() < 25 or d[c].max() < reach or d[c].min() > near:
            continue
        k = np.argmin(np.where(c, d, np.inf))
        ky, kx = np.unravel_index(k, P.shape)
        s = float(proj_arc(np.array([[kx + x0 + 0.5, ky + y0 + 0.5]]), Hs)[0])
        if best is None or d[ky, kx] < best[0]:
            best = (float(d[ky, kx]), s)
    return best


def analyse(m):
    mv = movie(m)
    lab, G = mv["lab"], mv["G"]
    census = lab["grains"]
    rows = []
    for gid, L in sorted(lab["labels"].items()):
        g = G.get(gid)
        if g is None:
            continue
        flood = "reader:flood" in g["flags"]
        for key, tr in sorted((L.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
            if tr["state"] != "full":
                continue
            b = int(key)
            H, _ = trace(m, gid, b)
            if H is None:
                continue
            Lh = float(tr["length_px"])
            Lp = length_at(g, b)
            tol = max(2.0, 0.1 * Lh)
            len_hit = abs(Lp - Lh) <= tol
            tip = tip_at(g, b)
            te = float(np.hypot(*(tip - H[-1]))) if tip is not None and Lp > 0 else None
            tip_hit = te is not None and te <= max(5.0, 0.1 * Lh)
            row = {"m": m, "g": gid, "b": b, "Lh": round(Lh, 1), "Lp": round(Lp, 1), "len_hit": len_hit,
                   "tip_err": None if te is None else round(te, 1), "both": bool(len_hit and tip_hit),
                   "flood": flood, "contact": bool(tr.get("contact"))}
            Hs, sH = routes.resample(H, 1.0)
            D = drawn(m, g, b)
            if D is not None and len(D) >= 2:
                Ds, sD = routes.resample(D, 1.0)
                dD = seg_dists(Ds, Hs)
                dH = seg_dists(Hs, Ds)
                row["cov_h"] = round(float(np.mean(dH <= 2.0)), 2)
                row["cov_p"] = round(float(np.mean(dD <= 2.0)), 2)
                off = dD > 4.0
                s_div = None
                for k in range(len(off)):
                    if off[k] and off[k:k + 6].all():
                        s_div = k
                        break
                if s_div is not None:
                    h_div = float(proj_arc(Ds[s_div:s_div + 1], Hs)[0])
                    tail = Ds[s_div:]
                    on = p_along(m, b, tail)
                    row.update(s_div=int(s_div), h_div=round(h_div, 1), tail_px=len(tail),
                               tail_on=round(float(np.mean(on >= 0.5)), 2),
                               tail_far=round(float(dD[s_div:].max()), 1))
                    # another labelled grain's trace at this bin under the off-route part
                    hits = []
                    for og, OL in lab["labels"].items():
                        if og == gid:
                            continue
                        ot = (OL.get("traces") or {}).get(key)
                        if ot and len(ot.get("path_xy_ref") or []) >= 2:
                            share = float(np.mean(seg_dists(tail, routes.resample(np.asarray(ot["path_xy_ref"]), 1.0)[0]) <= 4.0))
                            if share > 0.2:
                                hits.append((og, round(share, 2)))
                    row["on_other_trace"] = hits
                if tip is not None:
                    row["tip_to_H"] = round(float(seg_dists(tip[None], Hs)[0]), 1)
            nb = near_other_tube(m, gid, b, H, census)
            row["near_other"] = None if nb is None else (round(nb[0], 1), round(nb[1], 1))
            rows.append(row)
    return rows


def classify(r):
    if r["len_hit"] and r["both"]:
        return "hit"
    if "cov_h" not in r:
        return "no_route"
    if r.get("s_div") is not None and r["tail_px"] >= 8 and r["tail_on"] >= 0.5 and r["h_div"] <= r["Lh"] - 8:
        return "takeover_start" if r["s_div"] < 5 else "takeover"
    if r.get("s_div") is not None and r["tail_px"] >= 8 and r["tail_on"] >= 0.5:
        return "overshoot_onto_tube"   # left the route beyond the human apex
    if r.get("s_div") is not None and r["tail_px"] >= 8:
        return "off_route"
    if r["Lp"] < r["Lh"]:
        return "short_on_route"
    return "long_on_route"


if __name__ == "__main__":
    movies = sys.argv[1:] or ["m2", "m1", "ld"]
    allrows = []
    for m in movies:
        rows = analyse(m)
        for r in rows:
            r["cls"] = classify(r)
        allrows += rows
        n = len(rows)
        print(f"\n=== {m}: {n} full traces; length hits {sum(r['len_hit'] for r in rows)}, both {sum(r['both'] for r in rows)}")
        from collections import Counter
        print(" classes:", dict(Counter(r["cls"] for r in rows)))
        print(" near another tube (<=6 px from the edge):", sum(r["near_other"] is not None for r in rows),
              "of which missed (length):", sum(r["near_other"] is not None and not r["len_hit"] for r in rows))
        for r in rows:
            if r["cls"] in ("takeover", "takeover_start", "overshoot_onto_tube", "off_route"):
                print(f"  {r['cls']:20s} {r['g']}@{r['b']:<4d} Lh {r['Lh']:6.1f} Lp {r['Lp']:6.1f} tip {r['tip_err']} "
                      f"cov_h {r.get('cov_h')} s_div {r.get('s_div')} h_div {r.get('h_div')} tail {r.get('tail_px')} "
                      f"on {r.get('tail_on')} far {r.get('tail_far')} other {r.get('on_other_trace')} near {r['near_other']} "
                      f"{'flood' if r['flood'] else 'change'}{' contact' if r['contact'] else ''}")
    json.dump(allrows, open(OUT + "cases.json", "w"), indent=0, default=float)
