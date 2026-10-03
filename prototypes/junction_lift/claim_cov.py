"""Decision or claim? For each FULL trace of the re-flooded grains: how much of the human route the flood had claimed
by the trace bin, whether the flood's tip lay on the human route, and why the unclaimed rest was not claimed."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, idx, movie, routes, seg_dists, trace  # noqa: E402
import reflood  # noqa: E402

cases = {(r["m"], r["g"], r["b"]): r for r in json.load(open(SP + "jt/cases.json"))}
rows = []
for f in sorted(glob.glob(SP + "jt/flood_cache/*.pkl")):
    m, gid = os.path.basename(f)[:-4].split("_")
    st = reflood.run(m, gid)
    g = movie(m)["G"][gid]
    n = st["arr"].max()
    for key, tr in sorted((movie(m)["lab"]["labels"][gid].get("traces") or {}).items(), key=lambda kv: int(kv[0])):
        if tr["state"] != "full":
            continue
        b = int(key)
        H, _ = trace(m, gid, b)
        i = idx(g, b)
        Hc = reflood.to_crop(st, H, i)
        q, s = routes.resample(Hc, 1.0)
        far = s > 5.0
        if far.sum() < 3:
            continue
        sel = st["fl"]["tube"] & (st["fl"]["t_in"] <= i) & np.isfinite(st["fl"]["dist"])
        ys, xs = np.nonzero(sel)
        if len(ys):
            from scipy.spatial import cKDTree
            dn = cKDTree(np.stack([xs, ys], 1)).query(q)[0]
        else:
            dn = np.full(len(q), 99.0)
        claimed = dn <= 3.0
        t = (g.get("tip") or {}).get("xy") or []
        tip_on = None
        if i < len(t) and t[i] and t[i][0] is not None and g["length"]["px"][i] > 0:
            tc = reflood.to_crop(st, np.asarray(t[i], float)[None] + (np.asarray(g["drift"]["xy"][i]) if g.get("drift") else 0), i)
            tip_on = float(seg_dists(tc, q)[0])
        why = {"blocked": 0, "old": 0, "late": 0, "never": 0, "arrived_unclaimed": 0}
        arr, blk = st["arr"], st["blocked"]
        tl = int(st["fl"]["t_in"][sel].max()) if sel.any() else -1
        for (x, y), c in zip(q[far], claimed[far]):
            if c:
                continue
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]):
                why["never"] += 1
                continue
            win = (slice(max(yi - 1, 0), yi + 2), slice(max(xi - 1, 0), xi + 2))
            a = int(arr[win].min())
            if blk[win].all():
                why["blocked"] += 1
            elif a >= n:
                why["never"] += 1
            elif a > i:
                why["late"] += 1
            else:
                why["arrived_unclaimed"] += 1
        r = cases.get((m, gid, b), {})
        rows.append({"m": m, "g": gid, "b": b, "Lh": tr["length_px"], "Lp": g["length"]["px"][i],
                     "len_hit": r.get("len_hit"), "both": r.get("both"), "cls": r.get("cls"),
                     "claim_cov": round(float(claimed[far].mean()), 2), "tip_to_H": None if tip_on is None else round(tip_on, 1),
                     "why": why, "contact": bool(tr.get("contact"))})
        rr = rows[-1]
        print(f"{m} {gid}@{b:<4d} {str(rr['cls']):20s} Lh {rr['Lh']:6.1f} Lp {rr['Lp']:6.1f} claimed {rr['claim_cov']:.2f} "
              f"tip->H {rr['tip_to_H']}  unclaimed: {why} {'contact' if rr['contact'] else ''}")
json.dump(rows, open(SP + "jt/claim_cov.json", "w"))
