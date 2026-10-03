"""Which census grains' discs (r + 2 px) does each human route cross (beyond its own grain)?"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, movie, routes, trace  # noqa: E402

rows = json.load(open(SP + "jt/cases.json"))
for m in ["m2", "m1", "ld"]:
    mv = movie(m)
    census = mv["lab"]["grains"]
    n_full = n_cross = 0
    for r in rows:
        if r["m"] != m:
            continue
        H, tr = trace(m, r["g"], r["b"])
        Hs, _ = routes.resample(H, 1.0)
        n_full += 1
        hits = []
        for cid, c in census.items():
            if cid == r["g"] or c.get("exclude_reason") == "not_a_grain":
                continue
            d = np.hypot(Hs[:, 0] - c["x"], Hs[:, 1] - c["y"])
            inside = d < c["r"] + 2.0
            if inside.any():
                hits.append((cid, int(inside.sum()), round(float(c["r"]), 1), c.get("excluded"), c.get("exclude_reason")))
        if hits:
            n_cross += 1
            print(f"{m} {r['g']}@{r['b']} Lh {r['Lh']} {r['cls']} contact={r['contact']}: {hits}")
    print(f"== {m}: {n_cross}/{n_full} full traces cross another census grain's blocked disc")
