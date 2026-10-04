"""Wrong starts in 0.8.8: per flood-read grain, the model's exit angle around the grain (its route's first point at
the first FULL human trace) against the human exit angle there."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import drawn, idx, movie, trace  # noqa: E402

for m in ["m2", "m1", "ld"]:
    mv = movie(m)
    n = bad = 0
    for gid, L in sorted(mv["lab"]["labels"].items()):
        g = mv["G"].get(gid)
        if g is None or "reader:flood" not in g["flags"]:
            continue
        full = sorted((int(k), t) for k, t in (L.get("traces") or {}).items() if t["state"] == "full")
        if not full:
            continue
        for b, tr in full:
            D = drawn(m, g, b)
            if D is None or len(D) < 2:
                continue
            H, _ = trace(m, gid, b)
            i = idx(g, b)
            d = np.asarray(g["drift"]["xy"][i]) if g.get("drift") else np.zeros(2)
            cx, cy = g["x"] + d[0], g["y"] + d[1]
            # exit direction: the first point >= r + 6 px out along each route
            def out_angle(P):
                rr = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
                k = np.flatnonzero(rr >= g["r"] + 6)
                q = P[k[0]] if len(k) else P[-1]
                return math.atan2(q[1] - cy, q[0] - cx)
            from common import routes
            Hs, _ = routes.resample(H, 1.0)
            Ds, _ = routes.resample(D, 1.0)
            dang = abs(math.degrees(math.remainder(out_angle(Ds) - out_angle(Hs), 2 * math.pi)))
            n += 1
            flag = dang > 45
            bad += flag
            print(f"{m} {gid}@{b:<4d} human {tr['length_px']:6.1f} model {g['length']['px'][i]:6.1f}  exit angle off "
                  f"{dang:5.0f} deg{'  <-- WRONG START?' if flag else ''}{'  contact' if tr.get('contact') else ''}")
            break  # the first FULL trace with a model route only
    print(f"== {m}: {bad}/{n} flood-read grains whose route leaves the grain > 45 deg from the human exit at the first "
          f"full trace")
