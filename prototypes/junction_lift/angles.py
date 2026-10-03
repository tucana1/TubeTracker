"""At each junction: the turn from the incoming direction (human route, 10 px before J) onto the human branch and onto
the other branch (15 px after J along each). Which continues the incoming direction?"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, routes, trace  # noqa: E402

res = json.load(open(SP + "jt/route_test.json"))


def ang(v):
    return math.atan2(v[1], v[0])


def turn(a, b):
    return abs(math.degrees(math.remainder(b - a, 2 * math.pi)))


rows = []
for a in res:
    m, rest = a["case"].split()
    gid, b = rest.split("@")
    H, _ = trace(m, gid, int(b))
    Hs, sH = routes.resample(H, 1.0)
    J = np.asarray(a["J"])
    kJ = int(np.argmin(np.hypot(*(Hs - J).T)))
    back = Hs[max(kJ - 10, 0)]
    if kJ < 5:  # junction at the exit: incoming = outward from the grain is unknown; use the human route's first 10 px
        th_in = ang(Hs[min(10, len(Hs) - 1)] - Hs[0])
    else:
        th_in = ang(J - back)
    hb = Hs[min(kJ + 15, len(Hs) - 1)] if kJ + 5 < len(Hs) else None
    oth = np.asarray(a["plain"]["routeT"])  # exit -> other end (plain geodesic on P)
    oq, os_ = routes.resample(oth, 1.0)
    kO = int(np.argmin(np.hypot(*(oq - J).T)))
    # the other branch leaves the human route where routeT is last within 3 px of the human route
    from common import seg_dists
    dO = seg_dists(oq, Hs)
    close = np.flatnonzero(dO <= 3.0)
    kd = int(close[-1]) if len(close) else 0
    ob = oq[min(kd + 15, len(oq) - 1)]
    t_h = None if hb is None else turn(th_in, ang(hb - J))
    t_o = turn(th_in, ang(ob - oq[kd]))
    rows.append((a["case"], a["cls"], a["contact"], t_h, t_o))
    print(f"{a['case']:14s} {a['cls']:20s} {'C' if a['contact'] else ' '} turn onto human branch "
          f"{'-' if t_h is None else f'{t_h:5.0f}'}  onto other {t_o:5.0f}  "
          f"{'' if t_h is None else ('human straighter' if t_h < t_o else 'OTHER straighter')}")
ok = [r for r in rows if r[3] is not None]
print(f"\nhuman branch straighter in {sum(r[3] < r[4] for r in ok)}/{len(ok)}; median turn human {np.median([r[3] for r in ok]):.0f}, "
      f"other {np.median([r[4] for r in ok]):.0f} deg")
