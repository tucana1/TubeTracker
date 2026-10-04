"""After the flood's start: how its claimed tube, and the arrived structure joined to the start piece, spread in
tangent (along the rim) and radial coordinates about the start's foot, bin by bin."""
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP  # noqa: E402
import reflood  # noqa: E402


def coords(st, mask, a0):
    c, gr = st["centre"], st["gr"]
    foot = (c + (gr + 3.0) * math.cos(a0), c + (gr + 3.0) * math.sin(a0))
    yy, xx = np.nonzero(mask)
    rad = (xx - foot[0]) * math.cos(a0) + (yy - foot[1]) * math.sin(a0)
    tan = -(xx - foot[0]) * math.sin(a0) + (yy - foot[1]) * math.cos(a0)
    return tan, rad


def evo(st, ks=(0, 3, 6, 10, 15, 20, 30, 45)):
    arr, rg, ang, gr = st["arr"].astype(np.int32), st["rg"], st["ang"], st["gr"]
    fl = st["fl"]
    b = fl["emerge"]
    blocked = st["blocked"] | (rg < gr + 3.0)
    start = fl["tube"] & (fl["t_in"] == b)
    a0 = float(np.angle(np.mean(np.exp(1j * ang[start]))))
    rows = []
    for k in ks:
        t = b + k
        tube = fl["tube"] & (fl["t_in"] <= t)
        tn, rd = coords(st, tube, a0)
        _, lab = cv2.connectedComponents(((arr <= t) & ~blocked).astype(np.uint8), connectivity=8)
        S = np.isin(lab, np.unique(lab[start])) & (lab > 0)
        sn, sr = coords(st, S, a0)
        rows.append(f"+{k:<2d} tube t[{tn.min():5.1f},{tn.max():5.1f}] r<={rd.max():5.1f} | joined {S.sum():5d}px "
                    f"t[{sn.min():6.1f},{sn.max():6.1f}] r<={sr.max():6.1f}")
    return b, rows


if __name__ == "__main__":
    sel = sys.argv[1:]
    for f in sorted(glob.glob(SP + "jt/flood_cache/*.pkl")):
        m, gid = os.path.basename(f)[:-4].split("_")
        if sel and f"{m}_{gid}" not in sel:
            continue
        st = reflood.run(m, gid)
        b, rows = evo(st)
        print(f"== {m} {gid} start bin {b}")
        for r in rows:
            print("   " + r)
