"""What the flood's start looked like (cached 0.8.8 flood states): the piece it started on and the structure that
piece was joined to by then (everything arrived and connected), seen from the grain: how far it reaches, whether it
runs past the grain both ways along the rim, and when its far parts arrived."""
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP  # noqa: E402
import reflood  # noqa: E402


def features(st, halo=3.0, band=4.0):
    arr, rg, ang, gr = st["arr"].astype(np.int32), st["rg"], st["ang"], st["gr"]
    fl = st["fl"]
    b = fl["emerge"]
    if b is None:
        return None
    blocked = st["blocked"] | (rg < gr + halo)
    start_pix = fl["tube"] & (fl["t_in"] == b) & ((arr == b) & ~blocked)
    if not start_pix.any():
        start_pix = fl["tube"] & (fl["t_in"] == b)
    _, lab_all = cv2.connectedComponents(((arr <= b) & ~blocked).astype(np.uint8), connectivity=8)
    S = np.isin(lab_all, np.unique(lab_all[start_pix])) & (lab_all > 0)
    a0 = float(np.angle(np.mean(np.exp(1j * ang[start_pix]))))
    da = np.degrees(np.angle(np.exp(1j * (ang - a0))))
    out = S & (rg > gr + halo + band + 2)            # beyond the start band
    far = S & (rg > gr + 10)
    # tangent/radial coordinates about the start piece's foot on the rim
    c = st["centre"]
    foot = np.array([c + (gr + halo) * math.cos(a0), c + (gr + halo) * math.sin(a0)])  # (x, y)
    yy, xx = np.nonzero(S)
    rad = (xx - foot[0]) * math.cos(a0) + (yy - foot[1]) * math.sin(a0)
    tan = -(xx - foot[0]) * math.sin(a0) + (yy - foot[1]) * math.cos(a0)
    return {"b": int(b), "start_px": int(start_pix.sum()), "S_px": int(S.sum()),
            "reach": round(float((rg[S] - gr).max()), 1) if S.any() else 0.0,
            "da_min": round(float(da[out].min()), 0) if out.any() else 0.0,
            "da_max": round(float(da[out].max()), 0) if out.any() else 0.0,
            "tan_min": round(float(tan.min()), 1), "tan_max": round(float(tan.max()), 1),
            "rad_max": round(float(rad.max()), 1),
            "far_px": int(far.sum()),
            "far_arr_min": int(arr[far].min()) if far.any() else None,
            "far_arr_med": int(np.median(arr[far])) if far.any() else None}


if __name__ == "__main__":
    for f in sorted(glob.glob(SP + "jt/flood_cache/*.pkl")):
        m, gid = os.path.basename(f)[:-4].split("_")
        st = reflood.run(m, gid)
        ft = features(st)
        print(f"{m} {gid}: {ft}")
