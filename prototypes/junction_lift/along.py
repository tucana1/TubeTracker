"""Along the human route at the trace bin: arrival bin, claimed by the flood (and when), blocked, P then."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import idx, movie, routes, trace  # noqa: E402
import reflood  # noqa: E402


def along(m, gid, b, step=4.0):
    st = reflood.run(m, gid)
    g = movie(m)["G"][gid]
    i = idx(g, b)
    H, tr = trace(m, gid, b)
    Hc = reflood.to_crop(st, H, i)
    q, s = routes.resample(Hc, step)
    pc = reflood.pcrop(st, b).astype(np.float32) / 250
    arr, tube, t_in, blk = st["arr"], st["fl"]["tube"], st["fl"]["t_in"], st["blocked"]
    em = st["fl"]["emerge"]
    print(f"{m} {gid}@{b} i={i} emerge={em} Lh={tr['length_px']:.1f} model={g['length']['px'][i]:.1f}")
    print("   s   arr(min 3x3)  claimed(t_in)  blocked  P")
    for (x, y), sv in zip(q, s):
        xi, yi = int(round(x)), int(round(y))
        win = (slice(max(yi - 1, 0), yi + 2), slice(max(xi - 1, 0), xi + 2))
        a = int(arr[win].min())
        cl = tube[win] & (t_in[win] <= i)
        tin = int(t_in[win][tube[win]].min()) if tube[win].any() else -1
        print(f"{sv:5.0f}  {a:5d}  {'Y' if cl.any() else ('later' if tube[win].any() else '-'):>6s} {tin:5d}  "
              f"{'B' if blk[win].any() else ' '}  {pc[win].max():.2f}")


if __name__ == "__main__":
    m = sys.argv[1]
    for a in sys.argv[2:]:
        gid, b = a.split("@")
        along(m, gid, int(b))
