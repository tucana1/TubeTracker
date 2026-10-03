"""Timing at each m2 takeover junction: when the flood's claim reached the junction along the human route, when it
took the other branch, and when the human continuation beyond the junction arrived (or that it was blocked)."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, drawn, idx, movie, routes, seg_dists, trace  # noqa: E402
import reflood  # noqa: E402

res = {a["case"]: a for a in json.load(open(SP + "jt/route_test.json"))}
for case in ["m2 g005@349", "m2 g038@244", "m2 g038@349", "m2 g052@244", "m2 g052@349", "m2 g092@349", "m2 g011@349",
             "m1 g014@244", "m1 g065@349"]:
    m, rest = case.split()
    gid, b = rest.split("@")
    b = int(b)
    st = reflood.run(m, gid)
    g = movie(m)["G"][gid]
    i = idx(g, b)
    H, _ = trace(m, gid, b)
    Hc = reflood.to_crop(st, H, i)
    q, s = routes.resample(Hc, 1.0)
    tube, t_in, arr, blk = st["fl"]["tube"], st["fl"]["t_in"], st["arr"], st["blocked"]
    sel = tube & (t_in <= i)
    ys, xs = np.nonzero(sel)
    tree = cKDTree(np.stack([xs, ys], 1))
    dn, kn = tree.query(q)
    claimed = dn <= 3.0
    # junction: the last human-route point (beyond 5 px) still claimed before the first long unclaimed run
    far = s > 5
    k_end = None
    for k in range(len(q)):
        if far[k] and not claimed[k] and (~claimed[k:k + 8]).all():
            k_end = k
            break
    if k_end is None:
        print(f"{case}: human route claimed throughout (claimed share {claimed[far].mean():.2f})")
        continue
    t_reach = int(t_in[ys[kn[max(k_end - 1, 0)]], xs[kn[max(k_end - 1, 0)]]]) if k_end > 0 else None
    beyond = q[k_end:k_end + 30]
    a_b = []
    nblk = 0
    for x, y in beyond:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]):
            continue
        win = (slice(max(yi - 1, 0), yi + 2), slice(max(xi - 1, 0), xi + 2))
        if blk[win].all():
            nblk += 1
        else:
            a_b.append(int(arr[win].min()))
    # the other branch: claimed tube pixels farther than 6 px from the human route, claimed after t_reach - 12
    D = drawn(m, g, b)
    oth = sel & (t_in >= (t_reach or 0) - 12)
    oy, ox = np.nonzero(oth)
    if len(oy):
        d_h = seg_dists(np.stack([ox, oy], 1).astype(float), q)
        off = d_h > 6
        t_other = int(np.min(t_in[oy[off], ox[off]])) if off.any() else None
        n_other = int(off.sum())
    else:
        t_other, n_other = None, 0
    n = arr.max()
    print(f"{case}: claim along the human route ends at s={s[k_end]:.0f} of {s[-1]:.0f} (reached bin {t_reach}); "
          f"beyond it (next 30 px): {nblk} px blocked, arrivals {('median %d, min %d' % (np.median(a_b), min(a_b))) if a_b else '-'}"
          f" ({sum(v >= n for v in a_b)} never); other branch claimed from bin {t_other} ({n_other} px)")
