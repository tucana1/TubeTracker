"""Why a disc passage opens or not: per foreign disc on a human route, the components of material that would arrive
inside it, with the band tests (edge points apart, edge covered, width) and their arrival bins."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import idx, movie, routes, trace  # noqa: E402
import reflood  # noqa: E402
from sparsetrack.learned import arrivals  # noqa: E402

CASES = [("m2", "g005", 349, "g037"), ("m2", "g052", 349, "g061"), ("m2", "g038", 349, "g041"),
         ("m2", "g092", 349, "g068"), ("m2", "g048", 349, "g047"), ("m2", "g064", 244, "g065")]

for m, gid, b, oid in CASES:
    st = reflood.run(m, gid)
    p = reflood.base_params(m)
    n_bins = len(st["drift"])
    half, centre = st["half"], st["centre"]
    o = movie(m)["lab"]["grains"][oid]
    ox, oy, R = o["x"] - st["gx"] + centre, o["y"] - st["gy"] + centre, o["r"] + p.other_block_px
    # P crops (grain frame) for every bin, only the disc's window
    x0, x1 = max(int(math.floor(ox - R)) - 1, 0), min(int(math.ceil(ox + R)) + 2, 2 * half)
    y0, y1 = max(int(math.floor(oy - R)) - 1, 0), min(int(math.ceil(oy + R)) + 2, 2 * half)
    pres = np.stack([reflood.pcrop(st, st["rs"] + i)[y0:y1, x0:x1] >= p.flood_p * 250 for i in range(n_bins)])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - ox, yy - oy)
    disc = rr < R
    a = arrivals(pres, st["blocked"][y0:y1, x0:x1] & ~disc, p.flood_persist, p.flood_frac)
    ever = disc & (a < n_bins)
    print(f"== {m} {gid}@{b} through {oid} (R {R:.1f}): disc px {disc.sum()}, ever-tube px {ever.sum()}")
    nl, lab = cv2.connectedComponents(ever.astype(np.uint8), connectivity=8)
    for l in range(1, nl):
        c = lab == l
        edge = c & (rr >= R - 1.5)
        th = np.arctan2(yy[edge] - oy, xx[edge] - ox)
        apart = np.degrees(np.abs(np.angle(np.exp(1j * (th[:, None] - th[None, :])))).max()) if edge.sum() >= 2 else 0
        covered = 10.0 * len(np.unique(np.floor(np.degrees(th) / 10.0)))
        pts = np.stack([xx[c], yy[c]], 1).astype(float)
        pts -= pts.mean(0)
        ax = np.linalg.eigh(np.cov(pts.T) if len(pts) > 2 else np.eye(2))[1][:, -1]
        width = c.sum() / (float(np.ptp(pts @ ax)) + 1.0)
        print(f"   comp {l}: {c.sum()} px, edge px {edge.sum()}, apart {apart:.0f} deg, edge covered {covered:.0f} deg, "
              f"width {width:.1f} px, arrivals {int(a[c].min())}..{int(np.median(a[c]))}..{int(a[c].max())}")
    # when the flood's claim reached the disc edge, and the human route's arrival inside the disc
    g = movie(m)["G"][gid]
    i = idx(g, b)
    H, _ = trace(m, gid, b)
    q, s = routes.resample(reflood.to_crop(st, H, i), 1.0)
    inside = np.hypot(q[:, 0] - ox, q[:, 1] - oy) < R
    if inside.any():
        k0 = int(np.argmax(inside))
        tube, t_in = st["fl"]["tube"], st["fl"]["t_in"]
        near = (np.hypot(*np.mgrid[0:2 * half, 0:2 * half][::-1] - q[max(k0 - 2, 0)][:, None, None]) <= 3) & tube
        print(f"   human route enters the disc at s={s[k0]:.0f}; flood's claim there from bin "
              f"{int(t_in[near].min()) if near.any() else None}; arrivals along the route inside: "
              f"{[int(a[int(round(y)) - y0, int(round(x)) - x0]) for x, y in q[inside][::3] if y0 <= round(y) < y1 and x0 <= round(x) < x1]}")
