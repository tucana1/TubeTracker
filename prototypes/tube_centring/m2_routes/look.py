"""Look at one case: panels of the registered field (and the network's map) with the annotator's trace (green), the
model's tube as drawn (magenta), the rest of the model's route beyond the cut (dashed pink), this grain's census
circle moved by the model's drift (white), other census grains (grey), other labelled grains' traces at that bin
(cyan) and other grains' drawn model tubes (orange).

    python look.py GID BIN [--bins b1,b2,...] [--half H] [--centre x,y] [--size S] [--prob] [--growth] [--out NAME]
"""
import argparse

import cv2
import numpy as np

from common import (CENSUS, CYAN, G, GREEN, L, MAGENTA, ORANGE, OUT, PINK_DIM, R, WHITE, YELLOW, Tile, drawn,
                    drift_at, grid, length_at, trace)

ap = argparse.ArgumentParser()
ap.add_argument("gid")
ap.add_argument("bin", type=int)
ap.add_argument("--bins", default="")
ap.add_argument("--half", type=int, default=0)
ap.add_argument("--centre", default="")
ap.add_argument("--size", type=int, default=480)
ap.add_argument("--prob", action="store_true")
ap.add_argument("--growth", action="store_true")
ap.add_argument("--others", action="store_true", help="draw other grains' model tubes")
ap.add_argument("--ncol", type=int, default=3)
ap.add_argument("--out", default="")
ap.add_argument("--avg", type=int, default=0, help="average bins b-avg..b")
a = ap.parse_args()

gid, b = a.gid, a.bin
g = G[gid]
tr, t = trace(gid, b)
d = drawn(g, b)
full = drawn(g, b, cut=False)
pts = [q for q in (tr, d) if q is not None]
allp = np.vstack(pts) if pts else np.array([[G[gid]["x"], G[gid]["y"]]])
if a.centre:
    cx, cy = map(float, a.centre.split(","))
else:
    cx, cy = (allp.min(0) + allp.max(0)) / 2
half = a.half or int(max(np.ptp(allp, axis=0).max() / 2 + 20, 30))
nb = [b] + [int(x) for x in a.bins.split(",") if x]


def overlay(T, bb, main):
    # census grains
    for cid, c in CENSUS.items():
        if abs(c["x"] - T.cx) < T.half + 20 and abs(c["y"] - T.cy) < T.half + 20:
            p = np.array([c["x"], c["y"]])
            if cid in G:
                p = p + drift_at(G[cid], bb)
            T.circle(p, c["r"], (150, 150, 150) if cid != gid else WHITE, 1)
            T.text_at(p + [c["r"] * 0.7, -c["r"] * 0.7], cid, col=(200, 200, 200) if cid != gid else WHITE)
    # other labelled traces at this bin
    for oid in L["labels"]:
        if oid == gid:
            continue
        otr, ot = trace(oid, bb)
        if otr is not None:
            T.line(otr, CYAN, 1)
    if a.others:
        for oid, og in G.items():
            if oid == gid:
                continue
            od = drawn(og, bb)
            if od is not None:
                T.line(od, ORANGE, 1)
    gtr, gt = trace(gid, bb)
    if bb != b and tr is not None:
        T.dashed(tr, GREEN, 1)  # the trace at the bin being diagnosed, for reference
    if gtr is not None:
        T.line(gtr, GREEN, 2 if main else 2)
    fl = drawn(g, bb, cut=False)
    if fl is not None:
        T.dashed(fl, PINK_DIM, 1)
    dd = drawn(g, bb)
    if dd is not None:
        T.line(dd, MAGENTA, 2)
        T.dot(dd[-1], MAGENTA, 2)
    tip = g.get("tip", {}).get("xy")
    if tip:
        from common import idx
        tp = tip[idx(g, bb)]
        if tp is not None and len(tp) == 2 and tp[0] is not None:
            T.dot(np.asarray(tp, float), YELLOW, 2, filled=False)


tiles = []
for bb in nb:
    b0 = bb - a.avg if a.avg else bb
    T = Tile(bb, cx, cy, half, a.size, b0=b0, b1=bb)
    overlay(T, bb, bb == b)
    gtr, gt = trace(gid, bb)
    T.text(f"{gid} bin {bb}  model L={length_at(g, bb):.0f}" + (f"  traced L={gt['length_px']:.0f}" if gt else ""))
    tiles.append(T.im)
    if a.prob and bb == b:
        P = Tile(bb, cx, cy, half, a.size, prob=True)
        overlay(P, bb, True)
        P.text(f"P(tube) bin {bb}")
        tiles.append(P.im)
    if a.growth and bb == b:
        from common import to_u8
        img = R.growth_crop(bb - 2, bb, cx, cy, half)
        u8 = to_u8(img)
        T2 = Tile(bb, cx, cy, half, a.size)
        T2.im = cv2.cvtColor(cv2.resize(u8, (a.size, a.size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        overlay(T2, bb, True)
        T2.text(f"growth bin {bb}")
        tiles.append(T2.im)
out = OUT + (a.out or f"look_{gid}_{b}.png")
cv2.imwrite(out, grid(tiles, a.ncol))
print("wrote", out, "centre", (round(cx, 1), round(cy, 1)), "half", half)
