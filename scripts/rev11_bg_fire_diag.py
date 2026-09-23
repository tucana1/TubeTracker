"""rev11 diagnostic: where do the body heads' background fires sit?

The gate600 trajectory: recall on reviewed body is 96-100% for all
three owners, cross-owner firing is 0-11%, but 64-68% of each head's
positive pixels fall in UNREVIEWED background. This script asks WHERE
those fires are: a tube-continuation band just outside the reviewed
extent (an extent/labels question — 'unknown' treated as negative), or
scattered noise far from anything reviewed.
"""
import numpy as np
from scipy import ndimage as ndi

STEP = "569"
z = np.load(f"runs/prototypes/v30/rev11_fitcheck_gate600.step{STEP}.npz",
            allow_pickle=True)
for i in range(3):
    p = z[f"prob_{i}"] > 0.5
    own = z[f"mask_{i}"]
    others = np.zeros_like(own)
    for j in range(3):
        if j != i:
            others |= z[f"mask_{j}"]
    fband = z[f"foreign_{i}"]
    bg = ~(own | others | fband)
    d = ndi.distance_transform_edt(~(own | others))
    bf = p & bg
    dists = d[bf]
    hist = []
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 9999)):
        hist.append(int(((dists > lo) & (dists <= hi)).sum()))
    print(f"g{i}: bg-fire {int(bf.sum())}px at dist-to-nearest-reviewed:"
          f" <=10px {hist[0]}, 10-20 {hist[1]}, 20-40 {hist[2]},"
          f" 40-80 {hist[3]}, >80 {hist[4]}")
    d_own = ndi.distance_transform_edt(~own)
    near_own = int((p & (d_own <= 20) & ~own).sum())
    print(f"   fires within 20px of OWN (continuation band?):"
          f" {near_own}px; own mask {int(own.sum())}px")
