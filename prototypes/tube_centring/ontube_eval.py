"""Two measures of a predictions file on the labelled grains: drawn tube vs traces (median offset, within 2 px,
early and late), and the share of (grain, every-10th-bin) where the drawn tube is mostly off the network's map."""
import json, sys
import numpy as np
S = "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad"
sys.path.insert(0, S)
from ontube import on_tube
from sparsetrack import stack
movie, f, npz = sys.argv[1:4]
pb, pm = stack.load(f"runs/sparsetrack/{movie}/prob_tubes_bn_real_ld_m2")
pred = json.load(open(f))
off = tot = 0
for g in pred["grains"]:
    if not (g["status"].startswith("emerged") and g.get("path")):
        continue
    for i in range(0, len(g["length"]["frames"]), 10):
        v = on_tube(g, pred, i, pb, 300)
        if v is not None:
            tot += 1
            off += v < 0.5
d = np.load(npz)
mod, b = d["model"], d["bin"]
ok = np.isfinite(mod)
nb = {"ld": 176, "m2": 351, "m1": 351}[movie]
early = ok & (b < nb // 2)
late = ok & (b >= nb // 2)
print(f"{movie} {f.split('/')[-1]}: off the map at {off}/{tot} grain-bins ({off / max(tot, 1):.0%}); traces: all median "
      f"{np.median(np.abs(mod[ok])):.2f} px, <=2 px {np.mean(np.abs(mod[ok]) <= 2):.0%}; early half {np.median(np.abs(mod[early])):.2f} px "
      f"({np.mean(np.abs(mod[early]) <= 2):.0%}); late half {np.median(np.abs(mod[late])):.2f} px ({np.mean(np.abs(mod[late]) <= 2):.0%})")
