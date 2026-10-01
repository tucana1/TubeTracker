"""Apply sparsetrack.learned.centre_route (the pipeline's own) to saved predictions: python apply_centre.py MOVIE IN OUT [key=value ...]"""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from dataclasses import replace
from sparsetrack import learned, stack
from sparsetrack.analyze import Params
from sparsetrack.render import Renderer
CACHE = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
movie, src, out = sys.argv[1:4]
p = Params()
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    p = replace(p, **{k: type(getattr(p, k))(v)})
prob = Renderer(*stack.load("/Users/joshjiang/Documents/TubeTracker/" + CACHE[movie] + "/prob_tubes_bn_real_ld_m2"))
meta = stack.load("/Users/joshjiang/Documents/TubeTracker/" + CACHE[movie])[1]
d = json.load(open(src))
moved = []
for g in d["grains"]:
    if len(g.get("path") or []) >= 2:
        learned.centre_route(g, prob, meta, p)
        if "route_centred_px" in g:
            moved.append(g["route_centred_px"])
json.dump(d, open(out, "w"))
print(f"{movie}: {len(moved)} routes centred, median |shift| {np.median(moved) if moved else 0:.2f} px")
