"""Quick check of flood_lift_px on chosen grains: lengths at each human trace, baseline vs lift."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import idx, movie, trace  # noqa: E402
import reflood  # noqa: E402

m = sys.argv[1]
kw = {}
for a in sys.argv[2:]:
    if "=" in a:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
gids = [a for a in sys.argv[2:] if "=" not in a]
for gid in gids:
    base = movie(m)["G"][gid]
    st = reflood.run(m, gid, p=reflood.base_params(m, **kw), use_cache=False)
    res = st["res"]
    out = []
    for key, tr in sorted((movie(m)["lab"]["labels"][gid].get("traces") or {}).items(), key=lambda kv: int(kv[0])):
        if tr["state"] != "full":
            continue
        b = int(key)
        i = idx(base, b)
        Lh = tr["length_px"]
        tol = max(2.0, 0.1 * Lh)
        lb, ll = base["length"]["px"][i], res["length"]["px"][i]
        out.append(f"@{b}: human {Lh:.0f} base {lb:.0f}{'*' if abs(lb - Lh) <= tol else ' '} lift {ll:.0f}{'*' if abs(ll - Lh) <= tol else ' '}")
    print(m, gid, "final", base.get("final_length_px"), "->", res.get("final_length_px"), "|", "; ".join(out), flush=True)
