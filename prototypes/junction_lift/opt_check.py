"""Quick check of flood options on chosen grains: onset and the length at each human FULL/partial trace, 0.8.8 vs
the option (* = within tolerance; p = partial, c = contact).

    opt_check.py m2 g005 g052 flood_disc_pass=true
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import idx, movie  # noqa: E402
import reflood  # noqa: E402


def parse(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return float(v) if "." in v else int(v)


m = sys.argv[1]
kw = {a.split("=")[0]: parse(a.split("=")[1]) for a in sys.argv[2:] if "=" in a}
gids = [a for a in sys.argv[2:] if "=" not in a]
for gid in gids:
    base = movie(m)["G"][gid]
    st = reflood.run(m, gid, p=reflood.base_params(m, **kw), use_cache=False)
    res = st["res"]
    out = []
    for key, tr in sorted((movie(m)["lab"]["labels"][gid].get("traces") or {}).items(), key=lambda kv: int(kv[0])):
        if tr["state"] not in ("full", "partial"):
            continue
        b = int(key)
        i = idx(base, b)
        h = tr["length_px"]
        tol = max(2.0, 0.1 * h)
        lb, ln = base["length"]["px"][i], res["length"]["px"][i]
        ok = (lambda v: abs(v - h) <= tol) if tr["state"] == "full" else (lambda v: v >= h - 2.0)
        tag = ("" if tr["state"] == "full" else "p") + ("c" if tr.get("contact") else "")
        out.append(f"@{b}{tag}: human {h:.0f} base {lb:.0f}{'*' if ok(lb) else ' '} new {ln:.0f}{'*' if ok(ln) else ' '}")
    print(f"{m} {gid} onset {base.get('onset_frame')} -> {res.get('onset_frame')}, final {base.get('final_length_px')} -> "
          f"{res.get('final_length_px')} | " + "; ".join(out), flush=True)
