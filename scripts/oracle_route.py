"""Oracle: the human's longest traced route is each grain's only path candidate.

    .venv/bin/python scripts/oracle_route.py MOVIE [key=value ...]   # MOVIE = ld | m2; params as synth_bench --set

Separates route choice from growth reading: if the oracle scores far better than the model with the
same parameters, the route is the bottleneck. Grains whose change map has no region on the rim
never reach the candidate step (they stay "no emergence") and are listed.
"""
import json
import sys

import numpy as np

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
import sparsetrack.analyze as A  # noqa: E402
from sparsetrack.evaluate import load, score  # noqa: E402
from synth_bench import parse_value  # noqa: E402

movie = sys.argv[1]
extra = {k: parse_value(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
p = A.Params(**extra)
labels_path = f"{REPO}/benchmark/labels/{movie}_v1.json"
L = load(labels_path)
centre = p.half - 0.5
CUR = [None]


def human_route(gid):
    g = L["grains"][gid]
    trs = [t for t in L["labels"].get(gid, {}).get("traces", {}).values()
           if t["state"] in ("full", "partial") and len(t.get("path_xy_view") or []) >= 2]
    if not trs:
        return None
    t = max(trs, key=lambda t: t["length_px"])
    hp = np.asarray(t["path_xy_view"], float) - [g["x"] - centre, g["y"] - centre]  # crop (x, y)
    seg = np.diff(hp, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    s = np.arange(0, cum[-1] + 1e-9, 1.0)
    d = np.stack([np.interp(s, cum, hp[:, 0]), np.interp(s, cum, hp[:, 1])], 1)
    out = np.nonzero(np.hypot(d[:, 0] - centre, d[:, 1] - centre) >= g["r"])[0]
    if not len(out):
        return None
    yx = np.clip(np.round(d[out[0]:, ::-1]).astype(int), 0, 2 * p.half - 1)
    keep = np.ones(len(yx), bool)
    keep[1:] = np.any(np.diff(yx, axis=0) != 0, axis=1)
    return yx[keep]


orig_cands, orig_grain = A.candidate_paths, A.analyze_grain
never_reached = []


def cands(comp, ring, cost, gr, c, pp):
    route = human_route(CUR[0])
    return [route] if route is not None and len(route) >= 3 else orig_cands(comp, ring, cost, gr, c, pp)


def grain(renderer, meta, g, others, pp, _settled=False):
    CUR[0] = g["id"]
    res = orig_grain(renderer, meta, g, others, pp, _settled)
    if not res.get("path_candidates") and res["status"] == "no_emergence_by_end" and human_route(g["id"]) is not None:
        never_reached.append(g["id"])
    return res


A.candidate_paths, A.analyze_grain = cands, grain
pred = A.analyze(f"{REPO}/runs/sparsetrack/{movie}", f"/tmp/tt_bench/oracle_{movie}", grains_path=labels_path,
                 params=p, log=lambda *a: None)
rep = score(L, pred)
o, Lf = rep["onset"], rep["length_full"]
print(f"ORACLE {movie} {extra}: onset {o['hits']}/{o['n_timed']}, len {Lf['within_tolerance']}/{Lf['n']} "
      f"(med {Lf['median_abs_error']:.2f}, bias {Lf['bias']:+.2f})")
print("  no rim region (route never tried):", never_reached)
for r in rep["rows"]:
    errs = ", ".join(f"{f['error']:+.1f}/{f['human']:.0f}" for f in r.get("full", []))
    print(f"  {r['grain']}: {r.get('pred')} onset err {r.get('onset_error')} | {errs}")
