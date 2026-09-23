"""Ranking comparison at equal data: which exposed key picks the candidate
whose ENDPOINT lands nearest the gold tip?

rev11: the per-ranker quantity is the candidate's exported endpoint error
(`current_tip` vs the gold tip). The previous version measured distance
to the full support polyline — a long line can pass through the true tip
and end at a grain rim or empty background, so that number conflated
"line passes near the tip" with "the candidate ends at the tip". Support
distance is a proposal-coverage diagnostic and is reported as such in
the run report, not here.

Reads a run's candidates.json (which carries sel, q_max, route_p,
body_support, front_evidence, current_tip for every scored candidate)
plus its selection.json, and prints, per event: the live winner's
endpoint error, the endpoint error of the candidate each ranker would
choose, and the best-available endpoint error as the ceiling. Candidates
without an endpoint score as unavailable (worst).
"""
import json
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "runs/prototypes/v30/rev10_interval_frontrank"
SNAP = "runs/prototypes/v30/snapshots/snap25"

cands = json.load(open(f"{RUN}/candidates.json"))
sel = json.load(open(f"{RUN}/selection.json"))
obs = json.load(open(f"{SNAP}/observations.json"))
rows = obs if isinstance(obs, list) else obs.get("observations") or obs.get("rows") or []
gold = {o["obs_uuid"]: o.get("direct_xy") for o in rows
        if str(o.get("obs_uuid", "")).startswith("obs-r4-p0")}

UNAVAILABLE = 9e9


def endpoint_err(g, c):
    """The ENDPOINT-domain quantity (rev11): exported current_tip vs gold.

    A candidate without an endpoint is unavailable, not zero-distance.
    """
    tip = c.get("current_tip") or c.get("endpoint")
    if not tip:
        return UNAVAILABLE
    return float(np.hypot(float(tip[0]) - float(g[0]),
                          float(tip[1]) - float(g[1])))


KEYS = {
    "sel": lambda c: float(c.get("sel") or 0.0),
    "q_max(v1)": lambda c: float(c.get("tip_score") or 0.0),
    "front_peak": lambda c: float((c.get("front_evidence") or {}).get("peak_logit") or 0.0),
    "front_prob": lambda c: float((c.get("front_evidence") or {}).get("peak_prob") or 0.0),
    "front_margin": lambda c: float((c.get("front_evidence") or {}).get("margin") or 0.0),
    "body_mean": lambda c: float((c.get("body_support") or {}).get("mean") or 0.0),
    "prob_x_sel": lambda c: float((c.get("front_evidence") or {}).get("peak_prob") or 0.0) * float(c.get("sel") or 0.0),
    "ev(img)": lambda c: float(c.get("ev") or 0.0),
    "ev_x_sel": lambda c: float(c.get("ev") or 0.0) * float(c.get("sel") or 0.0),
}

by_owner = {}
for c in cands:
    by_owner.setdefault(c.get("owner_id"), []).append(c)

print(f"{'event':>6} | {'n':>3} | {'live(win)':>9} | " +
      " | ".join(f"{k:>11}" for k in KEYS) + f" | {'BEST':>7} | {'n_near':>6}")
for s in sel:
    ev = s.get("event")
    g = gold.get(ev)
    if not g:
        continue
    cs = by_owner.get(ev, [])
    if not cs:
        continue
    chosen = {k: max(cs, key=f) for k, f in KEYS.items()}
    ds = {k: endpoint_err(g, c) for k, c in chosen.items()}
    live = s.get("winner")
    lc = next((c for c in cs if c.get("route_id") == live), None)
    ld = endpoint_err(g, lc) if lc else None
    best = min(endpoint_err(g, c) for c in cs)
    n_near = sum(1 for c in cs if endpoint_err(g, c) <= 5.0)
    print(f"{ev[-4:]:>6} | {len(cs):>3} | {ld:>9.1f} | " +
          " | ".join(f"{ds[k]:>11.1f}" for k in KEYS) +
          f" | {best:>7.1f} | {n_near:>6}")
