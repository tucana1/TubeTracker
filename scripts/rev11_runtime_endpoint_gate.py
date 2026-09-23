"""rev11 runtime-proposal ENDPOINT gate (review section 8B, pre-experiment).

The rev11 review: "A selector cannot choose an endpoint absent from its
pool" — support proximity is 6/6 but ENDPOINT-oracle coverage at 5 px is
1/6 on the six developed events. This gate reads a saved runtime run
(the deployment proposer's own candidates) and reports, per event:

  * best endpoint error (any candidate)  — proposal availability
  * selected endpoint error              — what the pipeline emits
  * how many SEPARATED hypotheses (top-K) already localize within
    5/10/15 px, with their route ids    — is a >5px answer a selection
    problem or a missing-candidate problem?

Engineering targets (review): all six developed events should offer a
<=5-px front CANDIDATE with high correct selection. The gate reports the
current shortfall; it does not re-rank.

Usage:
  .venv/bin/python scripts/rev11_runtime_endpoint_gate.py \
      [--run runs/prototypes/v30/rev11_strict6] \
      [--out runs/prototypes/v30/rev11_strict6/endpoint_gate.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOL_PX = 5.0
SNAP = REPO / "runs/prototypes/v30/snapshots/snap25"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(
        REPO / "runs/prototypes/v30/rev11_strict6"))
    ap.add_argument("--snapshot", default=str(SNAP))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    run = Path(a.run)
    cands = json.loads((run / "candidates.json").read_text())
    sel = json.loads((run / "selection.json").read_text())
    rep = json.loads((run / "report.json").read_text())
    obs = json.loads((Path(a.snapshot) / "observations.json").read_text())
    gold = {o["obs_uuid"]: o.get("direct_xy") for o in obs
            if str(o.get("obs_uuid", "")).startswith("obs-r4-p")}

    by_owner: dict[str, list] = {}
    for c in cands:
        by_owner.setdefault(str(c.get("owner_id")), []).append(c)
    winners = {s.get("event"): (s.get("winner") if not isinstance(
        s.get("winner"), dict) else s["winner"].get("route_id"))
        for s in sel}

    rows = []
    n_oracle = n_sel = 0
    for ev in sorted(by_owner):
        g = gold.get(ev)
        if not g:
            continue
        cs = by_owner[ev]
        errs = []
        for c in cs:
            tip = c.get("current_tip") or c.get("endpoint")
            errs.append((float(np.hypot(tip[0] - g[0], tip[1] - g[1])),
                         str(c.get("route_id")))
                        if tip else (math.inf, str(c.get("route_id"))))
        errs.sort()
        best_e, best_r = errs[0]
        win_r = winners.get(ev)
        win_e = next((e for e, r in errs if r == win_r), None)
        within = {t: [r for e, r in errs if e <= t]
                  for t in (5.0, 10.0, 15.0)}
        n_oracle += int(best_e <= TOL_PX)
        n_sel += int(win_e is not None and win_e <= TOL_PX)
        rows.append({
            "event": ev, "n_candidates": len(cs),
            "best_endpoint_px": best_e, "best_route": best_r,
            "selected_route": win_r, "selected_endpoint_px": win_e,
            "n_within_5px": len(within[5.0]),
            "n_within_10px": len(within[10.0]),
            "n_within_15px": len(within[15.0]),
            "routes_within_5px": within[5.0][:8],
        })
    n = len(rows)
    out = {
        "run": str(run), "tol_px": TOL_PX,
        "n_events": n,
        "n_endpoint_oracle_within_tol": n_oracle,
        "n_selected_endpoint_within_tol": n_sel,
        "events": rows,
        "target": "all six developed events offering a <=5-px front "
                  "candidate with high correct selection (review 8B)",
        "gate_passed": bool(n) and n_oracle == n and n_sel == n,
        "note": "the perturbation gate (rev11_support_perturbation.json) "
                "shows the front head localizes <=1.2 px on ALL SIX "
                "events when the support is oracle+24; the shortfall here "
                "is the RUNTIME proposals' supports, which is what the "
                "proposal experiment retrains against",
    }
    outp = Path(a.out) if a.out else (run / "endpoint_gate.json")
    outp.write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(f"{r['event']:>12} | best {r['best_endpoint_px']:6.2f} "
              f"({r['best_route']}) | selected "
              f"{('%.2f' % r['selected_endpoint_px']) if r['selected_endpoint_px'] is not None else 'n/a':>6} "
              f"({r['selected_route']}) | within5 {r['n_within_5px']} "
              f"within10 {r['n_within_10px']} within15 {r['n_within_15px']}")
    print(f"\nendpoint-oracle {n_oracle}/{n}, selected {n_sel}/{n} "
          f"-> gate_passed {out['gate_passed']} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
