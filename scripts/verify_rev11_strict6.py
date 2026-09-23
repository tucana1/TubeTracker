"""Verify rev11_strict6 against the rev11 audit's saved counterfactual.

Checks, all as hard assertions:
  1. 220 candidates; candidate IDs identical to the audit's replay.
  2. route_p range == audit's preserved-head range (float-exact).
  3. selected winners identical to the audit's preserved run.
  4. coverage at 5 px: 6/6 support, 2/6 current-path, 1/6 endpoint
     oracle, 0/6 selected — from OUR new report block.
  5. selected endpoint errors identical; conventional median 89.23,
     upper-middle 89.86.
"""
import json
import math
import sys

AUDIT = "/Users/joshjiang/Documents/New project/TubeTracker-rev11-audit-2026-09-17"
RUN = "runs/prototypes/v30/rev11_strict6"

mine_c = json.load(open(f"{RUN}/candidates.json"))
mine_sel = json.load(open(f"{RUN}/selection.json"))
mine_rep = json.load(open(f"{RUN}/report.json"))
audit_c = json.load(open(f"{AUDIT}/movie-route-head-preserved/candidates.json"))
audit_sel = json.load(open(f"{AUDIT}/movie-route-head-preserved/selection.json"))
audit_cand = json.load(open(f"{AUDIT}/candidate-audit.json"))

fails = []


def check(name, got, want, ok):
    print(f"{'OK ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        fails.append(name)


check("n_candidates", len(mine_c), 220, len(mine_c) == 220)
ids_mine = sorted(c["candidate_id"] for c in mine_c)
ids_audit = sorted(c["candidate_id"] for c in audit_c)
check("candidate_ids_identical", "set-equal" if ids_mine == ids_audit
      else "DIFF", "set-equal", ids_mine == ids_audit)

rps = [c["route_p"] for c in mine_c if c.get("route_p") is not None]
want_lo, want_hi = audit_cand["route_probability_range_preserved"]
check("route_p_min", rps and min(rps), want_lo,
      rps and min(rps) == want_lo)
check("route_p_max", rps and max(rps), want_hi,
      rps and max(rps) == want_hi)

def _wid(w):
    return w.get("route_id") if isinstance(w, dict) else w


w_mine = {s["event"]: _wid(s.get("winner")) for s in mine_sel}
w_audit = {s["event"]: _wid(s.get("winner")) for s in audit_sel}
check("winners", w_mine, w_audit, w_mine == w_audit)

cov = mine_rep.get("proposal_coverage_summary") or {}
check("n_events", cov.get("n_events"), 6, cov.get("n_events") == 6)
check("support_within_tol", cov.get("n_support_within_tol"), 6,
      cov.get("n_support_within_tol") == 6)
check("current_path_within_tol", cov.get("n_current_path_within_tol"), 2,
      cov.get("n_current_path_within_tol") == 2)
check("endpoint_oracle_within_tol", cov.get("n_endpoint_oracle_within_tol"),
      1, cov.get("n_endpoint_oracle_within_tol") == 1)
check("selected_endpoint_within_tol", cov.get("n_selected_endpoint_within_tol"),
      0, cov.get("n_selected_endpoint_within_tol") == 0)

want_end = {e["event"]: e["selected_preserved"]["endpoint_error"]
            for e in audit_cand["events"]}
end_rows = {r["event"]: r["endpoint_error_px"]
            for r in mine_rep["final_selection_error"]["rows"]}
ok = all(math.isclose(end_rows.get(k, -1), v, rel_tol=0, abs_tol=1e-9)
         for k, v in want_end.items())
check("selected_endpoint_errors", end_rows, want_end, ok)

summ = mine_rep["final_selection_error"]["summary"]
check("median_conventional", round(summ["median_endpoint_error_px"], 2),
      89.23, round(summ["median_endpoint_error_px"], 2) == 89.23)
check("upper_middle", round(summ["upper_middle_endpoint_error_px"], 2),
      89.86, round(summ["upper_middle_endpoint_error_px"], 2) == 89.86)

fl = mine_rep.get("front_load") or {}
check("front_load_mode", fl.get("mode"), "strict-current",
      fl.get("mode") == "strict-current")
check("route_head_fallback", fl.get("route_head_fallback"), None,
      fl.get("route_head_fallback") is None)
check("front_manifest_epoch", (fl.get("manifest") or {}).get("epoch"), 19,
      (fl.get("manifest") or {}).get("epoch") == 19)

print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
