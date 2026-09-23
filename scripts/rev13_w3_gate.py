"""rev13 W3 bounded-decision gate — PREDECLARED thresholds.

Applied to a capwindow eval artifact (scripts/rev11_proposal_fit.py
--mode eval --scorer capwindow). The thresholds below were declared in
the rev13 work order before this run; they do not move after seeing
the numbers. The gate names which term fails rather than declaring a
pass.

Gates (fit/plumbing criteria, NOT a claimed validation result):
- cap-scorer present/absent discrimination: accuracy strictly above
  BOTH trivial baselines (always-present / always-absent) over the
  labeled rows, with the unknown band kept separate;
- zero confident decoy acceptance: no absent row (present==0) with
  cap_max_prob >= 0.5;
- the six precise development tips: aim for all six within the 5 px
  endpoint gate, using the actual selected top-1 (an oracle is diagnostic,
  never a selection pass — an 8 px corridor-positive route is not
  automatically capable of the 5 px gate, so native refinement counts);
- body target (existing): IoU >= 0.90 with rival-positive < 0.05 —
  reported from the body fit artifact, not recomputed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

THRESHOLDS = {
    "decoy_confident_prob": 0.5,
    "endpoint_gate_px": 5.0,
    "n_dev_tips": 6,
    "body_iou_ge": 0.90,
    "rival_positive_lt": 0.05,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--body-fit", default="")
    ap.add_argument("--scope", choices=("combined", "cap-only"), default="combined")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = json.loads(Path(a.eval).read_text())
    n_ev = int(d.get("n_events") or 0)
    cc = d.get("cap_scorer_confusion") or {}
    tp, fp, tn, fn = (int(cc.get(k, 0)) for k in ("tp", "fp", "tn", "fn"))
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(1, n)
    base_p = (tp + fn) / max(1, n)   # always-present accuracy
    base_a = (tn + fp) / max(1, n)   # always-absent accuracy
    top1 = int(d.get("top1_within_5px") or 0)
    oracle = int(d.get("endpoint_oracle_coverage_events") or 0)
    top3 = int(d.get("top3_within_5px") or 0)

    gates = {
        "cap_discrimination_above_baselines": {
            "passed": acc > max(base_p, base_a) and n > 0 and tp > 0 and fn == 0,
            "accuracy": round(acc, 4),
            "always_present": round(base_p, 4),
            "always_absent": round(base_a, 4),
            "n_rows": n},
        "zero_confident_decoy_acceptance": {
            "passed": fp == 0 and tn > 0,
            "false_positives": fp,
            "threshold": THRESHOLDS["decoy_confident_prob"]},
        "six_dev_tips_within_5px": {
            "passed": top1 == n_ev and n_ev >= THRESHOLDS["n_dev_tips"],
            "top1": top1, "top3": top3, "endpoint_oracle": oracle,
            "n_events": n_ev, "gate_px": THRESHOLDS["endpoint_gate_px"]},
    }
    if a.scope == "combined":
        gates["body_fit_target"] = {"passed": False, "reason": "required body evidence missing"}
    if a.body_fit and Path(a.body_fit).exists():
        bf = json.loads(Path(a.body_fit).read_text())
        fits = bf.get("fit_gate") or []
        ok = bool(fits) and all(
            f.get("iou_ge_090") and f.get("rivals_below_005")
            for f in fits)
        gates["body_fit_target"] = {
            "passed": ok,
            "artifact": a.body_fit,
            "per_owner": [{"owner": f.get("owner"),
                           "iou": f.get("iou"),
                           "rivals": f.get("rival_rate")}
                          for f in fits]}
    failed = [k for k, v in gates.items() if not v["passed"]]
    res = {
        "eval_artifact": a.eval,
        "checkpoint": d.get("checkpoint"),
        "panel": d.get("panel"),
        "scope": a.scope,
        "thresholds_predeclared": THRESHOLDS,
        "gates": gates,
        "failed_terms": failed,
        "verdict": ("fit-gate PASSED (fit/plumbing only — not a "
                    "generalization claim)" if not failed else
                    "fit-gate NOT passed: " + ", ".join(failed)),
        "note": "the six-event panel is a fit/dev panel; passing it is "
                "not validation. After fit, generalization requires a "
                "held-out owner/interval panel with coverage and error "
                "reported together (predeclare its operating thresholds "
                "before examining it).",
    }
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
