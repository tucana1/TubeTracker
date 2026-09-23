"""rev13 W3: the bounded-decision gate's verdict logic.

The gate must name the failing term, refuse a pass when the cap scorer
does not beat the trivial baselines, refuse when confident decoys are
accepted, and demand the six dev tips within the 5 px endpoint gate
(selected top-1; the endpoint oracle is diagnostic only).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _run(tmp_path, d, scope="cap-only", body=None) -> dict:
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    outp = tmp_path / "gate.json"
    extra = ["--scope", scope]
    if body is not None:
        bp = tmp_path / "body.json"
        bp.write_text(json.dumps(body))
        extra += ["--body-fit", str(bp)]
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/rev13_w3_gate.py"),
         "--eval", str(p), "--out", str(outp)] + extra,
        capture_output=True, text=True, cwd=str(REPO))
    return {"rc": r.returncode, "res": json.loads(outp.read_text())}


def _eval_json(cap_conf, top1, oracle, n_ev=6) -> dict:
    return {"checkpoint": "x", "panel": "dev",
            "cap_scorer_confusion": dict(
                {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                 "unknown_band": 0, "n_rows": 0}, **cap_conf),
            "top1_within_5px": top1, "top3_within_5px": top1,
            "endpoint_oracle_coverage_events": oracle,
            "n_events": n_ev}


def test_gate_refuses_constant_classifier(tmp_path):
    # all-present: tp=23 fp=212 -> accuracy == always-present baseline
    out = _run(tmp_path, _eval_json({"tp": 23, "fp": 212, "n_rows": 235},
                                    top1=1, oracle=6))
    assert out["rc"] == 1
    assert "cap_discrimination_above_baselines" in \
        out["res"]["failed_terms"]
    assert "zero_confident_decoy_acceptance" in out["res"]["failed_terms"]


def test_gate_passes_only_with_real_separation(tmp_path):
    out = _run(tmp_path, _eval_json({"tp": 23, "tn": 212, "n_rows": 235},
                                    top1=6, oracle=6))
    assert out["rc"] == 0 and not out["res"]["failed_terms"]


def test_gate_requires_the_six_tips(tmp_path):
    out = _run(tmp_path, _eval_json({"tp": 23, "tn": 212, "n_rows": 235},
                                    top1=3, oracle=5))
    assert out["rc"] == 1
    assert "six_dev_tips_within_5px" in out["res"]["failed_terms"]
    # Candidate availability can never substitute for correct selection.
    out2 = _run(tmp_path, _eval_json({"tp": 23, "tn": 212, "n_rows": 235},
                                     top1=3, oracle=6))
    assert out2["rc"] == 1
    assert "six_dev_tips_within_5px" in out2["res"]["failed_terms"]


def test_combined_requires_body_evidence(tmp_path):
    good = _eval_json({"tp": 23, "tn": 212, "n_rows": 235}, 6, 6)
    out = _run(tmp_path, good, scope="combined")
    assert out["rc"] == 1
    assert "body_fit_target" in out["res"]["failed_terms"]


def test_empty_negative_class_and_missed_positive_cannot_pass(tmp_path):
    for confusion in ({"tp": 23, "n_rows": 23},
                      {"tp": 22, "fn": 1, "tn": 212, "n_rows": 235}):
        out = _run(tmp_path, _eval_json(confusion, 6, 6))
        assert out["rc"] == 1


def test_gate_names_every_failing_term(tmp_path):
    out = _run(tmp_path, _eval_json({"fp": 40, "tp": 10, "tn": 20,
                                     "fn": 5, "n_rows": 75},
                                    top1=0, oracle=1))
    assert out["rc"] == 1
    assert set(out["res"]["failed_terms"]) == {
        "cap_discrimination_above_baselines",
        "zero_confident_decoy_acceptance",
        "six_dev_tips_within_5px"}
    assert "fit-gate NOT passed" in out["res"]["verdict"]


def test_within_row_hinge_semantics():
    """rev13 W3.5b: zero once the best positive beats the best negative
    of the same row by the margin; positive otherwise; gradient flows."""
    import numpy as np
    import torch
    from scripts.rev11_proposal_fit import within_row_hinge
    pos = np.array([False, True, False, False])
    neg = np.array([False, False, True, True])
    good = torch.tensor([0.0, 2.0, 0.5, 0.9], requires_grad=True)
    assert float(within_row_hinge(good, pos, neg, 1.0)) == 0.0
    bad = torch.tensor([0.0, 0.2, 1.5, 0.4], requires_grad=True)
    h = within_row_hinge(bad, pos, neg, 1.0)
    assert abs(float(h) - (1.5 - 0.2 + 1.0)) < 1e-6
    h.backward()
    assert bad.grad is not None and float(bad.grad[1]) < 0  # pos pushed up
    assert float(bad.grad[2]) > 0                          # neg pushed down
