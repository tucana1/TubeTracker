"""rev11 item 7 tail: the owned-absence rejection gate semantics.

Pre-declared, all-or-nothing: every human-verified absence grain must
be rejected (owned-visibility argmax == no_tube_visible). One miss
fails the gate — no partial credit, no thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_gate_all_rejected_passes():
    from scripts.rev10_fit_check import _absence_gate
    cases = [{"case": "a", "rejected": True},
             {"case": "b", "rejected": True}]
    g = _absence_gate(cases)
    assert g["n_cases"] == 2 and g["n_rejected"] == 2
    assert g["gate_passed"] is True


def test_gate_one_miss_fails():
    from scripts.rev10_fit_check import _absence_gate
    cases = [{"case": "a", "rejected": True},
             {"case": "b", "rejected": False}]
    g = _absence_gate(cases)
    assert g["gate_passed"] is False
    assert g["n_rejected"] == 1


def test_gate_empty_fails():
    """No cases = no evidence = not a pass (never vacuously true)."""
    from scripts.rev10_fit_check import _absence_gate
    assert _absence_gate([])["gate_passed"] is False
