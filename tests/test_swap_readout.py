"""rev8 step 3: the query-swap readout must be able to TELL the two
outcomes apart.

Diagonal-dominant = the query selects the tube (whole-instance
behaviour). Identical rows = query-invariant output — the failure the
review measured, where a distal answer cannot depend on the root
prompt. If the summary could not separate these, the check would be
worthless.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.eval_whole_instance import summarize_swap  # noqa: E402

OWNERS = ["g0", "g1", "g2"]


def test_diagonal_dominant_is_reported_as_swapping():
    iou = {q: {t: (0.62 if q == t else 0.03) for t in OWNERS}
           for q in OWNERS}
    s = summarize_swap(iou, OWNERS)
    assert s["diag_mean_iou"] > 0.5
    assert s["offdiag_mean_iou"] < 0.1
    assert s["rows_peaking_on_own_tube"] == 3
    assert s["max_row_spread_iou"] > 0.5, s


def test_query_invariant_rows_are_reported_as_no_swap():
    """Every query produces the SAME body: the model ignores the query."""
    same = {t: 0.4 for t in OWNERS}
    iou = {q: dict(same) for q in OWNERS}
    s = summarize_swap(iou, OWNERS)
    assert s["max_row_spread_iou"] == 0.0, s
    assert abs(s["diag_mean_iou"] - s["offdiag_mean_iou"]) < 1e-12, s
    # rev8: a tie is NOT a peak. Counting `diag >= offdiag` made every
    # row "peak on its own tube", so a model that answered the same
    # body for every query scored 3/3 on the very test built to catch
    # it -- and an all-zero (collapsed) model scored 3/3 as well.
    assert s["rows_peaking_on_own_tube"] == 0, s
    # ...and the spread is what exposes it
    assert s["degenerate"] is False, s          # it predicts, just equally


def test_partial_swapping_scores_in_between():
    iou = {"g0": {"g0": 0.5, "g1": 0.45, "g2": 0.02},
           "g1": {"g0": 0.40, "g1": 0.5, "g2": 0.03},
           "g2": {"g0": 0.02, "g1": 0.03, "g2": 0.5}}
    s = summarize_swap(iou, OWNERS)
    assert s["rows_peaking_on_own_tube"] == 3
    assert 0.3 < s["max_row_spread_iou"] < 0.5, s
