"""rev11 metric-domain regression fixtures.

The rev11 review: the runner's `proposal_coverage` measured distance from
the gold tip to the full support polyline and called it front
availability; a long line passing through the tip while ending at a
grain rim is NOT a front. These fixtures pin the three domains apart:

* Fixture A — the support line passes exactly through the gold tip but
  the predicted cutoff stops 30 px early: support distance 0, measured
  endpoint error 30, endpoint availability false at 5 px.
* Fixture B — the current path overshoots the true cap: current-path
  proximity passes (the path crosses the tip) while endpoint accuracy
  fails.

Both call `evaluate.candidate_metrics`, the exact function the movie
runner uses for its coverage rows, so a fixture pass is a runner-domain
statement, not a re-implementation.
"""
from __future__ import annotations

import pytest

from prototypes.v30_video_apex.evaluate import (
    candidate_metrics, conventional_median, upper_middle)


def test_early_cutoff_fixture_support_vs_endpoint():
    """Support line through the tip, predicted cutoff 30 px early."""
    gold = (100.0, 0.0)
    support = [(0.0, 0.0), (200.0, 0.0)]          # passes through gold
    current = [(0.0, 0.0), (70.0, 0.0)]           # stops 30 px early
    tip = (70.0, 0.0)
    m = candidate_metrics(gold, support, current, tip)
    assert m["support_px"] == pytest.approx(0.0, abs=1e-9)
    assert m["endpoint_px"] == pytest.approx(30.0, abs=1e-9)
    assert m["current_px"] == pytest.approx(30.0, abs=1e-9)
    # the endpoint-domain availability at 5 px is FALSE, even though the
    # support-domain proximity is 0 — the confusion the review found
    assert not (m["endpoint_px"] <= 5.0)
    assert m["support_px"] <= 5.0


def test_overshoot_fixture_current_path_passes_endpoint_fails():
    """The measured path crosses the true cap and keeps going: proximity
    passes while endpoint accuracy fails."""
    gold = (100.0, 0.0)
    support = [(0.0, 0.0), (160.0, 0.0)]
    current = [(0.0, 0.0), (160.0, 0.0)]          # overshoots by 60 px
    tip = (160.0, 0.0)
    m = candidate_metrics(gold, support, current, tip)
    assert m["current_px"] == pytest.approx(0.0, abs=1e-9)
    assert m["endpoint_px"] == pytest.approx(60.0, abs=1e-9)
    assert m["current_px"] <= 5.0 and not (m["endpoint_px"] <= 5.0)


def test_missing_current_path_is_unavailable_not_support():
    """No measured path: the measurement domain reports unavailable —
    it must not silently fall back to support geometry."""
    gold = (100.0, 0.0)
    m = candidate_metrics(gold, [(0.0, 0.0), (200.0, 0.0)], None, None)
    assert m["support_px"] == pytest.approx(0.0, abs=1e-9)
    assert m["current_px"] is None
    assert m["endpoint_px"] is None


def test_median_convention_matches_the_review_numbers():
    """The documented convention: conventional median of the six
    PUBLISHED endpoint errors is 72.41 px, while the old upper-middle
    statistic reports 86.53 px. The preserved-replay set: 89.23 / 89.86.
    (Values from the rev11 candidate audit.)"""
    published = [11.006714342229765, 86.52798509430258,
                 177.1475373805689, 58.29048274652777,
                 88.59343111880366, 5.662730809276109]
    assert round(conventional_median(published), 2) == 72.41
    assert round(upper_middle(published), 2) == 86.53
    preserved = [91.07885987974406, 86.52798509430258, 109.08057281393313,
                 89.86101438204412, 88.59343111880366, 5.662730809276109]
    assert round(conventional_median(preserved), 2) == 89.23
    assert round(upper_middle(preserved), 2) == 89.86
    # even n: the conventional median is the average of the two middle
    assert conventional_median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_score_probes_reports_both_medians():
    """score_probes labels both statistics explicitly (no field named
    'median' may carry the upper-middle value)."""
    from prototypes.v30_video_apex.evaluate import score_probes
    gold = [{"movie_id": "m", "owner_id": f"o{i}", "source_frame": i,
             "x_native": 0.0, "y_native": 0.0, "observation": "observed"}
            for i in range(1, 7)]
    preds = [{"movie_id": "m", "owner_id": f"o{i}", "source_frame": i,
              "x_native": float(i) * 10.0, "y_native": 0.0,
              "tip_score": 1.0, "selected": True}
             for i in range(1, 7)]
    rep = score_probes(preds, gold)
    # errors: 10, 20, 30, 40, 50, 60 -> conventional (30+40)/2, upper 40
    assert rep["median_selected_error_px"] == 35.0
    assert rep["upper_middle_selected_error_px"] == 40.0
    assert "conventional median" in rep["median_convention"]
