"""rev8 step-2 contracts: one selector, a sampling-invariant metric,
and an explicit null decision.

The review's counterexample is a regression test here: a two-point
straight line whose single endpoint touches the gold path scored 0.50
under the old vertex-coverage metric but is geometrically almost
entirely off-tube. The new metric must not reward it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.eval_route_evidence import (  # noqa: E402
    path_metrics)
from prototypes.v30_video_apex.selection import (  # noqa: E402
    SelectionPolicy, select_candidate)


def _row(route, ev, sel, route_p=None, q_max=None):
    r = {"route_id": route, "ev": ev, "sel": sel}
    if route_p is not None:
        r["route_p"] = route_p
    if q_max is not None:
        r["q_max"] = q_max
    return r


def test_selector_refuses_with_reasons():
    pol = SelectionPolicy(rank="sel", evidence_gate=0.6,
                          min_score=0.0, min_evidence=0.0)
    d = select_candidate([], pol)
    assert d.refused and d.reason == "no-candidates"
    # every candidate has zero evidence -> no winner can be justified
    d = select_candidate([_row("a", 0.0, 0.5), _row("b", 0.0, 0.4)],
                         SelectionPolicy(rank="sel", evidence_gate=0.6,
                                         min_evidence=0.05))
    assert d.refused and d.reason == "evidence-below-floor"
    # gate empties the pool
    d = select_candidate([_row("a", 1.0, 0.1), _row("b", 0.1, 0.9)],
                         SelectionPolicy(rank="sel", evidence_gate=0.6))
    assert not d.refused and d.winner["route_id"] == "a"
    d = select_candidate([_row("a", 1.0, 0.1)],
                         SelectionPolicy(rank="sel", evidence_gate=6.0))
    assert d.refused and d.reason == "evidence-gate-empty"
    # absolute score floor
    d = select_candidate([_row("a", 1.0, 0.2)],
                         SelectionPolicy(rank="sel", min_score=0.5))
    assert d.refused and d.reason == "score-below-floor"


def test_selector_rank_parity_and_gate_effect():
    rows = [_row("fan", 0.9, 0.95, route_p=0.02, q_max=0.95),
            _row("walker", 0.8, 0.09, route_p=0.90, q_max=0.10)]
    # shipped rule: q_max x route_p lets the fan win
    d = select_candidate(rows, SelectionPolicy(rank="sel"))
    assert d.winner["route_id"] == "fan"
    # ranking by the trained preference alone picks the walker
    d = select_candidate(rows, SelectionPolicy(rank="route_p"))
    assert d.winner["route_id"] == "walker"
    # a relative gate on a third, junk candidate excludes only it
    rows2 = rows + [_row("junk", 0.05, 9.99)]
    d = select_candidate(rows2, SelectionPolicy(rank="sel",
                                                evidence_gate=0.6))
    assert d.winner["route_id"] == "fan"
    assert d.n_kept == 2
    # the decision is serializable provenance
    j = d.as_dict()
    assert j["policy"]["rank"] == "sel" and j["n_candidates"] == 3


def test_path_metrics_rejects_sparse_endpoint_hit():
    """The review's counterexample: one good endpoint of a two-point
    line must not read as a correct route."""
    gold = [[100.0, 300.0], [140.0, 300.0], [180.0, 300.0],
            [220.0, 300.0], [260.0, 300.0]]
    near = [[100.0, 300.0], [104.0, 300.0]]      # genuinely on it
    hit = [[104.0, 300.0], [104.0, 500.0]]       # endpoint on, body off
    m_near = path_metrics(near, gold)
    m_hit = path_metrics(hit, gold)
    assert m_near["recall"] < 0.6  # a stub cannot recall the whole path
    assert m_hit["precision"] < 0.5, m_hit
    assert m_hit["recall"] <= 0.4, m_hit
    assert not m_hit["joint_pass"] and not m_near["joint_pass"]
    # old metric arithmetics: two vertices, one within 5px -> 0.5
    old_like = 1 / 2
    assert m_hit["precision"] < old_like, (m_hit, old_like)
    # a genuinely matching path passes the joint diagnostic
    good = [[float(x), 301.0] for x in range(100, 261, 4)]
    m_good = path_metrics(good, gold)
    assert m_good["precision"] > 0.95 and m_good["recall"] > 0.95
    assert m_good["joint_pass"], m_good


def test_path_metrics_detects_loop_and_length_ratio():
    gold = [[0.0, 0.0], [100.0, 0.0]]
    # a candidate that crosses itself (the loop-around-a-grain and
    # jump-onto-a-crossing signature) must be flagged
    loop = [[0.0, 0.0], [40.0, 40.0], [0.0, 40.0], [40.0, 0.0]]
    m = path_metrics(loop, gold)
    assert m["loop"] is True, m
    assert not m["joint_pass"]
    stub = path_metrics([[0.0, 0.0], [10.0, 0.0]], gold)
    assert stub["length_ratio"] < 0.3 and not stub["joint_pass"]
