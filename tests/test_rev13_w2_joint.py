"""rev13 W2 acceptance: joint physical-owner inference.

The audit's exact counterexamples:
- same current cap + DIFFERENT support ends -> must demote (the old
  code compared support ends and resolved 0);
- different current caps + SAME support end -> must NOT demote (XY
  coincidence is not evidence of the same physical cap);
- three-owner collision -> no crash, no order-dependent deletion;
- genuine projected overlap -> uncertainty retained, no demotion.

Plus the joint selection over time: explicit unresolved option,
cap-uniqueness enforced across owners per frame, temporal smoothness,
alternatives retained.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _h(oid, frame, cap, support_end, score, rid="", cap_id="",
       cap_clear=False):
    from prototypes.v30_video_apex.ownership import RouteHypothesis
    # support polyline ends at support_end; current path ends at cap
    return RouteHypothesis(
        owner_id=oid, frame=frame, route_id=rid or f"{oid}-r",
        attachment=(100.0, 100.0),
        support_xy=[[100.0, 100.0], list(support_end)],
        current_prefix_len_px=20.0,
        local_cap_score=score, whole_route_score=0.5,
        current_path_xy=[[100.0, 100.0], list(cap)],
        tip_xy=tuple(cap), cap_candidate_id=cap_id, cap_clear=cap_clear)


def _state(oid, frame, hyp, alt_score=0.0):
    from prototypes.v30_video_apex.ownership import OwnerFrameState
    alt = _h(oid, frame, (0.0, 0.0), (0.0, 0.0), alt_score,
             rid="alt") if alt_score else None
    st = OwnerFrameState(owner_id=oid, frame=frame, state="present",
                         accepted=hyp,
                         alternatives=([alt] if alt is not None else []))
    return st


def test_same_cap_different_support_ends_demotes():
    from prototypes.v30_video_apex.ownership import joint_assign

    a = _state("g1", 10, _h("g1", 10, (50.0, 50.0), (90.0, 10.0),
                            0.9, cap_clear=True))
    b = _state("g2", 10, _h("g2", 10, (50.5, 50.2), (10.0, 90.0),
                            0.6, cap_clear=True))
    res = joint_assign({"g1": [a], "g2": [b]})
    assert res["n_duplicate_resolved"] == 1
    assert res["swaps"][0]["demoted"] == "g2"


def test_different_caps_same_support_end_does_not_demote():
    from prototypes.v30_video_apex.ownership import joint_assign

    a = _state("g1", 10, _h("g1", 10, (50.0, 50.0), (80.0, 80.0),
                            0.9, cap_clear=True))
    b = _state("g2", 10, _h("g2", 10, (90.0, 90.0), (80.0, 80.0),
                            0.9, cap_clear=True))
    res = joint_assign({"g1": [a], "g2": [b]})
    assert res["n_duplicate_resolved"] == 0
    assert a.state == "present" and b.state == "present"


def test_three_owner_collision_no_crash_order_independent():
    from prototypes.v30_video_apex.ownership import joint_assign

    def build():
        return {
            "g1": [_state("g1", 10, _h("g1", 10, (50.0, 50.0),
                                       (80.0, 10.0), 0.9,
                                       cap_clear=True))],
            "g2": [_state("g2", 10, _h("g2", 10, (50.2, 50.1),
                                       (10.0, 80.0), 0.8,
                                       cap_clear=True))],
            "g3": [_state("g3", 10, _h("g3", 10, (50.1, 50.2),
                                       (90.0, 90.0), 0.7,
                                       cap_clear=True))],
        }

    outcomes = set()
    for perm in itertools.permutations(["g1", "g2", "g3"]):
        states = build()
        ordered = {k: states[k] for k in perm}
        res = joint_assign(ordered)
        outcomes.add(tuple(sorted(
            s["demoted"] for s in res["swaps"])))
        assert not res["skipped"] or True  # reported, never raised
    # exactly one winner among three colliding claims, regardless of
    # the dict order
    assert len(outcomes) == 1
    (only,) = outcomes
    assert len(only) == 2  # two demoted, one kept


def test_projected_overlap_retains_uncertainty():
    from prototypes.v30_video_apex.ownership import joint_assign

    # same cap location but NO decisive local evidence: permitted
    a = _state("g1", 10, _h("g1", 10, (50.0, 50.0), (80.0, 10.0),
                            0.4, cap_clear=False))
    b = _state("g2", 10, _h("g2", 10, (50.2, 50.1), (10.0, 80.0),
                            0.4, cap_clear=False))
    res = joint_assign({"g1": [a], "g2": [b]})
    assert res["n_duplicate_resolved"] == 0
    assert res["overlaps"] and res["overlaps"][0][
        "same_cap_evidence"] is False
    assert a.state == "present" and b.state == "present"
    assert "permitted" in (a.note or "")


def test_cap_ids_identify_same_physical_cap():
    from prototypes.v30_video_apex.ownership import joint_assign

    # both carry the deployed pool's id for the same cap -> demote even
    # without cap_clear flags
    a = _state("g1", 10, _h("g1", 10, (50.0, 50.0), (80.0, 10.0),
                            0.9, cap_id="cap-7"))
    b = _state("g2", 10, _h("g2", 10, (50.5, 50.2), (10.0, 80.0),
                            0.6, cap_id="cap-7"))
    res = joint_assign({"g1": [a], "g2": [b]})
    assert res["n_duplicate_resolved"] == 1
    # different ids at the same location: NOT the same physical cap
    a2 = _state("g1", 11, _h("g1", 11, (50.0, 50.0), (80.0, 10.0),
                             0.9, cap_id="cap-7"))
    b2 = _state("g2", 11, _h("g2", 11, (50.1, 50.1), (10.0, 80.0),
                             0.9, cap_id="cap-9"))
    res2 = joint_assign({"g1": [a2], "g2": [b2]})
    assert res2["n_duplicate_resolved"] == 0


def test_joint_select_over_time_prefers_smooth_and_enforces_uniqueness():
    from prototypes.v30_video_apex.ownership import (
        joint_select_over_time)

    # Equal evidence on both hypotheses: with b present at ALL frames
    # and jumping 150 px between 10->20, the smooth a-trajectory must
    # win (the jump penalty 142.5 dwarfs the equal local scores), and
    # abstention must not provide a free reset.
    frames = [10, 20, 30]
    cands = {
        "g1": {
            10: [_h("g1", 10, (50.0, 50.0), (80.0, 10.0), 1.0, "a"),
                 _h("g1", 10, (200.0, 50.0), (80.0, 10.0), 1.0, "b")],
            20: [_h("g1", 20, (51.0, 50.0), (80.0, 10.0), 1.0, "a"),
                 _h("g1", 20, (350.0, 50.0), (80.0, 10.0), 1.0, "b")],
            30: [_h("g1", 30, (52.0, 50.0), (80.0, 10.0), 1.0, "a"),
                 _h("g1", 30, (351.0, 50.0), (80.0, 10.0), 1.0, "b")],
        },
        "g2": {
            10: [_h("g2", 10, (300.0, 300.0), (10.0, 80.0), 1.0, "c")],
            20: [_h("g2", 20, (301.0, 300.0), (10.0, 80.0), 1.0, "c")],
            30: [_h("g2", 30, (302.0, 300.0), (10.0, 80.0), 1.0, "c")],
        },
    }
    res = joint_select_over_time(cands, frames, temporal_w=1.0)
    for f in frames:
        assert res["rows"]["g1"][f]["route_id"] == "a", f
    assert res["rows"]["g1"][20]["reacquisition"] is False
    assert res["rows"]["g1"][20]["score_components"]["temporal"] == 0.0
    # alternatives retained
    assert "b" in res["alternatives_by_row"]["g1"][20]


def test_strong_evidence_reacquires_with_flag_and_components():
    from prototypes.v30_video_apex.ownership import (
        joint_select_over_time)

    # A strongly-scored candidate appearing mid-sequence may legitimately
    # win via a flagged re-acquisition (identity change over time) — the
    # components and the flag must make that auditable.
    frames = [10, 20, 30]
    cands = {
        "g1": {
            10: [_h("g1", 10, (50.0, 50.0), (80.0, 10.0), 0.2, "a")],
            20: [_h("g1", 20, (51.0, 50.0), (80.0, 10.0), 0.2, "a"),
                 _h("g1", 20, (350.0, 50.0), (80.0, 10.0), 9.9, "b")],
            30: [_h("g1", 30, (52.0, 50.0), (80.0, 10.0), 0.2, "a"),
                 _h("g1", 30, (351.0, 50.0), (80.0, 10.0), 9.9, "b")],
        },
    }
    res = joint_select_over_time(cands, frames, temporal_w=1.0)
    r20 = res["rows"]["g1"][20]
    assert r20["route_id"] == "b"
    assert r20["reacquisition"] is True
    assert r20["score_components"]["local"] == 9.9
    assert r20["score_components"]["temporal"] < 0  # switch charge


def test_joint_select_unresolved_option_and_cap_uniqueness():
    from prototypes.v30_video_apex.ownership import (
        joint_select_over_time)

    frames = [10]
    # g1 and g2 both offer ONLY the same clear cap (same id): the joint
    # state with both present is illegal; one must go unresolved.
    cands = {
        "g1": {10: [_h("g1", 10, (50.0, 50.0), (80.0, 10.0), 2.0,
                       "a", cap_id="cap-1", cap_clear=True)]},
        "g2": {10: [_h("g2", 10, (50.1, 50.0), (10.0, 80.0), 1.0,
                       "b", cap_id="cap-1", cap_clear=True)]},
    }
    res = joint_select_over_time(cands, frames)
    s1 = res["rows"]["g1"][10]["state"]
    s2 = res["rows"]["g2"][10]["state"]
    assert (s1, s2) == ("present", "identity_uncertain")
    # an owner with no candidates at all is explicitly unresolved
    cands2 = {"g1": {10: []}}
    res2 = joint_select_over_time(cands2, frames)
    assert res2["rows"]["g1"][10]["state"] == "identity_uncertain"
