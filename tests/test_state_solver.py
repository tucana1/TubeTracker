import math

import pytest

from prototypes.v30_video_apex.state_solver import (
    ConstraintConflict, SolverConfig, solve_joint_states)


def candidate(cid, xy, cap=.9, route=.8, **extra):
    return {"candidate_id": cid, "cap_id": cid, "tip_xy": list(xy),
            "cap_probability": cap, "route_probability": route, **extra}


def test_future_human_point_disambiguates_before_a_hidden_frame():
    candidates = {"a": {0: [candidate("left", (0, 0), cap=.8),
                             candidate("right", (30, 0), cap=.81)],
                         15: [], 30: [candidate("wrong", (30, 0))]}}
    constraints = {("a", 15): {"state": "not_directly_visible", "source": "obs-hidden"},
                   ("a", 30): {"state": "direct_visible", "tip_xy": [0, 0], "source": "obs-point"}}
    result = solve_joint_states(candidates, [0, 15, 30], constraints=constraints)
    early, hidden, last = result["rows"]
    assert early["tip_xy"] == [0, 0] and early["future_assisted_identity"]
    assert hidden["state"] == "occluded" and hidden["tip_xy"] is None
    assert last["tip_xy"] == [0, 0] and last["as_inference_constraint"]
    assert last["current_path_xy"] == [] and last["length_px"] is None


def test_human_point_cannot_lose_to_motion_or_model_score():
    result = solve_joint_states({"a": {0: [candidate("old", (0, 0))], 30: [candidate("high", (1, 0), .999, .999)]}},
                                [0, 30], constraints={("a", 30): {"tip_xy": [300, 0], "source": "human"}})
    row = result["rows"][-1]
    assert row["tip_xy"] == [300, 0] and row["provenance"] == "human-corrected"


def test_three_owners_cannot_claim_one_global_cap():
    owners = {o: {0: [candidate("physical-cap", (50, 50), route=p)]}
              for o, p in [("a", .98), ("b", .4), ("c", .4)]}
    result = solve_joint_states(owners, [0])
    assert sum(r["state"] == "present" for r in result["rows"]) == 1
    assert next(r for r in result["rows"] if r["state"] == "present")["owner_id"] == "a"


def test_distinct_projected_caps_remain_representable():
    result = solve_joint_states({"a": {0: [candidate("cap-a", (50, 50))]},
                                 "b": {0: [candidate("cap-b", (50, 50))]}}, [0])
    assert all(r["state"] == "present" for r in result["rows"])


def test_conflicting_human_ownership_surfaces_conflict():
    with pytest.raises(ConstraintConflict, match="same identified cap"):
        solve_joint_states({"a": {0: []}, "b": {0: []}}, [0],
                           constraints={(o, 0): {"tip_xy": [50, 50], "source": "human"}
                                        for o in ["a", "b"]})


def test_weak_evidence_abstains_and_hidden_preserves_partial_body():
    partial = [[0, 0], [10, 0]]
    result = solve_joint_states({"a": {0: [candidate("weak", (20, 0), .1, .99)],
                                          30: [candidate("visible-body", (22, 0),
                                                          observed_partial_path_xy=partial)]}}, [0, 30],
                                constraints={("a", 30): {"state": "not_directly_visible", "source": "human"}})
    assert result["rows"][0]["state"] == "identity_uncertain"
    assert result["rows"][1]["partial_path_xy"] == partial
    assert result["rows"][1]["length_px"] is None


def test_current_geometry_and_objective_sum_are_exact():
    c = {("a", 0): {"tip_xy": [20, 0], "path_xy": [[0, 0], [20, 0]],
                     "path_complete": True, "source": "review-full"},
         ("a", 30): {"tip_xy": [26, 0], "source": "tip-only"}}
    result = solve_joint_states({"a": {0: [], 30: []}}, [0, 30], constraints=c)
    a, b = result["rows"]
    assert a["length_px"] == 20  # future extent never enters this measurement
    assert b["length_px"] is None and b["score_components"]["temporal"] < 0
    assert math.isclose(sum(sum(r["score_components"].values()) for r in result["rows"]), result["objective"])


def test_many_independent_grains_do_not_form_a_cartesian_search():
    owners = {f"grain-{i:02d}": {f: [
        candidate(f"g{i}-f{f}-c{k}", (i*100+k*12, f/10), cap=.95-k*.1)
        for k in range(4)] for f in (0, 15, 30)} for i in range(30)}
    result = solve_joint_states(owners, [0, 15, 30], config=SolverConfig(beam_width=16))
    assert len(result["rows"]) == 90 and len(result["components"]) == 30
    assert result["n_assignment_expansions"] <= 30*3*5
    assert all(row["tip_xy"] is not None for row in result["rows"])


def test_crowded_search_is_bounded_and_reserves_hard_caps_before_pruning():
    owners = {f"g{i:02d}": {0: [candidate(f"cap-{k}", (10*k, 0)) for k in range(4)]}
              for i in range(12)}
    limit = 16
    result = solve_joint_states(owners, [0],
        constraints={("g11", 0): {"tip_xy": [0, 0], "source": "review"}},
        config=SolverConfig(max_joint_assignments=limit, beam_width=16))
    assert result["assignment_search_truncated"]
    assert result["n_assignment_expansions"] <= 12*limit*5
    reviewed = next(r for r in result["rows"] if r["owner_id"] == "g11")
    assert reviewed["tip_xy"] == [0, 0] and reviewed["as_inference_constraint"]
    caps = [r["cap_id"] for r in result["rows"] if r["cap_id"]]
    assert len(caps) == len(set(caps))
    assert all(not r["search_margin_exhaustive"] for r in result["rows"])


def test_small_joint_search_matches_an_unpruned_budget():
    owners = {o: {f: [candidate("shared-"+str(f), (10, f/10), cap=.92),
                      candidate(o+str(f), (x, f/10), cap=.85)]
                  for f in (0, 30)} for o, x in [("a", 0), ("b", 20)]}
    bounded = solve_joint_states(owners, [0, 30], config=SolverConfig(max_joint_assignments=16))
    exhaustive = solve_joint_states(owners, [0, 30],
        config=SolverConfig(max_joint_assignments=10000, beam_width=10000))
    assert bounded["objective"] == exhaustive["objective"]
    assert [r["selected_candidate_id"] for r in bounded["rows"]] == [
        r["selected_candidate_id"] for r in exhaustive["rows"]]
