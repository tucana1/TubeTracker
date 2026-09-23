"""rev14 W2: frame-specific root evidence from human FULL traces.

A human FULL trace licenses a root for the frame it was drawn on (point 1
of the reviewed centreline is the reviewed grain exit). A PARTIAL or
hidden trace never does; workflow-test records never do; a withdrawal
shadows the root; and the root never travels to another frame.
"""
import numpy as np
import pytest

from tubetracker.analysis_contracts import (
    AnalysisRequest, resolve_review_constraints, stable_hash)
from tubetracker.movie_analysis import frame_root_evidence, route_owner_for


def observation(owner="a", **extra):
    base = dict(movie="m", owner_uuid=owner, source_frame=30,
                direct_state="direct_visible", direct_xy=[50., 32.],
                source="review", source_id="obs", source_revision=1)
    base.update(extra)
    return base


def request():
    return AnalysisRequest("", "m", [30], owners=[{"id": "a"}, {"id": "b"}])


def route_key(constraints):
    return stable_hash({
        "corrections": [{"owner": k[0], "frame": k[1], **v}
                        for k, v in sorted(constraints.items())],
        "frame_roots": frame_root_evidence(constraints)})


def test_full_human_trace_licenses_a_frame_scoped_root():
    path = [[11.0, 32.0], [30.0, 32.0], [50.0, 32.0]]
    o = observation(path_xy=path, path_complete=True, review_origin="human",
                    source_id="obs-full", source_revision=2)
    constraints, _ = resolve_review_constraints([o], request(), request().owners)
    c = constraints[("a", 30)]
    vr = c["verified_root"]
    assert vr is not None
    assert vr["xy"] == [11.0, 32.0]      # point 1 = the reviewed grain exit
    assert vr["scope_frame"] == 30
    assert vr["observation_id"] == "obs-full"
    assert vr["revision"] == 2
    roots = frame_root_evidence(constraints)
    assert roots[30]["a"]["xy"] == [11.0, 32.0]


def test_partial_or_hidden_trace_never_licenses_a_root():
    # The real 3756 shape: a PARTIAL visible span plus a SEPARATE precise
    # tip review. The tip survives, but no root is licensed.
    partial = observation(path_xy=[[11., 32.], [50., 32.]],
                          path_complete=False, source_id="obs-part")
    tip = observation(tip_source="explicit_point", source_id="obs-tip")
    hidden = observation(owner="b", direct_state="not_directly_visible",
                         path_xy=[[11., 32.], [50., 32.]],
                         path_complete=True, source_id="obs-h")
    constraints, _ = resolve_review_constraints([partial, tip, hidden],
                                                request(), request().owners)
    c = constraints[("a", 30)]
    assert c["tip_xy"] == [50., 32.]          # the independent tip survives
    assert c["path_complete"] is False
    assert c["verified_root"] is None         # the PARTIAL start is not a root
    assert constraints[("b", 30)]["verified_root"] is None
    assert frame_root_evidence(constraints) == {}


def test_bare_partial_path_is_not_a_tip_and_not_a_root():
    # A path endpoint alone is never licensed as a tip (P0 rule), so a
    # bare partial draft produces no constraint and no root at all.
    partial = observation(path_xy=[[11., 32.], [50., 32.]],
                          path_complete=False, source_id="obs-bare")
    constraints, ignored = resolve_review_constraints([partial], request(),
                                                      request().owners)
    assert ("a", 30) not in constraints
    assert any('tip' in str(i.get('reason', '')) for i in ignored)


def test_workflow_origin_never_licenses_a_root():
    o = observation(path_xy=[[11., 32.], [50., 32.]], path_complete=True,
                    review_origin="workflow_test")
    constraints, _ = resolve_review_constraints([o], request(), request().owners)
    assert constraints[("a", 30)]["verified_root"] is None


def test_withdrawn_review_shadows_its_root():
    live = observation(path_xy=[[11.0, 32.0], [50.0, 32.0]],
                       path_complete=True, obs_uuid="obs-x", obs_revision=2)
    constraints, _ = resolve_review_constraints([live], request(), request().owners)
    assert constraints[("a", 30)]["verified_root"] is not None
    withdrawn = dict(live, review_status="withdrawn")
    constraints, ignored = resolve_review_constraints([withdrawn], request(),
                                                      request().owners)
    assert ("a", 30) not in constraints
    assert frame_root_evidence(constraints) == {}
    assert any('withdrawn' in str(i.get('reason', '')) for i in ignored)


def test_route_owner_override_is_frame_scoped_and_never_transported():
    owner = {"id": "a", "grain_native": [100.0, 100.0],
             "attachment_native": [90.0, 90.0], "attachment_verified": True}
    roots = {30: {"a": {"xy": [11.0, 32.0], "observation_id": "obs",
                        "revision": 2, "scope_frame": 30,
                        "basis": "reviewed FULL path exit"}}}
    qo, ev = route_owner_for(owner, roots, 30)
    assert qo is not owner and qo["attachment_native"] == [11.0, 32.0]
    assert qo["attachment_verified"] is True
    assert qo["attachment_source"]["observation_id"] == "obs"
    assert owner["attachment_native"] == [90.0, 90.0]   # static untouched
    other, ev2 = route_owner_for(owner, roots, 60)      # no transport
    assert other is owner and ev2 is None
    assert route_owner_for(owner, {}, 30)[1] is None


def test_revision_or_withdrawal_changes_the_route_cache_inputs():
    path = [[11.0, 32.0], [50.0, 32.0]]
    base = observation(path_xy=path, path_complete=True, obs_uuid="obs-r",
                       obs_revision=1)
    c1, _ = resolve_review_constraints([base], request(), request().owners)
    k1 = route_key(c1)
    rev = dict(base, source_revision=2)
    c2, _ = resolve_review_constraints([rev], request(), request().owners)
    k2 = route_key(c2)
    assert k1 != k2                                    # revision invalidates
    w, _ = resolve_review_constraints([dict(base, review_status="withdrawn")],
                                      request(), request().owners)
    assert route_key(w) != k1                          # withdrawal invalidates
