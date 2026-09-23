"""rev12 P1.2: ownership interfaces + joint assignment tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.ownership import (  # noqa: E402
    FrameEvidence, Owner, OwnerFrameState, RouteHypothesis, joint_assign)


def _route(owner, frame, tip, score, alt=None):
    # rev13 W2: the current path ends AT the tip (the support end is
    # context only); cap evidence lives on tip_xy/current_path_xy.
    r = RouteHypothesis(owner_id=owner, frame=frame, route_id=f"r{frame}",
                        attachment=(0.0, 0.0),
                        support_xy=[[0.0, 0.0], [10.0, 0.0], list(tip)],
                        current_prefix_len_px=20.0,
                        local_cap_score=score, whole_route_score=score,
                        current_path_xy=[[0.0, 0.0], [10.0, 0.0],
                                         list(tip)],
                        tip_xy=(float(tip[0]), float(tip[1])),
                        cap_clear=bool(score >= 0.65))
    if alt is not None:
        r.alternatives = [alt]
    return r


def _state(owner, frame, tip, score, alt_score=None):
    alt = (_route(owner, frame, (tip[0] + 30, tip[1]), alt_score)
           if alt_score is not None else None)
    return OwnerFrameState(owner_id=owner, frame=frame, state="present",
                           accepted=_route(owner, frame, tip, score,
                                           alt=alt),
                           alternatives=[alt] if alt else [],
                           measurement_domain="partial",
                           time_provenance="none (frames only)")


def test_interface_validation():
    FrameEvidence(movie="ld", source_frame=1, crop_xywh=(0, 0, 8, 8),
                  validity_frac=1.0).validate()
    with pytest.raises(ValueError):
        FrameEvidence(movie="ld", source_frame=1, crop_xywh=(0, 0),
                      validity_frac=1.0).validate()
    Owner(owner_id="o", grain_id="g", movie="ld",
          grain_native=(1.0, 2.0), grain_radius_px=13.0).validate()
    with pytest.raises(ValueError):
        Owner(owner_id="o", grain_id="", movie="ld",
              grain_native=(1.0, 2.0), grain_radius_px=13.0).validate()
    st = OwnerFrameState(owner_id="o", frame=1, state="present")
    with pytest.raises(ValueError):
        st.validate()  # present without an accepted hypothesis


def test_joint_assign_removes_duplicate_clear_cap():
    # two owners, both clearly identified (margin >= 0.25), tips 2 px
    # apart: duplicate ownership of one cap — the weaker is demoted to
    # identity_uncertain with its hypothesis retained as alternative
    s1 = _state("o1", 10, (100.0, 100.0), 0.9, alt_score=0.5)
    s2 = _state("o2", 10, (101.5, 100.5), 0.7, alt_score=0.3)
    res = joint_assign({"o1": [s1], "o2": [s2]})
    assert res["n_duplicate_resolved"] == 1
    assert s2.state == "identity_uncertain"
    assert s2.accepted is None
    assert s2.alternatives  # retained
    assert s1.state == "present"
    assert res["swaps"][0]["kept"] == "o1"


def test_joint_assign_permits_uncertain_colocation():
    # co-located tips WITHOUT a clear margin stay permitted
    s1 = _state("o1", 10, (100.0, 100.0), 0.6, alt_score=0.55)
    s2 = _state("o2", 10, (101.0, 100.0), 0.6, alt_score=0.55)
    res = joint_assign({"o1": [s1], "o2": [s2]})
    assert res["n_duplicate_resolved"] == 0
    assert s1.state == "present" and s2.state == "present"
    assert "permitted" in s1.note or "permitted" in s2.note


def test_joint_assign_distant_caps_untouched():
    s1 = _state("o1", 10, (100.0, 100.0), 0.9, alt_score=0.5)
    s2 = _state("o2", 10, (160.0, 100.0), 0.9, alt_score=0.5)
    res = joint_assign({"o1": [s1], "o2": [s2]})
    assert res["n_duplicate_resolved"] == 0
    assert s1.note == "" and s2.note == ""
