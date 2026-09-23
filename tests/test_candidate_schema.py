"""rev9 WP-A.5: one candidate record; the replay must match the run.

The audit replayed the eight saved gated events and got 5/8 saved
winners unadapted (29-40 candidates kept) versus 8/8 adapted (6-21
kept). This module pins both numbers against the SAVED candidates and
the shared selector — the serialization boundary, not a heuristic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototypes.v30_video_apex.candidates import (
    CANDIDATE_SCHEMA_VERSION, CandidateSchemaError, accepted_path,
    export_candidates, normalize_candidate, normalize_rows,
    path_consistency, polyline_length, read_candidates)
from prototypes.v30_video_apex.selection import (
    SelectionPolicy, select_candidate)

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/prototypes/v30/movie_v18_evgate"


def _saved_rows() -> list[dict]:
    return json.loads((RUN / "candidates.json").read_text())


def _events(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("owner_id", "")), []).append(r)
    return out


@pytest.mark.skipif(not (RUN / "candidates.json").exists(),
                    reason="movie_v18_evgate candidates absent")
def test_legacy_export_is_legacy_and_normalization_fixes_the_replay():
    rows = _saved_rows()
    assert rows, "no saved candidates"
    # the export really is pre-v2 (this is the defect, documented)
    assert all(r.get("q_max") is None for r in rows)
    assert all(r.get("ev") is None for r in rows)
    assert all(r.get("tip_score") is not None for r in rows)
    assert all((r.get("evidence") or {}).get("wall_ev") is not None
               for r in rows)

    pol = SelectionPolicy(rank="sel", evidence_gate=0.6, evidence_key="ev")
    raw_matches = adapted_matches = 0
    raw_kept: list[int] = []
    adapted_kept: list[int] = []
    events = _events(rows)
    assert len(events) == 8, f"expected the gated eight, got {len(events)}"
    for event, group in sorted(events.items()):
        saved = [r for r in group if r.get("selected")]
        assert len(saved) == 1, f"{event}: {len(saved)} saved winners"
        saved_id = str(saved[0].get("route_id"))
        raw = select_candidate(group, pol)
        raw_kept.append(raw.n_kept)
        if raw.winner is not None and \
                str(raw.winner.get("route_id")) == saved_id:
            raw_matches += 1
        adapted = select_candidate(normalize_rows(group), pol)
        adapted_kept.append(adapted.n_kept)
        if adapted.winner is not None and \
                str(adapted.winner.get("route_id")) == saved_id:
            adapted_matches += 1
    # the audit's numbers, reproduced from the saved file
    assert raw_matches == 5, f"raw replay matched {raw_matches}/8"
    assert adapted_matches == 8, f"adapted replay matched {adapted_matches}/8"
    assert max(raw_kept) > max(adapted_kept), (
        f"raw kept {raw_kept} vs adapted {adapted_kept}")


def test_normalization_is_idempotent_and_maps_once():
    row = {"candidate_id": "c1", "movie_id": "ld", "owner_id": "o1",
           "source_frame": 10, "tip_score": 0.75, "route_p": 0.5,
           "evidence": {"wall_ev": 12.0, "seed": "walk"}}
    once = normalize_candidate(row)
    twice = normalize_candidate(once)
    assert once == twice
    assert once["q_max"] == 0.75 and once["ev"] == 12.0
    assert once["sel"] == pytest.approx(0.375)
    assert once["candidate_schema"] == CANDIDATE_SCHEMA_VERSION
    # legacy keys survive as provenance (nothing rewritten)
    assert once["tip_score"] == 0.75
    assert once["evidence"]["wall_ev"] == 12.0
    # the selector now sees real scores where it saw none
    assert once["sel"] == pytest.approx(once["q_max"] * once["route_p"])


def test_flat_evidence_is_accepted_too():
    row = {"candidate_id": "c1", "movie_id": "ld", "owner_id": "o1",
           "source_frame": 1, "q_max": 0.2, "route_p": 1.0, "ev": 3.0}
    out = normalize_candidate(row)
    assert out["q_max"] == 0.2 and out["ev"] == 3.0 and out["sel"] == 0.2


@pytest.mark.parametrize("bad,reason", [
    ({}, "missing required key"),
    ({"candidate_id": "c", "movie_id": "m", "owner_id": "o",
      "source_frame": 0}, "no score"),
    ({"candidate_id": "c", "movie_id": "m", "owner_id": "o",
      "source_frame": 0, "tip_score": 0.1}, "no evidence"),
    ({"candidate_id": "c", "movie_id": "m", "owner_id": "o",
      "source_frame": 0, "tip_score": float("nan"), "ev": 1.0},
     "not finite"),
])
def test_missing_score_or_evidence_is_rejected(bad, reason):
    with pytest.raises(CandidateSchemaError) as e:
        normalize_candidate(bad)
    assert reason in str(e.value)


def test_export_and_read_use_the_same_object(tmp_path):
    rows = [_saved_rows()[0]]
    exported = export_candidates(rows)
    p = tmp_path / "candidates.json"
    p.write_text(json.dumps(exported, indent=2))
    back = read_candidates(p)
    assert back == exported
    # and a legacy file reads back normalized
    p2 = tmp_path / "legacy.json"
    p2.write_text(json.dumps(rows, indent=2))
    legacy_back = read_candidates(p2)
    assert legacy_back[0]["q_max"] == rows[0]["tip_score"]
    assert legacy_back[0]["ev"] == rows[0]["evidence"]["wall_ev"]


def test_accepted_path_is_a_prefix_and_length_agrees():
    row = {"candidate_id": "c", "movie_id": "m", "owner_id": "o",
           "source_frame": 0, "tip_score": 0.5, "ev": 1.0,
           "polyline_native": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0],
                               [3.0, 4.0], [13.0, 4.0]],
           "front_s": 7.0, "root_source": "path-start"}
    ap = accepted_path(row)
    # truncated inside the second segment at exactly 7 px of arclength
    assert ap["current_path"][0] == [0.0, 0.0]
    assert ap["current_path"][-1] == [3.0, 4.0] or \
        ap["current_path"][-1] == [3.0, 4.0]
    assert ap["length_px"] == pytest.approx(7.0, abs=1e-6)
    assert ap["current_tip"] == ap["current_path"][-1]
    assert ap["support_length_px"] == pytest.approx(17.0)
    ok, problems = path_consistency(row)
    assert ok, problems
    # a declared length that does not match the accepted path is caught
    bad = dict(row, length_px=17.0)
    ok2, problems2 = path_consistency(bad)
    assert not ok2 and "declared length" in problems2[0]
    # and a front beyond the support is caught
    ok3, problems3 = path_consistency(dict(row, front_s=99.0))
    assert not ok3 and "outside the support" in problems3[0]


def test_polyline_length_basic():
    assert polyline_length([[0, 0]]) == 0.0
    assert polyline_length([[0, 0], [3, 4]]) == pytest.approx(5.0)
    assert polyline_length([[0, 0], [3, 4], [3, 4]]) == pytest.approx(5.0)
