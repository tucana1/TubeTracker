"""rev13 W5 acceptance: germination vs emergence timing, quarantine of
contradictory bounds, scoped census completeness.

The audit's counterexamples:
- a known-present grain (present evidence, no certified absence) must
  count as germinated under the declared protocol while its interval
  stays left-censored — it is NOT "unresolved";
- a reversed pair (absent frame AFTER the present frame) must never
  become a confirmed interval with negative duration: it is
  quarantined with both bounds and their revisions retained;
- a tile from another movie (or a tiny unrelated region, or one
  without a declared class scope) cannot certify the census.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _grain(gid, *, present=None, absent=None, movie="ld", pos=(10.0, 20.0)):
    from prototypes.v30_video_apex.census import GrainRecord
    return GrainRecord(grain_id=gid, movie=movie, position_native=pos,
                       present_observed_at=list(present or []),
                       no_tube_observed_at=list(absent or []),
                       observed_frames=[1],
                       provenance={"revision": 3})


def test_known_present_without_absence_is_germinated_not_unresolved():
    from prototypes.v30_video_apex.census import (
        emergence_intervals, germination_report)
    grain = _grain("ld|g1", present=[22890, 24000])
    grain.classification = {"class": "germinated", "rule": "ruling-001",
                            "source": "review of this grain", "revision": 3}
    ivs = emergence_intervals([grain])
    assert ivs[0].classified == "germinated"
    assert "left" in ivs[0].censoring
    rep = germination_report(ivs, germination_rule="ruling-001: ...")
    assert rep["numerator_germinated"] == 1
    assert rep["numerator_bounded_intervals"] == 0
    assert rep["unresolved"] == 1  # timing is unresolved; CLASS is not
    assert rep["censored_left"] == 1


def test_contradictory_bounds_are_quarantined():
    from prototypes.v30_video_apex.census import (
        emergence_intervals, germination_report)
    # absent at 30000 AFTER present at 20000 -> contradiction
    ivs = emergence_intervals(
        [_grain("ld|g2", present=[20000], absent=[30000])])
    iv = ivs[0]
    assert iv.contradictory is True
    assert iv.bounds() is None            # never a confirmed interval
    rep = germination_report(ivs)
    assert rep["numerator_bounded_intervals"] == 0
    assert rep["numerator_confirmed"] == 0
    assert rep["n_quarantined_contradictory"] == 1
    q = rep["quarantined"][0]
    assert q["absent"]["frame"] == 30000 and q["present"]["frame"] == 20000
    assert "quarantined" in q["reason"]
    assert all("interval_frames" not in t for t in rep["timing_bounds"])


def test_valid_order_still_confirms():
    from prototypes.v30_video_apex.census import (
        emergence_intervals, germination_report)
    ivs = emergence_intervals(
        [_grain("ld|g3", present=[30000], absent=[20000])])
    rep = germination_report(ivs, acquisition_cadence_s=3.0,
                             cadence_source="acquisition log")
    assert rep["numerator_bounded_intervals"] == 1
    t = rep["timing_bounds"][0]
    assert t["interval_frames"] == 10000 and t["interval_seconds"] == 30000.0


def test_wrong_movie_or_tiny_tile_cannot_certify():
    from prototypes.v30_video_apex.census import (
        CensusScope, CensusTile, build_census)
    scope = CensusScope("ld", (0, 0, 400, 400), (1,))
    protocol = dict(border_policy=scope.border_policy, clump_policy=scope.clump_policy)
    tiles = [
        CensusTile(tile_id="t-other", movie="m2", frame=1,
                   box_xywh=(0, 0, 400, 400), class_scopes=("grains",),
                   exhaustive=True, **protocol),
        CensusTile(tile_id="t-tiny", movie="ld", frame=1,
                   box_xywh=(0, 0, 30, 30), class_scopes=("grains",),
                   exhaustive=True, **protocol),
        CensusTile(tile_id="t-noscope", movie="ld", frame=1,
                   box_xywh=(0, 0, 400, 400), exhaustive=True),
    ]
    c = build_census([_grain("ld|g1")], tiles, scope=scope)
    assert c["census_completeness_certified"] is False
    assert c["n_exhaustive_tiles"] == 1  # the small tile covers only 900 of 160000 px²
    assert c["n_tiles_out_of_scope"] == 2
    # a proper in-scope tile certifies
    ok = CensusTile(tile_id="t-ok", movie="ld", frame=1,
                    box_xywh=(0, 0, 400, 400), class_scopes=("grains",),
                    exhaustive=True, **protocol)
    c2 = build_census([_grain("ld|g1")], [ok], scope=scope)
    assert c2["census_completeness_certified"] is True
    assert c2["n_tiles_out_of_scope"] == 0
