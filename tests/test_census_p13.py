"""rev12 P1.3: census, emergence bounds, germination report tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.census import (  # noqa: E402
    CensusScope, CensusTile, GrainRecord, build_census, emergence_intervals,
    germination_report)


def test_census_counts_and_clumps():
    grains = [
        GrainRecord("g1", "ld", (0, 0), present_observed_at=[100]),
        GrainRecord("g2", "ld", (10, 0), no_tube_observed_at=[50]),
        GrainRecord("g3", "ld", (20, 0), clump_id="clumpA"),
        GrainRecord("g4", "ld", (21, 0), clump_id="clumpA"),
    ]
    census = build_census(grains, [])
    assert census["n_grains"] == 4
    assert census["classes"]["present_evidence"] == 1
    assert census["classes"]["no_visible_tube_evidence_only"] == 1
    assert census["classes"]["unresolved"] == 2
    assert census["clumps_unresolved"]["clumpA"] == ["g3", "g4"]
    assert census["census_completeness_certified"] is False


def test_exhaustive_flag_requires_certified_tile():
    scope = CensusScope("ld", (0, 0, 64, 64), (6,))
    protocol = dict(border_policy=scope.border_policy, clump_policy=scope.clump_policy)
    t = CensusTile("t1", "ld", 5, (0, 0, 64, 64), exhaustive=False,
                   class_scopes=("grains",), **protocol)
    census = build_census([], [t], scope=scope)
    assert census["n_exhaustive_tiles"] == 0
    t2 = CensusTile("t2", "ld", 6, (0, 0, 64, 64), exhaustive=True,
                    class_scopes=("grains",), **protocol)
    census2 = build_census([], [t, t2], scope=scope)
    assert census2["n_exhaustive_tiles"] == 1
    assert census2["census_completeness_certified"] is True
    # an exhaustive tile WITHOUT a declared scope cannot certify
    t3 = CensusTile("t3", "ld", 7, (0, 0, 64, 64), exhaustive=True)
    census3 = build_census([], [t3], scope=scope)
    assert census3["n_exhaustive_tiles"] == 0
    assert census3["n_tiles_out_of_scope"] == 1


def test_emergence_bounds_and_censoring():
    g = GrainRecord("g1", "ld", (0, 0), no_tube_observed_at=[100, 200],
                    present_observed_at=[300, 400])
    iv = emergence_intervals([g])[0]
    assert iv.bounds() == (200, 300)
    g2 = GrainRecord("g2", "ld", (0, 0), present_observed_at=[300])
    iv2 = emergence_intervals([g2])[0]
    assert iv2.bounds() is None
    assert "left" in iv2.censoring
    g3 = GrainRecord("g3", "ld", (0, 0), no_tube_observed_at=[100])
    iv3 = emergence_intervals([g3])[0]
    assert iv3.bounds() is None
    assert "right" in iv3.censoring


def test_germination_report_units_and_rule():
    g1 = GrainRecord("g1", "ld", (0, 0), no_tube_observed_at=[100],
                     present_observed_at=[160])
    g2 = GrainRecord("g2", "ld", (0, 0), no_tube_observed_at=[100])
    rep = germination_report(emergence_intervals([g1, g2]))
    assert rep["numerator_confirmed"] == 1
    assert rep["denominator_grains"] == 2
    assert rep["unresolved"] == 1
    assert rep["germination_rule"] is None      # never invented
    assert rep["timing_bounds"][0]["interval_seconds"] is None
    assert "NULL" in rep["units"]["seconds"]
    # with provenance-carrying cadence, seconds appear
    rep2 = germination_report(emergence_intervals([g1, g2]),
                              acquisition_cadence_s=3.0,
                              cadence_source="test-cadence")
    assert rep2["timing_bounds"][0]["interval_seconds"] == 180.0
    with pytest.raises(ValueError):
        germination_report(emergence_intervals([g1]), 
                           acquisition_cadence_s=3.0)  # no source


def test_pipeline_backend_registry():
    from tubetracker.pipeline_backends import (
        available_backends, load_backend)
    assert "v29-legacy" in available_backends()
    fn, info = load_backend("v29-legacy")
    assert fn is None and info["backend"] == "v29-legacy"
    with pytest.raises(ValueError):
        load_backend("nope")
    # v30-strict with a missing checkpoint reports unavailability
    fn2, info2 = load_backend("v30-strict",
                              checkpoint="/nonexistent.pt")
    assert fn2 is None and info2["available"] is False
