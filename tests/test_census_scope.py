import pytest

from prototypes.v30_video_apex.census import (
    CensusScope, CensusTile, GrainRecord, build_census, emergence_intervals, germination_report)


def tile(name, box, *, frame=10, classes=("grains",)):
    return CensusTile(name, "ld", frame, box, classes, True, 1,
                      "centre-inside", "individual-physical-grains")


def test_exact_union_coverage_is_per_frame_and_class():
    scope = CensusScope("ld", (0, 0, 64, 64), (10, 20), ("grains", "tips"))
    tiles = [tile("left", (0, 0, 32, 64)), tile("right", (32, 0, 32, 64))]
    first = build_census([], tiles, scope=scope)
    assert not first["census_completeness_certified"]
    assert next(c for c in first["coverage"] if c["frame"] == 10 and c["class"] == "grains")["complete"]
    tiles += [tile("tips", (0, 0, 64, 64), classes=("tips",)),
              tile("later", (0, 0, 64, 64), frame=20, classes=("tips", "grains"))]
    assert build_census([], tiles, scope=scope)["census_completeness_certified"]
    overlap = [tile("a", (0, 0, 40, 64)), tile("b", (0, 0, 40, 64))]
    c = build_census([], overlap, scope=CensusScope("ld", (0, 0, 64, 64), (10,)))
    assert c["coverage"][0]["covered_area_px2"] == 40*64
    assert not c["census_completeness_certified"]


def test_unrelated_tile_duplicates_movie_time_and_border_do_not_change_denominator():
    scope = CensusScope("ld", (0, 0, 64, 64), (10,))
    grains = [GrainRecord("a", "ld", (10, 10), observed_frames=[10]),
              GrainRecord("a", "ld", (10, 10), observed_frames=[10], present_observed_at=[30]),
              GrainRecord("b", "m2", (10, 10), observed_frames=[10]),
              GrainRecord("border", "ld", (64, 10), observed_frames=[10]),
              GrainRecord("earlier", "ld", (20, 20), observed_frames=[9])]
    c = build_census(grains, [tile("remote", (1000, 1000, 64, 64))], scope=scope)
    assert c["grain_ids"] == ["a"] and c["n_grains"] == 1
    assert c["classes"]["present_evidence"] == 1
    assert not c["census_completeness_certified"]
    assert len(c["grains_out_of_scope"]) == 3
    assert not build_census(grains, [tile("ok", (0, 0, 64, 64))], scope_movie="ld")["census_completeness_certified"]


def test_unresolved_clump_and_unprovenanced_frame_prevent_certification():
    scope = CensusScope("ld", (0, 0, 64, 64), (10,))
    g = GrainRecord("a", "ld", (10, 10), clump_id="clump")
    tiles = [tile("full", (0, 0, 64, 64))]
    assert not build_census([g], tiles, scope=scope)["census_completeness_certified"]
    g.observed_frames = [10]
    assert not build_census([g], tiles, scope=scope)["census_completeness_certified"]
    g.membership_resolved = True
    assert build_census([g], tiles, scope=scope)["census_completeness_certified"]


def test_case_specific_rule_cannot_classify_other_present_or_absent_grains():
    a = GrainRecord("a", "ld", (0, 0), present_observed_at=[20],
        classification={"class": "germinated", "rule": "case A protocol", "source": "human A", "revision": 1})
    b = GrainRecord("b", "ld", (30, 0), present_observed_at=[20])
    c = GrainRecord("c", "ld", (60, 0), no_tube_observed_at=[10])
    report = germination_report(emergence_intervals([a, b, c]), germination_rule="case A protocol")
    assert report["numerator_germinated"] == 1
    assert report["n_with_present_evidence"] == 2
    assert report["n_biologically_unclassified"] == 2
    assert report["classification_records"][1]["rule"] is None
    for invalid in (-1, 0, float("nan")):
        with pytest.raises(ValueError):
            germination_report([], acquisition_cadence_s=invalid, cadence_source="log")


def test_canonical_merge_preserves_bound_lineage_in_either_order():
    a = GrainRecord("g", "ld", (10, 20), present_observed_at=[40], no_tube_observed_at=[10],
        provenance={"evidence": [{"frame": 40, "kind": "present", "source": "later-path", "revision": 2},
                                 {"frame": 10, "kind": "absent", "source": "first-check", "revision": 1}]})
    b = GrainRecord("g", "ld", (10, 20), present_observed_at=[30], no_tube_observed_at=[20],
        provenance={"evidence": [{"frame": 30, "kind": "present", "source": "earlier-root", "revision": 6},
                                 {"frame": 20, "kind": "absent", "source": "last-check", "revision": 4}]})
    for rows in ([a, b], [b, a], [a, b, b]):
        iv = emergence_intervals(rows)[0]
        assert iv.first_verified_present["frame"] == 30
        assert iv.first_verified_present["source"] == "earlier-root"
        assert iv.first_verified_present["revision"] == 6
        assert len(iv.first_verified_present["lineage"]) == 1
        assert iv.last_verified_absent["frame"] == 20
        assert iv.last_verified_absent["source"] == "last-check"
        assert iv.last_verified_absent["revision"] == 4
