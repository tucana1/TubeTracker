import copy
import sqlite3

from tubetracker.grain_registry import (
    DiscoveryRegistry, apply_reviewed_links, register_alias_record)


def test_explicit_merge_overrides_stale_task_alias_and_is_idempotent():
    record = {"movie": "ld", "grain_native": [20, 30], "first_seen_frame": 10,
              "linkage": "owner-linked", "provenance": {"project": "p", "task": "t",
                                                       "label": "A", "kind": "owner-task"}}
    registry = {"grains": {"old": copy.deepcopy(record)}, "aliases": {
        "task|p|t|A": "old", "emerge|p|10|A": "old", "region|r": "old"}}
    links = [{"canonical": "canonical", "movie": "ld", "basis": "reviewed same grain", "revision": 1,
              "observations": {"project": "p", "label": "A", "frames": [10]}}]
    explicit = apply_reviewed_links(registry, links)
    assert register_alias_record(registry, record, ["task|p|t|A", "emerge|p|10|A"], explicit) == "canonical"
    assert set(registry["grains"]) == {"canonical"}
    assert set(registry["aliases"].values()) == {"canonical"}
    assert registry["tombstones"]["old"]["record"]["grain_native"] == [20, 30]
    first = copy.deepcopy(registry)
    explicit = apply_reviewed_links(registry, links)
    register_alias_record(registry, record, ["task|p|t|A", "emerge|p|10|A"], explicit)
    assert registry == first


def test_same_frame_and_letter_in_different_tasks_is_not_merge_evidence():
    def grain(x, task):
        return {"movie": "m2", "grain_native": [x, 10], "first_seen_frame": 1,
                "provenance": {"project": "p", "task": task, "label": "A", "kind": "owner-task"}}
    registry = {"grains": {"one": grain(10, "t1"), "two": grain(50, "t2")},
                "aliases": {"task|p|t1|A": "one", "task|p|t2|A": "two", "emerge|p|1|A": "one"}}
    got = register_alias_record(registry, grain(50, "t2"),
                                ["task|p|t2|A", "emerge|p|1|A"])
    assert got == "two" and len(registry["grains"]) == 2
    assert not registry.get("tombstones")


def test_discovery_ids_survive_restart_translation_and_cache_removal(tmp_path):
    path = tmp_path / "grain_registry.sqlite"
    registry = DiscoveryRegistry(path)
    detections = [{"xy": [20., 20.], "radius": 12, "score": .9},
                  {"xy": [40., 20.], "radius": 12, "score": .8}]
    owners, first = registry.reconcile("ld", "hash-a", 0, [], detections)
    ids = [o["id"] for o in owners]
    shifted = [{**d, "xy": [d["xy"][0]+2, d["xy"][1]+1]} for d in reversed(detections)]
    again, report = DiscoveryRegistry(path).reconcile("ld", "hash-a", 30, [], shifted)
    assert [o["id"] for o in again] == ids and not report["new_ids"]
    assert len(report["matched_existing_ids"]) == 2
    other, _ = registry.reconcile("ld", "different-movie-bytes", 0, [], detections)
    assert {o["id"] for o in other}.isdisjoint(ids)
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM grains").fetchone()[0] == 4
    con.close()


def test_clumped_seed_grains_are_never_merged_and_ambiguous_matches_are_reported(tmp_path):
    seeds = [{"id": "a", "movie": "ld", "grain_native": [10, 10]},
             {"id": "b", "movie": "ld", "grain_native": [20, 10]}]
    registry = DiscoveryRegistry(tmp_path / "registry.sqlite")
    owners, report = registry.reconcile("ld", "h", 0, seeds,
        [{"xy": [15, 10], "radius": 12, "score": .9}])
    assert {o["id"] for o in owners} == {"a", "b"}
    assert report["ambiguous"][0]["candidate_ids"] == ["a", "b"]
    assert not report["new_ids"]
