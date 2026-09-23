"""rev13 W1 acceptance: the manifest comparator is recursive and
schema-aware.

The audit's counterexample: a changed NESTED snapshot content hash was
invisible (unknown top-level dicts were compared shallowly). Also:
- an input checkpoint / owner mapping / update-count change must
  invalidate a claimed seed-only comparison;
- identical runs report `identical`;
- only the exact declared path (or descendants) absorbs a difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _base_manifest() -> dict:
    return {
        "seed": 0,
        "updates": 120,
        "init": "runs/x/best_front_ep15.pt",
        "init_param_hash": "7c9941ea10197a00",
        "snapshot": {
            "content_sha256": {
                "obs-r4-p02": "aa11",
                "ld|track-1": "bb22",
            },
            "split_ids": {"train": ["a", "b"], "dev": ["c"]},
            "owner_versions": {"obs-r4-p02": 3},
        },
        "effective_config": {"base": 8, "multiscale": True,
                             "losses": {"visibility": 0.5}},
        "history": [{"epoch": 0}],
        "note": "run note text",
    }


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_nested_snapshot_hash_change_detected(tmp_path):
    """The audit counterexample: only a nested hash changed."""
    from scripts.compare_run_manifests import compare

    a = _base_manifest()
    b = _base_manifest()
    b["snapshot"]["content_sha256"]["obs-r4-p02"] = "CHANGED"
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    res = compare(pa, pb, "seed")
    assert res["n_diffs"] >= 1
    assert res["verdict"] == "NOT-single-variable"
    fields = [d["field"] for d in res["undeclared_diffs"]]
    assert "snapshot.content_sha256.obs-r4-p02" in fields


def test_input_checkpoint_change_invalidates_seed_claim(tmp_path):
    from scripts.compare_run_manifests import compare

    a = _base_manifest()
    b = _base_manifest()
    b["init"] = "runs/y/other.pt"
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    res = compare(pa, pb, "seed")
    assert res["verdict"] == "NOT-single-variable"
    assert any(d["field"] == "init" for d in res["undeclared_diffs"])


def test_owner_mapping_and_update_count_changes_detected(tmp_path):
    from scripts.compare_run_manifests import compare

    a = _base_manifest()
    b = _base_manifest()
    b["snapshot"]["owner_versions"]["obs-r4-p02"] = 4
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    assert compare(pa, pb, "seed")["verdict"] == "NOT-single-variable"
    c = _base_manifest()
    c["updates"] = 580
    pc = _write(tmp_path, "c.json", c)
    assert compare(pa, pc, "seed")["verdict"] == "NOT-single-variable"


def test_identical_runs_report_identical(tmp_path):
    from scripts.compare_run_manifests import compare

    pa = _write(tmp_path, "a.json", _base_manifest())
    pb = _write(tmp_path, "b.json", _base_manifest())
    res = compare(pa, pb, "seed")
    assert res["n_diffs"] == 0 and res["verdict"] == "identical"
    # prose notes and result structures never enter the comparison
    b = _base_manifest()
    b["note"] = "totally different prose"
    b["history"] = [{"epoch": 0}, {"epoch": 1}]
    pc = _write(tmp_path, "c.json", b)
    assert compare(pa, pc, "seed")["verdict"] == "identical"


def test_declared_variable_exact_paths_only(tmp_path):
    from scripts.compare_run_manifests import compare

    a = _base_manifest()
    b = _base_manifest()
    b["effective_config"]["losses"]["visibility"] = 0.25
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    ok = compare(pa, pb, "effective_config.losses.visibility")
    assert ok["verdict"] == "single-variable"
    # a substring match is NOT enough: 'losses' must not be declared
    # by naming 'visibility' elsewhere
    c = _base_manifest()
    c["effective_config"]["losses"]["visibility"] = 0.25
    pc = _write(tmp_path, "c.json", c)
    assert compare(pa, pc, "effective_config.base")[
        "verdict"] == "NOT-single-variable"
