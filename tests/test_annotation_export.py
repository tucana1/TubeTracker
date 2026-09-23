"""Legacy kit import: lossless, read-only, no fabricated coordinates."""

import hashlib
from pathlib import Path

from tubetracker.annotation_export import import_legacy_kit

KIT = Path("runs/reference_validation/lowdens_v27_6")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_legacy_import_counts_and_aliases():
    before = (_sha(KIT / "manifest.json"), _sha(KIT / "annotations.json"))
    out = import_legacy_kit(KIT / "manifest.json", KIT / "annotations.json")
    assert len(out["owners"]) == 42
    assert len(out["tasks"]) == 504
    assert out["observations"] == []  # all unreviewed: no coordinates
    assert sorted(o.pipeline_aliases[0] for o in out["owners"]) == sorted(
        o.pipeline_aliases[0] for o in out["owners"])
    assert {o.uuid for o in out["owners"]} == {
        t.owner_uuid for t in out["tasks"]}
    assert (_sha(KIT / "manifest.json"), _sha(KIT / "annotations.json")) == before


def test_imported_tasks_validate_clean():
    out = import_legacy_kit(KIT / "manifest.json", KIT / "annotations.json")
    bad = [e for t in out["tasks"] for e in t.validate()]
    assert bad == []
    assert all(not t.completed for t in out["tasks"])
