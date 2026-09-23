"""v2 annotation schema validation (P0A, Qt-free)."""

from tubetracker.annotation_schema import (
    Crossing,
    Observation,
    Owner,
    SnapshotLock,
    SupervisionRegion,
    new_uuid,
)


def test_hidden_apex_rejects_exact_point():
    o = Observation(uuid=new_uuid(), owner_uuid="x", source_frame=1,
                    direct_state="not_directly_visible", direct_xy=(5.0, 5.0))
    assert any("exact point" in e for e in o.validate())


def test_visible_apex_needs_geometry():
    o = Observation(uuid=new_uuid(), owner_uuid="x", source_frame=1,
                    direct_state="direct_visible")
    assert any("point or region" in e for e in o.validate())
    o.direct_xy = (5.0, 5.0)
    assert o.validate() == []


def test_region_and_crossing_minima():
    r = SupervisionRegion(uuid=new_uuid(), movie_uuid="m", source_frame=1,
                          polygon_xy=[(0.0, 0.0)], kind="verified_negative",
                          confirmed=True)
    assert any("vertices" in e for e in r.validate())
    c = Crossing(uuid=new_uuid(), movie_uuid="m", owner_uuids=["a"])
    assert any(">=2 traced lanes" in e for e in c.validate())


def test_finalized_snapshot_needs_hash():
    s = SnapshotLock(uuid=new_uuid(), schema_version="v2.0", finalized=True)
    assert any("content hash" in e for e in s.validate())
    # H260: v2.0 locks do not silently pass under v2.1; they migrate
    # explicitly (un-finalized) or fail validation.
    assert any("mismatch" in e for e in s.validate())
    m = s.migrate_to_current()
    assert m.schema_version == "v2.1" and m.finalized is False
    m.content_hash = "abc"
    m.finalized = True
    assert m.validate() == []
    s2 = SnapshotLock(uuid=new_uuid(), schema_version="v2.1", finalized=True)
    s2.content_hash = "abc"
    assert s2.validate() == []


def test_owner_and_schema_import_clean():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
        "import sys; from tubetracker.annotation_schema import Owner; "
        "assert 'napari' not in sys.modules; assert 'PyQt6' not in sys.modules"], check=True)
    assert Owner(uuid=new_uuid(), movie_uuid="m").validate() == []
