"""rev8: batch-built mask tasks must survive the store round-trip, and
the target ring must sit on the GRAIN.

Two failures this pins, both real:

* The builder set the ring anchor to the tube's PATH CENTROID, which
  put the magenta ring 7px off on p03 and **56px off** on v30t-004 —
  the annotator spotted it immediately ("the circle isn't actually on
  the pollen").
* `AnnotationStore.load()` returns a wrapper
  (`{kind, data, revision}`); writing that wrapper straight back with
  `save()` put the whole record one level down and the app died with
  `KeyError: 'query_frames'`. Every edit must go through
  `load(uuid)["data"]`.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def _snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "observations.json").write_text(json.dumps([{
        "obs_uuid": "obs-t-001", "obs_revision": 1, "task_uuid": "t0",
        "project": "p", "movie": "ld", "movie_path": "/m.mp4",
        "tube_uuid": "", "owner_uuid": "", "source_frame": 500,
        "direct_state": "direct_visible", "direct_xy": None,
        "direct_region": None, "context_xy": None, "context_frames": [],
        # a 60px path running right, so the centroid is well off the grain
        "path_xy": [[100.0, 200.0], [130.0, 200.0], [160.0, 200.0]],
        "path_visible": [], "path_complete": True,
        "focus_xy": [100.0, 200.0],       # the grain: at the path START
        "task_type": "trace", "task_showed_tip": True,
        "owner_decision": "", "annotator": "t"}]))
    (snap / "regions.json").write_text("[]")
    (snap / "snapshot_manifest.json").write_text(
        '{"movies": {"ld": {"path": "/m.mp4"}}}')
    return snap


def test_builder_ring_anchors_on_the_grain_and_round_trips(tmp_path):
    snap = _snapshot(tmp_path)
    proj = tmp_path / "proj"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_v30mask_batch.py"),
         "--snapshot", str(snap), "--project-dir", str(proj),
         "--tag", "t", "--refs", "obs-t-001"],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-800:]
    con = sqlite3.connect(str(proj / "annotations.db"))
    try:
        (data,) = con.execute(
            "SELECT data FROM entities WHERE uuid='t-000'").fetchone()
    finally:
        con.close()
    t = json.loads(data)
    assert t["query_frames"] == [500]
    # the ring marks the grain, the view is framed on the tube
    assert t["target_xy"] == [100.0, 200.0]
    assert t["focus_xy"] == [130.0, 200.0]
    assert t["target_xy"] != t["focus_xy"]

    # load()->edit->save() must keep the record flat
    store = AnnotationStore(proj / "annotations.db")
    try:
        rec = store.load("t-000")
        assert set(rec) == {"kind", "data", "revision"}
        d = rec["data"]
        d["target_r"] = 12.0
        store.save("task", "t-000", d, actor="t")
        after = store.load("t-000")["data"]
        assert after["query_frames"] == [500], after.keys()
        assert after["target_xy"] == [100.0, 200.0]
        assert after["target_r"] == 12.0
        assert "data" not in after, "record must not be double-wrapped"
    finally:
        store.close()
