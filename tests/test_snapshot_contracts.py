"""rev8 step-1 contracts: snapshot diffs and verified backups.

These tools exist because the rev7 corpus loss was only discoverable by
hand: snapshots must be diffable at entity level, and project backups
must prove restore before new labeling starts.
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

DIFF = REPO / "scripts" / "snapshot_diff.py"
BACKUP = REPO / "scripts" / "backup_project_dbs.py"


def _write_snap(d: Path, obs: list, regions: list, duels: list,
                census: list, masks: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "observations.json").write_text(json.dumps(obs))
    (d / "regions.json").write_text(json.dumps(regions))
    (d / "duels.json").write_text(json.dumps(duels))
    (d / "census.json").write_text(json.dumps(census))
    (d / "body_masks.json").write_text(json.dumps(masks))


def test_snapshot_diff_detects_loss(tmp_path):
    obs = [{"obs_uuid": "o1", "task_uuid": "t1", "movie": "ld",
            "source_frame": 100, "direct_state": "direct_visible",
            "direct_xy": [10.0, 10.0], "path_complete": True,
            "path_xy": [[8.0, 8.0], [10.0, 10.0]]},
           {"obs_uuid": "o2", "task_uuid": "t2", "movie": "ld",
            "source_frame": 200, "direct_state": "no_tube_visible"}]
    regs = [{"_region_uuid": "r1", "task_uuid": "t3", "movie_uuid": "ld",
             "source_frame": 100, "kind": "verified_negative",
             "polygon_xy": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]}]
    duels = [{"_duel_uuid": "d1", "task_uuid": "t4", "winner": "A"},
             {"_duel_uuid": "d2", "task_uuid": "t5", "winner": "neither"}]
    census = [{"task_uuid": "t6", "complete": True}]
    masks = [{"mask_uuid": "m1", "source_obs_uuid": "o1",
              "painted_xy": [[1.0, 1.0]]}]
    old = tmp_path / "old"
    _write_snap(old, obs, regs, duels, census, masks)

    # successor: one observation's tip moved, o2 gone, one duel gone,
    # census and region gone (the snap13 -> snap15 loss pattern).
    new = tmp_path / "new"
    _write_snap(new, [dict(obs[0], direct_xy=[12.0, 10.0])], [], [duels[0]],
                [], [])

    out = tmp_path / "diff.json"
    p = subprocess.run([sys.executable, str(DIFF), str(old), str(new),
                        "--json", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    rep = json.loads(out.read_text())
    ent = rep["entities"]
    assert ent["observations"]["n_old"] == 2 and ent["observations"]["n_new"] == 1
    assert any("o2" in k for k in ent["observations"]["removed"])
    assert ent["observations"]["changed"], "moved tip must be a change"
    assert "direct_xy" in ent["observations"]["changed"][0]["fields"]
    assert len(ent["regions"]["removed"]) == 1
    assert len(ent["census"]["removed"]) == 1
    assert len(ent["duels"]["removed"]) == 1
    assert len(ent["body_masks"]["removed"]) == 1
    assert rep["summary"]["n_path_samples"] == 1

    # strict mode refuses a lossy successor
    p2 = subprocess.run([sys.executable, str(DIFF), str(old), str(new),
                         "--fail-on-removal"], capture_output=True, text=True)
    assert p2.returncode == 2, p2.stdout[-800:]
    # an identical diff is clean
    p3 = subprocess.run([sys.executable, str(DIFF), str(old), str(old),
                         "--fail-on-removal"], capture_output=True, text=True)
    assert p3.returncode == 0


def test_backup_verifies_and_detects_drift(tmp_path):
    proj = tmp_path / "live_project"
    proj.mkdir()
    s = AnnotationStore(proj / "annotations.db")
    try:
        s.save("task", "t1", {"uuid": "t1", "completed": True}, actor="t")
        s.save("observation", "o1", {"task_uuid": "t1",
                                     "direct_state": "direct_visible"},
               actor="t")
        s.save("region", "r1", {"task_uuid": "t1", "kind":
                                "verified_negative"}, actor="t")
    finally:
        s.close()
    bdir = tmp_path / "backups"
    p = subprocess.run(
        [sys.executable, str(BACKUP), "--project-dir", str(proj),
         "--backup-dir", str(bdir)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    man = json.loads((bdir / "backup_manifest.json").read_text())
    rec = man["records"][0]
    assert rec["status"] == "OK", rec
    assert rec["integrity_check"] == "ok"
    assert rec["restored_counts"] == rec["live_counts"]
    assert rec["restored_digest"] == rec["live_digest"]
    assert rec["live_counts"]["task"] == 1
    assert rec["live_counts"]["_revisions"] == 3

    # label more, back up again: recorded as drift, never silent
    s = AnnotationStore(proj / "annotations.db")
    try:
        s.save("task", "t2", {"uuid": "t2", "completed": False}, actor="t")
    finally:
        s.close()
    p2 = subprocess.run(
        [sys.executable, str(BACKUP), "--project-dir", str(proj),
         "--backup-dir", str(bdir)], capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr[-1500:]
    man2 = json.loads((bdir / "backup_manifest.json").read_text())
    rec2 = man2["records"][0]
    assert rec2["drift_vs_previous"] == "changed", rec2
    assert rec2["live_counts"]["task"] == 2

    # an unreadable project is reported, not crashed through
    p3 = subprocess.run(
        [sys.executable, str(BACKUP), "--project-dir",
         str(tmp_path / "does-not-exist"), "--backup-dir", str(bdir)],
        capture_output=True, text=True)
    assert p3.returncode == 1
    man3 = json.loads((bdir / "backup_manifest.json").read_text())
    assert man3["n_failed"] == 1


def test_backup_copy_is_restorable_standalone(tmp_path):
    """The backup file itself must open and count without the source."""
    proj = tmp_path / "p2"
    proj.mkdir()
    s = AnnotationStore(proj / "annotations.db")
    try:
        for i in range(3):
            s.save("task", f"k{i}", {"uuid": f"k{i}"}, actor="t")
    finally:
        s.close()
    bdir = tmp_path / "b2"
    subprocess.run([sys.executable, str(BACKUP), "--project-dir", str(proj),
                    "--backup-dir", str(bdir)], capture_output=True, text=True,
                   check=True)
    copy = next(bdir.glob("*p2.db"))
    con = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        integ = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    assert n == 3 and integ == "ok"
