"""rev8: a ball-scoped "no tube here" verdict.

Some grains genuinely grow no visible tube. That verdict must be
recordable (otherwise the painter is stuck), and it must stay scoped
to the queried grain — H242: a scoped "can't tell" is never a
full-frame negative.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_app import AnnotatorController  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402

MOVIE_LD = ("/Users/joshjiang/Downloads/"
            "test1lowdensjoshua-28c-hz.mp4 .mp4")


def _task():
    return {"uuid": "p-000", "owner_uuid": "rev8p-42000-g0",
            "owner_key": "ld|rev8p-42000-g0",
            "tube_uuid": "rev8p-42000-g0",
            "query_frames": [42000], "task_type": "body_mask",
            "movie": "ld", "completed": False,
            "focus_xy": [624.0, 664.0],
            "target_xy": [557.0, 691.0], "target_r": 14.0}


def test_no_tube_verdict_is_scoped_and_recorded(tmp_path):
    proj = tmp_path / "pp"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.load_tasks([_task()])
        c.current = c.tasks[0]
        uuid = c.save_mask_none()
        assert uuid == "mask-p-000"
    finally:
        reader.close()
        store.close()
    import sqlite3
    con = sqlite3.connect(str(proj / "annotations.db"))
    try:
        rows = dict(con.execute(
            "SELECT uuid, data FROM entities WHERE kind='mask'").fetchall())
        trow = dict(con.execute(
            "SELECT uuid, data FROM entities WHERE kind='task'").fetchall())
    finally:
        con.close()
    md = json.loads(rows["mask-p-000"])
    assert md["no_tube"] is True
    assert md["scope"] == "ball"
    assert md["painted_xy"] == []
    assert md["target_xy"] == [557.0, 691.0]
    assert md["owner_uuid"] == "rev8p-42000-g0"
    assert json.loads(trow["p-000"])["no_tube"] is True


def test_snapshot_turns_the_verdict_into_owned_absence_not_a_cap_negative(
        tmp_path):
    """rev9 WP-A.6: owned absence is NOT a generic cap negative."""
    proj = tmp_path / "pp2"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.load_tasks([_task()])
        c.current = c.tasks[0]
        c.save_mask_none()
    finally:
        reader.close()
        store.close()
    out = tmp_path / "snap"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_v30_snapshot.py"),
         "--project-dir", str(proj), "--default-movie", "ld",
         "--movie", f"ld={MOVIE_LD}", "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    regions = json.loads((out / "regions.json").read_text())
    negs = [g for g in regions if g.get("source") == "mask-no-tube"]
    assert len(negs) == 1, regions
    g = negs[0]
    assert g["kind"] == "owned_absence"
    assert g["class_scope"] == "owned_tube"    # the owner, nothing else
    assert g["owner_key"] == "ld|rev8p-42000-g0"
    assert g["grain_xy"] == [557.0, 691.0] and g["grain_r"] == 14.0
    xs = [p[0] for p in g["polygon_xy"]]
    ys = [p[1] for p in g["polygon_xy"]]
    assert xs == [557.0 - 14, 557.0 + 14, 557.0 + 14, 557.0 - 14]
    assert ys == [691.0 - 14, 691.0 - 14, 691.0 + 14, 691.0 + 14]
    # the verdict never becomes a body mask
    rows = json.loads((out / "body_masks.json").read_text())
    assert rows == []
    # and it IS an owned presence/visibility sample for that owner
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    samples = samples_from_snapshot(str(out))
    absences = [s_ for s_ in samples if s_.kind == "no_tube"]
    assert len(absences) == 1, [s_.kind for s_ in samples]
    a = absences[0]
    assert a.owner_key == "ld|rev8p-42000-g0"
    assert a.direct_state == "no_tube_visible"
    assert a.target_xy == (557.0, 691.0)
    assert not a.mask_raster and not a.mask_points


def test_foreign_tip_inside_an_owned_absence_disc_is_never_negative():
    """The reviewer's constructed counterexample.

    Grain A's "no tube" verdict covers A's own disc. A real (unlabelled)
    cap of grain B lying inside that disc is positive for the GENERIC
    terminal head, so the region must not suppress it — neither in the
    new `owned_absence` form nor in the historical cap-scoped
    `verified_negative` form that snapshots up to snap22 carry.
    """
    from prototypes.v30_video_apex.targets import apex_pos_neg_masks
    disc_poly = [[86.0, 86.0], [114.0, 86.0], [114.0, 114.0], [86.0, 114.0]]
    foreign_tip = (104.0, 98.0)          # grain B's real cap, inside A's disc
    new_form = {"kind": "owned_absence", "class_scope": "owned_tube",
                "source": "mask-no-tube", "polygon_xy": disc_poly,
                "grain_xy": [100.0, 100.0], "grain_r": 14.0}
    _pos, neg = apex_pos_neg_masks(288, 288, (0.0, 0.0), None, [new_form],
                                   other_tips_crop=[foreign_tip])
    assert neg.sum() == 0, "owned absence must not paint a negative"
    legacy_form = dict(new_form, kind="verified_negative",
                       class_scope="cap")
    _pos2, neg2 = apex_pos_neg_masks(288, 288, (0.0, 0.0), None,
                                     [legacy_form],
                                     other_tips_crop=[foreign_tip])
    assert neg2.sum() == 0, (
        "a mask-no-tube region must never suppress the generic head")
    # an EXPLICITLY reviewed cap-free region (hand-drawn) still does
    hand = {"kind": "verified_negative", "class_scope": "cap",
            "polygon_xy": [[150.0, 150.0], [170.0, 150.0],
                           [170.0, 170.0], [150.0, 170.0]]}
    _p3, neg3 = apex_pos_neg_masks(288, 288, (0.0, 0.0), None, [hand])
    assert neg3.sum() > 0, "explicit review must still suppress"
    # ...and a known tip inside it stays protected even then
    _p4, neg4 = apex_pos_neg_masks(288, 288, (0.0, 0.0), None, [hand],
                                   other_tips_crop=[(160.0, 160.0)])
    yy, xx = np.mgrid[0:288, 0:288]
    d_tip = np.hypot(xx - 160.0, yy - 160.0)
    assert not (neg4[d_tip <= 8.0] > 0).any(), (
        "the known tip's own pixels must never be negative")
    assert neg4.sum() > 0, "the rest of the reviewed region stays negative"
