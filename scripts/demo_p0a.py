"""Behavioral demo transcript (P0A acceptance, headless part).

Walks the full user loop without a display: annotate -> edit ->
save -> undo -> close -> reopen -> resume -> complete -> reopen
completed (no starter regen). Run: .venv/bin/python scripts/demo_p0a.py
The display-only half (pixels, zoom, clicks) is verified live by the
user in the app; every state transition here is identical in both.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"

from tubetracker.annotation_app import AnnotatorController  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def log(*parts):
    print("[demo]", *parts, flush=True)


def main() -> None:
    proj = Path(tempfile.mkdtemp(prefix="p0a_demo_"))
    log("project:", proj)
    reader = FrameReader(MOVIE)

    def open_controller(actor="demo"):
        store = AnnotationStore(proj / "annotations.db")
        c = AnnotatorController(store, reader, viewer=None, actor=actor)
        c.add_reader("default", reader)
        c.tasks = store.unfinished_tasks(limit=500)
        return c, store

    # Fresh project: starter-equivalent tasks arrive from the builder.
    c, store = open_controller()
    assert store.count_tasks() == 0
    c.load_tasks([
        {"uuid": "d-apex", "owner_uuid": "o1", "query_frames": [15000],
         "focus_xy": [1017.0, 344.0], "task_type": "apex",
         "stratum": "demo", "completed": False},
        {"uuid": "d-path", "owner_uuid": "o1", "query_frames": [15100],
         "focus_xy": [1017.0, 344.0], "task_type": "centerline",
         "stratum": "demo", "completed": False},
        {"uuid": "d-x", "owner_uuid": "o1", "query_frames": [15200],
         "focus_xy": [1017.0, 344.0], "task_type": "crossing",
         "stratum": "demo", "completed": False},
    ])
    log("3 demo tasks loaded")

    # 1. Annotate an apex, then EDIT it (re-click overwrites).
    assert c.advance()["uuid"] == "d-apex"
    c.click(1010.0, 330.0)
    assert c.progress() == (1, 3)
    c.go_back()
    c.click(1012.0, 331.0)
    assert store.load("obs-d-apex")["data"]["direct_xy"] == [1012.0, 331.0]
    assert store.load("obs-d-apex")["revision"] == 2
    log("apex annotated + edited (revision 2, no duplicates)")

    # 2. Undo removes the mark and reopens the task.
    c.go_back()
    assert c.undo_mark() is True
    assert c.progress() == (0, 3)
    log("undo reopens the task, history retained")

    # 3. Re-mark, look at a neighbor frame, save typed state w/ audit.
    c.click(1012.0, 331.0)
    c.go_back()
    assert c.step_frame(3) == 3
    assert c.click(1.0, 1.0) is None
    c.back_to_task_frame()
    c.save_state("owner_uncertain")
    c.save_and_next()
    o = store.load("obs-d-apex")["data"]
    assert o["direct_state"] == "owner_uncertain"
    assert o["context_frames"] == [15003]
    log("typed state + consulted-frame audit recorded")

    # 4. Centerline path, then crossing lanes.
    assert c.current["uuid"] == "d-path"
    for px, py in [(900.0, 300.0), (950.0, 320.0), (1000.0, 330.0)]:
        c.click(px, py)
    c.finish_path_and_next()
    assert store.load("obs-d-path")["data"]["path_xy"] == [
        [900.0, 300.0], [950.0, 320.0], [1000.0, 330.0]]
    log("centerline path saved (3 vertices, complete)")
    assert c.current["uuid"] == "d-x"
    c.click(10.0, 10.0)
    c.select_lane("B")
    c.click(12.0, 40.0)
    c.click(30.0, 12.0)
    c.select_lane("A")
    c.click(32.0, 30.0)
    xid = c.save_crossing(unresolved=True)
    assert store.load(xid)["data"]["unresolved"] is True
    c.save_and_next()
    assert c.progress() == (3, 3)
    log("crossing saved unresolved, queue complete")

    # 5. Close + reopen: resume state, no starter regen.
    store.close()
    c2, store2 = open_controller(actor="demo2")
    assert store2.count_tasks() == 3  # completed project: no starters
    assert c2.tasks == [] and c2.advance() is None
    log("reopen of completed project: complete, zero regenerated tasks")

    store2.close()
    reader.close()
    shutil.rmtree(proj, ignore_errors=True)
    log("ALL P0A HEADLESS ACCEPTANCE STEPS PASS")


if __name__ == "__main__":
    main()
