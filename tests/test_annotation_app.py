"""Workbench controller logic headless (viewer=None, no Qt/display)."""

from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_frames import FrameReader
from tubetracker.annotation_store import AnnotationStore

MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"


def _tasks():
    return [
        {"uuid": "t1", "owner_uuid": "o1", "query_frames": [15000],
         "task_type": "apex", "completed": False},
        {"uuid": "t2", "owner_uuid": "o1", "query_frames": [15100],
         "task_type": "apex", "completed": False},
    ]


def test_click_advances_and_tally_counts(tmp_path):
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks(_tasks())
        assert c.advance()["uuid"] == "t1"
        assert c.progress() == (0, 2)
        c.click(1017.0, 331.0)  # one click saves + moves on
        assert c.progress() == (1, 2)
        assert c.current["uuid"] == "t2"
        obs = s.load("obs-t1")
        assert obs["data"]["direct_xy"] == [1017.0, 331.0]
    finally:
        r.close()
        s.close()


def test_back_revisits_and_remark_overwrites(tmp_path):
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks(_tasks())
        c.advance()
        c.click(100.0, 100.0)
        assert c.current["uuid"] == "t2"
        back = c.go_back()
        assert back["uuid"] == "t1"
        assert c.progress() == (0, 2)
        c.click(101.0, 102.0)  # re-mark overwrites, no duplicate
        assert s.load("obs-t1")["data"]["direct_xy"] == [101.0, 102.0]
        assert s.load("obs-t1")["revision"] == 2
        assert len(s.history("obs-t1")) == 2
    finally:
        r.close()
        s.close()


def test_centerline_path_flow(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "c1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "centerline",
                       "completed": False}])
        assert c.advance()["uuid"] == "c1"
        assert c.click(10.0, 10.0)["uuid"] == "c1"  # stays, extends path
        c.click(20.0, 20.0)
        c.click(30.0, 25.0)
        nxt = c.finish_path_and_next()
        assert nxt is None  # queue complete
        o = s.load("obs-c1")
        assert o["data"]["path_xy"] == [[10.0, 10.0], [20.0, 20.0],
                                        [30.0, 25.0]]
        assert o["data"]["path_complete"] is True
        assert o["data"]["direct_xy"] == [30.0, 25.0]
    finally:
        r.close()
        s.close()


def test_typed_states_and_context(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "t1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "apex", "completed": False},
            {"uuid": "t2", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "apex", "completed": False},
        ])
        c.advance()
        assert c.step_frame(5) == 5
        assert c.click(1.0, 1.0) is None  # refuse on context frame
        c.back_to_task_frame()
        c.save_state("no_tube_visible")
        c.save_and_next()
        o = s.load("obs-t1")
        assert o["data"]["direct_state"] == "no_tube_visible"
        assert "direct_xy" not in o["data"]  # never invented
        assert o["data"]["context_frames"] == [15005]
        assert c.current["uuid"] == "t2"
    finally:
        r.close()
        s.close()


def test_crossing_lanes_and_continuation(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "x1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "crossing",
                       "completed": False}])
        c.advance()
        c.click(10.0, 10.0)  # lane A via active lane
        c.select_lane("B")
        c.click(12.0, 40.0)
        c.click(30.0, 12.0)  # still lane B
        c.select_lane("A")
        c.click(32.0, 30.0)  # lane A second vertex
        xid = c.save_crossing(continuation={"A": "A", "B": "B"},
                              unresolved=False)
        rec = s.load(xid)
        assert rec["data"]["unresolved"] is False
        assert set(rec["data"]["lanes"]) == {"A", "B"}
        assert rec["data"]["lanes"]["A"] == [[10.0, 10.0], [32.0, 30.0]]
        c.save_and_next()
        assert c.progress() == (1, 1)
    finally:
        r.close()
        s.close()


def test_clump_grains_roots_revisit(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "g1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "owner", "completed": False},
            {"uuid": "g2", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "apex", "completed": False},
        ])
        c.advance()
        c.grain_click("A", 100.0, 100.0)
        c.root_click("A", 110.0, 105.0)
        c.grain_click("B", 200.0, 200.0)
        assert c.clump_complete() is False  # B lacks root
        c.root_click("B", 205.0, 195.0)
        assert c.clump_complete() is True
        c.save_and_next()  # g1 done
        assert c.current["uuid"] == "g2"
        back = c.go_back()  # revisit restores grains
        assert back["uuid"] == "g1"
        assert c._grains["B"]["root_xy"] == [205.0, 195.0]
        assert c.clump_complete() is True
    finally:
        r.close()
        s.close()


def test_multimovie_reader_routing(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r1 = FrameReader(MOVIE)
    r2 = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r1, viewer=None, actor="test")
        c.add_reader("default", r1)
        c.add_reader("m2", r2)
        assert c._reader_for({"movie": "m2"}) is r2
        assert c._reader_for({}) is r1  # default falls back to primary
        import pytest
        with pytest.raises(ValueError, match='No movie reader'):
            c._reader_for({"movie": "nope"})
    finally:
        r1.close()
        r2.close()
        s.close()


def test_undo_mark_reopens_task(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "t1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "apex",
                       "completed": False}])
        c.advance()
        assert c.undo_mark() is False  # nothing to undo
        c.click(5.0, 6.0)
        assert c.progress() == (1, 1)
        c.go_back()
        assert c.undo_mark() is True
        assert c.progress() == (0, 1)
        assert s.load("obs-t1") is None  # entity gone...
        assert len(s.history("obs-t1")) >= 2  # ...but audited
        assert c.undo_mark() is False  # twice = nothing left
    finally:
        r.close()
        s.close()


def test_completed_project_has_tasks_not_starters(tmp_path):
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    try:
        assert s.count_tasks() == 0
        s.save("task", "t1", {"completed": True}, actor="t")
        assert s.count_tasks() == 1
        assert s.unfinished_tasks() == []
    finally:
        s.close()


def test_owner_task_grain_root_routing(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        # Natural flow: no buttons needed — ball, tube-start, next
        # ball, its tube-start. The app auto-walks the pile.
        c.click(50.0, 50.0)  # Ball 1 itself
        assert c._grains["A"]["xy"] == [50.0, 50.0]
        assert c._mark_mode == "root"  # now wants Ball 1's tube start
        assert c.current["uuid"] == "o1"  # stays
        c.click(55.0, 56.0)  # Ball 1's tube start -> auto to Ball 2
        assert c._grains["A"]["root_xy"] == [55.0, 56.0]
        assert c._lane == "B"
        assert c._mark_mode == "grain"
        assert c.clump_next_needed() == ("B", "ball")
        c.click(200.0, 200.0)  # Ball 2 itself
        assert c._mark_mode == "root"
        c.click(205.0, 195.0)  # Ball 2's tube start
        assert c.clump_complete() is True
        # Pile walked itself to a fresh ball; every marked ball done.
        assert c._lane not in ("A", "B") or c.clump_complete()
        # Fix-up path: Ball buttons jump back to correct Ball 1.
        c.select_lane("A")
        assert c._lane == "A"  # complete ball re-arms ball click for redo
        c.finish_path_and_next()  # pile-done advances
        assert c.progress() == (1, 1)
    finally:
        r.close()
        s.close()


def test_skip_unresolvable_excludes_from_training(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "u1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        c.grain_click("A", 10.0, 10.0)  # partial work stays saved
        c.skip_unresolvable(reason="impossible clump")
        assert c.progress() == (1, 1)
        t = s.load("task-u1") if hasattr(s, "load") else None
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "p.db"))
        t = [r for r in con.execute(
            "SELECT data FROM entities WHERE kind='task'").fetchall()][0][0]
        import json
        assert json.loads(t)["resolution"] == "unresolvable"
        n_obs = con.execute(
            "SELECT COUNT(*) FROM entities WHERE kind='observation'"
        ).fetchone()[0]
        assert n_obs == 0  # NO observation entity: nothing to train on
        con.close()
    finally:
        r.close()
        s.close()


def test_pile_undo_takes_back_one_dot_at_a_time(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        c.click(10.0, 10.0)   # Ball 1
        c.click(12.0, 11.0)   # its tube start -> auto Ball 2
        c.click(50.0, 50.0)   # Ball 2
        assert c.undo_mark() is True   # removes Ball 2 dot
        assert "B" not in c._grains
        assert c._lane == "B" and c._mark_mode == "grain"
        assert c.undo_mark() is True   # removes Ball 1 tube start
        assert c._grains["A"] == {"xy": [10.0, 10.0]}
        assert c._mark_mode == "root"
        assert c.undo_mark() is True   # removes Ball 1 dot
        assert c._grains == {}
        assert c.undo_mark() is False  # nothing left
        assert c.current["uuid"] == "o1"  # still on the pile
    finally:
        r.close()
        s.close()


def test_pile_supports_five_balls(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        for i, lab in enumerate(["A", "B", "C", "D", "E"]):
            c.grain_click(lab, float(i), float(i))
            c.root_click(lab, float(i) + 1.0, float(i))
        assert c.clump_complete() is True
        assert c._grains["E"] == {"xy": [4.0, 4.0], "root_xy": [5.0, 4.0]}
        c.finish_path_and_next()
        assert c.progress() == (1, 1)
    finally:
        r.close()
        s.close()


def test_crossing_and_trace_undo(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "t01x", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "crossing", "completed": False},
            {"uuid": "t02c", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "centerline", "completed": False},
        ])
        c.advance()
        assert c.current["uuid"] == "t01x"
        c.click(1.0, 1.0)
        c.click(2.0, 2.0)
        assert c.undo_mark() is True
        assert c._lanes["A"] == [[1.0, 1.0]]
        c.click(3.0, 3.0)
        c.select_lane("B")
        c.click(9.0, 9.0)
        assert c.undo_mark() is True  # B's only dot gone
        assert c._lanes.get("B", []) == []
        c.save_and_next()  # parks the crossing, moves to the trace
        assert c.current["uuid"] == "t02c"
        c.click(5.0, 5.0)
        c.click(6.0, 6.0)
        assert c.undo_mark() is True
        assert c._path_pts == [[5.0, 5.0]]
    finally:
        r.close()
        s.close()


def test_pile_table_arm_and_growth(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        assert c.ordered_balls() == []
        assert c.next_fresh_ball() == "A"
        # Table LEFT cell aims at a fresh ball; picture click fills it.
        c.arm("A", "grain")
        assert (c._lane, c._mark_mode) == ("A", "grain")
        c.click(10.0, 10.0)
        assert c.ordered_balls() == ["A"]
        assert c.next_fresh_ball() == "B"  # fresh row opened below
        # Table RIGHT cell aims at its tube start.
        c.arm("A", "root")
        assert (c._lane, c._mark_mode) == ("A", "root")
        c.click(12.0, 11.0)
        assert c._grains["A"] == {"xy": [10.0, 10.0],
                                  "root_xy": [12.0, 11.0]}
        # Re-aim LEFT re-marks the ball without touching its tube dot.
        c.arm("A", "grain")
        c.click(11.0, 10.5)
        assert c._grains["A"]["xy"] == [11.0, 10.5]
        assert c._grains["A"]["root_xy"] == [12.0, 11.0]
    finally:
        r.close()
        s.close()


def test_pile_remove_ball_compacts(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        for i, lab in enumerate(["A", "B", "C"]):
            c.grain_click(lab, float(i), float(i))
            c.root_click(lab, float(i) + 1.0, float(i))
        assert c.remove_ball("B") is True
        # C slid up: table stays Ball 1, Ball 2 with no gaps.
        assert c.ordered_balls() == ["A", "B"]
        assert c._grains["B"] == {"xy": [2.0, 2.0], "root_xy": [3.0, 2.0]}
        assert c.next_fresh_ball() == "C"
        assert c.clump_complete() is True
        # Undo still works after compaction (history remapped, B's gone).
        assert c.undo_mark() is True
        assert c._grains["B"] == {"xy": [2.0, 2.0]}
        assert c.remove_ball("Z") is False  # empty row: no-op
        assert c.remove_ball("A") is True
        assert c.remove_ball("A") is True  # old B, now slid to A
        assert c._grains == {}
        assert c.undo_mark() is False
    finally:
        r.close()
        s.close()


def test_not_a_ball_fenced_off_from_training(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "t01o", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "owner", "completed": False},
            {"uuid": "t02a", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "apex", "completed": False},
        ])
        c.advance()
        assert c.current["uuid"] == "t01o"
        c.grain_click("A", 10.0, 10.0)  # partial work stays saved...
        c.save_not_a_ball(reason="debris")
        assert c.current["uuid"] == "t02a"  # ...but moves on
        import sqlite3, json
        con = sqlite3.connect(str(tmp_path / "p.db"))
        t = json.loads(con.execute(
            "SELECT data FROM entities WHERE kind='task' AND uuid='t01o'"
        ).fetchone()[0])
        assert t["resolution"] == "not_a_ball"
        assert t["completed"] is True
        n_obs = con.execute(
            "SELECT COUNT(*) FROM entities WHERE kind='observation'"
        ).fetchone()[0]
        assert n_obs == 0  # NO observation: nothing trainable, ever
        con.close()
    finally:
        r.close()
        s.close()


def test_pile_no_tube_flag_submits(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "o1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "owner",
                       "completed": False}])
        c.advance()
        c.click(10.0, 10.0)   # Ball 1
        c.click(12.0, 11.0)   # its tube -> auto Ball 2
        c.click(50.0, 50.0)   # Ball 2, no tube on it
        assert c.clump_complete() is False  # tube answer missing
        c.set_no_tube("B", True)            # flag it tube-less
        assert c._grains["B"] == {"xy": [50.0, 50.0], "no_tube": True}
        assert c.clump_complete() is True   # submits regardless now
        assert c.clump_next_needed()[0] == "C"
        c.finish_path_and_next()
        assert c.progress() == (1, 1)
        # Guard rails: flag needs the ball, refuses against a tube dot.
        c.go_back()
        try:
            c.set_no_tube("Z", True)
            assert False, "should refuse without a ball dot"
        except ValueError:
            pass
        try:
            c.set_no_tube("A", True)
            assert False, "should refuse over a tube dot"
        except ValueError:
            pass
        # Lifting the flag re-arms the tube click; drawing clears flags.
        c.set_no_tube("B", False)
        assert c._mark_mode == "root" and c._lane == "B"
        c.click(52.0, 49.0)
        assert c._grains["B"]["root_xy"] == [52.0, 49.0]
        assert "no_tube" not in c._grains["B"]
    finally:
        r.close()
        s.close()


def _repair_tasks():
    return [
        {"uuid": "ra1", "owner_uuid": "o1", "query_frames": [15000],
         "task_type": "review_tip", "completed": False,
         "draft_xy": [50.0, 51.0], "focus_xy": [50.0, 60.0]},
        {"uuid": "rp1", "owner_uuid": "o1", "query_frames": [15100],
         "task_type": "review_path", "completed": False,
         "draft_path": {"path_xy": [[10.0, 10.0], [20.0, 20.0],
                                    [30.0, 25.0]]}},
    ]


def test_review_tip_region_and_draft_confirm(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks(_repair_tasks())
        c.advance()
        assert c.current["uuid"] == "ra1"
        # Precise re-mark stays (no auto-advance on review tasks).
        c.click(51.0, 52.0)
        assert c.current["uuid"] == "ra1"
        c.finish_review_tip()
        assert c.current["uuid"] == "rp1"
        c.go_back()
        # Area mode: 3 corners + save.
        c._region_mode = True
        c.click(1.0, 1.0)
        c.click(9.0, 1.0)
        assert c.undo_mark() is True  # pops one corner, stays
        c.click(9.0, 1.0)
        c.click(5.0, 8.0)
        nxt = c.finish_region_and_next()
        assert nxt["uuid"] == "rp1"
        import sqlite3, json
        con = sqlite3.connect(str(tmp_path / "p.db"))
        o = json.loads(con.execute(
            "SELECT data FROM entities WHERE kind='observation' AND uuid='obs-ra1'"
        ).fetchone()[0])
        con.close()
        assert o["direct_state"] == "visible_imprecise"
        assert len(o["direct_region"]) == 3
        assert "direct_xy" not in o  # never an invented point
    finally:
        r.close()
        s.close()


def test_review_path_span_confirm(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "rp1", "owner_uuid": "o1",
                       "query_frames": [15100], "task_type": "review_path",
                       "completed": False,
                       "draft_path": {"path_xy": [[10.0, 10.0],
                                                  [20.0, 20.0]]}}])
        c.advance()
        assert c.click(99.0, 99.0)["uuid"] == "rp1"  # clicks don't draw
        c.confirm_path_span(False)
        c.save_and_next()
        assert c.progress() == (1, 1)
        import sqlite3, json
        con = sqlite3.connect(str(tmp_path / "p.db"))
        o = json.loads(con.execute(
            "SELECT data FROM entities WHERE kind='observation' AND uuid='obs-rp1'"
        ).fetchone()[0])
        con.close()
        assert o["path_complete"] is False
        assert o["direct_xy"] is None
        assert o["direct_state"] == "not_directly_visible"
        assert o["path_xy"] == [[10.0, 10.0], [20.0, 20.0]]
    finally:
        r.close()
        s.close()


def test_review_crossing_preload_and_confirm(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        s.save("crossing", "cross-1",
               {"lanes": {"A": [[1.0, 1.0], [2.0, 2.0]],
                          "B": [[5.0, 5.0], [6.0, 6.0]]},
                "unresolved": True}, actor="setup")
        c.load_tasks([{"uuid": "rx1", "owner_uuid": "o1",
                       "query_frames": [15000],
                       "task_type": "review_crossing", "completed": False,
                       "draft_crossing": "cross-1",
                       "draft_lanes": {"A": [[1.0, 1.0], [2.0, 2.0]],
                                       "B": [[5.0, 5.0], [6.0, 6.0]]}}])
        c.advance()
        assert c._lanes["A"] == [[1.0, 1.0], [2.0, 2.0]]  # preloaded draft
        c.click(3.0, 3.0)  # extend lane A
        xid = c.assign_continuation({"A": "tube-1"})
        assert xid == "cross-1"  # partial mapping stays unresolved
        rec = s.load("cross-1")
        assert rec["data"]["continuation"] == {"A": "tube-1"}
        assert rec["data"]["unresolved"] is True
        xid = c.assign_continuation({"A": "tube-1", "B": "tube-2"})
        assert s.load("cross-1")["data"]["unresolved"] is False
        c.save_and_next()
        assert c.progress() == (1, 1)
    finally:
        r.close()
        s.close()


def test_finish_neg_needs_a_box(tmp_path):
    # The done button still refuses an empty review (single clicks
    # advance on their own; the button is the left-over path).
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "n1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "neg_region",
                       "movie": "ld", "neg_class": "apex",
                       "completed": False}])
        c.advance()
        try:
            c.finish_neg()
        except ValueError:
            pass
        else:
            raise AssertionError("empty neg review must refuse")
        c.click(100.0, 100.0)  # far click: box + auto-advance
        assert c.progress() == (1, 1)
        assert len(c.tasks[0].get("neg_boxes", [])) == 1
    finally:
        r.close()
        s.close()


def test_all_dock_callbacks_defined():
    """Static audit: every self._on_* wired in TaskDock must exist.

    The neg/census/tube-id UI shipped layout+refresh without its
    callbacks and crashed the app on launch — twice. This fails
    headless instead of in front of the annotator.
    """
    import re
    from pathlib import Path

    from tubetracker.annotation_app import TaskDock
    src = (Path(__file__).resolve().parents[1] / "tubetracker" /
           "annotation_app.py").read_text()
    refs = set(re.findall(r"self\.(_on_\w+)", src))
    missing = sorted(m for m in refs if not hasattr(TaskDock, m))
    assert not missing, f"dock callbacks missing: {missing}"


def test_markup_draws_boxes_and_hides_on_switch(tmp_path):
    """draw_task_markup shows neg boxes/pending + census dots (fake viewer)."""

    class FakeLayer:
        def __init__(self):
            self.data = None
            self.visible = True
            self.editable = True

    class FakeLayers(dict):
        pass

    class FakeViewer:
        def __init__(self):
            self.layers = FakeLayers()
            self.added = []

        def add_shapes(self, data, **kw):
            lay = FakeLayer()
            lay.data = data
            self.layers[kw["name"]] = lay
            self.added.append(kw["name"])
            return lay

        def add_points(self, data, **kw):
            return self.add_shapes(data, **kw)

        def add_image(self, data, **kw):
            lay = FakeLayer()
            lay.data = data
            self.layers[kw.get("name", "raw")] = lay
            return lay

    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "n1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "neg_region", "movie": "ld", "completed": False},
            {"uuid": "c1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "census", "movie": "ld",
             "focus_xy": [500.0, 500.0], "completed": False},
        ])
        c.viewer = FakeViewer()  # type: ignore[assignment]
        c.add_reader('ld', r)
        c.current = next(t for t in c.tasks if t["uuid"] == "n1")
        c.click(100.0, 100.0)  # single click: fixed box + auto-advance
        assert c.current["uuid"] == "c1"
        c.go_back()  # revisit: the saved box must draw
        assert c.current["uuid"] == "n1"
        c.draw_task_markup()
        assert "neg-boxes" in c.viewer.layers
        box = c.viewer.layers["neg-boxes"].data[0]
        assert box[0] == [72.0, 72.0] and box[2] == [128.0, 128.0]
        # Switching tasks hides neg layers, shows census dots.
        c.current = next(t for t in c.tasks if t["uuid"] == "c1")
        c.click(490.0, 495.0)
        c.draw_task_markup()
        assert c.viewer.layers["neg-boxes"].visible is False
        assert "census-tips" in c.viewer.layers
    finally:
        r.close()
        s.close()


def test_neg_click_loop_tip_ring_box(tmp_path):
    """One click finishes a neg task: tip confirms, ring refuses, box saves."""
    import sqlite3
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = tmp_path / "p.db"
    s = AnnotationStore(db)
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "n1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "neg_region", "movie": "ld",
             "known_tip_xy": [500.0, 500.0], "completed": False},
            {"uuid": "n2", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "neg_region", "movie": "ld",
             "known_tip_xy": [500.0, 500.0], "completed": False},
            {"uuid": "n3", "owner_uuid": "o1", "query_frames": [15200],
             "task_type": "neg_region", "movie": "ld",
             "known_tip_xy": [500.0, 500.0], "completed": False},
        ])
        c.advance()
        c.click(500.0, 500.0)  # on the tip: confirm + advance
        assert c.current["uuid"] == "n2"
        obs = sqlite3.connect(str(db)).execute(
            "select data from entities where kind='observation'").fetchall()
        assert len(obs) == 1  # the confirmation
        c.click(520.0, 500.0)  # ring (20px): refuse, stay
        assert c.current["uuid"] == "n2"
        assert "Too close" in c._neg_note
        c.click(100.0, 100.0)  # far: fixed box + advance
        assert c.current["uuid"] == "n3"
        assert c.progress() == (2, 3)
    finally:
        r.close()
        s.close()


def test_neg_empty_verdict_advances_without_supervision(tmp_path):
    """'Nothing box-worthy' moves on and creates no regions."""
    import sqlite3
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = tmp_path / "p.db"
    s = AnnotationStore(db)
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "n1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "neg_region", "movie": "ld", "completed": False},
            {"uuid": "n2", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "neg_region", "movie": "ld", "completed": False},
        ])
        c.advance()
        nxt = c.finish_neg_empty()
        assert nxt is not None and nxt["uuid"] == "n2"
        rows = sqlite3.connect(str(db)).execute(
            "select data from entities where uuid='n1'").fetchall()
        import json
        assert json.loads(rows[-1][0]).get("reviewed_empty") is True
        nreg = sqlite3.connect(str(db)).execute(
            "select count(*) from entities where kind=\"region\"").fetchone()
        assert nreg[0] == 0
    finally:
        r.close()
        s.close()


def test_draft_click_confirms_inside_and_advances(tmp_path):
    """Draft adjudication: inside confirms+advances, outside stays."""
    import sqlite3
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = tmp_path / "p.db"
    s = AnnotationStore(db)
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "d1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "neg_draft", "movie": "ld",
             "draft_boxes": [[90.0, 90.0, 150.0, 150.0]],
             "neg_class": "apex", "completed": False},
            {"uuid": "d2", "owner_uuid": "o1", "query_frames": [15100],
             "task_type": "neg_draft", "movie": "ld",
             "draft_boxes": [[90.0, 90.0, 150.0, 150.0]],
             "neg_class": "apex", "completed": False},
        ])
        c.advance()
        c.click(10.0, 10.0)  # outside: stay + guidance
        assert c.current["uuid"] == "d1"
        assert "INSIDE" in c._neg_note
        c.click(120.0, 120.0)  # inside: confirm + advance
        assert c.current["uuid"] == "d2"
        nreg = sqlite3.connect(str(db)).execute(
            "select data from entities where kind=\"region\"").fetchall()
        assert len(nreg) == 1
        import json
        assert json.loads(nreg[0][0])["kind"] == "verified_negative"
    finally:
        r.close()
        s.close()


def test_draft_refresh_branch_offscreen():
    """The neg_draft panel text must render (real Qt, offscreen).

    Draft UI ships layout + controller logic; this executes the
    refresh branch a live open would hit first.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return  # Qt env only
    app = QApplication.instance() or QApplication([])
    from tubetracker.annotation_app import AnnotatorController, TaskDock
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    import tempfile
    tmp = tempfile.mkdtemp()
    s = AnnotationStore(tmp + "/p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "d1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "neg_draft",
                       "movie": "ld",
                       "draft_boxes": [[90.0, 90.0, 150.0, 150.0]],
                       "neg_class": "apex", "draft_source": "elbow",
                       "completed": False}])
        c.advance()
        dock = TaskDock(c)
        dock.refresh()
        assert "INSIDE" in dock.neg_info.text()
        # the app's current prompt for this branch (the old
        # "Nothing-box-worthy" wording is gone); pin the box question
        _t = dock.next_label.text()
        assert _t.strip() and "box" in _t.lower(), _t
    finally:
        r.close()
        s.close()


def test_mask_paint_erase_save_roundtrip(tmp_path):
    """rev6 body-mask pilot: paint stamps accumulate, erase removes by
    radius, empty paint refuses, save commits a mask entity."""
    import sys
    sys.path.insert(0, '.')
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = str(tmp_path / "m.db")
    s = AnnotationStore(db)
    r = FrameReader("/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4")
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([{"uuid": "bm-001", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "body_mask",
                       "movie": "ld", "brush_px": 9.0,
                       "completed": False}])
        c.advance()
        assert c.mask_coverage() == 0
        c.mask_paint([[100., 100.], [104., 100.], [200., 200.]])
        assert c.mask_coverage() == 3
        c.mask_erase_at(100., 100., 6.0)
        assert c.mask_coverage() == 1
        try:
            AnnotatorController(s, r, viewer=None,
                                actor="test").save_mask(True)
        except ValueError:
            pass
        else:
            raise AssertionError("empty paint must refuse")
        c.mask_paint([[102., 102.]])
        uuid = c.save_mask(True)
        rec = s.load(uuid)
        assert rec is not None and rec["kind"] == "mask"
        assert rec["data"]["complete"] is True
        assert len(rec["data"]["painted_xy"]) == 2
    finally:
        r.close()
        s.close()


def test_body_mask_from_points_complete_vs_partial():
    """Human mask raster (rev8 extent contract): a 'complete' mask
    supervises background ONLY inside the extent that was actually
    reviewed; without a recorded extent it stays band-only (unknown
    beyond), and an empty paint never supervises anything."""
    import sys
    import numpy as np
    sys.path.insert(0, '.')
    from prototypes.v30_video_apex.targets import body_mask_from_points
    pts = [[float(x), 50.] for x in range(40, 61)]
    # complete WITH a recorded reviewed extent -> background inside it
    extent = [[10.0, 10.0], [90.0, 90.0]]
    tc, vc = body_mask_from_points(100, 100, (0., 0.), pts,
                                   brush_px=9.0, complete=True,
                                   review_region=extent)
    tp, vp = body_mask_from_points(100, 100, (0., 0.), pts,
                                   brush_px=9.0, complete=False)
    assert int((tc > 0).sum()) > 50
    # valid = reviewed extent (10..90) union paint
    assert int(vc.sum()) == 81 * 81
    assert vc[5, 5] == 0.0, "outside the review stays unknown"
    assert vc[50, 50] == 1.0, "inside the review is checked background"
    assert vc[50, 45] == 1.0 and tc[50, 45] == 1.0
    assert int((vp.sum())) < int(vc.sum())
    assert int((vp[tc > 0]).sum()) == int((tc > 0).sum())
    # complete WITHOUT a recorded extent: no invented background
    tn, vn = body_mask_from_points(100, 100, (0., 0.), pts,
                                   brush_px=9.0, complete=True)
    assert int(vn.sum()) == int(vp.sum()), \
        "a flag alone must not license background"
    te, ve = body_mask_from_points(100, 100, (0., 0.), [],
                                   brush_px=9.0, complete=True)
    assert int(te.sum()) == 0 and int(ve.sum()) == 0


def test_mask_extent_reaches_db_and_snapshot(tmp_path):
    """rev8 step-1 exit: the reviewed extent survives UI save -> DB ->
    snapshot -> mask row, and a mask without a link is quarantined by
    the loader instead of being attached to another owner's tube."""
    import json as _json
    from tubetracker.annotation_store import AnnotationStore
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from prototypes.v30_video_apex.targets import samples_from_snapshot

    proj = tmp_path / "proj"
    proj.mkdir()
    s = AnnotationStore(proj / "annotations.db")
    r = FrameReader("/Users/joshjiang/Downloads/"
                    "test1lowdensjoshua-28c-hz.mp4 .mp4")
    c = AnnotatorController(s, r, viewer=None, actor="proof")
    try:
        c.load_tasks([{"uuid": "m1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o1",
                       "review_region": [[900.0, 250.0], [1220.0, 506.0]]}])
        c.current = c.tasks[0]
        c.mask_paint([[1000.0, 330.0]] * 5 + [[1005.0, 335.0]] * 5)
        c.save_mask(complete=True)
    finally:
        r.close()
    s.close()
    import sqlite3
    con = sqlite3.connect(str(proj / "annotations.db"))
    try:
        (data,) = con.execute(
            "SELECT data FROM entities WHERE kind='mask'").fetchone()
    finally:
        con.close()
    md = _json.loads(data)
    assert md["review_region"] == [[900.0, 250.0], [1220.0, 506.0]], md
    assert md["owner_uuid"] == "o1"
    # snapshot carries it; loader links by owner (same frame+owner)
    import subprocess
    import sys as _sys
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[1]
    snap = tmp_path / "snap"
    pr = subprocess.run(
        [_sys.executable, str(REPO / "scripts" / "build_v30_snapshot.py"),
         "--project-dir", str(proj),
         "--movie", f"ld=/Users/joshjiang/Downloads/"
         "test1lowdensjoshua-28c-hz.mp4 .mp4",
         "--default-movie", "ld", "--out", str(snap)],
        capture_output=True, text=True, cwd=str(REPO))
    assert pr.returncode == 0, pr.stderr[-1500:]
    rows = _json.loads((snap / "body_masks.json").read_text())
    assert rows and rows[0]["review_region"][0] == [900.0, 250.0]
    m = _json.loads((snap / "snapshot_manifest.json").read_text())
    assert m["n_masks_with_review_region"] == 1
    # loader: the mask names its own owner ("o1") and carries its
    # queried object, so rev8 links it as task-derived instead of
    # quarantining it. Quarantine now applies only to masks whose
    # owner can be nothing but the "|task:<uuid>" fallback (the legacy
    # case, covered by tests/test_mask_owner_links.py).
    ss = samples_from_snapshot(str(snap))
    masks = [x for x in ss if x.kind == "body_mask"]
    assert len(masks) == 1
    assert masks[0].quarantine_reason == ""
    assert masks[0].link_source == "owner-self"
    assert masks[0].owner_link_source == "task-derived"
    # this task declared no queried grain, so the mask carries none —
    # target_xy is recorded when the task names an object, never invented
    assert masks[0].target_xy == ()
    assert masks[0].review_region[0] == [900.0, 250.0]


def test_duel_vote_save_neither_roundtrip(tmp_path):
    """rev6 duels: click near a lane votes + advances; Neither saves a
    null verdict (both bad); far clicks guide without saving."""
    import sys
    sys.path.insert(0, '.')
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = str(tmp_path / "d.db")
    s = AnnotationStore(db)
    r = FrameReader("/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4")
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        lanes = {"A": [[100., 100.], [150., 150.]],
                 "B": [[300., 300.], [350., 350.]]}
        c.load_tasks([
            {"uuid": "q-001", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "route_duel", "movie": "ld",
             "draft_lanes": lanes, "completed": False},
            {"uuid": "q-002", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "route_duel", "movie": "ld",
             "draft_lanes": lanes, "completed": False}])
        c.advance()
        assert c.duel_vote(102., 101.) == "A"
        assert c.duel_vote(348., 352.) == "B"
        try:
            c.duel_vote(10., 10.)
        except ValueError:
            pass
        else:
            raise AssertionError("far click must refuse")
        c.click(102., 101.)  # votes A + advances
        assert c.current["uuid"] == "q-002"
        rec = s.load("duel-q-001")
        assert rec is not None and rec["data"]["winner"] == "A"
        c.save_duel("neither")
        rec2 = s.load("duel-q-002")
        assert rec2 is not None and rec2["data"]["winner"] == "neither"
    finally:
        r.close()
        s.close()


def test_trace_undo_repaints_and_advance_clears(tmp_path):
    """Undo on a trace must repaint the yellow line, not just pop data.

    Regression: path_click drew incrementally but undo only popped
    _path_pts, so the line on screen never shortened (undo "didn't
    work") and stale lines leaked across tasks.
    """
    class FakeLayer:
        def __init__(self):
            self.data = None
            self.visible = True

    class FakeLayers(dict):
        def remove(self, lay):
            for k, v in list(self.items()):
                if v is lay:
                    del self[k]

    class FakeViewer:
        def __init__(self):
            self.layers = FakeLayers()

        def add_shapes(self, data, **kw):
            lay = FakeLayer()
            lay.data = data
            self.layers[kw["name"]] = lay
            return lay

        def add_image(self, data, **kw):
            lay = FakeLayer()
            lay.data = data
            self.layers[kw.get("name", "raw")] = lay
            return lay

    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "t1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "centerline", "movie": "ld", "completed": False},
            {"uuid": "t2", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "centerline", "movie": "ld", "completed": False},
        ])
        c.viewer = FakeViewer()  # type: ignore[assignment]
        c.add_reader('ld', r)
        c.advance()
        c.path_click(100.0, 100.0)
        c.path_click(120.0, 110.0)
        assert len(c.viewer.layers["centerline"].data[0]) == 2
        assert c.undo_last_dot() is True
        assert c._path_pts == [[100.,100.]]
        assert 'centerline' not in c.viewer.layers  # one vertex is a point, not a native path
        assert c.undo_last_dot() is True
        assert "centerline" not in c.viewer.layers  # empty: layer removed
        c.path_click(100.0, 100.0)
        c.path_click(120.0, 110.0)
        c.finish_path_and_next()  # save t1, move on: no stale line leaks
        assert "centerline" not in c.viewer.layers
        assert c.current["uuid"] == "t2"
    finally:
        r.close()
        s.close()


def test_trace_not_a_ball_completes_with_no_observation(tmp_path):
    """A mined trace focus can be empty: 'Not a ball' must finish the
    task with NO observation (fenced from training, never a negative)."""
    import sqlite3
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = tmp_path / "p.db"
    s = AnnotationStore(db)
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "t1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "centerline", "movie": "ld", "completed": False},
            {"uuid": "t2", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "centerline", "movie": "ld", "completed": False},
        ])
        c.advance()
        c.save_not_a_ball()
        assert c.current["uuid"] == "t2"
        n = sqlite3.connect(str(db)).execute(
            "select count(*) from entities where kind='observation'"
        ).fetchone()[0]
        assert n == 0
    finally:
        r.close()
        s.close()


def test_census_undo_pops_newest_dot_of_either_kind(tmp_path):
    """Undo on a tile pops the last-placed dot, tips or balls.

    Regression: undo always popped tips first, so ball dots were
    uneatable while any tip existed.
    """
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "c1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "census", "movie": "ld",
             "focus_xy": [500.0, 500.0], "completed": False},
        ])
        c.advance()
        c.click(100.0, 100.0)  # tip mode default
        c._census_mode = "grain"
        c.click(200.0, 200.0)  # ball dot placed LAST
        assert c.undo_last_dot() is True
        assert c.current.get("census_grains", []) == []  # ball went first
        assert len(c.current.get("census_tips", [])) == 1
        assert c.undo_last_dot() is True
        assert c.current.get("census_tips", []) == []
    finally:
        r.close()
        s.close()


def test_dot_history_survives_reload(tmp_path):
    """Undo works after an app restart: history persists on the task."""
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    db = tmp_path / "p.db"
    s = AnnotationStore(db)
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "c1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "census", "movie": "ld",
             "focus_xy": [500.0, 500.0], "completed": False},
        ])
        c.advance()
        c.click(100.0, 100.0)
        c._census_mode = "grain"
        c.click(200.0, 200.0)
        # Simulate restart: fresh controller, same DB.
        c2 = AnnotatorController(s, r, viewer=None, actor="test")
        c2.load_tasks([dict(t) for t in [c.tasks[0]]])
        c2.advance()
        assert c2.undo_last_dot() is True
        assert c2.current.get("census_grains", []) == []
        assert len(c2.current.get("census_tips", [])) == 1
    finally:
        r.close()
        s.close()


def test_clear_task_marks_empties_dots_and_history(tmp_path):
    """'Clear my dots' restarts the task: dots + history gone, still open."""
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "c1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "census", "movie": "ld",
             "focus_xy": [500.0, 500.0], "completed": False},
        ])
        c.advance()
        c.click(100.0, 100.0)
        c._census_mode = "grain"
        c.click(200.0, 200.0)
        assert c.clear_task_marks() is True
        assert c.current.get("census_tips", []) == []
        assert c.current.get("census_grains", []) == []
        assert c._dot_history == []
        assert c.current.get("completed") is not True
        assert c.undo_last_dot() is False
    finally:
        r.close()
        s.close()


def test_look_only_blocks_all_marks(tmp_path):
    """Look-only mode: clicks move nothing into the data on any task."""
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    s = AnnotationStore(tmp_path / "p.db")
    r = FrameReader(MOVIE)
    try:
        c = AnnotatorController(s, r, viewer=None, actor="test")
        c.load_tasks([
            {"uuid": "c1", "owner_uuid": "o1", "query_frames": [15000],
             "task_type": "census", "movie": "ld",
             "focus_xy": [500.0, 500.0], "completed": False},
        ])
        c.advance()
        c._look_only = True
        c.click(100.0, 100.0)
        assert c.current.get("census_tips", []) == []
        assert c._dot_history == []
        c._look_only = False
        c.click(100.0, 100.0)
        assert len(c.current.get("census_tips", [])) == 1
    finally:
        r.close()
        s.close()
