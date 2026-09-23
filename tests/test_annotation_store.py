"""Annotation store: transactions, revisions, resume, snapshot locks."""

from tubetracker.annotation_store import AnnotationStore


def test_save_revision_resume_and_finalize(tmp_path):
    s = AnnotationStore(tmp_path / "proj.db")
    try:
        assert s.unfinished_tasks() == []
        r1 = s.save("task", "t1", {"completed": False, "frame": 10},
                    actor="u1")
        assert r1 == 1
        assert len(s.unfinished_tasks()) == 1
        r2 = s.save("task", "t1", {"completed": True, "frame": 10},
                    actor="u1")
        assert r2 == 2
        assert s.unfinished_tasks() == []  # resume skips finished
        assert s.load("t1")["revision"] == 2
        hist = s.history("t1")
        assert [h["revision"] for h in hist] == [1, 2]
        assert s.load("nope") is None
        h = s.finalize_snapshot("snap1", {"members": ["t1"]})
        assert len(h) == 64
        try:
            s.finalize_snapshot("snap1", {"members": ["t1", "t2"]})
        except ValueError as e:
            assert "rejects mutation" in str(e)
        else:
            raise AssertionError("finalized snapshot mutated")
    finally:
        s.close()
    # Reopen: state survives (resume after close).
    s2 = AnnotationStore(tmp_path / "proj.db")
    try:
        assert s2.load("t1")["data"] == {"completed": True, "frame": 10}
    finally:
        s2.close()
