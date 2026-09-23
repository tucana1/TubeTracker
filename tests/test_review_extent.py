"""rev8: the reviewed extent must survive a viewer that is not napari's
latest, and a saved mask must be revisitable.

napari 0.9.1 exposes no `camera.rect` (verified against the installed
package), so the extent is derived from camera center/zoom and the
canvas size. These tests use a deliberately minimal fake viewer: the
point is the contract, not napari.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_app import AnnotatorController  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402

MOVIE_LD = ("/Users/joshjiang/Downloads/"
            "test1lowdensjoshua-28c-hz.mp4 .mp4")


@pytest.fixture(autouse=True)
def fake_colormap_for_fake_viewer(monkeypatch):
    module = ModuleType('napari.utils.colormaps')
    module.DirectLabelColormap = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'napari.utils.colormaps', module)


class _Size:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def width(self):
        return self._w

    def height(self):
        return self._h


class _Canvas:
    def __init__(self, w, h):
        self.size = _Size(w, h)


class _Camera:
    """Stand-in for napari's camera.

    `center` is deliberately untyped: napari 0.9.1 can report a
    3-component centre (z, y, x), and the app must read the last two
    components in that case (see test_camera_centre_with_three_...).
    """

    def __init__(self):
        self.center = (500.0, 300.0)
        self.zoom = 5.0


class _Layer:
    def __init__(self, data, name):
        self.data = np.asarray(data)
        self.name = name
        self.editable = True
        event = SimpleNamespace(connect=lambda callback: None)
        self.events = SimpleNamespace(data=event, labels_update=event, set_data=event, reload=event)


class _Layers(dict):
    def __init__(self):
        super().__init__()
        self.selection = SimpleNamespace(active=None)


class _Viewer:
    """Minimal napari stand-in: camera, canvas, layers."""

    def __init__(self, frame_shape, show: bool = True):
        self.camera = _Camera()
        self.layers = _Layers()
        self._frame_shape = frame_shape
        self._show = show
        self.window = type("W", (), {})()
        if show:
            self.window.qt_viewer = type(
                "QV", (), {"canvas": _Canvas(1000.0, 800.0)})()

    def add_shapes(self, *a, **k):
        lay = _Layer(np.zeros((0, 2)), k.get("name", "shapes"))
        self.layers[lay.name] = lay
        self.shapes_calls = getattr(self, "shapes_calls", [])
        self.shapes_calls.append((a, k))
        return lay

    def add_labels(self, data, name="labels", **kwargs):
        lay = _Layer(data, name)
        for key, value in kwargs.items():
            setattr(lay, key, value)
        self.layers[name] = lay
        return lay


def test_review_extent_from_canvas_geometry(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m1", "owner_uuid": "o1",
                       "query_frames": [15000], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o1"}])
        c.current = c.tasks[0]
        # no viewer, nothing declared -> no invented extent
        assert c._current_review_region() == []
        # a viewer with camera + canvas yields the true visible rect:
        # 1000x800 canvas at zoom 5 centred on (y=500, x=300)
        v = _Viewer((1024, 1280))
        c.viewer = v
        ext = c._current_review_region()
        assert ext == [[300.0 - 100.0, 500.0 - 80.0],
                       [300.0 + 100.0, 500.0 + 80.0]], ext
        # declared region is the fallback when the canvas is unknown
        c.viewer = _Viewer((1024, 1280), show=False)
        c.current["review_region"] = [[10.0, 20.0], [30.0, 40.0]]
        assert c._current_review_region() == [[10.0, 20.0], [30.0, 40.0]]
        # save_mask persists the computed extent and the identity
        c.viewer = v
        c.mask_paint([[300.0, 500.0]] * 4)
        c.save_mask(complete=True)
    finally:
        reader.close()
        store.close()
    import json
    import sqlite3
    con = sqlite3.connect(str(proj / "annotations.db"))
    try:
        (data,) = con.execute(
            "SELECT data FROM entities WHERE kind='mask'").fetchone()
    finally:
        con.close()
    md = json.loads(data)
    assert md["review_region"] == [[200.0, 420.0], [400.0, 580.0]], md
    assert md["owner_uuid"] == "o1"
    assert md['review_region_provenance']['canvas_size_wh'] == [1000.,800.]
    from tubetracker.review_region import licensed_review_region
    assert licensed_review_region(md['review_region'],md['review_region_provenance']) == (
        md['review_region'],'native_canvas')


@pytest.mark.parametrize('native_widget', [True, False])
def test_napari_canvas_height_width_does_not_swap_the_reviewed_region(tmp_path, native_widget):
    store = AnnotationStore(tmp_path/'canvas-order.db')
    try:
        viewer = _Viewer((1024,1280))
        canvas = SimpleNamespace(size=(743,560), _scene_canvas=SimpleNamespace(size=(560,743)))
        if native_widget:
            canvas.native = SimpleNamespace(size=lambda: _Size(560,743))
        viewer.window.qt_viewer.canvas = canvas
        viewer.camera.center = (0.,500.,300.)
        viewer.camera.zoom = 4.
        c = AnnotatorController(store, SimpleNamespace(native_size=(1280,1024)), viewer=viewer)
        c.current = {'uuid':'extent','task_type':'body_mask'}
        assert c._canvas_size() == (560.,743.)
        assert c._current_review_region() == [[230.,407.125],[370.,592.875]]
        assert not store.entities()
    finally:
        store.close()


def test_mask_task_restores_committed_paint(tmp_path):
    """Reopening a painted mask task shows the paint again (no re-paint
    needed to correct or re-save it)."""
    proj = tmp_path / "p2"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m2", "owner_uuid": "o2",
                       "query_frames": [15000], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o2"}])
        c.current = c.tasks[0]
        frame = reader.read(15000)
        img = frame.frame
        img = img[:, :, 0] if img.ndim == 3 else img
        c._image_layer = _Layer(img, "raw")
        # paint a known block, commit it
        pts = [[x, 500.0] for x in range(300, 341, 5)]
        c.mask_paint(pts)
        from prototypes.v30_video_apex.targets import (
            encode_mask_raster)
        paint = np.zeros(img.shape[:2], dtype=bool)
        paint[498:503, 300:341] = True
        c.current["mask_raster"] = encode_mask_raster(paint)
        c.save_mask(complete=False)
        # reopen the task with a viewer: the paint must come back
        v = _Viewer(img.shape)
        c.viewer = v
        c.advance()
        lab = v.layers.get("mask-paint")
        if lab is None:  # task completed -> show it again explicitly
            c.current = c.tasks[0]
            c.refresh_task_display() if hasattr(
                c, "refresh_task_display") else c.advance()
            lab = v.layers.get("mask-paint")
        assert lab is not None, "mask-paint layer missing"
        restored = np.asarray(lab.data) > 0
        assert int(restored.sum()) == int(paint.sum()), (
            int(restored.sum()), int(paint.sum()))
        assert bool((restored == paint).all())
    finally:
        reader.close()
        store.close()


def test_mask_task_marks_the_target_grain(tmp_path):
    """A mask query must mark WHICH grain's tube is being asked about.

    In a paired crop the guide path exists for at most one of the two
    grains, so the target ring is what tells the painter where to
    start.
    """
    proj = tmp_path / "p3"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m3", "owner_uuid": "o3",
                       "query_frames": [15000], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o3"}])
        c.current = c.tasks[0]
        frame = reader.read(15000)
        img = frame.frame
        img = img[:, :, 0] if img.ndim == 3 else img
        c._image_layer = _Layer(img, "raw")
        v = _Viewer(img.shape)
        c.viewer = v
        # no target declared -> the focus is the object (honest default)
        c.current["focus_xy"] = [300.0, 500.0]
        c._show_mask_task()
        assert "mask-target" in v.layers
        (args, _kw), = v.shapes_calls[-1:]
        ring = np.asarray(args[0])[0]
        # a closed 32-vertex ring centred on (x=300, y=500), r=14
        assert ring.shape == (32, 2), ring.shape
        assert np.allclose(ring.mean(axis=0), [500.0, 300.0], atol=1e-6)
        rr = np.hypot(ring[:, 0] - 500.0, ring[:, 1] - 300.0)
        assert np.allclose(rr, 14.0, atol=1e-6), rr
        assert _kw.get("name") == "mask-target"
        assert _kw.get("shape_type") == "path"
        # an explicit target overrides the focus (paired crops)
        v2 = _Viewer(img.shape)
        c.viewer = v2
        c.current["target_xy"] = [310.0, 520.0]
        c.current["target_r"] = 20.0
        c._show_mask_task()
        ring2 = np.asarray(v2.shapes_calls[-1][0][0])[0]
        assert np.allclose(ring2.mean(axis=0), [520.0, 310.0], atol=1e-6)
        rr2 = np.hypot(ring2[:, 0] - 520.0, ring2[:, 1] - 310.0)
        assert np.allclose(rr2, 20.0, atol=1e-6), rr2
    finally:
        reader.close()
        store.close()


def test_camera_centres_the_ring_and_keeps_the_guide_in_frame(tmp_path):
    """Two complaints, two requirements, and they are not the same one.

    (a) The RING must sit in the middle of the canvas: its position is
    what the annotator reads as "which grain am I working on", and
    centring on the ring+guide union pushed it off to one side
    (reported twice against the live app). (b) The GUIDE still sets the
    zoom, so the whole tube the task asks to paint stays visible.
    """
    proj = tmp_path / "p4"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m4", "owner_uuid": "o4",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o4",
                       # ring at (357, 433); tube running to (391, 452)
                       "target_xy": [357.0, 433.0],
                       "guide_path": [[351.0, 400.0], [391.0, 452.0]]}])
        c.current = c.tasks[0]
        c._image_layer = _Layer(np.zeros((1200, 1600), np.uint8), "raw")
        v = _Viewer((1200, 1600))
        c.viewer = v
        assert c._fit_camera_to_task() is True
        cy, cx = v.camera.center
        # centred ON THE RING (note the (y, x) order napari uses)
        assert abs(cx - 357.0) < 2.0, cx
        assert abs(cy - 433.0) < 2.0, cy
        for x, y in c.current['guide_path']:
            assert abs(x-cx) * v.camera.zoom < 1000 / 2
            assert abs(y-cy) * v.camera.zoom < 800 / 2
    finally:
        reader.close()
        store.close()


@pytest.mark.parametrize('canvas,guide,fixed', [
    ((560, 743), [[399.73, 162.03], [469.46, 140.82]], False),
    ((743, 220), [[385., 166.], [410., 600.]], False),
    ((280, 200), [[385., 166.], [1250., 900.]], False),
    ((560, 743), [[399.73, 162.03], [469.46, 140.82]], True),
])
def test_camera_fits_one_sided_tubes_and_the_grain_ring_at_real_canvas_sizes(tmp_path, canvas, guide, fixed):
    store = AnnotationStore(tmp_path/'camera.db')
    try:
        viewer = _Viewer((1024, 1280))
        viewer.window.qt_viewer.canvas = _Canvas(*canvas)
        c = AnnotatorController(store, SimpleNamespace(native_size=(1280, 1024)), viewer=viewer)
        c.current = {'target_xy': [384.98, 165.90], 'target_r': 14., 'guide_path': guide,
                     'review_region_fixed': fixed,
                     'review_region': [[350, 100], [490, 100], [490, 200], [350, 200]]}
        assert c._fit_camera_to_task()
        cy, cx = viewer.camera.center
        if not fixed:
            assert (cx, cy) == (384.98, 165.90)
        points = guide + [[384.98-14, 165.90-14], [384.98+14, 165.90+14]]
        if fixed:
            points += c.current['review_region']
        for x, y in points:
            assert abs(x-cx) * viewer.camera.zoom < canvas[0] / 2
            assert abs(y-cy) * viewer.camera.zoom < canvas[1] / 2
        assert not store.entities()
    finally:
        store.close()


def test_camera_centre_with_three_components_is_read_as_yx(tmp_path):
    """napari 0.9.1 hands back (z, y, x): z is not the y centre.

    The measured signature of the old bug: every saved extent in every
    project had its y pair symmetric about 0, so the reviewed region
    fell outside the mask's crop and 'complete' masks silently
    degraded to band-only validity.
    """
    proj = tmp_path / "p5"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m6", "owner_uuid": "o6",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o6",
                       "target_xy": [351.3, 387.1],
                       "guide_path": [[351.0, 400.0], [391.0, 452.0]]}])
        c.current = c.tasks[0]
        c._image_layer = _Layer(np.zeros((1200, 1600), np.uint8), "raw")
        v = _Viewer((1200, 1600))
        v.camera.center = (0.0, 430.0, 355.0)     # (z, y, x)
        v.camera.zoom = 4.0
        c.viewer = v
        reg = c._current_review_region()
        assert reg, "extent must not be empty with a live canvas"
        (x0, y0), (x1, y1) = reg
        assert y0 < 430.0 < y1, reg               # y from the y component
        assert x0 < 355.0 < x1, reg
        assert y0 > 0.0, f"y centre read z: {reg}"
        # a 2-component centre must still work unchanged
        v.camera.center = (430.0, 355.0)
        reg2 = c._current_review_region()
        assert reg2[0][1] < 430.0 < reg2[1][1], reg2
        assert abs(reg2[0][0] - reg[0][0]) < 12.0, (reg, reg2)
    finally:
        reader.close()
        store.close()


def test_camera_falls_back_to_the_union_without_a_ring(tmp_path):
    """No ring to centre on: frame the guide instead of guessing."""
    proj = tmp_path / "p6"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m7", "owner_uuid": "o7",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o7",
                       "focus_xy": [360.0, 430.0],
                       "guide_path": [[351.0, 400.0], [391.0, 452.0]]}])
        c.current = c.tasks[0]
        c._image_layer = _Layer(np.zeros((1200, 1600), np.uint8), "raw")
        v = _Viewer((1200, 1600))
        c.viewer = v
        assert c._fit_camera_to_task() is True
        cy, cx = v.camera.center
        assert abs(cx - (351.0 + 391.0) / 2.0) < 8.0, cx
        assert abs(cy - (400.0 + 452.0) / 2.0) < 8.0, cy
    finally:
        reader.close()
        store.close()


def test_extent_that_misses_the_paint_is_never_stored(tmp_path):
    """WP-A.1 (H318): the H306 shape must not reach the database.

    Every historical extent had its y pair symmetric about 0. A task
    that still declares such a region must have it dropped at save
    time, with the reason recorded — never silently stored.
    """
    proj = tmp_path / "p7"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m7", "owner_uuid": "o7",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o7",
                       "target_xy": [355.0, 430.0],
                       "review_region": [[0.0, -50.0], [300.0, 50.0]],
                       "guide_path": [[351.0, 400.0], [391.0, 452.0]]}])
        c.current = c.tasks[0]
        c.current["mask_points"] = [[355.0, 430.0], [360.0, 430.0]]
        ok_region, note = c._validated_review_region(
            paint_xy=[[355.0, 430.0], [360.0, 430.0]])
        assert ok_region == [] and note == "misses-paint"
        uuid = c.save_mask(complete=True)
        stored = store.load(uuid)["data"]
        assert stored["review_region"] == [], (
            "a quarantined extent must not be stored")
        assert stored["review_region_note"] == "misses-paint"
        # the paint itself is still recorded exactly
        assert stored["painted_xy"] == [[355.0, 430.0], [360.0, 430.0]]
    finally:
        reader.close()
        store.close()


def test_good_extent_is_stored_and_the_camera_geometry_is_enough(tmp_path):
    """A region that encloses the paint survives validation untouched."""
    proj = tmp_path / "p8"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m8", "owner_uuid": "o8",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o8",
                       "target_xy": [355.0, 430.0],
                       "review_region": [[300.0, 360.0], [420.0, 500.0]],
                       "guide_path": [[351.0, 400.0], [391.0, 452.0]]}])
        c.current = c.tasks[0]
        region, note = c._validated_review_region(
            paint_xy=[[355.0, 430.0], [360.0, 430.0]])
        assert note == "ok" and region == [[300.0, 360.0], [420.0, 500.0]]
    finally:
        reader.close()
        store.close()


def test_no_tube_verdict_scopes_the_extent_to_the_ball(tmp_path):
    """A 'no tube' verdict has no paint: the ball is the scope."""
    proj = tmp_path / "p9"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m9", "owner_uuid": "o9",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o9",
                       "target_xy": [355.0, 430.0], "target_r": 14.0,
                       "review_region": [[300.0, 400.0], [400.0, 460.0]],
                       "guide_path": []}])
        c.current = c.tasks[0]
        region, note = c._validated_review_region(
            center_xy=[355.0, 430.0], radius=14.0)
        assert note == "ok" and region
        # a region nowhere near the ball is dropped
        c.current["review_region"] = [[900.0, 900.0], [980.0, 980.0]]
        region2, note2 = c._validated_review_region(
            center_xy=[355.0, 430.0], radius=14.0)
        assert region2 == [] and note2 == "misses-paint"
    finally:
        reader.close()
        store.close()


def test_reopen_marks_a_completed_task_unfinished(tmp_path):
    """WP-A.2 needs completed tasks to be revisitable.

    Re-saving a mask writes a new revision of the same uuid, so the
    original bytes stay in the store history.
    """
    from scripts.run_annotation_app import reopen_tasks
    proj = tmp_path / "p10"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    try:
        store.save("task", "t1", {"uuid": "t1", "completed": True})
        done = reopen_tasks(store, ["t1", "  ", "missing"], actor="t")
        assert done == ["t1"], done
        rec = store.load("t1")["data"]
        assert rec["completed"] is False
        assert rec["reopened_for_review"] is True
        # masks are never reopened: only tasks drive the queue
        store.save("mask", "m1", {"complete": True})
        assert reopen_tasks(store, ["m1"]) == []
        assert store.load("m1")["data"] == {"complete": True}
    finally:
        store.close()


def test_recommitting_a_reopened_mask_does_not_duplicate_stamps(tmp_path):
    """rev9 round-trip defect: 274 stamps became 548 on re-save.

    Reopening a mask restores its committed paint into the labels
    layer; committing then re-read the layer while the task already
    carried the same points, appending exact duplicates (measured:
    274 -> 548, 359 -> 718). The re-save must be idempotent in stamps.
    """
    proj = tmp_path / "p11"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    reader = FrameReader(MOVIE_LD)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor="t")
        c.add_reader("ld", reader)
        c.load_tasks([{"uuid": "m11", "owner_uuid": "o11",
                       "query_frames": [48300], "task_type": "body_mask",
                       "movie": "ld", "completed": False,
                       "owner_key": "ld|o11",
                       "target_xy": [520.0, 415.0],
                       "guide_path": []}])
        c.current = c.tasks[0]
        v = _Viewer((1200, 1600))
        c.viewer = v
        paint = np.zeros((1200, 1600), np.uint8)
        paint[400:430, 500:540] = 1          # the restored committed paint
        v.layers["mask-paint"] = _Layer(paint, "mask-paint")
        c.current["mask_points"] = [[500.0, 400.0], [503.0, 400.0]]
        c.commit_mask_labels(complete=True)
        n_first = len(c.current["mask_points"])
        assert n_first > 2, "the layer's paint must contribute stamps"
        c.commit_mask_labels(complete=True)
        assert len(c.current["mask_points"]) == n_first, (
            "re-committing duplicated stamps "
            f"({n_first} -> {len(c.current['mask_points'])})")
        uniq = {(round(q[0], 3), round(q[1], 3))
                for q in c.current["mask_points"]}
        assert len(uniq) == len(c.current["mask_points"])
    finally:
        reader.close()
        store.close()
