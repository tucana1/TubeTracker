"""Qt control contracts; run with the annotator environment, without OpenGL."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import pytest

QtWidgets = pytest.importorskip('qtpy.QtWidgets', exc_type=ImportError)
pytest.importorskip('napari.layers', exc_type=ImportError)

from tubetracker.analysis_project import QUEUE
from tubetracker.analysis_queue import ReviewWorkspace
from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore


@pytest.fixture
def workspace(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = AnnotationStore(tmp_path/'annotations.db')
    controller = AnnotatorController(store, SimpleNamespace(native_size=(100,100)))
    controller.add_reader('m', controller.reader)
    controller.analysis_session = SimpleNamespace(selection={})
    controller.load_tasks([{'uuid':'crossing', 'task_type':'crossing', 'movie':'m',
        'owner_uuid':'unassigned', 'query_frames':[30], 'review_queue':QUEUE, 'queue_order':0,
        'completed':False, 'review_origin':'workflow_test',
        'owner_choices':[{'id':'grain-1'}, {'id':'grain-2'}]}])
    controller.advance()
    widget = ReviewWorkspace(controller)
    widget.open_queue()
    yield widget, controller, store
    widget.close()
    store.close()
    app.processEvents()


def test_lane_controls_save_distinct_owners_and_partial_mapping_stays_unresolved(workspace):
    widget, c, store = workspace
    q = widget.queue
    for point in [(10,10),(30,30)]: c.click(*point)
    # Accessibility changes checked state without emitting a mouse click.
    q.lane.button(1).setChecked(True)
    for point in [(10,30),(30,10)]: c.click(*point)
    q.assignments['A'].button(1).setChecked(True)
    q.finish(q.save_crossing)
    saved = store.entities('crossing')[0]
    assert saved['data']['unresolved']
    q.assignments['B'].button(2).setChecked(True)
    q.finish(q.save_crossing)
    revised = store.load(saved['uuid'])
    assert revised['revision'] == 2
    assert not revised['data']['unresolved']
    assert revised['data']['owner_uuids'] == ['grain-1','grain-2']
    assert revised['data']['lanes']['A'] == [[10.,10.],[30.,30.]]
    assert revised['data']['lanes']['B'] == [[10.,30.],[30.,10.]]


def test_look_only_disables_native_mask_brush_as_well_as_controller_clicks(workspace):
    widget, c, _ = workspace
    class Layers(dict):
        selection = SimpleNamespace(active=None)
    layer = SimpleNamespace(selected_label=1, brush_size=5, mode='paint')
    c.current['task_type'] = 'body_mask'
    c.viewer = SimpleNamespace(layers=Layers({'mask-paint':layer}))
    q = widget.queue
    q.brush_mode.button(2).setChecked(True)
    assert layer.selected_label == 2
    q.look.setChecked(True)
    assert layer.mode == 'pan_zoom' and c._look_only
    q.brush_mode.button(0).click()
    assert layer.mode == 'pan_zoom'
    q.look.setChecked(False)
    assert layer.mode == 'paint' and layer.selected_label == 0
    c._context_offset = 30
    q.brush()
    assert layer.mode == 'pan_zoom'
    c.viewer = None


def test_native_paint_and_undo_persist_without_creating_confirmed_masks(workspace):
    import numpy as np
    from napari.layers import Labels
    from prototypes.v30_video_apex.targets import decode_mask_raster
    widget, c, store = workspace
    c.current['task_type'] = 'body_mask'
    layer = Labels(np.zeros((100,100),dtype=np.uint8), metadata={'task_uuid':c.current['uuid']})
    c.viewer = SimpleNamespace(layers={'mask-paint':layer})
    c._watch_mask_draft(layer)
    layer.brush_size = 5
    layer.paint([20,20], 1)
    draft = store.load(c.current['uuid'])['data']['mask_draft_raster']
    assert decode_mask_raster(100,100,(0,0),draft).sum() > 0
    assert 'Draft saved' in widget.queue.message.text()
    layer.paint([20,20], 2)
    task = store.load(c.current['uuid'])['data']
    assert decode_mask_raster(100,100,(0,0),task['mask_unknown_draft_raster']).sum() > 0
    widget.queue.undo()
    task = store.load(c.current['uuid'])['data']
    assert decode_mask_raster(100,100,(0,0),task['mask_unknown_draft_raster']).sum() == 0
    assert store.entities('mask') == []
    c.viewer = None


def test_cannot_judge_withdraws_saved_crossing_and_explicit_save_reconfirms(workspace):
    widget, c, store = workspace
    q = widget.queue
    c.click(10,10); c.click(30,30)
    q.lane.button(1).setChecked(True)
    c.click(10,30); c.click(30,10)
    q.assignments['A'].button(1).setChecked(True)
    q.assignments['B'].button(2).setChecked(True)
    q.finish(q.save_crossing)
    original = store.entities('crossing')[0]
    q.unresolved()
    withdrawn = store.load(original['uuid'])
    assert withdrawn['revision'] == original['revision'] + 1
    assert withdrawn['data']['review_status'] == 'withdrawn'
    # rev14 P1: an unresolved verdict is an answer; the queue advances and
    # (with nothing pending) closes with the batch-complete state.
    assert q.tasks == [] and 'Batch complete' in q.message.text()
    # Reopening the task from history and saving explicitly reconfirms it.
    q.open(c.current['uuid'])
    q.finish(q.save_crossing)
    revised = store.load(original['uuid'])
    assert revised['revision'] == original['revision'] + 2
    assert revised['data']['review_status'] == 'active'
    assert store.load(c.current['uuid'])['data']['review_verdict'] == 'confirmed'


def test_empty_queue_has_no_active_annotation_controls_and_recovers_when_a_task_arrives(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = AnnotationStore(tmp_path/'empty.db')
    c = AnnotatorController(store, SimpleNamespace(native_size=(100, 100)))
    c.add_reader('m', c.reader)
    c.analysis_session = SimpleNamespace(selection={})
    widget = ReviewWorkspace(c)
    try:
        widget.open_queue()
        q = widget.queue
        assert q.actions.isHidden() and not q.actions.isEnabled()
        assert q.context_navigation.isHidden() and q.look.isHidden()
        assert not q.navigation.isEnabled() and not q.picker.isEnabled()
        assert q.message.text() == 'No requested annotations in this project.'
        assert store.entities() == []
        c.load_tasks([{'uuid': 'new-path', 'task_type': 'centerline', 'movie': 'm',
            'owner_uuid': 'grain', 'query_frames': [30], 'review_queue': QUEUE,
            'queue_order': 0, 'completed': False, 'review_origin': 'workflow_test'}])
        widget.open_queue()
        assert not q.actions.isHidden() and q.actions.isEnabled()
        assert not q.path_actions.isHidden() and q.crossing_actions.isHidden()
        assert not q.context_navigation.isHidden() and not q.look.isHidden()
        assert q.navigation.isEnabled() and q.picker.isEnabled()
        assert store.entities('observation') == []
    finally:
        widget.close()
        store.close()
        app.processEvents()


@pytest.mark.parametrize('origin,label', [('human','human reviews'), ('workflow_test','workflow-test reviews')])
def test_imported_review_display_is_scoped_and_never_creates_a_local_answer(workspace, origin, label):
    widget, c, store = workspace
    c.current.update(task_type='review_tip', owner_uuid='grain-1')
    c.analysis_session.result = {'movie_id':'m', 'rows':[
        {'owner_id':'grain-1', 'source_frame':30, 'state':'present', 'provenance':'human-corrected',
         'path_complete':True, 'length_px':73.2186,
         'constraint':{'review_origin':origin, 'state':'direct_visible',
                       'lineage':[{'id':'prior-full', 'revision':2}]}}]}
    before = store.entities()
    widget.review.refresh()
    assert label in widget.review.message.text()
    assert 'FULL path, length 73.22 px' in widget.review.message.text()
    assert 'prior-full' in widget.review.message.toolTip()
    c.current['movie'] = 'another-movie'
    widget.review.refresh()
    assert 'Last computed' not in widget.review.message.text()
    assert widget.review.message.toolTip() == ''
    c.current.update(movie='m', query_frames=[31])
    widget.review.refresh()
    assert 'Last computed' not in widget.review.message.text()
    assert store.entities() == before


def test_napari_canvas_wrapper_uses_real_qt_logical_dimensions(workspace):
    _, c, _ = workspace
    native = QtWidgets.QWidget()
    native.resize(560,743)
    c.viewer = SimpleNamespace(window=SimpleNamespace(_qt_viewer=SimpleNamespace(
        canvas=SimpleNamespace(native=native, size=(743,560)))))
    try:
        assert c._canvas_size() == (560.,743.)
    finally:
        c.viewer = None
        native.close()
