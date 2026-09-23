"""rev14 P1 controls: the independent review's widget-probe sequences as
permanent regressions. These drive real connected signals (button clicks,
the queue button, the stale timer slot), never helper calls alone."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import pytest

QtWidgets = pytest.importorskip('qtpy.QtWidgets', exc_type=ImportError)
pytest.importorskip('napari.layers', exc_type=ImportError)

from tubetracker.analysis_project import QUEUE
from tubetracker.analysis_queue import ReviewWorkspace
from tubetracker.analysis_dock import AnalysisDock
from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore


class Reader:
    native_size = (100, 100)

    def __len__(self):
        return 100


def task(uid, order=None):
    d = dict(uuid=uid, task_type='review_tip', movie='m', owner_uuid='grain',
             query_frames=[30], completed=False, review_origin='workflow_test')
    if order is not None:
        d.update(task_type='centerline', review_queue=QUEUE, queue_order=order,
                 owner_choices=[])
    return d


@pytest.fixture
def rig(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = AnnotationStore(tmp_path/'annotations.db')
    c = AnnotatorController(store, Reader(), actor='workflow-test')
    c.add_reader('m', c.reader)
    c.analysis_session = SimpleNamespace(
        config={'request': {'frames': [30], 'roi_xyxy': None,
                            'discover_grains': False, 'acquisition': {}}},
        result=None, selection={}, stale=False,
        open_row=lambda *a: None, reassign=lambda *a: None,
        start=lambda *a: None, stop=lambda: None, shutdown=lambda: None)
    c.load_tasks([task('inspection'), task('q1', 0), task('q2', 1)])
    c.open_task('inspection')
    w = ReviewWorkspace(c)
    dock = AnalysisDock(c, w)
    dock.timer.stop()
    yield w, dock, c, store
    dock.close()
    w.close()
    store.close()
    app.processEvents()


def test_queue_button_transition_disables_result_review_controls(rig):
    w, dock, c, store = rig
    app = QtWidgets.QApplication.instance()
    dock.queue_button.click()          # real connected control
    app.processEvents()
    dock.refresh_stale()               # the timer's actual slot
    assert c.current['uuid'] == 'q1'
    assert not dock.reassign_button.isEnabled()
    assert not dock.owner.isEnabled()
    assert 'magenta' in dock.owner_label.text()
    assert c.current['owner_uuid'] in dock.owner_label.text()


def test_final_save_shows_batch_complete_and_stops_editing(rig):
    w, dock, c, store = rig
    app = QtWidgets.QApplication.instance()
    dock.queue_button.click()
    app.processEvents()
    full = next(b for b in w.queue.findChildren(QtWidgets.QPushButton)
                if b.text() == 'Save FULL current path')
    visited = []
    for _ in range(2):
        visited.append(c.current['uuid'])
        c.click(10, 10)
        c.click(30, 30)
        full.click()                   # no manual completion hook
        app.processEvents()
    assert visited == ['q1', 'q2']
    assert w.queue.tasks == []
    assert w.queue.actions.isHidden()  # editing stops after the final save
    assert 'Batch complete' in w.queue.message.text()


def test_returning_to_result_review_agrees_and_look_only_blocks_writes(rig):
    w, dock, c, store = rig
    app = QtWidgets.QApplication.instance()
    c.open_task('inspection')
    w.refresh()
    app.processEvents()
    assert w.review.look.isChecked() and c._look_only   # one authoritative mode
    c.click(55, 55)                    # a click in look-only must not write
    assert store.load('obs-inspection') is None


def test_explicit_edit_mode_survives_refresh_and_real_click_saves_then_undoes(rig):
    w, dock, c, store = rig
    app = QtWidgets.QApplication.instance()
    w.review.look.click()  # real toggled signal previously reset itself in refresh()
    app.processEvents()
    w.refresh()
    dock.refresh_stale()
    assert not w.review.look.isChecked() and not c._look_only
    c.click(55, 54)
    w.refresh()
    app.processEvents()
    assert store.load('obs-inspection')['data']['direct_xy'] == [55., 54.]
    assert not w.review.look.isChecked() and not c._look_only
    undo = next(b for b in w.review.findChildren(QtWidgets.QPushButton)
                if b.text() == 'Undo saved correction')
    undo.click()
    app.processEvents()
    assert store.load('obs-inspection') is None


def test_new_owner_context_defaults_to_look_only_without_overriding_same_context_editing(rig):
    w, dock, c, store = rig
    w.review.look.click()
    assert not c._look_only
    c.current['owner_uuid'] = 'different-grain'
    w.refresh()
    assert w.review.look.isChecked() and c._look_only


def test_region_save_is_unavailable_for_points_and_requires_a_polygon_draft(rig):
    w, dock, c, store = rig
    save_region = next(b for b in w.review.action_buttons if b.text() == 'Save tip region')
    assert not save_region.isEnabled()
    w.review.look.click()
    c.click(55, 54)
    w.refresh()
    assert store.load('obs-inspection')['data']['direct_xy'] == [55., 54.]
    assert not save_region.isEnabled()
    w.review.tip_mode.click()
    assert 'Outline the visible tip area' in w.review.info.text()
    for xy in [(20,20),(40,20)]:
        c.click(*xy)
        w.refresh()
        assert not save_region.isEnabled()
    c.click(30,40)
    w.refresh()
    assert save_region.isEnabled()
    save_region.click()
    saved = store.load('obs-inspection')['data']
    assert saved['direct_state'] == 'visible_imprecise'
    assert saved.get('direct_xy') is None and len(saved['direct_region']) == 3
    w.review.tip_mode.click()
    assert not save_region.isEnabled() and not c._region_pts


def test_two_undos_traverse_accepted_edits_not_restorations(rig):
    w, dock, c, store = rig
    c.load_tasks([task('undo-history')])
    c.open_task('undo-history')
    for xy in ((11, 11), (22, 22), (33, 33)):
        c.save_apex(*xy)
    c.undo_saved_correction()
    assert store.load('obs-undo-history')['data']['direct_xy'] == [22., 22.]
    c.undo_saved_correction()
    assert store.load('obs-undo-history')['data']['direct_xy'] == [11., 11.]


def test_delete_then_new_answer_then_undo_returns_to_no_answer(rig):
    w, dock, c, store = rig
    c.load_tasks([task('undo-removed')])
    c.open_task('undo-removed')
    c.save_apex(11, 11)
    c.undo_saved_correction()          # first answer removed
    assert store.load('obs-undo-removed') is None
    c.save_apex(44, 44)
    c.undo_saved_correction()          # the new answer is removed too
    assert store.load('obs-undo-removed') is None
    assert not any(t['uuid'] == 'obs-undo-removed' for t in store.entities('observation'))


def test_completed_history_includes_archived_batches_without_pending(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = AnnotationStore(tmp_path/'annotations.db')
    c = AnnotatorController(store, Reader(), actor='workflow-test')
    c.add_reader('m', c.reader)
    c.analysis_session = SimpleNamespace(selection={})
    archived = dict(task('old-1'), completed=True,
                    review_queue=QUEUE + '-completed-20260919',
                    queue_archive={'previous_review_queue': QUEUE,
                                   'reason': 'archived'})
    c.load_tasks([task('q1', 0)])
    store.save('task', 'old-1', archived, actor='workflow-test')  # as a real project holds it
    w = ReviewWorkspace(c)
    w.queue.populate()
    assert [t['uuid'] for t in w.queue.tasks] == ['q1']       # never pending
    assert 'old-1' in [t['uuid'] for t in w.queue.history_tasks]
    w.close()
    store.close()
    app.processEvents()
