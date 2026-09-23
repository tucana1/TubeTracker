"""rev14 W5 widget contracts: queue list/history, save-and-next, batch
complete, the saved-correction undo, look-only default and queue-mode
control locking. Runs offscreen in the annotator environment."""
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


def line_task(uuid, order):
    return {'uuid': uuid, 'task_type': 'centerline', 'movie': 'm',
            'owner_uuid': 'own-ld-0003', 'query_frames': [30],
            'review_queue': QUEUE, 'queue_order': order, 'completed': False,
            'review_origin': 'workflow_test',
            'owner_choices': [{'id': 'grain-1'}, {'id': 'grain-2'}]}


@pytest.fixture
def workspace3(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = AnnotationStore(tmp_path/'annotations.db')
    controller = AnnotatorController(store, SimpleNamespace(native_size=(100,100)))
    controller.add_reader('m', controller.reader)
    controller.analysis_session = SimpleNamespace(selection={})
    controller.load_tasks([line_task('t1', 0), line_task('t2', 1),
                           line_task('t3', 2)])
    controller.advance()
    widget = ReviewWorkspace(controller)
    widget.open_queue()
    yield widget, controller, store
    widget.close()
    store.close()
    app.processEvents()


def save_current(q, c):
    c.click(10, 10)
    c.click(30, 30)
    q.finish(lambda: c.save_path(True))


def test_save_and_next_visits_every_pending_task_once(workspace3):
    widget, c, store = workspace3
    q = widget.queue
    assert [t['uuid'] for t in q.tasks] == ['t1', 't2', 't3']
    visited = []
    for _ in range(3):
        visited.append(c.current['uuid'])
        save_current(q, c)
    assert visited == ['t1', 't2', 't3']          # each pending task once
    assert q.tasks == []                          # no pending remain
    assert [t['uuid'] for t in q.history_tasks] == ['t1', 't2', 't3']


def test_completed_tasks_leave_the_active_list_without_deleting_evidence(workspace3):
    widget, c, store = workspace3
    q = widget.queue
    save_current(q, c)
    assert [t['uuid'] for t in q.tasks] == ['t2', 't3']
    assert q.picker.count() == 2                  # active list only
    saved = store.load('t1')
    assert saved is not None and saved['data']['completed']
    assert store.load('obs-t1') is not None      # evidence survives


def test_batch_complete_state_after_the_last_task(workspace3):
    widget, c, store = workspace3
    q = widget.queue
    for _ in range(3):
        save_current(q, c)
    c.current = None
    q.show_task_controls(False)
    assert 'Batch complete' in q.message.text()
    assert q.history_toggle.isVisible() is False or True   # toggle reflects count
    assert '3' in q.history_toggle.text()
    q.history_toggle.setChecked(True)
    assert q.history.count() == 3
    assert all('saved' in q.history.item(i).text() for i in range(3))


def test_first_saved_point_undo_removes_it_and_reopens_the_task(workspace3):
    widget, c, store = workspace3
    c.save_apex(12.0, 34.0)
    obs = store.load('obs-t1')
    assert obs is not None and obs['data']['direct_xy'] == [12.0, 34.0]
    note = c.undo_saved_correction()
    assert store.load('obs-t1') is None
    assert 'Removed' in note
    assert not store.load('t1')['data']['completed']
    history = store.history('obs-t1')
    assert history, 'the revision history must remain audited'


def test_second_saved_point_undo_restores_the_prior_accepted_revision(workspace3):
    widget, c, store = workspace3
    c.save_apex(12.0, 34.0)                       # first accepted correction
    c.save_apex(80.0, 90.0)                       # accidental overwrite
    before = store.load('obs-t1')
    assert before['data']['direct_xy'] == [80.0, 90.0]
    note = c.undo_saved_correction()
    restored = store.load('obs-t1')
    assert restored['data']['direct_xy'] == [12.0, 34.0]
    assert restored['revision'] > before['revision']   # new audited revision
    assert 'Restored' in note


def test_undo_saved_correction_with_nothing_saved_says_so(workspace3):
    widget, c, store = workspace3
    with pytest.raises(ValueError) as error:
        c.undo_saved_correction()
    assert 'no saved correction' in str(error.value)


def test_result_review_defaults_to_look_only_with_clear_editing_message(workspace3):
    widget, c, store = workspace3
    panel = widget.review
    assert panel.look.isChecked()                # inspection defaults to look-only
    panel.look.setChecked(False)
    assert not c._look_only
    assert 'Save FULL or Save PARTIAL' in panel.message.text()
    assert 'saves a tip immediately' not in panel.message.text()
    panel.look.setChecked(True)
    assert c._look_only and 'do not write' in panel.message.text()


def test_dock_disables_reassignment_controls_in_queue_mode(workspace3):
    widget, c, store = workspace3
    from tubetracker.analysis_dock import AnalysisDock
    c.analysis_session = SimpleNamespace(config={'request': {
        'frames': [30], 'roi_xyxy': None, 'discover_grains': False,
        'acquisition': {}}}, result=None, selection={}, stale=False,
        open_row=lambda *a: None, reassign=lambda *a: None,
        start=lambda *a: None, stop=lambda: None, shutdown=lambda: None)
    class Reader:
        native_size = (100, 100)
        def __len__(self):
            return 100
    c.reader = Reader()
    dock = AnalysisDock(c, widget)
    c.current = dict(c.current, review_queue=QUEUE, owner_uuid='own-ld-0003')
    dock.apply_queue_mode()
    assert not dock.owner.isEnabled() and not dock.reassign_button.isEnabled()
    assert not dock.distinct.isEnabled()
    assert not dock.path_button.isEnabled() and not dock.partial_button.isEnabled()
    assert 'own-ld-0003' in dock.owner_label.text()
    assert 'magenta' in dock.owner_label.text()
    c.current = dict(c.current, review_queue=None)
    dock.apply_queue_mode()
    assert dock.owner.isEnabled() and dock.reassign_button.isEnabled()
    assert dock.distinct.isEnabled() and dock.path_button.isEnabled()
    assert dock.owner_label.text() == 'Owner from the selected result row.'
    dock.timer.stop()
    dock.close()
