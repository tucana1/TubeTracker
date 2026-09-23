import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
QtWidgets = pytest.importorskip('qtpy.QtWidgets', exc_type=ImportError)
pytest.importorskip('napari.layers', exc_type=ImportError)

from test_rev14_p1_controls import rig
from tubetracker.analysis_project import QUEUE
from tubetracker.grain_identity import grain_identity_reviews
from tubetracker.analysis_contracts import AnalysisRequest


def button(w, label):
    return next(b for b in w.queue.findChildren(QtWidgets.QPushButton) if b.text() == label)


def test_identity_context_confirmation_and_uncertainty_drive_connected_controls(rig):
    w, dock, c, store = rig
    for uid, frame in [('q1', 30), ('q2', 40)]:
        data = store.load(uid)['data']
        data.update(task_type='grain_identity', query_frames=[frame], reference_frame=60,
                    context_frames=[50, 60], review_origin='human')
        store.save('task', uid, data)
    c.actor = 'reviewer'
    w.open_queue()
    assert w.queue.identity_actions.isVisibleTo(w.queue)
    w.queue.context_choices.activated.emit(1)
    assert c._context_offset == 30 and not w.queue.actions.isEnabled()
    assert w.queue.context_choices.currentData() == 60
    c.click(20, 20)
    assert store.entities('grain_identity') == []
    button(w, 'Task frame').click()
    assert w.queue.context_choices.currentData() == 30 and w.queue.actions.isEnabled()
    button(w, '+1').click()
    assert w.queue.context_choices.currentData() == 31 and not w.queue.actions.isEnabled()
    button(w, 'Task frame').click()
    c.click(20, 20)
    button(w, 'Save grain centre and next').click()
    assert c.current['uuid'] == 'q2'
    button(w, 'Cannot judge this task').click()
    assert w.queue.tasks == [] and w.queue.batch_complete
    rows, _ = grain_identity_reviews(store.entities(), AnalysisRequest('', 'm', [0]),
                                    [{'id':'grain'}], source=str(store.path))
    assert [r['identity_state'] for r in rows] == ['confirmed', 'ambiguous']
    assert store.entities('observation') == []


def test_tip_point_and_visibility_queue_never_skip_a_task(rig):
    w, dock, c, store = rig
    for position, uid in enumerate(['q1', 'q2'], 1):
        data = store.load(uid)['data']
        data.update(task_type='review_tip', queue_batch='finite-two', queue_position=position)
        store.save('task', uid, data)
    w.open_queue()
    assert w.queue.picker.item(0).text().startswith('1/2')
    assert not w.queue.point_next.isEnabled() and not w.queue.area_save.isEnabled()
    c.click(25, 25)
    w.refresh()  # wire_canvas_click refreshes the workspace after each click.
    assert 'Point saved' in w.queue.message.text()
    assert w.queue.point_next.isEnabled() and not w.queue.area_save.isEnabled()
    button(w, 'Undo saved correction').click()
    assert store.load('obs-q1') is None
    assert 'No tip point saved' in w.queue.message.text()
    assert not w.queue.point_next.isEnabled()
    c.click(25, 25)
    w.refresh()
    button(w, 'Continue after saved point').click()
    assert c.current['uuid'] == 'q2'
    assert w.queue.picker.item(0).text().startswith('2/2')
    assert w.queue.progress.text() == '1/2 answered · 1 remaining in this batch'
    button(w, 'Tip hidden — save and next').click()
    assert w.queue.tasks == [] and w.queue.batch_complete
    assert 'all 2 tasks in this batch answered' in w.queue.message.text()
    assert len(store.entities('observation')) == 2


def test_queue_area_save_requires_area_mode_and_a_polygon(rig):
    w, _, c, store = rig
    data = store.load('q1')['data']; data['task_type'] = 'review_tip'
    store.save('task', 'q1', data)
    w.open_queue()
    c.click(25, 25)
    w.refresh()
    assert not w.queue.area_save.isEnabled()
    w.queue.tip_region.click()
    assert not w.queue.point_next.isEnabled() and not w.queue.area_save.isEnabled()
    for point in [(20,20),(40,20)]:
        c.click(*point)
        assert not w.queue.area_save.isEnabled()
    c.click(30,40)
    assert w.queue.area_save.isEnabled()
    w.queue.area_save.click()
    assert c.current['uuid'] == 'q2'
    answer = store.load('obs-q1')['data']
    assert answer['direct_state'] == 'visible_imprecise'
    assert answer.get('direct_xy') is None and len(answer['direct_region']) == 3


def test_reference_highlight_uses_real_napari_points_and_context_cannot_save(rig):
    from napari.components import ViewerModel
    import numpy as np
    w, dock, c, store = rig
    w.open_queue()
    c.current.update(task_type='grain_identity', reference_frame=60,
                     reference_owners=[{'id':'grain', 'grain_native':[25,40]}])
    c._context_offset = 30
    c.viewer = ViewerModel()
    before = store.entities()
    try:
        w.queue.draw_overlay()
        layer = c.viewer.layers['queue-reference-grain']
        np.testing.assert_allclose(layer.data, [[40,25]])
        assert not layer.border_width_is_relative and not layer.editable
        np.testing.assert_allclose(layer.border_width, 2)
        c.click(25, 40)
        assert store.entities() == before
        dock.apply_queue_mode()
        assert 'Reference grain frame' in dock.owner_label.text()
        assert not dock.owner.isEnabled() and dock.owner.isHidden()
    finally:
        c.viewer = None
