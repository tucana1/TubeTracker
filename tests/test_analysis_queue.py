import json
from types import SimpleNamespace

import numpy as np
import pytest

from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore
from prototypes.v30_video_apex.targets import encode_mask_raster, decode_mask_raster


def controller(tmp_path, tasks):
    store = AnnotationStore(tmp_path/'annotations.db')
    c = AnnotatorController(store, SimpleNamespace(native_size=(100, 100)))
    c.add_reader('m', c.reader)
    c.load_tasks(tasks)
    c.advance()
    return c, store


def task(uid, kind, **kwargs):
    return {'uuid': uid, 'movie': 'm', 'task_type': kind, 'owner_uuid': 'grain-1',
            'query_frames': [30], 'completed': False, **kwargs}


def test_unconfirmed_path_and_undo_survive_restart_without_creating_truth(tmp_path):
    c, store = controller(tmp_path, [task('a', 'centerline'), task('b', 'centerline')])
    c.path_click(10, 20); c.path_click(20, 30); c.path_click(40, 30)
    c.open_task('b'); c.path_click(70, 50)
    c.open_task('a')
    assert c._path_pts == [[10.,20.],[20.,30.],[40.,30.]]
    assert store.entities('observation') == []
    c.undo_last_dot()
    store.close()
    store = AnnotationStore(tmp_path/'annotations.db')
    resumed = AnnotatorController(store, SimpleNamespace(native_size=(100,100)))
    resumed.add_reader('m', resumed.reader)
    resumed.open_task('a')
    assert resumed._path_pts == [[10.,20.],[20.,30.]]
    resumed.save_path(False)
    assert not store.load('obs-a')['data']['path_complete']
    assert store.load('obs-a')['data']['direct_xy'] is None
    store.close()


def test_exact_negative_polygon_preserves_unknown_and_rejects_known_cap(tmp_path):
    c, store = controller(tmp_path, [task('n', 'neg_region', negative_geometry='polygon',
        review_region=[[0,0],[100,0],[100,100],[0,100]], known_caps=[[50,50]], review_origin='workflow_test')])
    for p in [[10,10],[30,10],[10,30]]:c.click(*p)
    assert store.entities('region') == []
    uid = c.save_negative_polygon()
    data = store.load(uid)['data']
    assert data['polygon_xy'] == [[10.,10.],[30.,10.],[10.,30.]]
    assert data['review_origin'] == 'workflow_test'
    c._region_pts = [[40,40],[60,40],[60,60],[40,60]]
    with pytest.raises(ValueError, match='contains a reviewed cap'):c.save_negative_polygon()
    assert len(store.entities('region')) == 1
    store.close()


def test_crossing_partial_assignment_stays_unresolved_and_lanes_survive(tmp_path):
    c, store = controller(tmp_path, [task('x', 'crossing', owner_choices=[{'id':'grain-1'},{'id':'grain-2'}])])
    for lab, points in [('A',[[10,10],[30,30]]),('B',[[10,30],[30,10]])]:
        for p in points:c.lane_click(lab,*p)
    uid = c.save_crossing({'A':'grain-1'}, unresolved=False)
    assert store.load(uid)['data']['unresolved']
    c.open_task('x')
    assert c._lanes['B'] == [[10.,30.],[30.,10.]]
    c.save_crossing({'A':'grain-1','B':'grain-2'}, unresolved=False)
    assert not store.load(uid)['data']['unresolved']
    assert store.load(uid)['data']['owner_uuids'] == ['grain-1','grain-2']
    store.close()


def test_mask_unknown_paint_and_erasure_roundtrip_with_zero_gradient(tmp_path):
    from prototypes.v30_video_apex.batch_builder import own_mask_target, finalize_body_supervision
    from prototypes.v30_video_apex.native_body import body_loss
    import torch
    c, store = controller(tmp_path, [task('paint','body_mask', review_region_fixed=True,
        review_region=[[0,0],[32,0],[32,32],[0,32]], target_xy=[10,10], review_origin='workflow_test')])
    labels = np.zeros((32,32),np.uint8)
    labels[8:12,8:24] = 1
    labels[8:12,15:18] = 2
    c.viewer = SimpleNamespace(layers={'mask-paint': SimpleNamespace(data=labels)})
    uid = c.commit_mask_labels(True)
    data = store.load(uid)['data']
    np.testing.assert_array_equal(decode_mask_raster(32,32,(0,0),data['mask_raster']),labels==1)
    np.testing.assert_array_equal(decode_mask_raster(32,32,(0,0),data['mask_unknown_raster']),labels==2)
    sample = SimpleNamespace(mask_raster=data['mask_raster'], mask_unknown_raster=data['mask_unknown_raster'],
                             complete=True, review_region=data['review_region'], mask_uuid=uid)
    bt = own_mask_target(sample,32,32,(0,0))
    bt = finalize_body_supervision(bt, confusable=labels==2)
    assert (bt.valid[labels==2] == 0).all()
    logits = torch.zeros((32,32), requires_grad=True)
    loss,_ = body_loss(logits, torch.tensor(bt.sel_self),torch.tensor(bt.sel_bg),torch.tensor(bt.sel_foreign))
    loss.backward()
    assert torch.count_nonzero(logits.grad[labels==2]) == 0
    labels[8:12,8:11] = 0
    c.commit_mask_labels(True)
    revised = store.load(uid)
    assert revised['revision'] == 2
    assert all(x>=11 for x,y in revised['data']['painted_xy'])
    store.close()


def test_empty_raster_is_valid_but_corruption_cannot_become_background():
    empty = encode_mask_raster(np.zeros((16,16), dtype=bool))
    assert not decode_mask_raster(16,16,(0,0),empty).any()
    corrupt = {'x0':0, 'y0':0, 'w':4, 'h':4, 'bits_b64':''}
    with pytest.raises(ValueError, match='Invalid mask raster'):
        decode_mask_raster(16,16,(0,0),corrupt)
