"""Viewport geometry must not invent reviewed background for training."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tubetracker.review_region import licensed_review_region, REVIEW_GEOMETRY_SCHEMA
from prototypes.v30_video_apex.targets import encode_mask_raster


def test_legacy_rectangles_keep_only_the_common_view_and_polygons_keep_declared_scope():
    wide = [[10,30],[110,70]]
    tall = [[40,0],[80,100]]
    for region in (wide,tall):
        licensed, policy = licensed_review_region(region)
        assert licensed == [[40.,30.],[80.,70.]]
        assert policy == 'legacy_rectangle_intersection'
    polygon = [[10,30],[110,30],[110,70],[10,70]]
    assert licensed_review_region(polygon) == (polygon, 'declared_polygon')


@pytest.mark.parametrize('change', [{}, {'canvas_size_wh':[64,128]}, {'camera_zoom':float('nan')},
                                  {'schema':'unknown'}, {'camera_center_xy':[65,32]}])
def test_canvas_provenance_must_reproduce_the_recorded_extent(change):
    region = [[32.,16.],[96.,48.]]
    provenance = {'schema':REVIEW_GEOMETRY_SCHEMA, 'kind':'native_canvas',
                  'canvas_size_wh':[128,64], 'camera_center_xy':[64,32], 'camera_zoom':2}
    provenance.update(change)
    licensed, policy = licensed_review_region(region, provenance)
    if change:
        assert licensed == [] and policy == 'invalid_provenance'
    else:
        assert licensed == region and policy == 'native_canvas'


def test_declared_field_provenance_cannot_be_reused_for_a_larger_rectangle():
    region = [[10,30],[110,70]]
    provenance = {'schema':REVIEW_GEOMETRY_SCHEMA,'kind':'declared_field','region_xy':copy.deepcopy(region)}
    assert licensed_review_region(region,provenance) == (region,'declared_field')
    assert licensed_review_region([[0,0],[128,128]],provenance) == ([], 'invalid_provenance')


def test_native_panel_and_loss_exclude_ambiguous_extent_and_unreviewed_paint_band(tmp_path,monkeypatch):
    from scripts import train_native_body as trainer
    from tubetracker import annotation_frames
    from prototypes.v30_video_apex import targets
    from prototypes.v30_video_apex.native_body import body_loss

    paint = np.zeros((64,128),bool)
    paint[28:34,20:24] = True  # Explicit paint outside the conservative rectangle remains positive.
    paint[28:34,58:62] = True
    sample = SimpleNamespace(kind='body_mask', mask_uuid='fixture', obs_revision=1,
        mask_raster=encode_mask_raster(paint), mask_unknown_raster=None, mask_points=[],
        review_region=[[0,16],[120,48]], review_region_provenance=None, complete=True,
        brush_px=5, quarantine_reason='', movie='m', source_frame=0,
        target_xy=[32,32], target_r=13, owner_key='m|a', provenance='test-fixture')
    class Reader:
        def __init__(self,_path):pass
        def read(self,_frame):return SimpleNamespace(exact=True,frame=np.full((64,128,3),128,np.uint8))
        def close(self):pass
    monkeypatch.setattr(annotation_frames,'FrameReader',Reader)
    monkeypatch.setattr(targets,'samples_from_snapshot',lambda _snapshot:[sample])
    monkeypatch.setattr(trainer,'render',lambda *args,**kwargs:None)
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    (snapshot/'snapshot_manifest.json').write_text(json.dumps({'movies':{'m':{'path':'fixture'}}}))
    (snapshot/'regions.json').write_text('[]')
    out=tmp_path/'panel'
    trainer.prepare(SimpleNamespace(snapshot=str(snapshot),out=str(out)))
    doc=json.loads((out/'panel.json').read_text())
    arrays=np.load(out/'panel.npz')
    origin=doc['cases'][0]['origin']
    def at(x,y):return (0,int(y-origin[1]),int(x-origin[0]))
    assert arrays['positive'][at(21,30)]
    assert arrays['ordinary'][at(50,30)]
    assert not arrays['ordinary'][at(100,30)]  # Ambiguous background strip.
    assert not arrays['ordinary'][at(19,30)]   # Geometric band is not a review.
    assert doc['cases'][0]['unlicensed_band_pixels_excluded'] > 0
    assert doc['cases'][0]['extent_scope_policy'] == 'legacy_rectangle_intersection'
    logits=torch.zeros_like(torch.from_numpy(arrays['positive']),dtype=torch.float32,requires_grad=True)
    domains=[torch.from_numpy(arrays[k]) for k in ('positive','ordinary','foreign')]
    loss,_=body_loss(logits,*domains)
    loss.backward()
    assert logits.grad[at(21,30)] < 0 and logits.grad[at(50,30)] > 0
    assert logits.grad[at(100,30)] == logits.grad[at(19,30)] == 0


def test_old_native_panels_cannot_silently_reintroduce_unreviewed_negative_pixels(tmp_path):
    from scripts import train_native_body as trainer
    panel=tmp_path/'old-panel';panel.mkdir()
    (panel/'panel.json').write_text(json.dumps({'schema':'tubetracker.native_body_panel.v1'}))
    out=tmp_path/'new-run'
    with pytest.raises(ValueError,match='predates the reviewed-geometry/selector fix'):
        trainer.fit(SimpleNamespace(panel=str(panel),out=str(out)))
    assert not out.exists()
