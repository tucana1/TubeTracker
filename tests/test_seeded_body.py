import copy
import csv
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tubetracker.analysis_contracts import AnalysisRequest, route_model_bindings
from tubetracker.analysis_dependencies import dependency_fingerprint
from tubetracker.movie_analysis import export_analysis
from tubetracker.seeded_body import (SeededBodyConfig, LosslessFrameSequence,
                                     prepare_body_assistance, mask_core_points)
from test_body_presence import mask_record
from test_movie_analysis import service_fixture


class FrameFixture:
    native_size = (64, 64)

    def __init__(self, path):
        self.closed = False

    def __len__(self):
        return 100

    def read(self, frame):
        image = np.zeros((64,64,3),np.uint8)
        image[:,:,2] = np.arange(64,dtype=np.uint8)
        image[:,:,0] = frame
        return SimpleNamespace(frame=image,exact=True)

    def close(self):
        self.closed = True


def assisted_fixture(tmp_path, monkeypatch, *, second_owner=False):
    service, request, native_calls = service_fixture(tmp_path)
    source = tmp_path/'sam-source'
    package = source/'sam2'
    (package/'configs').mkdir(parents=True)
    (package/'sam2_video_predictor.py').write_text('# package fixture\n')
    (package/'configs/fixture.yaml').write_text('fixture: true\n')
    checkpoint = tmp_path/'sam.pt'
    checkpoint.write_bytes(b'fixture weights; never loaded by the injected producer')
    request.body_assistance = {'source_dir':str(source),'checkpoint':str(checkpoint),
        'model_config':'configs/fixture.yaml','seed_mask_ids':['mask-a']}
    request.roi_xyxy = [0,0,64,64]
    request.owners[0]['identity_verified'] = True
    if second_owner:
        request.owners.append({'id':'b','grain_native':[8,56],'grain_radius_px':3,
                              'attachment_native':[10,56],'attachment_verified':True})
    entities = [{'kind':'task','uuid':'body-task','revision':1,
                 'data':{'movie':'m','task_type':'body_mask','review_origin':'human'}},
                {'kind':'mask','uuid':'mask-a','revision':1,'data':mask_record()}]
    monkeypatch.setattr('tubetracker.annotation_frames.FrameReader',FrameFixture)
    assisted_calls = []

    def producer(request, plan, arrays, *, device, progress):
        assisted_calls.append(copy.deepcopy(plan))
        bindings = {}
        roi = plan['roi_xyxy']
        for frame in request.frames:
            bindings[str(frame)] = {}
            for seed in plan['seeds']:
                key = f"assisted-{frame}-{seed['owner_id']}"
                arrays[key] = np.full((roi[3]-roi[1],roi[2]-roi[0]),.99,np.float32)
                bindings[str(frame)][seed['owner_id']] = {
                    'body_key':key,'origin':roi[:2],
                    'prompt':{'backend':'sam2_video','assistance_identity':plan['identity'],
                              'classification':plan['classification'],'root_channel':False,
                              'seed':{k:seed[k] for k in ('source_id','source_project','revision','frame','mask_sha256')}}}
        return bindings, {'plan_identity':plan['identity'],'frame_pixel_sha256':{}}

    monkeypatch.setattr('tubetracker.seeded_body.run_sam2_body',producer)
    return service,request,entities,native_calls,assisted_calls,producer


def test_seed_plan_uses_exact_positive_pixels_and_separate_future_exposure(tmp_path, monkeypatch):
    _, request, entities, _, _, _ = assisted_fixture(tmp_path,monkeypatch)
    request.frames = [0]
    plan = prepare_body_assistance(request,request.owners,entities,source=str(tmp_path))
    assert plan['source_frames'] == [0,30]
    assert plan['effective_device'] == 'cpu'
    request.body_assistance['device'] = 'mps'
    accelerated = prepare_body_assistance(request,request.owners,entities,source=str(tmp_path),device='cpu')
    assert accelerated['effective_device'] == 'mps' and accelerated['identity'] != plan['identity']
    seed = plan['seeds'][0]
    assert seed['frame'] == 30 and seed['revision'] == 1 and seed['labels'] == [1]*len(seed['points_native'])
    assert seed['source_id'] == 'mask-a'
    for x,y in seed['points_native']:
        assert 36 <= x < 41 and 26 <= y < 30
    assert len(seed['points_native']) <= 8
    pixels = {'cap_checkpoint_sha256':'cap','body_checkpoint_sha256':'body','body_assistance':plan}
    assert route_model_bindings(pixels)['body_assistance'] == plan['identity']
    plan['seeds'][0]['frame'] = 31
    with pytest.raises(ValueError,match='identity'):
        route_model_bindings(pixels)


def test_service_overlays_seeded_owners_keeps_fallback_and_caches_provenance(tmp_path, monkeypatch):
    service,request,entities,native_calls,assisted_calls,_ = assisted_fixture(tmp_path,monkeypatch,second_owner=True)
    first = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    pixels = json.loads(open(first['artifacts']['pixels']).read())
    assert pixels['frames']['30']['owners']['a']['body_key'].startswith('assisted-')
    assert pixels['frames']['30']['owners']['b']['body_key'] == 'b-30'
    assert first['body_assistance_receipt']['plan_identity'] == assisted_calls[0]['identity']
    assert first['population']['model_body_assistance']['seeds'][0]['source_id'] == 'mask-a'
    assert 'reviewed-mask-assisted' in first['modes']['baseline']
    again = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert again['cache']['measurement_hit'] and len(native_calls) == len(assisted_calls) == 1
    request.acquisition = {'seconds_per_source_frame':2,'cadence_source':'fixture acquisition'}
    timed = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert timed['cache']['inference_hit'] and len(assisted_calls) == 1
    entities[1]['revision'] = 2
    changed = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert len(assisted_calls) == 2 and changed['cache']['pixel_key'] != first['cache']['pixel_key']
    with pytest.raises(ValueError,match='seed mask is missing'):
        service.analyze(request,review_entities=entities[:1],review_source=str(tmp_path))
    entities[0]['data']['review_status'] = 'withdrawn'
    with pytest.raises(ValueError,match='seed mask is missing'):
        service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert len(assisted_calls) == 2


def test_assistance_survives_measurement_export_without_becoming_human_truth(tmp_path, monkeypatch):
    service,request,entities,_,_,_ = assisted_fixture(tmp_path,monkeypatch)
    result = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert all(r['provenance'] == 'model-inferred' for r in result['rows'])
    paths = export_analysis(result,tmp_path/'exports')
    rows = list(csv.DictReader(open(paths['csv'])))
    assert all(r['model_body_provider'] == 'sam2_video' for r in rows)
    assert all(r['body_seed_mask_id'] == 'mask-a' and r['body_seed_frame'] == '30' for r in rows)
    assert all(r['length_px'] == '' for r in rows)
    from prototypes.v30_video_apex.route_quality import compare_validation_cases
    truth = {'obs_uuid':'point','obs_revision':1,'owner_uuid':'a','movie':'m',
             'source_frame':30,'direct_state':'direct_visible','direct_xy':[50,32],'review_origin':'human'}
    panel = {'scope':{'movie':'m'},'cases':[],'visible_tip_cases':[
        {'kind':'visible_tip','owner_id':'a','source_frame':30,'observation_id':'point'}]}
    case = compare_validation_cases(result,[truth],panel,set())[0]
    assert not case['excluded_from_fit_frames']  # prompting is exposure, even without optimizer updates


def test_scoped_root_edit_reuses_sam_pixels_but_native_root_dependency_remains(tmp_path, monkeypatch):
    service,request,entities,native_calls,assisted_calls,_ = assisted_fixture(tmp_path,monkeypatch,second_owner=True)
    service._body_pixel_contract = lambda _expected: {'schema':'fixture-root-conditioned','uses_root':True}
    initial = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    def full(owner, y):
        return [
            {'kind':'task','uuid':'full-'+owner,'revision':1,
             'data':{'movie':'m','task_type':'review_path','owner_uuid':owner,'review_origin':'human'}},
            {'kind':'observation','uuid':'obs-'+owner,'revision':1,
             'data':{'task_uuid':'full-'+owner,'owner_uuid':owner,'source_frame':30,
                     'direct_state':'direct_visible','direct_xy':[50,y],
                     'path_xy':[[11,y],[50,y]],'path_complete':True,'review_origin':'human'}}]
    seeded_edit = service.analyze(request,review_entities=entities+full('a',32),review_source=str(tmp_path))
    assert seeded_edit['cache']['pixel_key'] == initial['cache']['pixel_key']
    assert len(assisted_calls) == 1
    fallback_edit = service.analyze(request,review_entities=entities+full('b',56),review_source=str(tmp_path))
    assert fallback_edit['cache']['pixel_key'] != initial['cache']['pixel_key']
    assert len(native_calls) == len(assisted_calls) == 2


def test_failed_provider_publishes_no_pixel_manifest_and_retry_is_safe(tmp_path, monkeypatch):
    service,request,entities,_,_,producer = assisted_fixture(tmp_path,monkeypatch)
    def broken(*args,**kwargs):
        bindings,receipt = producer(*args,**kwargs)
        bindings['30']['a']['origin'] = [1,0]
        return bindings,receipt
    monkeypatch.setattr('tubetracker.seeded_body.run_sam2_body',broken)
    with pytest.raises(ValueError,match='invalid native probability'):
        service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert not list(service.cache_dir.glob('pixels-*.json'))
    monkeypatch.setattr('tubetracker.seeded_body.run_sam2_body',producer)
    result = service.analyze(request,review_entities=entities,review_source=str(tmp_path))
    assert len(result['rows']) == 2 and not result['cache']['pixel_hit']


def test_assisted_weights_and_package_config_are_dependencies_without_sam_import(tmp_path, monkeypatch):
    service,request,_,_,_,_ = assisted_fixture(tmp_path,monkeypatch)
    config = {'request':request.to_dict(),'cap_checkpoint':service.cap_checkpoint,'body_checkpoint':service.body_checkpoint}
    first = dependency_fingerprint(config)
    assert 'body_assistance/checkpoint' in first['files']
    package_file = next(v['path'] for k,v in first['files'].items() if k.endswith('package/configs/fixture.yaml'))
    from pathlib import Path
    Path(package_file).write_text('fixture: changed\n')
    assert dependency_fingerprint(config) != first
    request.roi_xyxy = None
    with pytest.raises(ValueError,match='explicit analysis ROI'):
        request.validate()
    broken = dict(request.body_assistance,model_config='../outside.yaml')
    with pytest.raises(ValueError,match='inside'):
        SeededBodyConfig.from_dict(broken)
    broken = dict(request.body_assistance,seed_mask_ids=['mask-a','mask-a'])
    with pytest.raises(ValueError,match='distinct'):
        SeededBodyConfig.from_dict(broken)


def test_lossless_loader_bounds_memory_and_records_exact_native_crop(tmp_path, monkeypatch):
    monkeypatch.setattr('tubetracker.annotation_frames.FrameReader',FrameFixture)
    roi = [5,7,20,22]
    sequence = LosslessFrameSequence('fixture',[0,30,60],roi,16)
    try:
        first = sequence[0]
        assert first.shape == (3,16,16) and str(first.dtype) == 'torch.float32'
        sequence[1]; sequence[2]
        assert len(sequence.cache) == 2 and 0 not in sequence.cache
        raw = FrameFixture('').read(0).frame
        native = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)[7:22,5:20]
        assert sequence.pixel_hashes['0'] == hashlib.sha256(native.tobytes()).hexdigest()
        np.testing.assert_array_equal(sequence[0].numpy(),first.numpy())
    finally:
        sequence.close()
    assert sequence.reader.closed and not sequence.cache


def test_positive_seed_points_never_fill_holes_or_unknown_pixels():
    mask = np.zeros((40,40),bool)
    mask[10:30,10:30] = True
    mask[16:24,16:24] = False
    points = mask_core_points(mask,8).astype(int)
    assert len(points) == len(set(map(tuple,points))) == 8
    assert all(mask[y,x] for x,y in points)
    with pytest.raises(ValueError,match='positive'):
        mask_core_points(np.zeros((2,2)),8)


def test_seed_point_selection_is_independent_of_raster_bounding_box():
    tight = np.ones((5,15),bool)
    canvas = np.pad(tight,((11,13),(9,17)))
    np.testing.assert_array_equal(mask_core_points(tight,8)+[9,11],
                                  mask_core_points(canvas,8))
