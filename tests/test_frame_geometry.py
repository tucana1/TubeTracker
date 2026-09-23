import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tubetracker.analysis_contracts import file_hash, route_model_bindings
from tubetracker.analysis_dependencies import dependency_fingerprint
from tubetracker.frame_geometry import (SCHEMA, grain_anchor_inputs, load_grain_poses,
                                        owner_at_frame, pose_index, result_owner_at_frame)
from tubetracker.movie_analysis import export_analysis, route_owner_for
from prototypes.v30_video_apex.route_quality import geometry_fingerprint
from test_movie_analysis import service_fixture
from test_seeded_body import FrameFixture


def pose_fixture(tmp_path, request):
    request.owners[0].update(identified_at_frame=0, identity_verified=True,
                             census_source={'task':'census','revision':2,'review_origin':'human'})
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir(exist_ok=True)
    (snapshot/'observations.json').write_text('[]')
    (snapshot/'census.json').write_text(json.dumps([{
        'task_uuid':'census', 'task_revision':2, 'movie':request.movie_id,
        'source_frame':0, 'tile_xywh':[0,0,64,64], 'class_scopes':['grains'],
        'border_policy':'centre-inside', 'clump_policy':'individual-physical-grains',
        'complete':True, 'membership_resolved':True, 'review_origin':'human',
        'instances':[{'grain_id':request.owners[0]['id'],
                      'xy':request.owners[0]['grain_native'],'confirmed':True}]}]))
    request.snapshot=str(snapshot)
    artifact = {'schema':SCHEMA, 'movie_id':request.movie_id,
        'movie_sha256':file_hash(request.movie_path), 'coordinate_system':'native_pixels',
        'image_size':[64,64], 'review_origin':'model', 'provider':'fixture-motion',
        'anchor_owners':grain_anchor_inputs(request.owners), 'poses':[]}
    for owner in request.owners:
        for frame in request.frames:
            artifact['poses'].append({'owner_id':owner['id'], 'source_frame':frame,
                'grain_native':[owner['grain_native'][0] + frame/3,owner['grain_native'][1]],
                'grain_radius_px':owner['grain_radius_px'], 'pose_status':'current_image_supported',
                'identity_status':'tracked_provisional', 'geometry_complete':True,
                'review_origin':'model', 'evidence':{'detection_instance_id':owner['id']}})
    path=tmp_path/'poses.json';path.write_text(json.dumps(artifact))
    request.grain_pose_path=str(path)
    request.route_policy={'root_selection':'current_body_rim'}
    return path,artifact


def load(request):
    return load_grain_poses(request.grain_pose_path, request, request.owners, file_hash(request.movie_path))


def test_frame_geometry_transports_center_without_laundering_root_or_mutating_inventory(tmp_path):
    _,request,_=service_fixture(tmp_path);path,artifact=pose_fixture(tmp_path,request)
    before=copy.deepcopy(request.owners)
    plan=load(request);poses=pose_index(plan)
    moved=owner_at_frame(request.owners[0],30,poses)
    assert moved['grain_native']==[18.,32.] and moved['attachment_native']==[20.,32.]
    assert not moved['attachment_verified'] and moved['grain_geometry_complete']
    assert moved['grain_pose']['review_origin']=='model'
    assert not moved['grain_pose']['orientation_observed']
    assert moved['grain_pose']['reference_to_frame_translation'][0][-1]==10
    assert request.owners==before
    reviewed,root=route_owner_for(moved,{30:{'a':{'xy':[21,32],'scope_frame':30,
        'observation_id':'full','revision':3,'basis':'human_full_trace'}}},30)
    assert reviewed['attachment_verified'] and reviewed['attachment_native']==[21,32]
    assert result_owner_at_frame({'grain_geometry':plan},before[0],30)['grain_native']==[18.,32.]
    pixels={'cap_checkpoint_sha256':'cap','body_checkpoint_sha256':'body','grain_geometry':plan}
    assert route_model_bindings(pixels)['grain_geometry']==plan['identity']
    assert geometry_fingerprint(before,None,'grain_crop',grain_geometry=plan)!=geometry_fingerprint(before,None,'grain_crop')
    pixels['grain_geometry']['poses'][0]['grain_native'][0]+=1
    with pytest.raises(ValueError,match='identity'):
        route_model_bindings(pixels)


@pytest.mark.parametrize('mutation,message',[
    (lambda d:d.update(review_origin='human'),'model evidence'),
    (lambda d:d.update(movie_sha256='different'),'different source movie'),
    (lambda d:d['poses'].append(copy.deepcopy(d['poses'][0])),'duplicate'),
    (lambda d:d['poses'].pop(),'coverage'),
    (lambda d:d['poses'][0].update(grain_native=[float('nan'),32]),'finite'),
    (lambda d:d['poses'][0].update(review_origin='human'),'cannot become human'),
    (lambda d:d['poses'][0].update(grain_native=[65,32]),'outside'),
])
def test_pose_contract_rejects_false_provenance_or_wrong_coverage(tmp_path,mutation,message):
    _,request,_=service_fixture(tmp_path);path,artifact=pose_fixture(tmp_path,request)
    mutation(artifact);path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError,match=message):load(request)


def test_anchor_revision_and_provider_evidence_make_motion_stale(tmp_path):
    service,request,_=service_fixture(tmp_path);path,artifact=pose_fixture(tmp_path,request)
    source=tmp_path/'provider-result.json';source.write_text('{}')
    artifact['evidence_files']={'provider':{'path':str(source),'sha256':file_hash(source)}}
    path.write_text(json.dumps(artifact));load(request)
    config={'request':request.to_dict(),'cap_checkpoint':service.cap_checkpoint,'body_checkpoint':service.body_checkpoint}
    before=dependency_fingerprint(config)
    source.write_text('{"changed":true}')
    assert before!=dependency_fingerprint(config)
    with pytest.raises(ValueError,match='source changed'):load(request)
    source.write_text('{}')
    request.owners[0]['census_source']['revision']=3
    with pytest.raises(ValueError,match='anchor revision'):load(request)


def test_shared_detector_instance_cannot_become_two_physical_grains(tmp_path):
    _,request,_=service_fixture(tmp_path)
    request.owners.append({'id':'b','grain_native':[8,50],'grain_radius_px':3})
    path,artifact=pose_fixture(tmp_path,request)
    for p in artifact['poses']:p['evidence']['detection_instance_id']='merged-object'
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError,match='two grain owners'):load(request)


def test_native_body_queries_and_routes_use_the_same_frame_centers(tmp_path,monkeypatch):
    service,request,_=service_fixture(tmp_path);pose_fixture(tmp_path,request)
    service.pixel_provider=None
    service._cap_model=SimpleNamespace(temporal=False)
    service._body_model=SimpleNamespace(pooling_lattice=1,normalization='pixel',root_channel=False)
    service._body_kind='tubetracker.native_owned_body.v1'
    monkeypatch.setattr(service,'_body_pixel_contract',lambda _: {'schema':service._body_kind,'uses_root':False})
    monkeypatch.setattr('tubetracker.annotation_frames.FrameReader',FrameFixture)
    monkeypatch.setattr('prototypes.v30_video_apex.native_caps.detect_caps',lambda *args,**kw:
        {'caps':[{'cap_id':f"cap-{kw['source_frame']}",'tip_xy':[50,32],
                  'probability':.99,'source_frame':kw['source_frame']}],'tile_hashes':[]})
    queries=[]
    def predict(model,gray,owner,root_xy=None):
        queries.append(copy.deepcopy(owner))
        return np.full((64,64),.99,np.float32),(0,0)
    monkeypatch.setattr('prototypes.v30_video_apex.native_body.predict_owned_body',predict)
    result=service.analyze(request)
    assert [o['grain_native'] for o in queries]==[[8.,32.],[18.,32.]]
    assert all(not o['attachment_verified'] for o in queries)
    for row in result['rows']:
        assert row['grain_pose']['grain_native']==[8+row['source_frame']/3,32.]
        assert not row['root_verified'] and row['length_px'] is None
    first_key=result['cache']['pixel_key']
    assert service.analyze(request)['cache']['pixel_hit'] and len(queries)==2
    request.acquisition={'seconds_per_frame':2,'cadence_source':'test acquisition clock'}
    assert service.analyze(request)['cache']['pixel_key']==first_key and len(queries)==2
    paths=export_analysis(result,tmp_path/'export')
    with open(paths['csv']) as f:rows=list(csv.DictReader(f))
    assert rows[1]['grain_x']=='18.0' and rows[1]['grain_pose_origin']=='model'
    assert rows[1]['grain_geometry_identity']==result['grain_geometry']['identity']


def test_unresolved_motion_withholds_model_ownership_but_preserves_human_tip(tmp_path):
    service,request,_=service_fixture(tmp_path);path,artifact=pose_fixture(tmp_path,request)
    artifact['poses'][1].update(pose_status='ambiguous',identity_status='ambiguous')
    path.write_text(json.dumps(artifact))
    result=service.analyze(request)
    row=next(r for r in result['model_without_reviews'] if r['source_frame']==30)
    assert row['tip_xy'] is None and row['length_px'] is None
    assert not row['grain_pose']['usable_for_model_geometry']
    assert all(x['last_verified_absent_frame'] is None and x['first_verified_present_frame'] is None
               for x in result['population']['germination']['emergence_records'])
    entities=[{'uuid':'tip','kind':'task','revision':1,'data':{'movie':'m','owner_uuid':'a','task_type':'review_tip'}},
      {'uuid':'obs-tip','kind':'observation','revision':1,'data':{'task_uuid':'tip','owner_uuid':'a',
        'source_frame':30,'direct_state':'direct_visible','direct_xy':[50,32],
        'tip_source':'explicit_point','review_origin':'human'}}]
    revised=service.analyze(request,review_entities=entities,review_source='human-project')
    human=next(r for r in revised['rows'] if r['source_frame']==30)
    assert human['tip_xy']==[50.,32.] and not human['path_complete']
    assert next(r for r in revised['model_without_reviews'] if r['source_frame']==30)['tip_xy'] is None


def test_truncated_grain_cannot_authorize_a_rim_root(tmp_path):
    _,request,_=service_fixture(tmp_path);path,artifact=pose_fixture(tmp_path,request)
    artifact['poses'][0]['grain_native']=[1,32]
    path.write_text(json.dumps(artifact));plan=load(request)
    pose=pose_index(plan)[('a',0)]
    assert not pose['geometry_complete'] and not pose['usable_for_model_geometry']
