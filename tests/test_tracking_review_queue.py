import copy
import json
import sqlite3

import cv2
import numpy as np
import pytest

from scripts.build_tracking_review_queue import (build_manifest, apply_manifest,
    file_hash, read_entities)
from tubetracker.analysis_contracts import stable_hash
from tubetracker.annotation_store import AnnotationStore


@pytest.fixture
def fixture(tmp_path):
    movie=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
    assert writer.isOpened()
    for i in range(16): writer.write(np.full((64,64,3),80+i,np.uint8))
    writer.release()
    project=tmp_path/'project';project.mkdir()
    (project/'project-identity.json').write_text(json.dumps({'project_id':'fixture-project'}))
    store=AnnotationStore(project/'annotations.db')
    store.save('session','analysis-configuration',{'verification_only':False})
    store.save('task','completed-history',{'task_type':'centerline','movie':'m',
        'owner_uuid':'grain','query_frames':[10],'completed':True,'_drawing_draft':{'path_xy':[[1,2],[3,4]]}})
    store.save('observation','old-answer',{'task_uuid':'completed-history','path_xy':[[1,2],[3,4]]})
    store.close()
    owner={'id':'grain','movie':'m','grain_native':[32,32],'grain_radius_px':5,
           'identity_verified':True,'identified_at_frame':12}
    analysis={'movie_id':'m','movie_path':str(movie),'movie_sha256':file_hash(movie),
        'owners':[owner],'request':{'snapshot':''},'model_without_reviews':[
            {'owner_id':'grain','source_frame':f,'grain_pose':{'grain_native':[32,32]}}
            for f in [0,2,4,6]]}
    a=tmp_path/'analysis.json';a.write_text(json.dumps(analysis))
    plan=[{'owner_id':'grain','frame':f,'task_type':kind,'role':role,'roi':[0,0,64,64]}
          for f,kind,role in [(0,'grain_identity','ownership_review'),(2,'review_tip','development'),
                              (4,'centerline','validation'),(6,'review_tip','validation')]]
    return project,a,plan


def test_dry_run_and_publication_never_save_answers_or_reset_completed_work(fixture):
    project,a,plan=fixture
    before=read_entities(project)
    manifest=build_manifest(project,a,a,plan=plan)
    assert before==read_entities(project) and manifest['count']==4
    assert all(t['completed'] is False and t['suppress_analysis_overlays'] for t in manifest['tasks'])
    identity=manifest['tasks'][0]
    assert identity['owner_choices']==[] and identity['reference_owners'][0]['id']=='grain'
    assert 'target_xy' not in identity
    assert all(0<=f<16 for t in manifest['tasks'] for f in t['context_frames'])
    receipt=apply_manifest(manifest)
    assert receipt['inserted']==4
    after=read_entities(project)
    assert [e for e in after if e['uuid'] in {r['uuid'] for r in before}]==before
    assert [e for e in after if e['kind'] not in ('task','session')]==[e for e in before if e['kind'] not in ('task','session')]
    assert apply_manifest(manifest)['inserted']==0
    repeated=build_manifest(project,a,a,plan=plan)
    assert repeated['count']==0 and len(repeated['skipped'])==4
    assert after==read_entities(project)


def test_stale_project_and_source_change_refuse_before_task_writes(fixture):
    project,a,plan=fixture
    manifest=build_manifest(project,a,a,plan=plan)
    store=AnnotationStore(project/'annotations.db');store.save('observation','new-human',{'tip_xy':[5,5]});store.close()
    with pytest.raises(ValueError,match='project changed'):
        apply_manifest(manifest)
    manifest=build_manifest(project,a,a,plan=plan)
    a.write_text(a.read_text()+'\n')
    with pytest.raises(ValueError,match='analysis changed'):
        apply_manifest(manifest)
    assert not any(e['uuid'].startswith('tracking-review-') for e in read_entities(project))


def test_verification_tasks_cannot_be_inserted_into_a_human_project(fixture):
    project,a,plan=fixture
    manifest=build_manifest(project,a,a,plan=plan,verification_only=True)
    with pytest.raises(ValueError,match='matching project modes'):
        apply_manifest(manifest)
    store=AnnotationStore(project/'annotations.db');store.save('session','analysis-configuration',{'verification_only':True});store.close()
    manifest=build_manifest(project,a,a,plan=plan,verification_only=True)
    assert all(t['review_origin']=='workflow_test' for t in manifest['tasks'])
    assert apply_manifest(manifest)['inserted']==4


def test_duplicate_unsupported_or_out_of_range_requests_fail_without_writes(fixture):
    project,a,plan=fixture
    before=stable_hash(read_entities(project))
    for broken in [plan+[plan[0]], [dict(plan[0],frame=16)], [dict(plan[0],roi=[0,0,65,64])],
                   [dict(plan[0],owner_id='unknown')], [dict(plan[1],frame=7)]]:
        with pytest.raises(ValueError):build_manifest(project,a,a,plan=broken)
    assert stable_hash(read_entities(project))==before


def test_atomic_create_only_cannot_overwrite_a_concurrently_created_task(fixture):
    project,_,_=fixture
    store=AnnotationStore(project/'annotations.db')
    before=store.load('completed-history')
    with pytest.raises(sqlite3.IntegrityError):
        store.save_many([('task','new-task',{'completed':False}),
                         ('task','completed-history',{'completed':False})],create_only=True)
    assert store.load('new-task') is None and store.load('completed-history')==before
    store.close()


def test_unanswered_result_task_is_adopted_without_a_duplicate_or_model_guide(fixture):
    project,a,plan=fixture
    store=AnnotationStore(project/'annotations.db')
    old={'task_type':'review_tip','movie':'m','owner_uuid':'grain','query_frames':[2],
         'completed':False,'guide_path':[[32,32],[35,35]],'draft_xy':None,
         '_drawing_draft':{'path_xy':[],'region_xy':[],'lanes':{}}}
    revision=store.save('task','unanswered-result',old);store.close()
    manifest=build_manifest(project,a,a,plan=plan)
    adopted=next(t for t in manifest['tasks'] if t['uuid']=='unanswered-result')
    assert 'guide_path' not in adopted and adopted['annotation_role']=='development'
    assert manifest['adoptions']['unanswered-result']['expected_revision']==revision
    result=apply_manifest(manifest)
    assert result['inserted']==3 and result['adopted']==1
    assert apply_manifest(manifest)['inserted']==0
    store=AnnotationStore(project/'annotations.db')
    assert store.history('unanswered-result')[0]['data']==old
    assert store.load('unanswered-result')['revision']==revision+1
    store.close()


def test_conditional_batch_rolls_back_if_any_answer_changed(fixture):
    project,_,_=fixture
    store=AnnotationStore(project/'annotations.db')
    before=store.load('completed-history')
    with pytest.raises(ValueError,match='entity changed'):
        store.save_many([('task','new-task',{}),('task','completed-history',{})],
            expected_revisions={'new-task':None,'completed-history':before['revision']+1})
    assert store.load('new-task') is None and store.load('completed-history')==before
    store.close()
