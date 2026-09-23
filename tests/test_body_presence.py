import copy
from contextlib import closing
import json

import numpy as np
import pytest

from prototypes.v30_video_apex.targets import encode_mask_raster
from tubetracker.annotation_store import AnnotationStore
from tubetracker.analysis_workbench import AnalysisSession
from tubetracker.population import resolve_body_presence, reviewed_mask_positive_pixels
from test_movie_analysis import service_fixture


def mask_record():
    paint = np.zeros((64, 64), bool)
    paint[26:30, 36:41] = True
    return {'task_uuid': 'body-task', 'movie_uuid': 'm', 'owner_uuid': 'a',
            'source_frame': 30, 'mask_raster': encode_mask_raster(paint),
            'complete': False, 'review_origin': 'human', 'review_status': 'active'}


def test_owned_mask_updates_timing_and_export_identity_without_inventing_geometry(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    initial = service.analyze(request)
    db = tmp_path/'project'/'annotations.db'
    db.parent.mkdir()
    with closing(AnnotationStore(db)) as store:
        store.save('task', 'body-task', {'movie': 'm', 'task_type': 'body_mask', 'owner_uuid': 'a'})
        store.save('mask', 'mask-a', mask_record())
        result = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        summary = result['population']['grain_summaries'][0]
        assert summary['onset_by_frame'] == 30 and summary['timing_status'] == 'left_censored'
        assert summary['biological_class'] is None and summary['complete_measurement_count'] == 0
        assert result['rows'] == initial['rows']  # no tip, root, solver constraint or complete path added
        assert result['cache']['pixel_key'] == initial['cache']['pixel_key']
        assert result['cache']['inference_key'] == initial['cache']['inference_key']
        assert result['cache']['population_key'] != initial['cache']['population_key']
        fact = result['inputs']['population']['body_presence'][0]
        assert fact['source_id'] == 'mask-a' and fact['positive_pixels'] == 20
        # An exported historical copy must not survive a live withdrawal.
        snapshot = tmp_path/'snapshot'; snapshot.mkdir()
        (snapshot/'observations.json').write_text('[]')
        (snapshot/'snapshot_manifest.json').write_text('{}')
        historical = dict(mask_record(), movie='m', mask_uuid='mask-a', mask_revision=1, project=str(db.parent))
        (snapshot/'body_masks.json').write_text(json.dumps([historical]))
        request.snapshot = str(snapshot)
        store.delete('mask', 'mask-a')
        removed = service.analyze(request, review_entities=store.entities(), review_source=str(db),
                                  review_tombstones=store.tombstones())
        assert removed['population']['grain_summaries'][0]['timing_status'] == 'unknown'
        assert 'tombstone' in removed['population']['body_presence_audit'][0]['ignored_reason']
        assert len(calls) == 1


@pytest.mark.parametrize('change', ['workflow_parent', 'withdrawn_parent', 'unknown_paint',
                                   'unverified_owner', 'wrong_movie', 'wrong_frame'])
def test_ineligible_mask_cannot_narrow_emergence(change, tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = change != 'unverified_owner'
    task = {'movie': 'm', 'task_type': 'body_mask', 'owner_uuid': 'a'}
    mask = mask_record()
    if change == 'workflow_parent': task['review_origin'] = 'workflow_test'
    if change == 'withdrawn_parent': task['review_status'] = 'withdrawn'
    if change == 'unknown_paint': mask['mask_unknown_raster'] = copy.deepcopy(mask['mask_raster'])
    if change == 'wrong_movie': mask['movie_uuid'] = 'other'
    if change == 'wrong_frame': mask['source_frame'] = 4
    entities = [{'kind': 'task', 'uuid': 'body-task', 'revision': 1, 'data': task},
                {'kind': 'mask', 'uuid': 'mask-a', 'revision': 1, 'data': mask}]
    facts, _ = resolve_body_presence(entities, request, request.owners, source=str(tmp_path), image_size=(64,64))
    assert facts == []


def test_same_uuid_in_another_project_is_independent_and_unknown_paint_is_subtracted(tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    snapshot = tmp_path/'snapshot'; snapshot.mkdir()
    other_project = tmp_path/'other-project'; other_project.mkdir()
    mask = dict(mask_record(), mask_uuid='shared-id', mask_revision=1, movie='m', project=str(other_project))
    (snapshot/'body_masks.json').write_text(json.dumps([mask]))
    request.snapshot = str(snapshot)
    facts, _ = resolve_body_presence([], request, request.owners, source=str(tmp_path/'live'),
                                    tombstones=[{'uuid':'shared-id','source':str(tmp_path/'live')}])
    assert len(facts) == 1
    unknown = np.zeros((64,64), bool); unknown[26:28,36:41] = True
    mask['mask_unknown_raster'] = encode_mask_raster(unknown)
    assert reviewed_mask_positive_pixels(mask) == 10
    assert reviewed_mask_positive_pixels(mask, image_size=(32,32)) == 0


def test_workbench_marks_mask_and_parent_withdrawal_stale_but_ignores_drafts(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    with closing(AnnotationStore(tmp_path/'annotations.db')) as store:
        session = AnalysisSession(store, {'cap_checkpoint':service.cap_checkpoint,
            'body_checkpoint':service.body_checkpoint,'request':request.to_dict()}, service=service)
        session.recompute()
        task = {'movie':'m','owner_uuid':'a','task_type':'body_mask'}
        store.save('task','body-task',task)
        store.save('mask','mask-a',mask_record())
        assert session.stale
        session.recompute()
        assert not session.stale
        assert session.result['population']['grain_summaries'][0]['onset_by_frame'] == 30
        store.save('task','body-task',dict(task,_drawing_draft={'path_xy':[[1,2],[3,4]]}))
        assert not session.stale
        store.save('task','body-task',dict(task,review_status='withdrawn'))
        assert session.stale
        session.recompute()
        assert session.result['population']['grain_summaries'][0]['timing_status'] == 'unknown'
        store.delete('mask','mask-a')
        assert session.stale
        assert len(calls) == 1
