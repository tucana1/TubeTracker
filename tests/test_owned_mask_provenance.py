from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tubetracker.annotation_store import AnnotationStore
from tubetracker.population import resolve_body_presence, resolve_owned_masks
from test_body_presence import mask_record
from test_movie_analysis import service_fixture


@pytest.mark.parametrize('mask_origin,parent_origin', [
    ('model', 'human'), ('model', 'model'), (None, 'model'),
    ('unknown', 'human'), (None, 'unknown')])
def test_model_or_unknown_mask_cannot_supply_presence_or_seed(tmp_path, mask_origin, parent_origin):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    mask = mask_record()
    mask.pop('review_origin')
    if mask_origin is not None:
        mask['review_origin'] = mask_origin
    task = {'movie': 'm', 'task_type': 'body_mask', 'review_origin': parent_origin}
    entities = [{'kind': 'task', 'uuid': 'body-task', 'revision': 1, 'data': task},
                {'kind': 'mask', 'uuid': 'mask-a', 'revision': 1, 'data': mask}]
    for all_frames in (False, True):
        masks, audit = resolve_owned_masks(entities, request, request.owners,
            source=str(tmp_path), image_size=(64, 64), all_movie_frames=all_frames)
        assert not masks and len(audit) == 1
        assert audit[0]['review_origin'] == (mask_origin or parent_origin)
        assert 'human review provenance' in audit[0]['ignored_reason']
    assert not resolve_body_presence(entities, request, request.owners, source=str(tmp_path))[0]


def test_legacy_manual_and_explicit_human_review_keep_distinct_lineage(tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    mask = mask_record()
    mask.pop('review_origin')
    task = {'movie': 'm', 'task_type': 'body_mask'}
    entities = [{'kind': 'task', 'uuid': 'body-task', 'revision': 1, 'data': task},
                {'kind': 'mask', 'uuid': 'mask-a', 'revision': 1, 'data': mask}]
    facts, _ = resolve_body_presence(entities, request, request.owners, source=str(tmp_path))
    assert facts[0]['review_origin_basis'] == 'legacy_manual_mask'
    task['review_origin'] = 'human'
    facts, _ = resolve_body_presence(entities, request, request.owners, source=str(tmp_path))
    assert facts[0]['review_origin_basis'] == 'parent_task'
    # Explicit human adjudication of a proposed object remains human review.
    task['review_origin'], mask['review_origin'] = 'model', 'human'
    facts, _ = resolve_body_presence(entities, request, request.owners, source=str(tmp_path))
    assert facts[0]['review_origin_basis'] == 'explicit_mask'


def test_missing_origin_without_manual_lineage_is_unknown(tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    mask = mask_record()
    mask.pop('review_origin')
    entities = [{'kind': 'task', 'uuid': 'body-task', 'revision': 1, 'data': {'movie': 'm'}},
                {'kind': 'mask', 'uuid': 'mask-a', 'revision': 1, 'data': mask}]
    facts, audit = resolve_body_presence(entities, request, request.owners, source=str(tmp_path))
    assert not facts and audit[0]['review_origin'] == 'unknown'


def test_nonhuman_revision_shadows_historical_human_mask(tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    historical = dict(mask_record(), movie='m', mask_uuid='mask-a', mask_revision=1,
                      project=str(tmp_path))
    (snapshot/'body_masks.json').write_text(json.dumps([historical]))
    request.snapshot = str(snapshot)
    entities = [{'kind':'task','uuid':'body-task','revision':1,
                 'data':{'movie':'m','review_origin':'human','task_type':'body_mask'}},
                {'kind':'mask','uuid':'mask-a','revision':2,
                 'data':dict(mask_record(),review_origin='model')}]
    facts, audit = resolve_body_presence(entities, request, request.owners, source=str(tmp_path))
    assert not facts and audit[0]['revision'] == 2 and audit[0]['review_origin'] == 'model'


def test_future_review_is_available_to_assistance_not_earlier_presence(tmp_path):
    _, request, _ = service_fixture(tmp_path)
    request.owners[0]['identity_verified'] = True
    request.frames = [0]
    entities = [{'kind':'task','uuid':'body-task','revision':1,
                 'data':{'movie':'m','review_origin':'human','task_type':'body_mask'}},
                {'kind':'mask','uuid':'mask-a','revision':1,'data':mask_record()}]
    assert not resolve_body_presence(entities, request, request.owners, source=str(tmp_path))[0]
    seeds, _ = resolve_owned_masks(entities, request, request.owners, source=str(tmp_path),
                                  all_movie_frames=True)
    assert len(seeds) == 1 and seeds[0]['evidence']['frame'] == 30


def test_snapshot_export_cannot_launder_model_origin(tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    with closing(AnnotationStore(project/'annotations.db')) as store:
        for origin in ('human', 'model'):
            task = 'task-'+origin
            store.save('task', task, {'movie':'m','task_type':'body_mask',
                'review_origin':origin,'owner_uuid':'a'})
            mask = dict(mask_record(),task_uuid=task,review_origin=origin)
            store.save('mask','mask-'+origin,mask)
    output = tmp_path/'snapshot'
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/build_v30_snapshot.py'),
               '--project-dir',str(project),'--out',str(output)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    masks = json.loads((output/'body_masks.json').read_text())
    assert [m['mask_uuid'] for m in masks] == ['mask-human']
    assert masks[0]['review_origin'] == 'human'
    rejected = json.loads((output/'mask_provenance_audit.json').read_text())
    assert len(rejected) == 1 and rejected[0]['mask_uuid'] == 'mask-model'
    assert rejected[0]['review_origin'] == 'model'
