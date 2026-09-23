"""Historical owner edits retain geometry but cannot retain wrong roots."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tubetracker.annotation_corrections import (
    apply_project_correction, correction_change, reassign_tube_record)
from tubetracker.annotation_store import AnnotationStore
from tubetracker.analysis_contracts import AnalysisRequest, resolve_review_constraints, snapshot_review_records
from tubetracker.review_semantics import precise_tip


def route(full=True):
    return {'uuid': 'obs-route', 'kind': 'observation', 'revision': 2,
            'data': {'owner_uuid': 'right', 'movie': 'm', 'task_uuid': 'route',
                     'source_frame': 30, 'review_origin': 'human',
                     'task_type': 'centerline', 'direct_state': 'direct_visible',
                     'direct_xy': [50., 32.], 'path_xy': [[10., 32.], [50., 32.]],
                     'path_complete': full, 'tip_source': 'reviewed_full_path' if full else 'unobserved'}}


def reassign(entity):
    return reassign_tube_record(entity, 'middle', project='/review/project',
        correction={'id': 'swap', 'basis': 'investigator confirmed historical owner correction'})


def test_full_owner_correction_preserves_apex_but_cannot_certify_root_or_full_length():
    before = route()
    after = reassign(before)
    assert before == route()  # pure planning
    assert after['owner_uuid'] == 'middle'
    assert after['path_xy'] == before['data']['path_xy']
    assert precise_tip(after) == [50., 32.]
    assert not after['path_complete'] and after['root_review_required']
    request = AnalysisRequest(movie_path='movie.mp4', movie_id='m', frames=[30])
    record = dict(after, source_id='obs-route', source_revision=3)
    constraints, ignored = resolve_review_constraints([record], request, [{'id': 'middle'}])
    assert not ignored
    constraint = constraints['middle', 30]
    assert constraint['tip_xy'] == [50., 32.]
    assert constraint['verified_root'] is None and not constraint['path_complete']


def test_partial_owner_correction_never_creates_an_apex():
    after = reassign(route(False))
    assert precise_tip(after) is None
    assert 'tip_review_basis' not in after
    assert after['path_xy'] == route(False)['data']['path_xy']


@pytest.mark.parametrize('mutation', [
    {'direct_xy': [51., 32.]},
    {'tip_review_basis': {}},
    {'tip_review_basis': {'kind': 'preserved_reviewed_full_path_apex', 'path_complete': True}},
    {'review_status': 'withdrawn'},
])
def test_preserved_apex_requires_original_review_provenance(mutation):
    after = reassign(route())
    after.update(mutation)
    assert precise_tip(after) is None


@pytest.mark.parametrize('kind', ['grain_identity', 'task'])
def test_physical_grain_records_cannot_be_swapped_as_tubes(kind):
    entity = route(); entity['kind'] = kind
    with pytest.raises(ValueError, match='physical grain'):
        reassign(entity)


def test_owned_absence_cannot_be_swapped_as_a_tube():
    entity = route(); entity['data']['direct_state'] = 'no_tube_visible'
    with pytest.raises(ValueError, match='absence'):
        reassign(entity)


def test_correction_is_atomic_revision_checked_and_retry_cannot_swap_back(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    try:
        for uid, owner in [('a', 'right'), ('b', 'middle')]:
            store.save('observation', uid, {'owner_uuid': owner, 'direct_xy': [1, 2]})
        before = store.entities()
        changes = [correction_change(e, dict(e['data'], owner_uuid=('middle' if e['data']['owner_uuid']=='right' else 'right')),
                                     reason='confirmed swap') for e in before]
        # A concurrent human edit prevents every planned edit, even those
        # appearing earlier in the transaction.
        store.save('observation', 'b', {'owner_uuid': 'middle', 'direct_xy': [3, 4]})
        with pytest.raises(ValueError, match='changed after planning'):
            apply_project_correction(store, changes, correction_id='swap', actor='test')
        assert store.load('a')['revision'] == 1
        changes = [correction_change(e, dict(e['data'], owner_uuid=('middle' if e['data']['owner_uuid']=='right' else 'right')),
                                     reason='confirmed swap') for e in store.entities()]
        result = apply_project_correction(store, changes, correction_id='swap', actor='test')
        assert not result['already_applied']
        assert store.load('a')['data']['owner_uuid'] == 'middle'
        assert store.load('b')['data']['owner_uuid'] == 'right'
        assert len(store.history('a')) == 2
        # An independent subsequent review must survive replay of the receipt.
        new = dict(store.load('a')['data'], direct_xy=[8, 9])
        store.save('observation', 'a', new)
        assert apply_project_correction(store, changes, correction_id='swap', actor='test')['already_applied']
        assert store.load('a')['data'] == new
        altered = copy.deepcopy(changes); altered[0]['reason'] = 'different correction'
        with pytest.raises(ValueError, match='different plan'):
            apply_project_correction(store, altered, correction_id='swap', actor='test')
    finally:
        store.close()


def test_snapshot_keeps_corrected_tip_provenance_and_withholds_full_claim(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    store.save('task', 'route', {'movie': 'm', 'task_type': 'centerline',
                              'owner_uuid': 'middle', 'query_frames': [30], 'annotation_role': 'validation'})
    store.save('observation', 'obs-route', reassign(route()))
    store.close()
    out = tmp_path/'snapshot'
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'scripts/build_v30_snapshot.py'),
        '--project-dir', str(tmp_path), '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    record = snapshot_review_records(out)[0]
    assert record['owner_uuid'] == 'middle'
    assert record['label_corrections'][0]['source_revision'] == 2
    assert record['root_review_required'] and not record['path_complete']
    assert record['annotation_role'] == 'validation'
    assert precise_tip(record) == [50., 32.]
