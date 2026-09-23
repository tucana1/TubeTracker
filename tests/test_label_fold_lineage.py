"""Real CLI seam: withdrawal must reach both copied labels and their audit."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tubetracker.annotation_store import AnnotationStore
from tubetracker.label_lineage import read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corpus(tmp_path):
    movie = tmp_path/'movie.avi'
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'MJPG'), 1, (32, 32))
    assert writer.isOpened()
    writer.write(np.zeros((32, 32, 3), np.uint8)); writer.release()
    base = tmp_path/'base'; base.mkdir(); (base/'frames').mkdir()
    (base/'manifest.json').write_text(json.dumps({'frames': []}))
    write_csv(base/'labels.csv', [], ['image_name', 'source_frame', 'label_type', 'x', 'y', 'provenance', 'confidence'])
    write_csv(base/'frame_reviews.csv', [], ['image_name', 'source_frame', 'grain_reviewed', 'tip_reviewed'])
    project = tmp_path/'project'; project.mkdir()
    return tmp_path, movie, base, project


def save_point(project, uid='t', *, kind='review_tip', observation=True, **extra):
    store = AnnotationStore(project/'annotations.db')
    store.save('task', uid, {'uuid': uid, 'task_type': kind, 'query_frames': [0],
        'movie': 'test', 'completed': True, 'review_origin': extra.pop('review_origin', 'human'),
        'review_status': 'active', 'review_verdict': 'confirmed'})
    if observation:
        store.save('observation', 'obs-'+uid, dict(task_uuid=uid, direct_state='direct_visible',
            direct_xy=[10, 12], review_status='active', **extra))
    store.close()


def withdraw(project, uid='t'):
    store = AnnotationStore(project/'annotations.db'); store.withdraw_task_review(uid); store.close()


def fold(corpus, name, *, base=None, project=None, ok=True):
    root, movie, original, original_project = corpus
    out = root/name
    command = [sys.executable, str(ROOT/'scripts/fold_human_labels.py'), '--project-dir', str(project or original_project),
        '--movie', str(movie), '--dataset-dir', str(base or original), '--out-dir', str(out),
        '--movie-tag', 'test', '--task-movie', 'test']
    result = subprocess.run(command, text=True, capture_output=True)
    if ok:
        assert result.returncode == 0, result.stdout+result.stderr
    else:
        assert result.returncode != 0
        assert not out.exists(), 'failed fold must not publish a dataset'
    return out, result


def audit(corpus, dataset, *, project=None, ok=True):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/audit_export.py'),
        '--project-dir', str(project or corpus[3]), '--dataset-dir', str(dataset),
        '--movie-tag', 'test', '--task-movie', 'test'], text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stdout+result.stderr
    return json.JSONDecoder().raw_decode(result.stdout)[0]


def test_withdrawal_reconciles_inherited_labels_and_reconfirm_revisions(corpus):
    project = corpus[3]
    save_point(project)
    save_point(project, 'independent')
    before, _ = fold(corpus, 'before')
    assert len(read_csv(before/'labels.csv')) == 1
    assert len(read_csv(before/'label_lineage.csv')) == 2
    assert audit(corpus, before)['eligible_human_tips'] == 1
    withdraw(project)
    audit(corpus, before, ok=False)
    independent, _ = fold(corpus, 'independent', base=before)
    assert len(read_csv(independent/'labels.csv')) == 1
    assert {r['obs_uuid'] for r in read_csv(independent/'label_lineage.csv')} == {'obs-independent'}
    withdraw(project, 'independent')
    empty, _ = fold(corpus, 'empty', base=independent)
    assert read_csv(empty/'labels.csv') == []
    assert read_csv(empty/'frame_reviews.csv')[0]['tip_reviewed'] == '0'
    assert audit(corpus, empty)['eligible_human_tips'] == 0
    fresh, _ = fold(corpus, 'empty-from-clean-base')
    assert read_csv(fresh/'labels.csv') == []
    save_point(project)
    restored, _ = fold(corpus, 'restored', base=empty)
    assert read_csv(restored/'label_lineage.csv')[0]['obs_revision'] == '3'
    assert audit(corpus, restored)['eligible_human_tips'] == 1
    again, _ = fold(corpus, 'again', base=restored)
    assert read_csv(again/'labels.csv') == read_csv(restored/'labels.csv')
    assert read_csv(again/'label_lineage.csv') == read_csv(restored/'label_lineage.csv')


def test_partial_endpoints_missing_observations_and_workflow_are_excluded(corpus):
    p = corpus[3]
    save_point(p, 'partial', kind='centerline', path_xy=[[3, 4], [10, 12]], path_complete=False)
    save_point(p, 'missing', kind='review_path', observation=False)
    save_point(p, 'workflow', review_origin='workflow_test')
    out, _ = fold(corpus, 'excluded')
    assert read_csv(out/'labels.csv') == []
    assert json.loads((out/'manifest.json').read_text())['frames'] == []
    assert audit(corpus, out)['eligible_human_tips'] == 0
    save_point(p, 'explicit', kind='centerline', path_xy=[[3, 4], [10, 12]],
               path_complete=False, tip_source='explicit_point')
    precise, _ = fold(corpus, 'explicit-point')
    assert len(read_csv(precise/'labels.csv')) == 1
    assert audit(corpus, precise)['eligible_human_tips'] == 1


def test_equal_ids_in_other_project_are_independent_licenses(corpus):
    p = corpus[3]; other = corpus[0]/'other'; other.mkdir()
    save_point(p); save_point(other)
    first, _ = fold(corpus, 'first')
    both, _ = fold(corpus, 'both', project=other, base=first)
    assert len(read_csv(both/'labels.csv')) == 1
    assert len(read_csv(both/'label_lineage.csv')) == 2
    withdraw(p)
    remaining, _ = fold(corpus, 'remaining', base=both)
    assert read_csv(remaining/'label_lineage.csv')[0]['source_project'] == str(other.resolve())
    assert len(read_csv(remaining/'labels.csv')) == 1
    assert audit(corpus, remaining, project=other)['eligible_human_tips'] == 1


@pytest.mark.parametrize('with_sidecar', [False, True])
def test_ambiguous_legacy_base_is_refused_without_mutating_it(corpus, with_sidecar):
    save_point(corpus[3])
    old, _ = fold(corpus, 'old')
    lineage = read_csv(old/'label_lineage.csv')
    if with_sidecar:
        fields = ['image_name', 'label_type', 'x', 'y', 'task_uuid', 'obs_uuid', 'obs_revision', 'movie_tag', 'fold_actor']
        write_csv(old/'label_lineage.csv', [{k:r[k] for k in fields} for r in lineage], fields)
    else:
        (old/'label_lineage.csv').unlink()
    before = (old/'labels.csv').read_bytes()
    withdraw(corpus[3])
    _, failed = fold(corpus, 'refused', base=old, ok=False)
    assert 'known clean base' in failed.stderr
    assert (old/'labels.csv').read_bytes() == before
    audit(corpus, old, ok=False)


def test_blank_revision_and_masked_channel_cannot_pass_audit(corpus):
    save_point(corpus[3]); out, _ = fold(corpus, 'out')
    path = out/'label_lineage.csv'; rows = read_csv(path)
    rows[0]['obs_revision'] = ''
    write_csv(path, rows, rows[0].keys())
    assert audit(corpus, out, ok=False)['verified_human_labels'] == 0
    rows[0]['obs_revision'] = '1'; write_csv(path, rows, rows[0].keys())
    path = out/'frame_reviews.csv'; reviews = read_csv(path); reviews[0]['tip_reviewed'] = '0'
    write_csv(path, reviews, reviews[0].keys())
    assert audit(corpus, out, ok=False)['eligible_human_tips'] == 0


def test_owner_grain_has_real_task_revision_without_invented_observation(corpus):
    store = AnnotationStore(corpus[3]/'annotations.db')
    store.save('task', 'owner', {'task_type': 'owner', 'movie': 'test', 'query_frames': [0],
        'completed': True, 'grains': {'A': {'xy': [12, 14]}}}); store.close()
    out, _ = fold(corpus, 'owner')
    row = read_csv(out/'label_lineage.csv')[0]
    assert row['source_kind'] == 'task' and row['source_revision'] == '1'
    assert row['obs_uuid'] == row['obs_revision'] == ''
    assert audit(corpus, out)['verified_human_labels'] == 1


def test_failed_frame_read_does_not_publish_partial_dataset(corpus):
    save_point(corpus[3]); store = AnnotationStore(corpus[3]/'annotations.db')
    task = store.load('t')['data']; task['query_frames'] = [9000]
    store.save('task', 't', task); store.close()
    fold(corpus, 'failed-frame', ok=False)


def test_withdrawal_restores_original_channel_mask_while_preserving_weak_labels(corpus):
    base = corpus[2]
    write_csv(base/'labels.csv', [dict(image_name='test_000000.png', source_frame='0',
        label_type='tip', x='20', y='20', provenance='weak-v1', confidence='.3')],
        ['image_name', 'source_frame', 'label_type', 'x', 'y', 'provenance', 'confidence'])
    write_csv(base/'frame_reviews.csv', [dict(image_name='test_000000.png', source_frame='0',
        grain_reviewed='0', tip_reviewed='0')], ['image_name', 'source_frame', 'grain_reviewed', 'tip_reviewed'])
    save_point(corpus[3]); active, _ = fold(corpus, 'active')
    assert read_csv(active/'frame_reviews.csv')[0]['tip_reviewed'] == '1'
    withdraw(corpus[3]); removed, _ = fold(corpus, 'removed', base=active)
    assert read_csv(removed/'labels.csv')[0]['provenance'] == 'weak-v1'
    assert read_csv(removed/'frame_reviews.csv')[0]['tip_reviewed'] == '0'


def test_frame_filename_collision_cannot_mix_movies(corpus):
    save_point(corpus[3]); first, _ = fold(corpus, 'first-movie')
    writer = cv2.VideoWriter(str(corpus[1]), cv2.VideoWriter_fourcc(*'MJPG'), 1, (32, 32))
    writer.write(np.full((32, 32, 3), 200, np.uint8)); writer.release()
    _, result = fold(corpus, 'wrong-movie', base=first, ok=False)
    assert 'another movie' in result.stderr
