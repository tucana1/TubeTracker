"""Grain inventory and cap training must retain their separate review scopes."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.native_caps import native_cap_loss
from prototypes.v30_video_apex.targets import census_pos_neg_masks, samples_from_snapshot
from tubetracker.annotation_store import AnnotationStore
from tubetracker.review_semantics import census_grain_points


@pytest.mark.parametrize('complete', [False, True])
def test_grain_only_census_has_exactly_zero_cap_loss_and_gradient(complete):
    positive, negative = census_pos_neg_masks(
        64, 64, (0, 0), [], [[16, 16], [48, 48]], (0, 0, 63, 63),
        complete, class_scopes=['grains'])
    assert not positive.any() and not negative.any()
    logits = torch.full((1, 1, 64, 64), 20., requires_grad=True)
    loss, _ = native_cap_loss(
        {'logits': logits, 'offset_xy': torch.zeros(1, 2, 64, 64),
         'logvar': torch.zeros_like(logits)},
        torch.from_numpy(positive)[None, None], torch.from_numpy(negative)[None, None],
        torch.zeros(1, 2, 64, 64), hard_negative_weight=.5)
    loss.backward()
    assert float(loss.detach()) == 0 and torch.count_nonzero(logits.grad) == 0


def test_cap_census_supervises_only_its_class_and_explicit_extent():
    p, n = census_pos_neg_masks(64, 64, (0, 0), [[16, 16]], [[48, 48]],
        (8, 8, 55, 55), True, other_tips_crop=[(32, 32)], class_scopes=['tips'])
    assert p[16, 16] == 1 and n[16, 16] == 0
    assert p[48, 48] == 0 and n[48, 48] == 1  # Grain centre is not a cap.
    assert p[32, 32] == n[32, 32] == 0  # Other known cap remains unknown here.
    assert not p[:8].any() and not n[:8].any()
    assert not (p*n).any()
    # Historical completion with no declared class can retain explicit
    # clicks, but cannot assert cap absence throughout a field.
    old_p, old_n = census_pos_neg_masks(64, 64, (0, 0), [[16, 16]], [],
        (8, 8, 55, 55), True)
    np.testing.assert_array_equal(old_p, p)
    assert not old_n.any()


def test_census_export_and_training_loader_exclude_proposals_and_grain_only_truth(tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    store = AnnotationStore(project/'annotations.db')
    task = {'uuid': 'pending', 'task_type': 'census', 'movie': 'ld',
        'query_frames': [10], 'review_region': [[0,0], [64,0], [64,64], [0,64]],
        'analysis_census': True, 'completed': False, 'census_complete': False,
        'census_grains': [[48,48]], 'census_class_scopes': ['grains'],
        'census_instances': [{'grain_id': 'g', 'xy': [48,48], 'confirmed': False}]}
    for uid, confirmed, complete in [('pending', False, False), ('partial', True, False),
                                      ('grain-only-complete', True, True)]:
        row = copy.deepcopy(task)
        row.update(uuid=uid, completed=complete, census_complete=complete,
                   census_membership_resolved=complete)
        row['census_instances'][0]['confirmed'] = confirmed
        store.save('task', uid, row, actor='test-fixture')
    for uid, scopes, tips in [('caps', ['tips'], [[16,16]]),
                              ('legacy', [], [[16,16]]), ('empty-caps', ['tips'], [])]:
        row = copy.deepcopy(task)
        row.pop('census_instances')
        row.update(uuid=uid, analysis_census=False, completed=True, census_complete=True,
                   census_class_scopes=scopes, census_tips=tips, census_grains=[])
        store.save('task', uid, row, actor='test-fixture')
    store.close()
    movie = tmp_path/'identity-only.mp4'
    movie.write_bytes(b'No decoding is needed for the snapshot/target contract.')
    snapshot = tmp_path/'snapshot'
    builder = Path(__file__).resolve().parents[1]/'scripts/build_v30_snapshot.py'
    run = subprocess.run([sys.executable, str(builder), '--project-dir', str(project),
        '--movie', f'ld={movie}', '--out', str(snapshot)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    census = {c['task_uuid']: c for c in json.loads((snapshot/'census.json').read_text())}
    assert census['pending']['grains'] == []
    assert census['pending']['instances'][0]['confirmed'] is False
    assert census['partial']['grains'] == census['grain-only-complete']['grains'] == [[48.,48.]]
    manifest = json.loads((snapshot/'snapshot_manifest.json').read_text())
    assert manifest['n_census_confirmed_grains'] == 2
    assert manifest['n_census_unconfirmed_instances'] == 1
    samples = {s.task_uuid:s for s in samples_from_snapshot(str(snapshot))}
    assert set(samples) == {'caps', 'legacy', 'empty-caps'}
    assert samples['caps'].census_class_scopes == ['tips']
    assert samples['caps'].census_complete and samples['empty-caps'].census_complete
    assert not samples['legacy'].census_complete


def test_instance_aware_empty_census_cannot_reuse_stale_grain_points():
    assert census_grain_points({'grains': [[1,2]], 'instances': [],
                                'instance_review_required': True}) == []
    assert census_grain_points({'grains': [[1,2]]}) == [[1.,2.]]
