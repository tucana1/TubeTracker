"""rev14 P1: stable logical project identity + deletion tombstones.

The independent review reproduced: a withdrawal of the live FULL observation
left the staged snapshot copy alive as a second project, reinstating the FULL
path and its verified root. These tests pin the expected outcomes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tubetracker.analysis_contracts import (
    AnalysisRequest, project_identity, resolve_review_constraints,
    snapshot_review_records, snapshot_tombstones, store_review_records)
from tubetracker.annotation_store import AnnotationStore


def request():
    return AnalysisRequest("", "m", [30], owners=[{"id": "a"}])


def obs(project, **extra):
    base = dict(movie="m", owner_uuid="a", source_frame=30,
                direct_state="direct_visible", direct_xy=[50., 32.],
                source="review", source_id="obs-1", source_revision=1,
                source_project=project, live_review=True)
    base.update(extra)
    return base


def test_identity_file_survives_backup_staging(tmp_path):
    live = tmp_path / 'live-project'
    staged = tmp_path / 'staged-copy'
    for d in (live, staged):
        d.mkdir()
    (live / 'project-identity.json').write_text(json.dumps({'project_id': 'abc123'}))
    (staged / 'project-identity.json').write_text(json.dumps({'project_id': 'abc123'}))
    assert project_identity(live) == project_identity(staged) == 'project-id:abc123'
    # Without a declaration the path is the identity (genuinely separate).
    other = tmp_path / 'other'
    other.mkdir()
    assert project_identity(other) == str(other.resolve())
    assert project_identity('project-id:abc123') == 'project-id:abc123'


def test_live_withdrawal_shadows_the_staged_snapshot_copy():
    path = [[11., 32.], [30., 32.], [50., 32.]]
    live_withdrawn = obs('project-id:abc', source_revision=2,
                         review_status='withdrawn',
                         path_xy=path, path_complete=True)
    staged_active = obs('project-id:abc', source_revision=1,
                        path_xy=path, path_complete=True)
    constraints, ignored = resolve_review_constraints(
        [staged_active, live_withdrawn], request(), request().owners)
    assert ('a', 30) not in constraints        # nothing survives the withdrawal
    assert any('withdrawn' in str(i.get('reason', '')) for i in ignored)


def test_same_uuid_in_a_different_real_project_stays_independent():
    path = [[11., 32.], [50., 32.]]
    staged_active = obs('project-id:abc', path_xy=path, path_complete=True)
    other_project = obs('project-id:xyz', path_xy=path, path_complete=True,
                        source_revision=2, review_status='withdrawn')
    constraints, _ = resolve_review_constraints(
        [staged_active, other_project], request(), request().owners)
    c = constraints[('a', 30)]
    assert c['path_complete'] is True          # the other project cannot veto
    assert c['verified_root'] is not None


def test_deletion_tombstone_shadows_an_older_snapshot_copy():
    path = [[11., 32.], [50., 32.]]
    staged_active = obs('project-id:abc', path_xy=path, path_complete=True)
    tombstones = [{'uuid': 'obs-1', 'source_project': 'project-id:abc',
                   'deleted_utc': '2099-01-01T00:00:00+00:00'}]
    constraints, ignored = resolve_review_constraints(
        [staged_active], request(), request().owners, tombstones=tombstones)
    assert ('a', 30) not in constraints
    assert any('tombstone' in str(i.get('reason', '')) for i in ignored)


def test_tombstone_only_shadows_its_own_logical_project():
    path = [[11., 32.], [50., 32.]]
    staged_active = obs('project-id:abc', path_xy=path, path_complete=True)
    tombstones = [{'uuid': 'obs-1', 'source_project': 'project-id:other'}]
    constraints, _ = resolve_review_constraints(
        [staged_active], request(), request().owners, tombstones=tombstones)
    assert constraints[('a', 30)]['path_complete'] is True


def test_store_delete_tombstone_is_exported_and_superseded_by_a_new_save(tmp_path):
    store = AnnotationStore(tmp_path / 'annotations.db')
    store.save('observation', 'obs-x', {'movie': 'm', 'owner_uuid': 'a',
                                        'source_frame': 30, 'direct_xy': [11., 11.]},
               actor='tester')
    assert store.tombstones() == []
    store.delete('observation', 'obs-x', actor='tester')
    tombstones = store.tombstones()
    assert [t['uuid'] for t in tombstones] == ['obs-x']
    store.save('observation', 'obs-x', {'movie': 'm', 'owner_uuid': 'a',
                                        'source_frame': 30, 'direct_xy': [44., 44.]},
               actor='tester')
    assert store.tombstones() == []            # the new answer supersedes
    assert store.load('obs-x')['data']['direct_xy'] == [44., 44.]
    store.close()


def test_snapshot_tombstones_are_identity_keyed(tmp_path):
    snap = tmp_path / 'snap'
    snap.mkdir()
    (snap / 'project-identity.json').write_text(json.dumps({'project_id': 'abc'}))
    (snap / 'tombstones.json').write_text(json.dumps([
        {'uuid': 'obs-9', 'project': str(snap), 'deleted_utc': 'x'}]))
    rows = snapshot_tombstones(snap)
    assert rows[0]['uuid'] == 'obs-9'
    assert rows[0]['source_project'] == 'project-id:abc'
