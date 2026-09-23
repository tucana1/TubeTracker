import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tubetracker.analysis_contracts import AnalysisRequest, project_identity
from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore
from tubetracker.grain_identity import grain_identity_reviews, identity_review_hash


class Reader:
    native_size = (100, 100)

    def __len__(self):
        return 100


def task(uid='identity', origin='human'):
    return {'uuid': uid, 'task_type': 'grain_identity', 'movie': 'm', 'owner_uuid': 'grain',
            'query_frames': [30], 'reference_frame': 60, 'review_origin': origin,
            'completed': False}


@pytest.fixture
def rig(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    c = AnnotatorController(store, Reader(), actor='reviewer')
    c.add_reader('m', c.reader)
    c.load_tasks([task()]); c.open_task('identity')
    yield c, store
    store.close()


def resolve(store, *, snapshot='', tombstones=()):
    request = AnalysisRequest('', 'm', [0, 90], snapshot=snapshot)
    return grain_identity_reviews(store.entities(), request, [{'id': 'grain'}],
                                  source=str(store.path), tombstones=tombstones)


def test_grain_click_is_only_a_draft_then_saves_typed_identity(rig):
    c, store = rig
    c.click(20, 21)
    assert store.entities('grain_identity') == store.entities('observation') == []
    c.open_task('identity')
    assert c._region_pts == [[20., 21.]]
    with pytest.raises(ValueError, match='not tube observations'):
        c.save_apex(22, 23)
    c.save_grain_identity()
    answer = store.entities('grain_identity')[0]['data']
    assert answer['grain_native'] == [20., 21.]
    assert not {'direct_xy', 'path_xy', 'root_xy', 'no_tube'} & set(answer)
    assert store.entities('observation') == []
    rows, audit = resolve(store)
    assert len(rows) == 1 and not audit
    assert rows[0]['source_frame'] == 30  # anchor need not be in the analysis schedule


def test_context_is_view_only_and_undo_walks_grain_revisions(rig):
    c, store = rig
    c.step_frame(30)
    c.click(20, 21)
    with pytest.raises(ValueError, match='task frame'):
        c.save_grain_identity()
    c.back_to_task_frame()
    for x in [20, 25, 30]:
        c.click(x, 21); c.save_grain_identity()
    c.undo_saved_correction(); assert resolve(store)[0][0]['grain_native'] == [25., 21.]
    c.undo_saved_correction(); assert resolve(store)[0][0]['grain_native'] == [20., 21.]
    c.undo_saved_correction(); assert resolve(store)[0] == []
    assert store.tombstones()[0]['kind'] == 'grain_identity'


def test_snapshot_copy_cannot_resurrect_a_withdrawn_or_deleted_anchor(rig, tmp_path):
    c, store = rig
    c.click(20, 21); c.save_grain_identity()
    entity = store.entities('grain_identity')[0]
    snap = tmp_path/'snapshot'; snap.mkdir()
    (snap/'grain_identities.json').write_text(json.dumps([dict(entity, project=project_identity(store.path))]))
    before = identity_review_hash(resolve(store, snapshot=str(snap))[0])
    c.withdraw_review()
    assert resolve(store, snapshot=str(snap))[0] == []
    c.click(25, 21); c.save_grain_identity()
    changed = resolve(store, snapshot=str(snap))[0]
    assert len(changed) == 1 and identity_review_hash(changed) != before
    store.delete('grain_identity', entity['uuid'])
    deletion = store.tombstones()
    assert resolve(store, snapshot=str(snap), tombstones=deletion)[0] == []
    c.click(24, 21); c.save_grain_identity()
    assert resolve(store, snapshot=str(snap), tombstones=deletion)[0][0]['grain_native'] == [24.,21.]


def test_human_ambiguity_overrides_old_confirmation_and_workflow_does_not_supply_truth(rig):
    c, store = rig
    c.click(20, 21); c.save_grain_identity()
    c.save_grain_identity('ambiguous')
    rows, _ = resolve(store)
    assert rows[0]['identity_state'] == 'ambiguous' and rows[0]['grain_native'] is None
    c.actor = 'workflow-test'
    c.click(22, 21); c.save_grain_identity()
    rows, audit = resolve(store)
    assert not rows and 'human provenance' in audit[0]['reason']


def test_independent_conflicting_centres_are_not_silently_averaged(rig):
    c, store = rig
    c.click(20, 21); c.save_grain_identity()
    c.load_tasks([task('independent')]); c.open_task('independent')
    c.click(40, 21); c.save_grain_identity()
    rows, _ = resolve(store)
    assert len(rows) == 1 and rows[0]['identity_state'] == 'ambiguous'
    assert len(rows[0]['sources']) == 2


def test_snapshot_builder_keeps_identity_separate_from_tip_training(rig, tmp_path):
    c, store = rig
    c.click(20, 21); c.save_grain_identity()
    out = tmp_path/'fold'
    script = Path(__file__).resolve().parents[1]/'scripts/build_v30_snapshot.py'
    result = subprocess.run([sys.executable, str(script), '--project-dir', str(store.path.parent),
                             '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    identities = json.loads((out/'grain_identities.json').read_text())
    assert len(identities) == 1 and identities[0]['data']['grain_native'] == [20., 21.]
    assert json.loads((out/'observations.json').read_text()) == []
    manifest = json.loads((out/'snapshot_manifest.json').read_text())
    assert manifest['n_grain_identities'] == 1
    assert manifest['content_sha256']['grain_identities.json']
