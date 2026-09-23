"""A withdrawn review cannot return through cached or snapshotted evidence."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore
from tubetracker.analysis_contracts import snapshot_review_records, store_review_records
from tubetracker.population import census_review_records
from tubetracker.review_semantics import partition_review_entities


def controller(store):
    c = AnnotatorController(store, SimpleNamespace(native_size=(100, 100)))
    c.add_reader('m', c.reader)
    return c


def test_withdraw_restart_recompute_export_and_reconfirm_against_old_snapshot(tmp_path):
    from test_movie_analysis import service_fixture
    from tubetracker.movie_analysis import export_analysis
    service, request, calls = service_fixture(tmp_path)
    db = tmp_path/'annotations.db'
    store = AnnotationStore(db)
    c = controller(store)
    c.load_tasks([{'uuid':'route', 'movie':'m', 'owner_uuid':'a',
                  'task_type':'centerline', 'query_frames':[30]}])
    c.advance()
    c.click(10, 32); c.click(50, 32)
    c.save_path(True)
    original = store.load('obs-route')
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    row = dict(original['data'], obs_uuid='obs-route', obs_revision=original['revision'],
               movie='m', project=str(tmp_path), task_type='centerline')
    (snapshot/'observations.json').write_text(json.dumps([row]))
    (snapshot/'snapshot_manifest.json').write_text('{}')
    request.snapshot = str(snapshot)
    first = service.analyze(request, review_entities=store.entities(), review_source=str(db))
    assert first['rows'][-1]['path_complete'] and first['rows'][-1]['length_px'] == 40
    # A separate point review should survive withdrawal of the full route.
    store.save('task', 'point', {'movie':'m', 'owner_uuid':'a', 'task_type':'review_tip'})
    store.save('observation', 'obs-point', {'task_uuid':'point', 'source_frame':30,
        'owner_uuid':'a', 'direct_state':'direct_visible', 'direct_xy':[50,32],
        'tip_source':'explicit_point', 'source_obs_uuid':'obs-route'})
    c.withdraw_review()
    assert store.load('obs-route')['revision'] == original['revision'] + 1
    assert store.load('obs-route')['data']['path_xy'] == original['data']['path_xy']
    assert store.history('obs-route')[0]['data'] == original['data']
    assert store.load('obs-point')['revision'] == 1
    store.close()
    store = AnnotationStore(db)
    c = controller(store)
    c.open_task('route')
    try:
        withdrawn = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        row = withdrawn['rows'][-1]
        assert row['tip_xy'] == [50.,32.]  # independent point retained
        assert not row['path_complete'] and row['length_px'] is None
        assert any(r['reason'] == 'review withdrawn' for r in withdrawn['ignored_reviews'])
        exported = export_analysis(withdrawn, tmp_path/'export')
        assert json.loads(Path(exported['json']).read_text())['rows'][-1]['length_px'] is None
        # Restored geometry is an editable draft until explicitly confirmed.
        c.save_path(True)
        assert store.load('obs-route')['revision'] == original['revision'] + 2
        resumed = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        assert resumed['rows'][-1]['length_px'] == 40
        # rev14 P1: reviews invalidate inference (and the route stage), not
        # the image evidence -- this body model has no attachment channel.
        assert len(calls) == 1
    finally:
        store.close()


@pytest.mark.parametrize('kind', ['crossing', 'mask', 'region', 'duel', 'ruling'])
def test_task_withdrawal_is_scoped_and_reconfirmation_is_explicit(tmp_path, kind):
    store = AnnotationStore(tmp_path/'annotations.db')
    c = controller(store)
    c.load_tasks([{'uuid':'t', 'movie':'m', 'query_frames':[30], 'task_type':'crossing'}])
    c.advance()
    store.save(kind, 'evidence', {'task_uuid':'t', 'geometry':[[1,2],[3,4]]})
    store.save('task', 'independent', {'movie':'m', 'source_obs_uuid':'evidence'})
    store.save('observation', 'separate-tip', {'task_uuid':'independent', 'source_obs_uuid':'evidence'})
    c.withdraw_review()
    grouped = {}
    for entity in store.entities(): grouped.setdefault(entity['kind'], []).append(entity)
    kept, audit = partition_review_entities(grouped)
    assert {r['uuid'] for r in audit} == {'t','evidence'}
    assert kept['observation'][0]['uuid'] == 'separate-tip'
    c._save_review(kind, 'evidence', store.load('evidence')['data'])
    assert store.load('evidence')['revision'] == 3
    assert store.load('evidence')['data']['review_status'] == 'active'
    assert store.load('t')['data']['review_status'] == 'active'
    store.close()


def test_withdrawn_census_shadows_prior_exhaustive_snapshot(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    c = controller(store)
    c.load_tasks([{'uuid':'census', 'movie':'m', 'query_frames':[30], 'task_type':'census',
        'analysis_census':True, 'review_region':[[0,0],[100,0],[100,100],[0,100]],
        'census_instances':[], 'census_complete':True, 'census_membership_resolved':True}])
    c.advance()
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    old = census_review_records(store.entities())
    assert old[0]['complete']
    (snapshot/'census.json').write_text(json.dumps(old))
    c.withdraw_review()
    assert census_review_records(store.entities(), str(snapshot)) == []
    with pytest.raises(ValueError, match='whole field'):
        c.finish_census()
    c.current['census_membership_resolved'] = True
    c.finish_census()
    revised = census_review_records(store.entities(), str(snapshot))
    assert len(revised) == 1 and revised[0]['complete']
    store.close()


def test_snapshot_excludes_withdrawn_geometry_and_records_audit(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    c = controller(store)
    c.load_tasks([{'uuid':'t', 'movie':'m', 'query_frames':[30], 'task_type':'centerline'}])
    c.advance()
    c.click(10,10); c.click(20,20)
    c.save_path(True)
    store.save('region','r',{'task_uuid':'t','confirmed':True,'kind':'verified_negative',
                            'polygon_xy':[[50,50],[60,50],[60,60]]})
    c.withdraw_review()
    store.close()
    out = tmp_path/'snapshot'
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable,str(root/'scripts/build_v30_snapshot.py'),
        '--project-dir',str(tmp_path),'--out',str(out)],capture_output=True,text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads((out/'observations.json').read_text()) == []
    assert json.loads((out/'regions.json').read_text()) == []
    audit = json.loads((out/'withdrawn_review_records.json').read_text())
    assert {r['uuid'] for r in audit} == {'t','obs-t','r'}
    assert json.loads((out/'workflow_review_records.json').read_text()) == []


def test_atomic_review_save_and_monotonic_revision_after_undo(tmp_path):
    store = AnnotationStore(tmp_path/'annotations.db')
    store.save('observation','o',{'point':[1,2]})
    with pytest.raises(TypeError):
        store.save_many([('observation','o',{'point':[3,4]}),('task','t',{'bad':object()})])
    assert store.load('o')['revision'] == 1 and store.load('t') is None
    store.delete('observation','o')
    assert store.save('observation','o',{'point':[5,6]}) == 2
    store.close()
