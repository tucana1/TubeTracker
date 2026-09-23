"""rev16 (sparse germination) regression tests: WO1 role/exposure discipline."""
import importlib.util
import json
from pathlib import Path


def _trainer():
    spec = importlib.util.spec_from_file_location(
        'rev16_train_native_body', 'scripts/train_native_body.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_sampler_excludes_validation_and_test_roles():
    trainer = _trainer()
    owners = {'g1': {'id': 'g1', 'grain_native': [0.0, 0.0]}}
    obs = [
        {'obs_uuid': 'v-train', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 1,
         'direct_state': 'direct_visible', 'direct_xy': [5, 5],
         'annotation_role': 'training'},
        {'obs_uuid': 'v-val', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 2,
         'direct_state': 'direct_visible', 'direct_xy': [5, 5],
         'annotation_role': 'validation'},
        {'obs_uuid': 'a-test', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 3,
         'direct_state': 'no_tube_visible', 'annotation_role': 'test'},
    ]
    definitions = trainer.reviewed_presence_definitions(obs, owners, {'ld': {}})
    by_id = {s['_presence_uuid']: s for _, _, _, s in definitions}
    assert by_id['v-train']['_presence_role'] == 'training'
    assert by_id['v-val']['_presence_role'] == 'validation'
    assert by_id['a-test']['_presence_role'] == 'test'
    # The fit-time training split keeps only training/development.
    train_roles = ('training', 'development')
    train = [u for u, s in by_id.items() if s['_presence_role'] in train_roles]
    held = [u for u, s in by_id.items() if s['_presence_role'] not in train_roles]
    assert train == ['v-train']
    assert sorted(held) == ['a-test', 'v-val']


def test_snap34_panel_presence_role_split():
    doc = json.load(open('runs/prototypes/v30/rev15_body_panel_snap34/panel.json'))
    pres = [c for c in doc['cases'] if c['kind'] == 'owned_presence'
            and c.get('presence') in ('visible', 'absent')]
    assert len(pres) == 33, 'all 33 scalar cases stay in eval'
    train = [c for c in pres if c.get('presence_role', 'training') in ('training', 'development')]
    held = [c for c in pres if c.get('presence_role', 'training') not in ('training', 'development')]
    assert len(held) >= 1, 'held-out scalar cases must exist'
    assert len(train) + len(held) == 33
    assert all(c['loss_pixel_count'] == 0 if 'loss_pixel_count' in c else c.get('positive_pixels', 0) == 0
               for c in pres)


def test_posed_panel_uses_runtime_poses_with_provenance():
    doc = json.load(open('runs/prototypes/v30/rev16_body_panel_posed/panel.json'))
    assert doc.get('grain_poses_plans'), 'poses plan must be hashed into the panel'
    pres = [c for c in doc['cases'] if c['kind'] == 'owned_presence']
    by_src = {}
    for c in pres:
        by_src.setdefault(c['query_pose_source'], []).append(c)
    assert 'runtime_pose' in by_src
    cf70 = [c for c in by_src['runtime_pose']
            if 'cf70' in c['owner'] and c['frame'] in (0, 300, 6000, 9000)]
    assert len(cf70) == 4
    # Audit-measured mismatch was ~10.5-11.5px; posed queries must show it.
    for c in cf70:
        dx, dy = c['query_pose_translation_px']
        assert 10.0 <= (dx * dx + dy * dy) ** 0.5 <= 12.0
        assert c['query_pose_reference_frame'] == 51120
        assert c['query_pose_correlation'] is not None
    # Fallbacks stay explicit, never silent.
    assert all(c['query_pose_source'] == 'registry_fallback'
               for c in by_src.get('registry_fallback', []))


def test_solver_model_absent_admission_and_scoring():
    from prototypes.v30_video_apex import state_solver as solver
    frames = [0, 1]
    present = {'candidate_id':'c:1','state':'present','tip_xy':[10.0,10.0],'cap_id':'',
               'cap_probability':0.9,'route_probability':0.9,'path_complete':True,
               'root_verified':False,'observed_partial_path_xy':[[10.0,10.0]],
               'presence_logit':5.0}
    absentish = dict(present, candidate_id='c:2', presence_logit=-4.0,
                     cap_probability=0.6, route_probability=0.6)
    cands = {'g1': {0: [dict(present)], 1: [dict(absentish)]}}
    review = {'constraints': [], 'reserved_routes': []}
    # Disabled by default: no model_absent choice, behaviour preserved.
    out = solver.solve_joint_states(cands, frames, config=solver.SolverConfig())
    states = {(r['owner_id'], r['source_frame']): r['state'] for r in out['rows']}
    assert states[('g1',0)] == 'present' and states[('g1',1)] == 'present'
    assert not any(r['state'] == 'model_absent' for r in out['rows'])
    # Enabled with a threshold above the frame-1 logit: frame 1 admits the
    # scored alternative and its absence evidence (-(-4)=4) beats the route.
    out2 = solver.solve_joint_states(
        {'g1': {0: [dict(present)], 1: [dict(absentish)]}}, frames,
        config=solver.SolverConfig(presence_absent_logit=0.0))
    states2 = {(r['owner_id'], r['source_frame']): r['state'] for r in out2['rows']}
    assert states2[('g1',1)] == 'model_absent', states2
    assert states2[('g1',0)] == 'present', states2
    row1 = [r for r in out2['rows'] if r['source_frame']==1 and r['state']=='model_absent'][0]
    assert row1['tip_xy'] is None and row1['current_path_xy'] == []
    # verified_absent is never emitted by the model path.
    assert not any(r['state'] == 'verified_absent' for r in out2['rows'])


def test_solver_model_absent_needs_evidence():
    from prototypes.v30_video_apex import state_solver as solver
    frames = [0]
    no_evidence = {'candidate_id':'c:1','state':'present','tip_xy':[10.0,10.0],'cap_id':'',
                   'cap_probability':0.9,'route_probability':0.9,'path_complete':True,
                   'root_verified':False,'observed_partial_path_xy':[[10.0,10.0]]}
    out = solver.solve_joint_states(
        {'g1': {0: [dict(no_evidence)]}}, frames,
        config=solver.SolverConfig(presence_absent_logit=0.0))
    assert not any(r['state'] == 'model_absent' for r in out['rows'])


def test_germination_episode_requires_justification():
    import pytest
    from tubetracker.annotation_tasks import germination_episode
    base = dict(grain_id='g', movie_id='m', movie_content_hash='h',
                window_start=0, window_end=9000, unresolved_error='e',
                why_existing_insufficient='w', cheapest_answer='c',
                consuming_loss_or_eval='l', role='training',
                before_after_comparison='b', stratum='bare-rim-control')
    task = germination_episode(**base)
    assert task['task_type'] == 'germination_event'
    assert task['role'] == 'training'
    assert task['episode_justification']['consuming_loss_or_eval'] == 'l'
    bad = dict(base, consuming_loss_or_eval='')
    with pytest.raises(ValueError):
        germination_episode(**bad)


def test_germination_event_save_roundtrip(tmp_path):
    from tubetracker.annotation_app import AnnotatorController
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.annotation_store import AnnotationStore
    movie = '/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4'
    store = AnnotationStore(tmp_path / 'p.db')
    reader = FrameReader(movie)
    try:
        c = AnnotatorController(store, reader, viewer=None, actor='test')
        c.load_tasks([{'uuid': 'ev1', 'owner_uuid': 'o1', 'query_frames': [300],
                       'task_type': 'germination_event', 'completed': False,
                       'movie': 'ld', 'source_start': 0, 'source_end': 9000}])
        assert c.advance()['uuid'] == 'ev1'
        assert c.click(100.0, 100.0)['uuid'] == 'ev1'  # clicks never mark
        assert store.load('event-ev1') is None
        c._consulted = [0, 300, 6000]
        uuid = c.save_germination_event('emerged_within', 300, 6000)
        assert uuid == 'event-ev1'
        row = store.load('event-ev1')
        assert row['data']['verdict'] == 'emerged_within'
        assert row['data']['review_origin'] == 'human'
        assert row['data']['consulted_frames'] == [0, 300, 6000]
        import pytest as _pt
        with _pt.raises(ValueError):
            c.save_germination_event('emerged_within', 6000, 300)
    finally:
        reader.close()


def test_germination_correction_reaches_population_export(tmp_path):
    """Correction persistence (WO4): a saved human verdict flows through
    entities -> population -> export columns, and survives a reopen
    (store reload) with identical values."""
    from tubetracker.annotation_store import AnnotationStore
    from tubetracker.population import (
        analysis_population, germination_event_records)
    store = AnnotationStore(tmp_path / 'p.db')
    store.save('germination_event', 'gev1', {
        'owner_uuid': 'o1', 'movie_id': 'ld', 'verdict': 'emerged_within',
        'last_absent_frame': 300, 'first_visible_frame': 6000,
        'review_origin': 'human', 'review_status': 'reviewed',
        'task_uuid': 'evtask1'})
    owners = [{'id': 'o1', 'grain_native': [10.0, 32.0],
               'grain_radius_px': 13.0, 'identified_at_frame': 0}]

    def summarize(entities):
        pop = analysis_population(
            owners, [], {}, [], movie='ld',
            germination_events=germination_event_records(entities))
        return pop['grain_summaries'][0]

    first = summarize(store.entities())
    assert first['human_event_verdict'] == 'emerged_within'
    assert first['human_event_bracket'] == [300, 6000]
    assert first['human_event_source'] == 'evtask1'
    # Reopen: fresh handle on the same DB, no controller state.
    reopened = AnnotationStore(tmp_path / 'p.db')
    second = summarize(reopened.entities())
    assert second['human_event_verdict'] == first['human_event_verdict']
    assert second['human_event_bracket'] == first['human_event_bracket']
    assert second['human_event_source'] == first['human_event_source']


def test_germination_event_snapshot_ingestion(tmp_path):
    import subprocess
    from tubetracker.annotation_store import AnnotationStore
    proj = tmp_path / 'proj'
    proj.mkdir()
    store = AnnotationStore(proj / 'annotations.db')
    store.save('task', 'ev1', {'uuid': 'ev1', 'task_type': 'germination_event',
                               'owner_uuid': 'o1', 'movie': 'ld',
                               'query_frames': [300]}, actor='test')
    store.save('germination_event', 'event-ev1',
               {'uuid': 'event-ev1', 'movie_uuid': 'ld', 'movie_content_hash': 'h',
                'owner_uuid': 'o1', 'task_uuid': 'ev1', 'window_start': 0,
                'window_end': 9000, 'verdict': 'emerged_within',
                'last_absent_frame': 300, 'first_visible_frame': 6000,
                'consulted_frames': [0, 300, 6000], 'annotator': 'test',
                'revision': 1, 'lineage': ['t'], 'review_origin': 'human'},
               actor='test')
    store.close()
    out = tmp_path / 'snap'
    r = subprocess.run(['.venv/bin/python', 'scripts/build_v30_snapshot.py',
                        '--project-dir', str(proj), '--default-movie', 'ld',
                        '--out', str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import json as _json
    manifest = _json.load(open(out / 'snapshot_manifest.json'))
    assert manifest['n_germination_events'] == 1
    events = _json.load(open(out / 'germination_events.json'))
    assert events[0]['data']['verdict'] == 'emerged_within'


def test_solver_event_veto_beats_rim_route():
    from prototypes.v30_video_apex import state_solver as solver
    frames = [0]
    rim = {'candidate_id': 'c:rim', 'state': 'present', 'tip_xy': [931.0, 482.0],
           'cap_id': '', 'cap_probability': 0.9, 'route_probability': 0.95,
           'path_complete': False, 'root_verified': False,
           'observed_partial_path_xy': [[931.0, 482.0]], 'event_score': 0.0}
    cfg = solver.SolverConfig(event_veto_threshold=0.05)
    out = solver.solve_joint_states({'g1': {0: [dict(rim)]}}, frames, config=cfg)
    assert out['rows'][0]['state'] == 'model_absent', out['rows'][0]
    assert out['rows'][0]['event_score'] == 0.0
    # A real tube (score ~1) never faces the veto.
    tube = dict(rim, candidate_id='c:tube', event_score=1.0)
    out2 = solver.solve_joint_states({'g1': {0: [tube]}}, frames, config=cfg)
    assert out2['rows'][0]['state'] == 'present', out2['rows'][0]
    # Disabled by default: behaviour preserved.
    out3 = solver.solve_joint_states(
        {'g1': {0: [dict(rim)]}}, frames, config=solver.SolverConfig())
    assert out3['rows'][0]['state'] == 'present'


def test_event_bracket_emits_absent_and_visible():
    trainer = _trainer()
    event = {'uuid': 'event-g0', 'revision': 1, 'project': 'p',
             'data': {'uuid': 'event-g0', 'movie_uuid': 'ld', 'owner_uuid': 'survey-G0-isolated',
                      'window_start': 26000, 'window_end': 28000, 'verdict': 'emerged_within',
                      'last_absent_frame': 8000, 'first_visible_frame': 8050,
                      'revision': 1, 'review_origin': 'human'}}
    extra = {'survey-G0-isolated': {'grain_native': [535.1, 640.1],
                                    'grain_radius_px': 12.1, 'source': 'survey'}}
    defs = trainer.event_bracket_presence_definitions([event], {}, extra, {'ld': {}})
    assert len(defs) == 2
    by_presence = {s['_presence']: (movie, frame, grain, s) for movie, frame, grain, s in defs}
    assert set(by_presence) == {'absent', 'visible'}
    assert by_presence['absent'][1] == 8000
    assert by_presence['visible'][1] == 8050
    for _, _, grain, s in defs:
        assert grain == [535.1, 640.1]
        assert s['_presence_role'] == 'training'
        assert s['_presence_review_origin'] == 'human'
        assert s['_presence_grain_source'] == 'survey'
        assert s['_presence_window_mismatch'] is True  # bracket outside task window
        assert s['_presence_radius'] == 12.1


def test_event_bracket_skips_unknown_and_unobservable():
    trainer = _trainer()
    unobs = {'uuid': 'e1', 'project': 'p',
             'data': {'uuid': 'e1', 'movie_uuid': 'ld', 'owner_uuid': 'own-ld-0002',
                      'window_start': 50880, 'window_end': 51480, 'verdict': 'unobservable',
                      'last_absent_frame': None, 'first_visible_frame': None,
                      'revision': 1, 'review_origin': 'human'}}
    owners = {'own-ld-0002': {'id': 'own-ld-0002', 'grain_native': [986.02, 378.89]}}
    assert trainer.event_bracket_presence_definitions([unobs], owners, {}, {'ld': {}}) == []
    # emerged_within but no resolvable geometry: skipped, never invented
    no_geo = {'uuid': 'e2', 'project': 'p',
              'data': {'uuid': 'e2', 'movie_uuid': 'ld', 'owner_uuid': 'ghost',
                       'window_start': 0, 'window_end': 100, 'verdict': 'emerged_within',
                       'last_absent_frame': 10, 'first_visible_frame': 20,
                       'revision': 1, 'review_origin': 'human'}}
    assert trainer.event_bracket_presence_definitions([no_geo], {}, {}, {'ld': {}}) == []
    # non-human drafts never supervise
    draft = {'uuid': 'e3', 'project': 'p',
             'data': dict(no_geo['data'], owner_uuid='g1', review_origin='model')}
    owners_g1 = {'g1': {'id': 'g1', 'grain_native': [0.0, 0.0]}}
    assert trainer.event_bracket_presence_definitions([draft], owners_g1, {}, {'ld': {}}) == []
