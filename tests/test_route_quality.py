from dataclasses import asdict
import json

import numpy as np
import pytest

from prototypes.v30_video_apex.route_quality import (
    RoutePolicy, inspect_route, route_certificate, validation_failures,
    compare_validation_cases, load_route_validation, algorithm_fingerprint)


def fixture():
    body = np.full((96, 160), .99, np.float32)
    owner = {'id': 'a', 'grain_native': [12, 48], 'grain_radius_px': 5,
             'attachment_native': [18, 48], 'attachment_verified': True}
    cap = {'cap_id': 'tip', 'tip_xy': [140, 48], 'probability': .99, 'source_frame': 30}
    return body, owner, cap


def test_long_gap_and_crop_truncation_are_explicit_despite_high_mean_support():
    body, owner, cap = fixture()
    body[:, 72:77] = .1
    d = inspect_route(body, [0, 0], [[18, 48], [140, 48]], owner, cap)
    assert d['mean_support'] > .9 and d['longest_unsupported_gap_px'] >= 5
    assert 'unsupported_gap' in d['failures'] and not d['geometrically_supported']
    body[:] = .99
    cap['tip_xy'] = [158, 48]
    d = inspect_route(body, [0, 0], [[18, 48], [158, 48]], owner, cap)
    assert 'reaches_body_crop_edge' in d['failures']


def test_root_and_foreign_grain_checks_prevent_false_full_routes():
    body, owner, cap = fixture()
    other = {'id': 'b', 'grain_native': [80, 48], 'grain_radius_px': 12}
    d = inspect_route(body, [0, 0], [[18, 48], [140, 48]], owner, cap, other_owners=[other])
    assert d['grain_interior_conflicts'] == ['b']
    owner['attachment_verified'] = False
    d = inspect_route(body, [0, 0], [[18, 48], [140, 48]], owner, cap)
    assert 'root_not_verified' in d['failures']


def test_supported_geometry_requires_a_scoped_validation_certificate():
    body, owner, cap = fixture()
    d = inspect_route(body, [0, 0], [[18, 48], [140, 48]], owner, cap)
    assert d['geometrically_supported']
    cert, reasons = route_certificate(d, owner, cap, None, movie='m', frame=30)
    assert cert is None and reasons == ['full_route_validation_missing']
    validation = {'validation_sha256': 'test-only', 'scope': {
        'movie': 'm', 'owner_ids': ['a'], 'frame_interval': [0, 60], 'roi_xyxy': [0, 0, 160, 96]}}
    cert, reasons = route_certificate(d, owner, cap, validation, movie='m', frame=30)
    assert cert['source_frame'] == 30 and not reasons
    assert route_certificate(d, owner, cap, validation, movie='different', frame=30)[0] is None
    assert route_certificate(d, owner, cap, validation, movie='m', frame=90)[0] is None


def test_workflow_traces_and_missing_curves_never_satisfy_validation():
    panel = {'scope': {'movie': 'm'}, 'cases': [{'kind': 'full', 'owner_id': 'a',
        'source_frame': 30, 'observation_id': 'test-trace'}]}
    trace = {'obs_uuid': 'test-trace', 'owner_uuid': 'a', 'movie': 'm', 'source_frame': 30,
        'path_complete': True, 'path_xy': [[18, 48], [140, 48]], 'direct_state': 'direct_visible',
        'review_origin': 'workflow_test'}
    analysis = {'model_without_reviews': [{'owner_id': 'a', 'source_frame': 30,
        'tip_xy': [140, 48], 'current_path_xy': trace['path_xy'],
        'route_evidence': {'geometry': {'geometrically_supported': True}}}]}
    rows = compare_validation_cases(analysis, [trace], panel, set())
    assert rows[0]['truth'] is None
    assert 'truth_missing_workflow_or_used_for_fit' in validation_failures({'cases': rows})
    trace['review_origin'] = 'human'
    rows = compare_validation_cases(analysis, [trace], panel, {('m', 30)})
    assert rows[0]['path_max_error_px'] == 0 and not rows[0]['excluded_from_fit_frames']
    assert validation_failures({'cases': rows})
    trace['review_status'] = 'withdrawn'
    assert compare_validation_cases(analysis, [trace], panel, set())[0]['truth'] is None


def test_bare_passed_flag_cannot_enable_automatic_lengths(tmp_path):
    path = tmp_path/'validation.json'
    path.write_text(json.dumps({'schema': 'tubetracker.route_validation.v1', 'passed': True,
        'inputs': {'models': {'cap': 'c', 'body': 'b'}, 'algorithm': algorithm_fingerprint(),
                   'geometry': 'g', 'movie_sha256': 'movie-a', 'policy': asdict(RoutePolicy())}}))
    with pytest.raises(ValueError, match='evidence is incomplete'):
        load_route_validation(path, model_hashes={'cap': 'c', 'body': 'b'}, geometry_hash='g', movie_sha256='movie-a')


@pytest.mark.parametrize('assisted', [False, True])
def test_checked_report_can_license_supported_geometry_and_rejects_changed_truth(tmp_path, assisted):
    import hashlib
    import torch
    from prototypes.v30_video_apex.route_quality import geometry_fingerprint
    from prototypes.v30_video_apex.route_evidence import generate_owned_routes
    from prototypes.v30_video_apex.state_solver import solve_joint_states
    owners = [{'id': oid, 'grain_native': [12, y], 'grain_radius_px': 5,
               'attachment_native': [18, y], 'attachment_verified': True} for oid, y in [('a', 20), ('b', 60)]]
    roi = [0, 0, 160, 96]
    specs, observations, predictions = [], [], []
    for kind, oid, frame in [('full', 'a', 0), ('full', 'b', 30), ('full', 'a', 60),
                             ('hidden', 'b', 0), ('hidden', 'b', 60), ('visible_tip', 'a', 30)]:
        uid = f'{oid}-{frame}'; y = 20 if oid == 'a' else 60
        specs.append({'kind': kind, 'owner_id': oid, 'source_frame': frame, 'observation_id': uid})
        observations.append({'obs_uuid': uid, 'obs_revision': 1, 'owner_uuid': oid, 'movie': 'm',
            'source_frame': frame, 'direct_state': 'not_directly_visible' if kind == 'hidden' else 'direct_visible',
            'direct_xy': None if kind == 'hidden' else [140, y],
            'path_complete': kind == 'full', 'path_xy': [[18, y], [140, y]] if kind == 'full' else [],
            'review_origin': 'human'})
        predictions.append({'owner_id': oid, 'source_frame': frame,
            'tip_xy': None if kind == 'hidden' else [140, y],
            'current_path_xy': [[18, y], [140, y]] if kind == 'full' else [],
            'route_evidence': {'geometry': {'geometrically_supported': True}}})
    panel = {'schema': 'tubetracker.route_validation_panel.v1', 'frozen_utc': '2026-09-19T11:00:00Z',
             'scope': {'movie': 'm', 'owner_ids': ['a', 'b'], 'frame_interval': [0, 60], 'roi_xyxy': roi},
             'cases': [s for s in specs if s['kind'] != 'visible_tip'],
             'visible_tip_cases': [s for s in specs if s['kind'] == 'visible_tip']}
    algorithm = algorithm_fingerprint()
    geometry = geometry_fingerprint(owners, roi, 'analysis_roi')
    files, models, checkpoints = {}, {}, {}
    for name in ('cap', 'body'):
        directory = tmp_path/name; directory.mkdir()
        fit = directory/'panel.json'; fit.write_text(json.dumps({'cases': []}))
        digest = hashlib.sha256(fit.read_bytes()).hexdigest()
        ck = tmp_path/(name+'.pt')
        torch.save({'manifest': {'inputs': {'panel': str(directory), 'panel_sha256': digest}}}, ck)
        models[name] = hashlib.sha256(ck.read_bytes()).hexdigest()
        checkpoints[name+'_checkpoint'] = {'path': str(ck), 'sha256': models[name]}
        files[name+'_fit_panel'] = {'path': str(fit), 'sha256': digest}
    analysis = {'movie_id': 'm', 'movie_sha256': 'movie-a', 'owners': owners, 'model_without_reviews': predictions,
        'dependencies': {'files': checkpoints},
        'request': {'roi_xyxy': roi, 'body_extent': 'analysis_roi'},
        'inputs': {'pixel': {'cap_checkpoint_sha256': models['cap'], 'body_checkpoint_sha256': models['body'],
            'code': {'prototypes/v30_video_apex/'+k: v for k, v in algorithm.items()}},
            'inference': {'code': {}, 'route_policy': asdict(RoutePolicy())}}}
    if assisted:
        from tubetracker.analysis_contracts import stable_hash, route_model_bindings
        plan = {'movie':'m','seeds':[{'owner_id':'a','frame':90,'source_id':'synthetic-seed'}]}
        plan['identity'] = stable_hash(plan)
        analysis['inputs']['pixel']['body_assistance'] = plan
        models = route_model_bindings(analysis['inputs']['pixel'])
    for name, data in [('analysis', analysis), ('observations', observations), ('panel', panel)]:
        path = tmp_path/(name+'.json'); path.write_text(json.dumps(data))
        files[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {'schema': 'tubetracker.route_validation.v1', 'scope': panel['scope'],
        'inputs': {'models': models, 'algorithm': algorithm, 'movie_sha256': 'movie-a',
                   'geometry': geometry, 'policy': asdict(RoutePolicy())},
        'evidence_files': files, 'cases': compare_validation_cases(analysis, observations, panel, set())}
    path = tmp_path/'validation.json'; path.write_text(json.dumps(report))
    checked = load_route_validation(path, model_hashes=models, geometry_hash=geometry, movie_sha256='movie-a')
    if assisted:
        for changed_models in ({k:v for k,v in models.items() if k != 'body_assistance'},
                               {**models,'body_assistance':'different-seed-or-model'}):
            with pytest.raises(ValueError, match='does not match'):
                load_route_validation(path, model_hashes=changed_models,
                                      geometry_hash=geometry,movie_sha256='movie-a')
    with pytest.raises(ValueError, match='does not match'):
        load_route_validation(path, model_hashes=models,
            geometry_hash=geometry_fingerprint(owners, roi, 'analysis_roi', 'grain_reference'),
            movie_sha256='movie-a')
    for changed in ({'review_status':'withdrawn'}, {'source_revision':2}):
        for uid in ('a-0', 'a-30'):
            with pytest.raises(ValueError, match='withdrawn or revised'):
                load_route_validation(path, model_hashes=models, geometry_hash=geometry, movie_sha256='movie-a',
                    review_records=[{'source_id':uid, 'movie':'m', 'live_review':True, 'source_revision':1, **changed}])
    with pytest.raises(ValueError, match='does not match'):
        load_route_validation(path, model_hashes=models, geometry_hash=geometry, movie_sha256='movie-b')
    cap = {'cap_id': 'cap', 'tip_xy': [140, 20], 'probability': .99, 'source_frame': 0}
    routes = generate_owned_routes(np.full((96, 160), .99), [0, 0], owners[0], [cap],
        movie='m', source_frame=0, validation=checked, other_owners=owners, alternate_routes=1)
    result = solve_joint_states({'a': {0: routes}}, [0])
    assert result['rows'][0]['path_complete'] and result['rows'][0]['length_px'] == 122
    (tmp_path/'observations.json').write_text('[]')
    with pytest.raises(ValueError, match='evidence changed'):
        load_route_validation(path, model_hashes=models, geometry_hash=geometry, movie_sha256='movie-a')


def test_training_panel_is_verified_against_model_metadata(tmp_path):
    import hashlib
    import torch
    from prototypes.v30_video_apex.route_quality import verified_fit_panels
    panel = tmp_path/'panel.json'; panel.write_text('{"cases": []}')
    ck = tmp_path/'model.pt'
    torch.save({'manifest': {'panel': str(tmp_path), 'panel_sha256': hashlib.sha256(panel.read_bytes()).hexdigest()}}, ck)
    digest = hashlib.sha256(ck.read_bytes()).hexdigest()
    analysis = {'inputs': {'pixel': {n+'_checkpoint_sha256': digest for n in ('cap', 'body')}},
        'dependencies': {'files': {n+'_checkpoint': {'path': str(ck), 'sha256': digest} for n in ('cap', 'body')}}}
    assert len(verified_fit_panels(analysis)) == 2
    panel.write_text('{"cases": [{"movie": "m", "frame": 30}]}')
    with pytest.raises(ValueError, match='training panel changed'):
        verified_fit_panels(analysis)


def visible_case(truth_changes=None, *, prediction=(140, 20), fit_frames=()):
    spec = {'kind': 'visible_tip', 'owner_id': 'a', 'source_frame': 30,
            'observation_id': 'point', 'observation_revision': 3}
    truth = {'obs_uuid': 'point', 'obs_revision': 3, 'project': 'human-project',
             'movie': 'm', 'owner_uuid': 'a', 'source_frame': 30,
             'review_origin': 'human', 'direct_state': 'direct_visible',
             'direct_xy': [140, 20], **(truth_changes or {})}
    analysis = {'model_without_reviews': [{'owner_id': 'a', 'source_frame': 30,
                 'tip_xy': list(prediction) if prediction is not None else None}]}
    panel = {'scope': {'movie': 'm'}, 'cases': [], 'visible_tip_cases': [spec]}
    return compare_validation_cases(analysis, [truth], panel, set(fit_frames))[0]


def test_visible_tip_has_traceable_truth_and_does_not_supply_full_or_hidden_coverage():
    case = visible_case()
    assert case['excluded_from_fit_frames'] and case['tip_error_px'] == 0
    assert case['truth'] == {'observation_id': 'point', 'revision': 3,
                             'project': 'human-project', 'review_origin': 'human'}
    failures = validation_failures({'cases': [case]})
    assert 'need_three_full_paths_across_two_owners_and_frames' in failures
    assert 'need_two_scoped_hidden_tip_cases' in failures
    assert 'visible_tip_accuracy_failed' not in failures
    assert 'truth_missing_workflow_or_used_for_fit' not in failures
    assert 'unsupported_validation_case' not in failures


@pytest.mark.parametrize('prediction', [None, (146, 20), (float('nan'), 20)])
def test_visible_tip_error_or_withholding_fails_even_when_passed_flag_is_forged(prediction):
    case = visible_case(prediction=prediction)
    case['passed'] = True
    assert 'visible_tip_accuracy_failed' in validation_failures({'cases': [case]})


@pytest.mark.parametrize('changes', [
    {'review_origin': 'model'}, {'review_origin': 'workflow_test'},
    {'review_status': 'withdrawn'}, {'obs_revision': 4},
    {'direct_xy': None, 'tip_region_xy': [[130, 10], [150, 10], [150, 30]]},
    {'owner_uuid': 'b'}, {'source_frame': 31},
])
def test_visible_tip_requires_current_human_precise_truth(changes):
    case = visible_case(changes)
    assert case['truth'] is None
    assert 'truth_missing_workflow_or_used_for_fit' in validation_failures({'cases': [case]})


def test_visible_tip_cannot_be_used_for_fit_and_validation():
    case = visible_case(fit_frames={('m', 30)})
    assert case['tip_error_px'] == 0 and not case['excluded_from_fit_frames']
    assert 'truth_missing_workflow_or_used_for_fit' in validation_failures({'cases': [case]})


@pytest.mark.parametrize('scope', ['research assistance', None, ['m']])
def test_malformed_scope_fails_without_crashing(scope):
    assert 'malformed_validation_scope' in validation_failures({'scope': scope, 'cases': [visible_case()]})


def test_unsupported_case_still_fails_with_valid_truth():
    case = visible_case()
    case['kind'] = 'unknown_class'
    assert 'unsupported_validation_case' in validation_failures({'cases': [case]})
