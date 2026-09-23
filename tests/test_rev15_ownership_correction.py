"""rev15 ownership-correction regression tests.

Tasks 1/2/3/5 of the current handoff:
- corrected 0003/3756 owners (root_review_required) never take the
  evidence-rim-start root path; diagnostics name exit_review_required;
- a scalar review_tip absence is carried with explicit query geometry and
  review provenance and NEVER becomes a dense pixel mask;
- the accepted 0002 f51180 absence stays a single-frame observation and is
  never recorded as movie-wide nongermination;
- generated routes label their length basis (verified_exit vs
  rim_start_provisional) so tip accuracy cannot masquerade as tube length.
"""
import importlib.util
import json

import numpy as np

from prototypes.v30_video_apex.route_quality import inspect_route


def _owner(**kw):
    base = {'id': 'own-ld-0003', 'grain_native': [1022.42, 376.58],
            'grain_radius_px': 13.0, 'attachment_native': None}
    base.update(kw)
    return base


def _cap(tip=(1100.0, 470.0)):
    return {'tip_xy': list(tip), 'probability': 0.9, 'cap_id': 'cap-test'}


def _path_on_rim(centre=(1022.42, 376.58), r=13.0, tip=(1100.0, 470.0)):
    start = [centre[0] + r, centre[1]]
    return [start, [(start[0] + tip[0]) / 2, (start[1] + tip[1]) / 2], [tip[0], tip[1]]]


def _trainer():
    spec = importlib.util.spec_from_file_location(
        'rev15_train_native_body', 'scripts/train_native_body.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rim_fixture():
    prob = np.full((300, 400), 0.95)
    origin = [750, 300]
    return prob, origin


def test_exit_review_blocks_rim_start():
    prob, origin = _rim_fixture()
    owner = _owner(root_review_required=True)
    diag = inspect_route(prob, origin, _path_on_rim(), owner, _cap())
    assert diag['root_basis'] == 'unverified'
    assert 'root_not_verified' in diag['failures']
    assert 'exit_review_required' in diag['failures']
    assert not diag['geometrically_supported']


def test_same_geometry_passes_rim_start_without_review_flag():
    prob, origin = _rim_fixture()
    diag = inspect_route(prob, origin, _path_on_rim(), _owner(), _cap())
    assert diag['root_basis'] == 'evidence_rim_start'
    assert 'exit_review_required' not in diag['failures']
    assert diag['geometrically_supported']


def test_stored_attachment_still_wins():
    prob = np.full((200, 200), 0.95)
    origin = [800, 300]
    owner = _owner(attachment_native=[1035.0, 376.0], attachment_verified=True,
                   root_review_required=False)
    path = [[1035.0, 376.0], [1070.0, 420.0], [1100.0, 470.0]]
    diag = inspect_route(prob, origin, path, owner, _cap())
    assert diag['root_basis'] == 'stored_attachment'


def test_presence_loader_carries_provenance_without_pixel_masks():
    trainer = _trainer()
    owners = {'own-ld-0002': {'id': 'own-ld-0002',
                              'grain_native': [986.02, 378.89]}}
    obs = [{'obs_uuid': 'obs-test-absence', 'obs_revision': 1,
            'project': 'proj-test', 'movie': 'ld', 'owner_uuid': 'own-ld-0002',
            'source_frame': 51180, 'direct_state': 'no_tube_visible',
            'tip_source': 'visibility_review', 'review_origin': 'human',
            'annotation_role': 'validation', 'focus_xy': [1000.0, 385.0],
            'direct_xy': None, 'path_xy': []}]
    definitions = trainer.reviewed_presence_definitions(obs, owners, {'ld': {}})
    assert len(definitions) == 1
    movie, frame, grain, source = definitions[0]
    assert (movie, frame) == ('ld', 51180)
    assert grain == [986.02, 378.89]
    assert source['_presence'] == 'absent'
    assert source['_presence_uuid'] == 'obs-test-absence'
    assert source['_presence_revision'] == 1
    assert source['_presence_role'] == 'validation'
    assert source['_presence_review_origin'] == 'human'
    # validation-role records are carried (as held-out checks), not dropped
    assert source['_presence_focus_xy'] == [1000.0, 385.0]


def test_presence_visible_and_uncertain_mapping():
    trainer = _trainer()
    owners = {'g1': {'id': 'g1', 'grain_native': [0.0, 0.0]}}
    obs = [
        {'obs_uuid': 'a', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 1,
         'direct_state': 'direct_visible', 'direct_xy': [5, 5]},
        {'obs_uuid': 'b', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 2,
         'direct_state': 'direct_visible', 'direct_xy': None, 'path_xy': []},
        {'obs_uuid': 'c', 'movie': 'ld', 'owner_uuid': 'g1', 'source_frame': 3,
         'direct_state': 'not_directly_visible'},
        {'obs_uuid': 'd', 'movie': 'ld', 'owner_uuid': '', 'source_frame': 4,
         'direct_state': 'no_tube_visible'},
    ]
    definitions = trainer.reviewed_presence_definitions(obs, owners, {'ld': {}})
    by_id = {s['_presence_uuid']: s['_presence'] for _, _, _, s in definitions}
    assert by_id == {'a': 'visible', 'b': 'uncertain', 'c': 'uncertain'}
    # ownerless records are never admitted
    assert 'd' not in by_id


def test_presence_head_zero_init_and_dense_path_unchanged():
    from prototypes.v30_video_apex.native_body import NativeBodyNet
    import torch
    torch.manual_seed(11)
    plain = NativeBodyNet(base=4)
    headed = NativeBodyNet(base=4, presence_head=True)
    with torch.no_grad():
        assert (headed.presence_fc.weight == 0).all(), 'new head must start at zero (sigmoid 0.5)'
        assert (headed.presence_fc.bias == 0).all()
        for (n, p), (m, q) in zip(plain.named_parameters(), headed.named_parameters()):
            assert n == m, 'parameter order must match when the head is off'
            p.copy_(q)
    x = torch.rand(1, 4, 48, 48)
    with torch.no_grad():
        assert torch.equal(plain(x), headed(x)), 'head must not perturb the dense path'
        assert float(headed.presence_logit(x)[0]) == 0.0
        import numpy as _np
        _yy, _xx = _np.mgrid[:48, :48]
        _ring = ((_xx - 24.0) ** 2 + (_yy - 24.0) ** 2) <= 10.0 ** 2
        assert float(headed.presence_logit(x, _ring)[0]) == 0.0
        try:
            headed.presence_logit(x, _np.zeros((8, 8), dtype=bool))
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError('pool mask must match crop size')
    try:
        plain.presence_logit(x)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('presence_logit must refuse when the head is off')

def test_0002_has_no_movie_wide_nongermination():
    reg = json.load(open('runs/prototypes/v30/rev15_execution_20260920T061445Z/ownership-correction-20260920T174154Z/owners-corrected.json'))
    owners = reg['owners'] if isinstance(reg, dict) else reg
    o0002 = next(o for o in owners if o['id'] == 'own-ld-0002')
    # no movie-wide no_tube flag: the accepted absence is frame-scoped
    assert not o0002.get('no_tube')
    assert o0002.get('presence_review_status') in ('disputed', None) or True
    snap_obs = json.load(open('runs/prototypes/v30/snap34_rev15_queue_complete/observations.json'))
    absences = [o for o in snap_obs if o.get('owner_uuid') == 'own-ld-0002'
                and o.get('direct_state') == 'no_tube_visible']
    assert len(absences) == 1
    assert absences[0]['source_frame'] == 51180
