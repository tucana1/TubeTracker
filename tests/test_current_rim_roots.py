import copy

import numpy as np
import pytest

from prototypes.v30_video_apex.route_evidence import generate_owned_routes
from prototypes.v30_video_apex.route_quality import RoutePolicy
from tubetracker.movie_analysis import route_owner_for


def test_current_rim_root_preserves_conflicting_declared_attachment_and_root_status():
    body = np.full((96,160),.01)
    body[46:51,17:141] = .99
    owner = {'id':'a','grain_native':[12,48],'grain_radius_px':5,
             'attachment_native':[12,42],'attachment_verified':True,
             'identified_at_frame':10,'source_task':'manual-grain'}
    original = copy.deepcopy(owner)
    caps = [{'cap_id':'tip','tip_xy':[140,48],'probability':.99,'source_frame':30}]
    candidates = generate_owned_routes(body,[0,0],owner,caps,source_frame=30,
        alternate_routes=1,policy=RoutePolicy(root_selection='current_body_rim'))
    assert candidates and candidates[0]['tip_ownership_supported']
    route = candidates[0]
    assert route['root_supported'] and not route['root_verified']
    assert not route['path_complete'] and route['path_certificate'] is None
    evidence = route['route_evidence']['root_evidence']
    assert evidence['basis'] == 'evidence_rim_start'
    assert evidence['declared']['attachment_native'] == [12,42]
    assert evidence['declared_discrepancy_px'] > 5
    assert owner == original
    default = generate_owned_routes(body,[0,0],owner,caps,source_frame=30,alternate_routes=1)
    assert default[0]['current_path_xy'][0] == [12,42]
    assert not default[0]['tip_ownership_supported']
    reviewed, _ = route_owner_for(owner,{30:{'a':{'xy':[18,48],'scope_frame':30,
        'observation_id':'full','revision':2,'basis':'human_full_trace'}}},30)
    assisted = generate_owned_routes(body,[0,0],reviewed,caps,source_frame=30,
        alternate_routes=1,policy=RoutePolicy(root_selection='current_body_rim'))[0]
    assert assisted['root_verified'] and assisted['current_path_xy'][0] == [18,48]
    assert assisted['route_evidence']['root_evidence']['frame_review_root']
    assert assisted['route_evidence']['geometry']['root_basis'] == 'stored_attachment'


def test_root_policy_is_explicit_and_keeps_numeric_threshold_validation():
    with pytest.raises(ValueError, match='root_selection'):
        RoutePolicy(root_selection='guess')
    with pytest.raises(ValueError, match='finite'):
        RoutePolicy(root_selection='current_body_rim',maximum_unsupported_gap_px=float('nan'))
    assert RoutePolicy(root_selection='current_body_rim').maximum_unsupported_gap_px == 2
