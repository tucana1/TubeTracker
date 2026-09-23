"""Future identity evidence cannot license an unsupported current owned tip."""
import numpy as np
import pytest

from prototypes.v30_video_apex.route_evidence import generate_owned_routes
from prototypes.v30_video_apex.state_solver import solve_joint_states


def routes(frame, *, gap=False, stored_root=True, validation=None):
    body = np.full((96,160), .01)
    body[46:51,17:141] = .99
    if gap:
        body[:,78:87] = .01
    owner = {'id':'a', 'grain_native':[12,48], 'grain_radius_px':5}
    if stored_root:
        owner.update(attachment_native=[18,48], attachment_verified=True)
    cap = {'cap_id':f'cap-{frame}', 'tip_xy':[140,48], 'probability':.99, 'source_frame':frame}
    return generate_owned_routes(body,[0,0],owner,[cap],movie='m',source_frame=frame,
                                 alternate_routes=1,validation=validation)


def test_high_cap_and_future_frame_cannot_bridge_missing_current_owner_connection():
    broken, visible = routes(30,gap=True), routes(60)
    assert broken and broken[0]['cap_probability'] == .99
    assert 'unsupported_gap' in broken[0]['route_evidence']['tip_ownership']['failures']
    assert not broken[0]['tip_ownership_supported'] and visible[0]['tip_ownership_supported']
    rows = solve_joint_states({'a':{30:broken,60:visible}},[30,60])['rows']
    assert rows[0]['state'] == 'identity_uncertain' and rows[0]['tip_xy'] is None
    assert rows[0]['partial_path_xy'] and max(q[0] for q in rows[0]['partial_path_xy']) < 78
    assert rows[1]['state'] == 'present' and rows[1]['tip_xy'] == [140,48]
    assert rows[1]['length_px'] is None  # a connected tip is not a certified length


def test_human_tip_can_correct_a_body_model_miss_without_inventing_a_complete_path():
    rows = solve_joint_states({'a':{30:routes(30,gap=True)}},[30],movie='m',
        constraints={('a',30):{'state':'direct_visible','tip_xy':[140,48],
                               'source':'review-point','review_origin':'human'}})['rows']
    assert rows[0]['tip_xy'] == [140,48] and rows[0]['provenance'] == 'human-corrected'
    assert not rows[0]['path_complete'] and rows[0]['length_px'] is None


def test_validated_evidence_rim_root_can_measure_without_claiming_a_human_attachment():
    validation = {'validation_sha256':'unit-test-only', 'scope':{
        'movie':'m','owner_ids':['a'],'frame_interval':[0,60],'roi_xyxy':[0,0,160,96]}}
    candidates = routes(30,stored_root=False,validation=validation)
    assert candidates[0]['root_supported'] and not candidates[0]['root_verified']
    row = solve_joint_states({'a':{30:candidates}},[30],movie='m')['rows'][0]
    assert row['path_complete'] and row['length_px'] > 120
    assert row['root_supported'] and not row['root_verified']
    assert row['path_certificate']['geometry_evidence']['root_basis'] == 'evidence_rim_start'


def test_confirming_same_tip_preserves_certified_model_root_and_geometry():
    validation = {'validation_sha256':'unit-test-only', 'scope':{
        'movie':'m','owner_ids':['a'],'frame_interval':[0,60],'roi_xyxy':[0,0,160,96]}}
    candidates = routes(30, stored_root=False, validation=validation)
    before = solve_joint_states({'a':{30:candidates}}, [30], movie='m')['rows'][0]
    tip_evidence = {'id':'review-point', 'revision':1}
    after = solve_joint_states({'a':{30:candidates}}, [30], movie='m',
        constraints={('a',30):{'state':'direct_visible', 'tip_xy':[140,48],
            'source':'review-point', 'review_origin':'human',
            'tip_evidence':tip_evidence}})['rows'][0]
    assert after['path_complete'] and after['length_px'] == before['length_px']
    assert after['root_supported'] and not after['root_verified']
    assert after['path_certificate']['geometry_evidence'] == before['path_certificate']['geometry_evidence']
    assert after['path_certificate']['tip_evidence'] == tip_evidence
    assert after['route_evidence'] == before['route_evidence']
    assert after['provenance'] == 'human-corrected'
    assert 'tip_evidence' not in candidates[0]['path_certificate']


@pytest.mark.parametrize('certified,tip,partial', [
    (True, [139,48], False), (False, [140,48], False), (True, [140,48], True)])
def test_point_or_partial_review_cannot_borrow_unsupported_complete_geometry(certified, tip, partial):
    validation = {'validation_sha256':'unit-test-only', 'scope':{
        'movie':'m','owner_ids':['a'],'frame_interval':[0,60],'roi_xyxy':[0,0,160,96]}}
    candidates = routes(30, stored_root=False, validation=validation if certified else None)
    correction = {'state':'direct_visible', 'tip_xy':tip,
                  'source':'review-point', 'review_origin':'human'}
    if partial:
        correction.update(path_xy=[[100,48],tip], path_completeness='partial')
    row = solve_joint_states({'a':{30:candidates}}, [30], movie='m',
        constraints={('a',30):correction})['rows'][0]
    assert row['tip_xy'] == tip and not row['root_verified']
    assert not row['path_complete'] and row['length_px'] is None
    assert row['path_certificate'] is None
