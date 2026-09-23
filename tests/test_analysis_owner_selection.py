import copy

import pytest

from tubetracker.population import apply_reviewed_inventory
from test_movie_analysis import service_fixture
from test_frame_geometry import pose_fixture


def census_entity(frame=0, revision=2):
    return {'uuid': 'census', 'kind': 'task', 'revision': revision, 'data': {
        'task_type': 'census', 'movie': 'm', 'query_frames': [frame],
        'review_region': [[0, 0], [64, 0], [64, 64], [0, 64]],
        'census_instances': [
            {'grain_id': 'a', 'xy': [8, 32], 'confirmed': True},
            {'grain_id': 'b', 'xy': [42, 48], 'confirmed': True}],
        'census_class_scopes': ['grains'], 'census_border_policy': 'centre-inside',
        'census_clump_policy': 'individual-physical-grains',
        'census_membership_resolved': True, 'census_complete': True,
        'review_origin': 'human'}}


def test_selected_grain_does_not_expand_when_census_frame_enters_schedule(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    request.selected_owner_ids = ['a']
    original = copy.deepcopy(request.owners)
    request.frames = [30]
    first = service.analyze(request, review_entities=[census_entity()])
    request.frames = [0, 30]
    second = service.analyze(request, review_entities=[census_entity()])
    for result in [first, second]:
        assert [o['id'] for o in result['owners']] == ['a']
        assert result['owners'][0]['identified_at_frame'] == 0
        assert result['owners'][0]['census_source']['revision'] == 2
        assert result['inventory_review']['known_inventory_ids'] == ['a', 'b']
        assert {r['owner_id'] for r in result['rows']} == {'a'}
    assert request.owners == original  # selection never rewrites the physical registry


def test_reference_roi_selection_keeps_off_schedule_census_anchors(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    request.roi_xyxy = [0, 0, 32, 64]
    result = service.analyze(request, review_entities=[census_entity(frame=60)])
    assert [o['id'] for o in result['owners']] == ['a']
    assert result['owners'][0]['identified_at_frame'] == 60
    assert result['inventory_review']['excluded_from_analysis'] == [
        {'grain_id': 'b', 'reason': 'outside analysis owner selection'}]
    assert not result['population']['census']['census_completeness_certified']


def test_selected_subset_cannot_inherit_complete_field_denominator(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    request.selected_owner_ids = ['a']
    request.census_scope = {'movie': 'm', 'roi_xywh': [0, 0, 64, 64], 'frames': [0]}
    result = service.analyze(request, review_entities=[census_entity()])
    report = result['population']
    assert report['census']['n_grains'] == 1
    assert report['census']['coverage'][0]['complete']  # tile was exhaustive, selection was not
    assert not report['census']['census_completeness_certified']
    assert not report['germination']['denominator_complete']
    assert report['germination']['population_fraction'] is None


def test_anchor_revision_and_withdrawal_invalidate_saved_motion_off_schedule(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    pose_fixture(tmp_path, request)
    request.selected_owner_ids = ['a']
    request.frames = [30]
    service.analyze(request, review_entities=[census_entity()])
    before = len(calls)
    with pytest.raises(ValueError, match='anchor revision'):
        service.analyze(request, review_entities=[census_entity(revision=3)])
    withdrawn = census_entity(revision=3)
    withdrawn['data']['review_status'] = 'withdrawn'
    with pytest.raises(ValueError, match='census anchor'):
        service.analyze(request, review_entities=[withdrawn])
    assert len(calls) == before


def test_unknown_selected_id_is_rejected_before_pixel_inference(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    request.selected_owner_ids = ['nonexistent']
    with pytest.raises(ValueError, match='not in the current inventory'):
        service.analyze(request, review_entities=[census_entity()])
    assert calls == []


def test_later_census_does_not_delete_a_grain_missing_at_a_different_frame():
    from tubetracker.population import census_review_records
    owner = {'id': 'earlier', 'movie': 'm', 'grain_native': [20, 20],
             'identified_at_frame': 10, 'identity_verified': True}
    records = census_review_records([census_entity(frame=60)])
    resolved, audit = apply_reviewed_inventory([owner], records, movie='m')
    assert {o['id'] for o in resolved} == {'earlier', 'a', 'b'}
    assert audit['removed'] == []
