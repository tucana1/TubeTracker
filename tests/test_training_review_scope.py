import numpy as np

from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
from scripts.train_native_caps import reviewed_corpus_cases
from tubetracker.review_semantics import is_workflow_record


def tip(uid, frame=100, role='training', **extra):
    return dict(movie='m', source_frame=frame, obs_uuid=uid, owner_uuid=uid,
                obs_revision=1, project='source', task_type='review_tip',
                direct_state='direct_visible', direct_xy=[30., 30.],
                review_origin='human', annotation_role=role, **extra)


def negative(uid, frame=100, role='training'):
    return dict(movie='m', source_frame=frame, _region_uuid=uid,
                _region_revision=1, _project='source', kind='verified_negative',
                class_scope='cap', confirmed=True, annotation_role=role,
                polygon_xy=[[70, 70], [80, 70], [80, 80], [70, 80]])


def test_validation_tip_cannot_leak_through_another_training_crop_or_alias():
    truth = CapLabelIndex([tip('reserved', role='validation'),
                           tip('other-grain')], [negative('training-region')])
    training = truth.for_training(context_offsets=(-2, 0, 2))
    target = training.spatial('m', 100, [0, 0], (100, 100))
    assert not target['positive'].any() and not target['negative'].any()
    cases, audit = reviewed_corpus_cases(truth, ['m'], [])
    assert cases == []
    assert {r.get('obs_uuid') for r in audit['excluded']} >= {'reserved', 'other-grain'}
    assert truth.locations('m', 100, [[30, 30], [75, 75]])['positive'][0]
    assert len(truth.points[('m', 100)]) == 2
    assert not is_workflow_record(truth.resolved_observations[('m', 100, 'reserved')])


def test_hidden_or_negative_validation_also_reserves_temporal_context():
    hidden = tip('hidden', role='validation')
    hidden.update(direct_state='not_directly_visible', direct_xy=None)
    truth = CapLabelIndex([hidden, tip('near', 98), tip('safe', 95)],
                          [negative('reserved-region', 200, 'test'),
                           negative('near-region', 198), negative('safe-region', 195)])
    training = truth.for_training(context_offsets=(-2, 0, 2))
    assert set(training.points) == {('m', 95)}
    assert set(training.negatives) == {('m', 195)}
    assert len(training.training_scope['role_reservations']) == 2


def test_independent_full_validation_reserves_a_frame_despite_a_precise_tip_review():
    point = tip('point')
    full = tip('full', role='validation')
    full.update(owner_uuid='point', task_type='centerline', path_complete=True)
    truth = CapLabelIndex([point, full], [])
    assert truth.resolved_observations[('m', 100, 'point')]['obs_uuid'] == 'point'
    assert not truth.for_training().points


def test_latest_revision_resolves_before_role_and_never_revives_old_training_point():
    old = tip('same')
    revised = dict(old, obs_revision=2, annotation_role='validation')
    # Input ordering cannot make an older copied revision authoritative.
    view = CapLabelIndex([revised, old], []).for_training()
    assert not view.points
    assert view.training_scope['role_reservations'][0]['revision'] == 2
    current_training = dict(revised, obs_revision=3, annotation_role='training')
    assert CapLabelIndex([old, revised, current_training], []).for_training().points


def test_explicit_reservation_removes_pixel_targets_not_just_sampled_cases():
    truth = CapLabelIndex([tip('context', 98), tip('safe', 95)], [negative('context-negative', 102)])
    view = truth.for_training(context_offsets=(-2, 0, 2), reserved_intervals=[('m', 100, 100)])
    assert set(view.points) == {('m', 95)} and not view.negatives
    assert np.all(view.locations('m', 98, [[30, 30]])['unknown'])


def test_body_queries_and_foreign_mask_pool_exclude_every_held_out_frame():
    from types import SimpleNamespace
    from scripts.train_native_body import filter_training_records
    samples = [SimpleNamespace(movie='m', source_frame=f) for f in [10, 20, 30, 40]]
    observations = [tip('reserved', 10, 'validation'), tip('safe', 40)]
    regions = [negative('reserved-negative', 20, 'test'), negative('same-frame', 30)]
    masks = [{'movie':'m','source_frame':30,'mask_uuid':'reserved-mask','annotation_role':'validation'}]
    ss, oo, rr, audit = filter_training_records(samples, observations, regions, masks)
    assert [s.source_frame for s in ss] == [40]
    assert [o['obs_uuid'] for o in oo] == ['safe'] and rr == []
    assert {r['source_id'] for r in audit} == {'reserved','reserved-negative','reserved-mask'}
