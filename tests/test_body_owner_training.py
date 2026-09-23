"""Owner supervision must change the query, never the underlying image."""
import argparse
import json

import numpy as np
import pytest

from scripts import train_native_body as trainer


@pytest.mark.parametrize('turns', range(4))
@pytest.mark.parametrize('flipped', [False, True])
def test_query_follows_actual_image_rotation_and_reflection(turns, flipped):
    # Locate a marker by transforming pixels, independently of query math.
    pixels = np.zeros((32, 32), np.uint8)
    pixels[9, 21] = 255
    transformed = np.rot90(pixels, turns)
    if flipped:
        transformed = transformed[:, ::-1]
    y, x = np.argwhere(transformed == 255)[0]
    q = trainer.query_in_crop([100+3+21, 200+5+9], [100, 200], [3, 5],
                              turns, flipped, size=32)
    np.testing.assert_array_equal(q, [x, y])


def test_pairs_keep_every_distinct_owner_but_never_duplicates_or_weak_spans():
    cases = []
    for i, (owner, xy, kind) in enumerate([
        ('ld|a', [60, 60], 'owned_body'),
        ('ld|b', [100, 60], 'owned_body'),
        ('ld|c', [60, 100], 'owned_body'),
        ('ld|ld|a', [60, 60], 'owned_body'),
        ('another-observation-of-a', [60.5, 60.5], 'owned_body'),
        ('ld|d', [100, 100], 'owned_body_scoped_span'),
    ]):
        cases.append(dict(id=str(i), kind=kind, owner=owner, grain=xy,
                          origin=[0, 0], movie='ld', frame=10))
    positive = np.zeros((len(cases), 160, 160), bool)
    for i in range(len(cases)):
        positive[i, 60:64, 20+i*10:25+i*10] = True
    # Known shared overlap must never be trained as foreign.
    positive[0:2, 50:53, 50:53] = True
    pairs, masks, report = trainer.body_owner_pairs({'cases': cases}, {'positive': positive})
    assert set(pairs[0]) == {1, 2}
    assert 3 not in pairs[0] and 4 not in pairs[0] and 5 not in pairs
    assert not masks[0, 1][50:53, 50:53].any()
    assert masks[0, 1][60:64, 20:25].all()
    assert report


def test_foreign_overlap_is_compared_in_native_coordinates():
    cases = [dict(id='a', kind='owned_body', owner='a', grain=[40, 40],
                  origin=[0, 0], movie='ld', frame=10),
             dict(id='b', kind='owned_body', owner='b', grain=[60, 40],
                  origin=[20, 0], movie='ld', frame=10)]
    positive = np.zeros((2, 80, 80), bool)
    positive[0, 20, 30] = True
    positive[1, 20, 10] = True  # same native pixel, despite different local x
    positive[0, 20, 31] = True
    positive[1, 20, 12] = True
    pairs, masks, _ = trainer.body_owner_pairs({'cases': cases}, {'positive': positive})
    assert pairs == {0: [1], 1: [0]}
    assert not masks[0, 1][20, 30] and not masks[1, 0][20, 10]
    assert masks[0, 1][20, 31] and masks[1, 0][20, 12]


def test_high_wrong_owner_response_is_penalized_even_when_the_margin_is_satisfied():
    import torch
    correct = torch.tensor([9., 2.], requires_grad=True)
    wrong = torch.tensor([8., 8.], requires_grad=True)
    relative, foreign = trainer.owner_query_loss(correct, wrong, torch.tensor([True, False]),
        margin=1., margin_weight=.5, foreign_weight=1.)
    assert relative.item() == 0. and foreign.item() > 7.9
    (relative + foreign).backward()
    assert wrong.grad[0] > .99 and wrong.grad[1] == 0.
    assert correct.grad[1] == 0.  # unknown pixels never receive a gradient


def test_focus_excludes_remote_or_out_of_crop_queries():
    cases = [
        dict(grain=[120, 120], origin=[0, 0]),
        dict(grain=[140, 120], origin=[20, 0]),
        dict(grain=[600, 120], origin=[480, 0]),
        dict(grain=[-10, 120], origin=[-100, 0])]
    pairs = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    focused = trainer.local_owner_pairs({'cases': cases}, pairs, 150.)
    assert focused[0] == [1] and 2 not in focused


def test_real_fit_uses_partner_native_position_in_anchor_image(tmp_path, monkeypatch):
    panel = tmp_path/'panel'; panel.mkdir()
    arrays = {k: np.zeros((2, trainer.STORED, trainer.STORED),
                          np.uint8 if k == 'pixels' else bool)
              for k in ('pixels', 'positive', 'ordinary', 'foreign')}
    arrays['pixels'][:] = 128
    arrays['positive'][:, 90:95, 100:105] = True
    np.savez_compressed(panel/'panel.npz', **arrays)
    cases = [dict(id='a', kind='owned_body', owner='a', grain=[160, 160],
                  radius=13, origin=[0, 0], movie='ld', frame=10),
             dict(id='b', kind='owned_body', owner='b', grain=[200, 160],
                  radius=13, origin=[40, 0], movie='ld', frame=10)]
    (panel/'panel.json').write_text(json.dumps({'cases': cases,
        'supervision_contract': 'finalized_self_reviewed_background_foreign_v1',
        'dataset_sha256': trainer.file_hash(panel/'panel.npz')}))
    captured = []
    original = trainer.body_input
    def capture(image, origin, grain, *args, **kwargs):
        captured.append((image.copy(), np.asarray(grain).copy()))
        return original(image, origin, grain, *args, **kwargs)
    monkeypatch.setattr(trainer, 'body_input', capture)
    # This fixture isolates the actual optimizer's input wiring, not accuracy.
    monkeypatch.setattr(trainer, 'evaluate', lambda *a, **kw: {
        'fit_gate_passed': False, 'min_body_iou': 0., 'mean_body_iou': 0.,
        'max_absence_region_positive_rate': 0.})
    trainer.fit(argparse.Namespace(panel=str(panel), out=str(tmp_path/'fit'),
        init=None, image_gain=1., normalization='spatial', root_channel=False,
        allow_normalization_transfer=False, allow_legacy_panel_control=False,
        seed=53, updates=1, block=100, lr=1e-5, device='cpu',
        hard_negative_weight=0., owner_margin=1., owner_weight=1., stop_on_fit_gate=False))
    assert len(captured) == 4  # two anchors, each followed by its paired query
    for a, b in zip(captured[::2], captured[1::2]):
        np.testing.assert_array_equal(a[0], b[0])
        assert np.linalg.norm(a[1]-b[1]) == pytest.approx(40.)


def test_spans_use_project_movie_owner_and_visible_development_segments(tmp_path):
    import copy
    project = str(tmp_path/'p')
    census = [dict(project=project, movie='ld', source_frame=10, task_uuid='c', task_revision=3,
                   instances=[dict(grain_id='a', xy=[40, 40], confirmed=True)])]
    original = dict(project=project, movie='ld', owner_uuid='a', obs_uuid='span',
                    obs_revision=2, task_uuid='t', task_revision=4, source_frame=12,
                    review_origin='human', annotation_role='development',
                    path_complete=False, path_xy=[[40, 41], [41, 42], [42, 43], [43, 44]],
                    path_visible=[True, True, False, True])
    variants = [original]
    for updates in [dict(project=str(tmp_path/'other')), dict(movie='other-movie'),
                    dict(annotation_role='validation'), dict(review_origin='workflow_test'),
                    dict(review_status='withdrawn'), dict(path_visible=[True, True]),
                    dict(review_origin=None)]:
        variants.append(dict(copy.deepcopy(original), **updates))
    definitions, excluded = trainer.reviewed_span_definitions(variants, census, ['ld', 'other-movie'])
    assert len(definitions) == 1 and len(excluded) == 7
    source = definitions[0][-1]
    assert source['_span'] == [[[40, 41], [41, 42]]]
    assert source['_span_lineage']['census_revision'] == 3
    assert source['_span_lineage']['task_revision'] == 4
    census[0]['instances'][0]['confirmed'] = False
    assert not trainer.reviewed_span_definitions([original], census, ['ld'])[0]


def test_symmetric_samples_align_both_owners_and_leave_unknown_and_overlap_unlabeled():
    size = trainer.STORED
    native = np.random.default_rng(9).integers(0, 255, (size, size+20), dtype=np.uint8)
    arrays = {'pixels': np.stack([native[:, :size], native[:, 20:20+size]])}
    arrays.update({k: np.zeros((2, size, size), bool) for k in ('positive', 'ordinary', 'foreign')})
    # Native positions: A owns x100; B owns x150; x120 has ambiguous overlap.
    arrays['positive'][0, 100, [100, 120]] = True
    arrays['positive'][1, 100, [150-20, 120-20]] = True
    arrays['ordinary'][1, 101, 151-20] = True
    doc = {'cases': [dict(id=owner, owner=owner, kind='owned_body', movie='m', frame=30,
        origin=origin, grain=grain, radius=13) for owner,origin,grain in
        [('a', [0, 0], [160, 160]), ('b', [20, 0], [200, 160])]]}
    symmetric, audit = trainer.same_crop_owner_samples(doc, arrays, 'symmetric_positive')
    control, _ = trainer.same_crop_owner_samples(doc, arrays, 'negative_only')
    assert [(s['image_case'], s['query_case']) for s in symmetric] == [(0,0), (1,1), (0,1), (1,0)]
    a_to_b = symmetric[2]['targets']
    assert a_to_b['positive'][100, 150] and a_to_b['foreign'][100, 100]
    assert a_to_b['ordinary'][101, 151]
    for targets in (a_to_b, control[2]['targets']):
        assert not any(v[100, 120] for v in targets.values())
        assert not any(v[200, 200] for v in targets.values())
    assert not control[2]['targets']['positive'].any()
    assert not control[2]['targets']['ordinary'].any()
    assert control[2]['targets']['foreign'][100, 100]
    assert audit[0]['overlap_pixels_excluded'] == 1
    arrays['pixels'][1, 100, 100] ^= 1
    with pytest.raises(ValueError, match='identical native image'):
        trainer.same_crop_owner_samples(doc, arrays, 'symmetric_positive')


def test_root_and_decoder_additions_preserve_legacy_predictions_but_cannot_drop_learned_queries():
    import torch
    from prototypes.v30_video_apex.native_body import NativeBodyNet
    torch.manual_seed(17)
    old = NativeBodyNet(base=4, image_gain=8).eval()
    new = NativeBodyNet(base=4, image_gain=8, root_channel=True, query_adapters=True).eval()
    notes = trainer.transfer_body_parameters(new, old)
    assert notes['zero_padded'] and notes['zero_initialized_query_tensors']
    image = np.random.default_rng(17).integers(100, 140, (64,64), dtype=np.uint8)
    x = trainer.body_input(image, [0,0], [32,32], image_gain=8)
    xr = trainer.body_input(image, [0,0], [32,32], image_gain=8, with_root=True, root_xy=[40,30])
    with torch.no_grad():
        torch.testing.assert_close(old(torch.from_numpy(x)[None]), new(torch.from_numpy(xr)[None]), rtol=0, atol=0)
    with pytest.raises(ValueError, match='discard learned'):
        trainer.transfer_body_parameters(old, new)
