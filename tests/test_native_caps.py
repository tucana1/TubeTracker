import numpy as np
import torch
import pytest

from prototypes.v30_video_apex.cap_evidence import CapLabelIndex, emit_cap_candidates
from prototypes.v30_video_apex.native_caps import NativeCapNet, native_cap_loss


def test_broader_corpus_groups_sampling_without_merging_owners_or_leaking_context():
    from scripts.train_native_caps import reviewed_corpus_cases
    observations = []
    for uid, movie, frame, owner, xy in [
        ('a','ld',50,'grain-a',[30,30]), ('b','ld',50,'grain-b',[31,30]),
        ('c','ld',98,'grain-c',[50,50]), ('d','ld',105,'grain-d',[60,60]),
        ('e','m2',50,'grain-e',[30,30])]:
        observations.append({'movie':movie, 'source_frame':frame, 'obs_uuid':uid,
            'owner_uuid':owner, 'direct_state':'direct_visible', 'direct_xy':xy})
    index = CapLabelIndex(observations, [])
    cases, audit = reviewed_corpus_cases(index, ['ld'], [('ld',100,110)])
    assert len(cases) == 1
    assert cases[0]['sampling_group']['owner_aliases'] == ['grain-a','grain-b']
    assert len(index.points[('ld',50)]) == 2
    assert {r['obs_uuid'] for r in audit['excluded']} == {'c','d','e'}
    assert next(r for r in audit['excluded'] if r['obs_uuid']=='c')['reason'] == 'reserved_query_or_context_frame'


def test_negative_response_cannot_escape_rejection_by_refining_into_unknown():
    from scripts.train_native_caps import negative_response_counts
    probability = np.zeros((64, 64), np.float32)
    probability[30, 29] = .9
    negative = np.zeros((64, 64), bool); negative[16:48, 16:30] = True
    out = {'valid': np.ones_like(negative), 'probability': probability,
           'caps': [{'location_xy': [29, 30], 'unrefined_xy': [29, 30],
                     'tip_xy': [32, 30], 'probability': .9}]}
    counted = negative_response_counts(out, negative, np.array([0, 0]))
    assert counted['raw_negative_peaks'] == 1
    assert counted['licensed_negative_peaks'] == 0
    assert counted['negative_to_unknown_escapes'] == 1
    # Refusing an invalid refinement must not hide the classifier response either.
    out['caps'] = []
    assert negative_response_counts(out, negative, np.array([0, 0]))['raw_negative_peaks'] == 1


def test_generic_cap_truth_survives_partial_owner_route_and_absence():
    obs = [{"movie": "m", "source_frame": 10, "obs_uuid": "partial",
            "direct_state": "direct_visible", "direct_xy": [20., 20.],
            "path_complete": False, "owner_uuid": "b"}]
    regs = [{"movie_uuid": "m", "source_frame": 10, "kind": "owned_absence",
             "owner_uuid": "a", "polygon_xy": [[0, 0], [50, 0], [50, 50], [0, 50]]},
            {"movie_uuid": "m", "source_frame": 10, "kind": "verified_negative",
             "confirmed": True, "class_scope": "apex",
             "polygon_xy": [[80, 80], [100, 80], [100, 100], [80, 100]]}]
    idx = CapLabelIndex(obs, regs)
    t = idx.locations("m", 10, [[20, 20], [50, 50], [90, 90]])
    assert t["positive"].tolist() == [True, False, False]
    assert t["negative"].tolist() == [False, False, True]
    assert t["unknown"].tolist() == [False, True, False]
    assert len(idx.owned_absences) == 1


def test_contradictory_negative_region_is_quarantined():
    idx = CapLabelIndex(
        [{"movie": "m", "source_frame": 1, "obs_uuid": "tip", "direct_state": "direct_visible",
          "direct_xy": [10, 10]}],
        [{"movie_uuid": "m", "source_frame": 1, "kind": "verified_negative", "confirmed": True,
          "class_scope": "apex", "_region_uuid": "conflict",
          "polygon_xy": [[0, 0], [50, 0], [50, 50], [0, 50]]}])
    assert idx.summary()["licensed_negative_regions"] == 0
    assert idx.quarantined[0]["region"] == "conflict"


def test_no_invalid_sentinel_or_refinement_escape_emitted():
    xy = np.array([[20, 20], [40, 40], [60, 60], [80, 80]], float)
    assert not emit_cap_candidates([-1, -1, -1, -1], xy, [False] * 4, threshold=0)
    out = emit_cap_candidates([.9, np.nan, .8, .95], xy, [True] * 4,
                               offsets_xy=[[0, 0], [0, 0], [np.inf, 0], [30, 0]],
                               bounds_xyxy=(0, 0, 100, 100), movie="m", source_frame=2)
    assert len(out) == 1 and out[0]["tip_xy"] == [20, 20]
    assert emit_cap_candidates([.9], [[20, 20]], [True], movie="m", source_frame=2)[0]["cap_id"] == out[0]["cap_id"]


@pytest.mark.parametrize('hard_weight', [0., .5])
def test_sparse_cap_loss_has_exactly_zero_unknown_gradient(hard_weight):
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    off = torch.zeros(1, 2, 8, 8, requires_grad=True)
    var = torch.zeros_like(logits, requires_grad=True)
    pos, neg = torch.zeros_like(logits, dtype=torch.bool), torch.zeros_like(logits, dtype=torch.bool)
    pos[..., 1, 1], neg[..., 6, 6] = True, True
    loss, _ = native_cap_loss({"logits": logits, "offset_xy": off, "logvar": var}, pos, neg,
                              torch.ones_like(off), hard_negative_weight=hard_weight)
    loss.backward()
    unknown = ~(pos | neg)
    assert torch.count_nonzero(logits.grad[unknown]) == 0
    assert logits.grad[..., 1, 1] < 0 and logits.grad[..., 6, 6] > 0
    assert torch.count_nonzero(off.grad[~pos.expand_as(off)]) == 0


def test_hard_negative_term_targets_hot_licensed_pixels_without_using_unknowns():
    logits = torch.tensor([[[[-2., 3., 20.]]]], requires_grad=True)
    zero = torch.zeros_like(logits)
    pred = {'logits': logits, 'offset_xy': zero.expand(-1, 2, -1, -1), 'logvar': zero}
    pos = torch.zeros_like(logits, dtype=torch.bool)
    neg = torch.tensor([[[[True, True, False]]]])
    loss, terms = native_cap_loss(pred, pos, neg, pred['offset_xy'], hard_negative_weight=.5, hard_negative_k=1)
    loss.backward()
    assert logits.grad[0, 0, 0, 1] > 8 * logits.grad[0, 0, 0, 0]
    assert logits.grad[0, 0, 0, 2] == 0
    assert terms['hard_negative_bce'] == pytest.approx(torch.nn.functional.softplus(torch.tensor(3.)).item())


def test_single_frame_depends_only_on_query_and_temporal_can_use_context():
    torch.manual_seed(0)
    model = NativeCapNet(base=4)
    clip = torch.rand(1, 3, 1, 32, 32, requires_grad=True)
    out = model(clip)
    g = torch.autograd.grad(out["logits"].sum(), clip)[0].abs().sum((0, 2, 3, 4))
    assert g[1] > 0 and g[0] == g[2] == 0
    model.temporal = True
    with torch.no_grad():
        model.context_gain.fill_(.2)
    g = torch.autograd.grad(model(clip)["logits"].sum(), clip)[0].abs().sum((0, 2, 3, 4))
    assert torch.all(g > 0)
