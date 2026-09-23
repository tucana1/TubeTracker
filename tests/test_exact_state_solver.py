import itertools
import math

import numpy as np
import pytest

from prototypes.v30_video_apex.state_solver import SolverConfig, solve_joint_states
from test_state_solver import candidate


def exhaustive_sequences(candidates, frames, cfg):
    """Independent enumeration; no solver preparation or transition helpers."""
    scored = []
    options = [candidates['g'][f]+[None] for f in frames]
    for sequence in itertools.product(*options):
        score, last, previous_visible = 0., None, False
        ids = []
        for frame, c in zip(frames, sequence):
            ids.append(c['candidate_id'] if c else f'uncertain:g:{frame}')
            if c:
                p, q = c['cap_probability'], c['route_probability']
                score += math.log(p/(1-p))+2*math.log(q/(1-q))
                if last is not None:
                    sigma = cfg.position_uncertainty_px+cfg.speed_px_per_source_frame*(frame-last[0])
                    score -= .5*sum((x-y)**2 for x,y in zip(c['tip_xy'],last[1]))/sigma**2
                last = frame, c['tip_xy']
            elif previous_visible:
                score -= cfg.switch_cost
            previous_visible = c is not None
        scored.append((score, ids))
    return scored


def test_forward_backward_matches_every_exhaustive_candidate_margin():
    rng = np.random.default_rng(412)
    frames = [0, 3, 11, 27]
    cfg = SolverConfig(exact_single_owner=True, beam_width=1)
    for _ in range(20):
        candidates = {'g': {f: [candidate(f'{f}-{i}', rng.uniform(0,70,2),
            cap=rng.uniform(.52,.97),route=rng.uniform(.4,.9)) for i in range(2)] for f in frames}}
        reference = exhaustive_sequences(candidates, frames, cfg)
        best = max(score for score,_ in reference)
        result = solve_joint_states(candidates, frames, config=cfg)
        assert result['objective'] == pytest.approx(best)
        assert not result['search_truncated']
        for i, row in enumerate(result['rows']):
            chosen = row['selected_candidate_id']
            alternatives = [s for s,ids in reference if ids[i] != chosen]
            assert row['search_margin'] == pytest.approx(best-max(alternatives))
            assert row['path_search_margin'] == pytest.approx(best-max(alternatives))
            for alt in row['alternatives']:
                conditional = max(s for s,ids in reference if ids[i] == alt['candidate_id'])
                assert alt['objective_gap'] == pytest.approx(best-conditional)
        assert sum(sum(r['score_components'].values()) for r in result['rows']) == pytest.approx(best)


def test_equal_routes_keep_earlier_path_ambiguity_after_future_histories_merge():
    common = dict(cap_id='same-cap', current_path_xy=[[0,0],[5,0]],
                  root_supported=True, path_complete=True, path_certificate={'kind':'fixture'})
    candidates = {'g': {0: [candidate('a',(5,0),**common), candidate('b',(5,0),**common)],
                         30: [candidate('future',(5,0),cap=.99,route=.99)]}}
    result = solve_joint_states(candidates,[0,30],config=SolverConfig(exact_single_owner=True))
    early = result['rows'][0]
    assert early['state'] == 'present' and early['path_ambiguous']
    assert early['path_search_margin'] == pytest.approx(0)
    assert early['length_px'] is None
    assert {'a','b'} <= ({early['selected_candidate_id']} | {r['candidate_id'] for r in early['alternatives']})


def test_future_human_anchor_uses_memory_without_measuring_the_hidden_frame():
    candidates={'g':{0:[candidate('left',(0,0),cap=.8),candidate('right',(30,0),cap=.81)],
                     15:[],30:[candidate('wrong',(30,0))]}}
    reviews={('g',15):{'state':'not_directly_visible','source':'hidden'},
             ('g',30):{'state':'direct_visible','tip_xy':[0,0],'source':'visible'}}
    rows=solve_joint_states(candidates,[0,15,30],constraints=reviews,
        config=SolverConfig(exact_single_owner=True,beam_width=1))['rows']
    assert rows[0]['tip_xy']==[0,0] and rows[0]['future_assisted_identity']
    assert rows[1]['state']=='occluded' and rows[1]['tip_xy'] is None and rows[1]['length_px'] is None
    assert rows[2]['as_inference_constraint'] and rows[2]['length_px'] is None


def test_long_single_owner_search_is_exact_but_candidate_clipping_stays_explicit():
    frames=list(range(0,5701,30))
    candidates={'g':{f:[candidate(f'p-{f}',(f/100,0))] for f in frames}}
    result=solve_joint_states(candidates,frames,config=SolverConfig(exact_single_owner=True,beam_width=1))
    assert len(result['rows'])==191 and not result['temporal_search_truncated']
    assert all(r['search_margin_exhaustive'] for r in result['rows'])
    assert result['component_search'][0]['algorithm']=='exact_single_owner_forward_backward'
    assert result['component_search'][0]['states'] > 191
    candidates['g'][0].append(candidate('extra',(100,100)))
    clipped=solve_joint_states(candidates,frames,
        config=SolverConfig(exact_single_owner=True,candidates_per_owner=1))
    assert clipped['candidate_pool_truncated'] and not clipped['temporal_search_truncated']
    assert not any(r['search_margin_exhaustive'] for r in clipped['rows'])
    with pytest.raises(ValueError,match='state budget'):
        solve_joint_states(candidates,frames,config=SolverConfig(exact_single_owner=True,max_exact_states=10))


def test_joint_conflicts_still_use_the_exclusive_multi_owner_solver():
    data={o:{0:[candidate('shared',(5,5),route=.9 if o=='a' else .5)]} for o in ['a','b']}
    result=solve_joint_states(data,[0],config=SolverConfig(exact_single_owner=True))
    assert result['component_search'][0]['algorithm']=='temporal_beam'
    assert sum(r['tip_xy'] is not None for r in result['rows'])==1
