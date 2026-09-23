"""Separate endpoint detection from complete path support and ownership choice."""
import numpy as np
import pytest

from prototypes.v30_video_apex.route_diagnostics import diagnose_reference_route


def fixture():
    owner={'id':'a','grain_native':[5,20],'grain_radius_px':5,
           'attachment_native':[10,20],'attachment_verified':True}
    path=[[10,20],[40,20]]
    good={'candidate_id':'right','cap_id':'cap-right','tip_xy':[40,20],
          'cap_probability':.9,'route_probability':.8,'current_path_xy':path}
    bad={**good,'candidate_id':'wrong','tip_xy':[40,30],
         'cap_id':'cap-wrong','current_path_xy':[[10,20],[40,30]]}
    return owner,path,good,bad


@pytest.mark.parametrize('stage', ['body_support','cap_proposal','route_construction','joint_selection','route_matches_reference'])
def test_diagnosis_locates_the_failed_stage_with_identical_tip_truth(stage):
    owner,path,good,bad=fixture()
    body=np.ones((50,50),np.float32)
    candidates=[good,bad];selected=good
    caps=[{'tip_xy':[40,20]},{'tip_xy':[40,30]}]
    if stage=='body_support': body[:,23:27]=0
    if stage=='cap_proposal': caps=caps[1:];candidates=[bad];selected=bad
    if stage=='route_construction': candidates=[bad];selected=bad
    if stage=='joint_selection': selected=bad
    result=diagnose_reference_route(body,[0,0],owner,path,candidates,selected,
        caps=caps,source_frame=30,competing_candidates=[dict(good,owner_id='b')])
    assert result['first_failed_stage']==stage
    if stage=='body_support':
        assert result['unsupported_intervals'][0]['length_px'] > 3
        assert result['matching_candidate_ids']==['right']  # Proposal success does not repair missing image support.
    if stage=='joint_selection':
        assert result['matching_candidate_ids']==['right']
        assert result['competing_owners'][0]['owner_id']=='b'


def test_assisted_selection_cannot_hide_model_failure():
    owner,path,good,_=fixture()
    with pytest.raises(ValueError,match='model-only'):
        diagnose_reference_route(np.ones((50,50)),[0,0],owner,path,[good],
            dict(good,provenance='human-corrected'),caps=[{'tip_xy':[40,20]}],source_frame=30)


def test_matching_shape_does_not_hide_an_unverified_root_or_grain_conflict():
    owner,path,good,_=fixture()
    options=dict(caps=[{'tip_xy':[40,20]}],source_frame=30)
    result=diagnose_reference_route(np.ones((50,50)),[0,0],dict(owner,attachment_verified=False),
                                   path,[good],good,**options)
    assert result['first_failed_stage']=='root_not_verified'
    result=diagnose_reference_route(np.ones((50,50)),[0,0],owner,path,[good],good,
        other_owners=[{'id':'b','grain_native':[25,20],'grain_radius_px':5}],**options)
    assert result['first_failed_stage']=='grain_geometry_conflict'
