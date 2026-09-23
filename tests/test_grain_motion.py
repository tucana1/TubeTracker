from types import SimpleNamespace

import cv2
import numpy as np

from tubetracker.analysis_contracts import AnalysisRequest, file_hash
from tubetracker.frame_geometry import load_grain_poses, pose_index
from tubetracker.grain_motion import GrainMotionConfig, assign_instances, generate_grain_motion


class MovingGrainReader:
    native_size = (96, 96)
    reads = []
    blank = False

    def __init__(self, path):
        self.image = np.full((96,96),150,np.uint8)
        if not self.blank:
            cv2.circle(self.image,(32,48),8,55,2)
            cv2.circle(self.image,(32,48),6,180,-1)
            cv2.circle(self.image,(30,47),2,140,-1)

    def __len__(self):
        return 13

    def read(self, frame):
        self.reads.append(frame)
        image = cv2.warpAffine(self.image,np.array([[1.,0.,frame/2],[0.,1.,0.]]),(96,96),borderValue=150)
        return SimpleNamespace(exact=True,frame=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))

    def close(self):
        pass


def fixture(tmp_path):
    movie = tmp_path/'source.bin'; movie.write_bytes(b'moving grain fixture')
    owners = [{'id':'a','movie':'m','grain_native':[35,48],'grain_radius_px':8,
               'identified_at_frame':6,'identity_verified':True}]
    request = AnalysisRequest(str(movie),'m',[0,3,6,12],owners=owners,
                              grain_motion={'maximum_frame_step':3})
    return request,owners,file_hash(movie)


def test_source_frame_motion_recovers_translation_and_reuses_immutable_artifact(tmp_path):
    request,owners,digest = fixture(tmp_path)
    MovingGrainReader.reads = []
    path,receipt = generate_grain_motion(request,owners,[],digest,tmp_path/'cache',reader_factory=MovingGrainReader)
    assert not receipt['cache_hit']
    plan = load_grain_poses(path,request,owners,digest)
    for pose in plan['poses']:
        assert pose['usable_for_model_geometry']
        assert np.linalg.norm(np.asarray(pose['grain_native'])-[32+pose['source_frame']/2,48])<.25
        assert pose['review_origin']=='model'
    reads = len(MovingGrainReader.reads)
    again,receipt = generate_grain_motion(request,owners,[],digest,tmp_path/'cache',reader_factory=MovingGrainReader)
    assert receipt['cache_hit'] and path==again and len(MovingGrainReader.reads)==reads


def test_identity_review_is_consumed_and_disagreement_is_not_interpolated_as_truth(tmp_path):
    request,owners,digest = fixture(tmp_path)
    review = {'owner_id':'a','source_frame':0,'grain_native':[70,48],
              'identity_state':'confirmed','review_origin':'human','source_id':'review','revision':1}
    path,_ = generate_grain_motion(request,owners,[review],digest,tmp_path/'cache',reader_factory=MovingGrainReader)
    plan = load_grain_poses(path,request,owners,digest,identity_reviews=[review])
    poses = pose_index(plan)
    assert poses[('a',0)]['grain_native']==[70,48]
    assert poses[('a',0)]['evidence']['human_identity_reviews']==[review]
    assert not poses[('a',3)]['usable_for_model_geometry']
    assert poses[('a',3)]['pose_status']=='ambiguous'
    review = dict(review,identity_state='ambiguous',grain_native=None,revision=2)
    revised,_ = generate_grain_motion(request,owners,[review],digest,tmp_path/'cache',reader_factory=MovingGrainReader)
    assert revised!=path
    poses = pose_index(load_grain_poses(revised,request,owners,digest,identity_reviews=[review]))
    assert not poses[('a',0)]['usable_for_model_geometry']


def test_blank_current_image_does_not_turn_a_reference_center_into_supported_motion(tmp_path):
    request,owners,digest = fixture(tmp_path)
    class Blank(MovingGrainReader):
        blank = True
    path,_ = generate_grain_motion(request,owners,[],digest,tmp_path/'cache',reader_factory=Blank)
    plan = load_grain_poses(path,request,owners,digest)
    assert all(not p['usable_for_model_geometry'] for p in plan['poses'] if p['source_frame']!=6)


def test_instance_assignment_is_unique_and_equal_alternatives_remain_ambiguous():
    owners = [{'grain_radius_px':8},{'grain_radius_px':8}]
    points = np.array([[30.,30.],[40.,30.]])
    cfg = GrainMotionConfig()
    _,ambiguous = assign_instances(points,owners,[{'xy':[35,30],'radius':8}],np.zeros((2,2)),cfg)
    assert not any(r['assignment_accepted'] for r in ambiguous)
    result,evidence = assign_instances(points,owners,[{'xy':[30,30],'radius':8,'instance_label':1},
        {'xy':[40,30],'radius':8,'instance_label':2}],np.zeros((2,2)),cfg)
    assert all(r['assignment_accepted'] for r in evidence)
    assert {r['detection_instance_id'] for r in evidence}=={'1','2'}
    assert np.allclose(result,points)
