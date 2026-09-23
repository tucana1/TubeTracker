import copy
import hashlib
import json

from tubetracker.analysis_project import bootstrap_project, requested_tasks
from tubetracker.annotation_store import AnnotationStore


def test_setup_is_idempotent_and_never_overwrites_human_work(tmp_path):
    assets = tmp_path/'assets'; assets.mkdir()
    records = {}
    for key in ['movie','movie2','cap','body','snapshot','owners']:
        p = assets/(key+'.json')
        value = {'owners':[{'id':f'own-ld-000{i}', 'grain_native':[10+i,20],
                           'attachment_native':[10+i,22], 'identified_at_frame':51240} for i in [1,2,3]]} if key=='owners' else {}
        p.write_text(json.dumps(value))
        records[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    release = assets/'release.json'
    release.write_text(json.dumps({'id':'fixture','files':records,'owner_ids':[f'own-ld-000{i}' for i in [1,2,3]],
                                    'experimental_note':'test'}))
    project = tmp_path/'project'
    config = bootstrap_project(project, release_path=release, verification_only=True)
    tasks = requested_tasks(config, project)
    assert len(tasks)==9 and all(t['review_origin']=='workflow_test' for t in tasks)
    assert {t['query_frames'][0] for t in tasks if t.get('annotation_role')=='validation'} == {51120,51180,51240}
    assert {t['query_frames'][0] for t in tasks if t['task_type']=='body_mask'} == {51150,51210}
    s = AnnotationStore(project/'annotations.db')
    t = s.load(tasks[0]['uuid'])['data']; t['completed']=True; t['_drawing_draft']={'path_xy':[[1,2],[3,4]]}
    revision = s.save('task',t['uuid'],t,actor='reviewer')
    s.close()
    again = bootstrap_project(project, release_path=release)
    assert again==config
    s = AnnotationStore(project/'annotations.db')
    assert s.load(t['uuid'])['revision']==revision and s.load(t['uuid'])['data']==t
    assert s.entities('observation') == [] and s.entities('mask') == []
    s.close()
