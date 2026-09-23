"""Build a finite, source-bound review queue; applying inserts tasks only.

The default plan is the twelve outstanding low-density identity, emergence,
and length reviews. --plan accepts another explicit list of owner/frame tasks.
No proposed point, path, absence or identity becomes an accepted annotation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tubetracker.analysis_contracts import project_identity, stable_hash
from tubetracker.analysis_project import QUEUE
from tubetracker.annotation_store import AnnotationStore
from tubetracker.population import apply_reviewed_inventory, census_review_records

CF70 = 'ld|review-53b55d8191864f77b5ed91beef56cf70'
CLUMP = ['ld|review-141e74fa42fa46d98b2d5c55eec33756',
         'own-ld-0001', 'own-ld-0002', 'own-ld-0003']
PROTOCOL = 'grain-emergence-length-v1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_entities(project):
    connection = sqlite3.connect('file:'+str(Path(project).resolve()/'annotations.db')+'?mode=ro', uri=True)
    try:
        return [{'uuid': uid, 'kind': kind, 'data': json.loads(data), 'revision': revision}
                for uid,kind,data,revision in connection.execute(
                    'SELECT uuid,kind,data,revision FROM entities ORDER BY kind,uuid')]
    finally:
        connection.close()


def default_plan():
    plan = [{'owner_id': owner, 'frame': 0, 'task_type': 'grain_identity',
             'role': 'ownership_review', 'roi': [900,300,1100,455],
             'context_frames': [0,12000,24000,36000,48000,51120]}
            for owner in CLUMP]
    plan += [{'owner_id': CF70, 'frame': frame, 'task_type': 'review_tip',
              'role': 'development', 'roi': [830,420,975,525]}
             for frame in [0,300,6000,9000]]
    plan += [{'owner_id': CF70, 'frame': frame, 'task_type': 'centerline',
              'role': 'validation', 'roi': [830,420,975,525]}
             for frame in [10200,24300,52582]]
    plan += [{'owner_id': 'own-ld-0002', 'frame': 51180, 'task_type': 'review_tip',
              'role': 'validation', 'roi': [900,300,1100,470]}]
    return plan


def build_manifest(project, analysis_path, reference_path, *, plan=None, verification_only=False):
    from tubetracker.annotation_frames import FrameReader
    project = Path(project).resolve()
    entities = read_entities(project)
    analysis_path, reference_path = Path(analysis_path).resolve(), Path(reference_path).resolve()
    analysis, reference = [json.loads(p.read_text()) for p in [analysis_path, reference_path]]
    movie = analysis['movie_id']
    if (reference['movie_id'],reference['movie_sha256']) != (movie,analysis['movie_sha256']):
        raise ValueError('reference and current analyses must describe the same source movie')
    movie_path = Path(analysis['movie_path']).resolve()
    if file_hash(movie_path) != analysis['movie_sha256']:
        raise ValueError('source movie changed after the frozen analysis')
    reader = FrameReader(str(movie_path))
    try:
        width, height = reader.native_size
        frame_count = len(reader)
    finally:
        reader.close()
    owners, inventory = apply_reviewed_inventory(reference['owners'],
        census_review_records(entities, analysis['request'].get('snapshot','')), movie=movie)
    by_owner = {o['id']:o for o in owners}
    predictions = {(r['owner_id'],r['source_frame']):r
                   for a in [reference,analysis] for r in a['model_without_reviews']}
    prediction_sources = {(r['owner_id'],r['source_frame']):p
                          for a,p in [(reference,reference_path),(analysis,analysis_path)]
                          for r in a['model_without_reviews']}
    current_tasks = [dict(e['data'],uuid=e['uuid'],revision=e['revision'])
                     for e in entities if e['kind']=='task']
    plan = list(plan if plan is not None else default_plan())
    batch = PROTOCOL+'-'+stable_hash(plan)[:10]
    tasks, skipped, seen, adoptions = [], [], set(), {}
    for spec in plan:
        owner_id, frame, kind = spec['owner_id'], int(spec['frame']), spec['task_type']
        if owner_id not in by_owner or kind not in ('grain_identity','review_tip','centerline'):
            raise ValueError('queue item requires a known physical owner and supported task type')
        if spec.get('role') not in ('training','development','ownership_review','validation','test'):
            raise ValueError('queue item requires an explicit annotation role')
        owner = by_owner[owner_id]
        reference_frame = int(owner['identified_at_frame'])
        if not 0 <= frame < frame_count or not 0 <= reference_frame < frame_count:
            raise ValueError('queue frame is outside the source movie')
        key = movie,owner_id,frame,kind
        if key in seen:
            raise ValueError('duplicate owner/frame/type in requested plan')
        seen.add(key)
        existing = [t for t in current_tasks if (t.get('movie') or t.get('movie_uuid'))==movie
                    and t.get('owner_uuid')==owner_id and t.get('task_type')==kind
                    and t.get('query_frames')==[frame]]
        adopt = None
        if len(existing)==1:
            previous = existing[0]
            draft = previous.get('_drawing_draft') or {}
            has_drawing = any(draft.get(k) for k in ('path_xy','region_xy','lanes','grains','painted_xy'))
            has_answer = any(e['kind'] not in ('task','session') and
                (e['data'].get('task_uuid')==previous['uuid'] or e['data'].get('source_task_uuid')==previous['uuid'])
                for e in entities)
            if (not previous.get('completed') and not previous.get('review_queue')
                    and not previous.get('review_status') and not has_drawing and not has_answer
                    and not previous.get('draft_xy')):
                adopt = previous
        if existing and adopt is None:
            skipped.append({'owner_id':owner_id,'frame':frame,'task_type':kind,
                'reason':'existing task retained without resetting its answer',
                'tasks':[{'uuid':t['uuid'],'revision':t['revision'],'completed':bool(t.get('completed'))} for t in existing]})
            continue
        roi = list(spec['roi'])
        if (len(roi)!=4 or any(type(v) is not int for v in roi)
                or not (0<=roi[0]<roi[2]<=width and 0<=roi[1]<roi[3]<=height)):
            raise ValueError('queue ROI must be an exact native image rectangle')
        row = predictions.get((owner_id,frame))
        if kind!='grain_identity' and row is None:
            raise ValueError('freeze a model prediction before requesting tip/path validation')
        guidance = copy.deepcopy(owner)
        pose = (row or {}).get('grain_pose') or {}
        if pose.get('grain_native') is not None:
            guidance['grain_native'] = list(pose['grain_native'])
        x,y = guidance['grain_native']
        if kind!='grain_identity' and not (roi[0]<=x<roi[2] and roi[1]<=y<roi[3]):
            raise ValueError('review ROI does not contain the indicated grain')
        contexts = spec.get('context_frames') or [frame-300,frame-30,frame-1,frame,frame+1,frame+30,frame+300]
        contexts = sorted({int(f) for f in contexts if 0<=int(f)<frame_count})
        uid = 'tracking-review-'+stable_hash([project_identity(project),PROTOCOL,
                                            analysis['movie_sha256'],owner_id,frame,kind])[:20]
        if adopt:
            uid = adopt['uuid']
            adoptions[uid] = {'expected_revision':adopt['revision'],
                             'reason':'existing unanswered result-review task; no answer or drawing replaced'}
        task = {'uuid':uid,'task_type':kind,'movie':movie,'movie_uuid':movie,
            'movie_sha256':analysis['movie_sha256'],'owner_uuid':owner_id,'query_frames':[frame],
            'review_queue':QUEUE,'queue_batch':batch,'queue_order':100+len(tasks),
            'queue_position':len(tasks)+1,
            'completed':False,'review_origin':'workflow_test' if verification_only else 'human',
            'annotation_role':spec['role'],'suppress_analysis_overlays':True,
            'review_region':[[roi[0],roi[1]],[roi[2],roi[1]],[roi[2],roi[3]],[roi[0],roi[3]]],
            'review_region_fixed':True,'focus_xy':[(roi[0]+roi[2])/2,(roi[1]+roi[3])/2],
            'view_zoom':4.,'reference_frame':reference_frame,'reference_owners':[copy.deepcopy(owner)],
            'context_frames':contexts,'prior_answer':None,'reference_identity':copy.deepcopy(owner),
            'owner_choices':[] if kind=='grain_identity' else [guidance],
            'guidance_origin':'none on query image' if kind=='grain_identity' else 'model_or_reference_pointer',
            'source_analysis':{'path':str(prediction_sources.get((owner_id,frame),reference_path)),
                               'sha256':file_hash(prediction_sources.get((owner_id,frame),reference_path))}}
        if kind=='grain_identity':
            task.update(why=f'Match reference grain {owner_id[-4:]} to the SAME pollen grain at frame {frame}. '
                'Use the Reference grain frame and context selector. Click the pollen body CENTRE, then Save grain centre and next. '
                'If you cannot identify it confidently, use Cannot judge; if it is outside the image, use Grain out of field. '
                'Do not mark a tube tip. The task image has no proposed identity point.',
                unlocks='independent early identity anchor for offline grain motion',
                allowed_responses=['confirmed','ambiguous','out_of_field'])
        elif kind=='centerline':
            task.update(target_xy=guidance['grain_native'],target_r=guidance.get('grain_radius_px',13),
                why=f'Trace the tube of grain {owner_id[-4:]} at frame {frame}, from its actual exit to its current apex. '
                    'Trace the ENTIRE supported route for FULL; if either end or any segment is hidden, save PARTIAL or Cannot judge. '
                    'The magenta circle identifies the grain, not a root or tip. No model path is shown.',
                unlocks='independent root-to-tip length and geometry validation',
                allowed_responses=['full','partial','unresolved'])
        else:
            task.update(target_xy=guidance['grain_native'],target_r=guidance.get('grain_radius_px',13),
                why=f'Inspect grain {owner_id[-4:]} and its tube at frame {frame}. '
                    'Click the current apex once for a precise point, then Continue after saved point. '
                    'For a visible but blurry apex, choose tip area, outline only that apex, then Save tip area and next. '
                    'Use Tip hidden if the tube exists but its apex cannot be seen; use No owned tube visible only after checking the grain and exit. '
                    'Cannot judge is always allowed. Nearby tubes belong to other grains.',
                unlocks='emergence evidence' if spec['role']=='development' else 'independent tip visibility control; hidden is not presumed',
                allowed_responses=['precise_point','visible_area','not_directly_visible','no_tube_visible','unresolved'])
        tasks.append(task)
    return {'schema':'tubetracker.tracking_review_queue.v1','protocol':PROTOCOL,'batch':batch,
        'created_utc':datetime.now(timezone.utc).isoformat(),'project':str(project),
        'logical_project':project_identity(project),'input_entity_digest':stable_hash(entities),
        'verification_only':bool(verification_only),'movie_path':str(movie_path),
        'movie_sha256':analysis['movie_sha256'],'source_frame_count':frame_count,'image_size':[width,height],
        'count':len(tasks),'tasks':tasks,'skipped':skipped,'adoptions':adoptions,'inventory_review':inventory,
        'frozen_inputs':{str(p):file_hash(p) for p in [analysis_path,reference_path]},
        'stop_rule':'Answer this finite batch. Unknown answers are valid; do not search randomly for substitute examples.',
        'scope':'Task metadata only. No biological answer or model proposal is accepted by this manifest.'}


def apply_manifest(manifest, *, backup_dir=None):
    if manifest.get('schema')!='tubetracker.tracking_review_queue.v1':
        raise ValueError('unsupported queue manifest')
    project=Path(manifest['project']).resolve()
    if project_identity(project)!=manifest['logical_project']:
        raise ValueError('queue belongs to a different logical project')
    current=read_entities(project)
    config=next((e['data'] for e in current if e['uuid']=='analysis-configuration'),{})
    if bool(config.get('verification_only'))!=manifest['verification_only']:
        raise ValueError('human and verification queues require matching project modes')
    existing={e['uuid']:e for e in current}
    adoptions=manifest.get('adoptions',{})
    pending=[t for t in manifest['tasks'] if t['uuid'] not in existing or
             (t['uuid'] in adoptions and existing[t['uuid']]['data'].get('queue_batch')!=manifest['batch'])]
    if not pending:
        return {'inserted':0,'adopted':0,'existing':len(manifest['tasks']),'backup':None}
    if stable_hash(current)!=manifest['input_entity_digest']:
        raise ValueError('project changed since queue preparation; regenerate the manifest')
    for path,digest in manifest['frozen_inputs'].items():
        if file_hash(path)!=digest:
            raise ValueError('frozen analysis changed since queue preparation')
    if file_hash(manifest['movie_path'])!=manifest['movie_sha256']:
        raise ValueError('source movie changed since queue preparation')
    backup=Path(backup_dir or project/'backups')/('before-tracking-queue-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True,exist_ok=False)
    source=sqlite3.connect('file:'+str(project/'annotations.db')+'?mode=ro',uri=True)
    destination=sqlite3.connect(backup/'annotations.db')
    try: source.backup(destination)
    finally: destination.close();source.close()
    for p in project.glob('*.json'): shutil.copy2(p,backup/p.name)
    store=AnnotationStore(project/'annotations.db')
    expected={t['uuid']:adoptions[t['uuid']]['expected_revision'] if t['uuid'] in adoptions else None for t in pending}
    try: store.save_many([('task',t['uuid'],t) for t in pending],actor='requested-tracking-review',
                         expected_revisions=expected)
    finally: store.close()
    receipt_dir=project/'queue-manifests';receipt_dir.mkdir(exist_ok=True)
    receipt=receipt_dir/(stable_hash(manifest)+'.json')
    if not receipt.exists(): receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    adopted=sum(t['uuid'] in adoptions for t in pending)
    return {'inserted':len(pending)-adopted,'adopted':adopted,'existing':len(manifest['tasks'])-len(pending),
            'backup':str(backup),'manifest':str(receipt)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',required=True)
    parser.add_argument('--analysis',required=True)
    parser.add_argument('--reference-analysis',required=True)
    parser.add_argument('--plan',help='JSON list of explicit owner/frame/type/role/ROI tasks')
    parser.add_argument('--out',required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--verification-only',action='store_true')
    parser.add_argument('--backup-dir')
    args=parser.parse_args()
    path=Path(args.out)
    if path.exists(): raise FileExistsError(path)
    manifest=build_manifest(args.project_dir,args.analysis,args.reference_analysis,
        plan=json.loads(Path(args.plan).read_text()) if args.plan else None,
        verification_only=args.verification_only)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    result=apply_manifest(manifest,backup_dir=args.backup_dir) if args.apply else {'applied':False}
    print(json.dumps({'manifest':str(path),'pending_tasks':manifest['count'],**result}),flush=True)


if __name__=='__main__':
    main()
