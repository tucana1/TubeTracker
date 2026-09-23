"""Durable native-analysis project setup, independent of Qt and torch."""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path

from .analysis_contracts import file_hash, stable_hash
from .annotation_store import AnnotationStore

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/'prototypes/v30_video_apex/analysis_release.json'
QUEUE = 'rev14-whole-tube-review'


def release_configuration(project, *, verification_only=False, release_path=RELEASE):
    release = json.loads(Path(release_path).read_text())
    for name, record in release['files'].items():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Release input changed: ' + name)
    owners = json.loads(Path(release['files']['owners']['path']).read_text())['owners']
    owners = [copy.deepcopy(o) for o in owners if o['id'] in release['owner_ids']]
    if len(owners) != len(release['owner_ids']):
        raise ValueError('Release is missing a reviewed grain identity')
    for owner in owners:
        owner.update(identity_verified=True, attachment_verified=True, grain_radius_px=13.)
    project = Path(project).resolve()
    return {'release': release['id'], 'experimental_note': release['experimental_note'],
        'verification_only': bool(verification_only), 'release_files': release['files'],
        'cap_checkpoint': release['files']['cap']['path'],
        'body_checkpoint': release['files']['body']['path'], 'device': 'cpu',
        'inference_python': str(ROOT/'.venv/bin/python'),
        'cache_dir': str(project/'analysis_cache'), 'registry_path': str(project/'grain_registry.sqlite'),
        'extra_movies': {'m2': release['files']['movie2']['path']},
        'request': {'movie_path': release['files']['movie']['path'], 'movie_id': 'ld',
            'frames': list(range(51030, 51451, 30)), 'owners': owners,
            'roi_xyxy': [800, 200, 1160, 510], 'body_extent': 'grain_crop',
            'snapshot': str(Path(release['files']['snapshot']['path']).parent),
            'discover_grains': False, 'acquisition': {},
            'census_scope': {'movie': 'ld', 'roi_xywh': [800, 200, 360, 310],
                             'frames': [51120], 'class_scopes': ['grains']}}}


def requested_tasks(config, project):
    owners = config['request']['owners']
    by_id = {o['id']: o for o in owners}
    roi = [[800, 200], [1160, 200], [1160, 510], [800, 510]]
    origin = 'workflow_test' if config.get('verification_only') else 'human'
    base = {'review_queue': QUEUE, 'movie': 'ld', 'movie_uuid': 'ld',
        'movie_sha256': config['release_files']['movie']['sha256'], 'completed': False,
        'review_origin': origin, 'review_region': roi, 'review_region_fixed': True,
        'suppress_analysis_overlays': True, 'owner_choices': owners,
        'focus_xy': [980, 355], 'view_zoom': 1.5}
    tasks = []
    def add(uuid, task_type, frame, why, **extra):
        task = copy.deepcopy(base)
        task.update(uuid=uuid, task_type=task_type, query_frames=[frame], why=why, **extra)
        task['queue_order'] = len(tasks)
        tasks.append(task)
        return task
    for number, oid, frame, tip, reference in [
        (1, 'own-ld-0003', 51120, [950.7934912088398, 398.88957061048274], 'obs-rev13w4-004'),
        (2, 'own-ld-0003', 51240, None, None),
        (3, 'own-ld-0001', 51180, [940.59999621581, 318.09999548808116], 'obs-rev13w4-002')]:
        owner = by_id[oid]
        add(f'rev14-route-{number:03d}', 'centerline', frame,
            f'Trace the WHOLE current tube of grain {oid[-4:]}: begin at its actual exit, follow every visible bend and wrap, '
            'and stop at its current cap. Save FULL only when the entire route is visible; otherwise save PARTIAL. '
            'Later frames help identity only. Existing tip marks are prior human reviews; no model route is supplied. '
            'This trace is reserved for route validation.',
            owner_uuid=oid, target_xy=owner['grain_native'], target_r=13.,
            root_reference_xy=owner['attachment_native'], root_reference_frame=owner['identified_at_frame'],
            annotation_role='validation', draft_xy=tip,
            source_observation={'id': reference, 'revision': 1} if reference else None,
            allowed_responses=['full', 'partial', 'unresolved'])
    for number, frame in [(1, 51120), (2, 51240)]:
        add(f'rev14-crossing-{number:03d}', 'crossing', frame,
            'Review the crossing lanes in this field. Trace each distinguishable tube through its approach, intersection '
            'and departure, then assign each lane to its physical grain. Use context frames when helpful. '
            'Leave uncertain assignments unresolved. Lane ends are not tip labels.',
            owner_uuid='unassigned', annotation_role='ownership_review',
            allowed_responses=['assigned', 'partially_assigned', 'unresolved'])
    for number, oid, frame in [(1, 'own-ld-0003', 51150), (2, 'own-ld-0001', 51210)]:
        owner = by_id[oid]
        add(f'rev14-body-{number:03d}', 'body_mask', frame,
            f'Paint the WHOLE visible tube belonging to grain {oid[-4:]}, including its visible wrap. '
            'Paint uncertain overlaps with the Unknown brush; do not connect hidden gaps. Other grains have their own tubes. '
            'Only choose Reviewed field after checking all pixels inside the yellow rectangle; otherwise save partial paint. '
            'These masks are for training, separate from the reserved path frames.',
            owner_uuid=oid, target_xy=owner['grain_native'], target_r=13., brush_px=5.,
            annotation_role='training', allowed_responses=['reviewed_field', 'partial', 'unresolved'])
    add('rev14-noncap-001', 'neg_region', 23660,
        'Movie 2: draw exact polygons around visible rims, elbows or background that are NOT tube caps. '
        'Leave every real cap and uncertain patch outside the polygons. The white mark is an existing reviewed cap. '
        'Save each polygon, then Finish regions. This case will become training data and is no longer an independent transfer test.',
        movie='m2', movie_uuid='m2', movie_sha256=config['release_files']['movie2']['sha256'],
        owner_uuid='unassigned', owner_choices=[], annotation_role='training', neg_class='cap',
        negative_geometry='polygon', review_region=[[512,704],[896,704],[896,1024],[512,1024]],
        focus_xy=[704,864], known_tip_xy=[714.8,831.7], known_caps=[[714.8,831.7]],
        source_observation={'id': 'obs-r3-005', 'revision': 1},
        allowed_responses=['exact_polygons', 'unresolved'])
    uid = 'analysis-census-' + stable_hash([str(Path(project).resolve()/'annotations.db'), 'ld', [800,200,1160,510], 51120])[:18]
    instances = [{'grain_id': o['id'], 'xy': o['grain_native'], 'confirmed': False} for o in owners]
    add(uid, 'census', 51120,
        'Review EVERY individual physical grain whose centre is inside the yellow rectangle, including every member of '
        'touching clumps and grains without a tube. Confirm or remove the proposals and click missed grains. '
        'Top/left border centres are included; bottom/right are excluded. Count completeness is separate from germination classification.',
        analysis_census=True, annotation_role='census', census_instances=instances,
        census_grains=[g['xy'] for g in instances], census_class_scopes=['grains'],
        census_border_policy='centre-inside', census_clump_policy='individual-physical-grains',
        census_membership_resolved=False, census_complete=False)
    return tasks


def _ensure_project_identity(project):
    """Write the stable logical identity once; never rewrite it."""
    import json
    import uuid
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    path = project/'project-identity.json'
    if not path.exists():
        path.write_text(json.dumps({'project_id': uuid.uuid4().hex,
            'project': str(project),
            'note': 'Stable logical identity; survives backup/staging.'}, indent=1) + '\n')
    return path


def bootstrap_project(project, *, verification_only=False, release_path=RELEASE):
    _ensure_project_identity(Path(project))
    project = Path(project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(project/'annotations.db')
    try:
        previous = store.load('analysis-configuration')
        config = copy.deepcopy(previous['data']) if previous else release_configuration(
            project, verification_only=verification_only, release_path=release_path)
        if verification_only and not config.get('verification_only'):
            raise ValueError('Use a separate verification project; a genuine project cannot be converted into a test')
        if not previous:
            store.save('session', 'analysis-configuration', config, actor='project-setup')
        tasks = requested_tasks(config, project)
        for task in tasks:
            if store.load(task['uuid']) is None:
                store.save('task', task['uuid'], task, actor='requested-annotation')
        for name, value in [('analysis-config.json', config), ('requested-annotations.json', tasks)]:
            path = project/name
            if not path.exists():
                path.write_text(json.dumps(value, indent=2)+'\n')
        panel = project/'route-validation-panel.json'
        if not panel.exists():
            cases = [{'kind': 'full', 'owner_id': t['owner_uuid'], 'source_frame': t['query_frames'][0],
                      'observation_id': 'obs-'+t['uuid']} for t in tasks if t.get('annotation_role') == 'validation']
            cases += [{'kind': 'hidden', 'owner_id': o, 'source_frame': 51120, 'observation_id': uid}
                      for o, uid in [('own-ld-0001','obs-rev13w4-001'),('own-ld-0002','obs-rev13w4-003')]]
            panel.write_text(json.dumps({'schema': 'tubetracker.route_validation_panel.v1',
                'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': {'movie': 'ld', 'owner_ids': ['own-ld-0001','own-ld-0003'],
                          'frame_interval': [51120,51240], 'roi_xyxy': [800,200,1160,510]},
                'cases': cases, 'limits': 'Scoped development validation; previously inspected interval, not a blind movie benchmark.'}, indent=2)+'\n')
        return config
    finally:
        store.close()
