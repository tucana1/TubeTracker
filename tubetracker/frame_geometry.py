"""Frame-specific grain geometry, separate from inventory and human tube truth.

Motion providers publish model pose proposals in native pixels. The analysis
consumes their declared frame, provenance and uncertainty rather than treating
one inventory center as the grain position throughout a moving movie.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .analysis_contracts import stable_hash

SCHEMA = 'tubetracker.grain_poses.v1'
POSE_STATES = {'current_image_supported', 'interpolated', 'ambiguous', 'missing'}
IDENTITY_STATES = {'tracked_provisional', 'ambiguous', 'unresolved'}


def grain_anchor_inputs(owners):
    """Only geometry/identity lineage used by a motion provider, not tube roots."""
    names = ('id', 'movie', 'grain_native', 'grain_radius_px', 'identified_at_frame',
             'identity_verified', 'census_source', 'source_task', 'source_label')
    return [{k: copy.deepcopy(o.get(k)) for k in names}
            for o in sorted(owners, key=lambda x: x['id'])]


def pose_dependencies(path, fingerprinter):
    """Qt-safe dependency enumeration; no inference library imports."""
    p = Path(path)
    files = {'manifest': fingerprinter.file(p)}
    if files['manifest'].get('missing'):
        return files
    record = json.loads(p.read_text())
    for name, evidence in record.get('evidence_files', {}).items():
        files['evidence/' + name] = fingerprinter.file(evidence['path'])
    return files


def _point(value, name, *, optional=False):
    if value is None and optional:
        return None
    p = np.asarray(value, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError(name + ' must be a finite native pixel point')
    return p.tolist()


def load_grain_poses(path, request, owners, movie_sha256, *, fingerprinter=None,
                     identity_reviews=()):
    """Validate an immutable model-pose artifact against the current inventory.

    An artifact can cover a larger interval/inventory than this request, but
    every requested owner/frame must have an explicit record, including missing
    or ambiguous records. No implicit interpolation or static fallback is truth.
    """
    if not path:
        return None
    from .analysis_dependencies import FileFingerprinter
    fp = fingerprinter or FileFingerprinter()
    files = pose_dependencies(path, fp)
    if files['manifest'].get('missing'):
        raise FileNotFoundError(path)
    record = json.loads(Path(path).read_text())
    from .grain_identity import identity_review_hash
    if record.get('identity_review_hash', identity_review_hash([])) != identity_review_hash(list(identity_reviews)):
        raise ValueError('grain identity reviews changed; regenerate motion for the current anchors')
    if record.get('schema') != SCHEMA:
        raise ValueError('unsupported grain pose schema')
    if record.get('movie_id') != request.movie_id or record.get('movie_sha256') != movie_sha256:
        raise ValueError('grain poses belong to a different source movie')
    if record.get('coordinate_system') != 'native_pixels':
        raise ValueError('grain poses must use native source-frame pixels')
    image_size = record.get('image_size', [])
    if (len(image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                  for v in image_size)):
        raise ValueError('grain poses require the native image width and height')
    if record.get('review_origin') != 'model':
        raise ValueError('a motion artifact is model evidence, not a human annotation')
    if not record.get('provider'):
        raise ValueError('grain poses require a declared provider')
    for name, evidence in record.get('evidence_files', {}).items():
        if files['evidence/' + name].get('sha256') != evidence.get('sha256'):
            raise ValueError('grain pose source changed: ' + name)
    anchors = record.get('anchor_owners', [])
    by_id = {o['id']: o for o in anchors}
    if len(by_id) != len(anchors):
        raise ValueError('duplicate grain pose anchor owner')
    expected = grain_anchor_inputs(owners)
    for o in expected:
        if by_id.get(o['id']) != o:
            raise ValueError('grain pose inventory or anchor revision changed: ' + o['id'])
    wanted_ids = {o['id'] for o in owners}
    wanted_frames = set(request.frames)
    selected, seen = {}, set()
    for raw in record.get('poses', []):
        frame, oid = raw.get('source_frame'), raw.get('owner_id')
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError('grain pose source_frame must be a nonnegative integer')
        key = (oid, frame)
        if key in seen:
            raise ValueError('duplicate owner/frame grain pose')
        seen.add(key)
        if oid not in by_id:
            raise ValueError('grain pose has no physical owner anchor')
        if raw.get('review_origin', 'model') != 'model':
            raise ValueError('a propagated grain pose cannot become human reviewed')
        state = raw.get('pose_status')
        identity = raw.get('identity_status')
        if state not in POSE_STATES or identity not in IDENTITY_STATES:
            raise ValueError('grain pose requires explicit geometry and identity status')
        xy = _point(raw.get('grain_native'), 'grain pose', optional=state in ('missing', 'ambiguous'))
        radius = raw.get('grain_radius_px')
        if isinstance(radius, bool) or not isinstance(radius, (float, int)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('grain pose radius must be finite and positive')
        complete = raw.get('geometry_complete')
        if not isinstance(complete, bool):
            raise ValueError('grain pose geometry completeness must be explicit')
        if xy is not None and state == 'current_image_supported':
            if not (0 <= xy[0] < image_size[0] and 0 <= xy[1] < image_size[1]):
                raise ValueError('supported grain pose lies outside its source image')
            complete = complete and min(xy[0], xy[1], image_size[0]-1-xy[0], image_size[1]-1-xy[1]) >= radius
        if oid in wanted_ids and frame in wanted_frames:
            usable = (state == 'current_image_supported' and identity == 'tracked_provisional'
                      and complete and xy is not None)
            selected[key] = {**copy.deepcopy(raw), 'grain_native': xy,
                             'grain_radius_px': float(radius), 'review_origin': 'model',
                             'geometry_complete': bool(complete),
                             'usable_for_model_geometry': usable}
    missing = [(o['id'], f) for o in owners for f in request.frames if (o['id'], f) not in selected]
    if missing:
        raise ValueError(f'grain pose coverage is missing {len(missing)} owner/frame records; regenerate motion for this interval')
    consumed = [selected[k] for k in sorted(selected)]
    # Explicit instance IDs prevent two physical owners from consuming one
    # segmentation instance, even when their estimated centers differ slightly.
    instances = {}
    for pose in consumed:
        token = (pose.get('evidence') or {}).get('detection_instance_id')
        if token and pose['usable_for_model_geometry']:
            key = (pose['source_frame'], str(token))
            if key in instances:
                raise ValueError('two grain owners share one current detection instance')
            instances[key] = pose['owner_id']
    plan = {'schema': SCHEMA, 'source_file': str(Path(path).resolve()),
            'source_sha256': files['manifest']['sha256'], 'movie_sha256': movie_sha256,
            'image_size': image_size,
            'provider': record['provider'], 'review_origin': 'model',
            'anchor_owners': expected, 'evidence_files': files, 'poses': consumed,
            'identity_reviews': list(identity_reviews),
            'note': 'Frame positions are model estimates; these records do not create human presence, absence or roots.'}
    plan['identity'] = stable_hash(plan)
    return plan


def pose_index(plan):
    return {(p['owner_id'], p['source_frame']): p for p in (plan or {}).get('poses', [])}


def owner_at_frame(owner, frame, poses):
    """Resolve current geometry without mutating the reference owner or its root."""
    if not poses:
        return owner
    pose = poses.get((owner['id'], int(frame)))
    if pose is None:
        raise ValueError('missing explicit grain pose for the displayed/analyzed frame')
    current = copy.deepcopy(owner)
    current['grain_pose'] = copy.deepcopy(pose)
    current['grain_geometry_complete'] = bool(pose['usable_for_model_geometry'])
    if pose['grain_native'] is not None:
        current['grain_native'] = list(pose['grain_native'])
    current['grain_radius_px'] = pose['grain_radius_px']
    # A center trajectory provides translation, not the grain's orientation.
    # Keep a translated root prior for inspection but never call it reviewed.
    delta = np.asarray(current['grain_native']) - np.asarray(owner['grain_native'])
    current['grain_pose']['reference_grain_native'] = list(owner['grain_native'])
    current['grain_pose']['reference_frame'] = owner.get('identified_at_frame')
    current['grain_pose']['reference_to_frame_translation'] = [
        [1., 0., float(delta[0])], [0., 1., float(delta[1])], [0., 0., 1.]]
    current['grain_pose']['orientation_observed'] = False
    for key, verified in [('attachment_native', 'attachment_verified'), ('root_native', 'root_verified')]:
        if owner.get(key) is not None:
            current['grain_pose']['reference_' + key] = list(owner[key])
            current[key] = (np.asarray(owner[key]) + delta).tolist()
            current[verified] = False
    return current


def result_owner_at_frame(result, owner, frame):
    """The UI uses the same frame geometry as pixels and route generation."""
    plan = result.get('grain_geometry')
    return owner_at_frame(owner, frame, pose_index(plan)) if plan else owner
