"""Human grain identity anchors, independent of tube and emergence labels."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .analysis_contracts import _review_project, stable_hash
from .review_semantics import is_withdrawn, is_workflow_record


def validate_identity_observation(data):
    if not data.get('movie') or not data.get('owner_uuid') or not data.get('task_uuid'):
        raise ValueError('grain identity needs a movie, physical owner and task')
    frame = data.get('source_frame')
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError('grain identity needs an exact source frame')
    state = data.get('identity_state')
    if state not in ('confirmed', 'ambiguous', 'out_of_field'):
        raise ValueError('unknown grain identity verdict')
    point = data.get('grain_native')
    if state == 'confirmed':
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError('confirmed grain identity needs a finite native centre')
    elif point is not None:
        raise ValueError('uncertain grain identity cannot assert a centre')
    return data


def grain_identity_reviews(entities, request, owners, *, source='', tombstones=()):
    """Latest revisions and deletions win across logical project copies.

    Independent contradictory centres are quarantined as ambiguity. Workflow
    verification and model suggestions never supply human identity anchors.
    """
    candidates = []
    if request.snapshot:
        p = Path(request.snapshot)/'grain_identities.json'
        if p.exists():
            candidates += [(False, row) for row in json.loads(p.read_text())]
    project = _review_project(source)
    tasks = {e['uuid']: e['data'] for e in entities if e['kind'] == 'task'}
    for entity in entities:
        if entity['kind'] == 'grain_identity':
            candidates.append((True, {'uuid': entity['uuid'], 'revision': entity['revision'],
                'project': project, 'data': copy.deepcopy(entity['data'])}))
    latest = {}
    for live, record in candidates:
        key = (_review_project(record.get('project', '')), record['uuid'])
        rank = (live, int(record['revision']))
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, record)
    deleted = {(_review_project(t.get('project') or t.get('source_project') or source),
                t.get('uuid') or t.get('source_id')) for t in tombstones
               if t.get('kind') == 'grain_identity'}
    owner_ids = {o['id'] for o in owners}
    accepted, audit = [], []
    for key, (rank, record) in sorted(latest.items()):
        data = record['data']
        task = tasks.get(data.get('task_uuid'), {}) if key[0] == project else {}
        reason = None
        if key in deleted and not rank[0]:
            reason = 'deleted identity review'
        elif is_withdrawn(data) or is_withdrawn(task):
            reason = 'withdrawn identity review'
        elif data.get('review_origin') != 'human' or is_workflow_record(data) or is_workflow_record(task):
            reason = 'identity anchor requires explicit human provenance'
        elif data.get('movie') != request.movie_id or data.get('owner_uuid') not in owner_ids:
            reason = 'outside selected movie/owners'
        elif task and (task.get('task_type') != 'grain_identity'
                       or task.get('owner_uuid') != data.get('owner_uuid')
                       or data.get('source_frame') not in task.get('query_frames', [])):
            reason = 'identity answer does not match its task'
        else:
            try:
                validate_identity_observation(data)
            except ValueError as error:
                reason = str(error)
        if reason:
            audit.append({'uuid': record['uuid'], 'project': key[0], 'reason': reason})
            continue
        accepted.append({'owner_id': data['owner_uuid'], 'source_frame': data['source_frame'],
            'grain_native': copy.deepcopy(data.get('grain_native')),
            'identity_state': data['identity_state'], 'review_origin': 'human',
            'source_id': record['uuid'], 'source_project': key[0],
            'revision': int(record['revision']), 'task_uuid': data['task_uuid']})
    groups = {}
    for row in accepted:
        groups.setdefault((row['owner_id'], row['source_frame']), []).append(row)
    result = []
    for (owner, frame), group in sorted(groups.items()):
        states = {r['identity_state'] for r in group}
        points = [r['grain_native'] for r in group if r['identity_state'] == 'confirmed']
        conflict = len(states) > 1 or (points and np.ptp(np.asarray(points), axis=0).max() > 2.)
        if conflict:
            result.append({'owner_id': owner, 'source_frame': frame,
                'identity_state': 'ambiguous', 'grain_native': None, 'review_origin': 'human',
                'reason': 'conflicting independent identity reviews', 'sources': group})
        else:
            result.extend(group)
    return result, audit


def identity_review_hash(records):
    return stable_hash(records)
