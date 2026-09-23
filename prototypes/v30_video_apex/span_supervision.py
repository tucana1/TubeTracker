"""Visible, reviewed centreline spans with explicit grain and role lineage."""
from collections import defaultdict

import numpy as np

from tubetracker.analysis_contracts import project_identity
from tubetracker.review_semantics import is_withdrawn, is_workflow_record


def reviewed_span_definitions(observations, census, movies):
    grains, definitions, excluded = defaultdict(list), [], []
    for record in census:
        if is_withdrawn(record) or is_workflow_record(record) or not record.get('project'):
            continue
        for grain in record.get('instances', []):
            if grain.get('confirmed'):
                key = (project_identity(record['project']), record.get('movie'), grain.get('grain_id'))
                grains[key].append((record, grain))
    reserved = {'obs-rev14-route-001', 'obs-rev14-route-002', 'obs-rev14-route-003'}
    roles = {'training', 'development', 'ownership_review'}
    for o in observations:
        pts = o.get('path_xy') or []
        if len(pts) < 2 or o.get('path_complete'):
            continue
        role = o.get('annotation_role') or 'development'
        key = (project_identity(o.get('project')), o.get('movie'), o.get('owner_uuid'))
        reason = None
        if role not in roles or o.get('obs_uuid') in reserved:
            reason = 'reserved or non-development role'
        elif is_withdrawn(o) or is_workflow_record(o) or o.get('review_origin') != 'human':
            reason = 'not an active human review'
        elif o.get('movie') not in movies or not o.get('project') or key not in grains:
            reason = 'no confirmed grain in the same logical project and movie'
        elif np.asarray(pts).shape != (len(pts), 2) or not np.isfinite(pts).all():
            reason = 'invalid native path geometry'
        if reason:
            excluded.append({'id': o.get('obs_uuid'), 'reason': reason})
            continue
        visible = o.get('path_visible')
        if visible is None or visible == []:
            visible = [True] * len(pts)
            visibility_basis = 'legacy reviewed visible span; per-point flags absent'
        else:
            visibility_basis = 'explicit per-point visibility'
        if len(visible) != len(pts) or any(type(v) is not bool for v in visible):
            excluded.append({'id': o.get('obs_uuid'), 'reason': 'invalid per-point visibility'})
            continue
        segments = [[a, b] for a, b, va, vb in zip(pts[:-1], pts[1:], visible[:-1], visible[1:]) if va and vb]
        if not segments:
            excluded.append({'id': o.get('obs_uuid'), 'reason': 'no wholly visible segment'})
            continue
        candidates = sorted(grains[key], key=lambda g: (
            abs(int(g[0]['source_frame']) - int(o['source_frame'])),
            -int(g[0].get('task_revision', 0))))
        record, grain = candidates[0]
        if any(r['source_frame'] == record['source_frame'] and g['xy'] != grain['xy']
               for r, g in candidates[1:]):
            excluded.append({'id': o.get('obs_uuid'), 'reason': 'conflicting reviewed grain centres'})
            continue
        definitions.append((o['movie'], o['source_frame'], grain['xy'], {
            '_span': segments, '_span_uuid': o.get('obs_uuid'),
            '_span_revision': o.get('obs_revision', 1), '_span_owner': o['owner_uuid'],
            '_span_project': o['project'], '_span_role': role,
            '_span_lineage': {'observation': o.get('obs_uuid'), 'observation_revision': o.get('obs_revision'),
                'task': o.get('task_uuid'), 'task_revision': o.get('task_revision'),
                'logical_project': key[0], 'census_task': record.get('task_uuid'),
                'census_revision': record.get('task_revision'), 'census_frame': record['source_frame'],
                'visibility_basis': visibility_basis}}))
    return definitions, excluded
