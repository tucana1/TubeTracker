"""Active, revisioned point supervision shared by legacy folding and auditing.

The sidecar describes current licenses; removed licenses are kept in the fold
audit. Annotation databases and inherited datasets are never changed here.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from .review_semantics import partition_review_entities, precise_tip

LABEL_FIELDS = ['image_name', 'source_frame', 'label_type', 'x', 'y', 'provenance', 'confidence']
LINEAGE_FIELDS = ['image_name', 'label_type', 'x', 'y', 'task_uuid', 'obs_uuid',
                  'obs_revision', 'movie_tag', 'fold_actor', 'lineage_schema',
                  'source_project', 'movie_id', 'movie_sha256', 'task_revision',
                  'source_kind', 'source_uuid', 'source_revision', 'label_provenance']


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometry_key(row):
    return (row['image_name'], row['label_type'], round(float(row['x']), 3), round(float(row['y']), 3))


def label_key(row):
    return (*geometry_key(row), row.get('label_provenance', row.get('provenance', '')))


def license_key(row):
    return (label_key(row), *(str(row.get(k, '')) for k in
        ('source_project', 'movie_id', 'movie_sha256', 'task_uuid', 'task_revision',
         'obs_uuid', 'obs_revision', 'source_kind', 'source_uuid', 'source_revision')))


def load_project(project):
    project = Path(project).resolve()
    con = sqlite3.connect((project/'annotations.db').as_uri()+'?mode=ro', uri=True)
    try:
        entities = defaultdict(list)
        for uuid, kind, payload, revision in con.execute('SELECT uuid,kind,data,revision FROM entities'):
            entities[kind].append({'uuid': uuid, 'data': json.loads(payload), 'revision': revision})
    finally:
        con.close()
    # Old projects used observation IDs as their task link. Normalize only
    # exact known IDs in this in-memory view, before exclusion propagation.
    task_ids = {r['uuid'] for r in entities.get('task', [])}
    for row in entities.get('observation', []):
        if not row['data'].get('task_uuid'):
            matches = [u for u in task_ids if row['uuid'] in ('obs-'+u, 'obs-'+u+'-0')]
            if len(matches) == 1:
                row['data']['task_uuid'] = matches[0]
    active, excluded = partition_review_entities(entities)
    return {'project': str(project), 'all': dict(entities), 'active': active, 'excluded': excluded}


def licensed_points(evidence, movie_tag, movie_id='', movie_sha256='', actor='fold', *, active=True):
    entities = evidence['active' if active else 'all']
    by_task = defaultdict(list)
    for obs in entities.get('observation', []):
        by_task[obs['data'].get('task_uuid')].append(obs)
    result = []
    for task in sorted(entities.get('task', []), key=lambda r: r['uuid']):
        t, uid = task['data'], task['uuid']
        if (not t.get('completed') or t.get('resolution') in ('unresolvable', 'not_a_ball')
                or (movie_id and str(t.get('movie', '')) != movie_id)):
            continue
        frames = t.get('query_frames', [])
        if not frames:
            continue
        frame = int(frames[0]); image = f'{movie_tag}_{frame:06d}.png'
        kind = t.get('task_type', 'apex')
        task_hash = t.get('movie_sha256', '')
        if movie_sha256 and task_hash and task_hash != movie_sha256:
            raise ValueError('Movie content differs from reviewed task '+uid)

        def emit(label_type, xy, provenance, confidence, obs=None):
            if xy is None or len(xy) != 2:
                return
            source = obs or task
            label = dict(zip(LABEL_FIELDS, [image, str(frame), label_type,
                str(round(float(xy[0]), 3)), str(round(float(xy[1]), 3)), provenance, str(confidence)]))
            lineage = {k: label[k] for k in ('image_name', 'label_type', 'x', 'y')}
            lineage.update(task_uuid=uid, task_revision=str(task['revision']),
                obs_uuid=obs['uuid'] if obs else '', obs_revision=str(obs['revision']) if obs else '',
                source_kind='observation' if obs else 'task', source_uuid=source['uuid'],
                source_revision=str(source['revision']), lineage_schema='2',
                source_project=evidence['project'], movie_id=str(t.get('movie', movie_id)),
                movie_sha256=movie_sha256 or task_hash, movie_tag=movie_tag,
                fold_actor=actor, label_provenance=provenance)
            result.append((label, lineage))

        observations = by_task.get(uid, [])
        if observations and all(o['data'].get('direct_state') == 'unresolved_overlap' for o in observations):
            continue
        if kind == 'owner':
            for grain in (t.get('grains') or {}).values():
                if isinstance(grain, dict):
                    emit('grain', grain.get('xy'), 'human-v1', 1.0)
        elif kind == 'apex':
            emit('grain', t.get('focus_xy'), 'proposal-frst-v1', 0.5)
        if kind in ('apex', 'centerline', 'review_tip', 'review_path'):
            for obs in observations:
                record = dict(obs['data'], task_type=kind)
                if not active:
                    # Only for detecting ambiguous inherited labels, never for training.
                    record.pop('review_status', None); record.pop('review_verdict', None)
                emit('tip', precise_tip(record), 'human-v1', 1.0, obs)
    return result


def current_license(row, cache):
    """Refresh one existing license only if its current source still agrees."""
    project = row.get('source_project', '')
    if not project or row.get('lineage_schema') != '2' or not row.get('movie_sha256'):
        return None
    scope = (project, row.get('movie_tag', ''), row.get('movie_id', ''), row['movie_sha256'])
    if scope not in cache:
        evidence = load_project(project)
        cache[scope] = licensed_points(evidence, scope[1], scope[2], scope[3])
    for _, license in cache[scope]:
        if (label_key(license) == label_key(row) and license['source_kind'] == row.get('source_kind')
                and license['source_uuid'] == row.get('source_uuid')
                and license['task_uuid'] == row.get('task_uuid')):
            return dict(license, fold_actor=row.get('fold_actor', 'fold'))
    return None


def reconcile(labels, lineage, evidence, desired, *, scope):
    """Rebuild current licenses, preserving independent reviews and weak rows."""
    all_ids = {r['uuid'] for rows in evidence['all'].values() for r in rows}
    desired_by_key = {label_key(label): label for label, _ in desired}
    legacy, retained, removed, managed = [], [], [], set()
    cache = {}
    for row in lineage:
        if row.get('lineage_schema') != '2' or not row.get('source_project'):
            if row.get('task_uuid') in all_ids or row.get('obs_uuid') in all_ids:
                raise ValueError('Legacy lineage lacks project identity: rebuild from a known clean base before folding this project')
            legacy.append(row)
            continue
        managed.add(label_key(row))
        try:
            current = current_license(row, cache)
        except (sqlite3.Error, OSError) as error:
            raise ValueError('Cannot verify inherited lineage project '+row['source_project']) from error
        if current is None:
            removed.append(dict(row, exclusion='source no longer licenses this label'))
        else:
            retained.append(current)
    # A source-less old label at a known withdrawn point cannot be attributed
    # safely. A fresh independent point review can explicitly license it.
    legacy_geometry = {geometry_key(r) for r in legacy}
    known_inactive_geometry = set()
    tag, movie, digest = scope
    tasks = {r['uuid']: r['data'] for r in evidence['all'].get('task', [])}
    for obs in evidence['all'].get('observation', []):
        task = tasks.get(obs['data'].get('task_uuid'), {})
        xy = obs['data'].get('direct_xy')
        if task.get('query_frames') and xy is not None and (not movie or task.get('movie') == movie):
            known_inactive_geometry.add(geometry_key({'image_name': f"{tag}_{int(task['query_frames'][0]):06d}.png",
                'label_type': 'tip', 'x': xy[0], 'y': xy[1]}))
    active_keys = {label_key(r) for r in retained} | set(desired_by_key)
    kept_labels = {}
    for label in labels:
        key = label_key(label)
        if key in managed and key not in active_keys:
            if geometry_key(label) in legacy_geometry:
                raise ValueError('An inherited label has ambiguous legacy and withdrawn sources; rebuild from a known clean base')
            continue
        if (key not in managed and 'human' in label.get('provenance', '')
                and geometry_key(label) in known_inactive_geometry and key not in active_keys):
            raise ValueError('An inherited human label has no attributable active source; rebuild from a known clean base')
        kept_labels[key] = label
    kept_labels.update(desired_by_key)
    retained += [r for _, r in desired]
    unique = {license_key(r): r for r in retained}
    all_lineage = legacy + list(unique.values())
    all_lineage = [{k: r.get(k, '') for k in LINEAGE_FIELDS} for r in all_lineage]
    return list(kept_labels.values()), all_lineage, {
        'removed_licenses': removed, 'excluded_reviews': evidence['excluded'],
        'legacy_unverified_lineage_rows': len(legacy),
        'managed_channels': sorted({(k[0], k[1]) for k in managed | set(desired_by_key)}),
    }


def audit_labels(labels, lineage, reviews):
    """Return exact current license matches and errors without guessing provenance."""
    labels_by_key = {label_key(r): r for r in labels}
    validated, errors, legacy = [], [], []
    cache = {}
    for row in lineage:
        if row.get('lineage_schema') != '2':
            legacy.append(row)
            continue
        key = label_key(row)
        try:
            current = current_license(row, cache)
        except (ValueError, sqlite3.Error, OSError) as error:
            errors.append({'row': row, 'reason': str(error)}); continue
        if current is None or license_key(current) != license_key(row):
            errors.append({'row': row, 'reason': 'inactive, stale, incomplete or mismatched source'}); continue
        if key not in labels_by_key:
            errors.append({'row': row, 'reason': 'lineage has no exact label'}); continue
        reviewed = reviews.get(row['image_name'], {}).get(row['label_type']+'_reviewed')
        if str(reviewed) not in ('1', 'True', 'true'):
            errors.append({'row': row, 'reason': 'label channel is masked out'}); continue
        validated.append(row)
    return validated, errors, legacy
