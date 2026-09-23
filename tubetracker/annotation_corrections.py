"""Audited corrections to accepted labels without rewriting their history.

Plans name exact entities and revisions. Physical grain inventories and
identity observations are separate from a tube's corrected owner.
"""
from __future__ import annotations

import copy
import hashlib
import json


def payload_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def reassign_tube_record(entity, new_owner, *, project, correction):
    """Keep reviewed geometry; a changed owner requires a new root review.

    A formerly FULL trace already reviewed its apex. Preserve that precise
    point with its old revision as evidence, while withdrawing completeness.
    An originally PARTIAL trace cannot gain a tip through reassignment.
    """
    if entity['kind'] not in ('observation', 'mask', 'region', 'duel'):
        raise ValueError('reassign a tube label, not a physical grain record')
    data = copy.deepcopy(entity['data'])
    old_owner = data.get('owner_uuid')
    if not old_owner or not new_owner or old_owner == new_owner:
        raise ValueError('ownership correction requires two distinct owners')
    if data.get('direct_state') == 'no_tube_visible' or data.get('no_tube'):
        raise ValueError('a grain absence is not a tube to reassign')
    if data.get('review_status') == 'withdrawn':
        raise ValueError('do not reactivate a withdrawn label by reassigning it')
    audit = dict(copy.deepcopy(correction), old_owner=old_owner, new_owner=new_owner,
                 source_project=str(project), source_uuid=entity['uuid'],
                 source_revision=int(entity['revision']))
    data.setdefault('label_corrections', []).append(audit)
    data['owner_uuid'] = new_owner
    if 'owner_key' in data:
        movie = data.get('movie') or data.get('movie_uuid')
        if not movie:
            raise ValueError('owner_key correction requires its movie')
        data['owner_key'] = f'{movie}|{new_owner}'
    if entity['kind'] == 'observation' and data.get('path_xy'):
        was_full = bool(data.get('path_complete'))
        if was_full:
            from .review_semantics import precise_tip
            point = precise_tip(data)
            if point is not None:
                data['tip_source'] = 'reviewed_full_path_apex'
                data['tip_review_basis'] = {
                    'kind': 'preserved_reviewed_full_path_apex',
                    'source_project': str(project), 'observation_uuid': entity['uuid'],
                    'revision': int(entity['revision']), 'path_complete': True,
                    'review_origin': data.get('review_origin', 'human'),
                    'direct_xy': copy.deepcopy(point)}
        data['path_complete'] = False
        data['root_review_required'] = True
    return data


def correction_change(entity, after, *, reason):
    if entity['data'] == after:
        raise ValueError('correction must change the record')
    return {'uuid': entity['uuid'], 'kind': entity['kind'],
            'expected_revision': int(entity['revision']),
            'before_sha256': payload_hash(entity['data']),
            'after_sha256': payload_hash(after), 'reason': reason,
            'after': copy.deepcopy(after)}


def apply_project_correction(store, changes, *, correction_id, actor):
    """Compare the whole plan before a single atomic, append-only save.

    A receipt in the same transaction makes retrying a symmetric swap safe:
    it can never swap the same records back or overwrite subsequent reviews.
    """
    receipt_id = 'annotation-correction-' + correction_id
    digest = payload_hash(changes)
    receipt = store.load(receipt_id)
    if receipt is not None:
        if receipt['kind'] != 'annotation_correction' or receipt['data'].get('plan_sha256') != digest:
            raise ValueError('correction ID already belongs to a different plan')
        return dict(receipt['data'], already_applied=True)
    if len({c['uuid'] for c in changes}) != len(changes):
        raise ValueError('one correction per entity')
    records, expected, summary = [], {receipt_id: None}, []
    for change in changes:
        entity = store.load(change['uuid'])
        if (entity is None or entity['kind'] != change['kind']
                or entity['revision'] != change['expected_revision']
                or payload_hash(entity['data']) != change['before_sha256']):
            raise ValueError('annotation changed after planning: ' + change['uuid'])
        if payload_hash(change['after']) != change['after_sha256']:
            raise ValueError('correction plan payload changed: ' + change['uuid'])
        expected[change['uuid']] = change['expected_revision']
        records.append((change['kind'], change['uuid'], change['after']))
        summary.append({key: change[key] for key in
                        ('uuid', 'kind', 'expected_revision', 'before_sha256',
                         'after_sha256', 'reason')})
    data = {'correction_id': correction_id, 'plan_sha256': digest,
            'changes': summary, 'actor': actor}
    records.append(('annotation_correction', receipt_id, data))
    revisions = store.save_many(records, actor=actor, expected_revisions=expected)
    return dict(data, already_applied=False,
                new_revisions=dict(zip([r[1] for r in records], revisions)))
