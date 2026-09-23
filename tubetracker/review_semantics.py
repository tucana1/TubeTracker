"""Tip evidence licensed by an observation, independent of path endpoints."""
from __future__ import annotations

import math


TRAINING_REVIEW_ROLES = frozenset({'training', 'development', 'ownership_review', 'unspecified'})


def training_review_role(record):
    """Legacy unassigned roles remain development; human provenance is separate."""
    return record.get('annotation_role') or 'development'


def is_withdrawn(record):
    return (record.get('review_status') == 'withdrawn'
            or record.get('review_verdict') == 'unresolved')


def is_workflow_record(record):
    """Workflow receipts are executable checks, never biological truth."""
    normalize = lambda v: str(v or "").lower().replace("-", "_")
    return (normalize(record.get("review_origin")) in ("workflow_test", "synthetic_test")
            or normalize(record.get("annotator")) in ("workflow_test", "synthetic_test")
            or record.get("training_eligible") is False)


def is_workflow_measurement(row):
    """Follow the selected tip/geometry evidence, including mixed-source rows."""
    return (is_workflow_record(row)
            or is_workflow_record({'review_origin': row.get('provenance')})
            or any(is_workflow_measurement(value) for key in
                   ('constraint', 'path_certificate', 'geometry_source', 'tip_evidence')
                   if isinstance(value := row.get(key), dict)))


def mask_review_provenance(record, parent=None):
    """Resolve a mask's origin without upgrading explicit model proposals.

    Legacy masks came from the painting task and predate review_origin. Keep
    that compatibility only with a sourced task/annotator trail. A newer
    explicit origin always wins, including an explicit unknown or model.
    """
    parent = parent or {}
    for basis, source in (('explicit_mask', record), ('parent_task', parent)):
        value = source.get('review_origin')
        if value is not None and str(value).strip():
            return {'review_origin': str(value).strip().lower().replace('-', '_'),
                    'review_origin_basis': basis}
    sourced = record.get('source_project') or record.get('project') or record.get('_project')
    annotator = str(record.get('annotator') or '').strip().lower().replace('-', '_')
    manual_task = parent.get('task_type') == 'body_mask'
    known_nonhuman = annotator in ('model', 'machine', 'automatic', 'unknown',
                                   'workflow_test', 'synthetic_test')
    if sourced and record.get('task_uuid') and not known_nonhuman and (annotator or manual_task):
        return {'review_origin': 'human', 'review_origin_basis': 'legacy_manual_mask'}
    return {'review_origin': 'unknown', 'review_origin_basis': 'missing_human_lineage'}


def census_grain_points(record):
    """Return reviewed grain centres, keeping unconfirmed proposals out.

    Legacy census dots were manual clicks. Instance-aware census records
    carry an explicit confirmation per grain, even before field completion.
    """
    instances = record.get('instances') or []
    if record.get('instance_review_required', bool(instances)):
        points = [g['xy'] for g in instances if g.get('confirmed')]
    else:
        points = record.get('grains') or []
    if any(len(q) != 2 or not all(math.isfinite(float(v)) for v in q) for q in points):
        raise ValueError('Reviewed census grains require finite native centres')
    return [[float(q[0]), float(q[1])] for q in points]


def partition_review_entities(entities):
    """Keep verification records and their dependants in a separate audit.

    Input/output use the store's kind -> entity-list representation. The
    original records remain intact, including revisions and source links.
    """
    flat = [(kind, row) for kind, rows in entities.items() for row in rows]
    excluded = {row["uuid"] for _, row in flat if is_workflow_record(row["data"])}
    links = ("task_uuid", "source_task_uuid", "source_obs_uuid")
    while True:
        added = {row["uuid"] for _, row in flat
                 if any(row["data"].get(k) in excluded for k in links)}
        if added <= excluded:
            break
        excluded |= added
    withdrawn = {row['uuid'] for _, row in flat if is_withdrawn(row['data'])}
    # A task withdrawal affects evidence produced by that task, not an
    # independent human review that merely used it as viewing context.
    withdrawn |= {row['uuid'] for _, row in flat
                  if any(row['data'].get(k) in withdrawn for k in ('task_uuid', 'source_task_uuid'))}
    kept, audit = {}, []
    for kind, row in flat:
        if row["uuid"] in excluded:
            audit.append({"kind": kind, **row, "exclusion": "workflow_verification"})
        elif row['uuid'] in withdrawn:
            audit.append({'kind': kind, **row, 'exclusion': 'review_withdrawn'})
        else:
            kept.setdefault(kind, []).append(row)
    return kept, audit


def precise_tip(record):
    """Return a reviewed native point, or None for unsupported endpoints.

    Historical records remain unchanged. Partial path/lane endpoints are
    unknown unless a separate point action explicitly licensed the tip.
    """
    if is_withdrawn(record):
        return None
    xy = record.get("direct_xy")
    if record.get("direct_state") != "direct_visible" or xy is None:
        return None
    if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
        return None
    basis = record.get('tip_review_basis') or {}
    preserved_full_apex = (
        record.get('tip_source') == 'reviewed_full_path_apex'
        and basis.get('kind') == 'preserved_reviewed_full_path_apex'
        and basis.get('review_origin') == 'human'
        and basis.get('path_complete') is True
        and bool(basis.get('source_project')) and bool(basis.get('observation_uuid'))
        and isinstance(basis.get('revision'), int) and basis['revision'] > 0
        and basis.get('direct_xy') == list(xy))
    if (record.get("path_xy") and not record.get("path_complete")
            and record.get("task_type") not in ("apex", "review_tip")
            and record.get("tip_source") != "explicit_point"
            and not preserved_full_apex):
        return None
    return [float(xy[0]), float(xy[1])]
