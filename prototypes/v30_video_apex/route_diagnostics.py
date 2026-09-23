"""Locate a route failure using reviewed geometry without supplying it to inference."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .route_quality import RoutePolicy, inspect_route, sample_path
from .state_solver import _score


def route_errors(path, reference, tip=None):
    if len(path or []) < 2:
        return None
    points, truth = sample_path(path), sample_path(reference)
    distance = np.r_[cKDTree(truth).query(points)[0], cKDTree(points).query(truth)[0]]
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    true_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    result = {
        'tip_error_px': float(np.linalg.norm(np.asarray(tip if tip is not None else points[-1])-truth[-1])),
        'path_mean_error_px': float(distance.mean()), 'path_max_error_px': float(distance.max()),
        'length_relative_error': abs(length-true_length)/max(true_length, 1e-6),
    }
    result['matches_reference'] = all(result[k] <= limit for k, limit in (
        ('tip_error_px', 5.), ('path_mean_error_px', 2.), ('path_max_error_px', 5.),
        ('length_relative_error', .1)))
    return result


def candidate_readout(candidate):
    cap, route = _score(dict(candidate, state='present'))
    return {
        'candidate_id': candidate['candidate_id'], 'cap_id': candidate.get('cap_id'),
        'tip_xy': candidate.get('tip_xy'), 'cap_probability': candidate.get('cap_probability'),
        'route_probability': candidate.get('route_probability'),
        'score_components': {'cap_logit': cap, 'weighted_route_logit': route,
                             'current_evidence_total': cap+route},
    }


def diagnose_reference_route(probability, origin, owner, reference_path, candidates, selected,
                             *, caps, source_frame, other_owners=(), image_size=None,
                             policy=None, competing_candidates=()):
    """Report body support, proposal availability and model-only selection.

    The oracle endpoint below is used only to sample the reviewed path.
    It is never inserted into the model cap pool or used as a certificate.
    """
    if selected and (selected.get('provenance') == 'human-corrected'
                     or selected.get('as_inference_constraint') or selected.get('human_constraint')):
        raise ValueError('diagnose the model-only selection, not a human-corrected result')
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError('diagnosis requires a finite current-frame native body map')
    policy = policy or RoutePolicy()
    points = sample_path(reference_path)
    oracle = {'tip_xy': points[-1].tolist(), 'probability': 1., 'source_frame': source_frame}
    body = inspect_route(probability, origin, reference_path, owner, oracle,
                         image_size=image_size, other_owners=other_owners, policy=policy)
    local = points-np.asarray(origin)
    support = map_coordinates(probability, [local[:,1], local[:,0]], order=1,
                              mode='constant', cval=0., prefilter=False)
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    weak = (support[:-1] < policy.support_probability) | (support[1:] < policy.support_probability)
    intervals, start = [], None
    for i in range(len(weak)+1):
        if i < len(weak) and weak[i]:
            if start is None:
                start = i
        elif start is not None:
            intervals.append({'start_s_px': float(arclength[start]), 'end_s_px': float(arclength[i]),
                              'length_px': float(arclength[i]-arclength[start]),
                              'start_xy': points[start].tolist(), 'end_xy': points[i].tolist()})
            start = None
    rows = []
    for candidate in candidates:
        rows.append({**candidate_readout(candidate),
                     'errors': route_errors(candidate.get('current_path_xy'), reference_path,
                                            candidate.get('tip_xy'))})
    matches = [r['candidate_id'] for r in rows if r['errors'] and r['errors']['matches_reference']]
    selected_errors = route_errors((selected or {}).get('current_path_xy'), reference_path,
                                   (selected or {}).get('tip_xy'))
    nearest_cap = min((float(np.linalg.norm(np.asarray(c['tip_xy'])-points[-1])) for c in caps), default=None)
    cap_ids = {r['cap_id'] for r in rows}
    competition = [{**candidate_readout(c), 'owner_id': c['owner_id']}
                   for c in competing_candidates if c.get('cap_id') in cap_ids
                   and c.get('owner_id') != owner['id']]
    if body['crop_clearance_px'] < policy.crop_guard_px:
        stage = 'field_or_crop_coverage'
    elif not owner.get('attachment_verified') or body['root_error_px'] is None:
        stage = 'root_not_verified'
    elif body['root_error_px'] > 1.:
        stage = 'reviewed_root_mismatch'
    elif body['grain_interior_conflicts']:
        stage = 'grain_geometry_conflict'
    elif any(f in body['failures'] for f in ('weak_current_body_evidence',
              'insufficient_supported_extent', 'unsupported_gap')):
        stage = 'body_support'
    elif nearest_cap is None or nearest_cap > 5.:
        stage = 'cap_proposal'
    elif not matches:
        stage = 'route_construction'
    elif not selected_errors or not selected_errors['matches_reference']:
        stage = 'joint_selection'
    else:
        stage = 'route_matches_reference'
    return {
        'scope': 'Diagnostic only. Reviewed geometry is sampled after inference, never supplied as model input.',
        'owner_id': owner['id'], 'source_frame': int(source_frame), 'first_failed_stage': stage,
        'body_on_reference': body, 'unsupported_intervals': intervals,
        'reference_profile': {'arclength_px': arclength.tolist(), 'probability': support.tolist()},
        'nearest_model_cap_error_px': nearest_cap, 'candidate_count': len(rows),
        'matching_candidate_ids': matches, 'candidates': rows,
        'selected_candidate_id': (selected or {}).get('selected_candidate_id'),
        'selected_errors': selected_errors, 'competing_owners': competition,
        'score_note': 'Current evidence scores are uncalibrated. Offline transitions and cap exclusivity also affect joint selection.',
    }
