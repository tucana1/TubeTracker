"""Current-frame route evidence and a scoped, independently checked release gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class RoutePolicy:
    support_probability: float = .5
    minimum_mean_support: float = .8
    minimum_supported_fraction: float = .95
    maximum_unsupported_gap_px: float = 2.
    crop_guard_px: float = 2.
    minimum_cap_probability: float = .5
    root_selection: str = 'declared'

    def __post_init__(self):
        if self.root_selection not in ('declared', 'current_body_rim'):
            raise ValueError('root_selection must be declared or current_body_rim')
        for key, value in asdict(self).items():
            if key == 'root_selection':
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError('route policy values must be finite and nonnegative')
            if key not in ('maximum_unsupported_gap_px', 'crop_guard_px') and not 0 < value <= 1:
                raise ValueError('route support thresholds must lie in (0, 1]')


def sample_path(points, spacing=.5):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('route requires a finite native polyline')
    chunks = [points[:1]]
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing)))
        chunks.append(a + np.arange(1, n+1)[:, None]/n * (b-a))
    return np.concatenate(chunks)


def inspect_route(probability, origin, path, owner, cap, *, image_size=None,
                  other_owners=(), policy=None):
    policy = policy or RoutePolicy()
    points = sample_path(path)
    local = points - np.asarray(origin)
    h, w = probability.shape
    support = map_coordinates(np.asarray(probability, dtype=float), [local[:, 1], local[:, 0]],
                              order=1, mode='constant', cval=0., prefilter=False)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(steps.sum())
    weak = (support[:-1] < policy.support_probability) | (support[1:] < policy.support_probability)
    longest = current = 0.
    for gap, step in zip(weak, steps):
        current = current + float(step) if gap else 0.
        longest = max(longest, current)
    fraction = float(1 - steps[weak].sum()/length) if length else 0.
    mean = float(np.sum((support[:-1]+support[1:])*.5*steps)/length) if length else 0.
    clearance = float(np.min(np.stack([local[:, 0], local[:, 1], w-1-local[:, 0], h-1-local[:, 1]])))
    root = owner.get('attachment_native')
    root_error = float(np.linalg.norm(points[0]-root)) if root is not None else None
    root_basis = 'unverified'
    root_ok = False
    if owner.get('attachment_verified') and root_error is not None and root_error <= 1.:
        root_basis, root_ok = 'stored_attachment', True
    elif root is None and owner.get('grain_geometry_complete', True) \
            and not owner.get('root_review_required'):
        # rev14 sparse: an owner with NO declared attachment (census-added
        # grains) may root on an evidence-derived rim start: the route
        # generator anchors such routes at the highest-support point on the
        # grain rim, so the root is verified by geometry + body evidence,
        # never by an invented coordinate. The basis is recorded.
        # rev15 ownership correction: owners whose exit is under review
        # (root_review_required, e.g. corrected 0003/3756) NEVER take this
        # path — their starts cannot be assumed to be actual exits.
        centre = np.asarray(owner['grain_native'], float)
        radius = float(owner.get('grain_radius_px', 13.0))
        distance = float(np.linalg.norm(np.asarray(points[0], float) - centre))
        if (0.85 * radius <= distance <= 1.15 * radius
                and float(support[0]) >= policy.support_probability):
            root_basis, root_ok = 'evidence_rim_start', True
    tip_error = float(np.linalg.norm(points[-1]-cap['tip_xy']))
    image_clearance = None
    if image_size:
        iw, ih = image_size
        image_clearance = float(np.min(np.stack([points[:, 0], points[:, 1], iw-1-points[:, 0], ih-1-points[:, 1]])))
    intrusions = []
    distance_from_start = np.r_[0., np.cumsum(steps)]
    grains = {g['id']: g for g in [owner, *other_owners]}
    for grain in grains.values():
        interior = np.linalg.norm(points-np.asarray(grain['grain_native']), axis=1) < .72*float(grain.get('grain_radius_px', 13))
        if grain['id'] == owner['id']:
            interior &= distance_from_start > 3  # allow the reviewed exit's discretization
        if interior.any():
            intrusions.append(grain['id'])
    failures = []
    if owner.get('grain_pose') and not owner['grain_pose'].get('usable_for_model_geometry'):
        failures.append('grain_pose_unresolved')
        root_basis, root_ok = 'unverified', False
    if not root_ok: failures.append('root_not_verified')
    if owner.get('root_review_required') and not owner.get('attachment_verified'):
        failures.append('exit_review_required')
    if not root_ok and not owner.get('grain_geometry_complete', True):
        failures.append('grain_geometry_incomplete')
    if tip_error > 1: failures.append('path_does_not_reach_current_cap')
    if float(cap.get('probability', 0)) < policy.minimum_cap_probability: failures.append('cap_not_accepted')
    if mean < policy.minimum_mean_support: failures.append('weak_current_body_evidence')
    if fraction < policy.minimum_supported_fraction: failures.append('insufficient_supported_extent')
    if longest > policy.maximum_unsupported_gap_px: failures.append('unsupported_gap')
    if clearance < policy.crop_guard_px: failures.append('reaches_body_crop_edge')
    if image_clearance is not None and image_clearance < policy.crop_guard_px: failures.append('reaches_movie_edge')
    if intrusions: failures.append('enters_grain_interior')
    return {'policy': asdict(policy), 'geometrically_supported': not failures, 'failures': failures,
            'root_basis': root_basis,
            'mean_support': mean, 'supported_fraction': fraction, 'longest_unsupported_gap_px': longest,
            'unsupported_length_px': float(steps[weak].sum()), 'crop_clearance_px': clearance,
            'image_clearance_px': image_clearance, 'root_error_px': root_error,
            'cap_endpoint_error_px': tip_error, 'grain_interior_conflicts': intrusions,
            'proposal_length_px': length,
            'path_bounds_xyxy': [float(points[:, 0].min()), float(points[:, 1].min()),
                                 float(points[:, 0].max()), float(points[:, 1].max())],
            'evidence_frame': int(cap['source_frame']) if 'source_frame' in cap else None}


def algorithm_fingerprint():
    root = Path(__file__).resolve().parents[2]
    names = ('route_quality.py', 'route_evidence.py', 'state_solver.py', 'native_caps.py', 'native_body.py')
    return {name: hashlib.sha256((root/'prototypes/v30_video_apex'/name).read_bytes()).hexdigest()
            for name in names}


def validation_failures(report):
    """Recompute the declared gate; a bare passed=true cannot license lengths."""
    if not isinstance(report, Mapping):
        return ['malformed_validation_report']
    errors = []
    cases = report.get('cases', [])
    if not isinstance(cases, list) or any(not isinstance(c, Mapping) for c in cases):
        return ['malformed_validation_cases']
    full = [c for c in cases if c.get('kind') == 'full']
    hidden = [c for c in cases if c.get('kind') == 'hidden']
    if len(full) < 3 or len({c.get('owner_id') for c in full}) < 2 or len({c.get('source_frame') for c in full}) < 2:
        errors.append('need_three_full_paths_across_two_owners_and_frames')
    if len(hidden) < 2: errors.append('need_two_scoped_hidden_tip_cases')
    scope = report.get('scope', {})
    if not isinstance(scope, Mapping):
        errors.append('malformed_validation_scope')
        scope = {}
    interval = scope.get('frame_interval', [])
    full_owners = {c.get('owner_id') for c in full}
    full_frames = [c['source_frame'] for c in full if c.get('source_frame') is not None]
    owner_ids = scope.get('owner_ids')
    if (not isinstance(owner_ids, (list, tuple)) or not owner_ids
            or any(not isinstance(oid, str) for oid in owner_ids)
            or not set(owner_ids).issubset(full_owners)):
        errors.append('scope_contains_unvalidated_owners')
    finite_number = lambda value: isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(finite_number(v) for v in interval)
            or not full_frames or not all(finite_number(v) for v in full_frames)
            or interval[0] > interval[1]
            or interval[0] < min(full_frames) or interval[1] > max(full_frames)):
        errors.append('scope_exceeds_validated_interval')
    for c in cases:
        if (not isinstance(c.get('truth'), Mapping) or c['truth'].get('review_origin') != 'human'
                or not c.get('excluded_from_fit_frames')):
            errors.append('truth_missing_workflow_or_used_for_fit')
            continue
        if c.get('kind') == 'full':
            limits = {'tip_error_px': 5., 'path_mean_error_px': 2., 'path_max_error_px': 5.,
                      'length_relative_error': .1}
            if not c.get('geometry_supported') or any(
                not finite_number(c.get(k)) or not 0 <= c[k] <= limit for k, limit in limits.items()):
                errors.append('full_route_accuracy_failed')
        elif c.get('kind') == 'hidden':
            if not c.get('tip_withheld'): errors.append('hidden_tip_emitted')
        elif c.get('kind') == 'visible_tip':
            error = c.get('tip_error_px')
            if c.get('tip_withheld') or not finite_number(error) or not 0 <= error <= 5.:
                errors.append('visible_tip_accuracy_failed')
        else:
            errors.append('unsupported_validation_case')
    return sorted(set(errors))


def geometry_fingerprint(owners, roi, body_extent, body_normalization_context='per_tile', *,
                         grain_geometry=None):
    geometry = {'roi_xyxy': roi, 'body_extent': body_extent,
                'body_normalization_context': body_normalization_context, 'owners': [
        {k: owner.get(k) for k in ('id', 'grain_native', 'grain_radius_px',
                                  'attachment_native', 'attachment_verified', 'grain_geometry_complete')}
        for owner in sorted(owners, key=lambda o: o['id'])]}
    if grain_geometry:
        geometry['frame_geometry_identity'] = grain_geometry['identity']
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def compare_validation_cases(analysis, observations, panel, fit_frames):
    from scipy.spatial import cKDTree
    from tubetracker.review_semantics import is_withdrawn, is_workflow_record
    from tubetracker.analysis_contracts import route_model_bindings
    fit_frames = set(fit_frames)
    pixels = analysis.get('inputs', {}).get('pixel', {})
    if pixels.get('body_assistance'):
        route_model_bindings(pixels)  # a stale or edited seed plan is not evidence
        plan = pixels['body_assistance']
        fit_frames.update((plan['movie'], int(seed['frame'])) for seed in plan['seeds'])
    obs_by_id = {o['obs_uuid']: o for o in observations}
    predictions = {(r['owner_id'], r['source_frame']): r for r in analysis['model_without_reviews']}
    def eligible_truth(truth, spec):
        owner = (truth or {}).get('owner_uuid') or (truth or {}).get('tube_uuid')
        return bool(truth and not is_workflow_record(truth) and not is_withdrawn(truth)
            and truth.get('review_origin', 'human') == 'human'
            and truth.get('movie') == panel['scope']['movie'] and owner == spec['owner_id']
            and truth.get('source_frame') == spec['source_frame']
            and ('observation_revision' not in spec
                 or truth.get('obs_revision', 1) == spec['observation_revision']))
    result = []
    for spec in panel['cases']:
        case = dict(spec)
        truth = obs_by_id.get(spec['observation_id'])
        row = predictions.get((spec['owner_id'], spec['source_frame']))
        movie = panel['scope']['movie']
        case['excluded_from_fit_frames'] = (movie, spec['source_frame']) not in fit_frames
        eligible = eligible_truth(truth, spec)
        if spec['kind'] == 'full':
            eligible = bool(eligible and truth.get('path_complete') and len(truth.get('path_xy') or []) >= 2
                            and truth.get('direct_state') == 'direct_visible')
        elif spec['kind'] == 'hidden':
            eligible = bool(eligible and truth.get('direct_state') in ('not_directly_visible', 'no_tube_visible'))
        case['truth'] = ({'observation_id': truth['obs_uuid'], 'revision': truth.get('obs_revision', 1),
                          **({'project': truth['project']} if truth.get('project') else {}),
                          'review_origin': 'human'} if eligible else None)
        if spec['kind'] == 'hidden':
            case['tip_withheld'] = bool(row and row.get('tip_xy') is None)
        else:
            case.update(tip_error_px=None, path_mean_error_px=None, path_max_error_px=None,
                        length_relative_error=None, geometry_supported=False)
            if eligible and row and row.get('tip_xy') and len(row.get('current_path_xy') or []) >= 2:
                predicted = sample_path(row['current_path_xy'])
                actual = sample_path(truth['path_xy'])
                distance = np.r_[cKDTree(actual).query(predicted)[0], cKDTree(predicted).query(actual)[0]]
                pl = float(np.linalg.norm(np.diff(predicted, axis=0), axis=1).sum())
                tl = float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum())
                case.update(tip_error_px=float(np.linalg.norm(np.asarray(row['tip_xy'])-actual[-1])),
                            path_mean_error_px=float(distance.mean()), path_max_error_px=float(distance.max()),
                            length_relative_error=abs(pl-tl)/max(tl, 1e-6),
                            geometry_supported=bool((row.get('route_evidence') or {}).get('geometry', {}).get('geometrically_supported')))
        result.append(case)
    # rev14 P1: typed visible-tip regressions -- scored, never silently
    # excluded. The panel keeps them separate from the full/hidden cases.
    from tubetracker.review_semantics import precise_tip
    for spec in panel.get('visible_tip_cases', []):
        case = dict(spec)
        truth = obs_by_id.get(spec['observation_id'])
        row = predictions.get((spec['owner_id'], spec['source_frame']))
        movie = panel['scope']['movie']
        case['excluded_from_fit_frames'] = (movie, spec['source_frame']) not in fit_frames
        eligible = eligible_truth(truth, spec)
        tip = precise_tip(truth) if eligible else None
        eligible = eligible and tip is not None
        case['truth'] = ({'observation_id': truth['obs_uuid'],
                          'revision': truth.get('obs_revision', 1),
                          **({'project': truth['project']} if truth.get('project') else {}),
                          'review_origin': 'human'} if eligible else None)
        case['tip_withheld'] = bool(row and row.get('tip_xy') is None)
        case['tip_error_px'] = (float(np.linalg.norm(np.asarray(row['tip_xy']) - np.asarray(tip)))
                                if tip is not None and row and row.get('tip_xy') else None)
        case['passed'] = bool(case['tip_error_px'] is not None
                              and case['tip_error_px'] <= 5.0)
        result.append(case)
    return result


def verified_fit_panels(analysis, *, body_fit_panel=None):
    """Resolve label panels from the exact models which produced the analysis."""
    import torch
    panels = {}
    for name in ('cap', 'body'):
        checkpoint = analysis.get('dependencies', {}).get('files', {}).get(name + '_checkpoint')
        expected = analysis['inputs']['pixel'][name + '_checkpoint_sha256']
        if not checkpoint or checkpoint.get('sha256') != expected:
            raise ValueError('analysis lacks checkpoint provenance: ' + name)
        raw = Path(checkpoint['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('analysis checkpoint changed: ' + name)
        header = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
        training = header['manifest'].get('inputs', header['manifest'])
        directory = training.get('panel') or (body_fit_panel if name == 'body' else None)
        if not directory:
            raise ValueError('checkpoint lacks a verifiable training-panel location: ' + name)
        panel = Path(directory)/'panel.json'
        if hashlib.sha256(panel.read_bytes()).hexdigest() != training.get('panel_sha256'):
            raise ValueError('training panel changed: ' + name)
        panels[name + '_fit_panel'] = panel.resolve()
    return panels


def load_route_validation(path, *, model_hashes, geometry_hash, movie_sha256, policy=None,
                           review_records=()):
    from tubetracker.analysis_contracts import route_model_bindings
    if not path:
        return None
    report = json.loads(Path(path).read_text())
    if report.get('schema') != 'tubetracker.route_validation.v1':
        raise ValueError('unsupported route validation schema')
    expected = {'models': model_hashes, 'algorithm': algorithm_fingerprint(), 'geometry': geometry_hash,
                'movie_sha256': movie_sha256,
                'policy': asdict(policy or RoutePolicy())}
    if any(report.get('inputs', {}).get(k) != value for k, value in expected.items()):
        raise ValueError('route validation does not match current models, policy or algorithm')
    files = report.get('evidence_files', {})
    if not {'analysis', 'observations', 'panel', 'cap_fit_panel', 'body_fit_panel'}.issubset(files):
        raise ValueError('route validation evidence is incomplete')
    evidence = {}
    for name, rec in files.items():
        raw = Path(rec['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != rec.get('sha256'):
            raise ValueError('route validation evidence changed: ' + name)
        evidence[name] = json.loads(raw)
    panel = evidence['panel']
    if panel.get('schema') != 'tubetracker.route_validation_panel.v1' or not panel.get('frozen_utc'):
        raise ValueError('a frozen route validation panel is required')
    if evidence['analysis'].get('movie_sha256') != movie_sha256:
        raise ValueError('route validation belongs to different movie content')
    verified = verified_fit_panels(evidence['analysis'],
        body_fit_panel=Path(files['body_fit_panel']['path']).parent)
    for name, source in verified.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != files[name]['sha256']:
            raise ValueError('validation training panel does not match checkpoint: ' + name)
    fit_frames = {(c['movie'], int(c['frame'])) for name in ('cap_fit_panel', 'body_fit_panel')
                  for c in evidence[name]['cases']}
    actual = compare_validation_cases(evidence['analysis'], evidence['observations'], evidence['panel'], fit_frames)
    if actual != report.get('cases') or evidence['panel']['scope'] != report.get('scope'):
        raise ValueError('route validation metrics do not match the source evidence')
    from tubetracker.review_semantics import is_withdrawn
    from tubetracker.analysis_contracts import _review_project
    for case in actual:
        truth = case.get('truth') or {}
        for record in review_records:
            if (not record.get('live_review') or record.get('movie') != report['scope']['movie']
                    or record.get('source_id') != truth.get('observation_id')):
                continue
            if truth.get('project') and _review_project(truth['project']) != _review_project(record.get('source_project')):
                continue
            if is_withdrawn(record) or record.get('source_revision', 1) != truth.get('revision', 1):
                raise ValueError('route validation review was withdrawn or revised; rebuild validation')
    a = evidence['analysis']
    actual_models = route_model_bindings(a['inputs']['pixel'])
    actual_geometry = geometry_fingerprint(a['owners'], a['request'].get('roi_xyxy'),
        a['request'].get('body_extent', 'grain_crop'), a['request'].get('body_normalization_context', 'per_tile'),
        grain_geometry=a['inputs']['pixel'].get('grain_geometry'))
    actual_code = {Path(k).name: v for component in ('pixel', 'inference')
                   for k, v in a['inputs'][component]['code'].items()}
    if (actual_models != model_hashes or actual_geometry != geometry_hash
            or any(actual_code.get(k) != v for k, v in expected['algorithm'].items())
            or a['inputs']['inference'].get('route_policy') != expected['policy']):
        raise ValueError('route validation analysis was produced by different inputs')
    if report['scope'].get('roi_xyxy') != a['request'].get('roi_xyxy') or report['scope'].get('movie') != a['movie_id']:
        raise ValueError('route validation field does not match its analysis')
    failures = validation_failures(report)
    if failures:
        raise ValueError('route validation failed: ' + ', '.join(failures))
    return dict(report, validation_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def route_certificate(diagnostics, owner, cap, validation, *, movie, frame):
    if not diagnostics['geometrically_supported']:
        return None, list(diagnostics['failures'])
    if validation is None:
        return None, ['full_route_validation_missing']
    scope = validation.get('scope', {})
    interval = scope.get('frame_interval', [])
    if (movie != scope.get('movie') or owner['id'] not in scope.get('owner_ids', [])
            or len(interval) != 2 or not interval[0] <= frame <= interval[1]):
        return None, ['outside_validated_movie_owner_or_interval']
    roi = scope.get('roi_xyxy')
    if not roi or not (roi[0] <= cap['tip_xy'][0] < roi[2] and roi[1] <= cap['tip_xy'][1] < roi[3]):
        return None, ['outside_validated_field']
    x0, y0, x1, y1 = diagnostics['path_bounds_xyxy']
    if not (roi[0] <= x0 <= x1 < roi[2] and roi[1] <= y0 <= y1 < roi[3]):
        return None, ['route_leaves_validated_field']
    return {'kind': 'validated_current_owned_route', 'movie': movie, 'source_frame': frame,
            'owner_id': owner['id'], 'cap_id': cap['cap_id'],
            'validation_sha256': validation['validation_sha256'],
            'geometry_evidence': diagnostics, 'review_origin': 'model'}, []
