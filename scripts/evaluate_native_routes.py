"""Evaluate model-only current paths against a frozen, genuinely reviewed panel.

Missing reviews produce an explicit failed report. The report cannot
license measurement until every declared route and hidden-tip case passes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.route_quality import (
    RoutePolicy, algorithm_fingerprint, compare_validation_cases,
    geometry_fingerprint, validation_failures, verified_fit_panels)


def main():
    from tubetracker.analysis_contracts import route_model_bindings
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--panel', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--body-fit-panel', default=str(ROOT/'runs/prototypes/v30/rev14_native_body_panel'),
                   help='Path for historical body checkpoints that record the panel hash without its location')
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    files = {'analysis': Path(a.analysis).resolve(),
             'observations': Path(a.snapshot).resolve()/'observations.json',
             'panel': Path(a.panel).resolve()}
    analysis = json.loads(files['analysis'].read_text())
    panel = json.loads(files['panel'].read_text())
    if panel.get('schema') != 'tubetracker.route_validation_panel.v1':
        raise ValueError('a frozen route validation panel is required')
    if not panel.get('frozen_utc'):
        raise ValueError('panel must be frozen before evaluation')
    observations = json.loads(files['observations'].read_text())
    fits = set()
    for name, fit_path in verified_fit_panels(analysis, body_fit_panel=a.body_fit_panel).items():
        files[name] = fit_path
        fits.update((c['movie'], int(c['frame'])) for c in json.loads(fit_path.read_text())['cases'])
    policy = RoutePolicy(**analysis['request'].get('route_policy', {}))
    report = {'schema': 'tubetracker.route_validation.v1', 'scope': panel['scope'],
        'inputs': {'models': route_model_bindings(analysis['inputs']['pixel']),
                   'movie_sha256': analysis['movie_sha256'],
                   'algorithm': algorithm_fingerprint(), 'policy': asdict(policy),
                   'geometry': geometry_fingerprint(analysis['owners'], analysis['request'].get('roi_xyxy'),
                                                    analysis['request'].get('body_extent', 'grain_crop'),
                                                    analysis['request'].get('body_normalization_context', 'per_tile'),
                                                    grain_geometry=analysis['inputs']['pixel'].get('grain_geometry'))},
        'evidence_files': {k: {'path': str(v), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()}
                           for k, v in files.items()},
        'cases': compare_validation_cases(analysis, observations, panel, fits),
        'limits': ('Scoped development validation of reviewed-mask-assisted model routes; '
                   'seed frames and native training frames excluded; no whole-movie generalization claim.'
                   if analysis['inputs']['pixel'].get('body_assistance') else
                   'Scoped development validation of model-only routes, with source frames excluded from fitting; no whole-movie generalization claim.')}
    report['failures'] = validation_failures(report)
    report['passed'] = not report['failures']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(out), 'passed': report['passed'], 'failures': report['failures'],
                      'missing_genuine_reviews': [c['observation_id'] for c in report['cases'] if not c.get('truth')]}), flush=True)


if __name__ == '__main__':
    main()
