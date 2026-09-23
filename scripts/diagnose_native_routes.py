"""Diagnose body, cap, route and ownership stages from existing immutable evidence.

Reserved panel references must be genuine. --assisted-replay separately
exercises the machinery on already composed human-supported fitting examples.
Neither mode creates annotations or licenses automatic measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.native_caps import file_hash
from prototypes.v30_video_apex.route_diagnostics import diagnose_reference_route, candidate_readout
from prototypes.v30_video_apex.route_evidence import generate_owned_routes
from prototypes.v30_video_apex.route_quality import (
    RoutePolicy, algorithm_fingerprint, compare_validation_cases, verified_fit_panels)
from tubetracker.evidence_cache import BodyArrayStore


def render_case(path, probability, origin, reference, selected, diagnosis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h, w = probability.shape
    axes[0].imshow(probability, cmap='gray', vmin=0, vmax=1,
                   extent=[origin[0]-.5, origin[0]+w-.5, origin[1]+h-.5, origin[1]-.5])
    truth = np.asarray(reference)
    axes[0].plot(truth[:,0],truth[:,1],color='lime',label='reviewed current path')
    model = np.asarray((selected or {}).get('current_path_xy') or [])
    if model.size:
        axes[0].plot(model[:,0],model[:,1],color='cyan',linestyle='--',label='model selection')
    visible = np.vstack((truth, model)) if model.size else truth
    lo, hi = visible.min(axis=0)-12, visible.max(axis=0)+12
    axes[0].set_xlim(max(origin[0]-.5,lo[0]),min(origin[0]+w-.5,hi[0]))
    axes[0].set_ylim(min(origin[1]+h-.5,hi[1]),max(origin[1]-.5,lo[1]))
    axes[0].legend(fontsize=8)
    axes[0].set_title('Current-frame body probability / native pixels')
    profile = diagnosis['reference_profile']
    axes[1].plot(profile['arclength_px'],profile['probability'])
    axes[1].axhline(diagnosis['body_on_reference']['policy']['support_probability'],color='gray',linestyle='--')
    for interval in diagnosis['unsupported_intervals']:
        axes[1].axvspan(interval['start_s_px'],interval['end_s_px'],color='red',alpha=.15)
    axes[1].set(xlabel='Distance along reviewed path (px)',ylabel='Body probability',ylim=(-.02,1.02))
    fig.suptitle(f"{diagnosis['owner_id']} / {diagnosis['source_frame']}: {diagnosis['first_failed_stage']}",fontsize=10)
    fig.tight_layout()
    fig.savefig(path,dpi=140)
    plt.close(fig)


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    analysis_path = Path(args.analysis).resolve()
    analysis = json.loads(analysis_path.read_text())
    fits = verified_fit_panels(analysis, body_fit_panel=args.body_fit_panel)
    fit_frames = {(c['movie'],int(c['frame'])) for p in fits.values()
                  for c in json.loads(p.read_text())['cases']}
    recorded_code = {Path(k).name:v for group in ('pixel','inference')
                     for k,v in analysis['inputs'][group]['code'].items()}
    if any(recorded_code.get(k) != v for k,v in algorithm_fingerprint().items()):
        raise ValueError('analysis route/model algorithm is stale; regenerate it before diagnosis')
    pixels_path = Path(analysis['artifacts']['pixels'])
    pixels = json.loads(pixels_path.read_text())
    if pixels['inputs'] != analysis['inputs']['pixel']:
        raise ValueError('pixel artifact does not match this analysis')
    arrays_path = pixels_path.with_suffix('.arrays')
    if not arrays_path.is_dir() or pixels['body_arrays']['schema'] != BodyArrayStore.schema:
        raise ValueError('immutable body evidence is missing or unsupported')
    arrays = BodyArrayStore(arrays_path, pixels['body_arrays']['entries'])
    files = [analysis_path,pixels_path,*fits.values()]
    files += [ROOT/p for p in ('scripts/diagnose_native_routes.py',
              'prototypes/v30_video_apex/route_diagnostics.py', 'tubetracker/evidence_cache.py')]
    files += [ROOT/'prototypes/v30_video_apex'/name for name in algorithm_fingerprint()]
    references = []
    if args.assisted_replay:
        mode = 'known_assisted_fitting_replay'
        for row in analysis['rows']:
            certificate = row.get('path_certificate') or {}
            if not row.get('path_complete') or certificate.get('review_origin') != 'human':
                continue
            references.append({'owner_id':row['owner_id'],'source_frame':row['source_frame'],
                'kind':'full','truth':certificate,'path':row['current_path_xy'],
                'excluded_from_fit_frames':(analysis['movie_id'],row['source_frame']) not in fit_frames})
        if not references:
            raise ValueError('no genuine human-supported complete replay exists in this export')
    else:
        mode = 'reserved_genuine_panel'
        if not args.snapshot:
            raise ValueError('--snapshot is required for reserved panel diagnosis')
        obs_path = Path(args.snapshot)/'observations.json'
        panel_path = Path(args.panel)
        files += [obs_path,panel_path]
        observations, panel = json.loads(obs_path.read_text()),json.loads(panel_path.read_text())
        if (panel.get('schema') != 'tubetracker.route_validation_panel.v1'
                or not panel.get('frozen_utc') or panel['scope']['movie'] != analysis['movie_id']):
            raise ValueError('a matching frozen route panel is required')
        observations_by_id = {o['obs_uuid']:o for o in observations}
        for case in compare_validation_cases(analysis,observations,panel,fit_frames):
            path = observations_by_id[case['observation_id']]['path_xy'] if case['truth'] and case['kind']=='full' else None
            references.append(dict(case,path=path))
    inputs = {str(p.resolve()):file_hash(p) for p in files}
    for checkpoint in ('cap_checkpoint','body_checkpoint'):
        rec = analysis['dependencies']['files'][checkpoint]
        if file_hash(rec['path']) != rec['sha256']:
            raise ValueError('model changed after analysis')
        inputs[rec['path']] = rec['sha256']
    predictions = {(r['owner_id'],r['source_frame']):r for r in analysis['model_without_reviews']}
    owners = {o['id']:o for o in analysis['owners']}
    policy = RoutePolicy(**analysis['request'].get('route_policy',{}))
    generated, reports, plots = {},[],[]
    for reference in references:
        frame, oid = reference['source_frame'],reference['owner_id']
        evidence = pixels['frames'][str(frame)]
        if frame not in generated:
            generated[frame] = {}
            for owner_id, info in evidence['owners'].items():
                body = arrays[info['body_key']]
                entry = arrays.entries[info['body_key']]
                inputs[str((arrays_path/entry['file']).resolve())] = entry['sha256']
                generated[frame][owner_id] = generate_owned_routes(body,info['origin'],owners[owner_id],evidence['caps'],
                    image_size=pixels['image_size'],other_owners=list(owners.values()),policy=policy,
                    movie=analysis['movie_id'],source_frame=frame)
        selected = predictions.get((oid,frame))
        row = {k:v for k,v in reference.items() if k != 'path'}
        row['model_selection'] = {k:(selected or {}).get(k) for k in ('state','tip_xy','selected_candidate_id')}
        candidates = generated[frame][oid]
        if reference.get('path'):
            info = evidence['owners'][oid]
            competitors = [dict(c,owner_id=owner_id) for owner_id,rows in generated[frame].items() for c in rows]
            diagnosis = diagnose_reference_route(arrays[info['body_key']],info['origin'],owners[oid],reference['path'],
                candidates,selected,caps=evidence['caps'],source_frame=frame,other_owners=list(owners.values()),
                image_size=pixels['image_size'],policy=policy,competing_candidates=competitors)
            row.update(status='diagnosed',diagnosis=diagnosis)
            plots.append((arrays[info['body_key']],info['origin'],reference['path'],selected,diagnosis))
        else:
            row.update(status='hidden_state_only' if reference['truth'] and reference['kind']=='hidden' else 'missing_genuine_full_path',
                       candidates=[candidate_readout(c) for c in candidates])
        reports.append(row)
    changed = [p for p,digest in inputs.items() if file_hash(p) != digest]
    if changed:
        raise RuntimeError(f'diagnostic evidence changed during evaluation: {changed}')
    out.mkdir(parents=True,exist_ok=False)
    report = {'schema':'tubetracker.route_stage_diagnosis.v1','mode':mode,'inputs_sha256':inputs,
        'scope':'Post-inference diagnosis only; no annotations, model changes or measurement certificates are produced.',
        'cases':reports,'status_counts':dict(Counter(r['status'] for r in reports)),
        'stage_counts':dict(Counter(r['diagnosis']['first_failed_stage'] for r in reports if r.get('diagnosis')))}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for i,values in enumerate(plots):
        render_case(out/f'reference-{i}.png',*values)
    print(json.dumps({'out':str(out),'mode':mode,'status_counts':report['status_counts'],'stage_counts':report['stage_counts']}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--panel')
    choice.add_argument('--assisted-replay',action='store_true')
    parser.add_argument('--snapshot')
    parser.add_argument('--body-fit-panel',default=str(ROOT/'runs/prototypes/v30/rev14_native_body_panel'))
    parser.add_argument('--out',required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
