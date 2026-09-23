"""Compare body checkpoints on fixed predictions and explicitly named label domains.

The legacy domain is historical evidence only. A score change caused by
removing unlicensed background is not a change in model accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.native_body import TILE, body_input, load_body_checkpoint
from prototypes.v30_video_apex.native_caps import file_hash
from scripts.audit_native_body_reference import aggregate, metrics
from scripts.train_native_body import PAD, render


def matched_domains(old_arrays, old_doc, arrays, doc):
    """Require the scope-only comparison and return its common valid domain."""
    if doc.get('supervision_contract') != 'finalized_self_reviewed_background_foreign_v1':
        raise ValueError('the corrected panel must declare finalized review selectors')
    fields = ('id', 'kind', 'movie', 'frame', 'grain', 'radius', 'origin', 'owner', 'image_size')
    if (len(old_doc['cases']) != len(doc['cases']) or any(
            any(a.get(k) != b.get(k) for k in fields)
            for a, b in zip(old_doc['cases'], doc['cases']))):
        raise ValueError('case order or query geometry changed; not a scope-only comparison')
    for key in ('pixels', 'positive', 'foreign'):
        if not np.array_equal(old_arrays[key], arrays[key]):
            raise ValueError(f'{key} changed; not a scope-only comparison')
    if old_arrays['ordinary'].shape != arrays['ordinary'].shape:
        raise ValueError('background shape changed')
    if np.any(arrays['ordinary'] & ~old_arrays['ordinary']):
        raise ValueError('the repaired legacy scope must not add background labels')
    domains = {
        'legacy': {k: old_arrays[k] for k in ('positive', 'ordinary', 'foreign')},
        'corrected': {k: arrays[k] for k in ('positive', 'ordinary', 'foreign')},
        'common_reviewed': {k: arrays[k] & old_arrays[k] for k in ('positive', 'ordinary', 'foreign')},
    }
    for name, domain in domains.items():
        if any(np.any(domain[a] & domain[b]) for a, b in (
                ('positive', 'ordinary'), ('positive', 'foreign'), ('ordinary', 'foreign'))):
            raise ValueError(f'{name} supervision channels overlap')
    return domains


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    old_panel, panel = Path(args.legacy_panel), Path(args.panel)
    old_doc, doc = [json.loads((p/'panel.json').read_text()) for p in (old_panel, panel)]
    for directory, manifest in ((old_panel, old_doc), (panel, doc)):
        if file_hash(directory/'panel.npz') != manifest['dataset_sha256']:
            raise ValueError(f'body panel changed: {directory}')
    old_arrays, arrays = [dict(np.load(p/'panel.npz')) for p in (old_panel, panel)]
    domains = matched_domains(old_arrays, old_doc, arrays, doc)
    files = [old_panel/'panel.json', old_panel/'panel.npz', panel/'panel.json', panel/'panel.npz']
    files += [Path(p) for p in args.checkpoint]
    files += [ROOT/p for p in (
        'scripts/audit_native_body_domains.py', 'scripts/audit_native_body_reference.py',
        'scripts/train_native_body.py', 'prototypes/v30_video_apex/native_body.py',
        'prototypes/v30_video_apex/native_caps.py')]
    inputs = {str(p.resolve()): file_hash(p) for p in files}
    counts = {name: {k: int(v.sum()) for k, v in domain.items()} for name, domain in domains.items()}
    torch.set_num_threads(2)
    checkpoints, all_predictions = [], {}
    for index, path in enumerate(args.checkpoint):
        model, metadata = load_body_checkpoint(path)
        predictions, rows = [], []
        for i, case in enumerate(doc['cases']):
            crop = (slice(PAD, PAD+TILE), slice(PAD, PAD+TILE))
            gray = arrays['pixels'][i][crop]
            origin = np.asarray(case['origin'])+PAD
            x = body_input(gray, origin, case['grain'], case['radius'], model.image_gain)
            with torch.no_grad():
                probability = torch.sigmoid(model(torch.from_numpy(x)[None]))[0, 0].numpy()
            predictions.append(probability)
            row = {k: case[k] for k in ('id', 'kind', 'movie', 'frame', 'owner')}
            for name, domain in domains.items():
                row[name] = metrics(probability, *[domain[k][i][crop]
                    for k in ('positive', 'ordinary', 'foreign')])
            removed = (old_arrays['ordinary'][i] & ~arrays['ordinary'][i])[crop]
            row['unlicensed_background_probability_diagnostic'] = (
                float(probability[removed].mean()) if removed.any() else None)
            rows.append(row)
        result = {'checkpoint': str(Path(path).resolve()), 'sha256': metadata['sha256'],
                  'normalization': model.normalization, 'image_gain': model.image_gain,
                  'summary': {name: aggregate(rows, name) for name in domains}, 'rows': rows}
        checkpoints.append(result)
        all_predictions[f'checkpoint_{index}'] = np.stack(predictions)
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
    changed = [p for p, digest in inputs.items() if file_hash(p) != digest]
    if changed:
        raise RuntimeError(f'audit inputs changed: {changed}')
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'predictions.npz', **all_predictions)
    result = {
        'schema': 'tubetracker.body_domain_comparison.v1',
        'scope': 'Preserved development fitting cases only; no held-out route or movie accuracy claim.',
        'interpretation': 'Each checkpoint is inferred once per image/query. Legacy, corrected and common-domain metrics use exactly those same predictions. A within-checkpoint score change is solely an evaluation-domain change.',
        'inputs_sha256': inputs, 'legacy_panel': str(old_panel.resolve()), 'corrected_panel': str(panel.resolve()),
        'supervised_case_pixels': counts, 'checkpoints': checkpoints,
        'predictions_sha256': file_hash(out/'predictions.npz'),
    }
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for i in range(len(checkpoints)):
        render(doc['cases'], arrays, out/f'checkpoint-{i}-corrected-domain.png', all_predictions[f'checkpoint_{i}'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-panel', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
