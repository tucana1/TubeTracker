"""Audit grain-reference body inference on preserved native mask supervision."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.native_body import (
    TILE, VALID_MARGIN, OwnedBodyReference, load_body_checkpoint,
    predict_owned_body, predict_owned_body_region)
from prototypes.v30_video_apex.native_caps import file_hash
from scripts.train_native_body import PAD, render
from tubetracker.annotation_frames import FrameReader


def metrics(prediction, positive, ordinary, foreign):
    prediction = prediction >= .5
    valid = positive | ordinary | foreign
    return {
        'iou': float((prediction & positive).sum() / max(1, ((prediction | positive) & valid).sum())) if positive.any() else None,
        'recall': float((prediction & positive).sum() / positive.sum()) if positive.any() else None,
        'ordinary_positive_rate': float(prediction[ordinary].mean()) if ordinary.any() else None,
        'foreign_positive_rate': float(prediction[foreign].mean()) if foreign.any() else None}


def aggregate(rows, mode):
    body = [r[mode] for r in rows if r['kind'] == 'owned_body']
    absent = [r[mode] for r in rows if r['kind'] == 'owned_absence']
    result = {
        'mean_body_iou': float(np.mean([r['iou'] for r in body])),
        'min_body_iou': min(r['iou'] for r in body),
        'min_body_recall': min(r['recall'] for r in body),
        'max_foreign_positive_rate': max((r['foreign_positive_rate'] for r in body if r['foreign_positive_rate'] is not None), default=None),
        'max_absence_region_positive_rate': max((r['ordinary_positive_rate'] for r in absent if r['ordinary_positive_rate'] is not None), default=None)}
    result['fit_gate_passed'] = (result['min_body_iou'] >= .9 and result['min_body_recall'] >= .95
        and result['max_foreign_positive_rate'] is not None and result['max_foreign_positive_rate'] <= .05
        and result['max_absence_region_positive_rate'] is not None and result['max_absence_region_positive_rate'] <= .05)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--baseline-source', required=True, help='Preserved pre-change native_body.py')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out, panel = Path(args.out), Path(args.panel)
    if out.exists():
        raise FileExistsError(out)
    doc = json.loads((panel/'panel.json').read_text())
    if file_hash(panel/'panel.npz') != doc['dataset_sha256']:
        raise ValueError('preserved body panel changed')
    snapshot = Path(doc['snapshot'])/'snapshot_manifest.json'
    if file_hash(snapshot) != doc['snapshot_sha256']:
        raise ValueError('preserved body snapshot changed')
    arrays = dict(np.load(panel/'panel.npz'))
    movies = json.loads(snapshot.read_text())['movies']
    files = [Path(args.checkpoint), panel/'panel.json', panel/'panel.npz', snapshot,
             Path(args.baseline_source), Path(__file__), ROOT/'prototypes/v30_video_apex/native_body.py',
             ROOT/'prototypes/v30_video_apex/native_caps.py', ROOT/'scripts/train_native_body.py',
             ROOT/'tubetracker/annotation_frames.py']
    files += [Path(movies[m]['path']) for m in sorted({c['movie'] for c in doc['cases']})]
    inputs = {str(p.resolve()): file_hash(p) for p in files}
    torch.set_num_threads(2)
    model, meta = load_body_checkpoint(args.checkpoint)
    spec = importlib.util.spec_from_file_location('prototypes.v30_video_apex._body_reference_baseline', args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    old_model, _ = baseline.load_body_checkpoint(args.checkpoint)
    rows, readers, frames = [], {}, {}
    predictions = {k: [] for k in ['canonical', 'reference_canonical', 'per_tile_roi', 'reference_roi']}
    try:
        for i, case in enumerate(doc['cases']):
            movie, frame = case['movie'], case['frame']
            if movie not in readers:
                readers[movie] = FrameReader(movies[movie]['path'])
            if (movie, frame) not in frames:
                image = readers[movie].read(frame)
                if not image.exact:
                    raise ValueError(f'inexact source frame {movie}/{frame}')
                frames[(movie, frame)] = cv2.cvtColor(image.frame, cv2.COLOR_BGR2GRAY)
            gray = frames[(movie, frame)]
            owner = {'id': case['owner'], 'grain_native': case['grain'], 'grain_radius_px': case['radius']}
            canonical, origin = predict_owned_body(model, gray, owner)
            old, old_origin = baseline.predict_owned_body(old_model, gray, owner)
            if origin != (np.asarray(case['origin'])+PAD).tolist() or origin != old_origin:
                raise ValueError('body panel and deployment crop origins differ')
            reference = OwnedBodyReference(model, gray, owner)
            replay, _ = reference.predict()
            h, w = gray.shape
            x0, y0 = max(0, origin[0]), max(0, origin[1])
            x1, y1 = min(w, origin[0]+TILE), min(h, origin[1]+TILE)
            regions, details = {}, {}
            for mode, context in [('per_tile_roi', 'per_tile'), ('reference_roi', 'grain_reference')]:
                probability, _, detail = predict_owned_body_region(model, gray, owner, [x0, y0, x1, y1], spatial_context=context)
                regions[mode] = np.zeros((TILE, TILE), np.float32)
                regions[mode][y0-origin[1]:y1-origin[1], x0-origin[0]:x1-origin[0]] = probability
                details[mode] = detail
            pos, ordinary, foreign = [arrays[k][i, PAD:PAD+TILE, PAD:PAD+TILE] for k in ['positive', 'ordinary', 'foreign']]
            values = {'canonical': canonical, 'reference_canonical': replay, **regions}
            row = {'id': case['id'], 'kind': case['kind'], 'movie': movie, 'source_frame': frame,
                   'owner': case['owner'], 'default_matches_preserved_source_bit_exactly': bool(np.array_equal(canonical, old)),
                   'canonical_reference_max_error': float(np.abs(canonical-replay).max()),
                   'canonical_reference_binary_unchanged': bool(np.array_equal(canonical>=.5, replay>=.5)),
                   'reference_roi_interior_max_error': float(np.abs(regions['reference_roi']-replay)[
                       VALID_MARGIN:-VALID_MARGIN, VALID_MARGIN:-VALID_MARGIN].max()),
                   'roi_details': details}
            for mode, value in values.items():
                predictions[mode].append(value)
                row[mode] = metrics(value, pos, ordinary, foreign)
            rows.append(row)
            print(json.dumps({'case': case['id'], 'ious': {k:row[k]['iou'] for k in predictions},
                              'canonical_reference_max_error': row['canonical_reference_max_error']}), flush=True)
    finally:
        for reader in readers.values():
            reader.close()
    changed = [p for p, digest in inputs.items() if file_hash(p) != digest]
    if changed:
        raise RuntimeError(f'body reference audit inputs changed: {changed}')
    result = {'scope': 'Development fit on preserved owned masks and absence regions; no new annotation or generalization claim.',
              'checkpoint': meta['sha256'], 'inputs_sha256': inputs, 'rows': rows,
              'summary': {mode: aggregate(rows, mode) for mode in predictions},
              'default_matches_preserved_source_bit_exactly': all(r['default_matches_preserved_source_bit_exactly'] for r in rows),
              'canonical_reference_max_error': max(r['canonical_reference_max_error'] for r in rows),
              'canonical_reference_binary_unchanged': all(r['canonical_reference_binary_unchanged'] for r in rows)}
    result['reference_preserves_canonical_gate_passed'] = (
        result['default_matches_preserved_source_bit_exactly'] and result['canonical_reference_binary_unchanged']
        and result['canonical_reference_max_error'] <= 1e-5)
    out.mkdir(parents=True, exist_ok=False)
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for mode, values in predictions.items():
        render(doc['cases'], arrays, out/(mode+'.png'), values)
    print(json.dumps({'out': str(out), 'summary': result['summary'],
                      'reference_preserves_canonical_gate_passed': result['reference_preserves_canonical_gate_passed']}), flush=True)


if __name__ == '__main__':
    main()
