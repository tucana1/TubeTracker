"""Compare native body crop phase and context without treating predictions as truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.native_body import (
    VALID_MARGIN, TILE, POOLING_LATTICE, OwnedBodyReference, load_body_checkpoint,
    predict_owned_body, predict_owned_body_region)
from prototypes.v30_video_apex.native_caps import file_hash
from tubetracker.annotation_frames import FrameReader


def compare(a, ao, b, bo, image_size):
    """Compare only equal native pixels with sufficient context in both inputs."""
    w, h = image_size
    lx, ty = max(0, ao[0]+VALID_MARGIN, bo[0]+VALID_MARGIN), max(0, ao[1]+VALID_MARGIN, bo[1]+VALID_MARGIN)
    rx, by = min(w, ao[0]+TILE-VALID_MARGIN, bo[0]+TILE-VALID_MARGIN), min(h, ao[1]+TILE-VALID_MARGIN, bo[1]+TILE-VALID_MARGIN)
    if lx >= rx or ty >= by:
        raise ValueError('No native interior overlap between body audit tiles')
    av = a[ty-ao[1]:by-ao[1], lx-ao[0]:rx-ao[0]]
    bv = b[ty-bo[1]:by-bo[1], lx-bo[0]:rx-bo[0]]
    delta = np.abs(av-bv)
    union = (av >= .5) | (bv >= .5)
    intersection = (av >= .5) & (bv >= .5)
    return {'native_box_xyxy': [int(lx), int(ty), int(rx), int(by)],
            'pixels': int(av.size), 'foreground_union_pixels': int(union.sum()),
            'max_probability_difference': float(delta.max()),
            'mean_probability_difference': float(delta.mean()),
            'foreground_mean_probability_difference': float(delta[union].mean()) if union.any() else None,
            'binary_agreement_iou': float(intersection.sum()/union.sum()) if union.any() else None}


def audit(checkpoint, movie, owners, frames, roi, device='cpu'):
    model, meta = load_body_checkpoint(checkpoint, device)
    reader = FrameReader(movie)
    rows = []
    try:
        for frame in frames:
            image = reader.read(frame)
            if not image.exact:
                raise ValueError(f'inexact audit frame {frame}')
            gray = cv2.cvtColor(image.frame, cv2.COLOR_BGR2GRAY)
            for owner in owners:
                base = np.floor(np.asarray(owner['grain_native'])-TILE/2).astype(int)
                aligned = base//POOLING_LATTICE*POOLING_LATTICE
                reference, origin = predict_owned_body(model, gray, owner, aligned)
                normal, default_origin = predict_owned_body(model, gray, owner)
                phases = []
                for dx,dy in [(64,0),(0,64),(64,64)]:
                    shifted, so = predict_owned_body(model, gray, owner, aligned+[dx,dy])
                    phases.append({'origin': so, **compare(reference, origin, shifted, so, reader.native_size)})
                _, _, tiled = predict_owned_body_region(model, gray, owner, roi)
                rows.append({'owner_id': owner['id'], 'source_frame': frame,
                    'grain_native': owner['grain_native'], 'aligned_origin': origin,
                    'default_origin': default_origin,
                    'default_vs_aligned': compare(normal, default_origin, reference, origin, reader.native_size),
                    'aligned_context_shifts': phases, 'roi_tiling': tiled})
                if model.normalization == 'spatial':
                    context = OwnedBodyReference(model, gray, owner)
                    replay, ro = context.predict()
                    ref_phases = []
                    for dx, dy in [(64, 0), (0, 64), (64, 64)]:
                        shifted, so = context.predict(np.asarray(ro) + [dx, dy])
                        ref_phases.append({'origin': so, **compare(replay, ro, shifted, so, reader.native_size)})
                    _, _, ref_tiled = predict_owned_body_region(
                        model, gray, owner, roi, spatial_context='grain_reference')
                    rows[-1]['grain_reference'] = {
                        'context': context.metadata(),
                        'canonical_max_probability_difference': float(np.abs(replay-normal).max()),
                        'canonical_binary_mask_unchanged': bool(np.array_equal(replay>=.5, normal>=.5)),
                        'context_shifts': ref_phases, 'roi_tiling': ref_tiled}
    finally:
        reader.close()
    differences = [c['max_probability_difference'] for r in rows for c in r['aligned_context_shifts']]
    result = {'checkpoint': {'path': str(Path(checkpoint).resolve()), 'sha256': meta['sha256']},
            'model': meta['config'], 'rows': rows,
            'maximum_aligned_context_difference': max(differences),
            'maximum_roi_overlap_difference': max(r['roi_tiling']['max_overlap_probability_disagreement'] for r in rows),
            'invariance_tolerance': 1e-5, 'aligned_context_invariance_passed': max(differences) <= 1e-5,
            'roi_overlap_invariance_passed': max(r['roi_tiling']['max_overlap_probability_disagreement'] for r in rows) <= 1e-5}
    if model.normalization == 'spatial':
        reference_rows = [r['grain_reference'] for r in rows]
        maximum = max(c['max_probability_difference'] for r in reference_rows for c in r['context_shifts'])
        roi_maximum = max(r['roi_tiling']['max_overlap_probability_disagreement'] for r in reference_rows)
        canonical_maximum = max(r['canonical_max_probability_difference'] for r in reference_rows)
        binary_unchanged = all(r['canonical_binary_mask_unchanged'] for r in reference_rows)
        result['grain_reference_summary'] = {
            'canonical_max_probability_difference': canonical_maximum,
            'canonical_binary_mask_unchanged': binary_unchanged,
            'maximum_context_difference': maximum, 'maximum_roi_overlap_difference': roi_maximum,
            'consistency_and_canonical_gate_passed': max(maximum, roi_maximum, canonical_maximum) <= 1e-5 and binary_unchanged}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, action='append')
    p.add_argument('--config', required=True, help='Analysis configuration defining movie, owners and ROI')
    p.add_argument('--frames', type=int, nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=('cpu','mps','cuda'), default='cpu')
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    cfg = json.loads(Path(a.config).read_text())['request']
    if not cfg.get('roi_xyxy') or not cfg.get('owners') or not a.frames:
        raise ValueError('Native audit requires owners, source frames and a declared ROI')
    torch.set_num_threads(2)
    result = {'scope': 'Prediction consistency only; no new biological labels or accuracy claim',
              'movie': cfg['movie_id'], 'movie_sha256': file_hash(cfg['movie_path']),
              'config': {'path': str(Path(a.config).resolve()), 'sha256': file_hash(a.config)},
              'audit_source_sha256': file_hash(__file__),
              'runtime_source_sha256': {str(f.relative_to(ROOT)): file_hash(f) for f in [
                  ROOT/'prototypes/v30_video_apex/native_body.py', ROOT/'prototypes/v30_video_apex/native_caps.py',
                  ROOT/'tubetracker/annotation_frames.py']},
              'models': [audit(c, cfg['movie_path'], cfg['owners'], a.frames, cfg['roi_xyxy'], a.device)
                         for c in a.checkpoint]}
    changed = [p for p, digest in result['runtime_source_sha256'].items() if file_hash(ROOT/p) != digest]
    if (changed or file_hash(__file__) != result['audit_source_sha256']
            or file_hash(a.config) != result['config']['sha256']
            or file_hash(cfg['movie_path']) != result['movie_sha256']):
        raise RuntimeError('body tiling audit inputs changed during the run')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'out': str(out), 'results': [
        {k:m[k] for k in ('checkpoint','maximum_aligned_context_difference','maximum_roi_overlap_difference',
                          'aligned_context_invariance_passed','roi_overlap_invariance_passed')}
        for m in result['models']]}), flush=True)
    print(json.dumps({'grain_reference': [m.get('grain_reference_summary') for m in result['models']]}), flush=True)


if __name__ == '__main__':
    main()
