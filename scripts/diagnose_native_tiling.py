"""Bounded padding/valid-context diagnosis with frozen native cap weights."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from prototypes.v30_video_apex import native_caps as nc
from scripts.train_native_caps import evaluate
from tubetracker.annotation_frames import FrameReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((ROOT / 'runs/prototypes/v30/rev14_native_runtime_audit/frozen_panel.json').read_text())
    sm = json.loads((Path(frozen['snapshot']) / 'snapshot_manifest.json').read_text())
    panel = ROOT / 'runs/prototypes/v30/rev14_native_panel'
    fit = json.loads((panel / 'panel.json').read_text())
    arrays = dict(np.load(panel / 'panel.npz'))
    checkpoint = ROOT / 'runs/prototypes/v30/rev14_native_single_refine/step-0200.pt'
    model, _ = nc.load_native_checkpoint(checkpoint)
    manifest = {'checkpoint': str(checkpoint), 'checkpoint_sha256': nc.file_hash(checkpoint),
                'frozen_audit': frozen, 'fit_panel_sha256': nc.file_hash(panel / 'panel.json'),
                'variants': ['zeros-16', 'reflect-16', 'zeros-32', 'reflect-32'],
                'purpose': 'diagnose unchanged weights; no variant is automatically promoted',
                'source_sha256': nc.file_hash(__file__)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    frames = {}
    for movie in {c['movie'] for c in frozen['cases']}:
        reader = FrameReader(sm['movies'][movie]['path'])
        try:
            for c in frozen['cases']:
                if c['movie'] != movie:
                    continue
                clip = []
                for offset in nc.OFFSETS:
                    r = reader.read(c['frame'] + offset)
                    if not r.exact:
                        raise ValueError('Inexact movie frame')
                    clip.append(cv2.cvtColor(r.frame, cv2.COLOR_BGR2GRAY))
                frames[c['id']] = clip
        finally:
            reader.close()
    rows, flat, fits = [], [], []
    for mode in ('zeros', 'reflect'):
        candidate = copy.deepcopy(model)
        for module in candidate.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = mode
        for margin in (16, 32):
            nc.VALID_MARGIN = margin
            candidate.valid_margin = margin
            name = mode + '-' + str(margin)
            result = evaluate(candidate, arrays, fit)
            fit_gate = result['fit_gate_passed']
            fits.append({'variant': name, **result})
            for level in (100, 160, 220):
                clip = np.full((3, 128, 128), level, dtype=np.uint8)
                result = nc.score_native_tile(candidate, clip)
                flat.append({'variant': name, 'intensity': level, 'n_caps': len(result['caps']),
                             'max_valid_probability': float(result['probability'][result['valid']].max()),
                             'interior_max_probability': float(result['probability'][40:88, 40:88].max())})
            for c in frozen['cases']:
                for phase in frozen['tile_phases_xy']:
                    detected = nc.detect_caps(candidate, frames[c['id']], movie=c['movie'],
                        source_frame=c['frame'], roi=c['roi_xyxy'], tile_phase_xy=phase)
                    caps = detected['caps']
                    error = min((float(np.linalg.norm(np.array(cap['tip_xy']) - c['tip_xy']))
                                 for cap in caps), default=None)
                    stride = 128 - 2 * margin
                    x0, y0 = c['roi_xyxy'][:2]
                    near_seam = 0
                    for cap in caps:
                        x = (cap['tip_xy'][0] - x0 + phase[0]) % stride
                        near_seam += min(x, stride - x) <= 8
                    row = {'variant': name, 'case': c['id'], 'phase': phase,
                           'n_caps': len(caps), 'nearest_error_px': error,
                           'within_5px': error is not None and error <= 5,
                           'within_8px_vertical_tile_seam': near_seam, 'caps': caps}
                    rows.append(row)
            print(json.dumps({'variant': name, 'fit_gate': fit_gate,
                              'audit_hits': sum(r['within_5px'] for r in rows if r['variant'] == name),
                              'flat': [r for r in flat if r['variant'] == name]}), flush=True)
    nc.VALID_MARGIN = 16
    (out / 'diagnosis.json').write_text(json.dumps({'fit': fits, 'flat': flat, 'audit': rows}, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ('zeros', 'reflect')):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = mode
        result = nc.score_native_tile(model, np.full((3, 128, 128), 160, dtype=np.uint8))
        ax.imshow(result['probability'], vmin=0, vmax=1, cmap='magma')
        ax.add_patch(plt.Rectangle((16, 16), 96, 96, fill=False, color='cyan'))
        ax.set_title(mode + ': probability on uniform image')
    fig.tight_layout(); fig.savefig(out / 'uniform-padding.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
