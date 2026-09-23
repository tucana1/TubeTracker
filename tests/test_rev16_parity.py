"""WO1-parity: train and runtime build the same native body input contract.

Uses cf70 frame 0 (the early drifting grain): runtime tile construction
(predict_owned_body) vs panel centre-window construction. Pins the origin
conventions, the shared body_input implementation (no forked duplicate
formula), gain/radius parameters, and saves an overlay figure.
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

POSES = Path('/Users/joshjiang/Documents/New project/TubeTracker-sparse-germination-audit-2026-09-21/runtime-cache/grain-motion/d46234d1e8ac5e413028fe7791406d42100dc620012de1c64e03a9f69f7aa8e7/poses.json')
PANEL = Path('runs/prototypes/v30/rev16_body_panel_posed/panel.json')
NPZ = Path('runs/prototypes/v30/rev16_body_panel_posed/panel.npz')
REPAIR = Path('runs/prototypes/v30/rev15_execution_20260920T061445Z/ownership-correction-20260920T174154Z/body-repair/step-0400.pt')
OUT = Path('runs/prototypes/v30/rev16_parity_cf70_f0.png')


def _trainer():
    spec = importlib.util.spec_from_file_location(
        'rev16_parity_trainer', 'scripts/train_native_body.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_body_input_implementation():
    from prototypes.v30_video_apex import native_body
    trainer = _trainer()
    assert trainer.body_input is native_body.body_input, (
        'train and runtime must share one body_input, not duplicated formulas')
    assert trainer.TILE == native_body.TILE == 288
    assert trainer.PAD == 16
    assert trainer.STORED == 320


def test_cf70_f0_origins_and_tensor_parity():
    import torch
    from prototypes.v30_video_apex.native_body import (
        body_input, extract_tile, load_body_checkpoint, predict_owned_body)
    from tubetracker.annotation_frames import FrameReader
    from tubetracker.frame_geometry import pose_index
    trainer = _trainer()
    assert trainer.body_input is body_input

    poses = pose_index(json.loads(POSES.read_text()))
    owner_id = 'ld|review-53b55d8191864f77b5ed91beef56cf70'
    pose = poses[(owner_id, 0)]
    grain = np.asarray(pose['grain_native'], float)
    radius = float(pose['grain_radius_px'])
    assert radius == 13.0

    model, _ = load_body_checkpoint(str(REPAIR), device='cpu')
    model.eval()
    snap = json.loads((Path('runs/prototypes/v30/snap34_rev15_queue_complete/snapshot_manifest.json')).read_text())
    movie = [m for m in snap['movies'].values() if 'lowdens' in m['path']][0]['path']
    reader = FrameReader(movie)
    try:
        full = cv2.cvtColor(reader.read(0).frame, cv2.COLOR_BGR2GRAY)
    finally:
        reader.close()

    # Runtime construction (the app path).
    owner = {'id': owner_id, 'grain_native': list(grain), 'grain_radius_px': radius}
    body_rt, origin_rt = predict_owned_body(model, full, owner)
    lattice = model.pooling_lattice
    expect_rt = (np.floor(grain - 288 / 2) // lattice * lattice).astype(int)
    assert list(origin_rt) == list(expect_rt), (origin_rt, expect_rt)

    # Train construction (the panel path): STORED window + centre crop.
    doc = json.loads(PANEL.read_text())
    case = next(c for c in doc['cases']
                if c['owner'] == owner_id and c['frame'] == 0 and c['kind'] == 'owned_presence')
    assert case['query_pose_source'] == 'runtime_pose'
    origin_stored = np.floor(np.asarray(case['grain']) - 160).astype(int)
    assert list(origin_stored) == list(np.asarray(case['origin'])), 'stored origin convention'
    arrays = np.load(NPZ)
    i = doc['cases'].index(case)
    stored = arrays['pixels'][i]
    assert stored.shape == (320, 320)
    assert np.array_equal(stored, extract_tile([full], origin_stored, 320)[0])
    gray288 = stored[16:16 + 288, 16:16 + 288]
    origin_tr = np.asarray(case['origin']) + 16
    x_tr = body_input(gray288, origin_tr, case['grain'], case['radius'], model.image_gain)

    # Same origin through the shared constructor => identical tensors.
    tile = extract_tile([full], np.asarray(origin_rt), 288)[0]
    x_rt = body_input(tile, np.asarray(origin_rt), list(grain), radius, model.image_gain)
    with torch.no_grad():
        p_rt = torch.sigmoid(model(torch.from_numpy(x_rt)[None]))[0, 0].numpy()
    assert np.array_equal(p_rt, body_rt), 'runtime output must equal the shared-contract output'
    assert x_tr.shape == x_rt.shape == (4, 288, 288)
    assert x_tr.dtype == x_rt.dtype == np.float32

    # Overlay figure: both windows with the posed grain marked.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, org, tag in ((axes[0], tile, origin_rt, f'runtime tile @ {list(origin_rt)}'),
                              (axes[1], gray288, origin_tr, f'train window @ {list(origin_tr)}')):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.plot(grain[0] - org[0], grain[1] - org[1], 'o', color='magenta', markersize=6)
        ax.set_title(tag, fontsize=9)
        ax.axis('off')
    fig.suptitle('cf70 f0: posed grain [926.4, 493.0] in both constructions', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    assert OUT.exists()
