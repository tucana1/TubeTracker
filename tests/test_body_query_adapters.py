import copy

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.native_body import (
    NativeBodyNet, body_input, load_body_checkpoint, save_body_checkpoint,
)


def test_new_adapters_preserve_initial_native_prediction_and_checkpoint_contract(tmp_path):
    torch.manual_seed(71)
    old = NativeBodyNet(base=4, image_gain=8).eval()
    model = NativeBodyNet(base=4, image_gain=8, query_adapters=True).eval()
    result = model.load_state_dict(old.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith('query_') for k in result.missing_keys)
    image = np.random.default_rng(71).integers(100, 150, (64, 64), dtype=np.uint8)
    x = torch.from_numpy(body_input(image, [100, 200], [127, 231], image_gain=8))[None]
    with torch.no_grad():
        torch.testing.assert_close(model(x), old(x), rtol=0, atol=0)
    save_body_checkpoint(tmp_path/'candidate.pt', model, {'scope': 'test'})
    loaded, meta = load_body_checkpoint(tmp_path/'candidate.pt')
    assert meta['config']['query_adapters']['coordinate_clip'] == 8.
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x), rtol=0, atol=0)


def test_finer_query_is_native_and_grain_relative_on_shared_image():
    image = np.full((64, 64), 128, np.uint8)
    left = torch.from_numpy(body_input(image, [900, 300], [924, 331]))[None]
    right = torch.from_numpy(body_input(image, [900, 300], [940, 331]))[None]
    torch.testing.assert_close(left[:, 0], right[:, 0], rtol=0, atol=0)
    ql, qr = NativeBodyNet.decoder_query(left), NativeBodyNet.decoder_query(right)
    torch.testing.assert_close(ql[:, 1]-qr[:, 1], torch.ones_like(ql[:, 1]), rtol=0, atol=0)
    torch.testing.assert_close(ql[:, 2], qr[:, 2], rtol=0, atol=0)


def test_query_contract_cannot_be_reinterpreted_at_load(tmp_path):
    model = NativeBodyNet(base=4, query_adapters=True)
    save_body_checkpoint(tmp_path/'valid.pt', model, {})
    blob = torch.load(tmp_path/'valid.pt', weights_only=False)
    changed = copy.deepcopy(blob)
    changed['config']['query_adapters']['coordinate_clip'] = 4.
    torch.save(changed, tmp_path/'changed.pt')
    with pytest.raises(ValueError, match='preprocessing/query contract mismatch'):
        load_body_checkpoint(tmp_path/'changed.pt')
