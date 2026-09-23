import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.native_caps import (
    NativeCapNet, detect_caps, extract_tile, load_native_checkpoint,
    save_native_checkpoint, score_native_tile)


def test_local_features_agree_for_overlapping_native_pixels():
    torch.manual_seed(53)
    frames = np.random.default_rng(1).integers(0, 256, (3, 192, 192), dtype=np.uint8)
    model = NativeCapNet(base=4, normalization='pixel').eval()
    a = score_native_tile(model, extract_tile(frames, (0, 0)))
    b = score_native_tile(model, extract_tile(frames, (32, 32)))
    for key in ('probability', 'offset_xy', 'logvar'):
        np.testing.assert_allclose(a[key][64:96, 64:96], b[key][32:64, 32:64],
                                   atol=3e-6, rtol=3e-6)
    # This is the defect the local architecture must remove, not a test
    # that either normalization can accidentally satisfy on uniform input.
    old = NativeCapNet(base=4).eval()
    a = score_native_tile(old, extract_tile(frames, (0, 0)))
    b = score_native_tile(old, extract_tile(frames, (32, 32)))
    assert np.max(np.abs(a['probability'][64:96, 64:96] - b['probability'][32:64, 32:64])) > .001


def test_distant_pixels_cannot_change_a_local_cap():
    torch.manual_seed(17)
    model = NativeCapNet(base=4, normalization='pixel').eval()
    clip = torch.rand(1, 3, 1, 128, 128)
    changed = clip.clone(); changed[..., :12, :12] = 50
    with torch.no_grad():
        a, b = model(clip), model(changed)
    for key in a:
        torch.testing.assert_close(a[key][..., 64:80, 64:80], b[key][..., 64:80, 64:80], rtol=0, atol=0)


class BrightPointModel(torch.nn.Module):
    valid_margin = 32
    pooling_lattice = 4
    normalization = 'pixel'
    temporal = False

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, clip):
        value = clip[:, 1] * 40 - 20 + self.anchor
        return {'logits': value, 'offset_xy': torch.zeros_like(value).repeat(1, 2, 1, 1),
                'logvar': torch.zeros_like(value)}


def test_phase_and_roi_edges_preserve_points_and_complete_coverage():
    model = BrightPointModel()
    frames = np.zeros((3, 320, 320), np.uint8)
    points = [[41, 43], [88, 126], [219, 224], [110, 55]]
    for x, y in points:
        frames[:, y, x] = 255
    for phase in ((0, 0), (17, 29), (47, 53), (63, 63)):
        out = detect_caps(model, frames, movie='m', source_frame=5,
                          roi=(41, 43, 220, 225), tile_phase_xy=phase)
        assert sorted(c['tip_xy'] for c in out['caps']) == sorted(points)
        assert out['tiling']['covered_roi_pixels'] == out['tiling']['eligible_roi_pixels'] == 179*182
        assert all(all(v % 4 == 0 for v in o) for o in out['tiling']['tile_origins_xy'])
        assert all(c['unrefined_xy'] == c['tip_xy'] for c in out['caps'])
    with pytest.raises(ValueError, match='tile phase'):
        detect_caps(model, frames, movie='m', source_frame=5, tile_phase_xy=(-1, 0))
    with pytest.raises(ValueError, match='ROI'):
        detect_caps(model, frames, movie='m', source_frame=5, roi=(0, 0, 321, 300))


@pytest.mark.parametrize('normalization', ['spatial', 'pixel'])
def test_checkpoint_keeps_its_normalization_and_valid_context(tmp_path, normalization):
    model = NativeCapNet(base=4, normalization=normalization)
    path = tmp_path / 'model.pt'
    save_native_checkpoint(path, model, manifest={'purpose': 'regression'})
    loaded, _ = load_native_checkpoint(path)
    assert loaded.config() == model.config()
    assert loaded.valid_margin == (32 if normalization == 'pixel' else 16)
    x = torch.rand(1, 3, 1, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(x)['logits'], loaded(x)['logits'], rtol=0, atol=0)
