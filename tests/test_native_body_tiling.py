import numpy as np
import pytest
import torch

from prototypes.v30_video_apex.native_body import (
    NativeBodyNet, OwnedBodyReference, load_body_checkpoint, predict_owned_body,
    predict_owned_body_region, save_body_checkpoint)


@pytest.mark.parametrize('grain', ([211.25, 189.75], [12.5, 19.5]))
def test_actual_body_network_agrees_across_aligned_crop_interiors(grain):
    image = np.random.default_rng(15).integers(80, 180, (480, 512), dtype=np.uint8)
    owner = {'grain_native': grain, 'grain_radius_px': 13.}
    torch.manual_seed(53)
    model = NativeBodyNet(base=4, normalization='pixel', image_gain=8).eval()
    a, _ = predict_owned_body(model, image, owner, [64, 64])
    b, _ = predict_owned_body(model, image, owner, [128, 128])
    # These are identical native pixels with >=64 px of context in each
    # crop. The second owner lies outside both inputs; its query stays
    # native rather than being moved into a convenient crop location.
    np.testing.assert_allclose(a[128:224, 128:224], b[64:160, 64:160], atol=3e-6, rtol=3e-6)
    torch.manual_seed(53)
    spatial = NativeBodyNet(base=4, image_gain=8).eval()
    a, _ = predict_owned_body(spatial, image, owner, [64, 64])
    b, _ = predict_owned_body(spatial, image, owner, [128, 128])
    assert np.max(np.abs(a[128:224, 128:224] - b[64:160, 64:160])) > .001


def test_default_pixel_crop_and_region_share_native_origin_phase():
    torch.manual_seed(19)
    model = NativeBodyNet(base=4, normalization='pixel').eval()
    image = np.random.default_rng(2).integers(0, 256, (432, 448), dtype=np.uint8)
    owner = {'grain_native': [217.25, 211.75], 'grain_radius_px': 13.}
    crop, origin = predict_owned_body(model, image, owner)
    assert origin == [72, 64]
    roi = [151, 145, 231, 225]
    region, actual, report = predict_owned_body_region(model, image, owner, roi)
    np.testing.assert_allclose(region, crop[81:161, 79:159], atol=3e-6, rtol=3e-6)
    assert actual == roi[:2]
    assert report['normalization'] == 'pixel'
    assert report['max_overlap_probability_disagreement'] < 3e-6
    with pytest.raises(ValueError, match='8-pixel'):
        predict_owned_body(model, image, owner, [73, 64])


@pytest.mark.parametrize('normalization', ('spatial', 'pixel'))
def test_body_checkpoint_keeps_normalization_and_coordinate_contract(tmp_path, normalization):
    torch.manual_seed(53)
    model = NativeBodyNet(base=4, normalization=normalization, image_gain=8).eval()
    path = tmp_path/'body.pt'
    save_body_checkpoint(path, model, {'test': 'normalization-contract'})
    loaded, meta = load_body_checkpoint(path)
    assert meta['config'] == model.config()
    x = torch.rand(1, 4, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)


@pytest.mark.parametrize('schema', ('v1', 'v2'))
def test_legacy_body_checkpoint_preserves_spatial_normalization_and_crop(tmp_path, schema):
    model = NativeBodyNet(base=4, image_gain=1 if schema == 'v1' else 8).eval()
    config = {k:v for k,v in model.config().items()
              if k not in ('normalization', 'crop_origin_lattice_px', 'tile_valid_margin_px')}
    if schema == 'v1':
        del config['image_contrast_gain_about_midgray']
    path = tmp_path/'legacy.pt'
    torch.save({'schema': 'tubetracker.native_owned_body.'+schema,
                'config': config, 'manifest': {}, 'model_state': model.state_dict()}, path)
    loaded, _ = load_body_checkpoint(path)
    image = np.random.default_rng(2).integers(0, 256, (350, 350), dtype=np.uint8)
    owner = {'grain_native': [217.25, 211.75], 'grain_radius_px': 13.}
    actual, origin = predict_owned_body(loaded, image, owner)
    expected, _ = predict_owned_body(model, image, owner, [73, 67])
    assert origin == [73, 67] and loaded.normalization == 'spatial'
    np.testing.assert_array_equal(actual, expected)


def test_stat_capture_keeps_canonical_forward_and_references_reproduce_it():
    torch.manual_seed(53)
    model = NativeBodyNet(base=4, image_gain=8).eval()
    inputs = torch.rand(1, 4, 128, 128)
    stats = {}
    with torch.no_grad():
        old = model(inputs)
        captured = model(inputs, capture_stats=stats)
        replayed = model(inputs, reference_stats=stats)
    torch.testing.assert_close(old, captured, rtol=0, atol=0)
    torch.testing.assert_close(old.sigmoid(), replayed.sigmoid(), rtol=0, atol=3e-6)
    assert len(stats) == 14 and all(not t.requires_grad for pair in stats.values() for t in pair)
    with pytest.raises(ValueError, match='match'):
        model(inputs, reference_stats={})
    with pytest.raises(ValueError, match='empty'):
        model(inputs, capture_stats=stats)


@pytest.mark.parametrize('grain', ([217.25, 211.75], [12.5, 19.5]))
def test_reference_body_matches_native_interiors_even_when_owner_outside_tiles(grain):
    image = np.random.default_rng(15).integers(80, 180, (480, 512), dtype=np.uint8)
    owner = {'id': 'grain-one', 'grain_native': grain, 'grain_radius_px': 13.}
    torch.manual_seed(53)
    model = NativeBodyNet(base=4, image_gain=8).eval()
    reference = OwnedBodyReference(model, image, owner)
    canonical, origin = predict_owned_body(model, image, owner)
    replayed, actual = reference.predict()
    assert actual == origin
    np.testing.assert_allclose(canonical, replayed, atol=3e-6, rtol=0)
    phase = np.asarray(reference.phase)
    a, ao = reference.predict(np.array([64, 64]) + phase)
    b, bo = reference.predict(np.array([128, 128]) + phase)
    np.testing.assert_allclose(a[128:224, 128:224], b[64:160, 64:160], atol=3e-6, rtol=0)
    region, origin, meta = predict_owned_body_region(
        model, image, owner, [128, 129, 450, 431], spatial_context='grain_reference')
    assert region.shape == (302, 322) and origin == [128, 129]
    assert meta['normalization_context'] == 'grain_reference'
    assert meta['tile_origin_phase_xy'] == reference.metadata()['pooling_phase_xy']
    assert meta['max_overlap_probability_disagreement'] < 3e-6
    assert all((np.asarray(o) % 8 == reference.phase).all() for o in meta['tile_origins_xy'])
    # Compare native pixels in a reference tile's interior with the wider field.
    left, top = bo[0]+64, bo[1]+64
    np.testing.assert_allclose(region[top-129:top-129+96, left-128:left-128+96],
                               b[64:160, 64:160], atol=3e-6, rtol=0)
    if grain[0] < 64:
        assert meta['tiles_without_grain_in_input'] == len(meta['tile_origins_xy'])


def test_reference_is_bound_to_copied_frame_owner_weights_and_phase():
    image = np.random.default_rng(1).integers(0, 256, (384, 384), dtype=np.uint8)
    owner = {'id': 'grain', 'grain_native': [217.25, 211.75], 'grain_radius_px': 13.}
    model = NativeBodyNet(base=4).eval()
    reference = OwnedBodyReference(model, image, owner)
    expected, _ = reference.predict()
    image[:] = 0
    owner['grain_native'][0] += 50
    owner['grain_radius_px'] = 25
    actual, _ = reference.predict()
    np.testing.assert_array_equal(actual, expected)
    meta = reference.metadata()
    meta['grain_native'][0] = 0
    assert reference.metadata()['grain_native'][0] == 217.25
    with pytest.raises(ValueError, match='pooling phase'):
        reference.predict(np.asarray(reference.origin) + [1, 0])
    model.e1.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        reference.predict()
    model.eval()
    # Even .data edits, which do not reliably advance tensor version counters,
    # must invalidate captured statistics.
    next(model.parameters()).data.add_(.1)
    with pytest.raises(RuntimeError, match='changed'):
        reference.predict()


def test_reference_requires_spatial_eval_model_and_valid_native_query():
    image = np.ones((320, 320), np.uint8)
    owner = {'grain_native': [160., 160.]}
    with pytest.raises(ValueError, match='evaluation'):
        OwnedBodyReference(NativeBodyNet(base=4), image, owner)
    with pytest.raises(ValueError, match='spatial'):
        OwnedBodyReference(NativeBodyNet(base=4, normalization='pixel').eval(), image, owner)
    with pytest.raises(ValueError, match='finite physical grain'):
        OwnedBodyReference(NativeBodyNet(base=4).eval(), image,
                           {'grain_native': [160., 160.], 'grain_radius_px': float('nan')})


@pytest.mark.parametrize('root', [None, [224.5, 205.5]])
def test_root_channel_matches_crop_reference_and_region_inference(root):
    torch.manual_seed(11)
    model = NativeBodyNet(base=4, root_channel=True).eval()
    image = np.random.default_rng(4).integers(0, 256, (384, 384), dtype=np.uint8)
    owner = {'id': 'a', 'grain_native': [217.25, 211.75], 'grain_radius_px': 13., 'root_native': root}
    canonical, origin = predict_owned_body(model, image, owner)
    reference = OwnedBodyReference(model, image, owner)
    replayed, actual = reference.predict()
    assert actual == origin and reference.metadata()['root_native'] == root
    assert reference.metadata()['root_channel'] is True
    np.testing.assert_allclose(canonical, replayed, atol=3e-6, rtol=0)
    roi = [151, 145, 231, 225]
    region, _, meta = predict_owned_body_region(model, image, owner, roi, spatial_context='grain_reference')
    np.testing.assert_allclose(region, canonical[78:158, 78:158], atol=3e-6, rtol=0)
    assert meta['normalization_reference']['root_native'] == root
    owner['root_native'] = [0., 0.]
    np.testing.assert_array_equal(reference.predict()[0], replayed)
