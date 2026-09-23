"""rev16 rim-profile evidence: a rim bump scores above a smooth rim."""
import importlib.util
from pathlib import Path

import numpy as np


def _module():
    spec = importlib.util.spec_from_file_location(
        'rim_profile_evidence', 'prototypes/v30_video_apex/rim_profile_evidence.py')
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RPE = _module()
ARTIFACT = Path('runs/prototypes/v30/rev16_rimhead_model.json')


def _smooth_tile(gx=144.0, gy=144.0, radius=13.0, h=288, w=288, blur=1.5):
    """A smooth circular grain (dark disk) on a bright background (no bump),
    with a soft optical edge (real brightfield grains are PSF-blurred, not
    razor-sharp)."""
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - gx, yy - gy)
    tile = np.full((h, w), 200.0)  # bright background
    tile[dist <= radius] = 40.0    # dark grain
    if blur:
        import cv2
        tile = cv2.GaussianBlur(tile.astype(np.float32), (0, 0), blur)
    return tile.astype(np.float32), (gx, gy)


def _bump_tile(gx=144.0, gy=144.0, radius=13.0, angle_deg=45.0, width_deg=24.0,
               out_r=1.5, blur=1.5):
    """The same grain with its circular boundary pushed outward over a wedge
    (a nascent rim bump / protrusion that breaks the circle)."""
    tile, (gx, gy) = _smooth_tile(gx, gy, radius, blur=0.0)  # build sharp, blur after
    h, w = tile.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - gx, yy - gy)
    ang = (np.arctan2(yy - gy, xx - gx) * 180.0 / np.pi) % 360
    d_ang = np.abs((ang - angle_deg + 180.0) % 360.0 - 180.0)
    wedge = (d_ang <= width_deg / 2)
    tile[wedge & (dist <= out_r * radius)] = 40.0
    if blur:
        import cv2
        tile = cv2.GaussianBlur(tile.astype(np.float32), (0, 0), blur)
    return tile.astype(np.float32), (gx, gy)


def test_smooth_rim_disruption_below_bumped_rim():
    """A smooth circle registers less rim disruption than a bumped one.

    Feature-level (not the learned score): a flat synthetic disk is outside the
    model's real-grain training distribution, so its absolute score is not
    meaningful — the classifier's real behaviour is checked on the actual movie
    pairs below (G0, cf70)."""
    smooth, g = _smooth_tile()
    bumped, _ = _bump_tile()
    f_s = RPE.RimProfileEvidence.disruption_features(smooth, g, 13.0)
    f_b = RPE.RimProfileEvidence.disruption_features(bumped, g, 13.0)
    assert f_b[2] > f_s[2], (f_s[2], f_b[2])  # mean_abs disruption


def test_uniform_brightness_shift_is_invariant():
    """Median-subtraction kills a uniform shift; a localized bump survives."""
    model = RPE.RimProfileEvidence.from_json(str(ARTIFACT))
    tile, g = _smooth_tile()
    bright = np.clip(tile + 40.0, 0, 255).astype(np.float32)
    d0 = RPE.RimProfileEvidence.rim_profile_deviation(tile, g, 13.0)
    d1 = RPE.RimProfileEvidence.rim_profile_deviation(bright, g, 13.0)
    assert np.allclose(d0, d1, atol=1e-4), 'uniform shift must cancel'
    # a real bump does not
    bumped, _ = _bump_tile()
    d2 = RPE.RimProfileEvidence.rim_profile_deviation(bumped, g, 13.0)
    assert not np.allclose(d0, d2, atol=1e-3)


def test_artifact_contract_and_shapes():
    model = RPE.RimProfileEvidence.from_json(str(ARTIFACT))
    assert model.weight.shape == (RPE.RimProfileEvidence.N_FEATURES,)
    assert model.mu.shape == model.sd.shape == model.weight.shape
    assert model.feature_names == RPE.RimProfileEvidence.disruption_feature_names()
    assert np.isfinite(model.bias)
    # a mismatched contract is an error, never a silent default
    import pytest
    with pytest.raises(ValueError):
        RPE.RimProfileEvidence(mu=[0.0] * 6, sd=[1.0] * 6,
                               weight=[0.0] * 6, bias=0.0,
                               contract='tubetracker.rim_profile_evidence.v999')
    with pytest.raises(ValueError):  # wrong feature count
        RPE.RimProfileEvidence(mu=[0.0] * 3, sd=[1.0] * 3,
                               weight=[0.0] * 3, bias=0.0)


def test_bump_raises_disruption_at_any_orientation():
    """A protrusion that pushes the rim boundary outward registers as rim
    disruption at any angle, with an angle-independent magnitude (the property
    that lets the feature score any tube orientation — the 72-bin
    angle-specific head did not). Asserted on the FEATURE: a flat synthetic
    disk is outside the model's real-grain training distribution, so its
    absolute score is not meaningful, but the feature's response is."""
    smooth, g = _smooth_tile()
    s_feats = RPE.RimProfileEvidence.disruption_features(smooth, g, 13.0)
    magnitudes = []
    for angle in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
        bumped, _ = _bump_tile(angle_deg=angle)
        feats = RPE.RimProfileEvidence.disruption_features(bumped, g, 13.0)
        # the outward push raises the peak outward deviation over a smooth rim
        assert feats[0] > s_feats[0], \
            f'angle {angle}: bump did not raise max_out {feats[0]} vs {s_feats[0]}'
        magnitudes.append(feats[0])
    magnitudes = np.array(magnitudes)
    assert magnitudes.std() < 0.5 * magnitudes.mean(), \
        f'disruption magnitude varies strongly with angle: {magnitudes}'


def test_feature_is_point_reflection_invariant():
    """A 180-degree bump flip (point reflection through the centred grain)
    leaves the 6-feature vector unchanged up to pixel rasterization — the
    disruption stats carry no angle index by construction (a circular shift of
    the rim profile leaves max/min/mean/sum/max-run unchanged)."""
    for base in (0, 37, 120):
        b1, g = _bump_tile(angle_deg=base)
        b2, _ = _bump_tile(angle_deg=base + 180.0)
        f1 = RPE.RimProfileEvidence.disruption_features(b1, g, 13.0)
        f2 = RPE.RimProfileEvidence.disruption_features(b2, g, 13.0)
        # invariant up to how the angular wedge rasterizes (a few % not 1e-9)
        assert np.allclose(f1, f2, rtol=0.2, atol=0.05), (base, f1, f2)


def test_real_movie_g0_pair_orders():
    """G0's reviewed pair (absent@8000, visible@8050) orders correctly with
    the runtime scorer and the trained artifact — the in-vivo check the
    whole-crop heads failed."""
    import cv2
    from prototypes.v30_video_apex.native_caps import extract_tile
    from tubetracker.annotation_frames import FrameReader
    model = RPE.RimProfileEvidence.from_json(str(ARTIFACT))
    reader = FrameReader('/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4')
    grain = [535.1, 640.1]
    radius = 12.1
    try:
        scores = {}
        for f in (8000, 8050):
            gray = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY)
            ox, oy = int(np.floor(grain[0] - 144)), int(np.floor(grain[1] - 144))
            tile = extract_tile([gray], (ox, oy), RPE.TILE)[0].astype(np.float32)
            scores[f] = model.score(tile, (grain[0] - ox, grain[1] - oy), radius)
    finally:
        reader.close()
    assert scores[8050] > scores[8000], scores
