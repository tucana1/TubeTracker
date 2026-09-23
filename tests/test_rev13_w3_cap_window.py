"""rev13 W3: the native cap scorer over an ORDERED time window.

Pins the contract the review asks for:
- native-resolution 2D scoring with current-image features (not a
  scalar MLP over pooled features);
- the ordered time window is actually consumed (reversing it changes
  the output; the single-frame control is the same weights with a
  one-frame window);
- native XY refinement (px) and a log-variance (uncertainty) are
  produced per location;
- the sampled patch at a fixed location does not depend on any route
  end (route-end invariance by construction);
- the head is declared metadata: it exists only when configured, and
  the strict factory rebuilds it from the checkpoint config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

CS = 288


def _model(seed: int = 0, window_head: bool = True):
    from prototypes.v30_video_apex.model import (
        build_model, build_owner_prompt)
    torch.manual_seed(seed)
    m = build_model("temporal", base=4, multiscale=True,
                    cap_window_head=window_head)
    m.eval()
    clip = torch.rand(1, 9, 1, CS, CS)
    prompt = build_owner_prompt("o")
    return m, clip, prompt


def _locs(k=3):
    torch.manual_seed(7)
    xy = torch.tensor([[[100.0, 140.0], [150.0, 100.0],
                        [90.0, 90.0]]], dtype=torch.float32)[:, :k]
    tg = torch.tensor([[[1.0, 0.0], [0.0, 1.0],
                        [0.7, 0.7]]], dtype=torch.float32)[:, :k]
    tg = tg / tg.norm(dim=-1, keepdim=True)
    return xy, tg


def test_absent_unless_configured():
    m, clip, prompt = _model(window_head=False)
    assert not hasattr(m, "cw_logit")
    from prototypes.v30_video_apex.model_factory import (
        model_kwargs_from_config)
    assert "cap_window_head" not in model_kwargs_from_config(
        {"variant": "temporal", "base": 4, "multiscale": True})
    assert model_kwargs_from_config(
        {"variant": "temporal", "base": 4, "multiscale": True,
         "cap_window_head": True})["cap_window_head"] is True


def test_shapes_refinement_uncertainty():
    m, clip, prompt = _model()
    xy, tg = _locs()
    out = m.score_cap_window(clip, prompt, xy, tg)
    assert out["logits"].shape == (1, 3)
    assert out["dxy_px"].shape == (1, 3, 2)
    assert out["logvar"].shape == (1, 3)
    assert out["window"] == list(range(9))
    # refinement magnitude is bounded by the patch scale (declared)
    assert float(out["dxy_px"].abs().max()) <= m.cw_patch_px


def test_ordered_window_is_consumed():
    m, clip, prompt = _model()
    xy, tg = _locs()
    fwd = m.score_cap_window(clip, prompt, xy, tg,
                             window_indices=[0, 1, 2, 3, 4])
    rev = m.score_cap_window(clip, prompt, xy, tg,
                             window_indices=[4, 3, 2, 1, 0])
    # different acquisition order -> different temporal mix
    assert not torch.allclose(fwd["logits"], rev["logits"])
    # the single-frame control uses the same weights on one frame
    one = m.score_cap_window(clip, prompt, xy, tg, frame_only=True)
    assert one["window"] == [4]
    ctrl = m.score_cap_window(clip, prompt, xy, tg, window_indices=[4])
    assert torch.allclose(one["logits"], ctrl["logits"])


def test_route_end_invariance():
    """The patch at a fixed location cannot depend on route ends: the
    scorer never receives a route at all."""
    m, clip, prompt = _model()
    xy, tg = _locs()
    a = m.score_cap_window(clip, prompt, xy, tg)
    b = m.score_cap_window(clip, prompt, xy, tg)
    assert torch.allclose(a["logits"], b["logits"])
    # and a second location's patch is unaffected by adding a location
    xy2 = torch.cat([xy, torch.tensor([[[200.0, 200.0]]])], dim=1)
    tg2 = torch.cat([tg, torch.tensor([[[1.0, 0.0]]])], dim=1)
    c = m.score_cap_window(clip, prompt, xy2, tg2)
    assert torch.allclose(a["logits"], c["logits"][:, :3])


def test_patch_is_native_resolution():
    m, clip, prompt = _model()
    assert m.cw_patch_px == 24 and m.cw_hidden == 16
    # parameter count stays small (bounded first implementation)
    n = sum(p.numel() for p in m.cw_conv.parameters()) \
        + sum(p.numel() for p in m.cw_temporal.parameters()) \
        + sum(p.numel() for p in m.cw_logit.parameters()) \
        + sum(p.numel() for p in m.cw_dxy.parameters()) \
        + sum(p.numel() for p in m.cw_logvar.parameters())
    assert n < 20_000


def test_encoder_feeding_is_real_raw_variant_is_inert():
    """The regression that catches H429: in the RAW variant, perturbing
    the encoder cannot change the scorer (it never reads it) — adapting
    the encoder would be an inert variable. In the ENCODER-FED variant
    the same perturbation MUST change the logits."""
    from prototypes.v30_video_apex.model import build_model
    torch.manual_seed(0)
    m_raw = build_model("temporal", base=4, multiscale=True,
                        cap_window_head=True)
    m_raw.eval()
    clip = torch.rand(1, 9, 1, CS, CS)
    from prototypes.v30_video_apex.model import build_owner_prompt
    prompt = build_owner_prompt("o")
    xy, tg = _locs()
    a1 = m_raw.score_cap_window(clip, prompt, xy, tg)
    with torch.no_grad():
        for p in m_raw.unet.parameters():
            p.add_(0.1)
    a2 = m_raw.score_cap_window(clip, prompt, xy, tg)
    assert torch.allclose(a1["logits"], a2["logits"])  # inert by design

    torch.manual_seed(0)
    m_enc = build_model("temporal", base=4, multiscale=True,
                        cap_window_head=True, cap_window_encoder=True)
    m_enc.eval()
    assert m_enc.cap_window_encoder is True
    b1 = m_enc.score_cap_window(clip, prompt, xy, tg)
    with torch.no_grad():
        for p in m_enc.unet.parameters():
            p.add_(0.1)
    b2 = m_enc.score_cap_window(clip, prompt, xy, tg)
    assert not torch.allclose(b1["logits"], b2["logits"])
    assert b1["logits"].shape == (1, 3)


def test_factory_maps_the_encoder_flag():
    from prototypes.v30_video_apex.model_factory import (
        model_kwargs_from_config)
    cfg = {"variant": "temporal", "base": 4, "multiscale": True,
           "cap_window_head": True, "cap_window_encoder": True}
    assert model_kwargs_from_config(cfg)["cap_window_encoder"] is True
    cfg.pop("cap_window_encoder")
    assert "cap_window_encoder" not in model_kwargs_from_config(cfg)


def test_validity_mask_excludes_clamped_locations():
    """rev13 W3.4 (H431): a location whose 24px patch leaves the frame
    is invalid — the mask must say so, and valid locations must remain
    valid."""
    m, clip, prompt = _model()
    xy = torch.tensor([[[5.0, 5.0], [144.0, 144.0],
                        [-30.0, 144.0]]], dtype=torch.float32)
    tg = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    out = m.score_cap_window(clip, prompt, xy, tg)
    v = out["valid"].reshape(-1).tolist()
    assert v[0] is False or v[0] == False  # patch leaves the frame
    assert v[1] is True or v[1] == True    # interior: valid
    assert v[2] is False or v[2] == False  # center outside the frame
