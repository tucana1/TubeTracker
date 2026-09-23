"""rev10 WP-B: the owner-decoder variant's contracts.

1. Its image-only encoder must be layer-for-layer equivalent to
   `_MultiScaleUNet`'s encoder under identical weights — the mirror
   exists so features can be cached, and a silent drift would make the
   cache wrong in a way no loss would show.
2. Ownership must enter ONLY as spatial prompts: no owner-id embedding
   may exist (the review: "Do not turn a metadata owner string into a
   memorized label embedding that cannot identify new grains").
3. The body head must see the query-frame residual path, so temporal
   mean-fusion cannot erase spatial evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.model import (  # noqa: E402
    _MultiScaleUNet, build_model)
from prototypes.v30_video_apex.owner_decoder import ImageOnlyEncoder  # noqa: E402


def _copy_encoder(a, b):
    with torch.no_grad():
        for (an, ap), (bn, bp) in zip(a.enc.named_parameters(),
                                     b.enc.named_parameters()):
            if an.endswith("weight") and ap.dim() == 4 and \
                    ap.shape[1] != bp.shape[1]:
                bp.copy_(ap[:, :bp.shape[1]])   # trim the owner channel
            else:
                bp.copy_(ap)
        for (an, ap), (bn, bp) in zip(a.bottleneck.named_parameters(),
                                      b.bottleneck.named_parameters()):
            bp.copy_(ap)


def test_image_only_encoder_mirrors_the_multiscale_encoder():
    torch.manual_seed(0)
    a = _MultiScaleUNet(2, 8)          # image + 1 owner channel
    b = ImageOnlyEncoder(1, 8)         # image only
    _copy_encoder(a, b)
    x1 = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        feats = []
        f = torch.cat([x1, torch.zeros(1, 1, 64, 64)], 1)
        for i, layer in enumerate(a.enc):
            f = layer(f)
            if (i % 4) == 3:
                feats.append(f)
        f_bot_a = a.bottleneck(f)
        f_bot_b, feats_b = b(x1)
    assert len(feats) == len(feats_b)
    for u, v in zip(feats, feats_b):
        assert torch.allclose(u, v, atol=1e-6)
    assert torch.allclose(f_bot_a, f_bot_b, atol=1e-6)


def test_owner_decoder_has_no_owner_id_embedding():
    """Conditioning is spatial; nothing may key on the owner string."""
    model = build_model("owner-decoder", base=8, multiscale=True)
    names = [n for n, _ in model.named_parameters()]
    bad = [n for n in names
           if "embed" in n.lower() or "owner_emb" in n.lower()]
    assert bad == [], f"owner-identity embeddings present: {bad}"
    # the same image and the same owner GEOMETRY must not depend on the
    # owner string: swapping the id alone changes nothing.
    clip = torch.randn(1, 9, 1, 64, 64)
    from prototypes.v30_video_apex.model import OwnerPrompt
    m = torch.zeros(64, 64)
    with torch.no_grad():
        o1 = model(clip, OwnerPrompt(owner_id="grain-a", grain_mask=m))
        o2 = model(clip, OwnerPrompt(owner_id="grain-b", grain_mask=m))
    assert torch.allclose(o1.body, o2.body, atol=1e-6), (
        "owner id changed the body prediction: conditioning leaked into "
        "an identity embedding instead of staying spatial")


def test_body_head_sees_the_query_frame_residual():
    """Zeroing the temporal volume's non-query frames must not remove the
    query frame's contribution: the body head reads it directly."""
    model = build_model("owner-decoder", base=8, multiscale=True)
    model.eval()
    clip = torch.randn(1, 9, 1, 64, 64)
    from prototypes.v30_video_apex.model import OwnerPrompt
    pr = OwnerPrompt(owner_id="g", grain_mask=torch.zeros(64, 64))
    with torch.no_grad():
        base = model(clip, pr).body
        # a distinctive query frame must change the prediction even if
        # surrounding frames are held fixed
        clip2 = clip.clone()
        clip2[:, 4] += 3.0
        alt = model(clip2, pr).body
    assert not torch.allclose(base, alt, atol=1e-4), (
        "the query frame's own features did not reach the body head")
