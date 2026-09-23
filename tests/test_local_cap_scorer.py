"""rev11 section 8B: the native local-cap scorer and its invariants.

The review's architectural hypothesis: 'replace the ribbon-relative
front readout with a local cap scorer in native image coordinates ...
Its input neighborhood around a given cap should stay identical when a
distant support endpoint moves.' These tests pin that property: the
sampled window — and therefore the logit — at a fixed (x, y, tangent)
is bit-identical regardless of what else is in the location grid, how
many locations there are, or where any support ends sit.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from prototypes.v30_video_apex.model import (  # noqa: E402
    build_model, build_owner_prompt)
from prototypes.v30_video_apex.model_factory import (  # noqa: E402
    build_model_from_checkpoint)
from prototypes.v30_video_apex import train as T  # noqa: E402

CS = 288


def _clip(seed: int = 0) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 9, 1, CS, CS, generator=g)


def _model(local: bool = True, base: int = 4):
    torch.manual_seed(0)
    return build_model("temporal", base=base, multiscale=True,
                       presence_head=True, local_cap_head=local)


def _cq(locs, tangs):
    return {"locations_xy": torch.tensor([locs], dtype=torch.float32),
            "tangents_xy": torch.tensor([tangs], dtype=torch.float32)}


def test_declared_and_additive():
    """The scorer exists only when declared; baselines stay identical."""
    m_off = _model(local=False)
    assert not any(n.startswith("cap_scorer") for n, _ in m_off.named_parameters())
    m_on = _model(local=True)
    keys = [n for n, _ in m_on.named_parameters() if n.startswith("cap_scorer")]
    assert len(keys) == 4, keys
    # every non-scorer parameter is bit-identical between the two builds
    sd_off = m_off.state_dict()
    for n, p in m_on.named_parameters():
        if n.startswith("cap_scorer"):
            continue
        assert torch.equal(p.detach(), sd_off[n]), n


def test_forward_cap_logits_shape_and_none():
    m = _model(local=True)
    clip = _clip()
    prompt = build_owner_prompt("o")
    out = m.forward(clip, prompt,
                    cap_query=_cq([[10.0, 10.0], [20.0, 20.0]],
                                  [[1.0, 0.0], [0.0, 1.0]]))
    assert out.cap_logits is not None
    assert tuple(out.cap_logits.shape) == (1, 2)
    out2 = m.forward(clip, prompt)
    assert out2.cap_logits is None
    m2 = _model(local=False)
    out3 = m2.forward(clip, prompt,
                      cap_query=_cq([[10.0, 10.0]], [[1.0, 0.0]]))
    assert out3.cap_logits is None


def test_invariance_fixed_location_across_grid_contents():
    """The same location scores bit-identically no matter what other
    locations, tangents or grid sizes accompany it — the endpoint-move
    invariance at unit level."""
    m = _model(local=True)
    m.eval()
    clip = _clip(1)
    prompt = build_owner_prompt("o")
    loc = [100.0, 120.0]
    tang = [0.6, 0.8]
    with torch.no_grad():
        a = m.forward(clip, prompt, cap_query=_cq([loc], [tang]))
        # same location first, then a completely different grid context
        b = m.forward(clip, prompt, cap_query=_cq(
            [loc, [30.0, 30.0], [200.0, 200.0], [50.0, 250.0]],
            [tang, [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        c = m.forward(clip, prompt, cap_query=_cq(
            [loc] + [[float(5 + 7 * i % 270), float(15 + 11 * i % 260)]
                     for i in range(47)],
            [tang] + [[1.0, 0.0] for _ in range(47)]))
    la = float(a.cap_logits[0, 0])
    lb = float(b.cap_logits[0, 0])
    lc = float(c.cap_logits[0, 0])
    # b vs c (two nontrivial grid sizes): bit-equal. a (K=1) may land
    # in a different grid_sample kernel — accept float32 rounding only.
    assert lb == lc, (lb, lc)
    assert abs(la - lb) <= 1e-6 and abs(la - lc) <= 1e-6, (la, lb, lc)


def test_invariance_logit_at_cap_under_endpoint_move():
    """Two 'constructions' that share geometry up to the cap but move
    the far endpoint (12 px vs 48 px tail): the tip location's logit is
    bit-equal, while a location living only in the moved tail changes
    nothing about it."""
    m = _model(local=True)
    m.eval()
    clip = _clip(2)
    prompt = build_owner_prompt("o")
    # shared geometry: same polyline up to the tip
    path = [[60.0, 60.0], [90.0, 90.0], [120.0, 120.0]]
    tip = [120.0, 120.0]
    tang_tip = [np.sqrt(0.5), np.sqrt(0.5)]
    tail12 = [[132.0, 132.0]]
    tail48 = [[156.0, 156.0]]
    locs_a = [[60.0, 60.0], [90.0, 90.0], [120.0, 120.0]] + tail12
    locs_b = [[60.0, 60.0], [90.0, 90.0], [120.0, 120.0]] + tail48
    tangs_a = [[1.0, 0.0], tang_tip, tang_tip, tang_tip]
    tangs_b = [[1.0, 0.0], tang_tip, tang_tip, tang_tip]
    with torch.no_grad():
        a = m.forward(clip, prompt, cap_query=_cq(locs_a, tangs_a))
        b = m.forward(clip, prompt, cap_query=_cq(locs_b, tangs_b))
    la = a.cap_logits.reshape(-1)
    lb = b.cap_logits.reshape(-1)
    # the tip sample and everything before it: bit-equal
    assert torch.equal(la[:3], lb[:3])
    # sanity: the tip location actually points where we think
    assert tip == locs_a[2] == locs_b[2]


def test_factory_rebuilds_declared_local_cap_head():
    m = _model(local=True)
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "cap.pt"
        T.save_checkpoint(ck, m, config={
            "variant": "temporal", "base": 4, "multiscale": True,
            "presence_head": True, "local_cap_head": True})
        m2, info = build_model_from_checkpoint(ck)
        assert info.get("config", {}).get("local_cap_head") is True
        assert any(n.startswith("cap_scorer")
                   for n, _ in m2.named_parameters())
        # and a checkpoint WITHOUT the flag has no scorer at all
        # (its config must declare everything its weights contain —
        # presence_head, but not local_cap_head)
        ck2 = Path(td) / "no.pt"
        T.save_checkpoint(ck2, _model(local=False), config={
            "variant": "temporal", "base": 4, "multiscale": True,
            "presence_head": True})
        m3, _ = build_model_from_checkpoint(ck2)
        assert not any(n.startswith("cap_scorer")
                       for n, _ in m3.named_parameters())
